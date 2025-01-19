use feanor_math::algorithms::convolution::*;
use feanor_math::integer::*;
use feanor_math::ring::*;
use feanor_math::homomorphism::*;
use feanor_math::rings::zn::*;
use feanor_math::rings::zn::zn_64::*;
use feanor_math::rings::extension::*;
use feanor_math::seq::*;
use feanor_math::integer::BigIntRing;
use feanor_math::primitive_int::StaticRing;
use feanor_math::algorithms::matmul::ComputeInnerProduct;
use feanor_math::matrix::*;
use subvector::SubvectorView;

use std::alloc::Allocator;
use std::alloc::Global;
use std::marker::PhantomData;
use std::ops::Range;

use crate::ciphertext_ring::gadget_product::double_rns::prime_factor_groups;
use crate::ciphertext_ring::number_ring::HECyclotomicNumberRing;
use crate::ciphertext_ring::single_rns_ring::*;
use crate::cyclotomic::CyclotomicRing;
use crate::rnsconv::*;
use crate::IsEq;

type UsedBaseConversion<A> = lift::AlmostExactBaseConversion<A>;

pub struct GadgetProductRhsOperand<'a, NumberRing, A, C> 
    where NumberRing: HECyclotomicNumberRing,
        A: Allocator + Clone,
        C: PreparedConvolutionAlgorithm<ZnBase>
{
    ring: &'a SingleRNSRingBase<NumberRing, A, C>,
    data: Vec<Option<Vec<C::PreparedConvolutionOperand, A>>>,
    digits: Vec<Range<usize>>
}

impl<'a, NumberRing, A, C> GadgetProductRhsOperand<'a, NumberRing, A, C> 
    where NumberRing: HECyclotomicNumberRing,
        A: Allocator + Clone,
        C: PreparedConvolutionAlgorithm<ZnBase>
{
    pub fn gadget_vector<'b>(&'b self) -> &'b [Range<usize>] {
        &self.digits
    }

    pub fn set_rns_factor(&mut self, i: usize, el: SingleRNSRingEl<NumberRing, A, C>) {
        self.data[i] = Some(self.ring.prepare_multiplicant(&el).data);
    }
    
    pub fn create_empty<const LOG: bool>(ring: &'a SingleRNSRingBase<NumberRing, A, C>, digits: usize) -> Self {
        let mut operands = Vec::with_capacity(digits);
        operands.extend((0..digits).map(|_| None));
        return Self {
            ring: ring,
            data: operands,
            digits: prime_factor_groups(ring.rns_base().len(), digits).iter().collect()
        };
    }
}

pub struct GadgetProductLhsOperand<'a, NumberRing, A, C> 
    where NumberRing: HECyclotomicNumberRing,
        A: Allocator + Clone,
        C: PreparedConvolutionAlgorithm<ZnBase>
{
    data: Vec<Vec<C::PreparedConvolutionOperand>>,
    element: SingleRNSRingEl<NumberRing, A, C>,
    ring: PhantomData<&'a SingleRNSRingBase<NumberRing, A, C>>
}

impl<'a, NumberRing, A, C>  GadgetProductLhsOperand<'a, NumberRing, A, C> 
    where NumberRing: HECyclotomicNumberRing,
            A: Allocator + Clone,
            C: PreparedConvolutionAlgorithm<ZnBase>
{
    pub fn create_from_element(ring: &'a SingleRNSRingBase<NumberRing, A, C>, digits: usize, el: SingleRNSRingEl<NumberRing, A, C>) -> Self {
        let digits = prime_factor_groups(ring.rns_base().len(), digits).iter().collect::<Vec<_>>();
        return GadgetProductLhsOperand {
            ring: PhantomData,
            data: ring.gadget_decompose(&el, &digits, ring.rns_base().len()),
            element: el,
        };
    }

    pub(in crate::ciphertext_ring) fn element(&self) -> &SingleRNSRingEl<NumberRing, A, C> {
        &self.element
    }

    pub(in crate::ciphertext_ring) fn digits(&self) -> usize {
        self.data.len()
    }
}

impl<NumberRing, A, C> SingleRNSRingBase<NumberRing, A, C> 
    where NumberRing: HECyclotomicNumberRing,
        A: Allocator + Clone,
        C: PreparedConvolutionAlgorithm<ZnBase>
{
    ///
    /// `gadget_decompose()[decomposed_component][rns_base_index]` contains the prepared convolution 
    /// modulo `shortened_rns_base.at(rns_base_index)` of the `decomposed_component`-th element of the gadget 
    /// decomposition vector. Here `shortened_rns_base` is formed by the last `output_moduli_count` rns 
    /// components of the main rns base.
    /// 
    /// The order of the fourier coefficients is the same as specified by the corresponding [`GeneralizedFFT`].
    /// 
    fn gadget_decompose(&self, el: &SingleRNSRingEl<NumberRing, A, C>, digits: &[Range<usize>], output_moduli_count: usize) -> Vec<Vec<C::PreparedConvolutionOperand>> {
        let ZZbig = BigIntRing::RING;
        let digit_bases = digits.iter().map(|range| zn_rns::Zn::new(range.clone().map(|i| self.rns_base().at(i)).collect::<Vec<_>>(), ZZbig)).collect::<Vec<_>>();

        let mut result = Vec::new();
        let el_as_matrix = self.coefficients_as_matrix(el);

        let homs = (0..output_moduli_count).map(|k| self.rns_base().at(self.rns_base().len() - output_moduli_count + k).can_hom::<StaticRing<i64>>(&StaticRing::<i64>::RING).unwrap()).collect::<Vec<_>>();
        let mut current_row = Vec::with_capacity(self.n() * output_moduli_count);
        current_row.resize_with(self.n() * output_moduli_count, || self.base_ring().at(0).zero());
        
        for (digit, base) in digits.iter().zip(digit_bases.iter()) {
            
            let conversion = UsedBaseConversion::new_with(
                SubvectorView::new(self.rns_base()).restrict(digit.clone()).as_iter().map(|Zn| Zn.clone()).collect::<Vec<_>>(),
                homs.iter().map(|h| **h.codomain()).collect::<Vec<_>>(),
                Global
            );
            
            conversion.apply(
                el_as_matrix.restrict_rows(digit.clone()),
                SubmatrixMut::from_1d(&mut current_row[..], output_moduli_count, self.n())
            );

            let mut part = Vec::with_capacity(output_moduli_count);
            part.extend((0..output_moduli_count).map(|k| {
                self.convolutions().at(self.rns_base().len() - output_moduli_count + k).prepare_convolution_operand(
                    &current_row[(k * self.n())..((k + 1) * self.n())],
                    homs[k].codomain()
                )
            }));
            result.push(part);
        }
        return result;
    }

    pub fn gadget_product_base(&self, lhs: &GadgetProductLhsOperand<NumberRing, A, C>, rhs: &GadgetProductRhsOperand<NumberRing, A, C>) -> SingleRNSRingEl<NumberRing, A, C> {
        let local_rns_base_len = lhs.data[0].len();
        let digits = &rhs.digits;
        assert_eq!(lhs.data.len(), digits.len());
        assert_eq!(rhs.data.len(), digits.len());
        assert!(lhs.data.iter().all(|components| components.len() == local_rns_base_len));
        assert!(rhs.data.iter().all(|components| if let Some(components) = components { components.len() == local_rns_base_len } else { true }));

        // currently this is the only case we use
        assert!(self.rns_base().len() == local_rns_base_len);

        let mut unreduced_result = Vec::with_capacity_in(self.n() * 2, self.allocator());
        let mut result = self.zero();
        for i in 0..local_rns_base_len {
            let Zp = self.rns_base().at(self.rns_base().len() - local_rns_base_len + i);
            unreduced_result.clear();
            unreduced_result.resize_with(self.n() * 2, || Zp.zero());
            
            self.convolutions()[self.rns_base().len() - local_rns_base_len + i].compute_convolution_inner_product_prepared(
                (0..digits.len()).filter_map(|j| rhs.data.at(j).as_ref().map(|rhs_part| (&lhs.data[j][i], &rhs_part[i]))), 
                &mut unreduced_result, 
                Zp
            );
            self.reduce_modulus_partly(i, &mut unreduced_result, self.coefficients_as_matrix_mut(&mut result).row_mut_at(i));
        }
        return result;
    }
}
