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

use std::alloc::Allocator;
use std::alloc::Global;
use std::marker::PhantomData;

use crate::rings::ntt_convolution::NTTConvolution;
use crate::rings::single_rns_ring::*;
use crate::cyclotomic::CyclotomicRing;
use crate::rnsconv::*;
use crate::rings::decomposition::*;
use crate::IsEq;

pub struct GadgetProductRhsOperand<'a, NumberRing, A, C = NTTConvolution<Zn>> 
    where NumberRing: DecomposableNumberRing<Zn>,
        A: Allocator + Clone,
        C: PreparedConvolutionAlgorithm<ZnBase>
{
    ring: &'a SingleRNSRingBase<NumberRing, Zn, A, C>,
    data: Vec<Option<Vec<C::PreparedConvolutionOperand, A>>>
}

impl<'a, NumberRing, A, C> GadgetProductRhsOperand<'a, NumberRing, A, C> 
    where NumberRing: DecomposableNumberRing<Zn>,
        A: Allocator + Clone,
        C: PreparedConvolutionAlgorithm<ZnBase>
{
    pub fn set_rns_factor(&mut self, i: usize, el: SingleRNSRingEl<NumberRing, Zn, A, C>) {
        self.data[i] = Some(self.ring.prepare_multiplicant(&el).data);
    }
}

pub struct GadgetProductLhsOperand<'a, NumberRing, A, C> 
    where NumberRing: DecomposableNumberRing<Zn>,
        A: Allocator + Clone,
        C: PreparedConvolutionAlgorithm<ZnBase>
{
    data: Vec<Vec<C::PreparedConvolutionOperand, A>, A>,
    ring: PhantomData<&'a SingleRNSRingBase<NumberRing, Zn, A, C>>
}

impl<NumberRing, A, C> SingleRNSRingBase<NumberRing, Zn, A, C> 
    where NumberRing: DecomposableNumberRing<Zn>,
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
    fn gadget_decompose(&self, el: SingleRNSRingEl<NumberRing, Zn, A, C>, output_moduli_count: usize) -> Vec<Vec<C::PreparedConvolutionOperand, A>, A> {
        let mut result = Vec::new_in(self.allocator().clone());
        let el_as_matrix = self.as_matrix(&el);

        let homs = (0..output_moduli_count).map(|k| self.rns_base().at(self.rns_base().len() - output_moduli_count + k).can_hom::<StaticRing<i64>>(&StaticRing::<i64>::RING).unwrap()).collect::<Vec<_>>();
        let mut current_row = Vec::with_capacity_in(self.rank(), self.allocator());
        current_row.resize_with(self.rank(), || self.base_ring().at(0).zero());

        for i in 0..self.rns_base().len() {
            let mut part = Vec::with_capacity_in(output_moduli_count, self.allocator().clone());
            part.extend((0..output_moduli_count).map(|k| {
                for j in 0..self.rank() {
                    current_row[j] = homs[k].map(self.rns_base().at(i).smallest_lift(self.rns_base().at(i).clone_el(el_as_matrix.at(i, j))));
                }
                self.convolutions().at(self.rns_base().len() - output_moduli_count + k).prepare_convolution_operand(
                    &current_row[..],
                    homs[k].codomain()
                )
            }));
            result.push(part);
        }
        return result;
    }

    pub fn to_gadget_product_lhs<'a>(&'a self, el: SingleRNSRingEl<NumberRing, Zn, A, C>) -> GadgetProductLhsOperand<'a, NumberRing, A, C> {
        record_time!("SingleRNSRing::to_gadget_product_lhs", || {
            GadgetProductLhsOperand {
                ring: PhantomData,
                data: self.gadget_decompose(el, self.rns_base().len())
            }
        })
    }

    pub fn gadget_product_rhs_empty<'a>(&'a self) -> GadgetProductRhsOperand<'a, NumberRing, A, C> {
        GadgetProductRhsOperand {
            ring: self,
            data: (0..self.rns_base().len()).map(|_| None).collect()
        }
    }

    pub fn gadget_product(&self, lhs: &GadgetProductLhsOperand<NumberRing, A, C>, rhs: &GadgetProductRhsOperand<NumberRing, A, C>) -> SingleRNSRingEl<NumberRing, Zn, A, C> {
        record_time!("SingleRNSRing::gadget_product", || {
            let local_rns_base_len = lhs.data[0].len();
            assert!(lhs.data.iter().all(|components| components.len() == local_rns_base_len));
            assert!(rhs.data.iter().all(|components| if let Some(components) = components { components.len() == local_rns_base_len } else { true }));

            // currently this is the only case we use
            assert!(self.rns_base().len() == local_rns_base_len);

            let mut unreduced_result = Vec::with_capacity_in(self.rank() * 2 * local_rns_base_len, self.allocator());
            for i in 0..local_rns_base_len {
                let Zp = self.rns_base().at(self.rns_base().len() - local_rns_base_len + i);
                unreduced_result.extend((0..(self.rank() * 2)).map(|_| Zp.zero()));
                for j in 0..self.rns_base().len() {
                    if let Some(rhs_part) = &rhs.data.at(j) {
                        self.convolutions()[self.rns_base().len() - local_rns_base_len + i].compute_convolution_prepared(&lhs.data[j][i], &rhs_part[i], &mut unreduced_result[(i * 2 * self.rank())..((i + 1) * 2 * self.rank())], Zp);
                    }
                }
            }

            let mut result = self.zero();
            self.reduce_modulus(&mut unreduced_result, &mut result);
            return result;
        })
    }
}
