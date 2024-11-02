use std::alloc::{Allocator, Global};
use std::marker::PhantomData;
use std::ops::Range;

use feanor_math::algorithms::convolution::{ConvolutionAlgorithm, KaratsubaAlgorithm, PreparedConvolutionAlgorithm, STANDARD_CONVOLUTION};
use feanor_math::algorithms::cyclotomic::cyclotomic_polynomial;
use feanor_math::algorithms::int_factor::factor;
use feanor_math::algorithms::matmul::ComputeInnerProduct;
use feanor_math::algorithms::poly_gcd::factor;
use feanor_math::primitive_int::{StaticRing, StaticRingBase};
use feanor_math::{assert_el_eq, ring::*};
use feanor_math::homomorphism::*;
use feanor_math::integer::*;
use feanor_math::rings::extension::*;
use feanor_math::rings::finite::FiniteRingStore;
use feanor_math::rings::poly::dense_poly::DensePolyRing;
use feanor_math::rings::poly::sparse_poly::SparsePolyRing;
use feanor_math::rings::poly::*;
use feanor_math::rings::zn::*;
use feanor_math::seq::sparse::SparseMapVector;
use feanor_math::seq::*;
use feanor_math::matrix::*;
use zn_static::Fp;

use crate::{cyclotomic::*, euler_phi};
use crate::rings::double_rns_ring::DoubleRNSRing;
use crate::rnsconv::RNSOperation;

use super::decomposition_ring::{DecompositionRing, DecompositionRingBase};
use super::double_rns_ring::{CoeffEl, DoubleRNSRingBase};
use super::ntt_conv::NTTConvolution;
use super::number_ring::HECyclotomicNumberRing;

pub struct SingleRNSRingBase<NumberRing, FpTy, A, C> 
    where NumberRing: HECyclotomicNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<FpTy::Type>
{
    // we have to store the double-RNS ring as well, since we need double-RNS representation
    // for computing Galois operations. This is because I don't think there is any good way
    // of efficiently computing the galois action image using only coefficient representation
    base: DoubleRNSRing<NumberRing, FpTy, A>,
    convolutions: Vec<C>,
    poly_moduli: CyclotomicReducer<FpTy>
}

pub type SingleRNSRing<NumberRing, FpTy, A, C> = RingValue<SingleRNSRingBase<NumberRing, FpTy, A, C>>;

pub struct SingleRNSRingEl<NumberRing, FpTy, A, C>
    where NumberRing: HECyclotomicNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<FpTy::Type>
{
    el_wrt_coeff_basis: Vec<El<FpTy>, A>,
    convolutions: PhantomData<C>,
    number_ring: PhantomData<NumberRing>
}

pub struct SingleRNSRingPreparedMultiplicant<NumberRing, FpTy, A, C>
    where NumberRing: HECyclotomicNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: PreparedConvolutionAlgorithm<FpTy::Type>
{
    pub(super) element: PhantomData<SingleRNSRingEl<NumberRing, FpTy, A, C>>,
    pub(super) number_ring: PhantomData<NumberRing>,
    pub(super) data: Vec<C::PreparedConvolutionOperand, A>
}

#[cfg(feature = "use_hexl")]
impl<NumberRing> SingleRNSRingBase<NumberRing, zn_64::Zn, Global, crate::rings::hexl_conv::HEXLConv> 
    where NumberRing: HECyclotomicNumberRing<zn_64::Zn>
{
    pub fn new(number_ring: NumberRing, rns_base: zn_rns::Zn<zn_64::Zn, BigIntRing>) -> RingValue<Self> {
        let max_log2_n = StaticRing::<i64>::RING.abs_log2_ceil(&(number_ring.rank() as i64 * 2)).unwrap();
        let convolutions = rns_base.as_iter().map(|Zp| crate::rings::hexl_conv::HEXLConv::new(Zp.clone(), max_log2_n)).collect();
        Self::new_with(number_ring, rns_base, Global, convolutions)
    }
}

impl<NumberRing, FpTy> SingleRNSRingBase<NumberRing, FpTy, Global, NTTConvolution<FpTy>> 
    where NumberRing: HECyclotomicNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
{
    pub fn new(number_ring: NumberRing, rns_base: zn_rns::Zn<FpTy, BigIntRing>) -> RingValue<Self> {
        let max_log2_n = StaticRing::<i64>::RING.abs_log2_ceil(&(number_ring.rank() as i64 * 2)).unwrap();
        let convolutions = rns_base.as_iter().map(|Zp| NTTConvolution::new(Zp.clone(), max_log2_n)).collect();
        Self::new_with(number_ring, rns_base, Global, convolutions)
    }
}

struct CyclotomicReducer<FpTy>
    where FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>
{
    data: Vec<(usize, Vec<Vec<(El<FpTy>, usize)>>)>
} 

impl<FpTy> CyclotomicReducer<FpTy>
    where FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>
{
    fn new(n: i64, rns_base: &zn_rns::Zn<FpTy, BigIntRing>) -> Self {
        let homs = rns_base.as_iter().map(|Zn| Zn.can_hom(&BigIntRing::RING).unwrap().compose(BigIntRing::RING.can_hom(&StaticRing::<i64>::RING).unwrap())).collect::<Vec<_>>();
        let sparse_poly_ring = SparsePolyRing::new(StaticRing::<i64>::RING, "X");
        let mut poly_moduli = Vec::new();
        let mut current_power = n as usize;
        let mut current_m = 1;
        let factorization = factor(StaticRing::<i64>::RING, n as i64);
        for (p, e) in factorization.iter().chain([(1, 1)].iter()) {
            let Phi_m = cyclotomic_polynomial(&sparse_poly_ring, current_m);
            let degree = sparse_poly_ring.degree(&Phi_m).unwrap() * current_power;

            let mut poly_modulus = (0..rns_base.len()).map(|_| Vec::new()).collect::<Vec<_>>();
            for (c, i) in sparse_poly_ring.terms(&Phi_m) {
                if i * current_power != degree {
                    for j in 0..rns_base.len() {
                        poly_modulus[j].push((homs[j].codomain().negate(homs[j].map_ref(c)), i * current_power));
                    }
                }
            }

            poly_moduli.push((degree, poly_modulus));

            current_power /= StaticRing::<i64>::RING.pow(*p, *e) as usize;
            current_m *= StaticRing::<i64>::RING.pow(*p, *e) as usize;
        }
        return Self {
            data: poly_moduli
        };
    }

    fn reduce(&self, rns_base: &zn_rns::Zn<FpTy, BigIntRing>, mut data: SubmatrixMut<AsFirstElement<El<FpTy>>, El<FpTy>>) {
        assert_eq!(data.row_count(), self.data[0].1.len());
        let mut current_index = data.col_count();
        for (degree, modulus) in &self.data {
            for (k, (Zp, Zp_modulus)) in rns_base.as_iter().zip(modulus.iter()).enumerate() {
                let row = data.row_mut_at(k);
                for i in ((*degree)..current_index).rev() {
                    let lc = Zp.clone_el(row.at(i));
                    for (c, j) in Zp_modulus {
                        Zp.add_assign(row.at_mut(i - degree + j), Zp.mul_ref(&lc, c));
                    }
                }
            }
            current_index = *degree;
        }
    }
}

impl<NumberRing, FpTy, A, C> SingleRNSRingBase<NumberRing, FpTy, A, C> 
    where NumberRing: HECyclotomicNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<FpTy::Type>
{
    pub fn new_with(number_ring: NumberRing, rns_base: zn_rns::Zn<FpTy, BigIntRing>, allocator: A, convolutions: Vec<C>) -> RingValue<Self> {
        assert!(rns_base.len() > 0);
        assert_eq!(rns_base.len(), convolutions.len());

        let base = DoubleRNSRingBase::new_with(number_ring, rns_base, allocator);
        let number_ring = base.get_ring().number_ring();
        let rns_base = base.get_ring().rns_base();
        
        RingValue::from(Self {
            poly_moduli: CyclotomicReducer::new(number_ring.n() as i64, rns_base),
            base: base,
            convolutions: convolutions,
        })
    }

    pub(super) fn reduce_modulus(&self, buffer: &mut Vec<El<FpTy>, &A>, output: &mut SingleRNSRingEl<NumberRing, FpTy, A, C>) {
        assert_eq!(2 * self.rank() * self.rns_base().len(), buffer.len());

        record_time!("SingleRNSRing::reduce_modulus", || {

            self.poly_moduli.reduce(self.rns_base(), SubmatrixMut::from_1d(buffer, self.rns_base().len(), 2 * self.rank()));
            
            let mut result_matrix = self.as_matrix_mut(output);
            for k in 0..self.rns_base().len() {
                let Zp = self.rns_base().at(k);
                for i in 0..self.rank() {
                    *result_matrix.at_mut(k, i) = Zp.clone_el(&buffer[k * 2 * self.rank() + i]);
                }
            }
        });
    }

    pub fn rns_base(&self) -> &zn_rns::Zn<FpTy, BigIntRing> {
        self.base.get_ring().rns_base()
    }

    pub fn element_len(&self) -> usize {
        self.rank() * self.rns_base().len()
    }

    pub fn as_matrix<'a>(&self, element: &'a SingleRNSRingEl<NumberRing, FpTy, A, C>) -> Submatrix<'a, AsFirstElement<El<FpTy>>, El<FpTy>> {
        Submatrix::from_1d(&element.el_wrt_coeff_basis, self.rns_base().len(), self.rank())
    }

    pub fn as_matrix_mut<'a>(&self, element: &'a mut SingleRNSRingEl<NumberRing, FpTy, A, C>) -> SubmatrixMut<'a, AsFirstElement<El<FpTy>>, El<FpTy>> {
        SubmatrixMut::from_1d(&mut element.el_wrt_coeff_basis, self.rns_base().len(), self.rank())
    }

    pub fn number_ring(&self) -> &NumberRing {
        self.base.get_ring().number_ring()
    }

    pub fn allocator(&self) -> &A {
        self.base.get_ring().allocator()
    }

    pub fn convolutions(&self) -> &[C] {
        &self.convolutions
    }

    pub fn sample_from_coefficient_distribution<G: FnMut() -> i32>(&self, mut distribution: G) -> SingleRNSRingEl<NumberRing, FpTy, A, C> {
        let mut result = self.zero();
        let mut result_matrix = self.as_matrix_mut(&mut result);
        for j in 0..self.rank() {
            let c = distribution();
            for i in 0..self.rns_base().len() {
                *result_matrix.at_mut(i, j) = self.rns_base().at(i).int_hom().map(c);
            }
        }
        return result;
    }

    pub fn sample_uniform<G: FnMut() -> u64>(&self, mut rng: G) -> SingleRNSRingEl<NumberRing, FpTy, A, C> {
        let mut result = self.zero();
        let mut result_matrix = self.as_matrix_mut(&mut result);
        for j in 0..self.rank() {
            for i in 0..self.rns_base().len() {
                *result_matrix.at_mut(i, j) = self.rns_base().at(i).random_element(&mut rng);
            }
        }
        return result;
    }

    pub fn perform_rns_op_from<FpTy2, A2, C2, Op>(
        &self, 
        from: &SingleRNSRingBase<NumberRing, FpTy2, A2, C2>, 
        el: &SingleRNSRingEl<NumberRing, FpTy2, A2, C2>, 
        op: &Op
    ) -> SingleRNSRingEl<NumberRing, FpTy, A, C> 
        where NumberRing: HECyclotomicNumberRing<FpTy2>,
            FpTy2: RingStore<Type = FpTy::Type> + Clone,
            A2: Allocator + Clone,
            C2: ConvolutionAlgorithm<FpTy2::Type>,
            Op: RNSOperation<RingType = FpTy::Type>
    {
        assert!(self.number_ring() == from.number_ring());
        assert_eq!(self.rns_base().len(), op.output_rings().len());
        assert_eq!(from.rns_base().len(), op.input_rings().len());

        for i in 0..from.rns_base().len() {
            assert!(from.rns_base().at(i).get_ring() == op.input_rings().at(i).get_ring());
        }
        for i in 0..self.rns_base().len() {
            assert!(self.rns_base().at(i).get_ring() == op.output_rings().at(i).get_ring());
        }
        record_time!("SingleRNSRing::perform_rns_op_from", || {
            let mut result = self.zero();
            op.apply(from.as_matrix(el), self.as_matrix_mut(&mut result));
            return result;
        })
    }

    pub fn exact_convert_from_nttring<FpTy2, A2>(
        &self, 
        from: &DecompositionRing<NumberRing, FpTy2, A2>, 
        element: &<DecompositionRingBase<NumberRing, FpTy2, A2> as RingBase>::Element
    ) -> SingleRNSRingEl<NumberRing, FpTy, A, C> 
        where NumberRing: HECyclotomicNumberRing<FpTy2>,
            FpTy2: RingStore<Type = FpTy::Type> + Clone,
            A2: Allocator + Clone
    {
        assert!(self.number_ring() == from.get_ring().number_ring());

        let mut result = self.zero().el_wrt_coeff_basis;
        let el_wrt_coeff_basis = from.wrt_canonical_basis(element);
        for j in 0..self.rank() {
            let x = int_cast(from.base_ring().smallest_lift(el_wrt_coeff_basis.at(j)), &StaticRing::<i32>::RING, from.base_ring().integer_ring());
            for i in 0..self.rns_base().len() {
                result[j + i * self.rank()] = self.rns_base().at(i).int_hom().map(x);
            }
        }
        return SingleRNSRingEl {
            el_wrt_coeff_basis: result,
            convolutions: PhantomData,
            number_ring: PhantomData
        };
    }

    pub fn perform_rns_op_to_nttring<FpTy2, A2, Op>(
        &self, 
        to: &DecompositionRing<NumberRing, FpTy2, A2>, 
        element: &SingleRNSRingEl<NumberRing, FpTy, A, C>, 
        op: &Op
    ) -> <DecompositionRingBase<NumberRing, FpTy2, A2> as RingBase>::Element 
        where NumberRing: HECyclotomicNumberRing<FpTy2>,
            FpTy2: RingStore<Type = FpTy::Type> + Clone,
            A2: Allocator + Clone,
            Op: RNSOperation<RingType = FpTy::Type>
    {
        assert!(self.number_ring() == to.get_ring().number_ring());
        assert_eq!(self.rns_base().len(), op.input_rings().len());
        assert_eq!(1, op.output_rings().len());
        
        for i in 0..self.rns_base().len() {
            assert!(self.rns_base().at(i).get_ring() == op.input_rings().at(i).get_ring());
        }
        assert!(to.base_ring().get_ring() == op.output_rings().at(0).get_ring());

        record_time!("SingleRNSRing::perform_rns_op_to_decompring", || {
            let el_matrix = self.as_matrix(element);
            let mut result = to.zero();
            let result_matrix = SubmatrixMut::from_1d(to.get_ring().wrt_canonical_basis_mut(&mut result), 1, to.rank());
            op.apply(el_matrix, result_matrix);
            return result;
        })
    }
}

impl<NumberRing, FpTy, A, C> SingleRNSRingBase<NumberRing, FpTy, A, C> 
    where NumberRing: HECyclotomicNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: PreparedConvolutionAlgorithm<FpTy::Type>
{
    pub fn prepare_multiplicant(&self, el: &SingleRNSRingEl<NumberRing, FpTy, A, C>) -> SingleRNSRingPreparedMultiplicant<NumberRing, FpTy, A, C> {
        let el_as_matrix = self.as_matrix(&el);
        let mut result = Vec::new_in(self.allocator().clone());
        record_time!("SingleRNSRing::prepare_multiplicant", || {
            result.extend(self.rns_base().as_iter().enumerate().map(|(i, Zp)| self.convolutions[i].prepare_convolution_operand(el_as_matrix.row_at(i), Zp)));
        });
        SingleRNSRingPreparedMultiplicant {
            element: PhantomData,
            data: result,
            number_ring: PhantomData
        }
    }

    pub fn mul_assign_prepared(&self, lhs: &mut SingleRNSRingEl<NumberRing, FpTy, A, C>, rhs: &SingleRNSRingPreparedMultiplicant<NumberRing, FpTy, A, C>) {
        let mut unreduced_result = Vec::with_capacity_in(2 * self.rank() * self.rns_base().len(), self.allocator());
        
        record_time!("SingleRNSRing::mul_assign_prepared::convolution", || {
            let lhs_matrix = self.as_matrix_mut(lhs);
            unreduced_result.extend(self.rns_base().as_iter().flat_map(|Zp| (0..(2 * self.rank())).map(|_| Zp.zero())));
            for k in 0..self.rns_base().len() {
                let Zp = self.rns_base().at(k);
                self.convolutions[k].compute_convolution_lhs_prepared(
                    rhs.data.at(k),
                    lhs_matrix.row_at(k),
                    &mut unreduced_result[(2 * k * self.rank())..(2 * (k + 1) * self.rank())],
                    Zp
                );
            }
        });
        self.reduce_modulus(&mut unreduced_result, lhs);
    }

    pub fn mul_prepared(&self, lhs: &SingleRNSRingPreparedMultiplicant<NumberRing, FpTy, A, C>, rhs: &SingleRNSRingPreparedMultiplicant<NumberRing, FpTy, A, C>) -> SingleRNSRingEl<NumberRing, FpTy, A, C> {
        let mut unreduced_result = Vec::with_capacity_in(2 * self.rank() * self.rns_base().len(), self.allocator());
        
        record_time!("SingleRNSRing::mul_prepared::convolution", || {
            unreduced_result.extend(self.rns_base().as_iter().flat_map(|Zp| (0..(2 * self.rank())).map(|_| Zp.zero())));
            for k in 0..self.rns_base().len() {
                let Zp = self.rns_base().at(k);
                self.convolutions[k].compute_convolution_prepared(
                    rhs.data.at(k),
                    lhs.data.at(k),
                    &mut unreduced_result[(2 * k * self.rank())..(2 * (k + 1) * self.rank())],
                    Zp
                );
            }
        });
        
        let mut result = self.zero();
        self.reduce_modulus(&mut unreduced_result, &mut result);
        return result;
    }
}

pub struct SingleRNSRingBaseElVectorRepresentation<'a, NumberRing, FpTy, A, C> 
    where NumberRing: HECyclotomicNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<FpTy::Type>
{
    element: &'a SingleRNSRingEl<NumberRing, FpTy, A, C>,
    ring: &'a SingleRNSRingBase<NumberRing, FpTy, A, C>
}

impl<'a, NumberRing, FpTy, A, C> VectorFn<El<zn_rns::Zn<FpTy, BigIntRing>>> for SingleRNSRingBaseElVectorRepresentation<'a, NumberRing, FpTy, A, C> 
    where NumberRing: HECyclotomicNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<FpTy::Type>
{
    fn len(&self) -> usize {
        self.ring.rank()
    }

    fn at(&self, i: usize) -> El<zn_rns::Zn<FpTy, BigIntRing>> {
        self.ring.rns_base().from_congruence(self.ring.as_matrix(self.element).col_at(i).as_iter().enumerate().map(|(i, x)| self.ring.rns_base().at(i).clone_el(x)))
    }
}

impl<NumberRing, FpTy, A, C> PartialEq for SingleRNSRingBase<NumberRing, FpTy, A, C> 
    where NumberRing: HECyclotomicNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<FpTy::Type>
{
    fn eq(&self, other: &Self) -> bool {
        self.number_ring() == other.number_ring() && self.rns_base().get_ring() == other.rns_base().get_ring()
    }
}

impl<NumberRing, FpTy, A, C> RingBase for SingleRNSRingBase<NumberRing, FpTy, A, C> 
    where NumberRing: HECyclotomicNumberRing<FpTy>, 
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<FpTy::Type>
{
    type Element = SingleRNSRingEl<NumberRing, FpTy, A, C>;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        assert_eq!(self.element_len(), val.el_wrt_coeff_basis.len());
        let mut result = Vec::with_capacity_in(self.element_len(), self.allocator().clone());
        result.extend((0..self.element_len()).map(|i| self.rns_base().at(i / self.rank()).clone_el(&val.el_wrt_coeff_basis[i])));
        SingleRNSRingEl {
            el_wrt_coeff_basis: result,
            number_ring: PhantomData,
            convolutions: PhantomData
        }
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        assert_eq!(self.element_len(), lhs.el_wrt_coeff_basis.len());
        assert_eq!(self.element_len(), rhs.el_wrt_coeff_basis.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                self.rns_base().at(i).add_assign_ref(&mut lhs.el_wrt_coeff_basis[i * self.rank() + j], &rhs.el_wrt_coeff_basis[i * self.rank() + j]);
            }
        }
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.add_assign_ref(lhs, &rhs);
    }

    fn sub_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        assert_eq!(self.element_len(), lhs.el_wrt_coeff_basis.len());
        assert_eq!(self.element_len(), rhs.el_wrt_coeff_basis.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                self.rns_base().at(i).sub_assign_ref(&mut lhs.el_wrt_coeff_basis[i * self.rank() + j], &rhs.el_wrt_coeff_basis[i * self.rank() + j]);
            }
        }
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        assert_eq!(self.element_len(), lhs.el_wrt_coeff_basis.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                self.rns_base().at(i).negate_inplace(&mut lhs.el_wrt_coeff_basis[i * self.rank() + j]);
            }
        }
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.mul_assign_ref(lhs, &rhs);
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        let mut unreduced_result = Vec::with_capacity_in(2 * self.rank() * self.rns_base().len(), self.allocator());

        record_time!("SingleRNSRing::mul_assign_ref:convolution", || {
            let lhs_matrix = self.as_matrix_mut(lhs);
            let rhs_matrix = self.as_matrix(rhs);
            
            unreduced_result.extend(self.rns_base().as_iter().flat_map(|Zp| (0..(2 * self.rank())).map(|_| Zp.zero())));
            for k in 0..self.rns_base().len() {
                let Zp = self.rns_base().at(k);
                self.convolutions[k].compute_convolution(
                    lhs_matrix.row_at(k),
                    rhs_matrix.row_at(k),
                    &mut unreduced_result[(2 * k * self.rank())..(2 * (k + 1) * self.rank())],
                    Zp
                );
            }
        });
        self.reduce_modulus(&mut unreduced_result, lhs);
    }
    
    fn from_int(&self, value: i32) -> Self::Element {
        self.from(self.base_ring().get_ring().from_int(value))
    }

    fn zero(&self) -> Self::Element {
        let mut result = Vec::with_capacity_in(self.element_len(), self.allocator().clone());
        result.extend(self.rns_base().as_iter().flat_map(|Zp| (0..self.rank()).map(|_| Zp.zero())));
        return SingleRNSRingEl {
            el_wrt_coeff_basis: result,
            number_ring: PhantomData,
            convolutions: PhantomData
        };
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        let lhs = self.as_matrix(lhs);
        let rhs = self.as_matrix(rhs);
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                if !self.rns_base().at(i).eq_el(lhs.at(i, j), rhs.at(i, j)) {
                    return false;
                }
            }
        }
        return true;
    }
    
    fn is_commutative(&self) -> bool { true }
    fn is_noetherian(&self) -> bool { true }
    fn is_approximate(&self) -> bool { false }

    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        let poly_ring = DensePolyRing::new(self.base_ring(), "X");
        poly_ring.get_ring().dbg(&RingRef::new(self).poly_repr(&poly_ring, value, self.base_ring().identity()), out)
    }

    fn characteristic<I: IntegerRingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        self.base_ring().characteristic(ZZ)     
    }
}

impl<NumberRing, FpTy, A, C> RingExtension for SingleRNSRingBase<NumberRing, FpTy, A, C> 
    where NumberRing: HECyclotomicNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<FpTy::Type>
{
    type BaseRing = zn_rns::Zn<FpTy, BigIntRing>;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing {
        self.rns_base()
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        let mut result = self.zero();
        let mut result_matrix = self.as_matrix_mut(&mut result);
        let x_congruence = self.base_ring().get_congruence(&x);
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            *result_matrix.at_mut(i, 0) = Zp.clone_el(x_congruence.at(i));
        }
        return result;
    }

    fn mul_assign_base(&self, lhs: &mut Self::Element, rhs: &El<Self::BaseRing>) {
        let x_congruence = self.rns_base().get_congruence(rhs);
        let mut lhs_matrix = self.as_matrix_mut(lhs);
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            for j in 0..self.rank() {
                Zp.mul_assign_ref(lhs_matrix.at_mut(i, j), x_congruence.at(i));
            }
        }
    }
}

impl<NumberRing, FpTy, A, C> FreeAlgebra for SingleRNSRingBase<NumberRing, FpTy, A, C> 
    where NumberRing: HECyclotomicNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<FpTy::Type>
{
    type VectorRepresentation<'a> = SingleRNSRingBaseElVectorRepresentation<'a, NumberRing, FpTy, A, C> 
        where Self: 'a;

    fn canonical_gen(&self) -> SingleRNSRingEl<NumberRing, FpTy, A, C> {
        let mut result = self.zero();
        let mut result_matrix = self.as_matrix_mut(&mut result);
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            *result_matrix.at_mut(i, 1) = Zp.one();
        }
        return result;
    }

    fn rank(&self) -> usize {
        self.number_ring().rank()
    }

    fn wrt_canonical_basis<'a>(&'a self, el: &'a SingleRNSRingEl<NumberRing, FpTy, A, C>) -> Self::VectorRepresentation<'a> {
        SingleRNSRingBaseElVectorRepresentation {
            ring: self,
            element: el
        }
    }

    fn from_canonical_basis<V>(&self, vec: V) -> SingleRNSRingEl<NumberRing, FpTy, A, C>
        where V: IntoIterator<Item = El<Self::BaseRing>>
    {
        let mut result = self.zero();
        let mut result_matrix = self.as_matrix_mut(&mut result);
        for (j, x) in vec.into_iter().enumerate() {
            let congruence = self.base_ring().get_ring().get_congruence(&x);
            for i in 0..self.rns_base().len() {
                *result_matrix.at_mut(i, j) = self.rns_base().at(i).clone_el(congruence.at(i));
            }
        }
        return result;
    }
}

impl<NumberRing, FpTy, A, C> CyclotomicRing for SingleRNSRingBase<NumberRing, FpTy, A, C> 
    where NumberRing: HECyclotomicNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase> + SelfIso,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<FpTy::Type>
{
    fn n(&self) -> u64 {
        self.base.n()
    }

    fn cyclotomic_index_ring(&self) -> zn_64::Zn {
        self.base.cyclotomic_index_ring()
    }

    fn apply_galois_action(&self, el: &Self::Element, s: zn_64::ZnEl) -> Self::Element {
        let self_ref = RingRef::new(self);
        let iso = self.base.can_iso(&self_ref).unwrap();
        let el_double_rns = iso.inv().map_ref(el);
        let result_double_rns = self.base.apply_galois_action(&el_double_rns, s);
        return iso.map(result_double_rns);
    }
    
    fn apply_galois_action_many<'a>(&'a self, el: &'a Self::Element, gs: &'a [zn_64::ZnEl]) -> impl 'a + ExactSizeIterator<Item = Self::Element> {
        let self_ref = RingRef::new(self);
        let iso = (&self.base).into_can_iso(self_ref).ok().unwrap();
        let el_double_rns = iso.inv().map_ref(el);
        let result_double_rns = self.base.apply_galois_action_many(&el_double_rns, gs);
        return result_double_rns.map(|x| iso.map(x)).collect::<Vec<_>>().into_iter();
    }
}

impl<NumberRing, FpTy1, FpTy2, A1, A2, C1, C2> CanHomFrom<SingleRNSRingBase<NumberRing, FpTy2, A2, C2>> for SingleRNSRingBase<NumberRing, FpTy1, A1, C1>
    where NumberRing: HECyclotomicNumberRing<FpTy1> + HECyclotomicNumberRing<FpTy2>,
        FpTy1: RingStore + Clone,
        FpTy1::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A1: Allocator + Clone,
        FpTy2: RingStore + Clone,
        FpTy2::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A2: Allocator + Clone,
        FpTy1::Type: CanHomFrom<FpTy2::Type>,
        C1: ConvolutionAlgorithm<FpTy1::Type>,
        C2: ConvolutionAlgorithm<FpTy2::Type>
{
    type Homomorphism = Vec<<FpTy1::Type as CanHomFrom<FpTy2::Type>>::Homomorphism>;

    fn has_canonical_hom(&self, from: &SingleRNSRingBase<NumberRing, FpTy2, A2, C2>) -> Option<Self::Homomorphism> {
        if self.rns_base().len() == from.rns_base().len() && self.number_ring() == from.number_ring() {
            (0..self.rns_base().len()).map(|i| self.rns_base().at(i).get_ring().has_canonical_hom(from.rns_base().at(i).get_ring()).ok_or(())).collect::<Result<Vec<_>, ()>>().ok()
        } else {
            None
        }
    }

    fn map_in(&self, from: &SingleRNSRingBase<NumberRing, FpTy2, A2, C2>, el: <SingleRNSRingBase<NumberRing, FpTy2, A2, C2> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in_ref(from, &el, hom)
    }

    fn map_in_ref(&self, from: &SingleRNSRingBase<NumberRing, FpTy2, A2, C2>, el: &<SingleRNSRingBase<NumberRing, FpTy2, A2, C2> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        let el_as_matrix = from.as_matrix(&el);
        let mut result = self.zero();
        let mut result_matrix = self.as_matrix_mut(&mut result);
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            for j in 0..self.rank() {
                *result_matrix.at_mut(i, j) = Zp.get_ring().map_in_ref(from.rns_base().at(i).get_ring(), el_as_matrix.at(i, j), &hom[i]);
            }
        }
        return result;
    }
}

impl<NumberRing, FpTy1, FpTy2, A1, A2, C1> CanHomFrom<DoubleRNSRingBase<NumberRing, FpTy2, A2>> for SingleRNSRingBase<NumberRing, FpTy1, A1, C1>
    where NumberRing: HECyclotomicNumberRing<FpTy1> + HECyclotomicNumberRing<FpTy2>,
        FpTy1: RingStore + Clone,
        FpTy1::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A1: Allocator + Clone,
        FpTy2: RingStore + Clone,
        FpTy2::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A2: Allocator + Clone,
        FpTy2::Type: CanIsoFromTo<FpTy1::Type>,
        C1: ConvolutionAlgorithm<FpTy1::Type>
{
    type Homomorphism = <DoubleRNSRingBase<NumberRing, FpTy2, A2> as CanIsoFromTo<Self>>::Isomorphism;

    fn has_canonical_hom(&self, from: &DoubleRNSRingBase<NumberRing, FpTy2, A2>) -> Option<Self::Homomorphism> {
        from.has_canonical_iso(self)
    }

    fn map_in(&self, from: &DoubleRNSRingBase<NumberRing, FpTy2, A2>, el: <DoubleRNSRingBase<NumberRing, FpTy2, A2> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        from.map_out(self, el, hom)
    }
}

impl<NumberRing, FpTy1, FpTy2, A1, A2, C1, C2> CanIsoFromTo<SingleRNSRingBase<NumberRing, FpTy2, A2, C2>> for SingleRNSRingBase<NumberRing, FpTy1, A1, C1>
    where NumberRing: HECyclotomicNumberRing<FpTy1> + HECyclotomicNumberRing<FpTy2>,
        FpTy1: RingStore + Clone,
        FpTy1::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A1: Allocator + Clone,
        FpTy2: RingStore + Clone,
        FpTy2::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A2: Allocator + Clone,
        FpTy1::Type: CanIsoFromTo<FpTy2::Type>,
        C1: ConvolutionAlgorithm<FpTy1::Type>,
        C2: ConvolutionAlgorithm<FpTy2::Type>
{
    type Isomorphism = Vec<<FpTy1::Type as CanIsoFromTo<FpTy2::Type>>::Isomorphism>;

    fn has_canonical_iso(&self, from: &SingleRNSRingBase<NumberRing, FpTy2, A2, C2>) -> Option<Self::Isomorphism> {
        if self.rns_base().len() == from.rns_base().len() && self.number_ring() == from.number_ring() {
            (0..self.rns_base().len()).map(|i| self.rns_base().at(i).get_ring().has_canonical_iso(from.rns_base().at(i).get_ring()).ok_or(())).collect::<Result<Vec<_>, ()>>().ok()
        } else {
            None
        }
    }

    fn map_out(&self, from: &SingleRNSRingBase<NumberRing, FpTy2, A2, C2>, el: Self::Element, iso: &Self::Isomorphism) -> <SingleRNSRingBase<NumberRing, FpTy2, A2, C2> as RingBase>::Element {
        let el_as_matrix = self.as_matrix(&el);
        let mut result = from.zero();
        let mut result_matrix = from.as_matrix_mut(&mut result);
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            for j in 0..self.rank() {
                *result_matrix.at_mut(i, j) = Zp.get_ring().map_out(from.rns_base().at(i).get_ring(), Zp.clone_el(el_as_matrix.at(i, j)), &iso[i]);
            }
        }
        return result;
    }
}

#[test]
fn test_modulo_cylotomic_polynomial() {
    let rns_base = zn_rns::Zn::new(vec![zn_64::Zn::new(7), zn_64::Zn::new(11)], BigIntRing::RING);
    let reducer = CyclotomicReducer::new(15, &rns_base);
    assert_eq!(3, reducer.data.len());

    let poly_ring = SparsePolyRing::new(StaticRing::<i64>::RING, "X");
    let Phi_15 = cyclotomic_polynomial(&poly_ring, 15);
    for i in 0..16 {
        let mut actual = rns_base.as_iter().flat_map(|Zp| (0..16).map(|j| if i == j { Zp.one() } else { Zp.zero() })).collect::<Vec<_>>();
        let mut actual = SubmatrixMut::from_1d(&mut actual, rns_base.len(), 16);
        reducer.reduce(&rns_base, actual.reborrow());
        let expected = poly_ring.div_rem_monic(poly_ring.pow(poly_ring.indeterminate(), i), &Phi_15).1;

        for k in 0..rns_base.len() {
            for j in 0..8 {
                assert_el_eq!(rns_base.at(k), rns_base.at(k).coerce(&StaticRing::<i64>::RING, *poly_ring.coefficient_at(&expected, j)), actual.at(k, j));
            }
        }
    }
}

#[cfg(any(test, feature = "generic_tests"))]
pub fn test_with_number_ring<NumberRing: Clone + HECyclotomicNumberRing<zn_64::Zn>>(number_ring: NumberRing) {
    let p1 = number_ring.largest_suitable_prime(20000).unwrap();
    let p2 = number_ring.largest_suitable_prime(p1 - 1).unwrap();
    assert!(p1 != p2);
    let rank = number_ring.rank();
    let base_ring = zn_rns::Zn::new(vec![zn_64::Zn::new(p1 as u64), zn_64::Zn::new(p2 as u64)], BigIntRing::RING);
    let ring = SingleRNSRingBase::<_, _, _, NTTConvolution<_>>::new(number_ring.clone(), base_ring.clone());

    let base_ring = ring.base_ring();
    let elements = vec![
        ring.zero(),
        ring.one(),
        ring.neg_one(),
        ring.int_hom().map(p1 as i32),
        ring.int_hom().map(p2 as i32),
        ring.canonical_gen(),
        ring.pow(ring.canonical_gen(), rank - 1),
        ring.int_hom().mul_map(ring.canonical_gen(), p1 as i32),
        ring.int_hom().mul_map(ring.pow(ring.canonical_gen(), rank - 1), p1 as i32),
        ring.add(ring.canonical_gen(), ring.one())
    ];

    feanor_math::ring::generic_tests::test_ring_axioms(&ring, elements.iter().map(|x| ring.clone_el(x)));
    feanor_math::ring::generic_tests::test_self_iso(&ring, elements.iter().map(|x| ring.clone_el(x)));
    feanor_math::rings::extension::generic_tests::test_free_algebra_axioms(&ring);

    let double_rns_ring = DoubleRNSRingBase::new(number_ring.clone(), base_ring.clone());
    feanor_math::ring::generic_tests::test_hom_axioms(&ring, &double_rns_ring, elements.iter().map(|x| ring.clone_el(x)));
}