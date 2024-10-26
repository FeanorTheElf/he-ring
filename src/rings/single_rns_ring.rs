use std::alloc::{Allocator, Global};
use std::marker::PhantomData;

use feanor_math::algorithms::convolution::{ConvolutionAlgorithm, KaratsubaAlgorithm, PreparedConvolutionAlgorithm, STANDARD_CONVOLUTION};
use feanor_math::primitive_int::{StaticRing, StaticRingBase};
use feanor_math::ring::*;
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

use crate::cyclotomic::*;
use crate::rings::decomposition::*;
use crate::rings::double_rns_ring::DoubleRNSRing;
use crate::rnsconv::RNSOperation;

use super::decomposition_ring::{DecompositionRing, DecompositionRingBase};
use super::double_rns_ring::{CoeffEl, DoubleRNSRingBase};
use super::ntt_convolution::NTTConvolution;

pub struct SingleRNSRingBase<NumberRing, FpTy, A = Global, C = NTTConvolution<FpTy>> 
    where NumberRing: DecomposableNumberRing<FpTy>,
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
    modulus: SparseMapVector<zn_rns::Zn<FpTy, BigIntRing>>
}

pub type SingleRNSRing<NumberRing, FpTy, A = Global, C = NTTConvolution<FpTy>> = RingValue<SingleRNSRingBase<NumberRing, FpTy, A, C>>;

pub struct SingleRNSRingEl<NumberRing, FpTy, A = Global, C = NTTConvolution<FpTy>>
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<FpTy::Type>
{
    // should also be visible in gadget_product::single_rns
    pub(super) data: CoeffEl<NumberRing, FpTy, A>,
    pub(super) convolutions: PhantomData<C>
}

pub struct SingleRNSRingPreparedMultiplicant<NumberRing, FpTy, A, C>
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: PreparedConvolutionAlgorithm<FpTy::Type>
{
    // should also be visible in gadget_product::single_rns
    pub(super) element: PhantomData<SingleRNSRingEl<NumberRing, FpTy, A, C>>,
    pub(super) data: Vec<C::PreparedConvolutionOperand, A>
}

impl<NumberRing, FpTy> SingleRNSRingBase<NumberRing, FpTy> 
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<StaticRingBase<i128>>
{
    pub fn new(number_ring: NumberRing, rns_base: zn_rns::Zn<FpTy, BigIntRing>) -> RingValue<Self> {
        let max_len_log2 = StaticRing::<i64>::RING.abs_log2_ceil(&(2 * number_ring.rank() as i64)).unwrap();
        let convolutions = rns_base.as_iter().map(|Zp| NTTConvolution::new(Zp.clone())).collect();
        Self::new_with(number_ring, rns_base, Global, convolutions)
    }
}

impl<NumberRing, FpTy, A, C> SingleRNSRingBase<NumberRing, FpTy, A, C> 
    where NumberRing: DecomposableNumberRing<FpTy>,
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
        
        let mut modulus = SparseMapVector::new(number_ring.rank(), rns_base.clone());
        let sparse_poly_ring = SparsePolyRing::new(BigIntRing::RING, "X");
        let hom = rns_base.can_hom(&BigIntRing::RING).unwrap();
        for (c, i) in sparse_poly_ring.terms(&number_ring.generating_poly(&sparse_poly_ring)) {
            if i != number_ring.rank() {
                *modulus.at_mut(i) = rns_base.negate(hom.map_ref(c));
            }
        }
        modulus.at_mut(0);
        RingValue::from(Self {
            base: base,
            modulus: modulus,
            convolutions: convolutions,
        })
    }

    pub(super) fn reduce_modulus(&self, buffer: &mut Vec<El<FpTy>, &A>, output: &mut SingleRNSRingEl<NumberRing, FpTy, A, C>) {
        assert_eq!(2 * self.rank() * self.rns_base().len(), buffer.len());

        record_time!("SingleRNSRing::reduce_modulus", || {
            for i in (self.rank()..(2 * self.rank())).rev() {
                for (j, c) in self.modulus.nontrivial_entries() {
                    let congruence = self.rns_base().get_congruence(c);
                    for k in 0..self.rns_base().len() {
                        let Zp = self.rns_base().at(k);
                        let subtract = Zp.mul_ref(&buffer[2 * k * self.rank() + i], congruence.at(k));
                        Zp.add_assign(&mut buffer[2 * k * self.rank() + i - self.rank() + j], subtract);
                    }
                }
            }
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
        self.base.get_ring().as_matrix(&element.data)
    }

    pub fn as_matrix_mut<'a>(&self, element: &'a mut SingleRNSRingEl<NumberRing, FpTy, A, C>) -> SubmatrixMut<'a, AsFirstElement<El<FpTy>>, El<FpTy>> {
        self.base.get_ring().as_matrix_mut(&mut element.data)
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

    ///
    /// Returns `a mod pi` where `a` is the coefficient belonging to `X^j` of the given element.
    /// 
    /// Here `pi` is the `i`-th prime divisor of the base ring (using the order exposed by
    /// [`zn_rns::ZnBase`]).
    /// 
    pub fn at<'a>(&self, i: usize, j: usize, el: &'a SingleRNSRingEl<NumberRing, FpTy, A, C>) -> &'a El<FpTy> {
        self.base.get_ring().at(i, j, &el.data)
    }

    /// 
    /// Returns `a mod pi` where `a` is the coefficient belonging to `X^j` of the given element.
    /// 
    /// See [`Self::at()`] for details.
    /// 
    pub fn at_mut<'a>(&self, i: usize, j: usize, el: &'a mut SingleRNSRingEl<NumberRing, FpTy, A, C>) -> &'a mut El<FpTy> {
        self.base.get_ring().at_mut(i, j, &mut el.data)
    }
    
    pub fn sample_from_coefficient_distribution<G: FnMut() -> i32>(&self, distribution: G) -> SingleRNSRingEl<NumberRing, FpTy, A, C> {
        SingleRNSRingEl {
            data: self.base.get_ring().sample_from_coefficient_distribution(distribution),
            convolutions: PhantomData
        }
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
        where NumberRing: DecomposableNumberRing<FpTy2>,
            FpTy2: RingStore<Type = FpTy::Type> + Clone,
            A2: Allocator + Clone,
            C2: ConvolutionAlgorithm<FpTy2::Type>,
            Op: RNSOperation<RingType = FpTy::Type>
    {
        SingleRNSRingEl {
            data: self.base.get_ring().perform_rns_op_from(from.base.get_ring(), &el.data, op),
            convolutions: PhantomData
        }
    }

    pub fn exact_convert_from_nttring<FpTy2, A2>(
        &self, 
        from: &DecompositionRing<NumberRing, FpTy2, A2>, 
        element: &<DecompositionRingBase<NumberRing, FpTy2, A2> as RingBase>::Element
    ) -> SingleRNSRingEl<NumberRing, FpTy, A, C> 
        where NumberRing: DecomposableNumberRing<FpTy2>,
            FpTy2: RingStore<Type = FpTy::Type> + Clone,
            A2: Allocator + Clone
    {
        SingleRNSRingEl {
            data: self.base.get_ring().exact_convert_from_nttring(from, element),
            convolutions: PhantomData
        }
    }

    pub fn perform_rns_op_to_nttring<FpTy2, A2, Op>(
        &self, 
        to: &DecompositionRing<NumberRing, FpTy2, A2>, 
        element: &SingleRNSRingEl<NumberRing, FpTy, A, C>, 
        op: &Op
    ) -> <DecompositionRingBase<NumberRing, FpTy2, A2> as RingBase>::Element 
        where NumberRing: DecomposableNumberRing<FpTy2>,
            FpTy2: RingStore<Type = FpTy::Type> + Clone,
            A2: Allocator + Clone,
            Op: RNSOperation<RingType = FpTy::Type>
    {
        self.base.get_ring().perform_rns_op_to_nttring(to, &element.data, op)
    }
}

impl<NumberRing, FpTy, A, C> SingleRNSRingBase<NumberRing, FpTy, A, C> 
    where NumberRing: DecomposableNumberRing<FpTy>,
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
            data: result
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
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<FpTy::Type>
{
    element: &'a SingleRNSRingEl<NumberRing, FpTy, A, C>,
    ring: &'a SingleRNSRingBase<NumberRing, FpTy, A, C>
}

impl<'a, NumberRing, FpTy, A, C> VectorFn<El<zn_rns::Zn<FpTy, BigIntRing>>> for SingleRNSRingBaseElVectorRepresentation<'a, NumberRing, FpTy, A, C> 
    where NumberRing: DecomposableNumberRing<FpTy>,
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
    where NumberRing: DecomposableNumberRing<FpTy>,
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
    where NumberRing: DecomposableNumberRing<FpTy>, 
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<FpTy::Type>
{
    type Element = SingleRNSRingEl<NumberRing, FpTy, A, C>;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        SingleRNSRingEl {
            data: self.base.get_ring().clone_el_non_fft(&val.data),
            convolutions: PhantomData
        }
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        self.base.get_ring().add_assign_non_fft(&mut lhs.data, &rhs.data);
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.add_assign_ref(lhs, &rhs);
    }

    fn sub_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        self.base.get_ring().sub_assign_non_fft(&mut lhs.data, &rhs.data);
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        self.base.get_ring().negate_inplace_non_fft(&mut lhs.data);
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.mul_assign_ref(lhs, &rhs);
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        let mut unreduced_result = Vec::with_capacity_in(2 * self.rank() * self.rns_base().len(), self.allocator());

        record_time!("SingleRNSRing::mul_assign_ref:convolution", || {
            let lhs_matrix = self.as_matrix_mut(lhs);
            let rhs = self.as_matrix(rhs);
            
            unreduced_result.extend(self.rns_base().as_iter().flat_map(|Zp| (0..(2 * self.rank())).map(|_| Zp.zero())));
            for k in 0..self.rns_base().len() {
                let Zp = self.rns_base().at(k);
                self.convolutions[k].compute_convolution(
                    lhs_matrix.row_at(k),
                    rhs.row_at(k),
                    &mut unreduced_result[(2 * k * self.rank())..(2 * (k + 1) * self.rank())],
                    Zp
                );
            }
        });
        self.reduce_modulus(&mut unreduced_result, lhs);
    }
    
    fn from_int(&self, value: i32) -> Self::Element {
        SingleRNSRingEl {
            data: self.base.get_ring().non_fft_from(self.base_ring().get_ring().from_int(value)),
            convolutions: PhantomData,
        }
    }

    fn zero(&self) -> Self::Element {
        SingleRNSRingEl {
            data: self.base.get_ring().non_fft_zero(),
            convolutions: PhantomData,
        }
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
    where NumberRing: DecomposableNumberRing<FpTy>,
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
        SingleRNSRingEl {
            data: self.base.get_ring().non_fft_from(x),
            convolutions: PhantomData
        }
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
    where NumberRing: DecomposableNumberRing<FpTy>,
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
        SingleRNSRingEl {
            data: self.base.get_ring().from_canonical_basis_non_fft(vec),
            convolutions: PhantomData
        }
    }
}

impl<NumberRing, FpTy, A, C> CyclotomicRing for SingleRNSRingBase<NumberRing, FpTy, A, C> 
    where NumberRing: DecomposableCyclotomicNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
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
        let el_fft = self.base.get_ring().do_fft(self.base.get_ring().clone_el_non_fft(&el.data));
        let result_fft = self.base.apply_galois_action(&el_fft, s);
        return SingleRNSRingEl {
            data: self.base.get_ring().undo_fft(result_fft),
            convolutions: PhantomData
        };
    }
    
    fn apply_galois_action_many<'a>(&'a self, x: &'a Self::Element, gs: &'a [zn_64::ZnEl]) -> impl 'a + ExactSizeIterator<Item = Self::Element> {
        let el_fft = self.base.get_ring().do_fft(self.base.get_ring().clone_el_non_fft(&x.data));
        gs.into_iter().map(move |g| SingleRNSRingEl {
            data: self.base.get_ring().undo_fft(self.base.apply_galois_action(&el_fft, *g)),
            convolutions: PhantomData
        })
    }
}

impl<NumberRing, FpTy1, FpTy2, A1, A2, C1, C2> CanHomFrom<SingleRNSRingBase<NumberRing, FpTy2, A2, C2>> for SingleRNSRingBase<NumberRing, FpTy1, A1, C1>
    where NumberRing: DecomposableNumberRing<FpTy1> + DecomposableNumberRing<FpTy2>,
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
    where NumberRing: DecomposableNumberRing<FpTy1> + DecomposableNumberRing<FpTy2>,
        FpTy1: RingStore + Clone,
        FpTy1::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A1: Allocator + Clone,
        FpTy2: RingStore + Clone,
        FpTy2::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A2: Allocator + Clone,
        FpTy1::Type: CanHomFrom<FpTy2::Type>,
        C1: ConvolutionAlgorithm<FpTy1::Type>
{
    type Homomorphism = Vec<<FpTy1::Type as CanHomFrom<FpTy2::Type>>::Homomorphism>;

    fn has_canonical_hom(&self, from: &DoubleRNSRingBase<NumberRing, FpTy2, A2>) -> Option<Self::Homomorphism> {
        if self.rns_base().len() == from.rns_base().len() && self.number_ring() == from.number_ring() {
            (0..self.rns_base().len()).map(|i| self.rns_base().at(i).get_ring().has_canonical_hom(from.rns_base().at(i).get_ring()).ok_or(())).collect::<Result<Vec<_>, ()>>().ok()
        } else {
            None
        }
    }

    fn map_in(&self, from: &DoubleRNSRingBase<NumberRing, FpTy2, A2>, el: <DoubleRNSRingBase<NumberRing, FpTy2, A2> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        let non_fft_el = from.undo_fft(el);
        let el_as_matrix = from.as_matrix(&non_fft_el);
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

impl<NumberRing, FpTy1, FpTy2, A1, A2, C1, C2> CanIsoFromTo<SingleRNSRingBase<NumberRing, FpTy2, A2, C2>> for SingleRNSRingBase<NumberRing, FpTy1, A1, C1>
    where NumberRing: DecomposableNumberRing<FpTy1> + DecomposableNumberRing<FpTy2>,
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

#[cfg(any(test, feature = "generic_tests"))]
pub fn test_with_number_ring<NumberRing: Clone + DecomposableNumberRing<zn_64::Zn>>(number_ring: NumberRing) {
    let p1 = number_ring.largest_suitable_prime(20000).unwrap();
    let p2 = number_ring.largest_suitable_prime(p1 - 1).unwrap();
    assert!(p1 != p2);
    let rank = number_ring.rank();
    let base_ring = zn_rns::Zn::new(vec![zn_64::Zn::new(p1 as u64), zn_64::Zn::new(p2 as u64)], BigIntRing::RING);
    let ring = SingleRNSRingBase::new(number_ring.clone(), base_ring.clone());

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