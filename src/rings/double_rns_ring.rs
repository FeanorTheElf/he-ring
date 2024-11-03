use std::alloc::Allocator;
use std::alloc::Global;
use std::marker::PhantomData;

use feanor_math::algorithms::convolution::ConvolutionAlgorithm;
use feanor_math::divisibility::*;
use feanor_math::integer::*;
use feanor_math::iters::multi_cartesian_product;
use feanor_math::iters::MultiProduct;
use feanor_math::matrix::*;
use feanor_math::primitive_int::StaticRing;
use feanor_math::primitive_int::StaticRingBase;
use feanor_math::rings::extension::*;
use feanor_math::rings::finite::*;
use feanor_math::ring::*;
use feanor_math::rings::poly::dense_poly::DensePolyRing;
use feanor_math::rings::zn::*;
use feanor_math::homomorphism::*;
use feanor_math::seq::*;
use serde_json::Number;

use crate::cyclotomic::CyclotomicRing;
use crate::profiling::TimeRecorder;
use crate::rings::number_ring::*;
use crate::rnsconv::*;
use crate::IsEq;

use super::decomposition_ring::*;
use super::single_rns_ring::SingleRNSRingBase;
// use super::single_rns_ring::SingleRNSRingBase;

///
/// The ring `R/qR` specified by a collection of [`RingDecomposition`] for all prime factors `p | q`. 
/// Elements are (by default) stored in double-RNS-representation for efficient arithmetic.
/// 
/// When necessary, it is also possible by using [`DoubleRNSRingBase::do_fft()`] and
/// [`DoubleRNSRingBase::undo_fft()`] to work with ring elements not in double-RNS-representation,
/// but note that multiplication is not available for those.
/// 
pub struct DoubleRNSRingBase<NumberRing, FpTy, A = Global> 
    where NumberRing: HENumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    number_ring: NumberRing,
    ring_decompositions: Vec<<NumberRing as HENumberRing<FpTy>>::Decomposed>,
    rns_base: zn_rns::Zn<FpTy, BigIntRing>,
    allocator: A
}

pub type DoubleRNSRing<NumberRing, FpTy, A = Global> = RingValue<DoubleRNSRingBase<NumberRing, FpTy, A>>;

///
/// A [`DoubleRNSRing`] element, stored by its coefficients w.r.t. the "mult basis".
/// In particular, this is the only representation that allows for multiplications.
/// 
pub struct DoubleRNSEl<NumberRing, FpTy, A = Global>
    where NumberRing: HENumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    pub(super) number_ring: PhantomData<NumberRing>,
    pub(super) allocator: PhantomData<A>,
    pub(super) el_wrt_mult_basis: Vec<El<FpTy>, A>
}

///
/// A [`DoubleRNSRing`] element, stored by its coefficients w.r.t. the "small basis".
/// 
pub struct CoeffEl<NumberRing, FpTy, A = Global>
    where NumberRing: HENumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    number_ring: PhantomData<NumberRing>,
    allocator: PhantomData<A>,
    el_wrt_small_basis: Vec<El<FpTy>, A>
}

impl<NumberRing, FpTy> DoubleRNSRingBase<NumberRing, FpTy> 
    where NumberRing: HENumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>
{
    pub fn new(number_ring: NumberRing, rns_base: zn_rns::Zn<FpTy, BigIntRing>) -> RingValue<Self> {
        Self::new_with(number_ring, rns_base, Global)
    }
}

impl<NumberRing, FpTy, A> DoubleRNSRingBase<NumberRing, FpTy, A> 
    where NumberRing: HENumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    pub fn new_with(number_ring: NumberRing, rns_base: zn_rns::Zn<FpTy, BigIntRing>, allocator: A) -> RingValue<Self> {
        assert!(rns_base.len() > 0);
        RingValue::from(Self {
            ring_decompositions: rns_base.as_iter().map(|Fp| number_ring.mod_p(Fp.clone())).collect(),
            number_ring: number_ring,
            rns_base: rns_base,
            allocator: allocator
        })
    }

    pub fn ring_decompositions(&self) -> &[<NumberRing as HENumberRing<FpTy>>::Decomposed] {
        &self.ring_decompositions
    }

    pub fn rns_base(&self) -> &zn_rns::Zn<FpTy, BigIntRing> {
        &self.rns_base
    }

    pub fn element_len(&self) -> usize {
        self.rank() * self.rns_base().len()
    }

    pub fn as_matrix_wrt_small_basis<'a>(&self, element: &'a CoeffEl<NumberRing, FpTy, A>) -> Submatrix<'a, AsFirstElement<El<FpTy>>, El<FpTy>> {
        Submatrix::from_1d(&element.el_wrt_small_basis, self.rns_base().len(), self.rank())
    }

    pub fn as_matrix_wrt_small_basis_mut<'a>(&self, element: &'a mut CoeffEl<NumberRing, FpTy, A>) -> SubmatrixMut<'a, AsFirstElement<El<FpTy>>, El<FpTy>> {
        SubmatrixMut::from_1d(&mut element.el_wrt_small_basis, self.rns_base().len(), self.rank())
    }

    pub fn number_ring(&self) -> &NumberRing {
        &self.number_ring
    }

    pub fn undo_fft(&self, element: DoubleRNSEl<NumberRing, FpTy, A>) -> CoeffEl<NumberRing, FpTy, A> {
        record_time!(GLOBAL_TIME_RECORDER, "DoubleRNSRing::undo_fft", || {
            assert_eq!(element.el_wrt_mult_basis.len(), self.element_len());
            let mut result = element.el_wrt_mult_basis;
            for i in 0..self.rns_base().len() {
                self.ring_decompositions[i].mult_basis_to_small_basis(&mut result[(i * self.rank())..((i + 1) * self.rank())]);
            }
            CoeffEl {
                el_wrt_small_basis: result,
                number_ring: PhantomData,
                allocator: PhantomData
            }
        })
    }

    pub fn allocator(&self) -> &A {
        &self.allocator
    }

    pub fn zero_non_fft(&self) -> CoeffEl<NumberRing, FpTy, A> {
        CoeffEl {
            el_wrt_small_basis: self.zero().el_wrt_mult_basis,
            number_ring: PhantomData,
            allocator: PhantomData
        }
    }

    pub fn from_non_fft(&self, x: El<<Self as RingExtension>::BaseRing>) -> CoeffEl<NumberRing, FpTy, A> {
        let mut result = Vec::with_capacity_in(self.element_len(), self.allocator.clone());
        let x = self.base_ring().get_congruence(&x);
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            result.push(Zp.clone_el(x.at(i)));
            for _ in 1..self.rank() {
                result.push(Zp.zero());
            }
            self.ring_decompositions[i].coeff_basis_to_small_basis(&mut result[(i * self.rank())..((i + 1) * self.rank())]);
        }
        CoeffEl {
            el_wrt_small_basis: result,
            number_ring: PhantomData,
            allocator: PhantomData
        }
    }

    pub fn do_fft(&self, element: CoeffEl<NumberRing, FpTy, A>) -> DoubleRNSEl<NumberRing, FpTy, A> {
        record_time!(GLOBAL_TIME_RECORDER, "DoubleRNSRing::do_fft", || {
            assert_eq!(element.el_wrt_small_basis.len(), self.element_len());
            let mut result = element.el_wrt_small_basis;
            for i in 0..self.rns_base().len() {
                self.ring_decompositions[i].small_basis_to_mult_basis(&mut result[(i * self.rank())..((i + 1) * self.rank())]);
            }
            DoubleRNSEl {
                el_wrt_mult_basis: result,
                number_ring: PhantomData,
                allocator: PhantomData
            }
        })
    }

    pub fn sample_from_coefficient_distribution<G: FnMut() -> i32>(&self, mut distribution: G) -> CoeffEl<NumberRing, FpTy, A> {
        let mut result = self.zero_non_fft().el_wrt_small_basis;
        for j in 0..self.rank() {
            let c = distribution();
            for i in 0..self.rns_base().len() {
                result[j + i * self.rank()] = self.rns_base().at(i).int_hom().map(c);
            }
        }
        for i in 0..self.rns_base().len() {
            self.ring_decompositions[i].coeff_basis_to_small_basis(&mut result[(i * self.rank())..((i + 1) * self.rank())]);
        }
        return CoeffEl {
            el_wrt_small_basis: result,
            allocator: PhantomData,
            number_ring: PhantomData
        };
    }

    pub fn clone_el_non_fft(&self, val: &CoeffEl<NumberRing, FpTy, A>) -> CoeffEl<NumberRing, FpTy, A> {
        assert_eq!(self.element_len(), val.el_wrt_small_basis.len());
        let mut result = Vec::with_capacity_in(self.element_len(), self.allocator.clone());
        result.extend((0..self.element_len()).map(|i| self.rns_base().at(i / self.rank()).clone_el(&val.el_wrt_small_basis[i])));
        CoeffEl {
            el_wrt_small_basis: result,
            number_ring: PhantomData,
            allocator: PhantomData
        }
    }

    pub fn eq_el_non_fft(&self, lhs: &CoeffEl<NumberRing, FpTy, A>, rhs: &CoeffEl<NumberRing, FpTy, A>) -> bool {
        assert_eq!(self.element_len(), lhs.el_wrt_small_basis.len());
        assert_eq!(self.element_len(), rhs.el_wrt_small_basis.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                if !self.rns_base().at(i).eq_el(&lhs.el_wrt_small_basis[i * self.rank() + j], &rhs.el_wrt_small_basis[i * self.rank() + j]) {
                    return false;
                }
            }
        }
        return true;
    }

    pub fn negate_inplace_non_fft(&self, val: &mut CoeffEl<NumberRing, FpTy, A>) {
        assert_eq!(self.element_len(), val.el_wrt_small_basis.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                self.rns_base().at(i).negate_inplace(&mut val.el_wrt_small_basis[i * self.rank() + j]);
            }
        }
    }

    pub fn negate_non_fft(&self, mut val: CoeffEl<NumberRing, FpTy, A>) -> CoeffEl<NumberRing, FpTy, A> {
        self.negate_inplace_non_fft(&mut val);
        return val;
    }

    pub fn sub_assign_non_fft(&self, lhs: &mut CoeffEl<NumberRing, FpTy, A>, rhs: &CoeffEl<NumberRing, FpTy, A>) {
        assert_eq!(self.element_len(), lhs.el_wrt_small_basis.len());
        assert_eq!(self.element_len(), rhs.el_wrt_small_basis.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                self.rns_base().at(i).sub_assign_ref(&mut lhs.el_wrt_small_basis[i * self.rank() + j], &rhs.el_wrt_small_basis[i * self.rank() + j]);
            }
        }
    }

    pub fn mul_scalar_assign_non_fft(&self, lhs: &mut CoeffEl<NumberRing, FpTy, A>, rhs: &El<zn_rns::Zn<FpTy, BigIntRing>>) {
        assert_eq!(self.element_len(), lhs.el_wrt_small_basis.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                self.rns_base().at(i).mul_assign_ref(&mut lhs.el_wrt_small_basis[i * self.rank() + j], self.rns_base().get_congruence(rhs).at(i));
            }
        }
    }

    pub fn add_assign_non_fft(&self, lhs: &mut CoeffEl<NumberRing, FpTy, A>, rhs: &CoeffEl<NumberRing, FpTy, A>) {
        assert_eq!(self.element_len(), lhs.el_wrt_small_basis.len());
        assert_eq!(self.element_len(), rhs.el_wrt_small_basis.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                self.rns_base().at(i).add_assign_ref(&mut lhs.el_wrt_small_basis[i * self.rank() + j], &rhs.el_wrt_small_basis[i * self.rank() + j]);
            }
        }
    }

    pub fn from_canonical_basis_non_fft<V>(&self, vec: V) -> CoeffEl<NumberRing, FpTy, A>
        where V: IntoIterator<Item = El<<Self as RingExtension>::BaseRing>>
    {
        let mut result = self.zero_non_fft().el_wrt_small_basis;
        for (j, x) in vec.into_iter().enumerate() {
            let congruence = self.base_ring().get_ring().get_congruence(&x);
            for i in 0..self.rns_base().len() {
                result[i * self.rank() + j] = self.rns_base().at(i).clone_el(congruence.at(i));
            }
        }
        for i in 0..self.rns_base().len() {
            self.ring_decompositions[i].coeff_basis_to_small_basis(&mut result[(i * self.rank())..((i + 1) * self.rank())]);
        }
        return CoeffEl {
            el_wrt_small_basis: result,
            allocator: PhantomData,
            number_ring: PhantomData
        };
    }

    pub fn wrt_canonical_basis_non_fft<'a>(&'a self, el: CoeffEl<NumberRing, FpTy, A>) -> DoubleRNSRingBaseElVectorRepresentation<'a, NumberRing, FpTy, A> {
        let mut result = el.el_wrt_small_basis;
        for i in 0..self.rns_base().len() {
            self.ring_decompositions[i].small_basis_to_coeff_basis(&mut result[(i * self.rank())..((i + 1) * self.rank())]);
        }
        return DoubleRNSRingBaseElVectorRepresentation {
            ring: self,
            el_wrt_coeff_basis: result
        };
    }

    pub fn perform_rns_op_from<FpTy2, A2, Op>(
        &self, 
        from: &DoubleRNSRingBase<NumberRing, FpTy2, A2>, 
        el: &CoeffEl<NumberRing, FpTy2, A2>, 
        op: &Op
    ) -> CoeffEl<NumberRing, FpTy, A> 
        where NumberRing: HENumberRing<FpTy2>,
            FpTy2: RingStore<Type = FpTy::Type> + Clone,
            A2: Allocator + Clone,
            Op: RNSOperation<RingType = FpTy::Type>
    {
        record_time!(GLOBAL_TIME_RECORDER, "DoubleRNSRing::perform_rns_op_from", || {
            assert!(self.number_ring == from.number_ring);
            assert_eq!(self.rns_base().len(), op.output_rings().len());
            assert_eq!(from.rns_base().len(), op.input_rings().len());

            for i in 0..from.rns_base().len() {
                assert!(from.rns_base().at(i).get_ring() == op.input_rings().at(i).get_ring());
            }
            for i in 0..self.rns_base().len() {
                assert!(self.rns_base().at(i).get_ring() == op.output_rings().at(i).get_ring());
            }
            let mut result = self.zero_non_fft();
            op.apply(from.as_matrix_wrt_small_basis(el), self.as_matrix_wrt_small_basis_mut(&mut result));
            return result;
        })
    }

    pub fn exact_convert_from_decompring<FpTy2, A2>(
        &self, 
        from: &DecompositionRing<NumberRing, FpTy2, A2>, 
        element: &<DecompositionRingBase<NumberRing, FpTy2, A2> as RingBase>::Element
    ) -> CoeffEl<NumberRing, FpTy, A> 
        where NumberRing: HENumberRing<FpTy2>,
            FpTy2: RingStore<Type = FpTy::Type> + Clone,
            A2: Allocator + Clone
    {
        assert!(&self.number_ring == from.get_ring().number_ring());

        let mut result = self.zero_non_fft().el_wrt_small_basis;
        let el_wrt_coeff_basis = from.wrt_canonical_basis(element);
        for j in 0..self.rank() {
            let x = int_cast(from.base_ring().smallest_lift(el_wrt_coeff_basis.at(j)), &StaticRing::<i32>::RING, from.base_ring().integer_ring());
            for i in 0..self.rns_base().len() {
                result[j + i * self.rank()] = self.rns_base().at(i).int_hom().map(x);
            }
        }
        for i in 0..self.rns_base().len() {
            self.ring_decompositions[i].coeff_basis_to_small_basis(&mut result[(i * self.rank())..((i + 1) * self.rank())]);
        }
        return CoeffEl {
            el_wrt_small_basis: result,
            allocator: PhantomData,
            number_ring: PhantomData
        };
    }

    pub fn perform_rns_op_to_decompring<FpTy2, A2, Op>(
        &self, 
        to: &DecompositionRing<NumberRing, FpTy2, A2>, 
        element: &CoeffEl<NumberRing, FpTy, A>, 
        op: &Op
    ) -> <DecompositionRingBase<NumberRing, FpTy2, A2> as RingBase>::Element 
        where NumberRing: HENumberRing<FpTy2>,
            FpTy2: RingStore<Type = FpTy::Type> + Clone,
            A2: Allocator + Clone,
            Op: RNSOperation<RingType = FpTy::Type>
    {
        record_time!(GLOBAL_TIME_RECORDER, "DoubleRNSRing::perform_rns_op_to_decompring", || {
            assert!(&self.number_ring == to.get_ring().number_ring());
            assert_eq!(self.rns_base().len(), op.input_rings().len());
            assert_eq!(1, op.output_rings().len());
            
            for i in 0..self.rns_base().len() {
                assert!(self.rns_base().at(i).get_ring() == op.input_rings().at(i).get_ring());
            }
            assert!(to.base_ring().get_ring() == op.output_rings().at(0).get_ring());

            let mut el_wrt_coeff_basis = self.clone_el_non_fft(element).el_wrt_small_basis;
            for i in 0..self.rns_base().len() {
                self.ring_decompositions[i].small_basis_to_coeff_basis(&mut el_wrt_coeff_basis[(i * self.rank())..((i + 1) * self.rank())]);
            }
            let el_matrix = Submatrix::from_1d(&el_wrt_coeff_basis[..], self.rns_base().len(), self.rank());

            let mut result = to.zero();
            let result_matrix = SubmatrixMut::from_1d(to.get_ring().wrt_canonical_basis_mut(&mut result), 1, to.rank());
            op.apply(el_matrix, result_matrix);
            return result;
        })
    }

}

impl<NumberRing, FpTy, A> PartialEq for DoubleRNSRingBase<NumberRing, FpTy, A> 
    where NumberRing: HENumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    fn eq(&self, other: &Self) -> bool {
        self.number_ring == other.number_ring && self.rns_base.get_ring() == other.rns_base.get_ring()
    }
}

impl<NumberRing, FpTy, A> RingBase for DoubleRNSRingBase<NumberRing, FpTy, A> 
    where NumberRing: HENumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    type Element = DoubleRNSEl<NumberRing, FpTy, A>;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        assert_eq!(self.element_len(), val.el_wrt_mult_basis.len());
        let mut result = Vec::with_capacity_in(self.element_len(), self.allocator.clone());
        result.extend((0..self.element_len()).map(|i| self.rns_base().at(i / self.rank()).clone_el(&val.el_wrt_mult_basis[i])));
        DoubleRNSEl {
            el_wrt_mult_basis: result,
            number_ring: PhantomData,
            allocator: PhantomData
        }
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        assert_eq!(self.element_len(), lhs.el_wrt_mult_basis.len());
        assert_eq!(self.element_len(), rhs.el_wrt_mult_basis.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                self.rns_base().at(i).add_assign_ref(&mut lhs.el_wrt_mult_basis[i * self.rank() + j], &rhs.el_wrt_mult_basis[i * self.rank() + j]);
            }
        }
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.add_assign_ref(lhs, &rhs);
    }

    fn sub_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        assert_eq!(self.element_len(), lhs.el_wrt_mult_basis.len());
        assert_eq!(self.element_len(), rhs.el_wrt_mult_basis.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                self.rns_base().at(i).sub_assign_ref(&mut lhs.el_wrt_mult_basis[i * self.rank() + j], &rhs.el_wrt_mult_basis[i * self.rank() + j]);
            }
        }
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        assert_eq!(self.element_len(), lhs.el_wrt_mult_basis.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                self.rns_base().at(i).negate_inplace(&mut lhs.el_wrt_mult_basis[i * self.rank() + j]);
            }
        }
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.mul_assign_ref(lhs, &rhs);
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        assert_eq!(self.element_len(), lhs.el_wrt_mult_basis.len());
        assert_eq!(self.element_len(), rhs.el_wrt_mult_basis.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                self.rns_base().at(i).mul_assign_ref(&mut lhs.el_wrt_mult_basis[i * self.rank() + j], &rhs.el_wrt_mult_basis[i * self.rank() + j]);
            }
        }
    }
    
    fn from_int(&self, value: i32) -> Self::Element {
        self.from(self.base_ring().get_ring().from_int(value))
    }

    fn zero(&self) -> Self::Element {
        let mut result = Vec::with_capacity_in(self.element_len(), self.allocator.clone());
        result.extend(self.rns_base().as_iter().flat_map(|Zp| (0..self.rank()).map(|_| Zp.zero())));
        return DoubleRNSEl {
            el_wrt_mult_basis: result,
            number_ring: PhantomData,
            allocator: PhantomData
        };
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        assert_eq!(self.element_len(), lhs.el_wrt_mult_basis.len());
        assert_eq!(self.element_len(), rhs.el_wrt_mult_basis.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                if !self.rns_base().at(i).eq_el(&lhs.el_wrt_mult_basis[i * self.rank() + j], &rhs.el_wrt_mult_basis[i * self.rank() + j]) {
                    return false;
                }
            }
        }
        return true;
    }
    
    fn is_commutative(&self) -> bool { true }
    fn is_noetherian(&self) -> bool { true }
    fn is_approximate(&self) -> bool { false }

    fn dbg_within<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>, env: EnvBindingStrength) -> std::fmt::Result {
        let poly_ring = DensePolyRing::new(self.base_ring(), "X");
        poly_ring.get_ring().dbg_within(&RingRef::new(self).poly_repr(&poly_ring, value, self.base_ring().identity()), out, env)
    }

    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        self.dbg_within(value, out, EnvBindingStrength::Weakest)
    }

    fn square(&self, value: &mut Self::Element) {
        assert_eq!(self.element_len(), value.el_wrt_mult_basis.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                self.rns_base().at(i).square(&mut value.el_wrt_mult_basis[i * self.rank() + j]);
            }
        }
    }

    fn characteristic<I: IntegerRingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        self.base_ring().characteristic(ZZ)     
    }
}

impl<NumberRing, FpTy, A> CyclotomicRing for DoubleRNSRingBase<NumberRing, FpTy, A> 
    where NumberRing: HECyclotomicNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    fn n(&self) -> u64 {
        self.number_ring.n()
    }

    fn apply_galois_action(&self, el: &Self::Element, g: zn_64::ZnEl) -> Self::Element {
        let mut result = self.zero();
        for (i, _) in self.rns_base().as_iter().enumerate() {
            <NumberRing::DecomposedAsCyclotomic>::from_ref(&self.ring_decompositions()[i]).permute_galois_action(
                &el.el_wrt_mult_basis[(i * self.rank())..((i + 1) * self.rank())],
                &mut result.el_wrt_mult_basis[(i * self.rank())..((i + 1) * self.rank())],
                g
            );
        }
        return result;
    }
}

impl<NumberRing, FpTy, A> DivisibilityRing for DoubleRNSRingBase<NumberRing, FpTy, A> 
    where NumberRing: HENumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        let mut result = Vec::with_capacity_in(self.element_len(), self.allocator.clone());
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            for j in 0..self.rank() {
                result.push(Zp.checked_div(&lhs.el_wrt_mult_basis[i * self.rank() + j], &rhs.el_wrt_mult_basis[i * self.rank() + j])?);
            }
        }
        return Some(DoubleRNSEl { el_wrt_mult_basis: result, number_ring: PhantomData, allocator: PhantomData })
    }

    fn is_unit(&self, x: &Self::Element) -> bool {
        x.el_wrt_mult_basis.iter().enumerate().all(|(index, c)| self.rns_base().at(index / self.rank()).is_unit(c))
    }
}

pub struct DoubleRNSRingBaseElVectorRepresentation<'a, NumberRing, FpTy, A> 
    where NumberRing: HENumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    el_wrt_coeff_basis: Vec<El<FpTy>, A>,
    ring: &'a DoubleRNSRingBase<NumberRing, FpTy, A>
}

impl<'a, NumberRing, FpTy, A> VectorFn<El<zn_rns::Zn<FpTy, BigIntRing>>> for DoubleRNSRingBaseElVectorRepresentation<'a, NumberRing, FpTy, A> 
    where NumberRing: HENumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    fn len(&self) -> usize {
        self.ring.rank()
    }

    fn at(&self, i: usize) -> El<zn_rns::Zn<FpTy, BigIntRing>> {
        self.ring.rns_base().from_congruence(self.el_wrt_coeff_basis[i..].iter().step_by(self.ring.rank()).enumerate().map(|(i, x)| self.ring.rns_base().at(i).clone_el(x)))
    }
}

impl<NumberRing, FpTy, A> FreeAlgebra for DoubleRNSRingBase<NumberRing, FpTy, A> 
    where NumberRing: HENumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    type VectorRepresentation<'a> = DoubleRNSRingBaseElVectorRepresentation<'a, NumberRing, FpTy, A> 
        where Self: 'a;

    fn canonical_gen(&self) -> Self::Element {
        let mut result = self.zero_non_fft().el_wrt_small_basis;
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            result[i * self.rank() + 1] = Zp.one();
            self.ring_decompositions[i].coeff_basis_to_small_basis(&mut result[(i * self.rank())..((i + 1) * self.rank())]);
        }
        return self.do_fft(CoeffEl {
            el_wrt_small_basis: result,
            allocator: PhantomData,
            number_ring: PhantomData
        });
    }

    fn rank(&self) -> usize {
        self.ring_decompositions[0].rank()
    }

    fn wrt_canonical_basis<'a>(&'a self, el: &'a Self::Element) -> Self::VectorRepresentation<'a> {
        self.wrt_canonical_basis_non_fft(self.undo_fft(self.clone_el(el)))
    }

    fn from_canonical_basis<V>(&self, vec: V) -> Self::Element
        where V: IntoIterator<Item = El<Self::BaseRing>>
    {
        return self.do_fft(self.from_canonical_basis_non_fft(vec));
    }
}

impl<NumberRing, FpTy, A> RingExtension for DoubleRNSRingBase<NumberRing, FpTy, A> 
    where NumberRing: HENumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    type BaseRing = zn_rns::Zn<FpTy, BigIntRing>;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing {
        self.rns_base()
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        self.from_ref(&x)
    }

    fn from_ref(&self, x: &El<Self::BaseRing>) -> Self::Element {
        let x_congruence = self.rns_base().get_congruence(x);
        let mut result = Vec::with_capacity_in(self.element_len(), self.allocator.clone());
        // this works, since the mult basis is, by definition, given by an isomorphism `R/p -> Fp^n`, so
        // in particular `Fp` mapsto to the diagonal `(x, ..., x) <= Fp^n`
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            for _ in 0..self.rank() {
                result.push(Zp.clone_el(x_congruence.at(i)));
            }
        }
        return DoubleRNSEl {
            el_wrt_mult_basis: result,
            number_ring: PhantomData,
            allocator: PhantomData
        };
    }

    fn mul_assign_base(&self, lhs: &mut Self::Element, rhs: &El<Self::BaseRing>) {
        let x_congruence = self.rns_base().get_congruence(rhs);
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            for j in 0..self.rank() {
                Zp.mul_assign_ref(&mut lhs.el_wrt_mult_basis[i * self.rank() + j], x_congruence.at(i));
            }
        }
    }
}

pub struct WRTCanonicalBasisElementCreator<'a, NumberRing, FpTy, A>
    where NumberRing: HENumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    ring: &'a DoubleRNSRingBase<NumberRing, FpTy, A>
}

impl<'a, 'b, NumberRing, FpTy, A> Clone for WRTCanonicalBasisElementCreator<'a, NumberRing, FpTy, A>
    where NumberRing: HENumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    fn clone(&self) -> Self {
        Self { ring: self.ring }
    }
}

impl<'a, 'b, NumberRing, FpTy, A> Fn<(&'b [El<zn_rns::Zn<FpTy, BigIntRing>>],)> for WRTCanonicalBasisElementCreator<'a, NumberRing, FpTy, A>
    where NumberRing: HENumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    extern "rust-call" fn call(&self, args: (&'b [El<zn_rns::Zn<FpTy, BigIntRing>>],)) -> Self::Output {
        self.ring.from_canonical_basis(args.0.iter().map(|x| self.ring.base_ring().clone_el(x)))
    }
}

impl<'a, 'b, NumberRing, FpTy, A> FnMut<(&'b [El<zn_rns::Zn<FpTy, BigIntRing>>],)> for WRTCanonicalBasisElementCreator<'a, NumberRing, FpTy, A>
    where NumberRing: HENumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    extern "rust-call" fn call_mut(&mut self, args: (&'b [El<zn_rns::Zn<FpTy, BigIntRing>>],)) -> Self::Output {
        self.call(args)
    }
}

impl<'a, 'b, NumberRing, FpTy, A> FnOnce<(&'b [El<zn_rns::Zn<FpTy, BigIntRing>>],)> for WRTCanonicalBasisElementCreator<'a, NumberRing, FpTy, A>
    where NumberRing: HENumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    type Output = El<DoubleRNSRing<NumberRing, FpTy, A>>;

    extern "rust-call" fn call_once(self, args: (&'b [El<zn_rns::Zn<FpTy, BigIntRing>>],)) -> Self::Output {
        self.call(args)
    }
}

impl<NumberRing, FpTy, A> FiniteRing for DoubleRNSRingBase<NumberRing, FpTy, A> 
    where NumberRing: HENumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    type ElementsIter<'a> = MultiProduct<
        <zn_rns::ZnBase<FpTy, BigIntRing> as FiniteRing>::ElementsIter<'a>, 
        WRTCanonicalBasisElementCreator<'a, NumberRing, FpTy, A>, 
        CloneRingEl<&'a zn_rns::Zn<FpTy, BigIntRing>>,
        El<DoubleRNSRing<NumberRing, FpTy, A>>
    > where Self: 'a;

    fn elements<'a>(&'a self) -> Self::ElementsIter<'a> {
        multi_cartesian_product((0..self.rank()).map(|_| self.base_ring().elements()), WRTCanonicalBasisElementCreator { ring: self }, CloneRingEl(self.base_ring()))
    }

    fn size<I: IntegerRingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        let modulus = self.base_ring().size(ZZ)?;
        if ZZ.get_ring().representable_bits().is_none() || ZZ.get_ring().representable_bits().unwrap() >= self.rank() * ZZ.abs_log2_ceil(&modulus).unwrap() {
            Some(ZZ.pow(modulus, self.rank()))
        } else {
            None
        }
    }

    fn random_element<G: FnMut() -> u64>(&self, mut rng: G) -> <Self as RingBase>::Element {
        let mut result = self.zero();
        for j in 0..self.rank() {
            for i in 0..self.rns_base().len() {
                result.el_wrt_mult_basis[j + i * self.rank()] = self.rns_base().at(i).random_element(&mut rng);
            }
        }
        return result;
    }
}

impl<NumberRing, FpTy1, FpTy2, A1, A2> CanHomFrom<DoubleRNSRingBase<NumberRing, FpTy2, A2>> for DoubleRNSRingBase<NumberRing, FpTy1, A1>
    where NumberRing: HENumberRing<FpTy1> + HENumberRing<FpTy2>,

        FpTy1: RingStore + Clone,
        FpTy1::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A1: Allocator + Clone,
        FpTy2: RingStore + Clone,
        FpTy2::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A2: Allocator + Clone,

        FpTy1::Type: CanHomFrom<FpTy2::Type>
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
        self.map_in_ref(from, &el, hom)
    }

    fn map_in_ref(&self, from: &DoubleRNSRingBase<NumberRing, FpTy2, A2>, el: &<DoubleRNSRingBase<NumberRing, FpTy2, A2> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        let mut result = Vec::with_capacity_in(self.element_len(), self.allocator.clone());
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            for j in 0..self.rank() {
                result.push(Zp.get_ring().map_in_ref(from.rns_base().at(i).get_ring(), &el.el_wrt_mult_basis[i * self.rank() + j], &hom[i]));
            }
        }
        DoubleRNSEl {
            el_wrt_mult_basis: result,
            number_ring: PhantomData,
            allocator: PhantomData
        }
    }
}

impl<NumberRing, FpTy1, FpTy2, A1, A2, C2> CanHomFrom<SingleRNSRingBase<NumberRing, FpTy2, A2, C2>> for DoubleRNSRingBase<NumberRing, FpTy1, A1>
    where NumberRing: HECyclotomicNumberRing<FpTy1> + HECyclotomicNumberRing<FpTy2>,
        FpTy1: RingStore + Clone,
        FpTy1::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A1: Allocator + Clone,
        FpTy2: RingStore + Clone,
        FpTy2::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A2: Allocator + Clone,
        FpTy1::Type: CanHomFrom<FpTy2::Type>,
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
        let mut result = Vec::with_capacity_in(self.element_len(), self.allocator.clone());
        let el_as_matrix = from.as_matrix(el);
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            for j in 0..self.rank() {
                result.push(Zp.get_ring().map_in_ref(from.rns_base().at(i).get_ring(), el_as_matrix.at(i, j), &hom[i]));
            }
        }
        for i in 0..self.rns_base().len() {
            self.ring_decompositions().at(i).coeff_basis_to_small_basis(&mut result[(i * self.rank())..((i + 1) * self.rank())]);
        }
        self.do_fft(CoeffEl {
            el_wrt_small_basis: result,
            number_ring: PhantomData,
            allocator: PhantomData
        })
    }
}

impl<NumberRing, FpTy1, FpTy2, A1, A2, C2> CanIsoFromTo<SingleRNSRingBase<NumberRing, FpTy2, A2, C2>> for DoubleRNSRingBase<NumberRing, FpTy1, A1>
    where NumberRing: HECyclotomicNumberRing<FpTy1> + HECyclotomicNumberRing<FpTy2>,
        FpTy1: RingStore + Clone,
        FpTy1::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A1: Allocator + Clone,
        FpTy2: RingStore + Clone,
        FpTy2::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A2: Allocator + Clone,
        FpTy1::Type: CanIsoFromTo<FpTy2::Type>,
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

    fn map_out(&self, from: &SingleRNSRingBase<NumberRing, FpTy2, A2, C2>, el: <Self as RingBase>::Element, iso: &Self::Isomorphism) -> <SingleRNSRingBase<NumberRing, FpTy2, A2, C2> as RingBase>::Element {
        let mut result = from.zero();
        let mut result_matrix = from.as_matrix_mut(&mut result);
        let mut el_coeff = self.undo_fft(el).el_wrt_small_basis;
        for i in 0..self.rns_base().len() {
            self.ring_decompositions().at(i).small_basis_to_coeff_basis(&mut el_coeff[(i * self.rank())..((i + 1) * self.rank())]);
        }
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            for j in 0..self.rank() {
                *result_matrix.at_mut(i, j) = Zp.get_ring().map_out(from.rns_base().at(i).get_ring(), Zp.clone_el(&el_coeff[i * self.rank() + j]), &iso[i]);
            }
        }
        return result;
    }
}

impl<NumberRing, FpTy1, FpTy2, A1, A2> CanIsoFromTo<DoubleRNSRingBase<NumberRing, FpTy2, A2>> for DoubleRNSRingBase<NumberRing, FpTy1, A1>
    where NumberRing: HENumberRing<FpTy1> + HENumberRing<FpTy2>,

        FpTy1: RingStore + Clone,
        FpTy1::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A1: Allocator + Clone,
        FpTy2: RingStore + Clone,
        FpTy2::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A2: Allocator + Clone,

        FpTy1::Type: CanIsoFromTo<FpTy2::Type>
{
    type Isomorphism = Vec<<FpTy1::Type as CanIsoFromTo<FpTy2::Type>>::Isomorphism>;

    fn has_canonical_iso(&self, from: &DoubleRNSRingBase<NumberRing, FpTy2, A2>) -> Option<Self::Isomorphism> {
        if self.rns_base().len() == from.rns_base().len() && self.number_ring() == from.number_ring() {
            (0..self.rns_base().len()).map(|i| self.rns_base().at(i).get_ring().has_canonical_iso(from.rns_base().at(i).get_ring()).ok_or(())).collect::<Result<Vec<_>, ()>>().ok()
        } else {
            None
        }
    }

    fn map_out(&self, from: &DoubleRNSRingBase<NumberRing, FpTy2, A2>, el: Self::Element, iso: &Self::Isomorphism) -> <DoubleRNSRingBase<NumberRing, FpTy2, A2> as RingBase>::Element {
        let mut result = Vec::with_capacity_in(from.element_len(), from.allocator.clone());
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            for j in 0..self.rank() {
                result.push(Zp.get_ring().map_out(from.rns_base().at(i).get_ring(), Zp.clone_el(&el.el_wrt_mult_basis[i * self.rank() + j]), &iso[i]));
            }
        }
        DoubleRNSEl {
            el_wrt_mult_basis: result,
            number_ring: PhantomData,
            allocator: PhantomData
        }
    }
}

#[cfg(any(test, feature = "generic_tests"))]
pub fn test_with_number_ring<NumberRing: Clone + HECyclotomicNumberRing<zn_64::Zn>>(number_ring: NumberRing) {
    use crate::{profiling::{clear_all_timings, print_all_timings}, rings::ntt_conv::NTTConv};

    let p1 = number_ring.largest_suitable_prime(20000).unwrap();
    let p2 = number_ring.largest_suitable_prime(p1 - 1).unwrap();
    assert!(p1 != p2);
    let rank = number_ring.rank();
    let base_ring = zn_rns::Zn::new(vec![zn_64::Zn::new(p1 as u64), zn_64::Zn::new(p2 as u64)], BigIntRing::RING);
    let ring = DoubleRNSRingBase::new(number_ring.clone(), base_ring.clone());

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

    let single_rns_ring = SingleRNSRingBase::<_, _, _, NTTConv<_>>::new(number_ring.clone(), base_ring.clone());
    feanor_math::ring::generic_tests::test_hom_axioms(&ring, &single_rns_ring, elements.iter().map(|x| ring.clone_el(x)));
}
