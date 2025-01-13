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
use feanor_math::specialization::FiniteRingOperation;
use feanor_math::specialization::FiniteRingSpecializable;
use serde_json::Number;
use zn_64::ZnBase;

use crate::cyclotomic::CyclotomicGaloisGroupEl;
use crate::cyclotomic::CyclotomicRing;
use crate::profiling::TimeRecorder;
use crate::rings::number_ring::*;
use crate::rnsconv::*;
use crate::IsEq;

use super::decomposition_ring::*;
use super::single_rns_ring::SingleRNSRing;
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
pub struct DoubleRNSRingBase<NumberRing, A = Global> 
    where NumberRing: HENumberRing,
        A: Allocator + Clone
{
    number_ring: NumberRing,
    ring_decompositions: Vec<<NumberRing as HENumberRing>::Decomposed>,
    rns_base: zn_rns::Zn<zn_64::Zn, BigIntRing>,
    allocator: A
}

pub type DoubleRNSRing<NumberRing, A = Global> = RingValue<DoubleRNSRingBase<NumberRing, A>>;

///
/// A [`DoubleRNSRing`] element, stored by its coefficients w.r.t. the "mult basis".
/// In particular, this is the only representation that allows for multiplications.
/// 
pub struct DoubleRNSEl<NumberRing, A = Global>
    where NumberRing: HENumberRing,
        A: Allocator + Clone
{
    pub(super) number_ring: PhantomData<NumberRing>,
    pub(super) allocator: PhantomData<A>,
    pub(super) el_wrt_mult_basis: Vec<zn_64::ZnEl, A>
}

///
/// A [`DoubleRNSRing`] element, stored by its coefficients w.r.t. the "small basis".
/// 
pub struct SmallBasisEl<NumberRing, A = Global>
    where NumberRing: HENumberRing,
        A: Allocator + Clone
{
    number_ring: PhantomData<NumberRing>,
    allocator: PhantomData<A>,
    el_wrt_small_basis: Vec<zn_64::ZnEl, A>
}

impl<NumberRing> DoubleRNSRingBase<NumberRing> 
    where NumberRing: HENumberRing
{
    pub fn new(number_ring: NumberRing, rns_base: zn_rns::Zn<zn_64::Zn, BigIntRing>) -> RingValue<Self> {
        Self::new_with(number_ring, rns_base, Global)
    }
}

impl<NumberRing, A> DoubleRNSRingBase<NumberRing, A> 
    where NumberRing: HENumberRing,
        A: Allocator + Clone
{
    pub fn new_with(number_ring: NumberRing, rns_base: zn_rns::Zn<zn_64::Zn, BigIntRing>, allocator: A) -> RingValue<Self> {
        assert!(rns_base.len() > 0);
        RingValue::from(Self {
            ring_decompositions: rns_base.as_iter().map(|Fp| number_ring.mod_p(Fp.clone())).collect(),
            number_ring: number_ring,
            rns_base: rns_base,
            allocator: allocator
        })
    }

    pub fn ring_decompositions(&self) -> &[<NumberRing as HENumberRing>::Decomposed] {
        &self.ring_decompositions
    }

    pub fn rns_base(&self) -> &zn_rns::Zn<zn_64::Zn, BigIntRing> {
        &self.rns_base
    }

    pub fn element_len(&self) -> usize {
        self.rank() * self.rns_base().len()
    }

    pub fn as_matrix_wrt_small_basis<'a>(&self, element: &'a SmallBasisEl<NumberRing, A>) -> Submatrix<'a, AsFirstElement<zn_64::ZnEl>, zn_64::ZnEl> {
        Submatrix::from_1d(&element.el_wrt_small_basis, self.rns_base().len(), self.rank())
    }

    pub fn as_matrix_wrt_small_basis_mut<'a>(&self, element: &'a mut SmallBasisEl<NumberRing, A>) -> SubmatrixMut<'a, AsFirstElement<zn_64::ZnEl>, zn_64::ZnEl> {
        SubmatrixMut::from_1d(&mut element.el_wrt_small_basis, self.rns_base().len(), self.rank())
    }

    pub fn number_ring(&self) -> &NumberRing {
        &self.number_ring
    }

    pub fn undo_fft(&self, element: DoubleRNSEl<NumberRing, A>) -> SmallBasisEl<NumberRing, A> {
        record_time!(GLOBAL_TIME_RECORDER, "DoubleRNSRing::undo_fft", || {
            assert_eq!(element.el_wrt_mult_basis.len(), self.element_len());
            let mut result = element.el_wrt_mult_basis;
            for i in 0..self.rns_base().len() {
                self.ring_decompositions[i].mult_basis_to_small_basis(&mut result[(i * self.rank())..((i + 1) * self.rank())]);
            }
            SmallBasisEl {
                el_wrt_small_basis: result,
                number_ring: PhantomData,
                allocator: PhantomData
            }
        })
    }

    pub fn allocator(&self) -> &A {
        &self.allocator
    }

    pub fn zero_non_fft(&self) -> SmallBasisEl<NumberRing, A> {
        SmallBasisEl {
            el_wrt_small_basis: self.zero().el_wrt_mult_basis,
            number_ring: PhantomData,
            allocator: PhantomData
        }
    }

    pub fn from_non_fft(&self, x: El<<Self as RingExtension>::BaseRing>) -> SmallBasisEl<NumberRing, A> {
        let mut result = Vec::with_capacity_in(self.element_len(), self.allocator.clone());
        let x = self.base_ring().get_congruence(&x);
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            result.push(Zp.clone_el(x.at(i)));
            for _ in 1..self.rank() {
                result.push(Zp.zero());
            }
            self.ring_decompositions[i].coeff_basis_to_small_basis(&mut result[(i * self.rank())..((i + 1) * self.rank())]);
        }
        SmallBasisEl {
            el_wrt_small_basis: result,
            number_ring: PhantomData,
            allocator: PhantomData
        }
    }

    pub fn do_fft(&self, element: SmallBasisEl<NumberRing, A>) -> DoubleRNSEl<NumberRing, A> {
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

    pub fn sample_from_coefficient_distribution<G: FnMut() -> i32>(&self, mut distribution: G) -> SmallBasisEl<NumberRing, A> {
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
        return SmallBasisEl {
            el_wrt_small_basis: result,
            allocator: PhantomData,
            number_ring: PhantomData
        };
    }

    pub fn clone_el_non_fft(&self, val: &SmallBasisEl<NumberRing, A>) -> SmallBasisEl<NumberRing, A> {
        assert_eq!(self.element_len(), val.el_wrt_small_basis.len());
        let mut result = Vec::with_capacity_in(self.element_len(), self.allocator.clone());
        result.extend((0..self.element_len()).map(|i| self.rns_base().at(i / self.rank()).clone_el(&val.el_wrt_small_basis[i])));
        SmallBasisEl {
            el_wrt_small_basis: result,
            number_ring: PhantomData,
            allocator: PhantomData
        }
    }

    pub fn eq_el_non_fft(&self, lhs: &SmallBasisEl<NumberRing, A>, rhs: &SmallBasisEl<NumberRing, A>) -> bool {
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

    pub fn negate_inplace_non_fft(&self, val: &mut SmallBasisEl<NumberRing, A>) {
        assert_eq!(self.element_len(), val.el_wrt_small_basis.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                self.rns_base().at(i).negate_inplace(&mut val.el_wrt_small_basis[i * self.rank() + j]);
            }
        }
    }

    pub fn negate_non_fft(&self, mut val: SmallBasisEl<NumberRing, A>) -> SmallBasisEl<NumberRing, A> {
        self.negate_inplace_non_fft(&mut val);
        return val;
    }

    pub fn sub_assign_non_fft(&self, lhs: &mut SmallBasisEl<NumberRing, A>, rhs: &SmallBasisEl<NumberRing, A>) {
        assert_eq!(self.element_len(), lhs.el_wrt_small_basis.len());
        assert_eq!(self.element_len(), rhs.el_wrt_small_basis.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                self.rns_base().at(i).sub_assign_ref(&mut lhs.el_wrt_small_basis[i * self.rank() + j], &rhs.el_wrt_small_basis[i * self.rank() + j]);
            }
        }
    }

    pub fn mul_scalar_assign_non_fft(&self, lhs: &mut SmallBasisEl<NumberRing, A>, rhs: &El<zn_rns::Zn<zn_64::Zn, BigIntRing>>) {
        assert_eq!(self.element_len(), lhs.el_wrt_small_basis.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                self.rns_base().at(i).mul_assign_ref(&mut lhs.el_wrt_small_basis[i * self.rank() + j], self.rns_base().get_congruence(rhs).at(i));
            }
        }
    }

    pub fn add_assign_non_fft(&self, lhs: &mut SmallBasisEl<NumberRing, A>, rhs: &SmallBasisEl<NumberRing, A>) {
        assert_eq!(self.element_len(), lhs.el_wrt_small_basis.len());
        assert_eq!(self.element_len(), rhs.el_wrt_small_basis.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                self.rns_base().at(i).add_assign_ref(&mut lhs.el_wrt_small_basis[i * self.rank() + j], &rhs.el_wrt_small_basis[i * self.rank() + j]);
            }
        }
    }

    pub fn from_canonical_basis_non_fft<V>(&self, vec: V) -> SmallBasisEl<NumberRing, A>
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
        return SmallBasisEl {
            el_wrt_small_basis: result,
            allocator: PhantomData,
            number_ring: PhantomData
        };
    }

    pub fn wrt_canonical_basis_non_fft<'a>(&'a self, el: SmallBasisEl<NumberRing, A>) -> DoubleRNSRingBaseElVectorRepresentation<'a, NumberRing, A> {
        let mut result = el.el_wrt_small_basis;
        for i in 0..self.rns_base().len() {
            self.ring_decompositions[i].small_basis_to_coeff_basis(&mut result[(i * self.rank())..((i + 1) * self.rank())]);
        }
        return DoubleRNSRingBaseElVectorRepresentation {
            ring: self,
            el_wrt_coeff_basis: result
        };
    }

    pub fn perform_rns_op_from<A2, Op>(
        &self, 
        from: &DoubleRNSRingBase<NumberRing, A2>, 
        el: &SmallBasisEl<NumberRing, A2>, 
        op: &Op
    ) -> SmallBasisEl<NumberRing, A> 
        where NumberRing: HENumberRing,
            A2: Allocator + Clone,
            Op: RNSOperation<RingType = zn_64::ZnBase>
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

    pub fn exact_convert_from_decompring<ZnTy, A2>(
        &self, 
        from: &DecompositionRing<NumberRing, ZnTy, A2>, 
        element: &<DecompositionRingBase<NumberRing, ZnTy, A2> as RingBase>::Element
    ) -> SmallBasisEl<NumberRing, A> 
        where NumberRing: HENumberRing,
            ZnTy: RingStore,
            ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
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
        return SmallBasisEl {
            el_wrt_small_basis: result,
            allocator: PhantomData,
            number_ring: PhantomData
        };
    }

    pub fn perform_rns_op_to_decompring<ZnTy, A2, Op>(
        &self, 
        to: &DecompositionRing<NumberRing, ZnTy, A2>, 
        element: &SmallBasisEl<NumberRing, A>, 
        op: &Op
    ) -> <DecompositionRingBase<NumberRing, ZnTy, A2> as RingBase>::Element 
        where NumberRing: HENumberRing,
            A2: Allocator + Clone,
            ZnTy: RingStore<Type = zn_64::ZnBase>,
            Op: RNSOperation<RingType = zn_64::ZnBase>
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

    pub fn map_in_from_singlerns<A2, C>(&self, from: &SingleRNSRingBase<NumberRing, A2, C>, el: &El<SingleRNSRing<NumberRing, A2, C>>, hom: &<Self as CanHomFrom<SingleRNSRingBase<NumberRing, A2, C>>>::Homomorphism) -> SmallBasisEl<NumberRing, A>
        where NumberRing: HECyclotomicNumberRing,
            A2: Allocator + Clone,
            C: ConvolutionAlgorithm<ZnBase>
    {
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
        SmallBasisEl {
            el_wrt_small_basis: result,
            number_ring: PhantomData,
            allocator: PhantomData
        }
    }

    pub fn map_out_to_singlerns<A2, C>(&self, to: &SingleRNSRingBase<NumberRing, A2, C>, el: SmallBasisEl<NumberRing, A>, iso: &<Self as CanIsoFromTo<SingleRNSRingBase<NumberRing, A2, C>>>::Isomorphism) -> El<SingleRNSRing<NumberRing, A2, C>>
        where NumberRing: HECyclotomicNumberRing,
            A2: Allocator + Clone,
            C: ConvolutionAlgorithm<ZnBase>
    {
        let mut result = to.zero();
        let mut result_matrix = to.as_matrix_mut(&mut result);
        let mut el_coeff = el.el_wrt_small_basis;
        for i in 0..self.rns_base().len() {
            self.ring_decompositions().at(i).small_basis_to_coeff_basis(&mut el_coeff[(i * self.rank())..((i + 1) * self.rank())]);
        }
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            for j in 0..self.rank() {
                *result_matrix.at_mut(i, j) = Zp.get_ring().map_out(to.rns_base().at(i).get_ring(), Zp.clone_el(&el_coeff[i * self.rank() + j]), &iso[i]);
            }
        }
        return result;
    }
}

impl<NumberRing, A> PartialEq for DoubleRNSRingBase<NumberRing, A> 
    where NumberRing: HENumberRing,
        A: Allocator + Clone
{
    fn eq(&self, other: &Self) -> bool {
        self.number_ring == other.number_ring && self.rns_base.get_ring() == other.rns_base.get_ring()
    }
}

impl<NumberRing, A> RingBase for DoubleRNSRingBase<NumberRing, A> 
    where NumberRing: HENumberRing,
        A: Allocator + Clone
{
    type Element = DoubleRNSEl<NumberRing, A>;

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

    fn mul_assign_int(&self, lhs: &mut Self::Element, rhs: i32) {
        assert_eq!(self.element_len(), lhs.el_wrt_mult_basis.len());
        for i in 0..self.rns_base().len() {
            let rhs_mod_p = self.rns_base().at(i).get_ring().from_int(rhs);
            for j in 0..self.rank() {
                self.rns_base().at(i).mul_assign_ref(&mut lhs.el_wrt_mult_basis[i * self.rank() + j], &rhs_mod_p);
            }
        }
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

impl<NumberRing, A> CyclotomicRing for DoubleRNSRingBase<NumberRing, A> 
    where NumberRing: HECyclotomicNumberRing,
        A: Allocator + Clone
{
    fn n(&self) -> u64 {
        self.number_ring.n()
    }

    fn apply_galois_action(&self, el: &Self::Element, g: CyclotomicGaloisGroupEl) -> Self::Element {
        record_time!(GLOBAL_TIME_RECORDER, "DoubleRNSRing::apply_galois_action", || {
            let mut result = self.zero();
            for (i, _) in self.rns_base().as_iter().enumerate() {
                <NumberRing::DecomposedAsCyclotomic>::from_ref(&self.ring_decompositions()[i]).permute_galois_action(
                    &el.el_wrt_mult_basis[(i * self.rank())..((i + 1) * self.rank())],
                    &mut result.el_wrt_mult_basis[(i * self.rank())..((i + 1) * self.rank())],
                    g
                );
            }
            return result;
        })
    }
}

impl<NumberRing, A> DivisibilityRing for DoubleRNSRingBase<NumberRing, A> 
    where NumberRing: HENumberRing,
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

pub struct DoubleRNSRingBaseElVectorRepresentation<'a, NumberRing, A> 
    where NumberRing: HENumberRing,
        A: Allocator + Clone
{
    el_wrt_coeff_basis: Vec<zn_64::ZnEl, A>,
    ring: &'a DoubleRNSRingBase<NumberRing, A>
}

impl<'a, NumberRing, A> VectorFn<El<zn_rns::Zn<zn_64::Zn, BigIntRing>>> for DoubleRNSRingBaseElVectorRepresentation<'a, NumberRing, A> 
    where NumberRing: HENumberRing,
        A: Allocator + Clone
{
    fn len(&self) -> usize {
        self.ring.rank()
    }

    fn at(&self, i: usize) -> El<zn_rns::Zn<zn_64::Zn, BigIntRing>> {
        assert!(i < self.len());
        self.ring.rns_base().from_congruence(self.el_wrt_coeff_basis[i..].iter().step_by(self.ring.rank()).enumerate().map(|(i, x)| self.ring.rns_base().at(i).clone_el(x)))
    }
}

impl<NumberRing, A> FreeAlgebra for DoubleRNSRingBase<NumberRing, A> 
    where NumberRing: HENumberRing,
        A: Allocator + Clone
{
    type VectorRepresentation<'a> = DoubleRNSRingBaseElVectorRepresentation<'a, NumberRing, A> 
        where Self: 'a;

    fn canonical_gen(&self) -> Self::Element {
        let mut result = self.zero_non_fft().el_wrt_small_basis;
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            result[i * self.rank() + 1] = Zp.one();
            self.ring_decompositions[i].coeff_basis_to_small_basis(&mut result[(i * self.rank())..((i + 1) * self.rank())]);
        }
        return self.do_fft(SmallBasisEl {
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

impl<NumberRing, A> RingExtension for DoubleRNSRingBase<NumberRing, A> 
    where NumberRing: HENumberRing,
        A: Allocator + Clone
{
    type BaseRing = zn_rns::Zn<zn_64::Zn, BigIntRing>;

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

pub struct WRTCanonicalBasisElementCreator<'a, NumberRing, A>
    where NumberRing: HENumberRing,
        A: Allocator + Clone
{
    ring: &'a DoubleRNSRingBase<NumberRing, A>
}

impl<'a, 'b, NumberRing, A> Clone for WRTCanonicalBasisElementCreator<'a, NumberRing, A>
    where NumberRing: HENumberRing,
        A: Allocator + Clone
{
    fn clone(&self) -> Self {
        Self { ring: self.ring }
    }
}

impl<'a, 'b, NumberRing, A> Fn<(&'b [El<zn_rns::Zn<zn_64::Zn, BigIntRing>>],)> for WRTCanonicalBasisElementCreator<'a, NumberRing, A>
    where NumberRing: HENumberRing,
        A: Allocator + Clone
{
    extern "rust-call" fn call(&self, args: (&'b [El<zn_rns::Zn<zn_64::Zn, BigIntRing>>],)) -> Self::Output {
        self.ring.from_canonical_basis(args.0.iter().map(|x| self.ring.base_ring().clone_el(x)))
    }
}

impl<'a, 'b, NumberRing, A> FnMut<(&'b [El<zn_rns::Zn<zn_64::Zn, BigIntRing>>],)> for WRTCanonicalBasisElementCreator<'a, NumberRing, A>
    where NumberRing: HENumberRing,
        A: Allocator + Clone
{
    extern "rust-call" fn call_mut(&mut self, args: (&'b [El<zn_rns::Zn<zn_64::Zn, BigIntRing>>],)) -> Self::Output {
        self.call(args)
    }
}

impl<'a, 'b, NumberRing, A> FnOnce<(&'b [El<zn_rns::Zn<zn_64::Zn, BigIntRing>>],)> for WRTCanonicalBasisElementCreator<'a, NumberRing, A>
    where NumberRing: HENumberRing,
        A: Allocator + Clone
{
    type Output = El<DoubleRNSRing<NumberRing, A>>;

    extern "rust-call" fn call_once(self, args: (&'b [El<zn_rns::Zn<zn_64::Zn, BigIntRing>>],)) -> Self::Output {
        self.call(args)
    }
}

impl<NumberRing, A> FiniteRingSpecializable for DoubleRNSRingBase<NumberRing, A> 
    where NumberRing: HENumberRing,
        A: Allocator + Clone
{
    fn specialize<O: FiniteRingOperation<Self>>(op: O) -> Result<O::Output, ()> {
        Ok(op.execute())
    }
}

impl<NumberRing, A> FiniteRing for DoubleRNSRingBase<NumberRing, A> 
    where NumberRing: HENumberRing,
        A: Allocator + Clone
{
    type ElementsIter<'a> = MultiProduct<
        <zn_rns::ZnBase<zn_64::Zn, BigIntRing> as FiniteRing>::ElementsIter<'a>, 
        WRTCanonicalBasisElementCreator<'a, NumberRing, A>, 
        CloneRingEl<&'a zn_rns::Zn<zn_64::Zn, BigIntRing>>,
        El<DoubleRNSRing<NumberRing, A>>
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

impl<NumberRing, A1, A2> CanHomFrom<DoubleRNSRingBase<NumberRing, A2>> for DoubleRNSRingBase<NumberRing, A1>
    where NumberRing: HENumberRing,
        A1: Allocator + Clone,
        A2: Allocator + Clone,
{
    type Homomorphism = Vec<<zn_64::ZnBase as CanHomFrom<zn_64::ZnBase>>::Homomorphism>;

    fn has_canonical_hom(&self, from: &DoubleRNSRingBase<NumberRing, A2>) -> Option<Self::Homomorphism> {
        if self.rns_base().len() == from.rns_base().len() && self.number_ring() == from.number_ring() {
            (0..self.rns_base().len()).map(|i| self.rns_base().at(i).get_ring().has_canonical_hom(from.rns_base().at(i).get_ring()).ok_or(())).collect::<Result<Vec<_>, ()>>().ok()
        } else {
            None
        }
    }

    fn map_in(&self, from: &DoubleRNSRingBase<NumberRing, A2>, el: <DoubleRNSRingBase<NumberRing, A2> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in_ref(from, &el, hom)
    }

    fn map_in_ref(&self, from: &DoubleRNSRingBase<NumberRing, A2>, el: &<DoubleRNSRingBase<NumberRing, A2> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
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

impl<NumberRing, A1, A2, C2> CanHomFrom<SingleRNSRingBase<NumberRing, A2, C2>> for DoubleRNSRingBase<NumberRing, A1>
    where NumberRing: HECyclotomicNumberRing,
        A1: Allocator + Clone,
        A2: Allocator + Clone,
        C2: ConvolutionAlgorithm<zn_64::ZnBase>
{
    type Homomorphism = Vec<<zn_64::ZnBase as CanHomFrom<zn_64::ZnBase>>::Homomorphism>;

    fn has_canonical_hom(&self, from: &SingleRNSRingBase<NumberRing, A2, C2>) -> Option<Self::Homomorphism> {
        if self.rns_base().len() == from.rns_base().len() && self.number_ring() == from.number_ring() {
            (0..self.rns_base().len()).map(|i| self.rns_base().at(i).get_ring().has_canonical_hom(from.rns_base().at(i).get_ring()).ok_or(())).collect::<Result<Vec<_>, ()>>().ok()
        } else {
            None
        }
    }

    fn map_in(&self, from: &SingleRNSRingBase<NumberRing, A2, C2>, el: <SingleRNSRingBase<NumberRing, A2, C2> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in_ref(from, &el, hom)
    }

    fn map_in_ref(&self, from: &SingleRNSRingBase<NumberRing, A2, C2>, el: &<SingleRNSRingBase<NumberRing, A2, C2> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.do_fft(self.map_in_from_singlerns(from, el, hom))
    }
}

impl<NumberRing, A1, A2, C2> CanIsoFromTo<SingleRNSRingBase<NumberRing, A2, C2>> for DoubleRNSRingBase<NumberRing, A1>
    where NumberRing: HECyclotomicNumberRing,
        A1: Allocator + Clone,
        A2: Allocator + Clone,
        C2: ConvolutionAlgorithm<zn_64::ZnBase>
{
    type Isomorphism = Vec<<zn_64::ZnBase as CanIsoFromTo<zn_64::ZnBase>>::Isomorphism>;

    fn has_canonical_iso(&self, from: &SingleRNSRingBase<NumberRing, A2, C2>) -> Option<Self::Isomorphism> {
        if self.rns_base().len() == from.rns_base().len() && self.number_ring() == from.number_ring() {
            (0..self.rns_base().len()).map(|i| self.rns_base().at(i).get_ring().has_canonical_iso(from.rns_base().at(i).get_ring()).ok_or(())).collect::<Result<Vec<_>, ()>>().ok()
        } else {
            None
        }
    }

    fn map_out(&self, from: &SingleRNSRingBase<NumberRing, A2, C2>, el: <Self as RingBase>::Element, iso: &Self::Isomorphism) -> <SingleRNSRingBase<NumberRing, A2, C2> as RingBase>::Element {
        self.map_out_to_singlerns(from, self.undo_fft(el), iso)
    }
}

impl<NumberRing, A1, A2> CanIsoFromTo<DoubleRNSRingBase<NumberRing, A2>> for DoubleRNSRingBase<NumberRing, A1>
    where NumberRing: HENumberRing,
        A1: Allocator + Clone,
        A2: Allocator + Clone,
{
    type Isomorphism = Vec<<zn_64::ZnBase as CanIsoFromTo<zn_64::ZnBase>>::Isomorphism>;

    fn has_canonical_iso(&self, from: &DoubleRNSRingBase<NumberRing, A2>) -> Option<Self::Isomorphism> {
        if self.rns_base().len() == from.rns_base().len() && self.number_ring() == from.number_ring() {
            (0..self.rns_base().len()).map(|i| self.rns_base().at(i).get_ring().has_canonical_iso(from.rns_base().at(i).get_ring()).ok_or(())).collect::<Result<Vec<_>, ()>>().ok()
        } else {
            None
        }
    }

    fn map_out(&self, from: &DoubleRNSRingBase<NumberRing, A2>, el: Self::Element, iso: &Self::Isomorphism) -> <DoubleRNSRingBase<NumberRing, A2> as RingBase>::Element {
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
pub fn test_with_number_ring<NumberRing: Clone + HECyclotomicNumberRing>(number_ring: NumberRing) {
    use crate::{profiling::{clear_all_timings, print_all_timings}, rings::ntt_convolution::NTTConv};

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

    let single_rns_ring = SingleRNSRingBase::<_, _, NTTConv<_>>::new(number_ring.clone(), base_ring.clone());
    feanor_math::ring::generic_tests::test_hom_axioms(&ring, &single_rns_ring, elements.iter().map(|x| ring.clone_el(x)));
}
