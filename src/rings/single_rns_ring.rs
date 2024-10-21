use std::alloc::{Allocator, Global};
use std::marker::PhantomData;

use feanor_math::algorithms::convolution::fft::FFTBasedConvolutionZn;
use feanor_math::algorithms::convolution::{ConvolutionAlgorithm, KaratsubaAlgorithm, STANDARD_CONVOLUTION};
use feanor_math::ring::*;
use feanor_math::homomorphism::*;
use feanor_math::integer::*;
use feanor_math::rings::extension::*;
use feanor_math::rings::poly::sparse_poly::SparsePolyRing;
use feanor_math::rings::poly::dense_poly::DensePolyRing;
use feanor_math::rings::poly::*;
use feanor_math::rings::zn::*;
use feanor_math::seq::sparse::SparseMapVector;
use feanor_math::seq::*;
use feanor_math::matrix::*;

use crate::rings::decomposition::*;

use super::double_rns_ring::DoubleRNSRingBase;

pub struct SingleRNSRingBase<NumberRing, FpTy, A = Global, C = KaratsubaAlgorithm> 
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<FpTy::Type>
{
    number_ring: NumberRing,
    rns_base: zn_rns::Zn<FpTy, BigIntRing>,
    convolutions: Vec<C>,
    modulus: SparseMapVector<zn_rns::Zn<FpTy, BigIntRing>>,
    allocator: A
}

pub type SingleRNSRing<NumberRing, FpTy, A = Global, C = KaratsubaAlgorithm> = RingValue<SingleRNSRingBase<NumberRing, FpTy, A, C>>;

pub struct SingleRNSRingEl<NumberRing, FpTy, A = Global, C = KaratsubaAlgorithm>
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone,
        C: ConvolutionAlgorithm<FpTy::Type>
{
    pub(super) number_ring: PhantomData<NumberRing>,
    pub(super) allocator: PhantomData<A>,
    pub(super) convolutions: PhantomData<C>,
    pub(super) data: Vec<El<FpTy>, A>
}

impl<NumberRing, FpTy> SingleRNSRingBase<NumberRing, FpTy> 
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>
{
    pub fn new(number_ring: NumberRing, rns_base: zn_rns::Zn<FpTy, BigIntRing>) -> RingValue<Self> {
        let convolutions = rns_base.as_iter().map(|_| STANDARD_CONVOLUTION).collect();
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
            number_ring: number_ring,
            modulus: modulus,
            convolutions: convolutions,
            rns_base: rns_base,
            allocator: allocator
        })
    }

    pub fn rns_base(&self) -> &zn_rns::Zn<FpTy, BigIntRing> {
        &self.rns_base
    }

    pub fn element_len(&self) -> usize {
        self.rank() * self.rns_base().len()
    }

    pub fn as_matrix<'a>(&self, element: &'a SingleRNSRingEl<NumberRing, FpTy, A, C>) -> Submatrix<'a, AsFirstElement<El<FpTy>>, El<FpTy>> {
        Submatrix::from_1d(&element.data, self.rns_base().len(), self.rank())
    }

    pub fn as_matrix_mut<'a>(&self, element: &'a mut SingleRNSRingEl<NumberRing, FpTy, A, C>) -> SubmatrixMut<'a, AsFirstElement<El<FpTy>>, El<FpTy>> {
        SubmatrixMut::from_1d(&mut element.data, self.rns_base().len(), self.rank())
    }

    pub fn number_ring(&self) -> &NumberRing {
        &self.number_ring
    }

    ///
    /// Returns `a mod pi` where `a` is the coefficient belonging to `X^j` of the given element.
    /// 
    /// Here `pi` is the `i`-th prime divisor of the base ring (using the order exposed by
    /// [`zn_rns::ZnBase`]).
    /// 
    pub fn at<'a>(&self, i: usize, j: usize, el: &'a SingleRNSRingEl<NumberRing, FpTy, A, C>) -> &'a El<FpTy> {
        &el.data[i * self.rank() + j]
    }

    /// 
    /// Returns `a mod pi` where `a` is the coefficient belonging to `X^j` of the given element.
    /// 
    /// See [`Self::at()`] for details.
    /// 
    pub fn at_mut<'a>(&self, i: usize, j: usize, el: &'a mut SingleRNSRingEl<NumberRing, FpTy, A, C>) -> &'a mut El<FpTy> {
        &mut el.data[i * self.rank() + j]
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
        self.ring.rns_base().from_congruence(self.element.data[i..].iter().step_by(self.ring.rank()).enumerate().map(|(i, x)| self.ring.rns_base().at(i).clone_el(x)))
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
        self.number_ring == other.number_ring && self.rns_base.get_ring() == other.rns_base.get_ring()
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
        assert_eq!(self.element_len(), val.data.len());
        let mut result = Vec::with_capacity_in(self.element_len(), self.allocator.clone());
        result.extend((0..self.element_len()).map(|i| self.rns_base().at(i / self.rank()).clone_el(&val.data[i])));
        SingleRNSRingEl {
            data: result,
            number_ring: PhantomData,
            allocator: PhantomData,
            convolutions: PhantomData,
        }
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        assert_eq!(self.element_len(), lhs.data.len());
        assert_eq!(self.element_len(), rhs.data.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                self.rns_base().at(i).add_assign_ref(&mut lhs.data[i * self.rank() + j], &rhs.data[i * self.rank() + j]);
            }
        }
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.add_assign_ref(lhs, &rhs);
    }

    fn sub_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        assert_eq!(self.element_len(), lhs.data.len());
        assert_eq!(self.element_len(), rhs.data.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                self.rns_base().at(i).sub_assign_ref(&mut lhs.data[i * self.rank() + j], &rhs.data[i * self.rank() + j]);
            }
        }
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        assert_eq!(self.element_len(), lhs.data.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                self.rns_base().at(i).negate_inplace(&mut lhs.data[i * self.rank() + j]);
            }
        }
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.mul_assign_ref(lhs, &rhs);
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        assert_eq!(self.element_len(), lhs.data.len());
        assert_eq!(self.element_len(), rhs.data.len());
        
        let mut unreduced_result = Vec::with_capacity_in(2 * self.rank() * self.rns_base.len(), &self.allocator);
        unreduced_result.extend(self.rns_base.as_iter().flat_map(|Zp| (0..(2 * self.rank())).map(|_| Zp.zero())));
        for k in 0..self.rns_base.len() {
            let Zp = self.rns_base.at(k);
            self.convolutions[k].compute_convolution(
                &lhs.data[(k * self.rank())..((k + 1) * self.rank())],
                &rhs.data[(k * self.rank())..((k + 1) * self.rank())],
                &mut unreduced_result[(2 * k * self.rank())..(2 * (k + 1) * self.rank())],
                Zp
            );
        }
        for i in (self.rank()..(2 * self.rank())).rev() {
            for (j, c) in self.modulus.nontrivial_entries() {
                let congruence = self.rns_base.get_congruence(c);
                for k in 0..self.rns_base.len() {
                    let Zp = self.rns_base.at(k);
                    let subtract = Zp.mul_ref(&unreduced_result[2 * k * self.rank() + i], congruence.at(k));
                    Zp.add_assign(&mut unreduced_result[2 * k * self.rank() + i - self.rank() + j], subtract);
                }
            }
        }
        for k in 0..self.rns_base.len() {
            let Zp = self.rns_base.at(k);
            for i in 0..self.rank() {
                lhs.data[k * self.rank() + i] = Zp.clone_el(&unreduced_result[k * 2 * self.rank() + i]);
            }
        }
    }
    
    fn from_int(&self, value: i32) -> Self::Element {
        self.from(self.base_ring().get_ring().from_int(value))
    }

    fn zero(&self) -> Self::Element {
        let mut result = Vec::with_capacity_in(self.element_len(), self.allocator.clone());
        result.extend(self.rns_base().as_iter().flat_map(|Zp| (0..self.rank()).map(|_| Zp.zero())));
        return SingleRNSRingEl {
            data: result,
            number_ring: PhantomData,
            convolutions: PhantomData,
            allocator: PhantomData
        };
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        assert_eq!(self.element_len(), lhs.data.len());
        assert_eq!(self.element_len(), rhs.data.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                if !self.rns_base().at(i).eq_el(&lhs.data[i * self.rank() + j], &rhs.data[i * self.rank() + j]) {
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

    fn characteristic<I: IntegerRingStore>(&self, ZZ: &I) -> Option<El<I>>
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
        &self.rns_base
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        let x_congruence = self.rns_base.get_congruence(&x);
        let mut result = self.zero();
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            result.data[i * self.rank()] = Zp.clone_el(x_congruence.at(i));
        }
        return result;
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
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            result.data[i * self.rank() + 1] = Zp.one();
        }
        return result;
    }

    fn rank(&self) -> usize {
        self.number_ring.rank()
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
        for (j, x) in vec.into_iter().enumerate() {
            let congruence = self.base_ring().get_ring().get_congruence(&x);
            for i in 0..self.rns_base().len() {
                result.data[i * self.rank() + j] = self.rns_base().at(i).clone_el(congruence.at(i));
            }
        }
        return result;
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
        let mut result = Vec::with_capacity_in(self.element_len(), self.allocator.clone());
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            for j in 0..self.rank() {
                result.push(Zp.get_ring().map_in_ref(from.rns_base().at(i).get_ring(), &el.data[i * self.rank() + j], &hom[i]));
            }
        }
        SingleRNSRingEl {
            data: result,
            number_ring: PhantomData,
            convolutions: PhantomData,
            allocator: PhantomData
        }
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
        let mut result = Vec::with_capacity_in(self.element_len(), self.allocator.clone());
        let non_fft_el = from.undo_fft(el);
        let el_as_matrix = from.as_matrix(&non_fft_el);
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            for j in 0..self.rank() {
                result.push(Zp.get_ring().map_in_ref(from.rns_base().at(i).get_ring(), el_as_matrix.at(i, j), &hom[i]));
            }
        }
        SingleRNSRingEl {
            data: result,
            number_ring: PhantomData,
            convolutions: PhantomData,
            allocator: PhantomData
        }
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
        let mut result = Vec::with_capacity_in(from.element_len(), from.allocator.clone());
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            for j in 0..self.rank() {
                result.push(Zp.get_ring().map_out(from.rns_base().at(i).get_ring(), Zp.clone_el(&el.data[i * self.rank() + j]), &iso[i]));
            }
        }
        SingleRNSRingEl {
            data: result,
            number_ring: PhantomData,
            convolutions: PhantomData,
            allocator: PhantomData
        }
    }
}

#[cfg(test)]
pub fn test_with_number_ring<NumberRing: Clone + DecomposableNumberRing<zn_64::Zn>>(number_ring: NumberRing) {
    let p1 = number_ring.largest_suitable_prime(1000).unwrap();
    let p2 = number_ring.largest_suitable_prime(2000).unwrap();
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
        ring.int_hom().mul_map(ring.canonical_gen(), p1 as i32)
    ];

    feanor_math::ring::generic_tests::test_ring_axioms(&ring, elements.iter().map(|x| ring.clone_el(x)));
    feanor_math::ring::generic_tests::test_self_iso(&ring, elements.iter().map(|x| ring.clone_el(x)));
    feanor_math::rings::extension::generic_tests::test_free_algebra_axioms(&ring);

    let double_rns_ring = DoubleRNSRingBase::new(number_ring.clone(), base_ring.clone());
    feanor_math::ring::generic_tests::test_hom_axioms(&ring, &double_rns_ring, elements.iter().map(|x| ring.clone_el(x)));
}