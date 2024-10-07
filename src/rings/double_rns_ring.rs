use std::alloc::Allocator;
use std::alloc::Global;
use std::marker::PhantomData;

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

use crate::cyclotomic::CyclotomicRing;
use crate::rings::decomposition::*;
use crate::rnsconv::*;

///
/// The ring `R/qR` specified by a collection of [`RingDecomposition`] for all prime factors `p | q`. 
/// Elements are (by default) stored in double-RNS-representation for efficient arithmetic.
/// 
/// When necessary, it is also possible by using [`DoubleRNSRingBase::do_fft()`] and
/// [`DoubleRNSRingBase::undo_fft()`] to work with ring elements not in double-RNS-representation,
/// but note that multiplication is not available for those.
/// 
pub struct DoubleRNSRingBase<NumberRing, FpTy, A = Global> 
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    number_ring: NumberRing,
    ring_decompositions: Vec<<NumberRing as DecomposableNumberRing<FpTy>>::Decomposed>,
    rns_base: zn_rns::Zn<FpTy, BigIntRing>,
    allocator: A
}

pub type DoubleRNSRing<NumberRing, FpTy, A = Global> = RingValue<DoubleRNSRingBase<NumberRing, FpTy, A>>;

pub struct DoubleRNSEl<NumberRing, FpTy, A = Global>
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    pub(super) number_ring: PhantomData<NumberRing>,
    pub(super) allocator: PhantomData<A>,
    pub(super) data: Vec<El<FpTy>, A>
}

pub struct CoeffEl<NumberRing, FpTy, A = Global>
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    pub(super) number_ring: PhantomData<NumberRing>,
    pub(super) allocator: PhantomData<A>,
    pub(super) data: Vec<El<FpTy>, A>
}

impl<NumberRing, FpTy, A> DoubleRNSRingBase<NumberRing, FpTy, A> 
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    pub fn new_with(number_ring: NumberRing, rns_base: zn_rns::Zn<FpTy, BigIntRing>, allocator: A) -> Self {
        assert!(rns_base.len() > 0);
        Self {
            ring_decompositions: rns_base.as_iter().map(|Fp| number_ring.mod_p(Fp.clone())).collect(),
            number_ring: number_ring,
            rns_base: rns_base,
            allocator: allocator
        }
    }

    pub fn ring_decompositions(&self) -> &[<NumberRing as DecomposableNumberRing<FpTy>>::Decomposed] {
        &self.ring_decompositions
    }

    pub fn rns_base(&self) -> &zn_rns::Zn<FpTy, BigIntRing> {
        &self.rns_base
    }

    pub fn element_len(&self) -> usize {
        self.rank() * self.rns_base().len()
    }

    pub fn as_matrix<'a>(&self, element: &'a CoeffEl<NumberRing, FpTy, A>) -> Submatrix<'a, AsFirstElement<El<FpTy>>, El<FpTy>> {
        Submatrix::from_1d(&element.data, self.rns_base().len(), self.rank())
    }

    pub fn as_matrix_mut<'a>(&self, element: &'a mut CoeffEl<NumberRing, FpTy, A>) -> SubmatrixMut<'a, AsFirstElement<El<FpTy>>, El<FpTy>> {
        SubmatrixMut::from_1d(&mut element.data, self.rns_base().len(), self.rank())
    }

    ///
    /// Returns `a mod pi` where `a` is the coefficient belonging to `X^j` of the given element.
    /// 
    /// Here `pi` is the `i`-th prime divisor of the base ring (using the order exposed by
    /// [`zn_rns::ZnBase`]).
    /// 
    pub fn at<'a>(&self, i: usize, j: usize, el: &'a CoeffEl<NumberRing, FpTy, A>) -> &'a El<FpTy> {
        &el.data[i * self.rank() + j]
    }

    ///
    /// Returns the `(i, j)`-th component of the element in double-RNS-representation. 
    /// 
    /// Note that strictly speaking, these components are indexed by prime divisors `p | q` and
    /// `k in (Z/nZ)*` (assuming a root of unity `zeta` in `Zq` is fixed). Then, the `(p, k)`-th double-RNS
    /// components of a ring element `a` would be `a mod (p, X - zeta^k) in Fp`. However, for easier 
    /// handling in the program, we index all prime divisors `p | q` by `i` and all `k in (Z/nZ)*` by `j`.
    /// 
    /// The indexing of the primes is consistent with the order of the primes in the base ring (of type
    /// [`zn_rns::ZnBase`]). The indexing of the `k` is quite unpredictable, as it depends on the implementation
    /// of the underlying [`RingDecomposition`] (in particular, if it is implemented using standard FFTs, then
    /// that in turn depends on the ordering used by [`feanor_math::algorithms::fft::FFTAlgorithm::unordered_fft()`]).
    /// Therefore, you should not rely on any specific relationship between `j` and `k`, except that it will
    /// remain constant during the lifetime of the ring. Note also that changing the order corresponds to an
    /// automorphism of the ring.
    /// 
    pub fn doublerns_coeff_at<'a>(&self, i: usize, j: usize, el: &'a DoubleRNSEl<NumberRing, FpTy, A>) -> &'a El<FpTy> {
        &el.data[i * self.rank() + j]
    }

    /// 
    /// Returns `a mod pi` where `a` is the coefficient belonging to `X^j` of the given element.
    /// 
    /// See [`Self::at()`] for details.
    /// 
    pub fn at_mut<'a>(&self, i: usize, j: usize, el: &'a mut CoeffEl<NumberRing, FpTy, A>) -> &'a mut El<FpTy> {
        &mut el.data[i * self.rank() + j]
    }

    ///
    /// Returns the `(i, j)`-th component of the element in double-RNS-representation. 
    /// 
    /// See [`Self::doublerns_coeff_at()`] for details.
    /// 
    pub fn doublerns_coeff_at_mut<'a>(&self, i: usize, j: usize, el: &'a mut DoubleRNSEl<NumberRing, FpTy, A>) -> &'a mut El<FpTy> {
        &mut el.data[i * self.rank() + j]
    }

    pub fn undo_fft(&self, mut element: DoubleRNSEl<NumberRing, FpTy, A>) -> CoeffEl<NumberRing, FpTy, A> {
        assert_eq!(element.data.len(), self.element_len());
        timed!("undo_fft", || {
            for i in 0..self.rns_base().len() {
                self.ring_decompositions[i].fft_backward(&mut element.data[(i * self.rank())..((i + 1) * self.rank())]);
            }
        });
        CoeffEl {
            data: element.data,
            number_ring: PhantomData,
            allocator: PhantomData
        }
    }

    pub fn allocator(&self) -> &A {
        &self.allocator
    }

    pub fn non_fft_zero(&self) -> CoeffEl<NumberRing, FpTy, A> {
        CoeffEl {
            data: self.zero().data,
            number_ring: PhantomData,
            allocator: PhantomData
        }
    }

    pub fn non_fft_from(&self, x: El<<Self as RingExtension>::BaseRing>) -> CoeffEl<NumberRing, FpTy, A> {
        let mut result = Vec::with_capacity_in(self.element_len(), self.allocator.clone());
        let x = self.base_ring().get_congruence(&x);
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            result.push(Zp.clone_el(x.at(i)));
            for _ in 1..self.rank() {
                result.push(Zp.zero());
            }
        }
        CoeffEl {
            data: result,
            number_ring: PhantomData,
            allocator: PhantomData
        }
    }

    pub fn do_fft(&self, mut element: CoeffEl<NumberRing, FpTy, A>) -> DoubleRNSEl<NumberRing, FpTy, A> {
        assert_eq!(element.data.len(), self.element_len());
        timed!("do_fft", || {
            for i in 0..self.rns_base().len() {
                self.ring_decompositions[i].fft_forward(&mut element.data[(i * self.rank())..((i + 1) * self.rank())]);
            }
        });
        DoubleRNSEl {
            data: element.data,
            number_ring: PhantomData,
            allocator: PhantomData
        }
    }

    pub fn sample_from_coefficient_distribution<G: FnMut() -> i32>(&self, mut distribution: G) -> CoeffEl<NumberRing, FpTy, A> {
        let mut result = self.non_fft_zero();
        let mut data = Vec::new();
        for j in 0..self.rank() {
            let c = distribution();
            data.push(c);
            for i in 0..self.rns_base().len() {
                result.data[j + i * self.rank()] = self.rns_base().at(i).int_hom().map(c);
            }
        }
        return result;
    }

    pub fn sample_uniform<G: FnMut() -> u64>(&self, mut rng: G) -> DoubleRNSEl<NumberRing, FpTy, A> {
        let mut result = self.zero();
        for j in 0..self.rank() {
            for i in 0..self.rns_base().len() {
                result.data[j + i * self.rank()] = self.rns_base().at(i).random_element(&mut rng);
            }
        }
        return result;
    }

    pub fn clone_el_non_fft(&self, val: &CoeffEl<NumberRing, FpTy, A>) -> CoeffEl<NumberRing, FpTy, A> {
        assert_eq!(self.element_len(), val.data.len());
        let mut result = Vec::with_capacity_in(self.element_len(), self.allocator.clone());
        result.extend((0..self.element_len()).map(|i| self.rns_base().at(i / self.rank()).clone_el(&val.data[i])));
        CoeffEl {
            data: result,
            number_ring: PhantomData,
            allocator: PhantomData
        }
    }

    pub fn negate_non_fft(&self, mut val: CoeffEl<NumberRing, FpTy, A>) -> CoeffEl<NumberRing, FpTy, A> {
        assert_eq!(self.element_len(), val.data.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                self.rns_base().at(i).negate_inplace(&mut val.data[i * self.rank() + j]);
            }
        }
        return val;
    }

    pub fn sub_assign_non_fft(&self, lhs: &mut CoeffEl<NumberRing, FpTy, A>, rhs: &CoeffEl<NumberRing, FpTy, A>) {
        assert_eq!(self.element_len(), lhs.data.len());
        assert_eq!(self.element_len(), rhs.data.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                self.rns_base().at(i).sub_assign_ref(&mut lhs.data[i * self.rank() + j], &rhs.data[i * self.rank() + j]);
            }
        }
    }

    pub fn mul_scalar_assign_non_fft(&self, lhs: &mut CoeffEl<NumberRing, FpTy, A>, rhs: &El<zn_rns::Zn<FpTy, BigIntRing>>) {
        assert_eq!(self.element_len(), lhs.data.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                self.rns_base().at(i).mul_assign_ref(&mut lhs.data[i * self.rank() + j], self.rns_base().get_congruence(rhs).at(i));
            }
        }
    }

    pub fn add_assign_non_fft(&self, lhs: &mut CoeffEl<NumberRing, FpTy, A>, rhs: &CoeffEl<NumberRing, FpTy, A>) {
        assert_eq!(self.element_len(), lhs.data.len());
        assert_eq!(self.element_len(), rhs.data.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                self.rns_base().at(i).add_assign_ref(&mut lhs.data[i * self.rank() + j], &rhs.data[i * self.rank() + j]);
            }
        }
    }

    pub fn wrt_canonical_basis_non_fft<'a>(&'a self, el: &'a CoeffEl<NumberRing, FpTy, A>) -> DoubleRNSRingBaseElVectorRepresentation<'a, NumberRing, FpTy, A> {
        DoubleRNSRingBaseElVectorRepresentation {
            ring: self,
            inv_fft_data: self.clone_el_non_fft(el)
        }
    }
}

impl<NumberRing, FpTy, A> PartialEq for DoubleRNSRingBase<NumberRing, FpTy, A> 
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    fn eq(&self, other: &Self) -> bool {
        self.number_ring == other.number_ring && self.rns_base.get_ring() == other.rns_base.get_ring()
    }
}

impl<NumberRing, FpTy, A> RingBase for DoubleRNSRingBase<NumberRing, FpTy, A> 
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    type Element = DoubleRNSEl<NumberRing, FpTy, A>;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        assert_eq!(self.element_len(), val.data.len());
        let mut result = Vec::with_capacity_in(self.element_len(), self.allocator.clone());
        result.extend((0..self.element_len()).map(|i| self.rns_base().at(i / self.rank()).clone_el(&val.data[i])));
        DoubleRNSEl {
            data: result,
            number_ring: PhantomData,
            allocator: PhantomData
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
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                self.rns_base().at(i).mul_assign_ref(&mut lhs.data[i * self.rank() + j], &rhs.data[i * self.rank() + j]);
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
            data: result,
            number_ring: PhantomData,
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

    fn square(&self, value: &mut Self::Element) {
        assert_eq!(self.element_len(), value.data.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                self.rns_base().at(i).square(&mut value.data[i * self.rank() + j]);
            }
        }
    }

    fn characteristic<I: IntegerRingStore>(&self, ZZ: &I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        self.base_ring().characteristic(ZZ)     
    }
}

impl<NumberRing, FpTy, A> DivisibilityRing for DoubleRNSRingBase<NumberRing, FpTy, A> 
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        let mut result = Vec::with_capacity_in(self.element_len(), self.allocator.clone());
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            for j in 0..self.rank() {
                result.push(Zp.checked_div(&lhs.data[i * self.rank() + j], &rhs.data[i * self.rank() + j])?);
            }
        }
        return Some(DoubleRNSEl { data: result, number_ring: PhantomData, allocator: PhantomData })
    }

    fn is_unit(&self, x: &Self::Element) -> bool {
        x.data.iter().enumerate().all(|(index, c)| self.rns_base().at(index / self.rank()).is_unit(c))
    }
}

pub struct DoubleRNSRingBaseElVectorRepresentation<'a, NumberRing, FpTy, A> 
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    inv_fft_data: CoeffEl<NumberRing, FpTy, A>,
    ring: &'a DoubleRNSRingBase<NumberRing, FpTy, A>
}

impl<'a, NumberRing, FpTy, A> VectorFn<El<zn_rns::Zn<FpTy, BigIntRing>>> for DoubleRNSRingBaseElVectorRepresentation<'a, NumberRing, FpTy, A> 
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    fn len(&self) -> usize {
        self.ring.rank()
    }

    fn at(&self, i: usize) -> El<zn_rns::Zn<FpTy, BigIntRing>> {
        self.ring.rns_base().from_congruence(self.inv_fft_data.data[i..].iter().step_by(self.ring.rank()).enumerate().map(|(i, x)| self.ring.rns_base().at(i).clone_el(x)))
    }
}

impl<NumberRing, FpTy, A> FreeAlgebra for DoubleRNSRingBase<NumberRing, FpTy, A> 
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    type VectorRepresentation<'a> = DoubleRNSRingBaseElVectorRepresentation<'a, NumberRing, FpTy, A> 
        where Self: 'a;

    fn canonical_gen(&self) -> Self::Element {
        let mut result = self.non_fft_zero();
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            result.data[i * self.rank() + 1] = Zp.one();
        }
        return self.do_fft(result);
    }

    fn rank(&self) -> usize {
        self.ring_decompositions[0].rank()
    }

    fn wrt_canonical_basis<'a>(&'a self, el: &'a Self::Element) -> Self::VectorRepresentation<'a> {
        DoubleRNSRingBaseElVectorRepresentation {
            ring: self,
            inv_fft_data: self.undo_fft(self.clone_el(el))
        }
    }

    fn from_canonical_basis<V>(&self, vec: V) -> Self::Element
        where V: IntoIterator<Item = El<Self::BaseRing>>
    {
        let mut result = self.non_fft_zero();
        for (j, x) in vec.into_iter().enumerate() {
            let congruence = self.base_ring().get_ring().get_congruence(&x);
            for i in 0..self.rns_base().len() {
                result.data[i * self.rank() + j] = self.rns_base().at(i).clone_el(congruence.at(i));
            }
        }
        return self.do_fft(result);
    }
}

impl<NumberRing, FpTy, A> RingExtension for DoubleRNSRingBase<NumberRing, FpTy, A> 
    where NumberRing: DecomposableNumberRing<FpTy>,
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
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            for _ in 0..self.rank() {
                result.push(Zp.clone_el(x_congruence.at(i)));
            }
        }
        return DoubleRNSEl {
            data: result,
            number_ring: PhantomData,
            allocator: PhantomData
        };
    }
}

pub struct WRTCanonicalBasisElementCreator<'a, NumberRing, FpTy, A>
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    ring: &'a DoubleRNSRingBase<NumberRing, FpTy, A>
}

impl<'a, 'b, NumberRing, FpTy, A> Clone for WRTCanonicalBasisElementCreator<'a, NumberRing, FpTy, A>
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    fn clone(&self) -> Self {
        Self { ring: self.ring }
    }
}

impl<'a, 'b, NumberRing, FpTy, A> Fn<(&'b [El<zn_rns::Zn<FpTy, BigIntRing>>],)> for WRTCanonicalBasisElementCreator<'a, NumberRing, FpTy, A>
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    extern "rust-call" fn call(&self, args: (&'b [El<zn_rns::Zn<FpTy, BigIntRing>>],)) -> Self::Output {
        self.ring.from_canonical_basis(args.0.iter().map(|x| self.ring.base_ring().clone_el(x)))
    }
}

impl<'a, 'b, NumberRing, FpTy, A> FnMut<(&'b [El<zn_rns::Zn<FpTy, BigIntRing>>],)> for WRTCanonicalBasisElementCreator<'a, NumberRing, FpTy, A>
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    extern "rust-call" fn call_mut(&mut self, args: (&'b [El<zn_rns::Zn<FpTy, BigIntRing>>],)) -> Self::Output {
        self.call(args)
    }
}

impl<'a, 'b, NumberRing, FpTy, A> FnOnce<(&'b [El<zn_rns::Zn<FpTy, BigIntRing>>],)> for WRTCanonicalBasisElementCreator<'a, NumberRing, FpTy, A>
    where NumberRing: DecomposableNumberRing<FpTy>,
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
    where NumberRing: DecomposableNumberRing<FpTy>,
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

    fn size<I: IntegerRingStore>(&self, ZZ: &I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        let modulus = self.base_ring().size(ZZ)?;
        if ZZ.get_ring().representable_bits().is_none() || ZZ.get_ring().representable_bits().unwrap() >= self.rank() * ZZ.abs_log2_ceil(&modulus).unwrap() {
            Some(ZZ.pow(modulus, self.rank()))
        } else {
            None
        }
    }

    fn random_element<G: FnMut() -> u64>(&self, rng: G) -> <Self as RingBase>::Element {
        self.sample_uniform(rng)
    }
}

impl<NumberRing, FpTy1, FpTy2, A1, A2> CanHomFrom<DoubleRNSRingBase<NumberRing, FpTy2, A2>> for DoubleRNSRingBase<NumberRing, FpTy1, A1>
    where NumberRing: DecomposableNumberRing<FpTy1> + DecomposableNumberRing<FpTy2>,

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
        if self.rns_base().len() == from.rns_base().len() && self.number_ring == from.number_ring {
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
                result.push(Zp.get_ring().map_in_ref(from.rns_base().at(i).get_ring(), &el.data[i * self.rank() + j], &hom[i]));
            }
        }
        DoubleRNSEl {
            data: result,
            number_ring: PhantomData,
            allocator: PhantomData
        }
    }
}

impl<NumberRing, FpTy1, FpTy2, A1, A2> CanIsoFromTo<DoubleRNSRingBase<NumberRing, FpTy2, A2>> for DoubleRNSRingBase<NumberRing, FpTy1, A1>
    where NumberRing: DecomposableNumberRing<FpTy1> + DecomposableNumberRing<FpTy2>,

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
        if self.rns_base().len() == from.rns_base().len() && self.number_ring == from.number_ring {
            (0..self.rns_base().len()).map(|i| self.rns_base().at(i).get_ring().has_canonical_iso(from.rns_base().at(i).get_ring()).ok_or(())).collect::<Result<Vec<_>, ()>>().ok()
        } else {
            None
        }
    }

    fn map_out(&self, from: &DoubleRNSRingBase<NumberRing, FpTy2, A2>, el: Self::Element, iso: &Self::Isomorphism) -> <DoubleRNSRingBase<NumberRing, FpTy2, A2> as RingBase>::Element {
        let mut result = Vec::with_capacity_in(from.element_len(), from.allocator.clone());
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            for j in 0..self.rank() {
                result.push(Zp.get_ring().map_out(from.rns_base().at(i).get_ring(), Zp.clone_el(&el.data[i * self.rank() + j]), &iso[i]));
            }
        }
        DoubleRNSEl {
            data: result,
            number_ring: PhantomData,
            allocator: PhantomData
        }
    }
}

#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use crate::rnsconv::lift::*;