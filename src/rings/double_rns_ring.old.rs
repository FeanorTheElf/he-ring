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
use super::ntt_ring::NTTRingBase;

///
/// The ring `R/qR` specified by a collection of [`RingDecomposition`] for all prime factors `p | q`. 
/// Elements are (by default) stored in double-RNS-representation for efficient arithmetic.
/// 
/// When necessary, it is also possible by using [`DoubleRNSRingBase::do_fft()`] and
/// [`DoubleRNSRingBase::undo_fft()`] to work with ring elements not in double-RNS-representation,
/// but note that multiplication is not available for those.
/// 
pub struct DoubleRNSRingBase<R, F, A = Global> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    ring_decompositions: Vec<F>,
    scalar_ring: zn_rns::Zn<R, BigIntRing>,
    allocator: A
}

pub type DoubleRNSRing<R, F, A = Global> = RingValue<DoubleRNSRingBase<R, F, A>>;

pub struct DoubleRNSEl<R, F, A> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    pub(super) ring_decompositions: PhantomData<F>,
    pub(super) allocator: PhantomData<A>,
    pub(super) data: Vec<El<R>, A>
}

pub struct DoubleRNSNonFFTEl<R, F, A = Global> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    ring_decompositions: PhantomData<F>,
    allocator: PhantomData<A>,
    data: Vec<El<R>, A>
}

impl<R, F, A> DoubleRNSRingBase<R, F, A> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    pub fn from_ring_decompositions(rns_base: zn_rns::Zn<R, BigIntRing>, ring_decompositions: Vec<F>, allocator: A) -> Self {
        assert!(ring_decompositions.len() > 0);
        for i in 0..ring_decompositions.len() {
            assert!(ring_decompositions[i].is_same_number_ring(&ring_decompositions[0]));
            assert_eq!(ring_decompositions[i].rank(), ring_decompositions[0].rank());
        }
        let scalar_ring = rns_base;
        Self { ring_decompositions, allocator, scalar_ring }
    }

    pub fn ring_decompositions(&self) -> &Vec<F> {
        &self.ring_decompositions
    }

    pub fn rns_base(&self) -> &zn_rns::ZnBase<R, BigIntRing> {
        self.scalar_ring.get_ring()
    }

    pub fn element_len(&self) -> usize {
        self.rank() * self.rns_base().len()
    }

    pub fn as_matrix<'a>(&self, element: &'a DoubleRNSNonFFTEl<R, F, A>) -> Submatrix<'a, AsFirstElement<El<R>>, El<R>> {
        Submatrix::from_1d(&element.data, self.rns_base().len(), self.rank())
    }

    pub fn as_matrix_mut<'a>(&self, element: &'a mut DoubleRNSNonFFTEl<R, F, A>) -> SubmatrixMut<'a, AsFirstElement<El<R>>, El<R>> {
        SubmatrixMut::from_1d(&mut element.data, self.rns_base().len(), self.rank())
    }

    ///
    /// Returns `a mod pi` where `a` is the coefficient belonging to `X^j` of the given element.
    /// 
    /// Here `pi` is the `i`-th prime divisor of the base ring (using the order exposed by
    /// [`zn_rns::ZnBase`]).
    /// 
    pub fn at<'a>(&self, i: usize, j: usize, el: &'a DoubleRNSNonFFTEl<R, F, A>) -> &'a El<R> {
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
    pub fn doublerns_coeff_at<'a>(&self, i: usize, j: usize, el: &'a DoubleRNSEl<R, F, A>) -> &'a El<R> {
        &el.data[i * self.rank() + j]
    }

    /// 
    /// Returns `a mod pi` where `a` is the coefficient belonging to `X^j` of the given element.
    /// 
    /// See [`Self::at()`] for details.
    /// 
    pub fn at_mut<'a>(&self, i: usize, j: usize, el: &'a mut DoubleRNSNonFFTEl<R, F, A>) -> &'a mut El<R> {
        &mut el.data[i * self.rank() + j]
    }

    ///
    /// Returns the `(i, j)`-th component of the element in double-RNS-representation. 
    /// 
    /// See [`Self::doublerns_coeff_at()`] for details.
    /// 
    pub fn doublerns_coeff_at_mut<'a>(&self, i: usize, j: usize, el: &'a mut DoubleRNSEl<R, F, A>) -> &'a mut El<R> {
        &mut el.data[i * self.rank() + j]
    }

    pub fn undo_fft(&self, mut element: DoubleRNSEl<R, F, A>) -> DoubleRNSNonFFTEl<R, F, A> {
        assert_eq!(element.data.len(), self.element_len());
        timed!("undo_fft", || {
            for i in 0..self.rns_base().len() {
                self.ring_decompositions[i].fft_backward(&mut element.data[(i * self.rank())..((i + 1) * self.rank())], self.rns_base().at(i).get_ring());
            }
        });
        DoubleRNSNonFFTEl {
            data: element.data,
            ring_decompositions: PhantomData,
            allocator: PhantomData
        }
    }

    pub fn allocator(&self) -> &A {
        &self.allocator
    }

    pub fn non_fft_zero(&self) -> DoubleRNSNonFFTEl<R, F, A> {
        DoubleRNSNonFFTEl {
            data: self.zero().data,
            ring_decompositions: PhantomData,
            allocator: PhantomData
        }
    }

    pub fn non_fft_from(&self, x: El<<Self as RingExtension>::BaseRing>) -> DoubleRNSNonFFTEl<R, F, A> {
        let mut result = Vec::with_capacity_in(self.element_len(), self.allocator.clone());
        let x = self.base_ring().get_congruence(&x);
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            result.push(Zp.clone_el(x.at(i)));
            for _ in 1..self.rank() {
                result.push(Zp.zero());
            }
        }
        DoubleRNSNonFFTEl {
            data: result,
            ring_decompositions: PhantomData,
            allocator: PhantomData
        }
    }

    pub fn do_fft(&self, mut element: DoubleRNSNonFFTEl<R, F, A>) -> DoubleRNSEl<R, F, A> {
        assert_eq!(element.data.len(), self.element_len());
        timed!("do_fft", || {
            for i in 0..self.rns_base().len() {
                self.ring_decompositions[i].fft_forward(&mut element.data[(i * self.rank())..((i + 1) * self.rank())], self.rns_base().at(i).get_ring());
            }
        });
        DoubleRNSEl {
            data: element.data,
            ring_decompositions: PhantomData,
            allocator: PhantomData
        }
    }

    pub fn perform_rns_op_from<R2, F2, A2, Op>(
        &self, 
        from: &DoubleRNSRingBase<R2, F2, A2>, 
        el: &DoubleRNSNonFFTEl<R2, F2, A2>, 
        op: &Op
    ) -> DoubleRNSNonFFTEl<R, F, A> 
        where F: IsomorphismInfo<R::Type, R2::Type, F2>,
            // the constraings for DoubleRNSRingBase<R2, F2, A2> 
            R2: ZnRingStore<Type = R::Type>,
            R::Type: CanIsoFromTo<R2::Type> + SelfIso,
            F2: RingDecompositionSelfIso<R2::Type>,
            A2: Allocator + Clone,
            // constraints for Op
            Op: RNSOperation<RingType = R::Type>
    {
        assert!(self.ring_decompositions()[0].is_same_number_ring(&from.ring_decompositions()[0]));
        debug_assert_eq!(self.rank(), from.rank());
        assert_eq!(self.rns_base().len(), op.output_rings().len());
        assert_eq!(from.rns_base().len(), op.input_rings().len());

        for i in 0..from.rns_base().len() {
            assert!(from.rns_base().at(i).get_ring() == op.input_rings().at(i).get_ring());
        }
        for i in 0..self.rns_base().len() {
            assert!(self.rns_base().at(i).get_ring() == op.output_rings().at(i).get_ring());
        }
        timed!("perform_rns_op_from", || {
            let mut result = self.non_fft_zero();
            op.apply(from.as_matrix(el), self.as_matrix_mut(&mut result));
            return result;
        })
    }

    pub fn exact_convert_from_nttring<F2, A2>(
        &self, 
        from: &NTTRingBase<R, F2, A2>, 
        element: &<NTTRingBase<R, F2, A2> as RingBase>::Element
    ) -> DoubleRNSNonFFTEl<R, F, A> 
        where R: RingStore,
            R::Type: ZnRing + CanHomFrom<StaticRingBase<i128>>, 
            F: IsomorphismInfo<R::Type, R::Type, F2>,
            F2: RingDecompositionSelfIso<R::Type>,
            A2: Allocator + Clone
    {
        assert!(<_ as IsomorphismInfo<_, _, _>>::is_same_number_ring(&self.ring_decompositions()[0], &from.ring_decompositions()[0]));
        debug_assert_eq!(self.rank(), from.rank());

        let mut result = self.non_fft_zero();
        for j in 0..self.rank() {
            let x = int_cast(from.base_ring().smallest_lift(from.base_ring().clone_el(&element.data[j])), &StaticRing::<i32>::RING, from.base_ring().integer_ring());
            for i in 0..self.rns_base().len() {
                result.data[j + i * self.rank()] = self.rns_base().at(i).int_hom().map(x);
            }
        }
        return result;
    }

    pub fn perform_rns_op_to_nttring<F2, A2, Op>(
        &self, 
        to: &NTTRingBase<R, F2, A2>, 
        element: &DoubleRNSNonFFTEl<R, F, A>, 
        op: &Op
    ) -> <NTTRingBase<R, F2, A2> as RingBase>::Element 
        where R: RingStore,
            R::Type: ZnRing + CanHomFrom<StaticRingBase<i128>>, 
            F: IsomorphismInfo<R::Type, R::Type, F2>,
            R::Type: SelfIso,
            F2: RingDecompositionSelfIso<R::Type>,
            A2: Allocator + Clone, 
            Op: RNSOperation<RingType = R::Type>
    {
        assert!(<_ as IsomorphismInfo<_, _, _>>::is_same_number_ring(&self.ring_decompositions()[0], &to.ring_decompositions()[0]));
        debug_assert_eq!(self.rank(), to.rank());
        assert_eq!(self.rns_base().len(), op.input_rings().len());
        assert_eq!(1, op.output_rings().len());

        timed!("perform_rns_op_to_cfft", || {
            for i in 0..self.rns_base().len() {
                assert!(self.rns_base().at(i).get_ring() == op.input_rings().at(i).get_ring());
            }
            assert!(to.base_ring().get_ring() == op.output_rings().at(0).get_ring());
         
            let mut result = to.zero();
            let result_matrix = SubmatrixMut::from_1d(&mut result.data, 1, to.rank());
            op.apply(self.as_matrix(element), result_matrix);
            return result;
        })
    }

    pub fn sample_from_coefficient_distribution<G: FnMut() -> i32>(&self, mut distribution: G) -> DoubleRNSNonFFTEl<R, F, A> {
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

    pub fn sample_uniform<G: FnMut() -> u64>(&self, mut rng: G) -> <Self as RingBase>::Element {
        let mut result = self.zero();
        for j in 0..self.rank() {
            for i in 0..self.rns_base().len() {
                result.data[j + i * self.rank()] = self.rns_base().at(i).random_element(&mut rng);
            }
        }
        return result;
    }

    pub fn clone_el_non_fft(&self, val: &DoubleRNSNonFFTEl<R, F, A>) -> DoubleRNSNonFFTEl<R, F, A> {
        assert_eq!(self.element_len(), val.data.len());
        let mut result = Vec::with_capacity_in(self.element_len(), self.allocator.clone());
        result.extend((0..self.element_len()).map(|i| self.rns_base().at(i / self.rank()).clone_el(&val.data[i])));
        DoubleRNSNonFFTEl {
            data: result,
            ring_decompositions: PhantomData,
            allocator: PhantomData
        }
    }

    pub fn negate_non_fft(&self, mut val: DoubleRNSNonFFTEl<R, F, A>) -> DoubleRNSNonFFTEl<R, F, A> {
        assert_eq!(self.element_len(), val.data.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                self.rns_base().at(i).negate_inplace(&mut val.data[i * self.rank() + j]);
            }
        }
        return val;
    }

    pub fn sub_assign_non_fft(&self, lhs: &mut DoubleRNSNonFFTEl<R, F, A>, rhs: &DoubleRNSNonFFTEl<R, F, A>) {
        assert_eq!(self.element_len(), lhs.data.len());
        assert_eq!(self.element_len(), rhs.data.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                self.rns_base().at(i).sub_assign_ref(&mut lhs.data[i * self.rank() + j], &rhs.data[i * self.rank() + j]);
            }
        }
    }

    pub fn mul_scalar_assign_non_fft(&self, lhs: &mut DoubleRNSNonFFTEl<R, F, A>, rhs: &El<zn_rns::Zn<R, BigIntRing>>) {
        assert_eq!(self.element_len(), lhs.data.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                self.rns_base().at(i).mul_assign_ref(&mut lhs.data[i * self.rank() + j], self.rns_base().get_congruence(rhs).at(i));
            }
        }
    }

    pub fn add_assign_non_fft(&self, lhs: &mut DoubleRNSNonFFTEl<R, F, A>, rhs: &DoubleRNSNonFFTEl<R, F, A>) {
        assert_eq!(self.element_len(), lhs.data.len());
        assert_eq!(self.element_len(), rhs.data.len());
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                self.rns_base().at(i).add_assign_ref(&mut lhs.data[i * self.rank() + j], &rhs.data[i * self.rank() + j]);
            }
        }
    }

    pub fn wrt_canonical_basis_non_fft<'a>(&'a self, el: &'a DoubleRNSNonFFTEl<R, F, A>) -> DoubleRNSRingBaseElVectorRepresentation<'a, R, F, A> {
        DoubleRNSRingBaseElVectorRepresentation {
            ring: self,
            inv_fft_data: self.clone_el_non_fft(el)
        }
    }
}

impl<R, F, A> PartialEq for DoubleRNSRingBase<R, F, A> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    fn eq(&self, other: &Self) -> bool {
        self.scalar_ring.get_ring() == other.scalar_ring.get_ring() && self.ring_decompositions[0] == other.ring_decompositions[0]
    }
}

impl<R, F, A> RingBase for DoubleRNSRingBase<R, F, A> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    type Element = DoubleRNSEl<R, F, A>;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        assert_eq!(self.element_len(), val.data.len());
        let mut result = Vec::with_capacity_in(self.element_len(), self.allocator.clone());
        result.extend((0..self.element_len()).map(|i| self.rns_base().at(i / self.rank()).clone_el(&val.data[i])));
        DoubleRNSEl {
            data: result,
            ring_decompositions: PhantomData,
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
            ring_decompositions: PhantomData,
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

impl<R, F, A> DivisibilityRing for DoubleRNSRingBase<R, F, A> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + DivisibilityRing,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        let mut result = Vec::with_capacity_in(self.element_len(), self.allocator.clone());
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            for j in 0..self.rank() {
                result.push(Zp.checked_div(&lhs.data[i * self.rank() + j], &rhs.data[i * self.rank() + j])?);
            }
        }
        return Some(DoubleRNSEl { data: result, ring_decompositions: PhantomData, allocator: PhantomData })
    }

    fn is_unit(&self, x: &Self::Element) -> bool {
        x.data.iter().enumerate().all(|(index, c)| self.rns_base().at(index / self.rank()).is_unit(c))
    }
}

pub struct DoubleRNSRingBaseElVectorRepresentation<'a, R, F, A> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    inv_fft_data: DoubleRNSNonFFTEl<R, F, A>,
    ring: &'a DoubleRNSRingBase<R, F, A>
}

impl<'a, R, F, A> VectorFn<El<zn_rns::Zn<R, BigIntRing>>> for DoubleRNSRingBaseElVectorRepresentation<'a, R, F, A> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    fn len(&self) -> usize {
        self.ring.rank()
    }

    fn at(&self, i: usize) -> El<zn_rns::Zn<R, BigIntRing>> {
        self.ring.rns_base().from_congruence(self.inv_fft_data.data[i..].iter().step_by(self.ring.rank()).enumerate().map(|(i, x)| self.ring.rns_base().at(i).clone_el(x)))
    }
}

impl<R, F, A> CyclotomicRing for DoubleRNSRingBase<R, F, A>
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type> + CyclotomicRingDecomposition<R::Type>,
        A: Allocator + Clone
{
    fn n(&self) -> usize {
        *self.ring_decompositions()[0].galois_group_mulrepr().modulus() as usize
    }

    fn apply_galois_action(&self, el: &Self::Element, g: zn_64::ZnEl) -> Self::Element {
        let mut result = self.zero();
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            self.ring_decompositions()[i].permute_galois_action(
                &el.data[(i * self.rank())..((i + 1) * self.rank())],
                &mut result.data[(i * self.rank())..((i + 1) * self.rank())],
                g,
                Zp
            );
        }
        return result;
    }
}

impl<R, F, A> FreeAlgebra for DoubleRNSRingBase<R, F, A> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    type VectorRepresentation<'a> = DoubleRNSRingBaseElVectorRepresentation<'a, R, F, A> 
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

impl<R, F, A> RingExtension for DoubleRNSRingBase<R, F, A> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    type BaseRing = zn_rns::Zn<R, BigIntRing>;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing {
        &self.scalar_ring
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
            ring_decompositions: PhantomData,
            allocator: PhantomData
        };
    }
}

pub struct WRTCanonicalBasisElementCreator<'a, R, F, A>
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    ring: &'a DoubleRNSRingBase<R, F, A>
}

impl<'a, 'b, R, F, A> Clone for WRTCanonicalBasisElementCreator<'a, R, F, A>
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    fn clone(&self) -> Self {
        Self { ring: self.ring }
    }
}

impl<'a, 'b, R, F, A> Fn<(&'b [El<zn_rns::Zn<R, BigIntRing>>],)> for WRTCanonicalBasisElementCreator<'a, R, F, A>
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    extern "rust-call" fn call(&self, args: (&'b [El<zn_rns::Zn<R, BigIntRing>>],)) -> Self::Output {
        self.ring.from_canonical_basis(args.0.iter().map(|x| self.ring.base_ring().clone_el(x)))
    }
}

impl<'a, 'b, R, F, A> FnMut<(&'b [El<zn_rns::Zn<R, BigIntRing>>],)> for WRTCanonicalBasisElementCreator<'a, R, F, A>
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    extern "rust-call" fn call_mut(&mut self, args: (&'b [El<zn_rns::Zn<R, BigIntRing>>],)) -> Self::Output {
        self.call(args)
    }
}

impl<'a, 'b, R, F, A> FnOnce<(&'b [El<zn_rns::Zn<R, BigIntRing>>],)> for WRTCanonicalBasisElementCreator<'a, R, F, A>
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    type Output = El<DoubleRNSRing<R, F, A>>;

    extern "rust-call" fn call_once(self, args: (&'b [El<zn_rns::Zn<R, BigIntRing>>],)) -> Self::Output {
        self.call(args)
    }
}

impl<R, F, A> FiniteRing for DoubleRNSRingBase<R, F, A> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    type ElementsIter<'a> = MultiProduct<
        <zn_rns::ZnBase<R, BigIntRing> as FiniteRing>::ElementsIter<'a>, 
        WRTCanonicalBasisElementCreator<'a, R, F, A>, 
        CloneRingEl<&'a zn_rns::Zn<R, BigIntRing>>,
        El<DoubleRNSRing<R, F, A>>
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

impl<R1, R2, F1, F2, A1, A2> CanHomFrom<DoubleRNSRingBase<R2, F2, A2>> for DoubleRNSRingBase<R1, F1, A1>
    where R1: ZnRingStore,
        R1::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F1: RingDecompositionSelfIso<R1::Type> + PartialEq<F2>,
        A1: Allocator + Clone,

        R2: ZnRingStore,
        R2::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F2: RingDecompositionSelfIso<R2::Type>,
        A2: Allocator + Clone,

        R1::Type: CanHomFrom<R2::Type>,
        F1: IsomorphismInfo<R1::Type, R2::Type, F2>
{
    type Homomorphism = Vec<<R1::Type as CanHomFrom<R2::Type>>::Homomorphism>;

    fn has_canonical_hom(&self, from: &DoubleRNSRingBase<R2, F2, A2>) -> Option<Self::Homomorphism> {
        if self.rns_base().len() == from.rns_base().len() && self.ring_decompositions[0] == from.ring_decompositions[0] {
            debug_assert!(self.rank() == from.rank());
            assert!(self.ring_decompositions.iter().zip(from.ring_decompositions.iter()).all(|(l, r)| l == r));
            (0..self.rns_base().len()).map(|i| self.rns_base().at(i).get_ring().has_canonical_hom(from.rns_base().at(i).get_ring()).ok_or(())).collect::<Result<Vec<_>, ()>>().ok()
        } else {
            None
        }
    }

    fn map_in(&self, from: &DoubleRNSRingBase<R2, F2, A2>, el: <DoubleRNSRingBase<R2, F2, A2> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in_ref(from, &el, hom)
    }

    fn map_in_ref(&self, from: &DoubleRNSRingBase<R2, F2, A2>, el: &<DoubleRNSRingBase<R2, F2, A2> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        let mut result = Vec::with_capacity_in(self.element_len(), self.allocator.clone());
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            for j in 0..self.rank() {
                result.push(Zp.get_ring().map_in_ref(from.rns_base().at(i).get_ring(), &el.data[i * self.rank() + j], &hom[i]));
            }
        }
        DoubleRNSEl {
            data: result,
            ring_decompositions: PhantomData,
            allocator: PhantomData
        }
    }
}

impl<R1, R2, F1, F2, A1, A2> CanIsoFromTo<DoubleRNSRingBase<R2, F2, A2>> for DoubleRNSRingBase<R1, F1, A1>
    where R1: ZnRingStore,
        R1::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F1: RingDecompositionSelfIso<R1::Type> + PartialEq<F2>,
        A1: Allocator + Clone,

        R2: ZnRingStore,
        R2::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F2: RingDecompositionSelfIso<R2::Type>,
        A2: Allocator + Clone,

        R1::Type: CanIsoFromTo<R2::Type>,
        F1: IsomorphismInfo<R1::Type, R2::Type, F2>
{
    type Isomorphism = Vec<<R1::Type as CanIsoFromTo<R2::Type>>::Isomorphism>;

    fn has_canonical_iso(&self, from: &DoubleRNSRingBase<R2, F2, A2>) -> Option<Self::Isomorphism> {
        if self.rns_base().len() == from.rns_base().len() && self.ring_decompositions[0] == from.ring_decompositions[0] {
            debug_assert!(self.rank() == from.rank());
            assert!(self.ring_decompositions.iter().zip(from.ring_decompositions.iter()).all(|(l, r)| l == r));
            (0..self.rns_base().len()).map(|i| self.rns_base().at(i).get_ring().has_canonical_iso(from.rns_base().at(i).get_ring()).ok_or(())).collect::<Result<Vec<_>, ()>>().ok()
        } else {
            None
        }
    }

    fn map_out(&self, from: &DoubleRNSRingBase<R2, F2, A2>, el: Self::Element, iso: &Self::Isomorphism) -> <DoubleRNSRingBase<R2, F2, A2> as RingBase>::Element {
        let mut result = Vec::with_capacity_in(from.element_len(), from.allocator.clone());
        for (i, Zp) in self.rns_base().as_iter().enumerate() {
            for j in 0..self.rank() {
                result.push(Zp.get_ring().map_out(from.rns_base().at(i).get_ring(), Zp.clone_el(&el.data[i * self.rank() + j]), &iso[i]));
            }
        }
        DoubleRNSEl {
            data: result,
            ring_decompositions: PhantomData,
            allocator: PhantomData
        }
    }
}

#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use crate::rnsconv::lift::*;
#[cfg(test)]
use crate::feanor_math::rings::zn::zn_64::Zn;
#[cfg(test)]
use crate::rings::pow2_cyclotomic::DefaultPow2CyclotomicDoubleRNSRingBase;
#[cfg(test)]
use crate::rings::pow2_cyclotomic::DefaultPow2CyclotomicNTTRingBase;

//
// Most tests are done in the implementations of the corresponding ring decompositions
//

#[test]
fn test_almost_exact_convert_from() {
    let rns_base1 = zn_rns::Zn::new(vec![Zn::new(17), Zn::new(97)], BigIntRing::RING);
    let R1 = DefaultPow2CyclotomicDoubleRNSRingBase::new(rns_base1, 3);

    let rns_base2 = zn_rns::Zn::new(vec![Zn::new(17), Zn::new(113)], BigIntRing::RING);
    let R2 = DefaultPow2CyclotomicDoubleRNSRingBase::new(rns_base2, 3);

    let converter = AlmostExactBaseConversion::new_with(R1.base_ring().as_iter().cloned().collect(), R2.base_ring().as_iter().cloned().collect(), Global);

    assert_el_eq!(&R2, &R2.canonical_gen(), &R2.get_ring().do_fft(R2.get_ring().perform_rns_op_from(R1.get_ring(), &R1.get_ring().undo_fft(R1.canonical_gen()), &converter)));
    for i in (-4 * 97)..=(4 * 97) {
        assert_el_eq!(&R2, &R2.int_hom().map(i), &R2.get_ring().do_fft(R2.get_ring().perform_rns_op_from(R1.get_ring(), &R1.get_ring().undo_fft(R1.int_hom().map(i)), &converter)));
    }
}

#[test]
fn test_almost_exact_convert_to_nttring() {
    let rns_base1 = zn_rns::Zn::new(vec![Zn::new(17), Zn::new(97)], BigIntRing::RING);
    let R1 = DefaultPow2CyclotomicDoubleRNSRingBase::new(rns_base1, 3);
    let R2 = DefaultPow2CyclotomicNTTRingBase::new(Zn::new(7), 3);
    let converter = AlmostExactBaseConversion::new_with(R1.base_ring().get_ring().as_iter().cloned().collect(), vec![R2.base_ring().clone()], Global);
    assert_el_eq!(&R2, &R2.canonical_gen(), &R1.get_ring().perform_rns_op_to_nttring(R2.get_ring(), &R1.get_ring().undo_fft(R1.canonical_gen()), &converter));
    for i in (-4 * 97)..=(4 * 97) {
        assert_el_eq!(&R2, &R2.int_hom().map(i), &R1.get_ring().perform_rns_op_to_nttring(R2.get_ring(), &R1.get_ring().undo_fft(R1.int_hom().map(i)), &converter));
    }
}
