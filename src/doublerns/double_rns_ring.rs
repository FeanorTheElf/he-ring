use std::marker::PhantomData;

use feanor_math::divisibility::*;
use feanor_math::integer::*;
use feanor_math::matrix::submatrix::AsFirstElement;
use feanor_math::matrix::submatrix::Submatrix;
use feanor_math::matrix::submatrix::SubmatrixMut;
use feanor_math::rings::extension::*;
use feanor_math::mempool::*;
use feanor_math::ring::*;
use feanor_math::rings::float_complex::Complex64El;
use feanor_math::rings::poly::dense_poly::DensePolyRing;
use feanor_math::rings::zn::*;
use feanor_math::vector::*;
use feanor_math::homomorphism::*;
use feanor_math::vector::vec_fn::VectorFn;
use feanor_math::primitive_int::*;

use crate::complexfft::complex_fft_ring;
use crate::rnsconv::*;

///
/// The conversion from and to double-RNS-representation, as required for double-RNS-based rings.
/// Usually used to build a double-RNS-based ring via [`DoubleRNSRing`].
/// 
/// Concretely, we consider a ring `Z[X]/(f(X), q)` to have a double-RNS-representation, if it 
/// decomposes into a product of prime rings, which is equivalent to `f(X)` splitting completely
/// in `Zq` and `q = p1 ... pr` being square-free. In this case, the map
/// ```text
/// Z[X]/(f(X), q) -> Zq x ... x Zq,    g -> (g(x))_x where f(x) = 0
/// ```
/// is a generalization of the number-theoretic transform (NTT). Furthermore, each `Zq` decomposes
/// as `Zp1 x ... x Zpr`, and we arrive at a product of prime fields.
/// 
/// When a ring element is stored using the values in `Zq` corresponding to the right-hand side, 
/// we call this the double-RNS-representation and the values the "double-RNS coefficients".
/// This representation can be used to efficiently compute arithmetic operations. 
/// This trait now encapsulates this isomorphism and its inverse. In particular, the isomorphism is 
/// what [`GeneralizedFFT::fft_forward()`] must compute.
/// 
/// Note that this is most useful, if the map can be computed in time `o(deg(f)^2)`, which usually means
/// that fast fourier-transform techniques are used for the evaluation.
/// 
pub trait GeneralizedFFT {

    type BaseRingBase: ?Sized + ZnRing;
    type BaseRingStore: RingStore<Type = Self::BaseRingBase>;
    
    fn base_ring(&self) -> &Self::BaseRingStore;

    fn rank(&self) -> usize;

    ///
    /// Computes the map
    /// ```text
    /// Z[X]/(f(X), q) -> Zq x ... x Zq,    g -> (g(x))_x where f(x) = 0
    /// ```
    /// For a more detailed explanation, see the trait-level doc [`GeneralizedFFT`].
    /// 
    fn fft_forward<S, M>(&self, data: &mut [El<S>], ring: &S, memory_provider: &M)
        where S: ZnRingStore,
            S::Type: ZnRing + CanIsoFromTo<Self::BaseRingBase>,
            M: MemoryProvider<El<S>>;

    ///
    /// Computes the inverse of [`GeneralizedFFT::fft_forward()`].
    /// 
    fn fft_backward<S, M>(&self, data: &mut [El<S>], ring: &S, memory_provider: &M)
        where S: ZnRingStore,
            S::Type: ZnRing + CanIsoFromTo<Self::BaseRingBase>,
            M: MemoryProvider<El<S>>;
}

///
/// Used as a marker that indicates whether the ring structure induced by two [`GeneralizedFFT`]s
/// is the same. Note that this should not consider the modulus, so really considers the number rings
/// `Z[X]/(f(X))` without the reduction modulo `q`.
/// 
/// The latter is important, since it is common in HE to switch the modulus, e.g. by "rescaling".
/// 
/// Note that whenever `a.is_isomorphic(b)` is true, it is necessary that also `a.rank() == b.rank()`.
/// 
pub trait GeneralizedFFTIso<F: GeneralizedFFT>: GeneralizedFFT {

    fn is_isomorphic(&self, other: &F) -> bool;
}

///
/// Used as a marker that indicates whether the ring structure induced by two `GeneralizedFFT`s
/// is the same. Note that this should not consider the modulus, so really considers the number rings
/// `Z[X]/(f(X))` without the reduction modulo `q`.
/// 
/// As opposed to [`GeneralizedFFTIso`], this refers to the relationship between a [`crate::doublerns::double_rns_ring::GeneralizedFFT`]
/// and a [`crate::complexfft::complex_fft_ring::GeneralizedFFT`]. Since (apart from the underlying implementation), the 
/// only formal differences are the modulus, this notion still makes sense.
/// 
/// See also [`GeneralizedFFTIso`].
/// 
pub trait GeneralizedFFTCrossIso<F: complex_fft_ring::GeneralizedFFT>: GeneralizedFFT {

    fn is_isomorphic(&self, other: &F) -> bool;
}

pub trait GeneralizedFFTSelfIso: Sized + GeneralizedFFTIso<Self> {}

impl<F: GeneralizedFFT + GeneralizedFFTIso<F>> GeneralizedFFTSelfIso for F {}

///
/// The ring specified by a [`GeneralizedFFT`]. Elements are stored in double-RNS-representation
/// for efficient arithmetic.
/// 
/// When necessary, it is also possible by using [`DoubleRNSRingBase::do_fft()`] and
/// [`DoubleRNSRingBase::undo_fft()`] to work with ring elements not in double-RNS-representation,
/// but note that arithmetic operations are not available for those.
/// 
pub struct DoubleRNSRingBase<R, F, M> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanIsoFromTo<F::BaseRingBase>,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    data: Vec<F>,
    scalar_ring: zn_rns::Zn<R, BigIntRing>,
    memory_provider: M
}

pub type DoubleRNSRing<R, F, M> = RingValue<DoubleRNSRingBase<R, F, M>>;

pub struct DoubleRNSEl<R, F, M> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanIsoFromTo<F::BaseRingBase> + CanHomFrom<BigIntRingBase>,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    generalized_fft: PhantomData<F>,
    memory_provider: PhantomData<M>,
    data: M::Object
}

pub struct DoubleRNSNonFFTEl<R, F, M> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanIsoFromTo<F::BaseRingBase> + CanHomFrom<BigIntRingBase>,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    generalized_fft: PhantomData<F>,
    memory_provider: PhantomData<M>,
    data: M::Object
}

impl<R, F, M> DoubleRNSRingBase<R, F, M> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanIsoFromTo<F::BaseRingBase> + CanHomFrom<BigIntRingBase>,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    pub fn from_generalized_ffts(rns_base: zn_rns::Zn<R, BigIntRing>, data: Vec<F>, memory_provider: M) -> Self {
        assert!(data.len() > 0);
        for i in 0..data.len() {
            assert!(data[i].is_isomorphic(&data[0]));
            assert_eq!(data[i].rank(), data[0].rank());
        }
        let scalar_ring = rns_base;
        Self { data, memory_provider, scalar_ring }
    }
}

impl<R, F, M> DoubleRNSRingBase<R, F, M> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanIsoFromTo<F::BaseRingBase> + CanHomFrom<BigIntRingBase>,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    pub fn generalized_fft(&self) -> &Vec<F> {
        &self.data
    }

    pub fn rns_base(&self) -> &zn_rns::ZnBase<R, BigIntRing> {
        self.scalar_ring.get_ring()
    }

    pub fn element_len(&self) -> usize {
        self.rank() * self.rns_base().len()
    }

    fn as_matrix<'a>(&self, element: &'a DoubleRNSNonFFTEl<R, F, M>) -> Submatrix<'a, AsFirstElement<El<R>>, El<R>> {
        Submatrix::<AsFirstElement<_>, _>::new(&element.data, self.rns_base().len(), self.rank())
    }

    fn as_matrix_mut<'a>(&self, element: &'a mut DoubleRNSNonFFTEl<R, F, M>) -> SubmatrixMut<'a, AsFirstElement<El<R>>, El<R>> {
        SubmatrixMut::<AsFirstElement<_>, _>::new(&mut element.data, self.rns_base().len(), self.rank())
    }

    ///
    /// Returns `a mod pi` where `a` is the coefficient belonging to `X^j` of the given element.
    /// 
    /// Here `pi` is the `i`-th prime divisor of the base ring (using the order exposed by
    /// [`zn_rns::ZnBase`]).
    /// 
    pub fn at<'a>(&self, i: usize, j: usize, el: &'a DoubleRNSNonFFTEl<R, F, M>) -> &'a El<R> {
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
    /// of the underlying [`GeneralizedFFT`] (in particular, if it is implemented using standard FFTs, then
    /// that in turn depends on the ordering used by [`feanor_math::algorithms::fft::FFTTable::unordered_fft()`]).
    /// Therefore, you should not rely on any specific relationship between `j` and `k`, except that it will
    /// remain constant during the lifetime of the ring. Note also that changing the order corresponds to an
    /// automorphism of the ring.
    /// 
    pub fn fourier_coefficient<'a>(&self, i: usize, j: usize, el: &'a DoubleRNSEl<R, F, M>) -> &'a El<R> {
        &el.data[i * self.rank() + j]
    }

    /// 
    /// Returns `a mod pi` where `a` is the coefficient belonging to `X^j` of the given element.
    /// 
    /// See [`Self::at()`] for details.
    /// 
    pub fn at_mut<'a>(&self, i: usize, j: usize, el: &'a mut DoubleRNSNonFFTEl<R, F, M>) -> &'a mut El<R> {
        &mut el.data[i * self.rank() + j]
    }

    ///
    /// Returns the `(i, j)`-th component of the element in double-RNS-representation. 
    /// 
    /// See [`Self::fourier_coefficient()`] for details.
    /// 
    pub fn fourier_coefficient_mut<'a>(&self, i: usize, j: usize, el: &'a mut DoubleRNSEl<R, F, M>) -> &'a mut El<R> {
        &mut el.data[i * self.rank() + j]
    }

    pub fn undo_fft(&self, mut element: DoubleRNSEl<R, F, M>) -> DoubleRNSNonFFTEl<R, F, M> {
        assert_eq!(element.data.len(), self.element_len());
        timed!("undo_fft", || {
            for i in 0..self.rns_base().len() {
                self.data[i].fft_backward(&mut element.data[(i * self.rank())..((i + 1) * self.rank())], self.rns_base().at(i), &self.memory_provider);
            }
        });
        DoubleRNSNonFFTEl {
            data: element.data,
            generalized_fft: PhantomData,
            memory_provider: PhantomData
        }
    }

    pub fn memory_provider(&self) -> &M {
        &self.memory_provider
    }

    pub fn non_fft_zero(&self) -> DoubleRNSNonFFTEl<R, F, M> {
        DoubleRNSNonFFTEl {
            data: self.zero().data,
            generalized_fft: PhantomData,
            memory_provider: PhantomData
        }
    }

    pub fn do_fft(&self, mut element: DoubleRNSNonFFTEl<R, F, M>) -> DoubleRNSEl<R, F, M> {
        assert_eq!(element.data.len(), self.element_len());
        timed!("do_fft", || {
            for i in 0..self.rns_base().len() {
                self.data[i].fft_forward(&mut element.data[(i * self.rank())..((i + 1) * self.rank())], self.rns_base().at(i), &self.memory_provider);
            }
        });
        DoubleRNSEl {
            data: element.data,
            generalized_fft: PhantomData,
            memory_provider: PhantomData
        }
    }

    pub fn perform_rns_op_from<R2, F2, M2, Op>(
        &self, 
        from: &DoubleRNSRingBase<R2, F2, M2>, 
        el: &DoubleRNSNonFFTEl<R2, F2, M2>, 
        op: &Op
    ) -> DoubleRNSNonFFTEl<R, F, M> 
        where F: GeneralizedFFTIso<F2>,
            // the constraings for DoubleRNSRingBase<R2, F2, M2> 
            R2: ZnRingStore<Type = R::Type>,
            R::Type: CanIsoFromTo<F2::BaseRingBase>,
            F2: GeneralizedFFT + GeneralizedFFTSelfIso,
            M2: MemoryProvider<El<R2>>,
            // constraints for Op
            Op: RNSOperation<RingType = R::Type>
    {
        assert!(self.generalized_fft()[0].is_isomorphic(&from.generalized_fft()[0]));
        debug_assert_eq!(self.rank(), from.rank());
        assert_eq!(self.rns_base().len(), op.output_rings().len());
        assert_eq!(from.rns_base().len(), op.input_rings().len());

        for i in 0..from.rns_base().len() {
            assert!(from.rns_base().at(i).get_ring() == op.input_rings().at(i).get_ring());
        }
        for i in 0..self.rns_base().len() {
            assert!(self.rns_base().at(i).get_ring() == op.output_rings().at(i).get_ring());
        }
        let mut result = self.non_fft_zero();
        op.apply(from.as_matrix(el), self.as_matrix_mut(&mut result));
        return result;
    }

    pub fn exact_convert_from_cfft<F2, M2_Zn, M2_CC>(
        &self, 
        from: &complex_fft_ring::ComplexFFTBasedRingBase<F2, M2_Zn, M2_CC>, 
        element: &<complex_fft_ring::ComplexFFTBasedRingBase<F2, M2_Zn, M2_CC> as RingBase>::Element
    ) -> DoubleRNSNonFFTEl<R, F, M> 
        where F: GeneralizedFFTCrossIso<F2>,
            // the constraings for DoubleRNSRingBase<F2, M2>
            F2: complex_fft_ring::GeneralizedFFT + complex_fft_ring::GeneralizedFFTSelfIso,
            F2::BaseRingBase: CanHomFrom<BigIntRingBase>,
            M2_Zn: MemoryProvider<El<F2::BaseRingStore>>,
            M2_CC: MemoryProvider<Complex64El>
    {
        assert!(<_ as GeneralizedFFTCrossIso<_>>::is_isomorphic(&self.generalized_fft()[0], &from.generalized_fft()));
        debug_assert_eq!(self.rank(), from.rank());

        let mut result = self.memory_provider.get_new_init(self.element_len(), |i| self.rns_base().at(i / self.rank()).zero());
        for j in 0..self.rank() {
            let x = int_cast(from.base_ring().smallest_lift(from.base_ring().clone_el(&element[j])), &StaticRing::<i32>::RING, from.base_ring().integer_ring());
            for i in 0..self.rns_base().len() {
                result[j + i * self.rank()] = self.rns_base().at(i).int_hom().map(x);
            }
        }
        return DoubleRNSNonFFTEl {
            data: result,
            generalized_fft: PhantomData,
            memory_provider: PhantomData
        };
    }

    pub fn perform_rns_op_to_cfft<F2, M2_Zn, M2_CC, Op>(
        &self, 
        to: &complex_fft_ring::ComplexFFTBasedRingBase<F2, M2_Zn, M2_CC>, 
        element: &DoubleRNSNonFFTEl<R, F, M>, 
        op: &Op
    ) -> <complex_fft_ring::ComplexFFTBasedRingBase<F2, M2_Zn, M2_CC> as RingBase>::Element 
        where F: GeneralizedFFTCrossIso<F2>,
            // the constraings for DoubleRNSRingBase<F2, M2>
            F2: complex_fft_ring::GeneralizedFFT<BaseRingBase = R::Type> + complex_fft_ring::GeneralizedFFTSelfIso,
            M2_Zn: MemoryProvider<El<F2::BaseRingStore>>,
            M2_CC: MemoryProvider<Complex64El>,
            Op: RNSOperation<RingType = R::Type>
    {
        assert!(<_ as GeneralizedFFTCrossIso<_>>::is_isomorphic(&self.generalized_fft()[0], &to.generalized_fft()));
        debug_assert_eq!(self.rank(), to.rank());
        assert_eq!(self.rns_base().len(), op.input_rings().len());
        assert_eq!(1, op.output_rings().len());

        for i in 0..self.rns_base().len() {
            assert!(self.rns_base().at(i).get_ring() == op.input_rings().at(i).get_ring());
        }
        assert!(to.base_ring().get_ring() == op.output_rings().at(0).get_ring());
        
        let mut result = to.zero();
        let result_matrix = SubmatrixMut::<AsFirstElement<_>, _>::new(&mut result, 1, to.rank());
        op.apply(self.as_matrix(element), result_matrix);
        return result;
    }

    pub fn sample_from_coefficient_distribution<G: FnMut() -> i32>(&self, mut distribution: G) -> <Self as RingBase>::Element {
        let mut result = self.memory_provider.get_new_init(self.element_len(), |i| self.rns_base().at(i / self.rank()).zero());
        let mut data = Vec::new();
        for j in 0..self.rank() {
            let c = distribution();
            data.push(c);
            for i in 0..self.rns_base().len() {
                result[j + i * self.rank()] = self.rns_base().at(i).int_hom().map(c);
            }
        }
        return self.do_fft(DoubleRNSNonFFTEl {
            data: result,
            generalized_fft: PhantomData,
            memory_provider: PhantomData
        });
    }

    pub fn sample_uniform<G: FnMut() -> u64>(&self, mut rng: G) -> <Self as RingBase>::Element {
        let mut result = self.memory_provider.get_new_init(self.element_len(), |i| self.rns_base().at(i / self.rank()).zero());
        for j in 0..self.rank() {
            for i in 0..self.rns_base().len() {
                result[j + i * self.rank()] = self.rns_base().at(i).random_element(&mut rng);
            }
        }
        return DoubleRNSEl {
            data: result,
            generalized_fft: PhantomData,
            memory_provider: PhantomData
        };
    }
}

impl<R, F, M> PartialEq for DoubleRNSRingBase<R, F, M> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanIsoFromTo<F::BaseRingBase> + CanHomFrom<BigIntRingBase>,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    fn eq(&self, other: &Self) -> bool {
        self.scalar_ring.get_ring() == other.scalar_ring.get_ring() && self.data[0].is_isomorphic(&other.data[0])
    }
}

impl<R, F, M> RingBase for DoubleRNSRingBase<R, F, M> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanIsoFromTo<F::BaseRingBase> + CanHomFrom<BigIntRingBase>,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    type Element = DoubleRNSEl<R, F, M>;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        assert_eq!(self.element_len(), val.data.len());
        DoubleRNSEl {
            data: self.memory_provider.get_new_init(self.element_len(), |i| self.rns_base().at(i / self.rank()).clone_el(&val.data[i])),
            generalized_fft: PhantomData,
            memory_provider: PhantomData
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

impl<R, F, M> DivisibilityRing for DoubleRNSRingBase<R, F, M> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanIsoFromTo<F::BaseRingBase> + CanHomFrom<BigIntRingBase> + DivisibilityRing,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        self.memory_provider.try_get_new_init(self.element_len(), |index| {
            let i = index / self.rank();
            if let Some(quo) = self.rns_base().at(i).checked_div(&lhs.data[index], &rhs.data[index]) {
                return Ok(quo);
            } else {
                return Err(());
            }
        }).ok().map(|data| DoubleRNSEl { data: data, generalized_fft: PhantomData, memory_provider: PhantomData })
    }

    fn is_unit(&self, x: &Self::Element) -> bool {
        x.data.iter().enumerate().all(|(index, c)| self.rns_base().at(index / self.rank()).is_unit(c))
    }
}

pub struct DoubleRNSRingBaseElVectorRepresentation<'a, R, F, M> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanIsoFromTo<F::BaseRingBase> + CanHomFrom<BigIntRingBase>,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    inv_fft_data: DoubleRNSNonFFTEl<R, F, M>,
    ring: &'a DoubleRNSRingBase<R, F, M>
}

impl<'a, R, F, M> VectorFn<El<zn_rns::Zn<R, BigIntRing>>> for DoubleRNSRingBaseElVectorRepresentation<'a, R, F, M> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanIsoFromTo<F::BaseRingBase> + CanHomFrom<BigIntRingBase>,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    fn len(&self) -> usize {
        self.ring.rank()
    }

    fn at(&self, i: usize) -> El<zn_rns::Zn<R, BigIntRing>> {
        self.ring.rns_base().from_congruence(self.inv_fft_data.data[i..].iter().step_by(self.ring.rank()).enumerate().map(|(i, x)| self.ring.rns_base().at(i).clone_el(x)))
    }
}

impl<R, F, M> FreeAlgebra for DoubleRNSRingBase<R, F, M> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanIsoFromTo<F::BaseRingBase> + CanHomFrom<BigIntRingBase>,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    type VectorRepresentation<'a> = DoubleRNSRingBaseElVectorRepresentation<'a, R, F, M> 
        where Self: 'a;

    fn canonical_gen(&self) -> Self::Element {
        let result = self.memory_provider.get_new_init(self.element_len(), |index| {
            let i = index / self.rank();
            let j = index % self.rank();
            if j == 1 {
                self.rns_base().at(i).one()
            } else {
                self.rns_base().at(i).zero()
            }
        });
        return self.do_fft(DoubleRNSNonFFTEl {
            data: result,
            generalized_fft: PhantomData,
            memory_provider: PhantomData
        });
    }

    fn rank(&self) -> usize {
        self.data[0].rank()
    }

    fn wrt_canonical_basis<'a>(&'a self, el: &'a Self::Element) -> Self::VectorRepresentation<'a> {
        DoubleRNSRingBaseElVectorRepresentation {
            ring: self,
            inv_fft_data: self.undo_fft(self.clone_el(el))
        }
    }

    fn from_canonical_basis<V>(&self, vec: V) -> Self::Element
        where V: ExactSizeIterator + DoubleEndedIterator + Iterator<Item = El<Self::BaseRing>>
    {
        let mut result = self.memory_provider.get_new_init(self.element_len(), |index| self.rns_base().at(index / self.rank()).zero());
        for (j, x) in vec.enumerate() {
            let congruence = self.base_ring().get_ring().get_congruence(&x);
            for i in 0..self.rns_base().len() {
                result[i * self.rank() + j] = self.rns_base().at(i).clone_el(congruence.at(i));
            }
        }
        return self.do_fft(DoubleRNSNonFFTEl {
            data: result,
            generalized_fft: PhantomData,
            memory_provider: PhantomData
        });
    }
}

impl<R, F, M> RingExtension for DoubleRNSRingBase<R, F, M> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanIsoFromTo<F::BaseRingBase> + CanHomFrom<BigIntRingBase>,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    type BaseRing = zn_rns::Zn<R, BigIntRing>;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing {
        &self.scalar_ring
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        self.from_ref(&x)
    }

    fn from_ref(&self, x: &El<Self::BaseRing>) -> Self::Element {
        let x_data = self.rns_base().get_congruence(x);
        let result = self.memory_provider.get_new_init(self.element_len(), |index| {
            let i = index / self.rank();
            let j = index % self.rank();
            if j == 0 {
                self.rns_base().at(i).clone_el(x_data.at(i))
            } else {
                self.rns_base().at(i).zero()
            }
        });
        return self.do_fft(DoubleRNSNonFFTEl {
            data: result,
            generalized_fft: PhantomData,
            memory_provider: PhantomData
        });
    }
}

impl<R1, R2, F1, F2, M1, M2> CanHomFrom<DoubleRNSRingBase<R2, F2, M2>> for DoubleRNSRingBase<R1, F1, M1>
    where R1: ZnRingStore,
        R1::Type: ZnRing + CanIsoFromTo<F1::BaseRingBase> + CanHomFrom<BigIntRingBase>,
        F1: GeneralizedFFT + GeneralizedFFTSelfIso,
        M1: MemoryProvider<El<R1>>,

        R2: ZnRingStore,
        R2::Type: ZnRing + CanIsoFromTo<F2::BaseRingBase> + CanHomFrom<BigIntRingBase>,
        F2: GeneralizedFFT + GeneralizedFFTSelfIso,
        M2: MemoryProvider<El<R2>>,

        R1::Type: CanHomFrom<R2::Type>,
        F1: GeneralizedFFTIso<F2>
{
    type Homomorphism = Vec<<R1::Type as CanHomFrom<R2::Type>>::Homomorphism>;

    fn has_canonical_hom(&self, from: &DoubleRNSRingBase<R2, F2, M2>) -> Option<Self::Homomorphism> {
        if self.rns_base().len() == from.rns_base().len() && self.data[0].is_isomorphic(&from.data[0]) {
            debug_assert!(self.rank() == from.rank());
            debug_assert!(self.data.iter().zip(from.data.iter()).all(|(l, r)| l.is_isomorphic(r)));
            (0..self.rns_base().len()).map(|i| self.rns_base().at(i).get_ring().has_canonical_hom(from.rns_base().at(i).get_ring()).ok_or(())).collect::<Result<Vec<_>, ()>>().ok()
        } else {
            None
        }
    }

    fn map_in(&self, from: &DoubleRNSRingBase<R2, F2, M2>, el: <DoubleRNSRingBase<R2, F2, M2> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in_ref(from, &el, hom)
    }

    fn map_in_ref(&self, from: &DoubleRNSRingBase<R2, F2, M2>, el: &<DoubleRNSRingBase<R2, F2, M2> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        DoubleRNSEl {
            data: self.memory_provider.get_new_init(self.element_len(), |index| {
                let i = index / self.rank();
                self.rns_base().at(i).get_ring().map_in_ref(from.rns_base().at(i).get_ring(), &el.data[index], &hom[i])
            }),
            generalized_fft: PhantomData,
            memory_provider: PhantomData
        }
    }
}

impl<R1, R2, F1, F2, M1, M2> CanIsoFromTo<DoubleRNSRingBase<R2, F2, M2>> for DoubleRNSRingBase<R1, F1, M1>
    where R1: ZnRingStore,
        R1::Type: ZnRing + CanIsoFromTo<F1::BaseRingBase> + CanHomFrom<BigIntRingBase>,
        F1: GeneralizedFFT + GeneralizedFFTSelfIso,
        M1: MemoryProvider<El<R1>>,

        R2: ZnRingStore,
        R2::Type: ZnRing + CanIsoFromTo<F2::BaseRingBase> + CanHomFrom<BigIntRingBase>,
        F2: GeneralizedFFT + GeneralizedFFTSelfIso,
        M2: MemoryProvider<El<R2>>,

        R1::Type: CanIsoFromTo<R2::Type>,
        F1: GeneralizedFFTIso<F2>
{
    type Isomorphism = Vec<<R1::Type as CanIsoFromTo<R2::Type>>::Isomorphism>;

    fn has_canonical_iso(&self, from: &DoubleRNSRingBase<R2, F2, M2>) -> Option<Self::Isomorphism> {
        if self.rns_base().len() == from.rns_base().len() && self.data[0].is_isomorphic(&from.data[0]) {
            debug_assert!(self.rank() == from.rank());
            debug_assert!(self.data.iter().zip(from.data.iter()).all(|(l, r)| l.is_isomorphic(r)));
            (0..self.rns_base().len()).map(|i| self.rns_base().at(i).get_ring().has_canonical_iso(from.rns_base().at(i).get_ring()).ok_or(())).collect::<Result<Vec<_>, ()>>().ok()
        } else {
            None
        }
    }

    fn map_out(&self, from: &DoubleRNSRingBase<R2, F2, M2>, el: Self::Element, iso: &Self::Isomorphism) -> <DoubleRNSRingBase<R2, F2, M2> as RingBase>::Element {
        DoubleRNSEl {
            data: from.memory_provider.get_new_init(self.element_len(), |index| {
                let i = index / self.rank();
                self.rns_base().at(i).get_ring().map_out(from.rns_base().at(i).get_ring(), self.rns_base().at(i).clone_el(&el.data[index]), &iso[i])
            }),
            generalized_fft: PhantomData,
            memory_provider: PhantomData
        }
    }
}

#[cfg(test)]
use feanor_math::{assert_el_eq, default_memory_provider};
#[cfg(test)]
use crate::complexfft;
#[cfg(test)]
use crate::rnsconv::lift::*;
#[cfg(test)]
use crate::doublerns::pow2_cyclotomic::Pow2CyclotomicFFT;
#[cfg(test)]
use crate::complexfft::complex_fft_ring::ComplexFFTBasedRingBase;
#[cfg(test)]
use crate::feanor_math::rings::zn::zn_64::Zn;

#[test]
fn test_almost_exact_convert_from() {
    let rns_base1 = zn_rns::Zn::new(vec![Zn::new(17), Zn::new(97)], BigIntRing::RING, default_memory_provider!());
    let fft_rings1 = rns_base1.get_ring().iter().cloned().collect();
    let R1 = DoubleRNSRingBase::<_, Pow2CyclotomicFFT<_>, _>::new(rns_base1, fft_rings1, 3, default_memory_provider!());

    let rns_base2 = zn_rns::Zn::new(vec![Zn::new(17), Zn::new(113)], BigIntRing::RING, default_memory_provider!());
    let fft_rings2 = rns_base2.get_ring().iter().cloned().collect();
    let R2 = DoubleRNSRingBase::<_, Pow2CyclotomicFFT<_>, _>::new(rns_base2, fft_rings2, 3, default_memory_provider!());

    let converter = AlmostExactBaseConversion::new(R1.base_ring().get_ring(), R2.base_ring().get_ring(), Zn::new(7), default_memory_provider!(), default_memory_provider!());

    assert_el_eq!(&R2, &R2.canonical_gen(), &R2.get_ring().do_fft(R2.get_ring().perform_rns_op_from(R1.get_ring(), &R1.get_ring().undo_fft(R1.canonical_gen()), &converter)));
    for i in (-4 * 97)..=(4 * 97) {
        assert_el_eq!(&R2, &R2.int_hom().map(i), &R2.get_ring().do_fft(R2.get_ring().perform_rns_op_from(R1.get_ring(), &R1.get_ring().undo_fft(R1.int_hom().map(i)), &converter)));
    }
}

#[test]
fn test_almost_exact_convert_to_cfft() {
    let rns_base1 = zn_rns::Zn::new(vec![Zn::new(17), Zn::new(97)], BigIntRing::RING, default_memory_provider!());
    let fft_rings1 = rns_base1.get_ring().iter().cloned().collect();
    let R1 = DoubleRNSRingBase::<_, Pow2CyclotomicFFT<_>, _>::new(rns_base1, fft_rings1, 3, default_memory_provider!());

    let R2 = ComplexFFTBasedRingBase::<complexfft::pow2_cyclotomic::Pow2CyclotomicFFT<_, _>, _, _>::new(Zn::new(7), 3, default_memory_provider!(), default_memory_provider!());

    let converter = AlmostExactBaseConversion::new(R1.base_ring().get_ring(), [R2.base_ring().clone()], Zn::new(11), default_memory_provider!(), default_memory_provider!());

    assert_el_eq!(&R2, &R2.canonical_gen(), &R1.get_ring().perform_rns_op_to_cfft(R2.get_ring(), &R1.get_ring().undo_fft(R1.canonical_gen()), &converter));
    for i in (-4 * 97)..=(4 * 97) {
        assert_el_eq!(&R2, &R2.int_hom().map(i), &R1.get_ring().perform_rns_op_to_cfft(R2.get_ring(), &R1.get_ring().undo_fft(R1.int_hom().map(i)), &converter));
    }
}