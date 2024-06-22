use std::alloc::Allocator;
use std::alloc::Global;

use feanor_math::algorithms;
use feanor_math::integer::*;
use feanor_math::iters::multi_cartesian_product;
use feanor_math::iters::MultiProduct;
use feanor_math::rings::extension::*;
use feanor_math::ring::*;
use feanor_math::rings::finite::*;
use feanor_math::rings::float_complex::*;
use feanor_math::rings::poly::PolyRing;
use feanor_math::rings::poly::dense_poly::DensePolyRing;
use feanor_math::rings::zn::*;
use feanor_math::homomorphism::*;
use feanor_math::seq::*;

const CC: Complex64 = Complex64::RING;

///
/// The data necessary to construct a complex-FFT based ring.
/// 
/// Mathematically speaking, this corresponds to a number field `K = Q[X]/(f)` with
/// ring of integers `O_K` and a reduction `O_K -> O_K/(p) = Fp[X]/(f)`. Arithmetic in this ring
/// is then provided by giving "generalized FFTs" that compute the evaluation of a polynomial
/// `g` at all complex roots of `f` (where we assume that `f` is separable).
/// 
/// More concretely, `fft_forward` should compute the map
/// ```text
///     Fp[X]/(f) -> C^deg(f),  g -> (lift(g)(x))_x where f(x) = 0
/// ```
/// and `fft_backward` should naturally compute the inverse map.
/// 
/// In practice, this is only sensible if `fft_forward` and `fft_backward` can be computed in 
/// time `o(deg(f)^2)`, which would usually be the case if some FFT-techniques can be applied.
/// 
pub trait GeneralizedFFT<R: ?Sized + RingBase> {

    fn rank(&self) -> usize;

    ///
    /// Computes the map
    /// ```text
    ///     Fp[X]/(f) -> C^deg(f),  g -> (lift(g)(x))_x where f(x) = 0
    /// ```
    /// For more detail, see the trait-level doc of [`GeneralizedFFT`].
    /// 
    fn fft_forward(&self, input: &[R::Element], destination: &mut [Complex64El], ring: &R);

    ///
    /// Computes the inverse of [`fft_forward()`].
    /// 
    /// As opposed to [`fft_forward()`], this function is allowed to arbitrarily change the value
    /// stored in `input`, should that enable a more efficient implementation.
    /// 
    fn fft_backward(&self, input: &mut [Complex64El], destination: &mut [R::Element], ring: &R);

    ///
    /// Computes the product of `lhs` and `rhs`, and stores it in `lhs`. Note that `lhs` here is
    /// given in coefficient-representation and `rhs` is given in (complex-valued) FFT representation.
    /// 
    fn mul_assign_fft(&self, lhs: &mut [R::Element], rhs: &[Complex64El], ring: &R) {
        let mut tmp = (0..self.rank()).map(|_| CC.zero()).collect::<Vec<_>>();
        self.fft_forward(&lhs, &mut tmp, ring);
        for i in 0..self.rank() {
            CC.mul_assign_ref(&mut tmp[i], &rhs[i]);
        }
        self.fft_backward(&mut tmp, lhs, ring);
    }
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
pub trait GeneralizedFFTIso<R1: ?Sized + RingBase, R2: ?Sized + RingBase, F: ?Sized + GeneralizedFFT<R2>>: GeneralizedFFT<R1> {

    fn is_isomorphic(&self, other: &F) -> bool;
}

pub trait GeneralizedFFTSelfIso<R: ?Sized + RingBase>: GeneralizedFFTIso<R, R, Self> {}

impl<R: ?Sized + RingBase, F: ?Sized + GeneralizedFFT<R> + GeneralizedFFTIso<R, R, F>> GeneralizedFFTSelfIso<R> for F {}

///
/// Implementation of rings `Z[X]/(f(X), q)` that uses a [`GeneralizedFFT`] for fast arithmetic.
/// 
/// The concrete ring (i.e. the polynomial `f`) is determined by the used [`GeneralizedFFT`].
/// In the most general case, you can use [`ComplexFFTBasedRingBase::from_generalized_fft()`] to 
/// create an instance for any `f`, but in the case of cyclotomics (e.g. when using [`crate::complexfft::pow2_cyclotomic::Pow2CyclotomicFFT`]),
/// simpler creation functions are provided.
/// 
/// # Example
/// 
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// # use feanor_math::primitive_int::StaticRing;
/// # use feanor_math::integer::*;
/// # use feanor_math::mempool::DefaultMemoryProvider;
/// # use feanor_math::algorithms::fft::*;
/// # use feanor_math::{default_memory_provider, assert_el_eq};
/// # use feanor_math::homomorphism::Homomorphism;
/// # use feanor_math::rings::float_complex::Complex64;
/// # use feanor_math::rings::extension::FreeAlgebra;
/// # use feanor_math::rings::extension::FreeAlgebraStore;
/// # use he_ring::complexfft::complex_fft_ring::*;
/// # use he_ring::cyclotomic::*;
/// # use he_ring::complexfft::pow2_cyclotomic::Pow2CyclotomicFFT;
/// type TheRing = ComplexFFTBasedRing<Pow2CyclotomicFFT<Zn, cooley_tuckey::FFTTableCooleyTuckey<Complex64>>>;
/// 
/// // the ring `F7[X]/(X^8 + 1)`
/// let R = <TheRing as RingStore>::Type::new(Zn::new(7), 3);
/// let root_of_unity = R.canonical_gen();
/// assert_eq!(8, R.rank());
/// assert_eq!(16, R.n());
/// assert_el_eq!(&R, &R.neg_one(), &R.pow(root_of_unity, 8));
/// 
/// // instead of this, we can also explicity create the `GeneralizedFFT`
/// let generalized_fft: Pow2CyclotomicFFT<Zn, cooley_tuckey::FFTTableCooleyTuckey<Complex64>> = Pow2CyclotomicFFT::create(
///     Zn::new(7),
///     cooley_tuckey::FFTTableCooleyTuckey::for_complex(Complex64::RING, 3)
/// );
/// let R = RingValue::from(<TheRing as RingStore>::Type::from_generalized_fft(generalized_fft));
/// assert_eq!(8, R.rank());
/// assert_eq!(16, R.n());
/// assert_el_eq!(&R, &R.neg_one(), &R.pow(R.canonical_gen(), 8));
/// ```
/// 
pub struct ComplexFFTBasedRingBase<R, F, A = Global>
    where R: RingStore,
        R::Type: ZnRing,
        F: GeneralizedFFTSelfIso<R::Type>,
        A: Allocator + Clone
{
    ring: R,
    generalized_fft: F,
    allocator: A
}

///
/// The [`RingStore`] corresponding to [`ComplexFFTBasedRingBase`].
/// 
pub type ComplexFFTBasedRing<R, F, A = Global> = RingValue<ComplexFFTBasedRingBase<R, F, A>>;

impl<R, F, A> ComplexFFTBasedRingBase<R, F, A>
    where R: RingStore,
        R::Type: ZnRing,
        F: GeneralizedFFTSelfIso<R::Type>,
        A: Allocator + Clone
{
    pub fn from_generalized_fft(ring: R, generalized_fft: F, allocator: A) -> Self {
        Self { ring, generalized_fft, allocator }
    }

    pub fn generalized_fft(&self) -> &F {
        &self.generalized_fft
    }

    pub fn allocator(&self) -> &A {
        &self.allocator
    }
}

impl<R, F, A> PartialEq for ComplexFFTBasedRingBase<R, F, A>
    where R: RingStore,
        R::Type: ZnRing,
        F: GeneralizedFFTSelfIso<R::Type>,
        A: Allocator + Clone
{
    fn eq(&self, other: &Self) -> bool {
        self.ring.get_ring() == other.ring.get_ring() && self.generalized_fft.is_isomorphic(&other.generalized_fft)
    }
}

impl<R, F, A> RingBase for ComplexFFTBasedRingBase<R, F, A>
    where R: RingStore,
        R::Type: ZnRing,
        F: GeneralizedFFTSelfIso<R::Type>,
        A: Allocator + Clone
{
    type Element = Vec<El<R>, A>;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        assert_eq!(self.rank(), val.len());
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        result.extend((0..self.rank()).map(|i| self.base_ring().clone_el(&val[i])));
        return result;
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        assert_eq!(self.generalized_fft.rank(), lhs.len());
        assert_eq!(self.generalized_fft.rank(), rhs.len());
        for i in 0..self.generalized_fft.rank() {
            self.base_ring().add_assign_ref(&mut lhs[i], &rhs[i]);
        }
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.add_assign_ref(lhs, &rhs);
    }

    fn sub_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        self.negate_inplace(lhs);
        self.add_assign_ref(lhs, rhs);
        self.negate_inplace(lhs);
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        assert_eq!(self.generalized_fft.rank(), lhs.len());
        for i in 0..self.generalized_fft.rank() {
            self.base_ring().negate_inplace(&mut lhs[i]);
        }
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.mul_assign_ref(lhs, &rhs);
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        assert_eq!(self.rank(), lhs.len());
        assert_eq!(self.rank(), rhs.len());
        let mut rhs_fft = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        rhs_fft.extend((0..self.rank()).map(|_| CC.zero()));
        self.generalized_fft.fft_forward(rhs, &mut rhs_fft, self.base_ring().get_ring());
        self.generalized_fft.mul_assign_fft(lhs, &rhs_fft, self.base_ring().get_ring());
    }
    
    fn from_int(&self, value: i32) -> Self::Element {
        self.from(self.base_ring().get_ring().from_int(value))
    }

    fn zero(&self) -> Self::Element {
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        result.extend((0..self.rank()).map(|_| self.base_ring().zero()));
        return result;
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        assert_eq!(self.generalized_fft.rank(), lhs.len());
        assert_eq!(self.generalized_fft.rank(), rhs.len());
        (0..self.rank()).all(|i| self.base_ring().eq_el(&lhs[i], &rhs[i]))
    }
    
    fn is_commutative(&self) -> bool { true }
    fn is_noetherian(&self) -> bool { true }

    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        let poly_ring = DensePolyRing::new(self.base_ring(), "X");
        poly_ring.get_ring().dbg(&RingRef::new(self).poly_repr(&poly_ring, value, self.base_ring().identity()), out)
    }

    fn square(&self, value: &mut Self::Element) {
        assert_eq!(self.rank(), value.len());
        let mut value_fft = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        value_fft.extend((0..self.rank()).map(|_| CC.zero()));
        self.generalized_fft.fft_forward(value, &mut value_fft, self.base_ring().get_ring());
        for i in 0..self.rank() {
            Complex64::RING.square(&mut value_fft[i]);
        }
        self.generalized_fft.fft_backward(&mut value_fft, value, self.base_ring().get_ring());
    }

    fn pow_gen<I: IntegerRingStore>(&self, x: Self::Element, power: &El<I>, integers: I) -> Self::Element 
        where I::Type: IntegerRing
    {
        assert_eq!(self.rank(), x.len());
        let mut x_fft = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        x_fft.extend((0..self.rank()).map(|_| CC.zero()));
        self.generalized_fft.fft_forward(&x, &mut x_fft, self.base_ring().get_ring());
        algorithms::sqr_mul::generic_abs_square_and_multiply(
            x_fft, 
            power, 
            integers, 
            |mut a| {
                self.square(&mut a);
                return a;
            }, 
            |a, mut b| {
                self.generalized_fft.mul_assign_fft(&mut b, a, self.base_ring().get_ring());
                return b;
            }, 
            self.one()
        )
    }

    fn characteristic<I: IntegerRingStore>(&self, ZZ: &I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        self.base_ring().characteristic(ZZ)     
    }
}

pub struct WRTCanonicalBasisElementCreator<'a, R, F, A>
    where R: RingStore,
        R::Type: ZnRing,
        F: GeneralizedFFTSelfIso<R::Type>,
        A: Allocator + Clone
{
    ring: &'a ComplexFFTBasedRingBase<R, F, A>,
}

impl<'a, R, F, A> Copy for WRTCanonicalBasisElementCreator<'a, R, F, A>
    where R: RingStore,
        R::Type: ZnRing,
        F: GeneralizedFFTSelfIso<R::Type>,
        A: Allocator + Clone
{}

impl<'a, R, F, A> Clone for WRTCanonicalBasisElementCreator<'a, R, F, A>
    where R: RingStore,
        R::Type: ZnRing,
        F: GeneralizedFFTSelfIso<R::Type>,
        A: Allocator + Clone
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, 'b, R, F, A> FnOnce<(&'b [El<R>],)> for WRTCanonicalBasisElementCreator<'a, R, F, A>
    where R: RingStore,
        R::Type: ZnRing,
        F: GeneralizedFFTSelfIso<R::Type>,
        A: Allocator + Clone
{
    type Output = El<ComplexFFTBasedRing<R, F, A>>;

    extern "rust-call" fn call_once(self, args: (&'b [El<R>],)) -> Self::Output {
        self.call(args)
    }
}

impl<'a, 'b, R, F, A> FnMut<(&'b [El<R>],)> for WRTCanonicalBasisElementCreator<'a, R, F, A>
    where R: RingStore,
        R::Type: ZnRing,
        F: GeneralizedFFTSelfIso<R::Type>,
        A: Allocator + Clone
{
    extern "rust-call" fn call_mut(&mut self, args: (&'b [El<R>],)) -> Self::Output {
        self.call(args)
    }
}

impl<'a, 'b, R, F, A> Fn<(&'b [El<R>],)> for WRTCanonicalBasisElementCreator<'a, R, F, A>
    where R: RingStore,
        R::Type: ZnRing,
        F: GeneralizedFFTSelfIso<R::Type>,
        A: Allocator + Clone
{
    extern "rust-call" fn call(&self, args: (&'b [El<R>],)) -> Self::Output {
        self.ring.from_canonical_basis(args.0.iter().map(|x| self.ring.base_ring().clone_el(x)))
    }
}

impl<R, F, A> FiniteRing for ComplexFFTBasedRingBase<R, F, A>
    where R: RingStore,
        R::Type: ZnRing,
        F: GeneralizedFFTSelfIso<R::Type>,
        A: Allocator + Clone
{
    type ElementsIter<'a> = MultiProduct<<R::Type as FiniteRing>::ElementsIter<'a>, WRTCanonicalBasisElementCreator<'a, R, F, A>, CloneRingEl<&'a R>, Self::Element>
        where Self: 'a;

    fn elements<'a>(&'a self) -> Self::ElementsIter<'a> {
        multi_cartesian_product((0..self.rank()).map(|_| self.base_ring().elements()), WRTCanonicalBasisElementCreator { ring: self }, CloneRingEl(self.base_ring()))
    }

    fn random_element<G: FnMut() -> u64>(&self, mut rng: G) -> Self::Element {
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        result.extend((0..self.rank()).map(|_| self.base_ring().random_element(&mut rng)));
        return result;
    }

    fn size<I: IntegerRingStore>(&self, ZZ: &I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        let characteristic = self.base_ring().size(ZZ)?;
        if ZZ.get_ring().representable_bits().is_none() || ZZ.get_ring().representable_bits().unwrap() >= self.rank() * ZZ.abs_log2_ceil(&characteristic).unwrap() {
            Some(ZZ.pow(characteristic, self.rank()))
        } else {
            None
        }
    }
}

impl<R, F, A> FreeAlgebra for ComplexFFTBasedRingBase<R, F, A>
    where R: RingStore,
        R::Type: ZnRing,
        F: GeneralizedFFTSelfIso<R::Type>,
        A: Allocator + Clone
{
    type VectorRepresentation<'a> = CloneElFn<&'a [El<R>], El<R>, CloneRingEl<&'a R>>
        where Self: 'a;

    fn canonical_gen(&self) -> Self::Element {
        let mut result = self.zero();
        result[1] = self.base_ring().one();
        return result;
    }

    fn rank(&self) -> usize {
        self.generalized_fft.rank()
    }

    fn wrt_canonical_basis<'a>(&'a self, el: &'a Self::Element) -> Self::VectorRepresentation<'a> {
        (&el[..]).into_fn(CloneRingEl(self.base_ring()))
    }
}

impl<R, F, A> RingExtension for ComplexFFTBasedRingBase<R, F, A>
    where R: RingStore,
        R::Type: ZnRing,
        F: GeneralizedFFTSelfIso<R::Type>,
        A: Allocator + Clone
{
    type BaseRing = R;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing {
        &self.ring
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        let mut result = self.zero();
        result[0] = x;
        return result;
    }
}

impl<P, R, F, A> CanHomFrom<P> for ComplexFFTBasedRingBase<R, F, A>
    where R: RingStore,
        R::Type: ZnRing,
        F: GeneralizedFFTSelfIso<R::Type>,
        A: Allocator + Clone,
        P: PolyRing,
        R::Type: CanHomFrom<<P::BaseRing as RingStore>::Type>
{
    type Homomorphism = <R::Type as CanHomFrom<<P::BaseRing as RingStore>::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &P) -> Option<Self::Homomorphism> {
        self.base_ring().get_ring().has_canonical_hom(from.base_ring().get_ring())
    }

    fn map_in(&self, from: &P, el: <P as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in_ref(from, &el, hom)
    }

    fn map_in_ref(&self, from: &P, el: &<P as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        assert!(from.degree(&el).unwrap_or(0) < self.rank(), "`ComplexFFTBasedRing` currently only supports mapping in elements from polynomial rings that don't require additional reduction");
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        result.extend((0..self.rank()).map(|i| self.base_ring().get_ring().map_in_ref(from.base_ring().get_ring(), from.coefficient_at(&el, i), hom)));
        return result;
    }
}

impl<R1, R2, F1, F2, A1, A2> CanHomFrom<ComplexFFTBasedRingBase<R2, F2, A2>> for ComplexFFTBasedRingBase<R1, F1, A1>
    where R1: RingStore,
        R1::Type: ZnRing + CanHomFrom<R2::Type>,
        F1: GeneralizedFFTSelfIso<R1::Type>,
        A1: Allocator + Clone,
        R2: RingStore,
        R2::Type: ZnRing,
        F2: GeneralizedFFTSelfIso<R2::Type>,
        A2: Allocator + Clone,
        F1: GeneralizedFFTIso<R1::Type, R2::Type, F2>
{
    type Homomorphism = <R1::Type as CanHomFrom<R2::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &ComplexFFTBasedRingBase<R2, F2, A2>) -> Option<Self::Homomorphism> {
        if self.generalized_fft.is_isomorphic(&from.generalized_fft) {
            self.base_ring().get_ring().has_canonical_hom(from.base_ring().get_ring())
        } else {
            None
        }
    }

    fn map_in(&self, from: &ComplexFFTBasedRingBase<R2, F2, A2>, el: <ComplexFFTBasedRingBase<R2, F2, A2> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        result.extend((0..self.rank()).map(|i| self.base_ring().get_ring().map_in(from.base_ring().get_ring(), from.base_ring().clone_el(&el[i]), hom)));
        return result;
    }
}

impl<R1, R2, F1, F2, A1, A2> CanIsoFromTo<ComplexFFTBasedRingBase<R2, F2, A2>> for ComplexFFTBasedRingBase<R1, F1, A1>
    where R1: RingStore,
        R1::Type: ZnRing + CanIsoFromTo<R2::Type>,
        F1: GeneralizedFFTSelfIso<R1::Type>,
        A1: Allocator + Clone,
        R2: RingStore,
        R2::Type: ZnRing,
        F2: GeneralizedFFTSelfIso<R2::Type>,
        A2: Allocator + Clone,
        F1: GeneralizedFFTIso<R1::Type, R2::Type, F2>
{
    type Isomorphism = <R1::Type as CanIsoFromTo<R2::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &ComplexFFTBasedRingBase<R2, F2, A2>) -> Option<Self::Isomorphism> {
        if self.generalized_fft.is_isomorphic(&from.generalized_fft) {
            self.base_ring().get_ring().has_canonical_iso(from.base_ring().get_ring())
        } else {
            None
        }
    }

    fn map_out(&self, from: &ComplexFFTBasedRingBase<R2, F2, A2>, el: Self::Element, iso: &Self::Isomorphism) -> <ComplexFFTBasedRingBase<R2, F2, A2> as RingBase>::Element {
        let mut result = Vec::with_capacity_in(self.rank(), from.allocator.clone());
        result.extend((0..self.rank()).map(|i| self.base_ring().get_ring().map_out(from.base_ring().get_ring(), self.base_ring().clone_el(&el[i]), iso)));
        return result;
    }
}