use feanor_math::algorithms;
use feanor_math::integer::*;
use feanor_math::iters::multi_cartesian_product;
use feanor_math::iters::MultiProduct;
use feanor_math::iters::RingElementClone;
use feanor_math::rings::extension::*;
use feanor_math::mempool::*;
use feanor_math::ring::*;
use feanor_math::rings::finite::*;
use feanor_math::rings::float_complex::*;
use feanor_math::rings::poly::PolyRing;
use feanor_math::rings::poly::dense_poly::DensePolyRing;
use feanor_math::rings::zn::*;
use feanor_math::homomorphism::*;
use feanor_math::vector::*;
use feanor_math::vector::vec_fn::RingElVectorViewFn;

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
/// time `o(deg(f)^2)`, so if some FFT-techniques can be applied.
/// 
pub trait GeneralizedFFT {

    type BaseRingBase: ?Sized + ZnRing;
    type BaseRingStore: RingStore<Type = Self::BaseRingBase>;
    
    fn base_ring(&self) -> &Self::BaseRingStore;

    fn rank(&self) -> usize;

    fn fft_forward<M: MemoryProvider<Complex64El>>(&self, data: &[El<Self::BaseRingStore>], destination: &mut [Complex64El], memory_provider: &M);

    fn fft_backward<M_Zn: MemoryProvider<El<Self::BaseRingStore>>, M_CC: MemoryProvider<Complex64El>>(&self, data: &mut [Complex64El], destination: &mut [El<Self::BaseRingStore>], memory_provider_zn: &M_Zn, memory_provider_cc: &M_CC);

    fn mul_assign_fft<M_Zn: MemoryProvider<El<Self::BaseRingStore>>, M_CC: MemoryProvider<Complex64El>>(&self, lhs: &mut [El<Self::BaseRingStore>], rhs: &[Complex64El], memory_provider_zn: &M_Zn, memory_provider_cc: &M_CC) {
        let mut tmp = memory_provider_cc.get_new_init(self.rank(), |_| CC.zero());
        self.fft_forward(&lhs, &mut tmp, memory_provider_cc);
        for i in 0..self.rank() {
            CC.mul_assign_ref(&mut tmp[i], &rhs[i]);
        }
        self.fft_backward(&mut tmp, lhs, memory_provider_zn, memory_provider_cc);
    }
}

pub trait GeneralizedFFTIso<F: GeneralizedFFT>: GeneralizedFFT {

    fn is_isomorphic(&self, other: &F) -> bool;
}

pub trait GeneralizedFFTSelfIso: Sized + GeneralizedFFTIso<Self> {}

impl<F: GeneralizedFFT + GeneralizedFFTIso<F>> GeneralizedFFTSelfIso for F {}

pub struct ComplexFFTBasedRingBase<F: GeneralizedFFT + GeneralizedFFTSelfIso, M_Zn: MemoryProvider<El<F::BaseRingStore>>, M_CC: MemoryProvider<Complex64El>> {
    data: F,
    memory_provider_zn: M_Zn,
    memory_provider_cc: M_CC
}

pub type ComplexFFTBasedRing<F, M_Zn, M_CC> = RingValue<ComplexFFTBasedRingBase<F, M_Zn, M_CC>>;

impl<F: GeneralizedFFT + GeneralizedFFTSelfIso, M_Zn: MemoryProvider<El<F::BaseRingStore>>, M_CC: MemoryProvider<Complex64El>> ComplexFFTBasedRingBase<F, M_Zn, M_CC> {

    pub fn from_generalized_fft(data: F, memory_provider_zn: M_Zn, memory_provider_cc: M_CC) -> Self {
        Self { data, memory_provider_cc, memory_provider_zn }
    }

    pub fn generalized_fft(&self) -> &F {
        &self.data
    }

    pub fn memory_provider_zn(&self) -> &M_Zn {
        &self.memory_provider_zn
    }

    pub fn memory_provider_cc(&self) -> &M_CC {
        &self.memory_provider_cc
    }
}

impl<F: GeneralizedFFT + GeneralizedFFTSelfIso, M_Zn: MemoryProvider<El<F::BaseRingStore>>, M_CC: MemoryProvider<Complex64El>> PartialEq for ComplexFFTBasedRingBase<F, M_Zn, M_CC> {

    fn eq(&self, other: &Self) -> bool {
        self.data.base_ring().get_ring() == other.data.base_ring().get_ring() && self.data.is_isomorphic(&other.data)
    }
}

impl<F: GeneralizedFFT + GeneralizedFFTSelfIso, M_Zn: MemoryProvider<El<F::BaseRingStore>>, M_CC: MemoryProvider<Complex64El>> RingBase for ComplexFFTBasedRingBase<F, M_Zn, M_CC> {

    type Element = M_Zn::Object;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        assert_eq!(self.data.rank(), val.len());
        self.memory_provider_zn.get_new_init(self.data.rank(), |i| self.base_ring().clone_el(&val[i]))
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        assert_eq!(self.data.rank(), lhs.len());
        assert_eq!(self.data.rank(), rhs.len());
        for i in 0..self.data.rank() {
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
        assert_eq!(self.data.rank(), lhs.len());
        for i in 0..self.data.rank() {
            self.base_ring().negate_inplace(&mut lhs[i]);
        }
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.mul_assign_ref(lhs, &rhs);
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        assert_eq!(self.data.rank(), lhs.len());
        assert_eq!(self.data.rank(), rhs.len());
        let mut rhs_fft = self.memory_provider_cc.get_new_init(self.data.rank(), |_| CC.zero());
        self.data.fft_forward(rhs, &mut rhs_fft, &self.memory_provider_cc);
        self.data.mul_assign_fft(lhs, &rhs_fft, &self.memory_provider_zn, &self.memory_provider_cc);
    }
    
    fn from_int(&self, value: i32) -> Self::Element {
        self.from(self.base_ring().get_ring().from_int(value))
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        assert_eq!(self.data.rank(), lhs.len());
        assert_eq!(self.data.rank(), rhs.len());
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
        let mut value_fft = self.memory_provider_cc.get_new_init(self.data.rank(), |_| CC.zero());
        self.data.fft_forward(value, &mut value_fft, &self.memory_provider_cc);
        for i in 0..self.rank() {
            Complex64::RING.square(&mut value_fft[i]);
        }
        self.data.fft_backward(&mut value_fft, value, &self.memory_provider_zn, &self.memory_provider_cc);
    }

    fn pow_gen<I: IntegerRingStore>(&self, x: Self::Element, power: &El<I>, integers: I) -> Self::Element 
        where I::Type: IntegerRing
    {
        assert_eq!(self.rank(), x.len());
        let mut x_fft = self.memory_provider_cc.get_new_init(self.data.rank(), |_| CC.zero());
        self.data.fft_forward(&x, &mut x_fft, &self.memory_provider_cc);
        algorithms::sqr_mul::generic_abs_square_and_multiply(
            x_fft, 
            power, 
            integers, 
            |mut a| {
                self.square(&mut a);
                return a;
            }, 
            |a, mut b| {
                self.data.mul_assign_fft(&mut b, a, &self.memory_provider_zn, &self.memory_provider_cc);
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

pub struct WRTCanonicalBasisElementCreator<'a, F: GeneralizedFFT + GeneralizedFFTSelfIso, M_Zn: MemoryProvider<El<F::BaseRingStore>>, M_CC: MemoryProvider<Complex64El>> {
    ring: &'a ComplexFFTBasedRingBase<F, M_Zn, M_CC>,
}

impl<'a, F: GeneralizedFFT + GeneralizedFFTSelfIso, M_Zn: MemoryProvider<El<F::BaseRingStore>>, M_CC: MemoryProvider<Complex64El>> Copy for WRTCanonicalBasisElementCreator<'a, F, M_Zn, M_CC> {}

impl<'a, F: GeneralizedFFT + GeneralizedFFTSelfIso, M_Zn: MemoryProvider<El<F::BaseRingStore>>, M_CC: MemoryProvider<Complex64El>> Clone for WRTCanonicalBasisElementCreator<'a, F, M_Zn, M_CC> {

    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, 'b, F: GeneralizedFFT + GeneralizedFFTSelfIso, M_Zn: MemoryProvider<El<F::BaseRingStore>>, M_CC: MemoryProvider<Complex64El>> FnOnce<(&'b [El<F::BaseRingStore>],)> for WRTCanonicalBasisElementCreator<'a, F, M_Zn, M_CC> {

    type Output = M_Zn::Object;

    extern "rust-call" fn call_once(self, args: (&'b [El<F::BaseRingStore>],)) -> Self::Output {
        self.call(args)
    }
}

impl<'a, 'b, F: GeneralizedFFT + GeneralizedFFTSelfIso, M_Zn: MemoryProvider<El<F::BaseRingStore>>, M_CC: MemoryProvider<Complex64El>> FnMut<(&'b [El<F::BaseRingStore>],)> for WRTCanonicalBasisElementCreator<'a, F, M_Zn, M_CC> {

    extern "rust-call" fn call_mut(&mut self, args: (&'b [El<F::BaseRingStore>],)) -> Self::Output {
        self.call(args)
    }
}

impl<'a, 'b, F: GeneralizedFFT + GeneralizedFFTSelfIso, M_Zn: MemoryProvider<El<F::BaseRingStore>>, M_CC: MemoryProvider<Complex64El>> Fn<(&'b [El<F::BaseRingStore>],)> for WRTCanonicalBasisElementCreator<'a, F, M_Zn, M_CC> {

    extern "rust-call" fn call(&self, args: (&'b [El<F::BaseRingStore>],)) -> Self::Output {
        self.ring.from_canonical_basis(args.0.iter().map(|x| self.ring.base_ring().clone_el(x)))
    }
}

impl<F: GeneralizedFFT + GeneralizedFFTSelfIso, M_Zn: MemoryProvider<El<F::BaseRingStore>>, M_CC: MemoryProvider<Complex64El>> FiniteRing for ComplexFFTBasedRingBase<F, M_Zn, M_CC> {

    type ElementsIter<'a> = MultiProduct<<F::BaseRingBase as FiniteRing>::ElementsIter<'a>, WRTCanonicalBasisElementCreator<'a, F, M_Zn, M_CC>, RingElementClone<'a, F::BaseRingBase>, Self::Element>
        where Self: 'a;

    fn elements<'a>(&'a self) -> Self::ElementsIter<'a> {
        multi_cartesian_product((0..self.rank()).map(|_| self.base_ring().elements()), WRTCanonicalBasisElementCreator { ring: self }, RingElementClone::new(self.base_ring().get_ring()))
    }

    fn random_element<G: FnMut() -> u64>(&self, mut rng: G) -> Self::Element {
        self.memory_provider_zn().get_new_init(self.rank(), |_| self.base_ring().random_element(&mut rng))
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

impl<F: GeneralizedFFT + GeneralizedFFTSelfIso, M_Zn: MemoryProvider<El<F::BaseRingStore>>, M_CC: MemoryProvider<Complex64El>> FreeAlgebra for ComplexFFTBasedRingBase<F, M_Zn, M_CC> {

    type VectorRepresentation<'a> = RingElVectorViewFn<&'a F::BaseRingStore, &'a [El<F::BaseRingStore>], El<F::BaseRingStore>>
        where Self: 'a;

    fn canonical_gen(&self) -> Self::Element {
        self.memory_provider_zn.get_new_init(self.rank(), |i| if i == 1 { self.base_ring().one() } else { self.base_ring().zero() })
    }

    fn rank(&self) -> usize {
        self.data.rank()
    }

    fn wrt_canonical_basis<'a>(&'a self, el: &'a Self::Element) -> Self::VectorRepresentation<'a> {
        (&el[..]).as_el_fn(self.data.base_ring())
    }
}

impl<F: GeneralizedFFT + GeneralizedFFTSelfIso, M_Zn: MemoryProvider<El<F::BaseRingStore>>, M_CC: MemoryProvider<Complex64El>> RingExtension for ComplexFFTBasedRingBase<F, M_Zn, M_CC> {

    type BaseRing = F::BaseRingStore;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing {
        self.data.base_ring()
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        let mut x_opt = Some(x);
        self.memory_provider_zn.get_new_init(self.rank(), |i| if i == 0 { x_opt.take().unwrap() } else { self.base_ring().zero() })
    }
}

impl<P, F, M_Zn, M_CC> CanHomFrom<P> for ComplexFFTBasedRingBase<F, M_Zn, M_CC>
    where F: GeneralizedFFT + GeneralizedFFTSelfIso, 
        M_Zn: MemoryProvider<El<F::BaseRingStore>>, 
        M_CC: MemoryProvider<Complex64El>, 
        P: PolyRing,
        F::BaseRingBase: CanHomFrom<<P::BaseRing as RingStore>::Type>
{
    type Homomorphism = <F::BaseRingBase as CanHomFrom<<P::BaseRing as RingStore>::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &P) -> Option<Self::Homomorphism> {
        self.base_ring().get_ring().has_canonical_hom(from.base_ring().get_ring())
    }

    fn map_in(&self, from: &P, el: <P as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in_ref(from, &el, hom)
    }

    fn map_in_ref(&self, from: &P, el: &<P as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        assert!(from.degree(&el).unwrap_or(0) < self.rank());
        self.memory_provider_zn.get_new_init(self.data.rank(), |i| self.base_ring().get_ring().map_in_ref(from.base_ring().get_ring(), from.coefficient_at(&el, i), hom))
    }
}

impl<F1, F2, M1_Zn, M2_Zn, M1_CC, M2_CC> CanHomFrom<ComplexFFTBasedRingBase<F2, M2_Zn, M2_CC>> for ComplexFFTBasedRingBase<F1, M1_Zn, M1_CC>
    where F1: GeneralizedFFT + GeneralizedFFTSelfIso, 
        F2: GeneralizedFFT + GeneralizedFFTSelfIso, 
        M1_Zn: MemoryProvider<El<F1::BaseRingStore>>, 
        M1_CC: MemoryProvider<Complex64El>, 
        M2_Zn: MemoryProvider<El<F2::BaseRingStore>>, 
        M2_CC: MemoryProvider<Complex64El>,
        F1::BaseRingBase: CanHomFrom<F2::BaseRingBase>, 
        F1: GeneralizedFFTIso<F2>
{
    type Homomorphism = <F1::BaseRingBase as CanHomFrom<F2::BaseRingBase>>::Homomorphism;

    fn has_canonical_hom(&self, from: &ComplexFFTBasedRingBase<F2, M2_Zn, M2_CC>) -> Option<Self::Homomorphism> {
        if self.data.is_isomorphic(&from.data) {
            self.base_ring().get_ring().has_canonical_hom(from.base_ring().get_ring())
        } else {
            None
        }
    }

    fn map_in(&self, from: &ComplexFFTBasedRingBase<F2, M2_Zn, M2_CC>, el: <ComplexFFTBasedRingBase<F2, M2_Zn, M2_CC> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.memory_provider_zn.get_new_init(self.data.rank(), |i| self.base_ring().get_ring().map_in(from.base_ring().get_ring(), from.base_ring().clone_el(&el[i]), hom))
    }
}

impl<F1, F2, M1_Zn, M2_Zn, M1_CC, M2_CC> CanonicalIso<ComplexFFTBasedRingBase<F2, M2_Zn, M2_CC>> for ComplexFFTBasedRingBase<F1, M1_Zn, M1_CC>
    where F1: GeneralizedFFT + GeneralizedFFTSelfIso, 
        F2: GeneralizedFFT + GeneralizedFFTSelfIso, 
        M1_Zn: MemoryProvider<El<F1::BaseRingStore>>, 
        M1_CC: MemoryProvider<Complex64El>, 
        M2_Zn: MemoryProvider<El<F2::BaseRingStore>>, 
        M2_CC: MemoryProvider<Complex64El>,
        F1::BaseRingBase: CanonicalIso<F2::BaseRingBase>, 
        F1: GeneralizedFFTIso<F2>
{
    type Isomorphism = <F1::BaseRingBase as CanonicalIso<F2::BaseRingBase>>::Isomorphism;

    fn has_canonical_iso(&self, from: &ComplexFFTBasedRingBase<F2, M2_Zn, M2_CC>) -> Option<Self::Isomorphism> {
        if self.data.is_isomorphic(&from.data) {
            self.base_ring().get_ring().has_canonical_iso(from.data.base_ring().get_ring())
        } else {
            None
        }
    }

    fn map_out(&self, from: &ComplexFFTBasedRingBase<F2, M2_Zn, M2_CC>, el: Self::Element, iso: &Self::Isomorphism) -> <ComplexFFTBasedRingBase<F2, M2_Zn, M2_CC> as RingBase>::Element {
        from.memory_provider_zn.get_new_init(from.data.rank(), |i| self.base_ring().get_ring().map_out(from.base_ring().get_ring(), self.base_ring().clone_el(&el[i]), iso))
    }
}