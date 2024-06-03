use std::collections::HashMap;

use feanor_math::algorithms::fft::*;
use feanor_math::algorithms;
use feanor_math::integer::int_cast;
use feanor_math::rings::poly::*;
use feanor_math::algorithms::fft::complex_fft::ErrorEstimate;
use feanor_math::divisibility::DivisibilityRing;
use feanor_math::mempool::*;
use feanor_math::primitive_int::*;
use feanor_math::ring::*;
use feanor_math::homomorphism::*;
use feanor_math::rings::float_complex::{Complex64, Complex64El};
use feanor_math::rings::poly::sparse_poly::SparsePolyRing;
use feanor_math::rings::zn::{ZnRing, ZnRingStore};
use feanor_math::vector::*;

use super::complex_fft_ring::{GeneralizedFFT, ComplexFFTBasedRingBase, GeneralizedFFTIso};
use crate::cyclotomic::*;

const CC: Complex64 = Complex64::RING;
const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;

///
/// Euler's totient function
/// 
fn phi(factorization: &Vec<(i64, usize)>) -> i64 {
    ZZ.prod(factorization.iter().map(|(p, e)| (p - 1) * ZZ.pow(*p, e - 1)))
}

///
/// A [`GeneralizedFFT`] for odd-conductor cyclotomic rings, i.e. `Z[X]/(Phi_n(X), q)` for
/// `n` an odd integer. Usually, this will only be used together with [`ComplexFFTBasedRing`].
/// 
/// # See also
/// 
/// [`super::pow2_cyclotomic::Pow2CyclotomicFFT`] in the case that the cyclotomic conductor is a power of
/// two instead.
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
/// # use he_ring::complexfft::odd_cyclotomic::OddCyclotomicFFT;
/// type TheRing = ComplexFFTBasedRing<OddCyclotomicFFT<Zn, bluestein::FFTTableBluestein<Complex64>>, DefaultMemoryProvider, DefaultMemoryProvider>;
/// 
/// // the ring `F7[X]/(Phi_15(X)) = F7[X]/(X^8 - X^7 + X^5 - X^4 + X^3 - X + 1)`
/// let R = <TheRing as RingStore>::Type::new(Zn::new(7), 15, default_memory_provider!(), default_memory_provider!());
/// let root_of_unity = R.canonical_gen();
/// assert_eq!(8, R.rank());
/// assert_eq!(15, R.n());
/// assert_el_eq!(&R, &R.one(), &R.pow(root_of_unity, 15));
/// ```
/// 
pub struct OddCyclotomicFFT<R: ZnRingStore, F: FFTTable<Ring = Complex64> + ErrorEstimate> 
    where R::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>
{
    fft_table: F,
    base_ring: R,
    n_factorization: Vec<(i64, usize)>,
    zeta_pow_rank: HashMap<usize, Complex64El>,
    rank: usize
}

impl<R: ZnRingStore, F: FFTTable<Ring = Complex64> + ErrorEstimate> OddCyclotomicFFT<R, F> 
    where R::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>
{
    ///
    /// Computing this "generalized FFT" requires evaluating a polynomial at all primitive
    /// `n`-th roots of unity. However, the base FFT will compute the evaluation at all `n`-th
    /// roots of unity. This function gives an iterator over the indices (indices into the output of
    /// the base FFT) that correspond to the primitive roots. Note that the base FFT is used via
    /// [`FFTTable::unordered_fft()`], so this is nontrivial.
    /// 
    fn fft_output_indices<'a>(&'a self) -> impl 'a + Iterator<Item = usize> {
        (0..self.rank()).scan(-1, move |state: &mut i64, _| {
            *state += 1;
            let mut power = self.fft_table.unordered_fft_permutation(*state as usize) as i64;
            while self.n_factorization.iter().any(|(p, _)| power % *p == 0) {
                *state += 1;
                power = self.fft_table.unordered_fft_permutation(*state as usize) as i64;
            }
            debug_assert!(*state >= 0);
            debug_assert!(*state < self.fft_table.len() as i64);
            return Some(*state as usize);
        })
    }
}

impl<R: ZnRingStore, F: FFTTable<Ring = Complex64> + ErrorEstimate> OddCyclotomicFFT<R, F> 
    where R::Type: ZnRing + DivisibilityRing + CanHomFrom<StaticRingBase<i64>>, 
        R: Clone
{
    pub fn create(base_ring: R, fft_table: F) -> Self {
        let n_factorization = algorithms::int_factor::factor(&ZZ, fft_table.len() as i64);
        let rank = phi(&n_factorization) as usize;

        let poly_ring = SparsePolyRing::new(&base_ring, "X");
        let cyclotomic_poly = algorithms::cyclotomic::cyclotomic_polynomial(&poly_ring, fft_table.len());
        assert_eq!(poly_ring.degree(&cyclotomic_poly).unwrap(), rank);
        let mut zeta_pow_rank = HashMap::new();
        for (a, i) in poly_ring.terms(&cyclotomic_poly) {
            if i != rank {
                zeta_pow_rank.insert(i, CC.from_f64(int_cast(base_ring.smallest_lift(base_ring.negate(base_ring.clone_el(a))), &ZZ, base_ring.integer_ring()) as f64));
            }
        }

        return Self { fft_table, base_ring, n_factorization, zeta_pow_rank, rank };
    }
}

impl<R: ZnRingStore, F: FFTTable<Ring = Complex64> + ErrorEstimate> OddCyclotomicFFT<R, F> 
    where R::Type: ZnRing + DivisibilityRing + CanHomFrom<StaticRingBase<i64>>
{
    pub fn n(&self) -> usize {
        self.fft_table.len()
    }
}

impl<R: ZnRingStore, F: FFTTable<Ring = Complex64> + ErrorEstimate> GeneralizedFFT for OddCyclotomicFFT<R, F> 
    where R::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>
{
    type BaseRingBase = R::Type;
    type BaseRingStore = R;

    fn base_ring(&self) -> &Self::BaseRingStore {
        &self.base_ring
    }

    fn fft_backward<M_Zn: MemoryProvider<El<Self::BaseRingStore>>, M_CC: MemoryProvider<Complex64El>>(&self, data: &mut [Complex64El], destination: &mut [El<Self::BaseRingStore>], _: &M_Zn, memory_provider_cc: &M_CC) {
        let mut tmp = memory_provider_cc.get_new_init(self.fft_table.len(), |_| CC.zero());
        for (j, i) in self.fft_output_indices().enumerate() {
            tmp[i] = data[j];
        }
        self.fft_table.unordered_inv_fft(&mut tmp[..], memory_provider_cc, &CC.identity());
        for i in (self.rank()..self.fft_table.len()).rev() {
            let factor = tmp[i];
            for (j, c) in self.zeta_pow_rank.iter() {
                CC.add_assign(&mut tmp[i - self.rank() + *j], CC.mul_ref(&factor, c));
            }
        }

        for i in 0..self.rank() {
            let (re, im) = Complex64::RING.closest_gaussian_int(tmp[i]);
            debug_assert_eq!(0, im);
            debug_assert!(CC.abs(CC.sub(tmp[i], CC.from_f64(re as f64))) < 0.01);
            destination[i] = self.base_ring().coerce(&ZZ, re);
        }
    }

    fn fft_forward<M: MemoryProvider<Complex64El>>(&self, data: &[El<Self::BaseRingStore>], destination: &mut [Complex64El], memory_provider: &M) {
        let mut tmp = memory_provider.get_new_init(self.fft_table.len(), |_| CC.zero());
        for i in 0..self.rank() {
            tmp[i] = CC.from_f64(int_cast(self.base_ring().smallest_lift(self.base_ring().clone_el(&data[i])), &ZZ, self.base_ring().integer_ring()) as f64);
        }

        self.fft_table.unordered_fft(&mut tmp[..], memory_provider, &CC.identity());
        for (j, i) in self.fft_output_indices().enumerate() {
            destination[j] = tmp[i];
        }
    }

    fn rank(&self) -> usize {
        self.rank
    }
}

impl<R1: ZnRingStore, F1, R2: ZnRingStore, F2> GeneralizedFFTIso<OddCyclotomicFFT<R1, F1>> for OddCyclotomicFFT<R2, F2>
    where R1::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>, 
        F1: FFTTable<Ring = Complex64> + ErrorEstimate,
        R2::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>, 
        F2: FFTTable<Ring = Complex64> + ErrorEstimate
{
    fn is_isomorphic(&self, other: &OddCyclotomicFFT<R1, F1>) -> bool {
        self.fft_table.len() == other.fft_table.len()
    }
}

impl<R: ZnRingStore, F: FFTTable<Ring = Complex64> + ErrorEstimate, M_Zn, M_CC> CyclotomicRing for ComplexFFTBasedRingBase<OddCyclotomicFFT<R, F>, M_Zn, M_CC>
    where R::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>,
        M_Zn: MemoryProvider<El<R>>,
        M_CC: MemoryProvider<Complex64El>
{
    fn n(&self) -> usize {
        self.generalized_fft().fft_table.len()
    }
}

impl<R: ZnRingStore, M_Zn, M_CC> ComplexFFTBasedRingBase<OddCyclotomicFFT<R, bluestein::FFTTableBluestein<Complex64>>, M_Zn, M_CC>
    where R::Type: ZnRing + DivisibilityRing + CanHomFrom<StaticRingBase<i64>>, 
        R: Clone,
        M_Zn: MemoryProvider<El<R>>,
        M_CC: MemoryProvider<Complex64El>
{
    pub fn new(base_ring: R, n: usize, memory_provider_zn: M_Zn, memory_provider_cc: M_CC) -> RingValue<Self> {
        RingValue::from(
            Self::from_generalized_fft(
                OddCyclotomicFFT::create(
                    base_ring, 
                    bluestein::FFTTableBluestein::for_complex(Complex64::RING, n)
                ), 
                memory_provider_zn, 
                memory_provider_cc
            )
        )
    }
}

#[cfg(test)]
use feanor_math::rings::extension::generic_test_free_algebra_axioms;
#[cfg(test)]
use feanor_math::rings::zn::zn_64::Zn;
#[cfg(test)]
use feanor_math::default_memory_provider;

#[test]
fn test_ring_axioms() {
    let Fp = Zn::new(65537);
    let R = ComplexFFTBasedRingBase::<OddCyclotomicFFT<_, bluestein::FFTTableBluestein<_>>, _, _>::new(Fp, 9, default_memory_provider!(), default_memory_provider!());
    feanor_math::ring::generic_tests::test_ring_axioms(&R, [
        ring_literal!(&R, [0, 0, 0, 0, 0, 0]),
        ring_literal!(&R, [1, 0, 0, 0, 0, 0]),
        ring_literal!(&R, [-1, 0, 0, 0, 0, 0]),
        ring_literal!(&R, [0, 0, 0, 0, 0, 1]),
        ring_literal!(&R, [0, 0, 0, 0, 0, -1]),
        ring_literal!(&R, [1, 1, 1, 1, 1, 1]),
        ring_literal!(&R, [1, -1, 0, 0, 0, 0])
    ].into_iter());
}

#[test]
fn test_free_algebra_axioms() {
    let Fp = Zn::new(65537);
    let R = ComplexFFTBasedRingBase::<OddCyclotomicFFT<_, bluestein::FFTTableBluestein<_>>, _, _>::new(Fp, 9, default_memory_provider!(), default_memory_provider!());
    generic_test_free_algebra_axioms(R);
}

#[test]
fn test_cyclotomic_ring_axioms() {
    let Fp = Zn::new(65537);
    let R = ComplexFFTBasedRingBase::<OddCyclotomicFFT<_, bluestein::FFTTableBluestein<_>>, _, _>::new(Fp, 9, default_memory_provider!(), default_memory_provider!());
    generic_test_cyclotomic_ring_axioms(R);
}

#[test]
fn test_fft_output_indices() {
    let Fp = Zn::new(257);
    let S = ComplexFFTBasedRingBase::<OddCyclotomicFFT<_, bluestein::FFTTableBluestein<_>>, _, _>::new(Fp, 7, default_memory_provider!(), default_memory_provider!());

    assert_eq!(vec![1, 2, 3, 4, 5, 6], S.get_ring().generalized_fft().fft_output_indices().collect::<Vec<_>>());
}