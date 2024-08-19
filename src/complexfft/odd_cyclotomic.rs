use std::alloc::Allocator;
use std::alloc::Global;
use std::collections::HashMap;

use feanor_math::algorithms::fft::*;
use feanor_math::algorithms;
use feanor_math::integer::*;
use feanor_math::rings::float_complex::Complex64Base;
use feanor_math::rings::poly::*;
use feanor_math::algorithms::fft::complex_fft::FFTErrorEstimate;
use feanor_math::primitive_int::*;
use feanor_math::ring::*;
use feanor_math::homomorphism::*;
use feanor_math::rings::float_complex::{Complex64, Complex64El};
use feanor_math::rings::poly::sparse_poly::SparsePolyRing;
use feanor_math::rings::zn::zn_64;
use feanor_math::rings::zn::{ZnRing, ZnRingStore};

use super::automorphism::CyclotomicRingDecomposition;
use super::complex_fft_ring::CCFFTRing;
use super::complex_fft_ring::{RingDecomposition, CCFFTRingBase, SameNumberRing};
use crate::complexfft::complex_fft_ring::mul_assign_fft_base;
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
/// # use feanor_math::algorithms::fft::*;
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::homomorphism::Homomorphism;
/// # use feanor_math::rings::float_complex::*;
/// # use feanor_math::rings::extension::FreeAlgebra;
/// # use feanor_math::homomorphism::Identity;
/// # use feanor_math::rings::extension::FreeAlgebraStore;
/// # use he_ring::complexfft::complex_fft_ring::*;
/// # use he_ring::cyclotomic::*;
/// # use he_ring::complexfft::odd_cyclotomic::OddCyclotomicFFT;
/// type TheRing = CCFFTRing<Zn, OddCyclotomicFFT<Zn, bluestein::BluesteinFFT<Complex64Base, Complex64Base, Identity<Complex64>>>>;
/// 
/// // the ring `F7[X]/(Phi_15(X)) = F7[X]/(X^8 - X^7 + X^5 - X^4 + X^3 - X + 1)`
/// let R = <TheRing as RingStore>::Type::new(Zn::new(7), 15);
/// let root_of_unity = R.canonical_gen();
/// assert_eq!(8, R.rank());
/// assert_eq!(15, R.n());
/// assert_el_eq!(&R, &R.one(), &R.pow(root_of_unity, 15));
/// ```
/// 
pub struct OddCyclotomicFFT<R, F, A = Global> 
    where R: RingStore,
        R::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>,
        F: FFTAlgorithm<Complex64Base> + FFTErrorEstimate,
        A: Allocator + Clone
{
    ring: R,
    fft_table: F,
    /// contains `usize::MAX` whenenver the fft output index corresponds to a non-primitive root of unity, and an index otherwise
    fft_output_indices_to_indices: Vec<usize>,
    zeta_pow_rank: HashMap<usize, Complex64El>,
    rank: usize,
    allocator: A    
}

impl<R, F, A> OddCyclotomicFFT<R, F, A> 
    where R: RingStore,
        R::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>,
        F: FFTAlgorithm<Complex64Base> + FFTErrorEstimate,
        A: Allocator + Clone
{
    ///
    /// Computing this "generalized FFT" requires evaluating a polynomial at all primitive
    /// `n`-th roots of unity. However, the base FFT will compute the evaluation at all `n`-th
    /// roots of unity. This function gives an iterator over the index pairs `(i, j)`, where `i` 
    /// is an index into the vector of evaluations, and `j` is an index into the output of the base 
    /// FFT.
    /// 
    fn fft_output_indices<'a>(&'a self) -> impl 'a + Iterator<Item = (usize, usize)> {
        self.fft_output_indices_to_indices.iter().enumerate().filter_map(|(i, j)| if *j == usize::MAX { None } else { Some((*j, i)) })
    }
}

impl<R, F, A> OddCyclotomicFFT<R, F, A> 
    where R: RingStore,
        R::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>,
        F: FFTAlgorithm<Complex64Base> + FFTErrorEstimate,
        A: Allocator + Clone
{
    pub fn create(ring: R, fft_table: F, allocator: A) -> Self {
        let n_factorization = algorithms::int_factor::factor(&ZZ, fft_table.len() as i64);
        let rank = phi(&n_factorization) as usize;

        let modulus = ring.integer_ring().to_float_approx(ring.modulus());
        assert!(fft_table.expected_absolute_error(modulus * modulus, modulus * modulus * f64::EPSILON + fft_table.expected_absolute_error(modulus, 0.)) < 0.5);
        let poly_ring = SparsePolyRing::new(&ring, "X");
        let cyclotomic_poly = algorithms::cyclotomic::cyclotomic_polynomial(&poly_ring, fft_table.len());
        assert_eq!(poly_ring.degree(&cyclotomic_poly).unwrap(), rank);
        let mut zeta_pow_rank = HashMap::new();
        for (a, i) in poly_ring.terms(&cyclotomic_poly) {
            if i != rank {
                zeta_pow_rank.insert(i, CC.from_f64(int_cast(ring.smallest_lift(ring.negate(ring.clone_el(a))), &ZZ, ring.integer_ring()) as f64));
            }
        }

        let fft_output_indices_to_indices = (0..fft_table.len()).scan(0, |state, i| {
            if n_factorization.iter().all(|(p, _)| i as i64 % *p != 0) {
                *state += 1;
                return Some(*state - 1);
            } else {
                return Some(usize::MAX);
            }
        }).collect::<Vec<_>>();

        return Self { ring, fft_table, zeta_pow_rank, rank, allocator, fft_output_indices_to_indices };
    }
}

impl<R, F, A> OddCyclotomicFFT<R, F, A> 
    where R: RingStore,
        R::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>,
        F: FFTAlgorithm<Complex64Base> + FFTErrorEstimate,
        A: Allocator + Clone
{
    pub fn n(&self) -> usize {
        self.fft_table.len()
    }
}

impl<R, F, A> RingDecomposition<R::Type> for OddCyclotomicFFT<R, F, A> 
    where R: RingStore,
        R::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>,
        F: FFTAlgorithm<Complex64Base> + FFTErrorEstimate,
        A: Allocator + Clone
{
    fn fft_backward(&self, data: &mut [Complex64El], destination: &mut [El<R>], ring: &R::Type) {
        assert!(self.ring.get_ring() == ring);
        let mut tmp = Vec::with_capacity_in(self.fft_table.len(), self.allocator.clone());
        tmp.extend((0..self.fft_table.len()).map(|_| CC.zero()));
        for (j, i) in self.fft_output_indices() {
            tmp[i] = data[j];
        }
        self.fft_table.unordered_inv_fft(&mut tmp[..], CC.get_ring());
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
            destination[i] = self.ring.coerce(&ZZ, re);
        }
    }

    fn fft_forward(&self, data: &[El<R>], destination: &mut [Complex64El], ring: &R::Type) {
        assert!(self.ring.get_ring() == ring);
        let mut tmp = Vec::with_capacity_in(self.fft_table.len(), self.allocator.clone());
        tmp.extend((0..self.fft_table.len()).map(|_| CC.zero()));
        for i in 0..self.rank() {
            tmp[i] = CC.from_f64(int_cast(self.ring.smallest_lift(self.ring.clone_el(&data[i])), &ZZ, self.ring.integer_ring()) as f64);
        }

        self.fft_table.unordered_fft(&mut tmp[..], CC.get_ring());
        for (j, i) in self.fft_output_indices() {
            destination[j] = tmp[i];
        }
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn mul_assign_fft(&self, lhs: &mut [El<R>], rhs: &[Complex64El], ring: &R::Type) {
        assert!(self.ring.get_ring() == ring);
        let mut tmp = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        tmp.resize(self.rank(), CC.zero());
        mul_assign_fft_base(self, lhs, rhs, &mut tmp, ring);
    }
}

impl<R, F, A> CyclotomicRingDecomposition<R::Type> for OddCyclotomicFFT<R, F, A> 
    where R: RingStore,
        R::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>,
        F: FFTAlgorithm<Complex64Base> + FFTErrorEstimate,
        A: Allocator + Clone
{
    fn galois_group_mulrepr(&self) -> zn_64::Zn {
        zn_64::Zn::new(self.fft_table.len() as u64)
    }

    fn permute_galois_action(&self, src: &[Complex64El], dst: &mut [Complex64El], galois_element: El<zn_64::Zn>) {
        let Zn = self.galois_group_mulrepr();
        let hom = Zn.can_hom(&StaticRing::<i64>::RING).unwrap();
        
        for (j, i) in self.fft_output_indices() {
            dst[j] = src[self.fft_output_indices_to_indices[self.fft_table.unordered_fft_permutation_inv(
                Zn.smallest_positive_lift(Zn.mul(galois_element, hom.map(self.fft_table.unordered_fft_permutation(i) as i64))) as usize
            )]];
        }
    }
}

impl<R1, F1, A1, R2, F2, A2> SameNumberRing<R2::Type, R1::Type, OddCyclotomicFFT<R1, F1, A1>> for OddCyclotomicFFT<R2, F2, A2>
    where R1: RingStore,
        R1::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>,
        F1: FFTAlgorithm<Complex64Base> + FFTErrorEstimate,
        A1: Allocator + Clone,
        R2: RingStore,
        R2::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>,
        F2: FFTAlgorithm<Complex64Base> + FFTErrorEstimate,
        A2: Allocator + Clone
{
    fn is_isomorphic(&self, other: &OddCyclotomicFFT<R1, F1, A1>) -> bool {
        self.fft_table.len() == other.fft_table.len()
    }
}

impl<R, F, A1, A2> CyclotomicRing for CCFFTRingBase<R, OddCyclotomicFFT<R, F, A1>, A2>
    where R: RingStore,
        R::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>,
        F: FFTAlgorithm<Complex64Base> + FFTErrorEstimate,
        A1: Allocator + Clone,
        A2: Allocator + Clone
{
    fn n(&self) -> usize {
        self.generalized_fft().fft_table.len()
    }

    fn apply_galois_action(&self, el: &Self::Element, s: zn_64::ZnEl) -> Self::Element {
        unimplemented!()
    }
}

pub type DefaultOddCyclotomicCCFFTRingBase<R = zn_64::Zn> = CCFFTRingBase<R, OddCyclotomicFFT<R, bluestein::BluesteinFFT<Complex64Base, Complex64Base, Identity<Complex64>>>>;
pub type DefaultOddCyclotomicCCFFTRing<R = zn_64::Zn> = CCFFTRing<R, OddCyclotomicFFT<R, bluestein::BluesteinFFT<Complex64Base, Complex64Base, Identity<Complex64>>>>;

impl<R, A> CCFFTRingBase<R, OddCyclotomicFFT<R, bluestein::BluesteinFFT<Complex64Base, Complex64Base, Identity<Complex64>, A>, A>, A>
    where R: RingStore + Clone,
        R::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>,
        A: Allocator + Clone + Default
{
    pub fn new(ring: R, n: usize) -> RingValue<Self> {
        let allocator = A::default();
        RingValue::from(
            Self::from_generalized_fft(
                ring.clone(),
                OddCyclotomicFFT::create(
                    ring, 
                    bluestein::BluesteinFFT::for_complex(Complex64::RING, n, allocator.clone()),
                    allocator.clone()
                ), 
                allocator
            )
        )
    }
}

#[cfg(test)]
use feanor_math::rings::zn::zn_64::Zn;
#[cfg(test)]
use feanor_math::assert_el_eq;

#[test]
fn test_ring_axioms() {
    let Fp = Zn::new(65537);
    let R = DefaultOddCyclotomicCCFFTRingBase::new(Fp, 9);
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
    let R = DefaultOddCyclotomicCCFFTRingBase::new(Fp, 9);
    feanor_math::rings::extension::generic_tests::test_free_algebra_axioms(R);
}

#[test]
fn test_cyclotomic_ring_axioms() {
    let Fp = Zn::new(65537);
    let R = DefaultOddCyclotomicCCFFTRingBase::new(Fp, 9);
    generic_test_cyclotomic_ring_axioms(R);
}

#[test]
fn test_fft_output_indices() {
    let Fp = Zn::new(257);
    let S = DefaultOddCyclotomicCCFFTRingBase::new(Fp, 7);

    assert_eq!(vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)], S.get_ring().generalized_fft().fft_output_indices().collect::<Vec<_>>());
}

#[test]
fn test_permute_galois_automorphism() {
    let Fp = Zn::new(257);
    let R = DefaultOddCyclotomicCCFFTRingBase::new(Fp, 7);
    let hom = R.get_ring().galois_group_mulrepr().into_int_hom();

    assert_el_eq!(R, ring_literal!(&R, [0, 0, 1, 0, 0, 0]), R.get_ring().apply_galois_action(hom.map(2), ring_literal!(&R, [0, 1, 0, 0, 0, 0])));
    assert_el_eq!(R, ring_literal!(&R, [0, 0, 0, 1, 0, 0]), R.get_ring().apply_galois_action(hom.map(3), ring_literal!(&R, [0, 1, 0, 0, 0, 0])));
    assert_el_eq!(R, ring_literal!(&R, [0, 0, 0, 0, 1, 0]), R.get_ring().apply_galois_action(hom.map(2), ring_literal!(&R, [0, 0, 1, 0, 0, 0])));
    assert_el_eq!(R, ring_literal!(&R, [-1, -1, -1, -1, -1, -1]), R.get_ring().apply_galois_action(hom.map(3), ring_literal!(&R, [0, 0, 1, 0, 0, 0])));

    let R = DefaultOddCyclotomicCCFFTRingBase::new(Fp, 15);
    let hom = R.get_ring().galois_group_mulrepr().into_int_hom();

    assert_el_eq!(R, ring_literal!(&R, [0, 0, 1, 0, 0, 0, 0, 0]), R.get_ring().apply_galois_action(hom.map(2), ring_literal!(&R, [0, 1, 0, 0, 0, 0, 0, 0])));
    assert_el_eq!(R, ring_literal!(&R, [0, 0, 0, 0, 1, 0, 0, 0]), R.get_ring().apply_galois_action(hom.map(4), ring_literal!(&R, [0, 1, 0, 0, 0, 0, 0, 0])));
    assert_el_eq!(R, ring_literal!(&R, [-1, 1, 0, -1, 1, -1, 0, 1]), R.get_ring().apply_galois_action(hom.map(8), ring_literal!(&R, [0, 1, 0, 0, 0, 0, 0, 0])));
    assert_el_eq!(R, ring_literal!(&R, [-1, 1, 0, -1, 1, -1, 0, 1]), R.get_ring().apply_galois_action(hom.map(2), ring_literal!(&R, [0, 0, 0, 0, 1, 0, 0, 0])));
}