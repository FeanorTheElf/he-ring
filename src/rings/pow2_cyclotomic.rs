use cooley_tuckey::bitreverse;
use cooley_tuckey::CooleyTuckeyFFT;
use feanor_math::algorithms;
use feanor_math::algorithms::fft::*;
use feanor_math::algorithms::miller_rabin::is_prime;
use feanor_math::algorithms::unity_root::get_prim_root_of_unity_pow2;
use feanor_math::divisibility::DivisibilityRing;
use feanor_math::primitive_int::StaticRing;
use feanor_math::ring::*;
use feanor_math::assert_el_eq;
use feanor_math::homomorphism::*;
use feanor_math::integer::*;
use feanor_math::rings::extension::FreeAlgebra;
use feanor_math::rings::extension::FreeAlgebraStore;
use feanor_math::rings::poly::*;
use feanor_math::rings::zn::zn_64;
use feanor_math::rings::zn::zn_64::ZnEl;
use feanor_math::rings::zn::FromModulusCreateableZnRing;
use feanor_math::seq::*;
use feanor_math::rings::zn::{ZnRing, ZnRingStore, zn_rns};
use feanor_math::rings::zn::zn_64::Zn;
use std::alloc::Allocator;
use std::alloc::Global;

use super::decomposition_ring;
use super::double_rns_ring;
use super::double_rns_ring::*;
use super::decomposition_ring::*;
use super::single_rns_ring;
use crate::rings::decomposition::*;
use crate::sample_primes;
use crate::StdZn;
use crate::cyclotomic::CyclotomicRing;

#[derive(Clone)]
pub struct Pow2CyclotomicDecomposableNumberRing {
    log2_n: usize
}

impl Pow2CyclotomicDecomposableNumberRing {

    pub fn new(n: usize) -> Self {
        assert!(n > 2);
        let log2_n = StaticRing::<i64>::RING.abs_log2_floor(&(n as i64)).unwrap();
        assert_eq!(n, 1 << log2_n);
        Self {
            log2_n: log2_n
        }
    }
}

impl PartialEq for Pow2CyclotomicDecomposableNumberRing {

    fn eq(&self, other: &Self) -> bool {
        self.log2_n == other.log2_n
    }
}

impl<FpTy> DecomposableNumberRing<FpTy> for Pow2CyclotomicDecomposableNumberRing
    where FpTy: RingStore + Clone,
        FpTy::Type: ZnRing
{
    type Decomposed = Pow2CyclotomicDecomposedNumberRing<FpTy, CooleyTuckeyFFT<FpTy::Type, FpTy::Type, Identity<FpTy>>>;

    fn product_expansion_factor(&self) -> f64 {
        (1 << (self.log2_n - 1)) as f64
    }

    fn can_to_inf_norm_expansion_factor(&self) -> f64 {
        1. / ((1 << (self.log2_n - 1)) as f64).sqrt()
    }

    fn inf_to_can_norm_expansion_factor(&self) -> f64 {
        // the l2-norm of the coefficients of `x` is at most `sqrt(n) |x|_inf`, and
        // in the power-of-two case, the canonical embedding is a scaled isometry by `sqrt(n)`
        (1 << (self.log2_n - 1)) as f64
    }

    fn mod_p(&self, Fp: FpTy) -> Self::Decomposed {
        let rank = 1 << (self.log2_n - 1);
        let mut twiddles = Vec::with_capacity(rank as usize);
        let mut inv_twiddles = Vec::with_capacity(rank as usize);

        let Fp_as_field = (&Fp).as_field().ok().unwrap();
        let root_of_unity = get_prim_root_of_unity_pow2(&Fp_as_field, self.log2_n).unwrap();
        let zeta = Fp_as_field.get_ring().unwrap_element(root_of_unity);
        let fft_table = CooleyTuckeyFFT::new(Fp.clone(), Fp.pow(Fp.clone_el(&zeta), 2), self.log2_n - 1);

        let mut current = Fp.one();
        let mut current_inv = Fp.one();
        let zeta_inv = Fp.pow(Fp.clone_el(&zeta), 2 * rank as usize - 1);
        for _ in 0..rank {
            twiddles.push(Fp.clone_el(&current));
            inv_twiddles.push(Fp.clone_el(&current_inv));
            Fp.mul_assign_ref(&mut current, &zeta);
            Fp.mul_assign_ref(&mut current_inv, &zeta_inv);
        }
        return Pow2CyclotomicDecomposedNumberRing { ring: Fp, fft_table, twiddles, inv_twiddles };
    }

    fn largest_suitable_prime(&self, leq_than: i64) -> Option<i64> {
        let modulus = 1 << self.log2_n;
        let mut current = (leq_than - 1) - ((leq_than - 1) % modulus) + 1;
        while current > 0 && !is_prime(StaticRing::<i64>::RING, &current, 10) {
            current -= modulus;
        }
        if current <= 0 {
            return None;
        } else {
            assert!(current <= leq_than);
            return Some(current);
        }
    }
    
    fn generating_poly<P>(&self, poly_ring: P) -> El<P>
        where P: RingStore,
            P::Type: PolyRing,
            <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing
    {
        poly_ring.add(poly_ring.pow(poly_ring.indeterminate(), 1 << (self.log2_n - 1)), poly_ring.one())
    }

    fn rank(&self) -> usize {
        1 << (self.log2_n - 1)
    }
}

impl<FpTy> DecomposableCyclotomicNumberRing<FpTy> for Pow2CyclotomicDecomposableNumberRing
    where FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>
{
    type DecomposedAsCyclotomic = Pow2CyclotomicDecomposedNumberRing<FpTy, CooleyTuckeyFFT<FpTy::Type, FpTy::Type, Identity<FpTy>>>;

    fn n(&self) -> u64 {
        1 << self.log2_n
    }
}

impl<R, F> DecomposedCyclotomicNumberRing<R::Type> for Pow2CyclotomicDecomposedNumberRing<R, F> 
    where R: RingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + DivisibilityRing,
        F: FFTAlgorithm<R::Type> + PartialEq
{
    fn n(&self) -> u64 {
        2 * self.fft_table.len() as u64
    }

    fn permute_galois_action(&self, src: &[<R::Type as RingBase>::Element], dst: &mut [<R::Type as RingBase>::Element], galois_element: zn_64::ZnEl) {
        assert_eq!(self.rank(), src.len());
        assert_eq!(self.rank(), dst.len());

        let ring = self.base_ring();
        let index_ring = self.cyclotomic_index_ring();
        let hom = index_ring.can_hom(&StaticRing::<i64>::RING).unwrap();
        let bitlength = StaticRing::<i64>::RING.abs_log2_ceil(&(self.rank() as i64)).unwrap();
        debug_assert_eq!(1 << bitlength, self.rank());

        // the elements of src resp. dst follow an order derived from the bitreversing order of the underlying FFT
        let index_to_galois_el = |i: usize| hom.map(2 * bitreverse(i, bitlength) as i64 + 1);
        let galois_el_to_index = |s: ZnEl| bitreverse((index_ring.smallest_positive_lift(s) as usize - 1) / 2, bitlength);

        for i in 0..self.rank() {
            dst[i] = ring.clone_el(&src[galois_el_to_index(index_ring.mul(galois_element, index_to_galois_el(i)))]);
        }
    }
}

pub struct Pow2CyclotomicDecomposedNumberRing<R, F> 
    where R: RingStore,
        R::Type: ZnRing,
        F: FFTAlgorithm<R::Type> + PartialEq
{
    ring: R,
    fft_table: F,
    twiddles: Vec<El<R>>,
    inv_twiddles: Vec<El<R>>,
}

impl<R, F> PartialEq for Pow2CyclotomicDecomposedNumberRing<R, F> 
    where R: RingStore,
        R::Type: ZnRing,
        F: FFTAlgorithm<R::Type> + PartialEq
{
    fn eq(&self, other: &Self) -> bool {
        self.ring.get_ring() == other.ring.get_ring() && self.fft_table == other.fft_table && self.ring.eq_el(&self.twiddles[0], &other.twiddles[0])
    }
}

impl<R, F> DecomposedNumberRing<R::Type> for Pow2CyclotomicDecomposedNumberRing<R, F> 
    where R: RingStore,
        R::Type: ZnRing,
        F: FFTAlgorithm<R::Type> + PartialEq
{
    fn fft_backward(&self, data: &mut [El<R>]) {
        self.fft_table.unordered_inv_fft(&mut data[..], &self.ring);
        for i in 0..self.rank() {
            self.ring.mul_assign_ref(&mut data[i], &self.twiddles[i]);
        }
    }

    fn fft_forward(&self, data: &mut [El<R>]) {
        for i in 0..self.rank() {
            self.ring.mul_assign_ref(&mut data[i], &self.inv_twiddles[i]);
        }
        self.fft_table.unordered_fft(&mut data[..], &self.ring);
    }

    fn rank(&self) -> usize {
        self.fft_table.len()
    }

    fn base_ring(&self) -> RingRef<R::Type> {
        RingRef::new(self.ring.get_ring())
    }
}

#[test]
fn test_odd_cyclotomic_double_rns_ring() {
    double_rns_ring::test_with_number_ring(Pow2CyclotomicDecomposableNumberRing::new(8));
    double_rns_ring::test_with_number_ring(Pow2CyclotomicDecomposableNumberRing::new(16));
}

#[test]
fn test_odd_cyclotomic_decomposition_ring() {
    decomposition_ring::test_with_number_ring(Pow2CyclotomicDecomposableNumberRing::new(8));
    decomposition_ring::test_with_number_ring(Pow2CyclotomicDecomposableNumberRing::new(16));
}

#[test]
fn test_odd_cyclotomic_single_rns_ring() {
    single_rns_ring::test_with_number_ring(Pow2CyclotomicDecomposableNumberRing::new(8));
    single_rns_ring::test_with_number_ring(Pow2CyclotomicDecomposableNumberRing::new(16));
}

#[test]
fn test_permute_galois_automorphism() {
    let rns_base = zn_rns::Zn::new(vec![Zn::new(17), Zn::new(97)], BigIntRing::RING);
    let R = DoubleRNSRingBase::new_with(Pow2CyclotomicDecomposableNumberRing::new(16), rns_base, Global);
    assert_el_eq!(R, R.pow(R.canonical_gen(), 3), R.get_ring().apply_galois_action(&R.canonical_gen(), R.get_ring().cyclotomic_index_ring().int_hom().map(3)));
    assert_el_eq!(R, R.pow(R.canonical_gen(), 6), R.get_ring().apply_galois_action(&R.pow(R.canonical_gen(), 2), R.get_ring().cyclotomic_index_ring().int_hom().map(3)));
}
