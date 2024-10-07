use cooley_tuckey::bitreverse;
use cooley_tuckey::CooleyTuckeyFFT;
use feanor_math::algorithms;
use feanor_math::algorithms::fft::*;
use feanor_math::algorithms::miller_rabin::is_prime;
use feanor_math::algorithms::unity_root::get_prim_root_of_unity_pow2;
use feanor_math::primitive_int::StaticRing;
use feanor_math::ring::*;
use feanor_math::assert_el_eq;
use feanor_math::homomorphism::*;
use feanor_math::integer::*;
use feanor_math::rings::zn::zn_64;
use feanor_math::rings::zn::zn_64::ZnEl;
use feanor_math::rings::zn::FromModulusCreateableZnRing;
use feanor_math::seq::*;
use feanor_math::rings::zn::{ZnRing, ZnRingStore, zn_rns};
use feanor_math::rings::zn::zn_64::Zn;

use std::alloc::Allocator;
use std::alloc::Global;

use super::double_rns_ring::*;
use super::number_ring_quo::*;
use crate::rings::decomposition::*;
use crate::sample_primes;
use crate::StdZn;

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
        let n = 1 << self.log2_n;
        let mut current = leq_than - (leq_than % n) + 1;
        while current > 0 && !is_prime(StaticRing::<i64>::RING, &current, 10) {
            current -= n;
        }
        if current <= 0 {
            return None;
        } else {
            return Some(current);
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
