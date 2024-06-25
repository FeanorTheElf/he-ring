use std::alloc::Allocator;
use std::alloc::Global;

use feanor_math::algorithms::fft::*;
use feanor_math::algorithms::fft::complex_fft::FFTErrorEstimate;
use feanor_math::homomorphism::Identity;
use feanor_math::primitive_int::*;
use feanor_math::ring::*;
use feanor_math::rings::float_complex::Complex64Base;
use feanor_math::rings::float_complex::{Complex64, Complex64El};
use feanor_math::rings::zn::zn_64;
use feanor_math::rings::zn::{ZnRing, ZnRingStore};
use feanor_math::rings::extension::*;
use feanor_math::integer::*;

use super::complex_fft_ring::*;
use crate::cyclotomic::*;

const CC: Complex64 = Complex64::RING;

///
/// A [`GeneralizedFFT`] for power-of-two cyclotomic rings, i.e. `Z[X]/(X^(n/2) + 1, q)` for
/// `n` a power of two.
/// 
/// Usually, this will only be used together with [`ComplexFFTBasedRing`]. See that doc for a usage
/// example.
/// 
/// # See also
/// 
/// [`super::odd_cyclotomic::OddCyclotomicFFT`] in the case that the cyclotomic conductor is odd instead
/// of a power of two.
/// 
pub struct Pow2CyclotomicFFT<R, F, A = Global> 
    where R: RingStore,
        R::Type: ZnRing,
        F: FFTAlgorithm<Complex64Base> + FFTErrorEstimate,
        A: Allocator + Clone
{
    fft_table: F,
    ring: R,
    twiddles: Vec<Complex64El>,
    inv_twiddles: Vec<Complex64El>,
    allocator: A
}

impl<R, F, A> Pow2CyclotomicFFT<R, F, A> 
    where R: RingStore,
        R::Type: ZnRing,
        F: FFTAlgorithm<Complex64Base> + FFTErrorEstimate,
        A: Allocator + Clone
{
    pub fn create(ring: R, fft_table: F, allocator: A) -> Self {
        let rank = fft_table.len() as i64;
        let log2_n = StaticRing::<i64>::RING.abs_highest_set_bit(&rank).unwrap();
        assert!(rank == (1 << log2_n));
        let mut twiddles = Vec::with_capacity(rank as usize);
        let mut inv_twiddles = Vec::with_capacity(rank as usize);
        for i in 0..rank {
            twiddles.push(CC.root_of_unity(i, rank * 2));
            inv_twiddles.push(CC.root_of_unity(-i, rank * 2));
        }
        return Self { fft_table, ring, twiddles, inv_twiddles, allocator };
    }
}

impl<R, F, A> RingDecomposition<R::Type> for Pow2CyclotomicFFT<R, F, A> 
    where R: RingStore,
        R::Type: ZnRing,
        F: FFTAlgorithm<Complex64Base> + FFTErrorEstimate,
        A: Allocator + Clone
{
    fn fft_backward(&self, data: &mut [Complex64El], destination: &mut [El<R>], ring: &R::Type) {
        assert!(ring == self.ring.get_ring());
        self.fft_table.unordered_inv_fft(&mut data[..], CC.get_ring());
        for i in 0..self.rank() {
            CC.mul_assign_ref(&mut data[i], &self.twiddles[i]);
            let (re, im) = Complex64::RING.closest_gaussian_int(data[i]);
            debug_assert_eq!(0, im);
            debug_assert!(CC.abs(CC.sub(data[i], CC.from_f64(re as f64))) < 0.01);
            destination[i] = self.ring.coerce(self.ring.integer_ring(), int_cast(re as i64, self.ring.integer_ring(), &StaticRing::<i64>::RING));
        }
    }

    fn fft_forward(&self, data: &[El<R>], destination: &mut [Complex64El], ring: &R::Type) {
        assert!(ring == self.ring.get_ring());
        for i in 0..self.rank() {
            destination[i] = CC.from_f64(int_cast(self.ring.smallest_lift(self.ring.clone_el(&data[i])), StaticRing::<i64>::RING, self.ring.integer_ring()) as f64);
            CC.mul_assign_ref(&mut destination[i], &self.inv_twiddles[i]);
        }
        self.fft_table.unordered_fft(destination, CC.get_ring());
    }

    fn rank(&self) -> usize {
        self.fft_table.len()
    }

    fn mul_assign_fft(&self, lhs: &mut [El<R>], rhs: &[Complex64El], ring: &R::Type) {
        assert!(self.ring.get_ring() == ring);
        let mut tmp = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        tmp.resize(self.rank(), CC.zero());
        mul_assign_fft_base(self, lhs, rhs, &mut tmp, ring);
    }
}

impl<R1, F1, R2, F2, A1, A2> SameNumberRing<R2::Type, R1::Type, Pow2CyclotomicFFT<R1, F1, A1>> for Pow2CyclotomicFFT<R2, F2, A2>
    where R1: RingStore,
        R1::Type: ZnRing,
        F1: FFTAlgorithm<Complex64Base> + FFTErrorEstimate,
        R2: RingStore,
        R2::Type: ZnRing,
        F2: FFTAlgorithm<Complex64Base> + FFTErrorEstimate,
        A1: Allocator + Clone,
        A2: Allocator + Clone
{
    fn is_isomorphic(&self, other: &Pow2CyclotomicFFT<R1, F1, A1>) -> bool {
        self.rank() == other.rank()
    }
}

impl<R, F, A1, A2> CyclotomicRing for CCFFTRingBase<R, Pow2CyclotomicFFT<R, F, A2>, A1>
    where R: RingStore,
        R::Type: ZnRing,
        F: FFTAlgorithm<Complex64Base> + FFTErrorEstimate,
        A1: Allocator + Clone,
        A2: Allocator + Clone
{
    fn n(&self) -> usize {
        2 * self.rank()
    }
}

pub type DefaultPow2CyclotomicCCFFTRingBase<R = zn_64::Zn> = CCFFTRingBase<R, Pow2CyclotomicFFT<R, cooley_tuckey::CooleyTuckeyFFT<Complex64Base, Complex64Base, Identity<Complex64>>>>;
pub type DefaultPow2CyclotomicCCFFTRing<R = zn_64::Zn> = CCFFTRing<R, Pow2CyclotomicFFT<R, cooley_tuckey::CooleyTuckeyFFT<Complex64Base, Complex64Base, Identity<Complex64>>>>;

impl<R, A> CCFFTRingBase<R, Pow2CyclotomicFFT<R, cooley_tuckey::CooleyTuckeyFFT<Complex64Base, Complex64Base, Identity<Complex64>>>, A>
    where R: RingStore + Clone,
        R::Type: ZnRing,
        A: Allocator + Clone + Default
{
    pub fn new(ring: R, log2_ring_degree: usize) -> RingValue<Self> {
        RingValue::from(
            Self::from_generalized_fft(
                ring.clone(),
                Pow2CyclotomicFFT::create(
                    ring, 
                    cooley_tuckey::CooleyTuckeyFFT::for_complex(Complex64::RING, log2_ring_degree),
                    Global
                ),
                A::default()
            )
        )
    }
}

#[cfg(test)]
use feanor_math::rings::extension::generic_test_free_algebra_axioms;
#[cfg(test)]
use feanor_math::rings::zn::zn_64::Zn;

#[test]
fn test_ring_axioms() {
    let Fp = Zn::new(65537);
    let R = DefaultPow2CyclotomicCCFFTRingBase::<>::new(Fp, 3);
    feanor_math::ring::generic_tests::test_ring_axioms(&R, [
        ring_literal!(&R, [0, 0, 0, 0, 0, 0, 0, 0]),
        ring_literal!(&R, [1, 0, 0, 0, 0, 0, 0, 0]),
        ring_literal!(&R, [-1, 0, 0, 0, 0, 0, 0, 0]),
        ring_literal!(&R, [0, 0, 0, 0, 0, 0, 0, 1]),
        ring_literal!(&R, [0, 0, 0, 0, 0, 0, 0, -1]),
        ring_literal!(&R, [1, 1, 1, 1, 1, 1, 1, 1]),
        ring_literal!(&R, [1, -1, 0, 0, 0, 0, 0, 0])
    ].into_iter());
}

#[test]
fn test_free_algebra_axioms() {
    let Fp = Zn::new(65537);
    let R = DefaultPow2CyclotomicCCFFTRingBase::<>::new(Fp, 3);
    generic_test_free_algebra_axioms(R);
}

#[test]
fn test_cyclotomic_ring_axioms() {
    let Fp = Zn::new(65537);
    let R = DefaultPow2CyclotomicCCFFTRingBase::<>::new(Fp, 3);
    generic_test_cyclotomic_ring_axioms(R);

}