use feanor_math::algorithms::fft::*;
use feanor_math::algorithms::fft::complex_fft::ErrorEstimate;
use feanor_math::mempool::*;
use feanor_math::primitive_int::*;
use feanor_math::ring::*;
use feanor_math::rings::float_complex::{Complex64, Complex64El};
use feanor_math::rings::zn::{ZnRing, ZnRingStore};
use feanor_math::rings::extension::*;
use feanor_math::integer::*;

use super::complex_fft_ring::{GeneralizedFFT, ComplexFFTBasedRingBase, GeneralizedFFTIso};
use crate::cyclotomic::*;

const CC: Complex64 = Complex64::RING;

pub struct Pow2CyclotomicFFT<R: ZnRingStore, F: FFTTable<Ring = Complex64> + ErrorEstimate> 
    where R::Type: ZnRing
{
    fft_table: F,
    base_ring: R,
    twiddles: Vec<Complex64El>,
    inv_twiddles: Vec<Complex64El>,
}

impl<R: ZnRingStore, F: FFTTable<Ring = Complex64> + ErrorEstimate> Pow2CyclotomicFFT<R, F> 
    where R::Type: ZnRing
{
    pub fn create(base_ring: R, fft_table: F) -> Self {
        let rank = fft_table.len() as i64;
        let log2_n = StaticRing::<i64>::RING.abs_highest_set_bit(&rank).unwrap();
        assert!(rank == (1 << log2_n));
        let mut twiddles = Vec::with_capacity(rank as usize);
        let mut inv_twiddles = Vec::with_capacity(rank as usize);
        for i in 0..rank {
            twiddles.push(CC.root_of_unity(i, rank * 2));
            inv_twiddles.push(CC.root_of_unity(-i, rank * 2));
        }
        return Self { fft_table, base_ring, twiddles, inv_twiddles };
    }
}

impl<R: ZnRingStore, F: FFTTable<Ring = Complex64> + ErrorEstimate> GeneralizedFFT for Pow2CyclotomicFFT<R, F> 
    where R::Type: ZnRing
{
    type BaseRingBase = R::Type;
    type BaseRingStore = R;

    fn base_ring(&self) -> &Self::BaseRingStore {
        &self.base_ring
    }

    fn fft_backward<M_Zn: MemoryProvider<El<Self::BaseRingStore>>, M_CC: MemoryProvider<Complex64El>>(&self, data: &mut [Complex64El], destination: &mut [El<Self::BaseRingStore>], _: &M_Zn, memory_provider_cc: &M_CC) {
        self.fft_table.unordered_inv_fft(&mut data[..], memory_provider_cc, &Complex64::RING.identity());
        for i in 0..self.rank() {
            CC.mul_assign_ref(&mut data[i], &self.twiddles[i]);
            let (re, im) = Complex64::RING.closest_gaussian_int(data[i]);
            debug_assert_eq!(0, im);
            debug_assert!(CC.abs(CC.sub(data[i], CC.from_f64(re as f64))) < 0.01);
            destination[i] = self.base_ring().coerce(self.base_ring().integer_ring(), int_cast(re as i64, self.base_ring().integer_ring(), &StaticRing::<i64>::RING));
        }
    }

    fn fft_forward<M: MemoryProvider<Complex64El>>(&self, data: &[El<Self::BaseRingStore>], destination: &mut [Complex64El], memory_provider: &M) {
        for i in 0..self.rank() {
            destination[i] = Complex64::RING.from_f64(int_cast(self.base_ring().smallest_lift(self.base_ring().clone_el(&data[i])), StaticRing::<i64>::RING, self.base_ring().integer_ring()) as f64);
            CC.mul_assign_ref(&mut destination[i], &self.inv_twiddles[i]);
        }
        self.fft_table.unordered_fft(destination, memory_provider, &Complex64::RING.identity());
    }

    fn rank(&self) -> usize {
        self.fft_table.len()
    }
}

impl<R1: ZnRingStore, F1, R2: ZnRingStore, F2> GeneralizedFFTIso<Pow2CyclotomicFFT<R1, F1>> for Pow2CyclotomicFFT<R2, F2>
    where R1::Type: ZnRing, F1: FFTTable<Ring = Complex64> + ErrorEstimate,
        R2::Type: ZnRing, F2: FFTTable<Ring = Complex64> + ErrorEstimate
{
    fn is_isomorphic(&self, other: &Pow2CyclotomicFFT<R1, F1>) -> bool {
        self.rank() == other.rank()
    }
}

impl<R: ZnRingStore, F: FFTTable<Ring = Complex64> + ErrorEstimate, M_Zn, M_CC> CyclotomicRing for ComplexFFTBasedRingBase<Pow2CyclotomicFFT<R, F>, M_Zn, M_CC>
    where R::Type: ZnRing,
        M_Zn: MemoryProvider<El<R>>,
        M_CC: MemoryProvider<Complex64El>
{
    fn n(&self) -> usize {
        2 * self.rank()
    }
}

impl<R: ZnRingStore, M_Zn, M_CC> ComplexFFTBasedRingBase<Pow2CyclotomicFFT<R, cooley_tuckey::FFTTableCooleyTuckey<Complex64>>, M_Zn, M_CC>
    where R::Type: ZnRing,
        M_Zn: MemoryProvider<El<R>>,
        M_CC: MemoryProvider<Complex64El>
{
    pub fn new(base_ring: R, log2_n: usize, memory_provider_zn: M_Zn, memory_provider_cc: M_CC) -> RingValue<Self> {
        RingValue::from(
            Self::from_generalized_fft(
                Pow2CyclotomicFFT::create(
                    base_ring, 
                    cooley_tuckey::FFTTableCooleyTuckey::for_complex(Complex64::RING, log2_n)
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
use feanor_math::rings::zn::zn_42::Zn;
#[cfg(test)]
use feanor_math::default_memory_provider;

#[test]
fn test_ring_axioms() {
    let Fp = Zn::new(65537);
    let R = ComplexFFTBasedRingBase::<Pow2CyclotomicFFT<_, cooley_tuckey::FFTTableCooleyTuckey<Complex64>>, _, _>::new(Fp, 3, default_memory_provider!(), default_memory_provider!());
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
    let R = ComplexFFTBasedRingBase::<Pow2CyclotomicFFT<_, cooley_tuckey::FFTTableCooleyTuckey<Complex64>>, _, _>::new(Fp, 3, default_memory_provider!(), default_memory_provider!());
    generic_test_free_algebra_axioms(R);
}

#[test]
fn test_cyclotomic_ring_axioms() {
    let Fp = Zn::new(65537);
    let R = ComplexFFTBasedRingBase::<Pow2CyclotomicFFT<_, cooley_tuckey::FFTTableCooleyTuckey<Complex64>>, _, _>::new(Fp, 3, default_memory_provider!(), default_memory_provider!());
    generic_test_cyclotomic_ring_axioms(R);

}