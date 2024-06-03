use feanor_math::algorithms::fft::complex_fft::ErrorEstimate;
use feanor_math::algorithms;
use feanor_math::algorithms::fft::*;
use feanor_math::mempool::*;
use feanor_math::primitive_int::StaticRing;
use feanor_math::ring::*;
use feanor_math::assert_el_eq;
use feanor_math::homomorphism::*;
use feanor_math::integer::*;
use feanor_math::rings::float_complex::Complex64;
use feanor_math::rings::zn::{ZnRing, ZnRingStore, zn_rns};
use feanor_math::rings::extension::*;

use crate::complexfft;
use crate::cyclotomic::*;
use super::double_rns_ring::*;

pub struct Pow2CyclotomicFFT<F> 
    where F: FFTTable,
        F::Ring: Sized + ZnRingStore,
        <F::Ring as RingStore>::Type: ZnRing + CanHomFrom<BigIntRingBase>
{
    fft_table: F,
    twiddles: Vec<El<F::Ring>>,
    inv_twiddles: Vec<El<F::Ring>>
}

impl<F> Pow2CyclotomicFFT<F> 
    where F: FFTTable,
        F::Ring: Sized + ZnRingStore,
        <F::Ring as RingStore>::Type: ZnRing + CanHomFrom<BigIntRingBase>
{
    pub fn create(fft_table: F, root_of_unity: El<F::Ring>) -> Self {
        let rank = fft_table.len() as i64;
        let log2_n = StaticRing::<i64>::RING.abs_highest_set_bit(&rank).unwrap();
        assert!(rank == (1 << log2_n));
        let mut twiddles = Vec::with_capacity(rank as usize);
        let mut inv_twiddles = Vec::with_capacity(rank as usize);
        let ring = fft_table.ring();
        assert_el_eq!(ring, fft_table.root_of_unity(), &ring.pow(ring.clone_el(&root_of_unity), 2));

        let mut current = ring.one();
        let mut current_inv = ring.one();
        let zeta_inv = ring.pow(ring.clone_el(&root_of_unity), 2 * rank as usize - 1);
        for _ in 0..rank {
            twiddles.push(ring.clone_el(&current));
            inv_twiddles.push(ring.clone_el(&current_inv));
            ring.mul_assign_ref(&mut current, &root_of_unity);
            ring.mul_assign_ref(&mut current_inv, &zeta_inv);
        }
        return Self { fft_table, twiddles, inv_twiddles };
    }
}

impl<F> GeneralizedFFT for Pow2CyclotomicFFT<F> 
    where F: FFTTable,
        F::Ring: Sized + ZnRingStore,
        <F::Ring as RingStore>::Type: ZnRing + CanHomFrom<BigIntRingBase>
{
    type BaseRingBase = <F::Ring as RingStore>::Type;
    type BaseRingStore = F::Ring;

    fn base_ring(&self) -> &Self::BaseRingStore {
        self.fft_table.ring()
    }

    fn fft_backward<S, M>(&self, data: &mut [El<S>], ring: &S, memory_provider: &M)
        where S: ZnRingStore,
            S::Type: ZnRing + CanHomFrom<Self::BaseRingBase>,
            M: MemoryProvider<El<S>>
    {
        self.fft_table.unordered_inv_fft(&mut data[..], memory_provider, &ring.can_hom(self.fft_table.ring()).unwrap());
        let hom = ring.can_hom(self.base_ring()).unwrap();
        for i in 0..self.rank() {
            ring.get_ring().mul_assign_map_in_ref(self.base_ring().get_ring(), &mut data[i], &self.twiddles[i], hom.raw_hom());
        }
    }

    fn fft_forward<S, M>(&self, data: &mut [El<S>], ring: &S, memory_provider: &M)
        where S: ZnRingStore,
            S::Type: ZnRing + CanHomFrom<Self::BaseRingBase>,
            M: MemoryProvider<El<S>> 
    {
        let hom = ring.can_hom(self.base_ring()).unwrap();
        for i in 0..self.rank() {
            ring.get_ring().mul_assign_map_in_ref(self.base_ring().get_ring(), &mut data[i], &self.inv_twiddles[i], hom.raw_hom());
        }
        self.fft_table.unordered_fft(&mut data[..], memory_provider, &ring.can_hom(self.fft_table.ring()).unwrap());
    }

    fn rank(&self) -> usize {
        self.fft_table.len()
    }
}

impl<F1, F2> GeneralizedFFTIso<Pow2CyclotomicFFT<F1>> for Pow2CyclotomicFFT<F2>
    where F1: FFTTable,
        F1::Ring: Sized + ZnRingStore,
        <F1::Ring as RingStore>::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F2: FFTTable,
        F2::Ring: Sized + ZnRingStore,
        <F2::Ring as RingStore>::Type: ZnRing + CanHomFrom<BigIntRingBase>
{
    fn is_isomorphic(&self, other: &Pow2CyclotomicFFT<F1>) -> bool {
        self.rank() == other.rank()
    }
}

impl<R1, F1, F2> GeneralizedFFTCrossIso<complexfft::pow2_cyclotomic::Pow2CyclotomicFFT<R1, F1>> for Pow2CyclotomicFFT<F2>
    where R1: RingStore,
        F1: FFTTable<Ring = Complex64> + ErrorEstimate,
        R1::Type: ZnRing,
        F2: FFTTable,
        F2::Ring: Sized + ZnRingStore,
        <F2::Ring as RingStore>::Type: ZnRing + CanHomFrom<BigIntRingBase>
{
    fn is_isomorphic(&self, other: &complexfft::pow2_cyclotomic::Pow2CyclotomicFFT<R1, F1>) -> bool {
        self.rank() == <_ as complexfft::complex_fft_ring::GeneralizedFFT>::rank(other)
    }
}

impl<R, F, M> CyclotomicRing for DoubleRNSRingBase<R, Pow2CyclotomicFFT<F>, M>
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanIsoFromTo<<F::Ring as RingStore>::Type>,
        F: FFTTable,
        F::Ring: Sized + ZnRingStore,
        <F::Ring as RingStore>::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        M: MemoryProvider<El<R>>
{
    fn n(&self) -> usize {
        2 * self.rank()
    }
}

impl<R_main, R_fft, M> DoubleRNSRingBase<R_main, Pow2CyclotomicFFT<cooley_tuckey::FFTTableCooleyTuckey<R_fft>>, M>
    where R_main: ZnRingStore,
        R_fft: ZnRingStore,
        R_fft::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        R_main::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanIsoFromTo<R_fft::Type> + CanHomFrom<BigIntRingBase>,
        M: MemoryProvider<El<R_main>>
{
    pub fn new(base_ring: zn_rns::Zn<R_main, BigIntRing>, fft_rings: Vec<R_fft>, log2_n: usize, memory_provider: M) -> RingValue<Self> {
        let ffts = fft_rings.into_iter().map(|R| {
            let root_of_unity = algorithms::unity_root::get_prim_root_of_unity_pow2(&R, log2_n + 1).unwrap();
            let fft_table_root_of_unity = R.pow(R.clone_el(&root_of_unity), 2);
            Pow2CyclotomicFFT::create(
                cooley_tuckey::FFTTableCooleyTuckey::new(R, fft_table_root_of_unity, log2_n),
                root_of_unity
            )
        }).collect();
        RingValue::from(Self::from_generalized_ffts(
            base_ring,
            ffts, 
            memory_provider
        ))
    }
}

#[cfg(test)]
use feanor_math::rings::extension::generic_test_free_algebra_axioms;
#[cfg(test)]
use feanor_math::rings::zn::zn_64::Zn;
#[cfg(test)]
use feanor_math::vector::*;
#[cfg(test)]
use feanor_math::default_memory_provider;

#[cfg(test)]
fn edge_case_elements<'a, R, F, M>(R: &'a DoubleRNSRing<R, F, M>) -> impl 'a + Iterator<Item = El<DoubleRNSRing<R, F, M>>>
    where R: ZnRingStore,
        R::Type: ZnRing + CanIsoFromTo<F::BaseRingBase> + CanHomFrom<BigIntRingBase>,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    assert_eq!(2, R.get_ring().rns_base().len());
    assert_eq!(17, int_cast(R.get_ring().rns_base().at(0).integer_ring().clone_el(R.get_ring().rns_base().at(0).modulus()), StaticRing::<i64>::RING, R.get_ring().rns_base().at(0).integer_ring()));
    assert_eq!(8, R.rank());
    [
        ring_literal!(&R, [0, 0, 0, 0, 0, 0, 0, 0]),
        ring_literal!(&R, [1, 0, 0, 0, 0, 0, 0, 0]),
        ring_literal!(&R, [-1, 0, 0, 0, 0, 0, 0, 0]),
        ring_literal!(&R, [0, 0, 0, 0, 0, 0, 0, 1]),
        ring_literal!(&R, [0, 0, 0, 0, 0, 0, 0, -1]),
        ring_literal!(&R, [1, 1, 1, 1, 1, 1, 1, 1]),
        ring_literal!(&R, [1, -1, 0, 0, 0, 0, 0, 0]),
        // these elements are non-invertible, but in the same prime ideal `(X + 3)`
        ring_literal!(&R, [15, 8, 1, 0, 0, 0, 0, 0]),
        ring_literal!(&R, [3, 1, 0, 0, 0, 0, 0, 0]),
        ring_literal!(&R, [0, 15, 8, 1, 0, 0, 0, 0])
    ].into_iter()
}

#[test]
fn test_ring_axioms() {
    let rns_base = zn_rns::Zn::new(vec![Zn::new(17), Zn::new(97)], BigIntRing::RING, default_memory_provider!());
    let fft_rings = rns_base.get_ring().iter().cloned().collect();
    let R = DoubleRNSRingBase::<_, Pow2CyclotomicFFT<cooley_tuckey::FFTTableCooleyTuckey<_>>, _>::new(rns_base, fft_rings, 3, default_memory_provider!());
    feanor_math::ring::generic_tests::test_ring_axioms(&R, edge_case_elements(&R));
}

#[test]
fn test_divisibility_axioms() {
    let rns_base = zn_rns::Zn::new(vec![Zn::new(17), Zn::new(97)], BigIntRing::RING, default_memory_provider!());
    let fft_rings = rns_base.get_ring().iter().cloned().collect();
    let R = DoubleRNSRingBase::<_, Pow2CyclotomicFFT<cooley_tuckey::FFTTableCooleyTuckey<_>>, _>::new(rns_base, fft_rings, 3, default_memory_provider!());
    feanor_math::divisibility::generic_tests::test_divisibility_axioms(&R, edge_case_elements(&R));
}

#[test]
fn test_free_algebra_axioms() {
    let rns_base = zn_rns::Zn::new(vec![Zn::new(17), Zn::new(97)], BigIntRing::RING, default_memory_provider!());
    let fft_rings = rns_base.get_ring().iter().cloned().collect();
    let R = DoubleRNSRingBase::<_, Pow2CyclotomicFFT<cooley_tuckey::FFTTableCooleyTuckey<_>>, _>::new(rns_base, fft_rings, 3, default_memory_provider!());
    generic_test_free_algebra_axioms(R);
}

#[test]
fn test_cyclotomic_ring_axioms() {
    let rns_base = zn_rns::Zn::new(vec![Zn::new(17), Zn::new(97)], BigIntRing::RING, default_memory_provider!());
    let fft_rings = rns_base.get_ring().iter().cloned().collect();
    let R = DoubleRNSRingBase::<_, Pow2CyclotomicFFT<cooley_tuckey::FFTTableCooleyTuckey<_>>, _>::new(rns_base, fft_rings, 3, default_memory_provider!());
    generic_test_cyclotomic_ring_axioms(R);

}