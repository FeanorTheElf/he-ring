use feanor_math::algorithms::fft::complex_fft::FFTErrorEstimate;
use feanor_math::algorithms;
use feanor_math::algorithms::fft::*;
use feanor_math::primitive_int::StaticRing;
use feanor_math::ring::*;
use feanor_math::assert_el_eq;
use feanor_math::homomorphism::*;
use feanor_math::integer::*;
use feanor_math::rings::float_complex::Complex64Base;
use feanor_math::rings::zn::zn_64;
use feanor_math::seq::*;
use feanor_math::rings::zn::{ZnRing, ZnRingStore, zn_rns};
use feanor_math::rings::extension::*;

use std::alloc::Allocator;

use crate::complexfft;
use crate::cyclotomic::*;
use super::double_rns_ring::*;

///
/// [`GeneralizedFFT`] corresponding to the evaluation at all primitive `2n`-th roots of unity,
/// for `n` a power of two. 
/// 
/// More concretely, when `p = 1 mod 2n` there is a primitive `2n`-th root of unity `z` in `Fp` and
/// we have the isomorphism
/// ```text
///   Fp[X]/(X^n + 1) -> Fp^n,  f -> (z^i)
/// ```
/// where `i` runs through `Z/2nZ*`. This map and its inverse are stored by this object, and can
/// be used to construct a [`DoubleRNSRing`]. The map is computed using the underlying Fast
/// Fourier-Transform, so usually in time `O(n log(n))`.
/// 
/// Note that it would be possible to merge the multiplication with twiddle factors into the actual
/// FFT, thus saving a few multiplications on each execution. However, for the sake of modularity,
/// we currently don't do this and use the underlying FFT as a black box.
/// 
pub struct Pow2CyclotomicFFT<R, F> 
    where R: RingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: FFTAlgorithm<R::Type>
{
    fft_table: F,
    twiddles: Vec<El<R>>,
    inv_twiddles: Vec<El<R>>,
    ring: R
}

impl<R, F> Pow2CyclotomicFFT<R, F> 
    where R: RingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: FFTAlgorithm<R::Type>
{
    pub fn create(ring: R, fft_table: F, root_of_unity: El<R>) -> Self {
        let rank = fft_table.len() as i64;
        let log2_n = StaticRing::<i64>::RING.abs_log2_floor(&rank).unwrap();
        assert!(rank == (1 << log2_n));
        let mut twiddles = Vec::with_capacity(rank as usize);
        let mut inv_twiddles = Vec::with_capacity(rank as usize);
        assert_el_eq!(&ring, fft_table.root_of_unity(ring.get_ring()), &ring.pow(ring.clone_el(&root_of_unity), 2));

        let mut current = ring.one();
        let mut current_inv = ring.one();
        let zeta_inv = ring.pow(ring.clone_el(&root_of_unity), 2 * rank as usize - 1);
        for _ in 0..rank {
            twiddles.push(ring.clone_el(&current));
            inv_twiddles.push(ring.clone_el(&current_inv));
            ring.mul_assign_ref(&mut current, &root_of_unity);
            ring.mul_assign_ref(&mut current_inv, &zeta_inv);
        }
        return Self { ring, fft_table, twiddles, inv_twiddles };
    }
}

impl<R, F> RingDecomposition<R::Type> for Pow2CyclotomicFFT<R, F> 
    where R: RingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: FFTAlgorithm<R::Type>
{
    fn fft_backward(&self, data: &mut [El<R>], ring: &R::Type) {
        assert!(ring == self.ring.get_ring());
        self.fft_table.unordered_inv_fft(&mut data[..], ring);
        for i in 0..self.rank() {
            ring.mul_assign_ref(&mut data[i], &self.twiddles[i]);
        }
    }

    fn fft_forward(&self, data: &mut [El<R>], ring: &R::Type) {
        assert!(ring == self.ring.get_ring());
        for i in 0..self.rank() {
            ring.mul_assign_ref(&mut data[i], &self.inv_twiddles[i]);
        }
        self.fft_table.unordered_fft(&mut data[..], ring);
    }

    fn rank(&self) -> usize {
        self.fft_table.len()
    }
}

impl<R1, R2, F1, F2> SameNumberRing<R1::Type, R2::Type, Pow2CyclotomicFFT<R2, F2>> for Pow2CyclotomicFFT<R1, F1>
    where R1: RingStore,
        R1::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F1: FFTAlgorithm<R1::Type>,
        R2: RingStore,
        R2::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F2: FFTAlgorithm<R2::Type>
{
    fn is_isomorphic(&self, other: &Pow2CyclotomicFFT<R2, F2>) -> bool {
        self.rank() == other.rank()
    }
}

impl<R1, R2, F1, F2> SameNumberRingCross<R2::Type, R1::Type, complexfft::pow2_cyclotomic::Pow2CyclotomicFFT<R1, F1>> for Pow2CyclotomicFFT<R2, F2>
    where R1: RingStore,
        F1: FFTAlgorithm<Complex64Base> + FFTErrorEstimate,
        R1::Type: ZnRing,
        R2: RingStore,
        R2::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F2: FFTAlgorithm<R2::Type>
{
    fn is_isomorphic(&self, other: &complexfft::pow2_cyclotomic::Pow2CyclotomicFFT<R1, F1>) -> bool {
        self.rank() == <_ as complexfft::complex_fft_ring::RingDecomposition<_>>::rank(other)
    }
}

impl<R, F, A> CyclotomicRing for DoubleRNSRingBase<R, Pow2CyclotomicFFT<R, F>, A>
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: FFTAlgorithm<R::Type>,
        A: Allocator + Clone
{
    fn n(&self) -> usize {
        2 * self.rank()
    }
}

impl<R_main, R_twiddle, A> DoubleRNSRingBase<R_main, Pow2CyclotomicFFT<R_main, cooley_tuckey::CooleyTuckeyFFT<R_main::Type, R_twiddle::Type, CanHom<R_twiddle, R_main>>>, A>
    where R_main: RingStore + Clone,
        R_twiddle: RingStore,
        R_main::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<R_twiddle::Type>,
        R_twiddle::Type: ZnRing,
        A: Allocator + Default + Clone
{
    pub fn new(base_ring: zn_rns::Zn<R_main, BigIntRing>, fft_rings: Vec<R_twiddle>, log2_n: usize) -> RingValue<Self> {
        Self::new_with(base_ring, fft_rings, log2_n, A::default())
    }
}

impl<R_main, R_twiddle, A> DoubleRNSRingBase<R_main, Pow2CyclotomicFFT<R_main, cooley_tuckey::CooleyTuckeyFFT<R_main::Type, R_twiddle::Type, CanHom<R_twiddle, R_main>>>, A>
    where R_main: RingStore + Clone,
        R_twiddle: RingStore,
        R_main::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<R_twiddle::Type>,
        R_twiddle::Type: ZnRing,
        A: Allocator + Clone
{
    pub fn new_with(base_ring: zn_rns::Zn<R_main, BigIntRing>, fft_rings: Vec<R_twiddle>, log2_n: usize, allocator: A) -> RingValue<Self> {
        assert_eq!(base_ring.len(), fft_rings.len());
        let ffts = fft_rings.into_iter().enumerate().map(|(i, R)| {
            let R_as_field = (&R).as_field().ok().unwrap();
            let root_of_unity = R_as_field.get_ring().unwrap_element(algorithms::unity_root::get_prim_root_of_unity_pow2(&R_as_field, log2_n + 1).unwrap());
            let fft_table_root_of_unity = R.pow(R.clone_el(&root_of_unity), 2);
            let hom = base_ring.at(i).clone().into_can_hom(R).ok().unwrap();
            let root_of_unity = hom.map(root_of_unity);
            Pow2CyclotomicFFT::create(
                hom.codomain().clone(),
                cooley_tuckey::CooleyTuckeyFFT::new_with_hom(hom, fft_table_root_of_unity, log2_n),
                root_of_unity
            )
        }).collect();
        RingValue::from(Self::from_generalized_ffts(
            base_ring,
            ffts,
            allocator
        ))
    }
}

pub type DefaultPow2CyclotomicDoubleRNSRingBase<R = zn_64::Zn> = DoubleRNSRingBase<R, Pow2CyclotomicFFT<R, cooley_tuckey::CooleyTuckeyFFT<<R as RingStore>::Type, <R as RingStore>::Type, Identity<R>>>>;
pub type DefaultPow2CyclotomicDoubleRNSRing<R = zn_64::Zn> = DoubleRNSRing<R, Pow2CyclotomicFFT<R, cooley_tuckey::CooleyTuckeyFFT<<R as RingStore>::Type, <R as RingStore>::Type, Identity<R>>>>;

impl<R, A> DoubleRNSRingBase<R, Pow2CyclotomicFFT<R, cooley_tuckey::CooleyTuckeyFFT<R::Type, R::Type, Identity<R>>>, A>
    where R: RingStore + Clone,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Default + Clone
{
    pub fn new(base_ring: zn_rns::Zn<R, BigIntRing>, log2_n: usize) -> RingValue<Self> {
        Self::new_with(base_ring, log2_n, A::default())
    }
}

impl<R, A> DoubleRNSRingBase<R, Pow2CyclotomicFFT<R, cooley_tuckey::CooleyTuckeyFFT<R::Type, R::Type, Identity<R>>>, A>
    where R: RingStore + Clone,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    pub fn new_with(base_ring: zn_rns::Zn<R, BigIntRing>, log2_n: usize, allocator: A) -> RingValue<Self> {
        let ffts = base_ring.as_iter().enumerate().map(|(_, R)| {
            let R_as_field = R.as_field().ok().unwrap();
            let root_of_unity = R_as_field.get_ring().unwrap_element(algorithms::unity_root::get_prim_root_of_unity_pow2(&R_as_field, log2_n + 1).unwrap());
            let fft_table_root_of_unity = R.pow(R.clone_el(&root_of_unity), 2);
            Pow2CyclotomicFFT::create(
                R.clone(),
                cooley_tuckey::CooleyTuckeyFFT::new(R.clone() as R, fft_table_root_of_unity, log2_n),
                root_of_unity
            )
        }).collect();
        RingValue::from(Self::from_generalized_ffts(
            base_ring,
            ffts,
            allocator
        ))
    }
}

#[cfg(test)]
use feanor_math::rings::extension::generic_test_free_algebra_axioms;
#[cfg(test)]
use feanor_math::rings::zn::zn_64::Zn;

#[cfg(test)]
fn edge_case_elements<'a, R, F, A>(R: &'a DoubleRNSRing<R, F, A>) -> impl 'a + Iterator<Item = El<DoubleRNSRing<R, F, A>>>
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
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
    let rns_base = zn_rns::Zn::new(vec![Zn::new(17), Zn::new(97)], BigIntRing::RING);
    let R = DefaultPow2CyclotomicDoubleRNSRingBase::new(rns_base, 3);
    feanor_math::ring::generic_tests::test_ring_axioms(&R, edge_case_elements(&R));
}

#[test]
fn test_divisibility_axioms() {
    let rns_base = zn_rns::Zn::new(vec![Zn::new(17), Zn::new(97)], BigIntRing::RING);
    let R = DefaultPow2CyclotomicDoubleRNSRingBase::new(rns_base, 3);
    feanor_math::divisibility::generic_tests::test_divisibility_axioms(&R, edge_case_elements(&R));
}

#[test]
fn test_free_algebra_axioms() {
    let rns_base = zn_rns::Zn::new(vec![Zn::new(17), Zn::new(97)], BigIntRing::RING);
    let R = DefaultPow2CyclotomicDoubleRNSRingBase::new(rns_base, 3);
    generic_test_free_algebra_axioms(R);
}

#[test]
fn test_cyclotomic_ring_axioms() {
    let rns_base = zn_rns::Zn::new(vec![Zn::new(17), Zn::new(97)], BigIntRing::RING);
    let R = DefaultPow2CyclotomicDoubleRNSRingBase::new(rns_base, 3);
    generic_test_cyclotomic_ring_axioms(R);
}