use cooley_tuckey::bitreverse;
use feanor_math::algorithms;
use feanor_math::algorithms::fft::*;
use feanor_math::algorithms::unity_root::get_prim_root_of_unity_pow2;
use feanor_math::primitive_int::StaticRing;
use feanor_math::ring::*;
use feanor_math::assert_el_eq;
use feanor_math::homomorphism::*;
use feanor_math::integer::*;
use feanor_math::rings::zn::zn_64;
use feanor_math::rings::zn::zn_64::ZnEl;
use feanor_math::seq::*;
use feanor_math::rings::zn::{ZnRing, ZnRingStore, zn_rns};
use feanor_math::rings::zn::zn_64::Zn;

use std::alloc::Allocator;
use std::alloc::Global;

use super::double_rns_ring::*;
use super::ntt_ring::NTTRing;
use super::ntt_ring::NTTRingBase;
use crate::rings::decomposition::*;
use crate::sample_primes;
use crate::StdZn;

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
        R::Type: ZnRing,
        F: FFTAlgorithm<R::Type> + PartialEq
{
    fft_table: F,
    twiddles: Vec<El<R>>,
    inv_twiddles: Vec<El<R>>,
    ring: R
}

impl<R, F> Pow2CyclotomicFFT<R, F> 
    where R: RingStore,
        R::Type: ZnRing,
        F: FFTAlgorithm<R::Type> + PartialEq
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
        R::Type: ZnRing,
        F: FFTAlgorithm<R::Type> + PartialEq
{
    fn expansion_factor(&self) -> i64 {
        self.rank() as i64
    }

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

impl<R, F> PartialEq for Pow2CyclotomicFFT<R, F> 
    where R: RingStore,
        R::Type: ZnRing,
        F: FFTAlgorithm<R::Type> + PartialEq
{
    fn eq(&self, other: &Self) -> bool {
        self.ring.get_ring() == other.ring.get_ring() && self.fft_table == other.fft_table && self.ring.eq_el(&self.twiddles[1], &other.twiddles[1])
    }
}

impl<R1, R2, F1, F2> IsomorphismInfo<R1::Type, R2::Type, Pow2CyclotomicFFT<R2, F2>> for Pow2CyclotomicFFT<R1, F1>
    where R1: RingStore,
        R1::Type: ZnRing,
        F1: FFTAlgorithm<R1::Type> + PartialEq,
        R2: RingStore,
        R2::Type: ZnRing + PartialEq<R1::Type>,
        F2: FFTAlgorithm<R2::Type> + PartialEq
{
    fn is_same_number_ring(&self, other: &Pow2CyclotomicFFT<R2, F2>) -> bool {
        self.rank() == other.rank()
    }
}

impl<R, F> CyclotomicRingDecomposition<R::Type> for Pow2CyclotomicFFT<R, F> 
    where R: RingStore,
        R::Type: ZnRing,
        F: FFTAlgorithm<R::Type> + PartialEq
{
    fn permute_galois_action<S>(&self, src: &[<R::Type as RingBase>::Element], dst: &mut [<R::Type as RingBase>::Element], galois_element: ZnEl, ring: S)
        where S: RingStore<Type = R::Type>
    {
        assert_eq!(self.rank(), src.len());
        assert_eq!(self.rank(), dst.len());
        let Gal = self.galois_group_mulrepr();
        let hom = Gal.can_hom(&StaticRing::<i64>::RING).unwrap();
        let bitlength = StaticRing::<i64>::RING.abs_log2_ceil(&(self.rank() as i64)).unwrap();
        debug_assert_eq!(1 << bitlength, self.rank());

        // the elements of src resp. dst follow an order derived from the bitreversing order of the underlying FFT
        let index_to_galois_el = |i: usize| hom.map(2 * bitreverse(i, bitlength) as i64 + 1);
        let galois_el_to_index = |s: ZnEl| bitreverse((Gal.smallest_positive_lift(s) as usize - 1) / 2, bitlength);

        for i in 0..self.rank() {
            dst[i] = ring.clone_el(&src[galois_el_to_index(Gal.mul(galois_element, index_to_galois_el(i)))]);
        }
    }

    fn galois_group_mulrepr(&self) -> Zn {
        Zn::new(self.rank() as u64 * 2)
    }
}

impl<R_main, R_twiddle, A> DoubleRNSRingBase<R_main, Pow2CyclotomicFFT<R_main, cooley_tuckey::CooleyTuckeyFFT<R_main::Type, R_twiddle::Type, CanHom<R_twiddle, R_main>>>, A>
    where R_main: RingStore + Clone,
        R_twiddle: RingStore,
        R_main::Type: ZnRing + CanHomFrom<R_twiddle::Type> + CanHomFrom<BigIntRingBase>,
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
        R_main::Type: ZnRing + CanHomFrom<R_twiddle::Type> + CanHomFrom<BigIntRingBase>,
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
        RingValue::from(Self::from_ring_decompositions(
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
        RingValue::from(Self::from_ring_decompositions(
            base_ring,
            ffts,
            allocator
        ))
    }
}

pub type DefaultPow2CyclotomicNTTRingBase<R = zn_64::Zn> = NTTRingBase<R, Pow2CyclotomicFFT<R, cooley_tuckey::CooleyTuckeyFFT<<R as RingStore>::Type, <R as RingStore>::Type, Identity<R>>>>;
pub type DefaultPow2CyclotomicNTTRing<R = zn_64::Zn> = NTTRing<R, Pow2CyclotomicFFT<R, cooley_tuckey::CooleyTuckeyFFT<<R as RingStore>::Type, <R as RingStore>::Type, Identity<R>>>>;

impl<R, A> NTTRingBase<RingValue<R>, Pow2CyclotomicFFT<RingValue<R>, cooley_tuckey::CooleyTuckeyFFT<R, R, Identity<RingValue<R>>>>, A>
    where R: StdZn + Clone,
        A: Allocator + Default + Clone
{
    pub fn new_with(base_ring: RingValue<R>, log2_n: usize, allocator: A) -> RingValue<Self> {
        let modulus = int_cast(base_ring.integer_ring().clone_el(base_ring.modulus()), StaticRing::<i64>::RING, base_ring.integer_ring());
        let required_bits = ((modulus as f64).log2() * 2. + log2_n as f64).ceil() as usize;
        let primes = sample_primes(required_bits, required_bits + 4, 58, &BigIntRing::RING.power_of_two(log2_n + 1)).unwrap();

        let mut rns_base = Vec::new();
        let mut ring_decompositions = Vec::new();
        for p in primes {
            let Fp = RingValue::from(R::create(|int_ring| Ok(int_cast(p, RingRef::new(int_ring), &BigIntRing::RING))).unwrap_or_else(|x| x));
            let as_field = RingRef::new(Fp.get_ring()).as_field().ok().unwrap();
            let root_of_unity = Fp.coerce(&as_field, get_prim_root_of_unity_pow2(as_field, log2_n + 1).unwrap());
            let fft_table = cooley_tuckey::CooleyTuckeyFFT::new(Fp.clone(), Fp.pow(Fp.clone_el(&root_of_unity), 2), log2_n);
            ring_decompositions.push(Pow2CyclotomicFFT::create(Fp.clone(), fft_table, root_of_unity));
            rns_base.push(Fp);
        }
        return RingValue::from(NTTRingBase::from_ring_decompositions(
            base_ring, 
            zn_rns::Zn::new(rns_base, BigIntRing::RING), 
            ring_decompositions, 
            allocator
        ));
    }
}

impl DefaultPow2CyclotomicNTTRingBase {

    pub fn new(base_ring: zn_64::Zn, log2_n: usize) -> RingValue<Self> {
        Self::new_with(base_ring, log2_n, Global)
    }
}

#[cfg(test)]
use feanor_math::rings::extension::*;
#[cfg(test)]
use crate::cyclotomic::*;

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
    feanor_math::rings::extension::generic_tests::test_free_algebra_axioms(R);
}

#[test]
fn test_cyclotomic_ring_axioms() {
    let rns_base = zn_rns::Zn::new(vec![Zn::new(17), Zn::new(97)], BigIntRing::RING);
    let R = DefaultPow2CyclotomicDoubleRNSRingBase::new(rns_base, 3);
    feanor_math::rings::extension::generic_tests::test_free_algebra_axioms(R);
}

#[test]
fn test_permute_galois_automorphism() {
    let rns_base = zn_rns::Zn::new(vec![Zn::new(17), Zn::new(97)], BigIntRing::RING);
    let R = DefaultPow2CyclotomicDoubleRNSRingBase::new(rns_base, 3);
    assert_el_eq!(R, R.pow(R.canonical_gen(), 3), R.get_ring().apply_galois_action(&R.canonical_gen(), R.get_ring().galois_group_mulrepr().int_hom().map(3)));
    assert_el_eq!(R, R.pow(R.canonical_gen(), 6), R.get_ring().apply_galois_action(&R.pow(R.canonical_gen(), 2), R.get_ring().galois_group_mulrepr().int_hom().map(3)));
}