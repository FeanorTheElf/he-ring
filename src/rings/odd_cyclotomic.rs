use std::alloc::{Allocator, Global};
use std::collections::HashMap;

use feanor_math::algorithms::int_factor::factor;
use feanor_math::algorithms::unity_root::{get_prim_root_of_unity, get_prim_root_of_unity_pow2};
use feanor_math::algorithms::{fft::*, self};
use feanor_math::integer::IntegerRingStore;
use feanor_math::integer::*;
use feanor_math::rings::poly::*;
use feanor_math::divisibility::DivisibilityRing;
use feanor_math::primitive_int::*;
use feanor_math::ring::*;
use feanor_math::homomorphism::*;
use feanor_math::rings::poly::sparse_poly::SparsePolyRing;
use feanor_math::rings::zn::*;
use feanor_math::seq::*;

use crate::rings::double_rns_ring::*;
use crate::rings::decomposition::*;
use super::ntt_ring::{NTTRing, NTTRingBase};
use crate::{euler_phi, sample_primes, StdZn};

const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;

pub struct OddCyclotomicFFT<R, F, A = Global> 
    where R: RingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + DivisibilityRing,
        F: FFTAlgorithm<R::Type> + PartialEq,
        A: Allocator + Clone
{
    ring: R,
    fft_table: F,
    /// contains `usize::MAX` whenenver the fft output index corresponds to a non-primitive root of unity, and an index otherwise
    fft_output_indices_to_indices: Vec<usize>,
    zeta_pow_rank: HashMap<usize, El<R>>,
    rank: usize,
    allocator: A
}

impl<R, F, A> OddCyclotomicFFT<R, F, A> 
    where R: RingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + DivisibilityRing,
        F: FFTAlgorithm<R::Type> + PartialEq,
        A: Allocator + Clone
{
    pub fn create(ring: R, fft_table: F, allocator: A) -> Self {
        let n_factorization = algorithms::int_factor::factor(&ZZ, fft_table.len() as i64);
        let rank = euler_phi(&n_factorization) as usize;

        let poly_ring = SparsePolyRing::new(&ring, "X");
        let cyclotomic_poly = algorithms::cyclotomic::cyclotomic_polynomial(&poly_ring, fft_table.len());
        assert_eq!(poly_ring.degree(&cyclotomic_poly).unwrap(), rank);
        let mut zeta_pow_rank = HashMap::new();
        for (a, i) in poly_ring.terms(&cyclotomic_poly) {
            if i != rank {
                zeta_pow_rank.insert(i, ring.negate(ring.clone_el(a)));
            }
        }

        let fft_output_indices_to_indices = (0..fft_table.len()).scan(0, |state, i| {
            let power = fft_table.unordered_fft_permutation(i);
            if n_factorization.iter().all(|(p, _)| power as i64 % *p != 0) {
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
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + DivisibilityRing,
        F: FFTAlgorithm<R::Type> + PartialEq,
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

impl<R, F, A> PartialEq for OddCyclotomicFFT<R, F, A> 
    where R: RingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + DivisibilityRing,
        F: FFTAlgorithm<R::Type> + PartialEq,
        A: Allocator + Clone
{
    fn eq(&self, other: &Self) -> bool {
        self.ring.get_ring() == other.ring.get_ring() && self.fft_table == other.fft_table
    }
}

impl<R, F, A> RingDecomposition<R::Type> for OddCyclotomicFFT<R, F , A> 
    where R: RingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + DivisibilityRing,
        F: FFTAlgorithm<R::Type> + PartialEq,
        A: Allocator + Clone
{
    fn expansion_factor(&self) -> i64 {
        // an overestimate, but at least simple:
        // taking the complex DWT of `(a_i)` gives elements of size `n * max |a_i|`,
        // after multiplying we are at elements of size `n^2 * max |a_i| * max |b_i|`
        // and after the final DWT we have elements of size `n^3 * max |a_i| * max |b_i|`;
        let complex_dwt_operator_norm = self.rank() as i64;
        ZZ.pow(complex_dwt_operator_norm, 3)
    }

    fn fft_backward(&self, data: &mut [El<R>], ring: &R::Type) {
        assert!(ring == self.ring.get_ring());
        let mut tmp = Vec::with_capacity_in(self.fft_table.len(), self.allocator.clone());
        tmp.extend((0..self.fft_table.len()).map(|_| ring.zero()));
        for (i, j) in self.fft_output_indices() {
            tmp[j] = ring.clone_el(&data[i]);
        }
        self.fft_table.unordered_inv_fft(&mut tmp[..], ring);

        for i in (self.rank()..self.fft_table.len()).rev() {
            let factor = ring.clone_el(&tmp[i]);
            for (j, c) in self.zeta_pow_rank.iter() {
                let mut add = ring.clone_el(&factor);
                ring.mul_assign_ref(&mut add, c);
                ring.add_assign(&mut tmp[i - self.rank() + *j], add);
            }
        }

        for i in 0..self.rank() {
            data[i] = ring.clone_el(&tmp[i]);
        }
    }

    fn fft_forward(&self, data: &mut [El<R>], ring: &R::Type) {
        assert!(ring == self.ring.get_ring());
        let mut tmp = Vec::with_capacity_in(self.fft_table.len(), self.allocator.clone());
        tmp.extend((0..self.fft_table.len()).map(|_| ring.zero()));
        for i in 0..self.rank() {
            tmp[i] = ring.clone_el(&data[i]);
        }

        self.fft_table.unordered_fft(&mut tmp[..], ring);
        for (i, j) in self.fft_output_indices() {
            data[i] = ring.clone_el(&tmp[j]); 
        }
    }

    fn rank(&self) -> usize {
        self.rank
    }
}

impl<R, F, A> CyclotomicRingDecomposition<R::Type> for OddCyclotomicFFT<R, F , A> 
    where R: RingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + DivisibilityRing,
        F: FFTAlgorithm<R::Type> + PartialEq,
        A: Allocator + Clone
{
    fn galois_group_mulrepr(&self) -> zn_64::Zn {
        zn_64::Zn::new(self.fft_table.len() as u64)
    }

    fn permute_galois_action<S>(&self, src: &[<R::Type as RingBase>::Element], dst: &mut [<R::Type as RingBase>::Element], galois_element: zn_64::ZnEl, ring: S)
        where S: RingStore<Type = R::Type> 
    {
        let Zn = self.galois_group_mulrepr();
        let hom = Zn.can_hom(&StaticRing::<i64>::RING).unwrap();
        
        for (j, i) in self.fft_output_indices() {
            dst[j] = ring.clone_el(&src[self.fft_output_indices_to_indices[self.fft_table.unordered_fft_permutation_inv(
                Zn.smallest_positive_lift(Zn.mul(galois_element, hom.map(self.fft_table.unordered_fft_permutation(i) as i64))) as usize
            )]]);
        }
    }
}

impl<R1, R2, F1, F2, A1, A2> IsomorphismInfo<R2::Type, R1::Type, OddCyclotomicFFT<R1, F1, A1>> for OddCyclotomicFFT<R2, F2, A2>
    where R1: RingStore,
        R1::Type: ZnRing + CanHomFrom<BigIntRingBase> + DivisibilityRing + PartialEq<R2::Type>,
        F1: FFTAlgorithm<R1::Type> + PartialEq,
        R2: RingStore,
        R2::Type: ZnRing + CanHomFrom<BigIntRingBase> + DivisibilityRing,
        F2: FFTAlgorithm<R2::Type> + PartialEq,
        A1: Allocator + Clone,
        A2: Allocator + Clone
{
    fn is_same_number_ring(&self, other: &OddCyclotomicFFT<R1, F1, A1>) -> bool {
        self.fft_table.len() == other.fft_table.len()
    }
}

impl<R_main, R_twiddle, A> DoubleRNSRingBase<R_main, OddCyclotomicFFT<R_main, bluestein::BluesteinFFT<R_main::Type, R_twiddle::Type, CanHom<R_twiddle, R_main>, A>, A>, A>
    where R_main: ZnRingStore + Clone,
        R_main::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        R_twiddle: ZnRingStore + Clone,
        R_twiddle::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        R_main::Type: CanIsoFromTo<R_twiddle::Type> + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone + Default
{
    pub fn new(base_ring: zn_rns::Zn<R_main, BigIntRing>, fft_rings: Vec<R_twiddle>, n: usize) -> RingValue<Self> {
        let allocator = A::default();
        let ffts = fft_rings.into_iter().enumerate().map(|(i, R)| {
            let hom = base_ring.at(i).clone().into_can_hom(R).ok().unwrap();
            OddCyclotomicFFT::create(
                hom.codomain().clone(),
                bluestein::BluesteinFFT::for_zn_with_hom(hom, n, allocator.clone()).unwrap(),
                allocator.clone()
            )
        }).collect();
        RingValue::from(
            Self::from_ring_decompositions(
                base_ring,
                ffts, 
                allocator
            )
        )
    }
}

pub type DefaultOddCyclotomicDoubleRNSRingBase<R = zn_64::Zn> = DoubleRNSRingBase<R, OddCyclotomicFFT<R, bluestein::BluesteinFFT<<R as RingStore>::Type, <R as RingStore>::Type, Identity<R>>>, Global>;
pub type DefaultOddCyclotomicDoubleRNSRing<R = zn_64::Zn> = DoubleRNSRing<R, OddCyclotomicFFT<R, bluestein::BluesteinFFT<<R as RingStore>::Type, <R as RingStore>::Type, Identity<R>>>, Global>;

impl<R, A> DoubleRNSRingBase<R, OddCyclotomicFFT<R, bluestein::BluesteinFFT<R::Type, R::Type, Identity<R>, A>, A>, A>
    where R: ZnRingStore + Clone,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone + Default
{
    pub fn new(base_ring: zn_rns::Zn<R, BigIntRing>, n: usize) -> RingValue<Self> {
        let allocator = A::default();
        let ffts = base_ring.as_iter().map(|R| {
            OddCyclotomicFFT::create(
                R.clone(),
                bluestein::BluesteinFFT::for_zn(R.clone(), n, allocator.clone()).unwrap(),
                allocator.clone()
            )
        }).collect();
        RingValue::from(
            Self::from_ring_decompositions(
                base_ring,
                ffts, 
                allocator
            )
        )
    }
}

pub type DefaultOddCyclotomicNTTRingBase<R = zn_64::Zn> = NTTRingBase<R, OddCyclotomicFFT<R, bluestein::BluesteinFFT<<R as RingStore>::Type, <R as RingStore>::Type, Identity<R>>>>;
pub type DefaultOddCyclotomicNTTRing<R = zn_64::Zn> = NTTRing<R, OddCyclotomicFFT<R, bluestein::BluesteinFFT<<R as RingStore>::Type, <R as RingStore>::Type, Identity<R>>>>;

impl<R> NTTRingBase<RingValue<R>, OddCyclotomicFFT<RingValue<R>, bluestein::BluesteinFFT<R, R, Identity<RingValue<R>>>>>
    where R: StdZn + Clone
{
    pub fn new(base_ring: RingValue<R>, n: usize) -> RingValue<Self> {
        Self::new_with(base_ring, n, Global)
    }
}

impl<R, A> NTTRingBase<RingValue<R>, OddCyclotomicFFT<RingValue<R>, bluestein::BluesteinFFT<R, R, Identity<RingValue<R>>>>, A>
    where R: StdZn + Clone,
        A: Allocator + Default + Clone
{
    pub fn new_with(base_ring: RingValue<R>, n: usize, allocator: A) -> RingValue<Self> {
        assert!(n % 2 == 1);
        let modulus = int_cast(base_ring.integer_ring().clone_el(base_ring.modulus()), StaticRing::<i64>::RING, base_ring.integer_ring());
        let expansion_factor = ZZ.pow(euler_phi(&factor(&ZZ, n as i64)), 3);
        let log2_m = ZZ.abs_log2_ceil(&(n as i64)).unwrap() + 1;
        let required_bits = ((modulus as f64).log2() * 2. + (expansion_factor as f64).log2()).ceil() as usize;
        let congruent_to_one_mod = n << log2_m;
        let primes = sample_primes(required_bits, required_bits + 4, 58, &BigIntRing::RING.coerce(&ZZ, congruent_to_one_mod as i64)).unwrap();

        let mut rns_base = Vec::new();
        let mut ring_decompositions = Vec::new();
        for p in primes {
            let Fp = RingValue::from(R::create(|int_ring| Ok(int_cast(p, RingRef::new(int_ring), &BigIntRing::RING))).unwrap_or_else(|x| x));
            let as_field = RingRef::new(Fp.get_ring()).as_field().ok().unwrap();
            let pow2_root_of_unity = Fp.coerce(&as_field, get_prim_root_of_unity_pow2(as_field, log2_m).unwrap());
            let root_of_unity = Fp.coerce(&as_field, get_prim_root_of_unity(as_field, 2 * n).unwrap());
            let fft_table = bluestein::BluesteinFFT::new(Fp.clone(), root_of_unity, pow2_root_of_unity, n, log2_m, Global);
            ring_decompositions.push(OddCyclotomicFFT::create(Fp.clone(), fft_table, Global));
            assert_eq!(expansion_factor, ring_decompositions.last().unwrap().expansion_factor());
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

#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use feanor_math::rings::zn::zn_64::Zn;
#[cfg(test)]
use feanor_math::rings::extension::FreeAlgebraStore;
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
    assert_eq!(577, int_cast(R.get_ring().rns_base().at(0).integer_ring().clone_el(R.get_ring().rns_base().at(0).modulus()), StaticRing::<i64>::RING, R.get_ring().rns_base().at(0).integer_ring()));
    assert_eq!(6, R.rank());
    [
        ring_literal!(&R, [0, 0, 0, 0, 0, 0]),
        ring_literal!(&R, [1, 0, 0, 0, 0, 0]),
        ring_literal!(&R, [-1, 0, 0, 0, 0, 0]),
        ring_literal!(&R, [0, 0, 0, 0, 0, 1]),
        ring_literal!(&R, [0, 0, 0, 0, 0, -1]),
        ring_literal!(&R, [1, 1, 1, 1, 1, 1]),
        ring_literal!(&R, [1, -1, 0, 0, 0, 0]),
        // these elements are non-invertible, in the same prime ideal `(X + 256)`
        ring_literal!(&R, [435, 287, 1, 0, 0, 0]),
        ring_literal!(&R, [256, 1, 0, 0, 0, 0]),
        ring_literal!(&R, [0, 435, 287, 1, 0, 0]),
    ].into_iter()
}

#[test]
fn test_odd_cyclotomic_fft_factor_fft() {
    let ring = Zn::new(2801);
    let root_of_unity = ring.int_hom().map(3);
    let base_fft = factor_fft::CoprimeCooleyTuckeyFFT::new(
        ring,
        ring.pow(root_of_unity, 80),
        bluestein::BluesteinFFT::new(ring, ring.pow(root_of_unity, 280), ring.pow(root_of_unity, 7 * 25), 5, 4, Global),
        bluestein::BluesteinFFT::new(ring, ring.pow(root_of_unity, 200), ring.pow(root_of_unity, 7 * 25), 7, 4, Global)
    );
    let fft = OddCyclotomicFFT::create(ring, base_fft, Global);

    let original = [1, 0, 2800, 1, 1, 2800, 1, 2800, 0, 0, 0, 1, 2800, 0, 0, 1, 0, 1, 1, 1, 2800, 0, 0, 1];
    let mut data = original.iter().copied().map(|x| ring.int_hom().map(x)).collect::<Vec<_>>();
    fft.fft_forward(&mut data[..], ring.get_ring());

    fft.fft_backward(&mut data[..], ring.get_ring());
    for i in 0..data.len() {
        assert_el_eq!(&ring, &ring.int_hom().map(original[i]), &data[i]);
    }
}

#[test]
fn test_ring_axioms() {
    let rns_base = zn_rns::Zn::new(vec![Zn::new(577), Zn::new(1153)], BigIntRing::RING);
    let R = DefaultOddCyclotomicDoubleRNSRingBase::new(rns_base, 9);
    feanor_math::ring::generic_tests::test_ring_axioms(&R, edge_case_elements(&R));
}

#[test]
fn test_divisibility_axioms() {
    let rns_base = zn_rns::Zn::new(vec![Zn::new(577), Zn::new(1153)], BigIntRing::RING);
    let R = DefaultOddCyclotomicDoubleRNSRingBase::new(rns_base, 9);
    feanor_math::divisibility::generic_tests::test_divisibility_axioms(&R, edge_case_elements(&R));
}

#[test]
fn test_free_algebra_axioms() {
    let rns_base = zn_rns::Zn::new(vec![Zn::new(577), Zn::new(1153)], BigIntRing::RING);
    let R = DefaultOddCyclotomicDoubleRNSRingBase::new(rns_base, 9);
    feanor_math::rings::extension::generic_tests::test_free_algebra_axioms(R);
}

#[test]
fn test_cyclotomic_ring_axioms() {
    let rns_base = zn_rns::Zn::new(vec![Zn::new(577), Zn::new(1153)], BigIntRing::RING);
    let R = DefaultOddCyclotomicDoubleRNSRingBase::new(rns_base, 9);
    generic_test_cyclotomic_ring_axioms(R);
}

#[test]
fn test_permute_galois_automorphism() {
    let Fp = Zn::new(257);
    let R = DefaultOddCyclotomicNTTRingBase::new_with(Fp, 7, Global);
    let hom = R.get_ring().galois_group_mulrepr().into_int_hom();

    assert_el_eq!(R, ring_literal!(&R, [0, 0, 1, 0, 0, 0]), R.get_ring().apply_galois_action(&ring_literal!(&R, [0, 1, 0, 0, 0, 0]), hom.map(2)));
    assert_el_eq!(R, ring_literal!(&R, [0, 0, 0, 1, 0, 0]), R.get_ring().apply_galois_action(&ring_literal!(&R, [0, 1, 0, 0, 0, 0]), hom.map(3)));
    assert_el_eq!(R, ring_literal!(&R, [0, 0, 0, 0, 1, 0]), R.get_ring().apply_galois_action(&ring_literal!(&R, [0, 0, 1, 0, 0, 0]), hom.map(2)));
    assert_el_eq!(R, ring_literal!(&R, [-1, -1, -1, -1, -1, -1]), R.get_ring().apply_galois_action(&ring_literal!(&R, [0, 0, 1, 0, 0, 0]), hom.map(3)));

    let R = DefaultOddCyclotomicNTTRingBase::new_with(Fp, 15, Global);
    let hom = R.get_ring().galois_group_mulrepr().into_int_hom();

    assert_el_eq!(R, ring_literal!(&R, [0, 0, 1, 0, 0, 0, 0, 0]), R.get_ring().apply_galois_action(&ring_literal!(&R, [0, 1, 0, 0, 0, 0, 0, 0]), hom.map(2)));
    assert_el_eq!(R, ring_literal!(&R, [0, 0, 0, 0, 1, 0, 0, 0]), R.get_ring().apply_galois_action(&ring_literal!(&R, [0, 1, 0, 0, 0, 0, 0, 0]), hom.map(4)));
    assert_el_eq!(R, ring_literal!(&R, [-1, 1, 0, -1, 1, -1, 0, 1]), R.get_ring().apply_galois_action(&ring_literal!(&R, [0, 1, 0, 0, 0, 0, 0, 0]), hom.map(8)));
    assert_el_eq!(R, ring_literal!(&R, [-1, 1, 0, -1, 1, -1, 0, 1]), R.get_ring().apply_galois_action(&ring_literal!(&R, [0, 0, 0, 0, 1, 0, 0, 0]), hom.map(2)));
}
