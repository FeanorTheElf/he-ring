use std::collections::HashMap;

use feanor_math::algorithms::fft::complex_fft::ErrorEstimate;
use feanor_math::algorithms::{fft::*, self};
use feanor_math::integer::IntegerRingStore;
use feanor_math::integer::*;
use feanor_math::rings::float_complex::Complex64;
use feanor_math::rings::poly::*;
use feanor_math::divisibility::DivisibilityRing;
use feanor_math::mempool::*;
use feanor_math::primitive_int::{StaticRing, StaticRingBase};
use feanor_math::ring::*;
use feanor_math::homomorphism::*;
use feanor_math::rings::poly::sparse_poly::SparsePolyRing;
use feanor_math::rings::zn::*;
use feanor_math::vector::*;

use crate::complexfft;
use crate::cyclotomic::*;
use crate::doublerns::double_rns_ring::*;

const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;

fn phi(factorization: &Vec<(i64, usize)>) -> i64 {
    ZZ.prod(factorization.iter().map(|(p, e)| (p - 1) * ZZ.pow(*p, e - 1)))
}

pub struct OddCyclotomicFFT<F> 
    where F: FFTTable,
        F::Ring: Sized + ZnRingStore,
        <F::Ring as RingStore>::Type: ZnRing + CanHomFrom<BigIntRingBase>
{
    fft_table: F,
    n_factorization: Vec<(i64, usize)>,
    zeta_pow_rank: HashMap<usize, El<F::Ring>>,
    rank: usize
}

impl<F> OddCyclotomicFFT<F> 
    where F: FFTTable,
        F::Ring: Sized + ZnRingStore,
        <F::Ring as RingStore>::Type: ZnRing + DivisibilityRing + CanHomFrom<BigIntRingBase>
{
    fn create(fft_table: F) -> Self {
        let ring = fft_table.ring();

        let n_factorization = algorithms::int_factor::factor(&ZZ, fft_table.len() as i64);
        let rank = phi(&n_factorization) as usize;

        let poly_ring = SparsePolyRing::new(&ring, "X");
        let cyclotomic_poly = algorithms::cyclotomic::cyclotomic_polynomial(&poly_ring, fft_table.len());
        assert_eq!(poly_ring.degree(&cyclotomic_poly).unwrap(), rank);
        let mut zeta_pow_rank = HashMap::new();
        for (a, i) in poly_ring.terms(&cyclotomic_poly) {
            if i != rank {
                zeta_pow_rank.insert(i, ring.negate(ring.clone_el(a)));
            }
        }

        return Self { fft_table, n_factorization, zeta_pow_rank, rank };
    }
}

impl<F> OddCyclotomicFFT<F> 
    where F: FFTTable,
        F::Ring: Sized + ZnRingStore,
        <F::Ring as RingStore>::Type: ZnRing + CanHomFrom<BigIntRingBase>
{
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

impl<F> GeneralizedFFT for OddCyclotomicFFT<F> 
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
        let mut tmp = memory_provider.get_new_init(self.fft_table.len(), |_| ring.zero());
        for (i, j) in self.fft_output_indices().enumerate() {
            tmp[j] = ring.clone_el(&data[i]);
        }
        self.fft_table.unordered_inv_fft(&mut tmp[..], memory_provider, &ring.can_hom(self.fft_table.ring()).unwrap());

        let hom = ring.can_hom(self.base_ring()).unwrap();
        for i in (self.rank()..self.fft_table.len()).rev() {
            let factor = ring.clone_el(&tmp[i]);
            for (j, c) in self.zeta_pow_rank.iter() {
                let mut add = ring.clone_el(&factor);
                ring.get_ring().mul_assign_map_in_ref(self.base_ring().get_ring(), &mut add, c, hom.raw_hom());
                ring.add_assign(&mut tmp[i - self.rank() + *j], add);
            }
        }

        for i in 0..self.rank() {
            data[i] = ring.clone_el(&tmp[i]);
        }
    }

    fn fft_forward<S, M>(&self, data: &mut [El<S>], ring: &S, memory_provider: &M)
        where S: ZnRingStore,
            S::Type: ZnRing + CanHomFrom<Self::BaseRingBase>,
            M: MemoryProvider<El<S>>
    {
        let mut tmp = memory_provider.get_new_init(self.fft_table.len(), |_| ring.zero());
        for i in 0..self.rank() {
            tmp[i] = ring.clone_el(&data[i]);
        }

        self.fft_table.unordered_fft(&mut tmp[..], memory_provider, &ring.can_hom(self.fft_table.ring()).unwrap());
        for (i, j) in self.fft_output_indices().enumerate() {
            data[i] = ring.clone_el(&tmp[j]); 
        }
    }

    fn rank(&self) -> usize {
        self.rank
    }
}

impl<F1, F2> GeneralizedFFTIso<OddCyclotomicFFT<F1>> for OddCyclotomicFFT<F2>
    where F1: FFTTable,
        F1::Ring: Sized + ZnRingStore,
        <F1::Ring as RingStore>::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F2: FFTTable,
        F2::Ring: Sized + ZnRingStore,
        <F2::Ring as RingStore>::Type: ZnRing + CanHomFrom<BigIntRingBase>
{
    fn is_isomorphic(&self, other: &OddCyclotomicFFT<F1>) -> bool {
        self.fft_table.len() == other.fft_table.len()
    }
}

impl<R1, F1, F2> GeneralizedFFTCrossIso<complexfft::odd_cyclotomic::OddCyclotomicFFT<R1, F1>> for OddCyclotomicFFT<F2>
    where R1: RingStore,
        F1: FFTTable<Ring = Complex64> + ErrorEstimate,
        R1::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>,
        F2: FFTTable,
        F2::Ring: Sized + ZnRingStore,
        <F2::Ring as RingStore>::Type: ZnRing + CanHomFrom<BigIntRingBase>
{
    fn is_isomorphic(&self, other: &complexfft::odd_cyclotomic::OddCyclotomicFFT<R1, F1>) -> bool {
        self.fft_table.len() == other.n()
    }
}

impl<R, F, M> CyclotomicRing for DoubleRNSRingBase<R, OddCyclotomicFFT<F>, M>
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanIsoFromTo<<F::Ring as RingStore>::Type>,
        F: FFTTable,
        F::Ring: Sized + ZnRingStore,
        <F::Ring as RingStore>::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        M: MemoryProvider<El<R>>
{
    fn n(&self) -> usize {
        self.generalized_fft()[0].fft_table.len()
    }
}

impl<R_main, R_fft, M> DoubleRNSRingBase<R_main, OddCyclotomicFFT<bluestein::FFTTableBluestein<R_fft>>, M>
    where R_main: ZnRingStore,
        R_main::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        R_fft: ZnRingStore,
        R_fft::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        R_main::Type: CanIsoFromTo<R_fft::Type> + CanHomFrom<BigIntRingBase>,
        M: MemoryProvider<El<R_main>>
{
    pub fn new(base_ring: zn_rns::Zn<R_main, BigIntRing>, fft_rings: Vec<R_fft>, n: usize, memory_provider: M) -> RingValue<Self> {
        let ffts = fft_rings.into_iter().map(|R| OddCyclotomicFFT::create(
            bluestein::FFTTableBluestein::for_zn(R, n).unwrap(),
        )).collect();
        RingValue::from(
            Self::from_generalized_ffts(
                base_ring,
                ffts, 
                memory_provider
            )
        )
    }
}

impl<R_main, R_fft, M> DoubleRNSRingBase<R_main, OddCyclotomicFFT<factor_fft::FFTTableGenCooleyTuckey<R_fft, bluestein::FFTTableBluestein<R_fft>, bluestein::FFTTableBluestein<R_fft>>>, M>
    where R_main: ZnRingStore,
        R_fft: ZnRingStore + Clone,
        R_fft::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        R_main::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanIsoFromTo<R_fft::Type> + CanHomFrom<BigIntRingBase>,
        M: MemoryProvider<El<R_main>>
{
    pub fn new(base_ring: zn_rns::Zn<R_main, BigIntRing>, fft_rings: Vec<R_fft>, n: (i64, i64), memory_provider: M) -> RingValue<Self> {
        let bluestein_log2_n0 = ZZ.abs_log2_ceil(&n.0).unwrap() + 1;
        let bluestein_log2_n1 = ZZ.abs_log2_ceil(&n.1).unwrap() + 1;
        let ffts = fft_rings.into_iter().map(|R| {
            let root_of_unity = algorithms::unity_root::get_prim_root_of_unity(&R, (2 * n.0 * n.1) as usize).unwrap();
            OddCyclotomicFFT::create(factor_fft::FFTTableGenCooleyTuckey::new(
                R.pow(R.clone_el(&root_of_unity), 2),
                bluestein::FFTTableBluestein::new(
                    R.clone(), 
                    R.pow(R.clone_el(&root_of_unity), n.0 as usize), 
                    algorithms::unity_root::get_prim_root_of_unity_pow2(&R, bluestein_log2_n1).unwrap(), 
                    n.1 as usize, 
                    bluestein_log2_n1
                ),
                bluestein::FFTTableBluestein::new(
                    R.clone(), 
                    R.pow(root_of_unity, n.1 as usize), 
                    algorithms::unity_root::get_prim_root_of_unity_pow2(&R, bluestein_log2_n0).unwrap(), 
                    n.0 as usize, 
                    bluestein_log2_n0
                ),
            ))
        }).collect();
        RingValue::from(
            Self::from_generalized_ffts(
                base_ring,
                ffts, 
                memory_provider
            )
        )
    }
}

#[cfg(test)]
use feanor_math::{assert_el_eq, default_memory_provider};
#[cfg(test)]
use feanor_math::rings::extension::generic_test_free_algebra_axioms;
#[cfg(test)]
use feanor_math::rings::zn::zn_64::Zn;
#[cfg(test)]
use crate::feanor_math::rings::extension::FreeAlgebraStore;

#[cfg(test)]
fn edge_case_elements<'a, R, F, M>(R: &'a DoubleRNSRing<R, F, M>) -> impl 'a + Iterator<Item = El<DoubleRNSRing<R, F, M>>>
    where R: ZnRingStore,
        R::Type: ZnRing + CanIsoFromTo<F::BaseRingBase> + CanHomFrom<BigIntRingBase>,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
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
    let base_fft = factor_fft::FFTTableGenCooleyTuckey::new(
        ring.pow(root_of_unity, 80),
        bluestein::FFTTableBluestein::new(ring, ring.pow(root_of_unity, 280), ring.pow(root_of_unity, 7 * 25), 5, 4),
        bluestein::FFTTableBluestein::new(ring, ring.pow(root_of_unity, 200), ring.pow(root_of_unity, 7 * 25), 7, 4)
    );
    let fft = OddCyclotomicFFT::create(base_fft);

    let original = [1, 0, 2800, 1, 1, 2800, 1, 2800, 0, 0, 0, 1, 2800, 0, 0, 1, 0, 1, 1, 1, 2800, 0, 0, 1];
    let mut data = original.iter().copied().map(|x| ring.int_hom().map(x)).collect::<Vec<_>>();
    fft.fft_forward(&mut data[..], &ring, &default_memory_provider!());

    fft.fft_backward(&mut data[..], &ring, &default_memory_provider!());
    for i in 0..data.len() {
        assert_el_eq!(&ring, &ring.int_hom().map(original[i]), &data[i]);
    }
}

#[test]
fn test_ring_axioms() {
    let rns_base = zn_rns::Zn::new(vec![Zn::new(577), Zn::new(1153)], BigIntRing::RING, default_memory_provider!());
    let fft_rings = rns_base.get_ring().iter().cloned().collect();
    let R = DoubleRNSRingBase::<_, OddCyclotomicFFT<bluestein::FFTTableBluestein<_>>, _>::new(rns_base, fft_rings, 9, default_memory_provider!());
    feanor_math::ring::generic_tests::test_ring_axioms(&R, edge_case_elements(&R));
}

#[test]
fn test_divisibility_axioms() {
    let rns_base = zn_rns::Zn::new(vec![Zn::new(577), Zn::new(1153)], BigIntRing::RING, default_memory_provider!());
    let fft_rings = rns_base.get_ring().iter().cloned().collect();
    let R = DoubleRNSRingBase::<_, OddCyclotomicFFT<bluestein::FFTTableBluestein<_>>, _>::new(rns_base, fft_rings, 9, default_memory_provider!());
    feanor_math::divisibility::generic_tests::test_divisibility_axioms(&R, edge_case_elements(&R));
}

#[test]
fn test_free_algebra_axioms() {
    let rns_base = zn_rns::Zn::new(vec![Zn::new(577), Zn::new(1153)], BigIntRing::RING, default_memory_provider!());
    let fft_rings = rns_base.get_ring().iter().cloned().collect();
    let R = DoubleRNSRingBase::<_, OddCyclotomicFFT<bluestein::FFTTableBluestein<_>>, _>::new(rns_base, fft_rings, 9, default_memory_provider!());
    generic_test_free_algebra_axioms(R);
}

#[test]
fn test_cyclotomic_ring_axioms() {
    let rns_base = zn_rns::Zn::new(vec![Zn::new(577), Zn::new(1153)], BigIntRing::RING, default_memory_provider!());
    let fft_rings = rns_base.get_ring().iter().cloned().collect();
    let R = DoubleRNSRingBase::<_, OddCyclotomicFFT<bluestein::FFTTableBluestein<_>>, _>::new(rns_base, fft_rings, 9, default_memory_provider!());
    generic_test_cyclotomic_ring_axioms(R);
}
