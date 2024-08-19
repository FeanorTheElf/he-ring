use std::alloc::{Allocator, Global};
use std::collections::HashMap;

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

use crate::complexfft::automorphism::euler_phi;
use crate::rings::double_rns_ring::*;
use crate::rings::decomposition::*;

const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;

pub struct OddCyclotomicFFT<R, F, A = Global> 
    where R: RingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + DivisibilityRing,
        F: FFTAlgorithm<R::Type> + PartialEq,
        A: Allocator + Clone
{
    ring: R,
    fft_table: F,
    n_factorization: Vec<(i64, usize)>,
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
    fn create(ring: R, fft_table: F, allocator: A) -> Self {
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

        return Self { ring, fft_table, n_factorization, zeta_pow_rank, rank, allocator };
    }
}

impl<R, F, A> OddCyclotomicFFT<R, F, A> 
    where R: RingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + DivisibilityRing,
        F: FFTAlgorithm<R::Type> + PartialEq,
        A: Allocator + Clone
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
        for (i, j) in self.fft_output_indices().enumerate() {
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
        for (i, j) in self.fft_output_indices().enumerate() {
            data[i] = ring.clone_el(&tmp[j]); 
        }
    }

    fn rank(&self) -> usize {
        self.rank
    }
}

impl<R1, R2, F1, F2, A1, A2> IsomorphismInfo<R2::Type, R1::Type, OddCyclotomicFFT<R1, F1, A1>> for OddCyclotomicFFT<R2, F2, A2>
    where R1: RingStore,
        R1::Type: ZnRing + CanHomFrom<BigIntRingBase> + DivisibilityRing + PartialEq<R2::Type>,
        F1: FFTAlgorithm<R1::Type> + PartialEq,
        R2: RingStore,
        R2::Type: ZnRing + CanHomFrom<BigIntRingBase> + DivisibilityRing,
        F2: FFTAlgorithm<R2::Type> + PartialEq + PartialEq<F1>,
        A1: Allocator + Clone,
        A2: Allocator + Clone
{
    fn is_same_number_ring(&self, other: &OddCyclotomicFFT<R1, F1, A1>) -> bool {
        self.fft_table.len() == other.fft_table.len()
    }

    fn is_exactly_same(&self, other: &OddCyclotomicFFT<R1, F1, A1>) -> bool {
        self.is_same_number_ring(other) && other.ring.get_ring() == self.ring.get_ring() && self.fft_table == other.fft_table
    }
}

// impl<R1, R2, F1, F2, A1, A2> SameNumberRingCross<R2::Type, R1::Type, complexfft::odd_cyclotomic::OddCyclotomicFFT<R1, F1, A1>> for OddCyclotomicFFT<R2, F2, A2>
//     where R1: RingStore,
//         R1::Type: ZnRing + CanHomFrom<StaticRingBase<i64>> + DivisibilityRing,
//         F1: FFTAlgorithm<Complex64Base> + FFTErrorEstimate,
//         R2: RingStore,
//         R2::Type: ZnRing + CanHomFrom<BigIntRingBase> + DivisibilityRing,
//         F2: FFTAlgorithm<R2::Type>,
//         A1: Allocator + Clone,
//         A2: Allocator + Clone
// {
//     fn is_isomorphic(&self, other: &complexfft::odd_cyclotomic::OddCyclotomicFFT<R1, F1, A1>) -> bool {
//         self.fft_table.len() == other.n()
//     }
// }

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

impl<R_main, R_twiddle, A> DoubleRNSRingBase<
    R_main, 
    OddCyclotomicFFT<R_main, factor_fft::CoprimeCooleyTuckeyFFT<
        R_main::Type, 
        R_twiddle::Type, 
        CanHom<R_twiddle, R_main>, 
        bluestein::BluesteinFFT<R_main::Type, R_twiddle::Type, CanHom<R_twiddle, R_main>, A>, 
        bluestein::BluesteinFFT<R_main::Type, R_twiddle::Type, CanHom<R_twiddle, R_main>, A>
    >, A>, 
    A
>
    where R_main: ZnRingStore + Clone,
        R_twiddle: ZnRingStore + Clone,
        R_twiddle::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        R_main::Type: ZnRing + CanIsoFromTo<R_twiddle::Type> + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    pub fn new(base_ring: zn_rns::Zn<R_main, BigIntRing>, fft_rings: Vec<R_twiddle>, n: (i64, i64), allocator: A) -> RingValue<Self> {
        let bluestein_log2_n0 = ZZ.abs_log2_ceil(&n.0).unwrap() + 1;
        let bluestein_log2_n1 = ZZ.abs_log2_ceil(&n.1).unwrap() + 1;
        let ffts = fft_rings.into_iter().enumerate().map(|(i, R)| {
            let R_as_field = (&R).as_field().ok().unwrap();
            let root_of_unity = R_as_field.get_ring().unwrap_element(algorithms::unity_root::get_prim_root_of_unity(&R_as_field, (2 * n.0 * n.1) as usize).unwrap());
            let hom: CanHom<R_twiddle, R_main> = base_ring.at(i).clone().into_can_hom(R.clone()).ok().unwrap();
            OddCyclotomicFFT::create(hom.codomain().clone(), factor_fft::CoprimeCooleyTuckeyFFT::new_with_hom(
                hom.clone(),
                R.clone_el(&root_of_unity),
                bluestein::BluesteinFFT::new_with_hom(
                    hom.clone(), 
                    R.pow(R.clone_el(&root_of_unity), n.0 as usize), 
                    R_as_field.get_ring().unwrap_element(algorithms::unity_root::get_prim_root_of_unity_pow2(&R_as_field, bluestein_log2_n1).unwrap()), 
                    n.1 as usize, 
                    bluestein_log2_n1,
                    allocator.clone()
                ),
                bluestein::BluesteinFFT::new_with_hom(
                    hom.clone(), 
                    R.pow(root_of_unity, n.1 as usize), 
                    R_as_field.get_ring().unwrap_element(algorithms::unity_root::get_prim_root_of_unity_pow2(&R_as_field, bluestein_log2_n0).unwrap()), 
                    n.0 as usize, 
                    bluestein_log2_n0,
                    allocator.clone()
                ),
            ), allocator.clone())
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

#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use feanor_math::rings::zn::zn_64::Zn;
#[cfg(test)]
use feanor_math::rings::extension::FreeAlgebraStore;

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

// #[test]
// fn test_cyclotomic_ring_axioms() {
//     let rns_base = zn_rns::Zn::new(vec![Zn::new(577), Zn::new(1153)], BigIntRing::RING);
//     let R = DefaultOddCyclotomicDoubleRNSRingBase::new(rns_base, 9);
//     generic_test_cyclotomic_ring_axioms(R);
// }
