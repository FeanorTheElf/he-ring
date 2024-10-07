use std::alloc::{Allocator, Global};
use std::collections::HashMap;
use std::cmp::max;

use bluestein::BluesteinFFT;
use factor_fft::CoprimeCooleyTuckeyFFT;
use feanor_math::algorithms::eea::signed_gcd;
use feanor_math::algorithms::int_factor::{self, factor};
use feanor_math::algorithms::miller_rabin::is_prime;
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
use super::number_ring_quo::*;
use crate::{euler_phi, euler_phi_squarefree, sample_primes, StdZn};

pub struct OddCyclotomicDecomposableNumberRing {
    n_factorization_squarefree: Vec<i64>
}

impl OddCyclotomicDecomposableNumberRing {

    pub fn new(n: usize) -> Self {
        assert!(n % 2 == 1);
        assert!(n > 1);
        let factorization = int_factor::factor(StaticRing::<i64>::RING, n as i64);
        // while most of the arithmetic still works with non-squarefree n, our statements about the geometry
        // of the number ring as lattice don't hold anymore (currently this refers to the `norm1_to_norm2_expansion_factor`
        // functions)
        for (_, e) in &factorization {
            assert!(*e == 1, "n = {} is not squarefree", n);
        }
        Self {
            n_factorization_squarefree: factorization.iter().map(|(p, _)| *p).collect()
        }
    }

    ///
    /// Returns a bound on
    /// ```text
    ///   sup_(x in R \ {0}) | x |_can / | x |'_inf
    /// ```
    /// where `| . |'_inf` is similar to `| . |_inf`, but takes the inf-norm w.r.t.
    /// the powerful basis representation. The powerful basis is given by the monomials
    /// `X^(n i1 / p1 + ... + n ir / pr)` for `0 <= ik < phi(pk) = pk - 1`, and `n = p1 ... pr` is
    /// squarefree with prime factors `p1, ..., pr`.
    /// 
    /// To compare, the standard inf norm `| . |_inf` is the inf-norm w.r.t. the
    /// coefficient basis representation, which is just given by the monomials `X^i`
    /// for `0 <= i < phi(n)`. It has the disadvantage that it is not compatible with
    /// the tensor-product factorization
    /// ```text
    ///   Q[X]/(Phi_n) = Q[X]/(Phi_p1) ⊗ ... ⊗ Q[X]/(Phi_pr)
    /// ```
    /// 
    pub fn powful_inf_to_can_norm_expansion_factor(&self) -> f64 {
        let rank = euler_phi_squarefree(&self.n_factorization_squarefree);
        // a simple estimate; it holds, as for any `x` with `|x|_inf <= b`, the coefficients
        // under the canonical embedding are clearly `<= nb` in absolute value, thus the canonical
        // norm is at most `n sqrt(n)`
        (rank as f64).powi(3).sqrt()
    }

    ///
    /// Returns a bound on
    /// ```text
    ///   sup_(x in R \ {0}) | x |'_inf / | x |_can
    /// ```
    /// For the distinction of standard inf-norm and powerful inf-norm, see
    /// the doc of [`OddCyclotomicDecomposableNumberRing::powful_inf_to_can_norm_expansion_factor()`].
    /// 
    pub fn can_to_powful_inf_norm_expansion_factor(&self) -> f64 {
        // if `n = p` is a prime, we can give an explicit inverse to the matrix
        // `A = ( zeta^(ij) )` where `i in (Z/pZ)*` and `j in { 0, ..., p - 2 }` by
        // `A^-1 = ( zeta^(ij) - zeta^j ) / p` with `i in { 0, ..., p - 2 }` and `j in (Z/pZ)*`.
        // This clearly shows that in this case, then expansion factor is at most 
        // `(p - 1) | zeta^(ij) - zeta^j | / p < 2`. By the tensor product compatibility of
        // the powerful inf-norm, we thus get this bound
        2f64.powi(self.n_factorization_squarefree.len() as i32)
    }

    ///
    /// Returns a bound on
    /// ```text
    ///   sup_(x in R \ {0}) | x |_inf / | x |'_inf
    /// ```
    /// For the distinction of standard inf-norm and powerful inf-norm, see
    /// the doc of [`OddCyclotomicDecomposableNumberRing::powful_inf_to_can_norm_expansion_factor()`].
    /// 
    pub fn powful_inf_to_inf_norm_expansion_factor(&self) -> f64 {
        // the map we are considering is the identity on powerful monomials
        // `X^(n i1 / p1 + ... + n ir / pr)` if `n i1 / p1 + ... + n ir / pr < phi(n)`
        // and reduction modulo `Phi_n` otherwise
        unimplemented!()
    }
}

impl PartialEq for OddCyclotomicDecomposableNumberRing {
    fn eq(&self, other: &Self) -> bool {
        self.n_factorization_squarefree == other.n_factorization_squarefree
    }
}

impl<FpTy> DecomposableNumberRing<FpTy> for OddCyclotomicDecomposableNumberRing
    where FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>
{
    type Decomposed = OddCyclotomicDecomposedNumberRing<FpTy, BluesteinFFT<FpTy::Type, FpTy::Type, Identity<FpTy>>>;

    fn can_to_inf_norm_expansion_factor(&self) -> f64 {
        self.powful_inf_to_inf_norm_expansion_factor() * self.can_to_powful_inf_norm_expansion_factor()
    }

    fn inf_to_can_norm_expansion_factor(&self) -> f64 {
        let rank = euler_phi_squarefree(&self.n_factorization_squarefree);
        // a simple estimate; it holds, as for any `x` with `|x|_inf <= b`, the coefficients
        // under the canonical embedding are clearly `<= nb` in absolute value, thus the canonical
        // norm is at most `n sqrt(n)`
        (rank as f64).powi(3).sqrt()
    }

    fn mod_p(&self, Fp: FpTy) -> Self::Decomposed {
        let n_factorization = &self.n_factorization_squarefree;
        let n = n_factorization.iter().copied().product::<i64>();
        let rank = euler_phi_squarefree(&n_factorization) as usize;

        let Fp_as_field = (&Fp).as_field().ok().unwrap();
        let zeta = get_prim_root_of_unity(&Fp_as_field, 2 * n as usize).unwrap();
        let zeta = Fp_as_field.get_ring().unwrap_element(zeta);
        let log2_m = StaticRing::<i64>::RING.abs_log2_ceil(&n).unwrap() + 1;
        let zeta_m = get_prim_root_of_unity_pow2(&Fp_as_field, log2_m).unwrap();
        let zeta_m = Fp_as_field.get_ring().unwrap_element(zeta_m);
        let fft_table = BluesteinFFT::new(Fp.clone(), zeta, zeta_m, n as usize, log2_m, Global);

        return OddCyclotomicDecomposedNumberRing::create_squarefree(fft_table, Fp, &self.n_factorization_squarefree, Global);
    }

    fn largest_suitable_prime(&self, leq_than: i64) -> Option<i64> {
        let n = self.n_factorization_squarefree.iter().copied().product::<i64>();
        let log2_m = StaticRing::<i64>::RING.abs_log2_ceil(&n).unwrap() + 1;
        let modulus = n << log2_m;
        let mut current = leq_than - (leq_than % modulus) + 1;
        while current > 0 && !is_prime(StaticRing::<i64>::RING, &current, 10) {
            current -= modulus;
        }
        if current <= 0 {
            return None;
        } else {
            return Some(current);
        }
    }
}

pub struct CompositeCyclotomicDecomposableNumberRing {
    base: OddCyclotomicDecomposableNumberRing,
    n1: i64,
    n2: i64
}

impl CompositeCyclotomicDecomposableNumberRing {

    pub fn new(n1: usize, n2: usize) -> Self {
        assert!(n1 % 2 == 1);
        assert!(n2 % 2 == 1);
        assert!(n1 > 1);
        assert!(n2 > 1);
        assert!(signed_gcd(n1 as i64, n2 as i64, StaticRing::<i64>::RING) == 1);
        Self {
            base: OddCyclotomicDecomposableNumberRing::new(n1 * n2),
            n1: n1 as i64,
            n2: n2 as i64
        }
    }
}

impl PartialEq for CompositeCyclotomicDecomposableNumberRing {

    fn eq(&self, other: &Self) -> bool {
        self.base == other.base
    }
}

impl<FpTy> DecomposableNumberRing<FpTy> for CompositeCyclotomicDecomposableNumberRing
    where FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>
{
    type Decomposed = OddCyclotomicDecomposedNumberRing<FpTy, CoprimeCooleyTuckeyFFT<FpTy::Type, FpTy::Type, Identity<FpTy>, BluesteinFFT<FpTy::Type, FpTy::Type, Identity<FpTy>>, BluesteinFFT<FpTy::Type, FpTy::Type, Identity<FpTy>>>>;

    fn mod_p(&self, Fp: FpTy) -> Self::Decomposed {
        let n_factorization = &self.base.n_factorization_squarefree;
        let n = n_factorization.iter().copied().product::<i64>();
        let rank = euler_phi_squarefree(&n_factorization) as usize;

        let Fp_as_field = (&Fp).as_field().ok().unwrap();
        let zeta = get_prim_root_of_unity(&Fp_as_field, 2 * n as usize).unwrap();
        let zeta = Fp_as_field.get_ring().unwrap_element(zeta);

        let zeta_2n1 = Fp.pow(Fp.clone_el(&zeta), self.n2 as usize);
        let log2_m1 = StaticRing::<i64>::RING.abs_log2_ceil(&self.n1).unwrap() + 1;
        let zeta_m1 = get_prim_root_of_unity_pow2(&Fp_as_field, log2_m1).unwrap();
        let zeta_m1 = Fp_as_field.get_ring().unwrap_element(zeta_m1);
        let fft_table1 = BluesteinFFT::new(Fp.clone(), zeta_2n1, zeta_m1, self.n1 as usize, log2_m1, Global);

        let zeta_2n2 = Fp.pow(Fp.clone_el(&zeta), self.n1 as usize);
        let log2_m2 = StaticRing::<i64>::RING.abs_log2_ceil(&self.n2).unwrap() + 1;
        let zeta_m2 = get_prim_root_of_unity_pow2(&Fp_as_field, log2_m2).unwrap();
        let zeta_m2 = Fp_as_field.get_ring().unwrap_element(zeta_m2);
        let fft_table2 = BluesteinFFT::new(Fp.clone(), zeta_2n2, zeta_m2, self.n2 as usize, log2_m2, Global);

        let fft_table = CoprimeCooleyTuckeyFFT::new(Fp.clone(), Fp.pow(zeta, 2), fft_table1, fft_table2);

        return OddCyclotomicDecomposedNumberRing::create_squarefree(fft_table, Fp, &self.base.n_factorization_squarefree, Global);
    }

    fn largest_suitable_prime(&self, leq_than: i64) -> Option<i64> {
        let n = self.base.n_factorization_squarefree.iter().copied().product::<i64>();
        let log2_m = max(
            StaticRing::<i64>::RING.abs_log2_ceil(&self.n1).unwrap() + 1,
            StaticRing::<i64>::RING.abs_log2_ceil(&self.n2).unwrap() + 1
        );
        let modulus = n << log2_m;
        let mut current = leq_than - (leq_than % modulus) + 1;
        while current > 0 && !is_prime(StaticRing::<i64>::RING, &current, 10) {
            current -= modulus;
        }
        if current <= 0 {
            return None;
        } else {
            return Some(current);
        }
    }

    fn inf_to_can_norm_expansion_factor(&self) -> f64 {
        <_ as DecomposableNumberRing<FpTy>>::inf_to_can_norm_expansion_factor(&self.base)
    }

    fn can_to_inf_norm_expansion_factor(&self) -> f64 {
        <_ as DecomposableNumberRing<FpTy>>::can_to_inf_norm_expansion_factor(&self.base)
    }
}

pub struct OddCyclotomicDecomposedNumberRing<R, F, A = Global> 
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

impl<R, F, A> PartialEq for OddCyclotomicDecomposedNumberRing<R, F, A> 
    where R: RingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + DivisibilityRing,
        F: FFTAlgorithm<R::Type> + PartialEq,
        A: Allocator + Clone
{
    fn eq(&self, other: &Self) -> bool {
        self.ring.get_ring() == other.ring.get_ring() && self.fft_table == other.fft_table
    }
}

impl<R, F, A> OddCyclotomicDecomposedNumberRing<R, F, A> 
    where R: RingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + DivisibilityRing,
        F: FFTAlgorithm<R::Type> + PartialEq,
        A: Allocator + Clone
{
    fn create_squarefree(fft_table: F, Fp: R, n_factorization: &[i64], allocator: A) -> Self {
        let n = n_factorization.iter().copied().product::<i64>();
        let rank = euler_phi_squarefree(&n_factorization) as usize;

        let poly_ring = SparsePolyRing::new(&Fp, "X");
        let cyclotomic_poly = algorithms::cyclotomic::cyclotomic_polynomial(&poly_ring, n as usize);
        assert_eq!(poly_ring.degree(&cyclotomic_poly).unwrap(), rank);
        let mut zeta_pow_rank = HashMap::new();
        for (a, i) in poly_ring.terms(&cyclotomic_poly) {
            if i != rank {
                zeta_pow_rank.insert(i, Fp.negate(Fp.clone_el(a)));
            }
        }

        let fft_output_indices_to_indices = (0..fft_table.len()).scan(0, |state, i| {
            let power = fft_table.unordered_fft_permutation(i);
            if n_factorization.iter().all(|p| power as i64 % *p != 0) {
                *state += 1;
                return Some(*state - 1);
            } else {
                return Some(usize::MAX);
            }
        }).collect::<Vec<_>>();

        return Self { ring: Fp, fft_table, zeta_pow_rank, rank, allocator, fft_output_indices_to_indices };
    }

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

impl<R, F, A> DecomposedNumberRing<R::Type> for OddCyclotomicDecomposedNumberRing<R, F , A> 
    where R: RingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + DivisibilityRing,
        F: FFTAlgorithm<R::Type> + PartialEq,
        A: Allocator + Clone
{
    fn fft_backward(&self, data: &mut [El<R>]) {
        let ring = self.base_ring();
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

    fn fft_forward(&self, data: &mut [El<R>]) {
        let ring = self.base_ring();
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

    fn base_ring(&self) -> RingRef<R::Type> {
        RingRef::new(self.ring.get_ring())
    }
}