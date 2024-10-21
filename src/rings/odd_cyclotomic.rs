use std::alloc::{Allocator, Global};
use std::collections::HashMap;
use std::cmp::max;

use bluestein::BluesteinFFT;
use factor_fft::CoprimeCooleyTuckeyFFT;
use feanor_math::algorithms::eea::{poly, signed_gcd};
use feanor_math::algorithms::int_factor::factor;
use feanor_math::algorithms::miller_rabin::is_prime;
use feanor_math::algorithms::cyclotomic::cyclotomic_polynomial;
use feanor_math::algorithms::unity_root::{get_prim_root_of_unity, get_prim_root_of_unity_pow2};
use feanor_math::algorithms::fft::*;
use feanor_math::integer::IntegerRingStore;
use feanor_math::integer::*;
use feanor_math::rings::poly::*;
use feanor_math::divisibility::DivisibilityRing;
use feanor_math::primitive_int::*;
use feanor_math::ring::*;
use feanor_math::homomorphism::*;
use feanor_math::rings::poly::sparse_poly::SparsePolyRing;
use feanor_math::rings::zn::*;
use crate::feanor_math::rings::extension::*;
use feanor_math::seq::*;

use crate::rings::double_rns_ring::*;
use crate::rings::decomposition::*;
use super::single_rns_ring;
use super::{decomposition_ring::{self, *}, double_rns_ring};
use crate::{euler_phi, euler_phi_squarefree, sample_primes, StdZn};
use crate::cyclotomic::CyclotomicRing;

pub struct OddCyclotomicDecomposableNumberRing {
    n_factorization_squarefree: Vec<i64>,
    sparse_poly_ring: SparsePolyRing<BigIntRing>,
    cyclotomic_poly: El<SparsePolyRing<BigIntRing>>
}

impl OddCyclotomicDecomposableNumberRing {

    pub fn new(n: usize) -> Self {
        assert!(n % 2 == 1);
        assert!(n > 1);
        let factorization = factor(StaticRing::<i64>::RING, n as i64);
        // while most of the arithmetic still works with non-squarefree n, our statements about the geometry
        // of the number ring as lattice don't hold anymore (currently this refers to the `norm1_to_norm2_expansion_factor`
        // functions)
        for (_, e) in &factorization {
            assert!(*e == 1, "n = {} is not squarefree", n);
        }
        let poly_ring = SparsePolyRing::new(BigIntRing::RING, "X");
        let cyclotomic_poly = cyclotomic_polynomial(&poly_ring, n);
        Self {
            n_factorization_squarefree: factorization.iter().map(|(p, _)| *p).collect(),
            sparse_poly_ring: poly_ring,
            cyclotomic_poly: cyclotomic_poly
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
        // TODO: Fix
        // conjecture: this is `<= n`; I have no proof currently, but note the following:
        // If the powerful-basis indices `n1 i1 / p1 + ... + nr ir / pr` were distributed
        // at random, about `n / phi(n)` of them would have to be "reduced", i.e. fall
        // into `{ phi(n), ..., n - 1 }` modulo `n`. Each of them contributes to the inf-operator
        // norm, up to the maximal coefficient of `Phi_n`. This maximal coefficient seems
        // to behave as `n^(1/r)`, and `n / phi(n) ~ n^((r - 1)/r)`
        let rank = euler_phi_squarefree(&self.n_factorization_squarefree);
        return rank as f64;
    }
    
    fn generating_poly<P>(&self, poly_ring: P) -> El<P>
        where P: RingStore,
            P::Type: PolyRing,
            <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing
    {
        unimplemented!()
    }
}

impl Clone for OddCyclotomicDecomposableNumberRing {
    fn clone(&self) -> Self {
        Self {
            cyclotomic_poly: self.sparse_poly_ring.clone_el(&self.cyclotomic_poly),
            sparse_poly_ring: self.sparse_poly_ring.clone(),
            n_factorization_squarefree: self.n_factorization_squarefree.clone()
        }
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
        let mut current = (leq_than - 1) - ((leq_than - 1) % modulus) + 1;
        while current > 0 && !is_prime(StaticRing::<i64>::RING, &current, 10) {
            current -= modulus;
        }
        if current <= 0 {
            return None;
        } else {
            return Some(current);
        }
    }

    fn generating_poly<P>(&self, poly_ring: P) -> El<P>
        where P: RingStore,
            P::Type: PolyRing + DivisibilityRing,
            <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing
    {
        cyclotomic_polynomial(&poly_ring, <_ as DecomposableCyclotomicNumberRing<FpTy>>::n(self) as usize)
    }

    fn rank(&self) -> usize {
        euler_phi_squarefree(&self.n_factorization_squarefree) as usize
    }
}

impl<FpTy> DecomposableCyclotomicNumberRing<FpTy> for OddCyclotomicDecomposableNumberRing
    where FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>
{
    type DecomposedAsCyclotomic = OddCyclotomicDecomposedNumberRing<FpTy, BluesteinFFT<FpTy::Type, FpTy::Type, Identity<FpTy>>>;

    fn n(&self) -> u64 {
        self.n_factorization_squarefree.iter().copied().product::<i64>() as u64
    }
}

#[derive(Clone)]
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

impl<FpTy> DecomposableCyclotomicNumberRing<FpTy> for CompositeCyclotomicDecomposableNumberRing
    where FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>
{
    type DecomposedAsCyclotomic = OddCyclotomicDecomposedNumberRing<FpTy, CoprimeCooleyTuckeyFFT<FpTy::Type, FpTy::Type, Identity<FpTy>, BluesteinFFT<FpTy::Type, FpTy::Type, Identity<FpTy>>, BluesteinFFT<FpTy::Type, FpTy::Type, Identity<FpTy>>>>;

    fn n(&self) -> u64 {
        <_ as DecomposableCyclotomicNumberRing<FpTy>>::n(&self.base)
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
        let mut current = (leq_than - 1) - ((leq_than - 1) % modulus) + 1;
        while current > 0 && !is_prime(StaticRing::<i64>::RING, &current, 10) {
            current -= modulus;
        }
        if current <= 0 {
            return None;
        } else {
            assert!(current <= leq_than);
            return Some(current);
        }
    }

    fn inf_to_can_norm_expansion_factor(&self) -> f64 {
        <_ as DecomposableNumberRing<FpTy>>::inf_to_can_norm_expansion_factor(&self.base)
    }

    fn can_to_inf_norm_expansion_factor(&self) -> f64 {
        <_ as DecomposableNumberRing<FpTy>>::can_to_inf_norm_expansion_factor(&self.base)
    }

    fn generating_poly<P>(&self, poly_ring: P) -> El<P>
        where P: RingStore,
            P::Type: PolyRing + DivisibilityRing,
            <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing
    {
        cyclotomic_polynomial(&poly_ring, <_ as DecomposableCyclotomicNumberRing<FpTy>>::n(self) as usize)
    }

    fn rank(&self) -> usize {
        euler_phi_squarefree(&self.base.n_factorization_squarefree) as usize
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
        let cyclotomic_poly = cyclotomic_polynomial(&poly_ring, n as usize);
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

impl<R, F, A> DecomposedCyclotomicNumberRing<R::Type> for OddCyclotomicDecomposedNumberRing<R, F , A> 
    where R: RingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + DivisibilityRing,
        F: FFTAlgorithm<R::Type> + PartialEq,
        A: Allocator + Clone
{
    fn n(&self) -> u64 {
        self.fft_table.len() as u64
    }

    fn permute_galois_action(&self, src: &[<R::Type as RingBase>::Element], dst: &mut [<R::Type as RingBase>::Element], galois_element: zn_64::ZnEl) {
        assert_eq!(self.rank(), src.len());
        assert_eq!(self.rank(), dst.len());
        let ring = self.base_ring();
        let index_ring = self.cyclotomic_index_ring();
        let hom = index_ring.can_hom(&StaticRing::<i64>::RING).unwrap();
        
        for (j, i) in self.fft_output_indices() {
            dst[j] = ring.clone_el(&src[self.fft_output_indices_to_indices[self.fft_table.unordered_fft_permutation_inv(
                index_ring.smallest_positive_lift(index_ring.mul(galois_element, hom.map(self.fft_table.unordered_fft_permutation(i) as i64))) as usize
            )]]);
        }
    }
}

#[cfg(test)]
use feanor_math::assert_el_eq;

#[test]
fn test_odd_cyclotomic_double_rns_ring() {
    double_rns_ring::test_with_number_ring(OddCyclotomicDecomposableNumberRing::new(5));
    double_rns_ring::test_with_number_ring(OddCyclotomicDecomposableNumberRing::new(7));
    double_rns_ring::test_with_number_ring(CompositeCyclotomicDecomposableNumberRing::new(3, 5));
    double_rns_ring::test_with_number_ring(CompositeCyclotomicDecomposableNumberRing::new(3, 7));
}

#[test]
fn test_odd_cyclotomic_decomposition_ring() {
    decomposition_ring::test_with_number_ring(OddCyclotomicDecomposableNumberRing::new(5));
    decomposition_ring::test_with_number_ring(OddCyclotomicDecomposableNumberRing::new(7));
    decomposition_ring::test_with_number_ring(CompositeCyclotomicDecomposableNumberRing::new(3, 5));
    decomposition_ring::test_with_number_ring(CompositeCyclotomicDecomposableNumberRing::new(3, 7));
}

#[test]
fn test_odd_cyclotomic_single_rns_ring() {
    single_rns_ring::test_with_number_ring(OddCyclotomicDecomposableNumberRing::new(5));
    single_rns_ring::test_with_number_ring(OddCyclotomicDecomposableNumberRing::new(7));
    single_rns_ring::test_with_number_ring(CompositeCyclotomicDecomposableNumberRing::new(3, 5));
    single_rns_ring::test_with_number_ring(CompositeCyclotomicDecomposableNumberRing::new(3, 7));
}

#[test]
fn test_permute_galois_automorphism() {
    let Fp = zn_64::Zn::new(257);
    let R = DecompositionRingBase::new(OddCyclotomicDecomposableNumberRing::new(7), Fp);
    let hom = R.get_ring().cyclotomic_index_ring().into_int_hom();

    assert_el_eq!(R, ring_literal!(&R, [0, 0, 1, 0, 0, 0]), R.get_ring().apply_galois_action(&ring_literal!(&R, [0, 1, 0, 0, 0, 0]), hom.map(2)));
    assert_el_eq!(R, ring_literal!(&R, [0, 0, 0, 1, 0, 0]), R.get_ring().apply_galois_action(&ring_literal!(&R, [0, 1, 0, 0, 0, 0]), hom.map(3)));
    assert_el_eq!(R, ring_literal!(&R, [0, 0, 0, 0, 1, 0]), R.get_ring().apply_galois_action(&ring_literal!(&R, [0, 0, 1, 0, 0, 0]), hom.map(2)));
    assert_el_eq!(R, ring_literal!(&R, [-1, -1, -1, -1, -1, -1]), R.get_ring().apply_galois_action(&ring_literal!(&R, [0, 0, 1, 0, 0, 0]), hom.map(3)));

    let R = DecompositionRingBase::new(CompositeCyclotomicDecomposableNumberRing::new(5, 3), Fp);
    let hom = R.get_ring().cyclotomic_index_ring().into_int_hom();

    assert_el_eq!(R, ring_literal!(&R, [0, 0, 1, 0, 0, 0, 0, 0]), R.get_ring().apply_galois_action(&ring_literal!(&R, [0, 1, 0, 0, 0, 0, 0, 0]), hom.map(2)));
    assert_el_eq!(R, ring_literal!(&R, [0, 0, 0, 0, 1, 0, 0, 0]), R.get_ring().apply_galois_action(&ring_literal!(&R, [0, 1, 0, 0, 0, 0, 0, 0]), hom.map(4)));
    assert_el_eq!(R, ring_literal!(&R, [-1, 1, 0, -1, 1, -1, 0, 1]), R.get_ring().apply_galois_action(&ring_literal!(&R, [0, 1, 0, 0, 0, 0, 0, 0]), hom.map(8)));
    assert_el_eq!(R, ring_literal!(&R, [-1, 1, 0, -1, 1, -1, 0, 1]), R.get_ring().apply_galois_action(&ring_literal!(&R, [0, 0, 0, 0, 1, 0, 0, 0]), hom.map(2)));
}
