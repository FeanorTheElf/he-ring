use std::alloc::{Allocator, Global};
use std::collections::{BTreeMap, HashMap};
use std::cmp::max;

use bluestein::BluesteinFFT;
use dense_poly::DensePolyRing;
use factor_fft::CoprimeCooleyTuckeyFFT;
use feanor_math::algorithms::eea::{signed_eea, signed_gcd};
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
use subvector::SubvectorView;
use zn_static::Fp;
use crate::feanor_math::rings::extension::*;
use feanor_math::seq::*;

use crate::rings::double_rns_ring::*;
use crate::rings::number_ring::*;
use crate::rings::single_rns_ring;
use super::{decomposition_ring::{self, *}, double_rns_ring};
use crate::{euler_phi, euler_phi_squarefree, sample_primes, StdZn};
use crate::cyclotomic::CyclotomicRing;

pub struct OddCyclotomicNumberRing {
    n_factorization_squarefree: Vec<i64>,
    sparse_poly_ring: SparsePolyRing<BigIntRing>,
    cyclotomic_poly: El<SparsePolyRing<BigIntRing>>
}

impl OddCyclotomicNumberRing {

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
}

impl Clone for OddCyclotomicNumberRing {
    fn clone(&self) -> Self {
        Self {
            cyclotomic_poly: self.sparse_poly_ring.clone_el(&self.cyclotomic_poly),
            sparse_poly_ring: self.sparse_poly_ring.clone(),
            n_factorization_squarefree: self.n_factorization_squarefree.clone()
        }
    }
}

impl PartialEq for OddCyclotomicNumberRing {

    fn eq(&self, other: &Self) -> bool {
        self.n_factorization_squarefree == other.n_factorization_squarefree
    }
}

impl<FpTy> HENumberRing<FpTy> for OddCyclotomicNumberRing
    where FpTy: RingStore + Clone,
        FpTy::Type: ZnRing
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
        record_time!("OddCyclotomicDecomposedNumberRing::mod_p", || {
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
        })
    }

    fn largest_suitable_prime(&self, leq_than: i64) -> Option<i64> {
        let n = <_ as HECyclotomicNumberRing<FpTy>>::n(self) as i64;
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
        cyclotomic_polynomial(&poly_ring, <_ as HECyclotomicNumberRing<FpTy>>::n(self) as usize)
    }

    fn rank(&self) -> usize {
        euler_phi_squarefree(&self.n_factorization_squarefree) as usize
    }
}

impl<FpTy> HECyclotomicNumberRing<FpTy> for OddCyclotomicNumberRing
    where FpTy: RingStore + Clone,
        FpTy::Type: ZnRing
{
    type DecomposedAsCyclotomic = OddCyclotomicDecomposedNumberRing<FpTy, BluesteinFFT<FpTy::Type, FpTy::Type, Identity<FpTy>>>;

    fn n(&self) -> u64 {
        self.n_factorization_squarefree.iter().copied().product::<i64>() as u64
    }
}

#[derive(Clone)]
pub struct CompositeCyclotomicNumberRing {
    tensor_factor1: OddCyclotomicNumberRing,
    tensor_factor2: OddCyclotomicNumberRing
}

impl CompositeCyclotomicNumberRing {

    pub fn new(n1: usize, n2: usize) -> Self {
        assert!(n1 % 2 == 1);
        assert!(n2 % 2 == 1);
        assert!(n1 > 1);
        assert!(n2 > 1);
        assert!(signed_gcd(n1 as i64, n2 as i64, StaticRing::<i64>::RING) == 1);
        Self {
            tensor_factor1: OddCyclotomicNumberRing::new(n1),
            tensor_factor2: OddCyclotomicNumberRing::new(n2)
        }
    }
}

impl PartialEq for CompositeCyclotomicNumberRing {

    fn eq(&self, other: &Self) -> bool {
        self.tensor_factor1 == other.tensor_factor1 && self.tensor_factor2 == other.tensor_factor2
    }
}

impl<FpTy> HECyclotomicNumberRing<FpTy> for CompositeCyclotomicNumberRing
    where FpTy: RingStore + Clone,
        FpTy::Type: ZnRing
{
    type DecomposedAsCyclotomic = CompositeCyclotomicDecomposedNumberRing<FpTy, BluesteinFFT<FpTy::Type, FpTy::Type, Identity<FpTy>>>;

    fn n(&self) -> u64 {
        <_ as HECyclotomicNumberRing<FpTy>>::n(&self.tensor_factor1) * <_ as HECyclotomicNumberRing<FpTy>>::n(&self.tensor_factor2)
    }
}

impl<FpTy> HENumberRing<FpTy> for CompositeCyclotomicNumberRing
    where FpTy: RingStore + Clone,
        FpTy::Type: ZnRing
{
    type Decomposed = CompositeCyclotomicDecomposedNumberRing<FpTy, BluesteinFFT<FpTy::Type, FpTy::Type, Identity<FpTy>>>;

    fn mod_p(&self, Fp: FpTy) -> Self::Decomposed {
        record_time!("CompositeCyclotomicNumberRing::mod_p", || {
            let n1 = <_ as HECyclotomicNumberRing<FpTy>>::n(&self.tensor_factor1) as i64;
            let n2 = <_ as HECyclotomicNumberRing<FpTy>>::n(&self.tensor_factor2) as i64;
            let n = n1 * n2;
            let r1 = <_ as HENumberRing<FpTy>>::rank(&self.tensor_factor1) as i64;
            let r2 = <_ as HENumberRing<FpTy>>::rank(&self.tensor_factor2) as i64;

            let poly_ring = &self.tensor_factor1.sparse_poly_ring;
            let Phi_n1 = poly_ring.coerce(&self.tensor_factor1.sparse_poly_ring, self.tensor_factor1.sparse_poly_ring.clone_el(&self.tensor_factor1.cyclotomic_poly));
            let Phi_n2 = poly_ring.coerce(&self.tensor_factor2.sparse_poly_ring, self.tensor_factor2.sparse_poly_ring.clone_el(&self.tensor_factor2.cyclotomic_poly));
            let Phi_n = cyclotomic_polynomial(&poly_ring, n as usize);
            let hom = Fp.can_hom(Fp.integer_ring()).unwrap().compose(Fp.integer_ring().can_hom(poly_ring.base_ring()).unwrap());
            let hom_ref = &hom;

            let (s, t, d) = signed_eea(n1, n2, StaticRing::<i64>::RING);
            assert_eq!(1, d);

            let mut small_to_coeff_conversion_matrix = (0..(r1 * r2)).map(|_| Vec::new()).collect::<Vec<_>>();
            let mut coeff_to_small_conversion_matrix = (0..(r1 * r2)).map(|_| Vec::new()).collect::<Vec<_>>();

            let mut X_pow_i = poly_ring.one();
            for i in 0..(n1 * n2) {
                let i1 = ((t * i % n1) + n1) % n1;
                let i2 = ((s * i % n2) + n2) % n2;
                debug_assert_eq!(i, (i1 * n / n1 + i2 * n / n2) % n);

                if i < r1 * r2 {
                    let X1_power_reduced = poly_ring.div_rem_monic(poly_ring.pow(poly_ring.indeterminate(), i1 as usize), &Phi_n1).1;
                    let X2_power_reduced = poly_ring.div_rem_monic(poly_ring.pow(poly_ring.indeterminate(), i2 as usize), &Phi_n2).1;
                    
                    coeff_to_small_conversion_matrix[i as usize] = poly_ring.terms(&X1_power_reduced).flat_map(|(c1, j1)| poly_ring.terms(&X2_power_reduced).map(move |(c2, j2)| 
                        (j1 + j2 * r1 as usize, hom_ref.map(poly_ring.base_ring().mul_ref(c1, c2))
                    ))).collect::<Vec<_>>();
                }
                
                if i1 < r1 && i2 < r2 {
                    small_to_coeff_conversion_matrix[(i2 * r1 + i1) as usize] = poly_ring.terms(&X_pow_i).map(|(c, i)| (i, hom_ref.map_ref(c))).collect::<Vec<_>>();
                }

                record_time!("CompositeCyclotomicNumberRing::mod_p::mul_mod", || {
                    poly_ring.mul_assign(&mut X_pow_i, poly_ring.indeterminate());
                    X_pow_i = poly_ring.div_rem_monic(std::mem::replace(&mut X_pow_i, poly_ring.zero()), &Phi_n).1;
                });
            }

            CompositeCyclotomicDecomposedNumberRing {
                small_to_coeff_conversion_matrix: small_to_coeff_conversion_matrix,
                coeff_to_small_conversion_matrix: coeff_to_small_conversion_matrix,
                tensor_factor1: self.tensor_factor1.mod_p(Fp.clone()),
                tensor_factor2: self.tensor_factor2.mod_p(Fp)
            }
        })
    }

    fn largest_suitable_prime(&self, leq_than: i64) -> Option<i64> {
        let n = <_ as HECyclotomicNumberRing<FpTy>>::n(self) as i64;
        let log2_m = max(
            StaticRing::<i64>::RING.abs_log2_ceil(&(<_ as HECyclotomicNumberRing<FpTy>>::n(&self.tensor_factor1) as i64)).unwrap() + 1,
            StaticRing::<i64>::RING.abs_log2_ceil(&(<_ as HECyclotomicNumberRing<FpTy>>::n(&self.tensor_factor2) as i64)).unwrap() + 1
        );
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

    fn inf_to_can_norm_expansion_factor(&self) -> f64 {
        return <_ as HENumberRing<FpTy>>::inf_to_can_norm_expansion_factor(&self.tensor_factor1) * <_ as HENumberRing<FpTy>>::inf_to_can_norm_expansion_factor(&self.tensor_factor2);
    }

    fn can_to_inf_norm_expansion_factor(&self) -> f64 {
        return <_ as HENumberRing<FpTy>>::can_to_inf_norm_expansion_factor(&self.tensor_factor1) * <_ as HENumberRing<FpTy>>::can_to_inf_norm_expansion_factor(&self.tensor_factor2);
    }

    fn generating_poly<P>(&self, poly_ring: P) -> El<P>
        where P: RingStore,
            P::Type: PolyRing + DivisibilityRing,
            <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing
    {
        cyclotomic_polynomial(&poly_ring, <_ as HECyclotomicNumberRing<FpTy>>::n(self) as usize)
    }

    fn rank(&self) -> usize {
        <_ as HENumberRing<FpTy>>::rank(&self.tensor_factor1) * <_ as HENumberRing<FpTy>>::rank(&self.tensor_factor2)
    }
}

pub struct OddCyclotomicDecomposedNumberRing<R, F, A = Global> 
    where R: RingStore,
        R::Type: ZnRing + DivisibilityRing,
        F: FFTAlgorithm<R::Type> + PartialEq,
        A: Allocator + Clone
{
    ring: R,
    fft_table: F,
    /// contains `usize::MAX` whenenver the fft output index corresponds to a non-primitive root of unity, and an index otherwise
    fft_output_indices_to_indices: Vec<usize>,
    zeta_pow_rank: Vec<(usize, El<R>)>,
    rank: usize,
    allocator: A
}

impl<R, F, A> PartialEq for OddCyclotomicDecomposedNumberRing<R, F, A> 
    where R: RingStore,
        R::Type: ZnRing + DivisibilityRing,
        F: FFTAlgorithm<R::Type> + PartialEq,
        A: Allocator + Clone
{
    fn eq(&self, other: &Self) -> bool {
        self.ring.get_ring() == other.ring.get_ring() && self.fft_table == other.fft_table
    }
}

impl<R, F, A> OddCyclotomicDecomposedNumberRing<R, F, A> 
    where R: RingStore,
        R::Type: ZnRing + DivisibilityRing,
        F: FFTAlgorithm<R::Type> + PartialEq,
        A: Allocator + Clone
{
    fn create_squarefree(fft_table: F, Fp: R, n_factorization: &[i64], allocator: A) -> Self {
        let n = n_factorization.iter().copied().product::<i64>();
        let rank = euler_phi_squarefree(&n_factorization) as usize;

        let poly_ring = SparsePolyRing::new(&Fp, "X");
        let cyclotomic_poly = cyclotomic_polynomial(&poly_ring, n as usize);
        assert_eq!(poly_ring.degree(&cyclotomic_poly).unwrap(), rank);
        let mut zeta_pow_rank = Vec::new();
        for (a, i) in poly_ring.terms(&cyclotomic_poly) {
            if i != rank {
                zeta_pow_rank.push((i, Fp.negate(Fp.clone_el(a))));
            }
        }
        zeta_pow_rank.sort_unstable_by_key(|(i, _)| *i);

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
    fn fft_output_indices<'a>(&'a self) -> impl Iterator<Item = (usize, usize)> + 'a {
        self.fft_output_indices_to_indices.iter().enumerate().filter_map(|(i, j)| if *j == usize::MAX { None } else { Some((*j, i)) })
    }
}

impl<R, F, A> HENumberRingMod<R::Type> for OddCyclotomicDecomposedNumberRing<R, F , A> 
    where R: RingStore,
        R::Type: ZnRing + DivisibilityRing,
        F: FFTAlgorithm<R::Type> + PartialEq,
        A: Allocator + Clone
{
    fn mult_basis_to_small_basis<V>(&self, mut data: V)
        where V: VectorViewMut<El<R>>
    {
        let ring = self.base_ring();
        let mut tmp = Vec::with_capacity_in(self.fft_table.len(), &self.allocator);
        tmp.extend((0..self.fft_table.len()).map(|_| ring.zero()));
        for (i, j) in self.fft_output_indices() {
            tmp[j] = ring.clone_el(data.at(i));
        }

        record_time!("OddCyclotomicDecomposedNumberRing::mult_basis_to_small_basis::fft", || {
            self.fft_table.unordered_inv_fft(&mut tmp[..], ring);
        });

        record_time!("OddCyclotomicDecomposedNumberRing::mult_basis_to_small_basis::reduction", || {
            for i in (self.rank()..self.fft_table.len()).rev() {
                let factor = ring.clone_el(&tmp[i]);
                for (j, c) in self.zeta_pow_rank.iter() {
                    let mut add = ring.clone_el(&factor);
                    ring.mul_assign_ref(&mut add, c);
                    ring.add_assign(&mut tmp[i - self.rank() + *j], add);
                }
            }
        });

        for i in 0..self.rank() {
            *data.at_mut(i) = ring.clone_el(&tmp[i]);
        }
    }

    fn small_basis_to_mult_basis<V>(&self, mut data: V)
        where V: VectorViewMut<El<R>>
    {
        let ring = self.base_ring();
        let mut tmp = Vec::with_capacity_in(self.fft_table.len(), self.allocator.clone());
        tmp.extend((0..self.fft_table.len()).map(|_| ring.zero()));
        for i in 0..self.rank() {
            tmp[i] = ring.clone_el(data.at(i));
        }

        record_time!("OddCyclotomicDecomposedNumberRing::small_basis_to_mult_basis::fft", || {
            self.fft_table.unordered_fft(&mut tmp[..], ring);
        });
        for (i, j) in self.fft_output_indices() {
            *data.at_mut(i) = ring.clone_el(&tmp[j]); 
        }
    }

    fn coeff_basis_to_small_basis<V>(&self, data: V)
        where V: VectorViewMut<El<R>>
    {}

    fn small_basis_to_coeff_basis<V>(&self, data: V)
        where V: VectorViewMut<El<R>>
    {}

    fn rank(&self) -> usize {
        self.rank
    }

    fn base_ring(&self) -> RingRef<R::Type> {
        RingRef::new(self.ring.get_ring())
    }
}

impl<R, F, A> HECyclotomicNumberRingMod<R::Type> for OddCyclotomicDecomposedNumberRing<R, F , A> 
    where R: RingStore,
        R::Type: ZnRing + DivisibilityRing,
        F: FFTAlgorithm<R::Type> + PartialEq,
        A: Allocator + Clone
{
    fn n(&self) -> u64 {
        self.fft_table.len() as u64
    }

    fn permute_galois_action<V1, V2>(&self, src: V1, mut dst: V2, galois_element: zn_64::ZnEl)
        where V1: VectorView<El<R>>,
            V2: SwappableVectorViewMut<El<R>>
    {
        assert_eq!(self.rank(), src.len());
        assert_eq!(self.rank(), dst.len());
        let ring = self.base_ring();
        let index_ring = self.cyclotomic_index_ring();
        let hom = index_ring.can_hom(&StaticRing::<i64>::RING).unwrap();
        
        for (j, i) in self.fft_output_indices() {
            *dst.at_mut(j) = ring.clone_el(src.at(self.fft_output_indices_to_indices[self.fft_table.unordered_fft_permutation_inv(
                index_ring.smallest_positive_lift(index_ring.mul(galois_element, hom.map(self.fft_table.unordered_fft_permutation(i) as i64))) as usize
            )]));
        }
    }
}

///
/// The small basis is given by 
/// ```text
///   1 ⊗ 1,            ζ1 ⊗ 1,            ζ1^2 ⊗ 1,           ...,  ζ1^(n1 - 1) ⊗ 1,
///   1 ⊗ ζ2,           ζ1 ⊗ ζ2,           ζ1^2 ⊗ ζ2,          ...,  ζ1^(n1 - 1) ⊗ ζ2,
///   ...
///   1 ⊗ ζ2^(n2 - 1),  ζ1 ⊗ ζ2^(n2 - 1),  ζ1^2 ⊗ ζ2^(n2 - 1), ...,  ζ1^(n1 - 1) ⊗ ζ2^(n2 - 1)
/// ```
/// 
pub struct CompositeCyclotomicDecomposedNumberRing<R, F, A = Global> 
    where R: RingStore,
        R::Type: ZnRing + DivisibilityRing,
        F: FFTAlgorithm<R::Type> + PartialEq,
        A: Allocator + Clone
{
    tensor_factor1: OddCyclotomicDecomposedNumberRing<R, F, A>,
    tensor_factor2: OddCyclotomicDecomposedNumberRing<R, F, A>,
    // the `i`-th entry is none if the `i`-th small basis vector equals the `i`-th coeff basis vector,
    // and otherwise, it contains the coeff basis representation of the `i`-th small basis vector
    small_to_coeff_conversion_matrix: Vec<Vec<(usize, El<R>)>>,
    // same as `small_to_coeff_conversion_matrix` but with small and coeff basis swapped
    coeff_to_small_conversion_matrix: Vec<Vec<(usize, El<R>)>>
}

impl<R, F, A> PartialEq for CompositeCyclotomicDecomposedNumberRing<R, F, A> 
    where R: RingStore,
        R::Type: ZnRing + DivisibilityRing,
        F: FFTAlgorithm<R::Type> + PartialEq,
        A: Allocator + Clone
{
    fn eq(&self, other: &Self) -> bool {
        self.tensor_factor1 == other.tensor_factor1 && self.tensor_factor2 == other.tensor_factor2
    }
}

impl<R, F, A> HENumberRingMod<R::Type> for CompositeCyclotomicDecomposedNumberRing<R, F, A> 
    where R: RingStore,
        R::Type: ZnRing + DivisibilityRing,
        F: FFTAlgorithm<R::Type> + PartialEq,
        A: Allocator + Clone
{
    fn small_basis_to_mult_basis<V>(&self, mut data: V)
        where V: SwappableVectorViewMut<El<R>>
    {
        record_time!("CompositeCyclotomicDecomposedNumberRing::small_basis_to_mult_basis", || {
            for i in 0..self.tensor_factor2.rank() {
                self.tensor_factor1.small_basis_to_mult_basis(SubvectorView::new(&mut data).restrict((i * self.tensor_factor1.rank())..((i + 1) * self.tensor_factor1.rank())));
            }
            for j in 0..self.tensor_factor1.rank() {
                self.tensor_factor2.small_basis_to_mult_basis(SubvectorView::new(&mut data).restrict(j..).step_by(self.tensor_factor1.rank()));
            }
        })
    }

    fn mult_basis_to_small_basis<V>(&self, mut data: V)
        where V: SwappableVectorViewMut<El<R>>
    {
        record_time!("CompositeCyclotomicDecomposedNumberRing::mult_basis_to_small_basis", || {
            for j in 0..self.tensor_factor1.rank() {
                self.tensor_factor2.mult_basis_to_small_basis(SubvectorView::new(&mut data).restrict(j..).step_by(self.tensor_factor1.rank()));
            }
            for i in 0..self.tensor_factor2.rank() {
                self.tensor_factor1.mult_basis_to_small_basis(SubvectorView::new(&mut data).restrict((i * self.tensor_factor1.rank())..((i + 1) * self.tensor_factor1.rank())));
            }
        })
    }

    fn coeff_basis_to_small_basis<V>(&self, mut data: V)
        where V: SwappableVectorViewMut<El<R>>
    {
        let mut result = Vec::with_capacity_in(self.rank(), &self.tensor_factor1.allocator);
        result.resize_with(self.rank(), || self.base_ring().zero());
        for i in 0..self.rank() {
            for (j, c) in &self.coeff_to_small_conversion_matrix[i] {
                self.base_ring().add_assign(&mut result[*j], self.base_ring().mul_ref(data.at(i), c));
            }
        }
        for (i, c) in result.drain(..).enumerate() {
            *data.at_mut(i) = c;
        }
    }

    fn small_basis_to_coeff_basis<V>(&self, mut data: V)
        where V: SwappableVectorViewMut<El<R>>
    {
        let mut result = Vec::with_capacity_in(self.rank(), &self.tensor_factor1.allocator);
        result.resize_with(self.rank(), || self.base_ring().zero());
        for i in 0..self.rank() {
            for (j, c) in &self.small_to_coeff_conversion_matrix[i] {
                self.base_ring().add_assign(&mut result[*j], self.base_ring().mul_ref(data.at(i), c));
            }
        }
        for (i, c) in result.drain(..).enumerate() {
            *data.at_mut(i) = c;
        }
    }

    fn rank(&self) -> usize {
        self.tensor_factor1.rank() * self.tensor_factor2.rank()
    }

    fn base_ring(&self) -> RingRef<R::Type> {
        self.tensor_factor1.base_ring()
    }
}

impl<R, F, A> HECyclotomicNumberRingMod<R::Type> for CompositeCyclotomicDecomposedNumberRing<R, F , A> 
    where R: RingStore,
        R::Type: ZnRing + DivisibilityRing,
        F: FFTAlgorithm<R::Type> + PartialEq,
        A: Allocator + Clone
{
    fn n(&self) -> u64 {
        self.tensor_factor1.n() * self.tensor_factor2.n()
    }

    fn permute_galois_action<V1, V2>(&self, src: V1, mut dst: V2, galois_element: zn_64::ZnEl)
        where V1: VectorView<El<R>>,
            V2: SwappableVectorViewMut<El<R>>
    {
        let index_ring = self.cyclotomic_index_ring();
        let ring_factor1 = self.tensor_factor1.cyclotomic_index_ring();
        let ring_factor2 = self.tensor_factor2.cyclotomic_index_ring();

        let g1 = ring_factor1.can_hom(ring_factor1.integer_ring()).unwrap().map(index_ring.smallest_lift(galois_element));
        let g2 = ring_factor2.can_hom(ring_factor2.integer_ring()).unwrap().map(index_ring.smallest_lift(galois_element));
        let mut tmp = Vec::with_capacity_in(self.rank(), &self.tensor_factor1.allocator);
        tmp.resize_with(self.rank(), || self.base_ring().zero());
        for i in 0..self.tensor_factor2.rank() {
            self.tensor_factor1.permute_galois_action(SubvectorView::new(&src).restrict((i * self.tensor_factor1.rank())..((i + 1) * self.tensor_factor1.rank())), &mut tmp[(i * self.tensor_factor1.rank())..((i + 1) * self.tensor_factor1.rank())], g1);
        }
        for j in 0..self.tensor_factor1.rank() {
            self.tensor_factor2.permute_galois_action(SubvectorView::new(&tmp[..]).restrict(j..).step_by(self.tensor_factor1.rank()), SubvectorView::new(&mut dst).restrict(j..).step_by(self.tensor_factor1.rank()), g2);
        }
    }
}

#[cfg(test)]
use feanor_math::assert_el_eq;

#[test]
fn test_odd_cyclotomic_double_rns_ring() {
    double_rns_ring::test_with_number_ring(OddCyclotomicNumberRing::new(5));
    double_rns_ring::test_with_number_ring(OddCyclotomicNumberRing::new(7));
    double_rns_ring::test_with_number_ring(CompositeCyclotomicNumberRing::new(3, 5));
    double_rns_ring::test_with_number_ring(CompositeCyclotomicNumberRing::new(3, 7));
}

#[test]
fn test_odd_cyclotomic_decomposition_ring() {
    decomposition_ring::test_with_number_ring(OddCyclotomicNumberRing::new(5));
    decomposition_ring::test_with_number_ring(OddCyclotomicNumberRing::new(7));
    let ring = CompositeCyclotomicNumberRing::new(3, 5);
    decomposition_ring::test_with_number_ring(ring);
    decomposition_ring::test_with_number_ring(CompositeCyclotomicNumberRing::new(3, 7));
}

#[test]
fn test_small_coeff_basis_conversion() {
    let ring = Fp::<241>::RING;
    let number_ring = CompositeCyclotomicNumberRing::new(3, 5);
    let decomposition = number_ring.mod_p(ring);

    let original = [1, 0, 0, 0, 0, 0, 0, 0];
    let expected = [1, 0, 0, 0, 0, 0, 0, 0];
    let mut actual = original;
    decomposition.coeff_basis_to_small_basis(&mut actual);
    assert_eq!(expected, actual);
    decomposition.small_basis_to_coeff_basis(&mut actual);
    assert_eq!(original, actual);
    
    // ζ_15 = ζ_3^-1 ⊗ ζ_5^2 = (-1 - ζ_3) ⊗ ζ_5^2
    let original = [0, 1, 0, 0, 0, 0, 0, 0];
    let expected = [0, 0, 0, 0, 240, 240, 0, 0];
    let mut actual = original;
    decomposition.coeff_basis_to_small_basis(&mut actual);
    assert_eq!(expected, actual);
    decomposition.small_basis_to_coeff_basis(&mut actual);
    assert_eq!(original, actual);

    let original = [0, 0, 240, 0, 0, 0, 0, 0];
    let expected = [0, 1, 0, 1, 0, 1, 0, 1];
    let mut actual = original;
    decomposition.coeff_basis_to_small_basis(&mut actual);
    assert_eq!(expected, actual);
    decomposition.small_basis_to_coeff_basis(&mut actual);
    assert_eq!(original, actual);

    let original = [0, 0, 0, 1, 0, 0, 0, 0];
    let expected = [0, 0, 1, 0, 0, 0, 0, 0];
    let mut actual = original;
    decomposition.coeff_basis_to_small_basis(&mut actual);
    assert_eq!(expected, actual);
    decomposition.small_basis_to_coeff_basis(&mut actual);
    assert_eq!(original, actual);

    let original = [0, 0, 0, 0, 0, 1, 0, 0];
    let expected = [0, 1, 0, 0, 0, 0, 0, 0];
    let mut actual = original;
    decomposition.coeff_basis_to_small_basis(&mut actual);
    assert_eq!(expected, actual);
    decomposition.small_basis_to_coeff_basis(&mut actual);
    assert_eq!(original, actual);
}

#[test]
fn test_odd_cyclotomic_single_rns_ring() {
    single_rns_ring::test_with_number_ring(OddCyclotomicNumberRing::new(5));
    single_rns_ring::test_with_number_ring(OddCyclotomicNumberRing::new(7));
    single_rns_ring::test_with_number_ring(CompositeCyclotomicNumberRing::new(3, 5));
    single_rns_ring::test_with_number_ring(CompositeCyclotomicNumberRing::new(3, 7));
}

#[test]
fn test_permute_galois_automorphism() {
    let Fp = zn_64::Zn::new(257);
    let R = DecompositionRingBase::new(OddCyclotomicNumberRing::new(7), Fp);
    let hom = R.get_ring().cyclotomic_index_ring().into_int_hom();

    assert_el_eq!(R, ring_literal!(&R, [0, 0, 1, 0, 0, 0]), R.get_ring().apply_galois_action(&ring_literal!(&R, [0, 1, 0, 0, 0, 0]), hom.map(2)));
    assert_el_eq!(R, ring_literal!(&R, [0, 0, 0, 1, 0, 0]), R.get_ring().apply_galois_action(&ring_literal!(&R, [0, 1, 0, 0, 0, 0]), hom.map(3)));
    assert_el_eq!(R, ring_literal!(&R, [0, 0, 0, 0, 1, 0]), R.get_ring().apply_galois_action(&ring_literal!(&R, [0, 0, 1, 0, 0, 0]), hom.map(2)));
    assert_el_eq!(R, ring_literal!(&R, [-1, -1, -1, -1, -1, -1]), R.get_ring().apply_galois_action(&ring_literal!(&R, [0, 0, 1, 0, 0, 0]), hom.map(3)));

    let R = DecompositionRingBase::new(CompositeCyclotomicNumberRing::new(5, 3), Fp);
    let hom = R.get_ring().cyclotomic_index_ring().into_int_hom();

    assert_el_eq!(R, ring_literal!(&R, [0, 0, 1, 0, 0, 0, 0, 0]), R.get_ring().apply_galois_action(&ring_literal!(&R, [0, 1, 0, 0, 0, 0, 0, 0]), hom.map(2)));
    assert_el_eq!(R, ring_literal!(&R, [0, 0, 0, 0, 1, 0, 0, 0]), R.get_ring().apply_galois_action(&ring_literal!(&R, [0, 1, 0, 0, 0, 0, 0, 0]), hom.map(4)));
    assert_el_eq!(R, ring_literal!(&R, [-1, 1, 0, -1, 1, -1, 0, 1]), R.get_ring().apply_galois_action(&ring_literal!(&R, [0, 1, 0, 0, 0, 0, 0, 0]), hom.map(8)));
    assert_el_eq!(R, ring_literal!(&R, [-1, 1, 0, -1, 1, -1, 0, 1]), R.get_ring().apply_galois_action(&ring_literal!(&R, [0, 0, 0, 0, 1, 0, 0, 0]), hom.map(2)));
}