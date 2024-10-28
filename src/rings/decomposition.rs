use feanor_math::divisibility::DivisibilityRing;
use feanor_math::integer::IntegerRing;
use feanor_math::ring::*;
use feanor_math::rings::extension::FreeAlgebra;
use feanor_math::rings::float_complex::Complex64;
use feanor_math::rings::poly::PolyRing;
use feanor_math::rings::zn::zn_64::{self, Zn, ZnEl};
use feanor_math::rings::zn::ZnRing;

use crate::cyclotomic::CyclotomicRing;
use crate::IsEq;

///
/// Trait for objects that represent number rings `R = Z[X]/(f)` by the isomorphisms
/// ```text
///   R/(p) -> Fp^deg(f)
/// ```
/// that exist whenever `p` is a prime that splits in `R`.
/// 
/// As usually, an object of this trait describes a number ring together with a canonical
/// generator (or equivalently, the polynomial `f`). Two objects describing the same number
/// ring, but with different generators are considered to be inequal.
/// 
pub trait DecomposableNumberRing<R>: PartialEq
    where R: RingStore,
        R::Type: ZnRing
{
    type Decomposed: DecomposedNumberRing<R::Type>;

    fn mod_p(&self, Fp: R) -> Self::Decomposed;

    fn largest_suitable_prime(&self, leq_than: i64) -> Option<i64>;

    ///
    /// Returns an upper bound on the value
    /// ```text
    ///   sup_(x in R \ {0}) | x |_can / | x |_inf
    /// ```
    /// 
    /// Note that while the canonical norm `|.|_can` depends only on the
    /// number ring `R`, the infinity norm refers to the infinity norm
    /// in the coefficient-representation w.r.t. the canonical generator,
    /// and thus depends on the canonical generator.
    /// 
    fn inf_to_can_norm_expansion_factor(&self) -> f64;

    ///
    /// Returns an upper bound on the value
    /// ```text
    ///   sup_(x in R \ {0}) | x |_inf / | x |_can
    /// ```
    /// 
    /// Note that while the canonical norm `|.|_can` depends only on the
    /// number ring `R`, the infinity norm refers to the infinity norm
    /// in the coefficient-representation w.r.t. the canonical generator,
    /// and thus depends on the canonical generator.
    /// 
    fn can_to_inf_norm_expansion_factor(&self) -> f64;

    ///
    /// Returns an upper bound on the value
    /// ```text
    ///   sup_(x, y in R \ {0}) | xy |_inf / (| x |_inf | y |_inf)
    /// ```
    /// 
    fn product_expansion_factor(&self) -> f64 {
        self.inf_to_can_norm_expansion_factor().powi(2) * self.can_to_inf_norm_expansion_factor()
    }

    fn generating_poly<P>(&self, poly_ring: P) -> El<P>
        where P: RingStore,
            P::Type: PolyRing + DivisibilityRing,
            <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing;

    fn rank(&self) -> usize;
}

pub trait DecomposableCyclotomicNumberRing<R>: DecomposableNumberRing<R>
    where R: RingStore,
        R::Type: ZnRing
{    
    type DecomposedAsCyclotomic: DecomposedCyclotomicNumberRing<R::Type> + IsEq<Self::Decomposed>;

    fn n(&self) -> u64;

    fn cyclotomic_index_ring(&self) -> zn_64::Zn {
        zn_64::Zn::new(self.n())
    }
}

pub trait CCEmbeddedNumberRing<R>: DecomposableNumberRing<R>
    where R: RingStore,
        R::Type: ZnRing
{
    fn fft_forward(&self, data: &mut [El<Complex64>]);

    fn fft_backward(&self, data: &mut [El<Complex64>]);
}

///
/// A [`DecomposableNumberRing`] `R` modulo a prime `p` that splits completely in `R`.
/// 
/// This object may define up to three different basis of `R / p`, with the following
/// properties:
///  - the "small basis" should consist of elements whose shortest lift to `R` has small
///    canonical norm
///  - the "mult basis" should allow for component-wise multiplication, i.e. `bi * bi = bi`
///    and `bi * bj = 0` for `i != j`
///  - the "coeff basis" should consist of powers of a generator of the ring, which for
///    cyclotomic rings should be the root of unity.
/// Both "small basis" and "coeff basis" should be the reduction of a corresponding
/// canonical basis of `R`.
/// 
/// Note that it is valid for any of these basis to coincide, and then implement the 
/// corresponding conversions as no-ops.
/// 
/// This design is motivated by the example of `Z[ζ_n]` for a composite `n`, since in
/// this case, we need three different basis.
///  - The "small basis" is the powerful basis `ζ^(n/n1 * i1 + ... + n/nr * ir)` with
///    `0 <= ij < phi(nj)`, where `nj` runs through pairwise coprime factors of `n`
///  - The "mult basis" is the preimage of the unit vector basis under `Fp[ζ] -> Fp^phi(n)`
///  - The "coeff basis" is the basis `1, ζ, ζ^2, ..., ζ^phi(n)`
/// While one could choose "small basis" and "coeff basis" to be equal (after all, the
/// elements `ζ^i` are all "small"), staying in "small basis" whenever possible has
/// performance benefits, because of the tensor-decomposition.
/// 
pub trait DecomposedNumberRing<R: ?Sized + ZnRing>: PartialEq {

    fn base_ring(&self) -> RingRef<R>;

    fn rank(&self) -> usize;

    fn small_basis_to_mult_basis(&self, data: &mut [R::Element]);

    fn mult_basis_to_small_basis(&self, data: &mut [R::Element]);

    fn coeff_basis_to_small_basis(&self, data: &mut [R::Element]);

    fn small_basis_to_coeff_basis(&self, data: &mut [R::Element]);
}

pub trait DecomposedCyclotomicNumberRing<R: ?Sized + ZnRing>: DecomposedNumberRing<R> {

    fn n(&self) -> u64;

    fn cyclotomic_index_ring(&self) -> zn_64::Zn {
        zn_64::Zn::new(self.n())
    }

    ///
    /// Permutes the components of an element w.r.t. the mult basis to
    /// obtain its image under the given Galois action.
    /// 
    fn permute_galois_action(&self, src: &[R::Element], dst: &mut [R::Element], galois_element: zn_64::ZnEl);
}