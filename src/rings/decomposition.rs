use feanor_math::integer::IntegerRing;
use feanor_math::ring::*;
use feanor_math::rings::extension::FreeAlgebra;
use feanor_math::rings::zn::zn_64::{Zn, ZnEl};
use feanor_math::rings::zn::ZnRing;

use crate::cyclotomic::CyclotomicRing;

///
/// Trait for objects that represent number rings `R = Z[X]/(f)` by the isomorphisms
/// ```text
/// R/(p) -> Fp^deg(f)
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
}

///
/// A [`DecomposableNumberRing`] `R` modulo a prime `p` that splits completely in `R`.
/// 
pub trait DecomposedNumberRing<R: ?Sized + ZnRing>: PartialEq {

    fn base_ring(&self) -> RingRef<R>;

    fn rank(&self) -> usize;

    fn fft_forward(&self, data: &mut [R::Element]);

    fn fft_backward(&self, data: &mut [R::Element]);
}
