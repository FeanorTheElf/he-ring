use feanor_math::ring::*;
use feanor_math::rings::zn::zn_64::{Zn, ZnEl};

///
/// A number ring `R` with a canonical generator `a`, with arithmetic modulo a prime `p`.
/// This must be given by a way to compute the the decomposition of `R/pR` into prime fields `Fp`. 
/// This of course requires that `f` splits modulo `p`.
/// 
pub trait RingDecomposition<R: ?Sized + RingBase>: PartialEq {

    ///
    /// Rank of the ring `R`
    /// 
    fn rank(&self) -> usize;

    ///
    /// Value `C > 0` such that
    /// ```text
    /// (a0 + a1 X + ... + a(n - 1) X^(n - 1)) * (b0 + b1 X + ... + b(n - 1) X^(n - 1)) mod f
    /// ```
    /// has coefficients bounded by `C * max |ai| * max |bi|` in absolute value.
    /// Note that this depends only on the generating polynomial `f` of `R` and not on `p`.
    /// 
    /// Used for size estimation when implementing [`super::ntt_ring::NTTRing`]. Please
    /// don't use it for error estimation during FHE, since using the canonical norm will
    /// give tighter bounds.
    ///  
    fn expansion_factor(&self) -> i64;

    ///
    /// Computes the decomposition isomorphism
    /// ```text
    /// Z[X]/(f(X), p) -> Zp x ... x Zp,    g -> (g(x))_x where f(x) = 0
    /// ```
    /// For a more detailed explanation, see the trait-level doc [`RingDecomposition`].
    /// 
    fn fft_forward(&self, data: &mut [R::Element], ring: &R);

    ///
    /// Computes the inverse of [`RingDecomposition::fft_forward()`].
    /// 
    fn fft_backward(&self, data: &mut [R::Element], ring: &R);
}

pub trait IsomorphismInfo<R1: ?Sized + RingBase, R2: ?Sized + RingBase, F: RingDecomposition<R2>>: RingDecomposition<R1> {

    ///
    /// Returns if the number ring `R` and the generator `a` (inducing the generating polynomial `f`)
    /// are the same for both rings.
    /// 
    /// This considers neither the prime modulus `p` nor the the implementation of the decomposition isomorphism
    /// ```text
    /// Z[X]/(f(X), p) -> Zp x ... x Zp,    g -> (g(x))_x where f(x) = 0
    /// ```
    /// 
    fn is_same_number_ring(&self, other: &F) -> bool;
}

pub trait RingDecompositionSelfIso<R: ?Sized + RingBase>: Sized + PartialEq + IsomorphismInfo<R, R, Self> {}

impl<R: ?Sized + RingBase, F: RingDecomposition<R> + IsomorphismInfo<R, R, F>> RingDecompositionSelfIso<R> for F {}

pub trait CyclotomicRingDecomposition<R: ?Sized + RingBase>: RingDecomposition<R> {

    ///
    /// Returns `Z/nZ` such that the galois group of this number ring
    /// is `(Z/nZ)*`
    /// 
    fn galois_group_mulrepr(&self) -> Zn;

    fn permute_galois_action<S>(&self, src: &[R::Element], dst: &mut [R::Element], galois_element: ZnEl, ring: S)
        where S: RingStore<Type = R>;
}