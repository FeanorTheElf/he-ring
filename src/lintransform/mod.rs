
use std::alloc::Allocator;

use feanor_math::algorithms::unity_root::is_prim_root_of_unity;
use feanor_math::homomorphism::CanHomFrom;
use feanor_math::homomorphism::Homomorphism;
use feanor_math::integer::IntegerRingStore;
use feanor_math::primitive_int::*;
use feanor_math::ring::*;
use feanor_math::divisibility::*;
use feanor_math::rings::extension::*;
use feanor_math::rings::zn::zn_64::*;
use feanor_math::rings::zn::*;
use feanor_math::rings::extension::galois_field::GaloisFieldDyn;
use feanor_math::assert_el_eq;

use crate::complexfft::automorphism::*;
use crate::complexfft::complex_fft_ring::*;
use crate::cyclotomic::*;

const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;

pub struct LinearTransform<R, F, A>
    where R: RingStore,
        R::Type: ZnRing,
        F: CyclotomicRingDecomposition<R::Type> + RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    coeffs: Vec<El<CCFFTRing<R, F, A>>>,
    galois_elements: Vec<ZnEl>
}

impl<R, F, A> CCFFTRingBase<R, F, A> 
    where R: RingStore,
        R::Type: ZnRing,
        F: RingDecompositionSelfIso<R::Type> + CyclotomicRingDecomposition<R::Type>,
        A: Allocator + Clone
{
    pub fn compute_linear_transform(&self, el: &<Self as RingBase>::Element, transform: &LinearTransform<R, F, A>) -> <Self as RingBase>::Element {
        <_ as RingBase>::sum(self, transform.coeffs.iter().zip(transform.galois_elements.iter()).map(|(c, s)| self.mul_ref_fst(c, self.apply_galois_action(*s, self.clone_el(el)))))
    }
}

///
/// Works separately on each block `(b0, ..., b(l - 1))` of size `l = blocksize` along the given given hypercube dimension.
/// This function computes the length-`l` DWT
/// ```text
/// sum_(0 <= i < l) ai * zeta_(4l)^(i * g^j)
/// ```
/// from the length-`l/2` DWTs of the even-index resp. odd-index entries of `ai`. These two sub-DWTs are expected to be written
/// in the first resp. second half of the input block (i.e. not interleaved, this is where the "bitreversed" comes from).
/// 
/// Here `g` is the generator of the current hypercube dimension, i.e. usually `g = 5` in the power of two case.
/// 
/// # Notes
///  - This does not compute the evaluation at all primitive `4l`-th roots of unity, but only at half of them - namely `zeta_(4l)^(g^j)` for all `j`.
///    In particular, `g` does not generate `(Z/4lZ)*`, but `<g>` is an index 2 subgroup of it.
/// 
fn pow2_bitreversed_dwt_butterfly<'b, R, F, A>(H: &HypercubeIsomorphism<'b, R, F, A>, dim_index: usize, l: usize, root_of_unity_4l: El<SlotRing<'b, R, A>>) -> LinearTransform<R, F, A>
    where R: RingStore,
        R::Type: ZnRing + CanHomFrom<<<<GaloisFieldDyn as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type>,
        F: CyclotomicRingDecomposition<R::Type> + RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone,
        CCFFTRingBase<R, F, A>: CyclotomicRing + /* unfortunately, the type checker is not clever enough to know that this is always the case */ RingExtension<BaseRing = R>
{
    let m = H.len(dim_index);
    let g = H.galois_group_mulrepr().invert(&H.galois_forward(dim_index, 1)).unwrap();

    let smaller_galois_group_mulrepr = Zn::new(4 * l as u64);
    let red = ReductionMap::new(H.galois_group_mulrepr(), &smaller_galois_group_mulrepr).unwrap();
    assert_el_eq!(&smaller_galois_group_mulrepr, &smaller_galois_group_mulrepr.one(), &smaller_galois_group_mulrepr.pow(red.map(g), l));

    let log2_m = ZZ.abs_log2_ceil(&(m as i64)).unwrap();
    assert!(m == 1 << log2_m, "pow2_bitreversed_cooley_tuckey_butterfly() only valid for hypercube dimensions that have a power-of-2 length");
    let l = l;
    assert!(l > 1);
    assert!(m % l == 0);
    let zeta = root_of_unity_4l;
    assert_el_eq!(H.slot_ring(), &H.slot_ring().neg_one(), &H.slot_ring().pow(H.slot_ring().clone_el(&zeta), 2 * l));

    enum TwiddleFactor {
        Zero, PosPowerZeta(ZnEl), NegPowerZeta(ZnEl)
    }

    let pow_of_zeta = |factor: TwiddleFactor| match factor {
        TwiddleFactor::PosPowerZeta(pow) => H.slot_ring().pow(H.slot_ring().clone_el(&zeta), H.galois_group_mulrepr().smallest_positive_lift(pow) as usize),
        TwiddleFactor::NegPowerZeta(pow) => H.slot_ring().negate(H.slot_ring().pow(H.slot_ring().clone_el(&zeta), H.galois_group_mulrepr().smallest_positive_lift(pow) as usize)),
        TwiddleFactor::Zero => H.slot_ring().zero()
    };

    let forward_galois_element = H.galois_forward(dim_index, l as i64 / 2);
    let backward_galois_element = H.galois_group_mulrepr().invert(&forward_galois_element).unwrap();

    let forward_mask = H.from_slot_vec(H.slot_iter(|idxs| {
        let idx_in_block = idxs[dim_index] % l;
        if idx_in_block >= l / 2 {
            TwiddleFactor::PosPowerZeta(H.galois_group_mulrepr().zero())
        } else {
            TwiddleFactor::Zero
        }
    }).map(&pow_of_zeta));

    let diagonal_mask = H.from_slot_vec(H.slot_iter(|idxs| {
        let idx_in_block = idxs[dim_index] % l;
        if idx_in_block >= l / 2 {
            TwiddleFactor::NegPowerZeta(H.galois_group_mulrepr().pow(g, idx_in_block - l / 2))
        } else {
            TwiddleFactor::PosPowerZeta(H.galois_group_mulrepr().zero())
        }
    }).map(&pow_of_zeta));

    let backward_mask = H.from_slot_vec(H.slot_iter(|idxs| {
        let idx_in_block = idxs[dim_index] % l;
        if idx_in_block < l / 2 {
            TwiddleFactor::PosPowerZeta(H.galois_group_mulrepr().pow(g, idx_in_block))
        } else {
            TwiddleFactor::Zero
        }
    }).map(&pow_of_zeta));

    return LinearTransform { 
        coeffs: vec![diagonal_mask, forward_mask, backward_mask], 
        galois_elements: vec![H.galois_group_mulrepr().one(), forward_galois_element, backward_galois_element]
    };
}

///
/// Computes the evaluation of `f(X) = a0 + a1 X + a2 X^2 + ... + a(m - 1) X^(m - 1)` at the
/// `4m`-primitive roots of unity corresponding to the subgroup `<g>` of `(Z/4mZ)*`.
/// 
/// The input is assumed to be in bitreversed order, and the output is in order
/// ```text
/// f(z), f(z^g), f(z^(g^2)), f(z^(g^3)), ...
/// ```
/// 
fn pow2_bitreversed_dwt<R, F, A>(H: &HypercubeIsomorphism<R, F, A>, dim_index: usize) -> Vec<LinearTransform<R, F, A>>
    where R: RingStore,
        R::Type: ZnRing + CanHomFrom<<<<GaloisFieldDyn as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type>,
        F: CyclotomicRingDecomposition<R::Type> + RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone,
        CCFFTRingBase<R, F, A>: CyclotomicRing + /* unfortunately, the type checker is not clever enough to know that this is always the case */ RingExtension<BaseRing = R>
{
    let m = H.len(dim_index);
    let log2_m = ZZ.abs_log2_ceil(&(m as i64)).unwrap();
    assert!(m == 1 << log2_m, "pow2_bitreversed_dwt() only valid for hypercube dimensions that have a power-of-2 length");
    assert!((H.ring().n() / m) % 4 == 0, "pow2_bitreversed_dwt() only possible if there is a 4m-th primitive root of unity");

    let zeta = H.slot_ring().pow(H.slot_ring().canonical_gen(), H.ring().n() / m / 4);
    debug_assert!(is_prim_root_of_unity(H.slot_ring(), &H.slot_ring().canonical_gen(), H.ring().n()));
    debug_assert!(is_prim_root_of_unity(H.slot_ring(), &zeta, 4 * m));

    let mut result = Vec::new();
    for log2_l in 1..=log2_m {
        result.push(pow2_bitreversed_dwt_butterfly(H, dim_index, 1 << log2_l, H.slot_ring().pow(H.slot_ring().clone_el(&zeta), m / (1 << log2_l))));
    }

    return result;
}

#[cfg(test)]
use crate::complexfft::pow2_cyclotomic::DefaultPow2CyclotomicCCFFTRingBase;

#[test]
fn test_pow2_bitreversed_dwt() {
    // `F23[X]/(X^16 + 1) ~ F_(23^4)^4`
    let ring = DefaultPow2CyclotomicCCFFTRingBase::new(Zn::new(23), 4);
    let hypercube = HypercubeIsomorphism::new(ring.get_ring());

    let mut current = ring_literal!(&ring, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    for T in pow2_bitreversed_dwt(&hypercube, 0) {
        current = ring.get_ring().compute_linear_transform(&current, &T);
    }

    assert_el_eq!(&ring, &ring_literal!(&ring, [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), &current);

    let mut current = hypercube.from_slot_vec([1, 2, 3, 4].into_iter().map(|n| hypercube.slot_ring().int_hom().map(n)));
    for T in pow2_bitreversed_dwt(&hypercube, 0) {
        current = ring.get_ring().compute_linear_transform(&current, &T);
    }

    // remember that input is in bitreversed order
    assert_el_eq!(&ring, &ring_literal!(&ring, [1, 0, 3, 0, 2, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0]), &current);
}