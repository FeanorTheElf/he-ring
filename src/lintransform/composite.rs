use std::alloc::Allocator;

use feanor_math::homomorphism::*;
use feanor_math::ring::*;
use feanor_math::rings::extension::galois_field::GaloisFieldDyn;
use feanor_math::rings::zn::zn_64::*;
use feanor_math::rings::zn::*;
use feanor_math::rings::extension::FreeAlgebraStore;
use feanor_math::primitive_int::StaticRing;

use crate::complexfft::automorphism::*;
use crate::complexfft::complex_fft_ring::*;
use crate::cyclotomic::*;
use super::LinearTransform;

fn column_dwt<R, F, A, G>(H: &HypercubeIsomorphism<R, F, A>, dim_index: usize, row_autos: G) -> Vec<LinearTransform<R, F, A>>
    where R: RingStore,
        R::Type: ZnRing + CanHomFrom<<<<GaloisFieldDyn as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type>,
        F: CyclotomicRingDecomposition<R::Type> + RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone,
        CCFFTRingBase<R, F, A>: CyclotomicRing + /* unfortunately, the type checker is not clever enough to know that this is always the case */ RingExtension<BaseRing = R>,
        G: Fn(&[usize]) -> ZnEl
{
    let m = H.len(dim_index) as i64;
    let zeta = H.slot_ring().pow(H.slot_ring().canonical_gen(), H.ring().n() / H.dim(dim_index).corresponding_factor_n() as usize);
    let hom = H.galois_group_mulrepr().can_hom(&StaticRing::<i64>::RING).unwrap();
    vec![LinearTransform {
        data: ((1 - m)..m).map(|s| {
            let coeff = H.from_slot_vec(H.slot_iter(|idxs| if idxs[dim_index] as i64 >= s && idxs[dim_index] as i64 - s < m {
                H.slot_ring().pow(H.slot_ring().clone_el(&zeta), {
                    let res = H.galois_group_mulrepr().smallest_positive_lift(H.galois_group_mulrepr().prod([
                    H.shift_galois_element(dim_index, -(idxs[dim_index] as i64)),
                    hom.map(idxs[dim_index] as i64 - s),
                    row_autos(idxs)
                ].into_iter())) as usize; res })
            } else {
                H.slot_ring().zero()
            }));
            return (
                H.shift_galois_element(dim_index, s),
                coeff, 
                (0..H.dim_count()).map(|i| if i == dim_index { s } else { 0 }).collect()
            );
        }).collect()
    }]
}

///
/// Moves the values in the slots to the coefficients w.r.t. the powerful basis.
/// 
/// The powerful basis is the basis given by the `X^(i1 n/p1^e1 + ... + ik n/pk^ek mod n)` for `ij < lj`, 
/// where `n = p1^e1 ... pk^ek` is the prime factorization of `n` and `lj` is the length of the hypercube
/// dimension corresponding to `pj^ej`. Note that this is not exactly equal to the power-basis coefficients,
/// which are `X^i` for `0 <= i < l = l1 ... lk`. However, for bootstrapping, they can usually
/// be used interchangably, up to a small increase in error.
/// 
/// # Simple example
/// 
/// Consider the `35`th cyclotomic number ring `Z[X]/(Phi_35)` over `F71`.
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// # use he_ring::complexfft::complex_fft_ring::*;
/// # use he_ring::complexfft::odd_cyclotomic::*;
/// let R = DefaultOddCyclotomicCCFFTRingBase::new(Zn::new(71), 35);
/// ```
/// Since `71 = 1 mod 35`, we know that `R` decomposes completely into `phi(35) = 4 * 6 = 24` slots,
/// each isomorphic to `F71`. We can order these slots according to the hypercube `H`
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// # use he_ring::complexfft::complex_fft_ring::*;
/// # use he_ring::complexfft::odd_cyclotomic::*;
/// # use he_ring::complexfft::automorphism::*;
/// # let R = DefaultOddCyclotomicCCFFTRingBase::new(Zn::new(71), 35);
/// let H = HypercubeIsomorphism::new(R.get_ring());
/// assert_eq!(5, H.dim(0).corresponding_factor_n());
/// assert_eq!(7, H.dim(1).corresponding_factor_n());
/// ```
/// Now we can compute an element that has the values `1` to `24` in the slots
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// # use he_ring::complexfft::complex_fft_ring::*;
/// # use feanor_math::homomorphism::Homomorphism;
/// # use he_ring::complexfft::odd_cyclotomic::*;
/// # use he_ring::complexfft::automorphism::*;
/// # let R = DefaultOddCyclotomicCCFFTRingBase::new(Zn::new(71), 35);
/// # let H = HypercubeIsomorphism::new(R.get_ring());
/// let a = H.from_slot_vec((1..25).map(|n| H.slot_ring().int_hom().map(n)));
/// ```
/// Using `odd_slots_to_powcoeffs_thin()` will now move these values into the powerful basis components.
/// The powerful basis is given by
/// ```
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// # use he_ring::complexfft::complex_fft_ring::*;
/// # use he_ring::complexfft::odd_cyclotomic::*;
/// # use feanor_math::rings::extension::FreeAlgebraStore;
/// # use he_ring::complexfft::automorphism::*;
/// # let R = DefaultOddCyclotomicCCFFTRingBase::new(Zn::new(71), 35);
/// let powerful_basis = (0..4).flat_map(|i| (0..6).map(move |j| (i, j))).map(|(i, j)| (i * 7 + j * 5)).map(|k| R.pow(R.canonical_gen(), k)).collect::<Vec<_>>();
/// ```
/// Note that the powerful basis does not only have monomials, e.g.
/// ```
/// # use he_ring::ring_literal;
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// # use he_ring::complexfft::complex_fft_ring::*;
/// # use he_ring::complexfft::odd_cyclotomic::*;
/// # use feanor_math::rings::extension::FreeAlgebraStore;
/// # use he_ring::complexfft::automorphism::*;
/// # let R = DefaultOddCyclotomicCCFFTRingBase::new(Zn::new(71), 35);
/// # let powerful_basis = (0..4).flat_map(|i| (0..6).map(move |j| (i, j))).map(|(i, j)| (i * 7 + j * 5)).map(|k| R.pow(R.canonical_gen(), k)).collect::<Vec<_>>();
/// assert_el_eq!(R, ring_literal!(&R, [-1, 0, 0, 1, 0, -1, 0, -1, 1, 0, 0, 0, -1, 1, -1, 0, 0, 0, 1, -1, 0, -1, 0, 1]), &powerful_basis[3 * 6 + 1])
/// ```
/// Applying the transform
/// ```
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// # use he_ring::complexfft::complex_fft_ring::*;
/// # use he_ring::complexfft::odd_cyclotomic::*;
/// # use he_ring::complexfft::automorphism::*;
/// # use feanor_math::rings::extension::FreeAlgebraStore;
/// # use feanor_math::homomorphism::Homomorphism;
/// # use he_ring::lintransform::composite::odd_slots_to_powcoeffs_thin;
/// # let R = DefaultOddCyclotomicCCFFTRingBase::new(Zn::new(71), 35);
/// # let H = HypercubeIsomorphism::new(R.get_ring());
/// # let a = H.from_slot_vec((1..25).map(|n| H.slot_ring().int_hom().map(n)));
/// # let powerful_basis = (0..4).flat_map(|i| (0..6).map(move |j| (i, j))).map(|(i, j)| (i * 7 + j * 5)).map(|k| R.pow(R.canonical_gen(), k)).collect::<Vec<_>>();
/// let mut current = a;
/// for transform in odd_slots_to_powcoeffs_thin(&H) {
///    current = R.get_ring().compute_linear_transform(&current, &transform);
/// }
/// let transformed_a = current;
/// ```
/// we find
/// ```
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// # use he_ring::complexfft::complex_fft_ring::*;
/// # use he_ring::complexfft::odd_cyclotomic::*;
/// # use he_ring::complexfft::automorphism::*;
/// # use feanor_math::rings::extension::FreeAlgebraStore;
/// # use feanor_math::homomorphism::Homomorphism;
/// # use he_ring::lintransform::composite::odd_slots_to_powcoeffs_thin;
/// # let R = DefaultOddCyclotomicCCFFTRingBase::new(Zn::new(71), 35);
/// # let H = HypercubeIsomorphism::new(R.get_ring());
/// # let a = H.from_slot_vec((1..25).map(|n| H.slot_ring().int_hom().map(n)));
/// # let powerful_basis = (0..4).flat_map(|i| (0..6).map(move |j| (i, j))).map(|(i, j)| (i * 7 + j * 5)).map(|k| R.pow(R.canonical_gen(), k)).collect::<Vec<_>>();
/// # let mut current = a;
/// # for transform in odd_slots_to_powcoeffs_thin(&H) {
/// #    current = R.get_ring().compute_linear_transform(&current, &transform);
/// # }
/// # let transformed_a = current;
/// let expected = R.sum((1..25).zip(powerful_basis.iter()).map(|(c, b)| R.int_hom().mul_ref_map(b, &c)));
/// assert_el_eq!(R, expected, transformed_a);
/// ```
/// 
pub fn odd_slots_to_powcoeffs_thin<R, F, A>(H: &HypercubeIsomorphism<R, F, A>) -> Vec<LinearTransform<R, F, A>>
    where R: RingStore,
        R::Type: ZnRing + CanHomFrom<<<<GaloisFieldDyn as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type>,
        F: CyclotomicRingDecomposition<R::Type> + RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone,
        CCFFTRingBase<R, F, A>: CyclotomicRing + /* unfortunately, the type checker is not clever enough to know that this is always the case */ RingExtension<BaseRing = R>
{
    assert!(H.ring().n() % 2 != 0);
    let mut result = Vec::new();
    for i in 0..H.dim_count() {
        result.extend(column_dwt(H, i, |_| H.galois_group_mulrepr().one()));
    }
    return result;
}

#[cfg(test)]
use crate::complexfft::odd_cyclotomic::*;
#[cfg(test)]
use feanor_math::assert_el_eq;

#[test]
fn test_column_dwt() {
    // F2[X]/Phi_31(X) ~ F32^6
    let ring = DefaultOddCyclotomicCCFFTRingBase::new(Zn::new(2), 31);
    let H = HypercubeIsomorphism::new(ring.get_ring());

    let mut current = ring_literal!(&ring, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    for transform in column_dwt(&H, 0, |_| H.galois_group_mulrepr().one()) {
        current = ring.get_ring().compute_linear_transform(&current, &transform);
    }

    assert_el_eq!(ring, ring_literal!(&ring, [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), current);
}

#[test]
fn test_odd_slots_to_powcoeffs_thin() {
    // F11[X]/Phi_35(X) ~ F_(11^3)^8
    let ring = DefaultOddCyclotomicCCFFTRingBase::new(Zn::new(11), 35);
    let H = HypercubeIsomorphism::new(ring.get_ring());

    let mut current = ring_literal!(&ring, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    for transform in odd_slots_to_powcoeffs_thin(&H) {
        current = ring.get_ring().compute_linear_transform(&current, &transform);
    }
    let expected = ring.sum([0, 5, 7, 12, 14, 19, 21, 26].into_iter().map(|k| ring.pow(ring.canonical_gen(), k)));
    assert_el_eq!(ring, expected, current);

    let mut current = H.from_slot_vec((1..9).map(|n| H.slot_ring().int_hom().map(n)));
    for transform in odd_slots_to_powcoeffs_thin(&H) {
        current = ring.get_ring().compute_linear_transform(&current, &transform);
    }
    let ring_ref = &ring;
    let expected = ring.sum((0..4).flat_map(|i| (0..2).map(move |j| ring_ref.mul(ring_ref.pow(ring_ref.canonical_gen(), i * 7 + j * 5), ring_ref.int_hom().map((1 + j + i * 2) as i32)))));

    assert_el_eq!(ring, expected, current);

    // F71[X]/Phi_35(X) ~ F71^24
    let ring = DefaultOddCyclotomicCCFFTRingBase::new(Zn::new(71), 35);
    let H = HypercubeIsomorphism::new(ring.get_ring());

    let mut current = H.from_slot_vec((1..25).map(|n| H.slot_ring().int_hom().map(n)));
    for transform in odd_slots_to_powcoeffs_thin(&H) {
        current = ring.get_ring().compute_linear_transform(&current, &transform);
    }
    let ring_ref = &ring;
    let expected = ring.sum((0..4).flat_map(|i| (0..6).map(move |j| ring_ref.mul(ring_ref.pow(ring_ref.canonical_gen(), i * 7 + j * 5), ring_ref.int_hom().map((1 + j + i * 6) as i32)))));
    assert_el_eq!(ring, expected, current);
}