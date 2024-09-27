use std::alloc::Allocator;

use feanor_math::homomorphism::*;
use feanor_math::ring::*;
use feanor_math::rings::zn::zn_64::*;
use feanor_math::rings::zn::*;
use feanor_math::rings::extension::FreeAlgebraStore;
use feanor_math::primitive_int::*;
use trace::Trace;

use crate::*;
use crate::rings::slots::*;
use crate::cyclotomic::*;
use crate::StdZn;
use crate::lintransform::*;

fn column_dwt<'a, R, F, A, G>(H: &HypercubeIsomorphism<'a, R, F, A>, dim_index: usize, row_autos: G, zeta_powertable: &PowerTable<&SlotRing<'a, R, A>>) -> Vec<LinearTransform<R, F, A>>
    where R: RingStore,
        R::Type: StdZn,
        F: CyclotomicRingDecomposition<R::Type> + RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone,
        NTTRingBase<R, F, A>: CyclotomicRing + RingExtension<BaseRing = R>,
        G: Fn(&[usize]) -> ZnEl
{
    let ZZ_to_Gal = H.galois_group_mulrepr().can_hom(&StaticRing::<i64>::RING).unwrap();

    // multiplication with the matrix `A(i, j) = zeta^(j * shift_element(-i))` if we consider an element as multiple vectors along the `dim_index`-th dimension
    vec![LinearTransform::matmul1d(
        H, 
        dim_index, 
        |i, j, idxs| {
            let exponent = H.galois_group_mulrepr().prod([
                H.shift_galois_element(dim_index, -(i as i64)),
                ZZ_to_Gal.map(j as i64),
                ZZ_to_Gal.map(H.ring().n() as i64 / H.corresponding_factor_n(dim_index)),
                row_autos(idxs)
            ]);
            return H.slot_ring().clone_el(&*zeta_powertable.get_power(H.galois_group_mulrepr().smallest_lift(exponent)));
        }
    )]
}

///
/// Assumes the slot `(i1, ..., ir)` contains `sum_j a_(j, i1, ..., ir) zeta^j`.
/// 
/// Then moves the value `a_(j, i1, ..., ir)` to the powerful-basis coefficient of `X1^(j * m1 + i1) X2^i2 ... Xr^ir`.
/// 
fn slots_to_powcoeffs_fat<R, F, A>(H: &HypercubeIsomorphism<R, F, A>) -> Vec<LinearTransform<R, F, A>>
    where R: RingStore,
        R::Type: StdZn,
        F: CyclotomicRingDecomposition<R::Type> + RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone,
        NTTRingBase<R, F, A>: CyclotomicRing + RingExtension<BaseRing = R>
{
    assert!(H.ring().n() % 2 != 0);
    assert!(H.slot_ring().rank() == H.local_slot_rank(0));

    let mut result = Vec::new();
    let zeta_powertable = PowerTable::new(H.slot_ring(), H.slot_ring().canonical_gen(), H.ring().n());

    let Gal = H.galois_group_mulrepr();
    let ZZ_to_Gal = Gal.can_hom(&StaticRing::<i64>::RING).unwrap();
    
    // in the first step, we arrange the coefficients of each slot in the coefficients of the corresponding hypercube dimension;
    // in other words, we map the element `zeta^l e_U(j)` to `X1^(j + l m0) e_U(*) = X1^(j + l m0) sum_i e_U(i)`;
    // here `zeta` is the canonical generator of the slot ring, and `X1` is the image of `X1` under the isomorphism
    // `Fp[X1, ..., Xr]/(Phi_n1(X1), ..., Phi_nr(Xr)) -> Fp[X]/(Phi_n(X))`, i.e. is `X1 = X^(n/n1)`.
    result.push(LinearTransform::blockmatmul1d(
        H,
        0,
        |(i, k), (j, l), _idxs| {
            // the "work" that is left to do is to write `X1 e_U(*)` w.r.t. the basis `zeta^k e_U(i)`;
            // however, this is exactly `X1 = sum_i X^(n/n1) e_U(i) = sum_i zeta^(shift_element(-i) * n/n1) e_U(i)`
            let exponent = Gal.prod([
                H.shift_galois_element(0, -(i as i64)), 
                ZZ_to_Gal.map(H.ring().n() as i64 / H.corresponding_factor_n(0)),
                ZZ_to_Gal.map((j + l * H.len(0)) as i64)
            ]);
            return H.slot_ring().wrt_canonical_basis(&*zeta_powertable.get_power(Gal.smallest_lift(exponent))).at(k);
        }
    ));

    for i in 1..H.dim_count() {
        result.extend(column_dwt(H, i, |_| H.galois_group_mulrepr().one(), &zeta_powertable));
    }

    return result;
}

///
/// Assumes each slot contains only an element of `Fp`.
/// 
/// Then moves the value in slot `(i1, ..., ir)` to the powerful-basis coefficient of `X1^i1 ... Xr^ir`.
/// 
pub fn slots_to_powcoeffs_thin<R, F, A>(H: &HypercubeIsomorphism<R, F, A>) -> Vec<LinearTransform<R, F, A>>
    where R: RingStore,
        R::Type: StdZn,
        F: CyclotomicRingDecomposition<R::Type> + RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone,
        NTTRingBase<R, F, A>: CyclotomicRing + RingExtension<BaseRing = R>,
{
    assert!(H.ring().n() % 2 != 0);
    assert!(H.slot_ring().rank() == H.local_slot_rank(0));

    let zeta_powertable = PowerTable::new(H.slot_ring(), H.slot_ring().canonical_gen(), H.ring().n());
    let mut result = Vec::new();

    for i in 0..H.dim_count() {
        result.extend(column_dwt(H, i, |_| H.galois_group_mulrepr().one(), &zeta_powertable));
    }
    return result;
}

#[test]
fn test_slots_to_powcoeffs_thin() {
    // F11[X]/Phi_35(X) ~ F_(11^3)^8
    let ring = DefaultOddCyclotomicNTTRingBase::new(Zn::new(11), 35);
    let H = HypercubeIsomorphism::new::<false>(ring.get_ring());

    // first test very simple case
    let mut current = ring_literal!(&ring, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    for transform in slots_to_powcoeffs_thin(&H) {
        current = ring.get_ring().compute_linear_transform(&H, &current, &transform);
    }
    let expected = ring.sum([0, 5, 7, 12, 14, 19, 21, 26].into_iter().map(|k| ring.pow(ring.canonical_gen(), k)));
    assert_el_eq!(ring, expected, current);

    // then test "thin bootstrapping" case
    assert_eq!(7, H.corresponding_factor_n(0));
    assert_eq!(2, H.len(0));
    assert_eq!(5, H.corresponding_factor_n(1));
    assert_eq!(4, H.len(1));
    let mut current = H.from_slot_vec((1..9).map(|n| H.slot_ring().int_hom().map(n)));
    for transform in slots_to_powcoeffs_thin(&H) {
        current = ring.get_ring().compute_linear_transform(&H, &current, &transform);
    }
    let ring_ref = &ring;
    let expected = ring.sum((0..2).flat_map(|i| (0..4).map(move |j| ring_ref.mul(ring_ref.pow(ring_ref.canonical_gen(), i * 5 + j * 7), ring_ref.int_hom().map((1 + j + i * 4) as i32)))));
    assert_el_eq!(ring, expected, current);

    // F71[X]/Phi_35(X) ~ F71^24
    let ring = DefaultOddCyclotomicNTTRingBase::new(Zn::new(71), 35);
    let H = HypercubeIsomorphism::new::<false>(ring.get_ring());

    let mut current = H.from_slot_vec((1..25).map(|n| H.slot_ring().int_hom().map(n)));
    for transform in slots_to_powcoeffs_thin(&H) {
        current = ring.get_ring().compute_linear_transform(&H, &current, &transform);
    }
    let ring_ref = &ring;
    let expected = ring.sum((0..4).flat_map(|i| (0..6).map(move |j| ring_ref.mul(ring_ref.pow(ring_ref.canonical_gen(), i * 7 + j * 5), ring_ref.int_hom().map((1 + j + i * 6) as i32)))));
    assert_el_eq!(ring, expected, current);
}

#[test]
fn test_slots_to_powcoeffs_fat() {
    // F11[X]/Phi_35(X) ~ F_(11^3)^8
    let ring = DefaultOddCyclotomicNTTRingBase::new(Zn::new(11), 35);
    let H = HypercubeIsomorphism::new::<false>(ring.get_ring());

    // first test very simple case
    let mut current = ring_literal!(&ring, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    for transform in slots_to_powcoeffs_fat(&H) {
        current = ring.get_ring().compute_linear_transform(&H, &current, &transform);
    }
    let expected = ring.sum([0, 5, 7, 12, 14, 19, 21, 26].into_iter().map(|k| ring.pow(ring.canonical_gen(), k)));
    assert_el_eq!(ring, expected, current);

    // then test "thin bootstrapping" case
    assert_eq!(7, H.corresponding_factor_n(0));
    assert_eq!(2, H.len(0));
    assert_eq!(5, H.corresponding_factor_n(1));
    assert_eq!(4, H.len(1));
    let mut current = H.from_slot_vec((1..9).map(|n| H.slot_ring().int_hom().map(n)));
    for transform in slots_to_powcoeffs_fat(&H) {
        current = ring.get_ring().compute_linear_transform(&H, &current, &transform);
    }
    let ring_ref = &ring;
    let expected = ring.sum((0..2).flat_map(|i| (0..4).map(move |j| ring_ref.mul(ring_ref.pow(ring_ref.canonical_gen(), i * 5 + j * 7), ring_ref.int_hom().map((1 + j + i * 4) as i32)))));
    assert_el_eq!(ring, expected, current);

    // then test "fat bootstrapping" case
    let hom = H.slot_ring().base_ring().int_hom();
    let mut current = H.from_slot_vec((1..9).map(|n| H.slot_ring().from_canonical_basis([hom.map(n), hom.map(n + 100), hom.map(n + 200)])));
    for transform in slots_to_powcoeffs_fat(&H) {
        current = ring.get_ring().compute_linear_transform(&H, &current, &transform);
    }
    let ring_ref = &ring;
    let expected = ring.sum((0..3).flat_map(|k| (0..2).flat_map(move |i| (0..4).map(move |j| ring_ref.mul(ring_ref.pow(ring_ref.canonical_gen(), (i + k * 2) * 5 + j * 7), ring_ref.int_hom().map((1 + j + i * 4 + k * 100) as i32))))));

    assert_el_eq!(ring, expected, current);

    // F71[X]/Phi_35(X) ~ F71^24
    let ring = DefaultOddCyclotomicNTTRingBase::new(Zn::new(71), 35);
    let H = HypercubeIsomorphism::new::<false>(ring.get_ring());

    assert_eq!(5, H.corresponding_factor_n(0));
    assert_eq!(4, H.len(0));
    assert_eq!(7, H.corresponding_factor_n(1));
    assert_eq!(6, H.len(1));
    let mut current = H.from_slot_vec((1..25).map(|n| H.slot_ring().int_hom().map(n)));
    for transform in slots_to_powcoeffs_fat(&H) {
        current = ring.get_ring().compute_linear_transform(&H, &current, &transform);
    }
    let ring_ref = &ring;
    let expected = ring.sum((0..4).flat_map(|i| (0..6).map(move |j| ring_ref.mul(ring_ref.pow(ring_ref.canonical_gen(), i * 7 + j * 5), ring_ref.int_hom().map((1 + j + i * 6) as i32)))));
    assert_el_eq!(ring, expected, current);
}