use std::alloc::Allocator;

use feanor_math::algorithms::linsolve::LinSolveRingStore;
use feanor_math::homomorphism::*;
use feanor_math::ring::*;
use feanor_math::rings::zn::*;
use feanor_math::rings::extension::FreeAlgebraStore;
use feanor_math::primitive_int::*;

use crate::*;
use crate::rings::slots::*;
use crate::cyclotomic::*;
use crate::StdZn;
use crate::lintransform::*;

fn column_dwt_matrix<'a, NumberRing, A>(H: &HypercubeIsomorphism<'a, NumberRing, A>, dim_index: usize, zeta_powertable: &PowerTable<&SlotRing<'a, A>>) -> OwnedMatrix<El<SlotRing<'a, A>>>
    where NumberRing: DecomposableCyclotomicNumberRing<Zn>,
        A: Allocator + Clone,
{
    let index_ring = H.cyclotomic_index_ring();
    let ZZ_to_Gal = index_ring.can_hom(&StaticRing::<i64>::RING).unwrap();

    OwnedMatrix::from_fn(H.len(dim_index), H.len(dim_index), |i, j| {
        let exponent = H.cyclotomic_index_ring().prod([
            H.shift_galois_element(dim_index, -(i as i64)),
            ZZ_to_Gal.map(j as i64),
            ZZ_to_Gal.map(H.ring().n() as i64 / H.corresponding_factor_n(dim_index))
        ]);
        return H.slot_ring().clone_el(&*zeta_powertable.get_power(H.cyclotomic_index_ring().smallest_lift(exponent)));
    })
}

///
/// Interprets each hypercolumn as a vector of length `ni`, and computes the discrete weighted transform 
/// along this vector, i.e. the evaluation at the primitive roots of unity `𝜁^(n/ni * j)` for `j` coprime
/// to `ni`
/// 
fn column_dwt<'a, NumberRing, A>(H: &HypercubeIsomorphism<'a, NumberRing, A>, dim_index: usize, zeta_powertable: &PowerTable<&SlotRing<'a, A>>) -> Vec<LinearTransform<NumberRing, A>>
    where NumberRing: DecomposableCyclotomicNumberRing<Zn>,
        A: Allocator + Clone,
{
    // multiplication with the matrix `A(i, j) = 𝜁^(j * shift_element(-i))` if we consider an element as multiple vectors along the `dim_index`-th dimension
    let A = column_dwt_matrix(H, dim_index, zeta_powertable);

    vec![LinearTransform::matmul1d(
        H, 
        dim_index, 
        |i, j, _idxs| H.slot_ring().clone_el(A.at(i, j))
    )]
}

fn column_dwt_inv<'a, NumberRing, A>(H: &HypercubeIsomorphism<'a, NumberRing, A>, dim_index: usize, zeta_powertable: &PowerTable<&SlotRing<'a, A>>) -> Vec<LinearTransform<NumberRing, A>>
    where NumberRing: DecomposableCyclotomicNumberRing<Zn>,
        A: Allocator + Clone,
{
    let mut A = column_dwt_matrix(H, dim_index, zeta_powertable);
    let mut rhs = OwnedMatrix::identity(H.len(dim_index), H.len(dim_index), H.slot_ring());
    let mut sol = OwnedMatrix::zero(H.len(dim_index), H.len(dim_index), H.slot_ring());
    <_ as LinSolveRingStore>::solve_right(H.slot_ring(), A.data_mut(), rhs.data_mut(), sol.data_mut()).assert_solved();

    // multiplication with the matrix `A(i, j) = 𝜁^(j * shift_element(-i))` if we consider an element as multiple vectors along the `dim_index`-th dimension
    vec![LinearTransform::matmul1d(
        H, 
        dim_index, 
        |i, j, _idxs| H.slot_ring().clone_el(sol.at(i, j))
    )]
}

/// 
/// in the first step, we arrange the coefficients of each slot in the coefficients of the corresponding hypercube dimension;
/// in other words, we map the element `𝜁^l e_U(j)` to `X1^(j + l m0) e_U(*) = X1^(j + l m0) sum_i e_U(i)`;
/// here `𝜁` is the canonical generator of the slot ring, and `X1` is the image of `X1` under the isomorphism
/// `Fp[X1, ..., Xr]/(Phi_n1(X1), ..., Phi_nr(Xr)) -> Fp[X]/(Phi_n(X))`, i.e. is `X1 = X^(n/n1)`.
///
fn slots_to_powcoeffs_fat_fst_step<'a, NumberRing, A>(H: &HypercubeIsomorphism<'a, NumberRing, A>, dim_index: usize, zeta_powertable: &PowerTable<&SlotRing<'a, A>>) -> OwnedMatrix<El<Zn>>
    where NumberRing: DecomposableCyclotomicNumberRing<Zn>,
        A: Allocator + Clone,
{
    let Gal = H.cyclotomic_index_ring();
    let ZZ_to_Gal = Gal.can_hom(&StaticRing::<i64>::RING).unwrap();

    OwnedMatrix::from_fn(H.len(dim_index) * H.slot_ring().rank(), H.len(dim_index) * H.slot_ring().rank(), |row_idx, col_idx| {
        let i = row_idx / H.slot_ring().rank();
        let k = row_idx % H.slot_ring().rank();
        let j = col_idx / H.slot_ring().rank();
        let l = col_idx % H.slot_ring().rank(); 
        // the "work" that is left to do is to write `X1 e_U(*)` w.r.t. the basis `𝜁^k e_U(i)`;
        // however, this is exactly `X1 = sum_i X^(n/n1) e_U(i) = sum_i 𝜁^(shift_element(-i) * n/n1) e_U(i)`
        let exponent = Gal.prod([
            H.shift_galois_element(0, -(i as i64)), 
            ZZ_to_Gal.map(H.ring().n() as i64 / H.corresponding_factor_n(0)),
            ZZ_to_Gal.map((j + l * H.len(0)) as i64)
        ]);
        return H.slot_ring().wrt_canonical_basis(&*zeta_powertable.get_power(Gal.smallest_lift(exponent))).at(k);
    })
}

///
/// Computes the [https://ia.cr/2014/873]-style linear transform for fat bootstrapping with composite moduli.
/// 
/// If for the linear transform input, the slot `(i1, ..., ir)` contains `sum_j a_(j, i1, ..., ir) 𝜁^j`, this
/// this transform "puts" `a_(j, i1, ..., ir)` into the powerful-basis coefficient of `X1^(j * m1 + i1) X2^i2 ... Xr^ir`.
/// 
#[allow(unused)]
fn slots_to_powcoeffs_fat<NumberRing, A>(H: &HypercubeIsomorphism<NumberRing, A>) -> Vec<LinearTransform<NumberRing, A>>
    where NumberRing: DecomposableCyclotomicNumberRing<Zn>,
        A: Allocator + Clone,
{
    assert!(H.ring().n() % 2 != 0);
    assert!(H.slot_ring().rank() == H.local_slot_rank(0));

    let mut result = Vec::new();
    let zeta_powertable = PowerTable::new(H.slot_ring(), H.slot_ring().canonical_gen(), H.ring().n() as usize);

    let fst_step_matrix = slots_to_powcoeffs_fat_fst_step(H, 0, &zeta_powertable);
    result.push(LinearTransform::blockmatmul1d(
        H,
        0,
        |(i, k), (j, l), _idxs| H.slot_ring().base_ring().clone_el(fst_step_matrix.at(i * H.slot_ring().rank() + k, j * H.slot_ring().rank() + l))
    ));

    for i in 1..H.dim_count() {
        result.extend(column_dwt(H, i, &zeta_powertable));
    }

    return result;
}

///
/// Inverse of [`slots_to_powcoeffs_fat()`], i.e. moves the powerful-basis coefficient of `X1^(j * m1 + i1) X2^i2 ... Xr^ir`
/// to the slot ``(i1, ..., ir)`.
/// 
fn powcoeffs_to_slots_fat<NumberRing, A>(H: &HypercubeIsomorphism<NumberRing, A>) -> Vec<LinearTransform<NumberRing, A>>
    where NumberRing: DecomposableCyclotomicNumberRing<Zn>,
        A: Allocator + Clone,
{
    assert!(H.ring().n() % 2 != 0);
    assert!(H.slot_ring().rank() == H.local_slot_rank(0));

    let mut result = Vec::new();
    let zeta_powertable = PowerTable::new(H.slot_ring(), H.slot_ring().canonical_gen(), H.ring().n() as usize);

    for i in (1..H.dim_count()).rev() {
        result.extend(column_dwt_inv(H, i, &zeta_powertable));
    }

    let mut A = slots_to_powcoeffs_fat_fst_step(H, 0, &zeta_powertable);
    let mut rhs = OwnedMatrix::identity(H.len(0) * H.slot_ring().rank(), H.len(0) * H.slot_ring().rank(), H.slot_ring().base_ring());
    let mut sol = OwnedMatrix::zero(H.len(0) * H.slot_ring().rank(), H.len(0) * H.slot_ring().rank(), H.slot_ring().base_ring());
    <_ as LinSolveRingStore>::solve_right(H.slot_ring().base_ring(), A.data_mut(), rhs.data_mut(), sol.data_mut()).assert_solved();

    result.push(LinearTransform::blockmatmul1d(
        H,
        0,
        |(i, k), (j, l), _idxs| H.slot_ring().base_ring().clone_el(sol.at(i * H.slot_ring().rank() + k, j * H.slot_ring().rank() + l))
    ));

    return result;
}

///
/// Computes the [https://ia.cr/2014/873]-style linear transform for thin bootstrapping with composite moduli.
/// 
/// If for the linear transform input, the slot `(i1, ..., ir)` contains a scalar `a_(i1, ..., ir)`, this
/// transform "puts" `a_(i1, ..., ir)` into the powerful-basis coefficient of `X1^i1 ... Xr^ir`.
/// 
pub fn slots_to_powcoeffs_thin<NumberRing, A>(H: &HypercubeIsomorphism<NumberRing, A>) -> Vec<LinearTransform<NumberRing, A>>
    where NumberRing: DecomposableCyclotomicNumberRing<Zn>,
        A: Allocator + Clone,
{
    assert!(H.ring().n() % 2 != 0);
    assert!(H.slot_ring().rank() == H.local_slot_rank(0));

    let zeta_powertable = PowerTable::new(H.slot_ring(), H.slot_ring().canonical_gen(), H.ring().n() as usize);
    let mut result = Vec::new();

    for i in 0..H.dim_count() {
        result.extend(column_dwt(H, i, &zeta_powertable));
    }
    return result;
}

///
/// Conceptually, this is the inverse of [`slots_to_powcoeffs_thin()`].
/// 
/// It does move the value from the powerful-basis coefficients `X1^i1 ... Xr^ir` for `i1 < phi(n1)/d` and
/// `i2 < phi(n2), ..., ir < phi(nr)` to the slot `(i1, ..., ir)`; However, values corresponding to other 
/// powerful-basis coefficients are discarded, i.e. mapped to zero. In particular this transform does not have
/// full rank, and cannot be the mathematical inverse of [`slots_to_powcoeffs_thin()`].
/// 
pub fn powcoeffs_to_slots_thin<NumberRing, A>(H: &HypercubeIsomorphism<NumberRing, A>) -> Vec<LinearTransform<NumberRing, A>>
    where NumberRing: DecomposableCyclotomicNumberRing<Zn>,
        A: Allocator + Clone,
{
    let mut result = powcoeffs_to_slots_fat(H);
    let discard_unused = LinearTransform::blockmatmul0d(
        H, 
        |i, j, _idxs| if j == 0 && i == 0 { H.slot_ring().base_ring().one() } else { H.slot_ring().base_ring().zero() }
    );
    let last_step = result.last_mut().unwrap();
    *last_step = discard_unused.compose(last_step, H);
    return result;
}

#[test]
fn test_slots_to_powcoeffs_thin() {
    // F11[X]/Phi_35(X) ~ F_(11^3)^8
    let ring = DecompositionRingBase::new(CompositeCyclotomicDecomposableNumberRing::new(5, 7), Zn::new(11));
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
    let ring = DecompositionRingBase::new(CompositeCyclotomicDecomposableNumberRing::new(5, 7), Zn::new(71));
    let H = HypercubeIsomorphism::new::<false>(ring.get_ring());

    let mut current = H.from_slot_vec((1..25).map(|n| H.slot_ring().int_hom().map(n)));
    for transform in slots_to_powcoeffs_thin(&H) {
        current = ring.get_ring().compute_linear_transform(&H, &current, &transform);
    }
    let ring_ref = &ring;
    let expected = ring.sum((0..4).flat_map(|i| (0..6).map(move |j| ring_ref.mul(ring_ref.pow(ring_ref.canonical_gen(), i * 7 + j * 5), ring_ref.int_hom().map((1 + j + i * 6) as i32)))));
    assert_el_eq!(ring, expected, current);

    // Z/8Z[X]/Phi_341 ~ GR(2, 3, 10)^30
    let ring = DecompositionRingBase::new(CompositeCyclotomicDecomposableNumberRing::new(11, 31), Zn::new(8));
    let H = HypercubeIsomorphism::new::<false>(ring.get_ring());

    let mut current = H.from_slot_vec((1..=30).map(|n| H.slot_ring().int_hom().map(n)));
    for transform in slots_to_powcoeffs_thin(&H) {
        current = ring.get_ring().compute_linear_transform(&H, &current, &transform);
    }
    let expected = ring.sum((0..30).map(|j| ring.mul(ring.pow(ring.canonical_gen(), j * 11), ring.int_hom().map((j + 1) as i32))));
    assert_el_eq!(ring, expected, current);
}

#[test]
fn test_powcoeffs_to_slots_thin() {
    // F11[X]/Phi_35(X) ~ F_(11^3)^8
    let ring = DecompositionRingBase::new(CompositeCyclotomicDecomposableNumberRing::new(5, 7), Zn::new(11));
    let H = HypercubeIsomorphism::new::<false>(ring.get_ring());

    assert_eq!(7, H.corresponding_factor_n(0));
    assert_eq!(2, H.len(0));
    assert_eq!(5, H.corresponding_factor_n(1));
    assert_eq!(4, H.len(1));
    let mut current = ring_literal!(&ring, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    for transform in powcoeffs_to_slots_thin(&H) {
        current = ring.get_ring().compute_linear_transform(&H, &current, &transform);
    }
    let expected = H.from_slot_vec([H.slot_ring().one()].into_iter().chain((2..9).map(|_| H.slot_ring().zero())));
    assert_el_eq!(ring, expected, current);

    assert_eq!(7, H.corresponding_factor_n(0));
    assert_eq!(2, H.len(0));
    assert_eq!(5, H.corresponding_factor_n(1));
    assert_eq!(4, H.len(1));
    let ring_ref = &ring;
    let mut current = ring.sum((0..6).flat_map(|i| (0..4).map(move |j| ring_ref.mul(ring_ref.pow(ring_ref.canonical_gen(), i * 5 + j * 7), ring_ref.int_hom().map((1 + j + i * 4) as i32)))));
    for transform in powcoeffs_to_slots_thin(&H) {
        current = ring.get_ring().compute_linear_transform(&H, &current, &transform);
    }
    let expected = H.from_slot_vec([1, 2, 3, 4, 5, 6, 7, 8].into_iter().map(|n| H.slot_ring().int_hom().map(n)));
    assert_el_eq!(ring, expected, current);

    // F71[X]/Phi_35(X) ~ F71^24
    let ring = DecompositionRingBase::new(CompositeCyclotomicDecomposableNumberRing::new(5, 7), Zn::new(71));
    let H = HypercubeIsomorphism::new::<false>(ring.get_ring());

    let ring_ref = &ring;
    let mut current = ring.sum((0..4).flat_map(|i| (0..6).map(move |j| ring_ref.mul(ring_ref.pow(ring_ref.canonical_gen(), i * 7 + j * 5), ring_ref.int_hom().map((1 + j + i * 6) as i32)))));
    for transform in powcoeffs_to_slots_thin(&H) {
        current = ring.get_ring().compute_linear_transform(&H, &current, &transform);
    }
    let expected = H.from_slot_vec((1..25).map(|n| H.slot_ring().int_hom().map(n)));
    assert_el_eq!(ring, expected, current);
}

#[test]
fn test_slots_to_powcoeffs_fat() {
    // F11[X]/Phi_35(X) ~ F_(11^3)^8
    let ring = DecompositionRingBase::new(CompositeCyclotomicDecomposableNumberRing::new(5, 7), Zn::new(11));
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
    let ring = DecompositionRingBase::new(CompositeCyclotomicDecomposableNumberRing::new(5, 7), Zn::new(71));
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

#[test]
fn test_powcoeffs_to_slots_fat() {
    // F11[X]/Phi_35(X) ~ F_(11^3)^8
    let ring = DecompositionRingBase::new(CompositeCyclotomicDecomposableNumberRing::new(5, 7), Zn::new(11));
    let H = HypercubeIsomorphism::new::<false>(ring.get_ring());

    assert_eq!(7, H.corresponding_factor_n(0));
    assert_eq!(2, H.len(0));
    assert_eq!(5, H.corresponding_factor_n(1));
    assert_eq!(4, H.len(1));
    let ring_ref = &ring;
    let mut current = ring.sum((0..6).flat_map(|i| (0..4).map(move |j| ring_ref.mul(ring_ref.pow(ring_ref.canonical_gen(), i * 5 + j * 7), ring_ref.int_hom().map((1 + j + i * 4) as i32)))));
    for transform in powcoeffs_to_slots_fat(&H) {
        current = ring.get_ring().compute_linear_transform(&H, &current, &transform);
    }
    let expected = H.from_slot_vec([1, 2, 3, 4, 5, 6, 7, 8].into_iter().map(|n| H.slot_ring().from_canonical_basis([n, n + 8, n + 16].into_iter().map(|m| H.slot_ring().base_ring().int_hom().map(m)))));
    assert_el_eq!(ring, expected, current);
}