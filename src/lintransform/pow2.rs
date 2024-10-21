use std::alloc::Allocator;

use feanor_math::algorithms::fft::cooley_tuckey::bitreverse;
use feanor_math::algorithms::unity_root::is_prim_root_of_unity;
use feanor_math::homomorphism::Homomorphism;
use feanor_math::integer::IntegerRingStore;
use feanor_math::ring::*;
use feanor_math::rings::extension::*;
use feanor_math::rings::zn::zn_64::*;
use feanor_math::rings::zn::*;
use feanor_math::assert_el_eq;

use crate::cyclotomic::*;
use crate::rings::decomposition::*;
use crate::rings::decomposition_ring::*;
use crate::rings::pow2_cyclotomic::*;
use crate::StdZn;
use super::*;

///
/// Works separately on each block `(b0, ..., b(l - 1))` of size `l = blocksize` along the given given hypercube dimension.
/// This function computes the length-`l` DWT
/// ```text
/// sum_(0 <= i < l) ai * 𝜁_(4l)^(i * g^j)
/// ``` 
/// from the length-`l/2` DWTs of the even-index resp. odd-index entries of `ai`. These two sub-DWTs are expected to be written
/// in the first resp. second half of the input block (i.e. not interleaved, this is where the "bitreversed" comes from).
/// Here `g` is the generator of the current hypercube dimension, i.e. usually `g = 5`.
/// 
/// More concretely, it is expected that the input to the linear transform is
/// ```text
/// bj = sum_(0 <= i < l/2) a(2i) * 𝜁_(4l)^(2 * i * g^j)              if j < l/2
/// bj = sum_(0 <= i < l/2) a(2i + 1) * 𝜁_(4l)^(2 * i * g^j)
///    = sum_(0 <= i < l/2) a(2i + 1) * 𝜁_(4l)^(2 * i * g^(j - l/2))  otherwise
/// ```
/// In this case, the output is
/// ```text
/// bj = sum_(0 <= i < l) ai * 𝜁_(4l)^(i * g^j)
/// ```
/// 
/// # Notes
///  - This does not compute the evaluation at all primitive `4l`-th roots of unity, but only at half of them - namely `𝜁_(4l)^(g^j)` for all `j`.
///    In particular, `g` does not generate `(Z/4lZ)*`, but `<g>` is an index 2 subgroup of it.
///  - `row_autos` can be given to use different `𝜁`s for each block; in particular, for the block with hypercube indices `idxs`, the DWT with root
///    of unity `𝜁 = root_of_unity_4l^row_autos(idxs)` is used. Note that the index passed to `row_autos` is the hypercube index of some element in the
///    block. It does not make sense for `row_autos` to behave differently on different indices in the same block, this will lead to 
///    `pow2_bitreversed_dwt_butterfly` to give nonsensical results. If you pass `row_autos = |_| H.cyclotomic_index_ring().one()` then this uses the same
///    roots of unity everywhere, i.e. results in the behavior as outlined above.
/// 
fn pow2_bitreversed_dwt_butterfly<'b, A, G>(H: &HypercubeIsomorphism<'b, Pow2CyclotomicDecomposableNumberRing, A>, dim_index: usize, l: usize, root_of_unity_4l: El<SlotRing<'b, A>>, row_autos: G) -> LinearTransform<Pow2CyclotomicDecomposableNumberRing, A>
    where A: Allocator + Clone,
        G: Fn(&[usize]) -> ZnEl
{
    let m = H.len(dim_index);
    let g = H.shift_galois_element(dim_index, -1);
    let smaller_cyclotomic_index_ring = Zn::new(4 * l as u64);
    let red = ReductionMap::new(H.cyclotomic_index_ring(), &smaller_cyclotomic_index_ring).unwrap();
    assert_el_eq!(&smaller_cyclotomic_index_ring, &smaller_cyclotomic_index_ring.one(), &smaller_cyclotomic_index_ring.pow(red.map(g), l));

    let log2_m = ZZ.abs_log2_ceil(&(m as i64)).unwrap();
    assert!(m == 1 << log2_m, "pow2_bitreversed_cooley_tuckey_butterfly() only valid for hypercube dimensions that have a power-of-2 length");
    let l = l;
    assert!(l > 1);
    assert!(m % l == 0);
    let zeta_power_table = PowerTable::new(H.slot_ring(), root_of_unity_4l, 4 * l);

    enum TwiddleFactor {
        Zero, PosPowerZeta(ZnEl), NegPowerZeta(ZnEl)
    }

    let pow_of_zeta = |factor: TwiddleFactor| match factor {
        TwiddleFactor::PosPowerZeta(pow) => H.slot_ring().clone_el(&zeta_power_table.get_power(H.cyclotomic_index_ring().smallest_positive_lift(pow))),
        TwiddleFactor::NegPowerZeta(pow) => H.slot_ring().negate(H.slot_ring().clone_el(&zeta_power_table.get_power(H.cyclotomic_index_ring().smallest_positive_lift(pow)))),
        TwiddleFactor::Zero => H.slot_ring().zero()
    };

    let forward_mask = H.from_slot_vec(H.slot_iter(|idxs| {
        let idx_in_block = idxs[dim_index] % l;
        if idx_in_block >= l / 2 {
            TwiddleFactor::PosPowerZeta(H.cyclotomic_index_ring().zero())
        } else {
            TwiddleFactor::Zero
        }
    }).map(&pow_of_zeta));

    let diagonal_mask = H.from_slot_vec(H.slot_iter(|idxs| {
        let idx_in_block = idxs[dim_index] % l;
        if idx_in_block >= l / 2 {
            TwiddleFactor::NegPowerZeta(H.cyclotomic_index_ring().mul(H.cyclotomic_index_ring().pow(g, idx_in_block - l / 2), row_autos(&idxs)))
        } else {
            TwiddleFactor::PosPowerZeta(H.cyclotomic_index_ring().zero())
        }
    }).map(&pow_of_zeta));

    let backward_mask = H.from_slot_vec(H.slot_iter(|idxs| {
        let idx_in_block = idxs[dim_index] % l;
        if idx_in_block < l / 2 {
            TwiddleFactor::PosPowerZeta(H.cyclotomic_index_ring().mul(H.cyclotomic_index_ring().pow(g, idx_in_block), row_autos(&idxs)))
        } else {
            TwiddleFactor::Zero
        }
    }).map(&pow_of_zeta));

    let mut result = LinearTransform {
        data: vec![
            (
                GaloisElementIndex::shift_1d(H.dim_count(), dim_index, 0),
                diagonal_mask
            ),
            (
                GaloisElementIndex::shift_1d(H.dim_count(), dim_index, l as i64 / 2),
                forward_mask
            ),
            (
                GaloisElementIndex::shift_1d(H.dim_count(), dim_index, -(l as i64) / 2),
                backward_mask
            )
        ]
    };
    result.canonicalize(H);
    return result;
}

///
/// Inverse of [`pow2_bitreversed_dwt_butterfly()`]
/// 
fn pow2_bitreversed_inv_dwt_butterfly<'b, A, G>(H: &HypercubeIsomorphism<'b, Pow2CyclotomicDecomposableNumberRing, A>, dim_index: usize, l: usize, root_of_unity_4l: El<SlotRing<'b, A>>, row_autos: G) -> LinearTransform<Pow2CyclotomicDecomposableNumberRing, A>
    where A: Allocator + Clone,
        G: Fn(&[usize]) -> ZnEl
{
    let m = H.len(dim_index);
    let g = H.shift_galois_element(dim_index, -1);
    let smaller_cyclotomic_index_ring = Zn::new(4 * l as u64);
    let red = ReductionMap::new(H.cyclotomic_index_ring(), &smaller_cyclotomic_index_ring).unwrap();
    assert_el_eq!(&smaller_cyclotomic_index_ring, &smaller_cyclotomic_index_ring.one(), &smaller_cyclotomic_index_ring.pow(red.map(g), l));

    let log2_m = ZZ.abs_log2_ceil(&(m as i64)).unwrap();
    assert!(m == 1 << log2_m, "pow2_bitreversed_inv_dwt_butterfly() only valid for hypercube dimensions that have a power-of-2 length");
    let l = l;
    assert!(l > 1);
    assert!(m % l == 0);
    let zeta_power_table = PowerTable::new(H.slot_ring(), root_of_unity_4l, 4 * l);

    enum TwiddleFactor {
        Zero, PosPowerZeta(ZnEl), NegPowerZeta(ZnEl)
    }

    let pow_of_zeta = |factor: TwiddleFactor| match factor {
        TwiddleFactor::PosPowerZeta(pow) => H.slot_ring().clone_el(&zeta_power_table.get_power(H.cyclotomic_index_ring().smallest_positive_lift(pow))),
        TwiddleFactor::NegPowerZeta(pow) => H.slot_ring().negate(H.slot_ring().clone_el(&zeta_power_table.get_power(H.cyclotomic_index_ring().smallest_positive_lift(pow)))),
        TwiddleFactor::Zero => H.slot_ring().zero()
    };

    let inv_2 = H.ring().base_ring().invert(&H.ring().base_ring().int_hom().map(2)).unwrap();

    let mut forward_mask = H.from_slot_vec(H.slot_iter(|idxs| {
        let idx_in_block = idxs[dim_index] % l;
        if idx_in_block >= l / 2 {
            TwiddleFactor::PosPowerZeta(H.cyclotomic_index_ring().mul(H.cyclotomic_index_ring().negate(H.cyclotomic_index_ring().pow(g, idx_in_block - l / 2)), row_autos(&idxs)))
        } else {
            TwiddleFactor::Zero
        }
    }).map(&pow_of_zeta));
    H.ring().inclusion().mul_assign_ref_map(&mut forward_mask, &inv_2);

    let mut diagonal_mask = H.from_slot_vec(H.slot_iter(|idxs| {
        let idx_in_block = idxs[dim_index] % l;
        if idx_in_block >= l / 2 {
            TwiddleFactor::NegPowerZeta(H.cyclotomic_index_ring().mul(H.cyclotomic_index_ring().negate(H.cyclotomic_index_ring().pow(g, idx_in_block - l / 2)), row_autos(&idxs)))
        } else {
            TwiddleFactor::PosPowerZeta(H.cyclotomic_index_ring().zero())
        }
    }).map(&pow_of_zeta));
    H.ring().inclusion().mul_assign_ref_map(&mut diagonal_mask, &inv_2);

    let mut backward_mask = H.from_slot_vec(H.slot_iter(|idxs| {
        let idx_in_block = idxs[dim_index] % l;
        if idx_in_block < l / 2 {
            TwiddleFactor::PosPowerZeta(H.cyclotomic_index_ring().zero())
        } else {
            TwiddleFactor::Zero
        }
    }).map(&pow_of_zeta));
    H.ring().inclusion().mul_assign_ref_map(&mut backward_mask, &inv_2);

    let mut result = LinearTransform {
        data: vec![
            (
                GaloisElementIndex::shift_1d(H.dim_count(), dim_index, 0),
                diagonal_mask
            ),
            (
                GaloisElementIndex::shift_1d(H.dim_count(), dim_index, l as i64 / 2),
                forward_mask
            ),
            (
                GaloisElementIndex::shift_1d(H.dim_count(), dim_index, -(l as i64) / 2),
                backward_mask
            )
        ]
    };
    result.canonicalize(&H);
    #[cfg(debug_assertions)] {
        let expected = pow2_bitreversed_dwt_butterfly(H, dim_index, l, H.slot_ring().clone_el(&zeta_power_table.get_power(1)), row_autos).inverse(&H);
        debug_assert!(result.eq(&expected, H));
    }
    return result;
}

///
/// Computes the evaluation of `f(X) = a0 + a1 X + a2 X^2 + ... + a(m - 1) X^(m - 1)` at the
/// `4m`-primitive roots of unity corresponding to the subgroup `<g>` of `(Z/4mZ)*`.
/// Here `m` is the hypercube length of the given dimension and `g` is the generator of the hypercube
/// dimension.
/// 
/// More concretely, this computes
/// ```text
/// sum_(0 <= i < m) a(bitrev(i)) * 𝜁^(n / (4m) * row_autos(idxs) * g^j)
/// ```
/// for `j` from `0` to `m`.
/// Here `𝜁` is the canonical generator of the slot ring, which is a primitive `n`-th root of unity.
/// 
fn pow2_bitreversed_dwt<A, G>(H: &HypercubeIsomorphism<Pow2CyclotomicDecomposableNumberRing, A>, dim_index: usize, row_autos: G) -> Vec<LinearTransform<Pow2CyclotomicDecomposableNumberRing, A>>
    where A: Allocator + Clone,
        G: Fn(&[usize]) -> ZnEl
{
    let m = H.len(dim_index);
    let log2_m = ZZ.abs_log2_ceil(&(m as i64)).unwrap();
    assert!(m == 1 << log2_m, "pow2_bitreversed_dwt() only valid for hypercube dimensions that have a power-of-2 length");
    assert!((H.ring().n() / m as u64) % 4 == 0, "pow2_bitreversed_dwt() only possible if there is a 4m-th primitive root of unity");

    let zeta = H.slot_ring().pow(H.slot_ring().canonical_gen(), H.ring().n() as usize / m / 4);

    debug_assert!(is_prim_root_of_unity(H.slot_ring(), &H.slot_ring().canonical_gen(), H.ring().n() as usize));
    debug_assert!(is_prim_root_of_unity(H.slot_ring(), &zeta, 4 * m));

    let mut result = Vec::new();
    for log2_l in 1..=log2_m {
        result.push(pow2_bitreversed_dwt_butterfly(
            H, 
            dim_index, 
            1 << log2_l, 
            H.slot_ring().pow(H.slot_ring().clone_el(&zeta), m / (1 << log2_l)), 
            &row_autos
        ));
    }

    return result;
}

///
/// Inverse to [`pow2_bitreversed_dwt()`]
/// 
fn pow2_bitreversed_inv_dwt<A, G>(H: &HypercubeIsomorphism<Pow2CyclotomicDecomposableNumberRing, A>, dim_index: usize, row_autos: G) -> Vec<LinearTransform<Pow2CyclotomicDecomposableNumberRing, A>>
    where A: Allocator + Clone,
        G: Fn(&[usize]) -> ZnEl
{
    let m = H.len(dim_index);
    let log2_m = ZZ.abs_log2_ceil(&(m as i64)).unwrap();
    assert!(m == 1 << log2_m, "pow2_bitreversed_dwt() only valid for hypercube dimensions that have a power-of-2 length");
    assert!((H.ring().n() / m as u64) % 4 == 0, "pow2_bitreversed_dwt() only possible if there is a 4m-th primitive root of unity");

    let zeta = H.slot_ring().pow(H.slot_ring().canonical_gen(), H.ring().n() as usize / m / 4);
    debug_assert!(is_prim_root_of_unity(H.slot_ring(), &H.slot_ring().canonical_gen(), H.ring().n() as usize));
    debug_assert!(is_prim_root_of_unity(H.slot_ring(), &zeta, 4 * m));

    let mut result = Vec::new();
    for log2_l in (1..=log2_m).rev() {
        result.push(pow2_bitreversed_inv_dwt_butterfly(
            H, 
            dim_index, 
            1 << log2_l, 
            H.slot_ring().pow(H.slot_ring().clone_el(&zeta), m / (1 << log2_l)), 
            &row_autos
        ));
    }

    return result;
}

///
/// Computes the [https://ia.cr/2024/153]-style Slots-to-Coeffs linear transform for the thin-bootstrapping case,
/// i.e. where all slots contain elements in `Z/pZ`.
/// 
/// In the case `p = 1 mod 4`, the slots are enumerated by `i, j` with `0 <= i < m/2` and `j in {0, 1}`. If `p = 1 mod 4`.
/// Then the returned linear transform will then put the value of slot `(i, 0)` into the coefficient of `X^(bitrev(i, m/2) * n/(2m))`
/// and the value of slot `(i, 1)` into the coefficient of `X^(bitrev(i, m/2) * n/(2m) + n/4)`.
/// 
/// If `p = 3 mod 4`, the slots are enumerated by `i` with `0 <= i < m` and the transform will put the value of slot `i` into the coefficient
/// of `X^(bitrev(i, m) * n/(4m))`
/// 
pub fn slots_to_coeffs_thin<A>(H: &HypercubeIsomorphism<Pow2CyclotomicDecomposableNumberRing, A>) -> Vec<LinearTransform<Pow2CyclotomicDecomposableNumberRing, A>>
    where A: Allocator + Clone
{
    let n = H.ring().get_ring().n();
    let log2_n = ZZ.abs_log2_ceil(&(n as i64)).unwrap();
    assert!(n == 1 << log2_n);

    if H.dim_count() == 2 {
        // this is the `p = 1 mod 4` case
        assert_eq!(2, H.len(1));
        let zeta4 = H.slot_ring().pow(H.slot_ring().canonical_gen(), H.ring().n() as usize / 4);
        let mut result = Vec::new();

        // we first combine `ai0` and `ai1` to `(ai0 + 𝜁^(n/4) ai1, ai0 - 𝜁^(n/4) ai1)`
        result.push(LinearTransform {
            data: vec![
                (
                    GaloisElementIndex::shift_1d(H.dim_count(), 1, 0),
                    H.from_slot_vec(H.slot_iter(|idxs| if idxs[1] == 0 {
                        H.slot_ring().one()
                    } else {
                        debug_assert!(idxs[1] == 1);
                        H.slot_ring().negate(H.slot_ring().clone_el(&zeta4))
                    }))
                ), 
                (
                    GaloisElementIndex::shift_1d(H.dim_count(), 1, 1),
                    H.from_slot_vec(H.slot_iter(|idxs| if idxs[1] == 0 {
                        H.slot_ring().clone_el(&zeta4)
                    } else {
                        debug_assert!(idxs[1] == 1);
                        H.slot_ring().one()
                    }))
                )
            ]
        });
        result.last_mut().unwrap().canonicalize(H);

        // then map the `ai0 + 𝜁^(n/4) ai1` to `sum_i (ai0 + 𝜁^(n/4) ai1) 𝜁^(n/(2m) i g^k)` for each slot `(k, 0)`, and similarly
        // for the slots `(*, 1)`. The negation in the second hypercolumn comes from the fact that `-𝜁^(n/4) = 𝜁^(-n/4)`
        result.extend(pow2_bitreversed_dwt(H, 0, |idxs| if idxs[1] == 0 {
            H.cyclotomic_index_ring().one()
        } else {
            debug_assert!(idxs[1] == 1);
            H.cyclotomic_index_ring().neg_one()
        }));
        
        return result;
    } else {
        // this is the `p = 3 mod 4` case
        assert_eq!(1, H.dim_count());
        return pow2_bitreversed_dwt(H, 0, |_idxs| H.cyclotomic_index_ring().one());
    }
}

///
/// This is the inverse to [`slots_to_coeffs_thin()`]. Note that it is not the
/// "Coeffs-to-Slots" map, as it does not discard unused factors. However, it is not
/// too hard to build the "coeffs-to-slots" map from this, see [`coeffs_to_slots_thin()`]. 
/// 
pub fn slots_to_coeffs_thin_inv<A>(H: &HypercubeIsomorphism<Pow2CyclotomicDecomposableNumberRing, A>) -> Vec<LinearTransform<Pow2CyclotomicDecomposableNumberRing, A>>
    where A: Allocator + Clone
{
    let n = H.ring().get_ring().n();
    let log2_n = ZZ.abs_log2_ceil(&(n as i64)).unwrap();
    assert!(n == 1 << log2_n);

    if H.dim_count() == 2 {
        assert_eq!(2, H.len(1));
        let zeta4_inv = H.slot_ring().pow(H.slot_ring().canonical_gen(), 3 * H.ring().n() as usize / 4);
        let two_inv = H.ring().base_ring().invert(&H.slot_ring().base_ring().int_hom().map(2)).unwrap();
        let mut result = Vec::new();

        result.extend(pow2_bitreversed_inv_dwt(H, 0, |idxs| if idxs[1] == 0 {
            H.cyclotomic_index_ring().one()
        } else {
            debug_assert!(idxs[1] == 1);
            H.cyclotomic_index_ring().neg_one()
        }));

        result.push(LinearTransform {
            data: vec![
                (
                    GaloisElementIndex::shift_1d(H.dim_count(), 1, 0),
                    H.ring().inclusion().mul_map(H.from_slot_vec(H.slot_iter(|idxs| if idxs[1] == 0 {
                        H.slot_ring().one()
                    } else {
                        debug_assert!(idxs[1] == 1);
                        H.slot_ring().negate(H.slot_ring().clone_el(&zeta4_inv))
                    })), H.ring().base_ring().clone_el(&two_inv))
                ), 
                (
                    GaloisElementIndex::shift_1d(H.dim_count(), 1, 1),
                    H.ring().inclusion().mul_map(H.from_slot_vec(H.slot_iter(|idxs| if idxs[1] == 0 {
                        H.slot_ring().one()
                    } else {
                        debug_assert!(idxs[1] == 1);
                        H.slot_ring().clone_el(&zeta4_inv)
                    })), two_inv)
                )
            ]
        });
        result.last_mut().unwrap().canonicalize(H);

        return result;
    } else {
        assert_eq!(1, H.dim_count());
        return pow2_bitreversed_inv_dwt(H, 0, |_idxs| H.cyclotomic_index_ring().one());
    }
}

pub fn coeffs_to_slots_thin<A>(H: &HypercubeIsomorphism<Pow2CyclotomicDecomposableNumberRing, A>) -> (Vec<LinearTransform<Pow2CyclotomicDecomposableNumberRing, A>>, Trace)
    where A: Allocator + Clone,
{
    let trace = Trace::new(&H.cyclotomic_index_ring(), H.cyclotomic_index_ring().smallest_positive_lift(H.frobenius_element(1)), H.slot_ring().rank());
    let mut result = slots_to_coeffs_thin_inv(H);
    let last = LinearTransform::slot_scalar_mult(H, &H.slot_ring().inclusion().map(H.slot_ring().base_ring().invert(&H.slot_ring().base_ring().int_hom().map(H.slot_ring().rank() as i32)).unwrap()));
    *result.last_mut().unwrap() = result.last().unwrap().compose(&last, H);
    return (result, trace);
}

#[test]
fn test_slots_to_coeffs_thin() {
    // `F97[X]/(X^32 + 1) ~ F_(97^2)^16`
    let ring = DecompositionRingBase::new(Pow2CyclotomicDecomposableNumberRing::new(64), Zn::new(97));
    let H = HypercubeIsomorphism::new::<false>(ring.get_ring());
    
    let mut current = H.from_slot_vec((1..17).map(|n| H.slot_ring().int_hom().map(n)));
    for T in slots_to_coeffs_thin(&H) {
        current = ring.get_ring().compute_linear_transform(&H, &current, &T);
    }

    let mut expected = [0; 32];
    for i in 0..8 {
        for j in 0..2 {
            expected[bitreverse(i, 3) * 2 + j * 16] = i * 2 + j + 1;
        }
    }
    assert_el_eq!(&ring, &ring_literal!(&ring, expected), &current);

    // `F23[X]/(X^32 + 1) ~ F_(23^8)^4`
    let ring = DecompositionRingBase::new(Pow2CyclotomicDecomposableNumberRing::new(64), Zn::new(23));
    let H = HypercubeIsomorphism::new::<false>(ring.get_ring());

    let mut current = H.from_slot_vec([1, 2, 3, 4].into_iter().map(|n| H.slot_ring().int_hom().map(n)));
    for T in slots_to_coeffs_thin(&H) {
        current = ring.get_ring().compute_linear_transform(&H, &current, &T);
    }

    let mut expected = [0; 32];
    for i in 0..4 {
        expected[bitreverse(i, 2) * 4] = i + 1;
    }
    assert_el_eq!(&ring, &ring_literal!(&ring, expected), &current);
}

#[test]
fn test_slots_to_coeffs_thin_inv() {
    // `F23[X]/(X^32 + 1) ~ F_(23^8)^4`
    let ring = DecompositionRingBase::new(Pow2CyclotomicDecomposableNumberRing::new(64), Zn::new(23));
    let H = HypercubeIsomorphism::new::<false>(ring.get_ring());

    for (transform, actual) in slots_to_coeffs_thin(&H).into_iter().rev().zip(slots_to_coeffs_thin_inv(&H).into_iter()) {
        let expected = transform.inverse(&H);
        assert!(expected.eq(&actual, &H));
    }
    
    // `F97[X]/(X^32 + 1) ~ F_(97^2)^16`
    let ring = DecompositionRingBase::new(Pow2CyclotomicDecomposableNumberRing::new(64), Zn::new(97));
    let H = HypercubeIsomorphism::new::<false>(ring.get_ring());
    
    for (transform, actual) in slots_to_coeffs_thin(&H).into_iter().rev().zip(slots_to_coeffs_thin_inv(&H).into_iter()) {
        let expected = transform.inverse(&H);
        assert!(expected.eq(&actual, &H));
    }
}

#[test]
fn test_coeffs_to_slots_thin() {
    // `F97[X]/(X^32 + 1) ~ F_(97^2)^16`
    let ring = DecompositionRingBase::new(Pow2CyclotomicDecomposableNumberRing::new(64), Zn::new(97));
    let H = HypercubeIsomorphism::new::<false>(ring.get_ring());
    
    let mut input = [0; 32];
    for i in 0..8 {
        for j in 0..2 {
            input[bitreverse(i, 3) * 2 + j * 16] = i * 2 + j + 1;
            input[bitreverse(i, 3) * 2 + j * 16 + 1] = i * 2 + j + 1 + 16;
        }
    }
    let mut current = ring_literal!(&ring, input);
    let (main_transform, trace) = coeffs_to_slots_thin(&H);
    for T in main_transform {
        current = ring.get_ring().compute_linear_transform(&H, &current, &T);
    }
    current = ring.get_ring().compute_trace(&current, &trace);

    let expected = H.from_slot_vec((1..17).map(|n| H.slot_ring().int_hom().map(n)));
    assert_el_eq!(&ring, &expected, &current);

    // `F23[X]/(X^32 + 1) ~ F_(23^8)^4`
    let ring = DecompositionRingBase::new(Pow2CyclotomicDecomposableNumberRing::new(64), Zn::new(23));
    let H = HypercubeIsomorphism::new::<false>(ring.get_ring());

    let mut input = [0; 32];
    input[4] = 1;
    input[16] = 1;
    let mut current = ring_literal!(&ring, input);
    let (main_transform, trace) = coeffs_to_slots_thin(&H);
    for T in main_transform.iter().rev().skip(1).rev() {
        current = ring.get_ring().compute_linear_transform(&H, &current, &T);
    }
    for a in H.get_slot_values(&current) {
        H.slot_ring().println(&a);
    }
    println!();
    current = ring.get_ring().compute_linear_transform(&H, &current, main_transform.last().unwrap());
    current = ring.get_ring().compute_trace(&current, &trace);

    let expected = H.from_slot_vec([0, 0, 1, 0].into_iter().map(|n| H.slot_ring().int_hom().map(n)));
    
    for a in H.get_slot_values(&current) {
        H.slot_ring().println(&a);
    }
    println!();
    for a in H.get_slot_values(&expected) {
        H.slot_ring().println(&a);
    }

    assert_el_eq!(&ring, &expected, &current);

    let mut input = [0; 32];
    for i in 0..4 {
        input[bitreverse(i, 2) * 4] = i + 1;
        for k in 1..4 {
            input[bitreverse(i, 2) * 4 + k] = i + 1 + 4 * k;
        }
        for k in 0..4 {
            input[bitreverse(i, 2) * 4 + k + 16] = i + 1 + 4 * k + 16;
        }
    }
    let mut current = ring_literal!(&ring, input);
    ring.println(&current);
    let (main_transform, trace) = coeffs_to_slots_thin(&H);
    for T in main_transform.iter() {
        current = ring.get_ring().compute_linear_transform(&H, &current, &T);
    }
    current = ring.get_ring().compute_trace(&current, &trace);

    let expected = H.from_slot_vec([1, 2, 3, 4].into_iter().map(|n| H.slot_ring().int_hom().map(n)));

    assert_el_eq!(&ring, &expected, &current);
}
