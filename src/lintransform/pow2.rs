use std::alloc::Allocator;

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
use crate::rings::ntt_ring::*;
use crate::StdZn;
use super::*;

///
/// Works separately on each block `(b0, ..., b(l - 1))` of size `l = blocksize` along the given given hypercube dimension.
/// This function computes the length-`l` DWT
/// ```text
/// sum_(0 <= i < l) ai * zeta_(4l)^(i * g^j)
/// ``` 
/// from the length-`l/2` DWTs of the even-index resp. odd-index entries of `ai`. These two sub-DWTs are expected to be written
/// in the first resp. second half of the input block (i.e. not interleaved, this is where the "bitreversed" comes from).
/// Here `g` is the generator of the current hypercube dimension, i.e. usually `g = 5`.
/// 
/// More concretely, it is expected that the input to the linear transform is
/// ```text
/// bj = sum_(0 <= i < l/2) a(2i) * zeta_(4l)^(2 * i * g^j)              if j < l/2
/// bj = sum_(0 <= i < l/2) a(2i + 1) * zeta_(4l)^(2 * i * g^j)
///    = sum_(0 <= i < l/2) a(2i + 1) * zeta_(4l)^(2 * i * g^(j - l/2))  otherwise
/// ```
/// In this case, the output is
/// ```text
/// bj = sum_(0 <= i < l) ai * zeta_(4l)^(i * g^j)
/// ```
/// 
/// # Notes
///  - This does not compute the evaluation at all primitive `4l`-th roots of unity, but only at half of them - namely `zeta_(4l)^(g^j)` for all `j`.
///    In particular, `g` does not generate `(Z/4lZ)*`, but `<g>` is an index 2 subgroup of it.
///  - `row_autos` can be given to use different `zeta`s for each block; in particular, for the block with hypercube indices `idxs`, the DWT with root
///    of unity `zeta = root_of_unity_4l^row_autos(idxs)` is used. Note that the index passed to `row_autos` is the hypercube index of some element in the
///    block. It does not make sense for `row_autos` to behave differently on different indices in the same block, this will lead to 
///    `pow2_bitreversed_dwt_butterfly` to give nonsensical results. If you pass `row_autos = |_| H.galois_group_mulrepr().one()` then this uses the same
///    roots of unity everywhere, i.e. results in the behavior as outlined above.
/// 
fn pow2_bitreversed_dwt_butterfly<'b, R, F, A, G>(H: &HypercubeIsomorphism<'b, R, F, A>, dim_index: usize, l: usize, root_of_unity_4l: El<SlotRing<'b, R, A>>, row_autos: G) -> LinearTransform<R, F, A>
    where R: RingStore,
        R::Type: StdZn,
        F: CyclotomicRingDecomposition<R::Type> + RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone,
        NTTRingBase<R, F, A>: CyclotomicRing + RingExtension<BaseRing = R>,
        G: Fn(&[usize]) -> ZnEl
{
    let m = H.len(dim_index);
    let g = H.shift_galois_element(dim_index, -1);
    let smaller_galois_group_mulrepr = Zn::new(4 * l as u64);
    let red = ReductionMap::new(H.galois_group_mulrepr(), &smaller_galois_group_mulrepr).unwrap();
    assert_el_eq!(&smaller_galois_group_mulrepr, &smaller_galois_group_mulrepr.one(), &smaller_galois_group_mulrepr.pow(red.map(g), l));

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
        TwiddleFactor::PosPowerZeta(pow) => H.slot_ring().clone_el(&zeta_power_table.get_power(H.galois_group_mulrepr().smallest_positive_lift(pow))),
        TwiddleFactor::NegPowerZeta(pow) => H.slot_ring().negate(H.slot_ring().clone_el(&zeta_power_table.get_power(H.galois_group_mulrepr().smallest_positive_lift(pow)))),
        TwiddleFactor::Zero => H.slot_ring().zero()
    };

    let forward_galois_element = H.shift_galois_element(dim_index, l as i64 / 2);
    let backward_galois_element = H.shift_galois_element(dim_index, -(l as i64 / 2));

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
            TwiddleFactor::NegPowerZeta(H.galois_group_mulrepr().mul(H.galois_group_mulrepr().pow(g, idx_in_block - l / 2), row_autos(&idxs)))
        } else {
            TwiddleFactor::PosPowerZeta(H.galois_group_mulrepr().zero())
        }
    }).map(&pow_of_zeta));

    let backward_mask = H.from_slot_vec(H.slot_iter(|idxs| {
        let idx_in_block = idxs[dim_index] % l;
        if idx_in_block < l / 2 {
            TwiddleFactor::PosPowerZeta(H.galois_group_mulrepr().mul(H.galois_group_mulrepr().pow(g, idx_in_block), row_autos(&idxs)))
        } else {
            TwiddleFactor::Zero
        }
    }).map(&pow_of_zeta));

    let mut result = LinearTransform {
        data: vec![
            (
                H.galois_group_mulrepr().one(),
                diagonal_mask,
                (0..H.dim_count()).map(|_| 0).collect()
            ),
            (
                forward_galois_element,
                forward_mask,
                (0..H.dim_count()).map(|i| if i == dim_index { l as i64 / 2 } else { 0 }).collect()
            ),
            (
                backward_galois_element,
                backward_mask,
                (0..H.dim_count()).map(|i| if i == dim_index { -(l as i64 / 2) } else { 0 }).collect()
            )
        ]
    };
    result.optimize(H);
    return result;
}

///
/// Inverse of [`pow2_bitreversed_dwt_butterfly()`]
/// 
fn pow2_bitreversed_inv_dwt_butterfly<'b, R, F, A, G>(H: &HypercubeIsomorphism<'b, R, F, A>, dim_index: usize, l: usize, root_of_unity_4l: El<SlotRing<'b, R, A>>, row_autos: G) -> LinearTransform<R, F, A>
    where R: RingStore,
        R::Type: StdZn,
        F: CyclotomicRingDecomposition<R::Type> + RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone,
        NTTRingBase<R, F, A>: CyclotomicRing + RingExtension<BaseRing = R>,
        G: Fn(&[usize]) -> ZnEl
{
    let m = H.len(dim_index);
    let g = H.shift_galois_element(dim_index, -1);
    let smaller_galois_group_mulrepr = Zn::new(4 * l as u64);
    let red = ReductionMap::new(H.galois_group_mulrepr(), &smaller_galois_group_mulrepr).unwrap();
    assert_el_eq!(&smaller_galois_group_mulrepr, &smaller_galois_group_mulrepr.one(), &smaller_galois_group_mulrepr.pow(red.map(g), l));

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
        TwiddleFactor::PosPowerZeta(pow) => H.slot_ring().clone_el(&zeta_power_table.get_power(H.galois_group_mulrepr().smallest_positive_lift(pow))),
        TwiddleFactor::NegPowerZeta(pow) => H.slot_ring().negate(H.slot_ring().clone_el(&zeta_power_table.get_power(H.galois_group_mulrepr().smallest_positive_lift(pow)))),
        TwiddleFactor::Zero => H.slot_ring().zero()
    };

    let forward_galois_element = H.shift_galois_element(dim_index, l as i64 / 2);
    let backward_galois_element = H.shift_galois_element(dim_index, -(l as i64 / 2));

    let inv_2 = H.ring().base_ring().invert(&H.ring().base_ring().int_hom().map(2)).unwrap();

    let mut forward_mask = H.from_slot_vec(H.slot_iter(|idxs| {
        let idx_in_block = idxs[dim_index] % l;
        if idx_in_block >= l / 2 {
            TwiddleFactor::PosPowerZeta(H.galois_group_mulrepr().mul(H.galois_group_mulrepr().negate(H.galois_group_mulrepr().pow(g, idx_in_block - l / 2)), row_autos(&idxs)))
        } else {
            TwiddleFactor::Zero
        }
    }).map(&pow_of_zeta));
    H.ring().inclusion().mul_assign_map_ref(&mut forward_mask, &inv_2);

    let mut diagonal_mask = H.from_slot_vec(H.slot_iter(|idxs| {
        let idx_in_block = idxs[dim_index] % l;
        if idx_in_block >= l / 2 {
            TwiddleFactor::NegPowerZeta(H.galois_group_mulrepr().mul(H.galois_group_mulrepr().negate(H.galois_group_mulrepr().pow(g, idx_in_block - l / 2)), row_autos(&idxs)))
        } else {
            TwiddleFactor::PosPowerZeta(H.galois_group_mulrepr().zero())
        }
    }).map(&pow_of_zeta));
    H.ring().inclusion().mul_assign_map_ref(&mut diagonal_mask, &inv_2);

    let mut backward_mask = H.from_slot_vec(H.slot_iter(|idxs| {
        let idx_in_block = idxs[dim_index] % l;
        if idx_in_block < l / 2 {
            TwiddleFactor::PosPowerZeta(H.galois_group_mulrepr().zero())
        } else {
            TwiddleFactor::Zero
        }
    }).map(&pow_of_zeta));
    H.ring().inclusion().mul_assign_map_ref(&mut backward_mask, &inv_2);

    let mut result = LinearTransform {
        data: vec![
            (
                H.galois_group_mulrepr().one(),
                diagonal_mask,
                (0..H.dim_count()).map(|_| 0).collect()
            ),
            (
                forward_galois_element,
                forward_mask,
                (0..H.dim_count()).map(|i| if i == dim_index { l as i64 / 2 } else { 0 }).collect()
            ),
            (
                backward_galois_element,
                backward_mask,
                (0..H.dim_count()).map(|i| if i == dim_index { -(l as i64 / 2) } else { 0 }).collect()
            )
        ]
    };
    result.optimize(&H);
    #[cfg(debug_assertions)] {
        let expected = pow2_bitreversed_dwt_butterfly(H, dim_index, l, H.slot_ring().clone_el(&zeta_power_table.get_power(1)), row_autos).inverse(&H);
        
        for (d, c, _) in &expected.data {
            println!("{}, {}", H.galois_group_mulrepr().format(d), H.ring().format(c));
        }
        println!();
        for (d, c, _) in &result.data {
            println!("{}, {}", H.galois_group_mulrepr().format(d), H.ring().format(c));
        }
        println!();
        debug_assert!(result.eq(&expected, &H));
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
/// sum_(0 <= i < m) a(bitrev(i)) * zeta^(n / (4m) * row_autos(idxs) * g^j)
/// ```
/// for `j` from `0` to `m`.
/// Here `zeta` is the canonical generator of the slot ring, which is a primitive `n`-th root of unity.
/// 
fn pow2_bitreversed_dwt<R, F, A, G>(H: &HypercubeIsomorphism<R, F, A>, dim_index: usize, row_autos: G) -> Vec<LinearTransform<R, F, A>>
    where R: RingStore,
        R::Type: StdZn,
        F: CyclotomicRingDecomposition<R::Type> + RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone,
        NTTRingBase<R, F, A>: CyclotomicRing + RingExtension<BaseRing = R>,
        G: Fn(&[usize]) -> ZnEl
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
fn pow2_bitreversed_inv_dwt<R, F, A, G>(H: &HypercubeIsomorphism<R, F, A>, dim_index: usize, row_autos: G) -> Vec<LinearTransform<R, F, A>>
    where R: RingStore,
        R::Type: StdZn,
        F: CyclotomicRingDecomposition<R::Type> + RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone,
        NTTRingBase<R, F, A>: CyclotomicRing + RingExtension<BaseRing = R>,
        G: Fn(&[usize]) -> ZnEl
{
    let m = H.len(dim_index);
    let log2_m = ZZ.abs_log2_ceil(&(m as i64)).unwrap();
    assert!(m == 1 << log2_m, "pow2_bitreversed_dwt() only valid for hypercube dimensions that have a power-of-2 length");
    assert!((H.ring().n() / m) % 4 == 0, "pow2_bitreversed_dwt() only possible if there is a 4m-th primitive root of unity");

    let zeta = H.slot_ring().pow(H.slot_ring().canonical_gen(), H.ring().n() / m / 4);
    debug_assert!(is_prim_root_of_unity(H.slot_ring(), &H.slot_ring().canonical_gen(), H.ring().n()));
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
/// If `p = 1 mod 4`, the slots are enumerated by `i, j` with `i in {0, 1}` and `0 <= j < l/2`. The returned linear
/// transform will then put the value of slot `(0, j)` into the coefficient of `X^(bitrev(j, l/2) * n/(2l))` and the value of slot
/// `(1, j)` into the coefficient of `X^(bitrev(j, l/2) * n/(2l) + n/4)`. 
/// 
/// If `p = 3 mod 4`, the slots are enumerate by `j` with `0 <= j < l/2`. The returned linear transform will then
/// put the value of slot `j` into the coefficient of `X^(bitrev(j, l) * n/(4l))`.
/// 
pub fn pow2_slots_to_coeffs_thin<R, F, A>(H: &HypercubeIsomorphism<R, F, A>) -> Vec<LinearTransform<R, F, A>>
    where R: RingStore,
        R::Type: StdZn,
        F: CyclotomicRingDecomposition<R::Type> + RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone,
        NTTRingBase<R, F, A>: CyclotomicRing + RingExtension<BaseRing = R>,
{
    let n = H.ring().get_ring().n();
    let log2_n = ZZ.abs_log2_ceil(&(n as i64)).unwrap();
    assert!(n == 1 << log2_n);

    if H.dim_count() == 2 {
        assert_eq!(2, H.len(0));
        
        let mut result = Vec::new();

        let zeta4 = H.slot_ring().pow(H.slot_ring().canonical_gen(), H.ring().n() / 4);

        result.push(LinearTransform {
            data: vec![
                (
                    H.galois_group_mulrepr().one(),
                    H.from_slot_vec(H.slot_iter(|idxs| if idxs[0] == 0 {
                        H.slot_ring().one()
                    } else {
                        debug_assert!(idxs[0] == 1);
                        H.slot_ring().negate(H.slot_ring().clone_el(&zeta4))
                    })),
                    vec![0, 0]
                ), 
                (
                    H.galois_group_mulrepr().neg_one(),
                    H.from_slot_vec(H.slot_iter(|idxs| if idxs[0] == 0 {
                        H.slot_ring().clone_el(&zeta4)
                    } else {
                        debug_assert!(idxs[0] == 1);
                        H.slot_ring().one()
                    })),
                    vec![1, 0]
                )
            ]
        });

        result.extend(pow2_bitreversed_dwt(H, 1, |idxs| if idxs[0] == 0 {
            H.galois_group_mulrepr().one()
        } else {
            debug_assert!(idxs[0] == 1);
            H.galois_group_mulrepr().neg_one()
        }));
        return result;
    } else {
        assert_eq!(H.dim_count(), 1);
        return pow2_bitreversed_dwt(H, 0, |_| H.galois_group_mulrepr().one());
    }
}

///
/// Inverse to [`pow2_slots_to_coeffs_thin()`]
/// 
pub fn pow2_coeffs_to_slots_thin<R, F, A>(H: &HypercubeIsomorphism<R, F, A>) -> Vec<LinearTransform<R, F, A>>
    where R: RingStore,
        R::Type: StdZn,
        F: CyclotomicRingDecomposition<R::Type> + RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone,
        NTTRingBase<R, F, A>: CyclotomicRing + RingExtension<BaseRing = R>,
{
    let n = H.ring().get_ring().n();
    let log2_n = ZZ.abs_log2_ceil(&(n as i64)).unwrap();
    assert!(n == 1 << log2_n);

    if H.dim_count() == 2 {
        assert_eq!(2, H.len(0));
        
        let mut result = Vec::new();

        let zeta4_inv = H.slot_ring().pow(H.slot_ring().canonical_gen(), 3 * H.ring().n() / 4);
        let inv_2 = H.ring().base_ring().invert(&H.ring().base_ring().int_hom().map(2)).unwrap();

        result.extend(pow2_bitreversed_inv_dwt(H, 1, |idxs| if idxs[0] == 0 {
            H.galois_group_mulrepr().one()
        } else {
            debug_assert!(idxs[0] == 1);
            H.galois_group_mulrepr().neg_one()
        }));

        result.push(LinearTransform {
            data: vec![
                (
                    H.galois_group_mulrepr().one(),
                    H.ring().inclusion().mul_map(H.from_slot_vec(H.slot_iter(|idxs| if idxs[0] == 0 {
                        H.slot_ring().one()
                    } else {
                        debug_assert!(idxs[0] == 1);
                        H.slot_ring().negate(H.slot_ring().clone_el(&zeta4_inv))
                    })), H.ring().base_ring().clone_el(&inv_2)),
                    vec![0, 0]
                ), 
                (
                    H.galois_group_mulrepr().neg_one(),
                    H.ring().inclusion().mul_map(H.from_slot_vec(H.slot_iter(|idxs| if idxs[0] == 0 {
                        H.slot_ring().one()
                    } else {
                        debug_assert!(idxs[0] == 1);
                        H.slot_ring().clone_el(&zeta4_inv)
                    })), inv_2),
                    vec![1, 0]
                )
            ]
        });

        return result;
    } else {
        assert_eq!(H.dim_count(), 1);
        return pow2_bitreversed_inv_dwt(H, 0, |_| H.galois_group_mulrepr().one());
    }
}

#[cfg(test)]
use crate::rings::pow2_cyclotomic::DefaultPow2CyclotomicNTTRingBase;

#[test]
fn test_pow2_bitreversed_dwt() {
    // `F23[X]/(X^16 + 1) ~ F_(23^4)^4`
    let ring = DefaultPow2CyclotomicNTTRingBase::new(Zn::new(23), 4);
    let H = HypercubeIsomorphism::new(ring.get_ring());

    let mut current = ring_literal!(&ring, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    for T in pow2_bitreversed_dwt(&H, 0, |_| H.galois_group_mulrepr().one()) {
        current = ring.get_ring().compute_linear_transform(&current, &T);
    }

    assert_el_eq!(&ring, &ring_literal!(&ring, [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), &current);

    let mut current = H.from_slot_vec([1, 2, 3, 4].into_iter().map(|n| H.slot_ring().int_hom().map(n)));
    for T in pow2_bitreversed_dwt(&H, 0, |_| H.galois_group_mulrepr().one()) {
        current = ring.get_ring().compute_linear_transform(&current, &T);
    }

    // remember that input is in bitreversed order
    assert_el_eq!(&ring, &ring_literal!(&ring, [1, 0, 3, 0, 2, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0]), &current);
}

#[test]
fn test_pow2_slots_to_coeffs_thin() {
    // `F23[X]/(X^32 + 1) ~ F_(23^8)^4`
    let ring = DefaultPow2CyclotomicNTTRingBase::new(Zn::new(23), 5);
    let H = HypercubeIsomorphism::new(ring.get_ring());

    let mut current = H.from_slot_vec([1, 2, 3, 4].into_iter().map(|n| H.slot_ring().int_hom().map(n)));
    for T in pow2_slots_to_coeffs_thin(&H) {
        current = ring.get_ring().compute_linear_transform(&current, &T);
    }

    assert_el_eq!(&ring, &ring_literal!(&ring, [1, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), &current);
    
    // `F97[X]/(X^32 + 1) ~ F_(97^2)^16`
    let ring = DefaultPow2CyclotomicNTTRingBase::new(Zn::new(97), 5);
    let H = HypercubeIsomorphism::new(ring.get_ring());
    
    let mut current = H.from_slot_vec((1..17).map(|n| H.slot_ring().int_hom().map(n)));
    for T in pow2_slots_to_coeffs_thin(&H) {
        current = ring.get_ring().compute_linear_transform(&current, &T);
    }

    assert_el_eq!(&ring, &ring_literal!(&ring, [1, 0, 5, 0, 3, 0, 7, 0, 2, 0, 6, 0, 4, 0, 8, 0, 9, 0, 13, 0, 11, 0, 15, 0, 10, 0, 14, 0, 12, 0, 16, 0]), &current);
}

#[test]
fn test_pow2_coeffs_to_slots_thin() {
    // `F23[X]/(X^32 + 1) ~ F_(23^8)^4`
    let ring = DefaultPow2CyclotomicNTTRingBase::new(Zn::new(23), 5);
    let H = HypercubeIsomorphism::new(ring.get_ring());

    for (transform, actual) in pow2_slots_to_coeffs_thin(&H).into_iter().rev().zip(pow2_coeffs_to_slots_thin(&H).into_iter()) {
        let expected = transform.inverse(&H);
        assert!(expected.eq(&actual, &H));
    }
    
    // `F97[X]/(X^32 + 1) ~ F_(97^2)^16`
    let ring = DefaultPow2CyclotomicNTTRingBase::new(Zn::new(97), 5);
    let H = HypercubeIsomorphism::new(ring.get_ring());
    
    for (transform, actual) in pow2_slots_to_coeffs_thin(&H).into_iter().rev().zip(pow2_coeffs_to_slots_thin(&H).into_iter()) {
        let expected = transform.inverse(&H);
        assert!(expected.eq(&actual, &H));
    }
}