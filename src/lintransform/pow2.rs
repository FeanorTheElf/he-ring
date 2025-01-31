use std::alloc::Allocator;
use std::alloc::Global;

use feanor_math::algorithms::fft::cooley_tuckey::bitreverse;
use feanor_math::algorithms::unity_root::is_prim_root_of_unity;
use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::homomorphism::Homomorphism;
use feanor_math::integer::IntegerRingStore;
use feanor_math::primitive_int::StaticRing;
use feanor_math::ring::*;
use feanor_math::rings::extension::*;
use feanor_math::rings::zn::zn_64::*;
use feanor_math::rings::zn::*;
use feanor_math::assert_el_eq;
use feanor_math::seq::VectorView;
use tracing::instrument;

use crate::circuit::PlaintextCircuit;
use crate::cyclotomic::*;
use crate::lintransform::matmul::*;
use crate::lintransform::trace::trace_circuit;
use crate::lintransform::PowerTable;
use crate::number_ring::hypercube::DefaultHypercube;
use crate::number_ring::hypercube::HypercubeStructure;
use crate::number_ring::hypercube::SlotRingOver;
use crate::number_ring::pow2_cyclotomic::Pow2CyclotomicNumberRing;
use crate::number_ring::quotient::NumberRingQuotientBase;
use crate::number_ring::{HECyclotomicNumberRing, HENumberRing};
use crate::number_ring::quotient::NumberRingQuotient;
use crate::number_ring::hypercube::HypercubeIsomorphism;

const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;

///
/// Works separately on each block `(b0, ..., b(l - 1))` of size `l = blocksize` along the given given hypercube dimension.
/// This function computes the length-`l` DWT
/// ```text
/// sum_(0 <= i < l) ai * 𝝵_(4l)^(i * g^j)
/// ``` 
/// from the length-`l/2` DWTs of the even-index resp. odd-index entries of `ai`. These two sub-DWTs are expected to be written
/// in the first resp. second half of the input block (i.e. not interleaved, this is where the "bitreversed" comes from).
/// Here `g` is the generator of the current hypercube dimension, i.e. usually `g = 5`.
/// 
/// More concretely, it is expected that the input to the linear transform is
/// ```text
/// bj = sum_(0 <= i < l/2) a(2i) * 𝝵_(4l)^(2 * i * g^j)              if j < l/2
/// bj = sum_(0 <= i < l/2) a(2i + 1) * 𝝵_(4l)^(2 * i * g^j)
///    = sum_(0 <= i < l/2) a(2i + 1) * 𝝵_(4l)^(2 * i * g^(j - l/2))  otherwise
/// ```
/// In this case, the output is
/// ```text
/// bj = sum_(0 <= i < l) ai * 𝝵_(4l)^(i * g^j)
/// ```
/// 
/// # Notes
///  - This does not compute the evaluation at all primitive `4l`-th roots of unity, but only at half of them - namely `𝝵_(4l)^(g^j)` for all `j`.
///    In particular, `g` does not generate `(Z/4lZ)*`, but `<g>` is an index 2 subgroup of it.
///  - `row_autos` can be given to use different `𝝵`s for each block; in particular, for the block with hypercube indices `idxs`, the DWT with root
///    of unity `𝝵 = root_of_unity_4l^row_autos(idxs)` is used. Note that the index passed to `row_autos` is the hypercube index of some element in the
///    block. It does not make sense for `row_autos` to behave differently on different indices in the same block, this will lead to 
///    `pow2_bitreversed_dwt_butterfly` to give nonsensical results. If you pass `row_autos = |_| H.cyclotomic_index_ring().one()` then this uses the same
///    roots of unity everywhere, i.e. results in the behavior as outlined above.
/// u
fn pow2_bitreversed_dwt_butterfly<G, NumberRing>(H: &DefaultHypercube<NumberRing>, dim_index: usize, l: usize, root_of_unity_4l: El<SlotRingOver<Zn>>, row_autos: G) -> MatmulTransform<NumberRing>
    where G: Fn(&[usize]) -> CyclotomicGaloisGroupEl,
        NumberRing: HECyclotomicNumberRing + Clone
{
    let m = H.hypercube().m(dim_index);
    let log2_m = ZZ.abs_log2_ceil(&(m as i64)).unwrap();
    assert_eq!(m, 1 << log2_m);

    let g = H.hypercube().map_1d(dim_index, -1);
    let smaller_cyclotomic_index_ring = Zn::new(4 * l as u64);
    let Gal = H.galois_group();
    let red = ZnReductionMap::new(Gal.underlying_ring(), &smaller_cyclotomic_index_ring).unwrap();
    assert_el_eq!(&smaller_cyclotomic_index_ring, &smaller_cyclotomic_index_ring.one(), &smaller_cyclotomic_index_ring.pow(red.map(Gal.to_ring_el(g)), l));

    let l = l;
    assert!(l > 1);
    assert!(m % l == 0);
    let zeta_power_table = PowerTable::new(H.slot_ring(), root_of_unity_4l, 4 * l);

    enum TwiddleFactor {
        Zero, PosPowerZeta(ZnEl), NegPowerZeta(ZnEl)
    }

    let pow_of_zeta = |factor: TwiddleFactor| match factor {
        TwiddleFactor::PosPowerZeta(pow) => H.slot_ring().clone_el(&zeta_power_table.get_power(Gal.underlying_ring().smallest_positive_lift(pow))),
        TwiddleFactor::NegPowerZeta(pow) => H.slot_ring().negate(H.slot_ring().clone_el(&zeta_power_table.get_power(Gal.underlying_ring().smallest_positive_lift(pow)))),
        TwiddleFactor::Zero => H.slot_ring().zero()
    };

    let forward_mask = H.from_slot_values(H.hypercube().hypercube_iter(|idxs| {
        let idx_in_block = idxs[dim_index] % l;
        if idx_in_block >= l / 2 {
            TwiddleFactor::PosPowerZeta(Gal.underlying_ring().zero())
        } else {
            TwiddleFactor::Zero
        }
    }).map(&pow_of_zeta));

    let diagonal_mask = H.from_slot_values(H.hypercube().hypercube_iter(|idxs| {
        let idx_in_block = idxs[dim_index] % l;
        if idx_in_block >= l / 2 {
            TwiddleFactor::NegPowerZeta(Gal.to_ring_el(Gal.mul(Gal.pow(g, (idx_in_block - l / 2) as i64), row_autos(&idxs))))
        } else {
            TwiddleFactor::PosPowerZeta(Gal.underlying_ring().zero())
        }
    }).map(&pow_of_zeta));

    let backward_mask = H.from_slot_values(H.hypercube().hypercube_iter(|idxs| {
        let idx_in_block = idxs[dim_index] % l;
        if idx_in_block < l / 2 {
            TwiddleFactor::PosPowerZeta(Gal.to_ring_el(Gal.mul(Gal.pow(g, idx_in_block as i64), row_autos(&idxs))))
        } else {
            TwiddleFactor::Zero
        }
    }).map(&pow_of_zeta));

    let result = MatmulTransform::linear_combine_shifts(H, [
        (
            (0..H.hypercube().dim_count()).map(|i| 0).collect::<Vec<_>>(),
            diagonal_mask
        ),
        (
            (0..H.hypercube().dim_count()).map(|i| if i == dim_index { l as i64 / 2 } else { 0 }).collect::<Vec<_>>(),
            forward_mask
        ),
        (
            (0..H.hypercube().dim_count()).map(|i| if i == dim_index { -(l as i64) / 2 } else { 0 }).collect::<Vec<_>>(),
            backward_mask
        )
    ].iter().map(|(shift, coeff)| (shift.copy_els(), H.ring().clone_el(coeff))));
    return result;
}

///
/// Inverse of [`pow2_bitreversed_dwt_butterfly()`]
/// 
fn pow2_bitreversed_inv_dwt_butterfly<G, NumberRing>(H: &DefaultHypercube<NumberRing>, dim_index: usize, l: usize, root_of_unity_4l: El<SlotRingOver<Zn>>, row_autos: G) -> MatmulTransform<NumberRing>
    where G: Fn(&[usize]) -> CyclotomicGaloisGroupEl,
        NumberRing: HECyclotomicNumberRing + Clone
{
    let m = H.hypercube().m(dim_index);
    let log2_m = ZZ.abs_log2_ceil(&(m as i64)).unwrap();
    assert_eq!(m, 1 << log2_m);

    let g = H.hypercube().map_1d(dim_index, -1);
    let smaller_cyclotomic_index_ring = Zn::new(4 * l as u64);
    let Gal = H.galois_group();
    let red = ZnReductionMap::new(Gal.underlying_ring(), &smaller_cyclotomic_index_ring).unwrap();
    assert_el_eq!(&smaller_cyclotomic_index_ring, &smaller_cyclotomic_index_ring.one(), &smaller_cyclotomic_index_ring.pow(red.map(Gal.to_ring_el(g)), l));

    let l = l;
    assert!(l > 1);
    assert!(m % l == 0);
    let zeta_power_table = PowerTable::new(H.slot_ring(), root_of_unity_4l, 4 * l);

    enum TwiddleFactor {
        Zero, PosPowerZeta(ZnEl), NegPowerZeta(ZnEl)
    }

    let pow_of_zeta = |factor: TwiddleFactor| match factor {
        TwiddleFactor::PosPowerZeta(pow) => H.slot_ring().clone_el(&zeta_power_table.get_power(Gal.underlying_ring().smallest_positive_lift(pow))),
        TwiddleFactor::NegPowerZeta(pow) => H.slot_ring().negate(H.slot_ring().clone_el(&zeta_power_table.get_power(Gal.underlying_ring().smallest_positive_lift(pow)))),
        TwiddleFactor::Zero => H.slot_ring().zero()
    };

    let inv_2 = H.ring().base_ring().invert(&H.ring().base_ring().int_hom().map(2)).unwrap();

    let mut forward_mask = H.from_slot_values(H.hypercube().hypercube_iter(|idxs| {
        let idx_in_block = idxs[dim_index] % l;
        if idx_in_block >= l / 2 {
            TwiddleFactor::PosPowerZeta(Gal.to_ring_el(Gal.mul(Gal.negate(Gal.pow(g, (idx_in_block - l / 2) as i64)), row_autos(&idxs))))
        } else {
            TwiddleFactor::Zero
        }
    }).map(&pow_of_zeta));
    H.ring().inclusion().mul_assign_ref_map(&mut forward_mask, &inv_2);

    let mut diagonal_mask = H.from_slot_values(H.hypercube().hypercube_iter(|idxs| {
        let idx_in_block = idxs[dim_index] % l;
        if idx_in_block >= l / 2 {
            TwiddleFactor::NegPowerZeta(Gal.to_ring_el(Gal.mul(Gal.negate(Gal.pow(g, (idx_in_block - l / 2) as i64)), row_autos(&idxs))))
        } else {
            TwiddleFactor::PosPowerZeta(Gal.underlying_ring().zero())
        }
    }).map(&pow_of_zeta));
    H.ring().inclusion().mul_assign_ref_map(&mut diagonal_mask, &inv_2);

    let mut backward_mask = H.from_slot_values(H.hypercube().hypercube_iter(|idxs| {
        let idx_in_block = idxs[dim_index] % l;
        if idx_in_block < l / 2 {
            TwiddleFactor::PosPowerZeta(Gal.underlying_ring().zero())
        } else {
            TwiddleFactor::Zero
        }
    }).map(&pow_of_zeta));
    H.ring().inclusion().mul_assign_ref_map(&mut backward_mask, &inv_2);

    let result = MatmulTransform::linear_combine_shifts(H, [
        (
            (0..H.hypercube().dim_count()).map(|i| 0).collect::<Vec<_>>(),
            diagonal_mask
        ),
        (
            (0..H.hypercube().dim_count()).map(|i| if i == dim_index { l as i64 / 2 } else { 0 }).collect::<Vec<_>>(),
            forward_mask
        ),
        (
            (0..H.hypercube().dim_count()).map(|i| if i == dim_index { -(l as i64) / 2 } else { 0 }).collect::<Vec<_>>(),
            backward_mask
        )
    ].iter().map(|(shift, coeff)| (shift.copy_els(), H.ring().clone_el(coeff))));
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
/// sum_(0 <= i < m) a(bitrev(i)) * 𝝵^(n / (4m) * row_autos(idxs) * g^j)
/// ```
/// for `j` from `0` to `m`.
/// Here `𝝵` is the canonical generator of the slot ring, which is a primitive `n`-th root of unity.
/// 
fn pow2_bitreversed_dwt<G, NumberRing>(H: &DefaultHypercube<NumberRing>, dim_index: usize, row_autos: G) -> Vec<MatmulTransform<NumberRing>>
    where G: Fn(&[usize]) -> CyclotomicGaloisGroupEl,
        NumberRing: HECyclotomicNumberRing + Clone
{
    let m = H.hypercube().m(dim_index);
    let log2_m = ZZ.abs_log2_ceil(&(m as i64)).unwrap();
    assert_eq!(m, 1 << log2_m);
    assert!((H.ring().n() / m) % 4 == 0, "pow2_bitreversed_dwt() only possible if there is a 4m-th primitive root of unity");

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
#[instrument(skip_all)]
fn pow2_bitreversed_inv_dwt<G, NumberRing>(H: &DefaultHypercube<NumberRing>, dim_index: usize, row_autos: G) -> Vec<MatmulTransform<NumberRing>>
    where G: Fn(&[usize]) -> CyclotomicGaloisGroupEl,
        NumberRing: HECyclotomicNumberRing + Clone
{
    let m = H.hypercube().m(dim_index);
    let log2_m = ZZ.abs_log2_ceil(&(m as i64)).unwrap();
    assert_eq!(m, 1 << log2_m);
    assert!((H.ring().n() / m) % 4 == 0, "pow2_bitreversed_dwt() only possible if there is a 4m-th primitive root of unity");

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
/// Computes the <https://ia.cr/2024/153>-style Slots-to-Coeffs linear transform for the thin-bootstrapping case,
/// i.e. where all slots contain elements in `Z/pZ`.
/// 
/// In the case `p = 1 mod 4`, the slots are enumerated by `i, j` with `0 <= i < m/2` and `j in {0, 1}`. If `p = 1 mod 4`.
/// Then the returned linear transform will then put the value of slot `(i, 0)` into the coefficient of `X^(bitrev(i, m/2) * n/(2m))`
/// and the value of slot `(i, 1)` into the coefficient of `X^(bitrev(i, m/2) * n/(2m) + n/4)`.
/// 
/// If `p = 3 mod 4`, the slots are enumerated by `i` with `0 <= i < m` and the transform will put the value of slot `i` into the coefficient
/// of `X^(bitrev(i, m) * n/(4m))`
/// 
#[instrument(skip_all)]
pub fn slots_to_coeffs_thin<NumberRing>(H: &DefaultHypercube<NumberRing>) -> Vec<MatmulTransform<NumberRing>>
    where NumberRing: HECyclotomicNumberRing + Clone
{
    let n = H.ring().get_ring().n();
    let log2_n = ZZ.abs_log2_ceil(&(n as i64)).unwrap();
    assert!(n == 1 << log2_n);

    if H.hypercube().dim_count() == 2 {
        // this is the `p = 1 mod 4` case
        assert_eq!(2, H.hypercube().m(1));
        let zeta4 = H.slot_ring().pow(H.slot_ring().canonical_gen(), H.ring().n() as usize / 4);
        let mut result = Vec::new();

        // we first combine `ai0` and `ai1` to `(ai0 + 𝝵^(n/4) ai1, ai0 - 𝝵^(n/4) ai1)`
        result.push(MatmulTransform::linear_combine_shifts(H, [
            (
                (0..H.hypercube().dim_count()).map(|i| 0).collect::<Vec<_>>(),
                H.from_slot_values(H.hypercube().hypercube_iter(|idxs| if idxs[1] == 0 {
                    H.slot_ring().one()
                } else {
                    debug_assert!(idxs[1] == 1);
                    H.slot_ring().negate(H.slot_ring().clone_el(&zeta4))
                }))
            ),
            (
                (0..H.hypercube().dim_count()).map(|i| if i == 1 { 1 } else { 0 }).collect::<Vec<_>>(),
                H.from_slot_values(H.hypercube().hypercube_iter(|idxs| if idxs[1] == 0 {
                    H.slot_ring().clone_el(&zeta4)
                } else {
                    debug_assert!(idxs[1] == 1);
                    H.slot_ring().one()
                }))
            )
        ].iter().map(|(shift, coeff)| (shift.copy_els(), H.ring().clone_el(coeff)))));

        // then map the `ai0 + 𝝵^(n/4) ai1` to `sum_i (ai0 + 𝝵^(n/4) ai1) 𝝵^(n/(2m) i g^k)` for each slot `(k, 0)`, and similarly
        // for the slots `(*, 1)`. The negation in the second hypercolumn comes from the fact that `-𝝵^(n/4) = 𝝵^(-n/4)`
        result.extend(pow2_bitreversed_dwt(H, 0, |idxs| if idxs[1] == 0 {
            H.galois_group().identity()
        } else {
            debug_assert!(idxs[1] == 1);
            H.galois_group().negate(H.galois_group().identity())
        }));
        
        return result;
    } else {
        // this is the `p = 3 mod 4` case
        assert_eq!(1, H.hypercube().dim_count());
        return pow2_bitreversed_dwt(H, 0, |_idxs| H.galois_group().identity());
    }
}

///
/// This is the inverse to [`slots_to_coeffs_thin()`]. Note that it is not the
/// "Coeffs-to-Slots" map, as it does not discard unused factors. However, it is not
/// too hard to build the "coeffs-to-slots" map from this, see [`coeffs_to_slots_thin()`]. 
/// 
#[instrument(skip_all)]
pub fn slots_to_coeffs_thin_inv<NumberRing>(H: &DefaultHypercube<NumberRing>) -> Vec<MatmulTransform<NumberRing>>
    where NumberRing: HECyclotomicNumberRing + Clone
{
    let n = H.ring().get_ring().n();
    let log2_n = ZZ.abs_log2_ceil(&(n as i64)).unwrap();
    assert!(n == 1 << log2_n);

    if H.hypercube().dim_count() == 2 {
        assert_eq!(2, H.hypercube().m(1));
        let zeta4_inv = H.slot_ring().pow(H.slot_ring().canonical_gen(), 3 * H.ring().n() as usize / 4);
        let two_inv = H.ring().base_ring().invert(&H.slot_ring().base_ring().int_hom().map(2)).unwrap();
        let mut result = Vec::new();

        result.extend(pow2_bitreversed_inv_dwt(H, 0, |idxs| if idxs[1] == 0 {
            H.galois_group().identity()
        } else {
            debug_assert!(idxs[1] == 1);
            H.galois_group().negate(H.galois_group().identity())
        }));

        result.push(MatmulTransform::linear_combine_shifts(H, [
            (
                (0..H.hypercube().dim_count()).map(|i| 0).collect::<Vec<_>>(),
                H.ring().inclusion().mul_map(H.from_slot_values(H.hypercube().hypercube_iter(|idxs| if idxs[1] == 0 {
                    H.slot_ring().one()
                } else {
                    debug_assert!(idxs[1] == 1);
                    H.slot_ring().negate(H.slot_ring().clone_el(&zeta4_inv))
                })), H.ring().base_ring().clone_el(&two_inv))
            ),
            (
                (0..H.hypercube().dim_count()).map(|i| if i == 1 { 1 } else { 0 }).collect::<Vec<_>>(),
                H.ring().inclusion().mul_map(H.from_slot_values(H.hypercube().hypercube_iter(|idxs| if idxs[1] == 0 {
                    H.slot_ring().one()
                } else {
                    debug_assert!(idxs[1] == 1);
                    H.slot_ring().clone_el(&zeta4_inv)
                })), two_inv)
            )
        ].iter().map(|(shift, coeff)| (shift.copy_els(), H.ring().clone_el(coeff)))));

        return result;
    } else {
        assert_eq!(1, H.hypercube().dim_count());
        return pow2_bitreversed_inv_dwt(H, 0, |_idxs| H.galois_group().identity());
    }
}

///
/// Computes the <https://ia.cr/2024/153>-style Coeffs-to-Slots linear transform for the thin-bootstrapping case,
/// i.e. where all slots contain elements in `Z/pZ`.
/// 
/// In the case `p = 1 mod 4`, the slots are enumerated by `i, j` with `0 <= i < m/2` and `j in {0, 1}`. If `p = 1 mod 4`.
/// Then the returned linear transform will put the value of the coefficient of `X^(i * n/(2m))` into slot `(bitrev(i, m/2), 0)` 
/// and the value the coefficient of `X^(i * n/(2m) + n/4)` into slot `(bitrev(i, m/2), 1)`.
/// 
/// If `p = 3 mod 4`, the slots are enumerated by `i` with `0 <= i < m` and the transform will put the value of the coefficient
/// of `X^(i * n/(4m))` into slot `bitrev(i, m)`.
/// 
/// Note that the values of all other coefficients are discarded, hence this transform is not the inverse of [`slots_to_coeffs_thin()`].
/// However, `coeff_to_slots_thin()(slots_to_coeffs_thin()(x))` does give `x` for all thinly-packed plaintexts `x`, i.e. `x` where
/// each slot only contains an element in `Z/pZ`. 
/// 
#[instrument(skip_all)]
pub fn coeffs_to_slots_thin<NumberRing>(H: &DefaultHypercube<NumberRing>) -> PlaintextCircuit<NumberRingQuotientBase<NumberRing, Zn, Global>>
    where NumberRing: HECyclotomicNumberRing + Clone
{
    let log2_n = ZZ.abs_log2_ceil(&(H.hypercube().n() as i64)).unwrap();
    assert_eq!(H.hypercube().n(), 1 << log2_n);

    let mut result = slots_to_coeffs_thin_inv(H);
    let last = MatmulTransform::mult_scalar_slots(H, &H.slot_ring().inclusion().map(H.slot_ring().base_ring().invert(&H.slot_ring().base_ring().int_hom().map(H.slot_ring().rank() as i32)).unwrap()));
    *result.last_mut().unwrap() = result.last().unwrap().compose(&last, H);

    let trace_circuit = trace_circuit(H.ring().get_ring(), &H.galois_group(), H.hypercube().frobenius(1), H.slot_ring().rank());
    let result_circuit = MatmulTransform::to_circuit_many(result, H);
    return trace_circuit.compose(result_circuit, H.ring());
}

#[test]
fn test_slots_to_coeffs_thin() {
    // `F97[X]/(X^32 + 1) ~ F_(97^2)^16`
    let ring = NumberRingQuotientBase::new(Pow2CyclotomicNumberRing::new(64), Zn::new(97));
    let hypercube = HypercubeStructure::halevi_shoup_hypercube(CyclotomicGaloisGroup::new(64), 97);
    let H = HypercubeIsomorphism::new::<false>(&ring, hypercube);
    
    let mut current = H.from_slot_values((1..17).map(|n| H.slot_ring().int_hom().map(n)));
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
    let ring = NumberRingQuotientBase::new(Pow2CyclotomicNumberRing::new(64), Zn::new(23));
    let hypercube = HypercubeStructure::halevi_shoup_hypercube(CyclotomicGaloisGroup::new(64), 23);
    let H = HypercubeIsomorphism::new::<false>(&ring, hypercube);

    let mut current = H.from_slot_values([1, 2, 3, 4].into_iter().map(|n| H.slot_ring().int_hom().map(n)));
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
    let ring = NumberRingQuotientBase::new(Pow2CyclotomicNumberRing::new(64), Zn::new(23));
    let hypercube = HypercubeStructure::halevi_shoup_hypercube(CyclotomicGaloisGroup::new(64), 23);
    let H = HypercubeIsomorphism::new::<false>(&ring, hypercube);

    for (transform, actual) in slots_to_coeffs_thin(&H).into_iter().rev().zip(slots_to_coeffs_thin_inv(&H).into_iter()) {
        let expected = transform.inverse(&H);
        assert!(expected.eq(&actual, &H));
    }
    
    // `F97[X]/(X^32 + 1) ~ F_(97^2)^16`
    let ring = NumberRingQuotientBase::new(Pow2CyclotomicNumberRing::new(64), Zn::new(97));
    let hypercube = HypercubeStructure::halevi_shoup_hypercube(CyclotomicGaloisGroup::new(64), 97);
    let H = HypercubeIsomorphism::new::<false>(&ring, hypercube);
    
    for (transform, actual) in slots_to_coeffs_thin(&H).into_iter().rev().zip(slots_to_coeffs_thin_inv(&H).into_iter()) {
        let expected = transform.inverse(&H);
        assert!(expected.eq(&actual, &H));
    }
}

#[test]
fn test_coeffs_to_slots_thin() {
    // `F97[X]/(X^32 + 1) ~ F_(97^2)^16`
    let ring = NumberRingQuotientBase::new(Pow2CyclotomicNumberRing::new(64), Zn::new(97));
    let hypercube = HypercubeStructure::halevi_shoup_hypercube(CyclotomicGaloisGroup::new(64), 97);
    let H = HypercubeIsomorphism::new::<false>(&ring, hypercube);
    
    let mut input = [0; 32];
    for i in 0..8 {
        for j in 0..2 {
            input[bitreverse(i, 3) * 2 + j * 16] = i * 2 + j + 1;
            input[bitreverse(i, 3) * 2 + j * 16 + 1] = i * 2 + j + 1 + 16;
        }
    }
    let current = ring_literal!(&ring, input);
    let circuit = coeffs_to_slots_thin(&H);
    let actual = circuit.evaluate(std::slice::from_ref(&current), ring.identity()).pop().unwrap();
    let expected = H.from_slot_values((1..17).map(|n| H.slot_ring().int_hom().map(n)));
    assert_el_eq!(&ring, &expected, &actual);

    // `F23[X]/(X^32 + 1) ~ F_(23^8)^4`
    let ring = NumberRingQuotientBase::new(Pow2CyclotomicNumberRing::new(64), Zn::new(23));
    let hypercube = HypercubeStructure::halevi_shoup_hypercube(CyclotomicGaloisGroup::new(64), 23);
    let H = HypercubeIsomorphism::new::<false>(&ring, hypercube);

    let mut input = [0; 32];
    input[4] = 1;
    input[16] = 1;

    let current = ring_literal!(&ring, input);
    let circuit = coeffs_to_slots_thin(&H);
    let actual = circuit.evaluate(std::slice::from_ref(&current), ring.identity()).pop().unwrap();
    let expected = H.from_slot_values([0, 0, 1, 0].into_iter().map(|n| H.slot_ring().int_hom().map(n)));
    assert_el_eq!(&ring, &expected, &actual);

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
    let current = ring_literal!(&ring, input);
    let circuit = coeffs_to_slots_thin(&H);
    let actual = circuit.evaluate(std::slice::from_ref(&current), ring.identity()).pop().unwrap();
    let expected = H.from_slot_values([1, 2, 3, 4].into_iter().map(|n| H.slot_ring().int_hom().map(n)));
    assert_el_eq!(&ring, &expected, &actual);
}
