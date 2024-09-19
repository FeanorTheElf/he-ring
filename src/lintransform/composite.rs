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

fn column_dwt_matrix<'a, 'b, R, F, A, G>(H: &'b HypercubeIsomorphism<'a, R, F, A>, dim_index: usize, row_autos: G) -> impl 'b + Fn(usize, usize, &[usize]) -> El<SlotRing<'a, R, A>>
    where R: RingStore,
        R::Type: StdZn,
        F: CyclotomicRingDecomposition<R::Type> + RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone,
        NTTRingBase<R, F, A>: CyclotomicRing + RingExtension<BaseRing = R>,
        G: 'b + Fn(&[usize]) -> ZnEl
{
    let zeta = H.slot_ring().pow(H.slot_ring().canonical_gen(), H.ring().n() / H.corresponding_factor_n(dim_index) as usize);
    let hom = H.galois_group_mulrepr().can_hom(&StaticRing::<i64>::RING).unwrap();
    move |i, j, idxs| H.slot_ring().pow(
        H.slot_ring().clone_el(&zeta), 
        H.galois_group_mulrepr().smallest_positive_lift(H.galois_group_mulrepr().prod([
            H.shift_galois_element(dim_index, -(i as i64)),
            hom.map(j as i64),
            row_autos(idxs)
        ].into_iter())) as usize
    )
}

fn column_dwt<R, F, A, G>(H: &HypercubeIsomorphism<R, F, A>, dim_index: usize, row_autos: G) -> Vec<LinearTransform<R, F, A>>
    where R: RingStore,
        R::Type: StdZn,
        F: CyclotomicRingDecomposition<R::Type> + RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone,
        NTTRingBase<R, F, A>: CyclotomicRing + RingExtension<BaseRing = R>,
        G: Fn(&[usize]) -> ZnEl
{
    // multiplication with the matrix `A(i, j) = zeta^(j * g^i)` if we consider an element as multiple vectors along the `dim_index`-th dimension
    vec![LinearTransform::matmul1d(
        H, 
        dim_index, 
        column_dwt_matrix(H, dim_index, row_autos), 
    )]
}

fn slots_to_coeffs_fat<R, F, A, G>(H: &HypercubeIsomorphism<R, F, A>, dim_index: usize, row_autos: G) -> Vec<LinearTransform<R, F, A>>
    where R: RingStore,
        R::Type: StdZn,
        F: CyclotomicRingDecomposition<R::Type> + RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone,
        NTTRingBase<R, F, A>: CyclotomicRing + RingExtension<BaseRing = R>,
        G: Fn(&[usize]) -> ZnEl
{
    unimplemented!()
}