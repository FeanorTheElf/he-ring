use std::alloc::Allocator;

use feanor_math::homomorphism::*;
use feanor_math::ring::*;
use feanor_math::rings::extension::galois_field::GaloisFieldDyn;
use feanor_math::rings::zn::zn_64::*;
use feanor_math::rings::zn::ZnRing;
use feanor_math::rings::extension::FreeAlgebraStore;

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
    vec![LinearTransform {
        galois_elements: ((1 - m)..m).map(|s| H.shift_galois_element(dim_index, s)).collect(),
        coeffs: ((1 - m)..m).map(|s| unimplemented!()).collect()
    }]
}