
use std::alloc::Allocator;

use feanor_math::primitive_int::*;
use feanor_math::ring::*;
use feanor_math::rings::zn::zn_64::*;
use feanor_math::rings::zn::*;

use crate::complexfft::automorphism::*;
use crate::complexfft::complex_fft_ring::*;

pub mod pow2;
pub mod composite;

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
