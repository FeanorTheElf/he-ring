
use std::alloc::Allocator;

use feanor_math::algorithms::eea::signed_gcd;
use feanor_math::assert_el_eq;
use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::homomorphism::Homomorphism;
use feanor_math::iters::multi_cartesian_product;
use feanor_math::primitive_int::*;
use feanor_math::ring::*;
use feanor_math::rings::zn::zn_64::*;
use feanor_math::rings::zn::*;
use pow2::pow2_slots_to_coeffs_thin;

use crate::complexfft::automorphism::*;
use crate::complexfft::complex_fft_ring::*;
use crate::complexfft::pow2_cyclotomic::DefaultPow2CyclotomicCCFFTRingBase;
use crate::cyclotomic::CyclotomicRing;

pub mod pow2;
pub mod composite;

const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;

pub struct LinearTransform<R, F, A>
    where R: RingStore,
        R::Type: ZnRing,
        F: CyclotomicRingDecomposition<R::Type> + RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone,
        CCFFTRingBase<R, F, A>: CyclotomicRing + /* unfortunately, the type checker is not clever enough to know that this is always the case */ RingExtension<BaseRing = R>
{
    data: Vec<(ZnEl, El<CCFFTRing<R, F, A>>, Vec<i64>)>
}

impl<R, F, A> LinearTransform<R, F, A>
    where R: RingStore,
        R::Type: ZnRing,
        F: CyclotomicRingDecomposition<R::Type> + RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone,
        CCFFTRingBase<R, F, A>: CyclotomicRing + /* unfortunately, the type checker is not clever enough to know that this is always the case */ RingExtension<BaseRing = R>
{
    fn check_valid(&self, H: &HypercubeIsomorphism<R, F, A>) {
        for (g, _, steps) in &self.data {
            assert_el_eq!(
                H.galois_group_mulrepr(),
                g,
                H.galois_group_mulrepr().prod(steps.iter().enumerate().map(|(i, s)| H.shift_galois_element(i, *s)))
            );
        }
    }

    fn compose(&self, run_first: &LinearTransform<R, F, A>, H: &HypercubeIsomorphism<R, F, A>) -> Self {
        Self {
            data: self.data.iter().flat_map(|(self_g, self_coeff, self_indices)| run_first.data.iter().map(|(first_g, first_coeff, first_indices)| (
                H.galois_group_mulrepr().mul_ref(self_g, first_g), 
                H.ring().mul_ref_snd(H.ring().get_ring().apply_galois_action(*self_g, H.ring().clone_el(first_coeff)), self_coeff),
                self_indices.iter().zip(first_indices).map(|(self_i, first_i)| self_i + first_i).collect()
            ))).collect()
        }
    }

    fn identity(H: &HypercubeIsomorphism<R, F, A>) -> Self {
        Self {
            data: vec![(H.galois_group_mulrepr().one(), H.ring().one(), (0..H.dim_count()).map(|_| 0).collect())]
        }
    }

    fn optimize(&mut self, H: &HypercubeIsomorphism<R, F, A>) {
        self.data.sort_unstable_by_key(|(g, _, _)| H.galois_group_mulrepr().smallest_positive_lift(*g));
        self.data.dedup_by(|second, first| {
            if H.galois_group_mulrepr().eq_el(&second.0, &first.0) {
                H.ring().add_assign_ref(&mut first.1, &second.1);
                return true;
            } else {
                return false;
            }
        });
    }
}

impl<R, F, A> CCFFTRingBase<R, F, A> 
    where R: RingStore,
        R::Type: ZnRing,
        F: RingDecompositionSelfIso<R::Type> + CyclotomicRingDecomposition<R::Type>,
        A: Allocator + Clone,
        CCFFTRingBase<R, F, A>: CyclotomicRing + /* unfortunately, the type checker is not clever enough to know that this is always the case */ RingExtension<BaseRing = R>
{
    pub fn compute_linear_transform(&self, el: &<Self as RingBase>::Element, transform: &LinearTransform<R, F, A>) -> <Self as RingBase>::Element {
        <_ as RingBase>::sum(self, transform.data.iter().map(|(s, c, _)| self.mul_ref_fst(c, self.apply_galois_action(*s, self.clone_el(el)))))
    }
}

pub struct CompiledLinearTransform<R, F, A>
    where R: RingStore,
        R::Type: ZnRing,
        F: CyclotomicRingDecomposition<R::Type> + RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone,
        CCFFTRingBase<R, F, A>: CyclotomicRing + /* unfortunately, the type checker is not clever enough to know that this is always the case */ RingExtension<BaseRing = R>
{
    baby_step_galois_elements: Vec<ZnEl>,
    giant_step_galois_elements: Vec<Option<ZnEl>>,
    coeffs: Vec<Vec<El<CCFFTRing<R, F, A>>>>,
    one: El<CCFFTRing<R, F, A>>
}

impl<R, F, A> CompiledLinearTransform<R, F, A>
    where R: RingStore,
        R::Type: ZnRing,
        F: CyclotomicRingDecomposition<R::Type> + RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone,
        CCFFTRingBase<R, F, A>: CyclotomicRing + /* unfortunately, the type checker is not clever enough to know that this is always the case */ RingExtension<BaseRing = R>
{
    pub fn create_from_merged(H: &HypercubeIsomorphism<R, F, A>, lin_transforms: &[LinearTransform<R, F, A>], preferred_baby_steps: usize) -> CompiledLinearTransform<R, F, A> {
        Self::create_from(H, lin_transforms.iter().fold(LinearTransform::identity(&H), |current, next| next.compose(&current, &H)), preferred_baby_steps)
    }

    pub fn create_from(H: &HypercubeIsomorphism<R, F, A>, mut lin_transform: LinearTransform<R, F, A>, preferred_baby_steps: usize) -> CompiledLinearTransform<R, F, A> {
        lin_transform.optimize(&H);

        let mut max_step: Vec<i64> = Vec::new();
        let mut min_step: Vec<i64> = Vec::new();
        let mut gcd_step: Vec<i64> = Vec::new();
        let mut sizes: Vec<i64> = Vec::new();
        for i in 0..H.dim_count() {
            max_step.push(lin_transform.data.iter().map(|(_, _, steps)| steps[i]).max().unwrap());
            min_step.push(lin_transform.data.iter().map(|(_, _, steps)| steps[i]).min().unwrap());
            let gcd = lin_transform.data.iter().map(|(_, _, steps)| steps[i]).fold(0, |a, b| signed_gcd(a, b, StaticRing::<i64>::RING));
            if gcd == 0 {
                gcd_step.push(1);
            } else {
                gcd_step.push(gcd);
            }
            assert!(gcd_step[i] > 0);
            if gcd_step[i] != 0 {
                sizes.push(StaticRing::<i64>::RING.checked_div(&(max_step[i] - min_step[i] + gcd_step[i]), &gcd_step[i]).unwrap());
            } else {
                sizes.push(1);
            }
        }
        let mut baby_step_dims = 0;
        let mut baby_steps = 1;
        for i in 0..H.dim_count() {
            baby_step_dims += 1;
            baby_steps *= sizes[i];
            if baby_steps >= preferred_baby_steps as i64 {
                break;
            }
        }

        let giant_steps_galois_els = multi_cartesian_product((baby_step_dims..H.dim_count()).map(|i| (min_step[i]..=max_step[i]).step_by(gcd_step[i] as usize)), |indices| {
            H.galois_group_mulrepr().prod(indices.iter().enumerate().map(|(i, s)| H.shift_galois_element(i + baby_step_dims, *s)))
        }, |_, x| *x)
            .map(|g_el| if H.galois_group_mulrepr().is_one(&g_el) { None } else { Some(g_el) })
            .collect::<Vec<_>>();

        let baby_steps_galois_els = multi_cartesian_product((0..baby_step_dims).map(|i| (min_step[i]..=max_step[i]).step_by(gcd_step[i] as usize)), move |indices| {
            H.galois_group_mulrepr().prod(indices.iter().enumerate().map(|(i, s)| H.shift_galois_element(i, *s)))
        }, |_, x| *x).collect::<Vec<_>>();

        let compiled_coeffs = giant_steps_galois_els.iter().map(|gs_el| baby_steps_galois_els.iter().map(|bs_el| {
            let gs_el = gs_el.unwrap_or(H.galois_group_mulrepr().one());
            let total_el = H.galois_group_mulrepr().mul(gs_el, *bs_el);
            let coeff = &lin_transform.data.iter().filter(|(g, _, _)| H.galois_group_mulrepr().eq_el(g, &total_el)).next().unwrap().1;
            let result = H.ring().get_ring().apply_galois_action(H.galois_group_mulrepr().invert(&gs_el).unwrap(), H.ring().clone_el(coeff));
            return result;
        }).collect::<Vec<_>>()).collect::<Vec<_>>();

        return CompiledLinearTransform {
            baby_step_galois_elements: baby_steps_galois_els,
            giant_step_galois_elements: giant_steps_galois_els,
            coeffs: compiled_coeffs,
            one: H.ring().one()
        };
    }

    pub fn evaluate<S>(&self, input: &El<CCFFTRing<R, F, A>>, ring: S) -> El<CCFFTRing<R, F, A>>
        where S: RingStore<Type = CCFFTRingBase<R, F, A>>
    {
        self.evaluate_generic(input, |a, b, c| {
            ring.add_assign(a, ring.mul_ref(b, c));
        }, |x, galois_els| {
            galois_els.iter().map(|el| ring.get_ring().apply_galois_action(*el, ring.clone_el(x))).collect()
        }, || ring.zero())
    }
    
    pub fn evaluate_generic<T, AddScaled, ApplyGalois, Zero>(&self, input: &T, mut add_scaled_fn: AddScaled, mut apply_galois_fn: ApplyGalois, mut zero_fn: Zero) -> T
        where AddScaled: FnMut(&mut T, &T, &El<CCFFTRing<R, F, A>>),
            ApplyGalois: FnMut(&T, &[ZnEl]) -> Vec<T>,
            Zero: FnMut() -> T
    {
        let baby_steps = apply_galois_fn(input, &self.baby_step_galois_elements);
        assert_eq!(self.baby_step_galois_elements.len(), baby_steps.len());
        let mut result = zero_fn();
        for (gs_el, coeffs) in self.giant_step_galois_elements.iter().zip(self.coeffs.iter()) {
            let mut giant_step_result = zero_fn();
            for (coeff, x) in coeffs.iter().zip(baby_steps.iter()) {
                add_scaled_fn(&mut giant_step_result, x, coeff);
            }
            let summand = if let Some(gs_el) = gs_el {
                let summand = apply_galois_fn(&giant_step_result, &[*gs_el]);
                assert_eq!(summand.len(), 1);
                summand.into_iter().next().unwrap()
            } else {
                giant_step_result
            };
            add_scaled_fn(&mut result, &summand, &self.one);
        }
        return result;
    }
}

#[test]
fn test_compile() {
    let ring = DefaultPow2CyclotomicCCFFTRingBase::new(Zn::new(23), 5);
    let H = HypercubeIsomorphism::new(ring.get_ring());
    let compiled_transform = pow2_slots_to_coeffs_thin(&H).into_iter().map(|T| CompiledLinearTransform::create_from(&H, T, 2)).collect::<Vec<_>>();

    let mut current = H.from_slot_vec([1, 2, 3, 4].into_iter().map(|n| H.slot_ring().int_hom().map(n)));
    for T in &compiled_transform {
        current = T.evaluate(&current, &ring);
    }
    assert_el_eq!(&ring, &ring_literal!(&ring, [1, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), &current);
    
    let compiled_composed_transform = CompiledLinearTransform::create_from_merged(&H, &pow2_slots_to_coeffs_thin(&H), 2);
    let mut current = H.from_slot_vec([1, 2, 3, 4].into_iter().map(|n| H.slot_ring().int_hom().map(n)));
    current = compiled_composed_transform.evaluate(&current, &ring);
    assert_el_eq!(&ring, &ring_literal!(&ring, [1, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), &current);

    let ring = DefaultPow2CyclotomicCCFFTRingBase::new(Zn::new(97), 5);
    let H = HypercubeIsomorphism::new(ring.get_ring());
    let compiled_transform = pow2_slots_to_coeffs_thin(&H).into_iter().map(|T| CompiledLinearTransform::create_from(&H, T, 2)).collect::<Vec<_>>();
    
    let mut current = H.from_slot_vec((1..17).into_iter().map(|n| H.slot_ring().int_hom().map(n)));
    for T in compiled_transform {
        current = T.evaluate(&current, &ring);
    }
    assert_el_eq!(&ring, &ring_literal!(&ring, [1, 0, 5, 0, 3, 0, 7, 0, 2, 0, 6, 0, 4, 0, 8, 0, 9, 0, 13, 0, 11, 0, 15, 0, 10, 0, 14, 0, 12, 0, 16, 0]), &current);

    let compiled_composed_transform = CompiledLinearTransform::create_from_merged(&H, &pow2_slots_to_coeffs_thin(&H), 2);
    let mut current = H.from_slot_vec((1..17).into_iter().map(|n| H.slot_ring().int_hom().map(n)));
    current = compiled_composed_transform.evaluate(&current, &ring);
    assert_el_eq!(&ring, &ring_literal!(&ring, [1, 0, 5, 0, 3, 0, 7, 0, 2, 0, 6, 0, 4, 0, 8, 0, 9, 0, 13, 0, 11, 0, 15, 0, 10, 0, 14, 0, 12, 0, 16, 0]), &current);
}

#[test]
fn test_compose() {
    let ring = DefaultPow2CyclotomicCCFFTRingBase::new(Zn::new(23), 5);
    let H = HypercubeIsomorphism::new(ring.get_ring());
    let composed_transform = pow2_slots_to_coeffs_thin(&H).into_iter().fold(LinearTransform::identity(&H), |current, next| next.compose(&current, &H));

    let mut current = H.from_slot_vec([1, 2, 3, 4].into_iter().map(|n| H.slot_ring().int_hom().map(n)));
    current = ring.get_ring().compute_linear_transform(&current, &composed_transform);

    assert_el_eq!(&ring, &ring_literal!(&ring, [1, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), &current);
    
    let ring = DefaultPow2CyclotomicCCFFTRingBase::new(Zn::new(97), 5);
    let H = HypercubeIsomorphism::new(ring.get_ring());
    let composed_transform = pow2_slots_to_coeffs_thin(&H).into_iter().fold(LinearTransform::identity(&H), |current, next| next.compose(&current, &H));
    
    let mut current = H.from_slot_vec((1..17).into_iter().map(|n| H.slot_ring().int_hom().map(n)));
    current = ring.get_ring().compute_linear_transform(&current, &composed_transform);

    assert_el_eq!(&ring, &ring_literal!(&ring, [1, 0, 5, 0, 3, 0, 7, 0, 2, 0, 6, 0, 4, 0, 8, 0, 9, 0, 13, 0, 11, 0, 15, 0, 10, 0, 14, 0, 12, 0, 16, 0]), &current);
}