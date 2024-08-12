
use std::alloc::Allocator;
use std::alloc::Global;
use std::ops::Range;

use feanor_math::algorithms::eea::signed_gcd;
use feanor_math::assert_el_eq;
use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::iters::multi_cartesian_product;
use feanor_math::matrix::OwnedMatrix;
use feanor_math::primitive_int::*;
use feanor_math::ring::*;
use feanor_math::rings::extension::*;
use feanor_math::rings::zn::zn_64::*;
use feanor_math::rings::zn::*;
use feanor_math::seq::*;
use feanor_math::algorithms::linsolve::LinSolveRing;

use crate::complexfft::automorphism::*;
use crate::complexfft::complex_fft_ring::*;
use crate::cyclotomic::*;

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
    pub fn switch_ring(&self, H_from: &HypercubeIsomorphism<R, F, A>, to: &CCFFTRingBase<R, F, A>) -> Self {
        self.check_valid(H_from);
        assert_eq!(H_from.ring().n(), to.n());
        let from = H_from.ring();
        let red_map = ReductionMap::new(from.base_ring(), to.base_ring()).unwrap();
        let hom = |x: &El<CCFFTRing<R, F, A>>| to.from_canonical_basis(H_from.ring().wrt_canonical_basis(x).into_iter().map(|x| red_map.map(x)));
        Self {
            data: self.data.iter().map(|(g, coeff, steps)| (*g, hom(coeff), steps.clone())).collect()
        }
    }

    pub fn inverse(&self, H: &HypercubeIsomorphism<R, F, A>) -> Self {
        self.check_valid(H);
        let original_automorphisms = self.data.iter().map(|(g, _, _)| g);
        let inverse_automorphisms = original_automorphisms.clone().map(|g| H.galois_group_mulrepr().invert(g).unwrap()).collect::<Vec<_>>();
        let mut composed_automorphisms = original_automorphisms.clone().flat_map(|g| inverse_automorphisms.iter().map(|s| H.galois_group_mulrepr().mul_ref(g, s))).collect::<Vec<_>>();
        composed_automorphisms.sort_unstable_by_key(|g| H.galois_group_mulrepr().smallest_positive_lift(*g));
        composed_automorphisms.dedup_by(|a, b| H.galois_group_mulrepr().eq_el(a, b));

        let mut matrix: OwnedMatrix<_> = OwnedMatrix::zero(composed_automorphisms.len(), inverse_automorphisms.len(), H.ring());
        for (i, g) in original_automorphisms.enumerate() {
            for (j, s) in inverse_automorphisms.iter().enumerate() {
                let row_index = composed_automorphisms.binary_search_by_key(
                    &H.galois_group_mulrepr().smallest_positive_lift(H.galois_group_mulrepr().mul_ref(g, s)), 
                    |g| H.galois_group_mulrepr().smallest_positive_lift(*g)
                ).unwrap();
                let entry = H.ring().get_ring().apply_galois_action(*s, H.ring().clone_el(&self.data[i].1));
                *matrix.at_mut(row_index, j) = entry;
            }
        }

        let mut matrix_by_slots = (0..matrix.row_count()).map(|i| (0..matrix.col_count()).map(|j| H.get_slot_values(matrix.at(i, j))).collect::<Vec<_>>()).collect::<Vec<_>>();
        let mut result_by_slots = (0..inverse_automorphisms.len()).map(|_| Vec::new()).collect::<Vec<_>>();
        let mut lhs: OwnedMatrix<_> = OwnedMatrix::zero(matrix.row_count(), matrix.col_count(), H.slot_ring());
        let mut rhs: OwnedMatrix<_> = OwnedMatrix::zero(matrix.row_count(), 1, H.slot_ring());
        let mut sol: OwnedMatrix<_> = OwnedMatrix::zero(matrix.col_count(), 1, H.slot_ring());
        for _ in H.slot_iter(|_indices| ()) {
            for i in 0..matrix.row_count() {
                for j in 0..matrix.col_count() {
                    *lhs.at_mut(i, j) = matrix_by_slots[i][j].next().unwrap();
                }
            }
            assert!(H.galois_group_mulrepr().is_one(&composed_automorphisms[0]));
            *rhs.at_mut(0, 0) = H.slot_ring().one();
            for j in 1..matrix.row_count() {
                *rhs.at_mut(j, 0) = H.slot_ring().zero();
            }
            H.slot_ring().get_ring().solve_right(lhs.data_mut(), rhs.data_mut(), sol.data_mut(), Global).assert_solved();
            for j in 0..matrix.col_count() {
                result_by_slots[j].push(H.slot_ring().clone_el(sol.at(j, 0)));
            }
        }

        let result = result_by_slots.into_iter().map(|coeff_by_slots| H.from_slot_vec(coeff_by_slots.into_iter())).collect::<Vec<_>>();

        #[cfg(test)] {
            let sol = Submatrix::<AsFirstElement<_>, _>::new(&result, matrix.col_count(), 1);
            let mut check: OwnedMatrix<_> = OwnedMatrix::zero(matrix.row_count(), 1, H.ring());
            STANDARD_MATMUL.matmul(TransposableSubmatrix::from(matrix.data()), TransposableSubmatrix::from(sol), TransposableSubmatrixMut::from(check.data_mut()), H.ring().get_ring());
            assert!(H.ring().is_one(check.at(0, 0)));
            for i in 1..matrix.row_count() {
                assert!(H.ring().is_zero(check.at(i, 0)));
            }
        }

        let mut result = Self {
            data: self.data.iter().zip(result.into_iter()).map(|((g, _, steps), coeff)| (
                H.galois_group_mulrepr().invert(g).unwrap(),
                coeff,
                steps.into_iter().map(|k| -k).collect()
            )).collect()
        };
        result.optimize(H);

        #[cfg(test)] {
            let check = self.compose(&result, H);
            assert_eq!(1, check.data.len());
            assert!(H.galois_group_mulrepr().is_one(&check.data[0].0));
            assert!(H.ring().is_one(&check.data[0].1));
        }

        return result;
    }

    fn check_valid(&self, H: &HypercubeIsomorphism<R, F, A>) {
        for (i, (g, _, _)) in self.data.iter().enumerate() {
            for (j, (s, _, _)) in self.data.iter().enumerate() {
                assert!(i == j || !H.galois_group_mulrepr().eq_el(g, s));
            }
        }
        for (g, _, steps) in &self.data {
            assert_el_eq!(
                H.galois_group_mulrepr(),
                g,
                H.galois_group_mulrepr().prod(steps.iter().enumerate().map(|(i, s)| H.shift_galois_element(i, *s)))
            );
        }
    }

    fn compose(&self, run_first: &LinearTransform<R, F, A>, H: &HypercubeIsomorphism<R, F, A>) -> Self {
        self.check_valid(H);
        run_first.check_valid(H);
        let mut result = Self {
            data: self.data.iter().flat_map(|(self_g, self_coeff, self_indices)| run_first.data.iter().map(|(first_g, first_coeff, first_indices)| (
                H.galois_group_mulrepr().mul_ref(self_g, first_g), 
                H.ring().mul_ref_snd(H.ring().get_ring().apply_galois_action(*self_g, H.ring().clone_el(first_coeff)), self_coeff),
                self_indices.iter().zip(first_indices).map(|(self_i, first_i)| self_i + first_i).collect()
            ))).collect()
        };
        result.optimize(&H);
        return result;
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
        self.data.retain(|(_, coeff, _)| !H.ring().is_zero(coeff));
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
    coeffs: Vec<Vec<Option<El<CCFFTRing<R, F, A>>>>>,
    one: El<CCFFTRing<R, F, A>>
}

pub struct BabyStepGiantStepParams {
    pure_baby_step_dimensions: Range<usize>,
    pure_giant_step_dimensions: Range<usize>,
    mixed_step_dimension: usize,
    mixed_step_dimension_baby_steps: usize,
    hoisted_automorphism_count: usize,
    unhoisted_automorphism_count: usize
}

impl<R, F, A> CompiledLinearTransform<R, F, A>
    where R: RingStore,
        R::Type: ZnRing,
        F: CyclotomicRingDecomposition<R::Type> + RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone,
        CCFFTRingBase<R, F, A>: CyclotomicRing + /* unfortunately, the type checker is not clever enough to know that this is always the case */ RingExtension<BaseRing = R>
{
    fn compute_automorphisms_per_dimension(H: &HypercubeIsomorphism<R, F, A>, lin_transform: &LinearTransform<R, F, A>) -> (Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>) {
        lin_transform.check_valid(H);
        
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
        return (
            max_step,
            min_step,
            gcd_step,
            sizes
        );
    }

    pub fn baby_step_giant_step_params<V>(automorphisms_per_dim: V, preferred_baby_steps: usize) -> BabyStepGiantStepParams
        where V: VectorFn<usize>
    {
        let mut baby_step_dims = 0;
        let mut baby_steps = 1;
        for i in 0..automorphisms_per_dim.len() {
            let new_steps = baby_steps * automorphisms_per_dim.at(i);
            if new_steps >= preferred_baby_steps {
                break;
            }
            baby_step_dims += 1;
            baby_steps = new_steps;
        }
        let mixed_dim_i = baby_step_dims;
        let giant_step_start_dim = mixed_dim_i + 1;
        let mixed_dim_baby_steps = (preferred_baby_steps - 1) / baby_steps + 1;
        let baby_steps = baby_steps * mixed_dim_baby_steps;
        assert!(baby_steps >= preferred_baby_steps);
        let giant_steps = (giant_step_start_dim..automorphisms_per_dim.len()).map(|i| automorphisms_per_dim.at(i)).product::<usize>() * ((automorphisms_per_dim.at(mixed_dim_i) - 1) / mixed_dim_baby_steps + 1);
        return BabyStepGiantStepParams { 
            pure_baby_step_dimensions: 0..baby_step_dims, 
            pure_giant_step_dimensions: giant_step_start_dim..automorphisms_per_dim.len(), 
            mixed_step_dimension: mixed_dim_i, 
            mixed_step_dimension_baby_steps: mixed_dim_baby_steps, 
            // we assume both baby steps and giant steps contain the trivial automorphism once, thus subtract 1;
            // note that we cannot check this with the current information, but if it is wrong, at most the 
            // estimates will be slightly suboptimal
            hoisted_automorphism_count: baby_steps - 1, 
            unhoisted_automorphism_count: giant_steps - 1
        };
    }

    pub fn compile(H: &HypercubeIsomorphism<R, F, A>, lin_transform: LinearTransform<R, F, A>) -> Self {
        lin_transform.check_valid(H);

        let (_, _, _, sizes) = Self::compute_automorphisms_per_dimension(H, &lin_transform);

        const UNHOISTED_AUTO_COUNT_OVERHEAD: usize = 3;

        let preferred_baby_steps = (1..=(sizes.iter().copied().product::<i64>() as usize)).min_by_key(|preferred_baby_steps| {
            let params = Self::baby_step_giant_step_params(sizes.as_fn().map_fn(|s| *s as usize), *preferred_baby_steps);
            return params.hoisted_automorphism_count + params.unhoisted_automorphism_count * UNHOISTED_AUTO_COUNT_OVERHEAD;
        }).unwrap();

        return Self::create_from(H, lin_transform, preferred_baby_steps);
    }

    pub fn compile_merged(H: &HypercubeIsomorphism<R, F, A>, lin_transforms: &[LinearTransform<R, F, A>]) -> Self {
        Self::compile(H, lin_transforms.iter().fold(LinearTransform::identity(&H), |current, next| next.compose(&current, &H)))
    }

    pub fn create_from_merged(H: &HypercubeIsomorphism<R, F, A>, lin_transforms: &[LinearTransform<R, F, A>], preferred_baby_steps: usize) -> CompiledLinearTransform<R, F, A> {
        Self::create_from(H, lin_transforms.iter().fold(LinearTransform::identity(&H), |current, next| next.compose(&current, &H)), preferred_baby_steps)
    }

    pub fn create_from(H: &HypercubeIsomorphism<R, F, A>, lin_transform: LinearTransform<R, F, A>, preferred_baby_steps: usize) -> CompiledLinearTransform<R, F, A> {
        lin_transform.check_valid(H);

        let (max_step, min_step, gcd_step, sizes) = Self::compute_automorphisms_per_dimension(H, &lin_transform);

        let params = Self::baby_step_giant_step_params((0..H.dim_count()).map_fn(|i| sizes[i] as usize), preferred_baby_steps);
        let mixed_dim_i = params.mixed_step_dimension;
        let mixed_dim_baby_steps = params.mixed_step_dimension_baby_steps as i64;

        let giant_step_range_iters = [(min_step[mixed_dim_i]..=max_step[mixed_dim_i]).step_by((gcd_step[mixed_dim_i] * mixed_dim_baby_steps) as usize)].into_iter()
            .chain(params.pure_giant_step_dimensions.clone().map(|i| (min_step[i]..=max_step[i]).step_by(gcd_step[i] as usize)));

        let baby_step_range_iters = params.pure_baby_step_dimensions.clone().map(|i| (min_step[i]..=max_step[i]).step_by(gcd_step[i] as usize))
            .chain([(0..=((mixed_dim_baby_steps - 1) * gcd_step[mixed_dim_i])).step_by(gcd_step[mixed_dim_i] as usize)]);

        let giant_steps_galois_els = multi_cartesian_product(giant_step_range_iters, |indices| {
            H.galois_group_mulrepr().prod(indices.iter().enumerate().map(|(i, s)| H.shift_galois_element(if i == 0 { mixed_dim_i } else { i + params.pure_giant_step_dimensions.start - 1 }, *s)))
        }, |_, x| *x)
            .map(|g_el| if H.galois_group_mulrepr().is_one(&g_el) { None } else { Some(g_el) })
            .collect::<Vec<_>>();

        let baby_steps_galois_els = multi_cartesian_product(baby_step_range_iters, move |indices| {
            H.galois_group_mulrepr().prod(indices.iter().enumerate().map(|(i, s)| H.shift_galois_element(i, *s)))
        }, |_, x| *x).collect::<Vec<_>>();

        assert_eq!(params.hoisted_automorphism_count, baby_steps_galois_els.len() - 1);
        assert_eq!(params.unhoisted_automorphism_count, giant_steps_galois_els.len() - 1);

        let compiled_coeffs = giant_steps_galois_els.iter().map(|gs_el| baby_steps_galois_els.iter().map(|bs_el| {
            let gs_el = gs_el.unwrap_or(H.galois_group_mulrepr().one());
            let total_el = H.galois_group_mulrepr().mul(gs_el, *bs_el);
            let coeff = &lin_transform.data.iter().filter(|(g, _, _)| H.galois_group_mulrepr().eq_el(g, &total_el)).next().map(|(_, c, _)| c).and_then(|c| if H.ring().is_zero(c) { None } else { Some(c) });
            let result = coeff.map(|c| H.ring().get_ring().apply_galois_action(H.galois_group_mulrepr().invert(&gs_el).unwrap(), H.ring().clone_el(c)));
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
    
    pub fn required_galois_keys<'a>(&'a self) -> impl 'a + Iterator<Item = &'a ZnEl> {
        self.baby_step_galois_elements.iter().chain(self.giant_step_galois_elements.iter().filter_map(|x| x.as_ref()))
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
                if let Some(c) = coeff {
                    add_scaled_fn(&mut giant_step_result, x, c);
                }
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

use feanor_math::algorithms::matmul::MatmulAlgorithm;
use feanor_math::algorithms::matmul::STANDARD_MATMUL;
use feanor_math::matrix::AsFirstElement;
use feanor_math::matrix::Submatrix;
use feanor_math::matrix::TransposableSubmatrix;
use feanor_math::matrix::TransposableSubmatrixMut;
use feanor_math::homomorphism::Homomorphism;
use pow2::pow2_slots_to_coeffs_thin;
use crate::complexfft::pow2_cyclotomic::DefaultPow2CyclotomicCCFFTRingBase;

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
    
    let mut current = H.from_slot_vec((1..17).map(|n| H.slot_ring().int_hom().map(n)));
    for T in compiled_transform {
        current = T.evaluate(&current, &ring);
    }
    assert_el_eq!(&ring, &ring_literal!(&ring, [1, 0, 5, 0, 3, 0, 7, 0, 2, 0, 6, 0, 4, 0, 8, 0, 9, 0, 13, 0, 11, 0, 15, 0, 10, 0, 14, 0, 12, 0, 16, 0]), &current);

    let compiled_composed_transform = CompiledLinearTransform::create_from_merged(&H, &pow2_slots_to_coeffs_thin(&H), 2);
    let mut current = H.from_slot_vec((1..17).map(|n| H.slot_ring().int_hom().map(n)));
    current = compiled_composed_transform.evaluate(&current, &ring);
    assert_el_eq!(&ring, &ring_literal!(&ring, [1, 0, 5, 0, 3, 0, 7, 0, 2, 0, 6, 0, 4, 0, 8, 0, 9, 0, 13, 0, 11, 0, 15, 0, 10, 0, 14, 0, 12, 0, 16, 0]), &current);

    let compiled_composed_transform = CompiledLinearTransform::create_from_merged(&H, &pow2_slots_to_coeffs_thin(&H), 3);
    let mut current = H.from_slot_vec((1..17).map(|n| H.slot_ring().int_hom().map(n)));
    current = compiled_composed_transform.evaluate(&current, &ring);
    assert_el_eq!(&ring, &ring_literal!(&ring, [1, 0, 5, 0, 3, 0, 7, 0, 2, 0, 6, 0, 4, 0, 8, 0, 9, 0, 13, 0, 11, 0, 15, 0, 10, 0, 14, 0, 12, 0, 16, 0]), &current);

    let compiled_composed_transform = CompiledLinearTransform::create_from_merged(&H, &pow2_slots_to_coeffs_thin(&H), 5);
    let mut current = H.from_slot_vec((1..17).map(|n| H.slot_ring().int_hom().map(n)));
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
    
    let mut current = H.from_slot_vec((1..17).map(|n| H.slot_ring().int_hom().map(n)));
    current = ring.get_ring().compute_linear_transform(&current, &composed_transform);

    assert_el_eq!(&ring, &ring_literal!(&ring, [1, 0, 5, 0, 3, 0, 7, 0, 2, 0, 6, 0, 4, 0, 8, 0, 9, 0, 13, 0, 11, 0, 15, 0, 10, 0, 14, 0, 12, 0, 16, 0]), &current);
}

#[test]
fn test_invert() {
    let ring = DefaultPow2CyclotomicCCFFTRingBase::new(Zn::new(23), 5);
    let H = HypercubeIsomorphism::new(ring.get_ring());
    let composed_transform = pow2_slots_to_coeffs_thin(&H).into_iter().fold(LinearTransform::identity(&H), |current, next| next.compose(&current, &H));
    let inv_transform = composed_transform.inverse(&H);

    let current = ring_literal!(&ring, [1, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    let actual = ring.get_ring().compute_linear_transform(&current, &inv_transform);
    let expected = H.from_slot_vec([1, 2, 3, 4].into_iter().map(|n| H.slot_ring().int_hom().map(n)));
    assert_el_eq!(&ring, &expected, &actual);
    
    let ring = DefaultPow2CyclotomicCCFFTRingBase::new(Zn::new(97), 5);
    let H = HypercubeIsomorphism::new(ring.get_ring());
    let composed_transform = pow2_slots_to_coeffs_thin(&H).into_iter().fold(LinearTransform::identity(&H), |current, next| next.compose(&current, &H));
    let inv_transform = composed_transform.inverse(&H);
    
    let current = ring_literal!(&ring, [1, 0, 5, 0, 3, 0, 7, 0, 2, 0, 6, 0, 4, 0, 8, 0, 9, 0, 13, 0, 11, 0, 15, 0, 10, 0, 14, 0, 12, 0, 16, 0]);
    let actual = ring.get_ring().compute_linear_transform(&current, &inv_transform);
    let expected = H.from_slot_vec((1..17).map(|n| H.slot_ring().int_hom().map(n)));

    assert_el_eq!(&ring, &expected, &actual);
}