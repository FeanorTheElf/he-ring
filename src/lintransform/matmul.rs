
use std::alloc::Allocator;
use std::alloc::Global;
use std::fs::File;
use std::io::BufReader;
use std::io::BufWriter;
use std::ops::Range;

use feanor_math::algorithms::eea::signed_gcd;
use feanor_math::algorithms::matmul::ComputeInnerProduct;
use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::iters::multi_cartesian_product;
use feanor_math::matrix::OwnedMatrix;
use feanor_math::primitive_int::*;
use feanor_math::ring::*;
use feanor_math::rings::extension::*;
use feanor_math::rings::poly::dense_poly::DensePolyRing;
use feanor_math::rings::poly::PolyRingStore;
use feanor_math::rings::zn::zn_64::*;
use feanor_math::rings::zn::*;
use feanor_math::seq::*;
use feanor_math::algorithms::linsolve::LinSolveRing;
use feanor_math::homomorphism::*;
use serde::de::DeserializeSeed;

use crate::cyclotomic::*;
use crate::rings::number_ring::*;
use crate::rings::decomposition_ring::*;
use crate::rings::odd_cyclotomic::*;
use crate::rings::slots::*;
use crate::StdZn;

#[derive(Clone, PartialEq, Eq)]
pub(super) struct GaloisElementIndex {
    shift_steps: Vec<i64>,
    frobenius_count: i64
}

impl GaloisElementIndex {

    pub fn shift<V>(dim_count: usize, positions: V) -> Self
        where V: VectorFn<i64>
    {
        assert_eq!(dim_count, positions.len());
        GaloisElementIndex {
            shift_steps: positions.iter().collect(),
            frobenius_count: 0
        }
    }

    fn identity(dim_count: usize) -> GaloisElementIndex {
        GaloisElementIndex {
            shift_steps: (0..dim_count).map(|_| 0).collect(),
            frobenius_count: 0
        }
    }

    pub fn shift_1d(dim_count: usize, dim_index: usize, steps: i64) -> GaloisElementIndex {
        Self::shift(dim_count, (0..dim_count).map_fn(|i| if i == dim_index { steps } else { 0 }))
    }

    fn frobenius(dim_count: usize, count: i64) -> GaloisElementIndex {
        GaloisElementIndex {
            shift_steps: (0..dim_count).map(|_| 0).collect(),
            frobenius_count: count
        }
    }

    fn inverse(mut self) -> GaloisElementIndex {
        self.frobenius_count = -self.frobenius_count;
        for s in &mut self.shift_steps {
            *s = -*s;
        }
        return self;
    }

    fn compose(mut self, other: &GaloisElementIndex) -> GaloisElementIndex {
        assert!(self.shift_steps.len() == other.shift_steps.len());
        self.frobenius_count += other.frobenius_count;
        for i in 0..self.shift_steps.len() {
            self.shift_steps[i] += other.shift_steps[i];
        }
        return self;
    }

    fn galois_element<NumberRing, A>(&self, H: &HypercubeIsomorphism<NumberRing, A>) -> ZnEl
        where NumberRing: HECyclotomicNumberRing,
            A: Allocator + Clone
    {
        H.cyclotomic_index_ring().prod(self.shift_steps.iter().enumerate().map(|(i, s)| H.shift_galois_element(i, *s)).chain([H.frobenius_element(self.frobenius_count)].into_iter()))
    }

    ///
    /// If `dim_index_or_frobenius = 0`, returns the number of frobenius applications, otherwise
    /// returns the number of shifting steps along dimension `dim_index_or_frobenius - 1`.
    /// 
    fn index_at_including_frobenius(&self, dim_index_or_frobenius: usize) -> i64 {
        if dim_index_or_frobenius == 0 {
            self.frobenius_count
        } else {
            self.shift_steps[dim_index_or_frobenius - 1]
        }
    }

    fn canonicalize<NumberRing, A>(&mut self, H: &HypercubeIsomorphism<NumberRing, A>)
        where NumberRing: HECyclotomicNumberRing,
            A: Allocator + Clone
    {
        let canonicalize_mod = |a: i64, n: i64| (((a % n) + n) % n);
        let result = Self {
            frobenius_count: canonicalize_mod(self.frobenius_count, H.slot_ring().rank() as i64),
            shift_steps: self.shift_steps.iter().enumerate().map(|(i, s)| canonicalize_mod(*s, H.shift_order(i) as i64)).collect()
        };
        debug_assert!(H.cyclotomic_index_ring().eq_el(&self.galois_element(H), &result.galois_element(H)));
        *self = result;
    }
}

pub struct MatmulTransform<NumberRing, A = Global>
    where NumberRing: HECyclotomicNumberRing,
        A: Allocator + Clone
{
    pub(super) data: Vec<(GaloisElementIndex, El<DecompositionRing<NumberRing, Zn, A>>)>
}

impl<NumberRing, A> MatmulTransform<NumberRing, A>
    where NumberRing: HECyclotomicNumberRing + Clone,
        A: Allocator + Clone
{
    pub fn eq(&self, other: &Self, H: &HypercubeIsomorphism<NumberRing, A>) -> bool {
        self.check_valid(H);
        other.check_valid(H);
        if self.data.len() != other.data.len() {
            return false;
        }
        for (self_d, other_d) in self.data.iter().zip(other.data.iter()) {
            if !H.cyclotomic_index_ring().eq_el(&self_d.0.galois_element(H), &other_d.0.galois_element(H)) {
                return false;
            }
            if !H.ring().eq_el(&self_d.1, &other_d.1) {
                return false;
            }
        }
        return true;
    }

    ///
    /// `matrix` is called on `output_index, input_index, column_indices` to give the `(output_index, input_index)`-th
    /// entry of the matrix corresponding to the hypercolumn containing the slot `column_indices`.
    /// 
    pub fn matmul1d<'a, G>(H: &HypercubeIsomorphism<'a, NumberRing, A>, dim_index: usize, matrix: G) -> MatmulTransform<NumberRing, A>
        where G: Fn(usize, usize, &[usize]) -> El<SlotRing<'a, A>>
    {
        let m = H.len(dim_index) as i64;
        let mut result = MatmulTransform {
            data: ((1 - m)..m).map(|s| {
                let coeff = H.from_slot_vec(H.slot_iter(|idxs| if idxs[dim_index] as i64 >= s && idxs[dim_index] as i64 - s < m {
                    matrix(idxs[dim_index], (idxs[dim_index] as i64 - s) as usize, idxs)
                } else {
                    H.slot_ring().zero()
                }));
                return (
                    GaloisElementIndex::shift_1d(H.dim_count(), dim_index, s),
                    coeff, 
                );
            }).collect()
        };
        result.canonicalize(H);
        return result;
    }
    
    ///
    /// Applies a linea transform on each slot separately. The transform is given by its matrix w.r.t. the basis
    /// `1, X, ..., X^(d - 1)` where `X` is the canonical generator of the slot ring.
    /// 
    pub fn blockmatmul0d<'a, G>(H: &HypercubeIsomorphism<'a, NumberRing, A>, matrix: G) -> MatmulTransform<NumberRing, A>
        where G: Fn(usize, usize, &[usize]) -> El<Zn>
    {
        let d = H.slot_ring().rank();
        let Gal = H.cyclotomic_index_ring();
        let trace = Trace::new(H.ring().get_ring().number_ring().clone(), Gal.smallest_positive_lift(H.frobenius_element(1)), d);
        let extract_coeff_factors = (0..d).map(|j| trace.extract_coefficient_map(H.slot_ring(), j)).collect::<Vec<_>>();
        
        let poly_ring = DensePolyRing::new(H.slot_ring().base_ring(), "X");
        // this is the map `X -> X^p`, which is the frobenius in our case, since we choose the canonical generator of the slot ring as root of unity
        let apply_frobenius = |x: &El<SlotRing<'a, _>>, count: i64| {
            poly_ring.evaluate(&H.slot_ring().poly_repr(&poly_ring, x, &H.slot_ring().base_ring().identity()), &H.slot_ring().pow(H.slot_ring().canonical_gen(), Gal.smallest_positive_lift(H.frobenius_element(count)) as usize), &H.slot_ring().inclusion())
        };
        
        // similar to `blockmatmul1d()`, but simpler
        let mut result = MatmulTransform {
            data: (0..d).map(|frobenius_index| {
                let coeff = H.from_slot_vec(H.slot_iter(|idxs| {
                    <_ as ComputeInnerProduct>::inner_product(H.slot_ring().get_ring(), (0..d).map(|l| (
                        apply_frobenius(&extract_coeff_factors[l], frobenius_index as i64),
                        H.slot_ring().from_canonical_basis((0..d).map(|k| matrix(k, l, idxs)))
                    )))
                }));
                return (
                    GaloisElementIndex::frobenius(H.dim_count(), frobenius_index as i64),
                    coeff, 
                );
            }).collect()
        };
        result.canonicalize(H);
        return result;
    }

    ///
    /// For each hypercolumn along the `dim_index`-th dimension containing the slots of index 
    /// `U(i) = (u1, ..., u(dim_index - 1), i, u(dim_index + 1), ..., ur)` for all `i`, 
    /// we can consider the `Fp`-basis given by `X^k e_U(i)`. An `Fp`-linear transform that operates on 
    /// each set `{ X^k e_U(i) | k, k }` separately, for all `U`, is said to be of `blockmatmul1d`-type, 
    /// and can be created using this function.
    /// 
    /// More concretely, for each hypercolumn (represented by `U(.)`), the given function `matrix` 
    /// should return the `(i, k), (j, l)`-th entry of the matrix of the linear transform (w.r.t. basis `X^k e_U(i)`) 
    /// when called on `((i, k), (j, l), U(<unspecified value>))`.
    /// 
    pub fn blockmatmul1d<'a, G>(H: &HypercubeIsomorphism<'a, NumberRing, A>, dim_index: usize, matrix: G) -> MatmulTransform<NumberRing, A>
        where G: Fn((usize, usize), (usize, usize), &[usize]) -> El<Zn>
    {
        let m = H.len(dim_index) as i64;
        let d = H.slot_ring().rank();
        let Gal = H.cyclotomic_index_ring();
        let trace = Trace::new(H.ring().get_ring().number_ring().clone(), Gal.smallest_positive_lift(H.frobenius_element(1)), d);
        let extract_coeff_factors = (0..d).map(|j| trace.extract_coefficient_map(H.slot_ring(), j)).collect::<Vec<_>>();
        
        let poly_ring = DensePolyRing::new(H.slot_ring().base_ring(), "X");
        // this is the map `X -> X^p`, which is the frobenius in our case, since we choose the canonical generator of the slot ring as root of unity
        let apply_frobenius = |x: &El<SlotRing<'a, _>>, count: i64| {
            poly_ring.evaluate(&H.slot_ring().poly_repr(&poly_ring, x, &H.slot_ring().base_ring().identity()), &H.slot_ring().pow(H.slot_ring().canonical_gen(), Gal.smallest_positive_lift(H.frobenius_element(count)) as usize), &H.slot_ring().inclusion())
        };
        
        // the approach is as follows:
        // We consider the matrix by block-diagonals as in [`matmul1d()`], which correspond to shifting slots within a hypercolumn.
        // Additionally however, we need to take care of the transformation within a slot. Unfortunately, the matrix structure does
        // not nicely correspond to structure of the Frobenius anymore (more concretely, the basis `1, X, ..., X^(d - 1)` w.r.t. which
        // we represent the matrix is not normal). Thus, we have to solve a linear system, which is done by [`Trace::extract_coefficient_map`].
        // In other words, we compute the Frobenius-coefficients for the maps `sum a_k X^k -> a_l` for all `l`. Then we we take the
        // desired map as the linear combination of these extract-coefficient-maps.
        let mut result = MatmulTransform {
            data: ((1 - m)..m).flat_map(|s| (0..d).map(move |frobenius_index| (s, frobenius_index))).map(|(s, frobenius_index)| {
                let coeff = H.from_slot_vec(H.slot_iter(|idxs| if idxs[dim_index] as i64 >= s && idxs[dim_index] as i64 - s < m {
                    let i = idxs[dim_index];
                    let j = (idxs[dim_index] as i64 - s) as usize;
                    <_ as ComputeInnerProduct>::inner_product(H.slot_ring().get_ring(), (0..d).map(|l| (
                        apply_frobenius(&extract_coeff_factors[l], frobenius_index as i64),
                        H.slot_ring().from_canonical_basis((0..d).map(|k| matrix((i, k), (j, l), idxs)))
                    )))
                } else {
                    H.slot_ring().zero()
                }));
                return (
                    GaloisElementIndex::shift_1d(H.dim_count(), dim_index, s).compose(&GaloisElementIndex::frobenius(H.dim_count(), frobenius_index as i64)),
                    coeff, 
                );
            }).collect()
        };
        result.canonicalize(H);
        return result;
    }

    pub fn switch_ring(&self, H_from: &HypercubeIsomorphism<NumberRing, A>, to: &DecompositionRingBase<NumberRing, Zn, A>) -> Self {
        self.check_valid(H_from);
        assert_eq!(H_from.ring().n(), to.n());
        let from = H_from.ring();
        let red_map = ZnReductionMap::new(from.base_ring(), to.base_ring()).unwrap();
        let hom = |x: &El<DecompositionRing<NumberRing, Zn, A>>| to.from_canonical_basis(H_from.ring().wrt_canonical_basis(x).into_iter().map(|x| red_map.map(x)));
        Self {
            data: self.data.iter().map(|(g, coeff)| (g.clone(), hom(coeff))).collect()
        }
    }

    pub fn inverse(&self, H: &HypercubeIsomorphism<NumberRing, A>) -> Self {
        self.check_valid(H);
        let original_automorphisms = self.data.iter().map(|(g, _)| g.galois_element(H));
        let inverse_automorphisms = original_automorphisms.clone().map(|g| H.cyclotomic_index_ring().invert(&g).unwrap()).collect::<Vec<_>>();
        let mut composed_automorphisms = original_automorphisms.clone().flat_map(|g| inverse_automorphisms.iter().map(move |s| H.cyclotomic_index_ring().mul_ref(&g, s))).collect::<Vec<_>>();
        composed_automorphisms.sort_unstable_by_key(|g| H.cyclotomic_index_ring().smallest_positive_lift(*g));
        composed_automorphisms.dedup_by(|a, b| H.cyclotomic_index_ring().eq_el(a, b));

        let mut matrix: OwnedMatrix<_> = OwnedMatrix::zero(composed_automorphisms.len(), inverse_automorphisms.len(), H.ring());
        for (i, g) in original_automorphisms.enumerate() {
            for (j, s) in inverse_automorphisms.iter().enumerate() {
                let row_index = composed_automorphisms.binary_search_by_key(
                    &H.cyclotomic_index_ring().smallest_positive_lift(H.cyclotomic_index_ring().mul_ref(&g, s)), 
                    |g| H.cyclotomic_index_ring().smallest_positive_lift(*g)
                ).unwrap();
                let entry = H.ring().get_ring().apply_galois_action(&self.data[i].1, *s);
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
            assert!(H.cyclotomic_index_ring().is_one(&composed_automorphisms[0]));
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

        let mut result = Self {
            data: self.data.iter().zip(result.into_iter()).map(|((g, _), coeff)| (
                g.clone().inverse(),
                coeff
            )).collect()
        };
        result.canonicalize(H);

        #[cfg(test)] {
            let check = self.compose(&result, H);
            assert_eq!(1, check.data.len());
            assert!(H.cyclotomic_index_ring().is_one(&check.data[0].0.galois_element(H)));
            assert!(H.ring().is_one(&check.data[0].1));
        }

        return result;
    }

    fn check_valid(&self, _H: &HypercubeIsomorphism<NumberRing, A>) {
        for (i, (g, _)) in self.data.iter().enumerate() {
            for (j, (s, _)) in self.data.iter().enumerate() {
                assert!(i == j || g != s);
            }
        }
    }

    pub fn compose(&self, run_first: &MatmulTransform<NumberRing, A>, H: &HypercubeIsomorphism<NumberRing, A>) -> Self {
        self.check_valid(H);
        run_first.check_valid(H);
        let mut result = Self {
            data: self.data.iter().flat_map(|(self_g, self_coeff)| run_first.data.iter().map(|(first_g, first_coeff)| (
                self_g.clone().compose(first_g), 
                H.ring().mul_ref_snd(H.ring().get_ring().apply_galois_action(first_coeff, self_g.galois_element(H)), self_coeff)
            ))).collect()
        };
        result.canonicalize(&H);
        return result;
    }

    pub fn mult_scalar_slots<'a>(H: &HypercubeIsomorphism<'a, NumberRing, A>, scalar: &El<SlotRing<'a, A>>) -> MatmulTransform<NumberRing, A> {
        return MatmulTransform {
            data: vec![(GaloisElementIndex::identity(H.dim_count()), H.from_slot_vec((0..H.slot_count()).map(|_| H.slot_ring().clone_el(scalar))))]
        };
    }

    pub fn mult_ring_element<'a>(H: &HypercubeIsomorphism<'a, NumberRing, A>, factor: El<DecompositionRing<NumberRing, Zn, A>>) -> MatmulTransform<NumberRing, A> {
        return MatmulTransform {
            data: vec![(GaloisElementIndex::identity(H.dim_count()), factor)]
        };
    }

    pub fn identity(H: &HypercubeIsomorphism<NumberRing, A>) -> Self {
        Self {
            data: vec![(GaloisElementIndex::identity(H.dim_count()), H.ring().one())]
        }
    }

    pub fn shift(H: &HypercubeIsomorphism<NumberRing, A>, positions: &[i64]) -> Self {
        assert_eq!(H.dim_count(), positions.len());
        Self {
            data: vec![(GaloisElementIndex::shift(H.dim_count(), positions.copy_els()), H.ring().one())]
        }
    }

    ///
    /// Creates the linear transform that maps
    /// ```text
    ///   x  ->  sum_(i1, ..., ir, c) c σ_(i1, ..., ir)(x)
    /// ```
    /// where the sum is over all elements returned by the iterator and
    /// `σ_(i1, ..., ir)` is the Galois automorphism that corresponds to
    /// a shift by `ij` along the `j`-th hypercube dimension.
    /// 
    pub fn linear_combine_shifts<V, I>(H: &HypercubeIsomorphism<NumberRing, A>, summands: I) -> Self
        where I: Iterator<Item = (V, El<DecompositionRing<NumberRing, Zn, A>>)>,
            V: VectorFn<i64>
    {
        let mut result = Self {
            data: summands.map(|(positions, factor)| (GaloisElementIndex::shift(H.dim_count(), positions), factor)).collect()
        };
        result.canonicalize(H);
        return result;
    }

    fn canonicalize(&mut self, H: &HypercubeIsomorphism<NumberRing, A>) {
        self.data.sort_unstable_by_key(|(g, _)| H.cyclotomic_index_ring().smallest_positive_lift(g.galois_element(H)));
        for (steps, _) in &mut self.data {
            steps.canonicalize(H);
        }
        self.data.dedup_by(|second, first| {
            if H.cyclotomic_index_ring().eq_el(&second.0.galois_element(H), &first.0.galois_element(H)) {
                H.ring().add_assign_ref(&mut first.1, &second.1);
                return true;
            } else {
                return false;
            }
        });
        self.data.retain(|(_, coeff)| !H.ring().is_zero(coeff));
    }

    #[cfg(test)]
    #[allow(unused)]
    pub fn print(&self, H: &HypercubeIsomorphism<NumberRing, A>) {
        for (g, c) in &self.data {
            println!("p^{} {:?}: {}", g.frobenius_count, g.shift_steps, H.ring().format(c));
        }
    }
}

impl<NumberRing, A> DecompositionRingBase<NumberRing, Zn, A> 
    where NumberRing: HECyclotomicNumberRing,
        A: Allocator + Clone
{
    pub fn compute_linear_transform(&self, H: &HypercubeIsomorphism<NumberRing, A>, el: &<Self as RingBase>::Element, transform: &MatmulTransform<NumberRing, A>) -> <Self as RingBase>::Element {
        assert!(H.ring().get_ring() == self);
        <_ as RingBase>::sum(self, transform.data.iter().map(|(s, c)| self.mul_ref_fst(c, self.apply_galois_action(el, s.galois_element(H)))))
    }
}

pub struct CompiledLinearTransform<NumberRing, A = Global>
    where NumberRing: HECyclotomicNumberRing,
        A: Allocator + Clone
{
    baby_step_galois_elements: Vec<ZnEl>,
    giant_step_galois_elements: Vec<Option<ZnEl>>,
    coeffs: Vec<Vec<Option<El<DecompositionRing<NumberRing, Zn, A>>>>>,
    number_ring: NumberRing
}

#[derive(Debug)]
pub struct BabyStepGiantStepParams {
    pure_baby_step_dimensions: Range<usize>,
    pure_giant_step_dimensions: Range<usize>,
    mixed_step_dimension: usize,
    mixed_step_dimension_baby_steps: usize,
    hoisted_automorphism_count: usize,
    unhoisted_automorphism_count: usize
}

impl<NumberRing, A> HELinearTransform<NumberRing, A> for CompiledLinearTransform<NumberRing, A>
    where NumberRing: HECyclotomicNumberRing,
        A: Allocator + Clone
{
    fn number_ring(&self) -> &NumberRing {
        &self.number_ring
    }

    fn evaluate_generic<T, AddFn, ScaleFn, ApplyGaloisFn, CloneFn>(
            &self,
            input: T,
            mut add_fn: AddFn,
            mut scale_fn: ScaleFn,
            mut apply_galois_fn: ApplyGaloisFn,
            mut clone_fn: CloneFn
        ) -> T
            where AddFn: FnMut(T, &T) -> T,
                ScaleFn: FnMut(T, &El<DecompositionRing<NumberRing, Zn, A>>) -> T,
                ApplyGaloisFn: FnMut(T, &[ZnEl]) -> Vec<T>,
                CloneFn: FnMut(&T) -> T
    {
        let baby_steps = apply_galois_fn(input, &self.baby_step_galois_elements);

        assert_eq!(self.baby_step_galois_elements.len(), baby_steps.len());
        let mut result = None;
        for (gs_el, coeffs) in self.giant_step_galois_elements.iter().zip(self.coeffs.iter()) {
            let mut giant_step_result = None;
            for (coeff, x) in coeffs.iter().zip(baby_steps.iter()) {
                if let Some(c) = coeff {
                    if giant_step_result.is_some() {
                        giant_step_result = Some(add_fn(giant_step_result.unwrap(), &scale_fn(clone_fn(x), c)));
                    } else {
                        giant_step_result = Some(scale_fn(clone_fn(x), c));
                    }
                }
            }
            if let Some(giant_step_result) = giant_step_result {
                let summand = if let Some(gs_el) = gs_el {
                    let summand = apply_galois_fn(giant_step_result, &[*gs_el]);
                    assert_eq!(summand.len(), 1);
                    summand.into_iter().next().unwrap()
                } else {
                    giant_step_result
                };
                if result.is_some() {
                    result = Some(add_fn(result.unwrap(), &summand));
                } else {
                    result = Some(summand)
                };
            }
        }
        return result.unwrap();
    }
}

impl<NumberRing, A> CompiledLinearTransform<NumberRing, A>
    where NumberRing: HECyclotomicNumberRing + Clone,
        A: Allocator + Clone
{
    pub fn save(&self, filename: &str, ring: &DecompositionRing<NumberRing, Zn, A>, cyclotomic_index_ring: &Zn) {
        serde_json::to_writer_pretty(
            BufWriter::new(File::create(filename).unwrap()), 
            &serialization::CompiledLinearTransformSerializable::from(ring, cyclotomic_index_ring, self)
        ).unwrap()
    }

    pub fn load(filename: &str, ring: &DecompositionRing<NumberRing, Zn, A>, cyclotomic_index_ring: &Zn) -> Self {
        let mut deserializer = serde_json::Deserializer::from_reader(BufReader::new(File::open(filename).unwrap()));
        return <_ as DeserializeSeed>::deserialize(serialization::DeserializeLinearTransformSeed { cyclotomic_index_ring: cyclotomic_index_ring, ring: ring.get_ring() }, &mut deserializer).unwrap().into();
    }

    pub fn save_seq(data: &[Self], filename: &str, ring: &DecompositionRing<NumberRing, Zn, A>, cyclotomic_index_ring: &Zn) {
        serde_json::to_writer_pretty(
            BufWriter::new(File::create(filename).unwrap()), 
            &data.iter().map(|t| serialization::CompiledLinearTransformSerializable::from(ring, cyclotomic_index_ring, t)).collect::<Vec<_>>()
        ).unwrap()
    }

    pub fn load_seq(filename: &str, ring: &DecompositionRing<NumberRing, Zn, A>, cyclotomic_index_ring: &Zn) -> Vec<Self> {
        let mut deserializer = serde_json::Deserializer::from_reader(BufReader::new(File::open(filename).unwrap()));
        return <_ as DeserializeSeed>::deserialize(serialization::VecDeserializeSeed { base_seed: serialization::DeserializeLinearTransformSeed { cyclotomic_index_ring: cyclotomic_index_ring, ring: ring.get_ring() } }, &mut deserializer).unwrap().into();
    }

    /// 
    /// In the returned lists, we use the first entry for the "frobenius-dimension"
    /// 
    fn compute_automorphisms_per_dimension(H: &HypercubeIsomorphism<NumberRing, A>, lin_transform: &MatmulTransform<NumberRing, A>) -> (Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>) {
        lin_transform.check_valid(H);
        
        let mut max_step: Vec<i64> = Vec::new();
        let mut min_step: Vec<i64> = Vec::new();
        let mut gcd_step: Vec<i64> = Vec::new();
        let mut sizes: Vec<i64> = Vec::new();
        for i in 0..=H.dim_count() {
            max_step.push(lin_transform.data.iter().map(|(steps, _)| steps.index_at_including_frobenius(i)).max().unwrap());
            min_step.push(lin_transform.data.iter().map(|(steps, _)| steps.index_at_including_frobenius(i)).min().unwrap());
            let gcd = lin_transform.data.iter().map(|(steps, _)| steps.index_at_including_frobenius(i)).fold(0, |a, b| signed_gcd(a, b, StaticRing::<i64>::RING));
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

    pub fn compile(H: &HypercubeIsomorphism<NumberRing, A>, lin_transform: MatmulTransform<NumberRing, A>) -> Self {
        lin_transform.check_valid(H);

        let (_, _, _, sizes) = Self::compute_automorphisms_per_dimension(H, &lin_transform);

        const UNHOISTED_AUTO_COUNT_OVERHEAD: usize = 3;

        let preferred_baby_steps = (1..=(sizes.iter().copied().product::<i64>() as usize)).min_by_key(|preferred_baby_steps| {
            let params = Self::baby_step_giant_step_params(sizes.as_fn().map_fn(|s| *s as usize), *preferred_baby_steps);
            return params.hoisted_automorphism_count + params.unhoisted_automorphism_count * UNHOISTED_AUTO_COUNT_OVERHEAD;
        }).unwrap();

        return Self::create_from(H, lin_transform, preferred_baby_steps);
    }

    pub fn compile_merged(H: &HypercubeIsomorphism<NumberRing, A>, lin_transforms: &[MatmulTransform<NumberRing, A>]) -> Self {
        Self::compile(H, lin_transforms.iter().fold(MatmulTransform::identity(&H), |current, next| next.compose(&current, &H)))
    }

    pub fn create_from_merged(H: &HypercubeIsomorphism<NumberRing, A>, lin_transforms: &[MatmulTransform<NumberRing, A>], preferred_baby_steps: usize) -> CompiledLinearTransform<NumberRing, A> {
        Self::create_from(H, lin_transforms.iter().fold(MatmulTransform::identity(&H), |current, next| next.compose(&current, &H)), preferred_baby_steps)
    }

    pub fn create_from(H: &HypercubeIsomorphism<NumberRing, A>, lin_transform: MatmulTransform<NumberRing, A>, preferred_baby_steps: usize) -> CompiledLinearTransform<NumberRing, A> {
        lin_transform.check_valid(H);

        let (max_step, min_step, gcd_step, sizes) = Self::compute_automorphisms_per_dimension(H, &lin_transform);

        let params = Self::baby_step_giant_step_params((0..sizes.len()).map_fn(|i| sizes[i] as usize), preferred_baby_steps);

        let mixed_dim_i = params.mixed_step_dimension;
        let mixed_dim_baby_steps = params.mixed_step_dimension_baby_steps as i64;

        let giant_step_range_iters = [(min_step[mixed_dim_i]..=max_step[mixed_dim_i]).step_by((gcd_step[mixed_dim_i] * mixed_dim_baby_steps) as usize)].into_iter()
            .chain(params.pure_giant_step_dimensions.clone().map(|i| (min_step[i]..=max_step[i]).step_by(gcd_step[i] as usize)));

        let baby_step_range_iters = params.pure_baby_step_dimensions.clone().map(|i| (min_step[i]..=max_step[i]).step_by(gcd_step[i] as usize))
            .chain([(0..=((mixed_dim_baby_steps - 1) * gcd_step[mixed_dim_i])).step_by(gcd_step[mixed_dim_i] as usize)]);

        let shift_or_frobenius = |dim_or_frobenius: usize, steps: i64| if dim_or_frobenius == 0 {
            GaloisElementIndex::frobenius(H.dim_count(), steps)
        } else {
            GaloisElementIndex::shift_1d(H.dim_count(), dim_or_frobenius - 1, steps)
        };

        let giant_steps_galois_els = multi_cartesian_product(giant_step_range_iters, |indices| {
            indices[1..].iter().enumerate().map(|(i, s)| shift_or_frobenius(i + params.pure_giant_step_dimensions.start, *s)).fold(shift_or_frobenius(mixed_dim_i, indices[0]), |a, b| a.compose(&b)).galois_element(H)
        }, |_, x| *x)
            .map(|g_el| if H.cyclotomic_index_ring().is_one(&g_el) { None } else { Some(g_el) })
            .collect::<Vec<_>>();

        let baby_steps_galois_els = multi_cartesian_product(baby_step_range_iters, move |indices| {
            indices.iter().enumerate().map(|(i, s)| shift_or_frobenius(i, *s)).fold(shift_or_frobenius(0, 0), |a, b| a.compose(&b)).galois_element(H)
        }, |_, x| *x).collect::<Vec<_>>();

        assert_eq!(params.hoisted_automorphism_count, baby_steps_galois_els.len() - 1);
        assert_eq!(params.unhoisted_automorphism_count, giant_steps_galois_els.len() - 1);

        let mut lin_transform_data = lin_transform.data;
        let compiled_coeffs = giant_steps_galois_els.iter().map(|gs_el| baby_steps_galois_els.iter().map(|bs_el| {
            let gs_el = gs_el.unwrap_or(H.cyclotomic_index_ring().one());
            let total_el = H.cyclotomic_index_ring().mul(gs_el, *bs_el);
            let mut coeff = None;
            lin_transform_data.retain(|(g, c)| if H.cyclotomic_index_ring().eq_el(&g.galois_element(H), &total_el) {
                coeff = Some(H.ring().clone_el(c));
                false
            } else {
                true
            });
            coeff = coeff.and_then(|c| if H.ring().is_zero(&c) { None } else { Some(c) });
            let result = coeff.map(|c| H.ring().get_ring().apply_galois_action(&c, H.cyclotomic_index_ring().invert(&gs_el).unwrap()));
            return result;
        }).collect::<Vec<_>>()).collect::<Vec<_>>();

        return CompiledLinearTransform {
            baby_step_galois_elements: baby_steps_galois_els,
            giant_step_galois_elements: giant_steps_galois_els,
            coeffs: compiled_coeffs,
            number_ring: H.ring().get_ring().number_ring().clone()
        };
    }
}

mod serialization {
    use feanor_math::serialization::{DeserializeWithRing, SerializeWithRing};
    use serde::{de::{self, DeserializeSeed, Visitor}, ser::SerializeStruct, Deserialize, Serialize};
    use crate::rings::decomposition_ring::DecompositionRingBase;

    use super::*;

    pub struct CompiledLinearTransformSerializable<'a, NumberRing, A>
        where NumberRing: HECyclotomicNumberRing,
            A: Allocator + Clone
    {
        baby_step_galois_elements: Vec<SerializeWithRing<'a, &'a Zn>>,
        giant_step_galois_elements: Vec<Option<SerializeWithRing<'a, &'a Zn>>>,
        coeffs: Vec<Vec<Option<SerializeWithRing<'a, &'a DecompositionRing<NumberRing, Zn, A>>>>>,
    }

    impl<'a, NumberRing, A> Serialize for CompiledLinearTransformSerializable<'a, NumberRing, A>
        where NumberRing: HECyclotomicNumberRing,
            A: Allocator + Clone
    {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where S: serde::Serializer
        {
            let mut out = serializer.serialize_struct("CompiledLinearTransform", 7)?;
            out.serialize_field("baby_step_galois_elements", &self.baby_step_galois_elements)?;
            out.serialize_field("giant_step_galois_elements", &self.giant_step_galois_elements)?;
            out.serialize_field("coeffs", &self.coeffs)?;
            return out.end();
        }
    }

    impl<'a, NumberRing, A> CompiledLinearTransformSerializable<'a, NumberRing, A>
        where NumberRing: HECyclotomicNumberRing,
            A: Allocator + Clone
    {
        pub fn from(ring: &'a DecompositionRing<NumberRing, Zn, A>, galois_group_ring: &'a Zn, transform: &'a CompiledLinearTransform<NumberRing, A>) -> Self {
            Self {
                baby_step_galois_elements: transform.baby_step_galois_elements.iter().map(|x| SerializeWithRing::new(x, galois_group_ring)).collect(),
                giant_step_galois_elements: transform.giant_step_galois_elements.iter().map(|x| x.as_ref().map(|x| SerializeWithRing::new(x, galois_group_ring))).collect(),
                coeffs: transform.coeffs.iter().map(|cs| cs.iter().map(|c| c.as_ref().map(|c| SerializeWithRing::new(c, ring))).collect()).collect()
            }
        }
    }

    #[derive(Clone)]
    pub struct VecDeserializeSeed<S: Clone> {
        pub base_seed: S
    }

    impl<'de, S> DeserializeSeed<'de> for VecDeserializeSeed<S>
        where S: Clone + DeserializeSeed<'de>
    {
        type Value = Vec<S::Value>;

        fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
            where D: de::Deserializer<'de>
        {
            struct ElementsVisitor<S: Clone> {
                seed: S
            }
            impl<'de, S> Visitor<'de> for ElementsVisitor<S>
                where S: Clone + DeserializeSeed<'de>
            {
                type Value = Vec<S::Value>;

                fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                    write!(formatter, "sequence of value")
                }

                fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
                    where A: de::SeqAccess<'de>
                {
                    let mut result = Vec::new();
                    while let Some(el) = seq.next_element_seed(self.seed.clone())? {
                        result.push(el);
                    }
                    return Ok(result);
                }
            }
            deserializer.deserialize_seq(ElementsVisitor { seed: self.base_seed })
        }
    }
    
    #[derive(Clone)]
    pub struct OptionDeserializeSeed<S> {
        pub base_seed: S
    }

    impl<'de, S> DeserializeSeed<'de> for OptionDeserializeSeed<S>
        where S: DeserializeSeed<'de>
    {
        type Value = Option<S::Value>;

        fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
            where D: de::Deserializer<'de>
        {
            struct ElementVisitor<S> {
                seed: S
            }

            impl<'de, S> Visitor<'de> for ElementVisitor<S>
                where S: DeserializeSeed<'de>
            {
                type Value = Option<S::Value>;

                fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                    write!(formatter, "an optional value")
                }
                
                fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
                    where D: de::Deserializer<'de>
                {
                    Ok(Some(self.seed.deserialize(deserializer)?))                    
                }

                fn visit_none<E>(self) -> Result<Self::Value, E>
                    where E: de::Error
                {
                    Ok(None)
                }

                fn visit_unit<E>(self) -> Result<Self::Value, E>
                    where E: de::Error
                {
                    Ok(None)
                }
            }

            deserializer.deserialize_option(ElementVisitor {
                seed: self.base_seed
            })
        }
    }

    pub struct DeserializeLinearTransformSeed<'a, NumberRing, A>
        where NumberRing: HECyclotomicNumberRing,
            A: Allocator + Clone
    {
        pub cyclotomic_index_ring: &'a Zn,
        pub ring: &'a DecompositionRingBase<NumberRing, Zn, A>
    }
    
    impl<'a, NumberRing, A> Clone for DeserializeLinearTransformSeed<'a, NumberRing, A>
        where NumberRing: HECyclotomicNumberRing,
            A: Allocator + Clone
    {
        fn clone(&self) -> Self {
            *self
        }
    }

    impl<'a, NumberRing, A> Copy for DeserializeLinearTransformSeed<'a, NumberRing, A>
        where NumberRing: HECyclotomicNumberRing,
            A: Allocator + Clone
    {}

    impl<'a, 'de, NumberRing, A> DeserializeSeed<'de> for DeserializeLinearTransformSeed<'a, NumberRing, A>
        where NumberRing: HECyclotomicNumberRing + Clone,
            A: Allocator + Clone
    {
        type Value = CompiledLinearTransform<NumberRing, A>;

        fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
            where D: serde::Deserializer<'de>
        {
            struct FieldsVisitor<'a, NumberRing, A>
                where NumberRing: HECyclotomicNumberRing + Clone,
                    A: Allocator + Clone
            {
                cyclotomic_index_ring: &'a Zn,
                ring: &'a DecompositionRingBase<NumberRing, Zn, A>
            }
            
            impl<'a, 'de, NumberRing, A> Visitor<'de> for FieldsVisitor<'a, NumberRing, A>
                where NumberRing: HECyclotomicNumberRing + Clone,
                    A: Allocator + Clone
            {
                type Value = CompiledLinearTransform<NumberRing, A>;

                fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                    write!(formatter, "struct `CompiledLinearTransform` with fields `baby_step_galois_elements`, `giant_step_galois_elements`, `coeffs`")
                }

                fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
                    where S: serde::de::SeqAccess<'de>
                {
                    let baby_step_galois_elements = seq.next_element_seed(VecDeserializeSeed { base_seed: DeserializeWithRing::new(self.cyclotomic_index_ring) })?.ok_or_else(|| de::Error::invalid_length(0, &self))?;
                    let giant_step_galois_elements = seq.next_element_seed(VecDeserializeSeed { base_seed: OptionDeserializeSeed { base_seed: DeserializeWithRing::new(self.cyclotomic_index_ring) } })?.ok_or_else(|| de::Error::invalid_length(1, &self))?;
                    let coeffs = seq.next_element_seed(VecDeserializeSeed { base_seed: VecDeserializeSeed { base_seed: OptionDeserializeSeed { base_seed: DeserializeWithRing::new(RingRef::new(self.ring)) } } })?.ok_or_else(|| de::Error::invalid_length(1, &self))?;
                    return Ok(CompiledLinearTransform {
                        baby_step_galois_elements,
                        giant_step_galois_elements,
                        coeffs,
                        number_ring: self.ring.number_ring().clone()
                    });
                }
                
                fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
                    where M: de::MapAccess<'de>
                {
                    #[allow(non_camel_case_types)]
                    #[derive(Deserialize)]
                    enum Field {
                        baby_step_galois_elements,
                        giant_step_galois_elements,
                        coeffs
                    }
                    let mut baby_step_galois_elements = None;
                    let mut giant_step_galois_elements = None;
                    let mut coeffs = None;
                    while let Some(key) = map.next_key()? {
                        match key {
                            Field::baby_step_galois_elements => {
                                if baby_step_galois_elements.is_some() {
                                    return Err(de::Error::duplicate_field("baby_step_galois_elements"));
                                }
                                baby_step_galois_elements = Some(map.next_value_seed(VecDeserializeSeed { base_seed: DeserializeWithRing::new(self.cyclotomic_index_ring) })?);
                            },
                            Field::giant_step_galois_elements => {
                                if giant_step_galois_elements.is_some() {
                                    return Err(de::Error::duplicate_field("giant_step_galois_elements"));
                                }
                                giant_step_galois_elements = Some(map.next_value_seed(VecDeserializeSeed { base_seed: OptionDeserializeSeed { base_seed: DeserializeWithRing::new(self.cyclotomic_index_ring) } })?);
                            },
                            Field::coeffs => {
                                if coeffs.is_some() {
                                    return Err(de::Error::duplicate_field("coeffs"));
                                }
                                coeffs = Some(map.next_value_seed(VecDeserializeSeed { base_seed: VecDeserializeSeed { base_seed: OptionDeserializeSeed { base_seed: DeserializeWithRing::new(RingRef::new(self.ring)) } } })?);
                            },
                        }
                    }
                    let baby_step_galois_elements = baby_step_galois_elements.ok_or_else(|| de::Error::missing_field("baby_step_galois_elements"))?;
                    let giant_step_galois_elements = giant_step_galois_elements.ok_or_else(|| de::Error::missing_field("giant_step_galois_elements"))?;
                    let coeffs = coeffs.ok_or_else(|| de::Error::missing_field("coeffs"))?;
                    return Ok(CompiledLinearTransform {
                        baby_step_galois_elements,
                        giant_step_galois_elements,
                        coeffs,
                        number_ring: self.ring.number_ring().clone()
                    });
                }
            }

            deserializer.deserialize_struct("CompiledLinearTransform", &["baby_step_galois_elements", "giant_step_galois_elements", "coeffs"], FieldsVisitor {
                ring: self.ring,
                cyclotomic_index_ring: self.cyclotomic_index_ring
            })
        }
    }
}

#[cfg(test)]
use feanor_math::algorithms::matmul::MatmulAlgorithm;
#[cfg(test)]
use feanor_math::algorithms::matmul::STANDARD_MATMUL;
#[cfg(test)]
use feanor_math::matrix::Submatrix;
#[cfg(test)]
use feanor_math::matrix::TransposableSubmatrix;
#[cfg(test)]
use feanor_math::matrix::TransposableSubmatrixMut;
#[cfg(test)]
use super::pow2::slots_to_coeffs_thin;
#[cfg(test)]
use super::composite::slots_to_powcoeffs_thin;
use super::HELinearTransform;
#[cfg(test)]
use crate::rings::pow2_cyclotomic::*;
#[cfg(test)]
use feanor_math::assert_el_eq;

use super::trace::Trace;

#[test]
fn test_compile() {
    let ring = DecompositionRingBase::new(Pow2CyclotomicNumberRing::new(64), Zn::new(23));
    let H = HypercubeIsomorphism::new::<false>(ring.get_ring());
    let compiled_transform = slots_to_coeffs_thin(&H).into_iter().map(|T| CompiledLinearTransform::create_from(&H, T, 2)).collect::<Vec<_>>();

    let mut current = H.from_slot_vec([1, 2, 3, 4].into_iter().map(|n| H.slot_ring().int_hom().map(n)));
    let expected = slots_to_coeffs_thin(&H).into_iter().fold(ring.clone_el(&current), |c, T| ring.get_ring().compute_linear_transform(&H, &c, &T));
    for T in &compiled_transform {
        current = T.evaluate(&ring, current);
    }
    assert_el_eq!(&ring, &expected, &current);
    
    let compiled_composed_transform = CompiledLinearTransform::create_from_merged(&H, &slots_to_coeffs_thin(&H), 2);
    let mut current = H.from_slot_vec([1, 2, 3, 4].into_iter().map(|n| H.slot_ring().int_hom().map(n)));
    let expected = slots_to_coeffs_thin(&H).into_iter().fold(ring.clone_el(&current), |c, T| ring.get_ring().compute_linear_transform(&H, &c, &T));
    current = compiled_composed_transform.evaluate(&ring, current);
    assert_el_eq!(&ring, &expected, &current);

    let ring = DecompositionRingBase::new(Pow2CyclotomicNumberRing::new(64), Zn::new(97));
    let H = HypercubeIsomorphism::new::<false>(ring.get_ring());
    let compiled_transform = slots_to_coeffs_thin(&H).into_iter().map(|T| CompiledLinearTransform::create_from(&H, T, 2)).collect::<Vec<_>>();
    
    let mut current = H.from_slot_vec((1..17).map(|n| H.slot_ring().int_hom().map(n)));
    let expected = slots_to_coeffs_thin(&H).into_iter().fold(ring.clone_el(&current), |c, T| ring.get_ring().compute_linear_transform(&H, &c, &T));
    for T in compiled_transform {
        current = T.evaluate(&ring, current);
    }
    assert_el_eq!(&ring, &expected, &current);

    let compiled_composed_transform = CompiledLinearTransform::create_from_merged(&H, &slots_to_coeffs_thin(&H), 2);
    let mut current = H.from_slot_vec((1..17).map(|n| H.slot_ring().int_hom().map(n)));
    let expected = slots_to_coeffs_thin(&H).into_iter().fold(ring.clone_el(&current), |c, T| ring.get_ring().compute_linear_transform(&H, &c, &T));
    current = compiled_composed_transform.evaluate(&ring, current);
    assert_el_eq!(&ring, &expected, &current);

    let compiled_composed_transform = CompiledLinearTransform::create_from_merged(&H, &slots_to_coeffs_thin(&H), 3);
    let mut current = H.from_slot_vec((1..17).map(|n| H.slot_ring().int_hom().map(n)));
    let expected = slots_to_coeffs_thin(&H).into_iter().fold(ring.clone_el(&current), |c, T| ring.get_ring().compute_linear_transform(&H, &c, &T));
    current = compiled_composed_transform.evaluate(&ring, current);
    assert_el_eq!(&ring, &expected, &current);

    let compiled_composed_transform = CompiledLinearTransform::create_from_merged(&H, &slots_to_coeffs_thin(&H), 5);
    let mut current = H.from_slot_vec((1..17).map(|n| H.slot_ring().int_hom().map(n)));
    let expected = slots_to_coeffs_thin(&H).into_iter().fold(ring.clone_el(&current), |c, T| ring.get_ring().compute_linear_transform(&H, &c, &T));
    current = compiled_composed_transform.evaluate(&ring, current);
    assert_el_eq!(&ring, &expected, &current);
}

#[test]
fn test_compile_odd_case() {
    let ring = DecompositionRingBase::new(CompositeCyclotomicNumberRing::new(3, 19), Zn::new(7));
    let H = HypercubeIsomorphism::new::<false>(ring.get_ring());

    let mut current = H.from_slot_vec((1..=12).map(|n| H.slot_ring().int_hom().map(n)));
    let compiled_transform = slots_to_powcoeffs_thin(&H).into_iter().map(|T| CompiledLinearTransform::compile(&H, T)).collect::<Vec<_>>();
    let mut expected = ring.clone_el(&current);
    for (compiled_transform, transform) in compiled_transform.iter().zip(slots_to_powcoeffs_thin(&H).into_iter()) {
        current = compiled_transform.evaluate(&ring, current);
        expected = ring.get_ring().compute_linear_transform(&H, &expected, &transform);
        assert_el_eq!(&ring, &expected, &current);
    }
    println!();
    
    let ring = DecompositionRingBase::new(CompositeCyclotomicNumberRing::new(11, 31), Zn::new(2));
    let H = HypercubeIsomorphism::new::<false>(ring.get_ring());

    let mut current = H.from_slot_vec((1..=30).map(|n| H.slot_ring().int_hom().map(n)));
    let compiled_transform = slots_to_powcoeffs_thin(&H).into_iter().map(|T| CompiledLinearTransform::compile(&H, T)).collect::<Vec<_>>();
    let mut expected = ring.clone_el(&current);
    for (compiled_transform, transform) in compiled_transform.iter().zip(slots_to_powcoeffs_thin(&H).into_iter()) {
        current = compiled_transform.evaluate(&ring, current);
        expected = ring.get_ring().compute_linear_transform(&H, &expected, &transform);
        assert_el_eq!(&ring, &expected, &current);
    }
}

#[test]
fn test_compose() {
    let ring = DecompositionRingBase::new(Pow2CyclotomicNumberRing::new(64), Zn::new(23));
    let H = HypercubeIsomorphism::new::<false>(ring.get_ring());
    let composed_transform = slots_to_coeffs_thin(&H).into_iter().fold(MatmulTransform::identity(&H), |current, next| next.compose(&current, &H));

    let mut current = H.from_slot_vec([1, 2, 3, 4].into_iter().map(|n| H.slot_ring().int_hom().map(n)));
    let mut expected = ring.clone_el(&current);
    current = ring.get_ring().compute_linear_transform(&H, &current, &composed_transform);
    expected = slots_to_coeffs_thin(&H).into_iter().fold(expected, |c, T| ring.get_ring().compute_linear_transform(&H, &c, &T));

    assert_el_eq!(&ring, &expected, &current);
    
    let ring = DecompositionRingBase::new(Pow2CyclotomicNumberRing::new(64), Zn::new(97));
    let H = HypercubeIsomorphism::new::<false>(ring.get_ring());
    let composed_transform = slots_to_coeffs_thin(&H).into_iter().fold(MatmulTransform::identity(&H), |current, next| next.compose(&current, &H));
    
    let mut current = H.from_slot_vec((1..17).map(|n| H.slot_ring().int_hom().map(n)));
    let mut expected = ring.clone_el(&current);
    current = ring.get_ring().compute_linear_transform(&H, &current, &composed_transform);
    expected = slots_to_coeffs_thin(&H).into_iter().fold(expected, |c, T| ring.get_ring().compute_linear_transform(&H, &c, &T));

    assert_el_eq!(&ring, &expected, &current);
}

#[test]
fn test_invert() {
    let ring = DecompositionRingBase::new(Pow2CyclotomicNumberRing::new(64), Zn::new(23));
    let H = HypercubeIsomorphism::new::<false>(ring.get_ring());
    let composed_transform = slots_to_coeffs_thin(&H).into_iter().fold(MatmulTransform::identity(&H), |current, next| next.compose(&current, &H));
    let inv_transform = composed_transform.inverse(&H);

    let expected = H.from_slot_vec([1, 2, 3, 4].into_iter().map(|n| H.slot_ring().int_hom().map(n)));
    let current = slots_to_coeffs_thin(&H).into_iter().fold(ring.clone_el(&expected), |c, T| ring.get_ring().compute_linear_transform(&H, &c, &T));
    let actual = ring.get_ring().compute_linear_transform(&H, &current, &inv_transform);
    assert_el_eq!(&ring, &expected, &actual);
    
    let ring = DecompositionRingBase::new(Pow2CyclotomicNumberRing::new(64), Zn::new(97));
    let H = HypercubeIsomorphism::new::<false>(ring.get_ring());
    let composed_transform = slots_to_coeffs_thin(&H).into_iter().fold(MatmulTransform::identity(&H), |current, next| next.compose(&current, &H));
    let inv_transform = composed_transform.inverse(&H);
    
    let expected = H.from_slot_vec((1..17).map(|n| H.slot_ring().int_hom().map(n)));
    let current = slots_to_coeffs_thin(&H).into_iter().fold(ring.clone_el(&expected), |c, T| ring.get_ring().compute_linear_transform(&H, &c, &T));
    let actual = ring.get_ring().compute_linear_transform(&H, &current, &inv_transform);

    assert_el_eq!(&ring, &expected, &actual);
}

#[test]
fn test_blockmatmul1d() {
    // F23[X]/(Phi_5) ~ F_(23^4)
    let ring = DecompositionRingBase::new(OddCyclotomicNumberRing::new(5), Zn::new(23));
    let H = HypercubeIsomorphism::new::<false>(ring.get_ring());
    let matrix = [
        [1, 0, 1, 0],
        [0, 0, 0, 2],
        [0, 0, 0, 0],
        [5, 0, 8, 8]
    ];
    let lin_transform = MatmulTransform::blockmatmul1d(&H, 0, |(i, k), (j, l), idxs| {
        assert_eq!(0, i);
        assert_eq!(0, j);
        assert_eq!(&[0], idxs);
        H.slot_ring().base_ring().int_hom().map(matrix[k][l])
    });

    for i in 0..4 {
        let input = H.ring().pow(H.ring().canonical_gen(), i);
        let expected = H.ring().from_canonical_basis((0..4).map(|j| H.ring().base_ring().int_hom().map(matrix[j][i])));
        let actual = ring.get_ring().compute_linear_transform(&H, &input, &lin_transform);
        assert_el_eq!(H.ring(), &expected, &actual);
    }


    // F23[X]/(Phi_7) ~ F_(23^3)^2
    let ring = DecompositionRingBase::new(OddCyclotomicNumberRing::new(7), Zn::new(23));
    let H = HypercubeIsomorphism::new::<false>(ring.get_ring());
    let matrix = [
        [1, 0, 0],
        [2, 0, 0],
        [3, 0, 0]
    ];
    let lin_transform = MatmulTransform::blockmatmul1d(&H, 0, |(i, k), (j, l), _idxs| {
        if i == 0 && j == 1 {
            H.slot_ring().base_ring().int_hom().map(matrix[k][l])
        } else {
            H.slot_ring().base_ring().zero()
        }
    });

    for i in 0..3 {
        let input = H.from_slot_vec([H.slot_ring().zero(), H.slot_ring().pow(H.slot_ring().canonical_gen(), i)]);
        let expected = H.from_slot_vec([H.slot_ring().from_canonical_basis((0..3).map(|j| H.slot_ring().base_ring().int_hom().map(matrix[j][i]))), H.slot_ring().zero()]);
        let actual = ring.get_ring().compute_linear_transform(&H, &input, &lin_transform);
        assert_el_eq!(H.ring(), &expected, &actual);
    }

    for i in 0..3 {
        let input = H.from_slot_vec([H.slot_ring().pow(H.slot_ring().canonical_gen(), i), H.slot_ring().zero()]);
        let expected = H.ring().zero();
        let actual = ring.get_ring().compute_linear_transform(&H, &input, &lin_transform);
        assert_el_eq!(H.ring(), &expected, &actual);
    }
}