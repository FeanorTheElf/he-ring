
use std::alloc::Allocator;
use std::alloc::Global;
use std::fs::File;
use std::io::BufReader;
use std::io::BufWriter;
use std::ops::Range;
use std::cmp::min;

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
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::circuit::Coefficient;
use crate::circuit::PlaintextCircuit;
use crate::cyclotomic::*;
use crate::lintransform::PowerTable;
use crate::lintransform::CREATE_LINEAR_TRANSFORM_TIME_RECORDER;
use crate::rings::hypercube::*;
use crate::rings::number_ring::*;
use crate::rings::decomposition_ring::*;
use crate::rings::odd_cyclotomic::*;
use crate::lintransform::trace::*;
use crate::lintransform::HELinearTransform;
use crate::StdZn;

pub struct MatmulTransform<NumberRing, A = Global>
    where NumberRing: HECyclotomicNumberRing,
        A: Allocator + Clone
{
    pub(super) data: Vec<(CyclotomicGaloisGroupEl, El<DecompositionRing<NumberRing, Zn, A>>)>
}

impl<NumberRing, A> MatmulTransform<NumberRing, A>
    where NumberRing: HECyclotomicNumberRing + Clone,
        A: Allocator + Clone
{
    pub fn eq(&self, other: &Self, H: &DefaultHypercube<NumberRing, A>) -> bool {
        self.check_valid(H);
        other.check_valid(H);
        if self.data.len() != other.data.len() {
            return false;
        }
        for (self_d, other_d) in self.data.iter().zip(other.data.iter()) {
            if !H.galois_group().eq_el(self_d.0, other_d.0) {
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
    pub fn matmul1d<G>(H: &DefaultHypercube<NumberRing, A>, dim_index: usize, matrix: G) -> MatmulTransform<NumberRing, A>
        where G: Sync + Fn(usize, usize, &[usize]) -> El<SlotRingOver<Zn>>,
            A: Send + Sync
    {
        record_time!(CREATE_LINEAR_TRANSFORM_TIME_RECORDER, "matmul1d", || {
            let m = H.hypercube().m(dim_index) as i64;
            let mut result = MatmulTransform {
                data: ((1 - m)..m).into_par_iter().map(|s| {
                    let coeff = H.from_slot_values(H.hypercube().hypercube_iter(|idxs: &[usize]| if idxs[dim_index] as i64 >= s && idxs[dim_index] as i64 - s < m {
                        matrix(idxs[dim_index], (idxs[dim_index] as i64 - s) as usize, idxs)
                    } else {
                        H.slot_ring().zero()
                    }));
                    return (
                        H.hypercube().map_1d(dim_index, s),
                        coeff, 
                    );
                }).collect()
            };
            result.canonicalize(H);
            return result;
        })
    }
    
    ///
    /// Applies a linea transform on each slot separately. The transform is given by its matrix w.r.t. the basis
    /// `1, X, ..., X^(d - 1)` where `X` is the canonical generator of the slot ring.
    /// 
    pub fn blockmatmul0d<G>(H: &DefaultHypercube<NumberRing, A>, matrix: G) -> MatmulTransform<NumberRing, A>
        where G: Fn(usize, usize, &[usize]) -> El<Zn>
    {
        let d = H.slot_ring().rank();
        let Gal = H.galois_group();
        let trace = Trace::new(H.ring().get_ring().number_ring().clone(), Gal.representative(H.hypercube().p()) as i64, d);
        let extract_coeff_factors = (0..d).map(|j| trace.extract_coefficient_map(H.slot_ring(), j)).collect::<Vec<_>>();
        
        let poly_ring = DensePolyRing::new(H.slot_ring().base_ring(), "X");
        // this is the map `X -> X^p`, which is the frobenius in our case, since we choose the canonical generator of the slot ring as root of unity
        let apply_frobenius = |x: &El<SlotRingOver<Zn>>, count: i64| poly_ring.evaluate(
            &H.slot_ring().poly_repr(&poly_ring, x, &H.slot_ring().base_ring().identity()), 
            &H.slot_ring().pow(H.slot_ring().canonical_gen(), Gal.representative(H.hypercube().frobenius(count))), 
            &H.slot_ring().inclusion()
        );
        
        // similar to `blockmatmul1d()`, but simpler
        let mut result = MatmulTransform {
            data: (0..d).map(|frobenius_index| {
                let coeff = H.from_slot_values(H.hypercube().hypercube_iter(|idxs| {
                    <_ as ComputeInnerProduct>::inner_product(H.slot_ring().get_ring(), (0..d).map(|l| (
                        apply_frobenius(&extract_coeff_factors[l], frobenius_index as i64),
                        H.slot_ring().from_canonical_basis((0..d).map(|k| matrix(k, l, idxs)))
                    )))
                }));
                return (
                    H.hypercube().frobenius(frobenius_index as i64),
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
    pub fn blockmatmul1d<'a, G>(H: &DefaultHypercube<NumberRing, A>, dim_index: usize, matrix: G) -> MatmulTransform<NumberRing, A>
        where G: Sync + Fn((usize, usize), (usize, usize), &[usize]) -> El<Zn>,
            A: Send + Sync
    {
        record_time!(CREATE_LINEAR_TRANSFORM_TIME_RECORDER, "blockmatmul1d", || {
            let m = H.hypercube().m(dim_index) as i64;
            let d = H.slot_ring().rank();
            let Gal = H.galois_group();
            let trace = Trace::new(H.ring().get_ring().number_ring().clone(), Gal.representative(H.hypercube().p()) as i64, d);
            let extract_coeff_factors = (0..d).map(|j| trace.extract_coefficient_map(H.slot_ring(), j)).collect::<Vec<_>>();
            
            let poly_ring = DensePolyRing::new(H.slot_ring().base_ring(), "X");
            let canonical_gen_powertable = PowerTable::new(H.slot_ring(), H.slot_ring().canonical_gen(), H.ring().n() as usize);
            // this is the map `X -> X^p`, which is the frobenius in our case, since we choose the canonical generator of the slot ring as root of unity
            let apply_frobenius = |x: &El<SlotRingOver<Zn>>, count: i64| H.slot_ring().sum(
                H.slot_ring().wrt_canonical_basis(x).iter().enumerate().map(|(i, c)| H.slot_ring().inclusion().mul_ref_map(
                    &*canonical_gen_powertable.get_power(i as i64 * Gal.representative(H.hypercube().frobenius(count)) as i64),
                    &c
                )
            ));
            
            // the approach is as follows:
            // We consider the matrix by block-diagonals as in [`matmul1d()`], which correspond to shifting slots within a hypercolumn.
            // Additionally however, we need to take care of the transformation within a slot. Unfortunately, the matrix structure does
            // not nicely correspond to structure of the Frobenius anymore (more concretely, the basis `1, X, ..., X^(d - 1)` w.r.t. which
            // we represent the matrix is not normal). Thus, we have to solve a linear system, which is done by [`Trace::extract_coefficient_map`].
            // In other words, we compute the Frobenius-coefficients for the maps `sum a_k X^k -> a_l` for all `l`. Then we we take the
            // desired map as the linear combination of these extract-coefficient-maps.
            let mut result = MatmulTransform {
                data: ((1 - m)..m).into_par_iter().flat_map_iter(|s| (0..d).map(move |frobenius_index| (s, frobenius_index))).map(|(s, frobenius_index)| {
                    let coeff = H.from_slot_values(H.hypercube().hypercube_iter(|idxs| if idxs[dim_index] as i64 >= s && idxs[dim_index] as i64 - s < m {
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
                        H.galois_group().mul(H.hypercube().map_1d(dim_index, s), H.hypercube().frobenius(frobenius_index as i64)),
                        coeff, 
                    );
                }).collect()
            };
            result.canonicalize(H);
            return result;
        })
    }

    pub fn switch_ring(&self, H_from: &DefaultHypercube<NumberRing, A>, to: &DecompositionRingBase<NumberRing, Zn, A>) -> Self {
        self.check_valid(H_from);
        assert_eq!(H_from.ring().n(), to.n());
        let from = H_from.ring();
        let red_map = ZnReductionMap::new(from.base_ring(), to.base_ring()).unwrap();
        let hom = |x: &El<DecompositionRing<NumberRing, Zn, A>>| to.from_canonical_basis(H_from.ring().wrt_canonical_basis(x).into_iter().map(|x| red_map.map(x)));
        Self {
            data: self.data.iter().map(|(g, coeff)| (g.clone(), hom(coeff))).collect()
        }
    }

    pub fn inverse(&self, H: &DefaultHypercube<NumberRing, A>) -> Self {
        self.check_valid(H);
        let Gal = H.galois_group();
        let original_automorphisms = self.data.iter().map(|(g, _)| *g);
        let inverse_automorphisms = original_automorphisms.clone().map(|g| Gal.invert(g)).collect::<Vec<_>>();
        let mut composed_automorphisms = original_automorphisms.clone().flat_map(|g| inverse_automorphisms.iter().map(move |s| Gal.mul(g, *s))).collect::<Vec<_>>();
        composed_automorphisms.sort_unstable_by_key(|g| Gal.representative(*g));
        composed_automorphisms.dedup_by(|a, b| Gal.eq_el(*a, *b));

        let mut matrix: OwnedMatrix<_> = OwnedMatrix::zero(composed_automorphisms.len(), inverse_automorphisms.len(), H.ring());
        for (i, g) in original_automorphisms.enumerate() {
            for (j, s) in inverse_automorphisms.iter().enumerate() {
                let row_index = composed_automorphisms.binary_search_by_key(
                    &Gal.representative(Gal.mul(g, *s)), 
                    |g| Gal.representative(*g)
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

        for _ in H.hypercube().element_iter() {
            for i in 0..matrix.row_count() {
                for j in 0..matrix.col_count() {
                    *lhs.at_mut(i, j) = matrix_by_slots[i][j].next().unwrap();
                }
            }
            assert!(H.galois_group().is_identity(composed_automorphisms[0]));
            *rhs.at_mut(0, 0) = H.slot_ring().one();
            for j in 1..matrix.row_count() {
                *rhs.at_mut(j, 0) = H.slot_ring().zero();
            }
            H.slot_ring().get_ring().solve_right(lhs.data_mut(), rhs.data_mut(), sol.data_mut(), Global).assert_solved();
            for j in 0..matrix.col_count() {
                result_by_slots[j].push(H.slot_ring().clone_el(sol.at(j, 0)));
            }
        }

        let result = result_by_slots.into_iter().map(|coeff_by_slots| H.from_slot_values(coeff_by_slots.into_iter())).collect::<Vec<_>>();

        let mut result = Self {
            data: self.data.iter().zip(result.into_iter()).map(|((g, _), coeff)| (
                Gal.invert(*g),
                coeff
            )).collect()
        };
        result.canonicalize(H);

        #[cfg(test)] {
            let check = self.compose(&result, H);
            assert_eq!(1, check.data.len());
            assert!(Gal.is_identity(check.data[0].0));
            assert!(H.ring().is_one(&check.data[0].1));
        }

        return result;
    }

    fn check_valid(&self, H: &DefaultHypercube<NumberRing, A>) {
        let Gal = H.galois_group();
        assert!(self.data.is_sorted_by_key(|(g, _)| Gal.representative(*g)));
        for (i, (g, _)) in self.data.iter().enumerate() {
            for (j, (s, _)) in self.data.iter().enumerate() {
                assert!(i == j || !Gal.eq_el(*g, *s));
            }
        }
    }

    pub fn compose(&self, run_first: &MatmulTransform<NumberRing, A>, H: &DefaultHypercube<NumberRing, A>) -> Self {
        self.check_valid(H);
        run_first.check_valid(H);
        let Gal = H.galois_group();
        let mut result = Self {
            data: self.data.iter().flat_map(|(self_g, self_coeff)| run_first.data.iter().map(|(first_g, first_coeff)| (
                Gal.mul(*self_g, *first_g), 
                H.ring().mul_ref_snd(H.ring().get_ring().apply_galois_action(first_coeff, *self_g), self_coeff)
            ))).collect()
        };
        result.canonicalize(&H);
        return result;
    }

    pub fn mult_scalar_slots(H: &DefaultHypercube<NumberRing, A>, scalar: &El<SlotRingOver<Zn>>) -> MatmulTransform<NumberRing, A> {
        return MatmulTransform {
            data: vec![(H.galois_group().identity(), H.from_slot_values((0..H.slot_count()).map(|_| H.slot_ring().clone_el(scalar))))]
        };
    }

    pub fn mult_ring_element(H: &DefaultHypercube<NumberRing, A>, factor: El<DecompositionRing<NumberRing, Zn, A>>) -> MatmulTransform<NumberRing, A> {
        return MatmulTransform {
            data: vec![(H.galois_group().identity(), factor)]
        };
    }

    pub fn identity(H: &DefaultHypercube<NumberRing, A>) -> Self {
        Self {
            data: vec![(H.galois_group().identity(), H.ring().one())]
        }
    }

    pub fn shift(H: &DefaultHypercube<NumberRing, A>, positions: &[i64]) -> Self {
        assert_eq!(H.hypercube().dim_count(), positions.len());
        Self {
            data: vec![(H.hypercube().map(positions), H.ring().one())]
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
    pub fn linear_combine_shifts<V, I>(H: &DefaultHypercube<NumberRing, A>, summands: I) -> Self
        where I: Iterator<Item = (V, El<DecompositionRing<NumberRing, Zn, A>>)>,
            V: VectorFn<i64>
    {
        let mut tmp = (0..H.hypercube().dim_count()).map(|_| 0).collect::<Vec<_>>();
        let mut result = Self {
            data: summands.map(|(positions, factor)| {
                for i in 0..H.hypercube().dim_count() {
                    tmp[i] = positions.at(i);
                }
                (H.hypercube().map(&tmp), factor)
            }).collect()
        };
        result.canonicalize(H);
        return result;
    }

    fn canonicalize(&mut self, H: &DefaultHypercube<NumberRing, A>) {
        self.data.sort_unstable_by_key(|(g, _)| H.galois_group().representative(*g));
        self.data.dedup_by(|second, first| {
            if H.galois_group().eq_el(second.0, first.0) {
                H.ring().add_assign_ref(&mut first.1, &second.1);
                return true;
            } else {
                return false;
            }
        });
        self.data.retain(|(_, coeff)| !H.ring().is_zero(coeff));
    }

    
    /// 
    /// In the returned lists, we use the first entry for the "frobenius-dimension";
    /// 
    /// Note that `gcd_step[i]` will contain `1` instead of the expected `0` if there is only one entry
    /// in dimension `i` (i.e. `min_step[i] = max_step[i]`), since this makes using it via `step_by` easier.
    /// 
    fn compute_automorphisms_per_dimension(&self, H: &DefaultHypercube<NumberRing, A>) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
        self.check_valid(H);
        
        let mut max_step: Vec<usize> = Vec::new();
        let mut min_step: Vec<usize> = Vec::new();
        let mut gcd_step: Vec<usize> = Vec::new();
        let mut sizes: Vec<usize> = Vec::new();
        for i in 0..=H.hypercube().dim_count() {
            max_step.push(self.data.iter().map(|(g, _)| H.hypercube().std_preimage(*g)[i]).max().unwrap());
            min_step.push(self.data.iter().map(|(g, _)| H.hypercube().std_preimage(*g)[i]).min().unwrap());
            let gcd = self.data.iter().map(|(g, _)| H.hypercube().std_preimage(*g)[i]).fold(0, |a, b| signed_gcd(a as i64, b as i64, StaticRing::<i64>::RING) as usize);
            if gcd == 0 {
                gcd_step.push(1);
            } else {
                gcd_step.push(gcd);
            }
            assert!(gcd_step[i] > 0);
            if gcd_step[i] != 0 {
                sizes.push(StaticRing::<i64>::RING.checked_div(&((max_step[i] - min_step[i] + gcd_step[i]) as i64), &(gcd_step[i] as i64)).unwrap() as usize);
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

    pub fn to_circuit(self, H: &DefaultHypercube<NumberRing, A>) -> PlaintextCircuit<DecompositionRingBase<NumberRing, Zn, A>> {
        self.check_valid(H);

        let (_, _, _, sizes) = self.compute_automorphisms_per_dimension(H);

        const UNHOISTED_AUTO_COUNT_OVERHEAD: usize = 3;

        let preferred_baby_steps = (1..=(sizes.iter().copied().product::<usize>() as usize)).min_by_key(|preferred_baby_steps| {
            let params = Self::baby_step_giant_step_params(sizes.as_fn().map_fn(|s| *s as usize), *preferred_baby_steps);
            return params.hoisted_automorphism_count + params.unhoisted_automorphism_count * UNHOISTED_AUTO_COUNT_OVERHEAD;
        }).unwrap();

        return self.to_circuit_with_baby_steps(H, preferred_baby_steps);
    }

    pub fn to_circuit_many(transforms: Vec<Self>, H: &DefaultHypercube<NumberRing, A>) -> PlaintextCircuit<DecompositionRingBase<NumberRing, Zn, A>> {
        transforms.into_iter().fold(PlaintextCircuit::identity(1, H.ring()), |current, next| next.to_circuit(H).compose(current, H.ring()))
    }

    pub fn to_circuit_with_baby_steps(self, H: &DefaultHypercube<NumberRing, A>, preferred_baby_steps: usize) -> PlaintextCircuit<DecompositionRingBase<NumberRing, Zn, A>> {
        self.check_valid(H);

        let (max_step, min_step, gcd_step, sizes) = self.compute_automorphisms_per_dimension(H);

        let params = Self::baby_step_giant_step_params((0..sizes.len()).map_fn(|i| sizes[i] as usize), preferred_baby_steps);

        let mixed_dim_i = params.mixed_step_dimension;
        let mixed_dim_baby_steps = params.mixed_step_dimension_baby_steps;

        let giant_step_range_iters = [(min_step[mixed_dim_i]..=max_step[mixed_dim_i]).step_by(gcd_step[mixed_dim_i] * mixed_dim_baby_steps)].into_iter()
            .chain(params.pure_giant_step_dimensions.clone().map(|i| (min_step[i]..=max_step[i]).step_by(gcd_step[i])));

        let baby_step_range_iters = params.pure_baby_step_dimensions.clone().map(|i| (min_step[i]..=max_step[i]).step_by(gcd_step[i] as usize))
            .chain([(0..=((mixed_dim_baby_steps - 1) * gcd_step[mixed_dim_i])).step_by(gcd_step[mixed_dim_i])]);

        let shift_or_frobenius = |dim_or_frobenius: usize, steps: usize| if dim_or_frobenius == 0 {
            H.hypercube().frobenius(steps as i64)
        } else {
            H.hypercube().map_1d(dim_or_frobenius - 1, steps as i64)
        };

        let giant_steps_galois_els = multi_cartesian_product(giant_step_range_iters, |indices| {
            indices[1..].iter()
                .enumerate()
                .map(|(i, s)| shift_or_frobenius(i + params.pure_giant_step_dimensions.start, *s))
                .fold(shift_or_frobenius(mixed_dim_i, indices[0]), |a, b| H.galois_group().mul(a, b))
        }, |_, x| *x)
            .map(|g_el| if H.galois_group().is_identity(g_el) { None } else { Some(g_el) })
            .collect::<Vec<_>>();

        let baby_steps_galois_els = multi_cartesian_product(baby_step_range_iters, move |indices| {
            indices.iter()
                .enumerate()
                .map(|(i, s)| shift_or_frobenius(i, *s))
                .fold(shift_or_frobenius(0, 0), |a, b| H.galois_group().mul(a, b))
        }, |_, x| *x)
            .collect::<Vec<_>>();

        assert_eq!(params.hoisted_automorphism_count, baby_steps_galois_els.len() - 1);
        assert_eq!(params.unhoisted_automorphism_count, giant_steps_galois_els.len() - 1);

        let mut lin_transform_data = self.data;
        let compiled_coeffs = giant_steps_galois_els.iter().map(|gs_el| baby_steps_galois_els.iter().map(|bs_el| {
            let gs_el = gs_el.unwrap_or(H.galois_group().identity());
            let total_el = H.galois_group().mul(gs_el, *bs_el);
            let mut coeff = None;
            lin_transform_data.retain(|(g, c)| if H.galois_group().eq_el(*g, total_el) {
                debug_assert!(coeff.is_none());
                coeff = Some(H.ring().clone_el(c));
                false
            } else {
                true
            });
            if coeff.is_none() || H.ring().is_zero(coeff.as_ref().unwrap()) {
                return Coefficient::Zero;
            } else {
                return Coefficient::Other(H.ring().get_ring().apply_galois_action(coeff.as_ref().unwrap(), H.galois_group().invert(gs_el)));
            }
        }).collect::<Vec<_>>());

        let baby_step_circuit = PlaintextCircuit::gal_many(&baby_steps_galois_els, H.ring());

        let mut current = PlaintextCircuit::identity(1, H.ring()).tensor(baby_step_circuit, H.ring()).compose(PlaintextCircuit::identity(1, H.ring()).output_twice(H.ring()), H.ring());
        for (g, coeffs) in giant_steps_galois_els.iter().copied().zip(compiled_coeffs) {
            debug_assert_eq!(baby_steps_galois_els.len() + 1, current.output_count());
            let summand = if let Some(g) = g {
                let galois_of_lin_transform = PlaintextCircuit::gal(g, H.ring()).compose(PlaintextCircuit::linear_transform(&coeffs, H.ring()), H.ring());
                PlaintextCircuit::add(H.ring()).compose(
                    PlaintextCircuit::identity(1, H.ring()).tensor(galois_of_lin_transform, H.ring()),
                    H.ring()
                )
            } else {
                PlaintextCircuit::add(H.ring()).compose(
                    PlaintextCircuit::identity(1, H.ring()).tensor(PlaintextCircuit::linear_transform(&coeffs, H.ring()), H.ring()),
                    H.ring()
                )
            };
            current = summand.tensor(PlaintextCircuit::drop(1), H.ring()).tensor(PlaintextCircuit::identity(baby_steps_galois_els.len(), H.ring()), H.ring()).compose(
                current.output_twice(H.ring()), H.ring()
            );
            debug_assert_eq!(baby_steps_galois_els.len() + 1, current.output_count());
        }

        return PlaintextCircuit::identity(1, H.ring()).tensor(PlaintextCircuit::drop(baby_steps_galois_els.len()), H.ring()).compose(current, H.ring());
    }
}

impl<NumberRing, A> DecompositionRingBase<NumberRing, Zn, A> 
    where NumberRing: HECyclotomicNumberRing,
        A: Allocator + Clone
{
    pub fn compute_linear_transform(&self, H: &DefaultHypercube<NumberRing, A>, el: &<Self as RingBase>::Element, transform: &MatmulTransform<NumberRing, A>) -> <Self as RingBase>::Element {
        assert!(H.ring().get_ring() == self);
        <_ as RingBase>::sum(self, transform.data.iter().map(|(s, c)| self.mul_ref_fst(c, self.apply_galois_action(el, *s))))
    }
}

pub struct CompiledLinearTransform<NumberRing, A = Global>
    where NumberRing: HECyclotomicNumberRing,
        A: Allocator + Clone
{
    baby_step_galois_elements: Vec<CyclotomicGaloisGroupEl>,
    giant_step_galois_elements: Vec<Option<CyclotomicGaloisGroupEl>>,
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
                ApplyGaloisFn: FnMut(T, &[CyclotomicGaloisGroupEl]) -> Vec<T>,
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
    /// 
    /// In the returned lists, we use the first entry for the "frobenius-dimension";
    /// 
    /// Note that `gcd_step[i]` will contain `1` instead of the expected `0` if there is only one entry
    /// in dimension `i` (i.e. `min_step[i] = max_step[i]`), since this makes using it via `step_by` easier.
    /// 
    fn compute_automorphisms_per_dimension(H: &DefaultHypercube<NumberRing, A>, lin_transform: &MatmulTransform<NumberRing, A>) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
        lin_transform.check_valid(H);
        
        let mut max_step: Vec<usize> = Vec::new();
        let mut min_step: Vec<usize> = Vec::new();
        let mut gcd_step: Vec<usize> = Vec::new();
        let mut sizes: Vec<usize> = Vec::new();
        for i in 0..=H.hypercube().dim_count() {
            max_step.push(lin_transform.data.iter().map(|(g, _)| H.hypercube().std_preimage(*g)[i]).max().unwrap());
            min_step.push(lin_transform.data.iter().map(|(g, _)| H.hypercube().std_preimage(*g)[i]).min().unwrap());
            let gcd = lin_transform.data.iter().map(|(g, _)| H.hypercube().std_preimage(*g)[i]).fold(0, |a, b| signed_gcd(a as i64, b as i64, StaticRing::<i64>::RING) as usize);
            if gcd == 0 {
                gcd_step.push(1);
            } else {
                gcd_step.push(gcd);
            }
            assert!(gcd_step[i] > 0);
            if gcd_step[i] != 0 {
                sizes.push(StaticRing::<i64>::RING.checked_div(&((max_step[i] - min_step[i] + gcd_step[i]) as i64), &(gcd_step[i] as i64)).unwrap() as usize);
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

    pub fn compile(H: &DefaultHypercube<NumberRing, A>, lin_transform: MatmulTransform<NumberRing, A>) -> Self {
        lin_transform.check_valid(H);

        let (_, _, _, sizes) = Self::compute_automorphisms_per_dimension(H, &lin_transform);

        const UNHOISTED_AUTO_COUNT_OVERHEAD: usize = 3;

        let preferred_baby_steps = (1..=(sizes.iter().copied().product::<usize>() as usize)).min_by_key(|preferred_baby_steps| {
            let params = Self::baby_step_giant_step_params(sizes.as_fn().map_fn(|s| *s as usize), *preferred_baby_steps);
            return params.hoisted_automorphism_count + params.unhoisted_automorphism_count * UNHOISTED_AUTO_COUNT_OVERHEAD;
        }).unwrap();

        return Self::create_from(H, lin_transform, preferred_baby_steps);
    }

    pub fn compile_merged(H: &DefaultHypercube<NumberRing, A>, lin_transforms: &[MatmulTransform<NumberRing, A>]) -> Self {
        Self::compile(H, lin_transforms.iter().fold(MatmulTransform::identity(&H), |current, next| next.compose(&current, &H)))
    }

    pub fn create_from_merged(H: &DefaultHypercube<NumberRing, A>, lin_transforms: &[MatmulTransform<NumberRing, A>], preferred_baby_steps: usize) -> CompiledLinearTransform<NumberRing, A> {
        Self::create_from(H, lin_transforms.iter().fold(MatmulTransform::identity(&H), |current, next| next.compose(&current, &H)), preferred_baby_steps)
    }

    pub fn create_from(H: &DefaultHypercube<NumberRing, A>, lin_transform: MatmulTransform<NumberRing, A>, preferred_baby_steps: usize) -> CompiledLinearTransform<NumberRing, A> {
        lin_transform.check_valid(H);

        let (max_step, min_step, gcd_step, sizes) = Self::compute_automorphisms_per_dimension(H, &lin_transform);

        let params = Self::baby_step_giant_step_params((0..sizes.len()).map_fn(|i| sizes[i] as usize), preferred_baby_steps);

        let mixed_dim_i = params.mixed_step_dimension;
        let mixed_dim_baby_steps = params.mixed_step_dimension_baby_steps;

        let giant_step_range_iters = [(min_step[mixed_dim_i]..=max_step[mixed_dim_i]).step_by(gcd_step[mixed_dim_i] * mixed_dim_baby_steps)].into_iter()
            .chain(params.pure_giant_step_dimensions.clone().map(|i| (min_step[i]..=max_step[i]).step_by(gcd_step[i])));

        let baby_step_range_iters = params.pure_baby_step_dimensions.clone().map(|i| (min_step[i]..=max_step[i]).step_by(gcd_step[i] as usize))
            .chain([(0..=((mixed_dim_baby_steps - 1) * gcd_step[mixed_dim_i])).step_by(gcd_step[mixed_dim_i])]);

        let shift_or_frobenius = |dim_or_frobenius: usize, steps: usize| if dim_or_frobenius == 0 {
            H.hypercube().frobenius(steps as i64)
        } else {
            H.hypercube().map_1d(dim_or_frobenius - 1, steps as i64)
        };

        let giant_steps_galois_els = multi_cartesian_product(giant_step_range_iters, |indices| {
            indices[1..].iter()
                .enumerate()
                .map(|(i, s)| shift_or_frobenius(i + params.pure_giant_step_dimensions.start, *s))
                .fold(shift_or_frobenius(mixed_dim_i, indices[0]), |a, b| H.galois_group().mul(a, b))
        }, |_, x| *x)
            .map(|g_el| if H.galois_group().is_identity(g_el) { None } else { Some(g_el) })
            .collect::<Vec<_>>();

        let baby_steps_galois_els = multi_cartesian_product(baby_step_range_iters, move |indices| {
            indices.iter()
                .enumerate()
                .map(|(i, s)| shift_or_frobenius(i, *s))
                .fold(shift_or_frobenius(0, 0), |a, b| H.galois_group().mul(a, b))
        }, |_, x| *x)
            .collect::<Vec<_>>();

        assert_eq!(params.hoisted_automorphism_count, baby_steps_galois_els.len() - 1);
        assert_eq!(params.unhoisted_automorphism_count, giant_steps_galois_els.len() - 1);

        let mut lin_transform_data = lin_transform.data;
        let compiled_coeffs = giant_steps_galois_els.iter().map(|gs_el| baby_steps_galois_els.iter().map(|bs_el| {
            let gs_el = gs_el.unwrap_or(H.galois_group().identity());
            let total_el = H.galois_group().mul(gs_el, *bs_el);
            let mut coeff = None;
            lin_transform_data.retain(|(g, c)| if H.galois_group().eq_el(*g, total_el) {
                coeff = Some(H.ring().clone_el(c));
                false
            } else {
                true
            });
            coeff = coeff.and_then(|c| if H.ring().is_zero(&c) { None } else { Some(c) });
            let result = coeff.map(|c| H.ring().get_ring().apply_galois_action(&c, H.galois_group().invert(gs_el)));
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
#[cfg(test)]
use crate::rings::pow2_cyclotomic::*;
#[cfg(test)]
use feanor_math::assert_el_eq;

#[test]
fn test_to_circuit() {
    let ring = DecompositionRingBase::new(Pow2CyclotomicNumberRing::new(64), Zn::new(23));
    let hypercube = HypercubeStructure::halevi_shoup_hypercube(CyclotomicGaloisGroup::new(64), 23);
    let H = HypercubeIsomorphism::new::<false>(&ring, hypercube);
    let compiled_transform = MatmulTransform::to_circuit_many(slots_to_coeffs_thin(&H), &H);

    let input = H.from_slot_values([1, 2, 3, 4].into_iter().map(|n| H.slot_ring().int_hom().map(n)));
    let expected = slots_to_coeffs_thin(&H).into_iter().fold(ring.clone_el(&input), |c, T| ring.get_ring().compute_linear_transform(&H, &c, &T));
    assert_el_eq!(&ring, &expected, &compiled_transform.evaluate(&[input], ring.identity()).pop().unwrap());
    
    let ring = DecompositionRingBase::new(Pow2CyclotomicNumberRing::new(64), Zn::new(97));
    let hypercube = HypercubeStructure::halevi_shoup_hypercube(CyclotomicGaloisGroup::new(64), 97);
    let H = HypercubeIsomorphism::new::<false>(&ring, hypercube);
    let compiled_transform = MatmulTransform::to_circuit_many(slots_to_coeffs_thin(&H), &H);
    
    let input = H.from_slot_values((1..17).map(|n| H.slot_ring().int_hom().map(n)));
    let expected = slots_to_coeffs_thin(&H).into_iter().fold(ring.clone_el(&input), |c, T| ring.get_ring().compute_linear_transform(&H, &c, &T));
    assert_el_eq!(&ring, &expected, &compiled_transform.evaluate(&[input], ring.identity()).pop().unwrap());
}

#[test]
fn test_compile() {
    let ring = DecompositionRingBase::new(Pow2CyclotomicNumberRing::new(64), Zn::new(23));
    let hypercube = HypercubeStructure::halevi_shoup_hypercube(CyclotomicGaloisGroup::new(64), 23);
    let H = HypercubeIsomorphism::new::<false>(&ring, hypercube);
    let compiled_transform = slots_to_coeffs_thin(&H).into_iter().map(|T| CompiledLinearTransform::create_from(&H, T, 2)).collect::<Vec<_>>();

    let mut current = H.from_slot_values([1, 2, 3, 4].into_iter().map(|n| H.slot_ring().int_hom().map(n)));
    let expected = slots_to_coeffs_thin(&H).into_iter().fold(ring.clone_el(&current), |c, T| ring.get_ring().compute_linear_transform(&H, &c, &T));
    for T in &compiled_transform {
        current = T.evaluate(&ring, current);
    }
    assert_el_eq!(&ring, &expected, &current);
    
    let compiled_composed_transform = CompiledLinearTransform::create_from_merged(&H, &slots_to_coeffs_thin(&H), 2);
    let mut current = H.from_slot_values([1, 2, 3, 4].into_iter().map(|n| H.slot_ring().int_hom().map(n)));
    let expected = slots_to_coeffs_thin(&H).into_iter().fold(ring.clone_el(&current), |c, T| ring.get_ring().compute_linear_transform(&H, &c, &T));
    current = compiled_composed_transform.evaluate(&ring, current);
    assert_el_eq!(&ring, &expected, &current);

    let ring = DecompositionRingBase::new(Pow2CyclotomicNumberRing::new(64), Zn::new(97));
    let hypercube = HypercubeStructure::halevi_shoup_hypercube(CyclotomicGaloisGroup::new(64), 97);
    let H = HypercubeIsomorphism::new::<false>(&ring, hypercube);
    let compiled_transform = slots_to_coeffs_thin(&H).into_iter().map(|T| CompiledLinearTransform::create_from(&H, T, 2)).collect::<Vec<_>>();
    
    let mut current = H.from_slot_values((1..17).map(|n| H.slot_ring().int_hom().map(n)));
    let expected = slots_to_coeffs_thin(&H).into_iter().fold(ring.clone_el(&current), |c, T| ring.get_ring().compute_linear_transform(&H, &c, &T));
    for T in compiled_transform {
        current = T.evaluate(&ring, current);
    }
    assert_el_eq!(&ring, &expected, &current);

    let compiled_composed_transform = CompiledLinearTransform::create_from_merged(&H, &slots_to_coeffs_thin(&H), 2);
    let mut current = H.from_slot_values((1..17).map(|n| H.slot_ring().int_hom().map(n)));
    let expected = slots_to_coeffs_thin(&H).into_iter().fold(ring.clone_el(&current), |c, T| ring.get_ring().compute_linear_transform(&H, &c, &T));
    current = compiled_composed_transform.evaluate(&ring, current);
    assert_el_eq!(&ring, &expected, &current);

    let compiled_composed_transform = CompiledLinearTransform::create_from_merged(&H, &slots_to_coeffs_thin(&H), 3);
    let mut current = H.from_slot_values((1..17).map(|n| H.slot_ring().int_hom().map(n)));
    let expected = slots_to_coeffs_thin(&H).into_iter().fold(ring.clone_el(&current), |c, T| ring.get_ring().compute_linear_transform(&H, &c, &T));
    current = compiled_composed_transform.evaluate(&ring, current);
    assert_el_eq!(&ring, &expected, &current);

    let compiled_composed_transform = CompiledLinearTransform::create_from_merged(&H, &slots_to_coeffs_thin(&H), 5);
    let mut current = H.from_slot_values((1..17).map(|n| H.slot_ring().int_hom().map(n)));
    let expected = slots_to_coeffs_thin(&H).into_iter().fold(ring.clone_el(&current), |c, T| ring.get_ring().compute_linear_transform(&H, &c, &T));
    current = compiled_composed_transform.evaluate(&ring, current);
    assert_el_eq!(&ring, &expected, &current);
}

#[test]
fn test_compile_odd_case() {
    let ring = DecompositionRingBase::new(CompositeCyclotomicNumberRing::new(3, 19), Zn::new(7));
    let hypercube = HypercubeStructure::halevi_shoup_hypercube(CyclotomicGaloisGroup::new(3 * 19), 7);
    let H = HypercubeIsomorphism::new::<false>(&ring, hypercube);

    let mut current = H.from_slot_values((1..=12).map(|n| H.slot_ring().int_hom().map(n)));
    let compiled_transform = slots_to_powcoeffs_thin(&H).into_iter().map(|T| CompiledLinearTransform::compile(&H, T)).collect::<Vec<_>>();
    let mut expected = ring.clone_el(&current);
    for (compiled_transform, transform) in compiled_transform.iter().zip(slots_to_powcoeffs_thin(&H).into_iter()) {
        current = compiled_transform.evaluate(&ring, current);
        expected = ring.get_ring().compute_linear_transform(&H, &expected, &transform);
        assert_el_eq!(&ring, &expected, &current);
    }
    
    let ring = DecompositionRingBase::new(CompositeCyclotomicNumberRing::new(11, 31), Zn::new(2));
    let hypercube = HypercubeStructure::halevi_shoup_hypercube(CyclotomicGaloisGroup::new(11 * 31), 2);
    let H = HypercubeIsomorphism::new::<false>(&ring, hypercube);

    let mut current = H.from_slot_values((1..=30).map(|n| H.slot_ring().int_hom().map(n)));
    let compiled_transform = slots_to_powcoeffs_thin(&H).into_iter().map(|T| CompiledLinearTransform::compile(&H, T)).collect::<Vec<_>>();
    let mut expected = ring.clone_el(&current);
    for (compiled_transform, transform) in compiled_transform.iter().zip(slots_to_powcoeffs_thin(&H).into_iter()) {
        current = compiled_transform.evaluate(&ring, current);
        expected = ring.get_ring().compute_linear_transform(&H, &expected, &transform);
        assert_el_eq!(&ring, &expected, &current);
    }
}

#[test]
fn test_compute_automorphisms_per_dimension() {
    let ring = DecompositionRingBase::new(CompositeCyclotomicNumberRing::new(3, 19), Zn::new(7));
    let hypercube = HypercubeStructure::halevi_shoup_hypercube(CyclotomicGaloisGroup::new(3 * 19), 7);
    let H = HypercubeIsomorphism::new::<false>(&ring, hypercube);
    assert_eq!(2, H.hypercube().dim_count());
    assert_eq!(3, H.slot_ring().rank());
    assert_eq!(6, H.hypercube().m(0));
    assert_eq!(2, H.hypercube().m(1));

    let transform = MatmulTransform::blockmatmul1d(&H, 0, |i, j, _| H.slot_ring().base_ring().one());
    let (max, min, gcd, sizes) = CompiledLinearTransform::compute_automorphisms_per_dimension(&H, &transform);
    assert_eq!(vec![2, 5, 0], max);
    assert_eq!(vec![0, 0, 0], min);
    assert_eq!(vec![1, 1, 1], gcd);
    assert_eq!(vec![3, 6, 1], sizes);
}

#[test]
fn test_compose() {
    let ring = DecompositionRingBase::new(Pow2CyclotomicNumberRing::new(64), Zn::new(23));
    let hypercube = HypercubeStructure::halevi_shoup_hypercube(CyclotomicGaloisGroup::new(64), 23);
    let H = HypercubeIsomorphism::new::<false>(&ring, hypercube);
    let composed_transform = slots_to_coeffs_thin(&H).into_iter().fold(MatmulTransform::identity(&H), |current, next| next.compose(&current, &H));

    let mut current = H.from_slot_values([1, 2, 3, 4].into_iter().map(|n| H.slot_ring().int_hom().map(n)));
    let mut expected = ring.clone_el(&current);
    current = ring.get_ring().compute_linear_transform(&H, &current, &composed_transform);
    expected = slots_to_coeffs_thin(&H).into_iter().fold(expected, |c, T| ring.get_ring().compute_linear_transform(&H, &c, &T));

    assert_el_eq!(&ring, &expected, &current);
    
    let ring = DecompositionRingBase::new(Pow2CyclotomicNumberRing::new(64), Zn::new(97));
    let hypercube = HypercubeStructure::halevi_shoup_hypercube(CyclotomicGaloisGroup::new(64), 97);
    let H = HypercubeIsomorphism::new::<false>(&ring, hypercube);
    let composed_transform = slots_to_coeffs_thin(&H).into_iter().fold(MatmulTransform::identity(&H), |current, next| next.compose(&current, &H));
    
    let mut current = H.from_slot_values((1..17).map(|n| H.slot_ring().int_hom().map(n)));
    let mut expected = ring.clone_el(&current);
    current = ring.get_ring().compute_linear_transform(&H, &current, &composed_transform);
    expected = slots_to_coeffs_thin(&H).into_iter().fold(expected, |c, T| ring.get_ring().compute_linear_transform(&H, &c, &T));

    assert_el_eq!(&ring, &expected, &current);
}

#[test]
fn test_invert() {
    let ring = DecompositionRingBase::new(Pow2CyclotomicNumberRing::new(64), Zn::new(23));
    let hypercube = HypercubeStructure::halevi_shoup_hypercube(CyclotomicGaloisGroup::new(64), 23);
    let H = HypercubeIsomorphism::new::<false>(&ring, hypercube);
    let composed_transform = slots_to_coeffs_thin(&H).into_iter().fold(MatmulTransform::identity(&H), |current, next| next.compose(&current, &H));
    let inv_transform = composed_transform.inverse(&H);

    let expected = H.from_slot_values([1, 2, 3, 4].into_iter().map(|n| H.slot_ring().int_hom().map(n)));
    let current = slots_to_coeffs_thin(&H).into_iter().fold(ring.clone_el(&expected), |c, T| ring.get_ring().compute_linear_transform(&H, &c, &T));
    let actual = ring.get_ring().compute_linear_transform(&H, &current, &inv_transform);
    assert_el_eq!(&ring, &expected, &actual);
    
    let ring = DecompositionRingBase::new(Pow2CyclotomicNumberRing::new(64), Zn::new(97));
    let hypercube = HypercubeStructure::halevi_shoup_hypercube(CyclotomicGaloisGroup::new(64), 97);
    let H = HypercubeIsomorphism::new::<false>(&ring, hypercube);
    let composed_transform = slots_to_coeffs_thin(&H).into_iter().fold(MatmulTransform::identity(&H), |current, next| next.compose(&current, &H));
    let inv_transform = composed_transform.inverse(&H);
    
    let expected = H.from_slot_values((1..17).map(|n| H.slot_ring().int_hom().map(n)));
    let current = slots_to_coeffs_thin(&H).into_iter().fold(ring.clone_el(&expected), |c, T| ring.get_ring().compute_linear_transform(&H, &c, &T));
    let actual = ring.get_ring().compute_linear_transform(&H, &current, &inv_transform);

    assert_el_eq!(&ring, &expected, &actual);
}

#[test]
fn test_blockmatmul1d() {
    // F23[X]/(Phi_5) ~ F_(23^4)
    let ring = DecompositionRingBase::new(OddCyclotomicNumberRing::new(5), Zn::new(23));
    let hypercube = HypercubeStructure::halevi_shoup_hypercube(CyclotomicGaloisGroup::new(5), 23);
    let H = HypercubeIsomorphism::new::<false>(&ring, hypercube);
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
    let hypercube = HypercubeStructure::halevi_shoup_hypercube(CyclotomicGaloisGroup::new(7), 23);
    let H = HypercubeIsomorphism::new::<false>(&ring, hypercube);
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
        let input = H.from_slot_values([H.slot_ring().zero(), H.slot_ring().pow(H.slot_ring().canonical_gen(), i)]);
        let expected = H.from_slot_values([H.slot_ring().from_canonical_basis((0..3).map(|j| H.slot_ring().base_ring().int_hom().map(matrix[j][i]))), H.slot_ring().zero()]);
        let actual = ring.get_ring().compute_linear_transform(&H, &input, &lin_transform);
        assert_el_eq!(H.ring(), &expected, &actual);
    }

    for i in 0..3 {
        let input = H.from_slot_values([H.slot_ring().pow(H.slot_ring().canonical_gen(), i), H.slot_ring().zero()]);
        let expected = H.ring().zero();
        let actual = ring.get_ring().compute_linear_transform(&H, &input, &lin_transform);
        assert_el_eq!(H.ring(), &expected, &actual);
    }
}