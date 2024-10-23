use std::num::NonZeroI64;

use feanor_math::homomorphism::{CanHomFrom, Homomorphism};
use feanor_math::primitive_int::{StaticRing, StaticRingBase};
use feanor_math::ring::*;
use feanor_math::algorithms::matmul::ComputeInnerProduct;

pub mod polys;

#[derive(Clone, Debug, PartialEq, Eq)]
struct LinTransform {
    factors: Vec<Option<NonZeroI64>>,
    constant: Option<NonZeroI64>
}

impl LinTransform {

    pub fn evaluate_generic<T, AddScaled, FromI64>(&self, first_inputs: &[T], second_inputs: &[T], add_scaled: &mut AddScaled, from: &mut FromI64) -> T
        where AddScaled: FnMut(T, &T, i64) -> T,
            FromI64: FnMut(i64) -> T
    {
        assert_eq!(self.factors.len(), first_inputs.len() + second_inputs.len());
        let mut result = from(self.constant.map(|x| x.into()).unwrap_or(0));
        for (i, c) in self.factors.iter().enumerate() {
            if let Some(c) = c {
                if i < first_inputs.len() {
                    result = add_scaled(result, &first_inputs[i], (*c).into());
                } else {
                    result = add_scaled(result, &second_inputs[i - first_inputs.len()], (*c).into());
                }
            }
        }
        return result;
    }

    fn evaluate<R, H>(&self, first_inputs: &[El<R>], second_inputs: &[El<R>], ring: R, hom: H) -> El<R>
        where R: RingStore,
            H: Homomorphism<StaticRingBase<i64>, R::Type>
    {
        assert_eq!(self.factors.len(), first_inputs.len() + second_inputs.len());
        let result = ring.add(
            <_ as ComputeInnerProduct>::inner_product_ref_fst(
                ring.get_ring(), 
                self.factors[0..first_inputs.len()].iter().zip(first_inputs.iter()).filter(|(c, _)| c.is_some()).map(|(c, x)| (x, hom.map(c.unwrap().into())))
            ),
            <_ as ComputeInnerProduct>::inner_product_ref_fst(
                ring.get_ring(), 
                self.factors[first_inputs.len()..].iter().zip(second_inputs.iter()).filter(|(c, _)| c.is_some()).map(|(c, x)| (x, hom.map(c.unwrap().into())))
            ),
        );
        if let Some(c) = self.constant {
            return ring.add(result, hom.map(c.into()));
        } else {
            return result;
        }
    }

    fn compose(&self, input_transforms: &[LinTransform]) -> LinTransform {
        assert_eq!(self.factors.len(), input_transforms.len());
        if input_transforms.len() == 0 {
            return self.clone();
        }
        let new_input_count = input_transforms[0].factors.len();
        assert!(input_transforms.iter().all(|t| t.factors.len() == new_input_count));
        let mut result_factors: Vec<i64> = (0..new_input_count).map(|_| 0).collect();
        let mut result_constant: i64 = self.constant.map(|c| c.into()).unwrap_or(0);
        for (c, t) in self.factors.iter().zip(input_transforms.iter()) {
            if let Some(c) = c {
                let c: i64 = (*c).into();
                result_constant += c * t.constant.map(|c| c.into()).unwrap_or(0);
                for i in 0..new_input_count {
                    result_factors[i] += c * t.factors[i].map(|c| c.into()).unwrap_or(0);
                }
            }
        }
        return LinTransform {
            constant: NonZeroI64::new(result_constant),
            factors: result_factors.into_iter().map(|c| NonZeroI64::new(c)).collect()
        };
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ArithCircuitMul {
    lhs: LinTransform,
    rhs: LinTransform
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArithCircuit {
    input_count: usize,
    multiplications: Vec<ArithCircuitMul>,
    output_transforms: Vec<LinTransform>
}

impl ArithCircuit {

    pub fn check_invariants(&self) {
        let mut current_count = self.input_count;
        for mul in &self.multiplications {
            assert_eq!(current_count, mul.lhs.factors.len());
            assert_eq!(current_count, mul.rhs.factors.len());
            current_count += 1;
        }
        for out in &self.output_transforms {
            assert_eq!(current_count, out.factors.len());
        }
    }

    pub fn empty() -> ArithCircuit {
        ArithCircuit {
            input_count: 0,
            multiplications: Vec::new(),
            output_transforms: Vec::new()
        }
    }

    pub fn constant(c: i64) -> ArithCircuit {
        ArithCircuit {
            input_count: 0,
            multiplications: Vec::new(),
            output_transforms: vec![LinTransform {
                constant: NonZeroI64::new(c),
                factors: Vec::new()
            }]
        }
    }

    pub fn linear_transform(coeffs: &[i64]) -> ArithCircuit {
        ArithCircuit {
            input_count: coeffs.len(),
            multiplications: Vec::new(),
            output_transforms: vec![LinTransform {
                constant: None,
                factors: coeffs.iter().map(|c| NonZeroI64::new(*c)).collect()
            }]
        }
    }

    pub fn add() -> ArithCircuit {
        ArithCircuit {
            input_count: 2,
            multiplications: Vec::new(),
            output_transforms: vec![LinTransform {
                constant: None,
                factors: vec![NonZeroI64::new(1), NonZeroI64::new(1)]
            }]
        }
    }

    pub fn mul() -> ArithCircuit {
        ArithCircuit {
            input_count: 2,
            multiplications: vec![ArithCircuitMul {
                lhs: LinTransform {
                    constant: None,
                    factors: vec![NonZeroI64::new(1), None]
                },
                rhs: LinTransform {
                    constant: None,
                    factors: vec![None, NonZeroI64::new(1)]
                }
            }],
            output_transforms: vec![LinTransform {
                constant: None,
                factors: vec![None, None, NonZeroI64::new(1)]
            }]
        }
    }

    pub fn output_twice(&self) -> ArithCircuit {
        self.output_times(2)
    }

    pub fn identity(wire_count: usize) -> ArithCircuit {
        ArithCircuit {
            input_count: wire_count,
            multiplications: Vec::new(),
            output_transforms: (0..wire_count).map(|i| LinTransform {
                constant: None,
                factors: (0..wire_count).map(|j| if j == i { NonZeroI64::new(1) } else { None }).collect()
            }).collect()
        }
    }

    pub fn select(input_wire_count: usize, output_wires: &[usize]) -> ArithCircuit {
        ArithCircuit {
            input_count: input_wire_count,
            multiplications: Vec::new(),
            output_transforms: output_wires.iter().map(|i| {
                assert!(*i < input_wire_count);
                LinTransform {
                    constant: None,
                    factors: (0..input_wire_count).map(|j| if *i == j { NonZeroI64::new(1) } else { None }).collect()
                }
            }).collect()
        }
    }

    pub fn output_times(&self, times: usize) -> ArithCircuit {
        ArithCircuit {
            input_count: self.input_count,
            multiplications: self.multiplications.clone(),
            output_transforms: (0..times).flat_map(|_| self.output_transforms.iter()).cloned().collect()
        }
    }

    pub fn tensor(&self, rhs: &ArithCircuit) -> ArithCircuit {
        let add_nones = |vec: &[Option<NonZeroI64>], index: usize, count: usize| 
            vec[0..index].iter().copied().chain(std::iter::repeat(None).take(count)).chain(vec[index..].iter().copied()).collect::<Vec<_>>();
        let map_self_transform = |t: &LinTransform| LinTransform {
            constant: t.constant,
            factors: add_nones(&t.factors, self.input_count, rhs.input_count)
        };
        let map_rhs_transform = |t: &LinTransform| LinTransform {
            constant: t.constant,
            factors: add_nones(&add_nones(&t.factors, rhs.input_count, self.multiplications.len()), 0, self.input_count)
        };
        let result = ArithCircuit {
            input_count: self.input_count + rhs.input_count,
            multiplications: self.multiplications.iter().map(|mul| ArithCircuitMul {
                lhs: map_self_transform(&mul.lhs),
                rhs: map_self_transform(&mul.rhs)
            }).chain(
                rhs.multiplications.iter().map(|mul| ArithCircuitMul {
                    lhs: map_rhs_transform(&mul.lhs),
                    rhs: map_rhs_transform(&mul.rhs)
                })
            ).collect(),
            output_transforms: self.output_transforms.iter().map(|t| {
                let added_inputs_t = map_self_transform(t);
                LinTransform {
                    factors: add_nones(&added_inputs_t.factors, self.input_count + rhs.input_count + self.multiplications.len(), rhs.multiplications.len()),
                    constant: added_inputs_t.constant
                }
            }).chain(rhs.output_transforms.iter().map(map_rhs_transform)).collect()
        };
        result.check_invariants();
        return result;
    }

    pub fn compose(&self, first: &ArithCircuit) -> ArithCircuit {
        assert_eq!(first.output_count(), self.input_count());

        let map_transform = |t: &LinTransform| {
            let input_transform = LinTransform {
                constant: t.constant,
                factors: t.factors[0..self.input_count].iter().copied().collect()
            };
            let mut result = input_transform.compose(&first.output_transforms);
            result.factors.extend(t.factors[self.input_count..].iter().copied());
            return result;
        };
        let result = ArithCircuit {
            input_count: first.input_count,
            multiplications: first.multiplications.iter().cloned().chain(
                self.multiplications.iter().map(|mul| ArithCircuitMul {
                    lhs: map_transform(&mul.lhs),
                    rhs: map_transform(&mul.rhs)
                })
            ).collect(),
            output_transforms: self.output_transforms.iter().map(map_transform).collect()
        };
        result.check_invariants();
        return result;
    }

    pub fn input_count(&self) -> usize {
        self.input_count
    }

    pub fn output_count(&self) -> usize {
        self.output_transforms.len()
    }

    pub fn evaluate<'a, R>(&'a self, inputs: &'a [El<R>], ring: R) -> impl ExactSizeIterator<Item = El<R>> + 'a
        where R: 'a + RingStore + Clone,
            R::Type: CanHomFrom<StaticRingBase<i64>>
    {
        assert_eq!(inputs.len(), self.input_count());
        let hom = ring.clone().into_can_hom(&StaticRing::<i64>::RING).ok().unwrap();

        let mut current = Vec::new();
        for mul in &self.multiplications {
            let lhs = mul.lhs.evaluate(&inputs, &current, &ring, &hom);
            let rhs = mul.rhs.evaluate(&inputs, &current, &ring, &hom);
            let prod = ring.mul(lhs, rhs);
            current.push(prod);
        }
        return self.output_transforms.iter().map(move |t| t.evaluate(inputs, &current, &ring, &hom));
    }

    pub fn evaluate_generic<'a, T, AddScaled, Mul, FromI64>(&'a self, inputs: &'a [T], mut add_scaled_fn: AddScaled, mut mul_fn: Mul, mut from_fn: FromI64) -> impl ExactSizeIterator<Item = T> + 'a
        where AddScaled: 'a + FnMut(T, &T, i64) -> T,
            Mul: 'a + FnMut(T, T) -> T,
            FromI64: 'a + FnMut(i64) -> T
    {
        assert_eq!(inputs.len(), self.input_count());
        let mut current = Vec::new();
        for mul in &self.multiplications {
            let lhs = mul.lhs.evaluate_generic(inputs, &current, &mut add_scaled_fn, &mut from_fn);
            let rhs = mul.rhs.evaluate_generic(inputs, &current, &mut add_scaled_fn, &mut from_fn);
            let prod = mul_fn(lhs, rhs);
            current.push(prod);
        }
        return self.output_transforms.iter().map(move |t| t.evaluate_generic(inputs, &current, &mut add_scaled_fn, &mut from_fn));
    }

    pub fn mul_count(&self) -> usize {
        self.multiplications.len()
    }

    pub fn mul_depth(&self, output: usize) -> usize {
        let mut mul_depths = (0..self.input_count).map(|_| 0).collect::<Vec<_>>();
        for mul in &self.multiplications {
            mul_depths.push(
                mul.lhs.factors.iter().enumerate().chain(mul.rhs.factors.iter().enumerate()).filter(|(_, c)| c.is_some()).map(|(j, _)| mul_depths[j]).max().unwrap_or(0) + 1
            );
        }
        return self.output_transforms[output].factors.iter().enumerate().filter(|(_, c)| c.is_some()).map(|(j, _)| mul_depths[j]).max().unwrap_or(0);
    }

    pub fn max_mul_depth(&self) -> usize {
        let mut mul_depths = (0..self.input_count).map(|_| 0).collect::<Vec<_>>();
        for mul in &self.multiplications {
            mul_depths.push(
                mul.lhs.factors.iter().enumerate().chain(mul.rhs.factors.iter().enumerate()).filter(|(_, c)| c.is_some()).map(|(j, _)| mul_depths[j]).max().unwrap_or(0) + 1
            );
        }
        return *mul_depths.iter().max().unwrap_or(&0);
    }
}

#[test]
fn test_circuit_mul_depth() {
    let x = ArithCircuit::identity(1);
    let x_sqr = ArithCircuit::mul().compose(&x.output_twice());
    let xy = ArithCircuit::mul();
    let x_sqr_add_xy = ArithCircuit::add().compose(&x_sqr.tensor(&xy).compose(&ArithCircuit::select(2, &[0, 0, 1])));
    let result = ArithCircuit::mul().compose(&x_sqr_add_xy.output_twice());
    assert_eq!(2, result.max_mul_depth());
    assert_eq!(2, result.mul_depth(0));
    assert_eq!(3, result.mul_count());
}

#[test]
fn test_circuit_tensor_compose() {
    let x = ArithCircuit::linear_transform(&[1]);
    let x_sqr = ArithCircuit::mul().compose(&x.output_twice());
    assert_eq!(&ArithCircuit {
        input_count: 1,
        multiplications: vec![ArithCircuitMul {
            lhs: LinTransform {
                constant: None,
                factors: vec![NonZeroI64::new(1)]
            },
            rhs: LinTransform {
                constant: None,
                factors: vec![NonZeroI64::new(1)]
            }
        }],
        output_transforms: vec![LinTransform {
            constant: None,
            factors: vec![None, NonZeroI64::new(1)]
        }]
    }, &x_sqr);

    let x = ArithCircuit::linear_transform(&[1]);
    let y = ArithCircuit::linear_transform(&[1]);
    let x_y_x_y = x.tensor(&y).output_twice();
    // z = 2 * x + 3 * y
    let x_y_z = x.tensor(&x).tensor(&ArithCircuit::linear_transform(&[2, 3])).compose(&x_y_x_y);
    let xy_z = ArithCircuit::mul().tensor(&x).compose(&x_y_z);
    // w = x * y * (2 * x + 3 * y)
    let w = ArithCircuit::mul().compose(&xy_z);
    for x in -5..5 {
        for y in -5..5 {
            assert_eq!(x * y * (2 * x + 3 * y), w.evaluate(&[x, y], StaticRing::<i64>::RING).next().unwrap());
        }
    }

    let w_1_sqr = ArithCircuit::mul().compose(&ArithCircuit::add().compose(&w.tensor(&ArithCircuit::constant(1))).output_twice());
    for x in -5..5 {
        for y in -5..5 {
            assert_eq!(StaticRing::<i64>::RING.pow(x * y * (2 * x + 3 * y) + 1, 2), w_1_sqr.evaluate(&[x, y], StaticRing::<i64>::RING).next().unwrap());
        }
    }
}

#[test]
fn test_giant_step_circuit() {
    let powers = ArithCircuit::identity(1).tensor(&ArithCircuit::mul()).tensor(&ArithCircuit::mul()).compose(&ArithCircuit::mul().output_times(4).tensor(&ArithCircuit::identity(1))).compose(&ArithCircuit::identity(1).output_times(3));
    assert_eq!(vec![4, 16, 8], powers.evaluate(&[2], StaticRing::<i64>::RING).collect::<Vec<_>>());

    let permuted_baby_step_dupl_input = ArithCircuit::constant(1).tensor(&ArithCircuit::identity(1)).tensor(&powers);
    assert_eq!(vec![1, 2, 4, 16, 8], permuted_baby_step_dupl_input.evaluate(&[2, 2], StaticRing::<i64>::RING).collect::<Vec<_>>());

    let copy_input = ArithCircuit::identity(1).output_twice();
    assert_eq!(vec![2, 2], copy_input.evaluate(&[2], StaticRing::<i64>::RING).collect::<Vec<_>>());

    let permuted_baby_steps = permuted_baby_step_dupl_input.compose(&copy_input);
    assert_eq!(vec![1, 2, 4, 16, 8], permuted_baby_steps.evaluate(&[2], StaticRing::<i64>::RING).collect::<Vec<_>>());

    let baby_steps = ArithCircuit::select(5, &[0, 1, 2, 4, 3]).compose(&permuted_baby_steps);
    assert_eq!(1, baby_steps.input_count());
    assert_eq!(5, baby_steps.output_count());
    assert_eq!(vec![1, 2, 4, 8, 16], baby_steps.evaluate(&[2], StaticRing::<i64>::RING).collect::<Vec<_>>());

    let giant_steps_before_baby_steps = ArithCircuit::constant(1).tensor(&ArithCircuit::identity(1));
    let baby_and_giant_steps = ArithCircuit::identity(4).tensor(&giant_steps_before_baby_steps).compose(&baby_steps);
    assert_eq!(vec![1, 2, 4, 8, 1, 16], baby_and_giant_steps.evaluate(&[2], StaticRing::<i64>::RING).collect::<Vec<_>>());
}