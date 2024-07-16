use std::num::NonZeroI64;

use feanor_math::homomorphism::{CanHomFrom, Homomorphism};
use feanor_math::primitive_int::{StaticRing, StaticRingBase};
use feanor_math::ring::*;
use feanor_math::algorithms::matmul::ComputeInnerProduct;

pub mod precomputed;

#[derive(Clone, Debug, PartialEq, Eq)]
struct LinTransform {
    factors: Vec<Option<NonZeroI64>>,
    constant: Option<NonZeroI64>
}

impl LinTransform {

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
                constant: None,
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

    pub fn evaluate<'a, R>(&'a self, inputs: &'a [El<R>], ring: R) -> impl 'a + ExactSizeIterator<Item = El<R>>
        where R: 'a + RingStore + Clone,
            R::Type: CanHomFrom<StaticRingBase<i64>>
    {
        assert_eq!(inputs.len(), self.input_count());
        let hom = ring.clone().into_can_hom(&StaticRing::<i64>::RING).ok().unwrap();

        let mut current = Vec::new();
        for mul in &self.multiplications {
            let lhs = mul.lhs.evaluate(&inputs, &current, &ring, &hom);
            let rhs = mul.rhs.evaluate(&inputs, &current, &ring, &hom);
            current.push(ring.mul(lhs, rhs));
        }
        return self.output_transforms.iter().map(move |t| t.evaluate(inputs, &current, &ring, &hom));
    }
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