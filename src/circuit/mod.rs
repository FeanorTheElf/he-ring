use feanor_math::algorithms::matmul::ComputeInnerProduct;
use feanor_math::homomorphism::Homomorphism;
use feanor_math::primitive_int::StaticRing;
use feanor_math::ring::*;

use crate::cyclotomic::{CyclotomicGaloisGroupEl, CyclotomicRing, CyclotomicRingStore};

struct LinearCombination<R: ?Sized + RingBase> {
    factors: Vec<Option<R::Element>>,
    constant: Option<R::Element>
}

impl<R: ?Sized + RingBase> LinearCombination<R> {

    fn clone<S: RingStore<Type = R>>(&self, ring: S) -> Self {
        Self {
            factors: self.factors.iter().map(|c| c.as_ref().map(|c| ring.clone_el(c))).collect(),
            constant: self.constant.as_ref().map(|c| ring.clone_el(c))
        }
    }

    fn evaluate_generic<T, ContantFn, AddProductFn>(&self, first_inputs: &[T], second_inputs: &[T], mut constant: ContantFn, mut add_prod: AddProductFn) -> T
        where ContantFn: FnMut(&Option<R::Element>) -> T,
            AddProductFn: FnMut(T, &R::Element, &T) -> T
    {
        assert_eq!(self.factors.len(), first_inputs.len() + second_inputs.len());
        let mut current = constant(&self.constant);
        for (factor, input) in self.factors.iter().zip(first_inputs.iter().chain(second_inputs.iter())) {
            if let Some(factor) = factor {
                current = add_prod(current, factor, input);
            }
        }
        return current;
    }

    fn compose<S>(&self, input_transforms: &[LinearCombination<R>], ring: S) -> LinearCombination<R>
        where S: RingStore<Type = R>
    {
        assert_eq!(self.factors.len(), input_transforms.len());
        if input_transforms.len() == 0 {
            return self.clone(ring);
        }
        let new_input_count = input_transforms[0].factors.len();
        assert!(input_transforms.iter().all(|t| t.factors.len() == new_input_count));
        let mut result_factors = (0..new_input_count).map(|_| ring.zero()).collect::<Vec<_>>();
        let mut result_constant = self.constant.as_ref().map(|c| ring.clone_el(c)).unwrap_or(ring.zero());
        for (factor, t) in self.factors.iter().zip(input_transforms.iter()) {
            if let Some(factor) = factor {
                if let Some(t_constant) = &t.constant {
                    ring.add_assign(&mut result_constant, ring.mul_ref(factor, t_constant));
                }
                for i in 0..new_input_count {
                    if let Some(t_factor) = &t.factors[i] {
                        ring.add_assign(&mut result_factors[i], ring.mul_ref(factor, t_factor));
                    }
                }
            }
        }
        return LinearCombination {
            constant: if ring.is_zero(&result_constant) { None } else { Some(result_constant) },
            factors: result_factors.into_iter().map(|c| if ring.is_zero(&c) { None } else { Some(c) }).collect()
        };
    }
}

impl<R: RingBase + Default> PartialEq for LinearCombination<R> {
    
    fn eq(&self, other: &Self) -> bool {
        assert_eq!(self.factors.len(), other.factors.len());
        let ring = RingValue::<R>::default();
        return self.constant.is_some() == other.constant.is_some() &&
            (self.constant.is_none() || ring.eq_el(self.constant.as_ref().unwrap(), other.constant.as_ref().unwrap())) &&
            self.factors.iter().zip(other.factors.iter()).all(|(lhs, rhs)| 
                lhs.is_some() == rhs.is_some() &&
                (lhs.is_none() || ring.eq_el(lhs.as_ref().unwrap(), rhs.as_ref().unwrap()))
            );
    }
}

enum PlainCircuitGate<R: ?Sized + RingBase> {
    Mul(LinearCombination<R>, LinearCombination<R>),
    Gal(Vec<CyclotomicGaloisGroupEl>, LinearCombination<R>)
}

impl<R: ?Sized + RingBase> PlainCircuitGate<R> {
    
    fn clone<S: RingStore<Type = R> + Copy>(&self, ring: S) -> Self {
        match self {
            PlainCircuitGate::Mul(lhs, rhs) => PlainCircuitGate::Mul(lhs.clone(ring), rhs.clone(ring)),
            PlainCircuitGate::Gal(gs, t) => PlainCircuitGate::Gal(gs.clone(), t.clone(ring))
        }
    }
}

impl<R: RingBase + Default> PartialEq for PlainCircuitGate<R> {
    
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (PlainCircuitGate::Mul(self_lhs, self_rhs), PlainCircuitGate::Mul(other_lhs, other_rhs)) => self_lhs == other_lhs && self_rhs == other_rhs,
            _ => false
        }
    }
}

pub struct PlainCircuit<R: ?Sized + RingBase> {
    input_count: usize,
    gates: Vec<PlainCircuitGate<R>>,
    output_transforms: Vec<LinearCombination<R>>
}

impl<R: RingBase + Default> PartialEq for PlainCircuit<R> {

    fn eq(&self, other: &Self) -> bool {
        self.input_count == other.input_count && self.gates == other.gates && self.output_transforms == other.output_transforms
    }
}

impl<R: ?Sized + RingBase> PlainCircuit<R> {

    pub fn check_invariants(&self) {
        let mut current_count = self.input_count;
        for gate in &self.gates {
            match gate {
                PlainCircuitGate::Mul(lhs, rhs) => {
                    assert_eq!(current_count, lhs.factors.len());
                    assert_eq!(current_count, rhs.factors.len());
                    current_count += 1;
                },
                PlainCircuitGate::Gal(gs, t) => {
                    assert_eq!(current_count, t.factors.len());
                    current_count += gs.len();
                }
            }
        }
        for out in &self.output_transforms {
            assert_eq!(current_count, out.factors.len());
        }
    }

    fn computed_wire_count(&self) -> usize {
        self.gates.iter().map(|gate| match gate {
            PlainCircuitGate::Mul(_, _) => 1,
            PlainCircuitGate::Gal(gs, _) => gs.len()
        }).sum()
    }

    pub fn empty() -> Self {
        Self {
            input_count: 0,
            gates: Vec::new(),
            output_transforms: Vec::new()
        }
    }

    pub fn constant<S: RingStore<Type = R>>(el: El<S>, ring: S) -> Self {
        Self {
            input_count: 0,
            gates: Vec::new(),
            output_transforms: vec![LinearCombination {
                constant: if ring.is_zero(&el) { None } else { Some(el) },
                factors: Vec::new()
            }]
        }
    }

    pub fn linear_transform<S: RingStore<Type = R>>(coeffs: &[El<S>], ring: S) -> Self {
        Self {
            input_count: coeffs.len(),
            gates: Vec::new(),
            output_transforms: vec![LinearCombination {
                constant: None,
                factors: coeffs.iter().map(|c| if ring.is_zero(c) { None } else { Some(ring.clone_el(c)) }).collect()
            }]
        }
    }

    pub fn add<S: RingStore<Type = R>>(ring: S) -> Self {
        Self {
            input_count: 2,
            gates: Vec::new(),
            output_transforms: vec![LinearCombination {
                constant: None,
                factors: vec![Some(ring.one()), Some(ring.one())]
            }]
        }
    }

    pub fn mul<S: RingStore<Type = R>>(ring: S) -> Self {
        Self {
            input_count: 2,
            gates: vec![PlainCircuitGate::Mul(
                LinearCombination {
                    constant: None,
                    factors: vec![Some(ring.one()), None]
                },
                LinearCombination {
                    constant: None,
                    factors: vec![None, Some(ring.one())]
                }
            )],
            output_transforms: vec![LinearCombination {
                constant: None,
                factors: vec![None, None, Some(ring.one())]
            }]
        }
    }

    pub fn gal<S: RingStore<Type = R>>(g: CyclotomicGaloisGroupEl, ring: S) -> Self {
        Self {
            input_count: 1,
            gates: vec![PlainCircuitGate::Gal(vec![g], LinearCombination {
                constant: None,
                factors: vec![Some(ring.one())]
            })],
            output_transforms: vec![LinearCombination {
                constant: None,
                factors: vec![None, Some(ring.one())]
            }]
        }
    }

    pub fn gal_many<S: RingStore<Type = R>>(gs: &[CyclotomicGaloisGroupEl], ring: S) -> Self {
        Self {
            input_count: 1,
            gates: vec![PlainCircuitGate::Gal(
                gs.to_owned(), 
                LinearCombination {
                    constant: None,
                    factors: vec![Some(ring.one())]
                }
            )],
            output_transforms: vec![LinearCombination {
                constant: None,
                factors: vec![None, Some(ring.one())]
            }]
        }
    }

    pub fn output_twice<S: RingStore<Type = R> + Copy>(&self, ring: S) -> Self {
        self.output_times(2, ring)
    }

    pub fn identity<S: RingStore<Type = R>>(wire_count: usize, ring: S) -> Self {
        Self {
            input_count: wire_count,
            gates: Vec::new(),
            output_transforms: (0..wire_count).map(|i| LinearCombination {
                constant: None,
                factors: (0..wire_count).map(|j| if j == i { Some(ring.one()) } else { None }).collect()
            }).collect()
        }
    }

    pub fn select<S: RingStore<Type = R>>(input_wire_count: usize, output_wires: &[usize], ring: S) -> Self {
        Self {
            input_count: input_wire_count,
            gates: Vec::new(),
            output_transforms: output_wires.iter().map(|i| {
                assert!(*i < input_wire_count);
                LinearCombination {
                    constant: None,
                    factors: (0..input_wire_count).map(|j| if *i == j { Some(ring.one()) } else { None }).collect()
                }
            }).collect()
        }
    }

    pub fn output_times<S: RingStore<Type = R> + Copy>(&self, times: usize, ring: S) -> Self {
        Self {
            input_count: self.input_count,
            gates: self.gates.iter().map(|gate| gate.clone(ring)).collect(),
            output_transforms: (0..times).flat_map(|_| self.output_transforms.iter()).map(|lin| lin.clone(ring)).collect()
        }
    }

    pub fn tensor<S: RingStore<Type = R>>(&self, rhs: &Self, ring: S) -> Self {
        let add_nones = |vec: &[Option<El<S>>], index: usize, count: usize| 
            vec[0..index].iter().map(|c| c.as_ref().map(|c| ring.clone_el(c)))
                .chain(std::iter::repeat_with(|| None).take(count))
                .chain(vec[index..].iter().map(|c| c.as_ref().map(|c| ring.clone_el(c))))
                .collect::<Vec<_>>();

        let map_self_transform = |t: &LinearCombination<R>| LinearCombination {
            constant: t.constant.as_ref().map(|c| ring.clone_el(c)),
            factors: add_nones(&t.factors, self.input_count, rhs.input_count)
        };
        let map_rhs_transform = |t: &LinearCombination<R>| LinearCombination {
            constant: t.constant.as_ref().map(|c| ring.clone_el(c)),
            factors: add_nones(&add_nones(&t.factors, rhs.input_count, self.computed_wire_count()), 0, self.input_count)
        };
        let result = Self {
            input_count: self.input_count + rhs.input_count,
            gates: self.gates.iter().map(|gate| match gate {
                PlainCircuitGate::Mul(lhs, rhs) => PlainCircuitGate::Mul(
                    map_self_transform(&lhs),
                    map_self_transform(&rhs)
                ),
                PlainCircuitGate::Gal(gs, t) => PlainCircuitGate::Gal(
                    gs.clone(), 
                    map_self_transform(t)
                )
            }).chain(
                rhs.gates.iter().map(|gate| match gate {
                    PlainCircuitGate::Mul(lhs, rhs) => PlainCircuitGate::Mul(
                        map_rhs_transform(&lhs),
                        map_rhs_transform(&rhs)
                    ),
                    PlainCircuitGate::Gal(gs, t) => PlainCircuitGate::Gal(
                        gs.clone(), 
                        map_rhs_transform(t)
                    )
                })
            ).collect(),
            output_transforms: self.output_transforms.iter().map(|t| {
                let added_inputs_t = map_self_transform(t);
                LinearCombination {
                    factors: add_nones(&added_inputs_t.factors, self.input_count + rhs.input_count + self.computed_wire_count(), rhs.computed_wire_count()),
                    constant: added_inputs_t.constant
                }
            }).chain(rhs.output_transforms.iter().map(map_rhs_transform)).collect()
        };
        result.check_invariants();
        return result;
    }

    pub fn compose<S: RingStore<Type = R> + Copy>(&self, first: &Self, ring: S) -> Self {
        assert_eq!(first.output_count(), self.input_count());

        let map_transform = |t: &LinearCombination<R>| {
            let input_transform = LinearCombination {
                constant: t.constant.as_ref().map(|c| ring.clone_el(c)),
                factors: t.factors[0..self.input_count].iter().map(|c| c.as_ref().map(|c| ring.clone_el(c))).collect()
            };
            let mut result = input_transform.compose(&first.output_transforms, ring);
            result.factors.extend(t.factors[self.input_count..].iter().map(|c| c.as_ref().map(|c| ring.clone_el(c))));
            return result;
        };
        let result = Self {
            input_count: first.input_count,
            gates: first.gates.iter().map(|gate| gate.clone(ring)).chain(
                self.gates.iter().map(|gate| match gate {
                    PlainCircuitGate::Mul(lhs, rhs) => PlainCircuitGate::Mul(
                        map_transform(lhs),
                        map_transform(rhs),
                    ),
                    PlainCircuitGate::Gal(gs, t) => PlainCircuitGate::Gal(
                        gs.clone(),
                        map_transform(t)
                    )
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

    pub fn mul_count(&self) -> usize {
        self.gates.iter().filter(|gate| match gate {
            PlainCircuitGate::Mul(_, _) => true,
            _ => false
        }).count()
    }
    
    pub fn evaluate_generic<T, ContantFn, AddProductFn, MulFn, GaloisFn>(&self, inputs: &[T], mut constant: ContantFn, mut add_prod: AddProductFn, mut mul: MulFn, mut gal: GaloisFn) -> Vec<T>
        where ContantFn: FnMut(&Option<R::Element>) -> T,
            AddProductFn: FnMut(T, &R::Element, &T) -> T,
            MulFn: FnMut(T, T) -> T,
            GaloisFn: FnMut(&[CyclotomicGaloisGroupEl], T) -> Vec<T>
    {
        assert_eq!(self.input_count, inputs.len());
        let mut current = Vec::new();
        for gate in &self.gates {
            match gate {
                PlainCircuitGate::Mul(lhs, rhs) => current.push(mul(
                    lhs.evaluate_generic(inputs, &current, &mut constant, &mut add_prod),
                    rhs.evaluate_generic(inputs, &current, &mut constant, &mut add_prod)
                )),
                PlainCircuitGate::Gal(gs, t) => current.extend(gal(
                    &gs,
                    t.evaluate_generic(inputs, &current, &mut constant, &mut add_prod)
                ))
            }
        }
        return self.output_transforms.iter().map(|t| t.evaluate_generic(inputs, &current, &mut constant, &mut add_prod)).collect()
    }

    pub fn evaluate_no_galois<S, H>(&self, inputs: &[S::Element], hom: H) -> Vec<S::Element>
        where S: ?Sized + RingBase,
            H: Homomorphism<R, S>
    {
        assert!(!self.has_galois_gates());
        self.evaluate_generic(
            inputs,
            |c| c.as_ref().map(|c| hom.map_ref(c)).unwrap_or(hom.codomain().zero()),
            |x, lhs, rhs| hom.codomain().add(x, hom.mul_ref_map(rhs, lhs)),
            |lhs, rhs| hom.codomain().mul(lhs, rhs),
            |_, _| unreachable!()
        )
    }

    pub fn evaluate<S, H>(&self, inputs: &[S::Element], hom: H) -> Vec<S::Element>
        where S: ?Sized + RingBase + CyclotomicRing,
            H: Homomorphism<R, S>
    {
        assert!(!self.has_galois_gates());
        self.evaluate_generic(
            inputs,
            |c| c.as_ref().map(|c| hom.map_ref(c)).unwrap_or(hom.codomain().zero()),
            |x, lhs, rhs| hom.codomain().add(x, hom.mul_ref_map(rhs, lhs)),
            |lhs, rhs| hom.codomain().mul(lhs, rhs),
            |gs, x: S::Element| hom.codomain().apply_galois_action_many(&x, gs)
        )
    }

    pub fn has_galois_gates(&self) -> bool {
        self.gates.iter().any(|gate| match gate {
            PlainCircuitGate::Gal(_, _) => true,
            PlainCircuitGate::Mul(_, _) => false
        })
    }

    pub fn has_multiplication_gates(&self) -> bool {
        self.gates.iter().any(|gate| match gate {
            PlainCircuitGate::Gal(_, _) => false,
            PlainCircuitGate::Mul(_, _) => true
        })
    }
    
    pub fn is_linear(&self) -> bool {
        !self.has_multiplication_gates()
    }
}

#[test]
fn test_circuit_tensor_compose() {
    let ring = StaticRing::<i64>::RING;
    let x = PlainCircuit::linear_transform(&[1], ring);
    let x_sqr = PlainCircuit::mul(ring).compose(&x.output_twice(ring), ring);
    assert!(PlainCircuit {
        input_count: 1,
        gates: vec![PlainCircuitGate::Mul(
            LinearCombination {
                constant: None,
                factors: vec![Some(1)]
            },
            LinearCombination {
                constant: None,
                factors: vec![Some(1)]
            }
        )],
        output_transforms: vec![LinearCombination {
            constant: None,
            factors: vec![None, Some(1)]
        }]
    } == x_sqr);

    let x = PlainCircuit::linear_transform(&[1], ring);
    let y = PlainCircuit::linear_transform(&[1], ring);
    let x_y_x_y = x.tensor(&y, ring).output_twice(ring);
    // z = 2 * x + 3 * y
    let x_y_z = x.tensor(&x, ring).tensor(&PlainCircuit::linear_transform(&[2, 3], ring), ring).compose(&x_y_x_y, ring);
    let xy_z = PlainCircuit::mul(ring).tensor(&x, ring).compose(&x_y_z, ring);
    // w = x * y * (2 * x + 3 * y)
    let w = PlainCircuit::mul(ring).compose(&xy_z, ring);
    for x in -5..5 {
        for y in -5..5 {
            assert_eq!(x * y * (2 * x + 3 * y), w.evaluate_no_galois(&[x, y], ring.identity()).into_iter().next().unwrap());
        }
    }

    let w_1_sqr = PlainCircuit::mul(ring).compose(&PlainCircuit::add(ring).compose(&w.tensor(&PlainCircuit::constant(1, ring), ring), ring).output_twice(ring), ring);
    for x in -5..5 {
        for y in -5..5 {
            assert_eq!(StaticRing::<i64>::RING.pow(x * y * (2 * x + 3 * y) + 1, 2), w_1_sqr.evaluate_no_galois(&[x, y], ring.identity()).into_iter().next().unwrap());
        }
    }
}

#[test]
fn test_giant_step_circuit() {
    let ring = StaticRing::<i64>::RING;
    let powers = PlainCircuit::identity(1, ring).tensor(&PlainCircuit::mul(ring), ring).tensor(&PlainCircuit::mul(ring), ring).compose(
        &PlainCircuit::mul(ring).output_times(4, ring).tensor(&PlainCircuit::identity(1, ring), ring),
        ring
    ).compose(
        &PlainCircuit::identity(1, ring).output_times(3, ring),
        ring
    );
    assert_eq!(vec![4, 16, 8], powers.evaluate_no_galois(&[2], ring.identity()));

    let permuted_baby_step_dupl_input = PlainCircuit::constant(1, ring).tensor(&PlainCircuit::identity(1, ring), ring).tensor(&powers, ring);
    assert_eq!(vec![1, 2, 4, 16, 8], permuted_baby_step_dupl_input.evaluate_no_galois(&[2, 2], ring.identity()));

    let copy_input = PlainCircuit::identity(1, ring).output_twice(ring);
    assert_eq!(vec![2, 2], copy_input.evaluate_no_galois(&[2], ring.identity()));

    let permuted_baby_steps = permuted_baby_step_dupl_input.compose(&copy_input, ring);
    assert_eq!(vec![1, 2, 4, 16, 8], permuted_baby_steps.evaluate_no_galois(&[2], ring.identity()));

    let baby_steps = PlainCircuit::select(5, &[0, 1, 2, 4, 3], ring).compose(&permuted_baby_steps, ring);
    assert_eq!(1, baby_steps.input_count());
    assert_eq!(5, baby_steps.output_count());
    assert_eq!(vec![1, 2, 4, 8, 16], baby_steps.evaluate_no_galois(&[2], ring.identity()));

    let giant_steps_before_baby_steps = PlainCircuit::constant(1, ring).tensor(&PlainCircuit::identity(1, ring), ring);
    let baby_and_giant_steps = PlainCircuit::identity(4, ring).tensor(&giant_steps_before_baby_steps, ring).compose(&baby_steps, ring);
    assert_eq!(vec![1, 2, 4, 8, 1, 16], baby_and_giant_steps.evaluate_no_galois(&[2], ring.identity()));
}