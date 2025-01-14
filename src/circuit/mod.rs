use feanor_math::rings::extension::FreeAlgebraStore;
use feanor_math::{algorithms::matmul::ComputeInnerProduct, rings::zn::zn_64::Zn};
use feanor_math::homomorphism::Homomorphism;
use feanor_math::primitive_int::StaticRing;
use feanor_math::{assert_el_eq, ring::*};

use crate::cyclotomic::*;
use crate::rings::decomposition_ring::DecompositionRingBase;
use crate::rings::pow2_cyclotomic::Pow2CyclotomicNumberRing;

pub enum Coefficient<R: ?Sized + RingBase> {
    Zero, One, Integer(i32), Other(R::Element)
}

impl<R: ?Sized + RingBase> Coefficient<R> {

    pub fn clone<S: RingStore<Type = R>>(&self, ring: S) -> Self {
        match self {
            Coefficient::Zero => Coefficient::Zero,
            Coefficient::One => Coefficient::One,
            Coefficient::Integer(x) => Coefficient::Integer(*x),
            Coefficient::Other(x) => Coefficient::Other(ring.clone_el(x))
        }
    }

    fn eq<S: RingStore<Type = R> + Copy>(&self, other: &Self, ring: S) -> bool {
        ring.eq_el(&self.clone(ring).to_ring_el(ring), &other.clone(ring).to_ring_el(ring))
    }

    fn from<S: RingStore<Type = R> + Copy>(el: El<S>, ring: S) -> Self {
        if ring.is_zero(&el) {
            Coefficient::Zero
        } else if ring.is_one(&el) {
            Coefficient::One
        } else {
            Coefficient::Other(el)
        }
    }

    fn to_ring_el<S: RingStore<Type = R>>(self, ring: S) -> El<S> {
        match self {
            Coefficient::Zero => ring.zero(),
            Coefficient::One => ring.one(),
            Coefficient::Integer(x) => ring.int_hom().map(x),
            Coefficient::Other(x) => x
        }
    }

    fn add<S: RingStore<Type = R> + Copy>(self, other: Self, ring: S) -> Self {
        match (self, other) {
            (Coefficient::Zero, Coefficient::Zero) => Coefficient::Zero,
            (Coefficient::Zero, Coefficient::One) => Coefficient::One,
            (Coefficient::One, Coefficient::Zero) => Coefficient::One,
            (Coefficient::Zero, Coefficient::Integer(x)) => Coefficient::Integer(x),
            (Coefficient::Integer(x), Coefficient::Zero) => Coefficient::Integer(x),
            (Coefficient::One, Coefficient::Integer(x)) => Coefficient::Integer(x + 1),
            (Coefficient::Integer(x), Coefficient::One) => Coefficient::Integer(x + 1),
            (lhs, rhs) => Coefficient::Other(ring.add(lhs.to_ring_el(ring), rhs.to_ring_el(ring)))
        }
    }

    fn mul<S: RingStore<Type = R> + Copy>(self, other: Self, ring: S) -> Self {
        match (self, other) {
            (Coefficient::Zero, _) => Coefficient::Zero,
            (_, Coefficient::Zero) => Coefficient::Zero,
            (Coefficient::One, Coefficient::One) => Coefficient::One,
            (Coefficient::One, Coefficient::Integer(x)) => Coefficient::Integer(x),
            (Coefficient::Integer(x), Coefficient::One) => Coefficient::Integer(x),
            (lhs, rhs) => Coefficient::Other(ring.mul(lhs.to_ring_el(ring), rhs.to_ring_el(ring)))
        }
    }
}

struct LinearCombination<R: ?Sized + RingBase> {
    factors: Vec<Coefficient<R>>,
    constant: Coefficient<R>
}

impl<R: ?Sized + RingBase> LinearCombination<R> {

    fn clone<S: RingStore<Type = R> + Copy>(&self, ring: S) -> Self {
        Self {
            factors: self.factors.iter().map(|c| c.clone(ring)).collect(),
            constant: self.constant.clone(ring)
        }
    }

    fn evaluate_generic<T, ContantFn, AddProductFn>(&self, first_inputs: &[T], second_inputs: &[T], mut constant: ContantFn, mut add_prod: AddProductFn) -> T
        where ContantFn: FnMut(&Coefficient<R>) -> T,
            AddProductFn: FnMut(T, &Coefficient<R>, &T) -> T
    {
        assert_eq!(self.factors.len(), first_inputs.len() + second_inputs.len());
        let mut current = constant(&self.constant);
        for (factor, input) in self.factors.iter().zip(first_inputs.iter().chain(second_inputs.iter())) {
            current = add_prod(current, factor, input);
        }
        return current;
    }

    fn compose<S>(self, input_transforms: &[LinearCombination<R>], ring: S) -> LinearCombination<R>
        where S: RingStore<Type = R> + Copy
    {
        assert_eq!(self.factors.len(), input_transforms.len());
        if input_transforms.len() == 0 {
            return self.clone(ring);
        }
        let new_input_count = input_transforms[0].factors.len();
        assert!(input_transforms.iter().all(|t| t.factors.len() == new_input_count));
        let mut result_factors = (0..new_input_count).map(|_| Coefficient::Zero).collect::<Vec<_>>();
        let mut result_constant = self.constant.clone(ring);
        for (factor, t) in self.factors.into_iter().zip(input_transforms.iter()) {
            for i in 0..new_input_count {
                take_mut::take(&mut result_factors[i], |x| x.add(factor.clone(ring).mul(t.factors[i].clone(ring), ring), ring));
            }
            result_constant = result_constant.add(factor.mul(t.constant.clone(ring), ring), ring);
        }
        return LinearCombination {
            constant: result_constant,
            factors: result_factors
        };
    }
}

impl<R: RingBase + Default> PartialEq for LinearCombination<R> {
    
    fn eq(&self, other: &Self) -> bool {
        assert_eq!(self.factors.len(), other.factors.len());
        let ring = RingValue::<R>::default();
        return self.constant.eq(&other.constant, &ring) &&
            self.factors.iter().zip(other.factors.iter()).all(|(lhs, rhs)| lhs.eq(rhs, &ring));
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

pub struct PlaintextCircuit<R: ?Sized + RingBase> {
    input_count: usize,
    gates: Vec<PlainCircuitGate<R>>,
    output_transforms: Vec<LinearCombination<R>>
}

impl<R: RingBase + Default> PartialEq for PlaintextCircuit<R> {

    fn eq(&self, other: &Self) -> bool {
        self.input_count == other.input_count && self.gates == other.gates && self.output_transforms == other.output_transforms
    }
}

impl<R: ?Sized + RingBase> PlaintextCircuit<R> {

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

    pub fn clone<S: RingStore<Type = R> + Copy>(&self, ring: S) -> Self {
        Self {
            gates: self.gates.iter().map(|gate| gate.clone(ring)).collect(),
            input_count: self.input_count,
            output_transforms: self.output_transforms.iter().map(|t| t.clone(ring)).collect()
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

    pub fn constant_i32<S: RingStore<Type = R>>(el: i32, ring: S) -> Self {
        let result = Self {
            input_count: 0,
            gates: Vec::new(),
            output_transforms: vec![LinearCombination {
                constant: if el == 0{
                    Coefficient::Zero
                } else if el == 1 {
                    Coefficient::One
                } else {
                    Coefficient::Integer(el)
                },
                factors: Vec::new()
            }]
        };
        result.check_invariants();
        return result;
    }

    pub fn constant<S: RingStore<Type = R>>(el: El<S>, ring: S) -> Self {
        let result = Self {
            input_count: 0,
            gates: Vec::new(),
            output_transforms: vec![LinearCombination {
                constant: Coefficient::from(el, &ring),
                factors: Vec::new()
            }]
        };
        result.check_invariants();
        return result;
    }

    pub fn linear_transform<S: RingStore<Type = R>>(coeffs: &[Coefficient<R>], ring: S) -> Self {
        let result = Self {
            input_count: coeffs.len(),
            gates: Vec::new(),
            output_transforms: vec![LinearCombination {
                constant: Coefficient::Zero,
                factors: coeffs.iter().map(|c| c.clone(&ring)).collect()
            }]
        };
        result.check_invariants();
        return result;
    }

    pub fn linear_transform_ring<S: RingStore<Type = R>>(coeffs: &[El<S>], ring: S) -> Self {
        let result = Self {
            input_count: coeffs.len(),
            gates: Vec::new(),
            output_transforms: vec![LinearCombination {
                constant: Coefficient::Zero,
                factors: coeffs.iter().map(|c| Coefficient::from(ring.clone_el(c), &ring)).collect()
            }]
        };
        result.check_invariants();
        return result;
    }

    pub fn add<S: RingStore<Type = R>>(ring: S) -> Self {
        let result = Self {
            input_count: 2,
            gates: Vec::new(),
            output_transforms: vec![LinearCombination {
                constant: Coefficient::Zero,
                factors: vec![Coefficient::One, Coefficient::One]
            }]
        };
        return result;
    }

    pub fn mul<S: RingStore<Type = R>>(ring: S) -> Self {
        let result = Self {
            input_count: 2,
            gates: vec![PlainCircuitGate::Mul(
                LinearCombination {
                    constant: Coefficient::Zero,
                    factors: vec![Coefficient::One, Coefficient::Zero]
                },
                LinearCombination {
                    constant: Coefficient::Zero,
                    factors: vec![Coefficient::Zero, Coefficient::One]
                }
            )],
            output_transforms: vec![LinearCombination {
                constant: Coefficient::Zero,
                factors: vec![Coefficient::Zero, Coefficient::Zero, Coefficient::One]
            }]
        };
        result.check_invariants();
        return result;
    }

    pub fn gal<S: RingStore<Type = R>>(g: CyclotomicGaloisGroupEl, ring: S) -> Self {
        let result = Self {
            input_count: 1,
            gates: vec![PlainCircuitGate::Gal(vec![g], LinearCombination {
                constant: Coefficient::Zero,
                factors: vec![Coefficient::One]
            })],
            output_transforms: vec![LinearCombination {
                constant: Coefficient::Zero,
                factors: vec![Coefficient::Zero, Coefficient::One]
            }]
        };
        result.check_invariants();
        return result;
    }

    pub fn gal_many<S: RingStore<Type = R>>(gs: &[CyclotomicGaloisGroupEl], ring: S) -> Self {
        let result = Self {
            input_count: 1,
            gates: vec![PlainCircuitGate::Gal(
                gs.to_owned(), 
                LinearCombination {
                    constant: Coefficient::Zero,
                    factors: vec![Coefficient::One]
                }
            )],
            output_transforms: (0..gs.len()).map(|i| LinearCombination {
                constant: Coefficient::Zero,
                factors: (0..=gs.len()).map(|j| if j == i + 1 { Coefficient::One } else { Coefficient::Zero }).collect()
            }).collect()
        };
        result.check_invariants();
        return result;
    }

    pub fn output_twice<S: RingStore<Type = R> + Copy>(self, ring: S) -> Self {
        self.output_times(2, ring)
    }

    pub fn drop(wire_count: usize) -> Self {
        let result = Self {
            input_count: wire_count,
            gates: Vec::new(),
            output_transforms: Vec::new()
        };
        result.check_invariants();
        return result;
    }

    pub fn identity<S: RingStore<Type = R>>(wire_count: usize, ring: S) -> Self {
        let result = Self {
            input_count: wire_count,
            gates: Vec::new(),
            output_transforms: (0..wire_count).map(|i| LinearCombination {
                constant: Coefficient::Zero,
                factors: (0..wire_count).map(|j| if j == i { Coefficient::One } else { Coefficient::Zero }).collect()
            }).collect()
        };
        result.check_invariants();
        return result;
    }

    pub fn select<S: RingStore<Type = R>>(input_wire_count: usize, output_wires: &[usize], ring: S) -> Self {
        let result = Self {
            input_count: input_wire_count,
            gates: Vec::new(),
            output_transforms: output_wires.iter().map(|i| {
                assert!(*i < input_wire_count);
                LinearCombination {
                    constant: Coefficient::Zero,
                    factors: (0..input_wire_count).map(|j| if *i == j { Coefficient::One } else { Coefficient::Zero }).collect()
                }
            }).collect()
        };
        result.check_invariants();
        return result;
    }

    pub fn output_times<S: RingStore<Type = R> + Copy>(self, times: usize, ring: S) -> Self {
        let result = Self {
            input_count: self.input_count,
            gates: self.gates.iter().map(|gate| gate.clone(ring)).collect(),
            output_transforms: (0..times).flat_map(|_| self.output_transforms.iter()).map(|lin| lin.clone(ring)).collect()
        };
        result.check_invariants();
        return result;
    }

    pub fn tensor<S: RingStore<Type = R>>(self, rhs: Self, ring: S) -> Self {
        let add_zeros = |vec: &[Coefficient<R>], index: usize, count: usize| 
            vec[0..index].iter().map(|c| c.clone(&ring))
                .chain(std::iter::repeat_with(|| Coefficient::Zero).take(count))
                .chain(vec[index..].iter().map(|c| c.clone(&ring)))
                .collect::<Vec<_>>();

        let map_self_transform = |t: &LinearCombination<R>| LinearCombination {
            constant: t.constant.clone(&ring),
            factors: add_zeros(&t.factors, self.input_count, rhs.input_count)
        };
        let map_rhs_transform = |t: &LinearCombination<R>| LinearCombination {
            constant: t.constant.clone(&ring),
            factors: add_zeros(&add_zeros(&t.factors, rhs.input_count, self.computed_wire_count()), 0, self.input_count)
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
                assert_eq!(self.computed_wire_count() + self.input_count, t.factors.len());
                let added_inputs_t = map_self_transform(t);
                LinearCombination {
                    factors: add_zeros(&added_inputs_t.factors, self.input_count + rhs.input_count + self.computed_wire_count(), rhs.computed_wire_count()),
                    constant: added_inputs_t.constant
                }
            }).chain(rhs.output_transforms.iter().map(|t| {
                assert_eq!(rhs.computed_wire_count() + rhs.input_count, t.factors.len());
                map_rhs_transform(t)
            })).collect()
        };
        result.check_invariants();
        return result;
    }

    pub fn compose<S: RingStore<Type = R> + Copy>(self, first: Self, ring: S) -> Self {
        assert_eq!(first.output_count(), self.input_count());

        let map_transform = |t: &LinearCombination<R>| {
            let input_transform = LinearCombination {
                constant: t.constant.clone(&ring),
                factors: t.factors[0..self.input_count].iter().map(|c| c.clone(&ring)).collect()
            };
            let mut result = input_transform.compose(&first.output_transforms, ring);
            result.factors.extend(t.factors[self.input_count..].iter().map(|c| c.clone(&ring)));
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
        where ContantFn: FnMut(&Coefficient<R>) -> T,
            AddProductFn: FnMut(T, &Coefficient<R>, &T) -> T,
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
            |c| hom.map(c.clone(hom.domain()).to_ring_el(hom.domain())),
            |x, lhs, rhs| hom.codomain().add(x, hom.mul_ref_fst_map(rhs, lhs.clone(hom.domain()).to_ring_el(hom.domain()))),
            |lhs, rhs| hom.codomain().mul(lhs, rhs),
            |_, _| unreachable!()
        )
    }

    pub fn evaluate<S, H>(&self, inputs: &[S::Element], hom: H) -> Vec<S::Element>
        where S: ?Sized + RingBase + CyclotomicRing,
            H: Homomorphism<R, S>
    {
        self.evaluate_generic(
            inputs,
            |c| hom.map(c.clone(hom.domain()).to_ring_el(hom.domain())),
            |x, lhs, rhs| hom.codomain().add(x, hom.mul_ref_fst_map(rhs, lhs.clone(hom.domain()).to_ring_el(hom.domain()))),
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
    let x = PlaintextCircuit::linear_transform_ring(&[1], ring);
    let x_sqr = PlaintextCircuit::mul(ring).compose(x.output_twice(ring), ring);
    assert!(PlaintextCircuit {
        input_count: 1,
        gates: vec![PlainCircuitGate::Mul(
            LinearCombination {
                constant: Coefficient::Zero,
                factors: vec![Coefficient::One]
            },
            LinearCombination {
                constant: Coefficient::Zero,
                factors: vec![Coefficient::One]
            }
        )],
        output_transforms: vec![LinearCombination {
            constant: Coefficient::Zero,
            factors: vec![Coefficient::Zero, Coefficient::One]
        }]
    } == x_sqr);

    let x = PlaintextCircuit::identity(1, ring);
    let y = PlaintextCircuit::identity(1, ring);
    let x_y_x_y = x.clone(&ring).tensor(y, ring).output_twice(ring);
    // z = 2 * x + 3 * y
    let x_y_z = x.clone(ring).tensor(x.clone(ring), ring).tensor(PlaintextCircuit::linear_transform_ring(&[2, 3], ring), ring).compose(x_y_x_y, ring);
    let xy_z = PlaintextCircuit::mul(ring).tensor(x, ring).compose(x_y_z, ring);
    // w = x * y * (2 * x + 3 * y)
    let w = PlaintextCircuit::mul(ring).compose(xy_z, ring);
    for x in -5..5 {
        for y in -5..5 {
            assert_eq!(x * y * (2 * x + 3 * y), w.evaluate_no_galois(&[x, y], ring.identity()).into_iter().next().unwrap());
        }
    }

    let w_1_sqr = PlaintextCircuit::mul(ring).compose(PlaintextCircuit::add(ring).compose(w.tensor(PlaintextCircuit::constant(1, ring), ring), ring).output_twice(ring), ring);
    for x in -5..5 {
        for y in -5..5 {
            assert_eq!(StaticRing::<i64>::RING.pow(x * y * (2 * x + 3 * y) + 1, 2), w_1_sqr.evaluate_no_galois(&[x, y], ring.identity()).into_iter().next().unwrap());
        }
    }
}

#[test]
fn test_circuit_tensor_compose_with_galois() {
    let ring = DecompositionRingBase::new(Pow2CyclotomicNumberRing::new(16), Zn::new(17));

    let x = PlaintextCircuit::identity(1, &ring);
    let y = PlaintextCircuit::identity(1, &ring);
    let xy = PlaintextCircuit::mul(&ring).compose(x.tensor(y, &ring), &ring);
    let conj_xy = PlaintextCircuit::gal(ring.galois_group().from_representative(-1), &ring).compose(xy.clone(&ring), &ring);
    let partial_trace_xy = PlaintextCircuit::add(&ring).compose(xy.tensor(conj_xy, &ring), &ring).compose(PlaintextCircuit::identity(2, &ring).output_twice(&ring), &ring);

    for x_e in 0..8 {
        for y_e in 0..8 {
            let x = ring.pow(ring.canonical_gen(), x_e);
            let y = ring.pow(ring.canonical_gen(), y_e);
            let xy = ring.mul_ref(&x, &y);
            let conj_xy = ring.mul(ring.pow(ring.canonical_gen(), 16 - x_e), ring.pow(ring.canonical_gen(), 16 - y_e));
            assert_el_eq!(
                &ring,
                ring.add(xy, conj_xy),
                partial_trace_xy.evaluate(&[x, y], ring.identity()).into_iter().next().unwrap()
            );
        }
    }
}

#[test]
fn test_giant_step_circuit() {
    let ring = StaticRing::<i64>::RING;
    let powers = PlaintextCircuit::identity(1, ring).tensor(PlaintextCircuit::mul(ring), ring).tensor(PlaintextCircuit::mul(ring), ring).compose(
        PlaintextCircuit::mul(ring).output_times(4, ring).tensor(PlaintextCircuit::identity(1, ring), ring),
        ring
    ).compose(
        PlaintextCircuit::identity(1, ring).output_times(3, ring),
        ring
    );
    assert_eq!(vec![4, 16, 8], powers.evaluate_no_galois(&[2], ring.identity()));

    let permuted_baby_step_dupl_input = PlaintextCircuit::constant(1, ring).tensor(PlaintextCircuit::identity(1, ring), ring).tensor(powers, ring);
    assert_eq!(vec![1, 2, 4, 16, 8], permuted_baby_step_dupl_input.evaluate_no_galois(&[2, 2], ring.identity()));

    let copy_input = PlaintextCircuit::identity(1, ring).output_twice(ring);
    assert_eq!(vec![2, 2], copy_input.evaluate_no_galois(&[2], ring.identity()));

    let permuted_baby_steps = permuted_baby_step_dupl_input.compose(copy_input, ring);
    assert_eq!(vec![1, 2, 4, 16, 8], permuted_baby_steps.evaluate_no_galois(&[2], ring.identity()));

    let baby_steps = PlaintextCircuit::select(5, &[0, 1, 2, 4, 3], ring).compose(permuted_baby_steps, ring);
    assert_eq!(1, baby_steps.input_count());
    assert_eq!(5, baby_steps.output_count());
    assert_eq!(vec![1, 2, 4, 8, 16], baby_steps.evaluate_no_galois(&[2], ring.identity()));

    let giant_steps_before_baby_steps = PlaintextCircuit::constant(1, ring).tensor(PlaintextCircuit::identity(1, ring), ring);
    let baby_and_giant_steps = PlaintextCircuit::identity(4, ring).tensor(giant_steps_before_baby_steps, ring).compose(baby_steps, ring);
    assert_eq!(vec![1, 2, 4, 8, 1, 16], baby_and_giant_steps.evaluate_no_galois(&[2], ring.identity()));
}