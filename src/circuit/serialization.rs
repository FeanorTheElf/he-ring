use std::marker::PhantomData;

use feanor_math::ring::*;
use feanor_math::seq::{VectorFn, VectorView};
use feanor_math::serialization::{DeserializeSeedSeq, DeserializeWithRing, SerializableElementRing, SerializableSeq, SerializeWithRing};
use serde::de::DeserializeSeed;
use serde::Serialize;

use crate::cyclotomic::{CyclotomicGaloisGroup, CyclotomicGaloisGroupEl, DeserializeSeedCyclotomicGaloisGroupEl, SerializableCyclotomicGaloisGroupEl};
use crate::serialization_helper::DeserializeSeedTuple;
use crate::{impl_deserialize_seed_for_dependent_enum, impl_deserialize_seed_for_dependent_struct};

use super::{Coefficient, LinearCombination, PlaintextCircuit, PlaintextCircuitGate};

#[derive(Serialize)]
#[serde(rename = "CoefficientData", bound = "")]
enum SerializableCoefficient<'a, R: RingStore>
    where R::Type: SerializableElementRing
{
    Integer(i32),
    Other(SerializeWithRing<'a, R>)
}

#[derive(Serialize)]
#[serde(rename = "LinearCombinationData", bound = "")]
struct SerializableLinearCombination<C: Serialize, S: Serialize> {
    constant: C,
    factors: S
}

#[derive(Serialize)]
#[serde(rename = "GateData", bound = "")]
enum SerializablePlaintextCircuitGate<L: Serialize, G: Serialize> {
    Mul((L, L)),
    Gal((G, L))
}

#[derive(Serialize)]
#[serde(rename = "PlaintextCircuitData", bound = "")]
struct SerializablePlaintextCircuitData<G: Serialize, O: Serialize> {
    input_count: usize,
    gates: G,
    output_transforms: O
}

pub struct SerializablePlaintextCircuit<'a, R: RingStore> {
    circuit: &'a PlaintextCircuit<R::Type>,
    ring: R,
    galois_group: Option<&'a CyclotomicGaloisGroup>
}

impl<'a, R: RingStore + Copy> SerializablePlaintextCircuit<'a, R>
    where R::Type: SerializableElementRing
{
    pub fn new(ring: R, galois_group: &'a CyclotomicGaloisGroup, circuit: &'a PlaintextCircuit<R::Type>) -> Self {
        Self { circuit: circuit, ring: ring, galois_group: Some(galois_group) }
    }

    pub fn new_no_galois(ring: R, circuit: &'a PlaintextCircuit<R::Type>) -> Self {
        assert!(!circuit.has_galois_gates());
        Self { circuit: circuit, ring: ring, galois_group: None }
    }
}

impl<'a, R: RingStore + Copy> Serialize for SerializablePlaintextCircuit<'a, R> 
    where R::Type: SerializableElementRing
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: serde::Serializer
    {
        fn serialize_constant<'a, R: RingStore>(c: &'a Coefficient<R::Type>, ring: R) -> SerializableCoefficient<'a, R>
            where R::Type: SerializableElementRing
        {
            match c {
                Coefficient::Integer(x) => SerializableCoefficient::Integer(*x),
                Coefficient::One => SerializableCoefficient::Integer(1),
                Coefficient::NegOne => SerializableCoefficient::Integer(-1),
                Coefficient::Zero => SerializableCoefficient::Integer(0),
                Coefficient::Other(x) => SerializableCoefficient::Other(SerializeWithRing::new(x, ring))
            }
        }
        fn serialize_lin_transform<'a, R: Copy + RingStore>(t: &'a LinearCombination<R::Type>, ring: R) -> SerializableLinearCombination<SerializableCoefficient<'a, R>, impl use<'a, R> + Serialize>
            where R::Type: SerializableElementRing,
                R: 'a
        {
            SerializableLinearCombination {
                constant: serialize_constant(&t.constant, ring),
                factors: SerializableSeq::new(t.factors.as_fn().map_fn(move |c| serialize_constant(c, ring)))
            }
        }
        SerializablePlaintextCircuitData {
            input_count: self.circuit.input_count,
            gates: SerializableSeq::new(self.circuit.gates.as_fn().map_fn(|gate| match gate {
                PlaintextCircuitGate::Mul(lhs, rhs) => SerializablePlaintextCircuitGate::Mul((serialize_lin_transform(lhs, self.ring), serialize_lin_transform(rhs, self.ring))),
                PlaintextCircuitGate::Gal(gs, val) => SerializablePlaintextCircuitGate::Gal((SerializableSeq::new(gs.as_fn().map_fn(|g| SerializableCyclotomicGaloisGroupEl::new(self.galois_group.unwrap(), *g))), serialize_lin_transform(val, self.ring)))
            })),
            output_transforms: SerializableSeq::new(self.circuit.output_transforms.as_fn().map_fn(|t| serialize_lin_transform(t, self.ring)))
        }.serialize(serializer)
    }
}

#[derive(Clone)]
struct DeserializeSeedCoefficient<R: RingStore>
    where R::Type: SerializableElementRing
{
    deserializer: DeserializeWithRing<R>
}

impl_deserialize_seed_for_dependent_enum!{
    <{'de, R}> pub enum CoefficientData<{'de, R}> using DeserializeSeedCoefficient<R> {
        Integer(i32): |_: DeserializeSeedCoefficient<R>| PhantomData,
        Other(El<R>): |d: DeserializeSeedCoefficient<R>| d.deserializer
    } where R: RingStore, R::Type: SerializableElementRing
}

#[derive(Clone)]
struct DeserializeSeedLinearCombination<R: RingStore + Copy>
    where R::Type: SerializableElementRing
{
    deserializer: DeserializeWithRing<R>
}

impl_deserialize_seed_for_dependent_struct!{
    <{'de, R}> pub struct LinearCombinationData<{'de, R}> using DeserializeSeedLinearCombination<R> {
        constant: CoefficientData<'de, R>: |d: &DeserializeSeedLinearCombination<R>| DeserializeSeedCoefficient { deserializer: d.deserializer.clone() },
        factors: Vec<CoefficientData<'de, R>>: |d: &DeserializeSeedLinearCombination<R>| DeserializeSeedSeq::new(
            std::iter::repeat(DeserializeSeedCoefficient { deserializer: d.deserializer.clone() }),
            Vec::new(),
            |mut current, next| { current.push(next); current }
        )
    } where R: RingStore + Copy, R::Type: SerializableElementRing
}

#[derive(Clone)]
struct DeserializeSeedPlaintextCircuitGate<'a, R: RingStore + Copy>
    where R::Type: SerializableElementRing
{
    galois_group: Option<&'a CyclotomicGaloisGroup>,
    deserializer: DeserializeWithRing<R>
}

impl_deserialize_seed_for_dependent_enum!{
    <{'de, 'a, R}> pub enum GateData<{'de, R}> using DeserializeSeedPlaintextCircuitGate<'a, R> {
        Mul((LinearCombinationData<'de, R>, LinearCombinationData<'de, R>)): |d: DeserializeSeedPlaintextCircuitGate<'a, R>| DeserializeSeedTuple(
            DeserializeSeedLinearCombination { deserializer: d.deserializer.clone() },
            DeserializeSeedLinearCombination { deserializer: d.deserializer.clone() }
        ),
        Gal((Vec<CyclotomicGaloisGroupEl>, LinearCombinationData<'de, R>)): |d: DeserializeSeedPlaintextCircuitGate<'a, R>| DeserializeSeedTuple(
            DeserializeSeedSeq::new(
                std::iter::repeat(DeserializeSeedCyclotomicGaloisGroupEl::new(d.galois_group.expect("cannot deserialize a circuit containing galois gates when no galois group is specified"))),
                Vec::new(),
                |mut current, next| { current.push(next); current }
            ),
            DeserializeSeedLinearCombination { deserializer: d.deserializer.clone() }
        )
    } where R: RingStore + Copy, R::Type: SerializableElementRing
}

struct DeserializeSeedPlaintextCircuitData<'a, R: RingStore + Copy>
    where R::Type: SerializableElementRing
{
    galois_group: Option<&'a CyclotomicGaloisGroup>,
    deserializer: DeserializeWithRing<R>
}

impl_deserialize_seed_for_dependent_struct!{
    <{'de, 'a, R}> pub struct PlaintextCircuitData<{'de, R}> using DeserializeSeedPlaintextCircuitData<'a, R> {
        input_count: usize: |_| PhantomData,
        gates: Vec<GateData<'de, R>>: |d: &DeserializeSeedPlaintextCircuitData<'a, R>| DeserializeSeedSeq::new(
            std::iter::repeat(DeserializeSeedPlaintextCircuitGate { deserializer: d.deserializer.clone(), galois_group: d.galois_group }),
            Vec::new(),
            |mut current, next| { current.push(next); current }
        ),
        output_transforms: Vec<LinearCombinationData<'de, R>>: |d: &DeserializeSeedPlaintextCircuitData<'a, R>| DeserializeSeedSeq::new(
            std::iter::repeat(DeserializeSeedLinearCombination { deserializer: d.deserializer.clone() }),
            Vec::new(),
            |mut current, next| { current.push(next); current }
        )
    } where R: RingStore + Copy, R::Type: SerializableElementRing
}

pub struct DeserializeSeedPlaintextCircuit<'a, R: RingStore + Copy>
    where R::Type: SerializableElementRing
{
    ring: R,
    galois_group: Option<&'a CyclotomicGaloisGroup>
}

impl<'a, R: RingStore + Copy> DeserializeSeedPlaintextCircuit<'a, R>
    where R::Type: SerializableElementRing
{
    pub fn new(ring: R, galois_group: &'a CyclotomicGaloisGroup) -> Self {
        Self { ring: ring, galois_group: Some(galois_group) }
    }

    pub fn new_no_galois(ring: R) -> Self {
        Self { ring: ring, galois_group: None }
    }
}

impl<'de, 'a, R: RingStore + Copy> DeserializeSeed<'de> for DeserializeSeedPlaintextCircuit<'a, R>
    where R::Type: SerializableElementRing
{
    type Value = PlaintextCircuit<R::Type>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
        where D: serde::Deserializer<'de>
    {
        let convert_coefficient = |c: CoefficientData<_>| match c {
            CoefficientData::Integer((x, _)) if x == 0 => Coefficient::Zero,
            CoefficientData::Integer((x, _)) if x == 1 => Coefficient::One,
            CoefficientData::Integer((x, _)) if x == -1 => Coefficient::NegOne,
            CoefficientData::Integer((x, _)) => Coefficient::Integer(x),
            CoefficientData::Other((x, _)) => Coefficient::Other(x)
        };
        let convert_transform = |t: LinearCombinationData<_>| LinearCombination {
            constant: convert_coefficient(t.constant),
            factors: t.factors.into_iter().map(convert_coefficient).collect()
        };
        let res = DeserializeSeedPlaintextCircuitData {
            deserializer: DeserializeWithRing::new(self.ring),
            galois_group: self.galois_group
        }.deserialize(deserializer)?;
        let result = PlaintextCircuit {
            gates: res.gates.into_iter().map(|gate| match gate {
                GateData::Gal(((gs, val), _)) => PlaintextCircuitGate::Gal(gs, convert_transform(val)),
                GateData::Mul(((lhs, rhs), _)) => PlaintextCircuitGate::Mul(convert_transform(lhs), convert_transform(rhs))
            }).collect(),
            input_count: res.input_count,
            output_transforms: res.output_transforms.into_iter().map(convert_transform).collect()
        };
        return Ok(result);
    }
}