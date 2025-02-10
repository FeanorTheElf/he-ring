use std::marker::PhantomData;

use feanor_math::serialization::DeserializeSeedSeq;
use serde::Serialize;

use crate::cyclotomic::{CyclotomicGaloisGroup, DeserializeSeedCyclotomicGaloisGroupEl, SerializableCyclotomicGaloisGroupEl};
use crate::impl_deserialize_seed_for_dependent_struct;

use super::*;

#[derive(Serialize)]
#[serde(rename = "HypercubeStructureData")]
pub struct SerializableHypercubeStructureData<'a> {
    p: SerializableCyclotomicGaloisGroupEl<'a>,
    d: usize,
    ms: &'a [usize],
    gs: Vec<SerializableCyclotomicGaloisGroupEl<'a>>,
    choice: &'a HypercubeTypeData
}

impl<'a> SerializableHypercubeStructureData<'a> {

    pub fn new(hypercube_structure: &'a HypercubeStructure) -> Self {
        Self {
            p: SerializableCyclotomicGaloisGroupEl::new(&hypercube_structure.galois_group, hypercube_structure.p),
            d: hypercube_structure.d,
            ms: &hypercube_structure.ms,
            gs: hypercube_structure.gs.iter().map(|g| SerializableCyclotomicGaloisGroupEl::new(&hypercube_structure.galois_group, *g)).collect(),
            choice: &hypercube_structure.choice
        }
    }
}

pub struct HypercubeStructureDataDeserializer {
    galois_group: CyclotomicGaloisGroup
}

impl<'a> HypercubeStructureDataDeserializer {
    pub fn new(galois_group: CyclotomicGaloisGroup) -> Self {
        Self { galois_group }
    }
}

fn derive_single_galois_group_deserializer<'a>(deserializer: &'a HypercubeStructureDataDeserializer) -> DeserializeSeedCyclotomicGaloisGroupEl<'a> {
    DeserializeSeedCyclotomicGaloisGroupEl::new(&deserializer.galois_group)
}

fn derive_multiple_galois_group_deserializer<'de, 'a>(deserializer: &'a HypercubeStructureDataDeserializer) -> impl use<'a, 'de> + DeserializeSeed<'de, Value = Vec<CyclotomicGaloisGroupEl>> {
    DeserializeSeedSeq::new(
        std::iter::repeat(DeserializeSeedCyclotomicGaloisGroupEl::new(&deserializer.galois_group)),
        Vec::new(),
        |mut current, next| { current.push(next); current }
    )
}

impl_deserialize_seed_for_dependent_struct!{
    struct HypercubeStructureData<'de> using HypercubeStructureDataDeserializer {
        p: CyclotomicGaloisGroupEl: derive_single_galois_group_deserializer,
        d: usize: |_| PhantomData,
        ms: Vec<usize>: |_| PhantomData,
        gs: Vec<CyclotomicGaloisGroupEl>: derive_multiple_galois_group_deserializer,
        choice: HypercubeTypeData: |_| PhantomData
    }
}