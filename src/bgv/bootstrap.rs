use feanor_math::ring::*;

use crate::circuit::*;
use crate::digitextract::DigitExtract;

use super::modwitch::BGVModswitchStrategy;
use super::*;

#[derive(Clone, Debug)]
pub struct ThinBootstrapParams<Params: BGVParams> {
    pub scheme_params: Params,
    pub v: usize,
    pub t: i64
}

pub struct ThinBootstrapData<Params: BGVParams, M: BGVModswitchStrategy<Params>> {
    modswitch_strategy: M,
    digit_extract: DigitExtract,
    slots_to_coeffs_thin: PlaintextCircuit<<PlaintextRing<Params> as RingStore>::Type>,
    coeffs_to_slots_thin: PlaintextCircuit<<PlaintextRing<Params> as RingStore>::Type>,
    plaintext_ring_hierarchy: Vec<PlaintextRing<Params>>
}

impl DigitExtract {

    pub fn evaluate_bgv<'a, Params: BGVParams, M: BGVModswitchStrategy<Params>>(
        &self,
        modswitch_strategy: &M, 
        P_base: &PlaintextRing<Params>, 
        P: &[PlaintextRing<Params>], 
        C: &CiphertextRing<Params>, 
        input: (M::CiphertextInfo, Ciphertext<Params>), 
        rk: &RelinKey<'a, Params>
    ) -> ((M::CiphertextInfo, Ciphertext<Params>), (M::CiphertextInfo, Ciphertext<Params>)) {
        unimplemented!()
    }
}