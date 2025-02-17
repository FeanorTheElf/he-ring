use feanor_math::ring::*;

use crate::circuit::*;
use crate::digitextract::DigitExtract;

use super::*;

#[derive(Clone, Debug)]
pub struct ThinBootstrapParams<Params: BGVParams> {
    pub scheme_params: Params,
    pub v: usize,
    pub t: i64
}

pub struct ThinBootstrapData<Params: BGVParams> {
    digit_extract: DigitExtract,
    slots_to_coeffs_thin: PlaintextCircuit<<PlaintextRing<Params> as RingStore>::Type>,
    coeffs_to_slots_thin: PlaintextCircuit<<PlaintextRing<Params> as RingStore>::Type>,
    plaintext_ring_hierarchy: Vec<PlaintextRing<Params>>
}
