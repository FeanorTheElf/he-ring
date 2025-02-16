use feanor_math::homomorphism::Homomorphism;
use feanor_math::primitive_int::*;
use feanor_math::ring::*;

use crate::circuit::PlaintextCircuit;
use crate::cyclotomic::CyclotomicGaloisGroupEl;

use super::*;

pub trait BGVModswitchStrategy<Params: BGVParams> {

    type CiphertextData;

    fn evaluate_circuit(
        &self,
        circuit: &PlaintextCircuit<StaticRingBase<i64>>,
        params: &[(Self::CiphertextData, Ciphertext<Params>)],
        P: &PlaintextRing<Params>,
        C: &CiphertextRing<Params>,
        rk: Option<&RelinKey<Params>>,
        gks: &[(CyclotomicGaloisGroupEl, KeySwitchKey<Params>)], 
        key_switches: &mut usize
    );
}