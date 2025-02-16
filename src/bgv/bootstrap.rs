use feanor_math::ring::*;
use feanor_math::primitive_int::*;

use crate::circuit::*;
use crate::cyclotomic::*;
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

impl<NumberRing> PlaintextCircuit<NumberRingQuotientBase<NumberRing, Zn>>
    where NumberRing: HENumberRing
{
    #[instrument(skip_all)]
    pub fn evaluate_bgv_naive_modswitch_strategy<Params>(&self, 
        P: &PlaintextRing<Params>, 
        C: &CiphertextRing<Params>, 
        inputs: &[Ciphertext<Params>], 
        rk: Option<&RelinKey<Params>>, 
        gks: &[(CyclotomicGaloisGroupEl, KeySwitchKey<Params>)], 
        key_switches: &mut usize
    ) -> Vec<Ciphertext<Params>> 
        where Params: BGVParams,
            Params::CiphertextRing: BGFVCiphertextRing<NumberRing = NumberRing>
    {
        assert!(!self.has_multiplication_gates() || rk.is_some());
        let galois_group = C.galois_group();
        return self.evaluate_generic(
            inputs,
            |x| match x {
                Coefficient::Zero => Params::transparent_zero(P, C),
                x => Params::hom_add_plain(P, C, &x.clone(P).to_ring_el(P), Params::transparent_zero(P, C))
            },
            |dst, x, ct| Params::hom_add(P, C, dst, Params::hom_mul_plain(P, C, &x.clone(P).to_ring_el(P), Params::clone_ct(P, C, ct))),
            |lhs, rhs| Params::hom_mul(P, C, lhs, rhs, rk.unwrap()),
            |gs, x| Params::hom_galois_many(P, C, x, gs, gs.as_fn().map_fn(|expected_g| &gks.iter().filter(|(g, _)| galois_group.eq_el(*g, *expected_g)).next().unwrap().1))
        );
    }
}

impl PlaintextCircuit<StaticRingBase<i64>> {

    #[instrument(skip_all)]
    pub fn evaluate_bgv_naive_modswitch_strategy<Params>(&self, 
        P: &PlaintextRing<Params>, 
        C: &CiphertextRing<Params>, 
        inputs: &[Ciphertext<Params>], 
        rk: Option<&RelinKey<Params>>, 
        gks: &[(CyclotomicGaloisGroupEl, KeySwitchKey<Params>)], 
        key_switches: &mut usize
    ) -> Vec<Ciphertext<Params>> 
        where Params: BGVParams
    {
        assert!(!self.has_multiplication_gates() || rk.is_some());
        const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;
        let galois_group = C.galois_group();
        return self.evaluate_generic(
            inputs,
            |x| match x {
                Coefficient::Zero => Params::transparent_zero(P, C),
                x => Params::hom_add_plain(P, C, &P.int_hom().map(x.to_ring_el(ZZ) as i32), Params::transparent_zero(P, C))
            },
            |dst, x, ct| Params::hom_add(P, C, dst, Params::hom_mul_plain(P, C, &P.int_hom().map(x.to_ring_el(ZZ) as i32), Params::clone_ct(P, C, ct))),
            |lhs, rhs| Params::hom_mul(P, C, lhs, rhs, rk.unwrap()),
            |gs, x| Params::hom_galois_many(P, C, x, gs, gs.as_fn().map_fn(|expected_g| &gks.iter().filter(|(g, _)| galois_group.eq_el(*g, *expected_g)).next().unwrap().1))
        );
    }
}
impl DigitExtract {
    
    pub fn evaluate_bgv_naive_modswitch_strategy<'a, Params: BGVParams>(&self, 
        P_base: &PlaintextRing<Params>, 
        P: &[PlaintextRing<Params>], 
        C: &CiphertextRing<Params>, 
        Cmul: &CiphertextRing<Params>, 
        input: Ciphertext<Params>, 
        rk: &RelinKey<'a, Params>
    ) -> (Ciphertext<Params>, Ciphertext<Params>) {
        unimplemented!()
    }
}