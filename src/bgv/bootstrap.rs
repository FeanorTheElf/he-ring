use feanor_math::algorithms::int_factor::is_prime_power;
use feanor_math::ring::*;

use crate::bgv::modwitch::DefaultModswitchStrategy;
use crate::circuit::*;
use crate::log_time;
use crate::digitextract::DigitExtract;

use super::modwitch::BGVModswitchStrategy;
use super::modwitch::ModulusAwareCiphertext;
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


impl<Params: BGVParams, Strategy: BGVModswitchStrategy<Params>> ThinBootstrapData<Params, Strategy> {

    fn r(&self) -> usize {
        self.digit_extract.e() - self.digit_extract.v()
    }

    fn e(&self) -> usize {
        self.digit_extract.e()
    }

    fn v(&self) -> usize {
        self.digit_extract.v()
    }

    fn p(&self) -> i64 {
        self.digit_extract.p()
    }

    pub fn required_galois_keys(&self, P: &PlaintextRing<Params>) -> Vec<CyclotomicGaloisGroupEl> {
        let mut result = Vec::new();
        result.extend(self.slots_to_coeffs_thin.required_galois_keys(&P.galois_group()).into_iter());
        result.extend(self.coeffs_to_slots_thin.required_galois_keys(&P.galois_group()).into_iter());
        result.sort_by_key(|g| P.galois_group().representative(*g));
        result.dedup_by(|g, s| P.galois_group().eq_el(*g, *s));
        return result;
    }

    pub fn bootstrap_thin<'a, const LOG: bool>(
        &self,
        C_master: &CiphertextRing<Params>, 
        P_base: &PlaintextRing<Params>,
        ct_dropped_moduli: &RNSFactorIndexList,
        ct: Ciphertext<Params>,
        rk: &RelinKey<'a, Params>,
        gks: &[(CyclotomicGaloisGroupEl, KeySwitchKey<'a, Params>)],
        debug_sk: Option<&SecretKey<Params>>
    ) -> ModulusAwareCiphertext<Params, Strategy>
        where Params: 'a
    {
        assert!(LOG || debug_sk.is_none());
        assert_eq!(ZZ.pow(self.p(), self.r()), *P_base.base_ring().modulus());
        if LOG {
            println!("Starting Bootstrapping")
        }

        let P_main = self.plaintext_ring_hierarchy.last().unwrap();
        debug_assert_eq!(ZZ.pow(self.p(), self.e()), *P_main.base_ring().modulus());

        let values_in_coefficients = log_time::<_, _, LOG, _>("1. Computing Slots-to-Coeffs transform", |[key_switches]| {
            let result = DefaultModswitchStrategy::never_modswitch().evaluate_circuit_plaintext(
                &self.slots_to_coeffs_thin, 
                P_base, 
                C_master, 
                &[ModulusAwareCiphertext {
                    data: ct, 
                    info: (), 
                    dropped_rns_factor_indices: ct_dropped_moduli.to_owned()
                }], 
                None, 
                gks
            );
            assert_eq!(1, result.len());
            let result = result.into_iter().next().unwrap();
            debug_assert_eq!(*result.dropped_rns_factor_indices, *ct_dropped_moduli);
            return result.data;
        });

        let noisy_decryption = log_time::<_, _, LOG, _>("2. Computing noisy decryption c0 + c1 * s", |[key_switches]| {
            let C_current = Params::mod_switch_down_ciphertext_ring(C_master, ct_dropped_moduli);
            // let (c0, c1) = Params::mod_switch_to_plaintext(P_main, C_current, values_in_coefficients);
            let (c0, c1) = unimplemented!();
            let enc_sk = Params::enc_sk(P_main, C_master);
            *key_switches += 1;
            return ModulusAwareCiphertext {
                data: Params::hom_add_plain(P_main, C_master, &c0, Params::hom_mul_plain(P_main, C_master, &c1, enc_sk)),
                info: self.modswitch_strategy.info_for_fresh_encryption(P_main, C_master),
                dropped_rns_factor_indices: RNSFactorIndexList::empty()
            };
        });

        let noisy_decryption_in_slots = log_time::<_, _, LOG, _>("3. Computing Coeffs-to-Slots transform", |[key_switches]| {
            let result = self.modswitch_strategy.evaluate_circuit_plaintext(
                &self.slots_to_coeffs_thin, 
                P_base, 
                C_master, 
                &[noisy_decryption], 
                None, 
                gks
            );
            assert_eq!(1, result.len());
            return result.into_iter().next().unwrap();
        });

        if LOG {
            println!("4. Performing digit extraction");
        }
        let rounding_divisor_half = P_main.base_ring().coerce(&ZZbig, ZZbig.rounded_div(ZZbig.pow(int_cast(self.p(), ZZbig, ZZ), self.v()), &ZZbig.int_hom().map(2)));
        let digit_extraction_input = ModulusAwareCiphertext {
            data: Params::hom_add_plain(P_main, C_master, &P_main.inclusion().map(rounding_divisor_half), noisy_decryption_in_slots.data),
            info: noisy_decryption_in_slots.info,
            dropped_rns_factor_indices: noisy_decryption_in_slots.dropped_rns_factor_indices
        };

        return self.digit_extract.evaluate_bgv::<Params, Strategy>(
            &self.modswitch_strategy,
            P_base,
            &self.plaintext_ring_hierarchy,
            C_master,
            digit_extraction_input,
            rk
        ).0;
    }
}

impl DigitExtract {

    pub fn evaluate_bgv<'a, Params: BGVParams, Strategy: BGVModswitchStrategy<Params>>(
        &self,
        modswitch_strategy: &Strategy, 
        P_base: &PlaintextRing<Params>, 
        P: &[PlaintextRing<Params>], 
        C_master: &CiphertextRing<Params>, 
        input: ModulusAwareCiphertext<Params, Strategy>, 
        rk: &RelinKey<'a, Params>
    ) -> (ModulusAwareCiphertext<Params, Strategy>, ModulusAwareCiphertext<Params, Strategy>) {
        let (p, actual_r) = is_prime_power(StaticRing::<i64>::RING, P_base.base_ring().modulus()).unwrap();
        assert_eq!(self.p(), p);
        assert!(actual_r >= self.r());
        for i in 0..(self.e() - self.r()) {
            assert_eq!(ZZ.pow(self.p(), actual_r + i + 1), *P[i].base_ring().modulus());
        }
        let get_P = |exp: usize| if exp == self.r() {
            P_base
        } else {
            &P[exp - self.r() - 1]
        };
        return self.evaluate_generic(
            input,
            |exp, inputs, circuit|
                modswitch_strategy.evaluate_circuit_int(circuit, get_P(exp), C_master, inputs, Some(rk), &[]),
            |exp_old, exp_new, input| ModulusAwareCiphertext {
                data: Params::change_plaintext_modulus(get_P(exp_new), get_P(exp_old), C_master, input.data),
                dropped_rns_factor_indices: input.dropped_rns_factor_indices,
                info: input.info
            }
        );
    }
}