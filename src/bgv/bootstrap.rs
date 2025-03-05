use feanor_math::algorithms::int_factor::is_prime_power;
use feanor_math::ring::*;

use crate::bgv::modswitch::DefaultModswitchStrategy;
use crate::circuit::*;
use crate::lintransform::matmul::MatmulTransform;
use crate::log_time;
use crate::digitextract::DigitExtract;

use crate::lintransform::composite;
use crate::lintransform::pow2;

use super::modswitch::*;
use super::*;

#[derive(Clone, Debug)]
pub struct ThinBootstrapParams<Params: BGVParams> {
    pub scheme_params: Params,
    pub v: usize,
    pub t: i64
}

impl<Params: BGVParams> ThinBootstrapParams<Params>
    where NumberRing<Params>: Clone
{
    pub fn build_pow2<M: BGVModswitchStrategy<Params>, const LOG: bool>(&self, modswitch_strategy: M, cache_dir: Option<&str>) -> ThinBootstrapData<Params, M> {
        let log2_n = ZZ.abs_log2_ceil(&(self.scheme_params.number_ring().n() as i64)).unwrap();
        assert_eq!(self.scheme_params.number_ring().n(), 1 << log2_n);

        let (p, r) = is_prime_power(&ZZ, &self.t).unwrap();
        let v = self.v;
        let e = r + v;
        if LOG {
            println!("Setting up bootstrapping for plaintext modulus p^r = {}^{} = {} within the cyclotomic ring Q[X]/(Phi_{})", p, r, self.t, <_ as HECyclotomicNumberRing>::n(&self.scheme_params.number_ring()));
            println!("Choosing e = r + v = {} + {}", r, v);
        }

        let plaintext_ring = self.scheme_params.create_plaintext_ring(ZZ.pow(p, e));
        let original_plaintext_ring = self.scheme_params.create_plaintext_ring(ZZ.pow(p, r));

        let digit_extract = DigitExtract::new_default(p, e, r);

        let hypercube = HypercubeStructure::halevi_shoup_hypercube(CyclotomicGaloisGroup::new(plaintext_ring.n() as u64), p);
        let H = if let Some(cache_dir) = cache_dir {
            HypercubeIsomorphism::new_cache_file::<LOG>(&plaintext_ring, hypercube, cache_dir)
        } else {
            HypercubeIsomorphism::new::<LOG>(&plaintext_ring, hypercube)
        };
        let original_H = H.change_modulus(&original_plaintext_ring);
        let slots_to_coeffs = log_time::<_, _, LOG, _>("Creating Slots-to-Coeffs transform", |[]| MatmulTransform::to_circuit_many(pow2::slots_to_coeffs_thin(&original_H), &original_H));
        let coeffs_to_slots = log_time::<_, _, LOG, _>("Creating Coeffs-to-Slots transform", |[]| pow2::coeffs_to_slots_thin(&H));
        let plaintext_ring_hierarchy = ((r + 1)..=e).map(|k| self.scheme_params.create_plaintext_ring(ZZ.pow(p, k))).collect();

        return ThinBootstrapData {
            digit_extract,
            slots_to_coeffs_thin: slots_to_coeffs,
            coeffs_to_slots_thin: coeffs_to_slots,
            plaintext_ring_hierarchy: plaintext_ring_hierarchy,
            modswitch_strategy: modswitch_strategy,
            tmp_coprime_modulus_plaintext: self.scheme_params.create_plaintext_ring(ZZ.pow(p, e) + 1)
        };
    }

    pub fn build_odd<M: BGVModswitchStrategy<Params>, const LOG: bool>(&self, modswitch_strategy: M, cache_dir: Option<&str>) -> ThinBootstrapData<Params, M> {
        assert!(self.scheme_params.number_ring().n() % 2 != 0);

        let (p, r) = is_prime_power(&ZZ, &self.t).unwrap();
        let v = self.v;
        let e = r + v;
        if LOG {
            println!("Setting up bootstrapping for plaintext modulus p^r = {}^{} = {} within the cyclotomic ring Q[X]/(Phi_{})", p, r, self.t, self.scheme_params.number_ring().n());
            println!("Choosing e = r + v = {} + {}", r, v);
        }

        let plaintext_ring = self.scheme_params.create_plaintext_ring(ZZ.pow(p, e));
        let original_plaintext_ring = self.scheme_params.create_plaintext_ring(ZZ.pow(p, r));

        let digit_extract = if p == 2 && e <= 23 {
            DigitExtract::new_precomputed_p_is_2(p, e, r)
        } else {
            DigitExtract::new_default(p, e, r)
        };

        let hypercube = HypercubeStructure::halevi_shoup_hypercube(CyclotomicGaloisGroup::new(plaintext_ring.n() as u64), p);
        let H = if let Some(cache_dir) = cache_dir {
            HypercubeIsomorphism::new_cache_file::<LOG>(&plaintext_ring, hypercube, cache_dir)
        } else {
            HypercubeIsomorphism::new::<LOG>(&plaintext_ring, hypercube)
        };
        let original_H = H.change_modulus(&original_plaintext_ring);
        let slots_to_coeffs = log_time::<_, _, LOG, _>("Creating Slots-to-Coeffs transform", |[]| MatmulTransform::to_circuit_many(composite::slots_to_powcoeffs_thin(&original_H), &original_H));
        let coeffs_to_slots = log_time::<_, _, LOG, _>("Creating Coeffs-to-Slots transform", |[]| MatmulTransform::to_circuit_many(composite::powcoeffs_to_slots_thin(&H), &H));
        let plaintext_ring_hierarchy = ((r + 1)..=e).map(|k| self.scheme_params.create_plaintext_ring(ZZ.pow(p, k))).collect();

        return ThinBootstrapData {
            digit_extract,
            slots_to_coeffs_thin: slots_to_coeffs,
            coeffs_to_slots_thin: coeffs_to_slots,
            plaintext_ring_hierarchy: plaintext_ring_hierarchy,
            modswitch_strategy: modswitch_strategy,
            tmp_coprime_modulus_plaintext: self.scheme_params.create_plaintext_ring(ZZ.pow(p, e) + 1)
        };
    }
}

pub struct ThinBootstrapData<Params: BGVParams, M: BGVModswitchStrategy<Params>> {
    modswitch_strategy: M,
    digit_extract: DigitExtract,
    slots_to_coeffs_thin: PlaintextCircuit<<PlaintextRing<Params> as RingStore>::Type>,
    coeffs_to_slots_thin: PlaintextCircuit<<PlaintextRing<Params> as RingStore>::Type>,
    plaintext_ring_hierarchy: Vec<PlaintextRing<Params>>,
    tmp_coprime_modulus_plaintext: PlaintextRing<Params>
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

    pub fn largest_plaintext_ring(&self) -> &PlaintextRing<Params> {
        self.plaintext_ring_hierarchy.last().unwrap()
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
        
        let C_input = Params::mod_switch_down_ciphertext_ring(C_master, ct_dropped_moduli);
        let sk_input = debug_sk.map(|sk| Params::mod_switch_down_sk(&C_input, &C_master, ct_dropped_moduli, sk));
        if let Some(sk) = &sk_input {
            Params::dec_println_slots(P_base, &C_input, &ct, sk);
        }

        let P_main = self.plaintext_ring_hierarchy.last().unwrap();
        debug_assert_eq!(ZZ.pow(self.p(), self.e()), *P_main.base_ring().modulus());

        let values_in_coefficients = log_time::<_, _, LOG, _>("1. Computing Slots-to-Coeffs transform", |[key_switches]| {
            let result = DefaultModswitchStrategy::never_modswitch().evaluate_circuit_plaintext(
                &self.slots_to_coeffs_thin, 
                P_base, 
                &C_input, 
                &[ModulusAwareCiphertext {
                    data: ct, 
                    info: (), 
                    dropped_rns_factor_indices: RNSFactorIndexList::empty()
                }], 
                None, 
                gks,
                key_switches,
                debug_sk
            );
            assert_eq!(1, result.len());
            let result = result.into_iter().next().unwrap();
            debug_assert_eq!(*result.dropped_rns_factor_indices, *ct_dropped_moduli);
            return result.data;
        });
        if let Some(sk) = &sk_input {
            Params::dec_println(P_base, &C_input, &values_in_coefficients, sk);
        }

        let noisy_decryption = log_time::<_, _, LOG, _>("2. Computing noisy decryption c0 + c1 * s", |[]| {
            // this is slightly more complicated than in BFV, since we cannot mod-switch to a ciphertext modulus that is not coprime to `t = p^r`.
            // Instead, we first multiply by `p^v`, then mod-switch to `p^e + 1`, and then reduce the shortest lift of the result modulo `p^e`.
            // This will introduce the overflow modulo `p^e + 1` as error in the lower bits, which we will later remove during digit extraction
            let values_scaled = Ciphertext {
                c0: C_input.inclusion().mul_map(values_in_coefficients.c0, C_input.base_ring().coerce(&ZZ, ZZ.pow(self.p(), self.v()))),
                c1: C_input.inclusion().mul_map(values_in_coefficients.c1, C_input.base_ring().coerce(&ZZ, ZZ.pow(self.p(), self.v()))),
                implicit_scale: values_in_coefficients.implicit_scale
            };
            // change to `p^e + 1`
            let (c0, c1) = Params::mod_switch_to_plaintext(P_main, &self.tmp_coprime_modulus_plaintext, &C_input, values_scaled);
            // reduce modulo `p^e`, which will introduce additional error in the lower digits
            let mod_pe = P_main.base_ring().can_hom(&ZZ).unwrap();
            let (c0, c1) = (
                P_main.from_canonical_basis(self.tmp_coprime_modulus_plaintext.wrt_canonical_basis(&c0).iter().map(|x| mod_pe.map(self.tmp_coprime_modulus_plaintext.base_ring().smallest_lift(x)))),
                P_main.from_canonical_basis(self.tmp_coprime_modulus_plaintext.wrt_canonical_basis(&c1).iter().map(|x| mod_pe.map(self.tmp_coprime_modulus_plaintext.base_ring().smallest_lift(x))))
            );

            let enc_sk = Params::enc_sk(P_main, C_master);
            return ModulusAwareCiphertext {
                data: Params::hom_add_plain(P_main, C_master, &c0, Params::hom_mul_plain(P_main, C_master, &c1, enc_sk)),
                info: self.modswitch_strategy.info_for_fresh_encryption(P_main, C_master),
                dropped_rns_factor_indices: RNSFactorIndexList::empty()
            };
        });
        if let Some(sk) = debug_sk {
            Params::dec_println(P_main, &C_master, &noisy_decryption.data, sk);
        }

        let noisy_decryption_in_slots = log_time::<_, _, LOG, _>("3. Computing Coeffs-to-Slots transform", |[key_switches]| {
            let result = self.modswitch_strategy.evaluate_circuit_plaintext(
                &self.coeffs_to_slots_thin, 
                P_main, 
                C_master, 
                &[noisy_decryption], 
                None, 
                gks,
                key_switches,
                debug_sk
            );
            assert_eq!(1, result.len());
            return result.into_iter().next().unwrap();
        });
        if let Some(sk) = debug_sk {
            Params::dec_println_slots(P_main, C_master, &noisy_decryption_in_slots.data, sk);
        }

        let final_result = log_time::<_, _, LOG, _>("4. Computing digit extraction", |[key_switches]| {

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
                rk,
                key_switches,
                debug_sk
            ).0;
        });
        return final_result;
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
        rk: &RelinKey<'a, Params>,
        key_switches: &mut usize,
        debug_sk: Option<&SecretKey<Params>>
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
                modswitch_strategy.evaluate_circuit_int(circuit, get_P(exp), C_master, inputs, Some(rk), &[], key_switches, debug_sk),
            |exp_old, exp_new, input| {
                let C_current = Params::mod_switch_down_ciphertext_ring(C_master, &input.dropped_rns_factor_indices);
                let result = ModulusAwareCiphertext {
                    data: Params::change_plaintext_modulus(get_P(exp_new), get_P(exp_old), &C_current, input.data),
                    dropped_rns_factor_indices: input.dropped_rns_factor_indices.clone(),
                    info: input.info
                };
                return result;
            }
        );
    }
}

#[test]
fn test_pow2_bgv_thin_bootstrapping_17() {
    let mut rng = thread_rng();
    
    // 8 slots of rank 16
    let params = Pow2BGV {
        log2_q_min: 790,
        log2_q_max: 800,
        log2_N: 7,
        ciphertext_allocator: DefaultCiphertextAllocator::default(),
        negacyclic_ntt: PhantomData::<DefaultNegacyclicNTT>
    };
    let t = 17;
    let digits = 6;
    let bootstrap_params = ThinBootstrapParams {
        scheme_params: params.clone(),
        v: 2,
        t: t
    };
    let bootstrapper = bootstrap_params.build_pow2::<_, true>(DefaultModswitchStrategy::<_, _, true>::new(NaiveBGVNoiseEstimator), None);
    
    let P = params.create_plaintext_ring(t);
    let C_master = params.create_initial_ciphertext_ring();
    
    let sk = Pow2BGV::gen_sk(&C_master, &mut rng);
    let gk = bootstrapper.required_galois_keys(&P).into_iter().map(|g| (g, Pow2BGV::gen_gk(bootstrapper.largest_plaintext_ring(), &C_master, &mut rng, &sk, g, digits))).collect::<Vec<_>>();
    let rk = Pow2BGV::gen_rk(bootstrapper.largest_plaintext_ring(), &C_master, &mut rng, &sk, digits);
    
    let m = P.int_hom().map(2);
    let ct = Pow2BGV::enc_sym(&P, &C_master, &mut rng, &m, &sk);
    let ct_result = bootstrapper.bootstrap_thin::<true>(
        &C_master, 
        &P, 
        &RNSFactorIndexList::empty(),
        ct, 
        &rk, 
        &gk,
        Some(&sk)
    );
    let C_result = Pow2BGV::mod_switch_down_ciphertext_ring(&C_master, &ct_result.dropped_rns_factor_indices);
    let sk_result = Pow2BGV::mod_switch_down_sk(&C_result, &C_master, &ct_result.dropped_rns_factor_indices, &sk);

    assert_el_eq!(P, P.int_hom().map(2), Pow2BGV::dec(&P, &C_result, ct_result.data, &sk_result));
}
