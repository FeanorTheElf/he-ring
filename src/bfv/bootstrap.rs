
use feanor_math::algorithms::int_factor::is_prime_power;
use feanor_math::homomorphism::Homomorphism;
use feanor_math::integer::{int_cast, IntegerRingStore};
use feanor_math::primitive_int::StaticRing;
use feanor_math::ring::*;
use feanor_math::rings::poly::dense_poly::DensePolyRing;
use feanor_math::rings::zn::zn_64::ZnEl;
use feanor_math::rings::zn::ZnRingStore;
use feanor_math::seq::VectorView;
use polys::poly_to_circuit;
use rand::thread_rng;

use crate::cyclotomic::CyclotomicRingStore;
use crate::lintransform::composite;
use crate::lintransform::trace::Trace;
use crate::rnsconv;
use crate::digitextract::*;
use crate::lintransform::pow2;
use crate::digitextract::polys::digit_retain_poly;
use crate::lintransform::CompiledLinearTransform;
use crate::rings::slots::*;

use super::*;

pub struct ThinBootstrapParams<Params: BFVParams> {
    params: Params,
    r: usize,
    e: usize,
    p: i64,
    // the k-th circuit works modulo `e - k` and outputs values `yi` such that `yi = lift(x mod p) mod p^(i + 2)` for `0 <= i < v - k - 2` as well as a final `y'` with `y' = lift(x mod p)`
    digit_extract_circuits: Vec<ArithCircuit>,
    slots_to_coeffs_thin: Vec<CompiledLinearTransform<Params::NumberRing>>,
    coeffs_to_slots_thin: (Vec<CompiledLinearTransform<Params::NumberRing>>, Option<Trace>)
}

pub struct BootstrapperConfig {
    set_v: Option<usize>
}

impl BootstrapperConfig {

    pub fn set_v(mut self, v: usize) -> Self {
        self.set_v = Some(v);
        return self;
    }
}

const DEFAULT_CONFIG: BootstrapperConfig = BootstrapperConfig {
    set_v: None
};

pub struct BootstrappingDataBundle<Params: BFVParams> {
    ///
    /// Plaintext ring with modulus `p^i` for each `r < i <= e`; 
    /// Entries are stored reversed (i.e. starting with `i = e`).
    /// 
    pub plaintext_ring_hierarchy: Vec<PlaintextRing<Params>>,
    ///
    /// Data for multiplication with plaintext modulus `p^i` for each `r < i <= e`; 
    /// Entries are stored reversed (i.e. starting with `i = e`).
    /// 
    pub multiplication_rescale_hierarchy: Vec<MulConversionData>,
    ///
    /// Data for the modulus switch `q -> p^e`
    /// 
    pub mod_switch: ModSwitchData
}

impl ThinBootstrapParams<Pow2BFVParams> {

    pub fn build_pow2<const LOG: bool>(params: Pow2BFVParams, t: i64, config: BootstrapperConfig) -> Self {
        let (p, r) = is_prime_power(&ZZ, &t).unwrap();
        let s_can_norm = <_ as HENumberRing<Zn>>::inf_to_can_norm_expansion_factor(&params.number_ring());
        let v = config.set_v.unwrap_or(((s_can_norm + 1.).log2() / (p as f64).log2()).ceil() as usize);
        let e = r + v;
        if LOG {
            println!("Setting up bootstrapping for plaintext modulus p^r = {}^{} = {} within the cyclotomic ring Q[X]/(Phi_{})", p, r, t, <_ as HECyclotomicNumberRing<Zn>>::n(&params.number_ring()));
            println!("Choosing e = r + v = {} + {}", r, v);
        }

        let digit_extraction_circuits = log_time::<_, _, LOG, _>("Computing digit extraction polynomials", |[]| {
            (1..=v).rev().map(|remaining_v| {
                let poly_ring = DensePolyRing::new(Zn::new(ZZ.pow(p, remaining_v + r) as u64), "X");
                poly_to_circuit(&poly_ring, &(2..=remaining_v).chain([r + remaining_v].into_iter()).map(|j| digit_retain_poly(&poly_ring, j)).collect::<Vec<_>>())
            }).collect::<Vec<_>>()
        });

        let plaintext_ring = params.create_plaintext_ring(ZZ.pow(p, e));
        let original_plaintext_ring = params.create_plaintext_ring(ZZ.pow(p, r));

        let H = HypercubeIsomorphism::new::<LOG>(plaintext_ring.get_ring());
        let original_H = H.change_modulus(original_plaintext_ring.get_ring());
        let slots_to_coeffs = log_time::<_, _, LOG, _>("Creating Slots-to-Coeffs transform", |[]| pow2::slots_to_coeffs_thin(&original_H));
        let (coeffs_to_slots, trace) = log_time::<_, _, LOG, _>("Creating Coeffs-to-Slots transform", |[]| {
            let (transforms, trace) = pow2::coeffs_to_slots_thin(&H);
            (transforms, Some(trace))
        });
        let (compiled_coeffs_to_slots_thin, compiled_slots_to_coeffs_thin): (Vec<_>, Vec<_>) = log_time::<_, _, LOG, _>("Compiling transforms", |[]| (
            coeffs_to_slots.into_iter().map(|T| CompiledLinearTransform::compile(&H, T)).collect::<Vec<_>>(),
            slots_to_coeffs.into_iter().map(|T| CompiledLinearTransform::compile(&original_H, T)).collect::<Vec<_>>()
        ));

        return Self::new_with(
            params,
            r,
            e,
            p,
            digit_extraction_circuits,
            compiled_slots_to_coeffs_thin,
            (compiled_coeffs_to_slots_thin, trace)
        );
    }
}

impl<Params: BFVParams> ThinBootstrapParams<Params> {

    pub fn build_odd<const LOG: bool>(params: Params, t: i64, config: BootstrapperConfig) -> Self {
        assert!(params.number_ring().n() % 2 != 0);

        let (p, r) = is_prime_power(&ZZ, &t).unwrap();
        let s_can_norm = params.number_ring().inf_to_can_norm_expansion_factor();
        let v = config.set_v.unwrap_or(((s_can_norm + 1.).log2() / (p as f64).log2()).ceil() as usize);
        let e = r + v;
        if LOG {
            println!("Setting up bootstrapping for plaintext modulus p^r = {}^{} = {} within the cyclotomic ring Q[X]/(Phi_{})", p, r, t, params.number_ring().n());
            println!("Choosing e = r + v = {} + {}", r, v);
        }

        let digit_extraction_circuits = log_time::<_, _, LOG, _>("Computing digit extraction polynomials", |[]| {
            (1..=v).rev().map(|remaining_v| {
                let poly_ring = DensePolyRing::new(Zn::new(ZZ.pow(p, remaining_v + r) as u64), "X");
                poly_to_circuit(&poly_ring, &(2..=remaining_v).chain([r + remaining_v].into_iter()).map(|j| digit_retain_poly(&poly_ring, j)).collect::<Vec<_>>())
            }).collect::<Vec<_>>()
        });

        let plaintext_ring = params.create_plaintext_ring(ZZ.pow(p, e));
        let original_plaintext_ring = params.create_plaintext_ring(ZZ.pow(p, r));

        let H = HypercubeIsomorphism::new::<LOG>(plaintext_ring.get_ring());
        let original_H = H.change_modulus(original_plaintext_ring.get_ring());
        let slots_to_coeffs = log_time::<_, _, LOG, _>("Creating Slots-to-Coeffs transform", |[]| composite::slots_to_powcoeffs_thin(&original_H));
        let coeffs_to_slots = log_time::<_, _, LOG, _>("Creating Coeffs-to-Slots transform", |[]| composite::powcoeffs_to_slots_thin(&H));

        let (compiled_coeffs_to_slots_thin, compiled_slots_to_coeffs_thin): (Vec<_>, Vec<_>) = log_time::<_, _, LOG, _>("Compiling transforms", |[]| (
            coeffs_to_slots.into_iter().map(|T| CompiledLinearTransform::compile(&H, T)).collect::<Vec<_>>(),
            slots_to_coeffs.into_iter().map(|T| CompiledLinearTransform::compile(&original_H, T)).collect::<Vec<_>>()
        ));

        return Self::new_with(
            params,
            r,
            e,
            p,
            digit_extraction_circuits,
            compiled_slots_to_coeffs_thin,
            (compiled_coeffs_to_slots_thin, None)
        );
    }
}

impl<Params: BFVParams> ThinBootstrapParams<Params> {

    pub fn new_with(
        params: Params,
        r: usize,
        e: usize,
        p: i64,
        digit_extract_circuits: Vec<ArithCircuit>,
        slots_to_coeffs_thin: Vec<CompiledLinearTransform<<Params as BFVParams>::NumberRing>>,
        coeffs_to_slots_thin: (Vec<CompiledLinearTransform<<Params as BFVParams>::NumberRing>>, Option<Trace>)
    ) -> Self {
        let v = e - r;
        assert_eq!(v, digit_extract_circuits.len());
        for k in 0..v {
            assert_eq!(1, digit_extract_circuits[k].input_count());
            assert_eq!(v - k, digit_extract_circuits[k].output_count());
        }
        Self {
            params,
            r,
            e,
            p,
            digit_extract_circuits,
            slots_to_coeffs_thin,
            coeffs_to_slots_thin
        }
    }

    #[allow(unused)]
    pub fn load(params: Params, load_dir_name: &str) -> Self {
        unimplemented!()
    }

}

impl<Params: BFVParams> ThinBootstrapParams<Params> {

    pub fn create_bootstrapping_plaintext_ring_hierarchy(&self) -> Vec<PlaintextRing<Params>> {
        ((self.r + 1)..=self.e).rev().map(|k| self.params.create_plaintext_ring(ZZ.pow(self.p, k))).collect()
    }

    pub fn create_modulus_switch(&self, P_bootstrap: &[PlaintextRing<Params>], C: &CiphertextRing<Params>) -> ModSwitchData {
        let allocator = Global;
        ModSwitchData {
            scale: rnsconv::bfv_rescale::AlmostExactRescaling::new_with(
                C.base_ring().as_iter().map(|R| Zn::new(*R.modulus() as u64)).collect::<Vec<_>>(), 
                vec![ Zn::new(*P_bootstrap[0].base_ring().modulus() as u64) ], 
                C.base_ring().len(), 
                allocator.clone()
            )
        }
    }

    pub fn create_multiplication_rescale_hierarchy(&self, P_bootstrap: &[PlaintextRing<Params>], C: &CiphertextRing<Params>, C_mul: &CiphertextRing<Params>) -> Vec<MulConversionData> {
        ((self.r + 1)..=self.e).rev().enumerate().map(|(i, _k)| Params::create_multiplication_rescale(&P_bootstrap[i], C, C_mul)).collect()
    }

    pub fn create_all_bootstrapping_data(&self, C: &CiphertextRing<Params>, C_mul: &CiphertextRing<Params>) -> BootstrappingDataBundle<Params> {
        let plaintext_rings = self.create_bootstrapping_plaintext_ring_hierarchy();
        BootstrappingDataBundle {
            mod_switch: self.create_modulus_switch(&plaintext_rings, C),
            multiplication_rescale_hierarchy: self.create_multiplication_rescale_hierarchy(&plaintext_rings, C, C_mul),
            plaintext_ring_hierarchy: plaintext_rings
        }
    }

    pub fn required_galois_keys(&self, P: &PlaintextRing<Params>) -> Vec<ZnEl> {
        let mut result = Vec::new();
        result.extend(self.slots_to_coeffs_thin.iter().flat_map(|T| T.required_galois_keys().map(|g| *g)));
        result.extend(self.coeffs_to_slots_thin.0.iter().flat_map(|T| T.required_galois_keys().map(|g| *g)));
        if let Some(trace) = &self.coeffs_to_slots_thin.1 {
            result.extend(trace.required_galois_keys());
        }
        result.sort_by_key(|g| P.get_ring().cyclotomic_index_ring().smallest_positive_lift(*g));
        result.dedup_by(|g, s| P.get_ring().cyclotomic_index_ring().eq_el(g, s));
        return result;
    }

    pub fn bootstrap_thin<const LOG: bool>(
        &self,
        C: &CiphertextRing<Params>, 
        C_mul: &CiphertextRing<Params>, 
        P_base: &PlaintextRing<Params>,
        P_bootstrap: &[PlaintextRing<Params>],
        mul_rescale_bootstrap: &[MulConversionData],
        mod_switch: &ModSwitchData,
        ct: Ciphertext<Params>,
        rk: &RelinKey<Params>,
        gk: &[(ZnEl, KeySwitchKey<Params>)],
        debug_sk: Option<&SecretKey<Params>>
    ) -> Ciphertext<Params> {
        assert_eq!(ZZ.pow(self.p, self.r), *P_base.base_ring().modulus());
        if LOG {
            println!("Starting Bootstrapping")
        }
        if let Some(sk) = debug_sk {
            Params::dec_println_slots(P_base, C, &ct, sk);
        }

        let P_main = &P_bootstrap[0];
        let rounding_divisor_half = P_main.base_ring().coerce(&ZZbig, ZZbig.rounded_div(ZZbig.pow(int_cast(self.p, ZZbig, ZZ), self.e - self.r), &ZZbig.int_hom().map(2)));

        let values_in_coefficients = log_time::<_, _, LOG, _>("1. Computing Slots-to-Coeffs transform", |[key_switches]| {
            return hom_compute_linear_transform::<Params>(P_base, C, ct, &self.slots_to_coeffs_thin, gk, key_switches);
        });
        if let Some(sk) = debug_sk {
            Params::dec_println(P_base, C, &values_in_coefficients, sk);
        }

        let noisy_decryption = log_time::<_, _, LOG, _>("2. Computing noisy decryption c0 + c1 * s", |[key_switches]| {
            let (c0, c1) = Params::mod_switch_to_plaintext(P_main, C, values_in_coefficients, mod_switch);
            let enc_sk = Params::enc_sk(P_main, C);
            *key_switches += 1;
            return Params::hom_add_plain(P_main, C, &c0, Params::hom_mul_plain(P_main, C, &c1, enc_sk));
        });
        if let Some(sk) = debug_sk {
            Params::dec_println(P_main, C, &noisy_decryption, sk);
        }

        let noisy_decryption_in_slots = log_time::<_, _, LOG, _>("3. Computing Coeffs-to-Slots transform", |[key_switches]| {
            let moved_to_slots = hom_compute_linear_transform::<Params>(P_main, C, noisy_decryption, &self.coeffs_to_slots_thin.0, gk, key_switches);
            if let Some(trace) = &self.coeffs_to_slots_thin.1 {
                return hom_compute_trace::<Params>(P_main, C, moved_to_slots, trace, gk, key_switches);
            } else {
                return moved_to_slots;
            };
        });
        if let Some(sk) = debug_sk {
            Params::dec_println_slots(P_main, C, &noisy_decryption_in_slots, sk);
        }

        if LOG {
            println!("4. Performing digit extraction");
        }
        let digit_extraction_input = Params::hom_add_plain(P_main, C, &P_main.inclusion().map(rounding_divisor_half), noisy_decryption_in_slots);
        let mut result = Params::clone_ct(C, &digit_extraction_input);
        let result_ref = &mut result;
        let mut digit_extracted: Vec<Vec<Ciphertext<Params>>> = Vec::new();
        for (i, k) in ((self.r + 1)..(self.e + 1)).rev().enumerate() {
            let P_current =  &P_bootstrap[i];
            let current_mul_rescale = &mul_rescale_bootstrap[i];
            log_time::<_, _, LOG, _>(format!("Extracting {}-th digit", i).as_str(), |[key_switches]| {
                let mut current_ct = Params::clone_ct(C, &digit_extraction_input);
                assert_eq!(i, digit_extracted.len());
                for j in 0..i {
                    current_ct = Params::hom_sub(C, current_ct, &digit_extracted[j][i - j - 1]);
                }
    
                let lowest_digit = hom_evaluate_circuit::<Params>(P_current, C, C_mul, &current_ct, &self.digit_extract_circuits[i], rk, current_mul_rescale, key_switches);
                let mut lowest_digit = lowest_digit.collect::<Vec<_>>();
                assert_eq!(k - self.r, lowest_digit.len());
    
                *result_ref = Params::hom_sub(C, Params::clone_ct(C, result_ref), &lowest_digit.pop().unwrap());
                digit_extracted.push(lowest_digit);
    
            });
            if let Some(sk) = debug_sk {
                Params::dec_println_slots(P_current, C, result_ref, sk);
            }
        }

        return result;
    }
}

fn hom_compute_linear_transform<'a, Params: BFVParams>(
    P: &PlaintextRing<Params>, 
    C: &CiphertextRing<Params>, 
    input: Ciphertext<Params>, 
    transform: &[CompiledLinearTransform<<Params as BFVParams>::NumberRing>], 
    gk: &[(ZnEl, KeySwitchKey<'a, Params>)], 
    key_switches: &mut usize
) -> Ciphertext<Params> {
    let Gal = P.get_ring().cyclotomic_index_ring();
    let get_gk = |g: &ZnEl| &gk.iter().filter(|(s, _)| Gal.eq_el(g, s)).next().unwrap().1;

    return transform.iter().fold(input, |current, T| T.evaluate_generic(
        current,
        |lhs, rhs, factor| {
            *lhs = Params::hom_add(C, Params::hom_mul_plain(P, C, factor, Params::clone_ct(C, rhs)), lhs)
        }, 
        |value, gs| {
            *key_switches += gs.len();
            Params::hom_galois_many(C, value, gs, gs.as_fn().map_fn(|g| get_gk(g)))
        },
        || Params::transparent_zero(C)
    ));
}

fn hom_compute_trace<Params: BFVParams>(P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, input: Ciphertext<Params>, trace: &Trace, gk: &[(ZnEl, KeySwitchKey<Params>)], key_switches: &mut usize) -> Ciphertext<Params> {
    let Gal = P.get_ring().cyclotomic_index_ring();
    let get_gk = |g: &ZnEl| &gk.iter().filter(|(s, _)| Gal.eq_el(g, s)).next().unwrap().1;
    return trace.evaluate_generic(input, |x, y| Params::hom_add(C, x, y), |x, g| {
        if !Gal.is_one(&g) {
            *key_switches += 1;
            Params::hom_galois(C, Params::clone_ct(C, x), g, get_gk(&g))
        } else {
            Params::clone_ct(C, x)
        }
    }, |x| Params::clone_ct(C, x));
}

fn hom_evaluate_circuit<'a, 'b, Params: BFVParams>(
    P: &'a PlaintextRing<Params>, 
    C: &'a CiphertextRing<Params>, 
    C_mul: &'a CiphertextRing<Params>, 
    input: &'a Ciphertext<Params>, 
    circuit: &'a ArithCircuit, 
    rk: &'a RelinKey<'b, Params>, 
    mul_rescale: &'a MulConversionData, 
    key_switches: &'a mut usize
) -> impl ExactSizeIterator<Item = Ciphertext<Params>> + use<'a, 'b, Params> {
    return circuit.evaluate_generic(
        std::slice::from_ref(input), 
        |lhs, rhs, factor| {
            let result = Params::hom_add(C, Params::hom_mul_plain_i64(P, C, factor, Params::clone_ct(C, rhs)), &lhs);
            return result;
        }, 
        |lhs, rhs| {
            *key_switches += 1;
            let result =  Params::hom_mul(C, C_mul, lhs, rhs, rk, mul_rescale);
            return result;
        }, 
        move |x| {
            Params::hom_add_plain(P, C, &P.inclusion().compose(P.base_ring().can_hom(&ZZ).unwrap()).map(x), Params::transparent_zero(C))
        }
    );
}

#[test]
fn test_pow2_bfv_thin_bootstrapping_17() {
    let mut rng = thread_rng();
    
    // 8 slots of rank 16
    let params = Pow2BFVParams {
        log2_q_min: 790,
        log2_q_max: 800,
        log2_N: 7
    };
    let t = 17;
    let digits = 3;
    let bootstrapper = ThinBootstrapParams::build_pow2::<true>(params.clone(), t, DEFAULT_CONFIG);
    
    let P = params.create_plaintext_ring(t);
    let (C, C_mul) = params.create_ciphertext_rings();

    let bootstrapping_data = bootstrapper.create_all_bootstrapping_data(&C, &C_mul);
    
    let sk = Pow2BFVParams::gen_sk(&C, &mut rng);
    let gk = bootstrapper.required_galois_keys(&P).into_iter().map(|g| (g, Pow2BFVParams::gen_gk::<_, false>(&C, &mut rng, &sk, g, digits))).collect::<Vec<_>>();
    let rk = Pow2BFVParams::gen_rk::<_, false>(&C, &mut rng, &sk, digits);
    
    let m = P.int_hom().map(2);
    let ct = Pow2BFVParams::enc_sym(&P, &C, &mut rng, &m, &sk);
    let res_ct = bootstrapper.bootstrap_thin::<true>(
        &C, 
        &C_mul, 
        &P, 
        &bootstrapping_data.plaintext_ring_hierarchy, 
        &bootstrapping_data.multiplication_rescale_hierarchy, 
        &bootstrapping_data.mod_switch, 
        ct, 
        &rk, 
        &gk,
        None
    );

    assert_el_eq!(P, P.int_hom().map(2), Pow2BFVParams::dec(&P, &C, res_ct, &sk));
}

#[test]
fn test_pow2_bfv_thin_bootstrapping_23() {
    let mut rng = thread_rng();
    
    // 4 slots of rank 32
    let params = Pow2BFVParams {
        log2_q_min: 790,
        log2_q_max: 800,
        log2_N: 7
    };
    let t = 23;
    let digits = 3;
    let bootstrapper = ThinBootstrapParams::build_pow2::<true>(params.clone(), t, DEFAULT_CONFIG);
    
    let P = params.create_plaintext_ring(t);
    let (C, C_mul) = params.create_ciphertext_rings();

    let bootstrapping_data = bootstrapper.create_all_bootstrapping_data(&C, &C_mul);
    
    let sk = Pow2BFVParams::gen_sk(&C, &mut rng);
    let gk = bootstrapper.required_galois_keys(&P).into_iter().map(|g| (g, Pow2BFVParams::gen_gk::<_, false>(&C, &mut rng, &sk, g, digits))).collect::<Vec<_>>();
    let rk = Pow2BFVParams::gen_rk::<_, false>(&C, &mut rng, &sk, digits);
    
    let m = P.int_hom().map(2);
    let ct = Pow2BFVParams::enc_sym(&P, &C, &mut rng, &m, &sk);
    let res_ct = bootstrapper.bootstrap_thin::<true>(
        &C, 
        &C_mul, 
        &P, 
        &bootstrapping_data.plaintext_ring_hierarchy, 
        &bootstrapping_data.multiplication_rescale_hierarchy, 
        &bootstrapping_data.mod_switch, 
        ct, 
        &rk, 
        &gk,
        None
    );

    assert_el_eq!(P, P.int_hom().map(2), Pow2BFVParams::dec(&P, &C, res_ct, &sk));
}

#[test]
fn test_composite_bfv_thin_bootstrapping_2() {
    let mut rng = thread_rng();
    
    let params = CompositeBFVParams {
        log2_q_min: 750,
        log2_q_max: 800,
        n1: 31,
        n2: 11
    };
    let t = 8;
    let digits = 3;
    let bootstrapper = ThinBootstrapParams::build_odd::<true>(params.clone(), t, DEFAULT_CONFIG.set_v(11));
    
    let P = params.create_plaintext_ring(t);
    let (C, C_mul) = params.create_ciphertext_rings();

    let bootstrapping_data = bootstrapper.create_all_bootstrapping_data(&C, &C_mul);
    
    let sk = CompositeBFVParams::gen_sk(&C, &mut rng);
    let gk = bootstrapper.required_galois_keys(&P).into_iter().map(|g| (g, CompositeBFVParams::gen_gk::<_, false>(&C, &mut rng, &sk, g, digits))).collect::<Vec<_>>();
    let rk = CompositeBFVParams::gen_rk::<_, false>(&C, &mut rng, &sk, digits);
    
    let m = P.int_hom().map(2);
    let ct = CompositeBFVParams::enc_sym(&P, &C, &mut rng, &m, &sk);
    let res_ct = bootstrapper.bootstrap_thin::<true>(
        &C, 
        &C_mul, 
        &P, 
        &bootstrapping_data.plaintext_ring_hierarchy, 
        &bootstrapping_data.multiplication_rescale_hierarchy, 
        &bootstrapping_data.mod_switch, 
        ct, 
        &rk, 
        &gk,
        None
    );

    assert_el_eq!(P, P.int_hom().map(2), CompositeBFVParams::dec(&P, &C, res_ct, &sk));
}
