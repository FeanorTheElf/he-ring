use std::alloc::Global;

use feanor_math::algorithms::int_factor::is_prime_power;
use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::homomorphism::Homomorphism;
use feanor_math::integer::{int_cast, IntegerRingStore};
use feanor_math::primitive_int::StaticRing;
use feanor_math::ring::*;
use feanor_math::rings::extension::FreeAlgebraStore;
use feanor_math::rings::poly::dense_poly::DensePolyRing;
use feanor_math::rings::zn::zn_64::{Zn, ZnEl};
use feanor_math::rings::zn::ZnRingStore;
use feanor_math::seq::VectorView;
use polys::poly_to_circuit;
use rand::thread_rng;

use crate::complexfft::automorphism::HypercubeIsomorphism;
use crate::lintransform::pow2::pow2_coeffs_to_slots_thin;
use crate::rnsconv;
use crate::{digitextract::*, lintransform::pow2::pow2_slots_to_coeffs_thin};
use crate::digitextract::polys::digit_retain_poly;
use crate::lintransform::CompiledLinearTransform;

use super::*;

pub struct Pow2Trace {
    galois_elements: Vec<ZnEl>,
    trace_rank_quo: i64
}

impl Pow2Trace {

    pub fn slotwise_trace(Gal: &Zn, p: i64, slot_rank: usize) -> Pow2Trace {
        let log2_slot_rank = ZZ.abs_log2_ceil(&(slot_rank as i64)).unwrap();
        assert_eq!(slot_rank, 1 << log2_slot_rank);
        Pow2Trace { 
            galois_elements: (0..log2_slot_rank).map(|i| Gal.pow(Gal.coerce(&ZZ, p), 1 << i)).collect(),
            trace_rank_quo: slot_rank as i64
        }
    }

    pub fn evaluate_generic<T, Add, ApplyGalois>(&self, input: T, mut add_fn: Add, mut apply_galois_fn: ApplyGalois) -> T
        where Add: FnMut(T, &T) -> T,
            ApplyGalois: FnMut(&T, &ZnEl) -> T
    {
        self.galois_elements.iter().fold(input, |current, g| {
            let conjugate = apply_galois_fn(&current, g);
            add_fn(conjugate, &current)
        })
    }

    pub fn required_galois_keys<'a>(&'a self) -> impl 'a + Iterator<Item = &'a ZnEl> {
        self.galois_elements.iter()
    }
}

pub struct Pow2BFVThinBootstrapParams {
    params: Pow2BFVParams,
    r: usize,
    e: usize,
    p: i64,
    slotwise_trace: Pow2Trace,
    // the k-th circuit works modulo `e - k` and outputs values `yi` such that `yi = lift(x mod p) mod p^(i + 2)` for `0 <= i < v - k - 2` as well as a final `y'` with `y' = lift(x mod p)`
    digit_extract_circuits: Vec<ArithCircuit>,
    slots_to_coeffs_thin: Vec<CompiledLinearTransform<PlaintextZn, crate::complexfft::pow2_cyclotomic::Pow2CyclotomicFFT<PlaintextZn, PlaintextFFT>, Global>>,
    coeffs_to_slots_thin: Vec<CompiledLinearTransform<PlaintextZn, crate::complexfft::pow2_cyclotomic::Pow2CyclotomicFFT<PlaintextZn, PlaintextFFT>, Global>>
}

pub struct BootstrappingDataBundle {
    pub plaintext_ring_hierarchy: Vec<PlaintextRing>,
    pub multiplication_rescale_hierarchy: Vec<MulConversionData>,
    pub mod_switch: ModSwitchData
}

impl Pow2BFVThinBootstrapParams {

    pub fn create_for(params: Pow2BFVParams, load_hypercube: Option<&str>) -> Self {
        let (p, r) = is_prime_power(&ZZ, &params.t).unwrap();
        // this is the case if s is uniformly ternary
        let s_can_norm = 1 << params.log2_N;
        let v = ((s_can_norm as f64 + 1.).log2() / (p as f64).log2()).ceil() as usize;
        let e = r + v;
        println!("e = {}", e);

        println!("preparing digit extraction polys...");
        let start = Instant::now();
        let digit_extraction_circuits = (1..=v).rev().map(|remaining_v| {
            let poly_ring = DensePolyRing::new(PlaintextZn::new(ZZ.pow(p, remaining_v + r) as u64), "X");
            poly_to_circuit(&poly_ring, &(2..=remaining_v).chain([r + remaining_v].into_iter()).map(|j| digit_retain_poly(&poly_ring, j)).collect::<Vec<_>>())
        }).collect::<Vec<_>>();
        let end = Instant::now();
        println!("done in {} ms", (end - start).as_millis());

        let plaintext_ring = <PlaintextRing as RingStore>::Type::new(PlaintextZn::new(ZZ.pow(p, e) as u64), params.log2_N);
        let original_plaintext_ring = params.create_plaintext_ring();

        let H = if let Some(filename) = load_hypercube {
            HypercubeIsomorphism::load(filename, plaintext_ring.get_ring())
        } else {
            let H = HypercubeIsomorphism::new(plaintext_ring.get_ring());
            H.save("F:\\Users\\Simon\\Documents\\Projekte\\he-ring\\hypercube_bootstrap.json");
            H
        };
        let original_H = H.reduce_modulus(original_plaintext_ring.get_ring());
        original_H.save("F:\\Users\\Simon\\Documents\\Projekte\\he-ring\\hypercube_base.json");

        println!("computing slots-to-coeffs transforms...");
        let start = Instant::now();
        let slots_to_coeffs = pow2_slots_to_coeffs_thin(&original_H);
        let end = Instant::now();
        println!("done in {} ms", (end - start).as_millis());

        println!("computing coeffs-to-slots transforms...");
        let start = Instant::now();
        let coeffs_to_slots = pow2_coeffs_to_slots_thin(&H);
        let end = Instant::now();
        println!("done in {} ms", (end - start).as_millis());

        println!("compiling linear transforms...");
        let start = Instant::now();
        let compiled_coeffs_to_slots_thin = coeffs_to_slots.into_iter().map(|T| CompiledLinearTransform::compile(&H, T)).collect();
        let compiled_slots_to_coeffs_thin = slots_to_coeffs.into_iter().map(|T| CompiledLinearTransform::compile(&original_H, T)).collect();
        let end = Instant::now();
        println!("done in {} ms", (end - start).as_millis());

        println!("done");
        return Self {
            params: params,
            e: e,
            r: r,
            p: p,
            digit_extract_circuits: digit_extraction_circuits,
            slots_to_coeffs_thin: compiled_slots_to_coeffs_thin,
            coeffs_to_slots_thin: compiled_coeffs_to_slots_thin,
            slotwise_trace: Pow2Trace::slotwise_trace(H.galois_group_mulrepr(), p, H.slot_ring().rank())
        };
    }

    pub fn create_bootstrapping_plaintext_ring_hierarchy(&self) -> Vec<PlaintextRing> {
        ((self.r + 1)..=self.e).rev().map(|k| <PlaintextRing as RingStore>::Type::new(PlaintextZn::new(ZZ.pow(self.p, k) as u64), self.params.log2_N)).collect()
    }

    pub fn create_modulus_switch(&self, P_bootstrap: &[PlaintextRing], C: &CiphertextRing) -> ModSwitchData {
        let allocator = C.get_ring().allocator().clone();
        ModSwitchData {
            scale: rnsconv::bfv_rescale::AlmostExactRescaling::new_with(
                C.get_ring().rns_base().as_iter().map(|R| CiphertextZn::new(*R.modulus() as u64)).collect::<Vec<_>>(), 
                vec![ PlaintextZn::new(*P_bootstrap[0].base_ring().modulus() as u64) ], 
                C.get_ring().rns_base().len(), 
                allocator.clone()
            )
        }
    }

    pub fn create_multiplication_rescale_hierarchy(&self, C: &CiphertextRing, C_mul: &CiphertextRing) -> Vec<MulConversionData> {
        let allocator = C.get_ring().allocator().clone();
        ((self.r + 1)..=self.e).rev().map(|k| MulConversionData {
            lift_to_C_mul: rnsconv::shared_lift::AlmostExactSharedBaseConversion::new_with(
                C.get_ring().rns_base().as_iter().map(|R| CiphertextZn::new(*R.modulus() as u64)).collect::<Vec<_>>(), 
                Vec::new(),
                C_mul.get_ring().rns_base().as_iter().skip(C.get_ring().rns_base().len()).map(|R| CiphertextZn::new(*R.modulus() as u64)).collect::<Vec<_>>(),
                allocator.clone()
            ),
            scale_down_to_C: rnsconv::bfv_rescale::AlmostExactRescalingConvert::new_with(
                C_mul.get_ring().rns_base().as_iter().map(|R| CiphertextZn::new(*R.modulus() as u64)).collect::<Vec<_>>(), 
                vec![ CiphertextZn::new(ZZ.pow(self.p, k) as u64) ], 
                C.get_ring().rns_base().len(),
                allocator.clone()
            )
        }).collect()
    }

    pub fn create_all_bootstrapping_data(&self, C: &CiphertextRing, C_mul: &CiphertextRing) -> BootstrappingDataBundle {
        let plaintext_rings = self.create_bootstrapping_plaintext_ring_hierarchy();
        BootstrappingDataBundle {
            mod_switch: self.create_modulus_switch(&plaintext_rings, C),
            multiplication_rescale_hierarchy: self.create_multiplication_rescale_hierarchy(C, C_mul),
            plaintext_ring_hierarchy: plaintext_rings
        }
    }

    pub fn required_galois_keys(&self, P: &PlaintextRing) -> Vec<ZnEl> {
        let mut result = Vec::new();
        result.extend(self.coeffs_to_slots_thin.iter().flat_map(|T| T.required_galois_keys().map(|g| *g)));
        result.extend(self.slots_to_coeffs_thin.iter().flat_map(|T| T.required_galois_keys().map(|g| *g)));
        result.extend(self.slotwise_trace.required_galois_keys());
        result.sort_by_key(|g| P.get_ring().galois_group_mulrepr().smallest_positive_lift(*g));
        result.dedup_by(|g, s| P.get_ring().galois_group_mulrepr().eq_el(g, s));
        return result;
    }

    pub fn bootstrap_thin(
        &self,
        C: &CiphertextRing, 
        C_mul: &CiphertextRing, 
        P: &PlaintextRing,
        P_bootstrap: &[PlaintextRing],
        mul_rescale_bootstrap: &[MulConversionData],
        mod_switch: &ModSwitchData,
        ct: Ciphertext,
        rk: &RelinKey,
        gk: &[(ZnEl, KeySwitchKey)]
    ) -> Ciphertext {
        let Gal = P.get_ring().galois_group_mulrepr();
        let P_main = &P_bootstrap[0];
        let delta_bootstrap = C.base_ring().coerce(&ZZbig, ZZbig.rounded_div(
            ZZbig.clone_el(C.base_ring().modulus()), 
            &int_cast(*P_main.base_ring().modulus() as i32, &ZZbig, &StaticRing::<i32>::RING)
        ));
        let rounding_divisor_half = P_main.base_ring().coerce(&ZZbig, ZZbig.rounded_div(ZZbig.pow(int_cast(self.p, ZZbig, ZZ), self.e - self.r), &ZZbig.int_hom().map(2)));
        let undo_trace_scaling = P_main.base_ring().invert(&P_main.base_ring().coerce(&ZZ, self.slotwise_trace.trace_rank_quo)).unwrap();

        let clone_ct = |(c0, c1): &Ciphertext| (C.get_ring().clone_el_non_fft(c0), C.get_ring().clone_el_non_fft(c1));
        let get_gk = |g: &ZnEl| &gk.iter().filter(|(s, _)| Gal.eq_el(g, s)).next().unwrap().1;

        println!("performing slots-to-coeffs transform");
        let mut key_switches = 0;
        let start = Instant::now();
        let values_in_coefficients = self.slots_to_coeffs_thin.iter().fold(ct, |current, T| T.evaluate_generic(
            &current, 
            |lhs, rhs, factor| {
                *lhs = hom_add(C, hom_mul_plain(P_main, C, factor, clone_ct(rhs)), lhs)
            }, 
            |value, gs| {
                key_switches += 1;
                gs.iter().map(|g| hom_galois(C, clone_ct(value), *g, get_gk(g))).collect()
            },
            || (C.get_ring().non_fft_zero(), C.get_ring().non_fft_zero())
        ));
        let end = Instant::now();
        println!("done in {} ms and {} key switches", (end - start).as_millis(), key_switches);

        println!("computing c0 + c1 * s");
        let start = Instant::now();
        let (c0, c1) = mod_switch_to_plaintext(P_main, C, &values_in_coefficients, mod_switch);
        let enc_sk: Ciphertext = (C.get_ring().non_fft_zero(), C.get_ring().non_fft_from(delta_bootstrap));
        let noisy_decryption = hom_add_plain(P_main, C, &c0, hom_mul_plain(P_main, C, &c1, enc_sk));
        let end = Instant::now();
        println!("done in {} ms and 1 key switch", (end - start).as_millis());

        println!("cancelling pure noise coefficients");
        let mut key_switches = 0;
        let start = Instant::now();
        let cancelled_out_irrelevant_coeffs = self.slotwise_trace.evaluate_generic(
            noisy_decryption, 
            |lhs, rhs| hom_add(C, lhs, rhs), 
            |value, g| {
                key_switches += 1;
                hom_galois(C, clone_ct(value), *g, get_gk(g))
            }
        );
        let cancelled_out_irrelevant_coeffs = hom_mul_plain(&P_main, C, &P.inclusion().map(undo_trace_scaling), cancelled_out_irrelevant_coeffs);
        let end = Instant::now();
        println!("done in {} ms and {} key switches", (end - start).as_millis(), key_switches);

        println!("performing coeffs-to-slots transform");
        let start = Instant::now();
        let mut key_switches = 0;
        let moved_back_to_slots = self.coeffs_to_slots_thin.iter().fold(cancelled_out_irrelevant_coeffs, |current, T| T.evaluate_generic(
            &current, 
            |lhs, rhs, factor| {
                *lhs = hom_add(C, hom_mul_plain(P_main, C, factor, clone_ct(rhs)), lhs)
            }, 
            |value, gs| {
                key_switches += 1;
                gs.iter().map(|g| hom_galois(C, clone_ct(value), *g, get_gk(g))).collect()
            },
            || (C.get_ring().non_fft_zero(), C.get_ring().non_fft_zero())
        ));
        let end = Instant::now();
        println!("done in {} ms and {} key switches", (end - start).as_millis(), key_switches);

        debug_dec_print_slots(P_main, C, &moved_back_to_slots);
        
        let digit_extraction_input = hom_add_plain(P_main, C, &P.inclusion().map(rounding_divisor_half), moved_back_to_slots);

        let mut result = clone_ct(&digit_extraction_input);
        let mut digit_extracted: Vec<Vec<Ciphertext>> = Vec::new();
        for (i, k) in ((self.r + 1)..(self.e + 1)).rev().enumerate() {
            println!("extracting {}-th digit", i);
            let mut key_switches = 0;
            let start = Instant::now();
            let current_P =  &P_bootstrap[i];
            let current_mul_rescale = &mul_rescale_bootstrap[i];
            let delta = C.base_ring().coerce(&ZZbig, ZZbig.rounded_div(
                ZZbig.clone_el(C.base_ring().modulus()), 
                &int_cast(*current_P.base_ring().modulus() as i32, &ZZbig, &StaticRing::<i32>::RING)
            ));

            let mut current_ct = clone_ct(&digit_extraction_input);
            assert_eq!(i, digit_extracted.len());
            for j in 0..i {
                current_ct = hom_sub(C, current_ct, &digit_extracted[j][i - j - 1]);
            }

            let lowest_digit = self.digit_extract_circuits[i].evaluate_generic(
                std::slice::from_ref(&current_ct), 
                |lhs, rhs, factor| {
                    hom_add(C, hom_mul_plain(current_P, C, &current_P.inclusion().map(current_P.base_ring().coerce(&ZZ, factor)), clone_ct(rhs)), &lhs)
                }, 
                |lhs, rhs| {
                    key_switches += 1;
                    hom_mul(C, C_mul, &lhs, &rhs, rk, current_mul_rescale)
                }, 
                |x| {
                    (C.get_ring().non_fft_from(C.base_ring().mul_ref_fst(&delta, C.base_ring().coerce(&ZZ, x))), C.get_ring().non_fft_zero())
                }
            );
            let mut lowest_digit = lowest_digit.collect::<Vec<_>>();
            assert_eq!(k - self.r, lowest_digit.len());
            debug_dec_print_slots(P_main, C, lowest_digit.last().unwrap());
            result = hom_sub(C, result, &lowest_digit.pop().unwrap());
            digit_extracted.push(lowest_digit);
            let end = Instant::now();
            println!("done in {} ms and {} key switches", (end - start).as_millis(), key_switches);
        }

        return result;
    }
}

#[test]
fn test_bfv_thin_bootstrapping_17() {
    let mut rng = thread_rng();
    
    let params = Pow2BFVParams {
        t: 17,
        log2_q_min: 790,
        log2_q_max: 800,
        log2_N: 7
    };
    let bootstrapper = Pow2BFVThinBootstrapParams::create_for(params.clone(), None);
    
    let P = params.create_plaintext_ring();
    let (C, C_mul) = params.create_ciphertext_rings();
    let bootstrapping_data = bootstrapper.create_all_bootstrapping_data(&C, &C_mul);
    
    let sk = gen_sk(&C, &mut rng);
    let gk = bootstrapper.required_galois_keys(&P).into_iter().map(|g| (g, gen_gk(&C, &mut rng, &sk, g))).collect::<Vec<_>>();
    let rk = gen_rk(&C, &mut rng, &sk);
    
    let m = P.int_hom().map(2);
    let ct = enc_sym(&P, &C, &mut rng, &m, &sk);
    let res_ct = bootstrapper.bootstrap_thin(
        &C, 
        &C_mul, 
        &P, 
        &bootstrapping_data.plaintext_ring_hierarchy, 
        &bootstrapping_data.multiplication_rescale_hierarchy, 
        &bootstrapping_data.mod_switch, 
        ct, 
        &rk, 
        &gk
    );

    assert_el_eq!(P, P.int_hom().map(2), dec(&P, &C, res_ct, &sk));
}

#[test]
fn test_bfv_thin_bootstrapping_257() {
    let mut rng = thread_rng();
    
    let params = Pow2BFVParams {
        t: 257,
        log2_q_min: 790,
        log2_q_max: 800,
        log2_N: 10
    };
    let bootstrapper = Pow2BFVThinBootstrapParams::create_for(params.clone(), None);
    
    let P = params.create_plaintext_ring();
    let (C, C_mul) = params.create_ciphertext_rings();
    let bootstrapping_data = bootstrapper.create_all_bootstrapping_data(&C, &C_mul);
    
    let sk = gen_sk(&C, &mut rng);
    let gk = bootstrapper.required_galois_keys(&P).into_iter().map(|g| (g, gen_gk(&C, &mut rng, &sk, g))).collect::<Vec<_>>();
    let rk = gen_rk(&C, &mut rng, &sk);
    
    let m = P.int_hom().map(2);
    let ct = enc_sym(&P, &C, &mut rng, &m, &sk);
    let res_ct = bootstrapper.bootstrap_thin(
        &C, 
        &C_mul, 
        &P, 
        &bootstrapping_data.plaintext_ring_hierarchy, 
        &bootstrapping_data.multiplication_rescale_hierarchy, 
        &bootstrapping_data.mod_switch, 
        ct, 
        &rk, 
        &gk
    );

    assert_el_eq!(P, P.int_hom().map(2), dec(&P, &C, res_ct, &sk));
}

#[test]

#[ignore]
fn run_bfv_thin_bootstrapping() {
    let mut rng = thread_rng();
    
    let params = Pow2BFVParams {
        t: 257,
        log2_q_min: 790,
        log2_q_max: 800,
        log2_N: 15
    };
    println!("Preparing bootstrapper...");

    let bootstrapper = Pow2BFVThinBootstrapParams::create_for(params.clone(), Some("F:\\Users\\Simon\\Documents\\Projekte\\he-ring\\hypercube_bootstrap_257.json"));
    
    println!("Preparing utility data...");

    let P = params.create_plaintext_ring();
    let (C, C_mul) = params.create_ciphertext_rings();
    let bootstrapping_data = bootstrapper.create_all_bootstrapping_data(&C, &C_mul);
    
    println!("Generating keys...");

    let sk = gen_sk(&C, &mut rng);
    let gk = bootstrapper.required_galois_keys(&P).into_iter().map(|g| (g, gen_gk(&C, &mut rng, &sk, g))).collect::<Vec<_>>();
    let rk = gen_rk(&C, &mut rng, &sk);
    
    println!("Preparing message...");

    let m = P.int_hom().map(2);
    let ct = enc_sym(&P, &C, &mut rng, &m, &sk);

    println!("Running bootstrapping...");

    let start = Instant::now();
    let res_ct = bootstrapper.bootstrap_thin(
        &C, 
        &C_mul, 
        &P, 
        &bootstrapping_data.plaintext_ring_hierarchy, 
        &bootstrapping_data.multiplication_rescale_hierarchy, 
        &bootstrapping_data.mod_switch, 
        ct, 
        &rk, 
        &gk
    );
    let end = Instant::now();
    println!("Bootstrapping done in {} ms", (end - start).as_millis());

    let H = HypercubeIsomorphism::load("F:\\Users\\Simon\\Documents\\Projekte\\he-ring\\hypercube_base_257.json", P.get_ring());
    let result = dec(&P, &C, res_ct, &sk);
    for x in H.get_slot_values(&result) {
        H.slot_ring().println(&x);
    }
    // assert_el_eq!(P, P.int_hom().map(2), );
}