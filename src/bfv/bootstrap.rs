
use feanor_math::algorithms::int_factor::is_prime_power;
use feanor_math::divisibility::DivisibilityRingStore;
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
use crate::lintransform::trace::Trace;
use crate::rnsconv;
use crate::digitextract::*;
use crate::lintransform::pow2;
use crate::lintransform::HELinearTransform;
use crate::digitextract::polys::digit_retain_poly;
use crate::lintransform::composite;
use crate::lintransform::matmul::CompiledLinearTransform;
use crate::rings::slots::*;

use super::*;

pub struct ThinBootstrapParams<Params: BFVParams> {
    digit_extract: DigitExtract<Params>,
    slots_to_coeffs_thin: Vec<CompiledLinearTransform<NumberRing<Params>>>,
    coeffs_to_slots_thin: (Vec<CompiledLinearTransform<NumberRing<Params>>>, Option<Trace<NumberRing<Params>, PlaintextAllocator>>)
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

pub const DEFAULT_CONFIG: BootstrapperConfig = BootstrapperConfig {
    set_v: None
};

pub struct BootstrappingDataBundle<Params: BFVParams> {
    ///
    /// Plaintext ring with modulus `p^i` for each `r < i <= e`; 
    /// Entries are stored reversed (i.e. starting with `i = e`).
    /// 
    pub plaintext_ring_hierarchy: Vec<PlaintextRing<Params>>,
    ///
    /// Data for the modulus switch `q -> p^e`
    /// 
    pub mod_switch: ModSwitchData
}

impl ThinBootstrapParams<Pow2BFV> {

    pub fn build_pow2<const LOG: bool>(params: Pow2BFV, t: i64, config: BootstrapperConfig) -> Self {
        let (p, r) = is_prime_power(&ZZ, &t).unwrap();
        let s_can_norm = <_ as HENumberRing>::inf_to_can_norm_expansion_factor(&params.number_ring());
        let v = config.set_v.unwrap_or(((s_can_norm + 1.).log2() / (p as f64).log2()).ceil() as usize);
        let e = r + v;
        if LOG {
            println!("Setting up bootstrapping for plaintext modulus p^r = {}^{} = {} within the cyclotomic ring Q[X]/(Phi_{})", p, r, t, <_ as HECyclotomicNumberRing>::n(&params.number_ring()));
            println!("Choosing e = r + v = {} + {}", r, v);
        }

        let plaintext_ring = params.create_plaintext_ring(ZZ.pow(p, e));
        let original_plaintext_ring = params.create_plaintext_ring(ZZ.pow(p, r));

        let digit_extract = DigitExtract::new_default::<LOG>(params, &plaintext_ring, r);

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
            digit_extract,
            compiled_slots_to_coeffs_thin,
            (compiled_coeffs_to_slots_thin, trace)
        );
    }
}

impl<Params: BFVParams> ThinBootstrapParams<Params>
    where NumberRing<Params>: Clone
{
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

        let plaintext_ring = params.create_plaintext_ring(ZZ.pow(p, e));
        let original_plaintext_ring = params.create_plaintext_ring(ZZ.pow(p, r));

        let digit_extract = DigitExtract::new_default::<LOG>(params, &plaintext_ring, r);

        let H = HypercubeIsomorphism::new::<LOG>(plaintext_ring.get_ring());
        let original_H = H.change_modulus(original_plaintext_ring.get_ring());
        let slots_to_coeffs = log_time::<_, _, LOG, _>("Creating Slots-to-Coeffs transform", |[]| composite::slots_to_powcoeffs_thin(&original_H));
        let coeffs_to_slots = log_time::<_, _, LOG, _>("Creating Coeffs-to-Slots transform", |[]| composite::powcoeffs_to_slots_thin(&H));

        let (compiled_coeffs_to_slots_thin, compiled_slots_to_coeffs_thin): (Vec<_>, Vec<_>) = log_time::<_, _, LOG, _>("Compiling transforms", |[]| (
            coeffs_to_slots.into_iter().map(|T| CompiledLinearTransform::compile(&H, T)).collect::<Vec<_>>(),
            slots_to_coeffs.into_iter().map(|T| CompiledLinearTransform::compile(&original_H, T)).collect::<Vec<_>>()
        ));

        return Self::new_with(
            digit_extract,
            compiled_slots_to_coeffs_thin,
            (compiled_coeffs_to_slots_thin, None)
        );
    }
}

impl<Params: BFVParams> ThinBootstrapParams<Params> {

    pub fn new_with(
        digit_extract: DigitExtract<Params>,
        slots_to_coeffs_thin: Vec<CompiledLinearTransform<NumberRing<Params>>>,
        coeffs_to_slots_thin: (Vec<CompiledLinearTransform<NumberRing<Params>>>, Option<Trace<NumberRing<Params>, PlaintextAllocator>>)
    ) -> Self {
        Self {
            digit_extract,
            slots_to_coeffs_thin,
            coeffs_to_slots_thin
        }
    }

    #[allow(unused)]
    pub fn load(params: Params, load_dir_name: &str) -> Self {
        unimplemented!()
    }
}

///
/// The digit extraction operation, as required in BFV bootstrapping.
/// 
/// Concretely, this encapsulates an efficient implementation of the
/// per-slot digit extraction function
/// ```text
///   Z/p^eZ -> Z/p^rZ x Z/p^eZ,  x -> (x - (x mod p^v) / p^v, x mod p^v)
/// ```
/// for `v = e - r`. Here `x mod p^v` refers to the smallest positive element
/// of `Z/p^eZ` that is congruent to `x` modulo `p^v`.
/// 
pub struct DigitExtract<Params: BFVParams> {
    extraction_circuits: Vec<(Vec<usize>, ArithCircuit)>,
    v: usize,
    e: usize,
    p: i64,
    params: Params
}

impl<Params: BFVParams> DigitExtract<Params> {
    
    pub fn new_default<const LOG: bool>(params: Params, P: &PlaintextRing<Params>, r: usize) -> Self {
        let (p, e) = is_prime_power(ZZ, P.base_ring().modulus()).unwrap();
        assert!(e > r);
        let v = e - r;
        
        let digit_extraction_circuits = log_time::<_, _, LOG, _>("Computing digit extraction polynomials", |[]| {
            (1..=v).rev().map(|i| {
                let required_digits = (2..=(v - i + 1)).chain([r + v - i + 1].into_iter()).collect::<Vec<_>>();
                let poly_ring = DensePolyRing::new(Zn::new(ZZ.pow(p, *required_digits.last().unwrap()) as u64), "X");
                let circuit = poly_to_circuit(&poly_ring, &required_digits.iter().map(|j| digit_retain_poly(&poly_ring, *j)).collect::<Vec<_>>());
                return (required_digits, circuit);
            }).collect::<Vec<_>>()
        });
        assert!(digit_extraction_circuits.is_sorted_by_key(|(digits, _)| *digits.last().unwrap()));
        
        return Self::new_with(params, p, r, e, digit_extraction_circuits);
    }

    ///
    /// Creates a new [`DigitExtract`] from the given circuits.
    /// 
    /// This functions expects the list of circuits to contain tuples `(digits, C)`,
    /// where the circuit `C` takes a single input and computes `digits.len()` outputs, 
    /// such that the `i`-th output is congruent to `lift(input mod p)` modulo 
    /// `p^digits[i]`.
    /// 
    /// If you want to use the default choice of circuits, consider using [`DigitExtract::new_default()`].
    /// 
    pub fn new_with(params: Params, p: i64, r: usize, e: usize, extraction_circuits: Vec<(Vec<usize>, ArithCircuit)>) -> Self {
        assert!(e > r);
        for (digits, circuit) in &extraction_circuits {
            assert!(digits.is_sorted());
            assert_eq!(digits.len(), circuit.output_count());
            assert_eq!(1, circuit.input_count());
        }
        assert!(extraction_circuits.iter().any(|(digits, _)| *digits.last().unwrap() >= e));
        Self {
            extraction_circuits: extraction_circuits,
            v: e - r,
            p: p,
            e: e,
            params
        }
    }

    fn evaluate_homomorphic<'a, const LOG: bool>(&self, 
        P_base: &PlaintextRing<Params>, 
        P_bootstrap: &[PlaintextRing<Params>], 
        C: &CiphertextRing<Params>, 
        C_mul: &CiphertextRing<Params>, 
        ct: &Ciphertext<Params>, 
        rk: &RelinKey<'a, Params>,
        debug_sk: Option<&SecretKey<Params>>
    ) -> (Ciphertext<Params>, Ciphertext<Params>)
        where Params: 'a
    {
        assert!(LOG || debug_sk.is_none());
        let p = self.p;
        let e = self.e;
        let r = self.e - self.v;
        assert_eq!(ZZ.pow(p, r), *P_base.base_ring().modulus());
        for i in 0..self.v {
            assert_eq!(ZZ.pow(p, i + r + 1), *P_bootstrap[i].base_ring().modulus());
        }
        assert_eq!(self.p, p);
        assert_eq!(self.e, e);
        let P = |modulus_exponent: usize| if modulus_exponent <= r {
            assert_eq!(r, modulus_exponent);
            P_base
        } else {
            &P_bootstrap[modulus_exponent - r - 1]
        };

        let mut floor_div_result = Params::clone_ct(C, ct);
        let mut mod_result = Params::transparent_zero(C);
        let mut partial_floor_divs = (0..self.v).map(|_| Params::clone_ct(C, ct)).collect::<Vec<_>>();
        for i in 0..self.v {
            let remaining_digits = e - i;
            let mul_rescale = Params::create_multiplication_rescale(P(remaining_digits), C, &C_mul);
            debug_assert!(self.extraction_circuits.is_sorted_by_key(|(digits, _)| *digits.last().unwrap()));
            let (use_circuit_digits, use_circuit) = self.extraction_circuits.iter().filter(|(digits, _)| *digits.last().unwrap() >= remaining_digits).next().unwrap();
            debug_assert!(use_circuit_digits.is_sorted());

            log_time::<_, _, LOG, _>(format!("Extracting {}-th digit using digit extraction polys for {:?}", i, use_circuit_digits).as_str(), |[key_switches]| {
                let current = &partial_floor_divs[i];

                if let Some(sk) = debug_sk {
                    Params::dec_println_slots(P(remaining_digits), C, current, sk);
                }

                let digit_extracted = hom_evaluate_circuit::<Params>(P(remaining_digits), C, C_mul, current, use_circuit, rk, &mul_rescale, key_switches).collect::<Vec<_>>();
                
                for (res, modulo_exponent) in digit_extracted.iter().zip(use_circuit_digits.iter()) {
                    if let Some(sk) = debug_sk {
                        println!("Digit extraction modulo p^{} result", modulo_exponent);
                        Params::dec_println_slots(P(remaining_digits), C, res, sk);
                    }
                }

                take_mut::take(&mut floor_div_result, |current| Params::hom_sub(C, current, digit_extracted.last().unwrap()));
                take_mut::take(&mut mod_result, |current| Params::hom_add(C, current, digit_extracted.last().unwrap()));
                for j in (i + 1)..self.v {
                    let digit_extracted_index = use_circuit_digits.iter().enumerate().filter(|(_, cleared_digits)| **cleared_digits > j - i).next().unwrap().0;
                    take_mut::take(&mut partial_floor_divs[j], |current| Params::hom_sub(C, current, &digit_extracted[digit_extracted_index]));
                }
            });
        }

        if let Some(sk) = debug_sk {
            println!("Digit extraction final result");
            Params::dec_println_slots(P_base, C, &floor_div_result, sk);
        }
        return (floor_div_result, mod_result);
    }
}


impl<Params: BFVParams> ThinBootstrapParams<Params> {

    pub fn params(&self) -> &Params {
        &self.digit_extract.params
    }

    fn r(&self) -> usize {
        self.digit_extract.e - self.digit_extract.v
    }

    fn e(&self) -> usize {
        self.digit_extract.e
    }

    fn v(&self) -> usize {
        self.digit_extract.v
    }

    fn p(&self) -> i64 {
        self.digit_extract.p
    }

    pub fn create_bootstrapping_plaintext_ring_hierarchy(&self) -> Vec<PlaintextRing<Params>> {
        ((self.r() + 1)..=self.e()).map(|k| self.params().create_plaintext_ring(ZZ.pow(self.p(), k))).collect()
    }

    pub fn create_modulus_switch(&self, P_bootstrap: &[PlaintextRing<Params>], C: &CiphertextRing<Params>) -> ModSwitchData {
        let allocator = Global;
        ModSwitchData {
            scale: rnsconv::bfv_rescale::AlmostExactRescaling::new_with(
                C.base_ring().as_iter().map(|R| Zn::new(*R.modulus() as u64)).collect::<Vec<_>>(), 
                vec![ Zn::new(*P_bootstrap.last().unwrap().base_ring().modulus() as u64) ], 
                C.base_ring().len(), 
                allocator.clone()
            )
        }
    }

    pub fn create_all_bootstrapping_data(&self, C: &CiphertextRing<Params>, C_mul: &CiphertextRing<Params>) -> BootstrappingDataBundle<Params> {
        let plaintext_rings = self.create_bootstrapping_plaintext_ring_hierarchy();
        BootstrappingDataBundle {
            mod_switch: self.create_modulus_switch(&plaintext_rings, C),
            plaintext_ring_hierarchy: plaintext_rings
        }
    }

    pub fn required_galois_keys(&self, P: &PlaintextRing<Params>) -> Vec<ZnEl> {
        let mut result = Vec::new();
        result.extend(self.slots_to_coeffs_thin.iter().flat_map(|T| T.required_galois_keys().into_iter()));
        result.extend(self.coeffs_to_slots_thin.0.iter().flat_map(|T| T.required_galois_keys().into_iter()));
        if let Some(trace) = &self.coeffs_to_slots_thin.1 {
            result.extend(trace.required_galois_keys());
        }
        result.sort_by_key(|g| P.get_ring().cyclotomic_index_ring().smallest_positive_lift(*g));
        result.dedup_by(|g, s| P.get_ring().cyclotomic_index_ring().eq_el(g, s));
        return result;
    }

    pub fn bootstrap_thin<'a, const LOG: bool>(
        &self,
        C: &CiphertextRing<Params>, 
        C_mul: &CiphertextRing<Params>, 
        P_base: &PlaintextRing<Params>,
        P_bootstrap: &[PlaintextRing<Params>],
        mod_switch: &ModSwitchData,
        ct: Ciphertext<Params>,
        rk: &RelinKey<'a, Params>,
        gk: &[(ZnEl, KeySwitchKey<'a, Params>)],
        debug_sk: Option<&SecretKey<Params>>
    ) -> Ciphertext<Params>
        where Params: 'a
    {
        assert!(LOG || debug_sk.is_none());
        assert_eq!(ZZ.pow(self.p(), self.r()), *P_base.base_ring().modulus());
        if LOG {
            println!("Starting Bootstrapping")
        }
        if let Some(sk) = debug_sk {
            Params::dec_println_slots(P_base, C, &ct, sk);
        }

        let P_main = P_bootstrap.last().unwrap();
        debug_assert_eq!(ZZ.pow(self.p(), self.e()), *P_main.base_ring().modulus());

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
        let rounding_divisor_half = P_main.base_ring().coerce(&ZZbig, ZZbig.rounded_div(ZZbig.pow(int_cast(self.p(), ZZbig, ZZ), self.v()), &ZZbig.int_hom().map(2)));
        let digit_extraction_input = Params::hom_add_plain(P_main, C, &P_main.inclusion().map(rounding_divisor_half), noisy_decryption_in_slots);
        let result = self.digit_extract.evaluate_homomorphic::<LOG>(P_base, P_bootstrap, C, C_mul, &digit_extraction_input, rk, debug_sk).0;

        return result;
    }
}

fn hom_compute_linear_transform<'a, Params: BFVParams>(
    P: &PlaintextRing<Params>, 
    C: &CiphertextRing<Params>, 
    input: Ciphertext<Params>, 
    transform: &[CompiledLinearTransform<NumberRing<Params>>], 
    gk: &[(ZnEl, KeySwitchKey<'a, Params>)], 
    key_switches: &mut usize
) -> Ciphertext<Params>
    where Params: 'a
{
    let Gal = P.get_ring().cyclotomic_index_ring();
    let get_gk = |g: &ZnEl| &gk.iter().filter(|(s, _)| Gal.eq_el(g, s)).next().unwrap().1;

    return transform.iter().fold(input, |current, T| T.evaluate_generic(
        current,
        |lhs, rhs| {
            Params::hom_add(C, lhs, rhs)
        }, 
        |value, factor| {
            Params::hom_mul_plain(P, C, factor, value)
        },
        |value, gs| {
            *key_switches += gs.len();
            Params::hom_galois_many(C, value, gs, gs.as_fn().map_fn(|g| get_gk(g)))
        },
        |value| Params::clone_ct(C, value)
    ));
}

fn hom_compute_trace<'a, Params: BFVParams>(P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, input: Ciphertext<Params>, trace: &Trace<NumberRing<Params>, PlaintextAllocator>, gk: &[(ZnEl, KeySwitchKey<'a, Params>)], key_switches: &mut usize) -> Ciphertext<Params>
    where Params: 'a
{
    let Gal = P.get_ring().cyclotomic_index_ring();
    let get_gk = |g: &ZnEl| &gk.iter().filter(|(s, _)| Gal.eq_el(g, s)).next().unwrap().1;
    return trace.evaluate_generic(input, 
    |x, y| Params::hom_add(C, x, y), 
    |value, factor| Params::hom_mul_plain(P, C, factor, value),
    |x, gs| gs.iter().map(|g| {
        if !Gal.is_one(&g) {
            *key_switches += 1;
            Params::hom_galois(C, Params::clone_ct(C, &x), *g, get_gk(&g))
        } else {
            Params::clone_ct(C, &x)
        }
    }).collect(), |x| Params::clone_ct(C, x));
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
) -> impl ExactSizeIterator<Item = Ciphertext<Params>> + use<'a, 'b, Params> 
    where Params: 'b
{
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
    let params = Pow2BFV {
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
    
    let sk = Pow2BFV::gen_sk(&C, &mut rng);
    let gk = bootstrapper.required_galois_keys(&P).into_iter().map(|g| (g, Pow2BFV::gen_gk(&C, &mut rng, &sk, g, digits))).collect::<Vec<_>>();
    let rk = Pow2BFV::gen_rk(&C, &mut rng, &sk, digits);
    
    let m = P.int_hom().map(2);
    let ct = Pow2BFV::enc_sym(&P, &C, &mut rng, &m, &sk);
    let res_ct = bootstrapper.bootstrap_thin::<true>(
        &C, 
        &C_mul, 
        &P, 
        &bootstrapping_data.plaintext_ring_hierarchy, 
        &bootstrapping_data.mod_switch, 
        ct, 
        &rk, 
        &gk,
        None
    );

    assert_el_eq!(P, P.int_hom().map(2), Pow2BFV::dec(&P, &C, res_ct, &sk));
}

#[test]
fn test_pow2_bfv_thin_bootstrapping_23() {
    let mut rng = thread_rng();
    
    // 4 slots of rank 32
    let params = Pow2BFV {
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
    
    let sk = Pow2BFV::gen_sk(&C, &mut rng);
    let gk = bootstrapper.required_galois_keys(&P).into_iter().map(|g| (g, Pow2BFV::gen_gk(&C, &mut rng, &sk, g, digits))).collect::<Vec<_>>();
    let rk = Pow2BFV::gen_rk(&C, &mut rng, &sk, digits);
    
    let m = P.int_hom().map(2);
    let ct = Pow2BFV::enc_sym(&P, &C, &mut rng, &m, &sk);
    let res_ct = bootstrapper.bootstrap_thin::<true>(
        &C, 
        &C_mul, 
        &P, 
        &bootstrapping_data.plaintext_ring_hierarchy, 
        &bootstrapping_data.mod_switch, 
        ct, 
        &rk, 
        &gk,
        None
    );

    assert_el_eq!(P, P.int_hom().map(2), Pow2BFV::dec(&P, &C, res_ct, &sk));
}

#[test]
fn test_composite_bfv_thin_bootstrapping_2() {
    let mut rng = thread_rng();
    
    let params = CompositeBFV {
        log2_q_min: 660,
        log2_q_max: 700,
        n1: 31,
        n2: 11
    };
    let t = 8;
    let digits = 3;
    let bootstrapper = ThinBootstrapParams::build_odd::<true>(params.clone(), t, DEFAULT_CONFIG.set_v(11));
    
    let P = params.create_plaintext_ring(t);
    let (C, C_mul) = params.create_ciphertext_rings();

    let bootstrapping_data = bootstrapper.create_all_bootstrapping_data(&C, &C_mul);
    
    let sk = CompositeBFV::gen_sk(&C, &mut rng);
    let gk = bootstrapper.required_galois_keys(&P).into_iter().map(|g| (g, CompositeBFV::gen_gk(&C, &mut rng, &sk, g, digits))).collect::<Vec<_>>();
    let rk = CompositeBFV::gen_rk(&C, &mut rng, &sk, digits);
    
    let m = P.int_hom().map(2);
    let ct = CompositeBFV::enc_sym(&P, &C, &mut rng, &m, &sk);
    let res_ct = bootstrapper.bootstrap_thin::<true>(
        &C, 
        &C_mul, 
        &P, 
        &bootstrapping_data.plaintext_ring_hierarchy, 
        &bootstrapping_data.mod_switch, 
        ct, 
        &rk, 
        &gk,
        None
    );

    assert_el_eq!(P, P.int_hom().map(2), CompositeBFV::dec(&P, &C, res_ct, &sk));
}
