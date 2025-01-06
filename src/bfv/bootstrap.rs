
use feanor_math::algorithms::int_factor::is_prime_power;
use feanor_math::divisibility::{DivisibilityRing, DivisibilityRingStore};
use feanor_math::homomorphism::{CanHomFrom, Homomorphism};
use feanor_math::integer::{int_cast, IntegerRingStore};
use feanor_math::primitive_int::{StaticRing, StaticRingBase};
use feanor_math::ring::*;
use feanor_math::rings::poly::dense_poly::DensePolyRing;
use feanor_math::rings::zn::zn_64::ZnEl;
use feanor_math::rings::zn::ZnRingStore;
use feanor_math::seq::VectorView;
use polys::{poly_to_circuit, precomputed_p_2};
use rand::thread_rng;

use crate::cyclotomic::CyclotomicRingStore;
use crate::lintransform::trace::Trace;
use crate::rnsconv;
use crate::digitextract::*;
use crate::lintransform::pow2::{self, coeffs_to_slots_thin};
use crate::lintransform::HELinearTransform;
use crate::digitextract::polys::digit_retain_poly;
use crate::lintransform::composite;
use crate::lintransform::matmul::CompiledLinearTransform;

use super::*;

#[derive(Clone, Debug)]
pub struct ThinBootstrapParams<Params: BFVParams> {
    pub scheme_params: Params,
    pub v: usize,
    pub t: i64
}

///
/// Precomputed data required to perform BFV bootstrapping.
/// 
/// The standard way to create this data is to use [`ThinBootstrapParams::build_pow2()`]
/// or [`ThinBootstrapParams::build_odd()`], but note that this computation is very expensive.
/// 
pub struct ThinBootstrapData<Params: BFVParams> {
    digit_extract: DigitExtract,
    slots_to_coeffs_thin: Vec<CompiledLinearTransform<NumberRing<Params>>>,
    coeffs_to_slots_thin: (Vec<CompiledLinearTransform<NumberRing<Params>>>, Option<Trace<NumberRing<Params>>>),
    plaintext_ring_hierarchy: Vec<PlaintextRing<Params>>
}

impl<Params: BFVParams> ThinBootstrapParams<Params>
    where NumberRing<Params>: Clone
{
    pub fn build_pow2<const LOG: bool>(&self) -> ThinBootstrapData<Params> {
        let log2_n = ZZ.abs_log2_ceil(&(self.scheme_params.number_ring().n() as i64)).unwrap();
        assert_eq!(self.scheme_params.number_ring().n(), 1 << log2_n);

        let (p, r) = is_prime_power(&ZZ, &self.t).unwrap();
        let s_can_norm = <_ as HENumberRing>::inf_to_can_norm_expansion_factor(&self.scheme_params.number_ring());
        let v = self.v;
        let e = r + v;
        if LOG {
            println!("Setting up bootstrapping for plaintext modulus p^r = {}^{} = {} within the cyclotomic ring Q[X]/(Phi_{})", p, r, self.t, <_ as HECyclotomicNumberRing>::n(&self.scheme_params.number_ring()));
            println!("Choosing e = r + v = {} + {}", r, v);
        }

        let plaintext_ring = self.scheme_params.create_plaintext_ring(ZZ.pow(p, e));
        let original_plaintext_ring = self.scheme_params.create_plaintext_ring(ZZ.pow(p, r));

        let digit_extract = DigitExtract::new_default::<LOG>(plaintext_ring.base_ring(), r);

        let hypercube = HypercubeStructure::halevi_shoup_hypercube(CyclotomicGaloisGroup::new(plaintext_ring.n()), p);
        let H = HypercubeIsomorphism::new::<LOG>(&plaintext_ring, hypercube);
        let original_H = H.change_modulus(&original_plaintext_ring);
        let slots_to_coeffs = log_time::<_, _, LOG, _>("Creating Slots-to-Coeffs transform", |[]| pow2::slots_to_coeffs_thin(&original_H));
        let (coeffs_to_slots, trace) = log_time::<_, _, LOG, _>("Creating Coeffs-to-Slots transform", |[]| {
            let (transforms, trace) = pow2::coeffs_to_slots_thin(&H);
            (transforms, Some(trace))
        });
        let (compiled_coeffs_to_slots_thin, compiled_slots_to_coeffs_thin): (Vec<_>, Vec<_>) = log_time::<_, _, LOG, _>("Compiling transforms", |[]| (
            coeffs_to_slots.into_iter().map(|T| CompiledLinearTransform::compile(&H, T)).collect::<Vec<_>>(),
            slots_to_coeffs.into_iter().map(|T| CompiledLinearTransform::compile(&original_H, T)).collect::<Vec<_>>()
        ));

        let plaintext_ring_hierarchy = ((r + 1)..=e).map(|k| self.scheme_params.create_plaintext_ring(ZZ.pow(p, k))).collect();

        return ThinBootstrapData {
            digit_extract,
            slots_to_coeffs_thin: compiled_slots_to_coeffs_thin,
            coeffs_to_slots_thin: (compiled_coeffs_to_slots_thin, trace),
            plaintext_ring_hierarchy: plaintext_ring_hierarchy
        };
    }

    pub fn build_odd<const LOG: bool>(&self) -> ThinBootstrapData<Params> {
        assert!(self.scheme_params.number_ring().n() % 2 != 0);

        let (p, r) = is_prime_power(&ZZ, &self.t).unwrap();
        let s_can_norm = self.scheme_params.number_ring().inf_to_can_norm_expansion_factor();
        let v = self.v;
        let e = r + v;
        if LOG {
            println!("Setting up bootstrapping for plaintext modulus p^r = {}^{} = {} within the cyclotomic ring Q[X]/(Phi_{})", p, r, self.t, self.scheme_params.number_ring().n());
            println!("Choosing e = r + v = {} + {}", r, v);
        }

        let plaintext_ring = self.scheme_params.create_plaintext_ring(ZZ.pow(p, e));
        let original_plaintext_ring = self.scheme_params.create_plaintext_ring(ZZ.pow(p, r));

        let digit_extract = DigitExtract::new_default::<LOG>(plaintext_ring.base_ring(), r);

        let hypercube = HypercubeStructure::halevi_shoup_hypercube(CyclotomicGaloisGroup::new(plaintext_ring.n()), p);
        let H = HypercubeIsomorphism::new::<LOG>(&plaintext_ring, hypercube);
        let original_H = H.change_modulus(&original_plaintext_ring);
        let slots_to_coeffs = log_time::<_, _, LOG, _>("Creating Slots-to-Coeffs transform", |[]| composite::slots_to_powcoeffs_thin(&original_H));
        let coeffs_to_slots = log_time::<_, _, LOG, _>("Creating Coeffs-to-Slots transform", |[]| composite::powcoeffs_to_slots_thin(&H));

        let (compiled_coeffs_to_slots_thin, compiled_slots_to_coeffs_thin): (Vec<_>, Vec<_>) = log_time::<_, _, LOG, _>("Compiling transforms", |[]| (
            coeffs_to_slots.into_iter().map(|T| CompiledLinearTransform::compile(&H, T)).collect::<Vec<_>>(),
            slots_to_coeffs.into_iter().map(|T| CompiledLinearTransform::compile(&original_H, T)).collect::<Vec<_>>()
        ));

        let plaintext_ring_hierarchy = ((r + 1)..=e).map(|k| self.scheme_params.create_plaintext_ring(ZZ.pow(p, k))).collect();

        return ThinBootstrapData {
            digit_extract,
            slots_to_coeffs_thin: compiled_slots_to_coeffs_thin,
            coeffs_to_slots_thin: (compiled_coeffs_to_slots_thin, None),
            plaintext_ring_hierarchy: plaintext_ring_hierarchy
        };
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
pub struct DigitExtract {
    extraction_circuits: Vec<(Vec<usize>, ArithCircuit)>,
    v: usize,
    e: usize,
    p: i64
}

impl DigitExtract {

    ///
    /// Creates a [`DigitExtract`] for a scalar ring `Z/2^eZ`.
    /// 
    /// Uses the precomputed table of best digit extraction circuits for `e <= 23`.
    /// 
    pub fn new_precomputed_p_is_2(scalar_ring: &Zn, r: usize) -> Self {
        let (p, e) = is_prime_power(ZZ, scalar_ring.modulus()).unwrap();
        assert_eq!(2, p);
        return Self::new_with(
            p, 
            r, 
            e, 
            [1, 2, 4, 8, 16, 23].into_iter().map(|e| (
                [1, 2, 4, 8, 16, 23].into_iter().take_while(|i| *i <= e).collect(),
                precomputed_p_2(e)
            )).collect::<Vec<_>>()
        );
    }
    
    ///
    /// Creates a [`DigitExtract`] for a scalar ring `Z/p^eZ`.
    /// 
    /// Uses the Chen-Han digit retain polynomials [https://ia.cr/2018/067] together with
    /// a heuristic method to compile them into an arithmetic circuit, based on the
    /// Paterson-Stockmeyer method.
    /// 
    pub fn new_default<const LOG: bool>(scalar_ring: &Zn, r: usize) -> Self {
        let (p, e) = is_prime_power(ZZ, scalar_ring.modulus()).unwrap();
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
        
        return Self::new_with(p, r, e, digit_extraction_circuits);
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
    pub fn new_with(p: i64, r: usize, e: usize, extraction_circuits: Vec<(Vec<usize>, ArithCircuit)>) -> Self {
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
            e: e
        }
    }

    pub fn r(&self) -> usize {
        self.e - self.v
    }

    pub fn e(&self) -> usize {
        self.e
    }

    pub fn v(&self) -> usize {
        self.v
    }

    pub fn p(&self) -> i64 {
        self.p
    }

    ///
    /// Computes the function directly on a ring. Mainly designed for testing.
    /// 
    /// To avoid requiring many rings, this actually computes a slight variant
    /// of the digit extraction function on only one ring, namely
    /// ```text
    ///   Z/p^eZ -> Z/p^eZ x Z/p^eZ,  x -> (x - (x mod p^v), x mod p^v)
    /// ```
    /// In particular, the first returned value is divisible by `p^v`.
    /// 
    pub fn evaluate<R>(&self, ring: R, input: El<R>) -> (El<R>, El<R>)
        where R: RingStore + Copy,
            R::Type: CanHomFrom<StaticRingBase<i64>> + DivisibilityRing
    {
        let p = self.p;
        let e = self.e;
        let r = self.e - self.v;
        assert_eq!(ZZ.pow(p, e), ring.characteristic(ZZ).unwrap());

        let mut mod_result = ring.zero();
        let mut partial_floor_divs = (0..self.v).map(|_| ring.clone_el(&input)).collect::<Vec<_>>();
        let mut floor_div_result = input;
        for i in 0..self.v {
            let remaining_digits = e - i;
            debug_assert!(self.extraction_circuits.is_sorted_by_key(|(digits, _)| *digits.last().unwrap()));
            let (use_circuit_digits, use_circuit) = self.extraction_circuits.iter().filter(|(digits, _)| *digits.last().unwrap() >= remaining_digits).next().unwrap();
            debug_assert!(use_circuit_digits.is_sorted());

            let scale = ring.coerce(&ZZ, ZZ.pow(p, i));
            let current = ring.checked_div(&partial_floor_divs[i], &scale).unwrap();
            let digit_extracted = use_circuit.evaluate(std::slice::from_ref(&current), ring).collect::<Vec<_>>();

            ring.sub_assign(&mut floor_div_result, ring.mul_ref(digit_extracted.last().unwrap(), &scale));
            ring.add_assign(&mut mod_result, ring.mul_ref(digit_extracted.last().unwrap(), &scale));
            for j in (i + 1)..self.v {
                let digit_extracted_index = use_circuit_digits.iter().enumerate().filter(|(_, cleared_digits)| **cleared_digits > j - i).next().unwrap().0;
                ring.sub_assign(&mut partial_floor_divs[j], ring.mul_ref(&digit_extracted[digit_extracted_index], &scale));
            }
        }

        return (floor_div_result, mod_result);
    }

    pub fn evaluate_homomorphic<'a, Params: BFVParams, const LOG: bool>(&self, 
        P_base: &PlaintextRing<Params>, 
        P_bootstrap: &[PlaintextRing<Params>], 
        C: &CiphertextRing<Params>, 
        C_mul: &CiphertextRing<Params>, 
        ct: Ciphertext<Params>, 
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
        let P = |modulus_exponent: usize| if modulus_exponent <= r {
            assert_eq!(r, modulus_exponent);
            P_base
        } else {
            &P_bootstrap[modulus_exponent - r - 1]
        };

        let mut mod_result = Params::transparent_zero(C);
        let mut partial_floor_divs = (0..self.v).map(|_| Params::clone_ct(C, &ct)).collect::<Vec<_>>();
        let mut floor_div_result = ct;
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


impl<Params: BFVParams> ThinBootstrapData<Params> {

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

    fn create_modulus_switch(&self, P_bootstrap: &[PlaintextRing<Params>], C: &CiphertextRing<Params>) -> ModSwitchData {
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

    pub fn required_galois_keys(&self, P: &PlaintextRing<Params>) -> Vec<ZnEl> {
        let mut result = Vec::new();
        result.extend(self.slots_to_coeffs_thin.iter().flat_map(|T| T.required_galois_keys().into_iter()));
        result.extend(self.coeffs_to_slots_thin.0.iter().flat_map(|T| T.required_galois_keys().into_iter()));
        if let Some(trace) = &self.coeffs_to_slots_thin.1 {
            result.extend(<_ as HELinearTransform<_, Global>>::required_galois_keys(trace));
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

        let P_main = self.plaintext_ring_hierarchy.last().unwrap();
        debug_assert_eq!(ZZ.pow(self.p(), self.e()), *P_main.base_ring().modulus());
        let mod_switch = self.create_modulus_switch(&self.plaintext_ring_hierarchy, C);

        let values_in_coefficients = log_time::<_, _, LOG, _>("1. Computing Slots-to-Coeffs transform", |[key_switches]| {
            return Params::hom_compute_linear_transform::<_, false>(P_base, C, ct, &self.slots_to_coeffs_thin, gk, key_switches);
        });
        if let Some(sk) = debug_sk {
            Params::dec_println(P_base, C, &values_in_coefficients, sk);
        }

        let noisy_decryption = log_time::<_, _, LOG, _>("2. Computing noisy decryption c0 + c1 * s", |[key_switches]| {
            let (c0, c1) = Params::mod_switch_to_plaintext(P_main, C, values_in_coefficients, &mod_switch);
            let enc_sk = Params::enc_sk(P_main, C);
            *key_switches += 1;
            return Params::hom_add_plain(P_main, C, &c0, Params::hom_mul_plain(P_main, C, &c1, enc_sk));
        });
        if let Some(sk) = debug_sk {
            Params::dec_println(P_main, C, &noisy_decryption, sk);
        }

        let noisy_decryption_in_slots = log_time::<_, _, LOG, _>("3. Computing Coeffs-to-Slots transform", |[key_switches]| {
            let moved_to_slots = Params::hom_compute_linear_transform::<_, false>(P_main, C, noisy_decryption, &self.coeffs_to_slots_thin.0, gk, key_switches);
            if let Some(trace) = &self.coeffs_to_slots_thin.1 {
                return Params::hom_compute_linear_transform::<_, false>(P_main, C, moved_to_slots, std::slice::from_ref(trace), gk, key_switches);
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
        let result = self.digit_extract.evaluate_homomorphic::<Params, LOG>(P_base, &self.plaintext_ring_hierarchy, C, C_mul, digit_extraction_input, rk, debug_sk).0;

        return result;
    }
}

#[test]
fn test_pow2_bfv_thin_bootstrapping_17() {
    let mut rng = thread_rng();
    
    // 8 slots of rank 16
    let params = Pow2BFV {
        log2_q_min: 790,
        log2_q_max: 800,
        log2_N: 7,
        ciphertext_allocator: DefaultCiphertextAllocator::default(),
        negacyclic_ntt: PhantomData::<DefaultNegacyclicNTT>
    };
    let t = 17;
    let digits = 3;
    let bootstrap_params = ThinBootstrapParams {
        scheme_params: params.clone(),
        v: 2,
        t: t
    };
    let bootstrapper = bootstrap_params.build_pow2::<true>();
    
    let P = params.create_plaintext_ring(t);
    let (C, C_mul) = params.create_ciphertext_rings();
    
    let sk = Pow2BFV::gen_sk(&C, &mut rng);
    let gk = bootstrapper.required_galois_keys(&P).into_iter().map(|g| (g, Pow2BFV::gen_gk(&C, &mut rng, &sk, g, digits))).collect::<Vec<_>>();
    let rk = Pow2BFV::gen_rk(&C, &mut rng, &sk, digits);
    
    let m = P.int_hom().map(2);
    let ct = Pow2BFV::enc_sym(&P, &C, &mut rng, &m, &sk);
    let res_ct = bootstrapper.bootstrap_thin::<true>(
        &C, 
        &C_mul, 
        &P, 
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
        log2_N: 7,
        ciphertext_allocator: DefaultCiphertextAllocator::default(),
        negacyclic_ntt: PhantomData::<DefaultNegacyclicNTT>
    };
    let t = 23;
    let digits = 3;
    let bootstrap_params = ThinBootstrapParams {
        scheme_params: params.clone(),
        v: 2,
        t: t
    };
    let bootstrapper = bootstrap_params.build_pow2::<true>();
    
    let P = params.create_plaintext_ring(t);
    let (C, C_mul) = params.create_ciphertext_rings();
    
    let sk = Pow2BFV::gen_sk(&C, &mut rng);
    let gk = bootstrapper.required_galois_keys(&P).into_iter().map(|g| (g, Pow2BFV::gen_gk(&C, &mut rng, &sk, g, digits))).collect::<Vec<_>>();
    let rk = Pow2BFV::gen_rk(&C, &mut rng, &sk, digits);
    
    let m = P.int_hom().map(2);
    let ct = Pow2BFV::enc_sym(&P, &C, &mut rng, &m, &sk);
    let res_ct = bootstrapper.bootstrap_thin::<true>(
        &C, 
        &C_mul, 
        &P, 
        ct, 
        &rk, 
        &gk,
        None
    );

    assert_el_eq!(P, P.int_hom().map(2), Pow2BFV::dec(&P, &C, res_ct, &sk));
}

#[test]
fn test_composite_bfv_thin_bootstrapping_2_takes_long() {
    let mut rng = thread_rng();
    
    let params = CompositeBFV {
        log2_q_min: 660,
        log2_q_max: 700,
        n1: 31,
        n2: 11,
        ciphertext_allocator: DefaultCiphertextAllocator::default()
    };
    let t = 8;
    let digits = 3;
    let bootstrap_params = ThinBootstrapParams {
        scheme_params: params.clone(),
        v: 9,
        t: t
    };
    let bootstrapper = bootstrap_params.build_odd::<true>();
    
    let P = params.create_plaintext_ring(t);
    let (C, C_mul) = params.create_ciphertext_rings();
    
    let sk = CompositeBFV::gen_sk(&C, &mut rng);
    let gk = bootstrapper.required_galois_keys(&P).into_iter().map(|g| (g, CompositeBFV::gen_gk(&C, &mut rng, &sk, g, digits))).collect::<Vec<_>>();
    let rk = CompositeBFV::gen_rk(&C, &mut rng, &sk, digits);
    
    let m = P.int_hom().map(2);
    let ct = CompositeBFV::enc_sym(&P, &C, &mut rng, &m, &sk);
    let res_ct = bootstrapper.bootstrap_thin::<true>(
        &C, 
        &C_mul, 
        &P, 
        ct, 
        &rk, 
        &gk,
        None
    );

    assert_el_eq!(P, P.int_hom().map(2), CompositeBFV::dec(&P, &C, res_ct, &sk));
}

#[test]
fn test_digit_extract_evaluate() {
    let ring = Zn::new(16);
    let digit_extract = DigitExtract::new_default::<false>(&ring, 2);
    for x in 0..16 {
        let (actual_high, actual_low) = digit_extract.evaluate(&ring, ring.int_hom().map(x));
        assert_eq!((x / 4) * 4, ring.smallest_positive_lift(actual_high) as i32);
        assert_eq!(x % 4, ring.smallest_positive_lift(actual_low) as i32);
    }

    let ring = Zn::new(81);
    let digit_extract = DigitExtract::new_default::<false>(&ring, 2);
    for x in 0..81 {
        let (actual_high, actual_low) = digit_extract.evaluate(&ring, ring.int_hom().map(x));
        assert_eq!((x / 9) * 9, ring.smallest_positive_lift(actual_high) as i32);
        assert_eq!(x % 9, ring.smallest_positive_lift(actual_low) as i32);
    }

    let ring = Zn::new(125);
    let digit_extract = DigitExtract::new_default::<false>(&ring, 2);
    for x in 0..125 {
        let (actual_high, actual_low) = digit_extract.evaluate(&ring, ring.int_hom().map(x));
        assert_eq!((x / 5) * 5, ring.smallest_positive_lift(actual_high) as i32);
        assert_eq!(x % 5, ring.smallest_positive_lift(actual_low) as i32);
    }
}

#[cfg(test)]
fn test_circuit() -> ArithCircuit {
    
    let id = ArithCircuit::linear_transform(&[1]);
    let f0 = id.clone();
    let f1 = id.tensor(&ArithCircuit::mul()).compose(&ArithCircuit::select(1, &[0, 0, 0]).compose(&f0));
    let f2 = id.tensor(&id).tensor(&ArithCircuit::mul()).compose(&ArithCircuit::select(2, &[0, 1, 1, 1]).compose(&f1));
    
    let f3_comp = ArithCircuit::add().compose(&ArithCircuit::linear_transform(&[112]).tensor(
        &ArithCircuit::mul().compose(&ArithCircuit::linear_transform(&[94, 121]).output_twice())
    )).compose(&ArithCircuit::select(2, &[0, 0, 1]));
    let f3 = id.tensor(&id).tensor(&id).tensor(&f3_comp).compose(&ArithCircuit::select(3, &[0, 1, 2, 1, 2]).compose(&f2));

    let f4_comp = ArithCircuit::add().compose(&ArithCircuit::linear_transform(&[1984, 528, 22620]).tensor(
        &ArithCircuit::mul().compose(&ArithCircuit::linear_transform(&[226, 113]).tensor(&ArithCircuit::linear_transform(&[8, 2, 301])))
    )).compose(&ArithCircuit::select(3, &[0, 1, 2, 1, 2, 0, 1, 2]));
    let f4 = id.tensor(&id).tensor(&id).tensor(&id).tensor(&f4_comp).compose(&ArithCircuit::select(4, &[0, 1, 2, 3, 1, 2, 3]).compose(&f3));

    return f4;
}

#[test]
fn test_evaluate_circuit() {
    let mut rng = thread_rng();
    let params = CompositeSingleRNSBFV {
        log2_q_max: 800,
        log2_q_min: 780,
        n1: 11,
        n2: 31,
        ciphertext_allocator: DefaultCiphertextAllocator::default(),
        convolution: PhantomData::<DefaultConvolution>
    };
    let t = ZZ.pow(2, 15);
    let digits = 4;
    
    let P = params.create_plaintext_ring(t);
    let (C, C_mul) = params.create_ciphertext_rings();

    let sk = CompositeSingleRNSBFV::gen_sk(&C, &mut rng);
    let mul_rescale = CompositeSingleRNSBFV::create_multiplication_rescale(&P, &C, &C_mul);
    let rk = CompositeSingleRNSBFV::gen_rk(&C, &mut rng, &sk, digits);

    let m = P.zero();
    let ct = CompositeSingleRNSBFV::enc_sym(&P, &C, &mut rng, &m, &sk);

    let circuit = test_circuit();
    let mut key_switches = 0;
    let result = hom_evaluate_circuit::<CompositeSingleRNSBFV>(&P, &C, &C_mul, &ct, &circuit, &rk, &mul_rescale, &mut key_switches);

    let result_dec = result.map(|ct| CompositeSingleRNSBFV::dec(&P, &C, ct, &sk)).collect::<Vec<_>>();
    assert_el_eq!(&P, P.zero(), &result_dec[0]);
    assert_el_eq!(&P, P.zero(), &result_dec[1]);
    assert_el_eq!(&P, P.zero(), &result_dec[2]);
    assert_el_eq!(&P, P.zero(), &result_dec[4]);
}