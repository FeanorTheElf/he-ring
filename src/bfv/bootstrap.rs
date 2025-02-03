
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
use crate::lintransform::matmul::MatmulTransform;
use crate::rnsconv;
use crate::digitextract::*;
use crate::lintransform::pow2::{self, coeffs_to_slots_thin};
use crate::digitextract::polys::digit_retain_poly;
use crate::lintransform::composite;

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
    slots_to_coeffs_thin: PlaintextCircuit<<PlaintextRing<Params> as RingStore>::Type>,
    coeffs_to_slots_thin: PlaintextCircuit<<PlaintextRing<Params> as RingStore>::Type>,
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

        let digit_extract = DigitExtract::new_default::<LOG>(p, e, r);

        let hypercube = HypercubeStructure::halevi_shoup_hypercube(CyclotomicGaloisGroup::new(plaintext_ring.n() as u64), p);
        let H = HypercubeIsomorphism::new::<LOG>(&plaintext_ring, hypercube);
        let original_H = H.change_modulus(&original_plaintext_ring);
        let slots_to_coeffs = log_time::<_, _, LOG, _>("Creating Slots-to-Coeffs transform", |[]| MatmulTransform::to_circuit_many(pow2::slots_to_coeffs_thin(&original_H), &original_H));
        let coeffs_to_slots = log_time::<_, _, LOG, _>("Creating Coeffs-to-Slots transform", |[]| pow2::coeffs_to_slots_thin(&H));
        let plaintext_ring_hierarchy = ((r + 1)..=e).map(|k| self.scheme_params.create_plaintext_ring(ZZ.pow(p, k))).collect();

        return ThinBootstrapData {
            digit_extract,
            slots_to_coeffs_thin: slots_to_coeffs,
            coeffs_to_slots_thin: coeffs_to_slots,
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

        let digit_extract = DigitExtract::new_default::<LOG>(p, e, r);

        let hypercube = HypercubeStructure::halevi_shoup_hypercube(CyclotomicGaloisGroup::new(plaintext_ring.n() as u64), p);
        let H = HypercubeIsomorphism::new::<LOG>(&plaintext_ring, hypercube);
        let original_H = H.change_modulus(&original_plaintext_ring);
        let slots_to_coeffs = log_time::<_, _, LOG, _>("Creating Slots-to-Coeffs transform", |[]| MatmulTransform::to_circuit_many(composite::slots_to_powcoeffs_thin(&original_H), &original_H));
        let coeffs_to_slots = log_time::<_, _, LOG, _>("Creating Coeffs-to-Slots transform", |[]| MatmulTransform::to_circuit_many(composite::powcoeffs_to_slots_thin(&H), &H));
        let plaintext_ring_hierarchy = ((r + 1)..=e).map(|k| self.scheme_params.create_plaintext_ring(ZZ.pow(p, k))).collect();

        return ThinBootstrapData {
            digit_extract,
            slots_to_coeffs_thin: slots_to_coeffs,
            coeffs_to_slots_thin: coeffs_to_slots,
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
/// This function can also be applied to values in a ring `Z/p^e'Z` for
/// `e' > e`, in which case the results are only specified modulo `p^e`, i.e.
/// may be perturbed by an arbitrary value `p^e a`.
/// 
pub struct DigitExtract {
    extraction_circuits: Vec<(Vec<usize>, PlaintextCircuit<StaticRingBase<i64>>)>,
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
    pub fn new_precomputed_p_is_2(p: i64, e: usize, r: usize) -> Self {
        assert_eq!(2, p);
        assert!(is_prime(&ZZ, &p, 10));
        return Self::new_with(
            p, 
            e, 
            r, 
            [1, 2, 4, 8, 16, 23].into_iter().map(|e| (
                [1, 2, 4, 8, 16, 23].into_iter().take_while(|i| *i <= e).collect(),
                precomputed_p_2(e)
            )).collect::<Vec<_>>()
        );
    }
    
    ///
    /// Creates a [`DigitExtract`] for a scalar ring `Z/p^eZ`.
    /// 
    /// Uses the Chen-Han digit retain polynomials <https://ia.cr/2018/067> together with
    /// a heuristic method to compile them into an arithmetic circuit, based on the
    /// Paterson-Stockmeyer method.
    /// 
    pub fn new_default<const LOG: bool>(p: i64, e: usize, r: usize) -> Self {
        assert!(is_prime(&ZZ, &p, 10));
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
        
        return Self::new_with(p, e, r, digit_extraction_circuits);
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
    pub fn new_with(p: i64, e: usize, r: usize, extraction_circuits: Vec<(Vec<usize>, PlaintextCircuit<StaticRingBase<i64>>)>) -> Self {
        assert!(is_prime(&ZZ, &p, 10));
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
        assert!(ZZ.divides(&ring.characteristic(ZZ).unwrap(), &ZZ.pow(p, e)));

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
            let digit_extracted = use_circuit.evaluate_no_galois(std::slice::from_ref(&current), ring.can_hom(&ZZ).unwrap());

            ring.sub_assign(&mut floor_div_result, ring.mul_ref(digit_extracted.last().unwrap(), &scale));
            ring.add_assign(&mut mod_result, ring.mul_ref(digit_extracted.last().unwrap(), &scale));
            for j in (i + 1)..self.v {
                let digit_extracted_index = use_circuit_digits.iter().enumerate().filter(|(_, cleared_digits)| **cleared_digits > j - i).next().unwrap().0;
                ring.sub_assign(&mut partial_floor_divs[j], ring.mul_ref(&digit_extracted[digit_extracted_index], &scale));
            }
        }

        return (floor_div_result, mod_result);
    }

    pub fn evaluate_bfv<'a, Params: BFVParams, const LOG: bool>(&self, 
        P_base: &PlaintextRing<Params>, 
        P_bootstrap: &[PlaintextRing<Params>], 
        C: &CiphertextRing<Params>, 
        Cmul: &CiphertextRing<Params>, 
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
        let v = self.v;

        let (check_p, base_ring_e) = is_prime_power(&ZZ, P_base.base_ring().modulus()).unwrap();
        assert_eq!(check_p, p);
        assert!(base_ring_e >= r);
        assert_eq!(v, P_bootstrap.len());
        for i in 0..v {
            assert_eq!(ZZ.pow(p, base_ring_e + i + 1), *P_bootstrap[i].base_ring().modulus());
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
            debug_assert!(self.extraction_circuits.is_sorted_by_key(|(digits, _)| *digits.last().unwrap()));
            let (use_circuit_digits, use_circuit) = self.extraction_circuits.iter().filter(|(digits, _)| *digits.last().unwrap() >= remaining_digits).next().unwrap();
            debug_assert!(use_circuit_digits.is_sorted());

            log_time::<_, _, LOG, _>(format!("Extracting {}-th digit using digit extraction polys for {:?}", i, use_circuit_digits).as_str(), |[key_switches]| {
                let current = &partial_floor_divs[i];

                if let Some(sk) = debug_sk {
                    Params::dec_println_slots(P(remaining_digits), C, current, sk);
                }

                let digit_extracted = use_circuit.evaluate_bfv::<Params>(P(remaining_digits), C, Some(Cmul), std::slice::from_ref(current), Some(rk), &[], key_switches);
                
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
        C: &CiphertextRing<Params>, 
        Cmul: &CiphertextRing<Params>, 
        P_base: &PlaintextRing<Params>,
        ct: Ciphertext<Params>,
        rk: &RelinKey<'a, Params>,
        gk: &[(CyclotomicGaloisGroupEl, KeySwitchKey<'a, Params>)],
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

        let values_in_coefficients = log_time::<_, _, LOG, _>("1. Computing Slots-to-Coeffs transform", |[key_switches]| {
            let result = self.slots_to_coeffs_thin.evaluate_bfv::<Params>(P_base, C, None, std::slice::from_ref(&ct), None, gk, key_switches);
            assert_eq!(1, result.len());
            return result.into_iter().next().unwrap();
        });
        if let Some(sk) = debug_sk {
            Params::dec_println(P_base, C, &values_in_coefficients, sk);
        }

        let noisy_decryption = log_time::<_, _, LOG, _>("2. Computing noisy decryption c0 + c1 * s", |[key_switches]| {
            let (c0, c1) = Params::mod_switch_to_plaintext(P_main, C, values_in_coefficients);
            let enc_sk = Params::enc_sk(P_main, C);
            *key_switches += 1;
            return Params::hom_add_plain(P_main, C, &c0, Params::hom_mul_plain(P_main, C, &c1, enc_sk));
        });
        if let Some(sk) = debug_sk {
            Params::dec_println(P_main, C, &noisy_decryption, sk);
        }

        let noisy_decryption_in_slots = log_time::<_, _, LOG, _>("3. Computing Coeffs-to-Slots transform", |[key_switches]| {
            let result = self.coeffs_to_slots_thin.evaluate_bfv::<Params>(P_main, C, None, std::slice::from_ref(&noisy_decryption), None, gk, key_switches);
            assert_eq!(1, result.len());
            return result.into_iter().next().unwrap();
        });
        if let Some(sk) = debug_sk {
            Params::dec_println_slots(P_main, C, &noisy_decryption_in_slots, sk);
        }

        if LOG {
            println!("4. Performing digit extraction");
        }
        let rounding_divisor_half = P_main.base_ring().coerce(&ZZbig, ZZbig.rounded_div(ZZbig.pow(int_cast(self.p(), ZZbig, ZZ), self.v()), &ZZbig.int_hom().map(2)));
        let digit_extraction_input = Params::hom_add_plain(P_main, C, &P_main.inclusion().map(rounding_divisor_half), noisy_decryption_in_slots);
        let result = self.digit_extract.evaluate_bfv::<Params, LOG>(P_base, &self.plaintext_ring_hierarchy, C, Cmul, digit_extraction_input, rk, debug_sk).0;

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
    let (C, Cmul) = params.create_ciphertext_rings();
    
    let sk = Pow2BFV::gen_sk(&C, &mut rng);
    let gk = bootstrapper.required_galois_keys(&P).into_iter().map(|g| (g, Pow2BFV::gen_gk(&C, &mut rng, &sk, g, digits))).collect::<Vec<_>>();
    let rk = Pow2BFV::gen_rk(&C, &mut rng, &sk, digits);
    
    let m = P.int_hom().map(2);
    let ct = Pow2BFV::enc_sym(&P, &C, &mut rng, &m, &sk);
    let res_ct = bootstrapper.bootstrap_thin::<true>(
        &C, 
        &Cmul, 
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
    let (C, Cmul) = params.create_ciphertext_rings();
    
    let sk = Pow2BFV::gen_sk(&C, &mut rng);
    let gk = bootstrapper.required_galois_keys(&P).into_iter().map(|g| (g, Pow2BFV::gen_gk(&C, &mut rng, &sk, g, digits))).collect::<Vec<_>>();
    let rk = Pow2BFV::gen_rk(&C, &mut rng, &sk, digits);
    
    let m = P.int_hom().map(2);
    let ct = Pow2BFV::enc_sym(&P, &C, &mut rng, &m, &sk);
    let res_ct = bootstrapper.bootstrap_thin::<true>(
        &C, 
        &Cmul, 
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
    let (C, Cmul) = params.create_ciphertext_rings();
    
    let sk = CompositeBFV::gen_sk(&C, &mut rng);
    let gk = bootstrapper.required_galois_keys(&P).into_iter().map(|g| (g, CompositeBFV::gen_gk(&C, &mut rng, &sk, g, digits))).collect::<Vec<_>>();
    let rk = CompositeBFV::gen_rk(&C, &mut rng, &sk, digits);
    
    let m = P.int_hom().map(2);
    let ct = CompositeBFV::enc_sym(&P, &C, &mut rng, &m, &sk);
    let res_ct = bootstrapper.bootstrap_thin::<true>(
        &C, 
        &Cmul, 
        &P, 
        ct, 
        &rk, 
        &gk,
        None
    );

    assert_el_eq!(P, P.int_hom().map(2), CompositeBFV::dec(&P, &C, res_ct, &sk));
}

#[test]
fn test_digit_extract_homomorphic() {
    let mut rng = thread_rng();
    
    let params = Pow2BFV {
        log2_q_min: 500,
        log2_q_max: 520,
        log2_N: 6,
        ciphertext_allocator: DefaultCiphertextAllocator::default(),
        negacyclic_ntt: PhantomData::<DefaultNegacyclicNTT>
    };
    let digits = 3;
    
    let P1 = params.create_plaintext_ring(17 * 17);
    let P2 = params.create_plaintext_ring(17 * 17 * 17);
    let (C, Cmul) = params.create_ciphertext_rings();

    let sk = Pow2BFV::gen_sk(&C, &mut rng);
    let rk = Pow2BFV::gen_rk(&C, &mut rng, &sk, digits);

    let m = P2.int_hom().map(17 * 17 + 2 * 17 + 5);
    let ct = Pow2BFV::enc_sym(&P2, &C, &mut rng, &m, &sk);

    let digitextract = DigitExtract::new_default::<false>(17, 2, 1);

    let (ct_high, ct_low) = digitextract.evaluate_bfv::<Pow2BFV, true>(&P1, std::slice::from_ref(&P2), &C, &Cmul, ct, &rk, Some(&sk));

    let m_high = Pow2BFV::dec(&P1, &C, Pow2BFV::clone_ct(&C, &ct_high), &sk);
    assert!(P1.wrt_canonical_basis(&m_high).iter().skip(1).all(|x| P1.base_ring().is_zero(&x)));
    let m_high = P1.base_ring().smallest_positive_lift(P1.wrt_canonical_basis(&m_high).at(0));
    assert_eq!(2, m_high % 17);

    let m_low = Pow2BFV::dec(&P2, &C, Pow2BFV::clone_ct(&C, &ct_low), &sk);
    assert!(P1.wrt_canonical_basis(&m_low).iter().skip(1).all(|x| P2.base_ring().is_zero(&x)));
    let m_low = P1.base_ring().smallest_positive_lift(P1.wrt_canonical_basis(&m_low).at(0));
    assert_eq!(5, m_low % (17 * 17));
}

#[test]
fn test_digit_extract_evaluate() {
    let ring = Zn::new(16);
    let digit_extract = DigitExtract::new_default::<false>(2, 4, 2);
    for x in 0..16 {
        let (actual_high, actual_low) = digit_extract.evaluate(&ring, ring.int_hom().map(x));
        assert_eq!((x / 4) * 4, ring.smallest_positive_lift(actual_high) as i32);
        assert_eq!(x % 4, ring.smallest_positive_lift(actual_low) as i32);
    }

    let ring = Zn::new(81);
    let digit_extract = DigitExtract::new_default::<false>(3, 4, 2);
    for x in 0..81 {
        let (actual_high, actual_low) = digit_extract.evaluate(&ring, ring.int_hom().map(x));
        assert_eq!((x / 9) * 9, ring.smallest_positive_lift(actual_high) as i32);
        assert_eq!(x % 9, ring.smallest_positive_lift(actual_low) as i32);
    }

    let ring = Zn::new(125);
    let digit_extract = DigitExtract::new_default::<false>(5, 3, 2);
    for x in 0..125 {
        let (actual_high, actual_low) = digit_extract.evaluate(&ring, ring.int_hom().map(x));
        assert_eq!((x / 5) * 5, ring.smallest_positive_lift(actual_high) as i32);
        assert_eq!(x % 5, ring.smallest_positive_lift(actual_low) as i32);
    }
}

#[test]
fn test_digit_extract_evaluate_ignore_higher() {
    let ring = Zn::new(64);
    let digit_extract = DigitExtract::new_default::<false>(2, 4, 2);
    for x in 0..64 {
        let (actual_high, actual_low) = digit_extract.evaluate(&ring, ring.int_hom().map(x));
        assert_eq!(((x / 4) * 4) % 16, ring.smallest_positive_lift(actual_high) as i32 % 16);
        assert_eq!(x % 4, ring.smallest_positive_lift(actual_low) as i32 % 16);
    }

    let ring = Zn::new(243);
    let digit_extract = DigitExtract::new_default::<false>(3, 4, 2);
    for x in 0..243 {
        let (actual_high, actual_low) = digit_extract.evaluate(&ring, ring.int_hom().map(x));
        assert_eq!(((x / 9) * 9) % 81, ring.smallest_positive_lift(actual_high) as i32 % 81);
        assert_eq!(x % 9, ring.smallest_positive_lift(actual_low) as i32 % 81);
    }

    let ring = Zn::new(625);
    let digit_extract = DigitExtract::new_default::<false>(5, 3, 2);
    for x in 0..625 {
        let (actual_high, actual_low) = digit_extract.evaluate(&ring, ring.int_hom().map(x));
        assert_eq!(((x / 5) * 5) % 125, ring.smallest_positive_lift(actual_high) as i32 % 125);
        assert_eq!(x % 5, ring.smallest_positive_lift(actual_low) as i32 % 125);
    }
}
