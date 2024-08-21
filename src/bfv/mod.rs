#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use std::alloc::Global;
use std::ptr::Alignment;
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Instant;
use feanor_math::homomorphism::CanHom;
use feanor_math::homomorphism::Identity;
use feanor_math::ring::*;
use feanor_math::assert_el_eq;
use feanor_math::rings::zn::*;
use feanor_math::rings::zn::zn_64::*;
use feanor_math::primitive_int::StaticRing;
use feanor_math::integer::*;
use feanor_math::algorithms::miller_rabin::is_prime;
use feanor_math::algorithms::fft::*;
use feanor_math::homomorphism::Homomorphism;
use feanor_math::ordered::OrderedRingStore;
use feanor_math::rings::extension::FreeAlgebraStore;
use feanor_math::seq::*;
use feanor_mempool::dynsize::DynLayoutMempool;
use feanor_mempool::AllocArc;

use crate::cyclotomic::CyclotomicRing;
use crate::rings::pow2_cyclotomic::*;
use crate::rings::gadget_product::*;
use crate::rings::double_rns_ring::*;
use crate::rings::ntt_ring::*;
use crate::profiling::*;
use crate::rings::slots::HypercubeIsomorphism;
use crate::rnsconv;

use rand::thread_rng;
use rand::{Rng, CryptoRng};
use rand_distr::StandardNormal;

pub mod bootstrap;

type PlaintextZn = zn_64::Zn;
type PlaintextFFT = cooley_tuckey::CooleyTuckeyFFT<<PlaintextZn as RingStore>::Type, <PlaintextZn as RingStore>::Type, Identity<PlaintextZn>>;
pub type PlaintextRing = NTTRing<PlaintextZn, Pow2CyclotomicFFT<PlaintextZn, PlaintextFFT>>;

type CiphertextAllocator = AllocArc<DynLayoutMempool>;
type CiphertextZn = zn_64::Zn;
type CiphertextFastmulZn = zn_64::ZnFastmul;
type CiphertextFFT = cooley_tuckey::CooleyTuckeyFFT<<CiphertextZn as RingStore>::Type, <CiphertextFastmulZn as RingStore>::Type, CanHom<CiphertextFastmulZn, CiphertextZn>>;
type CiphertextGenFFT = Pow2CyclotomicFFT<CiphertextZn, CiphertextFFT>;
pub type CiphertextRing = DoubleRNSRing<CiphertextZn, CiphertextGenFFT, CiphertextAllocator>;

pub type Ciphertext = (DoubleRNSNonFFTEl<CiphertextZn, CiphertextGenFFT, CiphertextAllocator>, DoubleRNSNonFFTEl<CiphertextZn, CiphertextGenFFT, CiphertextAllocator>);
pub type SecretKey = El<CiphertextRing>;
pub type GadgetProductOperand<'a> = GadgetProductRhsOperand<'a, CiphertextGenFFT, CiphertextAllocator>;
pub type KeySwitchKey<'a> = (GadgetProductOperand<'a>, GadgetProductOperand<'a>);
pub type RelinKey<'a> = (GadgetProductOperand<'a>, GadgetProductOperand<'a>);

static DEBUG_SK: RwLock<Option<SecretKey>> = RwLock::new(None);

pub struct MulConversionData {
    lift_to_C_mul: rnsconv::shared_lift::AlmostExactSharedBaseConversion<CiphertextAllocator>,
    scale_down_to_C: rnsconv::bfv_rescale::AlmostExactRescalingConvert<CiphertextAllocator>
}

pub struct ModSwitchData {
    scale: rnsconv::bfv_rescale::AlmostExactRescaling<CiphertextAllocator>
}

const ZZbig: BigIntRing = BigIntRing::RING;
const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;

fn max_prime_congruent_one_lt_bound(n: i64, bound: i64) -> Option<i64> {
    let mut candidate = (bound - 2) - ((bound - 2) % n) + 1;

    while candidate > 0 {
        if is_prime(ZZ, &candidate, 10) {
            return Some(candidate);
        }
        candidate -= n;
    }
    return None;
}

#[derive(Clone, Debug)]
pub struct Pow2BFVParams {
    t: i64,
    log2_q_min: usize,
    log2_q_max: usize,
    log2_N: usize
}

impl Pow2BFVParams {

    pub fn create_ciphertext_rings(&self) -> (CiphertextRing, CiphertextRing) {
        let approx_moduli_size = (1 << 58) + 1;
    
        let mut rns_base_components = Vec::new();
        let mut p = max_prime_congruent_one_lt_bound(2 << self.log2_N, approx_moduli_size).unwrap();
        let mut current_bits = (p as f64).log2();
        while current_bits < self.log2_q_max as f64 {
            rns_base_components.push(p);
            p = max_prime_congruent_one_lt_bound(2 << self.log2_N, p).unwrap();
            current_bits += (p as f64).log2();
        }
        current_bits -= (*rns_base_components.last().unwrap() as f64).log2();
        let remaining_size = 2f64.powf(self.log2_q_max as f64 - current_bits).floor() as i64;
        if let Some(p) = max_prime_congruent_one_lt_bound(2 << self.log2_N, remaining_size) {
            rns_base_components.push(p);
            current_bits += (p as f64).log2();
        }
        rns_base_components.reverse();
    
        let rns_base = zn_rns::Zn::new(
            rns_base_components.into_iter().map(|p| p as u64).map(CiphertextZn::new).collect(), 
            ZZbig
        );
        let Q = ZZbig.prod(rns_base.as_iter().map(|Fp: &CiphertextZn| int_cast(*Fp.modulus(), ZZbig, ZZ)));
        assert!(ZZbig.is_geq(&Q, &ZZbig.power_of_two(self.log2_q_min)), "Failed to find a suitable ciphertext modulus within the given range");
        
        let mut current_mul_bits = 0.;
        let primes = (0..)
            .map(|k| approx_moduli_size + (k << (self.log2_N + 1)))
            .filter(|p| is_prime(ZZ, p, 10));
        let rns_base_mul = zn_rns::Zn::new(
            rns_base.as_iter().map(|R: &RingValue<ZnBase>| *R.modulus()).chain(
                primes.take_while(|p| if current_mul_bits >= current_bits as f64 + 1. {
                    false
                } else {
                    current_mul_bits += (*p as f64).log2();
                    true
                })
            ).map(|p| p as u64).map(CiphertextZn::new).collect(), 
            ZZbig
        );
        assert!(ZZbig.is_geq(&ZZbig.prod(rns_base_mul.as_iter().map(|Fp: &CiphertextZn| int_cast(*Fp.modulus(), ZZbig, ZZ))), &ZZbig.pow(ZZbig.mul(Q, ZZbig.int_hom().map(2)), 2)), "Failed to find a suitable ciphertext modulus within the given range");
    
        let allocator = AllocArc(Arc::new(DynLayoutMempool::new_in(Alignment::of::<u128>(), Global)));
        let C = <CiphertextRing as RingStore>::Type::new_with(rns_base.clone(), rns_base.as_iter().map(|R| CiphertextFastmulZn::new(*R)).collect(), self.log2_N, allocator.clone());
        let C_mul = <CiphertextRing as RingStore>::Type::new_with(rns_base_mul.clone(), rns_base_mul.as_iter().map(|R| CiphertextFastmulZn::new(*R)).collect(), self.log2_N, allocator.clone());
        return (C, C_mul);
    }
    
    pub fn create_plaintext_ring(&self) -> PlaintextRing {
        return <PlaintextRing as RingStore>::Type::new(PlaintextZn::new(self.t as u64), self.log2_N);
    }
    
    pub fn create_multiplication_rescale(&self, P: &PlaintextRing, C: &CiphertextRing, C_mul: &CiphertextRing) -> MulConversionData {
        let allocator = C.get_ring().allocator().clone();
        MulConversionData {
            lift_to_C_mul: rnsconv::shared_lift::AlmostExactSharedBaseConversion::new_with(
                C.get_ring().rns_base().as_iter().map(|R| CiphertextZn::new(*R.modulus() as u64)).collect::<Vec<_>>(), 
                Vec::new(),
                C_mul.get_ring().rns_base().as_iter().skip(C.get_ring().rns_base().len()).map(|R| CiphertextZn::new(*R.modulus() as u64)).collect::<Vec<_>>(),
                allocator.clone()
            ),
            scale_down_to_C: rnsconv::bfv_rescale::AlmostExactRescalingConvert::new_with(
                C_mul.get_ring().rns_base().as_iter().map(|R| CiphertextZn::new(*R.modulus() as u64)).collect::<Vec<_>>(), 
                vec![ CiphertextZn::new(*P.base_ring().modulus() as u64) ], 
                C.get_ring().rns_base().len(),
                allocator
            )
        }
    }
}

pub fn debug_dec_print(P: &PlaintextRing, C: &CiphertextRing, ct: &Ciphertext) {
    println!();
    P.println(&dec(P, C, (C.get_ring().clone_el_non_fft(&ct.0), C.get_ring().clone_el_non_fft(&ct.1)), DEBUG_SK.read().unwrap().as_ref().unwrap()));
    println!();
}

pub fn debug_dec_print_slots(P: &PlaintextRing, C: &CiphertextRing, ct: &Ciphertext) {
    let H = HypercubeIsomorphism::new(P.get_ring());
    println!();
    print!("[");
    for x in H.get_slot_values(&dec(P, C, (C.get_ring().clone_el_non_fft(&ct.0), C.get_ring().clone_el_non_fft(&ct.1)), DEBUG_SK.read().unwrap().as_ref().unwrap())) {
        print!("{}, ", H.slot_ring().format(&x));
    }
    println!("]");
    println!();
}

pub fn gen_sk<R: Rng + CryptoRng>(C: &CiphertextRing, mut rng: R) -> SecretKey {
    // we sample uniform ternary secrets 
    let result = C.get_ring().sample_from_coefficient_distribution(|| (rng.next_u32() % 3) as i32 - 1);
    let result = C.get_ring().do_fft(result);
    *DEBUG_SK.write().unwrap() = Some(C.clone_el(&result));
    return result;
}

pub fn enc_sym_zero<R: Rng + CryptoRng>(C: &CiphertextRing, mut rng: R, sk: &SecretKey) -> Ciphertext {
    let a = C.get_ring().sample_uniform(|| rng.next_u64());
    let mut b = C.get_ring().undo_fft(C.negate(C.mul_ref(&a, &sk)));
    let e = C.get_ring().sample_from_coefficient_distribution(|| (rng.sample::<f64, _>(StandardNormal) * 3.2).round() as i32);
    C.get_ring().add_assign_non_fft(&mut b, &e);
    return (b, C.get_ring().undo_fft(a));
}

pub fn enc_sym<R: Rng + CryptoRng>(P: &PlaintextRing, C: &CiphertextRing, rng: R, m: &El<PlaintextRing>, sk: &SecretKey) -> Ciphertext {
    hom_add_plain(P, C, m, enc_sym_zero(C, rng, sk))
}

pub fn remove_noise(P: &PlaintextRing, C: &CiphertextRing, c: &El<CiphertextRing>) -> El<PlaintextRing> {
    let coefficients = C.wrt_canonical_basis(c);
    let Delta = ZZbig.rounded_div(
        ZZbig.clone_el(C.base_ring().modulus()), 
        &int_cast(*P.base_ring().modulus() as i32, &ZZbig, &StaticRing::<i32>::RING)
    );
    let modulo = P.base_ring().can_hom(&ZZbig).unwrap();
    return P.from_canonical_basis((0..coefficients.len()).map(|i| modulo.map(ZZbig.rounded_div(C.base_ring().smallest_lift(coefficients.at(i)), &Delta))));
}

pub fn dec(P: &PlaintextRing, C: &CiphertextRing, ct: Ciphertext, sk: &SecretKey) -> El<PlaintextRing> {
    let (c0, c1) = ct;
    let (c0, c1) = (C.get_ring().do_fft(c0), C.get_ring().do_fft(c1));
    let noisy_m = C.add(c0, C.mul_ref_snd(c1, sk));
    return remove_noise(P, C, &noisy_m);
}

pub fn hom_add(C: &CiphertextRing, lhs: Ciphertext, rhs: &Ciphertext) -> Ciphertext {
    let (mut lhs0, mut lhs1) = lhs;
    let (rhs0, rhs1) = rhs;
    C.get_ring().add_assign_non_fft(&mut lhs0, rhs0);
    C.get_ring().add_assign_non_fft(&mut lhs1, rhs1);
    return (lhs0, lhs1);
}

pub fn hom_sub(C: &CiphertextRing, lhs: Ciphertext, rhs: &Ciphertext) -> Ciphertext {
    let (mut lhs0, mut lhs1) = lhs;
    let (rhs0, rhs1) = rhs;
    C.get_ring().sub_assign_non_fft(&mut lhs0, rhs0);
    C.get_ring().sub_assign_non_fft(&mut lhs1, rhs1);
    return (lhs0, lhs1);
}

pub fn hom_add_plain(P: &PlaintextRing, C: &CiphertextRing, m: &El<PlaintextRing>, ct: Ciphertext) -> Ciphertext {
    let mut m = C.get_ring().exact_convert_from_nttring(P.get_ring(), m);
    let Delta = C.base_ring().coerce(&ZZbig, ZZbig.rounded_div(
        ZZbig.clone_el(C.base_ring().modulus()), 
        &int_cast(*P.base_ring().modulus() as i32, &ZZbig, &StaticRing::<i32>::RING)
    ));
    C.get_ring().mul_scalar_assign_non_fft(&mut m, &Delta);
    let (mut c0, c1) = ct;
    C.get_ring().add_assign_non_fft(&mut c0, &m);
    return (c0, c1);
}

pub fn hom_mul_plain(P: &PlaintextRing, C: &CiphertextRing, m: &El<PlaintextRing>, ct: Ciphertext) -> Ciphertext {
    let m = C.get_ring().do_fft(C.get_ring().exact_convert_from_nttring(P.get_ring(), m));
    let (c0, c1) = ct;
    let (c0, c1) = (C.get_ring().do_fft(c0), C.get_ring().do_fft(c1));
    return (C.get_ring().undo_fft(C.mul_ref_snd(c0, &m)), C.get_ring().undo_fft(C.mul(c1, m)));
}

pub fn noise_budget(P: &PlaintextRing, C: &CiphertextRing, ct: &Ciphertext, sk: &SecretKey) -> usize {
    let (c0, c1) = ct;
    let (c0, c1) = (C.get_ring().do_fft(C.get_ring().clone_el_non_fft(c0)), C.get_ring().do_fft(C.get_ring().clone_el_non_fft(c1)));
    let noisy_m = C.add(c0, C.mul_ref_snd(c1, sk));
    let coefficients = C.wrt_canonical_basis(&noisy_m);
    let Delta = ZZbig.rounded_div(
        ZZbig.clone_el(C.base_ring().modulus()), 
        &int_cast(*P.base_ring().modulus() as i32, &ZZbig, &StaticRing::<i32>::RING)
    );
    return ZZbig.abs_log2_ceil(C.base_ring().modulus()).unwrap().saturating_sub((0..coefficients.len()).map(|i| {
        let c = C.base_ring().smallest_lift(coefficients.at(i));
        let size = ZZbig.abs_log2_ceil(&ZZbig.sub_ref_fst(&c, ZZbig.mul_ref_snd(ZZbig.rounded_div(ZZbig.clone_el(&c), &Delta), &Delta)));
        return size.unwrap_or(0);
    }).max().unwrap() + P.base_ring().integer_ring().abs_log2_ceil(P.base_ring().modulus()).unwrap() + 1);
}

pub fn gen_rk<'a, R: Rng + CryptoRng>(C: &'a CiphertextRing, rng: R, sk: &SecretKey) -> RelinKey<'a> {
    gen_switch_key(C, rng, &C.pow(C.clone_el(sk), 2), sk)
}

pub fn hom_mul(C: &CiphertextRing, C_mul: &CiphertextRing, lhs: &Ciphertext, rhs: &Ciphertext, rk: &RelinKey, conv_data: &MulConversionData) -> Ciphertext {
    let (c00, c01) = lhs;
    let (c10, c11) = rhs;
    let lift = |c: &DoubleRNSNonFFTEl<CiphertextZn, CiphertextGenFFT, CiphertextAllocator>| {
        // let c_fft = C.get_ring().do_fft(C.get_ring().clone_el_non_fft(c));
        // let input_wrt_basis = C.wrt_canonical_basis(&c_fft);
        let result = C_mul.get_ring().do_fft(C_mul.get_ring().perform_rns_op_from(C.get_ring(), &c, &conv_data.lift_to_C_mul));
        // let result_wrt_basis = C_mul.wrt_canonical_basis(&result);
        // for i in 0..C.rank() {
        //     let actual = C_mul.base_ring().smallest_lift(result_wrt_basis.at(i));
        //     let expected = C.base_ring().smallest_lift(input_wrt_basis.at(i));
        //     assert!(ZZbig.is_leq(&ZZbig.abs(ZZbig.sub_ref(&actual, &expected)), C.base_ring().modulus()), "{}, {}, {}", ZZbig.format(&actual), ZZbig.format(&expected), ZZbig.format(C.base_ring().modulus()));
        // }
        return result;
    };

    let lifted0 = C_mul.mul(lift(c00), lift(c10));
    let lifted1 = C_mul.add(C_mul.mul(lift(c00), lift(c11)), C_mul.mul(lift(c01), lift(c10)));
    let lifted2 = C_mul.mul(lift(c01), lift(c11));

    let scale_down = |c: El<CiphertextRing>| C.get_ring().perform_rns_op_from(C_mul.get_ring(), &C_mul.get_ring().undo_fft(c), &conv_data.scale_down_to_C);

    let mut res0 = scale_down(lifted0);
    let mut res1 = scale_down(lifted1);
    let res2 = scale_down(lifted2);
    
    let op = C.get_ring().to_gadget_product_lhs(res2);
    let (s0, s1) = rk;

    C.get_ring().add_assign_non_fft(&mut res0, &C.get_ring().gadget_product_base(&op, s0));
    C.get_ring().add_assign_non_fft(&mut res1, &C.get_ring().gadget_product_base(&op, s1));
    return (res0, res1);
}

pub fn gen_switch_key<'a, R: Rng + CryptoRng>(C: &'a CiphertextRing, mut rng: R, old_sk: &SecretKey, new_sk: &SecretKey) -> KeySwitchKey<'a> {
    let old_sk_non_fft = C.get_ring().undo_fft(C.clone_el(old_sk));
    let mut res_0 = C.get_ring().gadget_product_rhs_empty();
    let mut res_1 = C.get_ring().gadget_product_rhs_empty();
    for i in 0..C.get_ring().rns_base().len() {
        let (c0, c1) = enc_sym_zero(C, &mut rng, new_sk);
        let factor = C.base_ring().get_ring().from_congruence((0..C.get_ring().rns_base().len()).map(|i2| {
            let Fp = C.get_ring().rns_base().at(i2);
            if i2 == i { Fp.one() } else { Fp.zero() } 
        }));
        let mut payload = C.get_ring().clone_el_non_fft(&old_sk_non_fft);
        C.get_ring().mul_scalar_assign_non_fft(&mut payload, &factor);
        C.get_ring().add_assign_non_fft(&mut payload, &c0);
        res_0.set_rns_factor(i, payload);
        res_1.set_rns_factor(i, c1);
    }
    return (res_0, res_1);
}

pub fn key_switch(C: &CiphertextRing, ct: &Ciphertext, switch_key: &KeySwitchKey) -> Ciphertext {
    let (c0, c1) = ct;
    let (s0, s1) = switch_key;
    let op = C.get_ring().to_gadget_product_lhs(C.get_ring().clone_el_non_fft(c1));
    let mut r0 = C.get_ring().gadget_product_base(&op, s0);
    C.get_ring().add_assign_non_fft(&mut r0, c0);
    let r1 = C.get_ring().gadget_product_base(&op, s1);
    return (r0, r1);
}

pub fn mod_switch_to_plaintext(target: &PlaintextRing, C: &CiphertextRing, ct: &Ciphertext, switch_data: &ModSwitchData) -> (El<PlaintextRing>, El<PlaintextRing>) {
    let (c0, c1) = ct;
    return (
        C.get_ring().perform_rns_op_to_nttring::<Pow2CyclotomicFFT<PlaintextZn, PlaintextFFT>, _, _>(target.get_ring(), c0, &switch_data.scale),
        C.get_ring().perform_rns_op_to_nttring::<Pow2CyclotomicFFT<PlaintextZn, PlaintextFFT>, _, _>(target.get_ring(), c1, &switch_data.scale)
    );
}

pub fn gen_gk<'a, R: Rng + CryptoRng>(C: &'a CiphertextRing, rng: R, sk: &SecretKey, g: ZnEl) -> KeySwitchKey<'a> {
    gen_switch_key(C, rng, &C.get_ring().apply_galois_action(sk, g), sk)
}

pub fn hom_galois(C: &CiphertextRing, ct: Ciphertext, g: ZnEl, gk: &KeySwitchKey) -> Ciphertext {
    key_switch(C, &(
        C.get_ring().undo_fft(C.get_ring().apply_galois_action(&C.get_ring().do_fft(ct.0), g)),
        C.get_ring().undo_fft(C.get_ring().apply_galois_action(&C.get_ring().do_fft(ct.1), g))
    ), gk)
}

#[test]
#[ignore]
fn run_bfv() {
    let mut rng = thread_rng();
    
    let params = Pow2BFVParams {
        t: 3,
        log2_q_min: 790,
        log2_q_max: 800,
        log2_N: 15
    };
    
    let P = params.create_plaintext_ring();
    let (C, C_mul) = params.create_ciphertext_rings();
    println!("Created rings, RNS base length is {}", C.base_ring().get_ring().len());
    
    let sk = gen_sk(&C, &mut rng);
    
    let m = P.int_hom().map(2);
    let ct = enc_sym(&P, &C, &mut rng, &m, &sk);
    println!("Encrypted message");
    
    let mul_rescale_data = params.create_multiplication_rescale(&P, &C, &C_mul);
    let relin_key = gen_rk(&C, &mut rng, &sk);
    println!("Created relin key");

    const COUNT: usize = 10;

    clear_all_timings();
    let start = Instant::now();
    let mut ct_sqr = hom_mul(&C, &C_mul, &ct, &ct, &relin_key, &mul_rescale_data);
    for _ in 0..(COUNT - 1) {
        ct_sqr = hom_mul(&C, &C_mul, &ct, &ct, &relin_key, &mul_rescale_data);
    }
    let end = Instant::now();
    println!("Performed multiplication in {} ms", (end - start).as_millis() as f64 / COUNT as f64);
    print_all_timings();

    let m_sqr = dec(&P, &C, ct_sqr, &sk);
    println!("Decrypted result");
    assert_el_eq!(&P, &P.one(), &m_sqr);
}

#[test]
fn test_hom_galois() {
    let mut rng = thread_rng();
    
    let params = Pow2BFVParams {
        t: 3,
        log2_q_min: 100,
        log2_q_max: 120,
        log2_N: 7
    };
    
    let P = params.create_plaintext_ring();
    let (C, _C_mul) = params.create_ciphertext_rings();    
    let sk = gen_sk(&C, &mut rng);
    let gk = gen_gk(&C, &mut rng, &sk, P.get_ring().galois_group_mulrepr().int_hom().map(3));
    
    let m = P.canonical_gen();
    let ct = enc_sym(&P, &C, &mut rng, &m, &sk);
    let ct_res = hom_galois(&C, ct, P.get_ring().galois_group_mulrepr().int_hom().map(3), &gk);
    let res = dec(&P, &C, ct_res, &sk);

    assert_el_eq!(&P, &P.pow(P.canonical_gen(), 3), &res);
}

#[test]
fn test_bfv_mul() {
    let mut rng = thread_rng();
    
    let params = Pow2BFVParams {
        t: 257,
        log2_q_min: 1090,
        log2_q_max: 1100,
        log2_N: 10
    };
    
    let P = params.create_plaintext_ring();
    let (C, C_mul) = params.create_ciphertext_rings();

    for Fp in C_mul.base_ring().as_iter() {
        ZZ.println(Fp.modulus());
    }
    println!("{}", C.base_ring().len());

    let sk = gen_sk(&C, &mut rng);
    let mul_rescale_data = params.create_multiplication_rescale(&P, &C, &C_mul);
    let rk = gen_rk(&C, &mut rng, &sk);

    let m = P.int_hom().map(2);
    let ct = enc_sym(&P, &C, &mut rng, &m, &sk);

    let m = dec(&P, &C, (C.get_ring().clone_el_non_fft(&ct.0), C.get_ring().clone_el_non_fft(&ct.1)), &sk);
    assert_el_eq!(&P, &P.int_hom().map(2), &m);

    let ct_sqr = hom_mul(&C, &C_mul, &ct, &ct, &rk, &mul_rescale_data);
    let m_sqr = dec(&P, &C, ct_sqr, &sk);

    assert_el_eq!(&P, &P.int_hom().map(4), &m_sqr);
}
