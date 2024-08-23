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

type Ciphertext = (RingEl, RingEl);
pub type SecretKey = El<CiphertextRing>;
pub type GadgetProductOperand<'a> = GadgetProductRhsOperand<'a, CiphertextGenFFT, CiphertextAllocator>;
pub type KeySwitchKey<'a> = (GadgetProductOperand<'a>, GadgetProductOperand<'a>);
pub type RelinKey<'a> = (GadgetProductOperand<'a>, GadgetProductOperand<'a>);

static DEBUG_SK: RwLock<Option<SecretKey>> = RwLock::new(None);

pub struct RingEl {
    ntt_part: Option<DoubleRNSEl<CiphertextZn, CiphertextGenFFT, CiphertextAllocator>>,
    coeff_part: Option<DoubleRNSNonFFTEl<CiphertextZn, CiphertextGenFFT, CiphertextAllocator>>
}

impl RingEl {

    pub fn ntt_repr(self, C: &CiphertextRing) -> Self {
        RingEl::from_ntt(self.to_ntt(C))
    }

    pub fn coeff_repr(self, C: &CiphertextRing) -> Self {
        RingEl::from_coeff(self.to_coeff(C))
    }

    pub fn from_ntt(el: El<CiphertextRing>) -> Self {
        Self {
            coeff_part: None,
            ntt_part: Some(el)
        }
    }

    pub fn from_coeff(el: DoubleRNSNonFFTEl<CiphertextZn, CiphertextGenFFT, CiphertextAllocator>) -> Self {
        Self {
            coeff_part: Some(el),
            ntt_part: None
        }
    }

    pub fn zero() -> Self {
        Self {
            coeff_part: None,
            ntt_part: None
        }
    }

    pub fn to_ntt(self, C: &CiphertextRing) -> El<CiphertextRing> {
        if let Some(mut result) = self.ntt_part {
            if let Some(coeff) = self.coeff_part {
                C.add_assign(&mut result, C.get_ring().do_fft(coeff));
            }
            return result;
        } else if let Some(coeff) = self.coeff_part {
            return C.get_ring().do_fft(coeff);
        } else {
            return C.zero();
        }
    }

    pub fn to_coeff(self, C: &CiphertextRing) -> DoubleRNSNonFFTEl<CiphertextZn, CiphertextGenFFT, CiphertextAllocator> {
        if let Some(mut result) = self.coeff_part {
            if let Some(ntt_part) = self.ntt_part {
                C.get_ring().add_assign_non_fft(&mut result, &C.get_ring().undo_fft(ntt_part));
            }
            return result;
        } else if let Some(ntt_part) = self.ntt_part {
            return C.get_ring().undo_fft(ntt_part);
        } else {
            return C.get_ring().non_fft_zero();
        }
    }

    pub fn gadget_product<'a>(lhs: &GadgetProductLhsOperand<'a, CiphertextGenFFT, CiphertextAllocator>, rhs: &GadgetProductRhsOperand<'a, CiphertextGenFFT, CiphertextAllocator>, C: &CiphertextRing) -> RingEl {
        match C.get_ring().preferred_output_repr(lhs, rhs) {
            ElRepr::Coeff => RingEl { ntt_part: None, coeff_part: Some(C.get_ring().gadget_product_coeff(lhs, rhs)) },
            ElRepr::NTT => RingEl { ntt_part: Some(C.get_ring().gadget_product_ntt(lhs, rhs)), coeff_part: None },
        }
    }

    pub fn add(lhs: RingEl, rhs: &RingEl, C: &CiphertextRing) -> RingEl {
        RingEl {
            ntt_part: if lhs.ntt_part.is_some() && rhs.ntt_part.is_some() { Some(C.add_ref_snd(lhs.ntt_part.unwrap(), rhs.ntt_part.as_ref().unwrap())) } else { lhs.ntt_part.or(rhs.ntt_part.as_ref().map(|x| C.clone_el(x)))},
            coeff_part: if lhs.coeff_part.is_some() && rhs.coeff_part.is_some() {
                let mut result  = lhs.coeff_part.unwrap();
                C.get_ring().add_assign_non_fft(&mut result, rhs.coeff_part.as_ref().unwrap());
                Some(result)
            } else { lhs.coeff_part.or(rhs.coeff_part.as_ref().map(|x| C.get_ring().clone_el_non_fft(x))) }
        }
    }

    pub fn sub(lhs: RingEl, rhs: &RingEl, C: &CiphertextRing) -> RingEl {
        RingEl {
            ntt_part: if lhs.ntt_part.is_some() && rhs.ntt_part.is_some() { Some(C.sub_ref_snd(lhs.ntt_part.unwrap(), rhs.ntt_part.as_ref().unwrap())) } else { lhs.ntt_part.or(rhs.ntt_part.as_ref().map(|x| C.negate(C.clone_el(x))))},
            coeff_part: if lhs.coeff_part.is_some() && rhs.coeff_part.is_some() {
                let mut result  = lhs.coeff_part.unwrap();
                C.get_ring().sub_assign_non_fft(&mut result, rhs.coeff_part.as_ref().unwrap());
                Some(result)
            } else { lhs.coeff_part.or(rhs.coeff_part.as_ref().map(|x| C.get_ring().negate_non_fft(C.get_ring().clone_el_non_fft(x)))) }
        }
    }

    pub fn mul_i64(mut val: RingEl, scalar: i64, C: &CiphertextRing) -> RingEl {
        let hom = C.base_ring().can_hom(&StaticRing::<i64>::RING).unwrap();
        if let Some(ntt_part) = &mut val.ntt_part {
            C.inclusion().mul_assign_map(ntt_part, hom.map(scalar));
        }
        if let Some(coeff_part) = &mut val.coeff_part {
            C.get_ring().mul_scalar_assign_non_fft(coeff_part, &hom.map(scalar));
        }
        return val;
    }

    pub fn clone(&self, C: &CiphertextRing) -> RingEl {
        RingEl { 
            ntt_part: self.ntt_part.as_ref().map(|x| C.clone_el(x)), 
            coeff_part: self.coeff_part.as_ref().map(|x| C.get_ring().clone_el_non_fft(x))
        }
    }
}

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
    P.println(&dec(P, C, clone_ct(C, ct), DEBUG_SK.read().unwrap().as_ref().unwrap()));
    println!();
}

pub fn debug_dec_print_slots(P: &PlaintextRing, C: &CiphertextRing, ct: &Ciphertext) {
    let H = HypercubeIsomorphism::new(P.get_ring());
    println!();
    print!("[");
    for x in H.get_slot_values(&dec(P, C, clone_ct(C, ct), DEBUG_SK.read().unwrap().as_ref().unwrap())) {
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
    return (RingEl::from_coeff(b), RingEl::from_ntt(a));
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
    let noisy_m = C.add(c0.to_ntt(C), C.mul_ref_snd(c1.to_ntt(C), sk));
    return remove_noise(P, C, &noisy_m);
}

pub fn hom_add(C: &CiphertextRing, lhs: Ciphertext, rhs: &Ciphertext) -> Ciphertext {
    let (lhs0, lhs1) = lhs;
    let (rhs0, rhs1) = rhs;
    return (RingEl::add(lhs0, rhs0, C), RingEl::add(lhs1, rhs1, C));
}

pub fn hom_sub(C: &CiphertextRing, lhs: Ciphertext, rhs: &Ciphertext) -> Ciphertext {
    let (lhs0, lhs1) = lhs;
    let (rhs0, rhs1) = rhs;
    return (RingEl::sub(lhs0, rhs0, C), RingEl::sub(lhs1, rhs1, C));
}

pub fn clone_ct(C: &CiphertextRing, ct: &Ciphertext) -> Ciphertext {
    return (ct.0.clone(C), ct.1.clone(C));
}

pub fn hom_add_plain(P: &PlaintextRing, C: &CiphertextRing, m: &El<PlaintextRing>, ct: Ciphertext) -> Ciphertext {
    let mut m = C.get_ring().exact_convert_from_nttring(P.get_ring(), m);
    let Delta = C.base_ring().coerce(&ZZbig, ZZbig.rounded_div(
        ZZbig.clone_el(C.base_ring().modulus()), 
        &int_cast(*P.base_ring().modulus() as i32, &ZZbig, &StaticRing::<i32>::RING)
    ));
    C.get_ring().mul_scalar_assign_non_fft(&mut m, &Delta);
    return (RingEl::add(ct.0, &RingEl::from_coeff(m), C), ct.1);
}

pub fn hom_mul_plain(P: &PlaintextRing, C: &CiphertextRing, m: &El<PlaintextRing>, ct: Ciphertext) -> Ciphertext {
    let m = C.get_ring().do_fft(C.get_ring().exact_convert_from_nttring(P.get_ring(), m));
    let c0 = ct.0.to_ntt(C);
    let c1 = ct.1.to_ntt(C);
    return (RingEl::from_ntt(C.mul_ref_snd(c0, &m)), RingEl::from_ntt(C.mul(c1, m)));
}

pub fn hom_mul_plain_i64(_P: &PlaintextRing, C: &CiphertextRing, m: i64, ct: Ciphertext) -> Ciphertext {
    (RingEl::mul_i64(ct.0, m, C), RingEl::mul_i64(ct.1, m, C))
}

pub fn noise_budget(P: &PlaintextRing, C: &CiphertextRing, ct: &Ciphertext, sk: &SecretKey) -> usize {
    let (c0, c1) = clone_ct(C, ct);
    let (c0, c1) = (c0.to_ntt(C), c1.to_ntt(C));
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

pub fn hom_mul(C: &CiphertextRing, C_mul: &CiphertextRing, lhs: Ciphertext, rhs: Ciphertext, rk: &RelinKey, conv_data: &MulConversionData) -> Ciphertext {
    let (c00, c01) = lhs;
    let (c10, c11) = rhs;
    let lift = |c: DoubleRNSNonFFTEl<CiphertextZn, CiphertextGenFFT, CiphertextAllocator>| 
        C_mul.get_ring().do_fft(C_mul.get_ring().perform_rns_op_from(C.get_ring(), &c, &conv_data.lift_to_C_mul));

    let c00_lifted = lift(c00.to_coeff(C));
    let c01_lifted = lift(c01.to_coeff(C));
    let c10_lifted = lift(c10.to_coeff(C));
    let c11_lifted = lift(c11.to_coeff(C));

    let lifted0 = C_mul.mul_ref(&c00_lifted, &c10_lifted);
    let lifted1 = C_mul.add(C_mul.mul_ref_snd(c00_lifted, &c11_lifted), C_mul.mul_ref_fst(&c01_lifted, c10_lifted));
    let lifted2 = C_mul.mul(c01_lifted, c11_lifted);

    let scale_down = |c: El<CiphertextRing>| 
        C.get_ring().perform_rns_op_from(C_mul.get_ring(), &C_mul.get_ring().undo_fft(c), &conv_data.scale_down_to_C);

    let res0 = scale_down(lifted0);
    let res1 = scale_down(lifted1);
    let res2 = scale_down(lifted2);

    let op = C.get_ring().to_gadget_product_lhs(res2);
    let (s0, s1) = rk;

    return (RingEl::add(RingEl::from_coeff(res0), &RingEl::gadget_product(&op, s0, C), C), RingEl::add(RingEl::from_coeff(res1), &RingEl::gadget_product(&op, s1, C), C));
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
        C.get_ring().add_assign_non_fft(&mut payload, &c0.to_coeff(C));
        res_0.set_rns_factor(i, payload);
        res_1.set_rns_factor(i, c1.to_coeff(C));
    }
    return (res_0, res_1);
}

pub fn key_switch(C: &CiphertextRing, ct: Ciphertext, switch_key: &KeySwitchKey) -> Ciphertext {
    let (c0, c1) = ct;
    let (s0, s1) = switch_key;
    let op = C.get_ring().to_gadget_product_lhs(c1.to_coeff(C));
    return (
        RingEl::add(c0, &RingEl::gadget_product(&op, s0, C), C),
        RingEl::gadget_product(&op, s1, C)
    );
}

pub fn mod_switch_to_plaintext(target: &PlaintextRing, C: &CiphertextRing, ct: Ciphertext, switch_data: &ModSwitchData) -> (El<PlaintextRing>, El<PlaintextRing>) {
    let (c0, c1) = ct;
    return (
        C.get_ring().perform_rns_op_to_nttring::<Pow2CyclotomicFFT<PlaintextZn, PlaintextFFT>, _, _>(target.get_ring(), &c0.to_coeff(C), &switch_data.scale),
        C.get_ring().perform_rns_op_to_nttring::<Pow2CyclotomicFFT<PlaintextZn, PlaintextFFT>, _, _>(target.get_ring(), &c1.to_coeff(C), &switch_data.scale)
    );
}

pub fn gen_gk<'a, R: Rng + CryptoRng>(C: &'a CiphertextRing, rng: R, sk: &SecretKey, g: ZnEl) -> KeySwitchKey<'a> {
    gen_switch_key(C, rng, &C.get_ring().apply_galois_action(sk, g), sk)
}

pub fn hom_galois(C: &CiphertextRing, ct: Ciphertext, g: ZnEl, gk: &KeySwitchKey) -> Ciphertext {
    key_switch(C, (
        RingEl::from_ntt(C.get_ring().apply_galois_action(&ct.0.to_ntt(C), g)),
        RingEl::from_ntt(C.get_ring().apply_galois_action(&ct.1.to_ntt(C), g))
    ), gk)
}

pub fn hom_galois_many<'a, V>(C: &CiphertextRing, ct: Ciphertext, gs: &[ZnEl], gks: V) -> Vec<Ciphertext>
    where V: VectorView<KeySwitchKey<'a>>
{
    let (c0, c1) = ct;
    let c0_ntt = c0.to_ntt(&C);
    let lhs = C.get_ring().to_gadget_product_lhs(c1.to_coeff(&C));
    return (0..gs.len()).map(|i| {
        let c1_g = lhs.apply_galois_action(C.get_ring(), gs[i]);
        let (s0, s1) = gks.at(i);
        let r0 = RingEl::gadget_product(&c1_g, s0, C);
        let r1 = RingEl::gadget_product(&c1_g, s1, C);
        let c0_g = RingEl::from_ntt(C.get_ring().apply_galois_action(&c0_ntt, gs[i]));
        return (RingEl::add(r0, &c0_g, C), r1);
    }).collect();
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
    let mut ct_sqr = hom_mul(&C, &C_mul, clone_ct(&C, &ct), clone_ct(&C, &ct), &relin_key, &mul_rescale_data);
    for _ in 0..(COUNT - 1) {
        ct_sqr = hom_mul(&C, &C_mul, clone_ct(&C, &ct), clone_ct(&C, &ct), &relin_key, &mul_rescale_data);
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

    let sk = gen_sk(&C, &mut rng);
    let mul_rescale_data = params.create_multiplication_rescale(&P, &C, &C_mul);
    let rk = gen_rk(&C, &mut rng, &sk);

    let m = P.int_hom().map(2);
    let ct = enc_sym(&P, &C, &mut rng, &m, &sk);

    let m = dec(&P, &C, clone_ct(&C, &ct), &sk);
    assert_el_eq!(&P, &P.int_hom().map(2), &m);

    let ct_sqr = hom_mul(&C, &C_mul, clone_ct(&C, &ct), clone_ct(&C, &ct), &rk, &mul_rescale_data);
    let m_sqr = dec(&P, &C, ct_sqr, &sk);

    assert_el_eq!(&P, &P.int_hom().map(4), &m_sqr);
}
