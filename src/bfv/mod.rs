#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use std::alloc::Global;
use std::marker::PhantomData;
use std::time::Instant;
use std::ops::Range;
use std::cmp::max;

use feanor_math::algorithms::int_factor::factor;
use feanor_math::algorithms::miller_rabin::is_prime;
use feanor_math::algorithms::unity_root::get_prim_root_of_unity;
use feanor_math::algorithms::unity_root::get_prim_root_of_unity_pow2;
use feanor_math::homomorphism::CanHom;
use feanor_math::homomorphism::Identity;
use feanor_math::ring::*;
use feanor_math::assert_el_eq;
use feanor_math::rings::zn::*;
use feanor_math::rings::zn::zn_64::*;
use feanor_math::primitive_int::StaticRing;
use feanor_math::integer::*;
use feanor_math::algorithms::fft::*;
use feanor_math::homomorphism::Homomorphism;
use feanor_math::rings::extension::FreeAlgebraStore;
use feanor_math::seq::*;
use feanor_math::ordered::OrderedRingStore;
use feanor_math::pid::EuclideanRingStore;

use crate::cyclotomic::*;
use crate::euler_phi;
use crate::extend_sampled_primes;
use crate::rings::double_rns_managed::*;
use crate::rings::number_ring::*;
use crate::rings::decomposition_ring::*;
use crate::rings::odd_cyclotomic::*;
use crate::rings::pow2_cyclotomic::*;
use crate::rings::gadget_product::managed::*;
use crate::profiling::*;
use crate::rings::slots::HypercubeIsomorphism;
use crate::rnsconv;
use crate::sample_primes;

use rand::thread_rng;
use rand::{Rng, CryptoRng};
use rand_distr::StandardNormal;

pub mod bootstrap;
pub mod single_rns;

pub type PlaintextAllocator = Global;
pub type CiphertextAllocator = Global;
pub type PlaintextRing<Params: BFVParams> = DecompositionRing<Params::NumberRing, Zn, PlaintextAllocator>;
pub type SecretKey<Params: BFVParams> = El<CiphertextRing<Params>>;
pub type KeySwitchKey<'a, Params: BFVParams> = (usize, (GadgetProductOperand<'a, Params>, GadgetProductOperand<'a, Params>));
pub type RelinKey<'a, Params: BFVParams> = KeySwitchKey<'a, Params>;
pub type CiphertextRing<Params: BFVParams> = ManagedDoubleRNSRing<Params::NumberRing, Zn, Global>;
pub type Ciphertext<Params: BFVParams> = (ManagedDoubleRNSEl<Params::NumberRing, Zn, CiphertextAllocator>, ManagedDoubleRNSEl<Params::NumberRing, Zn, CiphertextAllocator>);
pub type GadgetProductOperand<'a, Params: BFVParams> = GadgetProductRhsOperand<'a, Params::NumberRing, CiphertextAllocator>;

const ZZbig: BigIntRing = BigIntRing::RING;
const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;

pub struct MulConversionData {
    lift_to_C_mul: rnsconv::shared_lift::AlmostExactSharedBaseConversion<CiphertextAllocator>,
    scale_down_to_C: rnsconv::bfv_rescale::AlmostExactRescalingConvert<CiphertextAllocator>
}

pub struct ModSwitchData {
    scale: rnsconv::bfv_rescale::AlmostExactRescaling<CiphertextAllocator>
}

pub trait BFVParams {
    
    type NumberRing: HECyclotomicNumberRing<Zn>;

    fn ciphertext_modulus_bits(&self) -> Range<usize>;
    fn number_ring(&self) -> Self::NumberRing;

    fn create_plaintext_ring(&self, modulus: i64) -> PlaintextRing<Self> {
        DecompositionRingBase::new(self.number_ring(), Zn::new(modulus as u64))
    }

    fn create_ciphertext_rings(&self) -> (CiphertextRing<Self>, CiphertextRing<Self>) {
        let log2_q = self.ciphertext_modulus_bits();
        let number_ring = self.number_ring();

        let mut C_rns_base = sample_primes(log2_q.start, log2_q.end, 56, |bound| number_ring.largest_suitable_prime(int_cast(bound, ZZ, ZZbig)).map(|p| int_cast(p, ZZbig, ZZ))).unwrap();
        C_rns_base.sort_unstable_by(|l, r| ZZbig.cmp(l, r));

        let mut Cmul_rns_base = extend_sampled_primes(&C_rns_base, log2_q.end * 2, log2_q.end * 2 + 57, 57, |bound| number_ring.largest_suitable_prime(int_cast(bound, ZZ, ZZbig)).map(|p| int_cast(p, ZZbig, ZZ))).unwrap();
        assert!(ZZbig.is_gt(&Cmul_rns_base[Cmul_rns_base.len() - 1], &C_rns_base[C_rns_base.len() - 1]));
        Cmul_rns_base.sort_unstable_by(|l, r| ZZbig.cmp(l, r));

        let C = ManagedDoubleRNSRingBase::new(
            self.number_ring(),
            zn_rns::Zn::new(C_rns_base.iter().map(|p| Zn::new(int_cast(ZZbig.clone_el(p), ZZ, ZZbig) as u64)).collect(), ZZbig)
        );
        let Cmul = ManagedDoubleRNSRingBase::new(
            number_ring,
            zn_rns::Zn::new(Cmul_rns_base.iter().map(|p| Zn::new(int_cast(ZZbig.clone_el(p), ZZ, ZZbig) as u64)).collect(), ZZbig)
        );
        return (C, Cmul);
    }

    fn gen_sk<R: Rng + CryptoRng>(C: &CiphertextRing<Self>, mut rng: R) -> SecretKey<Self> {
        // we sample uniform ternary secrets 
        let result = C.get_ring().sample_from_coefficient_distribution(|| (rng.next_u32() % 3) as i32 - 1);
        return result;
    }
    
    fn enc_sym_zero<R: Rng + CryptoRng>(C: &CiphertextRing<Self>, mut rng: R, sk: &SecretKey<Self>) -> Ciphertext<Self> {
        let a = C.get_ring().sample_uniform(|| rng.next_u64());
        let mut b: ManagedDoubleRNSEl<<Self as BFVParams>::NumberRing, RingValue<ZnBase>> = C.negate(C.mul_ref(&a, &sk));
        let e = C.get_ring().sample_from_coefficient_distribution(|| (rng.sample::<f64, _>(StandardNormal) * 3.2).round() as i32);
        C.add_assign(&mut b, e);
        return (b, a);
    }
    
    fn transparent_zero(C: &CiphertextRing<Self>) -> Ciphertext<Self> {
        (C.zero(), C.zero())
    }

    fn enc_sym<R: Rng + CryptoRng>(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, rng: R, m: &El<PlaintextRing<Self>>, sk: &SecretKey<Self>) -> Ciphertext<Self> {
        Self::hom_add_plain(P, C, m, Self::enc_sym_zero(C, rng, sk))
    }

    fn enc_sk(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>) -> Ciphertext<Self> {
        let Delta = ZZbig.rounded_div(
            ZZbig.clone_el(C.base_ring().modulus()), 
            &int_cast(*P.base_ring().modulus() as i32, &ZZbig, &StaticRing::<i32>::RING)
        );
        (C.zero(), C.inclusion().map(C.base_ring().coerce(&ZZbig, Delta)))
    }
    
    fn remove_noise(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, c: &El<CiphertextRing<Self>>) -> El<PlaintextRing<Self>> {
        let coefficients = C.wrt_canonical_basis(c);
        let Delta = ZZbig.rounded_div(
            ZZbig.clone_el(C.base_ring().modulus()), 
            &int_cast(*P.base_ring().modulus() as i32, &ZZbig, &StaticRing::<i32>::RING)
        );
        let modulo = P.base_ring().can_hom(&ZZbig).unwrap();
        return P.from_canonical_basis((0..coefficients.len()).map(|i| modulo.map(ZZbig.rounded_div(C.base_ring().smallest_lift(coefficients.at(i)), &Delta))));
    }
    
    fn dec(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: Ciphertext<Self>, sk: &SecretKey<Self>) -> El<PlaintextRing<Self>> {
        let (c0, c1) = ct;
        let noisy_m = C.add(c0, C.mul_ref_snd(c1, sk));
        return Self::remove_noise(P, C, &noisy_m);
    }
    
    fn dec_println(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: &Ciphertext<Self>, sk: &SecretKey<Self>) {
        let m = Self::dec(P, C, Self::clone_ct(C, ct), sk);
        P.println(&m);
        println!();
    }
    
    fn dec_println_slots(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: &Ciphertext<Self>, sk: &SecretKey<Self>) {
        let H = HypercubeIsomorphism::new::<false>(P.get_ring());
        let m = Self::dec(P, C, Self::clone_ct(C, ct), sk);
        for a in H.get_slot_values(&m) {
            H.slot_ring().println(&a);
        }
        println!();
    }
    
    fn hom_add(C: &CiphertextRing<Self>, lhs: Ciphertext<Self>, rhs: &Ciphertext<Self>) -> Ciphertext<Self> {
        let (lhs0, lhs1) = lhs;
        let (rhs0, rhs1) = rhs;
        return (C.add_ref(&lhs0, &rhs0), C.add_ref(&lhs1, &rhs1));
    }
    
    fn hom_sub(C: &CiphertextRing<Self>, lhs: Ciphertext<Self>, rhs: &Ciphertext<Self>) -> Ciphertext<Self> {
        let (lhs0, lhs1) = lhs;
        let (rhs0, rhs1) = rhs;
        return (C.sub_ref(&lhs0, rhs0), C.sub_ref(&lhs1, rhs1));
    }
    
    fn clone_ct(C: &CiphertextRing<Self>, ct: &Ciphertext<Self>) -> Ciphertext<Self> {
        (C.clone_el(&ct.0), C.clone_el(&ct.1))
    }
    
    fn hom_add_plain(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: &El<PlaintextRing<Self>>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        let mut m = C.get_ring().exact_convert_from_decompring(P, m);
        let Delta = C.base_ring().coerce(&ZZbig, ZZbig.rounded_div(
            ZZbig.clone_el(C.base_ring().modulus()), 
            &int_cast(*P.base_ring().modulus() as i32, &ZZbig, &StaticRing::<i32>::RING)
        ));
        C.inclusion().mul_assign_ref_map(&mut m, &Delta);
        return (C.add(ct.0, m), ct.1);
    }
    
    fn hom_mul_plain(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: &El<PlaintextRing<Self>>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        let m = C.get_ring().exact_convert_from_decompring(P, m);
        let (c0, c1) = ct;
        return (C.mul_ref_snd(c0, &m), C.mul(c1, m));
    }
    
    fn hom_mul_plain_i64(_P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: i64, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        (C.int_hom().mul_map(ct.0, m as i32), C.int_hom().mul_map(ct.1, m as i32))
    }
    
    fn noise_budget(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: &Ciphertext<Self>, sk: &SecretKey<Self>) -> usize {
        let (c0, c1) = Self::clone_ct(C, ct);
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
    
    fn gen_rk<'a, R: Rng + CryptoRng, const LOG: bool>(C: &'a CiphertextRing<Self>, rng: R, sk: &SecretKey<Self>, digits: usize) -> RelinKey<'a, Self>
        where Self: 'a
    {
        Self::gen_switch_key::<_, LOG>(C, rng, &C.pow(C.clone_el(sk), 2), sk, digits)
    }
    
    fn hom_mul<'a>(C: &CiphertextRing<Self>, C_mul: &CiphertextRing<Self>, lhs: Ciphertext<Self>, rhs: Ciphertext<Self>, rk: &RelinKey<'a, Self>, conv_data: &MulConversionData) -> Ciphertext<Self>
        where Self: 'a
    {
        let (c00, c01) = lhs;
        let (c10, c11) = rhs;
        let lift = |c| 
            C_mul.get_ring().perform_rns_op_from(C.get_ring(), &c, &conv_data.lift_to_C_mul);
    
        let c00_lifted = lift(c00);
        let c01_lifted = lift(c01);
        let c10_lifted = lift(c10);
        let c11_lifted = lift(c11);
    
        let lifted0 = C_mul.mul_ref(&c00_lifted, &c10_lifted);
        let lifted1 = C_mul.add(C_mul.mul_ref_snd(c00_lifted, &c11_lifted), C_mul.mul_ref_fst(&c01_lifted, c10_lifted));
        let lifted2 = C_mul.mul(c01_lifted, c11_lifted);
    
        let scale_down = |c: El<CiphertextRing<Self>>| C.get_ring().perform_rns_op_from(C_mul.get_ring(), &c, &conv_data.scale_down_to_C);
    
        let res0 = scale_down(lifted0);
        let res1 = scale_down(lifted1);
        let res2 = scale_down(lifted2);
    
        let op = C.get_ring().to_gadget_product_lhs(res2, rk.0);
        let (s0, s1) = &rk.1;
    
        return (C.add_ref(&res0, &C.get_ring().gadget_product(&op, s0)), C.add_ref(&res1, &C.get_ring().gadget_product(&op, s1)));
    }
    
    fn gen_switch_key<'a, R: Rng + CryptoRng, const LOG: bool>(C: &'a CiphertextRing<Self>, mut rng: R, old_sk: &SecretKey<Self>, new_sk: &SecretKey<Self>, digits: usize) -> KeySwitchKey<'a, Self>
        where Self: 'a
    {
        let mut res_0 = C.get_ring().gadget_product_rhs_empty::<LOG>(digits);
        let mut res_1 = C.get_ring().gadget_product_rhs_empty::<LOG>(digits);
        for digit_i in 0..res_0.gadget_vector().len() {
            let (c0, c1) = Self::enc_sym_zero(C, &mut rng, new_sk);
            let digit_range = res_0.gadget_vector().at(digit_i).clone();
            let factor = C.base_ring().get_ring().from_congruence((0..C.base_ring().len()).map(|i2| {
                let Fp = C.base_ring().at(i2);
                if digit_range.contains(&i2) { Fp.one() } else { Fp.zero() } 
            }));
            let mut payload = C.clone_el(&old_sk);
            C.inclusion().mul_assign_ref_map(&mut payload, &factor);
            C.add_assign_ref(&mut payload, &c0);
            res_0.set_rns_factor(digit_i, payload);
            res_1.set_rns_factor(digit_i, c1);
        }
        return (digits, (res_0, res_1));
    }
    
    fn key_switch<'a>(C: &CiphertextRing<Self>, ct: Ciphertext<Self>, switch_key: &KeySwitchKey<'a, Self>) -> Ciphertext<Self>
        where Self: 'a
    {
        let (c0, c1) = ct;
        let (s0, s1) = &switch_key.1;
        let op = C.get_ring().to_gadget_product_lhs(c1, switch_key.0);
        return (
            C.add_ref(&c0, &C.get_ring().gadget_product(&op, s0)),
            C.get_ring().gadget_product(&op, s1)
        );
    }
    
    fn mod_switch_to_plaintext(target: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: Ciphertext<Self>, switch_data: &ModSwitchData) -> (El<PlaintextRing<Self>>, El<PlaintextRing<Self>>) {
        let (c0, c1) = ct;
        return (
            C.get_ring().perform_rns_op_to_decompring(target, &c0, &switch_data.scale),
            C.get_ring().perform_rns_op_to_decompring(target, &c1, &switch_data.scale)
        );
    }
    
    fn gen_gk<'a, R: Rng + CryptoRng, const LOG: bool>(C: &'a CiphertextRing<Self>, rng: R, sk: &SecretKey<Self>, g: ZnEl, digits: usize) -> KeySwitchKey<'a, Self>
        where Self: 'a
    {
        Self::gen_switch_key::<_, LOG>(C, rng, &C.get_ring().apply_galois_action(sk, g), sk, digits)
    }
    
    fn hom_galois<'a>(C: &CiphertextRing<Self>, ct: Ciphertext<Self>, g: ZnEl, gk: &KeySwitchKey<'a, Self>) -> Ciphertext<Self>
        where Self: 'a
    {
        Self::key_switch(C, (
            C.get_ring().apply_galois_action(&ct.0, g),
            C.get_ring().apply_galois_action(&ct.1, g)
        ), gk)
    }
    
    fn hom_galois_many<'a, 'b, V>(C: &CiphertextRing<Self>, ct: Ciphertext<Self>, gs: &[ZnEl], gks: V) -> Vec<Ciphertext<Self>>
        where V: VectorFn<&'b KeySwitchKey<'a, Self>>,
            KeySwitchKey<'a, Self>: 'b,
            'a: 'b,
            Self: 'a
    {
        let digits = gks.at(0).0;
        assert!(gks.iter().all(|(d, _)| *d == digits));
        let (c0, c1) = ct;
        let lhs = C.get_ring().to_gadget_product_lhs(c1, digits);
        return (0..gs.len()).map(|i| {
            let c1_g = lhs.apply_galois_action(C.get_ring(), gs[i]);
            let (s0, s1) = &gks.at(i).1;
            let r0 = C.get_ring().gadget_product(&c1_g, s0);
            let r1 = C.get_ring().gadget_product(&c1_g, s1);
            let c0_g = C.apply_galois_action(&c0, gs[i]);
            return (C.add_ref(&r0, &c0_g), r1);
        }).collect();
    }

    fn create_multiplication_rescale(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, Cmul: &CiphertextRing<Self>) -> MulConversionData {
        MulConversionData {
            lift_to_C_mul: rnsconv::shared_lift::AlmostExactSharedBaseConversion::new_with(
                C.base_ring().as_iter().map(|R| Zn::new(*R.modulus() as u64)).collect::<Vec<_>>(), 
                Vec::new(),
                Cmul.base_ring().as_iter().skip(C.base_ring().len()).map(|R| Zn::new(*R.modulus() as u64)).collect::<Vec<_>>(),
                Global
            ),
            scale_down_to_C: rnsconv::bfv_rescale::AlmostExactRescalingConvert::new_with(
                Cmul.base_ring().as_iter().map(|R| Zn::new(*R.modulus() as u64)).collect::<Vec<_>>(), 
                vec![ Zn::new(*P.base_ring().modulus() as u64) ], 
                C.base_ring().len(),
                Global
            )
        }
    }
}

#[derive(Clone, Debug)]
pub struct Pow2BFVParams {
    pub log2_q_min: usize,
    pub log2_q_max: usize,
    pub log2_N: usize
}

impl BFVParams for Pow2BFVParams {

    type NumberRing = Pow2CyclotomicDecomposableNumberRing;

    fn number_ring(&self) -> Self::NumberRing {
        Pow2CyclotomicDecomposableNumberRing::new(2 << self.log2_N)
    }

    fn ciphertext_modulus_bits(&self) -> Range<usize> {
        self.log2_q_min..self.log2_q_max
    }
}

#[derive(Clone, Debug)]
pub struct CompositeBFVParams {
    pub log2_q_min: usize,
    pub log2_q_max: usize,
    pub n1: usize,
    pub n2: usize
}

impl BFVParams for CompositeBFVParams {

    type NumberRing = CompositeCyclotomicNumberRing;

    fn ciphertext_modulus_bits(&self) -> Range<usize> {
        self.log2_q_min..self.log2_q_max
    }

    fn number_ring(&self) -> Self::NumberRing {
        CompositeCyclotomicNumberRing::new(self.n1, self.n2)
    }
}

fn coeff_repr<Params: BFVParams>(C: &CiphertextRing<Params>, ct: Ciphertext<Params>) -> Ciphertext<Params> {
    C.get_ring().force_coeff_repr(&ct.0);
    C.get_ring().force_coeff_repr(&ct.1);
    return ct;
}

#[test]
fn test_pow2_bfv_hom_galois() {
    let mut rng = thread_rng();
    
    let params = Pow2BFVParams {
        log2_q_min: 500,
        log2_q_max: 520,
        log2_N: 7
    };
    let t = 3;
    let digits = 3;
    
    let P = params.create_plaintext_ring(t);
    let (C, _C_mul) = params.create_ciphertext_rings();    
    let sk = Pow2BFVParams::gen_sk(&C, &mut rng);
    let gk = Pow2BFVParams::gen_gk::<_, true>(&C, &mut rng, &sk, P.get_ring().cyclotomic_index_ring().int_hom().map(3), digits);
    
    let m = P.canonical_gen();
    let ct = Pow2BFVParams::enc_sym(&P, &C, &mut rng, &m, &sk);
    let ct_res = Pow2BFVParams::hom_galois(&C, ct, P.get_ring().cyclotomic_index_ring().int_hom().map(3), &gk);
    let res = Pow2BFVParams::dec(&P, &C, ct_res, &sk);

    assert_el_eq!(&P, &P.pow(P.canonical_gen(), 3), &res);
}

#[test]
fn test_pow2_bfv_mul() {
    let mut rng = thread_rng();
    
    let params = Pow2BFVParams {
        log2_q_min: 500,
        log2_q_max: 520,
        log2_N: 10
    };
    let t = 257;
    let digits = 3;
    
    let P = params.create_plaintext_ring(t);
    let (C, C_mul) = params.create_ciphertext_rings();

    let sk = Pow2BFVParams::gen_sk(&C, &mut rng);
    let mul_rescale_data = Pow2BFVParams::create_multiplication_rescale(&P, &C, &C_mul);
    let rk = Pow2BFVParams::gen_rk::<_, true>(&C, &mut rng, &sk, digits);

    let m = P.int_hom().map(2);
    let ct = Pow2BFVParams::enc_sym(&P, &C, &mut rng, &m, &sk);

    let m = Pow2BFVParams::dec(&P, &C, Pow2BFVParams::clone_ct(&C, &ct), &sk);
    assert_el_eq!(&P, &P.int_hom().map(2), &m);

    let ct_sqr = Pow2BFVParams::hom_mul(&C, &C_mul, Pow2BFVParams::clone_ct(&C, &ct), Pow2BFVParams::clone_ct(&C, &ct), &rk, &mul_rescale_data);
    let m_sqr = Pow2BFVParams::dec(&P, &C, ct_sqr, &sk);

    assert_el_eq!(&P, &P.int_hom().map(4), &m_sqr);
}

#[test]
fn test_composite_bfv_mul() {
    let mut rng = thread_rng();
    
    let params = CompositeBFVParams {
        log2_q_min: 500,
        log2_q_max: 520,
        n1: 17,
        n2: 97
    };
    let t = 8;
    let digits = 3;
    
    let P = params.create_plaintext_ring(t);
    let (C, C_mul) = params.create_ciphertext_rings();

    let sk = CompositeBFVParams::gen_sk(&C, &mut rng);
    let mul_rescale_data = CompositeBFVParams::create_multiplication_rescale(&P, &C, &C_mul);
    let rk = CompositeBFVParams::gen_rk::<_, true>(&C, &mut rng, &sk, digits);

    let m = P.int_hom().map(2);
    let ct = CompositeBFVParams::enc_sym(&P, &C, &mut rng, &m, &sk);

    let m = CompositeBFVParams::dec(&P, &C, CompositeBFVParams::clone_ct(&C, &ct), &sk);
    assert_el_eq!(&P, &P.int_hom().map(2), &m);
    
    let ct_sqr = CompositeBFVParams::hom_mul(&C, &C_mul, CompositeBFVParams::clone_ct(&C, &ct), CompositeBFVParams::clone_ct(&C, &ct), &rk, &mul_rescale_data);
    let m_sqr = CompositeBFVParams::dec(&P, &C, ct_sqr, &sk);

    assert_el_eq!(&P, &P.int_hom().map(4), &m_sqr);
}

#[test]
#[ignore]
fn print_timings_pow2_bfv_mul() {
    let mut rng = thread_rng();
    
    let params = Pow2BFVParams {
        log2_q_min: 790,
        log2_q_max: 800,
        log2_N: 15
    };
    let t = 257;
    let digits = 3;
    
    let P = log_time::<_, _, true, _>("CreatePtxtRing", |[]|
        params.create_plaintext_ring(t)
    );
    let (C, C_mul) = log_time::<_, _, true, _>("CreateCtxtRing", |[]|
        params.create_ciphertext_rings()
    );
    print_all_timings();

    let sk = log_time::<_, _, true, _>("GenSK", |[]| 
        Pow2BFVParams::gen_sk(&C, &mut rng)
    );
    let mul_rescale_data = log_time::<_, _, true, _>("CreateMulRescale", |[]|
        Pow2BFVParams::create_multiplication_rescale(&P, &C, &C_mul)
    );

    let m = P.int_hom().map(2);
    let ct = log_time::<_, _, true, _>("EncSym", |[]|
        coeff_repr::<Pow2BFVParams>(&C, Pow2BFVParams::enc_sym(&P, &C, &mut rng, &m, &sk))
    );

    // let res = log_time::<_, _, true, _>("HomAddPlain", |[]| 
    //     coeff_repr::<Pow2BFVParams>(&C, Pow2BFVParams::hom_add_plain(&P, &C, &m, Pow2BFVParams::clone_ct(&C, &ct)))
    // );
    // assert_el_eq!(&P, &P.int_hom().map(4), &Pow2BFVParams::dec(&P, &C, res, &sk));

    let res = log_time::<_, _, true, _>("HomAdd", |[]| 
        coeff_repr::<Pow2BFVParams>(&C, Pow2BFVParams::hom_add(&C, Pow2BFVParams::clone_ct(&C, &ct), &ct))
    );
    assert_el_eq!(&P, &P.int_hom().map(4), &Pow2BFVParams::dec(&P, &C, res, &sk));

    let res = log_time::<_, _, true, _>("HomMulPlain", |[]| 
        coeff_repr::<Pow2BFVParams>(&C, Pow2BFVParams::hom_mul_plain(&P, &C, &m, Pow2BFVParams::clone_ct(&C, &ct)))
    );
    assert_el_eq!(&P, &P.int_hom().map(4), &Pow2BFVParams::dec(&P, &C, res, &sk));

    let rk = log_time::<_, _, true, _>("GenRK", |[]| 
        Pow2BFVParams::gen_rk::<_, true>(&C, &mut rng, &sk, digits)
    );
    clear_all_timings();
    let res = log_time::<_, _, true, _>("HomMul", |[]| 
        coeff_repr::<Pow2BFVParams>(&C, Pow2BFVParams::hom_mul(&C, &C_mul, Pow2BFVParams::clone_ct(&C, &ct), Pow2BFVParams::clone_ct(&C, &ct), &rk, &mul_rescale_data))
    );
    print_all_timings();
    assert_el_eq!(&P, &P.int_hom().map(4), &Pow2BFVParams::dec(&P, &C, res, &sk));
}

#[test]
#[ignore]
fn print_timings_double_rns_composite_bfv_mul() {
    let mut rng = thread_rng();
    
    let params = CompositeBFVParams {
        log2_q_min: 790,
        log2_q_max: 800,
        n1: 127,
        n2: 337
    };
    let t = 4;
    let digits = 3;
    
    let P = log_time::<_, _, true, _>("CreatePtxtRing", |[]|
        params.create_plaintext_ring(t)
    );
    let (C, C_mul) = log_time::<_, _, true, _>("CreateCtxtRing", |[]|
        params.create_ciphertext_rings()
    );

    let sk = log_time::<_, _, true, _>("GenSK", |[]| 
        CompositeBFVParams::gen_sk(&C, &mut rng)
    );
    let mul_rescale_data = log_time::<_, _, true, _>("CreateMulRescale", |[]|
        CompositeBFVParams::create_multiplication_rescale(&P, &C, &C_mul)
    );

    let m = P.int_hom().map(3);
    let ct = log_time::<_, _, true, _>("EncSym", |[]|
        coeff_repr::<CompositeBFVParams>(&C, CompositeBFVParams::enc_sym(&P, &C, &mut rng, &m, &sk))
    );
    assert_el_eq!(&P, &P.int_hom().map(3), &CompositeBFVParams::dec(&P, &C, CompositeBFVParams::clone_ct(&C, &ct), &sk));

    let res = log_time::<_, _, true, _>("HomAdd", |[]| 
        coeff_repr::<CompositeBFVParams>(&C, CompositeBFVParams::hom_add(&C, CompositeBFVParams::clone_ct(&C, &ct), &ct))
    );
    assert_el_eq!(&P, &P.int_hom().map(2), &CompositeBFVParams::dec(&P, &C, res, &sk));

    let res = log_time::<_, _, true, _>("HomAddPlain", |[]| 
        coeff_repr::<CompositeBFVParams>(&C, CompositeBFVParams::hom_add_plain(&P, &C, &m, CompositeBFVParams::clone_ct(&C, &ct)))
    );
    assert_el_eq!(&P, &P.int_hom().map(2), &CompositeBFVParams::dec(&P, &C, res, &sk));

    let res = log_time::<_, _, true, _>("HomMulPlain", |[]| 
        coeff_repr::<CompositeBFVParams>(&C, CompositeBFVParams::hom_mul_plain(&P, &C, &m, CompositeBFVParams::clone_ct(&C, &ct)))
    );
    assert_el_eq!(&P, &P.int_hom().map(1), &CompositeBFVParams::dec(&P, &C, res, &sk));

    let rk = log_time::<_, _, true, _>("GenRK", |[]| 
        CompositeBFVParams::gen_rk::<_, true>(&C, &mut rng, &sk, digits)
    );
    clear_all_timings();
    let res = log_time::<_, _, true, _>("HomMul", |[]| 
        coeff_repr::<CompositeBFVParams>(&C, CompositeBFVParams::hom_mul(&C, &C_mul, CompositeBFVParams::clone_ct(&C, &ct), CompositeBFVParams::clone_ct(&C, &ct), &rk, &mul_rescale_data))
    );
    print_all_timings();
    assert_el_eq!(&P, &P.int_hom().map(1), &CompositeBFVParams::dec(&P, &C, res, &sk));
}
