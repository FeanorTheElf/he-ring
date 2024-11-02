use std::{alloc::Global, ops::Range};

use feanor_math::algorithms::convolution::fft::FFTRNSBasedConvolutionZn;
use feanor_math::algorithms::convolution::ConvolutionAlgorithm;
use feanor_math::algorithms::convolution::STANDARD_CONVOLUTION;
use feanor_math::assert_el_eq;
use feanor_math::homomorphism::Homomorphism;
use feanor_math::ordered::OrderedRingStore;
use feanor_math::primitive_int::StaticRing;
use feanor_math::rings::extension::FreeAlgebraStore;
use feanor_math::rings::zn::*;
use feanor_math::ring::*;
use feanor_math::integer::*;
use feanor_math::rings::zn::zn_64::Zn;
use feanor_math::seq::VectorFn;
use feanor_math::seq::VectorView;
use rand::thread_rng;
use rand::CryptoRng;
use rand::Rng;
use rand_distr::StandardNormal;
use zn_64::ZnEl;

use crate::bfv::clear_all_timings;
use crate::bfv::log_time;
use crate::bfv::print_all_timings;
use crate::bfv::HENumberRing;
use crate::cyclotomic::CyclotomicRing;
use crate::extend_sampled_primes;
use crate::rings::ntt_conv::NTTConvolution;
use crate::rings::slots::HypercubeIsomorphism;
use crate::rnsconv;
use crate::{cyclotomic::CyclotomicRingStore, sample_primes};
use crate::rings::gadget_product::single_rns::*;
use crate::rings::single_rns_ring::*;

use super::{CompositeCyclotomicNumberRing, DecompositionRing, DecompositionRingBase};

pub type PlaintextAllocator = Global;
pub type CiphertextAllocator = Global;

#[cfg(feature = "use_hexl")]
pub type UsedConvolution = crate::rings::hexl_conv::HEXLConv;
#[cfg(not(feature = "use_hexl"))]
pub type UsedConvolution = NTTConvolution<Zn>;

pub type NumberRing = CompositeCyclotomicNumberRing;
pub type PlaintextRing = DecompositionRing<NumberRing, Zn, PlaintextAllocator>;
pub type CiphertextRing = SingleRNSRing<NumberRing, Zn, CiphertextAllocator, UsedConvolution>;
pub type SecretKey = El<CiphertextRing>;
pub type GadgetProductOperand<'a> = GadgetProductRhsOperand<'a, NumberRing, CiphertextAllocator, UsedConvolution>;
pub type KeySwitchKey<'a> = (usize, (GadgetProductOperand<'a>, GadgetProductOperand<'a>));
pub type RelinKey<'a> = KeySwitchKey<'a>;
pub type Ciphertext = (El<CiphertextRing>, El<CiphertextRing>);

pub struct MulConversionData {
    lift_to_C_mul: rnsconv::shared_lift::AlmostExactSharedBaseConversion<CiphertextAllocator>,
    scale_down_to_C: rnsconv::bfv_rescale::AlmostExactRescalingConvert<CiphertextAllocator>
}

pub struct ModSwitchData {
    scale: rnsconv::bfv_rescale::AlmostExactRescaling<CiphertextAllocator>
}

const ZZbig: BigIntRing = BigIntRing::RING;
const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;

#[derive(Clone, Debug)]
pub struct CompositeSingleRNSBFVParams {
    pub log2_q_min: usize,
    pub log2_q_max: usize,
    pub n1: usize,
    pub n2: usize
}

impl CompositeSingleRNSBFVParams {

    fn ciphertext_modulus_bits(&self) -> Range<usize> {
        self.log2_q_min..self.log2_q_max
    }

    fn number_ring(&self) -> NumberRing {
        CompositeCyclotomicNumberRing::new(self.n1, self.n2)
    }

    fn create_plaintext_ring(&self, modulus: i64) -> PlaintextRing {
        DecompositionRingBase::new(self.number_ring(), Zn::new(modulus as u64))
    }

    fn create_ciphertext_rings(&self) -> (CiphertextRing, CiphertextRing) {
        let log2_q = self.ciphertext_modulus_bits();
        let number_ring = self.number_ring();

        let mut C_rns_base = sample_primes(log2_q.start, log2_q.end, 56, |bound| <_ as HENumberRing<Zn>>::largest_suitable_prime(&number_ring, int_cast(bound, ZZ, ZZbig)).map(|p| int_cast(p, ZZbig, ZZ))).unwrap();
        C_rns_base.sort_unstable_by(|l, r| ZZbig.cmp(l, r));

        let mut Cmul_rns_base = extend_sampled_primes(&C_rns_base, log2_q.end * 2, log2_q.end * 2 + 57, 57, |bound| <_ as HENumberRing<Zn>>::largest_suitable_prime(&number_ring, int_cast(bound, ZZ, ZZbig)).map(|p| int_cast(p, ZZbig, ZZ))).unwrap();
        assert!(ZZbig.is_gt(&Cmul_rns_base[Cmul_rns_base.len() - 1], &C_rns_base[C_rns_base.len() - 1]));
        Cmul_rns_base.sort_unstable_by(|l, r| ZZbig.cmp(l, r));

        let C = SingleRNSRingBase::<_, _, _, UsedConvolution>::new(
            self.number_ring(),
            zn_rns::Zn::new(C_rns_base.iter().map(|p| Zn::new(int_cast(ZZbig.clone_el(p), ZZ, ZZbig) as u64)).collect(), ZZbig)
        );
        let Cmul = SingleRNSRingBase::<_, _, _, UsedConvolution>::new(
            number_ring,
            zn_rns::Zn::new(Cmul_rns_base.iter().map(|p| Zn::new(int_cast(ZZbig.clone_el(p), ZZ, ZZbig) as u64)).collect(), ZZbig)
        );
        return (C, Cmul);
    }

    fn gen_sk<R: Rng + CryptoRng>(C: &CiphertextRing, mut rng: R) -> SecretKey {
        // we sample uniform ternary secrets 
        C.get_ring().sample_from_coefficient_distribution(|| (rng.next_u32() % 3) as i32 - 1)
    }

    fn enc_sym_zero<R: Rng + CryptoRng>(C: &CiphertextRing, mut rng: R, sk: &SecretKey) -> Ciphertext {
        let a = C.get_ring().sample_uniform(|| rng.next_u64());
        let mut b = C.negate(C.mul_ref(&a, &sk));
        let e = C.get_ring().sample_from_coefficient_distribution(|| (rng.sample::<f64, _>(StandardNormal) * 3.2).round() as i32);
        C.add_assign(&mut b, e);
        return (b, a);
    }

    fn transparent_zero(C: &CiphertextRing) -> Ciphertext {
        (C.zero(), C.zero())
    }

    fn enc_sym<R: Rng + CryptoRng>(P: &PlaintextRing, C: &CiphertextRing, rng: R, m: &El<PlaintextRing>, sk: &SecretKey) -> Ciphertext {
        Self::hom_add_plain(P, C, m, Self::enc_sym_zero(C, rng, sk))
    }

    fn enc_sk(P: &PlaintextRing, C: &CiphertextRing) -> Ciphertext {
        let Delta = ZZbig.rounded_div(
            ZZbig.clone_el(C.base_ring().modulus()), 
            &int_cast(*P.base_ring().modulus() as i32, &ZZbig, &StaticRing::<i32>::RING)
        );
        (C.zero(), C.inclusion().map(C.base_ring().coerce(&ZZbig, Delta)))
    }
    
    fn remove_noise(P: &PlaintextRing, C: &CiphertextRing, c: &El<CiphertextRing>) -> El<PlaintextRing> {
        let coefficients = C.wrt_canonical_basis(c);
        let Delta = ZZbig.rounded_div(
            ZZbig.clone_el(C.base_ring().modulus()), 
            &int_cast(*P.base_ring().modulus() as i32, &ZZbig, &StaticRing::<i32>::RING)
        );
        let modulo = P.base_ring().can_hom(&ZZbig).unwrap();
        return P.from_canonical_basis((0..coefficients.len()).map(|i| modulo.map(ZZbig.rounded_div(C.base_ring().smallest_lift(coefficients.at(i)), &Delta))));
    }
    
    fn dec(P: &PlaintextRing, C: &CiphertextRing, ct: Ciphertext, sk: &SecretKey) -> El<PlaintextRing> {
        let (c0, c1) = ct;
        let noisy_m = C.add(c0, C.mul_ref_snd(c1, sk));
        return Self::remove_noise(P, C, &noisy_m);
    }
    
    fn dec_println(P: &PlaintextRing, C: &CiphertextRing, ct: &Ciphertext, sk: &SecretKey) {
        let m = Self::dec(P, C, Self::clone_ct(C, ct), sk);
        P.println(&m);
        println!();
    }
    
    fn dec_println_slots(P: &PlaintextRing, C: &CiphertextRing, ct: &Ciphertext, sk: &SecretKey) {
        let H = HypercubeIsomorphism::new::<false>(P.get_ring());
        let m = Self::dec(P, C, Self::clone_ct(C, ct), sk);
        for a in H.get_slot_values(&m) {
            H.slot_ring().println(&a);
        }
        println!();
    }
    
    fn hom_add(C: &CiphertextRing, lhs: Ciphertext, rhs: &Ciphertext) -> Ciphertext {
        let (lhs0, lhs1) = lhs;
        let (rhs0, rhs1) = rhs;
        return (C.add_ref_snd(lhs0, rhs0), C.add_ref_snd(lhs1, rhs1));
    }
    
    fn hom_sub(C: &CiphertextRing, lhs: Ciphertext, rhs: &Ciphertext) -> Ciphertext {let (lhs0, lhs1) = lhs;
        let (rhs0, rhs1) = rhs;
        return (C.sub_ref_snd(lhs0, rhs0), C.sub_ref_snd(lhs1, rhs1));
    }
    
    fn clone_ct(C: &CiphertextRing, ct: &Ciphertext) -> Ciphertext {
        (C.clone_el(&ct.0), C.clone_el(&ct.1))
    }
    
    fn hom_add_plain(P: &PlaintextRing, C: &CiphertextRing, m: &El<PlaintextRing>, ct: Ciphertext) -> Ciphertext {
        let mut m = C.get_ring().exact_convert_from_nttring(P, m);
        let Delta = C.base_ring().coerce(&ZZbig, ZZbig.rounded_div(
            ZZbig.clone_el(C.base_ring().modulus()), 
            &int_cast(*P.base_ring().modulus() as i32, &ZZbig, &StaticRing::<i32>::RING)
        ));
        C.inclusion().mul_assign_map(&mut m, Delta);
        return (C.add(ct.0, m), ct.1);
    }
    
    fn hom_mul_plain(P: &PlaintextRing, C: &CiphertextRing, m: &El<PlaintextRing>, ct: Ciphertext) -> Ciphertext {
        let m = C.get_ring().exact_convert_from_nttring(P, m);
        let (c0, c1) = ct;
        return (C.mul_ref_snd(c0, &m), C.mul(c1, m));
    }
    
    fn hom_mul_plain_i64(_P: &PlaintextRing, C: &CiphertextRing, m: i64, ct: Ciphertext) -> Ciphertext {
        let inclusion = C.inclusion().compose(C.base_ring().can_hom(&StaticRing::<i64>::RING).unwrap());
        (inclusion.mul_map(ct.0, m), inclusion.mul_map(ct.1, m))
    }
    
    fn noise_budget(P: &PlaintextRing, C: &CiphertextRing, ct: &Ciphertext, sk: &SecretKey) -> usize {
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
    
    fn gen_rk<'a, R: Rng + CryptoRng>(C: &'a CiphertextRing, rng: R, sk: &SecretKey, digits: usize) -> RelinKey<'a>
        where Self: 'a
    {
        Self::gen_switch_key(C, rng, &C.pow(C.clone_el(sk), 2), sk, digits)
    }
    
    fn hom_mul<'a>(C: &CiphertextRing, C_mul: &CiphertextRing, lhs: Ciphertext, rhs: Ciphertext, rk: &RelinKey<'a>, conv_data: &MulConversionData) -> Ciphertext
        where Self: 'a
    {
        let (c00, c01) = lhs;
        let (c10, c11) = rhs;
        let lift = |c: SingleRNSRingEl<NumberRing, Zn, CiphertextAllocator, _>| 
            C_mul.get_ring().perform_rns_op_from(C.get_ring(), &c, &conv_data.lift_to_C_mul);
    
        let c0_lifted = [lift(c00), lift(c01)];
        let c1_lifted = [lift(c10), lift(c11)];
        let [lifted0, lifted1, lifted2] = C_mul.get_ring().two_by_two_convolution(&c0_lifted, &c1_lifted);
    
        let scale_down = |c: El<CiphertextRing>| 
            C.get_ring().perform_rns_op_from(C_mul.get_ring(), &c, &conv_data.scale_down_to_C);
    
        let res0 = scale_down(lifted0);
        let res1 = scale_down(lifted1);
        let res2 = scale_down(lifted2);
    
        let op = C.get_ring().to_gadget_product_lhs(res2, rk.0);
        let (s0, s1) = &rk.1;
    
        return (C.add(res0, C.get_ring().gadget_product(&op, s0)), C.add(res1, C.get_ring().gadget_product(&op, s1)));
    }
    
    fn gen_switch_key<'a, R: Rng + CryptoRng>(C: &'a CiphertextRing, mut rng: R, old_sk: &SecretKey, new_sk: &SecretKey, digits: usize) -> KeySwitchKey<'a>
        where Self: 'a
    {
        let mut res_0 = C.get_ring().gadget_product_rhs_empty(digits);
        let mut res_1 = C.get_ring().gadget_product_rhs_empty(digits);
        for digit_i in 0..res_0.gadget_vector().len() {
            let (c0, c1) = Self::enc_sym_zero(C, &mut rng, new_sk);
            let digit_range = res_0.gadget_vector().at(digit_i).clone();
            let factor = C.base_ring().get_ring().from_congruence((0..C.get_ring().rns_base().len()).map(|i2| {
                let Fp = C.get_ring().rns_base().at(i2);
                if digit_range.contains(&i2) { Fp.one() } else { Fp.zero() } 
            }));
            let mut payload = C.get_ring().clone_el(old_sk);
            C.inclusion().mul_assign_ref_map(&mut payload, &factor);
            C.add_assign(&mut payload, c0);
            res_0.set_rns_factor(digit_i, payload);
            res_1.set_rns_factor(digit_i, c1);
        }
        return (digits, (res_0, res_1));
    }
    
    fn key_switch<'a>(C: &CiphertextRing, ct: Ciphertext, switch_key: &KeySwitchKey<'a>) -> Ciphertext
        where Self: 'a
    {
        let (c0, c1) = ct;
        let (s0, s1) = &switch_key.1;
        let op = C.get_ring().to_gadget_product_lhs(c1, switch_key.0);
        return (
            C.add(c0, C.get_ring().gadget_product(&op, s0)),
            C.get_ring().gadget_product(&op, s1)
        );
    }
    
    fn mod_switch_to_plaintext(target: &PlaintextRing, C: &CiphertextRing, ct: Ciphertext, switch_data: &ModSwitchData) -> (El<PlaintextRing>, El<PlaintextRing>) {
        let (c0, c1) = ct;
        return (
            C.get_ring().perform_rns_op_to_nttring(target, &c0, &switch_data.scale),
            C.get_ring().perform_rns_op_to_nttring(target, &c1, &switch_data.scale)
        );
    }
    
    fn gen_gk<'a, R: Rng + CryptoRng>(C: &'a CiphertextRing, rng: R, sk: &SecretKey, g: ZnEl, digits: usize) -> KeySwitchKey<'a>
        where Self: 'a
    {
        Self::gen_switch_key(C, rng, &C.get_ring().apply_galois_action(sk, g), sk, digits)
    }
    
    fn hom_galois<'a>(C: &CiphertextRing, ct: Ciphertext, g: ZnEl, gk: &KeySwitchKey<'a>) -> Ciphertext
        where Self: 'a
    {
        Self::key_switch(C, (
            C.get_ring().apply_galois_action(&ct.0, g),
            C.get_ring().apply_galois_action(&ct.1, g)
        ), gk)
    }
    
    fn hom_galois_many<'a, 'b, V>(C: &CiphertextRing, ct: Ciphertext, gs: &[ZnEl], gks: V) -> Vec<Ciphertext>
        where V: VectorFn<&'b KeySwitchKey<'a>>,
            KeySwitchKey<'a>: 'b,
            'a: 'b,
            Self: 'a
    {
        let mut result = Vec::new();
        for ((ct0, ct1), gk) in C.apply_galois_action_many(&ct.0, gs).zip(C.apply_galois_action_many(&ct.1, gs)).zip(gks.iter()) {
            result.push(Self::key_switch(C, (ct0, ct1), gk));
        }
        return result;
    }

    fn create_multiplication_rescale(P: &PlaintextRing, C: &CiphertextRing, Cmul: &CiphertextRing) -> MulConversionData {
        let allocator = C.get_ring().allocator().clone();
        MulConversionData {
            lift_to_C_mul: rnsconv::shared_lift::AlmostExactSharedBaseConversion::new_with(
                C.get_ring().rns_base().as_iter().map(|R| Zn::new(*R.modulus() as u64)).collect::<Vec<_>>(), 
                Vec::new(),
                Cmul.get_ring().rns_base().as_iter().skip(C.get_ring().rns_base().len()).map(|R| Zn::new(*R.modulus() as u64)).collect::<Vec<_>>(),
                allocator.clone()
            ),
            scale_down_to_C: rnsconv::bfv_rescale::AlmostExactRescalingConvert::new_with(
                Cmul.get_ring().rns_base().as_iter().map(|R| Zn::new(*R.modulus() as u64)).collect::<Vec<_>>(), 
                vec![ Zn::new(*P.base_ring().modulus() as u64) ], 
                C.get_ring().rns_base().len(),
                allocator
            )
        }
    }
}

#[test]
fn test_bfv_hom_galois() {
    let mut rng = thread_rng();
    
    let params = CompositeSingleRNSBFVParams {
        log2_q_min: 500,
        log2_q_max: 520,
        n1: 7,
        n2: 11
    };
    let t = 3;
    let digits = 3;
    
    let P = params.create_plaintext_ring(t);
    let (C, _C_mul) = params.create_ciphertext_rings();    
    let sk = CompositeSingleRNSBFVParams::gen_sk(&C, &mut rng);
    let gk = CompositeSingleRNSBFVParams::gen_gk(&C, &mut rng, &sk, P.get_ring().cyclotomic_index_ring().int_hom().map(3), digits);
    
    let m = P.canonical_gen();
    let ct = CompositeSingleRNSBFVParams::enc_sym(&P, &C, &mut rng, &m, &sk);
    let ct_res = CompositeSingleRNSBFVParams::hom_galois(&C, ct, P.get_ring().cyclotomic_index_ring().int_hom().map(3), &gk);
    let res = CompositeSingleRNSBFVParams::dec(&P, &C, ct_res, &sk);

    assert_el_eq!(&P, &P.pow(P.canonical_gen(), 3), &res);
}

#[test]
fn test_bfv_mul() {
    let mut rng = thread_rng();
    
    let params = CompositeSingleRNSBFVParams {
        log2_q_min: 500,
        log2_q_max: 520,
        n1: 7,
        n2: 11
    };
    let t = 3;
    let digits = 3;

    let P = params.create_plaintext_ring(t);
    let (C, C_mul) = params.create_ciphertext_rings();

    let sk = CompositeSingleRNSBFVParams::gen_sk(&C, &mut rng);
    let mul_rescale_data = CompositeSingleRNSBFVParams::create_multiplication_rescale(&P, &C, &C_mul);
    let rk = CompositeSingleRNSBFVParams::gen_rk(&C, &mut rng, &sk, digits);

    let m = P.int_hom().map(2);
    let ct = CompositeSingleRNSBFVParams::enc_sym(&P, &C, &mut rng, &m, &sk);

    let m = CompositeSingleRNSBFVParams::dec(&P, &C, CompositeSingleRNSBFVParams::clone_ct(&C, &ct), &sk);
    assert_el_eq!(&P, &P.int_hom().map(2), &m);

    let ct_sqr = CompositeSingleRNSBFVParams::hom_mul(&C, &C_mul, CompositeSingleRNSBFVParams::clone_ct(&C, &ct), CompositeSingleRNSBFVParams::clone_ct(&C, &ct), &rk, &mul_rescale_data);
    let m_sqr = CompositeSingleRNSBFVParams::dec(&P, &C, ct_sqr, &sk);

    assert_el_eq!(&P, &P.int_hom().map(4), &m_sqr);
}

#[test]
#[ignore]
fn print_timings_single_rns_composite_bfv_mul() {
    let mut rng = thread_rng();
    
    let params = CompositeSingleRNSBFVParams {
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
    print_all_timings();

    let sk = log_time::<_, _, true, _>("GenSK", |[]| 
        CompositeSingleRNSBFVParams::gen_sk(&C, &mut rng)
    );
    let mul_rescale_data = log_time::<_, _, true, _>("CreateMulRescale", |[]|
        CompositeSingleRNSBFVParams::create_multiplication_rescale(&P, &C, &C_mul)
    );

    let m = P.int_hom().map(3);
    let ct = log_time::<_, _, true, _>("EncSym", |[]|
        CompositeSingleRNSBFVParams::enc_sym(&P, &C, &mut rng, &m, &sk)
    );

    let res = log_time::<_, _, true, _>("HomAddPlain", |[]| 
        CompositeSingleRNSBFVParams::hom_add_plain(&P, &C, &m, CompositeSingleRNSBFVParams::clone_ct(&C, &ct))
    );
    assert_el_eq!(&P, &P.int_hom().map(2), &CompositeSingleRNSBFVParams::dec(&P, &C, res, &sk));

    let res = log_time::<_, _, true, _>("HomAdd", |[]| 
        CompositeSingleRNSBFVParams::hom_add(&C, CompositeSingleRNSBFVParams::clone_ct(&C, &ct), &ct)
    );
    assert_el_eq!(&P, &P.int_hom().map(2), &CompositeSingleRNSBFVParams::dec(&P, &C, res, &sk));

    let res = log_time::<_, _, true, _>("HomMulPlain", |[]| 
        CompositeSingleRNSBFVParams::hom_mul_plain(&P, &C, &m, CompositeSingleRNSBFVParams::clone_ct(&C, &ct))
    );
    assert_el_eq!(&P, &P.int_hom().map(1), &CompositeSingleRNSBFVParams::dec(&P, &C, res, &sk));

    let rk = log_time::<_, _, true, _>("GenRK", |[]| 
        CompositeSingleRNSBFVParams::gen_rk(&C, &mut rng, &sk, digits)
    );
    clear_all_timings();
    let res = log_time::<_, _, true, _>("HomMul", |[]| 
        CompositeSingleRNSBFVParams::hom_mul(&C, &C_mul, CompositeSingleRNSBFVParams::clone_ct(&C, &ct), CompositeSingleRNSBFVParams::clone_ct(&C, &ct), &rk, &mul_rescale_data)
    );
    print_all_timings();
    assert_el_eq!(&P, &P.int_hom().map(1), &CompositeSingleRNSBFVParams::dec(&P, &C, res, &sk));
}
