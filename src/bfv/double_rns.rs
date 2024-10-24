
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
use rand::thread_rng;

use crate::cyclotomic::CyclotomicRing;
use crate::euler_phi;
use crate::extend_sampled_primes;
use crate::rings::decomposition::*;
use crate::rings::decomposition_ring::*;
use crate::rings::odd_cyclotomic::*;
use crate::rings::pow2_cyclotomic::*;
use crate::rings::gadget_product::double_rns::*;
use crate::rings::double_rns_ring::*;
use crate::profiling::*;
use crate::rings::slots::HypercubeIsomorphism;
use crate::rnsconv;
use crate::sample_primes;

use super::*;

pub type GadgetProductOperand<'a, Params: DoubleRNSBFVParams> = GadgetProductRhsOperand<'a, <Params as DoubleRNSBFVParams>::NumberRing, CiphertextAllocator>;

pub trait DoubleRNSBFVParams {
    
    type NumberRing: DecomposableCyclotomicNumberRing<Zn>;

    fn ciphertext_modulus_bits_(&self) -> Range<usize>;
    fn number_ring_(&self) -> Self::NumberRing;
    fn plaintext_modulus_(&self) -> i64;
}

impl<P: DoubleRNSBFVParams> BFVParams for P {

    type NumberRing = <Self as DoubleRNSBFVParams>::NumberRing;
    type CiphertextRing = DoubleRNSRing<Self::NumberRing, Zn, Global>;
    type CiphertextRingBase = DoubleRNSRingBase<Self::NumberRing, Zn, Global>;
    type Ciphertext = (CoeffOrDoubleRNSEl<Self::NumberRing>, CoeffOrDoubleRNSEl<Self::NumberRing>);
    type KeySwitchKey<'a> = (GadgetProductOperand<'a, P>, GadgetProductOperand<'a, P>)
            where Self: 'a;

    fn ciphertext_modulus_bits(&self) -> Range<usize> {
        self.ciphertext_modulus_bits_()
    }

    fn number_ring(&self) -> <Self as BFVParams>::NumberRing {
        self.number_ring_()
    }

    fn plaintext_modulus(&self) -> i64 {
        self.plaintext_modulus_()
    }

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

        let C = DoubleRNSRingBase::new(
            self.number_ring(),
            zn_rns::Zn::new(C_rns_base.iter().map(|p| Zn::new(int_cast(ZZbig.clone_el(p), ZZ, ZZbig) as u64)).collect(), ZZbig)
        );
        let Cmul = DoubleRNSRingBase::new(
            number_ring,
            zn_rns::Zn::new(Cmul_rns_base.iter().map(|p| Zn::new(int_cast(ZZbig.clone_el(p), ZZ, ZZbig) as u64)).collect(), ZZbig)
        );
        return (C, Cmul);
    }

    fn gen_sk<R: Rng + CryptoRng>(C: &CiphertextRing<Self>, mut rng: R) -> SecretKey<Self> {
        // we sample uniform ternary secrets 
        let result = C.get_ring().sample_from_coefficient_distribution(|| (rng.next_u32() % 3) as i32 - 1);
        let result = C.get_ring().do_fft(result);
        return result;
    }
    
    fn enc_sym_zero<R: Rng + CryptoRng>(C: &CiphertextRing<Self>, mut rng: R, sk: &SecretKey<Self>) -> Ciphertext<Self> {
        let a = C.get_ring().sample_uniform(|| rng.next_u64());
        let mut b = C.get_ring().undo_fft(C.negate(C.mul_ref(&a, &sk)));
        let e = C.get_ring().sample_from_coefficient_distribution(|| (rng.sample::<f64, _>(StandardNormal) * 3.2).round() as i32);
        C.get_ring().add_assign_non_fft(&mut b, &e);
        return (CoeffOrDoubleRNSEl::from_coeff(b), CoeffOrDoubleRNSEl::from_ntt(a));
    }
    
    fn transparent_zero(_C: &CiphertextRing<Self>) -> Ciphertext<Self> {
        (CoeffOrDoubleRNSEl::zero(), CoeffOrDoubleRNSEl::zero())
    }

    fn enc_sym<R: Rng + CryptoRng>(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, rng: R, m: &El<PlaintextRing<Self>>, sk: &SecretKey<Self>) -> Ciphertext<Self> {
        Self::hom_add_plain(P, C, m, Self::enc_sym_zero(C, rng, sk))
    }

    fn enc_sk(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>) -> Ciphertext<Self> {
        let Delta = ZZbig.rounded_div(
            ZZbig.clone_el(C.base_ring().modulus()), 
            &int_cast(*P.base_ring().modulus() as i32, &ZZbig, &StaticRing::<i32>::RING)
        );
        (CoeffOrDoubleRNSEl::zero(), CoeffOrDoubleRNSEl::from_coeff(C.get_ring().non_fft_from(C.base_ring().coerce(&ZZbig, Delta))))
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
        let noisy_m = C.add(c0.to_ntt(C), C.mul_ref_snd(c1.to_ntt(C), sk));
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
        return (CoeffOrDoubleRNSEl::add(lhs0, rhs0, C), CoeffOrDoubleRNSEl::add(lhs1, rhs1, C));
    }
    
    fn hom_sub(C: &CiphertextRing<Self>, lhs: Ciphertext<Self>, rhs: &Ciphertext<Self>) -> Ciphertext<Self> {
        let (lhs0, lhs1) = lhs;
        let (rhs0, rhs1) = rhs;
        return (CoeffOrDoubleRNSEl::sub(lhs0, rhs0, C), CoeffOrDoubleRNSEl::sub(lhs1, rhs1, C));
    }
    
    fn clone_ct(C: &CiphertextRing<Self>, ct: &Ciphertext<Self>) -> Ciphertext<Self> {
        (ct.0.clone(C), ct.1.clone(C))
    }
    
    fn hom_add_plain(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: &El<PlaintextRing<Self>>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        let mut m = C.get_ring().exact_convert_from_nttring(P, m);
        let Delta = C.base_ring().coerce(&ZZbig, ZZbig.rounded_div(
            ZZbig.clone_el(C.base_ring().modulus()), 
            &int_cast(*P.base_ring().modulus() as i32, &ZZbig, &StaticRing::<i32>::RING)
        ));
        C.get_ring().mul_scalar_assign_non_fft(&mut m, &Delta);
        return (CoeffOrDoubleRNSEl::add(ct.0, &CoeffOrDoubleRNSEl::from_coeff(m), C), ct.1);
    }
    
    fn hom_mul_plain(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: &El<PlaintextRing<Self>>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        let m = C.get_ring().do_fft(C.get_ring().exact_convert_from_nttring(P, m));
        let c0 = ct.0.to_ntt(C);
        let c1 = ct.1.to_ntt(C);
        return (CoeffOrDoubleRNSEl::from_ntt(C.mul_ref_snd(c0, &m)), CoeffOrDoubleRNSEl::from_ntt(C.mul(c1, m)));
    }
    
    fn hom_mul_plain_i64(_P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: i64, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        (CoeffOrDoubleRNSEl::mul_i64(ct.0, m, C), CoeffOrDoubleRNSEl::mul_i64(ct.1, m, C))
    }
    
    fn noise_budget(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: &Ciphertext<Self>, sk: &SecretKey<Self>) -> usize {
        let (c0, c1) = Self::clone_ct(C, ct);
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
    
    fn gen_rk<'a, R: Rng + CryptoRng>(C: &'a CiphertextRing<Self>, rng: R, sk: &SecretKey<Self>) -> RelinKey<'a, Self>
        where Self: 'a
    {
        Self::gen_switch_key(C, rng, &C.pow(C.clone_el(sk), 2), sk)
    }
    
    fn hom_mul<'a>(C: &CiphertextRing<Self>, C_mul: &CiphertextRing<Self>, lhs: Ciphertext<Self>, rhs: Ciphertext<Self>, rk: &RelinKey<'a, Self>, conv_data: &MulConversionData) -> Ciphertext<Self>
        where Self: 'a
    {
        let (c00, c01) = lhs;
        let (c10, c11) = rhs;
        let lift = |c: CoeffEl<Self::NumberRing, Zn, CiphertextAllocator>| 
            C_mul.get_ring().do_fft(C_mul.get_ring().perform_rns_op_from(C.get_ring(), &c, &conv_data.lift_to_C_mul));
    
        let c00_lifted = lift(c00.to_coeff(C));
        let c01_lifted = lift(c01.to_coeff(C));
        let c10_lifted = lift(c10.to_coeff(C));
        let c11_lifted = lift(c11.to_coeff(C));
    
        let lifted0 = C_mul.mul_ref(&c00_lifted, &c10_lifted);
        let lifted1 = C_mul.add(C_mul.mul_ref_snd(c00_lifted, &c11_lifted), C_mul.mul_ref_fst(&c01_lifted, c10_lifted));
        let lifted2 = C_mul.mul(c01_lifted, c11_lifted);
    
        let scale_down = |c: El<CiphertextRing<Self>>| 
            C.get_ring().perform_rns_op_from(C_mul.get_ring(), &C_mul.get_ring().undo_fft(c), &conv_data.scale_down_to_C);
    
        let res0 = scale_down(lifted0);
        let res1 = scale_down(lifted1);
        let res2 = scale_down(lifted2);
    
        let op = C.get_ring().to_gadget_product_lhs(res2);
        let (s0, s1) = rk;
    
        return (CoeffOrDoubleRNSEl::add(CoeffOrDoubleRNSEl::from_coeff(res0), &CoeffOrDoubleRNSEl::gadget_product(&op, s0, C), C), CoeffOrDoubleRNSEl::add(CoeffOrDoubleRNSEl::from_coeff(res1), &CoeffOrDoubleRNSEl::gadget_product(&op, s1, C), C));
    }
    
    fn gen_switch_key<'a, R: Rng + CryptoRng>(C: &'a CiphertextRing<Self>, mut rng: R, old_sk: &SecretKey<Self>, new_sk: &SecretKey<Self>) -> KeySwitchKey<'a, Self>
        where Self: 'a
    {
        let old_sk_non_fft = C.get_ring().undo_fft(C.clone_el(old_sk));
        let mut res_0 = C.get_ring().gadget_product_rhs_empty();
        let mut res_1 = C.get_ring().gadget_product_rhs_empty();
        for i in 0..C.get_ring().rns_base().len() {
            let (c0, c1) = Self::enc_sym_zero(C, &mut rng, new_sk);
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
    
    fn key_switch<'a>(C: &CiphertextRing<Self>, ct: Ciphertext<Self>, switch_key: &KeySwitchKey<'a, Self>) -> Ciphertext<Self>
        where Self: 'a
    {
        let (c0, c1) = ct;
        let (s0, s1) = switch_key;
        let op = C.get_ring().to_gadget_product_lhs(c1.to_coeff(C));
        return (
            CoeffOrDoubleRNSEl::add(c0, &CoeffOrDoubleRNSEl::gadget_product(&op, s0, C), C),
            CoeffOrDoubleRNSEl::gadget_product(&op, s1, C)
        );
    }
    
    fn mod_switch_to_plaintext(target: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: Ciphertext<Self>, switch_data: &ModSwitchData) -> (El<PlaintextRing<Self>>, El<PlaintextRing<Self>>) {
        let (c0, c1) = ct;
        return (
            C.get_ring().perform_rns_op_to_nttring(target, &c0.to_coeff(C), &switch_data.scale),
            C.get_ring().perform_rns_op_to_nttring(target, &c1.to_coeff(C), &switch_data.scale)
        );
    }
    
    fn gen_gk<'a, R: Rng + CryptoRng>(C: &'a CiphertextRing<Self>, rng: R, sk: &SecretKey<Self>, g: ZnEl) -> KeySwitchKey<'a, Self>
        where Self: 'a
    {
        Self::gen_switch_key(C, rng, &C.get_ring().apply_galois_action(sk, g), sk)
    }
    
    fn hom_galois<'a>(C: &CiphertextRing<Self>, ct: Ciphertext<Self>, g: ZnEl, gk: &KeySwitchKey<'a, Self>) -> Ciphertext<Self>
        where Self: 'a
    {
        Self::key_switch(C, (
            CoeffOrDoubleRNSEl::from_ntt(C.get_ring().apply_galois_action(&ct.0.to_ntt(C), g)),
            CoeffOrDoubleRNSEl::from_ntt(C.get_ring().apply_galois_action(&ct.1.to_ntt(C), g))
        ), gk)
    }
    
    fn hom_galois_many<'a, 'b, V>(C: &CiphertextRing<Self>, ct: Ciphertext<Self>, gs: &[ZnEl], gks: V) -> Vec<Ciphertext<Self>>
        where V: VectorFn<&'b KeySwitchKey<'a, Self>>,
            KeySwitchKey<'a, Self>: 'b,
            'a: 'b,
            Self: 'a
    {
        let (c0, c1) = ct;
        let c0_ntt = c0.to_ntt(&C);
        let lhs = C.get_ring().to_gadget_product_lhs(c1.to_coeff(&C));
        return (0..gs.len()).map(|i| {
            let c1_g = lhs.apply_galois_action(C.get_ring(), gs[i]);
            let (s0, s1) = gks.at(i);
            let r0 = CoeffOrDoubleRNSEl::gadget_product(&c1_g, s0, C);
            let r1 = CoeffOrDoubleRNSEl::gadget_product(&c1_g, s1, C);
            let c0_g = CoeffOrDoubleRNSEl::from_ntt(C.get_ring().apply_galois_action(&c0_ntt, gs[i]));
            return (CoeffOrDoubleRNSEl::add(r0, &c0_g, C), r1);
        }).collect();
    }

    fn create_multiplication_rescale(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, Cmul: &CiphertextRing<Self>) -> MulConversionData {
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

pub struct CoeffOrDoubleRNSEl<NumberRing: DecomposableNumberRing<Zn>> {
    ntt_part: Option<DoubleRNSEl<NumberRing, Zn, CiphertextAllocator>>,
    coeff_part: Option<CoeffEl<NumberRing, Zn, CiphertextAllocator>>
}

impl<NumberRing: DecomposableNumberRing<Zn>> CoeffOrDoubleRNSEl<NumberRing> {

    pub fn ntt_repr(self, C: &DoubleRNSRing<NumberRing, Zn, CiphertextAllocator>) -> Self {
        CoeffOrDoubleRNSEl::from_ntt(self.to_ntt(C))
    }

    pub fn coeff_repr(self, C: &DoubleRNSRing<NumberRing, Zn, CiphertextAllocator>) -> Self {
        CoeffOrDoubleRNSEl::from_coeff(self.to_coeff(C))
    }

    pub fn from_ntt(el: DoubleRNSEl<NumberRing, Zn, CiphertextAllocator>) -> Self {
        Self {
            coeff_part: None,
            ntt_part: Some(el)
        }
    }

    pub fn from_coeff(el: CoeffEl<NumberRing, Zn, CiphertextAllocator>) -> Self {
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

    pub fn to_ntt(self, C: &DoubleRNSRing<NumberRing, Zn, CiphertextAllocator>) -> DoubleRNSEl<NumberRing, Zn, CiphertextAllocator> {
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

    pub fn to_coeff(self, C: &DoubleRNSRing<NumberRing, Zn, CiphertextAllocator>) -> CoeffEl<NumberRing, Zn, CiphertextAllocator> {
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

    pub fn gadget_product<'a>(lhs: &GadgetProductLhsOperand<'a, NumberRing, CiphertextAllocator>, rhs: &GadgetProductRhsOperand<'a, NumberRing, CiphertextAllocator>, C: &DoubleRNSRing<NumberRing, Zn, CiphertextAllocator>) -> CoeffOrDoubleRNSEl<NumberRing> {
        match C.get_ring().preferred_output_repr(lhs, rhs) {
            ElRepr::Coeff => CoeffOrDoubleRNSEl { ntt_part: None, coeff_part: Some(C.get_ring().gadget_product_coeff(lhs, rhs)) },
            ElRepr::NTT => CoeffOrDoubleRNSEl { ntt_part: Some(C.get_ring().gadget_product_ntt(lhs, rhs)), coeff_part: None },
        }
    }

    pub fn add(lhs: CoeffOrDoubleRNSEl<NumberRing>, rhs: &CoeffOrDoubleRNSEl<NumberRing>, C: &DoubleRNSRing<NumberRing, Zn, CiphertextAllocator>) -> CoeffOrDoubleRNSEl<NumberRing> {
        CoeffOrDoubleRNSEl {
            ntt_part: if lhs.ntt_part.is_some() && rhs.ntt_part.is_some() { Some(C.add_ref_snd(lhs.ntt_part.unwrap(), rhs.ntt_part.as_ref().unwrap())) } else { lhs.ntt_part.or(rhs.ntt_part.as_ref().map(|x| C.clone_el(x)))},
            coeff_part: if lhs.coeff_part.is_some() && rhs.coeff_part.is_some() {
                let mut result  = lhs.coeff_part.unwrap();
                C.get_ring().add_assign_non_fft(&mut result, rhs.coeff_part.as_ref().unwrap());
                Some(result)
            } else { lhs.coeff_part.or(rhs.coeff_part.as_ref().map(|x| C.get_ring().clone_el_non_fft(x))) }
        }
    }

    pub fn sub(lhs: CoeffOrDoubleRNSEl<NumberRing>, rhs: &CoeffOrDoubleRNSEl<NumberRing>, C: &DoubleRNSRing<NumberRing, Zn, CiphertextAllocator>) -> CoeffOrDoubleRNSEl<NumberRing> {
        CoeffOrDoubleRNSEl {
            ntt_part: if lhs.ntt_part.is_some() && rhs.ntt_part.is_some() { Some(C.sub_ref_snd(lhs.ntt_part.unwrap(), rhs.ntt_part.as_ref().unwrap())) } else { lhs.ntt_part.or(rhs.ntt_part.as_ref().map(|x| C.negate(C.clone_el(x))))},
            coeff_part: if lhs.coeff_part.is_some() && rhs.coeff_part.is_some() {
                let mut result  = lhs.coeff_part.unwrap();
                C.get_ring().sub_assign_non_fft(&mut result, rhs.coeff_part.as_ref().unwrap());
                Some(result)
            } else { lhs.coeff_part.or(rhs.coeff_part.as_ref().map(|x| C.get_ring().negate_non_fft(C.get_ring().clone_el_non_fft(x)))) }
        }
    }

    pub fn mul_i64(mut val: CoeffOrDoubleRNSEl<NumberRing>, scalar: i64, C: &DoubleRNSRing<NumberRing, Zn, CiphertextAllocator>) -> CoeffOrDoubleRNSEl<NumberRing> {
        let hom = C.base_ring().can_hom(&StaticRing::<i64>::RING).unwrap();
        if let Some(ntt_part) = &mut val.ntt_part {
            C.inclusion().mul_assign_map(ntt_part, hom.map(scalar));
        }
        if let Some(coeff_part) = &mut val.coeff_part {
            C.get_ring().mul_scalar_assign_non_fft(coeff_part, &hom.map(scalar));
        }
        return val;
    }

    pub fn clone(&self, C: &DoubleRNSRing<NumberRing, Zn, CiphertextAllocator>) -> CoeffOrDoubleRNSEl<NumberRing> {
        CoeffOrDoubleRNSEl { 
            ntt_part: self.ntt_part.as_ref().map(|x| C.clone_el(x)), 
            coeff_part: self.coeff_part.as_ref().map(|x| C.get_ring().clone_el_non_fft(x))
        }
    }
}

#[derive(Clone, Debug)]
pub struct Pow2BFVParams {
    pub t: i64,
    pub log2_q_min: usize,
    pub log2_q_max: usize,
    pub log2_N: usize
}

impl DoubleRNSBFVParams for Pow2BFVParams {

    type NumberRing = Pow2CyclotomicDecomposableNumberRing;

    fn plaintext_modulus_(&self) -> i64 {
        self.t
    }

    fn number_ring_(&self) -> Self::NumberRing {
        Pow2CyclotomicDecomposableNumberRing::new(2 << self.log2_N)
    }

    fn ciphertext_modulus_bits_(&self) -> Range<usize> {
        self.log2_q_min..self.log2_q_max
    }
}

#[derive(Clone, Debug)]
pub struct CompositeDoubleRNSBFVParams {
    pub t: i64,
    pub log2_q_min: usize,
    pub log2_q_max: usize,
    pub n1: usize,
    pub n2: usize
}

impl DoubleRNSBFVParams for CompositeDoubleRNSBFVParams {

    type NumberRing = CompositeCyclotomicDecomposableNumberRing;

    fn plaintext_modulus_(&self) -> i64 {
        self.t
    }

    fn ciphertext_modulus_bits_(&self) -> Range<usize> {
        self.log2_q_min..self.log2_q_max
    }

    fn number_ring_(&self) -> Self::NumberRing {
        CompositeCyclotomicDecomposableNumberRing::new(self.n1, self.n2)
    }
}

#[test]
fn test_pow2_hom_galois() {
    let mut rng = thread_rng();
    
    let params = Pow2BFVParams {
        t: 3,
        log2_q_min: 100,
        log2_q_max: 120,
        log2_N: 7
    };
    
    let P = params.create_plaintext_ring(params.plaintext_modulus());
    let (C, _C_mul) = params.create_ciphertext_rings();    
    let sk = Pow2BFVParams::gen_sk(&C, &mut rng);
    let gk = Pow2BFVParams::gen_gk(&C, &mut rng, &sk, P.get_ring().cyclotomic_index_ring().int_hom().map(3));
    
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
        t: 257,
        log2_q_min: 1090,
        log2_q_max: 1100,
        log2_N: 10
    };
    
    let P = params.create_plaintext_ring(params.plaintext_modulus());
    let (C, C_mul) = params.create_ciphertext_rings();

    let sk = Pow2BFVParams::gen_sk(&C, &mut rng);
    let mul_rescale_data = Pow2BFVParams::create_multiplication_rescale(&P, &C, &C_mul);
    let rk = Pow2BFVParams::gen_rk(&C, &mut rng, &sk);

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
    
    let params = CompositeDoubleRNSBFVParams {
        t: 8,
        log2_q_min: 700,
        log2_q_max: 800,
        n1: 17,
        n2: 97
    };
    
    let P = params.create_plaintext_ring(params.plaintext_modulus());
    let (C, C_mul) = params.create_ciphertext_rings();

    let sk = CompositeDoubleRNSBFVParams::gen_sk(&C, &mut rng);
    let mul_rescale_data = CompositeDoubleRNSBFVParams::create_multiplication_rescale(&P, &C, &C_mul);
    let rk = CompositeDoubleRNSBFVParams::gen_rk(&C, &mut rng, &sk);

    let m = P.int_hom().map(2);
    let ct = CompositeDoubleRNSBFVParams::enc_sym(&P, &C, &mut rng, &m, &sk);

    let m = CompositeDoubleRNSBFVParams::dec(&P, &C, CompositeDoubleRNSBFVParams::clone_ct(&C, &ct), &sk);
    assert_el_eq!(&P, &P.int_hom().map(2), &m);
    
    let ct_sqr = CompositeDoubleRNSBFVParams::hom_mul(&C, &C_mul, CompositeDoubleRNSBFVParams::clone_ct(&C, &ct), CompositeDoubleRNSBFVParams::clone_ct(&C, &ct), &rk, &mul_rescale_data);
    let m_sqr = CompositeDoubleRNSBFVParams::dec(&P, &C, ct_sqr, &sk);

    assert_el_eq!(&P, &P.int_hom().map(4), &m_sqr);
}

#[test]
#[ignore]
fn print_timings_pow2_bfv_mul() {
    let mut rng = thread_rng();
    
    let params = Pow2BFVParams {
        t: 257,
        log2_q_min: 790,
        log2_q_max: 800,
        log2_N: 15
    };
    
    let P = params.create_plaintext_ring(params.plaintext_modulus());
    let (C, C_mul) = params.create_ciphertext_rings();

    let sk = Pow2BFVParams::gen_sk(&C, &mut rng);
    let mul_rescale_data = Pow2BFVParams::create_multiplication_rescale(&P, &C, &C_mul);
    let rk = Pow2BFVParams::gen_rk(&C, &mut rng, &sk);

    let m = P.int_hom().map(2);
    let ct = Pow2BFVParams::enc_sym(&P, &C, &mut rng, &m, &sk);

    let res = log_time::<_, _, true, _>("HomAddPlain", |[]| {
        Pow2BFVParams::hom_add_plain(&P, &C, &m, Pow2BFVParams::clone_ct(&C, &ct))
    });
    assert_el_eq!(&P, &P.int_hom().map(4), &Pow2BFVParams::dec(&P, &C, res, &sk));

    let res = log_time::<_, _, true, _>("HomAdd", |[]| {
        Pow2BFVParams::hom_add(&C, Pow2BFVParams::clone_ct(&C, &ct), &ct)
    });
    assert_el_eq!(&P, &P.int_hom().map(4), &Pow2BFVParams::dec(&P, &C, res, &sk));

    let res = log_time::<_, _, true, _>("HomMulPlain", |[]| {
        Pow2BFVParams::hom_mul_plain(&P, &C, &m, Pow2BFVParams::clone_ct(&C, &ct))
    });
    assert_el_eq!(&P, &P.int_hom().map(4), &Pow2BFVParams::dec(&P, &C, res, &sk));

    let res = log_time::<_, _, true, _>("HomMul", |[]| {
        Pow2BFVParams::hom_mul(&C, &C_mul, Pow2BFVParams::clone_ct(&C, &ct), Pow2BFVParams::clone_ct(&C, &ct), &rk, &mul_rescale_data)
    });
    assert_el_eq!(&P, &P.int_hom().map(4), &Pow2BFVParams::dec(&P, &C, res, &sk));

}
