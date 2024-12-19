#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use std::alloc::Allocator;
use std::alloc::Global;
use std::marker::PhantomData;
use std::time::Instant;
use std::ops::Range;
use std::cmp::max;

use feanor_math::algorithms::int_factor::factor;
use feanor_math::algorithms::int_factor::is_prime_power;
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
use feanor_math::rings::finite::FiniteRingStore;

use crate::cyclotomic::*;
use crate::digitextract::ArithCircuit;
use crate::euler_phi;
use crate::extend_sampled_primes;
use crate::lintransform::composite::powcoeffs_to_slots_fat;
use crate::lintransform::matmul::CompiledLinearTransform;
use crate::lintransform::HELinearTransform;
use crate::rings::bxv::BXVCiphertextRing;
use crate::rings::double_rns_managed::*;
use crate::rings::hypercube::CyclotomicGaloisGroup;
use crate::rings::hypercube::HypercubeIsomorphism;
use crate::rings::hypercube::HypercubeStructure;
use crate::rings::number_ring::*;
use crate::rings::decomposition_ring::*;
use crate::rings::odd_cyclotomic::*;
use crate::rings::pow2_cyclotomic::*;
use crate::profiling::*;
use crate::rings::single_rns_ring::SingleRNSRingBase;
use crate::rnsconv;
use crate::sample_primes;

use rand::thread_rng;
use rand::{Rng, CryptoRng};
use rand_distr::StandardNormal;

pub mod bootstrap;

#[cfg(feature = "use_hexl")]
pub type UsedConvolution = feanor_math_hexl::conv::HEXLConvolution;
#[cfg(not(feature = "use_hexl"))]
pub type UsedConvolution = crate::rings::ntt_conv::NTTConv<Zn>;

#[cfg(feature = "use_hexl")]
pub type UsedNegacyclicNTT = feanor_math_hexl::hexl::HEXLNegacyclicNTT;
#[cfg(not(feature = "use_hexl"))]
pub type UsedNegacyclicNTT = RustNegacyclicNTT<Zn>;

pub type PlaintextAllocator = Global;
pub type CiphertextAllocator = Global;
pub type RNSOperationAllocator = Global;
pub type NumberRing<Params: BFVParams> = <Params::CiphertextRing as BXVCiphertextRing>::NumberRing;
pub type PlaintextRing<Params: BFVParams> = DecompositionRing<NumberRing<Params>, Zn, PlaintextAllocator>;
pub type SecretKey<Params: BFVParams> = El<CiphertextRing<Params>>;
pub type KeySwitchKey<'a, Params: BFVParams> = (usize, (GadgetProductOperand<'a, Params>, GadgetProductOperand<'a, Params>));
pub type RelinKey<'a, Params: BFVParams> = KeySwitchKey<'a, Params>;
pub type CiphertextRing<Params: BFVParams> = RingValue<Params::CiphertextRing>;
pub type Ciphertext<Params: BFVParams> = (El<CiphertextRing<Params>>, El<CiphertextRing<Params>>);
pub type GadgetProductOperand<'a, Params: BFVParams> = <Params::CiphertextRing as BXVCiphertextRing>::GadgetProductRhsOperand<'a>;

const ZZbig: BigIntRing = BigIntRing::RING;
const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;

pub struct MulConversionData {
    pub lift_to_C_mul: rnsconv::shared_lift::AlmostExactSharedBaseConversion<RNSOperationAllocator>,
    pub scale_down_to_C: rnsconv::bfv_rescale::AlmostExactRescalingConvert<RNSOperationAllocator>
}

pub struct ModSwitchData {
    pub scale: rnsconv::bfv_rescale::AlmostExactRescaling<RNSOperationAllocator>
}

pub trait BFVParams {
    
    type CiphertextRing: BXVCiphertextRing + CyclotomicRing;

    fn ciphertext_modulus_bits(&self) -> Range<usize>;
    fn number_ring(&self) -> NumberRing<Self>;
    fn create_ciphertext_rings(&self) -> (CiphertextRing<Self>, CiphertextRing<Self>);

    fn create_plaintext_ring(&self, modulus: i64) -> PlaintextRing<Self> {
        DecompositionRingBase::new(self.number_ring(), Zn::new(modulus as u64))
    }

    fn gen_sk<R: Rng + CryptoRng>(C: &CiphertextRing<Self>, mut rng: R) -> SecretKey<Self> {
        // we sample uniform ternary secrets 
        let result = C.get_ring().sample_from_coefficient_distribution(|| (rng.next_u32() % 3) as i32 - 1);
        return result;
    }
    
    fn enc_sym_zero<R: Rng + CryptoRng>(C: &CiphertextRing<Self>, mut rng: R, sk: &SecretKey<Self>) -> Ciphertext<Self> {
        let a = C.random_element(|| rng.next_u64());
        let mut b = C.negate(C.mul_ref(&a, &sk));
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
        println!("ciphertext (noise budget: {}):", Self::noise_budget(P, C, ct, sk));
        P.println(&m);
        println!();
    }
    
    fn dec_println_slots(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: &Ciphertext<Self>, sk: &SecretKey<Self>) {
        let (p, _e) = is_prime_power(ZZ, P.base_ring().modulus()).unwrap();
        let hypercube = HypercubeStructure::halevi_shoup_hypercube(CyclotomicGaloisGroup::new(P.n()), p);
        let H = HypercubeIsomorphism::new::<false>(P, hypercube);
        let m = Self::dec(P, C, Self::clone_ct(C, ct), sk);
        println!("ciphertext (noise budget: {}):", Self::noise_budget(P, C, ct, sk));
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
    
    fn gen_rk<'a, R: Rng + CryptoRng>(C: &'a CiphertextRing<Self>, rng: R, sk: &SecretKey<Self>, digits: usize) -> RelinKey<'a, Self>
        where Self: 'a
    {
        Self::gen_switch_key(C, rng, &C.pow(C.clone_el(sk), 2), sk, digits)
    }
    
    fn hom_mul<'a>(C: &CiphertextRing<Self>, C_mul: &CiphertextRing<Self>, lhs: Ciphertext<Self>, rhs: Ciphertext<Self>, rk: &RelinKey<'a, Self>, conv_data: &MulConversionData) -> Ciphertext<Self>
        where Self: 'a
    {
        let (c00, c01) = lhs;
        let (c10, c11) = rhs;
        let lift = |c| C_mul.get_ring().perform_rns_op_from(C.get_ring(), &c, &conv_data.lift_to_C_mul);
    
        let c00_lifted = lift(c00);
        let c01_lifted = lift(c01);
        let c10_lifted = lift(c10);
        let c11_lifted = lift(c11);
    
        let [lifted0, lifted1, lifted2] = C_mul.get_ring().two_by_two_convolution([&c00_lifted, &c01_lifted], [&c10_lifted, &c11_lifted]);
    
        let scale_down = |c: El<CiphertextRing<Self>>| C.get_ring().perform_rns_op_from(C_mul.get_ring(), &c, &conv_data.scale_down_to_C);
    
        let res0 = scale_down(lifted0);
        let res1 = scale_down(lifted1);
        let res2 = scale_down(lifted2);
    
        let op = C.get_ring().to_gadget_product_lhs(res2, rk.0);
        let (s0, s1) = &rk.1;
    
        return (C.add_ref(&res0, &C.get_ring().gadget_product(&op, s0)), C.add_ref(&res1, &C.get_ring().gadget_product(&op, s1)));
    }
    
    fn gen_switch_key<'a, R: Rng + CryptoRng>(C: &'a CiphertextRing<Self>, mut rng: R, old_sk: &SecretKey<Self>, new_sk: &SecretKey<Self>, digits: usize) -> KeySwitchKey<'a, Self>
        where Self: 'a
    {
        let mut res0 = C.get_ring().gadget_product_rhs_empty(digits);
        let mut res1 = C.get_ring().gadget_product_rhs_empty(digits);
        for digit_i in 0..C.get_ring().gadget_vector(&res0).len() {
            let (c0, c1) = Self::enc_sym_zero(C, &mut rng, new_sk);
            let digit_range = C.get_ring().gadget_vector(&res0).at(digit_i).clone();
            let factor = C.base_ring().get_ring().from_congruence((0..C.base_ring().len()).map(|i2| {
                let Fp = C.base_ring().at(i2);
                if digit_range.contains(&i2) { Fp.one() } else { Fp.zero() } 
            }));
            let mut payload = C.clone_el(&old_sk);
            C.inclusion().mul_assign_ref_map(&mut payload, &factor);
            C.add_assign_ref(&mut payload, &c0);
            C.get_ring().set_rns_factor(&mut res0, digit_i, payload);
            C.get_ring().set_rns_factor(&mut res1, digit_i, c1);
        }
        return (digits, (res0, res1));
    }
    
    fn key_switch<'a>(C: &CiphertextRing<Self>, ct: Ciphertext<Self>, switch_key: &KeySwitchKey<'a, Self>) -> Ciphertext<Self>
        where Self: 'a
    {
        let (c0, c1) = ct;
        let (s0, s1) = &switch_key.1;
        let op = C.get_ring().to_gadget_product_lhs(c1, switch_key.0);
        return (
            C.add_ref_snd(c0, &C.get_ring().gadget_product(&op, s0)),
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
    
    fn gen_gk<'a, R: Rng + CryptoRng>(C: &'a CiphertextRing<Self>, rng: R, sk: &SecretKey<Self>, g: ZnEl, digits: usize) -> KeySwitchKey<'a, Self>
        where Self: 'a
    {
        Self::gen_switch_key(C, rng, &C.get_ring().apply_galois_action(sk, g), sk, digits)
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
        let c1_op = C.get_ring().to_gadget_product_lhs(c1, digits);
        let c1_op_gs = C.get_ring().apply_galois_action_many_gadget_product_operand(&c1_op, gs);
        let c0_gs = C.get_ring().apply_galois_action_many(&c0, gs);
        assert_eq!(gks.len(), c1_op_gs.len());
        assert_eq!(gks.len(), c0_gs.len());
        return c0_gs.zip(c1_op_gs.iter()).enumerate().map(|(i, (c0_g, c1_g))| {
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
pub struct Pow2BFV {
    pub log2_q_min: usize,
    pub log2_q_max: usize,
    pub log2_N: usize
}

impl BFVParams for Pow2BFV {

    type CiphertextRing = ManagedDoubleRNSRingBase<Pow2CyclotomicNumberRing<UsedNegacyclicNTT>, CiphertextAllocator>;

    fn number_ring(&self) -> Pow2CyclotomicNumberRing<UsedNegacyclicNTT> {
        Pow2CyclotomicNumberRing::new_with(2 << self.log2_N)
    }

    fn ciphertext_modulus_bits(&self) -> Range<usize> {
        self.log2_q_min..self.log2_q_max
    }

    fn create_ciphertext_rings(&self) -> (CiphertextRing<Self>, CiphertextRing<Self>)  {
        let log2_q = self.ciphertext_modulus_bits();
        let number_ring = self.number_ring();

        let mut C_rns_base = sample_primes(log2_q.start, log2_q.end, 56, |bound| <_ as HENumberRing>::largest_suitable_prime(&number_ring, int_cast(bound, ZZ, ZZbig)).map(|p| int_cast(p, ZZbig, ZZ))).unwrap();
        C_rns_base.sort_unstable_by(|l, r| ZZbig.cmp(l, r));

        let mut Cmul_rns_base = extend_sampled_primes(&C_rns_base, log2_q.end * 2, log2_q.end * 2 + 57, 57, |bound| <_ as HENumberRing>::largest_suitable_prime(&number_ring, int_cast(bound, ZZ, ZZbig)).map(|p| int_cast(p, ZZbig, ZZ))).unwrap();
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
}

#[derive(Clone, Debug)]
pub struct CompositeBFV {
    pub log2_q_min: usize,
    pub log2_q_max: usize,
    pub n1: usize,
    pub n2: usize
}

impl BFVParams for CompositeBFV {

    type CiphertextRing = ManagedDoubleRNSRingBase<CompositeCyclotomicNumberRing, CiphertextAllocator>;

    fn ciphertext_modulus_bits(&self) -> Range<usize> {
        self.log2_q_min..self.log2_q_max
    }

    fn number_ring(&self) -> CompositeCyclotomicNumberRing {
        CompositeCyclotomicNumberRing::new(self.n1, self.n2)
    }

    fn create_ciphertext_rings(&self) -> (CiphertextRing<Self>, CiphertextRing<Self>)  {
        let log2_q = self.ciphertext_modulus_bits();
        let number_ring = self.number_ring();

        let mut C_rns_base = sample_primes(log2_q.start, log2_q.end, 56, |bound| <_ as HENumberRing>::largest_suitable_prime(&number_ring, int_cast(bound, ZZ, ZZbig)).map(|p| int_cast(p, ZZbig, ZZ))).unwrap();
        C_rns_base.sort_unstable_by(|l, r| ZZbig.cmp(l, r));

        let mut Cmul_rns_base = extend_sampled_primes(&C_rns_base, log2_q.end * 2, log2_q.end * 2 + 57, 57, |bound| <_ as HENumberRing>::largest_suitable_prime(&number_ring, int_cast(bound, ZZ, ZZbig)).map(|p| int_cast(p, ZZbig, ZZ))).unwrap();
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
}

#[derive(Clone, Debug)]
pub struct CompositeSingleRNSBFV {
    pub log2_q_min: usize,
    pub log2_q_max: usize,
    pub n1: usize,
    pub n2: usize
}

impl BFVParams for CompositeSingleRNSBFV {

    type CiphertextRing = SingleRNSRingBase<CompositeCyclotomicNumberRing, CiphertextAllocator, UsedConvolution>;

    fn ciphertext_modulus_bits(&self) -> Range<usize> {
        self.log2_q_min..self.log2_q_max
    }

    fn number_ring(&self) -> CompositeCyclotomicNumberRing {
        CompositeCyclotomicNumberRing::new(self.n1, self.n2)
    }

    fn create_ciphertext_rings(&self) -> (CiphertextRing<Self>, CiphertextRing<Self>) {
        let log2_q = self.ciphertext_modulus_bits();
        let number_ring = self.number_ring();

        let mut C_rns_base = sample_primes(log2_q.start, log2_q.end, 56, |bound| <_ as HENumberRing>::largest_suitable_prime(&number_ring, int_cast(bound, ZZ, ZZbig)).map(|p| int_cast(p, ZZbig, ZZ))).unwrap();
        C_rns_base.sort_unstable_by(|l, r| ZZbig.cmp(l, r));

        let mut Cmul_rns_base = extend_sampled_primes(&C_rns_base, log2_q.end * 2, log2_q.end * 2 + 57, 57, |bound| <_ as HENumberRing>::largest_suitable_prime(&number_ring, int_cast(bound, ZZ, ZZbig)).map(|p| int_cast(p, ZZbig, ZZ))).unwrap();
        assert!(ZZbig.is_gt(&Cmul_rns_base[Cmul_rns_base.len() - 1], &C_rns_base[C_rns_base.len() - 1]));
        Cmul_rns_base.sort_unstable_by(|l, r| ZZbig.cmp(l, r));

        let C = SingleRNSRingBase::<_, _, UsedConvolution>::new(
            self.number_ring(),
            zn_rns::Zn::new(C_rns_base.iter().map(|p| Zn::new(int_cast(ZZbig.clone_el(p), ZZ, ZZbig) as u64)).collect(), ZZbig)
        );
        let Cmul = SingleRNSRingBase::<_, _, UsedConvolution>::new(
            number_ring,
            zn_rns::Zn::new(Cmul_rns_base.iter().map(|p| Zn::new(int_cast(ZZbig.clone_el(p), ZZ, ZZbig) as u64)).collect(), ZZbig)
        );
        return (C, Cmul);
    }
}

pub fn hom_compute_linear_transform<'a, Params, Transform, const LOG: bool>(
    P: &PlaintextRing<Params>, 
    C: &CiphertextRing<Params>, 
    input: Ciphertext<Params>, 
    transform: &[Transform], 
    gk: &[(ZnEl, KeySwitchKey<'a, Params>)], 
    key_switches: &mut usize
) -> Ciphertext<Params>
    where Params: 'a + BFVParams,
        Transform: HELinearTransform<NumberRing<Params>, Global>
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
            let result = log_time::<_, _, LOG, _>(format!("Computing {} galois automorphisms", gs.len()).as_str(), |[]| 
                Params::hom_galois_many(C, value, gs, gs.as_fn().map_fn(|g| get_gk(g)))
            );
            return result;
        },
        |value| Params::clone_ct(C, value)
    ));
}

pub fn hom_evaluate_circuit<'a, 'b, Params: BFVParams>(
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

pub fn coeff_repr<Params, NumberRing, A>(C: &CiphertextRing<Params>, ct: Ciphertext<Params>) -> Ciphertext<Params>
    where Params: BFVParams<CiphertextRing = ManagedDoubleRNSRingBase<NumberRing, A>>,
        NumberRing: HECyclotomicNumberRing,
        A: Allocator + Clone
{
    C.get_ring().force_coeff_repr(&ct.0);
    C.get_ring().force_coeff_repr(&ct.1);
    return ct;
}

#[test]
fn test_pow2_bfv_hom_galois() {
    let mut rng = thread_rng();
    
    let params = Pow2BFV {
        log2_q_min: 500,
        log2_q_max: 520,
        log2_N: 7
    };
    let t = 3;
    let digits = 3;
    
    let P = params.create_plaintext_ring(t);
    let (C, _C_mul) = params.create_ciphertext_rings();    
    let sk = Pow2BFV::gen_sk(&C, &mut rng);
    let gk = Pow2BFV::gen_gk(&C, &mut rng, &sk, P.get_ring().cyclotomic_index_ring().int_hom().map(3), digits);
    
    let m = P.canonical_gen();
    let ct = Pow2BFV::enc_sym(&P, &C, &mut rng, &m, &sk);
    let ct_res = Pow2BFV::hom_galois(&C, ct, P.get_ring().cyclotomic_index_ring().int_hom().map(3), &gk);
    let res = Pow2BFV::dec(&P, &C, ct_res, &sk);

    assert_el_eq!(&P, &P.pow(P.canonical_gen(), 3), &res);
}

#[test]
fn test_pow2_bfv_mul() {
    let mut rng = thread_rng();
    
    let params = Pow2BFV {
        log2_q_min: 500,
        log2_q_max: 520,
        log2_N: 10
    };
    let t = 257;
    let digits = 3;
    
    let P = params.create_plaintext_ring(t);
    let (C, C_mul) = params.create_ciphertext_rings();

    let sk = Pow2BFV::gen_sk(&C, &mut rng);
    let mul_rescale_data = Pow2BFV::create_multiplication_rescale(&P, &C, &C_mul);
    let rk = Pow2BFV::gen_rk(&C, &mut rng, &sk, digits);

    let m = P.int_hom().map(2);
    let ct = Pow2BFV::enc_sym(&P, &C, &mut rng, &m, &sk);

    let m = Pow2BFV::dec(&P, &C, Pow2BFV::clone_ct(&C, &ct), &sk);
    assert_el_eq!(&P, &P.int_hom().map(2), &m);

    let ct_sqr = Pow2BFV::hom_mul(&C, &C_mul, Pow2BFV::clone_ct(&C, &ct), Pow2BFV::clone_ct(&C, &ct), &rk, &mul_rescale_data);
    let m_sqr = Pow2BFV::dec(&P, &C, ct_sqr, &sk);

    assert_el_eq!(&P, &P.int_hom().map(4), &m_sqr);
}

#[test]
fn test_composite_bfv_mul() {
    let mut rng = thread_rng();
    
    let params = CompositeBFV {
        log2_q_min: 500,
        log2_q_max: 520,
        n1: 17,
        n2: 97
    };
    let t = 8;
    let digits = 3;
    
    let P = params.create_plaintext_ring(t);
    let (C, C_mul) = params.create_ciphertext_rings();

    let sk = CompositeBFV::gen_sk(&C, &mut rng);
    let mul_rescale_data = CompositeBFV::create_multiplication_rescale(&P, &C, &C_mul);
    let rk = CompositeBFV::gen_rk(&C, &mut rng, &sk, digits);

    let m = P.int_hom().map(2);
    let ct = CompositeBFV::enc_sym(&P, &C, &mut rng, &m, &sk);

    let m = CompositeBFV::dec(&P, &C, CompositeBFV::clone_ct(&C, &ct), &sk);
    assert_el_eq!(&P, &P.int_hom().map(2), &m);
    
    let ct_sqr = CompositeBFV::hom_mul(&C, &C_mul, CompositeBFV::clone_ct(&C, &ct), CompositeBFV::clone_ct(&C, &ct), &rk, &mul_rescale_data);
    let m_sqr = CompositeBFV::dec(&P, &C, ct_sqr, &sk);

    assert_el_eq!(&P, &P.int_hom().map(4), &m_sqr);
}

#[test]
#[ignore]
fn print_timings_pow2_bfv_mul() {
    let mut rng = thread_rng();
    
    let params = Pow2BFV {
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

    let sk = log_time::<_, _, true, _>("GenSK", |[]| 
        Pow2BFV::gen_sk(&C, &mut rng)
    );
    let mul_rescale_data = log_time::<_, _, true, _>("CreateMulRescale", |[]|
        Pow2BFV::create_multiplication_rescale(&P, &C, &C_mul)
    );

    let m = P.int_hom().map(2);
    let ct = log_time::<_, _, true, _>("EncSym", |[]|
        coeff_repr::<Pow2BFV, _, _>(&C, Pow2BFV::enc_sym(&P, &C, &mut rng, &m, &sk))
    );

    let res = log_time::<_, _, true, _>("HomAddPlain", |[]| 
        coeff_repr::<Pow2BFV, _, _>(&C, Pow2BFV::hom_add_plain(&P, &C, &m, Pow2BFV::clone_ct(&C, &ct)))
    );
    assert_el_eq!(&P, &P.int_hom().map(4), &Pow2BFV::dec(&P, &C, res, &sk));

    let res = log_time::<_, _, true, _>("HomAdd", |[]| 
        coeff_repr::<Pow2BFV, _, _>(&C, Pow2BFV::hom_add(&C, Pow2BFV::clone_ct(&C, &ct), &ct))
    );
    assert_el_eq!(&P, &P.int_hom().map(4), &Pow2BFV::dec(&P, &C, res, &sk));

    let res = log_time::<_, _, true, _>("HomMulPlain", |[]| 
        coeff_repr::<Pow2BFV, _, _>(&C, Pow2BFV::hom_mul_plain(&P, &C, &m, Pow2BFV::clone_ct(&C, &ct)))
    );
    assert_el_eq!(&P, &P.int_hom().map(4), &Pow2BFV::dec(&P, &C, res, &sk));

    let rk = log_time::<_, _, true, _>("GenRK", |[]| 
        Pow2BFV::gen_rk(&C, &mut rng, &sk, digits)
    );
    clear_all_timings();
    let res = log_time::<_, _, true, _>("HomMul", |[]| 
        coeff_repr::<Pow2BFV, _, _>(&C, Pow2BFV::hom_mul(&C, &C_mul, Pow2BFV::clone_ct(&C, &ct), Pow2BFV::clone_ct(&C, &ct), &rk, &mul_rescale_data))
    );
    print_all_timings();
    assert_el_eq!(&P, &P.int_hom().map(4), &Pow2BFV::dec(&P, &C, res, &sk));
}

#[test]
#[ignore]
fn print_timings_double_rns_composite_bfv_mul() {
    let mut rng = thread_rng();
    
    let params = CompositeBFV {
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
        CompositeBFV::gen_sk(&C, &mut rng)
    );
    let mul_rescale_data = log_time::<_, _, true, _>("CreateMulRescale", |[]|
        CompositeBFV::create_multiplication_rescale(&P, &C, &C_mul)
    );

    let m = P.int_hom().map(3);
    let ct = log_time::<_, _, true, _>("EncSym", |[]|
        coeff_repr::<CompositeBFV, _, _>(&C, CompositeBFV::enc_sym(&P, &C, &mut rng, &m, &sk))
    );
    assert_el_eq!(&P, &P.int_hom().map(3), &CompositeBFV::dec(&P, &C, CompositeBFV::clone_ct(&C, &ct), &sk));

    let res = log_time::<_, _, true, _>("HomAdd", |[]| 
        coeff_repr::<CompositeBFV, _, _>(&C, CompositeBFV::hom_add(&C, CompositeBFV::clone_ct(&C, &ct), &ct))
    );
    assert_el_eq!(&P, &P.int_hom().map(2), &CompositeBFV::dec(&P, &C, res, &sk));

    let res = log_time::<_, _, true, _>("HomAddPlain", |[]| 
        coeff_repr::<CompositeBFV, _, _>(&C, CompositeBFV::hom_add_plain(&P, &C, &m, CompositeBFV::clone_ct(&C, &ct)))
    );
    assert_el_eq!(&P, &P.int_hom().map(2), &CompositeBFV::dec(&P, &C, res, &sk));

    let res = log_time::<_, _, true, _>("HomMulPlain", |[]| 
        coeff_repr::<CompositeBFV, _, _>(&C, CompositeBFV::hom_mul_plain(&P, &C, &m, CompositeBFV::clone_ct(&C, &ct)))
    );
    assert_el_eq!(&P, &P.int_hom().map(1), &CompositeBFV::dec(&P, &C, res, &sk));

    let rk = log_time::<_, _, true, _>("GenRK", |[]| 
        CompositeBFV::gen_rk(&C, &mut rng, &sk, digits)
    );
    clear_all_timings();
    let res = log_time::<_, _, true, _>("HomMul", |[]| 
        coeff_repr::<CompositeBFV, _, _>(&C, CompositeBFV::hom_mul(&C, &C_mul, CompositeBFV::clone_ct(&C, &ct), CompositeBFV::clone_ct(&C, &ct), &rk, &mul_rescale_data))
    );
    print_all_timings();
    assert_el_eq!(&P, &P.int_hom().map(1), &CompositeBFV::dec(&P, &C, res, &sk));
}


#[test]
fn test_bfv_hom_galois() {
    let mut rng = thread_rng();
    
    let params = CompositeSingleRNSBFV {
        log2_q_min: 500,
        log2_q_max: 520,
        n1: 7,
        n2: 11
    };
    let t = 3;
    let digits = 3;
    
    let P = params.create_plaintext_ring(t);
    let (C, _C_mul) = params.create_ciphertext_rings();    
    let sk = CompositeSingleRNSBFV::gen_sk(&C, &mut rng);
    let gk = CompositeSingleRNSBFV::gen_gk(&C, &mut rng, &sk, P.get_ring().cyclotomic_index_ring().int_hom().map(3), digits);
    
    let m = P.canonical_gen();
    let ct = CompositeSingleRNSBFV::enc_sym(&P, &C, &mut rng, &m, &sk);
    let ct_res = CompositeSingleRNSBFV::hom_galois(&C, ct, P.get_ring().cyclotomic_index_ring().int_hom().map(3), &gk);
    let res = CompositeSingleRNSBFV::dec(&P, &C, ct_res, &sk);

    assert_el_eq!(&P, &P.pow(P.canonical_gen(), 3), &res);
}

#[test]
fn test_single_rns_composite_bfv_mul() {
    let mut rng = thread_rng();
    
    let params = CompositeSingleRNSBFV {
        log2_q_min: 500,
        log2_q_max: 520,
        n1: 7,
        n2: 11
    };
    let t = 3;
    let digits = 3;

    let P = params.create_plaintext_ring(t);
    let (C, C_mul) = params.create_ciphertext_rings();

    let sk = CompositeSingleRNSBFV::gen_sk(&C, &mut rng);
    let mul_rescale_data = CompositeSingleRNSBFV::create_multiplication_rescale(&P, &C, &C_mul);
    let rk = CompositeSingleRNSBFV::gen_rk(&C, &mut rng, &sk, digits);

    let m = P.int_hom().map(2);
    let ct = CompositeSingleRNSBFV::enc_sym(&P, &C, &mut rng, &m, &sk);

    let m = CompositeSingleRNSBFV::dec(&P, &C, CompositeSingleRNSBFV::clone_ct(&C, &ct), &sk);
    assert_el_eq!(&P, &P.int_hom().map(2), &m);

    let ct_sqr = CompositeSingleRNSBFV::hom_mul(&C, &C_mul, CompositeSingleRNSBFV::clone_ct(&C, &ct), CompositeSingleRNSBFV::clone_ct(&C, &ct), &rk, &mul_rescale_data);
    let m_sqr = CompositeSingleRNSBFV::dec(&P, &C, ct_sqr, &sk);

    assert_el_eq!(&P, &P.int_hom().map(4), &m_sqr);
}

#[test]
#[ignore]
fn print_timings_single_rns_composite_bfv_mul() {
    let mut rng = thread_rng();
    
    let params = CompositeSingleRNSBFV {
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
        CompositeSingleRNSBFV::gen_sk(&C, &mut rng)
    );
    let mul_rescale_data = log_time::<_, _, true, _>("CreateMulRescale", |[]|
        CompositeSingleRNSBFV::create_multiplication_rescale(&P, &C, &C_mul)
    );

    let m = P.int_hom().map(3);
    let ct = log_time::<_, _, true, _>("EncSym", |[]|
        CompositeSingleRNSBFV::enc_sym(&P, &C, &mut rng, &m, &sk)
    );

    let res = log_time::<_, _, true, _>("HomAddPlain", |[]| 
        CompositeSingleRNSBFV::hom_add_plain(&P, &C, &m, CompositeSingleRNSBFV::clone_ct(&C, &ct))
    );
    assert_el_eq!(&P, &P.int_hom().map(2), &CompositeSingleRNSBFV::dec(&P, &C, res, &sk));

    let res = log_time::<_, _, true, _>("HomAdd", |[]| 
        CompositeSingleRNSBFV::hom_add(&C, CompositeSingleRNSBFV::clone_ct(&C, &ct), &ct)
    );
    assert_el_eq!(&P, &P.int_hom().map(2), &CompositeSingleRNSBFV::dec(&P, &C, res, &sk));

    let res = log_time::<_, _, true, _>("HomMulPlain", |[]| 
        CompositeSingleRNSBFV::hom_mul_plain(&P, &C, &m, CompositeSingleRNSBFV::clone_ct(&C, &ct))
    );
    assert_el_eq!(&P, &P.int_hom().map(1), &CompositeSingleRNSBFV::dec(&P, &C, res, &sk));

    let rk = log_time::<_, _, true, _>("GenRK", |[]| 
        CompositeSingleRNSBFV::gen_rk(&C, &mut rng, &sk, digits)
    );
    clear_all_timings();
    let res = log_time::<_, _, true, _>("HomMul", |[]| 
        CompositeSingleRNSBFV::hom_mul(&C, &C_mul, CompositeSingleRNSBFV::clone_ct(&C, &ct), CompositeSingleRNSBFV::clone_ct(&C, &ct), &rk, &mul_rescale_data)
    );
    print_all_timings();
    assert_el_eq!(&P, &P.int_hom().map(1), &CompositeSingleRNSBFV::dec(&P, &C, res, &sk));
}

#[test]
#[ignore]
fn test_hom_eval_powcoeffs_to_slots_fat_large() {let mut rng = thread_rng();
    let params = CompositeSingleRNSBFV {
        log2_q_min: 790,
        log2_q_max: 800,
        n1: 127,
        n2: 337
    };
    let t = 65536;
    let digits = 3;
    
    let P = log_time::<_, _, true, _>("CreatePtxtRing", |[]|
        params.create_plaintext_ring(t)
    );
    let (C, C_mul) = log_time::<_, _, true, _>("CreateCtxtRing", |[]|
        params.create_ciphertext_rings()
    );

    let sk = log_time::<_, _, true, _>("GenSK", |[]| 
        CompositeSingleRNSBFV::gen_sk(&C, &mut rng)
    );

    let hypercube = HypercubeStructure::halevi_shoup_hypercube(CyclotomicGaloisGroup::new(337 * 127), 2);
    let H = HypercubeIsomorphism::new::<true>(&P, hypercube);
    assert_eq!(337, H.hypercube().factor_of_n(0).unwrap());
    assert_eq!(16, H.hypercube().m(0));
    assert_eq!(127, H.hypercube().factor_of_n(1).unwrap());
    assert_eq!(126, H.hypercube().m(1));

    let transform = log_time::<_, _, true, _>("CreateTransform", |[]| 
        powcoeffs_to_slots_fat(&H).into_iter().map(|t| CompiledLinearTransform::compile(&H, t)).collect::<Vec<_>>()
    );

    let m = P.int_hom().map(3);
    let ct = log_time::<_, _, true, _>("EncSym", |[]|
        CompositeSingleRNSBFV::enc_sym(&P, &C, &mut rng, &m, &sk)
    );

    let gks = log_time::<_, _, true, _>("GenGK", |[]| 
        transform.iter().flat_map(|t| t.required_galois_keys().into_iter()).map(|g| (g, CompositeSingleRNSBFV::gen_gk(&C, &mut rng, &sk, g, digits))).collect::<Vec<_>>()
    );

    clear_all_timings();
    let result = log_time::<_, _, true, _>("ApplyTransform", |[key_switches]| 
        hom_compute_linear_transform::<CompositeSingleRNSBFV, _, true>(&P, &C, ct, &transform, &gks, key_switches)
    );
    print_all_timings();
}