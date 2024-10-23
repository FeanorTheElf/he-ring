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

use crate::cyclotomic::CyclotomicRing;
use crate::euler_phi;
use crate::extend_sampled_primes;
use crate::rings::decomposition::*;
use crate::rings::decomposition_ring::*;
use crate::rings::odd_cyclotomic::*;
use crate::rings::pow2_cyclotomic::*;
use crate::rings::gadget_product::*;
use crate::rings::double_rns_ring::*;
use crate::profiling::*;
use crate::rings::slots::HypercubeIsomorphism;
use crate::rnsconv;
use crate::sample_primes;

use rand::thread_rng;
use rand::{Rng, CryptoRng};
use rand_distr::StandardNormal;

pub mod bootstrap;
pub mod double_rns;
pub mod single_rns;

pub type PlaintextAllocator = Global;
pub type CiphertextAllocator = Global;
pub type PlaintextRing<Params: BFVParams> = DecompositionRing<Params::NumberRing, Zn, PlaintextAllocator>;
pub type SecretKey<Params: BFVParams> = El<CiphertextRing<Params>>;
pub type KeySwitchKey<'a, Params: BFVParams> = Params::KeySwitchKey<'a>;
pub type RelinKey<'a, Params: BFVParams> = Params::KeySwitchKey<'a>;
pub type CiphertextRing<Params: BFVParams> = Params::CiphertextRing;
pub type Ciphertext<Params: BFVParams> = Params::Ciphertext;

const ZZbig: BigIntRing = BigIntRing::RING;
const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;

pub struct MulConversionData {
    lift_to_C_mul: rnsconv::shared_lift::AlmostExactSharedBaseConversion<CiphertextAllocator>,
    scale_down_to_C: rnsconv::bfv_rescale::AlmostExactRescalingConvert<CiphertextAllocator>
}

pub struct ModSwitchData {
    scale: rnsconv::bfv_rescale::AlmostExactRescaling<CiphertextAllocator>
}

pub trait BFVParams: Sized {

    type NumberRing: DecomposableCyclotomicNumberRing<Zn>;
    type CiphertextRing: RingStore<Type = Self::CiphertextRingBase>;
    type CiphertextRingBase: CyclotomicRing + RingExtension<BaseRing = zn_rns::Zn<Zn, BigIntRing>>;
    type Ciphertext;
    type KeySwitchKey<'a>
        where Self: 'a;

    fn ciphertext_modulus_bits(&self) -> Range<usize>;

    fn number_ring(&self) -> Self::NumberRing;

    fn plaintext_modulus(&self) -> i64;

    fn create_plaintext_ring(&self, modulus: i64) -> PlaintextRing<Self>;

    fn create_ciphertext_rings(&self) -> (CiphertextRing<Self>, CiphertextRing<Self>);

    fn gen_sk<R: Rng + CryptoRng>(C: &CiphertextRing<Self>, rng: R) -> SecretKey<Self>;

    fn enc_sym_zero<R: Rng + CryptoRng>(C: &CiphertextRing<Self>, rng: R, sk: &SecretKey<Self>) -> Ciphertext<Self>;

    fn transparent_zero(C: &CiphertextRing<Self>) -> Ciphertext<Self>;

    fn enc_sym<R: Rng + CryptoRng>(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, rng: R, m: &El<PlaintextRing<Self>>, sk: &SecretKey<Self>) -> Ciphertext<Self>;

    fn enc_sk(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>) -> Ciphertext<Self>;
    
    fn remove_noise(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, c: &El<CiphertextRing<Self>>) -> El<PlaintextRing<Self>>;
    
    fn dec(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: Ciphertext<Self>, sk: &SecretKey<Self>) -> El<PlaintextRing<Self>>;
    
    fn dec_println(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: &Ciphertext<Self>, sk: &SecretKey<Self>);
    
    fn dec_println_slots(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: &Ciphertext<Self>, sk: &SecretKey<Self>);
    
    fn hom_add(C: &CiphertextRing<Self>, lhs: Ciphertext<Self>, rhs: &Ciphertext<Self>) -> Ciphertext<Self>;
    
    fn hom_sub(C: &CiphertextRing<Self>, lhs: Ciphertext<Self>, rhs: &Ciphertext<Self>) -> Ciphertext<Self>;
    
    fn clone_ct(C: &CiphertextRing<Self>, ct: &Ciphertext<Self>) -> Ciphertext<Self>;
    
    fn hom_add_plain(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: &El<PlaintextRing<Self>>, ct: Ciphertext<Self>) -> Ciphertext<Self>;
    
    fn hom_mul_plain(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: &El<PlaintextRing<Self>>, ct: Ciphertext<Self>) -> Ciphertext<Self>;
    
    fn hom_mul_plain_i64(_P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: i64, ct: Ciphertext<Self>) -> Ciphertext<Self>;
    
    fn noise_budget(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: &Ciphertext<Self>, sk: &SecretKey<Self>) -> usize;
    
    fn gen_rk<'a, R: Rng + CryptoRng>(C: &'a CiphertextRing<Self>, rng: R, sk: &SecretKey<Self>) -> RelinKey<'a, Self>
        where Self: 'a;
    
    fn hom_mul<'a>(C: &CiphertextRing<Self>, C_mul: &CiphertextRing<Self>, lhs: Ciphertext<Self>, rhs: Ciphertext<Self>, rk: &RelinKey<'a, Self>, conv_data: &MulConversionData) -> Ciphertext<Self>
        where Self: 'a;
    
    fn gen_switch_key<'a, R: Rng + CryptoRng>(C: &'a CiphertextRing<Self>, rng: R, old_sk: &SecretKey<Self>, new_sk: &SecretKey<Self>) -> KeySwitchKey<'a, Self>
        where Self: 'a;
    
    fn key_switch<'a>(C: &CiphertextRing<Self>, ct: Ciphertext<Self>, switch_key: &KeySwitchKey<'a, Self>) -> Ciphertext<Self>
        where Self: 'a;
    
    fn mod_switch_to_plaintext(target: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: Ciphertext<Self>, switch_data: &ModSwitchData) -> (El<PlaintextRing<Self>>, El<PlaintextRing<Self>>);
    
    fn gen_gk<'a, R: Rng + CryptoRng>(C: &'a CiphertextRing<Self>, rng: R, sk: &SecretKey<Self>, g: ZnEl) -> KeySwitchKey<'a, Self>
        where Self: 'a;
    
    fn hom_galois<'a>(C: &CiphertextRing<Self>, ct: Ciphertext<Self>, g: ZnEl, gk: &KeySwitchKey<'a, Self>) -> Ciphertext<Self>
        where Self: 'a;
    
    fn hom_galois_many<'a, 'b, V>(C: &CiphertextRing<Self>, ct: Ciphertext<Self>, gs: &[ZnEl], gks: V) -> Vec<Ciphertext<Self>>
        where V: VectorFn<&'b KeySwitchKey<'a, Self>>,
            KeySwitchKey<'a, Self>: 'b,
            'a: 'b,
            Self: 'a;

    fn create_multiplication_rescale(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, Cmul: &CiphertextRing<Self>) -> MulConversionData;
}
