#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use std::alloc::Allocator;
use std::alloc::Global;
use std::marker::PhantomData;
use std::ptr::Alignment;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;
use std::ops::Range;
use std::cmp::max;

use feanor_math::algorithms::convolution::PreparedConvolutionAlgorithm;
use feanor_math::algorithms::eea::signed_lcm;
use feanor_math::algorithms::int_factor::{factor, is_prime_power};
use feanor_math::algorithms::miller_rabin::is_prime;
use feanor_math::algorithms::unity_root::{get_prim_root_of_unity, get_prim_root_of_unity_pow2};
use feanor_math::homomorphism::*;
use feanor_math::ring::*;
use feanor_math::assert_el_eq;
use feanor_math::rings::finite::FiniteRing;
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
use feanor_mempool::dynsize::DynLayoutMempool;
use feanor_mempool::AllocArc;
use feanor_mempool::AllocRc;

use crate::ciphertext_ring::perform_rns_op;
use crate::ciphertext_ring::perform_rns_op_to_plaintext_ring;
use crate::ciphertext_ring::BGFVCiphertextRing;
use crate::cyclotomic::*;
use crate::digitextract::ArithCircuit;
use crate::euler_phi;
use crate::extend_sampled_primes;
use crate::gadget_product::GadgetProductLhsOperand;
use crate::gadget_product::GadgetProductRhsOperand;
use crate::lintransform::composite::powcoeffs_to_slots_fat;
use crate::lintransform::matmul::CompiledLinearTransform;
use crate::lintransform::HELinearTransform;
use crate::ntt::{HERingNegacyclicNTT, HERingConvolution};
use crate::ciphertext_ring::double_rns_managed::*;
use crate::number_ring::hypercube::{HypercubeStructure, HypercubeIsomorphism};
use crate::number_ring::largest_prime_leq_congruent_to_one;
use crate::number_ring::quotient::*;
use crate::number_ring::{HECyclotomicNumberRing, HENumberRing};
use crate::number_ring::pow2_cyclotomic::*;
use crate::number_ring::odd_cyclotomic::*;
use crate::profiling::*;
use crate::ciphertext_ring::single_rns_ring::SingleRNSRingBase;
use crate::rnsconv::bfv_rescale::{AlmostExactRescaling, AlmostExactRescalingConvert};
use crate::rnsconv::shared_lift::AlmostExactSharedBaseConversion;
use crate::DefaultCiphertextAllocator;
use crate::sample_primes;
use crate::DefaultConvolution;
use crate::DefaultNegacyclicNTT;

use rand::thread_rng;
use rand::{Rng, CryptoRng};
use rand_distr::StandardNormal;

pub mod bootstrap;

pub type NumberRing<Params: BFVParams> = <Params::CiphertextRing as BGFVCiphertextRing>::NumberRing;
pub type PlaintextRing<Params: BFVParams> = NumberRingQuotient<NumberRing<Params>, Zn, Global>;
pub type SecretKey<Params: BFVParams> = El<CiphertextRing<Params>>;
pub type KeySwitchKey<'a, Params: BFVParams> = (GadgetProductOperand<'a, Params>, GadgetProductOperand<'a, Params>);
pub type RelinKey<'a, Params: BFVParams> = KeySwitchKey<'a, Params>;
pub type CiphertextRing<Params: BFVParams> = RingValue<Params::CiphertextRing>;
pub type Ciphertext<Params: BFVParams> = (El<CiphertextRing<Params>>, El<CiphertextRing<Params>>);
pub type GadgetProductOperand<'a, Params: BFVParams> = GadgetProductRhsOperand<Params::CiphertextRing>;

const ZZbig: BigIntRing = BigIntRing::RING;
const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;

pub struct MulConversionData {
    pub lift_to_C_mul: AlmostExactSharedBaseConversion<Global>,
    pub scale_down_to_C: AlmostExactRescalingConvert<Global>
}

///
/// Trait for types that represent an instantiation of BFV.
/// 
/// For example, the implementation [`Pow2BFV`] stores
///  - the binary logarithm `log(N)` of the ring degree `N`
///  - the length of the ciphertext modulus `q`
///  - the type of ciphertext ring to be used - in this case [`ManagedDoubleRNSRing`] with 
///    [`Pow2CyclotomicNumberRing`]
/// 
/// In particular, we consider the parameters for the ciphertext ring to be part
/// of the instantiation of BFV, but not the plaintext modulus.
/// The reason is that some applications (most notably bootstrapping)
/// consider the "same" BFV instantiation with different plaintext
/// moduli - since a BFV encryption of an element of `R/tR` is always
/// also a valid BFV encryption of the derived element in `R/t'R`, for
/// every `t'` with `t | t'`. 
/// 
/// Most functionality of BFV is provided using default functions,
/// but can be overloaded in case a specific instantiation allows for
/// a more efficient implementation.
/// 
/// ## Combining different ring implementations
/// 
/// I'm not yet completely sure what is the best way to handle this.
/// Currently, each implementor of [`BFVParams`] fixes the type of the
/// ciphertext ring to be used. However, since the different possible
/// ciphertext ring implementations have different performance characteristics,
/// it may be sensible to switch the implementation during working with the
/// scheme. Currently, this requires either creating two different `BFVParams`-
/// instantiations (recommended), or manually creating the ciphertext ring.
/// Note that mapping the ring elements between the implementations can be
/// done using [`feanor_math::homomorphism::CanIso`].
///  
pub trait BFVParams {
    
    ///
    /// Implementation of the ciphertext ring to use.
    /// 
    type CiphertextRing: BGFVCiphertextRing + CyclotomicRing + FiniteRing;

    ///
    /// Returns the allowed lengths of the ciphertext modulus.
    /// 
    fn ciphertext_modulus_bits(&self) -> Range<usize>;

    ///
    /// The number ring `R` we work in, i.e. the ciphertext ring is `R/qR` and
    /// the plaintext ring is `R/tR`.
    /// 
    fn number_ring(&self) -> NumberRing<Self>;

    ///
    /// Creates the ciphertext ring `R/qR` and the extended-modulus ciphertext ring
    /// `R/qq'R` that is necessary for homomorphic multiplication.
    /// 
    fn create_ciphertext_rings(&self) -> (CiphertextRing<Self>, CiphertextRing<Self>);

    ///
    /// Creates the plaintext ring `R/tR` for the given plaintext modulus `t`.
    /// 
    fn create_plaintext_ring(&self, modulus: i64) -> PlaintextRing<Self> {
        NumberRingQuotientBase::new(self.number_ring(), Zn::new(modulus as u64))
    }

    ///
    /// Generates a secret key, using the randomness of the given rng.
    /// 
    fn gen_sk<R: Rng + CryptoRng>(C: &CiphertextRing<Self>, mut rng: R) -> SecretKey<Self> {
        // we sample uniform ternary secrets 
        let result = C.from_canonical_basis((0..C.rank()).map(|_| C.base_ring().int_hom().map((rng.next_u32() % 3) as i32 - 1)));
        return result;
    }
    
    ///
    /// Generates a new encryption of zero using the secret key and the randomness of the given rng.
    /// 
    fn enc_sym_zero<R: Rng + CryptoRng>(C: &CiphertextRing<Self>, mut rng: R, sk: &SecretKey<Self>) -> Ciphertext<Self> {
        let a = C.random_element(|| rng.next_u64());
        let mut b = C.negate(C.mul_ref(&a, &sk));
        let e = C.from_canonical_basis((0..C.rank()).map(|_| C.base_ring().int_hom().map((rng.sample::<f64, _>(StandardNormal) * 3.2).round() as i32)));
        C.add_assign(&mut b, e);
        return (b, a);
    }
    
    ///
    /// Creates a transparent encryption of zero, i.e. a noiseless encryption that does not hide
    /// the encrypted value - everyone can read it, even without access to the secret key.
    /// 
    /// Often used to initialize an accumulator (or similar) during algorithms. 
    /// 
    fn transparent_zero(C: &CiphertextRing<Self>) -> Ciphertext<Self> {
        (C.zero(), C.zero())
    }

    ///
    /// Encrypts the given value, using the randomness of the given rng.
    /// 
    fn enc_sym<R: Rng + CryptoRng>(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, rng: R, m: &El<PlaintextRing<Self>>, sk: &SecretKey<Self>) -> Ciphertext<Self> {
        Self::hom_add_plain(P, C, m, Self::enc_sym_zero(C, rng, sk))
    }

    ///
    /// Creates an encryption of the secret key - this is always easily possible in BFV.
    /// 
    fn enc_sk(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>) -> Ciphertext<Self> {
        let Delta = ZZbig.rounded_div(
            ZZbig.clone_el(C.base_ring().modulus()), 
            &int_cast(*P.base_ring().modulus() as i32, &ZZbig, &StaticRing::<i32>::RING)
        );
        (C.zero(), C.inclusion().map(C.base_ring().coerce(&ZZbig, Delta)))
    }
    
    ///
    /// Given `q/t m + e`, removes the noise term `e`, thus returns `q/t m`.
    /// Used during [`BFVParams::dec()`] and [`BFVParams::noise_budget()`].
    /// 
    fn remove_noise(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, c: &El<CiphertextRing<Self>>) -> El<PlaintextRing<Self>> {
        let coefficients = C.wrt_canonical_basis(c);
        let Delta = ZZbig.rounded_div(
            ZZbig.clone_el(C.base_ring().modulus()), 
            &int_cast(*P.base_ring().modulus() as i32, &ZZbig, &StaticRing::<i32>::RING)
        );
        let modulo = P.base_ring().can_hom(&ZZbig).unwrap();
        return P.from_canonical_basis((0..coefficients.len()).map(|i| modulo.map(ZZbig.rounded_div(C.base_ring().smallest_lift(coefficients.at(i)), &Delta))));
    }
    
    ///
    /// Decrypts a given ciphertext.
    /// 
    /// This function does perform any semantic checks. In particular, it is up to the
    /// caller to ensure that the ciphertext is defined over the given ring, and is a valid
    /// BFV encryption w.r.t. the given plaintext modulus.
    /// 
    fn dec(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: Ciphertext<Self>, sk: &SecretKey<Self>) -> El<PlaintextRing<Self>> {
        let (c0, c1) = ct;
        let noisy_m = C.add(c0, C.mul_ref_snd(c1, sk));
        return Self::remove_noise(P, C, &noisy_m);
    }
    
    ///
    /// Decrypts a given ciphertext and prints its value to stdout.
    /// Designed for debugging purposes.
    /// 
    fn dec_println(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: &Ciphertext<Self>, sk: &SecretKey<Self>) {
        let m = Self::dec(P, C, Self::clone_ct(C, ct), sk);
        println!("ciphertext (noise budget: {}):", Self::noise_budget(P, C, ct, sk));
        P.println(&m);
        println!();
    }
    
    ///
    /// Decrypts a given ciphertext and prints the values in its slots to stdout.
    /// Designed for debugging purposes.
    /// 
    fn dec_println_slots(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: &Ciphertext<Self>, sk: &SecretKey<Self>) {
        let (p, _e) = is_prime_power(ZZ, P.base_ring().modulus()).unwrap();
        let hypercube = HypercubeStructure::halevi_shoup_hypercube(CyclotomicGaloisGroup::new(P.n() as u64), p);
        let H = HypercubeIsomorphism::new::<false>(P, hypercube);
        let m = Self::dec(P, C, Self::clone_ct(C, ct), sk);
        println!("ciphertext (noise budget: {}):", Self::noise_budget(P, C, ct, sk));
        for a in H.get_slot_values(&m) {
            H.slot_ring().println(&a);
        }
        println!();
    }
    
    ///
    /// Computes an encryption of the sum of two encrypted messages.
    /// 
    /// This function does perform any semantic checks. In particular, it is up to the
    /// caller to ensure that the ciphertexts are defined over the given ring, and are
    /// BFV encryptions w.r.t. compatible plaintext moduli.
    /// 
    fn hom_add(C: &CiphertextRing<Self>, lhs: Ciphertext<Self>, rhs: &Ciphertext<Self>) -> Ciphertext<Self> {
        record_time!(GLOBAL_TIME_RECORDER, "BFVParams::hom_add", || {
            let (lhs0, lhs1) = lhs;
            let (rhs0, rhs1) = rhs;
            return (C.add_ref(&lhs0, &rhs0), C.add_ref(&lhs1, &rhs1));
        })
    }
    
    ///
    /// Computes an encryption of the difference of two encrypted messages.
    /// 
    /// This function does perform any semantic checks. In particular, it is up to the
    /// caller to ensure that the ciphertexts are defined over the given ring, and are
    /// BFV encryptions w.r.t. compatible plaintext moduli.
    /// 
    fn hom_sub(C: &CiphertextRing<Self>, lhs: Ciphertext<Self>, rhs: &Ciphertext<Self>) -> Ciphertext<Self> {
        record_time!(GLOBAL_TIME_RECORDER, "BFVParams::hom_sub", || {
            let (lhs0, lhs1) = lhs;
            let (rhs0, rhs1) = rhs;
            return (C.sub_ref(&lhs0, rhs0), C.sub_ref(&lhs1, rhs1));
        })
    }
    
    ///
    /// Copies a ciphertext.
    /// 
    fn clone_ct(C: &CiphertextRing<Self>, ct: &Ciphertext<Self>) -> Ciphertext<Self> {
        (C.clone_el(&ct.0), C.clone_el(&ct.1))
    }
    
    ///
    /// Computes an encryption of the sum of an encrypted message and a plaintext.
    /// 
    /// This function does perform any semantic checks. In particular, it is up to the
    /// caller to ensure that the ciphertext is defined over the given ring, and is
    /// BFV encryption w.r.t. the given plaintext modulus.
    /// 
    fn hom_add_plain(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: &El<PlaintextRing<Self>>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        record_time!(GLOBAL_TIME_RECORDER, "BFVParams::hom_add_plain", || {
            let ZZ_to_Zq = C.base_ring().can_hom(P.base_ring().integer_ring()).unwrap();
            let mut m = C.from_canonical_basis(P.wrt_canonical_basis(m).iter().map(|c| ZZ_to_Zq.map(P.base_ring().smallest_lift(c))));
            let Delta = C.base_ring().coerce(&ZZbig, ZZbig.rounded_div(
                ZZbig.clone_el(C.base_ring().modulus()), 
                &int_cast(*P.base_ring().modulus() as i32, &ZZbig, &StaticRing::<i32>::RING)
            ));
            C.inclusion().mul_assign_ref_map(&mut m, &Delta);
            return (C.add(ct.0, m), ct.1);
        })
    }
    
    ///
    /// Computes an encryption of the product of an encrypted message and a plaintext.
    /// 
    /// This function does perform any semantic checks. In particular, it is up to the
    /// caller to ensure that the ciphertext is defined over the given ring, and is
    /// BFV encryption w.r.t. the given plaintext modulus.
    /// 
    fn hom_mul_plain(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: &El<PlaintextRing<Self>>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        record_time!(GLOBAL_TIME_RECORDER, "BFVParams::hom_mul_plain", || {
            let ZZ_to_Zq = C.base_ring().can_hom(P.base_ring().integer_ring()).unwrap();
            let m = C.from_canonical_basis(P.wrt_canonical_basis(m).iter().map(|c| ZZ_to_Zq.map(P.base_ring().smallest_lift(c))));
            let (c0, c1) = ct;
            return (C.mul_ref_snd(c0, &m), C.mul(c1, m));
        })
    }
    
    ///
    /// Computes an encryption of the product of an encrypted message and an integer plaintext.
    /// 
    /// This function does perform any semantic checks. In particular, it is up to the
    /// caller to ensure that the ciphertext is defined over the given ring, and is
    /// BFV encryption w.r.t. the given plaintext modulus.
    /// 
    fn hom_mul_plain_i64(_P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: i64, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        (C.int_hom().mul_map(ct.0, m as i32), C.int_hom().mul_map(ct.1, m as i32))
    }
    
    ///
    /// Computes the "noise budget" of a given ciphertext.
    /// 
    /// Concretely, the noise budget is `log(q/(t|e|))`, where `t` is the plaintext modulus
    /// and `|e|` is the `l_inf`-norm of the noise term. This will decrease during homomorphic
    /// operations, and if it reaches zero, decryption may yield incorrect results.
    /// 
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
    
    ///
    /// Generates a relinearization key, necessary to compute homomorphic multiplications.
    /// 
    /// The parameter `digits` refers to the number of "digits" to use for the gadget product
    /// during relinearization. More concretely, when performing relinearization, the ciphertext
    /// will be decomposed into multiple small parts, which are then multiplied with the components
    /// of the relinearization key. Thus, a larger value for `digits` will result in lower (additive)
    /// noise growth during relinearization, at the cost of higher performance.
    /// 
    fn gen_rk<'a, R: Rng + CryptoRng>(C: &'a CiphertextRing<Self>, rng: R, sk: &SecretKey<Self>, digits: usize) -> RelinKey<'a, Self>
        where Self: 'a
    {
        Self::gen_switch_key(C, rng, &C.pow(C.clone_el(sk), 2), sk, digits)
    }
    
    ///
    /// Computes an encryption of the product of two encrypted messages.
    /// 
    /// This function does perform any semantic checks. In particular, it is up to the
    /// caller to ensure that the ciphertexts are defined over the given ring, and are
    /// BFV encryptions w.r.t. the given plaintext modulus.
    /// 
    fn hom_mul<'a>(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, C_mul: &CiphertextRing<Self>, lhs: Ciphertext<Self>, rhs: Ciphertext<Self>, rk: &RelinKey<'a, Self>) -> Ciphertext<Self>
        where Self: 'a
    {
        record_time!(GLOBAL_TIME_RECORDER, "BFVParams::hom_mul", || {
            let conv_data = Self::create_multiplication_rescale(P, C, C_mul);

            let (c00, c01) = lhs;
            let (c10, c11) = rhs;
            let lift = |c| perform_rns_op(C_mul.get_ring(), C.get_ring(), &c, &conv_data.lift_to_C_mul);
        
            let c00_lifted = lift(c00);
            let c01_lifted = lift(c01);
            let c10_lifted = lift(c10);
            let c11_lifted = lift(c11);
        
            let [lifted0, lifted1, lifted2] = C_mul.get_ring().two_by_two_convolution([&c00_lifted, &c01_lifted], [&c10_lifted, &c11_lifted]);
        
            let scale_down = |c: El<CiphertextRing<Self>>| perform_rns_op(C.get_ring(), C_mul.get_ring(), &c, &conv_data.scale_down_to_C);
        
            let res0 = scale_down(lifted0);
            let res1 = scale_down(lifted1);
            let res2 = scale_down(lifted2);
        
            let op = GadgetProductLhsOperand::from_element_with(C.get_ring(), &res2, rk.0.gadget_vector_moduli_indices());
            let (s0, s1) = rk;
        
            return (C.add_ref(&res0, &op.gadget_product(s0, C.get_ring())), C.add_ref(&res1, &op.gadget_product(s1, C.get_ring())));
        })
    }
    
    ///
    /// Generates a key-switch key. 
    /// 
    /// In particular, this is used to generate relinearization keys (via [`BFVParams::gen_rk()`])
    /// or Galois keys (via [`BFVParams::gen_gk()`]).
    /// 
    /// The parameter `digits` refers to the number of "digits" to use for the gadget product
    /// during key-switching. More concretely, when performing key-switching, the ciphertext
    /// will be decomposed into multiple small parts, which are then multiplied with the components
    /// of the key-switching key. Thus, a larger value for `digits` will result in lower (additive)
    /// noise growth during key-switching, at the cost of higher performance.
    /// 
    fn gen_switch_key<'a, R: Rng + CryptoRng>(C: &'a CiphertextRing<Self>, mut rng: R, old_sk: &SecretKey<Self>, new_sk: &SecretKey<Self>, digits: usize) -> KeySwitchKey<'a, Self>
        where Self: 'a
    {
        let mut res0 = GadgetProductRhsOperand::new(C.get_ring(), digits);
        let mut res1 = GadgetProductRhsOperand::new(C.get_ring(), digits);
        for digit_i in 0..digits {
            let (c0, c1) = Self::enc_sym_zero(C, &mut rng, new_sk);
            let digit_range = res0.gadget_vector_moduli_indices().at(digit_i).clone();
            let factor = C.base_ring().get_ring().from_congruence((0..C.base_ring().len()).map(|i2| {
                let Fp = C.base_ring().at(i2);
                if digit_range.contains(&i2) { Fp.one() } else { Fp.zero() } 
            }));
            let mut payload = C.clone_el(&old_sk);
            C.inclusion().mul_assign_ref_map(&mut payload, &factor);
            C.add_assign_ref(&mut payload, &c0);
            res0.set_rns_factor(C.get_ring(), digit_i, payload);
            res1.set_rns_factor(C.get_ring(), digit_i, c1);
        }
        return (res0, res1);
    }
    
    ///
    /// Using a key-switch key, computes an encryption encrypting the same message as the given ciphertext
    /// under a different secret key.
    /// 
    fn key_switch<'a>(C: &CiphertextRing<Self>, ct: Ciphertext<Self>, switch_key: &KeySwitchKey<'a, Self>) -> Ciphertext<Self>
        where Self: 'a
    {
        let (c0, c1) = ct;
        let (s0, s1) = switch_key;
        let op = GadgetProductLhsOperand::from_element_with(C.get_ring(), &c1, switch_key.0.gadget_vector_moduli_indices());
        return (
            C.add_ref_snd(c0, &op.gadget_product(s0, C.get_ring())),
            op.gadget_product(s1, C.get_ring())
        );
    }
    
    ///
    /// Modulus-switches from `R/qR` to `R/tR`, where the latter one is given as a plaintext ring.
    /// In particular, this is necessary during bootstrapping.
    /// 
    fn mod_switch_to_plaintext(target: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: Ciphertext<Self>) -> (El<PlaintextRing<Self>>, El<PlaintextRing<Self>>) {
        let mod_switch = AlmostExactRescaling::new_with(
            C.base_ring().as_iter().map(|Zp| *Zp).collect(),
            vec![*target.base_ring()],
            (0..C.base_ring().len()).collect(),
            Global
        );
        let (c0, c1) = ct;
        return (
            perform_rns_op_to_plaintext_ring(target, C.get_ring(), &c0, &mod_switch),
            perform_rns_op_to_plaintext_ring(target, C.get_ring(), &c1, &mod_switch)
        );
    }
    
    ///
    /// Generates a Galois key, usable for homomorphically applying Galois automorphisms.
    /// 
    /// The parameter `digits` refers to the number of "digits" to use for the gadget product
    /// during key-switching. More concretely, when performing key-switching, the ciphertext
    /// will be decomposed into multiple small parts, which are then multiplied with the components
    /// of the key-switching key. Thus, a larger value for `digits` will result in lower (additive)
    /// noise growth during key-switching, at the cost of higher performance.
    /// 
    fn gen_gk<'a, R: Rng + CryptoRng>(C: &'a CiphertextRing<Self>, rng: R, sk: &SecretKey<Self>, g: CyclotomicGaloisGroupEl, digits: usize) -> KeySwitchKey<'a, Self>
        where Self: 'a
    {
        Self::gen_switch_key(C, rng, &C.get_ring().apply_galois_action(sk, g), sk, digits)
    }
    
    ///
    /// Computes an encryption of `sigma(x)`, where `x` is the message encrypted by the given ciphertext
    /// and `sigma` is the given Galois automorphism.
    /// 
    fn hom_galois<'a>(C: &CiphertextRing<Self>, ct: Ciphertext<Self>, g: CyclotomicGaloisGroupEl, gk: &KeySwitchKey<'a, Self>) -> Ciphertext<Self>
        where Self: 'a
    {
        record_time!(GLOBAL_TIME_RECORDER, "BFVParams::hom_galois", || {
            Self::key_switch(C, (
                C.get_ring().apply_galois_action(&ct.0, g),
                C.get_ring().apply_galois_action(&ct.1, g)
            ), gk)
        })
    }
    
    ///
    /// Homomorphically applies multiple Galois automorphisms at once.
    /// Functionally, this is equivalent to calling [`BFVParams::hom_galois()`]
    /// multiple times, but can be faster.
    /// 
    fn hom_galois_many<'a, 'b, V>(C: &CiphertextRing<Self>, ct: Ciphertext<Self>, gs: &[CyclotomicGaloisGroupEl], gks: V) -> Vec<Ciphertext<Self>>
        where V: VectorFn<&'b KeySwitchKey<'a, Self>>,
            KeySwitchKey<'a, Self>: 'b,
            'a: 'b,
            Self: 'a
    {
        record_time!(GLOBAL_TIME_RECORDER, "BFVParams::hom_galois_many", || {
            let digits = gks.at(0).0.gadget_vector_moduli_indices();
            let has_same_digits = |gk: &GadgetProductRhsOperand<_>| gk.gadget_vector_moduli_indices().len() == digits.len() && gk.gadget_vector_moduli_indices().iter().zip(digits.iter()).all(|(l, r)| l == r);
            assert!(gks.iter().all(|gk| has_same_digits(&gk.0) && has_same_digits(&gk.1)));
            let (c0, c1) = ct;
            let c1_op = GadgetProductLhsOperand::from_element_with(C.get_ring(), &c1, digits);
            let c1_op_gs = c1_op.apply_galois_action_many(C.get_ring(), gs);
            let c0_gs = C.get_ring().apply_galois_action_many(&c0, gs).into_iter();
            assert_eq!(gks.len(), c1_op_gs.len());
            assert_eq!(gks.len(), c0_gs.len());
            return c0_gs.zip(c1_op_gs.iter()).enumerate().map(|(i, (c0_g, c1_g))| {
                let (s0, s1) = gks.at(i);
                let r0 = c1_g.gadget_product(s0, C.get_ring());
                let r1 = c1_g.gadget_product(s1, C.get_ring());
                let c0_g = C.apply_galois_action(&c0, gs[i]);
                return (C.add_ref(&r0, &c0_g), r1);
            }).collect();
        })
    }

    fn create_multiplication_rescale(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, Cmul: &CiphertextRing<Self>) -> MulConversionData {
        MulConversionData {
            lift_to_C_mul: AlmostExactSharedBaseConversion::new_with(
                C.base_ring().as_iter().map(|R| Zn::new(*R.modulus() as u64)).collect::<Vec<_>>(), 
                Vec::new(),
                Cmul.base_ring().as_iter().skip(C.base_ring().len()).map(|R| Zn::new(*R.modulus() as u64)).collect::<Vec<_>>(),
                Global
            ),
            scale_down_to_C: AlmostExactRescalingConvert::new_with(
                Cmul.base_ring().as_iter().map(|R| Zn::new(*R.modulus() as u64)).collect::<Vec<_>>(), 
                vec![ Zn::new(*P.base_ring().modulus() as u64) ], 
                C.base_ring().len(),
                Global
            )
        }
    }
    
    ///
    /// Computes an encryption of the image of the given encrypted message under the given
    /// linear transform. Assumes all necessary galois keys are given.
    /// 
    fn hom_compute_linear_transform<'a, Transform, const LOG: bool>(
        P: &PlaintextRing<Self>, 
        C: &CiphertextRing<Self>, 
        input: Ciphertext<Self>, 
        transform: &[Transform], 
        gk: &[(CyclotomicGaloisGroupEl, KeySwitchKey<'a, Self>)], 
        key_switches: &mut usize
    ) -> Ciphertext<Self>
        where Self: 'a,
            Transform: HELinearTransform<NumberRing<Self>, Global>
    {
        let Gal = P.galois_group();
        let get_gk = |g: &CyclotomicGaloisGroupEl| &gk.iter().filter(|(s, _)| Gal.eq_el(*g, *s)).next().unwrap().1;
    
        return transform.iter().fold(input, |current, T| T.evaluate_generic(
            current,
            |lhs, rhs| {
                Self::hom_add(C, lhs, rhs)
            }, 
            |value, factor| {
                Self::hom_mul_plain(P, C, factor, value)
            },
            |value, gs| {
                *key_switches += gs.len();
                let result = log_time::<_, _, LOG, _>(format!("Computing {} galois automorphisms", gs.len()).as_str(), |[]| 
                    Self::hom_galois_many(C, value, gs, gs.as_fn().map_fn(|g| get_gk(g)))
                );
                return result;
            },
            |value| Self::clone_ct(C, value)
        ));
    }
}

///
/// Instantiation of BFV in power-of-two cyclotomic rings `Z[X]/(X^N + 1)` for `N`
/// a power of two.
/// 
/// For these rings, using a `DoubleRNSRing` as ciphertext ring is always the best
/// (i.e. fastest) solution.
/// 
#[derive(Debug)]
pub struct Pow2BFV<A: Allocator + Clone + Send + Sync = DefaultCiphertextAllocator, C: Send + Sync + HERingNegacyclicNTT<Zn> = DefaultNegacyclicNTT> {
    pub log2_q_min: usize,
    pub log2_q_max: usize,
    pub log2_N: usize,
    pub ciphertext_allocator: A,
    pub negacyclic_ntt: PhantomData<C>
}

impl<A: Allocator + Clone + Send + Sync, C: Send + Sync + HERingNegacyclicNTT<Zn>> Clone for Pow2BFV<A, C> {

    fn clone(&self) -> Self {
        Self {
            log2_q_min: self.log2_q_min,
            log2_q_max: self.log2_q_max,
            log2_N: self.log2_N,
            ciphertext_allocator: self.ciphertext_allocator.clone(),
            negacyclic_ntt: PhantomData
        }
    }
}

impl<A: Allocator + Clone + Send + Sync, C: Send + Sync + HERingNegacyclicNTT<Zn>> BFVParams for Pow2BFV<A, C> {

    type CiphertextRing = ManagedDoubleRNSRingBase<Pow2CyclotomicNumberRing<C>, A>;

    fn number_ring(&self) -> Pow2CyclotomicNumberRing<C> {
        Pow2CyclotomicNumberRing::new_with(2 << self.log2_N)
    }

    fn ciphertext_modulus_bits(&self) -> Range<usize> {
        self.log2_q_min..self.log2_q_max
    }

    fn create_ciphertext_rings(&self) -> (CiphertextRing<Self>, CiphertextRing<Self>)  {
        let log2_q = self.ciphertext_modulus_bits();
        let number_ring = self.number_ring();
        let required_root_of_unity = number_ring.mod_p_required_root_of_unity() as i64;

        let mut C_rns_base = sample_primes(log2_q.start, log2_q.end, 56, |bound| largest_prime_leq_congruent_to_one(int_cast(bound, ZZ, ZZbig), required_root_of_unity).map(|p| int_cast(p, ZZbig, ZZ))).unwrap();
        C_rns_base.sort_unstable_by(|l, r| ZZbig.cmp(l, r));

        let mut Cmul_rns_base = extend_sampled_primes(&C_rns_base, log2_q.end * 2, log2_q.end * 2 + 57, 57, |bound| largest_prime_leq_congruent_to_one(int_cast(bound, ZZ, ZZbig), required_root_of_unity).map(|p| int_cast(p, ZZbig, ZZ))).unwrap();
        assert!(ZZbig.is_gt(&Cmul_rns_base[Cmul_rns_base.len() - 1], &C_rns_base[C_rns_base.len() - 1]));
        Cmul_rns_base.sort_unstable_by(|l, r| ZZbig.cmp(l, r));

        let C = ManagedDoubleRNSRingBase::new_with(
            self.number_ring(),
            zn_rns::Zn::new(C_rns_base.iter().map(|p| Zn::new(int_cast(ZZbig.clone_el(p), ZZ, ZZbig) as u64)).collect(), ZZbig),
            self.ciphertext_allocator.clone()
        );
        let Cmul = ManagedDoubleRNSRingBase::new_with(
            number_ring,
            zn_rns::Zn::new(Cmul_rns_base.iter().map(|p| Zn::new(int_cast(ZZbig.clone_el(p), ZZ, ZZbig) as u64)).collect(), ZZbig),
            self.ciphertext_allocator.clone()
        );
        return (C, Cmul);
    }
}

///
/// Instantiation of BFV over odd, composite cyclotomic rings `Z[X]/(Phi_n(X))`
/// with `n = n1 n2` and `n2, n2` odd coprime integers. Ciphertexts are represented
/// in double-RNS form. If single-RNS form is instead requires, use [`CompositeSingleRNSBFV`].
/// 
#[derive(Clone, Debug)]
pub struct CompositeBFV<A: Allocator + Clone + Send + Sync = DefaultCiphertextAllocator> {
    pub log2_q_min: usize,
    pub log2_q_max: usize,
    pub n1: usize,
    pub n2: usize,
    pub ciphertext_allocator: A
}

impl<A: Allocator + Clone + Send + Sync> BFVParams for CompositeBFV<A> {

    type CiphertextRing = ManagedDoubleRNSRingBase<CompositeCyclotomicNumberRing, A>;

    fn ciphertext_modulus_bits(&self) -> Range<usize> {
        self.log2_q_min..self.log2_q_max
    }

    fn number_ring(&self) -> CompositeCyclotomicNumberRing {
        CompositeCyclotomicNumberRing::new(self.n1, self.n2)
    }

    fn create_ciphertext_rings(&self) -> (CiphertextRing<Self>, CiphertextRing<Self>)  {
        let log2_q = self.ciphertext_modulus_bits();
        let number_ring = self.number_ring();
        let required_root_of_unity = number_ring.mod_p_required_root_of_unity() as i64;

        let mut C_rns_base = sample_primes(log2_q.start, log2_q.end, 56, |bound| largest_prime_leq_congruent_to_one(int_cast(bound, ZZ, ZZbig), required_root_of_unity).map(|p| int_cast(p, ZZbig, ZZ))).unwrap();
        C_rns_base.sort_unstable_by(|l, r| ZZbig.cmp(l, r));

        let mut Cmul_rns_base = extend_sampled_primes(&C_rns_base, log2_q.end * 2, log2_q.end * 2 + 57, 57, |bound| largest_prime_leq_congruent_to_one(int_cast(bound, ZZ, ZZbig), required_root_of_unity).map(|p| int_cast(p, ZZbig, ZZ))).unwrap();
        assert!(ZZbig.is_gt(&Cmul_rns_base[Cmul_rns_base.len() - 1], &C_rns_base[C_rns_base.len() - 1]));
        Cmul_rns_base.sort_unstable_by(|l, r| ZZbig.cmp(l, r));

        let C = ManagedDoubleRNSRingBase::new_with(
            self.number_ring(),
            zn_rns::Zn::new(C_rns_base.iter().map(|p| Zn::new(int_cast(ZZbig.clone_el(p), ZZ, ZZbig) as u64)).collect(), ZZbig),
            self.ciphertext_allocator.clone()
        );
        let Cmul = ManagedDoubleRNSRingBase::new_with(
            number_ring,
            zn_rns::Zn::new(Cmul_rns_base.iter().map(|p| Zn::new(int_cast(ZZbig.clone_el(p), ZZ, ZZbig) as u64)).collect(), ZZbig),
            self.ciphertext_allocator.clone()
        );
        return (C, Cmul);
    }
}

///
/// Instantiation of BFV over odd, composite cyclotomic rings `Z[X]/(Phi_n(X))`
/// with `n = n1 n2` and `n2, n2` odd coprime integers. Ciphertexts are represented
/// in single-RNS form. If double-RNS form is instead requires, use [`CompositeBFV`].
/// 
/// This takes a type `C` as last generic argument, which is the type of the convolution
/// algorithm to use to instantiate the ciphertext ring. This has a major impact on 
/// performance.
/// 
#[derive(Clone, Debug)]
pub struct CompositeSingleRNSBFV<A: Allocator + Clone + Send + Sync = DefaultCiphertextAllocator, C: Send + Sync + HERingConvolution<Zn> = DefaultConvolution> {
    pub log2_q_min: usize,
    pub log2_q_max: usize,
    pub n1: usize,
    pub n2: usize,
    pub ciphertext_allocator: A,
    pub convolution: PhantomData<C>
}

impl<A: Allocator + Clone + Send + Sync, C: Send + Sync + HERingConvolution<Zn>> BFVParams for CompositeSingleRNSBFV<A, C> {

    type CiphertextRing = SingleRNSRingBase<CompositeCyclotomicNumberRing, A, C>;

    fn ciphertext_modulus_bits(&self) -> Range<usize> {
        self.log2_q_min..self.log2_q_max
    }

    fn number_ring(&self) -> CompositeCyclotomicNumberRing {
        CompositeCyclotomicNumberRing::new(self.n1, self.n2)
    }

    fn create_ciphertext_rings(&self) -> (CiphertextRing<Self>, CiphertextRing<Self>) {
        let log2_q = self.ciphertext_modulus_bits();
        let number_ring = self.number_ring();
        let required_root_of_unity = signed_lcm(
            number_ring.mod_p_required_root_of_unity() as i64,
            ZZ.abs_log2_ceil(&(number_ring.n() as i64 * 2)).unwrap() as i64,
            ZZ
        );

        let mut C_rns_base = sample_primes(log2_q.start, log2_q.end, 56, |bound| largest_prime_leq_congruent_to_one(int_cast(bound, ZZ, ZZbig), required_root_of_unity).map(|p| int_cast(p, ZZbig, ZZ))).unwrap();
        C_rns_base.sort_unstable_by(|l, r| ZZbig.cmp(l, r));

        let mut Cmul_rns_base = extend_sampled_primes(&C_rns_base, log2_q.end * 2, log2_q.end * 2 + 57, 57, |bound| largest_prime_leq_congruent_to_one(int_cast(bound, ZZ, ZZbig), required_root_of_unity).map(|p| int_cast(p, ZZbig, ZZ))).unwrap();
        assert!(ZZbig.is_gt(&Cmul_rns_base[Cmul_rns_base.len() - 1], &C_rns_base[C_rns_base.len() - 1]));
        Cmul_rns_base.sort_unstable_by(|l, r| ZZbig.cmp(l, r));

        let max_log2_n = 1 + ZZ.abs_log2_ceil(&((self.n1 * self.n2) as i64)).unwrap();
        let C_rns_base = C_rns_base.iter().map(|p| Zn::new(int_cast(ZZbig.clone_el(p), ZZ, ZZbig) as u64)).collect::<Vec<_>>();
        let Cmul_rns_base = Cmul_rns_base.iter().map(|p| Zn::new(int_cast(ZZbig.clone_el(p), ZZ, ZZbig) as u64)).collect::<Vec<_>>();

        let C = SingleRNSRingBase::new_with(
            self.number_ring(),
            zn_rns::Zn::new(C_rns_base.clone(), ZZbig),
            self.ciphertext_allocator.clone(),
            C_rns_base.iter().map(|Zp| C::new(*Zp, max_log2_n)).collect()
        );
        let Cmul = SingleRNSRingBase::new_with(
            number_ring,
            zn_rns::Zn::new(Cmul_rns_base.clone(), ZZbig),
            self.ciphertext_allocator.clone(),
            Cmul_rns_base.iter().map(|Zp| C::new(*Zp, max_log2_n)).collect()
        );
        return (C, Cmul);
    }
}

pub fn hom_evaluate_circuit<'a, 'b, Params: BFVParams>(
    P: &'a PlaintextRing<Params>, 
    C: &'a CiphertextRing<Params>, 
    C_mul: &'a CiphertextRing<Params>, 
    input: &'a Ciphertext<Params>, 
    circuit: &'a ArithCircuit, 
    rk: &'a RelinKey<'b, Params>, 
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
            let result =  Params::hom_mul(P, C, C_mul, lhs, rhs, rk);
            return result;
        }, 
        move |x| {
            Params::hom_add_plain(P, C, &P.inclusion().compose(P.base_ring().can_hom(&ZZ).unwrap()).map(x), Params::transparent_zero(C))
        }
    );
}

pub fn small_basis_repr<Params, NumberRing, A>(C: &CiphertextRing<Params>, ct: Ciphertext<Params>) -> Ciphertext<Params>
    where Params: BFVParams<CiphertextRing = ManagedDoubleRNSRingBase<NumberRing, A>>,
        NumberRing: HECyclotomicNumberRing,
        A: Allocator + Clone
{
    C.get_ring().force_small_basis_repr(&ct.0);
    C.get_ring().force_small_basis_repr(&ct.1);
    return ct;
}

#[test]
fn test_pow2_bfv_hom_galois() {
    let mut rng = thread_rng();
    
    let params = Pow2BFV {
        log2_q_min: 500,
        log2_q_max: 520,
        log2_N: 7,
        ciphertext_allocator: DefaultCiphertextAllocator::default(),
        negacyclic_ntt: PhantomData::<DefaultNegacyclicNTT>
    };
    let t = 3;
    let digits = 3;
    
    let P = params.create_plaintext_ring(t);
    let (C, _C_mul) = params.create_ciphertext_rings();    
    let sk = Pow2BFV::gen_sk(&C, &mut rng);
    let gk = Pow2BFV::gen_gk(&C, &mut rng, &sk, P.galois_group().from_representative(3), digits);
    
    let m = P.canonical_gen();
    let ct = Pow2BFV::enc_sym(&P, &C, &mut rng, &m, &sk);
    let ct_res = Pow2BFV::hom_galois(&C, ct, P.galois_group().from_representative(3), &gk);
    let res = Pow2BFV::dec(&P, &C, ct_res, &sk);

    assert_el_eq!(&P, &P.pow(P.canonical_gen(), 3), &res);
}

#[test]
fn test_pow2_bfv_mul() {
    let mut rng = thread_rng();
    
    let params = Pow2BFV {
        log2_q_min: 500,
        log2_q_max: 520,
        log2_N: 10,
        ciphertext_allocator: DefaultCiphertextAllocator::default(),
        negacyclic_ntt: PhantomData::<DefaultNegacyclicNTT>
    };
    let t = 257;
    let digits = 3;
    
    let P = params.create_plaintext_ring(t);
    let (C, C_mul) = params.create_ciphertext_rings();

    let sk = Pow2BFV::gen_sk(&C, &mut rng);
    let rk = Pow2BFV::gen_rk(&C, &mut rng, &sk, digits);

    let m = P.int_hom().map(2);
    let ct = Pow2BFV::enc_sym(&P, &C, &mut rng, &m, &sk);

    let m = Pow2BFV::dec(&P, &C, Pow2BFV::clone_ct(&C, &ct), &sk);
    assert_el_eq!(&P, &P.int_hom().map(2), &m);

    let ct_sqr = Pow2BFV::hom_mul(&P, &C, &C_mul, Pow2BFV::clone_ct(&C, &ct), Pow2BFV::clone_ct(&C, &ct), &rk);
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
        n2: 97,
        ciphertext_allocator: DefaultCiphertextAllocator::default()
    };
    let t = 8;
    let digits = 3;
    
    let P = params.create_plaintext_ring(t);
    let (C, C_mul) = params.create_ciphertext_rings();

    let sk = CompositeBFV::gen_sk(&C, &mut rng);
    let rk = CompositeBFV::gen_rk(&C, &mut rng, &sk, digits);

    let m = P.int_hom().map(2);
    let ct = CompositeBFV::enc_sym(&P, &C, &mut rng, &m, &sk);

    let m = CompositeBFV::dec(&P, &C, CompositeBFV::clone_ct(&C, &ct), &sk);
    assert_el_eq!(&P, &P.int_hom().map(2), &m);
    
    let ct_sqr = CompositeBFV::hom_mul(&P, &C, &C_mul, CompositeBFV::clone_ct(&C, &ct), CompositeBFV::clone_ct(&C, &ct), &rk);
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
        log2_N: 15,
        ciphertext_allocator: DefaultCiphertextAllocator::default(),
        negacyclic_ntt: PhantomData::<DefaultNegacyclicNTT>
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

    let m = P.int_hom().map(2);
    let ct = log_time::<_, _, true, _>("EncSym", |[]|
        small_basis_repr::<Pow2BFV, _, _>(&C, Pow2BFV::enc_sym(&P, &C, &mut rng, &m, &sk))
    );

    let res = log_time::<_, _, true, _>("HomAddPlain", |[]| 
        small_basis_repr::<Pow2BFV, _, _>(&C, Pow2BFV::hom_add_plain(&P, &C, &m, Pow2BFV::clone_ct(&C, &ct)))
    );
    assert_el_eq!(&P, &P.int_hom().map(4), &Pow2BFV::dec(&P, &C, res, &sk));

    let res = log_time::<_, _, true, _>("HomAdd", |[]| 
        small_basis_repr::<Pow2BFV, _, _>(&C, Pow2BFV::hom_add(&C, Pow2BFV::clone_ct(&C, &ct), &ct))
    );
    assert_el_eq!(&P, &P.int_hom().map(4), &Pow2BFV::dec(&P, &C, res, &sk));

    let res = log_time::<_, _, true, _>("HomMulPlain", |[]| 
        small_basis_repr::<Pow2BFV, _, _>(&C, Pow2BFV::hom_mul_plain(&P, &C, &m, Pow2BFV::clone_ct(&C, &ct)))
    );
    assert_el_eq!(&P, &P.int_hom().map(4), &Pow2BFV::dec(&P, &C, res, &sk));

    let rk = log_time::<_, _, true, _>("GenRK", |[]| 
        Pow2BFV::gen_rk(&C, &mut rng, &sk, digits)
    );
    clear_all_timings();
    let res = log_time::<_, _, true, _>("HomMul", |[]| 
        small_basis_repr::<Pow2BFV, _, _>(&C, Pow2BFV::hom_mul(&P, &C, &C_mul, Pow2BFV::clone_ct(&C, &ct), Pow2BFV::clone_ct(&C, &ct), &rk))
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
        n2: 337,
        ciphertext_allocator: DefaultCiphertextAllocator::default()
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
    
    let m = P.int_hom().map(3);
    let ct = log_time::<_, _, true, _>("EncSym", |[]|
        small_basis_repr::<CompositeBFV, _, _>(&C, CompositeBFV::enc_sym(&P, &C, &mut rng, &m, &sk))
    );
    assert_el_eq!(&P, &P.int_hom().map(3), &CompositeBFV::dec(&P, &C, CompositeBFV::clone_ct(&C, &ct), &sk));

    let res = log_time::<_, _, true, _>("HomAdd", |[]| 
        small_basis_repr::<CompositeBFV, _, _>(&C, CompositeBFV::hom_add(&C, CompositeBFV::clone_ct(&C, &ct), &ct))
    );
    assert_el_eq!(&P, &P.int_hom().map(2), &CompositeBFV::dec(&P, &C, res, &sk));

    let res = log_time::<_, _, true, _>("HomAddPlain", |[]| 
        small_basis_repr::<CompositeBFV, _, _>(&C, CompositeBFV::hom_add_plain(&P, &C, &m, CompositeBFV::clone_ct(&C, &ct)))
    );
    assert_el_eq!(&P, &P.int_hom().map(2), &CompositeBFV::dec(&P, &C, res, &sk));

    let res = log_time::<_, _, true, _>("HomMulPlain", |[]| 
        small_basis_repr::<CompositeBFV, _, _>(&C, CompositeBFV::hom_mul_plain(&P, &C, &m, CompositeBFV::clone_ct(&C, &ct)))
    );
    assert_el_eq!(&P, &P.int_hom().map(1), &CompositeBFV::dec(&P, &C, res, &sk));

    let rk = log_time::<_, _, true, _>("GenRK", |[]| 
        CompositeBFV::gen_rk(&C, &mut rng, &sk, digits)
    );
    clear_all_timings();
    let res = log_time::<_, _, true, _>("HomMul", |[]| 
        small_basis_repr::<CompositeBFV, _, _>(&C, CompositeBFV::hom_mul(&P, &C, &C_mul, CompositeBFV::clone_ct(&C, &ct), CompositeBFV::clone_ct(&C, &ct), &rk))
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
        n2: 11,
        ciphertext_allocator: DefaultCiphertextAllocator::default(),
        convolution: PhantomData::<DefaultConvolution>
    };
    let t = 3;
    let digits = 3;
    
    let P = params.create_plaintext_ring(t);
    let (C, _C_mul) = params.create_ciphertext_rings();    
    let sk = CompositeSingleRNSBFV::gen_sk(&C, &mut rng);
    let gk = CompositeSingleRNSBFV::gen_gk(&C, &mut rng, &sk, P.galois_group().from_representative(3), digits);
    
    let m = P.canonical_gen();
    let ct = CompositeSingleRNSBFV::enc_sym(&P, &C, &mut rng, &m, &sk);
    let ct_res = CompositeSingleRNSBFV::hom_galois(&C, ct, P.galois_group().from_representative(3), &gk);
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
        n2: 11,
        ciphertext_allocator: DefaultCiphertextAllocator::default(),
        convolution: PhantomData::<DefaultConvolution>
    };
    let t = 3;
    let digits = 3;

    let P = params.create_plaintext_ring(t);
    let (C, C_mul) = params.create_ciphertext_rings();

    let sk = CompositeSingleRNSBFV::gen_sk(&C, &mut rng);
    let rk = CompositeSingleRNSBFV::gen_rk(&C, &mut rng, &sk, digits);

    let m = P.int_hom().map(2);
    let ct = CompositeSingleRNSBFV::enc_sym(&P, &C, &mut rng, &m, &sk);

    let m = CompositeSingleRNSBFV::dec(&P, &C, CompositeSingleRNSBFV::clone_ct(&C, &ct), &sk);
    assert_el_eq!(&P, &P.int_hom().map(2), &m);

    let ct_sqr = CompositeSingleRNSBFV::hom_mul(&P, &C, &C_mul, CompositeSingleRNSBFV::clone_ct(&C, &ct), CompositeSingleRNSBFV::clone_ct(&C, &ct), &rk);
    let m_sqr = CompositeSingleRNSBFV::dec(&P, &C, ct_sqr, &sk);

    assert_el_eq!(&P, &P.int_hom().map(4), &m_sqr);
}

#[test]
#[ignore]
fn print_timings_single_rns_composite_bfv_mul() {
    let mut rng = thread_rng();
    
    let params = CompositeSingleRNSBFV {
        log2_q_min: 1090,
        log2_q_max: 1100,
        n1: 127,
        n2: 337,
        ciphertext_allocator: AllocArc(Arc::new(DynLayoutMempool::<Global>::new(Alignment::of::<u64>()))),
        convolution: PhantomData::<DefaultConvolution>
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
        CompositeSingleRNSBFV::hom_mul(&P, &C, &C_mul, CompositeSingleRNSBFV::clone_ct(&C, &ct), CompositeSingleRNSBFV::clone_ct(&C, &ct), &rk)
    );
    print_all_timings();
    assert_el_eq!(&P, &P.int_hom().map(1), &CompositeSingleRNSBFV::dec(&P, &C, res, &sk));
}

#[test]
#[ignore]
fn test_hom_eval_powcoeffs_to_slots_fat_large() {let mut rng = thread_rng();
    let params = CompositeBFV {
        log2_q_min: 790,
        log2_q_max: 800,
        n1: 127,
        n2: 337,
        ciphertext_allocator: DefaultCiphertextAllocator::default()
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
        CompositeBFV::gen_sk(&C, &mut rng)
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
        CompositeBFV::enc_sym(&P, &C, &mut rng, &m, &sk)
    );

    let gks = log_time::<_, _, true, _>("GenGK", |[]| 
        transform.iter().flat_map(|t| t.required_galois_keys().into_iter()).map(|g| (g, CompositeBFV::gen_gk(&C, &mut rng, &sk, g, digits))).collect::<Vec<_>>()
    );

    clear_all_timings();
    let result = log_time::<_, _, true, _>("ApplyTransform", |[key_switches]| 
        CompositeBFV::hom_compute_linear_transform::<_, true>(&P, &C, ct, &transform, &gks, key_switches)
    );
    print_all_timings();
}