use std::alloc::{Allocator, Global};
use std::fmt::Display;
use std::marker::PhantomData;
use std::sync::Arc;

use feanor_math::algorithms::eea::signed_gcd;
use feanor_math::algorithms::int_factor::is_prime_power;
use feanor_math::algorithms::rational_reconstruction::reduce_2d_modular_relation_basis;
use feanor_math::homomorphism::Homomorphism;
use feanor_math::integer::{int_cast, BigIntRing, IntegerRingStore};
use feanor_math::matrix::OwnedMatrix;
use feanor_math::ordered::OrderedRingStore;
use feanor_math::primitive_int::StaticRing;
use feanor_math::ring::*;
use feanor_math::rings::extension::*;
use feanor_math::rings::finite::{FiniteRing, FiniteRingStore};
use feanor_math::rings::zn::zn_64::Zn;
use feanor_math::rings::zn::zn_rns;
use feanor_math::rings::zn::ZnRingStore;
use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::seq::*;
use tracing::instrument;

use crate::ciphertext_ring::double_rns_managed::ManagedDoubleRNSRingBase;
use crate::ciphertext_ring::{perform_rns_op_to_plaintext_ring, single_rns_ring::*};
use crate::ciphertext_ring::BGFVCiphertextRing;
use crate::{cyclotomic::*, ZZi64};
use crate::gadget_product::digits::RNSFactorIndexList;
use crate::gadget_product::{GadgetProductLhsOperand, GadgetProductRhsOperand};
use crate::ntt::{HERingConvolution, HERingNegacyclicNTT};
use crate::number_ring::hypercube::isomorphism::*;
use crate::number_ring::hypercube::structure::HypercubeStructure;
use crate::number_ring::odd_cyclotomic::CompositeCyclotomicNumberRing;
use crate::number_ring::{sample_primes, largest_prime_leq_congruent_to_one, HECyclotomicNumberRing, HENumberRing};
use crate::number_ring::pow2_cyclotomic::Pow2CyclotomicNumberRing;
use crate::number_ring::quotient::{NumberRingQuotient, NumberRingQuotientBase};
use crate::rnsconv::bgv_rescale::{CongruencePreservingAlmostExactBaseConversion, CongruencePreservingRescaling};
use crate::rnsconv::RNSOperation;
use crate::{DefaultCiphertextAllocator, DefaultConvolution, DefaultNegacyclicNTT};

use rand_distr::StandardNormal;
use rand::*;

pub type NumberRing<Params: BGVCiphertextParams> = <Params::CiphertextRing as BGFVCiphertextRing>::NumberRing;
pub type CiphertextRing<Params: BGVCiphertextParams> = RingValue<Params::CiphertextRing>;
pub type PlaintextRing<Params: BGVCiphertextParams> = NumberRingQuotient<NumberRing<Params>, Zn>;
pub type SecretKey<Params: BGVCiphertextParams> = El<CiphertextRing<Params>>;
pub type KeySwitchKey<'a, Params: BGVCiphertextParams> = (GadgetProductRhsOperand<Params::CiphertextRing>, GadgetProductRhsOperand<Params::CiphertextRing>);
pub type RelinKey<'a, Params: BGVCiphertextParams> = KeySwitchKey<'a, Params>;

///
/// Contains the trait [`noise_estimator::BGVNoiseEstimator`] for objects that provide
/// estimates of the noise level of ciphertexts after BGV homomorphic operations.
/// Currently, the only provided implementation is the somewhat imprecise and not rigorously
/// justified [`noise_estimator::NaiveBGVNoiseEstimator`], which is based on simple asymptotic
/// formulas.
/// 
pub mod noise_estimator;
///
/// Contains the trait [`modswitch::BGVModswitchStrategy`] and the implementation
/// [`modswitch::DefaultModswitchStrategy`] for automatic modulus management in BGV.
/// 
pub mod modswitch;
///
/// Contains the implementation of BGV thin bootstrapping.
/// 
pub mod bootstrap;

const ZZbig: BigIntRing = BigIntRing::RING;
const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;

///
/// A BGV ciphertext w.r.t. some [`BGVCiphertextParams`]. Note that this implementation
/// does not include an automatic management of the ciphertext modulus chain,
/// it is up to the user to keep track of the RNS base used for each ciphertext.
/// 
pub struct Ciphertext<Params: ?Sized + BGVCiphertextParams> {
    /// the ciphertext represents the value `implicit_scale^-1 lift(c0 + c1 s) mod t`, 
    /// i.e. `implicit_scale` stores the factor in `Z/tZ` that is introduced by modulus-switching;
    /// Hence, `implicit_scale` is set to `1` when encrypting a value, and only changes when
    /// doing modulus-switching.
    pub implicit_scale: El<Zn>,
    pub c0: El<CiphertextRing<Params>>,
    pub c1: El<CiphertextRing<Params>>
}

///
/// Computes small `a, b` such that `a/b = implicit_scale_bound` modulo `t`.
/// 
pub fn equalize_implicit_scale(Zt: &Zn, implicit_scale_quotient: El<Zn>) -> (i64, i64) {
    let (u, v) = reduce_2d_modular_relation_basis(Zt, implicit_scale_quotient);
    let ZZ_to_Zt = Zt.can_hom(&StaticRing::<i64>::RING).unwrap();
    if Zt.is_unit(&ZZ_to_Zt.map(u[0])) {
        return (u[1], u[0]);
    } else {
        assert!(Zt.is_unit(&ZZ_to_Zt.map(v[0])), "handling this situation in the case of plaintext moduli with multiple different prime factors is not implemented");
        return (v[1], v[0]);
    }
}

///
/// Trait for types that represent an instantiation of BGV.
/// 
/// The design is very similar to [`super::bfv::BFVCiphertextParams`], for details
/// have a look at that. In particular, the plaintext modulus is not a part
/// of the [`super::bfv::BFVCiphertextParams`], but the (initial) ciphertext modulus size
/// is. Note however that BGV requires many ciphertext rings, with progressively
/// smaller ciphertext moduli. You can either manage these manually, or have a look
/// on [`super::modswitch::BGVModswitchStrategy`], which is built on top of this
/// trait and (partially at least) manages ciphertext moduli automatically.
/// In particular, this is different to how other libraries handle BGV ciphertexts, for
/// example HElib by default manages the moduli of all BGV ciphertexts.
/// 
/// For a few more details on how this works, see [`crate::examples::bgv_basics`].
/// 
pub trait BGVCiphertextParams {
    
    ///
    /// Type of the ciphertext ring that is used when creating a ciphertext
    /// ring with these parameters.
    /// 
    type CiphertextRing: BGFVCiphertextRing + CyclotomicRing + FiniteRing;

    ///
    /// Creates the maximal/initial RNS base, which we use to construct
    /// the ciphertext ring that is used for relinearization and Galois keys,
    /// and fresh encryptions.
    /// 
    /// For more details on the modulus chain, see [`crate::examples::bgv_basics`].
    /// 
    fn max_rns_base(&self) -> zn_rns::Zn<Zn, BigIntRing>;

    ///
    /// Creates the ciphertext ring corresponding to the given RNS base.
    /// 
    /// In many cases, you might already have access to a ciphertext ring
    /// with larger RNS base, in these cases it is more efficient to use
    /// [`BGVCiphertextParams::mod_switch_down_ciphertext_ring()`].
    /// 
    fn create_ciphertext_ring(&self, rns_base: zn_rns::Zn<Zn, BigIntRing>) -> CiphertextRing<Self>;

    ///
    /// Creates the maximal/initial ciphertext ring, which is used for relinearization
    /// and Galois keys, and fresh encryptions. This should use the maximal RNS base.
    /// 
    /// For more details on the modulus chain, see [`crate::examples::bgv_basics`].
    /// 
    fn create_initial_ciphertext_ring(&self) -> CiphertextRing<Self> {
        self.create_ciphertext_ring(self.max_rns_base())
    }

    ///
    /// The number ring `R` from which we derive the ciphertext rings `R/qR` and the
    /// plaintext ring `R/tR`.
    /// 
    fn number_ring(&self) -> NumberRing<Self>;

    ///
    /// Creates a plaintext ring `R/tR` for the given plaintext modulus `t`.
    /// 
    #[instrument(skip_all)]
    fn create_plaintext_ring(&self, modulus: i64) -> PlaintextRing<Self> {
        NumberRingQuotientBase::new(self.number_ring(), Zn::new(modulus as u64))
    }

    ///
    /// Generates a secret key, which is either a sparse ternary element of the
    /// ciphertext ring (with hamming weight `hwt`), or a uniform ternary element
    /// of the ciphertext ring (if `hwt == None`).
    /// 
    #[instrument(skip_all)]
    fn gen_sk<R: Rng + CryptoRng>(C: &CiphertextRing<Self>, mut rng: R, hwt: Option<usize>) -> SecretKey<Self> {
        assert!(hwt.is_none() || hwt.unwrap() * 3 <= C.rank() * 2, "it does not make sense to take more than 2/3 of secret key entries in {{-1, 1}}");
        if let Some(hwt) = hwt {
            let mut result_data = (0..C.rank()).map(|_| 0).collect::<Vec<_>>();
            for _ in 0..hwt {
                let mut i = rng.next_u32() as usize % C.rank();
                while result_data[i] != 0 {
                    i = rng.next_u32() as usize % C.rank();
                }
                result_data[i] = (rng.next_u32() % 2) as i32 * 2 - 1;
            }
            let result = C.from_canonical_basis(result_data.into_iter().map(|c| C.base_ring().int_hom().map(c)));
            return result;
        } else {
            let result = C.from_canonical_basis((0..C.rank()).map(|_| C.base_ring().int_hom().map((rng.next_u32() % 3) as i32 - 1)));
            return result;
        }
    }

    ///
    /// Creates an RLWE sample `(a, -as + e)`, where `s = sk` is the secret key and `a, e`
    /// are sampled using randomness from `rng`. Currently, the standard deviation of the
    /// error is fixed to `3.2`.
    /// 
    #[instrument(skip_all)]
    fn rlwe_sample<R: Rng + CryptoRng>(C: &CiphertextRing<Self>, mut rng: R, sk: &SecretKey<Self>) -> (El<CiphertextRing<Self>>, El<CiphertextRing<Self>>) {
        let a = C.random_element(|| rng.next_u64());
        let mut b = C.negate(C.mul_ref(&a, &sk));
        let e = C.from_canonical_basis((0..C.rank()).map(|_| C.base_ring().int_hom().map((rng.sample::<f64, _>(StandardNormal) * 3.2).round() as i32)));
        C.add_assign(&mut b, e);
        return (a, b);
    }

    ///
    /// Creates a fresh encryption of zero, i.e. a ciphertext `(c0, c1) = (-as + te, a)`
    /// where `s = sk` is the given secret key. `a` and `e` are sampled using the randomness
    /// of `rng`. Currently, the standard deviation of the error is fixed to `3.2`.
    /// 
    #[instrument(skip_all)]
    fn enc_sym_zero<R: Rng + CryptoRng>(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, rng: R, sk: &SecretKey<Self>) -> Ciphertext<Self> {
        let t = C.base_ring().coerce(&ZZ, *P.base_ring().modulus());
        let (a, b) = Self::rlwe_sample(C, rng, sk);
        return Ciphertext {
            c0: C.inclusion().mul_ref_snd_map(b, &t),
            c1: C.inclusion().mul_ref_snd_map(a, &t),
            implicit_scale: P.base_ring().one()
        };
    }

    ///
    /// Creates a "transparent" encryption of zero, i.e. a ciphertext that represents zero,
    /// but does not actually hide the value - everyone can see that it is zero, without the
    /// secret key.
    /// 
    /// Mathematically, this is just the ciphertext `(c0, c1) = (0, 0)`.
    /// 
    #[instrument(skip_all)]
    fn transparent_zero(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>) -> Ciphertext<Self> {
        return Ciphertext {
            c0: C.zero(),
            c1: C.zero(),
            implicit_scale: P.base_ring().one()
        };
    }

    ///
    /// Decrypts the given ciphertext and prints it to stdout. Designed for debugging.
    /// 
    #[instrument(skip_all)]
    fn dec_println(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: &Ciphertext<Self>, sk: &SecretKey<Self>) {
        let m = Self::dec(P, C, Self::clone_ct(P, C, ct), sk);
        println!("ciphertext (noise budget: {} / {}):", Self::noise_budget(P, C, ct, sk), ZZbig.abs_log2_ceil(C.base_ring().modulus()).unwrap());
        P.println(&m);
        println!();
    }
    
    ///
    /// Decrypts the given ciphertext and prints the values of its slots to stdout. 
    /// Designed for debugging.
    /// 
    #[instrument(skip_all)]
    fn dec_println_slots(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: &Ciphertext<Self>, sk: &SecretKey<Self>, cache_dir: Option<&str>) {
        let (p, _e) = is_prime_power(ZZ, P.base_ring().modulus()).unwrap();
        let hypercube = HypercubeStructure::halevi_shoup_hypercube(CyclotomicGaloisGroup::new(P.n() as u64), p);
        let H = if let Some(dir) = cache_dir {
            HypercubeIsomorphism::new_cache_file::<false>(P, hypercube, dir)
        } else {
            HypercubeIsomorphism::new::<false>(P, hypercube)
        };
        let m = Self::dec(P, C, Self::clone_ct(P, C, ct), sk);
        println!("ciphertext (noise budget: {} / {}):", Self::noise_budget(P, C, ct, sk), ZZbig.abs_log2_ceil(C.base_ring().modulus()).unwrap());
        for a in H.get_slot_values(&m) {
            H.slot_ring().println(&a);
        }
        println!();
    }

    ///
    /// Returns an encryption of the sum of the encrypted input and the given plaintext,
    /// which has already been lifted/encoded into the ciphertext ring.
    /// 
    /// When the plaintext is given as an element of `P`, use [`BGVCiphertextParams::hom_add_plain()`]
    /// instead. However, internally, the plaintext will be lifted into the ciphertext ring during
    /// the addition, and if this is performed in advance (via [`BGVCiphertextParams::encode_plain()`]),
    /// addition will be faster.
    /// 
    #[instrument(skip_all)]
    fn hom_add_plain_encoded(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: &El<CiphertextRing<Self>>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        assert!(P.base_ring().is_unit(&ct.implicit_scale));
        let implicit_scale = C.base_ring().coerce(&ZZ, P.base_ring().smallest_lift(ct.implicit_scale));
        let result = Ciphertext {
            c0: C.add(ct.c0, C.inclusion().mul_ref_map(m, &implicit_scale)),
            c1: ct.c1,
            implicit_scale: ct.implicit_scale
        };
        assert!(P.base_ring().is_unit(&result.implicit_scale));
        return result;
    }

    ///
    /// Returns an encryption of the sum of the encrypted input and the given plaintext.
    /// 
    #[instrument(skip_all)]
    fn hom_add_plain(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: &El<PlaintextRing<Self>>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        Self::hom_add_plain_encoded(P, C, &Self::encode_plain(P, C, m), ct)
    }

    ///
    /// Returns a fresh encryption of the given element, i.e. a ciphertext `(c0, c1) = (-as + te + m, a)`
    /// where `s = sk` is the given secret key. `a` and `e` are sampled using the randomness of `rng`. 
    /// Currently, the standard deviation of the error is fixed to `3.2`.
    /// 
    #[instrument(skip_all)]
    fn enc_sym<R: Rng + CryptoRng>(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, rng: R, m: &El<PlaintextRing<Self>>, sk: &SecretKey<Self>) -> Ciphertext<Self> {
        Self::hom_add_plain(P, C, m, Self::enc_sym_zero(P, C, rng, sk))
    }

    ///
    /// Decrypts the given ciphertext using the given secret key.
    /// 
    #[instrument(skip_all)]
    fn dec(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: Ciphertext<Self>, sk: &SecretKey<Self>) -> El<PlaintextRing<Self>> {
        let noisy_m = C.add(ct.c0, C.mul_ref_snd(ct.c1, sk));
        let mod_t = P.base_ring().can_hom(&ZZbig).unwrap();
        return P.inclusion().mul_map(
            P.from_canonical_basis(C.wrt_canonical_basis(&noisy_m).iter().map(|x| mod_t.map(C.base_ring().smallest_lift(x)))),
            P.base_ring().invert(&ct.implicit_scale).unwrap()
        );
    }

    ///
    /// Returns an encryption of the product of the encrypted input and the given plaintext,
    /// which has already been lifted/encoded into the ciphertext ring.
    /// 
    /// When the plaintext is given as an element of `P`, use [`BGVCiphertextParams::hom_mul_plain()`]
    /// instead. However, internally, the plaintext will be lifted into the ciphertext ring during
    /// the multiplication, and if this is performed in advance (via [`BGVCiphertextParams::encode_plain()`]),
    /// multiplication will be faster.
    /// 
    #[instrument(skip_all)]
    fn hom_mul_plain_encoded(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: &El<CiphertextRing<Self>>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        assert!(P.base_ring().is_unit(&ct.implicit_scale));
        let result = Ciphertext {
            c0: C.mul_ref_snd(ct.c0, m), 
            c1: C.mul_ref_snd(ct.c1, m),
            implicit_scale: ct.implicit_scale
        };
        assert!(P.base_ring().is_unit(&result.implicit_scale));
        return result;
    }

    ///
    /// Computes the smallest lift of the plaintext ring element to the ciphertext
    /// ring. The result can be used in [`BGVCiphertextParams::hom_add_plain_encoded()`]
    /// or [`BGVCiphertextParams::hom_mul_plain_encoded()`] to compute plaintext-ciphertext
    /// addition resp. multiplication faster.
    /// 
    #[instrument(skip_all)]
    fn encode_plain(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: &El<PlaintextRing<Self>>) -> El<CiphertextRing<Self>> {
        let ZZ_to_Zq = C.base_ring().can_hom(P.base_ring().integer_ring()).unwrap();
        return C.from_canonical_basis(P.wrt_canonical_basis(m).iter().map(|c| ZZ_to_Zq.map(P.base_ring().smallest_lift(c))));
    }

    ///
    /// Returns an encryption of the product of the encrypted input and the given plaintext.
    /// 
    #[instrument(skip_all)]
    fn hom_mul_plain(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: &El<PlaintextRing<Self>>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        Self::hom_mul_plain_encoded(P, C, &Self::encode_plain(P, C, m), ct)
    }

    ///
    /// Returns an encryption of the product of the encrypted input and the given plaintext.
    /// 
    #[instrument(skip_all)]
    fn hom_mul_plain_i64(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: i64, mut ct: Ciphertext<Self>) -> Ciphertext<Self> {
        assert!(P.base_ring().is_unit(&ct.implicit_scale));
        // we could try to do tricks involving `implicit_scale` here
        //  - if `m mod t` is a unit, we could just multiply `m^-1` to implicit scale;
        //    however, this makes handling the non-unit case ugly
        //  - otherwise, we could also use this opportunity to multiply `implicit_scale^-1`
        //    to the ciphertext as well, and reset the implicit scale to 1; however, this
        //    might not be helpful in all circumstances
        // In the end, I think there is no default behavior for this that makes sense
        // in most situations and is not to unintuitive. Hence, we leave any `implicit_scale`
        // tricks to the modswitching strategy, which has higher-level information and might
        // be able to do something with that
        C.int_hom().mul_assign_map(&mut ct.c0, m as i32);
        C.int_hom().mul_assign_map(&mut ct.c1, m as i32);
        assert!(P.base_ring().is_unit(&ct.implicit_scale));
        return ct;
    }

    ///
    /// Converts a ciphertext into a ciphertext with `implicit_scale = 1`, but slightly
    /// larger noise. Mainly used for internal purposes.
    /// 
    fn merge_implicit_scale(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        let mut result = Self::hom_mul_plain_i64(P, C, P.base_ring().smallest_lift(P.base_ring().invert(&ct.implicit_scale).unwrap()), ct);
        result.implicit_scale = P.base_ring().one();
        return result;
    }

    ///
    /// Copies a ciphertext.
    /// 
    #[instrument(skip_all)]
    fn clone_ct(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: &Ciphertext<Self>) -> Ciphertext<Self> {
        assert!(P.base_ring().is_unit(&ct.implicit_scale));
        Ciphertext {
            c0: C.clone_el(&ct.c0),
            c1: C.clone_el(&ct.c1),
            implicit_scale: P.base_ring().clone_el(&ct.implicit_scale)
        }
    }

    ///
    /// Returns the value
    /// ```text
    ///   log2( q / | c0 + c1 s |_inf )
    /// ```
    /// which roughly corresponds to the "noise budget" of the ciphertext, in bits.
    /// 
    #[instrument(skip_all)]
    fn noise_budget(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: &Ciphertext<Self>, sk: &SecretKey<Self>) -> usize {
        let ct = Self::clone_ct(P, C, ct);
        let noisy_m = C.add(ct.c0, C.mul_ref_snd(ct.c1, sk));
        let coefficients = C.wrt_canonical_basis(&noisy_m);
        let size_of_critical_quantity = <_ as Iterator>::max((0..coefficients.len()).map(|i| {
            let c = C.base_ring().smallest_lift(coefficients.at(i));
            let size = ZZbig.abs_log2_ceil(&c);
            return size.unwrap_or(0);
        })).unwrap();
        return ZZbig.abs_log2_ceil(C.base_ring().modulus()).unwrap().saturating_sub(size_of_critical_quantity + 1);
    }

    ///
    /// Generates a key-switch key, which can be used (by [`BGVCiphertextParams::key_switch()`]) to
    /// convert a ciphertext w.r.t. `old_sk` into a ciphertext w.r.t. `new_sk`.
    /// 
    /// The parameter `digits` refers to the number of "digits" to use for the gadget product
    /// during key-switching. More concretely, when performing key-switching, the ciphertext
    /// will be decomposed into multiple small parts, which are then multiplied with the components
    /// of the key-switching key. Thus, a larger value for `digits` will result in lower (additive)
    /// noise growth during key-switching, at the cost of higher performance.
    /// 
    #[instrument(skip_all)]
    fn gen_switch_key<'a, R: Rng + CryptoRng>(P: &PlaintextRing<Self>, C: &'a CiphertextRing<Self>, mut rng: R, old_sk: &SecretKey<Self>, new_sk: &SecretKey<Self>, digits: usize) -> KeySwitchKey<'a, Self>
        where Self: 'a
    {
        let mut res0 = GadgetProductRhsOperand::new(C.get_ring(), digits);
        let mut res1 = GadgetProductRhsOperand::new(C.get_ring(), digits);
        for digit_i in 0..digits {
            let base = Self::enc_sym_zero(P, C, &mut rng, new_sk);
            let digit_range = res0.gadget_vector_digits().at(digit_i).clone();
            let factor = C.base_ring().get_ring().from_congruence((0..C.base_ring().len()).map(|i2| {
                let Fp = C.base_ring().at(i2);
                if digit_range.contains(&i2) { Fp.one() } else { Fp.zero() } 
            }));
            let mut payload = C.clone_el(&old_sk);
            C.inclusion().mul_assign_ref_map(&mut payload, &factor);
            C.add_assign(&mut payload, base.c0);
            res0.set_rns_factor(C.get_ring(), digit_i, payload);
            res1.set_rns_factor(C.get_ring(), digit_i, base.c1);
        }
        return (res0, res1);
    }

    ///
    /// Converts a ciphertext w.r.t. a secret key `old_sk` to a ciphertext w.r.t. a
    /// secret key `new_sk`, where `switch_key` is a key-switching key for `old_sk` and
    /// `new_sk` (which can be generated using [`BGVCiphertextParams::gen_switch_key()`]).
    /// 
    #[instrument(skip_all)]
    fn key_switch<'a>(C: &CiphertextRing<Self>, ct: Ciphertext<Self>, switch_key: &KeySwitchKey<'a, Self>) -> Ciphertext<Self>
        where Self: 'a
    {
        let (s0, s1) = switch_key;
        let op = GadgetProductLhsOperand::from_element_with(C.get_ring(), &ct.c1, switch_key.0.gadget_vector_digits());
        return Ciphertext {
            c0: C.add_ref_snd(ct.c0, &op.gadget_product(s0, C.get_ring())),
            c1: op.gadget_product(s1, C.get_ring()),
            implicit_scale: ct.implicit_scale
        };
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
    #[instrument(skip_all)]
    fn gen_rk<'a, R: Rng + CryptoRng>(P: &PlaintextRing<Self>, C: &'a CiphertextRing<Self>, rng: R, sk: &SecretKey<Self>, digits: usize) -> RelinKey<'a, Self>
        where Self: 'a
    {
        Self::gen_switch_key(P, C, rng, &C.pow(C.clone_el(sk), 2), sk, digits)
    }

    ///
    /// Computes an encryption of the product of two encrypted inputs.
    /// 
    /// Since HE-Ring does not (at least not implicitly) perform automatic modulus management,
    /// it is necessary to modulus-switch between calls to `hom_mul()` in order to prevent
    /// `hom_mul()` from causing exponential noise growth. For more info on modulus-switching
    /// and the modulus chain, see [`crate::examples::bgv_basics`].
    /// 
    #[instrument(skip_all)]
    fn hom_mul<'a>(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, lhs: Ciphertext<Self>, rhs: Ciphertext<Self>, rk: &RelinKey<'a, Self>) -> Ciphertext<Self>
        where Self: 'a
    {
        assert!(P.base_ring().is_unit(&lhs.implicit_scale));
        assert!(P.base_ring().is_unit(&rhs.implicit_scale));

        let [res0, res1, res2] = C.get_ring().two_by_two_convolution([&lhs.c0, &lhs.c1], [&rhs.c0, &rhs.c1]);
        
        let op = GadgetProductLhsOperand::from_element_with(C.get_ring(), &res2, rk.0.gadget_vector_digits());
        let (s0, s1) = &rk;
        
        let result = Ciphertext {
            c0: C.add(res0, op.gadget_product(s0, C.get_ring())), 
            c1: C.add(res1, op.gadget_product(s1, C.get_ring())),
            implicit_scale: P.base_ring().mul(lhs.implicit_scale, rhs.implicit_scale)
        };
        assert!(P.base_ring().is_unit(&result.implicit_scale));
        return result;
    }
    
    ///
    /// Computes an encryption of the square of an encrypted input.
    /// 
    /// Since HE-Ring does not (at least not implicitly) perform automatic modulus management,
    /// it is necessary to modulus-switch between calls to `hom_square()` in order to prevent
    /// `hom_square()` from causing exponential noise growth. For more info on modulus-switching
    /// and the modulus chain, see [`crate::examples::bgv_basics`].
    /// 
    #[instrument(skip_all)]
    fn hom_square<'a>(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, val: Ciphertext<Self>, rk: &RelinKey<'a, Self>) -> Ciphertext<Self>
        where Self: 'a
    {
        assert!(P.base_ring().is_unit(&val.implicit_scale));

        let [res0, res1, res2] = C.get_ring().two_by_two_convolution([&val.c0, &val.c1], [&val.c0, &val.c1]);
        
        let op = GadgetProductLhsOperand::from_element_with(C.get_ring(), &res2, rk.0.gadget_vector_digits());
        let (s0, s1) = &rk;
        
        let result = Ciphertext {
            c0: C.add(res0, op.gadget_product(s0, C.get_ring())), 
            c1: C.add(res1, op.gadget_product(s1, C.get_ring())),
            implicit_scale: P.base_ring().pow(val.implicit_scale, 2)
        };
        assert!(P.base_ring().is_unit(&result.implicit_scale));
        return result;
    }
    
    ///
    /// Computes an encryption of the sum of two encrypted inputs.
    /// 
    #[instrument(skip_all)]
    fn hom_add(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, mut lhs: Ciphertext<Self>, mut rhs: Ciphertext<Self>) -> Ciphertext<Self> {
        assert!(P.base_ring().is_unit(&lhs.implicit_scale));
        assert!(P.base_ring().is_unit(&rhs.implicit_scale));

        let Zt = P.base_ring();
        let (a, b) = equalize_implicit_scale(Zt, Zt.checked_div(&lhs.implicit_scale, &rhs.implicit_scale).unwrap());

        debug_assert!(!Zt.eq_el(&lhs.implicit_scale, &rhs.implicit_scale) || (a == 1 && b == 1));
        if a != 1 {
            C.int_hom().mul_assign_map(&mut rhs.c0, a as i32);
            C.int_hom().mul_assign_map(&mut rhs.c1, a as i32);
            P.base_ring().int_hom().mul_assign_map(&mut rhs.implicit_scale, a as i32);
        }
        if b != 1 {
            C.int_hom().mul_assign_map(&mut lhs.c0, b as i32);
            C.int_hom().mul_assign_map(&mut lhs.c1, b as i32);
            P.base_ring().int_hom().mul_assign_map(&mut lhs.implicit_scale, b as i32);
        }

        assert!(Zt.eq_el(&lhs.implicit_scale, &rhs.implicit_scale));
        let result = Ciphertext {
            c0: C.add(lhs.c0, rhs.c0),
            c1: C.add(lhs.c1, rhs.c1),
            implicit_scale: lhs.implicit_scale
        };
        assert!(P.base_ring().is_unit(&result.implicit_scale));
        return result;
    }

    ///
    /// Computes an encryption of the difference of two encrypted inputs.
    /// 
    fn hom_sub(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, lhs: Ciphertext<Self>, rhs: Ciphertext<Self>) -> Ciphertext<Self> {
        Self::hom_add(P, C, lhs, Ciphertext { c0: rhs.c0, c1: rhs.c1, implicit_scale: P.base_ring().negate(rhs.implicit_scale) })
    }
    
    ///
    /// Computes an encryption of `sigma(x)`, where `x` is the message encrypted by the given ciphertext
    /// and `sigma` is the given Galois automorphism.
    /// 
    #[instrument(skip_all)]
    fn hom_galois<'a>(_P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: Ciphertext<Self>, g: CyclotomicGaloisGroupEl, gk: &KeySwitchKey<'a, Self>) -> Ciphertext<Self>
        where Self: 'a
    {
        Self::key_switch(C, Ciphertext {
            c0: C.get_ring().apply_galois_action(&ct.c0, g),
            c1: C.get_ring().apply_galois_action(&ct.c1, g),
            implicit_scale: ct.implicit_scale
        }, gk)
    }

    ///
    /// Homomorphically applies multiple Galois automorphisms at once.
    /// Functionally, this is equivalent to calling [`BGVCiphertextParams::hom_galois()`]
    /// multiple times, but can be faster.
    /// 
    #[instrument(skip_all)]
    fn hom_galois_many<'a, 'b, V>(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: Ciphertext<Self>, gs: &[CyclotomicGaloisGroupEl], gks: V) -> Vec<Ciphertext<Self>>
        where V: VectorFn<&'b KeySwitchKey<'a, Self>>,
            KeySwitchKey<'a, Self>: 'b,
            'a: 'b,
            Self: 'a
    {
        assert!(P.base_ring().is_unit(&ct.implicit_scale));

        let digits = gks.at(0).0.gadget_vector_digits();
        let has_same_digits = |gk: &GadgetProductRhsOperand<_>| gk.gadget_vector_digits().len() == digits.len() && gk.gadget_vector_digits().iter().zip(digits.iter()).all(|(l, r)| l == r);
        assert!(gks.iter().all(|gk| has_same_digits(&gk.0) && has_same_digits(&gk.1)));
        let c1_op = GadgetProductLhsOperand::from_element_with(C.get_ring(), &ct.c1, digits);
        let c1_op_gs = c1_op.apply_galois_action_many(C.get_ring(), gs);
        let c0_gs = C.get_ring().apply_galois_action_many(&ct.c0, gs).into_iter();
        assert_eq!(gks.len(), c1_op_gs.len());
        assert_eq!(gks.len(), c0_gs.len());
        return c0_gs.zip(c1_op_gs.iter()).enumerate().map(|(i, (c0_g, c1_g))| {
            let (s0, s1) = gks.at(i);
            let r0 = c1_g.gadget_product(s0, C.get_ring());
            let r1 = c1_g.gadget_product(s1, C.get_ring());
            let result = Ciphertext {
                c0: C.add_ref(&r0, &c0_g),
                c1: r1,
                implicit_scale: P.base_ring().clone_el(&ct.implicit_scale)
            };
            assert!(P.base_ring().is_unit(&result.implicit_scale));
            return result;
        }).collect();
    }

    ///
    /// Given `R/qR` this creates the ciphertext ring `R/q'R`, where the RNS base for `q'`
    /// is derived from the RNS base of `q` by removing the RNS factors whose indices are mentioned
    /// in `drop_moduli`.
    /// 
    /// Note that for the implementation in HE-Ring at least, the underlying rings will share
    /// most of their data, which means that this function is actually very cheap, in particular
    /// much cheaper than creating a new ciphertext ring (e.g. using [`BGVCiphertextParams::create_ciphertext_ring()`]).
    /// 
    #[instrument(skip_all)]
    fn mod_switch_down_ciphertext_ring(C: &CiphertextRing<Self>, drop_moduli: &RNSFactorIndexList) -> CiphertextRing<Self> {
        RingValue::from(C.get_ring().drop_rns_factor(&drop_moduli))
    }

    ///
    /// Modulus-switches a secret key in a way compatible with modulus-switching ciphertexts.
    /// 
    /// In more detail, given `R/q'R` and `R/qR` where the RNS base for `q'`is derived from the RNS
    /// base of `q` by removing the RNS factors whose indices are mentioned in `drop_moduli`, this
    /// computes the secret key `sk mod q'`. Note that, if `ct` is an encryption w.r.t. `sk` over `R/qR`
    /// and is modulus-switched to `ct'` over the ring `R/q'R`, then `sk mod q'` can be used to decrypt
    /// `ct'`.
    /// 
    #[instrument(skip_all)]
    fn mod_switch_down_sk(Cnew: &CiphertextRing<Self>, Cold: &CiphertextRing<Self>, drop_moduli: &RNSFactorIndexList, sk: &SecretKey<Self>) -> SecretKey<Self> {
        assert_rns_factor_drop_correct::<Self>(Cnew, Cold, drop_moduli);
        if drop_moduli.len() == 0 {
            Cnew.clone_el(sk)
        } else {
            Cnew.get_ring().drop_rns_factor_element(Cold.get_ring(), &drop_moduli, Cold.clone_el(sk))
        }
    }

    ///
    /// Modulus-switches a relinearization key in a way compatible with modulus-switching ciphertexts.
    /// 
    /// This is equivalent to creating a new relinearization key (using [`BGVCiphertextParams::gen_rk()`])
    /// over `Cnew` for the secret key `mod_switch_down_sk(Cnew, Cold, drop_moduli, sk)`, but does not require
    /// access to `sk`.
    /// 
    #[instrument(skip_all)]
    fn mod_switch_down_rk<'a, 'b>(Cnew: &'b CiphertextRing<Self>, Cold: &CiphertextRing<Self>, drop_moduli: &RNSFactorIndexList, rk: &RelinKey<'a, Self>) -> RelinKey<'b, Self> {
        assert_rns_factor_drop_correct::<Self>(Cnew, Cold, drop_moduli);
        if drop_moduli.len() == 0 {
            (rk.0.clone(Cnew.get_ring()), rk.1.clone(Cnew.get_ring()))
        } else {
            (
                rk.0.clone(Cold.get_ring()).modulus_switch(Cnew.get_ring(), &drop_moduli, Cold.get_ring()), 
                rk.1.clone(Cold.get_ring()).modulus_switch(Cnew.get_ring(), &drop_moduli, Cold.get_ring())
            )
        }
    }

    ///
    /// Modulus-switches a Galois key in a way compatible with modulus-switching ciphertexts.
    /// 
    /// This is equivalent to creating a new Galois key (using [`BGVCiphertextParams::gen_gk()`])
    /// over `Cnew` for the secret key `mod_switch_down_sk(Cnew, Cold, drop_moduli, sk)`, but does not require
    /// access to `sk`.
    /// 
    #[instrument(skip_all)]
    fn mod_switch_down_gk<'a, 'b>(Cnew: &'b CiphertextRing<Self>, Cold: &CiphertextRing<Self>, drop_moduli: &RNSFactorIndexList, gk: &KeySwitchKey<'a, Self>) -> KeySwitchKey<'b, Self> {
        assert_rns_factor_drop_correct::<Self>(Cnew, Cold, drop_moduli);
        if drop_moduli.len() == 0 {
            (gk.0.clone(Cnew.get_ring()), gk.1.clone(Cnew.get_ring()))
        } else {
            (
                gk.0.clone(Cold.get_ring()).modulus_switch(Cnew.get_ring(), &drop_moduli, Cold.get_ring()), 
                gk.1.clone(Cold.get_ring()).modulus_switch(Cnew.get_ring(), &drop_moduli, Cold.get_ring())
            )
        }
    }

    ///
    /// Internal function to compute how the implicit scale of a ciphertext changes
    /// once we modulus-switch it.
    /// 
    fn mod_switch_down_compute_implicit_scale_factor(P: &PlaintextRing<Self>, q_new: &El<BigIntRing>, q_old: &El<BigIntRing>) -> El<Zn> {
        let ZZbig_to_Zt = P.base_ring().can_hom(&ZZbig).unwrap();
        let result = P.base_ring().checked_div(
            &ZZbig_to_Zt.map_ref(q_new),
            &ZZbig_to_Zt.map_ref(q_old)
        ).unwrap();
        assert!(P.base_ring().is_unit(&result));
        return result;
    }

    ///
    /// Modulus-switches a ciphertext.
    /// 
    /// More concretely, we require that `Cold` is the ring `R/qR` and `Cnew` is the ring `R/q'R`,
    /// where the RNS base for `q'`is derived from the RNS base of `q` by removing the RNS factors
    /// whose indices are mentioned in `drop_moduli`. Given a ciphertext `ct` over `R/qR`, this function
    /// then computes a ciphertext encrypting the same message over `R/q'R` (w.r.t. the secret key
    /// `sk mod q'`, which can be accessed via [`BGVCiphertextParams::mod_switch_down_sk()`]).
    /// 
    #[instrument(skip_all)]
    fn mod_switch_down(P: &PlaintextRing<Self>, Cnew: &CiphertextRing<Self>, Cold: &CiphertextRing<Self>, drop_moduli: &RNSFactorIndexList, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        assert_rns_factor_drop_correct::<Self>(Cnew, Cold, drop_moduli);
        assert!(P.base_ring().is_unit(&ct.implicit_scale));

        if drop_moduli.len() == 0 {
            return ct;
        } else {

            let compute_delta = CongruencePreservingAlmostExactBaseConversion::new_with(
                drop_moduli.iter().map(|i| *Cold.base_ring().at(*i)).collect(),
                Cnew.base_ring().as_iter().cloned().collect(),
                *P.base_ring(),
                Global
            );
            let mod_switch_ring_element = |x: El<CiphertextRing<Self>>| {
                // this logic is slightly complicated, since we want to avoid using `perform_rns_op()`;
                // in particular, we only need to convert a part of `x` into coefficient/small-basis representation,
                // while just using `perform_rns_op()` would convert all of `x`.
                let mut mod_b_part_of_x = OwnedMatrix::zero(drop_moduli.len(), Cold.get_ring().small_generating_set_len(), Cold.base_ring().at(0));
                Cold.get_ring().partial_representation_wrt_small_generating_set(&x, &drop_moduli, mod_b_part_of_x.data_mut());
                // this is the "correction", subtracting it will make `x` divisible by the moduli to drop
                let mut delta = OwnedMatrix::zero(Cnew.base_ring().len(), Cnew.get_ring().small_generating_set_len(), Cnew.base_ring().at(0));
                compute_delta.apply(mod_b_part_of_x.data(), delta.data_mut());
                let delta = Cnew.get_ring().from_representation_wrt_small_generating_set(delta.data());
                // now subtract `delta` and scale by the moduli to drop - since `x - delta` is divisible by those,
                // this is actually a rescaling and not only a division in `Z/qZ`
                return Cnew.inclusion().mul_map(
                    Cnew.sub(
                        Cnew.get_ring().drop_rns_factor_element(Cold.get_ring(), &drop_moduli, x),
                        delta
                    ),
                    Cnew.base_ring().invert(&Cnew.base_ring().coerce(&ZZbig, ZZbig.prod(drop_moduli.iter().map(|i| int_cast(*Cold.base_ring().at(*i).modulus(), ZZbig, ZZ))))).unwrap()
                )
            };
            
            let result = Ciphertext {
                c0: mod_switch_ring_element(ct.c0),
                c1: mod_switch_ring_element(ct.c1),
                implicit_scale: P.base_ring().mul(ct.implicit_scale, Self::mod_switch_down_compute_implicit_scale_factor(P, Cnew.base_ring().modulus(), Cold.base_ring().modulus()))
            };
            assert!(P.base_ring().is_unit(&result.implicit_scale));
            return result;
        }
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
    #[instrument(skip_all)]
    fn gen_gk<'a, R: Rng + CryptoRng>(P: &PlaintextRing<Self>, C: &'a CiphertextRing<Self>, rng: R, sk: &SecretKey<Self>, g: CyclotomicGaloisGroupEl, digits: usize) -> KeySwitchKey<'a, Self>
        where Self: 'a
    {
        Self::gen_switch_key(P, C, rng, &C.get_ring().apply_galois_action(sk, g), sk, digits)
    }

    ///
    /// Converts an encrypted value `m` w.r.t. a plaintext modulus `t` to an encryption of `t' m / t` w.r.t.
    /// a plaintext modulus `t'`. This requires that `t' m / t` is an integral ring element (i.e. `t` divides
    /// `t' m`), otherwise this function will cause immediate noise overflow.
    /// 
    #[instrument(skip_all)]
    fn change_plaintext_modulus(Pnew: &PlaintextRing<Self>, Pold: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        assert!(Pold.base_ring().is_unit(&ct.implicit_scale));

        let x = C.base_ring().checked_div(
            &C.base_ring().coerce(&StaticRing::<i64>::RING, *Pnew.base_ring().modulus()),
            &C.base_ring().coerce(&StaticRing::<i64>::RING, *Pold.base_ring().modulus()),
        ).unwrap();
        let new_implicit_scale = Pnew.base_ring().coerce(&StaticRing::<i64>::RING, Pold.base_ring().smallest_positive_lift(ct.implicit_scale));
        let result = Ciphertext {
            c0: C.inclusion().mul_ref_snd_map(ct.c0, &x),
            c1: C.inclusion().mul_ref_snd_map(ct.c1, &x),
            implicit_scale: new_implicit_scale
        };
        assert!(Pnew.base_ring().is_unit(&result.implicit_scale));
        return result;
    }

    ///
    /// Creates an encryption of the secret key.
    /// 
    /// Note that this does not require access to the secret key.
    /// 
    #[instrument(skip_all)]
    fn enc_sk(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>) -> Ciphertext<Self> {
        Ciphertext {
            c0: C.zero(),
            c1: C.one(),
            implicit_scale: P.base_ring().one()
        }
    }

    ///
    /// Modulus-switches from `R/qR` to `R/t'R`, where the latter one is given as a plaintext ring `target`.
    /// In particular, this is necessary during bootstrapping.
    /// 
    /// As opposed to BFV however, the modulus `t'` of `target` must be coprime with the
    /// current plaintext modulus `t`.
    /// 
    #[instrument(skip_all)]
    fn mod_switch_to_plaintext(P: &PlaintextRing<Self>, target: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: Ciphertext<Self>) -> (El<PlaintextRing<Self>>, El<PlaintextRing<Self>>) {
        assert!(signed_gcd(*P.base_ring().modulus(), *target.base_ring().modulus(), ZZ) == 1, "can only mod-switch to ciphertext moduli that are coprime to t");
        assert!(P.base_ring().is_unit(&ct.implicit_scale));

        let mod_switch = CongruencePreservingRescaling::new_with(
            C.base_ring().as_iter().map(|Zp| *Zp).collect(),
            vec![*target.base_ring()],
            (0..C.base_ring().len()).collect(),
            *P.base_ring(),
            Global
        );
        let c0 = C.inclusion().mul_map(ct.c0, C.base_ring().coerce(&ZZ, P.base_ring().smallest_lift(P.base_ring().invert(&P.base_ring().mul(
            ct.implicit_scale,
            Self::mod_switch_down_compute_implicit_scale_factor(P, &int_cast(*target.base_ring().modulus(), ZZbig, ZZ), C.base_ring().modulus())
        )).unwrap())));
        let c1 = C.inclusion().mul_map(ct.c1, C.base_ring().coerce(&ZZ, P.base_ring().smallest_lift(P.base_ring().invert(&P.base_ring().mul(
            ct.implicit_scale,
            Self::mod_switch_down_compute_implicit_scale_factor(P, &int_cast(*target.base_ring().modulus(), ZZbig, ZZ), C.base_ring().modulus())
        )).unwrap())));
        return (
            perform_rns_op_to_plaintext_ring(target, C.get_ring(), &c0, &mod_switch),
            perform_rns_op_to_plaintext_ring(target, C.get_ring(), &c1, &mod_switch)
        );
    }
}

#[derive(Debug)]
pub struct Pow2BGV<A: Allocator + Clone + Send + Sync = DefaultCiphertextAllocator, C: Send + Sync + HERingNegacyclicNTT<Zn> = DefaultNegacyclicNTT> {
    pub log2_q_min: usize,
    pub log2_q_max: usize,
    pub log2_N: usize,
    pub ciphertext_allocator: A,
    pub negacyclic_ntt: PhantomData<C>
}

impl<A: Allocator + Clone + Send + Sync, C: Send + Sync + HERingNegacyclicNTT<Zn>> Clone for Pow2BGV<A, C> {

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

impl<A: Allocator + Clone + Send + Sync, C: Send + Sync + HERingNegacyclicNTT<Zn>> Display for Pow2BGV<A, C> {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BGV(n = 2^{}, log2(q) in {}..{})", self.log2_N + 1, self.log2_q_min, self.log2_q_max)
    }
}

impl<A: Allocator + Clone + Send + Sync, C: Send + Sync + HERingNegacyclicNTT<Zn>> BGVCiphertextParams for Pow2BGV<A, C> {

    type CiphertextRing = ManagedDoubleRNSRingBase<Pow2CyclotomicNumberRing<C>, A>;

    fn number_ring(&self) -> Pow2CyclotomicNumberRing<C> {
        Pow2CyclotomicNumberRing::new_with(2 << self.log2_N)
    }

    #[instrument(skip_all)]
    fn enc_sym_zero<R: Rng + CryptoRng>(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, rng: R, sk: &SecretKey<Self>) -> Ciphertext<Self> {
        let t = C.base_ring().coerce(&ZZ, *P.base_ring().modulus());
        let (a, b) = Self::rlwe_sample(C, rng, sk);
        let result = Ciphertext {
            c0: C.inclusion().mul_ref_snd_map(b, &t),
            c1: C.inclusion().mul_ref_snd_map(a, &t),
            implicit_scale: P.base_ring().one()
        };
        return double_rns_repr::<Self, _, _>(P, C, result);
    }

    #[instrument(skip_all)]
    fn max_rns_base(&self) -> zn_rns::Zn<Zn, BigIntRing> {
        let log2_q = self.log2_q_min..self.log2_q_max;
        let number_ring = self.number_ring();
        let required_root_of_unity = number_ring.mod_p_required_root_of_unity() as i64;
        let max_bits_per_modulus = 57;
        let mut rns_base = sample_primes(log2_q.start, log2_q.end, max_bits_per_modulus, |bound| largest_prime_leq_congruent_to_one(int_cast(bound, ZZ, ZZbig), required_root_of_unity).map(|p| int_cast(p, ZZbig, ZZ))).unwrap();
        rns_base.sort_unstable_by(|l, r| ZZbig.cmp(l, r));
        return zn_rns::Zn::new(rns_base.into_iter().map(|p| Zn::new(int_cast(p, ZZ, ZZbig) as u64)).collect(), ZZbig);
    }

    #[instrument(skip_all)]
    fn create_ciphertext_ring(&self, rns_base: zn_rns::Zn<Zn, BigIntRing>) -> CiphertextRing<Self> {
        return ManagedDoubleRNSRingBase::new_with(
            self.number_ring(),
            rns_base,
            self.ciphertext_allocator.clone()
        );
    }

    #[instrument(skip_all)]
    fn encode_plain(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: &El<PlaintextRing<Self>>) -> El<CiphertextRing<Self>> {
        let ZZ_to_Zq = C.base_ring().can_hom(P.base_ring().integer_ring()).unwrap();
        let result = C.from_canonical_basis(P.wrt_canonical_basis(m).iter().map(|c| ZZ_to_Zq.map(P.base_ring().smallest_lift(c))));
        return C.get_ring().to_doublerns(&result).map(|x| C.get_ring().from_double_rns_repr(C.get_ring().unmanaged_ring().clone_el(x))).unwrap_or(C.zero());
    }
}

#[derive(Clone, Debug)]
pub struct CompositeBGV<A: Allocator + Clone + Send + Sync = DefaultCiphertextAllocator> {
    pub log2_q_min: usize,
    pub log2_q_max: usize,
    pub n1: usize,
    pub n2: usize,
    pub ciphertext_allocator: A
}

impl<A: Allocator + Clone + Send + Sync> Display for CompositeBGV<A> {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BGV(n = {} * {}, log2(q) in {}..{})", self.n1, self.n2, self.log2_q_min, self.log2_q_max)
    }
}

impl<A: Allocator + Clone + Send + Sync> BGVCiphertextParams for CompositeBGV<A> {

    type CiphertextRing = ManagedDoubleRNSRingBase<CompositeCyclotomicNumberRing, A>;

    #[instrument(skip_all)]
    fn enc_sym_zero<R: Rng + CryptoRng>(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, rng: R, sk: &SecretKey<Self>) -> Ciphertext<Self> {
        let t = C.base_ring().coerce(&ZZ, *P.base_ring().modulus());
        let (a, b) = Self::rlwe_sample(C, rng, sk);
        let result = Ciphertext {
            c0: C.inclusion().mul_ref_snd_map(b, &t),
            c1: C.inclusion().mul_ref_snd_map(a, &t),
            implicit_scale: P.base_ring().one()
        };
        return double_rns_repr::<Self, _, _>(P, C, result);
    }

    fn number_ring(&self) -> CompositeCyclotomicNumberRing {
        CompositeCyclotomicNumberRing::new(self.n1, self.n2)
    }

    fn max_rns_base(&self) -> zn_rns::Zn<Zn, BigIntRing> {
        let log2_q = self.log2_q_min..self.log2_q_max;
        let number_ring = self.number_ring();
        let required_root_of_unity = number_ring.mod_p_required_root_of_unity() as i64;
        let max_bits_per_modulus = 57;
        let mut rns_base = sample_primes(log2_q.start, log2_q.end, max_bits_per_modulus, |bound| largest_prime_leq_congruent_to_one(int_cast(bound, ZZ, ZZbig), required_root_of_unity).map(|p| int_cast(p, ZZbig, ZZ))).unwrap();
        rns_base.sort_unstable_by(|l, r| ZZbig.cmp(l, r));
        return zn_rns::Zn::new(rns_base.into_iter().map(|p| Zn::new(int_cast(p, ZZ, ZZbig) as u64)).collect(), ZZbig);
    }

    #[instrument(skip_all)]
    fn create_ciphertext_ring(&self, rns_base: zn_rns::Zn<Zn, BigIntRing>) -> CiphertextRing<Self> {
        return ManagedDoubleRNSRingBase::new_with(
            self.number_ring(),
            rns_base,
            self.ciphertext_allocator.clone()
        );
    }

    #[instrument(skip_all)]
    fn encode_plain(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: &El<PlaintextRing<Self>>) -> El<CiphertextRing<Self>> {
        let ZZ_to_Zq = C.base_ring().can_hom(P.base_ring().integer_ring()).unwrap();
        let result = C.from_canonical_basis(P.wrt_canonical_basis(m).iter().map(|c| ZZ_to_Zq.map(P.base_ring().smallest_lift(c))));
        return C.get_ring().to_doublerns(&result).map(|x| C.get_ring().from_double_rns_repr(C.get_ring().unmanaged_ring().clone_el(x))).unwrap_or(C.zero());
    }
}

fn assert_rns_factor_drop_correct<Params>(Cnew: &CiphertextRing<Params>, Cold: &CiphertextRing<Params>, drop_moduli: &RNSFactorIndexList)
    where Params: ?Sized + BGVCiphertextParams
{
    assert_eq!(Cold.base_ring().len(), Cnew.base_ring().len() + drop_moduli.len());
    let mut i_new = 0;
    for i_old in 0..Cold.base_ring().len() {
        if drop_moduli.contains(i_old) {
            continue;
        }
        assert!(Cold.base_ring().at(i_old).get_ring() == Cnew.base_ring().at(i_new).get_ring());
        i_new += 1;
    }
}

pub fn small_basis_repr<Params, NumberRing, A>(_P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, ct: Ciphertext<Params>) -> Ciphertext<Params>
    where Params: BGVCiphertextParams<CiphertextRing = ManagedDoubleRNSRingBase<NumberRing, A>>,
        NumberRing: HECyclotomicNumberRing,
        A: Allocator + Clone
{
    return Ciphertext {
        c0: C.get_ring().from_small_basis_repr(C.get_ring().to_small_basis(&ct.c0).map(|x| C.get_ring().unmanaged_ring().get_ring().clone_el_non_fft(x)).unwrap_or_else(|| C.get_ring().unmanaged_ring().get_ring().zero_non_fft())), 
        c1: C.get_ring().from_small_basis_repr(C.get_ring().to_small_basis(&ct.c1).map(|x| C.get_ring().unmanaged_ring().get_ring().clone_el_non_fft(x)).unwrap_or_else(|| C.get_ring().unmanaged_ring().get_ring().zero_non_fft())), 
        implicit_scale: ct.implicit_scale
    };
}

pub fn double_rns_repr<Params, NumberRing, A>(_P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, ct: Ciphertext<Params>) -> Ciphertext<Params>
    where Params: BGVCiphertextParams<CiphertextRing = ManagedDoubleRNSRingBase<NumberRing, A>>,
        NumberRing: HECyclotomicNumberRing,
        A: Allocator + Clone
{
    return Ciphertext {
        c0: C.get_ring().from_double_rns_repr(C.get_ring().to_doublerns(&ct.c0).map(|x| C.get_ring().unmanaged_ring().get_ring().clone_el(x)).unwrap_or_else(|| C.get_ring().unmanaged_ring().get_ring().zero())), 
        c1: C.get_ring().from_double_rns_repr(C.get_ring().to_doublerns(&ct.c1).map(|x| C.get_ring().unmanaged_ring().get_ring().clone_el(x)).unwrap_or_else(|| C.get_ring().unmanaged_ring().get_ring().zero())), 
        implicit_scale: ct.implicit_scale
    };
}

#[derive(Clone, Debug)]
pub struct SingleRNSCompositeBGV<A: Allocator + Clone + Send + Sync = DefaultCiphertextAllocator, C: HERingConvolution<Zn> = DefaultConvolution> {
    pub log2_q_min: usize,
    pub log2_q_max: usize,
    pub n1: usize,
    pub n2: usize,
    pub ciphertext_allocator: A,
    pub convolution: PhantomData<C>
}

impl<A: Allocator + Clone + Send + Sync, C: HERingConvolution<Zn>> BGVCiphertextParams for SingleRNSCompositeBGV<A, C> {

    type CiphertextRing = SingleRNSRingBase<CompositeCyclotomicNumberRing, A, C>;

    fn number_ring(&self) -> CompositeCyclotomicNumberRing {
        CompositeCyclotomicNumberRing::new(self.n1, self.n2)
    }

    fn max_rns_base(&self) -> zn_rns::Zn<Zn, BigIntRing> {
        let log2_q = self.log2_q_min..self.log2_q_max;
        let number_ring = self.number_ring();
        let required_root_of_unity = number_ring.mod_p_required_root_of_unity() as i64;
        let max_bits_per_modulus = 57;
        let mut rns_base = sample_primes(log2_q.start, log2_q.end, max_bits_per_modulus, |bound| largest_prime_leq_congruent_to_one(int_cast(bound, ZZ, ZZbig), required_root_of_unity).map(|p| int_cast(p, ZZbig, ZZ))).unwrap();
        rns_base.sort_unstable_by(|l, r| ZZbig.cmp(l, r));
        return zn_rns::Zn::new(rns_base.into_iter().map(|p| Zn::new(int_cast(p, ZZ, ZZbig) as u64)).collect(), ZZbig);
    }

    #[instrument(skip_all)]
    fn create_ciphertext_ring(&self, rns_base: zn_rns::Zn<Zn, BigIntRing>) -> CiphertextRing<Self> {
        let max_log2_n = 1 + ZZ.abs_log2_ceil(&((self.n1 * self.n2) as i64)).unwrap();
        let convolutions = rns_base.as_iter().map(|Zp| C::new(*Zp, max_log2_n)).map(Arc::new).collect::<Vec<_>>();
        return SingleRNSRingBase::new_with(
            self.number_ring(),
            rns_base,
            self.ciphertext_allocator.clone(),
            convolutions
        );
    }
}

#[cfg(test)]
use tracing_subscriber::prelude::*;
#[cfg(test)]
use feanor_mempool::dynsize::DynLayoutMempool;
#[cfg(test)]
use feanor_mempool::AllocArc;
#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use std::ptr::Alignment;
#[cfg(test)]
use std::fmt::Debug;
#[cfg(test)]
use crate::log_time;
#[cfg(test)]
use rand::rngs::StdRng;

#[test]
fn test_pow2_bgv_enc_dec() {
    let mut rng = thread_rng();
    
    let params = Pow2BGV {
        log2_q_min: 500,
        log2_q_max: 520,
        log2_N: 7,
        ciphertext_allocator: DefaultCiphertextAllocator::default(),
        negacyclic_ntt: PhantomData::<DefaultNegacyclicNTT>
    };
    let t = 257;
    
    let P = params.create_plaintext_ring(t);
    let C = params.create_initial_ciphertext_ring();
    let sk = Pow2BGV::gen_sk(&C, &mut rng, Some(16));

    let input = P.int_hom().map(2);
    let ctxt = Pow2BGV::enc_sym(&P, &C, &mut rng, &input, &sk);
    let output = Pow2BGV::dec(&P, &C, Pow2BGV::clone_ct(&P, &C, &ctxt), &sk);
    assert_el_eq!(&P, input, output);
}

#[test]
fn test_pow2_bgv_gen_sk() {
    let mut rng = StdRng::from_seed([0; 32]);
    
    let params = Pow2BGV {
        log2_q_min: 500,
        log2_q_max: 520,
        log2_N: 7,
        ciphertext_allocator: DefaultCiphertextAllocator::default(),
        negacyclic_ntt: PhantomData::<DefaultNegacyclicNTT>
    };
    
    let C = params.create_initial_ciphertext_ring();

    let sk = Pow2BGV::gen_sk(&C, &mut rng, Some(0));
    assert_el_eq!(&C, C.zero(), &sk);
    
    let sk = Pow2BGV::gen_sk(&C, &mut rng, Some(1));
    assert!(C.wrt_canonical_basis(&sk).iter().filter(|c| C.base_ring().is_one(&c) || C.base_ring().is_neg_one(&c)).count() == 1);

    let sk = Pow2BGV::gen_sk(&C, &mut rng, None);
    assert!(C.wrt_canonical_basis(&sk).iter().filter(|c| C.base_ring().is_one(&c)).count() > 32);
    
}

#[test]
fn test_pow2_bgv_mul() {
    let mut rng = thread_rng();
    
    let params = Pow2BGV {
        log2_q_min: 500,
        log2_q_max: 520,
        log2_N: 7,
        ciphertext_allocator: DefaultCiphertextAllocator::default(),
        negacyclic_ntt: PhantomData::<DefaultNegacyclicNTT>
    };
    let t = 257;
    let digits = 3;
    
    let P = params.create_plaintext_ring(t);
    let C = params.create_initial_ciphertext_ring();
    let sk = Pow2BGV::gen_sk(&C, &mut rng, None);
    let rk = Pow2BGV::gen_rk(&P, &C, &mut rng, &sk, digits);

    let input = P.int_hom().map(2);
    let ctxt = Pow2BGV::enc_sym(&P, &C, &mut rng, &input, &sk);
    let result_ctxt = Pow2BGV::hom_mul(&P, &C, Pow2BGV::clone_ct(&P, &C, &ctxt), ctxt, &rk);
    let result = Pow2BGV::dec(&P, &C, result_ctxt, &sk);
    assert_el_eq!(&P, P.int_hom().map(4), result);
}

#[test]
fn test_pow2_bgv_modulus_switch() {
    let mut rng = thread_rng();
    
    let params = Pow2BGV {
        log2_q_min: 500,
        log2_q_max: 520,
        log2_N: 7,
        ciphertext_allocator: DefaultCiphertextAllocator::default(),
        negacyclic_ntt: PhantomData::<DefaultNegacyclicNTT>
    };
    let t = 257;
    
    let P = params.create_plaintext_ring(t);
    let C0 = params.create_initial_ciphertext_ring();
    assert_eq!(9, C0.base_ring().len());

    let sk = Pow2BGV::gen_sk(&C0, &mut rng, None);

    let input = P.int_hom().map(2);
    let ctxt = Pow2BGV::enc_sym(&P, &C0, &mut rng, &input, &sk);

    for i in [0, 1, 8] {
        let to_drop = RNSFactorIndexList::from(vec![i], C0.base_ring().len());
        let C1 = Pow2BGV::mod_switch_down_ciphertext_ring(&C0, &to_drop);
        let result_ctxt = Pow2BGV::mod_switch_down(&P, &C1, &C0, &to_drop, Pow2BGV::clone_ct(&P, &C0, &ctxt));
        let result = Pow2BGV::dec(&P, &C1, result_ctxt, &Pow2BGV::mod_switch_down_sk(&C1, &C0, &to_drop, &sk));
        assert_el_eq!(&P, P.int_hom().map(2), result);
    }
}

#[test]
fn test_pow2_change_plaintext_modulus() {
    let mut rng = thread_rng();
    
    let params = Pow2BGV {
        log2_q_min: 500,
        log2_q_max: 520,
        log2_N: 7,
        ciphertext_allocator: DefaultCiphertextAllocator::default(),
        negacyclic_ntt: PhantomData::<DefaultNegacyclicNTT>
    };
    
    let P0 = params.create_plaintext_ring(17 * 17);
    let P1 = params.create_plaintext_ring(17);
    let C = params.create_initial_ciphertext_ring();

    let sk = Pow2BGV::gen_sk(&C, &mut rng, None);

    let input = P0.int_hom().map(2 * 17);
    let ctxt = Pow2BGV::enc_sym(&P0, &C, &mut rng, &input, &sk);
    let result_ctxt = Pow2BGV::change_plaintext_modulus(&P1, &P0, &C, ctxt);
    let result = Pow2BGV::dec(&P1, &C, result_ctxt, &sk);
    assert_el_eq!(&P1, P1.int_hom().map(2), result);
}

#[test]
fn test_pow2_modulus_switch_hom_add() {
    let mut rng = thread_rng();
    
    let params = Pow2BGV {
        log2_q_min: 500,
        log2_q_max: 520,
        log2_N: 7,
        ciphertext_allocator: DefaultCiphertextAllocator::default(),
        negacyclic_ntt: PhantomData::<DefaultNegacyclicNTT>
    };
    let t = 257;
    
    let P = params.create_plaintext_ring(t);
    let C0 = params.create_initial_ciphertext_ring();
    assert_eq!(9, C0.base_ring().len());

    let sk = Pow2BGV::gen_sk(&C0, &mut rng, None);

    let input = P.int_hom().map(2);
    let ctxt = Pow2BGV::enc_sym(&P, &C0, &mut rng, &input, &sk);

    for i in [0, 1, 8] {
        let to_drop = RNSFactorIndexList::from(vec![i], C0.base_ring().len());
        let C1 = Pow2BGV::mod_switch_down_ciphertext_ring(&C0, &to_drop);
        let ctxt_modswitch = Pow2BGV::mod_switch_down(&P, &C1, &C0, &to_drop, Pow2BGV::clone_ct(&P, &C0, &ctxt));
        let sk_modswitch = Pow2BGV::mod_switch_down_sk(&C1, &C0, &to_drop, &sk);
        let ctxt_other = Pow2BGV::enc_sym(&P, &C1, &mut rng, &P.int_hom().map(30), &sk_modswitch);

        let ctxt_result = Pow2BGV::hom_add(&P, &C1, ctxt_modswitch, ctxt_other);

        let result = Pow2BGV::dec(&P, &C1, ctxt_result, &sk_modswitch);
        assert_el_eq!(&P, P.int_hom().map(32), result);
    }
}

#[test]
fn test_pow2_bgv_modulus_switch_rk() {
    let mut rng = thread_rng();
    
    let params = Pow2BGV {
        log2_q_min: 500,
        log2_q_max: 520,
        log2_N: 7,
        ciphertext_allocator: DefaultCiphertextAllocator::default(),
        negacyclic_ntt: PhantomData::<DefaultNegacyclicNTT>
    };
    let t = 257;
    let digits = 3;
    
    let P = params.create_plaintext_ring(t);
    let C0 = params.create_initial_ciphertext_ring();
    assert_eq!(9, C0.base_ring().len());

    let sk = Pow2BGV::gen_sk(&C0, &mut rng, None);
    let rk = Pow2BGV::gen_rk(&P, &C0, &mut rng, &sk, digits);

    let input = P.int_hom().map(2);
    let ctxt = Pow2BGV::enc_sym(&P, &C0, &mut rng, &input, &sk);

    for i in [0, 1, 8] {
        let to_drop = RNSFactorIndexList::from(vec![i], C0.base_ring().len());
        let C1 = Pow2BGV::mod_switch_down_ciphertext_ring(&C0, &to_drop);
        let new_sk = Pow2BGV::mod_switch_down_sk(&C1, &C0, &to_drop, &sk);
        let new_rk = Pow2BGV::mod_switch_down_rk(&C1, &C0, &to_drop, &rk);
        let ctxt2 = Pow2BGV::enc_sym(&P, &C1, &mut rng, &P.int_hom().map(3), &new_sk);
        let result_ctxt = Pow2BGV::hom_mul(
            &P,
            &C1,
            Pow2BGV::mod_switch_down(&P, &C1, &C0, &to_drop, Pow2BGV::clone_ct(&P, &C0, &ctxt)),
            ctxt2,
            &new_rk
        );
        let result = Pow2BGV::dec(&P, &C1, result_ctxt, &new_sk);
        assert_el_eq!(&P, P.int_hom().map(6), result);
    }
}

#[test]
#[ignore]
fn measure_time_pow2_bgv() {
    let (chrome_layer, _guard) = tracing_chrome::ChromeLayerBuilder::new().build();
    tracing_subscriber::registry().with(chrome_layer).init();

    let mut rng = thread_rng();
    
    let params = Pow2BGV {
        log2_q_min: 790,
        log2_q_max: 800,
        log2_N: 15,
        ciphertext_allocator: AllocArc(Arc::new(DynLayoutMempool::<Global>::new(Alignment::of::<u64>()))),
        negacyclic_ntt: PhantomData::<DefaultNegacyclicNTT>
    };
    let t = 257;
    let digits = 3;
    
    let P = log_time::<_, _, true, _>("CreatePtxtRing", |[]|
        params.create_plaintext_ring(t)
    );
    let C = log_time::<_, _, true, _>("CreateCtxtRing", |[]|
        params.create_initial_ciphertext_ring()
    );

    let sk = log_time::<_, _, true, _>("GenSK", |[]| 
        Pow2BGV::gen_sk(&C, &mut rng, None)
    );

    let m = P.int_hom().map(2);
    let ct = log_time::<_, _, true, _>("EncSym", |[]|
        Pow2BGV::enc_sym(&P, &C, &mut rng, &m, &sk)
    );

    let res = log_time::<_, _, true, _>("HomAddPlain", |[]| 
        Pow2BGV::hom_add_plain(&P, &C, &m, Pow2BGV::clone_ct(&P, &C, &ct))
    );
    assert_el_eq!(&P, &P.int_hom().map(4), &Pow2BGV::dec(&P, &C, res, &sk));

    let res = log_time::<_, _, true, _>("HomMulPlain", |[]| 
        Pow2BGV::hom_mul_plain(&P, &C, &m, Pow2BGV::clone_ct(&P, &C, &ct))
    );
    assert_el_eq!(&P, &P.int_hom().map(4), &Pow2BGV::dec(&P, &C, res, &sk));

    let rk = log_time::<_, _, true, _>("GenRK", |[]| 
        Pow2BGV::gen_rk(&P, &C, &mut rng, &sk, digits)
    );
    let ct2 = Pow2BGV::enc_sym(&P, &C, &mut rng, &m, &sk);
    let res = log_time::<_, _, true, _>("HomMul", |[]| 
        Pow2BGV::hom_mul(&P, &C, ct, ct2, &rk)
    );
    assert_el_eq!(&P, &P.int_hom().map(4), &Pow2BGV::dec(&P, &C, Pow2BGV::clone_ct(&P, &C, &res), &sk));

    let to_drop = RNSFactorIndexList::from(vec![0], C.base_ring().len());
    let C_new = Pow2BGV::mod_switch_down_ciphertext_ring(&C, &to_drop);
    let sk_new = Pow2BGV::mod_switch_down_sk(&C_new, &C, &to_drop, &sk);
    let res_new = log_time::<_, _, true, _>("ModSwitch", |[]| 
        Pow2BGV::mod_switch_down(&P, &C_new, &C, &to_drop, res)
    );
    assert_el_eq!(&P, &P.int_hom().map(4), &Pow2BGV::dec(&P, &C_new, res_new, &sk_new));
}

#[test]
#[ignore]
fn measure_time_double_rns_composite_bgv() {
    let (chrome_layer, _guard) = tracing_chrome::ChromeLayerBuilder::new().build();
    tracing_subscriber::registry().with(chrome_layer).init();

    let mut rng = thread_rng();
    
    let params = CompositeBGV {
        log2_q_min: 1090,
        log2_q_max: 1100,
        n1: 127,
        n2: 337,
        ciphertext_allocator: AllocArc(Arc::new(DynLayoutMempool::<Global>::new(Alignment::of::<u64>()))),
    };
    let t = 4;
    let digits = 3;
    
    let P = log_time::<_, _, true, _>("CreatePtxtRing", |[]|
        params.create_plaintext_ring(t)
    );
    let C = log_time::<_, _, true, _>("CreateCtxtRing", |[]|
        params.create_initial_ciphertext_ring()
    );

    let sk = log_time::<_, _, true, _>("GenSK", |[]| 
        CompositeBGV::gen_sk(&C, &mut rng, None)
    );
    
    let m = P.int_hom().map(3);
    let ct = log_time::<_, _, true, _>("EncSym", |[]|
        CompositeBGV::enc_sym(&P, &C, &mut rng, &m, &sk)
    );
    assert_el_eq!(&P, &P.int_hom().map(3), &CompositeBGV::dec(&P, &C, CompositeBGV::clone_ct(&P, &C, &ct), &sk));

    let res = log_time::<_, _, true, _>("HomAddPlain", |[]| 
        CompositeBGV::hom_add_plain(&P, &C, &m, CompositeBGV::clone_ct(&P, &C, &ct))
    );
    assert_el_eq!(&P, &P.int_hom().map(2), &CompositeBGV::dec(&P, &C, res, &sk));

    let res = log_time::<_, _, true, _>("HomMulPlain", |[]| 
        CompositeBGV::hom_mul_plain(&P, &C, &m, CompositeBGV::clone_ct(&P, &C, &ct))
    );
    assert_el_eq!(&P, &P.int_hom().map(1), &CompositeBGV::dec(&P, &C, res, &sk));

    let rk = log_time::<_, _, true, _>("GenRK", |[]| 
        CompositeBGV::gen_rk(&P, &C, &mut rng, &sk, digits)
    );
    let ct2 = CompositeBGV::enc_sym(&P, &C, &mut rng, &m, &sk);
    let res = log_time::<_, _, true, _>("HomMul", |[]| 
        CompositeBGV::hom_mul(&P, &C, ct, ct2, &rk)
    );
    assert_el_eq!(&P, &P.int_hom().map(1), &CompositeBGV::dec(&P, &C, CompositeBGV::clone_ct(&P, &C, &res), &sk));

    let to_drop = RNSFactorIndexList::from(vec![0], C.base_ring().len());
    let C_new = CompositeBGV::mod_switch_down_ciphertext_ring(&C, &to_drop);
    let sk_new = CompositeBGV::mod_switch_down_sk(&C_new, &C, &to_drop, &sk);
    let res_new = log_time::<_, _, true, _>("ModSwitch", |[]| 
        CompositeBGV::mod_switch_down(&P, &C_new, &C, &to_drop, res)
    );
    assert_el_eq!(&P, &P.int_hom().map(1), &CompositeBGV::dec(&P, &C_new, res_new, &sk_new));
}

#[test]
#[ignore]
fn measure_time_single_rns_composite_bgv() {
    let (chrome_layer, _guard) = tracing_chrome::ChromeLayerBuilder::new().build();
    tracing_subscriber::registry().with(chrome_layer).init();

    let mut rng = thread_rng();
    
    let params = SingleRNSCompositeBGV {
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
    let C = log_time::<_, _, true, _>("CreateCtxtRing", |[]|
        params.create_initial_ciphertext_ring()
    );

    let sk = log_time::<_, _, true, _>("GenSK", |[]| 
        SingleRNSCompositeBGV::gen_sk(&C, &mut rng, None)
    );
    
    let m = P.int_hom().map(3);
    let ct = log_time::<_, _, true, _>("EncSym", |[]|
        SingleRNSCompositeBGV::enc_sym(&P, &C, &mut rng, &m, &sk)
    );
    assert_el_eq!(&P, &P.int_hom().map(3), &SingleRNSCompositeBGV::dec(&P, &C, SingleRNSCompositeBGV::clone_ct(&P, &C, &ct), &sk));

    let res = log_time::<_, _, true, _>("HomAddPlain", |[]| 
        SingleRNSCompositeBGV::hom_add_plain(&P, &C, &m, SingleRNSCompositeBGV::clone_ct(&P, &C, &ct))
    );
    assert_el_eq!(&P, &P.int_hom().map(2), &SingleRNSCompositeBGV::dec(&P, &C, res, &sk));

    let res = log_time::<_, _, true, _>("HomMulPlain", |[]| 
        SingleRNSCompositeBGV::hom_mul_plain(&P, &C, &m, SingleRNSCompositeBGV::clone_ct(&P, &C, &ct))
    );
    assert_el_eq!(&P, &P.int_hom().map(1), &SingleRNSCompositeBGV::dec(&P, &C, res, &sk));

    let rk = log_time::<_, _, true, _>("GenRK", |[]| 
        SingleRNSCompositeBGV::gen_rk(&P, &C, &mut rng, &sk, digits)
    );
    let ct2 = SingleRNSCompositeBGV::enc_sym(&P, &C, &mut rng, &m, &sk);
    let res = log_time::<_, _, true, _>("HomMul", |[]| 
        SingleRNSCompositeBGV::hom_mul(&P, &C, ct, ct2, &rk)
    );
    assert_el_eq!(&P, &P.int_hom().map(1), &SingleRNSCompositeBGV::dec(&P, &C, SingleRNSCompositeBGV::clone_ct(&P, &C, &res), &sk));

    let to_drop = RNSFactorIndexList::from(vec![0], C.base_ring().len());
    let C_new = SingleRNSCompositeBGV::mod_switch_down_ciphertext_ring(&C, &to_drop);
    let sk_new = SingleRNSCompositeBGV::mod_switch_down_sk(&C_new, &C, &to_drop, &sk);
    let res_new = log_time::<_, _, true, _>("ModSwitch", |[]| 
        SingleRNSCompositeBGV::mod_switch_down(&P, &C_new, &C, &to_drop, res)
    );
    assert_el_eq!(&P, &P.int_hom().map(1), &SingleRNSCompositeBGV::dec(&P, &C_new, res_new, &sk_new));
}
