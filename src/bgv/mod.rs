use std::alloc::Global;
use std::ops::Range;

use feanor_math::homomorphism::Homomorphism;
use feanor_math::integer::{BigIntRing, IntegerRingStore};
use feanor_math::primitive_int::StaticRing;
use feanor_math::{assert_el_eq, ring::*};
use feanor_math::rings::extension::{FreeAlgebra, FreeAlgebraStore};
use feanor_math::rings::finite::{FiniteRing, FiniteRingStore};
use feanor_math::rings::zn::zn_64::Zn;
use feanor_math::rings::zn::zn_rns;
use feanor_math::rings::zn::ZnRingStore;
use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::seq::*;

use crate::cyclotomic::CyclotomicRing;
use crate::rings::bxv::BXVCiphertextRing;
use crate::rings::decomposition_ring::{DecompositionRing, DecompositionRingBase};
use crate::rnsconv;
use crate::rnsconv::bgv_rescale::CongruencePreservingRescaling;

use rand_distr::StandardNormal;
use rand::{Rng, CryptoRng};

pub type NumberRing<Params: BGVParams> = <Params::CiphertextRing as BXVCiphertextRing>::NumberRing;
pub type CiphertextRing<Params: BGVParams> = RingValue<Params::CiphertextRing>;
pub type PlaintextRing<Params: BGVParams> = DecompositionRing<NumberRing<Params>, Zn>;
pub type SecretKey<Params: BGVParams> = El<CiphertextRing<Params>>;
pub type GadgetProductOperand<'a, Params: BGVParams> = <Params::CiphertextRing as BXVCiphertextRing>::GadgetProductRhsOperand<'a>;
pub type KeySwitchKey<'a, Params: BGVParams> = (usize, (GadgetProductOperand<'a, Params>, GadgetProductOperand<'a, Params>));
pub type RelinKey<'a, Params: BGVParams> = KeySwitchKey<'a, Params>;

const ZZbig: BigIntRing = BigIntRing::RING;
const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;

///
/// A BGV ciphertext w.r.t. some [`BFVParams`]. Note that this implementation
/// does not include an automatic management of the ciphertext modulus chain,
/// it is up to the user to keep track of the RNS base used for each 
/// ciphertext.
/// 
pub struct Ciphertext<Params: ?Sized + BGVParams> {
    pub implicit_scale: El<Zn>,
    pub c0: El<CiphertextRing<Params>>,
    pub c1: El<CiphertextRing<Params>>
}

pub trait BGVParams {
    
    type CiphertextRing: BXVCiphertextRing + CyclotomicRing;

    fn max_rns_base(&self) -> zn_rns::Zn<Zn, BigIntRing>;

    fn create_ciphertext_ring<I>(&self, max_rns_base: &zn_rns::Zn<Zn, BigIntRing>, use_primes: I) -> CiphertextRing<Self>
        where I: Iterator<Item = usize>;

    fn number_ring(&self) -> NumberRing<Self>;

    fn create_plaintext_ring(&self, modulus: i64) -> PlaintextRing<Self> {
        DecompositionRingBase::new(self.number_ring(), Zn::new(modulus as u64))
    }

    fn gen_sk<R: Rng + CryptoRng>(C: &CiphertextRing<Self>, mut rng: R) -> SecretKey<Self> {
        // we sample uniform ternary secrets 
        let result = C.get_ring().sample_from_coefficient_distribution(|| (rng.next_u32() % 3) as i32 - 1);
        return result;
    }

    fn rlwe_sample<R: Rng + CryptoRng>(C: &CiphertextRing<Self>, mut rng: R, sk: &SecretKey<Self>) -> (El<CiphertextRing<Self>>, El<CiphertextRing<Self>>) {
        let a = C.random_element(|| rng.next_u64());
        let mut b = C.negate(C.mul_ref(&a, &sk));
        let e = C.get_ring().sample_from_coefficient_distribution(|| (rng.sample::<f64, _>(StandardNormal) * 3.2).round() as i32);
        C.add_assign(&mut b, e);
        return (a, b);
    }

    fn enc_sym_zero<R: Rng + CryptoRng>(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, rng: R, sk: &SecretKey<Self>) -> Ciphertext<Self> {
        let t = C.base_ring().coerce(&ZZ, *P.base_ring().modulus());
        let (a, b) = Self::rlwe_sample(C, rng, sk);
        return Ciphertext {
            c0: C.inclusion().mul_ref_snd_map(b, &t),
            c1: C.inclusion().mul_ref_snd_map(a, &t),
            implicit_scale: P.base_ring().one()
        };
    }

    fn transparent_zero(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>) -> Ciphertext<Self> {
        return Ciphertext {
            c0: C.zero(),
            c1: C.zero(),
            implicit_scale: P.base_ring().one()
        };
    }

    fn hom_add_plain(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: &El<PlaintextRing<Self>>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        record_time!(GLOBAL_TIME_RECORDER, "BGVParams::hom_add_plain", || {
            let m = C.get_ring().exact_convert_from_decompring(P, &P.inclusion().mul_ref_map(m, &ct.implicit_scale));
            return Ciphertext {
                c0: C.add(ct.c0, m),
                c1: ct.c1,
                implicit_scale: ct.implicit_scale
            };
        })
    }

    fn enc_sym<R: Rng + CryptoRng>(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, rng: R, m: &El<PlaintextRing<Self>>, sk: &SecretKey<Self>) -> Ciphertext<Self> {
        Self::hom_add_plain(P, C, m, Self::enc_sym_zero(P, C, rng, sk))
    }

    fn hom_mul_plain(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: &El<PlaintextRing<Self>>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        record_time!(GLOBAL_TIME_RECORDER, "BFVParams::hom_mul_plain", || {
            let m = C.get_ring().exact_convert_from_decompring(P, m);
            return Ciphertext {
                c0: C.mul_ref_snd(ct.c0, &m), 
                c1: C.mul(ct.c1, m),
                implicit_scale: ct.implicit_scale
            };
        })
    }

    fn hom_mul_plain_i64(_P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: i64, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        Ciphertext {
            c0: C.int_hom().mul_map(ct.c0, m as i32), 
            c1: C.int_hom().mul_map(ct.c1, m as i32),
            implicit_scale: ct.implicit_scale
        }
    }

    fn clone_ct(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: &Ciphertext<Self>) -> Ciphertext<Self> {
        Ciphertext {
            c0: C.clone_el(&ct.c0),
            c1: C.clone_el(&ct.c1),
            implicit_scale: P.base_ring().clone_el(&ct.implicit_scale)
        }
    }

    fn noise_budget(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: &Ciphertext<Self>, sk: &SecretKey<Self>) -> usize {
        let ct = Self::clone_ct(P, C, ct);
        let noisy_m = C.add(ct.c0, C.mul_ref_snd(ct.c1, sk));
        let coefficients = C.wrt_canonical_basis(&noisy_m);
        return ZZbig.abs_log2_ceil(C.base_ring().modulus()).unwrap().saturating_sub((0..coefficients.len()).map(|i| {
            let c = C.base_ring().smallest_lift(coefficients.at(i));
            let size = ZZbig.abs_log2_ceil(&c);
            return size.unwrap_or(0);
        }).max().unwrap());
    }

    fn gen_switch_key<'a, R: Rng + CryptoRng>(C: &'a CiphertextRing<Self>, mut rng: R, old_sk: &SecretKey<Self>, new_sk: &SecretKey<Self>, digits: usize) -> KeySwitchKey<'a, Self>
        where Self: 'a
    {
        let mut res0 = C.get_ring().gadget_product_rhs_empty(digits);
        let mut res1 = C.get_ring().gadget_product_rhs_empty(digits);
        for digit_i in 0..C.get_ring().gadget_vector(&res0).len() {
            let (c1, c0) = Self::rlwe_sample(C, &mut rng, new_sk);
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
        let (s0, s1) = &switch_key.1;
        let op = C.get_ring().to_gadget_product_lhs(ct.c1, switch_key.0);
        return Ciphertext {
            c0: C.add_ref_snd(ct.c0, &C.get_ring().gadget_product(&op, s0)),
            c1: C.get_ring().gadget_product(&op, s1),
            implicit_scale: ct.implicit_scale
        };
    }

    fn gen_rk<'a, R: Rng + CryptoRng>(C: &'a CiphertextRing<Self>, rng: R, sk: &SecretKey<Self>, digits: usize) -> RelinKey<'a, Self>
        where Self: 'a
    {
        Self::gen_switch_key(C, rng, &C.pow(C.clone_el(sk), 2), sk, digits)
    }

    fn hom_mul<'a>(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, lhs: Ciphertext<Self>, rhs: Ciphertext<Self>, rk: &RelinKey<'a, Self>) -> Ciphertext<Self>
        where Self: 'a
    {
        assert_el_eq!(P.base_ring(), &lhs.implicit_scale, &rhs.implicit_scale);
        record_time!(GLOBAL_TIME_RECORDER, "BFVParams::hom_mul", || {
            let [res0, res1, res2] = C.get_ring().two_by_two_convolution([&lhs.c0, &lhs.c1], [&rhs.c0, &rhs.c1]);
        
            let op = C.get_ring().to_gadget_product_lhs(res2, rk.0);
            let (s0, s1) = &rk.1;
        
            return (C.add_ref(&res0, &C.get_ring().gadget_product(&op, s0)), C.add_ref(&res1, &C.get_ring().gadget_product(&op, s1)));
        })
    }

    fn modulus_switch(P: &PlaintextRing<Self>, Cold: &CiphertextRing<Self>, Cnew: &CiphertextRing<Self>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        let shared_moduli_count = Cold.base_ring().as_iter().rev().zip(Cnew.base_ring().as_iter().rev()).take_while(|(Zn1, Zn2)| Zn1.get_ring() == Zn2.get_ring()).count();
        let rescaling = rnsconv::bgv_rescale::CongruencePreservingRescaling::new_with(
            Cold.base_ring().as_iter().copied().collect(),
            Cnew.base_ring().as_iter().rev().skip(shared_moduli_count).rev().copied().collect(),
            Cold.base_ring().len() - shared_moduli_count,
            *P.base_ring(),
            Global
        );
        let ZZbig_to_Zt = P.base_ring().can_hom(&ZZbig).unwrap();
        let implicit_scale = P.base_ring().checked_div(
            &ZZbig_to_Zt.map_ref(Cnew.base_ring().modulus()),
            &ZZbig_to_Zt.map_ref(&Cold.base_ring().modulus())
        ).unwrap();
        return Ciphertext {
            c0: Cnew.get_ring().perform_rns_op_from(Cold.get_ring(), &ct.c0, &rescaling),
            c1: Cnew.get_ring().perform_rns_op_from(Cold.get_ring(), &ct.c1, &rescaling),
            implicit_scale: P.base_ring().mul(ct.implicit_scale, implicit_scale)
        };
    }
}