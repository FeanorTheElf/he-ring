use std::alloc::{Allocator, Global};
use std::fmt::Display;
use std::marker::PhantomData;
use std::ops::Range;
use std::ptr::Alignment;
use std::sync::Arc;
use std::time::Instant;

use feanor_math::homomorphism::Homomorphism;
use feanor_math::integer::{int_cast, BigIntRing, IntegerRingStore};
use feanor_math::matrix::OwnedMatrix;
use feanor_math::ordered::OrderedRingStore;
use feanor_math::primitive_int::StaticRing;
use feanor_math::{assert_el_eq, ring::*};
use feanor_math::rings::extension::{FreeAlgebra, FreeAlgebraStore};
use feanor_math::rings::finite::{FiniteRing, FiniteRingStore};
use feanor_math::rings::zn::zn_64::Zn;
use feanor_math::rings::zn::zn_rns;
use feanor_math::rings::zn::ZnRingStore;
use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::seq::*;
use tracing::instrument;

use crate::ciphertext_ring::double_rns_managed::ManagedDoubleRNSRingBase;
use crate::ciphertext_ring::single_rns_ring::{SingleRNSRing, SingleRNSRingBase};
use crate::ciphertext_ring::{perform_rns_op, BGFVCiphertextRing};
use crate::cyclotomic::CyclotomicRing;
use crate::gadget_product::{GadgetProductLhsOperand, GadgetProductRhsOperand};
use crate::ntt::{HERingConvolution, HERingNegacyclicNTT};
use crate::number_ring::odd_cyclotomic::CompositeCyclotomicNumberRing;
use crate::number_ring::{sample_primes, largest_prime_leq_congruent_to_one, HECyclotomicNumberRing, HENumberRing};
use crate::number_ring::pow2_cyclotomic::Pow2CyclotomicNumberRing;
use crate::number_ring::quotient::{NumberRingQuotient, NumberRingQuotientBase};
use crate::rnsconv::bgv_rescale::{CongruenceAwareAlmostExactBaseConversion, CongruencePreservingRescaling};
use crate::rnsconv::RNSOperation;
use crate::{log_time, DefaultCiphertextAllocator, DefaultConvolution, DefaultNegacyclicNTT};

use rand_distr::StandardNormal;
use rand::{thread_rng, CryptoRng, Rng};

pub type NumberRing<Params: BGVParams> = <Params::CiphertextRing as BGFVCiphertextRing>::NumberRing;
pub type CiphertextRing<Params: BGVParams> = RingValue<Params::CiphertextRing>;
pub type PlaintextRing<Params: BGVParams> = NumberRingQuotient<NumberRing<Params>, Zn>;
pub type SecretKey<Params: BGVParams> = El<CiphertextRing<Params>>;
pub type KeySwitchKey<'a, Params: BGVParams> = (GadgetProductRhsOperand<Params::CiphertextRing>, GadgetProductRhsOperand<Params::CiphertextRing>);
pub type RelinKey<'a, Params: BGVParams> = KeySwitchKey<'a, Params>;

const ZZbig: BigIntRing = BigIntRing::RING;
const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;

///
/// A BGV ciphertext w.r.t. some [`BGVParams`]. Note that this implementation
/// does not include an automatic management of the ciphertext modulus chain,
/// it is up to the user to keep track of the RNS base used for each 
/// ciphertext.
/// 
pub struct Ciphertext<Params: ?Sized + BGVParams> {
    /// the ciphertext represents the value `implicit_scale^-1 lift(c0 + c1 s) mod t`, 
    /// i.e. implicit scale stores the factor in `Z/tZ` that is introduced by modulus-switching
    pub implicit_scale: El<Zn>,
    pub c0: El<CiphertextRing<Params>>,
    pub c1: El<CiphertextRing<Params>>
}

pub trait BGVParams {
    
    type CiphertextRing: BGFVCiphertextRing + CyclotomicRing + FiniteRing;

    fn max_rns_base(&self) -> zn_rns::Zn<Zn, BigIntRing>;

    fn create_ciphertext_ring(&self, rns_base: zn_rns::Zn<Zn, BigIntRing>) -> CiphertextRing<Self>;

    fn create_initial_ciphertext_ring(&self) -> CiphertextRing<Self> {
        self.create_ciphertext_ring(self.max_rns_base())
    }

    #[instrument(skip_all)]
    fn drop_rns_factor(&self, C: &CiphertextRing<Self>, drop_factor_indices: &[usize]) -> CiphertextRing<Self> {
        RingValue::from(C.get_ring().drop_rns_factor(drop_factor_indices))
    }

    fn number_ring(&self) -> NumberRing<Self>;

    #[instrument(skip_all)]
    fn create_plaintext_ring(&self, modulus: i64) -> PlaintextRing<Self> {
        NumberRingQuotientBase::new(self.number_ring(), Zn::new(modulus as u64))
    }

    #[instrument(skip_all)]
    fn gen_sk<R: Rng + CryptoRng>(C: &CiphertextRing<Self>, mut rng: R) -> SecretKey<Self> {
        // we sample uniform ternary secrets 
        let result = C.from_canonical_basis((0..C.rank()).map(|_| C.base_ring().int_hom().map((rng.next_u32() % 3) as i32 - 1)));
        return result;
    }

    #[instrument(skip_all)]
    fn rlwe_sample<R: Rng + CryptoRng>(C: &CiphertextRing<Self>, mut rng: R, sk: &SecretKey<Self>) -> (El<CiphertextRing<Self>>, El<CiphertextRing<Self>>) {
        let a = C.random_element(|| rng.next_u64());
        let mut b = C.negate(C.mul_ref(&a, &sk));
        let e = C.from_canonical_basis((0..C.rank()).map(|_| C.base_ring().int_hom().map((rng.sample::<f64, _>(StandardNormal) * 3.2).round() as i32)));
        C.add_assign(&mut b, e);
        return (a, b);
    }

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

    fn transparent_zero(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>) -> Ciphertext<Self> {
        return Ciphertext {
            c0: C.zero(),
            c1: C.zero(),
            implicit_scale: P.base_ring().one()
        };
    }

    #[instrument(skip_all)]
    fn dec_println(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: &Ciphertext<Self>, sk: &SecretKey<Self>) {
        let m = Self::dec(P, C, Self::clone_ct(P, C, ct), sk);
        println!("ciphertext (noise budget: {} / {}):", Self::noise_budget(P, C, ct, sk), ZZbig.abs_log2_ceil(C.base_ring().modulus()).unwrap());
        P.println(&m);
        println!();
    }

    #[instrument(skip_all)]
    fn hom_add_plain(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: &El<PlaintextRing<Self>>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        let ZZ_to_Zq = C.base_ring().can_hom(P.base_ring().integer_ring()).unwrap();
        let m = C.from_canonical_basis(P.wrt_canonical_basis(m).iter().map(|c| ZZ_to_Zq.map(P.base_ring().smallest_lift(c))));
        return Ciphertext {
            c0: C.add(ct.c0, m),
            c1: ct.c1,
            implicit_scale: ct.implicit_scale
        };
    }

    #[instrument(skip_all)]
    fn enc_sym<R: Rng + CryptoRng>(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, rng: R, m: &El<PlaintextRing<Self>>, sk: &SecretKey<Self>) -> Ciphertext<Self> {
        Self::hom_add_plain(P, C, m, Self::enc_sym_zero(P, C, rng, sk))
    }

    #[instrument(skip_all)]
    fn dec(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: Ciphertext<Self>, sk: &SecretKey<Self>) -> El<PlaintextRing<Self>> {
        let noisy_m = C.add(ct.c0, C.mul_ref_snd(ct.c1, sk));
        let mod_t = P.base_ring().can_hom(&ZZbig).unwrap();
        return P.inclusion().mul_map(
            P.from_canonical_basis(C.wrt_canonical_basis(&noisy_m).iter().map(|x| mod_t.map(C.base_ring().smallest_lift(x)))),
            P.base_ring().invert(&ct.implicit_scale).unwrap()
        );
    }

    #[instrument(skip_all)]
    fn hom_mul_plain(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: &El<PlaintextRing<Self>>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        let ZZ_to_Zq = C.base_ring().can_hom(P.base_ring().integer_ring()).unwrap();
        let m = C.from_canonical_basis(P.wrt_canonical_basis(m).iter().map(|c| ZZ_to_Zq.map(P.base_ring().smallest_lift(c))));
        return Ciphertext {
            c0: C.mul_ref_snd(ct.c0, &m), 
            c1: C.mul(ct.c1, m),
            implicit_scale: ct.implicit_scale
        };
    }

    #[instrument(skip_all)]
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

    #[instrument(skip_all)]
    fn gen_switch_key<'a, R: Rng + CryptoRng>(P: &PlaintextRing<Self>, C: &'a CiphertextRing<Self>, mut rng: R, old_sk: &SecretKey<Self>, new_sk: &SecretKey<Self>, digits: usize) -> KeySwitchKey<'a, Self>
        where Self: 'a
    {
        let mut res0 = GadgetProductRhsOperand::new(C.get_ring(), digits);
        let mut res1 = GadgetProductRhsOperand::new(C.get_ring(), digits);
        for digit_i in 0..digits {
            let ct = Self::enc_sym_zero(P, C, &mut rng, new_sk);
            let digit_range = res0.gadget_vector_moduli_indices().at(digit_i).clone();
            let factor = C.base_ring().get_ring().from_congruence((0..C.base_ring().len()).map(|i2| {
                let Fp = C.base_ring().at(i2);
                if digit_range.contains(&i2) { Fp.one() } else { Fp.zero() } 
            }));
            let mut payload = C.clone_el(&old_sk);
            C.inclusion().mul_assign_ref_map(&mut payload, &factor);
            C.add_assign(&mut payload, ct.c0);
            res0.set_rns_factor(C.get_ring(), digit_i, payload);
            res1.set_rns_factor(C.get_ring(), digit_i, ct.c1);
        }
        return (res0, res1);
    }

    #[instrument(skip_all)]
    fn key_switch<'a>(C: &CiphertextRing<Self>, ct: Ciphertext<Self>, switch_key: &KeySwitchKey<'a, Self>) -> Ciphertext<Self>
        where Self: 'a
    {
        let (s0, s1) = switch_key;
        let op = GadgetProductLhsOperand::from_element_with(C.get_ring(), &ct.c1, switch_key.0.gadget_vector_moduli_indices());
        return Ciphertext {
            c0: C.add_ref_snd(ct.c0, &op.gadget_product(s0, C.get_ring())),
            c1: op.gadget_product(s1, C.get_ring()),
            implicit_scale: ct.implicit_scale
        };
    }

    #[instrument(skip_all)]
    fn gen_rk<'a, R: Rng + CryptoRng>(P: &PlaintextRing<Self>, C: &'a CiphertextRing<Self>, rng: R, sk: &SecretKey<Self>, digits: usize) -> RelinKey<'a, Self>
        where Self: 'a
    {
        Self::gen_switch_key(P, C, rng, &C.pow(C.clone_el(sk), 2), sk, digits)
    }

    #[instrument(skip_all)]
    fn hom_mul<'a>(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, lhs: Ciphertext<Self>, rhs: Ciphertext<Self>, rk: &RelinKey<'a, Self>) -> Ciphertext<Self>
        where Self: 'a
    {
        let [res0, res1, res2] = C.get_ring().two_by_two_convolution([&lhs.c0, &lhs.c1], [&rhs.c0, &rhs.c1]);
        
        let op = GadgetProductLhsOperand::from_element_with(C.get_ring(), &res2, rk.0.gadget_vector_moduli_indices());
        let (s0, s1) = &rk;
        
        return Ciphertext {
            c0: C.add(res0, op.gadget_product(s0, C.get_ring())), 
            c1: C.add(res1, op.gadget_product(s1, C.get_ring())),
            implicit_scale: P.base_ring().mul(lhs.implicit_scale, rhs.implicit_scale)
        };
    }

    #[instrument(skip_all)]
    fn mod_switch_sk(P: &PlaintextRing<Self>, Cnew: &CiphertextRing<Self>, Cold: &CiphertextRing<Self>, dropped_rns_factor_indices: &[usize], sk: &SecretKey<Self>) -> SecretKey<Self> {
        assert_rns_factor_drop_correct::<Self>(Cnew, Cold, dropped_rns_factor_indices);
        if dropped_rns_factor_indices.len() == 0 {
            assert!(Cnew.base_ring().get_ring() == Cold.base_ring().get_ring());
            return Cnew.clone_el(sk);
        }
        return Cnew.get_ring().drop_rns_factor_element(Cold.get_ring(), dropped_rns_factor_indices, Cold.clone_el(sk));
    }

    #[instrument(skip_all)]
    fn mod_switch_rk<'a, 'b>(P: &PlaintextRing<Self>, Cnew: &'b CiphertextRing<Self>, Cold: &CiphertextRing<Self>, dropped_rns_factor_indices: &[usize], rk: &RelinKey<'a, Self>) -> RelinKey<'b, Self> {
        assert_rns_factor_drop_correct::<Self>(Cnew, Cold, dropped_rns_factor_indices);
        if dropped_rns_factor_indices.len() == 0 {
            assert!(Cnew.base_ring().get_ring() == Cold.base_ring().get_ring());
            return (rk.0.clone(Cnew.get_ring()), rk.1.clone(Cnew.get_ring()));
        }
        return (rk.0.clone(Cold.get_ring()).modulus_switch(Cnew.get_ring(), dropped_rns_factor_indices, Cold.get_ring()), rk.1.clone(Cold.get_ring()).modulus_switch(Cnew.get_ring(), dropped_rns_factor_indices, Cold.get_ring()));
    }

    #[instrument(skip_all)]
    fn mod_switch_gk<'a, 'b>(P: &PlaintextRing<Self>, Cnew: &'b CiphertextRing<Self>, Cold: &CiphertextRing<Self>, dropped_rns_factor_indices: &[usize], gk: &KeySwitchKey<'a, Self>) -> KeySwitchKey<'b, Self> {
        assert_rns_factor_drop_correct::<Self>(Cnew, Cold, dropped_rns_factor_indices);
        if dropped_rns_factor_indices.len() == 0 {
            assert!(Cnew.base_ring().get_ring() == Cold.base_ring().get_ring());
            return (gk.0.clone(Cnew.get_ring()), gk.1.clone(Cnew.get_ring()));
        }
        return (gk.0.clone(Cold.get_ring()).modulus_switch(Cnew.get_ring(), dropped_rns_factor_indices, Cold.get_ring()), gk.1.clone(Cold.get_ring()).modulus_switch(Cnew.get_ring(), dropped_rns_factor_indices, Cold.get_ring()));
    }

    #[instrument(skip_all)]
    fn mod_switch(P: &PlaintextRing<Self>, Cnew: &CiphertextRing<Self>, Cold: &CiphertextRing<Self>, dropped_rns_factor_indices: &[usize], ct: Ciphertext<Self>) -> Ciphertext<Self> {
        assert_rns_factor_drop_correct::<Self>(Cnew, Cold, dropped_rns_factor_indices);
        if dropped_rns_factor_indices.len() == 0 {
            assert!(Cnew.base_ring().get_ring() == Cold.base_ring().get_ring());
            return ct;
        }

        let compute_delta = CongruenceAwareAlmostExactBaseConversion::new_with(
            dropped_rns_factor_indices.iter().map(|i| *Cold.base_ring().at(*i)).collect(),
            Cnew.base_ring().as_iter().cloned().collect(),
            *P.base_ring(),
            Global
        );
        let mod_switch_ring_element = |x: El<CiphertextRing<Self>>| {
            // this logic is slightly complicated, since we want to avoid using `perform_rns_op()`;
            // in particular, we only need to convert a part of `x` into coefficient/small-basis representation,
            // while just using `perform_rns_op()` would convert all of `x`.
            let mut mod_b_part_of_x = OwnedMatrix::zero(dropped_rns_factor_indices.len(), Cold.get_ring().small_generating_set_len(), Cold.base_ring().at(0));
            Cold.get_ring().partial_representation_wrt_small_generating_set(&x, dropped_rns_factor_indices, mod_b_part_of_x.data_mut());
            // this is the "correction", subtracting it will make `x` divisible by the moduli to drop
            let mut delta = OwnedMatrix::zero(Cnew.base_ring().len(), Cnew.get_ring().small_generating_set_len(), Cnew.base_ring().at(0));
            compute_delta.apply(mod_b_part_of_x.data(), delta.data_mut());
            let delta = Cnew.get_ring().from_representation_wrt_small_generating_set(delta.data());
            // now subtract `delta` and scale by the moduli to drop - since `x - delta` is divisible by those,
            // this is actually a rescaling and not only a division in `Z/qZ`
            return Cnew.inclusion().mul_map(
                Cnew.sub(
                    Cnew.get_ring().drop_rns_factor_element(Cold.get_ring(), dropped_rns_factor_indices, x),
                    delta
                ),
                Cnew.base_ring().invert(&Cnew.base_ring().coerce(&ZZbig, ZZbig.prod(dropped_rns_factor_indices.iter().map(|i| int_cast(*Cold.base_ring().at(*i).modulus(), ZZbig, ZZ))))).unwrap()
            )
        };
        
        let ZZbig_to_Zt = P.base_ring().can_hom(&ZZbig).unwrap();
        let implicit_scale = P.base_ring().checked_div(
            &ZZbig_to_Zt.map_ref(Cnew.base_ring().modulus()),
            &ZZbig_to_Zt.map_ref(&Cold.base_ring().modulus())
        ).unwrap();
        return Ciphertext {
            c0: mod_switch_ring_element(ct.c0),
            c1: mod_switch_ring_element(ct.c1),
            implicit_scale: P.base_ring().mul(ct.implicit_scale, implicit_scale)
        };
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

impl<A: Allocator + Clone + Send + Sync, C: Send + Sync + HERingNegacyclicNTT<Zn>> BGVParams for Pow2BGV<A, C> {

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
        let mut rns_base = sample_primes(log2_q.start, log2_q.end, 56, |bound| largest_prime_leq_congruent_to_one(int_cast(bound, ZZ, ZZbig), required_root_of_unity).map(|p| int_cast(p, ZZbig, ZZ))).unwrap();
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

impl<A: Allocator + Clone + Send + Sync> BGVParams for CompositeBGV<A> {

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
        let mut rns_base = sample_primes(log2_q.start, log2_q.end, 56, |bound| largest_prime_leq_congruent_to_one(int_cast(bound, ZZ, ZZbig), required_root_of_unity).map(|p| int_cast(p, ZZbig, ZZ))).unwrap();
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
}

fn assert_rns_factor_drop_correct<Params>(Cnew: &CiphertextRing<Params>, Cold: &CiphertextRing<Params>, dropped_rns_factor_indices: &[usize])
    where Params: ?Sized + BGVParams
{
    assert_eq!(Cold.base_ring().len(), Cnew.base_ring().len() + dropped_rns_factor_indices.len());
    let mut i_new = 0;
    for i_old in 0..Cold.base_ring().len() {
        if dropped_rns_factor_indices.contains(&i_old) {
            continue;
        }
        assert!(Cold.base_ring().at(i_old).get_ring() == Cnew.base_ring().at(i_new).get_ring());
        i_new += 1;
    }
}

pub fn small_basis_repr<Params, NumberRing, A>(P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, ct: Ciphertext<Params>) -> Ciphertext<Params>
    where Params: BGVParams<CiphertextRing = ManagedDoubleRNSRingBase<NumberRing, A>>,
        NumberRing: HECyclotomicNumberRing,
        A: Allocator + Clone
{
    return Ciphertext {
        c0: C.get_ring().from_small_basis_repr(C.get_ring().to_small_basis(&ct.c0).map(|x| C.get_ring().unmanaged_ring().get_ring().clone_el_non_fft(x)).unwrap_or_else(|| C.get_ring().unmanaged_ring().get_ring().zero_non_fft())), 
        c1: C.get_ring().from_small_basis_repr(C.get_ring().to_small_basis(&ct.c1).map(|x| C.get_ring().unmanaged_ring().get_ring().clone_el_non_fft(x)).unwrap_or_else(|| C.get_ring().unmanaged_ring().get_ring().zero_non_fft())), 
        implicit_scale: ct.implicit_scale
    };
}

pub fn double_rns_repr<Params, NumberRing, A>(P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, ct: Ciphertext<Params>) -> Ciphertext<Params>
    where Params: BGVParams<CiphertextRing = ManagedDoubleRNSRingBase<NumberRing, A>>,
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

impl<A: Allocator + Clone + Send + Sync, C: HERingConvolution<Zn>> BGVParams for SingleRNSCompositeBGV<A, C> {

    type CiphertextRing = SingleRNSRingBase<CompositeCyclotomicNumberRing, A, C>;

    fn number_ring(&self) -> CompositeCyclotomicNumberRing {
        CompositeCyclotomicNumberRing::new(self.n1, self.n2)
    }

    fn max_rns_base(&self) -> zn_rns::Zn<Zn, BigIntRing> {
        let log2_q = self.log2_q_min..self.log2_q_max;
        let number_ring = self.number_ring();
        let required_root_of_unity = number_ring.mod_p_required_root_of_unity() as i64;
        let mut rns_base = sample_primes(log2_q.start, log2_q.end, 56, |bound| largest_prime_leq_congruent_to_one(int_cast(bound, ZZ, ZZbig), required_root_of_unity).map(|p| int_cast(p, ZZbig, ZZ))).unwrap();
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
use std::fmt::Debug;

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
    let max_rns_base = params.max_rns_base();
    let C = params.create_initial_ciphertext_ring();
    let sk = Pow2BGV::gen_sk(&C, &mut rng);

    let input = P.int_hom().map(2);
    let ctxt = Pow2BGV::enc_sym(&P, &C, &mut rng, &input, &sk);
    let output = Pow2BGV::dec(&P, &C, Pow2BGV::clone_ct(&P, &C, &ctxt), &sk);
    assert_el_eq!(&P, input, output);
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
    let max_rns_base = params.max_rns_base();
    let C = params.create_initial_ciphertext_ring();
    let sk = Pow2BGV::gen_sk(&C, &mut rng);
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
    let digits = 3;
    
    let P = params.create_plaintext_ring(t);
    let C0 = params.create_initial_ciphertext_ring();
    assert_eq!(9, C0.base_ring().len());

    let sk = Pow2BGV::gen_sk(&C0, &mut rng);
    let rk = Pow2BGV::gen_rk(&P, &C0, &mut rng, &sk, digits);

    let input = P.int_hom().map(2);
    let ctxt = Pow2BGV::enc_sym(&P, &C0, &mut rng, &input, &sk);

    let C1 = params.drop_rns_factor(&C0, &[0]);
    let result_ctxt = Pow2BGV::mod_switch(&P, &C1, &C0, &[0], Pow2BGV::clone_ct(&P, &C0, &ctxt));
    let result = Pow2BGV::dec(&P, &C1, result_ctxt, &Pow2BGV::mod_switch_sk(&P, &C1, &C0, &[0], &sk));
    assert_el_eq!(&P, P.int_hom().map(2), result);

    let C1 = params.drop_rns_factor(&C0, &[1]);
    let result_ctxt = Pow2BGV::mod_switch(&P, &C1, &C0, &[1], Pow2BGV::clone_ct(&P, &C0, &ctxt));
    let result = Pow2BGV::dec(&P, &C1, result_ctxt, &Pow2BGV::mod_switch_sk(&P, &C1, &C0, &[1], &sk));
    assert_el_eq!(&P, P.int_hom().map(2), result);

    let C1 = params.drop_rns_factor(&C0, &[8]);
    let result_ctxt = Pow2BGV::mod_switch(&P, &C1, &C0, &[8], Pow2BGV::clone_ct(&P, &C0, &ctxt));
    let result = Pow2BGV::dec(&P, &C1, result_ctxt, &Pow2BGV::mod_switch_sk(&P, &C1, &C0, &[8], &sk));
    assert_el_eq!(&P, P.int_hom().map(2), result);
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

    let sk = Pow2BGV::gen_sk(&C0, &mut rng);
    let rk = Pow2BGV::gen_rk(&P, &C0, &mut rng, &sk, digits);

    let input = P.int_hom().map(2);
    let ctxt = Pow2BGV::enc_sym(&P, &C0, &mut rng, &input, &sk);

    for i in [0, 1, 8] {
        let C1 = params.drop_rns_factor(&C0, &[i]);
        let new_sk = Pow2BGV::mod_switch_sk(&P, &C1, &C0, &[i], &sk);
        let new_rk = Pow2BGV::mod_switch_rk(&P, &C1, &C0, &[i], &rk);
        let ctxt2 = Pow2BGV::enc_sym(&P, &C1, &mut rng, &P.int_hom().map(3), &new_sk);
        let result_ctxt = Pow2BGV::hom_mul(
            &P,
            &C1,
            Pow2BGV::mod_switch(&P, &C1, &C0, &[i], Pow2BGV::clone_ct(&P, &C0, &ctxt)),
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
        Pow2BGV::gen_sk(&C, &mut rng)
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

    let C_new = params.drop_rns_factor(&C, &[0]);
    let sk_new = Pow2BGV::mod_switch_sk(&P, &C_new, &C, &[0], &sk);
    let res_new = log_time::<_, _, true, _>("ModSwitch", |[]| 
        Pow2BGV::mod_switch(&P, &C_new, &C, &[0], res)
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
        CompositeBGV::gen_sk(&C, &mut rng)
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

    let C_new = params.drop_rns_factor(&C, &[0]);
    let sk_new = CompositeBGV::mod_switch_sk(&P, &C_new, &C, &[0], &sk);
    let res_new = log_time::<_, _, true, _>("ModSwitch", |[]| 
        CompositeBGV::mod_switch(&P, &C_new, &C, &[0], res)
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
        SingleRNSCompositeBGV::gen_sk(&C, &mut rng)
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

    let C_new = params.drop_rns_factor(&C, &[0]);
    let sk_new = SingleRNSCompositeBGV::mod_switch_sk(&P, &C_new, &C, &[0], &sk);
    let res_new = log_time::<_, _, true, _>("ModSwitch", |[]| 
        SingleRNSCompositeBGV::mod_switch(&P, &C_new, &C, &[0], res)
    );
    assert_el_eq!(&P, &P.int_hom().map(1), &SingleRNSCompositeBGV::dec(&P, &C_new, res_new, &sk_new));
}

#[cfg(test)]
pub fn tree_mul_benchmark<Params>(params: Params, digits: usize)
    where Params: Display + BGVParams
{
    use crate::gadget_product::recommended_rns_factors_to_drop;

    let mut rng = thread_rng();
    let t = 4;

    let P = params.create_plaintext_ring(t);
    let C = params.create_initial_ciphertext_ring();
    let mut sk = Params::gen_sk(&C, &mut rng);
    let mut rk = Params::gen_rk(&P, &C, &mut rng, &sk, digits);

    let log2_count = 4;
    let mut current = (0..(1 << log2_count)).map(|i| Params::enc_sym(&P, &C, &mut rng, &P.int_hom().map(2 * i + 1), &sk)).map(Some).collect::<Vec<_>>();
    let start = Instant::now();
    let mut C_current = C;

    for i in 0..log2_count {
        let mid = current.len() / 2;
        let (left, right) = current.split_at_mut(mid);
        assert_eq!(left.len(), right.len());
        for j in 0..left.len() {
            left[j] = Some(Params::hom_mul(&P, &C_current, left[j].take().unwrap(), right[j].take().unwrap(), &rk));
        }
        current.truncate(mid);

        let drop_prime_count = if i == 0 { 4 } else if i + 1 == log2_count { 0 } else { 2 };
        let drop_rns_base_factors = recommended_rns_factors_to_drop(C_current.base_ring().len(), digits, drop_prime_count);
        let C_new = RingValue::from(C_current.get_ring().drop_rns_factor(&drop_rns_base_factors));
        for ct in &mut current {
            *ct = Some(Params::mod_switch(&P, &C_new, &C_current, &drop_rns_base_factors, ct.take().unwrap()));
        }
        sk = Params::mod_switch_sk(&P, &C_new, &C_current, &drop_rns_base_factors, &sk);
        rk = Params::mod_switch_rk(&P, &C_new, &C_current, &drop_rns_base_factors, &rk);
        C_current = C_new;
    }
    let end = Instant::now();

    println!("{}", params);
    println!("digits = {}", digits);
    println!("Tree-wise multiplication of {} inputs took {} ms, so {} ms/mul", 1 << log2_count, (end - start).as_millis(), (end - start).as_millis() / ((1 << log2_count) - 1));
    println!("Final noise budget: {}", Params::noise_budget(&P, &C_current, current[0].as_ref().unwrap(), &sk));
    assert_el_eq!(&P, &P.int_hom().map((0..(1 << log2_count)).map(|i| 2 * i + 1).product::<i32>()), Params::dec(&P, &C_current, current[0].take().unwrap(), &sk));
}

#[cfg(test)]
pub fn chain_mul_benchmark<Params>(params: Params, digits: usize)
    where Params: Display + BGVParams
{
    use crate::gadget_product::recommended_rns_factors_to_drop;

    let mut rng = thread_rng();
    let t = 4;

    let P = params.create_plaintext_ring(t);
    let C = params.create_initial_ciphertext_ring();
    let mut sk = Params::gen_sk(&C, &mut rng);
    let mut rk = Params::gen_rk(&P, &C, &mut rng, &sk, digits);

    let count = 4;
    let mut current = Params::enc_sym(&P, &C, &mut rng, &P.int_hom().map(1), &sk);
    let start = Instant::now();
    let mut C_current = C;

    for i in 0..count {
        let left = Params::clone_ct(&P, &C_current, &current);
        let right = current;
        current = Params::hom_mul(&P, &C_current, left, right, &rk);

        let drop_prime_count = if i == 0 { 4 } else if i + 1 == count { 0 } else { 2 };
        let drop_rns_base_factors = recommended_rns_factors_to_drop(C_current.base_ring().len(), digits, drop_prime_count);
        let C_new = RingValue::from(C_current.get_ring().drop_rns_factor(&drop_rns_base_factors));
        current = Params::mod_switch(&P, &C_new, &C_current, &drop_rns_base_factors, current);
        sk = Params::mod_switch_sk(&P, &C_new, &C_current, &drop_rns_base_factors, &sk);
        rk = Params::mod_switch_rk(&P, &C_new, &C_current, &drop_rns_base_factors, &rk);
        C_current = C_new;
    }
    let end = Instant::now();

    println!("{}", params);
    println!("digits = {}", digits);
    println!("Repeated squaring of ciphertext {} times took {} ms, so {} ms/mul", count, (end - start).as_millis(), (end - start).as_millis() / count as u128);
    println!("Final noise budget: {}", Params::noise_budget(&P, &C_current, &current, &sk));
    assert_el_eq!(&P, &P.one(), Params::dec(&P, &C_current, current, &sk));
}

#[test]
#[ignore]
fn bgv_mul_benchmark() {
    let (chrome_layer, _guard) = tracing_chrome::ChromeLayerBuilder::new().build();
    tracing_subscriber::registry().with(chrome_layer).init();
    
    let params = CompositeBGV {
        log2_q_min: 680,
        log2_q_max: 690,
        n1: 85,
        n2: 257,
        ciphertext_allocator: AllocArc(Arc::new(DynLayoutMempool::<Global>::new(Alignment::of::<u64>()))),
    };
    tree_mul_benchmark(params.clone(), 4);
    chain_mul_benchmark(params, 4);

    let params = CompositeBGV {
        log2_q_min: 1090,
        log2_q_max: 1100,
        n1: 127,
        n2: 337,
        ciphertext_allocator: AllocArc(Arc::new(DynLayoutMempool::<Global>::new(Alignment::of::<u64>()))),
    };
    tree_mul_benchmark(params.clone(), 4);
    chain_mul_benchmark(params, 4);
}