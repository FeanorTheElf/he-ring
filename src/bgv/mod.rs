use std::alloc::{Allocator, Global};
use std::marker::PhantomData;
use std::ops::Range;
use std::ptr::Alignment;
use std::sync::Arc;

use feanor_math::homomorphism::Homomorphism;
use feanor_math::integer::{int_cast, BigIntRing, IntegerRingStore};
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
use crate::ciphertext_ring::{perform_rns_op, BGFVCiphertextRing};
use crate::cyclotomic::CyclotomicRing;
use crate::gadget_product::{GadgetProductLhsOperand, GadgetProductRhsOperand};
use crate::ntt::HERingNegacyclicNTT;
use crate::number_ring::{largest_prime_leq_congruent_to_one, HECyclotomicNumberRing, HENumberRing};
use crate::number_ring::pow2_cyclotomic::Pow2CyclotomicNumberRing;
use crate::number_ring::quotient::{NumberRingQuotient, NumberRingQuotientBase};
use crate::profiling::log_time;
use crate::rnsconv::bgv_rescale::CongruencePreservingRescaling;
use crate::{sample_primes, DefaultCiphertextAllocator, DefaultNegacyclicNTT};

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
/// A BGV ciphertext w.r.t. some [`BFVParams`]. Note that this implementation
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

    fn create_ciphertext_ring<I>(&self, max_rns_base: &zn_rns::Zn<Zn, BigIntRing>, use_primes: I) -> CiphertextRing<Self>
        where I: Iterator<Item = usize>;

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
        return P.from_canonical_basis(C.wrt_canonical_basis(&noisy_m).iter().map(|x| mod_t.map(C.base_ring().smallest_lift(x))));
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
        return ZZbig.abs_log2_ceil(C.base_ring().modulus()).unwrap().saturating_sub(size_of_critical_quantity);
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
    fn hom_mul<'a>(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, lhs: Ciphertext<Self>, rhs: Ciphertext<Self>, rk: &RelinKey<'a, Self>, s: Option<&El<CiphertextRing<Self>>>) -> Ciphertext<Self>
        where Self: 'a
    {
        assert_el_eq!(P.base_ring(), &lhs.implicit_scale, &rhs.implicit_scale);
        let [res0, res1, res2] = C.get_ring().two_by_two_convolution([&lhs.c0, &lhs.c1], [&rhs.c0, &rhs.c1]);

        // let s = s.unwrap();
        // C.println(&C.add(C.clone_el(&res0), C.add(C.mul_ref(&res1, s), C.mul_ref(&res2, &C.pow(C.clone_el(s), 2)))));
        
        let op = GadgetProductLhsOperand::from_element_with(C.get_ring(), &res2, rk.0.gadget_vector_moduli_indices());
        let (s0, s1) = &rk;
        
        return Ciphertext {
            c0: C.add(res0, op.gadget_product(s0, C.get_ring())), 
            c1: C.add(res1, op.gadget_product(s1, C.get_ring())),
            implicit_scale: P.base_ring().mul(lhs.implicit_scale, rhs.implicit_scale)
        };
    }

    #[instrument(skip_all)]
    fn modulus_switch(P: &PlaintextRing<Self>, Cold: &CiphertextRing<Self>, Cnew: &CiphertextRing<Self>, dropped_rns_factor_indices: &[usize], ct: Ciphertext<Self>) -> Ciphertext<Self> {
        assert_eq!(Cold.base_ring().len(), Cnew.base_ring().len() + dropped_rns_factor_indices.len());
        let mut i_new = 0;
        for i_old in 0..Cold.base_ring().len() {
            if dropped_rns_factor_indices.contains(&i_old) {
                continue;
            }
            assert!(Cold.base_ring().at(i_old).get_ring() == Cnew.base_ring().at(i_new).get_ring());
            i_new += 1;
        }

        let rescaling = CongruencePreservingRescaling::new_with(
            Cold.base_ring().as_iter().copied().collect(),
            Vec::new(),
            dropped_rns_factor_indices.to_owned(),
            *P.base_ring(),
            Global
        );
        let ZZbig_to_Zt = P.base_ring().can_hom(&ZZbig).unwrap();
        let implicit_scale = P.base_ring().checked_div(
            &ZZbig_to_Zt.map_ref(Cnew.base_ring().modulus()),
            &ZZbig_to_Zt.map_ref(&Cold.base_ring().modulus())
        ).unwrap();
        return Ciphertext {
            c0: perform_rns_op(Cnew.get_ring(), Cold.get_ring(), &ct.c0, &rescaling),
            c1: perform_rns_op(Cnew.get_ring(), Cold.get_ring(), &ct.c1, &rescaling),
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
    fn max_rns_base(&self) -> zn_rns::Zn<Zn, BigIntRing> {
        let log2_q = self.log2_q_min..self.log2_q_max;
        let number_ring = self.number_ring();
        let required_root_of_unity = number_ring.mod_p_required_root_of_unity() as i64;
        let mut rns_base = sample_primes(log2_q.start, log2_q.end, 56, |bound| largest_prime_leq_congruent_to_one(int_cast(bound, ZZ, ZZbig), required_root_of_unity).map(|p| int_cast(p, ZZbig, ZZ))).unwrap();
        rns_base.sort_unstable_by(|l, r| ZZbig.cmp(l, r));
        return zn_rns::Zn::new(rns_base.into_iter().map(|p| Zn::new(int_cast(p, ZZ, ZZbig) as u64)).collect(), ZZbig);
    }

    #[instrument(skip_all)]
    fn create_ciphertext_ring<I>(&self, max_rns_base: &zn_rns::Zn<Zn, BigIntRing>, use_primes: I) -> CiphertextRing<Self>
        where I: Iterator<Item = usize>
    {
        let current_rns_base = use_primes.map(|i| *max_rns_base.at(i));
        return ManagedDoubleRNSRingBase::new_with(
            self.number_ring(),
            zn_rns::Zn::new(current_rns_base.collect(), ZZbig),
            self.ciphertext_allocator.clone()
        );
    }
}

pub fn small_basis_repr<Params, NumberRing, A>(C: &CiphertextRing<Params>, ct: Ciphertext<Params>) -> Ciphertext<Params>
    where Params: BGVParams<CiphertextRing = ManagedDoubleRNSRingBase<NumberRing, A>>,
        NumberRing: HECyclotomicNumberRing,
        A: Allocator + Clone
{
    C.get_ring().force_small_basis_repr(&ct.c0);
    C.get_ring().force_small_basis_repr(&ct.c1);
    return ct;
}

#[cfg(test)]
use tracing_subscriber::prelude::*;
#[cfg(test)]
use feanor_mempool::dynsize::DynLayoutMempool;
#[cfg(test)]
use feanor_mempool::AllocArc;

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
    let C = params.create_ciphertext_ring(&max_rns_base, 0..max_rns_base.len());
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
    let C = params.create_ciphertext_ring(&max_rns_base, 0..max_rns_base.len());
    let sk = Pow2BGV::gen_sk(&C, &mut rng);
    let rk = Pow2BGV::gen_rk(&P, &C, &mut rng, &sk, digits);

    let input = P.int_hom().map(2);
    let ctxt = Pow2BGV::enc_sym(&P, &C, &mut rng, &input, &sk);
    let result_ctxt = Pow2BGV::hom_mul(&P, &C, Pow2BGV::clone_ct(&P, &C, &ctxt), ctxt, &rk, None);
    let result = Pow2BGV::dec(&P, &C, result_ctxt, &sk);
    assert_el_eq!(&P, P.int_hom().map(4), result);
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
    let max_rns_base = params.max_rns_base();
    let C = log_time::<_, _, true, _>("CreateCtxtRing", |[]|
        params.create_ciphertext_ring(&max_rns_base, 0..max_rns_base.len())
    );

    let sk = log_time::<_, _, true, _>("GenSK", |[]| 
        Pow2BGV::gen_sk(&C, &mut rng)
    );

    let m = P.int_hom().map(2);
    let ct = log_time::<_, _, true, _>("EncSym", |[]|
        small_basis_repr::<Pow2BGV<_, _>, _, _>(&C, Pow2BGV::enc_sym(&P, &C, &mut rng, &m, &sk))
    );

    let res = log_time::<_, _, true, _>("HomAddPlain", |[]| 
        small_basis_repr::<Pow2BGV<_, _>, _, _>(&C, Pow2BGV::hom_add_plain(&P, &C, &m, Pow2BGV::clone_ct(&P, &C, &ct)))
    );
    assert_el_eq!(&P, &P.int_hom().map(4), &Pow2BGV::dec(&P, &C, res, &sk));

    let res = log_time::<_, _, true, _>("HomMulPlain", |[]| 
        small_basis_repr::<Pow2BGV<_, _>, _, _>(&C, Pow2BGV::hom_mul_plain(&P, &C, &m, Pow2BGV::clone_ct(&P, &C, &ct)))
    );
    assert_el_eq!(&P, &P.int_hom().map(4), &Pow2BGV::dec(&P, &C, res, &sk));

    let rk = log_time::<_, _, true, _>("GenRK", |[]| 
        Pow2BGV::gen_rk(&P, &C, &mut rng, &sk, digits)
    );
    let ct2 = small_basis_repr::<Pow2BGV<_, _>, _, _>(&C, Pow2BGV::enc_sym(&P, &C, &mut rng, &m, &sk));
    let res = log_time::<_, _, true, _>("HomMul", |[]| 
        small_basis_repr::<Pow2BGV<_, _>, _, _>(&C, Pow2BGV::hom_mul(&P, &C, ct, ct2, &rk, None))
    );
    assert_el_eq!(&P, &P.int_hom().map(4), &Pow2BGV::dec(&P, &C, res, &sk));
}
