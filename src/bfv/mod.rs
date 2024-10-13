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
use crate::rings::number_ring_quo::*;
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

pub type PlaintextAllocator = Global;
pub type CiphertextAllocator = Global;

pub type CiphertextRing<Params: BFVParams> = DoubleRNSRing<Params::NumberRing, Zn, CiphertextAllocator>;
pub type PlaintextRing<Params: BFVParams> = NumberRingQuo<Params::NumberRing, Zn, PlaintextAllocator>;
pub type SecretKey<Params: BFVParams> = El<CiphertextRing<Params>>;
pub type Ciphertext<Params: BFVParams> = (CoeffOrNTTRingEl<Params>, CoeffOrNTTRingEl<Params>);
pub type GadgetProductOperand<'a, Params: BFVParams> = GadgetProductRhsOperand<'a, Params::NumberRing, CiphertextAllocator>;
pub type KeySwitchKey<'a, Params: BFVParams> = (GadgetProductOperand<'a, Params>, GadgetProductOperand<'a, Params>);
pub type RelinKey<'a, Params: BFVParams> = (GadgetProductOperand<'a, Params>, GadgetProductOperand<'a, Params>);

pub struct MulConversionData {
    lift_to_C_mul: rnsconv::shared_lift::AlmostExactSharedBaseConversion<CiphertextAllocator>,
    scale_down_to_C: rnsconv::bfv_rescale::AlmostExactRescalingConvert<CiphertextAllocator>
}

pub struct ModSwitchData {
    scale: rnsconv::bfv_rescale::AlmostExactRescaling<CiphertextAllocator>
}

pub trait BFVParams {

    type NumberRing: DecomposableCyclotomicNumberRing<Zn> + DecomposableCyclotomicNumberRing<Zn>;

    fn log2_q(&self) -> Range<usize>;
    fn number_ring(&self) -> Self::NumberRing;
    fn t(&self) -> i64;

    fn create_plaintext_ring(&self, modulus: i64) -> PlaintextRing<Self> {
        NumberRingQuoBase::new(self.number_ring(), Zn::new(modulus as u64))
    }

    fn create_ciphertext_rings(&self) -> (CiphertextRing<Self>, CiphertextRing<Self>) {
        let log2_q = self.log2_q();
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

}

pub fn create_multiplication_rescale<Params: BFVParams>(P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, Cmul: &CiphertextRing<Params>) -> MulConversionData {
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

pub struct CoeffOrNTTRingEl<P: BFVParams> {
    ntt_part: Option<DoubleRNSEl<P::NumberRing, Zn, CiphertextAllocator>>,
    coeff_part: Option<CoeffEl<P::NumberRing, Zn, CiphertextAllocator>>,
    params: PhantomData<P>
}

impl<P: BFVParams> CoeffOrNTTRingEl<P> {

    pub fn ntt_repr(self, C: &CiphertextRing<P>) -> Self {
        CoeffOrNTTRingEl::from_ntt(self.to_ntt(C))
    }

    pub fn coeff_repr(self, C: &CiphertextRing<P>) -> Self {
        CoeffOrNTTRingEl::from_coeff(self.to_coeff(C))
    }

    pub fn from_ntt(el: El<CiphertextRing<P>>) -> Self {
        Self {
            coeff_part: None,
            ntt_part: Some(el), 
            params: PhantomData
        }
    }

    pub fn from_coeff(el: CoeffEl<P::NumberRing, Zn, CiphertextAllocator>) -> Self {
        Self {
            coeff_part: Some(el),
            ntt_part: None, 
            params: PhantomData
        }
    }

    pub fn zero() -> Self {
        Self {
            coeff_part: None,
            ntt_part: None,
            params: PhantomData
        }
    }

    pub fn to_ntt(self, C: &CiphertextRing<P>) -> El<CiphertextRing<P>> {
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

    pub fn to_coeff(self, C: &CiphertextRing<P>) -> CoeffEl<P::NumberRing, Zn, CiphertextAllocator> {
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

    pub fn gadget_product<'a>(lhs: &GadgetProductLhsOperand<'a, P::NumberRing, CiphertextAllocator>, rhs: &GadgetProductRhsOperand<'a, P::NumberRing, CiphertextAllocator>, C: &CiphertextRing<P>) -> CoeffOrNTTRingEl<P> {
        match C.get_ring().preferred_output_repr(lhs, rhs) {
            ElRepr::Coeff => CoeffOrNTTRingEl { ntt_part: None, coeff_part: Some(C.get_ring().gadget_product_coeff(lhs, rhs)), params: PhantomData },
            ElRepr::NTT => CoeffOrNTTRingEl { ntt_part: Some(C.get_ring().gadget_product_ntt(lhs, rhs)), coeff_part: None, params: PhantomData },
        }
    }

    pub fn add(lhs: CoeffOrNTTRingEl<P>, rhs: &CoeffOrNTTRingEl<P>, C: &CiphertextRing<P>) -> CoeffOrNTTRingEl<P> {
        CoeffOrNTTRingEl {
            ntt_part: if lhs.ntt_part.is_some() && rhs.ntt_part.is_some() { Some(C.add_ref_snd(lhs.ntt_part.unwrap(), rhs.ntt_part.as_ref().unwrap())) } else { lhs.ntt_part.or(rhs.ntt_part.as_ref().map(|x| C.clone_el(x)))},
            coeff_part: if lhs.coeff_part.is_some() && rhs.coeff_part.is_some() {
                let mut result  = lhs.coeff_part.unwrap();
                C.get_ring().add_assign_non_fft(&mut result, rhs.coeff_part.as_ref().unwrap());
                Some(result)
            } else { lhs.coeff_part.or(rhs.coeff_part.as_ref().map(|x| C.get_ring().clone_el_non_fft(x))) },
            params: PhantomData
        }
    }

    pub fn sub(lhs: CoeffOrNTTRingEl<P>, rhs: &CoeffOrNTTRingEl<P>, C: &CiphertextRing<P>) -> CoeffOrNTTRingEl<P> {
        CoeffOrNTTRingEl {
            ntt_part: if lhs.ntt_part.is_some() && rhs.ntt_part.is_some() { Some(C.sub_ref_snd(lhs.ntt_part.unwrap(), rhs.ntt_part.as_ref().unwrap())) } else { lhs.ntt_part.or(rhs.ntt_part.as_ref().map(|x| C.negate(C.clone_el(x))))},
            coeff_part: if lhs.coeff_part.is_some() && rhs.coeff_part.is_some() {
                let mut result  = lhs.coeff_part.unwrap();
                C.get_ring().sub_assign_non_fft(&mut result, rhs.coeff_part.as_ref().unwrap());
                Some(result)
            } else { lhs.coeff_part.or(rhs.coeff_part.as_ref().map(|x| C.get_ring().negate_non_fft(C.get_ring().clone_el_non_fft(x)))) },
            params: PhantomData
        }
    }

    pub fn mul_i64(mut val: CoeffOrNTTRingEl<P>, scalar: i64, C: &CiphertextRing<P>) -> CoeffOrNTTRingEl<P> {
        let hom = C.base_ring().can_hom(&StaticRing::<i64>::RING).unwrap();
        if let Some(ntt_part) = &mut val.ntt_part {
            C.inclusion().mul_assign_map(ntt_part, hom.map(scalar));
        }
        if let Some(coeff_part) = &mut val.coeff_part {
            C.get_ring().mul_scalar_assign_non_fft(coeff_part, &hom.map(scalar));
        }
        return val;
    }

    pub fn clone(&self, C: &CiphertextRing<P>) -> CoeffOrNTTRingEl<P> {
        CoeffOrNTTRingEl { 
            ntt_part: self.ntt_part.as_ref().map(|x| C.clone_el(x)), 
            coeff_part: self.coeff_part.as_ref().map(|x| C.get_ring().clone_el_non_fft(x)),
            params: PhantomData
        }
    }
}

const ZZbig: BigIntRing = BigIntRing::RING;
const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;

#[derive(Clone, Debug)]
pub struct Pow2BFVParams {
    t: i64,
    log2_q_min: usize,
    log2_q_max: usize,
    log2_N: usize
}

impl BFVParams for Pow2BFVParams {

    type NumberRing = Pow2CyclotomicDecomposableNumberRing;

    fn t(&self) -> i64 {
        self.t
    }

    fn number_ring(&self) -> Self::NumberRing {
        Pow2CyclotomicDecomposableNumberRing::new(2 << self.log2_N)
    }

    fn log2_q(&self) -> Range<usize> {
        self.log2_q_min..self.log2_q_max
    }
}

#[derive(Clone, Debug)]
pub struct CompositeBFVParams {
    t: i64,
    log2_q_min: usize,
    log2_q_max: usize,
    n1: usize,
    n2: usize
}

impl BFVParams for CompositeBFVParams {

    type NumberRing = CompositeCyclotomicDecomposableNumberRing;

    fn number_ring(&self) -> Self::NumberRing {
        CompositeCyclotomicDecomposableNumberRing::new(self.n1, self.n2)
    }

    fn t(&self) -> i64 {
        self.t
    }

    fn log2_q(&self) -> Range<usize> {
        self.log2_q_min..self.log2_q_max
    }
}

pub fn gen_sk<R: Rng + CryptoRng, Params: BFVParams>(C: &CiphertextRing<Params>, mut rng: R) -> SecretKey<Params> {
    // we sample uniform ternary secrets 
    let result = C.get_ring().sample_from_coefficient_distribution(|| (rng.next_u32() % 3) as i32 - 1);
    let result = C.get_ring().do_fft(result);
    return result;
}

pub fn enc_sym_zero<R: Rng + CryptoRng, Params: BFVParams>(C: &CiphertextRing<Params>, mut rng: R, sk: &SecretKey<Params>) -> Ciphertext<Params> {
    let a = C.get_ring().sample_uniform(|| rng.next_u64());
    let mut b = C.get_ring().undo_fft(C.negate(C.mul_ref(&a, &sk)));
    let e = C.get_ring().sample_from_coefficient_distribution(|| (rng.sample::<f64, _>(StandardNormal) * 3.2).round() as i32);
    C.get_ring().add_assign_non_fft(&mut b, &e);
    return (CoeffOrNTTRingEl::from_coeff(b), CoeffOrNTTRingEl::from_ntt(a));
}

pub fn enc_sym<R: Rng + CryptoRng, Params: BFVParams>(P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, rng: R, m: &El<PlaintextRing<Params>>, sk: &SecretKey<Params>) -> Ciphertext<Params> {
    hom_add_plain(P, C, m, enc_sym_zero(C, rng, sk))
}

pub fn remove_noise<Params: BFVParams>(P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, c: &El<CiphertextRing<Params>>) -> El<PlaintextRing<Params>> {
    let coefficients = C.wrt_canonical_basis(c);
    let Delta = ZZbig.rounded_div(
        ZZbig.clone_el(C.base_ring().modulus()), 
        &int_cast(*P.base_ring().modulus() as i32, &ZZbig, &StaticRing::<i32>::RING)
    );
    let modulo = P.base_ring().can_hom(&ZZbig).unwrap();
    return P.from_canonical_basis((0..coefficients.len()).map(|i| modulo.map(ZZbig.rounded_div(C.base_ring().smallest_lift(coefficients.at(i)), &Delta))));
}

pub fn dec<Params: BFVParams>(P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, ct: Ciphertext<Params>, sk: &SecretKey<Params>) -> El<PlaintextRing<Params>> {
    let (c0, c1) = ct;
    let noisy_m = C.add(c0.to_ntt(C), C.mul_ref_snd(c1.to_ntt(C), sk));
    return remove_noise::<Params>(P, C, &noisy_m);
}

pub fn dec_println<Params: BFVParams>(P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, ct: &Ciphertext<Params>, sk: &SecretKey<Params>) {
    let m = dec(P, C, clone_ct(C, ct), sk);
    P.println(&m);
    println!();
}

pub fn dec_println_slots<Params: BFVParams>(P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, ct: &Ciphertext<Params>, sk: &SecretKey<Params>) {
    let H = HypercubeIsomorphism::new::<false>(P.get_ring());
    let m = dec(P, C, clone_ct(C, ct), sk);
    for a in H.get_slot_values(&m) {
        H.slot_ring().println(&a);
    }
    println!();
}

pub fn hom_add<Params: BFVParams>(C: &CiphertextRing<Params>, lhs: Ciphertext<Params>, rhs: &Ciphertext<Params>) -> Ciphertext<Params> {
    let (lhs0, lhs1) = lhs;
    let (rhs0, rhs1) = rhs;
    return (CoeffOrNTTRingEl::add(lhs0, rhs0, C), CoeffOrNTTRingEl::add(lhs1, rhs1, C));
}

pub fn hom_sub<Params: BFVParams>(C: &CiphertextRing<Params>, lhs: Ciphertext<Params>, rhs: &Ciphertext<Params>) -> Ciphertext<Params> {
    let (lhs0, lhs1) = lhs;
    let (rhs0, rhs1) = rhs;
    return (CoeffOrNTTRingEl::sub(lhs0, rhs0, C), CoeffOrNTTRingEl::sub(lhs1, rhs1, C));
}

pub fn clone_ct<Params: BFVParams>(C: &CiphertextRing<Params>, ct: &Ciphertext<Params>) -> Ciphertext<Params> {
    return (ct.0.clone(C), ct.1.clone(C));
}

pub fn hom_add_plain<Params: BFVParams>(P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, m: &El<PlaintextRing<Params>>, ct: Ciphertext<Params>) -> Ciphertext<Params> {
    let mut m = C.get_ring().exact_convert_from_nttring(P, m);
    let Delta = C.base_ring().coerce(&ZZbig, ZZbig.rounded_div(
        ZZbig.clone_el(C.base_ring().modulus()), 
        &int_cast(*P.base_ring().modulus() as i32, &ZZbig, &StaticRing::<i32>::RING)
    ));
    C.get_ring().mul_scalar_assign_non_fft(&mut m, &Delta);
    return (CoeffOrNTTRingEl::add(ct.0, &CoeffOrNTTRingEl::from_coeff(m), C), ct.1);
}

pub fn hom_mul_plain<Params: BFVParams>(P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, m: &El<PlaintextRing<Params>>, ct: Ciphertext<Params>) -> Ciphertext<Params> {
    let m = C.get_ring().do_fft(C.get_ring().exact_convert_from_nttring(P, m));
    let c0 = ct.0.to_ntt(C);
    let c1 = ct.1.to_ntt(C);
    return (CoeffOrNTTRingEl::from_ntt(C.mul_ref_snd(c0, &m)), CoeffOrNTTRingEl::from_ntt(C.mul(c1, m)));
}

pub fn hom_mul_plain_i64<Params: BFVParams>(_P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, m: i64, ct: Ciphertext<Params>) -> Ciphertext<Params> {
    (CoeffOrNTTRingEl::mul_i64(ct.0, m, C), CoeffOrNTTRingEl::mul_i64(ct.1, m, C))
}

pub fn noise_budget<Params: BFVParams>(P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, ct: &Ciphertext<Params>, sk: &SecretKey<Params>) -> usize {
    let (c0, c1) = clone_ct(C, ct);
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

pub fn gen_rk<'a, R: Rng + CryptoRng, Params: BFVParams>(C: &'a CiphertextRing<Params>, rng: R, sk: &SecretKey<Params>) -> RelinKey<'a, Params> {
    gen_switch_key::<R, Params>(C, rng, &C.pow(C.clone_el(sk), 2), sk)
}

pub fn hom_mul<Params: BFVParams>(C: &CiphertextRing<Params>, C_mul: &CiphertextRing<Params>, lhs: Ciphertext<Params>, rhs: Ciphertext<Params>, rk: &RelinKey<Params>, conv_data: &MulConversionData) -> Ciphertext<Params> {
    let (c00, c01) = lhs;
    let (c10, c11) = rhs;
    let lift = |c: CoeffEl<Params::NumberRing, Zn, CiphertextAllocator>| 
        C_mul.get_ring().do_fft(C_mul.get_ring().perform_rns_op_from(C.get_ring(), &c, &conv_data.lift_to_C_mul));

    let c00_lifted = lift(c00.to_coeff(C));
    let c01_lifted = lift(c01.to_coeff(C));
    let c10_lifted = lift(c10.to_coeff(C));
    let c11_lifted = lift(c11.to_coeff(C));

    let lifted0 = C_mul.mul_ref(&c00_lifted, &c10_lifted);
    let lifted1 = C_mul.add(C_mul.mul_ref_snd(c00_lifted, &c11_lifted), C_mul.mul_ref_fst(&c01_lifted, c10_lifted));
    let lifted2 = C_mul.mul(c01_lifted, c11_lifted);

    let scale_down = |c: El<CiphertextRing<Params>>| 
        C.get_ring().perform_rns_op_from(C_mul.get_ring(), &C_mul.get_ring().undo_fft(c), &conv_data.scale_down_to_C);

    let res0 = scale_down(lifted0);
    let res1 = scale_down(lifted1);
    let res2 = scale_down(lifted2);

    let op = C.get_ring().to_gadget_product_lhs(res2);
    let (s0, s1) = rk;

    return (CoeffOrNTTRingEl::add(CoeffOrNTTRingEl::from_coeff(res0), &CoeffOrNTTRingEl::gadget_product(&op, s0, C), C), CoeffOrNTTRingEl::add(CoeffOrNTTRingEl::from_coeff(res1), &CoeffOrNTTRingEl::gadget_product(&op, s1, C), C));
}

pub fn gen_switch_key<'a, R: Rng + CryptoRng, Params: BFVParams>(C: &'a CiphertextRing<Params>, mut rng: R, old_sk: &SecretKey<Params>, new_sk: &SecretKey<Params>) -> KeySwitchKey<'a, Params> {
    let old_sk_non_fft = C.get_ring().undo_fft(C.clone_el(old_sk));
    let mut res_0 = C.get_ring().gadget_product_rhs_empty();
    let mut res_1 = C.get_ring().gadget_product_rhs_empty();
    for i in 0..C.get_ring().rns_base().len() {
        let (c0, c1) = enc_sym_zero::<_, Params>(C, &mut rng, new_sk);
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

pub fn key_switch<Params: BFVParams>(C: &CiphertextRing<Params>, ct: Ciphertext<Params>, switch_key: &KeySwitchKey<Params>) -> Ciphertext<Params> {
    let (c0, c1) = ct;
    let (s0, s1) = switch_key;
    let op = C.get_ring().to_gadget_product_lhs(c1.to_coeff(C));
    return (
        CoeffOrNTTRingEl::add(c0, &CoeffOrNTTRingEl::gadget_product(&op, s0, C), C),
        CoeffOrNTTRingEl::gadget_product(&op, s1, C)
    );
}

pub fn mod_switch_to_plaintext<Params: BFVParams>(target: &PlaintextRing<Params>, C: &CiphertextRing<Params>, ct: Ciphertext<Params>, switch_data: &ModSwitchData) -> (El<PlaintextRing<Params>>, El<PlaintextRing<Params>>) {
    let (c0, c1) = ct;
    return (
        C.get_ring().perform_rns_op_to_nttring(target, &c0.to_coeff(C), &switch_data.scale),
        C.get_ring().perform_rns_op_to_nttring(target, &c1.to_coeff(C), &switch_data.scale)
    );
}

pub fn gen_gk<'a, R: Rng + CryptoRng, Params: BFVParams>(C: &'a CiphertextRing<Params>, rng: R, sk: &SecretKey<Params>, g: ZnEl) -> KeySwitchKey<'a, Params> {
    gen_switch_key::<R, Params>(C, rng, &C.get_ring().apply_galois_action(sk, g), sk)
}

pub fn hom_galois<Params: BFVParams>(C: &CiphertextRing<Params>, ct: Ciphertext<Params>, g: ZnEl, gk: &KeySwitchKey<Params>) -> Ciphertext<Params> {
    key_switch(C, (
        CoeffOrNTTRingEl::from_ntt(C.get_ring().apply_galois_action(&ct.0.to_ntt(C), g)),
        CoeffOrNTTRingEl::from_ntt(C.get_ring().apply_galois_action(&ct.1.to_ntt(C), g))
    ), gk)
}

pub fn hom_galois_many<'a, V, Params: BFVParams>(C: &CiphertextRing<Params>, ct: Ciphertext<Params>, gs: &[ZnEl], gks: V) -> Vec<Ciphertext<Params>>
    where V: VectorView<KeySwitchKey<'a, Params>>,
        Params::NumberRing: 'a
{
    let (c0, c1) = ct;
    let c0_ntt = c0.to_ntt(&C);
    let lhs = C.get_ring().to_gadget_product_lhs(c1.to_coeff(&C));
    return (0..gs.len()).map(|i| {
        let c1_g = lhs.apply_galois_action(C.get_ring(), gs[i]);
        let (s0, s1) = gks.at(i);
        let r0 = CoeffOrNTTRingEl::gadget_product(&c1_g, s0, C);
        let r1 = CoeffOrNTTRingEl::gadget_product(&c1_g, s1, C);
        let c0_g = CoeffOrNTTRingEl::from_ntt(C.get_ring().apply_galois_action(&c0_ntt, gs[i]));
        return (CoeffOrNTTRingEl::add(r0, &c0_g, C), r1);
    }).collect();
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
    
    let P = params.create_plaintext_ring(params.t());
    let (C, _C_mul) = params.create_ciphertext_rings();    
    let sk = gen_sk::<_, Pow2BFVParams>(&C, &mut rng);
    let gk = gen_gk::<_, Pow2BFVParams>(&C, &mut rng, &sk, P.get_ring().cyclotomic_index_ring().int_hom().map(3));
    
    let m = P.canonical_gen();
    let ct = enc_sym(&P, &C, &mut rng, &m, &sk);
    let ct_res = hom_galois::<Pow2BFVParams>(&C, ct, P.get_ring().cyclotomic_index_ring().int_hom().map(3), &gk);
    let res = dec(&P, &C, ct_res, &sk);

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
    
    let P = params.create_plaintext_ring(params.t());
    let (C, C_mul) = params.create_ciphertext_rings();

    let sk = gen_sk::<_, Pow2BFVParams>(&C, &mut rng);
    let mul_rescale_data = create_multiplication_rescale::<Pow2BFVParams>(&P, &C, &C_mul);
    let rk = gen_rk::<_, Pow2BFVParams>(&C, &mut rng, &sk);

    let m = P.int_hom().map(2);
    let ct = enc_sym(&P, &C, &mut rng, &m, &sk);

    let m = dec(&P, &C, clone_ct::<Pow2BFVParams>(&C, &ct), &sk);
    assert_el_eq!(&P, &P.int_hom().map(2), &m);

    let ct_sqr = hom_mul(&C, &C_mul, clone_ct(&C, &ct), clone_ct(&C, &ct), &rk, &mul_rescale_data);
    let m_sqr = dec(&P, &C, ct_sqr, &sk);

    assert_el_eq!(&P, &P.int_hom().map(4), &m_sqr);
}

#[test]
fn test_composite_bfv_mul() {
    let mut rng = thread_rng();
    
    let params = CompositeBFVParams {
        t: 8,
        log2_q_min: 700,
        log2_q_max: 800,
        n1: 17,
        n2: 97
    };
    
    let P = params.create_plaintext_ring(params.t());
    let (C, C_mul) = params.create_ciphertext_rings();

    let sk = gen_sk::<_, CompositeBFVParams>(&C, &mut rng);
    let mul_rescale_data = create_multiplication_rescale::<CompositeBFVParams>(&P, &C, &C_mul);
    let rk = gen_rk::<_, CompositeBFVParams>(&C, &mut rng, &sk);

    let m = P.int_hom().map(2);
    let ct = enc_sym(&P, &C, &mut rng, &m, &sk);

    let m = dec(&P, &C, clone_ct::<CompositeBFVParams>(&C, &ct), &sk);
    assert_el_eq!(&P, &P.int_hom().map(2), &m);
    
    let ct_sqr = hom_mul(&C, &C_mul, clone_ct(&C, &ct), clone_ct(&C, &ct), &rk, &mul_rescale_data);
    let m_sqr = dec(&P, &C, ct_sqr, &sk);

    assert_el_eq!(&P, &P.int_hom().map(4), &m_sqr);
}
