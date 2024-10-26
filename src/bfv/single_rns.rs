use feanor_math::algorithms::convolution::fft::FFTRNSBasedConvolutionZn;
use feanor_math::rings::zn::*;
use feanor_math::ring::*;
use feanor_math::rings::zn::zn_64::Zn;

use crate::cyclotomic::CyclotomicRingStore;
use crate::rings::gadget_product::single_rns::*;
use crate::rings::single_rns_ring::*;

use super::*;

#[derive(Clone, Debug)]
pub struct CompositeSingleRNSBFVParams {
    pub t: i64,
    pub log2_q_min: usize,
    pub log2_q_max: usize,
    pub n1: usize,
    pub n2: usize
}

impl BFVParams for CompositeSingleRNSBFVParams {

    type NumberRing = CompositeCyclotomicDecomposableNumberRing;
    type CiphertextRing = SingleRNSRing<Self::NumberRing, Zn, CiphertextAllocator>;
    type CiphertextRingBase = SingleRNSRingBase<Self::NumberRing, Zn, CiphertextAllocator>;
    type Ciphertext = (El<Self::CiphertextRing>, El<Self::CiphertextRing>);
    type KeySwitchKey<'a> = (GadgetProductRhsOperand<'a, Self::NumberRing, CiphertextAllocator>, GadgetProductRhsOperand<'a, Self::NumberRing, CiphertextAllocator>)
        where Self: 'a;

    fn ciphertext_modulus_bits(&self) -> Range<usize> {
        self.log2_q_min..self.log2_q_max
    }

    fn number_ring(&self) -> Self::NumberRing {
        CompositeCyclotomicDecomposableNumberRing::new(self.n1, self.n2)
    }

    fn plaintext_modulus(&self) -> i64 {
        self.t
    }

    fn create_plaintext_ring(&self, modulus: i64) -> PlaintextRing<Self> {
        DecompositionRingBase::new(self.number_ring(), Zn::new(modulus as u64))
    }

    fn create_ciphertext_rings(&self) -> (CiphertextRing<Self>, CiphertextRing<Self>) {
        let log2_q = self.ciphertext_modulus_bits();
        let number_ring = self.number_ring();

        let mut C_rns_base = sample_primes(log2_q.start, log2_q.end, 56, |bound| <_ as DecomposableNumberRing<Zn>>::largest_suitable_prime(&number_ring, int_cast(bound, ZZ, ZZbig)).map(|p| int_cast(p, ZZbig, ZZ))).unwrap();
        C_rns_base.sort_unstable_by(|l, r| ZZbig.cmp(l, r));

        let mut Cmul_rns_base = extend_sampled_primes(&C_rns_base, log2_q.end * 2, log2_q.end * 2 + 57, 57, |bound| <_ as DecomposableNumberRing<Zn>>::largest_suitable_prime(&number_ring, int_cast(bound, ZZ, ZZbig)).map(|p| int_cast(p, ZZbig, ZZ))).unwrap();
        assert!(ZZbig.is_gt(&Cmul_rns_base[Cmul_rns_base.len() - 1], &C_rns_base[C_rns_base.len() - 1]));
        Cmul_rns_base.sort_unstable_by(|l, r| ZZbig.cmp(l, r));

        let C = SingleRNSRingBase::new(
            self.number_ring(),
            zn_rns::Zn::new(C_rns_base.iter().map(|p| Zn::new(int_cast(ZZbig.clone_el(p), ZZ, ZZbig) as u64)).collect(), ZZbig)
        );
        let Cmul = SingleRNSRingBase::new(
            number_ring,
            zn_rns::Zn::new(Cmul_rns_base.iter().map(|p| Zn::new(int_cast(ZZbig.clone_el(p), ZZ, ZZbig) as u64)).collect(), ZZbig)
        );
        return (C, Cmul);
    }

    fn gen_sk<R: Rng + CryptoRng>(C: &CiphertextRing<Self>, mut rng: R) -> SecretKey<Self> {
        // we sample uniform ternary secrets 
        C.get_ring().sample_from_coefficient_distribution(|| (rng.next_u32() % 3) as i32 - 1)
    }

    fn enc_sym_zero<R: Rng + CryptoRng>(C: &CiphertextRing<Self>, mut rng: R, sk: &SecretKey<Self>) -> Ciphertext<Self> {
        let a = C.get_ring().sample_uniform(|| rng.next_u64());
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
        return (C.add_ref_snd(lhs0, rhs0), C.add_ref_snd(lhs1, rhs1));
    }
    
    fn hom_sub(C: &CiphertextRing<Self>, lhs: Ciphertext<Self>, rhs: &Ciphertext<Self>) -> Ciphertext<Self> {let (lhs0, lhs1) = lhs;
        let (rhs0, rhs1) = rhs;
        return (C.sub_ref_snd(lhs0, rhs0), C.sub_ref_snd(lhs1, rhs1));
    }
    
    fn clone_ct(C: &CiphertextRing<Self>, ct: &Ciphertext<Self>) -> Ciphertext<Self> {
        (C.clone_el(&ct.0), C.clone_el(&ct.1))
    }
    
    fn hom_add_plain(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: &El<PlaintextRing<Self>>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        let mut m = C.get_ring().exact_convert_from_nttring(P, m);
        let Delta = C.base_ring().coerce(&ZZbig, ZZbig.rounded_div(
            ZZbig.clone_el(C.base_ring().modulus()), 
            &int_cast(*P.base_ring().modulus() as i32, &ZZbig, &StaticRing::<i32>::RING)
        ));
        C.inclusion().mul_assign_map(&mut m, Delta);
        return (C.add(ct.0, m), ct.1);
    }
    
    fn hom_mul_plain(P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: &El<PlaintextRing<Self>>, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        let m = C.get_ring().exact_convert_from_nttring(P, m);
        let (c0, c1) = ct;
        return (C.mul_ref_snd(c0, &m), C.mul(c1, m));
    }
    
    fn hom_mul_plain_i64(_P: &PlaintextRing<Self>, C: &CiphertextRing<Self>, m: i64, ct: Ciphertext<Self>) -> Ciphertext<Self> {
        let inclusion = C.inclusion().compose(C.base_ring().can_hom(&StaticRing::<i64>::RING).unwrap());
        (inclusion.mul_map(ct.0, m), inclusion.mul_map(ct.1, m))
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
        let lift = |c: SingleRNSRingEl<Self::NumberRing, Zn, CiphertextAllocator>| 
            C_mul.get_ring().perform_rns_op_from(C.get_ring(), &c, &conv_data.lift_to_C_mul);
    
        let c00_lifted = lift(c00);
        let c01_lifted = lift(c01);
        let c10_lifted = lift(c10);
        let c11_lifted = lift(c11);
    
        let lifted0 = C_mul.mul_ref(&c00_lifted, &c10_lifted);
        let lifted1 = C_mul.add(C_mul.mul_ref_snd(c00_lifted, &c11_lifted), C_mul.mul_ref_fst(&c01_lifted, c10_lifted));
        let lifted2 = C_mul.mul(c01_lifted, c11_lifted);
    
        let scale_down = |c: El<CiphertextRing<Self>>| 
            C.get_ring().perform_rns_op_from(C_mul.get_ring(), &c, &conv_data.scale_down_to_C);
    
        let res0 = scale_down(lifted0);
        let res1 = scale_down(lifted1);
        let res2 = scale_down(lifted2);
    
        let op = C.get_ring().to_gadget_product_lhs(res2);
        let (s0, s1) = rk;
    
        return (C.add(res0, C.get_ring().gadget_product(&op, s0)), C.add(res1, C.get_ring().gadget_product(&op, s1)));
    }
    
    fn gen_switch_key<'a, R: Rng + CryptoRng>(C: &'a CiphertextRing<Self>, mut rng: R, old_sk: &SecretKey<Self>, new_sk: &SecretKey<Self>) -> KeySwitchKey<'a, Self>
        where Self: 'a
    {
        let mut res_0 = C.get_ring().gadget_product_rhs_empty();
        let mut res_1 = C.get_ring().gadget_product_rhs_empty();
        for i in 0..C.get_ring().rns_base().len() {
            let (c0, c1) = Self::enc_sym_zero(C, &mut rng, new_sk);
            let factor = C.base_ring().get_ring().from_congruence((0..C.get_ring().rns_base().len()).map(|i2| {
                let Fp = C.get_ring().rns_base().at(i2);
                if i2 == i { Fp.one() } else { Fp.zero() } 
            }));
            let mut payload = C.clone_el(old_sk);
            C.inclusion().mul_assign_map(&mut payload, factor);
            C.add_assign(&mut payload, c0);
            res_0.set_rns_factor(i, payload);
            res_1.set_rns_factor(i, c1);
        }
        return (res_0, res_1);
    }
    
    fn key_switch<'a>(C: &CiphertextRing<Self>, ct: Ciphertext<Self>, switch_key: &KeySwitchKey<'a, Self>) -> Ciphertext<Self>
        where Self: 'a
    {
        let (c0, c1) = ct;
        let (s0, s1) = switch_key;
        let op = C.get_ring().to_gadget_product_lhs(c1);
        return (
            C.add(c0, C.get_ring().gadget_product(&op, s0)),
            C.get_ring().gadget_product(&op, s1)
        );
    }
    
    fn mod_switch_to_plaintext(target: &PlaintextRing<Self>, C: &CiphertextRing<Self>, ct: Ciphertext<Self>, switch_data: &ModSwitchData) -> (El<PlaintextRing<Self>>, El<PlaintextRing<Self>>) {
        let (c0, c1) = ct;
        return (
            C.get_ring().perform_rns_op_to_nttring(target, &c0, &switch_data.scale),
            C.get_ring().perform_rns_op_to_nttring(target, &c1, &switch_data.scale)
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
        let mut result = Vec::new();
        for ((ct0, ct1), gk) in C.apply_galois_action_many(&ct.0, gs).zip(C.apply_galois_action_many(&ct.1, gs)).zip(gks.iter()) {
            result.push(Self::key_switch(C, (ct0, ct1), gk));
        }
        return result;
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