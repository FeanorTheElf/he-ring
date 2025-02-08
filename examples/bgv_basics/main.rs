// he-ring completely relies on unstable Rust features
#![feature(allocator_api)]
#![allow(non_snake_case)]

// For a guided explanation of this example, see the doc
#![doc = include_str!("Readme.md")]

use he_ring::bgv::{BGVParams, CiphertextRing, PlaintextRing, Pow2BGV};
use he_ring::cyclotomic::CyclotomicRingStore;
use he_ring::DefaultNegacyclicNTT;
use he_ring::ciphertext_ring::BGFVCiphertextRing;
use he_ring::gadget_product::recommended_rns_factors_to_drop;
use rand::{SeedableRng, rngs::StdRng};
use std::alloc::Global;
use std::marker::PhantomData;
use feanor_math::integer::*;
use feanor_math::primitive_int::*;
use feanor_math::ring::*;
use feanor_math::rings::zn::ZnRingStore;
use feanor_math::ring::RingStore;
use feanor_math::algorithms::eea::signed_gcd;
use feanor_math::homomorphism::Homomorphism;
use feanor_math::rings::extension::FreeAlgebraStore;
use feanor_math::seq::VectorView;
use feanor_math::assert_el_eq;

fn main() {

    type ChosenBGVParamType = Pow2BGV;
    let params = ChosenBGVParamType {
        ciphertext_allocator: Global,
        log2_N: 13,
        log2_q_min: 210,
        log2_q_max: 220,
        negacyclic_ntt: PhantomData::<DefaultNegacyclicNTT>
    };

    let C_initial: CiphertextRing<ChosenBGVParamType> = params.create_initial_ciphertext_ring();

    let plaintext_modulus = 17;
    let P: PlaintextRing<ChosenBGVParamType> = params.create_plaintext_ring(plaintext_modulus);
    assert!(BigIntRing::RING.is_one(&signed_gcd(BigIntRing::RING.clone_el(C_initial.base_ring().modulus()), int_cast(plaintext_modulus, BigIntRing::RING, StaticRing::<i64>::RING), BigIntRing::RING)));

    println!("N        = {}", C_initial.rank());
    println!("n        = {}", C_initial.n());
    println!("log2(q)  = {}", BigIntRing::RING.abs_log2_ceil(C_initial.base_ring().modulus()).unwrap());

    let mut rng = StdRng::from_seed([1; 32]);
    let sk = ChosenBGVParamType::gen_sk(&C_initial, &mut rng);

    let digits = 2;
    let rk = ChosenBGVParamType::gen_rk(&P, &C_initial, &mut rng, &sk, digits);

    let x = P.from_canonical_basis((0..(1 << 13)).map(|i| 
        P.base_ring().int_hom().map(i)
    ));
    let enc_x = ChosenBGVParamType::enc_sym(&P, &C_initial, &mut rng, &x, &sk);

    let enc_x_sqr = ChosenBGVParamType::hom_mul(&P, &C_initial, ChosenBGVParamType::clone_ct(&P, &C_initial, &enc_x), enc_x, &rk);
    
    let num_digits_to_drop = 1;
    let digits_to_drop_indices = recommended_rns_factors_to_drop(C_initial.base_ring().len(), rk.0.gadget_vector_moduli_indices(), num_digits_to_drop);
    let C_new = RingValue::from(C_initial.get_ring().drop_rns_factor(&digits_to_drop_indices));
    
    println!("log2(q') = {}", BigIntRing::RING.abs_log2_ceil(C_new.base_ring().modulus()).unwrap());
    
    let enc_x_modswitch = ChosenBGVParamType::mod_switch(&P, &C_new, &C_initial, &digits_to_drop_indices, enc_x_sqr);
    let sk_modswitch = ChosenBGVParamType::mod_switch_sk(&P, &C_new, &C_initial, &digits_to_drop_indices, &sk);
    let rk_modswitch = ChosenBGVParamType::mod_switch_rk(&P, &C_new, &C_initial, &digits_to_drop_indices, &rk);
    
    let enc_x_pow4 = ChosenBGVParamType::hom_mul(&P, &C_new, ChosenBGVParamType::clone_ct(&P, &C_initial, &enc_x_modswitch), enc_x_modswitch, &rk_modswitch);
    assert_eq!(22, ChosenBGVParamType::noise_budget(&P, &C_new, &enc_x_pow4, &sk_modswitch));
    let dec_x_pow4 = ChosenBGVParamType::dec(&P, &C_new, enc_x_pow4, &sk_modswitch);
    assert_el_eq!(&P, P.pow(P.clone_el(&x), 4), &dec_x_pow4);
    println!("done");
}