# he-ring

Building on [feanor-math](https://crates.io/crates/feanor-math), this library provides efficient implementations of rings that are commonly used in homomorphic encryption (HE).
In particular, this means cyclotomic rings modulo an integer `R_q = Z[X]/(Phi_n(X), q)`.
For both `q`, there are two settings of relevance.
 - `q` is a relatively small integer, used as "plaintext modulus" in schemes and often denoted by `t`. For large `n`, the fastest way to implement arithmetic in these rings is by using a discrete Fourier transform (DFT) over the complex numbers (using floating-point numbers).
 - `q` is a product of moderately large primes that split completely in `R = Z[X]/(Phi_n(X))`. This means that `R_q` has a decomposition into prime fields, where arithmetic operations are performed component-wise, thus very efficiently (this is called "double-RNS-representation"). In this setting, ring elements are usually stored in double-RNS-representation, and only converted back to standard-resp. coefficient-representation when necessary. Such conversions require a number-theoretic transform (NTT) and a implementation of the Chinese Remainder theorem.

Both of these settings are implemented in this library, a general implementation and a specialized one for the case `n = 2^k` is a power-of-two.
In the latter case, the DFTs/NTTs are cheaper, which makes power-of-two cyclotomic rings the most common choice for applications.

Finally, the library also contains an implementation of various fast RNS-conversions.
This refers to algorithms that perform non-arithmetic operations (usually variants of rounding) on the double-RNS-representation, thus avoiding conversions.

## Disclaimer

This library has been designed for research on homomorphic encryption.
I did not have practical considerations (like side-channel resistance) in mind, and advise against using using it in production.

## Example

To demonstrate the use of this library, we give an example implementation of the BFV fully homomorphic encryption scheme.
```rust
use feanor_math::ring::*;
use feanor_math::rings::zn::*;
use feanor_math::rings::zn::zn_64::*;
use feanor_math::primitive_int::StaticRing;
use feanor_math::integer::*;
use feanor_math::rings::poly::dense_poly::DensePolyRing;
use feanor_math::rings::poly::*;
use feanor_math::algorithms::miller_rabin::is_prime;
use feanor_math::mempool::DefaultMemoryProvider;
use feanor_math::algorithms::fft::*;
use feanor_math::{default_memory_provider, assert_el_eq};
use feanor_math::homomorphism::Homomorphism;
use feanor_math::vector::VectorView;
use feanor_math::rings::extension::FreeAlgebraStore;
use feanor_math::vector::vec_fn::VectorFn;
use feanor_math::rings::float_complex::Complex64;

use he_ring::*;

use rand::thread_rng;
use rand::{Rng, CryptoRng};
use rand_distr::StandardNormal;

// in the spirit of feanor-math, all rings are highly generic and extensible using the type system.
// here we define the ring types we will use to implement the scheme. 
pub type PlaintextRing = complexfft::complex_fft_ring::ComplexFFTBasedRing<complexfft::pow2_cyclotomic::Pow2CyclotomicFFT<Zn, cooley_tuckey::FFTTableCooleyTuckey<Complex64>>, DefaultMemoryProvider, DefaultMemoryProvider>;
pub type FFTTable = doublerns::pow2_cyclotomic::Pow2CyclotomicFFT<cooley_tuckey::FFTTableCooleyTuckey<ZnFastmul>>;
pub type CiphertextRing = doublerns::double_rns_ring::DoubleRNSRing<Zn, FFTTable, DefaultMemoryProvider>;

pub type Ciphertext = (El<CiphertextRing>, El<CiphertextRing>);
pub type SecretKey = El<CiphertextRing>;
pub type ExtProdOperand = doublerns::external_product::ExternalProductRhsOperand<Zn, FFTTable, DefaultMemoryProvider>;
pub type KeySwitchKey = (ExtProdOperand, ExtProdOperand);
pub type RelinKey = (ExtProdOperand, ExtProdOperand);

//
// During BFV multiplication, we need a "rescaling operation" that computes `round(x * t / q)`. Doing
// this in a fast-RNS-conversion manner requires precomputing all kinds of data, encapsulated by `MulConversionData`.
//
pub struct MulConversionData {
    to_C_mul: rnsconv::lift::AlmostExactBaseConversion<Vec<Zn>, Vec<Zn>, Zn, Zn, DefaultMemoryProvider, DefaultMemoryProvider>,
    scale_down_to_C: rnsconv::bfv_rescale::AlmostExactRescalingConvert<Vec<Zn>, Zn, Zn, DefaultMemoryProvider, DefaultMemoryProvider>
};

const ZZbig: BigIntRing = BigIntRing::RING;
const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;

pub fn sample_primes_in_arithmetic_progression(n: u64) -> impl Iterator<Item = u64> {
    (1..).map(move |i| i * n + 1).filter(|p| is_prime(&ZZ, &(*p as i64), 8))
}

//
// BFV requires arithmetic in two different "ciphertext rings":
//  - the standard ring `R_q` containing all ciphertexts
//  - an "extended" ciphertext ring `R_Q` used during multiplication (where `q` divides `Q` and `Q > q^2`)
// This function creates both of them
//
pub fn create_ciphertext_rings(log2_ring_degree: usize, ciphertext_moduli_count: usize) -> (CiphertextRing, CiphertextRing) {
    let mut primes = sample_primes_in_arithmetic_progression(2 << log2_ring_degree).map(|p| p as u64);

    let rns_base = zn_rns::Zn::new(primes.by_ref().take(ciphertext_moduli_count).map(Zn::new).collect(), ZZbig ,default_memory_provider!());
    
    let rns_base_mul = zn_rns::Zn::new(
        rns_base.get_ring().iter().map(|R| *R.modulus() as u64).chain(
            primes.take(ciphertext_moduli_count)
        ).map(Zn::new).collect(), 
        ZZbig, 
        default_memory_provider!()
    );

    let C = <CiphertextRing as RingStore>::Type::new(rns_base.clone(), rns_base.get_ring().iter().cloned().map(ZnFastmul::new).collect(), log2_ring_degree,default_memory_provider!());
    let C_mul = <CiphertextRing as RingStore>::Type::new(rns_base_mul.clone(), rns_base_mul.get_ring().iter().cloned().map(ZnFastmul::new).collect(), log2_ring_degree,default_memory_provider!());
    return (C, C_mul);
}

pub fn create_plaintext_ring(log2_ring_degree: usize, plaintext_modulus: i64) -> PlaintextRing {
    return <PlaintextRing as RingStore>::Type::new(Zn::new(plaintext_modulus as u64), log2_ring_degree, default_memory_provider!(), default_memory_provider!());
}

//
// Creates the `MulConversionData` required to perform the rescaling during BFV multiplication in a
// fast-RNS-conversion manner.
//
pub fn create_multiplication_rescale(P: &PlaintextRing, C: &CiphertextRing, C_mul: &CiphertextRing) -> MulConversionData {
    let intermediate = Zn::new(65539);
    MulConversionData {
        to_C_mul: rnsconv::lift::AlmostExactBaseConversion::new(
            C.get_ring().rns_base().iter().map(|R| Zn::new(*R.modulus() as u64)).collect::<Vec<_>>(), 
            C_mul.get_ring().rns_base().iter().map(|R| Zn::new(*R.modulus() as u64)).collect::<Vec<_>>(), 
            intermediate,
            default_memory_provider!(),
            default_memory_provider!()
        ),
        scale_down_to_C: rnsconv::bfv_rescale::AlmostExactRescalingConvert::new(
            C_mul.get_ring().rns_base().iter().map(|R| Zn::new(*R.modulus() as u64)).collect::<Vec<_>>(), 
            Some(P.base_ring()).into_iter().map(|R| Zn::new(*R.modulus() as u64)).collect::<Vec<_>>(), 
            C.get_ring().rns_base().len(), 
            intermediate,
            default_memory_provider!(),
            default_memory_provider!()
        )
    }
}

pub fn gen_sk<R: Rng + CryptoRng>(C: &CiphertextRing, mut rng: R) -> SecretKey {
    // we sample uniform ternary secrets 
    let result = C.get_ring().sample_from_coefficient_distribution(|| (rng.next_u32() % 3) as i32 - 1);
    return result;
}

pub fn enc_sym_zero<R: Rng + CryptoRng>(C: &CiphertextRing, mut rng: R, sk: &SecretKey) -> Ciphertext {
    let a = C.get_ring().sample_uniform(|| rng.next_u64());
    let b = C.mul_ref(&a, &sk);
    let e = C.get_ring().sample_from_coefficient_distribution(|| (rng.sample::<f64, _>(StandardNormal) * 3.2).round() as i32);
    return (C.add(C.negate(b), e), a);
}

pub fn enc_sym<R: Rng + CryptoRng>(P: &PlaintextRing, C: &CiphertextRing, rng: R, m: &El<PlaintextRing>, sk: &SecretKey) -> Ciphertext {
    hom_add_plain(P, C, m, enc_sym_zero(C, rng, sk))
}

pub fn dec(P: &PlaintextRing, C: &CiphertextRing, ct: &Ciphertext, sk: &SecretKey) -> El<PlaintextRing> {
    let (c0, c1) = ct;
    let noisy_m = C.add_ref_fst(c0, C.mul_ref(c1, sk));
    let coefficients = C.wrt_canonical_basis(&noisy_m);
    let Delta = ZZbig.rounded_div(
        ZZbig.clone_el(C.base_ring().modulus()), 
        &int_cast(*P.base_ring().modulus() as i32, &ZZbig, &StaticRing::<i32>::RING)
    );
    let modulo = P.base_ring().can_hom(&ZZbig).unwrap();
    return P.from_canonical_basis((0..coefficients.len()).map(|i| modulo.map(ZZbig.rounded_div(C.base_ring().smallest_lift(coefficients.at(i)), &Delta))));
}

pub fn hom_add(C: &CiphertextRing, lhs: &Ciphertext, rhs: &Ciphertext) -> Ciphertext {
    let (lhs0, lhs1) = lhs;
    let (rhs0, rhs1) = rhs;
    (C.add_ref(lhs0, rhs0), C.add_ref(lhs1, rhs1))
}

pub fn hom_add_plain(P: &PlaintextRing, C: &CiphertextRing, m: &El<PlaintextRing>, ct: Ciphertext) -> Ciphertext {
    let mut m = C.get_ring().do_fft(C.get_ring().exact_convert_from_cfft(P.get_ring(), m));
    let Delta = C.base_ring().coerce(&ZZbig, ZZbig.rounded_div(
        ZZbig.clone_el(C.base_ring().modulus()), 
        &int_cast(*P.base_ring().modulus() as i32, &ZZbig, &StaticRing::<i32>::RING)
    ));
    C.inclusion().mul_assign_map_ref(&mut m, &Delta);
    let (c0, c1) = ct;
    return (C.add(c0, m), c1);

}

pub fn hom_mul_plain(P: &PlaintextRing, C: &CiphertextRing, m: &El<PlaintextRing>, ct: Ciphertext) -> Ciphertext {
    let m = C.get_ring().do_fft(C.get_ring().exact_convert_from_cfft(P.get_ring(), m));
    let (c0, c1) = ct;
    return (C.mul_ref_snd(c0, &m), C.mul(c1, m));
}

pub fn gen_rk<R: Rng + CryptoRng>(C: &CiphertextRing, rng: R, sk: &SecretKey) -> RelinKey {
    gen_switch_key(C, rng, &C.pow(C.clone_el(sk), 2), sk)
}

pub fn hom_mul(C: &CiphertextRing, C_mul: &CiphertextRing, lhs: &Ciphertext, rhs: &Ciphertext, rk: &RelinKey, conv_data: &MulConversionData) -> Ciphertext {
    let (c00, c01) = lhs;
    let (c10, c11) = rhs;
    let lift = |c: El<CiphertextRing>| C_mul.get_ring().do_fft(C_mul.get_ring().perform_rns_op_from(C.get_ring(), &C.get_ring().undo_fft(c), &conv_data.to_C_mul));

    let lifted0 = C_mul.mul(lift(C.clone_el(c00)), lift(C.clone_el(c10)));
    let lifted1 = C_mul.add(C_mul.mul(lift(C.clone_el(c00)), lift(C.clone_el(c11))), C_mul.mul(lift(C.clone_el(c01)), lift(C.clone_el(c10))));
    let lifted2 = C_mul.mul(lift(C.clone_el(c01)), lift(C.clone_el(c11)));

    let scale_down = |c: El<CiphertextRing>| C.get_ring().perform_rns_op_from(C_mul.get_ring(), &C_mul.get_ring().undo_fft(c), &conv_data.scale_down_to_C);

    let res0 = C.get_ring().do_fft(scale_down(lifted0));
    let res1 = C.get_ring().do_fft(scale_down(lifted1));
    let res2 = scale_down(lifted2);
    
    let op = C.get_ring().to_external_product_lhs(res2);
    let (s0, s1) = rk;
    return (
        C.add(res0, C.get_ring().external_product(&op, s0)), 
        C.add(res1, C.get_ring().external_product(&op, s1))
    );
    
}

pub fn gen_switch_key<R: Rng + CryptoRng>(C: &CiphertextRing, mut rng: R, old_sk: &SecretKey, new_sk: &SecretKey) -> KeySwitchKey {
    let mut res_0 = C.get_ring().external_product_rhs_zero();
    let mut res_1 = C.get_ring().external_product_rhs_zero();
    for i in 0..C.get_ring().rns_base().len() {
        let (c0, c1) = enc_sym_zero(C, &mut rng, new_sk);
        let factor = C.base_ring().get_ring().from_congruence((0..C.get_ring().rns_base().len()).map(|i2| {
            let Fp = C.get_ring().rns_base().at(i2);
            if i2 == i { Fp.one() } else { Fp.zero() } 
        }));
        let mut payload = C.clone_el(old_sk);
        C.inclusion().mul_assign_map_ref(&mut payload, &factor);
        res_0.set_rns_factor(i, C.add(payload, c0));
        res_1.set_rns_factor(i, c1);
    }
    return (res_0, res_1);
}

pub fn key_switch(C: &CiphertextRing, ct: &Ciphertext, switch_key: &KeySwitchKey) -> Ciphertext {
    let (c0, c1) = ct;
    let (s0, s1) = switch_key;
    let op = C.get_ring().to_external_product_lhs(C.get_ring().undo_fft(C.clone_el(c1)));
    return (C.add_ref_fst(c0, C.get_ring().external_product(&op, s0)), C.get_ring().external_product(&op, s1));
}

let mut rng = thread_rng();

// not a secure choice of parameters
let log2_ring_degree = 7;
let plaintext_modulus = 3;
let ciphertext_moduli_count = 5;

let P = create_plaintext_ring(log2_ring_degree, plaintext_modulus);
let (C, C_mul) = create_ciphertext_rings(log2_ring_degree, ciphertext_moduli_count);

let sk = gen_sk(&C, &mut rng);

// for simplicity, we encrypt only a scalar
let m = P.int_hom().map(2);
let ct = enc_sym(&P, &C, &mut rng, &m, &sk);

let mul_rescale_data = create_multiplication_rescale(&P, &C, &C_mul);
let relin_key = gen_rk(&C, &mut rng, &sk);
let ct_sqr = hom_mul(&C, &C_mul, &ct, &ct, &relin_key, &mul_rescale_data);

let m_sqr = dec(&P, &C, &ct_sqr, &sk);
assert_el_eq!(&P, &P.int_hom().map(1), &m_sqr);
```