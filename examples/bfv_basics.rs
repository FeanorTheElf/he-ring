// he-ring completely relies on unstable Rust features
#![feature(allocator_api)]
#![allow(non_snake_case)]

use std::{alloc::Global, marker::PhantomData};

use feanor_math::assert_el_eq;
use feanor_math::homomorphism::Homomorphism;
use feanor_math::integer::{BigIntRing, IntegerRingStore};
use feanor_math::ring::{RingExtensionStore, RingStore};
use feanor_math::rings::extension::FreeAlgebraStore;
use feanor_math::rings::zn::ZnRingStore;
use he_ring::bfv::{BFVParams, CiphertextRing, PlaintextRing, Pow2BFV};
use he_ring::cyclotomic::CyclotomicRingStore;
use he_ring::DefaultNegacyclicNTT;
use rand::thread_rng;

fn main() {

    // BFV parameters are modelled as a trait with multiple implementations, since this
    // allows each implementation to choose their own type for the BFV ciphertext ring;
    // for this example, we settle on the power-of-two case.
    type ChosenBFVParamType = Pow2BFV;
    
    // next, we choose the parameters; these contain the parameters used for the ciphertext
    // ring, but not the plaintext modulus. The rationale is that a BFV ciphertext usually
    // is a valid encryption (of different messages) w.r.t. multiple plaintext moduli
    let params = ChosenBFVParamType {
        // we can choose a custom allocator to allocate ciphertext ring elements; for now
        // just use the `Global` allocator
        ciphertext_allocator: Global,
        // `N` is the degree of the ring; since this must be a power-of-two, it is represented 
        // by it binary logarithm
        log2_N: 12,
        // the size of the ciphertext modulus is given as a range, since it must be the product
        // of multiple primes with certain properties, and we cannot always sample those with
        // an exactly fixed size;
        // according to the HE standard (https://homomorphicencryption.org/wp-content/uploads/2018/11/HomomorphicEncryptionStandardv1.1.pdf),
        // choosing `q` to have 110 bits gives 128 bits of security (for `N = 2^12` and noise standard deviation 3.2)
        log2_q_min: 100,
        log2_q_max: 110,
        // to compute the ring arithmetic, we need an NTT implementation; multiple instance of
        // this NTT will be created for different moduli, so here only its type is relevant (this
        // is why we only have a `PhantomData` here);
        // note that `DefaultNegacyclicNTT` points to either the somewhat slow native implementation,
        // or the NTT given by HEXL if the he-ring feature `use_hexl` is enabled
        negacyclic_ntt: PhantomData::<DefaultNegacyclicNTT>
    };

    // instead of creating an opaque "context" object as in other HE libraries, in he-ring the user
    // has to be somewhat aware of the internal structure of BFV. In particular:
    //  - ciphertexts consist of two elements of `R_q = Z[X]/(Phi_n(X), q)`, which we call "ciphertext ring"
    //  - plaintexts are elements of `R_t = Z[X]/(Phi_n(X), t)` where `t` is the "plaintext modulus"
    //  - for multiplication, we also require the ring `R_q'` for a `q' >> q` (more on that later)
    //  - the secret key is an element of `R_q`
    //  - to perform homomorphic multiplication, we require relinearization keys
    // hence, let's create all those components - note that their type depends on the type of the BFV parameters
    // we use
    let (C, C_for_multiplication): (CiphertextRing<ChosenBFVParamType>, CiphertextRing<ChosenBFVParamType>) = params.create_ciphertext_rings();

    // lower-case `n` is always used for the multiplicative order of the root of unity; in the power-of-two case,
    // we write capital `N` for the degree of `R`, so `N = phi(n)`
    println!("N        = {}", C.rank());
    println!("n        = {}", C.n());
    println!("log2(q)  = {}", BigIntRing::RING.abs_log2_ceil(C.base_ring().modulus()).unwrap());
    println!("log2(q') = {}", BigIntRing::RING.abs_log2_ceil(C_for_multiplication.base_ring().modulus()).unwrap());

    // since the plaintext modulus is not part of the parameters, we supply it separately
    let t = 17;
    let P: PlaintextRing<ChosenBFVParamType> = params.create_plaintext_ring(t);

    // from now on, we actually don't need the parameters anymore - the information is completely encoded in
    // `C`, `_C_for_multiplication` and `P`; in particular, it is possible to create these rings manually, or
    // reused them between different BFV instantiations, the functions `create_ciphertext_rings()` and
    // `create_plaintext_ring()` are only for convenience

    let mut rng = thread_rng();

    // create the secret key
    let sk = ChosenBFVParamType::gen_sk(&C, &mut rng);

    // for relinearization, we require a relinearization key; internally, it consists of multiple encryptions of `sk^2`
    // under `sk`. The number of these encryptions is `digits` (because it is equal to the number of "digits" used during
    // the gadget product); a higher number will lead to slower performance but lower noise growth - note however that
    // the noise growth is additive, hence will have a large effect on fresh, low-noise ciphertexts, but basically no
    // effect on medium noise ciphertexts, even for "small" values of `digits`;
    // It does not make sense for `digits` to be larger than the number of RNS factors in `q`
    let digits = 2;
    let rk = ChosenBFVParamType::gen_rk(&C, &mut rng, &sk, digits);

    // Let's create an example plaintext. The function `from_canonical_basis()` computes the ring element in
    // `Z[X]/(Phi_n(X), t)` that has the given list as coefficients w.r.t. the basis `1, X, X^2, ..., X^(phi(n) - 1)`;
    // in other words, these are just the coefficients of the polynomial
    let x = P.from_canonical_basis((0..(1 << 12)).map(|i| 
        // one of the easiest ways to get an element in any ring - here the "base ring" `Z/(t)` of `R_t` - is to use
        // `int_hom()`, which creates the homomorphism that maps integers (more concretely `i32`) into the ring;
        // note that we could also write `x = P.int_hom().map(42)` if we only wanted to encode a single scalar in `R_t`
        P.base_ring().int_hom().map(i)
    ));

    // symmetrically encrypt `x`
    let enc_x = ChosenBFVParamType::enc_sym(&P, &C, &mut rng, &x, &sk);
    

    // perform homomorphic multiplication;
    // here we require `C_for_multiplication`, since BFV homomorphic multiplication internally requires arithmetic
    // that does not wrap around `q`. `create_ciphertext_rings()` will choose the modulus `q'` large enough that
    // there is no wrap-around when doing this arithmetic in `C_for_multiplication`
    let enc_x_sqr = ChosenBFVParamType::hom_mul(&P, &C, &C_for_multiplication, ChosenBFVParamType::clone_ct(&C, &enc_x), enc_x, &rk);

    // decrypt and check the result
    let result = ChosenBFVParamType::dec(&P, &C, enc_x_sqr, &sk);
    assert_el_eq!(&P, P.pow(P.clone_el(&x), 2), result);
    println!("done");

    // uncommenting this will print quite q long response, since (when written as a polynomial), `result` has 4096 coefficients
    // println!("{}", P.format(&result));
}