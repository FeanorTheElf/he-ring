# he-ring
# A toolkit library to build Homomorphic Encryption

Building on [feanor-math](https://crates.io/crates/feanor-math), this library provides efficient implementations of various building blocks for Homomorphic Encryption (HE).
The focus is on implementations of the ring `R_q = Z[X]/(Phi_n(X), q)` as required for second-generation HE schemes (like BGV, BFV), but also contains many other components and schemes.

The goal of this library is **not** to provide an easy-to-use implementation of homomorphic encryptions for use in applications - there are many good libraries for that already.
Instead, the goal is to provide a toolkit for researchers that simplifies implementing variants of existing HE schemes, as well as new HE schemes.

## Features

### Provided rings

The number ring `R` is abstractly represented via the trait `HENumberRing`.
However, there are multiple ways of, based on the abstract specification, implement the ring arithmetic.
Three of them are implemented in this library, while one is already present in `feanor-math`.

| Representation of `q` | Multiplication implemented via   |                                      |
|-----------------------|----------------------------------|--------------------------------------|
| Single integer        | Convolution & explicit reduction | `FreeAlgebraImpl` from `feanor_math` |
| Single integer        | Ring factorization               | `NumberRingQuotient`                  |
| RNS basis             | Convolution & explicit reduction | `SingleRNSRing`                      |
| RNS basis             | Ring factorization               | `DoubleRNSRing`                      |

 - "Convolution & explicit reduction" means that two elements of `R_q` are multiplied by first multiplying them as polynomials over `Z_q[X]` and then explicitly performing reduction modulo `Phi_n(X)`
     - elements are stored coefficient-wise, which allows running non-arithmetic operations without requiring prior conversion
     - convolutions can usually be implemented to be quite fast
     - Galois operations are usually quite expensive
 - "Ring factorization" means that two elements of `R_q` are multiplied by computing their image under the isomorphism `R_q -> Z_q x ... x Z_q`, after which multiplication can be performed pointwise.
   However, for this isomorphism to exist, it is usually required to impose additional constraints on `q` resp. its factors (or switch to a different `q` before the multiplication, and switch back afterwards).
     - if `q` resp. its factors satisfy the constraints for the isomorphism to exist, then elements can be kept in decomposed form, leading to very fast arithmetic operations
     - mapping elements to their decomposed form and back is usually slower than computing a convolution, with the notable exception of power-of-two cyclotomics `R = Z[X]/(X^n + 1)`, for which this always beats standard convolutions
     - non-arithmetic operations usually require mapping elements back into coefficient form
     - Galois operations can usually be computed in linear time in decomposed form

Note that in most HE-related situations, you will want the "Ring factorization"-based methods, i.e. use `DoubleRNSRing` for the ciphertext ring and `NumberRingQuotient` for the plaintext ring.

This code is contained in the modules [`crate::ntt`] and [`crate::rings`].
In detail, the following is available:
 - The above implementations of rings
 - Description of the ring factorization `R/(p^e) = GR(p, e, d)^l` into copies of Galois rings via a "hypercube structure", as introduced by [Bootstrapping for HElib](https://ia.cr/2014/873).
 - Implementation of power-of-two cyclotomic number rings and odd cyclotomic number rings
 - A `ManagedDoubleRNSRing`, which is uses a `DoubleRNSRing` for arithmetic operations, but automatically switches between coefficient and double-RNS representation as needed
 - Efficient implementations of "gadget products", which are used in HE to limit the noise growth when multiplying noisy ring elements

### Provided RNS operations

In order to perform non-arithmetic operations (e.g. rounding, reduction modulo `t`, ...) on `R_q` when `q = p1 ... pr` is represented as an RNS basis, one has to use algorithms specifically designed for this setting.
Such algorithms are provided in `rnsconv` as implementations of the trait `RNSOperation`.

This code is contained in the modules [`crate::rnsconv`].
In detail, the following is available:
 - `AlmostExactBaseConversion` and `AlmostExactSharedBaseConversion` to compute the map `Z_q -> Z_q',  x -> lift(x) mod q'` in RNS form
 - `AlmostExactRescalingConvert` to compute the map `Z_q -> Z_q',  x -> round(a lift(x) / b) mod q'` (as required during BFV) in RNS form
 - `CongruencePreservingRescaling` to compute the special variant of rescaling that is needed during modulus-switching in the BGV scheme

### Homomorphic Encryption schemes

This library currently contains an implementation of the BFV scheme, and I hope to soon add an implementation of the CLPX scheme for large integers.
Note that using the well-abstracted building blocks mentioned before, these implementations are actually a straightforward adaption of the mathematical description of the schemes, so it should be very simple to implement another/custom HE scheme.

This code is contained in the modules [`crate::bfv`].

### Arithmetization and Bootstrapping

We include an implementation of bootstrapping for BFV.
Again, while the actual implementation is of course scheme-specific, components are, where possible, designed in a generic way.
In particular, this includes tools for arithmetization and computing arithmetic circuits.

This code is contained in the modules [`crate::bfv::bootstrap`], [`crate::lintransform`] and [`crate::digitextract`].
This in particular contains:
 - Methods for computing linear transforms and representing them as sum of Galois automorphisms (including HElib-style `matmul1d()` and `blockmatmul1d()`)
 - Low-depth Paterson-Stockmeyer to convert polynomials into arithmetic circuits
 - Table of precomputed, optimal digit extraction polynomials for `p = 2`
 - General operations like the algebraic trace, or a "broadcast" across slots

## Performance

When optimizing for performance, please use the Intel HEXL library (by enabling the feature `use_hexl` and providing a build of HEXL, as described in more detail in the documentation of [`feanor-math-hexl`](https://github.com/FeanorTheElf/feanor-math-hexl)), since the default NTT does not provide SOTA performance. Also note that `he-ring` is currently single-threaded.

Note that while this library is already quite optimized, it may not be fully competitive with other HE libraries that have existed for longer and thus received more optimization effort.
Also, our goal of providing a modular toolkit of building blocks makes some kinds of optimizations more difficult, since components cannot always make as many assumptions on the input as they could if they only support a single HE scheme.

## Disclaimer

This library has been designed for research on homomorphic encryption.
I did not have practical considerations (like side-channel resistance) in mind, and advise against using using it in production.
