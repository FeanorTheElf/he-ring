# he-ring

Building on [feanor-math](https://crates.io/crates/feanor-math), this library provides efficient implementations of rings that are commonly used in homomorphic encryption (HE).
Our focus lies on providing the building blocks for second-generation HE schemes like BGV or BFV, however most building blocks are also used in other schemes like FHEW/TFHE.
In particular, the core component are cyclotomic rings modulo an integer `R_q = Z[X]/(Phi_n(X), q)`.

## Features

### Provided rings

The number ring `R` is abstractly represented via the trait `HENumberRing`.
However, there are multiple ways of, based on the abstract specification, implement the ring arithmetic.
Three of them are implemented in this library, while one is already present in `feanor-math`.

| Representation of `q` | Computation of convolution       |                                      |
|-----------------------|----------------------------------|--------------------------------------|
| Single integer        | Convolution & explicit reduction | `FreeAlgebraImpl` from `feanor_math` |
| Single integer        | Ring factorization               | `DecompositionRing`                  |
| RNS basis             | Convolution & explicit reduction | `SingleRNSRing` *                    |
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

Note that in most HE-related situations, you will want the "Ring factorization"-based methods, i.e. use `DoubleRNSRing` for the ciphertext ring and `DecompositionRing` for the plaintext ring.

* I am considering removing this implementation again, since it does not seem to never have any performance benefits over `DoubleRNSRing`...

### Provided RNS operations

In order to perform non-arithmetic operations (e.g. rounding, reduction modulo `t`, ...) on `R_q` when `q = p1 ... pr` is represented as an RNS basis, one has to use algorithms specifically designed for this setting.
Such algorithms are provided in `rnsconv` as implementations of the trait `RNSOperation`.

### Bootstrapping operations

TODO

## Disclaimer

This library has been designed for research on homomorphic encryption.
I did not have practical considerations (like side-channel resistance) in mind, and advise against using using it in production.
