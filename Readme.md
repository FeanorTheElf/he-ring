# he-ring

Building on [feanor-math](https://crates.io/crates/feanor-math), this library provides efficient implementations of rings that are commonly used in homomorphic encryption (HE).
Our focus lies on providing the building blocks for second-generation HE schemes like BGV or BFV, however most building blocks are also used in other schemes like FHEW/TFHE.
In particular, the core component are cyclotomic rings modulo an integer `R_q = Z[X]/(Phi_n(X), q)`.
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
