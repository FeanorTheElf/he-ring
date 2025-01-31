# he-ring, a toolkit library to build Homomorphic Encryption

Building on [feanor-math](https://crates.io/crates/feanor-math), this library provides efficient implementations of various building blocks for Homomorphic Encryption (HE).
The focus is on implementations of the ring `R_q = Z[X]/(Phi_n(X), q)` as required for second-generation HE schemes (like BGV, BFV), but also contains many other components and schemes.

The goal of this library is **not** to provide an easy-to-use implementation of homomorphic encryptions for use in applications - there are many good libraries for that already.
Instead, the goal is to provide a toolkit for researchers that simplifies implementing variants of existing HE schemes, as well as new HE schemes.

## Features

In short, he-ring contains the following:
 - multiple efficient implementations of arithmetic in the ring `R_q`, which provide different performance characteristics (supporting arbitrary `n`)
 - an implementation of the isomorphism `R/(p^e) = GR(p, e, d) x ... x GR(p, e, d)` via "hypercube structures" (compare "Bootstrapping for HElib" by Halevi and Shoup, <https://ia.cr/2014/873>)
 - an implementation of "gadget products", i.e. the certain kind of inner product that is used in HE schemes to multiply ciphertexts with lower noise growth
 - implementations of the BFV and BGV encryption schemes
 - bootstrapping for BFV
 - tools for arithmetization, including modelling of arithmetic circuits, polynomial-to-circuit conversion via Paterson-Stockmeyer and HElib-style linear transforms

## Performance

When optimizing for performance, please use the Intel HEXL library (by enabling the feature `use_hexl` and providing a build of HEXL, as described in more detail in the documentation of [`feanor-math-hexl`](https://github.com/FeanorTheElf/feanor-math-hexl)), since the default NTT does not provide SOTA performance. Also note that `he-ring` is currently single-threaded.

Note that while this library is already quite optimized, it may not be fully competitive with other HE libraries that have existed for longer and thus received more optimization effort.
Also, our goal of providing a modular toolkit of building blocks makes some kinds of optimizations more difficult, since components cannot always make as many assumptions on the input as they could if they only support a single HE scheme.

### Profiling

`he-ring` is instrumented using the framework defined by the Rust library [`tracing`](https://crates.io/crates/tracing).
Hence, running any `he-ring` functions with an active tracing subscriber will generate corresponding tracing events that the subscriber can use for profiling purposes.
There are various crates that implement tracing subscribers with profiling functionality.

For tests within this crate, we use [`tracing-chrome`](https://crates.io/crates/tracing-chrome) which generates Perfetto json trace files (can be displayed by Google Chrome without requiring plugins).
In particular, if you enable ignored tests and run one of the  `measure_time_`-prefixed test in this crate, this will generate a trace file.
Of course, this is only included on test builds, in library builds, the parent application is free to configure `tracing` as desired.

## Disclaimer

This library has been designed for research on homomorphic encryption.
I did not have practical considerations (like side-channel resistance) in mind, and advise against using using it in production.
