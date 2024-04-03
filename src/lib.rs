#![feature(never_type)]
#![feature(fn_traits)]
#![feature(unboxed_closures)]
#![feature(test)]
#![feature(const_type_name)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

#![doc = include_str!("../Readme.md")]

#[cfg(test)]
macro_rules! ring_literal {
    ($ring:expr, $iter:expr) => {
        {
            let ring = $ring;
            let base_ring = ring.base_ring();
            <_ as feanor_math::rings::extension::FreeAlgebraStore>::from_canonical_basis(&ring, $iter.into_iter().map(|x| <_ as feanor_math::homomorphism::Homomorphism<_, _>>::map(&base_ring.int_hom(), x)))
        }
    };
}

extern crate feanor_math;
extern crate rand;
extern crate test;

#[macro_use]
pub mod profiling;

///
/// Defines the trait [`cyclotomic::CyclotomicRing`] for rings of the form `R[X]/(Phi_n)`, where `R` is any base ring.
/// 
pub mod cyclotomic;

///
/// Implementation of rings using complex-valued fast fourier transforms for efficient arithmetic.
/// 
pub mod complexfft;
pub mod doublerns;
pub mod rnsconv;