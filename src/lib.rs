#![feature(never_type)]
#![feature(fn_traits)]
#![feature(unboxed_closures)]
#![feature(test)]
#![feature(const_type_name)]
#![feature(allocator_api)]
#![feature(ptr_alignment_type)]
#![feature(associated_type_defaults)]
#![feature(generic_arg_infer)]
#![feature(min_specialization)]
#![feature(array_chunks)]
#![feature(mapped_lock_guards)]

#![allow(non_snake_case)]
#![allow(type_alias_bounds)]
#![allow(unused_imports)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(unused_variables)]

#![doc = include_str!("../Readme.md")]

use std::alloc::Global;
use std::time::Instant;

use feanor_math::algorithms::miller_rabin::is_prime;
use feanor_math::integer::BigIntRing;
use feanor_math::integer::BigIntRingBase;
use feanor_math::integer::IntegerRingStore;
use feanor_math::ordered::OrderedRingStore;
use feanor_math::pid::EuclideanRingStore;
use feanor_math::primitive_int::StaticRing;
use feanor_math::primitive_int::StaticRingBase;
use feanor_math::ring::*;
use feanor_math::homomorphism::*;
use feanor_math::rings::extension::galois_field::GaloisField;
use feanor_math::rings::field::AsFieldBase;
use feanor_math::rings::local::AsLocalPIRBase;
use feanor_math::rings::zn::zn_64;
use feanor_math::rings::zn::zn_64::Zn;
use feanor_math::rings::zn::{FromModulusCreateableZnRing, ZnRing};
use feanor_math::serialization::SerializableElementRing;
use number_ring::pow2_cyclotomic::RustNegacyclicNTT;

#[macro_export]
macro_rules! ring_literal {
    ($ring:expr, $iter:expr) => {
        {
            let ring = $ring;
            let base_ring = ring.base_ring();
            <_ as feanor_math::rings::extension::FreeAlgebraStore>::from_canonical_basis(&ring, $iter.into_iter().map(|x| <_ as feanor_math::homomorphism::Homomorphism<_, _>>::map(&base_ring.int_hom(), x as i32)))
        }
    };
}

extern crate feanor_math;
#[cfg(feature = "use_hexl")]
extern crate feanor_math_hexl;
extern crate test;
extern crate thread_local;
extern crate serde;
extern crate rand;
extern crate rand_distr;

#[cfg(feature = "use_hexl")]
pub type DefaultConvolution = feanor_math_hexl::conv::HEXLConvolution;
#[cfg(not(feature = "use_hexl"))]
pub type DefaultConvolution = crate::ntt::ntt_convolution::NTTConv<Zn>;

#[cfg(feature = "use_hexl")]
pub type DefaultNegacyclicNTT = feanor_math_hexl::hexl::HEXLNegacyclicNTT;
#[cfg(not(feature = "use_hexl"))]
pub type DefaultNegacyclicNTT = RustNegacyclicNTT<Zn>;

pub type DefaultCiphertextAllocator = Global;

///
/// Euler's totient function
/// 
#[allow(unused)]
fn euler_phi(factorization: &[(i64, usize)]) -> i64 {
    StaticRing::<i64>::RING.prod(factorization.iter().map(|(p, e)| (p - 1) * StaticRing::<i64>::RING.pow(*p, e - 1)))
}

///
/// Euler's totient function
/// 
fn euler_phi_squarefree(factorization: &[i64]) -> i64 {
    StaticRing::<i64>::RING.prod(factorization.iter().map(|p| p - 1))
}

///
/// Runs the given function. If `LOG` is true, its running time is printed to stdout.
/// 
pub fn log_time<F, T, const LOG: bool, const COUNTER_VAR_COUNT: usize>(description: &str, step_fn: F) -> T
    where F: FnOnce(&mut [usize; COUNTER_VAR_COUNT]) -> T
{
    if LOG {
        println!("{}", description);
    }
    let mut counters = [0; COUNTER_VAR_COUNT];
    let start = Instant::now();
    let result = step_fn(&mut counters);
    let end = Instant::now();
    if LOG {
        println!("done in {} ms, {:?}", (end - start).as_millis(), counters);
    }
    return result;
}

///
/// Contains some macros that mimic `#[derive(Deserialize)]` but for [`serde::de::DeserializeSeed`].
/// 
mod serialization_helper;

///
/// Contains an abstraction for NTTs and convolutions, which can then be
/// used to configure the ring implementations in this crate.
/// 
pub mod ntt;

///
/// Defines the trait [`cyclotomic::CyclotomicRing`] for rings of the form `R[X]/(Phi_n)`, where `R` is any base ring.
/// 
pub mod cyclotomic;

///
/// Implementation of fast RNS conversion algorithms.
/// 
pub mod rnsconv;

///
/// Contains an HE-specific abstraction for number rings.
/// 
pub mod number_ring;

///
/// Implementation of rings using double-RNS representation.
/// 
pub mod ciphertext_ring;

///
/// Contains an implementation of "gadget products", which are a form of inner
/// products that are commonly used in HE to compute multiplications of noisy values
/// in a way that reduces the increase in noise.
/// 
pub mod gadget_product;

///
/// Contains an implementation of the BFV scheme.
/// 
pub mod bfv;

///
/// The implementation of arithmetic-galois circuits (i.e. circuits built
/// from linear combination, multiplication and galois gates).
/// 
pub mod circuit;

///
/// Contains algorithms to compute linear transformations and represent
/// them as linear combination of Galois automorphisms, as required for
/// (second-generation) HE schemes.
/// 
pub mod lintransform;

///
/// Contains algorithms to build arithmetic circuits, with a focus on
/// digit extraction polynomials.
/// 
pub mod digitextract;

///
/// Contains an implementation of the BGV scheme.
/// 
pub mod bgv;

///
/// Hacky workaround for displaying examples on `docs.rs`.
/// 
/// Contains an empty submodule for each example, whose documentation gives
/// a guide to the corresponding concepts of HE-Ring.
/// 
#[cfg(any(doc, doctest))]
pub mod examples {
    #[doc = include_str!("../examples/bfv_basics/Readme.md")]
    pub mod bfv_basics {}
    #[doc = include_str!("../examples/bgv_basics/Readme.md")]
    pub mod bgv_basics {}
    #[doc = include_str!("../examples/bfv_impl_v1/Readme.md")]
    pub mod bfv_impl_v1 {}
    #[doc = include_str!("../examples/bfv_impl_v2/Readme.md")]
    pub mod bfv_impl_v2 {}
}