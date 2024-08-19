#![feature(never_type)]
#![feature(fn_traits)]
#![feature(unboxed_closures)]
#![feature(test)]
#![feature(const_type_name)]
#![feature(allocator_api)]#![feature(hint_assert_unchecked)]
#![feature(ptr_alignment_type)]

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

#![doc = include_str!("../Readme.md")]

use std::cell::RefCell;

use feanor_math::integer::BigIntRingBase;
use feanor_math::primitive_int::StaticRing;
use feanor_math::primitive_int::StaticRingBase;
use feanor_math::ring::*;
use feanor_math::homomorphism::*;
use feanor_math::rings::extension::galois_field::GaloisFieldDyn;
use feanor_math::rings::local::AsLocalPIRBase;
use feanor_math::rings::zn::zn_64;
use feanor_math::rings::zn::zn_64::Zn;
use feanor_math::rings::zn::ZnRingStore;
use feanor_math::rings::zn::{FromModulusCreateableZnRing, ZnRing};
use feanor_math::serialization::SerializableElementRing;

#[macro_export]
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
extern crate test;

pub trait StdZn: ZnRing 
    + FromModulusCreateableZnRing 
    + CanHomFrom<<<<GaloisFieldDyn as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type> 
    + CanHomFrom<AsLocalPIRBase<zn_64::Zn>>
    + CanHomFrom<StaticRingBase<i64>>
    + CanHomFrom<BigIntRingBase>
    + SerializableElementRing
{}

impl<R> StdZn for R
    where R: ZnRing 
    + FromModulusCreateableZnRing 
    + CanHomFrom<<<<GaloisFieldDyn as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type> 
    + CanHomFrom<AsLocalPIRBase<zn_64::Zn>>
    + CanHomFrom<StaticRingBase<i64>>
    + CanHomFrom<BigIntRingBase>
    + SerializableElementRing
{}

#[macro_use]
pub mod profiling;

///
/// Defines the trait [`cyclotomic::CyclotomicRing`] for rings of the form `R[X]/(Phi_n)`, where `R` is any base ring.
/// 
pub mod cyclotomic;

///
/// Implementation of fast RNS conversion algorithms.
/// 
pub mod rnsconv;

///
/// Implementation of rings using complex-valued fast fourier transforms for efficient arithmetic.
/// 
pub mod complexfft;

///
/// Implementation of rings using double-RNS representation.
/// 
pub mod rings;

pub mod lintransform;
pub mod digitextract;

// #[cfg(test)]
// pub mod bfv;
