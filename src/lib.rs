#![feature(never_type)]
#![feature(fn_traits)]
#![feature(unboxed_closures)]
#![feature(test)]
#![feature(const_type_name)]
#![feature(allocator_api)]#![feature(hint_assert_unchecked)]
#![feature(ptr_alignment_type)]
#![feature(associated_type_defaults)]

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

#![doc = include_str!("../Readme.md")]

use std::cmp::min;

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
use feanor_math::rings::extension::galois_field::GaloisFieldDyn;
use feanor_math::rings::field::AsFieldBase;
use feanor_math::rings::local::AsLocalPIRBase;
use feanor_math::rings::zn::zn_64;
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
    + CanHomFrom<StaticRingBase<i128>>
    + for<'a> CanHomFrom<AsFieldBase<&'a RingValue<Self>>>
    + CanHomFrom<BigIntRingBase>
    + SerializableElementRing
{}

impl<R> StdZn for R
    where R: ZnRing 
    + FromModulusCreateableZnRing 
    + CanHomFrom<<<<GaloisFieldDyn as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type> 
    + CanHomFrom<AsLocalPIRBase<zn_64::Zn>>
    + CanHomFrom<StaticRingBase<i64>>
    + CanHomFrom<StaticRingBase<i128>>
    + for<'a> CanHomFrom<AsFieldBase<&'a RingValue<Self>>>
    + CanHomFrom<BigIntRingBase>
    + SerializableElementRing
{}

pub fn sample_primes(min_bits: usize, max_bits: usize, max_bits_each_modulus: usize, congruent_to_one_mod: &El<BigIntRing>) -> Option<Vec<El<BigIntRing>>> {
    assert!(max_bits > min_bits);
    let ZZbig = BigIntRing::RING;
    let mut result = Vec::new();
    let mut current_bits = 0.;
    while current_bits < min_bits as f64 {
        let next_modulus_add_bound = min(max_bits_each_modulus, max_bits - current_bits.ceil() as usize);
        let mut current = ZZbig.add(ZZbig.sub(ZZbig.power_of_two(next_modulus_add_bound), ZZbig.euclidean_rem(ZZbig.power_of_two(next_modulus_add_bound), congruent_to_one_mod)), ZZbig.one());
        let mut added_any = false;
        while ZZbig.is_pos(&current) {
            if is_prime(&ZZbig, &current, 10) {
                let bits = ZZbig.to_float_approx(&current).log2();
                if current_bits + bits > max_bits as f64 {
                    break;
                } else {
                    added_any = true;
                    current_bits += bits;
                    result.push(ZZbig.clone_el(&current));
                }
            }
            ZZbig.sub_assign_ref(&mut current, congruent_to_one_mod);
        }
        if !added_any {
            return None;
        }
    }
    debug_assert!(ZZbig.is_geq(&ZZbig.prod(result.iter().map(|n| ZZbig.clone_el(n))), &ZZbig.power_of_two(min_bits)));
    debug_assert!(ZZbig.is_lt(&ZZbig.prod(result.iter().map(|n| ZZbig.clone_el(n))), &ZZbig.power_of_two(max_bits)));
    return Some(result);
}

///
/// Euler's totient function
/// 
fn euler_phi(factorization: &[(i64, usize)]) -> i64 {
    StaticRing::<i64>::RING.prod(factorization.iter().map(|(p, e)| (p - 1) * StaticRing::<i64>::RING.pow(*p, e - 1)))
}

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
/// Implementation of rings using double-RNS representation.
/// 
pub mod rings;

pub mod lintransform;

pub mod digitextract;

#[cfg(test)]
pub mod bfv;
