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

#![allow(non_snake_case)]
#![allow(type_alias_bounds)]
#![allow(unused_imports)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(unused_variables)]

#![doc = include_str!("../Readme.md")]

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
use feanor_math::rings::zn::{FromModulusCreateableZnRing, ZnRing};
use feanor_math::serialization::SerializableElementRing;

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

pub trait IsEq<T: ?Sized> {

    fn from_ref<'a>(t: &'a T) -> &'a Self;
    fn to_ref<'a>(&'a self) -> &'a T;
}

impl<T: ?Sized> IsEq<T> for T {
    
    fn from_ref<'a>(t: &'a T) -> &'a Self { t }
    fn to_ref<'a>(&'a self) -> &'a T { self }
}

pub trait StdZn: ZnRing 
    + FromModulusCreateableZnRing 
    + CanHomFrom<<<<GaloisField as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type> 
    + CanHomFrom<AsLocalPIRBase<zn_64::Zn>>
    + CanHomFrom<StaticRingBase<i64>>
    + CanHomFrom<StaticRingBase<i128>>
    + for<'a> CanHomFrom<AsFieldBase<RingRef<'a, Self>>>
    + for<'a> CanHomFrom<AsLocalPIRBase<RingRef<'a, Self>>>
    + CanHomFrom<BigIntRingBase>
    + SerializableElementRing
{}

impl<R> StdZn for R
    where R: ZnRing 
    + FromModulusCreateableZnRing 
    + CanHomFrom<<<<GaloisField as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type> 
    + CanHomFrom<AsLocalPIRBase<zn_64::Zn>>
    + CanHomFrom<StaticRingBase<i64>>
    + CanHomFrom<StaticRingBase<i128>>
    + for<'a> CanHomFrom<AsFieldBase<RingRef<'a, Self>>>
    + for<'a> CanHomFrom<AsLocalPIRBase<RingRef<'a, Self>>>
    + CanHomFrom<BigIntRingBase>
    + SerializableElementRing
{}

pub fn largest_prime_congruent_one(modulus: El<BigIntRing>) -> impl Fn(El<BigIntRing>) -> Option<El<BigIntRing>> {
    move |leq_than| {
        let ZZbig = BigIntRing::RING;
        let lt_than = ZZbig.sub(leq_than, ZZbig.one());
        let mut current = ZZbig.add(ZZbig.sub(ZZbig.clone_el(&lt_than), ZZbig.euclidean_rem(lt_than, &modulus)), ZZbig.one());
        while ZZbig.is_pos(&current) && !is_prime(ZZbig, &current, 10) {
            ZZbig.sub_assign_ref(&mut current, &modulus);
        }
        if ZZbig.is_pos(&current) {
            return Some(current);
        } else {
            return None;
        }
    }
}

pub fn sample_primes<F>(min_bits: usize, max_bits: usize, max_bits_each_modulus: usize, largest_prime_leq: F) -> Option<Vec<El<BigIntRing>>>
    where F: FnMut(El<BigIntRing>) -> Option<El<BigIntRing>>
{
    extend_sampled_primes(&[], min_bits, max_bits, max_bits_each_modulus, largest_prime_leq)
}

pub fn extend_sampled_primes<F>(begin_with: &[El<BigIntRing>], min_bits: usize, max_bits: usize, max_bits_each_modulus: usize, mut largest_prime_leq: F) -> Option<Vec<El<BigIntRing>>>
    where F: FnMut(El<BigIntRing>) -> Option<El<BigIntRing>>
{
    let ZZbig = BigIntRing::RING;
    assert!(max_bits > min_bits);

    let mut result = begin_with.iter().map(|p| ZZbig.clone_el(p)).collect::<Vec<_>>();
    let mut current_bits = result.iter().map(|n| ZZbig.to_float_approx(n).log2()).sum::<f64>();
    assert!((current_bits.floor() as usize) < max_bits);
    let mut current_upper_bound = ZZbig.power_of_two(max_bits_each_modulus);

    let min = |x, y| if ZZbig.is_gt(&x, &y) { y } else { x };

    while current_bits < min_bits as f64 {

        if min_bits as f64 - current_bits < max_bits_each_modulus as f64 {  
            current_upper_bound = min(current_upper_bound, ZZbig.power_of_two(f64::min(max_bits as f64 - current_bits, max_bits_each_modulus as f64).floor() as usize));
        } else {
            let required_number_of_primes = ((min_bits as f64 - current_bits) / max_bits_each_modulus as f64).ceil() as usize;
            current_upper_bound = min(current_upper_bound, ZZbig.power_of_two(f64::min((max_bits as f64 - current_bits) / required_number_of_primes as f64, max_bits_each_modulus as f64).floor() as usize));
        }

        let mut prime = largest_prime_leq(ZZbig.clone_el(&current_upper_bound))?;
        current_upper_bound = ZZbig.sub_ref_fst(&prime, ZZbig.one());
        while begin_with.iter().any(|p| ZZbig.eq_el(p, &prime)) {
            prime = largest_prime_leq(ZZbig.clone_el(&current_upper_bound))?;
            current_upper_bound = ZZbig.sub_ref_fst(&prime, ZZbig.one());
        }
        let bits = ZZbig.to_float_approx(&prime).log2();
        current_bits += bits;
        result.push(ZZbig.clone_el(&prime));
    }
    debug_assert!(ZZbig.is_geq(&ZZbig.prod(result.iter().map(|n| ZZbig.clone_el(n))), &ZZbig.power_of_two(min_bits)));
    debug_assert!(ZZbig.is_lt(&ZZbig.prod(result.iter().map(|n| ZZbig.clone_el(n))), &ZZbig.power_of_two(max_bits)));
    return Some(result);
}

///
/// Euler's totient function
/// 
#[allow(unused)]
fn euler_phi(factorization: &[(i64, usize)]) -> i64 {
    StaticRing::<i64>::RING.prod(factorization.iter().map(|(p, e)| (p - 1) * StaticRing::<i64>::RING.pow(*p, e - 1)))
}

fn euler_phi_squarefree(factorization: &[i64]) -> i64 {
    StaticRing::<i64>::RING.prod(factorization.iter().map(|p| p - 1))
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

#[cfg(test)]
use feanor_math::integer::int_cast;

#[test]
fn test_sample_primes() {
    let ZZbig = BigIntRing::RING;
    let result = sample_primes(60, 62, 58, largest_prime_congruent_one(int_cast(422144, ZZbig, StaticRing::<i64>::RING))).unwrap();
    assert_eq!(result.len(), 2);
    let prod = ZZbig.prod(result.iter().map(|n| ZZbig.clone_el(n)));
    assert!(ZZbig.abs_log2_floor(&prod).unwrap() >= 60);
    assert!(ZZbig.abs_log2_ceil(&prod).unwrap() <= 62);
    assert!(result.iter().all(|n| ZZbig.is_one(&ZZbig.euclidean_rem(ZZbig.clone_el(n), &int_cast(422144, ZZbig, StaticRing::<i64>::RING)))));

    let ZZbig = BigIntRing::RING;
    let result = sample_primes(135, 138, 58, largest_prime_congruent_one(int_cast(422144, ZZbig, StaticRing::<i64>::RING))).unwrap();
    assert_eq!(result.len(), 3);
    let prod = ZZbig.prod(result.iter().map(|n| ZZbig.clone_el(n)));
    assert!(ZZbig.abs_log2_floor(&prod).unwrap() >= 135);
    assert!(ZZbig.abs_log2_ceil(&prod).unwrap() <= 138);
    assert!(result.iter().all(|n| ZZbig.is_one(&ZZbig.euclidean_rem(ZZbig.clone_el(n), &int_cast(422144, ZZbig, StaticRing::<i64>::RING)))));

    let ZZbig = BigIntRing::RING;
    let result = sample_primes(115, 118, 58, largest_prime_congruent_one(int_cast(422144, ZZbig, StaticRing::<i64>::RING))).unwrap();
    assert_eq!(result.len(), 2);
    let prod = ZZbig.prod(result.iter().map(|n| ZZbig.clone_el(n)));
    assert!(ZZbig.abs_log2_floor(&prod).unwrap() >= 115);
    assert!(ZZbig.abs_log2_ceil(&prod).unwrap() <= 118);
    assert!(result.iter().all(|n| ZZbig.is_one(&ZZbig.euclidean_rem(ZZbig.clone_el(n), &int_cast(422144, ZZbig, StaticRing::<i64>::RING)))));
}