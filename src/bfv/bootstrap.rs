use std::alloc::Global;

use feanor_math::algorithms::int_factor::is_prime_power;
use feanor_math::ring::*;
use feanor_math::rings::poly::dense_poly::DensePolyRing;
use polys::poly_to_circuit;

use crate::complexfft::automorphism::HypercubeIsomorphism;
use crate::{digitextract::*, lintransform::pow2::pow2_slots_to_coeffs_thin};
use crate::digitextract::polys::digit_retain_poly;
use crate::lintransform::CompiledLinearTransform;

use super::{PlaintextFFT, PlaintextZn, Pow2BFVParams, ZZ};

pub struct Pow2BFVThinBootstrapParams {
    params: Pow2BFVParams,
    r: usize,
    e: usize,
    // the k-th circuit works modulo `e - k` and outputs values `yi` such that `yi = lift(x mod p) mod p^(i + 2)` for `0 <= i < v - k - 2` as well as a final `y'` with `y' = lift(x mod p)`
    digit_extract_circuits: Vec<ArithCircuit>,
    linear_transform: Vec<CompiledLinearTransform<PlaintextZn, crate::complexfft::pow2_cyclotomic::Pow2CyclotomicFFT<PlaintextZn, PlaintextFFT>, Global>>
}

impl Pow2BFVThinBootstrapParams {

    pub fn create_for(params: Pow2BFVParams) -> Self {
        let (p, r) = is_prime_power(&ZZ, &params.t).unwrap();
        // this is the case if s is uniformly ternary
        let s_can_norm = 1 << params.log2_N;
        let v = ((s_can_norm as f64 + 1.).log2() / (p as f64).log2()).ceil() as usize;
        let e = r + v;

        let digit_extraction_circuits = (1..=v).rev().map(|remaining_v| {
            let poly_ring = DensePolyRing::new(PlaintextZn::new(ZZ.pow(p, remaining_v + r) as u64), "X");
            poly_to_circuit(&poly_ring, &(2..remaining_v).chain([r + remaining_v].into_iter()).map(|j| digit_retain_poly(&poly_ring, j)).collect::<Vec<_>>())
        }).collect::<Vec<_>>();

        let plaintext_ring = params.create_plaintext_ring();
        let H = HypercubeIsomorphism::new(plaintext_ring.get_ring());
        let linear_transform = pow2_slots_to_coeffs_thin(&H);
        let compiled_linear_transform = linear_transform.into_iter().map(|T| CompiledLinearTransform::compile(&H, T)).collect();

        return Self {
            params: params,
            e: e,
            r: r,
            digit_extract_circuits: digit_extraction_circuits,
            linear_transform: compiled_linear_transform
        };
    }
}