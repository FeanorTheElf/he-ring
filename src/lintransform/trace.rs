use std::alloc::*;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

use feanor_math::algorithms::convolution::ConvolutionAlgorithm;
use feanor_math::algorithms::convolution::STANDARD_CONVOLUTION;
use feanor_math::algorithms::sqr_mul::generic_pow_shortest_chain_table;
use feanor_math::computation::no_error;
use feanor_math::homomorphism::*;
use feanor_math::integer::*;
use feanor_math::matrix::OwnedMatrix;
use feanor_math::rings::extension::FreeAlgebraStore;
use feanor_math::rings::field::AsFieldBase;
use feanor_math::rings::finite::FiniteRingStore;
use feanor_math::rings::local::AsLocalPIR;
use feanor_math::rings::local::AsLocalPIRBase;
use feanor_math::rings::poly::dense_poly::DensePolyRing;
use feanor_math::rings::poly::PolyRingStore;
use feanor_math::algorithms::sqr_mul;
use feanor_math::rings::zn::zn_64::ZnEl;
use feanor_math::primitive_int::StaticRing;
use feanor_math::ring::*;
use feanor_math::rings::zn::zn_64::*;
use feanor_math::rings::zn::*;
use feanor_math::seq::*;
use feanor_math::algorithms::linsolve::LinSolveRingStore;

use crate::circuit::PlaintextCircuit;
use crate::number_ring::hypercube::DefaultHypercube;
use crate::number_ring::hypercube::SlotRingOver;
use crate::number_ring::odd_cyclotomic::OddCyclotomicNumberRing;
use crate::number_ring::pow2_cyclotomic::Pow2CyclotomicNumberRing;
use crate::number_ring::quotient::NumberRingQuotientBase;
use crate::number_ring::{HECyclotomicNumberRing, HENumberRing};
use crate::number_ring::quotient::NumberRingQuotient;
use crate::number_ring::hypercube::HypercubeIsomorphism;
use crate::cyclotomic::*;

pub fn trace_circuit<R>(ring: &R, galois_group: &CyclotomicGaloisGroup, relative_galois_group_gen: CyclotomicGaloisGroupEl, relative_degree: usize) -> PlaintextCircuit<R>
    where R: ?Sized + RingBase
{
    assert!(galois_group.is_identity(galois_group.pow(relative_galois_group_gen, relative_degree as i64)));

    let ring = RingRef::new(ring);
    let mut circuit = PlaintextCircuit::identity(1, ring);

    let extend_circuit = RefCell::new(|l_idx: usize, r_idx: usize, l_num: usize| {
        take_mut::take(&mut circuit, |circuit| PlaintextCircuit::identity(circuit.output_count(), ring).tensor(PlaintextCircuit::add(ring).compose(
            PlaintextCircuit::identity(1, ring).tensor(PlaintextCircuit::gal(galois_group.pow(relative_galois_group_gen, l_num as i64), ring), ring), ring
        ), ring).compose(
            PlaintextCircuit::select(circuit.output_count(), &(0..circuit.output_count()).chain([l_idx, r_idx].into_iter()).collect::<Vec<_>>(), ring), ring
        ).compose(
            circuit, ring
        ));
        return circuit.output_count() - 1;
    });

    let result_idx = generic_pow_shortest_chain_table(
        (Some(0), 1),
        &(relative_degree as i64),
        StaticRing::<i64>::RING,
        |(idx, num)| {
            if let Some(idx) = idx {
                let result = extend_circuit.borrow_mut()(*idx, *idx, *num);
                Ok((Some(result), num + num))
            } else {
                Ok((None, 0))
            }
        },
        |(l_idx, l_num), (r_idx, r_num)| {
            if let Some(l_idx) = l_idx {
                if let Some(r_idx) = r_idx {
                    let result = extend_circuit.borrow_mut()(*l_idx, *r_idx, *l_num);
                    Ok((Some(result), l_num + r_num))
                } else {
                    Ok((Some(*l_idx), *l_num))
                }
            } else {
                Ok((*r_idx, *r_num))
            }
        },
        |x| *x,
        (None, 0)
    ).unwrap_or_else(no_error).0.unwrap();

    return PlaintextCircuit::select(circuit.output_count(), &[result_idx], ring).compose(circuit, ring);
}

///
/// Computes `a` such that `Tr(aY)` is the given `Fp`-linear map `GR(p, e, d) -> Z/p^eZ`.
/// 
/// We assume that the frobenius automorphism in the given ring is given by `X -> X^p`
/// where `X` is its canonical generator. At the moment this always true, since we currently
/// choose the canonical generator to be a root of unity.
/// 
/// If the given function `function` is not `Fp`-linear, results may be nonsensical.
/// 
pub fn extract_linear_map<G>(slot_ring: &SlotRingOver<Zn>, mut function: G) -> El<SlotRingOver<Zn>>
    where G: FnMut(El<SlotRingOver<Zn>>) -> El<Zn>
{
    let mut lhs = OwnedMatrix::zero(slot_ring.rank(), slot_ring.rank(), slot_ring.base_ring());
    let mut rhs = OwnedMatrix::zero(slot_ring.rank(), 1, slot_ring.base_ring());
    let mut sol = OwnedMatrix::zero(slot_ring.rank(), 1, slot_ring.base_ring());

    let poly_ring = DensePolyRing::new(slot_ring.base_ring(), "X");
    for i in 0..slot_ring.rank() {
        for j in 0..slot_ring.rank() {
            *lhs.at_mut(i, j) = slot_ring.trace(slot_ring.pow(slot_ring.canonical_gen(), i + j));
        }
    }
    for j in 0..slot_ring.rank() {
        *rhs.at_mut(j, 0) = function(slot_ring.pow(slot_ring.canonical_gen(), j));
    }

    slot_ring.base_ring().solve_right(lhs.data_mut(), rhs.data_mut(), sol.data_mut()).assert_solved();

    return slot_ring.from_canonical_basis((0..slot_ring.rank()).map(|i| slot_ring.base_ring().clone_el(sol.at(i, 0))));
}

#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use feanor_math::rings::extension::galois_field::GaloisField;
#[cfg(test)]
use feanor_math::seq::sparse::*;
#[cfg(test)]
use feanor_math::algorithms::convolution::fft::{FFTRNSBasedConvolution, FFTRNSBasedConvolutionZn};
#[cfg(test)]
use feanor_math::algorithms::unity_root::is_prim_root_of_unity;
#[cfg(test)]
use feanor_math::integer::BigIntRing;
#[cfg(test)]
use feanor_math::rings::extension::extension_impl::FreeAlgebraImpl;
#[cfg(test)]
use crate::ntt::dyn_convolution::*;

#[test]
fn test_extract_coefficient_map() {
    let convolution = DynConvolutionAlgorithmConvolution::<ZnBase, Arc<dyn Send + Sync + DynConvolutionAlgorithm<ZnBase>>>::new(Arc::new(STANDARD_CONVOLUTION));
    let base_ring = Zn::new(17 * 17);
    let modulus = (0..4).map(|_| base_ring.neg_one()).collect::<Vec<_>>();
    let slot_ring = FreeAlgebraImpl::new_with(base_ring, 4, modulus, "a", Global, convolution);
    let max_ideal_gen = slot_ring.int_hom().map(17);
    let slot_ring = AsLocalPIR::from(AsLocalPIRBase::promise_is_local_pir(slot_ring, max_ideal_gen, Some(2)));
    assert!(is_prim_root_of_unity(&slot_ring, &slot_ring.canonical_gen(), 5));
    let Gal = CyclotomicGaloisGroup::new(5);

    let extract_constant_coeff = extract_linear_map(&slot_ring, |c| slot_ring.wrt_canonical_basis(&c).at(0));
    let poly_ring = DensePolyRing::new(slot_ring.base_ring(), "X");
    for i in 0..4 {
        let b = slot_ring.pow(slot_ring.canonical_gen(), i);
        let actual = slot_ring.trace(slot_ring.mul_ref(&b, &extract_constant_coeff));
        if i == 0 {
            assert_el_eq!(slot_ring.base_ring(), slot_ring.base_ring().one(), actual);
        } else {
            assert_el_eq!(slot_ring.base_ring(), slot_ring.base_ring().zero(), actual);
        }
    }
}

#[test]
fn test_trace_circuit() {
    let ring = NumberRingQuotientBase::new(OddCyclotomicNumberRing::new(7), Zn::new(3));
    let trace = trace_circuit(ring.get_ring(), &ring.galois_group(), ring.galois_group().from_representative(3), 6);
    for x in ring.elements() {
        let actual = trace.evaluate(std::slice::from_ref(&x), ring.identity()).pop().unwrap();
        assert_el_eq!(&ring, ring.inclusion().map(ring.trace(x)), actual);
    }

    let relative_trace = trace_circuit(ring.get_ring(), &ring.galois_group(), ring.galois_group().from_representative(2), 3);
    assert_eq!(1, relative_trace.output_count());
    
    let input = ring.canonical_gen();
    let actual = relative_trace.evaluate(std::slice::from_ref(&input), ring.identity()).pop().unwrap();
    let expected = ring.sum([ring.canonical_gen(), ring.pow(ring.canonical_gen(), 2), ring.pow(ring.canonical_gen(), 4)]);
    assert_el_eq!(&ring, expected, actual);
}