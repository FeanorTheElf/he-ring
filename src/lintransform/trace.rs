use std::alloc::*;
use std::cell::RefCell;
use std::marker::PhantomData;

use feanor_math::algorithms::convolution::STANDARD_CONVOLUTION;
use feanor_math::algorithms::sqr_mul::generic_pow_shortest_chain_table;
use feanor_math::homomorphism::*;
use feanor_math::integer::*;
use feanor_math::matrix::OwnedMatrix;
use feanor_math::rings::extension::FreeAlgebraStore;
use feanor_math::rings::field::AsFieldBase;
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

use crate::rings::dynconv::DynConvolutionAlgorithmConvolution;
use crate::rings::number_ring::HECyclotomicNumberRing;
use crate::rings::odd_cyclotomic::OddCyclotomicNumberRing;
use crate::rings::pow2_cyclotomic::Pow2CyclotomicNumberRing;
use crate::StdZn;

use crate::rings::decomposition_ring::*;
use crate::rings::slots::*;
use crate::cyclotomic::*;

pub struct Trace<NumberRing, A>
    where NumberRing: HECyclotomicNumberRing,
        A: Allocator + Clone
{
    number_ring: NumberRing,
    frobenius: ZnEl,
    trace_rank_quo: i64,
    allocator: PhantomData<A>
}

impl<NumberRing, A> Trace<NumberRing, A>
    where NumberRing: HECyclotomicNumberRing,
        A: Allocator + Clone
{
    pub fn new(number_ring: NumberRing, p: i64, slot_rank: usize) -> Self {
        assert_eq!(0, number_ring.rank() % slot_rank);
        Trace {
            frobenius: number_ring.cyclotomic_index_ring().can_hom(&StaticRing::<i64>::RING).unwrap().map(p),
            number_ring: number_ring,
            trace_rank_quo: slot_rank as i64,
            allocator: PhantomData
        }
    }

    ///
    /// Computes `a` such that `Tr(aY)` is the given `Fp`-linear map `GR(p, e, d) -> Z/p^eZ`.
    /// 
    /// We assume that the frobenius automorphism in the given ring is given by `X -> X^p`
    /// where `X` is its canonical generator. At the moment this always true, since we currently
    /// choose the canonical generator to be a root of unity.
    /// 
    pub fn extract_linear_map<'a, G>(&self, slot_ring: &SlotRing<'a, A>, mut function: G) -> El<SlotRing<'a, A>>
        where G: FnMut(El<SlotRing<'a, A>>) -> El<Zn>
    {
        assert_eq!(self.trace_rank_quo as usize, slot_ring.rank());

        let mut lhs = OwnedMatrix::zero(slot_ring.rank(), slot_ring.rank(), slot_ring.base_ring());
        let mut rhs = OwnedMatrix::zero(slot_ring.rank(), 1, slot_ring.base_ring());
        let mut sol = OwnedMatrix::zero(slot_ring.rank(), 1, slot_ring.base_ring());

        let Gal = self.number_ring.cyclotomic_index_ring();
        let poly_ring = DensePolyRing::new(slot_ring.base_ring(), "X");
        let trace = |a: El<SlotRing<'a, A>>| {
            let result = self.evaluate_generic(
                a, 
                |x, y| slot_ring.add_ref_snd(x, y), 
                |_, _| unreachable!(),
                |x, gs| gs.iter().map(|g| poly_ring.evaluate(&slot_ring.poly_repr(&poly_ring, &x, &slot_ring.base_ring().identity()), &slot_ring.pow(slot_ring.canonical_gen(), Gal.smallest_positive_lift(*g) as usize), &slot_ring.inclusion())).collect(), 
                |x| slot_ring.clone_el(x)
            );
            let result_wrt_basis = slot_ring.wrt_canonical_basis(&result);
            assert!((1..slot_ring.rank()).all(|i| slot_ring.base_ring().is_zero(&result_wrt_basis.at(i))));
            return result_wrt_basis.at(0);
        };
        for i in 0..slot_ring.rank() {
            for j in 0..slot_ring.rank() {
                *lhs.at_mut(i, j) = trace(slot_ring.pow(slot_ring.canonical_gen(), i + j));
            }
        }
        for j in 0..slot_ring.rank() {
            *rhs.at_mut(j, 0) = function(slot_ring.pow(slot_ring.canonical_gen(), j));
        }

        slot_ring.base_ring().solve_right(lhs.data_mut(), rhs.data_mut(), sol.data_mut()).assert_solved();

        return slot_ring.from_canonical_basis((0..slot_ring.rank()).map(|i| slot_ring.base_ring().clone_el(sol.at(i, 0))));
    }

    ///
    /// Computes `a` such that `Tr(aY)` is the map that maps `Y = sum_i c_i X^i` to `c_j`.
    /// 
    /// We assume that the frobenius automorphism in the given ring is given by `X -> X^p`
    /// where `X` is its canonical generator. At the moment this always true, since we currently
    /// choose the canonical generator to be a root of unity.
    /// 
    pub fn extract_coefficient_map<'a>(&self, slot_ring: &SlotRing<'a, A>, coefficient_j: usize) -> El<SlotRing<'a, A>> {
        self.extract_linear_map(slot_ring, |a| slot_ring.wrt_canonical_basis(&a).at(coefficient_j))
    }
}

impl<NumberRing, A> HELinearTransform<NumberRing, A> for Trace<NumberRing, A>
    where NumberRing: HECyclotomicNumberRing,
        A: Allocator + Clone
{
    fn number_ring(&self) -> &NumberRing {
        &self.number_ring
    }

    fn evaluate_generic<T, AddFn, ScaleFn, ApplyGaloisFn, CloneFn>(
        &self,
        input: T,
        add_fn: AddFn,
        _scale_fn: ScaleFn,
        apply_galois_fn: ApplyGaloisFn,
        clone_fn: CloneFn
    ) -> T
        where AddFn: FnMut(T, &T) -> T,
            ScaleFn: FnMut(T, &El<DecompositionRing<NumberRing, Zn, A>>) -> T,
            ApplyGaloisFn: FnMut(T, &[ZnEl]) -> Vec<T>,
            CloneFn: FnMut(&T) -> T
    {
        let Gal = self.number_ring.cyclotomic_index_ring();
        let add_fn = RefCell::new(add_fn);
        let apply_galois_fn = RefCell::new(apply_galois_fn);
        let clone_fn = RefCell::new(clone_fn);
        generic_pow_shortest_chain_table::<_, _, _, _, _, !>(
            (1, Some(input)), 
            &self.trace_rank_quo, 
            StaticRing::<i64>::RING, 
            |(i, x)| {
                if let Some(x) = x {
                    Ok((2 * i, Some(add_fn.borrow_mut()(apply_galois_fn.borrow_mut()(clone_fn.borrow_mut()(x), &vec![Gal.pow(self.frobenius, *i)]).into_iter().next().unwrap(), x))))
                } else {
                    assert_eq!(0, *i);
                    Ok((0, None))
                }
            }, |(i, x), (j, y)| {
                if x.is_none() {
                    assert_eq!(0, *i);
                    return Ok((i + j, y.as_ref().map(|y| clone_fn.borrow_mut()(y))));
                } else if y.is_none() {
                    assert_eq!(0, *j);
                    return Ok((i + j, x.as_ref().map(|x| clone_fn.borrow_mut()(x))));
                }
                Ok((i + j, Some(add_fn.borrow_mut()(apply_galois_fn.borrow_mut()(clone_fn.borrow_mut()(x.as_ref().unwrap()), &vec![Gal.pow(self.frobenius, *j)]).into_iter().next().unwrap(), y.as_ref().unwrap()))))
            }, 
            |(i, x)| (*i, x.as_ref().map(|x| clone_fn.borrow_mut()(x))), 
            (0, None)
        ).unwrap_or_else(|x| x).1.unwrap()
    }
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

use super::HELinearTransform;

#[test]
fn test_extract_coefficient_map() {

    let do_test_for_slot_ring = |slot_ring: &SlotRing, trace: &Trace<OddCyclotomicNumberRing, Global>, Gal: &Zn| {
        let extract_constant_coeff = trace.extract_coefficient_map(slot_ring, 0);

        let poly_ring = DensePolyRing::new(slot_ring.base_ring(), "X");
        for i in 0..4 {
            let b = slot_ring.pow(slot_ring.canonical_gen(), i);
            let actual = trace.evaluate_generic(
                slot_ring.mul_ref(&b, &extract_constant_coeff), 
                |x, y| slot_ring.add_ref_snd(x, y), 
                |_, _| unreachable!(),
                |x, gs| gs.iter().map(|g| poly_ring.evaluate(&slot_ring.poly_repr(&poly_ring, &x, &slot_ring.base_ring().identity()), &slot_ring.pow(slot_ring.canonical_gen(), Gal.smallest_positive_lift(*g) as usize), &slot_ring.inclusion())).collect(),
                |x| slot_ring.clone_el(x)
            );
            if i == 0 {
                assert_el_eq!(&slot_ring, &slot_ring.one(), &actual);
            } else {
                assert_el_eq!(&slot_ring, &slot_ring.zero(), &actual);
            }
        }
    };

    let convolution = || DynConvolutionAlgorithmConvolution::<ZnBase>::new(Box::new(STANDARD_CONVOLUTION));

    let base_ring = Zn::new(17).as_field().ok().unwrap();
    let mut modulus = SparseMapVector::new(4, &base_ring);
    for i in 0..4 {
        *modulus.at_mut(i) = base_ring.neg_one();
    }

    let slot_ring = GaloisField::create(FreeAlgebraImpl::new(&base_ring, 4, modulus).as_field().ok().unwrap());
    assert!(is_prim_root_of_unity(&slot_ring, &slot_ring.canonical_gen(), 5));
    let Gal = Zn::new(5);
    let trace = Trace::new(OddCyclotomicNumberRing::new(5), 17, 4);

    let slot_ring_pir_base_ring = Zn::new(17);
    let slot_ring_pir = slot_ring.clone().into().galois_ring_with(&slot_ring_pir_base_ring, Global, convolution());
    do_test_for_slot_ring(&slot_ring_pir, &trace, &Gal);
    
    let slot_ring_pir_base_ring = Zn::new(17 * 17);
    let slot_ring_pir = slot_ring.clone().into().galois_ring_with(&slot_ring_pir_base_ring, Global, convolution());
    do_test_for_slot_ring(&slot_ring_pir, &trace, &Gal);
}