use std::alloc::{Allocator, Global};
use std::cell::RefCell;
use std::collections::BTreeMap;

use feanor_math::homomorphism::*;
use feanor_math::rings::zn::FromModulusCreateableZnRing;
use feanor_math::ring::*;
use feanor_math::rings::zn::*;
use feanor_math::integer::*;

use super::decomposition::DecomposableNumberRing;

pub struct NumberRingBase<NumberRing, FpTy, A = Global> 
    where NumberRing: DecomposableNumberRing<RingValue<FpTy>>,
        FpTy: ZnRing + CanHomFrom<BigIntRingBase> + FromModulusCreateableZnRing,
        A: Allocator + Clone
{
    number_ring: NumberRing,
    ring_decompositions: Vec<<NumberRing as DecomposableNumberRing<RingValue<FpTy>>>::Decomposed>,
    rns_bases: RefCell<BTreeMap<usize, zn_rns::Zn<RingValue<FpTy>, BigIntRing>>>,
    allocator: A
}
