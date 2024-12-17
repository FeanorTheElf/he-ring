use std::alloc::Allocator;
use std::cell::{Ref, RefCell};

use feanor_math::primitive_int::StaticRing;
use feanor_math::rings::zn::zn_64::{Zn, ZnEl};
use feanor_math::ring::*;
use feanor_math::rings::zn::ZnRingStore;

use crate::cyclotomic::CyclotomicRingStore;
use crate::profiling::TimeRecorder;
use crate::rings::number_ring::HECyclotomicNumberRing;
use crate::rings::{decomposition_ring::DecompositionRing, number_ring::HENumberRing};

pub static CREATE_LINEAR_TRANSFORM_TIME_RECORDER: TimeRecorder = TimeRecorder::new("CreateLinTransform");

pub mod matmul;
pub mod trace;
pub mod composite;
pub mod pow2;
pub mod broadcast;

pub trait HELinearTransform<NumberRing, A>
    where NumberRing: HECyclotomicNumberRing,
        A: Allocator + Clone
{
    fn number_ring(&self) -> &NumberRing;

    fn evaluate_generic<T, AddFn, ScaleFn, ApplyGaloisFn, CloneFn>(
        &self,
        input: T,
        add_fn: AddFn,
        scale_fn: ScaleFn,
        apply_galois_fn: ApplyGaloisFn,
        clone_fn: CloneFn
    ) -> T
        where AddFn: FnMut(T, &T) -> T,
            ScaleFn: FnMut(T, &El<DecompositionRing<NumberRing, Zn, A>>) -> T,
            ApplyGaloisFn: FnMut(T, &[ZnEl]) -> Vec<T>,
            CloneFn: FnMut(&T) -> T;

    fn required_galois_keys(&self) -> Vec<ZnEl> {
        let Gal = self.number_ring().cyclotomic_index_ring();
        let mut result = Vec::new();
        self.evaluate_generic((), |(), ()| (), |(), _| (), |(), gs| { result.extend(gs.iter().copied()); gs.iter().map(|_| ()).collect::<Vec<_>>() }, |()| ());
        result.sort_unstable_by_key(|g| Gal.smallest_positive_lift(*g));
        result.dedup_by(|g1, g2| Gal.eq_el(g1, g2));
        return result;
    }

    fn evaluate(&self, ring: &DecompositionRing<NumberRing, Zn, A>, input: El<DecompositionRing<NumberRing, Zn, A>>) -> El<DecompositionRing<NumberRing, Zn, A>> {
        self.evaluate_generic(input, |a, b| ring.add_ref_snd(a, b), |a, b| ring.mul_ref_snd(a, b), |x, gs| ring.apply_galois_action_many(&x, gs).collect(), |a| ring.clone_el(a))
    }
}

pub struct PowerTable<R>
    where R: RingStore
{
    ring: R,
    exponent_ring: Zn,
    powers: RefCell<Vec<El<R>>>
}

impl<R> PowerTable<R>
    where R: RingStore
{
    pub fn new(ring: R, base: El<R>, order_of_base: usize) -> Self {
        debug_assert!(ring.is_one(&ring.pow(ring.clone_el(&base), order_of_base)));
        Self {
            powers: RefCell::new(vec![ring.one(), base]),
            ring: ring,
            exponent_ring: Zn::new(order_of_base as u64),
        }
    }

    pub fn get_power<'a>(&'a self, power: i64) -> Ref<'a, El<R>> {
        let power = self.exponent_ring.smallest_positive_lift(self.exponent_ring.coerce(&StaticRing::<i64>::RING, power)) as usize;
        let mut powers = self.powers.borrow_mut();
        while powers.len() <= power {
            let new = self.ring.mul_ref(powers.last().unwrap(), &powers[1]);
            powers.push(new);
        }
        drop(powers);
        return Ref::map(self.powers.borrow(), |powers| &powers[power]);
    }
}