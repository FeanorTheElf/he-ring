use std::cell::RefCell;

use feanor_math::homomorphism::Homomorphism;
use feanor_math::{algorithms::sqr_mul, rings::zn::zn_64::ZnEl};
use feanor_math::primitive_int::StaticRing;
use feanor_math::ring::*;
use feanor_math::rings::zn::zn_64::Zn;

pub struct Trace {
    Gal: Zn,
    frobenius: ZnEl,
    trace_rank_quo: i64
}

impl Trace {

    pub fn new(Gal: &Zn, p: i64, rank: usize) -> Trace {
        Trace {
            Gal: *Gal,
            frobenius: Gal.can_hom(&StaticRing::<i64>::RING).unwrap().map(p),
            trace_rank_quo: rank as i64
        }
    }

    pub fn evaluate_generic<T, Add, ApplyGalois, Clone>(&self, input: T, add_fn: Add, apply_galois_fn: ApplyGalois, clone: Clone) -> T
        where Add: FnMut(T, &T) -> T,
            ApplyGalois: FnMut(&T, &ZnEl) -> T,
            Clone: FnMut(&T) -> T
    {
        let add_fn = RefCell::new(add_fn);
        let apply_galois_fn = RefCell::new(apply_galois_fn);
        let clone = RefCell::new(clone);
        sqr_mul::generic_pow_shortest_chain_table::<_, _, _, _, _, !>(
            (1, Some(input)), 
            &self.trace_rank_quo, 
            StaticRing::<i64>::RING, 
            |(i, x)| {
                if let Some(x) = x {
                    Ok((2 * i, Some(add_fn.borrow_mut()(apply_galois_fn.borrow_mut()(x, &self.Gal.pow(self.frobenius, *i)), x))))
                } else {
                    Ok((*i, None))
                }
            }, |(i, x), (j, y)| {
                if x.is_none() {
                    return Ok((i + j, y.as_ref().map(|y| clone.borrow_mut()(y))));
                } else if y.is_none() {
                    return Ok((i + j, x.as_ref().map(|x| clone.borrow_mut()(x))));
                }
                Ok((i + j, Some(add_fn.borrow_mut()(apply_galois_fn.borrow_mut()(x.as_ref().unwrap(), &self.Gal.pow(self.frobenius, *j)), y.as_ref().unwrap()))))
            }, 
            |(i, x)| (*i, x.as_ref().map(|x| clone.borrow_mut()(x))), 
            (0, None)
        ).unwrap_or_else(|x| x).1.unwrap()
    }

    pub fn required_galois_keys(&self) -> impl Iterator<Item = ZnEl> {
        let mut result = Vec::new();
        self.evaluate_generic((), |(), ()| (), |(), g| {
            result.push(*g);
            ()
        }, |()| ());
        return result.into_iter();
    }
}
