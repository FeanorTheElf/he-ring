use feanor_math::ring::*;
use feanor_math::rings::zn::zn_64::*;

use super::double_rns_ring::RingDecomposition;

pub trait CyclotomicRingDecomposition<R: ?Sized + RingBase>: RingDecomposition<R> {

    ///
    /// Returns `Z/nZ` such that the galois group of this number ring
    /// is `(Z/nZ)*`
    /// 
    fn galois_group_mulrepr(&self) -> Zn;

    fn permute_galois_action(&self, src: &[R::Element], dst: &mut [R::Element], galois_element: ZnEl);
}