use std::alloc::Allocator;
use std::hash::DefaultHasher;
use std::hash::Hasher;
use std::cmp::max;

use feanor_math::algorithms::discrete_log::discrete_log;
use feanor_math::algorithms::eea::signed_gcd;
use feanor_math::algorithms::int_factor;
use feanor_math::algorithms::unity_root::get_prim_root_of_unity;
use feanor_math::homomorphism::Homomorphism;
use feanor_math::integer::int_cast;
use feanor_math::primitive_int::*;
use feanor_math::ring::*;
use feanor_math::rings::extension::extension_impl::FreeAlgebraImpl;
use feanor_math::rings::finite::FiniteRingStore;
use feanor_math::rings::float_complex::Complex64;
use feanor_math::rings::float_complex::Complex64El;
use feanor_math::rings::zn::zn_64::*;
use feanor_math::rings::zn::*;
use feanor_math::seq::VectorView;
use feanor_math::wrapper::RingElementWrapper;

use crate::cyclotomic::CyclotomicRing;
use oorandom;

use super::complex_fft_ring::CCFFTRingBase;
use super::complex_fft_ring::RingDecomposition;
use super::complex_fft_ring::RingDecompositionSelfIso;

pub fn euler_phi(factorization: &[(i64, usize)]) -> i64 {
    const ZZ: StaticRing<i64> = StaticRing::RING;
    ZZ.prod(factorization.iter().map(|(p, e)| (p - 1) * ZZ.pow(*p, e - 1)))
}

fn get_multiplicative_generator(ring: Zn, factorization: &[(i64, usize)]) -> ZnEl {
    const ZZ: StaticRing<i64> = StaticRing::RING;
    let mut rng = oorandom::Rand64::new(ring.integer_ring().default_hash(ring.modulus()) as u128);
    let order = euler_phi(factorization);
    'test_generator: loop {
        let potential_generator = ring.random_element(|| rng.rand_u64());
        for (p, _) in int_factor::factor(&ZZ, order) {
            if ring.is_one(&ring.pow(potential_generator, (order / p) as usize)) {
                continue 'test_generator;
            }
        }
        return potential_generator;
    }
}

fn unit_group_dlog(ring: &Zn, base: ZnEl, order: i64, value: ZnEl) -> Option<i64> {
    discrete_log(
        RingElementWrapper::new(&ring, value), 
        &RingElementWrapper::new(&ring, base), 
        order, 
        |x, y| x * y, 
        RingElementWrapper::new(&ring, ring.one())
    )
}

pub(super) fn apply_galois_action_base<R: ?Sized + RingBase, F: ?Sized + CyclotomicRingDecomposition<R>>(decomp: &F, src: &[R::Element], dst: &mut [R::Element], galois_element: ZnEl, tmp: &mut [Complex64El], ring: &R) {
    let (tmp_src, tmp_dst) = tmp.split_at_mut(decomp.rank());
    decomp.fft_forward(src, tmp_src, ring);
    decomp.permute_galois_action(tmp_src, tmp_dst, galois_element);
    decomp.fft_backward(tmp_dst, dst, ring);
}

pub trait CyclotomicRingDecomposition<R: ?Sized + RingBase>: RingDecomposition<R> {

    ///
    /// Returns `Z/nZ` such that the galois group of this number ring
    /// is `(Z/nZ)*`
    /// 
    fn galois_group_mulrepr(&self) -> Zn;

    fn permute_galois_action(&self, src: &[Complex64El], dst: &mut [Complex64El], galois_element: ZnEl);
    
    fn apply_galois_action(&self, src: &[R::Element], dst: &mut [R::Element], galois_element: ZnEl, ring: &R) {
        apply_galois_action_base(self, src, dst, galois_element, &mut (0..(2 * self.rank())).map(|_| Complex64::RING.zero()).collect::<Vec<_>>(), ring)
    }
}

pub fn compute_hypercube_structure(n: i64, p: i64) -> (
    Vec<(usize, bool)>,
    Vec<ZnEl>,
    Zn
) {
    const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;

    // the unit group (Z/nZ)* decomposes as X (Z/niZ)*
    let factorization = int_factor::factor(&ZZ, n);
    let Zn_rns = zn_rns::Zn::new(factorization.iter().map(|(q, k)| Zn::new(ZZ.pow(*q, *k) as u64)).collect(), ZZ);
    let Zn = Zn::new(n as u64);
    let iso = Zn.into_can_hom(zn_big::Zn::new(ZZ, n)).ok().unwrap().compose((&Zn_rns).into_can_iso(zn_big::Zn::new(ZZ, n)).ok().unwrap());

    let mut dims = Vec::new();
    let mut gens = Vec::new();
    for (i, (q, k)) in factorization.iter().enumerate() {
        let Zqk = Zn_rns.at(i);
        if *q == 2 {
            // `(Z/2^kZ)*` is an exception, since it is not cyclic
            if *k == 1 {
                continue;
            } else if *k == 2 {
                unimplemented!()
            } else {
                // `(Z/2^kZ)*` is isomorphic to `<g1> x <g2>` where `<g1> ~ Z/2^(k - 2)Z` and `<g2> ~ Z/2Z`
                let g1 = Zqk.int_hom().map(5);
                let ord_g1 = ZZ.pow(*q, *k as usize - 2);
                let g2 = Zqk.can_hom(&ZZ).unwrap().map(*Zqk.modulus() - 1);
                if p % 4 == 1 {
                    // `p` is in `<g1>`
                    let logg1_p = unit_group_dlog(Zqk, g1, ord_g1, Zqk.can_hom(&ZZ).unwrap().map(p)).unwrap();
                    dims.push((2, true));
                    gens.push(iso.map(Zn_rns.from_congruence((0..factorization.len()).map(|j| if i == j { g2 } else { Zn_rns.at(j).one() }))));
                    let ord_p = ord_g1 / signed_gcd(logg1_p, ord_g1, ZZ);
                    if ord_p != ord_g1 {
                        dims.push(((ord_g1 / ord_p) as usize, ord_p == 1));
                        gens.push(iso.map(Zn_rns.from_congruence((0..factorization.len()).map(|j| if i == j { g1 } else { Zn_rns.at(j).one() }))));
                    }
                } else {
                    // `<p, g1> = (Z/2^kZ)*` and `p * g2 in <g1>`
                    let logg1_pg2 = unit_group_dlog(Zqk, g1, ord_g1, Zqk.mul(Zqk.can_hom(&ZZ).unwrap().map(p), g2)).unwrap();
                    let ord_p = max(2, ord_g1 / signed_gcd(logg1_pg2, ord_g1, ZZ));
                    dims.push(((2 * ord_g1 / ord_p) as usize, false));
                    gens.push(iso.map(Zn_rns.from_congruence((0..factorization.len()).map(|j| if i == j { g1 } else { Zn_rns.at(j).one() }))));
                }
            }
        } else {
            // `(Z/q^kZ)*` is cyclic
            let g = get_multiplicative_generator(*Zqk, &[(*q, *k)]);
            let ord_g = euler_phi(&[(*q, *k)]);
            let logg_p = unit_group_dlog(Zqk, g, ord_g, Zqk.can_hom(&ZZ).unwrap().map(p)).unwrap();
            let ord_p = ord_g / signed_gcd(logg_p, ord_g, ZZ);
            if ord_p == ord_g {
                // the hypercube dimension is of length 1, so ignore
                continue;
            }
            if signed_gcd(ord_g / ord_p, ord_p, &ZZ) == 1 {
                // good dimension
                dims.push(((ord_g / ord_p) as usize, true));
                let local_gen = Zqk.pow(g, ord_p as usize);
                gens.push(iso.map(Zn_rns.from_congruence((0..factorization.len()).map(|j| if i == j { local_gen } else { Zn_rns.at(j).one() }))));
            } else {
                dims.push(((ord_g / ord_p) as usize, false));
                gens.push(iso.map(Zn_rns.from_congruence((0..factorization.len()).map(|j| if i == j { g } else { Zn_rns.at(j).one() }))));
            }
        }
    }
    return (dims, gens, Zn);
}

pub struct HypercubeIsomorphism<'a, R, F, A>
    where R: RingStore,
        R::Type: ZnRing,
        F: CyclotomicRingDecomposition<R::Type> + RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    ring: &'a CCFFTRingBase<R, F, A>,
    slot: FreeAlgebraImpl<&'a R, Vec<El<R>>, A>,
    dims: Vec<(usize, /* is good dim? */ bool)>,
    gens: Vec<ZnEl>
}

impl<'a, R, F, A> HypercubeIsomorphism<'a, R, F, A>
    where R: RingStore,
        R::Type: ZnRing,
        F: CyclotomicRingDecomposition<R::Type> + RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone,
        CCFFTRingBase<R, F, A>: CyclotomicRing + /* unfortunately, the type checker is not clever enough to know that this is always the case */ RingExtension<BaseRing = R>
{
}

#[test]
fn test_compute_hypercube_structure_pow2() {
    {
        let (dims, gens, Z_2n) = compute_hypercube_structure(1024, 3);
        assert_eq!(1, dims.len());
        assert_eq!(&(2, false), &dims[0]);
        assert!(unit_group_dlog(&Z_2n, Z_2n.int_hom().map(3), 256, gens[0]).is_none());
        assert!(unit_group_dlog(&Z_2n, Z_2n.int_hom().map(3), 256, Z_2n.pow(gens[0], 2)).is_some());
    }
    {
        let (dims, gens, Z_2n) = compute_hypercube_structure(1024, 23);
        assert_eq!(1, dims.len());
        assert_eq!(&(4, false), &dims[0]);
        assert!(unit_group_dlog(&Z_2n, Z_2n.int_hom().map(23), 128, Z_2n.pow(gens[0], 2)).is_none());
        assert!(unit_group_dlog(&Z_2n, Z_2n.int_hom().map(23), 128, Z_2n.pow(gens[0], 4)).is_some());
    }
    {
        let (dims, gens, Z_2n) = compute_hypercube_structure(1024, 13);
        assert_eq!(1, dims.len());
        assert_eq!(&(2, true), &dims[0]);
        assert!(unit_group_dlog(&Z_2n, Z_2n.int_hom().map(13), 256, gens[0]).is_none());
        assert_eq!(Some(0), unit_group_dlog(&Z_2n, Z_2n.int_hom().map(13), 256, Z_2n.pow(gens[0], 2)));
    }
    {
        let (dims, gens, Z_2n) = compute_hypercube_structure(1024, 17);
        assert_eq!(2, dims.len());
        assert_eq!(&(2, true), &dims[0]);
        assert_eq!(&(4, false), &dims[1]);
        assert!(unit_group_dlog(&Z_2n, Z_2n.int_hom().map(17), 64, gens[0]).is_none());
        assert_eq!(Some(0), unit_group_dlog(&Z_2n, Z_2n.int_hom().map(17), 64, Z_2n.pow(gens[0], 2)));
        assert!(unit_group_dlog(&Z_2n, Z_2n.int_hom().map(17), 64, Z_2n.pow(gens[1], 2)).is_none());
        assert!(unit_group_dlog(&Z_2n, Z_2n.int_hom().map(17), 64, Z_2n.pow(gens[1], 4)).is_some());
    }
}

#[test]
fn test_compute_hypercube_structure_odd() {
    {
        let (dims, gens, Zn) = compute_hypercube_structure(257, 3);
        assert_eq!(0, dims.len());
    }
    {
        let (dims, gens, Zn) = compute_hypercube_structure(257, 11);
        assert_eq!(1, dims.len());
        assert_eq!(&(4, false), &dims[0]);
        assert!(unit_group_dlog(&Zn, Zn.int_hom().map(11), 64, Zn.pow(gens[0], 2)).is_none());
        assert!(unit_group_dlog(&Zn, Zn.int_hom().map(11), 64, Zn.pow(gens[0], 4)).is_some());
    }
    {
        let (dims, gens, Zn) = compute_hypercube_structure(257 * 101, 13);
        assert_eq!(2, dims.len());
        assert_eq!(&(2, false), &dims[0]);
        assert_eq!(&(2, false), &dims[1]);
        assert!(unit_group_dlog(&Zn, Zn.int_hom().map(13), 3200, gens[0]).is_none());
        assert!(unit_group_dlog(&Zn, Zn.int_hom().map(13), 3200, gens[1]).is_none());
    }
}