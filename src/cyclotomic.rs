use feanor_math::ring::*;
use feanor_math::delegate;
use feanor_math::rings::extension::*;

///
/// Trait for rings `R[X]/(Phi_n)`, for a base ring `R`. Note that `Phi_n` is allowed to factor in `R`, hence this ring
/// might not be integral. Furthermore, the residue class of `X`, i.e. the root of unity, must be given by 
/// [`feanor_math::rings::extension::FreeAlgebra::canonical_gen()`].
/// 
/// # Nontrivial automorphisms
/// 
/// See [`feanor_math::rings::extension::FreeAlgebra`].
/// 
/// Note that computing this particular map when given arbitrary isomorphic instances `R`, `S`
/// can be difficult for specific implementations, hence it is not required that for all isomorphic
/// instances `R, S: RingType` with `RingType: CyclotomicRing` we have `R.has_canonical_hom(S).is_some()`.
/// Naturally, it is always required that `R.has_canonical_iso(R).is_some()`.
/// 
/// # Ambiguity in some situations
/// 
/// There is some ambiguity, as for `m` odd, we have `R[X]/(Phi_m) ~ R[X]/(Phi_(2m))` are isomorphic.
/// It is up to implementations which of these representations should be exposed via this trait.
/// Naturally, this should be consistent - i.e. `self.canonical_gen()` should always return
/// a `self.n()`-th root of unity.
/// 
pub trait CyclotomicRing: FreeAlgebra {

    ///
    /// The cyclotomic order, i.e. the multiplicative order of `self.canonical_gen()`.
    /// The degree of this ring extension is `phi(self.n())` where `phi` is Euler's totient
    /// function.
    ///
    fn n(&self) -> u64;

    fn cyclotomic_index_ring(&self) -> zn_64::Zn {
        zn_64::Zn::new(self.n())
    }

    fn apply_galois_action(&self, x: &Self::Element, g: zn_64::ZnEl) -> Self::Element;

    fn apply_galois_action_many<'a>(&'a self, x: &'a Self::Element, gs: &'a [zn_64::ZnEl]) -> impl 'a + ExactSizeIterator<Item = Self::Element> {
        gs.into_iter().map(move |g| self.apply_galois_action(x, *g))
    }
}

///
/// The [`RingStore`] belonging to [`CyclotomicRing`]
/// 
pub trait CyclotomicRingStore: RingStore
    where Self::Type: CyclotomicRing
{
    delegate!{ CyclotomicRing, fn n(&self) -> u64 }
    delegate!{ CyclotomicRing, fn cyclotomic_index_ring(&self) -> zn_64::Zn }
    delegate!{ CyclotomicRing, fn apply_galois_action(&self, el: &El<Self>, s: zn_64::ZnEl) -> El<Self> }
}

impl<R: RingStore> CyclotomicRingStore for R
    where R::Type: CyclotomicRing
{}

#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use feanor_math::primitive_int::*;
#[cfg(test)]
use feanor_math::algorithms;
#[cfg(test)]
use feanor_math::rings::poly::*;
#[cfg(test)]
use feanor_math::rings::poly::sparse_poly::SparsePolyRing;
use feanor_math::rings::zn::zn_64;
#[cfg(test)]
use feanor_math::seq::*;

#[cfg(test)]
pub fn generic_test_cyclotomic_ring_axioms<R: CyclotomicRingStore>(ring: R)
    where R::Type: CyclotomicRing
{
    use feanor_math::homomorphism::Homomorphism;

    let zeta = ring.canonical_gen();
    let n = ring.n();
    
    assert_el_eq!(&ring, &ring.one(), &ring.pow(ring.clone_el(&zeta), n as usize));
    for (p, _) in algorithms::int_factor::factor(&StaticRing::<i64>::RING, n as i64) {
        assert!(!ring.eq_el(&ring.one(), &ring.pow(ring.clone_el(&zeta), n as usize / p as usize)));
    }

    // test minimal polynomial of zeta
    let poly_ring = SparsePolyRing::new(&StaticRing::<i64>::RING, "X");
    let cyclo_poly = algorithms::cyclotomic::cyclotomic_polynomial(&poly_ring, n as usize);

    let x = ring.pow(ring.clone_el(&zeta), ring.rank());
    let x_vec = ring.wrt_canonical_basis(&x);
    assert_eq!(ring.rank(), x_vec.len());
    for i in 0..x_vec.len() {
        assert_el_eq!(ring.base_ring(), &ring.base_ring().negate(ring.base_ring().int_hom().map(*poly_ring.coefficient_at(&cyclo_poly, i) as i32)), &x_vec.at(i));
    }
    assert_el_eq!(&ring, &x, &ring.from_canonical_basis((0..x_vec.len()).map(|i| x_vec.at(i))));

    // test basis wrt_root_of_unity_basis linearity and compatibility from_canonical_basis/wrt_root_of_unity_basis
    for i in (0..ring.rank()).step_by(5) {
        for j in (1..ring.rank()).step_by(7) {
            if i == j {
                continue;
            }
            let element = ring.from_canonical_basis((0..ring.rank()).map(|k| if k == i { ring.base_ring().one() } else if k == j { ring.base_ring().int_hom().map(2) } else { ring.base_ring().zero() }));
            let expected = ring.add(ring.pow(ring.clone_el(&zeta), i), ring.int_hom().mul_map(ring.pow(ring.clone_el(&zeta), j), 2));
            assert_el_eq!(&ring, &expected, &element);
            let element_vec = ring.wrt_canonical_basis(&expected);
            for k in 0..ring.rank() {
                if k == i {
                    assert_el_eq!(ring.base_ring(), &ring.base_ring().one(), &element_vec.at(k));
                } else if k == j {
                    assert_el_eq!(ring.base_ring(), &ring.base_ring().int_hom().map(2), &element_vec.at(k));
                } else {
                    assert_el_eq!(ring.base_ring(), &ring.base_ring().zero(), &element_vec.at(k));
                }
            }
        }
    }
}
