use std::alloc::Global;
use std::fmt::Debug;
use std::ptr::Alignment;
use std::rc::Rc;
use std::cmp::max;
use std::sync::Arc;
use std::time::Instant;

use feanor_math::algorithms::convolution::fft::{FFTRNSBasedConvolution, FFTRNSBasedConvolutionZn};
use feanor_math::algorithms::convolution::STANDARD_CONVOLUTION;
use feanor_math::algorithms::discrete_log::{discrete_log, order};
use feanor_math::algorithms::eea::{signed_gcd, signed_lcm};
use feanor_math::algorithms::int_factor::{factor, is_prime_power};
use feanor_math::algorithms::linsolve::LinSolveRing;
use feanor_math::algorithms::poly_gcd::factor;
use feanor_math::divisibility::{DivisibilityRing, DivisibilityRingStore};
use feanor_math::rings::field::AsFieldBase;
use feanor_math::rings::poly::generic_impls::Isomorphism;
use feanor_math::homomorphism::*;
use feanor_math::assert_el_eq;
use feanor_math::integer::{int_cast, BigIntRing, BigIntRingBase, IntegerRingStore};
use feanor_math::iters::multi_cartesian_product;
use feanor_math::local::PrincipalLocalRing;
use feanor_math::primitive_int::{StaticRing, StaticRingBase};
use feanor_math::rings::extension::extension_impl::{FreeAlgebraImpl, FreeAlgebraImplBase};
use feanor_math::rings::extension::galois_field::GaloisField;
use feanor_math::rings::extension::{FreeAlgebra, FreeAlgebraStore};
use feanor_math::rings::finite::{FiniteRing, FiniteRingStore};
use feanor_math::rings::local::{AsLocalPIR, AsLocalPIRBase};
use feanor_math::rings::poly::dense_poly::DensePolyRing;
use feanor_math::rings::poly::PolyRingStore;
use feanor_math::rings::zn::zn_64::{Zn, ZnBase, ZnEl};
use feanor_math::delegate::DelegateRing;
use feanor_math::ring::*;
use feanor_math::rings::zn::{zn_big, zn_rns, FromModulusCreateableZnRing, ZnReductionMap, ZnRing, ZnRingStore};
use feanor_math::seq::sparse::SparseMapVector;
use feanor_math::seq::*;
use feanor_math::wrapper::RingElementWrapper;

use crate::cyclotomic::{CyclotomicRing, CyclotomicRingStore};
use crate::euler_phi;
use crate::profiling::{log_time, print_all_timings};
use crate::rings::dynconv::DynConvolutionAlgorithm;
use crate::lintransform::CREATE_LINEAR_TRANSFORM_TIME_RECORDER;

use super::decomposition_ring::{DecompositionRing, DecompositionRingBase};
use super::dynconv::DynConvolutionAlgorithmConvolution;
use super::interpolate::FastPolyInterpolation;
use super::odd_cyclotomic::CompositeCyclotomicNumberRing;
use super::pow2_cyclotomic::Pow2CyclotomicNumberRing;

const ZZi64: StaticRing<i64> = StaticRing::RING;

pub fn get_multiplicative_generator(ring: Zn, factorization: &[(i64, usize)]) -> ZnEl {
    assert_eq!(*ring.modulus(), ZZi64.prod(factorization.iter().map(|(p, e)| ZZi64.pow(*p, *e))));
    assert!(*ring.modulus() % 2 == 1, "for even n, Z/nZ* is either equal to Z/(n/2)Z* or not cyclic");
    let mut rng = oorandom::Rand64::new(ring.integer_ring().default_hash(ring.modulus()) as u128);
    let order = euler_phi(factorization);
    let order_factorization = factor(&ZZi64, order);
    'test_generator: loop {
        let potential_generator = ring.random_element(|| rng.rand_u64());
        if !ring.is_unit(&potential_generator) {
            continue 'test_generator;
        }
        for (p, _) in &order_factorization {
            if ring.is_one(&ring.pow(potential_generator, (order / p) as usize)) {
                continue 'test_generator;
            }
        }
        return potential_generator;
    }
}

pub fn unit_group_dlog(ring: &Zn, base: ZnEl, order: i64, value: ZnEl) -> Option<i64> {
    discrete_log(
        RingElementWrapper::new(&ring, value), 
        &RingElementWrapper::new(&ring, base), 
        order, 
        |x, y| x * y, 
        RingElementWrapper::new(&ring, ring.one())
    )
}

#[derive(Clone, Copy)]
pub struct CyclotomicGaloisGroup {
    ring: Zn,
    order: usize
}

impl CyclotomicGaloisGroup {

    pub fn new(n: u64) -> Self {
        Self {
            ring: Zn::new(n),
            order: euler_phi(&factor(ZZi64, n as i64)) as usize
        }
    }

    pub fn identity(&self) -> CyclotomicGaloisGroupEl {
        CyclotomicGaloisGroupEl { value: self.ring.one() }
    }

    pub fn mul(&self, lhs: CyclotomicGaloisGroupEl, rhs: CyclotomicGaloisGroupEl) -> CyclotomicGaloisGroupEl {
        CyclotomicGaloisGroupEl { value: self.ring.mul(lhs.value, rhs.value) }
    }

    pub fn invert(&self, value: CyclotomicGaloisGroupEl) -> CyclotomicGaloisGroupEl {
        CyclotomicGaloisGroupEl { value: self.ring.invert(&value.value).unwrap() }
    }

    pub fn nonnegative_representative(&self, value: CyclotomicGaloisGroupEl) -> usize {
        self.ring.smallest_positive_lift(value.value) as usize
    }

    pub fn from_representative(&self, value: i64) -> CyclotomicGaloisGroupEl {
        self.from_ring_el(self.ring.coerce(&ZZi64, value))
    }

    pub fn from_ring_el(&self, value: ZnEl) -> CyclotomicGaloisGroupEl {
        assert!(self.ring.is_unit(&value));
        CyclotomicGaloisGroupEl { value }
    }

    pub fn negate(&self, value: CyclotomicGaloisGroupEl) -> CyclotomicGaloisGroupEl {
        CyclotomicGaloisGroupEl { value: self.ring.negate(value.value) }
    }

    pub fn prod<I>(&self, it: I) -> CyclotomicGaloisGroupEl
        where I: IntoIterator<Item = CyclotomicGaloisGroupEl>
    {
        it.into_iter().fold(self.identity(), |a, b| self.mul(a, b))
    }

    pub fn pow(&self, base: CyclotomicGaloisGroupEl, power: i64) -> CyclotomicGaloisGroupEl {
        if power >= 0 {
            CyclotomicGaloisGroupEl { value: self.ring.pow(base.value, power as usize) }
        } else {
            self.invert(CyclotomicGaloisGroupEl { value: self.ring.pow(base.value, (-power) as usize) })
        }
    }

    pub fn is_identity(&self, value: CyclotomicGaloisGroupEl) -> bool {
        self.ring.is_one(&value.value)
    }

    pub fn eq_el(&self, lhs: CyclotomicGaloisGroupEl, rhs: CyclotomicGaloisGroupEl) -> bool {
        self.ring.eq_el(&lhs.value, &rhs.value)
    }

    pub fn n(&self) -> usize {
        *self.ring.modulus() as usize
    }

    pub fn to_ring_el(&self, value: CyclotomicGaloisGroupEl) -> ZnEl {
        value.value
    }

    pub fn underlying_ring(&self) -> &Zn {
        &self.ring
    }

    pub fn group_order(&self) -> usize {
        self.order
    }

    pub fn element_order(&self, value: CyclotomicGaloisGroupEl) -> usize {
        order(
            &RingElementWrapper::new(&self.ring, value.value), 
            self.group_order() as i64, 
            |a, b| a * b, 
            RingElementWrapper::new(&self.ring, self.ring.one())
        ) as usize
    }
}

impl Debug for CyclotomicGaloisGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(Z/{}Z)*", self.ring.modulus())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CyclotomicGaloisGroupEl {
    value: El<Zn>
}

///
/// Represents a hypercube, which is a map
/// ```text
///   h: { 0, ..., m1 - 1 } x ... x { 0, ..., mr - 1 } -> (Z/nZ)^*
///                      a1,  ...,  ar                 -> prod_i gi^ai
/// ```
/// such that the composition `(mod <p>) ∘ h` is a bijection.
/// 
/// We use the following notation:
///  - `n` and `p` as above
///  - `d` is the order of `<p>` as subgroup of `(Z/nZ)*`
///  - `mi` is the length of the `i`-th "hypercube dimension" as above
///  - `gi` is the generator of the `i`-th hypercube dimension
/// 
#[derive(Clone)]
pub struct HypercubeStructure {
    galois_group: CyclotomicGaloisGroup,
    p: CyclotomicGaloisGroupEl,
    d: usize,
    ms: Vec<usize>,
    ord_gs: Vec<usize>,
    gs: Vec<CyclotomicGaloisGroupEl>,
    choice: HypercubeTypeData
}

#[derive(Clone)]
pub enum HypercubeTypeData {
    Generic, 
    /// if the hypercube dimensions correspond directly to prime power factors of `n`, 
    /// we store this correspondence here, as it can be used to explicitly work with the
    /// relationship between hypercube dimensions and tensor factors of `Z[𝝵]`
    CyclotomicTensorProductHypercube(Vec<(i64, usize)>)
}

impl HypercubeStructure {

    pub fn new(galois_group: CyclotomicGaloisGroup, p: CyclotomicGaloisGroupEl, d: usize, ms: Vec<usize>, gs: Vec<CyclotomicGaloisGroupEl>) -> Self {
        assert_eq!(ms.len(), gs.len());
        // check order of p
        assert!(galois_group.is_identity(galois_group.pow(p, d as i64)));
        for (factor, _) in factor(ZZi64, d as i64) {
            assert!(!galois_group.is_identity(galois_group.pow(p, d as i64 / factor)));
        }
        // check whether the given values indeed define a bijection modulo `<p>`
        let mut all_elements = multi_cartesian_product(ms.iter().map(|mi| 0..*mi), |idxs| galois_group.prod(idxs.iter().zip(gs.iter()).map(|(i, g)| galois_group.pow(*g, *i as i64))), |_, x| *x)
            .flat_map(|g| (0..d).map(move |i| galois_group.mul(g, galois_group.pow(p, i as i64))))
            .map(|g| galois_group.nonnegative_representative(g))
            .collect::<Vec<_>>();
        all_elements.sort_unstable();
        assert!((1..all_elements.len()).all(|i| all_elements[i - 1] != all_elements[i]), "not a bijection");
        assert_eq!(galois_group.group_order(), all_elements.len());

        return Self {
            galois_group: galois_group,
            p: p,
            d: d,
            ms: ms,
            ord_gs: gs.iter().map(|g| galois_group.element_order(*g)).collect(),
            gs: gs,
            choice: HypercubeTypeData::Generic
        };
    }

    pub fn halevi_shoup_hypercube(galois_group: CyclotomicGaloisGroup, p: i64) -> Self {

        ///
        /// Stores information about a factor in the representation `(Z/nZ)* = (Z/n1Z)* x ... (Z/nrZ)*`
        /// and about `<p> <= (Z/niZ)^*` (considering `p` to be the "orthogonal" projection of `p in (Z/nZ)*`
        /// into `(Z/niZ)*`).
        /// 
        /// The one exception is the case `ni = 2^e`, since `(Z/2^eZ)*` is not cyclic (for `e > 2`).
        /// We then store it as a single factor (if `(Z/2^eZ)* = <p, g>` for some generator `g`) or as
        /// two factors otherwise.
        /// 
        struct HypercubeDimension {
            g_main: ZnEl,
            order_of_projected_p: i64,
            group_order: i64,
            factor_n: (i64, usize)
        }

        let n = galois_group.n() as i64;
        assert!(signed_gcd(n, p, ZZi64) == 1, "n and p must be coprime");

        // the unit group (Z/nZ)* decomposes as X (Z/niZ)*; this gives rise to the natural hypercube structure,
        // although technically many possible hypercube structures are possible
        let mut factorization = factor(ZZi64, n);
        // this makes debugging easier, since we have a canonical order
        factorization.sort_unstable_by_key(|(p, _)| *p);
        let Zn_rns = zn_rns::Zn::new(factorization.iter().map(|(q, k)| Zn::new(ZZi64.pow(*q, *k) as u64)).collect(), ZZi64);
        let Zn = Zn::new(n as u64);
        let iso = Zn.into_can_hom(zn_big::Zn::new(ZZi64, n)).ok().unwrap().compose((&Zn_rns).into_can_iso(zn_big::Zn::new(ZZi64, n)).ok().unwrap());
        let from_crt = |index: usize, value: ZnEl| iso.map(Zn_rns.from_congruence((0..factorization.len()).map(|j| if j == index { value } else { Zn_rns.at(j).one() })));

        let mut dimensions = Vec::new();
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
                    let ord_g1 = ZZi64.pow(*q, *k as usize - 2);
                    let g2 = Zqk.can_hom(&ZZi64).unwrap().map(-1);
                    if p % 4 == 1 {
                        // `p` is in `<g1>`
                        let logg1_p = unit_group_dlog(Zqk, g1, ord_g1, Zqk.can_hom(&ZZi64).unwrap().map(p)).unwrap();
                        dimensions.push(HypercubeDimension {
                            order_of_projected_p: ord_g1 / signed_gcd(logg1_p, ord_g1, &ZZi64), 
                            group_order: ord_g1,
                            g_main: from_crt(i, g1),
                            factor_n: (2, *k),
                        });
                        dimensions.push(HypercubeDimension {
                            order_of_projected_p: 1, 
                            group_order: 2,
                            g_main: from_crt(i, g2),
                            factor_n: (2, *k),
                        });
                    } else {
                        // `<p, g1> = (Z/2^kZ)*` and `p * g2 in <g1>`
                        let logg1_pg2 = unit_group_dlog(Zqk, g1, ord_g1, Zqk.mul(Zqk.can_hom(&ZZi64).unwrap().map(p), g2)).unwrap();
                        dimensions.push(HypercubeDimension {
                            order_of_projected_p: max(ord_g1 / signed_gcd(logg1_pg2, ord_g1, &ZZi64), 2),
                            group_order: 2 * ord_g1,
                            g_main: from_crt(i, g1),
                            factor_n: (2, *k)
                        });
                    }
                }
            } else {
                // `(Z/q^kZ)*` is cyclic
                let g = get_multiplicative_generator(*Zqk, &[(*q, *k)]);
                let ord_g = euler_phi(&[(*q, *k)]);
                let logg_p = unit_group_dlog(Zqk, g, ord_g, Zqk.can_hom(&ZZi64).unwrap().map(p)).unwrap();
                let ord_p = ord_g / signed_gcd(logg_p, ord_g, ZZi64);
                dimensions.push(HypercubeDimension {
                    order_of_projected_p: ord_p, 
                    group_order: ord_g,
                    g_main: from_crt(i, g),
                    factor_n: (*q, *k)
                });
            }
        }

        dimensions.sort_by_key(|dim| -(dim.order_of_projected_p as i64));
        let mut current_d = 1;
        let lengths = dimensions.iter().map(|dim| {
            let new_d = signed_lcm(current_d, dim.order_of_projected_p as i64, ZZi64);
            let len = dim.group_order as i64 / (new_d / current_d);
            current_d = new_d;
            return len as usize;
        }).collect::<Vec<_>>();

        let mut result = Self::new(
            galois_group,
            galois_group.from_representative(p),
            current_d as usize,
            lengths,
            dimensions.iter().map(|dim| galois_group.from_ring_el(dim.g_main)).collect()
        );
        result.choice = HypercubeTypeData::CyclotomicTensorProductHypercube(dimensions.iter().map(|dim| dim.factor_n).collect());
        return result;
    }

    pub fn map_1d(&self, dim_idx: usize, steps: i64) -> CyclotomicGaloisGroupEl {
        assert!(dim_idx < self.dim_count());
        self.galois_group.pow(self.gs[dim_idx], steps)
    }

    pub fn map(&self, idxs: &[i64]) -> CyclotomicGaloisGroupEl {
        assert_eq!(self.ms.len(), idxs.len());
        self.galois_group.prod(idxs.iter().zip(self.gs.iter()).map(|(i, g)| self.galois_group.pow(*g, *i)))
    }

    pub fn map_usize(&self, idxs: &[usize]) -> CyclotomicGaloisGroupEl {
        assert_eq!(self.ms.len(), idxs.len());
        self.galois_group.prod(idxs.iter().zip(self.gs.iter()).map(|(i, g)| self.galois_group.pow(*g, *i as i64)))
    }

    pub fn is_tensor_product_compatible(&self) -> bool {
        match self.choice {
            HypercubeTypeData::CyclotomicTensorProductHypercube(_) => true,
            HypercubeTypeData::Generic => false
        }
    }

    pub fn factor_of_n(&self, dim_idx: usize) -> Option<i64> {
        assert!(dim_idx < self.dim_count());
        match &self.choice {
            HypercubeTypeData::CyclotomicTensorProductHypercube(factors_n) => Some(ZZi64.pow(factors_n[dim_idx].0, factors_n[dim_idx].1)),
            HypercubeTypeData::Generic => None
        }
    }

    pub fn p(&self) -> CyclotomicGaloisGroupEl {
        self.p
    }

    pub fn frobenius(&self, power: i64) -> CyclotomicGaloisGroupEl {
        self.galois_group.pow(self.p(), power)
    }

    pub fn d(&self) -> usize {
        self.d
    }

    pub fn m(&self, i: usize) -> usize {
        assert!(i < self.ms.len());
        self.ms[i]
    }

    pub fn g(&self, i: usize) -> CyclotomicGaloisGroupEl {
        assert!(i < self.ms.len());
        self.gs[i]
    }

    pub fn ord_g(&self, i: usize) -> usize {
        assert!(i < self.ms.len());
        self.ord_gs[i]
    }

    pub fn n(&self) -> usize {
        self.galois_group().n()
    }

    pub fn dim_count(&self) -> usize {
        self.gs.len()
    }

    pub fn galois_group(&self) -> &CyclotomicGaloisGroup {
        &self.galois_group
    }

    pub fn element_count(&self) -> usize {
        ZZi64.prod(self.ms.iter().map(|mi| *mi as i64)) as usize
    }

    pub fn hypercube_iter<'b, G, T>(&'b self, for_slot: G) -> impl ExactSizeIterator<Item = T> + use<'b, G, T>
        where G: 'b + Clone + FnMut(&[usize]) -> T,
            T: 'b
    {
        let mut it = multi_cartesian_product(
            self.ms.iter().map(|l| (0..*l)),
            for_slot,
            |_, x| *x
        );
        (0..self.element_count()).map(move |_| it.next().unwrap())
    }

    pub fn element_iter<'b>(&'b self) -> impl ExactSizeIterator<Item = CyclotomicGaloisGroupEl> + use<'b> {
        self.hypercube_iter(|idxs| self.map_usize(idxs))
    }
}

fn get_prim_root_of_unity<R>(ring: R, m: usize) -> El<R>
    where R: RingStore,
        R::Type: FiniteRing + FreeAlgebra + DivisibilityRing,
        <<R::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing + ZnRing + CanHomFrom<StaticRingBase<i64>>
{
    let (p, e) = is_prime_power(&ZZi64, &ring.characteristic(&ZZi64).unwrap()).unwrap();
    let max_log2_len = ZZi64.abs_log2_ceil(&(ring.rank() as i64)).unwrap() + 1;
    let convolution: FFTRNSBasedConvolutionZn = FFTRNSBasedConvolutionZn::from(FFTRNSBasedConvolution::new_with(max_log2_len, BigIntRing::RING, Global));
    let galois_field = GaloisField::new_with(Zn::new(p as u64).as_field().ok().unwrap(), ring.rank(), Global, convolution);

    let rou = feanor_math::algorithms::unity_root::get_prim_root_of_unity(&galois_field, m).unwrap();

    let red_map = ZnReductionMap::new(ring.base_ring(), galois_field.base_ring()).unwrap();
    let mut result = ring.from_canonical_basis(galois_field.wrt_canonical_basis(&rou).into_iter().map(|x| red_map.smallest_lift(x)));

    // perform hensel lifting
    for _ in 0..e {
        let delta = ring.checked_div(
            &ring.sub(ring.pow(ring.clone_el(&result), m), ring.one()),
            &ring.inclusion().mul_map(ring.pow(ring.clone_el(&result), m - 1), ring.base_ring().coerce(&ZZi64, m as i64)) 
        ).unwrap();
        ring.sub_assign(&mut result, delta);
    }
    assert!(ring.is_one(&ring.pow(ring.clone_el(&result), m)));
    return result;
}

pub type SlotRingOver<R> = AsLocalPIR<FreeAlgebraImpl<R, Vec<El<R>>, Global, DynConvolutionAlgorithmConvolution<<R as RingStore>::Type, Arc<dyn Send + Sync + DynConvolutionAlgorithm<<R as RingStore>::Type>>>>>;
pub type SlotRingOf<R> = SlotRingOver<RingValue<BaseRing<R>>>;

pub type DefaultHypercube<'a, NumberRing, A = Global> = HypercubeIsomorphism<&'a DecompositionRing<NumberRing, Zn, A>>;

pub type BaseRing<R> = <<<R as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type;
pub type DecoratedBaseRing<R> = AsLocalPIR<RingValue<BaseRing<R>>>;

///
/// Represents the isomorphism
/// ```text
///   Fp[X]/(Phi_n(X)) -> F_(p^d)^((Z/nZ)*/<p>)
/// ```
/// where `d` is the order of `p` in `(Z/nZ)*`.
/// 
pub struct HypercubeIsomorphism<R>
    where R: RingStore,
        R::Type: CyclotomicRing,
        BaseRing<R>: Clone + ZnRing + CanHomFrom<StaticRingBase<i64>> + CanHomFrom<BigIntRingBase> + LinSolveRing + FromModulusCreateableZnRing,
        AsFieldBase<DecoratedBaseRing<R>>: CanIsoFromTo<<DecoratedBaseRing<R> as RingStore>::Type> + SelfIso
{
    ring: R,
    e: usize,
    slot_rings: Vec<SlotRingOf<R>>,
    slot_to_ring_interpolation: FastPolyInterpolation<DensePolyRing<DecoratedBaseRing<R>, Global>>,
    hypercube_structure: HypercubeStructure,
}

impl<R> HypercubeIsomorphism<R>
    where R: RingStore,
        R::Type: CyclotomicRing,
        <<R::Type as RingExtension>::BaseRing as RingStore>::Type: Clone + ZnRing + CanHomFrom<StaticRingBase<i64>> + CanHomFrom<BigIntRingBase> + LinSolveRing + FromModulusCreateableZnRing,
        AsFieldBase<DecoratedBaseRing<R>>: CanIsoFromTo<<DecoratedBaseRing<R> as RingStore>::Type> + SelfIso
{
    pub fn new<const LOG: bool>(ring: R, hypercube_structure: HypercubeStructure) -> Self {
        let n = ring.n() as usize;
        let (p, e) = is_prime_power(&ZZi64, &ring.characteristic(&ZZi64).unwrap()).unwrap();
        let d = hypercube_structure.d();
        let max_log2_len = ZZi64.abs_log2_ceil(&(d as i64)).unwrap() + 1;
        let galois_group = hypercube_structure.galois_group();
        assert_eq!(n, galois_group.n());
        assert!(galois_group.eq_el(hypercube_structure.p(), galois_group.from_representative(p)));
        
        let tmp_slot_ring = log_time::<_, _, LOG, _>("[HypercubeIsomorphism::new] Creating temporary slot ring", |[]| {
            let convolution: FFTRNSBasedConvolutionZn = FFTRNSBasedConvolutionZn::from(FFTRNSBasedConvolution::new_with(max_log2_len, BigIntRing::RING, Global));
            let base_ring = Zn::new(p as u64).as_field().ok().unwrap();
            GaloisField::new_with(base_ring, d, Global, STANDARD_CONVOLUTION).get_ring()
                .galois_ring_with(AsLocalPIR::from_zn(RingRef::new(ring.base_ring().get_ring())).unwrap(), Global, convolution)
        });

        let root_of_unity = log_time::<_, _, LOG, _>("[HypercubeIsomorphism::new] Computing root of unity", |[]| 
            get_prim_root_of_unity(&tmp_slot_ring, n)
        );

        // let poly_ring_convolution: FFTRNSBasedConvolutionZn = FFTRNSBasedConvolutionZn::from(FFTRNSBasedConvolution::new_with(ZZi64.abs_log2_ceil(&(n as i64)).unwrap() + 1, BigIntRing::RING, Global));
        let decorated_base_ring: DecoratedBaseRing<R> = AsLocalPIR::from_zn(RingValue::from(ring.base_ring().get_ring().clone())).unwrap();
        let base_poly_ring = DensePolyRing::new_with(decorated_base_ring, "X", Global, STANDARD_CONVOLUTION);
        let slot_ring_moduli = log_time::<_, _, LOG, _>("[HypercubeIsomorphism::new] Computing factorization of cyclotomic polynomial", |[]| {
            let poly_ring = DensePolyRing::new(&tmp_slot_ring, "X");
            let mut slot_ring_moduli = Vec::new();
            for g in hypercube_structure.element_iter() {
                let mut result = poly_ring.prod((0..d).scan(
                    tmp_slot_ring.pow(tmp_slot_ring.clone_el(&root_of_unity), galois_group.nonnegative_representative(galois_group.invert(g))), 
                    |current_root_of_unity, _| {
                        let result = poly_ring.sub(poly_ring.indeterminate(), poly_ring.inclusion().map_ref(current_root_of_unity));
                        *current_root_of_unity = tmp_slot_ring.pow(tmp_slot_ring.clone_el(current_root_of_unity), p as usize);
                        return Some(result);
                    }
                ));
                let normalization_factor = poly_ring.base_ring().invert(poly_ring.lc(&result).unwrap()).unwrap();
                poly_ring.inclusion().mul_assign_map(&mut result, normalization_factor);

                slot_ring_moduli.push(base_poly_ring.from_terms(poly_ring.terms(&result).map(|(c, i)| {
                    let c_wrt_basis = tmp_slot_ring.wrt_canonical_basis(c);
                    debug_assert!(c_wrt_basis.iter().skip(1).all(|c| tmp_slot_ring.base_ring().is_zero(&c)));
                    return (base_poly_ring.base_ring().get_ring().rev_delegate(tmp_slot_ring.base_ring().get_ring().delegate(c_wrt_basis.at(0))), i);
                })));
            }
            slot_ring_moduli
        });

        let slot_ring_convolution = Self::create_slot_ring_convolution(d);
        let slot_rings = log_time::<_, _, LOG, _>("[HypercubeIsomorphism::new] Computing slot rings", |[]| slot_ring_moduli.iter().map(|f| {
            let modulus = (0..d).map(|i| base_poly_ring.base_ring().get_ring().delegate(base_poly_ring.base_ring().negate(base_poly_ring.base_ring().clone_el(base_poly_ring.coefficient_at(f, i))))).collect::<Vec<_>>();
            let slot_ring = FreeAlgebraImpl::new_with(
                RingValue::from(ring.base_ring().get_ring().clone()),
                d,
                modulus,
                "𝝵",
                Global,
                slot_ring_convolution.clone()
            );
            let max_ideal_gen = slot_ring.inclusion().map(slot_ring.base_ring().coerce(&ZZi64, p));
            return AsLocalPIR::from(AsLocalPIRBase::promise_is_local_pir(slot_ring, max_ideal_gen, Some(e)));
        }).collect::<Vec<_>>());

        let interpolation = log_time::<_, _, LOG, _>("[HypercubeIsomorphism::new] Computing interpolation data", |[]|
            FastPolyInterpolation::new(base_poly_ring, slot_ring_moduli)
        );

        return Self {
            hypercube_structure: hypercube_structure,
            ring: ring,
            e: e,
            slot_to_ring_interpolation: interpolation,
            slot_rings: slot_rings
        };
    }

    pub fn change_modulus<RNew>(&self, new_ring: RNew) -> HypercubeIsomorphism<RNew>
        where RNew: RingStore,
            RNew::Type: CyclotomicRing,
            BaseRing<RNew>: Clone + ZnRing + CanHomFrom<StaticRingBase<i64>> + CanHomFrom<BigIntRingBase> + LinSolveRing + FromModulusCreateableZnRing,
            AsFieldBase<DecoratedBaseRing<RNew>>: CanIsoFromTo<<DecoratedBaseRing<RNew> as RingStore>::Type> + SelfIso
    {
        let (p, e) = is_prime_power(&ZZi64, &new_ring.characteristic(&ZZi64).unwrap()).unwrap();
        let d = self.hypercube().d();
        let slot_ring_convolution = HypercubeIsomorphism::<RNew>::create_slot_ring_convolution(d);
        let red_map = ZnReductionMap::new(self.ring().base_ring(), new_ring.base_ring()).unwrap();
        let poly_ring = DensePolyRing::new(new_ring.base_ring(), "X");
        let slot_rings = self.slot_rings.iter().map(|slot_ring| {
            let gen_poly = slot_ring.generating_poly(&poly_ring, &red_map);
            let new_slot_ring = FreeAlgebraImpl::new_with(
                RingValue::from(new_ring.base_ring().get_ring().clone()),
                d,
                (0..d).map(|i| new_ring.base_ring().negate(new_ring.base_ring().clone_el(poly_ring.coefficient_at(&gen_poly, i)))).collect::<Vec<_>>(),
                "𝝵",
                Global,
                slot_ring_convolution.clone()
            );
            let max_ideal_gen = new_slot_ring.inclusion().map(new_slot_ring.base_ring().coerce(&ZZi64, p));
            return AsLocalPIR::from(AsLocalPIRBase::promise_is_local_pir(new_slot_ring, max_ideal_gen, Some(e)));
        }).collect::<Vec<_>>();

        let decorated_base_ring: DecoratedBaseRing<RNew> = AsLocalPIR::from_zn(RingValue::from(new_ring.base_ring().get_ring().clone())).unwrap();
        let base_poly_ring = DensePolyRing::new_with(decorated_base_ring, "X", Global, STANDARD_CONVOLUTION);
        return HypercubeIsomorphism {
            slot_to_ring_interpolation: self.slot_to_ring_interpolation.change_modulus(base_poly_ring),
            e: e,
            hypercube_structure: self.hypercube().clone(),
            ring: new_ring,
            slot_rings: slot_rings,
        };
    }

    fn create_slot_ring_convolution(d: usize) -> DynConvolutionAlgorithmConvolution<<<R::Type as RingExtension>::BaseRing as RingStore>::Type, Arc<dyn Send + Sync + DynConvolutionAlgorithm<<<R::Type as RingExtension>::BaseRing as RingStore>::Type>>> {
        let max_log2_len = ZZi64.abs_log2_ceil(&(d as i64)).unwrap() + 1;
        DynConvolutionAlgorithmConvolution::new(Arc::new(STANDARD_CONVOLUTION))
    }

    pub fn hypercube(&self) -> &HypercubeStructure {
        &self.hypercube_structure
    }

    pub fn ring(&self) -> &R {
        &self.ring
    }

    pub fn slot_ring_at<'a>(&'a self, i: usize) -> &'a SlotRingOf<R>
        where R: 'a
    {
        &self.slot_rings[i]
    }

    pub fn slot_ring<'a>(&'a self) -> &'a SlotRingOf<R>
        where R: 'a
    {
        self.slot_ring_at(0)
    }

    pub fn p(&self) -> i64 {
        self.galois_group().nonnegative_representative(self.hypercube_structure.p()) as i64
    }

    pub fn e(&self) -> usize {
        self.e
    }

    pub fn d(&self) -> usize {
        self.hypercube_structure.d()
    }

    pub fn galois_group(&self) -> &CyclotomicGaloisGroup {
        self.hypercube_structure.galois_group()
    }

    pub fn slot_count(&self) -> usize {
        self.hypercube_structure.element_count()
    }
    
    pub fn get_slot_value(&self, el: &El<R>, slot_index: CyclotomicGaloisGroupEl) -> El<SlotRingOf<R>> {
        let el = self.ring().apply_galois_action(el, self.galois_group().to_ring_el(self.galois_group().invert(slot_index)));
        let poly_ring = DensePolyRing::new(self.ring.base_ring(), "X");
        let el_as_poly = self.ring().poly_repr(&poly_ring, &el, self.ring.base_ring().identity());
        let poly_modulus = self.slot_ring().generating_poly(&poly_ring, self.ring.base_ring().identity());
        let (_, rem) = poly_ring.div_rem_monic(el_as_poly, &poly_modulus);
        self.slot_ring().from_canonical_basis((0..self.d()).map(|i| poly_ring.base_ring().clone_el(poly_ring.coefficient_at(&rem, i))))
    }

    pub fn get_slot_values<'a>(&'a self, el: &'a El<R>) -> impl ExactSizeIterator<Item = El<SlotRingOf<R>>> + use<'a, R> {
        self.hypercube_structure.element_iter().map(move |g| self.get_slot_value(el, g))
    }

    pub fn from_slot_values<'a, I>(&self, values: I) -> El<R>
        where I: IntoIterator<Item = El<SlotRingOf<R>>>
    {
        record_time!(CREATE_LINEAR_TRANSFORM_TIME_RECORDER, "from_slot_values", || {
            let poly_ring = self.slot_to_ring_interpolation.poly_ring();
            let first_slot_ring: &SlotRingOf<R> = self.slot_ring();
            let mut values_it = values.into_iter();
            let wrap = LambdaHom::new(first_slot_ring.base_ring(), poly_ring.base_ring(), |from, to, x| to.get_ring().rev_delegate(from.clone_el(x)));
            let unwrap = LambdaHom::new(poly_ring.base_ring(), first_slot_ring.base_ring(), |from, _to, x| from.get_ring().delegate(from.clone_el(x)));

            let remainders = record_time!(CREATE_LINEAR_TRANSFORM_TIME_RECORDER, "prepare_remainders", || values_it.by_ref().zip(self.hypercube_structure.element_iter()).enumerate().map(|(i, (a, g))| {
                let f = first_slot_ring.poly_repr(&poly_ring, &a, &wrap);
                let local_slot_ring = self.slot_ring_at(i);
                let image_zeta = local_slot_ring.pow(local_slot_ring.canonical_gen(), self.galois_group().nonnegative_representative(g));
                return local_slot_ring.poly_repr(&poly_ring, &poly_ring.evaluate(&f, &image_zeta, local_slot_ring.inclusion().compose(&unwrap)), &wrap);
            }).collect::<Vec<_>>());
            assert!(values_it.next().is_none(), "iterator should only have {} elements", self.slot_count());
            debug_assert!(remainders.iter().all(|r| poly_ring.degree(r).unwrap_or(0) < self.d()));

            let unreduced_result = record_time!(CREATE_LINEAR_TRANSFORM_TIME_RECORDER, "interpolate_unreduced", || self.slot_to_ring_interpolation.interpolate_unreduced(remainders));
            let unreduced_result = (0..=poly_ring.degree(&unreduced_result).unwrap_or(0)).map(|i| poly_ring.base_ring().clone_el(poly_ring.coefficient_at(&unreduced_result, i))).collect::<Vec<_>>();

            let canonical_gen_pow_rank = self.ring().mul(self.ring().canonical_gen(), self.ring().from_canonical_basis((1..self.ring().rank()).map(|_| self.ring().base_ring().zero()).chain([self.ring().base_ring().one()].into_iter())));
            let mut current = self.ring().one();
            return record_time!(CREATE_LINEAR_TRANSFORM_TIME_RECORDER, "sum_result", || <_ as RingStore>::sum(&self.ring, unreduced_result.chunks(self.ring.rank()).map(|chunk| self.ring.from_canonical_basis(
                chunk.iter().map(|a| poly_ring.base_ring().clone_el(a)).chain((0..(self.ring.rank() - chunk.len())).map(|_| poly_ring.base_ring().zero()))
                    .map(|x| unwrap.map(x))
            )).map(|x| {
                let result = self.ring().mul_ref_snd(x, &current);
                self.ring().mul_assign_ref(&mut current, &canonical_gen_pow_rank);
                return result;
            })));
        })
    }
}

#[cfg(test)]
fn test_ring1() -> (DecompositionRing<Pow2CyclotomicNumberRing, Zn>, HypercubeStructure) {
    let galois_group = CyclotomicGaloisGroup::new(32);
    let hypercube_structure = HypercubeStructure::new(
        galois_group,
        galois_group.from_representative(7),
        4,
        vec![4],
        vec![galois_group.from_representative(5)]
    );
    let ring = DecompositionRingBase::new(Pow2CyclotomicNumberRing::new(32), Zn::new(7));
    return (ring, hypercube_structure);
}

#[cfg(test)]
fn test_ring2() -> (DecompositionRing<Pow2CyclotomicNumberRing, Zn>, HypercubeStructure) {
    let galois_group = CyclotomicGaloisGroup::new(32);
    let hypercube_structure = HypercubeStructure::new(
        galois_group,
        galois_group.from_representative(17),
        2,
        vec![4, 2],
        vec![galois_group.from_representative(5), galois_group.from_representative(-1)]
    );
    let ring = DecompositionRingBase::new(Pow2CyclotomicNumberRing::new(32), Zn::new(17));
    return (ring, hypercube_structure);
}

#[cfg(test)]
fn test_ring3() -> (DecompositionRing<CompositeCyclotomicNumberRing, Zn>, HypercubeStructure) {
    let galois_group = CyclotomicGaloisGroup::new(11 * 13);
    let hypercube_structure = HypercubeStructure::new(
        galois_group,
        galois_group.from_representative(3),
        15,
        vec![2, 4],
        vec![galois_group.from_representative(79), galois_group.from_representative(67)]
    );
    let ring = DecompositionRingBase::new(CompositeCyclotomicNumberRing::new(11, 13), Zn::new(3));
    return (ring, hypercube_structure);
}

#[test]
fn test_hypercube_isomorphism_from_to_slot_vector() {
    let mut rng = oorandom::Rand64::new(1);

    let (ring, hypercube) = test_ring1();
    let isomorphism = HypercubeIsomorphism::new::<true>(ring, hypercube);
    assert_eq!(4, isomorphism.slot_count());
    for _ in 0..10 {
        let slot_ring = isomorphism.slot_ring();
        let expected = (0..isomorphism.slot_count()).map(|_| slot_ring.random_element(|| rng.rand_u64())).collect::<Vec<_>>();
        let element = isomorphism.from_slot_values(expected.iter().map(|a| slot_ring.clone_el(a)));
        let actual = isomorphism.get_slot_values(&element);
        for (expected, actual) in expected.iter().zip(actual) {
            assert_el_eq!(slot_ring, expected, actual);
        }
    }

    let (ring, hypercube) = test_ring2();
    let isomorphism = HypercubeIsomorphism::new::<true>(ring, hypercube);
    assert_eq!(8, isomorphism.slot_count());
    for _ in 0..10 {
        let slot_ring = isomorphism.slot_ring();
        let expected = (0..isomorphism.slot_count()).map(|_| slot_ring.random_element(|| rng.rand_u64())).collect::<Vec<_>>();
        let element = isomorphism.from_slot_values(expected.iter().map(|a| slot_ring.clone_el(a)));
        let actual = isomorphism.get_slot_values(&element);
        for (expected, actual) in expected.iter().zip(actual) {
            assert_el_eq!(slot_ring, expected, actual);
        }
    }

    let (ring, hypercube) = test_ring3();
    let isomorphism = HypercubeIsomorphism::new::<true>(ring, hypercube);
    assert_eq!(8, isomorphism.slot_count());
    for _ in 0..10 {
        let slot_ring = isomorphism.slot_ring();
        let expected = (0..isomorphism.slot_count()).map(|_| slot_ring.random_element(|| rng.rand_u64())).collect::<Vec<_>>();
        let element = isomorphism.from_slot_values(expected.iter().map(|a| slot_ring.clone_el(a)));
        let actual = isomorphism.get_slot_values(&element);
        for (expected, actual) in expected.iter().zip(actual) {
            assert_el_eq!(slot_ring, expected, actual);
        }
    }
}

#[test]
fn test_hypercube_isomorphism_is_isomorphic() {
    let mut rng = oorandom::Rand64::new(1);

    let (ring, hypercube) = test_ring1();
    let isomorphism = HypercubeIsomorphism::new::<true>(ring, hypercube);
    for _ in 0..10 {
        let slot_ring = isomorphism.slot_ring();
        let lhs = (0..isomorphism.slot_count()).map(|_| slot_ring.random_element(|| rng.rand_u64())).collect::<Vec<_>>();
        let rhs = (0..isomorphism.slot_count()).map(|_| slot_ring.random_element(|| rng.rand_u64())).collect::<Vec<_>>();
        let expected = (0..isomorphism.slot_count()).map(|i| slot_ring.mul_ref(&lhs[i], &rhs[i])).collect::<Vec<_>>();
        let element = isomorphism.ring().mul(
            isomorphism.from_slot_values(lhs.iter().map(|a| slot_ring.clone_el(a))),
            isomorphism.from_slot_values(rhs.iter().map(|a| slot_ring.clone_el(a)))
        );
        let actual = isomorphism.get_slot_values(&element);
        for (expected, actual) in expected.iter().zip(actual) {
            assert_el_eq!(slot_ring, expected, actual);
        }
    }

    let (ring, hypercube) = test_ring2();
    let isomorphism = HypercubeIsomorphism::new::<true>(ring, hypercube);
    for _ in 0..10 {
        let slot_ring = isomorphism.slot_ring();
        let lhs = (0..isomorphism.slot_count()).map(|_| slot_ring.random_element(|| rng.rand_u64())).collect::<Vec<_>>();
        let rhs = (0..isomorphism.slot_count()).map(|_| slot_ring.random_element(|| rng.rand_u64())).collect::<Vec<_>>();
        let expected = (0..isomorphism.slot_count()).map(|i| slot_ring.mul_ref(&lhs[i], &rhs[i])).collect::<Vec<_>>();
        let element = isomorphism.ring().mul(
            isomorphism.from_slot_values(lhs.iter().map(|a| slot_ring.clone_el(a))),
            isomorphism.from_slot_values(rhs.iter().map(|a| slot_ring.clone_el(a)))
        );
        let actual = isomorphism.get_slot_values(&element);
        for (expected, actual) in expected.iter().zip(actual) {
            assert_el_eq!(slot_ring, expected, actual);
        }
    }

    let (ring, hypercube) = test_ring3();
    let isomorphism = HypercubeIsomorphism::new::<true>(ring, hypercube);
    for _ in 0..10 {
        let slot_ring = isomorphism.slot_ring();
        let lhs = (0..isomorphism.slot_count()).map(|_| slot_ring.random_element(|| rng.rand_u64())).collect::<Vec<_>>();
        let rhs = (0..isomorphism.slot_count()).map(|_| slot_ring.random_element(|| rng.rand_u64())).collect::<Vec<_>>();
        let expected = (0..isomorphism.slot_count()).map(|i| slot_ring.mul_ref(&lhs[i], &rhs[i])).collect::<Vec<_>>();
        let element = isomorphism.ring().mul(
            isomorphism.from_slot_values(lhs.iter().map(|a| slot_ring.clone_el(a))),
            isomorphism.from_slot_values(rhs.iter().map(|a| slot_ring.clone_el(a)))
        );
        let actual = isomorphism.get_slot_values(&element);
        for (expected, actual) in expected.iter().zip(actual) {
            assert_el_eq!(slot_ring, expected, actual);
        }
    }
}

#[test]
fn test_hypercube_isomorphism_rotation() {
    let mut rng = oorandom::Rand64::new(1);

    let (ring, hypercube) = test_ring1();
    let isomorphism = HypercubeIsomorphism::new::<true>(ring, hypercube);
    let ring = isomorphism.ring();
    let hypercube = isomorphism.hypercube();
    for _ in 0..10 {
        let slot_ring = isomorphism.slot_ring();
        let a = slot_ring.random_element(|| rng.rand_u64());

        let mut input = (0..isomorphism.slot_count()).map(|_| slot_ring.zero()).collect::<Vec<_>>();
        input[0] = slot_ring.clone_el(&a);

        let mut expected = (0..isomorphism.slot_count()).map(|_| slot_ring.zero()).collect::<Vec<_>>();
        expected[hypercube.m(0) - 1] = slot_ring.clone_el(&a);

        let actual = ring.apply_galois_action(
            &isomorphism.from_slot_values(input.into_iter()),
            hypercube.galois_group().to_ring_el(hypercube.galois_group().pow(hypercube.g(0), hypercube.m(0) as i64 - 1))
        );
        let actual = isomorphism.get_slot_values(&actual);
        for (expected, actual) in expected.iter().zip(actual) {
            assert_el_eq!(slot_ring, expected, actual);
        }
    }

    let (ring, hypercube) = test_ring2();
    let isomorphism = HypercubeIsomorphism::new::<true>(ring, hypercube);
    let ring = isomorphism.ring();
    let hypercube = isomorphism.hypercube();
    for _ in 0..10 {
        let slot_ring = isomorphism.slot_ring();
        let a = slot_ring.random_element(|| rng.rand_u64());
        
        let mut input = (0..isomorphism.slot_count()).map(|_| slot_ring.zero()).collect::<Vec<_>>();
        input[0] = slot_ring.clone_el(&a);

        let mut expected = (0..isomorphism.slot_count()).map(|_| slot_ring.zero()).collect::<Vec<_>>();
        expected[(hypercube.m(0) - 1) * hypercube.m(1)] = slot_ring.clone_el(&a);

        let actual = ring.apply_galois_action(
            &isomorphism.from_slot_values(input.into_iter()),
            hypercube.galois_group().to_ring_el(hypercube.galois_group().pow(hypercube.g(0), hypercube.m(0) as i64 - 1))
        );
        let actual = isomorphism.get_slot_values(&actual).collect::<Vec<_>>();
        for (expected, actual) in expected.iter().zip(actual.iter()) {
            assert_el_eq!(slot_ring, expected, actual);
        }
    }

    let (ring, hypercube) = test_ring3();
    let isomorphism = HypercubeIsomorphism::new::<true>(ring, hypercube);
    let ring = isomorphism.ring();
    let hypercube = isomorphism.hypercube();
    for _ in 0..10 {
        let slot_ring = isomorphism.slot_ring();
        let a = slot_ring.random_element(|| rng.rand_u64());
        
        let mut input = (0..isomorphism.slot_count()).map(|_| slot_ring.zero()).collect::<Vec<_>>();
        input[0] = slot_ring.clone_el(&a);

        let mut expected = (0..isomorphism.slot_count()).map(|_| slot_ring.zero()).collect::<Vec<_>>();
        expected[(hypercube.m(0) - 1) * hypercube.m(1)] = slot_ring.clone_el(&a);

        let actual = ring.apply_galois_action(
            &isomorphism.from_slot_values(input.into_iter()),
            hypercube.galois_group().to_ring_el(hypercube.galois_group().pow(hypercube.g(0), hypercube.m(0) as i64 - 1))
        );
        let actual = isomorphism.get_slot_values(&actual);
        for (expected, actual) in expected.iter().zip(actual) {
            assert_el_eq!(slot_ring, expected, actual);
        }
    }
}

#[test]
#[ignore]
fn time_from_slot_values_large() {
    let mut rng = oorandom::Rand64::new(1);

    let allocator = feanor_mempool::AllocRc(Rc::new(feanor_mempool::dynsize::DynLayoutMempool::<Global>::new(Alignment::of::<u64>())));
    let ring = RingValue::from(DecompositionRingBase::new(CompositeCyclotomicNumberRing::new(337, 127), Zn::new(65536)).into().with_allocator(allocator));
    let galois_group = CyclotomicGaloisGroup::new(337 * 127);
    let hypercube = HypercubeStructure::new(
        galois_group,
        galois_group.from_representative(2),
        21,
        vec![16, 126],
        vec![galois_group.from_representative(37085), galois_group.from_representative(25276)]
    );
    let H = HypercubeIsomorphism::new::<true>(ring, hypercube);
    let slot_ring = H.slot_ring();

    let value = log_time::<_, _, true, _>("from_slot_values", |[]| {
        H.from_slot_values((0..H.slot_count()).map(|_| slot_ring.random_element(|| rng.rand_u64())))
    });
    std::hint::black_box(value);

    print_all_timings();
}

#[test]
fn test_halevi_shoup_hypercube() {
    let galois_group = CyclotomicGaloisGroup::new(11 * 31);
    let hypercube_structure = HypercubeStructure::halevi_shoup_hypercube(galois_group, 2);
    assert_eq!(10, hypercube_structure.d());
    assert_eq!(2, hypercube_structure.dim_count());

    assert_eq!(1, hypercube_structure.m(0));
    assert_eq!(30, hypercube_structure.m(1));

    let galois_group = CyclotomicGaloisGroup::new(32);
    let hypercube_structure = HypercubeStructure::halevi_shoup_hypercube(galois_group, 7);
    assert_eq!(4, hypercube_structure.d());
    assert_eq!(1, hypercube_structure.dim_count());

    assert_eq!(4, hypercube_structure.m(0));

    let galois_group = CyclotomicGaloisGroup::new(32);
    let hypercube_structure = HypercubeStructure::halevi_shoup_hypercube(galois_group, 17);
    assert_eq!(2, hypercube_structure.d());
    assert_eq!(2, hypercube_structure.dim_count());

    assert_eq!(4, hypercube_structure.m(0));
    assert_eq!(2, hypercube_structure.m(1));
}