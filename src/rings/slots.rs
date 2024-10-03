use std::alloc::Allocator;
use std::cell::Ref;
use std::cell::RefCell;
use std::fs::File;
use std::io::BufReader;
use std::io::BufWriter;
use std::alloc::Global;
use std::cmp::max;

use de::DeserializeSeed;
use de::Visitor;
use feanor_math::algorithms::eea::signed_lcm;
use feanor_math::serialization::*;
use feanor_math::serialization::SerializeWithRing;
use ser::SerializeStruct;
use serde::*;

use feanor_math::algorithms::convolution::fft::FFTRNSBasedConvolution;
use feanor_math::algorithms::convolution::fft::FFTRNSBasedConvolutionZn;
use feanor_math::algorithms::discrete_log::discrete_log;
use feanor_math::algorithms::eea::signed_gcd;
use feanor_math::algorithms::int_factor;
use feanor_math::algorithms::int_factor::is_prime_power;
use feanor_math::homomorphism::*;
use feanor_math::integer::int_cast;
use feanor_math::iters::multi_cartesian_product;
use feanor_math::local::PrincipalLocalRing;
use feanor_math::primitive_int::*;
use feanor_math::ring::*;
use feanor_math::rings::extension::extension_impl::FreeAlgebraImpl;
use feanor_math::divisibility::*;
use feanor_math::rings::extension::*;
use feanor_math::rings::finite::FiniteRing;
use feanor_math::rings::finite::FiniteRingStore;
use feanor_math::rings::local::AsLocalPIR;
use feanor_math::rings::local::AsLocalPIRBase;
use feanor_math::rings::zn::zn_64::*;
use feanor_math::rings::zn::*;
use feanor_math::rings::poly::PolyRingStore;
use feanor_math::seq::*;
use feanor_math::rings::poly::dense_poly::DensePolyRing;
use feanor_math::rings::extension::galois_field::*;
use feanor_math::wrapper::RingElementWrapper;
use serialization::DeserializeHypercubeIsomorphismSeed;
use feanor_math::integer::BigIntRing;
use feanor_math::integer::IntegerRingStore;
use sparse::SparseMapVector;

use crate::cyclotomic::*;
use crate::profiling::timed_step;
use crate::StdZn;
use crate::euler_phi;
use super::decomposition::*;
use super::ntt_ring::NTTRing;
use super::ntt_ring::NTTRingBase;
use oorandom;

const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;

fn get_multiplicative_generator(ring: Zn, factorization: &[(i64, usize)]) -> ZnEl {
    let mut rng = oorandom::Rand64::new(ring.integer_ring().default_hash(ring.modulus()) as u128);
    let order = euler_phi(factorization);
    let order_factorization = int_factor::factor(&ZZ, order);
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

///
/// Corresponds to a single dimension of the hypercube, i.e. the Galois group
/// of a tensor-factor of the main ring. In other words, this stores information
/// about one of the group/subgroup pairs `<p> <= Gi` where `Gal(R/Fp) = G = G1 x ... x Gr`.
/// The good thing about the tensor product factorization is that we can derive the 
/// structure of `G / <p>` from just the `Gi` and the orders `ord(p)` in `Gi`.
/// 
/// Note that usually the `Gi` are all cyclic, except for one special case where
/// `Gi` has an index-2 cyclic subgroup (if `2^k | n` and `p = 1 mod 4`). In this case
/// however, we can write `Gi = <-1> x <g>` and create two hypercube dimensions 
/// corresponding to `<-1>` and `<g>`. Now the two dimensions are not tensor-product
/// factors of `R` anymore, but things still work out.
/// 
#[derive(Clone)]
struct HypercubeDimension {
    g_main: ZnEl,
    local_order_of_p: i64,
    order_of_g_main: i64,
    base_group_order: i64,
    factor_n: (i64, usize),
    is_primary_dim: bool,
}

impl HypercubeDimension {

    pub fn factor_n(&self) -> (i64, usize) {
        self.factor_n
    }

    pub fn order_of_p(&self) -> i64 {
        self.local_order_of_p
    }

    pub fn base_group_order(&self) -> i64 {
        self.base_group_order
    }
}

fn get_prim_root_of_unity<R>(ring: R, m: usize) -> El<R>
    where R: RingStore,
        R::Type: FiniteRing + FreeAlgebra + DivisibilityRing,
        <<R::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing + ZnRing + CanHomFrom<StaticRingBase<i64>>
{
    let (p, e) = is_prime_power(&ZZ, &ring.characteristic(&ZZ).unwrap()).unwrap();
    let max_log2_len = ZZ.abs_log2_ceil(&(ring.rank() as i64)).unwrap() + 1;
    let convolution: FFTRNSBasedConvolutionZn = FFTRNSBasedConvolutionZn::from(FFTRNSBasedConvolution::new_with(max_log2_len, BigIntRing::RING, Global));
    let galois_field = GaloisField::new_with(Zn::new(p as u64).as_field().ok().unwrap(), ring.rank(), Global, convolution);

    let rou = feanor_math::algorithms::unity_root::get_prim_root_of_unity(&galois_field, m).unwrap();

    let red_map = ReductionMap::new(ring.base_ring(), galois_field.base_ring()).unwrap();
    let mut result = ring.from_canonical_basis(galois_field.wrt_canonical_basis(&rou).into_iter().map(|x| red_map.smallest_lift(x)));
    for _ in 0..e {
        let delta = ring.checked_div(
            &ring.sub(ring.pow(ring.clone_el(&result), m), ring.one()),
            &ring.inclusion().mul_map(ring.pow(ring.clone_el(&result), m - 1), ring.base_ring().coerce(&ZZ, m as i64)) 
        ).unwrap();
        ring.sub_assign(&mut result, delta);
    }
    assert!(ring.is_one(&ring.pow(ring.clone_el(&result), m)));
    return result;
}

fn compute_hypercube_structure(n: i64, p: i64) -> (Vec<HypercubeDimension>, Zn) {
    assert!(signed_gcd(n, p, ZZ) == 1, "n and p must be coprime");
    // the unit group (Z/nZ)* decomposes as X (Z/niZ)*; this gives rise to the natural hypercube structure,
    // although technically many possible hypercube structures are possible
    let mut factorization = int_factor::factor(&ZZ, n);
    // this makes debugging easier, since we have a canonical order
    factorization.sort_unstable_by_key(|(p, _)| *p);
    let Zn_rns = zn_rns::Zn::new(factorization.iter().map(|(q, k)| Zn::new(ZZ.pow(*q, *k) as u64)).collect(), ZZ);
    let Zn = Zn::new(n as u64);
    let iso = Zn.into_can_hom(zn_big::Zn::new(ZZ, n)).ok().unwrap().compose((&Zn_rns).into_can_iso(zn_big::Zn::new(ZZ, n)).ok().unwrap());
    let from_crt = |index: usize, value: ZnEl| iso.map(Zn_rns.from_congruence((0..factorization.len()).map(|j| if j == index { value } else { Zn_rns.at(j).one() })));

    let mut dims = Vec::new();
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
                let g2 = Zqk.can_hom(&ZZ).unwrap().map(-1);
                if p % 4 == 1 {
                    // `p` is in `<g1>`
                    let logg1_p = unit_group_dlog(Zqk, g1, ord_g1, Zqk.can_hom(&ZZ).unwrap().map(p)).unwrap();
                    dims.push(HypercubeDimension {
                        local_order_of_p: ord_g1 / signed_gcd(logg1_p, ord_g1, &ZZ), 
                        base_group_order: ord_g1,
                        order_of_g_main: ord_g1,
                        g_main: from_crt(i, g1),
                        factor_n: (2, k - 1),
                        is_primary_dim: true
                    });
                    dims.push(HypercubeDimension {
                        local_order_of_p: 1, 
                        order_of_g_main: 2,
                        base_group_order: 2,
                        g_main: from_crt(i, g2),
                        factor_n: (2, k - 1),
                        is_primary_dim: false
                    });
                } else {
                    // `<p, g1> = (Z/2^kZ)*` and `p * g2 in <g1>`
                    let logg1_pg2 = unit_group_dlog(Zqk, g1, ord_g1, Zqk.mul(Zqk.can_hom(&ZZ).unwrap().map(p), g2)).unwrap();
                    dims.push(HypercubeDimension {
                        local_order_of_p: max(ord_g1 / signed_gcd(logg1_pg2, ord_g1, &ZZ), 2),
                        order_of_g_main: ord_g1,
                        base_group_order: 2 * ord_g1,
                        g_main: from_crt(i, g1),
                        factor_n: (2, *k - 1),
                        is_primary_dim: true
                    });
                }
            }
        } else {
            // `(Z/q^kZ)*` is cyclic
            let g = get_multiplicative_generator(*Zqk, &[(*q, *k)]);
            let ord_g = euler_phi(&[(*q, *k)]);
            let logg_p = unit_group_dlog(Zqk, g, ord_g, Zqk.can_hom(&ZZ).unwrap().map(p)).unwrap();
            let ord_p = ord_g / signed_gcd(logg_p, ord_g, ZZ);
            dims.push(HypercubeDimension {
                local_order_of_p: ord_p, 
                order_of_g_main: ord_g,
                base_group_order: ord_g,
                g_main: from_crt(i, g),
                factor_n: (*q, *k),
                is_primary_dim: true
            });
        }
    }
    return (dims, Zn);
}

fn order_hypercube_dimensions(mut dims: Vec<HypercubeDimension>) -> (Vec<HypercubeDimension>, Vec<usize>) {
    dims.sort_by_key(|dim| -(dim.order_of_p() as i64));
    let result = dims.iter().scan(1, |current_d, dim| {
        let new_d = signed_lcm(*current_d, dim.order_of_p() as i64, ZZ);
        let len = dim.base_group_order() as i64 / (new_d / *current_d);
        *current_d = new_d;
        return Some(len as usize);
    }).collect::<Vec<_>>();
    return (dims, result);
}

pub type SlotRing<'a, R, A> = AsLocalPIR<FreeAlgebraImpl<&'a R, SparseMapVector<&'a R>, A, FFTRNSBasedConvolutionZn>>;

///
/// Encapsulates the isomorphism
/// ```text
/// Z[X] / (Phi_n(X), p)  ->  F_(p^d) ^ G
/// ```
/// and its extension to Galois rings, where `G = Gal(K / Q) / <p>` 
/// with `K = Q[X]/(Phi_n(X))` and `d = phi(n) / #G`.
/// 
/// This becomes a hypercube by considering the decomposition
/// ```text
/// Gal(K / Q) = Z/nZ* = Z/(p1^e1)Z* x ... x Z/(pr^er)Z*
/// ```
/// where `n = p1^e1 ... pr^er`. Note that when `pi = 2`, we have
/// `Z/2^eZ* = Z/2^(e - 2)Z x Z/2Z` (thus giving two hypercube dimensions).
/// If `pi != 2` on the other hand, we find that `Z/(p^e)Z*` is cyclic, thus
/// giving one hypercube dimension.
/// 
/// # Warning
/// 
/// Note that this is the "natural" way to write `Gal(K / Q)` as a hypercube,
/// but other ways would be possible to, since many decompositions of `Gal(K / Q)`
/// are conceivable.
/// 
pub struct HypercubeIsomorphism<'a, R, F, A>
    where R: ZnRingStore,
        R::Type: StdZn,
        F: RingDecompositionSelfIso<R::Type> + CyclotomicRingDecomposition<R::Type>,
        A: Allocator + Clone,
        NTTRingBase<R, F, A>: CyclotomicRing + RingExtension<BaseRing = R>,
{
    ring: &'a NTTRingBase<R, F, A>,
    slot_unit_vec: El<NTTRing<R, F, A>>,
    slot_ring: SlotRing<'a, R, A>,
    dims: Vec<HypercubeDimension>,
    dim_lengths: Vec<usize>,
    galois_group_ring: Zn,
    d: usize
}

impl<'a, R, F, A> HypercubeIsomorphism<'a, R, F, A>
    where R: ZnRingStore,
        R::Type: StdZn,
        F: RingDecompositionSelfIso<R::Type> + CyclotomicRingDecomposition<R::Type>,
        A: Allocator + Clone,
        NTTRingBase<R, F, A>: CyclotomicRing + RingExtension<BaseRing = R>,
{
    pub fn new<const LOG: bool>(ring: &'a NTTRingBase<R, F, A>) -> Self {
        let t = int_cast(ring.base_ring().integer_ring().clone_el(ring.base_ring().modulus()), ZZ, ring.base_ring().integer_ring());
        let (p, e) = is_prime_power(&ZZ, &t).unwrap();

        let (dims, dim_lengths, galois_group_ring) = timed_step::<_, _, LOG, _>("Computing hypercube structure", |[]| {
            let (dims, galois_group_ring) = compute_hypercube_structure(ring.n() as i64, p);
            let (dims, dim_lengths) = order_hypercube_dimensions(dims);
            (dims, dim_lengths, galois_group_ring)
        });

        let frobenius = galois_group_ring.can_hom(&ZZ).unwrap().map(p);
        let slot_count: usize = dim_lengths.iter().copied().product();
        let d = ring.rank() / slot_count;
        let max_log2_len = ZZ.abs_log2_ceil(&(d as i64)).unwrap() + 1;
    
        let tmp_slot_ring = timed_step::<_, _, LOG, _>("Computing representation of slot ring", |[]| {
            let convolution = FFTRNSBasedConvolutionZn::from(FFTRNSBasedConvolution::<R::Type>::new_with(max_log2_len, BigIntRing::RING, Global));
            GaloisField::new(p, d).get_ring().galois_ring_with(AsLocalPIR::from_zn(RingRef::new(ring.base_ring().get_ring())).unwrap(), Global, convolution)
        });

        let root_of_unity = timed_step::<_, _, LOG, _>("Computing root of unity in slot ring", |[]| 
            get_prim_root_of_unity(&tmp_slot_ring, ring.n())
        );
        
        let poly_ring = DensePolyRing::new(&tmp_slot_ring, "X");
        let slot_generating_poly = timed_step::<_, _, LOG, _>("Computing minimal polynomial of root of unity", |[]| {
            let mut result = poly_ring.prod((0..d).scan(tmp_slot_ring.clone_el(&root_of_unity), |current_root_of_unity, _| {
                let result = poly_ring.sub(poly_ring.indeterminate(), poly_ring.inclusion().map_ref(current_root_of_unity));
                *current_root_of_unity = tmp_slot_ring.pow(tmp_slot_ring.clone_el(current_root_of_unity), galois_group_ring.smallest_positive_lift(frobenius) as usize);
                return Some(result);
            }));
            let normalization_factor = poly_ring.base_ring().invert(poly_ring.lc(&result).unwrap()).unwrap();
            poly_ring.inclusion().mul_assign_map(&mut result, normalization_factor);
            result
        });

        let hom = ring.base_ring().can_hom(tmp_slot_ring.base_ring()).unwrap();
        let mut slot_ring_modulus = SparseMapVector::new(d, ring.base_ring());
        for (coeff, i) in poly_ring.terms(&slot_generating_poly) {
            let coeff_wrt_basis = tmp_slot_ring.wrt_canonical_basis(&coeff);
            assert!((1..tmp_slot_ring.rank()).all(|i| tmp_slot_ring.base_ring().is_zero(&coeff_wrt_basis.at(i))));
            if i != tmp_slot_ring.rank() {
                *slot_ring_modulus.at_mut(i) = hom.map(coeff_wrt_basis.at(0));
            }
        }

        let irred_factor = ring.from_canonical_basis((0..ring.rank()).map(|i| if i < slot_ring_modulus.len() { 
            ring.base_ring().clone_el(slot_ring_modulus.at(i))
        } else if i == slot_ring_modulus.len() {
            ring.base_ring().one()
        } else {
            ring.base_ring().zero()
        }));

        slot_ring_modulus.scan(|_, x| ring.base_ring().negate_inplace(x));

        let max_log2_len = ZZ.abs_log2_ceil(&(d as i64)).unwrap() + 1;
        let slot_ring = FreeAlgebraImpl::new_with(ring.base_ring(), tmp_slot_ring.rank(), slot_ring_modulus, "𝜁", ring.allocator().clone(), FFTRNSBasedConvolution::new_with(max_log2_len, BigIntRing::RING, Global).into());
        let max_ideal_gen = slot_ring.inclusion().map(slot_ring.base_ring().coerce(&ZZ, p));
        let slot_ring = AsLocalPIR::from(AsLocalPIRBase::promise_is_local_pir(slot_ring, max_ideal_gen, Some(e)));

        let mut result = HypercubeIsomorphism {
            d: d,
            dims: dims,
            dim_lengths: dim_lengths,
            ring: ring,
            galois_group_ring: galois_group_ring,
            slot_ring: slot_ring,
            slot_unit_vec: ring.zero()
        };

        let slot_unit_vector = timed_step::<_, _, LOG, _>("Computing slot unit vector representation", |[]| {
            let unormalized_result = ring.prod(result.slot_iter(|idxs| if idxs.iter().all(|x| *x == 0) {
                    None
                } else {
                    Some(galois_group_ring.prod(result.dims.iter().enumerate().map(|(i, dim)| galois_group_ring.pow(dim.g_main, idxs[i]))))
                })
                .filter_map(|x| x)
                .map(|s| ring.apply_galois_action(&irred_factor, s)));
                
            let normalization_factor = result.slot_ring().invert(&result.get_slot_values(&unormalized_result).next().unwrap()).unwrap();
            let normalization_factor_wrt_basis = result.slot_ring().wrt_canonical_basis(&normalization_factor);
            ring.mul(
                ring.from_canonical_basis((0..ring.rank()).map(|i| if i < normalization_factor_wrt_basis.len() { normalization_factor_wrt_basis.at(i) } else { ring.base_ring().zero() })),
                unormalized_result
            )
        });

        result.slot_unit_vec = slot_unit_vector;

        return result;
    }

    pub fn reduce_modulus<'b>(&self, new_ring: &'b NTTRingBase<R, F, A>) -> HypercubeIsomorphism<'b, R, F, A> {
        assert_eq!(new_ring.n(), self.ring().n());
        let t = int_cast(self.ring.base_ring().integer_ring().clone_el(self.ring.base_ring().modulus()), ZZ, self.ring.base_ring().integer_ring());
        let (p, e) = is_prime_power(&ZZ, &t).unwrap();
        let self_ring = self.ring();
        let red_map = ReductionMap::new(self_ring.base_ring(), new_ring.base_ring()).unwrap();

        let poly_ring = DensePolyRing::new(self.slot_ring().base_ring(), "X");
        let slot_ring_modulus = self.slot_ring().generating_poly(&poly_ring, &poly_ring.base_ring().identity());
        let mut new_slot_ring_modulus = SparseMapVector::new(self.slot_ring().rank(), new_ring.base_ring());
        for (c, i) in poly_ring.terms(&slot_ring_modulus) {
            if i != self.slot_ring().rank() {
                *new_slot_ring_modulus.at_mut(i) = new_ring.base_ring().negate(red_map.map_ref(c));
            }
        }

        let max_log2_len = ZZ.abs_log2_ceil(&(self.slot_ring().rank() as i64)).unwrap() + 1;
        let slot_ring = FreeAlgebraImpl::new_with(new_ring.base_ring(), self.slot_ring().rank(), new_slot_ring_modulus, "𝜁", new_ring.allocator().clone(), FFTRNSBasedConvolution::new_with(max_log2_len, BigIntRing::RING, Global).into());
        let max_ideal_gen = slot_ring.inclusion().map(slot_ring.base_ring().coerce(&ZZ, p));
        let slot_ring = AsLocalPIR::from(AsLocalPIRBase::promise_is_local_pir(slot_ring, max_ideal_gen, Some(e)));

        let slot_unit_vec = new_ring.from_canonical_basis(self_ring.wrt_canonical_basis(&self.slot_unit_vec).into_iter().map(|x| red_map.map(x)));

        return HypercubeIsomorphism {
            d: self.d,
            dims: self.dims.clone(),
            dim_lengths: self.dim_lengths.clone(),
            galois_group_ring: self.galois_group_ring.clone(),
            ring: new_ring,
            slot_unit_vec: slot_unit_vec,
            slot_ring: slot_ring
        };
    }

    pub fn save(&self, filename: &str) {
        self.with_serializable(|data|
            serde_json::to_writer_pretty(
                BufWriter::new(File::create(filename).unwrap()), 
                &data
            ).unwrap()
        );
    }

    pub fn load(filename: &str, ring: &'a NTTRingBase<R, F, A>) -> Self {
        let mut deserializer = serde_json::Deserializer::from_reader(BufReader::new(File::open(filename).unwrap()));
        return <_ as DeserializeSeed>::deserialize(DeserializeHypercubeIsomorphismSeed { ring }, &mut deserializer).unwrap().into();
    }

    pub fn len(&self, dim_index: usize) -> usize {
        self.dim_lengths[dim_index]
    }

    pub fn corresponding_factor_n(&self, dim_index: usize) -> i64 {
        StaticRing::<i64>::RING.pow(self.dims[dim_index].factor_n.0, self.dims[dim_index].factor_n.1)
    }

    ///
    /// Returns the "local" slot rank, which is the number `d'` such that `Fp[X]/(Phi_ni) ~ F_(p^d')^(phi(ni)/d')`.
    /// In particular, this always divides the actual slot rank. However, note that the product of local slot ranks
    /// is **not** the global slot rank, but in general can be larger.
    /// 
    pub fn local_slot_rank(&self, dim_index: usize) -> usize {
        self.dims[dim_index].order_of_p() as usize
    }

    pub fn dim_count(&self) -> usize {
        self.dims.len()
    }

    pub fn slot_ring<'b>(&'b self) -> &'b SlotRing<'a, R, A> {
        &self.slot_ring
    }

    pub fn ring<'b>(&'b self) -> RingRef<'b, NTTRingBase<R, F, A>> {
        RingRef::new(&self.ring)
    }

    pub fn galois_group_mulrepr(&self) -> &Zn {
        &self.galois_group_ring
    }

    pub fn slot_count(&self) -> usize {
        self.dim_lengths.iter().copied().product()
    }

    pub fn slot_iter<'b, G, T>(&'b self, for_slot: G) -> impl 'b + ExactSizeIterator<Item = T>
        where G: 'b + Clone + FnMut(&[usize]) -> T,
            T: 'b
    {
        let mut it = multi_cartesian_product(
            self.dim_lengths.iter().map(|l| (0..*l)),
            for_slot,
            |_, x| *x
        );
        (0..self.slot_count()).map(move |_| it.next().unwrap())
    }

    pub fn from_slot_vec<I>(&self, vec: I) -> El<NTTRing<R, F, A>>
        where I: IntoIterator<Item = El<SlotRing<'a, R, A>>>
    {
        let mut given_len = 0;
        let mut vec_it = vec.into_iter().inspect(|_| given_len += 1);

        let move_to_slot_gens = self.slot_iter(|idxs| self.galois_group_mulrepr().prod(idxs.iter().enumerate().map(|(j, e)| self.shift_galois_element(j, *e as i64))));
        let result = self.ring.sum_galois_transforms(move_to_slot_gens.zip(vec_it.by_ref())
            .filter(|(_, x)| !self.slot_ring().is_zero(&x))
            .map(|(g, x)| {
                let x_wrt_basis = self.slot_ring().wrt_canonical_basis(&x);
                let mut lift_of_x = self.ring.from_canonical_basis((0..self.ring.rank()).map(|i| if i < x_wrt_basis.len() { x_wrt_basis.at(i) } else { self.ring.base_ring().zero() }));
                self.ring.mul_assign_ref(&mut lift_of_x, &self.slot_unit_vec);
                return (lift_of_x, g);
            })
        );
        assert!(vec_it.next().is_none());
        assert_eq!(given_len, self.ring.rank() / self.d);
        return result;
    }

    pub fn get_slot_value(&self, el: &El<NTTRing<R, F, A>>, move_to_slot_zero_el: ZnEl) -> El<SlotRing<'a, R, A>> {
        let el = self.ring.apply_galois_action(el, move_to_slot_zero_el);
        let poly_ring = DensePolyRing::new(self.ring.base_ring(), "X");
        let el_as_poly = RingRef::new(self.ring).poly_repr(&poly_ring, &el, self.ring.base_ring().identity());
        let poly_modulus = self.slot_ring().generating_poly(&poly_ring, self.ring.base_ring().identity());
        let (_, rem) = poly_ring.div_rem_monic(el_as_poly, &poly_modulus);
        self.slot_ring().from_canonical_basis((0..self.d).map(|i| poly_ring.base_ring().clone_el(poly_ring.coefficient_at(&rem, i))))
    }

    pub fn get_slot_values<'b>(&'b self, el: &'b El<NTTRing<R, F, A>>) -> impl 'b + ExactSizeIterator<Item = El<SlotRing<'a, R, A>>> {
        // again we use only a "slow" O(n^2) algorithm, but we only have to run it during preprocessing;
        // maybe use an FFT later?
        let mut move_to_slot_gens = self.slot_iter(|idxs| self.galois_group_mulrepr().prod(idxs.iter().enumerate().map(|(j, e)| self.shift_galois_element(j, -(*e as i64)))));
        (0..self.slot_count()).map(move |_| self.get_slot_value(el, move_to_slot_gens.next().unwrap()))
    }

    ///
    /// Returns a galois element that represents a shift along the given hypercube dimension.
    /// 
    /// Note that the behavior on "overflow", i.e. when a non-null slot is "shifted out" at the
    /// end (or the beginning in case of negative `steps`) is quite complicated. If the current
    /// dimension corresponds to a power-of-two factor, this can even influence another hypercolumn.
    /// In other cases, this will stay within the current hypercolumn, but potentially affect every
    /// entry. On the other hand, when a slot holding zero is shifted out, this will just shift in zero
    /// at the other hand.
    /// 
    pub fn shift_galois_element(&self, dim_index: usize, steps: i64) -> ZnEl {
        let g = self.dims[dim_index].g_main;
        let forward_galois_element = self.galois_group_mulrepr().pow(g, steps.abs() as usize);
        if steps > 0 {
            self.galois_group_mulrepr().invert(&forward_galois_element).unwrap()
        } else {
            forward_galois_element
        }
    }

    ///
    /// Returns the element `g` in `(Z/nZ)*` that represents the `p^count`-th power Frobenius
    /// automorphism in the main ring `(Z/p^eZ)[X]/(Phi_n)`.
    /// 
    /// Since we choose the canonical generator of the slot ring to be just `X mod f` where `f | Phi_n`,
    /// this can also be used to compute the Frobenius isomorphism in the slot ring.
    /// Note however that in the case of Galois rings `GR(p^e, d)` (with `e > 1`), the Frobenius
    /// is no longer given as `x -> x^p`.
    /// 
    pub fn frobenius_element(&self, count: i64) -> ZnEl {
        let t = int_cast(self.ring().base_ring().integer_ring().clone_el(self.ring().base_ring().modulus()), ZZ, self.ring().base_ring().integer_ring());
        let (p, _) = is_prime_power(&ZZ, &t).unwrap();
        let frobenius = self.galois_group_mulrepr().can_hom(&ZZ).unwrap().map(p);
        if count < 0 {
            self.galois_group_mulrepr().pow(frobenius, self.slot_ring().rank() - ((-count) % self.slot_ring().rank() as i64) as usize)
        } else {
            self.galois_group_mulrepr().pow(frobenius, (count % self.slot_ring().rank() as i64) as usize)
        }
    }
}

pub mod serialization {
    
    use super::*;

    #[derive(Serialize, Deserialize)]
    struct HypercubeDimensionSerializable {
        g_main: i64,
        order_of_g_main: i64,
        local_order_of_p: i64,
        factor_n: (i64, usize),
        is_primary_dim: bool,
        base_group_order: i64
    }
    
    pub struct HypercubeIsomorphismSerializable<'a, R, F, A>
        where R: ZnRingStore,
            R::Type: StdZn,
            F: RingDecompositionSelfIso<R::Type> + CyclotomicRingDecomposition<R::Type>,
            A: Allocator + Clone,
            NTTRingBase<R, F, A>: CyclotomicRing + RingExtension<BaseRing = R>,
    {
        t: i64,
        n: usize,
        slot_rank: usize,
        slot_ring_modulus: SerializeWithRing<'a, DensePolyRing<&'a <NTTRingBase<R, F, A> as RingExtension>::BaseRing>>,
        slot_unit_vec: SerializeWithRing<'a, RingRef<'a, NTTRingBase<R, F, A>>>,
        galois_group_ring_modulus: u64,
        dims: Vec<HypercubeDimensionSerializable>,
        dim_lengths: Vec<usize>
    }

    impl<'a, R, F, A> Serialize for HypercubeIsomorphismSerializable<'a, R, F, A>
        where R: ZnRingStore,
            R::Type: StdZn,
            F: RingDecompositionSelfIso<R::Type> + CyclotomicRingDecomposition<R::Type>,
            A: Allocator + Clone,
            NTTRingBase<R, F, A>: CyclotomicRing + RingExtension<BaseRing = R>,
    {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where S: Serializer
        {
            let mut out = serializer.serialize_struct("HypercubeIsomorphism", 7)?;
            out.serialize_field("t", &self.t)?;
            out.serialize_field("n", &self.n)?;
            out.serialize_field("slot_rank", &self.slot_rank)?;
            out.serialize_field("slot_ring_modulus", &self.slot_ring_modulus)?;
            out.serialize_field("slot_unit_vec", &self.slot_unit_vec)?;
            out.serialize_field("galois_group_ring_modulus", &self.galois_group_ring_modulus)?;
            out.serialize_field("dims", &self.dims)?;
            out.serialize_field("dim_lengths", &self.dim_lengths)?;
            return out.end();
        }
    }

    pub struct DeserializeHypercubeIsomorphismSeed<'a, R, F, A>
        where R: ZnRingStore,
            R::Type: StdZn,
            F: RingDecompositionSelfIso<R::Type> + CyclotomicRingDecomposition<R::Type>,
            A: Allocator + Clone,
            NTTRingBase<R, F, A>: CyclotomicRing + RingExtension<BaseRing = R>
    {
        pub ring: &'a NTTRingBase<R, F, A>
    }

    pub struct HypercubeIsomorphismDeserializable<'a, R, F, A>
        where R: ZnRingStore,
            R::Type: StdZn,
            F: RingDecompositionSelfIso<R::Type> + CyclotomicRingDecomposition<R::Type>,
            A: Allocator + Clone,
            NTTRingBase<R, F, A>: CyclotomicRing + RingExtension<BaseRing = R>,
    {
        t: i64,
        n: usize,
        slot_rank: usize,
        slot_ring_modulus: El<DensePolyRing<&'a R>>,
        slot_unit_vec: El<NTTRing<R, F, A>>,
        galois_group_ring_modulus: u64,
        dims: Vec<HypercubeDimensionSerializable>,
        dim_lengths: Vec<usize>,
        ring: &'a NTTRingBase<R, F, A>,
        poly_ring: DensePolyRing<&'a R>
    }

    impl<'a, 'de, R, F, A> DeserializeSeed<'de> for DeserializeHypercubeIsomorphismSeed<'a, R, F, A>
        where R: ZnRingStore,
            R::Type: StdZn,
            F: RingDecompositionSelfIso<R::Type> + CyclotomicRingDecomposition<R::Type>,
            A: Allocator + Clone,
            NTTRingBase<R, F, A>: CyclotomicRing + RingExtension<BaseRing = R>,
    {
        type Value = HypercubeIsomorphismDeserializable<'a, R, F, A>;

        fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
            where D: Deserializer<'de>
        {
            struct FieldsVisitor<'a, R, F, A>
                where R: ZnRingStore,
                    R::Type: StdZn,
                    F: RingDecompositionSelfIso<R::Type>,
                    A: Allocator + Clone,
                    NTTRingBase<R, F, A>: CyclotomicRing + RingExtension<BaseRing = R>,
            {
                poly_ring: DensePolyRing<&'a R>,
                ring: &'a NTTRingBase<R, F, A>
            }

            impl<'a, 'de, R, F, A> Visitor<'de> for FieldsVisitor<'a, R, F, A>
                where R: ZnRingStore,
                    R::Type: StdZn,
                    F: RingDecompositionSelfIso<R::Type> + CyclotomicRingDecomposition<R::Type>,
                    A: Allocator + Clone,
                    NTTRingBase<R, F, A>: CyclotomicRing + RingExtension<BaseRing = R>,
            {
                type Value = HypercubeIsomorphismDeserializable<'a, R, F, A>;

                fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                    write!(formatter, "struct `HypercubeIsomorphism` with fields `slot_rank`, `slot_ring_modulus`, `slot_unit_vec`, `galois_group_ring_modulus`, `dims`, `dim_lengths")
                }

                fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
                    where S: de::SeqAccess<'de>
                {
                    let t: i64 = seq.next_element()?.ok_or_else(|| de::Error::invalid_length(0, &self))?;
                    let n: usize = seq.next_element()?.ok_or_else(|| de::Error::invalid_length(1, &self))?;
                    let slot_rank: usize = seq.next_element()?.ok_or_else(|| de::Error::invalid_length(2, &self))?;
                    let slot_ring_modulus = seq.next_element_seed(DeserializeWithRing::new(&self.poly_ring))?.ok_or_else(|| de::Error::invalid_length(3, &self))?;
                    let slot_unit_vec = seq.next_element_seed(DeserializeWithRing::new(RingRef::new(self.ring)))?.ok_or_else(|| de::Error::invalid_length(4, &self))?;
                    let galois_group_ring_modulus: u64 = seq.next_element()?.ok_or_else(|| de::Error::invalid_length(5, &self))?;
                    let dims: Vec<HypercubeDimensionSerializable> = seq.next_element()?.ok_or_else(|| de::Error::invalid_length(6, &self))?;
                    let dim_lengths: Vec<usize> = seq.next_element()?.ok_or_else(|| de::Error::invalid_length(7, &self))?;
                    let ring = self.ring;
                    let poly_ring = self.poly_ring;
                    return Ok(HypercubeIsomorphismDeserializable {
                        slot_rank, slot_ring_modulus, slot_unit_vec, galois_group_ring_modulus, dims, dim_lengths, ring, poly_ring, t, n
                    });
                }

                fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
                    where M: de::MapAccess<'de>
                {
                    #[allow(non_camel_case_types)]
                    #[derive(Deserialize)]
                    enum Field {
                        t,
                        n,
                        slot_rank,
                        slot_ring_modulus,
                        slot_unit_vec,
                        galois_group_ring_modulus,
                        dims,
                        dim_lengths
                    }
                    let mut t = None;
                    let mut n = None;
                    let mut slot_rank = None;
                    let mut slot_ring_modulus = None;
                    let mut slot_unit_vec = None;
                    let mut galois_group_ring_modulus = None;
                    let mut dims = None;
                    let mut dim_lengths = None;
                    while let Some(key) = map.next_key()? {
                        match key {
                            Field::t => {
                                if t.is_some() {
                                    return Err(de::Error::duplicate_field("t"));
                                }
                                t = Some(map.next_value()?);
                            },
                            Field::n => {
                                if n.is_some() {
                                    return Err(de::Error::duplicate_field("n"));
                                }
                                n = Some(map.next_value()?);
                            },
                            Field::slot_rank => {
                                if slot_rank.is_some() {
                                    return Err(de::Error::duplicate_field("slot_rank"));
                                }
                                slot_rank = Some(map.next_value()?);
                            },
                            Field::slot_ring_modulus => {
                                if slot_ring_modulus.is_some() {
                                    return Err(de::Error::duplicate_field("slot_ring_modulus"));
                                }
                                slot_ring_modulus = Some(map.next_value_seed(DeserializeWithRing::new(&self.poly_ring))?);
                            },
                            Field::slot_unit_vec => {
                                if slot_unit_vec.is_some() {
                                    return Err(de::Error::duplicate_field("slot_unit_vec"));
                                }
                                slot_unit_vec = Some(map.next_value_seed(DeserializeWithRing::new(RingRef::new(self.ring)))?);
                            },
                            Field::galois_group_ring_modulus => {
                                if galois_group_ring_modulus.is_some() {
                                    return Err(de::Error::duplicate_field("galois_group_ring_modulus"));
                                }
                                galois_group_ring_modulus = Some(map.next_value()?);
                            },
                            Field::dims => {
                                if dims.is_some() {
                                    return Err(de::Error::duplicate_field("dims"));
                                }
                                dims = Some(map.next_value()?);
                            }
                            Field::dim_lengths => {
                                if dim_lengths.is_some() {
                                    return Err(de::Error::duplicate_field("dim_lengths"));
                                }
                                dim_lengths = Some(map.next_value()?);
                            }
                        }
                    }
                    let t = t.ok_or_else(|| de::Error::missing_field("t"))?;
                    let n = n.ok_or_else(|| de::Error::missing_field("n"))?;
                    let slot_rank = slot_rank.ok_or_else(|| de::Error::missing_field("slot_rank"))?;
                    let slot_ring_modulus = slot_ring_modulus.ok_or_else(|| de::Error::missing_field("slot_ring_modulus"))?;
                    let slot_unit_vec = slot_unit_vec.ok_or_else(|| de::Error::missing_field("slot_unit_vec"))?;
                    let galois_group_ring_modulus = galois_group_ring_modulus.ok_or_else(|| de::Error::missing_field("galois_group_ring_modulus"))?;
                    let dims = dims.ok_or_else(|| de::Error::missing_field("dims"))?;
                    let dim_lengths = dim_lengths.ok_or_else(|| de::Error::missing_field("dim_lengths"))?;
                    let ring = self.ring;
                    let poly_ring = self.poly_ring;
                    return Ok(HypercubeIsomorphismDeserializable {
                        slot_rank, slot_ring_modulus, slot_unit_vec, galois_group_ring_modulus, dims, dim_lengths, ring, poly_ring, t, n
                    });
                }
            }
            deserializer.deserialize_struct("HypercubeIsomorphism", &["slot_rank", "slot_ring_modulus", "slot_unit_vec", "galois_group_ring_modulus", "dims"], FieldsVisitor {
                poly_ring: DensePolyRing::new(self.ring.base_ring(), "X"),
                ring: self.ring
            })
        }
    }

    impl<'a, R, F, A> From<HypercubeIsomorphismDeserializable<'a, R, F, A>> for HypercubeIsomorphism<'a, R, F, A>
        where R: ZnRingStore,
            R::Type: StdZn,
            F: RingDecompositionSelfIso<R::Type> + CyclotomicRingDecomposition<R::Type>,
            A: Allocator + Clone,
            NTTRingBase<R, F, A>: CyclotomicRing + RingExtension<BaseRing = R>,
    {
        fn from(value: HypercubeIsomorphismDeserializable<'a, R, F, A>) -> Self {
            let t = int_cast(value.ring.base_ring().integer_ring().clone_el(value.ring.base_ring().modulus()), ZZ, value.ring.base_ring().integer_ring());
            assert_eq!(t, value.t);
            assert_eq!(value.n, value.ring.n());
            let (p, e) = is_prime_power(&ZZ, &t).unwrap();

            let galois_group_ring = Zn::new(value.galois_group_ring_modulus);
            let hom = galois_group_ring.can_hom(&StaticRing::<i64>::RING).unwrap();

            let mut slot_ring_modulus = SparseMapVector::new(value.slot_rank, value.ring.base_ring());
            for (coeff, i) in value.poly_ring.terms(&value.slot_ring_modulus) {
                if i != value.slot_rank {
                    *slot_ring_modulus.at_mut(i) = value.poly_ring.base_ring().clone_el(coeff);
                }
            }
            let max_log2_len = ZZ.abs_log2_ceil(&(value.slot_rank as i64)).unwrap() + 1;
            let slot_ring = FreeAlgebraImpl::new_with(value.ring.base_ring(), value.slot_rank, slot_ring_modulus, "𝜁", value.ring.allocator().clone(), FFTRNSBasedConvolution::new_with(max_log2_len, BigIntRing::RING, Global).into());
            let max_ideal_gen = slot_ring.inclusion().map(slot_ring.base_ring().coerce(&ZZ, p));
            let slot_ring: SlotRing<'a, R, A> = AsLocalPIR::from(AsLocalPIRBase::promise_is_local_pir(slot_ring, max_ideal_gen, Some(e)));

            Self {
                d: value.slot_rank,
                ring: value.ring,
                slot_unit_vec: value.slot_unit_vec,
                slot_ring: slot_ring,
                dims: value.dims.into_iter().map(|d| HypercubeDimension {
                    factor_n: d.factor_n,
                    g_main: hom.map(d.g_main),
                    local_order_of_p: d.local_order_of_p,
                    order_of_g_main: d.order_of_g_main,
                    is_primary_dim: d.is_primary_dim,
                    base_group_order: d.base_group_order
                }).collect(),
                galois_group_ring: galois_group_ring,
                dim_lengths: value.dim_lengths
            }
        }
    }

    impl<'a, R, F, A> HypercubeIsomorphism<'a, R, F, A>
        where R: ZnRingStore,
            R::Type: StdZn,
            F: RingDecompositionSelfIso<R::Type> + CyclotomicRingDecomposition<R::Type>,
            A: Allocator + Clone,
            NTTRingBase<R, F, A>: CyclotomicRing + RingExtension<BaseRing = R>,
    {
        pub fn with_serializable<G>(&self, op: G)
            where G: FnOnce(HypercubeIsomorphismSerializable<R, F, A>)
        {
            let poly_ring = DensePolyRing::new(self.ring.base_ring(), "X");

            op(HypercubeIsomorphismSerializable {
                t: int_cast(self.ring().base_ring().integer_ring().clone_el(self.ring().base_ring().modulus()), &StaticRing::<i64>::RING, self.ring().base_ring().integer_ring()),
                n: self.ring().n(),
                dims: self.dims.iter().map(|d| HypercubeDimensionSerializable {
                    factor_n: d.factor_n,
                    local_order_of_p: d.local_order_of_p,
                    order_of_g_main: d.order_of_g_main,
                    g_main: self.galois_group_mulrepr().smallest_positive_lift(d.g_main) as i64,
                    is_primary_dim: d.is_primary_dim,
                    base_group_order: d.base_group_order
                }).collect(),
                galois_group_ring_modulus: *self.galois_group_mulrepr().modulus() as u64,
                slot_rank: self.d,
                slot_ring_modulus: SerializeWithRing::new(&self.slot_ring().generating_poly(&poly_ring, &poly_ring.base_ring().identity()), poly_ring),
                slot_unit_vec: SerializeWithRing::new(&self.slot_unit_vec, self.ring()),
                dim_lengths: self.dim_lengths.clone()
            });
        }
    }
}

#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use crate::rings::pow2_cyclotomic::{DefaultPow2CyclotomicNTTRing, DefaultPow2CyclotomicNTTRingBase};
#[cfg(test)]
use crate::rings::odd_cyclotomic::{DefaultOddCyclotomicNTTRing, DefaultOddCyclotomicNTTRingBase};

#[test]
fn test_compute_hypercube_structure_pow2() {
    {
        let (dims, Z_2n) = compute_hypercube_structure(1024, 3);
        assert_eq!(1, dims.len());
        assert_eq!(256, dims[0].order_of_p());
        assert!(unit_group_dlog(&Z_2n, Z_2n.int_hom().map(3), 256, dims[0].g_main).is_none());
        assert!(unit_group_dlog(&Z_2n, Z_2n.int_hom().map(3), 256, Z_2n.negate(dims[0].g_main)).is_some());
        assert!(unit_group_dlog(&Z_2n, Z_2n.int_hom().map(3), 256, Z_2n.pow(dims[0].g_main, 2)).is_some());
    }
    {
        let (dims, Z_2n) = compute_hypercube_structure(1024, 23);
        assert_eq!(1, dims.len());
        assert_eq!(128, dims[0].order_of_p());
        assert!(unit_group_dlog(&Z_2n, Z_2n.int_hom().map(23), 128, Z_2n.pow(dims[0].g_main, 2)).is_none());
        assert!(unit_group_dlog(&Z_2n, Z_2n.int_hom().map(23), 128, Z_2n.negate(Z_2n.pow(dims[0].g_main, 2))).is_some());
        assert!(unit_group_dlog(&Z_2n, Z_2n.int_hom().map(23), 128, Z_2n.pow(dims[0].g_main, 4)).is_some());
    }
    {
        let (dims, Z_2n) = compute_hypercube_structure(1024, 13);
        assert_eq!(2, dims.len());
        assert_eq!(256, dims[0].order_of_p());
        assert_eq!(1, dims[1].order_of_p());
        assert!(unit_group_dlog(&Z_2n, Z_2n.int_hom().map(13), 256, dims[0].g_main).is_some());
        assert!(unit_group_dlog(&Z_2n, Z_2n.int_hom().map(13), 256, dims[1].g_main).is_none());
    }
    {
        let (dims, Z_2n) = compute_hypercube_structure(1024, 17);
        assert_eq!(2, dims.len());
        assert_eq!(64, dims[0].order_of_p());
        assert_eq!(1, dims[1].order_of_p());
        assert!(unit_group_dlog(&Z_2n, Z_2n.int_hom().map(17), 64, Z_2n.pow(dims[0].g_main, 2)).is_none());
        assert!(unit_group_dlog(&Z_2n, Z_2n.int_hom().map(17), 64, Z_2n.pow(dims[0].g_main, 4)).is_some());
    }
}

#[test]
fn test_compute_hypercube_structure_composite() {
    {
        let (dims, _Zn) = compute_hypercube_structure(257, 3);
        assert_eq!(1, dims.len());
        assert_eq!(256, dims[0].order_of_p());
        assert_eq!(256, dims[0].order_of_g_main);
    }
    {
        let (dims, Zn) = compute_hypercube_structure(257, 11);
        assert_eq!(1, dims.len());
        assert_eq!(64, dims[0].order_of_p());
        assert_eq!(256, dims[0].order_of_g_main);
        assert!(unit_group_dlog(&Zn, Zn.int_hom().map(11), 64, Zn.pow(dims[0].g_main, 2)).is_none());
        assert!(unit_group_dlog(&Zn, Zn.int_hom().map(11), 64, Zn.pow(dims[0].g_main, 4)).is_some());
    }
    {
        let (dims, _Zn) = compute_hypercube_structure(257 * 101, 13);
        assert_eq!(2, dims.len());
        assert_eq!(50, dims[0].order_of_p());
        assert_eq!(100, dims[0].order_of_g_main);

        assert_eq!(128, dims[1].order_of_p());
        assert_eq!(256, dims[1].order_of_g_main);
    }
    {
        let (dims, _Zn) = compute_hypercube_structure(257 * 101, 2);
        assert_eq!(2, dims.len());
        assert_eq!(100, dims[0].order_of_p());
        assert_eq!(100, dims[0].order_of_g_main);

        assert_eq!(16, dims[1].order_of_p());
        assert_eq!(256, dims[1].order_of_g_main);
    }
    {
        let (dims, Zn) = compute_hypercube_structure(11 * 31, 2);
        assert_eq!(2, dims.len());

        assert_eq!(10, dims[0].order_of_p());
        assert!(unit_group_dlog(&Zn, Zn.int_hom().map(156), 10, dims[0].g_main).is_some());
        assert!(unit_group_dlog(&Zn, dims[0].g_main, dims[0].order_of_g_main, Zn.int_hom().map(156)).is_some());

        assert_eq!(5, dims[1].order_of_p());
        assert!(unit_group_dlog(&Zn, Zn.int_hom().map(188), 5, Zn.pow(dims[1].g_main, 2)).is_none());
        assert!(unit_group_dlog(&Zn, Zn.int_hom().map(188), 5, Zn.pow(dims[1].g_main, 3)).is_none());
        assert!(unit_group_dlog(&Zn, Zn.int_hom().map(188), 5, Zn.pow(dims[1].g_main, 6)).is_some());
    }
}

#[test]
fn test_create_hypercube() {
    let ring: DefaultOddCyclotomicNTTRing = DefaultOddCyclotomicNTTRingBase::new(Zn::new(2), 31 * 11);
    let hypercube = HypercubeIsomorphism::new::<false>(ring.get_ring());

    assert_eq!(2, hypercube.dim_count());
    assert_eq!(1, hypercube.len(0));
    assert_eq!(30, hypercube.len(1));
}

#[test]
fn test_order_hypercube_dimensions_pow2() {
    {
        let ring = Zn::new(32);
        // dims of `<23> <= (Z/32Z)*`
        let dims = vec![
            HypercubeDimension {
                order_of_g_main: 8,
                g_main: ring.int_hom().map(5),
                local_order_of_p: 4,
                factor_n: (2, 5),
                is_primary_dim: true,
                base_group_order: 16
            }
        ];
        let (dims, dim_lengths) = order_hypercube_dimensions(dims);
        assert_eq!(&[4][..], &dim_lengths);
        assert_eq!(1, dims.len());
        assert_el_eq!(ring, ring.int_hom().map(5), dims[0].g_main);
        assert!(dims[0].is_primary_dim);
    }
    {
        let ring = Zn::new(32);
        // dims of `<13> <= (Z/32Z)*`
        let dims = vec![
            HypercubeDimension {
                order_of_g_main: 8,
                g_main: ring.int_hom().map(5),
                local_order_of_p: 8,
                factor_n: (2, 5),
                is_primary_dim: true,
                base_group_order: 8
            },
            HypercubeDimension {
                order_of_g_main: 2,
                g_main: ring.int_hom().map(-1),
                local_order_of_p: 1,
                factor_n: (2, 5),
                is_primary_dim: false,
                base_group_order: 2
            }
        ];
        let (dims, dim_lengths) = order_hypercube_dimensions(dims);
        assert_eq!(&[1, 2][..], &dim_lengths);
        assert_eq!(2, dims.len());
        assert_el_eq!(ring, ring.int_hom().map(5), dims[0].g_main);
        assert!(dims[0].is_primary_dim);
        assert_el_eq!(ring, ring.int_hom().map(-1), dims[1].g_main);
        assert!(!dims[1].is_primary_dim);
    }
}

#[test]
fn test_order_hypercube_dimensions_composite() {
    {
        let ring = Zn::new(11 * 31);
        // dims of `<2> <= (Z/341Z)*`
        let dims = vec![
            HypercubeDimension {
                order_of_g_main: 30,
                g_main: ring.int_hom().map(3),
                local_order_of_p: 5,
                factor_n: (31, 1),
                is_primary_dim: true,
                base_group_order: 30
            },
            HypercubeDimension {
                order_of_g_main: 10,
                g_main: ring.int_hom().map(2),
                local_order_of_p: 10,
                factor_n: (11, 1),
                is_primary_dim: true,
                base_group_order: 10
            }
        ];
        let (dims, dim_lengths) = order_hypercube_dimensions(dims);
        assert_eq!(&[1, 30][..], &dim_lengths);
        assert_eq!(2, dims.len());

        assert_eq!((11, 1), dims[0].factor_n());
        assert_el_eq!(ring, ring.int_hom().map(2), dims[0].g_main);
        assert!(dims[0].is_primary_dim);
        assert_eq!(10, dims[0].order_of_p());

        assert_eq!((31, 1), dims[1].factor_n());
        assert_el_eq!(ring, ring.int_hom().map(3), dims[1].g_main);
        assert!(dims[1].is_primary_dim);
        assert_eq!(5, dims[1].order_of_p());
    }
}

#[test]
fn test_rotation() {
    // `F23[X]/(X^16 + 1) ~ F_(23^4)^4`
    let ring: DefaultPow2CyclotomicNTTRing = DefaultPow2CyclotomicNTTRingBase::new(Zn::new(23), 4);
    let hypercube = HypercubeIsomorphism::new::<false>(ring.get_ring());
    assert_eq!(1, hypercube.dim_count());

    let current = hypercube.from_slot_vec([1, 0, 0, 0].into_iter().map(|n| hypercube.slot_ring().int_hom().map(n)));
    assert_el_eq!(
        &ring, 
        &hypercube.from_slot_vec([0, 0, 1, 0].into_iter().map(|n| hypercube.slot_ring().int_hom().map(n))),
        &ring.get_ring().apply_galois_action(&current, hypercube.shift_galois_element(0, 2))
    );
    assert_el_eq!(
        &ring, 
        &hypercube.from_slot_vec([0, 1, 0, 0].into_iter().map(|n| hypercube.slot_ring().int_hom().map(n))),
        &ring.get_ring().apply_galois_action(&current, hypercube.shift_galois_element(0, 1))
    );
    let current = hypercube.from_slot_vec([0, 1, 0, 0].into_iter().map(|n| hypercube.slot_ring().int_hom().map(n)));
    assert_el_eq!(
        &ring, 
        &hypercube.from_slot_vec([0, 0, 0, 1].into_iter().map(|n| hypercube.slot_ring().int_hom().map(n))),
        &ring.get_ring().apply_galois_action(&current, hypercube.shift_galois_element(0, 2))
    );
    assert_el_eq!(
        &ring, 
        &hypercube.from_slot_vec([1, 0, 0, 0].into_iter().map(|n| hypercube.slot_ring().int_hom().map(n))),
        &ring.get_ring().apply_galois_action(&current, hypercube.shift_galois_element(0, 3))
    );
}

#[test]
fn test_hypercube_galois_ring() {
    
    // `F(23^3)[X]/(X^16 + 1) ~ GF(23, 3, 4)^4`
    let ring: DefaultPow2CyclotomicNTTRing = DefaultPow2CyclotomicNTTRingBase::new(Zn::new(23 * 23 * 23), 4);
    let hypercube = HypercubeIsomorphism::new::<false>(ring.get_ring());
    
    let a = hypercube.slot_ring().from_canonical_basis([1, 2, 0, 1].into_iter().map(|x| hypercube.slot_ring().base_ring().int_hom().map(x)));
    let base = hypercube.from_slot_vec([None, Some(hypercube.slot_ring().clone_el(&a)), None, None].into_iter().map(|x| x.unwrap_or(hypercube.slot_ring().zero())));
    let actual = ring.pow(ring.clone_el(&base), 2);
    let expected = hypercube.from_slot_vec([None, Some(hypercube.slot_ring().pow(hypercube.slot_ring().clone_el(&a), 2)), None, None].into_iter().map(|x| x.unwrap_or(hypercube.slot_ring().zero())));
    assert_el_eq!(ring, expected, actual);

    let actual = ring.get_ring().apply_galois_action(&base, hypercube.shift_galois_element(0, 1));
    let expected = hypercube.from_slot_vec([None, None, Some(hypercube.slot_ring().clone_el(&a)), None].into_iter().map(|x| x.unwrap_or(hypercube.slot_ring().zero())));
    assert_el_eq!(ring, expected, actual);
}
