use std::alloc::Allocator;
use std::cell::Ref;
use std::cell::RefCell;
use std::cmp::max;
use std::fs::File;
use std::io::BufReader;
use std::io::BufWriter;
use std::time::Instant;
use std::alloc::Global;

use de::DeserializeSeed;
use de::Visitor;
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
use feanor_math::delegate::DelegateRing;
use feanor_math::iters::multi_cartesian_product;
use feanor_math::local::PrincipalLocalRing;
use feanor_math::primitive_int::*;
use feanor_math::ring::*;
use feanor_math::rings::extension::impl_new::FreeAlgebraImpl;
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
use sparse::SparseHashMapVector;
use feanor_math::integer::BigIntRing;
use feanor_math::integer::IntegerRingStore;

use crate::cyclotomic::*;
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

#[derive(Serialize, Deserialize)]
struct HypercubeDimensionSerializable {
    length: usize,
    generator: i64,
    factor_n: (i64, usize)
}

#[derive(Clone)]
pub struct HypercubeDimension {
    length: usize,
    generator: ZnEl,
    factor_n: (i64, usize)
}

impl HypercubeDimension {

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn corresponding_factor_n(&self) -> i64 {
        ZZ.pow(self.factor_n.0, self.factor_n.1)
    }
}

fn get_prim_root_of_unity<R>(ring: R, m: usize) -> El<R>
    where R: RingStore,
        R::Type: FiniteRing + FreeAlgebra + DivisibilityRing,
        <<R::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing + StdZn
{
    let (p, e) = is_prime_power(&ZZ, &ring.characteristic(&ZZ).unwrap()).unwrap();
    let max_log2_len = ZZ.abs_log2_ceil(&(ring.rank() as i64)).unwrap() + 1;
    let galois_field = RingValue::from(galois_field_dyn(p, ring.rank()).get_ring().get_delegate().clone()).set_convolution(
        FFTRNSBasedConvolutionZn::from(FFTRNSBasedConvolution::<<<R::Type as RingExtension>::BaseRing as RingStore>::Type>::new_with(max_log2_len, BigIntRing::RING, Global))
    ).as_field().ok().unwrap();

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

pub fn compute_hypercube_structure(n: i64, p: i64) -> (Vec<HypercubeDimension>, Zn) {
    // the unit group (Z/nZ)* decomposes as X (Z/niZ)*; this gives rise to the natural hypercube structure,
    // although technically many possible hypercube structures are possible
    let factorization = int_factor::factor(&ZZ, n);
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
                        length: 2, 
                        generator: from_crt(i, g2),
                        factor_n: (2, 2)
                    });
                    let ord_p = ord_g1 / signed_gcd(logg1_p, ord_g1, ZZ);
                    if ord_p != ord_g1 {
                        dims.push(HypercubeDimension {
                            length: (ord_g1 / ord_p) as usize, 
                            generator: from_crt(i, g1),
                            factor_n: (2, k - 2)
                        });
                    }
                } else {
                    // `<p, g1> = (Z/2^kZ)*` and `p * g2 in <g1>`
                    let logg1_pg2 = unit_group_dlog(Zqk, g1, ord_g1, Zqk.mul(Zqk.can_hom(&ZZ).unwrap().map(p), g2)).unwrap();
                    let ord_p = max(2, ord_g1 / signed_gcd(logg1_pg2, ord_g1, ZZ));
                    dims.push(HypercubeDimension {
                        length: (2 * ord_g1 / ord_p) as usize,
                        generator: from_crt(i, g1),
                        factor_n: (2, *k)
                    });
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
                let local_gen = Zqk.pow(g, ord_p as usize);
                dims.push(HypercubeDimension {
                    length: (ord_g / ord_p) as usize, 
                    generator: from_crt(i, local_gen),
                    factor_n: (*q, *k)
                });
            } else {
                dims.push(HypercubeDimension {
                    length: (ord_g / ord_p) as usize, 
                    generator: from_crt(i, g),
                    factor_n: (*q, *k)
                });
            }
        }
    }
    return (dims, Zn);
}

pub type SlotRing<'a, R, A> = AsLocalPIR<FreeAlgebraImpl<&'a R, SparseHashMapVector<&'a R>, A, FFTRNSBasedConvolutionZn>>;

///
/// Encapsulates the isomorphism
/// ```text
/// Z[X] / (Phi_n(X), p)  ->  F_(p^d) ^ G
/// ```
/// where `G = Gal(K / Q) / <p>` with `K = Q[X]/(Phi_n(X))` and `d = phi(n) / #G`.
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
    pub fn new(ring: &'a NTTRingBase<R, F, A>) -> Self {
        let t = int_cast(ring.base_ring().integer_ring().clone_el(ring.base_ring().modulus()), ZZ, ring.base_ring().integer_ring());
        let (p, e) = is_prime_power(&ZZ, &t).unwrap();
        let (dims, galois_group_ring) = compute_hypercube_structure(ring.n() as i64, p);
        let frobenius = galois_group_ring.can_hom(&ZZ).unwrap().map(p);
        let slot_count: usize = dims.iter().map(|dim| dim.length).product();
        let d = ring.rank() / slot_count;

        // first task: compute a nice representation of the slot ring
        let max_log2_len = ZZ.abs_log2_ceil(&(d as i64)).unwrap() + 1;
        let tmp_slot_ring = RingValue::from(galois_ring_dyn(p, e, d).get_ring().get_delegate().clone()).set_convolution(FFTRNSBasedConvolutionZn::from(FFTRNSBasedConvolution::<R::Type>::new_with(max_log2_len, BigIntRing::RING, Global)));
        
        let root_of_unity = get_prim_root_of_unity(&tmp_slot_ring, ring.n());
        
        let poly_ring = DensePolyRing::new(&tmp_slot_ring, "X");
        let mut slot_generating_poly = poly_ring.prod((0..d).scan(tmp_slot_ring.clone_el(&root_of_unity), |current_root_of_unity, _| {
            let result = poly_ring.sub(poly_ring.indeterminate(), poly_ring.inclusion().map_ref(current_root_of_unity));
            *current_root_of_unity = tmp_slot_ring.pow(tmp_slot_ring.clone_el(current_root_of_unity), galois_group_ring.smallest_positive_lift(frobenius) as usize);
            return Some(result);
        }));
        
        let normalization_factor = poly_ring.base_ring().invert(poly_ring.lc(&slot_generating_poly).unwrap()).unwrap();
        poly_ring.inclusion().mul_assign_map(&mut slot_generating_poly, normalization_factor);

        let hom = ring.base_ring().can_hom(tmp_slot_ring.base_ring()).unwrap();
        let mut slot_ring_modulus = SparseHashMapVector::new(d, ring.base_ring());
        for (coeff, i) in poly_ring.terms(&slot_generating_poly) {
            let coeff_wrt_basis = tmp_slot_ring.wrt_canonical_basis(&coeff);
            assert!((1..tmp_slot_ring.rank()).all(|i| tmp_slot_ring.base_ring().is_zero(&coeff_wrt_basis.at(i))));
            if i != tmp_slot_ring.rank() {
                *slot_ring_modulus.at_mut(i) = hom.map(coeff_wrt_basis.at(0));
            }
        }

        // second task: compute one unit vector w.r.t. the CRT isomorphism, used to later compute the whole isomorphism
        // an irreducible factor of `Phi_n` in `Zp[X]/(Phi_n)`, thus zero in the first slot and nonzero in all others
        let irred_factor = ring.from_canonical_basis((0..ring.rank()).map(|i| if i < slot_ring_modulus.len() { 
            ring.base_ring().clone_el(slot_ring_modulus.at(i))
        } else if i == slot_ring_modulus.len() {
            ring.base_ring().one()
        } else {
            ring.base_ring().zero()
        }));

        slot_ring_modulus.scan(|_, x| ring.base_ring().negate_inplace(x));
        let max_log2_len = ZZ.abs_log2_ceil(&(d as i64)).unwrap() + 1;
        let slot_ring = FreeAlgebraImpl::new_with(ring.base_ring(), tmp_slot_ring.rank(), slot_ring_modulus, ring.allocator().clone(), FFTRNSBasedConvolution::new_with(max_log2_len, BigIntRing::RING, Global).into());
        let max_ideal_gen = slot_ring.inclusion().map(slot_ring.base_ring().coerce(&ZZ, p));
        let slot_ring = AsLocalPIR::from(AsLocalPIRBase::promise_is_local_pir(slot_ring, max_ideal_gen, Some(e)));

        let mut result = HypercubeIsomorphism {
            d: d,
            dims: dims,
            ring: ring,
            galois_group_ring: galois_group_ring,
            slot_ring: slot_ring,
            slot_unit_vec: ring.zero()
        };

        // nonzero in the first slot and zero in all others
        let unnormalized_slot_unit_vector = ring.prod(result.slot_iter(|idxs| if idxs.iter().all(|x| *x == 0) {
            None
        } else {
            Some(galois_group_ring.prod(result.dims.iter().enumerate().map(|(i, dim)| galois_group_ring.pow(dim.generator, idxs[i]))))
        })
            .filter_map(|x| x)
            .map(|s| ring.apply_galois_action(&irred_factor, s)));
        
        let normalization_factor = result.slot_ring().invert(&result.get_slot_values(&unnormalized_slot_unit_vector).next().unwrap()).unwrap();
        let normalization_factor_wrt_basis = result.slot_ring().wrt_canonical_basis(&normalization_factor);

        result.slot_unit_vec = ring.mul(
            ring.from_canonical_basis((0..ring.rank()).map(|i| if i < normalization_factor_wrt_basis.len() { normalization_factor_wrt_basis.at(i) } else { ring.base_ring().zero() })),
            unnormalized_slot_unit_vector
        );

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
        let mut new_slot_ring_modulus = SparseHashMapVector::new(self.slot_ring().rank(), new_ring.base_ring());
        for (c, i) in poly_ring.terms(&slot_ring_modulus) {
            if i != self.slot_ring().rank() {
                *new_slot_ring_modulus.at_mut(i) = new_ring.base_ring().negate(red_map.map_ref(c));
            }
        }

        let max_log2_len = ZZ.abs_log2_ceil(&(self.slot_ring().rank() as i64)).unwrap() + 1;
        let slot_ring = FreeAlgebraImpl::new_with(new_ring.base_ring(), self.slot_ring().rank(), new_slot_ring_modulus, new_ring.allocator().clone(), FFTRNSBasedConvolution::new_with(max_log2_len, BigIntRing::RING, Global).into());
        let max_ideal_gen = slot_ring.inclusion().map(slot_ring.base_ring().coerce(&ZZ, p));
        let slot_ring = AsLocalPIR::from(AsLocalPIRBase::promise_is_local_pir(slot_ring, max_ideal_gen, Some(e)));

        let slot_unit_vec = new_ring.from_canonical_basis(self_ring.wrt_canonical_basis(&self.slot_unit_vec).into_iter().map(|x| red_map.map(x)));

        return HypercubeIsomorphism {
            d: self.d,
            dims: self.dims.clone(),
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
        return <_ as DeserializeSeed>::deserialize(ring, &mut deserializer).unwrap().into();
    }

    pub fn len(&self, dim_index: usize) -> usize {
        self.dims[dim_index].length
    }

    pub fn dim(&self, dim_index: usize) -> &HypercubeDimension {
        &self.dims[dim_index]
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
        self.dims.iter().map(|dim| dim.length).product()
    }

    pub fn slot_iter<'b, G, T>(&'b self, for_slot: G) -> impl 'b + ExactSizeIterator<Item = T>
        where G: 'b + Clone + FnMut(&[usize]) -> T,
            T: 'b
    {
        let mut it = multi_cartesian_product(
            self.dims.iter().map(|dim| (0..dim.length)),
            for_slot,
            |_, x| *x
        );
        (0..self.slot_count()).map(move |_| it.next().unwrap())
    }

    pub fn from_slot_vec<I>(&self, vec: I) -> El<NTTRing<R, F, A>>
        where I: ExactSizeIterator<Item = El<SlotRing<'a, R, A>>>
    {
        assert_eq!(vec.len(), self.ring.rank() / self.d);
        let move_to_slot_gens = self.slot_iter(|idxs| self.galois_group_mulrepr().prod(idxs.iter().enumerate().map(|(j, e)| self.shift_galois_element(j, *e as i64))));
        return self.ring.sum_galois_transforms(move_to_slot_gens.zip(vec)
            .filter(|(_, x)| !self.slot_ring().is_zero(&x))
            .map(|(g, x)| {
                let x_wrt_basis = self.slot_ring().wrt_canonical_basis(&x);
                let mut lift_of_x = self.ring.from_canonical_basis((0..self.ring.rank()).map(|i| if i < x_wrt_basis.len() { x_wrt_basis.at(i) } else { self.ring.base_ring().zero() }));
                self.ring.mul_assign_ref(&mut lift_of_x, &self.slot_unit_vec);
                return (lift_of_x, g);
            })
        );
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

    pub fn shift_galois_element(&self, dim_index: usize, steps: i64) -> ZnEl {
        let g = self.dims[dim_index].generator;
        let forward_galois_element = self.galois_group_mulrepr().pow(g, steps.abs() as usize);
        if steps > 0 {
            self.galois_group_mulrepr().invert(&forward_galois_element).unwrap()
        } else {
            forward_galois_element
        }
    }
}

pub mod serialization {
    
    use super::*;
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
        dims: Vec<HypercubeDimensionSerializable>
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
            return out.end();
        }
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
        ring: &'a NTTRingBase<R, F, A>,
        poly_ring: DensePolyRing<&'a R>
    }

    impl<'a, 'de, R, F, A> DeserializeSeed<'de> for &'a NTTRingBase<R, F, A>
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
                    write!(formatter, "struct `HypercubeIsomorphism` with fields `slot_rank`, `slot_ring_modulus`, `slot_unit_vec`, `galois_group_ring_modulus`, `dims`")
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
                    let ring = self.ring;
                    let poly_ring = self.poly_ring;
                    return Ok(HypercubeIsomorphismDeserializable {
                        slot_rank, slot_ring_modulus, slot_unit_vec, galois_group_ring_modulus, dims, ring, poly_ring, t, n
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
                        dims
                    }
                    let mut t = None;
                    let mut n = None;
                    let mut slot_rank = None;
                    let mut slot_ring_modulus = None;
                    let mut slot_unit_vec = None;
                    let mut galois_group_ring_modulus = None;
                    let mut dims = None;
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
                        }
                    }
                    let t = t.ok_or_else(|| de::Error::missing_field("t"))?;
                    let n = n.ok_or_else(|| de::Error::missing_field("n"))?;
                    let slot_rank = slot_rank.ok_or_else(|| de::Error::missing_field("slot_rank"))?;
                    let slot_ring_modulus = slot_ring_modulus.ok_or_else(|| de::Error::missing_field("slot_ring_modulus"))?;
                    let slot_unit_vec = slot_unit_vec.ok_or_else(|| de::Error::missing_field("slot_unit_vec"))?;
                    let galois_group_ring_modulus = galois_group_ring_modulus.ok_or_else(|| de::Error::missing_field("galois_group_ring_modulus"))?;
                    let dims = dims.ok_or_else(|| de::Error::missing_field("dims"))?;
                    let ring = self.ring;
                    let poly_ring = self.poly_ring;
                    return Ok(HypercubeIsomorphismDeserializable {
                        slot_rank, slot_ring_modulus, slot_unit_vec, galois_group_ring_modulus, dims, ring, poly_ring, t, n
                    });
                }
            }
            deserializer.deserialize_struct("HypercubeIsomorphism", &["slot_rank", "slot_ring_modulus", "slot_unit_vec", "galois_group_ring_modulus", "dims"], FieldsVisitor {
                poly_ring: DensePolyRing::new(self.base_ring(), "X"),
                ring: self
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

            let mut slot_ring_modulus = SparseHashMapVector::new(value.slot_rank, value.ring.base_ring());
            for (coeff, i) in value.poly_ring.terms(&value.slot_ring_modulus) {
                if i != value.slot_rank {
                    *slot_ring_modulus.at_mut(i) = value.poly_ring.base_ring().clone_el(coeff);
                }
            }
            let max_log2_len = ZZ.abs_log2_ceil(&(value.slot_rank as i64)).unwrap() + 1;
            let slot_ring = FreeAlgebraImpl::new_with(value.ring.base_ring(), value.slot_rank, slot_ring_modulus, value.ring.allocator().clone(), FFTRNSBasedConvolution::new_with(max_log2_len, BigIntRing::RING, Global).into());
            let max_ideal_gen = slot_ring.inclusion().map(slot_ring.base_ring().coerce(&ZZ, p));
            let slot_ring: SlotRing<'a, R, A> = AsLocalPIR::from(AsLocalPIRBase::promise_is_local_pir(slot_ring, max_ideal_gen, Some(e)));

            Self {
                d: value.slot_rank,
                ring: value.ring,
                slot_unit_vec: value.slot_unit_vec,
                slot_ring: slot_ring,
                dims: value.dims.into_iter().map(|d| HypercubeDimension {
                    factor_n: d.factor_n,
                    generator: hom.map(d.generator),
                    length: d.length
                }).collect(),
                galois_group_ring: galois_group_ring
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
                    length: d.length,
                    generator: self.galois_group_mulrepr().smallest_positive_lift(d.generator) as i64
                }).collect(),
                galois_group_ring_modulus: *self.galois_group_mulrepr().modulus() as u64,
                slot_rank: self.d,
                slot_ring_modulus: SerializeWithRing::new(&self.slot_ring().generating_poly(&poly_ring, &poly_ring.base_ring().identity()), poly_ring),
                slot_unit_vec: SerializeWithRing::new(&self.slot_unit_vec, self.ring())
            });
        }
    }
}

#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use crate::rings::pow2_cyclotomic::DefaultPow2CyclotomicNTTRing;
#[cfg(test)]
use crate::rings::pow2_cyclotomic::DefaultPow2CyclotomicNTTRingBase;

#[test]
fn test_compute_hypercube_structure_pow2() {
    {
        let (dims, Z_2n) = compute_hypercube_structure(1024, 3);
        assert_eq!(1, dims.len());
        assert_eq!(2, dims[0].length);
        assert!(unit_group_dlog(&Z_2n, Z_2n.int_hom().map(3), 256, dims[0].generator).is_none());
        assert!(unit_group_dlog(&Z_2n, Z_2n.int_hom().map(3), 256, Z_2n.pow(dims[0].generator, 2)).is_some());
    }
    {
        let (dims, Z_2n) = compute_hypercube_structure(1024, 23);
        assert_eq!(1, dims.len());
        assert_eq!(4, dims[0].length);
        assert!(unit_group_dlog(&Z_2n, Z_2n.int_hom().map(23), 128, Z_2n.pow(dims[0].generator, 2)).is_none());
        assert!(unit_group_dlog(&Z_2n, Z_2n.int_hom().map(23), 128, Z_2n.pow(dims[0].generator, 4)).is_some());
    }
    {
        let (dims, Z_2n) = compute_hypercube_structure(1024, 13);
        assert_eq!(1, dims.len());
        assert_eq!(2, dims[0].length);
        assert!(unit_group_dlog(&Z_2n, Z_2n.int_hom().map(13), 256, dims[0].generator).is_none());
        assert_eq!(Some(0), unit_group_dlog(&Z_2n, Z_2n.int_hom().map(13), 256, Z_2n.pow(dims[0].generator, 2)));
    }
    {
        let (dims, Z_2n) = compute_hypercube_structure(1024, 17);
        assert_eq!(2, dims.len());
        assert_eq!(2, dims[0].length);
        assert_eq!(4, dims[1].length);
        assert!(unit_group_dlog(&Z_2n, Z_2n.int_hom().map(17), 64, dims[0].generator).is_none());
        assert_eq!(Some(0), unit_group_dlog(&Z_2n, Z_2n.int_hom().map(17), 64, Z_2n.pow(dims[0].generator, 2)));
        assert!(unit_group_dlog(&Z_2n, Z_2n.int_hom().map(17), 64, Z_2n.pow(dims[1].generator, 2)).is_none());
        assert!(unit_group_dlog(&Z_2n, Z_2n.int_hom().map(17), 64, Z_2n.pow(dims[1].generator, 4)).is_some());
    }
}

#[test]
fn test_compute_hypercube_structure_odd() {
    {
        let (dims, _Zn) = compute_hypercube_structure(257, 3);
        assert_eq!(0, dims.len());
    }
    {
        let (dims, Zn) = compute_hypercube_structure(257, 11);
        assert_eq!(1, dims.len());
        assert_eq!(4, dims[0].length);
        assert!(unit_group_dlog(&Zn, Zn.int_hom().map(11), 64, Zn.pow(dims[0].generator, 2)).is_none());
        assert!(unit_group_dlog(&Zn, Zn.int_hom().map(11), 64, Zn.pow(dims[0].generator, 4)).is_some());
    }
    {
        let (dims, Zn) = compute_hypercube_structure(257 * 101, 13);
        assert_eq!(2, dims.len());
        assert_eq!(2, dims[0].length);
        assert_eq!(2, dims[1].length);
        assert!(unit_group_dlog(&Zn, Zn.int_hom().map(13), 3200, dims[0].generator).is_none());
        assert!(unit_group_dlog(&Zn, Zn.int_hom().map(13), 3200, dims[1].generator).is_none());
    }
}

#[test]
fn test_rotation() {
    // `F23[X]/(X^16 + 1) ~ F_(23^4)^4`
    let ring: DefaultPow2CyclotomicNTTRing = DefaultPow2CyclotomicNTTRingBase::new(Zn::new(23), 4);
    let hypercube = HypercubeIsomorphism::new(ring.get_ring());

    let current = hypercube.from_slot_vec([0, 1, 0, 0].into_iter().map(|n| hypercube.slot_ring().int_hom().map(n)));
    assert_el_eq!(
        &ring, 
        &hypercube.from_slot_vec([0, 0, 1, 0].into_iter().map(|n| hypercube.slot_ring().int_hom().map(n))),
        &ring.get_ring().apply_galois_action(&current, hypercube.shift_galois_element(0, 1))
    );
    assert_el_eq!(
        &ring, 
        &hypercube.from_slot_vec([0, 0, 0, 1].into_iter().map(|n| hypercube.slot_ring().int_hom().map(n))),
        &ring.get_ring().apply_galois_action(&current, hypercube.shift_galois_element(0, 2))
    );
    assert_el_eq!(
        &ring, 
        &hypercube.from_slot_vec([1, 0, 0, 0].into_iter().map(|n| hypercube.slot_ring().int_hom().map(n))),
        &ring.get_ring().apply_galois_action(&current, hypercube.shift_galois_element(0, -1))
    );
}

#[test]
fn test_hypercube_galois_ring() {
    
    // `F(23^3)[X]/(X^16 + 1) ~ GF(23, 3, 4)^4`
    let ring: DefaultPow2CyclotomicNTTRing = DefaultPow2CyclotomicNTTRingBase::new(Zn::new(23 * 23 * 23), 4);
    let hypercube = HypercubeIsomorphism::new(ring.get_ring());
    
    let a = hypercube.slot_ring().from_canonical_basis([1, 2, 0, 1].into_iter().map(|x| hypercube.slot_ring().base_ring().int_hom().map(x)));
    let base = hypercube.from_slot_vec([None, Some(hypercube.slot_ring().clone_el(&a)), None, None].into_iter().map(|x| x.unwrap_or(hypercube.slot_ring().zero())));
    let actual = ring.pow(ring.clone_el(&base), 2);
    let expected = hypercube.from_slot_vec([None, Some(hypercube.slot_ring().pow(hypercube.slot_ring().clone_el(&a), 2)), None, None].into_iter().map(|x| x.unwrap_or(hypercube.slot_ring().zero())));
    assert_el_eq!(ring, expected, actual);

    let actual = ring.get_ring().apply_galois_action(&base, hypercube.shift_galois_element(0, 1));
    let expected = hypercube.from_slot_vec([None, None, Some(hypercube.slot_ring().clone_el(&a)), None].into_iter().map(|x| x.unwrap_or(hypercube.slot_ring().zero())));
    assert_el_eq!(ring, expected, actual);
}