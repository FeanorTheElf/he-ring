use std::alloc::Allocator;
use std::hash::DefaultHasher;
use std::hash::Hasher;
use std::cmp::max;

use feanor_math::algorithms;
use feanor_math::algorithms::cyclotomic::cyclotomic_polynomial;
use feanor_math::algorithms::discrete_log::discrete_log;
use feanor_math::algorithms::eea::signed_gcd;
use feanor_math::algorithms::fft::cooley_tuckey::bitreverse;
use feanor_math::algorithms::int_factor;
use feanor_math::algorithms::unity_root::get_prim_root_of_unity;
use feanor_math::algorithms::unity_root::is_prim_root_of_unity;
use feanor_math::homomorphism::CanHomFrom;
use feanor_math::homomorphism::Homomorphism;
use feanor_math::integer::int_cast;
use feanor_math::integer::BigIntRing;
use feanor_math::integer::IntegerRingStore;
use feanor_math::iters::multi_cartesian_product;
use feanor_math::primitive_int::*;
use feanor_math::ring::*;
use feanor_math::divisibility::*;
use feanor_math::rings::extension::*;
use feanor_math::rings::extension::extension_impl::FreeAlgebraImpl;
use feanor_math::rings::finite::FiniteRingStore;
use feanor_math::rings::float_complex::Complex64;
use feanor_math::rings::float_complex::Complex64El;
use feanor_math::rings::poly::sparse_poly::SparsePolyRing;
use feanor_math::rings::zn::zn_64::*;
use feanor_math::rings::zn::*;
use feanor_math::rings::poly::PolyRingStore;
use feanor_math::seq::*;
use feanor_math::rings::poly::dense_poly::DensePolyRing;
use feanor_math::rings::extension::galois_field::{GFdyn, GaloisFieldDyn};
use feanor_math::assert_el_eq;
use feanor_math::wrapper::RingElementWrapper;

use crate::cyclotomic::CyclotomicRing;
use oorandom;

use super::complex_fft_ring::*;
use super::pow2_cyclotomic::DefaultPow2CyclotomicCCFFTRingBase;

const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;


pub fn euler_phi(factorization: &[(i64, usize)]) -> i64 {
    ZZ.prod(factorization.iter().map(|(p, e)| (p - 1) * ZZ.pow(*p, e - 1)))
}

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

pub trait CyclotomicRingDecomposition<R: ?Sized + RingBase>: RingDecomposition<R> {

    ///
    /// Returns `Z/nZ` such that the galois group of this number ring
    /// is `(Z/nZ)*`
    /// 
    fn galois_group_mulrepr(&self) -> Zn;

    fn permute_galois_action(&self, src: &[Complex64El], dst: &mut [Complex64El], galois_element: ZnEl);
}

impl<R, F, A> CCFFTRingBase<R, F, A>
    where R: RingStore,
        R::Type: ZnRing,
        F: RingDecompositionSelfIso<R::Type> + CyclotomicRingDecomposition<R::Type>,
        A: Allocator + Clone
{
    pub fn galois_group_mulrepr(&self) -> Zn {
        self.generalized_fft().galois_group_mulrepr()
    }

    pub fn apply_galois_action(&self, galois_element: ZnEl, mut el: <Self as RingBase>::Element) -> <Self as RingBase>::Element {
        const CC: Complex64 = Complex64::RING;
        let mut tmp_src = Vec::with_capacity_in(self.rank(), self.allocator());
        tmp_src.resize(self.rank(), CC.zero());
        self.generalized_fft().fft_forward(&el, &mut tmp_src, self.base_ring().get_ring());
        let mut tmp_dst = Vec::with_capacity_in(self.rank(), self.allocator());
        tmp_dst.resize(self.rank(), CC.zero());
        self.generalized_fft().permute_galois_action(&tmp_src, &mut tmp_dst, galois_element);
        self.generalized_fft().fft_backward(&mut tmp_dst, &mut el, self.base_ring().get_ring());
        return el;
    }

    pub fn compute_linear_transform(&self, el: &<Self as RingBase>::Element, transform: &LinearTransform<R, F, A>) -> <Self as RingBase>::Element {
        <_ as RingBase>::sum(self, transform.coeffs.iter().zip(transform.galois_elements.iter()).map(|(c, s)| self.mul_ref_fst(c, self.apply_galois_action(*s, self.clone_el(el)))))
    }
}

pub fn compute_hypercube_structure(n: i64, p: i64) -> (
    Vec<(usize, bool)>,
    Vec<ZnEl>,
    Zn
) {
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
                let g2 = Zqk.can_hom(&ZZ).unwrap().map(-1);
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

pub struct LinearTransform<R, F, A>
    where R: RingStore,
        R::Type: ZnRing,
        F: CyclotomicRingDecomposition<R::Type> + RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    coeffs: Vec<El<CCFFTRing<R, F, A>>>,
    galois_elements: Vec<ZnEl>
}

pub type SlotRing<'a, R, A> = FreeAlgebraImpl<&'a R, Vec<El<R>>, A>;

pub struct HypercubeIsomorphism<'a, R, F, A>
    where R: RingStore,
        R::Type: ZnRing,
        F: CyclotomicRingDecomposition<R::Type> + RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    ring: &'a CCFFTRingBase<R, F, A>,
    slot_unit_vec: El<CCFFTRing<R, F, A>>,
    slot_ring: FreeAlgebraImpl<&'a R, Vec<El<R>>, A>,
    dims: Vec<(usize, /* is good dim? */ bool)>,
    gens: Vec<ZnEl>,
    galois_group_ring: Zn,
    d: usize
}

impl<'a, R, F, A> HypercubeIsomorphism<'a, R, F, A>
    where R: RingStore,
        R::Type: ZnRing + CanHomFrom<<<<GaloisFieldDyn as RingStore>::Type as RingExtension>::BaseRing as RingStore>::Type>,
        F: CyclotomicRingDecomposition<R::Type> + RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone,
        CCFFTRingBase<R, F, A>: CyclotomicRing + /* unfortunately, the type checker is not clever enough to know that this is always the case */ RingExtension<BaseRing = R>
{
    pub fn new(ring: &'a CCFFTRingBase<R, F, A>) -> Self {
        let p = int_cast(ring.base_ring().integer_ring().clone_el(ring.base_ring().modulus()), ZZ, ring.base_ring().integer_ring());
        let (dims, gens, galois_group_ring) = compute_hypercube_structure(ring.n() as i64, p);
        let frobenius = galois_group_ring.can_hom(&ZZ).unwrap().map(p);
        let slot_count: usize = dims.iter().map(|(m, _)| *m).product();
        let d = ring.rank() / slot_count;

        let tmp_slot_ring = GFdyn(ZZ.pow(p, d) as u64);
        let root_of_unity = get_prim_root_of_unity(&tmp_slot_ring, ring.n()).unwrap();

        // once we support p^e moduli, use Hensel lifting here

        let poly_ring = SparsePolyRing::new(&tmp_slot_ring, "X");
        let mut slot_generating_poly = poly_ring.prod((0..d).map(|i| poly_ring.sub(
            poly_ring.indeterminate(),
            poly_ring.inclusion().map(tmp_slot_ring.pow(tmp_slot_ring.clone_el(&root_of_unity), galois_group_ring.smallest_positive_lift(galois_group_ring.pow(frobenius, i)) as usize))
        )));
        let normalization_factor = poly_ring.base_ring().invert(poly_ring.lc(&slot_generating_poly).unwrap()).unwrap();
        poly_ring.inclusion().mul_assign_map(&mut slot_generating_poly, normalization_factor);
        debug_assert!(poly_ring.checked_div(&cyclotomic_polynomial(&poly_ring, ring.n()), &slot_generating_poly).is_some());
        poly_ring.println(&slot_generating_poly);

        let hom = ring.base_ring().can_hom(tmp_slot_ring.base_ring()).unwrap();
        let mut slot_ring_modulus = (0..d).map(|i| {
            let coeff = poly_ring.coefficient_at(&slot_generating_poly, i);
            let coeff_wrt_basis = tmp_slot_ring.wrt_canonical_basis(&coeff);
            assert!((1..tmp_slot_ring.rank()).all(|i| tmp_slot_ring.base_ring().is_zero(&coeff_wrt_basis.at(i))));
            return hom.map(coeff_wrt_basis.at(0));
        }).collect::<Vec<_>>();

        // an irreducible factor of `Phi_n` in `Zp[X]/(Phi_n)`, thus zero in the first slot and nonzero in all others
        let irred_factor = ring.from_canonical_basis((0..ring.rank()).map(|i| if i < slot_ring_modulus.len() { 
            ring.base_ring().clone_el(&slot_ring_modulus[i])
        } else if i == slot_ring_modulus.len() {
            ring.base_ring().one()
        } else {
            ring.base_ring().zero()
        }));

        for i in 0..d {
            ring.base_ring().negate_inplace(&mut slot_ring_modulus[i]);
        }
        let slot_ring = FreeAlgebraImpl::new_with(ring.base_ring(), slot_ring_modulus, ring.allocator().clone());

        let mut result = HypercubeIsomorphism {
            d: d,
            dims: dims,
            gens: gens,
            ring: ring,
            galois_group_ring: galois_group_ring,
            slot_ring: slot_ring,
            slot_unit_vec: ring.zero()
        };

        // nonzero in the first slot and zero in all others
        let unnormalized_slot_unit_vector = ring.prod(result.slot_iter(|idxs| if idxs.iter().all(|x| *x == 0) {
            None
        } else {
            Some(galois_group_ring.prod(result.gens.iter().enumerate().map(|(i, g)| galois_group_ring.pow(*g, idxs[i]))))
        })
            .filter_map(|x| x)
            .map(|s| ring.apply_galois_action(s, ring.clone_el(&irred_factor))));

        let normalization_factor = result.slot_ring().invert(&result.get_slot_values(&unnormalized_slot_unit_vector).next().unwrap()).unwrap();
        let normalization_factor_wrt_basis = result.slot_ring().wrt_canonical_basis(&normalization_factor);

        result.slot_unit_vec = ring.mul(
            ring.from_canonical_basis((0..ring.rank()).map(|i| if i < normalization_factor_wrt_basis.len() { normalization_factor_wrt_basis.at(i) } else { ring.base_ring().zero() })),
            unnormalized_slot_unit_vector
        );

        return result;
    }

    fn slot_ring<'b>(&'b self) -> &'b SlotRing<'a, R, A> {
        &self.slot_ring
    }

    fn galois_group_mulrepr(&self) -> &Zn {
        &self.galois_group_ring
    }

    fn slot_count(&self) -> usize {
        self.dims.iter().map(|(m, _)| *m).product()
    }

    fn slot_iter<'b, G, T>(&'b self, for_slot: G) -> impl 'b + ExactSizeIterator<Item = T>
        where G: 'b + Clone + FnMut(&[usize]) -> T,
            T: 'b
    {
        let mut it = multi_cartesian_product(
            self.dims.iter().map(|(m, _)| (0..*m)),
            for_slot,
            |_, x| *x
        );
        (0..self.slot_count()).map(move |_| it.next().unwrap())
    }

    fn from_slot_vec<'b, I>(&self, vec: I) -> El<CCFFTRing<R, F, A>>
        where I: ExactSizeIterator<Item = El<SlotRing<'b, R, A>>>,
            R: 'b,
            'a: 'b
    {
        assert_eq!(vec.len(), self.ring.rank() / self.d);
        let mut move_to_slot_gens = self.slot_iter(|idxs| self.galois_group_mulrepr().prod(idxs.iter().enumerate().map(|(j, e)| self.galois_forward(j, *e as i64))));
        // currently only a "slow" O(n^2) algorithm, but we only have to run it during preprocessing;
        // maybe use an FFT later?
        <_ as RingBase>::sum(self.ring, vec.map(|x| {
            let s = move_to_slot_gens.next().unwrap();
            let x_wrt_basis = self.slot_ring().wrt_canonical_basis(&x);
            let lift_of_x = self.ring.from_canonical_basis((0..self.ring.rank()).map(|i| if i < x_wrt_basis.len() { x_wrt_basis.at(i) } else { self.ring.base_ring().zero() }));
            self.ring.apply_galois_action(
                s,
                self.ring.mul_ref_fst(&self.slot_unit_vec, lift_of_x)
            )
        }))
    }

    fn get_slot_values<'b>(&'b self, el: &'b El<CCFFTRing<R, F, A>>) -> impl 'b + ExactSizeIterator<Item = El<SlotRing<'a, R, A>>> {
        let mut move_to_slot_gens = self.slot_iter(|idxs| self.galois_group_mulrepr().prod(idxs.iter().enumerate().map(|(j, e)| self.galois_forward(j, -(*e as i64)))));
        (0..self.slot_count()).map(move |_| {
            let el = self.ring.apply_galois_action(move_to_slot_gens.next().unwrap(), self.ring.clone_el(&el));
            // again we use only a "slow" O(n^2) algorithm, but we only have to run it during preprocessing;
            // maybe use an FFT later?
            let poly_ring = DensePolyRing::new(self.ring.base_ring(), "X");
            let el_as_poly = RingRef::new(self.ring).poly_repr(&poly_ring, &el, self.ring.base_ring().identity());
            let poly_modulus = self.slot_ring().generating_poly(&poly_ring, self.ring.base_ring().identity());
            let (_, rem) = poly_ring.div_rem_monic(el_as_poly, &poly_modulus);
            self.slot_ring().from_canonical_basis((0..self.d).map(|i| poly_ring.base_ring().clone_el(poly_ring.coefficient_at(&rem, i))))
        })
    }

    fn galois_forward(&self, dim_index: usize, steps: i64) -> ZnEl {
        let g = self.gens[dim_index];
        let forward_galois_element = self.galois_group_mulrepr().pow(g, steps.abs() as usize);
        if steps > 0 {
            self.galois_group_mulrepr().invert(&forward_galois_element).unwrap()
        } else {
            forward_galois_element
        }
    }

    ///
    /// Works separately on each block `(b0, ..., b(l - 1))` of size `l = blocksize` along the given given hypercube dimension.
    /// This function computes the length-`l` DWT
    /// ```text
    /// sum_(0 <= i < l) ai * zeta_(4l)^(i * g^j)
    /// ```
    /// from the length-`l/2` DWTs of the even-index resp. odd-index entries of `ai`. These two sub-DWTs are expected to be written
    /// in the first resp. second half of the input block (i.e. not interleaved, this is where the "bitreversed" comes from).
    /// 
    /// Here `g` is the generator of the current hypercube dimension, i.e. usually `g = 5` in the power of two case.
    /// 
    /// # Notes
    ///  - This does not compute the evaluation at all primitive `4l`-th roots of unity, but only at half of them - namely `zeta_(4l)^(g^j)` for all `j`.
    ///    In particular, `g` does not generate `(Z/4lZ)*`, but `<g>` is an index 2 subgroup of it.
    /// 
    fn pow2_bitreversed_dwt_butterfly<'b>(&'b self, dim_index: usize, l: usize, root_of_unity_4l: El<SlotRing<'b, R, A>>) -> LinearTransform<R, F, A> {
    
        let (m, _is_good) = self.dims[dim_index];
        let g = self.gens[dim_index];

        let smaller_galois_group_mulrepr = Zn::new(4 * l as u64);
        let red = ReductionMap::new(self.galois_group_mulrepr(), &smaller_galois_group_mulrepr).unwrap();
        assert_el_eq!(&smaller_galois_group_mulrepr, &smaller_galois_group_mulrepr.one(), &smaller_galois_group_mulrepr.pow(red.map(g), l));

        let log2_m = ZZ.abs_log2_ceil(&(m as i64)).unwrap();
        assert!(m == 1 << log2_m, "pow2_bitreversed_cooley_tuckey_butterfly() only valid for hypercube dimensions that have a power-of-2 length");
        let l = l;
        assert!(l > 1);
        assert!(m % l == 0);
        let zeta = root_of_unity_4l;
        assert_el_eq!(self.slot_ring(), &self.slot_ring().neg_one(), &self.slot_ring().pow(self.slot_ring().clone_el(&zeta), 2 * l));

        enum TwiddleFactor {
            Zero, PosPowerZeta(ZnEl), NegPowerZeta(ZnEl)
        }

        let pow_of_zeta = |factor: TwiddleFactor| match factor {
            TwiddleFactor::PosPowerZeta(pow) => self.slot_ring().pow(self.slot_ring().clone_el(&zeta), self.galois_group_mulrepr().smallest_positive_lift(pow) as usize),
            TwiddleFactor::NegPowerZeta(pow) => self.slot_ring().negate(self.slot_ring().pow(self.slot_ring().clone_el(&zeta), self.galois_group_mulrepr().smallest_positive_lift(pow) as usize)),
            TwiddleFactor::Zero => self.slot_ring().zero()
        };

        let forward_galois_element = self.galois_forward(dim_index, l as i64 / 2);
        let backward_galois_element = self.galois_group_mulrepr().invert(&forward_galois_element).unwrap();

        let forward_mask = self.from_slot_vec(self.slot_iter(|idxs| {
            let idx_in_block = idxs[dim_index] % l;
            if idx_in_block >= l / 2 {
                TwiddleFactor::PosPowerZeta(self.galois_group_mulrepr().zero())
            } else {
                TwiddleFactor::Zero
            }
        }).map(&pow_of_zeta));

        let diagonal_mask = self.from_slot_vec(self.slot_iter(|idxs| {
            let idx_in_block = idxs[dim_index] % l;
            if idx_in_block >= l / 2 {
                TwiddleFactor::NegPowerZeta(self.galois_group_mulrepr().pow(g, idx_in_block - l / 2))
            } else {
                TwiddleFactor::PosPowerZeta(self.galois_group_mulrepr().zero())
            }
        }).map(&pow_of_zeta));

        let backward_mask = self.from_slot_vec(self.slot_iter(|idxs| {
            let idx_in_block = idxs[dim_index] % l;
            if idx_in_block < l / 2 {
                TwiddleFactor::PosPowerZeta(self.galois_group_mulrepr().pow(g, idx_in_block))
            } else {
                TwiddleFactor::Zero
            }
        }).map(&pow_of_zeta));

        return LinearTransform { 
            coeffs: vec![diagonal_mask, forward_mask, backward_mask], 
            galois_elements: vec![self.galois_group_mulrepr().one(), forward_galois_element, backward_galois_element]
        };
    }

    ///
    /// Computes the evaluation of `f(X) = a0 + a1 X + a2 X^2 + ... + a(m - 1) X^(m - 1)` at the
    /// `4m`-primitive roots of unity corresponding to the subgroup `<g>` of `(Z/4mZ)*`.
    /// 
    /// The input is assumed to be in bitreversed order, and the output is in order
    /// ```text
    /// f(z), f(z^g), f(z^(g^2)), f(z^(g^3)), ...
    /// ```
    /// 
    fn pow2_bitreversed_dwt(&self, dim: usize) -> Vec<LinearTransform<R, F, A>> {
        let (m, _is_good) = self.dims[dim];
        let log2_m = ZZ.abs_log2_ceil(&(m as i64)).unwrap();
        assert!(m == 1 << log2_m, "pow2_bitreversed_dwt() only valid for hypercube dimensions that have a power-of-2 length");
        assert!((self.ring.n() / m) % 4 == 0, "pow2_bitreversed_dwt() only possible if there is a 4m-th primitive root of unity");

        let zeta = self.slot_ring().pow(self.slot_ring().canonical_gen(), self.ring.n() / m / 4);
        debug_assert!(is_prim_root_of_unity(self.slot_ring(), &self.slot_ring().canonical_gen(), self.ring.n()));
        debug_assert!(is_prim_root_of_unity(self.slot_ring(), &zeta, 4 * m));

        let mut result = Vec::new();
        for log2_l in 1..=log2_m {
            result.push(self.pow2_bitreversed_dwt_butterfly(dim, 1 << log2_l, self.slot_ring().pow(self.slot_ring().clone_el(&zeta), m / (1 << log2_l))));
        }

        return result;
    }
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
        let (dims, _gens, _Zn) = compute_hypercube_structure(257, 3);
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

#[test]
fn test_rotation() {
    // `F23[X]/(X^16 + 1) ~ F_(23^4)^4`
    let ring = DefaultPow2CyclotomicCCFFTRingBase::new(Zn::new(23), 4);
    let hypercube = HypercubeIsomorphism::new(ring.get_ring());

    let current = hypercube.from_slot_vec([0, 1, 0, 0].into_iter().map(|n| hypercube.slot_ring().int_hom().map(n)));
    assert_el_eq!(
        &ring, 
        &hypercube.from_slot_vec([0, 0, 1, 0].into_iter().map(|n| hypercube.slot_ring().int_hom().map(n))),
        &ring.get_ring().apply_galois_action(hypercube.galois_forward(0, 1), ring.clone_el(&current))
    );
    assert_el_eq!(
        &ring, 
        &hypercube.from_slot_vec([0, 0, 0, 1].into_iter().map(|n| hypercube.slot_ring().int_hom().map(n))),
        &ring.get_ring().apply_galois_action(hypercube.galois_forward(0, 2), ring.clone_el(&current))
    );
    assert_el_eq!(
        &ring, 
        &hypercube.from_slot_vec([1, 0, 0, 0].into_iter().map(|n| hypercube.slot_ring().int_hom().map(n))),
        &ring.get_ring().apply_galois_action(hypercube.galois_forward(0, -1), ring.clone_el(&current))
    );
}

#[test]
fn test_pow2_bitreversed_dwt() {
    // `F23[X]/(X^16 + 1) ~ F_(23^4)^4`
    let ring = DefaultPow2CyclotomicCCFFTRingBase::new(Zn::new(23), 4);
    let hypercube = HypercubeIsomorphism::new(ring.get_ring());

    let mut current = ring_literal!(&ring, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    for T in hypercube.pow2_bitreversed_dwt(0) {
        current = ring.get_ring().compute_linear_transform(&current, &T);
    }

    assert_el_eq!(&ring, &ring_literal!(&ring, [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), &current);

    let mut current = hypercube.from_slot_vec([1, 2, 3, 4].into_iter().map(|n| hypercube.slot_ring().int_hom().map(n)));
    for T in hypercube.pow2_bitreversed_dwt(0) {
        current = ring.get_ring().compute_linear_transform(&current, &T);
    }

    // remember that input is in bitreversed order
    assert_el_eq!(&ring, &ring_literal!(&ring, [1, 0, 3, 0, 2, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0]), &current);
}