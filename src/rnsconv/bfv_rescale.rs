use feanor_math::matrix::*;
use feanor_math::homomorphism::*;
use feanor_math::rings::zn::*;
use feanor_math::rings::zn::zn_64::*;
use feanor_math::integer::*;
use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::ring::*;
use feanor_math::seq::*;
use feanor_math::ordered::OrderedRingStore;

use std::alloc::Allocator;
use std::alloc::Global;

use super::lift::AlmostExactBaseConversion;

use super::sort_unstable_permutation;
use super::RNSOperation;

const ZZbig: BigIntRing = BigIntRing::RING;

///
/// Computes almost exact rescaling with final conversion.
/// The exact rescaling with conversion refers to the map
/// ```text
/// Z/qZ -> Z/bZ, x -> round(lift(x) * a/b) mod b
/// ```
/// where `b | q` and `gcd(a, q) = 1`. We allow this implementation to
/// make an error of `+/- 1` in the result.
/// This requires that the shortest lift of the input is bounded by `q/4`.
/// 
/// # Use
/// 
/// Primarily, this is relevant as it is used during multiplication for BFV.
/// This is also the reason why we restrict to the map as above, instead of
/// ```text
/// Z/qZ -> Z/cZ, x -> round(lift(x) * a/b) mod c
/// ```
/// for an arbitrary `c`, which can be implemented just as easily.
/// 
pub struct AlmostExactRescalingConvert<A = Global>
    where A: Allocator + Clone
{
    // rescale `Z/qZ -> Z/(aq/b)Z`
    rescaling: AlmostExactRescaling<A>,
    // convert `Z/(aq/b)Z -> Z/bZ`
    convert: AlmostExactBaseConversion<A>
}

impl<A> AlmostExactRescalingConvert<A>
    where A: Allocator + Clone
{
    ///
    /// Creates a new [`AlmostExactRescalingConvert`], where
    ///  - `q` is the product of `in_moduli`
    ///  - `a` is the product of `num_moduli`
    ///  - `b` is the product of the first `den_moduli_count` elements of `in_moduli`
    /// At least the moduli belonging to `b` are expected to be sorted.
    /// 
    pub fn new(in_moduli: Vec<Zn>, num_moduli: Vec<Zn>, den_moduli_count: usize, allocator: A) -> Self {
        let rescaling = AlmostExactRescaling::new(in_moduli.clone(), num_moduli, den_moduli_count, allocator.clone());
        let convert = AlmostExactBaseConversion::new(
            rescaling.output_rings().iter().cloned().collect(),
            in_moduli[..den_moduli_count].iter().cloned().collect(),
            allocator,
        );
        return Self { rescaling, convert };
    }
}

impl<A> RNSOperation for AlmostExactRescalingConvert<A>
    where A: Allocator + Clone
{
    type Ring = Zn;

    type RingType = ZnBase;

    fn input_rings<'a>(&'a self) -> &'a [Zn] {
        self.rescaling.input_rings()
    }

    fn output_rings<'a>(&'a self) -> &'a [Zn] {
        self.convert.output_rings()
    }

    fn apply<V1, V2>(&self, input: Submatrix<V1, El<Self::Ring>>, output: SubmatrixMut<V2, El<Self::Ring>>)
            where V1: AsPointerToSlice<El<Self::Ring>>,
                V2: AsPointerToSlice<El<Self::Ring>>
    {
        assert_eq!(input.col_count(), output.col_count());
        let mut tmp = (0..(self.rescaling.output_rings().len() * input.col_count())).map(|idx| self.rescaling.output_rings().at(idx  / input.col_count()).zero()).collect::<Vec<_>>();
        let mut tmp = SubmatrixMut::<AsFirstElement<_>, _>::new(&mut tmp, self.rescaling.output_rings().len(), input.col_count());
        self.rescaling.apply(input, tmp.reborrow());
        self.convert.apply(tmp.as_const(), output);
    }
}

///
/// Computes almost exact rescaling.
/// The exact rescaling refers to the map
/// ```text
/// Z/qZ -> Z/(aq/b)Z, x -> round(lift(x) * a/b) mod aq/b
/// ```
/// where `b | q` and `gcd(a, q) = 1`. We allow allow an error of `+/- 1`, 
/// as this enables a fast RNS implementation
/// 
/// # Examples
/// ```
/// #![feature(const_type_name)]
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::zn_42::*;
/// # use feanor_math::mempool::*;
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::matrix::submatrix::*;
/// # use feanor_math::default_memory_provider;
/// # use he_ring::rnsconv::*;
/// # use he_ring::rnsconv::bfv_rescale::AlmostExactRescaling;
/// let from = vec![Zn::new(17), Zn::new(19), Zn::new(23)];
/// let from_modulus = 17 * 19 * 23;
/// let to = vec![Zn::new(29)];
/// let rescaling = AlmostExactRescaling::new(from.clone(), to.clone(), 3);
/// let mut output = [to[0].zero()];
///
/// let x = 1000;
/// rescaling.apply(Submatrix::<AsFirstElement<_>, _>::new(&[from[0].int_hom().map(x), from[1].int_hom().map(x), from[2].int_hom().map(x)], 3, 1), SubmatrixMut::<AsFirstElement<_>, _>::new(&mut output, 1, 1));
/// assert_el_eq!(
///     &to[0],
///     &to[0].int_hom().map(/* rounded division */ (x * 29 + from_modulus / 2) / from_modulus),
///     &output[0]);
/// 
/// // here we get an error of -1
/// let x = 1152;
/// rescaling.apply(Submatrix::<AsFirstElement<_>, _>::new(&[from[0].int_hom().map(x), from[1].int_hom().map(x), from[2].int_hom().map(x)], 3, 1), SubmatrixMut::<AsFirstElement<_>, _>::new(&mut output, 1, 1));
/// assert_el_eq!(
///     &to[0],
///     &to[0].int_hom().map(/* rounded division */ (x * 29 + from_modulus / 2) / from_modulus + 1),
///     &output[0]);
/// ```
/// 
pub struct AlmostExactRescaling<A>
    where A: Allocator + Clone
{
    q_moduli: Vec<Zn>,
    /// `q_moduli[i + b_moduli_count] = aq_over_b_moduli[q_to_aq_over_b_permutation[i]]`
    q_over_b_to_aq_over_b_permutation: Vec<usize>,
    /// contains the moduli of `q` and then the moduli of `a`
    b_to_aq_over_b_lift: AlmostExactBaseConversion<A>,
    /// a as element of each modulus of `q` (ordered as `q_moduli`)
    a: Vec<El<Zn>>,
    /// 1/b as element of each modulus of `aq/b` (ordered as `self.aq_over_b_moduli()`)
    b_inv: Vec<El<Zn>>,
    allocator: A
}

impl<A> AlmostExactRescaling<A>
    where A: Allocator + Clone
{
    ///
    /// Creates a new [`AlmostExactRescaling`], where
    ///  - `q` is the product of `in_moduli`
    ///  - `a` is the product of `num_moduli`
    ///  - `b` is the product of the first `den_moduli_count` elements of `in_moduli`
    /// At least the moduli belonging to `b` are expected to be sorted.
    /// 
    pub fn new(in_moduli: Vec<Zn>, num_moduli: Vec<Zn>, den_moduli_count: usize, allocator: A) -> Self {
        let a_moduli_count = num_moduli.len();
        let ZZ = in_moduli[0].integer_ring();
        for ring in &in_moduli {
            assert!(ring.integer_ring().get_ring() == ZZ.get_ring());
        }
        for ring in &num_moduli {
            assert!(ring.integer_ring().get_ring() == ZZ.get_ring());
        }
        
        let a = ZZbig.prod(num_moduli.iter().map(|Zn| int_cast(Zn.integer_ring().clone_el(Zn.modulus()), &ZZbig, Zn.integer_ring())));
        let b = ZZbig.prod(in_moduli.iter().take(den_moduli_count).map(|Zn| int_cast(Zn.integer_ring().clone_el(Zn.modulus()), &ZZbig, Zn.integer_ring())));
        let a_mod = in_moduli.iter().map(|Zn| Zn.coerce(&ZZbig, ZZbig.clone_el(&a))).collect::<Vec<_>>();

        let mut in_moduli_iter = in_moduli.iter().cloned();
        let b_moduli = in_moduli_iter.by_ref().take(den_moduli_count).collect::<Vec<_>>();
        let aq_over_b_moduli_unsorted = in_moduli_iter.chain(num_moduli.into_iter()).collect::<Vec<_>>();
        let (aq_over_b_moduli, aq_permutation) = sort_unstable_permutation(aq_over_b_moduli_unsorted, |ring_l, ring_r| ZZ.cmp(ring_l.modulus(), ring_r.modulus()));
        let mut q_over_b_to_aq_over_b_permutation = aq_permutation;
        q_over_b_to_aq_over_b_permutation.truncate(aq_over_b_moduli.len() - a_moduli_count);

        let b_to_aq_over_b_lift = AlmostExactBaseConversion::new(
            b_moduli,
            aq_over_b_moduli,
            allocator.clone()
        );
        let inv_b = b_to_aq_over_b_lift.output_rings()
            .iter()
            .map(|ring: &Zn| ring.invert(&ring.coerce(&ZZbig, ZZbig.clone_el(&b))).unwrap())
            .collect();

        AlmostExactRescaling {
            b_inv: inv_b, 
            a: a_mod,
            b_to_aq_over_b_lift: b_to_aq_over_b_lift,
            q_over_b_to_aq_over_b_permutation: q_over_b_to_aq_over_b_permutation,
            q_moduli: in_moduli, 
            allocator: allocator
        }
    }
}

impl<A> AlmostExactRescaling<A>
    where A: Allocator + Clone
{
    fn b_moduli<'a>(&'a self) -> &'a [Zn] {
        self.b_to_aq_over_b_lift.input_rings()
    }

    fn q_moduli<'a>(&'a self) -> &'a [Zn] {
        &self.q_moduli
    }

    fn aq_over_b_moduli<'a>(&'a self) -> &'a [Zn] {
        self.b_to_aq_over_b_lift.output_rings()
    }
}

impl<A> RNSOperation for AlmostExactRescaling<A>
    where A: Allocator + Clone
{
    type Ring = Zn;

    type RingType = ZnBase;

    fn input_rings<'a>(&'a self) -> &'a [Zn] {
        self.q_moduli()
    }

    fn output_rings<'a>(&'a self) -> &'a [Zn] {
        self.aq_over_b_moduli()
    }

    fn apply<V1, V2>(&self, input: Submatrix<V1, El<Self::Ring>>, mut output: SubmatrixMut<V2, El<Self::Ring>>)
        where V1: AsPointerToSlice<El<Self::Ring>>,
            V2: AsPointerToSlice<El<Self::Ring>>
    {
        assert_eq!(input.row_count(), self.input_rings().len());
        assert_eq!(output.row_count(), self.output_rings().len());
        assert_eq!(input.col_count(), output.col_count());
        let in_len = input.row_count();
        let col_count = input.col_count();

        // Compute `x := el * a mod aq`, store it in `x_mod_b` and `x_mod_aq_over_b`
        let mut x_mod_b = Vec::with_capacity_in(self.b_moduli().len() * col_count, self.allocator.clone());
        x_mod_b.extend((0..(self.b_moduli().len() * col_count)).map(|_| self.b_moduli().at(0).get_ring().zero()));
        let mut x_mod_b = SubmatrixMut::<AsFirstElement<_>, _>::new(&mut x_mod_b, self.b_moduli().len(), col_count);
        for i in 0..self.b_moduli().len() {
            for j in 0..col_count {
                *x_mod_b.at_mut(i, j) = self.b_moduli().at(i).mul_ref(input.at(i, j), self.a.at(i));
            }
        }

        for i in 0..self.aq_over_b_moduli().len() {
            for j in 0..col_count {
                *output.at_mut(i, j) = self.aq_over_b_moduli().at(i).zero();
            }
        }
        for i in self.b_moduli().len()..in_len {
            for j in 0..col_count {
                let target_index = self.q_over_b_to_aq_over_b_permutation[i - self.b_moduli().len()];
                *output.at_mut(target_index, j) = self.aq_over_b_moduli().at(target_index).mul_ref(input.at(i, j), self.a.at(i));
            }
        }
        let mut x_mod_aq_over_b = output;

        // Compute the shortest lift of `x mod b` to `aq/b`; Here we might introduce an error of `+/- b`
        // that will later be rescaled to `+/- 1`.
        let mut x_mod_b_lift: Vec<ZnEl, _> = Vec::with_capacity_in(self.aq_over_b_moduli().len() * col_count, self.allocator.clone());
        x_mod_b_lift.extend((0..(self.aq_over_b_moduli().len() * col_count)).map(|idx| self.aq_over_b_moduli().at(idx / col_count).zero()));
        let mut x_mod_b_lift = SubmatrixMut::<AsFirstElement<_>, _>::new(&mut x_mod_b_lift, self.aq_over_b_moduli().len(), col_count);
        self.b_to_aq_over_b_lift.apply(x_mod_b.as_const(), x_mod_b_lift.reborrow());
        debug_assert!(x_mod_b_lift.row_count() == self.aq_over_b_moduli().len());

        for (i, Zk) in self.aq_over_b_moduli().iter().enumerate() {
            for j in 0..col_count {
                // Subtract `lift(x mod b) mod aq/b` from `x_mod_aq_over_b`
                Zk.sub_assign_ref(x_mod_aq_over_b.at_mut(i, j), x_mod_b_lift.at(i, j));
                // Now `x_mod_aq_over_b - lift(x mod b)` is divisibible by b
                Zk.mul_assign_ref(x_mod_aq_over_b.at_mut(i, j), self.b_inv.at(i));
            }
        }
    }

}

#[cfg(test)]
use feanor_math::assert_el_eq;

#[test]
fn test_rescale() {
    let from = vec![Zn::new(17), Zn::new(97), Zn::new(113)];
    let num = vec![Zn::new(257)];
    let to = vec![Zn::new(113), Zn::new(257)];
    let q = 17 * 97 * 113;

    let rescaling = AlmostExactRescaling::new(
        from.clone(), 
        num.clone(), 
        2,
        Global
    );

    for i in -(q/2)..=(q/2) {
        let input = from.iter().map(|Zn| Zn.int_hom().map(i)).collect::<Vec<_>>();
        let output = to.iter().map(|Zn| Zn.int_hom().map((i as f64 * 257. / 17. / 97.).round() as i32)).collect::<Vec<_>>();
        let mut actual = to.iter().map(|Zn| Zn.zero()).collect::<Vec<_>>();

        rescaling.apply(Submatrix::<AsFirstElement<_>, _>::new(&input, 3, 1), SubmatrixMut::<AsFirstElement<_>, _>::new(&mut actual, 2, 1));

        for j in 0..output.len() {
            // no errors seem to occur in this case
            assert_el_eq!(to.at(j), output.at(j), actual.at(j));
        }
    }
}

#[test]
fn test_rescale_small_num() {
    let from = vec![Zn::new(17), Zn::new(97), Zn::new(113)];
    let num = vec![Zn::new(19), Zn::new(23)];
    let to = vec![Zn::new(19), Zn::new(23), Zn::new(113)];
    let q = 17 * 97 * 113;

    let rescaling = AlmostExactRescaling::new(
        from.clone(), 
        num.clone(), 
        2,
        Global
    );

    for i in -(q/2)..=(q/2) {
        let input = from.iter().map(|Zn| Zn.int_hom().map(i)).collect::<Vec<_>>();
        let output = to.iter().map(|Zn| Zn.int_hom().map((i as f64 * 19. * 23. / 17. / 97.).round() as i32)).collect::<Vec<_>>();
        let mut actual = to.iter().map(|Zn| Zn.zero()).collect::<Vec<_>>();

        rescaling.apply(Submatrix::<AsFirstElement<_>, _>::new(&input, 3, 1), SubmatrixMut::<AsFirstElement<_>, _>::new(&mut actual, 3, 1));

        for j in 0..output.len() {
            // no errors seem to occur in this case
            assert_el_eq!(to.at(j), output.at(j), actual.at(j));
        }
    }
}

#[test]
fn test_rescale_small() {
    let from = vec![Zn::new(17), Zn::new(19), Zn::new(23)];
    let num = vec![Zn::new(29)];
    let q = 17 * 19 * 23;

    let rescaling = AlmostExactRescaling::new(
        from.clone(), 
        num.clone(), 
        3,
        Global
    );

    // since Zm_intermediate has a very large modulus, we can ignore errors here at the moment (I think)
    for i in -(q/2)..=(q/2) {
        let input = from.iter().map(|Zn| Zn.int_hom().map(i)).collect::<Vec<_>>();
        let output = num.iter().map(|Zn| Zn.int_hom().map((i as f64 * 29. / q as f64).round() as i32)).collect::<Vec<_>>();
        let mut actual = num.iter().map(|Zn| Zn.zero()).collect::<Vec<_>>();

        rescaling.apply(Submatrix::<AsFirstElement<_>, _>::new(&input, 3, 1), SubmatrixMut::<AsFirstElement<_>, _>::new(&mut actual, 1, 1));

        let Zk = num.at(0);
        assert!(Zk.eq_el(output.at(0), actual.at(0)) ||
            Zk.eq_el(output.at(0), &Zk.add_ref_fst(actual.at(0), Zk.one())) ||
            Zk.eq_el(output.at(0), &Zk.sub_ref_fst(actual.at(0), Zk.one()))
        );
    }
}