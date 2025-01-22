use feanor_math::matrix::*;
use feanor_math::pid::EuclideanRingStore;
use feanor_math::rings::zn::*;
use feanor_math::rings::zn::zn_64::*;
use feanor_math::integer::*;
use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::ring::*;
use feanor_math::homomorphism::*;
use feanor_math::seq::*;
use feanor_math::ordered::OrderedRingStore;

use std::alloc::Allocator;
use std::alloc::Global;

use super::sort_unstable_permutation;
use super::RNSOperation;

type UsedBaseConversion<A> = super::lift::AlmostExactBaseConversion<A>;

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
    /// rescale `Z/qZ -> Z/(aq/b)Z`
    rescaling: AlmostExactRescaling<A>,
    /// convert `Z/(aq/b)Z -> Z/bZ`
    convert: UsedBaseConversion<A>
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
    pub fn new_with(in_moduli: Vec<Zn>, num_moduli: Vec<Zn>, den_moduli_count: usize, allocator: A) -> Self {
        let rescaling = AlmostExactRescaling::new_with(in_moduli.clone(), num_moduli, (0..den_moduli_count).collect(), allocator.clone());
        let convert = UsedBaseConversion::new_with(
            rescaling.output_rings().iter().cloned().collect(),
            in_moduli[..den_moduli_count].iter().cloned().collect(),
            allocator,
        );
        return Self { rescaling, convert };
    }

    pub fn num(&self) -> &El<BigIntRing> {
        self.rescaling.num()
    }

    pub fn den(&self) -> &El<BigIntRing> {
        self.rescaling.den()
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
        #[cfg(debug_assertions)] {
            let rns_ring = zn_rns::Zn::new(self.input_rings().iter().cloned().collect(), ZZbig);
            for j in 0..input.col_count() {
                debug_assert!(
                    ZZbig.is_leq(&ZZbig.int_hom().mul_map(ZZbig.abs(rns_ring.smallest_lift(rns_ring.from_congruence((0..input.row_count()).map(|i| self.input_rings().at(i).clone_el(input.at(i, j)))))), 4), rns_ring.modulus()),
                    "Input is not <= q/4 in absolute value"
                );
            }
        }
        let mut tmp = (0..(self.rescaling.output_rings().len() * input.col_count())).map(|idx| self.rescaling.output_rings().at(idx  / input.col_count()).zero()).collect::<Vec<_>>();
        let mut tmp = SubmatrixMut::from_1d(&mut tmp, self.rescaling.output_rings().len(), input.col_count());
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
/// #![feature(allocator_api)]
/// # use std::alloc::Global;
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::matrix::*;
/// # use he_ring::rnsconv::*;
/// # use he_ring::rnsconv::bfv_rescale::AlmostExactRescaling;
/// let from = vec![Zn::new(17), Zn::new(19), Zn::new(23)];
/// let from_modulus = 17 * 19 * 23;
/// let to = vec![Zn::new(29)];
/// let rescaling = AlmostExactRescaling::new_with(from.clone(), to.clone(), vec![0, 1, 2], Global);
/// let mut output = [to[0].zero()];
///
/// let x = 1000;
/// rescaling.apply(Submatrix::from_1d(&[from[0].int_hom().map(x), from[1].int_hom().map(x), from[2].int_hom().map(x)], 3, 1), SubmatrixMut::from_1d(&mut output, 1, 1));
/// assert_el_eq!(
///     &to[0],
///     &to[0].int_hom().map(/* rounded division */ (x * 29 + from_modulus / 2) / from_modulus),
///     &output[0]);
/// ```
/// We sometimes get an error of `+/- 1`
/// ```should_panic
/// #![feature(allocator_api)]
/// # use std::alloc::Global;
/// # use feanor_math::ring::*;
/// # use feanor_math::rings::zn::zn_64::*;
/// # use feanor_math::assert_el_eq;
/// # use feanor_math::homomorphism::*;
/// # use feanor_math::matrix::*;
/// # use he_ring::rnsconv::*;
/// # use he_ring::rnsconv::bfv_rescale::AlmostExactRescaling;
/// # let from = vec![Zn::new(17), Zn::new(19), Zn::new(23)];
/// # let from_modulus = 17 * 19 * 23;
/// # let to = vec![Zn::new(29)];
/// # let rescaling = AlmostExactRescaling::new_with(from.clone(), to.clone(), vec![0, 1, 2], Global);
/// # let mut output = [to[0].zero()];
/// for x in 1000..2000 {
///     rescaling.apply(Submatrix::from_1d(&[from[0].int_hom().map(x), from[1].int_hom().map(x), from[2].int_hom().map(x)], 3, 1), SubmatrixMut::from_1d(&mut output, 1, 1));
///     assert_el_eq!(
///         &to[0],
///         &to[0].int_hom().map(/* rounded division */ (x * 29 + from_modulus / 2) / from_modulus + 1),
///         &output[0]);
/// }
/// ```
/// 
pub struct AlmostExactRescaling<A>
    where A: Allocator + Clone
{
    /// ordered as supplied when instantiating the object
    input_moduli_unordered: Vec<Zn>,
    /// ordered as supplied when instantiating the object
    output_moduli_unordered: Vec<Zn>,
    /// ordered in ascending order
    aq_over_b_moduli: Vec<Zn>,
    b_moduli: Vec<Zn>,
    /// the `i`-th entry points to the position of `aq_over_b_moduli[i]` in `input_moduli_unordered`, or `None` if 
    /// `aq_over_b_moduli[i]` belongs to `a`
    input_permutation_q_over_b: Vec<Option<usize>>,
    /// the `i`-th entry points to the position of `b_moduli[i]` in `input_moduli_unordered`
    input_permutation_b: Vec<usize>,
    /// the `i`-th entry points to the position of `aq_over_b_moduli[i]` in `output_moduli_unordered`
    output_permutation: Vec<usize>,
    /// contains the moduli of `q` and then the moduli of `a`
    b_to_aq_over_b_lift: UsedBaseConversion<A>,
    /// `a` as an element of each modulus of `q/b`
    a_mod_q_over_b: Vec<Option<El<Zn>>>,
    /// `a` as an element of each modulus of `b`
    a_mod_b: Vec<El<Zn>>,
    /// `b^-1` as an element of each modulus of `aq/b`
    b_inv: Vec<El<Zn>>,
    allocator: A,
    a_bigint: El<BigIntRing>,
    b_bigint: El<BigIntRing>
}

impl<A> AlmostExactRescaling<A>
    where A: Allocator + Clone
{
    pub fn num(&self) -> &El<BigIntRing> {
        &self.a_bigint
    }

    pub fn den(&self) -> &El<BigIntRing> {
        &self.b_bigint
    }

    ///
    /// Creates a new [`AlmostExactRescaling`], where
    ///  - `q` is the product of `in_moduli`
    ///  - `a` is the product of `num_moduli`
    ///  - `b` is the product of the first `den_moduli_count` elements of `in_moduli`
    /// 
    pub fn new_with(in_moduli: Vec<Zn>, num_moduli: Vec<Zn>, den_moduli_indices: Vec<usize>, allocator: A) -> Self {
        let a_moduli_count = num_moduli.len();
        let ZZ = in_moduli[0].integer_ring();
        for ring in &in_moduli {
            assert!(ring.integer_ring().get_ring() == ZZ.get_ring());
        }
        for ring in &num_moduli {
            assert!(ring.integer_ring().get_ring() == ZZ.get_ring());
        }
        let a = ZZbig.prod(num_moduli.iter().map(|Zn| int_cast(Zn.integer_ring().clone_el(Zn.modulus()), &ZZbig, Zn.integer_ring())));
        let b = ZZbig.prod(den_moduli_indices.iter().map(|i| &in_moduli[*i]).map(|Zn| int_cast(Zn.integer_ring().clone_el(Zn.modulus()), &ZZbig, Zn.integer_ring())));
        
        let aq_over_b_moduli_unsorted = num_moduli.iter().copied()
            .chain((0..in_moduli.len()).filter(|i| !den_moduli_indices.contains(i)).map(|i| in_moduli[i]))
            .collect::<Vec<_>>();
        let (aq_over_b_moduli, permutation) = sort_unstable_permutation(aq_over_b_moduli_unsorted, |ring_l, ring_r| ZZ.cmp(ring_l.modulus(), ring_r.modulus()));
        let b_moduli = den_moduli_indices.iter().map(|i| in_moduli[*i]).collect::<Vec<_>>();
        let input_permutation_b = den_moduli_indices;

        let input_moduli_unordered = in_moduli;
        let mut input_permutation_q_over_b = Vec::new();

        for i in 0..aq_over_b_moduli.len() {
            let mut in_moduli_index = input_moduli_unordered.iter().enumerate().filter(|(_, Zn)| Zn.modulus() == aq_over_b_moduli[i].modulus());
            input_permutation_q_over_b.push(in_moduli_index.next().map(|(i, _)| i));
            assert!(in_moduli_index.next().is_none());
        }

        let mut output_moduli_unordered = Vec::new();
        let mut output_permutation = Vec::new();
        output_permutation.resize(aq_over_b_moduli.len(), usize::MAX);

        for i in 0..aq_over_b_moduli.len() {
            let index_in_ordered = permutation[i];
            output_permutation[index_in_ordered] = i;
            output_moduli_unordered.push(aq_over_b_moduli[index_in_ordered]);
        }

        let mut a_mod_q_over_b = Vec::new();
        a_mod_q_over_b.resize(aq_over_b_moduli.len(), None);

        for i in num_moduli.len()..aq_over_b_moduli.len() {
            let index_in_ordered = permutation[i];
            a_mod_q_over_b[index_in_ordered] = Some(aq_over_b_moduli[index_in_ordered].coerce(&ZZbig, ZZbig.clone_el(&a)));
        }

        AlmostExactRescaling {
            a_mod_b: b_moduli.iter().map(|Zn| Zn.coerce(&ZZbig, ZZbig.clone_el(&a))).collect(),
            a_mod_q_over_b: a_mod_q_over_b,
            b_inv: aq_over_b_moduli.iter().map(|Zn| Zn.invert(&Zn.coerce(&ZZbig, ZZbig.clone_el(&b))).unwrap()).collect(),
            output_permutation: output_permutation,
            input_permutation_b: input_permutation_b,
            input_permutation_q_over_b: input_permutation_q_over_b,
            b_to_aq_over_b_lift: UsedBaseConversion::new_with(b_moduli.clone(), aq_over_b_moduli.clone(), allocator.clone()),
            b_moduli: b_moduli,
            aq_over_b_moduli: aq_over_b_moduli,
            input_moduli_unordered: input_moduli_unordered,
            output_moduli_unordered: output_moduli_unordered,
            allocator: allocator,
            a_bigint: a,
            b_bigint: b
        }
    }
}

impl<A> RNSOperation for AlmostExactRescaling<A>
    where A: Allocator + Clone
{
    type Ring = Zn;

    type RingType = ZnBase;

    fn input_rings<'a>(&'a self) -> &'a [Zn] {
        &self.input_moduli_unordered
    }

    fn output_rings<'a>(&'a self) -> &'a [Zn] {
        &self.output_moduli_unordered
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
        let mut x_mod_b = Vec::with_capacity_in(self.b_moduli.len() * col_count, self.allocator.clone());
        x_mod_b.extend((0..self.b_moduli.len()).flat_map(|i| (0..col_count).map(move |j| {
            let input_i = self.input_permutation_b[i];
            debug_assert!(self.b_moduli[i].get_ring() == self.input_rings()[input_i].get_ring());
            self.b_moduli[i].mul_ref(input.at(input_i, j), self.a_mod_b.at(i))
        })));
        let x_mod_b = Submatrix::from_1d(&x_mod_b, self.b_moduli.len(), col_count);

        let mut x_mod_aq_over_b = Vec::with_capacity_in(self.aq_over_b_moduli.len() * col_count, self.allocator.clone());
        x_mod_aq_over_b.extend((0..self.aq_over_b_moduli.len()).flat_map(|i| (0..col_count).map(move |j| {
            if let Some(input_i) = self.input_permutation_q_over_b[i] {
                debug_assert!(self.aq_over_b_moduli[i].get_ring() == self.input_rings()[input_i].get_ring());
                self.aq_over_b_moduli[i].mul_ref(input.at(input_i, j), self.a_mod_q_over_b.at(i).as_ref().unwrap())
            } else {
                self.aq_over_b_moduli[i].zero()
            }
        })));
        let x_mod_aq_over_b = SubmatrixMut::from_1d(&mut x_mod_aq_over_b, self.aq_over_b_moduli.len(), col_count);

        // Compute the shortest lift of `x mod b` to `aq/b`; Here we might introduce an error of `+/- b`
        // that will later be rescaled to `+/- 1`.
        let mut x_mod_b_lift = output.reborrow();
        self.b_to_aq_over_b_lift.apply(x_mod_b, x_mod_b_lift.reborrow());

        // compute the result
        let mut result = x_mod_aq_over_b;
        for i in 0..self.aq_over_b_moduli.len() {
            let Zk = self.aq_over_b_moduli.at(i);
            for j in 0..col_count {
                // Subtract `lift(x mod b) mod aq/b` from `result`
                let divisble_by_b = Zk.sub_ref(result.at(i, j), x_mod_b_lift.at(i, j));
                // Now `result - lift(x mod b)` is divisibible by b
                *result.at_mut(i, j) = Zk.mul_ref_snd(divisble_by_b, self.b_inv.at(i));
            }
        }

        // copy and permute the result to output
        for i in 0..self.aq_over_b_moduli.len() {
            let output_i = self.output_permutation[i];
            for j in 0..col_count {
                debug_assert!(self.aq_over_b_moduli.at(i).get_ring() == self.output_rings().at(output_i).get_ring());
                *output.at_mut(output_i, j) = *result.at(i, j);
            }
        }
    }

}

#[test]
fn test_rescale_partial() {
    let from = vec![Zn::new(17), Zn::new(97), Zn::new(113)];
    let num = vec![Zn::new(257)];
    let to = vec![Zn::new(257), Zn::new(113)];
    let q = 17 * 97 * 113;

    let rescaling = AlmostExactRescaling::new_with(
        from.clone(), 
        num.clone(), 
        vec![0, 1],
        Global
    );

    for i in -(q/2)..=(q/2) {
        let input = from.iter().map(|Zn| Zn.int_hom().map(i)).collect::<Vec<_>>();
        let expected = to.iter().map(|Zn| Zn.int_hom().map((i as f64 * 257. / 17. / 97.).round() as i32)).collect::<Vec<_>>();
        let mut actual = to.iter().map(|Zn| Zn.zero()).collect::<Vec<_>>();

        rescaling.apply(Submatrix::from_1d(&input, 3, 1), SubmatrixMut::from_1d(&mut actual, 2, 1));

        for j in 0..expected.len() {
            assert!(
                to.at(j).smallest_lift(to.at(j).sub_ref(expected.at(j), actual.at(j))).abs() <= 1,
                "Expected {} to be {} +/- 1",
                to.at(j).format(actual.at(j)),
                to.at(j).format(expected.at(j))
            );
        }
    }
}

#[test]
fn test_rescale_small_num() {
    let from = vec![Zn::new(17), Zn::new(97), Zn::new(113)];
    let num = vec![Zn::new(19), Zn::new(23)];
    let to = vec![Zn::new(19), Zn::new(23), Zn::new(113)];
    let q = 17 * 97 * 113;

    let rescaling = AlmostExactRescaling::new_with(
        from.clone(), 
        num.clone(), 
        vec![0, 1],
        Global
    );

    for i in -(q/2)..=(q/2) {
        let input = from.iter().map(|Zn| Zn.int_hom().map(i)).collect::<Vec<_>>();
        let expected = to.iter().map(|Zn| Zn.int_hom().map((i as f64 * 19. * 23. / 17. / 97.).round() as i32)).collect::<Vec<_>>();
        let mut actual = to.iter().map(|Zn| Zn.zero()).collect::<Vec<_>>();

        rescaling.apply(Submatrix::from_1d(&input, 3, 1), SubmatrixMut::from_1d(&mut actual, 3, 1));

        for j in 0..expected.len() {
            assert!(
                to.at(j).smallest_lift(to.at(j).sub_ref(expected.at(j), actual.at(j))).abs() <= 1,
                "Expected {} to be {} +/- 1",
                to.at(j).format(actual.at(j)),
                to.at(j).format(expected.at(j))
            );
        }
    }
}

#[test]
fn test_rescale_small() {
    let from = vec![Zn::new(17), Zn::new(19), Zn::new(23)];
    let num = vec![Zn::new(29)];
    let q = 17 * 19 * 23;

    let rescaling = AlmostExactRescaling::new_with(
        from.clone(), 
        num.clone(), 
        vec![0, 1, 2],
        Global
    );

    // since Zm_intermediate has a very large modulus, we can ignore errors here at the moment (I think)
    for i in -(q/2)..=(q/2) {
        let input = from.iter().map(|Zn| Zn.int_hom().map(i)).collect::<Vec<_>>();
        let output = num.iter().map(|Zn| Zn.int_hom().map((i as f64 * 29. / q as f64).round() as i32)).collect::<Vec<_>>();
        let mut actual = num.iter().map(|Zn| Zn.zero()).collect::<Vec<_>>();

        rescaling.apply(Submatrix::from_1d(&input, 3, 1), SubmatrixMut::from_1d(&mut actual, 1, 1));

        let Zk = num.at(0);
        assert!(Zk.eq_el(output.at(0), actual.at(0)) ||
            Zk.eq_el(output.at(0), &Zk.add_ref_fst(actual.at(0), Zk.one())) ||
            Zk.eq_el(output.at(0), &Zk.sub_ref_fst(actual.at(0), Zk.one()))
        );
    }
}
