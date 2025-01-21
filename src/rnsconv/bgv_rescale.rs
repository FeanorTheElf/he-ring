use std::alloc::Allocator;
use std::alloc::Global;

use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::homomorphism::*;
use feanor_math::matrix::*;
use feanor_math::rings::zn::*;
use feanor_math::rings::zn::zn_64::*;
use feanor_math::integer::int_cast;
use feanor_math::integer::*;
use feanor_math::ring::*;
use feanor_math::ordered::OrderedRingStore;
use feanor_math::seq::*;

use super::sort_unstable_permutation;
use super::RNSOperation;

const ZZbig: BigIntRing = BigIntRing::RING;

type UsedBaseConversion<A> = super::matrix_lift::AlmostExactMatrixBaseConversion<A>;

///
/// Computes the almost exact rescaling that preserves congruence modulo `t`,
/// as required in BGV.
/// Concretely, the image `y` of `x` is the closest integer to `ax/b` with `by = ax mod t`.
/// In particular, we compute the map
/// ```text
/// Z/qZ -> Z/(aq/b)Z,  x -> round*(ax / b) + lift(a b^-1 x - round*(ax / b) mod t)
/// ```
/// To allow an efficient RNS implementation, we allow `round*` to make an error of `+/- 1`.
/// This means that the "closest integer" above might only be the second-closest when there is
/// almost a tie.
/// 
pub struct CongruencePreservingRescaling<A = Global>
    where A: Allocator + Clone
{
    /// ordered as supplied when instantiating the object
    input_moduli_unordered: Vec<Zn>,
    /// ordered as supplied when instantiating the object
    output_moduli_unordered: Vec<Zn>,
    /// ordered in ascending order
    aq_moduli: Vec<Zn>,
    /// indices of moduli belonging to `b` as part of `aq_moduli`
    b_moduli_indices: Vec<usize>,
    /// the `i`-th entry points to the position of `aq_moduli[i]` in `input_moduli_unordered`, or `None` if `aq_moduli[i]`
    /// belongs to `a`
    input_permutation: Vec<Option<usize>>,
    /// the `i`-th entry points to the position of `aq_moduli[i]` in `output_moduli_unordered`, or `None` if `aq_moduli[i]`
    /// belongs to `b`
    output_permutation: Vec<Option<usize>>,
    /// moduli of `aq` are sorted as in `aq_moduli`
    b_to_aq_lift: UsedBaseConversion<A>,
    /// moduli of `aq` are sorted as in `aq_moduli`
    aq_to_t_conv: UsedBaseConversion<A>,
    allocator: A,
    /// `a` as an element of each modulus of `q`
    a: Vec<Option<El<Zn>>>,
    /// `b^-1` as an element of each modulus of `aq/b`
    b_inv: Vec<Option<El<Zn>>>,
    /// `b^-1` as an element of `Z/tZ`
    b_inv_mod_t: El<Zn>
}

impl<A> CongruencePreservingRescaling<A>
    where A: Allocator + Clone
{
    pub fn scale_down(q_moduli: Vec<Zn>, den_moduli_indices: Vec<usize>, plaintext_modulus: Zn, allocator: A) -> Self {
        Self::new_with(q_moduli, Vec::new(), den_moduli_indices, plaintext_modulus, allocator)
    }

    ///
    /// Creates a new [`CongruencePreservingRescaling`], where
    ///  - `q` is the product of `in_moduli`
    ///  - `a` is the product of `num_moduli`
    ///  - `b` is the product of the moduli of `in_moduli` indexed by `den_moduli_indices`
    /// 
    pub fn new_with(in_moduli: Vec<Zn>, num_moduli: Vec<Zn>, den_moduli_indices: Vec<usize>, plaintext_modulus: Zn, allocator: A) -> Self {
        let ZZ = plaintext_modulus.integer_ring();
        for ring in &in_moduli {
            assert!(ring.integer_ring().get_ring() == ZZ.get_ring());
        }
        for ring in &num_moduli {
            assert!(ring.integer_ring().get_ring() == ZZ.get_ring());
        }
        
        let a = ZZbig.prod(num_moduli.iter().map(|Zn| int_cast(Zn.integer_ring().clone_el(Zn.modulus()), &ZZbig, Zn.integer_ring())));
        let b = ZZbig.prod(den_moduli_indices.iter().map(|i| &in_moduli[*i]).map(|Zn| int_cast(Zn.integer_ring().clone_el(Zn.modulus()), &ZZbig, Zn.integer_ring())));
        
        let aq_moduli_unsorted = in_moduli.iter().copied().chain(num_moduli.iter().copied()).collect::<Vec<_>>();
        let (aq_moduli, aq_permutation) = sort_unstable_permutation(aq_moduli_unsorted.clone(), |ring_l, ring_r| ZZ.cmp(ring_l.modulus(), ring_r.modulus()));

        let input_moduli_unordered = in_moduli;
        let mut input_permutation = Vec::new();
        input_permutation.resize(aq_moduli.len(), None);
        for i in 0..input_moduli_unordered.len() {
            input_permutation[aq_permutation[i]] = Some(i);
        }

        let mut output_moduli_unordered = Vec::new();
        let mut output_permutation = Vec::new();
        output_permutation.resize(aq_moduli.len(), None);
        let mut b_moduli_indices = Vec::new();
        let mut b_inv_mod_aq_over_b = Vec::new();
        b_inv_mod_aq_over_b.resize(aq_moduli.len(), None);
        let mut a_mod_q = Vec::new();
        a_mod_q.resize(aq_moduli.len(), None);

        for i in 0..input_moduli_unordered.len() {
            let index_in_ordered = aq_permutation[i];
            if den_moduli_indices.contains(&i) {
                b_moduli_indices.push(index_in_ordered);
            } else {
                output_permutation[index_in_ordered] = Some(output_moduli_unordered.len());
                output_moduli_unordered.push(aq_moduli[index_in_ordered]);
                b_inv_mod_aq_over_b[index_in_ordered] = Some(aq_moduli[index_in_ordered].invert(&aq_moduli[index_in_ordered].coerce(&ZZbig, ZZbig.clone_el(&b))).unwrap());
            }
            a_mod_q[index_in_ordered] = Some(aq_moduli[index_in_ordered].coerce(&ZZbig, ZZbig.clone_el(&a)));
        }

        for i in 0..num_moduli.len() {
            let index_in_ordered = aq_permutation[i + input_moduli_unordered.len()];
            output_permutation[index_in_ordered] = Some(output_moduli_unordered.len());
            output_moduli_unordered.push(aq_moduli[index_in_ordered]);
            b_inv_mod_aq_over_b[index_in_ordered] = Some(aq_moduli[index_in_ordered].invert(&aq_moduli[index_in_ordered].coerce(&ZZbig, ZZbig.clone_el(&b))).unwrap());
        }

        Self {
            a: a_mod_q,
            aq_moduli: aq_moduli.clone(),
            b_inv: b_inv_mod_aq_over_b,
            input_moduli_unordered,
            input_permutation: input_permutation,
            output_moduli_unordered,
            output_permutation: output_permutation,
            b_inv_mod_t: plaintext_modulus.invert(&plaintext_modulus.coerce(&ZZbig, b)).unwrap(),
            b_to_aq_lift: UsedBaseConversion::new_with(b_moduli_indices.iter().map(|i| aq_moduli[*i]).collect(), aq_moduli.clone(), allocator.clone()),
            aq_to_t_conv: UsedBaseConversion::new_with(aq_moduli, vec![plaintext_modulus], allocator.clone()),
            b_moduli_indices: b_moduli_indices,
            allocator: allocator
        }
    }

    fn t_modulus(&self) -> &Zn {
        &self.aq_to_t_conv.output_rings()[0]
    }
}

impl<A> RNSOperation for CongruencePreservingRescaling<A>
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

    ///
    /// # Implementation notes
    /// 
    /// A lot of this code is the same as for [`crate::rnsconv::bfv_rescale::AlmostExactRescaling`], but
    /// some subtle differences make it simpler to re-implement it.
    /// 
    /// In particular, we later refer to `x_mod_b_lift` again, which would not be accessible
    /// if we used [`crate::rnsconv::bfv_rescale::AlmostExactRescaling`]. Also, we currently lift to `aq`
    /// instead of `aq/b`, but I am not sure if that is really necessary.
    /// 
    fn apply<V1, V2>(&self, input: Submatrix<V1, El<Self::Ring>>, mut output: SubmatrixMut<V2, El<Self::Ring>>)
        where V1: AsPointerToSlice<El<Self::Ring>>,
            V2: AsPointerToSlice<El<Self::Ring>>
    {
        // `input` is ordered as in `input_moduli_unordered`
        assert_eq!(input.row_count(), self.input_rings().len());
        assert_eq!(output.row_count(), self.output_rings().len());
        assert_eq!(input.col_count(), output.col_count());

        // Compute `x := el * a`, ordered as in `aq_moduli`
        let mut x = Vec::with_capacity_in(self.aq_moduli.len() * input.col_count(), self.allocator.clone());
        x.extend((0..self.aq_moduli.len()).flat_map(|i| (0..input.col_count()).map(move |j| {
            if let Some(input_i) = self.input_permutation[i] {
                debug_assert!(self.aq_moduli[i].get_ring() == self.input_rings()[input_i].get_ring());
                self.aq_moduli[i].mul_ref(input.at(input_i, j), self.a.at(i).as_ref().unwrap())
            } else {
                self.aq_moduli[i].zero()
            }
        })));
        let mut x = SubmatrixMut::from_1d(&mut x, self.aq_moduli.len(), input.col_count());

        let mut x_mod_b = Vec::with_capacity_in(self.b_moduli_indices.len() * input.col_count(), self.allocator.clone());
        x_mod_b.extend(self.b_moduli_indices.iter().flat_map(|i| (0..input.col_count()).map(|j| self.aq_moduli[*i].clone_el(x.at(*i, j)))));
        let x_mod_b = Submatrix::from_1d(&x_mod_b, self.b_moduli_indices.len(), input.col_count());

        // Compute `lift(x mod b)`; Here we introduce an error of `+/- b`; ordered as in `aq_moduli`
        let mut x_mod_b_lift = Vec::with_capacity_in(self.aq_moduli.len() * input.col_count(), self.allocator.clone());
        x_mod_b_lift.extend((0..(self.aq_moduli.len() * input.col_count())).map(|idx| self.aq_moduli.at(idx / input.col_count()).zero()));
        let mut x_mod_b_lift = SubmatrixMut::from_1d(&mut x_mod_b_lift, self.aq_moduli.len(), input.col_count());
        self.b_to_aq_lift.apply(x_mod_b, x_mod_b_lift.reborrow());

        // Make `x` divisible by `b` by subtracting `lift(x mod b)`
        for (i, Zk) in self.aq_moduli.iter().enumerate() {
            for j in 0..input.col_count() {
                Zk.sub_assign_ref(x.at_mut(i, j), x_mod_b_lift.at(i, j));
            }
        }

        // now we have to ensure congruence; the difference between `x = el * a` and `b * exactdiv(x, b)` is exactly `x_mod_b_lift`;
        // this is small, so no error here
        let mut diff_mod_t = Vec::with_capacity_in(input.col_count(), self.allocator.clone());
        diff_mod_t.extend((0..input.col_count()).map(|_j| self.t_modulus().zero()));
        let mut diff_mod_t = SubmatrixMut::from_1d(&mut diff_mod_t, 1, input.col_count());
        self.aq_to_t_conv.apply(x_mod_b_lift.as_const(), diff_mod_t.reborrow());
        for j in 0..input.col_count() {
            self.t_modulus().mul_assign_ref(diff_mod_t.at_mut(0, j), &self.b_inv_mod_t);
        }

        // this is now `round((aq/b) * el / q)`, possibly `+/- 1`
        for i in 0..self.aq_moduli.len() {
            if let Some(output_i) = self.output_permutation[i] {
                debug_assert!(self.aq_moduli[i].get_ring() == self.output_rings()[output_i].get_ring());
                let Zk = &self.aq_moduli[i];
                let modulo = Zk.can_hom(self.t_modulus().integer_ring()).unwrap();
                debug_assert!(Zk.integer_ring().get_ring() == self.t_modulus().integer_ring().get_ring());
                let b_inv = self.b_inv.at(i).as_ref().unwrap();
                for j in 0..input.col_count() {
                    *output.at_mut(output_i, j) = Zk.mul_ref(x.at(i, j), b_inv);
    
                    // fix the congruence
                    Zk.add_assign(output.at_mut(output_i, j), modulo.map(self.t_modulus().smallest_lift(self.t_modulus().clone_el(diff_mod_t.at(0, j)))));
                }
            }
        }
        // output should be ordered according to `output_moduli_unordered`
    }
}

#[cfg(test)]
use feanor_math::assert_el_eq;

#[test]
fn test_rescale_complete() {
    let from = vec![Zn::new(17), Zn::new(23), Zn::new(29)];
    let to = vec![Zn::new(19), Zn::new(31)];
    let Zt = Zn::new(5);
    let q = 17 * 23 * 29;
    let qprime = 19 * 31;

    let rescaling = CongruencePreservingRescaling::new_with(
        from.clone(), 
        to.clone(), 
        vec![0, 1, 2],
        Zt.clone(), 
        Global
    );

    let ZZ_to_Zt = Zt.int_hom();
    
    for i in -(q/2)..=(q/2) {
        let input = i;
        let rescaled = (input as f64 * qprime as f64 / q as f64).round() as i32;
        let expected = rescaled + Zt.smallest_lift(Zt.sub(
            Zt.checked_div(&ZZ_to_Zt.map(input * qprime), &ZZ_to_Zt.map(q)).unwrap(),
            ZZ_to_Zt.map(rescaled)
        )) as i32;

        assert!(Zt.is_zero(&ZZ_to_Zt.map(input * qprime - expected * q)));

        let input = from.iter().map(|Zn| Zn.int_hom().map(input)).collect::<Vec<_>>();
        let expected = to.iter().map(|Zn| Zn.int_hom().map(expected)).collect::<Vec<_>>();
        let mut actual = to.iter().map(|Zn| Zn.zero()).collect::<Vec<_>>();

        rescaling.apply(Submatrix::from_1d(&input, 3, 1), SubmatrixMut::from_1d(&mut actual, 2, 1));

        for j in 0..expected.len() {
            assert!(
                to.at(j).eq_el(expected.at(j), actual.at(j)) ||
                to.at(j).eq_el(&to.at(j).add_ref_fst(expected.at(j), to.at(j).int_hom().map(5)), actual.at(j)) ||
                to.at(j).eq_el(&to.at(j).sub_ref_fst(expected.at(j), to.at(j).int_hom().map(5)), actual.at(j)),
                "Got {}, expected one of {}, {}, {} (mod {})",
                to.at(j).format(actual.at(j)),
                to.at(j).format(expected.at(j)),
                to.at(j).format(&to.at(j).add_ref_fst(expected.at(j), to.at(j).int_hom().map(5))),
                to.at(j).format(&to.at(j).sub_ref_fst(expected.at(j), to.at(j).int_hom().map(5))),
                to.at(j).modulus()
            );
        }        
    }
}

#[test]
fn test_rescale_partial() {
    let from = vec![Zn::new(17), Zn::new(23), Zn::new(29)];
    let to = vec![Zn::new(17), Zn::new(29), Zn::new(13)];
    let Zt = Zn::new(7);
    let q = 17 * 23 * 29;
    let qprime = 17 * 29 * 13;

    let rescaling = CongruencePreservingRescaling::new_with(
        from.clone(), 
        vec![Zn::new(13)], 
        vec![1],
        Zt.clone(), 
        Global
    );

    let ZZ_to_Zt = Zt.int_hom();
    
    for i in -(q/2)..=(q/2) {
        let input = i;
        let rescaled = (input as f64 * qprime as f64 / q as f64).round() as i32;
        let expected = rescaled + Zt.smallest_lift(Zt.sub(
            Zt.checked_div(&ZZ_to_Zt.map(input * qprime), &ZZ_to_Zt.map(q)).unwrap(),
            ZZ_to_Zt.map(rescaled)
        )) as i32;

        assert!(Zt.is_zero(&ZZ_to_Zt.map(input * qprime - expected * q)));

        let input = from.iter().map(|Zn| Zn.int_hom().map(input)).collect::<Vec<_>>();
        let expected = to.iter().map(|Zn| Zn.int_hom().map(expected)).collect::<Vec<_>>();
        let mut actual = to.iter().map(|Zn| Zn.zero()).collect::<Vec<_>>();

        rescaling.apply(Submatrix::from_1d(&input, 3, 1), SubmatrixMut::from_1d(&mut actual, 3, 1));

        for j in 0..expected.len() {
            assert!(
                to.at(j).eq_el(expected.at(j), actual.at(j)) ||
                to.at(j).eq_el(&to.at(j).add_ref_fst(expected.at(j), to.at(j).int_hom().map(7)), actual.at(j)) ||
                to.at(j).eq_el(&to.at(j).sub_ref_fst(expected.at(j), to.at(j).int_hom().map(7)), actual.at(j)),
                "Got {}, expected one of {}, {}, {} (mod {})",
                to.at(j).format(actual.at(j)),
                to.at(j).format(expected.at(j)),
                to.at(j).format(&to.at(j).add_ref_fst(expected.at(j), to.at(j).int_hom().map(7))),
                to.at(j).format(&to.at(j).sub_ref_fst(expected.at(j), to.at(j).int_hom().map(7))),
                to.at(j).modulus()
            );
        }        
    }
}

#[test]
fn test_rescale_down() {
    let from = vec![Zn::new(17), Zn::new(23), Zn::new(29)];
    let to = vec![Zn::new(23), Zn::new(29)];
    let Zt = Zn::new(5);
    let q = 17 * 23 * 29;
    let qprime = 23 * 29;

    let rescaling = CongruencePreservingRescaling::scale_down(
        from.clone(), 
        vec![0],
        Zt.clone(), 
        Global
    );

    let ZZ_to_Zt = Zt.int_hom();

    for i in -(q/2)..=(q/2) {
        let input = i;
        let rescaled = (input as f64 * qprime as f64 / q as f64).round() as i32;
        let output = rescaled + Zt.smallest_lift(Zt.sub(
            Zt.checked_div(&ZZ_to_Zt.map(input * qprime), &ZZ_to_Zt.map(q)).unwrap(),
            ZZ_to_Zt.map(rescaled)
        )) as i32;

        assert!(Zt.is_zero(&ZZ_to_Zt.map(input * qprime - output * q)));

        let input = from.iter().map(|Zn| Zn.int_hom().map(input)).collect::<Vec<_>>();
        let output = to.iter().map(|Zn| Zn.int_hom().map(output)).collect::<Vec<_>>();
        let mut actual = to.iter().map(|Zn| Zn.zero()).collect::<Vec<_>>();

        rescaling.apply(Submatrix::from_1d(&input, 3, 1), SubmatrixMut::from_1d(&mut actual, 2, 1));

        for j in 0..output.len() {
            // we currently assume no error happens
            assert_el_eq!(to.at(j), output.at(j), actual.at(j));
        }        
    }
}
