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
    aq_moduli: Vec<Zn>,
    b_moduli_count: usize,
    q_moduli_count: usize,
    /// `aq_moduli[i] = aq_moduli_sorted[aq_permutation[i]]`
    aq_permutation: Vec<usize>,
    /// contains all the moduli, but sorted
    b_to_aq_lift: UsedBaseConversion<A>,
    aq_to_t_conv: UsedBaseConversion<A>,
    allocator: A,
    /// `a` as an element of each modulus of `q`
    a: Vec<El<Zn>>,
    /// `b^-1` as an element of each modulus of `aq/b`
    b_inv: Vec<El<Zn>>,
    /// `b^-1` as an element of `Z/tZ`
    b_inv_mod_t: El<Zn>
}

impl<A> CongruencePreservingRescaling<A>
    where A: Allocator + Clone
{
    pub fn scale_down(q_moduli: Vec<Zn>, den_moduli_count: usize, plaintext_modulus: Zn, allocator: A) -> Self {
        Self::new_with(q_moduli, Vec::new(), den_moduli_count, plaintext_modulus, allocator)
    }

    ///
    /// Creates a new [`CongruencePreservingRescaling`], where
    ///  - `q` is the product of `in_moduli`
    ///  - `a` is the product of `num_moduli`
    ///  - `b` is the product of the first `den_moduli_count` elements of `in_moduli`
    /// At least the moduli belonging to `b` are expected to be sorted.
    /// 
    pub fn new_with(in_moduli: Vec<Zn>, num_moduli: Vec<Zn>, den_moduli_count: usize, plaintext_modulus: Zn, allocator: A) -> Self {
        let ZZ = plaintext_modulus.integer_ring();
        for ring in &in_moduli {
            assert!(ring.integer_ring().get_ring() == ZZ.get_ring());
        }
        for ring in &num_moduli {
            assert!(ring.integer_ring().get_ring() == ZZ.get_ring());
        }
        
        let a = ZZbig.prod(num_moduli.iter().map(|Zn| int_cast(Zn.integer_ring().clone_el(Zn.modulus()), &ZZbig, Zn.integer_ring())));
        let b = ZZbig.prod(in_moduli.iter().take(den_moduli_count).map(|Zn| int_cast(Zn.integer_ring().clone_el(Zn.modulus()), &ZZbig, Zn.integer_ring())));
        
        let a_mod: Vec<_> = in_moduli.iter().map(|Zn| Zn.coerce(&ZZbig, ZZbig.clone_el(&a))).collect();
        let b_inv_mod = in_moduli.iter().skip(den_moduli_count).chain(num_moduli.iter()).map(|Zn| Zn.invert(&Zn.coerce(&ZZbig, ZZbig.clone_el(&b))).unwrap()).collect();

        let b_moduli = in_moduli.iter().cloned().take(den_moduli_count).collect::<Vec<_>>();
        let aq_moduli = in_moduli.into_iter().chain(num_moduli.into_iter()).collect::<Vec<_>>();
        let (aq_moduli_sorted, aq_permutation) = sort_unstable_permutation(aq_moduli.clone(), |ring_l, ring_r| ZZ.cmp(ring_l.modulus(), ring_r.modulus()));
        Self {
            q_moduli_count: a_mod.len(),
            b_moduli_count: den_moduli_count,
            a: a_mod,
            b_inv: b_inv_mod,
            aq_moduli: aq_moduli,
            aq_permutation: aq_permutation,
            b_inv_mod_t: plaintext_modulus.invert(&plaintext_modulus.coerce(&ZZbig, b)).unwrap(),
            b_to_aq_lift: UsedBaseConversion::new_with(b_moduli, aq_moduli_sorted.iter().cloned().collect(), allocator.clone()),
            aq_to_t_conv: UsedBaseConversion::new_with(aq_moduli_sorted, vec![plaintext_modulus], allocator.clone()),
            allocator: allocator
        }
    }

    fn t_modulus(&self) -> &Zn {
        &self.aq_to_t_conv.output_rings()[0]
    }

    fn aq_moduli(&self) -> &[Zn] {
        &self.aq_moduli
    }
}

impl<A> RNSOperation for CongruencePreservingRescaling<A>
    where A: Allocator + Clone
{
    type Ring = Zn;

    type RingType = ZnBase;

    fn input_rings<'a>(&'a self) -> &'a [Zn] {
        &self.aq_moduli()[..self.q_moduli_count]
    }

    fn output_rings<'a>(&'a self) -> &'a [Zn] {
        &self.aq_moduli()[self.b_moduli_count..]
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
        assert_eq!(input.row_count(), self.input_rings().len());
        assert_eq!(output.row_count(), self.output_rings().len());
        assert_eq!(input.col_count(), output.col_count());

        // Compute `x := el * a`
        let mut x = Vec::with_capacity_in(self.aq_moduli().len() * input.col_count(), self.allocator.clone());
        x.extend((0..(self.aq_moduli().len() * input.col_count())).map(|idx| {
            let i = idx / input.col_count();
            let j = idx % input.col_count();
            if i < self.input_rings().len() {
                self.input_rings().at(i).mul_ref(input.at(i, j), &self.a.at(i))
            } else {
                self.aq_moduli().at(i).zero()
            }
        }));
        let mut x = SubmatrixMut::<AsFirstElement<_>, _>::new(&mut x, self.aq_moduli().len(), input.col_count());

        let x_mod_b = x.as_const().restrict_rows(0..self.b_moduli_count);

        // Compute `lift(x mod b)`; Here we introduce an error of `+/- b`
        let mut x_mod_b_lift = Vec::with_capacity_in(self.aq_moduli().len() * input.col_count(), self.allocator.clone());
        x_mod_b_lift.extend((0..(self.aq_moduli().len() * input.col_count())).map(|idx| self.aq_moduli().at(idx / input.col_count()).zero()));
        let mut x_mod_b_lift = SubmatrixMut::<AsFirstElement<_>, _>::new(&mut x_mod_b_lift, self.aq_moduli().len(), input.col_count());
        self.b_to_aq_lift.apply(x_mod_b, x_mod_b_lift.reborrow());
        // `x_mod_b_lift` is ordered according to `self.aq_moduli_sorted()`, not `self.aq_moduli()`

        // Make `x` divisible by `b` by subtracting `lift(x mod b)`
        for (i, Zk) in self.aq_moduli().iter().enumerate() {
            for j in 0..input.col_count() {
                Zk.sub_assign_ref(x.at_mut(i, j), x_mod_b_lift.at(self.aq_permutation[i], j));
            }
        }

        // now we have to ensure congruence; the difference between `x = el * a` and `b * exactdiv(x, b)` is exactly `x_mod_b_lift`;
        // this is small, so no error here
        let mut diff_mod_t = Vec::with_capacity_in(input.col_count(), self.allocator.clone());
        diff_mod_t.extend((0..input.col_count()).map(|_j| self.t_modulus().zero()));
        let mut diff_mod_t = SubmatrixMut::<AsFirstElement<_>, _>::new(&mut diff_mod_t, 1, input.col_count());
        self.aq_to_t_conv.apply(x_mod_b_lift.as_const(), diff_mod_t.reborrow());
        for j in 0..input.col_count() {
            self.t_modulus().mul_assign_ref(diff_mod_t.at_mut(0, j), &self.b_inv_mod_t);
        }

        // this is now `round((aq/b) * el / q)`, possibly `+/- 1`
        for (i, Zk) in self.output_rings().iter().enumerate() {
            let modulo = Zk.can_hom(self.t_modulus().integer_ring()).unwrap();
            debug_assert!(Zk.integer_ring().get_ring() == self.t_modulus().integer_ring().get_ring());
            for j in 0..input.col_count() {
                *output.at_mut(i, j) = Zk.mul_ref(x.at(i + self.b_moduli_count, j), self.b_inv.at(i));

                // fix the congruence
                Zk.add_assign(output.at_mut(i, j), modulo.map(self.t_modulus().smallest_lift(self.t_modulus().clone_el(diff_mod_t.at(0, j)))));
            }
        }
    }
}

#[cfg(test)]
use feanor_math::assert_el_eq;

#[test]
fn test_rescale() {
    let from = vec![Zn::new(17), Zn::new(23), Zn::new(113)];
    let to = vec![Zn::new(19), Zn::new(257)];
    let Zt = Zn::new(5);
    let q = 17 * 23 * 113;
    let qprime = 19 * 257;

    let rescaling = CongruencePreservingRescaling::new_with(
        from.clone(), 
        to.clone(), 
        3,
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

        rescaling.apply(Submatrix::<AsFirstElement<_>, _>::new(&input, 3, 1), SubmatrixMut::<AsFirstElement<_>, _>::new(&mut actual, 2, 1));

        for j in 0..output.len() {
            assert!(
                to.at(j).eq_el(output.at(j), actual.at(j)) ||
                to.at(j).eq_el(&to.at(j).add_ref_fst(output.at(j), to.at(j).int_hom().map(5)), actual.at(j)) ||
                to.at(j).eq_el(&to.at(j).sub_ref_fst(output.at(j), to.at(j).int_hom().map(5)), actual.at(j))
            );
        }        
    }
}

#[test]
fn test_rescale_down() {
    let from = vec![Zn::new(17), Zn::new(23), Zn::new(113)];
    let to = vec![Zn::new(23), Zn::new(113)];
    let Zt = Zn::new(5);
    let q = 17 * 23 * 113;
    let qprime = 23 * 113;

    let rescaling = CongruencePreservingRescaling::scale_down(
        from.clone(), 
        1,
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

        rescaling.apply(Submatrix::<AsFirstElement<_>, _>::new(&input, 3, 1), SubmatrixMut::<AsFirstElement<_>, _>::new(&mut actual, 2, 1));

        for j in 0..output.len() {
            // we currently assume no error happens
            assert_el_eq!(to.at(j), output.at(j), actual.at(j));
        }        
    }
}
