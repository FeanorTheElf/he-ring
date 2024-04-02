use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::homomorphism::*;
use feanor_math::matrix::submatrix::*;
use feanor_math::mempool::*;
use feanor_math::primitive_int::*;
use feanor_math::rings::zn::*;
use feanor_math::integer::int_cast;
use feanor_math::integer::*;
use feanor_math::ring::*;
use feanor_math::vector::subvector::*;
use feanor_math::vector::*;

use crate::rnsconv::lift::AlmostExactBaseConversion;

use super::RNSOperation;

const ZZbig: BigIntRing = BigIntRing::RING;

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
pub struct CongruencePreservingRescaling<V, R, R_intermediate, M_Zn, M_Int>
    where V: VectorView<R>,
        R: ZnRingStore, R_intermediate: ZnRingStore<Type = R::Type>,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<StaticRingBase<i32>>,
        M_Zn: MemoryProvider<El<R>>,
        M_Int: MemoryProvider<El<<R::Type as ZnRing>::Integers>>
{
    b_moduli_count: usize,
    q_moduli_count: usize,
    /// contains all the moduli, in the order: moduli of `q` first, then moduli of `q'`
    b_to_aq_lift: AlmostExactBaseConversion<Subvector<R, V>, V, R, R_intermediate, M_Zn, M_Int>,
    aq_to_t_conv: AlmostExactBaseConversion<V, [R; 1], R, R_intermediate, M_Zn, M_Int>,
    memory_provider: M_Zn,
    /// `a` as an element of each modulus of `q`
    a: Vec<El<R>>,
    /// `b^-1` as an element of each modulus of `aq/b`
    b_inv: Vec<El<R>>,
    /// `b^-1` as an element of `Z/tZ`
    b_inv_mod_t: El<R>
}

impl<R, R_intermediate, M_Zn, M_Int> CongruencePreservingRescaling<Vec<R>, R, R_intermediate, M_Zn, M_Int>
    where R: ZnRingStore + Clone, 
        R_intermediate: ZnRingStore<Type = R::Type> + Clone,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<StaticRingBase<i32>>,
        M_Zn: MemoryProvider<El<R>> + Clone,
        M_Int: MemoryProvider<El<<R::Type as ZnRing>::Integers>> + Clone
{
    pub fn scale_down(q_moduli: Vec<R>, drop_moduli: usize, plaintext_modulus: R, Zm_intermediate: R_intermediate, memory_provider: M_Zn, memory_provider_int: M_Int) -> Self {
        Self::new(q_moduli, Vec::new(), drop_moduli, plaintext_modulus, Zm_intermediate, memory_provider, memory_provider_int)
    }

    pub fn new(in_moduli: Vec<R>, num_moduli: Vec<R>, denominator_count: usize, plaintext_modulus: R, Zm_intermediate: R_intermediate, memory_provider: M_Zn, memory_provider_int: M_Int) -> Self {
        let a = ZZbig.prod(num_moduli.iter().map(|R| int_cast(R.integer_ring().clone_el(R.modulus()), &ZZbig, R.integer_ring())));
        let b = ZZbig.prod(in_moduli.iter().take(denominator_count).map(|R| int_cast(R.integer_ring().clone_el(R.modulus()), &ZZbig, R.integer_ring())));
        
        let a_mod: Vec<_> = in_moduli.iter().map(|R| R.coerce(&ZZbig, ZZbig.clone_el(&a))).collect();
        let b_inv_mod = in_moduli.iter().skip(denominator_count).chain(num_moduli.iter()).map(|R| R.invert(&R.coerce(&ZZbig, ZZbig.clone_el(&b))).unwrap()).collect();

        let b_moduli = in_moduli.iter().cloned().take(denominator_count).collect::<Vec<_>>();
        let aq_moduli = in_moduli.into_iter().chain(num_moduli.into_iter()).collect::<Vec<_>>();
        Self {
            q_moduli_count: a_mod.len(),
            b_moduli_count: denominator_count,
            a: a_mod,
            b_inv: b_inv_mod,
            b_inv_mod_t: plaintext_modulus.invert(&plaintext_modulus.coerce(&ZZbig, b)).unwrap(),
            b_to_aq_lift: AlmostExactBaseConversion::new(Subvector::new(b_moduli), aq_moduli.clone(), Zm_intermediate.clone(), memory_provider_int.clone(), memory_provider.clone()),
            aq_to_t_conv: AlmostExactBaseConversion::new(aq_moduli, [plaintext_modulus], Zm_intermediate, memory_provider_int, memory_provider.clone()),
            memory_provider: memory_provider
        }
    }
}

impl<V, R, R_intermediate, M_Zn, M_Int> CongruencePreservingRescaling<V, R, R_intermediate, M_Zn, M_Int>
    where V: VectorView<R>,
        R: ZnRingStore, R_intermediate: ZnRingStore<Type = R::Type>,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<StaticRingBase<i32>>,
        M_Zn: MemoryProvider<El<R>>,
        M_Int: MemoryProvider<El<<R::Type as ZnRing>::Integers>>
{
    fn q_moduli<'a>(&'a self) -> Subvector<R, &'a V> {
        Subvector::new(self.aq_to_t_conv.input_rings()).subvector(..self.q_moduli_count)
    }

    fn aq_over_b_moduli<'a>(&'a self) -> Subvector<R, &'a V> {
        Subvector::new(self.aq_to_t_conv.input_rings()).subvector(self.b_moduli_count..)
    }

    fn aq_moduli<'a>(&'a self) -> &'a V {
        self.aq_to_t_conv.input_rings()
    }

    fn t_modulus<'a>(&'a self) -> &'a R {
        self.aq_to_t_conv.output_rings().at(0)
    }
}

impl<V, R, R_intermediate, M_Zn, M_Int> RNSOperation for CongruencePreservingRescaling<V, R, R_intermediate, M_Zn, M_Int>
    where V: VectorView<R>,
        R: ZnRingStore, R_intermediate: ZnRingStore<Type = R::Type>,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<StaticRingBase<i32>>,
        M_Zn: MemoryProvider<El<R>>,
        M_Int: MemoryProvider<<<<R as RingStore>::Type as ZnRing>::IntegerRingBase as RingBase>::Element>
{
    type Ring = R;

    type RingType = R::Type;

    type InRings<'a> = Subvector<R, &'a V>
        where Self: 'a;

    type OutRings<'a> = Subvector<R, &'a V>
        where Self: 'a;

    fn input_rings<'a>(&'a self) -> Self::InRings<'a> {
        self.q_moduli()
    }

    fn output_rings<'a>(&'a self) -> Self::OutRings<'a> {
        self.aq_over_b_moduli()
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
        let mut x = self.memory_provider.get_new_init(self.aq_moduli().len() * input.col_count(), |idx| {
            let i = idx / input.col_count();
            let j = idx % input.col_count();
            if i < self.q_moduli().len() {
                self.q_moduli().at(i).mul_ref(input.at(i, j), &self.a.at(i))
            } else {
                self.aq_moduli().at(i).zero()
            }
        });
        let mut x = SubmatrixMut::<AsFirstElement<_>, _>::new(&mut x, self.aq_moduli().len(), input.col_count());

        let x_mod_b = x.as_const().restrict_rows(0..self.b_moduli_count);

        // Compute `lift(x mod b)`; Here we introduce an error of `+/- b`
        let mut x_mod_b_lift = self.memory_provider.get_new_init(self.aq_moduli().len() * input.col_count(), |idx| self.aq_moduli().at(idx / input.col_count()).zero());
        let mut x_mod_b_lift = SubmatrixMut::<AsFirstElement<_>, _>::new(&mut x_mod_b_lift, self.aq_moduli().len(), input.col_count());
        self.b_to_aq_lift.apply(x_mod_b, x_mod_b_lift.reborrow());

        // Make `x` divisible by `b` by subtracting `lift(x mod b)`
        for (i, Zk) in self.aq_moduli().iter().enumerate() {
            for j in 0..input.col_count() {
                Zk.sub_assign_ref(x.at(i, j), x_mod_b_lift.at(i, j));
            }
        }

        // now we have to ensure congruence; the difference between `x = el * a` and `b * exactdiv(x, b)` is exactly `x_mod_b_lift`;
        // this is small, so no error here
        let mut diff_mod_t = self.memory_provider.get_new_init(input.col_count(), |_j| self.t_modulus().zero());
        let mut diff_mod_t = SubmatrixMut::<AsFirstElement<_>, _>::new(&mut diff_mod_t, 1, input.col_count());
        self.aq_to_t_conv.apply(x_mod_b_lift.as_const(), diff_mod_t.reborrow());
        for j in 0..input.col_count() {
            self.t_modulus().mul_assign_ref(diff_mod_t.at(0, j), &self.b_inv_mod_t);
        }

        // this is now `round((aq/b) * el / q)`, possibly `+/- 1`
        for (i, Zk) in self.aq_over_b_moduli().iter().enumerate() {
            let modulo = Zk.can_hom(self.t_modulus().integer_ring()).unwrap();
            debug_assert!(Zk.integer_ring().get_ring() == self.t_modulus().integer_ring().get_ring());
            for j in 0..input.col_count() {
                *output.at(i, j) = Zk.mul_ref(x.at(i + self.b_moduli_count, j), self.b_inv.at(i));

                // fix the congruence
                Zk.add_assign(output.at(i, j), modulo.map(self.t_modulus().smallest_lift(self.t_modulus().clone_el(diff_mod_t.at(0, j)))));
            }
        }
    }
}

#[cfg(test)]
use feanor_math::rings::zn::zn_42::*;
#[cfg(test)]
use feanor_math::{assert_el_eq, default_memory_provider};

#[test]
fn test_rescale() {
    let from = vec![Zn::new(17), Zn::new(97), Zn::new(113)];
    let to = vec![Zn::new(19), Zn::new(257)];
    let Zt = Zn::new(5);
    let q = 17 * 97 * 113;
    let qprime = 19 * 257;

    let rescaling = CongruencePreservingRescaling::new(
        from.clone(), 
        to.clone(), 
        3,
        Zt.clone(), 
        Zn::new(65537), 
        default_memory_provider!(), 
        default_memory_provider!()
    );

    let ZZ_to_Zt = Zt.int_hom();
    // since Zm_intermediate has a very large modulus, we can ignore the `+/- 1` error here at the moment (I think)
    for i in -(q/2)..=(q/2) {
        let input = i;
        let rescaled = (input as f64 * qprime as f64 / q as f64).round() as i32;
        let output = rescaled + Zt.smallest_lift(Zt.sub(
            Zt.checked_div(&ZZ_to_Zt.map(input * qprime), &ZZ_to_Zt.map(q)).unwrap(),
            ZZ_to_Zt.map(rescaled)
        )) as i32;

        assert!(Zt.is_zero(&ZZ_to_Zt.map(input * qprime - output * q)));

        let input = from.iter().map(|R| R.int_hom().map(input)).collect::<Vec<_>>();
        let output = to.iter().map(|R| R.int_hom().map(output)).collect::<Vec<_>>();
        let mut actual = to.iter().map(|R| R.zero()).collect::<Vec<_>>();

        rescaling.apply(Submatrix::<AsFirstElement<_>, _>::new(&input, 3, 1), SubmatrixMut::<AsFirstElement<_>, _>::new(&mut actual, 2, 1));

        for j in 0..output.len() {
            assert_el_eq!(to.at(j), output.at(j), actual.at(j));
        }        
    }
}

#[test]
fn test_rescale_down() {
    let from = vec![Zn::new(17), Zn::new(97), Zn::new(113)];
    let to = vec![Zn::new(97), Zn::new(113)];
    let Zt = Zn::new(5);
    let q = 17 * 97 * 113;
    let qprime = 97 * 113;

    let rescaling = CongruencePreservingRescaling::scale_down(
        from.clone(), 
        1,
        Zt.clone(), 
        Zn::new(65537), 
        default_memory_provider!(), 
        default_memory_provider!()
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

        let input = from.iter().map(|R| R.int_hom().map(input)).collect::<Vec<_>>();
        let output = to.iter().map(|R| R.int_hom().map(output)).collect::<Vec<_>>();
        let mut actual = to.iter().map(|R| R.zero()).collect::<Vec<_>>();

        rescaling.apply(Submatrix::<AsFirstElement<_>, _>::new(&input, 3, 1), SubmatrixMut::<AsFirstElement<_>, _>::new(&mut actual, 2, 1));

        for j in 0..output.len() {
            assert_el_eq!(to.at(j), output.at(j), actual.at(j));
        }        
    }
}
