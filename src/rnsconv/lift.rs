use feanor_math::integer::*;
use feanor_math::matrix::submatrix::*;
use feanor_math::mempool::*;
use feanor_math::homomorphism::*;
use feanor_math::vector::VectorView;
use feanor_math::ordered::OrderedRingStore;
use feanor_math::rings::zn::{ZnRingStore, ZnRing};
use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::ring::*;

use super::RNSOperation;

///
/// Stores values for an almost exact conversion between RNS bases.
/// A complete conversion refers to the function
/// ```text
/// Z/QZ -> Z/Q'Z, x -> [lift(x)]
/// ```
/// In our case, the output of the function is allowed to have an error of `{ -Q, 0, Q }`,
/// unless the shortest lift of the input is bounded by `Q/4`, in which case the result
/// is always correct.
/// 
/// # Implementation
/// 
/// The implementation is based on ["A Full RNS Variant of FV like Somewhat
/// Homomorphic Encryption Schemes"](eprint.iacr.org/2016/510.pdf).
/// 
/// The basic idea is to perform a fast base conversion to `{ to_summands, m }`
/// with a helper prime m. Furthermore, we can do this in a way to get the result
/// in montgomery form, i.e. get `x_q m` instead of `x_q`. Hence, we end up with y
/// w.r.t. `{ to_summands, m }` such that
/// ```text
/// y in [shortest_lift(x)]_Q' m + Q { -k/2, -k/2 + 1, ..., k/2 }
/// ```
/// From this, we can now perform a montgomery reduction to find `[shortest_lift(x)]_Q'`
/// 
pub struct AlmostExactBaseConversion<V_from, V_to, R, R_intermediate, M_Zn, M_Int>
    where V_from: VectorView<R>, V_to: VectorView<R>,
        R: ZnRingStore, R_intermediate: ZnRingStore<Type = R::Type>,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        M_Zn: MemoryProvider<El<R>>,
        M_Int: MemoryProvider<El<<R::Type as ZnRing>::Integers>>
{
    from_summands: V_from,
    to_summands: V_to,
    // F_m for the helper factor m
    Zm: R_intermediate,
    /// the values `m q/Q mod q` for each RNS factor q dividing Q
    qm_over_Q: Vec<El<R>>,
    /// the values `Q/q mod q'` for each RNS factor q dividing Q and q' dividing Q'
    Q_over_q: Vec<El<R>>,
    // the value `Q^-1 mod m`
    Q_inv_Zm: El<R>,
    // the value `Q mod q'` for q' dividing Q'
    Q_Zq: Vec<El<R>>,
    // the value `m^-1 mod q'` for q' dividing Q'
    m_inv_Zq: Vec<El<R>>,
    memory_provider_int: M_Int,
    memory_provider_zn: M_Zn
}

const ZZbig: BigIntRing = BigIntRing::RING;

///
/// Computes the inner product `Z^n x R^n -> R` where one side consists of integers.
/// 
fn inner_product<V, R_to, I, H>(lhs: Column<V, I::Element>, rhs: &[R_to::Element], hom: &H) -> R_to::Element
    where 
        I: ?Sized + IntegerRing,
        H: Homomorphism<I, R_to>,
        R_to: ?Sized + ZnRing,
        V: AsPointerToSlice<I::Element>
{
    debug_assert_eq!(lhs.len(), rhs.len());
    <_ as RingStore>::sum(hom.codomain(),
        lhs.iter()
            .zip(rhs.iter())
            .map(|(x, y)| hom.mul_ref_map(y, x))
    )
}

impl<V_from, V_to, R, R_intermediate, M_Zn, M_Int> AlmostExactBaseConversion<V_from, V_to, R, R_intermediate, M_Zn, M_Int> 
    where V_from: VectorView<R>, V_to: VectorView<R>,
        R: ZnRingStore, R_intermediate: ZnRingStore<Type = R::Type>,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        M_Zn: MemoryProvider<El<R>>,
        M_Int: MemoryProvider<El<<R::Type as ZnRing>::Integers>>
{
    pub fn new(from_summands: V_from, to_summands: V_to, Zm: R_intermediate, memory_provider_int: M_Int, memory_provider_zn: M_Zn) -> Self {
        assert!(from_summands.len() > 0);
        assert!(to_summands.len() > 0);
        // we need `2 * ` such that the almost-exact conversion is an exact conversion in the case of small input
        assert!(Zm.integer_ring().is_gt(Zm.modulus(), &Zm.integer_ring().int_hom().map(2 * from_summands.len() as i32)));
        
        let ZZ = Zm.integer_ring();
        // we want to use integers without casts, so check ring equality here
        for Zk in from_summands.iter() {
            assert!(Zk.integer_ring().get_ring() == ZZ.get_ring());
        }
        for Zk in to_summands.iter() {
            assert!(Zk.integer_ring().get_ring() == ZZ.get_ring());
        }

        let m = int_cast(Zm.integer_ring().clone_el(Zm.modulus()), &ZZ, Zm.integer_ring());

        let from_moduli = from_summands.iter().map(|R| ZZ.clone_el(R.modulus()));

        let Q = ZZbig.prod(from_moduli.clone().map(|n| int_cast(n, &ZZbig, ZZ)));

        let q_over_Q = from_summands.iter()
            .zip(from_moduli.clone())
            .map(|(R, n)| R.invert(&R.coerce(&ZZbig, ZZbig.checked_div(&Q, &int_cast(n, &ZZbig, ZZ)).unwrap())).unwrap())
            .collect::<Vec<_>>();

        let qm_over_Q = from_summands.iter()
            .zip(q_over_Q.iter())
            .map(|(R, x)| R.mul_ref_fst(x, R.coerce(ZZ, ZZ.clone_el(&m))))
            .collect::<Vec<_>>();

        let Q_over_q = to_summands.iter()
            .flat_map(|to| from_moduli.clone()
                .map(|n| ZZbig.checked_div(&Q, &int_cast(n, &ZZbig, ZZ)).unwrap())
                .map(|x| to.coerce(&ZZbig, x))
            )
            .chain(from_moduli.clone()
                .map(|n| ZZbig.checked_div(&Q, &int_cast(n, &ZZbig, ZZ)).unwrap())
                .map(|x| Zm.coerce(&ZZbig, x))
            ).collect::<Vec<_>>();

        let Q_inv_Zm = Zm.invert(&Zm.coerce(&ZZbig, ZZbig.clone_el(&Q))).unwrap();

        let Q_Zq = to_summands.iter().map(|R| R.coerce(&ZZbig, ZZbig.clone_el(&Q))).collect();
        
        let m_inv_Zq = to_summands.iter().map(|R| R.invert(&R.coerce(R.integer_ring(), R.integer_ring().coerce(&ZZ, ZZ.clone_el(&m)))).unwrap()).collect();

        AlmostExactBaseConversion { from_summands, to_summands, Zm, qm_over_Q, Q_over_q, Q_inv_Zm, Q_Zq, m_inv_Zq, memory_provider_int, memory_provider_zn }
    }

    ///
    /// Performs a variant of the fast base conversion to the intermediate 
    /// RNS base `{ to_summands, m }`. The only difference is that the result
    /// will be in Montgomery form, i.e. is
    /// ```text
    /// shortest_lift(x m mod Q) + Q { -k/2, -k/2 + 1, ..., k/2 }
    /// ```
    /// modulo `{ to_summands, m }`.
    /// 
    #[inline(never)]
    fn fast_convert_assign_montgomery<V1, V2>(&self, input: Submatrix<V1, El<R>>, mut target: SubmatrixMut<V2, El<R>>, mut target_Zm: SubmatrixMut<AsFirstElement<El<R_intermediate>>, El<R_intermediate>>)
        where V1: AsPointerToSlice<El<R>>,
            V2: AsPointerToSlice<El<R>> 
    {
        timed!("fast_convert_assign_montgomery", || {
            debug_assert_eq!(input.row_count(), self.from_summands.len());
            debug_assert_eq!(self.qm_over_Q.len(), self.from_summands.len());
            debug_assert!(target.row_count() == self.to_summands.len());
            debug_assert!(self.Q_over_q.len() == (self.to_summands.len() + 1) * self.from_summands.len());
            debug_assert_eq!(input.col_count(), target.col_count());

            // this is the same for all rings, as checked in [`Self::new()`]
            let ZZ = self.Zm.integer_ring();

            let el_qm_over_Q = self.memory_provider_int.get_new_init(self.from_summands.len() * input.col_count(), |idx| {
                let i = idx / input.col_count();
                let j = idx % input.col_count();
                self.from_summands.at(i).smallest_lift(self.from_summands.at(i).mul_ref(input.at(i, j), self.qm_over_Q.at(i)))
            });
            let el_qm_over_Q = Submatrix::<AsFirstElement<_>, _>::new(&el_qm_over_Q, input.row_count(), input.col_count());

            let Q_over_q = Submatrix::<AsFirstElement<_>, _>::new(&self.Q_over_q[..], self.to_summands.len() + 1, self.from_summands.len());
            // Now do matmul Q_over_q * el_qm_over_Q
            for i in 0..self.to_summands.len() {
                for j in 0..input.col_count() {
                    *target.at(i, j) = inner_product(el_qm_over_Q.col_at(j), Q_over_q.row_at(i), &self.to_summands.at(i).can_hom(&ZZ).unwrap());
                }
            }
            for j in 0..input.col_count() {
                *target_Zm.at(0, j) = inner_product(el_qm_over_Q.col_at(j), Q_over_q.row_at(self.to_summands.len()), &self.Zm.can_hom(&ZZ).unwrap());
            }
        });
    }

    ///
    /// Performs the reduction mod q of the output of the intermediate fast base conversion
    ///
    #[inline(never)]
    fn reduce_mod_q_inplace<V1>(&self, mut target: SubmatrixMut<V1, El<R>>, mut helper: SubmatrixMut<AsFirstElement<El<R_intermediate>>, El<R_intermediate>>)
        where V1: AsPointerToSlice<El<R>>
    {
        debug_assert!(target.row_count() == self.to_summands.len());

        // Lemma 4 in
        // "A Full RNS Variant of FV like Somewhat Homomorphic Encryption Schemes",
        // Jean-Claude Bajard, Julien Eynard, Anwar Hasan, and Vincent Zucca,
        // eprint.iacr.org/2016/510.pdf
        //
        // The input is `x = [cm]_q + uq`, where c is the smallest representative mod q
        // and `|u| <= #Q/2`;
        // Write `[cm]_q = cm + wq`, so `|w| < m/2`. Now we have
        //```text
        // [cm]_q + uq = cm + (u + w)q
        //```
        // and hence find `u + w = x q^-1 mod m`. This allows us to determine `u + w` up to `+/- 1`,
        // as `|u + w| <= m/2 + #Q/2 <= m`, assuming that `m > #Q`.
        //
        // Furthermore, if `c <= q/4` then `[cm]_q = cm + wq` for `|w| <= m/4`. Thus `|u + w| <= m/2`, assuming
        // that `m >= 2#Q`.

        let ZZ = self.Zm.integer_ring();
        for j in 0..target.col_count() {
            let r = self.Zm.smallest_lift(self.Zm.mul_ref(helper.at(0, j), &self.Q_inv_Zm));
            for i in 0..self.to_summands.len() {
                let diff = self.to_summands.at(i).mul_ref_fst(&self.Q_Zq[i], self.to_summands.at(i).coerce(ZZ, ZZ.clone_el(&r)));
                self.to_summands.at(i).sub_assign(target.at(i, j), diff);
                self.to_summands.at(i).mul_assign_ref(target.at(i, j), &self.m_inv_Zq[i]);
            }
        }
    }
}


impl<V_from, V_to, R, R_intermediate, M_Zn, M_Int> RNSOperation for AlmostExactBaseConversion<V_from, V_to, R, R_intermediate, M_Zn, M_Int> 
    where V_from: VectorView<R>, V_to: VectorView<R>,
        R: ZnRingStore, R_intermediate: ZnRingStore<Type = R::Type>,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        M_Zn: MemoryProvider<El<R>>,
        M_Int: MemoryProvider<<<<R as RingStore>::Type as ZnRing>::IntegerRingBase as RingBase>::Element>
{
    type Ring = R;

    type RingType = R::Type;

    type InRings<'a> = &'a V_from
        where Self: 'a;

    type OutRings<'a> = &'a V_to
        where Self: 'a;

    fn input_rings<'a>(&'a self) -> Self::InRings<'a> {
        &self.from_summands
    }

    fn output_rings<'a>(&'a self) -> Self::OutRings<'a> {
        &self.to_summands
    }

    ///
    /// Performs the (almost) exact RNS base conversion
    /// ```text
    ///     Z/QZ -> Z/Q'Z, x -> smallest_lift(x) + kQ mod Q''
    /// ```
    /// where `k in { -1, 0, 1 }`.
    /// 
    /// Furthermore, if the shortest lift of the input is bounded by `Q/4`,
    /// then the result is guaranteed to be exact.
    /// 
    fn apply<V1, V2>(&self, input: Submatrix<V1, El<Self::Ring>>, mut output: SubmatrixMut<V2, El<Self::Ring>>)
        where V1: AsPointerToSlice<El<Self::Ring>>,
            V2: AsPointerToSlice<El<Self::Ring>> 
    {
        let mut helper = self.memory_provider_zn.get_new_init(input.col_count(), |_| self.Zm.zero());
        let mut helper = SubmatrixMut::<AsFirstElement<_>, _>::new(&mut helper, 1, input.col_count());
        self.fast_convert_assign_montgomery(input, output.reborrow(), helper.reborrow());
        self.reduce_mod_q_inplace(output.reborrow(), helper.reborrow());
    }
}

#[cfg(test)]
use feanor_math::rings::zn::zn_64::*;
#[cfg(test)]
use feanor_math::{assert_el_eq, default_memory_provider};

#[test]
fn test_rns_base_conversion() {
    let from = vec![Zn::new(17), Zn::new(97)];
    let to = vec![Zn::new(17), Zn::new(97), Zn::new(113), Zn::new(257)];

    let table = AlmostExactBaseConversion::new(from.clone(), to.clone(), Zn::new(65537), default_memory_provider!(), default_memory_provider!());

    for k in -412..=412 {
        let input = from.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();
        let expected = to.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();
        let mut actual = to.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();

        table.apply(Submatrix::<AsFirstElement<_>, _>::new(&input, 2, 1), SubmatrixMut::<AsFirstElement<_>, _>::new(&mut actual, 4, 1));

        for j in 0..to.len() {
            assert_el_eq!(to.at(j), expected.at(j), actual.at(j));
        }
    }
}

#[test]
fn test_rns_base_conversion_small_helper() {
    let from = vec![Zn::new(97), Zn::new(3)];
    let to = vec![Zn::new(17)];
    let table = AlmostExactBaseConversion::new(from.clone(), to.clone(), Zn::new(5), default_memory_provider!(), default_memory_provider!());
    
    for k in 0..291 {
        let mut actual = to.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();
        table.apply(Submatrix::<AsFirstElement<_>, _>::new(&[from[0].int_hom().map(k), from[1].int_hom().map(k)], 2, 1), SubmatrixMut::<AsFirstElement<_>, _>::new(&mut actual, 1, 1));

        assert!(
            to[0].eq_el(&to[0].int_hom().map(k), actual.at(0)) ||
            to[0].eq_el(&to[0].int_hom().map(k + 97 * 3), actual.at(0)) ||
            to[0].eq_el(&to[0].int_hom().map(k - 97 * 3), actual.at(0))
        );
    }
}

#[test]
fn test_rns_base_conversion_not_coprime() {
    let from = vec![Zn::new(17), Zn::new(97), Zn::new(113)];
    let to = vec![Zn::new(17), Zn::new(97), Zn::new(113), Zn::new(257)];
    let table = AlmostExactBaseConversion::new(from.clone(), to.clone(), Zn::new(65537), default_memory_provider!(), default_memory_provider!());

    for k in &[0, 1, 2, 17, 97, 113, 17 * 113, 18, 98, 114] {
        let x = from.iter().map(|R| R.int_hom().map(*k)).collect::<Vec<_>>();
        let y = to.iter().map(|R| R.int_hom().map(*k)).collect::<Vec<_>>();
        let mut actual = to.iter().map(|R| R.int_hom().map(*k)).collect::<Vec<_>>();

        table.apply(Submatrix::<AsFirstElement<_>, _>::new(&x, 3, 1), SubmatrixMut::<AsFirstElement<_>, _>::new(&mut actual, 4, 1));
        
        for i in 0..y.len() {
            assert!(to[i].eq_el(&y[i], actual.at(i)));
        }
    }
}

#[test]
fn test_rns_base_conversion_coprime() {
    let from = vec![Zn::new(17), Zn::new(97), Zn::new(113)];
    let to = vec![Zn::new(19), Zn::new(23), Zn::new(257)];
    let table = AlmostExactBaseConversion::new(from.iter().cloned().collect::<Vec<_>>(), to.iter().cloned().collect::<Vec<_>>(), Zn::new(65537), default_memory_provider!(), default_memory_provider!());

    for k in &[0, 1, 2, 17, 97, 113, 17 * 113, 18, 98, 114] {
        let x = from.iter().map(|R| R.int_hom().map(*k)).collect::<Vec<_>>();
        let y = to.iter().map(|R| R.int_hom().map(*k)).collect::<Vec<_>>();
        let mut actual = to.iter().map(|R| R.int_hom().map(*k)).collect::<Vec<_>>();

        table.apply(Submatrix::<AsFirstElement<_>, _>::new(&x, 3, 1), SubmatrixMut::<AsFirstElement<_>, _>::new(&mut actual, 3, 1));
        
        for i in 0..y.len() {
            assert!(to[i].eq_el(&y[i], actual.at(i)));
        }
    }
}
