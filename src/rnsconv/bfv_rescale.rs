use feanor_math::matrix::submatrix::*;
use feanor_math::mempool::*;
use feanor_math::rings::zn::*;
use feanor_math::homomorphism::*;
use feanor_math::integer::*;
use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::ring::*;
use feanor_math::vector::*;

use crate::rnsconv::approx_lift::AlmostExactBaseConversion;

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
pub struct AlmostExactRescalingConvert<R, M_Zn, M_Int>
    where R: ZnRingStore + Clone,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + SelfIso,
        M_Zn: MemoryProvider<El<R>>,
        M_Int: MemoryProvider<El<<R::Type as ZnRing>::Integers>>
{
    // rescale `Z/qZ -> Z/(aq/b)Z`
    rescaling: AlmostExactRescaling<R, M_Zn, M_Int>,
    // convert `Z/(aq/b)Z -> Z/bZ`
    convert: AlmostExactBaseConversion<R, R, M_Int, M_Zn>
}

impl<'a, R, M_Zn, M_Int> AlmostExactRescalingConvert<R, M_Zn, M_Int>
    where R: ZnRingStore + Clone,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + SelfIso,
        M_Zn: MemoryProvider<El<R>> + Clone,
        M_Int: MemoryProvider<El<<R::Type as ZnRing>::Integers>> + Clone
{
    pub fn new(in_moduli: Vec<R>, numerator_moduli: Vec<R>, denominator_count: usize, memory_provider: M_Zn, memory_provider_int: M_Int) -> Self {
        let rescaling = AlmostExactRescaling::new(in_moduli.clone(), numerator_moduli, denominator_count, memory_provider.clone(), memory_provider_int.clone());
        let convert = AlmostExactBaseConversion::new(
            rescaling.output_rings(),
            &in_moduli[..denominator_count],
            memory_provider_int,
            memory_provider
        );
        return Self { rescaling, convert };
    }
}

impl<R, M_Zn, M_Int> RNSOperation for AlmostExactRescalingConvert<R, M_Zn, M_Int>
    where R: ZnRingStore + Clone,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + SelfIso,
        M_Zn: MemoryProvider<El<R>>,
        M_Int: MemoryProvider<<<<R as RingStore>::Type as ZnRing>::IntegerRingBase as RingBase>::Element>
{
    type Ring = R;

    type RingType = R::Type;

    fn input_rings<'a>(&'a self) -> &'a [R] {
        self.rescaling.input_rings()
    }

    fn output_rings<'a>(&'a self) -> &'a [R] {
        self.convert.output_rings()
    }

    fn apply<V1, V2>(&self, input: Submatrix<V1, El<Self::Ring>>, output: SubmatrixMut<V2, El<Self::Ring>>)
            where V1: AsPointerToSlice<El<Self::Ring>>,
                V2: AsPointerToSlice<El<Self::Ring>>
    {
        assert_eq!(input.col_count(), output.col_count());
        let mut tmp = self.rescaling.memory_provider.get_new_init(self.rescaling.output_rings().len() * input.col_count(), |idx| self.rescaling.output_rings().at(idx  / input.col_count()).zero());
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
/// let rescaling = AlmostExactRescaling::new(from.clone(), to.clone(), 3, default_memory_provider!(), default_memory_provider!());
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
/// let x = 1153;
/// rescaling.apply(Submatrix::<AsFirstElement<_>, _>::new(&[from[0].int_hom().map(x), from[1].int_hom().map(x), from[2].int_hom().map(x)], 3, 1), SubmatrixMut::<AsFirstElement<_>, _>::new(&mut output, 1, 1));
/// assert_el_eq!(
///     &to[0],
///     &to[0].int_hom().map(/* rounded division */ (x * 29 + from_modulus / 2) / from_modulus - 1),
///     &output[0]);
/// ```
/// 
pub struct AlmostExactRescaling<R, M_Zn, M_Int>
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + SelfIso,
        M_Zn: MemoryProvider<El<R>>,
        M_Int: MemoryProvider<El<<R::Type as ZnRing>::Integers>>
{
    a_moduli_count: usize,
    // contains the moduli of `q` and then the moduli of `a`
    b_to_aq_over_b_lift: AlmostExactBaseConversion<R, R, M_Int, M_Zn>,
    // a as element of each modulus of `q`
    a: Vec<El<R>>,
    // 1/b as element of each modulus of `aq/b`
    b_inv: Vec<El<R>>,
    q_moduli: Vec<R>,
    memory_provider: M_Zn
}

impl<'a, R, M_Zn, M_Int> AlmostExactRescaling<R, M_Zn, M_Int>
    where R: ZnRingStore + Clone,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + SelfIso,
        M_Zn: MemoryProvider<El<R>> + Clone,
        M_Int: MemoryProvider<El<<R::Type as ZnRing>::Integers>>
{
    pub fn new(in_moduli: Vec<R>, numerator_moduli: Vec<R>, denominator_count: usize, memory_provider: M_Zn, memory_provider_int: M_Int) -> Self {
        let numerator_len = numerator_moduli.len();
        
        let a = ZZbig.prod(numerator_moduli.iter().map(|R| int_cast(R.integer_ring().clone_el(R.modulus()), &ZZbig, R.integer_ring())));
        let b = ZZbig.prod(in_moduli.iter().take(denominator_count).map(|R| int_cast(R.integer_ring().clone_el(R.modulus()), &ZZbig, R.integer_ring())));
        let a_mod = in_moduli.iter().map(|R| R.coerce(&ZZbig, ZZbig.clone_el(&a))).collect::<Vec<_>>();

        let mut in_moduli_iter = in_moduli.iter().cloned();
        let b_moduli = in_moduli_iter.by_ref().take(denominator_count).collect::<Vec<_>>();
        let aq_over_b_moduli = in_moduli_iter.chain(numerator_moduli.into_iter()).collect::<Vec<_>>();

        let b_to_aq_over_b_lift = AlmostExactBaseConversion::new(
            &b_moduli,
            &aq_over_b_moduli,
            memory_provider_int,
            memory_provider.clone()
        );
        let inv_b = b_to_aq_over_b_lift.output_rings()
            .iter()
            .map(|R| R.invert(&R.coerce(&ZZbig, ZZbig.clone_el(&b))).unwrap())
            .collect();

        AlmostExactRescaling {
            a_moduli_count: numerator_len,  
            b_inv: inv_b, 
            a: a_mod,
            b_to_aq_over_b_lift: b_to_aq_over_b_lift,
            q_moduli: in_moduli, 
            memory_provider: memory_provider
        }
    }
}

impl<R, M_Zn, M_Int> AlmostExactRescaling<R, M_Zn, M_Int>
    where R: ZnRingStore + Clone,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + SelfIso,
        M_Zn: MemoryProvider<El<R>>,
        M_Int: MemoryProvider<El<<R::Type as ZnRing>::Integers>>
{
    fn a_moduli<'a>(&'a self) -> &'a [R] {
        &self.b_to_aq_over_b_lift.output_rings()[(self.b_to_aq_over_b_lift.output_rings().len() - self.a_moduli_count)..]
    }

    fn b_moduli<'a>(&'a self) -> &'a [R] {
        self.b_to_aq_over_b_lift.input_rings()
    }

    fn q_over_b_moduli<'a>(&'a self) -> &'a [R] {
        &self.b_to_aq_over_b_lift.output_rings()[..(self.b_to_aq_over_b_lift.output_rings().len() - self.a_moduli_count)]
    }

    fn q_moduli<'a>(&'a self) -> &'a [R] {
        &self.q_moduli
    }

    fn aq_over_b_moduli<'a>(&'a self) -> &'a [R] {
        self.b_to_aq_over_b_lift.output_rings()
    }
}

impl<R, M_Zn, M_Int> RNSOperation for AlmostExactRescaling<R, M_Zn, M_Int>
    where R: ZnRingStore + Clone,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + SelfIso,
        M_Zn: MemoryProvider<El<R>>,
        M_Int: MemoryProvider<<<<R as RingStore>::Type as ZnRing>::IntegerRingBase as RingBase>::Element>
{
    type Ring = R;

    type RingType = R::Type;

    fn input_rings<'a>(&'a self) -> &'a [R] {
        self.q_moduli()
    }

    fn output_rings<'a>(&'a self) -> &'a [R] {
        self.aq_over_b_moduli()
    }

    fn apply<V1, V2>(&self, input: Submatrix<V1, El<Self::Ring>>, mut output: SubmatrixMut<V2, El<Self::Ring>>)
        where V1: AsPointerToSlice<El<Self::Ring>>,
            V2: AsPointerToSlice<El<Self::Ring>>
    {
        assert_eq!(input.row_count(), self.input_rings().len());
        assert_eq!(output.row_count(), self.output_rings().len());
        assert_eq!(input.col_count(), output.col_count());

        // Compute `x := el * a mod aq`, store it in `x_mod_b` and `x_mod_aq_over_b`
        let mut x_mod_b = self.memory_provider.get_new_init(self.b_moduli().len() * input.col_count(), |idx| {
            let i = idx / input.col_count();
            let j = idx % input.col_count();
            self.b_moduli().at(i).mul_ref(input.at(i, j), self.a.at(i))
        });

        let mut x_mod_aq_over_b = output.reborrow();
        for (i, Zk) in self.q_over_b_moduli().iter().enumerate() {
            for j in 0..input.col_count() {
                *x_mod_aq_over_b.at(i, j) = Zk.mul_ref(input.at(i + self.b_moduli().len(), j), self.a.at(i + self.b_moduli().len()));
            }
        }
        for (i, Zk) in self.a_moduli().iter().enumerate() {
            for j in 0..input.col_count() {
                *x_mod_aq_over_b.at(i + self.q_over_b_moduli().len(), j) = Zk.zero();
            }
        }

        let x_mod_b = SubmatrixMut::<AsFirstElement<_>, _>::new(&mut x_mod_b, self.b_moduli().len(), input.col_count());

        // Compute the shortest lift of `x mod b` to `aq/b`; Here we might introduce an error of `+/- b`
        // that will later be rescaled to `+/- 1`.
        let mut x_mod_b_lift = self.memory_provider.get_new_init(self.aq_over_b_moduli().len() * input.col_count(), |idx| {
            let i = idx / input.col_count();
            self.aq_over_b_moduli().at(i).zero()
        });
        let mut x_mod_b_lift = SubmatrixMut::<AsFirstElement<_>, _>::new(&mut x_mod_b_lift, self.aq_over_b_moduli().len(), input.col_count());
        self.b_to_aq_over_b_lift.apply(x_mod_b.as_const(), x_mod_b_lift.reborrow());
        debug_assert!(x_mod_b_lift.row_count() == self.aq_over_b_moduli().len());

        for (i, Zk) in self.aq_over_b_moduli().iter().enumerate() {
            for j in 0..input.col_count() {
                // Subtract `lift(x mod b) mod aq/b` from `x_mod_aq_over_b`
                Zk.sub_assign_ref(x_mod_aq_over_b.at(i, j), x_mod_b_lift.at(i, j));
                // Now `x_mod_aq_over_b - lift(x mod b)` is divisibible by b
                Zk.mul_assign_ref(x_mod_aq_over_b.at(i, j), self.b_inv.at(i));
            }
        }
    }

}

#[cfg(test)]
use feanor_math::rings::zn::zn_64::*;
#[cfg(test)]
use feanor_math::{assert_el_eq, default_memory_provider};

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
        default_memory_provider!(), 
        default_memory_provider!()
    );

    // since Zm_intermediate has a very large modulus, we can ignore errors here at the moment (I think)
    for i in -(q/2)..=(q/2) {
        let input = from.iter().map(|R| R.int_hom().map(i)).collect::<Vec<_>>();
        let output = to.iter().map(|R| R.int_hom().map((i as f64 * 257. / 17. / 97.).round() as i32)).collect::<Vec<_>>();
        let mut actual = to.iter().map(|R| R.zero()).collect::<Vec<_>>();

        rescaling.apply(Submatrix::<AsFirstElement<_>, _>::new(&input, 3, 1), SubmatrixMut::<AsFirstElement<_>, _>::new(&mut actual, 2, 1));

        for j in 0..output.len() {
            assert_el_eq!(to.at(j), output.at(j), actual.at(j));
        }        
    }
}


#[test]
fn test_rescale_small_intermediate() {
    let from = vec![Zn::new(17), Zn::new(19), Zn::new(23)];
    let num = vec![Zn::new(29)];
    let q = 17 * 19 * 23;

    let rescaling = AlmostExactRescaling::new(
        from.clone(), 
        num.clone(), 
        3,
        default_memory_provider!(), 
        default_memory_provider!()
    );

    // since Zm_intermediate has a very large modulus, we can ignore errors here at the moment (I think)
    for i in -(q/2)..=(q/2) {
        let input = from.iter().map(|R| R.int_hom().map(i)).collect::<Vec<_>>();
        let output = num.iter().map(|R| R.int_hom().map((i as f64 * 29. / q as f64).round() as i32)).collect::<Vec<_>>();
        let mut actual = num.iter().map(|R| R.zero()).collect::<Vec<_>>();

        rescaling.apply(Submatrix::<AsFirstElement<_>, _>::new(&input, 3, 1), SubmatrixMut::<AsFirstElement<_>, _>::new(&mut actual, 1, 1));

        let Zk = num.at(0);
        assert!(Zk.eq_el(output.at(0), actual.at(0)) ||
            Zk.eq_el(output.at(0), &Zk.add_ref_fst(actual.at(0), Zk.one())) ||
            Zk.eq_el(output.at(0), &Zk.sub_ref_fst(actual.at(0), Zk.one()))
        );
    }
}