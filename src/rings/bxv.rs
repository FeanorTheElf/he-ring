use std::alloc::Allocator;
use std::ops::Range;

use feanor_math::homomorphism::CanHomFrom;
use feanor_math::integer::{BigIntRing, BigIntRingBase};
use feanor_math::ring::*;
use feanor_math::rings::extension::FreeAlgebra;
use feanor_math::rings::finite::FiniteRing;
use feanor_math::rings::zn::{zn_64, zn_rns, ZnRing};

use crate::rnsconv::RNSOperation;

use super::decomposition_ring::{DecompositionRing, DecompositionRingBase};
use super::number_ring::{HECyclotomicNumberRing, HENumberRing};

///
/// A ring that supports all the operations required during BFV/BGV-style homomorphic
/// encryption.
/// 
pub trait BXVCiphertextRing: FreeAlgebra<BaseRing = zn_rns::Zn<zn_64::Zn, BigIntRing>> + FiniteRing {

    type NumberRing: HECyclotomicNumberRing<zn_64::Zn>;
    type GadgetProductLhsOperand<'a>
        where Self: 'a;
    type GadgetProductRhsOperand<'a>
        where Self: 'a;
    
    fn number_ring(&self) -> &Self::NumberRing;

    fn sample_from_coefficient_distribution<G: FnMut() -> i32>(&self, distribution: G) -> Self::Element;

    fn perform_rns_op_from<Op>(
        &self, 
        from: &Self, 
        el: &Self::Element, 
        op: &Op
    ) -> Self::Element
        where Op: RNSOperation<RingType = zn_64::ZnBase>;
    
    
    fn exact_convert_from_decompring<FpTy2, A2>(
        &self, 
        from: &DecompositionRing<Self::NumberRing, FpTy2, A2>, 
        element: &<DecompositionRingBase<Self::NumberRing, FpTy2, A2> as RingBase>::Element
    ) -> Self::Element
        where Self::NumberRing: HENumberRing<FpTy2>,
            FpTy2: RingStore<Type = zn_64::ZnBase> + Clone,
            A2: Allocator + Clone;
    
    fn perform_rns_op_to_decompring<FpTy2, A2, Op>(
        &self, 
        to: &DecompositionRing<Self::NumberRing, FpTy2, A2>, 
        element: &Self::Element, 
        op: &Op
    ) -> <DecompositionRingBase<Self::NumberRing, FpTy2, A2> as RingBase>::Element 
        where Self::NumberRing: HENumberRing<FpTy2>,
            FpTy2: RingStore<Type = zn_64::ZnBase> + Clone,
            A2: Allocator + Clone,
            Op: RNSOperation<RingType = zn_64::ZnBase>;

    ///
    /// Computes the data necessary to perform a "gadget product" with the given operand as
    /// left-hand side. This can be though of computing the gadget decomposition of the argument.
    /// For more details, see [`BXVCiphertextRing::gadget_product()`].
    /// 
    fn to_gadget_product_lhs<'a>(&'a self, el: Self::Element, digits: usize) -> Self::GadgetProductLhsOperand<'a>;

    ///
    /// Creates a right-hand side gadget product operand fo 0. Its data (i.e. the noisy approximations
    /// of scalings of the base ring element) can be set later with [`BXVCiphertextRing::set_rns_factor()`].
    /// For more details, see [`BXVCiphertextRing::gadget_product()`].
    /// 
    fn gadget_product_rhs_empty<'a>(&'a self, digits: usize) -> Self::GadgetProductRhsOperand<'a>;

    ///
    /// Returns the gadget vector associated to the given right-hand side gadget product operand.
    /// The gadget vector is returned as a vector of ranges that should be considered as indices
    /// into the RNS base. In other words, the `i`-th entry of the gadget vector is the element
    /// `g[i] in Z/qZ` that satisfies
    /// ```text
    ///   g[i] = 1 mod p_j    if j in begin..end
    ///        = 0 mod p_j    otherwise
    /// ```
    /// where `begin..end` is the `i`-th entry of the returned slice, and `p1, ..., pr` is the RNS
    /// base as given by `self.base_ring()`.
    /// 
    /// For a more general explanation of gadget products, see [`BXVCiphertextRing::gadget_product()`].
    /// 
    /// While this is quite an implicit way of giving the gadget vector, it has the great advantage
    /// that it is simple to use for values in RNS representation. In particular, if we want to compute
    /// `x * g[i]` (which is what we usually do when filling the right-hand side operand with values),
    /// we find that `x * g[i]` is exactly the value whose RNS components with index in `begin..end` have
    /// the value `x`, and all the others are zero.
    /// 
    fn gadget_vector<'a, 'b>(&'a self, rhs_operand: &'a Self::GadgetProductRhsOperand<'b>) -> &'a [Range<usize>];

    fn set_rns_factor<'b>(&self, rhs_operand: &mut Self::GadgetProductRhsOperand<'b>, i: usize, el: Self::Element);

    ///
    /// Computes the "RNS-gadget product" of two elements in this ring, as often required
    /// in HE scenarios. A "gadget product" computes the approximate product of two
    /// ring elements `x` and `y` by using `y` and multiple scaled & noisy approximations 
    /// to `x`. This function only supports the gadget vector given by a decomposition
    /// `q = D1 ... Dr` into coprime "digits".
    /// 
    /// # What exactly is a "gadget product"?
    /// 
    /// In an HE setting, we often have a noisy approximation to some value `x`, say
    /// `x + e`. Now the normal product `(x + e)y = xy + ey` includes an error of `ey`, which
    /// (if `y` is arbitrary) is not in any way an approximation to `xy` anymore. Instead,
    /// we can take a so-called "gadget vector" `g` and provide multiple noisy scalings of `x`, 
    /// say `g[1] * x + e1` to `g[r] * x + er`.
    /// Using these, we can approximate `xy` by computing a gadget-decomposition 
    /// `y = g[1] * y1 + ... + g[m] * ym` of `y`, where the values `yi` are small, and then use 
    /// `y1 (g[1] * x + e1) + ... + ym (g[m] * x + em)` as an approximation to `xy`.
    /// 
    /// The gadget vector used for this "RNS-gadget product" is the one given by the unit vectors
    /// in the decomposition `q = D1 ... Dr` into pairwise coprime "digits". In the simplest case,
    /// those digits are just the prime factors of `q`. However, it is usually beneficial to
    /// group multiple prime factors into a single digit, since decreasing the number of digits
    /// will significantly decrease the work we have to do when computing the inner product
    /// `sum_i (g[i] xi + ei) yi`. Note that this will of course decrease the quality of 
    /// approximation to `xy` (i.e. increase the error `sum_i yi ei`). Hence, choose the
    /// parameter `digits` appropriately.
    /// 
    /// # Example
    /// ```
    /// # use feanor_math::ring::*;
    /// # use feanor_math::assert_el_eq;
    /// # use feanor_math::homomorphism::*;
    /// # use feanor_math::rings::zn::*;
    /// # use feanor_math::rings::zn::zn_64::*;
    /// # use feanor_math::rings::finite::*;
    /// # use feanor_math::integer::BigIntRing;
    /// # use feanor_math::algorithms::fft::cooley_tuckey::CooleyTuckeyFFT;
    /// # use he_ring::rings::double_rns_managed::*;
    /// # use he_ring::rings::bxv::*;
    /// # use he_ring::rings::pow2_cyclotomic::Pow2CyclotomicNumberRing;
    /// # use feanor_math::rings::extension::FreeAlgebraStore;
    /// # use feanor_math::seq::*;
    /// let rns_base = vec![Zn::new(17), Zn::new(97), Zn::new(113)];
    /// let ring = ManagedDoubleRNSRingBase::new(Pow2CyclotomicNumberRing::new(16), zn_rns::Zn::new(rns_base.clone(), BigIntRing::RING));
    /// let mut rng = oorandom::Rand64::new(1);
    /// // we have digits == rns_base.len(), so the gadget vector has entries exactly the "CRT unit vectors" ei with ei = 1 mod pi, ei = 0 mod pj for j != i
    /// let digits = 3;
    /// 
    /// // build the right-hand side operand
    /// let rhs = ring.random_element(|| rng.rand_u64());
    /// let mut rhs_op = ring.get_ring().gadget_product_rhs_empty(digits);
    /// let gadget_vector = |i: usize| ring.base_ring().get_ring().from_congruence((0..3).map(|j| ring.base_ring().get_ring().at(j).int_hom().map(if i == j { 1 } else { 0 })));
    /// for i in 0..3 {
    ///     // set the i-th component to `gadget_vector(i) * rhs`, for now without noise
    ///     ring.get_ring().set_rns_factor(&mut rhs_op, i, ring.inclusion().mul_ref_map(&rhs, &gadget_vector(i)));
    /// }
    /// 
    /// // compute the gadget product
    /// let lhs = ring.random_element(|| rng.rand_u64());
    /// let actual = ring.get_ring().gadget_product(&ring.get_ring().to_gadget_product_lhs(ring.clone_el(&lhs), digits), &rhs_op);
    /// assert_el_eq!(&ring, &ring.mul_ref(&lhs, &rhs), actual);
    /// ```
    /// To demonstrate how this keeps small error terms small, consider the following variation of the previous example:
    /// ```
    /// # use feanor_math::ring::*;
    /// # use feanor_math::assert_el_eq;
    /// # use feanor_math::homomorphism::*;
    /// # use feanor_math::rings::zn::*;
    /// # use feanor_math::rings::zn::zn_64::*;
    /// # use feanor_math::algorithms::fft::cooley_tuckey::CooleyTuckeyFFT;
    /// # use he_ring::rings::double_rns_managed::*;
    /// # use he_ring::rings::bxv::*;
    /// # use he_ring::rings::pow2_cyclotomic::Pow2CyclotomicNumberRing;
    /// # use feanor_math::rings::extension::FreeAlgebraStore;
    /// # use feanor_math::integer::BigIntRing;
    /// # use feanor_math::rings::finite::*;
    /// # use feanor_math::integer::int_cast;
    /// # use feanor_math::primitive_int::StaticRing;
    /// # use feanor_math::seq::*;
    /// // create the ring as before
    /// # let rns_base = vec![Zn::new(17), Zn::new(97), Zn::new(113)];
    /// # let ring = ManagedDoubleRNSRingBase::new(Pow2CyclotomicNumberRing::new(16), zn_rns::Zn::new(rns_base.clone(), BigIntRing::RING));
    /// # let mut rng = oorandom::Rand64::new(1);
    /// # let digits = 3;
    /// let rhs = ring.random_element(|| rng.rand_u64());
    /// let mut rhs_op = ring.get_ring().gadget_product_rhs_empty(digits);
    /// // this time include some error when building `rhs_op`
    /// let mut create_small_error = || ring.get_ring().sample_from_coefficient_distribution(|| (rng.rand_u64() % 3) as i32 - 1);
    /// let gadget_vector = |i: usize| ring.base_ring().get_ring().from_congruence((0..3).map(|j| ring.base_ring().get_ring().at(j).int_hom().map(if i == j { 1 } else { 0 })));
    /// for i in 0..3 {
    ///     // set the i-th component to `gadget_vector(i) * rhs`, with possibly some error included
    ///     ring.get_ring().set_rns_factor(&mut rhs_op, i, ring.add(ring.inclusion().mul_ref_map(&rhs, &gadget_vector(i)), create_small_error()));
    /// }
    /// 
    /// // compute the gadget product
    /// let lhs = ring.random_element(|| rng.rand_u64());
    /// let actual = ring.get_ring().gadget_product(&ring.get_ring().to_gadget_product_lhs(ring.clone_el(&lhs), digits), &rhs_op);
    /// 
    /// // the final result should be close to `lhs * rhs`, except for some noise
    /// let expected = ring.mul_ref(&lhs, &rhs);
    /// let error = ring.sub(expected, actual);
    /// let error_coefficients = ring.wrt_canonical_basis(&error);
    /// let max_allowed_error = (113 / 2) * 8 * 3;
    /// assert!((0..8).all(|i| int_cast(ring.base_ring().smallest_lift(error_coefficients.at(i)), StaticRing::<i64>::RING, BigIntRing::RING).abs() <= max_allowed_error));
    /// ```
    /// 
    fn gadget_product<'a, 'b>(&self, lhs: &Self::GadgetProductLhsOperand<'a>, rhs: &Self::GadgetProductRhsOperand<'b>) -> Self::Element;

    ///
    /// Computes the image of the ring element stored by the given gadget product operand under the Galois action of all given Galois elements,
    /// and create new gadget product operands for those new ring elements.
    /// 
    /// This allows only explicitly creating a gadget product operand once, i.e.
    /// ```
    /// # use feanor_math::ring::*;
    /// # use feanor_math::assert_el_eq;
    /// # use feanor_math::homomorphism::*;
    /// # use feanor_math::rings::zn::*;
    /// # use feanor_math::rings::zn::zn_64::*;
    /// # use feanor_math::algorithms::fft::cooley_tuckey::CooleyTuckeyFFT;
    /// # use he_ring::rings::double_rns_managed::*;
    /// # use he_ring::rings::bxv::*;
    /// # use he_ring::cyclotomic::*;
    /// # use he_ring::rings::pow2_cyclotomic::Pow2CyclotomicNumberRing;
    /// # use feanor_math::rings::extension::FreeAlgebraStore;
    /// # use feanor_math::integer::BigIntRing;
    /// # use feanor_math::rings::finite::*;
    /// # use feanor_math::integer::int_cast;
    /// # use feanor_math::primitive_int::StaticRing;
    /// # use feanor_math::seq::*;
    /// // we are given `ring, x, gs, digits`
    /// # let rns_base = vec![Zn::new(17), Zn::new(97), Zn::new(113)];
    /// # let ring = ManagedDoubleRNSRingBase::new(Pow2CyclotomicNumberRing::new(16), zn_rns::Zn::new(rns_base.clone(), BigIntRing::RING));
    /// # let digits = 3;
    /// # let x = ring.canonical_gen();
    /// # let gs = vec![ring.cyclotomic_index_ring().one(), ring.cyclotomic_index_ring().int_hom().map(-1)];
    /// # let gs = &gs;
    /// let x_op = ring.get_ring().to_gadget_product_lhs(x, digits);
    /// let galois_x_op = ring.get_ring().apply_galois_action_many_gadget_product_operand(&x_op, gs);
    /// ```
    /// which might be faster than the following, functionally equivalent code
    /// ```
    /// # use feanor_math::ring::*;
    /// # use feanor_math::assert_el_eq;
    /// # use feanor_math::homomorphism::*;
    /// # use feanor_math::rings::zn::*;
    /// # use feanor_math::rings::zn::zn_64::*;
    /// # use feanor_math::algorithms::fft::cooley_tuckey::CooleyTuckeyFFT;
    /// # use he_ring::rings::double_rns_managed::*;
    /// # use he_ring::rings::bxv::*;
    /// # use he_ring::cyclotomic::*;
    /// # use he_ring::rings::pow2_cyclotomic::Pow2CyclotomicNumberRing;
    /// # use feanor_math::rings::extension::FreeAlgebraStore;
    /// # use feanor_math::integer::BigIntRing;
    /// # use feanor_math::rings::finite::*;
    /// # use feanor_math::integer::int_cast;
    /// # use feanor_math::primitive_int::StaticRing;
    /// # use feanor_math::seq::*;
    /// # let rns_base = vec![Zn::new(17), Zn::new(97), Zn::new(113)];
    /// # let ring = ManagedDoubleRNSRingBase::new(Pow2CyclotomicNumberRing::new(16), zn_rns::Zn::new(rns_base.clone(), BigIntRing::RING));
    /// # let digits = 3;
    /// # let x = ring.canonical_gen();
    /// # let gs = vec![ring.cyclotomic_index_ring().one(), ring.cyclotomic_index_ring().int_hom().map(-1)];
    /// # let gs = &gs;
    /// let galois_x = ring.get_ring().apply_galois_action_many(&x, gs);
    /// let galois_x_op = galois_x.map(|y| ring.get_ring().to_gadget_product_lhs(y, digits)).collect::<Vec<_>>();
    /// 
    /// // using `galois_x_op` from before, we see that the result is the same
    /// # let x_op_prev = ring.get_ring().to_gadget_product_lhs(x, digits);
    /// # let galois_x_op_prev = ring.get_ring().apply_galois_action_many_gadget_product_operand(&x_op_prev, gs);
    /// # let mut some_rhs_operand = ring.get_ring().gadget_product_rhs_empty(digits);
    /// # ring.get_ring().set_rns_factor(&mut some_rhs_operand, 0, ring.inclusion().map(ring.base_ring().from_congruence([1000, 0, 0].into_iter().enumerate().map(|(i, c)| ring.base_ring().at(i).int_hom().map(c)))));
    /// # ring.get_ring().set_rns_factor(&mut some_rhs_operand, 1, ring.inclusion().map(ring.base_ring().from_congruence([0, 1000, 0].into_iter().enumerate().map(|(i, c)| ring.base_ring().at(i).int_hom().map(c)))));
    /// # ring.get_ring().set_rns_factor(&mut some_rhs_operand, 2, ring.inclusion().map(ring.base_ring().from_congruence([0, 0, 1000].into_iter().enumerate().map(|(i, c)| ring.base_ring().at(i).int_hom().map(c)))));
    /// assert_el_eq!(&ring, ring.get_ring().gadget_product(&galois_x_op[0], &some_rhs_operand), ring.get_ring().gadget_product(&galois_x_op_prev[0], &some_rhs_operand));
    /// ```
    /// 
    fn apply_galois_action_many_gadget_product_operand<'a>(&'a self, x: &Self::GadgetProductLhsOperand<'a>, gs: &[zn_64::ZnEl]) -> Vec<Self::GadgetProductLhsOperand<'a>>;

    ///
    /// Computes `[lhs[0] * rhs[0], lhs[0] * rhs[1] + lhs[1] * rhs[0], lhs[1] * rhs[1]]`, but might be
    /// faster than the naive way of evaluating this.
    /// 
    fn two_by_two_convolution(&self, lhs: [&Self::Element; 2], rhs: [&Self::Element; 2]) -> [Self::Element; 3] {
        [
            self.mul_ref(lhs[0], rhs[0]),
            self.add(self.mul_ref(lhs[0], rhs[1]), self.mul_ref(lhs[1], rhs[0])),
            self.mul_ref(lhs[1], rhs[1])
        ]
    }
}