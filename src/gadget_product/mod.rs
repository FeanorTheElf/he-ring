use std::alloc::Global;
use std::ops::Range;

use feanor_math::integer::BigIntRing;
use feanor_math::matrix::{OwnedMatrix, Submatrix, SubmatrixMut};
use feanor_math::primitive_int::StaticRing;
use feanor_math::rings::zn::ZnRingStore;
use feanor_math::{assert_el_eq, ring::*};
use feanor_math::rings::zn::{zn_64::Zn, zn_rns};
use feanor_math::seq::subvector::SubvectorView;
use feanor_math::seq::{VectorFn, VectorView};
use feanor_math::homomorphism::Homomorphism;

use crate::ciphertext_ring::double_rns_ring::DoubleRNSRingBase;
use crate::ciphertext_ring::single_rns_ring::SingleRNSRingBase;
use crate::ciphertext_ring::BGFVCiphertextRing;
use crate::cyclotomic::{CyclotomicGaloisGroupEl, CyclotomicRing};
use crate::number_ring::pow2_cyclotomic::Pow2CyclotomicNumberRing;
use crate::rnsconv::bfv_rescale::AlmostExactRescaling;
use crate::rnsconv::{lift, RNSOperation};
use crate::DefaultConvolution;

type UsedBaseConversion<A> = lift::AlmostExactBaseConversion<A>;

pub struct GadgetProductLhsOperand<R: BGFVCiphertextRing> {
    /// `i`-th entry stores a `i`-th part of the gadget decomposition of the represented element.
    /// We store the element once as `PreparedMultiplicant` for fast computation of gadget products, and 
    /// once as the element itself, since there currently is no way of getting the ring element out of
    /// a `PreparedMultiplicant`
    element_decomposition: Vec<(R::PreparedMultiplicant, R::Element)>
}

impl<R: BGFVCiphertextRing> GadgetProductLhsOperand<R> {

    ///
    /// Creates a [`GadgetProductLhsOperand`] w.r.t. the gadget vector given by `digits`.
    /// For an explanation of gadget products, see [`GadgetProductLhsOperand::gadget_product()`].
    /// 
    pub fn from_element_with<V>(ring: &R, el: &R::Element, digits: V) -> Self
        where V: VectorFn<Range<usize>>
    {
        let decomposition = gadget_decompose(ring, el, digits);
        return Self {
            element_decomposition: decomposition
        };
    }

    /// 
    /// Creates a [`GadgetProductLhsOperand`] w.r.t. the RNS gadget vector that has `digits` digits.
    /// For an explanation of gadget products, see [`GadgetProductLhsOperand::gadget_product()`].
    /// 
    pub fn from_element(ring: &R, el: &R::Element, digits: usize) -> Self {
        Self::from_element_with(ring, el, select_digits(digits, ring.base_ring().len()).clone_els())
    }

    pub fn apply_galois_action(&self, ring: &R, g: CyclotomicGaloisGroupEl) -> Self 
        where R: CyclotomicRing
    {
        Self {
            element_decomposition: self.element_decomposition.iter().map(|(_prepared_el, el)| {
                let new_el = ring.apply_galois_action(el, g);
                return (ring.prepare_multiplicant(&new_el), new_el);
            }).collect()
        }
    }

    pub fn apply_galois_action_many(&self, ring: &R, gs: &[CyclotomicGaloisGroupEl]) -> Vec<Self>
        where R: CyclotomicRing
    {
        let mut result = Vec::with_capacity(gs.len());
        result.resize_with(gs.len(), || GadgetProductLhsOperand { element_decomposition: Vec::new() });
        for (_prepared_el, el) in &self.element_decomposition {
            let new_els = ring.apply_galois_action_many(el, gs);
            for (i, new_el) in new_els.into_iter().enumerate() {
                result[i].element_decomposition.push((ring.prepare_multiplicant(&new_el), new_el));
            }
        }
        return result;
    }

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
    /// parameter `digits` appropriately. The gadget vector used in a specific case can be
    /// queried using [`GadgetProductRhsOperand::gadget_vector()`]. 
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
    /// # use feanor_math::rings::extension::FreeAlgebraStore;
    /// # use feanor_math::seq::*;
    /// # use he_ring::ciphertext_ring::double_rns_managed::*;
    /// # use he_ring::number_ring::pow2_cyclotomic::Pow2CyclotomicNumberRing;
    /// # use he_ring::gadget_product::*;
    /// let rns_base = vec![Zn::new(17), Zn::new(97), Zn::new(113)];
    /// let ring = ManagedDoubleRNSRingBase::new(Pow2CyclotomicNumberRing::new(16), zn_rns::Zn::new(rns_base.clone(), BigIntRing::RING));
    /// let mut rng = oorandom::Rand64::new(1);
    /// // we have digits == rns_base.len(), so the gadget vector has entries exactly the "CRT unit vectors" ei with ei = 1 mod pi, ei = 0 mod pj for j != i
    /// let digits = 3;
    /// 
    /// // build the right-hand side operand
    /// let rhs = ring.random_element(|| rng.rand_u64());
    /// let mut rhs_op = GadgetProductRhsOperand::new(ring.get_ring(), digits);
    /// for i in 0..3 {
    ///     // set the i-th component to `gadget_vector(i) * rhs`, for now without noise
    ///     let component_at_i = ring.inclusion().mul_ref_map(&rhs, &rhs_op.gadget_vector(ring.get_ring()).at(i));
    ///     rhs_op.set_rns_factor(ring.get_ring(), i, component_at_i);
    /// }
    /// 
    /// // compute the gadget product
    /// let lhs = ring.random_element(|| rng.rand_u64());
    /// let lhs_op = GadgetProductLhsOperand::from_element(ring.get_ring(), &lhs, digits);
    /// let actual = lhs_op.gadget_product(&rhs_op, ring.get_ring());
    /// assert_el_eq!(&ring, &ring.mul_ref(&lhs, &rhs), actual);
    /// ```
    /// To demonstrate how this keeps small error terms small, consider the following variation of the previous example:
    /// ```
    /// # use feanor_math::ring::*;
    /// # use feanor_math::assert_el_eq;
    /// # use feanor_math::homomorphism::*;
    /// # use feanor_math::rings::zn::*;
    /// # use feanor_math::rings::zn::zn_64::*;
    /// # use feanor_math::rings::finite::*;
    /// # use feanor_math::rings::extension::FreeAlgebra;
    /// # use feanor_math::integer::*;
    /// # use feanor_math::primitive_int::StaticRing;
    /// # use feanor_math::algorithms::fft::cooley_tuckey::CooleyTuckeyFFT;
    /// # use feanor_math::rings::extension::FreeAlgebraStore;
    /// # use feanor_math::seq::*;
    /// # use he_ring::ciphertext_ring::double_rns_managed::*;
    /// # use he_ring::number_ring::pow2_cyclotomic::Pow2CyclotomicNumberRing;
    /// # use he_ring::gadget_product::*;
    /// # let rns_base = vec![Zn::new(17), Zn::new(97), Zn::new(113)];
    /// # let ring = ManagedDoubleRNSRingBase::new(Pow2CyclotomicNumberRing::new(16), zn_rns::Zn::new(rns_base.clone(), BigIntRing::RING));
    /// # let mut rng = oorandom::Rand64::new(1);
    /// # let digits = 3;
    /// // build the ring just as before
    /// let rhs = ring.random_element(|| rng.rand_u64());
    /// let mut rhs_op = GadgetProductRhsOperand::new(ring.get_ring(), digits);
    /// // this time include some error when building `rhs_op`
    /// let mut create_small_error = || ring.get_ring().from_canonical_basis((0..ring.rank()).map(|i| ring.base_ring().int_hom().map((rng.rand_u64() % 3) as i32 - 1)));
    /// for i in 0..3 {
    ///     // set the i-th component to `gadget_vector(i) * rhs`, with possibly some error included
    ///     let component_at_i = ring.inclusion().mul_ref_map(&rhs, &rhs_op.gadget_vector(ring.get_ring()).at(i));
    ///     rhs_op.set_rns_factor(ring.get_ring(), i, ring.add(component_at_i, create_small_error()));
    /// }
    /// 
    /// // compute the gadget product
    /// let lhs = ring.random_element(|| rng.rand_u64());
    /// let lhs_op = GadgetProductLhsOperand::from_element(ring.get_ring(), &lhs, digits);
    /// let actual = lhs_op.gadget_product(&rhs_op, ring.get_ring());
    /// 
    /// // the final result should be close to `lhs * rhs`, except for some noise
    /// let expected = ring.mul_ref(&lhs, &rhs);
    /// let error = ring.sub(expected, actual);
    /// let error_coefficients = ring.wrt_canonical_basis(&error);
    /// let max_allowed_error = (113 / 2) * 8 * 3;
    /// assert!((0..8).all(|i| int_cast(ring.base_ring().smallest_lift(error_coefficients.at(i)), StaticRing::<i64>::RING, BigIntRing::RING).abs() <= max_allowed_error));
    /// ```
    /// 
    pub fn gadget_product(&self, rhs: &GadgetProductRhsOperand<R>, ring: &R) -> R::Element {
        assert_eq!(self.element_decomposition.len(), rhs.scaled_element.len(), "Gadget product operands created w.r.t. different digit sets");
        return ring.inner_product_prepared(self.element_decomposition.iter().zip(rhs.scaled_element.iter()).filter_map(|((lhs, _), rhs)| rhs.as_ref().map(|(rhs, _)| (lhs, rhs))));
    }
}

///
/// `gadget_decompose()[decomposed_component][rns_base_index]` contains the prepared convolution 
/// modulo `shortened_rns_base.at(rns_base_index)` of the `decomposed_component`-th element of the gadget 
/// decomposition vector. Here `shortened_rns_base` is formed by the last `output_moduli_count` rns 
/// components of the main rns base.
/// 
/// The order of the fourier coefficients is the same as specified by the corresponding [`GeneralizedFFT`].
/// 
fn gadget_decompose<R, V>(ring: &R, el: &R::Element, digits: V) -> Vec<(R::PreparedMultiplicant, R::Element)>
    where R: BGFVCiphertextRing,
        V: VectorFn<Range<usize>>
{
    let ZZbig = BigIntRing::RING;
    let ZZi64 = StaticRing::<i64>::RING;
    let mut result = Vec::new();
    let mut el_as_matrix = OwnedMatrix::zero(ring.base_ring().len(), ring.small_generating_set_len(), ring.base_ring().at(0));
    ring.as_representation_wrt_small_generating_set(el, el_as_matrix.data_mut());
    
    let homs = ring.base_ring().as_iter().map(|Zp| Zp.can_hom(&ZZi64).unwrap()).collect::<Vec<_>>();
    let mut current_row = Vec::with_capacity(el_as_matrix.col_count());
    current_row.resize_with(homs.len() * el_as_matrix.col_count(), || ring.base_ring().at(0).zero());
    
    for i in 0..digits.len() {
        
        let digit = digits.at(i);
        let conversion = UsedBaseConversion::new_with(
            digit.iter().map(|idx| *ring.base_ring().at(idx)).collect::<Vec<_>>(),
            homs.iter().map(|h| **h.codomain()).collect::<Vec<_>>(),
            Global
        );
        
        conversion.apply(
            el_as_matrix.data().restrict_rows(digit.clone()),
            SubmatrixMut::from_1d(&mut current_row[..], homs.len(), el_as_matrix.col_count())
        );

        let decomposition_part = ring.from_representation_wrt_small_generating_set(Submatrix::from_1d(&current_row[..], homs.len(), el_as_matrix.col_count()));
        result.push((
            ring.prepare_multiplicant(&decomposition_part),
            decomposition_part
        ));
    }
    return result;
}

pub struct GadgetProductRhsOperand<R: BGFVCiphertextRing> {
    /// `i`-th entry stores a (noisy) encryption/encoding/whatever of the represented element,
    /// scaled by the `i`-th entry of the gadget vector. `None` represents zero. We store the
    /// element once as `PreparedMultiplicant` for fast computation of gadget products, and once
    /// as the element itself, since there currently is no way of getting the ring element out of
    /// a `PreparedMultiplicant`
    scaled_element: Vec<Option<(R::PreparedMultiplicant, R::Element)>>,
    /// representation of the used gadget vector, the `i`-th entry of the gadget vector is the
    /// RNS unit vector that is 1 modulo exactly the RNS factors contained in the range at index
    /// `i` of this list
    digits: Vec<Range<usize>>
}

fn select_digits(digits: usize, rns_base_len: usize) -> Vec<Range<usize>> {
    let moduli_per_small_digit = rns_base_len / digits;
    let large_digits = rns_base_len % digits;
    let small_digits = digits - large_digits;
    return (0..large_digits).map(|_| moduli_per_small_digit + 1)
        .chain((0..small_digits).map(|_| moduli_per_small_digit))
        .scan(0, |current, next| {
            let result = *current..(*current + next);
            *current += next;
            return Some(result);
        }).collect();
}

impl<R: BGFVCiphertextRing> GadgetProductRhsOperand<R> {

    pub fn clone(&self, ring: &R) -> Self {
        Self {
            scaled_element: self.scaled_element.iter().map(|el| el.as_ref().map(|el| (ring.prepare_multiplicant(&el.1), ring.clone_el(&el.1)))).collect(),
            digits: self.digits.clone()
        }
    }

    pub fn gadget_vector<'b>(&'b self, ring: &'b R) -> impl VectorFn<El<zn_rns::Zn<Zn, BigIntRing>>> + use<'b, R> {
        self.digits.as_fn().map_fn(|digit| ring.base_ring().from_congruence((0..ring.base_ring().len()).map(|i| if digit.contains(&i) { ring.base_ring().at(i).one() } else { ring.base_ring().at(i).zero() })))
    }

    pub fn gadget_vector_moduli_indices<'b>(&'b self) -> impl VectorFn<Range<usize>> + use<'b, R> {
        self.digits.as_fn().map_fn(|digit| digit.clone())
    }

    pub fn set_rns_factor(&mut self, ring: &R, i: usize, el: R::Element) {
        self.scaled_element[i] = Some((ring.prepare_multiplicant(&el), el));
    }
    
    /// 
    /// Creates a [`GadgetProductRhsOperand`] representing `0` w.r.t. the RNS gadget vector that has `digits` digits.
    /// For an explanation of gadget products, see [`GadgetProductLhsOperand::gadget_product()`].
    /// 
    pub fn new(ring: &R, digits: usize) -> Self {
        Self::new_with(ring, select_digits(digits, ring.base_ring().len()))
    }

    /// 
    /// Creates a [`GadgetProductRhsOperand`] representing `0` w.r.t. the RNS gadget given by `digits`.
    /// For an explanation of gadget products, see [`GadgetProductLhsOperand::gadget_product()`].
    /// 
    pub fn new_with(ring: &R, digits: Vec<Range<usize>>) -> Self {
        let mut operands = Vec::with_capacity(digits.len());
        operands.extend((0..digits.len()).map(|_| None));
        return Self {
            scaled_element: operands,
            digits: digits
        };
    }

    pub fn modulus_switch(self, to: &R, dropped_rns_factors: &[usize], from: &R) -> Self {
        assert_eq!(to.base_ring().len() + dropped_rns_factors.len(), from.base_ring().len());
        debug_assert_eq!(self.digits.len(), self.scaled_element.len());
        let mut result_scaled_el = Vec::new();
        let mut result_digits = Vec::new();
        let mut current = 0;
        for (digit, scaled_el) in self.digits.iter().zip(self.scaled_element.into_iter()) {
            let old_digit_len = digit.end - digit.start;
            let dropped_from_digit = dropped_rns_factors.iter().filter(|i| digit.contains(&i)).count();
            assert!(dropped_from_digit <= old_digit_len);
            if dropped_from_digit == old_digit_len {
                continue;
            }
            result_digits.push(current..(current + old_digit_len - dropped_from_digit));
            current += old_digit_len - dropped_from_digit;
            if let Some((scaled_el_prepared, scaled_el)) = scaled_el {
                let new_scaled_el = to.drop_rns_factor_element(from, dropped_rns_factors, scaled_el);
                result_scaled_el.push(Some((to.drop_rns_factor_prepared(from, dropped_rns_factors, scaled_el_prepared), new_scaled_el)));
            } else {
                result_scaled_el.push(None);
            }
        }
        return Self {
            digits: result_digits,
            scaled_element: result_scaled_el
        };
    }
}

#[test]
fn test_gadget_decomposition() {
    let ring = SingleRNSRingBase::<_, Global, DefaultConvolution>::new(Pow2CyclotomicNumberRing::new(4), zn_rns::Zn::create_from_primes(vec![17, 97, 113], BigIntRing::RING));
    let rns_base = ring.base_ring();
    let from_congruence = |data: &[i32]| rns_base.from_congruence(data.iter().enumerate().map(|(i, c)| rns_base.at(i).int_hom().map(*c)));
    let hom_big = ring.base_ring().can_hom(&BigIntRing::RING).unwrap();
    let hom_i32 = ring.base_ring().can_hom(&StaticRing::<i32>::RING).unwrap();

    let mut rhs = GadgetProductRhsOperand::new(ring.get_ring(), 2);
    rhs.set_rns_factor(ring.get_ring(), 0, ring.inclusion().map(from_congruence(&[1, 1, 0])));
    rhs.set_rns_factor(ring.get_ring(), 1, ring.inclusion().map(from_congruence(&[0, 0, 1])));

    let lhs = GadgetProductLhsOperand::from_element(ring.get_ring(), &ring.inclusion().map(hom_i32.map(1000)), 2);

    assert_el_eq!(ring, ring.inclusion().map(hom_i32.map(1000)), lhs.gadget_product(&rhs, ring.get_ring()));
}

#[test]
fn test_modulus_switch() {
    let ring = SingleRNSRingBase::<_, Global, DefaultConvolution>::new(Pow2CyclotomicNumberRing::new(4), zn_rns::Zn::create_from_primes(vec![17, 97, 113], BigIntRing::RING));
    let rns_base = ring.base_ring();
    let from_congruence = |data: &[i32]| rns_base.from_congruence(data.iter().enumerate().map(|(i, c)| rns_base.at(i).int_hom().map(*c)));

    let mut rhs = GadgetProductRhsOperand::new(ring.get_ring(), 2);
    rhs.set_rns_factor(ring.get_ring(), 0, ring.inclusion().map(from_congruence(&[1, 1, 0])));
    rhs.set_rns_factor(ring.get_ring(), 1, ring.inclusion().map(from_congruence(&[0, 0, 1])));

    let smaller_ring = SingleRNSRingBase::<_, Global, DefaultConvolution>::new(Pow2CyclotomicNumberRing::new(4), zn_rns::Zn::create_from_primes(vec![17, 113], BigIntRing::RING));
    let rhs = rhs.modulus_switch(smaller_ring.get_ring(), &[1], ring.get_ring());
    let lhs = GadgetProductLhsOperand::from_element(smaller_ring.get_ring(), &smaller_ring.int_hom().map(1000), 2);

    assert_el_eq!(&smaller_ring, smaller_ring.int_hom().map(1000), lhs.gadget_product(&rhs, smaller_ring.get_ring()));

    let ring = SingleRNSRingBase::<_, Global, DefaultConvolution>::new(Pow2CyclotomicNumberRing::new(4), zn_rns::Zn::create_from_primes(vec![17, 97, 113, 193, 241], BigIntRing::RING));
    let rns_base = ring.base_ring();
    let from_congruence = |data: &[i32]| rns_base.from_congruence(data.iter().enumerate().map(|(i, c)| rns_base.at(i).int_hom().map(*c)));

    let mut rhs = GadgetProductRhsOperand::new(ring.get_ring(), 3);
    rhs.set_rns_factor(ring.get_ring(), 0, ring.inclusion().map(from_congruence(&[1000, 1000, 0, 0, 0])));
    rhs.set_rns_factor(ring.get_ring(), 1, ring.inclusion().map(from_congruence(&[0, 0, 1000, 1000, 0])));
    rhs.set_rns_factor(ring.get_ring(), 2, ring.inclusion().map(from_congruence(&[0, 0, 0, 0, 1000])));

    let smaller_ring = SingleRNSRingBase::<_, Global, DefaultConvolution>::new(Pow2CyclotomicNumberRing::new(4), zn_rns::Zn::create_from_primes(vec![17, 193, 241], BigIntRing::RING));
    let rhs = rhs.modulus_switch(smaller_ring.get_ring(), &[1, 2], ring.get_ring());
    let lhs = GadgetProductLhsOperand::from_element(smaller_ring.get_ring(), &smaller_ring.int_hom().map(1000), 3);

    assert_el_eq!(&smaller_ring, smaller_ring.int_hom().map(1000000), lhs.gadget_product(&rhs, smaller_ring.get_ring()));
}