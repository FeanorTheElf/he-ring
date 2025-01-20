use std::alloc::Global;
use std::ops::Range;

use feanor_math::integer::BigIntRing;
use feanor_math::matrix::{Submatrix, SubmatrixMut};
use feanor_math::primitive_int::StaticRing;
use feanor_math::{assert_el_eq, ring::*};
use feanor_math::rings::zn::{zn_64::Zn, zn_rns};
use feanor_math::seq::subvector::SubvectorView;
use feanor_math::seq::{VectorFn, VectorView};
use feanor_math::homomorphism::Homomorphism;

use crate::ciphertext_ring::double_rns_ring::DoubleRNSRingBase;
use crate::ciphertext_ring::single_rns_ring::SingleRNSRingBase;
use crate::ciphertext_ring::BGFVCiphertextRing;
use crate::number_ring::pow2_cyclotomic::Pow2CyclotomicNumberRing;
use crate::rnsconv::{lift, RNSOperation};
use crate::DefaultConvolution;

type UsedBaseConversion<A> = lift::AlmostExactBaseConversion<A>;

pub struct GadgetProductLhsOperand<R: BGFVCiphertextRing> {
    element_decomposition: Vec<R::PreparedMultiplicant>
}

impl<R: BGFVCiphertextRing> GadgetProductLhsOperand<R> {

    pub fn from_element_with<V>(ring: &R, el: &R::Element, digits: V) -> Self
        where V: VectorFn<Range<usize>>
    {
        let decomposition = gadget_decompose(ring, el, digits);
        return Self {
            element_decomposition: decomposition
        };
    }

    pub fn from_element(ring: &R, el: &R::Element, digits: usize) -> Self {
        Self::from_element_with(ring, el, select_digits(digits, ring.base_ring().len()).clone_els())
    }

    pub fn gadget_product(&self, rhs: &GadgetProductRhsOperand<R>, ring: &R) -> R::Element {
        assert_eq!(self.element_decomposition.len(), rhs.scaled_element.len(), "Gadget product operands created w.r.t. different digit sets");
        return ring.inner_product_prepared(self.element_decomposition.iter().zip(rhs.scaled_element.iter()).filter_map(|(lhs, rhs)| rhs.as_ref().map(|rhs| (lhs, rhs))));
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
fn gadget_decompose<R, V>(ring: &R, el: &R::Element, digits: V) -> Vec<R::PreparedMultiplicant>
    where R: BGFVCiphertextRing,
        V: VectorFn<Range<usize>>
{
    let ZZbig = BigIntRing::RING;
    let ZZi64 = StaticRing::<i64>::RING;
    let mut result = Vec::new();
    let el_as_matrix = ring.as_representation_wrt_small_generating_set(el);
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
            el_as_matrix.restrict_rows(digit.clone()),
            SubmatrixMut::from_1d(&mut current_row[..], homs.len(), el_as_matrix.col_count())
        );

        result.push(ring.prepare_multiplicant(ring.from_representation_wrt_small_generating_set(Submatrix::from_1d(&current_row[..], homs.len(), el_as_matrix.col_count()))));
    }
    return result;
}

pub struct GadgetProductRhsOperand<R: BGFVCiphertextRing> {
    scaled_element: Vec<Option<R::PreparedMultiplicant>>,
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

    pub fn gadget_vector<'b>(&'b self, ring: &'b R) -> impl VectorFn<El<zn_rns::Zn<Zn, BigIntRing>>> + use<'b, R> {
        self.digits.as_fn().map_fn(|digit| ring.base_ring().from_congruence((0..ring.base_ring().len()).map(|i| if digit.contains(&i) { ring.base_ring().at(i).one() } else { ring.base_ring().at(i).zero() })))
    }

    pub fn gadget_vector_moduli_indices<'b>(&'b self) -> impl VectorFn<Range<usize>> + use<'b, R> {
        self.digits.as_fn().map_fn(|digit| digit.clone())
    }

    pub fn set_rns_factor(&mut self, ring: &R, i: usize, el: R::Element) {
        self.scaled_element[i] = Some(ring.prepare_multiplicant(el));
    }
    
    pub fn new(ring: &R, digits: usize) -> Self {
        Self::new_with(ring, select_digits(digits, ring.base_ring().len()))
    }

    pub fn new_with(ring: &R, digits: Vec<Range<usize>>) -> Self {
        let mut operands = Vec::with_capacity(digits.len());
        operands.extend((0..digits.len()).map(|_| None));
        return Self {
            scaled_element: operands,
            digits: digits
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
