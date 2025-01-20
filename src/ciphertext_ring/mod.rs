use feanor_math::integer::BigIntRing;
use feanor_math::matrix::{AsFirstElement, AsPointerToSlice, Submatrix};
use feanor_math::ring::*;
use feanor_math::rings::extension::FreeAlgebra;
use feanor_math::rings::zn::zn_64::ZnEl;
use feanor_math::rings::zn::{zn_64::Zn, zn_rns};

use crate::number_ring::HECyclotomicNumberRing;

///
/// Code for fast polynomial division by a cyclotomic polynomial.
/// Used within [`single_rns_ring`].
/// 
pub mod poly_remainder;

///
/// Contains [`double_rns_ring::DoubleRNSRing`], an implementation of the ring `R/qR` for suitable `q`.
///  
pub mod double_rns_ring;

///
/// Contains [`single_rns_ring::SingleRNSRing`], an implementation of the ring `R/qR` for suitable `q`.
///  
pub mod single_rns_ring;

///
/// Contains [`double_rns_managed::ManagedDoubleRNSRing`], an implementation of the ring `R/qR` for suitable `q`
/// that is based on [`double_rns_ring::DoubleRNSRing`].
///  
pub mod double_rns_managed;

///
/// Trait for rings `R/qR` with a number ring `R` and modulus `q = p1 ... pr` represented as 
/// RNS basis, which provide all necessary operations for use as ciphertext ring in BFV/BGV-style
/// HE schemes.
/// 
pub trait BGFVCiphertextRing: RingBase + FreeAlgebra + RingExtension<BaseRing = zn_rns::Zn<Zn, BigIntRing>> {

    type NumberRing: HECyclotomicNumberRing;

    ///
    /// Returns a view on the underlying representation of `x`. 
    /// 
    /// This is the counterpart of [`BGFVCiphertextRing::from_representation_wrt_small_generating_set()`].
    /// 
    /// More concretely, for some `Zq`-linear generating set `{ a_i | i }` consisting
    /// of ring elements of small canonical norm, each column of the returned matrix contains
    /// the RNS representation of some `x_i`, satisfying `x = sum_i a_i x_i`. The actual choice
    /// of the `a_i` is left to the ring implementation, and may change in future releases.
    /// The order of the rows (corresponding to the RNS factors of `Zq`) is the same as the
    /// order of the RNS factors in `self.base_ring()`.
    /// 
    /// Hence, it is recommended to instead use [`FreeAlgebra::wrt_canonical_basis()`] and
    /// [`FreeAlgebra::from_canonical_basis()`], whose result is uniquely defined. However, note
    /// that these may incur costs for internal representation conversion, which may not always
    /// be acceptable.
    /// 
    fn as_representation_wrt_small_generating_set<'a>(&'a self, x: &'a Self::Element) -> Submatrix<'a, AsFirstElement<ZnEl>, ZnEl>;

    ///
    /// Creates a ring element from its underlying representation.
    /// 
    /// This is the counterpart of [`BGFVCiphertextRing::as_representation_wrt_small_generating_set()`].
    /// 
    /// More concretely, for some `Zq`-linear generating set `{ a_i | i }` consisting
    /// of ring elements of small canonical norm, each column of the given matrix is interpreted
    /// as the RNS representation of some `x_i`, and the returned ring element is then
    /// `x = sum_i a_i x_i`. The actual choice of the `a_i` is left to the ring implementation, 
    /// and may change in future releases. The order of the rows (corresponding to the RNS factors of `Zq`) 
    /// is the same as the order of the RNS factors in `self.base_ring()`.
    /// 
    /// Hence, it is recommended to instead use [`FreeAlgebra::wrt_canonical_basis()`] and
    /// [`FreeAlgebra::from_canonical_basis()`], whose result is uniquely defined. However, note
    /// that these may incur costs for internal representation conversion, which may not always
    /// be acceptable.
    /// 
    fn from_representation_wrt_small_generating_set<V>(&self, data: Submatrix<V, ZnEl>) -> Self::Element
        where V: AsPointerToSlice<ZnEl>;

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