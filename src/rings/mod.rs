
///
/// Code for fast, precomputed polynomial interpolation.
/// Used for packing of values into slots (within [`hypercube`]).
/// 
pub mod interpolate;

///
/// Code for fast polynomial division by a cyclotomic polynomial.
/// Used within [`single_rns_ring`].
/// 
pub mod poly_remainder;

///
/// Abstraction of number rings.
/// 
pub mod number_ring;

///
/// Contains an implementation of "gadget products". For details, see [`double_rns_ring::DoubleRNSRingBase::gadget_product()`].
/// 
pub mod gadget_product;

///
/// Contains [`double_rns_ring::DoubleRNSRing`], an implementation of the ring `R/qR` for suitable `q`.
///  
pub mod double_rns_ring;

///
/// Contains [`single_rns_ring::SingleRNSRing`], an implementation of the ring `R/qR` for suitable `q`.
///  
pub mod single_rns_ring;

///
/// Contains [`decomposition_ring::DecompositionRing`], an implementation of the ring `R/tR` for any `t`.
///  
pub mod decomposition_ring;

///
/// Contains an implementation of power-of-two cyclotomic number rings.
/// 
pub mod pow2_cyclotomic;

///
/// Contains an implementation of odd cyclotomic number rings.
/// 
pub mod odd_cyclotomic;

///
/// Contains [`double_rns_managed::ManagedDoubleRNSRing`], an implementation of the ring `R/qR` for suitable `q`
/// that is based on [`double_rns_ring::DoubleRNSRing`].
///  
pub mod double_rns_managed;

///
/// Contains the trait [`bxv::BXVCiphertextRing`] for rings that can be used as ciphertext
/// ring during second-generation "BXV" HE schemes.
/// 
pub mod bxv;

///
/// Contains the description of the ring isomorphism `R/p^eR = GF(p, e, d)^l` via a hypercube.
/// 
pub mod hypercube;