

pub mod interpolate;

pub mod poly_remainder;

pub mod number_ring;

///
/// Contains an implementation of "gadget products". For details, see [`double_rns_ring::DoubleRNSRingBase::gadget_product()`].
/// 
pub mod gadget_product;

///
/// Contains the main implementation [`double_rns_ring::DoubleRNSRing`], which implements most ring operations
/// for any ring that supports conversion from and to double-RNS-representation.
///  
pub mod double_rns_ring;

pub mod single_rns_ring;

pub mod decomposition_ring;

pub mod pow2_cyclotomic;

pub mod odd_cyclotomic;

pub mod double_rns_managed;

pub mod bxv;

pub mod hypercube;