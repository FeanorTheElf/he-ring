
///
/// Contains the main implementation [`double_rns_ring::DoubleRNSRing`], which implements most ring operations
/// for any ring that supports conversion from and to double-RNS-representation (via [`double_rns_ring::GeneralizedFFT`]).
///  
pub mod double_rns_ring;

///
/// Contains the [`double_rns_ring::GeneralizedFFT`] for power-of-two cyclotomics [`pow2_cyclotomic::Pow2CyclotomicFFT`].
/// 
pub mod pow2_cyclotomic;

///
/// Contains the [`double_rns_ring::GeneralizedFFT`] for odd-conductor cyclotomics [`odd_cyclotomic::OddCyclotomicFFT`].
/// 
pub mod odd_cyclotomic;

///
/// Contains an implementation of "external products". For details, see [`DoubleRNSRingBase::external_product()`].
/// 
pub mod gadget_product;
