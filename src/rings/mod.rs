
pub mod decomposition;

///
/// Contains the main implementation [`double_rns_ring::DoubleRNSRing`], which implements most ring operations
/// for any ring that supports conversion from and to double-RNS-representation.
///  
pub mod double_rns_ring;

pub mod ntt_ring;

///
/// Contains the [`double_rns_ring::GeneralizedFFT`] for power-of-two cyclotomics [`pow2_cyclotomic::Pow2CyclotomicFFT`].
/// 
pub mod pow2_cyclotomic;

///
/// Contains an implementation of "gadget products". For details, see [`double_rns_ring::DoubleRNSRingBase::gadget_product()`].
/// 
pub mod gadget_product;

///
/// Contains the [`double_rns_ring::GeneralizedFFT`] for odd-conductor cyclotomics [`odd_cyclotomic::OddCyclotomicFFT`].
/// 
pub mod odd_cyclotomic;
