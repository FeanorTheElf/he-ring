
///
/// Contains the main implementation [`complex_fft_ring::ComplexFFTBasedRing`], which implements most ring operations
/// without fixing on a concrete ring, but instead is configured by a [`complex_fft_ring::GeneralizedFFT`].
///  
pub mod complex_fft_ring;

///
/// Contains [`pow2_cyclotomic::Pow2CyclotomicFFT`], a [`complex_fft_ring::GeneralizedFFT`] for power-of-two cyclotomic rings.
/// 
pub mod pow2_cyclotomic;

///
/// Contains [`odd_cyclotomic::OddCyclotomicFFT`], a [`complex_fft_ring::GeneralizedFFT`] for odd-conductor cyclotomic rings.
/// 
pub mod odd_cyclotomic;
