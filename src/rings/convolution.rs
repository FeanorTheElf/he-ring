use feanor_math::algorithms::convolution::PreparedConvolutionAlgorithm;
use feanor_math::ring::*;

pub trait FromRingCreateableConvolution<R: RingStore>: PreparedConvolutionAlgorithm<R::Type> {

    fn create(ring: R, max_log2_len: usize) -> Self;
}

#[cfg(feature = "use_hexl")]
impl FromRingCreateableConvolution<feanor_math::rings::zn::zn_64::Zn> for feanor_math_hexl::conv::HEXLConvolution {

    fn create(ring: feanor_math::rings::zn::zn_64::Zn, max_log2_len: usize) -> Self {
        Self::new(ring, max_log2_len)
    }
}