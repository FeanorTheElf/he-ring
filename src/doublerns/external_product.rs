use feanor_math::integer::BigIntRingBase;
use feanor_math::mempool::*;
use feanor_math::ring::*;
use feanor_math::homomorphism::*;
use feanor_math::rings::zn::*;
use feanor_math::rings::extension::*;
use feanor_math::vector::*;

use super::double_rns_ring::*;

pub struct ExternalProductRhsOperand<R, F, M> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanonicalIso<F::BaseRingBase> + CanHomFrom<BigIntRingBase>,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    operands: Vec<<DoubleRNSRingBase<R, F, M> as RingBase>::Element>
}

impl<R, F, M> ExternalProductRhsOperand<R, F, M>
    where R: ZnRingStore,
        R::Type: ZnRing + CanonicalIso<F::BaseRingBase> + CanHomFrom<BigIntRingBase>,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    pub fn set_rns_factor(&mut self, i: usize, el: <DoubleRNSRingBase<R, F, M> as RingBase>::Element) {
        self.operands[i] = el;
    }
}

pub struct ExternalProductLhsOperand<R, F, M> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanonicalIso<F::BaseRingBase> + CanHomFrom<BigIntRingBase>,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    operands: Vec<<DoubleRNSRingBase<R, F, M> as RingBase>::Element>
}

impl<R, F, M> DoubleRNSRingBase<R, F, M>
    where R: ZnRingStore,
        R::Type: ZnRing + CanonicalIso<F::BaseRingBase> + CanHomFrom<BigIntRingBase>,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    pub fn to_external_product_lhs(&self, el: DoubleRNSNonFFTEl<R, F, M>) -> ExternalProductLhsOperand<R, F, M> {
        let mut result: Vec<DoubleRNSNonFFTEl<R, F, M>> = (0..self.rns_base().len()).map(|_| self.non_fft_zero()).collect();

        let non_fft_el = el;
        for i in 0..self.rns_base().len() {
            for j in 0..self.rank() {
                let int_ring = self.rns_base().at(i).integer_ring();
                let value = self.rns_base().at(i).smallest_lift(self.rns_base().at(i).clone_el(self.at(i, j, &non_fft_el)));
                for i2 in 0..self.rns_base().len() {
                    *self.at_mut(i2, j, &mut result[i]) = self.rns_base().at(i2).coerce(int_ring, int_ring.clone_el(&value));
                }
            }
        }

        return ExternalProductLhsOperand {
            operands: result.into_iter().map(|x| self.do_fft(x)).collect()
        }
    }

    pub fn external_product_rhs_zero(&self) -> ExternalProductRhsOperand<R, F, M> {
        ExternalProductRhsOperand {
            operands: (0..self.rns_base().len()).map(|_| self.zero()).collect()
        }
    }

    pub fn external_product(&self, lhs: &ExternalProductLhsOperand<R, F, M>, rhs: &ExternalProductRhsOperand<R, F, M>) -> <Self as RingBase>::Element {
        <_ as RingBase>::sum(self, lhs.operands.iter().zip(rhs.operands.iter()).map(|(l, r)| self.mul_ref(l, r)))
    }
}