use double_rns::{ElRepr, GadgetProductLhsOperand, GadgetProductRhsOperand};
use feanor_math::rings::zn::zn_64::Zn;

use super::*;


///
/// Puts the components of `ct` into coefficient representation.
/// 
/// This is seldomly sensible in an application, as the components are automatically
/// mapped from and to coefficient representation as deemed most efficient. However, this
/// makes getting reliable benchmarks difficult, since one never knows if a certain time
/// included representation-switching or not. Thus, in such benchmarks, every
/// operation should be followed by `coeff_repr()`.
/// 
pub fn coeff_repr<Params: BFVParams>(C: &CiphertextRing<Params>, ct: Ciphertext<Params>) -> Ciphertext<Params> {
    (ct.0.coeff_repr(C), ct.1.coeff_repr(C))
}

pub struct CoeffOrDoubleRNSEl<NumberRing: HENumberRing<Zn>> {
    ntt_part: Option<DoubleRNSEl<NumberRing, Zn, CiphertextAllocator>>,
    coeff_part: Option<CoeffEl<NumberRing, Zn, CiphertextAllocator>>
}

impl<NumberRing: HENumberRing<Zn>> CoeffOrDoubleRNSEl<NumberRing> {

    pub fn ntt_repr(self, C: &DoubleRNSRing<NumberRing, Zn, CiphertextAllocator>) -> Self {
        CoeffOrDoubleRNSEl::from_ntt(self.to_ntt(C))
    }

    pub fn coeff_repr(self, C: &DoubleRNSRing<NumberRing, Zn, CiphertextAllocator>) -> Self {
        CoeffOrDoubleRNSEl::from_coeff(self.to_coeff(C))
    }

    pub fn from_ntt(el: DoubleRNSEl<NumberRing, Zn, CiphertextAllocator>) -> Self {
        Self {
            coeff_part: None,
            ntt_part: Some(el)
        }
    }

    pub fn from_coeff(el: CoeffEl<NumberRing, Zn, CiphertextAllocator>) -> Self {
        Self {
            coeff_part: Some(el),
            ntt_part: None
        }
    }

    pub fn zero() -> Self {
        Self {
            coeff_part: None,
            ntt_part: None
        }
    }

    pub fn to_ntt(self, C: &DoubleRNSRing<NumberRing, Zn, CiphertextAllocator>) -> DoubleRNSEl<NumberRing, Zn, CiphertextAllocator> {
        if let Some(mut result) = self.ntt_part {
            if let Some(coeff) = self.coeff_part {
                C.add_assign(&mut result, C.get_ring().do_fft(coeff));
            }
            return result;
        } else if let Some(coeff) = self.coeff_part {
            return C.get_ring().do_fft(coeff);
        } else {
            return C.zero();
        }
    }

    pub fn to_coeff(self, C: &DoubleRNSRing<NumberRing, Zn, CiphertextAllocator>) -> CoeffEl<NumberRing, Zn, CiphertextAllocator> {
        if let Some(mut result) = self.coeff_part {
            if let Some(ntt_part) = self.ntt_part {
                C.get_ring().add_assign_non_fft(&mut result, &C.get_ring().undo_fft(ntt_part));
            }
            return result;
        } else if let Some(ntt_part) = self.ntt_part {
            return C.get_ring().undo_fft(ntt_part);
        } else {
            return C.get_ring().non_fft_zero();
        }
    }

    pub fn gadget_product<'a>(lhs: &GadgetProductLhsOperand<'a, NumberRing, CiphertextAllocator>, rhs: &GadgetProductRhsOperand<'a, NumberRing, CiphertextAllocator>, C: &DoubleRNSRing<NumberRing, Zn, CiphertextAllocator>) -> CoeffOrDoubleRNSEl<NumberRing> {
        match C.get_ring().preferred_output_repr(lhs, rhs) {
            ElRepr::Coeff => CoeffOrDoubleRNSEl { ntt_part: None, coeff_part: Some(C.get_ring().gadget_product_coeff(lhs, rhs)) },
            ElRepr::NTT => CoeffOrDoubleRNSEl { ntt_part: Some(C.get_ring().gadget_product_ntt(lhs, rhs)), coeff_part: None },
        }
    }

    pub fn add(lhs: CoeffOrDoubleRNSEl<NumberRing>, rhs: &CoeffOrDoubleRNSEl<NumberRing>, C: &DoubleRNSRing<NumberRing, Zn, CiphertextAllocator>) -> CoeffOrDoubleRNSEl<NumberRing> {
        CoeffOrDoubleRNSEl {
            ntt_part: if lhs.ntt_part.is_some() && rhs.ntt_part.is_some() { Some(C.add_ref_snd(lhs.ntt_part.unwrap(), rhs.ntt_part.as_ref().unwrap())) } else { lhs.ntt_part.or(rhs.ntt_part.as_ref().map(|x| C.clone_el(x)))},
            coeff_part: if lhs.coeff_part.is_some() && rhs.coeff_part.is_some() {
                let mut result  = lhs.coeff_part.unwrap();
                C.get_ring().add_assign_non_fft(&mut result, rhs.coeff_part.as_ref().unwrap());
                Some(result)
            } else { lhs.coeff_part.or(rhs.coeff_part.as_ref().map(|x| C.get_ring().clone_el_non_fft(x))) }
        }
    }

    pub fn sub(lhs: CoeffOrDoubleRNSEl<NumberRing>, rhs: &CoeffOrDoubleRNSEl<NumberRing>, C: &DoubleRNSRing<NumberRing, Zn, CiphertextAllocator>) -> CoeffOrDoubleRNSEl<NumberRing> {
        CoeffOrDoubleRNSEl {
            ntt_part: if lhs.ntt_part.is_some() && rhs.ntt_part.is_some() { Some(C.sub_ref_snd(lhs.ntt_part.unwrap(), rhs.ntt_part.as_ref().unwrap())) } else { lhs.ntt_part.or(rhs.ntt_part.as_ref().map(|x| C.negate(C.clone_el(x))))},
            coeff_part: if lhs.coeff_part.is_some() && rhs.coeff_part.is_some() {
                let mut result  = lhs.coeff_part.unwrap();
                C.get_ring().sub_assign_non_fft(&mut result, rhs.coeff_part.as_ref().unwrap());
                Some(result)
            } else { lhs.coeff_part.or(rhs.coeff_part.as_ref().map(|x| C.get_ring().negate_non_fft(C.get_ring().clone_el_non_fft(x)))) }
        }
    }

    pub fn mul_i64(mut val: CoeffOrDoubleRNSEl<NumberRing>, scalar: i64, C: &DoubleRNSRing<NumberRing, Zn, CiphertextAllocator>) -> CoeffOrDoubleRNSEl<NumberRing> {
        let hom = C.base_ring().can_hom(&StaticRing::<i64>::RING).unwrap();
        if let Some(ntt_part) = &mut val.ntt_part {
            C.inclusion().mul_assign_map(ntt_part, hom.map(scalar));
        }
        if let Some(coeff_part) = &mut val.coeff_part {
            C.get_ring().mul_scalar_assign_non_fft(coeff_part, &hom.map(scalar));
        }
        return val;
    }

    pub fn clone(&self, C: &DoubleRNSRing<NumberRing, Zn, CiphertextAllocator>) -> CoeffOrDoubleRNSEl<NumberRing> {
        CoeffOrDoubleRNSEl { 
            ntt_part: self.ntt_part.as_ref().map(|x| C.clone_el(x)), 
            coeff_part: self.coeff_part.as_ref().map(|x| C.get_ring().clone_el_non_fft(x))
        }
    }
}
