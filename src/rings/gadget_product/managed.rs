use std::cell::RefCell;
use std::rc::Rc;
use std::{alloc::Allocator, ops::Range};

use feanor_math::ring::*;
use feanor_math::rings::zn::zn_64::{Zn, ZnEl};

use crate::rings::double_rns_managed::*;
use crate::rings::number_ring::*;
use super::double_rns;

pub struct GadgetProductRhsOperand<'a, NumberRing, A> 
    where NumberRing: HENumberRing<Zn>,
        A: Allocator + Clone
{
    ring: &'a ManagedDoubleRNSRing<NumberRing, Zn, A>,
    base: double_rns::GadgetProductRhsOperand<'a, NumberRing, A>
}

impl<'a, NumberRing, A> GadgetProductRhsOperand<'a, NumberRing, A> 
    where NumberRing: HENumberRing<Zn>,
        A: Allocator + Clone
{
    pub fn gadget_vector<'b>(&'b self) -> &'b [Range<usize>] {
        self.base.gadget_vector()
    }

    pub fn set_rns_factor(&mut self, i: usize, el: ManagedDoubleRNSEl<NumberRing, Zn, A>) {
        if let Some(nonzero) = self.ring.get_ring().to_coeff(&el) {
            self.base.set_rns_factor(i, self.ring.get_ring().base.clone_el_non_fft(&*nonzero));
        }
    }
}

pub struct GadgetProductLhsOperand<'a, NumberRing, A> 
    where NumberRing: HENumberRing<Zn>,
        A: Allocator + Clone
{
    base: double_rns::GadgetProductLhsOperand<'a, NumberRing, A>
}

impl<'a, NumberRing, A> GadgetProductLhsOperand<'a, NumberRing, A> 
    where NumberRing: HECyclotomicNumberRing<Zn>,
        A: Allocator + Clone
{
    pub fn apply_galois_action(&self, ring: &ManagedDoubleRNSRingBase<NumberRing, Zn, A>, g: ZnEl) -> Self {
        GadgetProductLhsOperand {
            base: self.base.apply_galois_action(&ring.base, g)
        }
    }
}

impl<NumberRing, A> ManagedDoubleRNSRingBase<NumberRing, Zn, A> 
    where NumberRing: HENumberRing<Zn>,
        A: Allocator + Clone
{
    pub fn to_gadget_product_lhs<'a>(&'a self, el: ManagedDoubleRNSEl<NumberRing, Zn, A>, digits: usize) -> GadgetProductLhsOperand<'a, NumberRing, A> {
        if let Some(nonzero) = self.to_coeff(&el) {
            GadgetProductLhsOperand {
                base: self.base.to_gadget_product_lhs(self.base.clone_el_non_fft(&*nonzero), digits)
            }
        } else {
            GadgetProductLhsOperand {
                base: self.base.to_gadget_product_lhs(self.base.clone_el_non_fft(&self.zero), digits)
            }
        }
    }

    pub fn gadget_product_rhs_empty<'a, const LOG: bool>(&'a self, digits: usize) -> GadgetProductRhsOperand<'a, NumberRing, A> {
        GadgetProductRhsOperand {
            base: self.base.gadget_product_rhs_empty::<LOG>(digits),
            ring: RingValue::from_ref(self)
        }
    }

    pub fn gadget_product(&self, lhs: &GadgetProductLhsOperand<NumberRing, A>, rhs: &GadgetProductRhsOperand<NumberRing, A>) -> ManagedDoubleRNSEl<NumberRing, Zn, A> {
        match self.base.preferred_output_repr(&lhs.base, &rhs.base) {
            double_rns::ElRepr::Coeff => ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::Coeff(self.base.gadget_product_coeff(&lhs.base, &rhs.base)))) },
            double_rns::ElRepr::NTT => ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::DoubleRNS(self.base.gadget_product_ntt(&lhs.base, &rhs.base)))) }
        }
    }
}
