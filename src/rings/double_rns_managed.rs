use std::alloc::Allocator;
use std::alloc::Global;
use std::cell::Ref;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::AtomicU64;

use feanor_math::assert_el_eq;
use feanor_math::delegate::DelegateRing;
use feanor_math::homomorphism::*;
use feanor_math::integer::*;
use feanor_math::ring::*;
use feanor_math::rings::extension::*;
use feanor_math::rings::finite::FiniteRing;
use feanor_math::rings::zn::*;
use zn_64::Zn;

use crate::cyclotomic::CyclotomicRing;
use crate::rings::number_ring::HENumberRing;
use crate::rnsconv::RNSOperation;

use super::bxv::BXVCiphertextRing;
use super::decomposition_ring::DecompositionRing;
use super::decomposition_ring::DecompositionRingBase;
use super::double_rns_ring::*;
use super::gadget_product;
use super::number_ring::HECyclotomicNumberRing;
use super::pow2_cyclotomic::Pow2CyclotomicDecomposableNumberRing;

///
/// Like [`DoubleRNSRing`] but stores element in whatever representation they
/// currently are available, and automatically switches representation when 
/// necessary.
/// 
pub struct ManagedDoubleRNSRingBase<NumberRing, FpTy, A = Global> 
    where NumberRing: HENumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    base: DoubleRNSRingBase<NumberRing, FpTy, A>,
    zero: CoeffEl<NumberRing, FpTy, A>
}

pub type ManagedDoubleRNSRing<NumberRing, FpTy, A = Global> = RingValue<ManagedDoubleRNSRingBase<NumberRing, FpTy, A>>;

impl<NumberRing, FpTy> ManagedDoubleRNSRingBase<NumberRing, FpTy, Global> 
    where NumberRing: HENumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>
{
    pub fn new(number_ring: NumberRing, rns_base: zn_rns::Zn<FpTy, BigIntRing>) -> RingValue<Self> {
        let result = DoubleRNSRingBase::new(number_ring, rns_base);
        let zero = result.get_ring().zero_non_fft();
        ManagedDoubleRNSRing::from(ManagedDoubleRNSRingBase { base: result.into(), zero: zero })
    }
}

enum ManagedDoubleRNSElRepresentation {
    Sum,
    Coeff,
    DoubleRNS,
    Both,
    Zero
}

enum DoubleRNSElInternal<NumberRing, FpTy, A = Global> 
    where NumberRing: HENumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    Sum(CoeffEl<NumberRing, FpTy, A>, DoubleRNSEl<NumberRing, FpTy, A>),
    Coeff(CoeffEl<NumberRing, FpTy, A>),
    DoubleRNS(DoubleRNSEl<NumberRing, FpTy, A>),
    Both(CoeffEl<NumberRing, FpTy, A>, DoubleRNSEl<NumberRing, FpTy, A>),
    Zero
}

impl<NumberRing, FpTy, A> DoubleRNSElInternal<NumberRing, FpTy, A> 
    where NumberRing: HENumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    fn unwrap_sum(self) -> (CoeffEl<NumberRing, FpTy, A>, DoubleRNSEl<NumberRing, FpTy, A>) {
        match self {
            DoubleRNSElInternal::Sum(coeff, doublerns) => (coeff, doublerns),
            _ => unreachable!()
        }
    }

    fn unwrap_coeff(self) -> CoeffEl<NumberRing, FpTy, A> {
        match self {
            DoubleRNSElInternal::Coeff(coeff) => coeff,
            _ => unreachable!()
        }
    }

    fn unwrap_ref_coeff<'a>(&'a self) -> &'a CoeffEl<NumberRing, FpTy, A> {
        match self {
            DoubleRNSElInternal::Coeff(coeff) => coeff,
            DoubleRNSElInternal::Both(coeff, _) => coeff,
            _ => unreachable!()
        }
    }
    
    fn unwrap_doublerns(self) -> DoubleRNSEl<NumberRing, FpTy, A> {
        match self {
            DoubleRNSElInternal::DoubleRNS(doublerns) => doublerns,
            _ => unreachable!()
        }
    }

    fn unwrap_ref_doublerns<'a>(&'a self) -> &'a DoubleRNSEl<NumberRing, FpTy, A> {
        match self {
            DoubleRNSElInternal::DoubleRNS(doublerns) => doublerns,
            DoubleRNSElInternal::Both(_, doublerns) => doublerns,
            _ => unreachable!()
        }
    }
}

pub struct ManagedDoubleRNSEl<NumberRing, FpTy, A = Global> 
    where NumberRing: HENumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    internal: Rc<RefCell<DoubleRNSElInternal<NumberRing, FpTy, A>>>
}

impl<NumberRing, FpTy, A> ManagedDoubleRNSRingBase<NumberRing, FpTy, A>
    where NumberRing: HENumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    ///
    /// Puts the internal value into coefficient representation.
    /// 
    /// This is seldomly sensible in an application, as the representation is automatically
    /// managed as deemed most efficient. However, this makes getting reliable benchmarks 
    /// difficult, since one never knows if a certain time included representation-switching 
    /// or not. Thus, in such benchmarks, every operation should be followed by 
    /// `force_coeff_repr()`.
    /// 
    pub fn force_coeff_repr(&self, value: &ManagedDoubleRNSEl<NumberRing, FpTy, A>) {
        if let Some(coeff) = self.to_coeff(value) {
            let new_value = self.base.clone_el_non_fft(&*coeff);
            drop(coeff);
            *value.internal.borrow_mut() = DoubleRNSElInternal::Coeff(new_value);
        }
    }

    fn get_repr(&self, value: &ManagedDoubleRNSEl<NumberRing, FpTy, A>) -> ManagedDoubleRNSElRepresentation {
        match &*value.internal.borrow() {
            DoubleRNSElInternal::Sum(_, _) => ManagedDoubleRNSElRepresentation::Sum,
            DoubleRNSElInternal::Coeff(_) => ManagedDoubleRNSElRepresentation::Coeff,
            DoubleRNSElInternal::DoubleRNS(_) => ManagedDoubleRNSElRepresentation::DoubleRNS,
            DoubleRNSElInternal::Both(_, _) => ManagedDoubleRNSElRepresentation::Both,
            DoubleRNSElInternal::Zero => ManagedDoubleRNSElRepresentation::Zero
        }
    }

    fn to_coeff<'a>(&self, value: &'a ManagedDoubleRNSEl<NumberRing, FpTy, A>) -> Option<Ref<'a, CoeffEl<NumberRing, FpTy, A>>> {
        match self.get_repr(value) {
            ManagedDoubleRNSElRepresentation::Sum => {
                let mut result = value.internal.borrow_mut();
                let (mut coeff, doublerns) = std::mem::replace(&mut *result, DoubleRNSElInternal::Zero).unwrap_sum();
                self.base.add_assign_non_fft(&mut coeff, &self.base.undo_fft(doublerns));
                *result = DoubleRNSElInternal::Coeff(coeff);
                drop(result);
                return Some(Ref::map(value.internal.borrow(), |x| x.unwrap_ref_coeff()));
            },
            ManagedDoubleRNSElRepresentation::Coeff | ManagedDoubleRNSElRepresentation::Both => {
                return Some(Ref::map(value.internal.borrow(), |x| x.unwrap_ref_coeff()));
            },
            ManagedDoubleRNSElRepresentation::DoubleRNS => {
                let mut result = value.internal.borrow_mut();
                let coeff = self.base.undo_fft(self.base.clone_el(result.unwrap_ref_doublerns()));
                *result = DoubleRNSElInternal::Both(coeff, std::mem::replace(&mut *result, DoubleRNSElInternal::Zero).unwrap_doublerns());
                drop(result);
                return Some(Ref::map(value.internal.borrow(), |x| x.unwrap_ref_coeff()));
            },
            ManagedDoubleRNSElRepresentation::Zero => {
                return None;
            }
        }
    }

    fn to_doublerns<'a>(&self, value: &'a ManagedDoubleRNSEl<NumberRing, FpTy, A>) -> Option<Ref<'a, DoubleRNSEl<NumberRing, FpTy, A>>> {
        match self.get_repr(value) {
            ManagedDoubleRNSElRepresentation::Sum => {
                let mut result = value.internal.borrow_mut();
                let (coeff, mut doublerns) = std::mem::replace(&mut *result, DoubleRNSElInternal::Zero).unwrap_sum();
                self.base.add_assign(&mut doublerns, self.base.do_fft(coeff));
                *result = DoubleRNSElInternal::DoubleRNS(doublerns);
                drop(result);
                return Some(Ref::map(value.internal.borrow(), |x| x.unwrap_ref_doublerns()));
            },
            ManagedDoubleRNSElRepresentation::DoubleRNS | ManagedDoubleRNSElRepresentation::Both => {
                return Some(Ref::map(value.internal.borrow(), |x| x.unwrap_ref_doublerns()));
            },
            ManagedDoubleRNSElRepresentation::Coeff => {
                let mut result = value.internal.borrow_mut();
                let doublerns = self.base.do_fft(self.base.clone_el_non_fft(result.unwrap_ref_coeff()));
                *result = DoubleRNSElInternal::Both(std::mem::replace(&mut *result, DoubleRNSElInternal::Zero).unwrap_coeff(), doublerns);
                drop(result);
                return Some(Ref::map(value.internal.borrow(), |x| x.unwrap_ref_doublerns()));
            },
            ManagedDoubleRNSElRepresentation::Zero => {
                return None;
            }
        }
    }

    fn apply_linear_operation<F_coeff_bin, F_doublerns_bin, F_coeff_un, F_doublerns_un>(
        &self, 
        lhs: &ManagedDoubleRNSEl<NumberRing, FpTy, A>, 
        rhs: &ManagedDoubleRNSEl<NumberRing, FpTy, A>, 
        f1: F_coeff_bin, 
        f2: F_doublerns_bin, 
        f3: F_coeff_un, 
        f4: F_doublerns_un
    ) -> ManagedDoubleRNSEl<NumberRing, FpTy, A> 
        where F_coeff_bin: FnOnce(&mut CoeffEl<NumberRing, FpTy, A>, &CoeffEl<NumberRing, FpTy, A>),
            F_doublerns_bin: FnOnce(&mut DoubleRNSEl<NumberRing, FpTy, A>, &DoubleRNSEl<NumberRing, FpTy, A>),
            F_coeff_un: FnOnce(&mut CoeffEl<NumberRing, FpTy, A>),
            F_doublerns_un: FnOnce(&mut DoubleRNSEl<NumberRing, FpTy, A>),
    {
        match (&*lhs.internal.borrow(), &*rhs.internal.borrow()) {
            (_, DoubleRNSElInternal::Zero) => self.clone_el(lhs),
            (DoubleRNSElInternal::Zero, DoubleRNSElInternal::DoubleRNS(rhs_doublerns)) => {
                let mut result_fft = self.base.clone_el(rhs_doublerns);
                f4(&mut result_fft);
                return ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::DoubleRNS(result_fft))) };
            },
            (DoubleRNSElInternal::Zero, DoubleRNSElInternal::Coeff(rhs_coeff)) => {
                let mut result_coeff = self.base.clone_el_non_fft(rhs_coeff);
                f3(&mut result_coeff);
                return ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::Coeff(result_coeff))) };
            },
            (DoubleRNSElInternal::Zero, DoubleRNSElInternal::Sum(rhs_coeff, rhs_doublerns)) => {
                let mut result_fft = self.base.clone_el(rhs_doublerns);
                let mut result_coeff = self.base.clone_el_non_fft(rhs_coeff);
                f3(&mut result_coeff);
                f4(&mut result_fft);
                return ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::Sum(result_coeff, result_fft))) };
            },
            (DoubleRNSElInternal::Zero, DoubleRNSElInternal::Both(rhs_coeff, rhs_doublerns)) => {
                let mut result_fft = self.base.clone_el(rhs_doublerns);
                let mut result_coeff = self.base.clone_el_non_fft(rhs_coeff);
                f3(&mut result_coeff);
                f4(&mut result_fft);
                return ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::Both(result_coeff, result_fft))) };
            },
            (DoubleRNSElInternal::Sum(lhs_coeff, lhs_doublerns), DoubleRNSElInternal::Sum(rhs_coeff, rhs_doublerns)) => {
                let mut result_coeff = self.base.clone_el_non_fft(lhs_coeff);
                let mut result_fft = self.base.clone_el(lhs_doublerns);
                f1(&mut result_coeff, rhs_coeff);
                f2(&mut result_fft, rhs_doublerns);
                return ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::Sum(result_coeff, result_fft))) };
            },
            (DoubleRNSElInternal::Sum(lhs_coeff, lhs_doublerns), DoubleRNSElInternal::Coeff(rhs_coeff)) => {
                let mut result_coeff = self.base.clone_el_non_fft(lhs_coeff);
                let result_fft = self.base.clone_el(lhs_doublerns);
                f1(&mut result_coeff, rhs_coeff);
                return ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::Sum(result_coeff, result_fft))) };
            },
            (DoubleRNSElInternal::Sum(lhs_coeff, lhs_doublerns), DoubleRNSElInternal::DoubleRNS(rhs_doublerns)) => {
                let result_coeff = self.base.clone_el_non_fft(lhs_coeff);
                let mut result_fft = self.base.clone_el(lhs_doublerns);
                f2(&mut result_fft, rhs_doublerns);
                return ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::Sum(result_coeff, result_fft))) };
            },
            (DoubleRNSElInternal::Sum(lhs_coeff, lhs_doublerns), DoubleRNSElInternal::Both(rhs_coeff, _)) => {
                let mut result_coeff = self.base.clone_el_non_fft(lhs_coeff);
                let result_fft = self.base.clone_el(lhs_doublerns);
                f1(&mut result_coeff, rhs_coeff);
                return ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::Sum(result_coeff, result_fft))) };
            },
            (DoubleRNSElInternal::Coeff(lhs_coeff), DoubleRNSElInternal::Sum(rhs_coeff, rhs_doublerns)) => {
                let mut result_coeff = self.base.clone_el_non_fft(lhs_coeff);
                let mut result_fft = self.base.clone_el(rhs_doublerns);
                f1(&mut result_coeff, rhs_coeff);
                f4(&mut result_fft);
                return ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::Sum(result_coeff, result_fft))) };
            },
            (DoubleRNSElInternal::Coeff(lhs_coeff), DoubleRNSElInternal::Coeff(rhs_coeff)) => {
                let mut result_coeff = self.base.clone_el_non_fft(lhs_coeff);
                f1(&mut result_coeff, rhs_coeff);
                return ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::Coeff(result_coeff))) };
            },
            (DoubleRNSElInternal::Coeff(lhs_coeff), DoubleRNSElInternal::DoubleRNS(rhs_doublerns)) => {
                let result_coeff = self.base.clone_el_non_fft(lhs_coeff);
                let mut result_fft = self.base.clone_el(rhs_doublerns);
                f4(&mut result_fft);
                return ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::Sum(result_coeff, result_fft))) };
            },
            (DoubleRNSElInternal::Coeff(lhs_coeff), DoubleRNSElInternal::Both(rhs_coeff, _)) => {
                let mut result_coeff = self.base.clone_el_non_fft(lhs_coeff);
                f1(&mut result_coeff, rhs_coeff);
                return ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::Coeff(result_coeff))) };
            },
            (DoubleRNSElInternal::DoubleRNS(lhs_doublerns), DoubleRNSElInternal::Sum(rhs_coeff, rhs_doublerns)) => {
                let mut result_coeff = self.base.clone_el_non_fft(rhs_coeff);
                let mut result_fft = self.base.clone_el(lhs_doublerns);
                f2(&mut result_fft, rhs_doublerns);
                f3(&mut result_coeff);
                return ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::Sum(result_coeff, result_fft))) };
            },
            (DoubleRNSElInternal::DoubleRNS(lhs_doublerns), DoubleRNSElInternal::Coeff(rhs_coeff)) => {
                let mut result_coeff = self.base.clone_el_non_fft(rhs_coeff);
                let result_fft = self.base.clone_el(lhs_doublerns);
                f3(&mut result_coeff);
                return ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::Sum(result_coeff, result_fft))) };
            },
            (DoubleRNSElInternal::DoubleRNS(lhs_doublerns), DoubleRNSElInternal::DoubleRNS(rhs_doublerns)) => {
                let mut result_fft = self.base.clone_el(lhs_doublerns);
                f2(&mut result_fft, rhs_doublerns);
                return ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::DoubleRNS(result_fft))) };
            },
            (DoubleRNSElInternal::DoubleRNS(lhs_doublerns), DoubleRNSElInternal::Both(_, rhs_doublerns)) => {
                let mut result_fft = self.base.clone_el(lhs_doublerns);
                f2(&mut result_fft, rhs_doublerns);
                return ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::DoubleRNS(result_fft))) };
            },
            (DoubleRNSElInternal::Both(lhs_coeff, _), DoubleRNSElInternal::Sum(rhs_coeff, rhs_doublerns)) => {
                let mut result_coeff = self.base.clone_el_non_fft(lhs_coeff);
                let mut result_fft = self.base.clone_el(rhs_doublerns);
                f1(&mut result_coeff, rhs_coeff);
                f4(&mut result_fft);
                return ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::Sum(result_coeff, result_fft))) };
            },
            (DoubleRNSElInternal::Both(lhs_coeff, _), DoubleRNSElInternal::Coeff(rhs_coeff)) => {
                let mut result_coeff = self.base.clone_el_non_fft(lhs_coeff);
                f1(&mut result_coeff, rhs_coeff);
                return ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::Coeff(result_coeff))) };
            },
            (DoubleRNSElInternal::Both(_, lhs_doublerns), DoubleRNSElInternal::DoubleRNS(rhs_doublerns)) => {
                let mut result_fft = self.base.clone_el(lhs_doublerns);
                f2(&mut result_fft, rhs_doublerns);
                return ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::DoubleRNS(result_fft))) };
            },
            (DoubleRNSElInternal::Both(lhs_coeff, lhs_doublerns), DoubleRNSElInternal::Both(rhs_coeff, rhs_doublerns)) => {
                let mut result_coeff = self.base.clone_el_non_fft(lhs_coeff);
                let mut result_fft = self.base.clone_el(lhs_doublerns);
                f1(&mut result_coeff, rhs_coeff);
                f2(&mut result_fft, rhs_doublerns);
                return ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::Both(result_coeff, result_fft))) };
            },
        }
    }
}

pub struct GadgetProductRhsOperand<'a, NumberRing, A> 
    where NumberRing: HENumberRing<Zn>,
        A: Allocator + Clone
{
    base: gadget_product::double_rns::GadgetProductRhsOperand<'a, NumberRing, A>
}

pub struct GadgetProductLhsOperand<'a, NumberRing, A> 
    where NumberRing: HENumberRing<Zn>,
        A: Allocator + Clone
{
    base: gadget_product::double_rns::GadgetProductLhsOperand<'a, NumberRing, A>
}


impl<NumberRing, A> BXVCiphertextRing for ManagedDoubleRNSRingBase<NumberRing, zn_64::Zn, A> 
    where NumberRing: HECyclotomicNumberRing<zn_64::Zn>,
        A: Allocator + Clone
{
    type NumberRing = NumberRing;
    type GadgetProductLhsOperand<'a> = GadgetProductLhsOperand<'a, NumberRing, A>
        where Self: 'a;
    type GadgetProductRhsOperand<'a> = GadgetProductRhsOperand<'a, NumberRing, A>
        where Self: 'a;

    fn number_ring(&self) -> &Self::NumberRing {
        self.base.number_ring()
    }

    fn sample_from_coefficient_distribution<G: FnMut() -> i32>(&self, distribution: G) -> ManagedDoubleRNSEl<NumberRing, zn_64::Zn, A> {
        ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::Coeff(self.base.sample_from_coefficient_distribution(distribution)))) }
    }
    
    fn perform_rns_op_from<Op>(
        &self, 
        from: &Self, 
        el: &ManagedDoubleRNSEl<NumberRing, zn_64::Zn, A>, 
        op: &Op
    ) -> ManagedDoubleRNSEl<NumberRing, zn_64::Zn, A> 
        where NumberRing: HENumberRing<zn_64::Zn>,
            Op: RNSOperation<RingType = zn_64::ZnBase>
    {
        let result = if let Some(value) = from.to_coeff(el) {
            self.base.perform_rns_op_from(&from.base, &*value, op)
        } else {
            self.base.perform_rns_op_from(&from.base, &from.zero, op)
        };
        ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::Coeff(result))) }
    }
    
    fn exact_convert_from_decompring<FpTy2, A2>(
        &self, 
        from: &DecompositionRing<NumberRing, FpTy2, A2>, 
        element: &<DecompositionRingBase<NumberRing, FpTy2, A2> as RingBase>::Element
    ) -> ManagedDoubleRNSEl<NumberRing, zn_64::Zn, A> 
        where NumberRing: HENumberRing<FpTy2>,
            FpTy2: RingStore<Type = zn_64::ZnBase> + Clone,
            A2: Allocator + Clone
    {
        ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::Coeff(self.base.exact_convert_from_decompring(from, element)))) }
    }
    
    fn perform_rns_op_to_decompring<FpTy2, A2, Op>(
        &self, 
        to: &DecompositionRing<NumberRing, FpTy2, A2>, 
        element: &ManagedDoubleRNSEl<NumberRing, zn_64::Zn, A>, 
        op: &Op
    ) -> <DecompositionRingBase<NumberRing, FpTy2, A2> as RingBase>::Element 
        where NumberRing: HENumberRing<FpTy2>,
            FpTy2: RingStore<Type = zn_64::ZnBase> + Clone,
            A2: Allocator + Clone,
            Op: RNSOperation<RingType = zn_64::ZnBase>
    {
        if let Some(value) = self.to_coeff(element) {
            return self.base.perform_rns_op_to_decompring(to, &*value, op);
        } else {
            return self.base.perform_rns_op_to_decompring(to, &self.zero, op);
        }
    }

    fn gadget_product<'b>(&self, lhs: &Self::GadgetProductLhsOperand<'b>, rhs: &Self::GadgetProductRhsOperand<'b>) -> Self::Element {
        match self.base.preferred_output_repr(&lhs.base, &rhs.base) {
            gadget_product::double_rns::ElRepr::Coeff => ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::Coeff(self.base.gadget_product_coeff(&lhs.base, &rhs.base)))) },
            gadget_product::double_rns::ElRepr::NTT => ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::DoubleRNS(self.base.gadget_product_ntt(&lhs.base, &rhs.base)))) }
        }
    }

    fn gadget_product_rhs_empty<'a>(&'a self, digits: usize) -> Self::GadgetProductRhsOperand<'a> {
        GadgetProductRhsOperand {
            base: gadget_product::double_rns::GadgetProductRhsOperand::create_empty::<false>(&self.base, digits)
        }
    }

    fn to_gadget_product_lhs<'a>(&'a self, el: Self::Element, digits: usize) -> Self::GadgetProductLhsOperand<'a> {
        if let Some(nonzero) = self.to_coeff(&el) {
            GadgetProductLhsOperand {
                base: gadget_product::double_rns::GadgetProductLhsOperand::create_from_element(&self.base, digits, self.base.clone_el_non_fft(&*nonzero))
            }
        } else {
            GadgetProductLhsOperand {
                base: gadget_product::double_rns::GadgetProductLhsOperand::create_from_element(&self.base, digits, self.base.clone_el_non_fft(&self.zero))
            }
        }
    }

    fn gadget_vector<'a, 'b>(&'a self, rhs_operand: &'a Self::GadgetProductRhsOperand<'b>) -> &'a [std::ops::Range<usize>] {
        rhs_operand.base.gadget_vector()
    }

    fn set_rns_factor<'b>(&self, rhs_operand: &mut Self::GadgetProductRhsOperand<'b>, i: usize, el: Self::Element) {
        if let Some(nonzero) = self.to_coeff(&el) {
            rhs_operand.base.set_rns_factor(i, self.base.clone_el_non_fft(&*nonzero))
        } else {
            rhs_operand.base.set_rns_factor(i, self.base.clone_el_non_fft(&self.zero))
        }
    }

    fn apply_galois_action_many_gadget_product_operand<'a>(&'a self, x: &Self::GadgetProductLhsOperand<'a>, gs: &[zn_64::ZnEl]) -> Vec<Self::GadgetProductLhsOperand<'a>> {
        gs.iter().map(|g| GadgetProductLhsOperand {
            base: x.base.apply_galois_action(&self.base, *g)
        }).collect()
    }
}

impl<NumberRing, FpTy, A> CyclotomicRing for ManagedDoubleRNSRingBase<NumberRing, FpTy, A> 
    where NumberRing: HECyclotomicNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    fn n(&self) -> u64 {
        self.base.n()
    }

    fn apply_galois_action(&self, el: &Self::Element, g: zn_64::ZnEl) -> Self::Element {
        let result = if let Some(value) = self.to_doublerns(el) {
            self.base.apply_galois_action(&*value, g)
        } else {
            return self.zero();
        };
        ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::DoubleRNS(result))) }
    }
}

impl<NumberRing, FpTy, A> PartialEq for ManagedDoubleRNSRingBase<NumberRing, FpTy, A>
    where NumberRing: HENumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    fn eq(&self, other: &Self) -> bool {
        self.base == other.base
    }
}

impl<NumberRing, FpTy, A> RingBase for ManagedDoubleRNSRingBase<NumberRing, FpTy, A>
    where NumberRing: HENumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    type Element = ManagedDoubleRNSEl<NumberRing, FpTy, A>;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        ManagedDoubleRNSEl { internal: val.internal.clone() }
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        match (self.get_repr(lhs), self.get_repr(rhs)) {
            (ManagedDoubleRNSElRepresentation::Zero, _) | (_, ManagedDoubleRNSElRepresentation::Zero) => self.is_zero(lhs),
            (ManagedDoubleRNSElRepresentation::Coeff, _) | (_, ManagedDoubleRNSElRepresentation::Coeff) => self.base.eq_el_non_fft(&*self.to_coeff(lhs).unwrap(), &*self.to_coeff(rhs).unwrap()),
            _ => self.base.eq_el(&*self.to_doublerns(lhs).unwrap(), &*self.to_doublerns(rhs).unwrap())
        }
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        match self.get_repr(value) {
            ManagedDoubleRNSElRepresentation::Zero => true,
            ManagedDoubleRNSElRepresentation::Sum | ManagedDoubleRNSElRepresentation::Coeff | ManagedDoubleRNSElRepresentation::Both => self.base.eq_el_non_fft(&*self.to_coeff(value).unwrap(), &self.zero),
            ManagedDoubleRNSElRepresentation::DoubleRNS => self.base.is_zero(&*self.to_doublerns(value).unwrap())
        }
    }

    fn zero(&self) -> Self::Element {
        ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::Zero)) }
    }
    
    fn dbg_within<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>, env: EnvBindingStrength) -> std::fmt::Result {
        if let Some(nonzero) = self.to_doublerns(value) {
            self.base.dbg_within(&*nonzero, out, env)
        } else {
            write!(out, "0")
        }
    }

    fn square(&self, value: &mut Self::Element) {
        let mut result = if let Some(nonzero) = self.to_doublerns(value) {
            self.base.clone_el(&*nonzero)
        } else {
            return
        };
        self.base.square(&mut result);
        value.internal = Rc::new(RefCell::new(DoubleRNSElInternal::DoubleRNS(result)));
    }

    fn negate(&self, value: Self::Element) -> Self::Element {
        self.apply_linear_operation(
            &self.zero(), 
            &value,
            |_, _| unreachable!(),
            |_, _| unreachable!(),
            |a| self.base.negate_inplace_non_fft(a),
            |a| self.base.negate_inplace(a)
        )
    }
    
    fn mul_int_ref(&self, lhs: &Self::Element, rhs: i32) -> Self::Element {
        self.apply_linear_operation(
            &self.zero(), 
            lhs,
            |_, _| unreachable!(),
            |_, _| unreachable!(),
            |a| self.base.mul_scalar_assign_non_fft(a, &self.base_ring().int_hom().map(rhs)),
            |a| self.base.mul_assign_int(a, rhs)
        )
    }

    fn add_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        self.apply_linear_operation(
            lhs, 
            rhs,
            |a, b| self.base.add_assign_non_fft(a, b),
            |a, b| self.base.add_assign_ref(a, b),
            |a| {},
            |a| {}
        )
    }

    fn sub_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        self.apply_linear_operation(
            lhs,
            rhs,
            |a, b| self.base.sub_assign_non_fft(a, b),
            |a, b| self.base.sub_assign_ref(a, b),
            |a| self.base.negate_inplace_non_fft(a),
            |a| self.base.negate_inplace(a)
        )
    }

    fn mul_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        let result = if let (Some(lhs), Some(rhs)) = (self.to_doublerns(lhs), self.to_doublerns(rhs)) {
            self.base.mul_ref(&*lhs, &*rhs)
        } else {
            return self.zero();
        };
        return ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::DoubleRNS(result))) };
    }

    fn pow_gen<R: IntegerRingStore>(&self, x: Self::Element, power: &El<R>, integers: R) -> Self::Element 
        where R::Type: IntegerRing
    {
        let result = if let Some(nonzero) = self.to_doublerns(&x) {
            self.base.pow_gen(self.base.clone_el(&*nonzero), power, integers)
        } else {
            return self.zero();
        };
        return ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::DoubleRNS(result))) };
    }

    fn characteristic<I: IntegerRingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        self.base_ring().characteristic(ZZ)
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        lhs.internal = self.add_ref(lhs, rhs).internal;
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        lhs.internal = self.add_ref(lhs, &rhs).internal;
    }

    fn sub_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        lhs.internal = self.sub_ref(lhs, rhs).internal;
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        lhs.internal = self.negate(self.clone_el(lhs)).internal;
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        lhs.internal = self.mul_ref(lhs, &rhs).internal;
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        lhs.internal = self.mul_ref(lhs, rhs).internal;
    }

    fn is_commutative(&self) -> bool { true }
    fn is_noetherian(&self) -> bool { true }
    fn is_approximate(&self) -> bool { false }

    fn from_int(&self, value: i32) -> Self::Element {
        self.from(self.base_ring().get_ring().from_int(value))
    }

    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        self.dbg_within(value, out, EnvBindingStrength::Weakest)
    }

    fn sub_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        lhs.internal = self.sub_ref(lhs, &rhs).internal;
    }

    fn mul_assign_int(&self, lhs: &mut Self::Element, rhs: i32) {
        lhs.internal = self.mul_int(self.clone_el(lhs), rhs).internal;
    }

    fn mul_int(&self, lhs: Self::Element, rhs: i32) -> Self::Element {
        self.mul_int_ref(&lhs, rhs)
    }

    fn sub_self_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        lhs.internal = self.sub_ref(&rhs, lhs).internal;
    }

    fn sub_self_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        lhs.internal = self.sub_ref(rhs, lhs).internal;
    }

    fn add_ref_fst(&self, lhs: &Self::Element, rhs: Self::Element) -> Self::Element {
        self.add_ref(lhs, &rhs)
    }

    fn add_ref_snd(&self, lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        self.add_ref(&lhs, rhs)
    }

    fn add(&self, lhs: Self::Element, rhs: Self::Element) -> Self::Element {
        self.add_ref(&lhs, &rhs)
    }

    fn sub_ref_fst(&self, lhs: &Self::Element, rhs: Self::Element) -> Self::Element {
        self.sub_ref(lhs, &rhs)
    }

    fn sub_ref_snd(&self, lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        self.sub_ref(&lhs, rhs)
    }

    fn sub(&self, lhs: Self::Element, rhs: Self::Element) -> Self::Element {
        self.sub_ref(&lhs, &rhs)
    }

    fn mul_ref_fst(&self, lhs: &Self::Element, rhs: Self::Element) -> Self::Element {
        self.mul_ref(lhs, &rhs)
    }

    fn mul_ref_snd(&self, lhs: Self::Element, rhs: &Self::Element) -> Self::Element {
        self.mul_ref(&lhs, rhs)
    }

    fn mul(&self, lhs: Self::Element, rhs: Self::Element) -> Self::Element {
        self.mul_ref(&lhs, &rhs)
    }
}

impl<NumberRing, FpTy, A> RingExtension for ManagedDoubleRNSRingBase<NumberRing, FpTy, A>
    where NumberRing: HENumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    type BaseRing = <DoubleRNSRingBase<NumberRing, FpTy, A> as RingExtension>::BaseRing;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing {
        self.base.base_ring()
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        let result_fft = self.base.from(self.base_ring().clone_el(&x));
        let result_coeff = self.base.from_non_fft(x);
        return ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::Both(result_coeff, result_fft))) };
    }
}

impl<NumberRing, FpTy, A> FreeAlgebra for ManagedDoubleRNSRingBase<NumberRing, FpTy, A>
    where NumberRing: HENumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    type VectorRepresentation<'a> = DoubleRNSRingBaseElVectorRepresentation<'a, NumberRing, FpTy, A> 
        where Self: 'a;

    fn canonical_gen(&self) -> Self::Element {
        let result = self.base.canonical_gen();
        return ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::DoubleRNS(result))) };
    }

    fn rank(&self) -> usize {
        self.base.rank()
    }

    fn wrt_canonical_basis<'a>(&'a self, el: &'a Self::Element) -> Self::VectorRepresentation<'a> {
        if let Some(result) = self.to_coeff(el) {
            self.base.wrt_canonical_basis_non_fft(self.base.clone_el_non_fft(&result))
        } else {
            self.base.wrt_canonical_basis_non_fft(self.base.clone_el_non_fft(&self.zero))
        }
    }

    fn from_canonical_basis<V>(&self, vec: V) -> Self::Element
        where V: IntoIterator<Item = El<Self::BaseRing>>,
            V::IntoIter: DoubleEndedIterator
    {
        let result = self.base.from_canonical_basis(vec);
        return ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::DoubleRNS(result))) };
    }
}

impl<NumberRing, FpTy, A> FiniteRing for ManagedDoubleRNSRingBase<NumberRing, FpTy, A>
    where NumberRing: HENumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    type ElementsIter<'a> = std::iter::Map<<DoubleRNSRingBase<NumberRing, FpTy, A> as FiniteRing>::ElementsIter<'a>, fn(DoubleRNSEl<NumberRing, FpTy, A>) -> ManagedDoubleRNSEl<NumberRing, FpTy, A>>
        where Self: 'a;

    fn elements<'a>(&'a self) -> Self::ElementsIter<'a> {
        fn from_doublerns<NumberRing, FpTy, A>(x: DoubleRNSEl<NumberRing, FpTy, A>) -> ManagedDoubleRNSEl<NumberRing, FpTy, A>
            where NumberRing: HENumberRing<FpTy>,
                FpTy: RingStore + Clone,
                FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
                A: Allocator + Clone
        {
            ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::DoubleRNS(x))) }
        }
        self.base.elements().map(from_doublerns)
    }

    fn random_element<G: FnMut() -> u64>(&self, rng: G) -> <Self as RingBase>::Element {
        ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(DoubleRNSElInternal::DoubleRNS(self.base.random_element(rng)))) }
    }

    fn size<I: IntegerRingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        self.base.size(ZZ)
    }
}

#[test]
fn test_ring_axioms() {
    let rns_base = zn_rns::Zn::new(vec![Zn::new(17), Zn::new(97)], BigIntRing::RING);
    let ring = ManagedDoubleRNSRingBase::new(Pow2CyclotomicDecomposableNumberRing::new(16), rns_base);
    
    let base_ring = ring.base_ring();
    let elements = vec![
        ring.zero(),
        ring.one(),
        ring.neg_one(),
        ring.int_hom().map(17),
        ring.int_hom().map(97),
        ring.canonical_gen(),
        ring.pow(ring.canonical_gen(), 15),
        ring.int_hom().mul_map(ring.canonical_gen(), 17),
        ring.int_hom().mul_map(ring.pow(ring.canonical_gen(), 15), 17),
        ring.add(ring.canonical_gen(), ring.one())
    ];

    feanor_math::ring::generic_tests::test_ring_axioms(&ring, elements.iter().map(|x| ring.clone_el(x)));
}

#[test]
fn test_add_result_independent_of_repr() {
    let rns_base = zn_rns::Zn::new(vec![Zn::new(17), Zn::new(97)], BigIntRing::RING);
    let ring = ManagedDoubleRNSRingBase::new(Pow2CyclotomicDecomposableNumberRing::new(4), rns_base);
    let base = &ring.get_ring().base;
    let reprs_of_11: [Box<dyn Fn() -> DoubleRNSElInternal<_, _, _>>; 4] = [
        Box::new(|| DoubleRNSElInternal::Coeff(base.from_non_fft(base.base_ring().int_hom().map(11)))),
        Box::new(|| DoubleRNSElInternal::DoubleRNS(base.from_int(11))),
        Box::new(|| DoubleRNSElInternal::Sum(base.from_non_fft(base.base_ring().int_hom().map(1)), base.from_int(10))),
        Box::new(|| DoubleRNSElInternal::Both(base.from_non_fft(base.base_ring().int_hom().map(11)), base.from_int(11))),
    ];
    let reprs_of_102: [Box<dyn Fn() -> DoubleRNSElInternal<_, _, _>>; 4] = [
       Box::new(|| DoubleRNSElInternal::Coeff(base.from_non_fft(base.base_ring().int_hom().map(102)))),
       Box::new(|| DoubleRNSElInternal::DoubleRNS(base.from_int(102))),
       Box::new(|| DoubleRNSElInternal::Sum(base.from_non_fft(base.base_ring().int_hom().map(2)), base.from_int(100))),
       Box::new(|| DoubleRNSElInternal::Both(base.from_non_fft(base.base_ring().int_hom().map(102)), base.from_int(102))),
    ];
    for a in &reprs_of_11 {
        for b in &reprs_of_102 {
            let x = ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(a())) };
            assert_el_eq!(RingRef::new(base), base.from_int(22), ring.get_ring().to_doublerns(&ring.add_ref(&x, &x)).unwrap());

            let x = ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(a())) };
            let y = ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(b())) };
            assert_el_eq!(RingRef::new(base), base.from_int(113), ring.get_ring().to_doublerns(&ring.add_ref(&x, &y)).unwrap());

            let x = ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(a())) };
            let y = ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(b())) };
            assert!(base.eq_el_non_fft(&base.from_non_fft(base.base_ring().int_hom().map(113)), &*ring.get_ring().to_coeff(&ring.add_ref(&x, &y)).unwrap()));

            let x = ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(a())) };
            let y = ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(b())) };
            assert_el_eq!(RingRef::new(base), base.from_int(-91), ring.get_ring().to_doublerns(&ring.sub_ref(&x, &y)).unwrap());

            let x = ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(a())) };
            let y = ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(b())) };
            assert!(base.eq_el_non_fft(&base.from_non_fft(base.base_ring().int_hom().map(-91)), &*ring.get_ring().to_coeff(&ring.sub_ref(&x, &y)).unwrap()));

            let x = ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(a())) };
            assert_el_eq!(RingRef::new(base), base.from_int(121), ring.get_ring().to_doublerns(&ring.mul_ref(&x, &x)).unwrap());

            let x = ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(a())) };
            let y = ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(b())) };
            assert_el_eq!(RingRef::new(base), base.from_int(1122), ring.get_ring().to_doublerns(&ring.mul_ref(&x, &y)).unwrap());

            let x = ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(a())) };
            let y = ManagedDoubleRNSEl { internal: Rc::new(RefCell::new(b())) };
            assert!(base.eq_el_non_fft(&base.from_non_fft(base.base_ring().int_hom().map(1122)), &*ring.get_ring().to_coeff(&ring.mul_ref(&x, &y)).unwrap()));
        }
    }
}