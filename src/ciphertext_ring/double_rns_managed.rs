use std::alloc::Allocator;
use std::alloc::Global;
use std::cell::OnceCell;
use std::cell::Ref;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::AtomicU64;

use feanor_math::algorithms::convolution::ConvolutionAlgorithm;
use feanor_math::algorithms::convolution::PreparedConvolutionAlgorithm;
use feanor_math::assert_el_eq;
use feanor_math::delegate::DelegateRing;
use feanor_math::homomorphism::*;
use feanor_math::integer::*;
use feanor_math::matrix::AsFirstElement;
use feanor_math::matrix::AsPointerToSlice;
use feanor_math::matrix::Submatrix;
use feanor_math::ring::*;
use feanor_math::rings::extension::*;
use feanor_math::rings::finite::FiniteRing;
use feanor_math::rings::zn::*;
use feanor_math::seq::VectorView;
use feanor_math::specialization::FiniteRingOperation;
use feanor_math::specialization::FiniteRingSpecializable;
use zn_64::Zn;
use zn_64::ZnBase;
use zn_64::ZnEl;

use crate::cyclotomic::CyclotomicGaloisGroupEl;
use crate::cyclotomic::CyclotomicRing;
use crate::number_ring::pow2_cyclotomic::Pow2CyclotomicNumberRing;
use crate::number_ring::HECyclotomicNumberRing;
use crate::number_ring::HENumberRing;
use crate::DefaultConvolution;

use super::double_rns_ring;
use super::double_rns_ring::*;
use super::single_rns_ring::*;
use super::BGFVCiphertextRing;

///
/// Like [`DoubleRNSRing`] but stores element in whatever representation they
/// currently are available, and automatically switches representation when 
/// necessary.
/// 
pub struct ManagedDoubleRNSRingBase<NumberRing, A = Global> 
    where NumberRing: HENumberRing,
        A: Allocator + Clone
{
    base: DoubleRNSRingBase<NumberRing, A>,
    zero: SmallBasisEl<NumberRing, A>
}

pub type ManagedDoubleRNSRing<NumberRing, A = Global> = RingValue<ManagedDoubleRNSRingBase<NumberRing, A>>;

impl<NumberRing> ManagedDoubleRNSRingBase<NumberRing, Global> 
    where NumberRing: HENumberRing,
{
    pub fn new(number_ring: NumberRing, rns_base: zn_rns::Zn<zn_64::Zn, BigIntRing>) -> RingValue<Self> {
        Self::new_with(number_ring, rns_base, Global)
    }
}

#[derive(Clone, Copy)]
enum ManagedDoubleRNSElRepresentation {
    Sum,
    SmallBasis,
    DoubleRNS,
    Both,
    Zero
}

struct DoubleRNSElInternal<NumberRing, A = Global> 
    where NumberRing: HENumberRing,
        A: Allocator + Clone
{
    representation: RefCell<ManagedDoubleRNSElRepresentation>,
    small_basis_repr: OnceCell<SmallBasisEl<NumberRing, A>>,
    double_rns_repr_or_part: RefCell<Option<DoubleRNSEl<NumberRing, A>>>,
    small_basis_part: RefCell<Option<SmallBasisEl<NumberRing, A>>>
}

impl<NumberRing, A> DoubleRNSElInternal<NumberRing, A> 
    where NumberRing: HENumberRing,
        A: Allocator + Clone
{
    fn get_repr(&self) -> ManagedDoubleRNSElRepresentation {
        *self.representation.borrow()
    }
}

pub struct ManagedDoubleRNSEl<NumberRing, A = Global> 
    where NumberRing: HENumberRing,
        A: Allocator + Clone
{
    internal: Rc<DoubleRNSElInternal<NumberRing, A>>
}

impl<NumberRing, A> ManagedDoubleRNSRingBase<NumberRing, A>
    where NumberRing: HENumberRing,
        A: Allocator + Clone
{
    pub fn new_with(number_ring: NumberRing, rns_base: zn_rns::Zn<zn_64::Zn, BigIntRing>, allocator: A) -> RingValue<ManagedDoubleRNSRingBase<NumberRing, A>> {
        let result = DoubleRNSRingBase::new_with(number_ring, rns_base, allocator);
        let zero = result.get_ring().zero_non_fft();
        ManagedDoubleRNSRing::from(ManagedDoubleRNSRingBase { base: result.into(), zero: zero })
    }

    ///
    /// Puts the internal value into coefficient representation.
    /// 
    /// This is seldomly sensible in an application, as the representation is automatically
    /// managed as deemed most efficient. However, this makes getting reliable benchmarks 
    /// difficult, since one never knows if a certain time included representation-switching 
    /// or not. Thus, in such benchmarks, every operation should be followed by 
    /// `force_coeff_repr()`.
    /// 
    pub fn force_small_basis_repr(&self, value: &ManagedDoubleRNSEl<NumberRing, A>) {
        self.to_small_basis(value);
        *value.internal.representation.borrow_mut() = ManagedDoubleRNSElRepresentation::SmallBasis;
        *value.internal.small_basis_part.borrow_mut() = None;
        *value.internal.double_rns_repr_or_part.borrow_mut() = None;
    }

    fn to_small_basis<'a>(&self, value: &'a ManagedDoubleRNSEl<NumberRing, A>) -> Option<&'a SmallBasisEl<NumberRing, A>> {
        match value.internal.get_repr() {
            ManagedDoubleRNSElRepresentation::Sum => {
                let mut result = std::mem::replace(&mut *value.internal.small_basis_part.borrow_mut(), None).unwrap();
                let double_rns_part = std::mem::replace(&mut *value.internal.double_rns_repr_or_part.borrow_mut(), None).unwrap();
                self.base.add_assign_non_fft(&mut result, &self.base.undo_fft(double_rns_part));
                value.internal.small_basis_repr.set(result).ok().unwrap();
                *value.internal.representation.borrow_mut() = ManagedDoubleRNSElRepresentation::SmallBasis;
                return Some(value.internal.small_basis_repr.get().unwrap());
            },
            ManagedDoubleRNSElRepresentation::SmallBasis | ManagedDoubleRNSElRepresentation::Both => {
                return Some(value.internal.small_basis_repr.get().unwrap());
            },
            ManagedDoubleRNSElRepresentation::DoubleRNS => {
                let result = self.base.undo_fft(self.base.clone_el(value.internal.double_rns_repr_or_part.borrow().as_ref().unwrap()));
                value.internal.small_basis_repr.set(result).ok().unwrap();
                *value.internal.representation.borrow_mut() = ManagedDoubleRNSElRepresentation::Both;
                return Some(value.internal.small_basis_repr.get().unwrap());
            },
            ManagedDoubleRNSElRepresentation::Zero => {
                return None;
            }
        }
    }

    fn to_doublerns<'a>(&self, value: &'a ManagedDoubleRNSEl<NumberRing, A>) -> Option<Ref<'a, DoubleRNSEl<NumberRing, A>>> {
        match value.internal.get_repr() {
            ManagedDoubleRNSElRepresentation::Sum => {
                let coeff_part = std::mem::replace(&mut *value.internal.small_basis_part.borrow_mut(), None).unwrap();
                let mut result = std::mem::replace(&mut *value.internal.double_rns_repr_or_part.borrow_mut(), None).unwrap();
                self.base.add_assign(&mut result, self.base.do_fft(coeff_part));
                *value.internal.double_rns_repr_or_part.borrow_mut() = Some(result);
                *value.internal.representation.borrow_mut() = ManagedDoubleRNSElRepresentation::DoubleRNS;
                return Some(Ref::map(value.internal.double_rns_repr_or_part.borrow(), |x| x.as_ref().unwrap()));
            },
            ManagedDoubleRNSElRepresentation::DoubleRNS | ManagedDoubleRNSElRepresentation::Both => {
                return Some(Ref::map(value.internal.double_rns_repr_or_part.borrow(), |x| x.as_ref().unwrap()));
            },
            ManagedDoubleRNSElRepresentation::SmallBasis => {
                let result = self.base.do_fft(self.base.clone_el_non_fft(value.internal.small_basis_repr.get().unwrap()));
                *value.internal.double_rns_repr_or_part.borrow_mut() = Some(result);
                *value.internal.representation.borrow_mut() = ManagedDoubleRNSElRepresentation::Both;
                return Some(Ref::map(value.internal.double_rns_repr_or_part.borrow(), |x| x.as_ref().unwrap()));
            },
            ManagedDoubleRNSElRepresentation::Zero => {
                return None;
            }
        }
    }

    fn apply_linear_operation<F_coeff_bin, F_doublerns_bin, F_coeff_un, F_doublerns_un>(
        &self, 
        lhs: &ManagedDoubleRNSEl<NumberRing, A>, 
        rhs: &ManagedDoubleRNSEl<NumberRing, A>, 
        f1: F_coeff_bin, 
        f2: F_doublerns_bin, 
        f3: F_coeff_un, 
        f4: F_doublerns_un
    ) -> ManagedDoubleRNSEl<NumberRing, A> 
        where F_coeff_bin: FnOnce(&mut SmallBasisEl<NumberRing, A>, &SmallBasisEl<NumberRing, A>),
            F_doublerns_bin: FnOnce(&mut DoubleRNSEl<NumberRing, A>, &DoubleRNSEl<NumberRing, A>),
            F_coeff_un: FnOnce(&mut SmallBasisEl<NumberRing, A>),
            F_doublerns_un: FnOnce(&mut DoubleRNSEl<NumberRing, A>),
    {
        let sum = |x1: SmallBasisEl<NumberRing, A>, x2: DoubleRNSEl<NumberRing, A>| ManagedDoubleRNSEl { internal: Rc::new(DoubleRNSElInternal {
            double_rns_repr_or_part: RefCell::new(Some(x2)),
            representation: RefCell::new(ManagedDoubleRNSElRepresentation::Sum),
            small_basis_repr: OnceCell::new(),
            small_basis_part: RefCell::new(Some(x1))
        }) };
        let double_rns = |x: DoubleRNSEl<NumberRing, A>| ManagedDoubleRNSEl { internal: Rc::new(DoubleRNSElInternal {
            double_rns_repr_or_part: RefCell::new(Some(x)),
            representation: RefCell::new(ManagedDoubleRNSElRepresentation::DoubleRNS),
            small_basis_repr: OnceCell::new(),
            small_basis_part: RefCell::new(None)
        }) };
        let small_basis = |x: SmallBasisEl<NumberRing, A>| ManagedDoubleRNSEl { internal: Rc::new(DoubleRNSElInternal {
            double_rns_repr_or_part: RefCell::new(None),
            representation: RefCell::new(ManagedDoubleRNSElRepresentation::SmallBasis),
            small_basis_repr: {
                let result = OnceCell::new();
                result.set(x).ok().unwrap();
                result
            },
            small_basis_part: RefCell::new(None)
        }) };
        let both = |x1: SmallBasisEl<NumberRing, A>, x2: DoubleRNSEl<NumberRing, A>| ManagedDoubleRNSEl { internal: Rc::new(DoubleRNSElInternal {
            double_rns_repr_or_part: RefCell::new(Some(x2)),
            representation: RefCell::new(ManagedDoubleRNSElRepresentation::Both),
            small_basis_repr: {
                let result = OnceCell::new();
                result.set(x1).ok().unwrap();
                result
            },
            small_basis_part: RefCell::new(None)
        }) };

        match (lhs.internal.get_repr(), rhs.internal.get_repr()) {
            (_, ManagedDoubleRNSElRepresentation::Zero) => self.clone_el(lhs),
            (ManagedDoubleRNSElRepresentation::Zero, ManagedDoubleRNSElRepresentation::DoubleRNS) => {
                let mut result_fft = self.base.clone_el(rhs.internal.double_rns_repr_or_part.borrow().as_ref().unwrap());
                f4(&mut result_fft);
                return double_rns(result_fft);
            },
            (ManagedDoubleRNSElRepresentation::Zero, ManagedDoubleRNSElRepresentation::SmallBasis) => {
                let mut result_coeff = self.base.clone_el_non_fft(rhs.internal.small_basis_repr.get().unwrap());
                f3(&mut result_coeff);
                return small_basis(result_coeff);
            },
            (ManagedDoubleRNSElRepresentation::Zero, ManagedDoubleRNSElRepresentation::Sum) => {
                let mut result_fft = self.base.clone_el(rhs.internal.double_rns_repr_or_part.borrow().as_ref().unwrap());
                let mut result_coeff = self.base.clone_el_non_fft(rhs.internal.small_basis_part.borrow().as_ref().unwrap());
                f3(&mut result_coeff);
                f4(&mut result_fft);
                return sum(result_coeff, result_fft);
            },
            (ManagedDoubleRNSElRepresentation::Zero, ManagedDoubleRNSElRepresentation::Both) => {
                let mut result_fft = self.base.clone_el(rhs.internal.double_rns_repr_or_part.borrow().as_ref().unwrap());
                let mut result_coeff = self.base.clone_el_non_fft(rhs.internal.small_basis_repr.get().unwrap());
                f3(&mut result_coeff);
                f4(&mut result_fft);
                return both(result_coeff, result_fft);
            },
            (ManagedDoubleRNSElRepresentation::Sum, ManagedDoubleRNSElRepresentation::Sum) => {
                let mut result_fft = self.base.clone_el(lhs.internal.double_rns_repr_or_part.borrow().as_ref().unwrap());
                let mut result_coeff = self.base.clone_el_non_fft(lhs.internal.small_basis_part.borrow().as_ref().unwrap());
                f1(&mut result_coeff, rhs.internal.small_basis_part.borrow().as_ref().unwrap());
                f2(&mut result_fft, rhs.internal.double_rns_repr_or_part.borrow().as_ref().unwrap());
                return sum(result_coeff, result_fft);
            },
            (ManagedDoubleRNSElRepresentation::Sum, ManagedDoubleRNSElRepresentation::SmallBasis) => {
                let mut result_coeff = self.base.clone_el_non_fft(lhs.internal.small_basis_part.borrow().as_ref().unwrap());
                let result_fft = self.base.clone_el(lhs.internal.double_rns_repr_or_part.borrow().as_ref().unwrap());
                f1(&mut result_coeff, rhs.internal.small_basis_repr.get().unwrap());
                return sum(result_coeff, result_fft);
            },
            (ManagedDoubleRNSElRepresentation::Sum, ManagedDoubleRNSElRepresentation::DoubleRNS) => {
                let result_coeff = self.base.clone_el_non_fft(lhs.internal.small_basis_part.borrow().as_ref().unwrap());
                let mut result_fft = self.base.clone_el(lhs.internal.double_rns_repr_or_part.borrow().as_ref().unwrap());
                f2(&mut result_fft, rhs.internal.double_rns_repr_or_part.borrow().as_ref().unwrap());
                return sum(result_coeff, result_fft);
            },
            (ManagedDoubleRNSElRepresentation::Sum, ManagedDoubleRNSElRepresentation::Both) => {
                let mut result_coeff = self.base.clone_el_non_fft(lhs.internal.small_basis_part.borrow().as_ref().unwrap());
                let result_fft = self.base.clone_el(lhs.internal.double_rns_repr_or_part.borrow().as_ref().unwrap());
                f1(&mut result_coeff, rhs.internal.small_basis_repr.get().unwrap());
                return sum(result_coeff, result_fft);
            },
            (ManagedDoubleRNSElRepresentation::SmallBasis, ManagedDoubleRNSElRepresentation::Sum) => {
                let mut result_fft = self.base.clone_el(rhs.internal.double_rns_repr_or_part.borrow().as_ref().unwrap());
                let mut result_coeff = self.base.clone_el_non_fft(lhs.internal.small_basis_repr.get().unwrap());
                f1(&mut result_coeff, rhs.internal.small_basis_part.borrow().as_ref().unwrap());
                f4(&mut result_fft);
                return sum(result_coeff, result_fft);
            },
            (ManagedDoubleRNSElRepresentation::SmallBasis, ManagedDoubleRNSElRepresentation::SmallBasis) | 
                (ManagedDoubleRNSElRepresentation::SmallBasis, ManagedDoubleRNSElRepresentation::Both) | 
                (ManagedDoubleRNSElRepresentation::Both, ManagedDoubleRNSElRepresentation::SmallBasis) => 
            {
                let mut result_coeff = self.base.clone_el_non_fft(lhs.internal.small_basis_repr.get().unwrap());
                f1(&mut result_coeff, rhs.internal.small_basis_repr.get().unwrap());
                return small_basis(result_coeff);
            },
            (ManagedDoubleRNSElRepresentation::SmallBasis, ManagedDoubleRNSElRepresentation::DoubleRNS) => {
                let result_coeff = self.base.clone_el_non_fft(lhs.internal.small_basis_repr.get().unwrap());
                let mut result_fft = self.base.clone_el(rhs.internal.double_rns_repr_or_part.borrow().as_ref().unwrap());
                f4(&mut result_fft);
                return sum(result_coeff, result_fft);
            },
            (ManagedDoubleRNSElRepresentation::DoubleRNS, ManagedDoubleRNSElRepresentation::Sum) => {
                let mut result_fft = self.base.clone_el(lhs.internal.double_rns_repr_or_part.borrow().as_ref().unwrap());
                let mut result_coeff = self.base.clone_el_non_fft(rhs.internal.small_basis_part.borrow().as_ref().unwrap());
                f2(&mut result_fft, rhs.internal.double_rns_repr_or_part.borrow().as_ref().unwrap());
                f3(&mut result_coeff);
                return sum(result_coeff, result_fft);
            },
            (ManagedDoubleRNSElRepresentation::DoubleRNS, ManagedDoubleRNSElRepresentation::SmallBasis) => {
                let mut result_coeff = self.base.clone_el_non_fft(rhs.internal.small_basis_repr.get().unwrap());
                let result_fft = self.base.clone_el(lhs.internal.double_rns_repr_or_part.borrow().as_ref().unwrap());
                f3(&mut result_coeff);
                return sum(result_coeff, result_fft);
            },
            (ManagedDoubleRNSElRepresentation::DoubleRNS, ManagedDoubleRNSElRepresentation::DoubleRNS) | 
                (ManagedDoubleRNSElRepresentation::DoubleRNS, ManagedDoubleRNSElRepresentation::Both) | 
                (ManagedDoubleRNSElRepresentation::Both, ManagedDoubleRNSElRepresentation::DoubleRNS) =>
            {
                let mut result_fft = self.base.clone_el(lhs.internal.double_rns_repr_or_part.borrow().as_ref().unwrap());
                f2(&mut result_fft, rhs.internal.double_rns_repr_or_part.borrow().as_ref().unwrap());
                return double_rns(result_fft);
            },
            (ManagedDoubleRNSElRepresentation::Both, ManagedDoubleRNSElRepresentation::Sum) => {
                let mut result_fft = self.base.clone_el(rhs.internal.double_rns_repr_or_part.borrow().as_ref().unwrap());
                let mut result_coeff = self.base.clone_el_non_fft(lhs.internal.small_basis_repr.get().unwrap());
                f1(&mut result_coeff, rhs.internal.small_basis_part.borrow().as_ref().unwrap());
                f4(&mut result_fft);
                return sum(result_coeff, result_fft);
            },
            (ManagedDoubleRNSElRepresentation::Both, ManagedDoubleRNSElRepresentation::Both) => {
                let mut result_fft = self.base.clone_el(lhs.internal.double_rns_repr_or_part.borrow().as_ref().unwrap());
                let mut result_coeff = self.base.clone_el_non_fft(lhs.internal.small_basis_repr.get().unwrap());
                f1(&mut result_coeff, rhs.internal.small_basis_repr.get().unwrap());
                f2(&mut result_fft, rhs.internal.double_rns_repr_or_part.borrow().as_ref().unwrap());
                return both(result_coeff, result_fft);
            },
        }
    }
}

impl<NumberRing, A> BGFVCiphertextRing for ManagedDoubleRNSRingBase<NumberRing, A> 
    where NumberRing: HECyclotomicNumberRing,
        A: Allocator + Clone
{
    type NumberRing = NumberRing;
    type PreparedMultiplicant = Self::Element;

    fn drop_rns_factor(&self, from: &Self, dropped_rns_factors: &[usize], value: Self::Element) -> Self::Element {
        match value.internal.get_repr() {
            ManagedDoubleRNSElRepresentation::Zero => self.zero(),
            ManagedDoubleRNSElRepresentation::Sum => ManagedDoubleRNSEl { internal: Rc::new(DoubleRNSElInternal {
                double_rns_repr_or_part: RefCell::new(Some(self.base.drop_rns_factor(&from.base, dropped_rns_factors, value.internal.double_rns_repr_or_part.borrow().as_ref().unwrap()))),
                representation: RefCell::new(ManagedDoubleRNSElRepresentation::Sum),
                small_basis_part: RefCell::new(Some(self.base.drop_rns_factor_non_fft(&from.base, dropped_rns_factors, value.internal.small_basis_part.borrow().as_ref().unwrap()))),
                small_basis_repr: OnceCell::new()
            }) },
            ManagedDoubleRNSElRepresentation::SmallBasis => ManagedDoubleRNSEl { internal: Rc::new(DoubleRNSElInternal {
                double_rns_repr_or_part: RefCell::new(None),
                representation: RefCell::new(ManagedDoubleRNSElRepresentation::SmallBasis),
                small_basis_part: RefCell::new(None),
                small_basis_repr: {
                    let result = OnceCell::new();
                    result.set(self.base.drop_rns_factor_non_fft(&from.base, dropped_rns_factors, value.internal.small_basis_repr.get().unwrap())).ok().unwrap();
                    result
                }
            }) },
            ManagedDoubleRNSElRepresentation::DoubleRNS => ManagedDoubleRNSEl { internal: Rc::new(DoubleRNSElInternal {
                double_rns_repr_or_part: RefCell::new(Some(self.base.drop_rns_factor(&from.base, dropped_rns_factors, value.internal.double_rns_repr_or_part.borrow().as_ref().unwrap()))),
                representation: RefCell::new(ManagedDoubleRNSElRepresentation::DoubleRNS),
                small_basis_part: RefCell::new(None),
                small_basis_repr: OnceCell::new()
            }) },
            ManagedDoubleRNSElRepresentation::Both => ManagedDoubleRNSEl { internal: Rc::new(DoubleRNSElInternal {
                double_rns_repr_or_part: RefCell::new(Some(self.base.drop_rns_factor(&from.base, dropped_rns_factors, value.internal.double_rns_repr_or_part.borrow().as_ref().unwrap()))),
                representation: RefCell::new(ManagedDoubleRNSElRepresentation::Both),
                small_basis_part: RefCell::new(None),
                small_basis_repr: {
                    let result = OnceCell::new();
                    result.set(self.base.drop_rns_factor_non_fft(&from.base, dropped_rns_factors, value.internal.small_basis_repr.get().unwrap())).ok().unwrap();
                    result
                }
            }) }
        }
    }

    fn drop_rns_factor_prepared(&self, from: &Self, drop_factors: &[usize], value: Self::PreparedMultiplicant) -> Self::PreparedMultiplicant {
        self.drop_rns_factor(from, drop_factors, value)
    }

    fn mul_prepared(&self, lhs: &Self::PreparedMultiplicant, rhs: &Self::PreparedMultiplicant) -> Self::Element {
        self.mul_ref(lhs, rhs)
    }

    fn prepare_multiplicant(&self, x: &Self::Element) -> Self::PreparedMultiplicant {
        self.clone_el(x)
    }

    fn as_representation_wrt_small_generating_set<'a>(&'a self, x: &'a Self::Element) -> Submatrix<'a, AsFirstElement<ZnEl>, ZnEl> {
        self.base.as_matrix_wrt_small_basis(self.to_small_basis(x).unwrap_or(&self.zero))
    }

    fn from_representation_wrt_small_generating_set<V>(&self, data: Submatrix<V, ZnEl>) -> Self::Element
        where V: AsPointerToSlice<ZnEl>
    {
        let mut x = self.base.zero_non_fft();
        let mut x_as_matrix = self.base.as_matrix_wrt_small_basis_mut(&mut x);
        assert_eq!(data.row_count(), x_as_matrix.row_count());
        assert_eq!(data.col_count(), x_as_matrix.col_count());
        for i in 0..data.row_count() {
            for j in 0..data.col_count() {
                *x_as_matrix.at_mut(i, j) = self.base.base_ring().at(i).clone_el(data.at(i, j));
            }
        }
        ManagedDoubleRNSEl { internal: Rc::new(DoubleRNSElInternal { 
            representation: RefCell::new(ManagedDoubleRNSElRepresentation::SmallBasis), 
            small_basis_repr: {
                let result = OnceCell::new();
                result.set(x).ok().unwrap();
                result   
            }, 
            double_rns_repr_or_part: RefCell::new(None), 
            small_basis_part: RefCell::new(None)
        }) }
    }
}

impl<NumberRing, A> CyclotomicRing for ManagedDoubleRNSRingBase<NumberRing, A> 
    where NumberRing: HECyclotomicNumberRing,
        A: Allocator + Clone
{
    fn n(&self) -> usize {
        self.base.n()
    }

    fn apply_galois_action(&self, el: &Self::Element, g: CyclotomicGaloisGroupEl) -> Self::Element {
        let result = if let Some(value) = self.to_doublerns(el) {
            self.base.apply_galois_action(&*value, g)
        } else {
            return self.zero();
        };
        ManagedDoubleRNSEl { internal: Rc::new(DoubleRNSElInternal { 
            representation: RefCell::new(ManagedDoubleRNSElRepresentation::DoubleRNS), 
            small_basis_repr: OnceCell::new(), 
            double_rns_repr_or_part: RefCell::new(Some(result)), 
            small_basis_part: RefCell::new(None)
        }) }
    }
}

impl<NumberRing, A> PartialEq for ManagedDoubleRNSRingBase<NumberRing, A>
    where NumberRing: HENumberRing,
        A: Allocator + Clone
{
    fn eq(&self, other: &Self) -> bool {
        self.base == other.base
    }
}

impl<NumberRing, A> RingBase for ManagedDoubleRNSRingBase<NumberRing, A>
    where NumberRing: HENumberRing,
        A: Allocator + Clone
{
    type Element = ManagedDoubleRNSEl<NumberRing, A>;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        ManagedDoubleRNSEl { internal: val.internal.clone() }
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        match (lhs.internal.get_repr(), rhs.internal.get_repr()) {
            (ManagedDoubleRNSElRepresentation::Zero, _) | (_, ManagedDoubleRNSElRepresentation::Zero) => self.is_zero(lhs),
            (ManagedDoubleRNSElRepresentation::SmallBasis, _) | (_, ManagedDoubleRNSElRepresentation::SmallBasis) => self.base.eq_el_non_fft(&*self.to_small_basis(lhs).unwrap(), &*self.to_small_basis(rhs).unwrap()),
            _ => self.base.eq_el(&*self.to_doublerns(lhs).unwrap(), &*self.to_doublerns(rhs).unwrap())
        }
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        match value.internal.get_repr() {
            ManagedDoubleRNSElRepresentation::Zero => true,
            ManagedDoubleRNSElRepresentation::Sum | ManagedDoubleRNSElRepresentation::SmallBasis | ManagedDoubleRNSElRepresentation::Both => self.base.eq_el_non_fft(&*self.to_small_basis(value).unwrap(), &self.zero),
            ManagedDoubleRNSElRepresentation::DoubleRNS => self.base.is_zero(&*self.to_doublerns(value).unwrap())
        }
    }

    fn zero(&self) -> Self::Element {
        ManagedDoubleRNSEl { internal: Rc::new(DoubleRNSElInternal { 
            representation: RefCell::new(ManagedDoubleRNSElRepresentation::Zero), 
            small_basis_repr: OnceCell::new(), 
            double_rns_repr_or_part: RefCell::new(None), 
            small_basis_part: RefCell::new(None)
        }) }
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
        value.internal = Rc::new(DoubleRNSElInternal { 
            representation: RefCell::new(ManagedDoubleRNSElRepresentation::DoubleRNS), 
            small_basis_repr: OnceCell::new(), 
            double_rns_repr_or_part: RefCell::new(Some(result)), 
            small_basis_part: RefCell::new(None)
        });
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
        return ManagedDoubleRNSEl { internal: Rc::new(DoubleRNSElInternal { 
            representation: RefCell::new(ManagedDoubleRNSElRepresentation::DoubleRNS), 
            small_basis_repr: OnceCell::new(), 
            double_rns_repr_or_part: RefCell::new(Some(result)), 
            small_basis_part: RefCell::new(None)
        }) };
    }

    fn pow_gen<R: IntegerRingStore>(&self, x: Self::Element, power: &El<R>, integers: R) -> Self::Element 
        where R::Type: IntegerRing
    {
        let result = if let Some(nonzero) = self.to_doublerns(&x) {
            self.base.pow_gen(self.base.clone_el(&*nonzero), power, integers)
        } else {
            return self.zero();
        };
        return ManagedDoubleRNSEl { internal: Rc::new(DoubleRNSElInternal { 
            representation: RefCell::new(ManagedDoubleRNSElRepresentation::DoubleRNS), 
            small_basis_repr: OnceCell::new(), 
            double_rns_repr_or_part: RefCell::new(Some(result)), 
            small_basis_part: RefCell::new(None)
        }) };
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

impl<NumberRing, A> RingExtension for ManagedDoubleRNSRingBase<NumberRing, A>
    where NumberRing: HENumberRing,
        A: Allocator + Clone
{
    type BaseRing = <DoubleRNSRingBase<NumberRing, A> as RingExtension>::BaseRing;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing {
        self.base.base_ring()
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        let result_fft = self.base.from(self.base_ring().clone_el(&x));
        let result_coeff = self.base.from_non_fft(x);
        return ManagedDoubleRNSEl { internal: Rc::new(DoubleRNSElInternal {
            double_rns_repr_or_part: RefCell::new(Some(result_fft)),
            representation: RefCell::new(ManagedDoubleRNSElRepresentation::Both),
            small_basis_part: RefCell::new(None),
            small_basis_repr: {
                let result = OnceCell::new();
                result.set(result_coeff).ok().unwrap();
                result
            }
        })};
    }
}

impl<NumberRing, A> FreeAlgebra for ManagedDoubleRNSRingBase<NumberRing, A>
    where NumberRing: HENumberRing,
        A: Allocator + Clone
{
    type VectorRepresentation<'a> = DoubleRNSRingBaseElVectorRepresentation<'a, NumberRing, A> 
        where Self: 'a;

    fn canonical_gen(&self) -> Self::Element {
        let result = self.base.canonical_gen();
        return ManagedDoubleRNSEl { internal: Rc::new(DoubleRNSElInternal {
            double_rns_repr_or_part: RefCell::new(Some(result)),
            representation: RefCell::new(ManagedDoubleRNSElRepresentation::DoubleRNS),
            small_basis_part: RefCell::new(None),
            small_basis_repr: OnceCell::new()
        })};
    }

    fn rank(&self) -> usize {
        self.base.rank()
    }

    fn wrt_canonical_basis<'a>(&'a self, el: &'a Self::Element) -> Self::VectorRepresentation<'a> {
        if let Some(result) = self.to_small_basis(el) {
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
        return ManagedDoubleRNSEl { internal: Rc::new(DoubleRNSElInternal {
            double_rns_repr_or_part: RefCell::new(Some(result)),
            representation: RefCell::new(ManagedDoubleRNSElRepresentation::DoubleRNS),
            small_basis_part: RefCell::new(None),
            small_basis_repr: OnceCell::new()
        })};
    }
}

impl<NumberRing, A> FiniteRingSpecializable for ManagedDoubleRNSRingBase<NumberRing, A>
    where NumberRing: HENumberRing,
        A: Allocator + Clone
{
    fn specialize<O: FiniteRingOperation<Self>>(op: O) -> Result<O::Output, ()> {
        Ok(op.execute())
    }
}

impl<NumberRing, A> FiniteRing for ManagedDoubleRNSRingBase<NumberRing, A>
    where NumberRing: HENumberRing,
        A: Allocator + Clone
{
    type ElementsIter<'a> = std::iter::Map<<DoubleRNSRingBase<NumberRing, A> as FiniteRing>::ElementsIter<'a>, fn(DoubleRNSEl<NumberRing, A>) -> ManagedDoubleRNSEl<NumberRing, A>>
        where Self: 'a;

    fn elements<'a>(&'a self) -> Self::ElementsIter<'a> {
        fn from_doublerns<NumberRing, A>(x: DoubleRNSEl<NumberRing, A>) -> ManagedDoubleRNSEl<NumberRing, A>
            where NumberRing: HENumberRing,
                A: Allocator + Clone
        {
            return ManagedDoubleRNSEl { internal: Rc::new(DoubleRNSElInternal {
                double_rns_repr_or_part: RefCell::new(Some(x)),
                representation: RefCell::new(ManagedDoubleRNSElRepresentation::DoubleRNS),
                small_basis_part: RefCell::new(None),
                small_basis_repr: OnceCell::new()
            })};
        }
        self.base.elements().map(from_doublerns)
    }

    fn random_element<G: FnMut() -> u64>(&self, rng: G) -> <Self as RingBase>::Element {
        return ManagedDoubleRNSEl { internal: Rc::new(DoubleRNSElInternal {
            double_rns_repr_or_part: RefCell::new(Some(self.base.random_element(rng))),
            representation: RefCell::new(ManagedDoubleRNSElRepresentation::DoubleRNS),
            small_basis_part: RefCell::new(None),
            small_basis_repr: OnceCell::new()
        })};
    }

    fn size<I: IntegerRingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        self.base.size(ZZ)
    }
}

impl<NumberRing, A1, A2, C> CanHomFrom<SingleRNSRingBase<NumberRing, A1, C>> for ManagedDoubleRNSRingBase<NumberRing, A2>
    where NumberRing: HECyclotomicNumberRing,
        A1: Allocator + Clone,
        A2: Allocator + Clone,
        C: PreparedConvolutionAlgorithm<ZnBase>
{
    type Homomorphism = <DoubleRNSRingBase<NumberRing, A2> as CanHomFrom<SingleRNSRingBase<NumberRing, A1, C>>>::Homomorphism;

    fn has_canonical_hom(&self, from: &SingleRNSRingBase<NumberRing, A1, C>) -> Option<Self::Homomorphism> {
        self.base.has_canonical_hom(from)
    }

    fn map_in(&self, from: &SingleRNSRingBase<NumberRing, A1, C>, el: <SingleRNSRingBase<NumberRing, A1, C> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        if from.is_zero(&el) {
            return self.zero();
        }
        return ManagedDoubleRNSEl { internal: Rc::new(DoubleRNSElInternal {
            double_rns_repr_or_part: RefCell::new(None),
            representation: RefCell::new(ManagedDoubleRNSElRepresentation::SmallBasis),
            small_basis_part: RefCell::new(None),
            small_basis_repr: {
                let result = OnceCell::new();
                result.set(self.base.map_in_from_singlerns(from, el, hom)).ok().unwrap();
                result
            }
        })};
    }
}

impl<NumberRing, A1, A2> CanHomFrom<DoubleRNSRingBase<NumberRing, A1>> for ManagedDoubleRNSRingBase<NumberRing, A2>
    where NumberRing: HECyclotomicNumberRing,
        A1: Allocator + Clone,
        A2: Allocator + Clone
{
    type Homomorphism = <DoubleRNSRingBase<NumberRing, A2> as CanHomFrom<DoubleRNSRingBase<NumberRing, A2>>>::Homomorphism;

    fn has_canonical_hom(&self, from: &DoubleRNSRingBase<NumberRing, A1>) -> Option<Self::Homomorphism> {
        self.base.has_canonical_hom(from)
    }

    fn map_in(&self, from: &DoubleRNSRingBase<NumberRing, A1>, el: <DoubleRNSRingBase<NumberRing, A1> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        if from.is_zero(&el) {
            return self.zero();
        }
        return ManagedDoubleRNSEl { internal: Rc::new(DoubleRNSElInternal {
            double_rns_repr_or_part: RefCell::new(Some(self.base.map_in(from, el, hom))),
            representation: RefCell::new(ManagedDoubleRNSElRepresentation::DoubleRNS),
            small_basis_part: RefCell::new(None),
            small_basis_repr: OnceCell::new()
        })};
    }
}

impl<NumberRing, A1, A2> CanHomFrom<ManagedDoubleRNSRingBase<NumberRing, A1>> for ManagedDoubleRNSRingBase<NumberRing, A2>
    where NumberRing: HECyclotomicNumberRing,
        A1: Allocator + Clone,
        A2: Allocator + Clone
{
    type Homomorphism = <DoubleRNSRingBase<NumberRing, A2> as CanHomFrom<DoubleRNSRingBase<NumberRing, A2>>>::Homomorphism;

    fn has_canonical_hom(&self, from: &ManagedDoubleRNSRingBase<NumberRing, A1>) -> Option<Self::Homomorphism> {
        self.base.has_canonical_hom(&from.base)
    }

    fn map_in_ref(&self, from: &ManagedDoubleRNSRingBase<NumberRing, A1>, el: &<ManagedDoubleRNSRingBase<NumberRing, A1> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        if let Some(el) = from.to_doublerns(el) {
            return ManagedDoubleRNSEl { internal: Rc::new(DoubleRNSElInternal {
                double_rns_repr_or_part: RefCell::new(Some(self.base.map_in_ref(&from.base, &*el, hom))),
                representation: RefCell::new(ManagedDoubleRNSElRepresentation::DoubleRNS),
                small_basis_part: RefCell::new(None),
                small_basis_repr: OnceCell::new()
            })};
        } else {
            self.zero()
        }
    }

    fn map_in(&self, from: &ManagedDoubleRNSRingBase<NumberRing, A1>, el: <ManagedDoubleRNSRingBase<NumberRing, A1> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in_ref(from, &el, hom)
    }
}

impl<NumberRing, A1, A2, C> CanIsoFromTo<SingleRNSRingBase<NumberRing, A1, C>> for ManagedDoubleRNSRingBase<NumberRing, A2>
    where NumberRing: HECyclotomicNumberRing,
        A1: Allocator + Clone,
        A2: Allocator + Clone,
        C: PreparedConvolutionAlgorithm<ZnBase>
{
    type Isomorphism = <DoubleRNSRingBase<NumberRing, A2> as CanIsoFromTo<SingleRNSRingBase<NumberRing, A1, C>>>::Isomorphism;

    fn has_canonical_iso(&self, to: &SingleRNSRingBase<NumberRing, A1, C>) -> Option<Self::Isomorphism> {
        self.base.has_canonical_iso(to)
    }

    fn map_out(&self, to: &SingleRNSRingBase<NumberRing, A1, C>, el: Self::Element, iso: &Self::Isomorphism) -> <SingleRNSRingBase<NumberRing, A1, C> as RingBase>::Element {
        if let Some(el) = self.to_small_basis(&el) {
            self.base.map_out_to_singlerns(to, self.base.clone_el_non_fft(&*&el), iso)
        } else {
            to.zero()
        }
    }
}

impl<NumberRing, A1, A2> CanIsoFromTo<DoubleRNSRingBase<NumberRing, A1>> for ManagedDoubleRNSRingBase<NumberRing, A2>
    where NumberRing: HECyclotomicNumberRing,
        A1: Allocator + Clone,
        A2: Allocator + Clone
{
    type Isomorphism = <DoubleRNSRingBase<NumberRing, A2> as CanIsoFromTo<DoubleRNSRingBase<NumberRing, A2>>>::Isomorphism;

    fn has_canonical_iso(&self, to: &DoubleRNSRingBase<NumberRing, A1>) -> Option<Self::Isomorphism> {
        self.base.has_canonical_iso(to)
    }

    fn map_out(&self, to: &DoubleRNSRingBase<NumberRing, A1>, el: Self::Element, iso: &Self::Isomorphism) -> <DoubleRNSRingBase<NumberRing, A1> as RingBase>::Element {
        if let Some(el) = self.to_doublerns(&el) {
            self.base.map_out(to, self.base.clone_el(&*&el), iso)
        } else {
            to.zero()
        }
    }
}

impl<NumberRing, A1, A2> CanIsoFromTo<ManagedDoubleRNSRingBase<NumberRing, A1>> for ManagedDoubleRNSRingBase<NumberRing, A2>
    where NumberRing: HECyclotomicNumberRing,
        A1: Allocator + Clone,
        A2: Allocator + Clone
{
    type Isomorphism = <DoubleRNSRingBase<NumberRing, A1> as CanHomFrom<DoubleRNSRingBase<NumberRing, A2>>>::Homomorphism;

    fn has_canonical_iso(&self, to: &ManagedDoubleRNSRingBase<NumberRing, A1>) -> Option<Self::Isomorphism> {
        to.has_canonical_hom(self)
    }

    fn map_out(&self, to: &ManagedDoubleRNSRingBase<NumberRing, A1>, el: Self::Element, iso: &Self::Isomorphism) -> <ManagedDoubleRNSRingBase<NumberRing, A1> as RingBase>::Element {
        to.map_in(self, el, iso)
    }
}

#[test]
fn test_ring_axioms() {
    let rns_base = zn_rns::Zn::new(vec![Zn::new(17), Zn::new(97)], BigIntRing::RING);
    let ring = ManagedDoubleRNSRingBase::new(Pow2CyclotomicNumberRing::new(16), rns_base);
    
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
    feanor_math::ring::generic_tests::test_self_iso(&ring, elements.iter().map(|x| ring.clone_el(x)));
}

#[test]
fn test_canonical_hom_from_doublerns() {
    let rns_base = zn_rns::Zn::new(vec![Zn::new(17), Zn::new(97)], BigIntRing::RING);
    let ring = ManagedDoubleRNSRingBase::new(Pow2CyclotomicNumberRing::new(16), rns_base);

    let doublerns_ring = RingRef::new(&ring.get_ring().base);
    let elements = vec![
        doublerns_ring.zero(),
        doublerns_ring.one(),
        doublerns_ring.neg_one(),
        doublerns_ring.int_hom().map(17),
        doublerns_ring.int_hom().map(97),
        doublerns_ring.canonical_gen(),
        doublerns_ring.pow(doublerns_ring.canonical_gen(), 15),
        doublerns_ring.int_hom().mul_map(doublerns_ring.canonical_gen(), 17),
        doublerns_ring.int_hom().mul_map(doublerns_ring.pow(doublerns_ring.canonical_gen(), 15), 17),
        doublerns_ring.add(doublerns_ring.canonical_gen(), doublerns_ring.one())
    ];

    feanor_math::ring::generic_tests::test_hom_axioms(doublerns_ring, &ring, elements.iter().map(|x| doublerns_ring.clone_el(x)));
}

#[test]
fn test_canonical_hom_from_singlerns() {
    let rns_base = zn_rns::Zn::new(vec![Zn::new(97), Zn::new(193)], BigIntRing::RING);
    let ring = ManagedDoubleRNSRingBase::new(Pow2CyclotomicNumberRing::new(16), rns_base.clone());

    let singlerns_ring = SingleRNSRingBase::<_, _, DefaultConvolution>::new(Pow2CyclotomicNumberRing::new(16), rns_base);
    let elements = vec![
        singlerns_ring.zero(),
        singlerns_ring.one(),
        singlerns_ring.neg_one(),
        singlerns_ring.int_hom().map(97),
        singlerns_ring.int_hom().map(193),
        singlerns_ring.canonical_gen(),
        singlerns_ring.pow(singlerns_ring.canonical_gen(), 15),
        singlerns_ring.int_hom().mul_map(singlerns_ring.canonical_gen(), 97),
        singlerns_ring.int_hom().mul_map(singlerns_ring.pow(singlerns_ring.canonical_gen(), 15), 97),
        singlerns_ring.add(singlerns_ring.canonical_gen(), singlerns_ring.one())
    ];

    feanor_math::ring::generic_tests::test_hom_axioms(&singlerns_ring, &ring, elements.iter().map(|x| singlerns_ring.clone_el(x)));
}

#[test]
fn test_add_result_independent_of_repr() {
    let rns_base = zn_rns::Zn::new(vec![Zn::new(17), Zn::new(97)], BigIntRing::RING);
    let ring = ManagedDoubleRNSRingBase::new(Pow2CyclotomicNumberRing::new(4), rns_base);
    let base = &ring.get_ring().base;
    let reprs_of_11: [Box<dyn Fn() -> ManagedDoubleRNSEl<_, _>>; 4] = [
        Box::new(|| ManagedDoubleRNSEl { internal: Rc::new(DoubleRNSElInternal {
            double_rns_repr_or_part: RefCell::new(None),
            representation: RefCell::new(ManagedDoubleRNSElRepresentation::SmallBasis),
            small_basis_repr: {
                let result = OnceCell::new();
                result.set(base.from_non_fft(base.base_ring().int_hom().map(11))).ok().unwrap();
                result
            },
            small_basis_part: RefCell::new(None)
        })}),
        Box::new(|| ManagedDoubleRNSEl { internal: Rc::new(DoubleRNSElInternal {
            double_rns_repr_or_part: RefCell::new(Some(base.from_int(11))),
            representation: RefCell::new(ManagedDoubleRNSElRepresentation::DoubleRNS),
            small_basis_part: RefCell::new(None),
            small_basis_repr: OnceCell::new()
        })}),
        Box::new(|| ManagedDoubleRNSEl { internal: Rc::new(DoubleRNSElInternal {
            double_rns_repr_or_part: RefCell::new(Some(base.from_int(10))),
            representation: RefCell::new(ManagedDoubleRNSElRepresentation::Sum),
            small_basis_part: RefCell::new(Some(base.from_non_fft(base.base_ring().int_hom().map(1)))),
            small_basis_repr: OnceCell::new()
        })}),
        Box::new(|| ManagedDoubleRNSEl { internal: Rc::new(DoubleRNSElInternal {
            double_rns_repr_or_part: RefCell::new(Some(base.from_int(11))),
            representation: RefCell::new(ManagedDoubleRNSElRepresentation::Both),
            small_basis_repr: {
                let result = OnceCell::new();
                result.set(base.from_non_fft(base.base_ring().int_hom().map(11))).ok().unwrap();
                result
            },
            small_basis_part: RefCell::new(None)
        })})
    ];
    let reprs_of_102: [Box<dyn Fn() -> ManagedDoubleRNSEl<_, _>>; 4] = [
        Box::new(|| ManagedDoubleRNSEl { internal: Rc::new(DoubleRNSElInternal {
            double_rns_repr_or_part: RefCell::new(None),
            representation: RefCell::new(ManagedDoubleRNSElRepresentation::SmallBasis),
            small_basis_repr: {
                let result = OnceCell::new();
                result.set(base.from_non_fft(base.base_ring().int_hom().map(102))).ok().unwrap();
                result
            },
            small_basis_part: RefCell::new(None)
        })}),
        Box::new(|| ManagedDoubleRNSEl { internal: Rc::new(DoubleRNSElInternal {
            double_rns_repr_or_part: RefCell::new(Some(base.from_int(102))),
            representation: RefCell::new(ManagedDoubleRNSElRepresentation::DoubleRNS),
            small_basis_part: RefCell::new(None),
            small_basis_repr: OnceCell::new()
        })}),
        Box::new(|| ManagedDoubleRNSEl { internal: Rc::new(DoubleRNSElInternal {
            double_rns_repr_or_part: RefCell::new(Some(base.from_int(100))),
            representation: RefCell::new(ManagedDoubleRNSElRepresentation::Sum),
            small_basis_part: RefCell::new(Some(base.from_non_fft(base.base_ring().int_hom().map(2)))),
            small_basis_repr: OnceCell::new()
        })}),
        Box::new(|| ManagedDoubleRNSEl { internal: Rc::new(DoubleRNSElInternal {
            double_rns_repr_or_part: RefCell::new(Some(base.from_int(102))),
            representation: RefCell::new(ManagedDoubleRNSElRepresentation::Both),
            small_basis_repr: {
                let result = OnceCell::new();
                result.set(base.from_non_fft(base.base_ring().int_hom().map(102))).ok().unwrap();
                result
            },
            small_basis_part: RefCell::new(None)
        })})
    ];
    for a in &reprs_of_11 {
        for b in &reprs_of_102 {
            let x = a();
            assert_el_eq!(RingRef::new(base), base.from_int(22), ring.get_ring().to_doublerns(&ring.add_ref(&x, &x)).unwrap());

            let x = a();
            let y = b();
            assert_el_eq!(RingRef::new(base), base.from_int(113), ring.get_ring().to_doublerns(&ring.add_ref(&x, &y)).unwrap());

            let x = a();
            let y = b();
            assert!(base.eq_el_non_fft(&base.from_non_fft(base.base_ring().int_hom().map(113)), &*ring.get_ring().to_small_basis(&ring.add_ref(&x, &y)).unwrap()));

            let x = a();
            let y = b();
            assert_el_eq!(RingRef::new(base), base.from_int(-91), ring.get_ring().to_doublerns(&ring.sub_ref(&x, &y)).unwrap());

            let x = a();
            let y = b();
            assert!(base.eq_el_non_fft(&base.from_non_fft(base.base_ring().int_hom().map(-91)), &*ring.get_ring().to_small_basis(&ring.sub_ref(&x, &y)).unwrap()));

            let x = a();
            assert_el_eq!(RingRef::new(base), base.from_int(121), ring.get_ring().to_doublerns(&ring.mul_ref(&x, &x)).unwrap());

            let x = a();
            let y = b();
            assert_el_eq!(RingRef::new(base), base.from_int(1122), ring.get_ring().to_doublerns(&ring.mul_ref(&x, &y)).unwrap());

            let x = a();
            let y = b();
            assert!(base.eq_el_non_fft(&base.from_non_fft(base.base_ring().int_hom().map(1122)), &*ring.get_ring().to_small_basis(&ring.mul_ref(&x, &y)).unwrap()));
        }
    }
}