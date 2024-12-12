use std::alloc::{Allocator, Global};
use std::marker::PhantomData;

use feanor_math::homomorphism::{CanHomFrom, CanIsoFromTo, Homomorphism};
use feanor_math::integer::{int_cast, BigIntRing, BigIntRingBase, IntegerRing, IntegerRingStore};
use feanor_math::iters::{multi_cartesian_product, MultiProduct};
use feanor_math::primitive_int::StaticRing;
use feanor_math::rings::extension::{FreeAlgebra, FreeAlgebraStore};
use feanor_math::rings::finite::FiniteRing;
use feanor_math::rings::poly::dense_poly::DensePolyRing;
use feanor_math::rings::zn::{zn_64, zn_big, zn_rns, FromModulusCreateableZnRing, ZnRing, ZnRingStore};
use feanor_math::ring::*;
use feanor_math::seq::{CloneElFn, CloneRingEl};
use feanor_math::seq::VectorView;
use feanor_math::serialization::{DeserializeWithRing, SerializableElementRing, SerializeWithRing};
use feanor_math::ordered::OrderedRingStore;
use feanor_math::rings::finite::*;

use feanor_math::specialization::{FiniteRingOperation, FiniteRingSpecializable};
use serde::{de, Deserializer, Serializer};
use serde_json::Number;

use crate::cyclotomic::CyclotomicRing;
use crate::sample_primes;
use crate::IsEq;
use crate::rings::number_ring::*;

use super::pow2_cyclotomic::Pow2CyclotomicNumberRing;

pub struct DecompositionRingBase<NumberRing, ZnTy, A = Global> 
    where NumberRing: HENumberRing,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    number_ring: NumberRing,
    base_ring: ZnTy,
    ring_decompositions: Vec<<NumberRing as HENumberRing>::Decomposed>,
    rns_base: zn_rns::Zn<zn_64::Zn, BigIntRing>,
    allocator: A
}

///
/// [`RingStore`] for [`DecompositionRingBase`]
/// 
pub type DecompositionRing<NumberRing, ZnTy, A = Global> = RingValue<DecompositionRingBase<NumberRing, ZnTy, A>>;

pub struct DecompositionRingEl<NumberRing, ZnTy, A = Global> 
    where NumberRing: HENumberRing,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    number_ring: PhantomData<NumberRing>,
    allocator: PhantomData<A>,
    data: Vec<El<ZnTy>, A>
}

impl<NumberRing, ZnTy> DecompositionRingBase<NumberRing, ZnTy>
    where NumberRing: HENumberRing,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
{
    pub fn new(number_ring: NumberRing, base_ring: ZnTy) -> RingValue<Self> {
        let ZZbig = BigIntRing::RING;
        let max_product_expansion_factor = ZZbig.from_float_approx(number_ring.product_expansion_factor().ceil()).unwrap();
        let max_lift_size = ZZbig.ceil_div(int_cast(base_ring.integer_ring().clone_el(base_ring.modulus()), ZZbig, base_ring.integer_ring()), &ZZbig.int_hom().map(2));
        let max_product_size = ZZbig.mul(ZZbig.pow(max_lift_size, 2), max_product_expansion_factor);
        let required_bits = ZZbig.abs_log2_ceil(&max_product_size).unwrap();
        let rns_base_primes = sample_primes(required_bits, required_bits + 57, 57, |n| number_ring.largest_suitable_prime(int_cast(n, StaticRing::<i64>::RING, ZZbig)).map(|n| int_cast(n, ZZbig, StaticRing::<i64>::RING))).unwrap();
        let rns_base = zn_rns::Zn::new(rns_base_primes.into_iter().map(|p| zn_64::Zn::new(int_cast(p, StaticRing::<i64>::RING, ZZbig) as u64)).collect(), ZZbig);
        println!("rns base len: {}", rns_base.len());
        return Self::new_with(number_ring, base_ring, rns_base, Global);
    }
}

impl<NumberRing, ZnTy, A> DecompositionRingBase<NumberRing, ZnTy, A>
    where NumberRing: HENumberRing,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    pub fn new_with(number_ring: NumberRing, base_ring: ZnTy, rns_base: zn_rns::Zn<zn_64::Zn, BigIntRing>, allocator: A) -> RingValue<Self> {
        assert!(rns_base.len() > 0);
        let ZZbig = BigIntRing::RING;
        let max_product_expansion_factor = ZZbig.from_float_approx(number_ring.product_expansion_factor().ceil()).unwrap();
        let max_lift_size = int_cast(base_ring.integer_ring().clone_el(base_ring.modulus()), ZZbig, base_ring.integer_ring());
        let max_product_size = ZZbig.mul(ZZbig.pow(max_lift_size, 2), max_product_expansion_factor);
        assert!(ZZbig.is_gt(&ZZbig.prod(rns_base.as_iter().map(|rns_base_ring| int_cast(rns_base_ring.integer_ring().clone_el(rns_base_ring.modulus()), ZZbig, rns_base_ring.integer_ring()))), &max_product_size));
        RingValue::from(Self {
            base_ring: base_ring,
            ring_decompositions: rns_base.as_iter().map(|Fp| number_ring.mod_p(Fp.clone())).collect(),
            number_ring: number_ring,
            rns_base: rns_base,
            allocator: allocator
        })
    }

    pub fn with_allocator<ANew>(self, new_allocator: ANew) -> DecompositionRingBase<NumberRing, ZnTy, ANew>
        where ANew: Allocator + Clone
    {
        DecompositionRingBase {
            allocator: new_allocator,
            number_ring: self.number_ring,
            base_ring: self.base_ring,
            ring_decompositions: self.ring_decompositions,
            rns_base: self.rns_base
        }
    }

    pub fn allocator(&self) -> &A {
        &self.allocator
    }

    pub fn ring_decompositions(&self) -> &[<NumberRing as HENumberRing>::Decomposed] {
        &self.ring_decompositions
    }

    pub fn wrt_canonical_basis_mut<'a>(&self, el: &'a mut DecompositionRingEl<NumberRing, ZnTy, A>) -> &'a mut [El<ZnTy>] {
        &mut el.data
    }

    pub fn rns_base(&self) -> &zn_rns::Zn<zn_64::Zn, BigIntRing> {
        &self.rns_base
    }

    pub fn number_ring(&self) -> &NumberRing {
        &self.number_ring
    }
    
    ///
    /// Computes `sum sigma_i(x_i)` where `els` yields pairs `(x_i, sigma_i)` with `sigma_i` being
    /// a Galois automorphism.
    /// 
    /// Note that this can be faster than directly computing the sum, since we can avoid some of the 
    /// intermediate DFTs. This is possible, since we only add elements, so the coefficients grow quite
    /// slowly, as opposed to multiplications.
    /// 
    pub fn sum_galois_transforms<I>(&self, els: I) -> <Self as RingBase>::Element
        where NumberRing: HECyclotomicNumberRing,
            I: Iterator<Item = (<Self as RingBase>::Element, zn_64::ZnEl)>
    {
        let mut unreduced_result = Vec::with_capacity_in(self.rank() * self.rns_base.len(), &self.allocator);
        unreduced_result.resize_with(self.rank() * self.rns_base.len(), || self.rns_base.at(0).zero());
        let mut tmp = Vec::with_capacity_in(self.rank(), &self.allocator);
        tmp.resize_with(self.rank(), || self.rns_base.at(0).zero());
        let mut tmp_perm = Vec::with_capacity_in(self.rank(), &self.allocator);
        tmp_perm.resize_with(self.rank(), || self.rns_base.at(0).zero());

        let mut len = 0;
        for (x, g) in els {
            len += 1;
            for i in 0..self.rns_base.len() {
                let Zp = self.rns_base.at(i);
                let from_lifted = Zp.can_hom(self.base_ring().integer_ring()).unwrap();
                for j in 0..self.rank() {
                    tmp[j] = from_lifted.map(self.base_ring().smallest_lift(self.base_ring().clone_el(&x.data[j])));
                }
                self.ring_decompositions[i].coeff_basis_to_small_basis(&mut tmp[..]);
                self.ring_decompositions[i].small_basis_to_mult_basis(&mut tmp[..]);
                <_ as HECyclotomicNumberRingMod>::permute_galois_action(<NumberRing::DecomposedAsCyclotomic>::from_ref(&self.ring_decompositions[i]), &tmp[..], &mut tmp_perm[..], g);
                for j in 0..self.rank() {
                    Zp.add_assign_ref(&mut unreduced_result[i * self.rank() + j], &tmp_perm[j]);
                }
            }
        }
        drop(tmp);
        drop(tmp_perm);

        // if this is satisfied, we have enough precision to not get an overflow
        let toleratable_size_increase = int_cast(self.base_ring.integer_ring().clone_el(self.base_ring.modulus()), StaticRing::<i64>::RING, self.base_ring.integer_ring()) as f64 * self.number_ring.product_expansion_factor();
        let max_size_increase = len as f64  * self.number_ring.inf_to_can_norm_expansion_factor() * self.number_ring.can_to_inf_norm_expansion_factor();
        assert!(max_size_increase < toleratable_size_increase);
        for i in 0..self.rns_base.len() {
            self.ring_decompositions[i].mult_basis_to_small_basis(&mut unreduced_result[(i * self.rank())..((i + 1) * self.rank())]);
            self.ring_decompositions[i].small_basis_to_coeff_basis(&mut unreduced_result[(i * self.rank())..((i + 1) * self.rank())]);
        }

        let hom = self.base_ring().can_hom(&BigIntRing::RING).unwrap();
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        for j in 0..self.rank() {
            result.push(hom.map(self.rns_base.smallest_lift(
                self.rns_base.from_congruence((0..self.rns_base.len()).map(|i| self.rns_base.at(i).clone_el(&unreduced_result[i * self.rank() + j])))
            )));
        }
        return DecompositionRingEl {
            data: result,
            number_ring: PhantomData,
            allocator: PhantomData
        };
    }
}

impl<NumberRing, ZnTy, A> CyclotomicRing for DecompositionRingBase<NumberRing, ZnTy, A>
    where NumberRing: HECyclotomicNumberRing,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    fn n(&self) -> u64 {
        <NumberRing::DecomposedAsCyclotomic>::from_ref(&self.ring_decompositions()[0]).n()
    }

    fn apply_galois_action(&self, el: &<Self as RingBase>::Element, g: zn_64::ZnEl) -> <Self as RingBase>::Element {
        assert_eq!(el.data.len(), self.rank());

        let mut unreduced_result = Vec::with_capacity_in(self.rank() * self.rns_base.len(), &self.allocator);
        unreduced_result.resize_with(self.rank() * self.rns_base.len(), || self.rns_base.at(0).zero());
        let mut tmp = Vec::with_capacity_in(self.rank(), &self.allocator);
        tmp.resize_with(self.rank(), || self.rns_base.at(0).zero());

        for i in 0..self.rns_base.len() {
            let Zp = self.rns_base.at(i);
            let from_lifted = Zp.can_hom(self.base_ring().integer_ring()).unwrap();
            for j in 0..self.rank() {
                tmp[j] = from_lifted.map(self.base_ring().smallest_lift(self.base_ring().clone_el(&el.data[j])));
            }
            self.ring_decompositions[i].coeff_basis_to_small_basis(&mut tmp[..]);
            self.ring_decompositions[i].small_basis_to_mult_basis(&mut tmp[..]);
            <NumberRing::DecomposedAsCyclotomic>::from_ref(&self.ring_decompositions()[i]).permute_galois_action(&tmp[..], &mut unreduced_result[(i * self.rank())..((i + 1) * self.rank())], g);
            self.ring_decompositions[i].mult_basis_to_small_basis(&mut unreduced_result[(i * self.rank())..((i + 1) * self.rank())]);
            self.ring_decompositions[i].small_basis_to_coeff_basis(&mut unreduced_result[(i * self.rank())..((i + 1) * self.rank())]);
        }
        drop(tmp);

        let hom = self.base_ring().can_hom(&BigIntRing::RING).unwrap();
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        for j in 0..self.rank() {
            result.push(hom.map(self.rns_base.smallest_lift(
                self.rns_base.from_congruence((0..self.rns_base.len()).map(|i| self.rns_base.at(i).clone_el(&unreduced_result[i * self.rank() + j])))
            )));
        }
        return DecompositionRingEl {
            data: result,
            number_ring: PhantomData,
            allocator: PhantomData
        };
    }
}

impl<NumberRing, ZnTy, A> PartialEq for DecompositionRingBase<NumberRing, ZnTy, A>
    where NumberRing: HENumberRing,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    fn eq(&self, other: &Self) -> bool {
        self.number_ring == other.number_ring && self.base_ring.get_ring() == other.base_ring.get_ring()
    }
}

impl<NumberRing, ZnTy, A> RingBase for DecompositionRingBase<NumberRing, ZnTy, A>
    where NumberRing: HENumberRing,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    type Element = DecompositionRingEl<NumberRing, ZnTy, A>;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        result.extend(val.data.iter().map(|x| self.base_ring().clone_el(x)));
        return DecompositionRingEl {
            data: result,
            number_ring: PhantomData,
            allocator: PhantomData
        };
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        assert_eq!(lhs.data.len(), self.rank());
        assert_eq!(rhs.data.len(), self.rank());
        for (i, x) in rhs.data.into_iter().enumerate() {
            self.base_ring().add_assign(&mut lhs.data[i], x)
        }
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        assert_eq!(lhs.data.len(), self.rank());
        assert_eq!(rhs.data.len(), self.rank());
        for (i, x) in (&rhs.data).into_iter().enumerate() {
            self.base_ring().add_assign_ref(&mut lhs.data[i], x)
        }
    }

    fn sub_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        assert_eq!(lhs.data.len(), self.rank());
        assert_eq!(rhs.data.len(), self.rank());
        for (i, x) in (&rhs.data).into_iter().enumerate() {
            self.base_ring().sub_assign_ref(&mut lhs.data[i], x)
        }
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        assert_eq!(lhs.data.len(), self.rank());
        for i in 0..self.rank() {
            self.base_ring().negate_inplace(&mut lhs.data[i]);
        }
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        *lhs = self.mul_ref(lhs, &rhs);
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        *lhs = self.mul_ref(lhs, rhs);
    }

    fn mul_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        assert_eq!(lhs.data.len(), self.rank());
        assert_eq!(rhs.data.len(), self.rank());

        let mut unreduced_result = Vec::with_capacity_in(self.rank() * self.rns_base.len(), &self.allocator);
        let mut lhs_tmp = Vec::with_capacity_in(self.rank(), &self.allocator);
        lhs_tmp.resize_with(self.rank(), || self.rns_base.at(0).zero());
        let mut rhs_tmp = Vec::with_capacity_in(self.rank(), &self.allocator);
        rhs_tmp.resize_with(self.rank(), || self.rns_base.at(0).zero());

        for i in 0..self.rns_base.len() {
            let Zp = self.rns_base.at(i);
            let from_lifted = Zp.can_hom(self.base_ring().integer_ring()).unwrap();
            for j in 0..self.rank() {
                lhs_tmp[j] = from_lifted.map(self.base_ring().smallest_lift(self.base_ring().clone_el(&lhs.data[j])));
                rhs_tmp[j] = from_lifted.map(self.base_ring().smallest_lift(self.base_ring().clone_el(&rhs.data[j])));
            }
            self.ring_decompositions[i].coeff_basis_to_small_basis(&mut lhs_tmp[..]);
            self.ring_decompositions[i].small_basis_to_mult_basis(&mut lhs_tmp[..]);
            self.ring_decompositions[i].coeff_basis_to_small_basis(&mut rhs_tmp[..]);
            self.ring_decompositions[i].small_basis_to_mult_basis(&mut rhs_tmp[..]);
            let end_index = unreduced_result.len();
            unreduced_result.extend((0..self.rank()).map(|j| Zp.mul_ref(&lhs_tmp[j], &rhs_tmp[j])));
            self.ring_decompositions[i].mult_basis_to_small_basis(&mut unreduced_result[end_index..]);
            self.ring_decompositions[i].small_basis_to_coeff_basis(&mut unreduced_result[end_index..]);
        }
        drop(lhs_tmp);
        drop(rhs_tmp);

        let hom = self.base_ring().can_hom(&BigIntRing::RING).unwrap();
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        for j in 0..self.rank() {
            result.push(hom.map(self.rns_base.smallest_lift(
                self.rns_base.from_congruence((0..self.rns_base.len()).map(|i| self.rns_base.at(i).clone_el(&unreduced_result[i * self.rank() + j])))
            )));
        }
        return DecompositionRingEl {
            data: result,
            number_ring: PhantomData,
            allocator: PhantomData
        };
    }
    
    fn from_int(&self, value: i32) -> Self::Element {
        self.from(self.base_ring().get_ring().from_int(value))
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        assert_eq!(lhs.data.len(), self.rank());
        assert_eq!(rhs.data.len(), self.rank());
        for i in 0..self.rank() {
            if !self.base_ring().eq_el(&lhs.data[i], &rhs.data[i]) {
                return false;
            }
        }
        return true;
    }

    fn zero(&self) -> Self::Element {
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        result.extend((0..self.rank()).map(|_| self.base_ring().zero()));
        return DecompositionRingEl {
            data: result,
            number_ring: PhantomData,
            allocator: PhantomData
        };
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        assert_eq!(value.data.len(), self.rank());
        value.data.iter().all(|x| self.base_ring().is_zero(x))
    }

    fn is_one(&self, value: &Self::Element) -> bool {
        assert_eq!(value.data.len(), self.rank());
        self.base_ring().is_one(&value.data[0]) && value.data[1..].iter().all(|x| self.base_ring().is_zero(x))
    }

    fn is_neg_one(&self, value: &Self::Element) -> bool {
        assert_eq!(value.data.len(), self.rank());
        self.base_ring().is_neg_one(&value.data[0]) && value.data[1..].iter().all(|x| self.base_ring().is_zero(x))
    }
    
    fn is_commutative(&self) -> bool { true }
    fn is_noetherian(&self) -> bool { true }
    fn is_approximate(&self) -> bool { false }

    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        self.dbg_within(value, out, EnvBindingStrength::Weakest)
    }

    fn dbg_within<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>, _env: EnvBindingStrength) -> std::fmt::Result {
        let poly_ring = DensePolyRing::new(self.base_ring(), "X");
        poly_ring.get_ring().dbg(&RingRef::new(self).poly_repr(&poly_ring, value, self.base_ring().identity()), out)
    }

    fn characteristic<I: IntegerRingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        self.base_ring.characteristic(ZZ)
    }
}

impl<NumberRing, ZnTy, A> RingExtension for DecompositionRingBase<NumberRing, ZnTy, A>
    where NumberRing: HENumberRing,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    type BaseRing = ZnTy;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing {
        &self.base_ring
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        let mut result = self.zero();
        result.data[0] = x;
        return result;
    }
}

impl<NumberRing, ZnTy, A> FreeAlgebra for DecompositionRingBase<NumberRing, ZnTy, A>
    where NumberRing: HENumberRing,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    type VectorRepresentation<'a> = CloneElFn<&'a [El<ZnTy>], El<ZnTy>, CloneRingEl<&'a ZnTy>>
        where Self: 'a;

    fn from_canonical_basis<V>(&self, vec: V) -> Self::Element
        where V: IntoIterator<Item = El<Self::BaseRing>>
    {
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        result.extend(vec);
        assert_eq!(result.len(), self.rank());
        return DecompositionRingEl {
            data: result,
            number_ring: PhantomData,
            allocator: PhantomData
        };
    }

    fn wrt_canonical_basis<'a>(&'a self, el: &'a Self::Element) -> Self::VectorRepresentation<'a> {
        (&el.data[..]).clone_ring_els(self.base_ring())
    }

    fn canonical_gen(&self) -> Self::Element {
        let mut result = self.zero();
        result.data[1] = self.base_ring().one();
        return result;
    }

    fn rank(&self) -> usize {
        self.ring_decompositions[0].rank()
    }
}

pub struct WRTCanonicalBasisElementCreator<'a, NumberRing, ZnTy, A>
    where NumberRing: HENumberRing,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    ring: &'a DecompositionRingBase<NumberRing, ZnTy, A>,
}

impl<'a, NumberRing, ZnTy, A> Copy for WRTCanonicalBasisElementCreator<'a, NumberRing, ZnTy, A>
    where NumberRing: HENumberRing,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{}

impl<'a, NumberRing, ZnTy, A> Clone for WRTCanonicalBasisElementCreator<'a, NumberRing, ZnTy, A>
    where NumberRing: HENumberRing,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, 'b, NumberRing, ZnTy, A> FnOnce<(&'b [El<ZnTy>],)> for WRTCanonicalBasisElementCreator<'a, NumberRing, ZnTy, A>
    where NumberRing: HENumberRing,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    type Output = El<DecompositionRing<NumberRing, ZnTy, A>>;

    extern "rust-call" fn call_once(self, args: (&'b [El<ZnTy>],)) -> Self::Output {
        self.call(args)
    }
}

impl<'a, 'b, NumberRing, ZnTy, A> FnMut<(&'b [El<ZnTy>],)> for WRTCanonicalBasisElementCreator<'a, NumberRing, ZnTy, A>
    where NumberRing: HENumberRing,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    extern "rust-call" fn call_mut(&mut self, args: (&'b [El<ZnTy>],)) -> Self::Output {
        self.call(args)
    }
}

impl<'a, 'b, NumberRing, ZnTy, A> Fn<(&'b [El<ZnTy>],)> for WRTCanonicalBasisElementCreator<'a, NumberRing, ZnTy, A>
    where NumberRing: HENumberRing,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    extern "rust-call" fn call(&self, args: (&'b [El<ZnTy>],)) -> Self::Output {
        self.ring.from_canonical_basis(args.0.iter().map(|x| self.ring.base_ring().clone_el(x)))
    }
}

impl<NumberRing, ZnTy, A> FiniteRingSpecializable for DecompositionRingBase<NumberRing, ZnTy, A>
    where NumberRing: HENumberRing,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    fn specialize<O: FiniteRingOperation<Self>>(op: O) -> Result<O::Output, ()> {
        Ok(op.execute())
    }
}

impl<NumberRing, ZnTy, A> FiniteRing for DecompositionRingBase<NumberRing, ZnTy, A>
    where NumberRing: HENumberRing,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    type ElementsIter<'a> = MultiProduct<<ZnTy::Type as FiniteRing>::ElementsIter<'a>, WRTCanonicalBasisElementCreator<'a, NumberRing, ZnTy, A>, CloneRingEl<&'a ZnTy>, Self::Element>
        where Self: 'a;

    fn elements<'a>(&'a self) -> Self::ElementsIter<'a> {
        multi_cartesian_product((0..self.rank()).map(|_| self.base_ring().elements()), WRTCanonicalBasisElementCreator { ring: self }, CloneRingEl(self.base_ring()))
    }

    fn random_element<G: FnMut() -> u64>(&self, mut rng: G) -> Self::Element {
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        result.extend((0..self.rank()).map(|_| self.base_ring().random_element(&mut rng)));
        return DecompositionRingEl {
            data: result,
            allocator: PhantomData,
            number_ring: PhantomData
        };
    }

    fn size<I: IntegerRingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        let characteristic = self.base_ring().size(ZZ)?;
        if ZZ.get_ring().representable_bits().is_none() || ZZ.get_ring().representable_bits().unwrap() >= self.rank() * ZZ.abs_log2_ceil(&characteristic).unwrap() {
            Some(ZZ.pow(characteristic, self.rank()))
        } else {
            None
        }
    }
}

impl<NumberRing, ZnTy, A> SerializableElementRing for DecompositionRingBase<NumberRing, ZnTy, A>
    where NumberRing: HENumberRing,
        ZnTy: RingStore,
        ZnTy::Type: ZnRing + CanHomFrom<BigIntRingBase> + SerializableElementRing,
        A: Allocator + Clone
{
    fn serialize<S>(&self, el: &Self::Element, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        feanor_math::serialization::serialize_seq_helper(serializer, el.data.iter().map(|x| SerializeWithRing::new(x, self.base_ring())))
    }

    fn deserialize<'de, D>(&self, deserializer: D) -> Result<Self::Element, D::Error>
        where D: Deserializer<'de> 
    {
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        feanor_math::serialization::deserialize_seq_helper(deserializer, |x| {
            result.push(x);
        }, DeserializeWithRing::new(self.base_ring()))?;
        if result.len() != self.rank() {
            return Err(de::Error::custom(format!("expected {} elements, got {}", self.rank(), result.len())));
        }
        return Ok(DecompositionRingEl {
            data: result,
            number_ring: PhantomData,
            allocator: PhantomData
        });
    }
}

impl<NumberRing, ZnTy1, ZnTy2, A1, A2> CanHomFrom<DecompositionRingBase<NumberRing, ZnTy2, A2>> for DecompositionRingBase<NumberRing, ZnTy1, A1>
    where NumberRing: HENumberRing ,
        ZnTy1: RingStore,
        ZnTy1::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A1: Allocator + Clone,
        ZnTy2: RingStore,
        ZnTy2::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A2: Allocator + Clone,
        ZnTy1::Type: CanHomFrom<ZnTy2::Type>
{
    type Homomorphism = <ZnTy1::Type as CanHomFrom<ZnTy2::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &DecompositionRingBase<NumberRing, ZnTy2, A2>) -> Option<Self::Homomorphism> {
        if self.number_ring == from.number_ring {
            self.base_ring().get_ring().has_canonical_hom(from.base_ring().get_ring())
        } else {
            None
        }
    }

    fn map_in(&self, from: &DecompositionRingBase<NumberRing, ZnTy2, A2>, el: <DecompositionRingBase<NumberRing, ZnTy2, A2> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        assert_eq!(el.data.len(), self.rank());
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        result.extend((0..self.rank()).map(|i| self.base_ring().get_ring().map_in(from.base_ring().get_ring(), from.base_ring().clone_el(&el.data[i]), hom)));
        return DecompositionRingEl {
            data: result,
            allocator: PhantomData,
            number_ring: PhantomData
        };
    }
}

impl<NumberRing, ZnTy1, ZnTy2, A1, A2> CanIsoFromTo<DecompositionRingBase<NumberRing, ZnTy2, A2>> for DecompositionRingBase<NumberRing, ZnTy1, A1>
    where NumberRing: HENumberRing + HENumberRing,
        ZnTy1: RingStore,
        ZnTy1::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A1: Allocator + Clone,
        ZnTy2: RingStore,
        ZnTy2::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A2: Allocator + Clone,
        ZnTy1::Type: CanIsoFromTo<ZnTy2::Type>
{
    type Isomorphism = <ZnTy1::Type as CanIsoFromTo<ZnTy2::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &DecompositionRingBase<NumberRing, ZnTy2, A2>) -> Option<Self::Isomorphism> {
        if self.number_ring == from.number_ring {
            self.base_ring().get_ring().has_canonical_iso(from.base_ring().get_ring())
        } else {
            None
        }
    }

    fn map_out(&self, from: &DecompositionRingBase<NumberRing, ZnTy2, A2>, el: Self::Element, iso: &Self::Isomorphism) -> <DecompositionRingBase<NumberRing, ZnTy2, A2> as RingBase>::Element {
        assert_eq!(el.data.len(), self.rank());
        let mut result = Vec::with_capacity_in(self.rank(), from.allocator.clone());
        result.extend((0..self.rank()).map(|i| self.base_ring().get_ring().map_out(from.base_ring().get_ring(), self.base_ring().clone_el(&el.data[i]), iso)));
        return DecompositionRingEl {
            data: result,
            allocator: PhantomData,
            number_ring: PhantomData
        };
    }
}

#[cfg(test)]
pub fn test_with_number_ring<NumberRing: HENumberRing>(number_ring: NumberRing) {
    let base_ring = zn_64::Zn::new(5);
    let rank = number_ring.rank();
    let ring = DecompositionRingBase::new(number_ring, base_ring);

    let elements = vec![
        ring.zero(),
        ring.one(),
        ring.neg_one(),
        ring.int_hom().map(2),
        ring.canonical_gen(),
        ring.pow(ring.canonical_gen(), 2),
        ring.pow(ring.canonical_gen(), rank - 1),
        ring.int_hom().mul_map(ring.canonical_gen(), 2),
        ring.int_hom().mul_map(ring.pow(ring.canonical_gen(), rank - 1), 2)
    ];

    feanor_math::ring::generic_tests::test_ring_axioms(&ring, elements.iter().map(|x| ring.clone_el(x)));
    feanor_math::ring::generic_tests::test_self_iso(&ring, elements.iter().map(|x| ring.clone_el(x)));
    feanor_math::rings::extension::generic_tests::test_free_algebra_axioms(&ring);
}

#[test]
pub fn test_decomposition_ring_large_modulus() {
    let number_ring = Pow2CyclotomicNumberRing::new(32);
    let rank = number_ring.rank();
    let ring = DecompositionRingBase::new(number_ring, zn_big::Zn::new(BigIntRing::RING, BigIntRing::RING.get_ring().parse("1267650600228229401496703205653", 10).unwrap()));
    
    let large_base_ring_element = ring.base_ring().coerce(&BigIntRing::RING, BigIntRing::RING.power_of_two(51));
    let elements = vec![
        ring.zero(),
        ring.one(),
        ring.neg_one(),
        ring.inclusion().map_ref(&large_base_ring_element),
        ring.canonical_gen(),
        ring.pow(ring.canonical_gen(), 2),
        ring.pow(ring.canonical_gen(), rank - 1),
        ring.inclusion().mul_ref_map(&ring.canonical_gen(), &large_base_ring_element),
        ring.inclusion().mul_ref_map(&ring.pow(ring.canonical_gen(), rank - 1), &large_base_ring_element)
    ];
    
    feanor_math::ring::generic_tests::test_ring_axioms(&ring, elements.iter().map(|x| ring.clone_el(x)));
    feanor_math::ring::generic_tests::test_self_iso(&ring, elements.iter().map(|x| ring.clone_el(x)));
    feanor_math::rings::extension::generic_tests::test_free_algebra_axioms(&ring);
}