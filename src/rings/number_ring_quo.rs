use std::alloc::{Allocator, Global};
use std::marker::PhantomData;

use feanor_math::homomorphism::{CanHomFrom, CanIsoFromTo, Homomorphism};
use feanor_math::integer::{int_cast, BigIntRing, BigIntRingBase, IntegerRing, IntegerRingStore};
use feanor_math::iters::{multi_cartesian_product, MultiProduct};
use feanor_math::primitive_int::StaticRing;
use feanor_math::rings::extension::{FreeAlgebra, FreeAlgebraStore};
use feanor_math::rings::finite::FiniteRing;
use feanor_math::rings::poly::dense_poly::DensePolyRing;
use feanor_math::rings::zn::{zn_64, zn_rns, FromModulusCreateableZnRing, ZnRing, ZnRingStore};
use feanor_math::ring::*;
use feanor_math::seq::{CloneElFn, CloneRingEl};
use feanor_math::seq::VectorView;
use feanor_math::serialization::{DeserializeWithRing, SerializableElementRing, SerializeWithRing};
use feanor_math::ordered::OrderedRingStore;
use feanor_math::rings::finite::*;

use serde::{de, Deserializer, Serializer};
use serde_json::Number;

use crate::cyclotomic::CyclotomicRing;
use crate::sample_primes;
use crate::IsEq;
use crate::rings::decomposition::*;

pub struct NumberRingQuoBase<NumberRing, FpTy, A = Global> 
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    number_ring: NumberRing,
    base_ring: FpTy,
    ring_decompositions: Vec<<NumberRing as DecomposableNumberRing<FpTy>>::Decomposed>,
    rns_base: zn_rns::Zn<FpTy, BigIntRing>,
    allocator: A
}

pub type NumberRingQuo<NumberRing, FpTy, A = Global> = RingValue<NumberRingQuoBase<NumberRing, FpTy, A>>;

pub struct NumberRingQuoEl<NumberRing, FpTy, A = Global> 
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    pub(super) number_ring: PhantomData<NumberRing>,
    pub(super) allocator: PhantomData<A>,
    pub(super) data: Vec<El<FpTy>, A>
}

impl<NumberRing, FpTy> NumberRingQuoBase<NumberRing, RingValue<FpTy>, Global>
    where NumberRing: DecomposableNumberRing<RingValue<FpTy>>,
        FpTy: Clone + ZnRing + CanHomFrom<BigIntRingBase> + FromModulusCreateableZnRing
{
    pub fn new(number_ring: NumberRing, base_ring: RingValue<FpTy>) -> RingValue<Self> {
        let max_product_expansion_factor = number_ring.product_expansion_factor();
        let max_lift_size = int_cast(base_ring.integer_ring().clone_el(base_ring.modulus()), StaticRing::<i64>::RING, base_ring.integer_ring()) as f64 / 2.;
        let max_product_size = max_lift_size * max_lift_size * max_product_expansion_factor;
        let required_bits = max_product_size.log2().ceil() as usize;
        let rns_base_primes = sample_primes(required_bits, required_bits + 10, 57, |n| number_ring.largest_suitable_prime(int_cast(n, StaticRing::<i64>::RING, BigIntRing::RING)).map(|n| int_cast(n, BigIntRing::RING, StaticRing::<i64>::RING))).unwrap();
        let rns_base = zn_rns::Zn::new(rns_base_primes.into_iter().map(|p| RingValue::from(FpTy::create::<_, !>(|ZZ| Ok(int_cast(p, RingRef::new(ZZ), BigIntRing::RING))).unwrap())).collect(), BigIntRing::RING);
        return Self::new_with(number_ring, base_ring, rns_base, Global);
    }
}

impl<NumberRing, FpTy, A> NumberRingQuoBase<NumberRing, FpTy, A>
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    pub fn new_with(number_ring: NumberRing, base_ring: FpTy, rns_base: zn_rns::Zn<FpTy, BigIntRing>, allocator: A) -> RingValue<Self> {
        assert!(rns_base.len() > 0);
        let max_product_expansion_factor = number_ring.product_expansion_factor();
        let max_lift_size = int_cast(base_ring.integer_ring().clone_el(base_ring.modulus()), StaticRing::<i64>::RING, base_ring.integer_ring()) as f64 / 2.;
        let max_product_size = max_lift_size * max_lift_size * max_product_expansion_factor;
        let ZZbig = BigIntRing::RING;
        assert!(ZZbig.is_gt(&ZZbig.prod(rns_base.as_iter().map(|rns_base_ring| int_cast(rns_base_ring.integer_ring().clone_el(rns_base_ring.modulus()), ZZbig, rns_base_ring.integer_ring()))), &ZZbig.from_float_approx(max_product_size).unwrap()));
        RingValue::from(Self {
            base_ring: base_ring,
            ring_decompositions: rns_base.as_iter().map(|Fp| number_ring.mod_p(Fp.clone())).collect(),
            number_ring: number_ring,
            rns_base: rns_base,
            allocator: allocator
        })
    }

    pub fn allocator(&self) -> &A {
        &self.allocator
    }

    pub fn ring_decompositions(&self) -> &[<NumberRing as DecomposableNumberRing<FpTy>>::Decomposed] {
        &self.ring_decompositions
    }

    pub fn rns_base(&self) -> &zn_rns::Zn<FpTy, BigIntRing> {
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
        where NumberRing: DecomposableCyclotomicNumberRing<FpTy>,
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
                self.ring_decompositions[i].fft_forward(&mut tmp[..]);
                <_ as DecomposedCyclotomicNumberRing<_>>::permute_galois_action(<NumberRing::DecomposedAsCyclotomic>::from_ref(&self.ring_decompositions[i]), &tmp, &mut tmp_perm, g);
                for j in 0..self.rank() {
                    Zp.add_assign_ref(&mut unreduced_result[i * self.rank() + j], &tmp_perm[j]);
                }
            }
        }
        drop(tmp);
        drop(tmp_perm);

        // if this is satisfied, we have enough precision to not get an overflow
        assert!(len < int_cast(self.rns_base.integer_ring().clone_el(self.rns_base.modulus()), StaticRing::<i64>::RING, self.rns_base.integer_ring()) as usize);
        for i in 0..self.rns_base.len() {
            self.ring_decompositions[i].fft_backward(&mut unreduced_result[(i * self.rank())..((i + 1) * self.rank())]);
        }

        let hom = self.base_ring().can_hom(&BigIntRing::RING).unwrap();
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        for j in 0..self.rank() {
            result.push(hom.map(self.rns_base.smallest_lift(
                self.rns_base.from_congruence((0..self.rns_base.len()).map(|i| self.rns_base.at(i).clone_el(&unreduced_result[i * self.rank() + j])))
            )));
        }
        return NumberRingQuoEl {
            data: result,
            number_ring: PhantomData,
            allocator: PhantomData
        };
    }
}

impl<NumberRing, FpTy, A> CyclotomicRing for NumberRingQuoBase<NumberRing, FpTy, A>
    where NumberRing: DecomposableCyclotomicNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    fn n(&self) -> usize {
        *<NumberRing::DecomposedAsCyclotomic>::from_ref(&self.ring_decompositions()[0]).cyclotomic_index_ring().modulus() as usize
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
            self.ring_decompositions[i].fft_forward(&mut tmp[..]);
            <NumberRing::DecomposedAsCyclotomic>::from_ref(&self.ring_decompositions()[i]).permute_galois_action(&tmp, &mut unreduced_result[(i * self.rank())..((i + 1) * self.rank())], g);
            self.ring_decompositions[i].fft_backward(&mut unreduced_result[(i * self.rank())..((i + 1) * self.rank())]);
        }
        drop(tmp);

        let hom = self.base_ring().can_hom(&BigIntRing::RING).unwrap();
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        for j in 0..self.rank() {
            result.push(hom.map(self.rns_base.smallest_lift(
                self.rns_base.from_congruence((0..self.rns_base.len()).map(|i| self.rns_base.at(i).clone_el(&unreduced_result[i * self.rank() + j])))
            )));
        }
        return NumberRingQuoEl {
            data: result,
            number_ring: PhantomData,
            allocator: PhantomData
        };
    }
}

impl<NumberRing, FpTy, A> PartialEq for NumberRingQuoBase<NumberRing, FpTy, A>
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    fn eq(&self, other: &Self) -> bool {
        self.number_ring == other.number_ring && self.base_ring.get_ring() == other.base_ring.get_ring()
    }
}

impl<NumberRing, FpTy, A> RingBase for NumberRingQuoBase<NumberRing, FpTy, A>
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    type Element = NumberRingQuoEl<NumberRing, FpTy, A>;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        result.extend(val.data.iter().map(|x| self.base_ring().clone_el(x)));
        return NumberRingQuoEl {
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
            self.ring_decompositions[i].fft_forward(&mut lhs_tmp[..]);
            self.ring_decompositions[i].fft_forward(&mut rhs_tmp[..]);
            let end_index = unreduced_result.len();
            unreduced_result.extend((0..self.rank()).map(|j| Zp.mul_ref(&lhs_tmp[j], &rhs_tmp[j])));
            self.ring_decompositions[i].fft_backward(&mut unreduced_result[end_index..]);
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
        return NumberRingQuoEl {
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
        return NumberRingQuoEl {
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

    fn characteristic<I: IntegerRingStore>(&self, ZZ: &I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        self.base_ring.characteristic(ZZ)
    }
}

impl<NumberRing, FpTy, A> RingExtension for NumberRingQuoBase<NumberRing, FpTy, A>
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    type BaseRing = FpTy;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing {
        &self.base_ring
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        let mut result = self.zero();
        result.data[0] = x;
        return result;
    }
}

impl<NumberRing, FpTy, A> FreeAlgebra for NumberRingQuoBase<NumberRing, FpTy, A>
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    type VectorRepresentation<'a> = CloneElFn<&'a [El<FpTy>], El<FpTy>, CloneRingEl<&'a FpTy>>
        where Self: 'a;

    fn from_canonical_basis<V>(&self, vec: V) -> Self::Element
        where V: IntoIterator<Item = El<Self::BaseRing>>
    {
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        result.extend(vec);
        assert_eq!(result.len(), self.rank());
        return NumberRingQuoEl {
            data: result,
            number_ring: PhantomData,
            allocator: PhantomData
        };
    }

    fn wrt_canonical_basis<'a>(&'a self, el: &'a Self::Element) -> Self::VectorRepresentation<'a> {
        (&el.data[..]).into_ring_el_fn(self.base_ring())
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

pub struct WRTCanonicalBasisElementCreator<'a, NumberRing, FpTy, A>
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    ring: &'a NumberRingQuoBase<NumberRing, FpTy, A>,
}

impl<'a, NumberRing, FpTy, A> Copy for WRTCanonicalBasisElementCreator<'a, NumberRing, FpTy, A>
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{}

impl<'a, NumberRing, FpTy, A> Clone for WRTCanonicalBasisElementCreator<'a, NumberRing, FpTy, A>
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, 'b, NumberRing, FpTy, A> FnOnce<(&'b [El<FpTy>],)> for WRTCanonicalBasisElementCreator<'a, NumberRing, FpTy, A>
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    type Output = El<NumberRingQuo<NumberRing, FpTy, A>>;

    extern "rust-call" fn call_once(self, args: (&'b [El<FpTy>],)) -> Self::Output {
        self.call(args)
    }
}

impl<'a, 'b, NumberRing, FpTy, A> FnMut<(&'b [El<FpTy>],)> for WRTCanonicalBasisElementCreator<'a, NumberRing, FpTy, A>
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    extern "rust-call" fn call_mut(&mut self, args: (&'b [El<FpTy>],)) -> Self::Output {
        self.call(args)
    }
}

impl<'a, 'b, NumberRing, FpTy, A> Fn<(&'b [El<FpTy>],)> for WRTCanonicalBasisElementCreator<'a, NumberRing, FpTy, A>
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    extern "rust-call" fn call(&self, args: (&'b [El<FpTy>],)) -> Self::Output {
        self.ring.from_canonical_basis(args.0.iter().map(|x| self.ring.base_ring().clone_el(x)))
    }
}

impl<NumberRing, FpTy, A> FiniteRing for NumberRingQuoBase<NumberRing, FpTy, A>
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A: Allocator + Clone
{
    type ElementsIter<'a> = MultiProduct<<FpTy::Type as FiniteRing>::ElementsIter<'a>, WRTCanonicalBasisElementCreator<'a, NumberRing, FpTy, A>, CloneRingEl<&'a FpTy>, Self::Element>
        where Self: 'a;

    fn elements<'a>(&'a self) -> Self::ElementsIter<'a> {
        multi_cartesian_product((0..self.rank()).map(|_| self.base_ring().elements()), WRTCanonicalBasisElementCreator { ring: self }, CloneRingEl(self.base_ring()))
    }

    fn random_element<G: FnMut() -> u64>(&self, mut rng: G) -> Self::Element {
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        result.extend((0..self.rank()).map(|_| self.base_ring().random_element(&mut rng)));
        return NumberRingQuoEl {
            data: result,
            allocator: PhantomData,
            number_ring: PhantomData
        };
    }

    fn size<I: IntegerRingStore>(&self, ZZ: &I) -> Option<El<I>>
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

impl<NumberRing, FpTy, A> SerializableElementRing for NumberRingQuoBase<NumberRing, FpTy, A>
    where NumberRing: DecomposableNumberRing<FpTy>,
        FpTy: RingStore + Clone,
        FpTy::Type: ZnRing + CanHomFrom<BigIntRingBase> + SerializableElementRing,
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
        return Ok(NumberRingQuoEl {
            data: result,
            number_ring: PhantomData,
            allocator: PhantomData
        });
    }
}

impl<NumberRing, FpTy1, FpTy2, A1, A2> CanHomFrom<NumberRingQuoBase<NumberRing, FpTy2, A2>> for NumberRingQuoBase<NumberRing, FpTy1, A1>
    where NumberRing: DecomposableNumberRing<FpTy1> + DecomposableNumberRing<FpTy2>,
        FpTy1: RingStore + Clone,
        FpTy1::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A1: Allocator + Clone,
        FpTy2: RingStore + Clone,
        FpTy2::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A2: Allocator + Clone,
        FpTy1::Type: CanHomFrom<FpTy2::Type>
{
    type Homomorphism = <FpTy1::Type as CanHomFrom<FpTy2::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &NumberRingQuoBase<NumberRing, FpTy2, A2>) -> Option<Self::Homomorphism> {
        if self.number_ring == from.number_ring {
            self.base_ring().get_ring().has_canonical_hom(from.base_ring().get_ring())
        } else {
            None
        }
    }

    fn map_in(&self, from: &NumberRingQuoBase<NumberRing, FpTy2, A2>, el: <NumberRingQuoBase<NumberRing, FpTy2, A2> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        assert_eq!(el.data.len(), self.rank());
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        result.extend((0..self.rank()).map(|i| self.base_ring().get_ring().map_in(from.base_ring().get_ring(), from.base_ring().clone_el(&el.data[i]), hom)));
        return NumberRingQuoEl {
            data: result,
            allocator: PhantomData,
            number_ring: PhantomData
        };
    }
}

impl<NumberRing, FpTy1, FpTy2, A1, A2> CanIsoFromTo<NumberRingQuoBase<NumberRing, FpTy2, A2>> for NumberRingQuoBase<NumberRing, FpTy1, A1>
    where NumberRing: DecomposableNumberRing<FpTy1> + DecomposableNumberRing<FpTy2>,
        FpTy1: RingStore + Clone,
        FpTy1::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A1: Allocator + Clone,
        FpTy2: RingStore + Clone,
        FpTy2::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        A2: Allocator + Clone,
        FpTy1::Type: CanIsoFromTo<FpTy2::Type>
{
    type Isomorphism = <FpTy1::Type as CanIsoFromTo<FpTy2::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &NumberRingQuoBase<NumberRing, FpTy2, A2>) -> Option<Self::Isomorphism> {
        if self.number_ring == from.number_ring {
            self.base_ring().get_ring().has_canonical_iso(from.base_ring().get_ring())
        } else {
            None
        }
    }

    fn map_out(&self, from: &NumberRingQuoBase<NumberRing, FpTy2, A2>, el: Self::Element, iso: &Self::Isomorphism) -> <NumberRingQuoBase<NumberRing, FpTy2, A2> as RingBase>::Element {
        assert_eq!(el.data.len(), self.rank());
        let mut result = Vec::with_capacity_in(self.rank(), from.allocator.clone());
        result.extend((0..self.rank()).map(|i| self.base_ring().get_ring().map_out(from.base_ring().get_ring(), self.base_ring().clone_el(&el.data[i]), iso)));
        return NumberRingQuoEl {
            data: result,
            allocator: PhantomData,
            number_ring: PhantomData
        };
    }
}

#[cfg(test)]
use super::pow2_cyclotomic::Pow2CyclotomicDecomposableNumberRing;
#[cfg(test)]
use feanor_math::algorithms::fft::cooley_tuckey::CooleyTuckeyFFT;
#[cfg(test)]
use feanor_math::algorithms::unity_root::get_prim_root_of_unity_pow2;

#[cfg(test)]
fn test_ring_and_elements() -> (
    NumberRingQuo<Pow2CyclotomicDecomposableNumberRing, zn_64::Zn>,
    Vec<NumberRingQuoEl<Pow2CyclotomicDecomposableNumberRing, zn_64::Zn>>
) {
    let rns_base = zn_rns::Zn::new([113, 193, 241, 257, 337].into_iter().map(zn_64::Zn::new).collect(), BigIntRing::RING);
    let test_ring = NumberRingQuoBase::new_with(
        Pow2CyclotomicDecomposableNumberRing::new(16),
        zn_64::Zn::new(65536),
        rns_base,
        Global
    );
    assert_eq!(8, test_ring.rank());

    let mut test_elements = Vec::new();
    for i in [0, 1, 2, 4, 7] {
        for j in [0, 1] {
            for a in [-1, 1, 32768] {
                test_elements.push(test_ring.from_canonical_basis((0..8).map(|k| if k == i { test_ring.base_ring().int_hom().map(a) } else if k == j { test_ring.base_ring().one() } else { test_ring.base_ring().zero() })));
            }
        }
    }

    return (test_ring, test_elements);
}

#[test]
fn test_ring_axioms() {
    let (ring, els) = test_ring_and_elements();
    feanor_math::ring::generic_tests::test_ring_axioms(ring, els.into_iter());
}

#[test]
fn test_nilpotent() {
    let (ring, _els) = test_ring_and_elements();
    let two_x = ring.int_hom().mul_map(ring.canonical_gen(), 2);
    for i in 0..15 {
        assert!(!ring.is_zero(&ring.pow(ring.clone_el(&two_x), i)));
    }
    assert!(ring.is_zero(&ring.pow(ring.clone_el(&two_x), 16)));
}

#[test]
fn test_free_algebra_axioms() {
    let (ring, _els) = test_ring_and_elements();
    feanor_math::rings::extension::generic_tests::test_free_algebra_axioms(ring);
}
