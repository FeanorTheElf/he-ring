use std::alloc::{Allocator, Global};
use std::marker::PhantomData;

use feanor_math::homomorphism::{CanHomFrom, CanIsoFromTo, Homomorphism};
use feanor_math::integer::{int_cast, BigIntRing, BigIntRingBase, IntegerRing, IntegerRingStore};
use feanor_math::iters::{multi_cartesian_product, MultiProduct};
use feanor_math::primitive_int::StaticRing;
use feanor_math::rings::extension::{FreeAlgebra, FreeAlgebraStore};
use feanor_math::rings::finite::FiniteRing;
use feanor_math::rings::poly::dense_poly::DensePolyRing;
use feanor_math::rings::zn::{zn_rns, zn_64, ZnRing, ZnRingStore};
use feanor_math::ring::*;
use feanor_math::seq::{CloneElFn, CloneRingEl};
use feanor_math::seq::VectorView;
use feanor_math::serialization::{DeserializeWithRing, SerializableElementRing, SerializeWithRing};
use feanor_math::ordered::OrderedRingStore;

use serde::{de, Deserializer, Serializer};

use super::decomposition::{CyclotomicRingDecomposition, IsomorphismInfo, RingDecompositionSelfIso};

pub struct NTTRingBase<R, F, A = Global> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    ring_decompositions: Vec<F>,
    base_ring: R,
    rns_base: zn_rns::Zn<R, BigIntRing>,
    allocator: A
}

pub type NTTRing<R, F, A = Global> = RingValue<NTTRingBase<R, F, A>>;

pub struct NTTRingEl<R, F, A = Global> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    pub(super) ring_decompositions: PhantomData<F>,
    pub(super) allocator: PhantomData<A>,
    pub(super) data: Vec<El<R>, A>
}

impl<R, F, A> NTTRingBase<R, F, A>
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    pub fn from_ring_decompositions(base_ring: R, rns_base: zn_rns::Zn<R, BigIntRing>, ring_decompositions: Vec<F>, allocator: A) -> Self {
        assert!(ring_decompositions.len() > 0);
        for i in 0..ring_decompositions.len() {
            assert!(ring_decompositions[i].is_same_number_ring(&ring_decompositions[0]));
            assert_eq!(ring_decompositions[i].rank(), ring_decompositions[0].rank());
            assert_eq!(ring_decompositions[i].expansion_factor(), ring_decompositions[0].expansion_factor());
        }
        let modulus = int_cast(base_ring.integer_ring().clone_el(base_ring.modulus()), StaticRing::<i64>::RING, base_ring.integer_ring());
        assert!(BigIntRing::RING.is_geq(rns_base.modulus(), &BigIntRing::RING.prod([ring_decompositions[0].expansion_factor(), 2, ((modulus - 1) / 2 + 1), ((modulus - 1) / 2 + 1)].into_iter().map(|n| BigIntRing::RING.coerce(&StaticRing::<i64>::RING, n)))));
        Self { base_ring, rns_base, ring_decompositions, allocator }
    }

    pub fn allocator(&self) -> &A {
        &self.allocator
    }

    pub fn ring_decompositions(&self) -> &Vec<F> {
        &self.ring_decompositions
    }

    pub fn rns_base(&self) -> &zn_rns::ZnBase<R, BigIntRing> {
        self.rns_base.get_ring()
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
        where F: CyclotomicRingDecomposition<R::Type>,
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
                self.ring_decompositions[i].fft_forward(&mut tmp[..], Zp.get_ring());
                self.ring_decompositions[i].permute_galois_action(&tmp, &mut tmp_perm, g, self.base_ring());
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
            let Zp = self.rns_base.at(i);
            self.ring_decompositions[i].fft_backward(&mut unreduced_result[(i * self.rank())..((i + 1) * self.rank())], Zp.get_ring());
        }

        let hom = self.base_ring().can_hom(&BigIntRing::RING).unwrap();
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        for j in 0..self.rank() {
            result.push(hom.map(self.rns_base.smallest_lift(
                self.rns_base.from_congruence((0..self.rns_base.len()).map(|i| self.rns_base.at(i).clone_el(&unreduced_result[i * self.rank() + j])))
            )));
        }
        return NTTRingEl {
            data: result,
            ring_decompositions: PhantomData,
            allocator: PhantomData
        };
    }
}

impl<R, F, A> PartialEq for NTTRingBase<R, F, A>
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    fn eq(&self, other: &Self) -> bool {
        self.base_ring().get_ring() == other.base_ring().get_ring() && self.ring_decompositions[0].is_same_number_ring(&other.ring_decompositions[0])
    }
}

impl<R, F, A> RingBase for NTTRingBase<R, F, A>
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    type Element = NTTRingEl<R, F, A>;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        result.extend(val.data.iter().map(|x| self.base_ring().clone_el(x)));
        return NTTRingEl {
            data: result,
            ring_decompositions: PhantomData,
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
            self.ring_decompositions[i].fft_forward(&mut lhs_tmp[..], Zp.get_ring());
            self.ring_decompositions[i].fft_forward(&mut rhs_tmp[..], Zp.get_ring());
            let end_index = unreduced_result.len();
            unreduced_result.extend((0..self.rank()).map(|j| Zp.mul_ref(&lhs_tmp[j], &rhs_tmp[j])));
            self.ring_decompositions[i].fft_backward(&mut unreduced_result[end_index..], Zp.get_ring());
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
        return NTTRingEl {
            data: result,
            ring_decompositions: PhantomData,
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
        return NTTRingEl {
            data: result,
            ring_decompositions: PhantomData,
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

impl<R, F, A> RingExtension for NTTRingBase<R, F, A>
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    type BaseRing = R;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing {
        &self.base_ring
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        let mut result = self.zero();
        result.data[0] = x;
        return result;
    }
}

impl<R, F, A> CyclotomicRing for NTTRingBase<R, F, A>
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type> + CyclotomicRingDecomposition<R::Type>,
        A: Allocator + Clone
{
    fn n(&self) -> usize {
        *self.ring_decompositions()[0].galois_group_mulrepr().modulus() as usize
    }

    fn apply_galois_action(&self, el: &Self::Element, g: zn_64::ZnEl) -> Self::Element {
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
            self.ring_decompositions[i].fft_forward(&mut tmp[..], Zp.get_ring());
            self.ring_decompositions[i].permute_galois_action(&tmp, &mut unreduced_result[(i * self.rank())..((i + 1) * self.rank())], g, self.base_ring());
            self.ring_decompositions[i].fft_backward(&mut unreduced_result[(i * self.rank())..((i + 1) * self.rank())], Zp.get_ring());
        }
        drop(tmp);

        let hom = self.base_ring().can_hom(&BigIntRing::RING).unwrap();
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        for j in 0..self.rank() {
            result.push(hom.map(self.rns_base.smallest_lift(
                self.rns_base.from_congruence((0..self.rns_base.len()).map(|i| self.rns_base.at(i).clone_el(&unreduced_result[i * self.rank() + j])))
            )));
        }
        return NTTRingEl {
            data: result,
            ring_decompositions: PhantomData,
            allocator: PhantomData
        };
    }
}

impl<R, F, A> FreeAlgebra for NTTRingBase<R, F, A>
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    type VectorRepresentation<'a> = CloneElFn<&'a [El<R>], El<R>, CloneRingEl<&'a R>>
        where Self: 'a;

    fn from_canonical_basis<V>(&self, vec: V) -> Self::Element
        where V: IntoIterator<Item = El<Self::BaseRing>>
    {
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        result.extend(vec);
        assert_eq!(result.len(), self.rank());
        return NTTRingEl {
            data: result,
            ring_decompositions: PhantomData,
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

pub struct WRTCanonicalBasisElementCreator<'a, R, F, A>
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    ring: &'a NTTRingBase<R, F, A>,
}

impl<'a, R, F, A> Copy for WRTCanonicalBasisElementCreator<'a, R, F, A>
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{}

impl<'a, R, F, A> Clone for WRTCanonicalBasisElementCreator<'a, R, F, A>
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, 'b, R, F, A> FnOnce<(&'b [El<R>],)> for WRTCanonicalBasisElementCreator<'a, R, F, A>
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    type Output = El<NTTRing<R, F, A>>;

    extern "rust-call" fn call_once(self, args: (&'b [El<R>],)) -> Self::Output {
        self.call(args)
    }
}

impl<'a, 'b, R, F, A> FnMut<(&'b [El<R>],)> for WRTCanonicalBasisElementCreator<'a, R, F, A>
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    extern "rust-call" fn call_mut(&mut self, args: (&'b [El<R>],)) -> Self::Output {
        self.call(args)
    }
}

impl<'a, 'b, R, F, A> Fn<(&'b [El<R>],)> for WRTCanonicalBasisElementCreator<'a, R, F, A>
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    extern "rust-call" fn call(&self, args: (&'b [El<R>],)) -> Self::Output {
        self.ring.from_canonical_basis(args.0.iter().map(|x| self.ring.base_ring().clone_el(x)))
    }
}

impl<R, F, A> FiniteRing for NTTRingBase<R, F, A>
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F: RingDecompositionSelfIso<R::Type>,
        A: Allocator + Clone
{
    type ElementsIter<'a> = MultiProduct<<R::Type as FiniteRing>::ElementsIter<'a>, WRTCanonicalBasisElementCreator<'a, R, F, A>, CloneRingEl<&'a R>, Self::Element>
        where Self: 'a;

    fn elements<'a>(&'a self) -> Self::ElementsIter<'a> {
        multi_cartesian_product((0..self.rank()).map(|_| self.base_ring().elements()), WRTCanonicalBasisElementCreator { ring: self }, CloneRingEl(self.base_ring()))
    }

    fn random_element<G: FnMut() -> u64>(&self, mut rng: G) -> Self::Element {
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        result.extend((0..self.rank()).map(|_| self.base_ring().random_element(&mut rng)));
        return NTTRingEl {
            data: result,
            allocator: PhantomData,
            ring_decompositions: PhantomData
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

impl<R, F, A> SerializableElementRing for NTTRingBase<R, F, A>
    where R: ZnRingStore,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + SerializableElementRing,
        F: RingDecompositionSelfIso<R::Type>,
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
        return Ok(NTTRingEl {
            data: result,
            ring_decompositions: PhantomData,
            allocator: PhantomData
        });
    }
}

impl<R1, R2, F1, F2, A1, A2> CanHomFrom<NTTRingBase<R2, F2, A2>> for NTTRingBase<R1, F1, A1>
    where R1: ZnRingStore,
        R1::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F1: RingDecompositionSelfIso<R1::Type>,
        A1: Allocator + Clone,
        R2: ZnRingStore,
        R2::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F2: RingDecompositionSelfIso<R2::Type>,
        A2: Allocator + Clone,
        R1::Type: CanHomFrom<R2::Type>,
        F1: IsomorphismInfo<R1::Type, R2::Type, F2>
{
    type Homomorphism = <R1::Type as CanHomFrom<R2::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &NTTRingBase<R2, F2, A2>) -> Option<Self::Homomorphism> {
        // since we store elements by their canonical basis representation, `is_same_number_ring()` together
        // with compatibility of base rings is enough here
        if self.ring_decompositions[0].is_same_number_ring(&from.ring_decompositions[0]) {
            self.base_ring().get_ring().has_canonical_hom(from.base_ring().get_ring())
        } else {
            None
        }
    }

    fn map_in(&self, from: &NTTRingBase<R2, F2, A2>, el: <NTTRingBase<R2, F2, A2> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        assert_eq!(el.data.len(), self.rank());
        let mut result = Vec::with_capacity_in(self.rank(), self.allocator.clone());
        result.extend((0..self.rank()).map(|i| self.base_ring().get_ring().map_in(from.base_ring().get_ring(), from.base_ring().clone_el(&el.data[i]), hom)));
        return NTTRingEl {
            data: result,
            allocator: PhantomData,
            ring_decompositions: PhantomData
        };
    }
}

impl<R1, R2, F1, F2, A1, A2> CanIsoFromTo<NTTRingBase<R2, F2, A2>> for NTTRingBase<R1, F1, A1>
    where R1: ZnRingStore,
        R1::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F1: RingDecompositionSelfIso<R1::Type>,
        A1: Allocator + Clone,
        R2: ZnRingStore,
        R2::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        F2: RingDecompositionSelfIso<R2::Type>,
        A2: Allocator + Clone,
        R1::Type: CanIsoFromTo<R2::Type>,
        F1: IsomorphismInfo<R1::Type, R2::Type, F2>
{
    type Isomorphism = <R1::Type as CanIsoFromTo<R2::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &NTTRingBase<R2, F2, A2>) -> Option<Self::Isomorphism> {
        // since we store elements by their canonical basis representation, `is_same_number_ring()` together
        // with compatibility of base rings is enough here
        if self.ring_decompositions[0].is_same_number_ring(&from.ring_decompositions[0]) {
            self.base_ring().get_ring().has_canonical_iso(from.base_ring().get_ring())
        } else {
            None
        }
    }

    fn map_out(&self, from: &NTTRingBase<R2, F2, A2>, el: Self::Element, iso: &Self::Isomorphism) -> <NTTRingBase<R2, F2, A2> as RingBase>::Element {
        assert_eq!(el.data.len(), self.rank());
        let mut result = Vec::with_capacity_in(self.rank(), from.allocator.clone());
        result.extend((0..self.rank()).map(|i| self.base_ring().get_ring().map_out(from.base_ring().get_ring(), self.base_ring().clone_el(&el.data[i]), iso)));
        return NTTRingEl {
            data: result,
            allocator: PhantomData,
            ring_decompositions: PhantomData
        };
    }
}

#[cfg(test)]
use feanor_math::algorithms::fft::cooley_tuckey;
use crate::cyclotomic::CyclotomicRing;
#[cfg(test)]
use crate::rings::pow2_cyclotomic::Pow2CyclotomicFFT;
#[cfg(test)]
use feanor_math::algorithms::unity_root::get_prim_root_of_unity_pow2;
#[cfg(test)]
use feanor_math::rings::field::{AsField, AsFieldBase};
#[cfg(test)]
use feanor_math::homomorphism::CanHom;

#[cfg(test)]
fn test_ring_and_elements() -> (
    NTTRing<zn_64::Zn, Pow2CyclotomicFFT<zn_64::Zn, cooley_tuckey::CooleyTuckeyFFT<zn_64::ZnBase, AsFieldBase<zn_64::Zn>, CanHom<AsField<zn_64::Zn>, zn_64::Zn>>>>,
    Vec<NTTRingEl<zn_64::Zn, Pow2CyclotomicFFT<zn_64::Zn, cooley_tuckey::CooleyTuckeyFFT<zn_64::ZnBase, AsFieldBase<zn_64::Zn>, CanHom<AsField<zn_64::Zn>, zn_64::Zn>>>>>
) {
    let rns_base = zn_rns::Zn::new([113, 193, 241, 257, 337].into_iter().map(zn_64::Zn::new).collect(), BigIntRing::RING);
    let mut generalized_ffts = Vec::new();
    for Fp in rns_base.as_iter() {
        let as_field = (*Fp).as_field().ok().unwrap();
        let root_of_unity = get_prim_root_of_unity_pow2(as_field, 4).unwrap();
        let hom = (*Fp).into_can_hom(as_field).ok().unwrap();
        let root_of_unity_Fp = hom.map(root_of_unity);
        generalized_ffts.push(Pow2CyclotomicFFT::create(*Fp, cooley_tuckey::CooleyTuckeyFFT::new_with_hom(hom, as_field.pow(root_of_unity, 2), 3), root_of_unity_Fp));
    }

    let test_ring = RingValue::from(NTTRingBase::from_ring_decompositions(
        zn_64::Zn::new(65536),
        rns_base,
        generalized_ffts,
        Global
    ));

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
