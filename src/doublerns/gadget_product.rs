use feanor_math::integer::BigIntRingBase;
use feanor_math::mempool::*;
use feanor_math::ring::*;
use feanor_math::homomorphism::*;
use feanor_math::rings::zn::*;
use feanor_math::rings::extension::*;
use feanor_math::vector::*;

use super::double_rns_ring::*;

///
/// Right-hand side operand of an "RNS-gadget product", hence this struct stores
/// noisy approximations to `lift((q / pi)^-1 mod pi) * q / pi * x`, where `q = p1 * ... * pm` is the ring modulus.
/// For more details, see [`DoubleRNSRingBase::gadget_product()`].
/// 
pub struct GadgetProductRhsOperand<R, F, M> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanIsoFromTo<F::BaseRingBase> + CanHomFrom<BigIntRingBase>,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    operands: Vec<<DoubleRNSRingBase<R, F, M> as RingBase>::Element>
}

impl<R, F, M> GadgetProductRhsOperand<R, F, M>
    where R: ZnRingStore,
        R::Type: ZnRing + CanIsoFromTo<F::BaseRingBase> + CanHomFrom<BigIntRingBase>,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    pub fn set_rns_factor(&mut self, i: usize, el: <DoubleRNSRingBase<R, F, M> as RingBase>::Element) {
        self.operands[i] = el;
    }
}

///
/// Left-hand side operand of an "RNS-gadget product", hence this struct stores
/// the gadget decomposition of a ring element `y`, w.r.t. the RNS gadget vector
/// `(lift((q / pi)^-1 mod pi) * q / pi)_i` where `q = p1 * ... * pm` is the ring modulus.
/// For more details, see [`DoubleRNSRingBase::gadget_product()`].
/// 
pub struct GadgetProductLhsOperand<R, F, M> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanIsoFromTo<F::BaseRingBase> + CanHomFrom<BigIntRingBase>,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    operands: Vec<<DoubleRNSRingBase<R, F, M> as RingBase>::Element>
}

impl<R, F, M> DoubleRNSRingBase<R, F, M>
    where R: ZnRingStore,
        R::Type: ZnRing + CanIsoFromTo<F::BaseRingBase> + CanHomFrom<BigIntRingBase>,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    ///
    /// Computes the data necessary to perform a "gadget product" with the given operand as
    /// left-hand side. This can be though of computing the gadget decomposition of the argument.
    /// For more details, see [`DoubleRNSRingBase::gadget_product()`].
    /// 
    pub fn to_gadget_product_lhs(&self, el: DoubleRNSNonFFTEl<R, F, M>) -> GadgetProductLhsOperand<R, F, M> {
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

        return GadgetProductLhsOperand {
            operands: result.into_iter().map(|x| self.do_fft(x)).collect()
        }
    }

    ///
    /// Creates a [`GadgetProductRhsOperand`] representing 0. Its data (i.e. the noisy approximations
    /// of scalings of the base ring element) can be set later with [`GadgetProductRhsOperand::set_rns_factor()`].
    /// For more details, see [`DoubleRNSRingBase::gadget_product()`].
    /// 
    pub fn gadget_product_rhs_zero(&self) -> GadgetProductRhsOperand<R, F, M> {
        GadgetProductRhsOperand {
            operands: (0..self.rns_base().len()).map(|_| self.zero()).collect()
        }
    }

    ///
    /// Computes the "RNS-gadget product" of two elements in this ring, as often required
    /// in HE scenarios. A "gadget product" computes the approximate product of two
    /// ring elements `x` and `y` by using `y` and multiple scaled & noisy approximations 
    /// to `x`. This function only supports the gadget vector given by the RNS decomposition
    /// of the ring modulus `q`.
    /// 
    /// # What exactly is a "gadget product"?
    /// 
    /// In an HE setting, we often have a noisy approximation to some value `x`, say
    /// `x + e`. Now the normal product `(x + e)y = xy + ey` includes an error of `ey`, which
    /// (if `y` is arbitrary) is not in any way an approximation to `xy` anymore. Instead,
    /// one usually provides multiple noisy scalings of `x`, say `g1 * x + e1` to `gm * x + em`.
    /// The `g` form the so-called "gadget vector". Using these, we can approximate `xy` by 
    /// computing a gadget-decomposition `y = g1 * y1 + ... + gm * ym` for small values `yi` and
    /// then setting `xy ~ y1 (g1 * x + e1) + ... + ym (gm * x + em)`.
    /// 
    /// The gadget vector used for this "RNS-gadget product" is the one given by the RNS basis or
    /// CRT unit vectors, i.e. `gi = lift((q / pi)^-1 mod pi) * q / pi` where `q = p1 * ... * pm` 
    /// is the prime factorization of `q`.
    /// 
    pub fn gadget_product(&self, lhs: &GadgetProductLhsOperand<R, F, M>, rhs: &GadgetProductRhsOperand<R, F, M>) -> <Self as RingBase>::Element {
        <_ as RingBase>::sum(self, lhs.operands.iter().zip(rhs.operands.iter()).map(|(l, r)| self.mul_ref(l, r)))
    }
}

#[cfg(test)]
use crate::doublerns::pow2_cyclotomic::*;
#[cfg(test)]
use feanor_math::algorithms::fft::cooley_tuckey::FFTTableCooleyTuckey;
#[cfg(test)]
use zn_64::Zn;
#[cfg(test)]
use feanor_math::default_memory_provider;
#[cfg(test)]
use feanor_math::integer::BigIntRing;
#[cfg(test)]
use feanor_math::ordered::OrderedRingStore;
#[cfg(test)]
use vec_fn::VectorFn;
#[cfg(test)]
use test::Bencher;
#[cfg(test)]
use feanor_math::algorithms::miller_rabin::is_prime;
#[cfg(test)]
use feanor_math::rings::finite::FiniteRingStore;
#[cfg(test)]
use feanor_math::primitive_int::StaticRing;

#[test]
fn test_gadget_product() {
    const ZZbig: BigIntRing = BigIntRing::RING;
    let rns_base = vec![Zn::new(17), Zn::new(97)];
    let ring = DoubleRNSRingBase::<_, Pow2CyclotomicFFT<FFTTableCooleyTuckey<Zn>>, _>::new(zn_rns::Zn::new(rns_base.clone(), ZZbig, default_memory_provider!()), rns_base, 3, default_memory_provider!());

    let rhs_factor = ring_literal!(&ring, [0, 1000, 1200, 1, 600, 1600, 0, 800]);
    let mut rhs_op = ring.get_ring().gadget_product_rhs_zero();
    let error1 = ring_literal!(&ring, [1, 0, 0, -1, 0, 1, 1, 0]);
    let error2 = ring_literal!(&ring, [1, 1, 0, -1, 0, 0, -1, 0]);
    rhs_op.set_rns_factor(0, ring.add(ring.int_hom().mul_ref_fst_map(&rhs_factor, 97), error1));
    rhs_op.set_rns_factor(0, ring.add(ring.int_hom().mul_ref_fst_map(&rhs_factor, 17), error2));

    let lhs_factor = ring_literal!(&ring, [0, 10, 100, 1000, 50, 500, 800, 1]);
    let result_error = ring.sub(ring.mul_ref(&lhs_factor, &rhs_factor), ring.get_ring().gadget_product(&ring.get_ring().to_gadget_product_lhs(ring.get_ring().undo_fft(lhs_factor)), &rhs_op));
    let error_bound = 1 * 8 * (97 / 2) + 1 * 8 * (17 / 2);
    let result_error_vec = ring.wrt_canonical_basis(&result_error);
    for i in 0..8 {
        assert!(ZZbig.is_leq(&ring.base_ring().smallest_lift(result_error_vec.at(i)), &ZZbig.int_hom().map(error_bound)));
    }
}

#[bench]
fn bench_gadget_product(bencher: &mut Bencher) {
    const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;
    const ZZbig: BigIntRing = BigIntRing::RING;
    let log2_n = 14;
    let rns_base_len = 16;
    let rns_base: Vec<_> = (1..).map(|i| (i << (log2_n + 1)) + 1).filter(|p| is_prime(ZZ, &(*p as i64), 10)).map(Zn::new).take(rns_base_len).collect();
    let error_bound = ZZbig.can_hom(&ZZ).unwrap().map((rns_base.len() as i64 * *rns_base.last().unwrap().modulus() as i64) << log2_n);
    let ring = DoubleRNSRingBase::<_, Pow2CyclotomicFFT<FFTTableCooleyTuckey<Zn>>, _>::new(zn_rns::Zn::new(rns_base.clone(), ZZbig, default_memory_provider!()), rns_base, log2_n, default_memory_provider!());

    let mut rng = oorandom::Rand64::new(1);
    let rhs = ring.random_element(|| rng.rand_u64());
    let mut rhs_op = ring.get_ring().gadget_product_rhs_zero();
    let gadget_vec = |i: usize| ring.base_ring().get_ring().from_congruence((0..rns_base_len).map(|j| if i == j {
        ring.base_ring().get_ring().at(j).one()
    } else {
        ring.base_ring().get_ring().at(j).zero()
    }));
    for i in 0..rns_base_len {
        let error = ring.get_ring().sample_from_coefficient_distribution(|| (rng.rand_u64() % 3) as i32 - 1);
        rhs_op.set_rns_factor(i, ring.add(ring.inclusion().mul_ref_fst_map(&rhs, gadget_vec(i)), error));
    }

    bencher.iter(|| {
        let lhs = ring.random_element(|| rng.rand_u64());
        let expected_result = ring.mul_ref(&lhs, &rhs);
        let lhs_op = ring.get_ring().to_gadget_product_lhs(ring.get_ring().undo_fft(lhs));
        let result = ring.get_ring().gadget_product(&lhs_op, &rhs_op);
        let error = ring.sub(expected_result, result);
        let error_vec = ring.wrt_canonical_basis(&error);
        for i in 0..error_vec.len() {
            assert!(ZZbig.is_leq(&ZZbig.abs(ring.base_ring().smallest_lift(error_vec.at(i))), &error_bound));
        }
    });
}