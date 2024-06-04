use feanor_math::integer::int_cast;
use feanor_math::integer::BigIntRingBase;
use feanor_math::mempool::*;
use feanor_math::ring::*;
use feanor_math::homomorphism::*;
use feanor_math::rings::zn::*;
use feanor_math::rings::extension::*;
use feanor_math::vector::*;
use feanor_math::default_memory_provider;
use feanor_math::integer::BigIntRing;
use feanor_math::primitive_int::StaticRing;

use std::marker::PhantomData;

use super::double_rns_ring::*;

///
/// Right-hand side operand of an "RNS-gadget product", hence this struct can be thought
/// of as storing noisy approximations to `lift((q / pi)^-1 mod pi) * q / pi * x`, where 
/// `q = p1 * ... * pm` is the ring modulus. However, since we do KLSS-style gadget products,
/// some more data is stored to allow for faster computations later.
/// For more details, see [`DoubleRNSRingBase::gadget_product()`].
/// 
pub struct GadgetProductRhsOperand<'a, R, F, M> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanIsoFromTo<F::BaseRingBase> + CanHomFrom<BigIntRingBase>,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    shortened_rns_base: zn_rns::Zn<&'a R, BigIntRing>,
    ring: &'a DoubleRNSRingBase<R, F, M>,
    operands: Vec<Vec<Vec<Vec<El<R>>>>>
}

impl<'a, R, F, M> GadgetProductRhsOperand<'a, R, F, M>
    where R: ZnRingStore,
        R::Type: ZnRing + CanIsoFromTo<F::BaseRingBase> + CanHomFrom<BigIntRingBase>,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    pub fn set_rns_factor(&mut self, i: usize, el: <DoubleRNSRingBase<R, F, M> as RingBase>::Element) {
        self.operands[i] = self.ring.gadget_decompose(self.ring.undo_fft(el), self.shortened_rns_base.get_ring().len());
    }
}

///
/// Left-hand side operand of an "RNS-gadget product", hence this struct stores
/// the gadget decomposition of a ring element `y`, w.r.t. the RNS gadget vector
/// `(lift((q / pi)^-1 mod pi) * q / pi)_i` where `q = p1 * ... * pm` is the ring modulus.
/// For more details, see [`DoubleRNSRingBase::gadget_product()`].
/// 
pub struct GadgetProductLhsOperand<'a, R, F, M> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanIsoFromTo<F::BaseRingBase> + CanHomFrom<BigIntRingBase>,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    output_moduli_count: usize,
    ring: PhantomData<&'a DoubleRNSRingBase<R, F, M>>,
    operands: Vec<Vec<Vec<El<R>>>>
}

impl<R, F, M> DoubleRNSRingBase<R, F, M>
    where R: ZnRingStore,
        R::Type: ZnRing + CanIsoFromTo<F::BaseRingBase> + CanHomFrom<BigIntRingBase>,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    ///
    /// `gadget_decompose()[decomposed_component][rns_base_index][coefficient_index]` contains the 
    /// `coefficient_index`-th fourier coefficient modulo `rns_base.at(rns_base_index)` of the 
    /// `decomposed_component`-th element of the gadget decomposition vector.
    /// 
    /// The order of the fourier coefficients is the same as specified by the corresponding [`GeneralizedFFT`].
    /// 
    fn gadget_decompose(&self, el: DoubleRNSNonFFTEl<R, F, M>, output_moduli_count: usize) -> Vec<Vec<Vec<El<R>>>> {
        let mut result = Vec::new();

        for i in 0..self.rns_base().len() {
            result.push((0..output_moduli_count).map(|k| {
                let int_ring = self.rns_base().at(i).integer_ring();
                let ring_i = self.rns_base().len() - output_moduli_count + k;
                let hom = self.rns_base().at(ring_i).can_hom(int_ring).unwrap();
                let mut result = (0..self.rank())
                    .map(|j| self.rns_base().at(i).smallest_lift(self.rns_base().at(i).clone_el(self.at(i, j, &el))))
                    .map(|n| hom.map(n))
                    .collect::<Vec<_>>();
                self.generalized_fft().at(ring_i).fft_forward(&mut result, hom.codomain(), self.memory_provider());
                return result;
            }).collect::<Vec<_>>());
        }
        return result;
    }

    ///
    /// The number of moduli we need when performing the internal inner product
    /// during the KLSS-style product (ia.cr/2023/413)
    /// 
    fn get_gadget_product_modulo_count(&self) -> usize {
        let p_max = self.rns_base().iter().map(|Fp| int_cast(Fp.integer_ring().clone_el(Fp.modulus()), StaticRing::<i64>::RING, Fp.integer_ring())).max().unwrap();
        // the maximal size of the inner product of two gadget-decomposed elements
        let max_size_log2 = (self.rank() as f64 * p_max as f64 / 2.).log2() * 2. + (self.rns_base().len() as f64).log2();
        let mut current_size_log2 = 0.;
        self.rns_base().iter().rev().take_while(|Fp| {
            if current_size_log2 < max_size_log2 {
                current_size_log2 += (int_cast(Fp.integer_ring().clone_el(Fp.modulus()), StaticRing::<i64>::RING, Fp.integer_ring()) as f64).log2();
                true
            } else {
                false
            }
        }).count()
    }

    ///
    /// Computes the data necessary to perform a "gadget product" with the given operand as
    /// left-hand side. This can be though of computing the gadget decomposition of the argument.
    /// For more details, see [`DoubleRNSRingBase::gadget_product()`].
    /// 
    pub fn to_gadget_product_lhs<'a>(&'a self, el: DoubleRNSNonFFTEl<R, F, M>) -> GadgetProductLhsOperand<'a, R, F, M> {
        timed!("to_gadget_product_lhs", || {
            let output_moduli_count = self.get_gadget_product_modulo_count();
            return GadgetProductLhsOperand {
                ring: PhantomData,
                operands: self.gadget_decompose(el, output_moduli_count),
                output_moduli_count: output_moduli_count
            }
        })
    }

    ///
    /// Creates a [`GadgetProductRhsOperand`] representing 0. Its data (i.e. the noisy approximations
    /// of scalings of the base ring element) can be set later with [`GadgetProductRhsOperand::set_rns_factor()`].
    /// For more details, see [`DoubleRNSRingBase::gadget_product()`].
    /// 
    pub fn gadget_product_rhs_zero<'a>(&'a self) -> GadgetProductRhsOperand<'a, R, F, M> {
        let output_moduli_count = self.get_gadget_product_modulo_count();
        let shortened_rns_base = zn_rns::Zn::new(self.rns_base().iter().skip(self.rns_base().len() - output_moduli_count).collect(), BigIntRing::RING, default_memory_provider!());
        GadgetProductRhsOperand {
            ring: self,
            operands: (0..self.rns_base().len()).map(|_| (0..self.rns_base().len()).map(|_| (0..output_moduli_count).map(|i| (0..self.rank()).map(|_| shortened_rns_base.get_ring().at(i).zero()).collect()).collect()).collect()).collect(),
            shortened_rns_base: shortened_rns_base,
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
    /// # Example
    /// ```
    /// # use feanor_math::ring::*;
    /// # use feanor_math::assert_el_eq;
    /// # use feanor_math::default_memory_provider;
    /// # use feanor_math::homomorphism::*;
    /// # use feanor_math::rings::zn::*;
    /// # use feanor_math::rings::zn::zn_64::Zn;
    /// # use feanor_math::integer::BigIntRing;
    /// # use feanor_math::algorithms::fft::cooley_tuckey::FFTTableCooleyTuckey;
    /// # use he_ring::doublerns::double_rns_ring::DoubleRNSRingBase;
    /// # use he_ring::doublerns::pow2_cyclotomic::Pow2CyclotomicFFT;
    /// # use feanor_math::rings::extension::FreeAlgebraStore;
    /// # use feanor_math::vector::VectorView;
    /// // Set up the double-RNS ring
    /// let rns_base = vec![Zn::new(17), Zn::new(97), Zn::new(113)];
    /// let ring = DoubleRNSRingBase::<_, Pow2CyclotomicFFT<FFTTableCooleyTuckey<_>>, _>::new(zn_rns::Zn::new(rns_base.clone(), BigIntRing::RING, default_memory_provider!()), rns_base, 3, default_memory_provider!());
    /// 
    /// // Create the right-hand operand; this is usually considered to be constant
    /// let rhs = ring.from_canonical_basis([0, 1000, 1200, 1, 600, 1600, 0, 800].into_iter().map(|x| ring.base_ring().int_hom().map(x)));
    /// let mut rhs_op = ring.get_ring().gadget_product_rhs_zero();
    /// let gadget_vector = |i: usize| ring.base_ring().get_ring().from_congruence((0..3).map(|j| ring.base_ring().get_ring().at(j).int_hom().map(if i == j { 1 } else { 0 })));
    /// for i in 0..3 {
    ///     // here we might include a small error (after scaling), and then get an approximate result; See next example 
    ///     rhs_op.set_rns_factor(i, ring.inclusion().mul_ref_map(&rhs, &gadget_vector(i)));
    /// }
    /// 
    /// let lhs = ring.from_canonical_basis([0, 10, 100, 1000, 50, 500, 800, 1].into_iter().map(|x| ring.base_ring().int_hom().map(x)));
    /// assert_el_eq!(&ring, &ring.mul_ref(&lhs, &rhs), &ring.get_ring().gadget_product(&ring.get_ring().to_gadget_product_lhs(ring.get_ring().undo_fft(lhs)), &rhs_op));
    /// ```
    /// To demonstrate how this keeps small error terms small, consider the following variation of the previous example:
    /// 
    /// ```
    /// # use feanor_math::ring::*;
    /// # use feanor_math::assert_el_eq;
    /// # use feanor_math::default_memory_provider;
    /// # use feanor_math::homomorphism::*;
    /// # use feanor_math::rings::zn::*;
    /// # use feanor_math::rings::zn::zn_64::Zn;
    /// # use feanor_math::algorithms::fft::cooley_tuckey::FFTTableCooleyTuckey;
    /// # use he_ring::doublerns::double_rns_ring::DoubleRNSRingBase;
    /// # use he_ring::doublerns::pow2_cyclotomic::Pow2CyclotomicFFT;
    /// # use feanor_math::rings::extension::FreeAlgebraStore;
    /// # use feanor_math::vector::VectorView;
    /// # use feanor_math::integer::BigIntRing;
    /// # use feanor_math::integer::int_cast;
    /// # use feanor_math::primitive_int::StaticRing;
    /// # use feanor_math::vector::vec_fn::VectorFn;
    /// let rns_base = vec![Zn::new(17), Zn::new(97), Zn::new(113)];
    /// let ring = DoubleRNSRingBase::<_, Pow2CyclotomicFFT<FFTTableCooleyTuckey<_>>, _>::new(zn_rns::Zn::new(rns_base.clone(), BigIntRing::RING, default_memory_provider!()), rns_base, 3, default_memory_provider!());
    /// 
    /// let mut rng = oorandom::Rand64::new(1);
    /// let mut error = || ring.get_ring().sample_from_coefficient_distribution(|| (rng.rand_u64() % 3) as i32 - 1);
    /// 
    /// let rhs = ring.from_canonical_basis([0, 1000, 1200, 1, 600, 1600, 0, 800].into_iter().map(|x| ring.base_ring().int_hom().map(x)));
    /// let mut rhs_op = ring.get_ring().gadget_product_rhs_zero();
    /// let gadget_vector = |i: usize| ring.base_ring().get_ring().from_congruence((0..3).map(|j| ring.base_ring().get_ring().at(j).int_hom().map(if i == j { 1 } else { 0 })));
    /// for i in 0..3 {
    ///     rhs_op.set_rns_factor(i, ring.add(ring.inclusion().mul_ref_map(&rhs, &gadget_vector(i)), error()));
    /// }
    /// 
    /// let lhs = ring.from_canonical_basis([0, 10, 100, 1000, 50, 500, 800, 1].into_iter().map(|x| ring.base_ring().int_hom().map(x)));
    /// let expected = ring.mul_ref(&lhs, &rhs);
    /// let actual = ring.get_ring().gadget_product(&ring.get_ring().to_gadget_product_lhs(ring.get_ring().undo_fft(lhs)), &rhs_op);
    /// let error = ring.sub(expected, actual);
    /// let error_coefficients = ring.wrt_canonical_basis(&error);
    /// let max_allowed_error = (113 / 2) * 8 * 3;
    /// assert!((0..8).all(|i| int_cast(ring.base_ring().smallest_lift(error_coefficients.at(i)), StaticRing::<i64>::RING, BigIntRing::RING).abs() <= max_allowed_error));
    /// ```
    /// 
    pub fn gadget_product(&self, lhs: &GadgetProductLhsOperand<R, F, M>, rhs: &GadgetProductRhsOperand<R, F, M>) -> DoubleRNSEl<R, F, M> {
        self.do_fft(self.gadget_product_base(lhs, rhs))
    }

    ///
    /// The gadget product without final FFT. See [`Self::gadget_product()`] for a description.
    /// 
    pub fn gadget_product_base(&self, lhs: &GadgetProductLhsOperand<R, F, M>, rhs: &GadgetProductRhsOperand<R, F, M>) -> DoubleRNSNonFFTEl<R, F, M> {
        timed!("gadget_product_base", || {
            assert_eq!(lhs.output_moduli_count, rhs.shortened_rns_base.get_ring().len());
            let output_moduli_count = lhs.output_moduli_count;
            let shortened_rns_base = rhs.shortened_rns_base.get_ring();

            let mut result = self.non_fft_zero();
            for j in 0..self.rns_base().len() {
                let mut summand = timed!("gadget_product_base::sum", || {
                    (0..output_moduli_count).map(|k| (0..self.rank()).map(|l| {
                        let Fp = shortened_rns_base.at(k);
                        <_ as RingStore>::sum(&Fp, (0..self.rns_base().len()).map(|i| Fp.mul_ref(&lhs.operands[i][k][l], &rhs.operands[i][j][k][l])))
                    }).collect::<Vec<_>>()).collect::<Vec<_>>()
                });
                timed!("gadget_product_base::ffts", || {
                    for k in 0..output_moduli_count {
                        self.generalized_fft()[self.rns_base().len() - output_moduli_count + k].fft_backward(&mut summand[k], shortened_rns_base.at(k), self.memory_provider());
                    }
                });
                timed!("gadget_product_base::lifting", || {
                    let target_Fp = self.rns_base().at(j);
                    let mod_target_Fp = target_Fp.can_hom(&BigIntRing::RING).unwrap();
                    for i in 0..self.rank() {
                        let short_result = shortened_rns_base.from_congruence((0..output_moduli_count).map(|k| shortened_rns_base.at(k).clone_el(&summand[k][i])));
                        *self.at_mut(j, i, &mut result) = mod_target_Fp.map(shortened_rns_base.smallest_lift(short_result));
                    }
                });
            }
            return result;
        })
    }
}

#[cfg(test)]
use crate::doublerns::pow2_cyclotomic::*;
#[cfg(test)]
use crate::profiling::print_all_timings;
#[cfg(test)]
use feanor_math::algorithms::fft::cooley_tuckey::FFTTableCooleyTuckey;
#[cfg(test)]
use zn_64::Zn;
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
use feanor_math::assert_el_eq;

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

#[test]
fn test_gadget_product_zero() {
    const ZZbig: BigIntRing = BigIntRing::RING;
    let rns_base = vec![Zn::new(17), Zn::new(97)];
    let ring = DoubleRNSRingBase::<_, Pow2CyclotomicFFT<FFTTableCooleyTuckey<Zn>>, _>::new(zn_rns::Zn::new(rns_base.clone(), ZZbig, default_memory_provider!()), rns_base, 3, default_memory_provider!());

    let rhs_op = ring.get_ring().gadget_product_rhs_zero();
    let lhs = ring_literal!(&ring, [0, 10, 100, 1000, 50, 500, 800, 1]);
    assert_el_eq!(&ring, &ring.zero(), &ring.get_ring().gadget_product(&ring.get_ring().to_gadget_product_lhs(ring.get_ring().undo_fft(lhs)), &rhs_op));
}

#[bench]
fn bench_gadget_product(bencher: &mut Bencher) {
    const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;
    const ZZbig: BigIntRing = BigIntRing::RING;
    let log2_n = 10;
    let rns_base_len = 16;
    let rns_base: Vec<_> = (0..).map(|i| (i << (log2_n + 1)) + 1).filter(|p| is_prime(ZZ, &(*p as i64), 10)).map(Zn::new).take(rns_base_len).collect();
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

    print_all_timings();
}