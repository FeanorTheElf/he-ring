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
use feanor_math::algorithms::matmul::InnerProductComputation;
use feanor_math::matrix::submatrix::*;

use std::marker::PhantomData;

use super::double_rns_ring::*;
use crate::rnsconv::approx_lift::AlmostExactBaseConversion;
use crate::rnsconv::*;

///
/// Right-hand side operand of an "RNS-gadget product", hence this struct can be thought
/// of as storing noisy approximations to `lift((q / pi)^-1 mod pi) * q / pi * x`, where 
/// `q = p1 * ... * pm` is the ring modulus. However, since we do KLSS-style gadget products,
/// some more data is stored to allow for faster computations later.
/// For more details, see [`DoubleRNSRingBase::gadget_product()`].
/// 
pub enum GadgetProductRhsOperand<'a, R, F, M> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanIsoFromTo<F::BaseRingBase> + CanHomFrom<BigIntRingBase> + SelfIso,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    LKSSStyle(LKSSGadgetProductRhsOperand<'a, R, F, M>),
    Naive(Vec<El<DoubleRNSRing<R, F, M>>>)
}

pub struct LKSSGadgetProductRhsOperand<'a, R, F, M> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanIsoFromTo<F::BaseRingBase> + CanHomFrom<BigIntRingBase> + SelfIso,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    shortened_rns_base: zn_rns::Zn<&'a R, BigIntRing>,
    ring: &'a DoubleRNSRingBase<R, F, M>,
    operands: Vec<Vec<M::Object>>,
    conversions: Vec<AlmostExactBaseConversion<&'a R, DefaultMemoryProvider, DefaultMemoryProvider>>
}

impl<'a, R, F, M> LKSSGadgetProductRhsOperand<'a, R, F, M>
    where R: ZnRingStore,
        R::Type: ZnRing + CanIsoFromTo<F::BaseRingBase> + CanHomFrom<BigIntRingBase> + SelfIso,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    fn at(&self, i: usize, j: usize, k: usize, l: usize) -> &El<R> {
        debug_assert!(i < self.ring.rns_base().len());
        debug_assert!(j < self.ring.rns_base().len());
        debug_assert!(k < self.shortened_rns_base.get_ring().len());
        debug_assert!(l < self.ring.rank());
        &self.operands[i][j][k * self.ring.rank() + l]
    }

    fn set_rns_factor(&mut self, i: usize, el: <DoubleRNSRingBase<R, F, M> as RingBase>::Element) {
        self.operands[i] = self.ring.gadget_decompose(self.ring.undo_fft(el), self.shortened_rns_base.get_ring().len());
    }
}

impl<'a, R, F, M> GadgetProductRhsOperand<'a, R, F, M>
    where R: ZnRingStore,
        R::Type: ZnRing + CanIsoFromTo<F::BaseRingBase> + CanHomFrom<BigIntRingBase> + SelfIso,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    pub fn set_rns_factor(&mut self, i: usize, el: <DoubleRNSRingBase<R, F, M> as RingBase>::Element) {
        match self {
            GadgetProductRhsOperand::LKSSStyle(op) => op.set_rns_factor(i, el),
            GadgetProductRhsOperand::Naive(op) => op[i] = el
        }
    }
}

///
/// Left-hand side operand of an "RNS-gadget product", hence this struct stores
/// the gadget decomposition of a ring element `y`, w.r.t. the RNS gadget vector
/// `(lift((q / pi)^-1 mod pi) * q / pi)_i` where `q = p1 * ... * pm` is the ring modulus.
/// For more details, see [`DoubleRNSRingBase::gadget_product()`].
/// 
pub enum GadgetProductLhsOperand<'a, R, F, M> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanIsoFromTo<F::BaseRingBase> + CanHomFrom<BigIntRingBase> + SelfIso,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    LKSSStyle(LKSSGadgetProductLhsOperand<'a, R, F, M>),
    Naive(Vec<El<DoubleRNSRing<R, F, M>>>)
}

pub struct LKSSGadgetProductLhsOperand<'a, R, F, M> 
    where R: ZnRingStore,
        R::Type: ZnRing + CanIsoFromTo<F::BaseRingBase> + CanHomFrom<BigIntRingBase>,
        F: GeneralizedFFT + GeneralizedFFTSelfIso,
        M: MemoryProvider<El<R>>
{
    output_moduli_count: usize,
    ring: PhantomData<&'a DoubleRNSRingBase<R, F, M>>,
    operands: Vec<M::Object>
}

impl<R, F, M> DoubleRNSRingBase<R, F, M>
    where R: ZnRingStore,
        R::Type: ZnRing + CanIsoFromTo<F::BaseRingBase> + CanHomFrom<BigIntRingBase> + SelfIso,
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
    fn gadget_decompose(&self, el: DoubleRNSNonFFTEl<R, F, M>, output_moduli_count: usize) -> Vec<M::Object> {
        let mut result = Vec::new();

        for i in 0..self.rns_base().len() {
            let int_ring = self.rns_base().at(i).integer_ring();
            let homs = (0..output_moduli_count).map(|k| self.rns_base().at(self.rns_base().len() - output_moduli_count + k).can_hom(int_ring).unwrap()).collect::<Vec<_>>();
            result.push(self.memory_provider().get_new_init(output_moduli_count * self.rank(), |idx| {
                let k = idx / self.rank();
                let j = idx % self.rank();
                homs[k].map(self.rns_base().at(i).smallest_lift(self.rns_base().at(i).clone_el(self.at(i, j, &el))))
            }));
            for k in 0..output_moduli_count {
                let ring_i = self.rns_base().len() - output_moduli_count + k;
                self.generalized_fft().at(ring_i).fft_forward(&mut result.last_mut().unwrap()[(k * self.rank())..((k + 1) * self.rank())], homs[k].codomain(), self.memory_provider());
            }
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
        let max_size_log2 = (self.rank() as f64 * p_max as f64 / 2.).log2() * 2. + (self.rns_base().len() as f64).log2() + 1.;
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

    fn use_lkss_gadget_product(&self) -> bool {
        self.get_gadget_product_modulo_count() + 1 < self.rns_base().len() && !cfg!(feature = "force_naive_gadget_product")
    }

    ///
    /// Computes the data necessary to perform a "gadget product" with the given operand as
    /// left-hand side. This can be though of computing the gadget decomposition of the argument.
    /// For more details, see [`DoubleRNSRingBase::gadget_product()`].
    /// 
    pub fn to_gadget_product_lhs<'a>(&'a self, el: DoubleRNSNonFFTEl<R, F, M>) -> GadgetProductLhsOperand<'a, R, F, M> {
        timed!("to_gadget_product_lhs", || {
            let output_moduli_count = self.get_gadget_product_modulo_count();
            if self.use_lkss_gadget_product() {
                GadgetProductLhsOperand::LKSSStyle(LKSSGadgetProductLhsOperand {
                    ring: PhantomData,
                    operands: self.gadget_decompose(el, output_moduli_count),
                    output_moduli_count: output_moduli_count
                })
            } else {
                GadgetProductLhsOperand::Naive(
                    self.gadget_decompose(el, self.rns_base().len()).into_iter()
                        .map(|data| DoubleRNSEl {
                            generalized_fft: PhantomData,
                            memory_provider: PhantomData,
                            data: data
                        }).collect()
                )
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
        // if the RNS base is very short, we don't want to do LKSS style gadget products, mainly
        // for code simplicity, but we also don't expect much performance improvement
        if self.use_lkss_gadget_product() {
            let shortened_rns_base = zn_rns::Zn::new(self.rns_base().iter().skip(self.rns_base().len() - output_moduli_count).collect(), BigIntRing::RING, default_memory_provider!());
            GadgetProductRhsOperand::LKSSStyle(LKSSGadgetProductRhsOperand {
                ring: self,
                operands: (0..self.rns_base().len()).map(|_| (0..self.rns_base().len()).map(|_| self.memory_provider().get_new_init(output_moduli_count * self.rank(), |idx| {
                    let k = idx / self.rank();
                    self.rns_base().at(self.rns_base().len() - output_moduli_count + k).zero()
                })).collect()).collect(),
                conversions: (0..self.rns_base().len()).map(|i| AlmostExactBaseConversion::new(
                    shortened_rns_base.get_ring(), 
                    &[self.rns_base().at(i)], 
                    default_memory_provider!(),
                    default_memory_provider!()
                )).collect::<Vec<_>>(),
                shortened_rns_base: shortened_rns_base
            })
        } else {
            GadgetProductRhsOperand::Naive((0..self.rns_base().len()).map(|_| self.zero()).collect())
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
    /// # use feanor_math::rings::finite::*;
    /// # use feanor_math::integer::BigIntRing;
    /// # use feanor_math::algorithms::fft::cooley_tuckey::FFTTableCooleyTuckey;
    /// # use he_ring::doublerns::double_rns_ring::DoubleRNSRingBase;
    /// # use he_ring::doublerns::pow2_cyclotomic::Pow2CyclotomicFFT;
    /// # use feanor_math::rings::extension::FreeAlgebraStore;
    /// # use feanor_math::vector::VectorView;
    /// let rns_base = vec![Zn::new(17), Zn::new(97), Zn::new(113)];
    /// let ring = DoubleRNSRingBase::<_, Pow2CyclotomicFFT<FFTTableCooleyTuckey<_>>, _>::new(zn_rns::Zn::new(rns_base.clone(), BigIntRing::RING, default_memory_provider!()), rns_base, 3, default_memory_provider!());
    /// let mut rng = oorandom::Rand64::new(1);
    /// 
    /// // build the right-hand side operand
    /// let rhs = ring.random_element(|| rng.rand_u64());
    /// let mut rhs_op = ring.get_ring().gadget_product_rhs_zero();
    /// let gadget_vector = |i: usize| ring.base_ring().get_ring().from_congruence((0..3).map(|j| ring.base_ring().get_ring().at(j).int_hom().map(if i == j { 1 } else { 0 })));
    /// for i in 0..3 {
    ///     // set the i-th component to `gadget_vector(i) * rhs`, for now without noise
    ///     rhs_op.set_rns_factor(i, ring.inclusion().mul_ref_map(&rhs, &gadget_vector(i)));
    /// }
    /// 
    /// // compute the gadget product
    /// let lhs = ring.random_element(|| rng.rand_u64());
    /// let actual = ring.get_ring().gadget_product(&ring.get_ring().to_gadget_product_lhs(ring.get_ring().undo_fft(ring.clone_el(&lhs))), &rhs_op);
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
    /// # use feanor_math::rings::finite::*;
    /// # use feanor_math::integer::int_cast;
    /// # use feanor_math::primitive_int::StaticRing;
    /// # use feanor_math::vector::vec_fn::VectorFn;
    /// let rns_base = vec![Zn::new(17), Zn::new(97), Zn::new(113)];
    /// let ring = DoubleRNSRingBase::<_, Pow2CyclotomicFFT<FFTTableCooleyTuckey<_>>, _>::new(zn_rns::Zn::new(rns_base.clone(), BigIntRing::RING, default_memory_provider!()), rns_base, 3, default_memory_provider!());
    /// 
    /// let mut rng = oorandom::Rand64::new(1);
    /// 
    /// // build the right-hand side operand
    /// let rhs = ring.random_element(|| rng.rand_u64());
    /// let mut error = || ring.get_ring().sample_from_coefficient_distribution(|| (rng.rand_u64() % 3) as i32 - 1);
    /// let mut rhs_op = ring.get_ring().gadget_product_rhs_zero();
    /// let gadget_vector = |i: usize| ring.base_ring().get_ring().from_congruence((0..3).map(|j| ring.base_ring().get_ring().at(j).int_hom().map(if i == j { 1 } else { 0 })));
    /// for i in 0..3 {
    ///     // set the i-th component to `gadget_vector(i) * rhs`, with possibly some noise included
    ///     rhs_op.set_rns_factor(i, ring.add(ring.inclusion().mul_ref_map(&rhs, &gadget_vector(i)), error()));
    /// }
    /// 
    /// // compute the gadget product
    /// let lhs = ring.random_element(|| rng.rand_u64());
    /// let actual = ring.get_ring().gadget_product(&ring.get_ring().to_gadget_product_lhs(ring.get_ring().undo_fft(ring.clone_el(&lhs))), &rhs_op);
    /// 
    /// // the final result should be close to `lhs * rhs`, except for some noise
    /// let expected = ring.mul_ref(&lhs, &rhs);
    /// let error = ring.sub(expected, actual);
    /// let error_coefficients = ring.wrt_canonical_basis(&error);
    /// let max_allowed_error = (113 / 2) * 8 * 3;
    /// assert!((0..8).all(|i| int_cast(ring.base_ring().smallest_lift(error_coefficients.at(i)), StaticRing::<i64>::RING, BigIntRing::RING).abs() <= max_allowed_error));
    /// ```
    /// 
    pub fn gadget_product(&self, lhs: &GadgetProductLhsOperand<R, F, M>, rhs: &GadgetProductRhsOperand<R, F, M>) -> DoubleRNSEl<R, F, M> {
        self.do_fft(self.gadget_product_base(lhs, rhs))
    }

    fn gadget_product_lkss(&self, lhs: &LKSSGadgetProductLhsOperand<R, F, M>, rhs: &LKSSGadgetProductRhsOperand<R, F, M>) -> DoubleRNSNonFFTEl<R, F, M> {
        timed!("gadget_product_lkss", || {
            assert_eq!(lhs.output_moduli_count, rhs.shortened_rns_base.get_ring().len());
            let output_moduli_count = lhs.output_moduli_count;
            let shortened_rns_base = rhs.shortened_rns_base.get_ring();

            let mut result = self.non_fft_zero();
            for j in 0..self.rns_base().len() {
                let mut summand = timed!("gadget_product_lkss::sum", || self.memory_provider().get_new_init(shortened_rns_base.len() * self.rank(), |idx| {
                    let k = idx / self.rank();
                    let l = idx % self.rank();
                    let Fp = shortened_rns_base.at(k);
                    <_ as InnerProductComputation>::inner_product_ref(Fp.get_ring(), (0..self.rns_base().len()).map(|i| (&lhs.operands[i][k * self.rank() + l], rhs.at(i, j, k, l))))
                }));
                timed!("gadget_product_lkss::ffts", || {
                    for k in 0..output_moduli_count {
                        self.generalized_fft()[self.rns_base().len() - output_moduli_count + k].fft_backward(&mut summand[(k * self.rank())..((k + 1) * self.rank())], shortened_rns_base.at(k), self.memory_provider());
                    }
                });
                timed!("gadget_product_lkss::lifting", || {
                    rhs.conversions[j].apply(
                        Submatrix::<AsFirstElement<_>, _>::new(&summand, shortened_rns_base.len(), self.rank()),
                        self.as_matrix_mut(&mut result).restrict_rows(j..(j + 1))
                    );
                });
            }
            return result;
        })
    }

    ///
    /// The gadget product without final FFT. See [`Self::gadget_product()`] for a description.
    /// 
    /// The implementation uses the KLSS-style algorithm (ia.cr/2023/413).
    /// 
    pub fn gadget_product_base(&self, lhs: &GadgetProductLhsOperand<R, F, M>, rhs: &GadgetProductRhsOperand<R, F, M>) -> DoubleRNSNonFFTEl<R, F, M> {
        match (lhs, rhs) {
            (GadgetProductLhsOperand::LKSSStyle(lhs), GadgetProductRhsOperand::LKSSStyle(rhs)) => self.gadget_product_lkss(lhs, rhs),
            (GadgetProductLhsOperand::Naive(lhs), GadgetProductRhsOperand::Naive(rhs)) => timed!("gadget_product_base::naive", || {
                self.undo_fft(<_ as RingBase>::sum(self, lhs.iter().zip(rhs.iter()).map(|(l, r)| self.mul_ref(l, r))))
            }),
            _ => panic!("Illegal combination of GadgetProductOperands; Maybe they were created by different rings?")
        }
        
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

    {
        let rns_base = vec![Zn::new(17), Zn::new(97), Zn::new(113), Zn::new(193), Zn::new(241), Zn::new(257)];
        let ring = DoubleRNSRingBase::<_, Pow2CyclotomicFFT<FFTTableCooleyTuckey<Zn>>, _>::new(zn_rns::Zn::new(rns_base.clone(), ZZbig, default_memory_provider!()), rns_base, 3, default_memory_provider!());

        let rhs = ring_literal!(&ring, [0, 100000, 120000, 100, 60000, 160000, 0, 80000]);
        let mut rhs_op = ring.get_ring().gadget_product_rhs_zero();
        let errors = [
            ring_literal!(&ring, [1, 0, 0, -1, 0, 1, 1, 0]),
            ring_literal!(&ring, [1, 1, 0, -1, 0, 0, -1, 0]),
            ring_literal!(&ring, [0, 1, 0, -1, 0, 0, -1, 0]),
            ring_literal!(&ring, [0, 1, 0, 0, 0, 0, 0, 1]),
            ring_literal!(&ring, [1, 0, 0, -1, 1, 0, -1, 0]),
            ring_literal!(&ring, [-1, -1, 1, 0, 0, 0, 0, 1])
        ];
        for i in 0..ring.base_ring().get_ring().len() {
            let gadget_vector_i = ring.base_ring().get_ring().from_congruence((0..ring.base_ring().get_ring().len()).map(|j| ring.base_ring().get_ring().at(j).int_hom().map(if j == i { 1 } else { 0 })));
            rhs_op.set_rns_factor(i, ring.add_ref_snd(ring.inclusion().mul_ref_fst_map(&rhs, gadget_vector_i), &errors[i]));
        }

        let lhs_factor = ring_literal!(&ring, [0, 1000, 10000, 100000, 5000, 50000, 80000, 100]);
        let result_error = ring.sub(ring.mul_ref(&lhs_factor, &rhs), ring.get_ring().gadget_product(&ring.get_ring().to_gadget_product_lhs(ring.get_ring().undo_fft(lhs_factor)), &rhs_op));
        let error_bound = 1 * 8 * (97 / 2) + 1 * 8 * (17 / 2) + 1 * 8 * (113 / 2) + 1 * 8 * (193 / 2);
        let result_error_vec = ring.wrt_canonical_basis(&result_error);
        for i in 0..8 {
            assert!(ZZbig.is_leq(&ring.base_ring().smallest_lift(result_error_vec.at(i)), &ZZbig.int_hom().map(error_bound)));
        }
    }
    {
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
}

#[test]
fn test_gadget_product_zero() {
    const ZZbig: BigIntRing = BigIntRing::RING;
    
    {
        let rns_base = vec![Zn::new(17), Zn::new(97), Zn::new(113), Zn::new(193), Zn::new(241), Zn::new(257)];
        let ring = DoubleRNSRingBase::<_, Pow2CyclotomicFFT<FFTTableCooleyTuckey<Zn>>, _>::new(zn_rns::Zn::new(rns_base.clone(), ZZbig, default_memory_provider!()), rns_base, 3, default_memory_provider!());

        let rhs_op = ring.get_ring().gadget_product_rhs_zero();
        let lhs = ring_literal!(&ring, [0, 10, 100, 1000, 50, 500, 800, 1]);
        assert_el_eq!(&ring, &ring.zero(), &ring.get_ring().gadget_product(&ring.get_ring().to_gadget_product_lhs(ring.get_ring().undo_fft(lhs)), &rhs_op));
    }
    {
        let rns_base = vec![Zn::new(17), Zn::new(97)];
        let ring = DoubleRNSRingBase::<_, Pow2CyclotomicFFT<FFTTableCooleyTuckey<Zn>>, _>::new(zn_rns::Zn::new(rns_base.clone(), ZZbig, default_memory_provider!()), rns_base, 3, default_memory_provider!());

        let rhs_op = ring.get_ring().gadget_product_rhs_zero();
        let lhs = ring_literal!(&ring, [0, 10, 100, 1000, 50, 500, 800, 1]);
        assert_el_eq!(&ring, &ring.zero(), &ring.get_ring().gadget_product(&ring.get_ring().to_gadget_product_lhs(ring.get_ring().undo_fft(lhs)), &rhs_op));
    }
}

#[bench]
fn bench_gadget_product(bencher: &mut Bencher) {
    const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;
    const ZZbig: BigIntRing = BigIntRing::RING;
    let log2_n = 10;
    let rns_base_len = 32;
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

    let lhs = ring.random_element(|| rng.rand_u64());
    let expected_result = ring.get_ring().undo_fft(ring.mul_ref(&lhs, &rhs));

    bencher.iter(|| {
        let lhs_op = ring.get_ring().to_gadget_product_lhs(ring.get_ring().undo_fft(ring.clone_el(&lhs)));
        let result = ring.get_ring().gadget_product_base(&lhs_op, &rhs_op);
        let mut error = result;
        ring.get_ring().sub_assign_non_fft(&mut error, &expected_result);
        let error_vec = ring.get_ring().wrt_canonical_basis_non_fft(&error);
        // only check one random coordinate, otherwise this skews the performance completely
        assert!(ZZbig.is_leq(&ZZbig.abs(ring.base_ring().smallest_lift(error_vec.at(0))), &error_bound));
    });

    print_all_timings();
}