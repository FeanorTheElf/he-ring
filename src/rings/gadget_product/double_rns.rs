use feanor_math::integer::*;
use feanor_math::ring::*;
use feanor_math::homomorphism::*;
use feanor_math::rings::zn::*;
use feanor_math::rings::zn::zn_64::*;
use feanor_math::rings::extension::*;
use feanor_math::seq::*;
use feanor_math::integer::BigIntRing;
use feanor_math::primitive_int::StaticRing;
use feanor_math::algorithms::matmul::ComputeInnerProduct;
use feanor_math::matrix::*;

use std::alloc::Allocator;
use std::alloc::Global;
use std::marker::PhantomData;

use crate::rings::double_rns_ring::*;
use crate::cyclotomic::CyclotomicRing;
use crate::rnsconv::*;
use crate::rings::decomposition::*;
use crate::IsEq;

// don't use the matrix one here, as in our case, one dimension is very small (e.g. 3), and the matrix
// version pads to multiples of 16
type UsedBaseConversion<A> = lift::AlmostExactBaseConversion<A>;

///
/// Right-hand side operand of an "RNS-gadget product", hence this struct can be thought
/// of as storing noisy approximations to `lift((q / pi)^-1 mod pi) * q / pi * x`, where 
/// `q = p1 * ... * pm` is the ring modulus. However, since we do KLSS-style gadget products,
/// some more data is stored to allow for faster computations later.
/// For more details, see [`DoubleRNSRingBase::gadget_product()`].
/// 
pub enum GadgetProductRhsOperand<'a, NumberRing, A = Global> 
    where NumberRing: DecomposableNumberRing<Zn>,
        A: Allocator + Clone
{
    LKSSStyle(LKSSGadgetProductRhsOperand<'a, NumberRing, A>),
    Naive(&'a DoubleRNSRingBase<NumberRing, Zn, A>, Vec<El<DoubleRNSRing<NumberRing, Zn, A>>>)
}

///
/// LKSS-style lhs gadget product operand; 
/// Stores the noisy approximations to `lift((q / pi)^-1 mod pi) * q / pi * x`
/// 
pub struct LKSSGadgetProductRhsOperand<'a, NumberRing, A = Global> 
    where NumberRing: DecomposableNumberRing<Zn>,
        A: Allocator + Clone
{
    shortened_rns_base: zn_rns::Zn<Zn, BigIntRing>,
    ring: &'a DoubleRNSRingBase<NumberRing, Zn, A>,
    operands: Vec<Vec<Vec<ZnEl, A>, A>, A>,
    conversions: Vec<UsedBaseConversion<A>>
}

impl<'a, NumberRing, A> LKSSGadgetProductRhsOperand<'a, NumberRing, A> 
    where NumberRing: DecomposableNumberRing<Zn>,
        A: Allocator + Clone
{
    fn at(&self, i: usize, j: usize, k: usize, l: usize) -> &ZnEl {
        debug_assert!(i < self.ring.rns_base().len());
        debug_assert!(j < self.ring.rns_base().len());
        debug_assert!(k < self.shortened_rns_base.len());
        debug_assert!(l < self.ring.rank());
        &self.operands[i][j][k * self.ring.rank() + l]
    }

    fn set_rns_factor(&mut self, i: usize, el: CoeffEl<NumberRing, Zn, A>) {
        self.operands[i] = self.ring.gadget_decompose(el, self.shortened_rns_base.get_ring().len());
    }
}

impl<'a, NumberRing, A> GadgetProductRhsOperand<'a, NumberRing, A> 
    where NumberRing: DecomposableNumberRing<Zn>,
        A: Allocator + Clone
{
    pub fn set_rns_factor(&mut self, i: usize, el: CoeffEl<NumberRing, Zn, A>) {
        match self {
            GadgetProductRhsOperand::LKSSStyle(op) => op.set_rns_factor(i, el),
            GadgetProductRhsOperand::Naive(ring, op) => op[i] = ring.do_fft(el)
        }
    }
}

///
/// Left-hand side operand of an "RNS-gadget product", hence this struct stores
/// the gadget decomposition of a ring element `y`, w.r.t. the RNS gadget vector
/// `(lift((q / pi)^-1 mod pi) * q / pi)_i` where `q = p1 * ... * pm` is the ring modulus.
/// For more details, see [`DoubleRNSRingBase::gadget_product()`].
/// 
pub enum GadgetProductLhsOperand<'a, NumberRing, A = Global> 
    where NumberRing: DecomposableNumberRing<Zn>,
        A: Allocator + Clone
{
    LKSSStyle(LKSSGadgetProductLhsOperand<'a, NumberRing, A>),
    Naive(Vec<El<DoubleRNSRing<NumberRing, Zn, A>>, A>)
}

impl<'a, NumberRing, A> GadgetProductLhsOperand<'a, NumberRing, A>
    where NumberRing: DecomposableCyclotomicNumberRing<Zn>,
        DoubleRNSRingBase<NumberRing, Zn, A>: CyclotomicRing,
        A: Allocator + Clone
{
    pub fn apply_galois_action(&self, ring: &DoubleRNSRingBase<NumberRing, Zn, A>, g: ZnEl) -> Self {
        match self {
            GadgetProductLhsOperand::LKSSStyle(lkss_lhs_op) => GadgetProductLhsOperand::LKSSStyle(lkss_lhs_op.apply_galois_action(ring, g)),
            GadgetProductLhsOperand::Naive(els) => GadgetProductLhsOperand::Naive({
                let mut result = Vec::with_capacity_in(els.len(), els.allocator().clone());
                result.extend(els.iter().map(|el| ring.apply_galois_action(el, g)));
                result
            })
        }
    }
}

///
/// LKSS-style lhs gadget product operand; 
/// Stores the RNS-decomposition components of the element in NTT form, however not w.r.t. the whole RNS base
/// but only w.r.t. `output_moduli_count` of its RNS factors. Currently we use the last `output_moduli_count`
/// RNS factors.
/// 
pub struct LKSSGadgetProductLhsOperand<'a, NumberRing, A> 
    where NumberRing: DecomposableNumberRing<Zn>,
        A: Allocator + Clone
{
    output_moduli_count: usize,
    ring: PhantomData<&'a DoubleRNSRingBase<NumberRing, Zn, A>>,
    operands: Vec<Vec<ZnEl, A>, A>
}

impl<'a, NumberRing, A> LKSSGadgetProductLhsOperand<'a, NumberRing, A>
    where NumberRing: DecomposableCyclotomicNumberRing<Zn>,
        DoubleRNSRingBase<NumberRing, Zn, A>: CyclotomicRing,
        A: Allocator + Clone
{
    pub fn apply_galois_action(&self, ring: &DoubleRNSRingBase<NumberRing, Zn, A>, g: ZnEl) -> Self {
        let mut result_operands = Vec::with_capacity_in(self.operands.len(), self.operands.allocator().clone());
        for operand in self.operands.iter() {
            let mut result_op = Vec::with_capacity_in(operand.len(), operand.allocator().clone());
            result_op.resize_with(operand.len(), || ring.rns_base().at(0).zero());
            for k in 0..self.output_moduli_count {
                let ring_i = ring.rns_base().len() - self.output_moduli_count + k;
                <_ as DecomposedCyclotomicNumberRing<_>>::permute_galois_action(<NumberRing::DecomposedAsCyclotomic>::from_ref(ring.ring_decompositions().at(ring_i)), &operand[(k * ring.rank())..((k + 1) * ring.rank())], &mut result_op[(k * ring.rank())..((k + 1) * ring.rank())], g);
            }
            result_operands.push(result_op);
        }
        return LKSSGadgetProductLhsOperand {
            output_moduli_count: self.output_moduli_count,
            ring: PhantomData,
            operands: result_operands
        };
    }
}

pub enum ElRepr {
    Coeff, NTT
}

impl<NumberRing, A> DoubleRNSRingBase<NumberRing, Zn, A> 
    where NumberRing: DecomposableNumberRing<Zn>,
        A: Allocator + Clone
{
    ///
    /// `gadget_decompose()[decomposed_component][rns_base_index * self.rank() + coefficient_index]` contains the 
    /// `coefficient_index`-th fourier coefficient modulo `shortened_rns_base.at(rns_base_index)` of the 
    /// `decomposed_component`-th element of the gadget decomposition vector. Here `shortened_rns_base` is formed
    /// by the last `output_moduli_count` rns components of the main rns base.
    /// 
    /// The order of the fourier coefficients is the same as specified by the corresponding [`GeneralizedFFT`].
    /// 
    fn gadget_decompose(&self, el: CoeffEl<NumberRing, Zn, A>, output_moduli_count: usize) -> Vec<Vec<ZnEl, A>, A> {
        let mut result = Vec::new_in(self.allocator().clone());
        let el_as_matrix = self.as_matrix(&el);

        let homs = (0..output_moduli_count).map(|k| self.rns_base().at(self.rns_base().len() - output_moduli_count + k).can_hom::<StaticRing<i64>>(&StaticRing::<i64>::RING).unwrap()).collect::<Vec<_>>();
        for i in 0..self.rns_base().len() {
            let mut part = Vec::with_capacity_in(output_moduli_count * self.rank(), self.allocator().clone());
            part.extend((0..(output_moduli_count * self.rank())).map(|idx| {
                let k = idx / self.rank();
                let j = idx % self.rank();
                homs[k].map(self.rns_base().at(i).smallest_lift(self.rns_base().at(i).clone_el(el_as_matrix.at(i, j))))
            }));
            result.push(part);
            for k in 0..output_moduli_count {
                let ring_i = self.rns_base().len() - output_moduli_count + k;
                self.ring_decompositions().at(ring_i).fft_forward(&mut result.last_mut().unwrap()[(k * self.rank())..((k + 1) * self.rank())]);
            }
        }
        return result;
    }

    ///
    /// The number of moduli we need when performing the internal inner product
    /// during the KLSS-style product [https://ia.cr/2023/413]
    /// 
    fn get_gadget_product_modulo_count(&self) -> usize {
        let p_max = self.rns_base().as_iter().map(|Fp| int_cast(Fp.integer_ring().clone_el(Fp.modulus()), StaticRing::<i64>::RING, Fp.integer_ring())).max().unwrap();
        // the maximal size of the inner product of two gadget-decomposed elements
        let max_size_log2 = (self.rank() as f64 * p_max as f64 / 2.).log2() * 2. + (self.rns_base().len() as f64).log2() + 1.;
        let mut current_size_log2 = 0.;
        self.rns_base().as_iter().rev().take_while(|Fp| {
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
    pub fn to_gadget_product_lhs<'a>(&'a self, el: CoeffEl<NumberRing, Zn, A>) -> GadgetProductLhsOperand<'a, NumberRing, A> {
        record_time!("DoubleRNSRing::to_gadget_product_lhs", || {
            let output_moduli_count = self.get_gadget_product_modulo_count();
            if self.use_lkss_gadget_product() {
                GadgetProductLhsOperand::LKSSStyle(LKSSGadgetProductLhsOperand {
                    ring: PhantomData,
                    operands: self.gadget_decompose(el, output_moduli_count),
                    output_moduli_count: output_moduli_count
                })
            } else {
                let mut data = Vec::with_capacity_in(self.rns_base().len(), self.allocator().clone());
                for part in self.gadget_decompose(el, self.rns_base().len()).into_iter() {
                    data.push(DoubleRNSEl {
                        number_ring: PhantomData,
                        allocator: PhantomData,
                        data: part
                    })
                }
                GadgetProductLhsOperand::Naive(data)
            }
        })
    }

    ///
    /// Creates a [`GadgetProductRhsOperand`] representing 0. Its data (i.e. the noisy approximations
    /// of scalings of the base ring element) can be set later with [`GadgetProductRhsOperand::set_rns_factor()`].
    /// For more details, see [`DoubleRNSRingBase::gadget_product()`].
    /// 
    pub fn gadget_product_rhs_empty<'a>(&'a self) -> GadgetProductRhsOperand<'a, NumberRing, A> {
        let output_moduli_count = self.get_gadget_product_modulo_count();
        // if the RNS base is very short, we don't want to do LKSS style gadget products, mainly
        // for code simplicity, but we also don't expect much performance improvement
        if self.use_lkss_gadget_product() {
            let shortened_rns_base = zn_rns::Zn::<Zn, BigIntRing>::new(self.rns_base().as_iter().skip(self.rns_base().len() - output_moduli_count).cloned().collect(), BigIntRing::RING);
            let mut operands = Vec::with_capacity_in(self.rns_base().len(), self.allocator().clone());
            operands.extend((0..self.rns_base().len()).map(|_| Vec::new_in(self.allocator().clone())));
            GadgetProductRhsOperand::LKSSStyle(LKSSGadgetProductRhsOperand {
                ring: self,
                operands: operands,
                conversions: (0..self.rns_base().len()).map(|i| UsedBaseConversion::new_with(
                    shortened_rns_base.get_ring().as_iter().cloned().collect(), 
                    vec![*self.rns_base().at(i)],
                    self.allocator().clone()
                )).collect::<Vec<_>>(),
                shortened_rns_base: shortened_rns_base
            })
        } else {
            GadgetProductRhsOperand::Naive(self, (0..self.rns_base().len()).map(|_| self.zero()).collect())
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
    /// # use feanor_math::homomorphism::*;
    /// # use feanor_math::rings::zn::*;
    /// # use feanor_math::rings::zn::zn_64::*;
    /// # use feanor_math::rings::finite::*;
    /// # use feanor_math::integer::BigIntRing;
    /// # use feanor_math::algorithms::fft::cooley_tuckey::CooleyTuckeyFFT;
    /// # use he_ring::rings::double_rns_ring::DoubleRNSRingBase;
    /// # use he_ring::rings::pow2_cyclotomic::Pow2CyclotomicDecomposableNumberRing;
    /// # use feanor_math::rings::extension::FreeAlgebraStore;
    /// # use feanor_math::seq::*;
    /// let rns_base = vec![Zn::new(17), Zn::new(97), Zn::new(113)];
    /// let ring = DoubleRNSRingBase::new(Pow2CyclotomicDecomposableNumberRing::new(16), zn_rns::Zn::new(rns_base.clone(), BigIntRing::RING));
    /// let mut rng = oorandom::Rand64::new(1);
    /// 
    /// // build the right-hand side operand
    /// let rhs = ring.random_element(|| rng.rand_u64());
    /// let mut rhs_op = ring.get_ring().gadget_product_rhs_empty();
    /// let gadget_vector = |i: usize| ring.base_ring().get_ring().from_congruence((0..3).map(|j| ring.base_ring().get_ring().at(j).int_hom().map(if i == j { 1 } else { 0 })));
    /// for i in 0..3 {
    ///     // set the i-th component to `gadget_vector(i) * rhs`, for now without noise
    ///     rhs_op.set_rns_factor(i, ring.get_ring().undo_fft(ring.inclusion().mul_ref_map(&rhs, &gadget_vector(i))));
    /// }
    /// 
    /// // compute the gadget product
    /// let lhs = ring.random_element(|| rng.rand_u64());
    /// let actual = ring.get_ring().gadget_product_ntt(&ring.get_ring().to_gadget_product_lhs(ring.get_ring().undo_fft(ring.clone_el(&lhs))), &rhs_op);
    /// assert_el_eq!(&ring, &ring.mul_ref(&lhs, &rhs), &ring.get_ring().gadget_product_ntt(&ring.get_ring().to_gadget_product_lhs(ring.get_ring().undo_fft(lhs)), &rhs_op));
    /// ```
    /// To demonstrate how this keeps small error terms small, consider the following variation of the previous example:
    /// 
    /// ```
    /// # use feanor_math::ring::*;
    /// # use feanor_math::assert_el_eq;
    /// # use feanor_math::homomorphism::*;
    /// # use feanor_math::rings::zn::*;
    /// # use feanor_math::rings::zn::zn_64::*;
    /// # use feanor_math::algorithms::fft::cooley_tuckey::CooleyTuckeyFFT;
    /// # use he_ring::rings::double_rns_ring::DoubleRNSRingBase;
    /// # use he_ring::rings::pow2_cyclotomic::Pow2CyclotomicDecomposableNumberRing;
    /// # use feanor_math::rings::extension::FreeAlgebraStore;
    /// # use feanor_math::integer::BigIntRing;
    /// # use feanor_math::rings::finite::*;
    /// # use feanor_math::integer::int_cast;
    /// # use feanor_math::primitive_int::StaticRing;
    /// # use feanor_math::seq::*;
    /// let rns_base = vec![Zn::new(17), Zn::new(97), Zn::new(113)];
    /// let ring = DoubleRNSRingBase::new(Pow2CyclotomicDecomposableNumberRing::new(16), zn_rns::Zn::new(rns_base.clone(), BigIntRing::RING));
    /// 
    /// let mut rng = oorandom::Rand64::new(1);
    /// 
    /// // build the right-hand side operand
    /// let rhs = ring.random_element(|| rng.rand_u64());
    /// let mut error = || ring.get_ring().do_fft(ring.get_ring().sample_from_coefficient_distribution(|| (rng.rand_u64() % 3) as i32 - 1));
    /// let mut rhs_op = ring.get_ring().gadget_product_rhs_empty();
    /// let gadget_vector = |i: usize| ring.base_ring().get_ring().from_congruence((0..3).map(|j| ring.base_ring().get_ring().at(j).int_hom().map(if i == j { 1 } else { 0 })));
    /// for i in 0..3 {
    ///     // set the i-th component to `gadget_vector(i) * rhs`, with possibly some noise included
    ///     rhs_op.set_rns_factor(i, ring.get_ring().undo_fft(ring.add(ring.inclusion().mul_ref_map(&rhs, &gadget_vector(i)), error())));
    /// }
    /// 
    /// // compute the gadget product
    /// let lhs = ring.random_element(|| rng.rand_u64());
    /// let actual = ring.get_ring().gadget_product_ntt(&ring.get_ring().to_gadget_product_lhs(ring.get_ring().undo_fft(ring.clone_el(&lhs))), &rhs_op);
    /// 
    /// // the final result should be close to `lhs * rhs`, except for some noise
    /// let expected = ring.mul_ref(&lhs, &rhs);
    /// let error = ring.sub(expected, actual);
    /// let error_coefficients = ring.wrt_canonical_basis(&error);
    /// let max_allowed_error = (113 / 2) * 8 * 3;
    /// assert!((0..8).all(|i| int_cast(ring.base_ring().smallest_lift(error_coefficients.at(i)), StaticRing::<i64>::RING, BigIntRing::RING).abs() <= max_allowed_error));
    /// ```
    /// 
    pub fn gadget_product_ntt(&self, lhs: &GadgetProductLhsOperand<NumberRing, A>, rhs: &GadgetProductRhsOperand<NumberRing, A>) -> DoubleRNSEl<NumberRing, Zn, A> {
        match (lhs, rhs) {
            (GadgetProductLhsOperand::LKSSStyle(lhs), GadgetProductRhsOperand::LKSSStyle(rhs)) => self.do_fft(self.gadget_product_lkss(lhs, rhs)),
            (GadgetProductLhsOperand::Naive(lhs), GadgetProductRhsOperand::Naive(_, rhs)) => self.gadget_product_naive(lhs, rhs),
            _ => panic!("Illegal combination of GadgetProductOperands; Maybe they were created by different rings?")
        }
    }

    fn gadget_product_naive(&self, lhs: &[El<DoubleRNSRing<NumberRing, Zn, A>>], rhs: &[El<DoubleRNSRing<NumberRing, Zn, A>>]) -> El<DoubleRNSRing<NumberRing, Zn, A>> {
        record_time!("DoubleRNSRing::gadget_product_ntt::naive", || {
            <_ as RingBase>::sum(self, lhs.iter().zip(rhs.iter()).map(|(l, r)| self.mul_ref(l, r)))
        })
    }

    fn gadget_product_lkss(&self, lhs: &LKSSGadgetProductLhsOperand<NumberRing, A>, rhs: &LKSSGadgetProductRhsOperand<NumberRing, A>) -> CoeffEl<NumberRing, Zn, A> {
        record_time!("DoubleRNSRing::gadget_product_lkss", || {
            assert_eq!(lhs.output_moduli_count, rhs.shortened_rns_base.get_ring().len());
            let output_moduli_count = lhs.output_moduli_count;
            let shortened_rns_base = rhs.shortened_rns_base.get_ring();

            let mut result = self.non_fft_zero();
            for j in 0..self.rns_base().len() {
                let mut summand = record_time!("DoubleRNSRing::gadget_product_lkss::sum", || {
                    let mut summand = Vec::with_capacity_in(shortened_rns_base.len() * self.rank(), self.allocator().clone());
                    for k in 0..shortened_rns_base.len() {
                        for l in 0..self.rank() {
                            let Fp = shortened_rns_base.at(k);
                            summand.push(<_ as ComputeInnerProduct>::inner_product_ref(Fp.get_ring(), (0..self.rns_base().len()).map(|i| (&lhs.operands[i][k * self.rank() + l], rhs.at(i, j, k, l)))));
                        }
                    }
                    summand
                });
                record_time!("DoubleRNSRing::gadget_product_lkss::ffts", || {
                    for k in 0..output_moduli_count {
                        self.ring_decompositions()[self.rns_base().len() - output_moduli_count + k].fft_backward(&mut summand[(k * self.rank())..((k + 1) * self.rank())]);
                    }
                });
                record_time!("DoubleRNSRing::gadget_product_lkss::lifting", || {
                    rhs.conversions[j].apply(
                        Submatrix::from_1d(&summand, shortened_rns_base.len(), self.rank()),
                        self.as_matrix_mut(&mut result).restrict_rows(j..(j + 1))
                    );
                });
            }
            return result;
        })
    }

    pub fn preferred_output_repr(&self, lhs: &GadgetProductLhsOperand<NumberRing, A>, rhs: &GadgetProductRhsOperand<NumberRing, A>) -> ElRepr {
        match (lhs, rhs) {
            (GadgetProductLhsOperand::LKSSStyle(_), GadgetProductRhsOperand::LKSSStyle(_)) => ElRepr::Coeff,
            (GadgetProductLhsOperand::Naive(_), GadgetProductRhsOperand::Naive(_, _)) => ElRepr::NTT,
            _ => panic!("Illegal combination of GadgetProductOperands; Maybe they were created by different rings?")
        }
    }

    ///
    /// The gadget product without final FFT. See [`Self::gadget_product_ntt()`] for a description.
    /// 
    /// The implementation uses the KLSS-style algorithm [https://ia.cr/2023/413].
    /// 
    pub fn gadget_product_coeff(&self, lhs: &GadgetProductLhsOperand<NumberRing, A>, rhs: &GadgetProductRhsOperand<NumberRing, A>) -> CoeffEl<NumberRing, Zn, A> {
        match (lhs, rhs) {
            (GadgetProductLhsOperand::LKSSStyle(lhs), GadgetProductRhsOperand::LKSSStyle(rhs)) => self.gadget_product_lkss(lhs, rhs),
            (GadgetProductLhsOperand::Naive(lhs), GadgetProductRhsOperand::Naive(_, rhs)) => self.undo_fft(self.gadget_product_naive(lhs, rhs)),
            _ => panic!("Illegal combination of GadgetProductOperands; Maybe they were created by different rings?")
        }
    }
}
