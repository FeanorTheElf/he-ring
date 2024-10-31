use feanor_math::assert_el_eq;
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
use subvector::SubvectorView;

use std::alloc::Allocator;
use std::alloc::Global;
use std::marker::PhantomData;
use std::ops::Range;

use crate::rings::double_rns_ring::*;
use crate::cyclotomic::CyclotomicRing;
use crate::rings::pow2_cyclotomic::Pow2CyclotomicDecomposableNumberRing;
use crate::rings::pow2_cyclotomic::Pow2CyclotomicNumberRing;
use crate::rnsconv::*;
use crate::rings::number_ring::*;
use crate::IsEq;

// don't use the matrix one here, as in our case, one dimension is very small (e.g. 3), and the matrix
// version pads to multiples of 16
type UsedBaseConversion<A> = lift::AlmostExactBaseConversion<A>;

///
/// Right-hand side operand of a "gadget product".
/// 
/// For details, see [`DoubleRNSRingBase::gadget_product_ntt()`].
/// 
#[allow(private_interfaces)]
pub enum GadgetProductRhsOperand<'a, NumberRing, A = Global> 
    where NumberRing: HENumberRing<Zn>,
        A: Allocator + Clone
{
    LKSSStyle(LKSSGadgetProductRhsOperand<'a, NumberRing, A>),
    Naive(NaiveGadgetProductRhsOperand<'a, NumberRing, A>)
}

impl<'a, NumberRing, A> GadgetProductRhsOperand<'a, NumberRing, A> 
    where NumberRing: HENumberRing<Zn>,
        A: Allocator + Clone
{
    pub fn gadget_vector<'b>(&'b self) -> &'b [Range<usize>] {
        match self {
            GadgetProductRhsOperand::LKSSStyle(op) => op.gadget_vector(),
            GadgetProductRhsOperand::Naive(op) => op.gadget_vector()
        }
    }

    pub fn set_rns_factor(&mut self, i: usize, el: CoeffEl<NumberRing, Zn, A>) {
        match self {
            GadgetProductRhsOperand::LKSSStyle(op) => op.set_rns_factor(i, el),
            GadgetProductRhsOperand::Naive(op) => op.set_rns_factor(i, el)
        }
    }
}

///
/// Lhs gadget product operand for the straightforward gadget-product implementation;
/// 
/// Stores ring elements in double-RNS form that represent noisy approximations to `g[i]`.
/// 
struct NaiveGadgetProductRhsOperand<'a, NumberRing, A = Global> 
    where NumberRing: HENumberRing<Zn>,
        A: Allocator + Clone
{
    ring: &'a DoubleRNSRingBase<NumberRing, Zn, A>,
    operands: Vec<El<DoubleRNSRing<NumberRing, Zn, A>>, A>,
    digits: Vec<Range<usize>>
}

impl<'a, NumberRing, A> NaiveGadgetProductRhsOperand<'a, NumberRing, A> 
    where NumberRing: HENumberRing<Zn>,
        A: Allocator + Clone
{
    fn create_empty(ring: &'a DoubleRNSRingBase<NumberRing, Zn, A>, digits: usize) -> Self {
        let mut operands = Vec::with_capacity_in(digits, ring.allocator().clone());
        operands.extend((0..digits).map(|_| ring.zero()));
        return NaiveGadgetProductRhsOperand {
            ring: ring,
            operands: operands,
            digits: prime_factor_groups(ring.rns_base().len(), digits).iter().collect()
        };
    }

    fn gadget_vector<'b>(&'b self) -> &'b [Range<usize>] {
        &self.digits
    }

    fn set_rns_factor(&mut self, i: usize, el: CoeffEl<NumberRing, Zn, A>) {
        self.operands[i] = self.ring.do_fft(el);
    }
}

///
/// LKSS-style lhs gadget product operand; 
/// 
/// For each gadget vector entry `i`, we have a ring element `ai` that represents
/// a noisy approximation to `g[i]`. Instead of storing this element `ai` directly,
/// we store the gadget-decomposed parts `ai1, ..., aij` modulo a "shortened RNS base".
/// Since `aij` are outputs of gadget decomposition, they are small, thus already 
/// uniquely specified by a shorter RNS base.
/// 
struct LKSSGadgetProductRhsOperand<'a, NumberRing, A = Global> 
    where NumberRing: HENumberRing<Zn>,
        A: Allocator + Clone
{
    shortened_rns_base: zn_rns::Zn<Zn, BigIntRing>,
    ring: &'a DoubleRNSRingBase<NumberRing, Zn, A>,
    /// the `i`-th entry contains the gadget-decomposed approximation to `g[i]`, as returned
    /// by [`gadget_decompose()`]
    operands: Vec<Vec<Vec<ZnEl, A>>>,
    conversions: Vec<UsedBaseConversion<A>>,
    digits: Vec<Range<usize>>
}

impl<'a, NumberRing, A> LKSSGadgetProductRhsOperand<'a, NumberRing, A> 
    where NumberRing: HENumberRing<Zn>,
        A: Allocator + Clone
{
    fn at(&self, i: usize, j: usize, k: usize, l: usize) -> &ZnEl {
        debug_assert!(i < self.digits.len());
        debug_assert!(j < self.digits.len());
        debug_assert!(k < self.shortened_rns_base.len());
        debug_assert!(l < self.ring.rank());
        &self.operands[i][j][k * self.ring.rank() + l]
    }

    fn create_empty(ring: &'a DoubleRNSRingBase<NumberRing, Zn, A>, digits: usize, shortened_rns_base_len: usize) -> Self {
        let shortened_rns_base = zn_rns::Zn::<Zn, BigIntRing>::new(ring.rns_base().as_iter().skip(ring.rns_base().len() - shortened_rns_base_len).cloned().collect(), BigIntRing::RING);
        let mut operands = Vec::with_capacity(digits);
        operands.extend((0..digits).map(|_| Vec::new()));
        let digits = prime_factor_groups(ring.rns_base().len(), digits);
        return LKSSGadgetProductRhsOperand {
            ring: ring,
            operands: operands,
            conversions: digits.iter().map(|range| UsedBaseConversion::new_with(
                shortened_rns_base.get_ring().as_iter().cloned().collect(), 
                range.map(|i| *ring.rns_base().at(i)).collect(),
                ring.allocator().clone()
            )).collect(),
            shortened_rns_base: shortened_rns_base,
            digits: digits.iter().collect()
        };
    }

    fn gadget_vector<'b>(&'b self) -> &'b [Range<usize>] {
        &self.digits
    }

    fn set_rns_factor(&mut self, i: usize, el: CoeffEl<NumberRing, Zn, A>) {
        self.operands[i] = self.ring.gadget_decompose(el, &self.digits, self.shortened_rns_base.get_ring().len());
    }
}

///
/// Left-hand side operand of a "gadget product".
/// 
/// For details, see [`DoubleRNSRingBase::gadget_product_ntt()`].
/// 
#[allow(private_interfaces)]
pub enum GadgetProductLhsOperand<'a, NumberRing, A = Global> 
    where NumberRing: HENumberRing<Zn>,
        A: Allocator + Clone
{
    LKSSStyle(LKSSGadgetProductLhsOperand<'a, NumberRing, A>),
    Naive(NaiveGadgetProductLhsOperand<'a, NumberRing, A>)
}

impl<'a, NumberRing, A> GadgetProductLhsOperand<'a, NumberRing, A>
    where NumberRing: HECyclotomicNumberRing<Zn>,
        DoubleRNSRingBase<NumberRing, Zn, A>: CyclotomicRing,
        A: Allocator + Clone
{
    pub fn apply_galois_action(&self, ring: &DoubleRNSRingBase<NumberRing, Zn, A>, g: ZnEl) -> Self {
        match self {
            GadgetProductLhsOperand::LKSSStyle(lkss_lhs_op) => GadgetProductLhsOperand::LKSSStyle(lkss_lhs_op.apply_galois_action(ring, g)),
            GadgetProductLhsOperand::Naive(lhs_op) => GadgetProductLhsOperand::Naive(lhs_op.apply_galois_action(ring, g))
        }
    }
}

///
/// Direct implementation of the left-hand side operand of a gadget product.
/// 
/// Stores the gadget decomposition of a ring element `x`, i.e. small elements `xi`
/// such that `x = sum_i g[i] xi`.
/// 
struct NaiveGadgetProductLhsOperand<'a, NumberRing, A> 
    where NumberRing: HENumberRing<Zn>,
        A: Allocator + Clone
{
    ring: PhantomData<&'a DoubleRNSRingBase<NumberRing, Zn, A>>,
    operands: Vec<El<DoubleRNSRing<NumberRing, Zn, A>>, A>
}

impl<'a, NumberRing, A> NaiveGadgetProductLhsOperand<'a, NumberRing, A>
    where NumberRing: HECyclotomicNumberRing<Zn>,
        DoubleRNSRingBase<NumberRing, Zn, A>: CyclotomicRing,
        A: Allocator + Clone
{
    fn apply_galois_action(&self, ring: &DoubleRNSRingBase<NumberRing, Zn, A>, g: ZnEl) -> Self {
        let mut result = Vec::with_capacity_in(self.operands.len(), self.operands.allocator().clone());
        result.extend(self.operands.iter().map(|el| ring.apply_galois_action(el, g)));
        return NaiveGadgetProductLhsOperand {
            ring: PhantomData,
            operands: result
        };
    }
}

impl<'a, NumberRing, A> NaiveGadgetProductLhsOperand<'a, NumberRing, A>
    where NumberRing: HENumberRing<Zn>,
        A: Allocator + Clone
{
    fn new_from_element(ring: &DoubleRNSRingBase<NumberRing, Zn, A>, el: CoeffEl<NumberRing, Zn, A>, digits: usize) -> Self {
        let digits = prime_factor_groups(ring.rns_base().len(), digits).iter().collect::<Vec<_>>();
        let mut data = Vec::with_capacity_in(ring.rns_base().len(), ring.allocator().clone());
        for part in ring.gadget_decompose(el, &digits, ring.rns_base().len()).into_iter() {
            data.push(DoubleRNSEl {
                number_ring: PhantomData,
                allocator: PhantomData,
                el_wrt_mult_basis: part
            })
        }
        return NaiveGadgetProductLhsOperand {
            ring: PhantomData,
            operands: data
        };
    }
}

///
/// LKSS-style lhs gadget product operand.
/// 
/// Stores the gadget decomposition of a ring element `x`, i.e. small elements `xi`
/// such that `x = sum_i g[i] xi`. However, we store `xi` only modulo a "shortened
/// RNS base", which is fine since it is small, so already uniquely defined by less
/// RNS components.
/// 
struct LKSSGadgetProductLhsOperand<'a, NumberRing, A> 
    where NumberRing: HENumberRing<Zn>,
        A: Allocator + Clone
{
    output_moduli_count: usize,
    ring: PhantomData<&'a DoubleRNSRingBase<NumberRing, Zn, A>>,
    operands: Vec<Vec<ZnEl, A>>
}


impl<'a, NumberRing, A> LKSSGadgetProductLhsOperand<'a, NumberRing, A>
    where NumberRing: HENumberRing<Zn>,
        A: Allocator + Clone
{
    fn new_from_element(ring: &DoubleRNSRingBase<NumberRing, Zn, A>, el: CoeffEl<NumberRing, Zn, A>, digits: usize, output_moduli_count: usize) -> Self {
        let digits = prime_factor_groups(ring.rns_base().len(), digits).iter().collect::<Vec<_>>();
        LKSSGadgetProductLhsOperand {
            ring: PhantomData,
            operands: ring.gadget_decompose(el, &digits, output_moduli_count),
            output_moduli_count: output_moduli_count
        }
    }
}

impl<'a, NumberRing, A> LKSSGadgetProductLhsOperand<'a, NumberRing, A>
    where NumberRing: HECyclotomicNumberRing<Zn>,
        DoubleRNSRingBase<NumberRing, Zn, A>: CyclotomicRing,
        A: Allocator + Clone
{
    fn apply_galois_action(&self, ring: &DoubleRNSRingBase<NumberRing, Zn, A>, g: ZnEl) -> Self {
        let mut result_operands = Vec::with_capacity_in(self.operands.len(), self.operands.allocator().clone());
        for operand in self.operands.iter() {
            let mut result_op = Vec::with_capacity_in(operand.len(), operand.allocator().clone());
            result_op.resize_with(operand.len(), || ring.rns_base().at(0).zero());
            for k in 0..self.output_moduli_count {
                let ring_i = ring.rns_base().len() - self.output_moduli_count + k;
                <_ as HECyclotomicNumberRingMod<_>>::permute_galois_action(
                    <NumberRing::DecomposedAsCyclotomic>::from_ref(ring.ring_decompositions().at(ring_i)), 
                    &operand[(k * ring.rank())..((k + 1) * ring.rank())], &mut result_op[(k * ring.rank())..((k + 1) * ring.rank())], 
                    g
                );
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

fn prime_factor_groups<'a>(rns_base_len: usize, digits: usize) -> impl 'a + VectorFn<Range<usize>> {
    assert!(digits <= rns_base_len);
    assert!(digits >= 2);
    let primes_per_digit = rns_base_len / digits;
    let digits_with_one_more_prime = rns_base_len - digits * primes_per_digit;
    return (0..digits).map_fn(move |i| {
        if i < digits_with_one_more_prime {
            (i * (primes_per_digit + 1))..((i + 1) * (primes_per_digit + 1))
        } else {
            (digits_with_one_more_prime * (primes_per_digit + 1) + (i - digits_with_one_more_prime) * primes_per_digit)..(digits_with_one_more_prime * (primes_per_digit + 1) + (i + 1 - digits_with_one_more_prime) * primes_per_digit)
        }
    });
}

impl<NumberRing, A> DoubleRNSRingBase<NumberRing, Zn, A> 
    where NumberRing: HENumberRing<Zn>,
        A: Allocator + Clone
{
    ///
    /// `gadget_decompose()[decomposed_component][rns_base_index * self.rank() + coefficient_index]` contains the 
    /// `coefficient_index`-th fourier coefficient modulo `shortened_rns_base.at(rns_base_index)` of the 
    /// `decomposed_component`-th coefficient w.r.t. the gadget decomposition vector. Here `shortened_rns_base` 
    /// is formed by the last `output_moduli_count` rns components of the main rns base.
    /// 
    /// In particular, this means
    ///  - `decomposed_component` is between `0` (inclusive) and `digits` (exclusive)
    ///  - `rns_base_index` is between `0` (inclusive) and `output_moduli_count` (exclusive)
    ///  - `coefficient_index` is between `0` (inclusive) and `self.rank()` (exclusive)
    /// 
    /// The order of the fourier coefficients is the same as specified by the corresponding [`GeneralizedFFT`].
    /// 
    fn gadget_decompose(&self, el: CoeffEl<NumberRing, Zn, A>, digits: &[Range<usize>], output_moduli_count: usize) -> Vec<Vec<ZnEl, A>> {
        let ZZbig = BigIntRing::RING;
        let digit_bases = digits.iter().map(|range| zn_rns::Zn::new(range.clone().map(|i| self.rns_base().at(i)).collect::<Vec<_>>(), ZZbig)).collect::<Vec<_>>();

        let mut result = Vec::new();
        let el_as_matrix = self.as_matrix_wrt_small_basis(&el);

        let homs = (0..output_moduli_count).map(|k| self.rns_base().at(self.rns_base().len() - output_moduli_count + k).can_hom(&ZZbig).unwrap()).collect::<Vec<_>>();
        let homs_ref = &homs;
        for (digit, base) in digits.iter().zip(digit_bases.iter()) {

            let conversion = UsedBaseConversion::new_with(
                SubvectorView::new(self.rns_base()).restrict(digit.clone()).as_iter().map(|Zn| Zn.clone()).collect::<Vec<_>>(),
                homs.iter().map(|h| **h.codomain()).collect::<Vec<_>>(),
                Global
            );

            let mut part = Vec::with_capacity_in(output_moduli_count * self.rank(), self.allocator().clone());
            part.extend((0..output_moduli_count).flat_map(|i| (0..self.rank()).map(move |_| homs_ref[i].codomain().zero())));

            conversion.apply(
                self.as_matrix_wrt_small_basis(&el).restrict_rows(digit.clone()),
                SubmatrixMut::from_1d(&mut part[..], output_moduli_count, self.rank())
            );

            result.push(part);
            for i in 0..output_moduli_count {
                let ring_i = self.rns_base().len() - output_moduli_count + i;
                self.ring_decompositions().at(ring_i).small_basis_to_mult_basis(&mut result.last_mut().unwrap()[(i * self.rank())..((i + 1) * self.rank())]);
            }
        }
        return result;
    }

    ///
    /// The number of moduli we need when performing the internal inner product
    /// during the KLSS-style product [https://ia.cr/2023/413]
    /// 
    fn get_gadget_product_modulo_count(&self, digits: usize) -> usize {
        let ZZbig = BigIntRing::RING;
        let max_digit = prime_factor_groups(self.rns_base().len(), digits).iter()
            .map(|range| ZZbig.prod(range.map(|i| int_cast(*self.rns_base().at(i).modulus(), ZZbig, StaticRing::<i64>::RING))))
            .map(|digit| ZZbig.to_float_approx(&digit))
            .max_by(f64::total_cmp).unwrap();
        // the maximal size of the inner product of two gadget-decomposed elements
        let max_size_log2 = (max_digit.log2() - 1.) * 2. + self.number_ring().product_expansion_factor().log2() + (digits as f64).log2();
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

    fn use_lkss_gadget_product(&self, digits: usize) -> bool {
        self.get_gadget_product_modulo_count(digits) + 1 < digits && !cfg!(feature = "force_naive_gadget_product")
    }

    ///
    /// Computes the data necessary to perform a "gadget product" with the given operand as
    /// left-hand side. This can be though of computing the gadget decomposition of the argument.
    /// For more details, see [`DoubleRNSRingBase::gadget_product_ntt()`].
    /// 
    pub fn to_gadget_product_lhs<'a>(&'a self, el: CoeffEl<NumberRing, Zn, A>, digits: usize) -> GadgetProductLhsOperand<'a, NumberRing, A> {
        record_time!("DoubleRNSRing::to_gadget_product_lhs", || {
            let output_moduli_count = self.get_gadget_product_modulo_count(digits);
            if self.use_lkss_gadget_product(digits) {
                GadgetProductLhsOperand::LKSSStyle(LKSSGadgetProductLhsOperand::new_from_element(self, el, digits, output_moduli_count))
            } else {
                GadgetProductLhsOperand::Naive(NaiveGadgetProductLhsOperand::new_from_element(self, el, digits))
            }
        })
    }

    ///
    /// Creates a [`GadgetProductRhsOperand`] representing 0. Its data (i.e. the noisy approximations
    /// of scalings of the base ring element) can be set later with [`GadgetProductRhsOperand::set_rns_factor()`].
    /// For more details, see [`DoubleRNSRingBase::gadget_product_ntt()`].
    /// 
    pub fn gadget_product_rhs_empty<'a, const LOG: bool>(&'a self, digits: usize) -> GadgetProductRhsOperand<'a, NumberRing, A> {
        let output_moduli_count = self.get_gadget_product_modulo_count(digits);
        // if the RNS base is very short, we don't want to do LKSS style gadget products, mainly
        // for code simplicity, but we also don't expect much performance improvement
        if self.use_lkss_gadget_product(digits) {
            if LOG {
                println!("Use LKSS for gadget product, digits = {}, len(short RNS base) = {}, len(full RNS base) = {}", digits, output_moduli_count, self.rns_base().len());
            }
            GadgetProductRhsOperand::LKSSStyle(LKSSGadgetProductRhsOperand::create_empty(self, digits, output_moduli_count))
        } else {
            if LOG {
                println!("Use naive gadget product, digits = {}, len(full RNS base) = {}", digits, self.rns_base().len());
            }
            GadgetProductRhsOperand::Naive(NaiveGadgetProductRhsOperand::create_empty(self, digits))
        }
    }

    ///
    /// Computes the "RNS-gadget product" of two elements in this ring, as often required
    /// in HE scenarios. A "gadget product" computes the approximate product of two
    /// ring elements `x` and `y` by using `y` and multiple scaled & noisy approximations 
    /// to `x`. This function only supports the gadget vector given by a decomposition
    /// `q = D1 ... Dr` into coprime "digits".
    /// 
    /// # What exactly is a "gadget product"?
    /// 
    /// In an HE setting, we often have a noisy approximation to some value `x`, say
    /// `x + e`. Now the normal product `(x + e)y = xy + ey` includes an error of `ey`, which
    /// (if `y` is arbitrary) is not in any way an approximation to `xy` anymore. Instead,
    /// we can take a so-called "gadget vector" `g` and provide multiple noisy scalings of `x`, 
    /// say `g[1] * x + e1` to `g[r] * x + er`.
    /// Using these, we can approximate `xy` by computing a gadget-decomposition 
    /// `y = g[1] * y1 + ... + g[m] * ym` where the values `yi` are small, and then use 
    /// `y1 (g[1] * x + e1) + ... + ym (g[m] * x + em)` as an approximation to `xy`.
    /// 
    /// The gadget vector used for this "RNS-gadget product" is the one given by the unit vectors
    /// in the decomposition `q = D1 ... Dr` into pairwise coprime "digits". In the simplest case,
    /// those digits are just the prime factors of `q`. However, it is usually beneficial to
    /// group multiple prime factors into a single digit, since decreasing the number of digits
    /// will significantly decrease the work we have to do when computing the inner product
    /// `sum_i (g[i] xi + ei) yi`. Note that this will of course decrease the quality of 
    /// approximation to `xy` (i.e. increase the error `sum_i yi ei`). Hence, choose the
    /// parameter `digits` appropriately.
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
    /// let digits = 3;
    /// 
    /// // build the right-hand side operand
    /// let rhs = ring.random_element(|| rng.rand_u64());
    /// let mut rhs_op = ring.get_ring().gadget_product_rhs_empty::<false>(digits);
    /// let gadget_vector = |i: usize| ring.base_ring().get_ring().from_congruence((0..3).map(|j| ring.base_ring().get_ring().at(j).int_hom().map(if i == j { 1 } else { 0 })));
    /// for i in 0..3 {
    ///     // set the i-th component to `gadget_vector(i) * rhs`, for now without noise
    ///     rhs_op.set_rns_factor(i, ring.get_ring().undo_fft(ring.inclusion().mul_ref_map(&rhs, &gadget_vector(i))));
    /// }
    /// 
    /// // compute the gadget product
    /// let lhs = ring.random_element(|| rng.rand_u64());
    /// let actual = ring.get_ring().gadget_product_ntt(&ring.get_ring().to_gadget_product_lhs(ring.get_ring().undo_fft(ring.clone_el(&lhs)), digits), &rhs_op);
    /// assert_el_eq!(&ring, &ring.mul_ref(&lhs, &rhs), &ring.get_ring().gadget_product_ntt(&ring.get_ring().to_gadget_product_lhs(ring.get_ring().undo_fft(lhs), digits), &rhs_op));
    /// ```
    /// To demonstrate how this keeps small error terms small, consider the following variation of the previous example:
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
    /// let mut rng = oorandom::Rand64::new(1);
    /// // we have digits == rns_base.len(), so the gadget vector has entries exactly the "CRT unit vectors" ei with ei = 1 mod pi, ei = 0 mod pj for j != i
    /// let digits = 3;
    /// 
    /// // build the right-hand side operand
    /// let rhs = ring.random_element(|| rng.rand_u64());
    /// let mut error = || ring.get_ring().do_fft(ring.get_ring().sample_from_coefficient_distribution(|| (rng.rand_u64() % 3) as i32 - 1));
    /// let mut rhs_op = ring.get_ring().gadget_product_rhs_empty::<false>(digits);
    /// let gadget_vector = |i: usize| ring.base_ring().get_ring().from_congruence((0..3).map(|j| ring.base_ring().get_ring().at(j).int_hom().map(if i == j { 1 } else { 0 })));
    /// for i in 0..3 {
    ///     // set the i-th component to `gadget_vector(i) * rhs`, with possibly some noise included
    ///     rhs_op.set_rns_factor(i, ring.get_ring().undo_fft(ring.add(ring.inclusion().mul_ref_map(&rhs, &gadget_vector(i)), error())));
    /// }
    /// 
    /// // compute the gadget product
    /// let lhs = ring.random_element(|| rng.rand_u64());
    /// let actual = ring.get_ring().gadget_product_ntt(&ring.get_ring().to_gadget_product_lhs(ring.get_ring().undo_fft(ring.clone_el(&lhs)), digits), &rhs_op);
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
            (GadgetProductLhsOperand::Naive(lhs), GadgetProductRhsOperand::Naive(rhs)) => self.gadget_product_naive(lhs, rhs),
            _ => panic!("Illegal combination of GadgetProductOperands; Maybe they were created by different rings?")
        }
    }

    fn gadget_product_naive(&self, lhs: &NaiveGadgetProductLhsOperand<NumberRing, A>, rhs: &NaiveGadgetProductRhsOperand<NumberRing, A>) -> El<DoubleRNSRing<NumberRing, Zn, A>> {
        record_time!("DoubleRNSRing::gadget_product_ntt::naive", || {
            assert_eq!(lhs.operands.len(), rhs.operands.len());
            <_ as RingBase>::sum(self, lhs.operands.iter().zip(rhs.operands.iter()).map(|(l, r)| {
                let result = self.mul_ref(l, r);
                return result;
            }))
        })
    }

    fn gadget_product_lkss(&self, lhs: &LKSSGadgetProductLhsOperand<NumberRing, A>, rhs: &LKSSGadgetProductRhsOperand<NumberRing, A>) -> CoeffEl<NumberRing, Zn, A> {
        record_time!("DoubleRNSRing::gadget_product_lkss", || {
            assert_eq!(lhs.output_moduli_count, rhs.shortened_rns_base.get_ring().len());
            assert_eq!(lhs.operands.len(), rhs.operands.len());
            let digits = &rhs.digits;
            let output_moduli_count = lhs.output_moduli_count;
            let shortened_rns_base = rhs.shortened_rns_base.get_ring();

            let mut result = self.non_fft_zero();
            for j in 0..digits.len() {
                let mut summand = record_time!("DoubleRNSRing::gadget_product_lkss::sum", || {
                    let mut summand = Vec::with_capacity_in(shortened_rns_base.len() * self.rank(), self.allocator().clone());
                    for k in 0..shortened_rns_base.len() {
                        for l in 0..self.rank() {
                            let Fp = shortened_rns_base.at(k);
                            summand.push(<_ as ComputeInnerProduct>::inner_product_ref(Fp.get_ring(), (0..digits.len()).map(|i| (&lhs.operands[i][k * self.rank() + l], rhs.at(i, j, k, l)))));
                        }
                    }
                    summand
                });
                record_time!("DoubleRNSRing::gadget_product_lkss::ffts", || {
                    for k in 0..output_moduli_count {
                        self.ring_decompositions()[self.rns_base().len() - output_moduli_count + k].mult_basis_to_small_basis(&mut summand[(k * self.rank())..((k + 1) * self.rank())]);
                    }
                });
                record_time!("DoubleRNSRing::gadget_product_lkss::lifting", || {
                    rhs.conversions[j].apply(
                        Submatrix::from_1d(&summand, shortened_rns_base.len(), self.rank()),
                        self.as_matrix_wrt_small_basis_mut(&mut result).restrict_rows(digits.at(j).clone())
                    );
                });
            }
            return result;
        })
    }

    ///
    /// Whether computing this gadget product naturally returns the result in [`ElRepr::NTT`] 
    /// or [`ElRepr::Coeff`] representation. If you can use either, calling the appropriate function
    /// can save some unnecessary conversion.
    /// 
    pub fn preferred_output_repr(&self, lhs: &GadgetProductLhsOperand<NumberRing, A>, rhs: &GadgetProductRhsOperand<NumberRing, A>) -> ElRepr {
        match (lhs, rhs) {
            (GadgetProductLhsOperand::LKSSStyle(_), GadgetProductRhsOperand::LKSSStyle(_)) => ElRepr::Coeff,
            (GadgetProductLhsOperand::Naive(_), GadgetProductRhsOperand::Naive(_)) => ElRepr::NTT,
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
            (GadgetProductLhsOperand::Naive(lhs), GadgetProductRhsOperand::Naive(rhs)) => self.undo_fft(self.gadget_product_naive(lhs, rhs)),
            _ => panic!("Illegal combination of GadgetProductOperands; Maybe they were created by different rings?")
        }
    }
}

#[test]
fn test_naive_gadget_decomposition() {
    let ring = DoubleRNSRingBase::new(Pow2CyclotomicDecomposableNumberRing::new(4), zn_rns::Zn::create_from_primes(vec![17, 97, 113], BigIntRing::RING));
    let rns_base = ring.base_ring();
    let from_congruence = |data: &[i32]| rns_base.from_congruence(data.iter().enumerate().map(|(i, c)| rns_base.at(i).int_hom().map(*c)));
    let hom_big = ring.base_ring().can_hom(&BigIntRing::RING).unwrap();
    let hom_i32 = ring.base_ring().can_hom(&StaticRing::<i32>::RING).unwrap();

    let mut rhs = NaiveGadgetProductRhsOperand::create_empty(ring.get_ring(), 2);
    rhs.set_rns_factor(0, ring.get_ring().non_fft_from(from_congruence(&[1, 1, 0])));
    rhs.set_rns_factor(1, ring.get_ring().non_fft_from(from_congruence(&[0, 0, 1])));

    let lhs = NaiveGadgetProductLhsOperand::new_from_element(ring.get_ring(), ring.get_ring().non_fft_from(hom_i32.map(1000)), 2);

    assert_el_eq!(ring, ring.inclusion().map(hom_i32.map(1000)), ring.get_ring().gadget_product_naive(&lhs, &rhs));
}

#[test]
fn test_lkss_gadget_decomposition() {
    let ring = DoubleRNSRingBase::new(Pow2CyclotomicDecomposableNumberRing::new(4), zn_rns::Zn::create_from_primes(vec![17, 97, 113], BigIntRing::RING));
    let rns_base = ring.base_ring();
    let from_congruence = |data: &[i32]| rns_base.from_congruence(data.iter().enumerate().map(|(i, c)| rns_base.at(i).int_hom().map(*c)));
    let hom_big = ring.base_ring().can_hom(&BigIntRing::RING).unwrap();
    let hom_i32 = ring.base_ring().can_hom(&StaticRing::<i32>::RING).unwrap();

    let mut rhs = LKSSGadgetProductRhsOperand::create_empty(ring.get_ring(), 2, 2);
    rhs.set_rns_factor(0, ring.get_ring().non_fft_from(from_congruence(&[1, 1, 0])));
    rhs.set_rns_factor(1, ring.get_ring().non_fft_from(from_congruence(&[0, 0, 1])));

    let lhs = LKSSGadgetProductLhsOperand::new_from_element(ring.get_ring(), ring.get_ring().non_fft_from(hom_i32.map(1000)), 2, 2);

    assert_el_eq!(ring, ring.inclusion().map(hom_i32.map(1000)), ring.get_ring().do_fft(ring.get_ring().gadget_product_lkss(&lhs, &rhs)));
}