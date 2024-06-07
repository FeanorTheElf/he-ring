use feanor_math::algorithms::matmul::InnerProductComputation;
use feanor_math::integer::*;
use feanor_math::matrix::submatrix::*;
use feanor_math::mempool::*;
use feanor_math::homomorphism::*;
use feanor_math::vector::VectorView;
use feanor_math::rings::zn::{ZnRingStore, ZnRing};
use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::primitive_int::*;
use feanor_math::ring::*;
use feanor_math::ordered::OrderedRingStore;

use super::RNSOperation;

///
/// Stores values for an almost exact conversion between RNS bases.
/// A complete conversion refers to the function
/// ```text
/// Z/QZ -> Z/Q'Z, x -> [lift(x)]
/// ```
/// In our case, the output of the function is allowed to have an error of `{ -Q, 0, Q }`,
/// unless the shortest lift of the input is bounded by `Q/4`, in which case the result
/// is always correct.
/// 
/// # Implementation
/// 
/// Similar to [`super::approx_lift::AlmostExactBaseConversion`], but this
/// implementation makes some assumptions on the sizes of the moduli, which allows
/// to use a matrix multiplication for the performance-critical section.
/// 
pub struct AlmostExactMatrixBaseConversion<R, M_Int, M_Zn>
    where R: ZnRingStore + Clone,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<StaticRingBase<i128>>,
        M_Int: MemoryProvider<i64>,
        M_Zn: MemoryProvider<El<R>>
{
    from_summands: Vec<R>,
    to_summands: Vec<R>,
    /// the values `q/Q mod q` for each RNS factor q dividing Q (ordered as `from_summands`)
    q_over_Q: M_Zn::Object,
    /// shortest lifts of the values `Q/q mod q'` for each RNS factor q dividing Q (ordered as `from_summands_ordered`) and q' dividing Q'
    Q_over_q: M_Int::Object,
    /// the values `Q/q/2^drop_bits'` for each RNS factor q dividing Q (ordered as `from_summands_ordered`)
    Q_over_q_int: M_Int::Object,
    Q_dropped_bits: i64,
    /// `Q mod q'` for every `q'` dividing `Q'`
    Q_mod_q: M_Zn::Object,
    memory_provider_int: M_Int,
    _memory_provider_zn: M_Zn
}

const ZZbig: BigIntRing = BigIntRing::RING;
const ZZi64: StaticRing<i64> = StaticRing::<i64>::RING;
const ZZi128: StaticRing<i128> = StaticRing::<i128>::RING;

impl<R, M_Int, M_Zn> AlmostExactMatrixBaseConversion<R, M_Int, M_Zn> 
    where R: ZnRingStore + Clone,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<StaticRingBase<i128>>,
        M_Int: MemoryProvider<i64>,
        M_Zn: MemoryProvider<El<R>>
{
    ///
    /// Creates a new [`AlmostExactBaseConversion`] from `q` to `q'`. The moduli belonging to `q'`
    /// are expected to be sorted.
    /// 
    pub fn new(in_rings: Vec<R>, out_rings: Vec<R>, memory_provider_int: M_Int, memory_provider_zn: M_Zn) -> Self {
        let ZZ = in_rings.at(0).integer_ring();
        for i in 0..in_rings.len() {
            assert!(in_rings.at(i).integer_ring().get_ring() == ZZ.get_ring());
        }
        for i in 0..out_rings.len() {
            assert!(out_rings.at(i).integer_ring().get_ring() == ZZ.get_ring());
        }
        let Q = ZZbig.prod((0..in_rings.len()).map(|i| int_cast(ZZ.clone_el(in_rings.at(i).modulus()), ZZbig, ZZ)));

        let max = |l, r| if ZZbig.is_geq(&l, &r) { l } else { r };
        let max_computation_result = ZZbig.prod([
            in_rings.iter().map(|ring| int_cast(ZZ.clone_el(ring.modulus()), ZZbig, ZZ)).reduce(max).unwrap(),
            out_rings.iter().map(|ring| int_cast(ZZ.clone_el(ring.modulus()), ZZbig, ZZ)).reduce(max).unwrap(),
            ZZbig.int_hom().map(in_rings.len() as i32)
        ].into_iter());
        assert!(ZZbig.is_lt(&max_computation_result, &ZZbig.power_of_two(i128::BITS as usize - 1)), "AlmostExactMatrixBaseConversion requires that the unreduced result fits into an i128");
        
        // When computing the approximate lifted value, we can drop `k` bits where `k <= 1 + log(Q/(4 r max(q + 1)))` and `q | Q`
        let log2_r = ZZi64.abs_log2_ceil(&(in_rings.len() as i64)).unwrap() as i64;
        let log2_qmax = ZZi64.abs_log2_ceil(&(0..in_rings.len()).map(|i| int_cast(ZZ.clone_el(in_rings.at(i).modulus()), ZZi64, ZZ)).max().unwrap()).unwrap() as i64;
        let log2_Q = ZZbig.abs_log2_ceil(&Q).unwrap() as i64;
        let drop_bits = log2_Q - log2_r - log2_qmax - 5;
        let drop_bits = if drop_bits < 0 { 0 } else { drop_bits as usize };
        assert!((drop_bits as i64) < log2_Q);
        assert!(i64::BITS as i64 - 1 > log2_r + log2_Q - drop_bits as i64);

        let Q_over_q = memory_provider_int.get_new_init(in_rings.len() * out_rings.len(), |idx| {
            let i = idx / in_rings.len();
            let j = idx % in_rings.len();
            let ring = out_rings.at(i);
            int_cast(ring.smallest_lift(ring.coerce(&ZZbig, ZZbig.checked_div(&Q, &int_cast(ZZ.clone_el(in_rings.at(j).modulus()), ZZbig, ZZ)).unwrap())), ZZi64, ZZ)
        });
        let Q_over_q_int = memory_provider_int.get_new_init(in_rings.len(), |i| 
            int_cast(ZZbig.rounded_div(ZZbig.clone_el(&Q), &ZZbig.mul(int_cast(ZZ.clone_el(in_rings.at(i).modulus()), ZZbig, ZZ), ZZbig.power_of_two(drop_bits))), ZZi64, ZZbig)
        );
        let q_over_Q = memory_provider_zn.get_new_init(in_rings.len(), |i| 
            in_rings.at(i).invert(&in_rings.at(i).coerce(&ZZbig, ZZbig.checked_div(&Q, &int_cast(ZZ.clone_el(in_rings.at(i).modulus()), ZZbig, ZZ)).unwrap())).unwrap()
        );

        Self {
            Q_over_q: Q_over_q,
            Q_over_q_int: Q_over_q_int,
            q_over_Q: q_over_Q,
            Q_mod_q: memory_provider_zn.get_new_init(out_rings.len(), |i| out_rings.at(i).coerce(&ZZbig, ZZbig.clone_el(&Q))),
            Q_dropped_bits: int_cast(ZZbig.rounded_div(Q, &ZZbig.power_of_two(drop_bits)), ZZi64, ZZbig),
            memory_provider_int: memory_provider_int,
            _memory_provider_zn: memory_provider_zn,
            from_summands: in_rings,
            to_summands: out_rings
        }
    }
}

impl<R, M_Int, M_Zn> RNSOperation for AlmostExactMatrixBaseConversion<R, M_Int, M_Zn> 
    where R: ZnRingStore + Clone,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<StaticRingBase<i128>>,
        M_Int: MemoryProvider<i64>,
        M_Zn: MemoryProvider<El<R>>
{
    type Ring = R;

    type RingType = R::Type;

    fn input_rings<'a>(&'a self) -> &'a [R] {
        &self.from_summands
    }

    fn output_rings<'a>(&'a self) -> &'a [R] {
        &self.to_summands
    }

    ///
    /// Performs the (almost) exact RNS base conversion
    /// ```text
    ///     Z/QZ -> Z/Q'Z, x -> smallest_lift(x) + kQ mod Q''
    /// ```
    /// where `k in { -1, 0, 1 }`.
    /// 
    /// Furthermore, if the shortest lift of the input is bounded by `Q/4`,
    /// then the result is guaranteed to be exact.
    /// 
    fn apply<V1, V2>(&self, input: Submatrix<V1, El<Self::Ring>>, mut output: SubmatrixMut<V2, El<Self::Ring>>)
        where V1: AsPointerToSlice<El<Self::Ring>>,
            V2: AsPointerToSlice<El<Self::Ring>>
    {
        assert_eq!(input.row_count(), self.input_rings().len());
        assert_eq!(output.row_count(), self.output_rings().len());
        assert_eq!(input.col_count(), output.col_count());

        let in_len = input.row_count();
        let out_len = output.row_count();
        let col_count = input.col_count();

        let ZZ = self.output_rings().at(0).integer_ring();
        let int_to_homs = (0..self.output_rings().len()).map(|k| self.output_rings().at(k).can_hom(&ZZi128).unwrap()).collect::<Vec<_>>();

        let mut lifts = self.memory_provider_int.get_new_init(in_len * col_count, |_| 0);
        let mut lifts = SubmatrixMut::<AsFirstElement<_>, _>::new(&mut lifts, in_len, col_count);

        for i in 0..in_len {
            for j in 0..col_count {
                *lifts.at(i, j) = int_cast(
                    self.from_summands[i].smallest_lift(self.from_summands[i].mul_ref(input.at(i, j), self.q_over_Q.at(i))),
                    StaticRing::<i64>::RING,
                    ZZ
                );
            }
        }

        let mut output_unreduced = self.memory_provider_int.get_new_init(col_count * out_len * 2, |_| 0);
        let mut output_unreduced: &mut [i128] = bytemuck::cast_slice_mut(&mut output_unreduced);
        let mut output_unreduced = SubmatrixMut::<AsFirstElement<_>, _>::new(&mut output_unreduced, out_len, col_count);

        let Q_over_q = Submatrix::<AsFirstElement<_>, _>::new(&self.Q_over_q, out_len, in_len);

        for i in 0..out_len {
            for j in 0..col_count {
                for k in 0..in_len {
                    *output_unreduced.at(i, j) += *Q_over_q.at(i, k) as i128 * *lifts.at(k, j) as i128;
                }
            }
        }
        
        for j in 0..col_count {
            let correction = ZZi64.rounded_div(<_ as InnerProductComputation>::inner_product(ZZi64.get_ring(), 
                (0..input.row_count()).map(|i| (*lifts.at(i, j), *self.Q_over_q_int.at(i)))
            ), &self.Q_dropped_bits);

            for i in 0..out_len {
                *output.at(i, j) = self.to_summands[i].sub(
                    int_to_homs.at(i).map_ref(output_unreduced.at(i, j)), 
                    self.to_summands[i].mul_ref_snd(int_to_homs[i].map(correction as i128), &self.Q_mod_q[i])
                );
            }
        }
    }
}

#[cfg(test)]
use feanor_math::rings::zn::zn_64::*;
#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use test::Bencher;
#[cfg(test)]
use feanor_math::algorithms::miller_rabin::is_prime;
#[cfg(test)]
use feanor_math::rings::finite::FiniteRingStore;
#[cfg(test)]
use feanor_math::default_memory_provider;
#[cfg(test)]
use caching::CachingMemoryProvider;

#[test]
fn test_rns_base_conversion() {
    let from = vec![Zn::new(17), Zn::new(97)];
    let to = vec![Zn::new(17), Zn::new(97), Zn::new(113), Zn::new(257)];

    let table = AlmostExactMatrixBaseConversion::new(from.clone(), to.clone(), default_memory_provider!(), default_memory_provider!());

    // within this area, we guarantee that no error occurs
    for k in -(17 * 97 / 4)..=(17 * 97 / 4) {
        let input = from.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();
        let expected = to.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();
        let mut actual = to.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();

        table.apply(
            Submatrix::<AsFirstElement<_>, _>::new(&input, 2, 1), 
            SubmatrixMut::<AsFirstElement<_>, _>::new(&mut actual, 4, 1)
        );

        for j in 0..to.len() {
            assert_el_eq!(to.at(j), expected.at(j), actual.at(j));
        }
    }

    for k in (-17 * 97 / 2)..=(17 * 97 / 2) {
        let input = from.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();
        let expected = to.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();
        let mut actual = to.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();

        table.apply(
            Submatrix::<AsFirstElement<_>, _>::new(&input, 2, 1), 
            SubmatrixMut::<AsFirstElement<_>, _>::new(&mut actual, 4, 1)
        );

        for j in 0..to.len() {
            assert!(
                to.at(j).eq_el(expected.at(j), actual.at(j)) ||
                to.at(j).eq_el(&to.at(j).add_ref_fst(expected.at(j), to.at(j).int_hom().map(17 * 97)), actual.at(j)) ||
                to.at(j).eq_el(&to.at(j).sub_ref_fst(expected.at(j), to.at(j).int_hom().map(17 * 97)), actual.at(j))
            );
        }
    }
}

#[test]
fn test_rns_base_conversion_small() {
    let from = vec![Zn::new(3), Zn::new(97)];
    let to = vec![Zn::new(17)];
    let table = AlmostExactMatrixBaseConversion::new(from.clone(), to.clone(), default_memory_provider!(), default_memory_provider!());
    
    for k in -(97 * 3 / 2)..(97 * 3 / 2) {
        let mut actual = to.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();
        table.apply(
            Submatrix::<AsFirstElement<_>, _>::new(&[from[0].int_hom().map(k), from[1].int_hom().map(k)], 2, 1), 
            SubmatrixMut::<AsFirstElement<_>, _>::new(&mut actual, 1, 1)
        );

        assert!(
            to[0].eq_el(&to[0].int_hom().map(k), actual.at(0)) ||
            to[0].eq_el(&to[0].int_hom().map(k + 97 * 3), actual.at(0)) ||
            to[0].eq_el(&to[0].int_hom().map(k - 97 * 3), actual.at(0))
        );
    }
}

#[test]
fn test_rns_base_conversion_not_coprime() {
    let from = vec![Zn::new(17), Zn::new(97), Zn::new(113)];
    let to = vec![Zn::new(17), Zn::new(97), Zn::new(113), Zn::new(257)];
    let table = AlmostExactMatrixBaseConversion::new(from.clone(), to.clone(), default_memory_provider!(), default_memory_provider!());

    for k in -(17 * 97 * 113 / 4)..=(17 * 97 * 113 / 4) {
        let x = from.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();
        let y = to.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();
        let mut actual = to.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();

        table.apply(
            Submatrix::<AsFirstElement<_>, _>::new(&x, 3, 1), 
            SubmatrixMut::<AsFirstElement<_>, _>::new(&mut actual, 4, 1)
        );
        
        for i in 0..y.len() {
            assert!(to[i].eq_el(&y[i], actual.at(i)));
        }
    }
}

#[test]
fn test_rns_base_conversion_not_coprime_permuted() {
    let from = vec![Zn::new(113), Zn::new(17), Zn::new(97)];
    let to = vec![Zn::new(17), Zn::new(97), Zn::new(113), Zn::new(257)];
    let table = AlmostExactMatrixBaseConversion::new(from.clone(), to.clone(), default_memory_provider!(), default_memory_provider!());

    for k in -(17 * 97 * 113 / 4)..=(17 * 97 * 113 / 4) {
        let x = from.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();
        let y = to.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();
        let mut actual = to.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();

        table.apply(
            Submatrix::<AsFirstElement<_>, _>::new(&x, 3, 1), 
            SubmatrixMut::<AsFirstElement<_>, _>::new(&mut actual, 4, 1)
        );
        
        for i in 0..y.len() {
            assert!(to[i].eq_el(&y[i], actual.at(i)));
        }
    }
}

#[test]
fn test_rns_base_conversion_coprime() {
    let from = vec![Zn::new(17), Zn::new(97), Zn::new(113)];
    let to = vec![Zn::new(19), Zn::new(23), Zn::new(257)];
    let table = AlmostExactMatrixBaseConversion::new(from.clone(), to.clone(), default_memory_provider!(), default_memory_provider!());

    for k in -(17 * 97 * 113 / 4)..=(17 * 97 * 113 / 4) {
        let x = from.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();
        let y = to.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();
        let mut actual = to.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();

        table.apply(
            Submatrix::<AsFirstElement<_>, _>::new(&x, 3, 1), 
            SubmatrixMut::<AsFirstElement<_>, _>::new(&mut actual, 3, 1)
        );
        
        for i in 0..y.len() {
            assert!(to[i].eq_el(&y[i], actual.at(i)));
        }
    }
}

#[bench]
fn bench_rns_base_conversion(bencher: &mut Bencher) {
    let in_moduli_count = 20;
    let out_moduli_count = 40;
    let cols = 1000;
    let mut primes = ((1 << 30)..).map(|k| (1 << 10) * k + 1).filter(|p| is_prime(&StaticRing::<i64>::RING, p, 10)).map(|p| Zn::new(p as u64));
    let in_moduli = primes.by_ref().take(in_moduli_count).collect::<Vec<_>>();
    let out_moduli = primes.take(out_moduli_count).collect::<Vec<_>>();
    let conv = AlmostExactMatrixBaseConversion::new(in_moduli.clone(), out_moduli.clone(), CachingMemoryProvider::new(2), default_memory_provider!());
    
    let mut rng = oorandom::Rand64::new(1);
    let mut in_data = (0..(in_moduli_count * cols)).map(|idx| in_moduli[idx / cols].zero()).collect::<Vec<_>>();
    let mut in_matrix = SubmatrixMut::<AsFirstElement<_>, _>::new(&mut in_data, in_moduli_count, cols);
    let mut out_data = (0..(out_moduli_count * cols)).map(|idx| out_moduli[idx / cols].zero()).collect::<Vec<_>>();
    let mut out_matrix = SubmatrixMut::<AsFirstElement<_>, _>::new(&mut out_data, out_moduli_count, cols);

    bencher.iter(|| {
        for i in 0..in_moduli_count {
            for j in 0..cols {
                *in_matrix.at(i, j) = in_moduli[i].random_element(|| rng.rand_u64());
            }
        }
        conv.apply(in_matrix.as_const(), out_matrix.reborrow());
        for i in 0..out_moduli_count {
            for j in 0..cols {
                std::hint::black_box(out_matrix.at(i, j));
            }
        }
    });
}