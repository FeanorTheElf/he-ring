use feanor_math::algorithms::matmul::strassen::strassen_mem_size;
use feanor_math::integer::*;
use feanor_math::matrix::*;
use feanor_math::homomorphism::*;
use feanor_math::seq::*;
use feanor_math::rings::zn::*;
use feanor_math::rings::zn::zn_64::*;
use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::primitive_int::*;
use feanor_math::ring::*;
use feanor_math::ordered::OrderedRingStore;
use tracing::instrument;

use std::alloc::Allocator;
use std::alloc::Global;

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
pub struct AlmostExactMatrixBaseConversion<A = Global>
    where A: Allocator + Clone
{
    from_summands: Vec<Zn>,
    to_summands: Vec<Zn>,
    /// the values `q/Q mod q` for each RNS factor q dividing Q (ordered as `from_summands`)
    q_over_Q: Vec<ZnEl>,
    /// shortest lifts of the values `Q/q mod q'` for each RNS factor q dividing Q (ordered as `from_summands_ordered`) and q' dividing Q';
    /// finally, the last row are the values `gamma/q'` for each RNS factor q dividing Q (ordered as `from_summands_ordered`)
    Q_over_q_mod_and_downscaled: OwnedMatrix<i128>,
    gamma: i128,
    /// `Q mod q'` for every `q'` dividing `Q'`
    Q_mod_q: Vec<ZnEl>,
    allocator: A
}

// we currently use `any_lift()`; I haven't yet documented it anywhere, but in fact the largest output of `zn_64::Zn::any_lift()` is currently `6 * modulus()`
const ZN_ANY_LIFT_FACTOR: i64 = 6;

const BLOCK_SIZE_LOG2: usize = 4;

fn pad_to_block(len: usize) -> usize {
    ((len - 1) / (1 << BLOCK_SIZE_LOG2) + 1) * (1 << BLOCK_SIZE_LOG2)
}

const ZZbig: BigIntRing = BigIntRing::RING;
const ZZi64: StaticRing<i64> = StaticRing::<i64>::RING;
const ZZi128: StaticRing<i128> = StaticRing::<i128>::RING;

impl<A> AlmostExactMatrixBaseConversion<A> 
    where A: Allocator + Clone
{
    ///
    /// Creates a new [`AlmostExactBaseConversion`] from `q` to `q'`. The moduli belonging to `q'`
    /// are expected to be sorted.
    /// 
    #[instrument(skip_all)]
    pub fn new_with(in_rings: Vec<Zn>, out_rings: Vec<Zn>, allocator: A) -> Self {
        
        let Q = ZZbig.prod((0..in_rings.len()).map(|i| int_cast(*in_rings.at(i).modulus(), ZZbig, ZZi64)));

        let max = |l, r| if ZZbig.is_geq(&l, &r) { l } else { r };
        let max_computation_result = ZZbig.prod([
            in_rings.iter().map(|ring| int_cast(*ring.modulus() * ZN_ANY_LIFT_FACTOR, ZZbig, ZZi64)).reduce(max).unwrap(),
            out_rings.iter().map(|ring| int_cast(*ring.modulus(), ZZbig, ZZi64)).reduce(max).unwrap(),
            ZZbig.int_hom().map(in_rings.len() as i32)
        ].into_iter());
        assert!(ZZbig.is_lt(&max_computation_result, &ZZbig.power_of_two(i128::BITS as usize - 1)), "temporarily unreduced modular lift sum will overflow");

        // When computing the approximate lifted value, we can work with `gamma` in place of `Q`, where `gamma >= 4 r max(q)` (`q` runs through the input factors)
        let log2_r = ZZi64.abs_log2_ceil(&(in_rings.len() as i64)).unwrap();
        let log2_qmax = ZZi64.abs_log2_ceil(&(0..in_rings.len()).map(|i| *in_rings.at(i).modulus()).max().unwrap()).unwrap();
        let log2_any_lift_factor = ZZi64.abs_log2_ceil(&ZN_ANY_LIFT_FACTOR).unwrap();
        let gamma = ZZbig.power_of_two(log2_r + log2_qmax + log2_any_lift_factor + 2);
        // we compute a sum of `r` summands, each being a product of a lifted value (mod `q`, `q | Q`) and `gamma/q`; this must not overflow
        assert!(ZZbig.abs_log2_ceil(&gamma).unwrap() + log2_r + log2_any_lift_factor + 1 < ZZi128.get_ring().representable_bits().unwrap(), "correction computation will overflow");
        let gamma_log2 = ZZbig.abs_log2_ceil(&gamma).unwrap();
        assert!(gamma_log2 == ZZbig.abs_log2_floor(&gamma).unwrap());

        let Q_over_q = OwnedMatrix::from_fn_in(pad_to_block(out_rings.len() + 1), pad_to_block(in_rings.len()), |i, j| {
            if i < out_rings.len() && j < in_rings.len() {
                let ring = out_rings.at(i);
                ring.smallest_lift(ring.coerce(&ZZbig, ZZbig.checked_div(&Q, &int_cast(*in_rings.at(j).modulus(), ZZbig, ZZi64)).unwrap())) as i128
            } else if i == out_rings.len() && j < in_rings.len() {
                int_cast(ZZbig.rounded_div(ZZbig.clone_el(&gamma), &int_cast(*in_rings.at(j).modulus(), ZZbig, ZZi64)), ZZi128, ZZbig)
            } else {
                0
            }
        }, Global);
        let q_over_Q = (0..(in_rings.len())).map(|i| 
            in_rings.at(i).invert(&in_rings.at(i).coerce(&ZZbig, ZZbig.checked_div(&Q, &int_cast(*in_rings.at(i).modulus(), ZZbig, ZZi64)).unwrap())).unwrap()
        ).collect();

        Self {
            Q_over_q_mod_and_downscaled: Q_over_q,
            q_over_Q: q_over_Q,
            Q_mod_q: (0..out_rings.len()).map(|i| out_rings.at(i).coerce(&ZZbig, ZZbig.clone_el(&Q))).collect(),
            gamma: ZZi128.power_of_two(gamma_log2),
            allocator: allocator.clone(),
            from_summands: in_rings,
            to_summands: out_rings
        }
    }
}

impl<A> RNSOperation for AlmostExactMatrixBaseConversion<A> 
    where A: Allocator + Clone
{
    type Ring = Zn;

    type RingType = ZnBase;

    fn input_rings<'a>(&'a self) -> &'a [Zn] {
        &self.from_summands
    }

    fn output_rings<'a>(&'a self) -> &'a [Zn] {
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
    #[instrument(skip_all)]
    fn apply<V1, V2>(&self, input: Submatrix<V1, El<Self::Ring>>, mut output: SubmatrixMut<V2, El<Self::Ring>>)
        where V1: AsPointerToSlice<El<Self::Ring>>,
            V2: AsPointerToSlice<El<Self::Ring>>
    {
        {
            assert_eq!(input.row_count(), self.input_rings().len());
            assert_eq!(output.row_count(), self.output_rings().len());
            assert_eq!(input.col_count(), output.col_count());

            let in_len = input.row_count();
            let out_len = output.row_count();
            let col_count = input.col_count();

            let int_to_homs = (0..self.output_rings().len()).map(|k| self.output_rings().at(k).can_hom(&ZZi128).unwrap()).collect::<Vec<_>>();

            let mut lifts = OwnedMatrix::from_fn_in(pad_to_block(in_len), pad_to_block(col_count), |_, _| 0, self.allocator.clone());
            let mut lifts = lifts.data_mut();

            for i in 0..in_len {
                for j in 0..col_count {
                    // using `any_lift()` here is slightly dangerous, as I haven't documented anywhere that `zn_64::Zn::any_lift()` returns values `<= 6 * modulus()`, but
                    // it currently does, so this is currently fine
                    *lifts.at_mut(i, j) = self.from_summands[i].any_lift(self.from_summands[i].mul_ref(input.at(i, j), self.q_over_Q.at(i))) as i128;
                    debug_assert!(*lifts.at(i, 0) >= 0 && *lifts.at(i, 0) <= ZN_ANY_LIFT_FACTOR as i128 * *self.from_summands[i].modulus() as i128);
                }
            }

            let mut output_unreduced = OwnedMatrix::from_fn_in(pad_to_block(out_len + 1), pad_to_block(col_count), |_, _| 0, self.allocator.clone());
            let mut output_unreduced = output_unreduced.data_mut();

            // actually using Strassen's algorithm here doesn't make much of a difference, it is basically as fast as without for normal
            // parameters; however, this way we can claim superior asymptotic performance :)
            const STRASSEN_THRESHOLD_LOG2: usize = 3;
            let mem_size = strassen_mem_size(pad_to_block(in_len) > (1 << BLOCK_SIZE_LOG2), BLOCK_SIZE_LOG2, STRASSEN_THRESHOLD_LOG2);
            let mut memory = Vec::with_capacity_in(mem_size, self.allocator.clone());
            memory.resize(mem_size, 0);

            {
                for i in 0..(pad_to_block(out_len + 1) / (1 << BLOCK_SIZE_LOG2)) {
                    for k in 0..(pad_to_block(in_len) / (1 << BLOCK_SIZE_LOG2)) {
                        for j in 0..(pad_to_block(col_count) / (1 << BLOCK_SIZE_LOG2)) {
                            let rows = (i << BLOCK_SIZE_LOG2)..((i + 1) << BLOCK_SIZE_LOG2);
                            let cols = (j << BLOCK_SIZE_LOG2)..((j + 1) << BLOCK_SIZE_LOG2);
                            let ks = (k << BLOCK_SIZE_LOG2)..((k + 1) << BLOCK_SIZE_LOG2);
                            if k == 0 {
                                feanor_math::algorithms::matmul::strassen::dispatch_strassen_impl::<_, _, _, _, false, false, false, false>(
                                    BLOCK_SIZE_LOG2, 
                                    STRASSEN_THRESHOLD_LOG2, 
                                    TransposableSubmatrix::from(self.Q_over_q_mod_and_downscaled.data().submatrix(rows.clone(), ks.clone())), 
                                    TransposableSubmatrix::from(lifts.as_const().submatrix(ks, cols.clone())), 
                                    TransposableSubmatrixMut::from(output_unreduced.reborrow().submatrix(rows, cols)), 
                                    StaticRing::<i128>::RING, 
                                    &mut memory
                                );
                            } else {   
                                feanor_math::algorithms::matmul::strassen::dispatch_strassen_impl::<_, _, _, _, true, false, false, false>(
                                    BLOCK_SIZE_LOG2, 
                                    STRASSEN_THRESHOLD_LOG2, 
                                    TransposableSubmatrix::from(self.Q_over_q_mod_and_downscaled.data().submatrix(rows.clone(), ks.clone())), 
                                    TransposableSubmatrix::from(lifts.as_const().submatrix(ks, cols.clone())), 
                                    TransposableSubmatrixMut::from(output_unreduced.reborrow().submatrix(rows, cols)), 
                                    StaticRing::<i128>::RING, 
                                    &mut memory
                                );
                            }
                        }
                    }
                }
            };

            for j in 0..col_count {
                let mut correction = *output_unreduced.at(out_len, j);
                correction = ZZi128.rounded_div(correction, &self.gamma);

                for i in 0..out_len {
                    *output.at_mut(i, j) = self.to_summands[i].sub(
                        int_to_homs.at(i).map_ref(output_unreduced.at(i, j)), 
                        self.to_summands[i].mul_ref_snd(int_to_homs[i].map(correction), &self.Q_mod_q[i])
                    );
                }
            }
        }
    }
}

#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use test::Bencher;
#[cfg(test)]
use feanor_math::algorithms::miller_rabin::is_prime;
#[cfg(test)]
use feanor_math::rings::finite::FiniteRingStore;

#[test]
fn test_rns_base_conversion() {
    let from = vec![Zn::new(17), Zn::new(97)];
    let to = vec![Zn::new(17), Zn::new(97), Zn::new(113), Zn::new(257)];

    let table = AlmostExactMatrixBaseConversion::new_with(from.clone(), to.clone(), Global);

    // within this area, we guarantee that no error occurs
    for k in -(17 * 97 / 4)..=(17 * 97 / 4) {
        let input = from.iter().map(|Zn| Zn.int_hom().map(k)).collect::<Vec<_>>();
        let expected = to.iter().map(|Zn| Zn.int_hom().map(k)).collect::<Vec<_>>();
        let mut actual = to.iter().map(|Zn| Zn.int_hom().map(k)).collect::<Vec<_>>();

        table.apply(
            Submatrix::from_1d(&input, 2, 1), 
            SubmatrixMut::from_1d(&mut actual, 4, 1)
        );

        for j in 0..to.len() {
            assert_el_eq!(to.at(j), expected.at(j), actual.at(j));
        }
    }

    for k in (-17 * 97 / 2)..=(17 * 97 / 2) {
        let input = from.iter().map(|Zn| Zn.int_hom().map(k)).collect::<Vec<_>>();
        let expected = to.iter().map(|Zn| Zn.int_hom().map(k)).collect::<Vec<_>>();
        let mut actual = to.iter().map(|Zn| Zn.int_hom().map(k)).collect::<Vec<_>>();

        table.apply(
            Submatrix::from_1d(&input, 2, 1), 
            SubmatrixMut::from_1d(&mut actual, 4, 1)
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
    let table = AlmostExactMatrixBaseConversion::new_with(from.clone(), to.clone(), Global);
    
    for k in -(97 * 3 / 2)..(97 * 3 / 2) {
        let mut actual = to.iter().map(|Zn| Zn.int_hom().map(k)).collect::<Vec<_>>();
        table.apply(
            Submatrix::from_1d(&[from[0].int_hom().map(k), from[1].int_hom().map(k)], 2, 1), 
            SubmatrixMut::from_1d(&mut actual, 1, 1)
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
    let table = AlmostExactMatrixBaseConversion::new_with(from.clone(), to.clone(), Global);

    for k in -(17 * 97 * 113 / 4)..=(17 * 97 * 113 / 4) {
        let x = from.iter().map(|Zn| Zn.int_hom().map(k)).collect::<Vec<_>>();
        let y = to.iter().map(|Zn| Zn.int_hom().map(k)).collect::<Vec<_>>();
        let mut actual = to.iter().map(|Zn| Zn.int_hom().map(k)).collect::<Vec<_>>();

        table.apply(
            Submatrix::from_1d(&x, 3, 1), 
            SubmatrixMut::from_1d(&mut actual, 4, 1)
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
    let table = AlmostExactMatrixBaseConversion::new_with(from.clone(), to.clone(), Global);

    for k in -(17 * 97 * 113 / 4)..=(17 * 97 * 113 / 4) {
        let x = from.iter().map(|Zn| Zn.int_hom().map(k)).collect::<Vec<_>>();
        let y = to.iter().map(|Zn| Zn.int_hom().map(k)).collect::<Vec<_>>();
        let mut actual = to.iter().map(|Zn| Zn.int_hom().map(k)).collect::<Vec<_>>();

        table.apply(
            Submatrix::from_1d(&x, 3, 1), 
            SubmatrixMut::from_1d(&mut actual, 4, 1)
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
    let table = AlmostExactMatrixBaseConversion::new_with(from.clone(), to.clone(), Global);

    for k in -(17 * 97 * 113 / 4)..=(17 * 97 * 113 / 4) {
        let x = from.iter().map(|Zn| Zn.int_hom().map(k)).collect::<Vec<_>>();
        let y = to.iter().map(|Zn| Zn.int_hom().map(k)).collect::<Vec<_>>();
        let mut actual = to.iter().map(|Zn| Zn.int_hom().map(k)).collect::<Vec<_>>();

        table.apply(
            Submatrix::from_1d(&x, 3, 1), 
            SubmatrixMut::from_1d(&mut actual, 3, 1)
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
    let conv = AlmostExactMatrixBaseConversion::new_with(in_moduli.clone(), out_moduli.clone(), Global);
    
    let mut rng = oorandom::Rand64::new(1);
    let mut in_data = (0..(in_moduli_count * cols)).map(|idx| in_moduli[idx / cols].zero()).collect::<Vec<_>>();
    let mut in_matrix = SubmatrixMut::from_1d(&mut in_data, in_moduli_count, cols);
    let mut out_data = (0..(out_moduli_count * cols)).map(|idx| out_moduli[idx / cols].zero()).collect::<Vec<_>>();
    let mut out_matrix = SubmatrixMut::from_1d(&mut out_data, out_moduli_count, cols);

    bencher.iter(|| {
        for i in 0..in_moduli_count {
            for j in 0..cols {
                *in_matrix.at_mut(i, j) = in_moduli[i].random_element(|| rng.rand_u64());
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

#[test]
fn test_base_conversion_large() {
    let primes: [i64; 34] = [
        72057594040066049,
        288230376150870017,
        288230376150876161,
        288230376150878209,
        288230376150890497,
        288230376150945793,
        288230376150956033,
        288230376151062529,
        288230376151123969,
        288230376151130113,
        288230376151191553,
        288230376151388161,
        288230376151422977,
        288230376151529473,
        288230376151545857,
        288230376151554049,
        288230376151601153,
        288230376151625729,
        288230376151683073,
        288230376151748609,
        288230376151760897,
        288230376151779329,
        288230376151812097,
        288230376151902209,
        288230376151951361,
        288230376151994369,
        288230376152027137,
        288230376152061953,
        288230376152137729,
        288230376152154113,
        288230376152156161,
        288230376152205313,
        288230376152227841,
        288230376152340481,
    ];
    let in_len = 17;
    let from = &primes[..in_len];
    let from_prod = ZZbig.prod(from.iter().map(|p| int_cast(*p, ZZbig, StaticRing::<i64>::RING)));
    let to = &primes[in_len..];
    let number = ZZbig.get_ring().parse("156545561910861509258548850310120795193837265771491906959215072510998373539323526014165281634346450795208120921520265422129013635769405993324585707811035953253906720513250161495607960734366886366296007741500531044904559075687514262946086011957808717474666493477109586105297965072817051127737667010", 10).unwrap();
    assert!(ZZbig.is_lt(&number, &from_prod));
    
    let from = from.iter().map(|p| Zn::new(*p as u64)).collect::<Vec<_>>();
    let to = to.iter().map(|p| Zn::new(*p as u64)).collect::<Vec<_>>();
    let conversion = AlmostExactMatrixBaseConversion::new_with(from, to, Global);

    let input = (0..in_len).map(|i| conversion.input_rings().at(i).coerce(&ZZbig, ZZbig.clone_el(&number))).collect::<Vec<_>>();
    let expected = (0..(primes.len() - in_len)).map(|i| conversion.output_rings().at(i).coerce(&ZZbig, ZZbig.clone_el(&number))).collect::<Vec<_>>();
    let mut output = (0..(primes.len() - in_len)).map(|i| conversion.output_rings().at(i).zero()).collect::<Vec<_>>();
    conversion.apply(Submatrix::from_1d(&input, in_len, 1), SubmatrixMut::from_1d(&mut output, primes.len() - in_len, 1));

    assert!(
        expected.iter().zip(output.iter()).enumerate().all(|(i, (e, a))| conversion.output_rings().at(i).eq_el(e, a)) ||
        expected.iter().zip(output.iter()).enumerate().all(|(i, (e, a))| conversion.output_rings().at(i).eq_el(e, &conversion.output_rings().at(i).add_ref_fst(a, conversion.output_rings().at(i).coerce(&ZZbig, ZZbig.clone_el(&from_prod))))) ||
        expected.iter().zip(output.iter()).enumerate().all(|(i, (e, a))| conversion.output_rings().at(i).eq_el(e, &conversion.output_rings().at(i).sub_ref_fst(a, conversion.output_rings().at(i).coerce(&ZZbig, ZZbig.clone_el(&from_prod)))))
    );
}