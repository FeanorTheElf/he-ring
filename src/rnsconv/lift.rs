use std::alloc::Allocator;
use std::alloc::Global;

use feanor_math::algorithms::matmul::ComputeInnerProduct;
use feanor_math::integer::*;
use feanor_math::matrix::*;
use feanor_math::homomorphism::*;
use feanor_math::seq::*;
use feanor_math::rings::zn::{ZnRingStore, ZnRing};
use feanor_math::rings::zn::zn_64::*;
use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::primitive_int::*;
use feanor_math::ring::*;
use feanor_math::ordered::OrderedRingStore;

use super::sort_unstable_permutation;
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
/// Implementation is changed to approximating the lifted value using lower precision integers,
/// which can be used to determine the overflow when computing
/// ```text
/// lift(x) = sum_q lift(x * q/Q mod q) * Q/q
/// ```
/// modulo some `q'`.
/// 
pub struct AlmostExactBaseConversion<A = Global>
    where A: Allocator + Clone
{
    from_summands: Vec<Zn>,
    from_summands_ordered: Vec<Zn>,
    to_summands: Vec<Zn>,
    from_summands_permutation: Vec<usize>,
    /// the values `q/Q mod q` for each RNS factor q dividing Q (ordered as `from_summands`)
    q_over_Q: Vec<ZnEl>,
    /// the values `Q/q mod q'` for each RNS factor q dividing Q (ordered as `from_summands_ordered`) and q' dividing Q'
    Q_over_q: Vec<ZnEl>,
    /// the values `Q/q/2^drop_bits'` for each RNS factor q dividing Q (ordered as `from_summands_ordered`)
    Q_over_q_int: Vec<i128>,
    Q_downscaled: i128,
    /// `Q mod q'` for every `q'` dividing `Q'`
    Q_mod_q: Vec<ZnEl>,
    allocator: A
}

const ZZbig: BigIntRing = BigIntRing::RING;
const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;
const ZZi128: StaticRing<i128> = StaticRing::<i128>::RING;

impl<A> AlmostExactBaseConversion<A> 
    where A: Allocator + Clone
{
    ///
    /// Creates a new [`AlmostExactBaseConversion`] from `q` to `q'`. The moduli belonging to `q'`
    /// are expected to be sorted.
    /// 
    pub fn new_with(in_rings: Vec<Zn>, out_rings: Vec<Zn>, allocator: A) -> Self {
        for i in 0..in_rings.len() {
            assert!(in_rings.at(i).integer_ring().get_ring() == ZZ.get_ring());
        }
        for i in 0..out_rings.len() {
            assert!(out_rings.at(i).integer_ring().get_ring() == ZZ.get_ring());
        }
        for i in 1..out_rings.len() {
            assert!(ZZ.is_gt(out_rings.at(i).modulus(), out_rings.at(i - 1).modulus()), "Output RNS base must be sorted by increasing moduli");
        }
        let (in_rings_ordered, in_rings_permutation) = sort_unstable_permutation(in_rings.clone(), |ring_l, ring_r| ZZ.cmp(ring_l.modulus(), ring_r.modulus()));
        let Q = ZZbig.prod((0..in_rings.len()).map(|i| int_cast(*in_rings.at(i).modulus(), ZZbig, ZZ)));
        
        // When computing the approximate lifted value, we can drop `k` bits where `k <= 1 + log(Q/(4 r max(q + 1)))` and `q | Q`
        let log2_r = ZZ.abs_log2_ceil(&(in_rings.len() as i64)).unwrap() as i64;
        let log2_qmax = ZZ.abs_log2_ceil(&(0..in_rings.len()).map(|i| *in_rings.at(i).modulus()).max().unwrap()).unwrap() as i64;
        let log2_Q = ZZbig.abs_log2_ceil(&Q).unwrap() as i64;
        let drop_bits = log2_Q - log2_r - log2_qmax - 5;
        let drop_bits = if drop_bits < 0 { 0 } else { drop_bits as usize };
        assert!((drop_bits as i64) < log2_Q);
        assert!(i128::BITS as i64 - 1 > log2_r + log2_Q - drop_bits as i64);

        Self {
            Q_over_q: (0..(in_rings_ordered.len() * out_rings.len())).map(|idx| 
                out_rings.at(idx / in_rings_ordered.len()).coerce(&ZZbig, ZZbig.checked_div(&Q, &int_cast(*in_rings_ordered.at(idx % in_rings_ordered.len()).modulus(), ZZbig, ZZ)).unwrap())
            ).collect(),
            Q_over_q_int: (0..in_rings_ordered.len()).map(|i| 
                int_cast(ZZbig.rounded_div(ZZbig.clone_el(&Q), &ZZbig.mul(int_cast(*in_rings_ordered.at(i).modulus(), ZZbig, ZZ), ZZbig.power_of_two(drop_bits))), ZZi128, ZZbig)
            ).collect(),
            q_over_Q: (0..in_rings.len()).map(|i| 
                in_rings.at(i).invert(&in_rings.at(i).coerce(&ZZbig, ZZbig.checked_div(&Q, &int_cast(*in_rings.at(i).modulus(), ZZbig, ZZ)).unwrap())).unwrap()
            ).collect(),
            Q_mod_q: (0..out_rings.len()).map(|i| 
                out_rings.at(i).coerce(&ZZbig, ZZbig.clone_el(&Q))
            ).collect(),
            Q_downscaled: int_cast(ZZbig.rounded_div(Q, &ZZbig.power_of_two(drop_bits)), ZZi128, ZZbig),
            allocator: allocator,
            from_summands: in_rings,
            from_summands_ordered: in_rings_ordered,
            from_summands_permutation: in_rings_permutation,
            to_summands: out_rings
        }
    }
}

impl<A> RNSOperation for AlmostExactBaseConversion<A> 
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
    fn apply<V1, V2>(&self, input: Submatrix<V1, El<Self::Ring>>, mut output: SubmatrixMut<V2, El<Self::Ring>>)
        where V1: AsPointerToSlice<El<Self::Ring>>,
            V2: AsPointerToSlice<El<Self::Ring>>
    {
        assert_eq!(input.row_count(), self.input_rings().len());
        assert_eq!(output.row_count(), self.output_rings().len());
        assert_eq!(input.col_count(), output.col_count());

        let in_len = input.row_count();
        let col_count = input.col_count();

        let int_to_homs = (0..self.output_rings().len()).map(|k| self.output_rings().at(k).can_hom(&ZZi128).unwrap()).collect::<Vec<_>>();

        let mut lifts = Vec::with_capacity_in(col_count * in_len, self.allocator.clone());
        lifts.extend((0..(in_len * col_count)).map(|_| 0));
        for i in 0..in_len {
            for j in 0..col_count {
                lifts[self.from_summands_permutation[i] + j * in_len] = self.from_summands[i].smallest_positive_lift(self.from_summands[i].mul_ref(input.at(i, j), self.q_over_Q.at(i)))
            }
        }

        for k in 0..output.row_count() {
            let no_red_steps = (0..in_len).take_while(|i| ZZ.is_gt(self.output_rings().at(k).modulus(), self.from_summands_ordered.at(*i).modulus())).count();
            if cfg!(feature = "force_rns_conv_reduction") {
                for j in 0..col_count {
                    *output.at_mut(k, j) = <_ as ComputeInnerProduct>::inner_product_ref_fst(self.output_rings().at(k).get_ring(), (0..in_len).map(|i| {
                        (self.Q_over_q.at(i + in_len * k), int_to_homs[k].map(lifts[i + j * in_len] as i128))
                    }));
                }
            } else if no_red_steps == in_len {
                for j in 0..col_count {
                    *output.at_mut(k, j) = <_ as ComputeInnerProduct>::inner_product_ref_fst(self.output_rings().at(k).get_ring(), (0..no_red_steps).map(|i| {
                        (self.Q_over_q.at(i + in_len * k), self.output_rings().at(k).get_ring().from_int_promise_reduced(ZZ.clone_el(&lifts[i + j * in_len])))
                    }));
                }
            } else {
                for j in 0..col_count {
                    *output.at_mut(k, j) = <_ as ComputeInnerProduct>::inner_product_ref_fst(self.output_rings().at(k).get_ring(), (0..no_red_steps).map(|i| {
                        (self.Q_over_q.at(i + in_len * k), self.output_rings().at(k).get_ring().from_int_promise_reduced(ZZ.clone_el(&lifts[i + j * in_len])))
                    }).chain((no_red_steps..in_len).map(|i| {
                        (self.Q_over_q.at(i + in_len * k), int_to_homs[k].map(lifts[i + j * in_len] as i128))
                    })));
                }
            }
        }

        for j in 0..col_count {
            let correction = ZZi128.rounded_div(<_ as ComputeInnerProduct>::inner_product(ZZi128.get_ring(), 
                (0..input.row_count()).map(|i| (lifts[i + input.row_count() * j] as i128, self.Q_over_q_int[i]))
            ), &self.Q_downscaled);
            for i in 0..output.row_count() {
                self.output_rings().at(i).sub_assign(output.at_mut(i, j), self.to_summands[i].mul_ref_snd(int_to_homs[i].map_ref(&correction), &self.Q_mod_q[i]));
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

    let table = AlmostExactBaseConversion::new_with(from.clone(), to.clone(), Global);

    // within this area, we guarantee that no error occurs
    for k in -(17 * 97 / 4)..=(17 * 97 / 4) {
        let input = from.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();
        let expected = to.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();
        let mut actual = to.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();

        table.apply(
            Submatrix::from_1d(&input, 2, 1), 
            SubmatrixMut::from_1d(&mut actual, 4, 1)
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
    let table = AlmostExactBaseConversion::new_with(from.clone(), to.clone(), Global);
    
    for k in -(97 * 3 / 2)..(97 * 3 / 2) {
        let mut actual = to.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();
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
    let table = AlmostExactBaseConversion::new_with(from.clone(), to.clone(), Global);

    for k in -(17 * 97 * 113 / 4)..=(17 * 97 * 113 / 4) {
        let x = from.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();
        let y = to.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();
        let mut actual = to.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();

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
    let table = AlmostExactBaseConversion::new_with(from.clone(), to.clone(), Global);

    for k in -(17 * 97 * 113 / 4)..=(17 * 97 * 113 / 4) {
        let x = from.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();
        let y = to.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();
        let mut actual = to.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();

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
    let table = AlmostExactBaseConversion::new_with(from.clone(), to.clone(), Global);

    for k in -(17 * 97 * 113 / 4)..=(17 * 97 * 113 / 4) {
        let x = from.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();
        let y = to.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();
        let mut actual = to.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();

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
    let conv = AlmostExactBaseConversion::new_with(in_moduli.clone(), out_moduli.clone(), Global);
    
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
    let conversion = AlmostExactBaseConversion::new_with(from, to, Global);

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