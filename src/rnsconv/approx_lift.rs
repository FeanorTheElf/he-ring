use feanor_math::integer::*;
use feanor_math::matrix::submatrix::*;
use feanor_math::mempool::*;
use feanor_math::homomorphism::*;
use feanor_math::vector::VectorView;
use feanor_math::rings::zn::{ZnRingStore, ZnRing};
use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::primitive_int::*;
use feanor_math::ring::*;

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
pub struct AlmostExactBaseConversion<R, R_base, M_Int, M_Zn>
    where R: ZnRingStore + Clone,
        R_base: ZnRingStore + Clone,
        R_base::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<R_base::Type>,
        M_Int: MemoryProvider<<<R::Type as ZnRing>::IntegerRingBase as RingBase>::Element>,
        M_Zn: MemoryProvider<El<R_base>>
{
    from_summands: Vec<R>,
    to_summands: Vec<R>,
    /// the values `q/Q mod q` for each RNS factor q dividing Q
    q_over_Q: M_Zn::Object,
    /// the values `Q/q mod q'` for each RNS factor q dividing Q and q' dividing Q'
    Q_over_q: M_Zn::Object,
    /// the values `Q/q/2^drop_bits'` for each RNS factor q dividing Q
    Q_over_q_int: M_Int::Object,
    Q_dropped_bits: <<R::Type as ZnRing>::IntegerRingBase as RingBase>::Element,
    /// `Q mod q'` for every `q'` dividing `Q'`
    Q_mod_q: M_Zn::Object,
    memory_provider_int: M_Int,
    _memory_provider_zn: M_Zn,
    from_homs: Vec<CanHom<R_base, R>>,
    to_homs: Vec<CanHom<R_base, R>>
}

const ZZbig: BigIntRing = BigIntRing::RING;

impl<R, M_Int, M_Zn> AlmostExactBaseConversion<R, R, M_Int, M_Zn> 
    where R: ZnRingStore + Clone,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<R::Type>,
        M_Int: MemoryProvider<<<R::Type as ZnRing>::IntegerRingBase as RingBase>::Element>,
        M_Zn: MemoryProvider<El<R>>
{
    pub fn new<V1, V2>(in_rings: V1, out_rings: V2, mem_provider_int: M_Int, mem_provider_zn: M_Zn) -> Self
        where V1: VectorView<R>,
            V2: VectorView<R>
    {
        Self::new_generic(
            in_rings.iter().cloned().collect(), 
            in_rings.iter().cloned().collect(), 
            out_rings.iter().cloned().collect(), 
            out_rings.iter().cloned().collect(), 
            mem_provider_int, 
            mem_provider_zn
        )
    }
}

impl<R, R_base, M_Int, M_Zn> AlmostExactBaseConversion<R, R_base, M_Int, M_Zn> 
    where R: ZnRingStore + Clone,
        R_base: ZnRingStore + Clone,
        R_base::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<R_base::Type>,
        M_Int: MemoryProvider<<<R::Type as ZnRing>::IntegerRingBase as RingBase>::Element>,
        M_Zn: MemoryProvider<El<R_base>>
{
    pub fn new_generic(in_rings: Vec<R>, in_rings_base: Vec<R_base>, out_rings: Vec<R>, out_rings_base: Vec<R_base>, memory_provider_int: M_Int, memory_provider_zn: M_Zn) -> Self {
        let ZZ = in_rings.at(0).integer_ring();
        for i in 0..in_rings.len() {
            assert!(in_rings.at(i).integer_ring().get_ring() == ZZ.get_ring());
        }
        for i in 0..out_rings.len() {
            assert!(out_rings.at(i).integer_ring().get_ring() == ZZ.get_ring());
        }
        let Q = ZZbig.prod((0..in_rings.len()).map(|i| int_cast(ZZ.clone_el(in_rings.at(i).modulus()), ZZbig, ZZ)));
        
        // When computing the approximate lifted value, we can drop `k` bits where `k <= 1 + log(Q/(4 r max(q + 1)))` and `q | Q`
        let log2_r = StaticRing::<i64>::RING.abs_log2_ceil(&(in_rings.len() as i64)).unwrap() as i64;
        let log2_qmax = StaticRing::<i64>::RING.abs_log2_ceil(&(0..in_rings.len()).map(|i| int_cast(ZZ.clone_el(in_rings.at(i).modulus()), StaticRing::<i64>::RING, ZZ)).max().unwrap()).unwrap() as i64;
        let log2_Q = ZZbig.abs_log2_ceil(&Q).unwrap() as i64;
        let drop_bits = log2_Q - log2_r - log2_qmax - 5;
        let drop_bits = if drop_bits < 0 { 0 } else { drop_bits as usize };
        assert!((drop_bits as i64) < log2_Q);
        assert!(ZZ.get_ring().representable_bits().is_none() || ZZ.get_ring().representable_bits().unwrap() as i64 > log2_r + log2_Q - drop_bits as i64);

        Self {
            Q_over_q: memory_provider_zn.get_new_init(in_rings.len() * out_rings.len(), |idx| 
                out_rings_base.at(idx / in_rings.len()).coerce(&ZZbig, ZZbig.checked_div(&Q, &int_cast(ZZ.clone_el(in_rings.at(idx % in_rings.len()).modulus()), ZZbig, ZZ)).unwrap())
            ),
            Q_over_q_int: memory_provider_int.get_new_init(in_rings.len(), |i| 
                int_cast(ZZbig.rounded_div(ZZbig.clone_el(&Q), &ZZbig.mul(int_cast(ZZ.clone_el(in_rings.at(i).modulus()), ZZbig, ZZ), ZZbig.power_of_two(drop_bits))), ZZ, ZZbig)
            ),
            q_over_Q: memory_provider_zn.get_new_init(in_rings.len(), |i| 
                in_rings_base.at(i).invert(&in_rings_base.at(i).coerce(&ZZbig, ZZbig.checked_div(&Q, &int_cast(ZZ.clone_el(in_rings.at(i).modulus()), ZZbig, ZZ)).unwrap())).unwrap()
            ),
            Q_mod_q: memory_provider_zn.get_new_init(out_rings.len(), |i| out_rings_base.at(i).coerce(&ZZbig, ZZbig.clone_el(&Q))),
            Q_dropped_bits: int_cast(ZZbig.rounded_div(Q, &ZZbig.power_of_two(drop_bits)), ZZ, ZZbig),
            memory_provider_int: memory_provider_int,
            _memory_provider_zn: memory_provider_zn,
            from_homs: (0..in_rings.len()).map(|i| in_rings.at(i).clone().into_can_hom(in_rings_base.at(i).clone()).ok().unwrap()).collect::<Vec<_>>(),
            to_homs: (0..out_rings.len()).map(|i| out_rings.at(i).clone().into_can_hom(out_rings_base.at(i).clone()).ok().unwrap()).collect::<Vec<_>>(),
            from_summands: in_rings,
            to_summands: out_rings
        }
    }
}

impl<R, R_base, M_Int, M_Zn> RNSOperation for AlmostExactBaseConversion<R, R_base, M_Int, M_Zn> 
    where R: ZnRingStore + Clone,
        R_base: ZnRingStore + Clone,
        R_base::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<R_base::Type>,
        M_Int: MemoryProvider<<<R::Type as ZnRing>::IntegerRingBase as RingBase>::Element>,
        M_Zn: MemoryProvider<El<R_base>>
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
    #[inline(never)]
    fn apply<V1, V2>(&self, input: Submatrix<V1, El<Self::Ring>>, mut output: SubmatrixMut<V2, El<Self::Ring>>)
        where V1: AsPointerToSlice<El<Self::Ring>>,
            V2: AsPointerToSlice<El<Self::Ring>>
    {
            assert_eq!(input.row_count(), self.input_rings().len());
            assert_eq!(output.row_count(), self.output_rings().len());
            assert_eq!(input.col_count(), output.col_count());

            let ZZ = self.output_rings().at(0).integer_ring();

            let int_to_homs = (0..self.output_rings().len()).map(|k| self.output_rings().at(k).can_hom(&ZZ).unwrap()).collect::<Vec<_>>();

            let col_count = input.col_count();

            let lifts = timed!("AlmostExactBaseConversion::apply::lift", || self.memory_provider_int.get_new_init(col_count * input.row_count(), |idx| {
                let i = idx % input.row_count();
                let j = idx / input.row_count();
                self.from_homs[i].codomain().smallest_lift(self.from_homs[i].mul_ref_map(input.at(i, j), self.q_over_Q.at(i)))
            }));

            timed!("AlmostExactBaseConversion::apply::product", || {
                for j in 0..col_count {
                    for k in 0..output.row_count() {
                        *output.at(k, j) = <_ as RingStore>::sum(self.output_rings().at(k), (0..input.row_count()).map(|i| {
                            let lifted_red = int_to_homs[k].map_ref(&lifts[i + j * input.row_count()]);
                            let factor = self.Q_over_q.at(i + self.input_rings().len() * k);
                            self.to_homs[k].mul_ref_snd_map(lifted_red, factor)
                        }));
                    }
                }
            });

            timed!("AlmostExactBaseConversion::apply::correction", || {
                for j in 0..col_count {
                    let correction = ZZ.rounded_div(<_ as RingStore>::sum(&ZZ, (0..input.row_count()).map(|i| ZZ.mul_ref(&lifts[i + input.row_count() * j], self.Q_over_q_int.at(i)))), &self.Q_dropped_bits);
                    for i in 0..output.row_count() {
                        self.output_rings().at(i).sub_assign(output.at(i, j), self.to_homs[i].mul_ref_snd_map(int_to_homs[i].map_ref(&correction), &self.Q_mod_q[i]));
                    }
                }
            });
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

    let table = AlmostExactBaseConversion::new(&from, &to, default_memory_provider!(), default_memory_provider!());

    // within this area, we guarantee that no error occurs
    for k in -412..=412 {
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

    for k in (-17 * 97)..=(17 * 97) {
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
    let from = vec![Zn::new(97), Zn::new(3)];
    let to = vec![Zn::new(17)];
    let table = AlmostExactBaseConversion::new(&from, &to, default_memory_provider!(), default_memory_provider!());
    
    for k in 0..291 {
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
    let table = AlmostExactBaseConversion::new(&from, &to, default_memory_provider!(), default_memory_provider!());

    for k in &[0, 1, 2, 17, 97, 113, 17 * 113, 18, 98, 114] {
        let x = from.iter().map(|R| R.int_hom().map(*k)).collect::<Vec<_>>();
        let y = to.iter().map(|R| R.int_hom().map(*k)).collect::<Vec<_>>();
        let mut actual = to.iter().map(|R| R.int_hom().map(*k)).collect::<Vec<_>>();

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
    let table = AlmostExactBaseConversion::new(&from, &to, default_memory_provider!(), default_memory_provider!());

    for k in &[0, 1, 2, 17, 97, 113, 17 * 113, 18, 98, 114] {
        let x = from.iter().map(|R| R.int_hom().map(*k)).collect::<Vec<_>>();
        let y = to.iter().map(|R| R.int_hom().map(*k)).collect::<Vec<_>>();
        let mut actual = to.iter().map(|R| R.int_hom().map(*k)).collect::<Vec<_>>();

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
    let in_moduli_count = 80;
    let out_moduli_count = 50;
    let cols = 1000;
    let mut primes = (1000..).map(|k| 1024 * k + 1).filter(|p| is_prime(&StaticRing::<i64>::RING, p, 10)).map(|p| Zn::new(p as u64));
    let in_moduli = primes.by_ref().take(in_moduli_count).collect::<Vec<_>>();
    let out_moduli = primes.take(out_moduli_count).collect::<Vec<_>>();
    let fastmul_in_moduli = in_moduli.iter().map(|Fp| (*Fp)).collect();
    let fastmul_out_moduli = out_moduli.iter().map(|Fp| (*Fp)).collect();
    let conv = AlmostExactBaseConversion::new_generic(in_moduli.clone(), fastmul_in_moduli, out_moduli.clone(), fastmul_out_moduli, CachingMemoryProvider::new(1), default_memory_provider!());
    
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