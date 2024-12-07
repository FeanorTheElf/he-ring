use std::alloc::Global;
use std::cmp::min;

use feanor_math::algorithms::int_factor::is_prime_power;
use feanor_math::algorithms::interpolate::interpolate;
use feanor_math::divisibility::*;
use feanor_math::homomorphism::Homomorphism;
use feanor_math::integer::{int_cast, BigIntRing, IntegerRingStore};
use feanor_math::primitive_int::StaticRing;
use feanor_math::ring::*;
use feanor_math::rings::poly::dense_poly::DensePolyRing;
use feanor_math::seq::*;
use feanor_math::rings::poly::{PolyRing, PolyRingStore};
use feanor_math::rings::zn::{zn_64, ZnRing, ZnRingStore};

use crate::digitextract::ArithCircuit;

///
/// Returns the best arithmetic circuit that computes a function
/// ```text
/// digitex: Z/2^eZ -> (Z/2^eZ)^log(e)
/// ```
/// that satisfies `digitex(x)[i] = (x mod 2) mod 2^(2^i)`.
/// `e` must be a power of two.
/// 
/// Uses a lookup-table, consisting mainly of the values from [https://ia.cr/2022/1364], except for
/// `e = 16`, where there seemed to be a mistake in the paper.
/// 
pub fn precomputed_p_2(e: usize) -> ArithCircuit {
    let log2_e = StaticRing::<i64>::RING.abs_log2_ceil(&(e as i64)).unwrap();
    assert_eq!(e, 1 << log2_e);
    
    let id = ArithCircuit::linear_transform(&[1]);
    let f0 = id.clone();
    let f1 = id.tensor(&ArithCircuit::mul()).compose(&ArithCircuit::select(1, &[0, 0, 0]).compose(&f0));
    let f2 = id.tensor(&id).tensor(&ArithCircuit::mul()).compose(&ArithCircuit::select(2, &[0, 1, 1, 1]).compose(&f1));
    
    let f3_comp = ArithCircuit::add().compose(&ArithCircuit::linear_transform(&[112]).tensor(
        &ArithCircuit::mul().compose(&ArithCircuit::linear_transform(&[94, 121]).output_twice())
    )).compose(&ArithCircuit::select(2, &[0, 0, 1]));
    let f3 = id.tensor(&id).tensor(&id).tensor(&f3_comp).compose(&ArithCircuit::select(3, &[0, 1, 2, 1, 2]).compose(&f2));

    let f4_comp = ArithCircuit::add().compose(&ArithCircuit::linear_transform(&[1984, 528, 22620]).tensor(
        &ArithCircuit::mul().compose(&ArithCircuit::linear_transform(&[226, 113]).tensor(&ArithCircuit::linear_transform(&[8, 2, 301])))
    )).compose(&ArithCircuit::select(3, &[0, 1, 2, 1, 2, 0, 1, 2]));
    let f4 = id.tensor(&id).tensor(&id).tensor(&id).tensor(&f4_comp).compose(&ArithCircuit::select(4, &[0, 1, 2, 3, 1, 2, 3]).compose(&f3));

    return match log2_e {
        0 => f0,
        1 => f1,
        2 => f2,
        3 => f3,
        4 => f4,
        5.. => panic!("no table entry for {}", e)
    };
}

///
/// Heuristically chooses a low-depth, low-complexity circuit that
/// evaluates all the given univariate polynomials.
/// 
pub fn poly_to_circuit<P>(poly_ring: P, polys: &[El<P>]) -> ArithCircuit
    where P: RingStore,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing + DivisibilityRing
{
    let ZZ = StaticRing::<i64>::RING;
    let degrees = polys.iter().map(|f| poly_ring.degree(f).unwrap() as usize).collect::<Vec<_>>();
    let max_deg = degrees.iter().copied().max().unwrap();
    let optimal_depths = degrees.iter().copied().map(|d| ZZ.abs_log2_ceil(&(d as i64)).unwrap()).collect::<Vec<_>>();
    
    let baby_steps = (1..max_deg).filter(|bs| {
            let (depths, _) = low_depth_paterson_stockmeyer_cost((&degrees).copy_els(), *bs);
            (0..optimal_depths.len()).all(|i| depths.at(i) <= optimal_depths[i] + 1)
        })
        .min_by_key(|bs| low_depth_paterson_stockmeyer_cost((&degrees).copy_els(), *bs).1)
        .unwrap();

    return low_depth_paterson_stockmeyer(&poly_ring, polys, baby_steps);
}

pub fn low_depth_paterson_stockmeyer_cost<V>(degrees: V, baby_steps: usize) -> (/* mul depth */ impl VectorFn<usize>, /* mul count */ usize)
    where V: VectorFn<usize>
{
    let ZZ = StaticRing::<i64>::RING;
    let max_deg = degrees.iter().max().unwrap();
    let giant_steps = max_deg / baby_steps + 1;
    let giant_steps_half = giant_steps / 2 + 1;

    let baby_steps_mul_count = baby_steps - 1;
    let giant_steps_mul_count = giant_steps_half - 2;
    let mut final_mul_count = 0;
    for d in degrees.iter() {
        final_mul_count += d / baby_steps;
        // in this case we need one multiplication to get x^(d - (d % baby_steps)) and one to multiply it with the block
        if d / baby_steps > 1 && (d / baby_steps) % 2 == 1 {
            final_mul_count += 1;
        }
    }
    let mul_count = baby_steps_mul_count + giant_steps_mul_count + final_mul_count;

    let mul_depths = degrees.map_fn(move |d| ZZ.abs_log2_ceil(&min(baby_steps as i64, d as i64)).unwrap() as usize + ZZ.abs_log2_ceil(&((d / baby_steps) as i64)).map(|x| x + 1).unwrap_or(0) as usize);

    return (mul_depths, mul_count);
}

///
/// A low-depth variant of Paterson-Stockmeyer evaluation of polynomials
/// 
pub fn low_depth_paterson_stockmeyer<P>(poly_ring: P, polys: &[El<P>], baby_steps: usize) -> ArithCircuit
    where P: RingStore,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing
{
    let ZZ = StaticRing::<i64>::RING;
    let max_deg = polys.iter().map(|f| poly_ring.degree(f).unwrap_or(0)).max().unwrap();

    fn compute_power_circuit(deg_exclusive: usize) -> ArithCircuit {
        let mut result = ArithCircuit::constant(1).tensor(&ArithCircuit::identity(1));
        while result.output_count() < deg_exclusive {
            let n = result.output_count();
            let mut next_circuit = ArithCircuit::identity(n);
            for i in 1..=min(deg_exclusive - n, n - 1) {
                next_circuit = next_circuit.tensor(
                    &ArithCircuit::mul().compose(
                        &ArithCircuit::select(n, &[n - 1, i])
                    )
                );
            }
            assert_eq!(next_circuit.input_count(), (min(deg_exclusive - n, n - 1) + 1) * n);
            assert_eq!(next_circuit.output_count(), min(deg_exclusive - n, n - 1) + n);
            result = next_circuit.compose(&result.output_times(min(deg_exclusive - n, n - 1) + 1));
        }
        assert!(result.output_count() == deg_exclusive);
        return result;
    }

    let giant_steps = max_deg / baby_steps + 1;
    let giant_steps_half = giant_steps / 2 + 1;
    assert!((giant_steps - 1) * baby_steps + baby_steps > max_deg);
    assert!((giant_steps - 1) * baby_steps <= max_deg);

    // now baby_step_circuit computes (1, x, x^2, ..., x^(baby_steps - 1))
    let baby_step_circuit = compute_power_circuit(baby_steps + 1);
    assert_eq!(baby_steps - 1, baby_step_circuit.mul_count());
    assert_eq!(ZZ.abs_log2_ceil(&(baby_steps as i64)).unwrap() as usize, baby_step_circuit.max_mul_depth());

    // giant_step_circuit computes (1, x, ..., x^(baby_steps - 1), 1, x^baby_steps, x^(2 baby_steps), ..., x^(floor(giant_steps / 2) * baby_steps - baby_steps))
    let giant_step_circuit = ArithCircuit::identity(baby_steps).tensor(&compute_power_circuit(giant_steps_half)).compose(&baby_step_circuit);
    assert_eq!(baby_steps - 1 + giant_steps_half - 2, giant_step_circuit.mul_count());
    assert_eq!(ZZ.abs_log2_ceil(&(giant_steps_half as i64 - 1)).unwrap() as usize, giant_step_circuit.max_mul_depth() - baby_step_circuit.max_mul_depth());
    assert_eq!(giant_step_circuit.input_count(), 1);
    assert_eq!(giant_step_circuit.output_count(), baby_steps + giant_steps_half);

    let all_poly_parts: Vec<Vec<ArithCircuit>> = polys.iter().map(|f: &_| (0..(poly_ring.degree(f).unwrap() / baby_steps + 1)).map(|i| ArithCircuit::linear_transform(&(0..baby_steps).map(|j|
        int_cast(poly_ring.base_ring().smallest_lift(poly_ring.base_ring().clone_el(poly_ring.coefficient_at(f, i * baby_steps + j))), StaticRing::<i64>::RING, poly_ring.base_ring().integer_ring())
    ).collect::<Vec<_>>())).collect()).collect();

    let select_baby_steps = ArithCircuit::select(baby_steps + giant_steps_half, &(0..baby_steps).collect::<Vec<_>>());

    let mut result = ArithCircuit::empty();
    for (poly, poly_parts) in polys.iter().zip(all_poly_parts.iter()) {

        let mut compute_poly_circuit = poly_parts[0].compose(&select_baby_steps);
        let highest_block = poly_ring.degree(poly).unwrap() / baby_steps;
        
        for i in 1..=(highest_block / 2) {
            assert_eq!(baby_steps + giant_steps_half, compute_poly_circuit.input_count());
            assert_eq!(1, compute_poly_circuit.output_count());

            let low_part = &poly_parts[i];
            let high_part = &poly_parts[i + highest_block / 2];

            let compute_part = ArithCircuit::mul().compose(
                &ArithCircuit::add().compose(
                    &low_part.compose(&select_baby_steps).tensor(
                        &ArithCircuit::mul().compose(&high_part.compose(&select_baby_steps).tensor(&ArithCircuit::select(baby_steps + giant_steps_half, &[baby_steps + highest_block / 2])))
                    )
                ).tensor(&ArithCircuit::select(baby_steps + giant_steps_half, &[baby_steps + i]))
            ).compose(&ArithCircuit::identity(baby_steps + giant_steps_half).output_times(4));

            compute_poly_circuit = ArithCircuit::add().compose(&compute_poly_circuit.tensor(&compute_part))
                .compose(&ArithCircuit::identity(baby_steps + giant_steps_half).output_twice());
        }

        if highest_block == 1 {
            let compute_part = ArithCircuit::mul().compose(
                &poly_parts[highest_block].compose(&select_baby_steps).tensor(
                    &ArithCircuit::select(baby_steps + giant_steps_half, &[baby_steps + 1])
                )
            ).compose(&ArithCircuit::identity(baby_steps + giant_steps_half).output_times(2));  
            compute_poly_circuit = ArithCircuit::add().compose(&compute_poly_circuit.tensor(&compute_part))
                .compose(&ArithCircuit::identity(baby_steps + giant_steps_half).output_twice());
        } else if highest_block % 2 == 1 {
            let highest_block_power = ArithCircuit::mul().compose(&ArithCircuit::select(baby_steps + giant_steps_half, &[baby_steps + highest_block / 2]).tensor(
                &ArithCircuit::select(baby_steps + giant_steps_half, &[baby_steps + highest_block / 2 + 1])
            )).compose(&ArithCircuit::identity(baby_steps + giant_steps_half).output_twice());
            let compute_part = ArithCircuit::mul().compose(
                &poly_parts[highest_block].compose(&select_baby_steps).tensor(&highest_block_power)
            ).compose(&ArithCircuit::identity(baby_steps + giant_steps_half).output_twice());
            compute_poly_circuit = ArithCircuit::add().compose(&compute_poly_circuit.tensor(&compute_part))
                .compose(&ArithCircuit::identity(baby_steps + giant_steps_half).output_twice());
        }

        result = result.tensor(&compute_poly_circuit);
    }
    let result = result.compose(&giant_step_circuit.output_times(polys.len()));

    let (expected_mul_depths, expected_mul_count) = low_depth_paterson_stockmeyer_cost(polys.as_fn().map_fn(|f| poly_ring.degree(f).unwrap() as usize), baby_steps);
    for i in 0..polys.len() {
        assert_eq!(expected_mul_depths.at(i), result.mul_depth(i));
    }
    assert_eq!(expected_mul_count, result.mul_count());
    return result;
}

fn digit_extraction_poly<P>(poly_ring: P) -> El<P>
    where P: RingStore,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing + DivisibilityRing
{
    let Zn = poly_ring.base_ring();
    let (p, e) = is_prime_power(Zn.integer_ring(), Zn.modulus()).unwrap();
    let p = int_cast(p, StaticRing::<i64>::RING, Zn.integer_ring()) as usize;
    let hom = Zn.can_hom(Zn.integer_ring()).unwrap().compose(Zn.integer_ring().can_hom(&StaticRing::<i64>::RING).unwrap());
    let mut current = poly_ring.pow(poly_ring.indeterminate(), p);
    for i in 1..e {
        let mut correction = interpolate(
            &poly_ring, 
            (0..p).map_fn(|j| hom.map(j as i64)), 
            (0..p).map_fn(|j| Zn.checked_div(
                &Zn.sub(poly_ring.evaluate(&current, &hom.map(j as i64), &Zn.identity()), hom.map(j as i64)), 
                &Zn.pow(hom.map(p as i64), i as usize)
            ).unwrap()),
            Global
        ).unwrap();
        poly_ring.inclusion().mul_assign_ref_map(&mut correction, &Zn.pow(hom.map(p as i64), i as usize));
        poly_ring.sub_assign(&mut current, correction);
    }
    return current;
}

pub fn basic_digit_extract_circuit(p: i64, e: usize) -> ArithCircuit {
    let poly_ring = DensePolyRing::new(zn_64::Zn::new(StaticRing::<i64>::RING.pow(p, e) as u64), "X");
    let f = digit_extraction_poly(&poly_ring);
    let f_circuit = poly_to_circuit(&poly_ring, std::slice::from_ref(&f));
    let mut result = ArithCircuit::identity(1);
    for i in 1..e {
        result = ArithCircuit::identity(i).tensor(&f_circuit.compose(&ArithCircuit::select(i, &[i - 1]))).compose(&result.output_twice());
    }
    return ArithCircuit::select(e, &(1..e).collect::<Vec<_>>()).compose(&result);
}

///
/// Computes `min { n | n! % k == 0 }`
/// 
pub fn mu(k: i64) -> i64 {
    const ZZbig: BigIntRing = BigIntRing::RING;
    let mut n = 1;
    let mut n_fac = ZZbig.one();
    while ZZbig.checked_div(&n_fac, &int_cast(k, &ZZbig, &StaticRing::<i64>::RING)).is_none() {
        n += 1;
        ZZbig.mul_assign(&mut n_fac, int_cast(n, &ZZbig, &StaticRing::<i64>::RING));
    }
    return n;
}

pub fn falling_factorial_poly<P>(poly_ring: P, m: usize) -> El<P>
    where P: RingStore,
        P::Type: PolyRing
{
    poly_ring.prod((0..m).map(|j| poly_ring.sub(poly_ring.indeterminate(), poly_ring.int_hom().map(j as i32))))
}

///
/// Returns the lowest-degree polynomial `f` such that `f(x) = lift(x mod p) mod p^k`.
/// 
pub fn digit_retain_poly<P>(poly_ring: P, k: usize) -> El<P>
    where P: RingStore + Copy,
        P::Type: PolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: ZnRing + DivisibilityRing
{
    assert!(k > 0);
    if k == 1 {
        return poly_ring.indeterminate();
    }
    let Zn = poly_ring.base_ring();
    let hom = Zn.can_hom(Zn.integer_ring()).unwrap().compose(Zn.integer_ring().can_hom(&StaticRing::<i64>::RING).unwrap());
    let (p, e) = is_prime_power(Zn.integer_ring(), Zn.modulus()).unwrap();
    assert!(e >= k);
    let p = int_cast(p, StaticRing::<i64>::RING, Zn.integer_ring());
    let mut current = poly_ring.evaluate(&digit_extraction_poly(&poly_ring), &digit_retain_poly(poly_ring, k - 1), &poly_ring.inclusion());

    let mut current_e = 0;
    while Zn.checked_div(poly_ring.lc(&current).unwrap(), &Zn.pow(hom.map(p), current_e)).is_some() {
        let null_poly = poly_ring.inclusion().mul_map(
            falling_factorial_poly(&poly_ring, mu(StaticRing::<i64>::RING.pow(p, k - current_e)) as usize),
            Zn.pow(hom.map(p), current_e)
        );
        while let Some(quo) = Zn.checked_div(poly_ring.lc(&current).unwrap(), &poly_ring.lc(&null_poly).unwrap()) {
            if poly_ring.degree(&current).unwrap() < poly_ring.degree(&null_poly).unwrap() {
                break;
            }
            let mut subtractor = poly_ring.inclusion().mul_ref_map(&null_poly, &quo);
            poly_ring.mul_assign_monomial(&mut subtractor, poly_ring.degree(&current).unwrap() - poly_ring.degree(&null_poly).unwrap());
            poly_ring.sub_assign(&mut current, subtractor);
        }
        current_e += 1;
    }
    return current;
}

#[cfg(test)]
use feanor_math::rings::zn::zn_64::Zn;
#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use feanor_math::rings::finite::FiniteRingStore;

#[test]
fn test_digit_extraction_p_2() {
    let circuit = precomputed_p_2(16);
    let ring = Zn::new(1 << 16);
    let hom = ring.can_hom(&StaticRing::<i64>::RING).unwrap();
    for x in 0..(1 << 16) {
        for (i, actual) in (0..=4).zip(circuit.evaluate(&[hom.map(x)], &ring)) {
            assert_eq!(x % 2, ring.smallest_positive_lift(actual) % (1 << (1 << i)));
        }
    }
}

#[test]
fn test_digit_extraction_poly() {
    let Zn = Zn::new(17 * 17 * 17);
    let P = DensePolyRing::new(Zn, "X");
    let digit_extract = digit_extraction_poly(&P);
    for k in 0..(17 * 17 * 17) {
        assert_eq!(k % 17, Zn.smallest_positive_lift(P.evaluate(&digit_extract, &Zn.coerce(&StaticRing::<i64>::RING, k), &Zn.identity())) % (17 * 17));
    }
    for k_low in 0..17 {
        for k_high in (0..(17 * 17 * 17)).step_by(17 * 17) {
            assert_el_eq!(&Zn, &Zn.coerce(&StaticRing::<i64>::RING, k_low), &P.evaluate(&digit_extract, &Zn.coerce(&StaticRing::<i64>::RING, k_low + k_high), &Zn.identity()));
        }
    }
}

#[test]
fn test_digit_retain_poly() {
    let Zn = Zn::new(1024);
    let P = DensePolyRing::new(Zn, "X");
    let digit_retain = digit_retain_poly(&P, 3);
    assert_eq!(Some(3), P.degree(&digit_retain));
    for k in 0..1024 {
        assert_eq!(k % 2, Zn.smallest_positive_lift(P.evaluate(&digit_retain, &Zn.coerce(&StaticRing::<i64>::RING, k), &Zn.identity())) % 8);
    }
    let digit_retain = digit_retain_poly(&P, 6);
    assert_eq!(Some(6), P.degree(&digit_retain));
    for k in 0..1024 {
        assert_eq!(k % 2, Zn.smallest_positive_lift(P.evaluate(&digit_retain, &Zn.coerce(&StaticRing::<i64>::RING, k), &Zn.identity())) % 64);
    }

    let Zn = Zn::new(17 * 17 * 17);
    let P = DensePolyRing::new(Zn, "X");
    let digit_retain = digit_retain_poly(&P, 3);
    assert_eq!(Some(33), P.degree(&digit_retain));
    for k in 0..(17 * 17 * 17) {
        assert_el_eq!(&Zn, &Zn.coerce(&StaticRing::<i64>::RING, k % 17), &P.evaluate(&digit_retain, &Zn.coerce(&StaticRing::<i64>::RING, k), &Zn.identity()));
    }
    
    let Zn = Zn::new(257 * 257);
    let P = DensePolyRing::new(Zn, "X");
    let digit_retain = digit_retain_poly(&P, 2);
    assert_eq!(Some(257), P.degree(&digit_retain));
    for k in 0..257 {
        assert_el_eq!(&Zn, &Zn.coerce(&StaticRing::<i64>::RING, 2), &P.evaluate(&digit_retain, &Zn.coerce(&StaticRing::<i64>::RING, 2 + k * 257), &Zn.identity()));
    }
}

#[test]
#[ignore]
fn test_digit_retain_poly_large() {
    let Zn = Zn::new(257 * 257 * 257);
    let P = DensePolyRing::new(Zn, "X");
    let digit_retain = digit_retain_poly(&P, 3);
    assert_el_eq!(&Zn, &Zn.coerce(&StaticRing::<i64>::RING, 251), &P.evaluate(&digit_retain, &Zn.coerce(&StaticRing::<i64>::RING, 132092), &Zn.identity()));
    for k in 0..(257 * 257) {
        assert_el_eq!(&Zn, &Zn.coerce(&StaticRing::<i64>::RING, 2), &P.evaluate(&digit_retain, &Zn.coerce(&StaticRing::<i64>::RING, 2 + k * 257), &Zn.identity()));
    }
}

#[test]
fn test_paterson_stockmeyer() {
    let Zn = Zn::new(17);
    let P = DensePolyRing::new(Zn, "X");
    // 1 + 2 X^3 + 3 X^4 + 4 X^5 + 8 X^7
    let poly = P.from_terms([(1, 0), (2, 3), (3, 4), (4, 5), (8, 7)].into_iter().map(|(c, d)| (Zn.int_hom().map(c), d)));
    let circuit = low_depth_paterson_stockmeyer(&P, &[P.clone_el(&poly)], 3);
    assert_eq!(4, circuit.max_mul_depth());
    assert_eq!(4, circuit.mul_count());

    for x in Zn.elements() {
        assert_el_eq!(Zn, P.evaluate(&poly, &x, &P.base_ring().identity()), circuit.evaluate(&[x], P.base_ring()).next().unwrap());
    }
}

#[test]
fn test_paterson_stockmeyer_multiple_polys() {
    let Zn = Zn::new(17);
    let P = DensePolyRing::new(Zn, "X");
    // 1 + 2 X^3 + 3 X^4 + 4 X^5 + 8 X^7
    let f = P.from_terms([(1, 0), (2, 3), (3, 4), (4, 5), (8, 7)].into_iter().map(|(c, d)| (Zn.int_hom().map(c), d)));
    // 2 + X + 2 X^2 + 3 X^3 + 4 X^4 + 5 X^5 + 6 X^6 + 7 X^7 + 8 X^8 + 9 X^9
    let g = P.from_terms([(2, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)].into_iter().map(|(c, d)| (Zn.int_hom().map(c), d)));
    let circuit = low_depth_paterson_stockmeyer(&P, &[P.clone_el(&f), P.clone_el(&g)], 4);
    assert_eq!(4, circuit.max_mul_depth());
    assert_eq!(3, circuit.mul_depth(0));
    assert_eq!(4, circuit.mul_depth(1));
    assert_eq!(6, circuit.mul_count());

    for x in Zn.elements() {
        let mut result_it = circuit.evaluate(std::slice::from_ref(&x), P.base_ring());
        assert_el_eq!(Zn, P.evaluate(&f, &x, &P.base_ring().identity()), result_it.next().unwrap());
        assert_el_eq!(Zn, P.evaluate(&g, &x, &P.base_ring().identity()), result_it.next().unwrap());
    }

    // 1 + X^12
    let h = P.from_terms([(1, 0), (3, 6), (7, 9), (1, 12)].into_iter().map(|(c, d)| (Zn.int_hom().map(c), d)));
    let circuit = low_depth_paterson_stockmeyer(&P, &[P.clone_el(&f), P.clone_el(&g), P.clone_el(&h)], 4);
    assert_eq!(5, circuit.max_mul_depth());
    assert_eq!(3, circuit.mul_depth(0));
    assert_eq!(4, circuit.mul_depth(1));
    assert_eq!(5, circuit.mul_depth(2));
    assert_eq!(11, circuit.mul_count());

    for x in Zn.elements() {
        let mut result_it = circuit.evaluate(std::slice::from_ref(&x), P.base_ring());
        assert_el_eq!(Zn, P.evaluate(&f, &x, &P.base_ring().identity()), result_it.next().unwrap());
        assert_el_eq!(Zn, P.evaluate(&g, &x, &P.base_ring().identity()), result_it.next().unwrap());
        assert_el_eq!(Zn, P.evaluate(&h, &x, &P.base_ring().identity()), result_it.next().unwrap());
    }

    // 1 + X + X^2 + ... + X^15 + X^16
    let l = P.from_terms((0..=16).map(|i| (Zn.one(), i)));
    let circuit = low_depth_paterson_stockmeyer(&P, &[P.clone_el(&f), P.clone_el(&g), P.clone_el(&h), P.clone_el(&l)], 4);
    assert_eq!(5, circuit.max_mul_depth());
    assert_eq!(3, circuit.mul_depth(0));
    assert_eq!(4, circuit.mul_depth(1));
    assert_eq!(5, circuit.mul_depth(2));
    assert_eq!(5, circuit.mul_depth(3));
    assert_eq!(5 + 1 + 2 + 3 + 4, circuit.mul_count());

    for x in Zn.elements() {
        let mut result_it = circuit.evaluate(std::slice::from_ref(&x), P.base_ring());
        assert_el_eq!(Zn, P.evaluate(&f, &x, &P.base_ring().identity()), result_it.next().unwrap());
        assert_el_eq!(Zn, P.evaluate(&g, &x, &P.base_ring().identity()), result_it.next().unwrap());
        assert_el_eq!(Zn, P.evaluate(&h, &x, &P.base_ring().identity()), result_it.next().unwrap());
        assert_el_eq!(Zn, P.evaluate(&l, &x, &P.base_ring().identity()), result_it.next().unwrap());
    }
}

#[test]
fn test_best_circuit_multiple_polys() {
    let Zn = Zn::new(17);
    let P = DensePolyRing::new(Zn, "X");
    let f = P.from_terms([(1, 0), (2, 3), (3, 4), (4, 5), (8, 7)].into_iter().map(|(c, d)| (Zn.int_hom().map(c), d)));
    let g = P.from_terms([(2, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)].into_iter().map(|(c, d)| (Zn.int_hom().map(c), d)));
    let h = P.from_terms([(1, 0), (3, 6), (7, 9), (1, 12)].into_iter().map(|(c, d)| (Zn.int_hom().map(c), d)));
    let circuit = poly_to_circuit(&P, &[P.clone_el(&f), P.clone_el(&g), P.clone_el(&h)]);
    assert_eq!(5, circuit.max_mul_depth());
    assert_eq!(4, circuit.mul_depth(0));
    assert_eq!(4, circuit.mul_depth(1));
    assert_eq!(5, circuit.mul_depth(2));
    assert_eq!(8, circuit.mul_count());
    
    for x in Zn.elements() {
        let mut result_it = circuit.evaluate(std::slice::from_ref(&x), P.base_ring());
        assert_el_eq!(Zn, P.evaluate(&f, &x, &P.base_ring().identity()), result_it.next().unwrap());
        assert_el_eq!(Zn, P.evaluate(&g, &x, &P.base_ring().identity()), result_it.next().unwrap());
        assert_el_eq!(Zn, P.evaluate(&h, &x, &P.base_ring().identity()), result_it.next().unwrap());
    }

    let l = P.from_terms((0..=16).map(|i| (Zn.one(), i)));
    let circuit = poly_to_circuit(&P, &[P.clone_el(&f), P.clone_el(&g), P.clone_el(&h), P.clone_el(&l)]);
    assert_eq!(5, circuit.max_mul_depth());
    assert_eq!(4, circuit.mul_depth(0));
    assert_eq!(4, circuit.mul_depth(1));
    assert_eq!(5, circuit.mul_depth(2));
    assert_eq!(5, circuit.mul_depth(3));
    assert_eq!(11, circuit.mul_count());

    for x in Zn.elements() {
        let mut result_it = circuit.evaluate(std::slice::from_ref(&x), P.base_ring());
        assert_el_eq!(Zn, P.evaluate(&f, &x, &P.base_ring().identity()), result_it.next().unwrap());
        assert_el_eq!(Zn, P.evaluate(&g, &x, &P.base_ring().identity()), result_it.next().unwrap());
        assert_el_eq!(Zn, P.evaluate(&h, &x, &P.base_ring().identity()), result_it.next().unwrap());
        assert_el_eq!(Zn, P.evaluate(&l, &x, &P.base_ring().identity()), result_it.next().unwrap());
    }
}