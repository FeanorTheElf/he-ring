use feanor_math::homomorphism::Homomorphism;
use feanor_math::integer::IntegerRingStore;
use feanor_math::primitive_int::StaticRing;
use feanor_math::ring::{RingExtension, RingStore};
use feanor_math::rings::poly::{PolyRing, PolyRingStore};
use feanor_math::rings::zn::zn_64::Zn;
use feanor_math::rings::zn::{ZnRing, ZnRingStore};

use crate::digitextract::ArithCircuit;

///
/// Returns the best arithmetic circuit that computes a function
/// ```text
/// digitex: Z/2^eZ -> (Z/2^eZ)^log(e)
/// ```
/// that satisfies `digitex(x)[i] = (x mod 2) mod 2^(2^i)`.
/// `e` must be a power of two.
/// 
/// Uses a lookup-table, consisting mainly of the values from [https://ia.cr/2022/1364].
/// 
pub fn digit_extraction_p_2(e: usize) -> ArithCircuit {
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

#[test]
fn test_digit_extraction_p_2() {
    let circuit = digit_extraction_p_2(16);
    let ring = Zn::new(1 << 16);
    let hom = ring.can_hom(&StaticRing::<i64>::RING).unwrap();
    for x in 0..(1 << 16) {
        for (i, actual) in (0..=4).zip(circuit.evaluate(&[hom.map(x)], &ring)) {
            assert_eq!(x % 2, ring.smallest_positive_lift(actual) % (1 << (1 << i)));
        }
    }
}