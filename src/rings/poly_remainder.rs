use dense_poly::DensePolyRing;
use feanor_math::algorithms::cyclotomic::cyclotomic_polynomial;
use feanor_math::algorithms::int_factor::factor;
use feanor_math::primitive_int::StaticRing;
use feanor_math::ring::*;
use feanor_math::rings::poly::*;
use sparse_poly::SparsePolyRing;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum SmallCoeff {
    Zero = 0, One = 1, NegOne = -1
}

struct SparsePolyReducer {
    degree: usize,
    stride: usize,
    coefficients: Vec<SmallCoeff>
}

impl SparsePolyReducer {

    fn new<P>(poly_ring: P, poly: &El<P>, stride: usize) -> Self
        where P: RingStore,
            P::Type: PolyRing
    {
        assert!(stride > 0);
        assert!(!poly_ring.is_zero(poly));
        assert!(poly_ring.base_ring().is_one(poly_ring.lc(poly).unwrap()));
        let coefficients = (0..poly_ring.degree(poly).unwrap()).map(|i| {
            let c = poly_ring.coefficient_at(poly, i);
            if poly_ring.base_ring().is_zero(c) {
                SmallCoeff::Zero
            } else if poly_ring.base_ring().is_one(c) {
                SmallCoeff::NegOne
            } else {
                assert!(poly_ring.base_ring().is_neg_one(c));
                SmallCoeff::One
            }
        }).collect::<Vec<_>>();
        return Self {
            degree: poly_ring.degree(poly).unwrap() * stride,
            stride: stride,
            coefficients: coefficients
        };
    }

    fn remainder<R>(&self, data: &mut [El<R>], ring: R)
        where R: RingStore
    {
        let mut start_pos_it = (self.degree..data.len()).step_by(self.stride).rev();
        if let Some(i) = start_pos_it.next() {
            let stride = data.len() - i;
            let (base, reduce) = data[(i - self.degree)..].split_at_mut(self.degree);
            for j in 0..self.coefficients.len() {
                match self.coefficients[j] {
                    SmallCoeff::Zero => {},
                    SmallCoeff::One => for k in 0..stride {
                        ring.add_assign_ref(&mut base[k + j * self.stride], &reduce[k]);
                    },
                    SmallCoeff::NegOne => for k in 0..stride {
                        ring.sub_assign_ref(&mut base[k + j * self.stride], &reduce[k]);
                    }
                }
            }
        }
        for i in start_pos_it {
            let (base, reduce) = data[(i - self.degree)..].split_at_mut(self.degree);
            for j in 0..self.coefficients.len() {
                match self.coefficients[j] {
                    SmallCoeff::Zero => {},
                    SmallCoeff::One => for k in 0..self.stride {
                        ring.add_assign_ref(&mut base[k + j * self.stride], &reduce[k]);
                    },
                    SmallCoeff::NegOne => for k in 0..self.stride {
                        ring.sub_assign_ref(&mut base[k + j * self.stride], &reduce[k]);
                    }
                }
            }
        }
    }
}

pub struct CyclotomicPolyReducer {
    mod_poly_sequence: Vec<SparsePolyReducer>
}

impl CyclotomicPolyReducer {

    pub fn new(n: i64) -> Self {
        let factorization = factor(StaticRing::<i64>::RING, n);
        let poly_ring = SparsePolyRing::new(StaticRing::<i64>::RING, "X");
        let mut mod_poly_sequence = Vec::new();
        let mut current_n = 1;
        let mut current_stride = n;

        mod_poly_sequence.push(SparsePolyReducer::new(&poly_ring, &poly_ring.sub(poly_ring.indeterminate(), poly_ring.one()), current_stride as usize));

        for i in 0..factorization.len() {
            let (p, e) = factorization[i];
            current_n *= p;
            current_stride /= p;
            let cyclotomic_poly = cyclotomic_polynomial(&poly_ring, current_n as usize);
            mod_poly_sequence.push(SparsePolyReducer::new(&poly_ring, &cyclotomic_poly, current_stride as usize));
        }

        return Self {
            mod_poly_sequence: mod_poly_sequence
        };
    }

    pub fn remainder<R>(&self, data: &mut [El<R>], ring: R)
        where R: RingStore + Copy
    {
        let mut current_len = data.len();
        for reducer in &self.mod_poly_sequence {
            reducer.remainder(&mut data[..current_len], ring);
            current_len = reducer.degree;
        }
    }
}

#[test]
fn test_sparse_poly_remainder() {
    let poly_ring = DensePolyRing::new(StaticRing::<i64>::RING, "X");
    let poly = cyclotomic_polynomial(&poly_ring, 5);
    let reducer = SparsePolyReducer::new(&poly_ring, &poly, 3);
    let mut data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19];
    let expected = [4, 5, 6, 10, -9, -9, -6, -6, -6, -3, -3, -3];

    reducer.remainder(&mut data, StaticRing::<i32>::RING);

    assert_eq!(&expected, &data[..12]);

    let poly_ring = DensePolyRing::new(StaticRing::<i64>::RING, "X");
    let poly = poly_ring.sub(poly_ring.indeterminate(), poly_ring.one());
    let reducer = SparsePolyReducer::new(&poly_ring, &poly, 13);
    let mut data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25];
    let expected = [15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 13];

    reducer.remainder(&mut data, StaticRing::<i32>::RING);

    assert_eq!(&expected, &data[..13]);
}

#[test]
fn test_cyclotomic_poly_remainder() {
    let reducer = CyclotomicPolyReducer::new(4 * 5 * 7);
    let poly_ring = DensePolyRing::new(StaticRing::<i64>::RING, "X");
    let expected = poly_ring.div_rem_monic(poly_ring.from_terms((1..200).enumerate().map(|(i, x)| (x, i))), &cyclotomic_polynomial(&poly_ring, 4 * 5 * 7)).1;

    let mut actual = (1..200).collect::<Vec<_>>();
    reducer.remainder(&mut actual, StaticRing::<i64>::RING);

    for i in 0..48 {
        assert_eq!(poly_ring.coefficient_at(&expected, i), &actual[i]);
    }
}