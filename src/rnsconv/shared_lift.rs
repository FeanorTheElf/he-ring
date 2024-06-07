use feanor_math::matrix::submatrix::*;
use feanor_math::rings::zn::*;
use feanor_math::ring::*;
use feanor_math::homomorphism::*;
use feanor_math::integer::*;
use feanor_math::mempool::*;
use feanor_math::vector::VectorView;

use super::approx_lift::*;
use super::RNSOperation;

///
/// Computes almost exact base conversion with a shared factor.
/// The exact map would be
/// ```text
/// Z/aqZ -> Z/aq'Z, x -> lift(x) mod aq'
/// ```
/// but as usual, we allow an error of `+/- aq`, unless the shortest 
/// lift of the input is bounded by `aq/4`, in which case the result
/// is always correct.
/// 
/// The functionality is exactly as for [`AlmostExactBaseConversion`],
/// except that it might be faster by reusing the shared factor `a`.
/// 
pub struct AlmostExactSharedBaseConversion<R, M_Zn, M_Int>
    where R: ZnRingStore + Clone,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        M_Int: MemoryProvider<<<R::Type as ZnRing>::IntegerRingBase as RingBase>::Element>,
        M_Zn: MemoryProvider<El<R>>
{
    conversion: AlmostExactBaseConversion<R, M_Int, M_Zn>,
    out_moduli: Vec<R>
}

impl<R, M_Zn, M_Int> AlmostExactSharedBaseConversion<R, M_Zn, M_Int>
    where R: ZnRingStore + Clone,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        M_Int: MemoryProvider<<<R::Type as ZnRing>::IntegerRingBase as RingBase>::Element>,
        M_Zn: MemoryProvider<El<R>>
{
    pub fn new(shared_moduli: Vec<R>, additional_in_moduli: Vec<R>, additional_out_moduli: Vec<R>, memory_provider: M_Zn, memory_provider_int: M_Int) -> Self {
        let in_moduli = shared_moduli.iter().cloned().chain(additional_in_moduli.into_iter()).collect::<Vec<_>>();
        let out_moduli = shared_moduli.into_iter().chain(additional_out_moduli.iter().cloned()).collect::<Vec<_>>();
        let conversion = AlmostExactBaseConversion::new_base(in_moduli, additional_out_moduli, memory_provider_int, memory_provider);
        Self {
            out_moduli: out_moduli,
            conversion: conversion
        }
    }

    fn a_moduli_count(&self) -> usize {
        self.out_moduli.len() - self.conversion.output_rings().len()
    }
}

impl<R, M_Zn, M_Int> RNSOperation for AlmostExactSharedBaseConversion<R, M_Zn, M_Int>
    where R: ZnRingStore + Clone,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        M_Int: MemoryProvider<<<R::Type as ZnRing>::IntegerRingBase as RingBase>::Element>,
        M_Zn: MemoryProvider<El<R>>
{
    type Ring = R;
    type RingType = R::Type;

    fn input_rings<'a>(&'a self) -> &'a [Self::Ring] {
        self.conversion.input_rings()
    }

    fn output_rings<'a>(&'a self) -> &'a [Self::Ring] {
        &self.out_moduli
    }

    fn apply<V1, V2>(&self, input: Submatrix<V1, El<Self::Ring>>, mut output: SubmatrixMut<V2, El<Self::Ring>>)
        where V1: AsPointerToSlice<El<Self::Ring>>,
            V2: AsPointerToSlice<El<Self::Ring>>
    {
        assert_eq!(input.col_count(), output.col_count());
        assert_eq!(input.row_count(), self.input_rings().len());
        assert_eq!(output.row_count(), self.output_rings().len());

        self.conversion.apply(input, output.reborrow().restrict_rows(self.a_moduli_count()..self.output_rings().len()));
        for i in 0..self.a_moduli_count() {
            for j in 0..input.col_count() {
                *output.at(i, j) = self.output_rings()[i].clone_el(input.at(i, j));
            }
        }
    }
}

#[cfg(test)]
use feanor_math::rings::zn::zn_64::*;
#[cfg(test)]
use feanor_math::default_memory_provider;

#[test]
fn test_rns_shared_base_conversion() {
    let from = vec![Zn::new(17), Zn::new(97), Zn::new(113)];
    let to = vec![Zn::new(17), Zn::new(97), Zn::new(113), Zn::new(257)];
    let table = AlmostExactSharedBaseConversion::new(from.clone(), Vec::new(), vec![to[3]], default_memory_provider!(), default_memory_provider!());

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
