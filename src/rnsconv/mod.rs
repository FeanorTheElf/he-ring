use feanor_math::ring::*;
use feanor_math::rings::zn::{ZnRing, ZnRingStore};
use feanor_math::vector::*;
use feanor_math::matrix::submatrix::*;

pub mod lift;
pub mod bfv_rescale;
pub mod bgv_rescale;

pub trait RNSOperation {

    type Ring: ZnRingStore<Type = Self::RingType>;
    
    type RingType: ?Sized + ZnRing;

    type InRings<'a>: 'a + VectorView<Self::Ring>
        where Self: 'a;

    type OutRings<'a>: 'a + VectorView<Self::Ring>
        where Self: 'a;

    fn input_rings<'a>(&'a self) -> Self::InRings<'a>;

    fn output_rings<'a>(&'a self) -> Self::OutRings<'a>;

    ///
    /// Applies the RNS operation to each column of the given matrix, and writes the results to the columns
    /// of `output`. The entries of the `i`-th row are considered to be elements of `self.input_rings().at(i)`
    /// resp. `self.output_rings().at(i)`.
    ///
    fn apply<V1, V2>(&self, input: Submatrix<V1, El<Self::Ring>>, output: SubmatrixMut<V2, El<Self::Ring>>)
        where V1: AsPointerToSlice<El<Self::Ring>>,
            V2: AsPointerToSlice<El<Self::Ring>>;
}
