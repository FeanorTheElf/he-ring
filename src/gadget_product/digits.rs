use std::ops::{Deref, Range};

use feanor_math::seq::VectorFn;

///
/// A decomposition of the numbers `0..rns_len` into ranges, which we call digits.
/// 
/// The main use case is the construction of RNS gadget vectors, which are of the
/// form
/// ```text
///   g[i] = 1 mod pj  if j in digits[i]
///   g[i] = 0 mod pj  otherwise
/// ```
/// for a list of digits `digits` and `p0, ..., p(rns_len - 1)` being the RNS factors.
/// 
/// This trait (and many other components in HE-Ring) currently do not support
/// digits that are not a contiguous range of indices. More concretely, it would make
/// sense to decompose `0..6` into digits as `{0, 2, 3}, {1, 4, 5}`, but this is not
/// supported. The reason is that this allows us to take slices of data corresponding
/// to RNS factors, and get only the data corresponding to a single digit (hence avoid
/// copying the data around).
/// 
/// # Example
/// ```
/// # use feanor_math::seq::*;
/// # use he_ring::gadget_product::digits::*;
/// let digits = RNSGadgetVectorDigitList::from([3..7, 0..3, 7..10].clone_els());
/// assert_eq!(3, digits.len());
/// 
/// // the digits will be stored in an ordered way
/// assert_eq!(0..3, digits.at(0));
/// assert_eq!(3..7, digits.at(1));
/// assert_eq!(7..10, digits.at(2));
/// 
/// assert_eq!(10, digits.rns_base_len());
/// ```
/// 
#[repr(transparent)]
#[derive(Debug)]
pub struct RNSGadgetVectorDigitList {
    digit_boundaries: [usize]
}

impl RNSGadgetVectorDigitList {

    fn from_unchecked(digit_boundaries: Box<[usize]>) -> Box<Self> {
        unsafe { std::mem::transmute(digit_boundaries) }
    }

    pub fn from<V>(digits: V) -> Box<Self>
        where V: VectorFn<Range<usize>>
    {
        let mut result: Vec<usize> = Vec::with_capacity(digits.len());
        for _ in 0..digits.len() {
            let mut it = digits.iter().filter(|digit| digit.start == *result.last().unwrap_or(&0));
            if let Some(next) = it.next() {
                if it.next().is_some() {
                    panic!("multiple digits start at {}", result.last().unwrap_or(&0));
                }
                result.push(next.end);
            } else {
                panic!("no digit contains {}", result.last().unwrap_or(&0));
            }
        }
        return Self::from_unchecked(result.into_boxed_slice());
    }

    pub fn rns_base_len(&self) -> usize {
        *self.digit_boundaries.last().unwrap_or(&0)
    }

    ///
    /// Computes a balanced decomposition of `0..rns_base_len` into `digits` digits, which
    /// is often the best choice for an RNS gadget vector.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use feanor_math::seq::*;
    /// # use he_ring::gadget_product::digits::*;
    /// let digits = RNSGadgetVectorDigitList::select_digits(3, 10);
    /// assert_eq!(3, digits.len());
    /// assert_eq!(0..4, digits.at(0));
    /// assert_eq!(4..7, digits.at(1));
    /// assert_eq!(7..10, digits.at(2));
    /// ```
    /// 
    pub fn select_digits(digits: usize, rns_base_len: usize) -> Box<Self> {
        assert!(digits <= rns_base_len, "the number of gadget product digits may not exceed the number of RNS factors");
        let moduli_per_small_digit = rns_base_len / digits;
        let large_digits = rns_base_len % digits;
        let small_digits = digits - large_digits;
        let mut result = Vec::with_capacity(digits);
        let mut current = 0;
        for _ in 0..large_digits {
            current += moduli_per_small_digit + 1;
            result.push(current);
        }
        for _ in 0..small_digits {
            current += moduli_per_small_digit;
            result.push(current);
        }
        return Self::from_unchecked(result.into_boxed_slice());
    }

    ///
    /// Removes the given indices from each digit, and returns the resulting
    /// list of shorter digits.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use feanor_math::seq::*;
    /// # use he_ring::gadget_product::digits::*;
    /// let original_digits = RNSGadgetVectorDigitList::from([0..3, 3..5, 5..7].clone_els());
    /// let digits = original_digits.remove_indices(DropModuliIndices::from_ref(&[1, 2, 5], 7));
    /// assert_eq!(3, digits.len());
    /// assert_eq!(0..1, digits.at(0));
    /// assert_eq!(1..3, digits.at(1));
    /// assert_eq!(3..4, digits.at(2));
    /// ```
    /// If all indices from a given digit are removed, the whole digit is removed.
    /// ```
    /// # use feanor_math::seq::*;
    /// # use he_ring::gadget_product::digits::*;
    /// let original_digits = RNSGadgetVectorDigitList::from([0..3, 3..5, 5..7].clone_els());
    /// let digits = original_digits.remove_indices(DropModuliIndices::from_ref(&[0, 1, 2, 5], 7));
    /// assert_eq!(2, digits.len());
    /// assert_eq!(0..2, digits.at(0));
    /// assert_eq!(2..3, digits.at(1));
    /// ```
    /// 
    pub fn remove_indices(&self, drop_rns_factors: &DropModuliIndices) -> Box<Self> {
        let mut result = Vec::new();
        let mut current_len = 0;
        for range in self.iter() {
            let dropped_els = drop_rns_factors.num_within(&range);
            if dropped_els != range.end - range.start {
                current_len += range.end - range.start - dropped_els;
                result.push(current_len);
            }
        }
        debug_assert!(*result.last().unwrap_or(&0) == self.rns_base_len() - drop_rns_factors.len());
        return Self::from_unchecked(result.into_boxed_slice());
    }
}

impl VectorFn<Range<usize>> for RNSGadgetVectorDigitList {

    fn len(&self) -> usize {
        self.digit_boundaries.len()
    }

    fn at(&self, i: usize) -> Range<usize> {
        if i == 0 {
            0..self.digit_boundaries[0]
        } else {
            self.digit_boundaries[i - 1]..self.digit_boundaries[i]
        }
    }
}

impl Clone for Box<RNSGadgetVectorDigitList> {

    fn clone(&self) -> Self {
        RNSGadgetVectorDigitList::from_unchecked(self.digit_boundaries.to_owned().into_boxed_slice())
    }
}

///
/// Thin wrapper around ordered slices `[usize]`, used to store a set of indices
/// of RNS factors. In most cases, it refers to those RNS factors that should be
/// dropped from a "master RNS base" to get to the current state.
/// 
#[repr(transparent)]
#[derive(Debug)]
pub struct DropModuliIndices {
    drop_rns_moduli_indices: [usize]
}

impl Deref for DropModuliIndices {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        &self.drop_rns_moduli_indices
    }
}

impl DropModuliIndices {

    fn from_unchecked(indices: Box<[usize]>) -> Box<Self> {
        unsafe { std::mem::transmute(indices) }
    }

    fn from_ref_unchecked<'a>(indices: &'a [usize]) -> &'a Self {
        return unsafe { std::mem::transmute(indices) };
    }

    fn check_valid(indices: &[usize], rns_base_len: usize) {
        for i in indices {
            assert!(*i < rns_base_len, "all indices must be valid for an RNS base of length {}, but found {}", rns_base_len, *i);
        }
        for (i0, j0) in indices.iter().enumerate() {
            for (i1, j1) in indices.iter().enumerate() {
                assert!(i0 == i1 || j0 != j1, "all indices must be distinct, but found indices[{}] == {} == indices[{}]", i0, j0, i1);
            }
        }
    }

    pub fn from_ref<'a>(indices: &'a [usize], rns_base_len: usize) -> &'a Self {
        Self::check_valid(indices, rns_base_len);
        assert!(indices.is_sorted());
        return Self::from_ref_unchecked(indices);
    }

    pub fn from_ref_unsorted<'a>(indices: &'a mut [usize], rns_base_len: usize) -> &'a Self {
        Self::check_valid(indices, rns_base_len);
        indices.sort_unstable();
        return Self::from_ref_unchecked(indices);
    }

    pub fn from(mut indices: Vec<usize>, rns_base_len: usize) -> Box<Self> {
        Self::check_valid(&indices, rns_base_len);
        indices.sort_unstable();
        return Self::from_unchecked(indices.into_boxed_slice());
    }

    pub fn contains(&self, i: usize) -> bool {
        self.drop_rns_moduli_indices.binary_search(&i).is_ok()
    }

    pub fn num_within(&self, range: &Range<usize>) -> usize {
        match (self.drop_rns_moduli_indices.binary_search(&range.start), self.drop_rns_moduli_indices.binary_search(&range.end)) {
            (Ok(i), Ok(j)) |
            (Ok(i), Err(j)) |
            (Err(i), Ok(j)) |
            (Err(i), Err(j)) => j - i
        }
    }

    pub fn subtract(&self, other: &Self) -> Box<Self> {
        Self::from_unchecked(self.drop_rns_moduli_indices.iter().copied().filter(|i| !other.contains(*i)).collect())
    }

    pub fn intersect(&self, other: &Self) -> Box<Self> {
        Self::from_unchecked(self.drop_rns_moduli_indices.iter().copied().filter(|i| other.contains(*i)).collect())
    }

    ///
    /// Returns the indices contained in `self` but not in `context`, however relative to the
    /// RNS base that has `context` already removed.
    /// 
    pub fn within(&self, context: &Self) -> Box<Self> {
        if self.len() == 0 {
            assert!(context.len() == 0);
            return Self::empty();
        }
        let mut result = Vec::with_capacity(self.len() - context.len());
        let mut current = 0;
        let largest = self[self.len() - 1];
        assert!(context.len() == 0 || context[context.len() - 1] <= largest);

        // I guess this could be optimized, but it's fast enough
        for i in 0..=largest {
            if context.contains(i) {
                continue;
            }
            if self.contains(i) {
                result.push(current);
            }
            current += 1;
        }
        assert!(result.len() == self.len() - context.len());
        return Self::from_unchecked(result.into_boxed_slice());
    }

    pub fn union(&self, other: &Self) -> Box<Self> {
        let mut result = self.drop_rns_moduli_indices.iter().copied().chain(
            other.drop_rns_moduli_indices.iter().copied().filter(|i| !self.contains(*i)
        )).collect::<Box<[usize]>>();
        result.sort_unstable();
        return Self::from_unchecked(result);
    }

    pub fn empty() -> Box<Self> {
        Self::from_unchecked(Box::new([]))
    }
}
