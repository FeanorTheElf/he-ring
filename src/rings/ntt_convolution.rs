use std::alloc::{Allocator, Global};
use std::cell::{Ref, RefCell};
use std::cmp::{min, max};

use feanor_math::algorithms::convolution::{ConvolutionAlgorithm, STANDARD_CONVOLUTION};
use feanor_math::algorithms::convolution::PreparedConvolutionAlgorithm;
use feanor_math::algorithms::fft::cooley_tuckey::CooleyTuckeyFFT;
use feanor_math::homomorphism::Identity;
use feanor_math::primitive_int::StaticRing;
use feanor_math::assert_el_eq;
use feanor_math::ring::*;
use feanor_math::integer::*;
use feanor_math::rings::zn::*;
use feanor_math::seq::*;
use feanor_math::algorithms::fft::FFTAlgorithm;
use feanor_math::homomorphism::Homomorphism;
use feanor_math::rings::zn::zn_64::{Zn, ZnEl};

use super::convolution::FromRingCreateableConvolution;

pub struct NTTConv<R, A = Global>
    where R: RingStore + Clone,
        R::Type: ZnRing,
        A: Allocator + Clone
{
    ring: R,
    max_log2_n: usize,
    fft_algos: Vec<CooleyTuckeyFFT<R::Type, R::Type, Identity<R>>>,
    allocator: A
}

impl<R> NTTConv<R>
    where R: RingStore + Clone,
        R::Type: ZnRing
{
    pub fn new(ring: R, max_log2_n: usize) -> Self {
        Self::create(ring, max_log2_n)
    }
}

impl<R> FromRingCreateableConvolution<R> for NTTConv<R>
    where R: RingStore + Clone,
        R::Type: ZnRing
{
    fn create(ring: R, max_log2_n: usize) -> Self {
        Self::new_with(ring, max_log2_n, Global)
    }
}

impl<R, A> NTTConv<R, A>
    where R: RingStore + Clone,
        R::Type: ZnRing,
        A: Allocator + Clone
{
    pub fn new_with(ring: R, max_log2_n: usize, allocator: A) -> Self {
        assert!(max_log2_n <= ring.integer_ring().get_ring().abs_lowest_set_bit(&ring.integer_ring().sub_ref_fst(ring.modulus(), ring.integer_ring().one())).unwrap());
        Self {
            fft_algos: (0..=max_log2_n).map(|log2_n| CooleyTuckeyFFT::for_zn(ring.clone(), log2_n).unwrap()).collect(),
            ring: ring,
            allocator: allocator,
            max_log2_n: max_log2_n,
        }
    }

    pub fn max_supported_output_len(&self) -> usize {
        1 << self.max_log2_n
    }

    fn compute_convolution_base(&self, mut lhs: PreparedConvolutionOperand<R, A>, rhs: &PreparedConvolutionOperand<R, A>, out: &mut [El<R>]) {
        record_time!(GLOBAL_TIME_RECORDER, "NTTConv::compute_convolution_base", || {
            let log2_n = ZZ.abs_log2_ceil(&(lhs.data.len() as i64)).unwrap();
            assert_eq!(lhs.data.len(), 1 << log2_n);
            assert_eq!(rhs.data.len(), 1 << log2_n);
            assert!(lhs.len + rhs.len <= 1 << log2_n);
            for i in 0..(1 << log2_n) {
                self.ring.mul_assign_ref(&mut lhs.data[i], &rhs.data[i]);
            }
            self.get_fft(log2_n).unordered_inv_fft(&mut lhs.data[..], &self.ring);
            for i in 0..min(out.len(), 1 << log2_n) {
                self.ring.add_assign_ref(&mut out[i], &lhs.data[i]);
            }
        })
    }

    fn get_fft<'a>(&'a self, log2_n: usize) -> &'a CooleyTuckeyFFT<R::Type, R::Type, Identity<R>> {
        &self.fft_algos[log2_n]
    }

    fn clone_prepared_operand(&self, operand: &PreparedConvolutionOperand<R, A>) -> PreparedConvolutionOperand<R, A> {
        let mut result = Vec::with_capacity_in(operand.data.len(), self.allocator.clone());
        result.extend(operand.data.iter().map(|x| self.ring.clone_el(x)));
        return PreparedConvolutionOperand {
            len: operand.len,
            data: result
        };
    }
    
    fn prepare_convolution_base<V: VectorView<El<R>>>(&self, val: V, log2_n: usize) -> PreparedConvolutionOperand<R, A> {
        record_time!(GLOBAL_TIME_RECORDER, "NTTConv::prepare_convolution_base", || {
            let mut result = Vec::with_capacity_in(1 << log2_n, self.allocator.clone());
            result.extend(val.as_iter().map(|x| self.ring.clone_el(x)));
            result.resize_with(1 << log2_n, || self.ring.zero());
            let fft = self.get_fft(log2_n);
            fft.unordered_fft(&mut result[..], &self.ring);
            return PreparedConvolutionOperand {
                len: val.len(),
                data: result
            };
        })
    }
}

impl<R, A> ConvolutionAlgorithm<R::Type> for NTTConv<R, A>
    where R: RingStore + Clone,
        R::Type: ZnRing ,
        A: Allocator + Clone
{
    fn supports_ring<S: RingStore<Type = R::Type> + Copy>(&self, ring: S) -> bool {
        ring.get_ring() == self.ring.get_ring()
    }

    fn compute_convolution<S: RingStore<Type = R::Type> + Copy, V1: VectorView<<R::Type as RingBase>::Element>, V2: VectorView<<R::Type as RingBase>::Element>>(&self, lhs: V1, rhs: V2, dst: &mut [<R::Type as RingBase>::Element], ring: S) {
        assert!(ring.get_ring() == self.ring.get_ring());
        let log2_n = ZZ.abs_log2_ceil(&((lhs.len() + rhs.len()) as i64)).unwrap();
        let lhs_prep = self.prepare_convolution_base(lhs, log2_n);
        let rhs_prep = self.prepare_convolution_base(rhs, log2_n);
        self.compute_convolution_base(lhs_prep, &rhs_prep, dst);
    }
}

pub struct PreparedConvolutionOperand<R, A = Global>
    where R: RingStore + Clone,
        R::Type: ZnRing,
        A: Allocator + Clone
{
    len: usize,
    data: Vec<El<R>, A>
}

const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;

impl<R, A> PreparedConvolutionAlgorithm<R::Type> for NTTConv<R, A>
    where R: RingStore + Clone,
        R::Type: ZnRing ,
        A: Allocator + Clone
{
    type PreparedConvolutionOperand = PreparedConvolutionOperand<R, A>;

    fn prepare_convolution_operand<S: RingStore<Type = R::Type> + Copy, V: VectorView<El<R>>>(&self, val: V, ring: S) -> Self::PreparedConvolutionOperand {
        assert!(ring.get_ring() == self.ring.get_ring());
        let log2_n_in = ZZ.abs_log2_ceil(&(val.len() as i64)).unwrap();
        let log2_n_out = log2_n_in + 1;
        return self.prepare_convolution_base(val, log2_n_out);
    }

    fn compute_convolution_lhs_prepared<S: RingStore<Type = R::Type> + Copy, V: VectorView<El<R>>>(&self, lhs: &Self::PreparedConvolutionOperand, rhs: V, dst: &mut [El<R>], ring: S) {
        assert!(ring.get_ring() == self.ring.get_ring());
        let log2_n = ZZ.abs_log2_ceil(&((lhs.len + rhs.len()) as i64)).unwrap();
        if lhs.data.len() >= (1 << log2_n) {
            let log2_n = ZZ.abs_log2_ceil(&(lhs.data.len() as i64)).unwrap();
            assert!(lhs.data.len() == 1 << log2_n);
            self.compute_convolution_base(self.prepare_convolution_base(rhs, log2_n), lhs, dst);
        } else {
            self.compute_convolution_prepared(lhs, &self.prepare_convolution_base(rhs, log2_n), dst, ring)
        }
    }

    fn compute_convolution_prepared<S: RingStore<Type = R::Type> + Copy>(&self, lhs: &Self::PreparedConvolutionOperand, rhs: &Self::PreparedConvolutionOperand, dst: &mut [El<R>], ring: S) {
        assert!(ring.get_ring() == self.ring.get_ring());
        let log2_lhs = ZZ.abs_log2_ceil(&(lhs.data.len() as i64)).unwrap();
        assert_eq!(1 << log2_lhs, lhs.data.len());
        let log2_rhs = ZZ.abs_log2_ceil(&(rhs.data.len() as i64)).unwrap();
        assert_eq!(1 << log2_rhs, rhs.data.len());
        match log2_lhs.cmp(&log2_rhs) {
            std::cmp::Ordering::Equal => self.compute_convolution_base(self.clone_prepared_operand(lhs), rhs, dst),
            std::cmp::Ordering::Greater => self.compute_convolution_prepared(rhs, lhs, dst, ring),
            std::cmp::Ordering::Less => record_time!(GLOBAL_TIME_RECORDER, "NTTConv::compute_convolution_prepared::redo_fft", || {
                let mut lhs_new = Vec::with_capacity_in(lhs.data.len(), self.allocator.clone());
                lhs_new.extend(lhs.data.iter().map(|x| self.ring.clone_el(x)));
                self.get_fft(log2_lhs).unordered_inv_fft(&mut lhs_new[..], ring);
                lhs_new.resize_with(1 << log2_rhs, || ring.zero());
                self.get_fft(log2_rhs).unordered_fft(&mut lhs_new[..], ring);
                self.compute_convolution_base(PreparedConvolutionOperand { data: lhs_new, len: lhs.len }, rhs, dst);
            })
        }
    }
}

#[test]
fn test_convolution() {
    let ring = Zn::new(65537);
    let convolutor = NTTConv::new_with(ring, 16, Global);

    let check = |lhs: &[ZnEl], rhs: &[ZnEl], add: &[ZnEl]| {
        let mut expected = (0..(lhs.len() + rhs.len())).map(|i| if i < add.len() { add[i] } else { ring.zero() }).collect::<Vec<_>>();
        STANDARD_CONVOLUTION.compute_convolution(lhs, rhs, &mut expected, &ring);

        let mut actual1 = (0..(lhs.len() + rhs.len())).map(|i| if i < add.len() { add[i] } else { ring.zero() }).collect::<Vec<_>>();
        convolutor.compute_convolution(lhs, rhs, &mut actual1, &ring);
        for i in 0..(lhs.len() + rhs.len()) {
            assert_el_eq!(&ring, &expected[i], &actual1[i]);
        }
        
        let lhs_prepared = convolutor.prepare_convolution_operand(lhs, &ring);
        let rhs_prepared = convolutor.prepare_convolution_operand(rhs, &ring);

        let mut actual2 = (0..(lhs.len() + rhs.len())).map(|i| if i < add.len() { add[i] } else { ring.zero() }).collect::<Vec<_>>();
        convolutor.compute_convolution_lhs_prepared(&lhs_prepared, rhs, &mut actual2, &ring);
        for i in 0..(lhs.len() + rhs.len()) {
            assert_el_eq!(&ring, &expected[i], &actual2[i]);
        }
        
        let mut actual3 = (0..(lhs.len() + rhs.len())).map(|i| if i < add.len() { add[i] } else { ring.zero() }).collect::<Vec<_>>();
        convolutor.compute_convolution_rhs_prepared(lhs, &rhs_prepared, &mut actual3, &ring);
        for i in 0..(lhs.len() + rhs.len()) {
            assert_el_eq!(&ring, &expected[i], &actual3[i]);
        }
        
        let mut actual4 = (0..(lhs.len() + rhs.len())).map(|i| if i < add.len() { add[i] } else { ring.zero() }).collect::<Vec<_>>();
        convolutor.compute_convolution_prepared(&lhs_prepared, &rhs_prepared, &mut actual4, &ring);
        for i in 0..(lhs.len() + rhs.len()) {
            assert_el_eq!(&ring, &expected[i], &actual4[i]);
        }
    };

    for lhs_len in [1, 2, 3, 4, 7, 8, 9] {
        for rhs_len in [1, 5, 8, 16, 17] {
            let lhs = (0..lhs_len).map(|i| ring.int_hom().map(i)).collect::<Vec<_>>();
            let rhs = (0..rhs_len).map(|i| ring.int_hom().map(16 * i)).collect::<Vec<_>>();
            let add = (0..(lhs_len + rhs_len)).map(|i| ring.int_hom().map(32768 * i)).collect::<Vec<_>>();
            check(&lhs, &rhs, &add);
        }
    }
}