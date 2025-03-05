use core::f64;
use std::cell::RefCell;

use feanor_math::homomorphism::Homomorphism;
use feanor_math::primitive_int::*;
use feanor_math::ring::*;

use crate::circuit::Coefficient;
use crate::circuit::PlaintextCircuit;
use crate::cyclotomic::CyclotomicGaloisGroupEl;
use crate::gadget_product::digits::*;
use crate::ZZi64;

use super::*;

///
/// A [`Ciphertext`] which additionally stores w.r.t. which ciphertext modulus it is defined,
/// and which noise level (as measured by some [`BGVModswitchStrategy`]) it is estimated to have.
///
pub struct ModulusAwareCiphertext<Params: BGVParams, Strategy: ?Sized + BGVModswitchStrategy<Params>> {
    /// The stored raw ciphertext
    pub data: Ciphertext<Params>,
    /// The indices of those RNS components w.r.t. a "master RNS base" (specified by the context)
    /// that are not used for this ciphertext; in other words, the ciphertext modulus of this ciphertext
    /// is the product of all RNS factors of the master RNS base that are not mentioned in this list
    pub dropped_rns_factor_indices: Box<RNSFactorIndexList>,
    /// Additional information required by the modulus-switching strategy
    pub info: Strategy::CiphertextInfo
}

pub trait BGVNoiseEstimator<Params: BGVParams> {

    ///
    /// An estimate of the size and distribution of the critical quantity
    /// `c0 + c1 s = m + t e`. The only requirement is that the noise estimator
    /// can derive an estimate about its infinity norm via
    /// [`BGVNoiseEstimator::estimate_log2_relative_noise_level`], but estimators are free
    /// to store additional data to get more precise estimates on the noise growth
    /// of operations.
    ///
    type CriticalQuantityLevel;

    ///
    /// Should return an estimate of
    /// ```text
    ///   log2( | c0 + c1 * s |_inf / q )
    /// ```
    ///
    fn estimate_log2_relative_noise_level(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, noise: &Self::CriticalQuantityLevel) -> f64;

    fn enc_sym_zero(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>) -> Self::CriticalQuantityLevel;

    fn transparent_zero(&self) -> Self::CriticalQuantityLevel;

    fn hom_add_plain(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, m: &El<PlaintextRing<Params>>, ct: &Self::CriticalQuantityLevel, implicit_scale: El<Zn>) -> Self::CriticalQuantityLevel;

    fn enc_sym(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, m: &El<PlaintextRing<Params>>) -> Self::CriticalQuantityLevel {
        self.hom_add_plain(P, C, m, &self.enc_sym_zero(P, C), P.base_ring().one())
    }

    fn hom_mul_plain(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, m: &El<PlaintextRing<Params>>, ct: &Self::CriticalQuantityLevel, implicit_scale: El<Zn>) -> Self::CriticalQuantityLevel;

    fn hom_mul_plain_i64(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, m: i64, ct: &Self::CriticalQuantityLevel, implicit_scale: El<Zn>) -> Self::CriticalQuantityLevel;

    fn key_switch(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, ct: &Self::CriticalQuantityLevel, switch_key: &RNSGadgetVectorDigitIndices) -> Self::CriticalQuantityLevel;

    fn hom_mul(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, lhs: &Self::CriticalQuantityLevel, rhs: &Self::CriticalQuantityLevel, rk_digits: &RNSGadgetVectorDigitIndices) -> Self::CriticalQuantityLevel;

    fn hom_add(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, lhs: &Self::CriticalQuantityLevel, lhs_implicit_scale: Option<El<Zn>>, rhs: &Self::CriticalQuantityLevel, rhs_implicit_scale: Option<El<Zn>>) -> Self::CriticalQuantityLevel;

    fn hom_galois(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, ct: &Self::CriticalQuantityLevel, _g: CyclotomicGaloisGroupEl, gk: &RNSGadgetVectorDigitIndices) -> Self::CriticalQuantityLevel {
        self.key_switch(P, C, ct, gk)
    }

    fn mod_switch_down(&self, P: &PlaintextRing<Params>, Cnew: &CiphertextRing<Params>, Cold: &CiphertextRing<Params>, drop_moduli: &RNSFactorIndexList, ct: &Self::CriticalQuantityLevel) -> Self::CriticalQuantityLevel;

    fn change_plaintext_modulus(Pnew: &PlaintextRing<Params>, Pold: &PlaintextRing<Params>, C: &CiphertextRing<Params>, ct: &Self::CriticalQuantityLevel) -> Self::CriticalQuantityLevel;

    fn clone_critical_quantity_level(&self, val: &Self::CriticalQuantityLevel) -> Self::CriticalQuantityLevel;
}

fn log2_can_norm_sk_estimate<Params: BGVParams>(C: &CiphertextRing<Params>) -> f64 {
    (C.rank() as f64).log2()
}

///
/// A [`BGVNoiseEstimator`] that uses some very simple formulas to estimate the noise
/// growth of BGV operations. This is WIP and very likely to be replaced later by
/// a better and more rigorous estimator.
///
pub struct NaiveBGVNoiseEstimator;

impl<Params: BGVParams> BGVNoiseEstimator<Params> for NaiveBGVNoiseEstimator {

    /// We store `log2(| c0 + c1 s |_can / q)`; this is hopefully `< 0`
    type CriticalQuantityLevel = f64;

    fn estimate_log2_relative_noise_level(&self, _P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, noise: &Self::CriticalQuantityLevel) -> f64 {
        // we subtract `(C.rank() as f64).log2()`, since that should be about the difference between `l_inf` and canonical norm
        *noise - (C.rank() as f64).log2()
    }

    fn enc_sym_zero(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>) -> Self::CriticalQuantityLevel {
        let result = (*P.base_ring().modulus() as f64).log2() + log2_can_norm_sk_estimate::<Params>(C) - BigIntRing::RING.abs_log2_floor(C.base_ring().modulus()).unwrap() as f64;
        assert!(!result.is_nan());
        return result;
    }

    fn hom_add(&self, P: &PlaintextRing<Params>, _C: &CiphertextRing<Params>, lhs: &Self::CriticalQuantityLevel, lhs_implicit_scale: Option<El<Zn>>, rhs: &Self::CriticalQuantityLevel, rhs_implicit_scale: Option<El<Zn>>) -> Self::CriticalQuantityLevel {
        if lhs_implicit_scale.is_none() || rhs_implicit_scale.is_none() {
            let result = f64::max(*lhs, *rhs) + (*P.base_ring().modulus() as f64).log2() / 2.;
            assert!(!result.is_nan());
            return result;
        } else {
            let Zt = P.base_ring();
            let (a, b) = equalize_implicit_scale(Zt, Zt.checked_div(&lhs_implicit_scale.unwrap(), &rhs_implicit_scale.unwrap()).unwrap());
            debug_assert!(Zt.is_unit(&Zt.coerce(&ZZi64, a)));
            debug_assert!(Zt.is_unit(&Zt.coerce(&ZZi64, b)));
            let result = f64::max((b as f64).abs().log2() + *lhs, (a as f64).abs().log2() + *rhs);
            assert!(!result.is_nan());
            return result;
        }
    }

    fn hom_add_plain(&self, _P: &PlaintextRing<Params>, _C: &CiphertextRing<Params>, _m: &El<PlaintextRing<Params>>, ct: &Self::CriticalQuantityLevel, _implicit_scale: El<Zn>) -> Self::CriticalQuantityLevel {
        let result = *ct;
        assert!(!result.is_nan());
        return result;
    }

    fn hom_mul_plain(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, m: &El<PlaintextRing<Params>>, ct: &Self::CriticalQuantityLevel, _implicit_scale: El<Zn>) -> Self::CriticalQuantityLevel {
        let result = *ct + (P.wrt_canonical_basis(m).iter().map(|c| P.base_ring().smallest_lift(c).abs()).max().unwrap() as f64 * C.rank() as f64).log2();
        assert!(!result.is_nan());
        return result;
    }

    fn hom_mul_plain_i64(&self, _P: &PlaintextRing<Params>, _C: &CiphertextRing<Params>, m: i64, ct: &Self::CriticalQuantityLevel, _implicit_scale: El<Zn>) -> Self::CriticalQuantityLevel {
        let result = *ct + (m as f64).abs().log2();
        assert!(!result.is_nan());
        return result;
    }

    fn transparent_zero(&self) -> Self::CriticalQuantityLevel {
        let result = -f64::INFINITY;
        assert!(!result.is_nan());
        return result;
    }

    fn mod_switch_down(&self, P: &PlaintextRing<Params>, Cnew: &CiphertextRing<Params>, _Cold: &CiphertextRing<Params>, _drop_moduli: &RNSFactorIndexList, ct: &Self::CriticalQuantityLevel) -> Self::CriticalQuantityLevel {
        let result = f64::max(
            *ct,
            (*P.base_ring().modulus() as f64).log2() + log2_can_norm_sk_estimate::<Params>(Cnew) - BigIntRing::RING.abs_log2_ceil(Cnew.base_ring().modulus()).unwrap() as f64
        );
        assert!(!result.is_nan());
        return result;
    }

    fn hom_mul(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, lhs: &Self::CriticalQuantityLevel, rhs: &Self::CriticalQuantityLevel, rk_digits: &RNSGadgetVectorDigitIndices) -> Self::CriticalQuantityLevel {
        let result = <Self as BGVNoiseEstimator<Params>>::key_switch(self, P, C, &(*lhs + *rhs + BigIntRing::RING.abs_log2_ceil(C.base_ring().modulus()).unwrap() as f64), rk_digits);
        assert!(!result.is_nan());
        return result;
    }

    fn key_switch(&self, _P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, ct: &Self::CriticalQuantityLevel, switch_key: &RNSGadgetVectorDigitIndices) -> Self::CriticalQuantityLevel {
        let log2_largest_digit = switch_key.iter().map(|digit| digit.iter().map(|i| *C.base_ring().at(i).modulus() as f64).map(f64::log2).sum::<f64>()).max_by(f64::total_cmp).unwrap();
        let result = f64::max(
            *ct,
            log2_largest_digit + (C.rank() as f64).log2() * 2. - BigIntRing::RING.abs_log2_ceil(C.base_ring().modulus()).unwrap() as f64
        );
        assert!(!result.is_nan());
        return result;
    }

    fn change_plaintext_modulus(_Pnew: &PlaintextRing<Params>, _Pold: &PlaintextRing<Params>, _C: &CiphertextRing<Params>, ct: &Self::CriticalQuantityLevel) -> Self::CriticalQuantityLevel {
        let result = *ct;
        assert!(!result.is_nan());
        return result;
    }

    fn clone_critical_quantity_level(&self, val: &Self::CriticalQuantityLevel) -> Self::CriticalQuantityLevel {
        *val
    }
}

///
/// Noise estimator that always returns 0 as estimated noise budget.
///
/// Its only use is probably to have a default value in places where a
/// noise estimator is required but never used, as well as to implement
/// [`DefaultModswitchStrategy::never_modswitch()`].
///
pub struct AlwaysZeroNoiseEstimator;

impl<Params: BGVParams> BGVNoiseEstimator<Params> for AlwaysZeroNoiseEstimator {

    type CriticalQuantityLevel = ();

    fn estimate_log2_relative_noise_level(&self, _P: &PlaintextRing<Params>, _C: &CiphertextRing<Params>, _noise: &Self::CriticalQuantityLevel) -> f64 {
        0.
    }

    fn enc_sym_zero(&self, _P: &PlaintextRing<Params>, _C: &CiphertextRing<Params>) -> Self::CriticalQuantityLevel {}
    fn hom_add(&self, _P: &PlaintextRing<Params>, _C: &CiphertextRing<Params>, _lhs: &Self::CriticalQuantityLevel, _lhs_implicit_scale: Option<El<Zn>>, _rhs: &Self::CriticalQuantityLevel, _rhs_implicit_scale: Option<El<Zn>>) -> Self::CriticalQuantityLevel {}
    fn hom_add_plain(&self, _P: &PlaintextRing<Params>, _C: &CiphertextRing<Params>, _m: &El<PlaintextRing<Params>>, _ct: &Self::CriticalQuantityLevel, _implicit_scale: El<Zn>) -> Self::CriticalQuantityLevel {}
    fn hom_galois(&self, _P: &PlaintextRing<Params>, _C: &CiphertextRing<Params>, _ct: &Self::CriticalQuantityLevel, _g: CyclotomicGaloisGroupEl, _gk: &RNSGadgetVectorDigitIndices) -> Self::CriticalQuantityLevel {}
    fn hom_mul(&self, _P: &PlaintextRing<Params>, _C: &CiphertextRing<Params>, _lhs: &Self::CriticalQuantityLevel, _rhs: &Self::CriticalQuantityLevel, _rk_digits: &RNSGadgetVectorDigitIndices) -> Self::CriticalQuantityLevel {}
    fn hom_mul_plain(&self, _P: &PlaintextRing<Params>, _C: &CiphertextRing<Params>, _m: &El<PlaintextRing<Params>>, _ct: &Self::CriticalQuantityLevel, _implicit_scale: El<Zn>) -> Self::CriticalQuantityLevel {}
    fn hom_mul_plain_i64(&self, _P: &PlaintextRing<Params>, _C: &CiphertextRing<Params>, _m: i64, _ct: &Self::CriticalQuantityLevel, _implicit_scale: El<Zn>) -> Self::CriticalQuantityLevel {}
    fn key_switch(&self, _P: &PlaintextRing<Params>, _C: &CiphertextRing<Params>, _ct: &Self::CriticalQuantityLevel, _switch_key: &RNSGadgetVectorDigitIndices) -> Self::CriticalQuantityLevel {}
    fn mod_switch_down(&self, _P: &PlaintextRing<Params>, _Cnew: &CiphertextRing<Params>, _Cold: &CiphertextRing<Params>, _drop_moduli: &RNSFactorIndexList, _ct: &Self::CriticalQuantityLevel) -> Self::CriticalQuantityLevel {}
    fn transparent_zero(&self) -> Self::CriticalQuantityLevel {}
    fn change_plaintext_modulus(_Pnew: &PlaintextRing<Params>, _Pold: &PlaintextRing<Params>, _C: &CiphertextRing<Params>, _ct: &Self::CriticalQuantityLevel) -> Self::CriticalQuantityLevel {}
    fn clone_critical_quantity_level(&self, _val: &Self::CriticalQuantityLevel) -> Self::CriticalQuantityLevel {}
}

///
/// Trait for different modulus-switching strategies in BGV, currently WIP.
///
/// Basically, a [`BGVModswitchStrategy`] should be able to determine when (and
/// how) to modulus-switch during the evaluation of an arithmetic circuit.
/// The most powerful way to do this is by delegating the evaluation of the
/// circuit completely to the [`BGVModswitchStrategy`], which is our current
/// approach.
///
pub trait BGVModswitchStrategy<Params: BGVParams> {

    type CiphertextInfo;

    ///
    /// Evaluates the given circuit homomorphically on the given encrypted inputs.
    /// This includes performing modulus-switches at suitable times.
    ///
    /// The parameters are as follows:
    ///  - `circuit` is the circuit to evaluate, with integer constants
    ///  - `P` is the plaintext ring w.r.t. which the inputs are encrypted; this function does not
    ///    support mixing different plaintext moduli
    ///  - `C_master` is the ciphertext ring with the largest relevant RNS base, i.e. its RNS
    ///    base should contain all RNS factors that are referenced by any ciphertext, and may
    ///    have additional unused RNS factors
    ///  - `inputs` contains all inputs to the circuit, i.e. must be of the same length as the
    ///    circuit has input wires. Each entry should be of the form `(drop_rns_factors, info, ctxt)`
    ///    where `ctxt` is the ciphertext w.r.t. the RNS base that contains all RNS factors of `C_master`
    ///    except those mentioned in `drop_rns_fctors`, and `info` should store the additional information
    ///    associated to the ciphertext that is required to determine modulus-switching times.
    ///  - `rk` should be the relinearization key w.r.t. `C_master`, can be `None` if the circuit
    ///    contains no multiplication gates.
    ///  - `gks` should contain all Galois keys used by the circuit (may also contain unused ones);
    ///    if the circuit has no Galois gates, this may be an empty slice
    ///
    /// Note that the [`BGVModswitchStrategy::CiphertextInfo`]s currently cannot be created using
    /// functions of the trait, but only via functions on the concrete implementation of
    /// [`BGVModswitchStrategy`].
    ///
    fn evaluate_circuit_int(
        &self,
        circuit: &PlaintextCircuit<StaticRingBase<i64>>,
        P: &PlaintextRing<Params>,
        C_master: &CiphertextRing<Params>,
        inputs: &[ModulusAwareCiphertext<Params, Self>],
        rk: Option<&RelinKey<Params>>,
        gks: &[(CyclotomicGaloisGroupEl, KeySwitchKey<Params>)],
        key_switches: &mut usize,
        debug_sk: Option<&SecretKey<Params>>
    ) -> Vec<ModulusAwareCiphertext<Params, Self>>;

    ///
    /// Evaluates the given circuit homomorphically on the given encrypted inputs.
    /// This includes performing modulus-switches at suitable times.
    ///
    /// For a detailed description, see [`BGVModswitchStrategy::evaluate_circuit_int()`], the
    /// only difference is that for this function, circuit constants are arbitrary plaintext ring
    /// elements.
    ///
    fn evaluate_circuit_plaintext(
        &self,
        circuit: &PlaintextCircuit<<PlaintextRing<Params> as RingStore>::Type>,
        P: &PlaintextRing<Params>,
        C_master: &CiphertextRing<Params>,
        inputs: &[ModulusAwareCiphertext<Params, Self>],
        rk: Option<&RelinKey<Params>>,
        gks: &[(CyclotomicGaloisGroupEl, KeySwitchKey<Params>)],
        key_switches: &mut usize,
        debug_sk: Option<&SecretKey<Params>>
    ) -> Vec<ModulusAwareCiphertext<Params, Self>>;

    fn info_for_fresh_encryption(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>) -> Self::CiphertextInfo;

    fn clone_info(&self, info: &Self::CiphertextInfo) -> Self::CiphertextInfo;
}

///
/// Default modulus-switch strategy for BGV, which performs a certain number of modulus-switches
/// before each multiplication. Currently WIP.
///
/// The general strategy is as follows:
///  - only mod-switch before multiplications
///  - never introduce new RNS factors, only remove current ones
///  - use the provided [`BGVNoiseEstimator`] to determine when and by how much
///    we should reduce the ciphertext modulus
///
/// These points lead to a relatively simple and generally well-performing modulus switching strategy.
/// However, there may be situations where deviating from 1. could lead to a lower number of mod-switches
/// (and thus better performance), and deviating from 2. could be used for a finer-tuned mod-switching,
/// and thus less noise growth.
///
pub struct DefaultModswitchStrategy<Params: BGVParams, N: BGVNoiseEstimator<Params>, const LOG: bool> {
    params: PhantomData<Params>,
    noise_estimator: N
}

impl<Params: BGVParams> DefaultModswitchStrategy<Params, AlwaysZeroNoiseEstimator, false> {

    ///
    /// Create a [`DefaultModswitchStrategy`] that never performs modulus switching,
    /// except when necessary because operands are defined modulo different RNS bases.
    ///
    /// Using this is not recommended, except for linear circuits, or circuits with
    /// very low multiplicative depth.
    ///
    pub fn never_modswitch() -> Self {
        Self {
            params: PhantomData,
            noise_estimator: AlwaysZeroNoiseEstimator
        }
    }
}

///
/// Used internally when evaluating a circuit, since we want to store plaintexts
/// as plaintexts as long as possible - or rather until we know w.r.t. which RNS
/// base we should convert them into a ciphertext ring element
/// 
enum PlainOrCiphertext<'a, Params: BGVParams, Strategy: BGVModswitchStrategy<Params>, R: ?Sized + RingBase> {
    Plaintext(Coefficient<R>),
    PlaintextRef(&'a Coefficient<R>),
    CiphertextRef(&'a ModulusAwareCiphertext<Params, Strategy>),
    Ciphertext(ModulusAwareCiphertext<Params, Strategy>)
}

impl<'a, Params: BGVParams, Strategy: BGVModswitchStrategy<Params>, R: ?Sized + RingBase> PlainOrCiphertext<'a, Params, Strategy, R> {

    fn as_ciphertext_ref<'b>(&'b self) -> Result<&'b ModulusAwareCiphertext<Params, Strategy>, &'b Coefficient<R>> {
        match self {
            PlainOrCiphertext::Plaintext(x) => Err(x),
            PlainOrCiphertext::PlaintextRef(x) => Err(x),
            PlainOrCiphertext::Ciphertext(x) => Ok(x),
            PlainOrCiphertext::CiphertextRef(x) => Ok(x)
        }
    }

    fn as_ciphertext(self, P: &PlaintextRing<Params>, C_master: &CiphertextRing<Params>, ring: RingRef<R>, strategy: &Strategy) -> Result<(CiphertextRing<Params>, ModulusAwareCiphertext<Params, Strategy>), Coefficient<R>> {
        match self {
            PlainOrCiphertext::Plaintext(x) => Err(x),
            PlainOrCiphertext::PlaintextRef(x) => Err(x.clone(ring)),
            PlainOrCiphertext::CiphertextRef(x) => {
                let Cx = Params::mod_switch_down_ciphertext_ring(C_master, &x.dropped_rns_factor_indices);
                let x = ModulusAwareCiphertext {
                    data: Params::clone_ct(P, &Cx, &x.data),
                    dropped_rns_factor_indices: x.dropped_rns_factor_indices.clone(),
                    info: strategy.clone_info(&x.info)
                };
                Ok((Cx, x))
            },
            PlainOrCiphertext::Ciphertext(x) => {
                let Cx = Params::mod_switch_down_ciphertext_ring(C_master, &x.dropped_rns_factor_indices);
                Ok((Cx, x))
            }
        }
    }
}

impl<Params: BGVParams, N: BGVNoiseEstimator<Params>, const LOG: bool> DefaultModswitchStrategy<Params, N, LOG> {

    pub fn new(noise_estimator: N) -> Self {
        Self {
            params: PhantomData,
            noise_estimator: noise_estimator
        }
    }

    pub fn from_noise_level(&self, noise_level: N::CriticalQuantityLevel) -> <Self as BGVModswitchStrategy<Params>>::CiphertextInfo {
        noise_level
    }

    #[instrument(skip_all)]
    fn compute_optimal_mul_modswitch(
        &self,
        P: &PlaintextRing<Params>,
        C_master: &CiphertextRing<Params>,
        noise_x: &N::CriticalQuantityLevel,
        drop_x: &RNSFactorIndexList,
        noise_y: &N::CriticalQuantityLevel,
        drop_y: &RNSFactorIndexList,
        rk_digits: &RNSGadgetVectorDigitIndices
    ) -> Box<RNSFactorIndexList> {
        let Cx = Params::mod_switch_down_ciphertext_ring(C_master, drop_x);
        let Cy = Params::mod_switch_down_ciphertext_ring(C_master, drop_y);
        let outer_drop = drop_x.union(&drop_y);
        let after_first_drop = rk_digits.remove_indices(&outer_drop);
        let compute_result_noise = |num_to_drop: usize| {
            let inner_drop = recommended_rns_factors_to_drop(&after_first_drop, num_to_drop);
            let total_drop = inner_drop.pullback(&outer_drop);
            let C_target = Params::mod_switch_down_ciphertext_ring(C_master, &total_drop);
            let rk_after_total_drop = rk_digits.remove_indices(&total_drop);

            let expected_noise = self.noise_estimator.estimate_log2_relative_noise_level(
                P,
                &C_target,
                &self.noise_estimator.hom_mul(
                    P,
                    &C_target,
                    &self.noise_estimator.mod_switch_down(&P, &C_target, &Cx, drop_x, noise_x),
                    &self.noise_estimator.mod_switch_down(&P, &C_target, &Cy, drop_y, noise_y),
                    &rk_after_total_drop
                )
            );
            return (total_drop, expected_noise);
        };
        return (0..(C_master.base_ring().len() - outer_drop.len())).map(compute_result_noise).min_by(|(_, l), (_, r)| f64::total_cmp(l, r)).unwrap().0;
    }

    fn log_modulus_switch(
        &self,
        context: &str, 
        P: &PlaintextRing<Params>,
        C_master: &CiphertextRing<Params>,
        Cx: &CiphertextRing<Params>, 
        Cy: &CiphertextRing<Params>, 
        Ctarget: &CiphertextRing<Params>, 
        Cx_dropped: &RNSFactorIndexList, 
        Cy_dropped: &RNSFactorIndexList, 
        drop_x: &RNSFactorIndexList,
        drop_y: &RNSFactorIndexList, 
        total_drop: &RNSFactorIndexList, 
        x_data: &Option<Ciphertext<Params>>, 
        y_data: &Option<Ciphertext<Params>>, 
        res_data: &Ciphertext<Params>, 
        x_info: &N::CriticalQuantityLevel, 
        y_info: &N::CriticalQuantityLevel, 
        res_info: &N::CriticalQuantityLevel,
        debug_sk: Option<&SecretKey<Params>>
    ) {
        if LOG {
            println!("{}: Dropping {} from LHS with estimated noise {} and {} from RHS with estimated noise {}, estimated output noise is {}",
                context,
                drop_x,
                self.noise_estimator.estimate_log2_relative_noise_level(P, &Cx, &x_info),
                drop_y,
                self.noise_estimator.estimate_log2_relative_noise_level(P, &Cy, &y_info),
                self.noise_estimator.estimate_log2_relative_noise_level(P, &Ctarget, &res_info)
            );
            if let Some(sk) = debug_sk {
                let sk_x = Params::mod_switch_down_sk(&Cx, &C_master, &Cx_dropped, sk);
                let sk_y = Params::mod_switch_down_sk(&Cy, &C_master, &Cy_dropped, sk);
                let sk_res = Params::mod_switch_down_sk(&Ctarget, &C_master, &total_drop, sk);
                println!("  actual input noise is {}/{} resp. {}/{} and output noise is {}/{}",
                    Params::noise_budget(&P, &Cx, x_data.as_ref().unwrap(), &sk_x),
                    ZZbig.abs_log2_ceil(Cx.base_ring().modulus()).unwrap(),
                    Params::noise_budget(&P, &Cy, y_data.as_ref().unwrap(), &sk_y),
                    ZZbig.abs_log2_ceil(Cy.base_ring().modulus()).unwrap(),
                    Params::noise_budget(&P, &Ctarget, &res_data, &sk_res),
                    ZZbig.abs_log2_ceil(Ctarget.base_ring().modulus()).unwrap(),
                );
            }
        }
    }

    fn add_prod<'a, R, F1, F2>(
        &self,
        P: &PlaintextRing<Params>,
        C_master: &CiphertextRing<Params>,
        x: PlainOrCiphertext<'a, Params, Self, R>,
        c: &Coefficient<R>,
        y: &PlainOrCiphertext<'a, Params, Self, R>,
        ring: RingRef<R>,
        hom_add_plain: &mut F1,
        hom_mul_plain: &RefCell<F2>,
        debug_sk: Option<&SecretKey<Params>>
    ) -> PlainOrCiphertext<'a, Params, Self, R>
        where R: ?Sized + RingBase,
            F1: FnMut(&PlaintextRing<Params>, &CiphertextRing<Params>, &R::Element, Ciphertext<Params>, &<Self as BGVModswitchStrategy<Params>>::CiphertextInfo) -> (<Self as BGVModswitchStrategy<Params>>::CiphertextInfo, Ciphertext<Params>),
            F2: FnMut(&PlaintextRing<Params>, &CiphertextRing<Params>, &R::Element, Ciphertext<Params>, &<Self as BGVModswitchStrategy<Params>>::CiphertextInfo) -> (<Self as BGVModswitchStrategy<Params>>::CiphertextInfo, Ciphertext<Params>),
    {
        match (x.as_ciphertext(P, C_master, ring, self), c, y.as_ciphertext_ref()) {
            (Ok((_Cx, x)), Coefficient::Zero, _) => PlainOrCiphertext::Ciphertext(x),
            (Err(x), Coefficient::Zero, _) => PlainOrCiphertext::Plaintext(x),
            (Err(x), c, Err(y)) => PlainOrCiphertext::Plaintext(x.clone(ring).add(c.clone(ring).mul(y.clone(ring), ring), ring)),
            (Err(x), c, Ok(y)) => PlainOrCiphertext::Ciphertext({
                let total_drop = y.dropped_rns_factor_indices.clone();
                let Cy = Params::mod_switch_down_ciphertext_ring(C_master, &total_drop);
                let Ctarget = &Cy;

                let (y_scaled_info, y_scaled_data) = match c {
                    Coefficient::Zero => unreachable!(),
                    Coefficient::One => (self.clone_info(&y.info), Params::clone_ct(P, &Cy, &y.data)),
                    Coefficient::NegOne => (self.clone_info(&y.info), Params::hom_mul_plain_i64(P, &Ctarget, -1, Params::clone_ct(P, &Cy, &y.data))),
                    Coefficient::Integer(c) => (
                        self.noise_estimator.hom_mul_plain_i64(P, &Ctarget, *c as i64, &y.info, y.data.implicit_scale),
                        Params::hom_mul_plain_i64(P, &Ctarget, *c as i64, Params::clone_ct(P, &Cy, &y.data))
                    ),
                    Coefficient::Other(m) => (hom_mul_plain.borrow_mut())(P, &Cy, m, Params::clone_ct(P, &Cy, &y.data), &y.info)
                };
                let (res_info, res_data) = if let Coefficient::Zero = x {
                    (y_scaled_info, y_scaled_data)
                } else {
                    hom_add_plain(P, &Ctarget, &x.clone(ring).to_ring_el(ring), y_scaled_data, &y_scaled_info)
                };
                ModulusAwareCiphertext {
                    data: res_data,
                    info: res_info,
                    dropped_rns_factor_indices: total_drop
                }
            }),
            (Ok((Cx, x)), c, Err(y)) => PlainOrCiphertext::Ciphertext({
                let total_drop = x.dropped_rns_factor_indices.clone();
                let Ctarget = &Cx;
                let (result_info, result_data) = hom_add_plain(P, &Ctarget, &c.mul_to(y.clone(ring).to_ring_el(ring), ring), x.data, &x.info);
                ModulusAwareCiphertext {
                    data: result_data,
                    info: result_info,
                    dropped_rns_factor_indices: total_drop
                }
            }),
            // the most complex case, when both `x` and `y` are ciphertexts; in this case, it can happen that
            // they are defined w.r.t. different RNS bases, so we might have to perform a modulus-switch
            (Ok((Cx, x)), c, Ok(y)) => PlainOrCiphertext::Ciphertext({
                let total_drop = x.dropped_rns_factor_indices.union(&y.dropped_rns_factor_indices);
                let Ctarget = Params::mod_switch_down_ciphertext_ring(C_master, &total_drop);
                
                let x_data_copy = if debug_sk.is_some() { Some(Params::clone_ct(&P, &Cx, &x.data)) } else { None };
                let drop_x = total_drop.pushforward(&x.dropped_rns_factor_indices);
                let x_info = self.noise_estimator.mod_switch_down(&P, &Ctarget, &Cx, &drop_x, &x.info);
                let x_data = Params::mod_switch_down(P, &Ctarget, &Cx, &drop_x, x.data);

                let Cy = Params::mod_switch_down_ciphertext_ring(C_master, &y.dropped_rns_factor_indices);
                let y_data_copy = if debug_sk.is_some() { Some(Params::clone_ct(&P, &Cy, &y.data)) } else { None };
                let drop_y = total_drop.pushforward(&y.dropped_rns_factor_indices);
                let y_info = self.noise_estimator.mod_switch_down(&P, &Ctarget, &Cy, &drop_y, &y.info);
                let y_data = Params::mod_switch_down(P, &Ctarget, &Cy, &drop_y, Params::clone_ct(P, &Cy, &y.data));

                let (y_scaled_info, y_scaled_data) = match c {
                    Coefficient::Zero => unreachable!(),
                    Coefficient::One => (y_info, y_data),
                    Coefficient::NegOne => (y_info, Params::hom_mul_plain_i64(P, &Ctarget, -1, y_data)),
                    Coefficient::Integer(c) => (
                        self.noise_estimator.hom_mul_plain_i64(P, &Ctarget, *c as i64, &y_info, y_data.implicit_scale),
                        Params::hom_mul_plain_i64(P, &Ctarget, *c as i64, y_data)
                    ),
                    Coefficient::Other(m) => (hom_mul_plain.borrow_mut())(P, &Cy, m, y_data, &y_info)
                };
                let res_info = self.noise_estimator.hom_add(P, C_master, &x_info, Some(x_data.implicit_scale), &y_scaled_info, Some(y_scaled_data.implicit_scale));
                let res_data = Params::hom_add(P, &Ctarget, x_data, y_scaled_data);

                self.log_modulus_switch("HomAdd", P, C_master, &Cx, &Cy, &Ctarget, &x.dropped_rns_factor_indices, &y.dropped_rns_factor_indices, &drop_x, &drop_y, &total_drop, &x_data_copy, &y_data_copy, &res_data, &x.info, &y.info, &res_info, debug_sk);
                ModulusAwareCiphertext {
                    data: res_data,
                    info: res_info,
                    dropped_rns_factor_indices: total_drop
                }
            })
        }
    }

    fn mul<'a, R, F2>(
        &self,
        P: &PlaintextRing<Params>,
        C_master: &CiphertextRing<Params>,
        x: PlainOrCiphertext<'a, Params, Self, R>,
        y: PlainOrCiphertext<'a, Params, Self, R>,
        ring: RingRef<R>,
        hom_mul_plain: &RefCell<F2>,
        rk: Option<&RelinKey<Params>>,
        key_switches: &RefCell<&mut usize>,
        debug_sk: Option<&SecretKey<Params>>
    ) -> PlainOrCiphertext<'a, Params, Self, R>
        where R: ?Sized + RingBase,
            F2: FnMut(&PlaintextRing<Params>, &CiphertextRing<Params>, &R::Element, Ciphertext<Params>, &<Self as BGVModswitchStrategy<Params>>::CiphertextInfo) -> (<Self as BGVModswitchStrategy<Params>>::CiphertextInfo, Ciphertext<Params>),
    {
        match (x.as_ciphertext(P, C_master, ring, self), y.as_ciphertext(P, C_master, ring, self)) {
            (Err(x), Err(y)) => PlainOrCiphertext::Plaintext(x.add(y, ring)),
            // possibly swap `x` and `y` here so that we can handle both asymmetric cases in one statement
            (Ok((Cx, x)), Err(y)) | (Err(y), Ok((Cx, x))) => PlainOrCiphertext::Ciphertext({
                let total_drop = x.dropped_rns_factor_indices.clone();
                let Ctarget = &Cx;
                
                let (res_info, res_data) = match y {
                    Coefficient::Zero => unreachable!(),
                    Coefficient::One => (x.info, x.data),
                    Coefficient::NegOne => (x.info, Params::hom_mul_plain_i64(P, &Ctarget, -1, x.data)),
                    Coefficient::Integer(c) => (
                        self.noise_estimator.hom_mul_plain_i64(P, &Ctarget, c as i64, &x.info, x.data.implicit_scale),
                        Params::hom_mul_plain_i64(P, &Ctarget, c as i64, x.data)
                    ),
                    Coefficient::Other(m) => (hom_mul_plain.borrow_mut())(P, &Ctarget, &m, x.data, &x.info)
                };

                ModulusAwareCiphertext {
                    data: res_data,
                    info: res_info,
                    dropped_rns_factor_indices: total_drop
                }
            }),
            (Ok((Cx, x)), Ok((Cy, y))) => PlainOrCiphertext::Ciphertext({
                **key_switches.borrow_mut() += 1;

                let total_drop = self.compute_optimal_mul_modswitch(P, C_master, &x.info, &x.dropped_rns_factor_indices, &y.info, &y.dropped_rns_factor_indices, rk.unwrap().0.gadget_vector_digits());
                let Ctarget = Params::mod_switch_down_ciphertext_ring(C_master, &total_drop);
                let rk_modswitch = Params::mod_switch_down_rk(&Ctarget, C_master, &total_drop, rk.unwrap());
                debug_assert!(total_drop.len() >= x.dropped_rns_factor_indices.len());
                debug_assert!(total_drop.len() >= y.dropped_rns_factor_indices.len());

                if total_drop.len() == x.dropped_rns_factor_indices.len() && total_drop.len() == y.dropped_rns_factor_indices.len() {
                    if LOG {
                        println!("HomMul without modulus drop");
                    }
                    ModulusAwareCiphertext {
                        dropped_rns_factor_indices: total_drop,
                        info: self.noise_estimator.hom_mul(P, &Ctarget, &x.info, &y.info, rk_modswitch.0.gadget_vector_digits()),
                        data: Params::hom_mul(P, &Ctarget, x.data, y.data, &rk_modswitch)
                    }
                } else {
                    let x_data_copy = if debug_sk.is_some() { Some(Params::clone_ct(&P, &Cx, &x.data)) } else { None };
                    let drop_x = total_drop.pushforward(&x.dropped_rns_factor_indices);
                    let x_info = self.noise_estimator.mod_switch_down(P, &Ctarget, &Cx, &drop_x, &x.info);
                    let x_data = Params::mod_switch_down(P, &Ctarget, &Cx, &drop_x, x.data);

                    let y_data_copy = if debug_sk.is_some() { Some(Params::clone_ct(&P, &Cy, &y.data)) } else { None };
                    let drop_y = total_drop.pushforward(&y.dropped_rns_factor_indices);
                    let y_info = self.noise_estimator.mod_switch_down(P, &Ctarget, &Cy, &drop_y, &y.info);
                    let y_data = Params::mod_switch_down(P, &Ctarget, &Cy, &drop_y, y.data);

                    let res_data = Params::hom_mul(P, &Ctarget, x_data, y_data, &rk_modswitch);
                    let res_info = self.noise_estimator.hom_mul(P, &Ctarget, &x_info, &y_info, rk_modswitch.0.gadget_vector_digits());

                    self.log_modulus_switch("HomMul", P, C_master, &Cx, &Cy, &Ctarget, &x.dropped_rns_factor_indices, &y.dropped_rns_factor_indices, &drop_x, &drop_y, &total_drop, &x_data_copy, &y_data_copy, &res_data, &x.info, &y.info, &res_info, debug_sk);
                    ModulusAwareCiphertext {
                        dropped_rns_factor_indices: total_drop,
                        info: res_info,
                        data: res_data
                    }
                }
            })
        }
    }

    fn gal_many<'a, R, F3>(
        &self,
        P: &PlaintextRing<Params>,
        C_master: &CiphertextRing<Params>,
        x: PlainOrCiphertext<'a, Params, Self, R>,
        ring: RingRef<R>,
        gs: &[CyclotomicGaloisGroupEl],
        gks: &[(CyclotomicGaloisGroupEl, KeySwitchKey<Params>)],
        apply_galois_action_plaintext: &mut F3,
        key_switches: &RefCell<&mut usize>,
        _debug_sk: Option<&SecretKey<Params>>
    ) -> Vec<PlainOrCiphertext<'a, Params, Self, R>>
        where R: ?Sized + RingBase,
            F3: FnMut(R::Element, &[CyclotomicGaloisGroupEl]) -> Vec<R::Element>
    {
        match x.as_ciphertext(P, C_master, ring, self) {
            Ok((Cx, x)) => {
                **key_switches.borrow_mut() += gs.len();

                let gks_mod_switched = (0..gs.len()).map(|i| {
                    if let Some((_, gk)) = gks.iter().filter(|(provided_g, _)| C_master.galois_group().eq_el(gs[i], *provided_g)).next() {
                        Params::mod_switch_down_gk(&Cx, C_master, &x.dropped_rns_factor_indices, gk)
                    } else {
                        panic!("Galois key for {} not found", C_master.galois_group().representative(gs[i]))
                    }
                }).collect::<Vec<_>>();
        
                let result = if gs.len() == 1 {
                    vec![Params::hom_galois(P, &Cx, x.data, gs[0], gks_mod_switched.at(0))]
                } else {
                    Params::hom_galois_many(P, &Cx, x.data, gs, gks_mod_switched.as_fn())
                };
                result.into_iter().zip(gs.into_iter()).zip(gks_mod_switched.iter()).map(|((res, g), gk)| PlainOrCiphertext::Ciphertext(ModulusAwareCiphertext {
                    dropped_rns_factor_indices: x.dropped_rns_factor_indices.clone(),
                    info: self.noise_estimator.hom_galois(&P, &Cx, &x.info, *g, gk.0.gadget_vector_digits()),
                    data: res
                })).collect()
            },
            Err(Coefficient::Other(x)) => apply_galois_action_plaintext(x, gs).into_iter().map(|x| PlainOrCiphertext::Plaintext(Coefficient::Other(x))).collect(),
            // integers are preserved under all galois automorphisms
            Err(x) => gs.iter().map(|_| PlainOrCiphertext::Plaintext(x.clone(ring))).collect()
        }
    }

    fn evaluate_circuit_over_any_ring<R, F1, F2, F3>(
        &self,
        circuit: &PlaintextCircuit<R>,
        P: &PlaintextRing<Params>,
        C_master: &CiphertextRing<Params>,
        inputs: &[ModulusAwareCiphertext<Params, Self>],
        rk: Option<&RelinKey<Params>>,
        gks: &[(CyclotomicGaloisGroupEl, KeySwitchKey<Params>)],
        ring: &R,
        mut hom_add_plain: F1,
        hom_mul_plain: F2,
        mut apply_galois_action_plaintext: F3,
        key_switches: &mut usize,
        mut debug_sk: Option<&SecretKey<Params>>
    ) -> Vec<ModulusAwareCiphertext<Params, Self>>
        where R: ?Sized + RingBase,
            F1: FnMut(&PlaintextRing<Params>, &CiphertextRing<Params>, &R::Element, Ciphertext<Params>, &<Self as BGVModswitchStrategy<Params>>::CiphertextInfo) -> (<Self as BGVModswitchStrategy<Params>>::CiphertextInfo, Ciphertext<Params>),
            F2: FnMut(&PlaintextRing<Params>, &CiphertextRing<Params>, &R::Element, Ciphertext<Params>, &<Self as BGVModswitchStrategy<Params>>::CiphertextInfo) -> (<Self as BGVModswitchStrategy<Params>>::CiphertextInfo, Ciphertext<Params>),
            F3: FnMut(R::Element, &[CyclotomicGaloisGroupEl]) -> Vec<R::Element>
    {
        if !LOG {
            debug_sk = None;
        }
        let key_switches_refcell = std::cell::RefCell::new(key_switches);
        let hom_mul_plain_refcell = std::cell::RefCell::new(hom_mul_plain);
        let ring = RingRef::new(ring);

        let result = circuit.evaluate_generic(
            &inputs.iter().map(PlainOrCiphertext::CiphertextRef).collect::<Vec<_>>(),
            |m| PlainOrCiphertext::PlaintextRef(m),
            |x, c, y| self.add_prod(P, C_master, x, c, y, ring, &mut hom_add_plain, &hom_mul_plain_refcell, debug_sk),
            |x, y| self.mul(P, C_master, x, y, ring, &hom_mul_plain_refcell, rk, &key_switches_refcell, debug_sk),
            |gs, x| self.gal_many(P, C_master, x, ring, gs, gks, &mut apply_galois_action_plaintext, &key_switches_refcell, debug_sk)
        );
        return result.into_iter().map(|res| match res {
            PlainOrCiphertext::Ciphertext(x) => x,
            PlainOrCiphertext::CiphertextRef(x) => {
                let Cx = Params::mod_switch_down_ciphertext_ring(C_master, &x.dropped_rns_factor_indices);
                ModulusAwareCiphertext {
                    data: Params::clone_ct(&P, &Cx, &x.data),
                    dropped_rns_factor_indices: x.dropped_rns_factor_indices.clone(),
                    info: self.clone_info(&x.info)
                }
            },
            PlainOrCiphertext::Plaintext(x) => {
                let (res_info, res_data) = hom_add_plain(P, C_master, &x.to_ring_el(ring), Params::transparent_zero(P, C_master), &self.info_for_fresh_encryption(P, C_master));
                ModulusAwareCiphertext {
                    data: res_data,
                    dropped_rns_factor_indices: RNSFactorIndexList::empty(),
                    info: res_info
                }
            },
            PlainOrCiphertext::PlaintextRef(x) => {
                let (res_info, res_data) = hom_add_plain(P, C_master, &x.clone(ring).to_ring_el(ring), Params::transparent_zero(P, C_master), &self.info_for_fresh_encryption(P, C_master));
                ModulusAwareCiphertext {
                    data: res_data,
                    dropped_rns_factor_indices: RNSFactorIndexList::empty(),
                    info: res_info
                }
            }
        }).collect();
    }
}

impl<Params: BGVParams, N: BGVNoiseEstimator<Params>, const LOG: bool> BGVModswitchStrategy<Params> for DefaultModswitchStrategy<Params, N, LOG> {

    type CiphertextInfo = N::CriticalQuantityLevel;

    #[instrument(skip_all)]
    fn evaluate_circuit_int(
        &self,
        circuit: &PlaintextCircuit<StaticRingBase<i64>>,
        P: &PlaintextRing<Params>,
        C_master: &CiphertextRing<Params>,
        inputs: &[ModulusAwareCiphertext<Params, Self>],
        rk: Option<&RelinKey<Params>>,
        gks: &[(CyclotomicGaloisGroupEl, KeySwitchKey<Params>)],
        key_switches: &mut usize,
        debug_sk: Option<&SecretKey<Params>>
    ) -> Vec<ModulusAwareCiphertext<Params, Self>> {
        self.evaluate_circuit_over_any_ring(
            circuit,
            P,
            C_master,
            inputs,
            rk,
            gks,
            StaticRing::<i64>::RING.get_ring(),
            |P, C, m, ct, noise| (
                self.noise_estimator.hom_add_plain(P, C, &P.int_hom().map(*m as i32), noise, ct.implicit_scale),
                Params::hom_add_plain(P, C, &P.int_hom().map(*m as i32), ct)
            ),
            |P, C, m, ct, noise| (
                self.noise_estimator.hom_mul_plain_i64(P, C, *m, noise, ct.implicit_scale),
                Params::hom_mul_plain_i64(P, C, *m, ct)
            ),
            |x, gs| gs.iter().map(|_| x).collect(),
            key_switches,
            debug_sk
        )
    }

    #[instrument(skip_all)]
    fn evaluate_circuit_plaintext(
        &self,
        circuit: &PlaintextCircuit<<PlaintextRing<Params> as RingStore>::Type>,
        P: &PlaintextRing<Params>,
        C_master: &CiphertextRing<Params>,
        inputs: &[ModulusAwareCiphertext<Params, Self>],
        rk: Option<&RelinKey<Params>>,
        gks: &[(CyclotomicGaloisGroupEl, KeySwitchKey<Params>)],
        key_switches: &mut usize,
        debug_sk: Option<&SecretKey<Params>>
    ) -> Vec<ModulusAwareCiphertext<Params, Self>> {
        self.evaluate_circuit_over_any_ring(
            circuit,
            P,
            C_master,
            inputs,
            rk,
            gks,
            P.get_ring(),
            |P, C, m, ct, noise| (
                self.noise_estimator.hom_add_plain(P, C, m, noise, ct.implicit_scale),
                Params::hom_add_plain(P, C, m, ct)
            ),
            |P, C, m, ct, noise| (
                self.noise_estimator.hom_mul_plain(P, C, m, noise, ct.implicit_scale),
                Params::hom_mul_plain(P, C, m, ct)
            ),
            |x, gs| P.apply_galois_action_many(&x, gs),
            key_switches,
            debug_sk
        )
    }

    fn info_for_fresh_encryption(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>) -> <Self as BGVModswitchStrategy<Params>>::CiphertextInfo {
        self.from_noise_level(self.noise_estimator.enc_sym_zero(P, C))
    }

    fn clone_info(&self, info: &Self::CiphertextInfo) -> Self::CiphertextInfo {
        self.noise_estimator.clone_critical_quantity_level(info)
    }
}

#[test]
fn test_default_modswitch_strategy() {
    let mut rng = thread_rng();

    let params = Pow2BGV {
        log2_q_min: 500,
        log2_q_max: 520,
        log2_N: 7,
        ciphertext_allocator: DefaultCiphertextAllocator::default(),
        negacyclic_ntt: PhantomData::<DefaultNegacyclicNTT>
    };
    let t = 257;
    let digits = 3;

    let P = params.create_plaintext_ring(t);
    let C = params.create_initial_ciphertext_ring();

    let sk = Pow2BGV::gen_sk(&C, &mut rng);
    let rk = Pow2BGV::gen_rk(&P, &C, &mut rng, &sk, digits);

    let input = P.int_hom().map(2);
    let ctxt = Pow2BGV::enc_sym(&P, &C, &mut rng, &input, &sk);

    let modswitch_strategy: DefaultModswitchStrategy<Pow2BGV, _, true> = DefaultModswitchStrategy::new(NaiveBGVNoiseEstimator);
    let pow8_circuit = PlaintextCircuit::mul(ZZ)
        .compose(PlaintextCircuit::mul(ZZ).output_twice(ZZ), ZZ)
        .compose(PlaintextCircuit::mul(ZZ).output_twice(ZZ), ZZ)
        .compose(PlaintextCircuit::identity(1, ZZ).output_twice(ZZ), ZZ);

    let res = modswitch_strategy.evaluate_circuit_int(
        &pow8_circuit,
        &P,
        &C,
        &[ModulusAwareCiphertext {
            dropped_rns_factor_indices: RNSFactorIndexList::empty(),
            info: modswitch_strategy.info_for_fresh_encryption(&P, &C),
            data: ctxt
        }],
        Some(&rk),
        &[],
        &mut 0,
        None
    ).into_iter().next().unwrap();

    let res_C = Pow2BGV::mod_switch_down_ciphertext_ring(&C, &res.dropped_rns_factor_indices);
    let res_sk = Pow2BGV::mod_switch_down_sk(&res_C, &C, &res.dropped_rns_factor_indices, &sk);

    let res_noise = Pow2BGV::noise_budget(&P, &res_C, &res.data, &res_sk);
    println!("Actual output noise budget is {}", res_noise);
    assert_el_eq!(&P, &P.neg_one(), Pow2BGV::dec(&P, &res_C, res.data, &res_sk));
}

#[test]
fn test_never_modswitch_strategy() {
    let mut rng = thread_rng();

    let params = Pow2BGV {
        log2_q_min: 500,
        log2_q_max: 520,
        log2_N: 7,
        ciphertext_allocator: DefaultCiphertextAllocator::default(),
        negacyclic_ntt: PhantomData::<DefaultNegacyclicNTT>
    };
    let t = 257;
    let digits = 3;

    let P = params.create_plaintext_ring(t);
    let C = params.create_initial_ciphertext_ring();

    let sk = Pow2BGV::gen_sk(&C, &mut rng);
    let rk = Pow2BGV::gen_rk(&P, &C, &mut rng, &sk, digits);

    let input = P.int_hom().map(2);
    let ctxt = Pow2BGV::enc_sym(&P, &C, &mut rng, &input, &sk);

    {
        let modswitch_strategy = DefaultModswitchStrategy::never_modswitch();
        let pow4_circuit = PlaintextCircuit::mul(ZZ)
            .compose(PlaintextCircuit::mul(ZZ).output_twice(ZZ), ZZ)
            .compose(PlaintextCircuit::identity(1, ZZ).output_twice(ZZ), ZZ);

        let res = modswitch_strategy.evaluate_circuit_int(
            &pow4_circuit,
            &P,
            &C,
            &[ModulusAwareCiphertext {
                dropped_rns_factor_indices: RNSFactorIndexList::empty(),
                info: modswitch_strategy.info_for_fresh_encryption(&P, &C),
                data: Pow2BGV::clone_ct(&P, &C, &ctxt)
            }],
            Some(&rk),
            &[],
            &mut 0,
            None
        ).into_iter().next().unwrap();

        let res_C = Pow2BGV::mod_switch_down_ciphertext_ring(&C, &res.dropped_rns_factor_indices);
        let res_sk = Pow2BGV::mod_switch_down_sk(&res_C, &C, &res.dropped_rns_factor_indices, &sk);

        let res_noise = Pow2BGV::noise_budget(&P, &res_C, &res.data, &res_sk);
        println!("Actual output noise budget is {}", res_noise);
        assert_el_eq!(&P, &P.int_hom().map(16), Pow2BGV::dec(&P, &res_C, res.data, &res_sk));
    }
    {
        let modswitch_strategy = DefaultModswitchStrategy::never_modswitch();
        let pow8_circuit = PlaintextCircuit::mul(ZZ)
            .compose(PlaintextCircuit::mul(ZZ).output_twice(ZZ), ZZ)
            .compose(PlaintextCircuit::mul(ZZ).output_twice(ZZ), ZZ)
            .compose(PlaintextCircuit::identity(1, ZZ).output_twice(ZZ), ZZ);

        let res = modswitch_strategy.evaluate_circuit_int(
            &pow8_circuit,
            &P,
            &C,
            &[ModulusAwareCiphertext {
                dropped_rns_factor_indices: RNSFactorIndexList::empty(),
                info: modswitch_strategy.info_for_fresh_encryption(&P, &C),
                data: Pow2BGV::clone_ct(&P, &C, &ctxt)
            }],
            Some(&rk),
            &[],
            &mut 0,
            None
        ).into_iter().next().unwrap();

        let res_C = Pow2BGV::mod_switch_down_ciphertext_ring(&C, &res.dropped_rns_factor_indices);
        let res_sk = Pow2BGV::mod_switch_down_sk(&res_C, &C, &res.dropped_rns_factor_indices, &sk);

        let res_noise = Pow2BGV::noise_budget(&P, &res_C, &res.data, &res_sk);
        assert_eq!(0, res_noise);
    }
}