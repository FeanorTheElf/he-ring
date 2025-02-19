use core::f64;

use feanor_math::homomorphism::Homomorphism;
use feanor_math::primitive_int::*;
use feanor_math::ring::*;

use crate::circuit::Coefficient;
use crate::circuit::PlaintextCircuit;
use crate::cyclotomic::CyclotomicGaloisGroupEl;
use crate::gadget_product::digits::*;

use super::*;

pub trait BGVNoiseEstimator<Params: BGVParams> {

    ///
    /// An estimate of the size and distribution of the critical quantity
    /// `c0 + c1 s = m + t e`. The only requirement is that the noise estimator
    /// can derive an estimate about its infinity norm via
    /// [`BGVNoiseEstimator::get_ln_relative_noise_level`], but estimators are free
    /// to store additional data to get more precise estimates on the noise growth
    /// of operations. 
    /// 
    type CriticalQuantityLevel;

    ///
    /// Should return an estimate of
    /// ```text
    ///   ln( | c0 + c1 * s |_inf / q )
    /// ```
    /// 
    fn estimate_ln_relative_noise_level(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, noise: &Self::CriticalQuantityLevel) -> f64;

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

    fn hom_galois(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, ct: &Self::CriticalQuantityLevel, g: CyclotomicGaloisGroupEl, gk: &RNSGadgetVectorDigitIndices) -> Self::CriticalQuantityLevel {
        self.key_switch(P, C, ct, gk)
    }

    fn mod_switch_down(&self, P: &PlaintextRing<Params>, Cnew: &CiphertextRing<Params>, Cold: &CiphertextRing<Params>, drop_moduli: &DropModuliIndices, ct: &Self::CriticalQuantityLevel) -> Self::CriticalQuantityLevel;
    
    fn change_plaintext_modulus(Pnew: &PlaintextRing<Params>, Pold: &PlaintextRing<Params>, C: &CiphertextRing<Params>, ct: &Self::CriticalQuantityLevel) -> Self::CriticalQuantityLevel;
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

    fn estimate_ln_relative_noise_level(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, noise: &Self::CriticalQuantityLevel) -> f64 {
        // we subtract `(C.rank() as f64).log2()`, since that should be about the difference between `l_inf` and canonical norm
        *noise * 2f64.ln() - (C.rank() as f64).log2()
    }

    fn enc_sym_zero(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>) -> Self::CriticalQuantityLevel {
        (*P.base_ring().modulus() as f64).log2() + log2_can_norm_sk_estimate::<Params>(C) - BigIntRing::RING.abs_log2_floor(C.base_ring().modulus()).unwrap() as f64
    }

    fn hom_add(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, lhs: &Self::CriticalQuantityLevel, lhs_implicit_scale: Option<El<Zn>>, rhs: &Self::CriticalQuantityLevel, rhs_implicit_scale: Option<El<Zn>>) -> Self::CriticalQuantityLevel {
        if lhs_implicit_scale.is_none() || rhs_implicit_scale.is_none() {
            return f64::max(*lhs, *rhs) + (*P.base_ring().modulus() as f64).log2() / 2.;
        } else {
            let Zt = P.base_ring();
            let ZZ_to_Zt = Zt.can_hom(Zt.integer_ring()).unwrap();
            let (a, b) = rational_reconstruction(Zt, Zt.checked_div(&lhs_implicit_scale.unwrap(), &rhs_implicit_scale.unwrap()).unwrap());
            return f64::max((b as f64).log2() + *lhs, (a as f64).log2() + *rhs);
        }
    }

    fn hom_add_plain(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, m: &El<PlaintextRing<Params>>, ct: &Self::CriticalQuantityLevel, implicit_scale: El<Zn>) -> Self::CriticalQuantityLevel {
        *ct
    }

    fn hom_mul_plain(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, m: &El<PlaintextRing<Params>>, ct: &Self::CriticalQuantityLevel, implicit_scale: El<Zn>) -> Self::CriticalQuantityLevel {
        *ct + (P.wrt_canonical_basis(m).iter().map(|c| P.base_ring().smallest_lift(c).abs()).max().unwrap() as f64 * C.rank() as f64).log2()
    }

    fn hom_mul_plain_i64(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, m: i64, ct: &Self::CriticalQuantityLevel, implicit_scale: El<Zn>) -> Self::CriticalQuantityLevel {
        *ct + (m.abs() as f64).log2()
    }

    fn transparent_zero(&self) -> Self::CriticalQuantityLevel {
        -f64::INFINITY
    }

    fn mod_switch_down(&self, P: &PlaintextRing<Params>, Cnew: &CiphertextRing<Params>, Cold: &CiphertextRing<Params>, drop_moduli: &DropModuliIndices, ct: &Self::CriticalQuantityLevel) -> Self::CriticalQuantityLevel {
        f64::max(
            *ct,
            (*P.base_ring().modulus() as f64).log2() + log2_can_norm_sk_estimate::<Params>(Cnew) - BigIntRing::RING.abs_log2_ceil(Cnew.base_ring().modulus()).unwrap() as f64
        )
    }

    fn hom_mul(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, lhs: &Self::CriticalQuantityLevel, rhs: &Self::CriticalQuantityLevel, rk_digits: &RNSGadgetVectorDigitIndices) -> Self::CriticalQuantityLevel {
        <Self as BGVNoiseEstimator<Params>>::key_switch(self, P, C, &(*lhs + *rhs + BigIntRing::RING.abs_log2_ceil(C.base_ring().modulus()).unwrap() as f64), rk_digits)
    }

    fn key_switch(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, ct: &Self::CriticalQuantityLevel, switch_key: &RNSGadgetVectorDigitIndices) -> Self::CriticalQuantityLevel {
        let log2_largest_digit = switch_key.iter().map(|digit| digit.iter().map(|i| *C.base_ring().at(i).modulus() as f64).map(f64::log2).sum::<f64>()).max_by(f64::total_cmp).unwrap();
        f64::max(
            *ct,
            log2_largest_digit + (C.rank() as f64).log2() * 2. - BigIntRing::RING.abs_log2_ceil(C.base_ring().modulus()).unwrap() as f64
        )
    }

    fn change_plaintext_modulus(Pnew: &PlaintextRing<Params>, Pold: &PlaintextRing<Params>, C: &CiphertextRing<Params>, ct: &Self::CriticalQuantityLevel) -> Self::CriticalQuantityLevel {
        *ct
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

    fn estimate_ln_relative_noise_level(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, noise: &Self::CriticalQuantityLevel) -> f64 {
        0.
    }

    fn enc_sym_zero(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>) -> Self::CriticalQuantityLevel {}
    fn hom_add(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, lhs: &Self::CriticalQuantityLevel, lhs_implicit_scale: Option<El<Zn>>, rhs: &Self::CriticalQuantityLevel, rhs_implicit_scale: Option<El<Zn>>) -> Self::CriticalQuantityLevel {}
    fn hom_add_plain(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, m: &El<PlaintextRing<Params>>, ct: &Self::CriticalQuantityLevel, implicit_scale: El<Zn>) -> Self::CriticalQuantityLevel {}
    fn hom_galois(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, ct: &Self::CriticalQuantityLevel, g: CyclotomicGaloisGroupEl, gk: &RNSGadgetVectorDigitIndices) -> Self::CriticalQuantityLevel {}
    fn hom_mul(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, lhs: &Self::CriticalQuantityLevel, rhs: &Self::CriticalQuantityLevel, rk_digits: &RNSGadgetVectorDigitIndices) -> Self::CriticalQuantityLevel {}
    fn hom_mul_plain(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, m: &El<PlaintextRing<Params>>, ct: &Self::CriticalQuantityLevel, implicit_scale: El<Zn>) -> Self::CriticalQuantityLevel {}
    fn hom_mul_plain_i64(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, m: i64, ct: &Self::CriticalQuantityLevel, implicit_scale: El<Zn>) -> Self::CriticalQuantityLevel {}
    fn key_switch(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, ct: &Self::CriticalQuantityLevel, switch_key: &RNSGadgetVectorDigitIndices) -> Self::CriticalQuantityLevel {}
    fn mod_switch_down(&self, P: &PlaintextRing<Params>, Cnew: &CiphertextRing<Params>, Cold: &CiphertextRing<Params>, drop_moduli: &DropModuliIndices, ct: &Self::CriticalQuantityLevel) -> Self::CriticalQuantityLevel {}
    fn transparent_zero(&self) -> Self::CriticalQuantityLevel {}
    fn change_plaintext_modulus(Pnew: &PlaintextRing<Params>, Pold: &PlaintextRing<Params>, C: &CiphertextRing<Params>, ct: &Self::CriticalQuantityLevel) -> Self::CriticalQuantityLevel {}
}

///
/// Trait for different modulus-switching strategies in BGV, currently WIP
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
    fn evaluate_circuit(
        &self,
        circuit: &PlaintextCircuit<StaticRingBase<i64>>,
        P: &PlaintextRing<Params>,
        C_master: &CiphertextRing<Params>,
        inputs: &[(Box<DropModuliIndices>, Self::CiphertextInfo, Ciphertext<Params>)],
        rk: Option<&RelinKey<Params>>,
        gks: &[(CyclotomicGaloisGroupEl, KeySwitchKey<Params>)]
    ) -> Vec<(Box<DropModuliIndices>, Self::CiphertextInfo, Ciphertext<Params>)>;
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

impl<Params: BGVParams, N: BGVNoiseEstimator<Params>, const LOG: bool> DefaultModswitchStrategy<Params, N, LOG> {
    
    pub fn new(noise_estimator: N) -> Self {
        Self {
            params: PhantomData,
            noise_estimator: noise_estimator
        }
    }

    pub fn fresh_encryption(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>) -> <Self as BGVModswitchStrategy<Params>>::CiphertextInfo {
        self.from_noise_level(self.noise_estimator.enc_sym_zero(P, C))
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
        drop_x: &DropModuliIndices,
        noise_y: &N::CriticalQuantityLevel,
        drop_y: &DropModuliIndices,
        rk_digits: &RNSGadgetVectorDigitIndices
    ) -> Box<DropModuliIndices> {
        let Cx = Params::mod_switch_down_ciphertext_ring(C_master, drop_x);
        let Cy = Params::mod_switch_down_ciphertext_ring(C_master, drop_y);
        let outer_drop = drop_x.union(&drop_y);
        let after_first_drop = rk_digits.remove_indices(&outer_drop);
        let compute_result_noise = |num_to_drop: usize| {
            let inner_drop = recommended_rns_factors_to_drop(&after_first_drop, num_to_drop);
            let total_drop = inner_drop.pullback(&outer_drop);
            let C_target = Params::mod_switch_down_ciphertext_ring(C_master, &total_drop);
            let rk_after_total_drop = rk_digits.remove_indices(&total_drop);
            
            let expected_noise = self.noise_estimator.estimate_ln_relative_noise_level(
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
    
    fn evaluate_circuit_over_any_ring<R, F1, F2>(
        &self,
        circuit: &PlaintextCircuit<R>,
        P: &PlaintextRing<Params>,
        C_master: &CiphertextRing<Params>,
        inputs: &[(Box<DropModuliIndices>, <Self as BGVModswitchStrategy<Params>>::CiphertextInfo, Ciphertext<Params>)],
        rk: Option<&RelinKey<Params>>,
        gks: &[(CyclotomicGaloisGroupEl, KeySwitchKey<Params>)],
        ring: &R,
        mut hom_add_plain: F1,
        mut hom_mul_plain: F2
    ) -> Vec<(Box<DropModuliIndices>, <Self as BGVModswitchStrategy<Params>>::CiphertextInfo, Ciphertext<Params>)>
        where R: ?Sized + RingBase,
            F1: FnMut(&PlaintextRing<Params>, &CiphertextRing<Params>, &R::Element, Ciphertext<Params>, &<Self as BGVModswitchStrategy<Params>>::CiphertextInfo) -> (<Self as BGVModswitchStrategy<Params>>::CiphertextInfo, Ciphertext<Params>),
            F2: FnMut(&PlaintextRing<Params>, &CiphertextRing<Params>, &R::Element, Ciphertext<Params>, &<Self as BGVModswitchStrategy<Params>>::CiphertextInfo) -> (<Self as BGVModswitchStrategy<Params>>::CiphertextInfo, Ciphertext<Params>),
    {
        circuit.evaluate_generic(
            inputs,
            |m| {
                let (result_noise, result) = hom_add_plain(P, C_master, &m.clone(RingRef::new(ring)).to_ring_el(RingRef::new(ring)), Params::transparent_zero(P, C_master), &self.noise_estimator.transparent_zero());
                return (DropModuliIndices::empty(), result_noise, result);
            },
            |x, c, y| {

                if let Coefficient::Zero = c {
                    return x;
                }

                let res = x.0.union(&y.0);
                let Cres = Params::mod_switch_down_ciphertext_ring(C_master, &res);
        
                let Cx = Params::mod_switch_down_ciphertext_ring(C_master, &x.0);
                let drop_x = res.pushforward(&x.0);
                if LOG && drop_x.len() > 0 {
                    println!("Dropping RNS factors {:?} from addition operand to make addition well-defined", drop_x);
                }

                let x_noise = self.noise_estimator.mod_switch_down(&P, &Cres, &Cx, &drop_x, &x.1);
                let x_modswitch = Params::mod_switch_down(P, &Cres, &Cx, &drop_x, x.2);

                let Cy = Params::mod_switch_down_ciphertext_ring(C_master, &y.0);
                let drop_y = res.pushforward(&y.0);
                if LOG && drop_y.len() > 0 {
                    println!("Dropping RNS factors {:?} from addition operand to make addition well-defined", drop_x);
                }

                let y_noise = self.noise_estimator.mod_switch_down(&P, &Cres, &Cy, &drop_y, &y.1);
                let y_modswitch = Params::mod_switch_down(P, &Cres, &Cy, &drop_y, Params::clone_ct(P, &Cy, &y.2));
                
                let (y_noise, y_scaled) = match c {
                    Coefficient::Zero => unreachable!(),
                    Coefficient::One => (y_noise, y_modswitch),
                    Coefficient::NegOne => (y_noise, Params::hom_mul_plain_i64(P, &Cy, -1, y_modswitch)),
                    Coefficient::Integer(c) => (
                        self.noise_estimator.hom_mul_plain_i64(P, &Cy, *c as i64, &y_noise, y_modswitch.implicit_scale),
                        Params::hom_mul_plain_i64(P, &Cy, *c as i64, y_modswitch)
                    ),
                    Coefficient::Other(m) => hom_mul_plain(P, &Cy, m, y_modswitch, &y_noise)
                };

                return (
                    res, 
                    self.noise_estimator.hom_add(P, C_master, &x_noise, Some(x_modswitch.implicit_scale), &y_noise, Some(y_scaled.implicit_scale)), 
                    Params::hom_add(P, &Cres, x_modswitch, y_scaled)
                );
            },
            |x, y| {

                let total_drop = self.compute_optimal_mul_modswitch(P, C_master, &x.1, &x.0, &y.1, &y.0, rk.unwrap().0.gadget_vector_digits());
                let Ctarget = Params::mod_switch_down_ciphertext_ring(C_master, &total_drop);
                let rk_modswitch = Params::mod_switch_down_rk(P, &Ctarget, C_master, &total_drop, rk.unwrap());

                if total_drop.len() == x.0.len() {
                    debug_assert_eq!(total_drop.len(), y.0.len());
                    let Cx = &Ctarget;
                    let Cy = &Ctarget;
                    if LOG {
                        println!("Performing homomorphic multiplication without modulus drop");
                    }
                    return (
                        total_drop,
                        self.noise_estimator.hom_mul(P, &Ctarget, &x.1, &y.1, rk_modswitch.0.gadget_vector_digits()),
                        Params::hom_mul(P, &Ctarget, x.2, y.2, &rk_modswitch)
                    );
                }


                let Cx = Params::mod_switch_down_ciphertext_ring(C_master, &x.0);
                let drop_x = total_drop.pushforward(&x.0);
                if LOG {
                    println!("Dropping {:?} from homomorphic product operand with estimated noise {}", drop_x, self.noise_estimator.estimate_ln_relative_noise_level(P, &Cx, &x.1));
                }

                let x_noise = self.noise_estimator.mod_switch_down(P, &Ctarget, &Cx, &drop_x, &x.1);
                let x_modswitch = Params::mod_switch_down(P, &Ctarget, &Cx, &drop_x, x.2);

                let Cy = Params::mod_switch_down_ciphertext_ring(C_master, &y.0);
                let drop_y = total_drop.pushforward(&y.0);
                if LOG {
                    println!("Dropping {:?} from homomorphic product operand with estimated noise {}", drop_y, self.noise_estimator.estimate_ln_relative_noise_level(P, &Cy, &y.1));
                }

                let y_noise = self.noise_estimator.mod_switch_down(P, &Ctarget, &Cy, &drop_y, &y.1);
                let y_modswitch = Params::mod_switch_down(P, &Ctarget, &Cy, &drop_y, y.2);

                let result = Params::hom_mul(P, &Ctarget, x_modswitch, y_modswitch, &rk_modswitch);
                let result_noise = self.noise_estimator.hom_mul(P, &Ctarget, &x_noise, &y_noise, rk_modswitch.0.gadget_vector_digits());
                if LOG {
                    println!("Estimated output noise is {}", self.noise_estimator.estimate_ln_relative_noise_level(P, &Ctarget, &result_noise));
                }

                return (total_drop, result_noise, result);
            },
            |gs, x| {
                let Cx = Params::mod_switch_down_ciphertext_ring(C_master, &x.0);
                let gks_mod_switched = (0..gs.len()).map(|i| {
                    if let Some((_, gk)) = gks.iter().filter(|(provided_g, _)| C_master.galois_group().eq_el(gs[i], *provided_g)).next() {
                        Params::mod_switch_down_gk(P, &Cx, C_master, &x.0, gk)
                    } else {
                        panic!("Galois key for {} not found", C_master.galois_group().representative(gs[i]))
                    }
                }).collect::<Vec<_>>();

                let result = Params::hom_galois_many(P, &Cx, x.2, gs, gks_mod_switched.as_fn());
                return result.into_iter().zip(gs.into_iter()).zip(gks_mod_switched.iter()).map(|((res, g), gk)| (
                    x.0.clone(),
                    self.noise_estimator.hom_galois(&P, &Cx, &x.1, *g, gk.0.gadget_vector_digits()),
                    res
                )).collect();
            }
        )
    }
}

impl<Params: BGVParams, N: BGVNoiseEstimator<Params>, const LOG: bool> BGVModswitchStrategy<Params> for DefaultModswitchStrategy<Params, N, LOG> {

    type CiphertextInfo = N::CriticalQuantityLevel;

    #[instrument(skip_all)]
    fn evaluate_circuit(
        &self,
        circuit: &PlaintextCircuit<StaticRingBase<i64>>,
        P: &PlaintextRing<Params>,
        C_master: &CiphertextRing<Params>,
        inputs: &[(Box<DropModuliIndices>, Self::CiphertextInfo, Ciphertext<Params>)],
        rk: Option<&RelinKey<Params>>,
        gks: &[(CyclotomicGaloisGroupEl, KeySwitchKey<Params>)]
    ) -> Vec<(Box<DropModuliIndices>, Self::CiphertextInfo, Ciphertext<Params>)> {
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
            )
        )
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

    let (res_dropped_indices, res_noise_estimate, res_ctxt) = modswitch_strategy.evaluate_circuit(
        &pow8_circuit,
        &P, 
        &C,
        &[(DropModuliIndices::empty(), modswitch_strategy.fresh_encryption(&P, &C), ctxt)],
        Some(&rk),
        &[]
    ).into_iter().next().unwrap();

    let res_C = Pow2BGV::mod_switch_down_ciphertext_ring(&C, &res_dropped_indices);
    let res_sk = Pow2BGV::mod_switch_down_sk(&P, &res_C, &C, &res_dropped_indices, &sk);

    let res_noise = Pow2BGV::noise_budget(&P, &res_C, &res_ctxt, &res_sk);
    println!("Actual output noise budget is {}", res_noise);
    assert_el_eq!(&P, &P.neg_one(), Pow2BGV::dec(&P, &res_C, res_ctxt, &res_sk));
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

        let (res_dropped_indices, res_noise_estimate, res_ctxt) = modswitch_strategy.evaluate_circuit(
            &pow4_circuit,
            &P, 
            &C,
            &[(DropModuliIndices::empty(), modswitch_strategy.fresh_encryption(&P, &C), Pow2BGV::clone_ct(&P, &C, &ctxt))],
            Some(&rk),
            &[]
        ).into_iter().next().unwrap();

        let res_C = Pow2BGV::mod_switch_down_ciphertext_ring(&C, &res_dropped_indices);
        let res_sk = Pow2BGV::mod_switch_down_sk(&P, &res_C, &C, &res_dropped_indices, &sk);

        let res_noise = Pow2BGV::noise_budget(&P, &res_C, &res_ctxt, &res_sk);
        println!("Actual output noise budget is {}", res_noise);
        assert_el_eq!(&P, &P.int_hom().map(16), Pow2BGV::dec(&P, &res_C, res_ctxt, &res_sk));
    }
    {
        let modswitch_strategy = DefaultModswitchStrategy::never_modswitch();
        let pow8_circuit = PlaintextCircuit::mul(ZZ)
            .compose(PlaintextCircuit::mul(ZZ).output_twice(ZZ), ZZ)
            .compose(PlaintextCircuit::mul(ZZ).output_twice(ZZ), ZZ)
            .compose(PlaintextCircuit::identity(1, ZZ).output_twice(ZZ), ZZ);

        let (res_dropped_indices, res_noise_estimate, res_ctxt) = modswitch_strategy.evaluate_circuit(
            &pow8_circuit,
            &P, 
            &C,
            &[(DropModuliIndices::empty(), modswitch_strategy.fresh_encryption(&P, &C), Pow2BGV::clone_ct(&P, &C, &ctxt))],
            Some(&rk),
            &[]
        ).into_iter().next().unwrap();

        let res_C = Pow2BGV::mod_switch_down_ciphertext_ring(&C, &res_dropped_indices);
        let res_sk = Pow2BGV::mod_switch_down_sk(&P, &res_C, &C, &res_dropped_indices, &sk);

        let res_noise = Pow2BGV::noise_budget(&P, &res_C, &res_ctxt, &res_sk);
        assert_eq!(0, res_noise);
    }
}