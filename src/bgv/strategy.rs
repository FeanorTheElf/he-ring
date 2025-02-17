use feanor_math::homomorphism::Homomorphism;
use feanor_math::primitive_int::*;
use feanor_math::ring::*;

use crate::circuit::PlaintextCircuit;
use crate::cyclotomic::CyclotomicGaloisGroupEl;
use crate::gadget_product::digits::RNSGadgetVectorDigitList;

use super::*;

pub trait BGVNoiseEstimator<Params: BGVParams> {

    type CriticalQuantityLevel;

    fn get_relative_error_level(&self, noise: &Self::CriticalQuantityLevel) -> f64;

    fn enc_sym_zero(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>) -> Self::CriticalQuantityLevel;

    fn transparent_zero(&self) -> Self::CriticalQuantityLevel;

    fn hom_add_plain(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, m: &El<PlaintextRing<Params>>, ct: &Self::CriticalQuantityLevel, implicit_scale: El<Zn>) -> Self::CriticalQuantityLevel;

    fn enc_sym(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, m: &El<PlaintextRing<Params>>) -> Self::CriticalQuantityLevel {
        self.hom_add_plain(P, C, m, &self.enc_sym_zero(P, C), P.base_ring().one())
    }

    fn hom_mul_plain(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, m: &El<PlaintextRing<Params>>, ct: &Self::CriticalQuantityLevel, implicit_scale: El<Zn>) -> Self::CriticalQuantityLevel;

    fn key_switch(&self, C: &CiphertextRing<Params>, ct: &Self::CriticalQuantityLevel, implicit_scale: El<Zn>, switch_key: &RNSGadgetVectorDigitList) -> Self::CriticalQuantityLevel;

    fn hom_mul(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, lhs: &Self::CriticalQuantityLevel, lhs_implicit_scale: El<Zn>, rhs: &Self::CriticalQuantityLevel, rhs_implicit_scale: El<Zn>, rk_digits: &RNSGadgetVectorDigitList) -> Self::CriticalQuantityLevel;

    fn hom_add(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, lhs: &Self::CriticalQuantityLevel, lhs_implicit_scale: El<Zn>, rhs: &Self::CriticalQuantityLevel, rhs_implicit_scale: El<Zn>) -> Self::CriticalQuantityLevel;

    fn hom_galois_many<'a, 'b, V>(&self, P: &PlaintextRing<Params>, C: &CiphertextRing<Params>, ct: &Self::CriticalQuantityLevel, gs: &[CyclotomicGaloisGroupEl], gks: V) -> Vec<Self::CriticalQuantityLevel>
        where V: VectorFn<&'b KeySwitchKey<'a, Params>>,
            KeySwitchKey<'a, Params>: 'b,
            'a: 'b,
            Params: 'a;

    fn mod_switch_down(&self, P: &PlaintextRing<Params>, Cnew: &CiphertextRing<Params>, Cold: &CiphertextRing<Params>, drop_moduli: &DropModuliIndices, ct: &Self::CriticalQuantityLevel, implicit_scale: El<Zn>) -> Self::CriticalQuantityLevel;
}

///
/// Trait for different modulus-switching strategies in BGV, currently WIP
/// 
pub trait BGVModswitchStrategy<Params: BGVParams> {

    type CiphertextDescriptor;

    fn evaluate_circuit(
        &self,
        circuit: &PlaintextCircuit<StaticRingBase<i64>>,
        P: &PlaintextRing<Params>,
        C_master: &CiphertextRing<Params>,
        inputs: &[(Box<DropModuliIndices>, Self::CiphertextDescriptor, Ciphertext<Params>)],
        rk: Option<&RelinKey<Params>>
    ) -> Vec<(Box<DropModuliIndices>, Self::CiphertextDescriptor, Ciphertext<Params>)>;
}

///
/// Default modulus-switch strategy for BGV, which performs a certain number of modulus-switches
/// before each multiplication. Currently WIP.
/// 
pub struct DefaultModswitchStrategy<Params: BGVParams, N: BGVNoiseEstimator<Params>> {
    params: PhantomData<Params>,
    noise_estimator: N
}

impl<Params: BGVParams, N: BGVNoiseEstimator<Params>> DefaultModswitchStrategy<Params, N> {
    
    fn optimal_modswitch_to(
        &self,
        P: &PlaintextRing<Params>,
        C_master: &CiphertextRing<Params>,
        noise_x: &N::CriticalQuantityLevel,
        drop_x: &DropModuliIndices,
        noise_y: &N::CriticalQuantityLevel,
        drop_y: &DropModuliIndices,
        rk: &RelinKey<Params>
    ) {
        let min_drop = drop_x.union(&drop_y);
        let compute_result_noise = |num_to_drop: usize| {

        };
    }
}

impl<Params: BGVParams, N: BGVNoiseEstimator<Params>> BGVModswitchStrategy<Params> for DefaultModswitchStrategy<Params, N> {

    type CiphertextDescriptor = N::CriticalQuantityLevel;

    fn evaluate_circuit(
        &self,
        circuit: &PlaintextCircuit<StaticRingBase<i64>>,
        P: &PlaintextRing<Params>,
        C_master: &CiphertextRing<Params>,
        inputs: &[(Box<DropModuliIndices>, Self::CiphertextDescriptor, Ciphertext<Params>)],
        rk: Option<&RelinKey<Params>>
    ) -> Vec<(Box<DropModuliIndices>, Self::CiphertextDescriptor, Ciphertext<Params>)> {

        let result = circuit.evaluate_generic(
            inputs,
            |x| {
                let m = P.int_hom().map(x.to_ring_el(StaticRing::<i64>::RING) as i32);
                (
                    DropModuliIndices::empty(),
                    self.noise_estimator.hom_add_plain(P, C_master, &m, &self.noise_estimator.transparent_zero(), P.base_ring().one()),
                    Params::hom_add_plain(P, C_master, &m, Params::transparent_zero(P, C_master))
                )
            },
            |x, c, y| {
                let res = x.0.union(&y.0);
                let Cx = Params::mod_switch_down_ciphertext_ring(C_master, &x.0);
                let Cy = Params::mod_switch_down_ciphertext_ring(C_master, &y.0);
                let Cres = Params::mod_switch_down_ciphertext_ring(C_master, &res);
                
                let drop_x = res.within(&x.0);
                let x_noise = self.noise_estimator.mod_switch_down(&P, &Cres, &Cx, &drop_x, &x.1, x.2.implicit_scale);
                let x = Params::mod_switch_down(P, &Cres, &Cx, &drop_x, x.2);

                let drop_y = res.within(&y.0);
                let y_noise = self.noise_estimator.mod_switch_down(&P, &Cres, &Cy, &drop_y, &y.1, y.2.implicit_scale);
                let y = Params::mod_switch_down(P, &Cres, &Cy, &drop_y, Params::clone_ct(P, &Cy, &y.2));
                return (
                    res, 
                    self.noise_estimator.hom_add(P, C_master, &x_noise, x.implicit_scale, &y_noise, y.implicit_scale), 
                    Params::hom_add(P, &Cres, x, y)
                );
            },
            |x, y| {
                return unimplemented!();
            },
            |x, gs| {
                return unimplemented!();
            }
        );

        return unimplemented!();
    }
}