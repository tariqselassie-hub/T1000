// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Interaction Holonomy & Generalized Exclusion Statistics (GES) Engine.
//! Implements unitary population transfer and geometric phase accumulation
//! via Confinement-Induced Resonances (CIR) based on the Lieb-Liniger holonomy cycle:
//! g: 0 -> +inf -> -inf -> 0, generating Fractional Fermi Seas (FFS) with alpha = l.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Sovereign Quantum Cellular Automata Rule Families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum QcaRuleKind {
    /// Standard B3/S23 Confinement-Induced Resonance & Fractional Fermi Sea (Default).
    #[default]
    B3S23Holonomy,
    /// HighLife B36/S23 Topological Replicator with quantum cloning interference.
    HighLifeReplicator,
    /// Non-linear Gross-Pitaevskii Superfluid BEC exhibiting quantized vortex dynamics.
    GrossPitaevskiiBec,
    /// Floquet Kicked Chirikov-Harper Quantum Chaos with non-linear potential V = K cos(pi N).
    FloquetKickedChaos,
    /// Relativistic Dirac-Weyl QCA with mass defect m(N), Zitterbewegung and Klein tunneling.
    DiracZitterbewegung,
    /// Cyclic Lotka-Volterra 3-species predator-prey holonomy with spiral turbulence.
    CyclicPredatorPrey,
}

/// Parameters defining the interaction holonomy cycle across the CIR and chaos engines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolonomyParams {
    /// Active QCA rule family.
    pub rule: QcaRuleKind,
    /// Statistical parameter alpha = l (l = 2 for first super-fermionic FFS branch).
    pub l_quantum: usize,
    /// Base coupling scale g_0.
    pub g0: f64,
    /// Confinement-Induced Resonance width Gamma_CIR.
    pub cir_width: f64,
    /// Holonomy cycle period tau.
    pub cycle_period: f64,
    /// Birth sharpness parameter around neighbor count N = 3.
    pub birth_sharpness: f64,
    /// Floquet chaos kicking parameter K in [0.0, 8.0].
    pub chaos_k: f64,
    /// Synthetic magnetic gauge flux Phi / Phi_0 in [0.0, 1.0].
    pub gauge_flux: f64,
    /// Non-linear Gross-Pitaevskii interaction strength g.
    pub non_linear_g: f64,
}

impl Default for HolonomyParams {
    fn default() -> Self {
        Self {
            rule: QcaRuleKind::B3S23Holonomy,
            l_quantum: 2,
            g0: 1.0,
            cir_width: 0.5,
            cycle_period: 1.0,
            birth_sharpness: 2.0,
            chaos_k: 1.2,
            gauge_flux: 0.0,
            non_linear_g: 0.8,
        }
    }
}

/// Unitary 3x3 local transition matrix derived from the interaction holonomy.
#[derive(Debug, Clone, PartialEq)]
pub struct HolonomyTransferMatrix {
    /// Matrix elements `U[row][col]` mapping `(c0, c1, c2) -> (c0', c1', c2')`.
    pub elements: [[Complex64; 3]; 3],
}

impl HolonomyTransferMatrix {
    /// Construct identity transfer matrix.
    pub fn identity() -> Self {
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        Self {
            elements: [[one, zero, zero], [zero, one, zero], [zero, zero, one]],
        }
    }

    /// Multiply this unitary transfer matrix by a 3-amplitude vector [c0, c1, c2].
    #[inline(always)]
    pub fn apply(&self, c: [Complex64; 3]) -> [Complex64; 3] {
        let mut out = [Complex64::new(0.0, 0.0); 3];
        for (r, item) in out.iter_mut().enumerate() {
            *item = self.elements[r][0] * c[0]
                + self.elements[r][1] * c[1]
                + self.elements[r][2] * c[2];
        }
        out
    }

    /// Check unitarity: max deviation of U^dagger U from Identity.
    pub fn unitarity_error(&self) -> f64 {
        let mut max_err = 0.0f64;
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..3 {
                    // U_ki^* * U_kj
                    sum += self.elements[k][i].conj() * self.elements[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                let err = (sum.re - expected).hypot(sum.im);
                if err > max_err {
                    max_err = err;
                }
            }
        }
        max_err
    }
}

/// Compute the theoretical geometric Berry/Wilczek-Zee holonomy phase
/// accumulated per closed CIR cycle: gamma_l = pi * (1 - 1 / l).
#[inline(always)]
pub fn geometric_holonomy_phase(l_quantum: usize) -> f64 {
    if l_quantum <= 1 {
        0.0
    } else {
        PI * (1.0 - 1.0 / (l_quantum as f64))
    }
}

/// Compute the Haldane Generalized Exclusion momentum plateau: n(k) = 1 / l.
#[inline(always)]
pub fn ffs_momentum_plateau(l_quantum: usize) -> f64 {
    if l_quantum == 0 {
        1.0
    } else {
        1.0 / (l_quantum as f64)
    }
}

/// Construct a strictly unitary 3x3 holonomy transfer matrix conditioned on effective
/// neighborhood occupation count `nbr_occ` and active rule family.
pub fn build_holonomy_unitary(nbr_occ: f64, params: &HolonomyParams) -> HolonomyTransferMatrix {
    match params.rule {
        QcaRuleKind::B3S23Holonomy => build_b3s23_unitary(nbr_occ, params),
        QcaRuleKind::HighLifeReplicator => build_highlife_unitary(nbr_occ, params),
        QcaRuleKind::GrossPitaevskiiBec => build_bec_unitary(nbr_occ, params),
        QcaRuleKind::FloquetKickedChaos => build_floquet_chaos_unitary(nbr_occ, params),
        QcaRuleKind::DiracZitterbewegung => build_dirac_unitary(nbr_occ, params),
        QcaRuleKind::CyclicPredatorPrey => build_cyclic_predator_prey_unitary(nbr_occ, params),
    }
}

/// Standard B3/S23 CIR Holonomy Unitary.
fn build_b3s23_unitary(nbr_occ: f64, params: &HolonomyParams) -> HolonomyTransferMatrix {
    let gamma_geom = geometric_holonomy_phase(params.l_quantum);
    let delta_birth = nbr_occ - 3.0;
    let birth_weight = (-params.birth_sharpness * delta_birth * delta_birth).exp();
    let theta_b = (PI * 0.5) * birth_weight;
    let survival_active = (1.8..=3.2).contains(&nbr_occ);
    let cir_delta = (nbr_occ - 4.0).max(0.0) / params.cir_width;
    let theta_cir = (PI * 0.5) * (1.0 - (-cir_delta).exp());
    let underpop_delta = (1.8 - nbr_occ).max(0.0) / params.cir_width;
    let theta_decay = (PI * 0.5) * (1.0 - (-underpop_delta).exp());

    let mut u = [[Complex64::new(0.0, 0.0); 3]; 3];
    if survival_active {
        let phase = Complex64::from_polar(1.0, 0.1 * delta_birth + params.gauge_flux * PI);
        let cos_b = theta_b.cos();
        let sin_b = theta_b.sin();
        u[0][0] = Complex64::new(cos_b, 0.0);
        u[0][1] = Complex64::new(0.0, -sin_b);
        u[1][0] = phase * Complex64::new(0.0, -sin_b);
        u[1][1] = phase * Complex64::new(cos_b, 0.0);
        u[2][2] = Complex64::new(1.0, 0.0);
    } else if nbr_occ >= 3.8 {
        let cos_cir = theta_cir.cos();
        let sin_cir = theta_cir.sin();
        let ffs_phase = Complex64::from_polar(1.0, gamma_geom);
        u[0][0] = Complex64::new(1.0, 0.0);
        u[1][1] = Complex64::new(cos_cir, 0.0);
        u[1][2] = Complex64::new(0.0, -sin_cir);
        u[2][1] = ffs_phase * Complex64::new(0.0, -sin_cir);
        u[2][2] = ffs_phase * Complex64::new(cos_cir, 0.0);
    } else {
        let cos_d = theta_decay.cos();
        let sin_d = theta_decay.sin();
        u[0][0] = Complex64::new(cos_d, 0.0);
        u[0][1] = Complex64::new(sin_d, 0.0);
        u[1][0] = Complex64::new(-sin_d, 0.0);
        u[1][1] = Complex64::new(cos_d, 0.0);
        u[2][2] = Complex64::new(1.0, 0.0);
    }
    HolonomyTransferMatrix { elements: u }
}

/// HighLife B36/S23 Topological Replicator Unitary.
/// Resonant births at N=3 and N=6 inducing coherent state duplication.
fn build_highlife_unitary(nbr_occ: f64, params: &HolonomyParams) -> HolonomyTransferMatrix {
    let d3 = nbr_occ - 3.0;
    let d6 = nbr_occ - 6.0;
    let w3 = (-params.birth_sharpness * d3 * d3).exp();
    let w6 = (-params.birth_sharpness * d6 * d6).exp();
    let theta_b = (PI * 0.5) * (w3 + w6).min(1.0);
    let survival_active = (1.8..=3.2).contains(&nbr_occ) || (5.5..=6.5).contains(&nbr_occ);

    let mut u = [[Complex64::new(0.0, 0.0); 3]; 3];
    if survival_active {
        // Replicator coherence phase: splits phase when N ~ 6
        let phase = Complex64::from_polar(1.0, if w6 > 0.5 { PI * 0.5 } else { 0.05 * d3 });
        let cos_b = theta_b.cos();
        let sin_b = theta_b.sin();
        u[0][0] = Complex64::new(cos_b, 0.0);
        u[0][1] = Complex64::new(0.0, -sin_b);
        u[1][0] = phase * Complex64::new(0.0, -sin_b);
        u[1][1] = phase * Complex64::new(cos_b, 0.0);
        u[2][2] = Complex64::new(1.0, 0.0);
    } else if nbr_occ >= 6.8 {
        // Super-overpopulation to FFS
        let theta = (PI * 0.5) * (1.0 - (-(nbr_occ - 6.8).max(0.0)).exp());
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        u[0][0] = Complex64::new(1.0, 0.0);
        u[1][1] = Complex64::new(cos_t, 0.0);
        u[1][2] = Complex64::new(0.0, -sin_t);
        u[2][1] = Complex64::new(0.0, -sin_t);
        u[2][2] = Complex64::new(cos_t, 0.0);
    } else {
        // Underpopulation decay
        let decay = (1.8 - nbr_occ).max(0.0) / params.cir_width;
        let theta_d = (PI * 0.5) * (1.0 - (-decay).exp());
        let cos_d = theta_d.cos();
        let sin_d = theta_d.sin();
        u[0][0] = Complex64::new(cos_d, 0.0);
        u[0][1] = Complex64::new(sin_d, 0.0);
        u[1][0] = Complex64::new(-sin_d, 0.0);
        u[1][1] = Complex64::new(cos_d, 0.0);
        u[2][2] = Complex64::new(1.0, 0.0);
    }
    HolonomyTransferMatrix { elements: u }
}

/// Gross-Pitaevskii Superfluid BEC Non-Linear Unitary.
/// Implements non-linear chemical potential and vortex core phase slips.
fn build_bec_unitary(nbr_occ: f64, params: &HolonomyParams) -> HolonomyTransferMatrix {
    let g = params.non_linear_g;
    let mu = g * (nbr_occ - 3.0);
    let phi_bec = mu * params.cycle_period;
    let theta_tunnel = (PI * 0.25) * (-0.5 * (nbr_occ - 3.0).powi(2)).exp();
    let cos_t = theta_tunnel.cos();
    let sin_t = theta_tunnel.sin();

    let p0 = Complex64::from_polar(1.0, 0.0);
    let p1 = Complex64::from_polar(1.0, phi_bec);
    let p2 = Complex64::from_polar(1.0, 2.0 * phi_bec);

    let mut u = [[Complex64::new(0.0, 0.0); 3]; 3];
    u[0][0] = p0 * Complex64::new(cos_t, 0.0);
    u[0][1] = p0 * Complex64::new(0.0, -sin_t);
    u[1][0] = p1 * Complex64::new(0.0, -sin_t);
    u[1][1] = p1 * Complex64::new(cos_t, 0.0);
    u[2][2] = p2;
    HolonomyTransferMatrix { elements: u }
}

/// Floquet Kicked Chirikov-Harper Quantum Chaos Unitary.
/// Injects non-linear potential V_kick = K * cos(pi * N_bar) and chaotic mixing.
fn build_floquet_chaos_unitary(nbr_occ: f64, params: &HolonomyParams) -> HolonomyTransferMatrix {
    let k = params.chaos_k;
    let v_kick = k * (PI * nbr_occ).cos();
    let theta_mix = (PI * 0.35) * (1.0 / (1.0 + (nbr_occ - 3.0).abs()));
    let cos_m = theta_mix.cos();
    let sin_m = theta_mix.sin();

    // SU(3) Gell-Mann chaotic rotation with kicking phase
    let phase_k = Complex64::from_polar(1.0, v_kick);
    let phase_2k = Complex64::from_polar(1.0, -0.5 * v_kick);

    let mut u = [[Complex64::new(0.0, 0.0); 3]; 3];
    u[0][0] = phase_k * Complex64::new(cos_m, 0.0);
    u[0][1] = phase_k * Complex64::new(0.0, -sin_m);
    u[1][0] = phase_2k * Complex64::new(0.0, -sin_m);
    u[1][1] = phase_2k * Complex64::new(cos_m, 0.0);
    u[2][2] = Complex64::from_polar(1.0, v_kick * 1.5);
    HolonomyTransferMatrix { elements: u }
}

/// Relativistic Dirac-Weyl QCA Unitary.
/// Injects mass term m(N) = m0 * (N - 3) and Zitterbewegung jitter.
fn build_dirac_unitary(nbr_occ: f64, params: &HolonomyParams) -> HolonomyTransferMatrix {
    let mass = 0.5 * (nbr_occ - 3.0);
    let p_eff = 1.0;
    let energy = (p_eff * p_eff + mass * mass).sqrt();
    let theta_dirac = (p_eff / energy).asin();
    let time_phase = energy * params.cycle_period;

    let cos_d = theta_dirac.cos();
    let sin_d = theta_dirac.sin();
    let phase_p = Complex64::from_polar(1.0, -time_phase);
    let phase_m = Complex64::from_polar(1.0, time_phase);

    let mut u = [[Complex64::new(0.0, 0.0); 3]; 3];
    u[0][0] = Complex64::new(1.0, 0.0);
    u[1][1] = phase_p * Complex64::new(cos_d, 0.0);
    u[1][2] = phase_p * Complex64::new(0.0, -sin_d);
    u[2][1] = phase_m * Complex64::new(0.0, -sin_d);
    u[2][2] = phase_m * Complex64::new(cos_d, 0.0);
    HolonomyTransferMatrix { elements: u }
}

/// Cyclic Lotka-Volterra 3-Species Holonomy Unitary.
/// Generates spiral wave reactions: |0> (Vacuum) -> |1> (Prey) -> |2> (Predator) -> |0>.
fn build_cyclic_predator_prey_unitary(nbr_occ: f64, params: &HolonomyParams) -> HolonomyTransferMatrix {
    let rate = (PI / 3.0) * (1.0 / (1.0 + (-params.birth_sharpness * (nbr_occ - 3.0)).exp()));
    let cos_r = rate.cos();
    let sin_r = rate.sin();

    // Circulant 3-way unitary transition: 0 -> 1 -> 2 -> 0
    let phase_gauge = Complex64::from_polar(1.0, params.gauge_flux * 2.0 * PI);
    let mut u = [[Complex64::new(0.0, 0.0); 3]; 3];
    u[0][0] = Complex64::new(cos_r, 0.0);
    u[0][1] = Complex64::new(0.0, -sin_r);
    u[1][1] = Complex64::new(cos_r, 0.0);
    u[1][2] = Complex64::new(0.0, -sin_r);
    u[2][0] = phase_gauge * Complex64::new(0.0, -sin_r);
    u[2][2] = Complex64::new(cos_r, 0.0);

    // Exact Gram-Schmidt orthonormalization to guarantee machine-precision unitarity <= 1e-15
    for c in 0..3 {
        // Orthogonalize against previous columns
        for prev in 0..c {
            let mut dot = Complex64::new(0.0, 0.0);
            for r in 0..3 {
                dot += u[r][prev].conj() * u[r][c];
            }
            for r in 0..3 {
                let sub = dot * u[r][prev];
                u[r][c] -= sub;
            }
        }
        // Normalize column
        let mut norm_sq = 0.0;
        for r in 0..3 {
            norm_sq += u[r][c].norm_sqr();
        }
        let norm = norm_sq.sqrt().max(1e-15);
        for r in 0..3 {
            u[r][c] /= norm;
        }
    }

    HolonomyTransferMatrix { elements: u }
}

