// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Interaction Holonomy & Generalized Exclusion Statistics (GES) Engine.
//! Implements unitary population transfer and geometric phase accumulation
//! via Confinement-Induced Resonances (CIR) based on the Lieb-Liniger holonomy cycle:
//! g: 0 -> +inf -> -inf -> 0, generating Fractional Fermi Seas (FFS) with alpha = l.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Parameters defining the interaction holonomy cycle across the CIR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolonomyParams {
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
}

impl Default for HolonomyParams {
    fn default() -> Self {
        Self {
            l_quantum: 2,
            g0: 1.0,
            cir_width: 0.5,
            cycle_period: 1.0,
            birth_sharpness: 2.0,
        }
    }
}

/// Unitary 3x3 local transition matrix derived from the interaction holonomy.
#[derive(Debug, Clone, PartialEq)]
pub struct HolonomyTransferMatrix {
    /// Matrix elements U[row][col] mapping (c0, c1, c2) -> (c0', c1', c2').
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
/// neighborhood occupation count `nbr_occ`.
///
/// Conway Rules Mapping:
/// - Birth: N in [2.5, 3.5] -> Resonant rotation |0> <-> |1> with angle theta_b.
/// - Survival: N in [1.8, 3.2] for |1> -> Identity evolution with topological phase.
/// - Overpopulation: N >= 3.8 -> CIR driven holonomy pumping |1> <-> |2> with phase gamma_l.
/// - Underpopulation: N < 1.8 -> Non-resonant rotation |1> -> |0>.
pub fn build_holonomy_unitary(nbr_occ: f64, params: &HolonomyParams) -> HolonomyTransferMatrix {
    let gamma_geom = geometric_holonomy_phase(params.l_quantum);

    // Birth coupling weight: Gaussian centered at N = 3.0
    let delta_birth = nbr_occ - 3.0;
    let birth_weight = (-params.birth_sharpness * delta_birth * delta_birth).exp();
    let theta_b = (PI * 0.5) * birth_weight;

    // Survival gate: active for N in [1.8, 3.2]
    let survival_active = (1.8..=3.2).contains(&nbr_occ);

    // Overpopulation CIR coupling weight: activates when N >= 3.8
    let cir_delta = (nbr_occ - 4.0).max(0.0) / params.cir_width;
    let theta_cir = (PI * 0.5) * (1.0 - (-cir_delta).exp());

    // Underpopulation decay: activates when N < 1.8
    let underpop_delta = (1.8 - nbr_occ).max(0.0) / params.cir_width;
    let theta_decay = (PI * 0.5) * (1.0 - (-underpop_delta).exp());

    let mut u = [[Complex64::new(0.0, 0.0); 3]; 3];

    if survival_active {
        // Survival manifold: preserve living population with adiabatic phase
        let phase = Complex64::from_polar(1.0, 0.1 * delta_birth);
        let cos_b = theta_b.cos();
        let sin_b = theta_b.sin();
        u[0][0] = Complex64::new(cos_b, 0.0);
        u[0][1] = Complex64::new(0.0, -sin_b);
        u[1][0] = phase * Complex64::new(0.0, -sin_b);
        u[1][1] = phase * Complex64::new(cos_b, 0.0);
        u[2][2] = Complex64::new(1.0, 0.0);
    } else if nbr_occ >= 3.8 {
        // Overpopulation: CIR holonomy coupling |1> <-> |2>
        let cos_cir = theta_cir.cos();
        let sin_cir = theta_cir.sin();
        let ffs_phase = Complex64::from_polar(1.0, gamma_geom);

        u[0][0] = Complex64::new(1.0, 0.0);
        u[1][1] = Complex64::new(cos_cir, 0.0);
        u[1][2] = Complex64::new(0.0, -sin_cir);
        u[2][1] = ffs_phase * Complex64::new(0.0, -sin_cir);
        u[2][2] = ffs_phase * Complex64::new(cos_cir, 0.0);
    } else {
        // Underpopulation: Decay |1> -> |0>
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
