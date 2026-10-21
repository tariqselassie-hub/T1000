// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! The Second Hopf Fibration ($S^7 \to S^4 \cong \mathbb{H}P^1$) & Quaternionic Entanglement Invariants.
//!
//! Maps arbitrary 2-qubit Hilbert space states $|\psi\rangle \in S^7 \subset \mathbb{C}^4 \cong \mathbb{H}^2$
//! onto the 4-sphere $S^4 \subset \mathbb{R}^5$ in quaternionic projective space $\mathbb{H}P^1$.
//!
//! Under the Second Hopf Fibration:
//! - Fiber: $S^3 \cong SU(2)$ corresponds to simultaneous local gauge rotations.
//! - Base: $S^4$ parameterizes the orbit space $\mathbb{C}P^3 / SU(2)$.
//! - The equatorial 3-sphere $y_0 = 0$ corresponds to the maximally entangled Bell manifold ($C = 1$).
//! - The poles $y_0 = \pm 1$ correspond to pure product computational basis states ($C = 0$).
//! - For mixed states, the vector contracts into the 5D Quaternionic Bloch Ball $B^5$ ($\|Y\| \le 1$).

use crate::quaternion::Quaternion;
use serde::{Deserialize, Serialize};

/// 5D coordinate on the quaternionic Bloch sphere $S^4 \subset \mathbb{R}^5$ or inside the Bloch ball $B^5$.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HopfS4Coordinate {
    /// 5th dimensional scalar component: $|q_A|^2 - |q_B|^2 \in [-1, 1]$.
    pub y0: f64,
    /// Quaternionic components: $2 q_A q_B^* \in \mathbb{H}$.
    pub y1: f64,
    pub y2: f64,
    pub y3: f64,
    pub y4: f64,
    /// Radial norm $\|y\| = \sqrt{y_0^2 + y_1^2 + y_2^2 + y_3^2 + y_4^2}$.
    pub radius: f64,
    /// Norm residual from unit 4-sphere: $|\|y\| - 1.0|$.
    pub sphere_residual: f64,
}

impl HopfS4Coordinate {
    /// Creates a 5D coordinate from 5 scalar components.
    pub fn new(y0: f64, y1: f64, y2: f64, y3: f64, y4: f64) -> Self {
        let r_sq = y0 * y0 + y1 * y1 + y2 * y2 + y3 * y3 + y4 * y4;
        let radius = r_sq.sqrt();
        let sphere_residual = (radius - 1.0).abs();
        Self {
            y0,
            y1,
            y2,
            y3,
            y4,
            radius,
            sphere_residual,
        }
    }

    /// Evaluates if coordinate lies on the pure state $S^4$ manifold to within epsilon tolerance.
    #[inline(always)]
    pub fn is_pure_sphere(&self, tol: f64) -> bool {
        self.sphere_residual < tol
    }
}

/// Quantum entanglement invariants evaluated via the Second Hopf Fibration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EntanglementInvariant {
    /// Wootters concurrence $C(\psi) \in [0, 1]$.
    pub concurrence: f64,
    /// Bell-CHSH witness parameter: $S(\psi) = 2 \sqrt{1 + C^2} \in [2.0, 2\sqrt{2}]$.
    pub bell_chsh: f64,
    /// Single-qubit subsystem purity $\gamma = 1 - C^2 / 2 \in [0.5, 1.0]$.
    pub subsystem_purity: f64,
    /// Distance from equatorial Bell plane $y_0 = 0$.
    pub bell_plane_distance: f64,
}

/// Projects a 2-qubit pure state $|\psi\rangle \in S^7$ onto the Second Hopf base $S^4 \subset \mathbb{R}^5$.
///
/// Amplitudes must be ordered: $[|00\rangle, |01\rangle, |10\rangle, |11\rangle]$.
pub fn project_state_s7_to_s4(re: &[f64; 4], im: &[f64; 4]) -> HopfS4Coordinate {
    // Decompose into quaternionic 2-spinor (qA, qB) in H^2 via Cayley-Dickson
    // qA = c0 + c1 * j = (re[0], im[0], re[1], im[1])
    // qB = c2 + c3 * j = (re[2], im[2], re[3], im[3])
    let q_a = Quaternion::from_complex_pair(re[0], im[0], re[1], im[1]);
    let q_b = Quaternion::from_complex_pair(re[2], im[2], re[3], im[3]);

    let na_sq = q_a.norm_sq();
    let nb_sq = q_b.norm_sq();

    // y0 = |qA|^2 - |qB|^2
    let y0 = na_sq - nb_sq;

    // Y_1..4 = 2 * qA * qB^* in H
    let cross = (q_a * q_b.conjugate()).scale(2.0);

    HopfS4Coordinate::new(y0, cross.w, cross.x, cross.y, cross.z)
}

/// Computes exact Wootters concurrence and Bell-CHSH witness for a 2-qubit state.
pub fn compute_entanglement_invariants(re: &[f64; 4], im: &[f64; 4]) -> EntanglementInvariant {
    // c0 * c3 - c1 * c2
    // (re0 + i*im0)(re3 + i*im3) - (re1 + i*im1)(re2 + i*im2)
    let p03_re = re[0] * re[3] - im[0] * im[3];
    let p03_im = re[0] * im[3] + im[0] * re[3];

    let p12_re = re[1] * re[2] - im[1] * im[2];
    let p12_im = re[1] * im[2] + im[1] * re[2];

    let diff_re = p03_re - p12_re;
    let diff_im = p03_im - p12_im;

    let det_mag = (diff_re * diff_re + diff_im * diff_im).sqrt();
    let concurrence = (2.0 * det_mag).min(1.0);

    // Exact maximal Bell-CHSH parameter: S = 2 * sqrt(1 + C^2)
    let bell_chsh = 2.0 * (1.0 + concurrence * concurrence).sqrt();
    let subsystem_purity = (1.0 - 0.5 * concurrence * concurrence).max(0.5);

    let hopf = project_state_s7_to_s4(re, im);

    EntanglementInvariant {
        concurrence,
        bell_chsh,
        subsystem_purity,
        bell_plane_distance: hopf.y0.abs(),
    }
}

/// Generates parametric Bell rotation state $|\psi(\theta)\rangle = \cos(\theta)|00\rangle + \sin(\theta)|11\rangle$.
pub fn bell_rotation_state(theta: f64) -> ([f64; 4], [f64; 4]) {
    let re = [theta.cos(), 0.0, 0.0, theta.sin()];
    let im = [0.0; 4];
    (re, im)
}

/// Generates a Werner mixed state density matrix $\rho_W(p) = p |\Phi^+\rangle\langle\Phi^+| + \frac{1-p}{4} I_4$,
/// returning its 5D Quaternionic Bloch Ball vector $\vec{Y} \in B^5$ and analytical concurrence.
pub fn evaluate_werner_state(p: f64) -> (HopfS4Coordinate, EntanglementInvariant) {
    let p_clamped = p.clamp(0.0, 1.0);
    // For Werner state, the quaternionic vector contracts linearly with p:
    // Y = (0, 0, 0, -p, 0), with ||Y|| = p <= 1
    let hopf = HopfS4Coordinate::new(0.0, 0.0, 0.0, -p_clamped, 0.0);

    // Concurrence C(p) = max(0, (3p - 1) / 2)
    let concurrence = ((3.0 * p_clamped - 1.0) * 0.5).max(0.0);
    let bell_chsh = if p_clamped > 1.0 / std::f64::consts::SQRT_2 {
        2.0 * std::f64::consts::SQRT_2 * p_clamped
    } else {
        2.0
    };
    let subsystem_purity = 0.5 * (1.0 + p_clamped * p_clamped);

    let inv = EntanglementInvariant {
        concurrence,
        bell_chsh,
        subsystem_purity,
        bell_plane_distance: 0.0,
    };

    (hopf, inv)
}

/// Evaluates pairwise $S^4$ coordinates across an $N$-qubit statevector for qubit pair $(q_0, q_1)$.
pub fn compute_qubit_pair_hopf_s4(
    state_re: &[f64],
    state_imag: &[f64],
    q0: usize,
    q1: usize,
    n_qubits: usize,
) -> (HopfS4Coordinate, EntanglementInvariant) {
    let dim = 1usize << n_qubits;
    assert_eq!(state_re.len(), dim);
    assert_eq!(state_imag.len(), dim);
    assert!(q0 < n_qubits && q1 < n_qubits && q0 != q1);

    // Compute 2-qubit reduced density matrix rho_{00..11, 00..11} via partial trace
    let mut rho_re = [[0.0f64; 4]; 4];
    let mut rho_im = [[0.0f64; 4]; 4];

    for idx in 0..dim {
        let b0 = (idx >> q0) & 1;
        let b1 = (idx >> q1) & 1;
        let pair_idx_row = (b0 << 1) | b1;

        let re_i = state_re[idx];
        let im_i = state_imag[idx];

        for pair_col in 0..4 {
            let c0 = (pair_col >> 1) & 1;
            let c1 = pair_col & 1;

            let col_idx = (idx & !(1 << q0) & !(1 << q1)) | (c0 << q0) | (c1 << q1);
            let re_j = state_re[col_idx];
            let im_j = state_imag[col_idx];

            rho_re[pair_idx_row][pair_col] += re_i * re_j + im_i * im_j;
            rho_im[pair_idx_row][pair_col] += im_i * re_j - re_i * im_j;
        }
    }

    // Extract diagonal probabilities
    let p00 = rho_re[0][0];
    let p01 = rho_re[1][1];
    let p10 = rho_re[2][2];
    let p11 = rho_re[3][3];

    // Quaternionic Bloch coordinates from 2-qubit density matrix
    let y0 = (p00 + p01) - (p10 + p11);
    let y1 = 2.0 * rho_re[0][3];
    let y2 = 2.0 * rho_im[0][3];
    let y3 = 2.0 * rho_re[1][2];
    let y4 = 2.0 * rho_im[1][2];

    let coord = HopfS4Coordinate::new(y0, y1, y2, y3, y4);

    // 2-qubit concurrence proxy
    let p_offdiag = (rho_re[0][3] * rho_re[0][3] + rho_im[0][3] * rho_im[0][3]).sqrt();
    let p_inner = (rho_re[1][2] * rho_re[1][2] + rho_im[1][2] * rho_im[1][2]).sqrt();
    let concurrence = (2.0 * (p_offdiag - (p01 * p10).sqrt()).max(p_inner - (p00 * p11).sqrt())).clamp(0.0, 1.0);

    let bell_chsh = 2.0 * (1.0 + concurrence * concurrence).sqrt();
    let purity = rho_re.iter().flatten().map(|v| v * v).sum::<f64>()
        + rho_im.iter().flatten().map(|v| v * v).sum::<f64>();

    let inv = EntanglementInvariant {
        concurrence,
        bell_chsh,
        subsystem_purity: purity.min(1.0),
        bell_plane_distance: y0.abs(),
    };

    (coord, inv)
}

/// Computes the 2026 piecewise Planckian dissipation rate bound:
/// $\Gamma_{\max}(T, \Delta) = \min\left(\frac{2 k_B T}{\hbar}, \frac{\Delta}{\hbar}\right)$
/// and returns the exact lower-bound survival retention factor $\exp(-\Gamma_{\max} \cdot \delta t)$.
///
/// In natural units ($\hbar = 1, k_B = 1$):
/// - High-T Planckian rate: $2 k_B T / \hbar = 2 T$
/// - Low-T Gapped adiabatic limit: $\Delta / \hbar = \Delta$
/// - Piecewise crossover at $\beta \Delta = 2$ ($2 T = \Delta$).
#[inline(always)]
pub fn planckian_damping_step(
    temp_kbt: f64,      // Thermal energy scale (k_B * T) in natural units (\hbar = 1)
    spectral_gap: f64,  // Lowest excitation gap \Delta
    dt: f64,            // Discrete evolution step \delta t
) -> (f64, f64) {
    let rate_planckian = (2.0 * temp_kbt).max(0.0);
    let rate_gapped = spectral_gap.max(0.0);
    let gamma_bound = rate_planckian.min(rate_gapped);
    let min_retention_factor = (-gamma_bound * dt.max(0.0)).exp();
    (gamma_bound, min_retention_factor)
}

/// Applies Planckian-bounded contraction to a 5D Bloch ball ($B^5$) state vector.
///
/// Returns the new contracted $B^5$ radius $r(t + \delta t) \le 1.0$.
#[inline(always)]
pub fn damp_b5_bloch_vector(
    b5_vec: &mut [f64; 5],
    temp_kbt: f64,
    spectral_gap: f64,
    dt: f64,
) -> f64 {
    let (_gamma, retention) = planckian_damping_step(temp_kbt, spectral_gap, dt);
    let mut norm_sq = 0.0;
    for comp in b5_vec.iter_mut() {
        *comp *= retention;
        norm_sq += (*comp) * (*comp);
    }
    norm_sq.sqrt()
}

/// Evolves a Werner mixed state under Planckian-bounded dissipation over interval $\delta t$.
///
/// Initial mixing parameter $p_0$ contracts as $p(t + \delta t) = p_0 \cdot \exp(-\Gamma_{\max} \cdot \delta t)$.
pub fn damp_werner_state_planckian(
    p_initial: f64,
    temp_kbt: f64,
    spectral_gap: f64,
    dt: f64,
) -> (HopfS4Coordinate, EntanglementInvariant) {
    let (_gamma, retention) = planckian_damping_step(temp_kbt, spectral_gap, dt);
    let p_decayed = (p_initial.clamp(0.0, 1.0) * retention).clamp(0.0, 1.0);
    evaluate_werner_state(p_decayed)
}

/// Verifies whether a discrete state evolution step obeys the Bures / Fubini-Study geodesic speed limit:
/// $\Delta \theta_{\text{step}} \le \delta t \cdot \min\left(\frac{\pi k_B T}{\hbar}, \frac{\Delta}{\hbar}\right)$.
///
/// Returns `true` if causal Planckian speed limit is respected; `false` on Planckian causality violation.
#[inline(always)]
pub fn verify_planckian_bures_bound(
    theta_step: f64,
    dt: f64,
    temp_kbt: f64,
    spectral_gap: f64,
) -> bool {
    if dt <= 0.0 {
        return theta_step.abs() < 1e-12;
    }
    let max_angular_velocity = (std::f64::consts::PI * temp_kbt.max(0.0)).min(spectral_gap.max(0.0));
    let allowable_theta = max_angular_velocity * dt;
    theta_step.abs() <= allowable_theta + 1e-12
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_1_SQRT_2, PI};

    #[test]
    fn test_computational_basis_poles() {
        // State |00>: y0 = 1.0, cross = 0 -> North Pole
        let re_00 = [1.0, 0.0, 0.0, 0.0];
        let im = [0.0; 4];
        let h_00 = project_state_s7_to_s4(&re_00, &im);
        assert!((h_00.y0 - 1.0).abs() < 1e-12);
        assert!(h_00.is_pure_sphere(1e-12));

        let inv_00 = compute_entanglement_invariants(&re_00, &im);
        assert!(inv_00.concurrence < 1e-12);
        assert!((inv_00.bell_chsh - 2.0).abs() < 1e-12);
        assert!((inv_00.subsystem_purity - 1.0).abs() < 1e-12);

        // State |11>: y0 = -1.0, cross = 0 -> South Pole
        let re_11 = [0.0, 0.0, 0.0, 1.0];
        let h_11 = project_state_s7_to_s4(&re_11, &im);
        assert!((h_11.y0 - (-1.0)).abs() < 1e-12);
        assert!(h_11.is_pure_sphere(1e-12));
    }

    #[test]
    fn test_bell_states_equator_and_tsirelson() {
        let im = [0.0; 4];

        // Bell State |Phi+> = 1/√2 (|00> + |11>)
        let re_phi_plus = [FRAC_1_SQRT_2, 0.0, 0.0, FRAC_1_SQRT_2];
        let h_phi = project_state_s7_to_s4(&re_phi_plus, &im);

        // Lies strictly on equatorial S3: y0 = 0
        assert!(h_phi.y0.abs() < 1e-12);
        assert!(h_phi.is_pure_sphere(1e-12));

        let inv_phi = compute_entanglement_invariants(&re_phi_plus, &im);
        assert!((inv_phi.concurrence - 1.0).abs() < 1e-12);
        let tsirelson = 2.0 * std::f64::consts::SQRT_2;
        assert!((inv_phi.bell_chsh - tsirelson).abs() < 1e-12);
        assert!((inv_phi.subsystem_purity - 0.5).abs() < 1e-12);

        // Bell State |Psi+> = 1/√2 (|01> + |10>)
        let re_psi_plus = [0.0, FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0];
        let h_psi = project_state_s7_to_s4(&re_psi_plus, &im);
        assert!(h_psi.y0.abs() < 1e-12);
        assert!(h_psi.is_pure_sphere(1e-12));

        let inv_psi = compute_entanglement_invariants(&re_psi_plus, &im);
        assert!((inv_psi.concurrence - 1.0).abs() < 1e-12);
        assert!((inv_psi.bell_chsh - tsirelson).abs() < 1e-12);
    }

    #[test]
    fn test_continuous_bell_rotation() {
        for step in 0..=20 {
            let theta = (step as f64) * (PI * 0.25) / 20.0;
            let (re, im) = bell_rotation_state(theta);
            let h = project_state_s7_to_s4(&re, &im);
            assert!(h.is_pure_sphere(1e-12));

            let inv = compute_entanglement_invariants(&re, &im);
            let expected_c = (2.0 * theta).sin();
            assert!((inv.concurrence - expected_c).abs() < 1e-12);
        }
    }

    #[test]
    fn test_werner_state_contraction() {
        // Pure Bell state: p = 1.0
        let (h_pure, inv_pure) = evaluate_werner_state(1.0);
        assert!((h_pure.radius - 1.0).abs() < 1e-12);
        assert!((inv_pure.concurrence - 1.0).abs() < 1e-12);

        // Separable threshold: p = 1/3 -> C = 0
        let (_h_sep, inv_sep) = evaluate_werner_state(1.0 / 3.0);
        assert!(inv_sep.concurrence < 1e-12);

        // Maximally mixed state: p = 0.0 -> origin
        let (h_mixed, inv_mixed) = evaluate_werner_state(0.0);
        assert!(h_mixed.radius < 1e-12);
        assert!(inv_mixed.concurrence < 1e-12);
    }

    #[test]
    fn test_planckian_piecewise_crossover_and_scaling() {
        let spectral_gap = 1.0; // \Delta = 1.0

        // 1. Low-T regime: 2 * T < \Delta (e.g. T = 0.25 -> 2T = 0.5 < 1.0)
        let temp_low = 0.25;
        let (gamma_low, ret_low) = planckian_damping_step(temp_low, spectral_gap, 0.1);
        assert!((gamma_low - 0.5).abs() < 1e-12);
        assert!((ret_low - (-0.05f64).exp()).abs() < 1e-12);

        // 2. Crossover point: 2 * T = \Delta (T = 0.5 -> \beta \Delta = 2)
        let temp_cross = 0.5;
        let (gamma_cross, _) = planckian_damping_step(temp_cross, spectral_gap, 0.1);
        assert!((gamma_cross - 1.0).abs() < 1e-12);

        // 3. High-T regime: 2 * T > \Delta (e.g. T = 2.0 -> 2T = 4.0 > 1.0, saturates at \Delta)
        let temp_high = 2.0;
        let (gamma_high, ret_high) = planckian_damping_step(temp_high, spectral_gap, 0.1);
        assert!((gamma_high - 1.0).abs() < 1e-12);
        assert!((ret_high - (-0.1f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn test_b5_planckian_contraction_and_werner() {
        // Pure Bell vector on S4: radius = 1.0
        let mut b5 = [0.0, 0.0, 0.0, -1.0, 0.0];
        let r_init = damp_b5_bloch_vector(&mut b5, 1.0, 0.8, 0.0);
        assert!((r_init - 1.0).abs() < 1e-12);

        // Step under Planckian dissipation: gamma = min(2 * 0.3, 0.8) = 0.6
        let r_step = damp_b5_bloch_vector(&mut b5, 0.3, 0.8, 1.0);
        let expected_r = (-0.6f64).exp();
        assert!((r_step - expected_r).abs() < 1e-12);
        assert!(r_step < 1.0);

        // Werner state Planckian damping
        let (h_w, inv_w) = damp_werner_state_planckian(1.0, 0.3, 0.8, 1.0);
        assert!((h_w.radius - expected_r).abs() < 1e-12);
        assert!(inv_w.concurrence < 1.0);
    }

    #[test]
    fn test_planckian_bures_speed_limit_guardrail() {
        let temp = 0.5;
        let gap = 1.0;
        let dt = 0.1;
        // max_angular_velocity = min(pi * 0.5, 1.0) = 1.0
        // max_step = 1.0 * 0.1 = 0.1 rad

        // Sub-Planckian rotation: 0.05 rad <= 0.1 rad -> valid
        assert!(verify_planckian_bures_bound(0.05, dt, temp, gap));

        // Exact boundary: 0.10 rad <= 0.1 rad -> valid
        assert!(verify_planckian_bures_bound(0.10, dt, temp, gap));

        // Super-Planckian scrambling violation: 0.25 rad > 0.1 rad -> violation
        assert!(!verify_planckian_bures_bound(0.25, dt, temp, gap));
    }
}
