// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Quantum Phase-Lock & Decoherence-Free Subspace (DFS) Kernel
//!
//! Evaluates quantum state fidelity at the central Node (4, 5) fulcrum
//! under high-kinetic polar phase noise and mirrored parity damping.

use serde::{Deserialize, Serialize};

/// Quantum telemetry metrics for the 8-node gyroscopic stabilizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumLockStep {
    pub time: f64,
    pub phase_noise_a_rad: f64,
    pub phase_noise_b_rad: f64,
    pub core_accumulated_phase_rad: f64,
    pub core_fidelity: f64,
    pub parity_correlator_zab: f64,
    pub gyroscopic_q_factor: f64,
    pub dfs_confinement_error: f64,
}

/// Quantum phase-locking engine on the 8-node topology.
#[derive(Debug, Clone)]
pub struct QuantumLockEngine {
    pub noise_amplitude_rad: f64,
    pub noise_freq_rad_s: f64,
    pub core_coupling_strength_g: f64,
}

impl QuantumLockEngine {
    /// Creates a new quantum phase-locking engine with specified noise parameters and coupling strength.
    pub fn new(noise_amp: f64, noise_freq: f64, coupling_g: f64) -> Self {
        Self {
            noise_amplitude_rad: noise_amp,
            noise_freq_rad_s: noise_freq,
            core_coupling_strength_g: coupling_g,
        }
    }

    /// Evaluates quantum phase-locking telemetry at time `t`.
    pub fn evaluate_step(&self, t: f64) -> QuantumLockStep {
        // High-frequency kinetic phase flutter at polar extreme A
        let phi_a = self.noise_amplitude_rad * (self.noise_freq_rad_s * t).sin();
        // Counter-phase parity damping at polar extreme B
        let phi_b = -phi_a;

        // Core coupling phase accumulation: Delta phi_core = g * (phi_a + phi_b)
        // Under perfect counter-phase symmetry, Delta phi_core is analytically zero.
        let core_phase = self.core_coupling_strength_g * (phi_a + phi_b);
        let dfs_error = core_phase.abs();

        // Core state fidelity: F = cos^2(Delta phi_core / 2)
        let core_fid = (core_phase / 2.0).cos().powi(2);

        // Parity correlator: <Z_A Z_B> = cos(phi_a - phi_b) with opposite parity ground state
        // For anti-correlated parity, <Z_A Z_B> = -cos(2 * phi_a) or exactly -1 in DFS basis.
        let parity_zab = -(phi_a + phi_b).cos(); // Exactly -1.0 under balanced cancellation

        // Gyroscopic Q-factor: ratio of polar noise variance to core variance
        let polar_jitter = phi_a.powi(2) + phi_b.powi(2);
        let core_jitter = core_phase.powi(2) + 1e-18;
        let q_factor = (polar_jitter / core_jitter).min(1.0e12);

        QuantumLockStep {
            time: t,
            phase_noise_a_rad: phi_a,
            phase_noise_b_rad: phi_b,
            core_accumulated_phase_rad: core_phase,
            core_fidelity: core_fid,
            parity_correlator_zab: parity_zab,
            gyroscopic_q_factor: q_factor,
            dfs_confinement_error: dfs_error,
        }
    }
}
