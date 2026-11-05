// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Discretized Quantum Scalar Field & Mode Symplectic Evolution
//!
//! Models quantum field fluctuations v_k(eta) evolving on an expanding FLRW background.
//! Each spatial Fourier mode k satisfies the Mukhanov-Sasaki parametric oscillator equation:
//!   v_k'' + \Omega_k^2(eta) v_k = 0
//! with time-dependent frequency:
//!   \Omega_k^2(eta) = k^2 + m^2 a^2(eta) - a''(eta) / a(eta)
//!
//! Preserves the microscopic canonical commutation relations via the Wronskian invariant:
//!   W[v_k, v_k^*] = v_k (v_k')^* - v_k^* v_k' = i.

use super::metric::CosmologicalMetric;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// A single quantum field Fourier momentum mode in separated real/imaginary components.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantumFieldMode {
    pub mode_index: usize,
    /// Physical comoving momentum k = 2 * pi * n / L
    pub momentum_k: f64,
    /// Real and imaginary components of complex mode amplitude v_k
    pub amplitude_re: f64,
    pub amplitude_im: f64,
    /// Real and imaginary components of conjugate momentum pi_k = v_k' = dv_k / deta
    pub momentum_pi_re: f64,
    pub momentum_pi_im: f64,
    /// Effective instantaneous frequency Omega_k
    pub instantaneous_omega: f64,
}

impl QuantumFieldMode {
    /// Initializes a mode in the canonical Bunch-Davies quantum vacuum state.
    pub fn new_bunch_davies(
        mode_index: usize,
        momentum_k: f64,
        mass: f64,
        initial_scale: f64,
    ) -> Self {
        let omega = (momentum_k.powi(2) + (mass * initial_scale).powi(2))
            .sqrt()
            .max(1e-8);
        // Bunch-Davies vacuum condition: v_k = 1 / sqrt(2 * omega), pi_k = -i * sqrt(omega / 2)
        let v0_re = 1.0 / (2.0 * omega).sqrt();
        let pi0_im = -(omega / 2.0).sqrt();

        Self {
            mode_index,
            momentum_k,
            amplitude_re: v0_re,
            amplitude_im: 0.0,
            momentum_pi_re: 0.0,
            momentum_pi_im: pi0_im,
            instantaneous_omega: omega,
        }
    }

    /// Returns the complex amplitude v_k.
    pub fn amplitude(&self) -> Complex64 {
        Complex64::new(self.amplitude_re, self.amplitude_im)
    }

    /// Returns the complex conjugate momentum pi_k.
    pub fn momentum_pi(&self) -> Complex64 {
        Complex64::new(self.momentum_pi_re, self.momentum_pi_im)
    }

    /// Computes the microscopic Wronskian invariant: W = v_k * pi_k^* - v_k^* * pi_k.
    /// In an exact quantum unitary evolution, W must equal Complex64::new(0.0, 1.0).
    pub fn wronskian(&self) -> Complex64 {
        let v = self.amplitude();
        let pi = self.momentum_pi();
        v * pi.conj() - v.conj() * pi
    }

    /// Evaluates the deviation from canonical Wronskian conservation: |Im(W) - 1.0|.
    pub fn wronskian_unitarity_error(&self) -> f64 {
        let w = self.wronskian();
        let re_err = w.re.abs();
        let im_err = (w.im - 1.0).abs();
        re_err.max(im_err)
    }

    /// Advances the mode by a conformal time step d_eta using symplectic leapfrog / Störmer-Verlet.
    pub fn step_symplectic(&mut self, d_eta: f64, omega_sq: f64) {
        let half_dt = 0.5 * d_eta;

        // Kick: pi = pi - 0.5 * d_eta * omega^2 * v
        let kick_factor = half_dt * omega_sq;
        self.momentum_pi_re -= kick_factor * self.amplitude_re;
        self.momentum_pi_im -= kick_factor * self.amplitude_im;

        // Drift: v = v + d_eta * pi
        self.amplitude_re += d_eta * self.momentum_pi_re;
        self.amplitude_im += d_eta * self.momentum_pi_im;

        // Kick: pi = pi - 0.5 * d_eta * omega^2 * v
        self.momentum_pi_re -= kick_factor * self.amplitude_re;
        self.momentum_pi_im -= kick_factor * self.amplitude_im;

        self.instantaneous_omega = if omega_sq > 0.0 { omega_sq.sqrt() } else { 0.0 };
    }
}

/// Discretized spatial quantum scalar field lattice across a spectrum of momentum modes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscretizedQuantumField {
    pub box_length: f64,
    pub field_mass: f64,
    pub modes: Vec<QuantumFieldMode>,
}

impl DiscretizedQuantumField {
    /// Constructs a discretized field with `num_modes` spanning momentum space from k_min to k_max.
    pub fn new(num_modes: usize, box_length: f64, field_mass: f64, initial_scale: f64) -> Self {
        assert!(
            num_modes >= 2,
            "Field lattice requires at least 2 spatial modes"
        );
        let mut modes = Vec::with_capacity(num_modes);
        for n in 1..=num_modes {
            let k = 2.0 * std::f64::consts::PI * (n as f64) / box_length;
            modes.push(QuantumFieldMode::new_bunch_davies(
                n,
                k,
                field_mass,
                initial_scale,
            ));
        }

        Self {
            box_length,
            field_mass,
            modes,
        }
    }

    /// Steps all modes through conformal time d_eta under the expanding cosmological metric.
    pub fn step_field(&mut self, metric: &CosmologicalMetric, t_current: f64, d_eta: f64) {
        let v_eff = metric.effective_potential(t_current, self.field_mass);
        for mode in &mut self.modes {
            let omega_sq = mode.momentum_k.powi(2) + v_eff;
            mode.step_symplectic(d_eta, omega_sq);
        }
    }

    /// Returns the maximum Wronskian unitarity error across all active momentum modes.
    pub fn max_unitarity_error(&self) -> f64 {
        self.modes
            .iter()
            .map(|m| m.wronskian_unitarity_error())
            .fold(0.0_f64, f64::max)
    }
}
