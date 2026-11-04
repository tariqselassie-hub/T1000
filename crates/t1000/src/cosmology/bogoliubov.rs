// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Bogoliubov Transformation & Cosmological Particle Creation
//!
//! Evaluates the time-dependent Bogoliubov coefficients \alpha_k(t) and \beta_k(t)
//! relating the asymptotic in-vacuum to instantaneous field modes:
//!   v_k(eta) = \alpha_k(eta) u_k(eta) + \beta_k(eta) u_k^*(eta)
//!
//! Observables:
//! - Created particle number density: n_k = |\beta_k|^2
//! - Symplectic unitarity: |\alpha_k|^2 - |\beta_k|^2 = 1.0
//! - Squeezing parameter: r_k = asinh(|\beta_k|)
//! - Primordial curvature power spectrum: \mathcal{P}(k) = (k^3 / 2 pi^2) |v_k / a|^2
//! - Gibbons-Hawking horizon temperature: T_GH = H / (2 pi)

use super::field::QuantumFieldMode;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Mode-resolved Bogoliubov transformation and particle creation state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BogoliubovModeReport {
    pub mode_index: usize,
    pub momentum_k: f64,
    pub alpha_re: f64,
    pub alpha_im: f64,
    pub beta_re: f64,
    pub beta_im: f64,
    /// Number of particles created out of vacuum: n_k = |\beta_k|^2
    pub particle_number: f64,
    /// Quantum squeezing parameter r_k = asinh(|\beta_k|)
    pub squeezing_param: f64,
    /// Primordial curvature power spectrum contribution: \mathcal{P}(k)
    pub power_spectrum: f64,
    /// Unitarity error: ||\alpha|^2 - |\beta|^2 - 1.0|
    pub unitarity_error: f64,
}

/// Computes the exact Bogoliubov coefficients for a single field mode at time eta.
pub fn compute_mode_bogoliubov(
    mode: &QuantumFieldMode,
    scale_factor: f64,
    reference_omega: f64,
    eta_phase: f64,
) -> BogoliubovModeReport {
    let omega = reference_omega.max(1e-8);
    let sqrt_omega_half = (omega / 2.0).sqrt();
    let inv_omega = 1.0 / omega;

    // Phase rotation: exp(i * omega * eta)
    let phase_pos = Complex64::from_polar(1.0, omega * eta_phase);
    let phase_neg = Complex64::from_polar(1.0, -omega * eta_phase);

    // Invert for alpha_k and beta_k:
    // alpha_k = sqrt(omega / 2) * (v_k + (i / omega) * pi_k) * exp(i * omega * eta)
    let amp = mode.amplitude();
    let mom = mode.momentum_pi();

    let alpha = Complex64::from(sqrt_omega_half)
        * (amp + Complex64::new(0.0, inv_omega) * mom)
        * phase_pos;

    // beta_k = sqrt(omega / 2) * (v_k - (i / omega) * pi_k) * exp(-i * omega * eta)
    let beta = Complex64::from(sqrt_omega_half)
        * (amp - Complex64::new(0.0, inv_omega) * mom)
        * phase_neg;

    let alpha_norm_sq = alpha.norm_sqr();
    let beta_norm_sq = beta.norm_sqr();
    let n_k = beta_norm_sq;
    let r_k = n_k.sqrt().asinh();

    // Primordial power spectrum: P(k) = (k^3 / (2 pi^2)) * (|v_k|^2 / a^2)
    let k = mode.momentum_k;
    let a = scale_factor.max(1e-6);
    let p_k = (k.powi(3) / (2.0 * std::f64::consts::PI.powi(2))) * (amp.norm_sqr() / (a * a));

    let unitarity_err = (alpha_norm_sq - beta_norm_sq - 1.0).abs();

    BogoliubovModeReport {
        mode_index: mode.mode_index,
        momentum_k: k,
        alpha_re: alpha.re,
        alpha_im: alpha.im,
        beta_re: beta.re,
        beta_im: beta.im,
        particle_number: n_k,
        squeezing_param: r_k,
        power_spectrum: p_k,
        unitarity_error: unitarity_err,
    }
}

/// Instantaneous cosmic spectrum of created particles and perturbations across all modes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CosmicParticleSpectrum {
    pub cosmic_time: f64,
    pub scale_factor: f64,
    pub hubble_parameter: f64,
    pub gibbons_hawking_temp: f64,
    pub total_particle_density: f64,
    pub mean_squeezing: f64,
    pub mode_reports: Vec<BogoliubovModeReport>,
}

impl CosmicParticleSpectrum {
    pub fn evaluate(
        modes: &[QuantumFieldMode],
        field_mass: f64,
        t: f64,
        scale_factor: f64,
        hubble: f64,
        eta: f64,
    ) -> Self {
        let gh_temp = hubble / (2.0 * std::f64::consts::PI);
        let mut reports = Vec::with_capacity(modes.len());
        let mut total_n = 0.0;
        let mut total_r = 0.0;

        for mode in modes {
            let ref_omega = (mode.momentum_k.powi(2) + (field_mass * scale_factor).powi(2)).sqrt().max(1e-6);
            let rep = compute_mode_bogoliubov(mode, scale_factor, ref_omega, eta);
            total_n += rep.particle_number;
            total_r += rep.squeezing_param;
            reports.push(rep);
        }

        let count = modes.len().max(1) as f64;
        Self {
            cosmic_time: t,
            scale_factor,
            hubble_parameter: hubble,
            gibbons_hawking_temp: gh_temp,
            total_particle_density: total_n,
            mean_squeezing: total_r / count,
            mode_reports: reports,
        }
    }
}
