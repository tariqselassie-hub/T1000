// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Optical Waveguide Loss & Hardware Non-Ideality Engine
//!
//! Models physical non-idealities in integrated photonic circuits:
//! - Waveguide propagation loss $\alpha$ in $\text{dB/cm}$ ($P(L) = P_0 10^{-\alpha L / 10}$).
//! - Directional coupler splitting ratio deviation ($50:50 \to (0.5+\epsilon):(0.5-\epsilon)$).
//! - Thermal and electro-optic phase jitter $\sigma_\phi$.

use serde::{Deserialize, Serialize};

/// Physical loss and fabrication tolerance model for integrated optical circuits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WaveguideLossModel {
    /// Propagation attenuation $\alpha$ in dB/cm (e.g. 1.5 dB/cm for strip SOI, 0.2 dB/cm for TFLN).
    pub alpha_db_cm: f64,
    /// Total waveguide routing propagation distance in cm.
    pub total_length_cm: f64,
    /// Directional coupler splitting ratio tolerance $\epsilon$ (e.g. 0.01 for $\pm 1\%$ split imbalance).
    pub coupler_imbalance: f64,
    /// Static insertion loss per directional coupler in dB (e.g. 0.05 dB).
    pub coupler_insertion_loss_db: f64,
}

impl Default for WaveguideLossModel {
    fn default() -> Self {
        Self::lossless()
    }
}

impl WaveguideLossModel {
    /// Ideal, lossless physical model.
    pub fn lossless() -> Self {
        Self {
            alpha_db_cm: 0.0,
            total_length_cm: 0.0,
            coupler_imbalance: 0.0,
            coupler_insertion_loss_db: 0.0,
        }
    }

    /// Typical Silicon-on-Insulator (SOI) fabrication process parameters.
    pub fn silicon_on_insulator() -> Self {
        Self {
            alpha_db_cm: 1.8,                  // 1.8 dB/cm typical strip waveguide loss
            total_length_cm: 0.5,              // 5 mm total routing on chip
            coupler_imbalance: 0.015,          // 1.5% fabrication split imbalance
            coupler_insertion_loss_db: 0.08,   // 0.08 dB per directional coupler
        }
    }

    /// Low-loss Thin-Film Lithium Niobate (TFLN) process parameters.
    pub fn thin_film_lithium_niobate() -> Self {
        Self {
            alpha_db_cm: 0.25,                 // Ultra-low 0.25 dB/cm propagation loss
            total_length_cm: 1.2,              // 12 mm routing
            coupler_imbalance: 0.008,          // High precision 0.8% coupler split tolerance
            coupler_insertion_loss_db: 0.03,   // 0.03 dB insertion loss
        }
    }

    /// Calculates total transmission power fraction $T = P_{\text{out}} / P_{\text{in}} \in [0, 1]$.
    pub fn total_power_transmission(&self, n_couplers: usize) -> f64 {
        let prop_loss_db = self.alpha_db_cm * self.total_length_cm;
        let coupler_loss_db = self.coupler_insertion_loss_db * n_couplers as f64;
        let total_loss_db = prop_loss_db + coupler_loss_db;

        10.0f64.powf(-total_loss_db / 10.0)
    }

    /// Calculates optical field amplitude transmission factor $\eta = \sqrt{T}$.
    pub fn field_transmission_factor(&self, n_couplers: usize) -> f64 {
        self.total_power_transmission(n_couplers).sqrt()
    }

    /// Applies 2x2 directional coupler with splitting imbalance $\epsilon$ and insertion loss.
    ///
    /// Ideal: $H_2 = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$.
    /// Imbalanced: transmission $t = \sqrt{0.5 + \epsilon}$, reflection $r = \sqrt{0.5 - \epsilon}$.
    pub fn imbalanced_coupler(&self) -> [[f64; 2]; 2] {
        let eta = 10.0f64.powf(-self.coupler_insertion_loss_db / 20.0);
        let t = (0.5 + self.coupler_imbalance).clamp(0.0, 1.0).sqrt();
        let r = (0.5 - self.coupler_imbalance).clamp(0.0, 1.0).sqrt();

        [
            [eta * t, eta * r],
            [eta * r, -eta * t],
        ]
    }
}
