// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Cosmological Entanglement Horizon & Holographic Area Law
//!
//! Evaluates the quantum entanglement entropy generated across the cosmological Hubble
//! horizon as expanding spacetime freezes modes outside the causal wedge (k < a H).
//!
//! For each mode k undergoing parametric squeezing by the expanding metric, tracing out
//! super-horizon partners yields a thermal state with von Neumann entanglement entropy:
//!   S_vN(r_k) = cosh^2(r_k) * ln(cosh^2(r_k)) - sinh^2(r_k) * ln(sinh^2(r_k))
//! where r_k = asinh(|\beta_k|) is the cosmological squeezing parameter.

use super::bogoliubov::BogoliubovModeReport;
use super::metric::CosmologicalMetric;
use serde::{Deserialize, Serialize};

/// Telemetry regarding the cosmological causal horizon and entanglement entropy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HorizonEntanglementReport {
    pub cosmic_time: f64,
    pub scale_factor: f64,
    pub comoving_horizon_radius: f64,
    pub physical_horizon_area: f64,
    pub sub_horizon_modes_count: usize,
    pub super_horizon_modes_count: usize,
    /// Total von Neumann entanglement entropy summed over all horizon-exited modes
    pub total_entanglement_entropy: f64,
    /// Entropy per horizon boundary area: S / Area (evaluating area-law constancy)
    pub entropy_to_area_ratio: f64,
}

/// Computes the exact single-mode von Neumann entanglement entropy from squeezing r.
pub fn mode_entanglement_entropy(squeezing_r: f64) -> f64 {
    if squeezing_r < 1e-9 {
        return 0.0;
    }
    let cosh_sq = squeezing_r.cosh().powi(2);
    let sinh_sq = squeezing_r.sinh().powi(2);

    let term1 = cosh_sq * cosh_sq.ln();
    let term2 = if sinh_sq > 1e-12 {
        sinh_sq * sinh_sq.ln()
    } else {
        0.0
    };

    (term1 - term2).max(0.0)
}

/// Evaluates horizon crossing and entanglement entropy across all modes.
pub fn evaluate_horizon_entanglement(
    metric: &CosmologicalMetric,
    t: f64,
    mode_reports: &[BogoliubovModeReport],
) -> HorizonEntanglementReport {
    let a = metric.scale_factor(t);
    let r_c = metric.comoving_horizon(t);
    let r_p = metric.physical_horizon(t);
    let area_p = 4.0 * std::f64::consts::PI * r_p.powi(2);

    let mut sub_count = 0;
    let mut super_count = 0;
    let mut total_s = 0.0;

    for rep in mode_reports {
        let is_super = metric.is_super_horizon(rep.momentum_k, t);
        if is_super {
            super_count += 1;
            total_s += mode_entanglement_entropy(rep.squeezing_param);
        } else {
            sub_count += 1;
        }
    }

    let ratio = if area_p > 1e-9 && area_p < 1e11 {
        total_s / area_p
    } else {
        0.0
    };

    HorizonEntanglementReport {
        cosmic_time: t,
        scale_factor: a,
        comoving_horizon_radius: r_c,
        physical_horizon_area: area_p,
        sub_horizon_modes_count: sub_count,
        super_horizon_modes_count: super_count,
        total_entanglement_entropy: total_s,
        entropy_to_area_ratio: ratio,
    }
}

