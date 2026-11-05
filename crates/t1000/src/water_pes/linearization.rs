// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Water 180° Linearization Barrier & Walsh Diagram Engine
//!
//! Models the planar linearization transition of water:
//! - Evaluates the 180° barrier: Delta E_lin = E(180°) - E(104.5°).
//! - Analyzes Walsh orbital re-hybridization: 3a1 and 1b1 converge to degenerate 1pi_u at 180°.
//! - Tracks dipole moment collapse: |mu| -> 0 as C_2v elevates to centrosymmetric D_inf_h.

use super::surface::{evaluate_water_point, DEFAULT_THETA_EQUILIBRIUM};
use serde::{Deserialize, Serialize};

/// Energy conversion constants.
pub const HARTREE_TO_EV: f64 = 27.211386;
pub const HARTREE_TO_KCAL_MOL: f64 = 627.509474;
pub const HARTREE_TO_CM1: f64 = 219474.63;

/// Single point along the Walsh molecular orbital correlation path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WalshOrbitalState {
    pub theta_deg: f64,
    pub r_oh: f64,
    pub total_energy: f64,
    pub orbital_3a1: f64,
    pub orbital_1b1: f64,
    pub degeneracy_gap: f64,
    pub dipole_moment_debye: f64,
}

/// Comprehensive report of the 180° inversion/linearization barrier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WaterLinearizationReport {
    pub equilibrium_theta_deg: f64,
    pub equilibrium_energy: f64,
    pub linear_theta_deg: f64,
    pub linear_energy: f64,
    pub barrier_hartree: f64,
    pub barrier_ev: f64,
    pub barrier_kcal_mol: f64,
    pub barrier_cm1: f64,
    pub walsh_states: Vec<WalshOrbitalState>,
    pub degeneracy_gap_at_180: f64,
    pub dipole_moment_at_180: f64,
}

/// Evaluates the 180° linearization barrier and Walsh orbital correlation curve.
pub fn evaluate_linearization(
    r_angstrom: f64,
    theta_steps: usize,
) -> Result<WaterLinearizationReport, String> {
    let steps = theta_steps.max(5);
    let min_th = 80.0;
    let max_th = 180.0;
    let dth = (max_th - min_th) / ((steps - 1) as f64);

    let mut walsh_states = Vec::with_capacity(steps);
    let eq_point = evaluate_water_point(r_angstrom, DEFAULT_THETA_EQUILIBRIUM)?;
    let mut linear_point: Option<super::surface::WaterPesPoint> = None;

    for s in 0..steps {
        let th = min_th + (s as f64) * dth;
        let pt = evaluate_water_point(r_angstrom, th)?;

        // Orbital index mapping in STO-3G for H2O:
        // Orbitals 0..4 are occupied:
        // 0: 1a1 (O 1s core)
        // 1: 2a1 (O 2s bonding)
        // 2: 1b2 (O 2py bonding)
        // 3: 3a1 (O 2pz lone pair / hybrid)
        // 4: 1b1 (O 2px out-of-plane pure lone pair)
        let eps_3a1 = pt.orbital_energies.get(3).copied().unwrap_or(0.0);
        let eps_1b1 = pt.orbital_energies.get(4).copied().unwrap_or(0.0);
        let gap = (eps_3a1 - eps_1b1).abs();

        if (th - 180.0).abs() < 1e-3 {
            linear_point = Some(pt.clone());
        }

        walsh_states.push(WalshOrbitalState {
            theta_deg: th,
            r_oh: r_angstrom,
            total_energy: pt.total_energy,
            orbital_3a1: eps_3a1,
            orbital_1b1: eps_1b1,
            degeneracy_gap: gap,
            dipole_moment_debye: pt.dipole_moment_debye,
        });
    }

    let lin = linear_point.unwrap_or_else(|| evaluate_water_point(r_angstrom, 180.0).unwrap());
    let barrier_h = lin.total_energy - eq_point.total_energy;

    let final_gap = walsh_states.last().map(|w| w.degeneracy_gap).unwrap_or(0.0);

    Ok(WaterLinearizationReport {
        equilibrium_theta_deg: DEFAULT_THETA_EQUILIBRIUM,
        equilibrium_energy: eq_point.total_energy,
        linear_theta_deg: 180.0,
        linear_energy: lin.total_energy,
        barrier_hartree: barrier_h,
        barrier_ev: barrier_h * HARTREE_TO_EV,
        barrier_kcal_mol: barrier_h * HARTREE_TO_KCAL_MOL,
        barrier_cm1: barrier_h * HARTREE_TO_CM1,
        walsh_states,
        degeneracy_gap_at_180: final_gap,
        dipole_moment_at_180: lin.dipole_moment_debye,
    })
}
