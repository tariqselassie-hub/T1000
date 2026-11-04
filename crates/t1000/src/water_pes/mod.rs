// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Native Water (H2O) 2D Energy & Linearization Barrier Engine
//!
//! Models the complete ab-initio potential energy surface, vibrational manifold,
//! and 180° planar linearization barrier for water (H2O):
//! - **2D PES Surface E(r_OH, theta_HOH):** Full Restricted Hartree-Fock STO-3G electronic manifold.
//! - **180° Linearization Barrier:** Delta E_lin barrier separating bent C_2v wells from linear D_inf_h.
//! - **Walsh Molecular Orbital Correlation:** Degeneracy tracking of 3a_1 and 1b_1 -> 1pi_u.
//! - **Vibrational Normal Modes:** Wilson GF matrix harmonic frequencies (nu_1, nu_2, nu_3) and ZPE.

pub mod linearization;
pub mod surface;
pub mod vibrations;

pub use linearization::{
    evaluate_linearization, WalshOrbitalState, WaterLinearizationReport, HARTREE_TO_CM1,
    HARTREE_TO_EV, HARTREE_TO_KCAL_MOL,
};
pub use surface::{
    evaluate_water_point, water_atoms, WaterPesGrid, WaterPesPoint, DEFAULT_R_EQUILIBRIUM,
    DEFAULT_THETA_EQUILIBRIUM, WATER_ELECTRONS,
};
pub use vibrations::{
    evaluate_vibrational_manifold, VibrationalModeReport, WaterVibrationalReport,
};

use serde::{Deserialize, Serialize};

/// Configuration for the 2D Water PES and Linearization calculation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WaterPesConfig {
    pub r_min: f64,
    pub r_max: f64,
    pub r_steps: usize,
    pub theta_min: f64,
    pub theta_max: f64,
    pub theta_steps: usize,
    pub walsh_steps: usize,
}

impl Default for WaterPesConfig {
    fn default() -> Self {
        Self {
            r_min: 0.80,
            r_max: 1.30,
            r_steps: 11,
            theta_min: 80.0,
            theta_max: 180.0,
            theta_steps: 21,
            walsh_steps: 25,
        }
    }
}

/// Comprehensive report of the full H2O potential energy surface and vibrational manifold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WaterPesReport {
    pub config: WaterPesConfig,
    pub grid: WaterPesGrid,
    pub linearization: WaterLinearizationReport,
    pub vibrations: WaterVibrationalReport,
    pub equilibrium_r_oh: f64,
    pub equilibrium_theta_deg: f64,
    pub equilibrium_energy: f64,
    pub linearization_barrier_cm1: f64,
    pub linearization_barrier_ev: f64,
    pub linearization_barrier_kcal_mol: f64,
    pub zpe_cm1: f64,
}

/// The sovereign Water PES & Linearization Engine.
#[derive(Debug, Clone)]
pub struct WaterPesEngine {
    pub config: WaterPesConfig,
}

impl WaterPesEngine {
    pub fn new(config: WaterPesConfig) -> Self {
        Self { config }
    }

    /// Computes the complete potential energy manifold, barrier, and vibrational spectrum.
    pub fn compute_manifold(&self) -> Result<WaterPesReport, String> {
        let grid = WaterPesGrid::generate(
            self.config.r_min,
            self.config.r_max,
            self.config.r_steps,
            self.config.theta_min,
            self.config.theta_max,
            self.config.theta_steps,
        )?;

        let linearization = evaluate_linearization(
            grid.min_energy_point.r_oh,
            self.config.walsh_steps,
        )?;

        let vibrations = evaluate_vibrational_manifold(
            grid.min_energy_point.r_oh,
            grid.min_energy_point.theta_deg,
        )?;

        let eq_r = grid.min_energy_point.r_oh;
        let eq_th = grid.min_energy_point.theta_deg;
        let eq_energy = grid.min_energy_point.total_energy;
        let lin_barrier_cm1 = linearization.barrier_cm1;
        let lin_barrier_ev = linearization.barrier_ev;
        let lin_barrier_kcal = linearization.barrier_kcal_mol;
        let zpe_cm1 = vibrations.total_zpe_cm1;

        Ok(WaterPesReport {
            config: self.config.clone(),
            grid,
            linearization,
            vibrations,
            equilibrium_r_oh: eq_r,
            equilibrium_theta_deg: eq_th,
            equilibrium_energy: eq_energy,
            linearization_barrier_cm1: lin_barrier_cm1,
            linearization_barrier_ev: lin_barrier_ev,
            linearization_barrier_kcal_mol: lin_barrier_kcal,
            zpe_cm1,
        })
    }
}
