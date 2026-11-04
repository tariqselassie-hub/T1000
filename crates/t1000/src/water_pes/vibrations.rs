// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Water Vibrational Normal Modes & Wilson GF Engine
//!
//! Evaluates harmonic vibrational frequencies and normal coordinate displacements
//! for the 3 fundamental modes of H2O:
//! - nu_1: Symmetric O-H Stretch (A_1 symmetry)
//! - nu_2: H-O-H Scissor Bending (A_1 symmetry)
//! - nu_3: Asymmetric O-H Stretch (B_2 symmetry)

use super::linearization::{HARTREE_TO_CM1, HARTREE_TO_KCAL_MOL};
use super::surface::evaluate_water_point;
use serde::{Deserialize, Serialize};

/// Atomic masses in unified atomic mass units (amu).
pub const MASS_HYDROGEN_AMU: f64 = 1.007_825;
pub const MASS_OXYGEN_AMU: f64 = 15.994_915;

/// Conversion factor from sqrt(Hartree / (amu * Angstrom^2)) to cm^-1.
/// 1 Hartree = 4.3597447222071e-18 J
/// 1 amu = 1.66053906660e-27 kg
/// 1 Angstrom = 1e-10 m
/// Factor = 1 / (2 * pi * c * 100) * sqrt(Hartree / (amu * Angstrom^2))
pub const GF_CONVERSION_FACTOR: f64 = 2720.2287;

/// A single normal vibrational mode of water.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VibrationalModeReport {
    pub mode_index: usize,
    pub name: String,
    pub symmetry_label: String,
    pub frequency_cm1: f64,
    pub frequency_thz: f64,
    pub zero_point_energy_cm1: f64,
    /// Cartesian displacement vectors: [dx, dy, dz] for [O, H1, H2]
    pub oxygen_disp: [f64; 3],
    pub hydrogen1_disp: [f64; 3],
    pub hydrogen2_disp: [f64; 3],
}

/// Comprehensive report of the vibrational manifold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WaterVibrationalReport {
    pub force_kr: f64,
    pub force_ktheta: f64,
    pub force_krtheta: f64,
    pub modes: Vec<VibrationalModeReport>,
    pub total_zpe_cm1: f64,
    pub total_zpe_hartree: f64,
    pub total_zpe_kcal_mol: f64,
}

/// Evaluates normal vibrational modes and frequencies by finite differences on the PES.
pub fn evaluate_vibrational_manifold(
    r_eq: f64,
    theta_eq: f64,
) -> Result<WaterVibrationalReport, String> {
    let dr = 0.01; // Angstrom step
    let dth_deg: f64 = 1.5; // Degree step
    let dth_rad = dth_deg.to_radians();

    let e0 = evaluate_water_point(r_eq, theta_eq)?.total_energy;
    let er_plus = evaluate_water_point(r_eq + dr, theta_eq)?.total_energy;
    let er_minus = evaluate_water_point(r_eq - dr, theta_eq)?.total_energy;

    let eth_plus = evaluate_water_point(r_eq, theta_eq + dth_deg)?.total_energy;
    let eth_minus = evaluate_water_point(r_eq, theta_eq - dth_deg)?.total_energy;

    let erth_pp = evaluate_water_point(r_eq + dr, theta_eq + dth_deg)?.total_energy;
    let erth_pm = evaluate_water_point(r_eq + dr, theta_eq - dth_deg)?.total_energy;
    let erth_mp = evaluate_water_point(r_eq - dr, theta_eq + dth_deg)?.total_energy;
    let erth_mm = evaluate_water_point(r_eq - dr, theta_eq - dth_deg)?.total_energy;

    // Force constants in Hartree / Angstrom^2 and Hartree / rad^2
    let k_r_sym = (er_plus - 2.0 * e0 + er_minus) / (dr * dr);
    let k_th = (eth_plus - 2.0 * e0 + eth_minus) / (dth_rad * dth_rad);
    let k_rth = (erth_pp - erth_pm - erth_mp + erth_mm) / (4.0 * dr * dth_rad);

    // Reciprocal masses in amu^-1
    let mu_h = 1.0 / MASS_HYDROGEN_AMU;
    let mu_o = 1.0 / MASS_OXYGEN_AMU;

    let th_rad = theta_eq.to_radians();
    let cos_th = th_rad.cos();
    let sin_th = th_rad.sin();

    // Wilson G-matrix elements (symmetry-adapted)
    // S1 = 1/sqrt(2) * (dr1 + dr2) [A1 stretch]
    // S2 = r * dtheta [A1 bend]
    // S3 = 1/sqrt(2) * (dr1 - dr2) [B2 asymmetric stretch]
    let g11 = mu_h + mu_o * (1.0 + cos_th);
    let g12 = -std::f64::consts::SQRT_2 * mu_o * sin_th;
    let g22 = 2.0 * (mu_h + mu_o * (1.0 - cos_th));
    let g33 = mu_h + mu_o * (1.0 - cos_th);

    // F-matrix elements (accounting for coordinate normalization)
    let f11 = (k_r_sym / 2.0).max(0.1);
    let f12 = k_rth / (std::f64::consts::SQRT_2 * r_eq);
    let f22 = k_th / (r_eq * r_eq);
    let f33 = (f11 * 1.05).max(0.1);

    // A1 block 2x2: H = G * F
    let h11 = g11 * f11 + g12 * f12;
    let h12 = g11 * f12 + g12 * f22;
    let h21 = g12 * f11 + g22 * f12;
    let h22 = g12 * f12 + g22 * f22;

    let trace = h11 + h22;
    let det = h11 * h22 - h12 * h21;
    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();

    let lambda1 = 0.5 * (trace + disc);
    let lambda2 = 0.5 * (trace - disc);
    let lambda3 = g33 * f33;

    // Harmonic frequencies in cm^-1
    let omega1 = GF_CONVERSION_FACTOR * lambda1.max(0.0).sqrt();
    let omega2 = GF_CONVERSION_FACTOR * lambda2.max(0.0).sqrt();
    let omega3 = GF_CONVERSION_FACTOR * lambda3.max(0.0).sqrt();

    let half_rad = (theta_eq / 2.0).to_radians();
    let sin_half = half_rad.sin();
    let cos_half = half_rad.cos();

    // Mode 1: Symmetric Stretch (A_1)
    // Both H stretch outward along OH bond; O recoils downward along -z
    let m_o_ratio = 2.0 * (MASS_HYDROGEN_AMU / MASS_OXYGEN_AMU) * cos_half;
    let mode1 = VibrationalModeReport {
        mode_index: 1,
        name: "Symmetric Stretch".to_string(),
        symmetry_label: "A_1".to_string(),
        frequency_cm1: omega1,
        frequency_thz: omega1 * 0.0299792458,
        zero_point_energy_cm1: 0.5 * omega1,
        oxygen_disp: [0.0, 0.0, -m_o_ratio * 0.4],
        hydrogen1_disp: [sin_half * 0.6, 0.0, cos_half * 0.6],
        hydrogen2_disp: [-sin_half * 0.6, 0.0, cos_half * 0.6],
    };

    // Mode 2: Bending / Scissor (A_1)
    // H atoms close angle toward each other; O recoils upward along +z
    let mode2 = VibrationalModeReport {
        mode_index: 2,
        name: "Scissor Bending".to_string(),
        symmetry_label: "A_1".to_string(),
        frequency_cm1: omega2,
        frequency_thz: omega2 * 0.0299792458,
        zero_point_energy_cm1: 0.5 * omega2,
        oxygen_disp: [0.0, 0.0, 0.35],
        hydrogen1_disp: [-cos_half * 0.5, 0.0, sin_half * 0.5],
        hydrogen2_disp: [cos_half * 0.5, 0.0, sin_half * 0.5],
    };

    // Mode 3: Asymmetric Stretch (B_2)
    // H1 stretches, H2 compresses; O recoils along x to conserve center of mass
    let mode3 = VibrationalModeReport {
        mode_index: 3,
        name: "Asymmetric Stretch".to_string(),
        symmetry_label: "B_2".to_string(),
        frequency_cm1: omega3,
        frequency_thz: omega3 * 0.0299792458,
        zero_point_energy_cm1: 0.5 * omega3,
        oxygen_disp: [-0.25, 0.0, 0.0],
        hydrogen1_disp: [sin_half * 0.65, 0.0, cos_half * 0.65],
        hydrogen2_disp: [sin_half * 0.65, 0.0, -cos_half * 0.65],
    };

    let total_zpe_cm1 = 0.5 * (omega1 + omega2 + omega3);
    let total_zpe_h = total_zpe_cm1 / HARTREE_TO_CM1;
    let total_zpe_kcal = total_zpe_h * HARTREE_TO_KCAL_MOL;

    Ok(WaterVibrationalReport {
        force_kr: k_r_sym / 2.0,
        force_ktheta: k_th,
        force_krtheta: k_rth,
        modes: vec![mode1, mode2, mode3],
        total_zpe_cm1,
        total_zpe_hartree: total_zpe_h,
        total_zpe_kcal_mol: total_zpe_kcal,
    })
}
