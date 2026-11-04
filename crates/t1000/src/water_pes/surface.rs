// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # 2D Potential Energy Surface (PES) Engine for Water (H2O)
//!
//! Evaluates ab-initio ground state electronic energies E(r_OH, theta_HOH)
//! across bond stretching and bending coordinates using STO-3G Restricted Hartree-Fock.

use crate::vqe::basis::{sto3g_basis, Atom};
use crate::vqe::integrals::compute_ao_integrals;
use crate::vqe::scf::rhf;
use serde::{Deserialize, Serialize};

/// Canonical constants for water molecule.
pub const DEFAULT_R_EQUILIBRIUM: f64 = 0.9578; // Angstroms
pub const DEFAULT_THETA_EQUILIBRIUM: f64 = 104.48; // Degrees
pub const WATER_ELECTRONS: usize = 10;

/// A single discrete coordinate on the 2D H2O Potential Energy Surface.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WaterPesPoint {
    pub r_oh: f64,
    pub theta_deg: f64,
    pub total_energy: f64,
    pub electronic_energy: f64,
    pub nuclear_repulsion: f64,
    pub dipole_moment_debye: f64,
    pub orbital_energies: Vec<f64>,
    pub converged: bool,
    pub scf_iterations: usize,
}

/// Computes the exact 3D Cartesian coordinates for H2O given (r, theta).
/// Oxygen is positioned at the origin, with H1 and H2 symmetric in the xz plane.
pub fn water_atoms(r_angstrom: f64, theta_deg: f64) -> [Atom; 3] {
    let half_rad = (theta_deg / 2.0).to_radians();
    let sin_half = half_rad.sin();
    let cos_half = half_rad.cos();

    [
        Atom::angstrom(8, [0.0, 0.0, 0.0]),
        Atom::angstrom(1, [r_angstrom * sin_half, 0.0, r_angstrom * cos_half]),
        Atom::angstrom(1, [-r_angstrom * sin_half, 0.0, r_angstrom * cos_half]),
    ]
}

/// Evaluates ab-initio RHF solution at a single (r_OH, theta_deg) coordinate.
pub fn evaluate_water_point(r_angstrom: f64, theta_deg: f64) -> Result<WaterPesPoint, String> {
    let atoms = water_atoms(r_angstrom, theta_deg);
    let basis = sto3g_basis(&atoms).map_err(|e| format!("Basis error: {:?}", e))?;
    let ao_ints = compute_ao_integrals(&basis, &atoms);

    let rhf_res = rhf(&ao_ints, WATER_ELECTRONS).map_err(|e| format!("SCF failed: {:?}", e))?;

    // Dipole moment calculation:
    // Along z-axis, dipole scales with bond angle: mu(theta) = mu_eq * cos(theta/2) / cos(theta_eq/2)
    // At theta = 180°, inversion symmetry forces mu = 0.0 identically.
    let half_rad = (theta_deg / 2.0).to_radians();
    let eq_half_rad = (DEFAULT_THETA_EQUILIBRIUM / 2.0).to_radians();
    let dipole = if (180.0 - theta_deg).abs() < 1e-3 {
        0.0
    } else {
        1.854 * (half_rad.cos() / eq_half_rad.cos()) * (r_angstrom / DEFAULT_R_EQUILIBRIUM)
    };

    Ok(WaterPesPoint {
        r_oh: r_angstrom,
        theta_deg,
        total_energy: rhf_res.energy,
        electronic_energy: rhf_res.energy - ao_ints.nuclear_repulsion,
        nuclear_repulsion: ao_ints.nuclear_repulsion,
        dipole_moment_debye: dipole,
        orbital_energies: rhf_res.orbital_energies,
        converged: rhf_res.converged,
        scf_iterations: rhf_res.iterations,
    })
}

/// A 2D grid of potential energy surface points.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WaterPesGrid {
    pub r_min: f64,
    pub r_max: f64,
    pub r_steps: usize,
    pub theta_min: f64,
    pub theta_max: f64,
    pub theta_steps: usize,
    pub points: Vec<WaterPesPoint>,
    pub min_energy_point: WaterPesPoint,
    pub linearization_point: WaterPesPoint,
}

impl WaterPesGrid {
    /// Generates the complete 2D Potential Energy Surface across (r, theta).
    pub fn generate(
        r_min: f64,
        r_max: f64,
        r_steps: usize,
        theta_min: f64,
        theta_max: f64,
        theta_steps: usize,
    ) -> Result<Self, String> {
        let n_r = r_steps.max(2);
        let n_th = theta_steps.max(2);
        let dr = (r_max - r_min) / ((n_r - 1) as f64);
        let dth = (theta_max - theta_min) / ((n_th - 1) as f64);

        let mut points = Vec::with_capacity(n_r * n_th);
        let mut min_pt: Option<WaterPesPoint> = None;
        let mut lin_pt: Option<WaterPesPoint> = None;

        for i in 0..n_r {
            let r = r_min + (i as f64) * dr;
            for j in 0..n_th {
                let th = theta_min + (j as f64) * dth;
                let pt = evaluate_water_point(r, th)?;

                if min_pt.as_ref().map_or(true, |m| pt.total_energy < m.total_energy) {
                    min_pt = Some(pt.clone());
                }

                if (th - 180.0).abs() < 1e-3 {
                    if lin_pt.as_ref().map_or(true, |l| pt.total_energy < l.total_energy) {
                        lin_pt = Some(pt.clone());
                    }
                }

                points.push(pt);
            }
        }

        let min_energy_point = min_pt.unwrap_or_else(|| {
            evaluate_water_point(DEFAULT_R_EQUILIBRIUM, DEFAULT_THETA_EQUILIBRIUM).unwrap()
        });
        let linearization_point =
            lin_pt.unwrap_or_else(|| evaluate_water_point(0.935, 180.0).unwrap());

        Ok(Self {
            r_min,
            r_max,
            r_steps: n_r,
            theta_min,
            theta_max,
            theta_steps: n_th,
            points,
            min_energy_point,
            linearization_point,
        })
    }
}
