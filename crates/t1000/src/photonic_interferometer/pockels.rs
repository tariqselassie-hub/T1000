// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Electro-Optic Molecular Pockels Modulator (Silicon-Organic Hybrid)
//!
//! Models sub-0.5V organic slot modulators exploiting ultra-high hyperpolarizability
//! chromophores (e.g. Pitt #1 Lead) for low-power optical phase control.

use serde::{Deserialize, Serialize};

/// Electro-optic Pockels phase modulator driven by organic push-pull chromophores.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MolecularPockelsModulator {
    pub molecule_name: String,
    pub chemical_formula: String,
    pub dipole_moment_debye: f64,
    pub polarizability_angstrom3: f64,
    /// Electro-optic Pockels coefficient r_33 in pm/V.
    pub pockels_r33_pm_v: f64,
    /// Effective refractive index of the organic-clad waveguide.
    pub ref_index: f64,
    /// Optical mode confinement factor Gamma inside the active molecular cladding.
    pub confinement_gamma: f64,
    /// Electrode gap in microns (um).
    pub electrode_gap_um: f64,
    /// Interaction length in microns (um).
    pub waveguide_length_um: f64,
    /// Telecom optical wavelength in nm (e.g. 1550.0 nm).
    pub wavelength_nm: f64,
}

impl MolecularPockelsModulator {
    /// Constructs Silicon-Organic Hybrid (SOH) slot modulator using Pitt #1 Lead (C27H39N2O3S+).
    pub fn lead_1_soh() -> Self {
        Self {
            molecule_name: "4-(4-{4-[4-(dibutylamino)phenyl]-1,3-butadien-1-yl}-1-pyridiniumyl)-1-butanesulfonate".to_string(),
            chemical_formula: "C27H39N2O3S+".to_string(),
            dipole_moment_debye: 44.59,
            polarizability_angstrom3: 64.6,
            pockels_r33_pm_v: 342.5,
            ref_index: 1.82,
            confinement_gamma: 0.85,    // 85% mode confinement in nanoscale slot
            electrode_gap_um: 0.15,     // 150 nm slot electrode gap
            waveguide_length_um: 1000.0, // 1 mm compact on-chip interaction length
            wavelength_nm: 1550.0,
        }
    }

    /// Constructs baseline Lithium Niobate (LiNbO3) waveguide modulator for comparison.
    pub fn lithium_niobate_baseline() -> Self {
        Self {
            molecule_name: "Lithium Niobate (Inorganic LiNbO3)".to_string(),
            chemical_formula: "LiNbO3".to_string(),
            dipole_moment_debye: 0.0,
            polarizability_angstrom3: 0.0,
            pockels_r33_pm_v: 30.8,
            ref_index: 2.14,
            confinement_gamma: 1.0,
            electrode_gap_um: 4.0,       // 4.0 um standard electrode separation
            waveguide_length_um: 10000.0, // 10 mm (1 cm) standard device length
            wavelength_nm: 1550.0,
        }
    }

    /// Calculates the half-wave switching voltage V_pi in Volts.
    pub fn v_pi(&self) -> f64 {
        let lambda_m = self.wavelength_nm * 1e-9;
        let d_m = self.electrode_gap_um * 1e-6;
        let l_m = self.waveguide_length_um * 1e-6;
        let r33_m_v = self.pockels_r33_pm_v * 1e-12;
        let n3 = self.ref_index.powi(3);

        (lambda_m * d_m) / (n3 * r33_m_v * self.confinement_gamma * l_m)
    }

    /// Calculates the voltage-length product V_pi * L in V*cm.
    pub fn v_pi_l(&self) -> f64 {
        let v_pi = self.v_pi();
        let l_cm = self.waveguide_length_um * 1e-4;
        v_pi * l_cm
    }

    /// Computes the phase shift delta_phi (in radians) induced by voltage V.
    pub fn phase_shift(&self, voltage: f64) -> f64 {
        std::f64::consts::PI * (voltage / self.v_pi())
    }
}
