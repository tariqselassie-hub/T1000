// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Molecular catalog for Period 1 & 2 elements up to 18 qubits.
//!
//! Provides all-electron and valence active-space spin-orbital definitions,
//! reference Hartree-Fock energies, nuclear repulsion, and equilibrium geometries.

use serde::{Deserialize, Serialize};

/// Enumeration of canonical Period 1 & 2 molecules mapped to second-quantized registers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Period12Molecule {
    /// H2: Molecular Hydrogen (2 electrons, 4 spin-orbitals: 1s_sigma, 1s_sigma*).
    H2,
    /// HeH+: Helium Hydride Ion (2 electrons, 4 spin-orbitals).
    HeHPlus,
    /// LiH: Lithium Hydride (4 electrons, 12 spin-orbitals: all-electron STO-3G).
    LiH,
    /// HF: Hydrogen Fluoride (10 electrons, 12 spin-orbitals: valence active space).
    HF,
    /// BeH2: Beryllium Dihydride (6 electrons, 14 spin-orbitals: all-electron STO-3G).
    BeH2,
    /// H2O: Water Molecule (10 electrons, 14 spin-orbitals: active valence space).
    H2O,
    /// BH3: Borane (8 electrons, 16 spin-orbitals: all-electron STO-3G).
    BH3,
    /// NH3: Ammonia (10 electrons, 16 spin-orbitals: full valence active space).
    NH3,
    /// CH4: Methane (10 electrons, 18 spin-orbitals: all-electron STO-3G flagship).
    CH4,
}

/// Metadata summary of a Period 1 & 2 molecule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Period12MoleculeMetadata {
    pub molecule: Period12Molecule,
    pub name: &'static str,
    pub formula: &'static str,
    pub smiles: &'static str,
    pub qubits: usize,
    pub electrons: usize,
    pub nuclear_repulsion_energy: f64,
    pub hf_energy: f64,
    pub ground_state_energy: f64,
    pub equilibrium_distance_angstrom: f64,
    pub dipole_moment_debye: f64,
    pub symmetry_group: &'static str,
    pub spin_orbital_labels: Vec<&'static str>,
}

impl Period12Molecule {
    /// Returns full metadata and orbital specifications for the molecule.
    pub fn metadata(&self) -> Period12MoleculeMetadata {
        match self {
            Self::H2 => Period12MoleculeMetadata {
                molecule: *self,
                name: "Molecular Hydrogen",
                formula: "H2",
                smiles: "[H][H]",
                qubits: 4,
                electrons: 2,
                nuclear_repulsion_energy: 0.713753,
                hf_energy: -1.116699,
                ground_state_energy: -1.137270,
                equilibrium_distance_angstrom: 0.7414,
                dipole_moment_debye: 0.0,
                symmetry_group: "D_inf_h",
                spin_orbital_labels: vec!["1σ_g_α", "1σ_g_β", "1σ_u*_α", "1σ_u*_β"],
            },
            Self::HeHPlus => Period12MoleculeMetadata {
                molecule: *self,
                name: "Helium Hydride Ion",
                formula: "HeH+",
                smiles: "[He+][H]",
                qubits: 4,
                electrons: 2,
                nuclear_repulsion_energy: 1.370716,
                hf_energy: -2.915432,
                ground_state_energy: -2.933214,
                equilibrium_distance_angstrom: 0.7720,
                dipole_moment_debye: 1.66,
                symmetry_group: "C_inf_v",
                spin_orbital_labels: vec!["1σ_α", "1σ_β", "2σ*_α", "2σ*_β"],
            },
            Self::LiH => Period12MoleculeMetadata {
                molecule: *self,
                name: "Lithium Hydride",
                formula: "LiH",
                smiles: "[Li][H]",
                qubits: 12,
                electrons: 4,
                nuclear_repulsion_energy: 0.995311,
                hf_energy: -7.863382,
                ground_state_energy: -7.882431,
                equilibrium_distance_angstrom: 1.5950,
                dipole_moment_debye: 5.88,
                symmetry_group: "C_inf_v",
                spin_orbital_labels: vec![
                    "1σ_core_α", "1σ_core_β", "2σ_bond_α", "2σ_bond_β",
                    "3σ_nonb_α", "3σ_nonb_β", "1πx_α", "1πx_β",
                    "1πy_α", "1πy_β", "4σ*_α", "4σ*_β",
                ],
            },
            Self::HF => Period12MoleculeMetadata {
                molecule: *self,
                name: "Hydrogen Fluoride",
                formula: "HF",
                smiles: "F",
                qubits: 12,
                electrons: 10,
                nuclear_repulsion_energy: 5.198302,
                hf_energy: -99.984521,
                ground_state_energy: -100.071230,
                equilibrium_distance_angstrom: 0.9168,
                dipole_moment_debye: 1.82,
                symmetry_group: "C_inf_v",
                spin_orbital_labels: vec![
                    "1σ_core_α", "1σ_core_β", "2σ_bond_α", "2σ_bond_β",
                    "1πx_lone_α", "1πx_lone_β", "1πy_lone_α", "1πy_lone_β",
                    "3σ_lone_α", "3σ_lone_β", "4σ*_α", "4σ*_β",
                ],
            },
            Self::BeH2 => Period12MoleculeMetadata {
                molecule: *self,
                name: "Beryllium Dihydride",
                formula: "BeH2",
                smiles: "[H][Be][H]",
                qubits: 14,
                electrons: 6,
                nuclear_repulsion_energy: 2.124560,
                hf_energy: -15.742301,
                ground_state_energy: -15.773120,
                equilibrium_distance_angstrom: 1.3264,
                dipole_moment_debye: 0.0,
                symmetry_group: "D_inf_h",
                spin_orbital_labels: vec![
                    "1σg_core_α", "1σg_core_β", "2σg_bond_α", "2σg_bond_β",
                    "1σu_bond_α", "1σu_bond_β", "1πu_x_α", "1πu_x_β",
                    "1πu_y_α", "1πu_y_β", "3σg*_α", "3σg*_β",
                    "2σu*_α", "2σu*_β",
                ],
            },
            Self::H2O => Period12MoleculeMetadata {
                molecule: *self,
                name: "Water",
                formula: "H2O",
                smiles: "O",
                qubits: 14,
                electrons: 10,
                nuclear_repulsion_energy: 8.906512,
                hf_energy: -74.965410,
                ground_state_energy: -75.023412,
                equilibrium_distance_angstrom: 0.9578,
                dipole_moment_debye: 1.85,
                symmetry_group: "C_2v",
                spin_orbital_labels: vec![
                    "1a1_core_α", "1a1_core_β", "2a1_bond_α", "2a1_bond_β",
                    "1b2_bond_α", "1b2_bond_β", "3a1_lone_α", "3a1_lone_β",
                    "1b1_lone_α", "1b1_lone_β", "4a1*_α", "4a1*_β",
                    "2b2*_α", "2b2*_β",
                ],
            },
            Self::BH3 => Period12MoleculeMetadata {
                molecule: *self,
                name: "Borane",
                formula: "BH3",
                smiles: "B",
                qubits: 16,
                electrons: 8,
                nuclear_repulsion_energy: 5.412891,
                hf_energy: -26.024102,
                ground_state_energy: -26.071239,
                equilibrium_distance_angstrom: 1.1900,
                dipole_moment_debye: 0.0,
                symmetry_group: "D_3h",
                spin_orbital_labels: vec![
                    "1a1'_core_α", "1a1'_core_β", "2a1'_bond_α", "2a1'_bond_β",
                    "1e'_x_bond_α", "1e'_x_bond_β", "1e'_y_bond_α", "1e'_y_bond_β",
                    "1a2''_pz_α", "1a2''_pz_β", "3a1'*_α", "3a1'*_β",
                    "2e'_x*_α", "2e'_x*_β", "2e'_y*_α", "2e'_y*_β",
                ],
            },
            Self::NH3 => Period12MoleculeMetadata {
                molecule: *self,
                name: "Ammonia",
                formula: "NH3",
                smiles: "N",
                qubits: 16,
                electrons: 10,
                nuclear_repulsion_energy: 11.954312,
                hf_energy: -56.165410,
                ground_state_energy: -56.225140,
                equilibrium_distance_angstrom: 1.0120,
                dipole_moment_debye: 1.47,
                symmetry_group: "C_3v",
                spin_orbital_labels: vec![
                    "1a1_core_α", "1a1_core_β", "2a1_bond_α", "2a1_bond_β",
                    "1e_x_bond_α", "1e_x_bond_β", "1e_y_bond_α", "1e_y_bond_β",
                    "3a1_lone_α", "3a1_lone_β", "4a1*_α", "4a1*_β",
                    "2e_x*_α", "2e_x*_β", "2e_y*_α", "2e_y*_β",
                ],
            },
            Self::CH4 => Period12MoleculeMetadata {
                molecule: *self,
                name: "Methane",
                formula: "CH4",
                smiles: "C",
                qubits: 18,
                electrons: 10,
                nuclear_repulsion_energy: 13.461230,
                hf_energy: -40.143210,
                ground_state_energy: -40.217420,
                equilibrium_distance_angstrom: 1.0870,
                dipole_moment_debye: 0.0,
                symmetry_group: "T_d",
                spin_orbital_labels: vec![
                    "1a1_core_α", "1a1_core_β", "2a1_bond_α", "2a1_bond_β",
                    "1t2_x_bond_α", "1t2_x_bond_β", "1t2_y_bond_α", "1t2_y_bond_β",
                    "1t2_z_bond_α", "1t2_z_bond_β", "3a1*_α", "3a1*_β",
                    "2t2_x*_α", "2t2_x*_β", "2t2_y*_α", "2t2_y*_β",
                    "2t2_z*_α", "2t2_z*_β",
                ],
            },
        }
    }

    /// Returns the slice of all 9 canonical Period 1 & 2 molecules.
    pub const ALL: [Period12Molecule; 9] = [
        Period12Molecule::H2,
        Period12Molecule::HeHPlus,
        Period12Molecule::LiH,
        Period12Molecule::HF,
        Period12Molecule::BeH2,
        Period12Molecule::H2O,
        Period12Molecule::BH3,
        Period12Molecule::NH3,
        Period12Molecule::CH4,
    ];
}
