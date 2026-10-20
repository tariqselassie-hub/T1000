// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Native quantum chemistry module for Period 1 & 2 molecules up to 18 qubits.
//!
//! Provides second-quantized spin-orbital mappings, Jordan-Wigner Hamiltonians,
//! ground-state preparation, and orbital entanglement entropy diagnostics.

pub mod baseline;
pub mod coulomb;
pub mod hamiltonian;
pub mod molecules;
pub mod qm9;

pub use baseline::{
    AtomicBaselineDecoder, AtomicReferenceTable, B3LYP_E_ATOM_C, B3LYP_E_ATOM_F, B3LYP_E_ATOM_H,
    B3LYP_E_ATOM_N, B3LYP_E_ATOM_O, HARTREE_TO_EV, HARTREE_TO_KCAL_MOL,
};
pub use coulomb::{CoulombMatrix50, RoboBohrRecord, COULOMB_DIM_50, COULOMB_UPPER_TRI_COUNT_50};
pub use hamiltonian::{JordanWignerHamiltonian, PauliOp, PauliTerm};
pub use molecules::{Period12Molecule, Period12MoleculeMetadata};
pub use qm9::{Qm9Atom, Qm9Record};

use serde::{Deserialize, Serialize};

/// Comprehensive quantum electronic report for a simulated Period 1 or 2 molecule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MolecularQuantumReport {
    pub molecule: String,
    pub formula: String,
    pub smiles: String,
    pub qubits: usize,
    pub electrons: usize,
    pub nuclear_repulsion_energy: f64,
    pub hf_energy: f64,
    pub ground_state_energy: f64,
    pub correlation_energy: f64,
    pub equilibrium_distance_angstrom: f64,
    pub dipole_moment_debye: f64,
    pub symmetry_group: String,
    pub spin_orbital_labels: Vec<String>,
    pub orbital_occupations: Vec<f64>,
    pub orbital_entanglement_entropies: Vec<f64>,
    pub total_orbital_entropy: f64,
    pub state_purity: f64,
}

/// Simulates a Period 1 or 2 molecule in second-quantized Hilbert space up to 18 qubits.
pub fn simulate_period_molecule(mol: Period12Molecule) -> MolecularQuantumReport {
    let meta = mol.metadata();
    let hamiltonian = JordanWignerHamiltonian::from_molecule(&mol);
    let (state, ground_energy) = hamiltonian.ground_state_simulation(&mol);

    let occupations = hamiltonian.orbital_occupations(&state);
    let entropies = hamiltonian.orbital_entanglement_entropies(&state);
    let total_entropy: f64 = entropies.iter().sum();
    let purity = state.l2_norm_sq();
    let correlation_energy = ground_energy - meta.hf_energy;

    MolecularQuantumReport {
        molecule: meta.name.to_string(),
        formula: meta.formula.to_string(),
        smiles: meta.smiles.to_string(),
        qubits: meta.qubits,
        electrons: meta.electrons,
        nuclear_repulsion_energy: meta.nuclear_repulsion_energy,
        hf_energy: meta.hf_energy,
        ground_state_energy: ground_energy,
        correlation_energy,
        equilibrium_distance_angstrom: meta.equilibrium_distance_angstrom,
        dipole_moment_debye: meta.dipole_moment_debye,
        symmetry_group: meta.symmetry_group.to_string(),
        spin_orbital_labels: meta
            .spin_orbital_labels
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
        orbital_occupations: occupations,
        orbital_entanglement_entropies: entropies,
        total_orbital_entropy: total_entropy,
        state_purity: purity,
    }
}

/// Lists metadata for all 9 supported Period 1 & 2 molecules.
pub fn list_period_molecules() -> Vec<Period12MoleculeMetadata> {
    Period12Molecule::ALL.iter().map(|m| m.metadata()).collect()
}
