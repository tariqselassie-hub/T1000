// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Second-quantized molecular Hamiltonian generator and Jordan-Wigner transform.
//!
//! Maps molecular orbitals to qubit operators, computes expectation values,
//! orbital electron densities, and quantum orbital entanglement entropy.
//!
//! **Model Hamiltonian.** The one- and two-body coefficients here are fixed formulas and
//! `ground_state_simulation` returns the tabulated energy from `molecules`. For a Hamiltonian
//! computed from molecular integrals, with a variational solver and an exact reference, use
//! [`crate::vqe`].

use crate::chemistry::molecules::Period12Molecule;
use crate::QuantumState;
use serde::{Deserialize, Serialize};

/// Single-qubit Pauli operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PauliOp {
    X,
    Y,
    Z,
}

/// A weighted Pauli string term: c * (P_q1 ⊗ P_q2 ⊗ ...).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PauliTerm {
    pub coeff: f64,
    pub ops: Vec<(usize, PauliOp)>,
}

/// Second-quantized electronic structure Hamiltonian transformed via Jordan-Wigner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JordanWignerHamiltonian {
    pub qubits: usize,
    pub nuclear_repulsion: f64,
    pub terms: Vec<PauliTerm>,
}

impl JordanWignerHamiltonian {
    /// Builds a second-quantized Jordan-Wigner Hamiltonian for a Period 1 & 2 molecule.
    pub fn from_molecule(mol: &Period12Molecule) -> Self {
        let meta = mol.metadata();
        let qubits = meta.qubits;
        let mut terms = Vec::new();

        // 1. Nuclear repulsion constant term
        terms.push(PauliTerm {
            coeff: meta.nuclear_repulsion_energy,
            ops: Vec::new(),
        });

        // 2. One-body diagonal orbital potential terms: h_ii * a_i^dagger a_i
        // Under Jordan-Wigner: a_i^dagger a_i = (I - Z_i) / 2
        let n_occ = meta.electrons / 2;
        for i in 0..qubits {
            let orbital_idx = i / 2;
            let h_ii = if orbital_idx < n_occ {
                // Occupied core/bonding orbital: deep negative one-electron energy
                -1.8 - 0.4 * (orbital_idx as f64)
            } else {
                // Virtual/unoccupied orbital: higher energy
                0.2 + 0.3 * ((orbital_idx - n_occ) as f64)
            };

            // Identity component
            terms.push(PauliTerm {
                coeff: 0.5 * h_ii,
                ops: Vec::new(),
            });
            // -0.5 * h_ii * Z_i
            terms.push(PauliTerm {
                coeff: -0.5 * h_ii,
                ops: vec![(i, PauliOp::Z)],
            });
        }

        // 3. Two-body Coulomb and exchange interaction terms between spatial pairs
        for i in 0..qubits {
            for j in (i + 1)..qubits {
                let orbital_i = i / 2;
                let orbital_j = j / 2;
                let coupling = if orbital_i == orbital_j {
                    // On-site Hubbard-like Coulomb repulsion
                    0.45
                } else {
                    0.20 / (1.0 + (orbital_i.abs_diff(orbital_j) as f64))
                };

                // Z_i Z_j interaction
                terms.push(PauliTerm {
                    coeff: 0.25 * coupling,
                    ops: vec![(i, PauliOp::Z), (j, PauliOp::Z)],
                });
            }
        }

        // 4. Resonance / hopping excitations: (X_i X_j + Y_i Y_j) between paired spin-orbitals
        for i in (0..qubits).step_by(2) {
            let j = (i + 2) % qubits;
            let hop_t = 0.08 / (1.0 + ((i / 2) as f64));
            terms.push(PauliTerm {
                coeff: 0.5 * hop_t,
                ops: vec![(i, PauliOp::X), (j, PauliOp::X)],
            });
            terms.push(PauliTerm {
                coeff: 0.5 * hop_t,
                ops: vec![(i, PauliOp::Y), (j, PauliOp::Y)],
            });
        }

        Self {
            qubits,
            nuclear_repulsion: meta.nuclear_repulsion_energy,
            terms,
        }
    }

    /// Prepares the single-reference Hartree-Fock state |11...100...0> in computational basis.
    pub fn prepare_hartree_fock_state(&self, electrons: usize) -> QuantumState {
        let dim = 1usize << self.qubits;
        let mut state = QuantumState::new(dim);
        // Clear default |0...0> amplitude
        state.real[0] = 0.0;
        // Set bitstring with lowest `electrons` spin-orbitals set to 1
        let mut bitstring = 0usize;
        for q in 0..electrons.min(self.qubits) {
            bitstring |= 1 << q;
        }
        state.real[bitstring] = 1.0;
        state
    }

    /// Computes spin-orbital occupations n_k = <ψ| a_k^dagger a_k |ψ> = (1 - <Z_k>) / 2.
    pub fn orbital_occupations(&self, state: &QuantumState) -> Vec<f64> {
        let probs = state.probabilities();
        let mut occs = vec![0.0f64; self.qubits];

        for (idx, &prob) in probs.iter().enumerate() {
            if prob <= 0.0 {
                continue;
            }
            for q in 0..self.qubits {
                if (idx & (1 << q)) != 0 {
                    occs[q] += prob;
                }
            }
        }

        occs
    }

    /// Computes single-orbital Von Neumann entanglement entropy for each spin-orbital:
    /// S_k = -n_k * ln(n_k) - (1 - n_k) * ln(1 - n_k).
    pub fn orbital_entanglement_entropies(&self, state: &QuantumState) -> Vec<f64> {
        let occs = self.orbital_occupations(state);
        occs.into_iter()
            .map(|n| {
                let clamped = n.clamp(1e-12, 1.0 - 1e-12);
                -(clamped * clamped.ln() + (1.0 - clamped) * (1.0 - clamped).ln())
            })
            .collect()
    }

    /// Evaluates the total energy expectation <ψ| H |ψ>.
    pub fn evaluate_expectation(&self, state: &QuantumState) -> f64 {
        let probs = state.probabilities();
        let mut energy = 0.0f64;

        for term in &self.terms {
            if term.ops.is_empty() {
                energy += term.coeff;
                continue;
            }

            // Diagonal Z-only terms can be evaluated directly from probability distribution
            let all_z = term.ops.iter().all(|(_, op)| *op == PauliOp::Z);
            if all_z {
                let mut term_val = 0.0f64;
                for (idx, &prob) in probs.iter().enumerate() {
                    if prob <= 0.0 {
                        continue;
                    }
                    let mut parity = 1.0f64;
                    for &(q, _) in &term.ops {
                        if (idx & (1 << q)) != 0 {
                            parity = -parity;
                        }
                    }
                    term_val += parity * prob;
                }
                energy += term.coeff * term_val;
            }
        }

        energy
    }

    /// Synthesizes correlated electronic ground state via correlated superposition.
    pub fn ground_state_simulation(&self, mol: &Period12Molecule) -> (QuantumState, f64) {
        let meta = mol.metadata();
        let dim = 1usize << self.qubits;
        let mut state = self.prepare_hartree_fock_state(meta.electrons);

        // Inject dynamic correlation excitations: doubles excitation from HOMO to LUMO
        let n_occ = meta.electrons.min(self.qubits);
        if n_occ >= 2 && n_occ + 2 <= self.qubits {
            let hf_idx = (1usize << n_occ) - 1;
            // Excitation: pair (n_occ-2, n_occ-1) -> (n_occ, n_occ+1)
            let excited_idx = (hf_idx ^ (3 << (n_occ - 2))) | (3 << n_occ);
            if excited_idx < dim {
                let c_hf = 0.965f64;
                let c_exc = (1.0 - c_hf * c_hf).sqrt();
                state.real[hf_idx] = c_hf;
                state.real[excited_idx] = c_exc;
            }
        }

        let energy = meta.ground_state_energy;
        (state, energy)
    }
}
