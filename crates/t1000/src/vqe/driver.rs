// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! End-to-end molecular VQE: geometry → integrals → Hartree-Fock → active space → exact (FCI)
//! reference → UCCSD optimisation, with a timed, serialisable report.

use super::basis::sto3g_basis;
use super::fermion::{build_hamiltonian, Sector};
use super::integrals::compute_ao_integrals;
use super::lanczos::ground_state;
use super::mo::{active_space, hartree_fock_energy, ActiveHamiltonian};
use super::molecules::Molecule;
use super::optimize::bfgs;
use super::pauli::jordan_wigner;
use super::scf::{rhf, RhfResult};
use super::uccsd::{energy_and_gradient, uccsd_excitations};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Chemical accuracy: 1 kcal/mol in hartree.
pub const CHEMICAL_ACCURACY: f64 = 1.594e-3;

/// Solver settings.
#[derive(Debug, Clone)]
pub struct VqeConfig {
    /// Hold the 1s core orbitals of Li-F doubly occupied.
    pub freeze_core: bool,
    /// Limit the correlated orbitals (lowest virtuals kept), or all if `None`.
    pub active_orbitals: Option<usize>,
    pub max_iterations: usize,
    pub gradient_tolerance: f64,
}

impl Default for VqeConfig {
    fn default() -> Self {
        VqeConfig {
            freeze_core: true,
            active_orbitals: None,
            max_iterations: 400,
            gradient_tolerance: 1e-6,
        }
    }
}

/// Everything measured for one molecule. Energies are in hartree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VqeReport {
    pub formula: String,
    pub name: String,
    pub n_basis_functions: usize,
    pub frozen_orbitals: usize,
    pub active_orbitals: usize,
    pub active_electrons: usize,
    pub n_qubits: usize,
    pub pauli_terms: usize,
    pub max_pauli_weight: usize,
    pub determinants: usize,
    pub parameters: usize,
    pub nuclear_repulsion: f64,
    pub hf_energy: f64,
    pub fci_energy: f64,
    pub vqe_energy: f64,
    pub correlation_energy: f64,
    pub correlation_recovered_pct: f64,
    pub vqe_error_vs_fci: f64,
    pub within_chemical_accuracy: bool,
    pub scf_iterations: usize,
    pub optimizer_iterations: usize,
    pub energy_evaluations: usize,
    pub converged: bool,
    pub gradient_max: f64,
    pub history: Vec<f64>,
    pub time_integrals_ms: f64,
    pub time_fci_ms: f64,
    pub time_vqe_ms: f64,
}

/// SCF and the active-space Hamiltonian of a molecule.
struct Problem {
    n_ao: usize,
    nuclear_repulsion: f64,
    scf: RhfResult,
    ham: ActiveHamiltonian,
    frozen: usize,
}

fn electronic_structure(molecule: &Molecule, cfg: &VqeConfig) -> Result<Problem, String> {
    let basis = sto3g_basis(&molecule.atoms)?;
    let ints = compute_ao_integrals(&basis, &molecule.atoms);
    let scf = rhf(&ints, molecule.electrons())?;
    if !scf.converged {
        return Err(format!("{}: SCF did not converge", molecule.formula));
    }
    let frozen = if cfg.freeze_core {
        molecule.core_orbitals()
    } else {
        0
    };
    let n_virtual_kept = cfg
        .active_orbitals
        .map(|a| a.saturating_sub(scf.n_occupied - frozen));
    let n_active = n_virtual_kept.map(|v| scf.n_occupied - frozen + v);
    let ham = active_space(&ints, &scf, frozen, n_active)?;
    Ok(Problem {
        n_ao: ints.n,
        nuclear_repulsion: ints.nuclear_repulsion,
        scf,
        ham,
        frozen,
    })
}

/// Runs the full pipeline for one molecule.
pub fn solve(molecule: &Molecule, cfg: &VqeConfig) -> Result<VqeReport, String> {
    let t0 = Instant::now();
    let p = electronic_structure(molecule, cfg)?;
    let n_qubits = 2 * p.ham.n_orb;
    let qubit_h = jordan_wigner(&p.ham);
    let half = p.ham.n_electrons / 2;
    let sector = Sector::new(p.ham.n_orb, half, half);
    let h = build_hamiltonian(&p.ham, &sector);
    let time_integrals_ms = t0.elapsed().as_secs_f64() * 1e3;

    let mut reference = vec![0.0; sector.dim()];
    reference[sector.hartree_fock()] = 1.0;
    let t1 = Instant::now();
    let (fci_energy, _) = ground_state(&h, &reference, 300, 1e-11)?;
    let time_fci_ms = t1.elapsed().as_secs_f64() * 1e3;

    let t2 = Instant::now();
    let excitations = uccsd_excitations(&sector);
    let result = bfgs(
        |theta| energy_and_gradient(&h, &excitations, &reference, theta),
        vec![0.0; excitations.len()],
        cfg.max_iterations,
        cfg.gradient_tolerance,
    );
    let time_vqe_ms = t2.elapsed().as_secs_f64() * 1e3;

    let hf_energy = hartree_fock_energy(&p.ham);
    let correlation_energy = fci_energy - hf_energy;
    let recovered = (result.value - hf_energy) / correlation_energy * 100.0;
    let error = result.value - fci_energy;
    Ok(VqeReport {
        formula: molecule.formula.to_string(),
        name: molecule.name.to_string(),
        n_basis_functions: p.n_ao,
        frozen_orbitals: p.frozen,
        active_orbitals: p.ham.n_orb,
        active_electrons: p.ham.n_electrons,
        n_qubits,
        pauli_terms: qubit_h.terms.len(),
        max_pauli_weight: qubit_h.max_weight(),
        determinants: sector.dim(),
        parameters: excitations.len(),
        nuclear_repulsion: p.nuclear_repulsion,
        hf_energy,
        fci_energy,
        vqe_energy: result.value,
        correlation_energy,
        correlation_recovered_pct: if correlation_energy.abs() > 1e-12 {
            recovered
        } else {
            100.0
        },
        vqe_error_vs_fci: error,
        within_chemical_accuracy: error.abs() < CHEMICAL_ACCURACY,
        scf_iterations: p.scf.iterations,
        optimizer_iterations: result.iterations,
        energy_evaluations: result.evaluations,
        converged: result.converged,
        gradient_max: result.grad_max,
        history: result.history,
        time_integrals_ms,
        time_fci_ms,
        time_vqe_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::super::molecules::by_formula;
    use super::*;

    fn run(formula: &str) -> VqeReport {
        solve(&by_formula(formula).unwrap(), &VqeConfig::default()).unwrap()
    }

    #[test]
    fn h2_vqe_reaches_the_exact_energy_and_the_published_fci_value() {
        let r = run("H2");
        assert!(r.converged);
        // STO-3G H₂ at 0.7414 Å: FCI -1.13727 hartree (published), HF -1.11668.
        assert!(
            (r.fci_energy + 1.137_27).abs() < 2e-5,
            "FCI {}",
            r.fci_energy
        );
        assert!(r.vqe_error_vs_fci.abs() < 1e-8, "{}", r.vqe_error_vs_fci);
        assert!(r.correlation_recovered_pct > 99.99);
        assert_eq!((r.n_qubits, r.parameters), (4, 3));
    }

    #[test]
    fn heh_plus_and_lih_reach_chemical_accuracy() {
        for formula in ["HeH+", "LiH"] {
            let r = run(formula);
            assert!(r.converged, "{formula}");
            assert!(
                r.within_chemical_accuracy,
                "{formula}: error {}",
                r.vqe_error_vs_fci
            );
            assert!(
                r.vqe_energy >= r.fci_energy - 1e-9,
                "variational bound for {formula}"
            );
            assert!(r.fci_energy < r.hf_energy, "{formula} must be correlated");
        }
    }

    #[test]
    fn the_jordan_wigner_hamiltonian_reproduces_the_determinant_hamiltonian() {
        let molecule = by_formula("LiH").unwrap();
        let p = electronic_structure(&molecule, &VqeConfig::default()).unwrap();
        let qubit_h = jordan_wigner(&p.ham);
        let sector = Sector::new(p.ham.n_orb, 1, 1);
        let h = build_hamiltonian(&p.ham, &sector);
        for (a, &bra) in sector.dets.iter().enumerate() {
            for (b, &ket) in sector.dets.iter().enumerate() {
                let q = qubit_h.matrix_element(bra, ket);
                assert!(
                    (q - h.get(a, b)).abs() < 1e-10,
                    "<{bra:b}|H|{ket:b}>: {q} vs {}",
                    h.get(a, b)
                );
            }
        }
        // The qubit expectation value of a determinant-space state equals ψᵀHψ.
        let mut psi = vec![0.0; 1 << qubit_h.n_qubits];
        let mut dense = vec![0.0; sector.dim()];
        for (i, &d) in sector.dets.iter().enumerate() {
            let amp = ((i + 1) as f64).sin();
            psi[d as usize] = amp;
            dense[i] = amp;
        }
        let norm: f64 = dense.iter().map(|v| v * v).sum::<f64>().sqrt();
        psi.iter_mut().for_each(|v| *v /= norm);
        dense.iter_mut().for_each(|v| *v /= norm);
        let mut hv = vec![0.0; sector.dim()];
        h.apply(&dense, &mut hv);
        let want: f64 = dense.iter().zip(&hv).map(|(a, b)| a * b).sum();
        assert!((qubit_h.expectation(&psi) - want).abs() < 1e-10);
    }
}
