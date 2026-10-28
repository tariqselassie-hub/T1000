// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Ab initio chemistry (`t1000::vqe`): VQE reports, qubit Hamiltonians, classical baselines,
//! ADAPT-VQE with a choice of operator pool, and bond scans that carry SCF orbitals along.
//! Results cross the boundary as JSON; the Python layer turns them into dicts.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{json, Value};
use t1000::vqe::adapt::{adapt_vqe, AdaptConfig};
use t1000::vqe::basis::Atom;
use t1000::vqe::circuit::circuit_cost;
use t1000::vqe::classical::{ccsd, cisd_energy, mp2_correlation};
use t1000::vqe::driver::{electronic_structure, solve, solve_detailed, Problem, VqeConfig};
use t1000::vqe::fermion::{build_hamiltonian, Sector, SparseMatrix};
use t1000::vqe::lanczos::ground_state;
use t1000::vqe::molecules::{by_formula, catalog, Molecule};
use t1000::vqe::pauli::jordan_wigner;
use t1000::vqe::uccsd::{generalized_excitations, uccsd_excitations, Excitation};

/// Atoms as Python passes them: `(Z, [x, y, z])` in angstrom.
pub type AtomSpec = (u32, [f64; 3]);

fn err(e: impl ToString) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// A catalog molecule by formula, or explicit atoms (exactly one of the two).
fn molecule_from(
    formula: Option<String>,
    atoms: Option<Vec<AtomSpec>>,
    charge: i32,
) -> PyResult<Molecule> {
    match (formula, atoms) {
        (Some(f), None) => by_formula(&f).ok_or_else(|| err(format!("{f} is not in the catalog"))),
        (None, Some(list)) if !list.is_empty() => Ok(Molecule {
            name: "custom",
            formula: "custom",
            atoms: list
                .iter()
                .map(|&(z, pos)| Atom::angstrom(z, pos))
                .collect(),
            charge,
        }),
        _ => Err(err("pass exactly one of formula= or atoms=")),
    }
}

fn config(freeze_core: bool, active_orbitals: Option<usize>) -> VqeConfig {
    VqeConfig {
        freeze_core,
        active_orbitals,
        ..VqeConfig::default()
    }
}

/// The determinant problem of a molecule: sector, sparse Hamiltonian, HF reference vector, FCI.
struct Determinants {
    problem: Problem,
    sector: Sector,
    h: SparseMatrix,
    reference: Vec<f64>,
    hf: f64,
    fci: f64,
}

fn determinants(molecule: &Molecule, cfg: &VqeConfig) -> Result<Determinants, String> {
    let problem = electronic_structure(molecule, cfg)?;
    if !problem.scf.converged {
        return Err(format!("SCF did not converge ({})", problem.scf.strategy));
    }
    let half = problem.ham.n_electrons / 2;
    let sector = Sector::new(problem.ham.n_orb, half, half);
    let h = build_hamiltonian(&problem.ham, &sector);
    let mut reference = vec![0.0; sector.dim()];
    reference[sector.hartree_fock()] = 1.0;
    let (fci, _) = ground_state(&h, &reference, 400, 1e-12)?;
    let hf = h.get(sector.hartree_fock(), sector.hartree_fock());
    Ok(Determinants {
        problem,
        sector,
        h,
        reference,
        hf,
        fci,
    })
}

/// Ab initio VQE (STO-3G, RHF, UCCSD) for a catalog formula or explicit atoms; returns JSON.
#[pyfunction]
#[pyo3(signature = (formula=None, atoms=None, charge=0, freeze_core=true, active_orbitals=None))]
pub fn vqe_solve_json(
    py: Python<'_>,
    formula: Option<String>,
    atoms: Option<Vec<AtomSpec>>,
    charge: i32,
    freeze_core: bool,
    active_orbitals: Option<usize>,
) -> PyResult<String> {
    let molecule = molecule_from(formula, atoms, charge)?;
    let cfg = config(freeze_core, active_orbitals);
    let report = py.detach(move || solve(&molecule, &cfg)).map_err(err)?;
    serde_json::to_string(&report).map_err(err)
}

/// Formulas of the built-in VQE molecule catalog.
#[pyfunction]
pub fn vqe_catalog() -> Vec<String> {
    catalog().iter().map(|m| m.formula.to_string()).collect()
}

/// The Jordan-Wigner qubit Hamiltonian of the active space, as JSON.
///
/// Qubit `2p + s` is spatial orbital `p` with spin `s` (0 = alpha); the Hartree-Fock state is the
/// basis state with the lowest `n_electrons` qubits set.
#[pyfunction]
#[pyo3(signature = (formula=None, atoms=None, charge=0, freeze_core=true, active_orbitals=None))]
pub fn hamiltonian_json(
    py: Python<'_>,
    formula: Option<String>,
    atoms: Option<Vec<AtomSpec>>,
    charge: i32,
    freeze_core: bool,
    active_orbitals: Option<usize>,
) -> PyResult<String> {
    let molecule = molecule_from(formula, atoms, charge)?;
    let cfg = config(freeze_core, active_orbitals);
    let out: Result<Value, String> = py.detach(move || {
        let p = electronic_structure(&molecule, &cfg)?;
        let qh = jordan_wigner(&p.ham);
        let n_e = p.ham.n_electrons;
        let terms: Vec<Value> = qh
            .terms
            .iter()
            .map(|t| json!([t.coeff, qh.label(t)]))
            .collect();
        Ok(json!({
            "n_qubits": qh.n_qubits,
            "n_electrons": n_e,
            "constant": qh.constant,
            "terms": terms,
            "hf_state": (1u64 << n_e) - 1,
            "hf_energy": p.scf.energy,
            "scf_converged": p.scf.converged,
            "frozen_orbitals": p.frozen,
        }))
    });
    Ok(out.map_err(err)?.to_string())
}

/// HF, MP2, CISD, CCSD and exact FCI in one active space, as JSON (energies in hartree).
#[pyfunction]
#[pyo3(signature = (formula=None, atoms=None, charge=0, freeze_core=true, active_orbitals=None))]
pub fn classical_json(
    py: Python<'_>,
    formula: Option<String>,
    atoms: Option<Vec<AtomSpec>>,
    charge: i32,
    freeze_core: bool,
    active_orbitals: Option<usize>,
) -> PyResult<String> {
    let molecule = molecule_from(formula, atoms, charge)?;
    let cfg = config(freeze_core, active_orbitals);
    let out: Result<Value, String> = py.detach(move || {
        let d = determinants(&molecule, &cfg)?;
        let cc = ccsd(&d.h, &d.sector, &uccsd_excitations(&d.sector));
        Ok(json!({
            "n_qubits": 2 * d.problem.ham.n_orb,
            "determinants": d.sector.dim(),
            "hf": d.hf,
            "mp2": d.hf + mp2_correlation(&d.problem.ham),
            "cisd": cisd_energy(&d.problem.ham, &d.sector)?,
            "ccsd": { "energy": cc.energy, "converged": cc.converged, "iterations": cc.iterations,
                      "max_amplitude": cc.max_amplitude },
            "fci": d.fci,
        }))
    });
    Ok(out.map_err(err)?.to_string())
}

/// The ADAPT operator pool by name.
fn pool(name: &str, sector: &Sector) -> Result<Vec<Excitation>, String> {
    match name {
        "occ-virt" => Ok(uccsd_excitations(sector)),
        "gsd" => Ok(generalized_excitations(sector)),
        "qubit-gsd" => Ok(generalized_excitations(sector)
            .iter()
            .map(Excitation::to_qubit)
            .collect()),
        _ => Err(format!(
            "unknown pool {name:?}: use 'occ-virt', 'gsd' or 'qubit-gsd'"
        )),
    }
}

/// ADAPT-VQE with the chosen pool against exact FCI, as JSON.
#[pyfunction]
#[pyo3(signature = (formula=None, atoms=None, charge=0, freeze_core=true, active_orbitals=None,
                    pool_name="occ-virt".to_string(), gradient_tolerance=1e-3, max_operators=250))]
#[allow(clippy::too_many_arguments)]
pub fn adapt_json(
    py: Python<'_>,
    formula: Option<String>,
    atoms: Option<Vec<AtomSpec>>,
    charge: i32,
    freeze_core: bool,
    active_orbitals: Option<usize>,
    pool_name: String,
    gradient_tolerance: f64,
    max_operators: usize,
) -> PyResult<String> {
    let molecule = molecule_from(formula, atoms, charge)?;
    let cfg = config(freeze_core, active_orbitals);
    let adapt_cfg = AdaptConfig {
        gradient_norm_tolerance: gradient_tolerance,
        max_operators,
        ..AdaptConfig::default()
    };
    let out: Result<Value, String> = py.detach(move || {
        let d = determinants(&molecule, &cfg)?;
        let ops = pool(&pool_name, &d.sector)?;
        let r = adapt_vqe(&d.h, &ops, &d.reference, &adapt_cfg);
        let ansatz: Vec<Excitation> = r.selected.iter().map(|&k| ops[k].clone()).collect();
        let mut unique = r.selected.clone();
        unique.sort_unstable();
        unique.dedup();
        let (cnot, model) = if pool_name == "qubit-gsd" {
            (
                ansatz
                    .iter()
                    .map(|e| if e.is_double() { 13 } else { 2 })
                    .sum::<usize>(),
                "qubit-excitation circuits",
            )
        } else {
            (circuit_cost(&ansatz).cnot, "naive Jordan-Wigner")
        };
        Ok(json!({
            "pool": pool_name, "pool_size": ops.len(),
            "energy": r.energy, "hf": d.hf, "fci": d.fci, "error_mha": (r.energy - d.fci) * 1e3,
            "operators": ansatz.iter().map(|e| e.label.clone()).collect::<Vec<_>>(),
            "parameters": ansatz.len(), "unique_operators": unique.len(),
            "theta": r.theta, "cnot": cnot, "cnot_model": model,
            "converged": r.converged, "final_gradient_norm": r.final_gradient_norm,
            "energy_evaluations": r.energy_evaluations, "steps": r.steps,
        }))
    });
    Ok(out.map_err(err)?.to_string())
}

/// UCCSD-VQE along a list of geometries, starting each SCF from the previous point's orbitals
/// (`follow_scf`); `seed_only` keeps the followed SCF branch instead of the lowest one found.
#[pyfunction]
#[pyo3(signature = (geometries, charge=0, freeze_core=true, active_orbitals=None, follow_scf=true, seed_only=false))]
pub fn scan_json(
    py: Python<'_>,
    geometries: Vec<Vec<AtomSpec>>,
    charge: i32,
    freeze_core: bool,
    active_orbitals: Option<usize>,
    follow_scf: bool,
    seed_only: bool,
) -> PyResult<String> {
    let molecules = geometries
        .into_iter()
        .map(|g| molecule_from(None, Some(g), charge))
        .collect::<PyResult<Vec<_>>>()?;
    let points: Vec<Value> = py.detach(move || {
        let mut guess = None;
        molecules
            .iter()
            .map(|m| {
                let cfg = VqeConfig {
                    scf_guess: if follow_scf { guess.clone() } else { None },
                    scf_seed_only: seed_only,
                    ..config(freeze_core, active_orbitals)
                };
                match solve_detailed(m, &cfg) {
                    Ok(s) => {
                        guess = Some(s.scf_coefficients);
                        serde_json::to_value(&s.report).unwrap_or(Value::Null)
                    }
                    Err(e) => json!({ "error": e }),
                }
            })
            .collect()
    });
    Ok(Value::Array(points).to_string())
}
