// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Ab initio VQE across the Period 1-2 benchmark molecules.
//!
//! For each molecule: STO-3G integrals computed from the geometry, restricted Hartree-Fock, a
//! frozen-core active space, the Jordan-Wigner qubit Hamiltonian, the exact (FCI) energy by
//! Lanczos, and a UCCSD variational energy by BFGS with analytic gradients.
//!
//! Usage: `vqe_molecules [FORMULA,FORMULA,...] [--no-freeze-core] [--active N]`

use std::fs;
use std::path::Path;
use t1000::vqe::driver::{solve, VqeConfig, VqeReport, CHEMICAL_ACCURACY};
use t1000::vqe::molecules::{by_formula, catalog, Molecule};

fn parse_args() -> (Vec<Molecule>, VqeConfig) {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut cfg = VqeConfig::default();
    let mut names: Option<Vec<String>> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--no-freeze-core" => cfg.freeze_core = false,
            "--active" => {
                cfg.active_orbitals = args.get(i + 1).and_then(|v| v.parse().ok());
                i += 1;
            }
            other => names = Some(other.split(',').map(str::to_string).collect()),
        }
        i += 1;
    }
    let molecules = match names {
        Some(list) => list
            .iter()
            .map(|f| by_formula(f).unwrap_or_else(|| panic!("unknown molecule {f}")))
            .collect(),
        None => catalog(),
    };
    (molecules, cfg)
}

fn print_header() {
    println!("=========================================================================");
    println!(" T-1000 AB INITIO MOLECULAR VQE (STO-3G, UCCSD, exact FCI reference)");
    println!("=========================================================================");
    println!(
        " {:<5} {:>3} {:>3} {:>6} {:>5} {:>13} {:>13} {:>13} {:>9} {:>6} {:>8}",
        "mol", "q", "par", "terms", "dets", "E(HF)", "E(FCI)", "E(VQE)", "err mHa", "corr%", "ms"
    );
}

fn print_row(r: &VqeReport) {
    let mark = if r.within_chemical_accuracy {
        "✓"
    } else {
        "✗"
    };
    println!(
        " {:<5} {:>3} {:>3} {:>6} {:>5} {:>13.6} {:>13.6} {:>13.6} {:>8.4}{} {:>6.2} {:>8.0}",
        r.formula,
        r.n_qubits,
        r.parameters,
        r.pauli_terms,
        r.determinants,
        r.hf_energy,
        r.fci_energy,
        r.vqe_energy,
        r.vqe_error_vs_fci * 1e3,
        mark,
        r.correlation_recovered_pct,
        r.time_vqe_ms
    );
}

fn main() {
    let (molecules, cfg) = parse_args();
    print_header();
    let mut reports = Vec::new();
    for molecule in &molecules {
        match solve(molecule, &cfg) {
            Ok(r) => {
                print_row(&r);
                reports.push(r);
            }
            Err(e) => println!(" {:<5} FAILED: {e}", molecule.formula),
        }
    }
    let ok = reports
        .iter()
        .filter(|r| r.within_chemical_accuracy)
        .count();
    println!(
        "\n {ok}/{} molecules within chemical accuracy ({:.2} mHa) of exact FCI",
        reports.len(),
        CHEMICAL_ACCURACY * 1e3
    );

    let dir = Path::new("results/data");
    let _ = fs::create_dir_all(dir);
    let path = dir.join("vqe_molecules_report.json");
    let json = serde_json::to_string_pretty(&reports).expect("serialise report");
    t1000::results::write(&path, json).expect("write report");
    println!(" [OK] report: {}", path.display());
}
