// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Shots needed for chemical accuracy when Hamiltonian terms are measured in qubit-wise
//! commuting groups, against measuring every term on its own.
//!
//! Both counts use the exact converged UCCSD state, no device noise, and the variance-optimal split
//! of shots; the grouped count includes the covariances between terms that share a measurement.

use serde::Serialize;
use std::fs;
use std::path::Path;
use t1000::vqe::driver::{solve_detailed, VqeConfig, CHEMICAL_ACCURACY};
use t1000::vqe::grouping::{group_moments, grouped_total_shots, qwc_groups};
use t1000::vqe::molecules::catalog;
use t1000::vqe::noise::{embed_sector_state, TermStatistics};
use t1000::vqe::uccsd::prepare_state;

#[derive(Serialize)]
struct Row {
    formula: String,
    qubits: usize,
    pauli_terms: usize,
    groups: usize,
    shots_term_by_term: f64,
    shots_grouped: f64,
    reduction_factor: f64,
    energy_check_error: f64,
}

fn main() {
    println!("=========================================================================");
    println!(
        " MEASUREMENT GROUPING: SHOTS FOR CHEMICAL ACCURACY ({:.2} mHa)",
        CHEMICAL_ACCURACY * 1e3
    );
    println!("=========================================================================");
    println!(
        " {:<5} {:>3} {:>7} {:>7} {:>15} {:>15} {:>10} {:>12}",
        "mol", "q", "terms", "groups", "term-by-term", "grouped", "reduction", "|E check|"
    );
    let cfg = VqeConfig::default();
    let mut rows = Vec::new();
    for molecule in catalog() {
        let s = solve_detailed(&molecule, &cfg).expect("solve");
        let psi = prepare_state(&s.excitations, &s.reference, &s.theta);
        let stats = TermStatistics::new(&s.qubit_hamiltonian, &s.sector, &psi);
        let full = embed_sector_state(&s.sector, &psi, s.qubit_hamiltonian.n_qubits);
        let groups = qwc_groups(&s.qubit_hamiltonian);
        let moments = group_moments(&s.qubit_hamiltonian, &groups, &full);
        let energy: f64 = s.qubit_hamiltonian.constant + moments.iter().map(|m| m.0).sum::<f64>();
        let row = Row {
            formula: s.report.formula.clone(),
            qubits: s.report.n_qubits,
            pauli_terms: s.qubit_hamiltonian.terms.len(),
            groups: groups.len(),
            shots_term_by_term: stats.optimal_total_shots(1.0, 0.0, false, CHEMICAL_ACCURACY),
            shots_grouped: grouped_total_shots(&moments, CHEMICAL_ACCURACY),
            reduction_factor: 0.0,
            energy_check_error: (energy - stats.ideal_energy()).abs(),
        };
        let row = Row {
            reduction_factor: row.shots_term_by_term / row.shots_grouped,
            ..row
        };
        println!(
            " {:<5} {:>3} {:>7} {:>7} {:>15.3e} {:>15.3e} {:>9.1}x {:>12.1e}",
            row.formula,
            row.qubits,
            row.pauli_terms,
            row.groups,
            row.shots_term_by_term,
            row.shots_grouped,
            row.reduction_factor,
            row.energy_check_error
        );
        rows.push(row);
    }
    println!("\n |E check| = |constant + sum of group means - sum of term means|: the grouped estimator is unbiased.");
    let dir = Path::new("results/data");
    let _ = fs::create_dir_all(dir);
    let path = dir.join("vqe_grouping_report.json");
    t1000::results::write(
        &path,
        serde_json::to_string_pretty(&rows).expect("serialise"),
    )
    .expect("write report");
    println!(" [OK] report: {}", path.display());
}
