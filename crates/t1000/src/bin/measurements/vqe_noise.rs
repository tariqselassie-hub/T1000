// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! How much device noise and how many shots the molecular VQE can tolerate.
//!
//! Part 1 compiles each converged UCCSD circuit to Pauli rotations (CNOT and 1-qubit gate
//! counts) and applies depolarising gate error, readout error and shot noise to the exact
//! optimum: the energy bias, the shots needed, and the largest two-qubit error compatible with
//! chemical accuracy. Part 2 runs SPSA on sampled energies only, the way a device would.

use serde::Serialize;
use std::fs;
use std::path::Path;
use t1000::vqe::circuit::{circuit_cost, CircuitCost};
use t1000::vqe::driver::{solve_detailed, SolvedProblem, VqeConfig, CHEMICAL_ACCURACY};
use t1000::vqe::molecules::{by_formula, catalog};
use t1000::vqe::noise::{
    circuit_fidelity, max_two_qubit_error, profiles, DeviceProfile, TermStatistics,
};
use t1000::vqe::noisy_vqe::{spsa_vqe, NoisyConfig};
use t1000::vqe::uccsd::prepare_state;

#[derive(Serialize)]
struct ResourceRow {
    formula: String,
    qubits: usize,
    pauli_terms: usize,
    rotations: usize,
    cnot: usize,
    one_qubit_gates: usize,
    ideal_energy: f64,
    identity_coefficient: f64,
    noiseless_shots_for_chemical_accuracy: f64,
    max_two_qubit_error_for_chemical_accuracy: f64,
    by_profile: Vec<ProfileRow>,
}

#[derive(Serialize)]
struct ProfileRow {
    profile: String,
    circuit_fidelity: f64,
    unmitigated_bias_mha: f64,
    mitigated_shots_for_chemical_accuracy: f64,
}

#[derive(Serialize)]
struct SpsaRow {
    formula: String,
    scenario: String,
    shots_per_term: u64,
    mean_error_mha: f64,
    runs: usize,
}

fn statistics(s: &SolvedProblem) -> TermStatistics {
    let psi = prepare_state(&s.excitations, &s.reference, &s.theta);
    TermStatistics::new(&s.qubit_hamiltonian, &s.sector, &psi)
}

fn resource_row(s: &SolvedProblem, cost: &CircuitCost, stats: &TermStatistics) -> ResourceRow {
    let energy = s.report.vqe_energy;
    let by_profile = profiles()
        .iter()
        .map(|p: &DeviceProfile| {
            let f = circuit_fidelity(cost, p.p1, p.p2);
            ProfileRow {
                profile: p.name.to_string(),
                circuit_fidelity: f,
                unmitigated_bias_mha: (stats.noisy_energy(f, p.readout) - energy) * 1e3,
                mitigated_shots_for_chemical_accuracy: stats.optimal_total_shots(
                    f,
                    p.readout,
                    true,
                    CHEMICAL_ACCURACY,
                ),
            }
        })
        .collect();
    ResourceRow {
        formula: s.report.formula.clone(),
        qubits: s.report.n_qubits,
        pauli_terms: s.report.pauli_terms,
        rotations: cost.rotations,
        cnot: cost.cnot,
        one_qubit_gates: cost.one_qubit,
        ideal_energy: energy,
        identity_coefficient: stats.constant,
        noiseless_shots_for_chemical_accuracy: stats.optimal_total_shots(
            1.0,
            0.0,
            false,
            CHEMICAL_ACCURACY,
        ),
        max_two_qubit_error_for_chemical_accuracy: max_two_qubit_error(
            cost,
            energy,
            stats.constant,
            CHEMICAL_ACCURACY,
        ),
        by_profile,
    }
}

/// Scientific notation, or `inf` where a count is beyond any physical meaning.
fn sci(v: f64) -> String {
    if v.is_finite() && v < 1e30 {
        format!("{v:.2e}")
    } else {
        "inf".to_string()
    }
}

fn print_resources(rows: &[ResourceRow]) {
    println!("\n PART 1: RESOURCES AND NOISE TOLERANCE OF THE CONVERGED CIRCUIT");
    println!(
        " {:<5} {:>3} {:>6} {:>7} {:>7} {:>10} {:>11} {:>14} {:>13} {:>12}",
        "mol",
        "q",
        "terms",
        "CNOT",
        "1q",
        "F(cepheus)",
        "bias mHa",
        "shots (ideal)",
        "shots (mit.)",
        "max p2"
    );
    for r in rows {
        let cep = &r.by_profile[1];
        println!(
            " {:<5} {:>3} {:>6} {:>7} {:>7} {:>10.2e} {:>11.1} {:>14} {:>13} {:>12.1e}",
            r.formula,
            r.qubits,
            r.pauli_terms,
            r.cnot,
            r.one_qubit_gates,
            cep.circuit_fidelity,
            cep.unmitigated_bias_mha,
            sci(r.noiseless_shots_for_chemical_accuracy),
            sci(cep.mitigated_shots_for_chemical_accuracy),
            r.max_two_qubit_error_for_chemical_accuracy
        );
    }
    println!(
        "\n shots: total over all terms for {:.2} mHa standard error, variance-optimal split;",
        CHEMICAL_ACCURACY * 1e3
    );
    println!(" 'mit.' = known-noise mitigation at the cepheus profile. max p2: two-qubit error (p1 = p2/10)");
    println!(" at which depolarising bias alone reaches chemical accuracy.");
}

fn spsa_scenarios() -> Vec<(&'static str, f64, f64, bool)> {
    vec![
        ("shot noise only", 1.0, 0.0, false),
        ("100x-better device, raw", 0.97, 2.5e-4, false),
        ("100x-better device, mitigated", 0.97, 2.5e-4, true),
    ]
}

fn run_spsa(solved: &SolvedProblem, shots: u64, scenario: (&str, f64, f64, bool)) -> SpsaRow {
    let seeds = [11u64, 12, 13];
    let errors: Vec<f64> = seeds
        .iter()
        .map(|&seed| {
            let cfg = NoisyConfig {
                shots_per_term: shots,
                fidelity: scenario.1,
                readout: scenario.2,
                mitigate: scenario.3,
                iterations: 300,
                seed,
            };
            spsa_vqe(solved, &cfg).error_vs_fci
        })
        .collect();
    SpsaRow {
        formula: solved.report.formula.clone(),
        scenario: scenario.0.to_string(),
        shots_per_term: shots,
        mean_error_mha: errors.iter().sum::<f64>() / errors.len() as f64 * 1e3,
        runs: errors.len(),
    }
}

fn print_spsa(rows: &[SpsaRow]) {
    println!("\n PART 2: SPSA ON SAMPLED ENERGIES ONLY (300 iterations, mean of 3 seeds)");
    println!(" error = exact energy of the returned parameters minus FCI, mHa");
    println!(
        " {:<5} {:<32} {:>10} {:>12}",
        "mol", "scenario", "shots/term", "error mHa"
    );
    for r in rows {
        println!(
            " {:<5} {:<32} {:>10} {:>12.3}",
            r.formula, r.scenario, r.shots_per_term, r.mean_error_mha
        );
    }
}

fn main() {
    println!("=========================================================================");
    println!(" T-1000 NOISY MOLECULAR VQE: GATE COUNTS, SHOT BUDGETS, ERROR TOLERANCE");
    println!("=========================================================================");
    let cfg = VqeConfig::default();
    let mut resources = Vec::new();
    for molecule in catalog() {
        let solved = solve_detailed(&molecule, &cfg).expect("solve");
        let cost = circuit_cost(&solved.excitations);
        resources.push(resource_row(&solved, &cost, &statistics(&solved)));
    }
    print_resources(&resources);

    let mut spsa = Vec::new();
    for (formula, shot_counts) in [
        ("H2", vec![100u64, 1_000, 10_000, 100_000]),
        ("HeH+", vec![1_000, 100_000]),
        ("LiH", vec![1_000, 100_000]),
    ] {
        let solved = solve_detailed(&by_formula(formula).expect("molecule"), &cfg).expect("solve");
        for scenario in spsa_scenarios() {
            for &shots in &shot_counts {
                spsa.push(run_spsa(&solved, shots, scenario));
            }
        }
    }
    print_spsa(&spsa);

    let dir = Path::new("results/data");
    let _ = fs::create_dir_all(dir);
    let path = dir.join("vqe_noise_report.json");
    let json = serde_json::json!({ "resources": resources, "spsa": spsa });
    t1000::results::write(
        &path,
        serde_json::to_string_pretty(&json).expect("serialise"),
    )
    .expect("write report");
    println!("\n [OK] report: {}", path.display());
}
