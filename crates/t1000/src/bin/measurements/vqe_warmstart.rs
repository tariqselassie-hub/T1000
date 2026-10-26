// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Does a classical warm start save optimizer work for the UCCSD VQE?
//!
//! The cold start is `θ = 0` (the Hartree-Fock state), as in `vqe::driver`. Two warm starts:
//!
//! * `PT1` (cheap): first-order perturbation amplitudes with the Hamiltonian's own diagonal as
//!   denominators, `c_I = -H_{I0} / (H_II - H_00)` for every determinant `I`, projected on the
//!   first-order response of each excitation, so `θ_k = <response_k | c>`. It needs one
//!   matrix-vector product and the diagonal.
//! * `oracle`: the same projection of the exact (full-CI) coefficients `c_I = ψ_I / ψ_0`. This is the
//!   best a first-order classical estimate could do and needs the exact answer, so it is an upper
//!   bound on what a warm start can give, not a usable method.
//!
//! Each run uses the same BFGS and tolerances as the driver; the table gives the energy error of the
//! starting point and the iterations and gradient evaluations to converge.

use serde::Serialize;
use std::fs;
use std::path::Path;
use t1000::vqe::driver::{solve_detailed, VqeConfig};
use t1000::vqe::lanczos::ground_state;
use t1000::vqe::molecules::catalog;
use t1000::vqe::optimize::bfgs;
use t1000::vqe::uccsd::{energy_and_gradient, Excitation};

#[derive(Serialize)]
struct Start {
    start_error_mha: f64,
    iterations: usize,
    evaluations: usize,
    final_error_mha: f64,
    converged: bool,
}

#[derive(Serialize)]
struct Row {
    formula: String,
    parameters: usize,
    hf_error_mha: f64,
    cold: Start,
    pt1: Start,
    oracle: Start,
}

/// Projection of `c` (coefficients of determinants other than the reference) on each excitation's
/// first-order response from the reference.
fn project(excitations: &[Excitation], reference: &[f64], c: &[f64]) -> Vec<f64> {
    let eps = 1e-6;
    excitations
        .iter()
        .map(|e| {
            let mut psi = reference.to_vec();
            e.rotate(&mut psi, eps);
            psi.iter()
                .zip(reference)
                .zip(c)
                .map(|((p, r), ci)| (p - r) / eps * ci)
                .sum()
        })
        .collect()
}

fn run(cfg: &VqeConfig, s: &t1000::vqe::driver::SolvedProblem, fci: f64, x0: Vec<f64>) -> Start {
    let f = |t: &[f64]| energy_and_gradient(&s.hamiltonian, &s.excitations, &s.reference, t);
    let start_error = f(&x0).0 - fci;
    let r = bfgs(f, x0, cfg.max_iterations, cfg.gradient_tolerance);
    Start {
        start_error_mha: start_error * 1e3,
        iterations: r.iterations,
        evaluations: r.evaluations,
        final_error_mha: (r.value - fci) * 1e3,
        converged: r.converged,
    }
}

fn main() {
    println!("=========================================================================");
    println!(" UCCSD WARM STARTS: ITERATIONS AND GRADIENT EVALUATIONS TO CONVERGE");
    println!("=========================================================================");
    println!(
        " {:<5} {:>6} {:>10} | {:>22} | {:>22} | {:>22}",
        "mol",
        "params",
        "HF err mHa",
        "cold: start/it/evals",
        "PT1: start/it/evals",
        "oracle: start/it/evals"
    );
    let cfg = VqeConfig::default();
    let mut rows = Vec::new();
    for molecule in catalog() {
        let s = solve_detailed(&molecule, &cfg).expect("solve");
        let h = &s.hamiltonian;
        let n = h.n;
        let hf = s.sector.hartree_fock();
        let (fci, psi0) = ground_state(h, &s.reference, 300, 1e-11).expect("fci");

        // PT1 coefficients
        let mut col = vec![0.0; n];
        h.apply(&s.reference, &mut col);
        let h00 = h.get(hf, hf);
        let mut c_pt = vec![0.0; n];
        for i in 0..n {
            if i != hf {
                let gap = h.get(i, i) - h00;
                if gap.abs() > 1e-9 {
                    c_pt[i] = -col[i] / gap;
                }
            }
        }
        // oracle coefficients
        let sign = psi0[hf].signum();
        let c_fci: Vec<f64> = (0..n)
            .map(|i| {
                if i == hf {
                    0.0
                } else {
                    sign * psi0[i] / psi0[hf].abs()
                }
            })
            .collect();

        let theta_pt = project(&s.excitations, &s.reference, &c_pt);
        let theta_or = project(&s.excitations, &s.reference, &c_fci);

        let hf_error = (h00 - fci) * 1e3;
        let cold = run(&cfg, &s, fci, vec![0.0; s.excitations.len()]);
        let pt1 = run(&cfg, &s, fci, theta_pt);
        let oracle = run(&cfg, &s, fci, theta_or);
        let cell = |x: &Start| {
            format!(
                "{:>8.2}/{:>3}/{:>5}",
                x.start_error_mha, x.iterations, x.evaluations
            )
        };
        println!(
            " {:<5} {:>6} {:>10.2} | {:>22} | {:>22} | {:>22}",
            s.report.formula,
            s.excitations.len(),
            hf_error,
            cell(&cold),
            cell(&pt1),
            cell(&oracle)
        );
        rows.push(Row {
            formula: s.report.formula.clone(),
            parameters: s.excitations.len(),
            hf_error_mha: hf_error,
            cold,
            pt1,
            oracle,
        });
    }
    println!("\n start = energy error of the starting point in mHa; it = BFGS iterations; evals = energy+gradient evaluations.");
    println!(
        " Every run is checked for the same converged energy: see final_error_mha in the report."
    );
    let worst = rows
        .iter()
        .flat_map(|r| [&r.cold, &r.pt1, &r.oracle])
        .map(|x| x.final_error_mha.abs())
        .fold(0.0, f64::max);
    println!(" largest final error against full CI over all runs: {worst:.4} mHa");
    let dir = Path::new("results/data");
    let _ = fs::create_dir_all(dir);
    let path = dir.join("vqe_warmstart_report.json");
    t1000::results::write(
        &path,
        serde_json::to_string_pretty(&rows).expect("serialise"),
    )
    .expect("write report");
    println!(" [OK] report: {}", path.display());
}
