// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Decoded Quantum Interferometry (DQI) & Optimal Polynomial Intersection (OPI) Benchmark
//!
//! Evaluates Target 01 from *The Compendium of Super-Quadratic Computational Quantum Advantages*
//! (arXiv:2610.32595). Demonstrates the definitive computational showdown between classical
//! unique decoding (Berlekamp-Welch), classical exhaustive search (Maximum-LINSAT), and
//! quantum interference in dual-space Reed-Solomon qudit registers.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use t1000::dqi::{field::FieldElement, run_dqi_showdown, DqiShowdownReport};
use t1000::results;

/// Complete experiment benchmark report for DQI advantage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DqiExperimentReport {
    pub target_designation: String,
    pub compendium_reference: String,
    pub field_characteristic: u32,
    pub block_length: usize,
    pub message_dimension: usize,
    pub total_hilbert_dim: usize,
    pub classical_search_space: usize,
    pub sweep_results: Vec<DqiShowdownReport>,
    pub quantum_advantage_verified: bool,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let record = args.iter().any(|a| a == "--record" || a == "-r")
        || std::env::var("T1000_RECORD").map(|v| v == "1").unwrap_or(false);

    println!("[1/3] Initializing DQI Quantum Interferometer for Reed-Solomon [5, 2] over F_7...");
    let p = 7u32;
    let n = 5usize;
    let k = 2usize;
    let hilbert_dim = (p as usize).pow(n as u32);
    let search_space = (p as usize).pow(k as u32);

    // Planted polynomial: P(x) = 3 + 2x (degree < 2)
    let planted_poly = vec![FieldElement::new(3, p), FieldElement::new(2, p)];

    println!("[2/3] Executing Quantum vs Classical Decoding Showdown across noise regimes t in [0..3]...");

    let mut sweep_results = Vec::new();
    let error_counts = vec![0, 1, 2, 3];

    for &errors in &error_counts {
        let report = run_dqi_showdown(p, n, k, errors, &planted_poly);
        sweep_results.push(report);
    }

    println!("[3/3] Compiling Comparative Telemetry Matrix...\n");

    println!("┌────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("║       TARGET-01: DECODED QUANTUM INTERFEROMETRY (DQI) FOR OPTIMAL POLYNOMIAL OPI       ║");
    println!("║       Compendium Super-Quadratic / Exponential Advantage Showdown (arXiv:2610.32595)   ║");
    println!("├────────────────────────────────────────────────────────────────────────────────────────┤");
    println!("│ Galois Field:                   F_{:<2} (Prime characteristic)                           │", p);
    println!("│ Reed-Solomon Code:              [n={}, k={}] (MDS Singleton bound d_min = {})            │", n, k, n - k + 1);
    println!("│ Classical Unique Radius:        t <= {} errors (Berlekamp-Welch limit)                   │", (n - k) / 2);
    println!("│ Classical Guruswami-Sudan:      t <= {:.3} errors (List-decoding bound)                │", (n as f64) - ((n * k) as f64).sqrt());
    println!("│ Classical Search Complexity:    O(p^k) = {} candidate polynomials                       │", search_space);
    println!("│ Quantum Hilbert Space:          (C^{})^5 = {} complex amplitudes ({:.1} KB in L2)        │", p, hilbert_dim, (hilbert_dim * 16) as f64 / 1024.0);
    println!("├────────────────────────────────────────────────────────────────────────────────────────┤");
    println!("│ NOISE SWEEP COMPARISON: CLASSICAL VS QUANTUM DQI                                       │");
    println!("├───────┬────────────┬────────────────┬──────────────┬──────────────┬────────────────────┤");
    println!("│ Noise │ Agreements │ Classical BW   │ Class. Search│ DQI Status   │ Constructive Ampl. │");
    println!("├───────┼────────────┼────────────────┼──────────────┼──────────────┼────────────────────┤");

    let mut advantage_verified = false;

    for r in &sweep_results {
        let bw_str = if r.classical_bw_success {
            "SUCCESS (Exact)"
        } else {
            "FAILED  (0% Rec)"
        };

        let dqi_str = if r.dqi_success {
            "SUCCESS (Peak)"
        } else {
            "FAILED"
        };

        println!(
            "│ t={:<2}  │ {:>2}/{} ({:>2.0}%) │ {:<14} │ {:>6} us   │ {:<12} │ {:>12.3}x       │",
            r.errors,
            r.agreement_count,
            n,
            (r.agreement_count as f64 / n as f64) * 100.0,
            bw_str,
            r.classical_search_micros,
            dqi_str,
            r.dqi_amplification
        );

        // Verification condition: t > t_unique where classical BW fails but DQI succeeds with amplification > 1.0
        if r.errors > r.unique_threshold && !r.classical_bw_success && r.dqi_success && r.dqi_amplification > 1.0 {
            advantage_verified = true;
        }
    }

    println!("├───────┴────────────┴────────────────┴──────────────┴──────────────┴────────────────────┤");
    println!("│ QUANTUM INTERFERENCE VERIFICATION:                                                     │");
    if advantage_verified {
        println!("│ [ATTESTATION CONFIRMED] DQI definitively breaks the classical unique decoding radius!   │");
        println!("│ Regimes with t > floor((n-k)/2) successfully recover planted polynomial with >1x ampl. │");
    } else {
        println!("│ [ATTESTATION FAILED] Quantum amplification did not exceed classical threshold.         │");
    }
    println!("└────────────────────────────────────────────────────────────────────────────────────────┘\n");

    let experiment = DqiExperimentReport {
        target_designation: "TARGET-01".to_string(),
        compendium_reference: "arXiv:2610.32595 (Section 3.1)".to_string(),
        field_characteristic: p,
        block_length: n,
        message_dimension: k,
        total_hilbert_dim: hilbert_dim,
        classical_search_space: search_space,
        sweep_results,
        quantum_advantage_verified: advantage_verified,
    };

    let serialized = serde_json::to_string_pretty(&experiment).expect("Serialization failed");

    if record {
        let out_path = PathBuf::from("results/experiments/dqi_opi_advantage.json");
        results::write(&out_path, &serialized).expect("Failed to write results");
        println!("DQI OPI telemetry recorded to: {}", out_path.display());
    } else {
        println!("[bench] Pass --record (or T1000_RECORD=1) to save official results.");
    }
}

