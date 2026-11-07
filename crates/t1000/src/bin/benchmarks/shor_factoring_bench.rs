// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Reversible Quantum Order Finding & Shor Factorization Benchmark
//!
//! Evaluates Target 02 from *The Compendium of Super-Quadratic Computational Quantum Advantages*
//! (arXiv:2610.32595). Demonstrates end-to-end quantum factorization of composite semi-primes
//! using reversible modular multipliers, Quantum Phase Estimation (QPE), continued fraction
//! convergents, and exact Euclidean GCD extraction.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use t1000::results;
use t1000::shor::{ShorFactorizationReport, ShorFactorizer};

/// Shor Factorization Benchmark Suite Telemetry Report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShorBenchmarkSuiteReport {
    pub target_designation: String,
    pub compendium_reference: String,
    pub semi_primes_tested: Vec<u64>,
    pub factorizations: Vec<ShorFactorizationReport>,
    pub all_verified: bool,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let record = args.iter().any(|a| a == "--record" || a == "-r")
        || std::env::var("T1000_RECORD").map(|v| v == "1").unwrap_or(false);

    println!("[1/3] Initializing Quantum Phase Estimation & Shor Factorization Engine...");
    let semi_primes = vec![15u64, 21u64, 33u64, 35u64, 39u64];

    println!("[2/3] Executing Quantum Order Finding and Factorization across composite suite...");

    let mut reports = Vec::new();
    let mut all_verified = true;

    for &modulus in &semi_primes {
        if let Some(rep) = ShorFactorizer::factor(modulus) {
            if !rep.factorization_verified {
                all_verified = false;
            }
            reports.push(rep);
        } else {
            all_verified = false;
        }
    }

    println!("[3/3] Compiling Factorization Telemetry Matrix...\n");

    println!("┌────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("║       TARGET-02: REVERSIBLE QUANTUM ORDER FINDING & SHOR FACTORIZATION ENGINE          ║");
    println!("║       Compendium Exponential Advantage Showdown (Section 2.1, arXiv:2610.32595)        ║");
    println!("├────────────────────────────────────────────────────────────────────────────────────────┤");
    println!("│ Algorithm Type:                 Quantum Phase Estimation + Continued Fractions + GCD   │");
    println!("│ Classical Lower Bound:          General Number Field Sieve (GNFS): exp(O((log N)^1/3)) │");
    println!("│ Quantum Complexity:             Polynomial O((log N)^2 (log log N))                    │");
    println!("│ Register Architecture:          Counting Register (QPE) + In-Place Work Register       │");
    println!("├────────────────────────────────────────────────────────────────────────────────────────┤");
    println!("│ EXPERIMENTAL QUANTUM FACTORIZATION SUITE                                               │");
    println!("├───────┬─────────────────┬──────────┬──────────┬──────────┬──────────────┬──────────────┤");
    println!("│ N     │ Factors p x q   │ Base a   │ Order r  │ Qubits   │ Circuit Time │ Verification │");
    println!("├───────┼─────────────────┼──────────┼──────────┼──────────┼──────────────┼──────────────┤");

    for r in &reports {
        let factors_str = format!("{} x {}", r.factor_p, r.factor_q);
        let verif_str = if r.factorization_verified {
            "CONFIRMED"
        } else {
            "FAILED"
        };

        println!(
            "│ {:<5} │ {:<15} │ a = {:<4} │ r = {:<4} │ {:>2} Q     │ {:>6} us   │ {:<12} │",
            r.composite_n,
            factors_str,
            r.base_a,
            r.order_r,
            r.total_qubits,
            r.quantum_circuit_micros,
            verif_str
        );
    }

    println!("├───────┴─────────────────┴──────────┴──────────┴──────────┴──────────────┴──────────────┤");
    println!("│ QUANTUM FACTORIZATION ATTESTATION:                                                     │");
    if all_verified {
        println!("│ [ATTESTATION CONFIRMED] All semi-primes deterministically factored with zero errors!   │");
        println!("│ Exact non-trivial factors recovered via continued fraction convergents & Euclid GCD.   │");
    } else {
        println!("│ [ATTESTATION FAILED] One or more semi-primes failed factorization.                    │");
    }
    println!("└────────────────────────────────────────────────────────────────────────────────────────┘\n");

    let suite = ShorBenchmarkSuiteReport {
        target_designation: "TARGET-02".to_string(),
        compendium_reference: "arXiv:2610.32595 (Section 2.1)".to_string(),
        semi_primes_tested: semi_primes,
        factorizations: reports,
        all_verified,
    };

    let serialized = serde_json::to_string_pretty(&suite).expect("Serialization failed");

    if record {
        let out_path = PathBuf::from("results/experiments/shor_factoring_advantage.json");
        results::write(&out_path, &serialized).expect("Failed to write results");
        println!("Shor factorization telemetry recorded to: {}", out_path.display());
    } else {
        println!("[bench] Pass --record (or T1000_RECORD=1) to save official results.");
    }
}
