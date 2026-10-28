// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

use std::time::Instant;
use t1000::{CircuitAnsatz, QuantumState};

/// Standardized Cross-Entropy Benchmarking (XEB) between ideal and noisy statevectors:
/// F_XEB = 2^N * sum_x (P_ideal(x) * P_noisy(x)) - 1.0
pub fn compute_linear_xeb(p_ideal: &[f64], p_eval: &[f64]) -> f64 {
    let dim = p_ideal.len();
    assert_eq!(dim, p_eval.len());
    let mut cross_sum = 0.0;
    for i in 0..dim {
        cross_sum += p_ideal[i] * p_eval[i];
    }
    (dim as f64) * cross_sum - 1.0
}

fn main() {
    println!("=========================================================================");
    println!("      STANDARDIZED XEB BENCHMARK: CROSS-ENTROPY & DEPTH SCRAMBLING       ");
    println!("=========================================================================");
    println!(" Formula: F_XEB = 2^N * sum(P_ideal * P_eval) - 1.0                      ");
    println!("-------------------------------------------------------------------------");

    // PART 1: DEPTH SCRAMBLING STUDY (N = 12 and N = 16)
    // Demonstrates how increasing depth drives state space toward Haar chaos (F -> 1.0000)
    println!("\n[PART 1] DEPTH SCRAMBLING TRAJECTORY (Convergence toward Haar Limit 1.0000)");
    println!("┌────┬───────┬──────────────┬──────────────┬───────────────────────────────┐");
    println!("│ N  │ Depth │ States (2^N) │ Latency (ms) │ F_self (Purity Speckle)       │");
    println!("├────┼───────┼──────────────┼──────────────┼───────────────────────────────┤");

    let depth_sweep = vec![2, 4, 8, 12, 16, 20, 24, 32];
    for &n in &[12, 16] {
        let num_states = 1 << n;
        let inv_sqrt = 1.0 / (num_states as f64).sqrt();

        for &d in &depth_sweep {
            let ansatz = CircuitAnsatz::new(n, d).with_pi_layer(true);
            let mut state_in = QuantumState::new(num_states);
            for r in state_in.real.iter_mut() {
                *r = inv_sqrt;
            }
            let mut out = QuantumState::new(num_states);

            let t0 = Instant::now();
            ansatz.forward_into(&state_in, &ansatz.weights, &mut out);
            let lat_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let probs: Vec<f64> = out
                .real
                .iter()
                .zip(out.imag.iter())
                .map(|(&r, &i)| r * r + i * i)
                .collect();

            // Self-XEB
            let f_self = compute_linear_xeb(&probs, &probs);
            println!(
                "│ {:2} │ {:5} │ {:12} │ {:12.2} │ {:29.4} │",
                n, d, num_states, lat_ms, f_self
            );
        }
        if n == 12 {
            println!(
                "├────┼───────┼──────────────┼──────────────┼───────────────────────────────┤"
            );
        }
    }
    println!("└────┴───────┴──────────────┴──────────────┴───────────────────────────────┘");

    // PART 2: DUAL STATEVECTOR CROSS-ENTROPY VS NOISE MODEL
    // Demonstrates exact cross-entropy F_XEB(P_ideal, P_noisy) across depolarizing noise levels
    println!("\n[PART 2] DUAL STATEVECTOR CROSS-ENTROPY BENCHMARK (N = 16 Qubits, D = 16 Layers)");
    println!("┌───────────────────────┬──────────────────────────┬────────────────────────┐");
    println!("│ Noise Condition       │ Channel Parameter        │ F_XEB Fidelity         │");
    println!("├───────────────────────┼──────────────────────────┼────────────────────────┤");

    let n = 16;
    let d = 16;
    let num_states = 1 << n;
    let inv_sqrt = 1.0 / (num_states as f64).sqrt();

    let ansatz_ideal = CircuitAnsatz::new(n, d).with_pi_layer(true);
    let mut state_in = QuantumState::new(num_states);
    for r in state_in.real.iter_mut() {
        *r = inv_sqrt;
    }
    let mut out_ideal = QuantumState::new(num_states);
    ansatz_ideal.forward_into(&state_in, &ansatz_ideal.weights, &mut out_ideal);
    let p_ideal: Vec<f64> = out_ideal
        .real
        .iter()
        .zip(out_ideal.imag.iter())
        .map(|(&r, &i)| r * r + i * i)
        .collect();

    let noise_levels = vec![0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.95, 1.00];
    let p_uniform = 1.0 / (num_states as f64);

    for &eta in &noise_levels {
        // Depolarizing channel: P_noisy = (1 - eta) * P_ideal + eta * Uniform
        let p_noisy: Vec<f64> = p_ideal
            .iter()
            .map(|&p| (1.0 - eta) * p + eta * p_uniform)
            .collect();

        let f_xeb = compute_linear_xeb(&p_ideal, &p_noisy);
        let condition = if eta == 0.0 {
            "Ideal Noiseless State"
        } else if eta == 1.0 {
            "Total Decoherence (Noise)"
        } else if eta <= 0.10 {
            "High-Fidelity NISQ"
        } else if eta <= 0.50 {
            "Moderate Gate Error"
        } else {
            "Severe Decoherence"
        };

        println!(
            "│ {:21} │ Depolarizing eta = {:4.2} │ F_XEB = {:14.4} │",
            condition, eta, f_xeb
        );
    }
    println!("└───────────────────────┴──────────────────────────┴────────────────────────┘");
}
