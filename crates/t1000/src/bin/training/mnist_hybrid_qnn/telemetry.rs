// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Telemetry Structs & Quantum Interference Contrast Evaluator.

use serde::{Deserialize, Serialize};
use t1000::{CircuitAnsatz, QuantumState};
use t1000_core::interference::net_interference;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochTelemetry {
    pub epoch: usize,
    pub train_loss: f64,
    pub train_accuracy_pct: f64,
    pub test_loss: f64,
    pub test_accuracy_pct: f64,
    pub mean_coherence_l1: f64,
    pub intra_class_interference: f64,
    pub inter_class_interference: f64,
    pub epoch_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MnistHybridQnnReport {
    pub timestamp_rfc3339: String,
    pub architecture: String,
    pub n_qubits: usize,
    pub circuit_depth: usize,
    pub num_quantum_params: usize,
    pub num_classical_head_params: usize,
    pub total_trainable_params: usize,
    pub train_samples: usize,
    pub test_samples: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub baseline_vqc_linear_acc: f64,
    pub final_test_acc_pct: f64,
    pub final_test_loss: f64,
    pub final_intra_interference: f64,
    pub final_inter_interference: f64,
    pub total_train_time_sec: f64,
    pub per_sample_latency_us: f64,
    pub history: Vec<EpochTelemetry>,
}

pub fn compute_l1_coherence(state: &QuantumState) -> f64 {
    let sum_mag: f64 = (0..state.dim)
        .map(|i| (state.real[i] * state.real[i] + state.imag[i] * state.imag[i]).sqrt())
        .sum();
    (sum_mag * sum_mag - 1.0).max(0.0)
}

pub fn evaluate_interference_contrast(
    ansatz: &CircuitAnsatz,
    weights: &[f64],
    samples: &[(QuantumState, [f64; super::features::N_SPATIAL], usize)],
    n_pairs: usize,
) -> (f64, f64, f64) {
    let mut class_states: Vec<Vec<QuantumState>> = vec![Vec::new(); 10];
    for (state, _, label) in samples {
        if class_states[*label].len() < 30 {
            let mut evolved = state.clone();
            ansatz.forward_in_place(&mut evolved, weights);
            class_states[*label].push(evolved);
        }
    }

    let mut intra_scores = Vec::new();
    let mut inter_scores = Vec::new();

    for bucket in class_states.iter().take(10) {
        for i in 0..bucket.len() {
            for j in (i + 1)..bucket.len() {
                if intra_scores.len() >= n_pairs {
                    break;
                }
                let reg_a = bucket[i].to_register(100);
                let reg_b = bucket[j].to_register(101);
                intra_scores.push(net_interference(&reg_a, &reg_b));
            }
        }
    }

    for c1 in 0..10 {
        for c2 in (c1 + 1)..10 {
            let b1 = &class_states[c1];
            let b2 = &class_states[c2];
            for s1 in b1.iter().take(4) {
                for s2 in b2.iter().take(4) {
                    if inter_scores.len() >= n_pairs {
                        break;
                    }
                    let reg_a = s1.to_register(200);
                    let reg_b = s2.to_register(201);
                    inter_scores.push(net_interference(&reg_a, &reg_b));
                }
            }
        }
    }

    let mean_intra = if intra_scores.is_empty() { 0.0 } else { intra_scores.iter().sum::<f64>() / intra_scores.len() as f64 };
    let mean_inter = if inter_scores.is_empty() { 0.0 } else { inter_scores.iter().sum::<f64>() / inter_scores.len() as f64 };
    (mean_intra, mean_inter, mean_intra - mean_inter)
}
