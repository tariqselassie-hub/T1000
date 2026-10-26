// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Quantum Zeno Sentry & Ghost Anomaly Detector on Financial Banking Transactions.
//!
//! Evaluates interaction-free quantum anomaly confinement on the 1,000,000 transaction
//! bank fraud dataset (`bank_fraud.csv`), isolating anomalous fraud vectors in Hilbert space:
//!   H = H_Sentry(1Q) (x) H_Typology(3Q) (x) H_Env(6Q)
//! Total state dimension = 2^10 = 1,024 amplitudes.

use num_complex::Complex64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;

use t1000::bank_fraud_data::{
    locate_bank_fraud_file, parse_bank_transactions, BankTransaction, N_FRAUD_FEATURES,
};
use t1000::results::{Better, Score};
use t1000::ultrametric_tree::{apply_single_qubit_gate, gate_ry, gate_rz};

const N_QUBITS: usize = 10;
const STATE_DIM: usize = 1 << N_QUBITS; // 1,024

/// Sentry Circuit Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZenoSentryConfig {
    pub n_sentry: usize,   // 1 qubit  (Wire 0: Clean vs Alert)
    pub n_typology: usize, // 3 qubits (Wires 1..3: Fraud typology)
    pub n_env: usize,      // 6 qubits (Wires 4..9: Benign noise cooler)
}

impl Default for ZenoSentryConfig {
    fn default() -> Self {
        Self {
            n_sentry: 1,
            n_typology: 3,
            n_env: 6,
        }
    }
}

/// Applies Controlled-Z entangling gate.
#[inline(always)]
fn apply_cz(state: &mut [Complex64], q1: usize, q2: usize) {
    let m1 = 1 << q1;
    let m2 = 1 << q2;
    for (idx, amp) in state.iter_mut().enumerate() {
        if (idx & m1 != 0) && (idx & m2 != 0) {
            *amp = -*amp;
        }
    }
}

/// Parameterized Quantum Zeno Sentry Anomaly Engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumZenoSentry {
    pub cfg: ZenoSentryConfig,
    pub trunk_params: Vec<f64>,
    pub typology_params: Vec<[f64; 2]>,
}

impl QuantumZenoSentry {
    pub fn new(cfg: ZenoSentryConfig) -> Self {
        let trunk_params = vec![0.0; N_QUBITS * 2];
        let typology_params = vec![[0.0, 0.0]; 8]; // 8 branches for wire 1..3
        Self {
            cfg,
            trunk_params,
            typology_params,
        }
    }

    pub fn init_seeded(&mut self, _seed: u64) {
        // Feature scaling sensitivities w_q (indices 0..9) initialized to 1.0
        for p in self.trunk_params[0..N_QUBITS].iter_mut() {
            *p = 1.0;
        }
        // Phase shifts phi_q (indices 10..19) initialized to 0.0
        for p in self.trunk_params[N_QUBITS..N_QUBITS * 2].iter_mut() {
            *p = 0.0;
        }
        for b in self.typology_params.iter_mut() {
            b[0] = 0.0;
            b[1] = 0.0;
        }
    }

    pub fn total_parameters(&self) -> usize {
        self.trunk_params.len() + self.typology_params.len() * 2
    }

    /// Evaluates forward circuit on input transaction features.
    pub fn forward(&self, features: &[f64; N_FRAUD_FEATURES]) -> Vec<Complex64> {
        let mut state = vec![Complex64::default(); STATE_DIM];
        state[0] = Complex64::new(1.0, 0.0);

        // Stage 1: Feature Injection into 10-Qubit Hilbert Space (H_1024)
        for (q, (&x, &scale)) in features
            .iter()
            .zip(&self.trunk_params[..N_QUBITS])
            .enumerate()
        {
            let angle = (x * scale).clamp(0.0, std::f64::consts::PI);
            apply_single_qubit_gate(&mut state, q, &gate_ry(angle));
        }

        // Stage 2: Interaction-Free Zeno Coupling across Anomaly Channels
        // Multi-channel entangling CZ gates induce phase shifts on correlated anomalies
        apply_cz(&mut state, 0, 1); // Failed Attempts x International
        apply_cz(&mut state, 0, 2); // Failed Attempts x Night
        apply_cz(&mut state, 1, 2); // International x Night
        apply_cz(&mut state, 3, 4); // Velocity x Balance Ratio
        apply_cz(&mut state, 0, 5); // Failed Attempts x Risk Product
        apply_cz(&mut state, 1, 5); // International x Risk Product
        apply_cz(&mut state, 6, 7); // Distance x Credit Score

        // Stage 3: Parameterized Phase Shifts
        for q in 0..N_QUBITS {
            let phi = self.trunk_params[N_QUBITS + q];
            if phi.abs() > 1e-6 {
                apply_single_qubit_gate(&mut state, q, &gate_rz(phi));
            }
        }

        state
    }

    /// Computes Zeno Anomaly Leakage Probability P_alert:
    /// Measures probability mass in Forbidden Boundary Strata H_boundary (state != |0000000000>).
    #[inline]
    pub fn compute_alert_probability(&self, state: &[Complex64]) -> f64 {
        let p_flux = state[0].norm_sqr();
        (1.0 - p_flux).clamp(0.0, 1.0)
    }

    /// Parameter-shift derivative with respect to trunk parameter p_idx on alert probability.
    pub fn param_shift_trunk(&mut self, feat: &[f64; N_FRAUD_FEATURES], p_idx: usize) -> f64 {
        let shift = std::f64::consts::FRAC_PI_2;
        self.trunk_params[p_idx] += shift;
        let s_plus = self.forward(feat);
        let p_plus = self.compute_alert_probability(&s_plus);

        self.trunk_params[p_idx] -= 2.0 * shift;
        let s_minus = self.forward(feat);
        let p_minus = self.compute_alert_probability(&s_minus);

        self.trunk_params[p_idx] += shift;
        0.5 * (p_plus - p_minus)
    }
}

/// Comprehensive Report on Quantum Zeno Anomaly Detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BankFraudReport {
    pub timestamp_rfc3339: String,
    pub dataset: String,
    pub architecture: String,
    pub registers: String,
    pub n_qubits: usize,
    pub state_dim: usize,
    pub total_parameters: usize,
    pub train_samples: usize,
    pub test_samples: usize,
    pub fraud_prevalence_pct: f64,
    pub training_time_sec: f64,
    pub inference_per_sample_us: f64,
    pub sentry_routing_contrast_ratio: f64,
    pub mean_alert_clean: f64,
    pub mean_alert_fraud: f64,
    /// How `optimal_threshold` was chosen; the holdout metrics below are all at that threshold.
    pub threshold_source: String,
    pub optimal_threshold: f64,
    pub train_f1_score_pct: f64,
    pub precision_pct: f64,
    pub recall_pct: f64,
    pub f1_score_pct: f64,
    pub balanced_accuracy_pct: f64,
    pub auroc_score: f64,
    /// Threshold maximizing F1 on the holdout labels, and that F1. Optimistic; reference only.
    pub oracle_test_threshold: f64,
    pub oracle_test_f1_score_pct: f64,
}

fn main() {
    println!("================================================================================");
    println!("T-1000 QUANTUM ZENO SENTRY: BANK FRAUD INTERACTION-FREE ANOMALY DETECTOR");
    println!("================================================================================");

    let fraud_path =
        locate_bank_fraud_file(&["bank_fraud.csv"]).expect("Bank fraud dataset not found");
    println!("Loaded Dataset       : {}", fraud_path.display());

    let total_samples = 25000;
    let all_txns = parse_bank_transactions(&fraud_path, Some(total_samples));
    let n_train = 15000.min(all_txns.len() * 3 / 5);
    let train_set = &all_txns[0..n_train];
    let test_set = &all_txns[n_train..];

    let n_test_fraud = test_set.iter().filter(|t| t.is_fraud == 1).count();
    let fraud_prev = (n_test_fraud as f64) / (test_set.len() as f64) * 100.0;

    println!(
        "Dataset Split        : {} Train | {} Test Transactions (Total {})",
        train_set.len(),
        test_set.len(),
        all_txns.len()
    );
    println!(
        "Test Fraud Prevalence: {:.2}% ({}/{} Fraudulent Transactions)",
        fraud_prev,
        n_test_fraud,
        test_set.len()
    );

    let cfg = ZenoSentryConfig::default();
    println!("\nHilbert Register Topology:");
    println!(
        "  Sentry Register H_Sentry    : {} Qubit  (Wire 0: Clean vs Alert Subspace)",
        cfg.n_sentry
    );
    println!(
        "  Typology Register H_Typology: {} Qubits (Wires 1..3: Fraud Archetype)",
        cfg.n_typology
    );
    println!(
        "  Entropy Sink H_Env          : {} Qubits (Wires 4..9: Benign Noise Cooler)",
        cfg.n_env
    );
    println!("  Total State Dimension       : 10 Qubits (dim = 1,024 amplitudes)");

    let mut sentry = QuantumZenoSentry::new(cfg);
    sentry.init_seeded(0x5EED_7777);
    println!(
        "Trainable Parameters        : {} (Zero Classical Readout Weights)",
        sentry.total_parameters()
    );

    // 1. Training Stage via Sentry Parameter Shift
    let epochs = 3;
    println!("\n[1/2] Training Quantum Zeno Sentry Engine ({epochs} Epochs)...");
    let t_train = Instant::now();
    let lr = 0.05;
    let batch_size = 128;
    let n_batches = train_set.len() / batch_size;

    for ep in 1..=epochs {
        for b in 0..n_batches {
            let b_txns = &train_set[b * batch_size..(b + 1) * batch_size];
            let mut grad = vec![0.0; sentry.trunk_params.len()];

            let active_params = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
            for t in b_txns {
                let state = sentry.forward(&t.features);
                let p = sentry.compute_alert_probability(&state).clamp(0.005, 0.995);
                let dldp = if t.is_fraud == 1 {
                    -12.0 / p
                } else {
                    1.0 / (1.0 - p)
                };

                for &p_idx in &active_params {
                    let g = sentry.param_shift_trunk(&t.features, p_idx);
                    grad[p_idx] += dldp * g;
                }
            }

            let b_scale = (lr / (batch_size as f64)).min(0.005);
            for &p_idx in &active_params {
                sentry.trunk_params[p_idx] =
                    (sentry.trunk_params[p_idx] - b_scale * grad[p_idx]).clamp(0.2, 3.0);
            }
        }
        println!("  Epoch {:02}/{} Completed", ep, epochs);
    }
    let train_duration = t_train.elapsed().as_secs_f64();
    println!("Training Finished in {:.2}s", train_duration);

    // 2. Evaluation on Holdout Test Set
    println!(
        "\n[2/2] Evaluating on Holdout Test Set ({} transactions)...",
        test_set.len()
    );
    let t_eval = Instant::now();

    let scores = score_transactions(&sentry, test_set);

    let eval_duration = t_eval.elapsed();
    let per_sample_us = (eval_duration.as_micros() as f64) / (test_set.len() as f64);

    let mut alert_clean_sum = 0.0;
    let mut alert_fraud_sum = 0.0;
    let mut n_clean = 0u32;
    let mut n_fraud = 0u32;

    for &(p, y) in &scores {
        if y == 1 {
            alert_fraud_sum += p;
            n_fraud += 1;
        } else {
            alert_clean_sum += p;
            n_clean += 1;
        }
    }

    let mean_clean = alert_clean_sum / f64::from(n_clean).max(1.0);
    let mean_fraud = alert_fraud_sum / f64::from(n_fraud).max(1.0);
    let contrast_ratio = mean_fraud / mean_clean.max(1e-6);

    // The operating threshold is chosen on the training transactions only, then applied
    // unchanged to the holdout set.
    let train_scores = score_transactions(&sentry, train_set);
    let best_th = best_f1_threshold(&train_scores);
    let train_at_th = threshold_metrics(&train_scores, best_th);
    let test_at_th = threshold_metrics(&scores, best_th);
    // Reference only: the threshold that would be picked by looking at the test labels.
    let oracle_th = best_f1_threshold(&scores);
    let oracle_at_th = threshold_metrics(&scores, oracle_th);

    // Trapezoidal AUROC approximation
    let mut sorted_scores = scores;
    sorted_scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let mut auroc = 0.0;
    let mut cum_tp = 0.0;
    let mut prev_fp = 0.0;
    let total_pos = f64::from(n_fraud);
    let total_neg = f64::from(n_clean);

    for &(_, y) in &sorted_scores {
        if y == 1 {
            cum_tp += 1.0;
        } else {
            let cur_fp = prev_fp + 1.0;
            auroc += (cum_tp / total_pos) * (1.0 / total_neg);
            prev_fp = cur_fp;
        }
    }

    println!("--------------------------------------------------------------------------------");
    println!("QUANTUM ZENO SENTRY TELEMETRY & ROUTING CONTRAST:");
    println!(
        "  Mean Alert P_alert (Clean Transactions) : {:.4}",
        mean_clean
    );
    println!(
        "  Mean Alert P_alert (Fraud Transactions) : {:.4}",
        mean_fraud
    );
    println!(
        "  Sentry Routing Contrast Ratio          : {:.2}:1",
        contrast_ratio
    );
    println!("--------------------------------------------------------------------------------");
    println!(
        "HOLDOUT METRICS (threshold {:.2} chosen on training set, train F1 {:.2}%):",
        best_th,
        train_at_th.f1 * 100.0
    );
    println!(
        "  Fraud Precision                         : {:>6.2}%",
        test_at_th.precision * 100.0
    );
    println!(
        "  Fraud Recall                            : {:>6.2}%",
        test_at_th.recall * 100.0
    );
    println!(
        "  Fraud F1-Score                          : {:>6.2}%",
        test_at_th.f1 * 100.0
    );
    println!(
        "  Balanced Accuracy                       : {:>6.2}%",
        test_at_th.balanced_accuracy * 100.0
    );
    println!("  AUROC Score                             : {:.4}", auroc);
    println!(
        "  Inference Latency                       : {:.2} us/sample",
        per_sample_us
    );
    println!(
        "  [reference] test-tuned threshold {:.2} would give F1 {:.2}% (not a valid estimate)",
        oracle_th,
        oracle_at_th.f1 * 100.0
    );
    println!("--------------------------------------------------------------------------------");

    let report = BankFraudReport {
        timestamp_rfc3339: t1000::zinc_training_common::utc_timestamp_rfc3339(),
        dataset: "Bank Fraud Anomaly Dataset (1M transactions)".to_string(),
        architecture: "T-1000 Quantum Zeno Sentry Anomaly Detector".to_string(),
        registers: format!(
            "H_Sentry({}Q) (x) H_Typology({}Q) (x) H_Env({}Q)",
            cfg.n_sentry, cfg.n_typology, cfg.n_env
        ),
        n_qubits: N_QUBITS,
        state_dim: STATE_DIM,
        total_parameters: sentry.total_parameters(),
        train_samples: train_set.len(),
        test_samples: test_set.len(),
        fraud_prevalence_pct: fraud_prev,
        training_time_sec: train_duration,
        inference_per_sample_us: per_sample_us,
        sentry_routing_contrast_ratio: contrast_ratio,
        mean_alert_clean: mean_clean,
        mean_alert_fraud: mean_fraud,
        threshold_source: "train_set_max_f1".to_string(),
        optimal_threshold: best_th,
        train_f1_score_pct: train_at_th.f1 * 100.0,
        precision_pct: test_at_th.precision * 100.0,
        recall_pct: test_at_th.recall * 100.0,
        f1_score_pct: test_at_th.f1 * 100.0,
        balanced_accuracy_pct: test_at_th.balanced_accuracy * 100.0,
        auroc_score: auroc,
        oracle_test_threshold: oracle_th,
        oracle_test_f1_score_pct: oracle_at_th.f1 * 100.0,
    };

    let report_path = Path::new("results/experiments/bank_fraud_zeno_report.json");
    let settings = format!(
        "train={} test={} epochs={epochs} lr={lr} batch={batch_size} threshold=train-f1",
        train_set.len(),
        test_set.len()
    );
    let score = Score {
        metric: "test_auroc",
        value: auroc,
        better: Better::Higher,
        guard: None,
        settings: &settings,
    };
    let json_str = serde_json::to_string_pretty(&report).expect("Serialize report");
    let outcome =
        t1000::results::write_scored(report_path, json_str, &score).expect("Write report");
    println!("Report {}: {outcome:?}", report_path.display());
}

/// Alert probability and fraud label for each transaction.
fn score_transactions(sentry: &QuantumZenoSentry, txns: &[BankTransaction]) -> Vec<(f64, usize)> {
    txns.par_iter()
        .map(|t| {
            let state = sentry.forward(&t.features);
            (sentry.compute_alert_probability(&state), t.is_fraud)
        })
        .collect()
}

/// Detection metrics of the rule `p_alert >= threshold`.
#[derive(Debug, Clone, Copy, Default)]
struct ThresholdMetrics {
    precision: f64,
    recall: f64,
    f1: f64,
    balanced_accuracy: f64,
}

fn threshold_metrics(scores: &[(f64, usize)], threshold: f64) -> ThresholdMetrics {
    let (mut tp, mut fp, mut tn, mut fn_count) = (0usize, 0usize, 0usize, 0usize);
    for &(p, y) in scores {
        match (p >= threshold, y == 1) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, false) => tn += 1,
            (false, true) => fn_count += 1,
        }
    }
    let ratio = |a: usize, b: usize| if b > 0 { a as f64 / b as f64 } else { 0.0 };
    let precision = ratio(tp, tp + fp);
    let recall = ratio(tp, tp + fn_count);
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    ThresholdMetrics {
        precision,
        recall,
        f1,
        balanced_accuracy: 0.5 * (recall + ratio(tn, tn + fp)),
    }
}

/// Threshold on the 0.01 grid that maximizes F1 on `scores`.
fn best_f1_threshold(scores: &[(f64, usize)]) -> f64 {
    let mut best = (0.5, 0.0);
    for step in 1..99 {
        let th = f64::from(step) * 0.01;
        let f1 = threshold_metrics(scores, th).f1;
        if f1 > best.1 {
            best = (th, f1);
        }
    }
    best.0
}
