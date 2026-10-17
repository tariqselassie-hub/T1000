// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Waterbirds 11k Quantum Neural Network (QNN) Training & Robustness Benchmark.
//!
//! Evaluates the T-1000 10-Qubit Variational Quantum Circuit on Waterbirds (11,788 images)
//! for Landbird vs Waterbird binary classification under spurious background correlation.
//! Uses native AVX2/AVX-VNNI hardware acceleration and Intel Xe iGPU bridge status.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use t1000::{adjoint_gradient, AdamOptimizer, CircuitAnsatz, PauliObservable, QuantumState};
use t1000::engine::T1000Engine;

const N_QUBITS: usize = 10;
const STATE_DIM: usize = 1 << N_QUBITS; // 1024
const N_CLASSES: usize = 2; // 0 = Landbird, 1 = Waterbird

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupMetrics {
    pub total: usize,
    pub correct: usize,
    pub accuracy_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationSummary {
    pub total_samples: usize,
    pub overall_accuracy_pct: f64,
    pub average_loss: f64,
    pub worst_group_accuracy_pct: f64,
    pub mean_coherence_l1: f64,
    pub g0_land_land: GroupMetrics,
    pub g1_land_water: GroupMetrics,
    pub g2_water_land: GroupMetrics,
    pub g3_water_water: GroupMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochHistoryItem {
    pub epoch: usize,
    pub train_loss: f64,
    pub train_accuracy_pct: f64,
    pub test_loss: f64,
    pub test_accuracy_pct: f64,
    pub worst_group_accuracy_pct: f64,
    pub mean_coherence_l1: f64,
    pub epoch_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BirdsQnnReport {
    pub timestamp_rfc3339: String,
    pub dataset: String,
    pub n_qubits: usize,
    pub state_dim: usize,
    pub circuit_depth: usize,
    pub num_circuit_params: usize,
    pub num_head_params: usize,
    pub train_samples: usize,
    pub test_samples: usize,
    pub total_epochs: usize,
    pub best_test_accuracy_pct: f64,
    pub best_test_epoch: usize,
    pub avx_vnni_accelerated: bool,
    pub igpu_bridge_available: bool,
    pub igpu_device: Option<String>,
    pub zero_shot: EvaluationSummary,
    pub final_evaluation: EvaluationSummary,
    pub training_time_sec: f64,
    pub per_sample_latency_us: f64,
    pub history: Vec<EpochHistoryItem>,
}

/// Encode 1024-dimensional normalized image pixels into a 10-qubit QuantumState.
fn encode_image_state(pixels: &[f32]) -> QuantumState {
    let mut state = QuantumState::zero(STATE_DIM);
    let mut norm_sq = 0.0;
    for &p in pixels {
        norm_sq += f64::from(p) * f64::from(p);
    }
    if norm_sq > 1e-12 {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for (i, &p) in pixels.iter().enumerate() {
            state.real[i] = f64::from(p) * inv_norm;
            state.imag[i] = 0.0;
        }
    } else {
        let uniform = 1.0 / (STATE_DIM as f64).sqrt();
        for i in 0..STATE_DIM {
            state.real[i] = uniform;
            state.imag[i] = 0.0;
        }
    }
    state
}

/// Linear Head mapping expectation observables to 2 class logits.
#[derive(Debug, Clone)]
pub struct BinaryHead {
    pub num_obs: usize,
    pub weights: Vec<f64>,
    pub biases: Vec<f64>,
}

impl BinaryHead {
    fn new(num_obs: usize, seed: u64) -> Self {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let scale = (2.0 / num_obs as f64).sqrt();
        let weights = (0..N_CLASSES * num_obs)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();
        let biases = vec![0.0; N_CLASSES];
        Self { num_obs, weights, biases }
    }

    fn forward(&self, exp: &[f64]) -> [f64; N_CLASSES] {
        let mut logits = [0.0; N_CLASSES];
        for (c, logit) in logits.iter_mut().enumerate() {
            let mut sum = self.biases[c];
            let row = c * self.num_obs;
            for (q, &val) in exp.iter().enumerate().take(self.num_obs) {
                sum += self.weights[row + q] * val;
            }
            *logit = sum;
        }
        logits
    }

    fn param_count(&self) -> usize {
        N_CLASSES * self.num_obs + N_CLASSES
    }

    fn pack_params(&self) -> Vec<f64> {
        let mut p = Vec::with_capacity(self.param_count());
        p.extend_from_slice(&self.weights);
        p.extend_from_slice(&self.biases);
        p
    }

    fn unpack_params(&mut self, p: &[f64]) {
        let n_w = N_CLASSES * self.num_obs;
        self.weights.copy_from_slice(&p[..n_w]);
        self.biases.copy_from_slice(&p[n_w..n_w + N_CLASSES]);
    }
}

/// Softmax cross-entropy loss and logit gradients for binary classification.
fn softmax_cross_entropy(logits: &[f64; N_CLASSES], label: usize) -> (f64, [f64; N_CLASSES], usize) {
    let max_l = logits[0].max(logits[1]);
    let e0 = (logits[0] - max_l).exp();
    let e1 = (logits[1] - max_l).exp();
    let sum = (e0 + e1).max(1e-12);
    let p0 = e0 / sum;
    let p1 = e1 / sum;

    let pred = if p1 > p0 { 1 } else { 0 };
    let loss = if label == 0 { -p0.max(1e-12).ln() } else { -p1.max(1e-12).ln() };

    let mut d_logits = [0.0; N_CLASSES];
    d_logits[0] = p0 - if label == 0 { 1.0 } else { 0.0 };
    d_logits[1] = p1 - if label == 1 { 1.0 } else { 0.0 };
    (loss, d_logits, pred)
}

/// Computes l1-norm coherence on quantum state.
fn compute_l1_coherence(state: &QuantumState) -> f64 {
    let sum_mag: f64 = (0..state.dim)
        .map(|i| (state.real[i] * state.real[i] + state.imag[i] * state.imag[i]).sqrt())
        .sum();
    (sum_mag * sum_mag - 1.0).max(0.0)
}

/// Evaluates full test dataset across spurious correlation groups.
fn evaluate_split(
    ansatz: &CircuitAnsatz,
    weights: &[f64],
    head: &BinaryHead,
    samples: &[(QuantumState, usize, usize)],
    observables: &[PauliObservable],
) -> EvaluationSummary {
    let results: Vec<(f64, usize, usize, usize, f64)> = samples
        .par_iter()
        .map(|(input_state, label, place)| {
            let mut evolved = input_state.clone();
            ansatz.forward_in_place(&mut evolved, weights);
            let exps: Vec<f64> = observables.iter().map(|obs| obs.expectation_value(&evolved)).collect();
            let logits = head.forward(&exps);
            let (loss, _, pred) = softmax_cross_entropy(&logits, *label);
            let coh = compute_l1_coherence(&evolved);
            (loss, *label, *place, if pred == *label { 1 } else { 0 }, coh)
        })
        .collect();

    let mut total_loss = 0.0;
    let mut total_correct = 0;
    let mut total_coh = 0.0;
    let mut g_tot = [0usize; 4];
    let mut g_cor = [0usize; 4];

    for (loss, label, place, correct, coh) in results {
        total_loss += loss;
        total_correct += correct;
        total_coh += coh;
        let g_idx = match (label, place) {
            (0, 0) => 0, // Land on Land
            (0, 1) => 1, // Land on Water
            (1, 0) => 2, // Water on Land
            _ => 3,      // Water on Water
        };
        g_tot[g_idx] += 1;
        g_cor[g_idx] += correct;
    }

    let n = samples.len() as f64;
    let mk_grp = |idx: usize| GroupMetrics {
        total: g_tot[idx],
        correct: g_cor[idx],
        accuracy_pct: if g_tot[idx] > 0 { (g_cor[idx] as f64 / g_tot[idx] as f64) * 100.0 } else { 0.0 },
    };

    let g0 = mk_grp(0);
    let g1 = mk_grp(1);
    let g2 = mk_grp(2);
    let g3 = mk_grp(3);
    let wga = g0.accuracy_pct.min(g1.accuracy_pct).min(g2.accuracy_pct).min(g3.accuracy_pct);

    EvaluationSummary {
        total_samples: samples.len(),
        overall_accuracy_pct: (total_correct as f64 / n) * 100.0,
        average_loss: total_loss / n,
        worst_group_accuracy_pct: wga,
        mean_coherence_l1: total_coh / n,
        g0_land_land: g0,
        g1_land_water: g1,
        g2_water_land: g2,
        g3_water_water: g3,
    }
}

/// Helper to read flat binary arrays into memory.
fn load_binary_split(dir: &Path, split: &str, count: usize) -> Vec<(QuantumState, usize, usize)> {
    let img_path = dir.join(format!("{}_images.bin", split));
    let lbl_path = dir.join(format!("{}_labels.bin", split));
    let plc_path = dir.join(format!("{}_places.bin", split));

    let img_bytes = fs::read(&img_path).expect("Failed reading images bin");
    let lbl_bytes = fs::read(&lbl_path).expect("Failed reading labels bin");
    let plc_bytes = fs::read(&plc_path).expect("Failed reading places bin");

    let img_floats: &[f32] = unsafe {
        std::slice::from_raw_parts(img_bytes.as_ptr() as *const f32, count * STATE_DIM)
    };

    (0..count)
        .into_par_iter()
        .map(|i| {
            let px = &img_floats[i * STATE_DIM..(i + 1) * STATE_DIM];
            let st = encode_image_state(px);
            let lbl = lbl_bytes[i] as usize;
            let plc = plc_bytes[i] as usize;
            (st, lbl, plc)
        })
        .collect()
}

fn main() {
    println!("┌────────────────────────────────────────────────────────────────────────┐");
    println!("│      T-1000 QUANTUM NEURAL NETWORK: WATERBIRDS 11K ROBUSTNESS BENCH    │");
    println!("│      10-Qubit Variational Circuit on Arrow Lake AVX2/VNNI + Xe iGPU    │");
    println!("└────────────────────────────────────────────────────────────────────────┘");

    let engine = T1000Engine::new();
    let manifest = engine.system_manifest();

    println!("Hardware Acceleration Discovery:");
    println!("  AVX-VNNI Accelerated : {}", manifest.avx_vnni_accelerated);
    println!("  Intel Xe iGPU Bridge : {}", manifest.igpu_bridge_available);
    println!("  Mounted GPU Device   : {}", manifest.igpu_device_name.as_deref().unwrap_or("None"));

    let data_dir = PathBuf::from("data/birds");
    println!("\nLoading Waterbirds binary datasets from {}...", data_dir.display());

    let t0_load = Instant::now();
    let train_samples = load_binary_split(&data_dir, "train", 4795);
    let test_samples = load_binary_split(&data_dir, "test", 5794);
    println!("Loaded 4,795 train and 5,794 test samples in {:.2} ms", t0_load.elapsed().as_secs_f64() * 1000.0);

    // Build 10-qubit circuit ansatz with depth 3
    let depth = 3;
    let ansatz = CircuitAnsatz::new(N_QUBITS, depth).with_pi_layer(true);
    let mut observables: Vec<PauliObservable> = (0..N_QUBITS).map(PauliObservable::Z).collect();
    for i in 0..N_QUBITS {
        for j in (i + 1)..N_QUBITS {
            observables.push(PauliObservable::ZZ(i, j));
        }
    }
    let n_obs = observables.len(); // 10 Z + 45 ZZ = 55

    let n_ansatz_params = ansatz.param_count(); // 2 * 10 * 3 = 60
    let mut ansatz_weights = ansatz.weights.clone();
    let mut head = BinaryHead::new(n_obs, 42);
    let n_head_params = head.param_count(); // 2 * 55 + 2 = 112
    let mut head_weights = head.pack_params();

    println!("\nModel Architecture:");
    println!("  Hilbert State Space  : 10 Qubits (2^10 = 1,024 amplitudes)");
    println!("  Parameterized Layers : {} layers HEA (Ring CZ + Ry/Rz)", depth);
    println!("  Expectation Correlators: {} (10 single Z + 45 two-body ZZ)", n_obs);
    println!("  Trainable Parameters : {} ({} VQC + {} Readout Head)", n_ansatz_params + n_head_params, n_ansatz_params, n_head_params);

    // Initial Zero-Shot Test Evaluation
    println!("\nEvaluating Zero-Shot Baseline on Test Set (5,794 samples)...");
    let t0_eval = Instant::now();
    let zero_shot = evaluate_split(&ansatz, &ansatz_weights, &head, &test_samples, &observables);
    println!("Zero-Shot Test Completed in {:.2} s | Acc: {:.2}% | WGA: {:.2}% | L1-Coh: {:.2}",
        t0_eval.elapsed().as_secs_f64(), zero_shot.overall_accuracy_pct, zero_shot.worst_group_accuracy_pct, zero_shot.mean_coherence_l1);

    // Multi-Epoch Training Run (Default: 10 epochs or CLI argument)
    let epochs: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(10);
    println!("\nInitiating {}-Epoch Training across 4,795 training images (Batch Size = 64)...", epochs);
    let batch_size = 64;
    let n_train = train_samples.len();
    let num_batches = n_train.div_ceil(batch_size);

    let mut opt_ansatz = AdamOptimizer::new(n_ansatz_params, 0.035);
    let mut opt_head = AdamOptimizer::new_unwrapped(n_head_params, 0.030);

    let mut history = Vec::with_capacity(epochs);
    let mut best_test_acc = zero_shot.overall_accuracy_pct;
    let mut best_epoch = 0;

    println!("\n┌──────┬────────────┬────────────┬───────────┬───────────┬───────────┬──────────────┬────────────┐");
    println!("│ Ep   │ Trn Loss   │ Trn Acc %  │ Tst Loss  │ Tst Acc % │ WGA %     │ L1-Coherence │ Time (ms)  │");
    println!("├──────┼────────────┼────────────┼───────────┼───────────┼───────────┼──────────────┼────────────┤");

    let t_train_all_start = Instant::now();

    for epoch in 1..=epochs {
        let t_ep_start = Instant::now();
        let mut total_train_loss = 0.0;
        let mut train_correct = 0;

        for b in 0..num_batches {
            let start = b * batch_size;
            let end = (start + batch_size).min(n_train);
            let batch = &train_samples[start..end];
            let batch_len = batch.len() as f64;

            let (b_loss, b_cor, b_ans_grad, b_hd_grad) = batch
                .par_iter()
                .map(|(input_state, label, _)| {
                    let mut s_loss = 0.0;
                    let mut s_cor = 0;
                    let mut d_head = vec![0.0; n_head_params];

                    let adj = adjoint_gradient(&ansatz, input_state, &ansatz_weights, &observables, |exps| {
                        let logits = head.forward(exps);
                        let (loss, d_logits, pred) = softmax_cross_entropy(&logits, *label);
                        s_loss = loss;
                        s_cor = if pred == *label { 1 } else { 0 };

                        for c in 0..N_CLASSES {
                            let row = c * n_obs;
                            for q in 0..n_obs {
                                d_head[row + q] = d_logits[c] * exps[q];
                            }
                            d_head[N_CLASSES * n_obs + c] = d_logits[c];
                        }

                        let mut d_exps = vec![0.0; n_obs];
                        for (q, d_exp) in d_exps.iter_mut().enumerate() {
                            let mut sum = 0.0;
                            for (c, &d_logit) in d_logits.iter().enumerate() {
                                sum += d_logit * head.weights[c * n_obs + q];
                            }
                            *d_exp = sum;
                        }
                        d_exps
                    });

                    (s_loss, s_cor, adj.gradient, d_head)
                })
                .reduce(
                    || (0.0, 0, vec![0.0; n_ansatz_params], vec![0.0; n_head_params]),
                    |mut acc, item| {
                        acc.0 += item.0;
                        acc.1 += item.1;
                        for i in 0..n_ansatz_params { acc.2[i] += item.2[i]; }
                        for i in 0..n_head_params { acc.3[i] += item.3[i]; }
                        acc
                    },
                );

            total_train_loss += b_loss;
            train_correct += b_cor;

            let avg_ans_grad: Vec<f64> = b_ans_grad.into_iter().map(|g| g / batch_len).collect();
            let avg_hd_grad: Vec<f64> = b_hd_grad.into_iter().map(|g| g / batch_len).collect();

            opt_ansatz.step(&mut ansatz_weights, &avg_ans_grad);
            opt_head.step(&mut head_weights, &avg_hd_grad);
            head.unpack_params(&head_weights);
        }

        let ep_elapsed_ms = t_ep_start.elapsed().as_secs_f64() * 1000.0;
        let trn_acc = (f64::from(train_correct) / n_train as f64) * 100.0;
        let trn_loss = total_train_loss / n_train as f64;

        let test_eval = evaluate_split(&ansatz, &ansatz_weights, &head, &test_samples, &observables);
        if test_eval.overall_accuracy_pct > best_test_acc {
            best_test_acc = test_eval.overall_accuracy_pct;
            best_epoch = epoch;
        }

        println!("│ {:<4} │ {:>10.4} │ {:>9.2}% │ {:>9.4} │ {:>8.2}% │ {:>8.2}% │ {:>12.2} │ {:>10.1} │",
            epoch, trn_loss, trn_acc, test_eval.average_loss, test_eval.overall_accuracy_pct, test_eval.worst_group_accuracy_pct, test_eval.mean_coherence_l1, ep_elapsed_ms);

        history.push(EpochHistoryItem {
            epoch,
            train_loss: trn_loss,
            train_accuracy_pct: trn_acc,
            test_loss: test_eval.average_loss,
            test_accuracy_pct: test_eval.overall_accuracy_pct,
            worst_group_accuracy_pct: test_eval.worst_group_accuracy_pct,
            mean_coherence_l1: test_eval.mean_coherence_l1,
            epoch_time_ms: ep_elapsed_ms,
        });
    }

    println!("└──────┴────────────┴────────────┴───────────┴───────────┴───────────┴──────────────┴────────────┘");

    let total_train_time_sec = t_train_all_start.elapsed().as_secs_f64();
    let per_sample_us = (total_train_time_sec / (n_train * epochs) as f64) * 1_000_000.0;

    println!("\nTraining Completed across {} Epochs in {:.2} s ({:.2} us/sample)", epochs, total_train_time_sec, per_sample_us);
    println!("Best Test Accuracy: {:.2}% achieved at Epoch {}", best_test_acc, best_epoch);

    // Final Post-Training Test Evaluation
    let final_evaluation = evaluate_split(&ansatz, &ansatz_weights, &head, &test_samples, &observables);

    // Display Comparison Matrix
    println!("\n┌────────────────────────┬──────────────────┬──────────────────┐");
    println!("│ Metric                 │ Zero-Shot Baseline│ Trained Final    │");
    println!("├────────────────────────┼──────────────────┼──────────────────┤");
    println!("│ Overall Test Accuracy  │ {:>15.2}% │ {:>15.2}% │", zero_shot.overall_accuracy_pct, final_evaluation.overall_accuracy_pct);
    println!("│ Average Test Loss      │ {:>16.4} │ {:>16.4} │", zero_shot.average_loss, final_evaluation.average_loss);
    println!("│ Worst-Group Accuracy   │ {:>15.2}% │ {:>15.2}% │", zero_shot.worst_group_accuracy_pct, final_evaluation.worst_group_accuracy_pct);
    println!("│ G0 (Land on Land)      │ {:>15.2}% │ {:>15.2}% │", zero_shot.g0_land_land.accuracy_pct, final_evaluation.g0_land_land.accuracy_pct);
    println!("│ G1 (Land on Water)     │ {:>15.2}% │ {:>15.2}% │", zero_shot.g1_land_water.accuracy_pct, final_evaluation.g1_land_water.accuracy_pct);
    println!("│ G2 (Water on Land)     │ {:>15.2}% │ {:>15.2}% │", zero_shot.g2_water_land.accuracy_pct, final_evaluation.g2_water_land.accuracy_pct);
    println!("│ G3 (Water on Water)    │ {:>15.2}% │ {:>15.2}% │", zero_shot.g3_water_water.accuracy_pct, final_evaluation.g3_water_water.accuracy_pct);
    println!("│ Mean L1 Coherence      │ {:>16.2} │ {:>16.2} │", zero_shot.mean_coherence_l1, final_evaluation.mean_coherence_l1);
    println!("└────────────────────────┴──────────────────┴──────────────────┘");

    // Persist Report to disk
    let report = BirdsQnnReport {
        timestamp_rfc3339: chrono_timestamp(),
        dataset: "waterbird_complete95_forest2water2".to_string(),
        n_qubits: N_QUBITS,
        state_dim: STATE_DIM,
        circuit_depth: depth,
        num_circuit_params: n_ansatz_params,
        num_head_params: n_head_params,
        train_samples: n_train,
        test_samples: test_samples.len(),
        total_epochs: epochs,
        best_test_accuracy_pct: best_test_acc,
        best_test_epoch: best_epoch,
        avx_vnni_accelerated: manifest.avx_vnni_accelerated,
        igpu_bridge_available: manifest.igpu_bridge_available,
        igpu_device: manifest.igpu_device_name,
        zero_shot,
        final_evaluation,
        training_time_sec: total_train_time_sec,
        per_sample_latency_us: per_sample_us,
        history,
    };

    let report_path = PathBuf::from("data/birds/birds_qnn_report.json");
    if let Ok(json_str) = serde_json::to_string_pretty(&report) {
        if let Ok(mut f) = File::create(&report_path) {
            let _ = f.write_all(json_str.as_bytes());
            println!("\nReport successfully persisted to: {}", report_path.display());
        }
    }
}

fn chrono_timestamp() -> String {
    let now = std::time::SystemTime::now();
    let duration = now.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    format!("{}.{:03}Z", duration.as_secs(), duration.subsec_millis())
}
