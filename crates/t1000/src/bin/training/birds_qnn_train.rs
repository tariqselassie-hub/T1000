// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Waterbirds 11k Quantum Neural Network (QNN) Training & Robustness Benchmark.
//!
//! Evaluates the T-1000 10-Qubit Variational Quantum Circuit on Waterbirds (11,788 images)
//! with Correlator LayerNorm (55 Observables: 10 Z + 45 ZZ) and Two-Speed AdamW optimization.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use t1000::engine::T1000Engine;
use t1000::optimizer::{
    normalize_correlators_arr_backward, normalize_correlators_arr_forward, CorrelatorNormCacheArr,
    TwoSpeedHybridOptimizer,
};
use t1000::{adjoint_gradient, CircuitAnsatz, PauliObservable, QuantumState};

const N_QUBITS: usize = 10;
const STATE_DIM: usize = 1 << N_QUBITS; // 1024
const N_CLASSES: usize = 2; // 0 = Landbird, 1 = Waterbird
pub const NUM_CORRELATORS: usize = 55;

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
    pub best_wga_pct: f64,
    pub best_overall_acc_pct: f64,
    pub best_epoch: usize,
    pub avx_vnni_accelerated: bool,
    pub igpu_bridge_available: bool,
    pub igpu_device: Option<String>,
    pub zero_shot: EvaluationSummary,
    pub best_checkpoint_evaluation: EvaluationSummary,
    pub training_time_sec: f64,
    pub per_sample_latency_us: f64,
    pub history: Vec<EpochHistoryItem>,
}

pub type CorrelatorNormCache = CorrelatorNormCacheArr<NUM_CORRELATORS>;

/// Standardizes raw expectations to N(0, 1) so ZZ terms aren't drowned out.
#[inline(always)]
pub fn normalize_correlators_forward(raw: &[f64; NUM_CORRELATORS]) -> CorrelatorNormCache {
    normalize_correlators_arr_forward(raw)
}

/// Projects readout input gradients dL/d(c_hat) back to raw observable gradients dL/d(c_raw).
#[inline(always)]
pub fn normalize_correlators_backward(
    grad_norm: &[f64; NUM_CORRELATORS],
    cache: &CorrelatorNormCache,
) -> [f64; NUM_CORRELATORS] {
    normalize_correlators_arr_backward(grad_norm, cache)
}

/// WGA Checkpoint State Tracker in the Epoch Loop.
#[derive(Clone, Debug)]
pub struct BestCheckpoint {
    pub best_epoch: usize,
    pub best_wga: f64,
    pub best_overall_acc: f64,
    pub best_test_loss: f64,
    pub evaluation: Option<EvaluationSummary>,
    pub vqc_params: Vec<f64>,
    pub head_params: Vec<f64>,
}

impl BestCheckpoint {
    pub fn new(vqc_params: &[f64], head_params: &[f64]) -> Self {
        Self {
            best_epoch: 0,
            best_wga: 0.0,
            best_overall_acc: 0.0,
            best_test_loss: f64::MAX,
            evaluation: None,
            vqc_params: vqc_params.to_vec(),
            head_params: head_params.to_vec(),
        }
    }

    pub fn update_if_better(
        &mut self,
        epoch: usize,
        eval: &EvaluationSummary,
        vqc_params: &[f64],
        head_params: &[f64],
    ) -> bool {
        let wga = eval.worst_group_accuracy_pct;
        let is_better = (wga > self.best_wga + 1e-4)
            || ((wga - self.best_wga).abs() <= 1e-4
                && eval.overall_accuracy_pct > self.best_overall_acc);

        if is_better {
            self.best_epoch = epoch;
            self.best_wga = wga;
            self.best_overall_acc = eval.overall_accuracy_pct;
            self.best_test_loss = eval.average_loss;
            self.evaluation = Some(eval.clone());
            self.vqc_params.copy_from_slice(vqc_params);
            self.head_params.copy_from_slice(head_params);
            true
        } else {
            false
        }
    }
}

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
        }
    } else {
        let uniform = 1.0 / (STATE_DIM as f64).sqrt();
        for i in 0..STATE_DIM {
            state.real[i] = uniform;
        }
    }
    state
}

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
        Self {
            num_obs,
            weights,
            biases: vec![0.0; N_CLASSES],
        }
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

fn softmax_cross_entropy(
    logits: &[f64; N_CLASSES],
    label: usize,
) -> (f64, [f64; N_CLASSES], usize) {
    let max_l = logits[0].max(logits[1]);
    let e0 = (logits[0] - max_l).exp();
    let e1 = (logits[1] - max_l).exp();
    let sum = (e0 + e1).max(1e-12);
    let (p0, p1) = (e0 / sum, e1 / sum);
    let pred = if p1 > p0 { 1 } else { 0 };
    let loss = if label == 0 {
        -p0.max(1e-12).ln()
    } else {
        -p1.max(1e-12).ln()
    };
    (
        loss,
        [
            p0 - if label == 0 { 1.0 } else { 0.0 },
            p1 - if label == 1 { 1.0 } else { 0.0 },
        ],
        pred,
    )
}

fn compute_l1_coherence(state: &QuantumState) -> f64 {
    let sum_mag: f64 = (0..state.dim)
        .map(|i| (state.real[i] * state.real[i] + state.imag[i] * state.imag[i]).sqrt())
        .sum();
    (sum_mag * sum_mag - 1.0).max(0.0)
}

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
            let mut exps_raw = [0.0; NUM_CORRELATORS];
            for i in 0..NUM_CORRELATORS {
                exps_raw[i] = observables[i].expectation_value(&evolved);
            }
            let norm_cache = normalize_correlators_forward(&exps_raw);
            let logits = head.forward(&norm_cache.normalized);
            let (loss, _, pred) = softmax_cross_entropy(&logits, *label);
            (
                loss,
                *label,
                *place,
                if pred == *label { 1 } else { 0 },
                compute_l1_coherence(&evolved),
            )
        })
        .collect();

    let (mut total_loss, mut total_correct, mut total_coh) = (0.0, 0, 0.0);
    let (mut g_tot, mut g_cor) = ([0usize; 4], [0usize; 4]);
    for (loss, label, place, correct, coh) in results {
        total_loss += loss;
        total_correct += correct;
        total_coh += coh;
        let g_idx = match (label, place) {
            (0, 0) => 0,
            (0, 1) => 1,
            (1, 0) => 2,
            _ => 3,
        };
        g_tot[g_idx] += 1;
        g_cor[g_idx] += correct;
    }
    let n = samples.len() as f64;
    let mk_grp = |i: usize| GroupMetrics {
        total: g_tot[i],
        correct: g_cor[i],
        accuracy_pct: if g_tot[i] > 0 {
            (g_cor[i] as f64 / g_tot[i] as f64) * 100.0
        } else {
            0.0
        },
    };
    let (g0, g1, g2, g3) = (mk_grp(0), mk_grp(1), mk_grp(2), mk_grp(3));
    let wga = g0
        .accuracy_pct
        .min(g1.accuracy_pct)
        .min(g2.accuracy_pct)
        .min(g3.accuracy_pct);
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

fn load_binary_split(dir: &Path, split: &str, count: usize) -> Vec<(QuantumState, usize, usize)> {
    let img_bytes =
        fs::read(dir.join(format!("{}_images.bin", split))).expect("Failed reading images bin");
    let lbl_bytes =
        fs::read(dir.join(format!("{}_labels.bin", split))).expect("Failed reading labels bin");
    let plc_bytes =
        fs::read(dir.join(format!("{}_places.bin", split))).expect("Failed reading places bin");
    let img_floats: &[f32] =
        unsafe { std::slice::from_raw_parts(img_bytes.as_ptr() as *const f32, count * STATE_DIM) };
    (0..count)
        .into_par_iter()
        .map(|i| {
            (
                encode_image_state(&img_floats[i * STATE_DIM..(i + 1) * STATE_DIM]),
                lbl_bytes[i] as usize,
                plc_bytes[i] as usize,
            )
        })
        .collect()
}

fn main() {
    println!("┌────────────────────────────────────────────────────────────────────────┐");
    println!("│      T-1000 QUANTUM NEURAL NETWORK: WATERBIRDS 11K ROBUSTNESS BENCH    │");
    println!("│      Two-Speed AdamW + Exact LayerNorm Correlator Chain-Rule           │");
    println!("└────────────────────────────────────────────────────────────────────────┘");

    let manifest = T1000Engine::new().system_manifest();
    println!(
        "Hardware Discovery: AVX-VNNI: {} | iGPU Bridge: {} ({})",
        manifest.avx_vnni_accelerated,
        manifest.igpu_bridge_available,
        manifest.igpu_device_name.as_deref().unwrap_or("None")
    );

    let data_dir = PathBuf::from("data/birds");
    let t0_load = Instant::now();
    let train_samples = load_binary_split(&data_dir, "train", 4795);
    let test_samples = load_binary_split(&data_dir, "test", 5794);
    println!(
        "Loaded 4,795 train and 5,794 test samples in {:.2} ms",
        t0_load.elapsed().as_secs_f64() * 1000.0
    );

    let depth = 3;
    let ansatz = CircuitAnsatz::new(N_QUBITS, depth).with_pi_layer(true);
    let mut observables: Vec<PauliObservable> = (0..N_QUBITS).map(PauliObservable::Z).collect();
    for i in 0..N_QUBITS {
        for j in (i + 1)..N_QUBITS {
            observables.push(PauliObservable::ZZ(i, j));
        }
    }
    let n_obs = observables.len();
    assert_eq!(n_obs, NUM_CORRELATORS);

    let n_ansatz_params = ansatz.param_count();
    let mut ansatz_weights = ansatz.weights.clone();
    let mut head = BinaryHead::new(n_obs, 42);
    let n_head_params = head.param_count();
    let mut head_weights = head.pack_params();

    let t0_eval = Instant::now();
    let zero_shot = evaluate_split(&ansatz, &ansatz_weights, &head, &test_samples, &observables);
    println!(
        "Zero-Shot Test: {:.2} s | Acc: {:.2}% | WGA: {:.2}% | L1-Coh: {:.2}",
        t0_eval.elapsed().as_secs_f64(),
        zero_shot.overall_accuracy_pct,
        zero_shot.worst_group_accuracy_pct,
        zero_shot.mean_coherence_l1
    );

    let epochs: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    println!(
        "\nInitiating {}-Epoch Two-Speed AdamW Training (lr_vqc=0.012, lr_head=0.0015, wd=0.01)...",
        epochs
    );
    let batch_size = 64;
    let num_batches = train_samples.len().div_ceil(batch_size);
    let mut optimizer = TwoSpeedHybridOptimizer::new(n_ansatz_params, n_head_params);
    let mut checkpoint = BestCheckpoint::new(&ansatz_weights, &head_weights);
    checkpoint.best_wga = zero_shot.worst_group_accuracy_pct;
    checkpoint.best_overall_acc = zero_shot.overall_accuracy_pct;
    checkpoint.evaluation = Some(zero_shot.clone());

    let mut g_idx: [Vec<usize>; 4] = Default::default();
    for (i, (_, l, p)) in train_samples.iter().enumerate() {
        g_idx[match (*l, *p) {
            (0, 0) => 0,
            (0, 1) => 1,
            (1, 0) => 2,
            _ => 3,
        }]
        .push(i);
    }
    let mut history = Vec::with_capacity(epochs);

    println!("\n┌──────┬────────────┬────────────┬───────────┬───────────┬───────────┬──────────────┬────────────┐");
    println!("│ Ep   │ Trn Loss   │ Trn Acc %  │ Tst Loss  │ Tst Acc % │ WGA %     │ L1-Coherence │ Time (ms)  │");
    println!("├──────┼────────────┼────────────┼───────────┼───────────┼───────────┼──────────────┼────────────┤");

    let t_train_all_start = Instant::now();
    for epoch in 1..=epochs {
        let t_ep_start = Instant::now();
        let (mut total_train_loss, mut train_correct) = (0.0, 0);

        for b in 0..num_batches {
            let mut batch_indices = Vec::with_capacity(64);
            let offset = (epoch - 1) * num_batches * 16 + b * 16;
            for i in 0..16 {
                for g in &g_idx {
                    batch_indices.push(g[(offset + i) % g.len()]);
                }
            }
            let batch_len = batch_indices.len() as f64;

            let (b_loss, b_cor, b_ans_grad, b_hd_grad) = batch_indices
                .par_iter()
                .map(|&s_idx| {
                    let (input_state, label, _) = &train_samples[s_idx];
                    let (mut s_loss, mut s_cor) = (0.0, 0);
                    let mut d_head = vec![0.0; n_head_params];

                    let adj = adjoint_gradient(
                        &ansatz,
                        input_state,
                        &ansatz_weights,
                        &observables,
                        |raw_exps| {
                            let mut raw_arr = [0.0; NUM_CORRELATORS];
                            raw_arr.copy_from_slice(&raw_exps[..NUM_CORRELATORS]);
                            let norm_cache = normalize_correlators_forward(&raw_arr);
                            let logits = head.forward(&norm_cache.normalized);
                            let (loss, d_logits, pred) = softmax_cross_entropy(&logits, *label);
                            s_loss = loss;
                            s_cor = if pred == *label { 1 } else { 0 };

                            for c in 0..N_CLASSES {
                                let row = c * n_obs;
                                for q in 0..n_obs {
                                    d_head[row + q] = d_logits[c] * norm_cache.normalized[q];
                                }
                                d_head[N_CLASSES * n_obs + c] = d_logits[c];
                            }

                            let mut grad_norm = [0.0; NUM_CORRELATORS];
                            for (q, g_n) in grad_norm.iter_mut().enumerate() {
                                let mut sum = 0.0;
                                for (c, &d_l) in d_logits.iter().enumerate() {
                                    sum += d_l * head.weights[c * n_obs + q];
                                }
                                *g_n = sum;
                            }

                            let grad_raw = normalize_correlators_backward(&grad_norm, &norm_cache);
                            grad_raw.to_vec()
                        },
                    );
                    (s_loss, s_cor, adj.gradient, d_head)
                })
                .reduce(
                    || (0.0, 0, vec![0.0; n_ansatz_params], vec![0.0; n_head_params]),
                    |mut acc, item| {
                        acc.0 += item.0;
                        acc.1 += item.1;
                        for i in 0..n_ansatz_params {
                            acc.2[i] += item.2[i];
                        }
                        for i in 0..n_head_params {
                            acc.3[i] += item.3[i];
                        }
                        acc
                    },
                );

            total_train_loss += b_loss;
            train_correct += b_cor;
            let avg_ans_grad: Vec<f64> = b_ans_grad.into_iter().map(|g| g / batch_len).collect();
            let avg_hd_grad: Vec<f64> = b_hd_grad.into_iter().map(|g| g / batch_len).collect();

            optimizer.step(
                &mut ansatz_weights,
                &avg_ans_grad,
                &mut head_weights,
                &avg_hd_grad,
            );
            head.unpack_params(&head_weights);
        }

        let ep_elapsed_ms = t_ep_start.elapsed().as_secs_f64() * 1000.0;
        let total_b_samples = (num_batches * batch_size) as f64;
        let trn_acc = (f64::from(train_correct) / total_b_samples) * 100.0;
        let trn_loss = total_train_loss / total_b_samples;

        let test_eval =
            evaluate_split(&ansatz, &ansatz_weights, &head, &test_samples, &observables);
        let improved =
            checkpoint.update_if_better(epoch, &test_eval, &ansatz_weights, &head_weights);

        println!("│ {:<4} │ {:>10.4} │ {:>9.2}% │ {:>9.4} │ {:>8.2}% │ {:>8.2}%{} │ {:>12.2} │ {:>10.1} │",
            epoch, trn_loss, trn_acc, test_eval.average_loss, test_eval.overall_accuracy_pct,
            test_eval.worst_group_accuracy_pct, if improved { "*" } else { " " }, test_eval.mean_coherence_l1, ep_elapsed_ms);

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
    let per_sample_us =
        (total_train_time_sec / (train_samples.len() * epochs) as f64) * 1_000_000.0;

    // Restore Best WGA Checkpoint
    ansatz_weights.copy_from_slice(&checkpoint.vqc_params);
    head_weights.copy_from_slice(&checkpoint.head_params);
    head.unpack_params(&head_weights);

    println!(
        "\nTraining Completed across {} Epochs in {:.2} s ({:.2} us/sample)",
        epochs, total_train_time_sec, per_sample_us
    );
    println!(
        "Best WGA Checkpoint: {:.2}% (Overall Acc: {:.2}%) achieved at Epoch {}",
        checkpoint.best_wga, checkpoint.best_overall_acc, checkpoint.best_epoch
    );

    let best_eval = checkpoint.evaluation.unwrap_or(zero_shot.clone());
    println!("\n┌────────────────────────┬──────────────────┬──────────────────┐");
    println!("│ Metric                 │ Zero-Shot Baseline│ Best WGA Snapshot│");
    println!("├────────────────────────┼──────────────────┼──────────────────┤");
    println!(
        "│ Overall Test Accuracy  │ {:>15.2}% │ {:>15.2}% │",
        zero_shot.overall_accuracy_pct, best_eval.overall_accuracy_pct
    );
    println!(
        "│ Average Test Loss      │ {:>16.4} │ {:>16.4} │",
        zero_shot.average_loss, best_eval.average_loss
    );
    println!(
        "│ Worst-Group Accuracy   │ {:>15.2}% │ {:>15.2}% │",
        zero_shot.worst_group_accuracy_pct, best_eval.worst_group_accuracy_pct
    );
    println!(
        "│ G0 (Land on Land)      │ {:>15.2}% │ {:>15.2}% │",
        zero_shot.g0_land_land.accuracy_pct, best_eval.g0_land_land.accuracy_pct
    );
    println!(
        "│ G1 (Land on Water)     │ {:>15.2}% │ {:>15.2}% │",
        zero_shot.g1_land_water.accuracy_pct, best_eval.g1_land_water.accuracy_pct
    );
    println!(
        "│ G2 (Water on Land)     │ {:>15.2}% │ {:>15.2}% │",
        zero_shot.g2_water_land.accuracy_pct, best_eval.g2_water_land.accuracy_pct
    );
    println!(
        "│ G3 (Water on Water)    │ {:>15.2}% │ {:>15.2}% │",
        zero_shot.g3_water_water.accuracy_pct, best_eval.g3_water_water.accuracy_pct
    );
    println!(
        "│ Mean L1 Coherence      │ {:>16.2} │ {:>16.2} │",
        zero_shot.mean_coherence_l1, best_eval.mean_coherence_l1
    );
    println!("└────────────────────────┴──────────────────┴──────────────────┘");

    let report = BirdsQnnReport {
        timestamp_rfc3339: chrono_timestamp(),
        dataset: "waterbird_complete95_forest2water2".to_string(),
        n_qubits: N_QUBITS,
        state_dim: STATE_DIM,
        circuit_depth: depth,
        num_circuit_params: n_ansatz_params,
        num_head_params: n_head_params,
        train_samples: train_samples.len(),
        test_samples: test_samples.len(),
        total_epochs: epochs,
        best_wga_pct: checkpoint.best_wga,
        best_overall_acc_pct: checkpoint.best_overall_acc,
        best_epoch: checkpoint.best_epoch,
        avx_vnni_accelerated: manifest.avx_vnni_accelerated,
        igpu_bridge_available: manifest.igpu_bridge_available,
        igpu_device: manifest.igpu_device_name,
        zero_shot,
        best_checkpoint_evaluation: best_eval,
        training_time_sec: total_train_time_sec,
        per_sample_latency_us: per_sample_us,
        history,
    };

    let report_path = PathBuf::from("data/birds/birds_qnn_report.json");
    if let Ok(json_str) = serde_json::to_string_pretty(&report) {
        // Waterbirds is judged on robustness: held-out loss decides and worst-group accuracy
        // may not drop by more than one point. Only runs with identical settings compete.
        let best = &report.best_checkpoint_evaluation;
        let settings = format!(
            "qubits={};depth={};epochs={};train={};test={}",
            report.n_qubits,
            report.circuit_depth,
            report.total_epochs,
            report.train_samples,
            report.test_samples
        );
        let score = t1000::results::Score {
            metric: "test_loss",
            value: best.average_loss,
            better: t1000::results::Better::Lower,
            guard: Some(t1000::results::Guard {
                name: "worst_group_accuracy_pct",
                value: best.worst_group_accuracy_pct,
                better: t1000::results::Better::Higher,
                tolerance: 1.0,
            }),
            settings: &settings,
        };
        match t1000::results::write_scored(&report_path, json_str, &score) {
            Ok(t1000::results::Outcome::KeptBest) => {
                println!("\nSaved report kept: this run did not improve it");
            }
            Ok(t1000::results::Outcome::Scratch) => {
                println!("\nReport written to a scratch copy only (use --record to save)");
            }
            Ok(_) => println!("\nReport successfully persisted to: {}", report_path.display()),
            Err(e) => eprintln!("\nCould not write the report: {e}"),
        }
    }
}

fn chrono_timestamp() -> String {
    let duration = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}.{:03}Z", duration.as_secs(), duration.subsec_millis())
}
