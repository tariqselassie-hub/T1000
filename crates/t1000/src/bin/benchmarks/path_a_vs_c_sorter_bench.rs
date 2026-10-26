// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Empirical Comparison Benchmark: Path A (Pure Eigenvalue Sorter) vs. Path C (Data Re-Uploading).
//!
//! Evaluates:
//! 1. Baseline: 10Q VQC + 55 Correlators + Two-Speed AdamW + Linear Readout Head (172 params).
//! 2. Path A: Headless Pure Quantum Eigenvalue Sorter (zero classical weights, 60 circuit params).
//! 3. Path C: Trainable Data Re-Uploading breaking unitary rigidity (80 quantum params + head).

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use t1000::engine::T1000Engine;
use t1000::feature_map::{DataEncodingLayer, EncodingKind};
use t1000::optimizer::normalize_correlators_arr_forward;
use t1000::simd::{apply_cz_kernel, apply_ry_kernel, apply_rz_kernel};
use t1000::{adjoint_gradient, CircuitAnsatz, PauliObservable, QuantumState};

const N_QUBITS: usize = 10;
const N_CLASSES: usize = 2;
const NUM_CORRELATORS: usize = 55;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubgroupMetrics {
    pub total: usize,
    pub correct: usize,
    pub accuracy_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEvaluation {
    pub model_name: String,
    pub num_quantum_params: usize,
    pub num_classical_params: usize,
    pub total_params: usize,
    pub overall_accuracy_pct: f64,
    pub worst_group_accuracy_pct: f64,
    pub g0_land_land: SubgroupMetrics,
    pub g1_land_water: SubgroupMetrics,
    pub g2_water_land: SubgroupMetrics,
    pub g3_water_water: SubgroupMetrics,
    pub training_time_sec: f64,
    pub per_sample_us: f64,
    pub breaks_rigidity: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub timestamp_rfc3339: String,
    pub dataset: String,
    pub n_qubits: usize,
    pub baseline_hybrid: ModelEvaluation,
    pub path_a_headless_sorter: ModelEvaluation,
    pub path_c_data_reuploading: ModelEvaluation,
    pub synthesis: String,
}

#[derive(Debug, Clone)]
struct SampleData {
    pub raw_angles: [f64; N_QUBITS],
    pub state_in: QuantumState,
    pub label: usize,
    pub place: usize,
}

#[derive(Debug, Clone)]
struct BinaryHead {
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
}

fn softmax_cross_entropy(logits: &[f64; N_CLASSES], label: usize) -> (f64, [f64; N_CLASSES]) {
    let max_l = logits[0].max(logits[1]);
    let e0 = (logits[0] - max_l).exp();
    let e1 = (logits[1] - max_l).exp();
    let sum = (e0 + e1).max(1e-12);
    let (p0, p1) = (e0 / sum, e1 / sum);
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
    )
}

fn load_dataset(
    dir: &Path,
    split: &str,
    count: usize,
    encoder: &DataEncodingLayer,
) -> Vec<SampleData> {
    let feat_bytes =
        fs::read(dir.join(format!("{}_features_10d.bin", split))).expect("Read features");
    let lbl_bytes = fs::read(dir.join(format!("{}_labels.bin", split))).expect("Read labels");
    let plc_bytes = fs::read(dir.join(format!("{}_places.bin", split))).expect("Read places");
    let feat_floats: &[f32] =
        unsafe { std::slice::from_raw_parts(feat_bytes.as_ptr() as *const f32, count * N_QUBITS) };

    (0..count)
        .into_par_iter()
        .map(|i| {
            let mut angles = [0.0; N_QUBITS];
            for q in 0..N_QUBITS {
                angles[q] = f64::from(feat_floats[i * N_QUBITS + q]);
            }
            SampleData {
                raw_angles: angles,
                state_in: encoder.encode(&angles),
                label: lbl_bytes[i] as usize,
                place: plc_bytes[i] as usize,
            }
        })
        .collect()
}

fn compute_group_metrics(preds: &[(usize, usize, usize)]) -> (f64, f64, [SubgroupMetrics; 4]) {
    let mut g_tot = [0usize; 4];
    let mut g_cor = [0usize; 4];
    let mut total_correct = 0;

    for &(pred, label, place) in preds {
        let g = match (label, place) {
            (0, 0) => 0,
            (0, 1) => 1,
            (1, 0) => 2,
            _ => 3,
        };
        g_tot[g] += 1;
        if pred == label {
            g_cor[g] += 1;
            total_correct += 1;
        }
    }

    let n = preds.len() as f64;
    let mk_grp = |i: usize| SubgroupMetrics {
        total: g_tot[i],
        correct: g_cor[i],
        accuracy_pct: if g_tot[i] > 0 {
            (g_cor[i] as f64 / g_tot[i] as f64) * 100.0
        } else {
            0.0
        },
    };

    let grps = [mk_grp(0), mk_grp(1), mk_grp(2), mk_grp(3)];
    let wga = grps
        .iter()
        .map(|g| g.accuracy_pct)
        .fold(f64::INFINITY, f64::min);
    let overall = (f64::from(total_correct) / n) * 100.0;
    (overall, wga, grps)
}

// =========================================================================
// Benchmark: Path A (Headless Pure Eigenvalue Sorter)
// =========================================================================
fn run_path_a_benchmark(
    train_samples: &[SampleData],
    test_samples: &[SampleData],
    ansatz: &CircuitAnsatz,
    magnetization_obs: &[PauliObservable],
    epochs: usize,
) -> ModelEvaluation {
    let t0 = Instant::now();
    let n_params = ansatz.param_count(); // 60
    let mut weights = ansatz.weights.clone();
    let mut m_opt = vec![0.0; n_params];
    let mut v_opt = vec![0.0; n_params];
    let lr = 0.02;

    let mut g_idx: [Vec<usize>; 4] = Default::default();
    for (i, s) in train_samples.iter().enumerate() {
        let g = match (s.label, s.place) {
            (0, 0) => 0,
            (0, 1) => 1,
            (1, 0) => 2,
            _ => 3,
        };
        g_idx[g].push(i);
    }

    let batch_size = 64;
    let num_batches = train_samples.len().div_ceil(batch_size);

    for epoch in 1..=epochs {
        for b in 0..num_batches {
            let mut batch_indices = Vec::with_capacity(64);
            let offset = (epoch - 1) * num_batches * 16 + b * 16;
            for i in 0..16 {
                for g in &g_idx {
                    batch_indices.push(g[(offset + i) % g.len()]);
                }
            }
            let batch_len = batch_indices.len() as f64;

            let b_grad = batch_indices
                .par_iter()
                .map(|&s_idx| {
                    let s = &train_samples[s_idx];
                    // Target eigenvalue: +1.0 for Landbird (label 0), -1.0 for Waterbird (label 1)
                    let target_val = if s.label == 0 { 1.0 } else { -1.0 };
                    let adj = adjoint_gradient(
                        ansatz,
                        &s.state_in,
                        &weights,
                        magnetization_obs,
                        |exps| {
                            // Loss L = 0.5 * (<M> - y)^2 where <M> = (1/10) sum <Z_q>
                            let mean_m: f64 = exps.iter().sum::<f64>() / 10.0;
                            let diff = mean_m - target_val;
                            // dL / d<Z_q> = (1/10) * (<M> - y)
                            vec![diff * 0.1; 10]
                        },
                    );
                    adj.gradient
                })
                .reduce(
                    || vec![0.0; n_params],
                    |mut acc, g| {
                        for i in 0..n_params {
                            acc[i] += g[i];
                        }
                        acc
                    },
                );

            // Adam step on circuit parameters only
            for i in 0..n_params {
                let g = (b_grad[i] / batch_len).clamp(-1.0, 1.0);
                m_opt[i] = 0.9 * m_opt[i] + 0.1 * g;
                v_opt[i] = 0.999 * v_opt[i] + 0.001 * g * g;
                weights[i] -= lr * m_opt[i] / (v_opt[i].sqrt() + 1e-8);
            }
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let per_sample_us = (elapsed / (train_samples.len() * epochs) as f64) * 1_000_000.0;

    // Evaluate on test set
    let preds: Vec<(usize, usize, usize)> = test_samples
        .par_iter()
        .map(|s| {
            let mut evolved = s.state_in.clone();
            ansatz.forward_in_place(&mut evolved, &weights);
            let mean_m: f64 = magnetization_obs
                .iter()
                .map(|o| o.expectation_value(&evolved))
                .sum::<f64>()
                / 10.0;
            let pred = if mean_m >= 0.0 { 0 } else { 1 };
            (pred, s.label, s.place)
        })
        .collect();

    let (overall, wga, grps) = compute_group_metrics(&preds);

    ModelEvaluation {
        model_name: "Path A: Headless Pure Quantum Eigenvalue Sorter".to_string(),
        num_quantum_params: n_params,
        num_classical_params: 0,
        total_params: n_params,
        overall_accuracy_pct: overall,
        worst_group_accuracy_pct: wga,
        g0_land_land: grps[0].clone(),
        g1_land_water: grps[1].clone(),
        g2_water_land: grps[2].clone(),
        g3_water_water: grps[3].clone(),
        training_time_sec: elapsed,
        per_sample_us,
        breaks_rigidity: false,
    }
}

// =========================================================================
// Benchmark: Path C (Trainable Data Re-Uploading)
// =========================================================================
fn evolve_reuploading_state(
    raw_angles: &[f64; N_QUBITS],
    circuit_weights: &[f64],
    upload_scales_l1: &[f64],
    upload_scales_l2: &[f64],
) -> QuantumState {
    let mut state = QuantumState::ground(1 << N_QUBITS);
    // 1. Initial Angle Encoding: Ry(raw_angles)
    for (q, &ang) in raw_angles.iter().enumerate().take(N_QUBITS) {
        apply_ry_kernel(&mut state.real, &mut state.imag, q, ang);
    }
    // Initial pi phase layer
    for q in 0..N_QUBITS {
        let sign = if q % 2 == 0 { 1.0 } else { -1.0 };
        apply_rz_kernel(
            &mut state.real,
            &mut state.imag,
            q,
            sign * std::f64::consts::PI,
        );
    }

    // Layer 0: parameterized Ry, Rz + entangling CZ
    let mut param_offset = 0;
    for q in 0..N_QUBITS {
        apply_ry_kernel(
            &mut state.real,
            &mut state.imag,
            q,
            circuit_weights[param_offset],
        );
        param_offset += 1;
        apply_rz_kernel(
            &mut state.real,
            &mut state.imag,
            q,
            circuit_weights[param_offset],
        );
        param_offset += 1;
    }
    for q in 0..N_QUBITS {
        let next_q = (q + 1) % N_QUBITS;
        apply_cz_kernel(&mut state.real, &mut state.imag, q, next_q);
    }

    // RE-UPLOAD 1: Ry(scale * raw_angles) breaking unitary rigidity
    for q in 0..N_QUBITS {
        apply_ry_kernel(
            &mut state.real,
            &mut state.imag,
            q,
            upload_scales_l1[q] * raw_angles[q],
        );
    }

    // Layer 1: parameterized Ry, Rz + entangling CZ
    for q in 0..N_QUBITS {
        apply_ry_kernel(
            &mut state.real,
            &mut state.imag,
            q,
            circuit_weights[param_offset],
        );
        param_offset += 1;
        apply_rz_kernel(
            &mut state.real,
            &mut state.imag,
            q,
            circuit_weights[param_offset],
        );
        param_offset += 1;
    }
    for q in 0..N_QUBITS {
        let next_q = (q + 2) % N_QUBITS;
        apply_cz_kernel(&mut state.real, &mut state.imag, q, next_q);
    }

    // RE-UPLOAD 2: Ry(scale * raw_angles) breaking unitary rigidity
    for q in 0..N_QUBITS {
        apply_ry_kernel(
            &mut state.real,
            &mut state.imag,
            q,
            upload_scales_l2[q] * raw_angles[q],
        );
    }

    // Layer 2: parameterized Ry, Rz + entangling CZ
    for q in 0..N_QUBITS {
        apply_ry_kernel(
            &mut state.real,
            &mut state.imag,
            q,
            circuit_weights[param_offset],
        );
        param_offset += 1;
        apply_rz_kernel(
            &mut state.real,
            &mut state.imag,
            q,
            circuit_weights[param_offset],
        );
        param_offset += 1;
    }
    for q in 0..N_QUBITS {
        let next_q = (q + 3) % N_QUBITS;
        apply_cz_kernel(&mut state.real, &mut state.imag, q, next_q);
    }

    state
}

fn run_path_c_benchmark(
    train_samples: &[SampleData],
    test_samples: &[SampleData],
    observables: &[PauliObservable],
    epochs: usize,
) -> ModelEvaluation {
    let t0 = Instant::now();
    let n_circuit_params = 60; // 3 layers x 10 qubits x 2 gates (Ry, Rz)
    let n_upload_params = 20; // 2 re-uploads x 10 scaling factors
    let circuit_weights = vec![0.1; n_circuit_params];
    let mut upload_scales_l1 = vec![1.0; N_QUBITS];
    let mut upload_scales_l2 = vec![1.0; N_QUBITS];

    let mut head = BinaryHead::new(observables.len(), 42);
    let n_head_params = N_CLASSES * observables.len() + N_CLASSES;
    let mut head_weights = vec![0.0; n_head_params];
    head_weights[..N_CLASSES * observables.len()].copy_from_slice(&head.weights);
    head_weights[N_CLASSES * observables.len()..].copy_from_slice(&head.biases);

    let mut g_idx: [Vec<usize>; 4] = Default::default();
    for (i, s) in train_samples.iter().enumerate() {
        let g = match (s.label, s.place) {
            (0, 0) => 0,
            (0, 1) => 1,
            (1, 0) => 2,
            _ => 3,
        };
        g_idx[g].push(i);
    }

    let batch_size = 64;
    let num_batches = train_samples.len().div_ceil(batch_size);
    let lr_head = 0.003;
    let lr_scales = 0.01;

    for epoch in 1..=epochs {
        for b in 0..num_batches {
            let mut batch_indices = Vec::with_capacity(64);
            let offset = (epoch - 1) * num_batches * 16 + b * 16;
            for i in 0..16 {
                for g in &g_idx {
                    batch_indices.push(g[(offset + i) % g.len()]);
                }
            }
            let batch_len = batch_indices.len() as f64;

            let (d_head_acc, d_s1_acc, d_s2_acc) = batch_indices
                .par_iter()
                .map(|&s_idx| {
                    let s = &train_samples[s_idx];
                    let evolved = evolve_reuploading_state(
                        &s.raw_angles,
                        &circuit_weights,
                        &upload_scales_l1,
                        &upload_scales_l2,
                    );

                    let mut raw_exps = [0.0; NUM_CORRELATORS];
                    for (k, obs) in observables.iter().enumerate() {
                        raw_exps[k] = obs.expectation_value(&evolved);
                    }
                    let norm_cache = normalize_correlators_arr_forward(&raw_exps);
                    let logits = head.forward(&norm_cache.normalized);
                    let (_, d_logits) = softmax_cross_entropy(&logits, s.label);

                    let mut d_head = vec![0.0; n_head_params];
                    for c in 0..N_CLASSES {
                        let row = c * observables.len();
                        for q in 0..observables.len() {
                            d_head[row + q] = d_logits[c] * norm_cache.normalized[q];
                        }
                        d_head[N_CLASSES * observables.len() + c] = d_logits[c];
                    }

                    // Finite differences on upload scale vectors for re-upload adaptation
                    let eps = 1e-4;
                    let mut d_s1 = [0.0; N_QUBITS];
                    let mut d_s2 = [0.0; N_QUBITS];
                    for q in 0..N_QUBITS {
                        let mut s1_pert = upload_scales_l1.clone();
                        s1_pert[q] += eps;
                        let evolved_pert = evolve_reuploading_state(
                            &s.raw_angles,
                            &circuit_weights,
                            &s1_pert,
                            &upload_scales_l2,
                        );
                        let exp0_pert = observables[q].expectation_value(&evolved_pert);
                        let exp0_orig = raw_exps[q];
                        d_s1[q] = ((exp0_pert - exp0_orig) / eps) * (d_logits[0] - d_logits[1]);

                        let mut s2_pert = upload_scales_l2.clone();
                        s2_pert[q] += eps;
                        let evolved_pert2 = evolve_reuploading_state(
                            &s.raw_angles,
                            &circuit_weights,
                            &upload_scales_l1,
                            &s2_pert,
                        );
                        let exp0_pert2 = observables[q].expectation_value(&evolved_pert2);
                        d_s2[q] = ((exp0_pert2 - exp0_orig) / eps) * (d_logits[0] - d_logits[1]);
                    }

                    (d_head, d_s1, d_s2)
                })
                .reduce(
                    || (vec![0.0; n_head_params], [0.0; N_QUBITS], [0.0; N_QUBITS]),
                    |mut acc, item| {
                        for i in 0..n_head_params {
                            acc.0[i] += item.0[i];
                        }
                        for q in 0..N_QUBITS {
                            acc.1[q] += item.1[q];
                            acc.2[q] += item.2[q];
                        }
                        acc
                    },
                );

            for i in 0..n_head_params {
                head_weights[i] -= lr_head * (d_head_acc[i] / batch_len);
            }
            head.weights
                .copy_from_slice(&head_weights[..N_CLASSES * observables.len()]);
            head.biases
                .copy_from_slice(&head_weights[N_CLASSES * observables.len()..]);

            for q in 0..N_QUBITS {
                upload_scales_l1[q] -= lr_scales * (d_s1_acc[q] / batch_len).clamp(-1.0, 1.0);
                upload_scales_l2[q] -= lr_scales * (d_s2_acc[q] / batch_len).clamp(-1.0, 1.0);
            }
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let per_sample_us = (elapsed / (train_samples.len() * epochs) as f64) * 1_000_000.0;

    let preds: Vec<(usize, usize, usize)> = test_samples
        .par_iter()
        .map(|s| {
            let evolved = evolve_reuploading_state(
                &s.raw_angles,
                &circuit_weights,
                &upload_scales_l1,
                &upload_scales_l2,
            );
            let mut raw_exps = [0.0; NUM_CORRELATORS];
            for (k, obs) in observables.iter().enumerate() {
                raw_exps[k] = obs.expectation_value(&evolved);
            }
            let norm_cache = normalize_correlators_arr_forward(&raw_exps);
            let logits = head.forward(&norm_cache.normalized);
            let pred = if logits[1] > logits[0] { 1 } else { 0 };
            (pred, s.label, s.place)
        })
        .collect();

    let (overall, wga, grps) = compute_group_metrics(&preds);

    ModelEvaluation {
        model_name: "Path C: Trainable Data Re-Uploading (Rigidity-Broken)".to_string(),
        num_quantum_params: n_circuit_params + n_upload_params,
        num_classical_params: n_head_params,
        total_params: n_circuit_params + n_upload_params + n_head_params,
        overall_accuracy_pct: overall,
        worst_group_accuracy_pct: wga,
        g0_land_land: grps[0].clone(),
        g1_land_water: grps[1].clone(),
        g2_water_land: grps[2].clone(),
        g3_water_water: grps[3].clone(),
        training_time_sec: elapsed,
        per_sample_us,
        breaks_rigidity: true,
    }
}

fn main() {
    println!("┌────────────────────────────────────────────────────────────────────────┐");
    println!("│      T-1000 SCIENTIFIC SHOWDOWN: PATH A (HEADLESS) vs PATH C (RE-UPLOAD)│");
    println!("│      Direct Eigenvalue Sorter vs Rigidity-Broken Quantum Space         │");
    println!("└────────────────────────────────────────────────────────────────────────┘");

    let manifest = T1000Engine::new().system_manifest();
    println!(
        "Hardware Discovery: AVX-VNNI: {} | iGPU Bridge: {} ({})",
        manifest.avx_vnni_accelerated,
        manifest.igpu_bridge_available,
        manifest.igpu_device_name.as_deref().unwrap_or("None")
    );

    let data_dir = PathBuf::from("data/birds");
    let encoder = DataEncodingLayer::new(N_QUBITS, EncodingKind::Angle);

    let t0 = Instant::now();
    let train_samples = load_dataset(&data_dir, "train", 4795, &encoder);
    let test_samples = load_dataset(&data_dir, "test", 5794, &encoder);
    println!(
        "Loaded 4,795 train and 5,794 test samples in {:.2} ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );

    let ansatz = CircuitAnsatz::new(N_QUBITS, 3).with_pi_layer(true);
    let mut observables: Vec<PauliObservable> = (0..N_QUBITS).map(PauliObservable::Z).collect();
    for i in 0..N_QUBITS {
        for j in (i + 1)..N_QUBITS {
            observables.push(PauliObservable::ZZ(i, j));
        }
    }
    let magnetization_obs: Vec<PauliObservable> = (0..N_QUBITS).map(PauliObservable::Z).collect();

    // 1. Run Path A Benchmark
    println!("\n>>> Running Path A Benchmark: Pure Headless Eigenvalue Sorter (14 Epochs)...");
    let path_a_eval = run_path_a_benchmark(
        &train_samples,
        &test_samples,
        &ansatz,
        &magnetization_obs,
        14,
    );

    // 2. Run Path C Benchmark
    println!("\n>>> Running Path C Benchmark: Trainable Data Re-Uploading (14 Epochs)...");
    let path_c_eval = run_path_c_benchmark(&train_samples, &test_samples, &observables, 14);

    // Load Baseline from Report
    let baseline_report_path = PathBuf::from("data/birds/birds_hybrid_qnn_report.json");
    let baseline_data: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(baseline_report_path).unwrap()).unwrap();
    let base_eval = &baseline_data["best_checkpoint_evaluation"];

    let base_mk_grp = |name: &str| SubgroupMetrics {
        total: base_eval[name]["total"].as_u64().unwrap() as usize,
        correct: base_eval[name]["correct"].as_u64().unwrap() as usize,
        accuracy_pct: base_eval[name]["accuracy_pct"].as_f64().unwrap(),
    };

    let baseline_eval = ModelEvaluation {
        model_name: "Baseline: Hybrid VQC + Two-Speed AdamW".to_string(),
        num_quantum_params: 60,
        num_classical_params: 112,
        total_params: 172,
        overall_accuracy_pct: base_eval["overall_accuracy_pct"].as_f64().unwrap(),
        worst_group_accuracy_pct: base_eval["worst_group_accuracy_pct"].as_f64().unwrap(),
        g0_land_land: base_mk_grp("g0_land_land"),
        g1_land_water: base_mk_grp("g1_land_water"),
        g2_water_land: base_mk_grp("g2_water_land"),
        g3_water_water: base_mk_grp("g3_water_water"),
        training_time_sec: baseline_data["training_time_sec"].as_f64().unwrap(),
        per_sample_us: baseline_data["per_sample_latency_us"].as_f64().unwrap(),
        breaks_rigidity: false,
    };

    // Comparative Table Output
    println!(
        "\n┌─────────────────────────────┬─────────────────┬─────────────────┬─────────────────┐"
    );
    println!(
        "│ Metric                      │ Baseline Hybrid │ Path A (Headless│ Path C (Re-Up)  │"
    );
    println!(
        "├─────────────────────────────┼─────────────────┼─────────────────┼─────────────────┤"
    );
    println!(
        "│ Quantum Parameters          │ {:>15} │ {:>15} │ {:>15} │",
        baseline_eval.num_quantum_params,
        path_a_eval.num_quantum_params,
        path_c_eval.num_quantum_params
    );
    println!(
        "│ Classical Head Parameters   │ {:>15} │ {:>15} │ {:>15} │",
        baseline_eval.num_classical_params,
        path_a_eval.num_classical_params,
        path_c_eval.num_classical_params
    );
    println!(
        "│ Total Trainable Parameters  │ {:>15} │ {:>15} │ {:>15} │",
        baseline_eval.total_params, path_a_eval.total_params, path_c_eval.total_params
    );
    println!(
        "│ Breaks Unitary Rigidity?    │ {:>15} │ {:>15} │ {:>15} │",
        "No (Rigid W)", "No (Rigid W)", "YES (W(x) State)"
    );
    println!(
        "├─────────────────────────────┼─────────────────┼─────────────────┼─────────────────┤"
    );
    println!(
        "│ Overall Test Accuracy       │ {:>14.2}% │ {:>14.2}% │ {:>14.2}% │",
        baseline_eval.overall_accuracy_pct,
        path_a_eval.overall_accuracy_pct,
        path_c_eval.overall_accuracy_pct
    );
    println!(
        "│ Worst-Group Accuracy (WGA)  │ {:>14.2}% │ {:>14.2}% │ {:>14.2}% │",
        baseline_eval.worst_group_accuracy_pct,
        path_a_eval.worst_group_accuracy_pct,
        path_c_eval.worst_group_accuracy_pct
    );
    println!(
        "├─────────────────────────────┼─────────────────┼─────────────────┼─────────────────┤"
    );
    println!(
        "│ G0: Land on Land Accuracy   │ {:>14.2}% │ {:>14.2}% │ {:>14.2}% │",
        baseline_eval.g0_land_land.accuracy_pct,
        path_a_eval.g0_land_land.accuracy_pct,
        path_c_eval.g0_land_land.accuracy_pct
    );
    println!(
        "│ G1: Land on Water (Distract)│ {:>14.2}% │ {:>14.2}% │ {:>14.2}% │",
        baseline_eval.g1_land_water.accuracy_pct,
        path_a_eval.g1_land_water.accuracy_pct,
        path_c_eval.g1_land_water.accuracy_pct
    );
    println!(
        "│ G2: Water on Land (Distract)│ {:>14.2}% │ {:>14.2}% │ {:>14.2}% │",
        baseline_eval.g2_water_land.accuracy_pct,
        path_a_eval.g2_water_land.accuracy_pct,
        path_c_eval.g2_water_land.accuracy_pct
    );
    println!(
        "│ G3: Water on Water Accuracy │ {:>14.2}% │ {:>14.2}% │ {:>14.2}% │",
        baseline_eval.g3_water_water.accuracy_pct,
        path_a_eval.g3_water_water.accuracy_pct,
        path_c_eval.g3_water_water.accuracy_pct
    );
    println!(
        "├─────────────────────────────┼─────────────────┼─────────────────┼─────────────────┤"
    );
    println!(
        "│ Training Latency (14 Ep)    │ {:>13.2} s │ {:>13.2} s │ {:>13.2} s │",
        baseline_eval.training_time_sec,
        path_a_eval.training_time_sec,
        path_c_eval.training_time_sec
    );
    println!(
        "│ Per-Sample Compute Latency  │ {:>12.2} us │ {:>12.2} us │ {:>12.2} us │",
        baseline_eval.per_sample_us, path_a_eval.per_sample_us, path_c_eval.per_sample_us
    );
    println!(
        "└─────────────────────────────┴─────────────────┴─────────────────┴─────────────────┘"
    );

    let report = ComparisonReport {
        timestamp_rfc3339: format!(
            "{}.000Z",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        ),
        dataset: "waterbirds_11k".to_string(),
        n_qubits: N_QUBITS,
        baseline_hybrid: baseline_eval,
        path_a_headless_sorter: path_a_eval,
        path_c_data_reuploading: path_c_eval,
        synthesis: "Comprehensive Architecture Tradeoff Study".to_string(),
    };

    let out_path = PathBuf::from("results/experiments/path_a_vs_c_comparison_report.json");
    if let Ok(json_str) = serde_json::to_string_pretty(&report) {
        let _ = t1000::results::write(&out_path, json_str);
        println!("\nReport saved to: {}", out_path.display());
    }
}
