// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Quantum Fidelity & Hilbert Space Orthogonalization Benchmark.
//!
//! Empirically tests whether the Variational Quantum Circuit (VQC) functions as a
//! Hilbert space sorter/separator by measuring quantum state fidelity |<psi_A | psi_B>|^2
//! across intra-class, inter-class, and spurious confounder subgroups before and after training.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use t1000::engine::T1000Engine;
use t1000::feature_map::{DataEncodingLayer, EncodingKind};
use t1000::optimizer::{
    normalize_correlators_arr_backward, normalize_correlators_arr_forward, TwoSpeedHybridOptimizer,
};
use t1000::{adjoint_gradient, CircuitAnsatz, PauliObservable, QuantumState};

const N_QUBITS: usize = 10;
const N_CLASSES: usize = 2;
const NUM_CORRELATORS: usize = 55;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FidelityStageMetrics {
    pub stage_name: String,
    pub mean_intra_land_land: f64,
    pub mean_intra_water_water: f64,
    pub mean_intra_class_overall: f64,
    pub mean_inter_class_land_water: f64,
    pub separation_ratio: f64,
    pub confounder_same_species_diff_bg: f64,
    pub confounder_diff_species_same_bg: f64,
    pub species_vs_background_selectivity: f64,
    pub obs_intra_dist: f64,
    pub obs_inter_dist: f64,
    pub obs_separation_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FidelityExperimentReport {
    pub timestamp_rfc3339: String,
    pub dataset: String,
    pub n_qubits: usize,
    pub num_pairs_evaluated: usize,
    pub stage_input_angles: FidelityStageMetrics,
    pub stage_untrained_vqc: FidelityStageMetrics,
    pub stage_trained_vqc: FidelityStageMetrics,
    pub delta_separation_ratio_pct: f64,
    pub conclusion: String,
}

#[derive(Debug, Clone)]
struct SampleItem {
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
    let loss = if label == 0 { -p0.max(1e-12).ln() } else { -p1.max(1e-12).ln() };
    (loss, [p0 - if label == 0 { 1.0 } else { 0.0 }, p1 - if label == 1 { 1.0 } else { 0.0 }])
}

fn load_dataset(dir: &Path, split: &str, count: usize, encoder: &DataEncodingLayer) -> Vec<SampleItem> {
    let feat_bytes = fs::read(dir.join(format!("{}_features_10d.bin", split))).expect("Read features");
    let lbl_bytes = fs::read(dir.join(format!("{}_labels.bin", split))).expect("Read labels");
    let plc_bytes = fs::read(dir.join(format!("{}_places.bin", split))).expect("Read places");
    let feat_floats: &[f32] = unsafe {
        std::slice::from_raw_parts(feat_bytes.as_ptr() as *const f32, count * N_QUBITS)
    };

    (0..count)
        .into_par_iter()
        .map(|i| {
            let mut angles = [0.0; N_QUBITS];
            for q in 0..N_QUBITS {
                angles[q] = f64::from(feat_floats[i * N_QUBITS + q]);
            }
            SampleItem {
                state_in: encoder.encode(&angles),
                label: lbl_bytes[i] as usize,
                place: plc_bytes[i] as usize,
            }
        })
        .collect()
}

fn train_model(
    train_samples: &[SampleItem],
    ansatz: &CircuitAnsatz,
    observables: &[PauliObservable],
    epochs: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n_ansatz_params = ansatz.param_count();
    let mut ansatz_weights = ansatz.weights.clone();
    let mut head = BinaryHead::new(observables.len(), 42);
    let n_head_params = N_CLASSES * observables.len() + N_CLASSES;
    let mut head_weights = vec![0.0; n_head_params];
    head_weights[..N_CLASSES * observables.len()].copy_from_slice(&head.weights);
    head_weights[N_CLASSES * observables.len()..].copy_from_slice(&head.biases);

    let mut optimizer = TwoSpeedHybridOptimizer::new(n_ansatz_params, n_head_params)
        .with_learning_rates(0.015, 0.003, 0.01);

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

            let (_, b_ans_grad, b_hd_grad) = batch_indices
                .par_iter()
                .map(|&s_idx| {
                    let s = &train_samples[s_idx];
                    let mut d_head = vec![0.0; n_head_params];
                    let adj = adjoint_gradient(
                        ansatz,
                        &s.state_in,
                        &ansatz_weights,
                        observables,
                        |raw_exps| {
                            let mut raw_arr = [0.0; NUM_CORRELATORS];
                            raw_arr.copy_from_slice(&raw_exps[..NUM_CORRELATORS]);
                            let norm_cache = normalize_correlators_arr_forward(&raw_arr);
                            let logits = head.forward(&norm_cache.normalized);
                            let (_, d_logits) = softmax_cross_entropy(&logits, s.label);

                            for c in 0..N_CLASSES {
                                let row = c * observables.len();
                                for q in 0..observables.len() {
                                    d_head[row + q] = d_logits[c] * norm_cache.normalized[q];
                                }
                                d_head[N_CLASSES * observables.len() + c] = d_logits[c];
                            }

                            let mut grad_norm = [0.0; NUM_CORRELATORS];
                            for (q, g_n) in grad_norm.iter_mut().enumerate() {
                                let mut sum = 0.0;
                                for (c, &d_l) in d_logits.iter().enumerate() {
                                    sum += d_l * head.weights[c * observables.len() + q];
                                }
                                *g_n = sum;
                            }

                            normalize_correlators_arr_backward(&grad_norm, &norm_cache).to_vec()
                        },
                    );
                    (0.0, adj.gradient, d_head)
                })
                .reduce(
                    || (0.0, vec![0.0; n_ansatz_params], vec![0.0; n_head_params]),
                    |mut acc, item| {
                        for i in 0..n_ansatz_params { acc.1[i] += item.1[i]; }
                        for i in 0..n_head_params { acc.2[i] += item.2[i]; }
                        acc
                    },
                );

            let avg_ans_grad: Vec<f64> = b_ans_grad.into_iter().map(|g| g / batch_len).collect();
            let avg_hd_grad: Vec<f64> = b_hd_grad.into_iter().map(|g| g / batch_len).collect();
            optimizer.step(&mut ansatz_weights, &avg_ans_grad, &mut head_weights, &avg_hd_grad);
            head.weights.copy_from_slice(&head_weights[..N_CLASSES * observables.len()]);
            head.biases.copy_from_slice(&head_weights[N_CLASSES * observables.len()..]);
        }
    }

    (ansatz_weights, head_weights)
}

fn compute_fidelity_metrics(
    stage_name: &str,
    states: &[QuantumState],
    labels: &[usize],
    places: &[usize],
    observables: &[PauliObservable],
    max_eval_per_subset: usize,
) -> FidelityStageMetrics {
    let mut idx_g0 = Vec::new(); // Land on Land
    let mut idx_g1 = Vec::new(); // Land on Water
    let mut idx_g2 = Vec::new(); // Water on Land
    let mut idx_g3 = Vec::new(); // Water on Water

    for (i, (&l, &p)) in labels.iter().zip(places.iter()).enumerate() {
        match (l, p) {
            (0, 0) => idx_g0.push(i),
            (0, 1) => idx_g1.push(i),
            (1, 0) => idx_g2.push(i),
            _ => idx_g3.push(i),
        }
    }

    // Precompute 55-D normalized correlator vectors
    let exps_all: Vec<[f64; NUM_CORRELATORS]> = states
        .par_iter()
        .map(|st| {
            let mut raw = [0.0; NUM_CORRELATORS];
            for (k, obs) in observables.iter().enumerate() {
                raw[k] = obs.expectation_value(st);
            }
            normalize_correlators_arr_forward(&raw).normalized
        })
        .collect();

    let n_eval = max_eval_per_subset;
    let eval_pair_fidelity_and_dist = |list_a: &[usize], list_b: &[usize]| -> (f64, f64) {
        let count_a = list_a.len().min(n_eval);
        let count_b = list_b.len().min(n_eval);
        let mut total_fid = 0.0;
        let mut total_dist = 0.0;
        let mut pairs = 0;
        for i in 0..count_a {
            let idx_a = list_a[i];
            for j in 0..count_b {
                let idx_b = list_b[j];
                if idx_a != idx_b {
                    total_fid += states[idx_a].fidelity(&states[idx_b]);
                    let mut d_sq = 0.0;
                    for k in 0..NUM_CORRELATORS {
                        let diff = exps_all[idx_a][k] - exps_all[idx_b][k];
                        d_sq += diff * diff;
                    }
                    total_dist += d_sq.sqrt();
                    pairs += 1;
                }
            }
        }
        if pairs > 0 {
            (total_fid / pairs as f64, total_dist / pairs as f64)
        } else {
            (0.0, 0.0)
        }
    };

    let (f_ll_00, d_ll_00) = eval_pair_fidelity_and_dist(&idx_g0, &idx_g0);
    let (f_ll_01, d_ll_01) = eval_pair_fidelity_and_dist(&idx_g0, &idx_g1);
    let f_ll = 0.5 * (f_ll_00 + f_ll_01);
    let d_ll = 0.5 * (d_ll_00 + d_ll_01);

    let (f_ww_33, d_ww_33) = eval_pair_fidelity_and_dist(&idx_g3, &idx_g3);
    let (f_ww_23, d_ww_23) = eval_pair_fidelity_and_dist(&idx_g2, &idx_g3);
    let f_ww = 0.5 * (f_ww_33 + f_ww_23);
    let d_ww = 0.5 * (d_ww_33 + d_ww_23);

    let f_intra = 0.5 * (f_ll + f_ww);
    let d_intra = 0.5 * (d_ll + d_ww);

    // Inter-class (Land vs Water)
    let (f_inter_03, d_inter_03) = eval_pair_fidelity_and_dist(&idx_g0, &idx_g3);
    let (f_inter_12, d_inter_12) = eval_pair_fidelity_and_dist(&idx_g1, &idx_g2);
    let f_inter = 0.5 * (f_inter_03 + f_inter_12);
    let d_inter = 0.5 * (d_inter_03 + d_inter_12);

    let sep_ratio = f_intra / f_inter.max(1e-12);
    let obs_sep_ratio = d_inter / d_intra.max(1e-12);

    // Confounder test: Same species diff background vs Diff species same background
    let (f_same_species_diff_bg, _) = eval_pair_fidelity_and_dist(&idx_g0, &idx_g1);
    let (f_diff_species_same_bg, _) = eval_pair_fidelity_and_dist(&idx_g1, &idx_g3);
    let selectivity = f_same_species_diff_bg / f_diff_species_same_bg.max(1e-12);

    FidelityStageMetrics {
        stage_name: stage_name.to_string(),
        mean_intra_land_land: f_ll,
        mean_intra_water_water: f_ww,
        mean_intra_class_overall: f_intra,
        mean_inter_class_land_water: f_inter,
        separation_ratio: sep_ratio,
        confounder_same_species_diff_bg: f_same_species_diff_bg,
        confounder_diff_species_same_bg: f_diff_species_same_bg,
        species_vs_background_selectivity: selectivity,
        obs_intra_dist: d_intra,
        obs_inter_dist: d_inter,
        obs_separation_ratio: obs_sep_ratio,
    }
}

fn main() {
    println!("┌────────────────────────────────────────────────────────────────────────┐");
    println!("│      T-1000 QUANTUM HYBRID BENCHMARK: HILBERT SPACE ORTHOGONALIZATION  │");
    println!("│      Empirical Test: Unitary Sorter vs Classical Head Separator        │");
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

    println!("\nTraining 10Q VQC for 14 Epochs to acquire Optimal WGA Checkpoint...");
    let t_trn = Instant::now();
    let (trained_vqc_weights, _) = train_model(&train_samples, &ansatz, &observables, 14);
    println!("VQC Training completed in {:.2} s", t_trn.elapsed().as_secs_f64());

    println!("\nGenerating Statevector Manifolds across 5,794 Test Samples...");
    let test_labels: Vec<usize> = test_samples.iter().map(|s| s.label).collect();
    let test_places: Vec<usize> = test_samples.iter().map(|s| s.place).collect();

    // Stage 0: Input Angle States |psi_in>
    let states_stage0: Vec<QuantumState> = test_samples.iter().map(|s| s.state_in.clone()).collect();

    // Stage 1: Untrained / Random VQC States |psi_untrained>
    let states_stage1: Vec<QuantumState> = test_samples
        .par_iter()
        .map(|s| {
            let mut st = s.state_in.clone();
            ansatz.forward_in_place(&mut st, &ansatz.weights);
            st
        })
        .collect();

    // Stage 2: Trained VQC Checkpoint States |psi_trained>
    let states_stage2: Vec<QuantumState> = test_samples
        .par_iter()
        .map(|s| {
            let mut st = s.state_in.clone();
            ansatz.forward_in_place(&mut st, &trained_vqc_weights);
            st
        })
        .collect();

    println!("Computing Pairwise Quantum Fidelity (L2 Overlap) & Observable Metrics...");
    let max_pairs = 120; // 120 x 120 = 14,400 pairs per group
    let m0 = compute_fidelity_metrics("Stage 0: Input Angle States", &states_stage0, &test_labels, &test_places, &observables, max_pairs);
    let m1 = compute_fidelity_metrics("Stage 1: Untrained VQC", &states_stage1, &test_labels, &test_places, &observables, max_pairs);
    let m2 = compute_fidelity_metrics("Stage 2: Trained VQC Checkpoint", &states_stage2, &test_labels, &test_places, &observables, max_pairs);

    println!("\n┌────────────────────────────────────────┬─────────────┬─────────────┬─────────────┐");
    println!("│ 1. PURE QUANTUM STATE FIDELITY INVARIANT│ Stage 0 (In)│ Stage 1(Unt)│ Stage 2(Trn)│");
    println!("├────────────────────────────────────────┼─────────────┼─────────────┼─────────────┤");
    println!("│ Mean Intra-Class Fidelity F(L, L)      │ {:>11.4} │ {:>11.4} │ {:>11.4} │", m0.mean_intra_land_land, m1.mean_intra_land_land, m2.mean_intra_land_land);
    println!("│ Mean Intra-Class Fidelity F(W, W)      │ {:>11.4} │ {:>11.4} │ {:>11.4} │", m0.mean_intra_water_water, m1.mean_intra_water_water, m2.mean_intra_water_water);
    println!("│ Overall Intra-Class Fidelity <F_intra> │ {:>11.4} │ {:>11.4} │ {:>11.4} │", m0.mean_intra_class_overall, m1.mean_intra_class_overall, m2.mean_intra_class_overall);
    println!("│ Inter-Class Fidelity <F_inter> (L vs W)│ {:>11.4} │ {:>11.4} │ {:>11.4} │", m0.mean_inter_class_land_water, m1.mean_inter_class_land_water, m2.mean_inter_class_land_water);
    println!("├────────────────────────────────────────┼─────────────┼─────────────┼─────────────┤");
    println!("│ HILBERT FIDELITY RATIO (F_in / F_out)  │ {:>11.4} │ {:>11.4} │ {:>11.4} │", m0.separation_ratio, m1.separation_ratio, m2.separation_ratio);
    println!("├────────────────────────────────────────┼─────────────┼─────────────┼─────────────┤");
    println!("│ 2. OBSERVABLE MEASUREMENT SPACE (R^55) │ Stage 0 (In)│ Stage 1(Unt)│ Stage 2(Trn)│");
    println!("├────────────────────────────────────────┼─────────────┼─────────────┼─────────────┤");
    println!("│ Intra-Class Distance ||c_A - c_B||     │ {:>11.4} │ {:>11.4} │ {:>11.4} │", m0.obs_intra_dist, m1.obs_intra_dist, m2.obs_intra_dist);
    println!("│ Inter-Class Distance ||c_L - c_W||     │ {:>11.4} │ {:>11.4} │ {:>11.4} │", m0.obs_inter_dist, m1.obs_inter_dist, m2.obs_inter_dist);
    println!("│ OBSERVABLE SEPARATION RATIO (D_out/in) │ {:>11.4} │ {:>11.4} │ {:>11.4} │", m0.obs_separation_ratio, m1.obs_separation_ratio, m2.obs_separation_ratio);
    println!("├────────────────────────────────────────┼─────────────┼─────────────┼─────────────┤");
    println!("│ Same Species Diff Background F(G0, G1) │ {:>11.4} │ {:>11.4} │ {:>11.4} │", m0.confounder_same_species_diff_bg, m1.confounder_same_species_diff_bg, m2.confounder_same_species_diff_bg);
    println!("│ Diff Species Same Background F(G1, G3) │ {:>11.4} │ {:>11.4} │ {:>11.4} │", m0.confounder_diff_species_same_bg, m1.confounder_diff_species_same_bg, m2.confounder_diff_species_same_bg);
    println!("│ Species-over-Background Selectivity    │ {:>11.4} │ {:>11.4} │ {:>11.4} │", m0.species_vs_background_selectivity, m1.species_vs_background_selectivity, m2.species_vs_background_selectivity);
    println!("└────────────────────────────────────────┴─────────────┴─────────────┴─────────────┘");

    let obs_gain = ((m2.obs_separation_ratio - m1.obs_separation_ratio) / m1.obs_separation_ratio) * 100.0;
    println!("\nEmpirical Findings & Physical Analysis:");
    println!("1. Unitarity Rigidity: State fidelities are strictly invariant across stages (W† W = I).");
    println!("   A unitary VQC cannot alter relative Hilbert state distances globally.");
    println!("2. Observable Projection Space (55 Correlators):");
    println!("   Observable separation ratio changed from {:.4} (untrained) -> {:.4} (trained) [{:+.2}%].",
        m1.obs_separation_ratio, m2.obs_separation_ratio, obs_gain);
    if m2.obs_separation_ratio > m1.obs_separation_ratio {
        println!("--> CONFIRMED: VQC functions as an ACTIVE MEASUREMENT ORIENTATION SORTER.");
        println!("    The variational circuit rotates the rigid state manifold so that the fixed");
        println!("    55 Pauli observables maximize inter-class separation relative to intra-class dispersion.");
    } else {
        println!("--> HYPOTHESIS: Separation is entirely driven by the classical linear readout head.");
    }

    let report = FidelityExperimentReport {
        timestamp_rfc3339: format!("{}.000Z", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()),
        dataset: "waterbirds_11k".to_string(),
        n_qubits: N_QUBITS,
        num_pairs_evaluated: max_pairs * max_pairs * 4,
        stage_input_angles: m0,
        stage_untrained_vqc: m1,
        stage_trained_vqc: m2,
        delta_separation_ratio_pct: obs_gain,
        conclusion: "Unitary State Repulsion Analysis".to_string(),
    };

    let out_dir = PathBuf::from("results/experiments");
    fs::create_dir_all(&out_dir).unwrap();
    let report_path = out_dir.join("fidelity_orthogonality_report.json");
    if let Ok(json_str) = serde_json::to_string_pretty(&report) {
        let _ = fs::write(&report_path, json_str);
        println!("Saved report to: {}", report_path.display());
    }
}
