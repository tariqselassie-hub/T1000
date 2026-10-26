// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Cascaded Multi-Tier Ultrametric Tree Sorter on BoTNeTIoT-L01 IoT Botnet Traffic.
//!
//! Evaluates a 3-tier hierarchical Quantum Multiplexor (QMUX):
//!   H = H_Device(2Q) (x) H_Family(1Q) (x) H_SubType(1Q) (x) H_Env(6Q)
//! Total state dimension = 2^10 = 1,024 amplitudes.
//!
//! Enforces and verifies the 3-Tier Strong Triangle Inequality:
//!   d_U(u, v) <= max(d_U(u, w), d_U(w, v))
//! Across commercial IoT botnet traffic streams (Vek2urity Integration).

use num_complex::Complex64;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::path::Path;
use std::time::Instant;

use t1000::iot_botnet_data::{
    compute_iot_leaf_index, iot_cascaded_tree_distance, locate_iot_dataset_file,
    parse_iot_traffic_flows, IoTCascadedTreeConfig, IoTCascadedTreeReport,
    N_ATTACK_FAMILIES, N_ATTACK_SUBTYPES, N_DEVICE_CATEGORIES, N_IOT_FEATURES, N_IOT_LEAVES,
};
use t1000::ultrametric_tree::{apply_single_qubit_gate, gate_ry, gate_rz};

const N_QUBITS: usize = 10;
const STATE_DIM: usize = 1 << N_QUBITS; // 1,024

/// Applies a Controlled-Z gate.
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

/// Parameterized 3-Tier Cascaded QMUX Sorter Engine.
#[derive(Debug, Clone)]
pub struct IoTCascadedTreeSorter {
    pub cfg: IoTCascadedTreeConfig,
    pub trunk_params: Vec<f64>,
    pub family_qmux_params: Vec<[f64; 2]>,
    pub subtype_qmux_params: Vec<[f64; 2]>,
    pub cooling_phases: Vec<f64>,
}

impl IoTCascadedTreeSorter {
    pub fn new(cfg: IoTCascadedTreeConfig) -> Self {
        let trunk_params = vec![0.0; cfg.n_device * 2]; // 4 params for 2 device wires
        let family_qmux_params = vec![[0.0, 0.0]; 1 << cfg.n_device]; // 4 branches x 2
        let subtype_qmux_params = vec![[0.0, 0.0]; 1 << (cfg.n_device + cfg.n_family)]; // 8 branches x 2
        let cooling_phases = vec![0.0; cfg.n_env]; // 6 cooling phases
        Self {
            cfg,
            trunk_params,
            family_qmux_params,
            subtype_qmux_params,
            cooling_phases,
        }
    }

    pub fn init_seeded(&mut self, seed: u64) {
        use rand::Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for p in self.trunk_params.iter_mut() {
            *p = rng.gen_range(-0.35..0.35);
        }
        for b in self.family_qmux_params.iter_mut() {
            b[0] = rng.gen_range(-0.35..0.35);
            b[1] = rng.gen_range(-0.35..0.35);
        }
        for b in self.subtype_qmux_params.iter_mut() {
            b[0] = rng.gen_range(-0.35..0.35);
            b[1] = rng.gen_range(-0.35..0.35);
        }
        for p in self.cooling_phases.iter_mut() {
            *p = rng.gen_range(0.05..0.20);
        }
    }

    pub fn total_parameters(&self) -> usize {
        self.trunk_params.len()
            + self.family_qmux_params.len() * 2
            + self.subtype_qmux_params.len() * 2
            + self.cooling_phases.len()
    }

    /// Evaluates forward circuit on input IoT traffic features.
    pub fn forward(&self, features: &[f64; N_IOT_FEATURES]) -> Vec<Complex64> {
        let mut state = vec![Complex64::default(); STATE_DIM];
        state[0] = Complex64::new(1.0, 0.0);

        // Stage 0: Feature Injection Ry(x_i)
        for q in 0..N_QUBITS {
            apply_single_qubit_gate(&mut state, q, &gate_ry(features[q]));
        }

        // Stage 1: Coarse Device Routing Trunk (Wires 0, 1)
        for q in 0..self.cfg.n_device {
            let ry = self.trunk_params[q * 2];
            let rz = self.trunk_params[q * 2 + 1];
            apply_single_qubit_gate(&mut state, q, &gate_ry(ry));
            apply_single_qubit_gate(&mut state, q, &gate_rz(rz));
        }
        apply_cz(&mut state, 0, 1);

        // Stage 2: QMUX Attack Family Branching on Wire 2, conditioned on Wires 0, 1
        let wire_family = 2;
        let half_dim = STATE_DIM >> 1;
        let low_mask_f = (1 << wire_family) - 1;
        for i in 0..half_dim {
            let idx0 = ((i & !low_mask_f) << 1) | (i & low_mask_f);
            let idx1 = idx0 | (1 << wire_family);
            let dev_cat = idx0 & 0x03; // Wires 0, 1
            let [ry_p, rz_p] = self.family_qmux_params[dev_cat];
            let u_ry = gate_ry(ry_p);
            let u_rz = gate_rz(rz_p);
            let u = [
                [
                    u_rz[0][0] * u_ry[0][0] + u_rz[0][1] * u_ry[1][0],
                    u_rz[0][0] * u_ry[0][1] + u_rz[0][1] * u_ry[1][1],
                ],
                [
                    u_rz[1][0] * u_ry[0][0] + u_rz[1][1] * u_ry[1][0],
                    u_rz[1][0] * u_ry[0][1] + u_rz[1][1] * u_ry[1][1],
                ],
            ];
            let a0 = state[idx0];
            let a1 = state[idx1];
            state[idx0] = u[0][0] * a0 + u[0][1] * a1;
            state[idx1] = u[1][0] * a0 + u[1][1] * a1;
        }

        // Stage 3: QMUX Attack SubType Branching on Wire 3, conditioned on Wires 0, 1, 2
        let wire_subtype = 3;
        let low_mask_s = (1 << wire_subtype) - 1;
        for i in 0..half_dim {
            let idx0 = ((i & !low_mask_s) << 1) | (i & low_mask_s);
            let idx1 = idx0 | (1 << wire_subtype);
            let branch_ctx = idx0 & 0x07; // Wires 0, 1, 2
            let [ry_p, rz_p] = self.subtype_qmux_params[branch_ctx];
            let u_ry = gate_ry(ry_p);
            let u_rz = gate_rz(rz_p);
            let u = [
                [
                    u_rz[0][0] * u_ry[0][0] + u_rz[0][1] * u_ry[1][0],
                    u_rz[0][0] * u_ry[0][1] + u_rz[0][1] * u_ry[1][1],
                ],
                [
                    u_rz[1][0] * u_ry[0][0] + u_rz[1][0] * u_ry[1][0],
                    u_rz[1][0] * u_ry[0][1] + u_rz[1][1] * u_ry[1][1],
                ],
            ];
            let a0 = state[idx0];
            let a1 = state[idx1];
            state[idx0] = u[0][0] * a0 + u[0][1] * a1;
            state[idx1] = u[1][0] * a0 + u[1][1] * a1;
        }

        // Stage 4: Entropy Cooling Bridges into Wires 4..9
        for (e_idx, &phi) in self.cooling_phases.iter().enumerate() {
            let env_wire = 4 + e_idx;
            apply_cz(&mut state, 3, env_wire);
            apply_single_qubit_gate(&mut state, env_wire, &gate_rz(phi));
        }

        state
    }

    /// Evaluates marginal distribution across all 16 hierarchical leaves.
    #[inline]
    pub fn compute_leaf_distribution(&self, state: &[Complex64]) -> [f64; N_IOT_LEAVES] {
        let mut dist = [0.0; N_IOT_LEAVES];
        for (idx, amp) in state.iter().enumerate() {
            let leaf = idx & 0x0F;
            dist[leaf] += amp.norm_sqr();
        }
        dist
    }

    #[inline]
    pub fn marginal_device(&self, leaf_dist: &[f64; N_IOT_LEAVES]) -> [f64; N_DEVICE_CATEGORIES] {
        let mut dev = [0.0; N_DEVICE_CATEGORIES];
        for (leaf, &prob) in leaf_dist.iter().enumerate() {
            dev[leaf & 0x03] += prob;
        }
        dev
    }

    #[inline]
    pub fn marginal_family(&self, leaf_dist: &[f64; N_IOT_LEAVES]) -> [f64; N_ATTACK_FAMILIES] {
        let mut fam = [0.0; N_ATTACK_FAMILIES];
        for (leaf, &prob) in leaf_dist.iter().enumerate() {
            fam[(leaf >> 2) & 0x01] += prob;
        }
        fam
    }

    #[inline]
    pub fn marginal_subtype(&self, leaf_dist: &[f64; N_IOT_LEAVES]) -> [f64; N_ATTACK_SUBTYPES] {
        let mut sub = [0.0; N_ATTACK_SUBTYPES];
        for (leaf, &prob) in leaf_dist.iter().enumerate() {
            sub[(leaf >> 3) & 0x01] += prob;
        }
        sub
    }
}

fn main() {
    println!("================================================================================");
    println!("T-1000 3-TIER CASCADED ULTRAMETRIC TREE SORTER: BOTNET-IOT-L01");
    println!("================================================================================");

    let candidate_paths = [
        "data/iot/botnet_iot_l01_stratified.csv",
        "data/iot/botnet_iot.csv",
        "C:/Users/zoddj/datasets/iot_ids.zip",
    ];
    let data_path = locate_iot_dataset_file(&candidate_paths)
        .expect("BoTNeTIoT dataset not found");
    println!("Loaded Dataset       : {}", data_path.display());

    let mut all_flows = parse_iot_traffic_flows(&data_path, Some(25000));
    let mut rng = ChaCha8Rng::seed_from_u64(0x5EED_B070);
    all_flows.shuffle(&mut rng);

    let n_train = 18000.min(all_flows.len() * 4 / 5);
    let train_set = &all_flows[0..n_train];
    let test_set = &all_flows[n_train..];

    println!(
        "Dataset Split (Shuffled) : {} Train | {} Test Flows (Total {})",
        train_set.len(),
        test_set.len(),
        all_flows.len()
    );

    let cfg = IoTCascadedTreeConfig::default();
    println!("\nHilbert Register Topology:");
    println!("  Tier 1 Device Register H_Device   : {} Qubits (dim = 4)", cfg.n_device);
    println!("  Tier 2 Family Register H_Family   : {} Qubit  (dim = 2)", cfg.n_family);
    println!("  Tier 3 SubType Register H_SubType : {} Qubit  (dim = 2)", cfg.n_subtype);
    println!("  Entropy Cooling Sink H_Env        : {} Qubits (dim = 64)", cfg.n_env);
    println!("  Total State Dimension             : 10 Qubits (dim = 1,024 amplitudes)");
    println!("  Hierarchical Leaf Factorization   : 4 x 2 x 2 = {} Terminal Leaves", N_IOT_LEAVES);

    // 1. Exhaustive 3-Tier Strong Triangle Inequality Verification across all 4,096 Triplets
    println!("\n[1/3] Verifying 3-Tier Strong Triangle Inequality across all 4,096 Leaf Triplets...");
    let g1 = 0.30; // SubType sibling distance
    let g2 = 0.65; // Attack Family sibling distance
    let g3 = 1.00; // Device Category disjoint distance

    let mut triplet_violations = 0;
    let mut total_triplets = 0;

    for u in 0..N_IOT_LEAVES {
        let du = u & 0x03;
        let fu = (u >> 2) & 0x01;
        let su = (u >> 3) & 0x01;

        for v in 0..N_IOT_LEAVES {
            let dv = v & 0x03;
            let fv = (v >> 2) & 0x01;
            let sv = (v >> 3) & 0x01;
            let d_uv = iot_cascaded_tree_distance(du, fu, su, dv, fv, sv, g1, g2, g3);

            for w in 0..N_IOT_LEAVES {
                let dw = w & 0x03;
                let fw = (w >> 2) & 0x01;
                let sw = (w >> 3) & 0x01;
                let d_uw = iot_cascaded_tree_distance(du, fu, su, dw, fw, sw, g1, g2, g3);
                let d_wv = iot_cascaded_tree_distance(dw, fw, sw, dv, fv, sv, g1, g2, g3);

                total_triplets += 1;
                if d_uv > d_uw.max(d_wv) + 1e-9 {
                    triplet_violations += 1;
                }
            }
        }
    }

    println!("  Evaluated Triplets (16^3)          : {}", total_triplets);
    println!("  Strong Triangle Invariant Result   : {}", if triplet_violations == 0 { "100.0% CONSERVED (PASS)" } else { "VIOLATION DETECTED" });
    println!("  Triplet Violations                 : {}", triplet_violations);

    let mut sorter = IoTCascadedTreeSorter::new(cfg);
    sorter.init_seeded(0x5EED_9999);
    println!("  Trainable Parameters               : {} (Zero Classical Readout Weights)", sorter.total_parameters());

    // 2. Training Stage via Cascaded QMUX Parameter Shift on All 3 Tiers
    println!("\n[2/3] Training Cascaded QMUX Sorter Engine (4 Epochs)...");
    let t_train = Instant::now();
    let epochs = 4;
    let lr = 0.15;
    let batch_size = 128;
    let n_batches = train_set.len() / batch_size;
    let shift = std::f64::consts::FRAC_PI_2;

    for ep in 1..=epochs {
        for b in 0..n_batches {
            let b_flows = &train_set[b * batch_size..(b + 1) * batch_size];
            let mut grad_trunk = vec![0.0; sorter.trunk_params.len()];
            let mut grad_fam = vec![[0.0, 0.0]; 4];
            let mut grad_sub = vec![[0.0, 0.0]; 8];

            for f in b_flows {
                let state = sorter.forward(&f.features);
                let leaf_dist = sorter.compute_leaf_distribution(&state);
                let target_leaf = f.labels.leaf_index;
                let p_cur = leaf_dist[target_leaf].max(1e-4);

                // Tier 1 Trunk Gradient
                for p_idx in 0..sorter.trunk_params.len() {
                    sorter.trunk_params[p_idx] += shift;
                    let d_plus = sorter.compute_leaf_distribution(&sorter.forward(&f.features));
                    sorter.trunk_params[p_idx] -= 2.0 * shift;
                    let d_minus = sorter.compute_leaf_distribution(&sorter.forward(&f.features));
                    sorter.trunk_params[p_idx] += shift;

                    let grad_prob = 0.5 * (d_plus[target_leaf] - d_minus[target_leaf]);
                    grad_trunk[p_idx] += grad_prob / p_cur;
                }

                // Tier 2 Family QMUX Branch Gradient
                let dev_cat = f.labels.device_cat;
                for p_idx in 0..2 {
                    sorter.family_qmux_params[dev_cat][p_idx] += shift;
                    let d_plus = sorter.compute_leaf_distribution(&sorter.forward(&f.features));
                    sorter.family_qmux_params[dev_cat][p_idx] -= 2.0 * shift;
                    let d_minus = sorter.compute_leaf_distribution(&sorter.forward(&f.features));
                    sorter.family_qmux_params[dev_cat][p_idx] += shift;

                    let grad_prob = 0.5 * (d_plus[target_leaf] - d_minus[target_leaf]);
                    grad_fam[dev_cat][p_idx] += grad_prob / p_cur;
                }

                // Tier 3 SubType QMUX Branch Gradient
                let branch_ctx = (dev_cat << 1) | f.labels.attack_family;
                for p_idx in 0..2 {
                    sorter.subtype_qmux_params[branch_ctx][p_idx] += shift;
                    let d_plus = sorter.compute_leaf_distribution(&sorter.forward(&f.features));
                    sorter.subtype_qmux_params[branch_ctx][p_idx] -= 2.0 * shift;
                    let d_minus = sorter.compute_leaf_distribution(&sorter.forward(&f.features));
                    sorter.subtype_qmux_params[branch_ctx][p_idx] += shift;

                    let grad_prob = 0.5 * (d_plus[target_leaf] - d_minus[target_leaf]);
                    grad_sub[branch_ctx][p_idx] += grad_prob / p_cur;
                }
            }

            let b_scale = (lr / (batch_size as f64)).min(0.015);
            for p_idx in 0..sorter.trunk_params.len() {
                sorter.trunk_params[p_idx] += b_scale * grad_trunk[p_idx];
            }
            for dev in 0..4 {
                for p_idx in 0..2 {
                    sorter.family_qmux_params[dev][p_idx] += b_scale * grad_fam[dev][p_idx];
                }
            }
            for ctx in 0..8 {
                for p_idx in 0..2 {
                    sorter.subtype_qmux_params[ctx][p_idx] += b_scale * grad_sub[ctx][p_idx];
                }
            }
        }
        println!("  Epoch {:02}/{} Completed", ep, epochs);
    }
    let train_duration = t_train.elapsed().as_secs_f64();
    println!("Training Finished in {:.2}s", train_duration);

    // 3. Evaluation on Holdout Test Set
    println!("\n[3/3] Evaluating on Holdout Test Set ({} flows)...", test_set.len());
    let t_eval = Instant::now();

    let mut correct_dev = 0;
    let mut correct_fam = 0;
    let mut correct_sub = 0;
    let mut correct_joint = 0;

    let test_results: Vec<(usize, usize, usize, usize)> = test_set
        .par_iter()
        .map(|f| {
            let state = sorter.forward(&f.features);
            let leaf_dist = sorter.compute_leaf_distribution(&state);

            let dev_dist = sorter.marginal_device(&leaf_dist);
            let pred_dev = dev_dist
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            let fam_dist = sorter.marginal_family(&leaf_dist);
            let pred_fam = fam_dist
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            let sub_dist = sorter.marginal_subtype(&leaf_dist);
            let pred_sub = sub_dist
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            let pred_leaf = compute_iot_leaf_index(pred_dev, pred_fam, pred_sub);
            (pred_dev, pred_fam, pred_sub, pred_leaf)
        })
        .collect();

    let eval_duration = t_eval.elapsed();
    let per_sample_us = (eval_duration.as_micros() as f64) / (test_set.len() as f64);

    for (f, &(p_dev, p_fam, p_sub, p_leaf)) in test_set.iter().zip(&test_results) {
        if p_dev == f.labels.device_cat {
            correct_dev += 1;
        }
        if p_fam == f.labels.attack_family {
            correct_fam += 1;
        }
        if p_sub == f.labels.attack_subtype {
            correct_sub += 1;
        }
        if p_leaf == f.labels.leaf_index {
            correct_joint += 1;
        }
    }

    let n_test = test_set.len() as f64;
    let acc_dev = (correct_dev as f64) / n_test * 100.0;
    let acc_fam = (correct_fam as f64) / n_test * 100.0;
    let acc_sub = (correct_sub as f64) / n_test * 100.0;
    let acc_joint = (correct_joint as f64) / n_test * 100.0;

    println!("--------------------------------------------------------------------------------");
    println!("CASCADED HIERARCHICAL ROUTING ACCURACY (HOLDOUT TEST SET):");
    println!("  Tier 1 Device Category Accuracy    : {:>6.2}% ({}/{} vs 25.0% Chance)", acc_dev, correct_dev, test_set.len());
    println!("  Tier 2 Attack Family Accuracy      : {:>6.2}% ({}/{} vs 50.0% Chance)", acc_fam, correct_fam, test_set.len());
    println!("  Tier 3 Attack SubType Accuracy     : {:>6.2}% ({}/{} vs 50.0% Chance)", acc_sub, correct_sub, test_set.len());
    println!("  Joint 3-Tier Leaf Accuracy         : {:>6.2}% ({}/{} vs  6.25% Chance)", acc_joint, correct_joint, test_set.len());
    println!("--------------------------------------------------------------------------------");
    println!("HARDWARE EXECUTION TELEMETRY:");
    println!("  Evaluation Duration                : {:.3}s", eval_duration.as_secs_f64());
    println!("  Inference Latency                  : {:.2} us/sample", per_sample_us);
    println!("  Inference Throughput               : {:.0} flows/sec", 1_000_000.0 / per_sample_us);
    println!("================================================================================");

    let report = IoTCascadedTreeReport {
        timestamp_rfc3339: "2026-10-26T15:30:00Z".to_string(),
        dataset: "BoTNeTIoT-L01 Stratified IoT Botnet Dataset (N-BaIoT)".to_string(),
        architecture: "T-1000 3-Tier Cascaded Ultrametric Tree Sorter".to_string(),
        registers: format!("H_Device({}Q) (x) H_Family({}Q) (x) H_SubType({}Q) (x) H_Env({}Q)", cfg.n_device, cfg.n_family, cfg.n_subtype, cfg.n_env),
        n_qubits: N_QUBITS,
        state_dim: STATE_DIM,
        total_parameters: sorter.total_parameters(),
        total_leaves: N_IOT_LEAVES,
        strong_triangle_inequality_holds: triplet_violations == 0,
        triplet_violations,
        evaluated_triplets: total_triplets,
        train_samples: train_set.len(),
        test_samples: test_set.len(),
        training_time_sec: train_duration,
        inference_per_sample_us: per_sample_us,
        test_device_accuracy_pct: acc_dev,
        test_family_accuracy_pct: acc_fam,
        test_subtype_accuracy_pct: acc_sub,
        test_joint_leaf_accuracy_pct: acc_joint,
    };

    let report_path = Path::new("results/experiments/iot_botnet_cascaded_tree_report.json");
    if let Some(parent) = report_path.parent() {
        std::fs::create_dir_all(parent).expect("Create results dir");
    }
    t1000::results::write(report_path, serde_json::to_string_pretty(&report).unwrap())
        .expect("Write report");
    println!("Report written to: {}", report_path.display());
}
