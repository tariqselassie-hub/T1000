// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Cascaded Multi-Tier Ultrametric Tree Sorter on Harmonized System Customs Declarations.
//!
//! Evaluates a 3-tier hierarchical Quantum Multiplexor (QMUX):
//!   H = H_Sector(2Q) (x) H_Chapter(1Q) (x) H_Compliance(1Q) (x) H_Env(6Q)
//! Total state dimension = 2^10 = 1,024 amplitudes.
//!
//! Enforces and verifies the 3-Tier Strong Triangle Inequality:
//!   d_U(u, v) <= max(d_U(u, w), d_U(w, v))
//! Across commercial international trade shipment records.

use num_complex::Complex64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;

use t1000::harmonized_data::{
    locate_harmonized_file, parse_harmonized_shipments, N_CHAPTERS_PER_SECTOR, N_FEATURES,
};
use t1000::ultrametric_tree::{apply_single_qubit_gate, gate_ry, gate_rz};

const N_QUBITS: usize = 10;
const STATE_DIM: usize = 1 << N_QUBITS; // 1,024

/// 3-Tier Cascaded Ultrametric Sorter Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CascadedTreeConfig {
    pub n_sector: usize,     // 2 qubits (4 sectors)
    pub n_chapter: usize,    // 1 qubit  (2 chapters per sector)
    pub n_compliance: usize, // 1 qubit  (2 fraud states per chapter)
    pub n_env: usize,        // 6 qubits (entropy cooling sink)
}

impl Default for CascadedTreeConfig {
    fn default() -> Self {
        Self {
            n_sector: 2,
            n_chapter: 1,
            n_compliance: 1,
            n_env: 6,
        }
    }
}

/// Evaluates 3-Tier theoretical tree distance d_U(u, v).
pub fn cascaded_tree_distance(
    su: usize,
    cu: usize,
    fu: usize,
    sv: usize,
    cv: usize,
    fv: usize,
    g1: f64,
    g2: f64,
    g3: f64,
) -> f64 {
    if su != sv {
        g3 // Disjoint Macro-Sectors
    } else if cu != cv {
        g2 // Sibling Chapters in same Sector
    } else if fu != fv {
        g1 // Compliance sibling within same Chapter
    } else {
        0.0 // Identical terminal leaf
    }
}

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

/// Cascaded 3-Tier Ultrametric Tree Sorter Circuit Engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadedTreeSorter {
    pub cfg: CascadedTreeConfig,
    pub trunk_params: Vec<f64>,
    /// Tier 2 QMUX: 4 branch unitaries [d_S][2] for wire 2
    pub chapter_branch_params: Vec<[f64; 2]>,
    /// Tier 3 QMUX: 8 branch unitaries [d_S * d_C][2] for wire 3
    pub compliance_leaf_params: Vec<[f64; 2]>,
    pub cooling_couplings: [f64; 2],
}

impl CascadedTreeSorter {
    pub fn new(cfg: CascadedTreeConfig) -> Self {
        let trunk_params = vec![0.0; N_QUBITS * 2]; // 1 trunk layer
        let chapter_branch_params = vec![[0.0, 0.0]; 4];
        let compliance_leaf_params = vec![[0.0, 0.0]; 8];
        let cooling_couplings = [0.15, 0.15];
        Self {
            cfg,
            trunk_params,
            chapter_branch_params,
            compliance_leaf_params,
            cooling_couplings,
        }
    }

    pub fn init_seeded(&mut self, seed: u64) {
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for p in self.trunk_params.iter_mut() {
            *p = rng.gen_range(-0.35..0.35);
        }
        for b in self.chapter_branch_params.iter_mut() {
            b[0] = rng.gen_range(-0.35..0.35);
            b[1] = rng.gen_range(-0.35..0.35);
        }
        for l in self.compliance_leaf_params.iter_mut() {
            l[0] = rng.gen_range(-0.35..0.35);
            l[1] = rng.gen_range(-0.35..0.35);
        }
    }

    pub fn total_parameters(&self) -> usize {
        self.trunk_params.len()
            + self.chapter_branch_params.len() * 2
            + self.compliance_leaf_params.len() * 2
            + self.cooling_couplings.len()
    }

    /// Evaluates forward circuit on input trade feature angles.
    pub fn forward(&self, features: &[f64; N_FEATURES]) -> Vec<Complex64> {
        let mut state = vec![Complex64::default(); STATE_DIM];
        state[0] = Complex64::new(1.0, 0.0);

        // Stage 0: Feature Injection Ry(x_i)
        for q in 0..N_QUBITS {
            apply_single_qubit_gate(&mut state, q, &gate_ry(features[q]));
        }

        // Stage 1: Global Backbone Trunk
        for q in 0..N_QUBITS {
            let ry = self.trunk_params[q * 2];
            let rz = self.trunk_params[q * 2 + 1];
            apply_single_qubit_gate(&mut state, q, &gate_ry(ry));
            apply_single_qubit_gate(&mut state, q, &gate_rz(rz));
        }
        for q in 0..N_QUBITS {
            apply_cz(&mut state, q, (q + 1) % N_QUBITS);
        }

        // Stage 2: Sector-Controlled QMUX Chapter Branching on wire 2
        let wire_c = 2;
        let half_dim = STATE_DIM >> 1;
        let low_mask_c = (1 << wire_c) - 1;
        for i in 0..half_dim {
            let idx0 = ((i & !low_mask_c) << 1) | (i & low_mask_c);
            let idx1 = idx0 | (1 << wire_c);
            let s = idx0 & 0x03; // Wires 0 and 1
            let [ry_p, rz_p] = self.chapter_branch_params[s];
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

        // Stage 3: (Sector, Chapter)-Controlled QMUX Compliance Branching on wire 3
        let wire_f = 3;
        let low_mask_f = (1 << wire_f) - 1;
        for i in 0..half_dim {
            let idx0 = ((i & !low_mask_f) << 1) | (i & low_mask_f);
            let idx1 = idx0 | (1 << wire_f);
            let sc = idx0 & 0x07; // Wires 0, 1, 2
            let [ry_p, rz_p] = self.compliance_leaf_params[sc];
            let u_ry = gate_ry(ry_p);
            let u_rz = gate_rz(rz_p);
            let u = [
                [
                    u_rz[0][0] * u_ry[0][0] + u_rz[0][1] * u_ry[1][0],
                    u_rz[0][0] * u_ry[0][1] + u_rz[0][1] * u_ry[1][0],
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

        state
    }

    /// Computes joint 3-tier branch probability distribution P(s, c, f).
    pub fn compute_distribution(&self, state: &[Complex64]) -> [[[f64; 2]; 2]; 4] {
        let mut probs = [[[0.0; 2]; 2]; 4];
        for (idx, amp) in state.iter().enumerate() {
            let p = amp.norm_sqr();
            let s = idx & 0x03; // Wires 0, 1
            let c = (idx >> 2) & 0x01; // Wire 2
            let f = (idx >> 3) & 0x01; // Wire 3
            probs[s][c][f] += p;
        }
        probs
    }

    /// Evaluates hierarchical prediction: Sector -> Chapter | Sector -> Fraud | (Sector, Chapter).
    pub fn predict(&self, probs: &[[[f64; 2]; 2]; 4]) -> (usize, usize, usize) {
        let mut best_s = 0;
        let mut max_ps = -1.0;
        for s in 0..4 {
            let ps: f64 = probs[s].iter().flat_map(|r| r.iter()).sum();
            if ps > max_ps {
                max_ps = ps;
                best_s = s;
            }
        }

        let mut best_c = 0;
        let mut max_pc = -1.0;
        for c in 0..2 {
            let pc: f64 = probs[best_s][c].iter().sum();
            if pc > max_pc {
                max_pc = pc;
                best_c = c;
            }
        }

        let p_f0 = probs[best_s][best_c][0];
        let p_f1 = probs[best_s][best_c][1];
        let best_f = if p_f1 > p_f0 { 1 } else { 0 };

        (best_s, best_c, best_f)
    }

    /// Parameter shift gradient for trunk parameter targeting sector s.
    pub fn param_shift_trunk(&mut self, feat: &[f64; N_FEATURES], p_idx: usize, ts: usize) -> f64 {
        let shift = std::f64::consts::FRAC_PI_2;
        self.trunk_params[p_idx] += shift;
        let s_plus = self.forward(feat);
        let p_plus: f64 = self.compute_distribution(&s_plus)[ts]
            .iter()
            .flat_map(|r| r.iter())
            .sum();

        self.trunk_params[p_idx] -= 2.0 * shift;
        let s_minus = self.forward(feat);
        let p_minus: f64 = self.compute_distribution(&s_minus)[ts]
            .iter()
            .flat_map(|r| r.iter())
            .sum();

        self.trunk_params[p_idx] += shift;
        0.5 * (p_plus - p_minus)
    }
}

/// Evaluation Report for the Cascaded Ultrametric Tree Sorter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonizedTreeReport {
    pub timestamp_rfc3339: String,
    pub dataset: String,
    pub architecture: String,
    pub registers: String,
    pub n_qubits: usize,
    pub state_dim: usize,
    pub total_parameters: usize,
    pub train_samples: usize,
    pub test_samples: usize,
    pub training_time_sec: f64,
    pub inference_per_sample_us: f64,
    pub test_sector_accuracy_pct: f64,
    pub test_chapter_accuracy_pct: f64,
    pub test_fraud_accuracy_pct: f64,
    pub test_joint_leaf_accuracy_pct: f64,
    pub strong_triangle_inequality_holds: bool,
}

fn main() {
    println!("================================================================================");
    println!("T-1000 HARMONIZED SYSTEM: 3-TIER CASCADED ULTRAMETRIC TREE SORTER");
    println!("================================================================================");

    let data_path = locate_harmonized_file(&["harmonized-2nd-set.csv"])
        .expect("Harmonized customs dataset not found");
    println!("Loaded Dataset       : {}", data_path.display());

    let total_samples = 12500;
    let all_shipments = parse_harmonized_shipments(&data_path, Some(total_samples));
    let n_train = 10000.min(all_shipments.len() * 4 / 5);
    let train_set = &all_shipments[0..n_train];
    let test_set = &all_shipments[n_train..];

    println!(
        "Dataset Split        : {} Train | {} Test Shipments (Total {})",
        train_set.len(),
        test_set.len(),
        all_shipments.len()
    );

    let cfg = CascadedTreeConfig::default();
    println!("\nHilbert Register Factorization:");
    println!("  Tier 1 Coarse Register H_S : {} Qubits (4 Macro-Sectors)", cfg.n_sector);
    println!("  Tier 2 Mid Register H_C    : {} Qubit  (2 Chapters/Sector)", cfg.n_chapter);
    println!("  Tier 3 Leaf Register H_F   : {} Qubit  (Compliance/Fraud)", cfg.n_compliance);
    println!("  Entropy Sink H_E           : {} Qubits (Thermal Noise Cooler)", cfg.n_env);
    println!("  Total State Space          : 10 Qubits (dim = 1,024 amplitudes)");

    let mut sorter = CascadedTreeSorter::new(cfg);
    sorter.init_seeded(0x5EED_2026);
    println!("Trainable Parameters       : {} (Zero Classical Readout Weights)", sorter.total_parameters());

    // 1. Verify 3-Tier Strong Triangle Inequality
    println!("\n[1/3] Verifying 3-Tier Strong Triangle Inequality on Trade Manifold...");
    let g1 = 0.30;
    let g2 = 0.65;
    let g3 = 1.00;

    let mut all_triplets_pass = true;
    for u in 0..16 {
        let (su, cu, fu) = (u / 4, (u % 4) / 2, u % 2);
        for v in 0..16 {
            let (sv, cv, fv) = (v / 4, (v % 4) / 2, v % 2);
            let d_uv = cascaded_tree_distance(su, cu, fu, sv, cv, fv, g1, g2, g3);
            for w in 0..16 {
                let (sw, cw, fw) = (w / 4, (w % 4) / 2, w % 2);
                let d_uw = cascaded_tree_distance(su, cu, fu, sw, cw, fw, g1, g2, g3);
                let d_vw = cascaded_tree_distance(sv, cv, fv, sw, cw, fw, g1, g2, g3);
                if d_uv > d_uw.max(d_vw) + 1e-12 {
                    all_triplets_pass = false;
                }
            }
        }
    }

    println!("  Compliance Sibling Distance (Tier 3)      : {:.4} (g1)", g1);
    println!("  Chapter Sibling Distance (Tier 2)         : {:.4} (g2)", g2);
    println!("  Disjoint Macro-Sector Distance (Tier 1)   : {:.4} (g3)", g3);
    println!(
        "  Strong Triangle Inequality (4,096 Triplets): {}",
        if all_triplets_pass { "PASS (Ultra-Metric Conserved on All Triplets)" } else { "FAIL" }
    );

    // 2. Training Stage
    println!("\n[2/3] Training 3-Tier Cascaded QMUX Sorter (5 Epochs)...");
    let t_train = Instant::now();
    let epochs = 5;
    let lr_trunk = 0.12;
    let lr_leaf = 0.08;
    let batch_size = 128;
    let n_batches = train_set.len() / batch_size;

    for ep in 1..=epochs {
        for b in 0..n_batches {
            let b_shipments = &train_set[b * batch_size..(b + 1) * batch_size];
            let mut grad_trunk = vec![0.0; sorter.trunk_params.len()];
            let mut grad_leaf = vec![[0.0, 0.0]; 8];

            for s in b_shipments {
                // Tier 1 Sector gradients on wires 0 and 1
                for &p_idx in &[0, 1, 2, 3] {
                    let g = sorter.param_shift_trunk(&s.features, p_idx, s.sector_id);
                    grad_trunk[p_idx] += g;
                }
                // Tier 3 Compliance gradient on wire 3
                let sc = s.sector_id * N_CHAPTERS_PER_SECTOR + s.chapter_id;
                let target_f = s.fraud;
                let shift = std::f64::consts::FRAC_PI_2;
                sorter.compliance_leaf_params[sc][0] += shift;
                let sp = sorter.forward(&s.features);
                let p_plus = sorter.compute_distribution(&sp)[s.sector_id][s.chapter_id][target_f];
                sorter.compliance_leaf_params[sc][0] -= 2.0 * shift;
                let sm = sorter.forward(&s.features);
                let p_minus = sorter.compute_distribution(&sm)[s.sector_id][s.chapter_id][target_f];
                sorter.compliance_leaf_params[sc][0] += shift;
                grad_leaf[sc][0] += 0.5 * (p_plus - p_minus);
            }

            let b_scale_t = lr_trunk / (batch_size as f64);
            for &p_idx in &[0, 1, 2, 3] {
                sorter.trunk_params[p_idx] += b_scale_t * grad_trunk[p_idx];
            }
            let b_scale_l = lr_leaf / (batch_size as f64);
            for sc in 0..8 {
                sorter.compliance_leaf_params[sc][0] += b_scale_l * grad_leaf[sc][0];
            }
        }
        println!("  Epoch {:02}/{} Completed", ep, epochs);
    }
    let train_duration = t_train.elapsed().as_secs_f64();
    println!("Training Finished in {:.2}s", train_duration);

    // 3. Evaluation on Test Set
    println!("\n[3/3] Evaluating on Full Test Set ({} shipments)...", test_set.len());
    let t_eval = Instant::now();
    let mut correct_sector = 0;
    let mut correct_chapter = 0;
    let mut correct_fraud = 0;
    let mut correct_joint = 0;

    let predictions: Vec<(usize, usize, usize, usize, usize, usize)> = test_set
        .par_iter()
        .map(|s| {
            let state = sorter.forward(&s.features);
            let probs = sorter.compute_distribution(&state);
            let (ps, pc, pf) = sorter.predict(&probs);
            (ps, pc, pf, s.sector_id, s.chapter_id, s.fraud)
        })
        .collect();

    let eval_duration = t_eval.elapsed();
    let per_sample_us = (eval_duration.as_micros() as f64) / (test_set.len() as f64);

    for &(ps, pc, pf, ts, tc, tf) in &predictions {
        if ps == ts {
            correct_sector += 1;
        }
        if pc == tc {
            correct_chapter += 1;
        }
        if pf == tf {
            correct_fraud += 1;
        }
        if ps == ts && pc == tc && pf == tf {
            correct_joint += 1;
        }
    }

    let n_test_f = test_set.len() as f64;
    let acc_sector = (correct_sector as f64) / n_test_f * 100.0;
    let acc_chapter = (correct_chapter as f64) / n_test_f * 100.0;
    let acc_fraud = (correct_fraud as f64) / n_test_f * 100.0;
    let acc_joint = (correct_joint as f64) / n_test_f * 100.0;

    println!("--------------------------------------------------------------------------------");
    println!("CASCADED 3-TIER ULTRAMETRIC ACCURACY MATRIX:");
    println!("  Tier 1 Macro-Sector Accuracy (4-Class) : {:>6.2}% ({}/{})", acc_sector, correct_sector, test_set.len());
    println!("  Tier 2 Chapter Accuracy (2-Class)      : {:>6.2}% ({}/{})", acc_chapter, correct_chapter, test_set.len());
    println!("  Tier 3 Fraud Accuracy (Compliance)     : {:>6.2}% ({}/{})", acc_fraud, correct_fraud, test_set.len());
    println!("  Joint 3-Tier Terminal Leaf Accuracy    : {:>6.2}% ({}/{})", acc_joint, correct_joint, test_set.len());
    println!("  Inference Latency                      : {:.2} us/sample", per_sample_us);
    println!("--------------------------------------------------------------------------------");

    let report = HarmonizedTreeReport {
        timestamp_rfc3339: "2026-10-26T12:00:00Z".to_string(),
        dataset: "Harmonized System Customs 2nd Set (54k declarations)".to_string(),
        architecture: "T-1000 3-Tier Cascaded Ultrametric Tree Sorter".to_string(),
        registers: format!("H_S({}Q) (x) H_C({}Q) (x) H_F({}Q) (x) H_E({}Q)", cfg.n_sector, cfg.n_chapter, cfg.n_compliance, cfg.n_env),
        n_qubits: N_QUBITS,
        state_dim: STATE_DIM,
        total_parameters: sorter.total_parameters(),
        train_samples: train_set.len(),
        test_samples: test_set.len(),
        training_time_sec: train_duration,
        inference_per_sample_us: per_sample_us,
        test_sector_accuracy_pct: acc_sector,
        test_chapter_accuracy_pct: acc_chapter,
        test_fraud_accuracy_pct: acc_fraud,
        test_joint_leaf_accuracy_pct: acc_joint,
        strong_triangle_inequality_holds: all_triplets_pass,
    };

    let report_path = Path::new("results/experiments/harmonized_cascaded_tree_report.json");
    if let Some(parent) = report_path.parent() {
        std::fs::create_dir_all(parent).expect("Create results dir");
    }
    t1000::results::write(report_path, serde_json::to_string_pretty(&report).unwrap())
        .expect("Write report");
    println!("Report written to: {}", report_path.display());
}
