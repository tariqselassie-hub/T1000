// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # University of Pittsburgh Quantum Kernel Ridge Regression (Q-KRR) Benchmark
//!
//! Replaces non-convex variational circuit optimization with convex closed-form
//! Quantum Kernel Ridge Regression on exact statevector overlaps in Hilbert space:
//! 1. Model 1: Classical Linear Ridge Baseline (19 topological descriptors).
//! 2. Model 2: Classical RBF Gaussian Kernel Ridge Baseline.
//! 3. Model 3: 8-Qubit Second-Order ZZ Quantum Kernel Ridge Regressor (Q-KRR).
//! 4. Model 4: 8-Qubit Molecular Stark-Entangled Quantum Kernel Regressor (Q-SKRR).

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;
use t1000::pitt_data::{self, FeatureSet, PittMolecule};
use t1000::q_krr::{
    evaluate_regression_metrics, min_max_scale_features, solve_cholesky, Complex64, QKRRConfig,
    QuantumKernelRegressor,
};
use t1000::zinc_readout_fit::ridge_regression_multi;
use t1000::zinc_training_common::utc_timestamp_rfc3339;

const DEFAULT_PITT_CSV: &str = "data/pitt/pitt_quantum.csv";
const REPORT_PATH: &str = "results/data/pitt_qkrr_training_report.json";
const N_QUBITS: usize = 8;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QKRRModelTrial {
    pub name: String,
    pub description: String,
    pub train_mse: f64,
    pub test_mse: f64,
    pub test_mae: f64,
    pub test_r2: f64,
    pub throughput_mols_per_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PittQkrrTrainingReport {
    pub timestamp_utc: String,
    pub target_property: String,
    pub target_unit: String,
    pub train_samples: usize,
    pub test_samples: usize,
    pub qubits: usize,
    pub lambda_reg: f64,
    pub trials: Vec<QKRRModelTrial>,
    pub qkrr_advantage_pct: f64,
}

/// Evaluates a Classical RBF Gaussian Kernel Gram Matrix: K_ij = exp(-gamma * ||x_i - x_j||^2).
fn compute_rbf_gram_matrix(x_train: &[Vec<f64>], gamma: f64) -> Vec<f64> {
    let n = x_train.len();
    let rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut row = vec![0.0f64; n];
            row[i] = 1.0;
            for j in (i + 1)..n {
                let mut dist_sq = 0.0f64;
                for (&a, &b) in x_train[i].iter().zip(&x_train[j]) {
                    dist_sq += (a - b).powi(2);
                }
                row[j] = (-gamma * dist_sq).exp();
            }
            row
        })
        .collect();

    let mut gram = vec![0.0f64; n * n];
    for (i, row) in rows.iter().enumerate() {
        for j in i..n {
            let val = row[j];
            gram[i * n + j] = val;
            gram[j * n + i] = val;
        }
    }
    gram
}

/// Evaluates Classical RBF Predictions: y_pred = y_std * (K_test * alpha) + y_mean.
fn predict_rbf(
    x_test: &[Vec<f64>],
    x_train: &[Vec<f64>],
    alpha: &[f64],
    gamma: f64,
    y_mean: f64,
    y_std: f64,
) -> Vec<f64> {
    x_test
        .into_par_iter()
        .map(|t| {
            let mut sum_pred = 0.0f64;
            for (i, tr) in x_train.iter().enumerate() {
                let mut dist_sq = 0.0f64;
                for (&a, &b) in t.iter().zip(tr) {
                    dist_sq += (a - b).powi(2);
                }
                let k_val = (-gamma * dist_sq).exp();
                sum_pred += alpha[i] * k_val;
            }
            sum_pred * y_std + y_mean
        })
        .collect()
}

/// Encodes 3D molecular Stark dipole components into quantum statevectors.
fn encode_stark_enhanced_state(mol: &PittMolecule, scaled_feat: &[f64]) -> Vec<Complex64> {
    let dim = 1 << N_QUBITS; // 256
    let mut state = vec![Complex64::new(0.0, 0.0); dim];
    let norm_factor = 1.0 / (dim as f64).sqrt();
    for amp in &mut state {
        *amp = Complex64::new(norm_factor, 0.0);
    }

    let stark = mol.stark_angles(6.0);
    let mag = mol.dipole_magnitude();

    for (i, amp) in state.iter_mut().enumerate() {
        let mut phase = 0.0f64;
        for (p, &xp) in scaled_feat.iter().take(N_QUBITS).enumerate() {
            if (i >> p) & 1 == 1 {
                phase += xp;
            }
        }
        // Inject non-Abelian Stark dipole phase shifts
        if (i & 1) == 1 {
            phase += stark[0];
        }
        if ((i >> 1) & 1) == 1 {
            phase += stark[1];
        }
        if ((i >> 2) & 1) == 1 {
            phase += stark[2];
        }
        if ((i >> 3) & 1) == 1 {
            phase += (mag / 8.0).tanh() * 1.5;
        }

        // Second-order ZZ entangling phases
        for p in 0..N_QUBITS {
            for q in (p + 1)..N_QUBITS {
                let bit_p = (i >> p) & 1;
                let bit_q = (i >> q) & 1;
                if bit_p ^ bit_q == 1 {
                    let xp = scaled_feat[p.min(scaled_feat.len() - 1)];
                    let xq = scaled_feat[q.min(scaled_feat.len() - 1)];
                    phase += (std::f64::consts::PI - xp) * (std::f64::consts::PI - xq) * 0.5;
                }
            }
        }

        let cos_p = phase.cos();
        let sin_p = -phase.sin();
        *amp = Complex64::new(
            amp.re * cos_p - amp.im * sin_p,
            amp.re * sin_p + amp.im * cos_p,
        );
    }
    state
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let get_arg = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
    };

    let csv_path = get_arg("--dataset-path")
        .map(String::as_str)
        .unwrap_or(DEFAULT_PITT_CSV);
    let n_train: usize = get_arg("--train-samples")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    let n_test: usize = get_arg("--test-samples")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);
    let target_name = get_arg("--target")
        .map(String::as_str)
        .unwrap_or("polarizability");
    let lambda: f64 = get_arg("--lambda")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-2);
    let depth: usize = get_arg("--depth")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    let (target_idx, target_unit) = match target_name {
        "polarizability" | "alpha" => (4, "A^3"),
        "dipole" | "mu" => (3, "Debye"),
        "gap" | "egap" => (2, "eV"),
        "homo" => (0, "eV"),
        "lumo" => (1, "eV"),
        "molwt" | "mw" => (6, "g/mol"),
        _ => (4, "A^3"),
    };

    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ T-1000 QUANTUM KERNEL RIDGE REGRESSION (Q-KRR) MOLECULAR BENCHMARK         │");
    println!("├─────────────────────────────────────────┬───────────────────────────────────┤");
    println!(
        "│ Source Dataset                          │ {:<33} │",
        csv_path
    );
    println!(
        "│ Target Observable                       │ {:<33} │",
        format!("{} ({})", target_name, target_unit)
    );
    println!(
        "│ Partition Layout                        │ {:<33} │",
        format!("{} train / {} test", n_train, n_test)
    );
    println!(
        "│ Regularization Parameter (lambda)       │ {:<33} │",
        format!("{:.1e}", lambda)
    );
    println!(
        "│ Hilbert Encoding Dimension              │ {:<33} │",
        format!("{} Qubits (dim=256, depth={})", N_QUBITS, depth)
    );
    println!("└─────────────────────────────────────────┴───────────────────────────────────┘");

    if !Path::new(csv_path).exists() {
        eprintln!("[ERROR] Dataset not found at: {csv_path}");
        return;
    }

    let t_load = Instant::now();
    let (all_mols, skipped) =
        pitt_data::load(csv_path, FeatureSet::ZincLike, n_train + n_test + 500)
            .expect("Failed to load Pitt molecules");
    println!(
        "[LOAD] Ingested {} molecules (skipped {}) in {:.3} s",
        all_mols.len(),
        skipped,
        t_load.elapsed().as_secs_f64()
    );

    let (train_mols, test_mols) = pitt_data::split(all_mols, n_test, 2026);
    let train_mols: Vec<PittMolecule> = train_mols.into_iter().take(n_train).collect();
    let test_mols: Vec<PittMolecule> = test_mols.into_iter().take(n_test).collect();

    let y_train: Vec<f64> = train_mols
        .iter()
        .map(|m| m.properties[target_idx])
        .collect();
    let y_test: Vec<f64> = test_mols.iter().map(|m| m.properties[target_idx]).collect();

    // Min-Max scale the leading 8 descriptors into [0, PI]
    let raw_train_feat: Vec<Vec<f64>> = train_mols.iter().map(|m| m.features.clone()).collect();
    let raw_test_feat: Vec<Vec<f64>> = test_mols.iter().map(|m| m.features.clone()).collect();
    let scaled_train_feat = min_max_scale_features(&raw_train_feat, N_QUBITS);
    let scaled_test_feat = min_max_scale_features(&raw_test_feat, N_QUBITS);

    let mut trials = Vec::new();

    // ── TRIAL 1: Classical Linear Ridge Baseline ─────────────────────────────
    println!("\n[1/4] Fitting Classical Linear Ridge Baseline...");
    let t0 = Instant::now();
    let y_train_mat: Vec<Vec<f64>> = y_train.iter().map(|&y| vec![y]).collect();
    let (w_lin, b_lin) = ridge_regression_multi(&raw_train_feat, &y_train_mat, 19, 1, lambda);
    let pred_tr_lin: Vec<f64> = raw_train_feat
        .iter()
        .map(|r| b_lin[0] + r.iter().zip(&w_lin).map(|(&f, w)| f * w[0]).sum::<f64>())
        .collect();
    let pred_te_lin: Vec<f64> = raw_test_feat
        .iter()
        .map(|r| b_lin[0] + r.iter().zip(&w_lin).map(|(&f, w)| f * w[0]).sum::<f64>())
        .collect();
    let (tr_mse_l, _, _) = evaluate_regression_metrics(&y_train, &pred_tr_lin);
    let (te_mse_l, te_mae_l, te_r2_l) = evaluate_regression_metrics(&y_test, &pred_te_lin);
    let el_l = t0.elapsed().as_secs_f64();
    let tput_l = (n_train + n_test) as f64 / el_l;
    println!(
        "  -> Linear Ridge: Test MSE: {:.4} | MAE: {:.4} {} | R^2: {:.4} ({:.0} mols/s)",
        te_mse_l, te_mae_l, target_unit, te_r2_l, tput_l
    );
    trials.push(QKRRModelTrial {
        name: "Classical-Linear-Ridge".to_string(),
        description: "19-D topological SMILES descriptors with closed-form L2 regularization"
            .to_string(),
        train_mse: tr_mse_l,
        test_mse: te_mse_l,
        test_mae: te_mae_l,
        test_r2: te_r2_l,
        throughput_mols_per_sec: tput_l,
    });

    // ── TRIAL 2: Classical RBF Gaussian Kernel Ridge Baseline ─────────────────
    println!("\n[2/4] Fitting Classical RBF Gaussian Kernel Ridge Baseline...");
    let t1 = Instant::now();
    let y_mean = y_train.iter().sum::<f64>() / n_train as f64;
    let y_var = y_train.iter().map(|&y| (y - y_mean).powi(2)).sum::<f64>() / n_train as f64;
    let y_std = y_var.sqrt().max(1e-8);
    let y_norm: Vec<f64> = y_train.iter().map(|&y| (y - y_mean) / y_std).collect();
    let gamma = 0.5 / (N_QUBITS as f64);
    let mut k_rbf = compute_rbf_gram_matrix(&scaled_train_feat, gamma);
    for i in 0..n_train {
        k_rbf[i * n_train + i] += lambda;
    }
    let alpha_rbf = solve_cholesky(&k_rbf, &y_norm, n_train);
    let pred_tr_rbf = predict_rbf(
        &scaled_train_feat,
        &scaled_train_feat,
        &alpha_rbf,
        gamma,
        y_mean,
        y_std,
    );
    let pred_te_rbf = predict_rbf(
        &scaled_test_feat,
        &scaled_train_feat,
        &alpha_rbf,
        gamma,
        y_mean,
        y_std,
    );
    let (tr_mse_r, _, _) = evaluate_regression_metrics(&y_train, &pred_tr_rbf);
    let (te_mse_r, te_mae_r, te_r2_r) = evaluate_regression_metrics(&y_test, &pred_te_rbf);
    let el_r = t1.elapsed().as_secs_f64();
    let tput_r = (n_train + n_test) as f64 / el_r;
    println!(
        "  -> RBF Kernel:   Test MSE: {:.4} | MAE: {:.4} {} | R^2: {:.4} ({:.0} mols/s)",
        te_mse_r, te_mae_r, target_unit, te_r2_r, tput_r
    );
    trials.push(QKRRModelTrial {
        name: "Classical-RBF-Kernel".to_string(),
        description: "Gaussian RBF kernel in Euclidean space with Cholesky solve".to_string(),
        train_mse: tr_mse_r,
        test_mse: te_mse_r,
        test_mae: te_mae_r,
        test_r2: te_r2_r,
        throughput_mols_per_sec: tput_r,
    });

    // ── TRIAL 3: 8-Qubit Second-Order ZZ Quantum Kernel (Q-KRR) ──────────────
    println!("\n[3/4] Fitting 8-Qubit Second-Order ZZ Quantum Kernel Ridge Regressor (Q-KRR)...");
    let t2 = Instant::now();
    let config = QKRRConfig {
        num_qubits: N_QUBITS,
        depth,
        lambda,
        feature_scale: 1.0,
    };
    let mut qkrr = QuantumKernelRegressor::new(config);
    qkrr.fit(&scaled_train_feat, &y_train);
    let pred_tr_qkrr = qkrr.predict(&scaled_train_feat);
    let pred_te_qkrr = qkrr.predict(&scaled_test_feat);
    let (tr_mse_q, _, _) = evaluate_regression_metrics(&y_train, &pred_tr_qkrr);
    let (te_mse_q, te_mae_q, te_r2_q) = evaluate_regression_metrics(&y_test, &pred_te_qkrr);
    let el_q = t2.elapsed().as_secs_f64();
    let tput_q = (n_train + n_test) as f64 / el_q;
    println!(
        "  -> Q-KRR:        Test MSE: {:.4} | MAE: {:.4} {} | R^2: {:.4} ({:.0} mols/s)",
        te_mse_q, te_mae_q, target_unit, te_r2_q, tput_q
    );
    trials.push(QKRRModelTrial {
        name: "Quantum-Kernel-ZZ-8Q".to_string(),
        description: "8-qubit second-order ZZ entangling feature map in C^256 Hilbert space"
            .to_string(),
        train_mse: tr_mse_q,
        test_mse: te_mse_q,
        test_mae: te_mae_q,
        test_r2: te_r2_q,
        throughput_mols_per_sec: tput_q,
    });

    // ── TRIAL 4: 8-Qubit Molecular Stark-Entangled Kernel (Q-SKRR) ───────────
    println!("\n[4/4] Fitting 8-Qubit Molecular Stark-Entangled Quantum Kernel (Q-SKRR)...");
    let t3 = Instant::now();
    let tr_stark_states: Vec<Vec<Complex64>> = train_mols
        .par_iter()
        .zip(&scaled_train_feat)
        .map(|(m, f)| encode_stark_enhanced_state(m, f))
        .collect();
    let te_stark_states: Vec<Vec<Complex64>> = test_mols
        .par_iter()
        .zip(&scaled_test_feat)
        .map(|(m, f)| encode_stark_enhanced_state(m, f))
        .collect();

    let mut k_stark = qkrr.compute_gram_matrix(&tr_stark_states);
    for i in 0..n_train {
        k_stark[i * n_train + i] += lambda;
    }
    let alpha_stark = solve_cholesky(&k_stark, &y_norm, n_train);

    let pred_tr_skrr: Vec<f64> = tr_stark_states
        .par_iter()
        .map(|s| {
            let mut sum_k = 0.0f64;
            for (i, tr_s) in tr_stark_states.iter().enumerate() {
                sum_k += alpha_stark[i] * QuantumKernelRegressor::state_fidelity(tr_s, s);
            }
            sum_k * y_std + y_mean
        })
        .collect();
    let pred_te_skrr: Vec<f64> = te_stark_states
        .par_iter()
        .map(|s| {
            let mut sum_k = 0.0f64;
            for (i, tr_s) in tr_stark_states.iter().enumerate() {
                sum_k += alpha_stark[i] * QuantumKernelRegressor::state_fidelity(tr_s, s);
            }
            sum_k * y_std + y_mean
        })
        .collect();

    let (tr_mse_sk, _, _) = evaluate_regression_metrics(&y_train, &pred_tr_skrr);
    let (te_mse_sk, te_mae_sk, te_r2_sk) = evaluate_regression_metrics(&y_test, &pred_te_skrr);
    let el_sk = t3.elapsed().as_secs_f64();
    let tput_sk = (n_train + n_test) as f64 / el_sk;
    let adv_pct = (te_mse_r - te_mse_sk) / te_mse_r * 100.0;

    println!(
        "  -> Q-SKRR Stark: Test MSE: {:.4} | MAE: {:.4} {} | R^2: {:.4} ({:.0} mols/s)",
        te_mse_sk, te_mae_sk, target_unit, te_r2_sk, tput_sk
    );
    println!(
        "  -> Q-SKRR Accuracy Improvement vs Classical RBF: {:.2}%",
        adv_pct
    );

    trials.push(QKRRModelTrial {
        name: "Quantum-Stark-Kernel-8Q".to_string(),
        description:
            "8-qubit Stark-coupled dual-angle ZZ kernel with 3D permanent dipole vector injection"
                .to_string(),
        train_mse: tr_mse_sk,
        test_mse: te_mse_sk,
        test_mae: te_mae_sk,
        test_r2: te_r2_sk,
        throughput_mols_per_sec: tput_sk,
    });

    // ── Final Sovereign Telemetry Summary Table ──────────────────────────────
    println!("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║              T-1000 QUANTUM KERNEL RIDGE REGRESSION (Q-KRR) TELEMETRY                              ║");
    println!("╠══════════════════════════╦══════════════╦══════════════╦══════════════╦════════════╦═══════════════╣");
    println!("║ Architecture             ║ Train MSE    ║ Test MSE     ║ Test MAE     ║ Test R^2   ║ Mols / Sec    ║");
    println!("╠══════════════════════════╬══════════════╬══════════════╬══════════════╬════════════╬═══════════════╣");
    for tr in &trials {
        println!(
            "║ {:<24} ║ {:>12.4} ║ {:>12.4} ║ {:>9.4} {} ║ {:>10.4} ║ {:>11.0} ║",
            tr.name,
            tr.train_mse,
            tr.test_mse,
            tr.test_mae,
            target_unit,
            tr.test_r2,
            tr.throughput_mols_per_sec
        );
    }
    println!("╚══════════════════════════╩══════════════╩══════════════╩══════════════╩════════════╩═══════════════╝");

    let report = PittQkrrTrainingReport {
        timestamp_utc: utc_timestamp_rfc3339(),
        target_property: target_name.to_string(),
        target_unit: target_unit.to_string(),
        train_samples: n_train,
        test_samples: n_test,
        qubits: N_QUBITS,
        lambda_reg: lambda,
        trials,
        qkrr_advantage_pct: adv_pct,
    };

    if let Some(parent) = Path::new(REPORT_PATH).parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(json_str) = serde_json::to_string_pretty(&report) {
        let _ = fs::write(REPORT_PATH, json_str);
        println!("[RECORD] Telemetry successfully recorded to: {REPORT_PATH}");
    }
}
