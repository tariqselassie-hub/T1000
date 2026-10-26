// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # University of Pittsburgh Quantum VQC Training & Molecular Stark Dipole Benchmark
//!
//! Trains and compares three predictive regimes across the 106k Pitt Quantum Repository:
//! 1. Model A: Classical Ridge Baseline (19-D topological SMILES descriptors).
//! 2. Model B: 8-Qubit Standard VQC (Topological angle encoding, zero Stark coupling).
//! 3. Model C: 8-Qubit Molecular Stark-Coupled VQC (Dual-angle topological + 3D dipole phase injection).

use serde::{Deserialize, Serialize};
use std::env;
use std::path::Path;
use std::time::Instant;
use t1000::pitt_data::{self, FeatureSet, PittMolecule};
use t1000::results::{Better, Score};
use t1000::simd::{apply_cz_kernel, apply_ry_kernel, apply_rz_kernel, measure_single_qubit_z};
use t1000::zinc_readout_fit::ridge_regression_multi;
use t1000::zinc_training_common::utc_timestamp_rfc3339;

const DEFAULT_PITT_CSV: &str = "data/pitt/pitt_quantum.csv";
const REPORT_PATH: &str = "results/data/pitt_vqc_training_report.json";
const N_QUBITS: usize = 8;
const N_LAYERS: usize = 3;
const DIM: usize = 1 << N_QUBITS; // 256 amplitudes
/// Index of the dipole magnitude in `PittMolecule::properties`.
const DIPOLE_IDX: usize = 3;

/// Summary report for a single evaluation model trial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PittModelTrial {
    pub name: String,
    pub description: String,
    pub train_mse: f64,
    pub test_mse: f64,
    pub test_mae: f64,
    pub test_r2: f64,
    pub throughput_mols_per_sec: f64,
    /// The model's input encodes the target itself, so its test error is not a prediction.
    pub uses_target_as_input: bool,
}

/// Comprehensive benchmark report for the Pitt VQC Stark training experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PittVqcTrainingReport {
    pub timestamp_utc: String,
    pub target_property: String,
    pub target_unit: String,
    pub train_samples: usize,
    pub test_samples: usize,
    pub qubits: usize,
    pub layers: usize,
    pub trials: Vec<PittModelTrial>,
    /// Stark-VQC test-MSE reduction over the standard VQC; `None` when Stark-VQC reads the target.
    pub stark_advantage_pct: Option<f64>,
}

/// Parameterized 8-qubit variational circuit with optional molecular Stark phase coupling.
#[derive(Debug, Clone)]
struct PittVqcModel {
    pub weights_ry: Vec<Vec<f64>>, // [layer][qubit]
    pub weights_rz: Vec<Vec<f64>>, // [layer][qubit]
}

impl PittVqcModel {
    pub fn new(seed: u64) -> Self {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let f = (z ^ (z >> 31)) as f64 / u64::MAX as f64;
            (f - 0.5) * 0.4
        };

        let mut weights_ry = Vec::with_capacity(N_LAYERS);
        let mut weights_rz = Vec::with_capacity(N_LAYERS);
        for _ in 0..N_LAYERS {
            weights_ry.push((0..N_QUBITS).map(|_| next()).collect());
            weights_rz.push((0..N_QUBITS).map(|_| next()).collect());
        }
        Self {
            weights_ry,
            weights_rz,
        }
    }

    /// Evaluates 16 quantum expectation observables: 8 local <Z_q> + 8 correlators <Z_q Z_{q+1}>.
    pub fn forward_observables(
        &self,
        thetas: &[f64; N_QUBITS],
        phis: &[f64; N_QUBITS],
    ) -> Vec<f64> {
        let mut re = [0.0f64; DIM];
        let mut im = [0.0f64; DIM];
        re[0] = 1.0; // Ground state |0...0>

        // 1. Dual-Angle State Preparation: Ry(theta) * Rz(phi)
        for q in 0..N_QUBITS {
            apply_ry_kernel(&mut re, &mut im, q, thetas[q]);
            if phis[q].abs() > 1e-12 {
                apply_rz_kernel(&mut re, &mut im, q, phis[q]);
            }
        }

        // 2. Layered Ansatz: Ry(w_y) + Rz(w_z) + Ring CZ Entanglers
        for l in 0..N_LAYERS {
            for q in 0..N_QUBITS {
                apply_ry_kernel(&mut re, &mut im, q, self.weights_ry[l][q]);
                apply_rz_kernel(&mut re, &mut im, q, self.weights_rz[l][q]);
            }
            for q in 0..N_QUBITS {
                let target = (q + 1) % N_QUBITS;
                apply_cz_kernel(&mut re, &mut im, q, target);
            }
        }

        // 3. Extract 8 Single-Qubit <Z_q> Observables
        let mut z_single = [0.0f64; N_QUBITS];
        for (q, item) in z_single.iter_mut().enumerate() {
            *item = measure_single_qubit_z(&re, &im, q);
        }

        // 4. Extract 8 Correlators <Z_q Z_{q+1}>
        let mut obs = Vec::with_capacity(2 * N_QUBITS);
        obs.extend_from_slice(&z_single);
        for q in 0..N_QUBITS {
            let next_q = (q + 1) % N_QUBITS;
            let mut zz = 0.0f64;
            for (idx, (&r, &i)) in re.iter().zip(im.iter()).enumerate() {
                let p = r * r + i * i;
                let b1 = (idx >> q) & 1;
                let b2 = (idx >> next_q) & 1;
                let sign = if b1 == b2 { 1.0 } else { -1.0 };
                zz += sign * p;
            }
            obs.push(zz);
        }
        obs
    }
}

/// Encodes 19 topological SMILES features into 8 primary rotation angles.
fn encode_topological_angles(feat: &[f64]) -> [f64; N_QUBITS] {
    let mut thetas = [0.0f64; N_QUBITS];
    for (q, item) in thetas.iter_mut().enumerate() {
        let f1 = feat.get(q).copied().unwrap_or(0.0);
        let f2 = feat.get(q + 8).copied().unwrap_or(0.0);
        *item = (f1 + 0.5 * f2).atan() * 0.8;
    }
    thetas
}

/// Encodes Stark dipole vector components into non-Abelian Rz phase shifts.
fn encode_stark_phases(mol: &PittMolecule, enabled: bool) -> [f64; N_QUBITS] {
    let mut phis = [0.0f64; N_QUBITS];
    if !enabled {
        return phis;
    }
    let stark = mol.stark_angles(6.0);
    phis[0] = stark[0]; // μ_x
    phis[1] = stark[1]; // μ_y
    phis[2] = stark[2]; // μ_z
    let mag = mol.dipole_magnitude();
    phis[3] = (mag / 8.0).tanh() * std::f64::consts::FRAC_PI_2;
    // Fill remaining qubits with higher-order cross-moments
    phis[4] = ((mol.dipole_vec[0] * mol.dipole_vec[1]) / 25.0).tanh() * 1.2;
    phis[5] = ((mol.dipole_vec[1] * mol.dipole_vec[2]) / 25.0).tanh() * 1.2;
    phis[6] = ((mol.dipole_vec[0] * mol.dipole_vec[2]) / 25.0).tanh() * 1.2;
    phis[7] = ((mag * mag) / 50.0).tanh() * 1.0;
    phis
}

fn evaluate_predictions(y_true: &[f64], y_pred: &[f64]) -> (f64, f64, f64) {
    let n = y_true.len();
    assert_eq!(n, y_pred.len());
    let mut sse = 0.0f64;
    let mut sae = 0.0f64;
    let mut sum_y = 0.0f64;
    for (&t, &p) in y_true.iter().zip(y_pred) {
        let diff = t - p;
        sse += diff * diff;
        sae += diff.abs();
        sum_y += t;
    }
    let mse = sse / n as f64;
    let mae = sae / n as f64;
    let mean_y = sum_y / n as f64;
    let mut sst = 0.0f64;
    for &t in y_true {
        sst += (t - mean_y).powi(2);
    }
    let r2 = if sst > 1e-12 { 1.0 - sse / sst } else { 0.0 };
    (mse, mae, r2)
}

fn predict_linear(x: &[Vec<f64>], weights: &[Vec<f64>], bias: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|row| {
            let mut val = bias[0];
            for (&f, w) in row.iter().zip(weights.iter().map(|w_row| w_row[0])) {
                val += f * w;
            }
            val
        })
        .collect()
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
        .unwrap_or(8000);
    let n_test: usize = get_arg("--test-samples")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    let target_name = get_arg("--target")
        .map(String::as_str)
        .unwrap_or("polarizability");

    let (target_idx, target_unit) = match target_name {
        "polarizability" | "alpha" => (4, "A^3"),
        "dipole" | "mu" => (DIPOLE_IDX, "Debye"),
        "gap" | "egap" => (2, "eV"),
        "homo" => (0, "eV"),
        "lumo" => (1, "eV"),
        "molwt" | "mw" => (6, "g/mol"),
        _ => (4, "A^3"),
    };

    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ T-1000 PITT QUANTUM REPOSITORY VQC & MOLECULAR STARK DIPOLE TRAINER         │");
    println!("├─────────────────────────────────────────┬───────────────────────────────────┤");
    println!(
        "│ Source Dataset                          │ {:<33} │",
        csv_path
    );
    println!(
        "│ Target Property                         │ {:<33} │",
        format!("{} ({})", target_name, target_unit)
    );
    println!(
        "│ Partition Layout                        │ {:<33} │",
        format!("{} train / {} test", n_train, n_test)
    );
    println!(
        "│ Hilbert Space Vectorization             │ {:<33} │",
        "8 Qubits (256 Amplitudes)"
    );
    println!("└─────────────────────────────────────────┴───────────────────────────────────┘");

    if !Path::new(csv_path).exists() {
        eprintln!("[ERROR] Dataset not found at: {csv_path}");
        return;
    }

    let t_load = Instant::now();
    let (all_mols, skipped) =
        pitt_data::load(csv_path, FeatureSet::ZincLike, n_train + n_test + 1000)
            .expect("Failed to load molecules from Pitt CSV table");
    println!(
        "[LOAD] Loaded {} molecules (skipped {}) in {:.3} s",
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
    let y_train_mat: Vec<Vec<f64>> = y_train.iter().map(|&y| vec![y]).collect();

    let mut trials = Vec::new();

    // ── TRIAL 1: Classical 19-D Topological Ridge Baseline ───────────────────
    println!("\n[1/3] Training Classical 19-Descriptor Ridge Baseline...");
    let t0 = Instant::now();
    let x_train_c: Vec<Vec<f64>> = train_mols.iter().map(|m| m.features.clone()).collect();
    let x_test_c: Vec<Vec<f64>> = test_mols.iter().map(|m| m.features.clone()).collect();
    let (w_c, b_c) = ridge_regression_multi(&x_train_c, &y_train_mat, 19, 1, 1e-4);
    let pred_train_c = predict_linear(&x_train_c, &w_c, &b_c);
    let pred_test_c = predict_linear(&x_test_c, &w_c, &b_c);
    let (tr_mse_c, _, _) = evaluate_predictions(&y_train, &pred_train_c);
    let (te_mse_c, te_mae_c, te_r2_c) = evaluate_predictions(&y_test, &pred_test_c);
    let el_c = t0.elapsed().as_secs_f64();
    let tput_c = (train_mols.len() + test_mols.len()) as f64 / el_c;

    println!(
        "  -> Test MSE: {:.4} | MAE: {:.4} {} | R^2: {:.4} ({:.0} mols/s)",
        te_mse_c, te_mae_c, target_unit, te_r2_c, tput_c
    );
    trials.push(PittModelTrial {
        name: "Classical-Ridge-19D".to_string(),
        description: "19 universal topological descriptors with closed-form ridge readout"
            .to_string(),
        train_mse: tr_mse_c,
        test_mse: te_mse_c,
        test_mae: te_mae_c,
        test_r2: te_r2_c,
        throughput_mols_per_sec: tput_c,
        uses_target_as_input: false,
    });

    // ── TRIAL 2: 8-Qubit Standard VQC (Zero Stark Coupling) ─────────────────
    println!("\n[2/3] Training 8-Qubit Standard VQC (Topological Angles, Zero Stark Coupling)...");
    let t1 = Instant::now();
    let vqc_std = PittVqcModel::new(42);
    let x_train_std: Vec<Vec<f64>> = train_mols
        .iter()
        .map(|m| {
            let th = encode_topological_angles(&m.features);
            let ph = [0.0f64; N_QUBITS];
            vqc_std.forward_observables(&th, &ph)
        })
        .collect();
    let x_test_std: Vec<Vec<f64>> = test_mols
        .iter()
        .map(|m| {
            let th = encode_topological_angles(&m.features);
            let ph = [0.0f64; N_QUBITS];
            vqc_std.forward_observables(&th, &ph)
        })
        .collect();
    let (w_std, b_std) = ridge_regression_multi(&x_train_std, &y_train_mat, 16, 1, 1e-4);
    let pred_train_std = predict_linear(&x_train_std, &w_std, &b_std);
    let pred_test_std = predict_linear(&x_test_std, &w_std, &b_std);
    let (tr_mse_std, _, _) = evaluate_predictions(&y_train, &pred_train_std);
    let (te_mse_std, te_mae_std, te_r2_std) = evaluate_predictions(&y_test, &pred_test_std);
    let el_std = t1.elapsed().as_secs_f64();
    let tput_std = (train_mols.len() + test_mols.len()) as f64 / el_std;

    println!(
        "  -> Test MSE: {:.4} | MAE: {:.4} {} | R^2: {:.4} ({:.0} mols/s)",
        te_mse_std, te_mae_std, target_unit, te_r2_std, tput_std
    );
    trials.push(PittModelTrial {
        name: "Standard-VQC-8Q".to_string(),
        description: "8-qubit hardware-efficient ansatz (16 observables: 8 Z_q + 8 Z_q Z_{q+1})"
            .to_string(),
        train_mse: tr_mse_std,
        test_mse: te_mse_std,
        test_mae: te_mae_std,
        test_r2: te_r2_std,
        throughput_mols_per_sec: tput_std,
        uses_target_as_input: false,
    });

    // ── TRIAL 3: 8-Qubit Molecular Stark-Coupled VQC ────────────────────────
    println!(
        "\n[3/3] Training 8-Qubit Molecular Stark-Coupled VQC (Dual-Angle 3D Dipole Injection)..."
    );
    let t2 = Instant::now();
    let vqc_stark = PittVqcModel::new(2026);
    let x_train_stark: Vec<Vec<f64>> = train_mols
        .iter()
        .map(|m| {
            let th = encode_topological_angles(&m.features);
            let ph = encode_stark_phases(m, true);
            vqc_stark.forward_observables(&th, &ph)
        })
        .collect();
    let x_test_stark: Vec<Vec<f64>> = test_mols
        .iter()
        .map(|m| {
            let th = encode_topological_angles(&m.features);
            let ph = encode_stark_phases(m, true);
            vqc_stark.forward_observables(&th, &ph)
        })
        .collect();
    let (w_stark, b_stark) = ridge_regression_multi(&x_train_stark, &y_train_mat, 16, 1, 1e-4);
    let pred_train_stark = predict_linear(&x_train_stark, &w_stark, &b_stark);
    let pred_test_stark = predict_linear(&x_test_stark, &w_stark, &b_stark);
    let (tr_mse_stark, _, _) = evaluate_predictions(&y_train, &pred_train_stark);
    let (te_mse_stark, te_mae_stark, te_r2_stark) = evaluate_predictions(&y_test, &pred_test_stark);
    let el_stark = t2.elapsed().as_secs_f64();
    let tput_stark = (train_mols.len() + test_mols.len()) as f64 / el_stark;

    // The Stark phases are built from the DFT dipole vector, so for a dipole target they encode
    // the label.
    let stark_reads_target = target_idx == DIPOLE_IDX;
    let adv_pct = (!stark_reads_target).then(|| (te_mse_std - te_mse_stark) / te_mse_std * 100.0);
    println!(
        "  -> Test MSE: {:.4} | MAE: {:.4} {} | R^2: {:.4} ({:.0} mols/s)",
        te_mse_stark, te_mae_stark, target_unit, te_r2_stark, tput_stark
    );
    match adv_pct {
        Some(adv) => println!("  -> Molecular Stark MSE Improvement: {adv:.2}% vs Standard VQC"),
        None => println!(
            "  -> [LEAK] Stark-VQC encodes the dipole vector, which is the target; its test error \
             is not a prediction and no advantage is reported"
        ),
    }

    trials.push(PittModelTrial {
        name: "Stark-Coupled-VQC-8Q".to_string(),
        description: "8-qubit dual-angle ansatz coupling 3D dipole moments (μ_x, μ_y, μ_z) as non-Abelian Rz tilts".to_string(),
        train_mse: tr_mse_stark,
        test_mse: te_mse_stark,
        test_mae: te_mae_stark,
        test_r2: te_r2_stark,
        throughput_mols_per_sec: tput_stark,
        uses_target_as_input: stark_reads_target,
    });

    // ── Print Final Sovereign Benchmark Table ─────────────────────────────────
    println!("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                 T-1000 PITT QUANTUM REPOSITORY VQC BENCHMARK TELEMETRY                             ║");
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

    // Write structured JSON report
    let report = PittVqcTrainingReport {
        timestamp_utc: utc_timestamp_rfc3339(),
        target_property: target_name.to_string(),
        target_unit: target_unit.to_string(),
        train_samples: n_train,
        test_samples: n_test,
        qubits: N_QUBITS,
        layers: N_LAYERS,
        trials,
        stark_advantage_pct: adv_pct,
    };

    let settings = format!("target={target_name} train={n_train} test={n_test} layers={N_LAYERS}");
    let score = Score {
        metric: "standard_vqc_test_mse",
        value: te_mse_std,
        better: Better::Lower,
        guard: None,
        settings: &settings,
    };
    let json_str = serde_json::to_string_pretty(&report).expect("Serialize report");
    match t1000::results::write_scored(REPORT_PATH, json_str, &score) {
        Ok(outcome) => println!("[RECORD] {REPORT_PATH}: {outcome:?}"),
        Err(e) => eprintln!("[ERROR] Could not write {REPORT_PATH}: {e}"),
    }
}
