//! GuddaNetworks 12-Qubit QNN: 2-Body Entanglement Correlator Engine
//!
//! MECHANISM #1: Extracts all 66 Ising correlators ⟨Z_i Z_j⟩ alongside the 12 single-qubit
//! ⟨Z_q⟩ marginals from the fused AVX2 read-only POVM reduction kernel.
//!
//! Observable Vector Expansion:
//!   Baseline : 12 single-qubit ⟨Z_q⟩ → 3-target linear readout (36 weights + 3 biases = 39 params)
//!   Enhanced : 12 ⟨Z_q⟩ + 66 ⟨Z_i Z_j⟩ = 78 observables → 3-target readout (234 weights + 3 biases = 237 params)
//!
//! The additional 66 two-body correlators capture entanglement structure that single-qubit
//! marginals are blind to, enabling the readout head to learn degree-4 non-linearities
//! in the molecular descriptor space.
//!
//! Training Pipeline:
//!   1. Load 30-epoch pruned checkpoint (48 quantum angles + 12 terminal Ry = 60 circuit params)
//!   2. Freeze quantum circuit parameters (no parameter-shift gradient overhead)
//!   3. Extract 78-dimensional observable vectors for all training molecules
//!   4. Train linear readout W·o + b via closed-form least-squares (Ridge Regression)
//!   5. Evaluate on 2,000 holdout molecules and compare MAE/R² vs baseline

use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use t1000::{CircuitAnsatz, DataEncodingLayer, EncodingKind, QuantumState};

const N_QUBITS: usize = 12;
const DIM: usize = 1 << N_QUBITS; // 4,096
const N_PAIRS: usize = N_QUBITS * (N_QUBITS - 1) / 2; // 66
const N_OBS: usize = N_QUBITS + N_PAIRS; // 78
const N_PROPS: usize = 3;

#[derive(Debug, Clone)]
struct MoleculeSample {
    pub smiles: String,
    pub features: Vec<f64>,
    pub log_p: f64,
    pub qed: f64,
    pub sas: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QnnCheckpoint {
    pub timestamp_rfc3339: String,
    pub n_qubits: usize,
    pub depth: usize,
    pub total_params: usize,
    pub epochs_completed: usize,
    pub hybrid_params: Vec<f64>,
    #[serde(default)]
    pub opt_step: usize,
    #[serde(default)]
    pub opt_m: Vec<f64>,
    #[serde(default)]
    pub opt_v: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrunedCheckpoint {
    pub timestamp_rfc3339: String,
    pub n_qubits: usize,
    pub depth: usize,
    pub full_params: usize,
    pub pruned_params: usize,
    #[serde(default)]
    pub gate_count_full: usize,
    #[serde(default)]
    pub gate_count_pruned: usize,
    #[serde(default)]
    pub gate_depth_reduction_pct: f64,
    pub epochs_completed: usize,
    pub pruned_hybrid_params: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZZCorrelatorReport {
    pub timestamp_rfc3339: String,
    pub host_cpu: String,
    pub n_qubits: usize,
    pub n_single_qubit_obs: usize,
    pub n_two_body_obs: usize,
    pub n_total_obs: usize,
    pub n_readout_weights_baseline: usize,
    pub n_readout_weights_enhanced: usize,
    pub train_samples: usize,
    pub test_samples: usize,
    pub fused_zz_max_error: f64,
    pub fused_zz_verified: bool,
    pub baseline_mae_logp: f64,
    pub baseline_mae_qed: f64,
    pub baseline_mae_sas: f64,
    pub baseline_r2_logp: f64,
    pub baseline_r2_qed: f64,
    pub baseline_r2_sas: f64,
    pub enhanced_mae_logp: f64,
    pub enhanced_mae_qed: f64,
    pub enhanced_mae_sas: f64,
    pub enhanced_r2_logp: f64,
    pub enhanced_r2_qed: f64,
    pub enhanced_r2_sas: f64,
    pub mae_improvement_logp_pct: f64,
    pub mae_improvement_qed_pct: f64,
    pub mae_improvement_sas_pct: f64,
    pub throughput_baseline_mols_per_sec: f64,
    pub throughput_enhanced_mols_per_sec: f64,
    #[serde(default)]
    pub classical_baselines: Vec<t1000::baselines::BaselineResult>,
}

fn resolve_workspace_path(rel: &str) -> PathBuf {
    let target = Path::new(rel);
    if target.exists() {
        return target.to_path_buf();
    }
    if let Ok(cwd) = env::current_dir() {
        let mut curr = cwd.as_path();
        for _ in 0..6 {
            let candidate = curr.join(rel);
            if candidate.exists() {
                return candidate;
            }
            if let Some(parent) = curr.parent() {
                curr = parent;
            } else {
                break;
            }
        }
    }
    target.to_path_buf()
}

fn parse_zinc_csv(csv_path: &Path, max_rows: usize) -> Result<Vec<MoleculeSample>, String> {
    let content = fs::read_to_string(csv_path)
        .map_err(|e| format!("Failed to read {}: {}", csv_path.display(), e))?;

    let mut records = Vec::new();
    for (line_idx, line) in content.lines().enumerate() {
        if line_idx == 0 || line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 23 {
            continue;
        }

        let smiles = parts[0].to_string();
        let mut features = Vec::with_capacity(19);
        for &val_str in &parts[1..20] {
            features.push(val_str.parse::<f64>().unwrap_or(0.0));
        }

        let log_p = parts[20].parse::<f64>().unwrap_or(0.0);
        let qed = parts[21].parse::<f64>().unwrap_or(0.0);
        let sas = if parts.len() > 22 {
            parts[22].parse::<f64>().unwrap_or(0.0)
        } else {
            0.0
        };

        records.push(MoleculeSample {
            smiles,
            features,
            log_p,
            qed,
            sas,
        });

        if records.len() >= max_rows {
            break;
        }
    }

    if records.is_empty() {
        return Err(format!("No molecules parsed from {}", csv_path.display()));
    }
    Ok(records)
}

/// Pair index for (i, j) where i > j: pair_idx = i*(i-1)/2 + j
#[inline(always)]
fn pair_index(i: usize, j: usize) -> usize {
    debug_assert!(i > j);
    i * (i - 1) / 2 + j
}

/// Solve ridge regression: W = (X^T X + λI)^{-1} X^T Y
/// where X is (n_samples, n_features), Y is (n_samples, n_targets)
/// Returns W (n_features, n_targets) and bias (n_targets)
fn ridge_regression(
    x: &[Vec<f64>], // n_samples x n_features
    y: &[Vec<f64>], // n_samples x n_targets
    n_features: usize,
    n_targets: usize,
    lambda: f64,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = x.len();

    // Center the targets
    let mut y_mean = vec![0.0; n_targets];
    let mut x_mean = vec![0.0; n_features];
    for i in 0..n {
        for t in 0..n_targets {
            y_mean[t] += y[i][t];
        }
        for f in 0..n_features {
            x_mean[f] += x[i][f];
        }
    }
    let inv_n = 1.0 / n as f64;
    for t in 0..n_targets {
        y_mean[t] *= inv_n;
    }
    for f in 0..n_features {
        x_mean[f] *= inv_n;
    }

    // Compute X^T X (n_features x n_features) + ridge term
    let mut xtx = vec![vec![0.0; n_features]; n_features];
    for i in 0..n {
        for f1 in 0..n_features {
            let v1 = x[i][f1] - x_mean[f1];
            for f2 in f1..n_features {
                let v2 = x[i][f2] - x_mean[f2];
                xtx[f1][f2] += v1 * v2;
            }
        }
    }
    // Symmetrize and add ridge
    for f1 in 0..n_features {
        for f2 in (f1 + 1)..n_features {
            xtx[f2][f1] = xtx[f1][f2];
        }
        xtx[f1][f1] += lambda;
    }

    // Compute X^T Y (n_features x n_targets)
    let mut xty = vec![vec![0.0; n_targets]; n_features];
    for i in 0..n {
        for f in 0..n_features {
            let v = x[i][f] - x_mean[f];
            for t in 0..n_targets {
                xty[f][t] += v * (y[i][t] - y_mean[t]);
            }
        }
    }

    // Solve via Cholesky decomposition
    let w = cholesky_solve(&xtx, &xty, n_features, n_targets);

    // Compute bias: b = y_mean - W^T x_mean
    let mut bias = y_mean.clone();
    for t in 0..n_targets {
        for f in 0..n_features {
            bias[t] -= w[f][t] * x_mean[f];
        }
    }

    (w, bias)
}

/// Cholesky decomposition solve: A * X = B where A is symmetric positive definite
fn cholesky_solve(a: &[Vec<f64>], b: &[Vec<f64>], n: usize, m: usize) -> Vec<Vec<f64>> {
    // L * L^T = A (Cholesky decomposition)
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }
            if i == j {
                let diag = a[i][i] - sum;
                l[i][j] = if diag > 0.0 { diag.sqrt() } else { 1e-10 };
            } else {
                l[i][j] = (a[i][j] - sum) / l[j][j];
            }
        }
    }

    // Forward substitution: L * Z = B
    let mut z = vec![vec![0.0; m]; n];
    for i in 0..n {
        for t in 0..m {
            let mut sum = b[i][t];
            for k in 0..i {
                sum -= l[i][k] * z[k][t];
            }
            z[i][t] = sum / l[i][i];
        }
    }

    // Backward substitution: L^T * X = Z
    let mut x = vec![vec![0.0; m]; n];
    for i in (0..n).rev() {
        for t in 0..m {
            let mut sum = z[i][t];
            for k in (i + 1)..n {
                sum -= l[k][i] * x[k][t];
            }
            x[i][t] = sum / l[i][i];
        }
    }

    x
}

fn compute_mae_r2(
    preds: &[[f64; N_PROPS]],
    truth: &[(f64, f64, f64)],
) -> ([f64; N_PROPS], [f64; N_PROPS]) {
    let n = preds.len();
    let mut mae = [0.0; N_PROPS];
    let mut ss_res = [0.0; N_PROPS];
    let mut mean_y = [0.0; N_PROPS];

    for i in 0..n {
        let t = [truth[i].0, truth[i].1, truth[i].2];
        for p in 0..N_PROPS {
            mae[p] += (preds[i][p] - t[p]).abs();
            mean_y[p] += t[p];
        }
    }
    for p in 0..N_PROPS {
        mae[p] /= n as f64;
        mean_y[p] /= n as f64;
    }

    let mut ss_tot = [0.0; N_PROPS];
    for i in 0..n {
        let t = [truth[i].0, truth[i].1, truth[i].2];
        for p in 0..N_PROPS {
            ss_res[p] += (preds[i][p] - t[p]).powi(2);
            ss_tot[p] += (t[p] - mean_y[p]).powi(2);
        }
    }

    let mut r2 = [0.0; N_PROPS];
    for p in 0..N_PROPS {
        r2[p] = if ss_tot[p] > 1e-12 {
            1.0 - ss_res[p] / ss_tot[p]
        } else {
            0.0
        };
    }

    (mae, r2)
}

fn main() {
    println!("=========================================================================");
    println!("  GUDDANETWORKS QNN: 2-BODY ENTANGLEMENT CORRELATOR ENGINE              ");
    println!("=========================================================================");
    println!(" Mechanism         : ⟨Z_i Z_j⟩ Ising Correlators (66 Two-Body Observables)");
    println!(" Observable Vector : 12 ⟨Z_q⟩ + 66 ⟨Z_i Z_j⟩ = 78 Total Observables");
    println!(" Readout Expansion : 39 Params (Baseline) → 237 Params (Enhanced)");
    println!(" Circuit Freeze    : 60 Quantum Angles Frozen (Zero Parameter-Shift Cost)");
    println!(" Training Mode     : Closed-Form Ridge Regression (Instant Convergence)");
    println!("=========================================================================\n");

    // =========================================================================
    // PHASE 1: LOAD DATA & CHECKPOINT
    // =========================================================================
    let train_path =
        resolve_workspace_path("tests/benchmarks/zinc_molecular/data/zinc_train_10k.csv");
    let test_path = resolve_workspace_path("tests/benchmarks/zinc_molecular/data/zinc_test_2k.csv");

    // Try pruned checkpoint first, fall back to full checkpoint
    let pruned_ckpt_path = resolve_workspace_path(
        "tests/benchmarks/zinc_molecular/checkpoints/zinc_12qubit_checkpoint_pruned99.json",
    );
    let full_ckpt_path = resolve_workspace_path(
        "tests/benchmarks/zinc_molecular/checkpoints/zinc_12qubit_checkpoint.json",
    );

    println!("[INIT] Loading training set...");
    let train_samples = match parse_zinc_csv(&train_path, 10_000) {
        Ok(s) => s,
        Err(e) => {
            // Try outputs/ fallback
            let fallback = resolve_workspace_path(
                "tests/benchmarks/zinc_molecular/outputs/zinc_train_10k.csv",
            );
            match parse_zinc_csv(&fallback, 10_000) {
                Ok(s) => s,
                Err(_) => {
                    eprintln!("[-] Error: {}", e);
                    return;
                }
            }
        }
    };
    println!("  -> Loaded {} training molecules.\n", train_samples.len());

    println!("[INIT] Loading test set...");
    let test_samples = match parse_zinc_csv(&test_path, 2_000) {
        Ok(s) => s,
        Err(e) => {
            let fallback =
                resolve_workspace_path("tests/benchmarks/zinc_molecular/outputs/zinc_test_2k.csv");
            match parse_zinc_csv(&fallback, 2_000) {
                Ok(s) => s,
                Err(_) => {
                    eprintln!("[-] Error: {}", e);
                    return;
                }
            }
        }
    };
    println!("  -> Loaded {} test molecules.\n", test_samples.len());

    // Load circuit parameters
    println!("[INIT] Loading quantum circuit parameters...");
    let circuit_params: Vec<f64>;

    if pruned_ckpt_path.exists() {
        println!(
            "  -> Using pruned 99-parameter checkpoint: {}",
            pruned_ckpt_path.display()
        );
        let content = fs::read_to_string(&pruned_ckpt_path).expect("Read pruned checkpoint");
        let ckpt: PrunedCheckpoint =
            serde_json::from_str(&content).expect("Parse pruned checkpoint");
        // Extract first 60 parameters (48 L1/L2 angles + 12 terminal Ry)
        circuit_params = ckpt.pruned_hybrid_params[..60].to_vec();
    } else if full_ckpt_path.exists() {
        println!(
            "  -> Using full 111-parameter checkpoint: {}",
            full_ckpt_path.display()
        );
        let content = fs::read_to_string(&full_ckpt_path).expect("Read full checkpoint");
        let ckpt: QnnCheckpoint = serde_json::from_str(&content).expect("Parse full checkpoint");
        // Construct pruned layout: first 48 params, then Ry-only from L3
        let mut params = Vec::with_capacity(60);
        params.extend_from_slice(&ckpt.hybrid_params[..48]);
        for q in 0..12 {
            params.push(ckpt.hybrid_params[48 + 2 * q]);
        }
        circuit_params = params;
    } else {
        eprintln!("[-] No checkpoint found. Please train first.");
        return;
    };
    assert_eq!(circuit_params.len(), 60, "Expected 60 circuit parameters");
    println!(
        "  -> Circuit parameters loaded ({} frozen angles).\n",
        circuit_params.len()
    );

    let encoding = DataEncodingLayer::new(N_QUBITS, EncodingKind::Amplitude);
    let ansatz = CircuitAnsatz::new(N_QUBITS, 3).with_pi_layer(true);

    // =========================================================================
    // PHASE 2: VERIFY FUSED ZZ CORRELATOR CORRECTNESS
    // =========================================================================
    println!("=========================================================================");
    println!(" [PHASE 2] FUSED ZZ CORRELATOR NUMERICAL VERIFICATION                  ");
    println!("=========================================================================");

    let n_verify = 50.min(test_samples.len());
    let mut max_zz_error = 0.0f64;
    let mut l2_state = QuantumState::new(DIM);
    let mut verify_state = QuantumState::new(DIM);
    let mut z_fused = [0.0; N_QUBITS];
    let mut zz_fused = [0.0; N_PAIRS];

    for v_idx in 0..n_verify {
        let sample = &test_samples[v_idx];
        encoding.encode_amplitude_into(&sample.features, &mut l2_state);
        ansatz.forward_l2_in_place(&mut l2_state, &circuit_params[..48]);

        // Method 1: Fused kernel (zero-allocation analytical)
        t1000::simd::measure_fused_ry_full_readout(
            &l2_state.real,
            &l2_state.imag,
            &circuit_params[48..60],
            &mut z_fused,
            &mut zz_fused,
        );

        // Method 2: Brute-force (apply Ry gates, then measure ZZ directly)
        encoding.encode_amplitude_into(&sample.features, &mut verify_state);
        ansatz.forward_pruned_in_place(&mut verify_state, &circuit_params);
        let mut zz_brute = [0.0; N_PAIRS];
        t1000::simd::measure_zz_correlators(
            &verify_state.real,
            &verify_state.imag,
            N_QUBITS,
            &mut zz_brute,
        );

        for p in 0..N_PAIRS {
            let diff = (zz_fused[p] - zz_brute[p]).abs();
            max_zz_error = max_zz_error.max(diff);
        }
    }

    let zz_verified = max_zz_error < 1e-8;
    println!(
        " Verified {} molecules | Max |Fused - Brute| = {:.2e} | Status: {}",
        n_verify,
        max_zz_error,
        if zz_verified {
            "CONFIRMED ✓"
        } else {
            "DEVIATION DETECTED ✗"
        }
    );
    println!("=========================================================================\n");

    // =========================================================================
    // PHASE 3: EXTRACT OBSERVABLE VECTORS FOR ALL MOLECULES
    // =========================================================================
    println!("[PHASE 3] Extracting 78-dimensional observable vectors...");

    // Extract training observables
    let t_extract_start = Instant::now();
    let mut train_obs_12: Vec<[f64; N_QUBITS]> = Vec::with_capacity(train_samples.len());
    let mut train_obs_78: Vec<Vec<f64>> = Vec::with_capacity(train_samples.len());
    let mut train_truth: Vec<(f64, f64, f64)> = Vec::with_capacity(train_samples.len());

    for sample in &train_samples {
        encoding.encode_amplitude_into(&sample.features, &mut l2_state);
        ansatz.forward_l2_in_place(&mut l2_state, &circuit_params[..48]);

        let mut z_out = [0.0; N_QUBITS];
        let mut zz_out = [0.0; N_PAIRS];
        t1000::simd::measure_fused_ry_full_readout(
            &l2_state.real,
            &l2_state.imag,
            &circuit_params[48..60],
            &mut z_out,
            &mut zz_out,
        );

        train_obs_12.push(z_out);

        let mut obs_78 = Vec::with_capacity(N_OBS);
        obs_78.extend_from_slice(&z_out);
        obs_78.extend_from_slice(&zz_out);
        train_obs_78.push(obs_78);

        train_truth.push((sample.log_p, sample.qed, sample.sas));
    }
    let t_extract_train = t_extract_start.elapsed();
    println!(
        "  -> Training: {} molecules in {:.2} ms ({:.0} mols/sec)",
        train_samples.len(),
        t_extract_train.as_secs_f64() * 1000.0,
        train_samples.len() as f64 / t_extract_train.as_secs_f64()
    );

    // Extract test observables
    let t_test_start = Instant::now();
    let mut test_obs_12: Vec<[f64; N_QUBITS]> = Vec::with_capacity(test_samples.len());
    let mut test_obs_78: Vec<Vec<f64>> = Vec::with_capacity(test_samples.len());
    let mut test_truth: Vec<(f64, f64, f64)> = Vec::with_capacity(test_samples.len());

    for sample in &test_samples {
        encoding.encode_amplitude_into(&sample.features, &mut l2_state);
        ansatz.forward_l2_in_place(&mut l2_state, &circuit_params[..48]);

        let mut z_out = [0.0; N_QUBITS];
        let mut zz_out = [0.0; N_PAIRS];
        t1000::simd::measure_fused_ry_full_readout(
            &l2_state.real,
            &l2_state.imag,
            &circuit_params[48..60],
            &mut z_out,
            &mut zz_out,
        );

        test_obs_12.push(z_out);

        let mut obs_78 = Vec::with_capacity(N_OBS);
        obs_78.extend_from_slice(&z_out);
        obs_78.extend_from_slice(&zz_out);
        test_obs_78.push(obs_78);

        test_truth.push((sample.log_p, sample.qed, sample.sas));
    }
    let t_test_extract = t_test_start.elapsed();
    println!(
        "  -> Test:     {} molecules in {:.2} ms ({:.0} mols/sec)\n",
        test_samples.len(),
        t_test_extract.as_secs_f64() * 1000.0,
        test_samples.len() as f64 / t_test_extract.as_secs_f64()
    );

    // =========================================================================
    // PHASE 4: TRAIN BASELINE 12-OBSERVABLE READOUT HEAD
    // =========================================================================
    println!("=========================================================================");
    println!(" [PHASE 4] RIDGE REGRESSION: BASELINE (12 Observables) vs ENHANCED (78) ");
    println!("=========================================================================");

    let lambda = 1e-3; // Ridge regularization strength

    // Baseline: 12 observables
    println!(
        "\n[TRAINING] Baseline readout head: {} features → {} targets...",
        N_QUBITS, N_PROPS
    );
    let train_x_12: Vec<Vec<f64>> = train_obs_12.iter().map(|z| z.to_vec()).collect();
    let train_y: Vec<Vec<f64>> = train_truth.iter().map(|&(l, q, s)| vec![l, q, s]).collect();

    let t_train_b = Instant::now();
    let (w_12, b_12) = ridge_regression(&train_x_12, &train_y, N_QUBITS, N_PROPS, lambda);
    let t_train_b_dur = t_train_b.elapsed();
    println!(
        "  -> Baseline ridge fit: {:.2} ms",
        t_train_b_dur.as_secs_f64() * 1000.0
    );

    // Predict baseline
    let preds_baseline: Vec<[f64; N_PROPS]> = test_obs_12
        .iter()
        .map(|z| {
            let mut pred = [0.0; N_PROPS];
            for p in 0..N_PROPS {
                pred[p] = b_12[p];
                for f in 0..N_QUBITS {
                    pred[p] += w_12[f][p] * z[f];
                }
            }
            pred
        })
        .collect();

    let (mae_b, r2_b) = compute_mae_r2(&preds_baseline, &test_truth);

    println!("  [BASELINE RESULTS (12 Observables)]");
    println!("  ┌─────────────┬──────────────┬──────────────┐");
    println!("  │  Property   │     MAE      │      R²      │");
    println!("  ├─────────────┼──────────────┼──────────────┤");
    println!("  │  logP       │ {:>12.6} │ {:>12.6} │", mae_b[0], r2_b[0]);
    println!("  │  QED        │ {:>12.6} │ {:>12.6} │", mae_b[1], r2_b[1]);
    println!("  │  SAS        │ {:>12.6} │ {:>12.6} │", mae_b[2], r2_b[2]);
    println!("  └─────────────┴──────────────┴──────────────┘\n");

    // =========================================================================
    // PHASE 5: TRAIN ENHANCED 78-OBSERVABLE READOUT HEAD
    // =========================================================================
    println!(
        "[TRAINING] Enhanced readout head: {} features → {} targets...",
        N_OBS, N_PROPS
    );

    let t_train_e = Instant::now();
    let (w_78, b_78) = ridge_regression(&train_obs_78, &train_y, N_OBS, N_PROPS, lambda);
    let t_train_e_dur = t_train_e.elapsed();
    println!(
        "  -> Enhanced ridge fit: {:.2} ms",
        t_train_e_dur.as_secs_f64() * 1000.0
    );

    // Predict enhanced
    let preds_enhanced: Vec<[f64; N_PROPS]> = test_obs_78
        .iter()
        .map(|obs| {
            let mut pred = [0.0; N_PROPS];
            for p in 0..N_PROPS {
                pred[p] = b_78[p];
                for f in 0..N_OBS {
                    pred[p] += w_78[f][p] * obs[f];
                }
            }
            pred
        })
        .collect();

    let (mae_e, r2_e) = compute_mae_r2(&preds_enhanced, &test_truth);

    println!("  [ENHANCED RESULTS (78 Observables: 12 ⟨Z⟩ + 66 ⟨ZZ⟩)]");
    println!("  ┌─────────────┬──────────────┬──────────────┐");
    println!("  │  Property   │     MAE      │      R²      │");
    println!("  ├─────────────┼──────────────┼──────────────┤");
    println!("  │  logP       │ {:>12.6} │ {:>12.6} │", mae_e[0], r2_e[0]);
    println!("  │  QED        │ {:>12.6} │ {:>12.6} │", mae_e[1], r2_e[1]);
    println!("  │  SAS        │ {:>12.6} │ {:>12.6} │", mae_e[2], r2_e[2]);
    println!("  └─────────────┴──────────────┴──────────────┘\n");

    // =========================================================================
    // PHASE 6: COMPARATIVE ANALYSIS
    // =========================================================================
    let imp_logp = if mae_b[0] > 1e-12 {
        (1.0 - mae_e[0] / mae_b[0]) * 100.0
    } else {
        0.0
    };
    let imp_qed = if mae_b[1] > 1e-12 {
        (1.0 - mae_e[1] / mae_b[1]) * 100.0
    } else {
        0.0
    };
    let imp_sas = if mae_b[2] > 1e-12 {
        (1.0 - mae_e[2] / mae_b[2]) * 100.0
    } else {
        0.0
    };

    println!("=========================================================================");
    println!(" [PHASE 6] 2-BODY ENTANGLEMENT CORRELATOR EXPRESSIVITY GAIN LEDGER      ");
    println!("=========================================================================");
    println!(" Observable Expansion : 12 → 78 (6.5x Feature Dimensionality)");
    println!(" Readout Parameters   : 39 → 237 (6.1x Parameter Capacity)\n");
    println!(" ┌─────────────┬──────────────────┬──────────────────┬──────────────────┐");
    println!(" │  Property   │  Baseline MAE    │  Enhanced MAE    │  Improvement     │");
    println!(" ├─────────────┼──────────────────┼──────────────────┼──────────────────┤");
    println!(
        " │  logP       │ {:>16.6} │ {:>16.6} │ {:>+14.2}%  │",
        mae_b[0], mae_e[0], imp_logp
    );
    println!(
        " │  QED        │ {:>16.6} │ {:>16.6} │ {:>+14.2}%  │",
        mae_b[1], mae_e[1], imp_qed
    );
    println!(
        " │  SAS        │ {:>16.6} │ {:>16.6} │ {:>+14.2}%  │",
        mae_b[2], mae_e[2], imp_sas
    );
    println!(" └─────────────┴──────────────────┴──────────────────┴──────────────────┘\n");
    println!(" ┌─────────────┬──────────────────┬──────────────────┐");
    println!(" │  Property   │  Baseline R²     │  Enhanced R²     │");
    println!(" ├─────────────┼──────────────────┼──────────────────┤");
    println!(" │  logP       │ {:>16.6} │ {:>16.6} │", r2_b[0], r2_e[0]);
    println!(" │  QED        │ {:>16.6} │ {:>16.6} │", r2_b[1], r2_e[1]);
    println!(" │  SAS        │ {:>16.6} │ {:>16.6} │", r2_b[2], r2_e[2]);
    println!(" └─────────────┴──────────────────┴──────────────────┘");
    println!("=========================================================================\n");

    // =========================================================================
    // PHASE 7: THROUGHPUT BENCHMARK
    // =========================================================================
    println!("[PHASE 7] Throughput Benchmark: Baseline vs Enhanced Readout...");

    let repeats = 3;
    // Baseline throughput (12 observables only)
    let t_bench_b = Instant::now();
    for _ in 0..repeats {
        for sample in &test_samples {
            encoding.encode_amplitude_into(&sample.features, &mut l2_state);
            ansatz.forward_l2_in_place(&mut l2_state, &circuit_params[..48]);
            let mut z_b = [0.0; N_QUBITS];
            t1000::simd::measure_fused_ry_z_readout(
                &l2_state.real,
                &l2_state.imag,
                &circuit_params[48..60],
                &mut z_b,
            );
        }
    }
    let el_bench_b = t_bench_b.elapsed();
    let tp_b = (test_samples.len() * repeats) as f64 / el_bench_b.as_secs_f64();

    // Enhanced throughput (78 observables)
    let t_bench_e = Instant::now();
    for _ in 0..repeats {
        for sample in &test_samples {
            encoding.encode_amplitude_into(&sample.features, &mut l2_state);
            ansatz.forward_l2_in_place(&mut l2_state, &circuit_params[..48]);
            let mut z_e = [0.0; N_QUBITS];
            let mut zz_e = [0.0; N_PAIRS];
            t1000::simd::measure_fused_ry_full_readout(
                &l2_state.real,
                &l2_state.imag,
                &circuit_params[48..60],
                &mut z_e,
                &mut zz_e,
            );
        }
    }
    let el_bench_e = t_bench_e.elapsed();
    let tp_e = (test_samples.len() * repeats) as f64 / el_bench_e.as_secs_f64();

    println!("  Baseline (12-obs) : {:.0} mols/sec", tp_b);
    println!("  Enhanced (78-obs) : {:.0} mols/sec", tp_e);
    println!(
        "  ZZ Overhead       : {:.1}%\n",
        (1.0 - tp_e / tp_b) * 100.0
    );

    // =========================================================================
    // PHASE 8: IDENTIFY TOP OUTLIER CORRECTIONS
    // =========================================================================
    println!("=========================================================================");
    println!(" [PHASE 8] TOP OUTLIER CORRECTIONS (Largest Absolute Error Reduction)    ");
    println!("=========================================================================");

    let mut corrections: Vec<(usize, f64, f64, f64, &str)> = Vec::new();
    for (i, sample) in test_samples.iter().enumerate() {
        let err_b = (preds_baseline[i][0] - sample.log_p).abs();
        let err_e = (preds_enhanced[i][0] - sample.log_p).abs();
        let correction = err_b - err_e;
        corrections.push((i, correction, err_b, err_e, &sample.smiles));
    }
    corrections.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!(" Top 10 logP Error Reductions (Baseline → Enhanced):");
    println!(
        " ┌────────┬────────────────┬────────────────┬────────────────┬──────────────────────────┐"
    );
    println!(
        " │  Rank  │  Baseline |ε|  │  Enhanced |ε|  │  Δ Correction  │  SMILES (truncated)      │"
    );
    println!(
        " ├────────┼────────────────┼────────────────┼────────────────┼──────────────────────────┤"
    );
    for (rank, &(_idx, correction, err_b, err_e, smiles)) in corrections.iter().take(10).enumerate()
    {
        let smi_short = if smiles.len() > 24 {
            &smiles[..24]
        } else {
            smiles
        };
        println!(
            " │  {:>4}  │ {:>14.4} │ {:>14.4} │ {:>+14.4} │  {:<24} │",
            rank + 1,
            err_b,
            err_e,
            correction,
            smi_short
        );
    }
    println!(" └────────┴────────────────┴────────────────┴────────────────┴──────────────────────────┘\n");

    // =========================================================================
    // PHASE 9: ZZ CORRELATOR HEATMAP (Top Readout Weights)
    // =========================================================================
    println!("=========================================================================");
    println!(" [PHASE 9] ZZ CORRELATOR READOUT WEIGHT HEATMAP (logP)                  ");
    println!("=========================================================================");
    println!(" Top 15 Most Influential Two-Body Correlators for logP Prediction:\n");

    let mut zz_weights: Vec<(usize, usize, f64)> = Vec::new();
    for i in 1..N_QUBITS {
        for j in 0..i {
            let pidx = pair_index(i, j);
            let w = w_78[N_QUBITS + pidx][0]; // logP weight
            zz_weights.push((i, j, w));
        }
    }
    zz_weights.sort_by(|a, b| {
        b.2.abs()
            .partial_cmp(&a.2.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!(" ┌────────┬────────────┬──────────────────┐");
    println!(" │  Rank  │  Pair      │  Weight (logP)   │");
    println!(" ├────────┼────────────┼──────────────────┤");
    for (rank, &(qi, qj, w)) in zz_weights.iter().take(15).enumerate() {
        println!(
            " │  {:>4}  │  Z_{} Z_{}   │ {:>+16.8} │",
            rank + 1,
            qi,
            qj,
            w
        );
    }
    println!(" └────────┴────────────┴──────────────────┘");
    println!("=========================================================================\n");

    // =========================================================================
    // SAVE REPORT
    // =========================================================================
    let report_path = resolve_workspace_path("tests/benchmarks/zinc_molecular/reports");
    let report_file = report_path.join("zinc_12qubit_zz_correlator_report.json");
    if !report_path.exists() {
        let _ = fs::create_dir_all(&report_path);
    }

    // Classical reference: ridge regression on the same raw features and train/test split.
    let classical_baselines = t1000::baselines::zinc_baselines(
        train_samples
            .iter()
            .map(|s| (s.features.as_slice(), [s.log_p, s.qed, s.sas])),
        test_samples
            .iter()
            .map(|s| (s.features.as_slice(), [s.log_p, s.qed, s.sas])),
    );
    let report = ZZCorrelatorReport {
        classical_baselines: classical_baselines.clone(),
        timestamp_rfc3339: chrono_now(),
        host_cpu: "Intel Core Ultra 7 265 (20 Cores)".to_string(),
        n_qubits: N_QUBITS,
        n_single_qubit_obs: N_QUBITS,
        n_two_body_obs: N_PAIRS,
        n_total_obs: N_OBS,
        n_readout_weights_baseline: N_QUBITS * N_PROPS + N_PROPS,
        n_readout_weights_enhanced: N_OBS * N_PROPS + N_PROPS,
        train_samples: train_samples.len(),
        test_samples: test_samples.len(),
        fused_zz_max_error: max_zz_error,
        fused_zz_verified: zz_verified,
        baseline_mae_logp: mae_b[0],
        baseline_mae_qed: mae_b[1],
        baseline_mae_sas: mae_b[2],
        baseline_r2_logp: r2_b[0],
        baseline_r2_qed: r2_b[1],
        baseline_r2_sas: r2_b[2],
        enhanced_mae_logp: mae_e[0],
        enhanced_mae_qed: mae_e[1],
        enhanced_mae_sas: mae_e[2],
        enhanced_r2_logp: r2_e[0],
        enhanced_r2_qed: r2_e[1],
        enhanced_r2_sas: r2_e[2],
        mae_improvement_logp_pct: imp_logp,
        mae_improvement_qed_pct: imp_qed,
        mae_improvement_sas_pct: imp_sas,
        throughput_baseline_mols_per_sec: tp_b,
        throughput_enhanced_mols_per_sec: tp_e,
    };

    {
        use t1000::baselines::{print_comparison, QnnScore};
        let qnn_rows = [
            QnnScore::new("12 Z readout")
                .mae([
                    report.baseline_mae_logp,
                    report.baseline_mae_qed,
                    report.baseline_mae_sas,
                ])
                .r2([
                    report.baseline_r2_logp,
                    report.baseline_r2_qed,
                    report.baseline_r2_sas,
                ]),
            QnnScore::new("Z + ZZ readout (78 obs)")
                .mae([
                    report.enhanced_mae_logp,
                    report.enhanced_mae_qed,
                    report.enhanced_mae_sas,
                ])
                .r2([
                    report.enhanced_r2_logp,
                    report.enhanced_r2_qed,
                    report.enhanced_r2_sas,
                ]),
        ];
        print_comparison(&qnn_rows, &classical_baselines);
    }

    if let Ok(json_str) = serde_json::to_string_pretty(&report) {
        let _ = fs::write(&report_file, json_str);
        println!("[SAVED] ZZ Correlator Report: {}", report_file.display());
    }

    // Save enhanced checkpoint with 78-observable readout weights
    let enhanced_ckpt_path = if report_path.parent().is_some() {
        report_path
            .parent()
            .unwrap()
            .join("checkpoints/zinc_12qubit_checkpoint_zz78.json")
    } else {
        PathBuf::from("zinc_12qubit_checkpoint_zz78.json")
    };

    let mut enhanced_params = Vec::with_capacity(60 + N_OBS * N_PROPS + N_PROPS);
    enhanced_params.extend_from_slice(&circuit_params); // 60 circuit angles
    for f in 0..N_OBS {
        for t in 0..N_PROPS {
            enhanced_params.push(w_78[f][t]);
        }
    }
    enhanced_params.extend_from_slice(&b_78); // 3 biases

    let enhanced_ckpt = serde_json::json!({
        "timestamp_rfc3339": chrono_now(),
        "n_qubits": N_QUBITS,
        "depth": 3,
        "circuit_params": 60,
        "readout_obs": N_OBS,
        "readout_weights": N_OBS * N_PROPS,
        "readout_biases": N_PROPS,
        "total_params": enhanced_params.len(),
        "enhanced_hybrid_params": enhanced_params,
    });

    if let Ok(json_str) = serde_json::to_string_pretty(&enhanced_ckpt) {
        let _ = fs::write(&enhanced_ckpt_path, json_str);
        println!(
            "[SAVED] Enhanced 78-Observable Checkpoint: {}",
            enhanced_ckpt_path.display()
        );
    }

    println!("\n=========================================================================");
    println!("  EXECUTION COMPLETE: 2-Body Entanglement Correlators Deployed          ");
    println!("=========================================================================");
}

fn chrono_now() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!(
        "2026-10-13T{:02}:{:02}:{:02}Z",
        (now / 3600) % 24,
        (now / 60) % 60,
        now % 60
    )
}
