// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Shared utilities for the `zinc_qnn_12qubit_*` / `zinc_qnn_train` training binaries.
//!
//! Each item below used to be copied into several binaries. Only code that was
//! identical across those copies lives here, so moving to it changed no numerics:
//!
//! - [`resolve_workspace_path`] — workspace-root path resolution
//! - checkpoint contracts between pipeline stages: [`QnnCheckpoint`] (hybrid ->
//!   pruned / fused / zz), [`PrunedCheckpoint`] (pruned -> midcircuit / reuploading /
//!   zz), [`MidCircuitCheckpoint`] (midcircuit -> reuploading),
//!   [`ReuploadingCheckpoint`] (reuploading -> qng)
//! - [`eval_all_z12`], [`compute_state_norm`] — statevector readouts
//! - [`solve_ridge_regression`], [`compute_mae_and_r2`] — the linear readout head
//! - [`PropertyMetric`] — per-property report row
//! - [`utc_timestamp_rfc3339`] — checkpoint timestamps
//! - [`print_epoch_row`], [`save_checkpoint`], [`load_checkpoint`]
//!
//! Molecule CSVs are read with [`crate::zinc_data::load`] (bit-identical to the
//! per-binary parsers it replaced).

use crate::QuantumState;
use serde::{Deserialize, Serialize};
use std::env;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

// ── Path resolution ───────────────────────────────────────────────────────────

/// Workspace root: `crates/t1000` -> `crates` -> workspace.
fn workspace_root() -> Option<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2)?;
    root.join("Cargo.toml")
        .is_file()
        .then(|| root.to_path_buf())
}

/// Resolves a workspace-relative path such as `tests/benchmarks/zinc_molecular/...`.
///
/// An existing file under the workspace root wins, then an existing file found by
/// walking up from the current directory. A path that does not exist yet (an output
/// file) is anchored at the workspace root, so it lands in the same place whatever
/// the current directory is.
pub fn resolve_workspace_path(rel: &str) -> PathBuf {
    let root = workspace_root();
    if let Some(candidate) = root.as_ref().map(|r| r.join(rel)) {
        if candidate.exists() {
            return candidate;
        }
    }
    if let Ok(cwd) = env::current_dir() {
        for dir in cwd.ancestors().take(7) {
            let candidate = dir.join(rel);
            if candidate.exists() {
                return candidate;
            }
        }
    }
    match root {
        Some(r) => r.join(rel),
        None => PathBuf::from(rel),
    }
}

// ── Checkpoint contracts ──────────────────────────────────────────────────────

/// Full 12-qubit hybrid model, written by `zinc_qnn_12qubit_hybrid`.
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

/// Magnitude-pruned model, written by `zinc_qnn_12qubit_pruned`.
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

/// Mid-circuit feed-forward model, written by `zinc_qnn_12qubit_midcircuit`.
///
/// The readout heads default to empty so readers that only need the circuit
/// (`zinc_qnn_12qubit_reuploading`) accept files without them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MidCircuitCheckpoint {
    pub timestamp_rfc3339: String,
    pub n_qubits: usize,
    pub depth: usize,
    pub source_qubit: usize,
    pub circuit_params: Vec<f64>,     // 60 params
    pub feedforward_alphas: Vec<f64>, // 12 alphas
    #[serde(default)]
    pub readout_weights_12: Vec<Vec<f64>>, // 12 x 3
    #[serde(default)]
    pub readout_biases_12: Vec<f64>, // 3
    #[serde(default)]
    pub readout_weights_78: Vec<Vec<f64>>, // 78 x 3
    #[serde(default)]
    pub readout_biases_78: Vec<f64>, // 3
}

/// Data re-uploading model, written by `zinc_qnn_12qubit_reuploading`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReuploadingCheckpoint {
    pub timestamp_rfc3339: String,
    pub n_qubits: usize,
    pub depth: usize,
    pub source_qubit: usize,
    pub circuit_params: Vec<f64>,     // 60 params
    pub feedforward_alphas: Vec<f64>, // 12 alphas
    pub upload_scales_l0: Vec<f64>,   // 12 DRU scales
    pub readout_weights_12: Vec<Vec<f64>>,
    pub readout_biases_12: Vec<f64>,
    pub readout_weights_78: Vec<Vec<f64>>,
    pub readout_biases_78: Vec<f64>,
}

/// Per-property error metrics in a training report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyMetric {
    pub initial_mse: f64,
    pub tuned_mse: f64,
    pub tuned_mae: f64,
    pub pearson_r: f64,
    pub error_reduction_pct: f64,
}

// ── Statevector readouts ──────────────────────────────────────────────────────

/// Fused single-pass expectation evaluation for all 12 Pauli-Z observables.
/// Operates on a 4096-dim statevector resident in L2 cache.
#[inline(always)]
pub fn eval_all_z12(state: &QuantumState, z_out: &mut [f64; 12]) {
    debug_assert_eq!(state.dim, 1 << 12, "eval_all_z12 expects a 12-qubit state");
    z_out.fill(0.0);
    for idx in 0..state.dim {
        let prob = state.real[idx] * state.real[idx] + state.imag[idx] * state.imag[idx];
        for (q, z) in z_out.iter_mut().enumerate() {
            if (idx >> q) & 1 == 0 {
                *z += prob;
            } else {
                *z -= prob;
            }
        }
    }
}

/// L2 norm of a statevector.
pub fn compute_state_norm(state: &QuantumState) -> f64 {
    state
        .real
        .iter()
        .zip(&state.imag)
        .map(|(&r, &i)| r * r + i * i)
        .sum::<f64>()
        .sqrt()
}

// ── Linear readout head ───────────────────────────────────────────────────────

/// Solves (X^T X + lambda I) W = X^T Y via Gauss-Jordan elimination with partial pivoting.
///
/// The design matrix gets an appended bias column; the bias is not penalized.
/// Returns `(weights[D x P], biases[P])`.
#[allow(clippy::needless_range_loop)] // index form mirrors the linear algebra
pub fn solve_ridge_regression(
    x_matrix: &[Vec<f64>], // N rows x D features
    y_matrix: &[Vec<f64>], // N rows x P targets
    lambda: f64,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = x_matrix.len();
    assert!(n > 0);
    let d = x_matrix[0].len();
    let p = y_matrix[0].len();

    // Design matrix augmented with bias column: [x, 1.0] -> dim = D + 1
    let d_aug = d + 1;
    let mut xtx = vec![vec![0.0; d_aug]; d_aug];
    let mut xty = vec![vec![0.0; p]; d_aug];

    for (row_x, row_y) in x_matrix.iter().zip(y_matrix) {
        for i in 0..d_aug {
            let val_i = if i < d { row_x[i] } else { 1.0 };
            for j in 0..d_aug {
                let val_j = if j < d { row_x[j] } else { 1.0 };
                xtx[i][j] += val_i * val_j;
            }
            for k in 0..p {
                xty[i][k] += val_i * row_y[k];
            }
        }
    }

    // Add ridge penalty (do not penalize bias)
    for i in 0..d {
        xtx[i][i] += lambda;
    }

    // Solve for each target column via Gauss-Jordan elimination
    let mut weights = vec![vec![0.0; p]; d];
    let mut biases = vec![0.0; p];

    for k in 0..p {
        // Augmented system: [xtx | xty[:, k]]
        let mut aug = vec![vec![0.0; d_aug + 1]; d_aug];
        for i in 0..d_aug {
            for j in 0..d_aug {
                aug[i][j] = xtx[i][j];
            }
            aug[i][d_aug] = xty[i][k];
        }

        // Gauss-Jordan elimination
        for col in 0..d_aug {
            let mut max_row = col;
            let mut max_val = aug[col][col].abs();
            for row in (col + 1)..d_aug {
                if aug[row][col].abs() > max_val {
                    max_val = aug[row][col].abs();
                    max_row = row;
                }
            }
            aug.swap(col, max_row);

            let pivot = aug[col][col];
            if pivot.abs() < 1e-14 {
                continue;
            }

            for j in col..=d_aug {
                aug[col][j] /= pivot;
            }

            for row in 0..d_aug {
                if row != col {
                    let factor = aug[row][col];
                    for j in col..=d_aug {
                        aug[row][j] -= factor * aug[col][j];
                    }
                }
            }
        }

        for i in 0..d {
            weights[i][k] = aug[i][d_aug];
        }
        biases[k] = aug[d][d_aug];
    }

    (weights, biases)
}

/// Mean absolute error and R^2 for each of the three targets (logP, QED, SAS).
#[allow(clippy::needless_range_loop)] // index form mirrors the per-target sums
pub fn compute_mae_and_r2(preds: &[Vec<f64>], targets: &[Vec<f64>]) -> ([f64; 3], [f64; 3]) {
    let n = preds.len() as f64;
    assert!(n > 1.0);

    let mut mean_t = [0.0; 3];
    for row in targets {
        for k in 0..3 {
            mean_t[k] += row[k];
        }
    }
    for k in 0..3 {
        mean_t[k] /= n;
    }

    let mut mae = [0.0; 3];
    let mut ss_res = [0.0; 3];
    let mut ss_tot = [0.0; 3];

    for (p, t) in preds.iter().zip(targets) {
        for k in 0..3 {
            let err = p[k] - t[k];
            mae[k] += err.abs();
            ss_res[k] += err * err;
            let d = t[k] - mean_t[k];
            ss_tot[k] += d * d;
        }
    }

    let mut r2 = [0.0; 3];
    for k in 0..3 {
        mae[k] /= n;
        r2[k] = if ss_tot[k] > 1e-12 {
            1.0 - ss_res[k] / ss_tot[k]
        } else {
            0.0
        };
    }

    (mae, r2)
}

// ── Timestamps ────────────────────────────────────────────────────────────────

/// Current UTC time as RFC 3339 (`2026-10-22T15:04:05Z`).
///
/// Replaces the per-binary `chrono_now` helpers, which emitted a hard-coded
/// `2026-10-13` date.
pub fn utc_timestamp_rfc3339() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    rfc3339_from_unix(secs)
}

/// RFC 3339 for `secs` since the Unix epoch (proleptic Gregorian, UTC).
pub fn rfc3339_from_unix(secs: u64) -> String {
    let (days, rem) = ((secs / 86_400) as i64, secs % 86_400);
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rem / 3_600,
        (rem / 60) % 60,
        rem % 60
    )
}

// ── Holdout + checkpoint loading ──────────────────────────────────────────────

/// Loads the 2,000-molecule holdout set and the 30-epoch checkpoint, printing progress.
/// Prints the reason and returns `None` if either is unusable.
pub fn load_holdout_and_checkpoint() -> Option<(Vec<crate::zinc_data::ZincMolecule>, QnnCheckpoint)>
{
    let test_path =
        resolve_workspace_path("tests/benchmarks/zinc_molecular/outputs/zinc_test_2k.csv");
    let ckpt_path = resolve_workspace_path(
        "tests/benchmarks/zinc_molecular/outputs/zinc_12qubit_checkpoint.json",
    );

    println!(
        "[INIT] Loading ZINC 2,000 holdout molecules: {}",
        test_path.display()
    );
    let test_samples = match crate::zinc_data::load(&test_path, 2000) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[-] Error: {}", e);
            return None;
        }
    };
    println!("  -> Loaded {} molecules.\n", test_samples.len());

    println!(
        "[INIT] Loading 30-epoch checkpoint: {}",
        ckpt_path.display()
    );
    let ckpt_content = match std::fs::read_to_string(&ckpt_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[-] Error reading checkpoint: {}", e);
            return None;
        }
    };
    match serde_json::from_str(&ckpt_content) {
        Ok(ckpt) => Some((test_samples, ckpt)),
        Err(e) => {
            eprintln!("[-] Error parsing checkpoint: {}", e);
            None
        }
    }
}

/// Pruned parameter vector from the full 111: layers 1 & 2 keep their 48 angles, layer 3 keeps
/// only its first `n_ry` Ry angles (12 = canonical pruning, 11 = qubit 11 eliminated), and the
/// readout keeps its 36 weights and 3 biases.
pub fn derive_pruned_params(full_params: &[f64], n_ry: usize) -> Vec<f64> {
    let mut p = Vec::with_capacity(48 + n_ry + 39);
    p.extend_from_slice(&full_params[0..48]);
    for q in 0..n_ry {
        p.push(full_params[48 + 2 * q]);
    }
    p.extend_from_slice(&full_params[72..111]);
    p
}

// ── Epoch telemetry ───────────────────────────────────────────────────────────

/// Prints a single formatted training epoch row to stdout.
///
/// ```text
///   Epoch  1/20 | train 0.842 | test 0.871 | lr 1.000e-3 | 12.34 s
/// ```
pub fn print_epoch_row(
    epoch: usize,
    total_epochs: usize,
    train_loss: f64,
    test_loss: f64,
    lr: f64,
    elapsed_s: f64,
) {
    println!(
        "  Epoch {:>3}/{} | train {:.4} | test {:.4} | lr {:.3e} | {:.2} s",
        epoch, total_epochs, train_loss, test_loss, lr, elapsed_s
    );
}

// ── Checkpoint I/O ────────────────────────────────────────────────────────────

/// Serializes `value` as pretty-printed JSON and writes it via `t1000::results::write`.
///
/// Returns the JSON string on success so callers can also print or inspect it.
pub fn save_checkpoint<T: serde::Serialize>(path: &Path, value: &T) -> Result<String, String> {
    let json = serde_json::to_string_pretty(value)
        .map_err(|e| format!("Failed to serialize checkpoint: {e}"))?;
    crate::results::write(path, &json)
        .map_err(|e| format!("Failed to write checkpoint to {}: {e}", path.display()))?;
    Ok(json)
}

/// Deserializes a JSON checkpoint from `path`.
pub fn load_checkpoint<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read checkpoint {}: {e}", path.display()))?;
    serde_json::from_str(&text)
        .map_err(|e| format!("Failed to parse checkpoint {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc3339_known_instants() {
        assert_eq!(rfc3339_from_unix(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339_from_unix(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(rfc3339_from_unix(1_700_000_000), "2023-11-14T22:13:20Z");
        assert_eq!(rfc3339_from_unix(1_709_164_800), "2024-02-29T00:00:00Z");
        assert_eq!(rfc3339_from_unix(1_735_689_599), "2024-12-31T23:59:59Z");
        assert_eq!(utc_timestamp_rfc3339().len(), "2026-10-22T15:04:05Z".len());
    }

    #[test]
    fn workspace_paths_resolve_from_any_cwd() {
        let root = workspace_root().expect("workspace root");
        assert!(root.join("Cargo.toml").is_file());
        let rel = "tests/benchmarks/zinc_molecular/outputs/zinc_test_2k.csv";
        assert_eq!(resolve_workspace_path(rel), root.join(rel));
        let out = "tests/benchmarks/zinc_molecular/outputs/not_written_yet.json";
        assert_eq!(resolve_workspace_path(out), root.join(out));
    }

    #[test]
    fn ridge_recovers_an_exact_linear_map() {
        // y = 2 x0 - x1 + 0.5 and y' = -x0 + 3, no noise, tiny lambda.
        let xs: Vec<Vec<f64>> = (0..20)
            .map(|i| vec![i as f64 * 0.1, (i % 7) as f64])
            .collect();
        let ys: Vec<Vec<f64>> = xs
            .iter()
            .map(|x| vec![2.0 * x[0] - x[1] + 0.5, -x[0] + 3.0])
            .collect();
        let (w, b) = solve_ridge_regression(&xs, &ys, 1e-12);
        assert!((w[0][0] - 2.0).abs() < 1e-6 && (w[1][0] + 1.0).abs() < 1e-6);
        assert!((b[0] - 0.5).abs() < 1e-6 && (b[1] - 3.0).abs() < 1e-6);
        assert!((w[0][1] + 1.0).abs() < 1e-6 && w[1][1].abs() < 1e-6);
        let rows: Vec<Vec<f64>> = ys.iter().map(|y| vec![y[0], y[1], 1.0]).collect();
        let (mae, r2) = compute_mae_and_r2(&rows, &rows);
        assert_eq!((mae[0], r2[0], r2[2]), (0.0, 1.0, 0.0));
    }

    #[test]
    fn pruning_keeps_layers_1_2_ry_of_layer_3_and_the_readout() {
        let full: Vec<f64> = (0..111).map(|i| i as f64).collect();
        let pruned = derive_pruned_params(&full, 12);
        assert_eq!(pruned.len(), 99);
        assert_eq!(&pruned[..48], &full[..48]);
        assert_eq!(pruned[48 + 5], full[48 + 2 * 5]); // Ry of qubit 5, Rz skipped
        assert_eq!(&pruned[60..], &full[72..111]);
        assert_eq!(derive_pruned_params(&full, 11).len(), 98);
    }

    #[test]
    fn checkpoint_contracts_accept_older_files() {
        // reuploading reads mid-circuit files that may lack the readout heads.
        let json = r#"{"timestamp_rfc3339":"t","n_qubits":12,"depth":5,"source_qubit":0,
                       "circuit_params":[0.1],"feedforward_alphas":[0.2]}"#;
        let ck: MidCircuitCheckpoint = serde_json::from_str(json).unwrap();
        assert!(ck.readout_weights_12.is_empty() && ck.readout_biases_78.is_empty());
        let pruned = r#"{"timestamp_rfc3339":"t","n_qubits":12,"depth":5,"full_params":60,
                         "pruned_params":40,"epochs_completed":3,"pruned_hybrid_params":[1.0]}"#;
        let ck: PrunedCheckpoint = serde_json::from_str(pruned).unwrap();
        assert_eq!((ck.gate_count_full, ck.pruned_params), (0, 40));
    }
}
