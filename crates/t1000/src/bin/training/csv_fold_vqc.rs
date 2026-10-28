// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Trains a variational quantum circuit on cross-validation folds stored as CSV files.
//!
//! ```text
//! csv_fold_vqc --dir <folds> [--qubits 8] [--depth 3] [--epochs 30] [--lr 0.05] [--seed 11]
//! ```
//!
//! `<folds>` holds `fold_0_train.csv`, `fold_0_eval.csv`, `fold_1_train.csv`, ... Each file has a
//! header row and then one row per sample: `--qubits` feature columns (angles, typically scaled to
//! `[0, pi]`) followed by a label of `+1` or `-1`. For every fold the binary trains a fresh circuit
//! (`t1000::vqc_classifier`: angle encoding, hardware-efficient Ry/Rz + CZ ring, mean-Z readout,
//! exact gradients, full-batch Adam) and writes the readout of every `fold_k_eval.csv` row to
//! `fold_k_pred.csv` in the same folder. Scoring the predictions is left to the caller (for
//! example the Titanic project's `vqc_compare.py`). The Python bindings (`t1000_py`) expose the
//! same classifier without the CSV files.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use t1000::vqc_classifier::{train_vqc, VqcConfig};

fn read_csv(path: &Path, n_features: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: Vec<f64> = l
                .split(',')
                .map(|t| t.trim().parse().expect("number"))
                .collect();
            assert_eq!(
                v.len(),
                n_features + 1,
                "{}: expected {} columns",
                path.display(),
                n_features + 1
            );
            (v[..n_features].to_vec(), v[n_features])
        })
        .unzip()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str, default: &str| -> String {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1).cloned())
            .unwrap_or_else(|| default.to_string())
    };
    let Some(dir) = args
        .iter()
        .position(|a| a == "--dir")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
    else {
        eprintln!("usage: csv_fold_vqc --dir <folds> [--qubits 8] [--depth 3] [--epochs 30] [--lr 0.05] [--seed 11]");
        std::process::exit(2);
    };
    let base = VqcConfig {
        n_qubits: get("--qubits", "8").parse().expect("--qubits"),
        depth: get("--depth", "3").parse().expect("--depth"),
        epochs: get("--epochs", "30").parse().expect("--epochs"),
        lr: get("--lr", "0.05").parse().expect("--lr"),
        seed: get("--seed", "11").parse().expect("--seed"),
    };

    let mut fold = 0;
    while dir.join(format!("fold_{fold}_train.csv")).exists() {
        let t0 = Instant::now();
        let (train_x, train_y) =
            read_csv(&dir.join(format!("fold_{fold}_train.csv")), base.n_qubits);
        let (eval_x, _) = read_csv(&dir.join(format!("fold_{fold}_eval.csv")), base.n_qubits);
        // Each fold starts from its own seed, as before the training loop moved into the library.
        let cfg = VqcConfig {
            seed: base.seed + fold as u64,
            ..base
        };
        let model = train_vqc(&train_x, &train_y, &cfg);
        let preds = model.decisions(&eval_x);
        let train_acc = model
            .decisions(&train_x)
            .iter()
            .zip(&train_y)
            .filter(|(d, y)| (**d >= 0.0) == (**y > 0.0))
            .count() as f64
            / train_x.len() as f64;
        let out: String = std::iter::once("pred\n".to_string())
            .chain(preds.iter().map(|p| format!("{p:.9}\n")))
            .collect();
        fs::write(dir.join(format!("fold_{fold}_pred.csv")), out).expect("write predictions");
        println!(
            "fold {fold}: {} train, {} eval, {} parameters, final loss {:.4}, train acc {:.3}, {:.1} s",
            train_x.len(),
            eval_x.len(),
            model.parameter_count(),
            model.final_loss,
            train_acc,
            t0.elapsed().as_secs_f64()
        );
        fold += 1;
    }
    assert!(fold > 0, "no fold_0_train.csv in {}", dir.display());
}
