// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Trains the variational quantum circuit on the Titanic cross-validation folds.
//!
//! ```text
//! titanic_vqc --dir titanic/vqc_folds [--qubits 8] [--depth 3] [--epochs 30] [--lr 0.05]
//! ```
//!
//! For every `fold_k_train.csv` it trains a fresh circuit (angle encoding, hardware-efficient
//! Ry/Rz + CZ ring, mean-Z readout, exact parameter shift, full-batch Adam) and writes the
//! readout of every row of `fold_k_eval.csv` to `fold_k_pred.csv`. The scoring against the classical
//! models is done by `titanic/vqc_compare.py`, on the same folds and the same features.

use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use t1000::{
    AdamOptimizer, CircuitAnsatz, DataEncodingLayer, EncodingKind, ParameterShiftEngine,
    PauliObservable,
};

fn read_csv(path: &Path, n_features: usize) -> Vec<(Vec<f64>, f64)> {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: Vec<f64> = l.split(',').map(|t| t.trim().parse().expect("number")).collect();
            assert_eq!(v.len(), n_features + 1, "{}: expected {} columns", path.display(), n_features + 1);
            (v[..n_features].to_vec(), v[n_features])
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str, default: &str| -> String {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1).cloned())
            .unwrap_or_else(|| default.to_string())
    };
    let default_dir = if Path::new("titanic-contest/data/vqc_folds").exists() {
        "titanic-contest/data/vqc_folds"
    } else if Path::new("titanic-contest/vqc_folds").exists() {
        "titanic-contest/vqc_folds"
    } else {
        "titanic/vqc_folds"
    };
    let dir = PathBuf::from(get("--dir", default_dir));
    let n_qubits: usize = get("--qubits", "8").parse().expect("--qubits");
    let depth: usize = get("--depth", "3").parse().expect("--depth");
    let epochs: usize = get("--epochs", "30").parse().expect("--epochs");
    let lr: f64 = get("--lr", "0.05").parse().expect("--lr");
    let seed: u64 = get("--seed", "11").parse().expect("--seed");

    let mut fold = 0;
    while dir.join(format!("fold_{fold}_train.csv")).exists() {
        let t0 = Instant::now();
        let train = read_csv(&dir.join(format!("fold_{fold}_train.csv")), n_qubits);
        let eval = read_csv(&dir.join(format!("fold_{fold}_eval.csv")), n_qubits);

        let encoding = DataEncodingLayer::new(n_qubits, EncodingKind::Angle);
        let ansatz = CircuitAnsatz::new_seeded(n_qubits, depth, seed + fold as u64);
        let n_params = ansatz.param_count();
        let mut weights = ansatz.weights.clone();
        let mut adam = AdamOptimizer::new(n_params, lr);
        let engine = ParameterShiftEngine::new(encoding, ansatz, PauliObservable::GlobalMeanZ);

        let mut loss = 0.0;
        for ep in 0..epochs {
            adam.step_epoch_decay(ep, 0.9, 10);
            let (sum_loss, sum_grad) = train
                .par_iter()
                .map(|(x, y)| {
                    let (_, l, g) = engine.compute_loss_gradients(x, *y, &weights);
                    (l, g)
                })
                .reduce(
                    || (0.0, vec![0.0; n_params]),
                    |mut a, b| {
                        a.0 += b.0;
                        for (u, v) in a.1.iter_mut().zip(b.1) {
                            *u += v;
                        }
                        a
                    },
                );
            let n = train.len() as f64;
            loss = sum_loss / n;
            let grad: Vec<f64> = sum_grad.iter().map(|g| g / n).collect();
            adam.step(&mut weights, &grad);
        }

        let preds: Vec<f64> = eval.par_iter().map(|(x, _)| engine.predict(x, &weights)).collect();
        let train_acc = train
            .par_iter()
            .filter(|(x, y)| (engine.predict(x, &weights) >= 0.0) == (*y > 0.0))
            .count() as f64
            / train.len() as f64;
        let out: String = std::iter::once("pred\n".to_string())
            .chain(preds.iter().map(|p| format!("{p:.9}\n")))
            .collect();
        fs::write(dir.join(format!("fold_{fold}_pred.csv")), out).expect("write predictions");
        println!(
            "fold {fold}: {} train, {} eval, {n_params} parameters, final loss {loss:.4}, train acc {:.3}, {:.1} s",
            train.len(),
            eval.len(),
            train_acc,
            t0.elapsed().as_secs_f64()
        );
        fold += 1;
    }
    assert!(fold > 0, "no fold_0_train.csv in {}", dir.display());
}
