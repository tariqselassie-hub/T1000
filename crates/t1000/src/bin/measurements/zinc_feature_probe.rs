// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Diagnostic: how much of the ZINC signal survives amplitude encoding and the circuit?
//!
//! The same ridge regression is fitted to several feature sets on one train/test split:
//! the raw molecule features (the classical reference), the amplitudes the circuit actually
//! receives (L2-normalised features), and the Z and ZZ expectation values of the encoded state
//! and of the circuit output. Comparing the rows shows where information is lost.
//!
//! Usage: `zinc_feature_probe [--weights report.json] [--samples N] [--eval-samples N]`
//! where `report.json` is a `zinc_qnn_train` report containing `final_weights`.

use serde::Serialize;
use std::{env, fs};
use t1000::baselines::{fit_ridge_validated, quadratic_features, regression_metrics};
use t1000::linear_head::{z_features_into, Observables};
use t1000::zinc_data::{self, ZincMolecule};
use t1000::{CircuitAnsatz, DataEncodingLayer, EncodingKind, QuantumState};

const N_QUBITS: usize = 10;
const DEPTH: usize = 4;
const LAMBDAS: [f64; 7] = [1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0];
const TARGETS: [&str; 3] = ["logP", "QED", "SAS"];
const TRAIN_CSV: &str = "tests/benchmarks/zinc_molecular/outputs/zinc_train_10k.csv";
const TEST_CSV: &str = "tests/benchmarks/zinc_molecular/outputs/zinc_test_2k.csv";

/// Feature matrix: one row per molecule.
type Matrix = Vec<Vec<f64>>;
/// A named feature set: train rows, held-out rows, and whether to also fit a quadratic expansion.
type FeatureSet = (String, Matrix, Matrix, bool);

#[derive(Serialize)]
struct Row {
    feature_set: String,
    n_features: usize,
    model: String,
    relative_mse: f64,
    r2: [f64; 3],
    pearson_r: [f64; 3],
}

/// Per-feature mean and standard deviation of the training set.
fn moments(rows: &[ZincMolecule]) -> (Vec<f64>, Vec<f64>) {
    let n = rows.len() as f64;
    let k = rows[0].features.len();
    let mean: Vec<f64> = (0..k)
        .map(|j| rows.iter().map(|m| m.features[j]).sum::<f64>() / n)
        .collect();
    let std = (0..k)
        .map(|j| {
            let v = rows
                .iter()
                .map(|m| (m.features[j] - mean[j]).powi(2))
                .sum::<f64>()
                / n;
            v.sqrt().max(1e-12)
        })
        .collect();
    (mean, std)
}

/// `Z_q` (10 values) followed by `Z_q Z_r` for `q < r` (45 values) of a state.
fn z_zz_features(state: &QuantumState) -> Vec<f64> {
    let mut row = vec![0.0; Observables::ZAndZz.count(N_QUBITS)];
    z_features_into(state, Observables::ZAndZz, &mut row);
    row
}

/// Fraction of the squared norm carried by the largest feature, averaged over molecules.
fn dominance(inputs: &[Vec<f64>]) -> f64 {
    let share = |x: &Vec<f64>| {
        let total: f64 = x.iter().map(|v| v * v).sum();
        x.iter().map(|v| v * v).fold(0.0, f64::max) / total.max(1e-300)
    };
    inputs.iter().map(share).sum::<f64>() / inputs.len() as f64
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
    };
    let n_train: usize = value("--samples")
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000);
    let n_test: usize = value("--eval-samples")
        .and_then(|v| v.parse().ok())
        .unwrap_or(2_000);
    let train = zinc_data::load(TRAIN_CSV, n_train).expect("load training set");
    let test = zinc_data::load(TEST_CSV, n_test).expect("load held-out set");
    let train_y: Vec<Vec<f64>> = train.iter().map(|m| m.targets().to_vec()).collect();
    let test_y: Vec<Vec<f64>> = test.iter().map(|m| m.targets().to_vec()).collect();

    let (mean, std) = moments(&train);
    let standardize = |m: &ZincMolecule| -> Vec<f64> {
        m.features
            .iter()
            .enumerate()
            .map(|(j, v)| (v - mean[j]) / std[j])
            .collect()
    };
    let raw = |m: &ZincMolecule| m.features.clone();
    let std_inputs = |m: &ZincMolecule| standardize(m);

    let encoding = DataEncodingLayer::new(N_QUBITS, EncodingKind::Amplitude);
    let ansatz = CircuitAnsatz::new(N_QUBITS, DEPTH).with_pi_layer(true);
    let trained: Option<Vec<f64>> = value("--weights").map(|path| {
        let report: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(path).expect("read weights report"))
                .expect("parse weights report");
        let w: Vec<f64> =
            serde_json::from_value(report["final_weights"].clone()).expect("final_weights");
        assert_eq!(
            w.len(),
            ansatz.param_count(),
            "weights do not match the {DEPTH}-layer ansatz"
        );
        w
    });

    println!(
        "Preparing feature sets for {} training and {} held-out molecules...",
        train.len(),
        test.len()
    );
    let amplitudes =
        |input: &dyn Fn(&ZincMolecule) -> Vec<f64>, rows: &[ZincMolecule]| -> Vec<Vec<f64>> {
            rows.iter()
                .map(|m| {
                    let state = encoding.encode(&input(m));
                    state.real[..m.features.len()].to_vec()
                })
                .collect()
        };
    let circuit = |input: &dyn Fn(&ZincMolecule) -> Vec<f64>,
                   weights: Option<&[f64]>,
                   rows: &[ZincMolecule]|
     -> Vec<Vec<f64>> {
        let mut out = QuantumState::new(encoding.dim);
        rows.iter()
            .map(|m| {
                let state = encoding.encode(&input(m));
                match weights {
                    Some(w) => {
                        ansatz.forward_into(&state, w, &mut out);
                        z_zz_features(&out)
                    }
                    None => z_zz_features(&state),
                }
            })
            .collect()
    };
    let untrained = ansatz.weights.clone();

    // (name, train features, test features, also fit the quadratic expansion?)
    let mut sets: Vec<FeatureSet> = vec![
        (
            "raw features".into(),
            train.iter().map(raw).collect(),
            test.iter().map(raw).collect(),
            true,
        ),
        (
            "encoded amplitudes, raw inputs".into(),
            amplitudes(&raw, &train),
            amplitudes(&raw, &test),
            true,
        ),
        (
            "encoded amplitudes, standardized inputs".into(),
            amplitudes(&std_inputs, &train),
            amplitudes(&std_inputs, &test),
            true,
        ),
        (
            "Z/ZZ of encoded state, raw inputs".into(),
            circuit(&raw, None, &train),
            circuit(&raw, None, &test),
            false,
        ),
        (
            "Z/ZZ after untrained circuit, raw inputs".into(),
            circuit(&raw, Some(&untrained), &train),
            circuit(&raw, Some(&untrained), &test),
            false,
        ),
    ];
    if let Some(w) = &trained {
        sets.push((
            "Z/ZZ after trained circuit, raw inputs".into(),
            circuit(&raw, Some(w), &train),
            circuit(&raw, Some(w), &test),
            false,
        ));
    }
    // The model's readout sees only Z0..Z2; these rows show what a linear head on fewer
    // observables of the same circuits can reach.
    let leading = |x: &[Vec<f64>], k: usize| x.iter().map(|r| r[..k].to_vec()).collect::<Vec<_>>();
    let untrained_cols = [(
        circuit(&raw, Some(&untrained), &train),
        circuit(&raw, Some(&untrained), &test),
        "untrained",
    )];
    let trained_cols: Vec<_> = trained
        .iter()
        .map(|w| {
            (
                circuit(&raw, Some(w), &train),
                circuit(&raw, Some(w), &test),
                "trained",
            )
        })
        .collect();
    for (xt, xe, which) in untrained_cols.iter().chain(trained_cols.iter()) {
        for (k, what) in [(3, "Z0..Z2 only"), (10, "all 10 single-qubit Z")] {
            sets.push((
                format!("{what}, {which} circuit, raw inputs"),
                leading(xt, k),
                leading(xe, k),
                false,
            ));
        }
    }
    sets.push((
        "Z/ZZ of encoded state, standardized inputs".into(),
        circuit(&std_inputs, None, &train),
        circuit(&std_inputs, None, &test),
        false,
    ));
    sets.push((
        "Z/ZZ after untrained circuit, standardized inputs".into(),
        circuit(&std_inputs, Some(&untrained), &train),
        circuit(&std_inputs, Some(&untrained), &test),
        false,
    ));

    let raw_dom = dominance(&train.iter().map(raw).collect::<Vec<_>>());
    let std_dom = dominance(&train.iter().map(std_inputs).collect::<Vec<_>>());
    println!("Largest-feature share of the squared norm (what amplitude encoding keeps):");
    println!(
        "  raw inputs {:.4}   standardized inputs {:.4}\n",
        raw_dom, std_dom
    );

    let mut rows = Vec::new();
    for (name, xt, xe, quadratic) in &sets {
        let mut variants: Vec<(&str, Matrix, Matrix)> =
            vec![("linear ridge", xt.clone(), xe.clone())];
        if *quadratic {
            let q = |x: &Vec<Vec<f64>>| x.iter().map(|r| quadratic_features(r)).collect::<Vec<_>>();
            variants.push(("quadratic ridge", q(xt), q(xe)));
        }
        for (model, a, b) in variants {
            let fit = fit_ridge_validated(&a, &train_y, &LAMBDAS);
            let m = regression_metrics(&fit.predict(&b), &test_y, &TARGETS);
            let r2 = [m[0].r2, m[1].r2, m[2].r2];
            rows.push(Row {
                feature_set: name.clone(),
                n_features: xt[0].len(),
                model: model.into(),
                relative_mse: 1.0 - r2.iter().sum::<f64>() / 3.0,
                r2,
                pearson_r: [m[0].pearson_r, m[1].pearson_r, m[2].pearson_r],
            });
        }
    }

    println!("┌──────────────────────────────────────────────────┬─────┬─────────────────┬──────────┬────────┬────────┬────────┐");
    println!("│ Feature set                                      │  n  │ Model           │ Rel. MSE │ R2 logP│ R2 QED │ R2 SAS │");
    println!("├──────────────────────────────────────────────────┼─────┼─────────────────┼──────────┼────────┼────────┼────────┤");
    for r in &rows {
        println!(
            "│ {:<48} │ {:>3} │ {:<15} │ {:>8.4} │ {:>6.3} │ {:>6.3} │ {:>6.3} │",
            r.feature_set, r.n_features, r.model, r.relative_mse, r.r2[0], r.r2[1], r.r2[2]
        );
    }
    println!("└──────────────────────────────────────────────────┴─────┴─────────────────┴──────────┴────────┴────────┴────────┘");

    if let Ok(json) = serde_json::to_string_pretty(&rows) {
        let _ = t1000::results::write("results/experiments/zinc_feature_probe.json", json);
    }
}
