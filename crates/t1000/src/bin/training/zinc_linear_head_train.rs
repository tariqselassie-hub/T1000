// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! ZINC property regression with a fixed circuit and a closed-form linear head.
//!
//! Each molecule is amplitude-encoded, run through the circuit, and read out as the Z and ZZ
//! expectation values of the output state (all from one computational-basis measurement). A ridge
//! regression fitted on the training molecules maps those features to logP, QED and SAS. There is
//! no gradient training: the circuit is either the default initialisation or the weights stored
//! in a `zinc_qnn_train` report, and the head is exactly reproducible.
//!
//! Usage: `zinc_linear_head_train [--depth N] [--weights report.json] [--observables w4|wK|all|z]
//!         [--scale maxabs|minmax|raw] [--samples N] [--eval-samples N] [--record]`

use serde::{Deserialize, Serialize};
use std::{env, fs, time::Instant};
use t1000::baselines::{print_comparison, regression_metrics, zinc_baselines, BaselineResult, QnnScore, TargetMetrics, ZINC_TARGETS};
use t1000::holdout_score::ZincHoldout;
use t1000::linear_head::{feature_spreads, shots_for_noise_equal_to, weights_fingerprint, CircuitFeatureMap, LinearHead, Observables};
use t1000::zinc_data::{self, InputScaling, Scaler, ZincMolecule};
use t1000::{CircuitAnsatz, DataEncodingLayer, EncodingKind};

const N_QUBITS: usize = 10;
const TRAIN_CSV: &str = "tests/benchmarks/zinc_molecular/outputs/zinc_train_10k.csv";
const TEST_CSV: &str = "tests/benchmarks/zinc_molecular/outputs/zinc_test_2k.csv";
const REPORT_PATH: &str = "tests/benchmarks/zinc_molecular/outputs/zinc_linear_head_results.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LinearHeadReport {
    host_cpu: String,
    host_ram_gb: f64,
    n_qubits: usize,
    circuit_depth: usize,
    circuit_weights_source: String,
    n_observables: usize,
    observables: String,
    train_sample_count: usize,
    test_sample_count: usize,
    ridge_lambda: f64,
    feature_time_ms: f64,
    fit_time_ms: f64,
    per_molecule_latency_us: f64,
    /// Median standard deviation of the readout features across training molecules.
    median_feature_spread: f64,
    /// Shots per molecule at which sampling noise equals that spread (an estimate).
    shots_for_median_spread: f64,
    scaler: Scaler,
    metrics: Vec<TargetMetrics>,
    relative_mse: f64,
    mean_pearson_r: f64,
    classical_baselines: Vec<BaselineResult>,
    circuit_weights: Vec<f64>,
    head: LinearHead,
}

struct Options {
    depth: usize,
    weights_path: Option<String>,
    observables: Observables,
    scaling: InputScaling,
    samples: usize,
    eval_samples: usize,
}

fn options() -> Options {
    let args: Vec<String> = env::args().collect();
    let value = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned();
    Options {
        depth: value("--depth").and_then(|v| v.parse().ok()).unwrap_or(4),
        weights_path: value("--weights"),
        observables: match value("--observables").as_deref() {
            Some("z") => Observables::SingleZ,
            Some("all") => Observables::ZAndZz,
            Some(w) if w.starts_with('w') => Observables::ParityUpTo(w[1..].parse().unwrap_or(4)),
            _ => Observables::ParityUpTo(4),
        },
        scaling: match value("--scale").as_deref() {
            Some("raw") => InputScaling::Raw,
            Some("minmax") => InputScaling::MinMax,
            _ => InputScaling::MaxAbs,
        },
        samples: value("--samples").and_then(|v| v.parse().ok()).unwrap_or(10_000),
        eval_samples: value("--eval-samples").and_then(|v| v.parse().ok()).unwrap_or(2_000),
    }
}

/// Circuit weights from a `zinc_qnn_train` report, or the ansatz default.
fn circuit_weights(opts: &Options, ansatz: &CircuitAnsatz) -> (Vec<f64>, String) {
    let Some(path) = &opts.weights_path else {
        return (ansatz.weights.clone(), "default initialisation".to_string());
    };
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let report: serde_json::Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("cannot parse {path}: {e}"));
    let weights: Vec<f64> = serde_json::from_value(report["final_weights"].clone())
        .unwrap_or_else(|_| panic!("{path} has no final_weights (re-run zinc_qnn_train to store them)"));
    assert_eq!(weights.len(), ansatz.param_count(), "{path} was trained with a different depth (--depth)");
    (weights, format!("trained circuit from {path}"))
}

fn load(path: &str, max: usize) -> Vec<ZincMolecule> {
    zinc_data::load(path, max).unwrap_or_else(|e| panic!("{e}"))
}

fn main() {
    let opts = options();
    let host = t1000::hostinfo::HostProfile::detect();
    println!("=========================================================================");
    println!("   ZINC QUANTUM FEATURE MAP + CLOSED-FORM LINEAR HEAD                     ");
    println!("=========================================================================");
    println!(" Host            : {}", host.label());

    let train = load(TRAIN_CSV, opts.samples);
    let test = load(TEST_CSV, opts.eval_samples);
    let encoding = DataEncodingLayer::new(N_QUBITS, EncodingKind::Amplitude);
    let ansatz = CircuitAnsatz::new(N_QUBITS, opts.depth).with_pi_layer(true);
    let (weights, source) = circuit_weights(&opts, &ansatz);
    let map = CircuitFeatureMap { encoding: &encoding, ansatz: &ansatz, weights: &weights, observables: opts.observables };
    println!(" Circuit         : {N_QUBITS} qubits, depth {}, {source}", opts.depth);
    println!(" Readout         : {} observables ({:?}), ridge head fitted in closed form", map.n_features(), opts.observables);
    println!(" Input scaling   : {:?}", opts.scaling);
    println!(" Molecules       : {} training, {} held-out\n", train.len(), test.len());

    let t = Instant::now();
    let scaler = Scaler::fit(opts.scaling, &train);
    let scaled_train: Vec<Vec<f64>> = train.iter().map(|m| scaler.apply(&m.features)).collect();
    let scaled_test: Vec<Vec<f64>> = test.iter().map(|m| scaler.apply(&m.features)).collect();
    let train_x = map.feature_matrix(scaled_train.iter().map(Vec::as_slice));
    let train_ms = t.elapsed().as_secs_f64() * 1e3;
    let train_y: Vec<Vec<f64>> = train.iter().map(|m| m.targets().to_vec()).collect();

    let t = Instant::now();
    let head = LinearHead::fit(&train_x, &train_y);
    let fit_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let test_x = map.feature_matrix(scaled_test.iter().map(Vec::as_slice));
    let preds = head.predict(&test_x);
    let eval_ms = t.elapsed().as_secs_f64() * 1e3;
    let test_y: Vec<Vec<f64>> = test.iter().map(|m| m.targets().to_vec()).collect();
    let metrics = regression_metrics(&preds, &test_y, &ZINC_TARGETS);

    let target_cols: Vec<Vec<f64>> = (0..3).map(|k| test_y.iter().map(|r| r[k]).collect()).collect();
    let holdout = ZincHoldout::new(
        [metrics[0].mse, metrics[1].mse, metrics[2].mse],
        [metrics[0].pearson_r, metrics[1].pearson_r, metrics[2].pearson_r],
        [&target_cols[0], &target_cols[1], &target_cols[2]],
    );

    println!("  Feature extraction : {train_ms:>8.1} ms for {} training molecules", train.len());
    println!("  Head fit           : {fit_ms:>8.1} ms (ridge lambda {:e})", head.lambda());
    println!("  Held-out inference : {eval_ms:>8.1} ms ({:.1} us/molecule including the circuit)", eval_ms * 1e3 / test.len() as f64);
    let mut spreads = feature_spreads(&train_x);
    spreads.sort_by(f64::total_cmp);
    let median_spread = spreads[spreads.len() / 2];
    let shots = shots_for_noise_equal_to(median_spread);
    println!("  Feature spread     : median {median_spread:.2e} (min {:.1e}, max {:.1e}); sampling noise equals it at ~{shots:.1e} shots", spreads[0], spreads[spreads.len() - 1]);
    println!("\n  Target    MAE       MSE       R2      Pearson r");
    for m in &metrics {
        println!("  {:<6} {:>8.4}  {:>8.4}  {:>7.4}  {:>8.4}", m.target, m.mae, m.mse, m.r2, m.pearson_r);
    }
    println!("\n  Held-out relative MSE {:.4} | mean Pearson r {:.4}", holdout.relative_mse, holdout.mean_pearson_r);

    let baselines = zinc_baselines(
        train.iter().map(|m| (m.features.as_slice(), m.targets())),
        test.iter().map(|m| (m.features.as_slice(), m.targets())),
    );
    let col = |f: fn(&TargetMetrics) -> f64| [f(&metrics[0]), f(&metrics[1]), f(&metrics[2])];
    let row = QnnScore::new("circuit + linear head")
        .mae(col(|m| m.mae))
        .mse(col(|m| m.mse))
        .r2(col(|m| m.r2))
        .pearson_r(col(|m| m.pearson_r));
    print_comparison(&[row], &baselines);

    let report = LinearHeadReport {
        host_cpu: host.label(),
        host_ram_gb: host.ram_gb,
        n_qubits: N_QUBITS,
        circuit_depth: opts.depth,
        circuit_weights_source: source,
        n_observables: map.n_features(),
        observables: format!("{:?}", opts.observables),
        train_sample_count: train.len(),
        test_sample_count: test.len(),
        ridge_lambda: head.lambda(),
        feature_time_ms: train_ms,
        fit_time_ms: fit_ms,
        per_molecule_latency_us: eval_ms * 1e3 / test.len() as f64,
        median_feature_spread: median_spread,
        shots_for_median_spread: shots,
        scaler,
        metrics,
        relative_mse: holdout.relative_mse,
        mean_pearson_r: holdout.mean_pearson_r,
        classical_baselines: baselines,
        circuit_weights: weights.clone(),
        head,
    };
    let settings = format!(
        "qubits={N_QUBITS};depth={};weights={};obs={:?};scale={:?};train={};test={}",
        opts.depth,
        weights_fingerprint(&weights),
        opts.observables,
        opts.scaling,
        train.len(),
        test.len()
    );
    let json = serde_json::to_string_pretty(&report).expect("serialize report");
    match t1000::results::write_scored(REPORT_PATH, json, &holdout.score(&settings)) {
        Ok(t1000::results::Outcome::KeptBest) => println!("\n[=] Saved report kept: this run did not improve it"),
        Ok(t1000::results::Outcome::Scratch) => println!("\n[~] Report written to a scratch copy only (use --record to save)"),
        Ok(t1000::results::Outcome::WrittenVariant) => println!("\n[SAVED] Different settings from the saved report: recorded as a variant"),
        Ok(t1000::results::Outcome::Written) => println!("\n[SAVED] Report exported to: {REPORT_PATH}"),
        Err(e) => eprintln!("\n[!] Could not write the report: {e}"),
    }
}
