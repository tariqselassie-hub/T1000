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
//! `--dataset qm9` runs the same model on the QM9 DFT properties (`data/qm9/qm9.csv`, see
//! `data/qm9/README.md`): `--targets mu,gap,alpha` picks three of the nine properties and
//! `--features zinc|qm9` the descriptor set.
//!
//! Usage: `zinc_linear_head_train [--depth N] [--weights report.json] [--observables w4|wK|all|z]
//!         [--scale maxabs|minmax|raw] [--poly-baseline D] [--poly-normalized D] [--append-raw] [--qubits N] [--samples N] [--eval-samples N] [--record]`

use serde::{Deserialize, Serialize};
use std::{env, fs, time::Instant};
use t1000::baselines::{evaluate_baselines, print_comparison_named, regression_metrics, BaselineResult, QnnScore, TargetMetrics};
use t1000::datasets::{Data, DataOptions, Dataset, Sample};
use t1000::holdout_score::ZincHoldout;
use t1000::polynomial::polynomial_features;
use t1000::linear_head::{feature_spreads, shots_for_noise_equal_to, weights_fingerprint, CircuitFeatureMap, LinearHead, Observables};
use t1000::zinc_data::{InputScaling, Scaler};
use t1000::{CircuitAnsatz, DataEncodingLayer, EncodingKind};

const DEFAULT_QUBITS: usize = 10;
const ZINC_REPORT_PATH: &str = "tests/benchmarks/zinc_molecular/outputs/zinc_linear_head_results.json";
const QM9_REPORT_PATH: &str = "results/experiments/qm9_linear_head_results.json";

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
    data: DataOptions,
    /// Also fit a classical ridge on all monomials up to this degree of the scaled descriptors.
    poly_baseline: Option<usize>,
    /// Also fit ridge on monomials of the L2-normalised scaled descriptors (the amplitude-encoded
    /// vector): every circuit feature is a quadratic form in that vector.
    poly_normalized: Option<usize>,
    /// The head also reads the scaled descriptors next to the circuit features.
    append_raw: bool,
    /// Register size (at least 5, so the 19 amplitudes fit).
    qubits: usize,
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
        data: DataOptions::from_args(&args),
        poly_baseline: value("--poly-baseline").and_then(|v| v.parse().ok()),
        poly_normalized: value("--poly-normalized").and_then(|v| v.parse().ok()),
        append_raw: args.iter().any(|a| a == "--append-raw"),
        qubits: value("--qubits").and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_QUBITS).max(5),
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

/// Ridge regression on every monomial up to `degree` of the max-abs scaled descriptors.
fn polynomial_baseline(degree: usize, normalize: bool, train: &[Sample], test: &[Sample], names: &[&str]) -> BaselineResult {
    let rows: Vec<&[f64]> = train.iter().map(|m| m.features.as_slice()).collect();
    let scaler = Scaler::fit_rows(InputScaling::MaxAbs, &rows);
    let prepare = |m: &Sample| -> Vec<f64> {
        let x = scaler.apply(&m.features);
        if !normalize {
            return x;
        }
        let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
        x.iter().map(|v| v / norm).collect()
    };
    let expand = |set: &[Sample]| -> Vec<Vec<f64>> { set.iter().map(|m| polynomial_features(&prepare(m), degree)).collect() };
    let (x_train, x_test) = (expand(train), expand(test));
    let y_train: Vec<Vec<f64>> = train.iter().map(|m| m.targets.to_vec()).collect();
    let y_test: Vec<Vec<f64>> = test.iter().map(|m| m.targets.to_vec()).collect();
    let head = LinearHead::fit(&x_train, &y_train);
    BaselineResult {
        model: format!("Ridge (degree {degree}{})", if normalize { ", normalised input" } else { "" }),
        n_features: x_train[0].len(),
        lambda: Some(head.lambda()),
        metrics: regression_metrics(&head.predict(&x_test), &y_test, names),
    }
}

fn main() {
    let opts = options();
    let host = t1000::hostinfo::HostProfile::detect();
    println!("=========================================================================");
    println!("   QUANTUM FEATURE MAP + CLOSED-FORM LINEAR HEAD ({:?})", opts.data.dataset);
    println!("=========================================================================");
    println!(" Host            : {}", host.label());

    let Data { train, test, target_names, label } = opts.data.load();
    let report_path = if opts.data.dataset == Dataset::Qm9 { QM9_REPORT_PATH } else { ZINC_REPORT_PATH };
    let names: Vec<&str> = target_names.iter().map(String::as_str).collect();
    println!(" Data set        : {label}; targets {names:?}");
    let encoding = DataEncodingLayer::new(opts.qubits, EncodingKind::Amplitude);
    let ansatz = CircuitAnsatz::new(opts.qubits, opts.depth).with_pi_layer(true);
    let (weights, source) = circuit_weights(&opts, &ansatz);
    let map = CircuitFeatureMap { encoding: &encoding, ansatz: &ansatz, weights: &weights, observables: opts.observables };
    println!(" Circuit         : {} qubits, depth {}, {source}", opts.qubits, opts.depth);
    println!(" Readout         : {} observables ({:?}), ridge head fitted in closed form", map.n_features(), opts.observables);
    println!(" Input scaling   : {:?}", opts.scaling);
    println!(" Molecules       : {} training, {} held-out\n", train.len(), test.len());

    let t = Instant::now();
    let rows: Vec<&[f64]> = train.iter().map(|m| m.features.as_slice()).collect();
    let scaler = Scaler::fit_rows(opts.scaling, &rows);
    let scaled_train: Vec<Vec<f64>> = train.iter().map(|m| scaler.apply(&m.features)).collect();
    let scaled_test: Vec<Vec<f64>> = test.iter().map(|m| scaler.apply(&m.features)).collect();
    let with_raw = |mut rows: Vec<Vec<f64>>, raw: &[Vec<f64>]| -> Vec<Vec<f64>> {
        if opts.append_raw {
            rows.iter_mut().zip(raw).for_each(|(r, x)| r.extend_from_slice(x));
        }
        rows
    };
    let train_x = with_raw(map.feature_matrix(scaled_train.iter().map(Vec::as_slice)), &scaled_train);
    let train_ms = t.elapsed().as_secs_f64() * 1e3;
    let train_y: Vec<Vec<f64>> = train.iter().map(|m| m.targets.to_vec()).collect();

    let t = Instant::now();
    let head = LinearHead::fit(&train_x, &train_y);
    let fit_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let test_x = with_raw(map.feature_matrix(scaled_test.iter().map(Vec::as_slice)), &scaled_test);
    let preds = head.predict(&test_x);
    let eval_ms = t.elapsed().as_secs_f64() * 1e3;
    let test_y: Vec<Vec<f64>> = test.iter().map(|m| m.targets.to_vec()).collect();
    let metrics = regression_metrics(&preds, &test_y, &names);

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

    let pairs = |rows: &[Sample]| -> Vec<(Vec<f64>, Vec<f64>)> { rows.iter().map(|m| (m.features.clone(), m.targets.to_vec())).collect() };
    let mut baselines = evaluate_baselines(&pairs(&train), &pairs(&test), &names);
    if let Some(degree) = opts.poly_baseline {
        baselines.push(polynomial_baseline(degree, false, &train, &test, &names));
    }
    if let Some(degree) = opts.poly_normalized {
        baselines.push(polynomial_baseline(degree, true, &train, &test, &names));
    }
    let col = |f: fn(&TargetMetrics) -> f64| [f(&metrics[0]), f(&metrics[1]), f(&metrics[2])];
    let row = QnnScore::new("circuit + linear head")
        .mae(col(|m| m.mae))
        .mse(col(|m| m.mse))
        .r2(col(|m| m.r2))
        .pearson_r(col(|m| m.pearson_r));
    print_comparison_named(&[row], &baselines, &names);

    let report = LinearHeadReport {
        host_cpu: host.label(),
        host_ram_gb: host.ram_gb,
        n_qubits: opts.qubits,
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
    // ZINC keeps its original fingerprint so saved ZINC records stay comparable.
    let data_prefix = if opts.data.dataset == Dataset::Qm9 { format!("data={label}:{names:?};seed={};", opts.data.seed) } else { String::new() };
    let settings = format!(
        "{data_prefix}{}qubits={};depth={};weights={};obs={:?};scale={:?};train={};test={}",
        if opts.append_raw { "descriptors;" } else { "" },
        opts.qubits,
        opts.depth,
        weights_fingerprint(&weights),
        opts.observables,
        opts.scaling,
        train.len(),
        test.len()
    );
    let json = serde_json::to_string_pretty(&report).expect("serialize report");
    match t1000::results::write_scored(report_path, json, &holdout.score(&settings)) {
        Ok(t1000::results::Outcome::KeptBest) => println!("\n[=] Saved report kept: this run did not improve it"),
        Ok(t1000::results::Outcome::Scratch) => println!("\n[~] Report written to a scratch copy only (use --record to save)"),
        Ok(t1000::results::Outcome::WrittenVariant) => println!("\n[SAVED] Different settings from the saved report: recorded as a variant"),
        Ok(t1000::results::Outcome::Written) => println!("\n[SAVED] Report exported to: {report_path}"),
        Err(e) => eprintln!("\n[!] Could not write the report: {e}"),
    }
}
