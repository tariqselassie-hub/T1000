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
//!         [--scale maxabs|minmax|raw] [--poly-baseline D] [--poly-normalized D] [--append-raw] [--qubits N] [--validation N] [--feature-map amplitude|reupload] [--uploads K] [--angle-scale S] [--samples N] [--eval-samples N] [--record]`

use serde::{Deserialize, Serialize};
use std::{env, fs, time::Instant};
use t1000::baselines::{
    evaluate_baselines, print_comparison_named, regression_metrics, BaselineResult, QnnScore,
    TargetMetrics,
};
use t1000::datasets::{Data, DataOptions, Dataset, Sample};
use t1000::feature_head::{fit_head, targets_of, validation_mse, FeatureExtractor, FittedHead};
use t1000::holdout_score::ZincHoldout;
use t1000::linear_head::{
    feature_spreads, shots_for_noise_equal_to, weights_fingerprint, CircuitFeatureMap, LinearHead,
    Observables,
};
use t1000::polynomial::polynomial_ridge_baseline;
use t1000::reuploading::ReuploadingMap;
use t1000::zinc_data::{InputScaling, Scaler};
use t1000::{CircuitAnsatz, DataEncodingLayer, EncodingKind};

const DEFAULT_QUBITS: usize = 10;
const ZINC_REPORT_PATH: &str =
    "tests/benchmarks/zinc_molecular/outputs/zinc_linear_head_results.json";
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
    #[serde(default)]
    validation_relative_mse: Option<f64>,
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
    /// Data re-uploading instead of amplitude encoding: `Some((uploads, angle_scale))`.
    reupload: Option<(usize, f64)>,
}

fn options() -> Options {
    let args: Vec<String> = env::args().collect();
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
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
        qubits: value("--qubits")
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_QUBITS)
            .max(5),
        reupload: (value("--feature-map").as_deref() == Some("reupload")).then(|| {
            (
                value("--uploads").and_then(|v| v.parse().ok()).unwrap_or(4),
                value("--angle-scale")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(std::f64::consts::PI),
            )
        }),
    }
}

/// Circuit weights from a `zinc_qnn_train` report, or the ansatz default.
fn circuit_weights(opts: &Options, ansatz: &CircuitAnsatz) -> (Vec<f64>, String) {
    let Some(path) = &opts.weights_path else {
        return (ansatz.weights.clone(), "default initialisation".to_string());
    };
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let report: serde_json::Value =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("cannot parse {path}: {e}"));
    let weights: Vec<f64> =
        serde_json::from_value(report["final_weights"].clone()).unwrap_or_else(|_| {
            panic!("{path} has no final_weights (re-run zinc_qnn_train to store them)")
        });
    assert_eq!(
        weights.len(),
        ansatz.param_count(),
        "{path} was trained with a different depth (--depth)"
    );
    (weights, format!("trained circuit from {path}"))
}

/// The circuit under test: register encoding, ansatz and its weights.
struct Circuit {
    encoding: DataEncodingLayer,
    ansatz: CircuitAnsatz,
    depth: usize,
    weights: Vec<f64>,
    source: String,
}

impl Circuit {
    fn new(opts: &Options) -> Self {
        let encoding = DataEncodingLayer::new(opts.qubits, EncodingKind::Amplitude);
        // Re-uploading needs one ansatz layer per upload.
        let depth = opts
            .reupload
            .map_or(opts.depth, |(uploads, _)| opts.depth.max(uploads));
        let ansatz = CircuitAnsatz::new(opts.qubits, depth).with_pi_layer(true);
        let (weights, source) = circuit_weights(opts, &ansatz);
        Self {
            encoding,
            ansatz,
            depth,
            weights,
            source,
        }
    }

    fn features(&self, opts: &Options) -> FeatureExtractor<'_> {
        let (uploads, angle_scale) = opts.reupload.unwrap_or((0, 0.0));
        FeatureExtractor {
            map: CircuitFeatureMap {
                encoding: &self.encoding,
                ansatz: &self.ansatz,
                weights: &self.weights,
                observables: opts.observables,
            },
            reupload: ReuploadingMap {
                ansatz: &self.ansatz,
                weights: &self.weights,
                uploads,
                angle_scale,
                observables: opts.observables,
            },
            use_reupload: opts.reupload.is_some(),
            append_raw: opts.append_raw,
        }
    }
}

/// What the run is about, shared by the printers and the report.
struct RunInfo<'a> {
    opts: &'a Options,
    label: &'a str,
    names: &'a [&'a str],
    depth: usize,
    source: &'a str,
}

/// Held-out results.
struct Evaluation {
    metrics: Vec<TargetMetrics>,
    holdout: ZincHoldout,
    validation_relative_mse: Option<f64>,
    eval_ms: f64,
}

fn print_title(opts: &Options, host: &t1000::hostinfo::HostProfile) {
    println!("=========================================================================");
    println!(
        "   QUANTUM FEATURE MAP + CLOSED-FORM LINEAR HEAD ({:?})",
        opts.data.dataset
    );
    println!("=========================================================================");
    println!(" Host            : {}", host.label());
}

fn print_setup(run: &RunInfo, features: &FeatureExtractor, n_train: usize, n_test: usize) {
    let opts = run.opts;
    match opts.reupload {
        Some((uploads, scale)) => println!(" Circuit         : {} qubits, depth {}, data re-uploading ({uploads} uploads, angle scale {scale:.3}), {}", opts.qubits, run.depth, run.source),
        None => println!(" Circuit         : {} qubits, depth {}, {}", opts.qubits, opts.depth, run.source),
    }
    println!(
        " Readout         : {} observables ({:?}), ridge head fitted in closed form",
        features.map.n_features(),
        opts.observables
    );
    println!(" Input scaling   : {:?}", opts.scaling);
    println!(" Molecules       : {n_train} training, {n_test} held-out\n");
}

fn evaluate(
    fit: &FittedHead,
    features: &FeatureExtractor,
    validation: &[Sample],
    test: &[Sample],
    names: &[&str],
) -> Evaluation {
    let t = Instant::now();
    let test_x = features.matrix(&fit.scaled_test);
    let preds = fit.head.predict(&test_x);
    let eval_ms = t.elapsed().as_secs_f64() * 1e3;
    let test_y = targets_of(test);
    let metrics = regression_metrics(&preds, &test_y, names);
    let validation_relative_mse = validation_mse(fit, features, validation, names);
    if let Some(v) = validation_relative_mse {
        println!("  Validation relative MSE {v:.4} on {} molecules (choose settings on this, not on the test set)", validation.len());
    }

    let target_cols: Vec<Vec<f64>> = (0..3)
        .map(|k| test_y.iter().map(|r| r[k]).collect())
        .collect();
    let holdout = ZincHoldout::new(
        [metrics[0].mse, metrics[1].mse, metrics[2].mse],
        [
            metrics[0].pearson_r,
            metrics[1].pearson_r,
            metrics[2].pearson_r,
        ],
        [&target_cols[0], &target_cols[1], &target_cols[2]],
    );
    Evaluation {
        metrics,
        holdout,
        validation_relative_mse,
        eval_ms,
    }
}

/// Prints timings, feature spread and the per-target table. Returns (median spread, shots).
fn print_results(fit: &FittedHead, eval: &Evaluation, n_train: usize, n_test: usize) -> (f64, f64) {
    println!(
        "  Feature extraction : {:>8.1} ms for {n_train} training molecules",
        fit.train_ms
    );
    println!(
        "  Head fit           : {:>8.1} ms (ridge lambda {:e})",
        fit.fit_ms,
        fit.head.lambda()
    );
    println!(
        "  Held-out inference : {:>8.1} ms ({:.1} us/molecule including the circuit)",
        eval.eval_ms,
        eval.eval_ms * 1e3 / n_test as f64
    );
    let mut spreads = feature_spreads(&fit.train_x);
    spreads.sort_by(f64::total_cmp);
    let median_spread = spreads[spreads.len() / 2];
    let shots = shots_for_noise_equal_to(median_spread);
    println!("  Feature spread     : median {median_spread:.2e} (min {:.1e}, max {:.1e}); sampling noise equals it at ~{shots:.1e} shots", spreads[0], spreads[spreads.len() - 1]);
    println!("\n  Target    MAE       MSE       R2      Pearson r");
    for m in &eval.metrics {
        println!(
            "  {:<6} {:>8.4}  {:>8.4}  {:>7.4}  {:>8.4}",
            m.target, m.mae, m.mse, m.r2, m.pearson_r
        );
    }
    println!(
        "\n  Held-out relative MSE {:.4} | mean Pearson r {:.4}",
        eval.holdout.relative_mse, eval.holdout.mean_pearson_r
    );
    (median_spread, shots)
}

/// Classical baselines (plus the optional polynomial ridges), printed next to the circuit row.
fn compare_with_baselines(
    opts: &Options,
    train: &[Sample],
    test: &[Sample],
    metrics: &[TargetMetrics],
    names: &[&str],
) -> Vec<BaselineResult> {
    let pairs = |rows: &[Sample]| -> Vec<(Vec<f64>, Vec<f64>)> {
        rows.iter()
            .map(|m| (m.features.clone(), m.targets.to_vec()))
            .collect()
    };
    let mut baselines = evaluate_baselines(&pairs(train), &pairs(test), names);
    if let Some(degree) = opts.poly_baseline {
        baselines.push(polynomial_ridge_baseline(degree, false, train, test, names));
    }
    if let Some(degree) = opts.poly_normalized {
        baselines.push(polynomial_ridge_baseline(degree, true, train, test, names));
    }
    let col = |f: fn(&TargetMetrics) -> f64| [f(&metrics[0]), f(&metrics[1]), f(&metrics[2])];
    let row = QnnScore::new("circuit + linear head")
        .mae(col(|m| m.mae))
        .mse(col(|m| m.mse))
        .r2(col(|m| m.r2))
        .pearson_r(col(|m| m.pearson_r));
    print_comparison_named(&[row], &baselines, names);
    baselines
}

/// Identifies a run for `write_scored`: only runs with identical settings compete.
fn settings_fingerprint(run: &RunInfo, weights: &[f64], n_train: usize, n_test: usize) -> String {
    let opts = run.opts;
    // ZINC keeps its original fingerprint so saved ZINC records stay comparable.
    let data_prefix = if opts.data.dataset == Dataset::Qm9 {
        format!(
            "data={};{:?};seed={};",
            run.label, run.names, opts.data.seed
        )
    } else {
        String::new()
    };
    format!(
        "{data_prefix}{}{}qubits={};depth={};weights={};obs={:?};scale={:?};train={n_train};test={n_test}",
        if opts.append_raw { "descriptors;" } else { "" },
        opts.reupload.map_or(String::new(), |(u, s)| format!("reupload={u}x{s:.3};")),
        opts.qubits,
        run.depth,
        weights_fingerprint(weights),
        opts.observables,
        opts.scaling,
    )
}

fn save_report(
    report_path: &str,
    report: &LinearHeadReport,
    holdout: &ZincHoldout,
    settings: &str,
) {
    let json = serde_json::to_string_pretty(report).expect("serialize report");
    match t1000::results::write_scored(report_path, json, &holdout.score(settings)) {
        Ok(t1000::results::Outcome::KeptBest) => {
            println!("\n[=] Saved report kept: this run did not improve it")
        }
        Ok(t1000::results::Outcome::Scratch) => {
            println!("\n[~] Report written to a scratch copy only (use --record to save)")
        }
        Ok(t1000::results::Outcome::WrittenVariant) => {
            println!("\n[SAVED] Different settings from the saved report: recorded as a variant")
        }
        Ok(t1000::results::Outcome::Written) => {
            println!("\n[SAVED] Report exported to: {report_path}")
        }
        Err(e) => eprintln!("\n[!] Could not write the report: {e}"),
    }
}

fn main() {
    let opts = options();
    let host = t1000::hostinfo::HostProfile::detect();
    print_title(&opts, &host);

    let Data {
        train,
        validation,
        test,
        target_names,
        label,
    } = opts.data.load();
    let report_path = if opts.data.dataset == Dataset::Qm9 {
        QM9_REPORT_PATH
    } else {
        ZINC_REPORT_PATH
    };
    let names: Vec<&str> = target_names.iter().map(String::as_str).collect();
    println!(" Data set        : {label}; targets {names:?}");
    let circuit = Circuit::new(&opts);
    let features = circuit.features(&opts);
    let run = RunInfo {
        opts: &opts,
        label: &label,
        names: &names,
        depth: circuit.depth,
        source: &circuit.source,
    };
    print_setup(&run, &features, train.len(), test.len());

    let fit = fit_head(opts.scaling, &features, &train, &test);
    let eval = evaluate(&fit, &features, &validation, &test, &names);
    let (median_spread, shots) = print_results(&fit, &eval, train.len(), test.len());
    let baselines = compare_with_baselines(&opts, &train, &test, &eval.metrics, &names);

    let settings = settings_fingerprint(&run, &circuit.weights, train.len(), test.len());
    let report = LinearHeadReport {
        host_cpu: host.label(),
        host_ram_gb: host.ram_gb,
        n_qubits: opts.qubits,
        circuit_depth: circuit.depth,
        circuit_weights_source: circuit.source.clone(),
        n_observables: features.map.n_features(),
        observables: format!("{:?}", opts.observables),
        train_sample_count: train.len(),
        test_sample_count: test.len(),
        ridge_lambda: fit.head.lambda(),
        feature_time_ms: fit.train_ms,
        fit_time_ms: fit.fit_ms,
        per_molecule_latency_us: eval.eval_ms * 1e3 / test.len() as f64,
        median_feature_spread: median_spread,
        shots_for_median_spread: shots,
        scaler: fit.scaler,
        metrics: eval.metrics,
        relative_mse: eval.holdout.relative_mse,
        mean_pearson_r: eval.holdout.mean_pearson_r,
        validation_relative_mse: eval.validation_relative_mse,
        classical_baselines: baselines,
        circuit_weights: circuit.weights.clone(),
        head: fit.head,
    };
    save_report(report_path, &report, &eval.holdout, &settings);
}
