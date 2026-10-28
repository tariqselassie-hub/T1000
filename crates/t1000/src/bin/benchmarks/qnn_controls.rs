// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Does the trained circuit add anything? Circuit controls and capacity-matched classical
//! models for the best QNN configuration, under the ZINC evaluation protocol.
//!
//! On each stored split (random and scaffold, seeds 0-2), with descriptors standardized on the
//! training rows:
//!
//! | Model | Trainable params | Question |
//! |---|---|---|
//! | QNN, trained (gradient-descent readout) | 321 | reference |
//! | QNN trained circuit + ridge readout | 321 | is the readout under-fit? |
//! | Random fixed circuit + ridge readout | 237 | does training the circuit matter? |
//! | Identity circuit (encoding only) + ridge readout | 237 | does the circuit add anything beyond the encoding? |
//! | Random Fourier features (78) + ridge readout | 237 | classical random feature map of the same size |
//! | MLP 26-10-3 (tanh) | 303 | capacity-matched classical nonlinear model |
//! | Ridge (linear / quadratic) on descriptors | 81 / 1,134 | protocol baselines |
//!
//! Ridge readouts and the Fourier bandwidth are chosen on validation; every model's test set is
//! scored once. Output: `tests/benchmarks/zinc_molecular/reports/qnn_controls.json`.

use serde::Serialize;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use t1000::baselines::{
    regression_metrics, train_mlp, MlpTraining, RandomFourierMap, RIDGE_LAMBDAS, ZINC_TARGETS,
};
use t1000::protocol::{
    aggregate, baseline_on_split, compute_feature_set, dedupe_by_smiles, fit_ridge_on_validation,
    load_zinc_csv, AggregateResult, BaselineModel, DataSplit, SplitMethod, SplitResult,
};
use t1000::trainer::{
    observable_feature_rows, prepare_split, train_on_split, Encoding, HybridModel, ModelConfig,
    PreparedSplit, Readout, TrainConfig,
};

fn workspace(rel: &str) -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .ancestors()
        .nth(3)
        .unwrap_or(&manifest_dir)
        .join(rel)
}

fn result(
    split: &DataSplit,
    model: &str,
    preds: Vec<Vec<f64>>,
    prep: &PreparedSplit,
) -> SplitResult {
    SplitResult {
        split: split.name.clone(),
        features: "descriptors".to_string(),
        model: model.to_string(),
        lambda: None,
        k: None,
        test: regression_metrics(&preds, &prep.y_test, &ZINC_TARGETS),
    }
}

/// Ridge readout on fixed feature rows, penalty chosen on validation; returns (test predictions,
/// validation score = mean over targets of 1 - R^2).
fn ridge_readout(
    f_train: &[Vec<f64>],
    f_valid: &[Vec<f64>],
    f_test: &[Vec<f64>],
    prep: &PreparedSplit,
) -> (Vec<Vec<f64>>, f64) {
    let ridge = fit_ridge_on_validation(
        f_train,
        &prep.y_train,
        f_valid,
        &prep.y_valid,
        &RIDGE_LAMBDAS,
    );
    let valid = regression_metrics(&ridge.predict(f_valid), &prep.y_valid, &ZINC_TARGETS);
    let score = valid.iter().map(|m| 1.0 - m.r2).sum::<f64>() / 3.0;
    (ridge.predict(f_test), score)
}

fn circuit_readout(model: &HybridModel, prep: &PreparedSplit) -> Vec<Vec<f64>> {
    let f = |xs: &[Vec<f64>]| observable_feature_rows(model, xs);
    ridge_readout(&f(&prep.x_train), &f(&prep.x_valid), &f(&prep.x_test), prep).0
}

#[derive(Serialize)]
struct Report {
    qnn_config: TrainConfig,
    trainable_params: Vec<(String, usize)>,
    per_split: Vec<SplitResult>,
    summary: Vec<AggregateResult>,
}

fn main() {
    let mut pool = Vec::new();
    for src in [
        "tests/benchmarks/zinc_molecular/data/zinc_train_10k.csv",
        "tests/benchmarks/zinc_molecular/data/zinc_test_2k.csv",
    ] {
        pool.extend(load_zinc_csv(&workspace(src)).expect("load pool"));
    }
    let pool = dedupe_by_smiles(pool);
    let desc = compute_feature_set("descriptors", &pool).expect("descriptors");
    let n_in = desc[0].len();

    let best = ModelConfig {
        n_qubits: 12,
        depth: 3,
        pi_layer: true,
        encoding: Encoding::Amplitude,
        readout: Readout::ZZ,
        reupload: true,
    };
    let identity = ModelConfig {
        depth: 0,
        pi_layer: false,
        reupload: false,
        ..best.clone()
    };
    let cfg = TrainConfig {
        name: "qnn_best".to_string(),
        features: "descriptors".to_string(),
        model: best.clone(),
        epochs: 60,
        patience: 8,
        ..TrainConfig::default()
    };
    let n_obs = HybridModel::new(best.clone(), n_in, 0).n_observables();
    let readout_params = 3 * n_obs + 3;
    let mlp_hidden = 10;
    let names = [
        (
            "QNN trained (gradient-descent readout)",
            HybridModel::new(best.clone(), n_in, 0).n_params(),
        ),
        (
            "QNN trained circuit + ridge readout",
            HybridModel::new(best.clone(), n_in, 0).n_params(),
        ),
        ("Random fixed circuit + ridge readout", readout_params),
        ("Identity circuit + ridge readout", readout_params),
        ("Random Fourier features + ridge readout", readout_params),
        (
            "MLP 26-10-3",
            mlp_hidden * n_in + mlp_hidden + 3 * mlp_hidden + 3,
        ),
        ("Ridge (linear)", 3 * (n_in + 1)),
        ("Ridge (quadratic)", 3 * (n_in + n_in * (n_in + 1) / 2 + 1)),
    ];

    println!("=========================================================================");
    println!(" QNN CIRCUIT CONTROLS + CAPACITY-MATCHED CLASSICAL MODELS");
    println!("=========================================================================");
    for (name, p) in &names {
        println!("  {name:<44} {p:>5} trainable params");
    }

    let mut per_split = Vec::new();
    for tag in ["random", "scaffold"] {
        for seed in 0..3 {
            let path = workspace(&format!(
                "tests/benchmarks/zinc_molecular/splits/zinc12k_{tag}_seed{seed}.json"
            ));
            let split: DataSplit =
                serde_json::from_str(&fs::read_to_string(&path).expect("run zinc_protocol first"))
                    .expect("parse split");
            let prep = prepare_split(&desc, &pool, &split).expect("prepare split");
            let t0 = Instant::now();
            println!("\n  [{}]", split.name);

            let run = train_on_split(&cfg, &desc, &pool, &split, None, |_| {}).expect("train QNN");
            println!(
                "    QNN trained: best epoch {} of {} ({:.0} s)",
                run.best_epoch,
                run.epochs_run,
                t0.elapsed().as_secs_f64()
            );
            per_split.push(SplitResult {
                model: names[0].0.to_string(),
                ..run.test.clone()
            });

            let mut trained = HybridModel::new(best.clone(), n_in, split.seed);
            trained.params = run.params.clone();
            per_split.push(result(
                &split,
                names[1].0,
                circuit_readout(&trained, &prep),
                &prep,
            ));

            let random = HybridModel::new(best.clone(), n_in, split.seed);
            per_split.push(result(
                &split,
                names[2].0,
                circuit_readout(&random, &prep),
                &prep,
            ));

            let ident = HybridModel::new(identity.clone(), n_in, split.seed);
            per_split.push(result(
                &split,
                names[3].0,
                circuit_readout(&ident, &prep),
                &prep,
            ));

            let best_rff = [0.5, 1.0, 2.0]
                .iter()
                .map(|&f| {
                    let map =
                        RandomFourierMap::new(n_in, n_obs, f * (n_in as f64).sqrt(), split.seed);
                    let t =
                        |xs: &[Vec<f64>]| xs.iter().map(|x| map.transform(x)).collect::<Vec<_>>();
                    ridge_readout(
                        &t(&prep.x_train),
                        &t(&prep.x_valid),
                        &t(&prep.x_test),
                        &prep,
                    )
                })
                .reduce(|a, b| if b.1 < a.1 { b } else { a })
                .expect("bandwidth grid");
            per_split.push(result(&split, names[4].0, best_rff.0, &prep));

            let mlp = train_mlp(
                &prep.x_train,
                &prep.y_train,
                &prep.x_valid,
                &prep.y_valid,
                mlp_hidden,
                split.seed,
                MlpTraining::default(),
            );
            per_split.push(result(
                &split,
                names[5].0,
                prep.x_test.iter().map(|x| mlp.predict(x)).collect(),
                &prep,
            ));

            for model in [BaselineModel::RidgeLinear, BaselineModel::RidgeQuadratic] {
                per_split.push(
                    baseline_on_split(model, "descriptors", &desc, &pool, &split)
                        .expect("baseline"),
                );
            }
            println!("    split done in {:.0} s", t0.elapsed().as_secs_f64());
        }
    }

    let mut summary = Vec::new();
    for (method, tag) in [
        (SplitMethod::Random, "random"),
        (SplitMethod::Scaffold, "scaffold"),
    ] {
        let rows: Vec<SplitResult> = per_split
            .iter()
            .filter(|r| r.split.contains(tag))
            .cloned()
            .collect();
        summary.extend(aggregate(method, &rows));
    }
    for method in [SplitMethod::Random, SplitMethod::Scaffold] {
        println!("\n  {method:?} splits, test R2 mean +/- std over 3 seeds");
        println!(
            "    {:<44} {:>6} {:>17} {:>17} {:>17}",
            "model", "params", "logP", "QED", "SAS"
        );
        for a in summary.iter().filter(|a| a.method == method) {
            let p = names
                .iter()
                .find(|(n, _)| *n == a.model)
                .map_or(0, |(_, p)| *p);
            let c: Vec<String> =
                a.r2.iter()
                    .map(|m| format!("{:.3} +/- {:.3}", m.mean, m.std))
                    .collect();
            println!(
                "    {:<44} {:>6} {:>17} {:>17} {:>17}",
                a.model, p, c[0], c[1], c[2]
            );
        }
    }

    let report = Report {
        qnn_config: cfg,
        trainable_params: names.iter().map(|(n, p)| (n.to_string(), *p)).collect(),
        per_split,
        summary,
    };
    let out = workspace("tests/benchmarks/zinc_molecular/reports/qnn_controls.json");
    t1000::results::write(
        &out,
        serde_json::to_string_pretty(&report).expect("serialize"),
    )
    .expect("write report");
    println!("\n[OK] report written to {}", out.display());
}
