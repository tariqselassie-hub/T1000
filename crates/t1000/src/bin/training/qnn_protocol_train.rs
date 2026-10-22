//! Configurable QNN training under the ZINC evaluation protocol.
//!
//! Uses the stored splits written by `zinc_protocol` (run it first) and one model definition
//! (`t1000::trainer`) whose circuit mechanisms are options.
//!
//! ```text
//! cargo run --release -p t1000 --bin qnn_protocol_train -- \
//!     [--config cfg.json] [--name NAME] [--features descriptors] [--qubits 12] [--depth 3]
//!     [--encoding amplitude|angle] [--readout z|z+zz] [--reupload] [--no-pi-layer]
//!     [--epochs 20] [--batch-size 64] [--lr 0.02] [--patience 5]
//!     [--splits random|scaffold|both] [--max-train N]
//! ```
//!
//! Flags override the config file. `--max-train` subsamples training data for quick checks; such
//! runs are labelled as not protocol-compliant in the report. Results (validation and test,
//! mean +/- std over seeds, plus the classical baseline bar) go to
//! `tests/benchmarks/zinc_molecular/reports/qnn_protocol/<name>.json`.

use serde::Serialize;
use std::fs;
use std::path::PathBuf;
use t1000::protocol::{
    aggregate, compute_feature_set, dedupe_by_smiles, load_zinc_csv, DataSplit, MeanStd,
    SplitMethod,
};
use t1000::trainer::{train_on_split, Encoding, Readout, TrainConfig, TrainRun};

fn workspace(rel: &str) -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .ancestors()
        .nth(3)
        .unwrap_or(&manifest_dir)
        .join(rel)
}

fn fail(msg: impl std::fmt::Display) -> ! {
    eprintln!("[-] {msg}");
    std::process::exit(1)
}

#[derive(Serialize)]
struct Report {
    config: TrainConfig,
    protocol_compliant: bool,
    max_train: Option<usize>,
    runs: Vec<TrainRun>,
    summary: Vec<SummaryRow>,
}

#[derive(Serialize)]
struct SummaryRow {
    method: SplitMethod,
    seeds: usize,
    valid_r2: Vec<MeanStd>,
    test_r2: Vec<MeanStd>,
    test_mae: Vec<MeanStd>,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let has = |flag: &str| args.iter().any(|a| a == flag);

    let mut cfg = match value("--config") {
        Some(path) => {
            let text =
                fs::read_to_string(&path).unwrap_or_else(|e| fail(format!("read {path}: {e}")));
            serde_json::from_str::<TrainConfig>(&text)
                .unwrap_or_else(|e| fail(format!("parse {path}: {e}")))
        }
        None => TrainConfig::default(),
    };
    let parse = |flag: &str| {
        value(flag).map(|v| {
            v.parse::<f64>()
                .unwrap_or_else(|_| fail(format!("bad {flag}: {v}")))
        })
    };
    if let Some(v) = value("--name") {
        cfg.name = v;
    }
    if let Some(v) = value("--features") {
        cfg.features = v;
    }
    if let Some(v) = parse("--qubits") {
        cfg.model.n_qubits = v as usize;
    }
    if let Some(v) = parse("--depth") {
        cfg.model.depth = v as usize;
    }
    if let Some(v) = value("--encoding") {
        cfg.model.encoding = match v.as_str() {
            "amplitude" => Encoding::Amplitude,
            "angle" => Encoding::Angle,
            other => fail(format!("unknown encoding {other}")),
        };
    }
    if let Some(v) = value("--readout") {
        cfg.model.readout = match v.as_str() {
            "z" => Readout::Z,
            "z+zz" => Readout::ZZ,
            other => fail(format!("unknown readout {other}")),
        };
    }
    if has("--reupload") {
        cfg.model.reupload = true;
    }
    if has("--no-pi-layer") {
        cfg.model.pi_layer = false;
    }
    if let Some(v) = parse("--epochs") {
        cfg.epochs = v as usize;
    }
    if let Some(v) = parse("--batch-size") {
        cfg.batch_size = v as usize;
    }
    if let Some(v) = parse("--lr") {
        cfg.lr = v;
    }
    if let Some(v) = parse("--patience") {
        cfg.patience = v as usize;
    }
    let max_train = parse("--max-train").map(|v| v as usize);
    let methods: Vec<SplitMethod> = match value("--splits").as_deref().unwrap_or("both") {
        "random" => vec![SplitMethod::Random],
        "scaffold" => vec![SplitMethod::Scaffold],
        "both" => vec![SplitMethod::Random, SplitMethod::Scaffold],
        other => fail(format!("unknown --splits {other}")),
    };

    // Pool, features and stored splits.
    let mut pool = Vec::new();
    for src in [
        "tests/benchmarks/zinc_molecular/data/zinc_train_10k.csv",
        "tests/benchmarks/zinc_molecular/data/zinc_test_2k.csv",
    ] {
        pool.extend(load_zinc_csv(&workspace(src)).unwrap_or_else(|e| fail(e)));
    }
    let pool = dedupe_by_smiles(pool);
    let features = compute_feature_set(&cfg.features, &pool).unwrap_or_else(|e| fail(e));
    let mut splits: Vec<DataSplit> = Vec::new();
    for method in &methods {
        let tag = match method {
            SplitMethod::Random => "random",
            SplitMethod::Scaffold => "scaffold",
        };
        for seed in 0..3 {
            let path = workspace(&format!(
                "tests/benchmarks/zinc_molecular/splits/zinc12k_{tag}_seed{seed}.json"
            ));
            let text = fs::read_to_string(&path).unwrap_or_else(|_| {
                fail(format!(
                    "{} missing; run zinc_protocol first",
                    path.display()
                ))
            });
            splits.push(
                serde_json::from_str(&text)
                    .unwrap_or_else(|e| fail(format!("parse {}: {e}", path.display()))),
            );
        }
    }

    println!("=========================================================================");
    println!(" QNN UNDER THE ZINC PROTOCOL: {}", cfg.name);
    println!("=========================================================================");
    println!(
        "  features {} ({} dims) | {:?}",
        cfg.features,
        features[0].len(),
        cfg.model
    );
    println!(
        "  epochs {} | batch {} | lr {} | clip {} | patience {} | {} threads{}",
        cfg.epochs,
        cfg.batch_size,
        cfg.lr,
        cfg.max_grad_norm,
        cfg.patience,
        t1000::parallel::num_threads(),
        max_train
            .map(|m| format!(" | max-train {m} (NOT protocol-compliant)"))
            .unwrap_or_default()
    );

    let mut runs = Vec::new();
    for split in &splits {
        println!("\n  [{}]", split.name);
        let run = train_on_split(&cfg, &features, &pool, split, max_train, |line| {
            println!("{line}")
        })
        .unwrap_or_else(|e| fail(e));
        let r2 = |m: &[t1000::baselines::TargetMetrics]| {
            m.iter()
                .map(|t| format!("{:.3}", t.r2))
                .collect::<Vec<_>>()
                .join(" / ")
        };
        println!(
            "    best epoch {} of {} | valid R2 {} | test R2 {} | {} params",
            run.best_epoch,
            run.epochs_run,
            r2(&run.valid),
            r2(&run.test.test),
            run.n_params
        );
        runs.push(run);
    }

    // Summary and comparison with the classical bar.
    let baseline_path =
        workspace("tests/benchmarks/zinc_molecular/reports/zinc_protocol_baselines.json");
    let baselines: Option<serde_json::Value> = fs::read_to_string(&baseline_path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok());
    let mut summary = Vec::new();
    for method in &methods {
        let mine: Vec<&TrainRun> = runs
            .iter()
            .filter(|r| {
                splits
                    .iter()
                    .any(|s| s.name == r.split && s.method == *method)
            })
            .collect();
        let tests: Vec<_> = mine.iter().map(|r| r.test.clone()).collect();
        let agg = aggregate(*method, &tests).remove(0);
        let valid_r2 = (0..3)
            .map(|t| MeanStd::of(&mine.iter().map(|r| r.valid[t].r2).collect::<Vec<_>>()))
            .collect();
        println!(
            "\n  {method:?} splits, mean +/- std over {} seeds (test R2: logP / QED / SAS)",
            mine.len()
        );
        let cells = |v: &[MeanStd]| {
            v.iter()
                .map(|m| format!("{:.3} +/- {:.3}", m.mean, m.std))
                .collect::<Vec<_>>()
                .join(" | ")
        };
        println!(
            "    {:<44} {}",
            format!("QNN: {}", cfg.name),
            cells(&agg.r2)
        );
        if let Some(b) = &baselines {
            let tag = match method {
                SplitMethod::Random => "random",
                SplitMethod::Scaffold => "scaffold",
            };
            for row in b["summary"].as_array().into_iter().flatten() {
                let wanted = row["method"] == tag
                    && ((row["model"] == "Ridge (linear)"
                        && row["features"] == cfg.features.as_str())
                        || (row["model"] == "Ridge (linear)"
                            && row["features"] == "quad_descriptors+ecfp4")
                        || row["model"] == "Mean predictor");
                if wanted {
                    let r2: Vec<MeanStd> =
                        serde_json::from_value(row["r2"].clone()).unwrap_or_default();
                    if r2.len() == 3 {
                        let label = format!(
                            "Classical: {} [{}]",
                            row["model"].as_str().unwrap_or(""),
                            row["features"].as_str().unwrap_or("")
                        );
                        println!("    {:<44} {}", label, cells(&r2));
                    }
                }
            }
        }
        summary.push(SummaryRow {
            method: *method,
            seeds: mine.len(),
            valid_r2,
            test_r2: agg.r2,
            test_mae: agg.mae,
        });
    }

    let report = Report {
        config: cfg.clone(),
        protocol_compliant: max_train.is_none(),
        max_train,
        runs,
        summary,
    };
    let out_dir = workspace("tests/benchmarks/zinc_molecular/reports/qnn_protocol");
    fs::create_dir_all(&out_dir).unwrap_or_else(|e| fail(e));
    let out = out_dir.join(format!("{}.json", cfg.name));
    t1000::results::write(
        &out,
        serde_json::to_string_pretty(&report).expect("serialize"),
    )
    .unwrap_or_else(|e| fail(e));
    println!("\n[OK] report written to {}", out.display());
}
