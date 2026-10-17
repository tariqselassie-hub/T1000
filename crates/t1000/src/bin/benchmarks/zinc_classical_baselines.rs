//! Classical baselines for the ZINC (logP, QED, SAS) benchmark.
//!
//! Fits a mean predictor, linear ridge and quadratic ridge on the same 19 raw features and
//! train/test splits the QNN experiments use, then prints them next to the QNN numbers stored
//! in the saved experiment reports (when present). Writes
//! `tests/benchmarks/zinc_molecular/reports/zinc_classical_baselines.json`.
//!
//! Splits:
//! - main: first 10,000 rows of `zinc_train_10k.csv` / 2,000 of `zinc_test_2k.csv`
//!   (12-qubit hybrid, mid-circuit, re-uploading, QNG, ZZ, 10k-train and 20-epoch runs)
//! - bench: first 2,000 rows of `zinc_train_5k.csv` / 1,000 of `zinc_test_1k.csv` (`zinc_qnn_bench`)

use serde::Serialize;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use t1000::baselines::{print_comparison, zinc_baselines, BaselineResult, QnnScore};

struct Sample {
    features: Vec<f64>,
    targets: [f64; 3],
}

fn resolve_workspace_path(rel: &str) -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let ws_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .unwrap_or(&manifest_dir);
    ws_root.join(rel)
}

/// Same column layout the QNN programs read: smiles, 19 features, logP, QED, SAS.
fn parse_zinc_csv(path: &Path, max_rows: usize) -> Result<Vec<Sample>, String> {
    let content = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut out = Vec::new();
    for line in content.lines().skip(1) {
        let parts: Vec<&str> = line.split(',').map(str::trim).collect();
        if parts.len() < 23 {
            continue;
        }
        let num = |s: &str| s.parse::<f64>().unwrap_or(0.0);
        out.push(Sample {
            features: parts[1..20].iter().map(|s| num(s)).collect(),
            targets: [num(parts[20]), num(parts[21]), num(parts[22])],
        });
        if out.len() >= max_rows {
            break;
        }
    }
    if out.is_empty() {
        return Err(format!("no rows parsed from {}", path.display()));
    }
    Ok(out)
}

fn load(rel: &str, max_rows: usize) -> Result<Vec<Sample>, String> {
    let primary = resolve_workspace_path(&format!("tests/benchmarks/zinc_molecular/data/{rel}"));
    parse_zinc_csv(&primary, max_rows).or_else(|_| {
        parse_zinc_csv(
            &resolve_workspace_path(&format!("tests/benchmarks/zinc_molecular/outputs/{rel}")),
            max_rows,
        )
    })
}

fn baselines_for(train: &[Sample], test: &[Sample]) -> Vec<BaselineResult> {
    zinc_baselines(
        train.iter().map(|s| (s.features.as_slice(), s.targets)),
        test.iter().map(|s| (s.features.as_slice(), s.targets)),
    )
}

fn read_report(rel: &str) -> Option<Value> {
    let path = resolve_workspace_path(&format!("tests/benchmarks/zinc_molecular/{rel}"));
    serde_json::from_str(&fs::read_to_string(path).ok()?).ok()
}

fn triple(v: &Value, f: impl Fn(&str) -> String) -> Option<[f64; 3]> {
    let get = |t: &str| v.get(f(t)).and_then(Value::as_f64);
    Some([get("logp")?, get("qed")?, get("sas")?])
}

/// Rows from reports that store `<variant>_mae_<target>` / `<variant>_r2_<target>`.
fn mae_r2_rows(rel: &str, experiment: &str, variants: &[(&str, &str)]) -> Vec<QnnScore> {
    let Some(v) = read_report(rel) else {
        return Vec::new();
    };
    variants
        .iter()
        .filter_map(|(key, label)| {
            let mae = triple(&v, |t| format!("{key}_mae_{t}"))?;
            let r2 = triple(&v, |t| format!("{key}_r2_{t}"))?;
            Some(
                QnnScore::new(format!("{experiment}: {label}"))
                    .mae(mae)
                    .r2(r2),
            )
        })
        .collect()
}

/// Rows from reports with `<target>_metrics { tuned_mae, tuned_mse, pearson_r }`.
fn property_metric_row(rel: &str, label: &str) -> Option<QnnScore> {
    let v = read_report(rel)?;
    let field = |name: &str| -> Option<[f64; 3]> {
        let get = |t: &str| v.get(format!("{t}_metrics"))?.get(name)?.as_f64();
        Some([get("logp")?, get("qed")?, get("sas")?])
    };
    Some(
        QnnScore::new(label)
            .mae(field("tuned_mae")?)
            .mse(field("tuned_mse")?)
            .pearson_r(field("pearson_r")?),
    )
}

#[derive(Serialize)]
struct SplitReport {
    split: String,
    train_rows: usize,
    test_rows: usize,
    baselines: Vec<BaselineResult>,
}

fn main() {
    println!("=========================================================================");
    println!(" ZINC CLASSICAL BASELINES (mean, linear ridge, quadratic ridge)");
    println!("=========================================================================");

    let mut reports = Vec::new();

    // ---- main split ----
    let (train, test) = match (
        load("zinc_train_10k.csv", 10_000),
        load("zinc_test_2k.csv", 2_000),
    ) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("[-] {e}");
            std::process::exit(1);
        }
    };
    println!(
        "\n[main split] train = {}, test = {}",
        train.len(),
        test.len()
    );
    let main_baselines = baselines_for(&train, &test);

    let mut rows = Vec::new();
    rows.extend(mae_r2_rows(
        "reports/zinc_12qubit_midcircuit_report.json",
        "midcircuit",
        &[
            ("baseline", "no feed-forward"),
            ("feedforward", "feed-forward"),
            ("combined", "feed-forward + ZZ"),
        ],
    ));
    rows.extend(mae_r2_rows(
        "reports/zinc_12qubit_reuploading_report.json",
        "reuploading",
        &[
            ("baseline", "no mechanisms"),
            ("reuploading", "re-uploading"),
            ("unified", "re-uploading + ZZ"),
        ],
    ));
    rows.extend(mae_r2_rows(
        "reports/zinc_12qubit_qng_report.json",
        "qng",
        &[("euclidean", "Euclidean Adam"), ("qng", "natural gradient")],
    ));
    rows.extend(mae_r2_rows(
        "reports/zinc_12qubit_zz_correlator_report.json",
        "zz",
        &[
            ("baseline", "12 Z readout"),
            ("enhanced", "Z + ZZ (78 obs)"),
        ],
    ));
    rows.extend(property_metric_row(
        "outputs/zinc_12qubit_30ep_results.json",
        "hybrid 12q (30 epochs)",
    ));
    rows.extend(property_metric_row(
        "outputs/zinc_qnn_10k_results.json",
        "10q train (10k)",
    ));
    rows.extend(property_metric_row(
        "outputs/zinc_qnn_20ep_eval_results.json",
        "10q train (20 epochs)",
    ));
    if rows.is_empty() {
        println!("  (no saved QNN reports found; showing baselines only)");
    }
    print_comparison(&rows, &main_baselines);
    reports.push(SplitReport {
        split: "main: zinc_train_10k / zinc_test_2k".to_string(),
        train_rows: train.len(),
        test_rows: test.len(),
        baselines: main_baselines,
    });

    // ---- zinc_qnn_bench split ----
    if let (Ok(train), Ok(test)) = (
        load("zinc_train_5k.csv", 2_000),
        load("zinc_test_1k.csv", 1_000),
    ) {
        println!(
            "\n[bench split] train = {}, test = {}",
            train.len(),
            test.len()
        );
        let bench_baselines = baselines_for(&train, &test);
        let bench_rows: Vec<QnnScore> = read_report("outputs/zinc_qnn_results.json")
            .and_then(|v| triple(&v, |t| format!("tuned_{t}_mse")))
            .map(|mse| vec![QnnScore::new("zinc_qnn_bench (tuned)").mse(mse)])
            .unwrap_or_default();
        print_comparison(&bench_rows, &bench_baselines);
        reports.push(SplitReport {
            split: "bench: zinc_train_5k[..2000] / zinc_test_1k".to_string(),
            train_rows: train.len(),
            test_rows: test.len(),
            baselines: bench_baselines,
        });
    }

    let out = resolve_workspace_path(
        "tests/benchmarks/zinc_molecular/reports/zinc_classical_baselines.json",
    );
    match serde_json::to_string_pretty(&reports) {
        Ok(json) => match fs::write(&out, json) {
            Ok(()) => println!("\n[OK] Baselines written to {}", out.display()),
            Err(e) => eprintln!("[-] could not write {}: {e}", out.display()),
        },
        Err(e) => eprintln!("[-] could not serialize baselines: {e}"),
    }
}
