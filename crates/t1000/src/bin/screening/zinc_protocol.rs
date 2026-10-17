//! Builds the shared ZINC evaluation protocol and scores the classical baselines under it.
//!
//! Pool: the 12,000 molecules in `zinc_train_10k.csv` + `zinc_test_2k.csv` (no overlap).
//! Splits (80/10/10, stored by SMILES in `tests/benchmarks/zinc_molecular/splits/`):
//! random and Bemis-Murcko scaffold, seeds 0, 1, 2. Existing split files are reused (and checked
//! against the pool); pass `--regenerate` to rebuild them.
//!
//! For every split the mean predictor, linear ridge and quadratic ridge are fit on train, the
//! ridge penalty is chosen on validation, and test metrics are computed once. Results are
//! summarised per split kind as mean +/- standard deviation over the three seeds and written to
//! `tests/benchmarks/zinc_molecular/reports/zinc_protocol_baselines.json`.

use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use t1000::protocol::{
    aggregate, baseline_on_split, compute_feature_set, dedupe_by_smiles, load_zinc_csv, make_split,
    murcko_scaffold_key, AggregateResult, BaselineModel, DataSplit, SplitFractions, SplitMethod,
    SplitResult,
};

const SEEDS: [u64; 3] = [0, 1, 2];

fn workspace(rel: &str) -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .ancestors()
        .nth(3)
        .unwrap_or(&manifest_dir)
        .join(rel)
}

#[derive(Serialize)]
struct SplitStats {
    name: String,
    train: usize,
    valid: usize,
    test: usize,
    test_scaffolds: usize,
    /// Fraction of test molecules whose (non-empty) scaffold also occurs in training.
    test_molecules_with_seen_scaffold: f64,
}

#[derive(Serialize)]
struct ProtocolReport {
    pool_sources: Vec<String>,
    pool_size: usize,
    pool_checksum: String,
    distinct_scaffolds: usize,
    acyclic_molecules: usize,
    unparsed_smiles: usize,
    fractions: SplitFractions,
    splits: Vec<SplitStats>,
    per_split: Vec<SplitResult>,
    summary: Vec<AggregateResult>,
}

fn main() {
    let regenerate = std::env::args().any(|a| a == "--regenerate");

    let sources = [
        "tests/benchmarks/zinc_molecular/data/zinc_train_10k.csv",
        "tests/benchmarks/zinc_molecular/data/zinc_test_2k.csv",
    ];
    let mut pool = Vec::new();
    for src in sources {
        match load_zinc_csv(&workspace(src)) {
            Ok(rows) => pool.extend(rows),
            Err(e) => {
                eprintln!("[-] {e}");
                std::process::exit(1);
            }
        }
    }
    let pool = dedupe_by_smiles(pool);
    let smiles: Vec<String> = pool.iter().map(|r| r.smiles.clone()).collect();
    let checksum = t1000::protocol::pool_checksum(smiles.iter().map(String::as_str));

    // Scaffold keys; SMILES that fail to parse each get their own group.
    let mut unparsed = 0;
    let keys: Vec<String> = smiles
        .iter()
        .map(|s| {
            murcko_scaffold_key(s).unwrap_or_else(|_| {
                unparsed += 1;
                format!("unparsed:{s}")
            })
        })
        .collect();
    let distinct: HashSet<&str> = keys
        .iter()
        .map(String::as_str)
        .filter(|k| !k.is_empty())
        .collect();
    let acyclic = keys.iter().filter(|k| k.is_empty()).count();
    println!("=========================================================================");
    println!(" ZINC EVALUATION PROTOCOL");
    println!("=========================================================================");
    println!("  pool: {} molecules (checksum {checksum})", pool.len());
    println!(
        "  scaffolds: {} distinct, {acyclic} acyclic molecules, {unparsed} unparsed SMILES",
        distinct.len()
    );

    let split_dir = workspace("tests/benchmarks/zinc_molecular/splits");
    fs::create_dir_all(&split_dir).expect("create splits directory");
    let fractions = SplitFractions::default();
    let mut splits: Vec<DataSplit> = Vec::new();
    for method in [SplitMethod::Random, SplitMethod::Scaffold] {
        for seed in SEEDS {
            let tag = match method {
                SplitMethod::Random => "random",
                SplitMethod::Scaffold => "scaffold",
            };
            let name = format!("zinc12k_{tag}_seed{seed}");
            let path = split_dir.join(format!("{name}.json"));
            let existing: Option<DataSplit> = if regenerate {
                None
            } else {
                fs::read_to_string(&path)
                    .ok()
                    .and_then(|t| serde_json::from_str(&t).ok())
            };
            let split = match existing {
                Some(s) => {
                    if let Err(e) = s.indices(&pool) {
                        eprintln!("[-] stored split does not match the pool: {e}\n    re-run with --regenerate");
                        std::process::exit(1);
                    }
                    s
                }
                None => {
                    let s = make_split(&name, &smiles, Some(&keys), method, seed, fractions);
                    fs::write(
                        &path,
                        serde_json::to_string_pretty(&s).expect("serialize split"),
                    )
                    .expect("write split");
                    println!("  [new] {}", path.display());
                    s
                }
            };
            splits.push(split);
        }
    }

    // Split statistics and baselines.
    let key_of: HashMap<&str, &str> = smiles
        .iter()
        .map(String::as_str)
        .zip(keys.iter().map(String::as_str))
        .collect();
    let mut stats = Vec::new();
    // Feature sets, one row per pool molecule.
    let t0 = Instant::now();
    let feature = |name: &str| compute_feature_set(name, &pool).unwrap_or_else(|e| panic!("{e}"));
    let csv19 = feature("csv19_legacy");
    let desc = feature("descriptors");
    let ecfp = feature("ecfp4_2048");
    let desc_ecfp = feature("descriptors+ecfp4");
    let quad_desc_ecfp = feature("quad_descriptors+ecfp4");
    println!(
        "\n  features computed in {:.2} s (descriptors: {}, ECFP4 bits: 2048)",
        t0.elapsed().as_secs_f64(),
        desc[0].len()
    );
    let runs: [(BaselineModel, &str, &Vec<Vec<f64>>); 9] = [
        (BaselineModel::Mean, "none", &csv19),
        (BaselineModel::RidgeLinear, "csv19_legacy", &csv19),
        (BaselineModel::RidgeQuadratic, "csv19_legacy", &csv19),
        (BaselineModel::RidgeLinear, "descriptors", &desc),
        (BaselineModel::RidgeQuadratic, "descriptors", &desc),
        (BaselineModel::RidgeLinear, "ecfp4_2048", &ecfp),
        (BaselineModel::KnnTanimoto, "ecfp4_2048", &ecfp),
        (BaselineModel::RidgeLinear, "descriptors+ecfp4", &desc_ecfp),
        (
            BaselineModel::RidgeLinear,
            "quad_descriptors+ecfp4",
            &quad_desc_ecfp,
        ),
    ];

    let mut per_split = Vec::new();
    for split in &splits {
        let idx = split.indices(&pool).expect("checked above");
        let train_keys: HashSet<&str> = idx
            .train
            .iter()
            .map(|&i| key_of[smiles[i].as_str()])
            .collect();
        let test_keys: Vec<&str> = idx
            .test
            .iter()
            .map(|&i| key_of[smiles[i].as_str()])
            .collect();
        let seen = test_keys
            .iter()
            .filter(|k| !k.is_empty() && train_keys.contains(*k))
            .count();
        stats.push(SplitStats {
            name: split.name.clone(),
            train: idx.train.len(),
            valid: idx.valid.len(),
            test: idx.test.len(),
            test_scaffolds: test_keys.iter().collect::<HashSet<_>>().len(),
            test_molecules_with_seen_scaffold: seen as f64 / test_keys.len() as f64,
        });
        for (model, name, feats) in &runs {
            let t = Instant::now();
            per_split.push(baseline_on_split(*model, name, feats, &pool, split).expect("baseline"));
            println!(
                "  {:<24} {:<18} {:<20} {:>6.2} s",
                split.name,
                model.name(),
                name,
                t.elapsed().as_secs_f64()
            );
        }
    }

    println!(
        "\n  {:<26} {:>6} {:>6} {:>6} {:>18}",
        "split", "train", "valid", "test", "test scaffold seen"
    );
    for s in &stats {
        println!(
            "  {:<26} {:>6} {:>6} {:>6} {:>17.1}%",
            s.name,
            s.train,
            s.valid,
            s.test,
            100.0 * s.test_molecules_with_seen_scaffold
        );
    }

    let mut summary = Vec::new();
    for method in [SplitMethod::Random, SplitMethod::Scaffold] {
        let names: HashSet<&str> = splits
            .iter()
            .filter(|s| s.method == method)
            .map(|s| s.name.as_str())
            .collect();
        let results: Vec<SplitResult> = per_split
            .iter()
            .filter(|r| names.contains(r.split.as_str()))
            .cloned()
            .collect();
        summary.extend(aggregate(method, &results));
    }

    for method in [SplitMethod::Random, SplitMethod::Scaffold] {
        println!(
            "\n  {method:?} split, test set, mean +/- std over {} seeds",
            SEEDS.len()
        );
        println!(
            "    {:<40} {:>17} {:>17} {:>17}",
            "model [features]", "logP R2", "QED R2", "SAS R2"
        );
        for a in summary.iter().filter(|a| a.method == method) {
            let cells: Vec<String> =
                a.r2.iter()
                    .map(|m| format!("{:.3} +/- {:.3}", m.mean, m.std))
                    .collect();
            let label = format!("{} [{}]", a.model, a.features);
            println!(
                "    {:<40} {:>17} {:>17} {:>17}",
                label, cells[0], cells[1], cells[2]
            );
        }
    }

    let report = ProtocolReport {
        pool_sources: sources.iter().map(|s| s.to_string()).collect(),
        pool_size: pool.len(),
        pool_checksum: checksum,
        distinct_scaffolds: distinct.len(),
        acyclic_molecules: acyclic,
        unparsed_smiles: unparsed,
        fractions,
        splits: stats,
        per_split,
        summary,
    };
    let out = workspace("tests/benchmarks/zinc_molecular/reports/zinc_protocol_baselines.json");
    fs::write(
        &out,
        serde_json::to_string_pretty(&report).expect("serialize report"),
    )
    .expect("write report");
    println!("\n[OK] report written to {}", out.display());
}
