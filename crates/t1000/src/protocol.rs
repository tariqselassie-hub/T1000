//! Shared evaluation protocol for molecular property experiments.
//!
//! Every experiment that predicts molecular properties should be evaluated the same way, so that
//! results can be compared and a change can be judged by evidence:
//!
//! 1. **Fixed splits.** A pool of molecules is split into train / validation / test (80/10/10 by
//!    default). Splits are stored by SMILES (not row index), so they stay valid if the feature
//!    columns change. Two kinds are provided:
//!    - *random*: a seeded shuffle;
//!    - *scaffold*: molecules are grouped by Bemis-Murcko scaffold (ring systems plus the linkers
//!      between them) and whole groups are assigned to one side, so test molecules have ring
//!      cores never seen in training. This measures generalisation to new chemistry, which a
//!      random split overstates.
//! 2. **Validation for selection.** Hyperparameters (e.g. the ridge penalty, epochs, learning
//!    rate) are chosen on the validation set only; the test set is evaluated once per model.
//! 3. **Several seeds.** Each split kind is repeated over several seeds and metrics are reported
//!    as mean +/- sample standard deviation.
//! 4. **Same baselines.** Every report includes the classical baselines on the same split.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::baselines::{
    quadratic_features, regression_metrics, RidgeProblem, RidgeRegression, TargetMetrics,
    RIDGE_LAMBDAS, ZINC_TARGETS,
};
use crate::smiles::{parse, BondKind, SmilesError};
use rayon::prelude::*;

// =========================================================================
// Data
// =========================================================================

/// One molecule: SMILES, input features and the (logP, QED, SAS) targets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoleculeRecord {
    pub smiles: String,
    pub features: Vec<f64>,
    pub targets: [f64; 3],
}

/// Reads a ZINC benchmark CSV: `smiles, 19 feature columns, logP, QED, SAS` with a header row.
pub fn load_zinc_csv(path: &Path) -> Result<Vec<MoleculeRecord>, String> {
    let content = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut out = Vec::new();
    for (line_no, line) in content.lines().enumerate().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').map(str::trim).collect();
        if parts.len() != 23 {
            return Err(format!(
                "{}:{}: expected 23 columns, got {}",
                path.display(),
                line_no + 1,
                parts.len()
            ));
        }
        let num = |s: &str| {
            s.parse::<f64>()
                .map_err(|_| format!("{}:{}: not a number: {s:?}", path.display(), line_no + 1))
        };
        let features = parts[1..20]
            .iter()
            .map(|s| num(s))
            .collect::<Result<Vec<_>, _>>()?;
        out.push(MoleculeRecord {
            smiles: parts[0].to_string(),
            features,
            targets: [num(parts[20])?, num(parts[21])?, num(parts[22])?],
        });
    }
    Ok(out)
}

/// Concatenates pools, dropping repeated SMILES (first occurrence wins).
pub fn dedupe_by_smiles(records: Vec<MoleculeRecord>) -> Vec<MoleculeRecord> {
    let mut seen = HashSet::new();
    records
        .into_iter()
        .filter(|r| seen.insert(r.smiles.clone()))
        .collect()
}

/// Order-independent fingerprint of a pool's SMILES set (FNV-1a over the sorted SMILES).
pub fn pool_checksum<'a>(smiles: impl IntoIterator<Item = &'a str>) -> String {
    let mut all: Vec<&str> = smiles.into_iter().collect();
    all.sort_unstable();
    let mut h = Fnv::new();
    for s in all {
        h.write(s.as_bytes());
        h.write(&[0]);
    }
    format!("{:016x}", h.finish())
}

// =========================================================================
// Feature sets
// =========================================================================

/// Feature sets understood by [`compute_feature_set`].
pub const FEATURE_SETS: [&str; 5] = [
    "csv19_legacy",
    "descriptors",
    "ecfp4_2048",
    "descriptors+ecfp4",
    "quad_descriptors+ecfp4",
];

/// Computes a named feature set for every pool molecule (rows in pool order).
///
/// - `csv19_legacy`: the CSV feature columns (character-counted by `scripts/preprocess_zinc.py`).
/// - `descriptors`: [`crate::molfeatures::descriptors`] (26 values).
/// - `ecfp4_2048`: ECFP count fingerprint, radius 2, folded to 2048.
/// - `descriptors+ecfp4`: concatenation of the two.
/// - `quad_descriptors+ecfp4`: descriptors with all pairwise products, then ECFP4.
pub fn compute_feature_set(name: &str, pool: &[MoleculeRecord]) -> Result<Vec<Vec<f64>>, String> {
    use crate::molfeatures::{descriptors, ecfp_counts};
    let desc = || -> Result<Vec<Vec<f64>>, String> {
        pool.iter()
            .map(|r| {
                descriptors(&r.smiles).map_err(|e| format!("descriptors for {}: {e}", r.smiles))
            })
            .collect()
    };
    let ecfp = || -> Result<Vec<Vec<f64>>, String> {
        pool.iter()
            .map(|r| {
                ecfp_counts(&r.smiles, 2, 2048).map_err(|e| format!("ECFP for {}: {e}", r.smiles))
            })
            .collect()
    };
    let concat = |a: Vec<Vec<f64>>, b: Vec<Vec<f64>>| -> Vec<Vec<f64>> {
        a.into_iter()
            .zip(b)
            .map(|(mut x, y)| {
                x.extend(y);
                x
            })
            .collect()
    };
    match name {
        "csv19_legacy" => Ok(pool.iter().map(|r| r.features.clone()).collect()),
        "descriptors" => desc(),
        "ecfp4_2048" => ecfp(),
        "descriptors+ecfp4" => Ok(concat(desc()?, ecfp()?)),
        "quad_descriptors+ecfp4" => Ok(concat(
            desc()?.iter().map(|d| quadratic_features(d)).collect(),
            ecfp()?,
        )),
        other => Err(format!(
            "unknown feature set {other:?}; expected one of {FEATURE_SETS:?}"
        )),
    }
}

/// Per-column z-scoring fitted on training rows (constant columns are centred only).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Standardizer {
    pub mean: Vec<f64>,
    pub scale: Vec<f64>,
}

impl Standardizer {
    pub fn fit<'a>(rows: impl IntoIterator<Item = &'a [f64]>) -> Self {
        let rows: Vec<&[f64]> = rows.into_iter().collect();
        assert!(!rows.is_empty(), "cannot fit a standardizer on no rows");
        let d = rows[0].len();
        let n = rows.len() as f64;
        let mut mean = vec![0.0; d];
        for r in &rows {
            for (m, v) in mean.iter_mut().zip(*r) {
                *m += v / n;
            }
        }
        let mut var = vec![0.0; d];
        for r in &rows {
            for j in 0..d {
                var[j] += (r[j] - mean[j]).powi(2) / n;
            }
        }
        let scale = var
            .into_iter()
            .map(|v| if v > 1e-24 { v.sqrt() } else { 1.0 })
            .collect();
        Self { mean, scale }
    }

    pub fn transform(&self, row: &[f64]) -> Vec<f64> {
        row.iter()
            .zip(&self.mean)
            .zip(&self.scale)
            .map(|((x, m), s)| (x - m) / s)
            .collect()
    }

    pub fn inverse(&self, row: &[f64]) -> Vec<f64> {
        row.iter()
            .zip(&self.mean)
            .zip(&self.scale)
            .map(|((z, m), s)| z * s + m)
            .collect()
    }
}

// =========================================================================
// Bemis-Murcko scaffolds
// =========================================================================

/// Stable 64-bit FNV-1a hash (independent of Rust version, unlike `DefaultHasher`).
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
        }
    }
    fn write_u64(&mut self, v: u64) {
        self.write(&v.to_le_bytes());
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

/// Key identifying a molecule's Bemis-Murcko scaffold, or `""` for acyclic molecules.
///
/// The scaffold keeps ring atoms and the atoms on paths between rings: all other atoms are
/// stripped by repeatedly removing atoms with at most one remaining neighbour (side chains,
/// exocyclic atoms such as `=O`, and fragments without rings such as counter-ions). Atom labels
/// (element, aromatic flag) and bond kinds are kept.
///
/// The key is a Weisfeiler-Lehman hash of the scaffold graph rather than a canonical SMILES:
/// isomorphic scaffolds always get the same key and different scaffolds get different keys
/// except in rare hash or WL-indistinguishable cases. Aromaticity is not perceived, so a Kekulé
/// and an aromatic SMILES of the same ring give different keys; ZINC SMILES are aromatic-form.
pub fn murcko_scaffold_key(smiles: &str) -> Result<String, SmilesError> {
    let mol = parse(smiles)?;
    let n = mol.atoms.len();
    let mut adj: Vec<Vec<(usize, u8)>> = vec![Vec::new(); n];
    for &(a, b, kind) in &mol.bonds {
        let code = match kind {
            BondKind::Single | BondKind::Up | BondKind::Down => 1,
            BondKind::Double => 2,
            BondKind::Triple => 3,
            BondKind::Quadruple => 4,
            BondKind::Aromatic => 5,
        };
        adj[a].push((b, code));
        adj[b].push((a, code));
    }

    // Prune atoms with <= 1 live neighbour until only ring systems and linkers remain.
    let mut alive = vec![true; n];
    let mut degree: Vec<usize> = adj.iter().map(Vec::len).collect();
    let mut stack: Vec<usize> = (0..n).filter(|&i| degree[i] <= 1).collect();
    while let Some(i) = stack.pop() {
        if !alive[i] {
            continue;
        }
        alive[i] = false;
        for &(j, _) in &adj[i] {
            if alive[j] {
                degree[j] -= 1;
                if degree[j] == 1 {
                    stack.push(j);
                }
            }
        }
    }
    let kept: Vec<usize> = (0..n).filter(|&i| alive[i]).collect();
    if kept.is_empty() {
        return Ok(String::new());
    }

    // Weisfeiler-Lehman refinement on the scaffold graph.
    let mut label: Vec<u64> = (0..n)
        .map(|i| {
            let mut h = Fnv::new();
            h.write(mol.atoms[i].element.as_bytes());
            h.write(&[u8::from(mol.atoms[i].aromatic)]);
            h.finish()
        })
        .collect();
    let rounds = kept.len().min(16);
    for _ in 0..rounds {
        let next: Vec<u64> = (0..n)
            .map(|i| {
                if !alive[i] {
                    return 0;
                }
                let mut neigh: Vec<(u8, u64)> = adj[i]
                    .iter()
                    .filter(|&&(j, _)| alive[j])
                    .map(|&(j, code)| (code, label[j]))
                    .collect();
                neigh.sort_unstable();
                let mut h = Fnv::new();
                h.write_u64(label[i]);
                for (code, l) in neigh {
                    h.write(&[code]);
                    h.write_u64(l);
                }
                h.finish()
            })
            .collect();
        label = next;
    }

    let mut final_labels: Vec<u64> = kept.iter().map(|&i| label[i]).collect();
    final_labels.sort_unstable();
    let n_bonds = mol
        .bonds
        .iter()
        .filter(|&&(a, b, _)| alive[a] && alive[b])
        .count();
    let mut h = Fnv::new();
    h.write_u64(kept.len() as u64);
    h.write_u64(n_bonds as u64);
    for l in final_labels {
        h.write_u64(l);
    }
    Ok(format!("{:016x}", h.finish()))
}

// =========================================================================
// Splits
// =========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitMethod {
    Random,
    Scaffold,
}

/// Fractions of the pool assigned to train / validation / test.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SplitFractions {
    pub train: f64,
    pub valid: f64,
    pub test: f64,
}

impl Default for SplitFractions {
    fn default() -> Self {
        Self {
            train: 0.8,
            valid: 0.1,
            test: 0.1,
        }
    }
}

/// A stored split. Only validation and test SMILES are listed; every other pool molecule is
/// training data. `pool_checksum` ties the split to the exact pool it was made from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataSplit {
    pub name: String,
    pub method: SplitMethod,
    pub seed: u64,
    pub fractions: SplitFractions,
    pub pool_size: usize,
    pub pool_checksum: String,
    pub valid: Vec<String>,
    pub test: Vec<String>,
}

/// Pool indices for each side of a split.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitIndices {
    pub train: Vec<usize>,
    pub valid: Vec<usize>,
    pub test: Vec<usize>,
}

/// Splits `smiles` (a pool without duplicates). `scaffold_keys` is required for
/// [`SplitMethod::Scaffold`] and must align with `smiles`.
pub fn make_split(
    name: &str,
    smiles: &[String],
    scaffold_keys: Option<&[String]>,
    method: SplitMethod,
    seed: u64,
    fractions: SplitFractions,
) -> DataSplit {
    let n = smiles.len();
    assert!(n >= 3, "need at least 3 molecules to split");
    let total = fractions.train + fractions.valid + fractions.test;
    assert!(
        (total - 1.0).abs() < 1e-9,
        "split fractions must sum to 1 (got {total})"
    );
    let valid_size = (fractions.valid * n as f64).round() as usize;
    let test_size = (fractions.test * n as f64).round() as usize;
    let train_size = n - valid_size - test_size;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Deterministic base order regardless of input order.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| smiles[a].cmp(&smiles[b]));

    let (valid, test): (Vec<usize>, Vec<usize>) = match method {
        SplitMethod::Random => {
            order.shuffle(&mut rng);
            (
                order[train_size..train_size + valid_size].to_vec(),
                order[train_size + valid_size..].to_vec(),
            )
        }
        SplitMethod::Scaffold => {
            let keys = scaffold_keys.expect("scaffold split needs scaffold keys");
            assert_eq!(keys.len(), n, "one scaffold key per molecule");
            let mut groups: HashMap<&str, Vec<usize>> = HashMap::new();
            for &i in &order {
                groups.entry(keys[i].as_str()).or_default().push(i);
            }
            let mut groups: Vec<(&str, Vec<usize>)> = groups.into_iter().collect();
            groups.sort_by(|a, b| a.0.cmp(b.0));
            // Balanced scaffold split (as in Chemprop): groups too big to fit in half of the
            // validation or test set go to training first, largest first; the rest are shuffled.
            let (mut big, mut small): (Vec<_>, Vec<_>) = groups
                .into_iter()
                .partition(|(_, g)| g.len() * 2 > valid_size || g.len() * 2 > test_size);
            big.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(b.0)));
            small.shuffle(&mut rng);
            let (mut tr, mut va, mut te) = (Vec::new(), Vec::new(), Vec::new());
            for (_, g) in big.into_iter().chain(small) {
                if tr.len() + g.len() <= train_size {
                    tr.extend(g);
                } else if va.len() + g.len() <= valid_size {
                    va.extend(g);
                } else {
                    te.extend(g);
                }
            }
            (va, te)
        }
    };

    let mut valid: Vec<String> = valid.into_iter().map(|i| smiles[i].clone()).collect();
    let mut test: Vec<String> = test.into_iter().map(|i| smiles[i].clone()).collect();
    valid.sort();
    test.sort();
    DataSplit {
        name: name.to_string(),
        method,
        seed,
        fractions,
        pool_size: n,
        pool_checksum: pool_checksum(smiles.iter().map(String::as_str)),
        valid,
        test,
    }
}

impl DataSplit {
    /// Resolves the split against a pool. Fails if the pool is not the one the split was made from.
    pub fn indices(&self, pool: &[MoleculeRecord]) -> Result<SplitIndices, String> {
        let checksum = pool_checksum(pool.iter().map(|r| r.smiles.as_str()));
        if pool.len() != self.pool_size || checksum != self.pool_checksum {
            return Err(format!(
                "split {} was made for a pool of {} molecules (checksum {}), got {} (checksum {})",
                self.name,
                self.pool_size,
                self.pool_checksum,
                pool.len(),
                checksum
            ));
        }
        let valid: HashSet<&str> = self.valid.iter().map(String::as_str).collect();
        let test: HashSet<&str> = self.test.iter().map(String::as_str).collect();
        let mut out = SplitIndices {
            train: Vec::new(),
            valid: Vec::new(),
            test: Vec::new(),
        };
        for (i, r) in pool.iter().enumerate() {
            let s = r.smiles.as_str();
            if test.contains(s) {
                out.test.push(i);
            } else if valid.contains(s) {
                out.valid.push(i);
            } else {
                out.train.push(i);
            }
        }
        Ok(out)
    }
}

// =========================================================================
// Evaluation and aggregation
// =========================================================================

/// Ridge regression with the penalty chosen by validation MSE (normalised by target variance,
/// averaged over targets). The model is fit on the training rows only.
pub fn fit_ridge_on_validation(
    x_train: &[Vec<f64>],
    y_train: &[Vec<f64>],
    x_valid: &[Vec<f64>],
    y_valid: &[Vec<f64>],
    lambdas: &[f64],
) -> RidgeRegression {
    let k = y_train[0].len();
    let names: Vec<String> = (0..k).map(|t| t.to_string()).collect();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let var: Vec<f64> = (0..k)
        .map(|t| {
            let m = y_valid.iter().map(|r| r[t]).sum::<f64>() / y_valid.len() as f64;
            (y_valid.iter().map(|r| (r[t] - m).powi(2)).sum::<f64>() / y_valid.len() as f64)
                .max(1e-300)
        })
        .collect();
    // Build the (expensive) normal equations once and solve them for each penalty in parallel.
    let problem = RidgeProblem::new(x_train, y_train);
    let scored: Vec<(RidgeRegression, f64)> = crate::parallel::install(|| {
        lambdas
            .par_iter()
            .map(|&lambda| {
                let model = problem.solve(lambda);
                let m = regression_metrics(&model.predict(x_valid), y_valid, &refs);
                let score = m.iter().zip(&var).map(|(tm, v)| tm.mse / v).sum::<f64>() / k as f64;
                (model, score)
            })
            .collect()
    });
    // First minimum in grid order, so ties resolve the same way for any thread count.
    scored
        .into_iter()
        .reduce(|best, cand| if cand.1 < best.1 { cand } else { best })
        .map(|(model, _)| model)
        .expect("lambda grid is non-empty")
}

/// Test-set metrics of one model on one split.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplitResult {
    pub split: String,
    /// Input representation (e.g. `ecfp4_2048`), or `none` for the mean predictor.
    pub features: String,
    pub model: String,
    /// Ridge penalty chosen on validation, if any.
    pub lambda: Option<f64>,
    /// Number of neighbours chosen on validation, if any.
    pub k: Option<usize>,
    pub test: Vec<TargetMetrics>,
}

/// Mean and sample standard deviation of a metric over seeds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MeanStd {
    pub mean: f64,
    pub std: f64,
}

impl MeanStd {
    pub fn of(values: &[f64]) -> Self {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let std = if values.len() > 1 {
            (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
        } else {
            0.0
        };
        Self { mean, std }
    }
}

/// Per-target summary of one model across the seeds of one split method.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AggregateResult {
    pub method: SplitMethod,
    pub features: String,
    pub model: String,
    pub seeds: usize,
    pub targets: Vec<String>,
    pub mae: Vec<MeanStd>,
    pub r2: Vec<MeanStd>,
}

/// Groups split results by (features, model) and summarises each target over the seeds.
pub fn aggregate(method: SplitMethod, results: &[SplitResult]) -> Vec<AggregateResult> {
    let mut groups: Vec<(&str, &str)> = Vec::new();
    for r in results {
        let g = (r.features.as_str(), r.model.as_str());
        if !groups.contains(&g) {
            groups.push(g);
        }
    }
    groups
        .into_iter()
        .map(|(features, model)| {
            let runs: Vec<&SplitResult> = results
                .iter()
                .filter(|r| r.features == features && r.model == model)
                .collect();
            let targets: Vec<String> = runs[0].test.iter().map(|m| m.target.clone()).collect();
            let per = |f: fn(&TargetMetrics) -> f64| -> Vec<MeanStd> {
                (0..targets.len())
                    .map(|t| MeanStd::of(&runs.iter().map(|r| f(&r.test[t])).collect::<Vec<_>>()))
                    .collect()
            };
            AggregateResult {
                method,
                features: features.to_string(),
                model: model.to_string(),
                seeds: runs.len(),
                targets: targets.clone(),
                mae: per(|m| m.mae),
                r2: per(|m| m.r2),
            }
        })
        .collect()
}

/// Classical baseline models evaluated under the protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaselineModel {
    /// Predicts the training mean.
    Mean,
    /// Ridge on the standardised features; penalty chosen on validation.
    RidgeLinear,
    /// Ridge on features plus all pairwise products; penalty chosen on validation.
    RidgeQuadratic,
    /// Similarity-weighted mean of the k most Tanimoto-similar training molecules, using the
    /// non-zero positions of the features as the bit set (intended for fingerprints); k chosen
    /// on validation from [`KNN_K_GRID`].
    KnnTanimoto,
}

/// Candidate neighbour counts for [`BaselineModel::KnnTanimoto`].
pub const KNN_K_GRID: [usize; 6] = [1, 3, 5, 10, 20, 50];

impl BaselineModel {
    pub fn name(self) -> &'static str {
        match self {
            BaselineModel::Mean => "Mean predictor",
            BaselineModel::RidgeLinear => "Ridge (linear)",
            BaselineModel::RidgeQuadratic => "Ridge (quadratic)",
            BaselineModel::KnnTanimoto => "kNN (Tanimoto)",
        }
    }
}

fn normalised_mse(preds: &[Vec<f64>], targets: &[Vec<f64>]) -> f64 {
    let k = targets[0].len();
    let n = targets.len() as f64;
    (0..k)
        .map(|t| {
            let mean = targets.iter().map(|r| r[t]).sum::<f64>() / n;
            let var = (targets.iter().map(|r| (r[t] - mean).powi(2)).sum::<f64>() / n).max(1e-300);
            preds
                .iter()
                .zip(targets)
                .map(|(p, y)| (p[t] - y[t]).powi(2))
                .sum::<f64>()
                / n
                / var
        })
        .sum::<f64>()
        / k as f64
}

/// For each query, the `k_max` most similar training molecules as (similarity, train position),
/// most similar first; ties broken by training position.
fn tanimoto_neighbours(
    train: &[Vec<u32>],
    queries: &[Vec<u32>],
    k_max: usize,
) -> Vec<Vec<(f64, usize)>> {
    crate::parallel::install(|| {
        queries
            .par_iter()
            .map(|q| {
                let mut sims: Vec<(f64, usize)> = train
                    .iter()
                    .enumerate()
                    .map(|(i, t)| (crate::molfeatures::tanimoto(q, t), i))
                    .collect();
                let cmp =
                    |a: &(f64, usize), b: &(f64, usize)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
                let k = k_max.min(sims.len());
                if k < sims.len() {
                    sims.select_nth_unstable_by(k - 1, cmp);
                    sims.truncate(k);
                }
                sims.sort_by(cmp);
                sims
            })
            .collect()
    })
}

fn knn_predict(neighbours: &[Vec<(f64, usize)>], y_train: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    neighbours
        .iter()
        .map(|nb| {
            let top = &nb[..k.min(nb.len())];
            let wsum: f64 = top.iter().map(|(s, _)| s).sum();
            let n_t = y_train[0].len();
            (0..n_t)
                .map(|t| {
                    if wsum > 0.0 {
                        top.iter().map(|(s, i)| s * y_train[*i][t]).sum::<f64>() / wsum
                    } else {
                        top.iter().map(|(_, i)| y_train[*i][t]).sum::<f64>() / top.len() as f64
                    }
                })
                .collect()
        })
        .collect()
}

/// Fits `model` on the training part of `split` using `features` (one row per pool molecule, in
/// pool order), selects its hyperparameter on validation, and scores the test set once.
pub fn baseline_on_split(
    model: BaselineModel,
    feature_name: &str,
    features: &[Vec<f64>],
    pool: &[MoleculeRecord],
    split: &DataSplit,
) -> Result<SplitResult, String> {
    if features.len() != pool.len() {
        return Err(format!(
            "{} feature rows for a pool of {}",
            features.len(),
            pool.len()
        ));
    }
    let idx = split.indices(pool)?;
    let take = |ids: &[usize], f: &dyn Fn(usize) -> Vec<f64>| -> Vec<Vec<f64>> {
        ids.iter().map(|&i| f(i)).collect()
    };
    let targ = |i: usize| pool[i].targets.to_vec();
    let (y_tr, y_va, y_te) = (
        take(&idx.train, &targ),
        take(&idx.valid, &targ),
        take(&idx.test, &targ),
    );
    let result =
        |preds: Vec<Vec<f64>>, features: &str, lambda: Option<f64>, k: Option<usize>| SplitResult {
            split: split.name.clone(),
            features: features.to_string(),
            model: model.name().to_string(),
            lambda,
            k,
            test: regression_metrics(&preds, &y_te, &ZINC_TARGETS),
        };

    Ok(match model {
        BaselineModel::Mean => {
            let mean: Vec<f64> = (0..3)
                .map(|t| y_tr.iter().map(|r| r[t]).sum::<f64>() / y_tr.len() as f64)
                .collect();
            result(vec![mean; y_te.len()], "none", None, None)
        }
        BaselineModel::RidgeLinear | BaselineModel::RidgeQuadratic => {
            let quadratic = model == BaselineModel::RidgeQuadratic;
            let row = |i: usize| {
                if quadratic {
                    quadratic_features(&features[i])
                } else {
                    features[i].clone()
                }
            };
            let (x_tr, x_va, x_te) = (
                take(&idx.train, &row),
                take(&idx.valid, &row),
                take(&idx.test, &row),
            );
            let ridge = fit_ridge_on_validation(&x_tr, &y_tr, &x_va, &y_va, &RIDGE_LAMBDAS);
            result(ridge.predict(&x_te), feature_name, Some(ridge.lambda), None)
        }
        BaselineModel::KnnTanimoto => {
            let bits = |ids: &[usize]| -> Vec<Vec<u32>> {
                ids.iter()
                    .map(|&i| crate::molfeatures::on_bits(&features[i]))
                    .collect()
            };
            let (b_tr, b_va, b_te) = (bits(&idx.train), bits(&idx.valid), bits(&idx.test));
            let k_max = *KNN_K_GRID.iter().max().expect("non-empty grid");
            let nb_va = tanimoto_neighbours(&b_tr, &b_va, k_max);
            let best_k = KNN_K_GRID
                .iter()
                .map(|&k| (k, normalised_mse(&knn_predict(&nb_va, &y_tr, k), &y_va)))
                .reduce(|best, cand| if cand.1 < best.1 { cand } else { best })
                .map(|(k, _)| k)
                .expect("non-empty grid");
            let nb_te = tanimoto_neighbours(&b_tr, &b_te, best_k);
            result(
                knn_predict(&nb_te, &y_tr, best_k),
                feature_name,
                None,
                Some(best_k),
            )
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(s: &str) -> String {
        murcko_scaffold_key(s).unwrap()
    }

    #[test]
    fn scaffold_strips_side_chains_and_is_order_invariant() {
        let benzene = key("c1ccccc1");
        assert!(!benzene.is_empty());
        assert_eq!(key("Cc1ccccc1"), benzene);
        assert_eq!(key("c1ccccc1C"), benzene);
        assert_eq!(
            key("CC(=O)Oc1ccccc1C(=O)O"),
            benzene,
            "aspirin's scaffold is benzene"
        );
        assert_eq!(
            key("C1CCNCC1"),
            key("N1CCCCC1"),
            "ring written from a different atom"
        );
        assert_eq!(
            key("[Na+].[O-]C(=O)c1ccccc1"),
            benzene,
            "counter-ion and side chain removed"
        );
    }

    #[test]
    fn scaffold_distinguishes_different_cores() {
        let benzene = key("c1ccccc1");
        assert_ne!(key("c1ccncc1"), benzene, "pyridine");
        assert_ne!(key("c1ccc2ccccc2c1"), benzene, "naphthalene");
        assert_ne!(key("C1CCCCC1"), benzene, "cyclohexane");
        assert_ne!(
            key("c1ccc(-c2ccccc2)cc1"),
            key("c1ccc(Cc2ccccc2)cc1"),
            "biphenyl vs diphenylmethane linker"
        );
        assert_eq!(
            key("Cc1ccc(Cc2ccccc2)cc1O"),
            key("c1ccc(Cc2ccccc2)cc1"),
            "substituents removed, linker kept"
        );
    }

    #[test]
    fn acyclic_molecules_have_empty_scaffold() {
        assert_eq!(key("CCO"), "");
        assert_eq!(key("CC(C)(C)C"), "");
        assert_eq!(key("[Na+].[Cl-]"), "");
    }

    fn toy_pool() -> (Vec<String>, Vec<String>) {
        // 32 small scaffold groups (ring sizes 3-10, with or without one N/O/S) x 4 substituent
        // lengths, plus 30 acyclic molecules (one large group that must go to training).
        let mut smiles = Vec::new();
        for ring in 3..=10 {
            for hetero in ["", "N", "O", "S"] {
                let core = format!("C1{hetero}{}C1", "C".repeat(ring - 2 - hetero.len()));
                for k in 1..=4 {
                    smiles.push(format!("{}{}", "C".repeat(k), core));
                }
            }
        }
        for k in 1..=30 {
            smiles.push(format!("{}O", "C".repeat(k)));
        }
        let keys = smiles.iter().map(|s| key(s)).collect();
        (smiles, keys)
    }

    #[test]
    fn splits_partition_the_pool_and_are_deterministic() {
        let (smiles, keys) = toy_pool();
        for method in [SplitMethod::Random, SplitMethod::Scaffold] {
            let a = make_split(
                "t",
                &smiles,
                Some(&keys),
                method,
                7,
                SplitFractions::default(),
            );
            let b = make_split(
                "t",
                &smiles,
                Some(&keys),
                method,
                7,
                SplitFractions::default(),
            );
            assert_eq!(a, b, "{method:?} split must be deterministic for a seed");
            let v: HashSet<_> = a.valid.iter().collect();
            let t: HashSet<_> = a.test.iter().collect();
            assert!(v.is_disjoint(&t));
            let pool: Vec<MoleculeRecord> = smiles
                .iter()
                .map(|s| MoleculeRecord {
                    smiles: s.clone(),
                    features: vec![0.0],
                    targets: [0.0; 3],
                })
                .collect();
            let idx = a.indices(&pool).unwrap();
            assert_eq!(
                idx.train.len() + idx.valid.len() + idx.test.len(),
                pool.len()
            );
        }
        let r0 = make_split(
            "t",
            &smiles,
            None,
            SplitMethod::Random,
            0,
            SplitFractions::default(),
        );
        let r1 = make_split(
            "t",
            &smiles,
            None,
            SplitMethod::Random,
            1,
            SplitFractions::default(),
        );
        assert_ne!(
            r0.test, r1.test,
            "different seeds give different random splits"
        );
        // 158 molecules: round(0.1 * 158) = 16 each for validation and test.
        assert_eq!(r0.test.len(), 16);
        assert_eq!(r0.valid.len(), 16);
    }

    #[test]
    fn scaffold_split_keeps_scaffolds_on_one_side() {
        let (smiles, keys) = toy_pool();
        let split = make_split(
            "s",
            &smiles,
            Some(&keys),
            SplitMethod::Scaffold,
            3,
            SplitFractions::default(),
        );
        let key_of: HashMap<&str, &str> = smiles
            .iter()
            .map(String::as_str)
            .zip(keys.iter().map(String::as_str))
            .collect();
        let test_keys: HashSet<&str> = split.test.iter().map(|s| key_of[s.as_str()]).collect();
        let valid_keys: HashSet<&str> = split.valid.iter().map(|s| key_of[s.as_str()]).collect();
        let held: HashSet<&str> = split
            .valid
            .iter()
            .chain(&split.test)
            .map(String::as_str)
            .collect();
        let train_keys: HashSet<&str> = smiles
            .iter()
            .filter(|s| !held.contains(s.as_str()))
            .map(|s| key_of[s.as_str()])
            .collect();
        assert!(
            train_keys.is_disjoint(&test_keys),
            "a scaffold appears in both train and test"
        );
        assert!(
            train_keys.is_disjoint(&valid_keys),
            "a scaffold appears in both train and valid"
        );
        assert!(!split.test.is_empty() && !split.valid.is_empty());
    }

    #[test]
    fn split_rejects_a_different_pool() {
        let (smiles, _) = toy_pool();
        let split = make_split(
            "t",
            &smiles,
            None,
            SplitMethod::Random,
            0,
            SplitFractions::default(),
        );
        let other: Vec<MoleculeRecord> = smiles[1..]
            .iter()
            .map(|s| MoleculeRecord {
                smiles: s.clone(),
                features: vec![0.0],
                targets: [0.0; 3],
            })
            .collect();
        assert!(split.indices(&other).is_err());
    }

    #[test]
    fn knn_neighbours_and_predictions() {
        let train = vec![vec![1, 2, 3], vec![1, 2], vec![7, 8], vec![]];
        let y = vec![
            vec![1.0, 10.0, 0.0],
            vec![2.0, 20.0, 0.0],
            vec![3.0, 30.0, 0.0],
            vec![4.0, 40.0, 0.0],
        ];
        let nb = tanimoto_neighbours(&train, &[vec![1, 2, 3]], 3);
        // Similarities: 1.0 (exact), 2/3, 0, 0 -> ties at 0 broken by training position.
        assert_eq!(
            nb[0].iter().map(|&(_, i)| i).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert!((nb[0][1].0 - 2.0 / 3.0).abs() < 1e-12);
        let p1 = knn_predict(&nb, &y, 1);
        assert_eq!(p1[0], vec![1.0, 10.0, 0.0]);
        let p2 = knn_predict(&nb, &y, 2);
        let w = 1.0 + 2.0 / 3.0;
        assert!((p2[0][0] - (1.0 * 1.0 + 2.0 / 3.0 * 2.0) / w).abs() < 1e-12);
        // All-zero similarity falls back to a plain mean.
        let far = tanimoto_neighbours(&train, &[vec![99]], 2);
        assert_eq!(knn_predict(&far, &y, 2)[0][0], 1.5);
    }

    #[test]
    fn baseline_on_split_recovers_a_linear_target() {
        // Targets are linear in the single feature: linear ridge must be near-perfect on test.
        let (smiles, _) = toy_pool();
        let pool: Vec<MoleculeRecord> = smiles
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let x = i as f64 * 0.1;
                MoleculeRecord {
                    smiles: s.clone(),
                    features: vec![x],
                    targets: [2.0 * x + 1.0, -x, 3.0 * x],
                }
            })
            .collect();
        let feats: Vec<Vec<f64>> = pool.iter().map(|r| r.features.clone()).collect();
        let split = make_split(
            "t",
            &smiles,
            None,
            SplitMethod::Random,
            0,
            SplitFractions::default(),
        );
        let r = baseline_on_split(BaselineModel::RidgeLinear, "x", &feats, &pool, &split).unwrap();
        assert!(r.test.iter().all(|m| m.r2 > 0.999), "{:?}", r.test);
        let mean = baseline_on_split(BaselineModel::Mean, "x", &feats, &pool, &split).unwrap();
        assert_eq!(mean.features, "none");
        assert!(mean.test.iter().all(|m| m.r2 < 0.05));
    }

    #[test]
    fn mean_std_matches_hand_values() {
        let m = MeanStd::of(&[1.0, 2.0, 3.0]);
        assert!((m.mean - 2.0).abs() < 1e-12);
        assert!((m.std - 1.0).abs() < 1e-12);
    }
}
