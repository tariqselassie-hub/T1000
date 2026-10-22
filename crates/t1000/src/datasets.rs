// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! The molecular data sets of the feature-map studies behind one interface: descriptors plus
//! three regression targets, split into training and held-out molecules.
//!
//! * `zinc`: the 10,000 / 2,000 featurised ZINC CSVs (logP, QED, SAS).
//! * `qm9`: the MoleculeNet `qm9.csv` (see `data/qm9/README.md`), shuffled with a fixed seed and
//!   split; any three of the nine properties of [`PROPERTIES`] can be chosen.

use crate::baselines::ZINC_TARGETS;
use crate::qm9_data::{self, FeatureSet, Qm9Molecule, PROPERTIES};
use crate::zinc_data;

const ZINC_TRAIN_CSV: &str = "tests/benchmarks/zinc_molecular/outputs/zinc_train_10k.csv";
const ZINC_TEST_CSV: &str = "tests/benchmarks/zinc_molecular/outputs/zinc_test_2k.csv";
const QM9_CSV: &str = "data/qm9/qm9.csv";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dataset {
    Zinc,
    Qm9,
}

/// One molecule: descriptors and the three regression targets.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    pub features: Vec<f64>,
    pub targets: [f64; 3],
}

/// A loaded data set.
#[derive(Debug)]
pub struct Data {
    pub train: Vec<Sample>,
    /// Molecules for choosing hyperparameters, disjoint from `train` and `test` (empty unless
    /// `--validation N` is given). Choose settings on these, then report `test` once.
    pub validation: Vec<Sample>,
    pub test: Vec<Sample>,
    pub target_names: [String; 3],
    /// Human-readable description, for example `QM9 (ZincLike descriptors)`.
    pub label: String,
}

/// Command-line selectable data set options.
#[derive(Debug, Clone)]
pub struct DataOptions {
    pub dataset: Dataset,
    pub samples: usize,
    pub eval_samples: usize,
    /// QM9 only: indices into [`PROPERTIES`].
    pub qm9_targets: [usize; 3],
    pub qm9_features: FeatureSet,
    pub seed: u64,
    pub validation: usize,
}

/// Indices into [`PROPERTIES`] for a comma-separated list of exactly three property names.
pub fn parse_targets(list: Option<&str>) -> [usize; 3] {
    let names: Vec<&str> = list.unwrap_or("mu,gap,alpha").split(',').map(str::trim).collect();
    assert_eq!(names.len(), 3, "--targets needs three property names from {PROPERTIES:?}");
    let index = |n: &str| PROPERTIES.iter().position(|p| *p == n).unwrap_or_else(|| panic!("unknown property {n}; choose from {PROPERTIES:?}"));
    [index(names[0]), index(names[1]), index(names[2])]
}

impl DataOptions {
    /// Reads `--dataset zinc|qm9`, `--samples`, `--eval-samples`, `--targets a,b,c`,
    /// `--features zinc|qm9` and `--seed` from the command line.
    pub fn from_args(args: &[String]) -> Self {
        let value = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned();
        let qm9 = value("--dataset").as_deref() == Some("qm9");
        let number = |flag: &str, default: usize| value(flag).and_then(|v| v.parse().ok()).unwrap_or(default);
        Self {
            dataset: if qm9 { Dataset::Qm9 } else { Dataset::Zinc },
            samples: number("--samples", if qm9 { 20_000 } else { 10_000 }),
            eval_samples: number("--eval-samples", if qm9 { 10_000 } else { 2_000 }),
            qm9_targets: parse_targets(value("--targets").as_deref()),
            qm9_features: if value("--features").as_deref() == Some("qm9") { FeatureSet::Qm9 } else { FeatureSet::ZincLike },
            seed: number("--seed", 7) as u64,
            validation: number("--validation", 0),
        }
    }

    /// Loads and splits the data. Panics with a readable message if a file is missing.
    pub fn load(&self) -> Data {
        match self.dataset {
            Dataset::Zinc => {
                let load = |path: &str, max: usize| -> Vec<Sample> {
                    zinc_data::load(path, max)
                        .unwrap_or_else(|e| panic!("{e}"))
                        .into_iter()
                        .map(|m| Sample { targets: m.targets(), features: m.features })
                        .collect()
                };
                // ZINC has no spare molecules: a validation set is taken from the end of the
                // training file, which shortens the training set by the same amount.
                let mut train = load(ZINC_TRAIN_CSV, self.samples);
                let validation = train.split_off(train.len() - self.validation.min(train.len() / 2));
                Data {
                    train,
                    validation,
                    test: load(ZINC_TEST_CSV, self.eval_samples),
                    target_names: ZINC_TARGETS.map(String::from),
                    label: "ZINC".to_string(),
                }
            }
            Dataset::Qm9 => {
                let (rows, skipped) = qm9_data::load(QM9_CSV, self.qm9_features, usize::MAX)
                    .unwrap_or_else(|e| panic!("{e} (see data/qm9/README.md to fetch the data)"));
                println!(" QM9 rows         : {} featurised, {skipped} skipped", rows.len());
                let (mut train, test) = qm9_data::split(rows, self.eval_samples, self.seed);
                let validation = if train.len() > self.samples { train.split_off(self.samples) } else { Vec::new() };
                train.truncate(self.samples);
                let sample = |m: Qm9Molecule| Sample { targets: m.targets(self.qm9_targets), features: m.features };
                Data {
                    train: train.into_iter().map(sample).collect(),
                    validation: validation.into_iter().take(self.validation).map(sample).collect(),
                    test: test.into_iter().map(sample).collect(),
                    target_names: self.qm9_targets.map(|i| PROPERTIES[i].to_string()),
                    label: format!("QM9 ({:?} descriptors)", self.qm9_features),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn defaults_follow_the_dataset() {
        let zinc = DataOptions::from_args(&args(&[]));
        assert_eq!((zinc.dataset, zinc.samples, zinc.eval_samples), (Dataset::Zinc, 10_000, 2_000));
        let qm9 = DataOptions::from_args(&args(&["--dataset", "qm9"]));
        assert_eq!((qm9.dataset, qm9.samples, qm9.eval_samples), (Dataset::Qm9, 20_000, 10_000));
        assert_eq!(qm9.qm9_targets, [0, 4, 1]); // mu, gap, alpha
    }

    #[test]
    fn explicit_options_override_the_defaults() {
        let o = DataOptions::from_args(&args(&["--dataset", "qm9", "--samples", "500", "--eval-samples", "100", "--targets", "homo,lumo,r2", "--features", "qm9", "--seed", "3"]));
        assert_eq!((o.samples, o.eval_samples, o.seed), (500, 100, 3));
        assert_eq!(o.qm9_targets, [2, 3, 5]);
        assert_eq!(o.qm9_features, FeatureSet::Qm9);
    }

    #[test]
    fn the_validation_size_is_read_from_the_command_line() {
        assert_eq!(DataOptions::from_args(&args(&[])).validation, 0);
        assert_eq!(DataOptions::from_args(&args(&["--validation", "5000"])).validation, 5000);
    }

    #[test]
    #[should_panic(expected = "unknown property")]
    fn an_unknown_property_is_rejected() {
        parse_targets(Some("mu,gap,banana"));
    }
}
