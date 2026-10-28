// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Multi-Domain Dataset Loaders & Adapters (`t1000::datasets`)
//!
//! Unified interface for featurized datasets across quantum chemistry, financial fraud graphs,
//! network botnet telemetry, and supply chain logistics:
//!
//! | Module | Dataset Domain | Description |
//! | :--- | :--- | :--- |
//! | [`qm9`] | Molecular Quantum | MoleculeNet 134k organic molecules with STO-3G quantum properties. |
//! | [`pitt`] | Molecular Quantum | Pitt Quantum Repository (106k molecules, PM7) with 3D dipole vectors. |
//! | [`bank_fraud`] | Graph Cyber Defense | Transaction graph telemetry with synthetic fraud injection labels. |
//! | [`iot_botnet`] | Network Security | IoT network traffic features with multi-label attack taxonomy. |
//! | [`harmonized`] | Logistics / Trade | Harmonized tariff code logistics and sector classification. |

pub mod bank_fraud;
pub mod harmonized;
pub mod iot_botnet;
pub mod pitt;
pub mod qm9;

pub use bank_fraud as bank_fraud_data;
pub use harmonized as harmonized_data;
pub use iot_botnet as iot_botnet_data;
pub use pitt as pitt_data;
pub use qm9 as qm9_data;

use crate::baselines::ZINC_TARGETS;
use self::pitt::{PittMolecule, PROPERTIES as PITT_PROPERTIES};
use self::qm9::{FeatureSet, Qm9Molecule, PROPERTIES};
use crate::zinc_data;

const ZINC_TRAIN_CSV: &str = "tests/benchmarks/zinc_molecular/outputs/zinc_train_10k.csv";
const ZINC_TEST_CSV: &str = "tests/benchmarks/zinc_molecular/outputs/zinc_test_2k.csv";
const QM9_CSV: &str = "data/qm9/qm9.csv";
const PITT_CSV: &str = "data/pitt/pitt_quantum.csv";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dataset {
    Zinc,
    Qm9,
    Pitt,
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
    /// Pitt only: indices into [`PITT_PROPERTIES`].
    pub pitt_targets: [usize; 3],
    pub qm9_features: FeatureSet,
    pub seed: u64,
    pub validation: usize,
}

/// Indices into [`PROPERTIES`] for a comma-separated list of exactly three property names.
pub fn parse_targets(list: Option<&str>) -> [usize; 3] {
    let names: Vec<&str> = list
        .unwrap_or("mu,gap,alpha")
        .split(',')
        .map(str::trim)
        .collect();
    assert_eq!(
        names.len(),
        3,
        "--targets needs three property names from {PROPERTIES:?}"
    );
    let index = |n: &str| {
        PROPERTIES
            .iter()
            .position(|p| *p == n)
            .unwrap_or_else(|| panic!("unknown property {n}; choose from {PROPERTIES:?}"))
    };
    [index(names[0]), index(names[1]), index(names[2])]
}

/// Indices into [`PITT_PROPERTIES`] for a comma-separated list of exactly three property names.
pub fn parse_pitt_targets(list: Option<&str>) -> [usize; 3] {
    let names: Vec<&str> = list
        .unwrap_or("homo,lumo,gap")
        .split(',')
        .map(str::trim)
        .collect();
    assert_eq!(
        names.len(),
        3,
        "--targets needs three property names from {PITT_PROPERTIES:?}"
    );
    let index = |n: &str| {
        PITT_PROPERTIES
            .iter()
            .position(|p| *p == n)
            .unwrap_or_else(|| panic!("unknown property {n}; choose from {PITT_PROPERTIES:?}"))
    };
    [index(names[0]), index(names[1]), index(names[2])]
}

impl DataOptions {
    /// Reads `--dataset zinc|qm9|pitt`, `--samples`, `--eval-samples`, `--targets a,b,c`,
    /// `--features zinc|qm9` and `--seed` from the command line.
    pub fn from_args(args: &[String]) -> Self {
        let value = |flag: &str| {
            args.iter()
                .position(|a| a == flag)
                .and_then(|i| args.get(i + 1))
                .cloned()
        };
        let ds_name = value("--dataset");
        let dataset = match ds_name.as_deref() {
            Some("qm9") => Dataset::Qm9,
            Some("pitt") => Dataset::Pitt,
            _ => Dataset::Zinc,
        };
        let is_large = dataset == Dataset::Qm9 || dataset == Dataset::Pitt;
        let number = |flag: &str, default: usize| {
            value(flag).and_then(|v| v.parse().ok()).unwrap_or(default)
        };
        let targets_arg = value("--targets");
        let qm9_targets = if dataset == Dataset::Qm9 {
            parse_targets(targets_arg.as_deref())
        } else {
            parse_targets(None)
        };
        let pitt_targets = if dataset == Dataset::Pitt {
            parse_pitt_targets(targets_arg.as_deref())
        } else {
            parse_pitt_targets(None)
        };
        Self {
            dataset,
            samples: number("--samples", if is_large { 20_000 } else { 10_000 }),
            eval_samples: number("--eval-samples", if is_large { 10_000 } else { 2_000 }),
            qm9_targets,
            pitt_targets,
            qm9_features: if value("--features").as_deref() == Some("qm9") {
                FeatureSet::Qm9
            } else {
                FeatureSet::ZincLike
            },
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
                        .map(|m| Sample {
                            targets: m.targets(),
                            features: m.features,
                        })
                        .collect()
                };
                let mut train = load(ZINC_TRAIN_CSV, self.samples);
                let validation =
                    train.split_off(train.len() - self.validation.min(train.len() / 2));
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
                println!(
                    " QM9 rows         : {} featurised, {skipped} skipped",
                    rows.len()
                );
                let (mut train, test) = qm9_data::split(rows, self.eval_samples, self.seed);
                let validation = if train.len() > self.samples {
                    train.split_off(self.samples)
                } else {
                    Vec::new()
                };
                train.truncate(self.samples);
                let sample = |m: Qm9Molecule| Sample {
                    targets: m.targets(self.qm9_targets),
                    features: m.features,
                };
                Data {
                    train: train.into_iter().map(sample).collect(),
                    validation: validation
                        .into_iter()
                        .take(self.validation)
                        .map(sample)
                        .collect(),
                    test: test.into_iter().map(sample).collect(),
                    target_names: self.qm9_targets.map(|i| PROPERTIES[i].to_string()),
                    label: format!("QM9 ({:?} descriptors)", self.qm9_features),
                }
            }
            Dataset::Pitt => {
                let (rows, skipped) = pitt_data::load(PITT_CSV, self.qm9_features, usize::MAX)
                    .unwrap_or_else(|e| panic!("{e} (see data/pitt/README.md to fetch the data)"));
                println!(
                    " Pitt rows        : {} featurised, {skipped} skipped",
                    rows.len()
                );
                let (mut train, test) = pitt_data::split(rows, self.eval_samples, self.seed);
                let validation = if train.len() > self.samples {
                    train.split_off(self.samples)
                } else {
                    Vec::new()
                };
                train.truncate(self.samples);
                let sample = |m: PittMolecule| Sample {
                    targets: m.targets(self.pitt_targets),
                    features: m.features,
                };
                Data {
                    train: train.into_iter().map(sample).collect(),
                    validation: validation
                        .into_iter()
                        .take(self.validation)
                        .map(sample)
                        .collect(),
                    test: test.into_iter().map(sample).collect(),
                    target_names: self.pitt_targets.map(|i| PITT_PROPERTIES[i].to_string()),
                    label: format!("Pitt PM7 ({:?} descriptors)", self.qm9_features),
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
        assert_eq!(
            (zinc.dataset, zinc.samples, zinc.eval_samples),
            (Dataset::Zinc, 10_000, 2_000)
        );
        let qm9 = DataOptions::from_args(&args(&["--dataset", "qm9"]));
        assert_eq!(
            (qm9.dataset, qm9.samples, qm9.eval_samples),
            (Dataset::Qm9, 20_000, 10_000)
        );
        assert_eq!(qm9.qm9_targets, [0, 4, 1]); // mu, gap, alpha
    }

    #[test]
    fn explicit_options_override_the_defaults() {
        let o = DataOptions::from_args(&args(&[
            "--dataset",
            "qm9",
            "--samples",
            "500",
            "--eval-samples",
            "100",
            "--targets",
            "homo,lumo,r2",
            "--features",
            "qm9",
            "--seed",
            "3",
        ]));
        assert_eq!((o.samples, o.eval_samples, o.seed), (500, 100, 3));
        assert_eq!(o.qm9_targets, [2, 3, 5]);
        assert_eq!(o.qm9_features, FeatureSet::Qm9);
    }

    #[test]
    fn the_validation_size_is_read_from_the_command_line() {
        assert_eq!(DataOptions::from_args(&args(&[])).validation, 0);
        assert_eq!(
            DataOptions::from_args(&args(&["--validation", "5000"])).validation,
            5000
        );
    }

    #[test]
    fn defaults_and_options_for_pitt_dataset() {
        let pitt = DataOptions::from_args(&args(&["--dataset", "pitt"]));
        assert_eq!(
            (pitt.dataset, pitt.samples, pitt.eval_samples),
            (Dataset::Pitt, 20_000, 10_000)
        );
        assert_eq!(pitt.pitt_targets, [0, 1, 2]); // homo, lumo, gap

        let custom = DataOptions::from_args(&args(&[
            "--dataset",
            "pitt",
            "--targets",
            "dipole,polarizability,molwt",
            "--samples",
            "1000",
            "--eval-samples",
            "200",
        ]));
        assert_eq!(custom.pitt_targets, [3, 4, 6]);
        assert_eq!((custom.samples, custom.eval_samples), (1000, 200));
    }

    #[test]
    #[should_panic(expected = "unknown property")]
    fn an_unknown_property_is_rejected() {
        parse_targets(Some("mu,gap,banana"));
    }
}
