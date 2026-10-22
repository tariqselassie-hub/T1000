// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Loader for the featurized ZINC CSV files used by the molecular regression binaries.
//!
//! Columns: `smiles`, 19 numeric features, then the targets `logP`, `QED`, `SAS`.

use std::fs;
use std::path::Path;

/// Number of molecule features per row.
pub const N_FEATURES: usize = 19;

/// One featurized molecule with its three regression targets.
#[derive(Debug, Clone, PartialEq)]
pub struct ZincMolecule {
    pub smiles: String,
    pub features: Vec<f64>,
    pub log_p: f64,
    pub qed: f64,
    pub sas: f64,
}

impl ZincMolecule {
    /// Targets in the order `[logP, QED, SAS]`.
    pub fn targets(&self) -> [f64; 3] {
        [self.log_p, self.qed, self.sas]
    }
}

/// Reads up to `max_rows` molecules. Rows with fewer than 23 columns are skipped; numbers that
/// fail to parse become 0.
pub fn load(path: impl AsRef<Path>, max_rows: usize) -> Result<Vec<ZincMolecule>, String> {
    let path = path.as_ref();
    let text = fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let rows: Vec<ZincMolecule> = text
        .lines()
        .skip(1)
        .filter_map(parse_row)
        .take(max_rows)
        .collect();
    if rows.is_empty() {
        return Err(format!("no molecules parsed from {}", path.display()));
    }
    Ok(rows)
}

fn parse_row(line: &str) -> Option<ZincMolecule> {
    let p: Vec<&str> = line.split(',').map(str::trim).collect();
    if p.len() < N_FEATURES + 4 {
        return None;
    }
    let num = |s: &str| s.parse::<f64>().unwrap_or(0.0);
    Some(ZincMolecule {
        smiles: p[0].to_string(),
        features: p[1..=N_FEATURES].iter().map(|v| num(v)).collect(),
        log_p: num(p[N_FEATURES + 1]),
        qed: num(p[N_FEATURES + 2]),
        sas: num(p[N_FEATURES + 3]),
    })
}

/// How raw molecule features are rescaled before amplitude encoding.
///
/// Amplitude encoding L2-normalises the vector, so one large column (molecular weight is in the
/// hundreds next to counts below 25) can carry almost all of the norm. Z and ZZ readouts depend
/// only on squared amplitudes, so the scaled inputs must stay non-negative.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum InputScaling {
    /// Use the features as they are.
    Raw,
    /// `(x - min) / (max - min)` per feature, clamped to `[0, 1]`.
    MinMax,
    /// `x / max` per feature (features are non-negative), clamped to `[0, 1]`.
    MaxAbs,
}

/// A fitted rescaling of feature vectors (statistics from the training set only).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Scaler {
    pub kind: InputScaling,
    pub offset: Vec<f64>,
    pub range: Vec<f64>,
}

impl Scaler {
    pub fn fit(kind: InputScaling, rows: &[ZincMolecule]) -> Self {
        let features: Vec<&[f64]> = rows.iter().map(|m| m.features.as_slice()).collect();
        Self::fit_rows(kind, &features)
    }

    /// As [`Scaler::fit`] for plain feature rows (any dataset).
    pub fn fit_rows(kind: InputScaling, rows: &[&[f64]]) -> Self {
        let d = rows[0].len();
        let stat = |j: usize, f: fn(f64, f64) -> f64, init: f64| rows.iter().map(|r| r[j]).fold(init, f);
        let (mut offset, mut range) = (vec![0.0; d], vec![1.0; d]);
        for j in 0..d {
            let (lo, hi) = (stat(j, f64::min, f64::INFINITY), stat(j, f64::max, f64::NEG_INFINITY));
            match kind {
                InputScaling::Raw => {}
                InputScaling::MinMax => (offset[j], range[j]) = (lo, hi - lo),
                InputScaling::MaxAbs => range[j] = hi.abs().max(lo.abs()),
            }
            if range[j] <= 1e-12 {
                range[j] = 1.0;
            }
        }
        Self { kind, offset, range }
    }

    /// Rescales one feature vector; `Raw` returns it unchanged.
    pub fn apply(&self, features: &[f64]) -> Vec<f64> {
        if self.kind == InputScaling::Raw {
            return features.to_vec();
        }
        features
            .iter()
            .zip(self.offset.iter().zip(&self.range))
            .map(|(v, (o, r))| ((v - o) / r).clamp(0.0, 1.0))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn molecule(features: Vec<f64>) -> ZincMolecule {
        ZincMolecule { smiles: String::new(), features, log_p: 0.0, qed: 0.0, sas: 0.0 }
    }

    #[test]
    fn scalers_map_training_data_into_the_unit_interval() {
        let rows = vec![molecule(vec![2.0, 100.0, 5.0]), molecule(vec![4.0, 300.0, 5.0]), molecule(vec![3.0, 200.0, 5.0])];
        let minmax = Scaler::fit(InputScaling::MinMax, &rows);
        assert_eq!(minmax.apply(&rows[0].features), vec![0.0, 0.0, 0.0]);
        assert_eq!(minmax.apply(&rows[1].features), vec![1.0, 1.0, 0.0]);
        let maxabs = Scaler::fit(InputScaling::MaxAbs, &rows);
        assert_eq!(maxabs.apply(&rows[2].features), vec![0.75, 2.0 / 3.0, 1.0]);
        // Values outside the training range are clamped, never negative.
        assert_eq!(minmax.apply(&[1.0, 400.0, 9.0]), vec![0.0, 1.0, 1.0]);
        let raw = Scaler::fit(InputScaling::Raw, &rows);
        assert_eq!(raw.apply(&[7.0, 8.0, 9.0]), vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn parses_a_row_and_skips_short_ones() {
        let features = (1..=19).map(|i| i.to_string()).collect::<Vec<_>>().join(",");
        let good = format!("CCO,{features},1.5,0.7,2.5");
        let m = parse_row(&good).unwrap();
        assert_eq!(m.smiles, "CCO");
        assert_eq!(m.features.len(), N_FEATURES);
        assert_eq!(m.features[18], 19.0);
        assert_eq!(m.targets(), [1.5, 0.7, 2.5]);
        assert!(parse_row("CCO,1,2,3").is_none());
    }
}
