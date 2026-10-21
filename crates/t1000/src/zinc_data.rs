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

#[cfg(test)]
mod tests {
    use super::*;

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
