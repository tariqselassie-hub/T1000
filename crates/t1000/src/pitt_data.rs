// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Loader for the University of Pittsburgh Quantum Repository (`data/pitt/pitt_quantum.csv`,
//! 106k molecules, PM7 semi-empirical quantum observables) featurised with the universal 19-D
//! topological chemical descriptor map.

use crate::qm9_data::{descriptors, FeatureSet};
use crate::smiles::SmilesFeaturizer;
use std::fs;
use std::path::Path;

/// Properties available from the Pitt Quantum CSV table.
pub const PROPERTIES: [&str; 9] = [
    "homo",
    "lumo",
    "gap",
    "dipole",
    "polarizability",
    "heat_of_formation",
    "molwt",
    "surface_area",
    "volume",
];

/// One molecule with its 19-D topological features and the nine quantum/physical properties.
#[derive(Debug, Clone, PartialEq)]
pub struct PittMolecule {
    pub smiles: String,
    pub features: Vec<f64>,
    /// `homo` (eV), `lumo` (eV), `gap` (eV), `dipole` (Debye), `polarizability` (A^3),
    /// `heat_of_formation` (kJ/mol), `molwt` (g/mol), `surface_area` (A^2), `volume` (A^3).
    pub properties: [f64; 9],
}

impl PittMolecule {
    /// The properties at the given indices of [`PROPERTIES`].
    pub fn targets<const K: usize>(&self, which: [usize; K]) -> [f64; K] {
        which.map(|i| self.properties[i])
    }
}

fn parse_row(line: &str, set: FeatureSet) -> Option<PittMolecule> {
    let c: Vec<&str> = line.split(',').collect();
    if c.len() < 16 {
        return None;
    }
    let smiles = c[1].trim();
    // Exclude disconnected adduct mixtures
    if smiles.is_empty() || smiles.contains('.') {
        return None;
    }

    let num = |i: usize| c[i].trim().parse::<f64>().ok();

    let feat = SmilesFeaturizer::featurize(smiles).ok()?;
    let features = descriptors(&feat, set);

    // Columns:
    // 3: molwt, 5: homo, 6: lumo, 7: gap, 8: dipole, 12: polarizability,
    // 13: heat_of_formation, 14: surface_area, 15: volume
    let properties = [
        num(5)?,  // homo
        num(6)?,  // lumo
        num(7)?,  // gap
        num(8)?,  // dipole
        num(12)?, // polarizability
        num(13)?, // heat_of_formation
        num(3)?,  // molwt
        num(14)?, // surface_area
        num(15)?, // volume
    ];

    Some(PittMolecule {
        smiles: smiles.to_string(),
        features,
        properties,
    })
}

/// Loads up to `max_rows` molecules from the Pitt Quantum CSV table.
pub fn load(
    path: impl AsRef<Path>,
    set: FeatureSet,
    max_rows: usize,
) -> Result<(Vec<PittMolecule>, usize), String> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let (mut rows, mut skipped) = (Vec::new(), 0);

    for line in text.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        match parse_row(line, set) {
            Some(m) => rows.push(m),
            None => skipped += 1,
        }
        if rows.len() >= max_rows {
            break;
        }
    }

    if rows.is_empty() {
        return Err(format!("no molecules parsed from {}", path.display()));
    }
    Ok((rows, skipped))
}

/// Deterministic shuffle (seeded SplitMix64) then split into `(train, test)` with `test_size` test rows.
pub fn split(
    mut rows: Vec<PittMolecule>,
    test_size: usize,
    seed: u64,
) -> (Vec<PittMolecule>, Vec<PittMolecule>) {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    for i in (1..rows.len()).rev() {
        rows.swap(i, (next() % (i as u64 + 1)) as usize);
    }
    let split_at = rows.len().saturating_sub(test_size);
    let test = rows.split_off(split_at);
    (rows, test)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_pitt_row() {
        let sample_line = "AAABMNXUOFPYQK-AYUQAVNMSA-N,OC[C@H](C[C@H]1O[C@H]2[C@H](OCO[C@@H]2[C@H](C1(C)C)OC)NC(=O)[C@H]([C@]1(OC)O[C@H](C)[C@@H](C(=C)C1)C)O)OC,C25H43NO10,517.6096,C1,-9.3430,0.9400,10.2830,5.4210,2.8050,4.6390,0.0900,51.4016,-1808.8600,457.0200,622.1200,0.005775,0.002453,0.001938,187,tags,name";
        let mol = parse_row(sample_line, FeatureSet::ZincLike).expect("should parse");
        assert_eq!(mol.properties[0], -9.3430); // homo
        assert_eq!(mol.properties[1], 0.9400);  // lumo
        assert_eq!(mol.properties[2], 10.2830); // gap
        assert_eq!(mol.properties[3], 5.4210);  // dipole
        assert_eq!(mol.features.len(), 19);
    }

    #[test]
    fn rejects_multicomponent_adducts() {
        let adduct_line = "FAKE,C#C.O=O,C2H2O2,58.0,C1,-5.0,1.0,6.0,12.0,0,0,0,10,0,100,100,0,0,0,10,tags,name";
        assert!(parse_row(adduct_line, FeatureSet::ZincLike).is_none());
    }
}
