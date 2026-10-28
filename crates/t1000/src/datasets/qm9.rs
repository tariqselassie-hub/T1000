// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Loader for the MoleculeNet `qm9.csv` (133,885 molecules, DFT properties) with the SMILES
//! featurised like the ZINC pipeline.
//!
//! The 19 "ZINC-like" columns are the same descriptors as the ZINC CSVs: heavy atoms, estimated
//! molecular weight, element counts, aromatic atoms and ratio, heteroatoms and ratio, double and
//! triple bonds, branches, chiral atoms, rings. QM9 contains only C, H, N, O and F, so sulfur,
//! chlorine and bromine are always zero; the "QM9" variant replaces those three and the halogen
//! total with the hydrogen count, formal charge, aromatic carbons and total atom count.

use crate::smiles::{SmilesFeatures, SmilesFeaturizer};
use std::fs;
use std::path::Path;

/// Properties kept from the CSV, with units after loading.
pub const PROPERTIES: [&str; 9] = [
    "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "cv", "u0_atom",
];

const HARTREE_TO_EV: f64 = 27.211_386_245_988;

/// Which 19 descriptors describe a molecule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureSet {
    /// The ZINC pipeline's columns.
    ZincLike,
    /// The ZINC columns with the always-zero ones replaced by hydrogen-aware descriptors.
    Qm9,
}

/// One molecule with its descriptors and the nine properties of [`PROPERTIES`].
#[derive(Debug, Clone, PartialEq)]
pub struct Qm9Molecule {
    pub smiles: String,
    pub features: Vec<f64>,
    /// `mu` (Debye), `alpha` (Bohr^3), `homo`, `lumo`, `gap` (eV), `r2` (Bohr^2), `zpve` (Hartree),
    /// `cv` (cal/mol K), `u0_atom` (kcal/mol, atomization energy).
    pub properties: [f64; 9],
}

impl Qm9Molecule {
    /// The properties at the given indices of [`PROPERTIES`].
    pub fn targets<const K: usize>(&self, which: [usize; K]) -> [f64; K] {
        which.map(|i| self.properties[i])
    }
}

/// Descriptor vector of featurised SMILES.
pub fn descriptors(f: &SmilesFeatures, set: FeatureSet) -> Vec<f64> {
    let ha = f.heavy_atoms as f64;
    let n_arom = (f.n_arom_c + f.n_arom_n + f.n_arom_o) as f64;
    let hetero = f.heavy_atoms.saturating_sub(f.n_c) as f64;
    let mut v = vec![
        ha,
        f.estimated_mw,
        f.n_c as f64,
        f.n_n as f64,
        f.n_o as f64,
        f.n_s as f64,
        f.n_f as f64,
        f.n_cl as f64,
        f.n_br as f64,
        (f.n_f + f.n_cl + f.n_br + f.n_i) as f64,
        n_arom,
        if ha > 0.0 { n_arom / ha } else { 0.0 },
        hetero,
        if ha > 0.0 { hetero / ha } else { 0.0 },
        f.n_double as f64,
        f.n_triple as f64,
        f.n_branches as f64,
        f.n_chiral as f64,
        f.n_rings as f64,
    ];
    if set == FeatureSet::Qm9 {
        // Replace n_s, n_cl, n_br and n_hal (columns 5, 7, 8, 9).
        v[5] = f.n_hydrogens as f64;
        v[7] = f64::from(f.formal_charge);
        v[8] = f.n_arom_c as f64;
        v[9] = (f.heavy_atoms + f.n_hydrogens) as f64;
    }
    v
}

fn parse_row(line: &str, set: FeatureSet) -> Option<Qm9Molecule> {
    let c: Vec<&str> = line.split(',').collect();
    if c.len() < 18 {
        return None;
    }
    let num = |i: usize| c[i].trim().parse::<f64>().ok();
    let smiles = c[1].trim().to_string();
    let features = descriptors(&SmilesFeaturizer::featurize(&smiles).ok()?, set);
    // Columns: 5 mu, 6 alpha, 7 homo, 8 lumo, 9 gap, 10 r2, 11 zpve, 16 cv, 17 u0_atom.
    let properties = [
        num(5)?,
        num(6)?,
        num(7)? * HARTREE_TO_EV,
        num(8)? * HARTREE_TO_EV,
        num(9)? * HARTREE_TO_EV,
        num(10)?,
        num(11)?,
        num(16)?,
        num(17)?,
    ];
    Some(Qm9Molecule {
        smiles,
        features,
        properties,
    })
}

/// Loads up to `max_rows` molecules; returns them with the number of rows that could not be
/// parsed or featurised.
pub fn load(
    path: impl AsRef<Path>,
    set: FeatureSet,
    max_rows: usize,
) -> Result<(Vec<Qm9Molecule>, usize), String> {
    let path = path.as_ref();
    let text =
        fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
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
    mut rows: Vec<Qm9Molecule>,
    test_size: usize,
    seed: u64,
) -> (Vec<Qm9Molecule>, Vec<Qm9Molecule>) {
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
    let test = rows.split_off(rows.len() - test_size.min(rows.len() / 2));
    (rows, test)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ETHANOL: &str =
        "gdb_9,CCO,1,1,1,1.4,20.0,-0.26,0.08,0.34,0,0.08,0,0,0,0,10.0,-800.0,0,0,0";

    #[test]
    fn parses_a_row_into_descriptors_and_unit_converted_properties() {
        let m = parse_row(ETHANOL, FeatureSet::ZincLike).expect("row parses");
        assert_eq!(m.features.len(), 19);
        assert_eq!(m.features[0], 3.0); // heavy atoms
        assert_eq!(m.features[2], 2.0); // carbons
        assert_eq!(m.features[4], 1.0); // oxygens
        assert!((m.properties[4] - 0.34 * HARTREE_TO_EV).abs() < 1e-9);
        assert_eq!(m.targets([1, 8]), [20.0, -800.0]);
    }

    #[test]
    fn the_qm9_variant_adds_hydrogens_and_keeps_nineteen_columns() {
        let zinc = parse_row(ETHANOL, FeatureSet::ZincLike).unwrap();
        let qm9 = parse_row(ETHANOL, FeatureSet::Qm9).unwrap();
        assert_eq!(qm9.features.len(), 19);
        assert_eq!(zinc.features[5], 0.0); // no sulfur
        assert_eq!(qm9.features[5], 6.0); // six hydrogens in ethanol
        assert_eq!(qm9.features[9], 9.0); // nine atoms in total
        assert_eq!(&zinc.features[..5], &qm9.features[..5]);
    }

    #[test]
    fn short_or_unparsable_rows_are_rejected() {
        assert!(parse_row("gdb_1,C,1", FeatureSet::ZincLike).is_none());
        assert!(parse_row(&ETHANOL.replace("CCO", "C(("), FeatureSet::ZincLike).is_none());
    }

    #[test]
    fn the_split_is_deterministic_disjoint_and_complete() {
        let rows: Vec<Qm9Molecule> = (0..100)
            .map(|i| Qm9Molecule {
                smiles: i.to_string(),
                features: vec![i as f64],
                properties: [0.0; 9],
            })
            .collect();
        let (a_train, a_test) = split(rows.clone(), 20, 5);
        let (b_train, b_test) = split(rows.clone(), 20, 5);
        assert_eq!((&a_train, &a_test), (&b_train, &b_test));
        assert_eq!((a_train.len(), a_test.len()), (80, 20));
        let mut all: Vec<&str> = a_train
            .iter()
            .chain(&a_test)
            .map(|m| m.smiles.as_str())
            .collect();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), 100);
        assert_ne!(split(rows, 20, 6).1, a_test);
    }
}
