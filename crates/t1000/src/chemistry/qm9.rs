// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! QM9 Quantum Chemical Dataset Ingestion & Observable Subsystem.
//!
//! Provides zero-allocation extended XYZ parsing, streaming ZIP archive extraction,
//! and direct integration with T-1000's OpenSMILES and 19-dimensional quantum Hilbert spaces.

use crate::smiles::{SmilesFeaturizer, SmilesFeatures};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Seek};
use std::path::Path;
use zip::ZipArchive;

/// Conversion factor from Hartrees to electron-volts (CODATA standard).
pub const HARTREE_TO_EV: f64 = 27.211386245988;

/// Single atom entry from an extended XYZ geometry block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qm9Atom {
    /// Atomic element symbol (e.g. "C", "H", "O", "N", "F").
    pub symbol: String,
    /// 3D Cartesian coordinates in Angstroms: [x, y, z].
    pub coords: [f64; 3],
    /// Mulliken partial atomic charge (e).
    pub mulliken_charge: f64,
}

/// Comprehensive quantum chemical record for a single QM9 molecule.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qm9Record {
    /// Molecule index within the GDB-9 space (1..133885).
    pub index: usize,
    /// Origin tag (typically "gdb").
    pub tag: String,
    /// Total atom count (heavy atoms + hydrogens).
    pub n_atoms: usize,
    /// Rotational constants A, B, C in GHz.
    pub rotational_constants: [f64; 3],
    /// Dipole moment in Debye (mu).
    pub dipole_moment: f64,
    /// Isotropic polarizability in Bohr^3 (alpha).
    pub polarizability: f64,
    /// Energy of Highest Occupied Molecular Orbital in Hartrees (epsilon_HOMO).
    pub homo: f64,
    /// Energy of Lowest Unoccupied Molecular Orbital in Hartrees (epsilon_LUMO).
    pub lumo: f64,
    /// Fundamental HOMO-LUMO gap in Hartrees: lumo - homo.
    pub gap: f64,
    /// Electronic spatial extent in Bohr^2 (<R^2>).
    pub spatial_extent: f64,
    /// Zero-point vibrational energy in Hartrees (ZPVE).
    pub zpve: f64,
    /// Internal electronic ground-state energy at 0 K in Hartrees (U_0).
    pub u0: f64,
    /// Internal energy at 298.15 K in Hartrees (U).
    pub u: f64,
    /// Enthalpy at 298.15 K in Hartrees (H).
    pub h: f64,
    /// Gibbs free energy at 298.15 K in Hartrees (G).
    pub g: f64,
    /// Constant-volume heat capacity in cal/(mol*K) (C_v).
    pub cv: f64,
    /// Atom-by-atom 3D coordinates and Mulliken partial charges.
    pub atoms: Vec<Qm9Atom>,
    /// Harmonic vibrational frequencies (cm^-1).
    pub frequencies: Vec<f64>,
    /// Canonical GDB-17 OpenSMILES string.
    pub smiles_gdb: String,
    /// B3LYP geometry-relaxed OpenSMILES string.
    pub smiles_b3lyp: String,
    /// International Chemical Identifier (InChI).
    pub inchi: String,
}

impl Qm9Record {
    /// Parses a single extended XYZ format string from the QM9 dataset.
    pub fn parse_xyz_str(content: &str) -> Result<Self, String> {
        let lines: Vec<&str> = content.lines().collect();
        if lines.len() < 5 {
            return Err("Extended XYZ content too short".to_string());
        }

        // Line 0: Atom count
        let n_atoms = lines[0]
            .trim()
            .parse::<usize>()
            .map_err(|e| format!("Invalid atom count line '{}': {}", lines[0], e))?;

        if lines.len() < 3 + n_atoms {
            return Err(format!(
                "Incomplete XYZ block: declared {} atoms, found {} lines",
                n_atoms,
                lines.len()
            ));
        }

        // Line 1: Scalar DFT properties
        // Columns: tag, index, A, B, C, mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv
        let p: Vec<&str> = lines[1].split_whitespace().collect();
        if p.len() < 17 {
            return Err(format!(
                "Invalid properties line: expected 17 columns, found {}",
                p.len()
            ));
        }

        let tag = p[0].to_string();
        let index = p[1].parse::<usize>().map_err(|e| format!("index: {e}"))?;
        let a = p[2].parse::<f64>().map_err(|e| format!("A: {e}"))?;
        let b = p[3].parse::<f64>().map_err(|e| format!("B: {e}"))?;
        let c = p[4].parse::<f64>().map_err(|e| format!("C: {e}"))?;
        let dipole_moment = p[5].parse::<f64>().map_err(|e| format!("mu: {e}"))?;
        let polarizability = p[6].parse::<f64>().map_err(|e| format!("alpha: {e}"))?;
        let homo = p[7].parse::<f64>().map_err(|e| format!("homo: {e}"))?;
        let lumo = p[8].parse::<f64>().map_err(|e| format!("lumo: {e}"))?;
        let gap = p[9].parse::<f64>().map_err(|e| format!("gap: {e}"))?;
        let spatial_extent = p[10].parse::<f64>().map_err(|e| format!("r2: {e}"))?;
        let zpve = p[11].parse::<f64>().map_err(|e| format!("zpve: {e}"))?;
        let u0 = p[12].parse::<f64>().map_err(|e| format!("u0: {e}"))?;
        let u = p[13].parse::<f64>().map_err(|e| format!("u: {e}"))?;
        let h = p[14].parse::<f64>().map_err(|e| format!("h: {e}"))?;
        let g = p[15].parse::<f64>().map_err(|e| format!("g: {e}"))?;
        let cv = p[16].parse::<f64>().map_err(|e| format!("cv: {e}"))?;

        // Lines 2 to 2 + n_atoms: Atomic coordinates & Mulliken charges
        let mut atoms = Vec::with_capacity(n_atoms);
        for i in 0..n_atoms {
            let row = lines[2 + i].split_whitespace().collect::<Vec<&str>>();
            if row.len() < 5 {
                return Err(format!("Malformed atom line: {}", lines[2 + i]));
            }
            let symbol = row[0].to_string();
            let x = row[1].parse::<f64>().map_err(|e| format!("atom x: {e}"))?;
            let y = row[2].parse::<f64>().map_err(|e| format!("atom y: {e}"))?;
            let z = row[3].parse::<f64>().map_err(|e| format!("atom z: {e}"))?;
            let q = row[4].parse::<f64>().map_err(|e| format!("atom q: {e}"))?;
            atoms.push(Qm9Atom {
                symbol,
                coords: [x, y, z],
                mulliken_charge: q,
            });
        }

        // Line 2 + n_atoms: Vibrational frequencies
        let freq_line_idx = 2 + n_atoms;
        let mut frequencies = Vec::new();
        if freq_line_idx < lines.len() {
            for tok in lines[freq_line_idx].split_whitespace() {
                if let Ok(val) = tok.parse::<f64>() {
                    frequencies.push(val);
                }
            }
        }

        // Line 3 + n_atoms: SMILES strings (GDB-17 and B3LYP)
        let smiles_line_idx = 3 + n_atoms;
        let mut smiles_gdb = String::new();
        let mut smiles_b3lyp = String::new();
        if smiles_line_idx < lines.len() {
            let s_toks: Vec<&str> = lines[smiles_line_idx].split_whitespace().collect();
            if !s_toks.is_empty() {
                smiles_gdb = s_toks[0].to_string();
            }
            if s_toks.len() > 1 {
                smiles_b3lyp = s_toks[1].to_string();
            } else {
                smiles_b3lyp = smiles_gdb.clone();
            }
        }

        // Line 4 + n_atoms: InChI strings
        let inchi_line_idx = 4 + n_atoms;
        let mut inchi = String::new();
        if inchi_line_idx < lines.len() {
            let i_toks: Vec<&str> = lines[inchi_line_idx].split_whitespace().collect();
            if !i_toks.is_empty() {
                inchi = i_toks[0].to_string();
            }
        }

        Ok(Self {
            index,
            tag,
            n_atoms,
            rotational_constants: [a, b, c],
            dipole_moment,
            polarizability,
            homo,
            lumo,
            gap,
            spatial_extent,
            zpve,
            u0,
            u,
            h,
            g,
            cv,
            atoms,
            frequencies,
            smiles_gdb,
            smiles_b3lyp,
            inchi,
        })
    }

    /// Returns the fundamental HOMO-LUMO gap in electron-volts (eV).
    #[inline(always)]
    pub fn gap_ev(&self) -> f64 {
        self.gap * HARTREE_TO_EV
    }

    /// Returns the electronic ground state energy at 0 K in electron-volts (eV).
    #[inline(always)]
    pub fn u0_ev(&self) -> f64 {
        self.u0 * HARTREE_TO_EV
    }

    /// Extracts structural physical-chemistry features from the B3LYP SMILES string.
    pub fn smiles_features(&self) -> Result<SmilesFeatures, String> {
        SmilesFeaturizer::featurize(&self.smiles_b3lyp)
            .map_err(|e| format!("Failed to featurize SMILES '{}': {:?}", self.smiles_b3lyp, e))
    }

    /// Extracts the canonical 19-dimensional normalized feature vector for quantum Hilbert space embedding.
    pub fn quantum_feature_vector(&self) -> Result<[f64; 19], String> {
        let feat = self.smiles_features()?;
        Ok(feat.to_normalized_vector())
    }

    /// Streams QM9 records directly from a `.zip` archive on disk.
    pub fn stream_from_zip(
        zip_path: &Path,
        max_records: Option<usize>,
    ) -> Result<Vec<Self>, String> {
        let file = File::open(zip_path)
            .map_err(|e| format!("Failed to open zip archive {:?}: {}", zip_path, e))?;
        Self::stream_from_reader(file, max_records)
    }

    /// Streams QM9 records from any seekable reader containing a ZIP archive.
    pub fn stream_from_reader<R: Read + Seek>(
        reader: R,
        max_records: Option<usize>,
    ) -> Result<Vec<Self>, String> {
        let mut archive = ZipArchive::new(reader)
            .map_err(|e| format!("Failed to parse zip archive: {e}"))?;

        let mut records = Vec::new();
        let total_files = archive.len();

        for i in 0..total_files {
            let mut file = match archive.by_index(i) {
                Ok(f) => f,
                Err(_) => continue,
            };

            // Only process canonical files from the dsgdb9nsd.xyz/ directory
            let name = file.name().to_string();
            if !name.starts_with("dsgdb9nsd.xyz/dsgdb9nsd_") || !name.ends_with(".xyz") {
                continue;
            }

            let mut content = String::new();
            if file.read_to_string(&mut content).is_ok() {
                if let Ok(rec) = Self::parse_xyz_str(&content) {
                    records.push(rec);
                    if let Some(limit) = max_records {
                        if records.len() >= limit {
                            break;
                        }
                    }
                }
            }
        }

        Ok(records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const METHANE_XYZ: &str = "\
5
gdb 1\t157.7118\t157.70997\t157.70699\t0.\t13.21\t-0.3877\t0.1171\t0.5048\t35.3641\t0.044749\t-40.47893\t-40.476062\t-40.475117\t-40.498597\t6.469\t
C\t-0.0126981359\t 1.0858041578\t 0.0080009958\t-0.535689
H\t 0.002150416\t-0.0060313176\t 0.0019761204\t 0.133921
H\t 1.0117308433\t 1.4637511618\t 0.0002765748\t 0.133922
H\t-0.540815069\t 1.4475266138\t-0.8766437152\t 0.133923
H\t-0.5238136345\t 1.4379326443\t 0.9063972942\t 0.133923
1341.307\t1341.3284\t1341.365\t1562.6731\t1562.7453\t3038.3205\t3151.6034\t3151.6788\t3151.7078
C\tC\t
InChI=1S/CH4/h1H4\tInChI=1S/CH4/h1H4
";

    #[test]
    fn test_qm9_parse_methane() {
        let rec = Qm9Record::parse_xyz_str(METHANE_XYZ).unwrap();
        assert_eq!(rec.index, 1);
        assert_eq!(rec.n_atoms, 5);
        assert!((rec.u0 - (-40.47893)).abs() < 1e-5);
        assert!((rec.gap - 0.5048).abs() < 1e-4);
        assert!((rec.gap_ev() - 13.7363).abs() < 1e-2);
        assert_eq!(rec.atoms.len(), 5);
        assert_eq!(rec.atoms[0].symbol, "C");
        assert_eq!(rec.atoms[1].symbol, "H");
        assert_eq!(rec.smiles_b3lyp, "C");

        // Verify SMILES 19-dim quantum encoding
        let q_vec = rec.quantum_feature_vector().unwrap();
        assert_eq!(q_vec.len(), 19);
    }
}
