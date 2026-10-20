// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Molecular Coulomb Matrix Representation & Eigenspectrum Subsystem.
//!
//! Provides dense symmetric Coulomb matrix reconstruction, atomic number recovery,
//! invariant eigenspectrum decomposition, and high-throughput RoboBohr CSV parsing.

use crate::simd::eigensystem::solve_coulomb_eigensystem_50;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read};

/// Fixed dimension for standard Coulomb matrix representation.
pub const COULOMB_DIM_50: usize = 50;

/// Number of unique upper-triangular elements for a 50x50 matrix: 50 * 51 / 2 = 1275.
pub const COULOMB_UPPER_TRI_COUNT_50: usize = 1275;

/// Real symmetric 50x50 Coulomb matrix representation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoulombMatrix50 {
    /// Full symmetric row-major matrix: 50 * 50 = 2500 entries.
    pub matrix: Vec<f64>,
}

impl Default for CoulombMatrix50 {
    fn default() -> Self {
        Self {
            matrix: vec![0.0; 2500],
        }
    }
}

impl CoulombMatrix50 {
    /// Creates a new empty Coulomb matrix.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reconstructs the symmetric 50x50 Coulomb matrix from 1,275 upper-triangular entries.
    ///
    /// The upper triangle is expected in row-major order:
    /// - Row 0: elements (0, 0..50)
    /// - Row 1: elements (1, 1..50)
    /// - ...
    /// - Row 49: element (49, 49)
    pub fn from_upper_triangle(upper_tri: &[f64; COULOMB_UPPER_TRI_COUNT_50]) -> Self {
        let mut mat = vec![0.0; 2500];
        let mut idx = 0;

        for i in 0..COULOMB_DIM_50 {
            for j in i..COULOMB_DIM_50 {
                let val = upper_tri[idx];
                mat[i * COULOMB_DIM_50 + j] = val;
                mat[j * COULOMB_DIM_50 + i] = val;
                idx += 1;
            }
        }

        Self { matrix: mat }
    }

    /// Retrieves element (i, j).
    #[inline(always)]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        if i < COULOMB_DIM_50 && j < COULOMB_DIM_50 {
            self.matrix[i * COULOMB_DIM_50 + j]
        } else {
            0.0
        }
    }

    /// Returns the matrix trace: sum of diagonal entries (nuclear self-energies).
    pub fn trace(&self) -> f64 {
        let mut tr = 0.0;
        for i in 0..COULOMB_DIM_50 {
            tr += self.matrix[i * COULOMB_DIM_50 + i];
        }
        tr
    }

    /// Returns the Frobenius norm squared: sum of all squared entries.
    pub fn frobenius_norm_sq(&self) -> f64 {
        self.matrix.iter().map(|&x| x * x).sum()
    }

    /// Returns the Euclidean norm of row `i`: ||C_i||_2.
    pub fn row_norm(&self, row: usize) -> f64 {
        if row >= COULOMB_DIM_50 {
            return 0.0;
        }
        let start = row * COULOMB_DIM_50;
        let mut sum_sq = 0.0;
        for j in 0..COULOMB_DIM_50 {
            let v = self.matrix[start + j];
            sum_sq += v * v;
        }
        sum_sq.sqrt()
    }

    /// Recovers inferred atomic numbers Z_i from diagonal entries C_ii = 0.5 * Z_i^2.4.
    pub fn inferred_atomic_numbers(&self) -> [f64; COULOMB_DIM_50] {
        let mut z = [0.0; COULOMB_DIM_50];
        let exponent = 1.0 / 2.4;
        for i in 0..COULOMB_DIM_50 {
            let c_ii = self.matrix[i * COULOMB_DIM_50 + i];
            if c_ii > 1e-4 {
                z[i] = (2.0 * c_ii).powf(exponent);
            }
        }
        z
    }

    /// Evaluates the complete 50-dimensional invariant eigenspectrum.
    ///
    /// Eigenvalues are returned in descending order: lambda_0 >= lambda_1 >= ... >= lambda_49.
    /// Invariant under all 3D rotations SO(3), translations, and atom permutations.
    pub fn eigenspectrum(&self) -> [f64; COULOMB_DIM_50] {
        let mut fixed = [0.0; 2500];
        let copy_len = self.matrix.len().min(2500);
        fixed[..copy_len].copy_from_slice(&self.matrix[..copy_len]);
        solve_coulomb_eigensystem_50(&fixed)
    }
}

/// A parsed record from the RoboBohr molecular ground state energy dataset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoboBohrRecord {
    /// Row index in the dataset.
    pub row_index: usize,
    /// PubChem Compound ID.
    pub pubchem_id: u64,
    /// Ground-state atomization energy in Hartrees (E_at).
    pub eat_hartree: f64,
    /// Reconstructed symmetric 50x50 Coulomb matrix.
    pub coulomb: CoulombMatrix50,
}

impl RoboBohrRecord {
    /// Parses a single CSV line from `roboBohr.csv`.
    pub fn parse_csv_line(line: &str) -> Result<Self, String> {
        let parts: Vec<&str> = line.split(',').collect();
        // Expected columns: Unnamed: 0, 0..1274, pubchem_id, Eat (total 1278 columns)
        if parts.len() < 1278 {
            return Err(format!(
                "Invalid roboBohr CSV line: expected at least 1278 columns, found {}",
                parts.len()
            ));
        }

        let row_index = parts[0]
            .trim()
            .parse::<usize>()
            .map_err(|e| format!("Invalid row index '{}': {}", parts[0], e))?;

        let mut upper_tri = [0.0; COULOMB_UPPER_TRI_COUNT_50];
        for i in 0..COULOMB_UPPER_TRI_COUNT_50 {
            upper_tri[i] = parts[1 + i]
                .trim()
                .parse::<f64>()
                .map_err(|e| format!("Invalid feature {} '{}': {}", i, parts[1 + i], e))?;
        }

        let pubchem_id = parts[1276]
            .trim()
            .parse::<f64>()
            .map(|x| x as u64)
            .map_err(|e| format!("Invalid pubchem_id '{}': {}", parts[1276], e))?;

        let eat_hartree = parts[1277]
            .trim()
            .parse::<f64>()
            .map_err(|e| format!("Invalid Eat '{}': {}", parts[1277], e))?;

        let coulomb = CoulombMatrix50::from_upper_triangle(&upper_tri);

        Ok(Self {
            row_index,
            pubchem_id,
            eat_hartree,
            coulomb,
        })
    }

    /// Streams up to `max_records` from any buffered CSV reader.
    pub fn stream_from_reader<R: Read>(
        reader: R,
        max_records: Option<usize>,
    ) -> Result<Vec<Self>, String> {
        let buf_reader = BufReader::new(reader);
        let mut records = Vec::new();

        for (line_num, line_res) in buf_reader.lines().enumerate() {
            let line = line_res.map_err(|e| format!("Read error at line {line_num}: {e}"))?;
            if line_num == 0 && (line.contains("pubchem_id") || line.contains("Eat")) {
                continue; // Skip header
            }
            if line.trim().is_empty() {
                continue;
            }

            let record = Self::parse_csv_line(&line)?;
            records.push(record);

            if let Some(limit) = max_records {
                if records.len() >= limit {
                    break;
                }
            }
        }

        Ok(records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coulomb_matrix_reconstruction_and_trace() {
        let mut upper_tri = [0.0; COULOMB_UPPER_TRI_COUNT_50];
        // Set diagonal entries: 0.5 * 8^2.4 = 73.5167, 0.5 * 6^2.4 = 36.8581
        upper_tri[0] = 73.516695; // C_00
        upper_tri[50] = 36.858105; // C_11
        upper_tri[1] = 17.817765; // C_01 off-diagonal

        let mat = CoulombMatrix50::from_upper_triangle(&upper_tri);
        assert!((mat.get(0, 0) - 73.516695).abs() < 1e-6);
        assert!((mat.get(1, 1) - 36.858105).abs() < 1e-6);
        assert!((mat.get(0, 1) - 17.817765).abs() < 1e-6);
        assert!((mat.get(1, 0) - 17.817765).abs() < 1e-6); // Symmetric

        let z = mat.inferred_atomic_numbers();
        assert!((z[0] - 8.0).abs() < 1e-3); // Oxygen
        assert!((z[1] - 6.0).abs() < 1e-3); // Carbon

        let eigs = mat.eigenspectrum();
        assert_eq!(eigs.len(), 50);
        // Trace invariant: sum(eigs) == trace(C)
        let tr = mat.trace();
        let eig_sum: f64 = eigs.iter().sum();
        assert!((tr - eig_sum).abs() < 1e-6);
    }
}
