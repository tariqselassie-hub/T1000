// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Structure-of-Arrays (SoA) Quantum Cell State for 2D lattices.
//! Separates real and imaginary components across $|0\rangle$, $|1\rangle$, and $|2\rangle$
//! (Fractional Fermi Sea) basis sectors for SIMD streaming and zero-allocation execution.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Basis sector for local 3-level cell Hilbert space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CellBasis {
    /// Vacuum / Dead state (|0>).
    Vacuum = 0,
    /// Standard Living Fermi state (|1>).
    Living = 1,
    /// Super-Fermionic Fractional Fermi Sea branch (|2>).
    FractionalFermiSea = 2,
}

/// Structure-of-Arrays (SoA) quantum wavefunction for all lattice cells.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCellStateSoA {
    /// Total number of cells.
    pub num_cells: usize,
    /// Real amplitude buffer for vacuum state |0>.
    pub re0: Vec<f64>,
    /// Imaginary amplitude buffer for vacuum state |0>.
    pub im0: Vec<f64>,
    /// Real amplitude buffer for living state |1>.
    pub re1: Vec<f64>,
    /// Imaginary amplitude buffer for living state |1>.
    pub im1: Vec<f64>,
    /// Real amplitude buffer for super-fermionic FFS state |2>.
    pub re2: Vec<f64>,
    /// Imaginary amplitude buffer for super-fermionic FFS state |2>.
    pub im2: Vec<f64>,
}

impl QuantumCellStateSoA {
    /// Allocate an empty quantum state initialized entirely to vacuum (|0>).
    pub fn new_vacuum(num_cells: usize) -> Self {
        Self {
            num_cells,
            re0: vec![1.0; num_cells],
            im0: vec![0.0; num_cells],
            re1: vec![0.0; num_cells],
            im1: vec![0.0; num_cells],
            re2: vec![0.0; num_cells],
            im2: vec![0.0; num_cells],
        }
    }

    /// Set a cell to a pure classical living state (|1>).
    pub fn set_cell_living(&mut self, idx: usize) {
        self.re0[idx] = 0.0;
        self.im0[idx] = 0.0;
        self.re1[idx] = 1.0;
        self.im1[idx] = 0.0;
        self.re2[idx] = 0.0;
        self.im2[idx] = 0.0;
    }

    /// Set a cell to a pure Fractional Fermi Sea state (|2>).
    pub fn set_cell_ffs(&mut self, idx: usize) {
        self.re0[idx] = 0.0;
        self.im0[idx] = 0.0;
        self.re1[idx] = 0.0;
        self.im1[idx] = 0.0;
        self.re2[idx] = 1.0;
        self.im2[idx] = 0.0;
    }

    /// Set a cell to an arbitrary superposition (|0>, |1>, |2>) and normalize.
    pub fn set_cell_superposition(&mut self, idx: usize, c0: Complex64, c1: Complex64, c2: Complex64) {
        let norm_sq = c0.norm_sqr() + c1.norm_sqr() + c2.norm_sqr();
        let inv_norm = if norm_sq > 1e-15 { 1.0 / norm_sq.sqrt() } else { 1.0 };
        self.re0[idx] = c0.re * inv_norm;
        self.im0[idx] = c0.im * inv_norm;
        self.re1[idx] = c1.re * inv_norm;
        self.im1[idx] = c1.im * inv_norm;
        self.re2[idx] = c2.re * inv_norm;
        self.im2[idx] = c2.im * inv_norm;
    }

    /// Compute living state occupation probability P(|1>) at cell `idx`.
    #[inline(always)]
    pub fn prob_living(&self, idx: usize) -> f64 {
        self.re1[idx] * self.re1[idx] + self.im1[idx] * self.im1[idx]
    }

    /// Compute Fractional Fermi Sea occupation probability P(|2>) at cell `idx`.
    #[inline(always)]
    pub fn prob_ffs(&self, idx: usize) -> f64 {
        self.re2[idx] * self.re2[idx] + self.im2[idx] * self.im2[idx]
    }

    /// Compute vacuum state probability P(|0>) at cell `idx`.
    #[inline(always)]
    pub fn prob_vacuum(&self, idx: usize) -> f64 {
        self.re0[idx] * self.re0[idx] + self.im0[idx] * self.im0[idx]
    }

    /// Compute total expected particle occupation density <n_i> = P(|1>) + 2*P(|2>).
    #[inline(always)]
    pub fn cell_density(&self, idx: usize) -> f64 {
        self.prob_living(idx) + 2.0 * self.prob_ffs(idx)
    }

    /// Compute total lattice population N_tot = sum_i <n_i>.
    pub fn total_population(&self) -> f64 {
        let mut total = 0.0;
        for i in 0..self.num_cells {
            total += self.cell_density(i);
        }
        total
    }

    /// Project lattice amplitudes to a binary classical grid (alive = 1, dead = 0)
    /// based on occupation threshold P(|1>) + P(|2>) >= threshold.
    pub fn to_binary_projection(&self, threshold: f64) -> Vec<bool> {
        (0..self.num_cells)
            .map(|i| (self.prob_living(i) + self.prob_ffs(i)) >= threshold)
            .collect()
    }

    /// Check total normalization across all cells. Returns maximum deviation from 1.0.
    pub fn max_normalization_error(&self) -> f64 {
        let mut max_err = 0.0f64;
        for i in 0..self.num_cells {
            let p0 = self.prob_vacuum(i);
            let p1 = self.prob_living(i);
            let p2 = self.prob_ffs(i);
            let err = ((p0 + p1 + p2) - 1.0).abs();
            if err > max_err {
                max_err = err;
            }
        }
        max_err
    }

    /// Re-normalize all cells in-place.
    pub fn renormalize(&mut self) {
        for i in 0..self.num_cells {
            let p0 = self.prob_vacuum(i);
            let p1 = self.prob_living(i);
            let p2 = self.prob_ffs(i);
            let total = p0 + p1 + p2;
            if total > 1e-15 && (total - 1.0).abs() > 1e-14 {
                let inv = 1.0 / total.sqrt();
                self.re0[i] *= inv;
                self.im0[i] *= inv;
                self.re1[i] *= inv;
                self.im1[i] *= inv;
                self.re2[i] *= inv;
                self.im2[i] *= inv;
            }
        }
    }
}
