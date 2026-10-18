// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Quantum and Classical Transition Rules for Conway's Game of Life.
//! Implements classical B3/S23 reference execution, neighborhood density accumulation,
//! and unitary multi-cell block operators for reversible Margolus partitioning.

use crate::lattice::Lattice2D;
use crate::state::QuantumCellStateSoA;
use serde::{Deserialize, Serialize};

/// Classical Conway B3/S23 execution engine for benchmarking and fidelity comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassicalConwayEngine {
    /// 2D grid dimensions.
    pub width: usize,
    pub height: usize,
    /// Current binary cell status (true = alive, false = dead).
    pub grid: Vec<bool>,
    /// Scratch buffer for next generation.
    scratch: Vec<bool>,
}

impl ClassicalConwayEngine {
    /// Create a new classical Conway engine on an Lx * Ly grid initialized to dead.
    pub fn new(width: usize, height: usize) -> Self {
        let total = width * height;
        Self {
            width,
            height,
            grid: vec![false; total],
            scratch: vec![false; total],
        }
    }

    /// Set an individual cell alive or dead.
    pub fn set_cell(&mut self, x: usize, y: usize, alive: bool) {
        if x < self.width && y < self.height {
            let idx = y * self.width + x;
            self.grid[idx] = alive;
        }
    }

    /// Advance one generation using standard Conway B3/S23 rules.
    pub fn step(&mut self, lattice: &Lattice2D) {
        assert_eq!(lattice.width, self.width);
        assert_eq!(lattice.height, self.height);

        for i in 0..lattice.total_cells {
            let nbrs = lattice.moore_neighbors(i);
            let mut live_count = 0usize;
            for &nbr_idx in nbrs {
                if self.grid[nbr_idx] {
                    live_count += 1;
                }
            }

            let was_alive = self.grid[i];
            let next_alive = match (was_alive, live_count) {
                (true, 2) | (true, 3) => true, // Survival
                (false, 3) => true,            // Birth
                _ => false,                    // Underpopulation / Overpopulation
            };
            self.scratch[i] = next_alive;
        }

        self.grid.copy_from_slice(&self.scratch);
    }

    /// Count total live cells.
    pub fn live_population(&self) -> usize {
        self.grid.iter().filter(|&&alive| alive).count()
    }
}

/// Compute Moore neighborhood occupation vector for all cells in `state`.
/// Returns a vector of length `lattice.total_cells` where each entry is
/// sum_{j in Moore(i)} ( P(|1>_j) + 2 * P(|2>_j) ).
pub fn compute_neighborhood_occupations(
    lattice: &Lattice2D,
    state: &QuantumCellStateSoA,
    out_occupations: &mut [f64],
) {
    assert_eq!(lattice.total_cells, state.num_cells);
    assert_eq!(out_occupations.len(), state.num_cells);

    // Pre-calculate individual cell densities
    let mut densities = vec![0.0f64; state.num_cells];
    for (i, d) in densities.iter_mut().enumerate().take(state.num_cells) {
        *d = state.cell_density(i);
    }

    for (i, occ) in out_occupations
        .iter_mut()
        .enumerate()
        .take(lattice.total_cells)
    {
        let nbrs = lattice.moore_neighbors(i);
        let mut sum = 0.0f64;
        for &nbr_idx in nbrs {
            sum += densities[nbr_idx];
        }
        *occ = sum;
    }
}

/// Compute Hamming distance between a classical binary grid and a quantum projected binary grid.
pub fn compute_hamming_distance(classical: &[bool], quantum: &[bool]) -> usize {
    assert_eq!(classical.len(), quantum.len());
    classical
        .iter()
        .zip(quantum.iter())
        .filter(|(&c, &q)| c != q)
        .count()
}
