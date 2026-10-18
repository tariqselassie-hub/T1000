// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! 2D Lattice geometry, boundary conditions, and neighborhood indexing
//! for Quantum Cellular Automata and Fractional Fermi Sea evolution.

use serde::{Deserialize, Serialize};

/// Boundary condition governing edge wrapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryCondition {
    /// Toroidal periodic boundary condition (wraps in both dimensions).
    Toroidal,
    /// Open boundary condition (clamped with vacuum boundary cells).
    Open,
}

/// 2D discrete lattice topology with precomputed neighborhood lookups.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lattice2D {
    /// Width of the 2D grid in cells.
    pub width: usize,
    /// Height of the 2D grid in cells.
    pub height: usize,
    /// Total number of cells (width * height).
    pub total_cells: usize,
    /// Boundary condition applied.
    pub boundary: BoundaryCondition,
    /// Precomputed 8-neighbor Moore table for each flat index.
    neighbors: Vec<[usize; 8]>,
}

impl Lattice2D {
    /// Create a new 2D lattice and precompute the Moore neighborhood tables.
    pub fn new(width: usize, height: usize, boundary: BoundaryCondition) -> Self {
        assert!(width >= 2 && height >= 2, "Lattice dimensions must be at least 2x2");
        let total_cells = width * height;
        let mut neighbors = Vec::with_capacity(total_cells);

        for y in 0..height {
            for x in 0..width {
                let cell_neighbors = Self::compute_moore_neighbors(x, y, width, height, boundary);
                neighbors.push(cell_neighbors);
            }
        }

        Self {
            width,
            height,
            total_cells,
            boundary,
            neighbors,
        }
    }

    /// Convert 2D coordinates (x, y) into a flat 1D index.
    #[inline(always)]
    pub fn coord_to_idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    /// Convert a flat 1D index into 2D coordinates (x, y).
    #[inline(always)]
    pub fn idx_to_coord(&self, idx: usize) -> (usize, usize) {
        (idx % self.width, idx / self.width)
    }

    /// Retrieve the 8 precomputed Moore neighbor indices for a cell.
    #[inline(always)]
    pub fn moore_neighbors(&self, idx: usize) -> &[usize; 8] {
        &self.neighbors[idx]
    }

    /// Compute the Moore 8-neighbor coordinates for cell (x, y).
    fn compute_moore_neighbors(
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        b: BoundaryCondition,
    ) -> [usize; 8] {
        let offsets = [
            (-1, -1), (0, -1), (1, -1),
            (-1,  0),          (1,  0),
            (-1,  1), (0,  1), (1,  1),
        ];
        let mut nbrs = [0usize; 8];

        for (i, (dx, dy)) in offsets.iter().enumerate() {
            let nx = match b {
                BoundaryCondition::Toroidal => ((x as isize + dx).rem_euclid(w as isize)) as usize,
                BoundaryCondition::Open => (x as isize + dx).clamp(0, (w - 1) as isize) as usize,
            };
            let ny = match b {
                BoundaryCondition::Toroidal => ((y as isize + dy).rem_euclid(h as isize)) as usize,
                BoundaryCondition::Open => (y as isize + dy).clamp(0, (h - 1) as isize) as usize,
            };
            nbrs[i] = ny * w + nx;
        }

        nbrs
    }

    /// Retrieve Margolus $2 \times 2$ plaquettes for reversible block partitioning.
    /// `offset = 0` yields even-step plaquettes, `offset = 1` yields odd-step shifted plaquettes.
    pub fn margolus_plaquettes(&self, offset: usize) -> Vec<[usize; 4]> {
        let mut plaquettes = Vec::new();
        let step = 2;

        let start_x = offset % 2;
        let start_y = offset % 2;

        let mut y = start_y;
        while y + 1 < self.height {
            let mut x = start_x;
            while x + 1 < self.width {
                let p00 = self.coord_to_idx(x, y);
                let p10 = self.coord_to_idx(x + 1, y);
                let p01 = self.coord_to_idx(x, y + 1);
                let p11 = self.coord_to_idx(x + 1, y + 1);
                plaquettes.push([p00, p10, p01, p11]);
                x += step;
            }
            y += step;
        }

        plaquettes
    }

    /// Retrieve checkerboard partition: Sublattice A ($x+y \equiv 0$) and Sublattice B ($x+y \equiv 1$).
    pub fn checkerboard_sublattices(&self) -> (Vec<usize>, Vec<usize>) {
        let mut sub_a = Vec::with_capacity(self.total_cells / 2);
        let mut sub_b = Vec::with_capacity(self.total_cells / 2);

        for y in 0..self.height {
            for x in 0..self.width {
                let idx = self.coord_to_idx(x, y);
                if (x + y) % 2 == 0 {
                    sub_a.push(idx);
                } else {
                    sub_b.push(idx);
                }
            }
        }

        (sub_a, sub_b)
    }
}
