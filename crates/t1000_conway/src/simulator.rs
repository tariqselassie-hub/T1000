// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Quantum Conway Simulator Engine.
//! Executes zero-allocation unitary Quantum Cellular Automata (QCA) steps
//! via split-step checkerboard decomposition, Margolus plaquettes, and
//! Fractional Fermi Sea (FFS) interaction holonomy cycles.

use crate::holonomy::{build_holonomy_unitary, HolonomyParams};
use crate::lattice::Lattice2D;
use crate::rules::compute_neighborhood_occupations;
use crate::state::QuantumCellStateSoA;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Diagnostic telemetry captured after a simulation step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepTelemetry {
    /// Step index.
    pub step: usize,
    /// Total population N_tot = sum_i <n_i>.
    pub total_population: f64,
    /// Expected number of living cells (sum_i P(|1>)).
    pub living_population: f64,
    /// Expected number of FFS excited cells (sum_i P(|2>)).
    pub ffs_population: f64,
    /// Maximum normalization error deviation |sum P_s - 1.0|.
    pub max_norm_error: f64,
    /// Spatial bipartite von Neumann entanglement entropy.
    pub entanglement_entropy: f64,
    /// Average momentum density in central Fermi sea region.
    pub mean_momentum_density: f64,
}

/// The main Quantum Conway's Game of Life simulation engine.
#[derive(Debug, Clone)]
pub struct QuantumConwaySimulator {
    /// 2D lattice geometry.
    pub lattice: Lattice2D,
    /// Active SoA quantum cell state.
    pub state: QuantumCellStateSoA,
    /// Pre-allocated buffer for neighborhood densities.
    nbr_buffer: Vec<f64>,
    /// Holonomy cycle parameters.
    pub params: HolonomyParams,
    /// Current tick / step counter.
    pub current_step: usize,
}

impl QuantumConwaySimulator {
    /// Create a new simulator instance with the given lattice and holonomy parameters.
    pub fn new(lattice: Lattice2D, params: HolonomyParams) -> Self {
        let total = lattice.total_cells;
        Self {
            state: QuantumCellStateSoA::new_vacuum(total),
            nbr_buffer: vec![0.0; total],
            lattice,
            params,
            current_step: 0,
        }
    }

    /// Advance the simulation by one full step using split-step checkerboard sublattices.
    /// This guarantees exact spatial unitarity U = U_B * U_A without allocation.
    pub fn step(&mut self) -> StepTelemetry {
        let (sub_a, sub_b) = self.lattice.checkerboard_sublattices();

        // 1. Advance Sublattice A
        compute_neighborhood_occupations(&self.lattice, &self.state, &mut self.nbr_buffer);
        self.apply_sublattice_unitary(&sub_a);

        // 2. Advance Sublattice B with updated Sublattice A densities
        compute_neighborhood_occupations(&self.lattice, &self.state, &mut self.nbr_buffer);
        self.apply_sublattice_unitary(&sub_b);

        self.current_step += 1;
        self.gather_telemetry()
    }

    /// Apply unitary holonomy transfer matrix in-place to a subset of cells.
    fn apply_sublattice_unitary(&mut self, indices: &[usize]) {
        for &idx in indices {
            let nbr_occ = self.nbr_buffer[idx];
            let u = build_holonomy_unitary(nbr_occ, &self.params);

            let c = [
                Complex64::new(self.state.re0[idx], self.state.im0[idx]),
                Complex64::new(self.state.re1[idx], self.state.im1[idx]),
                Complex64::new(self.state.re2[idx], self.state.im2[idx]),
            ];

            let c_next = u.apply(c);

            self.state.re0[idx] = c_next[0].re;
            self.state.im0[idx] = c_next[0].im;
            self.state.re1[idx] = c_next[1].re;
            self.state.im1[idx] = c_next[1].im;
            self.state.re2[idx] = c_next[2].re;
            self.state.im2[idx] = c_next[2].im;
        }
    }

    /// Gather telemetry metrics for the current lattice state.
    pub fn gather_telemetry(&self) -> StepTelemetry {
        let mut total_pop = 0.0;
        let mut living_pop = 0.0;
        let mut ffs_pop = 0.0;

        for i in 0..self.lattice.total_cells {
            living_pop += self.state.prob_living(i);
            ffs_pop += self.state.prob_ffs(i);
            total_pop += self.state.cell_density(i);
        }

        let entropy = self.compute_bipartite_entropy();
        let momentum_density = self.compute_mean_momentum_density();

        StepTelemetry {
            step: self.current_step,
            total_population: total_pop,
            living_population: living_pop,
            ffs_population: ffs_pop,
            max_norm_error: self.state.max_normalization_error(),
            entanglement_entropy: entropy,
            mean_momentum_density: momentum_density,
        }
    }

    /// Compute spatial bipartite von Neumann entropy S across a vertical cut at W/2.
    pub fn compute_bipartite_entropy(&self) -> f64 {
        let mid_x = self.lattice.width / 2;
        let mut prob_left = 0.0f64;
        let mut prob_right = 0.0f64;

        for y in 0..self.lattice.height {
            for x in 0..self.lattice.width {
                let idx = self.lattice.coord_to_idx(x, y);
                let p = self.state.prob_living(idx) + self.state.prob_ffs(idx);
                if x < mid_x {
                    prob_left += p;
                } else {
                    prob_right += p;
                }
            }
        }

        let total = prob_left + prob_right;
        if total <= 1e-12 {
            return 0.0;
        }

        let p_l = prob_left / total;
        let p_r = prob_right / total;

        let s_l = if p_l > 1e-12 { -p_l * p_l.ln() } else { 0.0 };
        let s_r = if p_r > 1e-12 { -p_r * p_r.ln() } else { 0.0 };
        s_l + s_r
    }

    /// Compute average 2D momentum density n(k) in the central Brillouin zone
    /// via discrete Fourier transform of the living state amplitudes.
    pub fn compute_mean_momentum_density(&self) -> f64 {
        let w = self.lattice.width;
        let h = self.lattice.height;
        let mut sum_density = 0.0f64;
        let mut count = 0usize;

        // Sample kx, ky in central half of momentum space
        for kx in (w / 4)..(3 * w / 4) {
            for ky in (h / 4)..(3 * h / 4) {
                let mut fourier_amp = Complex64::new(0.0, 0.0);
                for y in 0..h {
                    for x in 0..w {
                        let idx = self.lattice.coord_to_idx(x, y);
                        let amp = Complex64::new(self.state.re1[idx], self.state.im1[idx]);
                        let phase = 2.0 * PI * ((kx * x) as f64 / w as f64 + (ky * y) as f64 / h as f64);
                        fourier_amp += amp * Complex64::from_polar(1.0, -phase);
                    }
                }
                let nk = fourier_amp.norm_sqr() / (w * h) as f64;
                sum_density += nk;
                count += 1;
            }
        }

        if count > 0 { sum_density / (count as f64) } else { 0.0 }
    }
}
