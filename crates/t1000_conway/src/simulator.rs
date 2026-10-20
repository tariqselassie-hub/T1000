// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Quantum Conway Simulator Engine.
//! Executes zero-allocation unitary Quantum Cellular Automata (QCA) steps
//! via split-step checkerboard decomposition, Margolus plaquettes, and
//! Fractional Fermi Sea (FFS) interaction holonomy cycles.

use crate::holonomy::{build_holonomy_unitary, HolonomyParams};
use crate::lattice::Lattice2D;
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
    /// Maximum absolute local quantum vorticity |omega|.
    pub max_vorticity: f64,
    /// Number of identified quantized vortex cores (|omega| > 0.15).
    pub vortex_count: usize,
    /// Mean magnitude of quantum probability flux |j|.
    pub mean_current_magnitude: f64,
    /// Quantum scrambling operator non-commutativity proxy (OTOC).
    pub scrambling_otoc: f64,
}

/// The main Quantum Conway's Game of Life simulation engine.
#[derive(Debug, Clone)]
pub struct QuantumConwaySimulator {
    /// 2D lattice geometry.
    pub lattice: Lattice2D,
    /// Active SoA quantum cell state.
    pub state: QuantumCellStateSoA,
    /// Pre-allocated buffer for intermediate cell densities.
    density_buffer: Vec<f64>,
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
            density_buffer: vec![0.0; total],
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
        crate::rules::compute_neighborhood_occupations_buffered(
            &self.lattice,
            &self.state,
            &mut self.density_buffer,
            &mut self.nbr_buffer,
        );
        self.apply_sublattice_unitary(&sub_a);

        // 2. Advance Sublattice B with updated Sublattice A densities
        crate::rules::compute_neighborhood_occupations_buffered(
            &self.lattice,
            &self.state,
            &mut self.density_buffer,
            &mut self.nbr_buffer,
        );
        self.apply_sublattice_unitary(&sub_b);

        self.current_step += 1;
        self.gather_telemetry()
    }

    /// Apply unitary holonomy transfer matrix in-place to a subset of cells
    /// using AVX2+FMA 4-cell packet streaming.
    fn apply_sublattice_unitary(&mut self, indices: &[usize]) {
        let chunks = indices.chunks_exact(4);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let idx4 = [chunk[0], chunk[1], chunk[2], chunk[3]];
            let u4 = [
                build_holonomy_unitary(self.nbr_buffer[idx4[0]], &self.params),
                build_holonomy_unitary(self.nbr_buffer[idx4[1]], &self.params),
                build_holonomy_unitary(self.nbr_buffer[idx4[2]], &self.params),
                build_holonomy_unitary(self.nbr_buffer[idx4[3]], &self.params),
            ];
            crate::simd::apply_unitary_4cells(
                &mut self.state.re0,
                &mut self.state.im0,
                &mut self.state.re1,
                &mut self.state.im1,
                &mut self.state.re2,
                &mut self.state.im2,
                idx4,
                &u4,
            );
        }

        for &idx in remainder {
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
        let currents = self.compute_probability_current();
        let vorticities = self.compute_vorticity(&currents);

        let mut max_vort = 0.0f64;
        let mut vortex_cnt = 0usize;
        let mut current_mag_sum = 0.0f64;

        for (curr, &vort) in currents.iter().zip(vorticities.iter()) {
            let v_abs = vort.abs();
            if v_abs > max_vort {
                max_vort = v_abs;
            }
            if v_abs > 0.15 {
                vortex_cnt += 1;
            }
            current_mag_sum += (curr[0] * curr[0] + curr[1] * curr[1]).sqrt();
        }

        let mean_current = if self.lattice.total_cells > 0 {
            current_mag_sum / (self.lattice.total_cells as f64)
        } else {
            0.0
        };

        let scrambling = self.compute_scrambling_otoc();

        StepTelemetry {
            step: self.current_step,
            total_population: total_pop,
            living_population: living_pop,
            ffs_population: ffs_pop,
            max_norm_error: self.state.max_normalization_error(),
            entanglement_entropy: entropy,
            mean_momentum_density: momentum_density,
            max_vorticity: max_vort,
            vortex_count: vortex_cnt,
            mean_current_magnitude: mean_current,
            scrambling_otoc: scrambling,
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
                        let phase =
                            2.0 * PI * ((kx * x) as f64 / w as f64 + (ky * y) as f64 / h as f64);
                        fourier_amp += amp * Complex64::from_polar(1.0, -phase);
                    }
                }
                let nk = fourier_amp.norm_sqr() / (w * h) as f64;
                sum_density += nk;
                count += 1;
            }
        }

        if count > 0 {
            sum_density / (count as f64)
        } else {
            0.0
        }
    }

    /// Compute quantum probability flux current vector field j(x,y) = [jx, jy].
    pub fn compute_probability_current(&self) -> Vec<[f64; 2]> {
        let w = self.lattice.width;
        let h = self.lattice.height;
        let mut currents = vec![[0.0, 0.0]; self.lattice.total_cells];

        for y in 0..h {
            for x in 0..w {
                let idx = self.lattice.coord_to_idx(x, y);
                let x_next = self.lattice.coord_to_idx((x + 1) % w, y);
                let x_prev = self.lattice.coord_to_idx((x + w - 1) % w, y);
                let y_next = self.lattice.coord_to_idx(x, (y + 1) % h);
                let y_prev = self.lattice.coord_to_idx(x, (y + h - 1) % h);

                let c1 = Complex64::new(self.state.re1[idx], self.state.im1[idx]);
                let c2 = Complex64::new(self.state.re2[idx], self.state.im2[idx]);

                let dc1_dx = Complex64::new(
                    self.state.re1[x_next] - self.state.re1[x_prev],
                    self.state.im1[x_next] - self.state.im1[x_prev],
                ) * 0.5;
                let dc2_dx = Complex64::new(
                    self.state.re2[x_next] - self.state.re2[x_prev],
                    self.state.im2[x_next] - self.state.im2[x_prev],
                ) * 0.5;

                let dc1_dy = Complex64::new(
                    self.state.re1[y_next] - self.state.re1[y_prev],
                    self.state.im1[y_next] - self.state.im1[y_prev],
                ) * 0.5;
                let dc2_dy = Complex64::new(
                    self.state.re2[y_next] - self.state.re2[y_prev],
                    self.state.im2[y_next] - self.state.im2[y_prev],
                ) * 0.5;

                let jx = (c1.conj() * dc1_dx + 2.0 * c2.conj() * dc2_dx).im;
                let jy = (c1.conj() * dc1_dy + 2.0 * c2.conj() * dc2_dy).im;
                currents[idx] = [jx, jy];
            }
        }
        currents
    }

    /// Compute local quantum vorticity omega(x,y) = curl(j) = d(jy)/dx - d(jx)/dy.
    pub fn compute_vorticity(&self, currents: &[[f64; 2]]) -> Vec<f64> {
        let w = self.lattice.width;
        let h = self.lattice.height;
        let mut vort = vec![0.0; self.lattice.total_cells];

        for y in 0..h {
            for x in 0..w {
                let idx = self.lattice.coord_to_idx(x, y);
                let x_next = self.lattice.coord_to_idx((x + 1) % w, y);
                let x_prev = self.lattice.coord_to_idx((x + w - 1) % w, y);
                let y_next = self.lattice.coord_to_idx(x, (y + 1) % h);
                let y_prev = self.lattice.coord_to_idx(x, (y + h - 1) % h);

                let djy_dx = (currents[x_next][1] - currents[x_prev][1]) * 0.5;
                let djx_dy = (currents[y_next][0] - currents[y_prev][0]) * 0.5;
                vort[idx] = djy_dx - djx_dy;
            }
        }
        vort
    }

    /// Compute quantum scrambling / operator non-commutativity proxy (OTOC).
    pub fn compute_scrambling_otoc(&self) -> f64 {
        let w = self.lattice.width as f64;
        let h = self.lattice.height as f64;
        let cx = w * 0.5;
        let cy = h * 0.5;
        let mut otoc_sum = 0.0f64;
        let mut tot_weight = 0.0f64;

        for y in 0..self.lattice.height {
            for x in 0..self.lattice.width {
                let idx = self.lattice.coord_to_idx(x, y);
                let dens = self.state.cell_density(idx);
                let dx = (x as f64) - cx;
                let dy = (y as f64) - cy;
                let r2 = dx * dx + dy * dy;
                otoc_sum += dens * r2;
                tot_weight += dens;
            }
        }

        if tot_weight > 1e-9 {
            (otoc_sum / tot_weight) / (w * h)
        } else {
            0.0
        }
    }
}
