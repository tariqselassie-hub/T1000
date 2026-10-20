// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Canonical Pattern Library for Quantum Conway's Game of Life.
//! Provides classical and quantum initial configurations: Gliders (Bethe-String Solitons),
//! Blinkers, Blocks, Toads, and Fractional Fermi Sea Bubbles.

use crate::lattice::Lattice2D;
use crate::state::QuantumCellStateSoA;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Type of canonical pattern to initialize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    /// Stationary 2x2 still-life block.
    Block,
    /// Coherent 3-cell period-2 oscillator.
    Blinker,
    /// 5-cell ballistic Bethe-string glider.
    Glider,
    /// 6-cell period-2 oscillator.
    Toad,
    /// 4-cell Fractional Fermi Sea topological bubble (alpha = 2).
    FractionalFermiSeaBubble,
    /// Opposing pair of colliding gliders.
    GliderPair,
    /// Superposed quantum beam splitter with pi/2 relative phase.
    QuantumBeamSplitter,
    /// Period-15 oscillator (Pentadecathlon).
    Pentadecathlon,
    /// Quantum Vortex Dipole: Counter-rotating phase vortex and anti-vortex pair.
    VortexDipole,
    /// HighLife B36/S23 Topological Replicator Seed.
    HighLifeReplicator,
    /// Ergodic Turbulent Quantum Soup with randomized phase distribution.
    TurbulentSoup,
    /// Relativistic Dirac wavepacket with momentum boost.
    DiracWavepacket,
    /// Belousov-Zhabotinsky spiral reaction seed.
    BelousovSpiral,
}

/// Helper to instantiate canonical patterns onto 2D lattices.
#[derive(Debug, Clone, Default)]
pub struct PatternFactory;

impl PatternFactory {
    /// Inject a canonical pattern onto the quantum state at top-left anchor (x0, y0).
    pub fn inject_pattern(
        pattern: PatternType,
        lattice: &Lattice2D,
        state: &mut QuantumCellStateSoA,
        x0: usize,
        y0: usize,
    ) {
        match pattern {
            PatternType::Block => Self::inject_block(lattice, state, x0, y0),
            PatternType::Blinker => Self::inject_blinker(lattice, state, x0, y0),
            PatternType::Glider => Self::inject_glider(lattice, state, x0, y0),
            PatternType::Toad => Self::inject_toad(lattice, state, x0, y0),
            PatternType::FractionalFermiSeaBubble => {
                Self::inject_ffs_bubble(lattice, state, x0, y0)
            }
            PatternType::GliderPair => Self::inject_glider_pair(lattice, state, x0, y0, 0.0),
            PatternType::QuantumBeamSplitter => {
                Self::inject_glider_pair(lattice, state, x0, y0, std::f64::consts::PI * 0.5)
            }
            PatternType::Pentadecathlon => Self::inject_pentadecathlon(lattice, state, x0, y0),
            PatternType::VortexDipole => Self::inject_vortex_dipole(lattice, state, x0, y0),
            PatternType::HighLifeReplicator => Self::inject_highlife_replicator(lattice, state, x0, y0),
            PatternType::TurbulentSoup => Self::inject_turbulent_soup(lattice, state, 42),
            PatternType::DiracWavepacket => Self::inject_dirac_wavepacket(lattice, state, x0, y0),
            PatternType::BelousovSpiral => Self::inject_belousov_spiral(lattice, state, x0, y0),
        }
    }

    /// 2x2 still-life block.
    fn inject_block(lattice: &Lattice2D, state: &mut QuantumCellStateSoA, x0: usize, y0: usize) {
        let offsets = [(0, 0), (1, 0), (0, 1), (1, 1)];
        for (dx, dy) in offsets {
            let x = (x0 + dx) % lattice.width;
            let y = (y0 + dy) % lattice.height;
            let idx = lattice.coord_to_idx(x, y);
            state.set_cell_living(idx);
        }
    }

    /// 3-cell blinker oscillator.
    fn inject_blinker(lattice: &Lattice2D, state: &mut QuantumCellStateSoA, x0: usize, y0: usize) {
        let offsets = [(0, 1), (1, 1), (2, 1)];
        for (dx, dy) in offsets {
            let x = (x0 + dx) % lattice.width;
            let y = (y0 + dy) % lattice.height;
            let idx = lattice.coord_to_idx(x, y);
            state.set_cell_living(idx);
        }
    }

    /// 5-cell traveling glider soliton.
    fn inject_glider(lattice: &Lattice2D, state: &mut QuantumCellStateSoA, x0: usize, y0: usize) {
        let coords = [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)];
        for (dx, dy) in coords {
            let x = (x0 + dx) % lattice.width;
            let y = (y0 + dy) % lattice.height;
            let idx = lattice.coord_to_idx(x, y);
            state.set_cell_living(idx);
        }
    }

    /// 6-cell toad oscillator.
    fn inject_toad(lattice: &Lattice2D, state: &mut QuantumCellStateSoA, x0: usize, y0: usize) {
        let coords = [(1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2)];
        for (dx, dy) in coords {
            let x = (x0 + dx) % lattice.width;
            let y = (y0 + dy) % lattice.height;
            let idx = lattice.coord_to_idx(x, y);
            state.set_cell_living(idx);
        }
    }

    /// 4-cell Fractional Fermi Sea topological bubble (alpha = 2 excitation branch).
    fn inject_ffs_bubble(
        lattice: &Lattice2D,
        state: &mut QuantumCellStateSoA,
        x0: usize,
        y0: usize,
    ) {
        let coords = [(0, 0), (1, 0), (0, 1), (1, 1)];
        for (dx, dy) in coords {
            let x = (x0 + dx) % lattice.width;
            let y = (y0 + dy) % lattice.height;
            let idx = lattice.coord_to_idx(x, y);
            state.set_cell_ffs(idx);
        }
    }

    /// Create a Schrödinger Cat Blinker: coherent equal superposition of horizontal and vertical blinker.
    pub fn inject_cat_blinker(
        lattice: &Lattice2D,
        state: &mut QuantumCellStateSoA,
        x0: usize,
        y0: usize,
    ) {
        let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
        let amp = Complex64::new(inv_sqrt2, 0.0);
        let vac = Complex64::new(0.0, 0.0);

        // Center cell (1, 1) is active in both horizontal and vertical
        let center_idx = lattice.coord_to_idx((x0 + 1) % lattice.width, (y0 + 1) % lattice.height);
        state.set_cell_living(center_idx);

        // Outer cells (0, 1) & (2, 1) horizontal; (1, 0) & (1, 2) vertical
        let outer_coords = [(0, 1), (2, 1), (1, 0), (1, 2)];
        for (dx, dy) in outer_coords {
            let idx = lattice.coord_to_idx((x0 + dx) % lattice.width, (y0 + dy) % lattice.height);
            state.set_cell_superposition(idx, amp, amp, vac);
        }
    }

    /// Inject two opposing gliders on collision trajectory with relative quantum phase.
    pub fn inject_glider_pair(
        lattice: &Lattice2D,
        state: &mut QuantumCellStateSoA,
        x0: usize,
        y0: usize,
        phase_rad: f64,
    ) {
        let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
        let amp_a = Complex64::new(inv_sqrt2, 0.0);
        let amp_b = Complex64::from_polar(inv_sqrt2, phase_rad);

        // Glider A (SE bound) at (x0, y0)
        let coords_a = [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)];
        for (dx, dy) in coords_a {
            let x = (x0 + dx) % lattice.width;
            let y = (y0 + dy) % lattice.height;
            let idx = lattice.coord_to_idx(x, y);
            let c0 = Complex64::new(state.re0[idx], state.im0[idx]);
            let c2 = Complex64::new(state.re2[idx], state.im2[idx]);
            let cur_c1 = Complex64::new(state.re1[idx], state.im1[idx]);
            state.set_cell_superposition(idx, c0, cur_c1 + amp_a, c2);
        }

        // Glider B (SW bound) at (x0 + 8, y0)
        let coords_b = [(1, 0), (0, 1), (0, 2), (1, 2), (2, 2)];
        let x_b0 = (x0 + 8) % lattice.width;
        for (dx, dy) in coords_b {
            let x = (x_b0 + dx) % lattice.width;
            let y = (y0 + dy) % lattice.height;
            let idx = lattice.coord_to_idx(x, y);
            let c0 = Complex64::new(state.re0[idx], state.im0[idx]);
            let c2 = Complex64::new(state.re2[idx], state.im2[idx]);
            let cur_c1 = Complex64::new(state.re1[idx], state.im1[idx]);
            state.set_cell_superposition(idx, c0, cur_c1 + amp_b, c2);
        }
    }

    /// Pentadecathlon: 10-cell period-15 oscillator.
    pub fn inject_pentadecathlon(
        lattice: &Lattice2D,
        state: &mut QuantumCellStateSoA,
        x0: usize,
        y0: usize,
    ) {
        let offsets = [
            (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0)
        ];
        for (dx, dy) in offsets {
            let x = (x0 + dx) % lattice.width;
            let y = (y0 + dy) % lattice.height;
            let idx = lattice.coord_to_idx(x, y);
            state.set_cell_living(idx);
        }
    }

    /// Quantum Vortex Dipole: Pair of counter-rotating phase vortices (+1 and -1).
    pub fn inject_vortex_dipole(
        lattice: &Lattice2D,
        state: &mut QuantumCellStateSoA,
        x0: usize,
        y0: usize,
    ) {
        let cx1 = x0 as f64;
        let cy1 = y0 as f64;
        let cx2 = ((x0 + 6) % lattice.width) as f64;
        let cy2 = y0 as f64;

        for y in 0..lattice.height {
            for x in 0..lattice.width {
                let dx1 = (x as f64) - cx1;
                let dy1 = (y as f64) - cy1;
                let dx2 = (x as f64) - cx2;
                let dy2 = (y as f64) - cy2;

                let r1 = (dx1 * dx1 + dy1 * dy1).sqrt();
                let r2 = (dx2 * dx2 + dy2 * dy2).sqrt();

                if r1 < 6.0 || r2 < 6.0 {
                    let theta1 = dy1.atan2(dx1);
                    let theta2 = -dy2.atan2(dx2);
                    let phase = theta1 + theta2;
                    let envelope = ((-0.15 * r1.min(r2)).exp())
                        * (1.0 - (-r1 * 0.8).exp())
                        * (1.0 - (-r2 * 0.8).exp());

                    let amp1 = Complex64::from_polar(envelope.clamp(0.0, 0.95), phase);
                    let rem = (1.0 - amp1.norm_sqr()).max(0.0).sqrt();
                    let amp0 = Complex64::new(rem, 0.0);

                    let idx = lattice.coord_to_idx(x, y);
                    state.set_cell_superposition(idx, amp0, amp1, Complex64::new(0.0, 0.0));
                }
            }
        }
    }

    /// HighLife B36/S23 Topological Replicator Seed: 12-cell symmetric replicator.
    pub fn inject_highlife_replicator(
        lattice: &Lattice2D,
        state: &mut QuantumCellStateSoA,
        x0: usize,
        y0: usize,
    ) {
        let coords = [
            (1, 0), (2, 0), (3, 0),
            (0, 1), (3, 1),
            (0, 2), (3, 2),
            (0, 3), (1, 3), (2, 3), (3, 3),
            (2, 4),
        ];
        for (dx, dy) in coords {
            let x = (x0 + dx) % lattice.width;
            let y = (y0 + dy) % lattice.height;
            let idx = lattice.coord_to_idx(x, y);
            state.set_cell_living(idx);
        }
    }

    /// Ergodic Turbulent Quantum Soup: Random coherent phases across lattice.
    pub fn inject_turbulent_soup(
        lattice: &Lattice2D,
        state: &mut QuantumCellStateSoA,
        seed: u64,
    ) {
        let mut rng = seed;
        let mut next_f64 = || -> f64 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 32) as f64) / (u32::MAX as f64)
        };

        for i in 0..lattice.total_cells {
            let p_alive = next_f64();
            if p_alive > 0.65 {
                let phase = next_f64() * 2.0 * std::f64::consts::PI;
                let is_ffs = p_alive > 0.88;
                if is_ffs {
                    let amp2 = Complex64::from_polar(1.0, phase);
                    state.set_cell_superposition(i, Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), amp2);
                } else {
                    let amp1 = Complex64::from_polar(1.0, phase);
                    state.set_cell_superposition(i, Complex64::new(0.0, 0.0), amp1, Complex64::new(0.0, 0.0));
                }
            } else {
                state.set_cell_vacuum(i);
            }
        }
    }

    /// Relativistic Dirac wavepacket with momentum boost.
    pub fn inject_dirac_wavepacket(
        lattice: &Lattice2D,
        state: &mut QuantumCellStateSoA,
        x0: usize,
        y0: usize,
    ) {
        let cx = x0 as f64;
        let cy = y0 as f64;
        let p_x = 0.8;
        let p_y = 0.4;
        let sigma = 3.5;

        for y in 0..lattice.height {
            for x in 0..lattice.width {
                let dx = (x as f64) - cx;
                let dy = (y as f64) - cy;
                let r2 = dx * dx + dy * dy;
                let envelope = (-r2 / (2.0 * sigma * sigma)).exp();

                if envelope > 0.04 {
                    let phase = p_x * dx + p_y * dy;
                    let amp1 = Complex64::from_polar(envelope * 0.8, phase);
                    let amp2 = Complex64::from_polar(envelope * 0.4, phase + std::f64::consts::PI * 0.5);
                    let rem_sq = (1.0 - (amp1.norm_sqr() + amp2.norm_sqr())).max(0.0);
                    let amp0 = Complex64::new(rem_sq.sqrt(), 0.0);

                    let idx = lattice.coord_to_idx(x, y);
                    state.set_cell_superposition(idx, amp0, amp1, amp2);
                }
            }
        }
    }

    /// Belousov-Zhabotinsky spiral reaction seed: Cross-phase pinning.
    pub fn inject_belousov_spiral(
        lattice: &Lattice2D,
        state: &mut QuantumCellStateSoA,
        x0: usize,
        y0: usize,
    ) {
        let arm_len = 5;
        for d in 0..arm_len {
            let x = (x0 + d) % lattice.width;
            let y = y0 % lattice.height;
            let idx = lattice.coord_to_idx(x, y);
            state.set_cell_superposition(
                idx,
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            );
        }
        for d in 0..arm_len {
            let x = x0 % lattice.width;
            let y = (y0 + d) % lattice.height;
            let idx = lattice.coord_to_idx(x, y);
            state.set_cell_superposition(
                idx,
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 1.0),
            );
        }
    }
}
