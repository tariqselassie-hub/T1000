// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Quantum Glider Collision S-Matrix & Beam-Splitter Interferometry.
//! Formulates coherent soliton collisions in Unitary Quantum Cellular Automata (QCA).
//! Tracks phase-dependent constructive/destructive interference, bound-state formation,
//! and Mach-Zehnder quantum visibility V = (P_max - P_min) / (P_max + P_min).

use crate::lattice::Lattice2D;
use crate::simulator::QuantumConwaySimulator;
use crate::state::QuantumCellStateSoA;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Direction of glider soliton propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GliderDirection {
    /// Moving South-East (+x, +y).
    SouthEast,
    /// Moving South-West (-x, +y).
    SouthWest,
    /// Moving North-East (+x, -y).
    NorthEast,
    /// Moving North-West (-x, -y).
    NorthWest,
}

impl GliderDirection {
    /// Return the 5 relative cell offsets for the glider oriented in this direction.
    pub fn cell_offsets(&self) -> [(isize, isize); 5] {
        match self {
            Self::SouthEast => [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)],
            Self::SouthWest => [(1, 0), (0, 1), (0, 2), (1, 2), (2, 2)],
            Self::NorthEast => [(0, 0), (1, 0), (2, 0), (2, 1), (1, 2)],
            Self::NorthWest => [(0, 0), (1, 0), (2, 0), (0, 1), (1, 2)],
        }
    }
}

/// Diagnostic telemetry captured from a quantum glider collision run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollisionTelemetry {
    /// Relative quantum phase between the two gliders in radians.
    pub relative_phase_rad: f64,
    /// Relative phase in degrees.
    pub phase_deg: f64,
    /// Simulation steps executed through collision zone.
    pub steps: usize,
    /// Final surviving living population P_living.
    pub living_population: f64,
    /// Final FFS excited population P_ffs.
    pub ffs_population: f64,
    /// Transmission ratio into forward scatter quadrant.
    pub transmission_prob: f64,
    /// Bound-state formation ratio in central collision zone.
    pub bound_state_prob: f64,
    /// Annihilation ratio into vacuum.
    pub annihilation_prob: f64,
    /// Peak bipartite von Neumann entanglement entropy generated.
    pub peak_entanglement: f64,
    /// Final state normalization error.
    pub max_norm_error: f64,
}

/// S-Matrix and Mach-Zehnder Visibility analysis over a phase sweep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GliderInterferometryReport {
    /// Number of phase sweep points.
    pub phase_points: usize,
    /// Measured collision telemetry across relative phases.
    pub sweep_results: Vec<CollisionTelemetry>,
    /// Minimum transmission probability.
    pub min_transmission: f64,
    /// Maximum transmission probability.
    pub max_transmission: f64,
    /// Quantum interferometric visibility V = (P_max - P_min) / (P_max + P_min).
    pub visibility: f64,
    /// Peak entanglement entropy observed across the sweep.
    pub max_entanglement_nats: f64,
}

/// Inject a directed glider soliton with complex amplitude scaling.
pub fn inject_directed_glider(
    lattice: &Lattice2D,
    state: &mut QuantumCellStateSoA,
    x0: usize,
    y0: usize,
    dir: GliderDirection,
    amplitude: Complex64,
) {
    let offsets = dir.cell_offsets();
    let w = lattice.width as isize;
    let h = lattice.height as isize;

    for (dx, dy) in offsets {
        let x = ((x0 as isize + dx).rem_euclid(w)) as usize;
        let y = ((y0 as isize + dy).rem_euclid(h)) as usize;
        let idx = lattice.coord_to_idx(x, y);

        // Accumulate amplitude in state (living channel)
        let cur_re1 = state.re1[idx];
        let cur_im1 = state.im1[idx];
        let next_c1 = Complex64::new(cur_re1, cur_im1) + amplitude;
        let c0 = Complex64::new(state.re0[idx], state.im0[idx]);
        let c2 = Complex64::new(state.re2[idx], state.im2[idx]);
        state.set_cell_superposition(idx, c0, next_c1, c2);
    }
}

/// Set up a symmetrical dual-glider collision experiment on a lattice.
/// Glider A starts at (x_a, y_a) heading SE.
/// Glider B starts at (x_b, y_b) heading SW.
/// The relative phase `phase_rad` tunes quantum interference in the interaction zone.
pub fn setup_glider_collision(
    sim: &mut QuantumConwaySimulator,
    x_a: usize,
    y_a: usize,
    x_b: usize,
    y_b: usize,
    phase_rad: f64,
) {
    let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
    let amp_a = Complex64::new(inv_sqrt2, 0.0);
    let amp_b = Complex64::from_polar(inv_sqrt2, phase_rad);

    inject_directed_glider(&sim.lattice, &mut sim.state, x_a, y_a, GliderDirection::SouthEast, amp_a);
    inject_directed_glider(&sim.lattice, &mut sim.state, x_b, y_b, GliderDirection::SouthWest, amp_b);
}

/// Execute a glider collision simulation run and compute scattering matrix observables.
pub fn run_collision_experiment(
    mut sim: QuantumConwaySimulator,
    x_a: usize,
    y_a: usize,
    x_b: usize,
    y_b: usize,
    phase_rad: f64,
    steps: usize,
) -> CollisionTelemetry {
    setup_glider_collision(&mut sim, x_a, y_a, x_b, y_b, phase_rad);

    let mut peak_entropy = 0.0f64;
    let mut last_telem = sim.gather_telemetry();

    for _ in 0..steps {
        let telem = sim.step();
        if telem.entanglement_entropy > peak_entropy {
            peak_entropy = telem.entanglement_entropy;
        }
        last_telem = telem;
    }

    // Partition lattice into quadrants to measure transmission vs bound-state retention
    let mid_y = sim.lattice.height / 2;
    let mut lower_pop = 0.0f64;
    let mut center_bound_pop = 0.0f64;
    let total_pop = last_telem.total_population;

    let center_x_min = sim.lattice.width / 4;
    let center_x_max = 3 * sim.lattice.width / 4;
    let center_y_min = sim.lattice.height / 4;
    let center_y_max = 3 * sim.lattice.height / 4;

    for y in 0..sim.lattice.height {
        for x in 0..sim.lattice.width {
            let idx = sim.lattice.coord_to_idx(x, y);
            let d = sim.state.cell_density(idx);
            if y >= mid_y {
                lower_pop += d;
            }
            if x >= center_x_min && x <= center_x_max && y >= center_y_min && y <= center_y_max {
                center_bound_pop += d;
            }
        }
    }

    let initial_nominal = 10.0f64; // Two 5-cell gliders
    let transmission = (lower_pop / initial_nominal).clamp(0.0, 1.0);
    let bound = (center_bound_pop / initial_nominal).clamp(0.0, 1.0);
    let annihilation = (1.0 - (total_pop / initial_nominal)).clamp(0.0, 1.0);

    CollisionTelemetry {
        relative_phase_rad: phase_rad,
        phase_deg: phase_rad.to_degrees(),
        steps,
        living_population: last_telem.living_population,
        ffs_population: last_telem.ffs_population,
        transmission_prob: transmission,
        bound_state_prob: bound,
        annihilation_prob: annihilation,
        peak_entanglement: peak_entropy,
        max_norm_error: last_telem.max_norm_error,
    }
}

/// Perform a full Mach-Zehnder relative phase sweep to measure quantum visibility.
pub fn sweep_glider_interferometry(
    width: usize,
    height: usize,
    num_phases: usize,
    steps: usize,
) -> GliderInterferometryReport {
    use crate::lattice::BoundaryCondition;
    use crate::holonomy::HolonomyParams;

    let mut sweep_results = Vec::with_capacity(num_phases);
    let mut min_t = 1.0f64;
    let mut max_t = 0.0f64;
    let mut max_ent = 0.0f64;

    // Symmetrical launch positions
    let x_a = 2;
    let y_a = 2;
    let x_b = width.saturating_sub(6);
    let y_b = 2;

    for i in 0..num_phases {
        let phase = 2.0 * std::f64::consts::PI * (f64::from(i as u32) / f64::from(num_phases as u32));
        let lattice = Lattice2D::new(width, height, BoundaryCondition::Toroidal);
        let sim = QuantumConwaySimulator::new(lattice, HolonomyParams::default());

        let telem = run_collision_experiment(sim, x_a, y_a, x_b, y_b, phase, steps);
        if telem.transmission_prob < min_t {
            min_t = telem.transmission_prob;
        }
        if telem.transmission_prob > max_t {
            max_t = telem.transmission_prob;
        }
        if telem.peak_entanglement > max_ent {
            max_ent = telem.peak_entanglement;
        }
        sweep_results.push(telem);
    }

    let vis = if (max_t + min_t) > 1e-12 {
        (max_t - min_t) / (max_t + min_t)
    } else {
        0.0
    };

    GliderInterferometryReport {
        phase_points: num_phases,
        sweep_results,
        min_transmission: min_t,
        max_transmission: max_t,
        visibility: vis,
        max_entanglement_nats: max_ent,
    }
}
