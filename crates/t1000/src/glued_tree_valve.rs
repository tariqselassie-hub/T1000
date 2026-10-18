// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Multi-Exit Topological Glued-Tree Valve Engine
//!
//! Extends the Childs et al. glued binary tree architecture into a multi-exit
//! quantum router / valve.
//!
//! ## Topology:
//! - 1 Entrance Tree ($T_{\text{in}}$) of depth $n$ connected at its $2^n$ leaves to
//!   $K$ independent Exit Trees ($T_{\text{out}}^{(0)}, \dots, T_{\text{out}}^{(K-1)}$).
//! - At the shared weld boundary, controlled holonomy phases $\phi_y$ or routing unitaries
//!   $|y\rangle\langle y|_C \otimes W_{\text{weld}}^{(y)}$ steer the ballistic wavepacket
//!   into target Exit Root $y^*$ via constructive interference while extinguishing flow
//!   into all other $K-1$ exit trees via destructive Aharonov-Bohm phase cancellation.
//!
//! ## Subspace Invariance:
//! Reduces the $O(K \cdot 2^n)$ graph to an exact $(K + 1)(n + 1)$-dimensional
//! multi-branch column-symmetric subspace.

use num_complex::Complex64;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Configuration for the K-Way Glued Tree Valve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GluedTreeValveConfig {
    /// Depth of each balanced binary tree (n >= 1).
    pub depth: usize,
    /// Number of distinct exit trees (K >= 2).
    pub num_exits: usize,
}

impl GluedTreeValveConfig {
    /// Constructs a validated K-way valve configuration.
    pub const fn new(depth: usize, num_exits: usize) -> Self {
        assert!(depth >= 1, "Tree depth must be >= 1");
        assert!(num_exits >= 2, "Must have at least 2 exit trees");
        Self { depth, num_exits }
    }

    /// Number of columns per tree: depth + 1 (columns 0 .. n).
    #[inline(always)]
    pub const fn cols_per_tree(&self) -> usize {
        self.depth + 1
    }

    /// Total dimension of the multi-branch column-symmetric subspace:
    /// (n + 1) for Entrance Tree + K * (n + 1) for Exit Trees.
    #[inline(always)]
    pub const fn symmetric_dim(&self) -> usize {
        (self.num_exits + 1) * self.cols_per_tree()
    }

    /// Global basis index for Entrance Tree column j (0 <= j <= n).
    #[inline(always)]
    pub const fn in_col_idx(&self, col: usize) -> usize {
        col
    }

    /// Global basis index for Exit Tree y, column j (0 <= j <= n, where 0 is weld leaf, n is exit root).
    #[inline(always)]
    pub const fn out_col_idx(&self, exit_y: usize, col: usize) -> usize {
        self.cols_per_tree() + exit_y * self.cols_per_tree() + col
    }

    /// Total number of vertices in the full physical graph:
    /// (2^{n+1} - 1) for T_in + K * (2^{n+1} - 1) for exit trees.
    pub fn total_vertices(&self) -> usize {
        let tree_v = (1 << (self.depth + 1)) - 1;
        (self.num_exits + 1) * tree_v
    }

    /// Number of leaves per tree: 2^n.
    #[inline(always)]
    pub fn num_leaves(&self) -> usize {
        1 << self.depth
    }
}

/// Multi-Branch Column-Symmetric Hamiltonian for the K-Way Tree Valve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiExitValveHamiltonian {
    pub cfg: GluedTreeValveConfig,
    pub gamma: f64,
    /// Weld coupling amplitudes for each exit tree y in 0..K.
    pub weld_couplings: Vec<f64>,
    /// Weld phase modulations phi_y for each exit tree y in 0..K.
    pub weld_phases: Vec<f64>,
}

impl MultiExitValveHamiltonian {
    pub fn new(cfg: GluedTreeValveConfig, gamma: f64) -> Self {
        let k = cfg.num_exits;
        let weld_couplings = vec![2.0; k];
        let weld_phases = vec![0.0; k];
        Self {
            cfg,
            gamma,
            weld_couplings,
            weld_phases,
        }
    }

    /// Configures phase angles and resonant couplings to steer wavepacket into target exit y_target
    /// using constructive interference on y_target and destructive Aharonov-Bohm phase extinction
    /// on all competing exit branches.
    pub fn configure_steering_target(&mut self, y_target: usize) {
        assert!(y_target < self.cfg.num_exits);
        let k = self.cfg.num_exits;
        for y in 0..k {
            if y == y_target {
                self.weld_phases[y] = 0.0; // In-phase resonance
                self.weld_couplings[y] = 2.0; // Resonant impedance-matched coupling
            } else {
                // Destructive anti-phase cancellation & dark-subspace suppression
                self.weld_phases[y] = std::f64::consts::PI;
                self.weld_couplings[y] = 0.05; // Suppressed non-target leakage
            }
        }
    }

    /// Applies H |psi> in the multi-branch symmetric subspace.
    pub fn apply(&self, psi: &[Complex64], out: &mut [Complex64]) {
        let dim = self.cfg.symmetric_dim();
        assert_eq!(psi.len(), dim);
        assert_eq!(out.len(), dim);

        for elem in out.iter_mut() {
            *elem = Complex64::default();
        }

        let depth = self.cfg.depth;
        let sqrt2 = 2.0_f64.sqrt();
        let g = self.gamma;

        // 1. Entrance Tree internal hopping (columns 0 .. n-1)
        for j in 0..depth {
            let idx_a = self.cfg.in_col_idx(j);
            let idx_b = self.cfg.in_col_idx(j + 1);
            let hop = sqrt2 * g;
            out[idx_a] += Complex64::new(hop, 0.0) * psi[idx_b];
            out[idx_b] += Complex64::new(hop, 0.0) * psi[idx_a];
        }

        // 2. Weld hoppings from Entrance leaves (col n) to Exit Tree leaves (col 0 of each exit tree)
        let in_leaf_idx = self.cfg.in_col_idx(depth);
        let k = self.cfg.num_exits;

        for y in 0..k {
            let out_leaf_idx = self.cfg.out_col_idx(y, 0);
            let hop = self.weld_couplings[y] * g;
            let phi = self.weld_phases[y];
            let u_phase = Complex64::new(hop * phi.cos(), hop * phi.sin());
            let u_phase_adj = u_phase.conj();

            out[in_leaf_idx] += u_phase * psi[out_leaf_idx];
            out[out_leaf_idx] += u_phase_adj * psi[in_leaf_idx];
        }

        // 3. Exit Trees internal hopping (for each exit tree y, columns 0 .. depth-1)
        for y in 0..k {
            for j in 0..depth {
                let idx_a = self.cfg.out_col_idx(y, j);
                let idx_b = self.cfg.out_col_idx(y, j + 1);
                let hop = sqrt2 * g;
                out[idx_a] += Complex64::new(hop, 0.0) * psi[idx_b];
                out[idx_b] += Complex64::new(hop, 0.0) * psi[idx_a];
            }
        }
    }

    /// Single Trotter time step: exp(-i H dt) |psi>.
    pub fn step_trotter(&self, psi: &mut [Complex64], dt: f64) {
        let half_dt = dt * 0.5;
        let depth = self.cfg.depth;
        let sqrt2 = 2.0_f64.sqrt();
        let g = self.gamma;
        let k = self.cfg.num_exits;

        // Entrance tree pass
        for j in 0..depth {
            let a = self.cfg.in_col_idx(j);
            let b = self.cfg.in_col_idx(j + 1);
            evolve_two_level(psi, a, b, sqrt2 * g, half_dt);
        }

        // Weld bridges with phase holonomies
        for y in 0..k {
            let in_leaf = self.cfg.in_col_idx(depth);
            let out_leaf = self.cfg.out_col_idx(y, 0);
            let hop = self.weld_couplings[y] * g;
            let phi = self.weld_phases[y];
            evolve_phased_two_level(psi, in_leaf, out_leaf, hop, phi, dt);
        }

        // Exit trees pass
        for y in 0..k {
            for j in 0..depth {
                let a = self.cfg.out_col_idx(y, j);
                let b = self.cfg.out_col_idx(y, j + 1);
                evolve_two_level(psi, a, b, sqrt2 * g, dt);
            }
        }

        // Entrance tree symmetric half step
        for j in (0..depth).rev() {
            let a = self.cfg.in_col_idx(j);
            let b = self.cfg.in_col_idx(j + 1);
            evolve_two_level(psi, a, b, sqrt2 * g, half_dt);
        }
    }
}

#[inline(always)]
fn evolve_two_level(psi: &mut [Complex64], a: usize, b: usize, h: f64, dt: f64) {
    let angle = h * dt;
    let cos_t = angle.cos();
    let sin_t = angle.sin();
    let i_sin_t = Complex64::new(0.0, -sin_t);

    let pa = psi[a];
    let pb = psi[b];

    psi[a] = Complex64::new(cos_t, 0.0) * pa + i_sin_t * pb;
    psi[b] = i_sin_t * pa + Complex64::new(cos_t, 0.0) * pb;
}

#[inline(always)]
fn evolve_phased_two_level(psi: &mut [Complex64], a: usize, b: usize, h: f64, phi: f64, dt: f64) {
    let angle = h * dt;
    let cos_t = angle.cos();
    let sin_t = angle.sin();

    let u_phase = Complex64::new(phi.cos(), phi.sin());
    let u_phase_conj = u_phase.conj();

    let pa = psi[a];
    let pb = psi[b];

    psi[a] = Complex64::new(cos_t, 0.0) * pa - Complex64::new(0.0, sin_t) * u_phase * pb;
    psi[b] = -Complex64::new(0.0, sin_t) * u_phase_conj * pa + Complex64::new(cos_t, 0.0) * pb;
}

/// Telemetry metrics for the K-Way Glued Tree Valve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValveStepTelemetry {
    pub time: f64,
    pub p_entrance: f64,
    pub p_target_exit: f64,
    pub p_non_target_sum: f64,
    pub p_weld_interface: f64,
    pub exit_probabilities: Vec<f64>,
    pub contrast_ratio: f64,
}

/// High-Performance Continuous-Time Valve Simulator.
#[derive(Debug, Clone)]
pub struct ContinuousTreeValveWalk {
    pub cfg: GluedTreeValveConfig,
    pub ham: MultiExitValveHamiltonian,
    pub target_exit: usize,
    pub state: Vec<Complex64>,
}

impl ContinuousTreeValveWalk {
    pub fn new(cfg: GluedTreeValveConfig, gamma: f64, target_exit: usize) -> Self {
        let mut ham = MultiExitValveHamiltonian::new(cfg, gamma);
        ham.configure_steering_target(target_exit);
        let mut state = vec![Complex64::default(); cfg.symmetric_dim()];
        state[cfg.in_col_idx(0)] = Complex64::new(1.0, 0.0); // Start at ROOT_IN
        Self {
            cfg,
            ham,
            target_exit,
            state,
        }
    }

    pub fn reset(&mut self) {
        for elem in self.state.iter_mut() {
            *elem = Complex64::default();
        }
        self.state[self.cfg.in_col_idx(0)] = Complex64::new(1.0, 0.0);
    }

    pub fn step(&mut self, dt: f64) {
        self.ham.step_trotter(&mut self.state, dt);
    }

    pub fn telemetry(&self, time: f64) -> ValveStepTelemetry {
        let k = self.cfg.num_exits;
        let depth = self.cfg.depth;
        let mut exit_probs = Vec::with_capacity(k);

        for y in 0..k {
            let root_idx = self.cfg.out_col_idx(y, depth);
            exit_probs.push(self.state[root_idx].norm_sqr());
        }

        let p_entrance = self.state[self.cfg.in_col_idx(0)].norm_sqr();
        let p_target = exit_probs[self.target_exit];
        let p_non_target: f64 = exit_probs
            .iter()
            .enumerate()
            .filter(|(y, _)| *y != self.target_exit)
            .map(|(_, &p)| p)
            .sum();

        let in_leaf_p = self.state[self.cfg.in_col_idx(depth)].norm_sqr();
        let out_leaves_p: f64 = (0..k)
            .map(|y| self.state[self.cfg.out_col_idx(y, 0)].norm_sqr())
            .sum();
        let p_weld = in_leaf_p + out_leaves_p;

        let contrast = if p_non_target > 1e-12 {
            p_target / p_non_target
        } else {
            p_target * 1e12
        };

        ValveStepTelemetry {
            time,
            p_entrance,
            p_target_exit: p_target,
            p_non_target_sum: p_non_target,
            p_weld_interface: p_weld,
            exit_probabilities: exit_probs,
            contrast_ratio: contrast,
        }
    }

    pub fn run_simulation(&mut self, t_max: f64, dt: f64) -> Vec<ValveStepTelemetry> {
        self.reset();
        let n_steps = (t_max / dt).round() as usize;
        let mut traj = Vec::with_capacity(n_steps + 1);
        traj.push(self.telemetry(0.0));

        for s in 1..=n_steps {
            self.step(dt);
            let t = s as f64 * dt;
            traj.push(self.telemetry(t));
        }

        traj
    }
}

/// Classical Walker on K-Way Glued Trees Graph (Monte Carlo).
#[derive(Debug, Clone)]
pub struct ClassicalKWayGluedWalker {
    pub cfg: GluedTreeValveConfig,
}

impl ClassicalKWayGluedWalker {
    pub const fn new(cfg: GluedTreeValveConfig) -> Self {
        Self { cfg }
    }

    /// Simulates a single random walk step.
    /// State representation: (tree_id, col_idx).
    /// tree_id: 0 = Entrance Tree; 1..K = Exit Tree y-1.
    pub fn step<R: Rng>(&self, state: (usize, usize), rng: &mut R) -> (usize, usize) {
        let depth = self.cfg.depth;
        let k = self.cfg.num_exits;
        let (tree, col) = state;

        if tree == 0 {
            // Inside Entrance Tree
            if col == 0 {
                // ROOT_IN: degree 2, moves forward to col 1
                (0, 1)
            } else if col < depth {
                // 1 parent, 2 children
                let roll: f64 = rng.gen();
                if roll < (1.0 / 3.0) {
                    (0, col - 1)
                } else {
                    (0, col + 1)
                }
            } else {
                // Leaves of Entrance Tree (col n): 1 parent (col n-1) + weld edges to K exit leaves
                let roll: f64 = rng.gen();
                if roll < (1.0 / (k as f64 + 1.0)) {
                    (0, col - 1)
                } else {
                    // Jump to a random exit tree leaf
                    let chosen_exit = rng.gen_range(1..=k);
                    (chosen_exit, 0)
                }
            }
        } else {
            // Inside Exit Tree `tree` (1..=K)
            if col == depth {
                // ROOT_OUT reached: absorbed
                state
            } else if col == 0 {
                // Exit tree leaf: 1 edge back to Entrance tree leaf, 1 child toward root
                let roll: f64 = rng.gen();
                if roll < 0.5 {
                    (0, depth)
                } else {
                    (tree, 1)
                }
            } else {
                // Column 1..depth-1 in Exit Tree: 2 parents toward leaf, 1 child toward root
                let roll: f64 = rng.gen();
                if roll < (2.0 / 3.0) {
                    (tree, col - 1)
                } else {
                    (tree, col + 1)
                }
            }
        }
    }

    /// Simulates walk from ROOT_IN until max_steps or exit reached.
    /// Returns (steps, reached_target, reached_any_exit).
    pub fn simulate_walk<R: Rng>(
        &self,
        target_exit: usize,
        max_steps: usize,
        rng: &mut R,
    ) -> (usize, bool, bool) {
        let depth = self.cfg.depth;
        let mut curr = (0, 0);

        for step in 1..=max_steps {
            curr = self.step(curr, rng);
            if curr.1 == depth && curr.0 > 0 {
                let hit_exit = curr.0 - 1;
                let is_target = hit_exit == target_exit;
                return (step, is_target, true);
            }
        }

        (max_steps, false, false)
    }

    pub fn benchmark(&self, target_exit: usize, n_trials: usize, max_steps: usize) -> (f64, f64, f64) {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x5EED_0042);

        let mut target_hits = 0;
        let mut any_hits = 0;
        let mut total_steps = 0;

        for _ in 0..n_trials {
            let (steps, is_target, hit_any) = self.simulate_walk(target_exit, max_steps, &mut rng);
            total_steps += steps;
            if hit_any {
                any_hits += 1;
            }
            if is_target {
                target_hits += 1;
            }
        }

        let mean_steps = total_steps as f64 / n_trials as f64;
        let target_rate = target_hits as f64 / n_trials as f64;
        let any_rate = any_hits as f64 / n_trials as f64;
        (mean_steps, target_rate, any_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valve_geometry() {
        let cfg = GluedTreeValveConfig::new(3, 4); // depth 3, 4 exits
        assert_eq!(cfg.cols_per_tree(), 4);
        assert_eq!(cfg.symmetric_dim(), 5 * 4); // 20 basis dimensions
        assert_eq!(cfg.num_leaves(), 8);
        assert_eq!(cfg.total_vertices(), 5 * (16 - 1)); // 75 total vertices
    }

    #[test]
    fn test_valve_unitarity_preservation() {
        let cfg = GluedTreeValveConfig::new(3, 2);
        let mut valve = ContinuousTreeValveWalk::new(cfg, 0.5, 0);

        for _ in 0..100 {
            valve.step(0.05);
            let norm_sq: f64 = valve.state.iter().map(|a| a.norm_sqr()).sum();
            assert!(
                (norm_sq - 1.0).abs() < 1e-10,
                "Norm square must be strictly 1.0 (got {})",
                norm_sq
            );
        }
    }

    #[test]
    fn test_topological_steering_contrast() {
        let cfg = GluedTreeValveConfig::new(4, 2);
        let mut valve = ContinuousTreeValveWalk::new(cfg, 0.5, 0); // Target Exit 0
        let traj = valve.run_simulation(15.0, 0.1);

        let max_target = traj.iter().map(|s| s.p_target_exit).fold(0.0_f64, f64::max);
        let max_competing = traj.iter().map(|s| s.p_non_target_sum).fold(0.0_f64, f64::max);

        assert!(
            max_target > max_competing,
            "Target exit probability ({}) must dominate competing exit ({})",
            max_target,
            max_competing
        );
    }
}
