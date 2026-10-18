// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Glued Trees Quantum Walk Engine (Childs et al. Welded-Tree Traversal)
//!
//! Provides an exponential quantum speedup over any classical algorithm on a graph
//! traversal problem.
//!
//! ## Topology:
//! Two balanced binary (or d-ary) trees of depth $n$ connected at their leaves by a
//! random cycle (the "weld").
//! Total vertices: $N = 2 \cdot (2^{n+1} - 1) = 2^{n+2} - 2$.
//!
//! ## Theoretical Foundations:
//! - **Classical Bottleneck:** At each step from level $j < n$, a classical random walk
//!   moves forward to level $j+1$ with probability $2/3$ and backward with probability $1/3$.
//!   Upon hitting the weld ($j=n$), it becomes trapped in an exponentially large layer
//!   ($2^n$ leaves). Classical hitting time to ROOT_B is $\Omega(2^{n/2})$.
//! - **Quantum Speedup:** Continuous-Time Quantum Walk (CTQW) preserves the column-symmetric
//!   subspace:
//!   $$|\text{col}_j\rangle = \frac{1}{\sqrt{N_j}} \sum_{v \in \text{col } j} |v\rangle$$
//!   Destructive interference eliminates all paths outside the symmetric subspace,
//!   reducing the exponential graph to a 1D chain of length $2n+2$. The quantum walker
//!   crosses the weld and hits ROOT_B in $O(\text{poly}(n))$ time.

use num_complex::Complex64;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Configuration parameters for the Glued Trees Graph and Quantum Walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GluedTreesConfig {
    /// Depth of each balanced binary tree (n >= 1).
    pub depth: usize,
    /// Branching factor (default = 2 for binary trees).
    pub branching: usize,
}

impl GluedTreesConfig {
    /// Constructs a validated Glued Trees configuration.
    pub const fn new(depth: usize) -> Self {
        assert!(depth >= 1, "Tree depth must be at least 1");
        Self {
            depth,
            branching: 2,
        }
    }

    /// Number of columns in the glued graph: 0 to 2n + 1 (total 2n + 2 columns).
    #[inline(always)]
    pub const fn num_columns(&self) -> usize {
        2 * self.depth + 2
    }

    /// Number of vertices at column j (0 <= j <= 2n + 1).
    pub fn column_size(&self, col: usize) -> usize {
        assert!(col < self.num_columns(), "Column index out of bounds");
        if col <= self.depth {
            1 << col
        } else {
            1 << (2 * self.depth + 1 - col)
        }
    }

    /// Total number of vertices in the full glued graph: 2 * (2^{n+1} - 1).
    pub fn total_vertices(&self) -> usize {
        2 * ((1 << (self.depth + 1)) - 1)
    }

    /// Number of leaves per tree: 2^n.
    #[inline(always)]
    pub fn num_leaves(&self) -> usize {
        1 << self.depth
    }
}

/// Tridiagonal Hamiltonian in the Column-Symmetric Subspace (dimension 2n + 2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymmetricChainHamiltonian {
    /// Dimension of the symmetric subspace (2n + 2).
    pub dim: usize,
    /// Coupling parameter gamma (hopping amplitude).
    pub gamma: f64,
    /// Off-diagonal hopping elements: coupling between col j and col j+1.
    pub couplings: Vec<f64>,
}

impl SymmetricChainHamiltonian {
    /// Constructs the column-symmetric tridiagonal Hamiltonian for given tree depth and gamma.
    pub fn new(cfg: &GluedTreesConfig, gamma: f64) -> Self {
        let dim = cfg.num_columns();
        let mut couplings = Vec::with_capacity(dim.saturating_sub(1));

        let depth = cfg.depth;
        let sqrt2 = 2.0_f64.sqrt();

        for j in 0..(dim - 1) {
            let hop = if j < depth {
                // Forward inside Tree 1: root to weld
                sqrt2 * gamma
            } else if j == depth {
                // Crossing the weld: col n to col n+1 (each leaf connected to 2 leaves)
                2.0 * gamma
            } else {
                // Forward inside Tree 2: weld to exit root
                sqrt2 * gamma
            };
            couplings.push(hop);
        }

        Self {
            dim,
            gamma,
            couplings,
        }
    }

    /// Computes H |psi> for a state in the symmetric column basis.
    pub fn apply(&self, psi: &[Complex64], out: &mut [Complex64]) {
        assert_eq!(psi.len(), self.dim);
        assert_eq!(out.len(), self.dim);

        for elem in out.iter_mut() {
            *elem = Complex64::default();
        }

        for j in 0..(self.dim - 1) {
            let h_val = self.couplings[j];
            out[j] += Complex64::new(h_val, 0.0) * psi[j + 1];
            out[j + 1] += Complex64::new(h_val, 0.0) * psi[j];
        }
    }

    /// Single time-step evolution via second-order Suzuki-Trotter step:
    /// exp(-i H dt) |psi>.
    pub fn step_trotter(&self, psi: &mut [Complex64], dt: f64) {
        let half_dt = dt * 0.5;

        // Pass 1: Even bonds (0-1, 2-3, 4-5, ...) with half step
        for j in (0..(self.dim - 1)).step_by(2) {
            evolve_two_level(psi, j, j + 1, self.couplings[j], half_dt);
        }

        // Pass 2: Odd bonds (1-2, 3-4, 5-6, ...) with full step
        for j in (1..(self.dim - 1)).step_by(2) {
            evolve_two_level(psi, j, j + 1, self.couplings[j], dt);
        }

        // Pass 3: Even bonds with half step
        for j in (0..(self.dim - 1)).step_by(2) {
            evolve_two_level(psi, j, j + 1, self.couplings[j], half_dt);
        }
    }
}

/// Local 2-level unitary time-evolution: exp(-i * [0, h; h, 0] * dt).
#[inline(always)]
fn evolve_two_level(psi: &mut [Complex64], a: usize, b: usize, h: f64, dt: f64) {
    let angle = h * dt;
    let cos_t = angle.cos();
    let sin_t = angle.sin();
    let i_sin_t = Complex64::new(0.0, -sin_t);

    let psi_a = psi[a];
    let psi_b = psi[b];

    psi[a] = Complex64::new(cos_t, 0.0) * psi_a + i_sin_t * psi_b;
    psi[b] = i_sin_t * psi_a + Complex64::new(cos_t, 0.0) * psi_b;
}

/// Classical Walker on Glued Trees Graph (Monte Carlo Simulator).
#[derive(Debug, Clone)]
pub struct ClassicalGluedWalker {
    pub cfg: GluedTreesConfig,
}

impl ClassicalGluedWalker {
    pub const fn new(cfg: GluedTreesConfig) -> Self {
        Self { cfg }
    }

    /// Simulates a single classical random walk step from current column `col`.
    /// Returns the new column after taking a step along one of the available edges.
    pub fn step<R: Rng>(&self, col: usize, rng: &mut R) -> usize {
        let depth = self.cfg.depth;
        let exit_col = 2 * depth + 1;

        if col == 0 {
            // Entrance Root: degree 2, both edges go to col 1
            1
        } else if col == exit_col {
            // Exit Root: absorbed / reached
            exit_col
        } else if col < depth {
            // Inside Tree 1: 1 parent edge (col - 1), 2 children edges (col + 1)
            let roll: f64 = rng.gen();
            if roll < (1.0 / 3.0) {
                col - 1
            } else {
                col + 1
            }
        } else if col == depth {
            // Weld layer Tree 1 leaves: 1 parent (col - 1), 2 weld edges to Tree 2 leaves (col + 1)
            let roll: f64 = rng.gen();
            if roll < (1.0 / 3.0) {
                col - 1
            } else {
                col + 1
            }
        } else if col == depth + 1 {
            // Weld layer Tree 2 leaves: 2 weld edges to Tree 1 (col - 1), 1 child toward exit (col + 1)
            let roll: f64 = rng.gen();
            if roll < (2.0 / 3.0) {
                col - 1
            } else {
                col + 1
            }
        } else {
            // Inside Tree 2: 2 parents toward weld (col - 1), 1 child toward exit (col + 1)
            let roll: f64 = rng.gen();
            if roll < (2.0 / 3.0) {
                col - 1
            } else {
                col + 1
            }
        }
    }

    /// Simulates classical walk starting at ROOT_A (col 0) up to `max_steps`.
    /// Returns (steps_taken, reached_exit).
    pub fn simulate_walk<R: Rng>(&self, max_steps: usize, rng: &mut R) -> (usize, bool) {
        let exit_col = 2 * self.cfg.depth + 1;
        let mut current_col = 0;

        for step in 1..=max_steps {
            current_col = self.step(current_col, rng);
            if current_col == exit_col {
                return (step, true);
            }
        }

        (max_steps, false)
    }

    /// Estimates classical hitting time and success probability over `n_trials`.
    pub fn benchmark_classical(&self, n_trials: usize, max_steps: usize) -> (f64, f64) {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x5EED_0042);

        let mut successes = 0;
        let mut total_steps = 0;

        for _ in 0..n_trials {
            let (steps, reached) = self.simulate_walk(max_steps, &mut rng);
            total_steps += steps;
            if reached {
                successes += 1;
            }
        }

        let mean_steps = total_steps as f64 / n_trials as f64;
        let success_rate = successes as f64 / n_trials as f64;
        (mean_steps, success_rate)
    }
}

/// Telemetry metrics for Continuous-Time Quantum Walk (CTQW) on Glued Trees.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumWalkTelemetry {
    /// Time point t.
    pub time: f64,
    /// Probability at Entrance Root (col 0): |<col_0 | psi(t)>|^2.
    pub p_entrance: f64,
    /// Probability in Weld region (cols n and n+1).
    pub p_weld: f64,
    /// Probability at Exit Root (col 2n + 1): |<col_{2n+1} | psi(t)>|^2.
    pub p_exit: f64,
    /// Column probability distribution across all columns 0 .. 2n + 1.
    pub col_probabilities: Vec<f64>,
}

/// Continuous-Time Quantum Walk Simulator on Glued Trees.
#[derive(Debug, Clone)]
pub struct ContinuousQuantumWalk {
    pub cfg: GluedTreesConfig,
    pub ham: SymmetricChainHamiltonian,
    pub state: Vec<Complex64>,
}

impl ContinuousQuantumWalk {
    /// Initializes quantum walk with state localized at Entrance Root (column 0).
    pub fn new(cfg: GluedTreesConfig, gamma: f64) -> Self {
        let ham = SymmetricChainHamiltonian::new(&cfg, gamma);
        let mut state = vec![Complex64::default(); cfg.num_columns()];
        state[0] = Complex64::new(1.0, 0.0); // |col_0> = ROOT_A
        Self { cfg, ham, state }
    }

    /// Resets state to Entrance Root |col_0>.
    pub fn reset(&mut self) {
        for elem in self.state.iter_mut() {
            *elem = Complex64::default();
        }
        self.state[0] = Complex64::new(1.0, 0.0);
    }

    /// Advances the quantum walk by duration `dt` using Suzuki-Trotter evolution.
    pub fn step(&mut self, dt: f64) {
        self.ham.step_trotter(&mut self.state, dt);
    }

    /// Injects a phase defect or damping into one specific column to test router steering.
    pub fn apply_column_phase_perturbation(&mut self, col: usize, phase: f64) {
        if col < self.state.len() {
            let u_phase = Complex64::new(phase.cos(), phase.sin());
            self.state[col] *= u_phase;
        }
    }

    /// Evaluates current quantum telemetry and column distribution.
    pub fn telemetry(&self, time: f64) -> QuantumWalkTelemetry {
        let probs: Vec<f64> = self.state.iter().map(|a| a.norm_sqr()).collect();
        let depth = self.cfg.depth;
        let exit_idx = 2 * depth + 1;

        let p_entrance = probs[0];
        let p_weld = probs[depth] + probs[depth + 1];
        let p_exit = probs[exit_idx];

        QuantumWalkTelemetry {
            time,
            p_entrance,
            p_weld,
            p_exit,
            col_probabilities: probs,
        }
    }

    /// Simulates CTQW from t = 0 to t = t_max and records spatiotemporal trajectory.
    pub fn run_simulation(&mut self, t_max: f64, dt: f64) -> Vec<QuantumWalkTelemetry> {
        self.reset();
        let n_steps = (t_max / dt).round() as usize;
        let mut trajectory = Vec::with_capacity(n_steps + 1);

        trajectory.push(self.telemetry(0.0));

        for s in 1..=n_steps {
            self.step(dt);
            let t = s as f64 * dt;
            trajectory.push(self.telemetry(t));
        }

        trajectory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glued_trees_geometry() {
        let cfg = GluedTreesConfig::new(3);
        assert_eq!(cfg.num_columns(), 8); // 0..7
        assert_eq!(cfg.column_size(0), 1);
        assert_eq!(cfg.column_size(1), 2);
        assert_eq!(cfg.column_size(2), 4);
        assert_eq!(cfg.column_size(3), 8); // Weld Tree 1
        assert_eq!(cfg.column_size(4), 8); // Weld Tree 2
        assert_eq!(cfg.column_size(5), 4);
        assert_eq!(cfg.column_size(6), 2);
        assert_eq!(cfg.column_size(7), 1); // Exit Root
        assert_eq!(cfg.total_vertices(), 2 * (16 - 1)); // 30 vertices
    }

    #[test]
    fn test_ctqw_unitarity_preservation() {
        let cfg = GluedTreesConfig::new(4);
        let mut qw = ContinuousQuantumWalk::new(cfg, 0.5);

        for _ in 0..100 {
            qw.step(0.05);
            let norm_sq: f64 = qw.state.iter().map(|a| a.norm_sqr()).sum();
            assert!(
                (norm_sq - 1.0).abs() < 1e-10,
                "Norm square must remain 1.0 (got {})",
                norm_sq
            );
        }
    }

    #[test]
    fn test_quantum_hits_exit_in_polynomial_time() {
        let cfg = GluedTreesConfig::new(4);
        let mut qw = ContinuousQuantumWalk::new(cfg, 0.5);
        let trajectory = qw.run_simulation(20.0, 0.1);

        let max_exit_prob = trajectory
            .iter()
            .map(|snap| snap.p_exit)
            .fold(0.0_f64, f64::max);

        assert!(
            max_exit_prob > 0.05,
            "Quantum walker must reach exit with significant probability (got {})",
            max_exit_prob
        );
    }
}
