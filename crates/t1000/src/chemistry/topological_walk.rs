// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Continuous-Time Topological Quantum Walk Molecular Screener
//!
//! Evolves molecular electronic wavepackets under continuous graph Hamiltonians
//! constructed directly from ab initio Coulomb interaction matrices and Mulliken charges:
//!
//! $$H = -\gamma L_{\text{mol}} + \sum_{i=1}^N \epsilon_i |i\rangle\langle i|$$
//!
//! Where $L_{\text{mol}} = D - A$ is the molecular graph Laplacian derived from
//! inter-nuclear Coulomb potentials, and $\epsilon_i = \kappa \cdot Z_i (1 + q_i)$ are
//! atomic on-site potentials. Evaluates ballistic hitting times, wavepacket routing contrast,
//! and continuous Zeno projective sentry stabilization without discrete gate Trotterization.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use super::qm9::Qm9Record;
use crate::simd::eigensystem::solve_symmetric_eigensystem;

/// Real-symmetric graph Laplacian and continuous Hamiltonian representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MolecularLaplacian {
    /// Dimension (atom count N).
    pub dim: usize,
    /// Molecular Coulomb adjacency matrix A (N x N, row-major).
    pub adjacency: Vec<f64>,
    /// Degree vector D_ii = sum_j A_ij.
    pub degrees: Vec<f64>,
    /// Graph Laplacian matrix L = D - A (N x N, row-major).
    pub laplacian: Vec<f64>,
    /// Continuous molecular Hamiltonian H = -gamma * L + diag(epsilon) (N x N).
    pub hamiltonian: Vec<f64>,
    /// Sorted eigenvalues of H (descending).
    pub eigenvalues: Vec<f64>,
    /// Corresponding orthonormal eigenvectors (column vectors).
    pub eigenvectors: Vec<Vec<f64>>,
    /// Core atom index (highest nuclear charge / primary backbone carbon).
    pub core_atom_idx: usize,
    /// Frontier reactive site index (electrophilic or heteroatom valence site).
    pub frontier_site_idx: usize,
}

impl MolecularLaplacian {
    /// Constructs a molecular Laplacian and continuous Hamiltonian from a QM9 record.
    pub fn from_qm9_record(
        record: &Qm9Record,
        gamma: f64,
        potential_scale: f64,
    ) -> Result<Self, String> {
        let n = record.atoms.len();
        if n < 2 {
            return Err("Molecule must contain at least 2 atoms".to_string());
        }

        let z_vals: Vec<f64> = record
            .atoms
            .iter()
            .map(|a| element_symbol_to_z(&a.symbol))
            .collect();

        // 1. Build Coulomb adjacency matrix A_ij = (Z_i * Z_j) / ||r_i - r_j||
        let mut adj = vec![0.0; n * n];
        let mut deg = vec![0.0; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let dx = record.atoms[i].coords[0] - record.atoms[j].coords[0];
                let dy = record.atoms[i].coords[1] - record.atoms[j].coords[1];
                let dz = record.atoms[i].coords[2] - record.atoms[j].coords[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(0.1);
                let coupling = (z_vals[i] * z_vals[j]) / dist;

                adj[i * n + j] = coupling;
                adj[j * n + i] = coupling;
                deg[i] += coupling;
                deg[j] += coupling;
            }
        }

        // 2. Build graph Laplacian L = D - A
        let mut lap = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                lap[i * n + j] = if i == j {
                    deg[i] - adj[i * n + j]
                } else {
                    -adj[i * n + j]
                };
            }
        }

        // 3. Build Hamiltonian H = -gamma * L + diag(epsilon_i)
        // epsilon_i = potential_scale * Z_i * (1.0 + q_i)
        let mut ham = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let base = -gamma * lap[i * n + j];
                ham[i * n + j] = if i == j {
                    let eps = potential_scale * z_vals[i] * (1.0 + record.atoms[i].mulliken_charge);
                    base + eps
                } else {
                    base
                };
            }
        }

        // 4. Solve real-symmetric eigensystem
        let eigensystem = solve_symmetric_eigensystem(&ham, n, true)?;
        let evecs = eigensystem.eigenvectors.ok_or_else(|| {
            "Failed to compute eigenvectors for molecular Hamiltonian".to_string()
        })?;

        // 5. Select core atom and frontier site
        let frontier_idx = select_frontier_site(&z_vals, &record.atoms);
        let core_idx = select_core_atom(&z_vals, frontier_idx);

        Ok(Self {
            dim: n,
            adjacency: adj,
            degrees: deg,
            laplacian: lap,
            hamiltonian: ham,
            eigenvalues: eigensystem.eigenvalues,
            eigenvectors: evecs,
            core_atom_idx: core_idx,
            frontier_site_idx: frontier_idx,
        })
    }
}

/// Continuous-Time Topological Quantum Walk Engine on Molecular Graphs.
#[derive(Debug, Clone)]
pub struct ContinuousMolecularWalk {
    /// Underlying molecular graph Laplacian and Hamiltonian.
    pub laplacian: MolecularLaplacian,
}

impl ContinuousMolecularWalk {
    /// Constructs a continuous molecular walk from a solved Laplacian.
    pub fn new(laplacian: MolecularLaplacian) -> Self {
        Self { laplacian }
    }

    /// Exact analytical continuous-time wavepacket evolution:
    /// |psi(t)> = sum_k exp(-i * lambda_k * t) * (v_k^T psi(0)) * v_k
    pub fn evolve_analytic(&self, psi0: &[Complex64], t: f64) -> Vec<Complex64> {
        let n = self.laplacian.dim;
        let mut out = vec![Complex64::new(0.0, 0.0); n];

        for (k, val) in self.laplacian.eigenvalues.iter().enumerate().take(n) {
            let v_k = &self.laplacian.eigenvectors[k];

            // Compute projection coefficient c_k = <v_k | psi(0)>
            let mut c_k = Complex64::new(0.0, 0.0);
            for j in 0..n {
                c_k += psi0[j] * v_k[j];
            }

            // Phase factor exp(-i * lambda_k * t)
            let phase = -val * t;
            let phase_factor = Complex64::new(phase.cos(), phase.sin());
            let coeff = c_k * phase_factor;

            for j in 0..n {
                out[j] += coeff * v_k[j];
            }
        }

        out
    }

    /// Evolves under intermediate Weld Zeno projective sentries.
    ///
    /// Periodically applies projection operator onto physically bound valence sites,
    /// suppressing phase drift and unphysical radical dissociation.
    pub fn evolve_zeno_stepped(
        &self,
        psi0: &[Complex64],
        total_time: f64,
        num_zeno_steps: usize,
        bound_mask: &[bool],
    ) -> (Vec<Complex64>, f64) {
        let n = self.laplacian.dim;
        let dt = total_time / (num_zeno_steps.max(1) as f64);
        let mut current_state = psi0.to_vec();
        let mut cumulative_survival = 1.0;

        for _ in 0..num_zeno_steps {
            // 1. Coherent unitary evolution over duration dt
            let evolved = self.evolve_analytic(&current_state, dt);

            // 2. Continuous Zeno projection onto bound valence subspace
            let mut projected_norm_sq = 0.0;
            let mut projected = vec![Complex64::new(0.0, 0.0); n];

            for j in 0..n {
                if j < bound_mask.len() && bound_mask[j] {
                    projected[j] = evolved[j];
                    projected_norm_sq += evolved[j].norm_sqr();
                }
            }

            cumulative_survival *= projected_norm_sq.min(1.0);

            // Renormalize if non-zero
            if projected_norm_sq > 1e-12 {
                let inv_norm = 1.0 / projected_norm_sq.sqrt();
                for j in 0..n {
                    projected[j] *= inv_norm;
                }
                current_state = projected;
            } else {
                current_state = evolved;
            }
        }

        (current_state, cumulative_survival)
    }

    /// Evaluates ballistic hitting probability curve P_target(t) across a discrete time mesh.
    pub fn compute_ballistic_hitting_curve(
        &self,
        source_idx: usize,
        target_idx: usize,
        times: &[f64],
    ) -> Vec<f64> {
        let n = self.laplacian.dim;
        let mut psi0 = vec![Complex64::new(0.0, 0.0); n];
        if source_idx < n {
            psi0[source_idx] = Complex64::new(1.0, 0.0);
        }

        times
            .iter()
            .map(|&t| {
                let state = self.evolve_analytic(&psi0, t);
                if target_idx < n {
                    state[target_idx].norm_sqr()
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Computes routing contrast: ratio of target exit probability to max off-target probability.
    pub fn routing_contrast(&self, state: &[Complex64], target_idx: usize) -> f64 {
        let target_prob = if target_idx < state.len() {
            state[target_idx].norm_sqr()
        } else {
            0.0
        };

        let mut max_other = 1e-9;
        for (i, item) in state.iter().enumerate() {
            if i != target_idx {
                let p = item.norm_sqr();
                if p > max_other {
                    max_other = p;
                }
            }
        }

        target_prob / max_other
    }
}

/// Classical Markov Random Walk baseline for comparative hitting time analysis.
#[derive(Debug, Clone)]
pub struct ClassicalRandomWalk {
    dim: usize,
    transition_matrix: Vec<f64>,
}

impl ClassicalRandomWalk {
    /// Constructs a classical random walk engine from a molecular Laplacian.
    pub fn from_laplacian(laplacian: &MolecularLaplacian) -> Self {
        let n = laplacian.dim;
        let mut trans = vec![0.0; n * n];

        for i in 0..n {
            let deg_i = laplacian.degrees[i];
            if deg_i > 1e-9 {
                let inv_d = 1.0 / deg_i;
                for j in 0..n {
                    trans[i * n + j] = laplacian.adjacency[i * n + j] * inv_d;
                }
            }
        }

        Self {
            dim: n,
            transition_matrix: trans,
        }
    }

    /// Diffuses classical probability distribution over discrete steps.
    pub fn diffuse(&self, p0: &[f64], steps: usize) -> Vec<f64> {
        let n = self.dim;
        let mut current = p0.to_vec();
        let mut next = vec![0.0; n];

        for _ in 0..steps {
            for j in 0..n {
                let mut sum = 0.0;
                for i in 0..n {
                    sum += current[i] * self.transition_matrix[i * n + j];
                }
                next[j] = sum;
            }
            current.copy_from_slice(&next);
        }

        current
    }

    /// Computes classical hitting probability at target site after given steps.
    pub fn hitting_prob(&self, source: usize, target: usize, steps: usize) -> f64 {
        let mut p0 = vec![0.0; self.dim];
        if source < self.dim {
            p0[source] = 1.0;
        }
        let pf = self.diffuse(&p0, steps);
        if target < self.dim {
            pf[target]
        } else {
            0.0
        }
    }
}

// ----------------------------------------------------------------------------
// Helper Functions
// ----------------------------------------------------------------------------

#[inline]
fn element_symbol_to_z(symbol: &str) -> f64 {
    match symbol.trim() {
        "H" => 1.0,
        "He" => 2.0,
        "Li" => 3.0,
        "Be" => 4.0,
        "B" => 5.0,
        "C" => 6.0,
        "N" => 7.0,
        "O" => 8.0,
        "F" => 9.0,
        "Ne" => 10.0,
        _ => 6.0, // Default to Carbon
    }
}

#[inline]
fn select_core_atom(z_vals: &[f64], frontier_idx: usize) -> usize {
    let mut best_idx = 0;
    let mut best_score = f64::NEG_INFINITY;
    for (i, &z) in z_vals.iter().enumerate() {
        if i == frontier_idx {
            continue;
        }
        let score = if (z - 6.0).abs() < 1e-3 {
            10.0 // Carbon preferred as backbone core
        } else if z > 1.0 {
            5.0 // Other heavy atom
        } else {
            0.0 // Hydrogen
        };
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    best_idx
}

#[inline]
fn select_frontier_site(z_vals: &[f64], atoms: &[super::qm9::Qm9Atom]) -> usize {
    // Prefer heteroatoms (O, N, F) with negative Mulliken charge as reactive frontier
    let mut best_idx = 0;
    let mut best_score = f64::NEG_INFINITY;

    for (i, atom) in atoms.iter().enumerate() {
        let z = z_vals[i];
        let is_hetero = (z - 7.0).abs() < 1e-3 || (z - 8.0).abs() < 1e-3 || (z - 9.0).abs() < 1e-3;
        let score = if is_hetero {
            10.0 - atom.mulliken_charge
        } else if (z - 6.0).abs() < 1e-3 {
            2.0 - atom.mulliken_charge
        } else {
            -1.0
        };

        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    best_idx
}
