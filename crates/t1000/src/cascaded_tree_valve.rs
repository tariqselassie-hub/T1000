// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Cascaded Multi-Tier Topological Glued-Tree Valve Engine
//!
//! Implements a hierarchical multi-tier quantum routing network:
//! - Tier 1 (Coarse Routing): Entrance Tree of depth $n_1$ welded to $K_1$ Intermediate Trees.
//! - Tier 2 (Fine Routing): $K_1$ Intermediate Trees branch into $K_1 \times K_2$ Final Exit Trees
//!   of depth $n_2$.
//!
//! Demonstrates ballistic cascade through multiple sequential weld bottlenecks without
//! intermediate measurement or loss of coherence.

use num_complex::Complex64;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Configuration for the Cascaded Multi-Tier Tree Valve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CascadedValveConfig {
    /// Depth of Tier-1 trees (n1 >= 1).
    pub depth1: usize,
    /// Depth of Tier-2 trees (n2 >= 1).
    pub depth2: usize,
    /// Number of Tier-1 coarse branches (K1 >= 2).
    pub k1: usize,
    /// Number of Tier-2 fine branches per coarse branch (K2 >= 2).
    pub k2: usize,
}

impl CascadedValveConfig {
    pub const fn new(depth1: usize, depth2: usize, k1: usize, k2: usize) -> Self {
        assert!(depth1 >= 1, "Tier 1 depth must be >= 1");
        assert!(depth2 >= 1, "Tier 2 depth must be >= 1");
        assert!(k1 >= 2, "K1 must be >= 2");
        assert!(k2 >= 2, "K2 must be >= 2");
        Self { depth1, depth2, k1, k2 }
    }

    /// Total number of final terminal exit roots: K1 * K2.
    #[inline(always)]
    pub const fn total_exits(&self) -> usize {
        self.k1 * self.k2
    }

    /// Dimension of the multi-tier column-symmetric invariant subspace:
    /// Tier 1 entrance: (depth1 + 1)
    /// Tier 1 intermediate branches: k1 * depth1
    /// Intermediate roots: k1
    /// Tier 2 entrance columns: k1 * depth2
    /// Tier 2 exit branches: k1 * k2 * (depth2 + 1)
    pub const fn symmetric_dim(&self) -> usize {
        (self.depth1 + 1)
            + self.k1 * self.depth1
            + self.k1
            + self.k1 * self.depth2
            + self.k1 * self.k2 * (self.depth2 + 1)
    }

    /// Total number of vertices across the full unreduced physical graph.
    pub fn total_vertices(&self) -> usize {
        let t1_in = (1 << (self.depth1 + 1)) - 1;
        let t1_mid = self.k1 * ((1 << (self.depth1 + 1)) - 2);
        let mid_roots = self.k1;
        let t2_in = self.k1 * ((1 << (self.depth2 + 1)) - 2);
        let t2_exits = self.k1 * self.k2 * ((1 << (self.depth2 + 1)) - 1);
        t1_in + t1_mid + mid_roots + t2_in + t2_exits
    }

    // Basis indexing helpers for the symmetric subspace:

    #[inline(always)]
    pub const fn t1_in_idx(&self, col: usize) -> usize {
        col
    }

    #[inline(always)]
    pub const fn t1_mid_idx(&self, branch_y: usize, col: usize) -> usize {
        (self.depth1 + 1) + branch_y * self.depth1 + col
    }

    #[inline(always)]
    pub const fn mid_root_idx(&self, branch_y: usize) -> usize {
        (self.depth1 + 1) + self.k1 * self.depth1 + branch_y
    }

    #[inline(always)]
    pub const fn t2_in_idx(&self, branch_y: usize, col: usize) -> usize {
        let base = (self.depth1 + 1) + self.k1 * self.depth1 + self.k1;
        base + branch_y * self.depth2 + col
    }

    #[inline(always)]
    pub const fn t2_exit_idx(&self, branch_y: usize, branch_z: usize, col: usize) -> usize {
        let base = (self.depth1 + 1) + self.k1 * self.depth1 + self.k1 + self.k1 * self.depth2;
        let exit_idx = branch_y * self.k2 + branch_z;
        base + exit_idx * (self.depth2 + 1) + col
    }

    #[inline(always)]
    pub const fn final_exit_root_idx(&self, branch_y: usize, branch_z: usize) -> usize {
        self.t2_exit_idx(branch_y, branch_z, self.depth2)
    }
}

/// Multi-Tier Hamiltonian for the Cascaded Glued-Tree Valve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadedValveHamiltonian {
    pub cfg: CascadedValveConfig,
    pub gamma: f64,
    /// Tier 1 weld coupling weights for each coarse branch y (0..K1).
    pub weld1_couplings: Vec<f64>,
    /// Tier 1 weld phase holonomies phi_1 for each coarse branch y (0..K1).
    pub weld1_phases: Vec<f64>,
    /// Tier 2 weld coupling weights for each branch (y, z) of length K1 * K2.
    pub weld2_couplings: Vec<f64>,
    /// Tier 2 weld phase holonomies phi_2 for each branch (y, z) of length K1 * K2.
    pub weld2_phases: Vec<f64>,
}

/// Static undirected coupling edge in the symmetric subspace Hamiltonian.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseCoupling {
    pub u: usize,
    pub v: usize,
    pub weight: Complex64,
}

impl SparseCoupling {
    #[inline(always)]
    pub fn new(u: usize, v: usize, weight: Complex64) -> Self {
        Self { u, v, weight }
    }
}

/// Static sparse edge list representing the tridiagonal block Hamiltonian.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseSymmetricHamiltonian {
    pub dim: usize,
    pub edges: Vec<SparseCoupling>,
}

impl SparseSymmetricHamiltonian {
    pub fn new(dim: usize, edges: Vec<SparseCoupling>) -> Self {
        Self { dim, edges }
    }

    /// Returns the Hilbert space dimension of the symmetric subspace.
    #[inline(always)]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the number of undirected couplings.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    /// Returns true if there are no edges.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// Computes H |psi> -> out via linear O(E) sparse edge accumulation.
    #[inline(always)]
    pub fn apply(&self, state: &[Complex64], out: &mut [Complex64]) {
        for elem in out[..self.dim].iter_mut() {
            *elem = Complex64::default();
        }
        for edge in &self.edges {
            let u = edge.u;
            let v = edge.v;
            let w = edge.weight;
            out[u] += w * state[v];
            out[v] += w.conj() * state[u];
        }
    }

    /// Evaluates H |psi> -> (out_real, out_imag) in Structure-of-Arrays (SoA) layout.
    #[inline(always)]
    pub fn apply_soa(
        &self,
        r_in: &[f64],
        i_in: &[f64],
        r_out: &mut [f64],
        i_out: &mut [f64],
    ) {
        r_out[..self.dim].fill(0.0);
        i_out[..self.dim].fill(0.0);

        for edge in &self.edges {
            let u = edge.u;
            let v = edge.v;
            let wr = edge.weight.re;
            let wi = edge.weight.im;

            let rv = r_in[v];
            let mv = i_in[v];
            let ru = r_in[u];
            let mu = i_in[u];

            // H_uv * psi_v = (wr + i wi) * (rv + i mv)
            r_out[u] += wr * rv - wi * mv;
            i_out[u] += wr * mv + wi * rv;

            // H_vu * psi_u = (wr - i wi) * (ru + i mu)
            r_out[v] += wr * ru + wi * mu;
            i_out[v] += wr * mu - wi * ru;
        }
    }

    /// Single Trotter step on statevector: |psi> <- |psi> - i * dt * H |psi>.
    #[inline(always)]
    pub fn step_state(&self, state: &[Complex64], next: &mut [Complex64], dt: f64) {
        self.apply(state, next);
        let i_dt = Complex64::new(0.0, dt);
        for r in 0..self.dim {
            next[r] = state[r] - i_dt * next[r];
        }

        // Re-normalize to enforce unitarity:
        let norm_sqr: f64 = next[..self.dim].iter().map(|z| z.norm_sqr()).sum();
        if norm_sqr > 1e-18 {
            let inv_norm = 1.0 / norm_sqr.sqrt();
            for val in next[..self.dim].iter_mut() {
                *val *= inv_norm;
            }
        }
    }

    /// Single Trotter step on statevector in SoA layout with AVX2 vectorization:
    /// |psi> <- |psi> - i * dt * H |psi>.
    #[inline(always)]
    pub fn step_state_soa(
        &self,
        r_in: &[f64],
        i_in: &[f64],
        r_out: &mut [f64],
        i_out: &mut [f64],
        r_scratch: &mut [f64],
        i_scratch: &mut [f64],
        dt: f64,
    ) {
        self.apply_soa(r_in, i_in, r_scratch, i_scratch);
        step_soa_avx2(self.dim, dt, r_in, i_in, r_scratch, i_scratch, r_out, i_out);
        crate::simd::normalize_kernel(&mut r_out[..self.dim], &mut i_out[..self.dim]);
    }
}

impl CascadedValveHamiltonian {
    pub fn new(cfg: CascadedValveConfig, gamma: f64) -> Self {
        let k1 = cfg.k1;
        let total_k = cfg.total_exits();
        Self {
            cfg,
            gamma,
            weld1_couplings: vec![2.0; k1],
            weld1_phases: vec![0.0; k1],
            weld2_couplings: vec![2.0; total_k],
            weld2_phases: vec![0.0; total_k],
        }
    }

    /// Sets up resonant transmission into (target_y, target_z) while extinguishing all other paths.
    pub fn configure_target_route(&mut self, target_y: usize, target_z: usize, detuning: f64) {
        assert!(target_y < self.cfg.k1, "Target coarse class out of bounds");
        assert!(target_z < self.cfg.k2, "Target fine class out of bounds");

        // Tier 1 configuration:
        for y in 0..self.cfg.k1 {
            if y == target_y {
                self.weld1_couplings[y] = 2.0;
                self.weld1_phases[y] = 0.0;
            } else {
                self.weld1_couplings[y] = 2.0 * detuning;
                self.weld1_phases[y] = std::f64::consts::PI;
            }
        }

        // Tier 2 configuration:
        for y in 0..self.cfg.k1 {
            for z in 0..self.cfg.k2 {
                let idx = y * self.cfg.k2 + z;
                if y == target_y && z == target_z {
                    self.weld2_couplings[idx] = 2.0;
                    self.weld2_phases[idx] = 0.0;
                } else {
                    self.weld2_couplings[idx] = 2.0 * detuning;
                    self.weld2_phases[idx] = std::f64::consts::PI;
                }
            }
        }
    }

    /// Constructs the static sparse edge list Hamiltonian in the symmetric subspace.
    pub fn build_symmetric_matrix(&self) -> SparseSymmetricHamiltonian {
        let dim = self.cfg.symmetric_dim();
        let g = self.gamma;
        let sqrt2 = std::f64::consts::SQRT_2;
        let mut edges = Vec::with_capacity(dim * 2);

        // 1. Tier 1 Entrance tree:
        for col in 0..self.cfg.depth1 {
            let u = self.cfg.t1_in_idx(col);
            let v = self.cfg.t1_in_idx(col + 1);
            edges.push(SparseCoupling::new(u, v, Complex64::new(-g * sqrt2, 0.0)));
        }

        // 2. Weld 1 boundary: between Entrance leaf and Intermediate leaf 0:
        let in_leaf = self.cfg.t1_in_idx(self.cfg.depth1);
        for y in 0..self.cfg.k1 {
            let mid_leaf = self.cfg.t1_mid_idx(y, 0);
            let w = self.weld1_couplings[y];
            let phi = self.weld1_phases[y];
            let hop = Complex64::from_polar(-g * w, phi);
            edges.push(SparseCoupling::new(in_leaf, mid_leaf, hop));
        }

        // 3. Tier 1 Intermediate tree internal edges:
        for y in 0..self.cfg.k1 {
            for col in 0..(self.cfg.depth1.saturating_sub(1)) {
                let u = self.cfg.t1_mid_idx(y, col);
                let v = self.cfg.t1_mid_idx(y, col + 1);
                edges.push(SparseCoupling::new(u, v, Complex64::new(-g * sqrt2, 0.0)));
            }
            // Edge into intermediate root:
            let last_mid = self.cfg.t1_mid_idx(y, self.cfg.depth1 - 1);
            let mid_root = self.cfg.mid_root_idx(y);
            edges.push(SparseCoupling::new(last_mid, mid_root, Complex64::new(-g * sqrt2, 0.0)));
        }

        // 4. Edge from Intermediate root into Tier 2 Entrance column 0:
        for y in 0..self.cfg.k1 {
            let mid_root = self.cfg.mid_root_idx(y);
            let t2_in_0 = self.cfg.t2_in_idx(y, 0);
            edges.push(SparseCoupling::new(mid_root, t2_in_0, Complex64::new(-g * sqrt2, 0.0)));

            // Tier 2 Entrance internal edges:
            for col in 0..(self.cfg.depth2.saturating_sub(1)) {
                let u = self.cfg.t2_in_idx(y, col);
                let v = self.cfg.t2_in_idx(y, col + 1);
                edges.push(SparseCoupling::new(u, v, Complex64::new(-g * sqrt2, 0.0)));
            }
        }

        // 5. Weld 2 boundary: between Tier 2 Entrance leaf and Tier 2 Exit leaf 0:
        for y in 0..self.cfg.k1 {
            let t2_leaf = self.cfg.t2_in_idx(y, self.cfg.depth2 - 1);
            for z in 0..self.cfg.k2 {
                let exit_leaf = self.cfg.t2_exit_idx(y, z, 0);
                let idx = y * self.cfg.k2 + z;
                let w = self.weld2_couplings[idx];
                let phi = self.weld2_phases[idx];
                let hop = Complex64::from_polar(-g * w, phi);
                edges.push(SparseCoupling::new(t2_leaf, exit_leaf, hop));

                // Tier 2 Exit tree internal edges:
                for col in 0..self.cfg.depth2 {
                    let u = self.cfg.t2_exit_idx(y, z, col);
                    let v = self.cfg.t2_exit_idx(y, z, col + 1);
                    edges.push(SparseCoupling::new(u, v, Complex64::new(-g * sqrt2, 0.0)));
                }
            }
        }

        SparseSymmetricHamiltonian::new(dim, edges)
    }

    /// Single Trotter step on statevector: delegates to the sparse matrix representation.
    #[inline(always)]
    pub fn step_state(
        &self,
        mat: &SparseSymmetricHamiltonian,
        state: &[Complex64],
        next: &mut [Complex64],
        dt: f64,
    ) {
        mat.step_state(state, next, dt);
    }

    /// Single Trotter step on SoA statevector with AVX2 acceleration.
    #[inline(always)]
    pub fn step_state_soa(
        &self,
        mat: &SparseSymmetricHamiltonian,
        r_in: &[f64],
        i_in: &[f64],
        r_out: &mut [f64],
        i_out: &mut [f64],
        r_scratch: &mut [f64],
        i_scratch: &mut [f64],
        dt: f64,
    ) {
        mat.step_state_soa(r_in, i_in, r_out, i_out, r_scratch, i_scratch, dt);
    }
}

/// Structure-of-Arrays (SoA) double-precision statevector for AVX2 accelerated propagation.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantumWalkStateSoA {
    pub dim: usize,
    pub real: Vec<f64>,
    pub imag: Vec<f64>,
}

impl QuantumWalkStateSoA {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            real: vec![0.0; dim],
            imag: vec![0.0; dim],
        }
    }

    #[inline(always)]
    pub fn norm_sqr(&self, idx: usize) -> f64 {
        let r = self.real[idx];
        let i = self.imag[idx];
        r * r + i * i
    }

    #[inline(always)]
    pub fn inject_root(&mut self, root_idx: usize) {
        self.real.fill(0.0);
        self.imag.fill(0.0);
        self.real[root_idx] = 1.0;
    }

    #[inline(always)]
    pub fn copy_from(&mut self, other: &Self) {
        self.real[..self.dim].copy_from_slice(&other.real[..self.dim]);
        self.imag[..self.dim].copy_from_slice(&other.imag[..self.dim]);
    }
}

/// Vectorized SoA state update: next_r = r + dt * acc_i, next_i = i - dt * acc_r.
#[inline(always)]
pub fn step_soa_avx2(
    dim: usize,
    dt: f64,
    r_in: &[f64],
    i_in: &[f64],
    acc_r: &[f64],
    acc_i: &[f64],
    r_out: &mut [f64],
    i_out: &mut [f64],
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        step_soa_avx2_inner(dim, dt, r_in, i_in, acc_r, acc_i, r_out, i_out);
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                step_soa_avx2_inner(dim, dt, r_in, i_in, acc_r, acc_i, r_out, i_out);
                return;
            }
        }
        step_soa_scalar(dim, dt, r_in, i_in, acc_r, acc_i, r_out, i_out);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn step_soa_avx2_inner(
    dim: usize,
    dt: f64,
    r_in: &[f64],
    i_in: &[f64],
    acc_r: &[f64],
    acc_i: &[f64],
    r_out: &mut [f64],
    i_out: &mut [f64],
) {
    let dt_vec = _mm256_set1_pd(dt);
    let mut k = 0;
    while k + 4 <= dim {
        let r = _mm256_loadu_pd(r_in.as_ptr().add(k));
        let i = _mm256_loadu_pd(i_in.as_ptr().add(k));
        let ar = _mm256_loadu_pd(acc_r.as_ptr().add(k));
        let ai = _mm256_loadu_pd(acc_i.as_ptr().add(k));

        let nr = _mm256_fmadd_pd(dt_vec, ai, r);
        let ni = _mm256_fnmadd_pd(dt_vec, ar, i);

        _mm256_storeu_pd(r_out.as_mut_ptr().add(k), nr);
        _mm256_storeu_pd(i_out.as_mut_ptr().add(k), ni);

        k += 4;
    }

    while k < dim {
        r_out[k] = r_in[k] + dt * acc_i[k];
        i_out[k] = i_in[k] - dt * acc_r[k];
        k += 1;
    }
}

#[inline(always)]
fn step_soa_scalar(
    dim: usize,
    dt: f64,
    r_in: &[f64],
    i_in: &[f64],
    acc_r: &[f64],
    acc_i: &[f64],
    r_out: &mut [f64],
    i_out: &mut [f64],
) {
    for k in 0..dim {
        r_out[k] = r_in[k] + dt * acc_i[k];
        i_out[k] = i_in[k] - dt * acc_r[k];
    }
}

/// Instantaneous trajectory snapshot for the cascaded valve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadedTrajectoryPoint {
    pub time: f64,
    pub p_entrance: f64,
    pub p_target_intermediate: f64,
    pub p_target_final_exit: f64,
    pub p_sibling_exits: f64,
    pub p_cross_class_exits: f64,
    pub routing_contrast: f64,
}

/// Benchmark result struct comparing cascaded CTQW with classical walk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadedBenchmarkResult {
    pub depth1: usize,
    pub depth2: usize,
    pub k1: usize,
    pub k2: usize,
    pub total_vertices: usize,
    pub symmetric_dim: usize,
    pub classical_mean_steps: f64,
    pub classical_target_hit_pct: f64,
    pub classical_any_exit_pct: f64,
    pub quantum_target_peak_prob: f64,
    pub quantum_time_to_peak: f64,
    pub quantum_sibling_leakage: f64,
    pub quantum_cross_class_leakage: f64,
    pub quantum_routing_contrast: f64,
}

/// Runs continuous-time quantum walk simulation on the cascaded valve with SoA AVX2 acceleration.
pub fn simulate_cascaded_quantum_walk(
    ham: &CascadedValveHamiltonian,
    target_y: usize,
    target_z: usize,
    t_max: f64,
    dt: f64,
) -> (Vec<CascadedTrajectoryPoint>, f64, f64) {
    let dim = ham.cfg.symmetric_dim();
    let mat = ham.build_symmetric_matrix();

    let mut state = QuantumWalkStateSoA::new(dim);
    let mut next = QuantumWalkStateSoA::new(dim);
    let mut scratch = QuantumWalkStateSoA::new(dim);
    state.inject_root(0); // Injected at ROOT_IN

    let steps = (t_max / dt).round() as usize;
    let mut trajectory = Vec::with_capacity(steps / 20 + 2);

    let target_mid = ham.cfg.mid_root_idx(target_y);
    let target_exit = ham.cfg.final_exit_root_idx(target_y, target_z);

    let mut peak_prob = 0.0f64;
    let mut time_at_peak = 0.0f64;

    for step in 0..=steps {
        let current_time = step as f64 * dt;
        let p_in = state.norm_sqr(0);
        let p_mid = state.norm_sqr(target_mid);
        let p_target = state.norm_sqr(target_exit);

        if p_target > peak_prob {
            peak_prob = p_target;
            time_at_peak = current_time;
        }

        if step % 20 == 0 || step == steps {
            let mut p_sib = 0.0;
            for z in 0..ham.cfg.k2 {
                if z != target_z {
                    let idx = ham.cfg.final_exit_root_idx(target_y, z);
                    p_sib += state.norm_sqr(idx);
                }
            }

            let mut p_cross = 0.0;
            for y in 0..ham.cfg.k1 {
                if y != target_y {
                    for z in 0..ham.cfg.k2 {
                        let idx = ham.cfg.final_exit_root_idx(y, z);
                        p_cross += state.norm_sqr(idx);
                    }
                }
            }

            let non_target_total = p_sib + p_cross;
            let contrast = if non_target_total > 1e-12 {
                p_target / non_target_total
            } else {
                p_target * 1e12
            };

            trajectory.push(CascadedTrajectoryPoint {
                time: current_time,
                p_entrance: p_in,
                p_target_intermediate: p_mid,
                p_target_final_exit: p_target,
                p_sibling_exits: p_sib,
                p_cross_class_exits: p_cross,
                routing_contrast: contrast,
            });
        }

        mat.step_state_soa(
            &state.real,
            &state.imag,
            &mut next.real,
            &mut next.imag,
            &mut scratch.real,
            &mut scratch.imag,
            dt,
        );
        state.copy_from(&next);
    }

    (trajectory, peak_prob, time_at_peak)
}

/// Simulates classical random walk on the cascaded multi-tier graph.
pub fn simulate_cascaded_classical_walk(
    cfg: &CascadedValveConfig,
    target_y: usize,
    target_z: usize,
    num_trials: usize,
    step_cutoff: usize,
    rng: &mut impl Rng,
) -> (f64, f64, f64) {
    let mut target_hits = 0;
    let mut any_exit_hits = 0;
    let mut step_sum = 0usize;

    for _ in 0..num_trials {
        let mut steps = 0usize;
        let mut current_pos = 0usize; // 0 = entrance root

        while steps < step_cutoff {
            steps += 1;

            // Simplified column-position Markov model across the two cascaded tiers:
            // Distance from ROOT_IN: 0 .. (2*depth1 + 2*depth2)
            // Forward vs backward transitions modeled by branch degrees:
            let at_exit = current_pos >= (2 * cfg.depth1 + 2 * cfg.depth2);
            if at_exit {
                any_exit_hits += 1;
                // Check if random dispersal hits target exit (1 / (K1 * K2)):
                let chosen_y = rng.gen_range(0..cfg.k1);
                let chosen_z = rng.gen_range(0..cfg.k2);
                if chosen_y == target_y && chosen_z == target_z {
                    target_hits += 1;
                }
                step_sum += steps;
                break;
            }

            // Move forward with probability 2/3, backward with 1/3:
            let forward_prob = if current_pos == 0 { 1.0 } else { 0.6667 };
            if rng.gen_bool(forward_prob) {
                current_pos += 1;
            } else if current_pos > 0 {
                current_pos -= 1;
            }
        }
    }

    let target_pct = (target_hits as f64 / num_trials as f64) * 100.0;
    let any_pct = (any_exit_hits as f64 / num_trials as f64) * 100.0;
    let mean_steps = if any_exit_hits > 0 {
        step_sum as f64 / any_exit_hits as f64
    } else {
        step_cutoff as f64
    };

    (mean_steps, target_pct, any_pct)
}
