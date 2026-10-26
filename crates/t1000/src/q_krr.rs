// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Quantum Kernel Ridge Regression (Q-KRR) Engine
//!
//! Replaces variational iterative circuit training with convex closed-form Quantum Kernel
//! Ridge Regression over exact Hilbert space statevector overlaps.
//!
//! ## Mathematical Formulation
//! Given feature vectors $\boldsymbol{x} \in \mathbb{R}^D$, a fixed parameter-free unitary map
//! $U(\boldsymbol{x})$ embeds $\boldsymbol{x}$ into an $N$-qubit statevector:
//! $$|\psi(\boldsymbol{x})\rangle = U(\boldsymbol{x}) |0\rangle^{\otimes N}$$
//!
//! The kernel metric between feature vectors $\boldsymbol{x}_i$ and $\boldsymbol{x}_j$ is the exact
//! quantum state fidelity:
//! $$k(\boldsymbol{x}_i, \boldsymbol{x}_j) = |\langle \psi(\boldsymbol{x}_i) | \psi(\boldsymbol{x}_j) \rangle|^2
//!   = \left| \sum_{k=0}^{2^N - 1} \psi_k^*(\boldsymbol{x}_i) \psi_k(\boldsymbol{x}_j) \right|^2$$
//!
//! The optimal regularized weights $\boldsymbol{\alpha}$ are obtained via closed-form Cholesky solve:
//! $$\boldsymbol{\alpha} = (\boldsymbol{K}_{\text{train}} + \lambda \boldsymbol{I})^{-1} \boldsymbol{y}_{\text{norm}}$$

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Hyperparameters for Quantum Kernel Ridge Regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QKRRConfig {
    /// Number of qubits in statevector encoding (e.g. 8 qubits -> 256 complex amplitudes).
    pub num_qubits: usize,
    /// Circuit depth (number of interleaved Hadamard + ZZ entangling layers, default 1).
    pub depth: usize,
    /// Ridge regularization parameter lambda > 0 (e.g. 1e-3 to 1e-1).
    pub lambda: f64,
    /// Angular scale factor for single-qubit phase rotations.
    pub feature_scale: f64,
}

impl Default for QKRRConfig {
    fn default() -> Self {
        Self {
            num_qubits: 8,
            depth: 1,
            lambda: 1e-2,
            feature_scale: 1.0,
        }
    }
}

/// Lightweight complex number for zero-allocation statevector fidelity computations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    #[inline(always)]
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline(always)]
    pub fn conj(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    #[inline(always)]
    pub fn mul(&self, rhs: &Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

/// Quantum Kernel Ridge Regressor caching vectorized Hilbert states and Cholesky weights.
#[derive(Debug, Clone)]
pub struct QuantumKernelRegressor {
    pub config: QKRRConfig,
    pub train_states: Vec<Vec<Complex64>>,
    pub alpha_weights: Vec<f64>,
    pub y_mean: f64,
    pub y_std: f64,
}

impl QuantumKernelRegressor {
    pub fn new(config: QKRRConfig) -> Self {
        Self {
            config,
            train_states: Vec::new(),
            alpha_weights: Vec::new(),
            y_mean: 0.0,
            y_std: 1.0,
        }
    }

    /// Evaluates exact quantum state fidelity: |<psi_a | psi_b>|^2.
    #[inline]
    pub fn state_fidelity(a: &[Complex64], b: &[Complex64]) -> f64 {
        let mut inner_re = 0.0f64;
        let mut inner_im = 0.0f64;

        for (ca, cb) in a.iter().zip(b.iter()) {
            // ca* * cb = (re_a - i*im_a) * (re_b + i*im_b)
            inner_re += ca.re * cb.re + ca.im * cb.im;
            inner_im += ca.re * cb.im - ca.im * cb.re;
        }

        inner_re * inner_re + inner_im * inner_im
    }

    /// Encodes normalized molecular features into an N-qubit statevector (dim = 2^N).
    /// Employs Hadamard base, single-qubit Rz phases, and second-order entangling ZZ phases,
    /// repeated for `config.depth` interleaved layers.
    pub fn encode_state(&self, features: &[f64]) -> Vec<Complex64> {
        let dim = 1 << self.config.num_qubits;
        let mut state = vec![Complex64::new(0.0, 0.0); dim];

        // Equal superposition initialization |+>^N
        let norm_factor = 1.0 / (dim as f64).sqrt();
        for amp in &mut state {
            *amp = Complex64::new(norm_factor, 0.0);
        }

        let depth = self.config.depth.max(1);
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;

        for layer in 0..depth {
            if layer > 0 {
                // Interleaved global Hadamard transform H^{\otimes N}
                for w in 0..self.config.num_qubits {
                    let bit_step = 1 << w;
                    for i in 0..dim {
                        if (i & bit_step) == 0 {
                            let a = state[i];
                            let b = state[i + bit_step];
                            state[i] = Complex64::new((a.re + b.re) * inv_sqrt2, (a.im + b.im) * inv_sqrt2);
                            state[i + bit_step] = Complex64::new((a.re - b.re) * inv_sqrt2, (a.im - b.im) * inv_sqrt2);
                        }
                    }
                }
            }

            // Apply product state phases: exp(-i * phi_k)
            for (i, amp) in state.iter_mut().enumerate() {
                let mut phase = 0.0f64;

                // Single-qubit phase contributions
                for (p, &xp) in features.iter().take(self.config.num_qubits).enumerate() {
                    if (i >> p) & 1 == 1 {
                        phase += xp * self.config.feature_scale;
                    }
                }

                // Entangling two-qubit ZZ phase contributions: (PI - x_p) * (PI - x_q) * 0.5
                let n_q = self.config.num_qubits;
                for p in 0..n_q {
                    for q in (p + 1)..n_q {
                        let bit_p = (i >> p) & 1;
                        let bit_q = (i >> q) & 1;
                        if bit_p ^ bit_q == 1 {
                            let xp = features[p.min(features.len() - 1)];
                            let xq = features[q.min(features.len() - 1)];
                            phase += (PI - xp) * (PI - xq) * 0.5;
                        }
                    }
                }

                let cos_p = phase.cos();
                let sin_p = -phase.sin();
                *amp = Complex64::new(
                    amp.re * cos_p - amp.im * sin_p,
                    amp.re * sin_p + amp.im * cos_p,
                );
            }
        }

        state
    }

    /// Builds the Gram matrix K in parallel across threads via Rayon.
    pub fn compute_gram_matrix(&self, states: &[Vec<Complex64>]) -> Vec<f64> {
        let n = states.len();
        let mut gram = vec![0.0f64; n * n];

        // Parallelize row-wise over upper triangle
        let rows: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut row = vec![0.0f64; n];
                row[i] = 1.0; // Self-fidelity |<psi|psi>|^2 = 1.0
                for j in (i + 1)..n {
                    row[j] = Self::state_fidelity(&states[i], &states[j]);
                }
                row
            })
            .collect();

        // Assemble symmetric Gram matrix
        for (i, row) in rows.iter().enumerate() {
            for j in i..n {
                let val = row[j];
                gram[i * n + j] = val;
                gram[j * n + i] = val;
            }
        }

        gram
    }

    /// Computes summary geometry metrics on a Gram matrix K:
    /// returns `(mean_off_diagonal_fidelity, max_off_diag, min_off_diag)`.
    pub fn compute_gram_metrics(gram: &[f64], n: usize) -> (f64, f64, f64) {
        if n <= 1 {
            return (1.0, 1.0, 1.0);
        }
        let mut sum_off = 0.0;
        let mut count = 0usize;
        let mut max_off = f64::NEG_INFINITY;
        let mut min_off = f64::INFINITY;

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let val = gram[i * n + j];
                    sum_off += val;
                    count += 1;
                    if val > max_off {
                        max_off = val;
                    }
                    if val < min_off {
                        min_off = val;
                    }
                }
            }
        }

        (sum_off / (count as f64), max_off, min_off)
    }

    /// Trains the kernel model via Cholesky decomposition on (K + lambda * I).
    pub fn fit(&mut self, x_train: &[Vec<f64>], y_train: &[f64]) {
        let n = x_train.len();
        assert_eq!(n, y_train.len());

        // 1. Target standard scaling
        self.y_mean = y_train.iter().sum::<f64>() / n as f64;
        let var = y_train
            .iter()
            .map(|&y| (y - self.y_mean).powi(2))
            .sum::<f64>()
            / n as f64;
        self.y_std = var.sqrt().max(1e-8);

        let y_norm: Vec<f64> = y_train
            .iter()
            .map(|&y| (y - self.y_mean) / self.y_std)
            .collect();

        // 2. Vectorize Hilbert states in parallel
        self.train_states = x_train
            .into_par_iter()
            .map(|x| self.encode_state(x))
            .collect();

        // 3. Compute Gram Matrix
        let mut k_mat = self.compute_gram_matrix(&self.train_states);

        // 4. Regularize diagonal: K + lambda * I
        for i in 0..n {
            k_mat[i * n + i] += self.config.lambda;
        }

        // 5. Invert system: (K + lambda * I) * alpha = y_norm via Cholesky
        self.alpha_weights = solve_cholesky(&k_mat, &y_norm, n);
    }

    /// Evaluates out-of-sample test molecules against the cached training states in parallel.
    pub fn predict(&self, x_test: &[Vec<f64>]) -> Vec<f64> {
        let test_states: Vec<Vec<Complex64>> = x_test
            .into_par_iter()
            .map(|x| self.encode_state(x))
            .collect();

        test_states
            .into_par_iter()
            .map(|t_state| {
                let mut sum_pred = 0.0f64;
                for (train_idx, tr_state) in self.train_states.iter().enumerate() {
                    let k_val = Self::state_fidelity(tr_state, &t_state);
                    sum_pred += self.alpha_weights[train_idx] * k_val;
                }
                // Rescale to target physical units
                sum_pred * self.y_std + self.y_mean
            })
            .collect()
    }
}

/// Solves regularized symmetric linear system L * L^T * x = b via Cholesky factorization.
pub fn solve_cholesky(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0f64; n * n];

    // Factorization A = L * L^T
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0f64;
            for k in 0..j {
                sum += l[i * n + k] * l[j * n + k];
            }

            if i == j {
                let diag = a[i * n + i] - sum;
                l[i * n + j] = if diag > 1e-11 { diag.sqrt() } else { 1e-5 };
            } else {
                let d = l[j * n + j];
                l[i * n + j] = if d.abs() > 1e-12 {
                    (a[i * n + j] - sum) / d
                } else {
                    0.0
                };
            }
        }
    }

    // Forward substitution L * y = b
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut sum = 0.0f64;
        for k in 0..i {
            sum += l[i * n + k] * y[k];
        }
        let diag = l[i * n + i];
        y[i] = if diag.abs() > 1e-12 {
            (b[i] - sum) / diag
        } else {
            0.0
        };
    }

    // Back substitution L^T * x = y
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = 0.0f64;
        for k in (i + 1)..n {
            sum += l[k * n + i] * x[k];
        }
        let diag = l[i * n + i];
        x[i] = if diag.abs() > 1e-12 {
            (y[i] - sum) / diag
        } else {
            0.0
        };
    }

    x
}

/// Min-max scales raw molecular descriptors into $[0, \pi]$ along each feature dimension.
pub fn min_max_scale_features(x: &[Vec<f64>], target_dim: usize) -> Vec<Vec<f64>> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let d = x[0].len().min(target_dim);

    let mut mins = vec![f64::INFINITY; d];
    let mut maxs = vec![f64::NEG_INFINITY; d];

    for row in x {
        for (f, &val) in row.iter().take(d).enumerate() {
            if val < mins[f] {
                mins[f] = val;
            }
            if val > maxs[f] {
                maxs[f] = val;
            }
        }
    }

    x.iter()
        .map(|row| {
            let mut scaled = vec![0.0f64; d];
            for f in 0..d {
                let span = maxs[f] - mins[f];
                let val = row.get(f).copied().unwrap_or(0.0);
                scaled[f] = if span > 1e-8 {
                    ((val - mins[f]) / span) * PI
                } else {
                    0.5 * PI
                };
            }
            scaled
        })
        .collect()
}

/// Computes (MSE, MAE, R^2) metrics between target labels and predictions.
pub fn evaluate_regression_metrics(y_true: &[f64], y_pred: &[f64]) -> (f64, f64, f64) {
    let n = y_true.len();
    assert_eq!(n, y_pred.len());
    let mut sse = 0.0f64;
    let mut sae = 0.0f64;
    let mut sum_y = 0.0f64;

    for (&t, &p) in y_true.iter().zip(y_pred) {
        let diff = t - p;
        sse += diff * diff;
        sae += diff.abs();
        sum_y += t;
    }

    let mse = sse / n as f64;
    let mae = sae / n as f64;
    let mean_y = sum_y / n as f64;

    let mut sst = 0.0f64;
    for &t in y_true {
        sst += (t - mean_y).powi(2);
    }

    let r2 = if sst > 1e-12 { 1.0 - sse / sst } else { 0.0 };
    (mse, mae, r2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_fidelity_identity_and_orthogonality() {
        let s1 = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let s2 = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];

        assert!((QuantumKernelRegressor::state_fidelity(&s1, &s1) - 1.0).abs() < 1e-12);
        assert!(QuantumKernelRegressor::state_fidelity(&s1, &s2) < 1e-12);
    }

    #[test]
    fn test_gram_matrix_symmetry_and_unit_diagonal() {
        let reg = QuantumKernelRegressor::new(QKRRConfig {
            num_qubits: 3,
            depth: 1,
            lambda: 0.01,
            feature_scale: 1.0,
        });

        let samples = [
            vec![0.1, 0.2, 0.3],
            vec![1.2, 0.8, 0.5],
            vec![2.5, 1.9, 0.1],
        ];

        let states: Vec<Vec<Complex64>> = samples.iter().map(|s| reg.encode_state(s)).collect();
        let gram = reg.compute_gram_matrix(&states);

        for i in 0..3 {
            assert!((gram[i * 3 + i] - 1.0).abs() < 1e-12);
            for j in 0..3 {
                assert!((gram[i * 3 + j] - gram[j * 3 + i]).abs() < 1e-12);
                assert!(gram[i * 3 + j] >= 0.0 && gram[i * 3 + j] <= 1.000001);
            }
        }
    }

    #[test]
    fn test_cholesky_solver_accuracy() {
        // Simple 2x2 positive-definite matrix [[4, 2], [2, 5]] * [x1, x2] = [8, 12] -> [1, 2]
        let a = vec![4.0, 2.0, 2.0, 5.0];
        let b = vec![8.0, 12.0];
        let x = solve_cholesky(&a, &b, 2);

        assert!((x[0] - 1.0).abs() < 1e-8);
        assert!((x[1] - 2.0).abs() < 1e-8);
    }

    #[test]
    fn test_qkrr_fit_and_predict_toy_convergence() {
        let config = QKRRConfig {
            num_qubits: 4,
            depth: 1,
            lambda: 1e-3,
            feature_scale: 0.5,
        };
        let mut model = QuantumKernelRegressor::new(config);

        let x_train = vec![
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.5, 0.6, 0.7, 0.8],
            vec![1.1, 1.2, 1.3, 1.4],
            vec![2.1, 2.2, 2.3, 2.4],
        ];
        let y_train = vec![10.0, 20.0, 35.0, 60.0];

        model.fit(&x_train, &y_train);
        let preds = model.predict(&x_train);

        let (mse, _mae, r2) = evaluate_regression_metrics(&y_train, &preds);
        assert!(
            mse < 0.1,
            "Q-KRR must fit training points closely, got MSE {mse}"
        );
        assert!(r2 > 0.99, "Q-KRR R^2 must be > 0.99, got {r2}");
    }

    #[test]
    fn test_depth_two_fidelity_concentration() {
        // 8-qubit comparison: Depth L=1 retains chemical overlap; Depth L=2 concentrates toward orthogonal collapse
        let reg_l1 = QuantumKernelRegressor::new(QKRRConfig {
            num_qubits: 8,
            depth: 1,
            lambda: 0.01,
            feature_scale: 1.0,
        });
        let reg_l2 = QuantumKernelRegressor::new(QKRRConfig {
            num_qubits: 8,
            depth: 2,
            lambda: 0.01,
            feature_scale: 1.0,
        });

        let mol_a = vec![0.5, 1.2, 0.8, 1.5, 0.3, 2.1, 0.7, 1.9];
        let mol_b = vec![0.6, 1.1, 0.9, 1.4, 0.4, 2.0, 0.8, 1.8]; // Minor perturbation

        let state_a_l1 = reg_l1.encode_state(&mol_a);
        let state_b_l1 = reg_l1.encode_state(&mol_b);
        let fid_l1 = QuantumKernelRegressor::state_fidelity(&state_a_l1, &state_b_l1);

        let state_a_l2 = reg_l2.encode_state(&mol_a);
        let state_b_l2 = reg_l2.encode_state(&mol_b);
        let fid_l2 = QuantumKernelRegressor::state_fidelity(&state_a_l2, &state_b_l2);

        // L=1 retains continuous overlap (> 0.20), whereas L=2 drops significantly closer to 1/256
        assert!(fid_l1 > fid_l2, "L=1 fidelity ({fid_l1}) must exceed L=2 fidelity ({fid_l2})");
        assert!(fid_l1 > 0.20, "L=1 must preserve chemical continuity, got {fid_l1}");
    }
}
