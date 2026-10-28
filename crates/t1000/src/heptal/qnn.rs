// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Sovereign Heptal Qudit ($d = 7, \mathbb{Z}_7$) Quantum Neural Network Engine
//!
//! Implements high-dimensional molecular feature mapping and variational inference:
//! 1. 19-dimensional molecular descriptor injection into $\mathbb{Z}_7$ cyclic phase spaces.
//! 2. 1D Matrix Product State (MPS) Heptal Linear Zipper with 784-byte L1 cache footprint.
//! 3. Radix-7 Chrestenson DFT quantum interference and modular entangling gates.
//! 4. Gauge-invariant $\mathbb{Z}_7$ clock observables: $\langle \text{Re}(Z_7) \rangle$, $\langle \text{Im}(Z_7) \rangle$, and local entropy.

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use crate::heptal_wave::HeptalWaveState;

/// Canonical cyclic 7th-root of unity constants for Z_7 clock observables.
const COS_7: [f64; 7] = [
    1.0,
    0.6234898018587335,
    -0.22252093395631445,
    -0.9009688679024191,
    -0.9009688679024191,
    -0.22252093395631445,
    0.6234898018587335,
];

const SIN_7: [f64; 7] = [
    0.0,
    0.7818314824680298,
    0.9749279121818236,
    0.4338837391175581,
    -0.4338837391175581,
    -0.9749279121818236,
    -0.7818314824680298,
];

/// Configuration for the Heptal Molecular Zipper QNN.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeptalQnnConfig {
    pub n_qudits: usize,
    pub entangle_strength: f64,
    pub epoxy_threshold: f64,
    pub use_chrestenson: bool,
}

impl Default for HeptalQnnConfig {
    fn default() -> Self {
        Self {
            n_qudits: 19,
            entangle_strength: 1.0,
            epoxy_threshold: 0.999,
            use_chrestenson: true,
        }
    }
}

/// Extracted cyclic clock observables and entropy at a single Heptal qudit site.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HeptalSiteObservables {
    pub qudit_idx: usize,
    pub clock_re: f64,
    pub clock_im: f64,
    pub entropy_nats: f64,
    pub dominant_digit: usize,
    pub dominant_p: f64,
}

/// Multi-task prediction outputs emitted by the Heptal QNN.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HeptalPrediction {
    pub log_p: f64,
    pub qed: f64,
    pub sas: f64,
}

/// Parameterized Heptal Molecular Zipper QNN.
/// Evaluates an arbitrary $N$-qudit molecular graph with a constant 784-byte L1 cache active window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeptalMolecularZipperQnn {
    pub config: HeptalQnnConfig,
    /// Variational phase weights: `[n_qudits][7]`
    pub phase_weights: Vec<[f64; 7]>,
    /// Linear readout weights for logP: `[n_qudits * 3 + 1]` (Re, Im, Entropy per qudit + bias)
    pub logp_weights: Vec<f64>,
    /// Linear readout weights for QED
    pub qed_weights: Vec<f64>,
    /// Linear readout weights for SAS
    pub sas_weights: Vec<f64>,
}

impl HeptalMolecularZipperQnn {
    /// Constructs a new Heptal Molecular Zipper QNN with seeded initial weights.
    pub fn new(config: HeptalQnnConfig, seed: u64) -> Self {
        let n = config.n_qudits;
        let mut phase_weights = Vec::with_capacity(n);

        // Deterministic initial phase weights based on modular golden ratio sweeps
        for i in 0..n {
            let mut w = [0.0f64; 7];
            for d in 0..7 {
                let angle = (seed as f64 + (i * 7 + d) as f64 * 1.618033988749895).sin() * PI;
                w[d] = angle;
            }
            phase_weights.push(w);
        }

        let n_features = n * 3 + 1;
        let logp_weights = Self::init_readout_weights(seed, 1, n_features, 0.05);
        let qed_weights = Self::init_readout_weights(seed, 2, n_features, 0.02);
        let sas_weights = Self::init_readout_weights(seed, 3, n_features, 0.04);

        Self {
            config,
            phase_weights,
            logp_weights,
            qed_weights,
            sas_weights,
        }
    }

    fn init_readout_weights(seed: u64, offset: u64, len: usize, scale: f64) -> Vec<f64> {
        (0..len)
            .map(|i| {
                let val = ((seed + offset * 1000 + i as u64) as f64 * 0.7548776662).sin();
                val * scale
            })
            .collect()
    }

    /// Maps continuous molecular descriptors into a 7-point cyclic rotation angles array.
    #[inline(always)]
    fn feature_to_angles(feat: f64, weights: &[f64; 7]) -> [f64; 7] {
        let mut angles = [0.0f64; 7];
        let norm_feat = feat.tanh();
        for d in 0..7 {
            angles[d] = weights[d] + (2.0 * PI * d as f64 / 7.0) * norm_feat;
        }
        angles
    }

    /// Evaluates the forward pass of the molecular descriptors through the Heptal Zipper chain.
    /// Operates entirely in CPU L1 cache with 784 bytes working state memory.
    pub fn forward_chain(&self, features: &[f64]) -> Vec<HeptalSiteObservables> {
        let n = self.config.n_qudits;
        let mut observables = Vec::with_capacity(n);

        // Active 2-qudit state: 7^2 = 49 complex amplitudes (784 bytes)
        let mut active_state = HeptalWaveState::new_ground(2);

        for i in 0..n {
            let feat = features.get(i).copied().unwrap_or(0.0);
            let angles = Self::feature_to_angles(feat, &self.phase_weights[i]);

            // 1. Parameterized phase rotation on active qudit 0
            active_state.apply_qudit_phase(0, &angles);

            // 2. Chrestenson DFT quantum interference
            if self.config.use_chrestenson {
                active_state.apply_chrestenson(0);
            }

            // 3. Extract 7-simplex probabilities on qudit 0
            let probs = active_state.qudit_probabilities(0);
            let mut clock_re = 0.0f64;
            let mut clock_im = 0.0f64;
            let mut entropy = 0.0f64;
            let mut dom_digit = 0usize;
            let mut dom_p = 0.0f64;

            for d in 0..7 {
                let p = probs[d];
                clock_re += p * COS_7[d];
                clock_im += p * SIN_7[d];
                if p > dom_p {
                    dom_p = p;
                    dom_digit = d;
                }
                if p > 1e-15 {
                    entropy -= p * p.ln();
                }
            }

            observables.push(HeptalSiteObservables {
                qudit_idx: i,
                clock_re,
                clock_im,
                entropy_nats: entropy,
                dominant_digit: dom_digit,
                dominant_p: dom_p,
            });

            // 4. Modular coupling with neighbor qudit 1 and sliding window advance
            if i + 1 < n {
                // Modular complementarity entangling gate
                active_state.apply_modular_complement(0, 1);

                // Slide window forward by 1 site
                active_state = Self::slide_window_qudit(&active_state, dom_digit);
            }
        }

        observables
    }

    /// Predicts multi-task properties (logP, QED, SAS) for an input molecular descriptor vector.
    pub fn predict(&self, features: &[f64]) -> HeptalPrediction {
        let obs = self.forward_chain(features);
        let n = self.config.n_qudits;
        let mut readout_vec = Vec::with_capacity(n * 3 + 1);

        for site in &obs {
            readout_vec.push(site.clock_re);
            readout_vec.push(site.clock_im);
            readout_vec.push(site.entropy_nats);
        }
        readout_vec.push(1.0); // Bias term

        let log_p = Self::dot_product(&self.logp_weights, &readout_vec);
        let qed = Self::dot_product(&self.qed_weights, &readout_vec).clamp(0.0, 1.0);
        let sas = Self::dot_product(&self.sas_weights, &readout_vec).clamp(1.0, 10.0);

        HeptalPrediction { log_p, qed, sas }
    }

    #[inline(always)]
    fn dot_product(w: &[f64], x: &[f64]) -> f64 {
        w.iter().zip(x).map(|(&wi, &xi)| wi * xi).sum()
    }

    /// Slides the 2-qudit window forward by 1 site in radix-7:
    /// Shifts qudit 1 into position 0, introduces ground state |0> at position 1.
    fn slide_window_qudit(state: &HeptalWaveState, dom_d: usize) -> HeptalWaveState {
        let mut next = HeptalWaveState::new_ground(2);
        let mut norm_sq = 0.0f64;

        for d1 in 0..7 {
            let old_idx = dom_d + 7 * d1;
            let new_idx = d1;
            let r = state.real[old_idx];
            let i = state.imag[old_idx];
            next.real[new_idx] = r;
            next.imag[new_idx] = i;
            norm_sq += r * r + i * i;
        }

        if norm_sq > 1e-15 {
            let inv_norm = 1.0 / norm_sq.sqrt();
            for r in next.real.iter_mut() {
                *r *= inv_norm;
            }
            for i in next.imag.iter_mut() {
                *i *= inv_norm;
            }
        } else {
            next.real[0] = 1.0;
        }

        next
    }

    /// Fits linear readout weights using ridge regression closed form: W = (X^T X + lambda I)^-1 X^T Y.
    pub fn fit_readouts(
        &mut self,
        x_features: &[Vec<f64>],
        y_logp: &[f64],
        y_qed: &[f64],
        y_sas: &[f64],
        lambda: f64,
    ) {
        let m = x_features.len();
        let n_dim = self.config.n_qudits * 3 + 1;
        if m == 0 {
            return;
        }

        // Build design matrix X: [m][n_dim]
        let mut x_mat: Vec<Vec<f64>> = Vec::with_capacity(m);
        for feats in x_features {
            let obs = self.forward_chain(feats);
            let mut row = Vec::with_capacity(n_dim);
            for site in &obs {
                row.push(site.clock_re);
                row.push(site.clock_im);
                row.push(site.entropy_nats);
            }
            row.push(1.0); // Bias
            x_mat.push(row);
        }

        self.logp_weights = Self::solve_ridge(&x_mat, y_logp, lambda);
        self.qed_weights = Self::solve_ridge(&x_mat, y_qed, lambda);
        self.sas_weights = Self::solve_ridge(&x_mat, y_sas, lambda);
    }

    /// Solves linear ridge regression via normal equations with L2 regularization.
    fn solve_ridge(x: &[Vec<f64>], y: &[f64], lambda: f64) -> Vec<f64> {
        let m = x.len();
        let p = x[0].len();

        // Compute X^T X + lambda * I
        let mut xt_x = vec![vec![0.0f64; p]; p];
        for i in 0..p {
            for j in 0..p {
                let mut sum = 0.0;
                for k in 0..m {
                    sum += x[k][i] * x[k][j];
                }
                xt_x[i][j] = sum;
            }
            xt_x[i][i] += lambda;
        }

        // Compute X^T Y
        let mut xt_y = vec![0.0f64; p];
        for i in 0..p {
            let mut sum = 0.0;
            for k in 0..m {
                sum += x[k][i] * y[k];
            }
            xt_y[i] = sum;
        }

        // Gaussian elimination with partial pivoting to solve (X^T X + lambda I) w = X^T Y
        let mut a = xt_x;
        let mut b = xt_y;

        for i in 0..p {
            let mut max_row = i;
            let mut max_val = a[i][i].abs();
            for r in (i + 1)..p {
                if a[r][i].abs() > max_val {
                    max_val = a[r][i].abs();
                    max_row = r;
                }
            }
            a.swap(i, max_row);
            b.swap(i, max_row);

            let pivot = a[i][i];
            if pivot.abs() > 1e-12 {
                for r in (i + 1)..p {
                    let factor = a[r][i] / pivot;
                    for c in i..p {
                        a[r][c] -= factor * a[i][c];
                    }
                    b[r] -= factor * b[i];
                }
            }
        }

        // Back-substitution
        let mut w = vec![0.0f64; p];
        for i in (0..p).rev() {
            let mut sum = b[i];
            for j in (i + 1)..p {
                sum -= a[i][j] * w[j];
            }
            let diag = a[i][i];
            w[i] = if diag.abs() > 1e-12 { sum / diag } else { 0.0 };
        }

        w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heptal_molecular_zipper_execution_bounds() {
        let config = HeptalQnnConfig {
            n_qudits: 19,
            entangle_strength: 1.0,
            epoxy_threshold: 0.999,
            use_chrestenson: true,
        };
        let qnn = HeptalMolecularZipperQnn::new(config, 42);

        // Dummy 19-dimensional molecular feature vector
        let features = vec![1.0; 19];
        let obs = qnn.forward_chain(&features);

        assert_eq!(obs.len(), 19);
        for site in &obs {
            assert!(site.entropy_nats >= 0.0 && site.entropy_nats <= 7.0f64.ln() + 1e-8);
            assert!(site.dominant_p >= 1.0 / 7.0 - 1e-8 && site.dominant_p <= 1.0 + 1e-8);
            assert!(site.dominant_digit < 7);
            let clock_mag = (site.clock_re.powi(2) + site.clock_im.powi(2)).sqrt();
            assert!(clock_mag <= 1.0 + 1e-8);
        }

        let pred = qnn.predict(&features);
        assert!(!pred.log_p.is_nan());
        assert!(pred.qed >= 0.0 && pred.qed <= 1.0);
        assert!(pred.sas >= 1.0 && pred.sas <= 10.0);
    }
}
