// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Dual-Axis Continuous Feature Re-Uploader with Smooth SU(2) Spectral Scaling.
//! Maps classical descriptors into full 3D Bloch trajectories via
//! Rz(b) * Ry(w_y * tanh(x)) * Rx(w_x * tanh(x)), eliminating 1D great-circle trapping
//! and preventing multi-wrap phase aliasing on extreme chemical outliers (e.g. SMILES logP).

use serde::{Deserialize, Serialize};

use crate::ansatz::dispatch::apply_rx_op;
use crate::simd::{apply_ry_kernel, apply_rz_kernel};

/// Continuous SU(2) dual-axis parameter encoder.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DualAxisEncoder {
    pub weights_rx: Vec<f64>,
    pub weights_ry: Vec<f64>,
    pub biases_rz: Vec<f64>,
}

impl DualAxisEncoder {
    /// Creates a new dual-axis encoder initialized with unit scale factors.
    pub fn new(n_qubits: usize) -> Self {
        Self {
            weights_rx: vec![1.0; n_qubits],
            weights_ry: vec![1.0; n_qubits],
            biases_rz: vec![0.0; n_qubits],
        }
    }

    /// Creates an encoder with explicit weight vectors.
    pub fn with_weights(
        weights_rx: Vec<f64>,
        weights_ry: Vec<f64>,
        biases_rz: Vec<f64>,
    ) -> Self {
        assert_eq!(weights_rx.len(), weights_ry.len());
        assert_eq!(weights_rx.len(), biases_rz.len());
        Self {
            weights_rx,
            weights_ry,
            biases_rz,
        }
    }

    /// Maps scalar feature `x` onto full SU(2) trajectory:
    /// Rz(b) * Ry(w_y * tanh(x)) * Rx(w_x * tanh(x))
    #[inline(always)]
    pub fn encode_qubit_angles(&self, q: usize, raw_feature: f64) -> (f64, f64, f64) {
        // Smooth spectral clamping via tanh prevents multi-wrap phase aliasing on extreme outliers
        let x_norm = raw_feature.tanh() * std::f64::consts::PI;
        let theta_x = self.weights_rx[q] * x_norm;
        let theta_y = self.weights_ry[q] * x_norm;
        let phi_z = self.biases_rz[q];
        (theta_x, theta_y, phi_z)
    }

    /// Encodes and applies the full SU(2) rotation tuple to qubit `q` in-place.
    #[inline(always)]
    pub fn apply_to_qubit(
        &self,
        re: &mut [f64],
        im: &mut [f64],
        q: usize,
        raw_feature: f64,
    ) {
        let (tx, ty, pz) = self.encode_qubit_angles(q, raw_feature);
        if tx.abs() > 1e-14 {
            apply_rx_op(re, im, q, tx);
        }
        if ty.abs() > 1e-14 {
            apply_ry_kernel(re, im, q, ty);
        }
        if pz.abs() > 1e-14 {
            apply_rz_kernel(re, im, q, pz);
        }
    }

    /// Encodes and applies a feature vector across all registered qubits.
    pub fn apply_all_features(
        &self,
        re: &mut [f64],
        im: &mut [f64],
        features: &[f64],
    ) {
        let n = self.weights_rx.len();
        for q in 0..n {
            let feat = if !features.is_empty() {
                features[q % features.len()]
            } else {
                0.0
            };
            self.apply_to_qubit(re, im, q, feat);
        }
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_axis_outlier_boundedness() {
        let encoder = DualAxisEncoder::new(4);
        // Extreme outlier feature (+10,000.0)
        let (tx, ty, pz) = encoder.encode_qubit_angles(0, 10_000.0);
        assert!(tx <= std::f64::consts::PI);
        assert!(ty <= std::f64::consts::PI);
        assert_eq!(pz, 0.0);

        // Negative outlier (-50,000.0)
        let (tx_neg, ty_neg, _) = encoder.encode_qubit_angles(1, -50_000.0);
        assert!(tx_neg >= -std::f64::consts::PI);
        assert!(ty_neg >= -std::f64::consts::PI);
    }

    #[test]
    fn test_dual_axis_state_normalization() {
        let n = 3;
        let encoder = DualAxisEncoder::new(n);
        let mut re = vec![0.0; 1 << n];
        let mut im = vec![0.0; 1 << n];
        re[0] = 1.0;

        let feats = [1.2, -4.5, 8.9];
        encoder.apply_all_features(&mut re, &mut im, &feats);

        let norm_sq: f64 = re.iter().zip(&im).map(|(r, i)| r * r + i * i).sum();
        assert!((norm_sq - 1.0).abs() < 1e-12);
    }
}
