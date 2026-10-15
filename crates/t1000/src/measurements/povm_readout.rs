// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

use crate::state::QuantumState;
use t1000_core::{interference, POVM};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Generalized Positive Operator-Valued Measure (POVM) Readout Classifier.
///
/// **Experimental.** The operators are arbitrary unit vectors, not a valid POVM (they do not
/// sum to the identity). Scores are min-max normalised before the softmax, so with two
/// classes the output is always about 0.0002 / 0.9998 whatever the input; the
/// probabilities carry ranking information only, not calibrated confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct POVMClassifier {
    pub dim: usize,
    pub class_labels: Vec<String>,
    /// POVM measurement operators: each class corresponds to an operator vector
    pub operators: Vec<Vec<(f64, f64)>>,
}

impl POVMClassifier {
    /// Deterministic classifier with operators initialised from [`crate::DEFAULT_SEED`].
    pub fn new(dim: usize, class_labels: Vec<String>) -> Self {
        Self::new_seeded(dim, class_labels, crate::DEFAULT_SEED)
    }

    /// Deterministic classifier with random unit operators drawn using `seed`.
    pub fn new_seeded(dim: usize, class_labels: Vec<String>, seed: u64) -> Self {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

        let mut ops = Vec::with_capacity(class_labels.len());
        for _ in 0..class_labels.len() {
            let mut op = Vec::with_capacity(dim);
            let mut norm_sq = 0.0;
            for _ in 0..dim {
                let r: f64 = rng.gen_range(-1.0..1.0);
                let i: f64 = rng.gen_range(-1.0..1.0);
                norm_sq += r * r + i * i;
                op.push((r, i));
            }
            let norm = norm_sq.sqrt();
            for (r, i) in &mut op {
                *r /= norm;
                *i /= norm;
            }
            ops.push(op);
        }

        Self {
            dim,
            class_labels,
            operators: ops,
        }
    }

    /// Soft measurement readout yielding probability distribution over classes with Boltzmann temperature sharpening:
    /// p(m) = exp((s_m - max(s)) / T) / sum_k exp((s_k - max(s)) / T)
    pub fn predict_probabilities(&self, state: &QuantumState) -> Vec<(String, f64)> {
        self.predict_probabilities_with_temp(state, 0.12)
    }

    /// Predict probabilities with configurable inverse temperature T.
    /// Lower T (< 0.2) sharpens confidence toward top resonant category.
    pub fn predict_probabilities_with_temp(
        &self,
        state: &QuantumState,
        temperature: f64,
    ) -> Vec<(String, f64)> {
        assert_eq!(self.dim, state.dim);
        let povm_ops: Vec<Vec<Complex64>> = self
            .operators
            .iter()
            .map(|op| op.iter().map(|&(r, i)| Complex64::new(r, i)).collect())
            .collect();

        let povm = POVM::new(povm_ops);
        let reg = state.to_register(42);
        let raw_scores = povm.measure_soft(&reg);

        let max_score = raw_scores
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_score = raw_scores.iter().copied().fold(f64::INFINITY, f64::min);
        let spread = (max_score - min_score).max(1e-9);

        // Normalize raw scores to [0, 1] relative contrast before Boltzmann exponentiation
        let exp_scores: Vec<f64> = raw_scores
            .iter()
            .map(|&s| {
                let norm_s = (s - min_score) / spread;
                (norm_s / temperature).exp()
            })
            .collect();

        let sum_exp: f64 = exp_scores.iter().sum();
        let probs: Vec<f64> = if sum_exp > 1e-12 {
            exp_scores.iter().map(|e| e / sum_exp).collect()
        } else {
            vec![1.0 / self.class_labels.len() as f64; self.class_labels.len()]
        };

        self.class_labels
            .iter()
            .cloned()
            .zip(probs)
            .collect()
    }

    /// Set category operator directly from a calibrated category centroid state.
    pub fn set_class_centroid(&mut self, class_idx: usize, centroid: &QuantumState) {
        assert!(class_idx < self.operators.len());
        assert_eq!(self.dim, centroid.dim);
        self.operators[class_idx] = centroid.amplitudes();
    }

    /// Calibrate / adapt class operator toward the target document quantum state.
    pub fn adapt_class_operator(&mut self, class_idx: usize, state: &QuantumState, alpha: f64) {
        assert!(class_idx < self.operators.len());
        assert_eq!(self.dim, state.dim);

        let op = &mut self.operators[class_idx];
        let mut norm_sq = 0.0;
        for (idx, (op_r, op_i)) in op.iter_mut().enumerate() {
            let (target_r, target_i) = state.amplitude(idx);
            *op_r = *op_r * (1.0 - alpha) + target_r * alpha;
            *op_i = *op_i * (1.0 - alpha) + target_i * alpha;
            norm_sq += *op_r * *op_r + *op_i * *op_i;
        }

        let norm = norm_sq.sqrt();
        if norm > 1e-12 {
            for (op_r, op_i) in op.iter_mut() {
                *op_r /= norm;
                *op_i /= norm;
            }
        }
    }

    /// Net quantum interference score between two candidate states:
    /// I_net = I_constructive - I_destructive
    pub fn net_interference(state_a: &QuantumState, state_b: &QuantumState) -> f64 {
        let reg_a = state_a.to_register(1);
        let reg_b = state_b.to_register(2);
        interference::net_interference(&reg_a, &reg_b)
    }
}
