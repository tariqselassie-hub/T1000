// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! A variational quantum circuit classifier for tabular data.
//!
//! Angle encoding of one feature per qubit, a hardware-efficient Ry/Rz + CZ-ring ansatz, a
//! mean-Z readout in `[-1, 1]`, and full-batch Adam on the squared error against labels of `+1`
//! or `-1`, with exact adjoint gradients. Shared by the `csv_fold_vqc` binary and the Python
//! bindings, so both train the same model.

use crate::{
    AdamOptimizer, CircuitAnsatz, DataEncodingLayer, EncodingKind, ParameterShiftEngine,
    PauliObservable,
};
use rayon::prelude::*;

/// Training settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VqcConfig {
    /// Qubits, equal to the number of input features.
    pub n_qubits: usize,
    /// Ansatz layers.
    pub depth: usize,
    /// Full-batch Adam epochs.
    pub epochs: usize,
    /// Adam learning rate (decayed by 0.9 every 10 epochs).
    pub lr: f64,
    /// Seed of the initial ansatz weights.
    pub seed: u64,
}

impl Default for VqcConfig {
    fn default() -> Self {
        VqcConfig {
            n_qubits: 8,
            depth: 3,
            epochs: 30,
            lr: 0.05,
            seed: 11,
        }
    }
}

/// A trained classifier: the circuit and its weights.
#[derive(Debug, Clone)]
pub struct TrainedVqc {
    pub engine: ParameterShiftEngine,
    pub weights: Vec<f64>,
    /// Mean squared-error loss of the last epoch, `0.5 (y_pred - y)^2` averaged over samples.
    pub final_loss: f64,
}

impl TrainedVqc {
    /// Readout `<Z>` in `[-1, 1]` for one sample; its sign is the predicted class.
    pub fn decision(&self, x: &[f64]) -> f64 {
        self.engine.predict(x, &self.weights)
    }

    /// Readouts for many samples, in parallel.
    pub fn decisions(&self, xs: &[Vec<f64>]) -> Vec<f64> {
        xs.par_iter().map(|x| self.decision(x)).collect()
    }

    /// Number of trainable parameters.
    pub fn parameter_count(&self) -> usize {
        self.weights.len()
    }
}

/// Trains a fresh circuit on rows `xs` (each of length `cfg.n_qubits`) with labels `ys` in {+1, -1}.
///
/// # Panics
/// If `xs` is empty, the row lengths differ from `cfg.n_qubits`, or `xs` and `ys` differ in length.
pub fn train_vqc(xs: &[Vec<f64>], ys: &[f64], cfg: &VqcConfig) -> TrainedVqc {
    assert!(!xs.is_empty(), "no training samples");
    assert_eq!(xs.len(), ys.len(), "features and labels differ in length");
    assert!(
        xs.iter().all(|x| x.len() == cfg.n_qubits),
        "every row needs exactly n_qubits = {} features",
        cfg.n_qubits
    );
    let encoding = DataEncodingLayer::new(cfg.n_qubits, EncodingKind::Angle);
    let ansatz = CircuitAnsatz::new_seeded(cfg.n_qubits, cfg.depth, cfg.seed);
    let n_params = ansatz.param_count();
    let mut weights = ansatz.weights.clone();
    let mut adam = AdamOptimizer::new(n_params, cfg.lr);
    let engine = ParameterShiftEngine::new(encoding, ansatz, PauliObservable::GlobalMeanZ);
    let n = xs.len() as f64;
    let mut final_loss = 0.0;
    for ep in 0..cfg.epochs {
        adam.step_epoch_decay(ep, 0.9, 10);
        let (sum_loss, sum_grad) = xs
            .par_iter()
            .zip(ys)
            .map(|(x, y)| {
                let (_, l, g) = engine.compute_loss_gradients(x, *y, &weights);
                (l, g)
            })
            .reduce(
                || (0.0, vec![0.0; n_params]),
                |mut a, b| {
                    a.0 += b.0;
                    for (u, v) in a.1.iter_mut().zip(b.1) {
                        *u += v;
                    }
                    a
                },
            );
        final_loss = sum_loss / n;
        let grad: Vec<f64> = sum_grad.iter().map(|g| g / n).collect();
        adam.step(&mut weights, &grad);
    }
    TrainedVqc {
        engine,
        weights,
        final_loss,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_a_separable_one_feature_rule() {
        // Label +1 when the first angle is small, -1 when it is large.
        let xs: Vec<Vec<f64>> = (0..40)
            .map(|k| {
                let a = if k % 2 == 0 { 0.3 } else { 2.8 };
                vec![a + 0.01 * f64::from(k), 0.5]
            })
            .collect();
        let ys: Vec<f64> = (0..40)
            .map(|k| if k % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let cfg = VqcConfig {
            n_qubits: 2,
            depth: 2,
            epochs: 60,
            lr: 0.1,
            seed: 3,
        };
        let model = train_vqc(&xs, &ys, &cfg);
        let correct = model
            .decisions(&xs)
            .iter()
            .zip(&ys)
            .filter(|(d, y)| (**d >= 0.0) == (**y > 0.0))
            .count();
        assert!(
            correct >= 36,
            "{correct} of 40 correct, loss {}",
            model.final_loss
        );
    }

    #[test]
    fn training_is_deterministic_for_a_seed() {
        let xs = vec![vec![0.1, 1.0], vec![2.0, 0.4], vec![1.2, 2.9]];
        let ys = vec![1.0, -1.0, 1.0];
        let cfg = VqcConfig {
            n_qubits: 2,
            depth: 1,
            epochs: 5,
            ..VqcConfig::default()
        };
        let a = train_vqc(&xs, &ys, &cfg);
        let b = train_vqc(&xs, &ys, &cfg);
        assert_eq!(a.weights, b.weights);
    }
}
