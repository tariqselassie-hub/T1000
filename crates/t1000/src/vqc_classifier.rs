// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! A variational quantum circuit classifier for tabular data.
//!
//! Angle encoding of one feature per qubit, a hardware-efficient Ry/Rz + CZ-ring ansatz, a
//! mean-Z readout in `[-1, 1]`, and full-batch Adam on the squared error against labels of `+1`
//! or `-1`, with exact adjoint gradients. Shared by the `csv_fold_vqc` binary and the Python
//! bindings, so both train the same model.

use crate::{
    quantum_fisher_information, AdamOptimizer, AdjointWorkspace, CircuitAnsatz, DataEncodingLayer,
    EncodingKind, ParameterShiftEngine, PauliObservable, QuantumNaturalGradient, QuantumState,
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

/// Optimizer used by [`train_vqc_with`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VqcOptimizer {
    /// Adam on the adjoint gradient.
    Adam,
    /// Quantum natural gradient: precondition the batch gradient by the batch-averaged, damped
    /// quantum Fisher information.
    QuantumNatural {
        /// Tikhonov damping added to the metric diagonal.
        damping: f64,
    },
}

/// Trains a fresh circuit on rows `xs` (each of length `cfg.n_qubits`) with labels `ys` in {+1, -1}.
///
/// # Panics
/// If `xs` is empty, the row lengths differ from `cfg.n_qubits`, or `xs` and `ys` differ in length.
pub fn train_vqc(xs: &[Vec<f64>], ys: &[f64], cfg: &VqcConfig) -> TrainedVqc {
    train_vqc_with(xs, ys, cfg, VqcOptimizer::Adam)
}

/// Trains with quantum natural gradient and the given metric damping.
pub fn train_vqc_qng(xs: &[Vec<f64>], ys: &[f64], cfg: &VqcConfig, damping: f64) -> TrainedVqc {
    train_vqc_with(xs, ys, cfg, VqcOptimizer::QuantumNatural { damping })
}

/// Trains with an explicit optimizer choice.
pub fn train_vqc_with(
    xs: &[Vec<f64>],
    ys: &[f64],
    cfg: &VqcConfig,
    optimizer: VqcOptimizer,
) -> TrainedVqc {
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
    let dim = engine.ansatz.dim;
    let nq = engine.ansatz.n_qubits;
    let n = xs.len() as f64;
    let mut qng = match optimizer {
        VqcOptimizer::Adam => None,
        VqcOptimizer::QuantumNatural { damping } => {
            Some(QuantumNaturalGradient::new(cfg.lr).with_damping(damping))
        }
    };
    let mut final_loss = 0.0;
    for ep in 0..cfg.epochs {
        adam.step_epoch_decay(ep, 0.9, 10);
        if let Some(qng) = qng.as_mut() {
            // Build the gradient and the quantum Fisher information in the same sample pass.
            let (sum_loss, sum_grad, sum_metric) = xs
                .par_iter()
                .zip(ys)
                .map_init(
                    || (AdjointWorkspace::new(dim, nq), QuantumState::zero(dim)),
                    |(ws, input_buf), (x, y)| {
                        engine.encoding.encode_into(x, input_buf);
                        let (_, l, out) = engine
                            .compute_loss_gradients_for_state_into(input_buf, *y, &weights, ws);
                        let metric =
                            quantum_fisher_information(&engine.ansatz, input_buf, &weights);
                        (l, out.gradient.to_vec(), metric)
                    },
                )
                .reduce(
                    || (0.0, vec![0.0; n_params], vec![0.0; n_params * n_params]),
                    |mut a, b| {
                        a.0 += b.0;
                        for (u, v) in a.1.iter_mut().zip(b.1) {
                            *u += v;
                        }
                        for (u, v) in a.2.iter_mut().zip(b.2) {
                            *u += v;
                        }
                        a
                    },
                );
            final_loss = sum_loss / n;
            let inv_n = 1.0 / n;
            let grad: Vec<f64> = sum_grad.iter().map(|g| g * inv_n).collect();
            let metric: Vec<f64> = sum_metric.iter().map(|m| m * inv_n).collect();
            // Mirror Adam's exponential step decay (0.9 every 10 epochs).
            qng.lr = cfg.lr * 0.9f64.powi((ep / 10) as i32);
            qng.step_with_metric(&mut weights, &grad, &metric);
        } else {
            let (sum_loss, sum_grad) = xs
                .par_iter()
                .zip(ys)
                .map_init(
                    || AdjointWorkspace::new(dim, nq),
                    |ws, (x, y)| {
                        // Fused encoding + adjoint: no intermediate encoded state is allocated.
                        let (_, l, out) = engine.compute_loss_gradients_into(x, *y, &weights, ws);
                        (l, out.gradient.to_vec())
                    },
                )
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

    #[test]
    fn quantum_natural_gradient_training_reduces_loss() {
        // Learnable rule: label +1 for a small first angle, -1 for a large one.
        let xs: Vec<Vec<f64>> = (0..24)
            .map(|k| {
                let a = if k < 12 {
                    0.2 + 0.05 * f64::from(k)
                } else {
                    2.2 + 0.05 * f64::from(k - 12)
                };
                vec![a, 0.5 + 0.1 * f64::from(k % 4)]
            })
            .collect();
        let ys: Vec<f64> = (0..24).map(|k| if k < 12 { 1.0 } else { -1.0 }).collect();
        let cfg = VqcConfig {
            n_qubits: 2,
            depth: 1,
            epochs: 30,
            lr: 0.05,
            seed: 5,
        };

        let engine = ParameterShiftEngine::new(
            DataEncodingLayer::new(2, EncodingKind::Angle),
            CircuitAnsatz::new_seeded(2, 1, cfg.seed),
            PauliObservable::GlobalMeanZ,
        );
        let initial_loss: f64 = xs
            .iter()
            .zip(&ys)
            .map(|(x, y)| 0.5 * (engine.predict(x, &engine.ansatz.weights) - y).powi(2))
            .sum::<f64>()
            / xs.len() as f64;

        let model = train_vqc_qng(&xs, &ys, &cfg, 1e-2);
        assert!(model.final_loss.is_finite());
        assert!(
            model.final_loss < 0.75 * initial_loss,
            "QNG did not reduce the loss: {initial_loss} -> {}",
            model.final_loss
        );
        // Same seed and settings must reproduce exactly.
        let again = train_vqc_qng(&xs, &ys, &cfg, 1e-2);
        assert_eq!(model.weights, again.weights);
    }
}
