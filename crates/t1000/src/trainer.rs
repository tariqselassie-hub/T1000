//! One configurable hybrid QNN for molecular property prediction, trained under the
//! evaluation protocol ([`crate::protocol`]).
//!
//! Model: standardized features -> encoding -> parameterized circuit -> linear readout of
//! expectation values -> standardized (logP, QED, SAS). The circuit mechanisms explored in the
//! earlier experiment programs are options of this single model:
//!
//! | Option | Values |
//! |---|---|
//! | `encoding` | `amplitude` (features + bias term as amplitudes) or `angle` (fixed seeded random projection of the features to one Ry angle per qubit) |
//! | `pi_layer` | +/-pi Rz phase layer before the trainable layers |
//! | `readout` | `z` (all <Z_q>) or `z+zz` (plus all two-body <Z_i Z_j>) |
//! | `reupload` | re-inject the projected features as Ry(s_q * u_q) between layers, with trainable scales s_q |
//! | `n_qubits`, `depth` | circuit size |
//!
//! Not available here: mid-circuit feed-forward (angles depend on a measured expectation, which
//! the adjoint method does not differentiate through) and the quantum natural gradient optimizer.
//!
//! Training: Adam (no angle wrapping) on mini-batches, gradients from [`adjoint_program`]
//! computed in parallel across molecules, gradient-norm clipping, best epoch chosen on the
//! validation split (with patience), test set evaluated once with the selected parameters.

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::adjoint::{adjoint_program, run_program, Angle, GateOp};
use crate::ansatz::entangling_pairs;
use crate::baselines::{regression_metrics, TargetMetrics, ZINC_TARGETS};
use crate::observable::PauliObservable;
use crate::optimizer::AdamOptimizer;
use crate::protocol::{DataSplit, MoleculeRecord, SplitResult, Standardizer};
use crate::state::QuantumState;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Encoding {
    Amplitude,
    Angle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Readout {
    #[serde(rename = "z")]
    Z,
    #[serde(rename = "z+zz")]
    ZZ,
}

/// Circuit architecture options.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub n_qubits: usize,
    pub depth: usize,
    pub pi_layer: bool,
    pub encoding: Encoding,
    pub readout: Readout,
    pub reupload: bool,
}

/// One experiment: features, model and training settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainConfig {
    pub name: String,
    /// A name from [`crate::protocol::FEATURE_SETS`].
    pub features: String,
    pub model: ModelConfig,
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub max_grad_norm: f64,
    /// Stop when validation has not improved for this many epochs.
    pub patience: usize,
    /// Loss weights for (logP, QED, SAS) on standardized targets.
    pub task_weights: [f64; 3],
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            name: "qnn".to_string(),
            features: "descriptors".to_string(),
            model: ModelConfig {
                n_qubits: 12,
                depth: 3,
                pi_layer: true,
                encoding: Encoding::Amplitude,
                readout: Readout::ZZ,
                reupload: false,
            },
            epochs: 20,
            batch_size: 64,
            lr: 0.02,
            max_grad_norm: 5.0,
            patience: 5,
            task_weights: [1.0, 1.0, 1.0],
        }
    }
}

/// The trainable model. Parameter layout:
/// `[circuit weights (2 * n * depth) | re-upload scales (n, if enabled) | readout W (3 x n_obs) | readout b (3)]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HybridModel {
    pub config: ModelConfig,
    pub n_inputs: usize,
    /// Fixed `n_qubits x n_inputs` projection used by angle encoding and re-uploading.
    pub projection: Vec<Vec<f64>>,
    pub params: Vec<f64>,
}

impl HybridModel {
    pub fn new(config: ModelConfig, n_inputs: usize, seed: u64) -> Self {
        assert!(config.n_qubits >= 2, "need at least 2 qubits");
        // depth 0 is allowed: the circuit is then only the encoding (and optional pi layer),
        // which is the "identity circuit" control.
        if config.encoding == Encoding::Amplitude {
            assert!(
                n_inputs + 1 <= 1 << config.n_qubits,
                "amplitude encoding of {} features (+ bias) needs more than {} qubits",
                n_inputs,
                config.n_qubits
            );
        }
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let norm = 1.0 / (n_inputs.max(1) as f64).sqrt();
        // Gaussian entries via Box-Muller, scaled so projected values are O(1).
        let mut gauss = || {
            let (u1, u2): (f64, f64) = (rng.gen_range(1e-12..1.0), rng.gen_range(0.0..1.0));
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        let projection: Vec<Vec<f64>> =
            (0..config.n_qubits).map(|_| (0..n_inputs).map(|_| gauss() * norm).collect()).collect();
        let mut model = Self { config, n_inputs, projection, params: Vec::new() };
        let n_params = model.readout_bias_offset() + 3;
        let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(1));
        let mut params = vec![0.0; n_params];
        for p in params[..model.n_circuit()].iter_mut() {
            *p = rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI);
        }
        for p in params[model.n_circuit()..model.readout_offset()].iter_mut() {
            *p = 1.0; // re-upload scales
        }
        for p in params[model.readout_offset()..model.readout_bias_offset()].iter_mut() {
            *p = rng.gen_range(-0.01..0.01);
        }
        model.params = params;
        model
    }

    fn n_circuit(&self) -> usize {
        2 * self.config.n_qubits * self.config.depth
    }

    fn n_scales(&self) -> usize {
        if self.config.reupload { self.config.n_qubits } else { 0 }
    }

    pub fn n_observables(&self) -> usize {
        let n = self.config.n_qubits;
        match self.config.readout {
            Readout::Z => n,
            Readout::ZZ => n + n * (n - 1) / 2,
        }
    }

    fn readout_offset(&self) -> usize {
        self.n_circuit() + self.n_scales()
    }

    fn readout_bias_offset(&self) -> usize {
        self.readout_offset() + 3 * self.n_observables()
    }

    pub fn observables(&self) -> Vec<PauliObservable> {
        let n = self.config.n_qubits;
        let mut obs: Vec<PauliObservable> = (0..n).map(PauliObservable::Z).collect();
        if self.config.readout == Readout::ZZ {
            for i in 1..n {
                for j in 0..i {
                    obs.push(PauliObservable::ZZ(i, j));
                }
            }
        }
        obs
    }

    fn project(&self, x: &[f64]) -> Vec<f64> {
        self.projection.iter().map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum()).collect()
    }

    /// Initial state and gate program for one standardized input.
    fn circuit(&self, x: &[f64]) -> (QuantumState, Vec<GateOp>) {
        let n = self.config.n_qubits;
        let dim = 1usize << n;
        let needs_projection = self.config.encoding == Encoding::Angle || self.config.reupload;
        let u = if needs_projection { self.project(x) } else { Vec::new() };
        let mut ops = Vec::new();

        let mut state = QuantumState::zero(dim);
        match self.config.encoding {
            Encoding::Amplitude => {
                // Features plus a constant 1 so the overall feature scale is not normalised away.
                for (i, v) in x.iter().chain(std::iter::once(&1.0)).enumerate().take(dim) {
                    state.real[i] = *v;
                }
                state.normalize();
            }
            Encoding::Angle => {
                state.real[0] = 1.0;
                for (q, &uq) in u.iter().enumerate() {
                    ops.push(GateOp::Ry { qubit: q, angle: Angle::fixed(uq) });
                }
            }
        }
        if self.config.pi_layer {
            for q in 0..n {
                let sign = if q % 2 == 0 { 1.0 } else { -1.0 };
                ops.push(GateOp::Rz { qubit: q, angle: Angle::fixed(sign * std::f64::consts::PI) });
            }
        }
        for layer in 0..self.config.depth {
            let base = layer * 2 * n;
            for q in 0..n {
                ops.push(GateOp::Ry { qubit: q, angle: Angle::param(base + 2 * q) });
                ops.push(GateOp::Rz { qubit: q, angle: Angle::param(base + 2 * q + 1) });
            }
            for (a, b) in entangling_pairs(n, layer) {
                ops.push(GateOp::Cz(a, b));
            }
            if self.config.reupload && layer + 1 < self.config.depth {
                for (q, &uq) in u.iter().enumerate() {
                    ops.push(GateOp::Ry { qubit: q, angle: Angle::scaled(self.n_circuit() + q, uq) });
                }
            }
        }
        (state, ops)
    }

    fn readout(&self, z: &[f64]) -> [f64; 3] {
        let (w, b) = (self.readout_offset(), self.readout_bias_offset());
        let n_obs = self.n_observables();
        let mut out = [0.0; 3];
        for (t, o) in out.iter_mut().enumerate() {
            *o = self.params[b + t] + (0..n_obs).map(|k| self.params[w + t * n_obs + k] * z[k]).sum::<f64>();
        }
        out
    }

    /// Expectation values of the readout observables for one standardized input (the features
    /// the linear readout sees).
    pub fn observable_features(&self, x: &[f64]) -> Vec<f64> {
        let (state, ops) = self.circuit(x);
        let out = run_program(&state, &ops, &self.params);
        self.observables().iter().map(|o| o.expectation_value(&out)).collect()
    }

    /// Number of trainable parameters (circuit + re-upload scales + readout).
    pub fn n_params(&self) -> usize {
        self.params.len()
    }

    /// Prediction in standardized target units.
    pub fn predict(&self, x: &[f64]) -> [f64; 3] {
        self.readout(&self.observable_features(x))
    }

    /// Weighted squared error `sum_t w_t (pred_t - y_t)^2 / sum_t w_t` and its gradient.
    pub fn loss_and_gradient(&self, x: &[f64], y: &[f64; 3], task_weights: &[f64; 3]) -> (f64, Vec<f64>) {
        let (state, ops) = self.circuit(x);
        let obs = self.observables();
        let n_obs = obs.len();
        let wsum: f64 = task_weights.iter().sum();
        let d_pred = |z: &[f64]| -> [f64; 3] {
            let p = self.readout(z);
            let mut d = [0.0; 3];
            for t in 0..3 {
                d[t] = 2.0 * task_weights[t] * (p[t] - y[t]) / wsum;
            }
            d
        };
        let w_off = self.readout_offset();
        let res = adjoint_program(&state, &ops, &self.params, &obs, |z| {
            let d = d_pred(z);
            (0..n_obs).map(|k| (0..3).map(|t| d[t] * self.params[w_off + t * n_obs + k]).sum()).collect()
        });
        let z = &res.expectations;
        let p = self.readout(z);
        let loss = (0..3).map(|t| task_weights[t] * (p[t] - y[t]).powi(2)).sum::<f64>() / wsum;
        let d = d_pred(z);
        let mut grad = res.gradient;
        let b_off = self.readout_bias_offset();
        for t in 0..3 {
            grad[b_off + t] += d[t];
            for k in 0..n_obs {
                grad[w_off + t * n_obs + k] += d[t] * z[k];
            }
        }
        (loss, grad)
    }
}

/// Per-epoch training record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpochRecord {
    pub epoch: usize,
    pub train_loss: f64,
    /// Mean over targets of validation MSE / variance (standardized units); lower is better.
    pub valid_score: f64,
    pub seconds: f64,
}

/// One training run on one split.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainRun {
    pub split: String,
    pub seed: u64,
    pub best_epoch: usize,
    pub epochs_run: usize,
    pub history: Vec<EpochRecord>,
    /// Validation metrics at the selected epoch (original units).
    pub valid: Vec<TargetMetrics>,
    /// Test metrics at the selected epoch (original units), evaluated once.
    pub test: SplitResult,
    pub n_params: usize,
    /// Parameters at the selected epoch.
    pub params: Vec<f64>,
}

fn predict_rows(model: &HybridModel, xs: &[Vec<f64>]) -> Vec<[f64; 3]> {
    crate::parallel::install(|| xs.par_iter().map(|x| model.predict(x)).collect())
}

fn score(preds: &[[f64; 3]], ys: &[[f64; 3]]) -> f64 {
    (0..3)
        .map(|t| {
            let n = ys.len() as f64;
            let mean = ys.iter().map(|y| y[t]).sum::<f64>() / n;
            let var = (ys.iter().map(|y| (y[t] - mean).powi(2)).sum::<f64>() / n).max(1e-300);
            preds.iter().zip(ys).map(|(p, y)| (p[t] - y[t]).powi(2)).sum::<f64>() / n / var
        })
        .sum::<f64>()
        / 3.0
}

/// Trains `cfg` on `split` with `features` (one row per pool molecule, in pool order).
/// `max_train`, if set, subsamples the training rows (for quick checks; not a protocol result).
pub fn train_on_split(
    cfg: &TrainConfig,
    features: &[Vec<f64>],
    pool: &[MoleculeRecord],
    split: &DataSplit,
    max_train: Option<usize>,
    log: impl Fn(&str),
) -> Result<TrainRun, String> {
    if features.len() != pool.len() {
        return Err(format!("{} feature rows for a pool of {}", features.len(), pool.len()));
    }
    let idx = split.indices(pool)?;
    let mut train_ids = idx.train.clone();
    if let Some(m) = max_train {
        train_ids.truncate(m);
    }
    let x_scaler = Standardizer::fit(train_ids.iter().map(|&i| features[i].as_slice()));
    let y_scaler = Standardizer::fit(train_ids.iter().map(|&i| pool[i].targets.as_slice()));
    let xs = |ids: &[usize]| -> Vec<Vec<f64>> { ids.iter().map(|&i| x_scaler.transform(&features[i])).collect() };
    let ys = |ids: &[usize]| -> Vec<[f64; 3]> {
        ids.iter()
            .map(|&i| {
                let v = y_scaler.transform(&pool[i].targets);
                [v[0], v[1], v[2]]
            })
            .collect()
    };
    let (x_tr, y_tr) = (xs(&train_ids), ys(&train_ids));
    let (x_va, y_va) = (xs(&idx.valid), ys(&idx.valid));
    let x_te = xs(&idx.test);

    let mut model = HybridModel::new(cfg.model.clone(), x_tr[0].len(), split.seed);
    let mut opt = AdamOptimizer::new_unwrapped(model.params.len(), cfg.lr);
    let mut rng = ChaCha8Rng::seed_from_u64(split.seed.wrapping_add(17));
    let mut order: Vec<usize> = (0..x_tr.len()).collect();

    let mut best = (f64::INFINITY, 0usize, model.params.clone());
    let mut history = Vec::new();
    for epoch in 1..=cfg.epochs {
        let t0 = Instant::now();
        order.shuffle(&mut rng);
        let mut epoch_loss = 0.0;
        for batch in order.chunks(cfg.batch_size.max(1)) {
            let per_sample: Vec<(f64, Vec<f64>)> = crate::parallel::install(|| {
                batch
                    .par_iter()
                    .map(|&i| model.loss_and_gradient(&x_tr[i], &y_tr[i], &cfg.task_weights))
                    .collect()
            });
            let mut grad = vec![0.0; model.params.len()];
            for (loss, g) in &per_sample {
                epoch_loss += loss;
                for (a, b) in grad.iter_mut().zip(g) {
                    *a += b / batch.len() as f64;
                }
            }
            let norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            if cfg.max_grad_norm > 0.0 && norm > cfg.max_grad_norm {
                grad.iter_mut().for_each(|g| *g *= cfg.max_grad_norm / norm);
            }
            opt.step(&mut model.params, &grad);
        }
        let valid_score = score(&predict_rows(&model, &x_va), &y_va);
        let rec = EpochRecord {
            epoch,
            train_loss: epoch_loss / x_tr.len() as f64,
            valid_score,
            seconds: t0.elapsed().as_secs_f64(),
        };
        log(&format!(
            "    epoch {:>3}: train loss {:.4} | valid score {:.4} | {:.1} s",
            rec.epoch, rec.train_loss, rec.valid_score, rec.seconds
        ));
        history.push(rec);
        if valid_score < best.0 {
            best = (valid_score, epoch, model.params.clone());
        } else if epoch - best.1 >= cfg.patience.max(1) {
            break;
        }
    }

    model.params = best.2;
    let to_units = |preds: Vec<[f64; 3]>| -> Vec<Vec<f64>> { preds.iter().map(|p| y_scaler.inverse(p)).collect() };
    let truth = |ids: &[usize]| -> Vec<Vec<f64>> { ids.iter().map(|&i| pool[i].targets.to_vec()).collect() };
    let valid = regression_metrics(&to_units(predict_rows(&model, &x_va)), &truth(&idx.valid), &ZINC_TARGETS);
    let test_metrics = regression_metrics(&to_units(predict_rows(&model, &x_te)), &truth(&idx.test), &ZINC_TARGETS);
    Ok(TrainRun {
        split: split.name.clone(),
        seed: split.seed,
        best_epoch: best.1,
        epochs_run: history.len(),
        history,
        valid,
        test: SplitResult {
            split: split.name.clone(),
            features: cfg.features.clone(),
            model: cfg.name.clone(),
            lambda: None,
            k: None,
            test: test_metrics,
        },
        n_params: model.params.len(),
        params: model.params.clone(),
    })
}

/// Standardized inputs (fitted on training rows) and original-unit targets for one split.
pub struct PreparedSplit {
    pub x_train: Vec<Vec<f64>>,
    pub x_valid: Vec<Vec<f64>>,
    pub x_test: Vec<Vec<f64>>,
    pub y_train: Vec<Vec<f64>>,
    pub y_valid: Vec<Vec<f64>>,
    pub y_test: Vec<Vec<f64>>,
}

/// Same input standardization as [`train_on_split`], for building controls on identical data.
pub fn prepare_split(features: &[Vec<f64>], pool: &[MoleculeRecord], split: &DataSplit) -> Result<PreparedSplit, String> {
    if features.len() != pool.len() {
        return Err(format!("{} feature rows for a pool of {}", features.len(), pool.len()));
    }
    let idx = split.indices(pool)?;
    let scaler = Standardizer::fit(idx.train.iter().map(|&i| features[i].as_slice()));
    let xs = |ids: &[usize]| ids.iter().map(|&i| scaler.transform(&features[i])).collect();
    let ys = |ids: &[usize]| ids.iter().map(|&i| pool[i].targets.to_vec()).collect();
    Ok(PreparedSplit {
        x_train: xs(&idx.train),
        x_valid: xs(&idx.valid),
        x_test: xs(&idx.test),
        y_train: ys(&idx.train),
        y_valid: ys(&idx.valid),
        y_test: ys(&idx.test),
    })
}

/// Observable features of `model` for each row, computed in parallel.
pub fn observable_feature_rows(model: &HybridModel, xs: &[Vec<f64>]) -> Vec<Vec<f64>> {
    crate::parallel::install(|| xs.par_iter().map(|x| model.observable_features(x)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config(encoding: Encoding, readout: Readout, reupload: bool) -> ModelConfig {
        ModelConfig { n_qubits: 3, depth: 2, pi_layer: true, encoding, readout, reupload }
    }

    #[test]
    fn model_gradient_matches_finite_differences() {
        for (encoding, readout, reupload) in [
            (Encoding::Amplitude, Readout::Z, false),
            (Encoding::Amplitude, Readout::ZZ, true),
            (Encoding::Angle, Readout::ZZ, true),
        ] {
            let mut model = HybridModel::new(tiny_config(encoding, readout, reupload), 5, 3);
            let mut rng = ChaCha8Rng::seed_from_u64(9);
            let readout_start = model.readout_offset();
            for p in model.params.iter_mut().skip(readout_start) {
                *p = rng.gen_range(-0.5..0.5);
            }
            let x = [0.3, -1.2, 0.7, 2.0, -0.4];
            let y = [0.5, -0.2, 1.1];
            let w = [1.0, 2.0, 0.5];
            let (loss, grad) = model.loss_and_gradient(&x, &y, &w);
            let loss_at = |m: &HybridModel| {
                let p = m.predict(&x);
                (0..3).map(|t| w[t] * (p[t] - y[t]).powi(2)).sum::<f64>() / 3.5
            };
            assert!((loss - loss_at(&model)).abs() < 1e-12);
            for j in 0..model.params.len() {
                let h = 1e-6;
                let mut mp = model.clone();
                mp.params[j] += h;
                let mut mm = model.clone();
                mm.params[j] -= h;
                let fd = (loss_at(&mp) - loss_at(&mm)) / (2.0 * h);
                assert!(
                    (grad[j] - fd).abs() < 1e-6,
                    "{encoding:?}/{readout:?}/reupload={reupload} param {j}: {} vs FD {fd}",
                    grad[j]
                );
            }
        }
    }

    #[test]
    fn parameter_layout_and_observables() {
        let m = HybridModel::new(
            ModelConfig { n_qubits: 4, depth: 3, pi_layer: false, encoding: Encoding::Amplitude, readout: Readout::ZZ, reupload: true },
            10,
            0,
        );
        assert_eq!(m.n_observables(), 4 + 6);
        assert_eq!(m.params.len(), 2 * 4 * 3 + 4 + 3 * 10 + 3);
        assert_eq!(m.observables().len(), m.n_observables());
        let again = HybridModel::new(m.config.clone(), 10, 0);
        assert_eq!(m, again, "construction is deterministic for a seed");
    }

    #[test]
    fn identity_circuit_reads_out_the_encoded_state() {
        // depth 0, no pi layer: the features are exactly the Z / ZZ expectations of the
        // amplitude-encoded input (features + bias, normalised).
        let m = HybridModel::new(
            ModelConfig { n_qubits: 3, depth: 0, pi_layer: false, encoding: Encoding::Amplitude, readout: Readout::ZZ, reupload: false },
            4,
            0,
        );
        assert_eq!(m.n_params(), 3 * 6 + 3, "only readout parameters");
        let x = [0.5, -1.0, 2.0, 0.25];
        let mut amps: Vec<f64> = x.iter().copied().chain([1.0]).chain([0.0; 3]).collect();
        let norm = amps.iter().map(|a| a * a).sum::<f64>().sqrt();
        amps.iter_mut().for_each(|a| *a /= norm);
        let z = |q: usize| (0..8).map(|i| if (i >> q) & 1 == 0 { amps[i] * amps[i] } else { -amps[i] * amps[i] }).sum::<f64>();
        let zz = |a: usize, b: usize| {
            (0..8).map(|i| if ((i >> a) ^ (i >> b)) & 1 == 0 { amps[i] * amps[i] } else { -amps[i] * amps[i] }).sum::<f64>()
        };
        let want = [z(0), z(1), z(2), zz(1, 0), zz(2, 0), zz(2, 1)];
        let got = m.observable_features(&x);
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-12, "{got:?} vs {want:?}");
        }
    }

    #[test]
    fn training_reduces_loss_on_a_learnable_toy_task() {
        // Targets are smooth functions of the features; a few epochs must beat the mean predictor
        // on validation, and the run must be deterministic.
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let pool: Vec<MoleculeRecord> = (0..200)
            .map(|i| {
                let f: Vec<f64> = (0..4).map(|_| rng.gen_range(-1.0..1.0)).collect();
                let t = [f[0] + 0.5 * f[1], (f[2] * 2.0).tanh(), f[3] - f[0]];
                MoleculeRecord { smiles: format!("C{i}"), features: f, targets: t }
            })
            .collect();
        let smiles: Vec<String> = pool.iter().map(|r| r.smiles.clone()).collect();
        let split = crate::protocol::make_split(
            "toy",
            &smiles,
            None,
            crate::protocol::SplitMethod::Random,
            0,
            crate::protocol::SplitFractions::default(),
        );
        let features: Vec<Vec<f64>> = pool.iter().map(|r| r.features.clone()).collect();
        let cfg = TrainConfig {
            features: "toy".to_string(),
            model: ModelConfig { n_qubits: 3, depth: 2, pi_layer: false, encoding: Encoding::Amplitude, readout: Readout::ZZ, reupload: false },
            epochs: 15,
            batch_size: 16,
            lr: 0.05,
            patience: 15,
            ..TrainConfig::default()
        };
        let run = train_on_split(&cfg, &features, &pool, &split, None, |_| {}).unwrap();
        let first = run.history[0].valid_score;
        let best = run.history.iter().map(|r| r.valid_score).fold(f64::INFINITY, f64::min);
        assert!(best < first, "validation score did not improve: {first} -> {best}");
        assert!(best < 0.9, "model should beat the mean predictor (score 1.0), got {best}");
        let again = train_on_split(&cfg, &features, &pool, &split, None, |_| {}).unwrap();
        let curve = |r: &TrainRun| r.history.iter().map(|e| (e.epoch, e.train_loss, e.valid_score)).collect::<Vec<_>>();
        assert_eq!(curve(&run), curve(&again), "training must be deterministic");
        assert_eq!(run.test.test, again.test.test);
    }
}
