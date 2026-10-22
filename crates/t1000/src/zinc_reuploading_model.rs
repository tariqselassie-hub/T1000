// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! The 12-qubit data re-uploading ZINC model and the pieces an optimizer comparison needs:
//! readout-weight fitting, a finite-difference gradient, the Fubini-Study diagonal metric, an
//! Adam step (optionally preconditioned by that metric) and holdout scoring.

use crate::zinc_data::ZincMolecule;
use crate::zinc_training_common::{compute_mae_and_r2_single, solve_ridge_cholesky};
use crate::{CircuitAnsatz, DataEncodingLayer, QuantumState};
use serde::{Deserialize, Serialize};

pub const N_QUBITS: usize = 12;
pub const DIM: usize = 1 << N_QUBITS; // 4,096
/// Qubit whose Z expectation drives the mid-circuit feed-forward.
pub const SOURCE_QUBIT: usize = 8;

/// The circuit, its input encoding and the (fixed) feed-forward strengths.
#[derive(Debug)]
pub struct Model<'a> {
    pub ansatz: &'a CircuitAnsatz,
    pub encoding: &'a DataEncodingLayer,
    pub feedforward_alphas: &'a [f64],
}

impl Model<'_> {
    /// Evolves `state` through the re-uploading circuit and writes the 12 Z expectations to `z`.
    ///
    /// `state` is evolved in place and is *not* reset: encode into it first
    /// (`self.encoding.encode_amplitude_into`) unless you mean to continue from its contents.
    pub fn run(
        &self,
        state: &mut QuantumState,
        params: &[f64],
        dru_scales: &[f64],
        features: &[f64],
        z: &mut [f64; N_QUBITS],
    ) {
        self.ansatz.forward_reuploading_in_place(
            state,
            params,
            features,
            dru_scales,
            SOURCE_QUBIT,
            self.feedforward_alphas,
            z,
        );
    }

    /// Encodes `features` into `state`, then [`run`](Self::run)s the circuit.
    pub fn encode_and_run(
        &self,
        state: &mut QuantumState,
        params: &[f64],
        dru_scales: &[f64],
        features: &[f64],
        z: &mut [f64; N_QUBITS],
    ) {
        self.encoding.encode_amplitude_into(features, state);
        self.run(state, params, dru_scales, features, z);
    }
}

// ── Readout weights ───────────────────────────────────────────────────────────

/// Ridge-fits the readout on `batch` for the current circuit parameters.
///
/// Returns the mean standardized squared error, the weights (logP, QED, SAS rows) and biases.
#[allow(clippy::needless_range_loop)] // index form mirrors the maths
pub fn eval_loss_and_weights(
    model: &Model,
    params: &[f64],
    dru_scales: &[f64],
    batch: &[ZincMolecule],
    lambda_ridge: f64,
) -> (f64, Vec<Vec<f64>>, Vec<f64>) {
    let mut z_mat = Vec::with_capacity(batch.len());
    let mut l2_state = QuantumState::new(DIM);
    let mut z = [0.0; N_QUBITS];
    for s in batch {
        model.encode_and_run(&mut l2_state, params, dru_scales, &s.features, &mut z);
        z_mat.push(z.to_vec());
    }
    let t_logp: Vec<f64> = batch.iter().map(|s| s.log_p).collect();
    let t_qed: Vec<f64> = batch.iter().map(|s| s.qed).collect();
    let t_sas: Vec<f64> = batch.iter().map(|s| s.sas).collect();

    let (w_logp, b_logp) = solve_ridge_cholesky(&z_mat, &t_logp, lambda_ridge);
    let (w_qed, b_qed) = solve_ridge_cholesky(&z_mat, &t_qed, lambda_ridge);
    let (w_sas, b_sas) = solve_ridge_cholesky(&z_mat, &t_sas, lambda_ridge);

    let n = batch.len() as f64;
    let mut total_loss = 0.0;
    for (i, row) in z_mat.iter().enumerate() {
        let mut p_logp = b_logp;
        let mut p_qed = b_qed;
        let mut p_sas = b_sas;
        for q in 0..N_QUBITS {
            p_logp += w_logp[q] * row[q];
            p_qed += w_qed[q] * row[q];
            p_sas += w_sas[q] * row[q];
        }
        let e_logp = (p_logp - t_logp[i]) / 1.5;
        let e_qed = (p_qed - t_qed[i]) / 0.15;
        let e_sas = (p_sas - t_sas[i]) / 0.8;
        total_loss += e_logp * e_logp + e_qed * e_qed + e_sas * e_sas;
    }
    (
        total_loss / n,
        vec![w_logp, w_qed, w_sas],
        vec![b_logp, b_qed, b_sas],
    )
}

// ── Holdout scoring ───────────────────────────────────────────────────────────

/// MAE and R^2 for (logP, QED, SAS).
#[derive(Debug, Clone, Copy)]
pub struct Scores {
    pub mae: [f64; 3],
    pub r2: [f64; 3],
}

/// Predicts the holdout molecules with fitted readout weights and scores them.
#[allow(clippy::needless_range_loop)] // index form mirrors the maths
pub fn holdout_scores(
    model: &Model,
    params: &[f64],
    dru_scales: &[f64],
    weights: &[Vec<f64>],
    biases: &[f64],
    test: &[ZincMolecule],
) -> Scores {
    let mut preds: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::with_capacity(test.len()));
    let mut l2_state = QuantumState::new(DIM);
    let mut z = [0.0; N_QUBITS];
    for s in test {
        model.encode_and_run(&mut l2_state, params, dru_scales, &s.features, &mut z);
        for (k, pred) in preds.iter_mut().enumerate() {
            let mut p = biases[k];
            for q in 0..N_QUBITS {
                p += weights[k][q] * z[q];
            }
            pred.push(p);
        }
    }
    let targets: [Vec<f64>; 3] = [
        test.iter().map(|s| s.log_p).collect(),
        test.iter().map(|s| s.qed).collect(),
        test.iter().map(|s| s.sas).collect(),
    ];
    let scored: Vec<(f64, f64)> = (0..3)
        .map(|k| compute_mae_and_r2_single(&preds[k], &targets[k]))
        .collect();
    Scores {
        mae: [scored[0].0, scored[1].0, scored[2].0],
        r2: [scored[0].1, scored[1].1, scored[2].1],
    }
}

/// One parameter set a molecule is tracked through.
#[derive(Debug, Clone, Copy)]
pub struct TrackedPath<'a> {
    pub params: &'a [f64],
    pub dru_scales: &'a [f64],
    /// logP readout row and bias.
    pub logp_weights: &'a [f64],
    pub logp_bias: f64,
}

/// logP predictions of one molecule under several parameter sets.
///
/// The molecule is encoded once and each path then continues from the state the previous one
/// left, so every path after the first starts from an already-evolved state. This reproduces
/// the sequencing of the original script, whose reports depend on it.
#[allow(clippy::needless_range_loop)] // index form mirrors the maths
pub fn track_molecule(model: &Model, features: &[f64], paths: &[TrackedPath]) -> Vec<f64> {
    let mut state = QuantumState::new(DIM);
    model.encoding.encode_amplitude_into(features, &mut state);
    let mut z = [0.0; N_QUBITS];
    paths
        .iter()
        .map(|p| {
            model.run(&mut state, p.params, p.dru_scales, features, &mut z);
            let mut pred = p.logp_bias;
            for q in 0..N_QUBITS {
                pred += p.logp_weights[q] * z[q];
            }
            pred
        })
        .collect()
}

// ── Gradient, metric and optimizer step ───────────────────────────────────────

/// Central finite-difference gradient of the readout-refit loss over `batch`.
#[allow(clippy::needless_range_loop)] // index form mirrors the maths
pub fn finite_difference_gradient(
    model: &Model,
    params: &[f64],
    dru_scales: &[f64],
    batch: &[ZincMolecule],
    eps: f64,
) -> Vec<f64> {
    let mut grad = vec![0.0; params.len()];
    for i in 0..params.len() {
        let mut p_plus = params.to_vec();
        p_plus[i] += eps;
        let (l_plus, _, _) = eval_loss_and_weights(model, &p_plus, dru_scales, batch, 1e-3);
        let mut p_minus = params.to_vec();
        p_minus[i] -= eps;
        let (l_minus, _, _) = eval_loss_and_weights(model, &p_minus, dru_scales, batch, 1e-3);
        grad[i] = (l_plus - l_minus) / (2.0 * eps);
    }
    grad
}

#[inline(always)]
fn state_fidelity(s1: &QuantumState, s2: &QuantumState) -> f64 {
    let mut sum_re = 0.0;
    let mut sum_im = 0.0;
    for i in 0..DIM {
        sum_re += s1.real[i] * s2.real[i] + s1.imag[i] * s2.imag[i];
        sum_im += s1.real[i] * s2.imag[i] - s1.imag[i] * s2.real[i];
    }
    sum_re * sum_re + sum_im * sum_im
}

/// The exact Fubini-Study diagonal metric tensor g_ii(θ), averaged over (up to 64 molecules of)
/// the batch: g_ii = E_x[(1 - |<ψ(θ+ε e_i)|ψ(θ-ε e_i)>|^2) / (4 ε^2)] + λ.
pub fn fubini_study_diagonal(
    model: &Model,
    circuit_params: &[f64],
    dru_scales: &[f64],
    batch: &[ZincMolecule],
    lambda_damping: f64,
) -> Vec<f64> {
    let p_len = circuit_params.len();
    let mut g_diag = vec![lambda_damping; p_len];
    let eps = 0.005;
    let eps2_4 = 4.0 * eps * eps;

    let mut state_pos = QuantumState::new(DIM);
    let mut state_neg = QuantumState::new(DIM);

    // Sample a subset of the batch for efficient Riemannian metric estimation.
    let metric_sub = &batch[..batch.len().min(64)];
    let scale_fac = 1.0 / metric_sub.len() as f64;

    for s in metric_sub {
        for i in 0..p_len {
            let mut p_pos = circuit_params.to_vec();
            p_pos[i] += eps;
            let mut p_neg = circuit_params.to_vec();
            p_neg[i] -= eps;

            let mut dummy_z = [0.0; N_QUBITS];
            model.encode_and_run(
                &mut state_pos,
                &p_pos,
                dru_scales,
                &s.features,
                &mut dummy_z,
            );
            model.encode_and_run(
                &mut state_neg,
                &p_neg,
                dru_scales,
                &s.features,
                &mut dummy_z,
            );

            let fid = state_fidelity(&state_pos, &state_neg).min(1.0);
            let g_val = (1.0 - fid) / eps2_4;
            g_diag[i] += scale_fac * g_val.max(0.0);
        }
    }
    g_diag
}

/// Circuit parameters plus their Adam moments.
#[derive(Debug, Clone)]
pub struct AdamPath {
    pub params: Vec<f64>,
    pub dru_scales: Vec<f64>,
    m: Vec<f64>,
    v: Vec<f64>,
}

impl AdamPath {
    pub fn new(params: &[f64], dru_scales: &[f64]) -> Self {
        Self {
            params: params.to_vec(),
            dru_scales: dru_scales.to_vec(),
            m: vec![0.0; params.len()],
            v: vec![0.0; params.len()],
        }
    }

    /// One Adam step (trust-region clip [-0.15, 0.15]) at 1-based `epoch`.
    ///
    /// With `metric` the gradient is first preconditioned by the diagonal Fubini-Study metric:
    /// g~_i = grad_i / (max(g_i, 0) + `lambda_damping`).
    pub fn step(&mut self, grad: &[f64], metric: Option<(&[f64], f64)>, epoch: usize, lr: f64) {
        let beta1 = 0.9;
        let beta2 = 0.999;
        for i in 0..self.params.len() {
            let g = match metric {
                Some((diag, lambda_damping)) => grad[i] / (diag[i].max(0.0) + lambda_damping),
                None => grad[i],
            };
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * g;
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * g * g;
            let m_hat = self.m[i] / (1.0 - beta1.powi(epoch as i32));
            let v_hat = self.v[i] / (1.0 - beta2.powi(epoch as i32));
            let step = lr * m_hat / (v_hat.sqrt() + 1e-8);
            self.params[i] -= step.clamp(-0.15, 0.15);
        }
    }
}

// ── Persisted contracts ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QngCheckpoint {
    pub timestamp_rfc3339: String,
    pub n_qubits: usize,
    pub depth: usize,
    pub source_qubit: usize,
    pub circuit_params_euclidean: Vec<f64>,
    pub circuit_params_qng: Vec<f64>,
    pub upload_scales_qng: Vec<f64>,
    pub feedforward_alphas: Vec<f64>,
    pub readout_weights_78: Vec<Vec<f64>>,
    pub readout_biases_78: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QngEpochTelemetry {
    pub epoch: usize,
    pub loss_euclidean: f64,
    pub loss_qng: f64,
    pub g_cond_number: f64,
    pub g_min_eigenval: f64,
    pub g_max_eigenval: f64,
    pub pred_mol10640_euclidean: f64,
    pub pred_mol10640_qng: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QngComparisonReport {
    pub timestamp_rfc3339: String,
    pub host_cpu: String,
    pub n_qubits: usize,
    pub train_samples: usize,
    pub test_samples: usize,
    pub pre_qng_mae_logp: f64,
    pub pre_qng_mae_qed: f64,
    pub pre_qng_mae_sas: f64,
    pub pre_qng_r2_logp: f64,
    pub pre_qng_r2_qed: f64,
    pub pre_qng_r2_sas: f64,
    pub euclidean_mae_logp: f64,
    pub euclidean_mae_qed: f64,
    pub euclidean_mae_sas: f64,
    pub euclidean_r2_logp: f64,
    pub euclidean_r2_qed: f64,
    pub euclidean_r2_sas: f64,
    pub qng_mae_logp: f64,
    pub qng_mae_qed: f64,
    pub qng_mae_sas: f64,
    pub qng_r2_logp: f64,
    pub qng_r2_qed: f64,
    pub qng_r2_sas: f64,
    pub mol10640_true_logp: f64,
    pub mol10640_pre_qng_pred: f64,
    pub mol10640_euclidean_pred: f64,
    pub mol10640_qng_pred: f64,
    pub epoch_history: Vec<QngEpochTelemetry>,
    #[serde(default)]
    pub classical_baselines: Vec<crate::baselines::BaselineResult>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EncodingKind;

    fn molecule(seed: usize) -> ZincMolecule {
        ZincMolecule {
            smiles: String::new(),
            features: (0..19)
                .map(|i| ((i * 5 + seed * 7) % 13) as f64 / 6.0 + 0.1)
                .collect(),
            log_p: seed as f64 * 0.3 - 1.0,
            qed: 0.5,
            sas: 2.0,
        }
    }

    fn circuit() -> (
        CircuitAnsatz,
        DataEncodingLayer,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
    ) {
        let params: Vec<f64> = (0..60).map(|i| (f64::from(i) * 0.41).sin()).collect();
        let dru = vec![1.0; 12];
        let alphas: Vec<f64> = (0..12).map(|i| 0.05 * f64::from(i)).collect();
        (
            CircuitAnsatz::new(N_QUBITS, 3).with_pi_layer(true),
            DataEncodingLayer::new(N_QUBITS, EncodingKind::Amplitude),
            params,
            dru,
            alphas,
        )
    }

    #[test]
    fn adam_first_step_moves_each_parameter_by_the_learning_rate_against_the_gradient() {
        let mut path = AdamPath::new(&[0.5, -0.5, 0.0], &[1.0]);
        path.step(&[2.0, -3.0, 0.0], None, 1, 0.04);
        assert!((path.params[0] - (0.5 - 0.04)).abs() < 1e-6);
        assert!((path.params[1] - (-0.5 + 0.04)).abs() < 1e-6);
        assert_eq!(path.params[2], 0.0);
        assert_eq!(
            path.dru_scales,
            vec![1.0],
            "upload scales are not optimized"
        );
    }

    #[test]
    fn adam_step_is_clipped_to_the_trust_region() {
        let mut path = AdamPath::new(&[0.0], &[]);
        path.step(&[1e6], None, 1, 5.0);
        assert!((path.params[0] + 0.15).abs() < 1e-12);
    }

    #[test]
    fn metric_preconditioning_divides_the_gradient_by_the_damped_metric() {
        // Two steps with gradients g and g/k per metric k give the same moments as raw gradient
        // g/k, so the preconditioned path must match an unpreconditioned one fed grad/k.
        let metric = [4.0, 0.0];
        let mut pre = AdamPath::new(&[0.0, 0.0], &[]);
        let mut raw = AdamPath::new(&[0.0, 0.0], &[]);
        for epoch in 1..=3 {
            pre.step(&[2.0, 1.0], Some((&metric, 1.0)), epoch, 0.04);
            raw.step(&[2.0 / 5.0, 1.0 / 1.0], None, epoch, 0.04);
        }
        assert_eq!(pre.params, raw.params);
    }

    #[test]
    fn negative_metric_entries_are_floored_at_zero() {
        let mut a = AdamPath::new(&[0.0], &[]);
        let mut b = AdamPath::new(&[0.0], &[]);
        a.step(&[1.0], Some((&[-7.0], 0.05)), 1, 0.04);
        b.step(&[1.0], Some((&[0.0], 0.05)), 1, 0.04);
        assert_eq!(a.params, b.params);
    }

    #[test]
    fn ridge_loss_is_small_for_a_noiseless_target_and_weights_have_the_right_shape() {
        let (ansatz, encoding, params, dru, alphas) = circuit();
        let model = Model {
            ansatz: &ansatz,
            encoding: &encoding,
            feedforward_alphas: &alphas,
        };
        let batch: Vec<ZincMolecule> = (0..40).map(molecule).collect();
        let (loss, w, b) = eval_loss_and_weights(&model, &params, &dru, &batch, 1e-3);
        assert!(loss.is_finite() && loss >= 0.0);
        assert_eq!((w.len(), w[0].len(), b.len()), (3, N_QUBITS, 3));
        // QED / SAS are constant, so their fitted readout is a pure bias.
        assert!((b[1] - 0.5).abs() < 0.05 && (b[2] - 2.0).abs() < 0.05);
    }

    #[test]
    fn fubini_study_diagonal_is_at_least_the_damping_and_finite() {
        let (ansatz, encoding, params, dru, alphas) = circuit();
        let model = Model {
            ansatz: &ansatz,
            encoding: &encoding,
            feedforward_alphas: &alphas,
        };
        let batch = [molecule(1)];
        let g = fubini_study_diagonal(&model, &params, &dru, &batch, 0.02);
        assert_eq!(g.len(), 60);
        assert!(g.iter().all(|v| v.is_finite() && *v >= 0.02));
        assert!(
            g.iter().any(|v| *v > 0.02),
            "some parameter must move the state"
        );
    }

    #[test]
    fn tracked_paths_continue_from_the_previous_paths_state() {
        let (ansatz, encoding, params, dru, alphas) = circuit();
        let model = Model {
            ansatz: &ansatz,
            encoding: &encoding,
            feedforward_alphas: &alphas,
        };
        let features = molecule(3).features;
        let weights = vec![0.1; N_QUBITS];
        let path = TrackedPath {
            params: &params,
            dru_scales: &dru,
            logp_weights: &weights,
            logp_bias: 0.0,
        };

        // The first path equals a fresh encode-and-run ...
        let mut state = QuantumState::new(DIM);
        let mut z = [0.0; N_QUBITS];
        model.encode_and_run(&mut state, &params, &dru, &features, &mut z);
        let fresh: f64 = z.iter().map(|v| v * 0.1).sum();
        let tracked = track_molecule(&model, &features, &[path, path]);
        assert!((tracked[0] - fresh).abs() < 1e-12);
        // ... but the same parameters again start from the evolved state, so differ.
        assert!((tracked[1] - tracked[0]).abs() > 1e-6);
    }

    #[test]
    fn holdout_scores_are_finite_and_perfect_for_a_constant_target_gap() {
        let (ansatz, encoding, params, dru, alphas) = circuit();
        let model = Model {
            ansatz: &ansatz,
            encoding: &encoding,
            feedforward_alphas: &alphas,
        };
        let test: Vec<ZincMolecule> = (0..30).map(molecule).collect();
        let (_, w, b) = eval_loss_and_weights(&model, &params, &dru, &test, 1e-3);
        let s = holdout_scores(&model, &params, &dru, &w, &b, &test);
        assert!(s.mae.iter().chain(&s.r2).all(|v| v.is_finite()));
        assert!(
            s.mae[1] < 0.05 && s.mae[2] < 0.05,
            "constant QED / SAS are fitted by the bias"
        );
    }
}
