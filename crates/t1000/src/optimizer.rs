use crate::adjoint::adjoint_gradient;
use crate::ansatz::CircuitAnsatz;
use crate::feature_map::DataEncodingLayer;
use crate::observable::PauliObservable;
use crate::state::QuantumState;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

fn default_true() -> bool {
    true
}

/// Classical Adam Optimizer for updating quantum circuit parameters θ.
///
/// By default every parameter is wrapped into [-π, π) after each step, which is only correct when
/// all parameters are rotation angles. Use [`AdamOptimizer::new_unwrapped`] (or set
/// `wrap_angles = false`) when the parameter vector also holds non-angle values such as readout
/// weights, biases or scales.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamOptimizer {
    pub lr: f64,
    pub initial_lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub m: Vec<f64>,
    pub v: Vec<f64>,
    pub step: usize,
    #[serde(default = "default_true")]
    pub wrap_angles: bool,
}

impl AdamOptimizer {
    pub fn new(param_count: usize, lr: f64) -> Self {
        Self {
            lr,
            initial_lr: lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m: vec![0.0; param_count],
            v: vec![0.0; param_count],
            step: 0,
            wrap_angles: true,
        }
    }

    /// Adam without angle wrapping, for parameter vectors that are not all rotation angles.
    pub fn new_unwrapped(param_count: usize, lr: f64) -> Self {
        Self {
            wrap_angles: false,
            ..Self::new(param_count, lr)
        }
    }

    /// Update the learning rate using exponential step decay schedule:
    /// \eta_epoch = \eta_0 * \gamma^{\lfloor \text{epoch} / k \rfloor}
    pub fn step_epoch_decay(&mut self, epoch: usize, gamma: f64, step_interval: usize) -> f64 {
        if step_interval == 0 {
            return self.lr;
        }
        let power = (epoch / step_interval) as i32;
        self.lr = self.initial_lr * gamma.powi(power);
        self.lr
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    /// Update parameters θ using Adam moment estimators.
    pub fn step(&mut self, weights: &mut [f64], grads: &[f64]) {
        self.step += 1;
        let beta1_t = self.beta1.powi(self.step as i32);
        let beta2_t = self.beta2.powi(self.step as i32);

        for (i, (&g, w)) in grads.iter().zip(weights.iter_mut()).enumerate() {
            // First moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * g
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            // Second moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;

            // Bias-corrected estimates
            let m_hat = self.m[i] / (1.0 - beta1_t).max(1e-12);
            let v_hat = self.v[i] / (1.0 - beta2_t).max(1e-12);

            // Update θ
            *w -= self.lr * m_hat / (v_hat.sqrt() + self.epsilon);

            if self.wrap_angles {
                // Keep rotation angles within canonical [-π, π]
                *w = (*w + PI).rem_euclid(2.0 * PI) - PI;
            }
        }
    }
}

/// Hybrid quantum-classical gradient engine. Gradients use adjoint differentiation; the
/// `*_parameter_shift` methods keep the parameter-shift rule for cross-checks. (The type name
/// is kept for compatibility.)
#[derive(Debug, Clone)]
pub struct ParameterShiftEngine {
    pub encoding: DataEncodingLayer,
    pub ansatz: CircuitAnsatz,
    pub observable: PauliObservable,
}

impl ParameterShiftEngine {
    pub fn new(
        encoding: DataEncodingLayer,
        ansatz: CircuitAnsatz,
        observable: PauliObservable,
    ) -> Self {
        Self {
            encoding,
            ansatz,
            observable,
        }
    }

    /// Forward pass predicting <O> on input features x:
    /// y_pred = <0| U^†(x) W^†(θ) O W(θ) U(x) |0>
    pub fn predict(&self, x: &[f64], weights: &[f64]) -> f64 {
        let state_in = self.encoding.encode(x);
        let state_out = self.ansatz.forward(&state_in, weights);
        self.observable.expectation_value(&state_out)
    }

    /// Exact analytic gradient d<O>/dθ via adjoint differentiation (one forward and one
    /// backward sweep, independent of the parameter count).
    pub fn compute_circuit_gradients(&self, x: &[f64], weights: &[f64]) -> Vec<f64> {
        self.compute_circuit_gradients_for_state(&self.encoding.encode(x), weights)
    }

    /// Loss gradient dL/dθ = (dL/dy_pred) * (dy_pred/dθ) for MSE loss L = 0.5 * (y_pred - y)^2.
    /// Returns (prediction, loss, gradient).
    pub fn compute_loss_gradients(
        &self,
        x: &[f64],
        y_target: f64,
        weights: &[f64],
    ) -> (f64, f64, Vec<f64>) {
        self.compute_loss_gradients_for_state(&self.encoding.encode(x), y_target, weights)
    }

    /// Forward pass on an already encoded or prepared quantum state.
    pub fn predict_state(&self, state: &QuantumState, weights: &[f64]) -> f64 {
        let state_out = self.ansatz.forward(state, weights);
        self.observable.expectation_value(&state_out)
    }

    /// Exact analytic gradient d<O>/dθ on an arbitrary input state (adjoint differentiation).
    pub fn compute_circuit_gradients_for_state(
        &self,
        state: &QuantumState,
        weights: &[f64],
    ) -> Vec<f64> {
        adjoint_gradient(
            &self.ansatz,
            state,
            weights,
            std::slice::from_ref(&self.observable),
            |_| vec![1.0],
        )
        .gradient
    }

    /// Loss and gradient on an input state for scalar target y (adjoint differentiation).
    /// Returns (prediction, loss, gradient).
    pub fn compute_loss_gradients_for_state(
        &self,
        state: &QuantumState,
        y_target: f64,
        weights: &[f64],
    ) -> (f64, f64, Vec<f64>) {
        let res = adjoint_gradient(
            &self.ansatz,
            state,
            weights,
            std::slice::from_ref(&self.observable),
            |e| vec![e[0] - y_target],
        );
        let pred = res.expectations[0];
        let error = pred - y_target;
        (pred, 0.5 * error * error, res.gradient)
    }

    /// Forward pass evaluating multiple observables simultaneously on an evolved quantum state.
    pub fn predict_multi_observables(
        &self,
        state: &QuantumState,
        observables: &[PauliObservable],
        weights: &[f64],
    ) -> Vec<f64> {
        let state_out = self.ansatz.forward(state, weights);
        observables
            .iter()
            .map(|obs| obs.expectation_value(&state_out))
            .collect()
    }

    /// Loss L = (1 / 2K) * sum_k (<O_k> - y_k)^2 over K observables and its gradient, computed in
    /// one adjoint sweep. Returns (predictions, loss, gradient).
    pub fn compute_multi_observable_loss_gradients(
        &self,
        state: &QuantumState,
        observables: &[PauliObservable],
        targets: &[f64],
        weights: &[f64],
    ) -> (Vec<f64>, f64, Vec<f64>) {
        assert_eq!(observables.len(), targets.len());
        let inv_k = 1.0 / observables.len() as f64;
        let res = adjoint_gradient(&self.ansatz, state, weights, observables, |preds| {
            preds
                .iter()
                .zip(targets)
                .map(|(p, y)| (p - y) * inv_k)
                .collect()
        });
        let loss = 0.5
            * inv_k
            * res
                .expectations
                .iter()
                .zip(targets)
                .map(|(p, y)| (p - y) * (p - y))
                .sum::<f64>();
        (res.expectations, loss, res.gradient)
    }

    // ----- Parameter-shift reference implementations (2 * P circuit runs each) -----

    /// d<O>/dθ_j = (<O>_{θ_j + π/2} - <O>_{θ_j - π/2}) / 2, evaluated by brute force.
    /// Kept for cross-checking the adjoint gradients and for benchmarks.
    pub fn compute_circuit_gradients_parameter_shift(
        &self,
        x: &[f64],
        weights: &[f64],
    ) -> Vec<f64> {
        let state_in = self.encoding.encode(x);
        self.parameter_shift(&state_in, weights, |s| self.observable.expectation_value(s))
    }

    /// Parameter-shift version of [`Self::compute_multi_observable_loss_gradients`].
    pub fn compute_multi_observable_loss_gradients_parameter_shift(
        &self,
        state: &QuantumState,
        observables: &[PauliObservable],
        targets: &[f64],
        weights: &[f64],
    ) -> (Vec<f64>, f64, Vec<f64>) {
        assert_eq!(observables.len(), targets.len());
        let inv_k = 1.0 / observables.len() as f64;
        let preds = self.predict_multi_observables(state, observables, weights);
        let errors: Vec<f64> = preds.iter().zip(targets).map(|(p, y)| p - y).collect();
        let loss = 0.5 * errors.iter().map(|e| e * e).sum::<f64>() * inv_k;
        let grads = self.parameter_shift(state, weights, |s| {
            observables
                .iter()
                .zip(&errors)
                .map(|(o, e)| e * inv_k * o.expectation_value(s))
                .sum()
        });
        (preds, loss, grads)
    }

    fn parameter_shift(
        &self,
        state: &QuantumState,
        weights: &[f64],
        f: impl Fn(&QuantumState) -> f64,
    ) -> Vec<f64> {
        let mut shifted = weights.to_vec();
        let shift = PI * 0.5;
        (0..weights.len())
            .map(|j| {
                shifted[j] = weights[j] + shift;
                let pos = f(&self.ansatz.forward(state, &shifted));
                shifted[j] = weights[j] - shift;
                let neg = f(&self.ansatz.forward(state, &shifted));
                shifted[j] = weights[j];
                0.5 * (pos - neg)
            })
            .collect()
    }
}

/// Cache holding normalized correlator expectations and reciprocal standard deviation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorrelatorNormCacheArr<const N: usize> {
    pub normalized: [f64; N],
    pub inv_std: f64,
}

/// Forward pass: Standardizes raw observable expectation array to zero mean and unit variance.
/// Eliminates magnitude disparity between single-qubit Z (~0.7) and two-body ZZ (~0.1).
#[inline(always)]
pub fn normalize_correlators_arr_forward<const N: usize>(
    raw: &[f64; N],
) -> CorrelatorNormCacheArr<N> {
    let n = N as f64;
    let mean = raw.iter().sum::<f64>() / n;
    let var = raw
        .iter()
        .map(|&c| {
            let d = c - mean;
            d * d
        })
        .sum::<f64>()
        / n;

    let inv_std = 1.0 / (var + 1e-6).sqrt();
    let mut normalized = [0.0; N];
    for i in 0..N {
        normalized[i] = (raw[i] - mean) * inv_std;
    }
    CorrelatorNormCacheArr {
        normalized,
        inv_std,
    }
}

/// Backward pass: Propagates normalized input gradients dL/d(c_hat) back through the LayerNorm
/// Jacobian to raw observable gradients dL/d(c_raw) for adjoint circuit differentiation:
/// dL/dc_i = (1/sigma) * [ g_i - mean(g) - c_hat_i * mean(g . c_hat) ]
#[inline(always)]
pub fn normalize_correlators_arr_backward<const N: usize>(
    grad_norm: &[f64; N],
    cache: &CorrelatorNormCacheArr<N>,
) -> [f64; N] {
    let n = N as f64;
    let mean_grad = grad_norm.iter().sum::<f64>() / n;
    let mean_grad_dot_norm = grad_norm
        .iter()
        .zip(cache.normalized.iter())
        .map(|(&g, &c_hat)| g * c_hat)
        .sum::<f64>()
        / n;

    let mut grad_raw = [0.0; N];
    for i in 0..N {
        grad_raw[i] =
            cache.inv_std * (grad_norm[i] - mean_grad - cache.normalized[i] * mean_grad_dot_norm);
    }
    grad_raw
}

/// Two-Speed AdamW Optimizer partitioning periodic quantum angles from Euclidean classical weights.
///
/// 1. VQC Angles: Driven with higher learning rate (default 0.012) and zero weight decay.
/// 2. Classical Head: Driven with damped learning rate (default 0.0015) and decoupled L2 weight decay (0.01).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwoSpeedHybridOptimizer {
    pub lr_vqc: f64,
    pub lr_head: f64,
    pub wd_head: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub step_t: usize,
    m_vqc: Vec<f64>,
    v_vqc: Vec<f64>,
    m_head: Vec<f64>,
    v_head: Vec<f64>,
}

impl TwoSpeedHybridOptimizer {
    pub fn new(n_vqc: usize, n_head: usize) -> Self {
        Self {
            lr_vqc: 0.012,
            lr_head: 0.0015,
            wd_head: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            step_t: 0,
            m_vqc: vec![0.0; n_vqc],
            v_vqc: vec![0.0; n_vqc],
            m_head: vec![0.0; n_head],
            v_head: vec![0.0; n_head],
        }
    }

    pub fn with_learning_rates(mut self, lr_vqc: f64, lr_head: f64, wd_head: f64) -> Self {
        self.lr_vqc = lr_vqc;
        self.lr_head = lr_head;
        self.wd_head = wd_head;
        self
    }

    pub fn step(
        &mut self,
        vqc_params: &mut [f64],
        vqc_grads: &[f64],
        head_params: &mut [f64],
        head_grads: &[f64],
    ) {
        self.step_t += 1;
        let t = self.step_t as i32;
        let bias_corr1 = (1.0 - self.beta1.powi(t)).max(1e-12);
        let bias_corr2 = (1.0 - self.beta2.powi(t)).max(1e-12);

        // 1. Quantum parameters (periodic mod 2pi, zero weight decay)
        let eff_lr_vqc = self.lr_vqc * bias_corr2.sqrt() / bias_corr1;
        for (i, w) in vqc_params.iter_mut().enumerate() {
            let g = vqc_grads[i].clamp(-1.0, 1.0);
            self.m_vqc[i] = self.beta1 * self.m_vqc[i] + (1.0 - self.beta1) * g;
            self.v_vqc[i] = self.beta2 * self.v_vqc[i] + (1.0 - self.beta2) * (g * g);
            *w -= eff_lr_vqc * self.m_vqc[i] / (self.v_vqc[i].sqrt() + self.eps);
        }

        // 2. Classical parameters (decoupled weight decay, skipping final 2 bias terms if present)
        let eff_lr_head = self.lr_head * bias_corr2.sqrt() / bias_corr1;
        let head_len = head_params.len();
        for (i, w) in head_params.iter_mut().enumerate() {
            let g = head_grads[i].clamp(-1.0, 1.0);
            self.m_head[i] = self.beta1 * self.m_head[i] + (1.0 - self.beta1) * g;
            self.v_head[i] = self.beta2 * self.v_head[i] + (1.0 - self.beta2) * (g * g);
            if i + 2 < head_len {
                *w *= 1.0 - self.lr_head * self.wd_head;
            }
            *w -= eff_lr_head * self.m_head[i] / (self.v_head[i].sqrt() + self.eps);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlator_layernorm_finite_difference_exactness() {
        // Test with 8 correlators representing a small register
        let raw: [f64; 8] = [0.82, -0.45, 0.12, 0.05, -0.08, 0.73, -0.22, 0.19];
        let weights: [f64; 8] = [1.2, -0.7, 0.5, 2.1, -1.4, 0.9, -0.3, 0.8];

        let cache = normalize_correlators_arr_forward(&raw);
        let grad_raw_analytical = normalize_correlators_arr_backward(&weights, &cache);

        // Finite differences verification: L(c) = sum_k w_k * c_hat_k
        let eps = 1e-6;
        for i in 0..8 {
            let mut raw_pos = raw;
            raw_pos[i] += eps;
            let cache_pos = normalize_correlators_arr_forward(&raw_pos);
            let loss_pos: f64 = cache_pos
                .normalized
                .iter()
                .zip(&weights)
                .map(|(c, w)| c * w)
                .sum();

            let mut raw_neg = raw;
            raw_neg[i] -= eps;
            let cache_neg = normalize_correlators_arr_forward(&raw_neg);
            let loss_neg: f64 = cache_neg
                .normalized
                .iter()
                .zip(&weights)
                .map(|(c, w)| c * w)
                .sum();

            let grad_fd = (loss_pos - loss_neg) / (2.0 * eps);
            let diff = (grad_raw_analytical[i] - grad_fd).abs();
            assert!(
                diff < 1e-5,
                "Jacobian mismatch at index {}: analytical={}, fd={}",
                i,
                grad_raw_analytical[i],
                grad_fd
            );
        }
    }

    #[test]
    fn test_two_speed_optimizer_partitioning() {
        let mut vqc = vec![0.5, -0.5];
        let mut head = vec![1.0, 1.0, 0.0, 0.0];
        let mut opt = TwoSpeedHybridOptimizer::new(2, 4);

        let vqc_grads = vec![0.1, -0.1];
        let head_grads = vec![0.2, 0.2, 0.1, 0.1];

        let vqc_before = vqc.clone();
        let head_before = head.clone();

        opt.step(&mut vqc, &vqc_grads, &mut head, &head_grads);

        // Both sets of parameters should be updated
        assert_ne!(vqc, vqc_before);
        assert_ne!(head, head_before);

        // VQC update delta should reflect the 8x higher learning rate
        let vqc_delta = (vqc[0] - vqc_before[0]).abs();
        let head_delta = (head[0] - head_before[0]).abs();
        assert!(vqc_delta > head_delta);
    }
}
