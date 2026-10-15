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
        Self { wrap_angles: false, ..Self::new(param_count, lr) }
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
    pub fn new(encoding: DataEncodingLayer, ansatz: CircuitAnsatz, observable: PauliObservable) -> Self {
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
    pub fn compute_loss_gradients(&self, x: &[f64], y_target: f64, weights: &[f64]) -> (f64, f64, Vec<f64>) {
        self.compute_loss_gradients_for_state(&self.encoding.encode(x), y_target, weights)
    }

    /// Forward pass on an already encoded or prepared quantum state.
    pub fn predict_state(&self, state: &QuantumState, weights: &[f64]) -> f64 {
        let state_out = self.ansatz.forward(state, weights);
        self.observable.expectation_value(&state_out)
    }

    /// Exact analytic gradient d<O>/dθ on an arbitrary input state (adjoint differentiation).
    pub fn compute_circuit_gradients_for_state(&self, state: &QuantumState, weights: &[f64]) -> Vec<f64> {
        adjoint_gradient(&self.ansatz, state, weights, std::slice::from_ref(&self.observable), |_| vec![1.0])
            .gradient
    }

    /// Loss and gradient on an input state for scalar target y (adjoint differentiation).
    /// Returns (prediction, loss, gradient).
    pub fn compute_loss_gradients_for_state(&self, state: &QuantumState, y_target: f64, weights: &[f64]) -> (f64, f64, Vec<f64>) {
        let res = adjoint_gradient(&self.ansatz, state, weights, std::slice::from_ref(&self.observable), |e| {
            vec![e[0] - y_target]
        });
        let pred = res.expectations[0];
        let error = pred - y_target;
        (pred, 0.5 * error * error, res.gradient)
    }

    /// Forward pass evaluating multiple observables simultaneously on an evolved quantum state.
    pub fn predict_multi_observables(&self, state: &QuantumState, observables: &[PauliObservable], weights: &[f64]) -> Vec<f64> {
        let state_out = self.ansatz.forward(state, weights);
        observables.iter().map(|obs| obs.expectation_value(&state_out)).collect()
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
            preds.iter().zip(targets).map(|(p, y)| (p - y) * inv_k).collect()
        });
        let loss = 0.5 * inv_k * res.expectations.iter().zip(targets).map(|(p, y)| (p - y) * (p - y)).sum::<f64>();
        (res.expectations, loss, res.gradient)
    }

    // ----- Parameter-shift reference implementations (2 * P circuit runs each) -----

    /// d<O>/dθ_j = (<O>_{θ_j + π/2} - <O>_{θ_j - π/2}) / 2, evaluated by brute force.
    /// Kept for cross-checking the adjoint gradients and for benchmarks.
    pub fn compute_circuit_gradients_parameter_shift(&self, x: &[f64], weights: &[f64]) -> Vec<f64> {
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
            observables.iter().zip(&errors).map(|(o, e)| e * inv_k * o.expectation_value(s)).sum()
        });
        (preds, loss, grads)
    }

    fn parameter_shift(&self, state: &QuantumState, weights: &[f64], f: impl Fn(&QuantumState) -> f64) -> Vec<f64> {
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
