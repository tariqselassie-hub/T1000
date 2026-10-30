// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Quantum Natural Gradient (QNG) preconditioning.
//!
//! Ordinary gradient descent moves in parameter space, so the effective step size depends on how
//! the ansatz is parameterised. Natural gradient moves along the steepest direction in state
//! space, preconditioning the gradient by the inverse of the Fubini-Study metric (the quantum
//! Fisher information matrix):
//!
//! ```text
//! theta <- theta - lr * (F + lambda I)^{-1} g
//! ```
//!
//! For a pure state `|psi(theta)>`,
//!
//! ```text
//! F_ij = Re( <d_i psi | d_j psi> - <d_i psi | psi> <psi | d_j psi> ).
//! ```
//!
//! The tangent vectors use the exact shift identity for `Ry`/`Rz` generators,
//! `d_i |psi> = (|psi(theta_i + pi)> - |psi(theta_i - pi)>) / 4`, so no finite differences and
//! no extra structure is assumed about the ansatz. (The more familiar `pi/2` shift computes
//! expectation-value derivatives, not state tangents: it differs from the true state derivative by
//! a factor of `sqrt(2)`.) Building the full metric costs `2P` forward passes; the damped linear
//! solve is `O(P^3)`. This is the expected QNG trade-off: more work per step in exchange for
//! parameterisation-invariant, often much faster convergence.
//!
//! Damping keeps the metric positive definite (states are only normalised to rounding), and a
//! giant-damping fallback degrades to plain gradient descent instead of failing.

use crate::ansatz::CircuitAnsatz;
use crate::state::QuantumState;
use std::f64::consts::PI;

/// Quantum Fisher information matrix of the state prepared by `ansatz` on `input`.
///
/// Returns the real, symmetric `P x P` matrix in row-major order, where `P = weights.len()`.
///
/// # Panics
/// If `weights` does not match the ansatz parameter count or `input.dim != ansatz.dim`.
pub fn quantum_fisher_information(
    ansatz: &CircuitAnsatz,
    input: &QuantumState,
    weights: &[f64],
) -> Vec<f64> {
    let p = weights.len();
    assert_eq!(
        p,
        ansatz.param_count(),
        "weight count does not match the ansatz"
    );
    assert_eq!(
        input.dim, ansatz.dim,
        "input dimension does not match the ansatz"
    );
    let dim = ansatz.dim;

    let psi = ansatz.forward(input, weights);
    if p == 0 {
        return Vec::new();
    }

    // Tangent vectors t_i = d|psi>/dtheta_i, obtained exactly by the +-pi shift identity.
    let mut plus = QuantumState::zero(dim);
    let mut minus = QuantumState::zero(dim);
    let mut tangent = QuantumState::zero(dim);
    let mut shifted = weights.to_vec();
    let mut tangents = Vec::with_capacity(p);
    for (i, &theta) in weights.iter().enumerate() {
        shifted[i] = theta + PI;
        ansatz.forward_into(input, &shifted, &mut plus);
        shifted[i] = theta - PI;
        ansatz.forward_into(input, &shifted, &mut minus);
        shifted[i] = theta;

        for k in 0..dim {
            tangent.real[k] = 0.25 * (plus.real[k] - minus.real[k]);
            tangent.imag[k] = 0.25 * (plus.imag[k] - minus.imag[k]);
        }
        tangents.push(tangent.clone());
    }

    // a_i = <t_i|psi>; the projection removes the global-phase direction.
    let a: Vec<(f64, f64)> = tangents.iter().map(|t| inner_conj(t, &psi)).collect();

    let mut fisher = vec![0.0; p * p];
    for i in 0..p {
        for j in i..p {
            let gram = inner_real(&tangents[i], &tangents[j]);
            let projected = a[i].0 * a[j].0 + a[i].1 * a[j].1;
            let value = gram - projected;
            fisher[i * p + j] = value;
            fisher[j * p + i] = value;
        }
    }
    fisher
}

/// Real part of `<x|y>`.
fn inner_real(x: &QuantumState, y: &QuantumState) -> f64 {
    x.real
        .iter()
        .zip(&x.imag)
        .zip(y.real.iter().zip(&y.imag))
        .map(|((&xr, &xi), (&yr, &yi))| xr * yr + xi * yi)
        .sum()
}

/// `<x|y>` as `(re, im)`.
fn inner_conj(x: &QuantumState, y: &QuantumState) -> (f64, f64) {
    let mut re = 0.0;
    let mut im = 0.0;
    for k in 0..x.real.len() {
        let (xr, xi) = (x.real[k], x.imag[k]);
        let (yr, yi) = (y.real[k], y.imag[k]);
        re += xr * yr + xi * yi;
        im += xr * yi - xi * yr;
    }
    (re, im)
}

/// Solves `A x = b` for a symmetric positive-definite row-major `n x n` matrix.
///
/// `a` is overwritten with its Cholesky factor and `x` (initialised with `b`) receives the
/// solution. Returns `false` when `A` is not numerically positive definite.
fn cholesky_solve(a: &mut [f64], n: usize, x: &mut [f64]) -> bool {
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= a[i * n + k] * a[j * n + k];
            }
            if i == j {
                if !(sum.is_finite() && sum > 0.0) {
                    return false;
                }
                a[i * n + i] = sum.sqrt();
            } else {
                a[i * n + j] = sum / a[j * n + j];
            }
        }
    }

    // Forward substitution: L y = b.
    for i in 0..n {
        let mut sum = x[i];
        for k in 0..i {
            sum -= a[i * n + k] * x[k];
        }
        x[i] = sum / a[i * n + i];
    }
    // Back substitution: L^T x = y.
    for i in (0..n).rev() {
        let mut sum = x[i];
        for k in (i + 1)..n {
            sum -= a[k * n + i] * x[k];
        }
        x[i] = sum / a[i * n + i];
    }
    true
}

/// Outcome of one QNG update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QngStepInfo {
    /// Damping that produced a positive-definite metric (may exceed the configured value).
    pub damping: f64,
    /// True when the solve failed and the raw gradient direction was used instead.
    pub used_fallback: bool,
}

/// Quantum Natural Gradient optimizer.
#[derive(Debug, Clone)]
pub struct QuantumNaturalGradient {
    /// Learning rate multiplying the preconditioned direction.
    pub lr: f64,
    /// Initial Tikhonov damping added to the metric diagonal.
    pub damping: f64,
    /// Applied updates so far.
    pub steps: usize,
    /// Wrap every updated parameter into `[-pi, pi)`.
    pub wrap_angles: bool,
    scratch_metric: Vec<f64>,
    scratch_x: Vec<f64>,
}

impl QuantumNaturalGradient {
    /// Creates an optimizer with damping `1e-2`.
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            damping: 1e-2,
            steps: 0,
            wrap_angles: true,
            scratch_metric: Vec::new(),
            scratch_x: Vec::new(),
        }
    }

    /// Sets the initial damping added to the metric diagonal.
    pub fn with_damping(mut self, damping: f64) -> Self {
        self.damping = damping.max(0.0);
        self
    }

    /// Updates `weights` from `grads` and a precomputed row-major `P x P` metric.
    ///
    /// Retries with ten times more damping when the damped metric is not positive definite, and
    /// finally falls back to a plain gradient step.
    pub fn step_with_metric(
        &mut self,
        weights: &mut [f64],
        grads: &[f64],
        metric: &[f64],
    ) -> QngStepInfo {
        let p = weights.len();
        assert_eq!(grads.len(), p, "gradient and weight counts differ");
        assert_eq!(
            metric.len(),
            p * p,
            "metric must be a row-major P x P matrix"
        );
        if p == 0 {
            return QngStepInfo {
                damping: self.damping,
                used_fallback: false,
            };
        }

        let mut damping = self.damping.max(1e-12);
        loop {
            self.scratch_metric.clear();
            self.scratch_metric.extend_from_slice(metric);
            for i in 0..p {
                self.scratch_metric[i * p + i] += damping;
            }
            self.scratch_x.clear();
            self.scratch_x.extend_from_slice(grads);

            if cholesky_solve(&mut self.scratch_metric, p, &mut self.scratch_x) {
                for (w, &d) in weights.iter_mut().zip(&self.scratch_x) {
                    *w -= self.lr * d;
                    self.wrap(w);
                }
                self.steps += 1;
                return QngStepInfo {
                    damping,
                    used_fallback: false,
                };
            }

            damping *= 10.0;
            if !damping.is_finite() || damping > 1e8 {
                for (w, &g) in weights.iter_mut().zip(grads) {
                    *w -= self.lr * g;
                    self.wrap(w);
                }
                self.steps += 1;
                return QngStepInfo {
                    damping,
                    used_fallback: true,
                };
            }
        }
    }

    /// Updates `weights` from `grads`, building the metric at `input` internally.
    pub fn step(
        &mut self,
        ansatz: &CircuitAnsatz,
        input: &QuantumState,
        weights: &mut [f64],
        grads: &[f64],
    ) -> QngStepInfo {
        let metric = quantum_fisher_information(ansatz, input, weights);
        self.step_with_metric(weights, grads, &metric)
    }

    #[inline(always)]
    fn wrap(&self, w: &mut f64) {
        if self.wrap_angles {
            *w = (*w + PI).rem_euclid(2.0 * PI) - PI;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observable::PauliObservable;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn cholesky_solves_known_spd_system() {
        // A = [[4, 1, 0], [1, 3, 1], [0, 1, 2]], b = [1, 2, 3].
        let mut a = vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
        let mut x = vec![1.0, 2.0, 3.0];
        assert!(cholesky_solve(&mut a, 3, &mut x));
        // A x = b gives x = [2/9, 1/9, 13/9].
        let want = [2.0 / 9.0, 1.0 / 9.0, 13.0 / 9.0];
        for k in 0..3 {
            assert!(close(x[k], want[k], 1e-12), "x[{k}] = {}", x[k]);
        }
    }

    #[test]
    fn fisher_matches_analytic_single_qubit() {
        // |psi> = Rz(phi) Ry(theta) |0>, so F_theta,theta = 1/4,
        // F_phi,phi = sin^2(theta)/4, and the cross term vanishes.
        let ansatz = CircuitAnsatz::new(1, 1);
        let input = QuantumState::ground(2);
        let theta = std::f64::consts::FRAC_PI_2;
        let phi = 0.3;
        let fisher = quantum_fisher_information(&ansatz, &input, &[theta, phi]);
        assert_eq!(fisher.len(), 4);
        assert!(close(fisher[0], 0.25, 1e-12), "F00 = {}", fisher[0]);
        assert!(close(fisher[3], 0.25, 1e-12), "F11 = {}", fisher[3]);
        assert!(close(fisher[1], 0.0, 1e-12), "F01 = {}", fisher[1]);
        assert!(close(fisher[2], 0.0, 1e-12), "F10 = {}", fisher[2]);
    }

    #[test]
    fn fisher_is_symmetric_positive_semidefinite() {
        let ansatz = CircuitAnsatz::new(3, 2);
        let input = QuantumState::new(8);
        let weights: Vec<f64> = (0..ansatz.param_count())
            .map(|i| 0.11 * (i as f64 + 1.0))
            .collect();
        let p = weights.len();
        let fisher = quantum_fisher_information(&ansatz, &input, &weights);

        for i in 0..p {
            for j in 0..p {
                assert!(close(fisher[i * p + j], fisher[j * p + i], 1e-12));
            }
            assert!(fisher[i * p + i] >= -1e-12, "negative diagonal at {i}");
        }
        // a^T F a >= 0 for a few deterministic directions.
        for seed in 0..5 {
            let dir: Vec<f64> = (0..p)
                .map(|k| ((k * 7 + seed * 13 + 1) as f64 * 0.37).sin())
                .collect();
            let mut q = 0.0;
            for i in 0..p {
                for j in 0..p {
                    q += dir[i] * fisher[i * p + j] * dir[j];
                }
            }
            assert!(q >= -1e-10, "quadratic form {q} at seed {seed}");
        }
    }

    #[test]
    fn step_with_metric_applies_inverse_fisher_direction() {
        // F = diag(2, 4), g = [2, -4]  =>  F^{-1} g = [1, -1].
        let metric = vec![2.0, 0.0, 0.0, 4.0];
        let grads = vec![2.0, -4.0];
        let mut weights = vec![0.0, 0.0];
        let mut qng = QuantumNaturalGradient::new(0.5).with_damping(0.0);
        let info = qng.step_with_metric(&mut weights, &grads, &metric);
        assert!(!info.used_fallback);
        assert!(close(weights[0], -0.5, 1e-12));
        assert!(close(weights[1], 0.5, 1e-12));
    }

    #[test]
    fn qng_step_reduces_a_single_qubit_loss() {
        // Target <Z> = 1: the optimum is theta = 0 for Ry(theta)|0>.
        let ansatz = CircuitAnsatz::new(1, 1);
        let input = QuantumState::ground(2);
        let mut weights = vec![0.4, 0.0];
        let obs = PauliObservable::Z(0);
        let mut qng = QuantumNaturalGradient::new(0.2).with_damping(1e-3);

        let loss = |w: &[f64]| {
            let psi = ansatz.forward(&input, w);
            0.5 * (obs.expectation_value(&psi) - 1.0).powi(2)
        };
        let before = loss(&weights);
        for _ in 0..5 {
            let grads = crate::adjoint::adjoint_gradient(
                &ansatz,
                &input,
                &weights,
                std::slice::from_ref(&obs),
                |e| vec![e[0] - 1.0],
            )
            .gradient;
            qng.step(&ansatz, &input, &mut weights, &grads);
        }
        let after = loss(&weights);
        assert!(after < before, "loss did not decrease: {before} -> {after}");
    }
}
