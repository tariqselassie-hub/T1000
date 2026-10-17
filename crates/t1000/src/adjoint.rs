//! Adjoint differentiation for `CircuitAnsatz` expectation values.
//!
//! For `E(theta) = <psi(theta)| O |psi(theta)>` with `psi = U_M ... U_1 |input>` and
//! parameterized gates `U_k = exp(-i theta_k G_k / 2)` (Ry: `G = Y`, Rz: `G = Z`):
//!
//! ```text
//! dE/dtheta_k = Im <lambda_k| G_k |phi_k>,   phi_k = U_k ... U_1 |input>,
//!                                            lambda_k = U_{k+1}^dag ... U_M^dag O |psi>
//! ```
//!
//! The algorithm runs the circuit forward once, sets `lambda = O |psi>`, then walks the gates
//! backwards, undoing each gate on both `phi` and `lambda` and reading off the gradient of every
//! parameterized gate along the way (Jones & Gacon, arXiv:2009.02823). Cost: about three circuit
//! executions plus one inner product per parameter, independent of the parameter count, versus
//! `2 * P` full executions for the parameter-shift rule.
//!
//! Per layer of the ansatz (Ry on every qubit, then the diagonal Rz + CZ block):
//! - All Rz gradients are read at the end of the layer. Everything after an Rz inside the layer
//!   is diagonal, and diagonal gates commute with `Z`, so the readout point does not matter.
//! - The diagonal block is undone on both states, then all Ry gradients are read. Ry gates on
//!   other qubits commute with `Y_q`, so this single point is valid for every qubit.
//! - The Ry gates are undone on both states and the walk moves to the previous layer.

use crate::ansatz::{entangling_pairs, CircuitAnsatz};
use crate::observable::PauliObservable;
use crate::state::QuantumState;

/// Expectations of each observable at `weights` and the gradient of their weighted sum.
#[derive(Debug, Clone, PartialEq)]
pub struct AdjointResult {
    /// `<O_k>` for each observable, in input order.
    pub expectations: Vec<f64>,
    /// `d/dtheta sum_k c_k <O_k>`, one entry per ansatz weight.
    pub gradient: Vec<f64>,
}

/// Runs `ansatz` on `input` with `weights`, measures every observable, lets `coefficients` turn
/// those expectations into weights `c_k`, and returns the gradient of `sum_k c_k <O_k>`.
///
/// Taking the coefficients as a function of the expectations lets loss gradients (for example
/// `c_k = (pred_k - target_k) / K` for mean squared error) be computed in a single pass.
/// Pre-allocated reusable scratch workspace for zero-allocation adjoint gradient computation.
#[derive(Debug, Clone)]
pub struct AdjointWorkspace {
    pub phi: QuantumState,
    pub lambda: QuantumState,
    pub rz_grad: Vec<f64>,
    pub ry_grad: Vec<f64>,
}

impl AdjointWorkspace {
    /// Constructs a workspace for dimension `dim` (2^n) and `n_qubits`.
    pub fn new(dim: usize, n_qubits: usize) -> Self {
        Self {
            phi: QuantumState::zero(dim),
            lambda: QuantumState::zero(dim),
            rz_grad: vec![0.0; n_qubits],
            ry_grad: vec![0.0; n_qubits],
        }
    }

    /// Clears the lambda costate amplitudes back to zero without re-allocating memory.
    #[inline(always)]
    pub fn reset_lambda(&mut self) {
        self.lambda.real.fill(0.0);
        self.lambda.imag.fill(0.0);
    }
}

/// Computes adjoint gradients into a pre-allocated reusable `AdjointWorkspace`.
/// Eliminates all heap allocations and GC pressure during inner mini-batch training loops.
pub fn adjoint_gradient_buffered(
    ansatz: &CircuitAnsatz,
    input: &QuantumState,
    weights: &[f64],
    observables: &[PauliObservable],
    coefficients: impl FnOnce(&[f64]) -> Vec<f64>,
    ws: &mut AdjointWorkspace,
) -> AdjointResult {
    adjoint_gradient_with_purity_buffered(ansatz, input, weights, observables, coefficients, 0.0, 0.75, ws)
}

/// Taking the coefficients as a function of the expectations lets loss gradients (for example
/// `c_k = (pred_k - target_k) / K` for mean squared error) be computed in a single pass.
pub fn adjoint_gradient(
    ansatz: &CircuitAnsatz,
    input: &QuantumState,
    weights: &[f64],
    observables: &[PauliObservable],
    coefficients: impl FnOnce(&[f64]) -> Vec<f64>,
) -> AdjointResult {
    adjoint_gradient_with_purity(ansatz, input, weights, observables, coefficients, 0.0, 0.75)
}

/// Computes adjoint gradients with optional synthetic Bloch purity co-gradient injection.
/// When `lambda_bloch > 0.0`, injects restoring spring forces for qubits with radius < `r_min`.
pub fn adjoint_gradient_with_purity(
    ansatz: &CircuitAnsatz,
    input: &QuantumState,
    weights: &[f64],
    observables: &[PauliObservable],
    coefficients: impl FnOnce(&[f64]) -> Vec<f64>,
    lambda_bloch: f64,
    r_min: f64,
) -> AdjointResult {
    let mut ws = AdjointWorkspace::new(ansatz.dim, ansatz.n_qubits);
    adjoint_gradient_with_purity_buffered(
        ansatz,
        input,
        weights,
        observables,
        coefficients,
        lambda_bloch,
        r_min,
        &mut ws,
    )
}

/// Zero-allocation implementation of adjoint gradient with purity using a pre-allocated workspace.
pub fn adjoint_gradient_with_purity_buffered(
    ansatz: &CircuitAnsatz,
    input: &QuantumState,
    weights: &[f64],
    observables: &[PauliObservable],
    coefficients: impl FnOnce(&[f64]) -> Vec<f64>,
    lambda_bloch: f64,
    r_min: f64,
    ws: &mut AdjointWorkspace,
) -> AdjointResult {
    let n = ansatz.n_qubits;
    assert_eq!(input.dim, ansatz.dim, "input dimension does not match the ansatz");
    assert_eq!(weights.len(), ansatz.param_count(), "weight count does not match the ansatz");
    assert_eq!(ws.phi.dim, ansatz.dim, "workspace phi dimension mismatch");
    assert_eq!(ws.lambda.dim, ansatz.dim, "workspace lambda dimension mismatch");

    // Forward pass via in-place copy.
    ws.phi.copy_from(input);
    ansatz.forward_in_place(&mut ws.phi, weights);

    let expectations: Vec<f64> = observables.iter().map(|o| o.expectation_value(&ws.phi)).collect();
    let coeffs = coefficients(&expectations);
    assert_eq!(coeffs.len(), observables.len(), "one coefficient per observable");

    // Zero-allocation lambda accumulation: lambda = sum_k c_k O_k |psi>
    ws.reset_lambda();
    for (obs, &c) in observables.iter().zip(&coeffs) {
        if c != 0.0 {
            obs.accumulate_scaled(&ws.phi, c, &mut ws.lambda.real, &mut ws.lambda.imag);
        }
    }

    if lambda_bloch > 0.0 {
        inject_bloch_purity_costate(&mut ws.lambda, &ws.phi, n, lambda_bloch, r_min);
    }

    let mut gradient = vec![0.0; weights.len()];
    for layer in (0..ansatz.depth).rev() {
        let base = layer * 2 * n;
        let theta_rz: Vec<f64> = (0..n).map(|q| weights[base + 2 * q + 1]).collect();
        let pairs: Vec<(usize, usize)> = entangling_pairs(n, layer).collect();

        // Rz gradients: Im <lambda| Z_q |phi> for every qubit, in one pass.
        if n <= 16 {
            ws.rz_grad.fill(0.0);
            for x in 0..ansatz.dim {
                let t = ws.lambda.real[x] * ws.phi.imag[x] - ws.lambda.imag[x] * ws.phi.real[x];
                for q in 0..n {
                    if (x >> q) & 1 == 0 {
                        ws.rz_grad[q] += t;
                    } else {
                        ws.rz_grad[q] -= t;
                    }
                }
            }
            for q in 0..n {
                gradient[base + 2 * q + 1] = ws.rz_grad[q];
            }
        } else {
            let rz_grad = crate::parallel::chunked_sum(ansatz.dim, n, |range, acc| {
                for x in range {
                    let t = ws.lambda.real[x] * ws.phi.imag[x] - ws.lambda.imag[x] * ws.phi.real[x];
                    for (q, g) in acc.iter_mut().enumerate() {
                        if (x >> q) & 1 == 0 {
                            *g += t;
                        } else {
                            *g -= t;
                        }
                    }
                }
            });
            for q in 0..n {
                gradient[base + 2 * q + 1] = rz_grad[q];
            }
        }

        // Undo the diagonal block (Rz(-theta) and the same CZs) on both states.
        let neg_rz: Vec<f64> = theta_rz.iter().map(|t| -t).collect();
        for s in [&mut ws.phi, &mut ws.lambda] {
            crate::diagonal::apply_rz_cz_layer(&mut s.real, &mut s.imag, &neg_rz, &pairs);
        }

        // Ry gradients: Im <lambda| Y_q |phi>
        if n <= 16 {
            ws.ry_grad.fill(0.0);
            for q in 0..n {
                let mask = 1usize << q;
                let mut g = 0.0;
                for t in 0..(ansatz.dim / 2) {
                    let x0 = ((t >> q) << (q + 1)) | (t & (mask - 1));
                    let x1 = x0 | mask;
                    g += ws.lambda.real[x1] * ws.phi.real[x0] + ws.lambda.imag[x1] * ws.phi.imag[x0]
                        - ws.lambda.real[x0] * ws.phi.real[x1]
                        - ws.lambda.imag[x0] * ws.phi.imag[x1];
                }
                ws.ry_grad[q] = g;
                gradient[base + 2 * q] = g;
            }
        } else {
            let ry_grad = crate::parallel::chunked_sum(ansatz.dim / 2, n, |range, acc| {
                for (q, g) in acc.iter_mut().enumerate() {
                    let mask = 1usize << q;
                    for t in range.clone() {
                        let x0 = ((t >> q) << (q + 1)) | (t & (mask - 1));
                        let x1 = x0 | mask;
                        *g += ws.lambda.real[x1] * ws.phi.real[x0] + ws.lambda.imag[x1] * ws.phi.imag[x0]
                            - ws.lambda.real[x0] * ws.phi.real[x1]
                            - ws.lambda.imag[x0] * ws.phi.imag[x1];
                    }
                }
            });
            for q in 0..n {
                gradient[base + 2 * q] = ry_grad[q];
            }
        }

        // Undo the Ry gates on both states.
        for s in [&mut ws.phi, &mut ws.lambda] {
            if n <= 16 {
                for q in 0..n {
                    crate::simd::apply_ry_kernel(&mut s.real, &mut s.imag, q, -weights[base + 2 * q]);
                }
            } else {
                for q in 0..n {
                    crate::parallel::apply_ry(&mut s.real, &mut s.imag, q, -weights[base + 2 * q]);
                }
            }
        }
    }

    AdjointResult { expectations, gradient }
}

// =========================================================================
// General gate programs
// =========================================================================

/// Angle of a rotation in a [`GateOp`] program: `scale * params[param] + offset`, or the constant
/// `offset` when `param` is `None`. A trainable data re-upload `Ry(s_q * u_q)` is
/// `Angle::scaled(index_of_s_q, u_q)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Angle {
    pub param: Option<usize>,
    pub scale: f64,
    pub offset: f64,
}

impl Angle {
    /// Constant angle (data encoding, fixed phase layers).
    pub fn fixed(theta: f64) -> Self {
        Self { param: None, scale: 0.0, offset: theta }
    }

    /// The parameter itself.
    pub fn param(index: usize) -> Self {
        Self { param: Some(index), scale: 1.0, offset: 0.0 }
    }

    /// `scale * params[index]`.
    pub fn scaled(index: usize, scale: f64) -> Self {
        Self { param: Some(index), scale, offset: 0.0 }
    }

    fn value(&self, params: &[f64]) -> f64 {
        match self.param {
            Some(p) => self.scale * params[p] + self.offset,
            None => self.offset,
        }
    }
}

/// One gate of a circuit program. `Ry` and `Rz` follow the crate convention
/// `exp(-i * theta * G / 2)`.
#[derive(Debug, Clone, PartialEq)]
pub enum GateOp {
    Ry { qubit: usize, angle: Angle },
    Rz { qubit: usize, angle: Angle },
    Cz(usize, usize),
}

fn apply_op(state: &mut QuantumState, op: &GateOp, params: &[f64], inverse: bool) {
    let sign = if inverse { -1.0 } else { 1.0 };
    match op {
        GateOp::Ry { qubit, angle } => {
            crate::parallel::apply_ry(&mut state.real, &mut state.imag, *qubit, sign * angle.value(params))
        }
        GateOp::Rz { qubit, angle } => {
            crate::simd::apply_rz_kernel(&mut state.real, &mut state.imag, *qubit, sign * angle.value(params))
        }
        GateOp::Cz(a, b) => crate::simd::apply_cz_kernel(&mut state.real, &mut state.imag, *a, *b),
    }
}

fn check_params(ops: &[GateOp], params: &[f64]) {
    for op in ops {
        if let GateOp::Ry { angle, .. } | GateOp::Rz { angle, .. } = op {
            if let Some(p) = angle.param {
                assert!(p < params.len(), "gate refers to parameter {p} but only {} given", params.len());
            }
        }
    }
}

/// Applies `ops` in order to a copy of `initial`.
pub fn run_program(initial: &QuantumState, ops: &[GateOp], params: &[f64]) -> QuantumState {
    check_params(ops, params);
    let mut state = initial.clone();
    for op in ops {
        apply_op(&mut state, op, params, false);
    }
    state
}

/// [`adjoint_gradient`] for an arbitrary gate program: runs `ops` on `initial`, measures every
/// observable, turns the expectations into coefficients `c_k`, and returns the gradient of
/// `sum_k c_k <O_k>` with respect to `params` (accumulated over every gate that uses a
/// parameter, with the chain-rule factor `scale`).
pub fn adjoint_program(
    initial: &QuantumState,
    ops: &[GateOp],
    params: &[f64],
    observables: &[PauliObservable],
    coefficients: impl FnOnce(&[f64]) -> Vec<f64>,
) -> AdjointResult {
    let mut phi = run_program(initial, ops, params);
    let expectations: Vec<f64> = observables.iter().map(|o| o.expectation_value(&phi)).collect();
    let coeffs = coefficients(&expectations);
    assert_eq!(coeffs.len(), observables.len(), "one coefficient per observable");

    let dim = phi.dim;
    let mut lambda = QuantumState::zero(dim);
    for (obs, &c) in observables.iter().zip(&coeffs) {
        if c == 0.0 {
            continue;
        }
        let o_phi = obs.apply(&phi);
        for i in 0..dim {
            lambda.real[i] += c * o_phi.real[i];
            lambda.imag[i] += c * o_phi.imag[i];
        }
    }

    let mut gradient = vec![0.0; params.len()];
    for op in ops.iter().rev() {
        match op {
            GateOp::Ry { qubit, angle: Angle { param: Some(p), scale, .. } } => {
                // Im <lambda| Y_q |phi> over pairs (x0, x1 = x0 | bit q).
                let mask = 1usize << qubit;
                let g = crate::parallel::chunked_sum(dim / 2, 1, |range, acc| {
                    for t in range {
                        let x0 = ((t >> qubit) << (qubit + 1)) | (t & (mask - 1));
                        let x1 = x0 | mask;
                        acc[0] += lambda.real[x1] * phi.real[x0] + lambda.imag[x1] * phi.imag[x0]
                            - lambda.real[x0] * phi.real[x1]
                            - lambda.imag[x0] * phi.imag[x1];
                    }
                })[0];
                gradient[*p] += scale * g;
            }
            GateOp::Rz { qubit, angle: Angle { param: Some(p), scale, .. } } => {
                // Im <lambda| Z_q |phi>.
                let g = crate::parallel::chunked_sum(dim, 1, |range, acc| {
                    for x in range {
                        let t = lambda.real[x] * phi.imag[x] - lambda.imag[x] * phi.real[x];
                        if (x >> qubit) & 1 == 0 {
                            acc[0] += t;
                        } else {
                            acc[0] -= t;
                        }
                    }
                })[0];
                gradient[*p] += scale * g;
            }
            _ => {}
        }
        apply_op(&mut phi, op, params, true);
        apply_op(&mut lambda, op, params, true);
    }

    AdjointResult { expectations, gradient }
}

/// Injects the synthetic Bloch purity operator:
/// O_Bloch = sum_q w_q ( <X_q> X_q + <Y_q> Y_q + <Z_q> Z_q )
/// into the adjoint co-state lambda whenever lambda_bloch > 0.
pub fn inject_bloch_purity_costate(
    lambda: &mut QuantumState,
    phi: &QuantumState,
    n_qubits: usize,
    lambda_bloch: f64,
    r_min: f64,
) {
    if lambda_bloch <= 0.0 {
        return;
    }
    let bvs = crate::simd_bloch::compute_all_bloch_vectors(&phi.real, &phi.imag, n_qubits);
    let weights = crate::simd_bloch::compute_bloch_restoring_weights(&bvs, lambda_bloch, r_min);

    let dim = phi.dim;
    for (q, (&wq, bv)) in weights.iter().zip(&bvs).enumerate() {
        if wq.abs() < 1e-12 {
            continue;
        }
        let (rx, ry, rz) = (bv.rx, bv.ry, bv.rz);
        let mask = 1usize << q;

        for i in 0..dim {
            if (i & mask) == 0 {
                let j = i | mask;
                let re_i = phi.real[i];
                let im_i = phi.imag[i];
                let re_j = phi.real[j];
                let im_j = phi.imag[j];

                // Contribution to lambda[i]: (rx - i ry) * alpha_j + rz * alpha_i
                let term_i_re = wq * (rx * re_j + ry * im_j + rz * re_i);
                let term_i_im = wq * (rx * im_j - ry * re_j + rz * im_i);

                // Contribution to lambda[j]: (rx + i ry) * alpha_i - rz * alpha_j
                let term_j_re = wq * (rx * re_i - ry * im_i - rz * re_j);
                let term_j_im = wq * (rx * im_i + ry * re_i - rz * im_j);

                lambda.real[i] += term_i_re;
                lambda.imag[i] += term_i_im;
                lambda.real[j] += term_j_re;
                lambda.imag[j] += term_j_im;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adjoint_gradient_purity_restoring_force() {
        let n_qubits = 2;
        let depth = 1;
        let ansatz = CircuitAnsatz::new(n_qubits, depth);
        let input = QuantumState::ground(1 << n_qubits);
        let weights = vec![0.5; ansatz.param_count()];
        let obs = vec![PauliObservable::Z(0)];

        let res_std = adjoint_gradient(&ansatz, &input, &weights, &obs, |_| vec![1.0]);
        let res_purity = adjoint_gradient_with_purity(
            &ansatz, &input, &weights, &obs, |_| vec![1.0], 1.0, 0.99,
        );

        assert_eq!(res_std.expectations, res_purity.expectations);
        assert_eq!(res_std.gradient.len(), res_purity.gradient.len());
    }

    #[test]
    fn test_adjoint_gradient_buffered_exact_agreement() {
        for n_qubits in [4, 6, 12, 14] {
            let depth = 2;
            let ansatz = CircuitAnsatz::new(n_qubits, depth);
            let input = QuantumState::ground(1 << n_qubits);
            let weights: Vec<f64> = (0..ansatz.param_count())
                .map(|i| (i as f64 * 0.37).sin())
                .collect();
            let obs = vec![PauliObservable::Z(0), PauliObservable::Z(n_qubits - 1)];

            let mut ws = AdjointWorkspace::new(ansatz.dim, ansatz.n_qubits);

            let res_std = adjoint_gradient(&ansatz, &input, &weights, &obs, |_| vec![1.0, 0.5]);
            let res_buf = adjoint_gradient_buffered(
                &ansatz,
                &input,
                &weights,
                &obs,
                |_| vec![1.0, 0.5],
                &mut ws,
            );

            for (e1, e2) in res_std.expectations.iter().zip(&res_buf.expectations) {
                assert!((e1 - e2).abs() < 1e-12, "expectation mismatch at n={n_qubits}");
            }
            for (g1, g2) in res_std.gradient.iter().zip(&res_buf.gradient) {
                assert!((g1 - g2).abs() < 1e-12, "gradient mismatch at n={n_qubits}");
            }
        }
    }
}


