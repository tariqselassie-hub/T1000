// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Exact adjoint differentiation for the data re-uploading + feed-forward readout.
//!
//! The pipeline (`CircuitAnsatz::forward_reuploading_in_place`) is
//!
//! ```text
//! |encoded> -> pre-init -> L0 -> Ry(s_q * x_q) upload -> L1..L_{d-2} -> m = <Z_src>
//!            -> t_q = theta_q + alpha_q * m,   z_q = cos t_q <Z_q> - sin t_q <X_q>
//! ```
//!
//! The mid-circuit "measurement" is a read-only expectation reduction (the statevector is never
//! collapsed), so every readout is analytic in the state and the trained angles. Holding the
//! readout head fixed (envelope theorem at the ridge optimum), the chain rule collapses into
//!
//! ```text
//! dL/dtheta_q           = dL/dz_q * (-sin t_q <Z_q> - cos t_q <X_q>)   (closed form)
//! dL/dtheta_{j<p_start} = Im <lambda | dU/dtheta_j | phi>              (one backward sweep)
//! ```
//!
//! with the costate assembled from the composite observable
//! `lambda = sum_q (dL/dz_q cos t_q Z_q - dL/dz_q sin t_q X_q)|phi> + dL/dm Z_src|phi>`, where
//! `dL/dm = sum_q alpha_q dL/dz_q (-sin t_q <Z_q> - cos t_q <X_q>)` carries the feed-forward
//! chain rule. One fused adjoint sweep per sample replaces hundreds of finite-difference
//! circuit evaluations.

use crate::adjoint::{AdjointWorkspace, BatchAdjointWorkspace};
use crate::adjoint_fusion::{ry_grad_and_undo, rz_grad_all};
use crate::ansatz::layers::{apply_single_qubit_ry, entangling_pairs};
use crate::ansatz::CircuitAnsatz;
use crate::diagonal::apply_rz_cz_layer;
use crate::measurements::PauliObservable;
use crate::simd::guards::MAX_FUSED_READOUT_QUBITS;
use crate::simd::{measure_dual_zx_expectations, measure_single_qubit_z};
use crate::state::QuantumState;
use rayon::prelude::*;

/// Re-upload forward + readout on an already-encoded `state`, gate-for-gate identical to
/// [`CircuitAnsatz::forward_reuploading_in_place`] (same feature normalization, upload guards,
/// layer order, `<Z_src>` read and fused readout arithmetic). Returns the mid-circuit `<Z_src>`
/// expectation and writes the readout values; `state` holds the post-sweep state on return.
pub fn reuploading_forward_readout(
    ansatz: &CircuitAnsatz,
    state: &mut QuantumState,
    features: &[f64],
    params: &[f64],
    upload_scales: &[f64],
    feedforward_alphas: &[f64],
    source_qubit: usize,
    readout: &mut [f64],
) -> f64 {
    let n = ansatz.n_qubits;
    let depth = ansatz.depth;
    let p_start = (depth - 1) * 2 * n;
    let n_eval = readout.len();
    assert!(depth >= 2, "re-uploading pipeline needs depth >= 2");
    assert!(source_qubit < n, "source qubit out of range");
    assert!(
        n_eval <= MAX_FUSED_READOUT_QUBITS,
        "readout wider than the fused kernel"
    );
    assert!(
        params.len() >= p_start + n_eval,
        "parameter buffer shorter than the trainable blocks"
    );

    ansatz.apply_pre_init(state);
    ansatz.forward_layer_into(0, params, state);

    let norm_sq: f64 = features.iter().map(|v| v * v).sum();
    let inv_norm = 1.0 / norm_sq.sqrt().max(1e-12);
    for q in 0..n {
        let feat_val = if q < features.len() {
            features[q] * inv_norm
        } else {
            0.0
        };
        let scale = upload_scales.get(q).copied().unwrap_or(0.0);
        if scale.abs() > 1e-12 {
            apply_single_qubit_ry(state, q, scale * feat_val);
        }
    }
    for layer in 1..(depth - 1) {
        ansatz.forward_layer_into(layer, params, state);
    }

    let m_src = measure_single_qubit_z(&state.real, &state.imag, source_qubit);
    let mut z_l2 = [0.0; MAX_FUSED_READOUT_QUBITS];
    let mut x_l2 = [0.0; MAX_FUSED_READOUT_QUBITS];
    measure_dual_zx_expectations(
        &state.real,
        &state.imag,
        &mut z_l2[..n_eval],
        &mut x_l2[..n_eval],
    );
    for q in 0..n_eval {
        let alpha = feedforward_alphas.get(q).copied().unwrap_or(0.0);
        let (s, c) = (params[p_start + q] + alpha * m_src).sin_cos();
        readout[q] = c * z_l2[q] - s * x_l2[q];
    }
    m_src
}

/// Parallel exact-gradient driver over a mini-batch: one fused costate sweep per sample.
///
/// Each workspace state is encoded first via `encode(i, state)`; `features_of(i)` supplies the
/// sample's upload features and `z_gradient(i, readout)` the loss gradient in readout space
/// (`dL/dz_k`, one value per readout qubit). Returns the per-sample readouts in batch order;
/// `grad_out` (zeroed on entry, length `2*n*(depth-1) + n_eval`) receives the sample-ordered
/// gradient sum, matching the finite-difference summation convention.
#[allow(clippy::too_many_arguments)]
pub fn adjoint_reuploading_batch<'a>(
    ansatz: &CircuitAnsatz,
    source_qubit: usize,
    encode: impl Fn(usize, &mut QuantumState) + Sync,
    features_of: impl Fn(usize) -> &'a [f64] + Sync,
    params: &[f64],
    upload_scales: &[f64],
    feedforward_alphas: &[f64],
    z_gradient: impl Fn(usize, &[f64]) -> Vec<f64> + Sync,
    ws: &mut BatchAdjointWorkspace,
    count: usize,
    grad_out: &mut [f64],
) -> Vec<Vec<f64>> {
    let n_eval = feedforward_alphas.len().min(ansatz.n_qubits);
    let p_len = (ansatz.depth - 1) * 2 * ansatz.n_qubits + n_eval;
    assert!(
        grad_out.len() == p_len,
        "grad_out must cover the sweep + readout parameter blocks"
    );
    assert!(
        count <= ws.workspaces.len(),
        "batch exceeds the pre-allocated workspace capacity"
    );

    for g in grad_out.iter_mut() {
        *g = 0.0;
    }

    let per_sample: Vec<(Vec<f64>, Vec<f64>)> = (0..count)
        .into_par_iter()
        .zip(ws.workspaces.par_iter_mut())
        .map(|(i, w)| {
            encode(i, &mut w.phi);
            let features = features_of(i);
            let mut readout = vec![0.0; n_eval];
            let m_src = reuploading_forward_readout(
                ansatz,
                &mut w.phi,
                features,
                params,
                upload_scales,
                feedforward_alphas,
                source_qubit,
                &mut readout,
            );
            let dz = z_gradient(i, &readout);
            let grad = assemble_costate_and_backward(
                ansatz,
                w,
                features,
                params,
                upload_scales,
                feedforward_alphas,
                &dz,
                source_qubit,
                m_src,
            );
            (readout, grad)
        })
        .collect();

    for (_, grad) in &per_sample {
        for (out, &g) in grad_out.iter_mut().zip(grad) {
            *out += g;
        }
    }
    per_sample.into_iter().map(|(r, _)| r).collect()
}

/// Assembles the composite costate, applies the closed-form readout-angle gradients and runs
/// the reverse-mode sweep over the actually-applied layers. Returns the gradient vector.
#[allow(clippy::too_many_arguments)] // mirrors the pipeline's natural argument list
fn assemble_costate_and_backward(
    ansatz: &CircuitAnsatz,
    w: &mut AdjointWorkspace,
    features: &[f64],
    params: &[f64],
    upload_scales: &[f64],
    alphas: &[f64],
    dz: &[f64],
    source_qubit: usize,
    m_src: f64,
) -> Vec<f64> {
    let n = ansatz.n_qubits;
    let p_start = (ansatz.depth - 1) * 2 * n;
    let n_eval = alphas.len().min(n);

    let mut z_l2 = [0.0; MAX_FUSED_READOUT_QUBITS];
    let mut x_l2 = [0.0; MAX_FUSED_READOUT_QUBITS];
    measure_dual_zx_expectations(
        &w.phi.real,
        &w.phi.imag,
        &mut z_l2[..n_eval],
        &mut x_l2[..n_eval],
    );

    let mut grad = vec![0.0; p_start + n_eval];
    let mut m_chain = 0.0;
    for q in 0..n_eval {
        let alpha = alphas.get(q).copied().unwrap_or(0.0);
        let (s, c) = (params[p_start + q] + alpha * m_src).sin_cos();
        grad[p_start + q] = dz[q] * (-s * z_l2[q] - c * x_l2[q]);
        m_chain += alpha * grad[p_start + q];
        if c != 0.0 {
            PauliObservable::Z(q).accumulate_scaled(
                &w.phi,
                dz[q] * c,
                &mut w.lambda.real,
                &mut w.lambda.imag,
            );
        }
        if s != 0.0 {
            PauliObservable::X(q).accumulate_scaled(
                &w.phi,
                -dz[q] * s,
                &mut w.lambda.real,
                &mut w.lambda.imag,
            );
        }
    }
    if m_chain != 0.0 {
        // Linear accumulation: adding Z(src)|phi> merges with any cost_z[src] term.
        PauliObservable::Z(source_qubit).accumulate_scaled(
            &w.phi,
            m_chain,
            &mut w.lambda.real,
            &mut w.lambda.imag,
        );
    }

    backward_sweep(ansatz, w, features, params, upload_scales, &mut grad);
    grad
}

/// Reverse-mode sweep: layers `1..=d-2` (post-upload), upload undo (parameter-independent, so
/// no gradients), then layer 0. Undoes every applied gate on both `phi` and `lambda`.
fn backward_sweep(
    ansatz: &CircuitAnsatz,
    w: &mut AdjointWorkspace,
    features: &[f64],
    params: &[f64],
    upload_scales: &[f64],
    grad: &mut [f64],
) {
    for layer in (1..(ansatz.depth - 1)).rev() {
        sweep_layer(ansatz, w, params, layer, grad);
    }
    undo_uploads(ansatz, &mut w.phi, &mut w.lambda, features, upload_scales);
    sweep_layer(ansatz, w, params, 0, grad);
}

/// One layer's Rz+CZ gradient block, diagonal undo, then the fused Ry gradient + undo,
/// replicating the standard adjoint sweep exactly.
fn sweep_layer(
    ansatz: &CircuitAnsatz,
    w: &mut AdjointWorkspace,
    params: &[f64],
    layer_idx: usize,
    grad: &mut [f64],
) {
    let n = ansatz.n_qubits;
    let base = layer_idx * 2 * n;
    let pairs: Vec<(usize, usize)> = entangling_pairs(n, layer_idx).collect();

    w.theta_rz.clear();
    w.theta_rz.extend((0..n).map(|q| params[base + 2 * q + 1]));
    rz_grad_all(&w.phi, &w.lambda, &mut w.rz_grad);
    for q in 0..n {
        grad[base + 2 * q + 1] = w.rz_grad[q];
    }

    w.neg_rz.clear();
    w.neg_rz.extend(w.theta_rz.iter().map(|t| -t));
    for s in [&mut w.phi, &mut w.lambda] {
        apply_rz_cz_layer(&mut s.real, &mut s.imag, &w.neg_rz, &pairs);
    }

    w.theta_ry.clear();
    w.theta_ry.extend((0..n).map(|q| params[base + 2 * q]));
    ry_grad_and_undo(&mut w.phi, &mut w.lambda, &w.theta_ry, &mut w.ry_grad);
    for q in 0..n {
        grad[base + 2 * q] = w.ry_grad[q];
    }
}

/// Undoes the data re-upload `Ry(s_q * x_q)` gates on both states with negated angles.
/// The upload scales are parameter-independent François into the loss in this regime, so no
/// gradient is taken here (the driver's parameter layout covers circuit angles only).
fn undo_uploads(
    ansatz: &CircuitAnsatz,
    phi: &mut QuantumState,
    lambda: &mut QuantumState,
    features: &[f64],
    upload_scales: &[f64],
) {
    let n = ansatz.n_qubits;
    let norm_sq: f64 = features.iter().map(|v| v * v).sum();
    let inv_norm = 1.0 / norm_sq.sqrt().max(1e-12);
    for q in 0..n {
        let feat_val = if q < features.len() {
            features[q] * inv_norm
        } else {
            0.0
        };
        let scale = upload_scales.get(q).copied().unwrap_or(0.0);
        if scale.abs() > 1e-12 {
            let angle = -(scale * feat_val);
            apply_single_qubit_ry(phi, q, angle);
            apply_single_qubit_ry(lambda, q, angle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DataEncodingLayer, EncodingKind};

    const N_QUBITS: usize = 12;
    const DIM: usize = 1 << N_QUBITS;
    const SOURCE: usize = 8;

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

    fn molecule(seed: usize) -> Vec<f64> {
        (0..19)
            .map(|i| ((i * 5 + seed * 7) % 13) as f64 / 6.0 + 0.1)
            .collect()
    }

    fn linear_loss(
        ansatz: &CircuitAnsatz,
        encoding: &DataEncodingLayer,
        params: &[f64],
        dru: &[f64],
        alphas: &[f64],
        feats: &[Vec<f64>],
        v: &[f64],
    ) -> f64 {
        let mut state = QuantumState::new(DIM);
        let mut z = [0.0; 12];
        let mut loss = 0.0;
        for s in feats {
            encoding.encode_amplitude_into(s, &mut state);
            ansatz.forward_reuploading_in_place(&mut state, params, s, dru, SOURCE, alphas, &mut z);
            loss += (0..12).map(|q| v[q] * z[q]).sum::<f64>();
        }
        loss
    }

    #[test]
    fn forward_readout_matches_the_reference_pipeline_bitwise() {
        let (ansatz, encoding, params, dru, alphas) = circuit();
        let feats = [molecule(1), molecule(4)];
        let mut z_ref = [0.0; 12];
        let mut z_kernel = [0.0; 12];

        let mut state = QuantumState::new(DIM);
        encoding.encode_amplitude_into(&feats[0], &mut state);
        let m_ref = ansatz.forward_reuploading_in_place(
            &mut state, &params, &feats[0], &dru, SOURCE, &alphas, &mut z_ref,
        );

        let mut ws = AdjointWorkspace::new(DIM, N_QUBITS);
        encoding.encode_amplitude_into(&feats[0], &mut ws.phi);
        let m_kernel = reuploading_forward_readout(
            &ansatz,
            &mut ws.phi,
            &feats[0],
            &params,
            &dru,
            &alphas,
            SOURCE,
            &mut z_kernel,
        );

        assert_eq!(m_ref.to_bits(), m_kernel.to_bits());
        for q in 0..12 {
            assert_eq!(z_ref[q].to_bits(), z_kernel[q].to_bits());
        }
    }

    #[test]
    fn adjoint_gradient_matches_central_finite_difference() {
        let (ansatz, encoding, params, dru, alphas) = circuit();
        let feats: Vec<Vec<f64>> = (1..=3).map(molecule).collect();
        let v: Vec<f64> = (0..12).map(|q| 0.25 * f64::from(q) - 1.0).collect();

        let mut ws = BatchAdjointWorkspace::new(DIM, N_QUBITS, feats.len());
        let mut grad = vec![0.0; params.len()];
        adjoint_reuploading_batch(
            &ansatz,
            SOURCE,
            |i, state| encoding.encode_amplitude_into(&feats[i], state),
            |i| feats[i].as_slice(),
            &params,
            &dru,
            &alphas,
            |_, _| v.clone(),
            &mut ws,
            feats.len(),
            &mut grad,
        );

        let eps = 0.01;
        for i in 0..params.len() {
            let mut p_plus = params.clone();
            p_plus[i] += eps;
            let mut p_minus = params.clone();
            p_minus[i] -= eps;
            let fd = (linear_loss(&ansatz, &encoding, &p_plus, &dru, &alphas, &feats, &v)
                - linear_loss(&ansatz, &encoding, &p_minus, &dru, &alphas, &feats, &v))
                / (2.0 * eps);
            let tol = 2e-3 + 0.02 * fd.abs();
            assert!(
                (grad[i] - fd).abs() <= tol,
                "component {i}: adjoint {:.9} vs finite-difference {:.9}",
                grad[i],
                fd
            );
        }
    }
}
