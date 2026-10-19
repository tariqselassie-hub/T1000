// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Barren Plateau Diagnostics & Lie-Algebraic Scaling Verifier.
//!
//! Empirically validates that Lie-algebraic equivariant graph convolution circumvents
//! McClean et al. barren plateaus, exhibiting polynomial gradient variance scaling O(1/poly(N))
//! rather than exponential collapse O(2^-N).

use super::convolution::EquivariantGraphConvLayer;
use super::graph::MolecularGraph;
use crate::simd::measurements::measure_single_qubit_z;
use crate::simd::rotations::{apply_cz_kernel, apply_ry_kernel, apply_rz_kernel};
use crate::state::QuantumState;
use serde::{Deserialize, Serialize};

/// Telemetry report capturing comparative gradient variance and Lie algebra dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarrenPlateauReport {
    pub num_qubits: usize,
    pub num_samples: usize,
    pub hea_gradient_variance: f64,
    pub qgnn_gradient_variance: f64,
    pub variance_ratio: f64,
    pub hea_lie_dim: f64,
    pub qgnn_lie_dim: usize,
    pub is_resilient: bool,
}

/// Evaluates empirical gradient variance for a standard Hardware-Efficient Ansatz (HEA).
pub fn compute_hea_gradient_variance(
    num_qubits: usize,
    depth: usize,
    num_samples: usize,
    seed: u64,
) -> f64 {
    let dim = 1usize << num_qubits;
    let mut rng_state = seed ^ 0xFEED_CAFE;
    let mut next_f64 = || -> f64 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state as f64 / u64::MAX as f64) * 2.0 * std::f64::consts::PI - std::f64::consts::PI
    };

    let mut gradients = Vec::with_capacity(num_samples);
    let mut state_plus = QuantumState::ground(dim);
    let mut state_minus = QuantumState::ground(dim);
    let shift = std::f64::consts::FRAC_PI_2;

    for _ in 0..num_samples {
        let angles: Vec<f64> = (0..(depth * num_qubits * 2)).map(|_| next_f64()).collect();

        // Parameter shift on the first parameter (target qubit 0, layer 0)
        eval_hea(&mut state_plus, num_qubits, depth, &angles, 0, shift);
        eval_hea(&mut state_minus, num_qubits, depth, &angles, 0, -shift);

        let exp_plus = measure_single_qubit_z(&state_plus.real, &state_plus.imag, 0);
        let exp_minus = measure_single_qubit_z(&state_minus.real, &state_minus.imag, 0);
        let grad = 0.5 * (exp_plus - exp_minus);
        gradients.push(grad);
    }

    variance(&gradients)
}

/// Evaluates empirical gradient variance for the Geometric Equivariant QGNN.
pub fn compute_qgnn_gradient_variance(
    graph: &MolecularGraph,
    depth: usize,
    num_samples: usize,
    seed: u64,
) -> f64 {
    let dim = 1usize << graph.num_nodes;
    let mut gradients = Vec::with_capacity(num_samples);
    let mut state_plus = QuantumState::ground(dim);
    let mut state_minus = QuantumState::ground(dim);
    let shift = std::f64::consts::FRAC_PI_2;

    for s in 0..num_samples {
        let mut layers = Vec::with_capacity(depth);
        for d in 0..depth {
            layers.push(EquivariantGraphConvLayer::new(
                graph.num_nodes,
                seed + (s * 100 + d) as u64,
            ));
        }

        // Perturb the node rotation parameter in layer 0
        let mut layers_plus = layers.clone();
        let mut layers_minus = layers;
        layers_plus[0].node_bias_y += shift;
        layers_minus[0].node_bias_y -= shift;

        state_plus.real.fill(0.0);
        state_plus.imag.fill(0.0);
        state_plus.real[0] = 1.0;
        for layer in &layers_plus {
            layer.forward(&mut state_plus, graph);
        }

        state_minus.real.fill(0.0);
        state_minus.imag.fill(0.0);
        state_minus.real[0] = 1.0;
        for layer in &layers_minus {
            layer.forward(&mut state_minus, graph);
        }

        let exp_plus = measure_single_qubit_z(&state_plus.real, &state_plus.imag, 0);
        let exp_minus = measure_single_qubit_z(&state_minus.real, &state_minus.imag, 0);
        let grad = 0.5 * (exp_plus - exp_minus);
        gradients.push(grad);
    }

    variance(&gradients)
}

/// Helper evaluating a standard Hardware-Efficient Ansatz with circular CZ entangling.
fn eval_hea(
    state: &mut QuantumState,
    num_qubits: usize,
    depth: usize,
    angles: &[f64],
    target_idx: usize,
    pert: f64,
) {
    state.real.fill(0.0);
    state.imag.fill(0.0);
    state.real[0] = 1.0;

    let mut idx = 0;
    for _ in 0..depth {
        for q in 0..num_qubits {
            let theta_y = angles[idx] + if idx == target_idx { pert } else { 0.0 };
            idx += 1;
            let theta_z = angles[idx] + if idx == target_idx { pert } else { 0.0 };
            idx += 1;
            apply_ry_kernel(&mut state.real, &mut state.imag, q, theta_y);
            apply_rz_kernel(&mut state.real, &mut state.imag, q, theta_z);
        }
        for q in 0..num_qubits {
            let next_q = (q + 1) % num_qubits;
            apply_cz_kernel(&mut state.real, &mut state.imag, q, next_q);
        }
    }
    state.normalize();
}

/// Helper computing sample variance.
fn variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let sum_sq_diff: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum();
    sum_sq_diff / (n - 1.0)
}
