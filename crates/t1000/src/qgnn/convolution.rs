// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Lie-algebraic equivariant quantum graph convolution kernels and layers.
//!
//! Evaluates permutation-equivariant node rotations and Heisenberg exchange entangling
//! operations with zero heap allocation in hot execution loops.

use super::graph::{MolecularGraph, NODE_FEATURE_DIM};
use crate::simd::rotations::{apply_ry_kernel, apply_rz_kernel};
use crate::state::QuantumState;
use serde::{Deserialize, Serialize};

/// Number of discrete bond classes parameterized in edge convolutions.
pub const NUM_BOND_CLASSES: usize = 4;

/// A single Lie-algebraic equivariant Quantum Graph Convolution (QGCN) Layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquivariantGraphConvLayer {
    /// Dimension of the Hilbert space register (2^N).
    pub dim: usize,
    /// Number of active atom qubits in the graph.
    pub num_nodes: usize,
    /// Shared node feature projection weights for Ry rotations.
    pub node_weight_y: [f64; NODE_FEATURE_DIM],
    pub node_bias_y: f64,
    /// Shared node feature projection weights for Rz rotations.
    pub node_weight_z: [f64; NODE_FEATURE_DIM],
    pub node_bias_z: f64,
    /// Shared exchange interaction angles for each bond class.
    pub bond_couplings: [f64; NUM_BOND_CLASSES],
    /// Anisotropic ratio for ZZ relative to XY exchange (1.0 = isotropic Heisenberg).
    pub anisotropy: f64,
}

impl EquivariantGraphConvLayer {
    /// Constructs a layer initialized with deterministic small parameters.
    pub fn new(num_nodes: usize, seed: u64) -> Self {
        let dim = 1usize << num_nodes;
        let mut s = seed ^ 0x5EED_0042;
        let mut next_f64 = || -> f64 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s as f64 / u64::MAX as f64) - 0.5) * 0.4
        };

        let mut node_weight_y = [0.0; NODE_FEATURE_DIM];
        let mut node_weight_z = [0.0; NODE_FEATURE_DIM];
        for i in 0..NODE_FEATURE_DIM {
            node_weight_y[i] = next_f64();
            node_weight_z[i] = next_f64();
        }

        let mut bond_couplings = [0.0; NUM_BOND_CLASSES];
        for i in 0..NUM_BOND_CLASSES {
            bond_couplings[i] = next_f64();
        }

        Self {
            dim,
            num_nodes,
            node_weight_y,
            node_bias_y: next_f64(),
            node_weight_z,
            node_bias_z: next_f64(),
            bond_couplings,
            anisotropy: 1.0,
        }
    }

    /// Evaluates forward graph quantum state evolution on a molecular topology.
    pub fn forward(&self, state: &mut QuantumState, graph: &MolecularGraph) {
        assert_eq!(
            state.dim,
            1usize << graph.num_nodes,
            "State dimension must match 2^(graph.num_nodes)"
        );

        // 1. Equivariant Node Rotation Pass
        for (v, feats) in graph.node_features.iter().enumerate() {
            let theta_y = dot_features(&self.node_weight_y, feats) + self.node_bias_y;
            let theta_z = dot_features(&self.node_weight_z, feats) + self.node_bias_z;
            apply_ry_kernel(&mut state.real, &mut state.imag, v, theta_y);
            apply_rz_kernel(&mut state.real, &mut state.imag, v, theta_z);
        }

        // 2. Lie-Equivariant Heisenberg Exchange Pass across Graph Edges
        for edge in &graph.edges {
            let b_class = edge.bond_class();
            let phi = self.bond_couplings[b_class];
            apply_heisenberg_exchange(
                &mut state.real,
                &mut state.imag,
                edge.u,
                edge.v,
                phi,
                self.anisotropy,
            );
        }

        state.normalize();
    }
}

/// Inlined dot product for 4-element node feature vectors.
#[inline(always)]
fn dot_features(w: &[f64; NODE_FEATURE_DIM], x: &[f64; NODE_FEATURE_DIM]) -> f64 {
    w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + w[3] * x[3]
}

/// Applies Heisenberg exchange interaction U = exp(-i phi/2 (XX + YY + delta ZZ)).
#[inline(always)]
pub fn apply_heisenberg_exchange(
    real: &mut [f64],
    imag: &mut [f64],
    q1: usize,
    q2: usize,
    phi: f64,
    delta: f64,
) {
    if phi.abs() < 1e-12 {
        return;
    }
    apply_zz_phase_kernel(real, imag, q1, q2, phi * delta);
    apply_xy_rotation_kernel(real, imag, q1, q2, phi);
}

/// In-place diagonal ZZ phase rotation exp(-i theta/2 Z_1 Z_2).
#[inline(always)]
pub fn apply_zz_phase_kernel(
    real: &mut [f64],
    imag: &mut [f64],
    q1: usize,
    q2: usize,
    theta: f64,
) {
    let half_theta = theta * 0.5;
    let (cos_val, sin_val) = (half_theta.cos(), half_theta.sin());
    let u_shift = 1usize << q1;
    let v_shift = 1usize << q2;
    let len = real.len();

    for k in 0..len {
        let same_parity = ((k & u_shift) != 0) == ((k & v_shift) != 0);
        let s = if same_parity { -sin_val } else { sin_val };
        let r = real[k];
        let i = imag[k];
        real[k] = r * cos_val - i * s;
        imag[k] = r * s + i * cos_val;
    }
}

/// In-place planar XY Givens rotation exp(-i phi/2 (X_1 X_2 + Y_1 Y_2)).
#[inline(always)]
pub fn apply_xy_rotation_kernel(
    real: &mut [f64],
    imag: &mut [f64],
    q1: usize,
    q2: usize,
    phi: f64,
) {
    let (cos_val, sin_val) = (phi.cos(), phi.sin());
    let low = q1.min(q2);
    let high = q1.max(q2);
    let low_bit = 1usize << low;
    let high_bit = 1usize << high;
    let len = real.len();

    let mut k = 0;
    while k < len {
        if (k & low_bit) == 0 && (k & high_bit) != 0 {
            let idx_10 = k;
            let idx_01 = (k ^ high_bit) | low_bit;

            let (r01, i01) = (real[idx_01], imag[idx_01]);
            let (r10, i10) = (real[idx_10], imag[idx_10]);

            // [01] -> c * [01] - i * s * [10]
            real[idx_01] = cos_val * r01 + sin_val * i10;
            imag[idx_01] = cos_val * i01 - sin_val * r10;

            // [10] -> -i * s * [01] + c * [10]
            real[idx_10] = cos_val * r10 + sin_val * i01;
            imag[idx_10] = cos_val * i10 - sin_val * r01;
        }
        k += 1;
    }
}
