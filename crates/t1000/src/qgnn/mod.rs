// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Geometric Quantum Graph Neural Network (QGNN) Engine
//!
//! Implements Lie-algebraic equivariant quantum graph convolutions for molecular latent
//! space exploration. Preserves permutation and physical symmetries, provably immune
//! to McClean et al. barren plateaus via polynomial dynamical Lie algebra scaling.

pub mod barren_plateau;
pub mod convolution;
pub mod graph;

pub use barren_plateau::{
    compute_hea_gradient_variance, compute_qgnn_gradient_variance, BarrenPlateauReport,
};
pub use convolution::{
    apply_heisenberg_exchange, apply_xy_rotation_kernel, apply_zz_phase_kernel,
    EquivariantGraphConvLayer, NUM_BOND_CLASSES,
};
pub use graph::{GraphEdge, MolecularGraph, NodeFeatures, NODE_FEATURE_DIM};

use crate::simd::measurements::measure_single_qubit_z;
use crate::state::QuantumState;
use serde::{Deserialize, Serialize};

/// High-level Geometric Quantum Graph Neural Network Model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometricQgnn {
    pub num_nodes: usize,
    pub layers: Vec<EquivariantGraphConvLayer>,
    /// Linear readout weights for [mean_Z, mean_ZZ, graph_dirichlet].
    pub readout_weights: [f64; 3],
    pub readout_bias: f64,
}

impl GeometricQgnn {
    /// Constructs a multi-layer Geometric QGNN.
    pub fn new(num_nodes: usize, depth: usize, seed: u64) -> Self {
        let mut layers = Vec::with_capacity(depth);
        for d in 0..depth {
            layers.push(EquivariantGraphConvLayer::new(
                num_nodes,
                seed + (d * 101) as u64,
            ));
        }
        Self {
            num_nodes,
            layers,
            readout_weights: [0.5, 0.3, -0.2],
            readout_bias: 0.0,
        }
    }

    /// Evaluates forward graph quantum state evolution and extracts graph observables.
    pub fn forward_state(&self, graph: &MolecularGraph) -> QuantumState {
        let dim = 1usize << graph.num_nodes;
        let mut state = QuantumState::ground(dim);
        for layer in &self.layers {
            layer.forward(&mut state, graph);
        }
        state
    }

    /// Computes permutation-invariant graph observables: [mean_Z, mean_ZZ, dirichlet_energy].
    pub fn extract_graph_observables(&self, graph: &MolecularGraph) -> [f64; 3] {
        let state = self.forward_state(graph);
        let n = graph.num_nodes;

        // 1. Single-qubit node expectation values <Z_v>
        let mut z_nodes = vec![0.0; n];
        for v in 0..n {
            z_nodes[v] = measure_single_qubit_z(&state.real, &state.imag, v);
        }
        let mean_z = z_nodes.iter().sum::<f64>() / (n as f64);

        // 2. Edge correlators <Z_u Z_v> and Dirichlet Laplacian
        let mut mean_zz = 0.0;
        let mut dirichlet = 0.0;
        if !graph.edges.is_empty() {
            let mut sum_zz = 0.0;
            let mut sum_dirichlet = 0.0;
            let dim = state.dim;

            for edge in &graph.edges {
                let u_mask = 1usize << edge.u;
                let v_mask = 1usize << edge.v;
                let mut zz_val = 0.0;
                for k in 0..dim {
                    let prob = state.real[k].powi(2) + state.imag[k].powi(2);
                    let sign = if ((k & u_mask) != 0) == ((k & v_mask) != 0) {
                        1.0
                    } else {
                        -1.0
                    };
                    zz_val += sign * prob;
                }
                sum_zz += zz_val;
                sum_dirichlet += (z_nodes[edge.u] - z_nodes[edge.v]).powi(2);
            }
            let m = graph.edges.len() as f64;
            mean_zz = sum_zz / m;
            dirichlet = sum_dirichlet / m;
        }

        [mean_z, mean_zz, dirichlet]
    }

    /// Predicts a continuous scalar molecular property (e.g. logP, QED, affinity score).
    pub fn predict(&self, graph: &MolecularGraph) -> f64 {
        let obs = self.extract_graph_observables(graph);
        self.readout_weights[0] * obs[0]
            + self.readout_weights[1] * obs[1]
            + self.readout_weights[2] * obs[2]
            + self.readout_bias
    }

    /// Trains the model for a single epoch across a batch of molecular graphs.
    pub fn train_step(&mut self, dataset: &[(&MolecularGraph, f64)], lr: f64) -> f64 {
        let mut total_loss = 0.0;
        let mut grad_w = [0.0; 3];
        let mut grad_b = 0.0;
        let batch_size = dataset.len() as f64;

        for &(graph, target) in dataset {
            let obs = self.extract_graph_observables(graph);
            let pred = self.readout_weights[0] * obs[0]
                + self.readout_weights[1] * obs[1]
                + self.readout_weights[2] * obs[2]
                + self.readout_bias;
            let err = pred - target;
            total_loss += err.powi(2);

            for i in 0..3 {
                grad_w[i] += (2.0 / batch_size) * err * obs[i];
            }
            grad_b += (2.0 / batch_size) * err;
        }

        // Apply gradient step to readout parameters
        for i in 0..3 {
            self.readout_weights[i] -= lr * grad_w[i];
        }
        self.readout_bias -= lr * grad_b;

        total_loss / batch_size
    }
}
