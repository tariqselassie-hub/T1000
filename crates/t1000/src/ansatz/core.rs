// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! `CircuitAnsatz`: hardware-efficient parameterized VQC with fused forward paths,
//! mid-circuit feed-forward, data re-uploading, and DAG compilation.

use super::layers::{apply_single_qubit_ry, entangling_pairs, PiPhaseLayer};
use super::pre_init::{apply_alt_bell_ghz, apply_bell_ghz, apply_iqft, apply_qft, PreInitCircuit};
use crate::state::QuantumState;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// Canonical Hardware-Efficient Parameterized Circuit Ansatz (W(θ)).
/// Alternates Single-Qubit Rotations (Ry, Rz) with Entanglement Blocks (CZ / CNOT).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitAnsatz {
    pub n_qubits: usize,
    pub dim: usize,
    pub depth: usize,
    pub weights: Vec<f64>,
    pub use_pi_layer: bool,
    pub pre_init: PreInitCircuit,
}

impl CircuitAnsatz {
    /// Deterministic ansatz with weights initialised from [`crate::DEFAULT_SEED`].
    pub fn new(n_qubits: usize, depth: usize) -> Self {
        Self::new_seeded(n_qubits, depth, crate::DEFAULT_SEED)
    }

    /// Deterministic ansatz with weights drawn uniformly from [-π, π) using `seed`.
    pub fn new_seeded(n_qubits: usize, depth: usize, seed: u64) -> Self {
        assert!(
            (1..usize::BITS as usize).contains(&n_qubits),
            "CircuitAnsatz needs between 1 and {} qubits (got {})",
            usize::BITS - 1,
            n_qubits
        );
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let num_params = 2 * n_qubits * depth;
        let weights: Vec<f64> = (0..num_params)
            .map(|_| rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI))
            .collect();

        Self {
            n_qubits,
            dim: 1 << n_qubits,
            depth,
            weights,
            use_pi_layer: false,
            pre_init: PreInitCircuit::None,
        }
    }

    pub fn with_pi_layer(mut self, enabled: bool) -> Self {
        self.use_pi_layer = enabled;
        if enabled {
            self.pre_init = PreInitCircuit::RademacherPi;
        } else if self.pre_init == PreInitCircuit::RademacherPi {
            self.pre_init = PreInitCircuit::None;
        }
        self
    }

    pub fn with_pre_init(mut self, pre_init: PreInitCircuit) -> Self {
        self.use_pi_layer = pre_init == PreInitCircuit::RademacherPi;
        self.pre_init = pre_init;
        self
    }

    pub fn with_bell_ghz(self) -> Self {
        self.with_pre_init(PreInitCircuit::BellGhz)
    }

    pub fn with_alt_bell_ghz(self) -> Self {
        self.with_pre_init(PreInitCircuit::AltBellGhz)
    }

    pub fn with_qft(self) -> Self {
        self.with_pre_init(PreInitCircuit::Qft)
    }

    pub fn with_iqft(self) -> Self {
        self.with_pre_init(PreInitCircuit::Iqft)
    }

    /// In-place pre-initialization functional applied to statevector prior to variational layers.
    pub fn apply_pre_init(&self, state: &mut QuantumState) {
        match self.pre_init {
            PreInitCircuit::None => {
                if self.use_pi_layer {
                    let pi_layer = PiPhaseLayer::new(self.n_qubits);
                    pi_layer.forward_in_place(state);
                }
            }
            PreInitCircuit::RademacherPi => {
                let pi_layer = PiPhaseLayer::new(self.n_qubits);
                pi_layer.forward_in_place(state);
            }
            PreInitCircuit::BellGhz => apply_bell_ghz(state, self.n_qubits),
            PreInitCircuit::AltBellGhz => apply_alt_bell_ghz(state, self.n_qubits),
            PreInitCircuit::Qft => apply_qft(state, self.n_qubits),
            PreInitCircuit::Iqft => apply_iqft(state, self.n_qubits),
        }
    }

    pub fn param_count(&self) -> usize {
        2 * self.n_qubits * self.depth
    }

    /// Update internal weights buffer.
    pub fn update_params(&mut self, weights: &[f64]) {
        assert_eq!(weights.len(), self.param_count());
        self.weights = weights.to_vec();
    }

    /// In-place forward pass on an existing state (zero allocations, zero copy).
    pub fn forward_in_place(&self, state: &mut QuantumState, weights: &[f64]) {
        assert_eq!(self.dim, state.dim);
        assert_eq!(weights.len(), self.param_count());

        self.apply_pre_init(state);

        for layer in 0..self.depth {
            self.forward_layer_into(layer, weights, state);

            if (layer + 1) % 4 == 0 || layer + 1 == self.depth {
                state.assert_normalized(1e-10);
            }
        }

        state.normalize();
    }

    /// In-place forward pass writing into a pre-allocated `out` buffer.
    /// Eliminates heap allocations and cache thrashing during 2P parameter-shift backward passes.
    pub fn forward_into(&self, state: &QuantumState, weights: &[f64], out: &mut QuantumState) {
        assert_eq!(self.dim, state.dim);
        assert_eq!(self.dim, out.dim);
        assert_eq!(weights.len(), self.param_count());

        out.copy_from(state);
        self.forward_in_place(out, weights);
    }

    /// Zero-allocation in-place evolution with terminal layer diagonal commutation pruning.
    pub fn forward_pruned_in_place(&self, state: &mut QuantumState, pruned_weights: &[f64]) {
        assert_eq!(self.dim, state.dim);
        assert!(self.depth >= 1, "Ansatz depth must be at least 1");
        let expected_len = 2 * self.n_qubits * (self.depth - 1) + self.n_qubits;
        assert_eq!(pruned_weights.len(), expected_len);

        self.apply_pre_init(state);

        for layer in 0..(self.depth - 1) {
            self.forward_layer_into(layer, pruned_weights, state);
        }

        let p_start = (self.depth - 1) * 2 * self.n_qubits;
        for q in 0..self.n_qubits {
            let theta_y = pruned_weights[p_start + q];
            apply_single_qubit_ry(state, q, theta_y);
        }

        state.normalize();
    }

    /// Zero-allocation in-place evolution stopping at end of Layer 2 (Gate 72).
    /// Statevector stops mutating here; terminal Ry rotation is evaluated via POVM Observable Fusion.
    pub fn forward_l2_in_place(&self, state: &mut QuantumState, pruned_weights: &[f64]) {
        assert_eq!(self.dim, state.dim);
        self.assert_full_layer_weights(pruned_weights);

        self.apply_pre_init(state);

        for layer in 0..(self.depth - 1) {
            self.forward_layer_into(layer, pruned_weights, state);
        }
    }

    /// In-place evolution with terminal layer diagonal commutation pruning.
    /// Expected weights length: 2 * n_qubits * (depth - 1) + n_qubits (e.g. 24 + 24 + 12 = 60).
    pub fn forward_pruned_into(
        &self,
        state: &QuantumState,
        pruned_weights: &[f64],
        out: &mut QuantumState,
    ) {
        assert_eq!(self.dim, state.dim);
        assert_eq!(self.dim, out.dim);
        assert!(self.depth >= 1, "Ansatz depth must be at least 1");
        let expected_len = 2 * self.n_qubits * (self.depth - 1) + self.n_qubits;
        assert_eq!(
            pruned_weights.len(),
            expected_len,
            "Pruned weights length mismatch: expected {expected_len}, got {}",
            pruned_weights.len()
        );

        out.copy_from(state);
        self.apply_pre_init(out);

        for layer in 0..(self.depth - 1) {
            self.forward_layer_into(layer, pruned_weights, out);
        }

        let p_start = (self.depth - 1) * 2 * self.n_qubits;
        for q in 0..self.n_qubits {
            let theta_y = pruned_weights[p_start + q];
            apply_single_qubit_ry(out, q, theta_y);
        }

        out.normalize();
    }

    /// In-place evolution stopping at the end of Layer (depth - 2).
    pub fn forward_l2_into(
        &self,
        state: &QuantumState,
        pruned_weights: &[f64],
        out: &mut QuantumState,
    ) {
        assert_eq!(self.dim, state.dim);
        assert_eq!(self.dim, out.dim);
        self.assert_full_layer_weights(pruned_weights);

        out.copy_from(state);
        self.apply_pre_init(out);

        for layer in 0..(self.depth - 1) {
            self.forward_layer_into(layer, pruned_weights, out);
        }

        out.normalize();
    }

    /// Full end-to-end forward evaluation with Next-Stage Observable Rotation Fusion:
    /// evaluates Pauli-Z via: ⟨Z_q⟩_final = cos(θ_q) ⟨Z_q⟩_L2 - sin(θ_q) ⟨X_q⟩_L2.
    pub fn forward_fused_readout_into(
        &self,
        state: &QuantumState,
        pruned_weights: &[f64],
        out_state: &mut QuantumState,
        z_final: &mut [f64],
    ) {
        self.assert_pruned_weights(pruned_weights);
        self.assert_readout_width(z_final.len());
        self.forward_l2_into(state, pruned_weights, out_state);
        let p_start = (self.depth - 1) * 2 * self.n_qubits;
        let n_eval = z_final.len();
        let theta_y = &pruned_weights[p_start..p_start + n_eval];
        crate::simd::measure_fused_ry_z_readout(&out_state.real, &out_state.imag, theta_y, z_final);
    }

    /// Full end-to-end forward evaluation extracting BOTH ⟨Z_q⟩ AND ⟨Z_i Z_j⟩ correlators.
    pub fn forward_fused_full_readout_into(
        &self,
        state: &QuantumState,
        pruned_weights: &[f64],
        out_state: &mut QuantumState,
        z_final: &mut [f64],
        zz_final: &mut [f64],
    ) {
        self.assert_pruned_weights(pruned_weights);
        self.assert_readout_width(z_final.len());
        self.forward_l2_into(state, pruned_weights, out_state);
        let p_start = (self.depth - 1) * 2 * self.n_qubits;
        let n_eval = z_final.len();
        let theta_y = &pruned_weights[p_start..p_start + n_eval];
        crate::simd::measure_fused_ry_full_readout(
            &out_state.real,
            &out_state.imag,
            theta_y,
            z_final,
            zz_final,
        );
    }

    /// MECHANISM #2: Mid-Circuit Feed-Forward In-Place.
    pub fn forward_midcircuit_feedforward_in_place(
        &self,
        state: &mut QuantumState,
        pruned_weights: &[f64],
        source_qubit: usize,
        feedforward_weights: &[f64],
        z_final: &mut [f64],
    ) -> f64 {
        self.assert_pruned_weights(pruned_weights);
        self.assert_readout_width(z_final.len());
        self.forward_l2_in_place(state, pruned_weights);
        let m_src = crate::simd::measure_single_qubit_z(&state.real, &state.imag, source_qubit);

        let n_eval = z_final.len();
        let dynamic_thetas =
            self.feedforward_thetas(pruned_weights, feedforward_weights, m_src, n_eval);

        crate::simd::measure_fused_ry_z_readout(
            &state.real,
            &state.imag,
            &dynamic_thetas[..n_eval],
            z_final,
        );

        m_src
    }

    /// MECHANISM #2 + full 2-body readout In-Place.
    pub fn forward_midcircuit_full_readout_in_place(
        &self,
        state: &mut QuantumState,
        pruned_weights: &[f64],
        source_qubit: usize,
        feedforward_weights: &[f64],
        z_final: &mut [f64],
        zz_final: &mut [f64],
    ) -> f64 {
        self.assert_pruned_weights(pruned_weights);
        self.assert_readout_width(z_final.len());
        self.forward_l2_in_place(state, pruned_weights);
        let m_src = crate::simd::measure_single_qubit_z(&state.real, &state.imag, source_qubit);

        let n_eval = z_final.len();
        let dynamic_thetas =
            self.feedforward_thetas(pruned_weights, feedforward_weights, m_src, n_eval);

        crate::simd::measure_fused_ry_full_readout(
            &state.real,
            &state.imag,
            &dynamic_thetas[..n_eval],
            z_final,
            zz_final,
        );

        m_src
    }

    /// MECHANISM #2: Mid-Circuit Feed-Forward into pre-allocated output buffer.
    pub fn forward_midcircuit_feedforward_into(
        &self,
        state: &QuantumState,
        pruned_weights: &[f64],
        source_qubit: usize,
        feedforward_weights: &[f64],
        out_state: &mut QuantumState,
        z_final: &mut [f64],
    ) -> f64 {
        self.assert_pruned_weights(pruned_weights);
        self.assert_readout_width(z_final.len());
        self.forward_l2_into(state, pruned_weights, out_state);
        let m_src =
            crate::simd::measure_single_qubit_z(&out_state.real, &out_state.imag, source_qubit);

        let n_eval = z_final.len();
        let dynamic_thetas =
            self.feedforward_thetas(pruned_weights, feedforward_weights, m_src, n_eval);

        crate::simd::measure_fused_ry_z_readout(
            &out_state.real,
            &out_state.imag,
            &dynamic_thetas[..n_eval],
            z_final,
        );

        m_src
    }

    /// Combined MECHANISM #1 + #2: Mid-Circuit Feed-Forward with Full 2-Body Readout.
    pub fn forward_midcircuit_full_readout_into(
        &self,
        state: &QuantumState,
        pruned_weights: &[f64],
        source_qubit: usize,
        feedforward_weights: &[f64],
        out_state: &mut QuantumState,
        z_final: &mut [f64],
        zz_final: &mut [f64],
    ) -> f64 {
        self.assert_pruned_weights(pruned_weights);
        self.assert_readout_width(z_final.len());
        self.forward_l2_into(state, pruned_weights, out_state);
        let m_src =
            crate::simd::measure_single_qubit_z(&out_state.real, &out_state.imag, source_qubit);

        let n_eval = z_final.len();
        let dynamic_thetas =
            self.feedforward_thetas(pruned_weights, feedforward_weights, m_src, n_eval);

        crate::simd::measure_fused_ry_full_readout(
            &out_state.real,
            &out_state.imag,
            &dynamic_thetas[..n_eval],
            z_final,
            zz_final,
        );

        m_src
    }

    /// Entanglement-Propagated Mid-Circuit Feed-Forward (L0 -> L1 In-Place Steering).
    pub fn forward_midcircuit_l0_feedforward_into(
        &self,
        state: &QuantumState,
        pruned_weights: &[f64],
        source_qubit: usize,
        feedforward_weights: &[f64],
        out_state: &mut QuantumState,
        z_final: &mut [f64],
    ) -> f64 {
        assert_eq!(self.dim, state.dim);
        assert_eq!(self.dim, out_state.dim);
        self.assert_multi_layer_depth();
        self.assert_pruned_weights(pruned_weights);
        self.assert_readout_width(z_final.len());
        out_state.copy_from(state);

        self.apply_pre_init(out_state);
        self.forward_layer_into(0, pruned_weights, out_state);

        let m_src =
            crate::simd::measure_single_qubit_z(&out_state.real, &out_state.imag, source_qubit);

        for q in 0..self.n_qubits.min(feedforward_weights.len()) {
            let alpha = feedforward_weights[q];
            if alpha.abs() > 1e-12 {
                apply_single_qubit_ry(out_state, q, alpha * m_src);
            }
        }

        for layer in 1..(self.depth - 1) {
            self.forward_layer_into(layer, pruned_weights, out_state);
        }
        out_state.normalize();

        let p_start = (self.depth - 1) * 2 * self.n_qubits;
        let n_eval = z_final.len();
        let theta_y = &pruned_weights[p_start..p_start + n_eval];
        crate::simd::measure_fused_ry_z_readout(&out_state.real, &out_state.imag, theta_y, z_final);

        m_src
    }

    /// MECHANISM #3: Data Re-Uploading In-Place Forward Evaluation.
    pub fn forward_reuploading_in_place(
        &self,
        state: &mut QuantumState,
        pruned_weights: &[f64],
        features: &[f64],
        upload_scales_l0: &[f64],
        source_qubit: usize,
        feedforward_alphas: &[f64],
        z_final: &mut [f64],
    ) -> f64 {
        assert_eq!(self.dim, state.dim);
        self.assert_multi_layer_depth();
        self.assert_pruned_weights(pruned_weights);
        self.assert_readout_width(z_final.len());
        self.apply_pre_init(state);

        self.forward_layer_into(0, pruned_weights, state);

        let norm_sq: f64 = features.iter().map(|v| v * v).sum();
        let inv_norm = 1.0 / norm_sq.sqrt().max(1e-12);
        let n_feats = features.len();

        for q in 0..self.n_qubits {
            let feat_val = if q < n_feats {
                features[q] * inv_norm
            } else {
                0.0
            };
            let scale = upload_scales_l0.get(q).copied().unwrap_or(0.0);
            if scale.abs() > 1e-12 {
                apply_single_qubit_ry(state, q, scale * feat_val);
            }
        }

        for layer in 1..(self.depth - 1) {
            self.forward_layer_into(layer, pruned_weights, state);
        }

        let m_src = crate::simd::measure_single_qubit_z(&state.real, &state.imag, source_qubit);
        let n_eval = z_final.len();
        let dynamic_thetas =
            self.feedforward_thetas(pruned_weights, feedforward_alphas, m_src, n_eval);

        crate::simd::measure_fused_ry_z_readout(
            &state.real,
            &state.imag,
            &dynamic_thetas[..n_eval],
            z_final,
        );

        m_src
    }

    /// Combined MECHANISMS #1 + #2 + #3: Full Re-Uploading + Feed-Forward + ZZ Readout.
    pub fn forward_reuploading_full_readout_in_place(
        &self,
        state: &mut QuantumState,
        pruned_weights: &[f64],
        features: &[f64],
        upload_scales_l0: &[f64],
        source_qubit: usize,
        feedforward_alphas: &[f64],
        z_final: &mut [f64],
        zz_final: &mut [f64],
    ) -> f64 {
        assert_eq!(self.dim, state.dim);
        self.assert_multi_layer_depth();
        self.assert_pruned_weights(pruned_weights);
        self.assert_readout_width(z_final.len());
        self.apply_pre_init(state);

        self.forward_layer_into(0, pruned_weights, state);

        let norm_sq: f64 = features.iter().map(|v| v * v).sum();
        let inv_norm = 1.0 / norm_sq.sqrt().max(1e-12);
        let n_feats = features.len();

        for q in 0..self.n_qubits {
            let feat_val = if q < n_feats {
                features[q] * inv_norm
            } else {
                0.0
            };
            let scale = upload_scales_l0.get(q).copied().unwrap_or(0.0);
            if scale.abs() > 1e-12 {
                apply_single_qubit_ry(state, q, scale * feat_val);
            }
        }

        for layer in 1..(self.depth - 1) {
            self.forward_layer_into(layer, pruned_weights, state);
        }

        let m_src = crate::simd::measure_single_qubit_z(&state.real, &state.imag, source_qubit);
        let n_eval = z_final.len();
        let dynamic_thetas =
            self.feedforward_thetas(pruned_weights, feedforward_alphas, m_src, n_eval);

        crate::simd::measure_fused_ry_full_readout(
            &state.real,
            &state.imag,
            &dynamic_thetas[..n_eval],
            z_final,
            zz_final,
        );

        m_src
    }

    // ── Assertion helpers ────────────────────────────────────────────────────

    fn assert_full_layer_weights(&self, weights: &[f64]) {
        assert!(self.depth >= 1, "Ansatz depth must be at least 1");
        let needed = 2 * self.n_qubits * (self.depth - 1);
        assert!(
            weights.len() >= needed,
            "Pruned weights length mismatch: need at least {needed} full-layer weights, got {}",
            weights.len()
        );
    }

    fn assert_pruned_weights(&self, weights: &[f64]) {
        assert!(self.depth >= 1, "Ansatz depth must be at least 1");
        let needed = 2 * self.n_qubits * (self.depth - 1) + self.n_qubits;
        assert!(
            weights.len() >= needed,
            "Pruned weights length mismatch: expected at least {needed}, got {}",
            weights.len()
        );
    }

    fn assert_readout_width(&self, n_eval: usize) {
        assert!(
            n_eval <= self.n_qubits,
            "readout of {n_eval} qubits requested from a {}-qubit ansatz",
            self.n_qubits
        );
        crate::simd::assert_fused_readout_width(n_eval);
    }

    fn assert_multi_layer_depth(&self) {
        assert!(
            self.depth >= 2,
            "this forward path needs depth >= 2 (got {})",
            self.depth
        );
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Terminal Ry angles modulated by the mid-circuit expectation: θ_q + α_q · m_src.
    fn feedforward_thetas(
        &self,
        pruned_weights: &[f64],
        alphas: &[f64],
        m_src: f64,
        n_eval: usize,
    ) -> [f64; crate::simd::MAX_FUSED_READOUT_QUBITS] {
        let p_start = (self.depth - 1) * 2 * self.n_qubits;
        let mut thetas = [0.0; crate::simd::MAX_FUSED_READOUT_QUBITS];
        for q in 0..n_eval {
            let alpha = alphas.get(q).copied().unwrap_or(0.0);
            thetas[q] = pruned_weights[p_start + q] + alpha * m_src;
        }
        thetas
    }

    /// In-place evolution for a single circuit layer: Ry on every qubit, then Rz + CZ block.
    pub fn forward_layer_into(&self, layer_idx: usize, weights: &[f64], out: &mut QuantumState) {
        assert_eq!(self.dim, out.dim);
        let n = self.n_qubits;
        let p_start = layer_idx * 2 * n;
        assert!(
            weights.len() >= p_start + 2 * n,
            "layer {layer_idx} needs weights up to index {}, got {}",
            p_start + 2 * n,
            weights.len()
        );

        let mut theta_ry = [0.0f64; 64];
        assert!(
            n <= theta_ry.len(),
            "{n} qubits exceed the layer angle buffer"
        );
        for (q, t) in theta_ry.iter_mut().enumerate().take(n) {
            *t = weights[p_start + 2 * q];
        }
        crate::layer_fusion::apply_ry_layer(&mut out.real, &mut out.imag, &theta_ry[..n]);

        let theta_rz: Vec<f64> = (0..n).map(|q| weights[p_start + 2 * q + 1]).collect();
        let pairs: Vec<(usize, usize)> = entangling_pairs(n, layer_idx).collect();
        crate::diagonal::apply_rz_cz_layer(&mut out.real, &mut out.imag, &theta_rz, &pairs);
    }

    /// Forward pass through the parameterized ansatz W(θ): |psi_out> = W(θ) |psi_in>.
    pub fn forward(&self, state: &QuantumState, weights: &[f64]) -> QuantumState {
        let mut out = QuantumState::new(self.dim);
        self.forward_into(state, weights, &mut out);
        out
    }

    /// Converts the parameterized ansatz into a dependency DAG for compilation and scheduling.
    pub fn to_dag(&self, weights: &[f64]) -> crate::compiler::CircuitDag {
        assert_eq!(weights.len(), self.param_count());
        let mut builder = crate::compiler::CircuitBuilder::new(self.n_qubits);

        match self.pre_init {
            PreInitCircuit::None => {
                if self.use_pi_layer {
                    let signs: Vec<f64> = (0..self.n_qubits)
                        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                        .collect();
                    builder.rademacher(signs);
                }
            }
            PreInitCircuit::RademacherPi => {
                let signs: Vec<f64> = (0..self.n_qubits)
                    .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                    .collect();
                builder.rademacher(signs);
            }
            PreInitCircuit::BellGhz => {
                let qubits: Vec<usize> = (0..self.n_qubits).collect();
                builder.bell_ghz(&qubits);
            }
            PreInitCircuit::AltBellGhz => {
                let qubits: Vec<usize> = (0..self.n_qubits).collect();
                builder.alt_bell_ghz(&qubits);
            }
            PreInitCircuit::Qft => {
                let qubits: Vec<usize> = (0..self.n_qubits).collect();
                builder.qft(&qubits);
            }
            PreInitCircuit::Iqft => {
                let qubits: Vec<usize> = (0..self.n_qubits).collect();
                builder.iqft(&qubits);
            }
        }

        let mut p_idx = 0;
        for layer in 0..self.depth {
            for q in 0..self.n_qubits {
                let theta_y = weights[p_idx];
                let theta_z = weights[p_idx + 1];
                p_idx += 2;
                builder.ry(q, theta_y);
                builder.rz(q, theta_z);
            }
            for (q, target) in entangling_pairs(self.n_qubits, layer) {
                builder.cz(q, target);
            }
        }

        builder.dag
    }

    /// Compiles the ansatz through the DAG dependency tracker, gate fuser, and cache scheduler.
    pub fn compile(&self, weights: &[f64]) -> crate::compiler::CompiledCircuit {
        let dag = self.to_dag(weights);
        crate::compiler::CompiledCircuit::compile(&dag)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_ansatz_with_pre_init_dag_equivalence() {
        let n_qubits = 5;
        let depth = 2;
        let ansatz = CircuitAnsatz::new(n_qubits, depth).with_bell_ghz();
        let weights = ansatz.weights.clone();

        let input = QuantumState::new(1 << n_qubits);

        let mut state_direct = input.clone();
        ansatz.forward_in_place(&mut state_direct, &weights);

        let compiled = ansatz.compile(&weights);
        let mut state_compiled = input.clone();
        compiled.execute(&mut state_compiled);

        let fidelity = state_direct.fidelity(&state_compiled);
        assert!(
            (fidelity - 1.0).abs() < 1e-12,
            "Direct and DAG compiled evolution must match, got fidelity {fidelity}"
        );
    }
}
