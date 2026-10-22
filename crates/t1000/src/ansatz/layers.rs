// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Layer-level primitives used by `CircuitAnsatz`:
//! single-qubit Ry kernel, entangling pair generator, and the deep VQC layer types
//! (`VariationalQuantumLayer`, `DeepQuantumNeuralNetwork`, `PiPhaseLayer`).

use crate::state::QuantumState;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use t1000_core::{Hamiltonian, PhaseGate, QubitRegister};

// ── Internal kernel ──────────────────────────────────────────────────────────

/// Single-qubit Ry rotation applied in-place via the fused AVX2 kernel.
#[inline(always)]
pub fn apply_single_qubit_ry(state: &mut QuantumState, qubit: usize, theta: f64) {
    crate::simd::apply_ry_kernel(&mut state.real, &mut state.imag, qubit, theta);
}

// ── Entangling pair generator ────────────────────────────────────────────────

/// CZ pairs of the entangling block for `layer_idx`: `(q, (q + 2^(layer_idx mod 4)) mod n)`.
///
/// Each unordered pair is yielded exactly once. When `2 * stride == n` the naive ring visits
/// every pair twice (e.g. `(0, 8)` and `(8, 0)` for 16 qubits) and the two CZs cancel, so only
/// the first `stride` qubits emit a pair. A stride that wraps to 0 (n divides it) emits nothing.
pub fn entangling_pairs(
    n_qubits: usize,
    layer_idx: usize,
) -> impl Iterator<Item = (usize, usize)> {
    let stride = (1usize << (layer_idx % 4)) % n_qubits.max(1);
    let count = match stride {
        0 => 0,
        s if 2 * s == n_qubits => s,
        _ => n_qubits,
    };
    (0..count).map(move |q| (q, (q + stride) % n_qubits))
}

// ── VariationalQuantumLayer ──────────────────────────────────────────────────

/// Parameterized Variational Quantum Circuit (VQC) Layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariationalQuantumLayer {
    pub dim: usize,
    /// Rotational phase parameters θ_k ∈ [-π, π]
    pub phase_weights: Vec<f64>,
    /// Diagonal Hamiltonian energy eigenvalues E_k
    pub hamiltonian_energies: Vec<f64>,
    /// Evolution time step Δt
    pub dt: f64,
    /// Werner entanglement coupling coefficient p ∈ [0, 1]
    pub entangle_coupling: f64,
}

impl VariationalQuantumLayer {
    /// Deterministic layer initialised from [`crate::DEFAULT_SEED`].
    pub fn new(dim: usize, dt: f64, entangle_coupling: f64) -> Self {
        Self::new_seeded(dim, dt, entangle_coupling, crate::DEFAULT_SEED)
    }

    /// Deterministic layer initialised from `seed`.
    pub fn new_seeded(dim: usize, dt: f64, entangle_coupling: f64, seed: u64) -> Self {
        Self::from_rng(dim, dt, entangle_coupling, &mut ChaCha8Rng::seed_from_u64(seed))
    }

    /// Layer initialised from a caller-supplied RNG (lets stacked layers share one stream).
    pub fn from_rng<R: Rng>(dim: usize, dt: f64, entangle_coupling: f64, rng: &mut R) -> Self {
        let phase_weights: Vec<f64> = (0..dim)
            .map(|_| rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI))
            .collect();
        let hamiltonian_energies: Vec<f64> =
            (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

        Self {
            dim,
            phase_weights,
            hamiltonian_energies,
            dt,
            entangle_coupling: entangle_coupling.clamp(0.0, 1.0),
        }
    }

    /// Forward pass through the variational ansatz layer:
    /// 1. Phase rotation R_z(θ)
    /// 2. Hamiltonian time evolution exp(-i H Δt)
    /// 3. Werner entanglement diffusion
    pub fn forward(&self, state: &QuantumState) -> QuantumState {
        assert_eq!(self.dim, state.dim);
        let mut reg = state.to_register(42);

        let gate = PhaseGate::new(self.phase_weights.clone());
        gate.apply(&mut reg);

        let ham = Hamiltonian::new(self.dim, self.hamiltonian_energies.clone());
        ham.time_evolve(&mut reg, self.dt);

        if self.entangle_coupling > 0.0 {
            apply_entangling_coupler(&mut reg, self.entangle_coupling);
        }

        reg.normalize();
        QuantumState::from_register(&reg)
    }

    /// Parameter update via learning rate and energy gradient descent.
    pub fn update_parameters(&mut self, phase_grads: &[f64], energy_grads: &[f64], lr: f64) {
        assert_eq!(self.dim, phase_grads.len());
        assert_eq!(self.dim, energy_grads.len());

        for (w, &g) in self.phase_weights.iter_mut().zip(phase_grads.iter()) {
            *w -= lr * g;
            *w = (*w + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI)
                - std::f64::consts::PI;
        }

        for (e, &g) in self.hamiltonian_energies.iter_mut().zip(energy_grads.iter()) {
            *e -= lr * g;
            *e = e.clamp(-10.0, 10.0);
        }
    }
}

/// Applies nearest-neighbor bipartite entanglement mixing to emulate spatial patch coupling.
fn apply_entangling_coupler(reg: &mut QubitRegister, coupling: f64) {
    let n = reg.basis;
    let mut mixed = reg.amplitudes.clone();

    for (i, target) in mixed.iter_mut().enumerate() {
        let next = (i + 1) % n;
        let prev = (i + n - 1) % n;
        let neighbor_mix = (reg.amplitudes[prev] + reg.amplitudes[next]) * 0.5;
        *target = reg.amplitudes[i] * (1.0 - coupling) + neighbor_mix * coupling;
    }

    reg.amplitudes = mixed;
}

// ── DeepQuantumNeuralNetwork ─────────────────────────────────────────────────

/// Multi-layer Deep Quantum Neural Network (Deep QNN).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepQuantumNeuralNetwork {
    pub dim: usize,
    pub layers: Vec<VariationalQuantumLayer>,
}

impl DeepQuantumNeuralNetwork {
    /// Deterministic network initialised from [`crate::DEFAULT_SEED`].
    pub fn new(dim: usize, depth: usize, dt: f64, entangle_coupling: f64) -> Self {
        Self::new_seeded(dim, depth, dt, entangle_coupling, crate::DEFAULT_SEED)
    }

    /// Deterministic network initialised from `seed`; layers draw from one shared stream,
    /// so they are distinct from each other.
    pub fn new_seeded(
        dim: usize,
        depth: usize,
        dt: f64,
        entangle_coupling: f64,
        seed: u64,
    ) -> Self {
        assert!(depth >= 1, "Depth must be at least 1");
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let layers = (0..depth)
            .map(|_| VariationalQuantumLayer::from_rng(dim, dt, entangle_coupling, &mut rng))
            .collect();
        Self { dim, layers }
    }

    pub fn forward(&self, state: &QuantumState) -> QuantumState {
        let mut current = state.clone();
        for layer in &self.layers {
            current = layer.forward(&current);
        }
        current
    }
}

// ── PiPhaseLayer ─────────────────────────────────────────────────────────────

/// Deterministic Bipolar ±π Phase Modulation Layer (Walsh-Rademacher Scrambler).
/// Applies single-qubit R_z(s_q * π) with alternating signs s_q ∈ {+1.0, -1.0} across all N qubits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiPhaseLayer {
    pub n_qubits: usize,
    pub signs: Vec<f64>,
}

impl PiPhaseLayer {
    pub fn new(n_qubits: usize) -> Self {
        let signs: Vec<f64> = (0..n_qubits)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        Self { n_qubits, signs }
    }

    pub fn with_signs(signs: Vec<f64>) -> Self {
        let n_qubits = signs.len();
        Self { n_qubits, signs }
    }

    /// Forward pass applying ±π phase shifts across all qubits via fused AVX2 kernel.
    pub fn forward(&self, state: &QuantumState) -> QuantumState {
        let mut current = state.clone();
        self.forward_in_place(&mut current);
        current
    }

    /// In-place fused Walsh-Rademacher phase transformation (zero allocations, single AVX2 pass).
    pub fn forward_in_place(&self, state: &mut QuantumState) {
        crate::simd::apply_rademacher_phase_kernel(&mut state.real, &mut state.imag, &self.signs);
    }
}

/// Type alias for bare-metal Walsh-Rademacher phase operations.
pub type RademacherPhaseLayer = PiPhaseLayer;
