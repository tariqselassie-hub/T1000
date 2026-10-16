pub mod cut_bridge;
pub mod dispatch;
pub mod reupload;

pub use cut_bridge::{
    eval_36q_bipartite_bridge, eval_36q_bipartite_midcircuit, eval_cut_bridge_general,
    CutBridgeReport,
};
pub use dispatch::*;
pub use reupload::DualAxisEncoder;

use crate::state::QuantumState;
use t1000_core::{Hamiltonian, PhaseGate, QubitRegister};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

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
        let hamiltonian_energies: Vec<f64> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

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

        // 1. Parameterized Phase Gate
        let gate = PhaseGate::new(self.phase_weights.clone());
        gate.apply(&mut reg);

        // 2. Hamiltonian Time Evolution
        let ham = Hamiltonian::new(self.dim, self.hamiltonian_energies.clone());
        ham.time_evolve(&mut reg, self.dt);

        // 3. Werner Entanglement Coupling across neighboring modes
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
            // Wrap into [-π, π]
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
        // Entangling mixer: W_p Werner superposition with neighbors
        let neighbor_mix = (reg.amplitudes[prev] + reg.amplitudes[next]) * 0.5;
        *target = reg.amplitudes[i] * (1.0 - coupling) + neighbor_mix * coupling;
    }

    reg.amplitudes = mixed;
}

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
    pub fn new_seeded(dim: usize, depth: usize, dt: f64, entangle_coupling: f64, seed: u64) -> Self {
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

    /// Forward pass applying ±π phase shifts across all qubits via fused single-pass AVX2 kernel:
    pub fn forward(&self, state: &QuantumState) -> QuantumState {
        let mut current = state.clone();
        self.forward_in_place(&mut current);
        current
    }

    /// In-place fused Walsh-Rademacher phase transformation (zero allocations, single AVX2 linear pass):
    pub fn forward_in_place(&self, state: &mut QuantumState) {
        crate::simd::apply_rademacher_phase_kernel(&mut state.real, &mut state.imag, &self.signs);
    }
}

/// Type alias for bare-metal Walsh-Rademacher phase operations.
pub type RademacherPhaseLayer = PiPhaseLayer;

/// Pre-initialization functional applied to Hilbert space prior to the variational ansatz layers.
/// Modeled after Weed / Qrack pre-quantum functionals (BELL_GHZ, ALT_BELL_GHZ, QFT, IQFT).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum PreInitCircuit {
    /// Zero pre-initialization; state starts in |0...0> or caller-provided input.
    #[default]
    None,
    /// Bipolar ±π Walsh-Rademacher phase scramble.
    RademacherPi,
    /// Canonical Greenberger-Horne-Zeilinger maximally entangled superposition:
    /// H(0) followed by CNOT chain (0->1, 1->2, ..., n-2->n-1).
    /// Prepares 1/√2 (|00...0> + |11...1>).
    BellGhz,
    /// Alternate Greenberger-Horne-Zeilinger superposition with alternating parity:
    /// H(0) followed by AntiCNOT chain (0-controlled NOT).
    /// Prepares 1/√2 (|0101...> + |1010...>).
    AltBellGhz,
    /// Full register Quantum Fourier Transform (QFT).
    Qft,
    /// Full register Inverse Quantum Fourier Transform (IQFT).
    Iqft,
}

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

    /// In-place pre-initialization functional applied to statevector prior to variational layers:
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
            PreInitCircuit::BellGhz => {
                apply_bell_ghz(state, self.n_qubits);
            }
            PreInitCircuit::AltBellGhz => {
                apply_alt_bell_ghz(state, self.n_qubits);
            }
            PreInitCircuit::Qft => {
                apply_qft(state, self.n_qubits);
            }
            PreInitCircuit::Iqft => {
                apply_iqft(state, self.n_qubits);
            }
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

            // Periodic L2 norm verification to verify numerical stability under deep passes
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

    /// In-place evolution with terminal layer diagonal commutation pruning:
    /// In the final layer (depth - 1), executes ONLY single-qubit Ry rotations.
    /// Skips all terminal Rz and Ring-CZ gates because [Rz, Z_j] = 0 and [CZ, Z_j] = 0,
    /// yielding exact mathematical invariance for Pauli-Z expectation measurements.
    /// Expected weights length: 2 * n_qubits * (depth - 1) + n_qubits (e.g. 24 + 24 + 12 = 60).
    pub fn forward_pruned_into(&self, state: &QuantumState, pruned_weights: &[f64], out: &mut QuantumState) {
        assert_eq!(self.dim, state.dim);
        assert_eq!(self.dim, out.dim);
        assert!(self.depth >= 1, "Ansatz depth must be at least 1");
        let expected_len = 2 * self.n_qubits * (self.depth - 1) + self.n_qubits;
        assert_eq!(pruned_weights.len(), expected_len, "Pruned weights length mismatch: expected {}, got {}", expected_len, pruned_weights.len());

        out.copy_from(state);

        self.apply_pre_init(out);

        // Full layers 0..(depth - 1)
        for layer in 0..(self.depth - 1) {
            self.forward_layer_into(layer, pruned_weights, out);
        }

        // Terminal Layer (depth - 1): apply ONLY Ry gates, omitting commuting Rz and Ring-CZ
        let p_start = (self.depth - 1) * 2 * self.n_qubits;
        for q in 0..self.n_qubits {
            let theta_y = pruned_weights[p_start + q];
            apply_single_qubit_ry(out, q, theta_y);
        }

        out.normalize();
    }

    /// In-place evolution stopping at the end of Layer (depth - 2).
    /// Used for Observable Rotation Fusion where terminal Ry gates are evaluated analytically
    /// during POVM / Pauli measurement without mutating the 64 KB statevector in Layer 3.
    pub fn forward_l2_into(&self, state: &QuantumState, pruned_weights: &[f64], out: &mut QuantumState) {
        assert_eq!(self.dim, state.dim);
        assert_eq!(self.dim, out.dim);
        self.assert_full_layer_weights(pruned_weights);

        out.copy_from(state);

        self.apply_pre_init(out);

        // Full layers 0..(depth - 1)
        for layer in 0..(self.depth - 1) {
            self.forward_layer_into(layer, pruned_weights, out);
        }

        out.normalize();
    }

    /// Full end-to-end forward evaluation with Next-Stage Observable Rotation Fusion:
    /// 1. Evolves statevector up to Layer 2 (Gate 72) with zero mutation in Layer 3.
    /// 2. Performs read-only AVX2 reduction to measure Pauli-Z and Pauli-X expectations.
    /// 3. Applies terminal rotation identity analytically:
    ///    <Z_q>_final = cos(θ_q) <Z_q>_L2 - sin(θ_q) <X_q>_L2.
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

    /// Full end-to-end forward evaluation with Next-Stage Observable Rotation Fusion
    /// extracting BOTH single-qubit ⟨Z_q⟩ AND 2-body ⟨Z_i Z_j⟩ correlators:
    /// 1. Evolves statevector up to Layer 2 (Gate 72) with zero mutation in Layer 3.
    /// 2. Performs read-only AVX2 reduction to measure all Pauli-Z, Pauli-X, and
    ///    two-body cross-observable correlators.
    /// 3. Applies terminal rotation identity analytically for both 1-body and 2-body terms.
    ///
    /// Output: z_final[0..n] = single-qubit expectations,
    ///         zz_final[0..n*(n-1)/2] = two-body Ising correlators.
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
            &out_state.real, &out_state.imag, theta_y, z_final, zz_final,
        );
    }

    /// In-place MECHANISM #2: Mid-Circuit Feed-Forward Evaluation.
    /// Evolves the input statevector in-place through Layers 0 & 1, reads ⟨Z_source_qubit⟩,
    /// modulates terminal Ry angles θ̃_{y, q} = θ_{y, q} + α_q · ⟨Z_source_qubit⟩,
    /// and performs fused terminal Ry + Pauli-Z observable reduction.
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
        let dynamic_thetas = self.feedforward_thetas(pruned_weights, feedforward_weights, m_src, n_eval);

        crate::simd::measure_fused_ry_z_readout(
            &state.real, &state.imag, &dynamic_thetas[..n_eval], z_final,
        );

        m_src
    }

    /// In-place COMBINED MECHANISM #1 + #2: Mid-Circuit Feed-Forward + Full 2-Body Readout.
    /// Evolves in-place, reads ⟨Z_source_qubit⟩, modulates terminal angles, and extracts
    /// both single-qubit ⟨Z_q⟩ and all 66 two-body correlators ⟨Z_i Z_j⟩.
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
        let dynamic_thetas = self.feedforward_thetas(pruned_weights, feedforward_weights, m_src, n_eval);

        crate::simd::measure_fused_ry_full_readout(
            &state.real, &state.imag, &dynamic_thetas[..n_eval], z_final, zz_final,
        );

        m_src
    }

    /// MECHANISM #2: Mid-Circuit Feed-Forward Forward Evaluation (L1 -> L2 Terminal Modulation).
    ///
    /// 1. Evolves the statevector through Layer 0 and Layer 1.
    /// 2. Reads the mid-circuit expectation ⟨Z_source_qubit⟩ (e.g. Qubit 8) without state collapse.
    /// 3. Dynamically modulates the terminal Layer 2 Ry angles:
    ///      θ̃_{y, q} = θ_{y, q} + feedforward_weights[q] · ⟨Z_source_qubit⟩
    ///    Because [Ry, Ry] = 0, this is the exact analytical equivalent of applying
    ///    mid-circuit conditional rotations Ry(α_q · ⟨Z_src⟩) before terminal Ry(θ_{y,q}).
    /// 4. Evaluates Pauli-Z expectations via fused AVX2 reduction.
    ///
    /// This introduces degree-4+ polynomial non-linearities into the expectation values,
    /// directly breaking the degree-2 quadratic limitation of single-upload unitary circuits.
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
        let m_src = crate::simd::measure_single_qubit_z(&out_state.real, &out_state.imag, source_qubit);

        let n_eval = z_final.len();
        let dynamic_thetas = self.feedforward_thetas(pruned_weights, feedforward_weights, m_src, n_eval);

        crate::simd::measure_fused_ry_z_readout(
            &out_state.real, &out_state.imag, &dynamic_thetas[..n_eval], z_final,
        );

        m_src
    }

    /// Combined MECHANISM #1 + #2: Mid-Circuit Feed-Forward with Full 2-Body Readout.
    /// Evaluates both single-qubit ⟨Z_q⟩ and all 66 two-body correlators ⟨Z_i Z_j⟩
    /// under dynamic feed-forward modulated terminal rotation angles.
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
        let m_src = crate::simd::measure_single_qubit_z(&out_state.real, &out_state.imag, source_qubit);

        let n_eval = z_final.len();
        let dynamic_thetas = self.feedforward_thetas(pruned_weights, feedforward_weights, m_src, n_eval);

        crate::simd::measure_fused_ry_full_readout(
            &out_state.real, &out_state.imag, &dynamic_thetas[..n_eval], z_final, zz_final,
        );

        m_src
    }

    /// Entanglement-Propagated Mid-Circuit Feed-Forward (L0 -> L1 In-Place Steering):
    /// 1. Evolves through Layer 0.
    /// 2. Measures ⟨Z_source_qubit⟩.
    /// 3. Applies Ry(α_q · ⟨Z_src⟩) directly to statevector before Layer 1.
    /// 4. Evolves through Layer 1 CZ ring entanglement, propagating the correction non-locally.
    /// 5. Evaluates terminal readout with fused Ry.
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

        // Layer 0
        self.forward_layer_into(0, pruned_weights, out_state);

        // Mid-circuit measurement at end of Layer 0
        let m_src = crate::simd::measure_single_qubit_z(&out_state.real, &out_state.imag, source_qubit);

        // Apply feed-forward rotations Ry(alpha_q * m_src) before Layer 1
        for q in 0..self.n_qubits.min(feedforward_weights.len()) {
            let alpha = feedforward_weights[q];
            if alpha.abs() > 1e-12 {
                apply_single_qubit_ry(out_state, q, alpha * m_src);
            }
        }

        // Remaining full layers 1..depth-1
        for layer in 1..(self.depth - 1) {
            self.forward_layer_into(layer, pruned_weights, out_state);
        }
        out_state.normalize();

        // Terminal layer fused readout
        let p_start = (self.depth - 1) * 2 * self.n_qubits;
        let n_eval = z_final.len();
        let theta_y = &pruned_weights[p_start..p_start + n_eval];
        crate::simd::measure_fused_ry_z_readout(
            &out_state.real, &out_state.imag, theta_y, z_final,
        );

        m_src
    }

    /// MECHANISM #3: Data Re-Uploading In-Place Forward Evaluation.
    ///
    /// Implements universal multi-frequency QNN representation (Schuld et al. 2021).
    /// Re-injects classical descriptors x between Layer 0 and Layer 1 via feature-scaled rotations:
    ///   R_y(s_q · x̃_q)
    /// prior to Layer 1's CZ ring entanglement block.
    ///
    /// Incorporates Mechanism #2 (Mid-Circuit Feed-Forward from Qubit 8)
    /// and terminal Observable Rotation Fusion.
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

        // Layer 0 evolution
        self.forward_layer_into(0, pruned_weights, state);

        // DATA RE-UPLOADING STAGE 1: Interleave normalized input features between Layer 0 and Layer 1
        let norm_sq: f64 = features.iter().map(|v| v * v).sum();
        let inv_norm = 1.0 / norm_sq.sqrt().max(1e-12);
        let n_feats = features.len();

        for q in 0..self.n_qubits {
            let feat_val = if q < n_feats { features[q] * inv_norm } else { 0.0 };
            let scale = if q < upload_scales_l0.len() { upload_scales_l0[q] } else { 0.0 };
            if scale.abs() > 1e-12 {
                apply_single_qubit_ry(state, q, scale * feat_val);
            }
        }

        // Remaining full layers 1..depth-1 (entangle re-uploaded features through the CZ blocks)
        for layer in 1..(self.depth - 1) {
            self.forward_layer_into(layer, pruned_weights, state);
        }

        // Mid-circuit measurement after the last full layer (Mechanism #2)
        let m_src = crate::simd::measure_single_qubit_z(&state.real, &state.imag, source_qubit);

        // Dynamic terminal rotation angles
        let n_eval = z_final.len();
        let dynamic_thetas = self.feedforward_thetas(pruned_weights, feedforward_alphas, m_src, n_eval);

        // Terminal fused readout
        crate::simd::measure_fused_ry_z_readout(
            &state.real, &state.imag, &dynamic_thetas[..n_eval], z_final,
        );

        m_src
    }

    /// Combined MECHANISMS #1 + #2 + #3: Full Re-Uploading + Feed-Forward + 78 ZZ Readout.
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

        // Layer 0 evolution
        self.forward_layer_into(0, pruned_weights, state);

        // DATA RE-UPLOADING STAGE 1: Interleave normalized input features between Layer 0 and Layer 1
        let norm_sq: f64 = features.iter().map(|v| v * v).sum();
        let inv_norm = 1.0 / norm_sq.sqrt().max(1e-12);
        let n_feats = features.len();

        for q in 0..self.n_qubits {
            let feat_val = if q < n_feats { features[q] * inv_norm } else { 0.0 };
            let scale = if q < upload_scales_l0.len() { upload_scales_l0[q] } else { 0.0 };
            if scale.abs() > 1e-12 {
                apply_single_qubit_ry(state, q, scale * feat_val);
            }
        }

        // Remaining full layers 1..depth-1
        for layer in 1..(self.depth - 1) {
            self.forward_layer_into(layer, pruned_weights, state);
        }

        // Mid-circuit measurement after the last full layer (Mechanism #2)
        let m_src = crate::simd::measure_single_qubit_z(&state.real, &state.imag, source_qubit);

        // Dynamic terminal rotation angles
        let n_eval = z_final.len();
        let dynamic_thetas = self.feedforward_thetas(pruned_weights, feedforward_alphas, m_src, n_eval);

        // Terminal fused readout (12 single-qubit + 66 two-body correlators)
        crate::simd::measure_fused_ry_full_readout(
            &state.real, &state.imag, &dynamic_thetas[..n_eval], z_final, zz_final,
        );

        m_src
    }

    /// Requires the full-layer prefix of a pruned weight vector: `2 * n_qubits * (depth - 1)`.
    fn assert_full_layer_weights(&self, weights: &[f64]) {
        assert!(self.depth >= 1, "Ansatz depth must be at least 1");
        let needed = 2 * self.n_qubits * (self.depth - 1);
        assert!(
            weights.len() >= needed,
            "Pruned weights length mismatch: need at least {} full-layer weights, got {}",
            needed,
            weights.len()
        );
    }

    /// Requires a full pruned weight vector: full layers plus one terminal Ry per qubit.
    fn assert_pruned_weights(&self, weights: &[f64]) {
        assert!(self.depth >= 1, "Ansatz depth must be at least 1");
        let needed = 2 * self.n_qubits * (self.depth - 1) + self.n_qubits;
        assert!(
            weights.len() >= needed,
            "Pruned weights length mismatch: expected at least {}, got {}",
            needed,
            weights.len()
        );
    }

    fn assert_readout_width(&self, n_eval: usize) {
        assert!(
            n_eval <= self.n_qubits,
            "readout of {} qubits requested from a {}-qubit ansatz",
            n_eval,
            self.n_qubits
        );
        crate::simd::assert_fused_readout_width(n_eval);
    }

    /// Mid-circuit feed-forward and re-uploading paths interleave work after layer 0,
    /// so they need at least one full layer before the terminal Ry layer.
    fn assert_multi_layer_depth(&self) {
        assert!(
            self.depth >= 2,
            "this forward path needs depth >= 2 (got {})",
            self.depth
        );
    }

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

    /// In-place evolution for a single circuit layer: Ry on every qubit, then the layer's Rz
    /// rotations and CZ entangling block.
    ///
    /// Ry gates on different qubits commute with the Rz gates of other qubits, so applying all Ry
    /// first preserves each qubit's Ry-then-Rz order. Every Rz and CZ is diagonal, so they are
    /// applied together as one phase pass (`diagonal::apply_rz_cz_layer`) instead of one pass per
    /// gate.
    pub fn forward_layer_into(&self, layer_idx: usize, weights: &[f64], out: &mut QuantumState) {
        assert_eq!(self.dim, out.dim);
        let n = self.n_qubits;
        let p_start = layer_idx * 2 * n;
        assert!(
            weights.len() >= p_start + 2 * n,
            "layer {} needs weights up to index {}, got {}",
            layer_idx,
            p_start + 2 * n,
            weights.len()
        );

        if n <= 16 {
            // Qubit Threshold Guard: Sequential AVX2 path for <=16 qubits (reclaims threadpool dispatch overhead)
            for q in 0..n {
                crate::simd::apply_ry_kernel(&mut out.real, &mut out.imag, q, weights[p_start + 2 * q]);
            }
        } else {
            // Multithreaded AVX2 + Tiled Butterfly path for >=17 qubits
            for q in 0..n {
                crate::parallel::apply_ry(&mut out.real, &mut out.imag, q, weights[p_start + 2 * q]);
            }
        }

        let theta_rz: Vec<f64> = (0..n).map(|q| weights[p_start + 2 * q + 1]).collect();
        let pairs: Vec<(usize, usize)> = entangling_pairs(n, layer_idx).collect();
        crate::diagonal::apply_rz_cz_layer(&mut out.real, &mut out.imag, &theta_rz, &pairs);
    }

    /// Forward pass through the parameterized ansatz W(θ):
    /// |psi_out> = W(θ) |psi_in>
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
        for _layer in 0..self.depth {
            // Rotation block
            for q in 0..self.n_qubits {
                let theta_y = weights[p_idx];
                let theta_z = weights[p_idx + 1];
                p_idx += 2;
                builder.ry(q, theta_y);
                builder.rz(q, theta_z);
            }
            // Entanglement block with logarithmic hypercube stride
            for (q, target) in entangling_pairs(self.n_qubits, _layer) {
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

/// CZ pairs of the entangling block for `layer_idx`: `(q, (q + 2^(layer_idx mod 4)) mod n)`.
///
/// Each unordered pair is yielded exactly once. When `2 * stride == n` the naive ring visits
/// every pair twice (e.g. `(0, 8)` and `(8, 0)` for 16 qubits) and the two CZs cancel, so only
/// the first `stride` qubits emit a pair. A stride that wraps to 0 (n divides it) emits nothing.
pub(crate) fn entangling_pairs(n_qubits: usize, layer_idx: usize) -> impl Iterator<Item = (usize, usize)> {
    let stride = (1usize << (layer_idx % 4)) % n_qubits.max(1);
    let count = match stride {
        0 => 0,
        s if 2 * s == n_qubits => s,
        _ => n_qubits,
    };
    (0..count).map(move |q| (q, (q + stride) % n_qubits))
}

#[inline(always)]
fn apply_single_qubit_ry(state: &mut QuantumState, qubit: usize, theta: f64) {
    crate::simd::apply_ry_kernel(&mut state.real, &mut state.imag, qubit, theta);
}

/// Prepares canonical Greenberger-Horne-Zeilinger state 1/√2 (|00...0> + |11...1>) on statevector.
pub fn apply_bell_ghz(state: &mut QuantumState, n_qubits: usize) {
    if n_qubits == 0 {
        return;
    }
    crate::compiler::executor::apply_hadamard_op(&mut state.real, &mut state.imag, 0);
    for i in 1..n_qubits {
        crate::compiler::executor::apply_cnot_op(&mut state.real, &mut state.imag, i - 1, i);
    }
}

/// Prepares alternate Greenberger-Horne-Zeilinger state 1/√2 (|0101...> + |1010...>) on statevector.
pub fn apply_alt_bell_ghz(state: &mut QuantumState, n_qubits: usize) {
    if n_qubits == 0 {
        return;
    }
    crate::compiler::executor::apply_hadamard_op(&mut state.real, &mut state.imag, 0);
    for i in 1..n_qubits {
        crate::compiler::executor::apply_anticnot_op(&mut state.real, &mut state.imag, i - 1, i);
    }
}

/// Performs full register Quantum Fourier Transform (QFT) in-place on statevector.
pub fn apply_qft(state: &mut QuantumState, n_qubits: usize) {
    if n_qubits == 0 {
        return;
    }
    for j in 0..n_qubits {
        crate::compiler::executor::apply_hadamard_op(&mut state.real, &mut state.imag, j);
        for k in (j + 1)..n_qubits {
            let theta = std::f64::consts::PI / (1usize << (k - j)) as f64;
            crate::compiler::executor::apply_controlled_phase_op(
                &mut state.real,
                &mut state.imag,
                k,
                j,
                theta,
            );
        }
    }
    for j in 0..(n_qubits / 2) {
        crate::compiler::executor::apply_swap_op(
            &mut state.real,
            &mut state.imag,
            j,
            n_qubits - 1 - j,
        );
    }
}

/// Performs full register Inverse Quantum Fourier Transform (IQFT) in-place on statevector.
pub fn apply_iqft(state: &mut QuantumState, n_qubits: usize) {
    if n_qubits == 0 {
        return;
    }
    for j in 0..(n_qubits / 2) {
        crate::compiler::executor::apply_swap_op(
            &mut state.real,
            &mut state.imag,
            j,
            n_qubits - 1 - j,
        );
    }
    for j in (0..n_qubits).rev() {
        for k in ((j + 1)..n_qubits).rev() {
            let theta = -std::f64::consts::PI / (1usize << (k - j)) as f64;
            crate::compiler::executor::apply_controlled_phase_op(
                &mut state.real,
                &mut state.imag,
                k,
                j,
                theta,
            );
        }
        crate::compiler::executor::apply_hadamard_op(&mut state.real, &mut state.imag, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bell_ghz_preparation() {
        let n_qubits = 4;
        let mut state = QuantumState::ground(1 << n_qubits);
        apply_bell_ghz(&mut state, n_qubits);

        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let last_idx = (1 << n_qubits) - 1;

        assert!((state.real[0] - inv_sqrt2).abs() < 1e-12);
        assert!((state.real[last_idx] - inv_sqrt2).abs() < 1e-12);

        for i in 1..last_idx {
            assert!(state.real[i].abs() < 1e-12);
            assert!(state.imag[i].abs() < 1e-12);
        }
    }

    #[test]
    fn test_alt_bell_ghz_preparation() {
        let n_qubits = 4;
        let mut state = QuantumState::ground(1 << n_qubits);
        apply_alt_bell_ghz(&mut state, n_qubits);

        // Branch 0 starts at 0 -> 1 -> 0 -> 1 -> binary 0101_2 = 5
        // Branch 1 starts at 1 -> 0 -> 1 -> 0 -> binary 1010_2 = 10
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let mut branch0 = 0usize;
        let mut branch1 = 0usize;
        let mut b0 = 0usize;
        let mut b1 = 1usize;
        for q in 0..n_qubits {
            branch0 |= b0 << q;
            branch1 |= b1 << q;
            b0 = 1 - b0;
            b1 = 1 - b1;
        }

        assert!((state.real[branch0] - inv_sqrt2).abs() < 1e-12);
        assert!((state.real[branch1] - inv_sqrt2).abs() < 1e-12);

        for i in 0..(1 << n_qubits) {
            if i != branch0 && i != branch1 {
                assert!(state.real[i].abs() < 1e-12);
                assert!(state.imag[i].abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_qft_iqft_roundtrip_identity() {
        let n_qubits = 6;
        let dim = 1 << n_qubits;
        let mut state = QuantumState::new(dim);

        // Initial arbitrary superposition state
        for i in 0..dim {
            state.real[i] = ((i * 7 + 3) as f64).sin();
            state.imag[i] = ((i * 11 + 5) as f64).cos();
        }
        state.normalize();

        let original = state.clone();

        // Apply QFT then IQFT
        apply_qft(&mut state, n_qubits);
        apply_iqft(&mut state, n_qubits);

        let fidelity = state.fidelity(&original);
        assert!(
            (fidelity - 1.0).abs() < 1e-12,
            "QFT -> IQFT fidelity must be 1.0, got {}",
            fidelity
        );
    }

    #[test]
    fn test_circuit_ansatz_with_pre_init_dag_equivalence() {
        let n_qubits = 5;
        let depth = 2;
        let ansatz = CircuitAnsatz::new(n_qubits, depth).with_bell_ghz();
        let weights = ansatz.weights.clone();

        let input = QuantumState::new(1 << n_qubits);

        // 1. Direct in-place evolution
        let mut state_direct = input.clone();
        ansatz.forward_in_place(&mut state_direct, &weights);

        // 2. DAG compiled evolution
        let compiled = ansatz.compile(&weights);
        let mut state_compiled = input.clone();
        compiled.execute(&mut state_compiled);

        let fidelity = state_direct.fidelity(&state_compiled);
        assert!(
            (fidelity - 1.0).abs() < 1e-12,
            "Direct and DAG compiled evolution must match, got fidelity {}",
            fidelity
        );
    }
}

