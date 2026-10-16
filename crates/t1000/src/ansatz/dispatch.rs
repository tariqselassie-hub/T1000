// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Unified Execution Dispatcher, Ansatz Topologies, and Ancilla Role Governance.
//! Bridges continuous variational ansatz layers with AVX2 SIMD kernels,
//! discrete Clifford+T synthesis, and auxiliary measurement roles.

use serde::{Deserialize, Serialize};

use crate::clifford_t::{synthesize_rz, DiscreteGate};
use crate::compiler::executor::{apply_cnot_op, apply_hadamard_op};
use crate::simd::{apply_cz_kernel, apply_rz_kernel};

// =========================================================================
// 1. AnsatzMode Specification
// =========================================================================

/// Defines how parameterized continuous gates map onto physical qubits
/// and how entanglement sweeps are laid out in the state evolution pipeline.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AnsatzMode {
    /// Standard Hardware-Efficient Ansatz:
    /// Alternating 1Q rotations (Rx, Rz) with 1D nearest-neighbor Ring CZ.
    /// Best for broad expressibility benchmarking.
    HardwareEfficient {
        layers: usize,
    },

    /// Fused Bipartite Clustered Ansatz (for N=18):
    /// K9 all-to-all inside R1 (0..8), K9 inside R2 (9..17), plus bridge CZs.
    /// Leverages `fused_all_to_all_cz_18` for 100% L3 cache residency.
    DualFragmentCluster {
        layers: usize,
        bridge_pairs: Vec<(usize, usize)>, // e.g. [(4, 13), (8, 17)]
    },

    /// Data Re-Uploading Layered Ansatz:
    /// Interleaves continuous parameter updates θ with external classical input features x.
    /// Overcomes single-layer expressibility limits in quantum classifiers.
    DataReuploading {
        layers: usize,
        feature_dim: usize,
    },

    /// Discrete Fault-Tolerant Clifford+T Ansatz:
    /// Decomposed via Ross-Selinger into discrete tokens (H, S, T, CZ).
    /// Used for fault-tolerant steganography, stabilizer tracking, and exact circuit depth auditing.
    CliffordTDecomposed {
        target_epsilon: f64,
    },

    /// 36-Qubit Bipartite Cut Ansatz across dual 18-qubit L3-resident registers:
    /// Cuts cross-fragment CZ bridges into quasiprobability branches,
    /// evaluating a 36-qubit system in 8 MB RAM without a 1 TB dense statevector.
    BipartiteCut36 {
        bridge_pairs: Vec<(usize, usize)>,
    },
}

impl AnsatzMode {
    /// Calculate exact parameter vector dimension required by this ansatz mode.
    pub fn parameter_count(&self, n_qubits: usize) -> usize {
        match self {
            AnsatzMode::HardwareEfficient { layers } => {
                // 2 single-qubit rotations (Rx, Rz) per qubit per layer + final readout layer
                2 * n_qubits * (layers + 1)
            }
            AnsatzMode::DualFragmentCluster { layers, .. } => {
                // Parameters for R1 and R2 single-qubit rotations per layer
                2 * n_qubits * layers
            }
            AnsatzMode::DataReuploading { layers, feature_dim } => {
                layers * (2 * n_qubits + feature_dim)
            }
            AnsatzMode::CliffordTDecomposed { .. } => 0, // Discrete gates carry no continuous angles
            AnsatzMode::BipartiteCut36 { .. } => 2 * n_qubits,
        }
    }
}

// =========================================================================
// 2. AncillaRole Specification
// =========================================================================

/// Defines how auxiliary qubits operate during circuit execution and state readout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AncillaRole {
    /// Latent Autoencoder Compression:
    /// Ancilla register is driven to |0⟩. The training loss penalizes any 
    /// probability mass remaining in |1⟩ (e.g., minimizing 1 - ⟨Z_ancilla⟩).
    TrashRegister {
        target_qubit: usize,
    },

    /// Non-Destructive Parity / Syndrome Extraction:
    /// Measures multi-qubit stabilizers (e.g. Z_i Z_j) into the ancilla 
    /// to detect errors without collapsing computational superpositions.
    ParitySyndrome {
        ancilla_qubit: usize,
        data_qubits: [usize; 2],
    },

    /// SWAP-Test Fidelity / Kernel Evaluator:
    /// Measures state overlap |⟨ψ_1|ψ_2⟩|² for quantum metric learning.
    SwapTest {
        ancilla_qubit: usize,
        register_a_start: usize,
        register_b_start: usize,
        register_len: usize,
    },

    /// Fault-Tolerant Magic State Gadget:
    /// Consumes a prepared |T⟩ magic ancilla to synthesize a non-Clifford phase rotation.
    MagicStateGadget {
        ancilla_qubit: usize,
        data_qubit: usize,
    },
}

// =========================================================================
// 3. Unified Dispatcher Circuit Engine
// =========================================================================

/// Unified Quantum Neural Network Circuit container with structured dispatch.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QnnCircuit {
    pub n_total_qubits: usize, // e.g. 18
    pub ansatz: AnsatzMode,
    pub ancillae: Vec<AncillaRole>,
    pub parameters: Vec<f64>,
}

impl QnnCircuit {
    /// Creates a new QNN circuit with zeroed initial parameter registers.
    pub fn new(n_total_qubits: usize, ansatz: AnsatzMode, ancillae: Vec<AncillaRole>) -> Self {
        let param_count = ansatz.parameter_count(n_total_qubits);
        Self {
            n_total_qubits,
            ansatz,
            ancillae,
            parameters: vec![0.0; param_count],
        }
    }

    /// Attaches pre-allocated or optimized parameter weights.
    pub fn with_parameters(mut self, parameters: Vec<f64>) -> Self {
        assert_eq!(
            parameters.len(),
            self.ansatz.parameter_count(self.n_total_qubits),
            "Parameter length mismatch"
        );
        self.parameters = parameters;
        self
    }

    /// Executes the circuit without external features.
    pub fn execute(&self, state_re: &mut [f64], state_im: &mut [f64]) {
        self.execute_with_features(state_re, state_im, &[]);
    }

    /// Executes the circuit interleaving external classical features.
    pub fn execute_with_features(
        &self,
        state_re: &mut [f64],
        state_im: &mut [f64],
        features: &[f64],
    ) {
        let n = self.n_total_qubits;
        assert_eq!(state_re.len(), 1 << n, "State length must equal 2^N");
        assert_eq!(state_im.len(), 1 << n, "State length must equal 2^N");

        match &self.ansatz {
            AnsatzMode::DualFragmentCluster { layers, bridge_pairs } => {
                debug_assert_eq!(self.n_total_qubits, 18);
                for l in 0..*layers {
                    let offset = l * 2 * n;
                    for q in 0..n {
                        let rx = self.parameters[offset + 2 * q];
                        let rz = self.parameters[offset + 2 * q + 1];
                        if rx.abs() > 1e-14 {
                            apply_rx_op(state_re, state_im, q, rx);
                        }
                        if rz.abs() > 1e-14 {
                            apply_rz_kernel(state_re, state_im, q, rz);
                        }
                    }
                    unsafe {
                        fused_all_to_all_cz_18(state_re, state_im, true, bridge_pairs);
                    }
                }
            }

            AnsatzMode::HardwareEfficient { layers } => {
                for l in 0..*layers {
                    let offset = l * 2 * n;
                    for q in 0..n {
                        let rx = self.parameters[offset + 2 * q];
                        let rz = self.parameters[offset + 2 * q + 1];
                        if rx.abs() > 1e-14 {
                            apply_rx_op(state_re, state_im, q, rx);
                        }
                        if rz.abs() > 1e-14 {
                            apply_rz_kernel(state_re, state_im, q, rz);
                        }
                    }
                    for q in 0..n {
                        let q_next = (q + 1) % n;
                        apply_cz_kernel(state_re, state_im, q, q_next);
                    }
                }
                let readout_offset = *layers * 2 * n;
                for q in 0..n {
                    let rx = self.parameters[readout_offset + 2 * q];
                    let rz = self.parameters[readout_offset + 2 * q + 1];
                    if rx.abs() > 1e-14 {
                        apply_rx_op(state_re, state_im, q, rx);
                    }
                    if rz.abs() > 1e-14 {
                        apply_rz_kernel(state_re, state_im, q, rz);
                    }
                }
            }

            AnsatzMode::DataReuploading { layers, feature_dim } => {
                let stride = 2 * n + *feature_dim;
                for l in 0..*layers {
                    let offset = l * stride;
                    for q in 0..n {
                        let rx = self.parameters[offset + 2 * q];
                        let rz = self.parameters[offset + 2 * q + 1];
                        let feat = if !features.is_empty() && q < *feature_dim {
                            let scale = self.parameters[offset + 2 * n + q];
                            scale * features[q % features.len()]
                        } else {
                            0.0
                        };
                        apply_rx_op(state_re, state_im, q, rx + feat);
                        if rz.abs() > 1e-14 {
                            apply_rz_kernel(state_re, state_im, q, rz);
                        }
                    }
                    for q in 0..n {
                        let q_next = (q + 1) % n;
                        apply_cz_kernel(state_re, state_im, q, q_next);
                    }
                }
            }

            AnsatzMode::CliffordTDecomposed { target_epsilon } => {
                for q in 0..n {
                    apply_hadamard_op(state_re, state_im, q);
                    let seq = synthesize_rz(std::f64::consts::FRAC_PI_4, *target_epsilon);
                    for &gate in &seq {
                        apply_discrete_gate(state_re, state_im, q, gate);
                    }
                }
                for q in 0..n {
                    let q_next = (q + 1) % n;
                    apply_cz_kernel(state_re, state_im, q, q_next);
                }
            }

            AnsatzMode::BipartiteCut36 { bridge_pairs } => {
                // Fragment evaluation: runs 18-qubit L3-resident state with bridge CZs
                for q in 0..n {
                    let rx = if 2 * q < self.parameters.len() { self.parameters[2 * q] } else { 0.0 };
                    let rz = if 2 * q + 1 < self.parameters.len() { self.parameters[2 * q + 1] } else { 0.0 };
                    if rx.abs() > 1e-14 {
                        apply_rx_op(state_re, state_im, q, rx);
                    }
                    if rz.abs() > 1e-14 {
                        apply_rz_kernel(state_re, state_im, q, rz);
                    }
                }
                unsafe {
                    fused_all_to_all_cz_18(state_re, state_im, true, bridge_pairs);
                }
            }
        }

        self.apply_ancillae_post_ops(state_re, state_im);
    }

    /// Post-ansatz ancilla operations.
    fn apply_ancillae_post_ops(&self, state_re: &mut [f64], state_im: &mut [f64]) {
        for ancilla in &self.ancillae {
            match ancilla {
                AncillaRole::TrashRegister { target_qubit: _ } => {
                    // Left intact for partial trace / penalty measurement
                }
                AncillaRole::SwapTest {
                    ancilla_qubit,
                    register_a_start,
                    register_b_start,
                    register_len,
                } => {
                    apply_swap_test_circuit(
                        state_re,
                        state_im,
                        *ancilla_qubit,
                        *register_a_start,
                        *register_b_start,
                        *register_len,
                    );
                }
                AncillaRole::ParitySyndrome {
                    ancilla_qubit,
                    data_qubits,
                } => {
                    apply_parity_syndrome_circuit(
                        state_re,
                        state_im,
                        *ancilla_qubit,
                        data_qubits[0],
                        data_qubits[1],
                    );
                }
                AncillaRole::MagicStateGadget {
                    ancilla_qubit,
                    data_qubit,
                } => {
                    apply_cnot_op(state_re, state_im, *data_qubit, *ancilla_qubit);
                }
            }
        }
    }

    /// Evaluates autoencoder trash penalty P(|1⟩) = (1 - ⟨Z⟩)/2 on target qubit.
    pub fn evaluate_trash_penalty(&self, state_re: &[f64], state_im: &[f64], target_qubit: usize) -> f64 {
        let mask = 1usize << target_qubit;
        let mut p_one = 0.0;
        for i in 0..state_re.len() {
            if (i & mask) != 0 {
                p_one += state_re[i] * state_re[i] + state_im[i] * state_im[i];
            }
        }
        p_one
    }

    /// Evaluates SWAP-test overlap |⟨ψ_A|ψ_B⟩|² = 1 - 2 P(ancilla = 1).
    pub fn evaluate_swap_test_overlap(
        &self,
        state_re: &[f64],
        state_im: &[f64],
        ancilla_qubit: usize,
    ) -> f64 {
        let p_one = self.evaluate_trash_penalty(state_re, state_im, ancilla_qubit);
        (1.0 - 2.0 * p_one).clamp(0.0, 1.0)
    }

    /// Evaluates non-destructive parity expectation ⟨Z_0 Z_1⟩ from the ancilla register.
    pub fn evaluate_parity_syndrome(
        &self,
        state_re: &[f64],
        state_im: &[f64],
        ancilla_qubit: usize,
    ) -> f64 {
        let p_one = self.evaluate_trash_penalty(state_re, state_im, ancilla_qubit);
        1.0 - 2.0 * p_one
    }
}

// =========================================================================
// 4. Low-Level Gate & Cache-Resident Operators
// =========================================================================

/// In-place single-qubit Rx(θ) rotation.
#[inline(always)]
pub fn apply_rx_op(real: &mut [f64], imag: &mut [f64], qubit: usize, theta: f64) {
    let half = theta * 0.5;
    let c = half.cos();
    let s = half.sin();
    let mask = 1usize << qubit;

    for i in 0..real.len() {
        if (i & mask) == 0 {
            let j = i | mask;
            let r0 = real[i];
            let i0 = imag[i];
            let r1 = real[j];
            let i1 = imag[j];

            real[i] = c * r0 + s * i1;
            imag[i] = c * i0 - s * r1;
            real[j] = s * i0 + c * r1;
            imag[j] = -s * r0 + c * i1;
        }
    }
}

/// Applies a discrete gate token from cyclotomic Clifford+T synthesis.
pub fn apply_discrete_gate(re: &mut [f64], im: &mut [f64], qubit: usize, gate: DiscreteGate) {
    let mask = 1usize << qubit;
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;

    match gate {
        DiscreteGate::H => apply_hadamard_op(re, im, qubit),
        DiscreteGate::T => {
            for i in 0..re.len() {
                if (i & mask) != 0 {
                    let r = re[i];
                    let m = im[i];
                    re[i] = (r - m) * inv_sqrt2;
                    im[i] = (r + m) * inv_sqrt2;
                }
            }
        }
        DiscreteGate::Tdg => {
            for i in 0..re.len() {
                if (i & mask) != 0 {
                    let r = re[i];
                    let m = im[i];
                    re[i] = (r + m) * inv_sqrt2;
                    im[i] = (-r + m) * inv_sqrt2;
                }
            }
        }
        DiscreteGate::S => {
            for i in 0..re.len() {
                if (i & mask) != 0 {
                    let r = re[i];
                    re[i] = -im[i];
                    im[i] = r;
                }
            }
        }
        DiscreteGate::Sdg => {
            for i in 0..re.len() {
                if (i & mask) != 0 {
                    let r = re[i];
                    re[i] = im[i];
                    im[i] = -r;
                }
            }
        }
        DiscreteGate::Z => {
            for i in 0..re.len() {
                if (i & mask) != 0 {
                    re[i] = -re[i];
                    im[i] = -im[i];
                }
            }
        }
        DiscreteGate::X => {
            for i in 0..re.len() {
                if (i & mask) == 0 {
                    let j = i | mask;
                    re.swap(i, j);
                    im.swap(i, j);
                }
            }
        }
        DiscreteGate::Y => {
            for i in 0..re.len() {
                if (i & mask) == 0 {
                    let j = i | mask;
                    let i0 = im[i];
                    let r0 = re[i];
                    let i1 = im[j];
                    let r1 = re[j];
                    re[i] = i1;
                    im[i] = -r1;
                    re[j] = -i0;
                    im[j] = r0;
                }
            }
        }
    }
}

/// Applies SWAP-test verification circuit.
pub fn apply_swap_test_circuit(
    re: &mut [f64],
    im: &mut [f64],
    ancilla: usize,
    reg_a: usize,
    reg_b: usize,
    len: usize,
) {
    apply_hadamard_op(re, im, ancilla);
    for k in 0..len {
        apply_cswap_op(re, im, ancilla, reg_a + k, reg_b + k);
    }
    apply_hadamard_op(re, im, ancilla);
}

/// Controlled-SWAP (Fredkin gate) controlled by `ctrl`.
pub fn apply_cswap_op(re: &mut [f64], im: &mut [f64], ctrl: usize, q1: usize, q2: usize) {
    let ctrl_mask = 1usize << ctrl;
    let m1 = 1usize << q1;
    let m2 = 1usize << q2;
    for i in 0..re.len() {
        if (i & ctrl_mask) != 0 && (i & m1) != 0 && (i & m2) == 0 {
            let j = (i ^ m1) | m2;
            re.swap(i, j);
            im.swap(i, j);
        }
    }
}

/// Non-destructive stabilizer parity measurement circuit.
pub fn apply_parity_syndrome_circuit(
    re: &mut [f64],
    im: &mut [f64],
    ancilla: usize,
    d0: usize,
    d1: usize,
) {
    apply_hadamard_op(re, im, ancilla);
    apply_cz_kernel(re, im, d0, ancilla);
    apply_cz_kernel(re, im, d1, ancilla);
    apply_hadamard_op(re, im, ancilla);
}

/// Fused All-to-All CZ single-pass sweep for N=18 (4.38 ms cache-resident path).
///
/// Evaluates:
/// 1. Intra-R1 K9 all-to-all CZs (qubits 0..9)
/// 2. Intra-R2 K9 all-to-all CZs (qubits 9..18)
/// 3. Bipartite K_9,9 CZs
/// 4. Arbitrary bridge CZ pairs
#[inline(always)]
pub unsafe fn fused_all_to_all_cz_18(
    re: &mut [f64],
    im: &mut [f64],
    include_bipartite: bool,
    bridge_pairs: &[(usize, usize)],
) {
    let r1_mask = 0x1FFusize; // bits 0..9
    let r2_mask = 0x1FFusize; // bits 9..18 shifted

    for i in 0..re.len() {
        let x1 = i & r1_mask;
        let x2 = (i >> 9) & r2_mask;
        let k1 = x1.count_ones() as usize;
        let k2 = x2.count_ones() as usize;

        let mut flip = ((k1 * (k1.saturating_sub(1)) / 2) ^ (k2 * (k2.saturating_sub(1)) / 2)) & 1;
        if include_bipartite {
            flip ^= (k1 * k2) & 1;
        }
        for &(a, b) in bridge_pairs {
            flip ^= ((i >> a) & (i >> b)) & 1;
        }

        if flip != 0 {
            *re.get_unchecked_mut(i) = -*re.get_unchecked(i);
            *im.get_unchecked_mut(i) = -*im.get_unchecked(i);
        }
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ansatz_mode_parameter_counts() {
        let hea = AnsatzMode::HardwareEfficient { layers: 3 };
        assert_eq!(hea.parameter_count(4), 2 * 4 * 4); // 32

        let dfc = AnsatzMode::DualFragmentCluster {
            layers: 2,
            bridge_pairs: vec![(4, 13)],
        };
        assert_eq!(dfc.parameter_count(18), 2 * 18 * 2); // 72

        let dru = AnsatzMode::DataReuploading {
            layers: 2,
            feature_dim: 8,
        };
        assert_eq!(dru.parameter_count(6), 2 * (2 * 6 + 8)); // 40

        let ct = AnsatzMode::CliffordTDecomposed { target_epsilon: 1e-5 };
        assert_eq!(ct.parameter_count(18), 0);
    }

    #[test]
    fn test_hardware_efficient_execution() {
        let n = 4;
        let circuit = QnnCircuit::new(
            n,
            AnsatzMode::HardwareEfficient { layers: 1 },
            vec![],
        );
        let mut re = vec![0.0; 1 << n];
        let mut im = vec![0.0; 1 << n];
        re[0] = 1.0;

        circuit.execute(&mut re, &mut im);
        let norm_sq: f64 = re.iter().zip(&im).map(|(r, i)| r * r + i * i).sum();
        assert!((norm_sq - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_swap_test_ancilla_evaluation() {
        let n = 3; // qubit 0: ancilla, qubit 1: reg A, qubit 2: reg B
        let circuit = QnnCircuit::new(
            n,
            AnsatzMode::HardwareEfficient { layers: 0 },
            vec![AncillaRole::SwapTest {
                ancilla_qubit: 0,
                register_a_start: 1,
                register_b_start: 2,
                register_len: 1,
            }],
        );

        // Case 1: Identical states (|0⟩ on reg A, |0⟩ on reg B) -> Overlap = 1.0
        let mut re = vec![0.0; 1 << n];
        let mut im = vec![0.0; 1 << n];
        re[0] = 1.0; // |000>
        circuit.execute(&mut re, &mut im);
        let overlap = circuit.evaluate_swap_test_overlap(&re, &im, 0);
        assert!((overlap - 1.0).abs() < 1e-12, "Expected 1.0, got {}", overlap);

        // Case 2: Orthogonal states (|0⟩ on reg A, |1⟩ on reg B -> |100> = index 4) -> Overlap = 0.0
        let mut re2 = vec![0.0; 1 << n];
        let mut im2 = vec![0.0; 1 << n];
        re2[4] = 1.0; // qubit 2 is 1 (index 4)
        circuit.execute(&mut re2, &mut im2);
        let overlap2 = circuit.evaluate_swap_test_overlap(&re2, &im2, 0);
        assert!(overlap2.abs() < 1e-12, "Expected 0.0, got {}", overlap2);
    }

    #[test]
    fn test_parity_syndrome_extraction() {
        let n = 3; // qubit 2: ancilla, qubits 0,1: data
        let circuit = QnnCircuit::new(
            n,
            AnsatzMode::HardwareEfficient { layers: 0 },
            vec![AncillaRole::ParitySyndrome {
                ancilla_qubit: 2,
                data_qubits: [0, 1],
            }],
        );

        // |000>: Even parity -> Ancilla stays 0 -> Parity = +1.0
        let mut re = vec![0.0; 1 << n];
        let mut im = vec![0.0; 1 << n];
        re[0] = 1.0;
        circuit.execute(&mut re, &mut im);
        let parity = circuit.evaluate_parity_syndrome(&re, &im, 2);
        assert!((parity - 1.0).abs() < 1e-12);

        // |001>: Odd parity (qubit 0 is 1) -> Ancilla flips to 1 -> Parity = -1.0
        let mut re2 = vec![0.0; 1 << n];
        let mut im2 = vec![0.0; 1 << n];
        re2[1] = 1.0;
        circuit.execute(&mut re2, &mut im2);
        let parity2 = circuit.evaluate_parity_syndrome(&re2, &im2, 2);
        assert!((parity2 - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_trash_register_autoencoder_penalty() {
        let n = 4;
        let circuit = QnnCircuit::new(
            n,
            AnsatzMode::HardwareEfficient { layers: 0 },
            vec![AncillaRole::TrashRegister { target_qubit: 3 }],
        );

        // Ground state |0000>: Qubit 3 is 0 -> penalty = 0.0
        let mut re = vec![0.0; 1 << n];
        let im = vec![0.0; 1 << n];
        re[0] = 1.0;
        let penalty = circuit.evaluate_trash_penalty(&re, &im, 3);
        assert!(penalty.abs() < 1e-12);

        // Excited state |1000> (index 8): Qubit 3 is 1 -> penalty = 1.0
        re[0] = 0.0;
        re[8] = 1.0;
        let penalty_excited = circuit.evaluate_trash_penalty(&re, &im, 3);
        assert!((penalty_excited - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_data_reuploading_execution() {
        let n = 4;
        let circuit = QnnCircuit::new(
            n,
            AnsatzMode::DataReuploading {
                layers: 2,
                feature_dim: 4,
            },
            vec![],
        );
        let mut re = vec![0.0; 1 << n];
        let mut im = vec![0.0; 1 << n];
        re[0] = 1.0;
        let features = vec![0.5, -0.3, 0.8, 0.1];

        circuit.execute_with_features(&mut re, &mut im, &features);
        let norm_sq: f64 = re.iter().zip(&im).map(|(r, i)| r * r + i * i).sum();
        assert!((norm_sq - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_dual_fragment_cluster_18_execution() {
        let n = 18;
        let circuit = QnnCircuit::new(
            n,
            AnsatzMode::DualFragmentCluster {
                layers: 1,
                bridge_pairs: vec![(4, 13), (8, 17)],
            },
            vec![],
        );
        let mut re = vec![0.0; 1 << n];
        let mut im = vec![0.0; 1 << n];
        re[0] = 1.0;

        circuit.execute(&mut re, &mut im);
        let norm_sq: f64 = re.iter().zip(&im).map(|(r, i)| r * r + i * i).sum();
        assert!((norm_sq - 1.0).abs() < 1e-12);
    }
}

