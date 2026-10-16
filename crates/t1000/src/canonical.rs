// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Canonical Quantum Circuits, Algorithmic Primitives & Error Mitigation
//!
//! Formalizes the standard, foundational quantum algorithms and protocols in `T-1000`:
//! - **Grover's Search Diffusion Operator**: Amplitude inversion about the mean $2|\psi\rangle\langle\psi| - I$.
//! - **Bernstein-Vazirani Oracle**: Single-query determination of hidden bitstrings $s \in \{0, 1\}^n$.
//! - **Quantum Phase Estimation (QPE)**: High-precision eigenvalue phase extraction via controlled evolutions and IQFT.
//! - **Hardware-Efficient Ansatz (HEA / EfficientSU2)**: Variational rotation and entangler ladders.
//! - **Trotter-Suzuki Spin Hamiltonian**: Transverse-Field Ising and Heisenberg spin-chain time evolution.
//! - **Readout Error Mitigation**: Measurement assignment confusion tensor calibration and inversion.

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use crate::coherence::NoiseChannelParams;
use crate::compiler::op::CircuitOp;
use crate::compiler::CircuitBuilder;
use crate::state::QuantumState;

/// Entangling topology options for variational quantum ansatze.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntanglerTopology {
    /// Linear chain: (0, 1), (1, 2), ..., (n-2, n-1)
    Linear,
    /// Circular periodic ring: linear chain plus wrap-around (n-1, 0)
    Circular,
    /// All-to-all pairs: (i, j) for all i < j
    AllToAll,
}

/// 2x2 Readout confusion matrix per physical qubit:
/// M = [[P(meas 0 | prep 0), P(meas 0 | prep 1)],
///      [P(meas 1 | prep 0), P(meas 1 | prep 1)]]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReadoutConfusionMatrix {
    /// P(measure 0 | prepared 0)
    pub p00: f64,
    /// P(measure 1 | prepared 0)
    pub p10: f64,
    /// P(measure 0 | prepared 1)
    pub p01: f64,
    /// P(measure 1 | prepared 1)
    pub p11: f64,
}

impl Default for ReadoutConfusionMatrix {
    fn default() -> Self {
        Self {
            p00: 0.985,
            p10: 0.015,
            p01: 0.025,
            p11: 0.975,
        }
    }
}

// =========================================================================
// 1. Canonical Algorithmic Circuit Constructors
// =========================================================================

/// Appends Grover's Diffusion Operator (inversion about the mean) to the circuit:
/// $D = 2|\psi\rangle\langle\psi| - I = H^{\otimes n} X^{\otimes n} (CZ_{\text{multi}}) X^{\otimes n} H^{\otimes n}$.
pub fn build_grover_diffusion(builder: &mut CircuitBuilder, qubits: &[usize]) {
    let n = qubits.len();
    if n == 0 {
        return;
    }

    // 1. H on all qubits
    for &q in qubits {
        builder.hadamard(q);
    }
    // 2. X on all qubits
    for &q in qubits {
        builder.dag.add_op(CircuitOp::Unitary1Q {
            qubit: q,
            matrix: crate::compiler::op::Unitary2x2::PAULI_X,
        });
    }

    // 3. Multi-controlled phase flip
    if n == 1 {
        builder.dag.add_op(CircuitOp::Rz { qubit: qubits[0], theta: PI });
    } else if n == 2 {
        builder.cz(qubits[0], qubits[1]);
    } else {
        // Controlled-phase ladder to invert sign of |11...1>
        for i in 0..(n - 1) {
            builder.cz(qubits[i], qubits[i + 1]);
        }
    }

    // 4. X on all qubits
    for &q in qubits {
        builder.dag.add_op(CircuitOp::Unitary1Q {
            qubit: q,
            matrix: crate::compiler::op::Unitary2x2::PAULI_X,
        });
    }
    // 5. H on all qubits
    for &q in qubits {
        builder.hadamard(q);
    }
}

/// Constructs a Bernstein-Vazirani circuit to solve for `hidden_mask` in a single query.
/// Requires `qubits` for input register plus `ancilla` initialized in state $|-\rangle$.
pub fn build_bernstein_vazirani(
    builder: &mut CircuitBuilder,
    qubits: &[usize],
    ancilla: usize,
    hidden_mask: usize,
) {
    // 1. Prepare ancilla in |-> = H X |0>
    builder.dag.add_op(CircuitOp::Unitary1Q {
        qubit: ancilla,
        matrix: crate::compiler::op::Unitary2x2::PAULI_X,
    });
    builder.hadamard(ancilla);

    // 2. Prepare inputs in equal superposition
    for &q in qubits {
        builder.hadamard(q);
    }

    // 3. Oracle phase-kickback: CNOT(q, ancilla) wherever hidden_mask bit is 1
    for (i, &q) in qubits.iter().enumerate() {
        if (hidden_mask & (1 << i)) != 0 {
            builder.cnot(q, ancilla);
        }
    }

    // 4. Measure transform: H on all input qubits
    for &q in qubits {
        builder.hadamard(q);
    }
}

/// Builds Quantum Phase Estimation (QPE) kernel to extract eigenvalue phase $\phi \in [0, 1)$.
/// Precision register contains `precision_qubits`; target qubit undergoes controlled rotations.
pub fn build_quantum_phase_estimation(
    builder: &mut CircuitBuilder,
    precision_qubits: &[usize],
    target_qubit: usize,
    phase_phi: f64,
) {
    let m = precision_qubits.len();
    if m == 0 {
        return;
    }

    // 1. Prepare precision register in equal superposition
    for &q in precision_qubits {
        builder.hadamard(q);
    }

    // 2. Controlled-U^(2^j) rotations: U = Rz(2 * pi * phi)
    for (j, &q_ctrl) in precision_qubits.iter().enumerate() {
        let power = 1usize << j;
        let theta = 2.0 * PI * phase_phi * (power as f64);
        builder.controlled_phase(q_ctrl, target_qubit, theta);
    }

    // 3. Inverse QFT on precision register
    builder.iqft(precision_qubits);
}

/// Appends a Hardware-Efficient Ansatz (HEA) variational layer across `qubits`:
/// Single-qubit parameterized rotations $R_y(\theta_q), R_z(\phi_q)$ followed by entangling ladder.
pub fn build_hea_layer(
    builder: &mut CircuitBuilder,
    qubits: &[usize],
    thetas_y: &[f64],
    thetas_z: &[f64],
    topology: EntanglerTopology,
) {
    let n = qubits.len();
    assert_eq!(thetas_y.len(), n, "thetas_y count must match qubit count");
    assert_eq!(thetas_z.len(), n, "thetas_z count must match qubit count");

    // Single-qubit rotations
    for (i, &q) in qubits.iter().enumerate() {
        builder.ry(q, thetas_y[i]);
        builder.rz(q, thetas_z[i]);
    }

    // Entangling gates
    match topology {
        EntanglerTopology::Linear => {
            for i in 0..(n.saturating_sub(1)) {
                builder.cz(qubits[i], qubits[i + 1]);
            }
        }
        EntanglerTopology::Circular => {
            for i in 0..(n.saturating_sub(1)) {
                builder.cz(qubits[i], qubits[i + 1]);
            }
            if n > 2 {
                builder.cz(qubits[n - 1], qubits[0]);
            }
        }
        EntanglerTopology::AllToAll => {
            for i in 0..n {
                for j in (i + 1)..n {
                    builder.cz(qubits[i], qubits[j]);
                }
            }
        }
    }
}

/// Appends a Trotter-Suzuki time step for Transverse-Field Ising Model:
/// $H = -J \sum_{\langle i, j \rangle} Z_i Z_j - h \sum_i X_i$.
/// $e^{-\mathrm{i} H dt} \approx \prod e^{-\mathrm{i} J Z_i Z_j dt} \prod e^{-\mathrm{i} h X_i dt}$.
pub fn build_trotter_ising_step(
    builder: &mut CircuitBuilder,
    qubit_pairs: &[(usize, usize)],
    qubits: &[usize],
    j_coupling: f64,
    h_transverse: f64,
    dt: f64,
) {
    // 1. Entangling ZZ interactions: CX -> Rz(2 * J * dt) -> CX
    let theta_zz = 2.0 * j_coupling * dt;
    for &(q1, q2) in qubit_pairs {
        builder.cnot(q1, q2);
        builder.rz(q2, theta_zz);
        builder.cnot(q1, q2);
    }

    // 2. Transverse field X rotations: Rx(2 * h * dt) = H -> Rz(2 * h * dt) -> H
    let theta_x = 2.0 * h_transverse * dt;
    for &q in qubits {
        builder.hadamard(q);
        builder.rz(q, theta_x);
        builder.hadamard(q);
    }
}

// =========================================================================
// 2. Readout Error Mitigation (Confusion Matrix Inversion)
// =========================================================================

/// Mitigates measurement readout noise by inverting single-qubit confusion matrices.
/// Enforces physical probability simplex constraints ($\sum p_i = 1, p_i \ge 0$).
pub fn mitigate_readout_errors(
    raw_probabilities: &[f64],
    n_qubits: usize,
    matrix: &ReadoutConfusionMatrix,
) -> Vec<f64> {
    let dim = 1 << n_qubits;
    assert_eq!(raw_probabilities.len(), dim);

    // Compute single-qubit inverse matrix
    // det = p00 * p11 - p01 * p10
    let det = matrix.p00 * matrix.p11 - matrix.p01 * matrix.p10;
    if det.abs() < 1e-6 {
        return raw_probabilities.to_vec();
    }
    let inv_det = 1.0 / det;
    let inv_00 = matrix.p11 * inv_det;
    let inv_01 = -matrix.p01 * inv_det;
    let inv_10 = -matrix.p10 * inv_det;
    let inv_11 = matrix.p00 * inv_det;

    let mut mitigated = raw_probabilities.to_vec();

    // Invert each qubit tensor product line
    for q in 0..n_qubits {
        let stride = 1 << q;
        let step = stride << 1;
        for i in (0..dim).step_by(step) {
            for j in 0..stride {
                let idx0 = i + j;
                let idx1 = idx0 + stride;
                let p0 = mitigated[idx0];
                let p1 = mitigated[idx1];
                mitigated[idx0] = inv_00 * p0 + inv_01 * p1;
                mitigated[idx1] = inv_10 * p0 + inv_11 * p1;
            }
        }
    }

    // Clip non-physical negative probabilities and re-normalize onto simplex
    for p in &mut mitigated {
        if *p < 0.0 {
            *p = 0.0;
        }
    }
    let sum: f64 = mitigated.iter().sum();
    if sum > 1e-15 {
        for p in &mut mitigated {
            *p /= sum;
        }
    }

    mitigated
}

// =========================================================================
// 3. End-to-End Canonical Algorithmic Execution Lifecycle
// =========================================================================

/// Formal catalog of foundational quantum algorithms.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CanonicalAlgorithm {
    /// Bernstein-Vazirani single-query hidden linear bitstring recovery.
    BernsteinVazirani {
        n_inputs: usize,
        hidden_mask: usize,
    },
    /// Grover amplitude diffusion operator on an n-qubit register.
    GroverDiffusion {
        n_qubits: usize,
    },
    /// Quantum Phase Estimation for an eigenvalue phase theta in [0, 2*pi).
    QuantumPhaseEstimation {
        n_counting: usize,
        phase_phi: f64,
    },
    /// Hardware-Efficient Ansatz parameterized layer.
    HardwareEfficientAnsatz {
        n_qubits: usize,
        thetas_y: Vec<f64>,
        thetas_z: Vec<f64>,
        topology: EntanglerTopology,
    },
    /// Transverse-Field 1D Ising model Trotter step.
    TrotterIsing {
        n_qubits: usize,
        j_coupling: f64,
        h_transverse: f64,
        dt: f64,
    },
}

/// Telemetry and distribution report from a canonical algorithm execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanonicalExecutionReport {
    /// Number of physical qubits in the execution register
    pub num_qubits: usize,
    /// Raw Born rule measurement probabilities
    pub probabilities_raw: Vec<f64>,
    /// Probabilities after measurement assignment error mitigation
    pub probabilities_mitigated: Vec<f64>,
    /// Most likely basis state index (argmax over probabilities)
    pub most_likely_state: usize,
    /// Shannon entropy in nats of the measurement distribution
    pub shannon_entropy_nats: f64,
    /// Algorithmic success metric (e.g. probability of target hidden string)
    pub success_probability: f64,
}

/// Fluent orchestrator for constructing, simulating, and evaluating canonical algorithms.
#[derive(Debug, Clone)]
pub struct CanonicalRunner {
    pub algorithm: CanonicalAlgorithm,
    pub noise_params: Option<NoiseChannelParams>,
    pub confusion_matrix: Option<ReadoutConfusionMatrix>,
    pub seed: u64,
}

impl CanonicalRunner {
    pub fn new(algorithm: CanonicalAlgorithm) -> Self {
        Self {
            algorithm,
            noise_params: None,
            confusion_matrix: None,
            seed: crate::DEFAULT_SEED,
        }
    }

    pub fn with_noise(mut self, noise: NoiseChannelParams) -> Self {
        self.noise_params = Some(noise);
        self
    }

    pub fn with_readout_mitigation(mut self, confusion: ReadoutConfusionMatrix) -> Self {
        self.confusion_matrix = Some(confusion);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Builds the circuit DAG for this algorithm.
    pub fn build_circuit(&self) -> CircuitBuilder {
        match &self.algorithm {
            CanonicalAlgorithm::BernsteinVazirani { n_inputs, hidden_mask } => {
                let total_qubits = n_inputs + 1;
                let ancilla = *n_inputs;
                let mut builder = CircuitBuilder::new(total_qubits);
                let input_qubits: Vec<usize> = (0..*n_inputs).collect();
                build_bernstein_vazirani(&mut builder, &input_qubits, ancilla, *hidden_mask);
                builder
            }
            CanonicalAlgorithm::GroverDiffusion { n_qubits } => {
                let mut builder = CircuitBuilder::new(*n_qubits);
                let qubits: Vec<usize> = (0..*n_qubits).collect();
                build_grover_diffusion(&mut builder, &qubits);
                builder
            }
            CanonicalAlgorithm::QuantumPhaseEstimation { n_counting, phase_phi } => {
                let total_qubits = n_counting + 1;
                let state_qubit = *n_counting;
                let mut builder = CircuitBuilder::new(total_qubits);
                let counting_qubits: Vec<usize> = (0..*n_counting).collect();
                build_quantum_phase_estimation(&mut builder, &counting_qubits, state_qubit, *phase_phi);
                builder
            }
            CanonicalAlgorithm::HardwareEfficientAnsatz { n_qubits, thetas_y, thetas_z, topology } => {
                let mut builder = CircuitBuilder::new(*n_qubits);
                let qubits: Vec<usize> = (0..*n_qubits).collect();
                build_hea_layer(&mut builder, &qubits, thetas_y, thetas_z, *topology);
                builder
            }
            CanonicalAlgorithm::TrotterIsing { n_qubits, j_coupling, h_transverse, dt } => {
                let mut builder = CircuitBuilder::new(*n_qubits);
                let qubits: Vec<usize> = (0..*n_qubits).collect();
                let pairs: Vec<(usize, usize)> = (0..n_qubits.saturating_sub(1)).map(|i| (i, i + 1)).collect();
                build_trotter_ising_step(&mut builder, &pairs, &qubits, *j_coupling, *h_transverse, *dt);
                builder
            }
        }
    }

    /// Executes the canonical algorithm through circuit compilation, optional noise, and readout mitigation.
    pub fn execute(&self) -> CanonicalExecutionReport {
        let builder = self.build_circuit();
        let num_qubits = builder.dag.num_qubits;
        let dim = 1 << num_qubits;
        let mut state = QuantumState::ground(dim);

        // For QPE, prepare state qubit in |1> eigenstate of phase gate
        if let CanonicalAlgorithm::QuantumPhaseEstimation { n_counting, .. } = self.algorithm {
            crate::simd::apply_ry_kernel(&mut state.real, &mut state.imag, n_counting, PI);
        }

        // Execute compiled circuit with or without open-system noise
        if let Some(ref noise) = self.noise_params {
            builder.execute_with_noise(&mut state, noise, 0.05, self.seed);
        } else {
            builder.execute(&mut state);
        }

        let probabilities_raw = state.probabilities();

        let probabilities_mitigated = if let Some(ref confusion) = self.confusion_matrix {
            mitigate_readout_errors(&probabilities_raw, num_qubits, confusion)
        } else {
            probabilities_raw.clone()
        };

        // Find most likely state
        let mut most_likely_state = 0;
        let mut max_prob = -1.0;
        for (idx, &p) in probabilities_mitigated.iter().enumerate() {
            if p > max_prob {
                max_prob = p;
                most_likely_state = idx;
            }
        }

        // Compute Shannon entropy in nats
        let mut shannon_entropy_nats = 0.0;
        for &p in &probabilities_mitigated {
            if p > 1e-15 {
                shannon_entropy_nats -= p * p.ln();
            }
        }

        // Determine success probability
        let success_probability = match self.algorithm {
            CanonicalAlgorithm::BernsteinVazirani { n_inputs, hidden_mask } => {
                let mask = (1 << n_inputs) - 1;
                let mut sum = 0.0;
                for (idx, &p) in probabilities_mitigated.iter().enumerate() {
                    if (idx & mask) == hidden_mask {
                        sum += p;
                    }
                }
                sum
            }
            CanonicalAlgorithm::GroverDiffusion { .. } => max_prob,
            CanonicalAlgorithm::QuantumPhaseEstimation { .. } => max_prob,
            CanonicalAlgorithm::HardwareEfficientAnsatz { .. } => max_prob,
            CanonicalAlgorithm::TrotterIsing { .. } => probabilities_mitigated[0],
        };

        CanonicalExecutionReport {
            num_qubits,
            probabilities_raw,
            probabilities_mitigated,
            most_likely_state,
            shannon_entropy_nats,
            success_probability,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::QuantumState;

    #[test]
    fn test_bernstein_vazirani_exact_solution() {
        let n_inputs = 3;
        let total_qubits = n_inputs + 1;
        let ancilla = 3;
        let hidden_mask = 0b101; // Should find mask 5 (bits 0 and 2 set)

        let mut builder = CircuitBuilder::new(total_qubits);
        let input_qubits = [0, 1, 2];
        build_bernstein_vazirani(&mut builder, &input_qubits, ancilla, hidden_mask);

        let compiled = builder.compile();
        let mut state = QuantumState::ground(1 << total_qubits);
        compiled.execute(&mut state);

        // Compute marginal probabilities over input qubits 0, 1, 2
        let mut prob_map = vec![0.0f64; 1 << n_inputs];
        for i in 0..state.dim {
            let input_val = i & 0b111;
            prob_map[input_val] += state.real[i] * state.real[i] + state.imag[i] * state.imag[i];
        }

        // Hidden mask 5 must have probability ~1.0
        assert!(prob_map[hidden_mask] > 0.999);
    }

    #[test]
    fn test_trotter_ising_evolution_preserves_unitarity() {
        let qubits = [0, 1, 2];
        let pairs = [(0, 1), (1, 2)];
        let mut builder = CircuitBuilder::new(3);

        build_trotter_ising_step(&mut builder, &pairs, &qubits, 1.0, 0.5, 0.05);
        let compiled = builder.compile();
        let mut state = QuantumState::ground(8);
        compiled.execute(&mut state);

        let norm_sq: f64 = state.real.iter().zip(&state.imag).map(|(&r, &i)| r * r + i * i).sum();
        assert!((norm_sq - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_readout_error_mitigation_inversion() {
        let n_qubits = 2;
        let raw = vec![0.90, 0.05, 0.03, 0.02];
        let m = ReadoutConfusionMatrix::default();

        let mitigated = mitigate_readout_errors(&raw, n_qubits, &m);
        assert_eq!(mitigated.len(), 4);
        let sum: f64 = mitigated.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Mitigation should sharpen dominant ground state probability
        assert!(mitigated[0] >= raw[0]);
    }

    #[test]
    fn test_canonical_runner_bernstein_vazirani() {
        let runner = CanonicalRunner::new(CanonicalAlgorithm::BernsteinVazirani {
            n_inputs: 3,
            hidden_mask: 0b101, // 5
        });
        let report = runner.execute();
        assert_eq!(report.num_qubits, 4);
        assert!(report.success_probability > 0.99);
    }

    #[test]
    fn test_canonical_runner_with_noise_and_mitigation() {
        let noise = NoiseChannelParams {
            t1_us: 100.0,
            t2_us: 120.0,
            depolarizing_prob: 0.005,
        };
        let confusion = ReadoutConfusionMatrix::default();
        let runner = CanonicalRunner::new(CanonicalAlgorithm::BernsteinVazirani {
            n_inputs: 3,
            hidden_mask: 0b011, // 3
        })
        .with_noise(noise)
        .with_readout_mitigation(confusion);

        let report = runner.execute();
        assert_eq!(report.num_qubits, 4);
        assert!(report.success_probability > 0.85);
        assert_eq!(report.probabilities_mitigated.len(), 16);
    }

    #[test]
    fn test_canonical_runner_grover_diffusion() {
        let runner = CanonicalRunner::new(CanonicalAlgorithm::GroverDiffusion { n_qubits: 3 });
        let report = runner.execute();
        assert_eq!(report.num_qubits, 3);
        assert_eq!(report.probabilities_mitigated.len(), 8);
    }
}
