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

#[cfg(test)]
mod tests {
    use super::*;

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
}
