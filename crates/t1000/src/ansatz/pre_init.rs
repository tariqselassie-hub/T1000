// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Pre-initialization circuits applied to the statevector before variational layers.
//!
//! Covers `PreInitCircuit` enum and the four gate families it selects from:
//! Walsh-Rademacher, GHZ, and QFT/IQFT.

use crate::state::QuantumState;

/// Pre-initialization functional applied to Hilbert space prior to the variational ansatz layers.
/// Modeled after Weed / Qrack pre-quantum functionals (BELL_GHZ, ALT_BELL_GHZ, QFT, IQFT).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
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

        for i in 0..dim {
            state.real[i] = ((i * 7 + 3) as f64).sin();
            state.imag[i] = ((i * 11 + 5) as f64).cos();
        }
        state.normalize();

        let original = state.clone();
        apply_qft(&mut state, n_qubits);
        apply_iqft(&mut state, n_qubits);

        let fidelity = state.fidelity(&original);
        assert!(
            (fidelity - 1.0).abs() < 1e-12,
            "QFT -> IQFT fidelity must be 1.0, got {fidelity}"
        );
    }
}
