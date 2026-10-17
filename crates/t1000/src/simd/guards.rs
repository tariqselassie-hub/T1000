// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Precondition assertion guards and structural validation for SIMD kernels.

/// Maximum qubit count supported by the fused terminal-readout kernels (fixed stack buffers).
pub const MAX_FUSED_READOUT_QUBITS: usize = 16;

/// Asserts a valid statevector buffer pair and returns its qubit count.
#[inline(always)]
pub fn assert_statevector(real: &[f64], imag: &[f64]) -> usize {
    assert_eq!(
        real.len(),
        imag.len(),
        "real/imag length mismatch ({} vs {})",
        real.len(),
        imag.len()
    );
    assert!(
        real.len() >= 2 && real.len().is_power_of_two(),
        "statevector length must be a power of two >= 2 (got {})",
        real.len()
    );
    real.len().trailing_zeros() as usize
}

/// Asserts that a single qubit index is within valid register bounds.
#[inline(always)]
pub fn assert_qubit(qubit: usize, n_qubits: usize) {
    assert!(
        qubit < n_qubits,
        "qubit index {} out of range for {}-qubit state",
        qubit,
        n_qubits
    );
}

/// Asserts that two qubit indices are distinct and within valid register bounds.
#[inline(always)]
pub fn assert_qubit_pair(q1: usize, q2: usize, n_qubits: usize) {
    assert_qubit(q1, n_qubits);
    assert_qubit(q2, n_qubits);
    assert_ne!(
        q1, q2,
        "two-qubit gate requires distinct qubits (got {} twice)",
        q1
    );
}

/// Asserts that the requested fused readout width is supported by stack scratch space.
#[inline(always)]
pub fn assert_fused_readout_width(n: usize) {
    assert!(
        n <= MAX_FUSED_READOUT_QUBITS,
        "fused readout supports at most {} qubits (got {})",
        MAX_FUSED_READOUT_QUBITS,
        n
    );
}
