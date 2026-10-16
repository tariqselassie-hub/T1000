// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Quantum Error Correction (QEC) & Stabilizer Syndrome Engine
//!
//! Provides mathematically rigorous stabilizer codes and fault-tolerant primitives:
//! - 3-Qubit Bit-Flip Code $[[3, 1, 1]]$ ($Z_0 Z_1, Z_1 Z_2$ stabilizers).
//! - 3-Qubit Phase-Flip Code $[[3, 1, 1]]$ ($X_0 X_1, X_1 X_2$ stabilizers).
//! - Steane 7-Qubit CSS Code $[[7, 1, 3]]$ (protects against arbitrary single-qubit errors).
//! - Non-destructive stabilizer syndrome measurements $\langle S_i \rangle = \pm 1$.
//! - Canonical look-up table Pauli recovery operators ($X_k, Z_k, Y_k$).
//! - Realistic Pauli channel error injection for benchmarking noise robustness.

use serde::{Deserialize, Serialize};
use crate::state::QuantumState;

/// Supported quantum error correction code families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QecCodeKind {
    /// 3-Qubit Bit-Flip code protecting against single X errors
    BitFlip3Q,
    /// 3-Qubit Phase-Flip code protecting against single Z errors
    PhaseFlip3Q,
    /// Steane 7-Qubit CSS code protecting against arbitrary single-qubit X, Y, Z errors
    Steane7Q,
}

/// Elementary Pauli error kinds for noise emulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PauliErrorKind {
    /// Pauli-X (Bit Flip)
    BitFlip,
    /// Pauli-Z (Phase Flip)
    PhaseFlip,
    /// Pauli-Y (Combined Bit and Phase Flip)
    BitPhaseFlip,
}

/// Stabilizer measurement syndrome outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyndromeReport {
    pub code: QecCodeKind,
    /// Discrete binary eigenvalues for each stabilizer generator (+1 or -1)
    pub syndrome_bits: Vec<i8>,
    /// Whether any error was detected
    pub error_detected: bool,
    /// Target physical qubit identified as faulty (if any)
    pub faulty_qubit: Option<usize>,
    /// Recommended recovery Pauli operation
    pub recommended_recovery: Option<PauliErrorKind>,
}

/// Complete report emitted after error detection and recovery.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QecRecoveryResult {
    pub syndrome: SyndromeReport,
    /// Overlap fidelity of the state with the uncorrupted logical reference
    pub fidelity_restored: f64,
    /// Purity of the recovered statevector
    pub purity: f64,
    pub success: bool,
}

// =========================================================================
// 1. Logical State Encoding
// =========================================================================

/// Encodes an arbitrary single-qubit state $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$
/// into the designated stabilizer code subspace starting at physical qubit `offset`.
pub fn encode_logical_state(
    alpha: (f64, f64),
    beta: (f64, f64),
    code: QecCodeKind,
    total_qubits: usize,
    offset: usize,
) -> QuantumState {
    let dim = 1 << total_qubits;
    let mut state = QuantumState::ground(dim);

    // Normalize input amplitudes
    let norm_sq = alpha.0 * alpha.0 + alpha.1 * alpha.1 + beta.0 * beta.0 + beta.1 * beta.1;
    let inv_norm = if norm_sq > 1e-15 { 1.0 / norm_sq.sqrt() } else { 1.0 };
    let a = (alpha.0 * inv_norm, alpha.1 * inv_norm);
    let b = (beta.0 * inv_norm, beta.1 * inv_norm);

    match code {
        QecCodeKind::BitFlip3Q => {
            assert!(offset + 3 <= total_qubits, "BitFlip3Q requires 3 physical qubits");
            // |0_L> = |000>, |1_L> = |111>
            let idx0 = 0usize;
            let idx1 = (1 << offset) | (1 << (offset + 1)) | (1 << (offset + 2));
            state.real[0] = 0.0;
            state.real[idx0] = a.0;
            state.imag[idx0] = a.1;
            state.real[idx1] = b.0;
            state.imag[idx1] = b.1;
        }
        QecCodeKind::PhaseFlip3Q => {
            assert!(offset + 3 <= total_qubits, "PhaseFlip3Q requires 3 physical qubits");
            // |0_L> = |+++> = 1/2*sqrt(2) (|000> + |001> + ... + |111>)
            // |1_L> = |---> = 1/2*sqrt(2) sum (-1)^wt |x>
            state.real[0] = 0.0;
            let norm = 1.0 / 8.0f64.sqrt();
            for pattern in 0..8usize {
                let p0 = (pattern & 1) << offset;
                let p1 = ((pattern >> 1) & 1) << (offset + 1);
                let p2 = ((pattern >> 2) & 1) << (offset + 2);
                let idx = p0 | p1 | p2;
                let parity = (pattern.count_ones() % 2 == 1) as i32;
                let sign = if parity == 1 { -1.0 } else { 1.0 };

                // state[idx] = norm * (alpha + sign * beta)
                state.real[idx] = norm * (a.0 + sign * b.0);
                state.imag[idx] = norm * (a.1 + sign * b.1);
            }
        }
        QecCodeKind::Steane7Q => {
            assert!(offset + 7 <= total_qubits, "Steane7Q requires 7 physical qubits");
            // Steane [[7,1,3]] code codewords:
            // The 8 codewords of the dual [7,3,4] simplex code C^\perp spanned by rows of H:
            // r1 = bits 3,4,5,6 (120)
            // r2 = bits 1,2,5,6 (102)
            // r3 = bits 0,2,4,6 (85)
            let steane_words = [0usize, 120, 102, 85, 30, 45, 51, 75];
            state.real[0] = 0.0;
            let norm = 1.0 / 8.0f64.sqrt();

            for &w in &steane_words {
                // Shift word into physical register at `offset`
                let idx0 = shift_pattern(w, offset);
                // |1_L> is X_L |0_L> where X_L = 1111111 (all 7 bits inverted)
                let idx1 = shift_pattern(w ^ 0b1111111, offset);

                state.real[idx0] += norm * a.0;
                state.imag[idx0] += norm * a.1;
                state.real[idx1] += norm * b.0;
                state.imag[idx1] += norm * b.1;
            }
        }
    }
    state.normalize();
    state
}

/// Helper to shift a bit pattern to a qubit offset.
#[inline(always)]
fn shift_pattern(pattern: usize, offset: usize) -> usize {
    let mut out = 0usize;
    for bit in 0..7 {
        if (pattern & (1 << bit)) != 0 {
            out |= 1 << (offset + bit);
        }
    }
    out
}

// =========================================================================
// 2. Syndrome Extraction & Non-Destructive Parity Measurement
// =========================================================================

/// Extracts the error syndrome for `code` without destroying the logical information.
pub fn extract_syndrome(state: &QuantumState, code: QecCodeKind, offset: usize) -> SyndromeReport {
    match code {
        QecCodeKind::BitFlip3Q => {
            // S1 = Z_0 Z_1, S2 = Z_1 Z_2
            let s1 = measure_pauli_zz_parity(state, offset, offset + 1);
            let s2 = measure_pauli_zz_parity(state, offset + 1, offset + 2);
            let bits = vec![s1, s2];

            let (error_detected, faulty_qubit, recovery) = match (s1, s2) {
                (1, 1) => (false, None, None),
                (-1, 1) => (true, Some(offset), Some(PauliErrorKind::BitFlip)),
                (-1, -1) => (true, Some(offset + 1), Some(PauliErrorKind::BitFlip)),
                (1, -1) => (true, Some(offset + 2), Some(PauliErrorKind::BitFlip)),
                _ => (false, None, None),
            };

            SyndromeReport {
                code,
                syndrome_bits: bits,
                error_detected,
                faulty_qubit,
                recommended_recovery: recovery,
            }
        }
        QecCodeKind::PhaseFlip3Q => {
            // S1 = X_0 X_1, S2 = X_1 X_2
            let s1 = measure_pauli_xx_parity(state, offset, offset + 1);
            let s2 = measure_pauli_xx_parity(state, offset + 1, offset + 2);
            let bits = vec![s1, s2];

            let (error_detected, faulty_qubit, recovery) = match (s1, s2) {
                (1, 1) => (false, None, None),
                (-1, 1) => (true, Some(offset), Some(PauliErrorKind::PhaseFlip)),
                (-1, -1) => (true, Some(offset + 1), Some(PauliErrorKind::PhaseFlip)),
                (1, -1) => (true, Some(offset + 2), Some(PauliErrorKind::PhaseFlip)),
                _ => (false, None, None),
            };

            SyndromeReport {
                code,
                syndrome_bits: bits,
                error_detected,
                faulty_qubit,
                recommended_recovery: recovery,
            }
        }
        QecCodeKind::Steane7Q => {
            // Steane 6 stabilizer generators:
            // Z-stabilizers detect X errors: S_z1 = Z3 Z4 Z5 Z6, S_z2 = Z1 Z2 Z5 Z6, S_z3 = Z0 Z2 Z4 Z6
            // X-stabilizers detect Z errors: S_x1 = X3 X4 X5 X6, S_x2 = X1 X2 X5 X6, S_x3 = X0 X2 X4 X6
            let sz1 = measure_pauli_multi_z(state, &[offset + 3, offset + 4, offset + 5, offset + 6]);
            let sz2 = measure_pauli_multi_z(state, &[offset + 1, offset + 2, offset + 5, offset + 6]);
            let sz3 = measure_pauli_multi_z(state, &[offset + 0, offset + 2, offset + 4, offset + 6]);

            let sx1 = measure_pauli_multi_x(state, &[offset + 3, offset + 4, offset + 5, offset + 6]);
            let sx2 = measure_pauli_multi_x(state, &[offset + 1, offset + 2, offset + 5, offset + 6]);
            let sx3 = measure_pauli_multi_x(state, &[offset + 0, offset + 2, offset + 4, offset + 6]);

            let bits = vec![sz1, sz2, sz3, sx1, sx2, sx3];

            // Decode Z-syndrome (binary index of bit-flip error)
            let z_idx = decode_syndrome_index(sz1, sz2, sz3);
            // Decode X-syndrome (binary index of phase-flip error)
            let x_idx = decode_syndrome_index(sx1, sx2, sx3);

            let (error_detected, faulty_qubit, recovery) = if z_idx == 0 && x_idx == 0 {
                (false, None, None)
            } else if z_idx > 0 && x_idx == 0 {
                // Pure Bit-Flip X on qubit (z_idx - 1)
                (true, Some(offset + (z_idx - 1)), Some(PauliErrorKind::BitFlip))
            } else if z_idx == 0 && x_idx > 0 {
                // Pure Phase-Flip Z on qubit (x_idx - 1)
                (true, Some(offset + (x_idx - 1)), Some(PauliErrorKind::PhaseFlip))
            } else {
                // Combined Y error (both syndromes fire on same qubit)
                let q = if z_idx == x_idx { z_idx - 1 } else { z_idx - 1 };
                (true, Some(offset + q), Some(PauliErrorKind::BitPhaseFlip))
            };

            SyndromeReport {
                code,
                syndrome_bits: bits,
                error_detected,
                faulty_qubit,
                recommended_recovery: recovery,
            }
        }
    }
}

/// Computes the integer index 0..7 from a 3-bit syndrome (s1, s2, s3) where s_i in {+1, -1}.
#[inline(always)]
fn decode_syndrome_index(s1: i8, s2: i8, s3: i8) -> usize {
    let b0 = if s3 == -1 { 1 } else { 0 };
    let b1 = if s2 == -1 { 1 } else { 0 };
    let b2 = if s1 == -1 { 1 } else { 0 };
    b0 | (b1 << 1) | (b2 << 2)
}

// =========================================================================
// 3. Error Recovery & Correction Execution
// =========================================================================

/// Automatically measures the syndrome, applies Pauli recovery, and evaluates recovered fidelity.
pub fn detect_and_recover(
    state: &mut QuantumState,
    reference_clean: &QuantumState,
    code: QecCodeKind,
    offset: usize,
) -> QecRecoveryResult {
    let syndrome = extract_syndrome(state, code, offset);

    if syndrome.error_detected {
        if let (Some(q), Some(rec)) = (syndrome.faulty_qubit, syndrome.recommended_recovery) {
            match rec {
                PauliErrorKind::BitFlip => apply_pauli_x(state, q),
                PauliErrorKind::PhaseFlip => apply_pauli_z(state, q),
                PauliErrorKind::BitPhaseFlip => apply_pauli_y(state, q),
            }
        }
    }

    state.normalize();
    let fid = state.fidelity(reference_clean);
    let purity = state.real.iter().zip(&state.imag).map(|(&r, &i)| (r * r + i * i).powi(2)).sum();

    QecRecoveryResult {
        syndrome,
        fidelity_restored: fid,
        purity,
        success: (fid - 1.0).abs() < 1e-4,
    }
}

// =========================================================================
// 4. Pauli Parity Measurement & Projection Primitives
// =========================================================================

/// Measures Pauli Z_a Z_b expectation value and returns discrete eigenvalue +1 or -1.
fn measure_pauli_zz_parity(state: &QuantumState, qa: usize, qb: usize) -> i8 {
    let mut exp_z = 0.0f64;
    for i in 0..state.dim {
        let ba = (i >> qa) & 1;
        let bb = (i >> qb) & 1;
        let p = state.real[i] * state.real[i] + state.imag[i] * state.imag[i];
        let sign = if ba ^ bb == 0 { 1.0 } else { -1.0 };
        exp_z += sign * p;
    }
    if exp_z >= 0.0 { 1 } else { -1 }
}

/// Measures multi-qubit Pauli Z_q1 Z_q2 ... Z_qk expectation value.
fn measure_pauli_multi_z(state: &QuantumState, qubits: &[usize]) -> i8 {
    let mut exp_val = 0.0f64;
    for i in 0..state.dim {
        let mut parity = 0;
        for &q in qubits {
            parity ^= (i >> q) & 1;
        }
        let sign = if parity == 0 { 1.0 } else { -1.0 };
        let p = state.real[i] * state.real[i] + state.imag[i] * state.imag[i];
        exp_val += sign * p;
    }
    if exp_val >= 0.0 { 1 } else { -1 }
}

/// Measures Pauli X_a X_b expectation value by rotating to X-basis via Hadamard gates.
fn measure_pauli_xx_parity(state: &QuantumState, qa: usize, qb: usize) -> i8 {
    let mut rotated = state.clone();
    apply_hadamard(&mut rotated, qa);
    apply_hadamard(&mut rotated, qb);
    measure_pauli_zz_parity(&rotated, qa, qb)
}

/// Measures multi-qubit Pauli X expectation value.
fn measure_pauli_multi_x(state: &QuantumState, qubits: &[usize]) -> i8 {
    let mut rotated = state.clone();
    for &q in qubits {
        apply_hadamard(&mut rotated, q);
    }
    measure_pauli_multi_z(&rotated, qubits)
}

// =========================================================================
// 5. In-Place Pauli & Clifford Kernels
// =========================================================================

/// Injects an arbitrary Pauli error onto `qubit` for testing robustness.
pub fn inject_pauli_error(state: &mut QuantumState, qubit: usize, error: PauliErrorKind) {
    match error {
        PauliErrorKind::BitFlip => apply_pauli_x(state, qubit),
        PauliErrorKind::PhaseFlip => apply_pauli_z(state, qubit),
        PauliErrorKind::BitPhaseFlip => apply_pauli_y(state, qubit),
    }
}

pub fn apply_pauli_x(state: &mut QuantumState, target: usize) {
    let stride = 1 << target;
    let step = stride << 1;
    for i in (0..state.dim).step_by(step) {
        for j in 0..stride {
            let idx0 = i + j;
            let idx1 = idx0 + stride;
            state.real.swap(idx0, idx1);
            state.imag.swap(idx0, idx1);
        }
    }
}

pub fn apply_pauli_z(state: &mut QuantumState, target: usize) {
    let stride = 1 << target;
    let step = stride << 1;
    for i in (0..state.dim).step_by(step) {
        for j in 0..stride {
            let idx1 = i + j + stride;
            state.real[idx1] = -state.real[idx1];
            state.imag[idx1] = -state.imag[idx1];
        }
    }
}

pub fn apply_pauli_y(state: &mut QuantumState, target: usize) {
    let stride = 1 << target;
    let step = stride << 1;
    for i in (0..state.dim).step_by(step) {
        for j in 0..stride {
            let idx0 = i + j;
            let idx1 = idx0 + stride;
            let r0 = state.real[idx0];
            let i0 = state.imag[idx0];
            let r1 = state.real[idx1];
            let i1 = state.imag[idx1];

            state.real[idx0] = i1;
            state.imag[idx0] = -r1;
            state.real[idx1] = -i0;
            state.imag[idx1] = r0;
        }
    }
}

pub fn apply_hadamard(state: &mut QuantumState, target: usize) {
    let stride = 1 << target;
    let step = stride << 1;
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;

    for i in (0..state.dim).step_by(step) {
        for j in 0..stride {
            let idx0 = i + j;
            let idx1 = idx0 + stride;
            let r0 = state.real[idx0];
            let i0 = state.imag[idx0];
            let r1 = state.real[idx1];
            let i1 = state.imag[idx1];

            state.real[idx0] = inv_sqrt2 * (r0 + r1);
            state.imag[idx0] = inv_sqrt2 * (i0 + i1);
            state.real[idx1] = inv_sqrt2 * (r0 - r1);
            state.imag[idx1] = inv_sqrt2 * (i0 - i1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_flip_3q_recovery_on_all_qubits() {
        let n_qubits = 3;
        // Test arbitrary superposition (0.6 |0> + 0.8 |1>)
        let clean = encode_logical_state((0.6, 0.0), (0.8, 0.0), QecCodeKind::BitFlip3Q, n_qubits, 0);

        for target in 0..3 {
            let mut corrupted = clean.clone();
            inject_pauli_error(&mut corrupted, target, PauliErrorKind::BitFlip);

            // Pre-recovery fidelity is zero
            assert!(corrupted.fidelity(&clean) < 0.5);

            // Autonomous detection and recovery
            let res = detect_and_recover(&mut corrupted, &clean, QecCodeKind::BitFlip3Q, 0);
            assert!(res.success, "Failed to recover bit-flip on qubit {}", target);
            assert_eq!(res.syndrome.faulty_qubit, Some(target));
            assert!((res.fidelity_restored - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_phase_flip_3q_recovery_on_all_qubits() {
        let n_qubits = 3;
        let clean = encode_logical_state((0.8, 0.0), (0.6, 0.0), QecCodeKind::PhaseFlip3Q, n_qubits, 0);

        for target in 0..3 {
            let mut corrupted = clean.clone();
            inject_pauli_error(&mut corrupted, target, PauliErrorKind::PhaseFlip);

            let res = detect_and_recover(&mut corrupted, &clean, QecCodeKind::PhaseFlip3Q, 0);
            assert!(res.success, "Failed to recover phase-flip on qubit {}", target);
            assert_eq!(res.syndrome.faulty_qubit, Some(target));
            assert!((res.fidelity_restored - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_steane_7q_recovers_arbitrary_single_qubit_x_and_z() {
        let n_qubits = 7;
        let clean = encode_logical_state((1.0, 0.0), (0.0, 0.0), QecCodeKind::Steane7Q, n_qubits, 0);

        // Test single X error on qubit 2
        let mut x_err = clean.clone();
        inject_pauli_error(&mut x_err, 2, PauliErrorKind::BitFlip);
        let res_x = detect_and_recover(&mut x_err, &clean, QecCodeKind::Steane7Q, 0);
        assert!(res_x.success);
        assert_eq!(res_x.syndrome.faulty_qubit, Some(2));
        assert_eq!(res_x.syndrome.recommended_recovery, Some(PauliErrorKind::BitFlip));

        // Test single Z error on qubit 5
        let mut z_err = clean.clone();
        inject_pauli_error(&mut z_err, 5, PauliErrorKind::PhaseFlip);
        let res_z = detect_and_recover(&mut z_err, &clean, QecCodeKind::Steane7Q, 0);
        assert!(res_z.success);
        assert_eq!(res_z.syndrome.faulty_qubit, Some(5));
        assert_eq!(res_z.syndrome.recommended_recovery, Some(PauliErrorKind::PhaseFlip));

        // Test combined Y error on qubit 4
        let mut y_err = clean.clone();
        inject_pauli_error(&mut y_err, 4, PauliErrorKind::BitPhaseFlip);
        let res_y = detect_and_recover(&mut y_err, &clean, QecCodeKind::Steane7Q, 0);
        assert!(res_y.success);
        assert_eq!(res_y.syndrome.faulty_qubit, Some(4));
        assert_eq!(res_y.syndrome.recommended_recovery, Some(PauliErrorKind::BitPhaseFlip));
    }
}
