// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Simple device-error models acting on a circuit's output probability distribution.
//!
//! * **Readout error**: each measured bit is flipped independently, `P(read 1 | true 0) = p01` and
//!   `P(read 0 | true 1) = p10` (decay makes `p10 > p01` on real devices). This is exactly a
//!   tensor product of 2x2 confusion matrices, applied one qubit at a time.
//! * **Gate error**: a global depolarizing approximation, `p -> F p + (1 - F) / 2^n`, where
//!   `F = (1 - p1)^(n1) (1 - p2)^(n2)` for `n1` single-qubit and `n2` two-qubit gates. It models
//!   errors that scramble the state; it ignores their structure and the cost of state preparation.
//! * **Readout mitigation**: applying the inverse confusion matrices to measured frequencies.

/// Independent per-qubit readout flip probabilities.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReadoutError {
    /// `P(read 1 | true 0)`.
    pub p01: f64,
    /// `P(read 0 | true 1)`.
    pub p10: f64,
}

impl ReadoutError {
    /// Typical asymmetric device: `p10 = e`, `p01 = e / 2`.
    pub fn typical(e: f64) -> Self {
        Self {
            p01: e / 2.0,
            p10: e,
        }
    }
}

/// Applies the 2x2 matrix `[[a, b], [c, d]]` to every qubit's (bit 0, bit 1) pair of `p`.
fn apply_per_qubit(p: &mut [f64], n: usize, m: [[f64; 2]; 2]) {
    assert_eq!(p.len(), 1 << n, "distribution must have 2^n entries");
    for q in 0..n {
        let bit = 1usize << q;
        for base in (0..p.len()).step_by(2 * bit) {
            for i in base..base + bit {
                let (x0, x1) = (p[i], p[i + bit]);
                p[i] = m[0][0] * x0 + m[0][1] * x1;
                p[i + bit] = m[1][0] * x0 + m[1][1] * x1;
            }
        }
    }
}

/// The distribution a device with readout error reports for true distribution `p`.
pub fn apply_readout(p: &mut [f64], n: usize, e: ReadoutError) {
    apply_per_qubit(p, n, [[1.0 - e.p01, e.p10], [e.p01, 1.0 - e.p10]]);
}

/// Inverts [`apply_readout`] (readout mitigation). Entries may become slightly negative.
pub fn invert_readout(p: &mut [f64], n: usize, e: ReadoutError) {
    let det = (1.0 - e.p01) * (1.0 - e.p10) - e.p01 * e.p10;
    assert!(det.abs() > 1e-9, "readout confusion matrix is singular");
    apply_per_qubit(
        p,
        n,
        [
            [(1.0 - e.p10) / det, -e.p10 / det],
            [-e.p01 / det, (1.0 - e.p01) / det],
        ],
    );
}

/// Global depolarizing channel: `p -> fidelity * p + (1 - fidelity) / len`.
pub fn depolarize(p: &mut [f64], fidelity: f64) {
    let uniform = (1.0 - fidelity) / p.len() as f64;
    p.iter_mut().for_each(|v| *v = fidelity * *v + uniform);
}

/// Circuit fidelity `(1 - p1)^n1 (1 - p2)^n2` for the hardware-efficient ansatz: each layer has
/// `n` Ry, `n` Rz and a ring of `n` CZ gates.
pub fn ansatz_fidelity(n_qubits: usize, depth: usize, p1: f64, p2: f64) -> f64 {
    (1.0 - p1).powi((2 * n_qubits * depth) as i32) * (1.0 - p2).powi((n_qubits * depth) as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn distribution(n: usize) -> Vec<f64> {
        let raw: Vec<f64> = (0..1usize << n).map(|i| 1.0 + (i * 7 % 5) as f64).collect();
        let total: f64 = raw.iter().sum();
        raw.iter().map(|v| v / total).collect()
    }

    #[test]
    fn readout_keeps_total_probability_and_matches_the_closed_form() {
        let n = 4;
        let e = ReadoutError {
            p01: 0.01,
            p10: 0.03,
        };
        let mut p = vec![0.0; 1 << n];
        p[0] = 1.0; // true state |0000>
        apply_readout(&mut p, n, e);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((p[0] - (1.0f64 - e.p01).powi(4)).abs() < 1e-12);
        let mut q = vec![0.0; 1 << n];
        q[(1 << n) - 1] = 1.0; // true state |1111>
        apply_readout(&mut q, n, e);
        assert!((q[(1 << n) - 1] - (1.0f64 - e.p10).powi(4)).abs() < 1e-12);
    }

    #[test]
    fn mitigation_inverts_readout_error() {
        let n = 5;
        let e = ReadoutError::typical(0.04);
        let original = distribution(n);
        let mut p = original.clone();
        apply_readout(&mut p, n, e);
        invert_readout(&mut p, n, e);
        assert!(p.iter().zip(&original).all(|(a, b)| (a - b).abs() < 1e-12));
    }

    #[test]
    fn symmetric_readout_error_shrinks_a_weight_w_parity_by_one_minus_two_e_to_the_w() {
        let (n, e) = (4, 0.05);
        let original = distribution(n);
        let parity = |p: &[f64], mask: usize| -> f64 {
            p.iter()
                .enumerate()
                .map(|(x, v)| {
                    if (x & mask).count_ones() % 2 == 0 {
                        *v
                    } else {
                        -*v
                    }
                })
                .sum()
        };
        let mut noisy = original.clone();
        apply_readout(&mut noisy, n, ReadoutError { p01: e, p10: e });
        for mask in [0b0001usize, 0b0011, 0b0111, 0b1111] {
            let w = mask.count_ones() as i32;
            let want = parity(&original, mask) * (1.0f64 - 2.0 * e).powi(w);
            assert!((parity(&noisy, mask) - want).abs() < 1e-12, "mask {mask:b}");
        }
    }

    #[test]
    fn depolarizing_scales_non_trivial_parities_by_the_fidelity() {
        let n = 3;
        let original = distribution(n);
        let mut p = original.clone();
        depolarize(&mut p, 0.8);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let z0 = |p: &[f64]| -> f64 {
            p.iter()
                .enumerate()
                .map(|(x, v)| if x & 1 == 0 { *v } else { -*v })
                .sum()
        };
        assert!((z0(&p) - 0.8 * z0(&original)).abs() < 1e-12);
        let mut fully = original;
        depolarize(&mut fully, 0.0);
        assert!(fully.iter().all(|v| (v - 0.125).abs() < 1e-12));
    }

    #[test]
    fn fidelity_follows_the_gate_counts() {
        // depth 4, 10 qubits: 80 single-qubit rotations and 40 CZ gates.
        let f = ansatz_fidelity(10, 4, 1e-3, 1e-2);
        assert!((f - 0.999f64.powi(80) * 0.99f64.powi(40)).abs() < 1e-12);
        assert_eq!(ansatz_fidelity(10, 4, 0.0, 0.0), 1.0);
    }
}
