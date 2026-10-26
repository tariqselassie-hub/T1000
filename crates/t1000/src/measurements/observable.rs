// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

use crate::state::QuantumState;
use serde::{Deserialize, Serialize};

/// Pauli observable string or single-qubit measurement operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PauliObservable {
    /// Measure sigma_z on qubit `i`: eigenvalues +1 (|0>) and -1 (|1>)
    Z(usize),
    /// Measure sigma_x on qubit `i`: expectation via Hadamard transform / amplitude overlap
    X(usize),
    /// Bipartite correlation sigma_z(i) * sigma_z(j)
    ZZ(usize, usize),
    /// Transverse bipartite correlation sigma_x(i) * sigma_x(j)
    XX(usize, usize),
    /// Transverse bipartite correlation sigma_y(i) * sigma_y(j)
    YY(usize, usize),
    /// Full state parity operator: Z^{\otimes N} = \bigotimes_{i=0}^{N-1} Z_i
    AllZ,
    /// Multi-qubit parity or linear combination: sum_k w_k Z_k
    LinearZ(Vec<(usize, f64)>),
    /// Global multi-qubit observable: `<H> = (1/N) * sum_{i=0}^{N-1} <Z_i>`
    /// Distributes gradient backpropagation directly across all qubits.
    GlobalMeanZ,
}

impl PauliObservable {
    /// Returns `O |state>` (all variants are Hermitian, so this is also `O^dagger |state>`).
    pub fn apply(&self, state: &QuantumState) -> QuantumState {
        let n_qubits = crate::simd::assert_statevector(&state.real, &state.imag);
        let z = |idx: usize, q: usize| if (idx >> q) & 1 == 0 { 1.0 } else { -1.0 };
        fn scale_by(out: &mut QuantumState, f: impl Fn(usize) -> f64) {
            for idx in 0..out.dim {
                let w = f(idx);
                out.real[idx] *= w;
                out.imag[idx] *= w;
            }
        }
        let mut out = state.clone();
        match self {
            PauliObservable::Z(q) => {
                crate::simd::assert_qubit(*q, n_qubits);
                scale_by(&mut out, |idx| z(idx, *q));
            }
            PauliObservable::ZZ(a, b) => {
                crate::simd::assert_qubit(*a, n_qubits);
                crate::simd::assert_qubit(*b, n_qubits);
                scale_by(&mut out, |idx| z(idx, *a) * z(idx, *b));
            }
            PauliObservable::X(q) => {
                crate::simd::assert_qubit(*q, n_qubits);
                let mask = 1usize << q;
                for idx in 0..state.dim {
                    out.real[idx] = state.real[idx ^ mask];
                    out.imag[idx] = state.imag[idx ^ mask];
                }
            }
            PauliObservable::XX(a, b) => {
                crate::simd::assert_qubit(*a, n_qubits);
                crate::simd::assert_qubit(*b, n_qubits);
                let mask = (1usize << *a) | (1usize << *b);
                for idx in 0..state.dim {
                    out.real[idx] = state.real[idx ^ mask];
                    out.imag[idx] = state.imag[idx ^ mask];
                }
            }
            PauliObservable::YY(a, b) => {
                crate::simd::assert_qubit(*a, n_qubits);
                crate::simd::assert_qubit(*b, n_qubits);
                let mask = (1usize << *a) | (1usize << *b);
                for idx in 0..state.dim {
                    let bit_a = (idx >> *a) & 1;
                    let bit_b = (idx >> *b) & 1;
                    let sign = if bit_a == bit_b { -1.0 } else { 1.0 };
                    let partner = idx ^ mask;
                    out.real[idx] = sign * state.real[partner];
                    out.imag[idx] = sign * state.imag[partner];
                }
            }
            PauliObservable::AllZ => {
                scale_by(
                    &mut out,
                    |idx| {
                        if idx.count_ones() % 2 == 0 {
                            1.0
                        } else {
                            -1.0
                        }
                    },
                );
            }
            PauliObservable::LinearZ(weights) => {
                for &(q, _) in weights {
                    crate::simd::assert_qubit(q, n_qubits);
                }
                scale_by(&mut out, |idx| {
                    weights.iter().map(|&(q, w)| w * z(idx, q)).sum()
                });
            }
            PauliObservable::GlobalMeanZ => {
                let n = n_qubits as f64;
                scale_by(&mut out, |idx| {
                    let ones = idx.count_ones() as f64;
                    (n - 2.0 * ones) / n
                });
            }
        }
        out
    }

    /// Directly accumulates `coeff * (O |state>)` into `out_real` and `out_imag` with zero heap allocations.
    pub fn accumulate_scaled(
        &self,
        state: &QuantumState,
        coeff: f64,
        out_real: &mut [f64],
        out_imag: &mut [f64],
    ) {
        if coeff == 0.0 {
            return;
        }
        let n_qubits = crate::simd::assert_statevector(&state.real, &state.imag);
        match self {
            PauliObservable::Z(q) => {
                crate::simd::assert_qubit(*q, n_qubits);
                let mask = 1usize << q;
                for idx in 0..state.dim {
                    let s = if (idx & mask) == 0 { coeff } else { -coeff };
                    out_real[idx] += s * state.real[idx];
                    out_imag[idx] += s * state.imag[idx];
                }
            }
            PauliObservable::ZZ(a, b) => {
                crate::simd::assert_qubit(*a, n_qubits);
                crate::simd::assert_qubit(*b, n_qubits);
                let mask_a = 1usize << a;
                let mask_b = 1usize << b;
                for idx in 0..state.dim {
                    let sign_a = (idx & mask_a) != 0;
                    let sign_b = (idx & mask_b) != 0;
                    let s = if sign_a == sign_b { coeff } else { -coeff };
                    out_real[idx] += s * state.real[idx];
                    out_imag[idx] += s * state.imag[idx];
                }
            }
            PauliObservable::X(q) => {
                crate::simd::assert_qubit(*q, n_qubits);
                let mask = 1usize << q;
                for idx in 0..state.dim {
                    let partner = idx ^ mask;
                    out_real[idx] += coeff * state.real[partner];
                    out_imag[idx] += coeff * state.imag[partner];
                }
            }
            PauliObservable::XX(a, b) => {
                crate::simd::assert_qubit(*a, n_qubits);
                crate::simd::assert_qubit(*b, n_qubits);
                let mask = (1usize << *a) | (1usize << *b);
                for idx in 0..state.dim {
                    let partner = idx ^ mask;
                    out_real[idx] += coeff * state.real[partner];
                    out_imag[idx] += coeff * state.imag[partner];
                }
            }
            PauliObservable::YY(a, b) => {
                crate::simd::assert_qubit(*a, n_qubits);
                crate::simd::assert_qubit(*b, n_qubits);
                let mask = (1usize << *a) | (1usize << *b);
                for idx in 0..state.dim {
                    let bit_a = (idx >> *a) & 1;
                    let bit_b = (idx >> *b) & 1;
                    let sign = if bit_a == bit_b { -1.0 } else { 1.0 };
                    let partner = idx ^ mask;
                    let s = coeff * sign;
                    out_real[idx] += s * state.real[partner];
                    out_imag[idx] += s * state.imag[partner];
                }
            }
            PauliObservable::AllZ => {
                for idx in 0..state.dim {
                    let sign = if idx.count_ones() % 2 == 0 { 1.0 } else { -1.0 };
                    let s = coeff * sign;
                    out_real[idx] += s * state.real[idx];
                    out_imag[idx] += s * state.imag[idx];
                }
            }
            PauliObservable::LinearZ(weights) => {
                for &(q, _) in weights {
                    crate::simd::assert_qubit(q, n_qubits);
                }
                for idx in 0..state.dim {
                    let mut sum_w = 0.0;
                    for &(q, w) in weights {
                        let sign = if (idx & (1usize << q)) == 0 {
                            1.0
                        } else {
                            -1.0
                        };
                        sum_w += w * sign;
                    }
                    let s = coeff * sum_w;
                    out_real[idx] += s * state.real[idx];
                    out_imag[idx] += s * state.imag[idx];
                }
            }
            PauliObservable::GlobalMeanZ => {
                let n = n_qubits as f64;
                for idx in 0..state.dim {
                    let ones = idx.count_ones() as f64;
                    let net = (n - 2.0 * ones) / n;
                    let s = coeff * net;
                    out_real[idx] += s * state.real[idx];
                    out_imag[idx] += s * state.imag[idx];
                }
            }
        }
    }

    /// Compute exact quantum expectation value:
    /// `<O> = <psi | O | psi>`
    pub fn expectation_value(&self, state: &QuantumState) -> f64 {
        let n_qubits = (state.dim as f64).log2().round() as usize;

        match self {
            PauliObservable::Z(qubit_idx) => {
                assert!(*qubit_idx < n_qubits, "Qubit index out of bounds");
                let mask = 1 << qubit_idx;
                let mut exp = 0.0;
                for idx in 0..state.dim {
                    let prob =
                        state.real[idx] * state.real[idx] + state.imag[idx] * state.imag[idx];
                    let sign = if (idx & mask) == 0 { 1.0 } else { -1.0 };
                    exp += sign * prob;
                }
                exp
            }
            PauliObservable::ZZ(q1, q2) => {
                assert!(
                    *q1 < n_qubits && *q2 < n_qubits,
                    "Qubit index out of bounds"
                );
                let mask1 = 1 << q1;
                let mask2 = 1 << q2;
                let mut exp = 0.0;
                for idx in 0..state.dim {
                    let prob =
                        state.real[idx] * state.real[idx] + state.imag[idx] * state.imag[idx];
                    let b1 = (idx & mask1) != 0;
                    let b2 = (idx & mask2) != 0;
                    let sign = if b1 == b2 { 1.0 } else { -1.0 };
                    exp += sign * prob;
                }
                exp
            }
            PauliObservable::X(qubit_idx) => {
                assert!(*qubit_idx < n_qubits, "Qubit index out of bounds");
                let flip_mask = 1 << qubit_idx;
                let mut exp = 0.0;
                for idx in 0..state.dim {
                    let partner = idx ^ flip_mask;
                    if idx < partner {
                        let r1 = state.real[idx];
                        let i1 = state.imag[idx];
                        let r2 = state.real[partner];
                        let i2 = state.imag[partner];
                        let real_cross = r1 * r2 + i1 * i2;
                        exp += 2.0 * real_cross;
                    }
                }
                exp
            }
            PauliObservable::XX(q1, q2) => {
                assert!(
                    *q1 < n_qubits && *q2 < n_qubits,
                    "Qubit index out of bounds"
                );
                let mask = (1usize << *q1) | (1usize << *q2);
                let mut exp = 0.0;
                for idx in 0..state.dim {
                    let partner = idx ^ mask;
                    if idx < partner {
                        let r1 = state.real[idx];
                        let i1 = state.imag[idx];
                        let r2 = state.real[partner];
                        let i2 = state.imag[partner];
                        exp += 2.0 * (r1 * r2 + i1 * i2);
                    }
                }
                exp
            }
            PauliObservable::YY(q1, q2) => {
                assert!(
                    *q1 < n_qubits && *q2 < n_qubits,
                    "Qubit index out of bounds"
                );
                let mask = (1usize << *q1) | (1usize << *q2);
                let mut exp = 0.0;
                for idx in 0..state.dim {
                    let partner = idx ^ mask;
                    if idx < partner {
                        let bit_a = (idx >> *q1) & 1;
                        let bit_b = (idx >> *q2) & 1;
                        let sign = if bit_a == bit_b { -1.0 } else { 1.0 };
                        let r1 = state.real[idx];
                        let i1 = state.imag[idx];
                        let r2 = state.real[partner];
                        let i2 = state.imag[partner];
                        exp += 2.0 * sign * (r1 * r2 + i1 * i2);
                    }
                }
                exp
            }
            PauliObservable::AllZ => {
                let mut exp = 0.0;
                for idx in 0..state.dim {
                    let prob =
                        state.real[idx] * state.real[idx] + state.imag[idx] * state.imag[idx];
                    let sign = if idx.count_ones() % 2 == 0 { 1.0 } else { -1.0 };
                    exp += sign * prob;
                }
                exp
            }
            PauliObservable::LinearZ(weights) => weights
                .iter()
                .map(|&(q, w)| w * PauliObservable::Z(q).expectation_value(state))
                .sum(),
            PauliObservable::GlobalMeanZ => {
                let mut exp = 0.0;
                let inv_n = 1.0 / (n_qubits as f64);
                for idx in 0..state.dim {
                    let prob =
                        state.real[idx] * state.real[idx] + state.imag[idx] * state.imag[idx];
                    let ones = idx.count_ones() as usize;
                    let zeros = n_qubits.saturating_sub(ones);
                    let net_z = (zeros as f64 - ones as f64) * inv_n;
                    exp += net_z * prob;
                }
                exp
            }
        }
    }
}
