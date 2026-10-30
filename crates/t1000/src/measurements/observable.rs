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
    ///
    /// Every variant dispatches to a single-pass AVX2+FMA kernel in
    /// [`crate::simd::pauli`] (with a scalar fallback), instead of a scalar loop over the
    /// statevector. `LinearZ` folds the weighted single-qubit Z reductions directly.
    pub fn expectation_value(&self, state: &QuantumState) -> f64 {
        let (real, imag) = (&state.real[..], &state.imag[..]);
        match self {
            PauliObservable::Z(q) => crate::simd::measure_single_qubit_z(real, imag, *q),
            PauliObservable::ZZ(q1, q2) => crate::simd::measure_zz_pair(real, imag, *q1, *q2),
            PauliObservable::X(q) => crate::simd::measure_single_qubit_x(real, imag, *q),
            PauliObservable::XX(q1, q2) => crate::simd::measure_xx_pair(real, imag, *q1, *q2),
            PauliObservable::YY(q1, q2) => crate::simd::measure_yy_pair(real, imag, *q1, *q2),
            PauliObservable::AllZ => crate::simd::measure_all_z_parity(real, imag),
            PauliObservable::LinearZ(weights) => weights
                .iter()
                .map(|&(q, w)| w * crate::simd::measure_single_qubit_z(real, imag, q))
                .sum(),
            PauliObservable::GlobalMeanZ => crate::simd::measure_global_mean_z(real, imag),
        }
    }
}
