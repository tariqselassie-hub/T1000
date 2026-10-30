// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Vectorized Pauli observable expectation kernels.
//!
//! These back [`crate::measurements::observable::PauliObservable::expectation_value`], which used
//! to be a scalar pass over the statevector. Every kernel here is a single read-only pass and
//! runtime-dispatches to AVX2+FMA when available:
//!
//! * [`measure_single_qubit_x`] — `<X_q> = 2 * Re<psi|psi ^ mask>`.
//! * [`measure_zz_pair`], [`measure_xx_pair`], [`measure_yy_pair`] — two-body correlators.
//! * [`measure_all_z_parity`] — `<Z^⊗N> = sum_k (-1)^popcount(k) |psi_k|^2`.
//! * [`measure_global_mean_z`] — `(1/N) sum_q <Z_q>` in one fused pass.
//!
//! Qubits 0 and 1 pair amplitudes inside a 4-wide vector, so those cases fall back to the scalar
//! loop (they are a negligible share of readout cost); higher qubits keep the partner vector
//! `base ^ mask` contiguous and are handled by full AVX2 loads.

use super::guards::{assert_qubit, assert_qubit_pair, assert_statevector};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// `<X_q>` for a single qubit.
#[inline(always)]
pub fn measure_single_qubit_x(real: &[f64], imag: &[f64], qubit: usize) -> f64 {
    let n_qubits = assert_statevector(real, imag);
    assert_qubit(qubit, n_qubits);
    let mask = 1usize << qubit;
    #[cfg(target_arch = "x86_64")]
    if avx2_fma_available() {
        // SAFETY: the feature check above guarantees AVX2+FMA.
        return unsafe { measure_cross_avx2(real, imag, mask, false) };
    }
    measure_cross_scalar(real, imag, mask, false)
}

/// `<Z_a Z_b>`.
#[inline(always)]
pub fn measure_zz_pair(real: &[f64], imag: &[f64], qa: usize, qb: usize) -> f64 {
    let n_qubits = assert_statevector(real, imag);
    assert_qubit_pair(qa, qb, n_qubits);
    #[cfg(target_arch = "x86_64")]
    if avx2_fma_available() {
        // SAFETY: the feature check above guarantees AVX2+FMA.
        return unsafe { measure_zz_pair_avx2(real, imag, qa, qb) };
    }
    measure_zz_pair_scalar(real, imag, qa, qb)
}

/// `<X_a X_b>`.
#[inline(always)]
pub fn measure_xx_pair(real: &[f64], imag: &[f64], qa: usize, qb: usize) -> f64 {
    let n_qubits = assert_statevector(real, imag);
    assert_qubit_pair(qa, qb, n_qubits);
    let mask = (1usize << qa) | (1usize << qb);
    #[cfg(target_arch = "x86_64")]
    if avx2_fma_available() {
        // SAFETY: the feature check above guarantees AVX2+FMA.
        return unsafe { measure_cross_avx2(real, imag, mask, false) };
    }
    measure_cross_scalar(real, imag, mask, false)
}

/// `<Y_a Y_b>`.
#[inline(always)]
pub fn measure_yy_pair(real: &[f64], imag: &[f64], qa: usize, qb: usize) -> f64 {
    let n_qubits = assert_statevector(real, imag);
    assert_qubit_pair(qa, qb, n_qubits);
    let mask = (1usize << qa) | (1usize << qb);
    #[cfg(target_arch = "x86_64")]
    if avx2_fma_available() {
        // SAFETY: the feature check above guarantees AVX2+FMA.
        return unsafe { measure_cross_avx2(real, imag, mask, true) };
    }
    measure_cross_scalar(real, imag, mask, true)
}

/// `<Z^⊗N>`, the full parity operator.
#[inline(always)]
pub fn measure_all_z_parity(real: &[f64], imag: &[f64]) -> f64 {
    assert_statevector(real, imag);
    #[cfg(target_arch = "x86_64")]
    if avx2_fma_available() {
        // SAFETY: the feature check above guarantees AVX2+FMA.
        return unsafe { measure_all_z_parity_avx2(real, imag) };
    }
    measure_all_z_parity_scalar(real, imag)
}

/// `(1/N) sum_q <Z_q>`, evaluated in a single fused pass.
#[inline(always)]
pub fn measure_global_mean_z(real: &[f64], imag: &[f64]) -> f64 {
    assert_statevector(real, imag);
    #[cfg(target_arch = "x86_64")]
    if avx2_fma_available() {
        // SAFETY: the feature check above guarantees AVX2+FMA.
        return unsafe { measure_global_mean_z_avx2(real, imag) };
    }
    measure_global_mean_z_scalar(real, imag)
}

// =========================================================================
// SCALAR FALLBACKS
// =========================================================================

fn measure_zz_pair_scalar(real: &[f64], imag: &[f64], qa: usize, qb: usize) -> f64 {
    let mask_a = 1usize << qa;
    let mask_b = 1usize << qb;
    let mut acc = 0.0;
    for idx in 0..real.len() {
        let prob = real[idx] * real[idx] + imag[idx] * imag[idx];
        let sign = if ((idx & mask_a) != 0) == ((idx & mask_b) != 0) {
            1.0
        } else {
            -1.0
        };
        acc += sign * prob;
    }
    acc
}

/// `2 * sum_{idx < idx^mask} Re(conj(psi_idx) psi_partner)`, with an optional YY sign.
fn measure_cross_scalar(real: &[f64], imag: &[f64], mask: usize, yy: bool) -> f64 {
    let mut acc = 0.0;
    for idx in 0..real.len() {
        let partner = idx ^ mask;
        if idx >= partner {
            continue;
        }
        let cross = real[idx] * real[partner] + imag[idx] * imag[partner];
        let sign = if yy && (idx & mask).count_ones() & 1 == 0 {
            -1.0
        } else {
            1.0
        };
        acc += 2.0 * sign * cross;
    }
    acc
}

fn measure_all_z_parity_scalar(real: &[f64], imag: &[f64]) -> f64 {
    let mut acc = 0.0;
    for idx in 0..real.len() {
        let prob = real[idx] * real[idx] + imag[idx] * imag[idx];
        acc += if idx.count_ones() & 1 == 0 {
            prob
        } else {
            -prob
        };
    }
    acc
}

fn measure_global_mean_z_scalar(real: &[f64], imag: &[f64]) -> f64 {
    let n_qubits = (real.len() as f64).log2().round() as usize;
    let inv_n = 1.0 / n_qubits as f64;
    let mut acc = 0.0;
    for idx in 0..real.len() {
        let prob = real[idx] * real[idx] + imag[idx] * imag[idx];
        let ones = f64::from(idx.count_ones());
        acc += (n_qubits as f64 - 2.0 * ones) * inv_n * prob;
    }
    acc
}

// =========================================================================
// AVX2 INTRINSICS
// =========================================================================

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn avx2_fma_available() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn horizontal_sum(v: __m256d) -> f64 {
    let mut lanes = [0.0f64; 4];
    _mm256_storeu_pd(lanes.as_mut_ptr(), v);
    lanes[0] + lanes[1] + lanes[2] + lanes[3]
}

/// Amplitudes `base` and `base ^ mask` live in two disjoint, 4-aligned vectors whenever
/// `mask & 3 == 0`; lower qubits pair inside a vector and use the scalar loop.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn measure_cross_avx2(real: &[f64], imag: &[f64], mask: usize, yy: bool) -> f64 {
    let dim = real.len();
    if mask & 3 != 0 || dim < 4 {
        return measure_cross_scalar(real, imag, mask, yy);
    }
    let mut acc = _mm256_setzero_pd();
    let mut base = 0usize;
    while base < dim {
        let partner = base ^ mask;
        if base < partner {
            let r1 = _mm256_loadu_pd(real.as_ptr().add(base));
            let i1 = _mm256_loadu_pd(imag.as_ptr().add(base));
            let r2 = _mm256_loadu_pd(real.as_ptr().add(partner));
            let i2 = _mm256_loadu_pd(imag.as_ptr().add(partner));
            let mut cross = _mm256_fmadd_pd(r1, r2, _mm256_mul_pd(i1, i2));
            if yy {
                // YY is -1 when the two measured bits agree; both flip under the partner mask.
                let sign = if (base & mask).count_ones() & 1 == 0 {
                    -1.0
                } else {
                    1.0
                };
                cross = _mm256_mul_pd(cross, _mm256_set1_pd(sign));
            }
            acc = _mm256_add_pd(acc, cross);
        }
        base += 4;
    }
    2.0 * horizontal_sum(acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn measure_zz_pair_avx2(real: &[f64], imag: &[f64], qa: usize, qb: usize) -> f64 {
    let dim = real.len();
    let mask = (1usize << qa) | (1usize << qb);
    if mask & 3 != 0 || dim < 4 {
        return measure_zz_pair_scalar(real, imag, qa, qb);
    }
    let mut acc = _mm256_setzero_pd();
    let mut base = 0usize;
    while base < dim {
        let r = _mm256_loadu_pd(real.as_ptr().add(base));
        let i = _mm256_loadu_pd(imag.as_ptr().add(base));
        let prob = _mm256_fmadd_pd(r, r, _mm256_mul_pd(i, i));
        let sign = if (base & mask).count_ones() & 1 == 1 {
            -1.0
        } else {
            1.0
        };
        acc = _mm256_fmadd_pd(prob, _mm256_set1_pd(sign), acc);
        base += 4;
    }
    horizontal_sum(acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn measure_all_z_parity_avx2(real: &[f64], imag: &[f64]) -> f64 {
    let dim = real.len();
    if dim < 4 {
        return measure_all_z_parity_scalar(real, imag);
    }
    // Low two index bits have popcount parity [even, odd, odd, even] => [+, -, -, +].
    let pattern = _mm256_set_pd(1.0, -1.0, -1.0, 1.0);
    let mut acc = _mm256_setzero_pd();
    let mut base = 0usize;
    while base < dim {
        let r = _mm256_loadu_pd(real.as_ptr().add(base));
        let i = _mm256_loadu_pd(imag.as_ptr().add(base));
        let prob = _mm256_fmadd_pd(r, r, _mm256_mul_pd(i, i));
        let high = if base.count_ones() & 1 == 0 {
            1.0
        } else {
            -1.0
        };
        let signs = _mm256_mul_pd(pattern, _mm256_set1_pd(high));
        acc = _mm256_fmadd_pd(prob, signs, acc);
        base += 4;
    }
    horizontal_sum(acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn measure_global_mean_z_avx2(real: &[f64], imag: &[f64]) -> f64 {
    let dim = real.len();
    if dim < 4 {
        return measure_global_mean_z_scalar(real, imag);
    }
    let n_qubits = (dim as f64).log2().round() as usize;
    let inv_n = 1.0 / n_qubits as f64;
    // popcount of the low two bits is [0, 1, 1, 2], so the per-lane correction is [-2h/n].
    let pattern = _mm256_set_pd(-4.0 * inv_n, -2.0 * inv_n, -2.0 * inv_n, 0.0);
    let mut acc = _mm256_setzero_pd();
    let mut base = 0usize;
    while base < dim {
        let r = _mm256_loadu_pd(real.as_ptr().add(base));
        let i = _mm256_loadu_pd(imag.as_ptr().add(base));
        let prob = _mm256_fmadd_pd(r, r, _mm256_mul_pd(i, i));
        let high_ones = f64::from(base.count_ones());
        let high = (n_qubits as f64 - 2.0 * high_ones) * inv_n;
        let weights = _mm256_add_pd(pattern, _mm256_set1_pd(high));
        acc = _mm256_fmadd_pd(prob, weights, acc);
        base += 4;
    }
    horizontal_sum(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_state(n_qubits: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let dim = 1usize << n_qubits;
        let mut state = seed | 1;
        let mut real = Vec::with_capacity(dim);
        let mut imag = Vec::with_capacity(dim);
        let mut norm = 0.0;
        for _ in 0..dim {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let i = (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            norm += r * r + i * i;
            real.push(r);
            imag.push(i);
        }
        let inv = 1.0 / norm.sqrt();
        for v in real.iter_mut() {
            *v *= inv;
        }
        for v in imag.iter_mut() {
            *v *= inv;
        }
        (real, imag)
    }

    fn assert_close(got: f64, want: f64, label: &str) {
        assert!(
            (got - want).abs() < 1e-11,
            "{label}: got {got}, want {want}"
        );
    }

    #[test]
    fn avx2_kernels_match_scalar() {
        for n in 1..=8 {
            let (real, imag) = random_state(n, 7 + n as u64);
            let global_want = measure_global_mean_z_scalar(&real, &imag);
            assert_close(
                measure_global_mean_z(&real, &imag),
                global_want,
                "global mean Z",
            );
            assert_close(
                measure_all_z_parity(&real, &imag),
                measure_all_z_parity_scalar(&real, &imag),
                "all Z parity",
            );
            for q in 0..n {
                assert_close(
                    measure_single_qubit_x(&real, &imag, q),
                    measure_cross_scalar(&real, &imag, 1usize << q, false),
                    "single X",
                );
            }
            for qa in 0..n {
                for qb in (qa + 1)..n {
                    assert_close(
                        measure_zz_pair(&real, &imag, qa, qb),
                        measure_zz_pair_scalar(&real, &imag, qa, qb),
                        "ZZ pair",
                    );
                    assert_close(
                        measure_xx_pair(&real, &imag, qa, qb),
                        measure_cross_scalar(&real, &imag, (1usize << qa) | (1usize << qb), false),
                        "XX pair",
                    );
                    assert_close(
                        measure_yy_pair(&real, &imag, qa, qb),
                        measure_cross_scalar(&real, &imag, (1usize << qa) | (1usize << qb), true),
                        "YY pair",
                    );
                }
            }
        }
    }

    #[test]
    fn global_mean_equals_mean_of_z() {
        for n in 1..=6 {
            let (real, imag) = random_state(n, 99 + n as u64);
            let z_of = |q: usize| -> f64 {
                let mask = 1usize << q;
                let mut acc = 0.0;
                for idx in 0..real.len() {
                    let prob = real[idx] * real[idx] + imag[idx] * imag[idx];
                    acc += if idx & mask == 0 { prob } else { -prob };
                }
                acc
            };
            let mean = (0..n).map(z_of).sum::<f64>() / n as f64;
            assert_close(measure_global_mean_z(&real, &imag), mean, "mean of Z");
        }
    }
}
