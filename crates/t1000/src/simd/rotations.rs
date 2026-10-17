// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Vectorized single- and two-qubit rotation kernels (Ry, Rz, CZ) with AVX2+FMA acceleration.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::guards::{assert_qubit, assert_qubit_pair, assert_statevector};

/// Vectorized Single-Qubit Ry rotation:
/// [new_r0, new_r1]^T = [c, -s; s, c] [r0, r1]^T
/// Executes via 256-bit FMA instructions (4 x f64 per SIMD lane).
#[inline(always)]
pub fn apply_ry_kernel(real: &mut [f64], imag: &mut [f64], qubit: usize, theta: f64) {
    let n_qubits = assert_statevector(real, imag);
    assert_qubit(qubit, n_qubits);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        apply_ry_avx2(real, imag, qubit, theta);
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                apply_ry_avx2(real, imag, qubit, theta);
                return;
            }
        }
        apply_ry_scalar(real, imag, qubit, theta);
    }
}

/// Vectorized Single-Qubit Rz rotation:
/// Multiplies |0> by exp(-i theta/2) and |1> by exp(i theta/2).
#[inline(always)]
pub fn apply_rz_kernel(real: &mut [f64], imag: &mut [f64], qubit: usize, theta: f64) {
    let n_qubits = assert_statevector(real, imag);
    assert_qubit(qubit, n_qubits);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        apply_rz_avx2(real, imag, qubit, theta);
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                apply_rz_avx2(real, imag, qubit, theta);
                return;
            }
        }
        apply_rz_scalar(real, imag, qubit, theta);
    }
}

/// Vectorized Ring Controlled-Z (CZ) gate:
/// Flips sign of amplitudes where both bit q1 and bit q2 are 1.
#[inline(always)]
pub fn apply_cz_kernel(real: &mut [f64], imag: &mut [f64], q1: usize, q2: usize) {
    let n_qubits = assert_statevector(real, imag);
    assert_qubit_pair(q1, q2, n_qubits);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        apply_cz_avx2(real, imag, q1, q2);
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            unsafe {
                apply_cz_avx2(real, imag, q1, q2);
                return;
            }
        }
        apply_cz_scalar(real, imag, q1, q2);
    }
}

// =========================================================================
// AVX2 INTRINSICS
// =========================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn apply_ry_avx2(real: &mut [f64], imag: &mut [f64], qubit: usize, theta: f64) {
    let dim = real.len();
    let (s, c) = (theta * 0.5).sin_cos();

    if qubit >= 2 {
        let block_size = 1 << qubit;
        let stride = block_size << 1;
        let c_vec = _mm256_set1_pd(c);
        let s_vec = _mm256_set1_pd(s);
        let neg_s_vec = _mm256_set1_pd(-s);

        let use_stream = dim >= (1 << 22)
            && ((real.as_ptr() as usize) & 31 == 0)
            && ((imag.as_ptr() as usize) & 31 == 0);

        let mut base = 0;
        while base < dim {
            let p_base = base + block_size;
            let mut k = 0;
            while k < block_size {
                let idx0 = base + k;
                let idx1 = p_base + k;

                let r0 = _mm256_loadu_pd(real.as_ptr().add(idx0));
                let r1 = _mm256_loadu_pd(real.as_ptr().add(idx1));
                let i0 = _mm256_loadu_pd(imag.as_ptr().add(idx0));
                let i1 = _mm256_loadu_pd(imag.as_ptr().add(idx1));

                let new_r0 = _mm256_fmadd_pd(r0, c_vec, _mm256_mul_pd(r1, neg_s_vec));
                let new_i0 = _mm256_fmadd_pd(i0, c_vec, _mm256_mul_pd(i1, neg_s_vec));
                let new_r1 = _mm256_fmadd_pd(r0, s_vec, _mm256_mul_pd(r1, c_vec));
                let new_i1 = _mm256_fmadd_pd(i0, s_vec, _mm256_mul_pd(i1, c_vec));

                if use_stream {
                    _mm256_stream_pd(real.as_mut_ptr().add(idx0), new_r0);
                    _mm256_stream_pd(real.as_mut_ptr().add(idx1), new_r1);
                    _mm256_stream_pd(imag.as_mut_ptr().add(idx0), new_i0);
                    _mm256_stream_pd(imag.as_mut_ptr().add(idx1), new_i1);
                } else {
                    _mm256_storeu_pd(real.as_mut_ptr().add(idx0), new_r0);
                    _mm256_storeu_pd(real.as_mut_ptr().add(idx1), new_r1);
                    _mm256_storeu_pd(imag.as_mut_ptr().add(idx0), new_i0);
                    _mm256_storeu_pd(imag.as_mut_ptr().add(idx1), new_i1);
                }

                k += 4;
            }
            base += stride;
        }

        if use_stream {
            _mm_sfence();
        }
    } else {
        apply_ry_scalar(real, imag, qubit, theta);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn apply_rz_avx2(real: &mut [f64], imag: &mut [f64], qubit: usize, theta: f64) {
    let dim = real.len();
    let (s_pos, c_pos) = (theta * 0.5).sin_cos();
    let (s_neg, c_neg) = (-theta * 0.5).sin_cos();

    if qubit >= 2 {
        let block_size = 1 << qubit;
        let stride = block_size << 1;

        let c_neg_v = _mm256_set1_pd(c_neg);
        let s_neg_v = _mm256_set1_pd(s_neg);
        let c_pos_v = _mm256_set1_pd(c_pos);
        let s_pos_v = _mm256_set1_pd(s_pos);

        let use_stream = dim >= (1 << 22)
            && ((real.as_ptr() as usize) & 31 == 0)
            && ((imag.as_ptr() as usize) & 31 == 0);

        let mut base = 0;
        while base < dim {
            let mut k = 0;
            while k < block_size {
                let idx = base + k;
                let r = _mm256_loadu_pd(real.as_ptr().add(idx));
                let im = _mm256_loadu_pd(imag.as_ptr().add(idx));

                let new_r = _mm256_fmsub_pd(r, c_neg_v, _mm256_mul_pd(im, s_neg_v));
                let new_im = _mm256_fmadd_pd(r, s_neg_v, _mm256_mul_pd(im, c_neg_v));

                if use_stream {
                    _mm256_stream_pd(real.as_mut_ptr().add(idx), new_r);
                    _mm256_stream_pd(imag.as_mut_ptr().add(idx), new_im);
                } else {
                    _mm256_storeu_pd(real.as_mut_ptr().add(idx), new_r);
                    _mm256_storeu_pd(imag.as_mut_ptr().add(idx), new_im);
                }
                k += 4;
            }

            let p_base = base + block_size;
            let mut k = 0;
            while k < block_size {
                let idx = p_base + k;
                let r = _mm256_loadu_pd(real.as_ptr().add(idx));
                let im = _mm256_loadu_pd(imag.as_ptr().add(idx));

                let new_r = _mm256_fmsub_pd(r, c_pos_v, _mm256_mul_pd(im, s_pos_v));
                let new_im = _mm256_fmadd_pd(r, s_pos_v, _mm256_mul_pd(im, c_pos_v));

                if use_stream {
                    _mm256_stream_pd(real.as_mut_ptr().add(idx), new_r);
                    _mm256_stream_pd(imag.as_mut_ptr().add(idx), new_im);
                } else {
                    _mm256_storeu_pd(real.as_mut_ptr().add(idx), new_r);
                    _mm256_storeu_pd(imag.as_mut_ptr().add(idx), new_im);
                }
                k += 4;
            }

            base += stride;
        }

        if use_stream {
            _mm_sfence();
        }
    } else {
        apply_rz_scalar(real, imag, qubit, theta);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_cz_avx2(real: &mut [f64], imag: &mut [f64], q1: usize, q2: usize) {
    let dim = real.len();
    let mask1 = 1 << q1;
    let mask2 = 1 << q2;
    let mask = mask1 | mask2;
    let sign_mask = _mm256_set1_pd(-0.0);

    let use_stream = dim >= (1 << 22)
        && ((real.as_ptr() as usize) & 31 == 0)
        && ((imag.as_ptr() as usize) & 31 == 0);

    let mut i = 0;
    while i < dim {
        let in_cz = (i & mask) == mask
            && ((i + 1) & mask) == mask
            && ((i + 2) & mask) == mask
            && ((i + 3) & mask) == mask;

        if in_cz {
            let r = _mm256_loadu_pd(real.as_ptr().add(i));
            let im = _mm256_loadu_pd(imag.as_ptr().add(i));
            if use_stream {
                _mm256_stream_pd(real.as_mut_ptr().add(i), _mm256_xor_pd(r, sign_mask));
                _mm256_stream_pd(imag.as_mut_ptr().add(i), _mm256_xor_pd(im, sign_mask));
            } else {
                _mm256_storeu_pd(real.as_mut_ptr().add(i), _mm256_xor_pd(r, sign_mask));
                _mm256_storeu_pd(imag.as_mut_ptr().add(i), _mm256_xor_pd(im, sign_mask));
            }
        } else {
            for k in 0..4 {
                let idx = i + k;
                if (idx & mask) == mask {
                    real[idx] = -real[idx];
                    imag[idx] = -imag[idx];
                }
            }
        }
        i += 4;
    }

    if use_stream {
        _mm_sfence();
    }
}

// =========================================================================
// SCALAR FALLBACK IMPLEMENTATIONS
// =========================================================================

#[allow(dead_code)]
pub(crate) fn apply_ry_scalar(real: &mut [f64], imag: &mut [f64], qubit: usize, theta: f64) {
    let dim = real.len();
    let mask = 1 << qubit;
    let (s, c) = (theta * 0.5).sin_cos();

    for i in 0..dim {
        if (i & mask) == 0 {
            let partner = i | mask;
            let r0 = real[i];
            let i0 = imag[i];
            let r1 = real[partner];
            let i1 = imag[partner];

            real[i] = c * r0 - s * r1;
            imag[i] = c * i0 - s * i1;
            real[partner] = s * r0 + c * r1;
            imag[partner] = s * i0 + c * i1;
        }
    }
}

#[allow(dead_code)]
pub(crate) fn apply_rz_scalar(real: &mut [f64], imag: &mut [f64], qubit: usize, theta: f64) {
    let dim = real.len();
    let mask = 1 << qubit;
    let (s_pos, c_pos) = (theta * 0.5).sin_cos();
    let (s_neg, c_neg) = (-theta * 0.5).sin_cos();

    for i in 0..dim {
        let r = real[i];
        let im = imag[i];
        if (i & mask) == 0 {
            real[i] = r * c_neg - im * s_neg;
            imag[i] = r * s_neg + im * c_neg;
        } else {
            real[i] = r * c_pos - im * s_pos;
            imag[i] = r * s_pos + im * c_pos;
        }
    }
}

#[allow(dead_code)]
pub(crate) fn apply_cz_scalar(real: &mut [f64], imag: &mut [f64], q1: usize, q2: usize) {
    let dim = real.len();
    let mask1 = 1 << q1;
    let mask2 = 1 << q2;
    for i in 0..dim {
        if (i & mask1) != 0 && (i & mask2) != 0 {
            real[i] = -real[i];
            imag[i] = -imag[i];
        }
    }
}
