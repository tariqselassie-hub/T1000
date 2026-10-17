// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Vectorized Hilbert space algebraic kernels: L2 normalization, inner products, and Rademacher phases.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::guards::assert_statevector;

/// Vectorized L2 Norm normalization: sum_i (r_i^2 + im_i^2) via FMA.
#[inline(always)]
pub fn normalize_kernel(real: &mut [f64], imag: &mut [f64]) {
    assert_eq!(real.len(), imag.len(), "real/imag length mismatch");
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        normalize_avx2(real, imag);
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                normalize_avx2(real, imag);
                return;
            }
        }
        normalize_scalar(real, imag);
    }
}

/// Vectorized Complex Inner Product <state_a | state_b>.
#[inline(always)]
pub fn inner_product_kernel(r1: &[f64], i1: &[f64], r2: &[f64], i2: &[f64]) -> (f64, f64) {
    let dim = r1.len();
    assert!(
        i1.len() == dim && r2.len() == dim && i2.len() == dim,
        "inner product operands must all have length {} (got {}, {}, {})",
        dim,
        i1.len(),
        r2.len(),
        i2.len()
    );
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        return inner_product_avx2(r1, i1, r2, i2);
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                return inner_product_avx2(r1, i1, r2, i2);
            }
        }
        inner_product_scalar(r1, i1, r2, i2)
    }
}

/// Vectorized Fused Walsh-Rademacher Phase Layer:
/// Transforms basis amplitudes according to the diagonal unitary:
/// U_Rad |x> = C * (-1)^popcount(x) |x>
/// where C = (-i)^n * prod_q s_q in {+1, -1, +i, -i}.
/// Executes via a single linear memory pass using 256-bit AVX2 XOR sign-bit inversions.
#[inline(always)]
pub fn apply_rademacher_phase_kernel(real: &mut [f64], imag: &mut [f64], signs: &[f64]) {
    let n_qubits = assert_statevector(real, imag);
    assert_eq!(
        signs.len(),
        n_qubits,
        "Rademacher layer needs one sign per qubit ({} signs for {} qubits)",
        signs.len(),
        n_qubits
    );
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        apply_rademacher_phase_avx2(real, imag, signs);
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            unsafe {
                apply_rademacher_phase_avx2(real, imag, signs);
                return;
            }
        }
        apply_rademacher_phase_scalar(real, imag, signs);
    }
}

// =========================================================================
// AVX2 INTRINSICS
// =========================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn normalize_avx2(real: &mut [f64], imag: &mut [f64]) {
    let dim = real.len();
    let mut sum_v0 = _mm256_setzero_pd();
    let mut sum_v1 = _mm256_setzero_pd();

    let mut i = 0;
    while i + 8 <= dim {
        let r0 = _mm256_loadu_pd(real.as_ptr().add(i));
        let i0 = _mm256_loadu_pd(imag.as_ptr().add(i));
        let r1 = _mm256_loadu_pd(real.as_ptr().add(i + 4));
        let i1 = _mm256_loadu_pd(imag.as_ptr().add(i + 4));

        sum_v0 = _mm256_fmadd_pd(r0, r0, sum_v0);
        sum_v0 = _mm256_fmadd_pd(i0, i0, sum_v0);
        sum_v1 = _mm256_fmadd_pd(r1, r1, sum_v1);
        sum_v1 = _mm256_fmadd_pd(i1, i1, sum_v1);

        i += 8;
    }

    let sum_all = _mm256_add_pd(sum_v0, sum_v1);
    let mut buf = [0.0; 4];
    _mm256_storeu_pd(buf.as_mut_ptr(), sum_all);
    let mut norm_sq = buf[0] + buf[1] + buf[2] + buf[3];

    while i < dim {
        norm_sq += real[i] * real[i] + imag[i] * imag[i];
        i += 1;
    }

    let norm = norm_sq.sqrt();
    if norm > 1e-12 {
        let inv_norm = 1.0 / norm;
        let inv_vec = _mm256_set1_pd(inv_norm);

        let mut j = 0;
        while j + 4 <= dim {
            let r = _mm256_loadu_pd(real.as_ptr().add(j));
            let im = _mm256_loadu_pd(imag.as_ptr().add(j));
            _mm256_storeu_pd(real.as_mut_ptr().add(j), _mm256_mul_pd(r, inv_vec));
            _mm256_storeu_pd(imag.as_mut_ptr().add(j), _mm256_mul_pd(im, inv_vec));
            j += 4;
        }

        while j < dim {
            real[j] *= inv_norm;
            imag[j] *= inv_norm;
            j += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn inner_product_avx2(r1: &[f64], i1: &[f64], r2: &[f64], i2: &[f64]) -> (f64, f64) {
    let dim = r1.len();
    let mut re_sum = _mm256_setzero_pd();
    let mut im_sum = _mm256_setzero_pd();

    let mut i = 0;
    while i + 4 <= dim {
        let a_r = _mm256_loadu_pd(r1.as_ptr().add(i));
        let a_i = _mm256_loadu_pd(i1.as_ptr().add(i));
        let b_r = _mm256_loadu_pd(r2.as_ptr().add(i));
        let b_i = _mm256_loadu_pd(i2.as_ptr().add(i));

        // conj(A) * B = (a_r - i a_i) * (b_r + i b_i)
        // Re = a_r * b_r + a_i * b_i
        // Im = a_r * b_i - a_i * b_r
        re_sum = _mm256_fmadd_pd(a_r, b_r, re_sum);
        re_sum = _mm256_fmadd_pd(a_i, b_i, re_sum);

        im_sum = _mm256_fmadd_pd(a_r, b_i, im_sum);
        im_sum = _mm256_fnmadd_pd(a_i, b_r, im_sum);

        i += 4;
    }

    let mut buf_re = [0.0; 4];
    let mut buf_im = [0.0; 4];
    _mm256_storeu_pd(buf_re.as_mut_ptr(), re_sum);
    _mm256_storeu_pd(buf_im.as_mut_ptr(), im_sum);

    let mut final_re = buf_re[0] + buf_re[1] + buf_re[2] + buf_re[3];
    let mut final_im = buf_im[0] + buf_im[1] + buf_im[2] + buf_im[3];

    while i < dim {
        final_re += r1[i] * r2[i] + i1[i] * i2[i];
        final_im += r1[i] * i2[i] - i1[i] * r2[i];
        i += 1;
    }

    (final_re, final_im)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_rademacher_phase_avx2(real: &mut [f64], imag: &mut [f64], signs: &[f64]) {
    let n = signs.len();
    let dim = real.len();
    assert_eq!(dim, imag.len());

    let mut prod_s = 1.0;
    for &s in signs {
        if s < 0.0 {
            prod_s = -prod_s;
        }
    }
    let rem = n % 4;

    // Mask for [+1, -1, -1, +1] where p=0 (offsets 0, 1, 2, 3)
    let mask_pos = _mm256_set_pd(0.0, -0.0, -0.0, 0.0);
    // Mask for [-1, +1, +1, -1] where p=1 (offsets 0, 1, 2, 3)
    let mask_neg = _mm256_set_pd(-0.0, 0.0, 0.0, -0.0);

    let chunks = dim / 4;
    let use_stream = dim >= (1 << 22)
        && ((real.as_ptr() as usize) & 31 == 0)
        && ((imag.as_ptr() as usize) & 31 == 0);

    if rem == 0 || rem == 2 {
        let c_sign = if rem == 0 { prod_s } else { -prod_s };
        let c_is_neg = c_sign < 0.0;

        for c_idx in 0..chunks {
            let i = c_idx * 4;
            let p = (i.count_ones() & 1) != 0;
            let flip_0 = if c_is_neg { !p } else { p };
            let mask = if !flip_0 { mask_pos } else { mask_neg };

            let r_ptr = real.as_mut_ptr().add(i);
            let im_ptr = imag.as_mut_ptr().add(i);

            let r_vec = _mm256_loadu_pd(r_ptr);
            let im_vec = _mm256_loadu_pd(im_ptr);

            if use_stream {
                _mm256_stream_pd(r_ptr, _mm256_xor_pd(r_vec, mask));
                _mm256_stream_pd(im_ptr, _mm256_xor_pd(im_vec, mask));
            } else {
                _mm256_storeu_pd(r_ptr, _mm256_xor_pd(r_vec, mask));
                _mm256_storeu_pd(im_ptr, _mm256_xor_pd(im_vec, mask));
            }
        }
    } else {
        let c_im = if rem == 1 { -prod_s } else { prod_s };
        let c_is_pos_i = c_im > 0.0;

        for c_idx in 0..chunks {
            let i = c_idx * 4;
            let p = (i.count_ones() & 1) != 0;

            let flip_r = if c_is_pos_i { !p } else { p };
            let flip_im = if c_is_pos_i { p } else { !p };

            let mask_r = if !flip_r { mask_pos } else { mask_neg };
            let mask_im = if !flip_im { mask_pos } else { mask_neg };

            let r_ptr = real.as_mut_ptr().add(i);
            let im_ptr = imag.as_mut_ptr().add(i);

            let r_vec = _mm256_loadu_pd(r_ptr);
            let im_vec = _mm256_loadu_pd(im_ptr);

            if use_stream {
                _mm256_stream_pd(r_ptr, _mm256_xor_pd(im_vec, mask_r));
                _mm256_stream_pd(im_ptr, _mm256_xor_pd(r_vec, mask_im));
            } else {
                _mm256_storeu_pd(r_ptr, _mm256_xor_pd(im_vec, mask_r));
                _mm256_storeu_pd(im_ptr, _mm256_xor_pd(r_vec, mask_im));
            }
        }
    }

    if use_stream {
        _mm_sfence();
    }

    let rem_start = chunks * 4;
    for idx in rem_start..dim {
        let p = (idx.count_ones() & 1) != 0;
        let factor = if p { -1.0 } else { 1.0 };
        let r = real[idx];
        let im = imag[idx];
        match rem {
            0 => {
                real[idx] = r * prod_s * factor;
                imag[idx] = im * prod_s * factor;
            }
            2 => {
                real[idx] = -r * prod_s * factor;
                imag[idx] = -im * prod_s * factor;
            }
            1 => {
                real[idx] = im * prod_s * factor;
                imag[idx] = -r * prod_s * factor;
            }
            3 => {
                real[idx] = -im * prod_s * factor;
                imag[idx] = r * prod_s * factor;
            }
            _ => unreachable!(),
        }
    }
}

// =========================================================================
// SCALAR FALLBACK IMPLEMENTATIONS
// =========================================================================

#[allow(dead_code)]
pub(crate) fn normalize_scalar(real: &mut [f64], imag: &mut [f64]) {
    let dim = real.len();
    let mut norm_sq = 0.0;
    for i in 0..dim {
        norm_sq += real[i] * real[i] + imag[i] * imag[i];
    }
    let norm = norm_sq.sqrt();
    if norm > 1e-12 {
        let inv = 1.0 / norm;
        for i in 0..dim {
            real[i] *= inv;
            imag[i] *= inv;
        }
    }
}

#[allow(dead_code)]
pub(crate) fn inner_product_scalar(r1: &[f64], i1: &[f64], r2: &[f64], i2: &[f64]) -> (f64, f64) {
    let mut re = 0.0;
    let mut im = 0.0;
    for idx in 0..r1.len() {
        re += r1[idx] * r2[idx] + i1[idx] * i2[idx];
        im += r1[idx] * i2[idx] - i1[idx] * r2[idx];
    }
    (re, im)
}

#[allow(dead_code)]
pub(crate) fn apply_rademacher_phase_scalar(real: &mut [f64], imag: &mut [f64], signs: &[f64]) {
    let n = signs.len();
    let dim = real.len();
    let mut prod_s = 1.0;
    for &s in signs {
        if s < 0.0 {
            prod_s = -prod_s;
        }
    }
    let rem = n % 4;

    for idx in 0..dim {
        let p = (idx.count_ones() & 1) != 0;
        let factor = if p { -1.0 } else { 1.0 };
        let r = real[idx];
        let im = imag[idx];
        match rem {
            0 => {
                real[idx] = r * prod_s * factor;
                imag[idx] = im * prod_s * factor;
            }
            2 => {
                real[idx] = -r * prod_s * factor;
                imag[idx] = -im * prod_s * factor;
            }
            1 => {
                real[idx] = im * prod_s * factor;
                imag[idx] = -r * prod_s * factor;
            }
            3 => {
                real[idx] = -im * prod_s * factor;
                imag[idx] = r * prod_s * factor;
            }
            _ => unreachable!(),
        }
    }
}
