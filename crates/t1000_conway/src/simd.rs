// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! AVX2+FMA SIMD Vectorization & Parallel Kernel Subsystem.
//! Vectorizes the 3x3 local unitary transformation [c0', c1', c2'] = U [c0, c1, c2]
//! across 4 cells simultaneously using 256-bit SIMD registers and FMA instructions.

use crate::holonomy::HolonomyTransferMatrix;
use num_complex::Complex64;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// 4-cell lane packet for SIMD vectorized evolution.
#[derive(Debug, Clone, Copy)]
pub struct SimdCellPacket4 {
    /// Indices of the 4 cells in the lattice.
    pub indices: [usize; 4],
}

/// Apply 3x3 unitary transformations to 4 cells using AVX2+FMA when available,
/// falling back to inline scalar execution otherwise.
#[inline(always)]
pub fn apply_unitary_4cells(
    re0: &mut [f64],
    im0: &mut [f64],
    re1: &mut [f64],
    im1: &mut [f64],
    re2: &mut [f64],
    im2: &mut [f64],
    indices: [usize; 4],
    unitaries: &[HolonomyTransferMatrix; 4],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                apply_unitary_4cells_avx2(
                    re0, im0, re1, im1, re2, im2, indices, unitaries,
                );
                return;
            }
        }
    }

    apply_unitary_4cells_scalar(re0, im0, re1, im1, re2, im2, indices, unitaries);
}

/// Scalar reference implementation for 4 cells.
#[inline(always)]
pub fn apply_unitary_4cells_scalar(
    re0: &mut [f64],
    im0: &mut [f64],
    re1: &mut [f64],
    im1: &mut [f64],
    re2: &mut [f64],
    im2: &mut [f64],
    indices: [usize; 4],
    unitaries: &[HolonomyTransferMatrix; 4],
) {
    for (lane, &idx) in indices.iter().enumerate() {
        let u = &unitaries[lane];
        let c = [
            Complex64::new(re0[idx], im0[idx]),
            Complex64::new(re1[idx], im1[idx]),
            Complex64::new(re2[idx], im2[idx]),
        ];
        let c_next = u.apply(c);
        re0[idx] = c_next[0].re;
        im0[idx] = c_next[0].im;
        re1[idx] = c_next[1].re;
        im1[idx] = c_next[1].im;
        re2[idx] = c_next[2].re;
        im2[idx] = c_next[2].im;
    }
}

/// AVX2+FMA hardware-accelerated 4-cell batch evolution.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn apply_unitary_4cells_avx2(
    re0: &mut [f64],
    im0: &mut [f64],
    re1: &mut [f64],
    im1: &mut [f64],
    re2: &mut [f64],
    im2: &mut [f64],
    indices: [usize; 4],
    unitaries: &[HolonomyTransferMatrix; 4],
) {
    let [i0, i1, i2, i3] = indices;

    // Load inputs into 4-lane SIMD registers
    let v_re0 = _mm256_set_pd(re0[i3], re0[i2], re0[i1], re0[i0]);
    let v_im0 = _mm256_set_pd(im0[i3], im0[i2], im0[i1], im0[i0]);
    let v_re1 = _mm256_set_pd(re1[i3], re1[i2], re1[i1], re1[i0]);
    let v_im1 = _mm256_set_pd(im1[i3], im1[i2], im1[i1], im1[i0]);
    let v_re2 = _mm256_set_pd(re2[i3], re2[i2], re2[i1], re2[i0]);
    let v_im2 = _mm256_set_pd(im2[i3], im2[i2], im2[i1], im2[i0]);

    let input_re = [v_re0, v_re1, v_re2];
    let input_im = [v_im0, v_im1, v_im2];

    // Compute row r in {0, 1, 2}
    let mut out_re = [_mm256_setzero_pd(); 3];
    let mut out_im = [_mm256_setzero_pd(); 3];

    for r in 0..3 {
        for k in 0..3 {
            // Load U_rk across 4 lanes
            let u_re = _mm256_set_pd(
                unitaries[3].elements[r][k].re,
                unitaries[2].elements[r][k].re,
                unitaries[1].elements[r][k].re,
                unitaries[0].elements[r][k].re,
            );
            let u_im = _mm256_set_pd(
                unitaries[3].elements[r][k].im,
                unitaries[2].elements[r][k].im,
                unitaries[1].elements[r][k].im,
                unitaries[0].elements[r][k].im,
            );

            // Re(c'_r) += u_re * in_re - u_im * in_im
            out_re[r] = _mm256_fmadd_pd(u_re, input_re[k], out_re[r]);
            out_re[r] = _mm256_fnmadd_pd(u_im, input_im[k], out_re[r]);

            // Im(c'_r) += u_re * in_im + u_im * in_re
            out_im[r] = _mm256_fmadd_pd(u_re, input_im[k], out_im[r]);
            out_im[r] = _mm256_fmadd_pd(u_im, input_re[k], out_im[r]);
        }
    }

    // Extract lanes and store back to SoA buffers
    let mut buf_re = [0.0f64; 4];
    let mut buf_im = [0.0f64; 4];

    _mm256_storeu_pd(buf_re.as_mut_ptr(), out_re[0]);
    _mm256_storeu_pd(buf_im.as_mut_ptr(), out_im[0]);
    re0[i0] = buf_re[0]; re0[i1] = buf_re[1]; re0[i2] = buf_re[2]; re0[i3] = buf_re[3];
    im0[i0] = buf_im[0]; im0[i1] = buf_im[1]; im0[i2] = buf_im[2]; im0[i3] = buf_im[3];

    _mm256_storeu_pd(buf_re.as_mut_ptr(), out_re[1]);
    _mm256_storeu_pd(buf_im.as_mut_ptr(), out_im[1]);
    re1[i0] = buf_re[0]; re1[i1] = buf_re[1]; re1[i2] = buf_re[2]; re1[i3] = buf_re[3];
    im1[i0] = buf_im[0]; im1[i1] = buf_im[1]; im1[i2] = buf_im[2]; im1[i3] = buf_im[3];

    _mm256_storeu_pd(buf_re.as_mut_ptr(), out_re[2]);
    _mm256_storeu_pd(buf_im.as_mut_ptr(), out_im[2]);
    re2[i0] = buf_re[0]; re2[i1] = buf_re[1]; re2[i2] = buf_re[2]; re2[i3] = buf_re[3];
    im2[i0] = buf_im[0]; im2[i1] = buf_im[1]; im2[i2] = buf_im[2]; im2[i3] = buf_im[3];
}
