// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! AVX2+FMA Ry/Rz kernels for qubits 0 and 1, where a butterfly pair sits inside one 256-bit
//! vector (or in adjacent ones) and the block-wise kernels cannot be used.
//!
//! Two vectors `v0 = [x0 x1 x2 x3]`, `v1 = [x4 x5 x6 x7]` are split into the elements whose
//! qubit bit is 0 and 1 with one unpack (qubit 0) or one 128-bit permute (qubit 1) each, rotated
//! as ordinary vectors, then interleaved back. Requires `dim >= 8`.

use super::unitary::gate_256;
use crate::compiler::Unitary2x2;
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn split<const Q: usize>(v0: __m256d, v1: __m256d) -> (__m256d, __m256d) {
    if Q == 0 {
        (_mm256_unpacklo_pd(v0, v1), _mm256_unpackhi_pd(v0, v1))
    } else {
        (_mm256_permute2f128_pd::<0x20>(v0, v1), _mm256_permute2f128_pd::<0x31>(v0, v1))
    }
}

#[inline(always)]
unsafe fn merge<const Q: usize>(lo: __m256d, hi: __m256d) -> (__m256d, __m256d) {
    // The unpack/permute pairs above are their own inverses on (lo, hi).
    split::<Q>(lo, hi)
}

/// Ry on qubit `Q` (0 or 1) of a statevector with `dim >= 8`.
#[target_feature(enable = "avx2", enable = "fma")]
pub(super) unsafe fn ry<const Q: usize>(real: &mut [f64], imag: &mut [f64], theta: f64) {
    let (s, c) = (theta * 0.5).sin_cos();
    let (vc, vs) = (_mm256_set1_pd(c), _mm256_set1_pd(s));
    for plane in [real.as_mut_ptr(), imag.as_mut_ptr()] {
        for k in (0..real.len()).step_by(8) {
            let (lo, hi) = split::<Q>(_mm256_loadu_pd(plane.add(k)), _mm256_loadu_pd(plane.add(k + 4)));
            let new_lo = _mm256_fmsub_pd(vc, lo, _mm256_mul_pd(vs, hi));
            let new_hi = _mm256_fmadd_pd(vs, lo, _mm256_mul_pd(vc, hi));
            let (n0, n1) = merge::<Q>(new_lo, new_hi);
            _mm256_storeu_pd(plane.add(k), n0);
            _mm256_storeu_pd(plane.add(k + 4), n1);
        }
    }
}

/// Rz on qubit `Q` (0 or 1) of a statevector with `dim >= 4`.
#[target_feature(enable = "avx2", enable = "fma")]
pub(super) unsafe fn rz<const Q: usize>(real: &mut [f64], imag: &mut [f64], theta: f64) {
    let (s_pos, c_pos) = (theta * 0.5).sin_cos();
    let (s_neg, c_neg) = (-theta * 0.5).sin_cos();
    // Lane j of a vector at offset k (a multiple of 4) holds index k + j, so bit Q is bit Q of j.
    let (vc, vs) = if Q == 0 {
        (_mm256_set_pd(c_pos, c_neg, c_pos, c_neg), _mm256_set_pd(s_pos, s_neg, s_pos, s_neg))
    } else {
        (_mm256_set_pd(c_pos, c_pos, c_neg, c_neg), _mm256_set_pd(s_pos, s_pos, s_neg, s_neg))
    };
    for k in (0..real.len()).step_by(4) {
        let r = _mm256_loadu_pd(real.as_ptr().add(k));
        let im = _mm256_loadu_pd(imag.as_ptr().add(k));
        _mm256_storeu_pd(real.as_mut_ptr().add(k), _mm256_fmsub_pd(r, vc, _mm256_mul_pd(im, vs)));
        _mm256_storeu_pd(imag.as_mut_ptr().add(k), _mm256_fmadd_pd(r, vs, _mm256_mul_pd(im, vc)));
    }
}

/// General complex 2x2 gate on qubit `Q` (0 or 1) of a block of `len` amplitudes, `len >= 8`.
///
/// # Safety
/// `re` and `im` must be valid for `len` reads and writes; the CPU must support AVX2 and FMA.
#[target_feature(enable = "avx2", enable = "fma")]
pub(super) unsafe fn u<const Q: usize>(re: *mut f64, im: *mut f64, len: usize, gate: &Unitary2x2) {
    for k in (0..len).step_by(8) {
        let (xr, yr) = split::<Q>(_mm256_loadu_pd(re.add(k)), _mm256_loadu_pd(re.add(k + 4)));
        let (xi, yi) = split::<Q>(_mm256_loadu_pd(im.add(k)), _mm256_loadu_pd(im.add(k + 4)));
        let (n0r, n0i, n1r, n1i) = gate_256(gate, xr, xi, yr, yi);
        let (a, b) = merge::<Q>(n0r, n1r);
        _mm256_storeu_pd(re.add(k), a);
        _mm256_storeu_pd(re.add(k + 4), b);
        let (a, b) = merge::<Q>(n0i, n1i);
        _mm256_storeu_pd(im.add(k), a);
        _mm256_storeu_pd(im.add(k + 4), b);
    }
}
