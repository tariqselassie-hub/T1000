// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! AVX-512F Ry/Rz kernels (8 x f64 per vector) for cache-resident statevectors.
//!
//! Used when the CPU supports AVX-512F and the state is at most [`MAX_DIM`] amplitudes. Larger
//! states are DRAM-bound, where the AVX2 kernels (with streaming stores) are kept. Set
//! `T1000_SIMD=avx2` to force the 256-bit kernels, for example to compare the two.
//!
//! Qubits 0 to 2 pair elements inside a vector: two vectors are split into the elements with the
//! qubit bit 0 and 1 by `vpermt2pd`, rotated, and merged back with a second permute.

use super::unitary::gate_512;
use crate::compiler::Unitary2x2;
use std::arch::x86_64::*;
use std::sync::OnceLock;

/// Largest statevector (amplitudes) handled here: 2^21 amplitudes = 32 MB of re + im.
pub(crate) const MAX_DIM: usize = 1 << 21;

/// True when AVX-512F is available and not disabled by `T1000_SIMD=avx2`.
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let forced_off = std::env::var("T1000_SIMD").is_ok_and(|v| v.eq_ignore_ascii_case("avx2"));
        !forced_off && is_x86_feature_detected!("avx512f")
    })
}

/// True when the AVX-512 kernels should run for a state of `dim` amplitudes.
#[inline(always)]
pub(crate) fn use_for(dim: usize) -> bool {
    (16..=MAX_DIM).contains(&dim) && enabled()
}

/// Element picks (0-7 from the first vector, 8-15 from the second) for qubits 0, 1, 2.
const SPLIT_LO: [[i64; 8]; 3] = [
    [0, 2, 4, 6, 8, 10, 12, 14],
    [0, 1, 4, 5, 8, 9, 12, 13],
    [0, 1, 2, 3, 8, 9, 10, 11],
];
const SPLIT_HI: [[i64; 8]; 3] = [
    [1, 3, 5, 7, 9, 11, 13, 15],
    [2, 3, 6, 7, 10, 11, 14, 15],
    [4, 5, 6, 7, 12, 13, 14, 15],
];
const MERGE_A: [[i64; 8]; 3] = [
    [0, 8, 1, 9, 2, 10, 3, 11],
    [0, 1, 8, 9, 2, 3, 10, 11],
    [0, 1, 2, 3, 8, 9, 10, 11],
];
const MERGE_B: [[i64; 8]; 3] = [
    [4, 12, 5, 13, 6, 14, 7, 15],
    [4, 5, 12, 13, 6, 7, 14, 15],
    [4, 5, 6, 7, 12, 13, 14, 15],
];

#[target_feature(enable = "avx512f")]
unsafe fn idx(table: &[i64; 8]) -> __m512i {
    _mm512_loadu_si512(table.as_ptr().cast())
}

/// Ry on any qubit of a statevector with `dim >= 16`.
///
/// # Safety
/// The CPU must support AVX-512F.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn apply_ry(real: &mut [f64], imag: &mut [f64], qubit: usize, theta: f64) {
    let (s, c) = (theta * 0.5).sin_cos();
    let (vc, vs) = (_mm512_set1_pd(c), _mm512_set1_pd(s));
    if qubit < 3 {
        let (lo_i, hi_i) = (idx(&SPLIT_LO[qubit]), idx(&SPLIT_HI[qubit]));
        let (a_i, b_i) = (idx(&MERGE_A[qubit]), idx(&MERGE_B[qubit]));
        for plane in [real.as_mut_ptr(), imag.as_mut_ptr()] {
            for k in (0..real.len()).step_by(16) {
                let (v0, v1) = (
                    _mm512_loadu_pd(plane.add(k)),
                    _mm512_loadu_pd(plane.add(k + 8)),
                );
                let lo = _mm512_permutex2var_pd(v0, lo_i, v1);
                let hi = _mm512_permutex2var_pd(v0, hi_i, v1);
                let new_lo = _mm512_fmsub_pd(vc, lo, _mm512_mul_pd(vs, hi));
                let new_hi = _mm512_fmadd_pd(vs, lo, _mm512_mul_pd(vc, hi));
                _mm512_storeu_pd(plane.add(k), _mm512_permutex2var_pd(new_lo, a_i, new_hi));
                _mm512_storeu_pd(
                    plane.add(k + 8),
                    _mm512_permutex2var_pd(new_lo, b_i, new_hi),
                );
            }
        }
        return;
    }
    let half = 1usize << qubit;
    for plane in [real.as_mut_ptr(), imag.as_mut_ptr()] {
        for base in (0..real.len()).step_by(2 * half) {
            for k in (base..base + half).step_by(8) {
                let (x0, x1) = (
                    _mm512_loadu_pd(plane.add(k)),
                    _mm512_loadu_pd(plane.add(k + half)),
                );
                _mm512_storeu_pd(plane.add(k), _mm512_fmsub_pd(vc, x0, _mm512_mul_pd(vs, x1)));
                _mm512_storeu_pd(
                    plane.add(k + half),
                    _mm512_fmadd_pd(vs, x0, _mm512_mul_pd(vc, x1)),
                );
            }
        }
    }
}

/// Rz on any qubit of a statevector with `dim >= 16`: |0> gets exp(-i theta/2), |1> exp(+i theta/2).
///
/// # Safety
/// The CPU must support AVX-512F.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn apply_rz(real: &mut [f64], imag: &mut [f64], qubit: usize, theta: f64) {
    let (s_pos, c_pos) = (theta * 0.5).sin_cos();
    let (s_neg, c_neg) = (-theta * 0.5).sin_cos();
    let rotate = |plane_r: *mut f64, plane_i: *mut f64, k: usize, vc: __m512d, vs: __m512d| {
        let (r, im) = (
            _mm512_loadu_pd(plane_r.add(k)),
            _mm512_loadu_pd(plane_i.add(k)),
        );
        _mm512_storeu_pd(
            plane_r.add(k),
            _mm512_fmsub_pd(r, vc, _mm512_mul_pd(im, vs)),
        );
        _mm512_storeu_pd(
            plane_i.add(k),
            _mm512_fmadd_pd(r, vs, _mm512_mul_pd(im, vc)),
        );
    };
    let (pr, pi) = (real.as_mut_ptr(), imag.as_mut_ptr());
    if qubit < 3 {
        // Bit `qubit` of lane j is bit `qubit` of the index, since vectors start at multiples of 8.
        let lane = |neg: f64, pos: f64| {
            let v: [f64; 8] =
                std::array::from_fn(|j| if (j >> qubit) & 1 == 1 { pos } else { neg });
            _mm512_loadu_pd(v.as_ptr())
        };
        let (vc, vs) = (lane(c_neg, c_pos), lane(s_neg, s_pos));
        for k in (0..real.len()).step_by(8) {
            rotate(pr, pi, k, vc, vs);
        }
        return;
    }
    let half = 1usize << qubit;
    let (c0, s0, c1, s1) = (
        _mm512_set1_pd(c_neg),
        _mm512_set1_pd(s_neg),
        _mm512_set1_pd(c_pos),
        _mm512_set1_pd(s_pos),
    );
    for base in (0..real.len()).step_by(2 * half) {
        for k in (base..base + half).step_by(8) {
            rotate(pr, pi, k, c0, s0);
            rotate(pr, pi, k + half, c1, s1);
        }
    }
}

/// General complex 2x2 gate on qubit `qubit` (0 to 2) of a block of `len >= 16` amplitudes.
///
/// # Safety
/// `re` and `im` must be valid for `len` reads and writes; the CPU must support AVX-512F.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn u_low(
    re: *mut f64,
    im: *mut f64,
    len: usize,
    qubit: usize,
    gate: &Unitary2x2,
) {
    let (lo_i, hi_i) = (idx(&SPLIT_LO[qubit]), idx(&SPLIT_HI[qubit]));
    let (a_i, b_i) = (idx(&MERGE_A[qubit]), idx(&MERGE_B[qubit]));
    let split = |p: *mut f64, k: usize| {
        let (v0, v1) = (_mm512_loadu_pd(p.add(k)), _mm512_loadu_pd(p.add(k + 8)));
        (
            _mm512_permutex2var_pd(v0, lo_i, v1),
            _mm512_permutex2var_pd(v0, hi_i, v1),
        )
    };
    for k in (0..len).step_by(16) {
        let ((xr, yr), (xi, yi)) = (split(re, k), split(im, k));
        let (n0r, n0i, n1r, n1i) = gate_512(gate, xr, xi, yr, yi);
        _mm512_storeu_pd(re.add(k), _mm512_permutex2var_pd(n0r, a_i, n1r));
        _mm512_storeu_pd(re.add(k + 8), _mm512_permutex2var_pd(n0r, b_i, n1r));
        _mm512_storeu_pd(im.add(k), _mm512_permutex2var_pd(n0i, a_i, n1i));
        _mm512_storeu_pd(im.add(k + 8), _mm512_permutex2var_pd(n0i, b_i, n1i));
    }
}
