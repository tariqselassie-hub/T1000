// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Pair and block kernels for general complex 2x2 gates and real rotations, used by the fused
//! layer passes. Each entry point picks AVX-512F, AVX2+FMA or scalar code at run time.
//!
//! * [`u_pair`] / [`rot_pair`]: the two halves of a butterfly are separate contiguous runs.
//! * [`u_block`]: a gate on a low qubit of a cache-resident block, where the halves interleave
//!   within a vector (qubits 0 and 1 for AVX2, 0 to 2 for AVX-512) or are separate runs.

use crate::compiler::Unitary2x2;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// True when the 512-bit kernels should be used (AVX-512F present, not disabled).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn wide() -> bool {
    super::avx512::enabled()
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn avx2() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

/// `[x0; x1] <- U [x0; x1]` over `len` amplitudes of each plane.
///
/// # Safety
/// The four pointers must be valid for `len` reads and writes, and the ranges must not overlap.
pub(crate) unsafe fn u_pair(r0: *mut f64, r1: *mut f64, i0: *mut f64, i1: *mut f64, len: usize, u: &Unitary2x2) {
    #[cfg(target_arch = "x86_64")]
    {
        if len >= 8 && wide() {
            return u_pair_avx512(r0, r1, i0, i1, len, u);
        }
        if len >= 4 && avx2() {
            return u_pair_avx2(r0, r1, i0, i1, len, u);
        }
    }
    u_pair_scalar(r0, r1, i0, i1, 0, len, u);
}

/// Real rotation `[c -s; s c]` over `len` amplitudes of each plane.
///
/// # Safety
/// As for [`u_pair`].
pub(crate) unsafe fn rot_pair(r0: *mut f64, r1: *mut f64, i0: *mut f64, i1: *mut f64, len: usize, s: f64, c: f64) {
    #[cfg(target_arch = "x86_64")]
    {
        if len >= 8 && wide() {
            return rot_pair_avx512(r0, r1, i0, i1, len, s, c);
        }
        if len >= 4 && avx2() {
            return rot_pair_avx2(r0, r1, i0, i1, len, s, c);
        }
    }
    for (x0, x1) in [(r0, r1), (i0, i1)] {
        for k in 0..len {
            let (a, b) = (*x0.add(k), *x1.add(k));
            *x0.add(k) = c * a - s * b;
            *x1.add(k) = s * a + c * b;
        }
    }
}

/// Applies `u` to qubit `q` of a block of `re.len()` amplitudes (a power of two, at least 2).
pub(crate) fn u_block(re: &mut [f64], im: &mut [f64], q: usize, u: &Unitary2x2) {
    let len = re.len();
    // SAFETY: each branch checks the CPU features it needs and the minimum block length.
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if q < 3 && len >= 16 && wide() {
            return super::avx512::u_low(re.as_mut_ptr(), im.as_mut_ptr(), len, q, u);
        }
        if q < 2 && len >= 8 && avx2() {
            return if q == 0 {
                super::lowq::u::<0>(re.as_mut_ptr(), im.as_mut_ptr(), len, u)
            } else {
                super::lowq::u::<1>(re.as_mut_ptr(), im.as_mut_ptr(), len, u)
            };
        }
    }
    let half = 1usize << q;
    for (re_b, im_b) in re.chunks_mut(2 * half).zip(im.chunks_mut(2 * half)) {
        let (r0, r1) = re_b.split_at_mut(half);
        let (i0, i1) = im_b.split_at_mut(half);
        // SAFETY: the four slices are disjoint and each holds `half` elements.
        unsafe { u_pair(r0.as_mut_ptr(), r1.as_mut_ptr(), i0.as_mut_ptr(), i1.as_mut_ptr(), half, u) };
    }
}

pub(crate) unsafe fn u_pair_scalar(
    r0: *mut f64,
    r1: *mut f64,
    i0: *mut f64,
    i1: *mut f64,
    from: usize,
    len: usize,
    u: &Unitary2x2,
) {
    for k in from..len {
        let (ar, ai, br, bi) = (*r0.add(k), *i0.add(k), *r1.add(k), *i1.add(k));
        *r0.add(k) = u.u00_re * ar - u.u00_im * ai + u.u01_re * br - u.u01_im * bi;
        *i0.add(k) = u.u00_re * ai + u.u00_im * ar + u.u01_re * bi + u.u01_im * br;
        *r1.add(k) = u.u10_re * ar - u.u10_im * ai + u.u11_re * br - u.u11_im * bi;
        *i1.add(k) = u.u10_re * ai + u.u10_im * ar + u.u11_re * bi + u.u11_im * br;
    }
}

/// Broadcast coefficients of a 2x2 complex matrix for one vector width.
#[cfg(target_arch = "x86_64")]
macro_rules! complex_gate {
    ($name:ident, $vec:ty, $set1:ident, $mul:ident, $fmadd:ident, $fmsub:ident, $fnmadd:ident) => {
        /// `(new0_re, new0_im, new1_re, new1_im)` for `new0 = a x + b y`, `new1 = c x + d y`.
        #[inline(always)]
        unsafe fn $name(u: &Unitary2x2, xr: $vec, xi: $vec, yr: $vec, yi: $vec) -> ($vec, $vec, $vec, $vec) {
            let v = |x: f64| $set1(x);
            let n0r = $fmadd(v(u.u01_re), yr, $fnmadd(v(u.u01_im), yi, $fmsub(v(u.u00_re), xr, $mul(v(u.u00_im), xi))));
            let n0i = $fmadd(v(u.u01_re), yi, $fmadd(v(u.u01_im), yr, $fmadd(v(u.u00_re), xi, $mul(v(u.u00_im), xr))));
            let n1r = $fmadd(v(u.u11_re), yr, $fnmadd(v(u.u11_im), yi, $fmsub(v(u.u10_re), xr, $mul(v(u.u10_im), xi))));
            let n1i = $fmadd(v(u.u11_re), yi, $fmadd(v(u.u11_im), yr, $fmadd(v(u.u10_re), xi, $mul(v(u.u10_im), xr))));
            (n0r, n0i, n1r, n1i)
        }
    };
}

#[cfg(target_arch = "x86_64")]
complex_gate!(gate256, __m256d, _mm256_set1_pd, _mm256_mul_pd, _mm256_fmadd_pd, _mm256_fmsub_pd, _mm256_fnmadd_pd);
#[cfg(target_arch = "x86_64")]
complex_gate!(gate512, __m512d, _mm512_set1_pd, _mm512_mul_pd, _mm512_fmadd_pd, _mm512_fmsub_pd, _mm512_fnmadd_pd);

#[cfg(target_arch = "x86_64")]
/// The 256-bit complex gate on vectors, for the interleaved low-qubit kernels in `lowq`.
#[inline(always)]
pub(super) unsafe fn gate_256(u: &Unitary2x2, xr: __m256d, xi: __m256d, yr: __m256d, yi: __m256d) -> (__m256d, __m256d, __m256d, __m256d) {
    gate256(u, xr, xi, yr, yi)
}

#[cfg(target_arch = "x86_64")]
/// The 512-bit complex gate on vectors, for the interleaved low-qubit kernels in `avx512`.
#[inline(always)]
pub(super) unsafe fn gate_512(u: &Unitary2x2, xr: __m512d, xi: __m512d, yr: __m512d, yi: __m512d) -> (__m512d, __m512d, __m512d, __m512d) {
    gate512(u, xr, xi, yr, yi)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn u_pair_avx2(r0: *mut f64, r1: *mut f64, i0: *mut f64, i1: *mut f64, len: usize, u: &Unitary2x2) {
    let vec_len = len - len % 4;
    for k in (0..vec_len).step_by(4) {
        let ld = |p: *mut f64| _mm256_loadu_pd(p.add(k));
        let (n0r, n0i, n1r, n1i) = gate256(u, ld(r0), ld(i0), ld(r1), ld(i1));
        _mm256_storeu_pd(r0.add(k), n0r);
        _mm256_storeu_pd(i0.add(k), n0i);
        _mm256_storeu_pd(r1.add(k), n1r);
        _mm256_storeu_pd(i1.add(k), n1i);
    }
    u_pair_scalar(r0, r1, i0, i1, vec_len, len, u);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn u_pair_avx512(r0: *mut f64, r1: *mut f64, i0: *mut f64, i1: *mut f64, len: usize, u: &Unitary2x2) {
    let vec_len = len - len % 8;
    for k in (0..vec_len).step_by(8) {
        let ld = |p: *mut f64| _mm512_loadu_pd(p.add(k));
        let (n0r, n0i, n1r, n1i) = gate512(u, ld(r0), ld(i0), ld(r1), ld(i1));
        _mm512_storeu_pd(r0.add(k), n0r);
        _mm512_storeu_pd(i0.add(k), n0i);
        _mm512_storeu_pd(r1.add(k), n1r);
        _mm512_storeu_pd(i1.add(k), n1i);
    }
    u_pair_scalar(r0, r1, i0, i1, vec_len, len, u);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn rot_pair_avx2(r0: *mut f64, r1: *mut f64, i0: *mut f64, i1: *mut f64, len: usize, s: f64, c: f64) {
    let (vc, vs) = (_mm256_set1_pd(c), _mm256_set1_pd(s));
    let vec_len = len - len % 4;
    for (x0, x1) in [(r0, r1), (i0, i1)] {
        for k in (0..vec_len).step_by(4) {
            let (a, b) = (_mm256_loadu_pd(x0.add(k)), _mm256_loadu_pd(x1.add(k)));
            _mm256_storeu_pd(x0.add(k), _mm256_fmsub_pd(vc, a, _mm256_mul_pd(vs, b)));
            _mm256_storeu_pd(x1.add(k), _mm256_fmadd_pd(vs, a, _mm256_mul_pd(vc, b)));
        }
        for k in vec_len..len {
            let (a, b) = (*x0.add(k), *x1.add(k));
            *x0.add(k) = c * a - s * b;
            *x1.add(k) = s * a + c * b;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn rot_pair_avx512(r0: *mut f64, r1: *mut f64, i0: *mut f64, i1: *mut f64, len: usize, s: f64, c: f64) {
    let (vc, vs) = (_mm512_set1_pd(c), _mm512_set1_pd(s));
    let vec_len = len - len % 8;
    for (x0, x1) in [(r0, r1), (i0, i1)] {
        for k in (0..vec_len).step_by(8) {
            let (a, b) = (_mm512_loadu_pd(x0.add(k)), _mm512_loadu_pd(x1.add(k)));
            _mm512_storeu_pd(x0.add(k), _mm512_fmsub_pd(vc, a, _mm512_mul_pd(vs, b)));
            _mm512_storeu_pd(x1.add(k), _mm512_fmadd_pd(vs, a, _mm512_mul_pd(vc, b)));
        }
        for k in vec_len..len {
            let (a, b) = (*x0.add(k), *x1.add(k));
            *x0.add(k) = c * a - s * b;
            *x1.add(k) = s * a + c * b;
        }
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    fn gate() -> Unitary2x2 {
        // A non-trivial complex matrix (not required to be unitary for kernel equivalence).
        Unitary2x2 {
            u00_re: 0.6, u00_im: -0.3, u01_re: 0.2, u01_im: 0.5,
            u10_re: -0.4, u10_im: 0.7, u11_re: 0.9, u11_im: 0.1,
        }
    }

    fn planes(len: usize) -> (Vec<f64>, Vec<f64>) {
        (
            (0..len).map(|i| (i as f64 * 0.37).sin() + 0.1).collect(),
            (0..len).map(|i| (i as f64 * 0.91).cos() - 0.2).collect(),
        )
    }

    /// Reference: the gate applied pair by pair in scalar code.
    fn reference(re: &mut [f64], im: &mut [f64], q: usize, u: &Unitary2x2) {
        let half = 1usize << q;
        for base in (0..re.len()).step_by(2 * half) {
            for k in base..base + half {
                let (r, i) = (re.as_mut_ptr(), im.as_mut_ptr());
                unsafe { u_pair_scalar(r.add(k), r.add(k + half), i.add(k), i.add(k + half), 0, 1, u) };
            }
        }
    }

    fn dev(a: &(Vec<f64>, Vec<f64>), b: &(Vec<f64>, Vec<f64>)) -> f64 {
        let d = |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max);
        d(&a.0, &b.0).max(d(&a.1, &b.1))
    }

    #[test]
    fn u_block_matches_reference_on_the_active_path() {
        for n in 1..=12 {
            for q in 0..n {
                let (mut got, mut want) = (planes(1 << n), planes(1 << n));
                u_block(&mut got.0, &mut got.1, q, &gate());
                reference(&mut want.0, &mut want.1, q, &gate());
                assert!(dev(&got, &want) < 1e-13, "n={n} q={q}");
            }
        }
    }

    #[test]
    fn each_simd_variant_matches_reference() {
        for n in 4..=12 {
            let len = 1usize << n;
            for q in 0..n {
                let mut want = planes(len);
                reference(&mut want.0, &mut want.1, q, &gate());
                if avx2() && q < 2 {
                    let mut got = planes(len);
                    unsafe {
                        if q == 0 {
                            super::super::lowq::u::<0>(got.0.as_mut_ptr(), got.1.as_mut_ptr(), len, &gate());
                        } else {
                            super::super::lowq::u::<1>(got.0.as_mut_ptr(), got.1.as_mut_ptr(), len, &gate());
                        }
                    }
                    assert!(dev(&got, &want) < 1e-13, "avx2 low n={n} q={q}");
                }
                if is_x86_feature_detected!("avx512f") && q < 3 {
                    let mut got = planes(len);
                    unsafe { super::super::avx512::u_low(got.0.as_mut_ptr(), got.1.as_mut_ptr(), len, q, &gate()) };
                    assert!(dev(&got, &want) < 1e-13, "avx512 low n={n} q={q}");
                }
            }
        }
    }

    #[test]
    fn pair_kernels_match_scalar_including_ragged_tails() {
        for len in [1usize, 3, 4, 7, 8, 9, 15, 16, 17, 100, 512] {
            let (mut want_a, mut want_b) = (planes(len), planes(len));
            want_b.0.iter_mut().for_each(|v| *v += 0.5);
            let run = |f: unsafe fn(*mut f64, *mut f64, *mut f64, *mut f64, usize, &Unitary2x2)| {
                let (mut a, mut b) = (planes(len), planes(len));
                b.0.iter_mut().for_each(|v| *v += 0.5);
                unsafe { f(a.0.as_mut_ptr(), b.0.as_mut_ptr(), a.1.as_mut_ptr(), b.1.as_mut_ptr(), len, &gate()) };
                (a, b)
            };
            unsafe {
                u_pair_scalar(want_a.0.as_mut_ptr(), want_b.0.as_mut_ptr(), want_a.1.as_mut_ptr(), want_b.1.as_mut_ptr(), 0, len, &gate());
            }
            let mut variants: Vec<(&str, unsafe fn(*mut f64, *mut f64, *mut f64, *mut f64, usize, &Unitary2x2))> = vec![("dispatch", u_pair)];
            if avx2() {
                variants.push(("avx2", u_pair_avx2));
            }
            if is_x86_feature_detected!("avx512f") {
                variants.push(("avx512", u_pair_avx512));
            }
            for (name, f) in variants {
                let (a, b) = run(f);
                assert!(dev(&a, &want_a).max(dev(&b, &want_b)) < 1e-13, "{name} len={len}");
            }
        }
    }
}
