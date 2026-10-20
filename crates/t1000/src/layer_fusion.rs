// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Cache-blocked Ry layer: one statevector pass per group of qubits instead of one per gate.
//!
//! Ry rotations on different qubits commute, so a layer's rotations can be regrouped freely.
//! Once the state is larger than the last-level cache every gate pass is DRAM-bound, so the
//! number of passes is the cost. This module applies
//!
//! * qubits `0..LOW_BITS` together, block by block, on `2^LOW_BITS`-amplitude blocks that stay
//!   in L2 while all of those gates run, and
//! * the remaining qubits in groups of up to `GROUP_QUBITS`: each group visits `2^g` chunks of
//!   `CHUNK` amplitudes (one per combination of the group's qubit bits) and runs the `g`
//!   butterfly stages on those cache-resident chunks before moving on.
//!
//! A 26-qubit layer drops from 26 DRAM passes to 1 + 4. Per-element arithmetic is the same as
//! the per-gate kernels, so results agree to rounding.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use rayon::prelude::*;

/// Qubits applied together on one in-cache block (`2^12` amplitudes = 64 KB of re + im).
const LOW_BITS: usize = 12;
/// Chunk length (amplitudes) streamed per group member; must be a multiple of 4.
const CHUNK_BITS: usize = 9;
const CHUNK: usize = 1 << CHUNK_BITS;
/// Maximum high qubits fused into one pass (`2^4` chunks resident).
const GROUP_QUBITS: usize = 4;

/// Raw statevector base pointers shared across worker threads.
///
/// Each worker only touches the chunks of its own group of bases, which never overlap.
#[derive(Clone, Copy)]
struct Planes {
    re: *mut f64,
    im: *mut f64,
}

// SAFETY: workers write disjoint index sets (see `apply_group`); the pointers outlive the call.
unsafe impl Send for Planes {}
unsafe impl Sync for Planes {}

/// Applies `Ry(angles[q])` to every qubit `q < angles.len()` of the statevector.
pub fn apply_ry_layer(real: &mut [f64], imag: &mut [f64], angles: &[f64]) {
    let n_qubits = crate::simd::assert_statevector(real, imag);
    assert!(
        angles.len() <= n_qubits,
        "{} angles for {} qubits",
        angles.len(),
        n_qubits
    );
    let parallel = crate::parallel::use_parallel(real.len());

    let low = angles.len().min(LOW_BITS);
    apply_low_block(real, imag, &angles[..low], parallel);

    let mut q = LOW_BITS;
    while q < angles.len() {
        let end = angles.len().min(q + GROUP_QUBITS);
        apply_high_group(real, imag, q, &angles[q..end], parallel);
        q = end;
    }
}

/// Qubits `0..angles.len()` (all `< LOW_BITS`) on each `2^LOW_BITS` block in turn.
fn apply_low_block(real: &mut [f64], imag: &mut [f64], angles: &[f64], parallel: bool) {
    if angles.is_empty() {
        return;
    }
    let block = real.len().min(1 << LOW_BITS);
    let run = |re: &mut [f64], im: &mut [f64]| {
        for (q, &theta) in angles.iter().enumerate() {
            crate::simd::apply_ry_kernel(re, im, q, theta);
        }
    };
    if parallel {
        crate::parallel::install(|| {
            real.par_chunks_mut(block)
                .zip(imag.par_chunks_mut(block))
                .for_each(|(re, im)| run(re, im));
        });
    } else {
        real.chunks_mut(block)
            .zip(imag.chunks_mut(block))
            .for_each(|(re, im)| run(re, im));
    }
}

/// Qubits `first..first + angles.len()` (all `>= CHUNK_BITS`) in a single pass over the state.
fn apply_high_group(real: &mut [f64], imag: &mut [f64], first: usize, angles: &[f64], par: bool) {
    let g = angles.len();
    let bases = real.len() >> (CHUNK_BITS + g);
    let planes = Planes {
        re: real.as_mut_ptr(),
        im: imag.as_mut_ptr(),
    };
    let rot: [(f64, f64); GROUP_QUBITS] =
        std::array::from_fn(|j| angles.get(j).map_or((0.0, 1.0), |t| (t * 0.5).sin_cos()));
    let run = |b: usize| apply_group(planes, b, first, g, &rot);
    if par {
        crate::parallel::install(|| (0..bases).into_par_iter().for_each(run));
    } else {
        (0..bases).for_each(run);
    }
}

/// Inserts a zero bit at final bit position `p` of `x`.
#[inline(always)]
fn insert_zero_bit(x: usize, p: usize) -> usize {
    let low = x & ((1usize << p) - 1);
    ((x ^ low) << 1) | low
}

/// Runs the `g` butterfly stages on the `2^g` chunks addressed by base index `b`.
fn apply_group(p: Planes, b: usize, first: usize, g: usize, rot: &[(f64, f64); GROUP_QUBITS]) {
    let mut x = b;
    for j in 0..g {
        x = insert_zero_bit(x, first + j - CHUNK_BITS);
    }
    let base = x << CHUNK_BITS;
    for (j, &(s, c)) in rot.iter().enumerate().take(g) {
        let bit = 1usize << (first + j);
        for mask in 0..(1usize << g) {
            if mask & (1 << j) != 0 {
                continue;
            }
            let off: usize = (0..g)
                .filter(|k| mask & (1 << k) != 0)
                .map(|k| 1usize << (first + k))
                .sum();
            // SAFETY: `base + off` and `base + off + bit` are distinct chunks of this group,
            // inside the statevector, and no other base touches them.
            unsafe {
                ry_pair(
                    p.re.add(base + off),
                    p.re.add(base + off + bit),
                    p.im.add(base + off),
                    p.im.add(base + off + bit),
                    c,
                    s,
                );
            }
        }
    }
}

/// `[x0; x1] <- [c -s; s c] [x0; x1]` over `CHUNK` elements of each plane.
///
/// # Safety
/// All four pointers must be valid for `CHUNK` reads/writes and the ranges must not overlap.
#[inline(always)]
unsafe fn ry_pair(r0: *mut f64, r1: *mut f64, i0: *mut f64, i1: *mut f64, c: f64, s: f64) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return ry_pair_avx2(r0, r1, i0, i1, c, s);
    }
    for k in 0..CHUNK {
        for (x0, x1) in [(r0, r1), (i0, i1)] {
            let (a, b) = (*x0.add(k), *x1.add(k));
            *x0.add(k) = c * a - s * b;
            *x1.add(k) = s * a + c * b;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn ry_pair_avx2(r0: *mut f64, r1: *mut f64, i0: *mut f64, i1: *mut f64, c: f64, s: f64) {
    let (vc, vs) = (_mm256_set1_pd(c), _mm256_set1_pd(s));
    for (x0, x1) in [(r0, r1), (i0, i1)] {
        for k in (0..CHUNK).step_by(4) {
            let a = _mm256_loadu_pd(x0.add(k));
            let b = _mm256_loadu_pd(x1.add(k));
            _mm256_storeu_pd(x0.add(k), _mm256_fmsub_pd(vc, a, _mm256_mul_pd(vs, b)));
            _mm256_storeu_pd(x1.add(k), _mm256_fmadd_pd(vs, a, _mm256_mul_pd(vc, b)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_state(dim: usize) -> (Vec<f64>, Vec<f64>) {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        (
            (0..dim).map(|_| next()).collect(),
            (0..dim).map(|_| next()).collect(),
        )
    }

    fn check(n: usize) {
        let (mut re, mut im) = pseudo_state(1 << n);
        let (mut re_ref, mut im_ref) = (re.clone(), im.clone());
        let angles: Vec<f64> = (0..n).map(|q| 0.17 + 0.31 * q as f64).collect();
        apply_ry_layer(&mut re, &mut im, &angles);
        for (q, &t) in angles.iter().enumerate() {
            crate::simd::apply_ry_kernel(&mut re_ref, &mut im_ref, q, t);
        }
        let err = re
            .iter()
            .zip(&re_ref)
            .chain(im.iter().zip(&im_ref))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-12, "n={n}: max deviation {err}");
    }

    #[test]
    fn fused_layer_matches_per_gate_kernels() {
        for n in [4, 12, 13, 16, 17, 19, 20] {
            check(n);
        }
    }

    #[test]
    fn partial_angle_list_is_supported() {
        let (mut re, mut im) = pseudo_state(1 << 15);
        let (mut re_ref, mut im_ref) = (re.clone(), im.clone());
        let angles = [
            0.4, -1.1, 0.9, 2.2, 0.3, 0.8, 1.4, -0.6, 0.5, 0.2, 1.1, 0.7, -0.9, 0.35,
        ];
        apply_ry_layer(&mut re, &mut im, &angles);
        for (q, &t) in angles.iter().enumerate() {
            crate::simd::apply_ry_kernel(&mut re_ref, &mut im_ref, q, t);
        }
        let err = re
            .iter()
            .zip(&re_ref)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-12, "max deviation {err}");
    }
}
