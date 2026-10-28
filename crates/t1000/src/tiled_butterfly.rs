// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Tiled kernels for large statevectors.
//!
//! Three code paths, all verified amplitude-by-amplitude against the reference simulator in
//! `tests/oracle.rs`:
//!
//! - **`apply_layer_fused`** (no longer used by `CircuitAnsatz`, which now applies Ry gates and
//!   then one fused diagonal pass from `crate::diagonal`; kept as a public kernel):
//!   applies the Ry + Rz pairs for qubits 0..=8 in one pass over 512-amplitude tiles (scalar
//!   inner loops), then applies each remaining qubit's Ry and Rz as separate passes. For a
//!   20-qubit layer that is 1 + 2 * 11 = 23 rotation passes instead of 40, not a single pass.
//! - **`tiled_apply_ry` / `tiled_apply_rz`** for qubit index >= 16: process each butterfly
//!   half in 512-element chunks. The Ry path copies chunks into an interleaved (re, im) stack
//!   buffer and back; this moves the same bytes as the plain SoA AVX2 kernel, so it does not
//!   reduce DRAM traffic.
//! - **`tiled_apply_cz`** for high qubit pairs: a scalar tile scan that skips tiles with no
//!   index having both bits set.
//!
//! Performance notes: the plain AVX2 kernels in `simd` already stream each half-block
//! contiguously, so large strides do not cause the cache-miss pattern these kernels were
//! written to avoid. No measured speedup of the tiled paths over the plain kernels has been
//! shown. The large end-to-end gains came from fusing each layer's diagonal gates into one pass
//! (`crate::diagonal`) and from multithreading (`crate::parallel`); see `frontier_rcs`.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Tile width in f64 elements for cache-oblivious blocking.
/// 512 × f64 = 4 KB → fits 8 tiles in L1 (32 KB). Tunable.
const W_TILE: usize = 512;

/// Threshold qubit index above which to use interleaved AoS layout for butterflies.
/// Below this, stride < 2^INTERLEAVE_THRESHOLD × 8 bytes (power-of-2 stride ≤ 1 MB for threshold=17).
const INTERLEAVE_THRESHOLD: usize = 16;

/// Number of f64 elements to prefetch ahead (2 × 64-byte cache lines = 16 f64s).
const PREFETCH_DIST: usize = 16;

/// Minimum qubit count at which the layer-fused tiled path activates.
/// Below this, the statevector fits comfortably in L3 and the standard per-gate
/// AVX2 path is already optimal. Above this (≥ 17 qubits → 1 MB+), the tiled
/// fused pass reduces DRAM sweeps from 2N to 1 per layer.
pub const TILED_LAYER_THRESHOLD: usize = 17;

// =============================================================================
// PUBLIC ENTRY POINTS (mirroring simd.rs API, drop-in replacements)
// =============================================================================

/// Cache-oblivious tiled single-qubit Ry gate.
///
/// For `qubit < INTERLEAVE_THRESHOLD`: uses existing AVX2 butterfly (stride fits in L2).
/// For `qubit >= INTERLEAVE_THRESHOLD`: uses tiled cache-oblivious butterfly with AoS repacking.
#[inline(always)]
pub fn tiled_apply_ry(real: &mut [f64], imag: &mut [f64], qubit: usize, theta: f64) {
    let n_qubits = crate::simd::assert_statevector(real, imag);
    crate::simd::assert_qubit(qubit, n_qubits);
    if qubit < INTERLEAVE_THRESHOLD {
        crate::simd::apply_ry_kernel(real, imag, qubit, theta);
    } else {
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma"
        ))]
        unsafe {
            apply_ry_tiled_interleaved(real, imag, qubit, theta);
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
                    apply_ry_tiled_interleaved(real, imag, qubit, theta);
                    return;
                }
            }
            apply_ry_tiled_scalar(real, imag, qubit, theta);
        }
    }
}

/// Cache-oblivious tiled single-qubit Rz gate.
#[inline(always)]
pub fn tiled_apply_rz(real: &mut [f64], imag: &mut [f64], qubit: usize, theta: f64) {
    let n_qubits = crate::simd::assert_statevector(real, imag);
    crate::simd::assert_qubit(qubit, n_qubits);
    if qubit < INTERLEAVE_THRESHOLD {
        crate::simd::apply_rz_kernel(real, imag, qubit, theta);
    } else {
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma"
        ))]
        unsafe {
            apply_rz_tiled_interleaved(real, imag, qubit, theta);
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
                    apply_rz_tiled_interleaved(real, imag, qubit, theta);
                    return;
                }
            }
            apply_rz_tiled_scalar(real, imag, qubit, theta);
        }
    }
}

/// Layer-fused tiled gate execution for all Ry + Rz rotations in a circuit layer.
///
/// **Key insight**: Single-qubit Ry and Rz gates acting on different qubits commute, but their
/// butterfly pairs (j0, j1 = j0 + 2^q) span stride-2^q apart. For small q the stride is short
/// (j0 and j1 fit in the same tile), enabling true butterfly batching within a W_TILE window.
/// For large q (high-qubit gates) the stride exceeds the tile, so we fall back to the tiled
/// AoS interleaved path which applies per-gate W_TILE blocking across each butterfly half.
///
/// **Phase 1 (q < log2(W_TILE))**: butterfly pairs (j0, j1) ALWAYS fit within one W_TILE chunk.
///   → Single tiled streaming pass handles ALL these gates, visiting each amplitude once.
///   → Memory bandwidth: 1 × 16 MB instead of 2Q × 16 MB.
///
/// **Phase 2 (q >= log2(W_TILE), q < INTERLEAVE_THRESHOLD)**: stride exceeds W_TILE.
///   → Apply each gate's two butterfly halves independently with W_TILE tiling + prefetch.
///   → Each half is a contiguous stream → good hardware prefetcher hit rate.
///
/// **Phase 3 (q >= INTERLEAVE_THRESHOLD)**: stride exceeds L2 → AoS interleaved kernel.
///   → Repack 2×W_TILE window into AoS, apply butterfly, write back → halves DRAM fetches.
///
/// `theta_ry[q]` and `theta_rz[q]` are the rotation angles for qubit `q`.
pub fn apply_layer_fused(real: &mut [f64], imag: &mut [f64], theta_ry: &[f64], theta_rz: &[f64]) {
    let state_qubits = crate::simd::assert_statevector(real, imag);
    assert_eq!(
        theta_ry.len(),
        theta_rz.len(),
        "Ry/Rz angle lists must have equal length"
    );
    let n_qubits = theta_ry.len();
    assert!(
        n_qubits <= state_qubits && n_qubits <= 32,
        "layer has {} rotation pairs but the state has {} qubits (max 32)",
        n_qubits,
        state_qubits
    );
    let dim = real.len();

    // Precompute (sin, cos) for all Ry and Rz gates
    let mut ry_cos = [0.0f64; 32];
    let mut ry_sin = [0.0f64; 32];
    let mut rz_cn = [0.0f64; 32]; // cos(-θ/2) for |0⟩
    let mut rz_sn = [0.0f64; 32]; // sin(-θ/2) for |0⟩
    let mut rz_cp = [0.0f64; 32]; // cos(+θ/2) for |1⟩
    let mut rz_sp = [0.0f64; 32]; // sin(+θ/2) for |1⟩

    for q in 0..n_qubits.min(32) {
        let (sy, cy) = (theta_ry[q] * 0.5).sin_cos();
        ry_cos[q] = cy;
        ry_sin[q] = sy;

        let (sn, cn) = (-theta_rz[q] * 0.5).sin_cos();
        let (sp, cp) = (theta_rz[q] * 0.5).sin_cos();
        rz_cn[q] = cn;
        rz_sn[q] = sn;
        rz_cp[q] = cp;
        rz_sp[q] = sp;
    }

    // Phase 1: low-stride gates (block <= W_TILE/2) → pairs always fit within one tile.
    //   Both j0 and j1 = j0+block are in [tile_base, tile_base + 2*block) ⊆ W_TILE window.
    //   We process in strides of 2*block, visiting each amplitude exactly once per gate,
    //   with ALL gates applied to the tile before moving to the next.
    let max_inline_qubit = usize::BITS as usize - W_TILE.leading_zeros() as usize - 1; // log2(W_TILE)
    let inline_n = n_qubits.min(max_inline_qubit).min(INTERLEAVE_THRESHOLD);

    if inline_n > 0 {
        // Stride for the outermost inline gate determines the base stepping size
        // We process ALL inline qubits' butterfly pairs together in chunks of W_TILE
        let mut tile_base = 0;
        while tile_base < dim {
            let tile_end = (tile_base + W_TILE).min(dim);

            // Apply each inline-qubit Ry within this W_TILE window
            for q in 0..inline_n {
                let block = 1usize << q;
                let stride = block << 1;
                let cy = ry_cos[q];
                let sy = ry_sin[q];

                // Iterate over all butterfly bases that fall within [tile_base, tile_end)
                let first_base = (tile_base / stride) * stride;
                let mut base = first_base;
                while base < tile_end {
                    // The "low half" of this stride is [base, base+block)
                    // Only process indices where bit_q == 0 (j0) and j1 = j0+block
                    let j0_start = base.max(tile_base);
                    let j0_end = (base + block).min(tile_end);

                    // j1 = j0 + block; for inline qubits block <= W_TILE/2, so j1 can be
                    // outside [tile_base, tile_end). We apply the butterfly unconditionally.
                    let mut k = j0_start;
                    while k < j0_end {
                        let j0 = k;
                        let j1 = k + block;
                        if j1 < dim {
                            let r0 = real[j0];
                            let r1 = real[j1];
                            let i0 = imag[j0];
                            let i1 = imag[j1];
                            real[j0] = cy * r0 - sy * r1;
                            real[j1] = sy * r0 + cy * r1;
                            imag[j0] = cy * i0 - sy * i1;
                            imag[j1] = sy * i0 + cy * i1;
                        }
                        k += 1;
                    }
                    base += stride;
                }
            }

            // Apply each inline-qubit Rz within this W_TILE window
            for q in 0..inline_n {
                let block = 1usize << q;
                let stride = block << 1;
                let cn = rz_cn[q];
                let sn = rz_sn[q];
                let cp = rz_cp[q];
                let sp = rz_sp[q];

                let first_base = (tile_base / stride) * stride;
                let mut base = first_base;
                while base < tile_end {
                    // Bit = 0 half
                    let b0_s = base.max(tile_base);
                    let b0_e = (base + block).min(tile_end);
                    let mut k = b0_s;
                    while k < b0_e && k < dim {
                        let r = real[k];
                        let im = imag[k];
                        real[k] = r * cn - im * sn;
                        imag[k] = r * sn + im * cn;
                        k += 1;
                    }
                    // Bit = 1 half
                    let b1_s = (base + block).max(tile_base);
                    let b1_e = (base + stride).min(tile_end);
                    let mut k = b1_s;
                    while k < b1_e && k < dim {
                        let r = real[k];
                        let im = imag[k];
                        real[k] = r * cp - im * sp;
                        imag[k] = r * sp + im * cp;
                        k += 1;
                    }
                    base += stride;
                }
            }

            tile_base = tile_end;
        }
    }

    // Phase 2 & 3: Remaining qubits (stride > W_TILE/2 or >= INTERLEAVE_THRESHOLD).
    // Apply each gate independently with its own W_TILE-blocked tiled loop.
    for q in inline_n..n_qubits {
        tiled_apply_ry(real, imag, q, theta_ry[q]);
        tiled_apply_rz(real, imag, q, theta_rz[q]);
    }
}

/// Cache-oblivious tiled CZ gate.
///
/// For low-qubit pairs (max(q1,q2) < INTERLEAVE_THRESHOLD): delegates to existing AVX2 kernel.
/// For high-qubit pairs: uses a tile-gated linear scan with prefetching.
#[inline(always)]
pub fn tiled_apply_cz(real: &mut [f64], imag: &mut [f64], q1: usize, q2: usize) {
    let n_qubits = crate::simd::assert_statevector(real, imag);
    crate::simd::assert_qubit_pair(q1, q2, n_qubits);
    if q1.max(q2) < INTERLEAVE_THRESHOLD {
        crate::simd::apply_cz_kernel(real, imag, q1, q2);
    } else {
        apply_cz_tiled(real, imag, q1, q2);
    }
}

// =============================================================================
// CACHE-OBLIVIOUS TILED BUTTERFLY: SCALAR FALLBACK
// =============================================================================

fn apply_ry_tiled_scalar(real: &mut [f64], imag: &mut [f64], qubit: usize, theta: f64) {
    let (s, c) = (theta * 0.5).sin_cos();
    let dim = real.len();
    let block = 1usize << qubit;
    let stride = block << 1;

    // Tile over pairs (tile_base, tile_end) in the |0⟩ half of each butterfly block
    let mut outer = 0;
    while outer < dim {
        let pair_start = outer;
        let pair_end = (outer + block).min(dim);

        // Tile the inner loop by W_TILE to keep both halves in cache
        let mut ti = pair_start;
        while ti < pair_end {
            let te = (ti + W_TILE).min(pair_end);
            for k in ti..te {
                let j0 = k;
                let j1 = k + block;
                if j1 < dim {
                    let r0 = real[j0];
                    let r1 = real[j1];
                    let i0 = imag[j0];
                    let i1 = imag[j1];
                    real[j0] = c * r0 - s * r1;
                    real[j1] = s * r0 + c * r1;
                    imag[j0] = c * i0 - s * i1;
                    imag[j1] = s * i0 + c * i1;
                }
            }
            ti = te;
        }
        outer += stride;
    }
}

fn apply_rz_tiled_scalar(real: &mut [f64], imag: &mut [f64], qubit: usize, theta: f64) {
    let (sn, cn) = (-theta * 0.5).sin_cos();
    let (sp, cp) = (theta * 0.5).sin_cos();
    let dim = real.len();
    let block = 1usize << qubit;
    let stride = block << 1;

    let mut outer = 0;
    while outer < dim {
        // Bit = 0 block → exp(-i θ/2)
        let b0_start = outer;
        let b0_end = (outer + block).min(dim);
        let mut ti = b0_start;
        while ti < b0_end {
            let te = (ti + W_TILE).min(b0_end);
            for k in ti..te {
                let r = real[k];
                let im = imag[k];
                real[k] = r * cn - im * sn;
                imag[k] = r * sn + im * cn;
            }
            ti = te;
        }

        // Bit = 1 block → exp(+i θ/2)
        let b1_start = outer + block;
        let b1_end = (outer + stride).min(dim);
        let mut ti = b1_start;
        while ti < b1_end {
            let te = (ti + W_TILE).min(b1_end);
            for k in ti..te {
                let r = real[k];
                let im = imag[k];
                real[k] = r * cp - im * sp;
                imag[k] = r * sp + im * cp;
            }
            ti = te;
        }
        outer += stride;
    }
}

fn apply_cz_tiled(real: &mut [f64], imag: &mut [f64], q1: usize, q2: usize) {
    let dim = real.len();
    let mask1 = 1usize << q1;
    let mask2 = 1usize << q2;
    let both = mask1 | mask2;

    // Tiled linear scan: skip entire tiles that can't contain any CZ-active index.
    let mut tile_base = 0;
    while tile_base < dim {
        let tile_end = (tile_base + W_TILE).min(dim);

        // Check if any index in [tile_base, tile_end) can have both bits set.
        // If neither bit is set in tile_base..tile_end range, skip the tile entirely.
        let tile_has_candidates = (tile_base..tile_end).any(|k| (k & both) == both);

        if tile_has_candidates {
            for k in tile_base..tile_end {
                if (k & both) == both {
                    real[k] = -real[k];
                    imag[k] = -imag[k];
                }
            }
        }
        tile_base = tile_end;
    }
}

// =============================================================================
// AVX2 + FMA INTERLEAVED HIGH-QUBIT BUTTERFLY IMPLEMENTATIONS
// =============================================================================

/// Ry butterfly for `qubit >= INTERLEAVE_THRESHOLD` using tiled interleaved (AoS) complex layout.
///
/// Strategy: Repack a W_TILE-sized window of (real[j0..j0+W_TILE], real[j1..j1+W_TILE]) into
/// an interleaved buffer [re_j0, im_j0, re_j1, im_j1, ...]. Apply the butterfly in interleaved
/// form using 256-bit loads that bring BOTH real and imag into registers simultaneously.
/// Write back to split layout.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn apply_ry_tiled_interleaved(real: &mut [f64], imag: &mut [f64], qubit: usize, theta: f64) {
    let (s, c) = (theta * 0.5).sin_cos();
    let dim = real.len();
    let block = 1usize << qubit;
    let stride = block << 1;

    let c_v = _mm256_set1_pd(c);
    let s_v = _mm256_set1_pd(s);
    let neg_s_v = _mm256_set1_pd(-s);

    // Reusable interleaved work buffer on stack: holds W_TILE/2 pairs × 4 f64 = 2 × W_TILE f64
    // For W_TILE = 512, this is 4 KB → stays in L1 cache.
    let mut buf_j0: [f64; W_TILE * 2] = [0.0; W_TILE * 2]; // [re0, im0, re0, im0, ...]
    let mut buf_j1: [f64; W_TILE * 2] = [0.0; W_TILE * 2];

    let mut outer = 0usize;
    while outer < dim {
        let pair_start = outer;
        let pair_end = (outer + block).min(dim);

        // Tile over the |0⟩ sub-block in chunks of W_TILE
        let mut ti = pair_start;
        while ti < pair_end {
            let te = (ti + W_TILE).min(pair_end);
            let chunk = te - ti;

            let j0_base = ti;
            let j1_base = ti + block;

            // Software prefetch the NEXT tile's data
            if te + PREFETCH_DIST < pair_end {
                unsafe {
                    let next_j0 = real.as_ptr().add(te + PREFETCH_DIST);
                    let next_j1 = real.as_ptr().add(te + block + PREFETCH_DIST);
                    _mm_prefetch(next_j0 as *const i8, _MM_HINT_T0);
                    _mm_prefetch(next_j1 as *const i8, _MM_HINT_T0);
                    let next_i0 = imag.as_ptr().add(te + PREFETCH_DIST);
                    let next_i1 = imag.as_ptr().add(te + block + PREFETCH_DIST);
                    _mm_prefetch(next_i0 as *const i8, _MM_HINT_T0);
                    _mm_prefetch(next_i1 as *const i8, _MM_HINT_T0);
                }
            }

            // Interleave into local buffer: buf_j0[2k] = real[j0+k], buf_j0[2k+1] = imag[j0+k]
            for k in 0..chunk {
                buf_j0[2 * k] = real[j0_base + k];
                buf_j0[2 * k + 1] = imag[j0_base + k];
                if j1_base + k < dim {
                    buf_j1[2 * k] = real[j1_base + k];
                    buf_j1[2 * k + 1] = imag[j1_base + k];
                }
            }

            // AVX2 butterfly over interleaved buffer: 4 f64 per __m256d
            // Each register holds [re_a, im_a, re_b, im_b] for two consecutive amplitudes.
            let mut vi = 0;
            while vi + 4 <= chunk * 2 {
                // Load two (re, im) pairs from j0 and j1 interleaved buffers
                let v_j0 = _mm256_loadu_pd(buf_j0.as_ptr().add(vi));
                let v_j1 = _mm256_loadu_pd(buf_j1.as_ptr().add(vi));

                // Butterfly: new_j0 = c * j0 - s * j1, new_j1 = s * j0 + c * j1
                let new_j0 = _mm256_fmadd_pd(v_j0, c_v, _mm256_mul_pd(v_j1, neg_s_v));
                let new_j1 = _mm256_fmadd_pd(v_j0, s_v, _mm256_mul_pd(v_j1, c_v));

                _mm256_storeu_pd(buf_j0.as_mut_ptr().add(vi), new_j0);
                _mm256_storeu_pd(buf_j1.as_mut_ptr().add(vi), new_j1);

                vi += 4;
            }

            // De-interleave back to split layout
            for k in 0..chunk {
                real[j0_base + k] = buf_j0[2 * k];
                imag[j0_base + k] = buf_j0[2 * k + 1];
                if j1_base + k < dim {
                    real[j1_base + k] = buf_j1[2 * k];
                    imag[j1_base + k] = buf_j1[2 * k + 1];
                }
            }

            ti = te;
        }
        outer += stride;
    }
}

/// Rz butterfly for `qubit >= INTERLEAVE_THRESHOLD` using tiled interleaved layout.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn apply_rz_tiled_interleaved(real: &mut [f64], imag: &mut [f64], qubit: usize, theta: f64) {
    let (sn, cn) = (-theta * 0.5).sin_cos();
    let (sp, cp) = (theta * 0.5).sin_cos();
    let dim = real.len();
    let block = 1usize << qubit;
    let stride = block << 1;

    let cn_v = _mm256_set1_pd(cn);
    let sn_v = _mm256_set1_pd(sn);
    let cp_v = _mm256_set1_pd(cp);
    let sp_v = _mm256_set1_pd(sp);

    let mut outer = 0usize;
    while outer < dim {
        // Bit = 0 block: multiply by exp(-i θ/2)
        let b0_start = outer;
        let b0_end = (outer + block).min(dim);
        let mut ti = b0_start;
        while ti < b0_end {
            let te = (ti + W_TILE).min(b0_end);

            // Prefetch next tile
            if te + PREFETCH_DIST < b0_end {
                unsafe {
                    _mm_prefetch(
                        real.as_ptr().add(te + PREFETCH_DIST) as *const i8,
                        _MM_HINT_T0,
                    );
                    _mm_prefetch(
                        imag.as_ptr().add(te + PREFETCH_DIST) as *const i8,
                        _MM_HINT_T0,
                    );
                }
            }

            let mut k = ti;
            while k + 4 <= te {
                let r = _mm256_loadu_pd(real.as_ptr().add(k));
                let im = _mm256_loadu_pd(imag.as_ptr().add(k));
                // new_r = r * cn - im * sn
                let new_r = _mm256_fmsub_pd(r, cn_v, _mm256_mul_pd(im, sn_v));
                // new_im = r * sn + im * cn
                let new_im = _mm256_fmadd_pd(r, sn_v, _mm256_mul_pd(im, cn_v));
                _mm256_storeu_pd(real.as_mut_ptr().add(k), new_r);
                _mm256_storeu_pd(imag.as_mut_ptr().add(k), new_im);
                k += 4;
            }
            while k < te {
                let r = real[k];
                let im = imag[k];
                real[k] = r * cn - im * sn;
                imag[k] = r * sn + im * cn;
                k += 1;
            }
            ti = te;
        }

        // Bit = 1 block: multiply by exp(+i θ/2)
        let b1_start = outer + block;
        let b1_end = (outer + stride).min(dim);
        let mut ti = b1_start;
        while ti < b1_end {
            let te = (ti + W_TILE).min(b1_end);

            if te + PREFETCH_DIST < b1_end {
                unsafe {
                    _mm_prefetch(
                        real.as_ptr().add(te + PREFETCH_DIST) as *const i8,
                        _MM_HINT_T0,
                    );
                    _mm_prefetch(
                        imag.as_ptr().add(te + PREFETCH_DIST) as *const i8,
                        _MM_HINT_T0,
                    );
                }
            }

            let mut k = ti;
            while k + 4 <= te {
                let r = _mm256_loadu_pd(real.as_ptr().add(k));
                let im = _mm256_loadu_pd(imag.as_ptr().add(k));
                let new_r = _mm256_fmsub_pd(r, cp_v, _mm256_mul_pd(im, sp_v));
                let new_im = _mm256_fmadd_pd(r, sp_v, _mm256_mul_pd(im, cp_v));
                _mm256_storeu_pd(real.as_mut_ptr().add(k), new_r);
                _mm256_storeu_pd(imag.as_mut_ptr().add(k), new_im);
                k += 4;
            }
            while k < te {
                let r = real[k];
                let im = imag[k];
                real[k] = r * cp - im * sp;
                imag[k] = r * sp + im * cp;
                k += 1;
            }
            ti = te;
        }
        outer += stride;
    }
}

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_haar(dim: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
        use t1000_core::sample_ideal_porter_thomas;
        let probs = sample_ideal_porter_thomas(dim, seed);
        let real: Vec<f64> = probs.iter().map(|&p| p.sqrt()).collect();
        let imag = vec![0.0f64; dim];
        (real, imag)
    }

    fn norm_sq(real: &[f64], imag: &[f64]) -> f64 {
        real.iter()
            .zip(imag.iter())
            .map(|(&r, &i)| r * r + i * i)
            .sum()
    }

    #[test]
    fn tiled_ry_matches_standard_for_low_qubit() {
        let (mut r_std, mut i_std) = make_haar(1024, 1);
        let (mut r_tiled, mut i_tiled) = (r_std.clone(), i_std.clone());

        let theta = 1.234;
        crate::simd::apply_ry_kernel(&mut r_std, &mut i_std, 3, theta);
        tiled_apply_ry(&mut r_tiled, &mut i_tiled, 3, theta);

        for k in 0..1024 {
            assert!(
                (r_std[k] - r_tiled[k]).abs() < 1e-12,
                "k={}: re mismatch: {} vs {}",
                k,
                r_std[k],
                r_tiled[k]
            );
            assert!(
                (i_std[k] - i_tiled[k]).abs() < 1e-12,
                "k={}: im mismatch",
                k
            );
        }
    }

    #[test]
    fn tiled_ry_high_qubit_norm_preserved() {
        // 20-qubit register, test high-qubit (>= 16) gate
        let n = 1 << 20;
        let (mut real, mut imag) = make_haar(n, 42);

        let norm_before = norm_sq(&real, &imag);
        tiled_apply_ry(&mut real, &mut imag, 18, 0.77);
        let norm_after = norm_sq(&real, &imag);

        assert!(
            (norm_before - norm_after).abs() < 1e-8,
            "Norm not preserved: before={}, after={}",
            norm_before,
            norm_after
        );
    }

    #[test]
    fn tiled_rz_high_qubit_norm_preserved() {
        let n = 1 << 20;
        let (mut real, mut imag) = make_haar(n, 99);

        let norm_before = norm_sq(&real, &imag);
        tiled_apply_rz(&mut real, &mut imag, 17, 1.21);
        let norm_after = norm_sq(&real, &imag);

        assert!(
            (norm_before - norm_after).abs() < 1e-8,
            "Norm not preserved: before={}, after={}",
            norm_before,
            norm_after
        );
    }

    #[test]
    fn apply_layer_fused_matches_sequential() {
        let dim = 1024; // 10-qubit
        let n_qubits = 10;
        let (mut r_seq, mut i_seq) = make_haar(dim, 55);
        let (mut r_fused, mut i_fused) = (r_seq.clone(), i_seq.clone());

        let theta_ry: Vec<f64> = (0..n_qubits).map(|q| q as f64 * 0.1 + 0.3).collect();
        let theta_rz: Vec<f64> = (0..n_qubits).map(|q| q as f64 * 0.2 - 0.5).collect();

        // Sequential baseline
        for q in 0..n_qubits {
            crate::simd::apply_ry_kernel(&mut r_seq, &mut i_seq, q, theta_ry[q]);
            crate::simd::apply_rz_kernel(&mut r_seq, &mut i_seq, q, theta_rz[q]);
        }

        // Fused kernel
        apply_layer_fused(&mut r_fused, &mut i_fused, &theta_ry, &theta_rz);

        for k in 0..dim {
            assert!(
                (r_seq[k] - r_fused[k]).abs() < 1e-10,
                "k={}: real mismatch: seq={}, fused={}",
                k,
                r_seq[k],
                r_fused[k]
            );
            assert!(
                (i_seq[k] - i_fused[k]).abs() < 1e-10,
                "k={}: imag mismatch: seq={}, fused={}",
                k,
                i_seq[k],
                i_fused[k]
            );
        }
    }

    #[test]
    fn tiled_cz_matches_standard() {
        let (mut r_std, mut i_std) = make_haar(512, 7);
        let (mut r_tiled, mut i_tiled) = (r_std.clone(), i_std.clone());

        crate::simd::apply_cz_kernel(&mut r_std, &mut i_std, 2, 5);
        tiled_apply_cz(&mut r_tiled, &mut i_tiled, 2, 5);

        for k in 0..512 {
            assert!(
                (r_std[k] - r_tiled[k]).abs() < 1e-14,
                "k={}: CZ real mismatch",
                k
            );
        }
    }
}
