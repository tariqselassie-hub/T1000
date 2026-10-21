// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Cache-blocked backward-sweep kernels for the layered ansatz adjoint gradient.
//!
//! Per layer the sweep needs, for every qubit, an Rz gradient `Im <lambda| Z_q |phi>` and an Ry
//! gradient `Im <lambda| Y_q |phi>`, then undoes the Ry gates on both states. Done gate by gate
//! that is about `4n` passes over a state; here it is
//!
//! * one pass for all Rz gradients ([`rz_grad_all`]): with `t(x) = Im(conj(lambda) phi)`, the
//!   gradient is `T - 2 B_q` where `T = sum t` and `B_q` sums `t` over indices with bit `q` set,
//!   computed per cache block by folding halves; and
//! * the layer-fusion block/chunk passes for the Ry gradients and the Ry undo together
//!   ([`ry_grad_and_undo`]). `Y_q` commutes with `Ry` on any other qubit, so a gate's gradient
//!   may be read after other qubits' gates were already undone on both states.
//!
//! Sums are reduced per fixed-size task and combined in task order, so results do not depend on
//! the thread count.

use crate::layer_fusion::{
    chunk_pair, insert_zero_bit, pair_op, PairOp, CHUNK, CHUNK_BITS, GROUP_QUBITS, LOW_BITS,
};
use crate::state::QuantumState;
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const BLOCKS_PER_TASK: usize = 8;
const BASES_PER_TASK: usize = 32;

/// Base pointers of the four planes, shared across workers that touch disjoint blocks.
#[derive(Clone, Copy)]
struct Joint {
    phi_re: *mut f64,
    phi_im: *mut f64,
    lam_re: *mut f64,
    lam_im: *mut f64,
}

// SAFETY: workers write disjoint blocks/chunks (see the callers); the pointers outlive the call.
unsafe impl Send for Joint {}
unsafe impl Sync for Joint {}

/// Runs `f(item, acc)` over `items`, grouped into fixed-size tasks with one `width`-wide partial
/// per task, and returns the partials summed in task order.
fn reduce_tasks(
    items: usize,
    per_task: usize,
    width: usize,
    parallel: bool,
    f: impl Fn(usize, &mut [f64]) + Sync,
) -> Vec<f64> {
    let tasks = items.div_ceil(per_task).max(1);
    let mut partial = vec![0.0; tasks * width];
    let run = |(t, acc): (usize, &mut [f64])| {
        for item in t * per_task..((t + 1) * per_task).min(items) {
            f(item, acc);
        }
    };
    if parallel {
        crate::parallel::install(|| partial.par_chunks_mut(width).enumerate().for_each(run));
    } else {
        partial.chunks_mut(width).enumerate().for_each(run);
    }
    let mut total = vec![0.0; width];
    for row in partial.chunks(width) {
        total.iter_mut().zip(row).for_each(|(t, v)| *t += v);
    }
    total
}

/// Sum with four independent accumulators.
fn sum4(x: &[f64]) -> f64 {
    let mut acc = [0.0f64; 4];
    let (quads, rest) = x.as_chunks::<4>();
    for c in quads {
        acc.iter_mut().zip(c).for_each(|(a, v)| *a += v);
    }
    let tail: f64 = rest.iter().sum();
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
}

/// Writes `out[q] = sum_x s_q(x) * Im(conj(lambda_x) phi_x)` for every qubit, with
/// `s_q(x) = +1` when bit `q` of `x` is 0 and `-1` otherwise.
pub fn rz_grad_all(phi: &QuantumState, lambda: &QuantumState, out: &mut [f64]) {
    let n = out.len();
    let dim = phi.dim;
    assert_eq!(dim, 1 << n, "gradient length must equal the qubit count");
    let blk = dim.min(1 << LOW_BITS);
    let low = n.min(LOW_BITS);
    let parallel = crate::parallel::use_parallel(dim);

    let total = reduce_tasks(dim / blk, BLOCKS_PER_TASK, n + 1, parallel, |h, acc| {
        let s = h * blk;
        let mut scratch = [0.0f64; 1 << LOW_BITS];
        let t = &mut scratch[..blk];
        let (pr, pi) = (&phi.real[s..s + blk], &phi.imag[s..s + blk]);
        let (lr, li) = (&lambda.real[s..s + blk], &lambda.imag[s..s + blk]);
        for k in 0..blk {
            t[k] = lr[k] * pi[k] - li[k] * pr[k];
        }
        // Fold from the top bit down: u[half..2 * half] holds the bit-q-set partial sums.
        let mut len = blk;
        for q in (0..low).rev() {
            let half = len / 2;
            let (lo, hi) = t[..len].split_at_mut(half);
            acc[q] += sum4(hi);
            lo.iter_mut().zip(hi.iter()).for_each(|(a, b)| *a += b);
            len = half;
        }
        let block_sum = t[0];
        acc[n] += block_sum;
        for (k, a) in acc.iter_mut().take(n).skip(LOW_BITS).enumerate() {
            if (h >> k) & 1 == 1 {
                *a += block_sum;
            }
        }
    });
    for q in 0..n {
        out[q] = total[n] - 2.0 * total[q];
    }
}

/// Computes the Ry gradients `out[q] = Im <lambda| Y_q |phi>` and then applies `Ry(-theta[q])` to
/// both states, for every qubit `q < theta.len()`.
pub fn ry_grad_and_undo(
    phi: &mut QuantumState,
    lambda: &mut QuantumState,
    theta: &[f64],
    out: &mut [f64],
) {
    let n = theta.len();
    let dim = phi.dim;
    assert_eq!(dim, 1 << n, "angle count must equal the qubit count");
    assert_eq!(out.len(), n);
    let joint = Joint {
        phi_re: phi.real.as_mut_ptr(),
        phi_im: phi.imag.as_mut_ptr(),
        lam_re: lambda.real.as_mut_ptr(),
        lam_im: lambda.imag.as_mut_ptr(),
    };
    let parallel = crate::parallel::use_parallel(dim);

    let blk = dim.min(1 << LOW_BITS);
    let low = n.min(LOW_BITS);
    let low_grad = reduce_tasks(dim / blk, BLOCKS_PER_TASK, low, parallel, |h, acc| {
        // SAFETY: block `h` is touched by exactly one worker.
        unsafe { low_block(joint, h * blk, blk, &theta[..low], acc) };
    });
    out[..low].copy_from_slice(&low_grad);

    let high: Vec<usize> = (LOW_BITS..n).collect();
    for group in high.chunks(GROUP_QUBITS) {
        let g = group.len();
        let ops: Vec<PairOp> = group.iter().map(|&q| pair_op(crate::layer_fusion::Gate1Q::Ry(-theta[q]))).collect();
        let bases = dim >> (CHUNK_BITS + g);
        let grads = reduce_tasks(bases, BASES_PER_TASK, g, parallel, |b, acc| {
            // SAFETY: base `b` addresses chunks no other base touches.
            unsafe { high_group(joint, b, group, &ops, acc) };
        });
        out[group[0]..group[0] + g].copy_from_slice(&grads);
    }
}

/// Gradient and undo for qubits `0..theta.len()` on the block at `start`.
///
/// # Safety
/// `[start, start + blk)` of all four planes must be valid and not accessed concurrently.
unsafe fn low_block(joint: Joint, start: usize, blk: usize, theta: &[f64], acc: &mut [f64]) {
    let planes = [joint.phi_re, joint.phi_im, joint.lam_re, joint.lam_im].map(|p| p.add(start));
    let [pr, pi, lr, li] = planes;
    for (q, &th) in theta.iter().enumerate() {
        let half = 1usize << q;
        for off in (0..blk).step_by(2 * half) {
            acc[q] += cross(
                [lr.add(off), li.add(off), pr.add(off), pi.add(off)],
                [lr.add(off + half), li.add(off + half), pr.add(off + half), pi.add(off + half)],
                half,
            );
        }
        let re = std::slice::from_raw_parts_mut(pr, blk);
        let im = std::slice::from_raw_parts_mut(pi, blk);
        crate::simd::apply_ry_kernel(re, im, q, -th);
        let re = std::slice::from_raw_parts_mut(lr, blk);
        let im = std::slice::from_raw_parts_mut(li, blk);
        crate::simd::apply_ry_kernel(re, im, q, -th);
    }
}

/// Gradient and undo for one group of high qubits at base index `b`.
///
/// # Safety
/// The group's chunks for `b` must be valid and not accessed concurrently.
unsafe fn high_group(joint: Joint, b: usize, qs: &[usize], ops: &[PairOp], acc: &mut [f64]) {
    let g = qs.len();
    let x = qs.iter().fold(b, |x, &q| insert_zero_bit(x, q - CHUNK_BITS));
    let base = x << CHUNK_BITS;
    for j in 0..g {
        let bit = 1usize << qs[j];
        for mask in 0..(1usize << g) {
            if mask & (1 << j) != 0 {
                continue;
            }
            let off: usize = (0..g).filter(|k| mask & (1 << k) != 0).map(|k| 1usize << qs[k]).sum();
            let (a0, a1) = (base + off, base + off + bit);
            let at = |a: usize| [joint.lam_re.add(a), joint.lam_im.add(a), joint.phi_re.add(a), joint.phi_im.add(a)];
            let (x0, x1) = (at(a0), at(a1));
            acc[j] += cross(x0, x1, CHUNK);
            chunk_pair(x0[2], x1[2], x0[3], x1[3], &ops[j]);
            chunk_pair(x0[0], x1[0], x0[1], x1[1], &ops[j]);
        }
    }
}

/// `sum lr1 * pr0 + li1 * pi0 - lr0 * pr1 - li0 * pi1` over `len` amplitudes.
///
/// Each argument is `[lambda_re, lambda_im, phi_re, phi_im]` for the |0> / |1> half.
unsafe fn cross(x0: [*mut f64; 4], x1: [*mut f64; 4], len: usize) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if len >= 4 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return cross_avx2(x0, x1, len);
    }
    cross_scalar(x0, x1, 0, len)
}

unsafe fn cross_scalar(x0: [*mut f64; 4], x1: [*mut f64; 4], from: usize, len: usize) -> f64 {
    let [lr0, li0, pr0, pi0] = x0;
    let [lr1, li1, pr1, pi1] = x1;
    let mut g = 0.0;
    for k in from..len {
        g += *lr1.add(k) * *pr0.add(k) + *li1.add(k) * *pi0.add(k)
            - *lr0.add(k) * *pr1.add(k)
            - *li0.add(k) * *pi1.add(k);
    }
    g
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn cross_avx2(x0: [*mut f64; 4], x1: [*mut f64; 4], len: usize) -> f64 {
    let [lr0, li0, pr0, pi0] = x0;
    let [lr1, li1, pr1, pi1] = x1;
    let mut acc = _mm256_setzero_pd();
    let vec_len = len - len % 4;
    for k in (0..vec_len).step_by(4) {
        let ld = |p: *mut f64| _mm256_loadu_pd(p.add(k));
        acc = _mm256_fmadd_pd(ld(lr1), ld(pr0), acc);
        acc = _mm256_fmadd_pd(ld(li1), ld(pi0), acc);
        acc = _mm256_fnmadd_pd(ld(lr0), ld(pr1), acc);
        acc = _mm256_fnmadd_pd(ld(li0), ld(pi1), acc);
    }
    let mut lanes = [0.0f64; 4];
    _mm256_storeu_pd(lanes.as_mut_ptr(), acc);
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]) + cross_scalar(x0, x1, vec_len, len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::apply_ry_kernel;

    fn random_state(n: usize, seed: u64) -> QuantumState {
        let mut x = seed;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let mut s = QuantumState::new(1 << n);
        s.real.iter_mut().for_each(|v| *v = next());
        s.imag.iter_mut().for_each(|v| *v = next());
        s
    }

    #[test]
    fn rz_grad_matches_direct_sum() {
        for n in [3, 8, 12, 14, 16] {
            let (phi, lam) = (random_state(n, 11), random_state(n, 29));
            let mut got = vec![0.0; n];
            rz_grad_all(&phi, &lam, &mut got);
            for q in 0..n {
                let want: f64 = (0..1usize << n)
                    .map(|x| {
                        let t = lam.real[x] * phi.imag[x] - lam.imag[x] * phi.real[x];
                        if (x >> q) & 1 == 0 { t } else { -t }
                    })
                    .sum();
                assert!((got[q] - want).abs() < 1e-10, "n={n} q={q}: {} vs {want}", got[q]);
            }
        }
    }

    #[test]
    fn ry_grad_and_undo_matches_per_gate_sequence() {
        for n in [3, 8, 12, 13, 16, 17] {
            let (mut phi, mut lam) = (random_state(n, 5), random_state(n, 77));
            let (mut phi_ref, mut lam_ref) = (phi.clone(), lam.clone());
            let theta: Vec<f64> = (0..n).map(|q| 0.3 + 0.41 * q as f64).collect();
            let mut got = vec![0.0; n];
            ry_grad_and_undo(&mut phi, &mut lam, &theta, &mut got);

            for q in 0..n {
                let mask = 1usize << q;
                let want: f64 = (0..1usize << (n - 1))
                    .map(|t| {
                        let x0 = ((t >> q) << (q + 1)) | (t & (mask - 1));
                        let x1 = x0 | mask;
                        lam_ref.real[x1] * phi_ref.real[x0] + lam_ref.imag[x1] * phi_ref.imag[x0]
                            - lam_ref.real[x0] * phi_ref.real[x1]
                            - lam_ref.imag[x0] * phi_ref.imag[x1]
                    })
                    .sum();
                assert!((got[q] - want).abs() < 1e-10, "n={n} q={q}: {} vs {want}", got[q]);
            }
            for (q, &t) in theta.iter().enumerate() {
                for s in [&mut phi_ref, &mut lam_ref] {
                    apply_ry_kernel(&mut s.real, &mut s.imag, q, -t);
                }
            }
            let dev = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
            assert!(dev(&phi.real, &phi_ref.real).max(dev(&phi.imag, &phi_ref.imag)) < 1e-12, "n={n} phi");
            assert!(dev(&lam.real, &lam_ref.real).max(dev(&lam.imag, &lam_ref.imag)) < 1e-12, "n={n} lambda");
        }
    }
}
