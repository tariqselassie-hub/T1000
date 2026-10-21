// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Cache-blocked single-qubit layer: one statevector pass per group of qubits instead of one
//! per gate.
//!
//! Single-qubit gates on different qubits commute, so a layer of them (at most one gate per
//! qubit) can be regrouped freely. Once the state is larger than the last-level cache every gate
//! pass is DRAM-bound, so the number of passes is the cost. This module applies
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

use crate::compiler::Unitary2x2;
use rayon::prelude::*;

/// Qubits applied together on one in-cache block (`2^12` amplitudes = 64 KB of re + im).
const LOW_BITS: usize = 12;
/// Chunk length (amplitudes) streamed per group member; a multiple of 4.
const CHUNK_BITS: usize = 9;
const CHUNK: usize = 1 << CHUNK_BITS;
/// Maximum high qubits fused into one pass (`2^4` chunks resident).
const GROUP_QUBITS: usize = 4;
/// Largest layer accepted (one gate per qubit).
pub const MAX_LAYER_QUBITS: usize = 64;

/// A single-qubit gate in a layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Gate1Q {
    /// Real rotation `[c -s; s c]` with `(s, c) = sin_cos(theta / 2)`.
    Ry(f64),
    /// Arbitrary complex 2x2 unitary.
    U(Unitary2x2),
}

/// A layer entry: `(qubit, gate)`.
pub type LayerGate = (usize, Gate1Q);

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
    assert!(angles.len() <= MAX_LAYER_QUBITS, "{} angles exceed the layer limit", angles.len());
    let mut gates = [(0usize, Gate1Q::Ry(0.0)); MAX_LAYER_QUBITS];
    for (q, (slot, &theta)) in gates.iter_mut().zip(angles).enumerate() {
        *slot = (q, Gate1Q::Ry(theta));
    }
    apply_1q_layer(real, imag, &mut gates[..angles.len()]);
}

/// Applies one gate per listed qubit. Qubits must be distinct; `gates` is reordered in place.
pub fn apply_1q_layer(real: &mut [f64], imag: &mut [f64], gates: &mut [LayerGate]) {
    let n_qubits = crate::simd::assert_statevector(real, imag);
    gates.sort_unstable_by_key(|g| g.0);
    assert!(gates.windows(2).all(|w| w[0].0 != w[1].0), "layer qubits must be distinct");
    assert!(gates.iter().all(|g| g.0 < n_qubits), "layer qubit out of range");
    let parallel = crate::parallel::use_parallel(real.len());

    let split = gates.partition_point(|g| g.0 < LOW_BITS);
    let (low, high) = gates.split_at(split);
    apply_low_block(real, imag, low, parallel);
    for group in high.chunks(GROUP_QUBITS) {
        apply_high_group(real, imag, group, parallel);
    }
}

/// Gates on qubits `< LOW_BITS`, applied to each `2^LOW_BITS` block in turn.
fn apply_low_block(real: &mut [f64], imag: &mut [f64], gates: &[LayerGate], parallel: bool) {
    if gates.is_empty() {
        return;
    }
    let block = real.len().min(1 << LOW_BITS);
    let run = |re: &mut [f64], im: &mut [f64]| {
        for &(q, gate) in gates {
            match gate {
                Gate1Q::Ry(theta) => crate::simd::apply_ry_kernel(re, im, q, theta),
                Gate1Q::U(u) => apply_u_block(re, im, q, &u),
            }
        }
    };
    if parallel {
        crate::parallel::install(|| {
            real.par_chunks_mut(block)
                .zip(imag.par_chunks_mut(block))
                .for_each(|(re, im)| run(re, im));
        });
    } else {
        real.chunks_mut(block).zip(imag.chunks_mut(block)).for_each(|(re, im)| run(re, im));
    }
}

/// Complex 2x2 on qubit `q` of an in-cache block, pairing the two halves of each sub-block.
fn apply_u_block(re: &mut [f64], im: &mut [f64], q: usize, u: &Unitary2x2) {
    let half = 1usize << q;
    for (re_b, im_b) in re.chunks_mut(2 * half).zip(im.chunks_mut(2 * half)) {
        let (r0, r1) = re_b.split_at_mut(half);
        let (i0, i1) = im_b.split_at_mut(half);
        // SAFETY: the four slices are disjoint and each holds `half` elements.
        unsafe { u_pair(r0.as_mut_ptr(), r1.as_mut_ptr(), i0.as_mut_ptr(), i1.as_mut_ptr(), half, u) };
    }
}

/// One group of high qubits (all `>= CHUNK_BITS`, ascending) in a single pass over the state.
fn apply_high_group(real: &mut [f64], imag: &mut [f64], group: &[LayerGate], par: bool) {
    let g = group.len();
    let bases = real.len() >> (CHUNK_BITS + g);
    let planes = Planes { re: real.as_mut_ptr(), im: imag.as_mut_ptr() };
    let mut qs = [0usize; GROUP_QUBITS];
    let mut ops = [PairOp::Rot { s: 0.0, c: 1.0 }; GROUP_QUBITS];
    for (j, &(q, gate)) in group.iter().enumerate() {
        qs[j] = q;
        ops[j] = pair_op(gate);
    }
    let run = |b: usize| apply_group(planes, b, &qs[..g], &ops[..g]);
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

/// Runs the butterfly stages on the `2^g` chunks addressed by base index `b`.
fn apply_group(p: Planes, b: usize, qs: &[usize], ops: &[PairOp]) {
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
            // SAFETY: `a0` and `a1` are distinct chunks of this group, inside the statevector,
            // and no other base touches them.
            unsafe {
                let (r0, r1) = (p.re.add(a0), p.re.add(a1));
                let (i0, i1) = (p.im.add(a0), p.im.add(a1));
                chunk_pair(r0, r1, i0, i1, &ops[j]);
            }
        }
    }
}

/// A gate with its trigonometry resolved, ready for the pair kernels.
#[derive(Clone, Copy)]
enum PairOp {
    /// Real rotation `[c -s; s c]`: half the arithmetic of the complex kernel.
    Rot { s: f64, c: f64 },
    Mat(Unitary2x2),
}

fn pair_op(gate: Gate1Q) -> PairOp {
    match gate {
        Gate1Q::U(u) => PairOp::Mat(u),
        Gate1Q::Ry(theta) => {
            let (s, c) = (theta * 0.5).sin_cos();
            PairOp::Rot { s, c }
        }
    }
}

/// Applies `op` to a pair of `CHUNK`-element chunks.
///
/// # Safety
/// The four pointers must be valid for `CHUNK` reads and writes, and the ranges must not overlap.
unsafe fn chunk_pair(r0: *mut f64, r1: *mut f64, i0: *mut f64, i1: *mut f64, op: &PairOp) {
    match op {
        PairOp::Mat(u) => u_pair(r0, r1, i0, i1, CHUNK, u),
        PairOp::Rot { s, c } => {
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return rot_pair_avx2(r0, r1, i0, i1, *s, *c);
            }
            for (x0, x1) in [(r0, r1), (i0, i1)] {
                for k in 0..CHUNK {
                    let (a, b) = (*x0.add(k), *x1.add(k));
                    *x0.add(k) = c * a - s * b;
                    *x1.add(k) = s * a + c * b;
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn rot_pair_avx2(r0: *mut f64, r1: *mut f64, i0: *mut f64, i1: *mut f64, s: f64, c: f64) {
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

/// `[x0; x1] <- U [x0; x1]` over `len` amplitudes of each plane.
///
/// # Safety
/// The four pointers must be valid for `len` reads and writes, and the ranges must not overlap.
unsafe fn u_pair(r0: *mut f64, r1: *mut f64, i0: *mut f64, i1: *mut f64, len: usize, u: &Unitary2x2) {
    #[cfg(target_arch = "x86_64")]
    if len >= 4 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return u_pair_avx2(r0, r1, i0, i1, len, u);
    }
    u_pair_scalar(r0, r1, i0, i1, 0, len, u);
}

unsafe fn u_pair_scalar(
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn u_pair_avx2(r0: *mut f64, r1: *mut f64, i0: *mut f64, i1: *mut f64, len: usize, u: &Unitary2x2) {
    let v = |x: f64| _mm256_set1_pd(x);
    let (a_r, a_i, b_r, b_i) = (v(u.u00_re), v(u.u00_im), v(u.u01_re), v(u.u01_im));
    let (c_r, c_i, d_r, d_i) = (v(u.u10_re), v(u.u10_im), v(u.u11_re), v(u.u11_im));
    let vec_len = len - len % 4;
    for k in (0..vec_len).step_by(4) {
        let (xr, xi) = (_mm256_loadu_pd(r0.add(k)), _mm256_loadu_pd(i0.add(k)));
        let (yr, yi) = (_mm256_loadu_pd(r1.add(k)), _mm256_loadu_pd(i1.add(k)));
        // new0 = a*x + b*y ; new1 = c*x + d*y  (complex)
        let n0r = _mm256_fnmadd_pd(b_i, yi, _mm256_fmsub_pd(a_r, xr, _mm256_mul_pd(a_i, xi)));
        let n0r = _mm256_fmadd_pd(b_r, yr, n0r);
        let n0i = _mm256_fmadd_pd(b_i, yr, _mm256_fmadd_pd(a_r, xi, _mm256_mul_pd(a_i, xr)));
        let n0i = _mm256_fmadd_pd(b_r, yi, n0i);
        let n1r = _mm256_fnmadd_pd(d_i, yi, _mm256_fmsub_pd(c_r, xr, _mm256_mul_pd(c_i, xi)));
        let n1r = _mm256_fmadd_pd(d_r, yr, n1r);
        let n1i = _mm256_fmadd_pd(d_i, yr, _mm256_fmadd_pd(c_r, xi, _mm256_mul_pd(c_i, xr)));
        let n1i = _mm256_fmadd_pd(d_r, yi, n1i);
        _mm256_storeu_pd(r0.add(k), n0r);
        _mm256_storeu_pd(i0.add(k), n0i);
        _mm256_storeu_pd(r1.add(k), n1r);
        _mm256_storeu_pd(i1.add(k), n1i);
    }
    u_pair_scalar(r0, r1, i0, i1, vec_len, len, u);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::CircuitOp;
    use crate::state::QuantumState;

    fn pseudo_state(dim: usize) -> QuantumState {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let mut s = QuantumState::new(dim);
        s.real.iter_mut().for_each(|v| *v = next());
        s.imag.iter_mut().for_each(|v| *v = next());
        s
    }

    fn rz_matrix(theta: f64) -> Unitary2x2 {
        let (s, c) = (theta * 0.5).sin_cos();
        Unitary2x2 { u00_re: c, u00_im: -s, u11_re: c, u11_im: s, ..Unitary2x2::IDENTITY }
    }

    fn max_dev(a: &QuantumState, b: &QuantumState) -> f64 {
        let d = |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max);
        d(&a.real, &b.real).max(d(&a.imag, &b.imag))
    }

    fn reference(state: &mut QuantumState, gates: &[LayerGate]) {
        for &(q, g) in gates {
            let op = match g {
                Gate1Q::Ry(theta) => CircuitOp::Ry { qubit: q, theta },
                Gate1Q::U(matrix) => CircuitOp::Unitary1Q { qubit: q, matrix },
            };
            crate::compiler::CompiledCircuit::execute_op(state, &op);
        }
    }

    #[test]
    fn ry_layer_matches_per_gate_kernels() {
        for n in [4, 12, 13, 16, 17, 19, 20] {
            let mut s = pseudo_state(1 << n);
            let mut r = s.clone();
            let angles: Vec<f64> = (0..n).map(|q| 0.17 + 0.31 * q as f64).collect();
            apply_ry_layer(&mut s.real, &mut s.imag, &angles);
            let gates: Vec<LayerGate> = angles.iter().enumerate().map(|(q, &t)| (q, Gate1Q::Ry(t))).collect();
            reference(&mut r, &gates);
            assert!(max_dev(&s, &r) < 1e-12, "n={n}");
        }
    }

    #[test]
    fn mixed_unitary_layer_on_sparse_qubits_matches() {
        for n in [5, 13, 16, 19] {
            let mut s = pseudo_state(1 << n);
            let mut r = s.clone();
            let mut gates: Vec<LayerGate> = (0..n)
                .rev()
                .step_by(2)
                .map(|q| {
                    let t = 0.4 + 0.23 * q as f64;
                    (q, if q % 4 == 0 { Gate1Q::Ry(t) } else { Gate1Q::U(rz_matrix(t)) })
                })
                .collect();
            let want = gates.clone();
            apply_1q_layer(&mut s.real, &mut s.imag, &mut gates);
            reference(&mut r, &want);
            assert!(max_dev(&s, &r) < 1e-12, "n={n}");
        }
    }
}
