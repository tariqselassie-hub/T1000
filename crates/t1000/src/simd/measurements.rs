// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Vectorized Pauli reduction and expectation measurement kernels with AVX2+FMA acceleration.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::guards::{assert_fused_readout_width, assert_qubit, assert_qubit_pair, assert_statevector};

/// Vectorized Dual Pauli-Z and Pauli-X Read-Only Reduction Kernel:
/// Simultaneously accumulates <Z_q> and <X_q> for each qubit q across the statevector.
/// Eliminates all writeback traffic and cache line dirtying.
#[inline(always)]
pub fn measure_dual_zx_expectations(
    real: &[f64],
    imag: &[f64],
    z_out: &mut [f64],
    x_out: &mut [f64],
) {
    let n_qubits = assert_statevector(real, imag);
    let n = z_out.len().min(x_out.len());
    assert!(
        n <= n_qubits,
        "requested {} qubit expectations from a {}-qubit state",
        n,
        n_qubits
    );
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        measure_dual_zx_avx2(real, imag, &mut z_out[..n], &mut x_out[..n]);
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                measure_dual_zx_avx2(real, imag, &mut z_out[..n], &mut x_out[..n]);
                return;
            }
        }
        measure_dual_zx_scalar(real, imag, &mut z_out[..n], &mut x_out[..n]);
    }
}

/// Zero-Allocation Fused Terminal Ry + Pauli-Z Observable Readout Kernel:
/// Evaluates:
///   <Z_q>_final = cos(θ_q) <Z_q>_L2 - sin(θ_q) <X_q>_L2
/// directly from the unmutated Layer 2 statevector |ψ_L2> via read-only AVX2 reduction.
/// Never mutates the statevector in Layer 3.
#[inline(always)]
pub fn measure_fused_ry_z_readout(
    real: &[f64],
    imag: &[f64],
    theta_y: &[f64],
    z_final: &mut [f64],
) {
    let n = theta_y.len().min(z_final.len());
    assert_fused_readout_width(n);
    let mut z_l2 = [0.0; 16];
    let mut x_l2 = [0.0; 16];
    measure_dual_zx_expectations(real, imag, &mut z_l2[..n], &mut x_l2[..n]);
    for q in 0..n {
        let (s, c) = theta_y[q].sin_cos();
        z_final[q] = c * z_l2[q] - s * x_l2[q];
    }
}

/// Read-Only Single-Qubit Pauli-Z Expectation Kernel:
/// Computes ⟨Z_q⟩ = Σ_k |ψ_k|² · (-1)^(bit_q(k)) directly from statevector in a single pass.
#[inline(always)]
pub fn measure_single_qubit_z(real: &[f64], imag: &[f64], qubit: usize) -> f64 {
    let n_qubits = assert_statevector(real, imag);
    assert_qubit(qubit, n_qubits);
    let mask = 1usize << qubit;
    let mut diff = 0.0;
    for idx in 0..real.len() {
        let prob = real[idx] * real[idx] + imag[idx] * imag[idx];
        if (idx & mask) == 0 {
            diff += prob;
        } else {
            diff -= prob;
        }
    }
    diff
}

/// Read-Only 2-Body Ising Correlator Measurement Kernel:
/// Computes all C(n,2) two-body correlators ⟨Z_i Z_j⟩ in a single linear pass over the
/// statevector. For 12 qubits this produces 66 correlator values.
///
/// Output layout: zz_out[pair_idx] where pair_idx = i * (i-1)/2 + j for i > j.
/// Total pairs = n * (n-1) / 2.
#[inline(always)]
pub fn measure_zz_correlators(
    real: &[f64],
    imag: &[f64],
    n_qubits: usize,
    zz_out: &mut [f64],
) {
    let state_qubits = assert_statevector(real, imag);
    assert!(
        n_qubits <= state_qubits,
        "requested correlators over {} qubits from a {}-qubit state",
        n_qubits,
        state_qubits
    );
    let dim = real.len();
    let n_pairs = n_qubits * n_qubits.saturating_sub(1) / 2;
    assert!(zz_out.len() >= n_pairs, "zz_out must have room for {} pairs", n_pairs);

    for v in zz_out[..n_pairs].iter_mut() {
        *v = 0.0;
    }

    for idx in 0..dim {
        let prob = real[idx] * real[idx] + imag[idx] * imag[idx];
        if prob < 1e-30 {
            continue;
        }

        let mut pair_idx = 0;
        for i in 1..n_qubits {
            let bit_i = (idx >> i) & 1;
            for j in 0..i {
                let bit_j = (idx >> j) & 1;
                let sign = if bit_i == bit_j { 1.0 } else { -1.0 };
                zz_out[pair_idx] += sign * prob;
                pair_idx += 1;
            }
        }
    }
}

/// Read-Only 2-Body Cross-Observable Measurement Kernel:
/// Computes the four 2-qubit Pauli correlators needed for fused Ry-rotated ZZ readout:
///   ⟨Z_i Z_j⟩, ⟨Z_i X_j⟩, ⟨X_i Z_j⟩, ⟨X_i X_j⟩
/// for a single pair (qi, qj) from the unmutated statevector.
#[inline(always)]
pub fn measure_2qubit_pauli_correlators(
    real: &[f64],
    imag: &[f64],
    qi: usize,
    qj: usize,
) -> (f64, f64, f64, f64) {
    let n_qubits = assert_statevector(real, imag);
    assert_qubit_pair(qi, qj, n_qubits);
    let dim = real.len();
    let mask_i = 1usize << qi;
    let mask_j = 1usize << qj;

    let mut zz = 0.0;
    let mut zx = 0.0;
    let mut xz = 0.0;
    let mut xx = 0.0;

    for idx in 0..dim {
        let prob = real[idx] * real[idx] + imag[idx] * imag[idx];

        let bi = (idx & mask_i) != 0;
        let bj = (idx & mask_j) != 0;
        let sign_zi: f64 = if bi { -1.0 } else { 1.0 };
        let sign_zj: f64 = if bj { -1.0 } else { 1.0 };

        zz += sign_zi * sign_zj * prob;

        let partner_j = idx ^ mask_j;
        if idx < partner_j {
            let cross_j = real[idx] * real[partner_j] + imag[idx] * imag[partner_j];
            zx += 2.0 * sign_zi * cross_j;
        }

        let partner_i = idx ^ mask_i;
        if idx < partner_i {
            let cross_i = real[idx] * real[partner_i] + imag[idx] * imag[partner_i];
            xz += 2.0 * sign_zj * cross_i;
        }

        let partner_ij = idx ^ mask_i ^ mask_j;
        if idx < partner_ij {
            let cross_ij = real[idx] * real[partner_ij] + imag[idx] * imag[partner_ij];
            xx += 2.0 * cross_ij;
        }
    }

    (zz, zx, xz, xx)
}

/// Zero-Allocation Fused Terminal Ry + Full 2-Body Correlator Readout Kernel:
/// Computes both single-qubit ⟨Z_q⟩ AND all two-body ⟨Z_i Z_j⟩
/// expectations after terminal Ry rotation, entirely from the unmutated statevector.
#[inline(always)]
pub fn measure_fused_ry_full_readout(
    real: &[f64],
    imag: &[f64],
    theta_y: &[f64],
    z_final: &mut [f64],
    zz_final: &mut [f64],
) {
    let n = theta_y.len().min(z_final.len());
    assert_fused_readout_width(n);
    let n_pairs = n * n.saturating_sub(1) / 2;
    assert!(zz_final.len() >= n_pairs);

    let mut z_l2 = [0.0; 16];
    let mut x_l2 = [0.0; 16];
    measure_dual_zx_expectations(real, imag, &mut z_l2[..n], &mut x_l2[..n]);

    let mut sin_t = [0.0; 16];
    let mut cos_t = [0.0; 16];
    for q in 0..n {
        let (s, c) = theta_y[q].sin_cos();
        sin_t[q] = s;
        cos_t[q] = c;
        z_final[q] = c * z_l2[q] - s * x_l2[q];
    }

    let mut pair_idx = 0;
    for i in 1..n {
        for j in 0..i {
            let (zz, zx, xz, xx) = measure_2qubit_pauli_correlators(real, imag, i, j);
            zz_final[pair_idx] = cos_t[i] * cos_t[j] * zz
                - cos_t[i] * sin_t[j] * zx
                - sin_t[i] * cos_t[j] * xz
                + sin_t[i] * sin_t[j] * xx;
            pair_idx += 1;
        }
    }
}

// =========================================================================
// AVX2 INTRINSICS
// =========================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn accumulate_zx_q0(real: &[f64], imag: &[f64], dim: usize) -> (f64, f64) {
    let mut z_acc0 = 0.0;
    let mut z_acc1 = 0.0;
    let mut x_acc0 = 0.0;
    let mut x_acc1 = 0.0;
    let mut i = 0;
    while i + 4 <= dim {
        let r0 = *real.get_unchecked(i);
        let i0 = *imag.get_unchecked(i);
        let r1 = *real.get_unchecked(i + 1);
        let i1 = *imag.get_unchecked(i + 1);
        let r2 = *real.get_unchecked(i + 2);
        let i2 = *imag.get_unchecked(i + 2);
        let r3 = *real.get_unchecked(i + 3);
        let i3 = *imag.get_unchecked(i + 3);

        let p0 = r0 * r0 + i0 * i0;
        let p1 = r1 * r1 + i1 * i1;
        z_acc0 += p0 - p1;
        x_acc0 += r0 * r1 + i0 * i1;

        let p2 = r2 * r2 + i2 * i2;
        let p3 = r3 * r3 + i3 * i3;
        z_acc1 += p2 - p3;
        x_acc1 += r2 * r3 + i2 * i3;

        i += 4;
    }
    while i + 2 <= dim {
        let r0 = *real.get_unchecked(i);
        let i0 = *imag.get_unchecked(i);
        let r1 = *real.get_unchecked(i + 1);
        let i1 = *imag.get_unchecked(i + 1);
        let p0 = r0 * r0 + i0 * i0;
        let p1 = r1 * r1 + i1 * i1;
        z_acc0 += p0 - p1;
        x_acc0 += r0 * r1 + i0 * i1;
        i += 2;
    }
    (z_acc0 + z_acc1, 2.0 * (x_acc0 + x_acc1))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn accumulate_zx_q1(real: &[f64], imag: &[f64], dim: usize) -> (f64, f64) {
    let mut z_acc0 = 0.0;
    let mut z_acc1 = 0.0;
    let mut x_acc0 = 0.0;
    let mut x_acc1 = 0.0;
    let mut i = 0;
    while i + 4 <= dim {
        let r0 = *real.get_unchecked(i);
        let i0 = *imag.get_unchecked(i);
        let r1 = *real.get_unchecked(i + 1);
        let i1 = *imag.get_unchecked(i + 1);
        let r2 = *real.get_unchecked(i + 2);
        let i2 = *imag.get_unchecked(i + 2);
        let r3 = *real.get_unchecked(i + 3);
        let i3 = *imag.get_unchecked(i + 3);

        let p0 = r0 * r0 + i0 * i0;
        let p1 = r1 * r1 + i1 * i1;
        let p2 = r2 * r2 + i2 * i2;
        let p3 = r3 * r3 + i3 * i3;

        z_acc0 += p0 - p2;
        z_acc1 += p1 - p3;
        x_acc0 += r0 * r2 + i0 * i2;
        x_acc1 += r1 * r3 + i1 * i3;

        i += 4;
    }
    (z_acc0 + z_acc1, 2.0 * (x_acc0 + x_acc1))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn accumulate_zx_q_high(real: &[f64], imag: &[f64], dim: usize, q: usize) -> (f64, f64) {
    let block_size = 1 << q;
    let stride = block_size << 1;

    let mut z_vec0 = _mm256_setzero_pd();
    let mut z_vec1 = _mm256_setzero_pd();
    let mut x_vec0 = _mm256_setzero_pd();
    let mut x_vec1 = _mm256_setzero_pd();

    let mut base = 0;
    while base < dim {
        let p_base = base + block_size;
        let mut k = 0;
        while k + 8 <= block_size {
            let idx0 = base + k;
            let idx1 = p_base + k;

            let r0_0 = _mm256_loadu_pd(real.as_ptr().add(idx0));
            let i0_0 = _mm256_loadu_pd(imag.as_ptr().add(idx0));
            let r1_0 = _mm256_loadu_pd(real.as_ptr().add(idx1));
            let i1_0 = _mm256_loadu_pd(imag.as_ptr().add(idx1));

            let r0_1 = _mm256_loadu_pd(real.as_ptr().add(idx0 + 4));
            let i0_1 = _mm256_loadu_pd(imag.as_ptr().add(idx0 + 4));
            let r1_1 = _mm256_loadu_pd(real.as_ptr().add(idx1 + 4));
            let i1_1 = _mm256_loadu_pd(imag.as_ptr().add(idx1 + 4));

            let s0_0 = _mm256_fmadd_pd(i0_0, i0_0, z_vec0);
            let s0_1 = _mm256_fmadd_pd(r0_0, r0_0, s0_0);
            let s0_2 = _mm256_fnmadd_pd(i1_0, i1_0, s0_1);
            z_vec0 = _mm256_fnmadd_pd(r1_0, r1_0, s0_2);

            let s1_0 = _mm256_fmadd_pd(i0_1, i0_1, z_vec1);
            let s1_1 = _mm256_fmadd_pd(r0_1, r0_1, s1_0);
            let s1_2 = _mm256_fnmadd_pd(i1_1, i1_1, s1_1);
            z_vec1 = _mm256_fnmadd_pd(r1_1, r1_1, s1_2);

            x_vec0 = _mm256_fmadd_pd(r0_0, r1_0, _mm256_fmadd_pd(i0_0, i1_0, x_vec0));
            x_vec1 = _mm256_fmadd_pd(r0_1, r1_1, _mm256_fmadd_pd(i0_1, i1_1, x_vec1));

            k += 8;
        }
        while k < block_size {
            let idx0 = base + k;
            let idx1 = p_base + k;

            let r0 = _mm256_loadu_pd(real.as_ptr().add(idx0));
            let i0 = _mm256_loadu_pd(imag.as_ptr().add(idx0));
            let r1 = _mm256_loadu_pd(real.as_ptr().add(idx1));
            let i1 = _mm256_loadu_pd(imag.as_ptr().add(idx1));

            let s0 = _mm256_fmadd_pd(i0, i0, z_vec0);
            let s1 = _mm256_fmadd_pd(r0, r0, s0);
            let s2 = _mm256_fnmadd_pd(i1, i1, s1);
            z_vec0 = _mm256_fnmadd_pd(r1, r1, s2);

            x_vec0 = _mm256_fmadd_pd(r0, r1, _mm256_fmadd_pd(i0, i1, x_vec0));

            k += 4;
        }
        base += stride;
    }

    let z_sum = _mm256_add_pd(z_vec0, z_vec1);
    let x_sum = _mm256_add_pd(x_vec0, x_vec1);
    let mut z_buf = [0.0; 4];
    let mut x_buf = [0.0; 4];
    _mm256_storeu_pd(z_buf.as_mut_ptr(), z_sum);
    _mm256_storeu_pd(x_buf.as_mut_ptr(), x_sum);

    (
        z_buf[0] + z_buf[1] + z_buf[2] + z_buf[3],
        2.0 * (x_buf[0] + x_buf[1] + x_buf[2] + x_buf[3]),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn measure_dual_zx_avx2(real: &[f64], imag: &[f64], z_out: &mut [f64], x_out: &mut [f64]) {
    let dim = real.len();
    let n = z_out.len().min(x_out.len());

    for q in 0..n {
        let (z_val, x_val) = match q {
            0 => accumulate_zx_q0(real, imag, dim),
            1 => accumulate_zx_q1(real, imag, dim),
            _ => accumulate_zx_q_high(real, imag, dim, q),
        };
        z_out[q] = z_val;
        x_out[q] = x_val;
    }
}

// =========================================================================
// SCALAR FALLBACK IMPLEMENTATIONS
// =========================================================================

#[allow(dead_code)]
pub(crate) fn measure_dual_zx_scalar(real: &[f64], imag: &[f64], z_out: &mut [f64], x_out: &mut [f64]) {
    let dim = real.len();
    let n = z_out.len().min(x_out.len());
    for q in 0..n {
        let mask = 1 << q;
        let mut z_acc = 0.0;
        let mut x_acc = 0.0;
        for i in 0..dim {
            if (i & mask) == 0 {
                let partner = i | mask;
                let r0 = real[i];
                let i0 = imag[i];
                let r1 = real[partner];
                let i1 = imag[partner];

                let p0 = r0 * r0 + i0 * i0;
                let p1 = r1 * r1 + i1 * i1;
                z_acc += p0 - p1;
                x_acc += r0 * r1 + i0 * i1;
            }
        }
        z_out[q] = z_acc;
        x_out[q] = 2.0 * x_acc;
    }
}
