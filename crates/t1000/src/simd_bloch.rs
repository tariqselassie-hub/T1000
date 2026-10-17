// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Single-Pass AVX2 Vectorized Bloch Vector & Qubit Purity Evaluation.
//! Computes ⟨X_q⟩, ⟨Y_q⟩, and ⟨Z_q⟩ in a single memory traversal over the statevector,
//! calculating single-qubit reduced density matrix purity γ_q = (1 + r_q²) / 2.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use serde::{Deserialize, Serialize};

/// 3D Bloch sphere coordinates and purity metrics for a single qubit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BlochVector {
    pub rx: f64,
    pub ry: f64,
    pub rz: f64,
    pub radius: f64,
    pub purity: f64,
}

/// Evaluates (⟨X_q⟩, ⟨Y_q⟩, ⟨Z_q⟩) for qubit `q` in a single pass over statevector.
pub fn compute_qubit_bloch(
    state_re: &[f64],
    state_imag: &[f64],
    q: usize,
    n_qubits: usize,
) -> BlochVector {
    assert_eq!(state_re.len(), 1 << n_qubits);
    assert_eq!(state_imag.len(), 1 << n_qubits);
    assert!(q < n_qubits);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                return compute_qubit_bloch_avx2(state_re, state_imag, q, n_qubits);
            }
        }
    }

    compute_qubit_bloch_scalar(state_re, state_imag, q, n_qubits)
}

/// Evaluates Bloch vectors across all `n_qubits`.
pub fn compute_all_bloch_vectors(
    state_re: &[f64],
    state_imag: &[f64],
    n_qubits: usize,
) -> Vec<BlochVector> {
    (0..n_qubits)
        .map(|q| compute_qubit_bloch(state_re, state_imag, q, n_qubits))
        .collect()
}

/// Computes the restoring spring force w_q = -2 * λ_Bloch * (r_min - r_q) / r_q
/// pulling the state back toward the Bloch surface whenever r_q < r_min (e.g. r_min = 0.75).
pub fn compute_bloch_restoring_weights(
    bloch_vectors: &[BlochVector],
    lambda_bloch: f64,
    r_min: f64,
) -> Vec<f64> {
    bloch_vectors
        .iter()
        .map(|bv| {
            if bv.radius < r_min {
                -2.0 * lambda_bloch * (r_min - bv.radius) / bv.radius.max(1e-6)
            } else {
                0.0
            }
        })
        .collect()
}

/// AVX2 + FMA hardware-accelerated Bloch vector evaluation.
///
/// # Safety
/// Caller must ensure AVX2 and FMA CPU features are available on the host processor and that
/// `state_re` and `state_imag` contain at least `1 << n_qubits` elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn compute_qubit_bloch_avx2(
    state_re: &[f64],
    state_imag: &[f64],
    q: usize,
    n_qubits: usize,
) -> BlochVector {
    let dim = 1usize << n_qubits;
    let stride = 1usize << q;
    let step = stride << 1;

    let mut sum_x = _mm256_setzero_pd();
    let mut sum_y = _mm256_setzero_pd();
    let mut sum_z = _mm256_setzero_pd();

    let re_ptr = state_re.as_ptr();
    let im_ptr = state_imag.as_ptr();

    if stride >= 4 {
        let mut base = 0;
        while base < dim {
            let mut offset = 0;
            while offset < stride {
                let i0 = base + offset;
                let i1 = i0 + stride;

                let re0 = _mm256_loadu_pd(re_ptr.add(i0));
                let im0 = _mm256_loadu_pd(im_ptr.add(i0));
                let re1 = _mm256_loadu_pd(re_ptr.add(i1));
                let im1 = _mm256_loadu_pd(im_ptr.add(i1));

                // ⟨Z_q⟩: |ψ_0|² - |ψ_1|²
                let p0 = _mm256_fmadd_pd(re0, re0, _mm256_mul_pd(im0, im0));
                let p1 = _mm256_fmadd_pd(re1, re1, _mm256_mul_pd(im1, im1));
                sum_z = _mm256_add_pd(sum_z, _mm256_sub_pd(p0, p1));

                // ⟨X_q⟩: 2 * (re0*re1 + im0*im1)
                let x_term = _mm256_fmadd_pd(re0, re1, _mm256_mul_pd(im0, im1));
                sum_x = _mm256_add_pd(sum_x, x_term);

                // ⟨Y_q⟩: 2 * (re0*im1 - im0*re1)
                let y_term = _mm256_fmsub_pd(re0, im1, _mm256_mul_pd(im0, re1));
                sum_y = _mm256_add_pd(sum_y, y_term);

                offset += 4;
            }
            base += step;
        }
    } else {
        return compute_qubit_bloch_scalar(state_re, state_imag, q, n_qubits);
    }

    let rx = 2.0 * hsum_avx2(sum_x);
    let ry = 2.0 * hsum_avx2(sum_y);
    let rz = hsum_avx2(sum_z);
    let r2 = (rx * rx + ry * ry + rz * rz).min(1.0);
    let radius = r2.sqrt();

    BlochVector {
        rx,
        ry,
        rz,
        radius,
        purity: 0.5 * (1.0 + r2),
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum_avx2(v: __m256d) -> f64 {
    let hi = _mm256_extractf128_pd(v, 1);
    let lo = _mm256_castpd256_pd128(v);
    let sum2 = _mm_add_pd(lo, hi);
    let hi64 = _mm_unpackhi_pd(sum2, sum2);
    _mm_cvtsd_f64(_mm_add_sd(sum2, hi64))
}

/// Scalar fallback for stride < 4 or non-x86_64 platforms.
pub fn compute_qubit_bloch_scalar(
    state_re: &[f64],
    state_imag: &[f64],
    q: usize,
    n_qubits: usize,
) -> BlochVector {
    let dim = 1usize << n_qubits;
    let stride = 1usize << q;
    let step = stride << 1;

    let mut rx = 0.0;
    let mut ry = 0.0;
    let mut rz = 0.0;

    let mut base = 0;
    while base < dim {
        for offset in 0..stride {
            let i0 = base + offset;
            let i1 = i0 + stride;
            let (re0, im0) = (state_re[i0], state_imag[i0]);
            let (re1, im1) = (state_re[i1], state_imag[i1]);

            rz += (re0 * re0 + im0 * im0) - (re1 * re1 + im1 * im1);
            rx += re0 * re1 + im0 * im1;
            ry += re0 * im1 - im0 * re1;
        }
        base += step;
    }

    let rx = 2.0 * rx;
    let ry = 2.0 * ry;
    let r2 = (rx * rx + ry * ry + rz * rz).min(1.0);
    let radius = r2.sqrt();

    BlochVector {
        rx,
        ry,
        rz,
        radius,
        purity: 0.5 * (1.0 + r2),
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloch_computational_basis_states() {
        let n = 3;
        let mut re = vec![0.0; 1 << n];
        let im = vec![0.0; 1 << n];

        // State |000>: Qubit 0, 1, 2 must have rz = +1.0, rx = 0, ry = 0
        re[0] = 1.0;
        for q in 0..n {
            let bv = compute_qubit_bloch(&re, &im, q, n);
            assert!((bv.rz - 1.0).abs() < 1e-12);
            assert!(bv.rx.abs() < 1e-12);
            assert!(bv.ry.abs() < 1e-12);
            assert!((bv.radius - 1.0).abs() < 1e-12);
            assert!((bv.purity - 1.0).abs() < 1e-12);
        }

        // State |001> (qubit 0 = 1): Qubit 0 must have rz = -1.0
        re[0] = 0.0;
        re[1] = 1.0;
        let bv0 = compute_qubit_bloch(&re, &im, 0, n);
        assert!((bv0.rz - (-1.0)).abs() < 1e-12);
        assert!((bv0.radius - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_bloch_superposition_plus_state() {
        let n = 2;
        let mut re = vec![0.0; 1 << n];
        let im = vec![0.0; 1 << n];

        // State |++> = 0.5 (|00> + |01> + |10> + |11>)
        for i in 0..4 {
            re[i] = 0.5;
        }

        for q in 0..n {
            let bv = compute_qubit_bloch(&re, &im, q, n);
            assert!((bv.rx - 1.0).abs() < 1e-12);
            assert!(bv.ry.abs() < 1e-12);
            assert!(bv.rz.abs() < 1e-12);
            assert!((bv.radius - 1.0).abs() < 1e-12);
            assert!((bv.purity - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_bloch_maximally_entangled_bell_state() {
        let n = 2;
        let mut re = vec![0.0; 1 << n];
        let im = vec![0.0; 1 << n];

        // Bell state 1/√2 (|00> + |11>)
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        re[0] = inv_sqrt2;
        re[3] = inv_sqrt2;

        for q in 0..n {
            let bv = compute_qubit_bloch(&re, &im, q, n);
            // Reduced density matrix is 1/2 I -> rx = ry = rz = 0
            assert!(bv.rx.abs() < 1e-12);
            assert!(bv.ry.abs() < 1e-12);
            assert!(bv.rz.abs() < 1e-12);
            assert!(bv.radius < 1e-12);
            assert!((bv.purity - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn test_restoring_force_calculation() {
        let bvs = vec![
            BlochVector {
                radius: 0.90,
                ..Default::default()
            }, // pure enough, no force
            BlochVector {
                radius: 0.50,
                ..Default::default()
            }, // collapsed, restoring force active
        ];
        let weights = compute_bloch_restoring_weights(&bvs, 0.1, 0.75);
        assert_eq!(weights[0], 0.0);
        assert!(weights[1] < 0.0); // negative restoring spring pulling back
    }
}
