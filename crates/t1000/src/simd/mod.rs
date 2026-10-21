// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! 256-bit AVX2 & FMA Hardware-Accelerated Vectorization Subsystem.
//!
//! Architected for Intel Arrow Lake (Lion Cove / Skymont) and x86_64 processors,
//! processing 4 double-precision complex amplitudes (8 x f64) per SIMD lane.
//!
//! Decomposed into modular domains:
//! - [`guards`]: Statevector dimension and index precondition assertion guards.
//! - [`rotations`]: Single-qubit (Ry, Rz) and two-qubit (CZ) rotation kernels.
//! - [`algebra`]: Hilbert space normalization, inner products, and Rademacher phase layers.
//! - [`measurements`]: Pauli reductions, dual ZX accumulation, and fused observable readouts.
//! - [`model`]: Strongly-typed [`SimdStateView`], [`SimdStateViewMut`], and [`SimdBackend`] domain models.

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx512;
pub mod algebra;
pub mod eigensystem;
pub mod guards;
#[cfg(target_arch = "x86_64")]
mod lowq;
pub mod measurements;
pub mod model;
pub mod rotations;
pub(crate) mod unitary;

// Re-export all primary kernel functions for 100% backward compatibility
pub use algebra::{apply_rademacher_phase_kernel, inner_product_kernel, normalize_kernel};
pub use eigensystem::{
    solve_coulomb_eigensystem_50, solve_symmetric_eigensystem, SymmetricEigensystem,
};
pub use guards::{
    assert_fused_readout_width, assert_qubit, assert_qubit_pair, assert_statevector,
    MAX_FUSED_READOUT_QUBITS,
};
pub use measurements::{
    measure_2qubit_pauli_correlators, measure_dual_zx_expectations, measure_fused_ry_full_readout,
    measure_fused_ry_z_readout, measure_single_qubit_z, measure_zz_correlators,
};
pub use model::{SimdBackend, SimdStateView, SimdStateViewMut};
pub use rotations::{apply_cz_kernel, apply_ry_kernel, apply_rz_kernel};

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_simd_backend_detection() {
        let backend = SimdBackend::detect();
        println!("Active SIMD Backend: {}", backend.name());
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            assert_eq!(backend, SimdBackend::Avx2Fma);
            assert!(backend.is_vectorized());
        }
    }

    #[test]
    fn test_simd_state_view_model() {
        let mut real = vec![0.0; 8];
        let mut imag = vec![0.0; 8];
        real[0] = 1.0;

        let mut view = SimdStateViewMut::new(&mut real, &mut imag);
        assert_eq!(view.qubits(), 3);
        assert_eq!(view.dim(), 8);

        // Apply Ry(pi/2) to qubit 0
        view.ry(0, PI * 0.5);

        // Expectation of Z on qubit 0 should now be 0.0 (superposition)
        let z0 = view.measure_single_z(0);
        assert!((z0).abs() < 1e-12, "expected z0 ~ 0, got {}", z0);

        // Expectation of Z on qubit 1 should still be 1.0
        let z1 = view.measure_single_z(1);
        assert!((z1 - 1.0).abs() < 1e-12, "expected z1 ~ 1, got {}", z1);

        // Normalize
        view.normalize();

        // Inner product with ground state
        let mut g_re = vec![0.0; 8];
        let g_im = vec![0.0; 8];
        g_re[0] = 1.0;
        let ground = SimdStateView::new(&g_re, &g_im);

        let (re, im) = view.inner_product(&ground);
        let expected = (PI * 0.25).cos();
        assert!((re - expected).abs() < 1e-12);
        assert!(im.abs() < 1e-12);
    }
}
