// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Strongly-typed SIMD State Vector model and hardware execution engine abstractions.

use super::algebra::{apply_rademacher_phase_kernel, inner_product_kernel, normalize_kernel};
use super::guards::assert_statevector;
use super::measurements::{
    measure_2qubit_pauli_correlators, measure_dual_zx_expectations, measure_fused_ry_full_readout,
    measure_fused_ry_z_readout, measure_single_qubit_z, measure_zz_correlators,
};
use super::rotations::{apply_cz_kernel, apply_ry_kernel, apply_rz_kernel};

/// Hardware execution backend capabilities detected at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdBackend {
    /// 256-bit AVX2 vectorization with Fused Multiply-Add (FMA).
    Avx2Fma,
    /// Portable scalar fallback for non-AVX2 architectures.
    Scalar,
}

impl SimdBackend {
    /// Detects current hardware vector capabilities.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return Self::Avx2Fma;
            }
        }
        Self::Scalar
    }

    /// Descriptive name of the execution backend.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Avx2Fma => "AVX2+FMA (256-bit)",
            Self::Scalar => "Scalar Fallback",
        }
    }

    /// Returns `true` if hardware SIMD acceleration is active.
    pub fn is_vectorized(&self) -> bool {
        matches!(self, Self::Avx2Fma)
    }
}

/// Read-only zero-allocation SIMD statevector view.
#[derive(Debug, Copy, Clone)]
pub struct SimdStateView<'a> {
    pub real: &'a [f64],
    pub imag: &'a [f64],
    pub n_qubits: usize,
}

impl<'a> SimdStateView<'a> {
    /// Creates a verified statevector view. Panics if dimension invariants are violated.
    pub fn new(real: &'a [f64], imag: &'a [f64]) -> Self {
        let n_qubits = assert_statevector(real, imag);
        Self {
            real,
            imag,
            n_qubits,
        }
    }

    /// Returns the number of qubits in the Hilbert register.
    #[inline(always)]
    pub fn qubits(&self) -> usize {
        self.n_qubits
    }

    /// Returns the Hilbert space dimension (2^n).
    #[inline(always)]
    pub fn dim(&self) -> usize {
        self.real.len()
    }

    /// Computes the complex inner product ⟨self | other⟩.
    #[inline(always)]
    pub fn inner_product(&self, other: &SimdStateView<'_>) -> (f64, f64) {
        inner_product_kernel(self.real, self.imag, other.real, other.imag)
    }

    /// Single-qubit Pauli-Z expectation value.
    #[inline(always)]
    pub fn measure_single_z(&self, qubit: usize) -> f64 {
        measure_single_qubit_z(self.real, self.imag, qubit)
    }

    /// Simultaneous dual Pauli-Z and Pauli-X expectation accumulation.
    #[inline(always)]
    pub fn measure_dual_zx(&self, z_out: &mut [f64], x_out: &mut [f64]) {
        measure_dual_zx_expectations(self.real, self.imag, z_out, x_out);
    }

    /// All C(n,2) two-body ⟨Z_i Z_j⟩ correlators.
    #[inline(always)]
    pub fn measure_zz(&self, n_qubits: usize, zz_out: &mut [f64]) {
        measure_zz_correlators(self.real, self.imag, n_qubits, zz_out);
    }

    /// Four 2-qubit Pauli correlators (ZZ, ZX, XZ, XX).
    #[inline(always)]
    pub fn measure_2qubit_pauli(&self, qi: usize, qj: usize) -> (f64, f64, f64, f64) {
        measure_2qubit_pauli_correlators(self.real, self.imag, qi, qj)
    }

    /// Fused terminal Ry + Pauli-Z readout without mutating statevector.
    #[inline(always)]
    pub fn measure_fused_ry_z(&self, theta_y: &[f64], z_final: &mut [f64]) {
        measure_fused_ry_z_readout(self.real, self.imag, theta_y, z_final);
    }

    /// Fused terminal Ry + full 2-body correlator readout without mutating statevector.
    #[inline(always)]
    pub fn measure_fused_ry_full(
        &self,
        theta_y: &[f64],
        z_final: &mut [f64],
        zz_final: &mut [f64],
    ) {
        measure_fused_ry_full_readout(self.real, self.imag, theta_y, z_final, zz_final);
    }
}

/// Mutable zero-allocation SIMD statevector view for high-throughput unitary evolution.
#[derive(Debug)]
pub struct SimdStateViewMut<'a> {
    pub real: &'a mut [f64],
    pub imag: &'a mut [f64],
    pub n_qubits: usize,
}

impl<'a> SimdStateViewMut<'a> {
    /// Creates a verified mutable statevector view. Panics if dimension invariants are violated.
    pub fn new(real: &'a mut [f64], imag: &'a mut [f64]) -> Self {
        let n_qubits = assert_statevector(real, imag);
        Self {
            real,
            imag,
            n_qubits,
        }
    }

    /// Downgrades to a read-only `SimdStateView`.
    #[inline(always)]
    pub fn as_view(&self) -> SimdStateView<'_> {
        SimdStateView {
            real: self.real,
            imag: self.imag,
            n_qubits: self.n_qubits,
        }
    }

    /// Returns the number of qubits in the Hilbert register.
    #[inline(always)]
    pub fn qubits(&self) -> usize {
        self.n_qubits
    }

    /// Returns the Hilbert space dimension (2^n).
    #[inline(always)]
    pub fn dim(&self) -> usize {
        self.real.len()
    }

    /// Applies in-place single-qubit Ry rotation.
    #[inline(always)]
    pub fn ry(&mut self, qubit: usize, theta: f64) -> &mut Self {
        apply_ry_kernel(self.real, self.imag, qubit, theta);
        self
    }

    /// Applies in-place single-qubit Rz rotation.
    #[inline(always)]
    pub fn rz(&mut self, qubit: usize, theta: f64) -> &mut Self {
        apply_rz_kernel(self.real, self.imag, qubit, theta);
        self
    }

    /// Applies in-place two-qubit Controlled-Z (CZ) gate.
    #[inline(always)]
    pub fn cz(&mut self, q1: usize, q2: usize) -> &mut Self {
        apply_cz_kernel(self.real, self.imag, q1, q2);
        self
    }

    /// Applies in-place Walsh-Rademacher diagonal phase layer.
    #[inline(always)]
    pub fn rademacher_phase(&mut self, signs: &[f64]) -> &mut Self {
        apply_rademacher_phase_kernel(self.real, self.imag, signs);
        self
    }

    /// In-place L2 normalization.
    #[inline(always)]
    pub fn normalize(&mut self) -> &mut Self {
        normalize_kernel(self.real, self.imag);
        self
    }

    /// Computes the complex inner product ⟨self | other⟩.
    #[inline(always)]
    pub fn inner_product(&self, other: &SimdStateView<'_>) -> (f64, f64) {
        self.as_view().inner_product(other)
    }

    /// Single-qubit Pauli-Z expectation value.
    #[inline(always)]
    pub fn measure_single_z(&self, qubit: usize) -> f64 {
        self.as_view().measure_single_z(qubit)
    }

    /// Simultaneous dual Pauli-Z and Pauli-X expectation accumulation.
    #[inline(always)]
    pub fn measure_dual_zx(&self, z_out: &mut [f64], x_out: &mut [f64]) {
        self.as_view().measure_dual_zx(z_out, x_out);
    }

    /// All C(n,2) two-body ⟨Z_i Z_j⟩ correlators.
    #[inline(always)]
    pub fn measure_zz(&self, n_qubits: usize, zz_out: &mut [f64]) {
        self.as_view().measure_zz(n_qubits, zz_out);
    }

    /// Four 2-qubit Pauli correlators (ZZ, ZX, XZ, XX).
    #[inline(always)]
    pub fn measure_2qubit_pauli(&self, qi: usize, qj: usize) -> (f64, f64, f64, f64) {
        self.as_view().measure_2qubit_pauli(qi, qj)
    }

    /// Fused terminal Ry + Pauli-Z readout without mutating statevector.
    #[inline(always)]
    pub fn measure_fused_ry_z(&self, theta_y: &[f64], z_final: &mut [f64]) {
        self.as_view().measure_fused_ry_z(theta_y, z_final);
    }

    /// Fused terminal Ry + full 2-body correlator readout without mutating statevector.
    #[inline(always)]
    pub fn measure_fused_ry_full(
        &self,
        theta_y: &[f64],
        z_final: &mut [f64],
        zz_final: &mut [f64],
    ) {
        self.as_view().measure_fused_ry_full(theta_y, z_final, zz_final);
    }
}
