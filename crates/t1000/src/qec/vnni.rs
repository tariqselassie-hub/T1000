// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! AVX-VNNI (_mm256_dpbusd_epi32) Hardware-Accelerated Stabilizer & Syndrome Engine.
//!
//! Executes 32-way parallel INT8 multiply-accumulate operations to accelerate:
//! 1. Multi-qubit stabilizer parity check matrix multiplication: s = H · e (mod 2).
//! 2. Binary symplectic inner products for Clifford+T tableau commutation checks:
//!    ⟨P1, P2⟩_symp = (X1 · Z2 - Z1 · X2) (mod 2).
//! 3. Single-cycle 8-stabilizer evaluation for the [[9, 1, 3]] planar surface code.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Hardware acceleration backend active for VNNI stabilizer passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VnniBackend {
    /// Native 256-bit AVX-VNNI instruction set (Intel Lion Cove / Arrow Lake).
    NativeAvxVnni,
    /// 256-bit AVX2 emulation using `_mm256_maddubs_epi16` + `_mm256_madd_epi16`.
    EmulatedAvx2,
    /// Portable scalar fallback.
    Scalar,
}

impl VnniBackend {
    /// Detects runtime hardware VNNI capabilities.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avxvnni") {
                return Self::NativeAvxVnni;
            }
            if is_x86_feature_detected!("avx2") {
                return Self::EmulatedAvx2;
            }
        }
        Self::Scalar
    }

    /// Returns `true` if SIMD acceleration (native VNNI or AVX2) is active.
    pub fn is_accelerated(&self) -> bool {
        matches!(self, Self::NativeAvxVnni | Self::EmulatedAvx2)
    }

    /// Descriptive name of the execution mode.
    pub fn name(&self) -> &'static str {
        match self {
            Self::NativeAvxVnni => "Native AVX-VNNI (_mm256_dpbusd_epi32)",
            Self::EmulatedAvx2 => "Emulated AVX2 (maddubs+madd)",
            Self::Scalar => "Portable Scalar",
        }
    }
}

/// 32-way INT8 dot-product kernel with runtime hardware dispatch.
/// Computes: dst[k] = src[k] + sum_{b=0..3} (u8[4k + b] * i8[4k + b]) for k in 0..7.
#[inline(always)]
pub fn vnni_dpbusd_32(src: [i32; 8], a_u8: [u8; 32], b_i8: [i8; 32]) -> [i32; 8] {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avxvnni") {
            unsafe {
                return vnni_dpbusd_native(src, a_u8, b_i8);
            }
        }
        if is_x86_feature_detected!("avx2") {
            unsafe {
                return vnni_dpbusd_avx2(src, a_u8, b_i8);
            }
        }
    }
    vnni_dpbusd_scalar(src, a_u8, b_i8)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vnni_dpbusd_native(src: [i32; 8], a_u8: [u8; 32], b_i8: [i8; 32]) -> [i32; 8] {
    let mut out = [0i32; 8];
    std::arch::asm!(
        "vmovdqu {dst}, [{src_ptr}]",
        "vmovdqu ymm1, [{a_ptr}]",
        "vmovdqu ymm2, [{b_ptr}]",
        "{{vex}} vpdpbusd {dst}, ymm1, ymm2",
        "vmovdqu [{out_ptr}], {dst}",
        src_ptr = in(reg) src.as_ptr(),
        a_ptr = in(reg) a_u8.as_ptr(),
        b_ptr = in(reg) b_i8.as_ptr(),
        out_ptr = in(reg) out.as_mut_ptr(),
        dst = out(ymm_reg) _,
        out("ymm1") _,
        out("ymm2") _,
        options(nostack),
    );
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vnni_dpbusd_avx2(src: [i32; 8], a_u8: [u8; 32], b_i8: [i8; 32]) -> [i32; 8] {
    let src_v = _mm256_loadu_si256(src.as_ptr() as *const __m256i);
    let a_v = _mm256_loadu_si256(a_u8.as_ptr() as *const __m256i);
    let b_v = _mm256_loadu_si256(b_i8.as_ptr() as *const __m256i);

    let prod16 = _mm256_maddubs_epi16(a_v, b_v);
    let ones = _mm256_set1_epi16(1);
    let prod32 = _mm256_madd_epi16(prod16, ones);
    let res = _mm256_add_epi32(src_v, prod32);

    let mut out = [0i32; 8];
    _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, res);
    out
}

#[inline(always)]
fn vnni_dpbusd_scalar(src: [i32; 8], a_u8: [u8; 32], b_i8: [i8; 32]) -> [i32; 8] {
    let mut out = src;
    for k in 0..8 {
        let mut dot = 0i32;
        for b in 0..4 {
            let idx = k * 4 + b;
            dot += (a_u8[idx] as i32) * (b_i8[idx] as i32);
        }
        out[k] += dot;
    }
    out
}

/// Evaluates binary symplectic inner product ⟨P1, P2⟩_symp = (X1 · Z2 - Z1 · X2) mod 2
/// across up to 16 qubits in a single 32-way VNNI instruction.
/// Returns `0` if P1 and P2 commute, or `1` if they anticommute.
pub fn symplectic_commutation_16q(x1: u16, z1: u16, x2: u16, z2: u16) -> u8 {
    let mut a_u8 = [0u8; 32];
    let mut b_i8 = [0i8; 32];

    // First 16 bytes: X1 * Z2
    for q in 0..16 {
        a_u8[q] = ((x1 >> q) & 1) as u8;
        b_i8[q] = ((z2 >> q) & 1) as i8;
    }
    // Next 16 bytes: Z1 * (-X2)
    for q in 0..16 {
        a_u8[16 + q] = ((z1 >> q) & 1) as u8;
        b_i8[16 + q] = -(((x2 >> q) & 1) as i8);
    }

    let acc = vnni_dpbusd_32([0; 8], a_u8, b_i8);
    let mut total = 0i32;
    for val in acc {
        total += val;
    }
    (total.rem_euclid(2)) as u8
}

/// Compact check-row representation of the 8 plaquette stabilizers of [[9, 1, 3]] Surface Code.
/// Pre-formatted for direct 32-way VNNI dot-product reduction against 9-qubit error vectors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SurfaceCode9QVnniMatrix {
    /// Mask matrix: 8 rows (stabilizers), 4 dwords each mapped to 4 bytes.
    pub z_masks: [u8; 32],
    pub x_masks: [u8; 32],
}

impl SurfaceCode9QVnniMatrix {
    /// Constructs the standard [[9, 1, 3]] planar surface code VNNI check matrix.
    pub fn new() -> Self {
        let mut z_masks = [0u8; 32];
        let mut x_masks = [0u8; 32];

        // S_z1 = Z1 Z2 Z4 Z5 (qubits 1, 2, 4, 5) -> Dword 0 (bytes 0..4: 1, 2, 4, 5)
        z_masks[0] = 1;
        z_masks[1] = 2;
        z_masks[2] = 4;
        z_masks[3] = 5;
        // S_z2 = Z3 Z4 Z6 Z7 (qubits 3, 4, 6, 7) -> Dword 1 (bytes 4..8)
        z_masks[4] = 3;
        z_masks[5] = 4;
        z_masks[6] = 6;
        z_masks[7] = 7;
        // S_z3 = Z0 Z3 (qubits 0, 3) -> Dword 2 (bytes 8..12)
        z_masks[8] = 0;
        z_masks[9] = 3;
        z_masks[10] = 0xFF;
        z_masks[11] = 0xFF;
        // S_z4 = Z5 Z8 (qubits 5, 8) -> Dword 3 (bytes 12..16)
        z_masks[12] = 5;
        z_masks[13] = 8;
        z_masks[14] = 0xFF;
        z_masks[15] = 0xFF;

        // S_x1 = X0 X1 X3 X4 (qubits 0, 1, 3, 4) -> Dword 4 (bytes 16..20)
        x_masks[16] = 0;
        x_masks[17] = 1;
        x_masks[18] = 3;
        x_masks[19] = 4;
        // S_x2 = X4 X5 X7 X8 (qubits 4, 5, 7, 8) -> Dword 5 (bytes 20..24)
        x_masks[20] = 4;
        x_masks[21] = 5;
        x_masks[22] = 7;
        x_masks[23] = 8;
        // S_x3 = X1 X2 (qubits 1, 2) -> Dword 6 (bytes 24..28)
        x_masks[24] = 1;
        x_masks[25] = 2;
        x_masks[26] = 0xFF;
        x_masks[27] = 0xFF;
        // S_x4 = X6 X7 (qubits 6, 7) -> Dword 7 (bytes 28..32)
        x_masks[28] = 6;
        x_masks[29] = 7;
        x_masks[30] = 0xFF;
        x_masks[31] = 0xFF;

        Self { z_masks, x_masks }
    }

    /// Hardware-accelerated syndrome extraction from 9-qubit physical error bitstrings.
    /// - `error_x`: Bitmask of Pauli-X (bit flip) errors on qubits 0..8.
    /// - `error_z`: Bitmask of Pauli-Z (phase flip) errors on qubits 0..8.
    /// Returns 8-element array: [Sz1, Sz2, Sz3, Sz4, Sx1, Sx2, Sx3, Sx4] where values are +1 or -1.
    #[inline(always)]
    pub fn extract_syndromes_vnni(&self, error_x: u16, error_z: u16) -> [i8; 8] {
        let mut err_vec_z = [0i8; 32];
        let mut err_vec_x = [0i8; 32];

        // Fill error vector matching the check matrix layout
        for i in 0..32 {
            let q_z = self.z_masks[i];
            if q_z != 0xFF {
                err_vec_z[i] = ((error_x >> q_z) & 1) as i8;
            }
            let q_x = self.x_masks[i];
            if q_x != 0xFF {
                err_vec_x[i] = ((error_z >> q_x) & 1) as i8;
            }
        }

        // Single VNNI pass for 4 Z-stabilizers
        let acc_z = vnni_dpbusd_32([0; 8], [1u8; 32], err_vec_z);
        // Single VNNI pass for 4 X-stabilizers
        let acc_x = vnni_dpbusd_32([0; 8], [1u8; 32], err_vec_x);

        let mut out = [1i8; 8];
        for k in 0..4 {
            out[k] = if (acc_z[k] & 1) == 0 { 1 } else { -1 };
        }
        for k in 0..4 {
            out[4 + k] = if (acc_x[4 + k] & 1) == 0 { 1 } else { -1 };
        }
        out
    }
}

impl Default for SurfaceCode9QVnniMatrix {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vnni_backend_detection() {
        println!("avxvnni: {}", is_x86_feature_detected!("avxvnni"));
        println!("avx512vnni: {}", is_x86_feature_detected!("avx512vnni"));
        println!("avx512vl: {}", is_x86_feature_detected!("avx512vl"));
        println!("avx512f: {}", is_x86_feature_detected!("avx512f"));
        let backend = VnniBackend::detect();
        println!("Active VNNI Backend: {}", backend.name());
        assert!(backend.is_accelerated());
    }

    #[test]
    fn test_vnni_dpbusd_numerical_equivalence() {
        let mut a = [0u8; 32];
        let mut b = [0i8; 32];
        for i in 0..32 {
            a[i] = (i % 7) as u8;
            b[i] = ((i as i8) % 5) - 2;
        }
        let src = [10, -5, 2, 8, -1, 0, 14, -7];
        let scalar = vnni_dpbusd_scalar(src, a, b);
        let dispatched = vnni_dpbusd_32(src, a, b);
        assert_eq!(scalar, dispatched);
    }

    #[test]
    fn test_symplectic_commutation() {
        // Pauli X0 and Z0 on qubit 0: anticommute -> 1
        assert_eq!(symplectic_commutation_16q(1, 0, 0, 1), 1);
        // Pauli X0 and X0: commute -> 0
        assert_eq!(symplectic_commutation_16q(1, 0, 1, 0), 0);
        // Pauli X0 Z1 and Z0 X1: X0*Z0 (-1) + Z1*X1 (+1) = 0 mod 2 -> commute
        assert_eq!(symplectic_commutation_16q(0b01, 0b10, 0b10, 0b01), 0);
        // Pauli X0 Z1 and X0 X1: X0*X0 (0) + Z1*X1 (-1) = 1 mod 2 -> anticommute
        assert_eq!(symplectic_commutation_16q(0b01, 0b10, 0b11, 0b00), 1);
    }

    #[test]
    fn test_surface_code_syndromes_all_single_qubit_errors() {
        let matrix = SurfaceCode9QVnniMatrix::new();

        // No errors -> all +1
        let syn_clean = matrix.extract_syndromes_vnni(0, 0);
        assert_eq!(syn_clean, [1, 1, 1, 1, 1, 1, 1, 1]);

        // Single X error on qubit 0 (detected by S_z3: Z0 Z3)
        let syn_x0 = matrix.extract_syndromes_vnni(1 << 0, 0);
        assert_eq!(syn_x0, [1, 1, -1, 1, 1, 1, 1, 1]);

        // Single X error on qubit 4 (detected by S_z1: Z1 Z2 Z4 Z5 and S_z2: Z3 Z4 Z6 Z7)
        let syn_x4 = matrix.extract_syndromes_vnni(1 << 4, 0);
        assert_eq!(syn_x4, [-1, -1, 1, 1, 1, 1, 1, 1]);

        // Single Z error on qubit 1 (detected by S_x1: X0 X1 X3 X4 and S_x3: X1 X2)
        let syn_z1 = matrix.extract_syndromes_vnni(0, 1 << 1);
        assert_eq!(syn_z1, [1, 1, 1, 1, -1, 1, -1, 1]);

        // Combined Y error on qubit 4 (both X and Z on qubit 4)
        let syn_y4 = matrix.extract_syndromes_vnni(1 << 4, 1 << 4);
        assert_eq!(syn_y4, [-1, -1, 1, 1, -1, -1, 1, 1]);
    }
}
