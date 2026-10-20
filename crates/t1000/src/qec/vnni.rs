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

/// 4-way unrolled independent accumulator pipeline breaking the 5-cycle loop-carried dependency chain.
/// Saturates dual execution ports on Intel Lion Cove (Arrow Lake) architectures.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn bench_vnni_unrolled_4x(
    u8_syndrome: __m256i,
    i8_weights: __m256i,
    iterations: usize,
) -> [i32; 8] {
    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let mut acc2 = _mm256_setzero_si256();
    let mut acc3 = _mm256_setzero_si256();

    let chunks = iterations / 4;
    for _ in 0..chunks {
        std::arch::asm!(
            "{{vex}} vpdpbusd {acc0}, {u8_s}, {i8_w}",
            "{{vex}} vpdpbusd {acc1}, {u8_s}, {i8_w}",
            "{{vex}} vpdpbusd {acc2}, {u8_s}, {i8_w}",
            "{{vex}} vpdpbusd {acc3}, {u8_s}, {i8_w}",
            acc0 = inout(ymm_reg) acc0,
            acc1 = inout(ymm_reg) acc1,
            acc2 = inout(ymm_reg) acc2,
            acc3 = inout(ymm_reg) acc3,
            u8_s = in(ymm_reg) u8_syndrome,
            i8_w = in(ymm_reg) i8_weights,
            options(nostack, pure, nomem),
        );
    }

    let sum01 = _mm256_add_epi32(acc0, acc1);
    let sum23 = _mm256_add_epi32(acc2, acc3);
    let total = _mm256_add_epi32(sum01, sum23);

    let mut out = [0i32; 8];
    _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, total);
    out
}

/// 8-way unrolled independent accumulator pipeline achieving full saturation of
/// dual-issue VNNI vector execution ports (Ports 0 and 5).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn bench_vnni_unrolled_8x(
    u8_syndrome: __m256i,
    i8_weights: __m256i,
    iterations: usize,
) -> [i32; 8] {
    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let mut acc2 = _mm256_setzero_si256();
    let mut acc3 = _mm256_setzero_si256();
    let mut acc4 = _mm256_setzero_si256();
    let mut acc5 = _mm256_setzero_si256();
    let mut acc6 = _mm256_setzero_si256();
    let mut acc7 = _mm256_setzero_si256();

    let chunks = iterations / 8;
    for _ in 0..chunks {
        std::arch::asm!(
            "{{vex}} vpdpbusd {acc0}, {u8_s}, {i8_w}",
            "{{vex}} vpdpbusd {acc1}, {u8_s}, {i8_w}",
            "{{vex}} vpdpbusd {acc2}, {u8_s}, {i8_w}",
            "{{vex}} vpdpbusd {acc3}, {u8_s}, {i8_w}",
            "{{vex}} vpdpbusd {acc4}, {u8_s}, {i8_w}",
            "{{vex}} vpdpbusd {acc5}, {u8_s}, {i8_w}",
            "{{vex}} vpdpbusd {acc6}, {u8_s}, {i8_w}",
            "{{vex}} vpdpbusd {acc7}, {u8_s}, {i8_w}",
            acc0 = inout(ymm_reg) acc0,
            acc1 = inout(ymm_reg) acc1,
            acc2 = inout(ymm_reg) acc2,
            acc3 = inout(ymm_reg) acc3,
            acc4 = inout(ymm_reg) acc4,
            acc5 = inout(ymm_reg) acc5,
            acc6 = inout(ymm_reg) acc6,
            acc7 = inout(ymm_reg) acc7,
            u8_s = in(ymm_reg) u8_syndrome,
            i8_w = in(ymm_reg) i8_weights,
            options(nostack, pure, nomem),
        );
    }

    let s01 = _mm256_add_epi32(acc0, acc1);
    let s23 = _mm256_add_epi32(acc2, acc3);
    let s45 = _mm256_add_epi32(acc4, acc5);
    let s67 = _mm256_add_epi32(acc6, acc7);
    let s0123 = _mm256_add_epi32(s01, s23);
    let s4567 = _mm256_add_epi32(s45, s67);
    let total = _mm256_add_epi32(s0123, s4567);

    let mut out = [0i32; 8];
    _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, total);
    out
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

/// Minimum-weight homologous X-correction lookup table for all 16 Z-stabilizer syndromes.
pub const LOOKUP_X_CORRECTION: [u16; 16] = [
    0,                             // 0: ( 1,  1,  1,  1) -> No error
    1 << 1,                        // 1: (-1,  1,  1,  1) -> X1
    1 << 6,                        // 2: ( 1, -1,  1,  1) -> X6
    1 << 4,                        // 3: (-1, -1,  1,  1) -> X4
    1 << 0,                        // 4: ( 1,  1, -1,  1) -> X0
    (1 << 0) | (1 << 1),           // 5: (-1,  1, -1,  1) -> X0 X1
    1 << 3,                        // 6: ( 1, -1, -1,  1) -> X3
    (1 << 1) | (1 << 3),           // 7: (-1, -1, -1,  1) -> X1 X3
    1 << 8,                        // 8: ( 1,  1,  1, -1) -> X8
    1 << 5,                        // 9: (-1,  1,  1, -1) -> X5
    (1 << 6) | (1 << 8),           // 10: ( 1, -1,  1, -1) -> X6 X8
    (1 << 4) | (1 << 8),           // 11: (-1, -1,  1, -1) -> X4 X8
    (1 << 0) | (1 << 8),           // 12: ( 1,  1, -1, -1) -> X0 X8
    (1 << 0) | (1 << 5),           // 13: (-1,  1, -1, -1) -> X0 X5
    (1 << 3) | (1 << 8),           // 14: ( 1, -1, -1, -1) -> X3 X8
    (1 << 3) | (1 << 5),           // 15: (-1, -1, -1, -1) -> X3 X5
];

/// Minimum-weight homologous Z-correction lookup table for all 16 X-stabilizer syndromes.
pub const LOOKUP_Z_CORRECTION: [u16; 16] = [
    0,                             // 0: ( 1,  1,  1,  1) -> No error
    1 << 0,                        // 1: (-1,  1,  1,  1) -> Z0
    1 << 5,                        // 2: ( 1, -1,  1,  1) -> Z5
    1 << 4,                        // 3: (-1, -1,  1,  1) -> Z4
    1 << 2,                        // 4: ( 1,  1, -1,  1) -> Z2
    1 << 1,                        // 5: (-1,  1, -1,  1) -> Z1
    (1 << 2) | (1 << 5),           // 6: ( 1, -1, -1,  1) -> Z2 Z5
    (1 << 1) | (1 << 5),           // 7: (-1, -1, -1,  1) -> Z1 Z5
    1 << 6,                        // 8: ( 1,  1,  1, -1) -> Z6
    (1 << 0) | (1 << 6),           // 9: (-1,  1,  1, -1) -> Z0 Z6
    1 << 7,                        // 10: ( 1, -1,  1, -1) -> Z7
    (1 << 0) | (1 << 7),           // 11: (-1, -1,  1, -1) -> Z0 Z7
    (1 << 2) | (1 << 6),           // 12: ( 1,  1, -1, -1) -> Z2 Z6
    (1 << 1) | (1 << 6),           // 13: (-1,  1, -1, -1) -> Z1 Z6
    (1 << 2) | (1 << 7),           // 14: ( 1, -1, -1, -1) -> Z2 Z7
    (1 << 1) | (1 << 7),           // 15: (-1, -1, -1, -1) -> Z1 Z7
];

/// Point datum in an empirical fault-tolerant threshold curve.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VnniThresholdPoint {
    pub p_phys: f64,
    pub shots: usize,
    pub logical_errors: usize,
    pub logical_error_rate: f64,
    pub suppression_ratio: f64,
    pub latency_ns_per_shot: f64,
}

/// Comprehensive telemetry report from the AVX-VNNI Monte Carlo threshold engine.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VnniThresholdReport {
    pub points: Vec<VnniThresholdPoint>,
    pub empirical_threshold: Option<f64>,
    pub fit_scaling_coefficient: f64,
    pub total_shots: usize,
    pub total_duration_seconds: f64,
    pub throughput_mshots_sec: f64,
    pub hardware_backend: String,
}

/// Real-time closed-loop QEC decoder combining 32-way VNNI syndrome extraction with O(1) table correction.
#[derive(Debug, Clone, Default)]
pub struct VnniQecDecoder {
    pub matrix: SurfaceCode9QVnniMatrix,
}

impl VnniQecDecoder {
    /// Constructs a new real-time VNNI decoder instance.
    pub fn new() -> Self {
        Self {
            matrix: SurfaceCode9QVnniMatrix::new(),
        }
    }

    /// Evaluates syndrome and retrieves minimum-weight correction `(cx, cz)` in single-cycle lookup.
    #[inline(always)]
    pub fn decode(&self, error_x: u16, error_z: u16) -> (u16, u16) {
        let syn = self.matrix.extract_syndromes_vnni(error_x, error_z);
        let syn_z_idx = ((syn[0] == -1) as usize)
            | (((syn[1] == -1) as usize) << 1)
            | (((syn[2] == -1) as usize) << 2)
            | (((syn[3] == -1) as usize) << 3);
        let syn_x_idx = ((syn[4] == -1) as usize)
            | (((syn[5] == -1) as usize) << 1)
            | (((syn[6] == -1) as usize) << 2)
            | (((syn[7] == -1) as usize) << 3);

        (
            LOOKUP_X_CORRECTION[syn_z_idx],
            LOOKUP_Z_CORRECTION[syn_x_idx],
        )
    }

    /// Checks if residual error `R = E ^ C` causes a non-trivial logical operator flip.
    /// Returns `(bit_flip_failure, phase_flip_failure)`.
    #[inline(always)]
    pub fn evaluate_logical_error(
        &self,
        error_x: u16,
        error_z: u16,
        cx: u16,
        cz: u16,
    ) -> (bool, bool) {
        let rx = error_x ^ cx;
        let rz = error_z ^ cz;
        // Z_L = Z0 Z1 Z2 (row 0: qubits 0, 1, 2)
        let bit_flip_fail = (rx & 0b000000111).count_ones() & 1 == 1;
        // X_L = X0 X3 X6 (col 0: qubits 0, 3, 6)
        let phase_flip_fail = (rz & 0b001001001).count_ones() & 1 == 1;
        (bit_flip_fail, phase_flip_fail)
    }

    /// Executes full pipeline for error bitstrings: extract -> decode -> check logical error.
    #[inline(always)]
    pub fn run_trial(&self, error_x: u16, error_z: u16) -> bool {
        let (cx, cz) = self.decode(error_x, error_z);
        let (bf, pf) = self.evaluate_logical_error(error_x, error_z, cx, cz);
        bf || pf
    }

    /// Simulates single Monte Carlo trial under symmetric physical depolarizing noise `p_phys`.
    #[inline(always)]
    pub fn simulate_trial_depolarizing<R: rand::Rng>(&self, p_phys: f64, rng: &mut R) -> bool {
        let mut err_x = 0u16;
        let mut err_z = 0u16;

        for q in 0..9 {
            let r: f64 = rng.gen();
            if r < p_phys {
                let u = r / p_phys;
                if u < 1.0 / 3.0 {
                    err_x |= 1 << q;
                } else if u < 2.0 / 3.0 {
                    err_z |= 1 << q;
                } else {
                    err_x |= 1 << q;
                    err_z |= 1 << q;
                }
            }
        }

        self.run_trial(err_x, err_z)
    }

    /// Sweeps physical noise rates and compiles complete threshold scaling telemetry.
    pub fn monte_carlo_threshold_sweep(
        &self,
        p_values: &[f64],
        shots_per_point: usize,
        seed: u64,
    ) -> VnniThresholdReport {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use std::time::Instant;

        let backend = VnniBackend::detect();
        let mut points = Vec::with_capacity(p_values.len());
        let t_total = Instant::now();
        let mut total_shots = 0;

        for &p in p_values {
            let mut rng = ChaCha8Rng::seed_from_u64(seed ^ (p.to_bits()));
            let t0 = Instant::now();
            let mut failures = 0;

            for _ in 0..shots_per_point {
                if self.simulate_trial_depolarizing(p, &mut rng) {
                    failures += 1;
                }
            }
            let dur = t0.elapsed();
            total_shots += shots_per_point;

            let p_l = failures as f64 / shots_per_point as f64;
            let ratio = if p_l > 1e-12 { p / p_l } else { 1.0 / p };
            let lat_ns = dur.as_nanos() as f64 / shots_per_point as f64;

            points.push(VnniThresholdPoint {
                p_phys: p,
                shots: shots_per_point,
                logical_errors: failures,
                logical_error_rate: p_l,
                suppression_ratio: ratio,
                latency_ns_per_shot: lat_ns,
            });
        }

        let total_dur = t_total.elapsed().as_secs_f64();
        let throughput = (total_shots as f64 / total_dur) / 1e6;

        // Find threshold crossover: where p_l crosses p_phys from below to above
        let mut threshold = None;
        for i in 0..points.len().saturating_sub(1) {
            let p1 = points[i].p_phys;
            let pl1 = points[i].logical_error_rate;
            let p2 = points[i + 1].p_phys;
            let pl2 = points[i + 1].logical_error_rate;

            if pl1 <= p1 && pl2 >= p2 {
                let frac = (p1 - pl1) / ((pl2 - p2) + (p1 - pl1));
                threshold = Some(p1 + frac * (p2 - p1));
                break;
            }
        }

        let mut fit_a = 0.0;
        for pt in &points {
            if pt.logical_error_rate > 0.0 && pt.p_phys <= 0.02 {
                fit_a = pt.logical_error_rate / (pt.p_phys * pt.p_phys);
                break;
            }
        }

        VnniThresholdReport {
            points,
            empirical_threshold: threshold,
            fit_scaling_coefficient: fit_a,
            total_shots,
            total_duration_seconds: total_dur,
            throughput_mshots_sec: throughput,
            hardware_backend: backend.name().to_string(),
        }
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
        assert_eq!(symplectic_commutation_16q(1, 0, 0, 1), 1);
        assert_eq!(symplectic_commutation_16q(1, 0, 1, 0), 0);
        assert_eq!(symplectic_commutation_16q(0b01, 0b10, 0b10, 0b01), 0);
        assert_eq!(symplectic_commutation_16q(0b01, 0b10, 0b11, 0b00), 1);
    }

    #[test]
    fn test_surface_code_syndromes_all_single_qubit_errors() {
        let matrix = SurfaceCode9QVnniMatrix::new();
        let syn_clean = matrix.extract_syndromes_vnni(0, 0);
        assert_eq!(syn_clean, [1, 1, 1, 1, 1, 1, 1, 1]);

        let syn_x0 = matrix.extract_syndromes_vnni(1 << 0, 0);
        assert_eq!(syn_x0, [1, 1, -1, 1, 1, 1, 1, 1]);

        let syn_x4 = matrix.extract_syndromes_vnni(1 << 4, 0);
        assert_eq!(syn_x4, [-1, -1, 1, 1, 1, 1, 1, 1]);

        let syn_z1 = matrix.extract_syndromes_vnni(0, 1 << 1);
        assert_eq!(syn_z1, [1, 1, 1, 1, -1, 1, -1, 1]);

        let syn_y4 = matrix.extract_syndromes_vnni(1 << 4, 1 << 4);
        assert_eq!(syn_y4, [-1, -1, 1, 1, -1, -1, 1, 1]);
    }

    #[test]
    fn test_vnni_decoder_single_error_zero_logical_failure() {
        let decoder = VnniQecDecoder::new();
        for q in 0..9 {
            let failed = decoder.run_trial(1 << q, 0);
            assert!(!failed, "Single X error on qubit {q} caused logical failure!");
        }
        for q in 0..9 {
            let failed = decoder.run_trial(0, 1 << q);
            assert!(!failed, "Single Z error on qubit {q} caused logical failure!");
        }
        for q in 0..9 {
            let failed = decoder.run_trial(1 << q, 1 << q);
            assert!(!failed, "Single Y error on qubit {q} caused logical failure!");
        }
    }

    #[test]
    fn test_vnni_decoder_two_qubit_logical_string_detected() {
        let decoder = VnniQecDecoder::new();
        let failed = decoder.run_trial((1 << 0) | (1 << 3), 0);
        assert!(failed, "Two-qubit error along column 0 should trigger logical failure!");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vnni_unrolled_numerical_equivalence() {
        if is_x86_feature_detected!("avxvnni") {
            let mut a = [0u8; 32];
            let mut b = [0i8; 32];
            for i in 0..32 {
                a[i] = ((i * 3 + 1) % 15) as u8;
                b[i] = (((i * 7 + 2) % 11) as i8) - 5;
            }
            let iters = 80usize;
            let u8_syn = unsafe { _mm256_loadu_si256(a.as_ptr() as *const __m256i) };
            let i8_w = unsafe { _mm256_loadu_si256(b.as_ptr() as *const __m256i) };

            let mut serial = [0i32; 8];
            for _ in 0..iters {
                serial = vnni_dpbusd_32(serial, a, b);
            }

            let res_4x = unsafe { bench_vnni_unrolled_4x(u8_syn, i8_w, iters) };
            let res_8x = unsafe { bench_vnni_unrolled_8x(u8_syn, i8_w, iters) };

            assert_eq!(serial, res_4x, "4x unrolled VNNI output must match serial accumulation");
            assert_eq!(serial, res_8x, "8x unrolled VNNI output must match serial accumulation");
        }
    }
}



