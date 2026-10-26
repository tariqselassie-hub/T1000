// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # $\mathbb{Z}_7$ Heptal Qudit Stabilizer & $W_7$ Wheel Graph Syndrome Engine
//!
//! Generalizes stabilizer formalisms from qubits ($\mathbb{Z}_2$) to 7-state qudits ($\mathbb{Z}_7$ / $\mathbb{F}_7$):
//! - Generalized Weyl-Heisenberg Pauli operators: $X^a |j\rangle = |(j+a) \pmod 7\rangle$, $Z^b |j\rangle = \omega^{bj} |j\rangle$ with $\omega = e^{2\pi i / 7}$.
//! - Modulo 7 Symplectic Commutation: $\langle P_1, P_2 \rangle_7 = (\vec{u}_1 \cdot \vec{v}_2 - \vec{v}_1 \cdot \vec{u}_2) \pmod 7$.
//! - AVX-VNNI accelerated 32-way dot products with fast modulo 7 reduction.
//! - 7-Node Wheel Graph ($W_7$) Stabilizer Code: Central Hub $v_0$ + 6-node cyclic rim $v_1..v_6$.
//! - Exact single-qudit error decoding across all $7 \times 48 = 336$ Pauli error channels.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Single qudit Weyl-Heisenberg operator: P = X^a Z^b with a, b in {0..6}.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HeptalPauli {
    pub a: u8, // Shift power (Pauli-X) mod 7
    pub b: u8, // Clock power (Pauli-Z) mod 7
}

impl HeptalPauli {
    pub const IDENTITY: Self = Self { a: 0, b: 0 };

    #[inline(always)]
    pub fn new(a: u8, b: u8) -> Self {
        Self { a: a % 7, b: b % 7 }
    }

    #[inline(always)]
    pub fn is_identity(&self) -> bool {
        self.a == 0 && self.b == 0
    }
}

/// 16-qudit Pauli operator string over Z_7: P = X^{vec{u}} Z^{vec{v}}.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeptalPauliString16 {
    pub u: [u8; 16], // Shift powers a_0..a_15
    pub v: [u8; 16], // Phase powers b_0..b_15
}

impl Default for HeptalPauliString16 {
    fn default() -> Self {
        Self {
            u: [0; 16],
            v: [0; 16],
        }
    }
}

impl HeptalPauliString16 {
    pub fn new(u: [u8; 16], v: [u8; 16]) -> Self {
        let mut u_mod = [0u8; 16];
        let mut v_mod = [0u8; 16];
        for i in 0..16 {
            u_mod[i] = u[i] % 7;
            v_mod[i] = v[i] % 7;
        }
        Self { u: u_mod, v: v_mod }
    }
}

/// Computes the mod 7 symplectic inner product between two 16-qudit Pauli strings (Scalar).
/// Returns c in {0..6} where P1 P2 = omega^c P2 P1 with omega = exp(2*pi*i / 7).
#[inline]
pub fn symplectic_commutation_z7_scalar(p1: &HeptalPauliString16, p2: &HeptalPauliString16) -> u8 {
    let mut acc: i32 = 0;
    for k in 0..16 {
        let term1 = (p1.u[k] as i32) * (p2.v[k] as i32);
        let term2 = (p1.v[k] as i32) * (p2.u[k] as i32);
        acc += term1 - term2;
    }
    ((acc % 7 + 7) % 7) as u8
}

/// Hardware-accelerated mod 7 symplectic commutation for 16-qudit strings using AVX-VNNI.
/// Packs u1, v2, v1, u2 into a 32-way INT8 dot product pass and computes (u1*v2 - v1*u2) mod 7.
#[inline]
pub fn symplectic_commutation_z7_vnni(p1: &HeptalPauliString16, p2: &HeptalPauliString16) -> u8 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avxvnni") {
            unsafe {
                return symplectic_commutation_z7_native(p1, p2);
            }
        }
    }
    symplectic_commutation_z7_scalar(p1, p2)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn symplectic_commutation_z7_native(
    p1: &HeptalPauliString16,
    p2: &HeptalPauliString16,
) -> u8 {
    // Pack 16 elements of u1 (unsigned) and v1 (unsigned) into 32-byte vector
    let mut u1_v1 = [0u8; 32];
    // Pack 16 elements of +v2 (signed) and -u2 (signed) into 32-byte vector
    let mut v2_neg_u2 = [0i8; 32];

    for k in 0..16 {
        u1_v1[k] = p1.u[k];
        u1_v1[16 + k] = p1.v[k];

        v2_neg_u2[k] = p2.v[k] as i8;
        v2_neg_u2[16 + k] = -(p2.u[k] as i8);
    }

    let a_vec = _mm256_loadu_si256(u1_v1.as_ptr() as *const __m256i);
    let b_vec = _mm256_loadu_si256(v2_neg_u2.as_ptr() as *const __m256i);

    let mut out_accum = [0i32; 8];
    std::arch::asm!(
        "vpxor {acc}, {acc}, {acc}",
        "{{vex}} vpdpbusd {acc}, {a}, {b}",
        "vmovdqu [{out_ptr}], {acc}",
        acc = out(ymm_reg) _,
        a = in(ymm_reg) a_vec,
        b = in(ymm_reg) b_vec,
        out_ptr = in(reg) out_accum.as_mut_ptr(),
        options(nostack),
    );

    // Sum the 8 dwords: sum(u1*v2 - v1*u2)
    let total_sum = out_accum[0]
        + out_accum[1]
        + out_accum[2]
        + out_accum[3]
        + out_accum[4]
        + out_accum[5]
        + out_accum[6]
        + out_accum[7];

    ((total_sum % 7 + 7) % 7) as u8
}

/// Wheel Graph $W_7$ Heptal Stabilizer Code Engine.
///
/// 7 qudits arranged in a wheel graph topology:
/// - Node 0: Central Hub
/// - Nodes 1..6: Cyclic Rim ($1 - 2 - 3 - 4 - 5 - 6 - 1$)
///
/// Stabilizer generators K_v = X_v prod_{w in N(v)} Z_w:
/// - K_0 = X_0 * Z_1 * Z_2 * Z_3 * Z_4 * Z_5 * Z_6 (Hub Generator)
/// - K_i = X_i * Z_0 * Z_{i-1} * Z_{i+1} (Rim Generators for i in 1..6)
#[derive(Debug, Clone)]
pub struct Wheel7HeptalStabilizerMatrix {
    /// Generator X-powers: `x_powers[gen_idx][qudit_idx]` in `{0..6}`
    pub x_powers: [[u8; 7]; 7],
    /// Generator Z-powers: `z_powers[gen_idx][qudit_idx]` in `{0..6}`
    pub z_powers: [[u8; 7]; 7],
    /// Lookup table mapping 7-syndrome hash to (faulty_qudit, a_shift, b_clock)
    pub syndrome_lut: Vec<Option<(u8, u8, u8)>>,
}

impl Wheel7HeptalStabilizerMatrix {
    pub const SYNDROME_LUT_SIZE: usize = 7 * 7 * 7 * 7 * 7 * 7 * 7; // 7^7 = 823,543 entries

    /// Constructs the canonical $W_7$ Heptal wheel stabilizer matrix and precomputes the syndrome decoder LUT.
    pub fn new() -> Self {
        let mut x_powers = [[0u8; 7]; 7];
        let mut z_powers = [[0u8; 7]; 7];

        // Generator 0 (Hub): X_0 Z_1 Z_2 Z_3 Z_4 Z_5 Z_6
        x_powers[0][0] = 1;
        for rim in 1..=6 {
            z_powers[0][rim] = 1;
        }

        // Generators 1..6 (Rim): K_i = X_i Z_0 Z_{prev} Z_{next}
        for i in 1..=6 {
            x_powers[i][i] = 1;
            z_powers[i][0] = 1; // Connected to central hub

            let prev = if i == 1 { 6 } else { i - 1 };
            let next = if i == 6 { 1 } else { i + 1 };
            z_powers[i][prev] = 1;
            z_powers[i][next] = 1;
        }

        let mut lut = vec![None; Self::SYNDROME_LUT_SIZE];

        // Populate lookup table for all 7 qudits and all 48 non-trivial Pauli errors (a, b) in Z_7^2
        for q in 0..7 {
            for a in 0..7u8 {
                for b in 0..7u8 {
                    if a == 0 && b == 0 {
                        continue;
                    }
                    let mut err_x = [0u8; 7];
                    let mut err_z = [0u8; 7];
                    err_x[q] = a;
                    err_z[q] = b;

                    let syn = Self::compute_syndromes_scalar(&x_powers, &z_powers, &err_x, &err_z);
                    let hash = Self::syndrome_hash(&syn);
                    lut[hash] = Some((q as u8, a, b));
                }
            }
        }

        Self {
            x_powers,
            z_powers,
            syndrome_lut: lut,
        }
    }

    /// Hashes a 7-element Z_7 syndrome vector to a unique integer index in 0..7^7.
    #[inline(always)]
    pub fn syndrome_hash(syn: &[u8; 7]) -> usize {
        let mut h = 0usize;
        for &s in syn.iter().rev() {
            h = h * 7 + (s as usize);
        }
        h
    }

    /// Evaluates 7 stabilizer generator syndromes for an error `E = prod_j X_j^{err_x[j]} Z_j^{err_z[j]}`.
    /// Syndrome `s_k = <K_k, E>_7 = sum_j (X_k[j]*err_z[j] - Z_k[j]*err_x[j]) mod 7`.
    #[inline]
    pub fn compute_syndromes_scalar(
        x_p: &[[u8; 7]; 7],
        z_p: &[[u8; 7]; 7],
        err_x: &[u8; 7],
        err_z: &[u8; 7],
    ) -> [u8; 7] {
        let mut syn = [0u8; 7];
        for k in 0..7 {
            let mut acc: i32 = 0;
            for j in 0..7 {
                let term1 = (x_p[k][j] as i32) * (err_z[j] as i32);
                let term2 = (z_p[k][j] as i32) * (err_x[j] as i32);
                acc += term1 - term2;
            }
            syn[k] = ((acc % 7 + 7) % 7) as u8;
        }
        syn
    }

    /// Hardware-accelerated syndrome extraction for $W_7$ using AVX-VNNI.
    #[inline]
    pub fn extract_syndromes_vnni(&self, err_x: &[u8; 7], err_z: &[u8; 7]) -> [u8; 7] {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avxvnni") {
                unsafe {
                    return self.extract_syndromes_native(err_x, err_z);
                }
            }
        }
        Self::compute_syndromes_scalar(&self.x_powers, &self.z_powers, err_x, err_z)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn extract_syndromes_native(&self, err_x: &[u8; 7], err_z: &[u8; 7]) -> [u8; 7] {
        let mut syn = [0u8; 7];

        // Pack err_z into low 7 bytes, -err_x into next 7 bytes
        let mut err_packed = [0i8; 32];
        for j in 0..7 {
            err_packed[j] = err_z[j] as i8;
            err_packed[8 + j] = -(err_x[j] as i8);
        }
        let err_vec = _mm256_loadu_si256(err_packed.as_ptr() as *const __m256i);

        // Evaluate each of the 7 stabilizer generators
        for k in 0..7 {
            let mut gen_packed = [0u8; 32];
            for j in 0..7 {
                gen_packed[j] = self.x_powers[k][j];
                gen_packed[8 + j] = self.z_powers[k][j];
            }
            let gen_vec = _mm256_loadu_si256(gen_packed.as_ptr() as *const __m256i);

            let mut out = [0i32; 8];
            std::arch::asm!(
                "vpxor {acc}, {acc}, {acc}",
                "{{vex}} vpdpbusd {acc}, {g}, {e}",
                "vmovdqu [{out_ptr}], {acc}",
                acc = out(ymm_reg) _,
                g = in(ymm_reg) gen_vec,
                e = in(ymm_reg) err_vec,
                out_ptr = in(reg) out.as_mut_ptr(),
                options(nostack),
            );

            let sum = out[0] + out[1] + out[2] + out[3];
            syn[k] = ((sum % 7 + 7) % 7) as u8;
        }
        syn
    }

    /// Decodes a measured 7-element syndrome into the unique faulty physical qudit and (a, b) error.
    /// Returns Some((qudit_index, a_shift, b_clock)) or None if no error (all syndromes zero).
    #[inline(always)]
    pub fn decode_syndrome(&self, syn: &[u8; 7]) -> Option<(u8, u8, u8)> {
        let hash = Self::syndrome_hash(syn);
        self.syndrome_lut[hash]
    }
}

impl Default for Wheel7HeptalStabilizerMatrix {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symplectic_commutation_z7_basic() {
        let mut p1 = HeptalPauliString16::default();
        let mut p2 = HeptalPauliString16::default();

        // X0 and Z0 on qudit 0: [X, Z] has phase <X, Z>_7 = 1*1 - 0*0 = 1 mod 7
        p1.u[0] = 1;
        p2.v[0] = 1;
        assert_eq!(symplectic_commutation_z7_scalar(&p1, &p2), 1);
        assert_eq!(symplectic_commutation_z7_vnni(&p1, &p2), 1);

        // Z0 and X0 on qudit 0: <Z, X>_7 = 0*0 - 1*1 = -1 = 6 mod 7
        assert_eq!(symplectic_commutation_z7_scalar(&p2, &p1), 6);
        assert_eq!(symplectic_commutation_z7_vnni(&p2, &p1), 6);

        // Commuting operators: X0 and X1
        let mut p3 = HeptalPauliString16::default();
        p3.u[1] = 3;
        assert_eq!(symplectic_commutation_z7_scalar(&p1, &p3), 0);
        assert_eq!(symplectic_commutation_z7_vnni(&p1, &p3), 0);
    }

    #[test]
    fn test_wheel7_graph_stabilizers_strictly_commute() {
        let w7 = Wheel7HeptalStabilizerMatrix::new();

        // All 7 graph state stabilizers on W7 must commute with each other!
        for i in 0..7 {
            for j in 0..7 {
                let mut p_i = HeptalPauliString16::default();
                let mut p_j = HeptalPauliString16::default();
                for q in 0..7 {
                    p_i.u[q] = w7.x_powers[i][q];
                    p_i.v[q] = w7.z_powers[i][q];
                    p_j.u[q] = w7.x_powers[j][q];
                    p_j.v[q] = w7.z_powers[j][q];
                }
                let comm = symplectic_commutation_z7_vnni(&p_i, &p_j);
                assert_eq!(
                    comm, 0,
                    "W7 stabilizer generator K_{i} must commute with K_{j} in Z_7!"
                );
            }
        }
    }

    #[test]
    fn test_wheel7_all_single_qudit_errors_uniquely_decoded() {
        let w7 = Wheel7HeptalStabilizerMatrix::new();

        let mut errors_tested = 0usize;
        for q in 0..7u8 {
            for a in 0..7u8 {
                for b in 0..7u8 {
                    if a == 0 && b == 0 {
                        continue;
                    }
                    let mut err_x = [0u8; 7];
                    let mut err_z = [0u8; 7];
                    err_x[q as usize] = a;
                    err_z[q as usize] = b;

                    let syn = w7.extract_syndromes_vnni(&err_x, &err_z);
                    let decoded = w7.decode_syndrome(&syn);

                    assert_eq!(
                        decoded,
                        Some((q, a, b)),
                        "Failed to uniquely decode single-qudit error X^{a} Z^{b} on qudit {q}!"
                    );
                    errors_tested += 1;
                }
            }
        }
        assert_eq!(
            errors_tested, 336,
            "Must test exactly 7 * 48 = 336 single-qudit error channels!"
        );
    }
}
