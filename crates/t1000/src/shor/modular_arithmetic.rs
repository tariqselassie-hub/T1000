// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Reversible Modular Arithmetic & Unitary Multipliers
//!
//! Provides reversible modular multiplication $U_a |x\rangle = |a \cdot x \pmod N\rangle$
//! and modular exponentiation for quantum order finding in Shor's algorithm.

use serde::{Deserialize, Serialize};

/// Computes greatest common divisor via Euclid's algorithm.
#[inline(always)]
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Computes $(base^{exp}) \pmod{modulus}$ in $O(\log exp)$ operations.
pub fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    assert!(modulus > 1, "Modulus must be > 1");
    if modulus == 1 {
        return 0;
    }
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp % 2 == 1 {
            result = ((result as u128 * base as u128) % modulus as u128) as u64;
        }
        exp /= 2;
        base = ((base as u128 * base as u128) % modulus as u128) as u64;
    }
    result
}

/// Classical multiplicative order $r = \text{ord}_N(a)$ such that $a^r \equiv 1 \pmod N$.
pub fn classical_multiplicative_order(a: u64, modulus: u64) -> Option<u64> {
    if gcd(a, modulus) != 1 {
        return None;
    }
    let mut current = a % modulus;
    let mut r = 1u64;
    while current != 1 {
        current = ((current as u128 * a as u128) % modulus as u128) as u64;
        r += 1;
        if r > modulus {
            return None;
        }
    }
    Some(r)
}

/// Reversible modular multiplication operator $U_a |x\rangle = |a \cdot x \pmod N\rangle$.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModularMultiplier {
    /// Multiplier base $a$ ($\gcd(a, N) = 1$).
    pub a: u64,
    /// Semi-prime modulus $N$.
    pub modulus: u64,
    /// Number of work register qubits ($2^{n_{\text{work}}} > N$).
    pub work_qubits: usize,
    /// Work register dimension $2^{n_{\text{work}}}$.
    pub work_dim: usize,
}

impl ModularMultiplier {
    /// Creates a new modular multiplier operator $U_a$ modulo $N$.
    pub fn new(a: u64, modulus: u64) -> Self {
        assert!(modulus >= 3, "Modulus N must be at least 3");
        assert!(gcd(a, modulus) == 1, "Base 'a' must be coprime to N");

        let mut work_qubits = 1;
        while (1u64 << work_qubits) <= modulus {
            work_qubits += 1;
        }
        let work_dim = 1 << work_qubits;

        Self {
            a: a % modulus,
            modulus,
            work_qubits,
            work_dim,
        }
    }

    /// Evaluates the classical permutation action $x \mapsto a \cdot x \pmod N$.
    #[inline(always)]
    pub fn permute(&self, x: usize) -> usize {
        let x_u64 = x as u64;
        if x_u64 < self.modulus {
            ((self.a as u128 * x_u64 as u128) % self.modulus as u128) as usize
        } else {
            // Identity on states outside computational subspace
            x
        }
    }

    /// Applies controlled power $C\text{-}U_a^{2^j}$ to a joint statevector:
    ///
    /// State shape: $2^{m_{\text{counting}}} \times 2^{n_{\text{work}}}$.
    pub fn apply_controlled_power(
        &self,
        power_2j: u64,
        state_real: &mut [f64],
        state_imag: &mut [f64],
        ctrl_qubit: usize,
        total_qubits: usize,
    ) {
        let effective_a = mod_pow(self.a, power_2j, self.modulus);
        let ctrl_stride = 1 << ctrl_qubit;
        let dim = 1 << total_qubits;

        let mut scratch_real = vec![0.0f64; self.work_dim];
        let mut scratch_imag = vec![0.0f64; self.work_dim];

        // Iterate over blocks where control qubit is |1>
        for block_start in (0..dim).step_by(ctrl_stride * 2) {
            let ctrl_1_start = block_start + ctrl_stride;

            for chunk_offset in 0..ctrl_stride {
                let base_idx = ctrl_1_start + chunk_offset;
                if base_idx >= dim {
                    continue;
                }

                // Copy current work amplitudes
                for w in 0..self.work_dim {
                    let idx = base_idx + w * (dim / self.work_dim);
                    if idx < dim {
                        scratch_real[w] = state_real[idx];
                        scratch_imag[w] = state_imag[idx];
                    }
                }

                // Permute work amplitudes by x -> (effective_a * x) mod N
                for w in 0..self.work_dim {
                    let target_w = if (w as u64) < self.modulus {
                        ((effective_a as u128 * w as u128) % self.modulus as u128) as usize
                    } else {
                        w
                    };
                    let src_idx = base_idx + w * (dim / self.work_dim);
                    let dst_idx = base_idx + target_w * (dim / self.work_dim);
                    if src_idx < dim && dst_idx < dim {
                        state_real[dst_idx] = scratch_real[w];
                        state_imag[dst_idx] = scratch_imag[w];
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcd_and_mod_pow() {
        assert_eq!(gcd(15, 7), 1);
        assert_eq!(gcd(15, 5), 5);
        assert_eq!(gcd(21, 14), 7);
        assert_eq!(mod_pow(2, 4, 15), 1); // 2^4 = 16 = 1 mod 15
        assert_eq!(mod_pow(7, 4, 15), 1); // 7^4 = 2401 = 1 mod 15
    }

    #[test]
    fn test_classical_order() {
        assert_eq!(classical_multiplicative_order(2, 15), Some(4));
        assert_eq!(classical_multiplicative_order(7, 15), Some(4));
        assert_eq!(classical_multiplicative_order(4, 15), Some(2));
        assert_eq!(classical_multiplicative_order(2, 21), Some(6)); // 2^6 = 64 = 1 mod 21
    }

    #[test]
    fn test_modular_multiplier_permutation() {
        let mult = ModularMultiplier::new(7, 15);
        assert_eq!(mult.work_qubits, 4); // 2^4 = 16 > 15
        assert_eq!(mult.work_dim, 16);

        // x=1 -> 7*1 = 7 mod 15
        assert_eq!(mult.permute(1), 7);
        // x=7 -> 7*7 = 49 = 4 mod 15
        assert_eq!(mult.permute(7), 4);
        // x=4 -> 7*4 = 28 = 13 mod 15
        assert_eq!(mult.permute(4), 13);
        // x=13 -> 7*13 = 91 = 1 mod 15 (order 4 cycle)
        assert_eq!(mult.permute(13), 1);
        // Non-computational basis state x=15 -> unchanged
        assert_eq!(mult.permute(15), 15);
    }
}

