// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Quantum Order Finding via Quantum Phase Estimation (QPE)
//!
//! Executes the full quantum circuit for order finding:
//! Hadamards on counting register, controlled modular exponentiation $C\text{-}U_a^{2^j}$,
//! inverse Quantum Fourier Transform (IQFT), and probability peak extraction.

use super::continued_fractions::ContinuedFractions;
use super::modular_arithmetic::{gcd, mod_pow, ModularMultiplier};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Telemetry and results of quantum order finding on $(a, N)$.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumOrderResult {
    pub a: u64,
    pub modulus: u64,
    pub counting_qubits: usize,
    pub work_qubits: usize,
    pub total_qubits: usize,
    pub order: Option<u64>,
    pub measured_phases: Vec<f64>,
    pub peak_probabilities: Vec<f64>,
    pub quantum_success: bool,
    pub circuit_micros: u64,
}

/// Quantum Order Finding Engine.
#[derive(Debug, Clone)]
pub struct QuantumOrderFinder {
    pub a: u64,
    pub modulus: u64,
    pub counting_qubits: usize,
}

impl QuantumOrderFinder {
    /// Creates a new quantum order finder for base $a$ and modulus $N$.
    pub fn new(a: u64, modulus: u64) -> Self {
        assert!(gcd(a, modulus) == 1, "Base 'a' must be coprime to N");
        // Counting register: at least 2 * ceil(log2(N)) or 6 qubits for small semi-primes
        let mut bits = 1;
        while (1u64 << bits) <= modulus {
            bits += 1;
        }
        let counting_qubits = (bits * 2).clamp(4, 8);
        Self {
            a,
            modulus,
            counting_qubits,
        }
    }

    /// Sets the number of precision counting qubits.
    pub fn with_counting_qubits(mut self, qubits: usize) -> Self {
        self.counting_qubits = qubits;
        self
    }

    /// Executes the quantum order finding circuit and extracts order $r$.
    pub fn find_order(&self) -> QuantumOrderResult {
        let t0 = std::time::Instant::now();
        let multiplier = ModularMultiplier::new(self.a, self.modulus);
        let m = self.counting_qubits;
        let nw = multiplier.work_qubits;
        let total_qubits = m + nw;
        let dim = 1 << total_qubits;

        let mut real = vec![0.0f64; dim];
        let mut imag = vec![0.0f64; dim];

        // 1. Initialize work register to |1> (x = 1) and counting register to |0>
        // State |00...0>|1> corresponds to index 1 (since work register is lowest bits or highest bits)
        // Layout: high m bits = counting register, low nw bits = work register
        let initial_idx = 1; // work register = 1, counting = 0
        real[initial_idx] = 1.0;

        // 2. Apply Hadamard gates H to each of the m counting qubits
        let inv_sqrt_2 = 1.0 / std::f64::consts::SQRT_2;
        for c in 0..m {
            let ctrl_wire = nw + c; // counting qubit wire
            let stride = 1 << ctrl_wire;
            for i in (0..dim).step_by(stride * 2) {
                for j in 0..stride {
                    let idx0 = i + j;
                    let idx1 = idx0 + stride;
                    let r0 = real[idx0];
                    let i0 = imag[idx0];
                    let r1 = real[idx1];
                    let i1 = imag[idx1];
                    real[idx0] = (r0 + r1) * inv_sqrt_2;
                    imag[idx0] = (i0 + i1) * inv_sqrt_2;
                    real[idx1] = (r0 - r1) * inv_sqrt_2;
                    imag[idx1] = (i0 - i1) * inv_sqrt_2;
                }
            }
        }

        // 3. Apply controlled modular exponentiation gates C-U_a^{2^c}
        for c in 0..m {
            let power = 1u64 << c;
            let effective_a = mod_pow(self.a, power, self.modulus);
            let ctrl_wire = nw + c;
            let ctrl_stride = 1 << ctrl_wire;

            // Apply permutation on work register when counting qubit c is 1
            for block_start in (0..dim).step_by(ctrl_stride * 2) {
                let ctrl_1_start = block_start + ctrl_stride;
                for chunk in 0..ctrl_stride {
                    let base = ctrl_1_start + chunk;
                    // work register is lowest nw bits
                    let work_val = base & ((1 << nw) - 1);
                    if (work_val as u64) < self.modulus {
                        let target_work = ((effective_a as u128 * work_val as u128)
                            % self.modulus as u128) as usize;
                        if target_work != work_val {
                            let target_base = (base & !((1 << nw) - 1)) | target_work;
                            if target_base > base {
                                let r_tmp = real[base];
                                let i_tmp = imag[base];
                                real[base] = real[target_base];
                                imag[base] = imag[target_base];
                                real[target_base] = r_tmp;
                                imag[target_base] = i_tmp;
                            }
                        }
                    }
                }
            }
        }

        // 4. Apply Inverse Quantum Fourier Transform (IQFT) on counting register
        for i in 0..m {
            let wire = nw + i;
            for j in 0..i {
                let wire_j = nw + j;
                let k = i - j + 1;
                let theta = -2.0 * PI / (1 << k) as f64;
                let c_rot = theta.cos();
                let s_rot = theta.sin();

                let stride_i = 1 << wire;
                let stride_j = 1 << wire_j;
                for idx in 0..dim {
                    if (idx & stride_i != 0) && (idx & stride_j != 0) {
                        let r = real[idx];
                        let im = imag[idx];
                        real[idx] = r * c_rot - im * s_rot;
                        imag[idx] = r * s_rot + im * c_rot;
                    }
                }
            }

            // Hadamard on wire
            let stride = 1 << wire;
            for block in (0..dim).step_by(stride * 2) {
                for offset in 0..stride {
                    let idx0 = block + offset;
                    let idx1 = idx0 + stride;
                    let r0 = real[idx0];
                    let i0 = imag[idx0];
                    let r1 = real[idx1];
                    let i1 = imag[idx1];
                    real[idx0] = (r0 + r1) * inv_sqrt_2;
                    imag[idx0] = (i0 + i1) * inv_sqrt_2;
                    real[idx1] = (r0 - r1) * inv_sqrt_2;
                    imag[idx1] = (i0 - i1) * inv_sqrt_2;
                }
            }
        }

        // 5. Marginalize probabilities over counting register states
        let counting_dim = 1 << m;
        let mut counting_probs = vec![0.0f64; counting_dim];
        for idx in 0..dim {
            let counting_val = idx >> nw;
            let prob = real[idx] * real[idx] + imag[idx] * imag[idx];
            counting_probs[counting_val] += prob;
        }

        // Find top peaks
        let mut indexed_probs: Vec<(usize, f64)> = counting_probs
            .iter()
            .copied()
            .enumerate()
            .collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut measured_phases = Vec::new();
        let mut peak_probabilities = Vec::new();
        let mut extracted_order = None;

        for (s, prob) in indexed_probs.iter().take(8) {
            let phase = *s as f64 / counting_dim as f64;
            measured_phases.push(phase);
            peak_probabilities.push(*prob);

            if extracted_order.is_none() && phase > 1e-6 {
                if let Some(r) = ContinuedFractions::extract_order(phase, self.a, self.modulus) {
                    extracted_order = Some(r);
                }
            }
        }

        let elapsed = t0.elapsed().as_micros() as u64;
        let quantum_success = extracted_order.is_some();

        QuantumOrderResult {
            a: self.a,
            modulus: self.modulus,
            counting_qubits: m,
            work_qubits: nw,
            total_qubits,
            order: extracted_order,
            measured_phases,
            peak_probabilities,
            quantum_success,
            circuit_micros: elapsed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_order_finding_15() {
        // Base a=7, N=15, order is r=4
        let finder = QuantumOrderFinder::new(7, 15).with_counting_qubits(6);
        let res = finder.find_order();
        assert!(res.quantum_success);
        assert_eq!(res.order, Some(4));
    }

    #[test]
    fn test_quantum_order_finding_21() {
        // Base a=2, N=21, 2^6 = 64 = 1 mod 21 => r=6
        let finder = QuantumOrderFinder::new(2, 21).with_counting_qubits(6);
        let res = finder.find_order();
        assert!(res.quantum_success);
        assert_eq!(res.order, Some(6));
    }
}
