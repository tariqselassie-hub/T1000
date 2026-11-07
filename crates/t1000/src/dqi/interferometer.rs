// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Decoded Quantum Interferometry (DQI) Quantum Core
//!
//! Implements multi-qudit statevector evolution over $(\mathbb{C}^p)^{\otimes n}$,
//! discrete quantum Fourier transform tensor passes $(F_p)^{\otimes n}$, syndrome
//! phase oracles, and constructive interference peak extraction for Optimal
//! Polynomial Intersection (OPI) and Reed-Solomon decoding.

use super::field::FieldElement;
use super::reed_solomon::ReedSolomonCode;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Quantum qudit statevector over $(\mathbb{C}^p)^{\otimes n}$.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DqiStatevector {
    /// Field characteristic / qudit dimension $p$.
    pub p: usize,
    /// Number of qudits $n$.
    pub n: usize,
    /// Total Hilbert space dimension $N = p^n$.
    pub dim: usize,
    /// Real components of amplitudes.
    pub real: Vec<f64>,
    /// Imaginary components of amplitudes.
    pub imag: Vec<f64>,
}

impl DqiStatevector {
    /// Creates a statevector initialized to $|0\rangle^{\otimes n}$.
    pub fn new_zero(p: usize, n: usize) -> Self {
        let dim = p.pow(n as u32);
        let mut real = vec![0.0; dim];
        let imag = vec![0.0; dim];
        real[0] = 1.0;
        Self { p, n, dim, real, imag }
    }

    /// Creates a statevector in uniform superposition $\frac{1}{\sqrt{p^n}} \sum_{v} |v\rangle$.
    pub fn new_uniform(p: usize, n: usize) -> Self {
        let dim = p.pow(n as u32);
        let inv_sqrt_dim = 1.0 / (dim as f64).sqrt();
        let real = vec![inv_sqrt_dim; dim];
        let imag = vec![0.0; dim];
        Self { p, n, dim, real, imag }
    }

    /// Total state norm squared $\sum_i |a_i|^2$.
    pub fn norm_squared(&self) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.dim {
            sum += self.real[i] * self.real[i] + self.imag[i] * self.imag[i];
        }
        sum
    }

    /// Normalizes statevector to unit norm.
    pub fn normalize(&mut self) {
        let norm = self.norm_squared().sqrt();
        if norm > 1e-15 {
            let inv_norm = 1.0 / norm;
            for i in 0..self.dim {
                self.real[i] *= inv_norm;
                self.imag[i] *= inv_norm;
            }
        }
    }

    /// Converts computational basis index to $n$-length digit vector in $\mathbb{F}_p^n$.
    #[inline(always)]
    pub fn index_to_digits(&self, mut index: usize) -> Vec<usize> {
        let mut digits = vec![0usize; self.n];
        for d in digits.iter_mut() {
            *d = index % self.p;
            index /= self.p;
        }
        digits
    }

    /// Converts $n$-length digit vector in $\mathbb{F}_p^n$ to basis index.
    #[inline(always)]
    pub fn digits_to_index(&self, digits: &[usize]) -> usize {
        let mut index = 0usize;
        let mut stride = 1usize;
        for &d in digits.iter() {
            index += d * stride;
            stride *= self.p;
        }
        index
    }

    /// Applies the 1-qudit Discrete Fourier Transform $F_p$ (or $F_p^\dagger$) on target wire.
    ///
    /// $$(F_p)_{a, b} = \frac{1}{\sqrt{p}} e^{\pm 2\pi i (a \cdot b) / p}$$
    pub fn apply_qft_wire(&mut self, wire: usize, inverse: bool) {
        assert!(wire < self.n, "Wire index out of range");
        let p = self.p;
        let stride = p.pow(wire as u32);
        let chunk_size = stride * p;
        let sign = if inverse { -1.0 } else { 1.0 };
        let inv_sqrt_p = 1.0 / (p as f64).sqrt();

        // Precompute F_p matrix elements
        let mut f_real = vec![vec![0.0f64; p]; p];
        let mut f_imag = vec![vec![0.0f64; p]; p];
        for a in 0..p {
            for b in 0..p {
                let angle = sign * 2.0 * PI * ((a * b) as f64) / (p as f64);
                f_real[a][b] = angle.cos() * inv_sqrt_p;
                f_imag[a][b] = angle.sin() * inv_sqrt_p;
            }
        }

        let mut scratch_real = vec![0.0f64; p];
        let mut scratch_imag = vec![0.0f64; p];

        for chunk_start in (0..self.dim).step_by(chunk_size) {
            for offset in 0..stride {
                // Load p amplitudes along this wire stride
                for k in 0..p {
                    let idx = chunk_start + k * stride + offset;
                    scratch_real[k] = self.real[idx];
                    scratch_imag[k] = self.imag[idx];
                }

                // Matrix-vector product with F_p
                for a in 0..p {
                    let mut sum_r = 0.0f64;
                    let mut sum_i = 0.0f64;
                    for b in 0..p {
                        let ar = f_real[a][b];
                        let ai = f_imag[a][b];
                        let br = scratch_real[b];
                        let bi = scratch_imag[b];
                        sum_r += ar * br - ai * bi;
                        sum_i += ar * bi + ai * br;
                    }
                    let idx = chunk_start + a * stride + offset;
                    self.real[idx] = sum_r;
                    self.imag[idx] = sum_i;
                }
            }
        }
    }

    /// Applies the full $n$-qudit Quantum Fourier Transform $(F_p)^{\otimes n}$.
    pub fn apply_full_qft(&mut self, inverse: bool) {
        for wire in 0..self.n {
            self.apply_qft_wire(wire, inverse);
        }
    }

    /// Applies agreement phase oracle with received word $\vec{y}$:
    ///
    /// $$U_{y}(\phi) |v\rangle = e^{i \phi \cdot \text{agr}(v, y)} |v\rangle$$
    pub fn apply_agreement_oracle(&mut self, y: &[FieldElement], phi: f64) {
        let y_vals: Vec<usize> = y.iter().map(|e| e.value as usize).collect();

        for i in 0..self.dim {
            let digits = self.index_to_digits(i);
            let mut agreement_count = 0usize;
            for j in 0..self.n {
                if digits[j] == y_vals[j] {
                    agreement_count += 1;
                }
            }

            if agreement_count > 0 {
                let total_angle = (agreement_count as f64) * phi;
                let c = total_angle.cos();
                let s = total_angle.sin();
                let r = self.real[i];
                let im = self.imag[i];
                self.real[i] = r * c - im * s;
                self.imag[i] = r * s + im * c;
            }
        }
    }

    /// Applies parity-check syndrome phase kickback for Reed-Solomon code $C$:
    ///
    /// If $H v = \vec{0} \pmod p$ (i.e. $v \in C$), applies phase $\theta_0$.
    /// Otherwise applies phase $\theta_1$.
    pub fn apply_syndrome_phase(&mut self, code: &ReedSolomonCode, theta_codeword: f64, theta_noncode: f64) {
        let c_code = theta_codeword.cos();
        let s_code = theta_codeword.sin();
        let c_non = theta_noncode.cos();
        let s_non = theta_noncode.sin();

        for i in 0..self.dim {
            let digits = self.index_to_digits(i);
            let vec_fe: Vec<FieldElement> = digits
                .iter()
                .map(|&d| FieldElement::new(d as u32, self.p as u32))
                .collect();

            let is_codeword = code.syndrome(&vec_fe).iter().all(|s| s.value == 0);
            let (c, s) = if is_codeword {
                (c_code, s_code)
            } else {
                (c_non, s_non)
            };

            let r = self.real[i];
            let im = self.imag[i];
            self.real[i] = r * c - im * s;
            self.imag[i] = r * s + im * c;
        }
    }

    /// Applies dual-space syndrome phase kickback for dual code $C^\perp$:
    ///
    /// If $G w = \vec{0} \pmod p$ (i.e. $w \in C^\perp$), applies phase $\theta_0$.
    /// Otherwise applies phase $\theta_1$.
    pub fn apply_dual_syndrome_phase(&mut self, code: &ReedSolomonCode, theta_dual: f64, theta_nondual: f64) {
        let c_dual = theta_dual.cos();
        let s_dual = theta_dual.sin();
        let c_non = theta_nondual.cos();
        let s_non = theta_nondual.sin();

        for i in 0..self.dim {
            let digits = self.index_to_digits(i);
            let vec_fe: Vec<FieldElement> = digits
                .iter()
                .map(|&d| FieldElement::new(d as u32, self.p as u32))
                .collect();

            // w in C_perp <=> G w = 0
            let mut is_dual = true;
            for row in &code.generator_matrix {
                let dot = super::field::field_dot(row, &vec_fe);
                if dot.value != 0 {
                    is_dual = false;
                    break;
                }
            }

            let (c, s) = if is_dual {
                (c_dual, s_dual)
            } else {
                (c_non, s_non)
            };

            let r = self.real[i];
            let im = self.imag[i];
            self.real[i] = r * c - im * s;
            self.imag[i] = r * s + im * c;
        }
    }

    /// Calculates probability of measuring basis state $|i\rangle$.
    #[inline(always)]
    pub fn probability(&self, index: usize) -> f64 {
        self.real[index] * self.real[index] + self.imag[index] * self.imag[index]
    }

    /// Computes full measurement probability distribution over all basis states.
    pub fn probability_distribution(&self) -> Vec<f64> {
        (0..self.dim).map(|i| self.probability(i)).collect()
    }
}

/// Decoded Quantum Interferometry engine configuration and solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DqiInterferometer {
    /// Reed-Solomon code instance.
    pub code: ReedSolomonCode,
    /// Agreement phase parameter $\phi$.
    pub phi: f64,
    /// Dual syndrome phase parameter $\theta$.
    pub theta: f64,
}

/// DQI Interferometry Execution Report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DqiInterferenceReport {
    /// Field characteristic $p$.
    pub p: u32,
    /// Codeword length $n$.
    pub n: usize,
    /// Message length $k$.
    pub k: usize,
    /// Number of injected errors $t$.
    pub errors: usize,
    /// Classical unique decoding limit $\lfloor(n-k)/2\rfloor$.
    pub classical_unique_bound: usize,
    /// Whether classical Berlekamp-Welch successfully decoded.
    pub classical_bw_success: bool,
    /// Whether DQI top peak recovered planted polynomial.
    pub dqi_success: bool,
    /// Planted codeword probability under DQI.
    pub planted_codeword_prob: f64,
    /// Uniform baseline probability $1 / p^k$.
    pub uniform_code_baseline: f64,
    /// Interference amplification ratio $P(c^*) / (1 / p^k)$.
    pub amplification_factor: f64,
    /// Recovered polynomial coefficients.
    pub recovered_polynomial: Vec<u32>,
    /// Agreement of top recovered candidate with received word.
    pub candidate_agreement: usize,
}

impl DqiInterferometer {
    /// Creates a new DQI interferometer for the given Reed-Solomon code.
    pub fn new(code: ReedSolomonCode) -> Self {
        Self {
            code,
            phi: PI / 3.0,
            theta: PI,
        }
    }

    /// Sets agreement phase angle $\phi$.
    pub fn with_phi(mut self, phi: f64) -> Self {
        self.phi = phi;
        self
    }

    /// Sets dual phase angle $\theta$.
    pub fn with_theta(mut self, theta: f64) -> Self {
        self.theta = theta;
        self
    }

    /// Runs DQI decoding on a received vector $\vec{y}$:
    ///
    /// 1. Initialize uniform superposition $|\psi\rangle = \frac{1}{\sqrt{p^n}} \sum |v\rangle$.
    /// 2. Apply agreement phase oracle $U_y(\phi)$.
    /// 3. Apply QFT $(F_p)^{\otimes n}$ into dual space.
    /// 4. Apply dual syndrome phase filter $U_{C^\perp}(\theta)$.
    /// 5. Apply inverse QFT $(F_p^\dagger)^{\otimes n}$ returning to primal space.
    /// 6. Extract probabilities on the code subspace $C$ and return top candidate.
    pub fn run_dqi(
        &self,
        received_y: &[FieldElement],
        planted_codeword: &[FieldElement],
        planted_poly: &[FieldElement],
        num_errors: usize,
    ) -> DqiInterferenceReport {
        let p = self.code.p as usize;
        let n = self.code.n;
        let k = self.code.k;
        let classical_bound = self.code.unique_decoding_radius();

        // 1. Classical Berlekamp-Welch baseline
        let bw_result = self.code.decode_berlekamp_welch(received_y);
        let classical_bw_success = if let Some(poly) = bw_result {
            poly == planted_poly
        } else {
            false
        };

        // 2. Quantum DQI circuit execution
        let mut state = DqiStatevector::new_uniform(p, n);

        // Step 1: Agreement phase oracle
        state.apply_agreement_oracle(received_y, self.phi);

        // Step 2: Fourier transform to dual space
        state.apply_full_qft(false);

        // Step 3: Dual syndrome phase kickback
        state.apply_dual_syndrome_phase(&self.code, self.theta, 0.0);

        // Step 4: Inverse Fourier transform back to primal basis
        state.apply_full_qft(true);

        // Step 5: Normalize and evaluate probabilities on valid codewords
        state.normalize();

        // Find index of planted codeword
        let planted_digits: Vec<usize> = planted_codeword
            .iter()
            .map(|fe| fe.value as usize)
            .collect();
        let planted_idx = state.digits_to_index(&planted_digits);
        let planted_codeword_prob = state.probability(planted_idx);

        // Evaluate all codewords in C and rank by probability
        let all_polys = self.code.all_polynomials();
        let mut best_poly = all_polys[0].clone();
        let mut best_prob = -1.0f64;
        let mut total_code_prob = 0.0f64;

        for poly in &all_polys {
            let cw = self.code.encode(poly);
            let cw_digits: Vec<usize> = cw.iter().map(|fe| fe.value as usize).collect();
            let idx = state.digits_to_index(&cw_digits);
            let prob = state.probability(idx);
            total_code_prob += prob;
            if prob > best_prob {
                best_prob = prob;
                best_poly = poly.clone();
            }
        }

        let dqi_success = best_poly == planted_poly;
        let uniform_code_baseline = 1.0 / (all_polys.len() as f64);
        let conditional_prob = if total_code_prob > 1e-15 {
            planted_codeword_prob / total_code_prob
        } else {
            0.0
        };
        let amplification_factor = if uniform_code_baseline > 0.0 {
            conditional_prob / uniform_code_baseline
        } else {
            0.0
        };

        let best_cw = self.code.encode(&best_poly);
        let mut candidate_agreement = 0usize;
        for i in 0..n {
            if best_cw[i] == received_y[i] {
                candidate_agreement += 1;
            }
        }

        DqiInterferenceReport {
            p: self.code.p,
            n,
            k,
            errors: num_errors,
            classical_unique_bound: classical_bound,
            classical_bw_success,
            dqi_success,
            planted_codeword_prob,
            uniform_code_baseline,
            amplification_factor,
            recovered_polynomial: best_poly.iter().map(|fe| fe.value).collect(),
            candidate_agreement,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qft_unitarity_single_wire() {
        let p = 5;
        let n = 2;
        let mut state = DqiStatevector::new_zero(p, n);
        let initial_norm = state.norm_squared();
        assert!((initial_norm - 1.0).abs() < 1e-12);

        // Apply QFT then inverse QFT
        state.apply_qft_wire(0, false);
        assert!((state.norm_squared() - 1.0).abs() < 1e-12);

        state.apply_qft_wire(0, true);
        assert!((state.norm_squared() - 1.0).abs() < 1e-12);

        // Should return to |0>
        assert!((state.real[0] - 1.0).abs() < 1e-12);
        for i in 1..state.dim {
            assert!(state.real[i].abs() < 1e-12);
            assert!(state.imag[i].abs() < 1e-12);
        }
    }

    #[test]
    fn test_full_qft_roundtrip() {
        let p = 3;
        let n = 3;
        let mut state = DqiStatevector::new_uniform(p, n);
        let norm0 = state.norm_squared();
        assert!((norm0 - 1.0).abs() < 1e-12);

        state.apply_full_qft(false);
        assert!((state.norm_squared() - 1.0).abs() < 1e-12);

        state.apply_full_qft(true);
        assert!((state.norm_squared() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_dqi_constructive_interference_beyond_unique_radius() {
        // [n=5, k=2] code over F_7:
        let p = 7;
        let n = 5;
        let k = 2;
        let code = ReedSolomonCode::new(p, n, k);

        // Planted polynomial: P(x) = 3 + 2x
        let planted_poly = vec![FieldElement::new(3, p), FieldElement::new(2, p)];
        let planted_codeword = code.encode(&planted_poly);

        // Corrupt at 2 positions: index 1 and 3
        let mut received = planted_codeword.clone();
        received[1] = received[1].add(FieldElement::new(2, p));
        received[3] = received[3].add(FieldElement::new(4, p));

        println!("Planted poly: {:?}", planted_poly);
        println!("Planted codeword: {:?}", planted_codeword);
        println!("Received vector: {:?}", received);

        let interferometer = DqiInterferometer::new(code.clone())
            .with_phi(PI / 2.5)
            .with_theta(PI);

        let report = interferometer.run_dqi(&received, &planted_codeword, &planted_poly, 2);
        println!("Report: {:#?}", report);

        // Check top 5 states in the statevector
        let mut state = DqiStatevector::new_uniform(p as usize, n);
        state.apply_agreement_oracle(&received, PI / 2.5);
        state.apply_full_qft(false);
        state.apply_dual_syndrome_phase(&code, PI, 0.0);
        state.apply_full_qft(true);
        state.normalize();

        let mut indexed_probs: Vec<(usize, f64)> = (0..state.dim)
            .map(|i| (i, state.probability(i)))
            .collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("Top 5 states after DQI:");
        for idx in 0..5 {
            let (state_idx, prob) = indexed_probs[idx];
            let digits = state.index_to_digits(state_idx);
            let is_cw = code.is_codeword(&digits.iter().map(|&d| FieldElement::new(d as u32, p)).collect::<Vec<_>>());
            println!("  Rank {}: idx={}, digits={:?}, prob={:.6}, is_codeword={}", idx, state_idx, digits, prob, is_cw);
        }

        assert!(!report.classical_bw_success);
        assert!(report.dqi_success, "DQI must recover the planted polynomial");
        assert!(
            report.amplification_factor > 1.0,
            "DQI must achieve constructive interference amplification > 1.0"
        );
    }
}
