// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Semiclassical 1-Qubit Adaptive Zipper Factoring Engine
//!
//! Implements sliding-window semiclassical phase estimation (Griffiths-Niu / Kitaev)
//! combined with the T-1000 Bloch-zipper discipline:
//!
//! - **Constant Counting Register:** Exactly 1 counting qubit recycled across $m$ phase steps.
//! - **Sliding Active Window:** Total statevector $Q = n_{\text{work}} + 1$ qubits resides
//!   in CPU L1/L2/L3 cache without allocating $2^{m + n_w}$ statevectors.
//! - **Adaptive Phase Feedforward:** Classical bit measurement $b_k$ drives phase
//!   corrections $\theta_j = 2\pi \sum_{k > j} b_k 2^{-(k - j + 1)}$ without quantum branching.
//! - **Zero Allocation Hot-Loop:** Work register and scratch amplitude buffers are pre-allocated
//!   once and reused across all phase steps.

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use super::continued_fractions::ContinuedFractions;
use super::modular_arithmetic::{gcd, mod_pow};

/// Simple xorshift64 PRNG for deterministic, reproducible quantum state collapse.
#[derive(Debug, Clone)]
pub struct ZipperRng {
    state: u64,
}

impl ZipperRng {
    /// Creates a new PRNG with non-zero seed.
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x5EED_0042 } else { seed },
        }
    }

    /// Generates uniformly distributed float in $[0, 1)$.
    #[inline(always)]
    pub fn next_f64(&mut self) -> f64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state & 0x001F_FFFF_FFFF_FFFF) as f64 / (0x0020_0000_0000_0000u64 as f64)
    }
}

/// Telemetry recorded for each bit along the adaptive phase estimation zipper.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZipperStepTelemetry {
    pub step_index: usize,
    pub power_2j: usize,
    pub effective_a: u64,
    pub z_expectation: f64,
    pub p0: f64,
    pub p1: f64,
    pub coherence: f64,
    pub entropy_nats: f64,
    pub adaptive_angle_rad: f64,
    pub measured_bit: u8,
}

/// Comprehensive report emitted after a semiclassical zipper factoring pass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZipperFactoringReport {
    pub composite_n: u64,
    pub digit_count_n: usize,
    pub factor_p: Option<u64>,
    pub factor_q: Option<u64>,
    pub digit_count_p: usize,
    pub base_a: u64,
    pub order_r: Option<u64>,
    pub work_qubits: usize,
    pub counting_qubits: usize,
    pub total_active_qubits: usize,
    pub active_state_bytes: usize,
    pub cache_tier: String,
    pub mean_coherence: f64,
    pub min_coherence: f64,
    pub estimated_phase: f64,
    pub factoring_verified: bool,
    pub wall_time_micros: u64,
    pub steps: Vec<ZipperStepTelemetry>,
}

/// Semiclassical Adaptive Zipper Factoring Engine.
#[derive(Debug, Clone)]
pub struct ZipperShorFactorizer;

impl ZipperShorFactorizer {
    /// Determines the physical cache placement tier for given memory footprint.
    pub fn classify_cache_tier(bytes: usize) -> String {
        if bytes <= 64 * 1024 {
            "L1 Data Cache (32-64 KB)".to_string()
        } else if bytes <= 1024 * 1024 {
            "L2 Cache (512-1024 KB)".to_string()
        } else if bytes <= 36 * 1024 * 1024 {
            "L3 Cache (24-36 MB)".to_string()
        } else {
            "Main DRAM (Bus Bottleneck)".to_string()
        }
    }

    /// Factors composite integer $N$ using the sliding 1-qubit adaptive zipper.
    pub fn factor(modulus: u64, seed: u64) -> Option<ZipperFactoringReport> {
        assert!(modulus >= 4, "Modulus must be >= 4");
        if modulus % 2 == 0 {
            return None;
        }

        let candidate_bases = [2u64, 7, 3, 5, 11, 13, 17, 19, 23, 29];
        for &a in &candidate_bases {
            if a >= modulus {
                continue;
            }
            if gcd(a, modulus) != 1 {
                continue;
            }
            let report = Self::run_single_attempt(a, modulus, seed);
            if report.factoring_verified {
                return Some(report);
            }
        }
        None
    }

    /// Executes single factoring pass with specific base $a$ and modulus $N$.
    pub fn run_single_attempt(a: u64, modulus: u64, seed: u64) -> ZipperFactoringReport {
        let t0 = std::time::Instant::now();
        let mut nw = 1;
        while (1u64 << nw) <= modulus {
            nw += 1;
        }
        let num_bits = nw * 2;
        let work_dim = 1 << nw;
        let total_dim = work_dim * 2;
        let active_bytes = total_dim * 16;
        let cache_tier = Self::classify_cache_tier(active_bytes);

        let mut real = vec![0.0f64; total_dim];
        let mut imag = vec![0.0f64; total_dim];
        let mut scratch_r = vec![0.0f64; work_dim];
        let mut scratch_i = vec![0.0f64; work_dim];

        // State layout: counting qubit is wire 0, work register is wires 1..=nw
        // Initial state: counting = 0, work = 1 -> index = (1 << 1) | 0 = 2
        real[2] = 1.0;

        let mut rng = ZipperRng::new(seed);
        let mut measured_bits = vec![0u8; num_bits];
        let mut steps = Vec::with_capacity(num_bits);
        let mut coh_sum = 0.0f64;
        let mut min_coh = 1.0f64;

        for j in (0..num_bits).rev() {
            Self::apply_counting_hadamard(&mut real, &mut imag, work_dim);
            Self::apply_controlled_modular_power(
                &mut real,
                &mut imag,
                &mut scratch_r,
                &mut scratch_i,
                a,
                j,
                modulus,
                work_dim,
            );
            let theta = Self::compute_adaptive_angle(&measured_bits, j, num_bits);
            Self::apply_adaptive_phase(&mut real, &mut imag, theta, work_dim);
            Self::apply_counting_hadamard(&mut real, &mut imag, work_dim);

            let (p0, p1, z_exp, coh, entropy) =
                Self::measure_counting_qubit(&real, &imag, work_dim);
            coh_sum += coh;
            if coh < min_coh {
                min_coh = coh;
            }

            let roll = rng.next_f64();
            let bit = if roll < p0 { 0u8 } else { 1u8 };
            measured_bits[j] = bit;

            let power_2j = 1usize << j.min(63);
            let effective_a = mod_pow(a, power_2j as u64, modulus);

            steps.push(ZipperStepTelemetry {
                step_index: j,
                power_2j: j,
                effective_a,
                z_expectation: z_exp,
                p0,
                p1,
                coherence: coh,
                entropy_nats: entropy,
                adaptive_angle_rad: theta,
                measured_bit: bit,
            });

            Self::epoxy_collapse_and_reset(&mut real, &mut imag, bit, p0, p1, work_dim);
        }

        let mut phase = 0.0f64;
        for (k, &b) in measured_bits.iter().enumerate() {
            if b == 1 {
                phase += (0.5f64).powi((k + 1) as i32);
            }
        }

        let mean_coh = coh_sum / (num_bits as f64);
        let order_r = if phase > 1e-8 {
            ContinuedFractions::extract_order(phase, a, modulus)
        } else {
            None
        };

        let mut factor_p = None;
        let mut factor_q = None;
        let mut verified = false;

        if let Some(r) = order_r {
            if r % 2 == 0 {
                let half_r = r / 2;
                let x = mod_pow(a, half_r, modulus);
                if x != modulus - 1 && x != 1 {
                    let p_cand = gcd(x.abs_diff(1), modulus);
                    let q_cand = modulus / p_cand;
                    if p_cand > 1 && q_cand > 1 && p_cand * q_cand == modulus {
                        factor_p = Some(p_cand.min(q_cand));
                        factor_q = Some(p_cand.max(q_cand));
                        verified = true;
                    }
                }
            }
        }

        let digits_n = modulus.to_string().len();
        let digits_p = factor_p.map(|p| p.to_string().len()).unwrap_or(0);
        let wall_time_micros = t0.elapsed().as_micros() as u64;

        ZipperFactoringReport {
            composite_n: modulus,
            digit_count_n: digits_n,
            factor_p,
            factor_q,
            digit_count_p: digits_p,
            base_a: a,
            order_r,
            work_qubits: nw,
            counting_qubits: 1,
            total_active_qubits: nw + 1,
            active_state_bytes: active_bytes,
            cache_tier,
            mean_coherence: mean_coh,
            min_coherence: min_coh,
            estimated_phase: phase,
            factoring_verified: verified,
            wall_time_micros,
            steps,
        }
    }

    /// Single-pass Hadamard gate on active wire 0 (counting qubit).
    #[inline(always)]
    fn apply_counting_hadamard(real: &mut [f64], imag: &mut [f64], work_dim: usize) {
        let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
        for w in 0..work_dim {
            let idx0 = w * 2;
            let idx1 = idx0 + 1;
            let r0 = real[idx0];
            let i0 = imag[idx0];
            let r1 = real[idx1];
            let i1 = imag[idx1];
            real[idx0] = (r0 + r1) * inv_sqrt2;
            imag[idx0] = (i0 + i1) * inv_sqrt2;
            real[idx1] = (r0 - r1) * inv_sqrt2;
            imag[idx1] = (i0 - i1) * inv_sqrt2;
        }
    }

    /// Controlled modular exponentiation $C\text{-}U_a^{2^j}$ acting on slice $c = 1$.
    #[inline(always)]
    fn apply_controlled_modular_power(
        real: &mut [f64],
        imag: &mut [f64],
        scratch_r: &mut [f64],
        scratch_i: &mut [f64],
        a: u64,
        j: usize,
        modulus: u64,
        work_dim: usize,
    ) {
        let power = 1u64 << j.min(63);
        let eff_a = mod_pow(a, power, modulus);

        for w in 0..work_dim {
            scratch_r[w] = real[w * 2 + 1];
            scratch_i[w] = imag[w * 2 + 1];
            real[w * 2 + 1] = 0.0;
            imag[w * 2 + 1] = 0.0;
        }

        for w in 0..work_dim {
            let target_w = if (w as u64) < modulus {
                ((eff_a as u128 * w as u128) % modulus as u128) as usize
            } else {
                w
            };
            real[target_w * 2 + 1] = scratch_r[w];
            imag[target_w * 2 + 1] = scratch_i[w];
        }
    }

    /// Adaptive feedback angle $\theta_j = 2\pi \sum_{k > j} b_k 2^{-(k - j + 1)}$.
    #[inline(always)]
    fn compute_adaptive_angle(bits: &[u8], j: usize, num_bits: usize) -> f64 {
        let mut theta = 0.0f64;
        for k in (j + 1)..num_bits {
            if bits[k] == 1 {
                let shift = (k - j + 1) as i32;
                theta += 2.0 * PI * (0.5f64).powi(shift);
            }
        }
        theta
    }

    /// In-place adaptive phase rotation $R_z(-\theta)$ on wire 0 ($c = 1$ branch).
    #[inline(always)]
    fn apply_adaptive_phase(real: &mut [f64], imag: &mut [f64], theta: f64, work_dim: usize) {
        let cos_t = (-theta).cos();
        let sin_t = (-theta).sin();
        for w in 0..work_dim {
            let idx1 = w * 2 + 1;
            let r = real[idx1];
            let i = imag[idx1];
            real[idx1] = r * cos_t - i * sin_t;
            imag[idx1] = r * sin_t + i * cos_t;
        }
    }

    /// Evaluates counting qubit expectation, coherence, and von Neumann entropy.
    #[inline(always)]
    fn measure_counting_qubit(
        real: &[f64],
        imag: &[f64],
        work_dim: usize,
    ) -> (f64, f64, f64, f64, f64) {
        let mut p0 = 0.0f64;
        let mut p1 = 0.0f64;
        for w in 0..work_dim {
            let r0 = real[w * 2];
            let i0 = imag[w * 2];
            let r1 = real[w * 2 + 1];
            let i1 = imag[w * 2 + 1];
            p0 += r0 * r0 + i0 * i0;
            p1 += r1 * r1 + i1 * i1;
        }
        let tot = p0 + p1;
        if tot > 1e-15 {
            p0 /= tot;
            p1 /= tot;
        }
        let z_exp = p0 - p1;
        let coh = z_exp.abs();
        let entropy = if p0 > 1e-15 && p1 > 1e-15 {
            -(p0 * p0.ln() + p1 * p1.ln())
        } else {
            0.0
        };
        (p0, p1, z_exp, coh, entropy)
    }

    /// Epoxy collapses active wire 0 and resets to ground state $|0\rangle$.
    #[inline(always)]
    fn epoxy_collapse_and_reset(
        real: &mut [f64],
        imag: &mut [f64],
        bit: u8,
        p0: f64,
        p1: f64,
        work_dim: usize,
    ) {
        let norm_factor = if bit == 0 {
            1.0 / p0.max(1e-15).sqrt()
        } else {
            1.0 / p1.max(1e-15).sqrt()
        };

        for w in 0..work_dim {
            if bit == 0 {
                real[w * 2] *= norm_factor;
                imag[w * 2] *= norm_factor;
                real[w * 2 + 1] = 0.0;
                imag[w * 2 + 1] = 0.0;
            } else {
                real[w * 2] = real[w * 2 + 1] * norm_factor;
                imag[w * 2] = imag[w * 2 + 1] * norm_factor;
                real[w * 2 + 1] = 0.0;
                imag[w * 2 + 1] = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zipper_factors_1_digit_prime() {
        // N = 15 (3 * 5)
        let report = ZipperShorFactorizer::run_single_attempt(7, 15, 42);
        assert!(report.factoring_verified);
        assert_eq!(report.factor_p, Some(3));
        assert_eq!(report.factor_q, Some(5));
        assert_eq!(report.digit_count_p, 1);
        assert_eq!(report.order_r, Some(4));
    }

    #[test]
    fn test_zipper_factors_2_digit_prime() {
        // N = 143 (11 * 13)
        let mut factored = false;
        for seed in [42, 12345, 9999, 777] {
            let report = ZipperShorFactorizer::run_single_attempt(2, 143, seed);
            if report.factoring_verified {
                assert_eq!(report.factor_p, Some(11));
                assert_eq!(report.factor_q, Some(13));
                assert_eq!(report.digit_count_p, 2);
                factored = true;
                break;
            }
        }
        assert!(factored, "Failed to factor N=143 across seeds");
    }

    #[test]
    fn test_zipper_factors_3_digit_prime() {
        // N = 10403 (101 * 103)
        let mut factored = false;
        for seed in [42, 12345, 9999, 777, 31415] {
            let report = ZipperShorFactorizer::run_single_attempt(2, 10403, seed);
            if report.factoring_verified {
                assert_eq!(report.factor_p, Some(101));
                assert_eq!(report.factor_q, Some(103));
                assert_eq!(report.digit_count_p, 3);
                factored = true;
                break;
            }
        }
        assert!(factored, "Failed to factor N=10403 across seeds");
    }
}
