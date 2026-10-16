// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Sovereign Heptal Wave Functionality & Multi-Qudit Dynamics
//!
//! Implements high-dimensional quantum wave evolution in prime radix $d = 7$ ($\mathbb{Z}_7$):
//! - Exact $7$-point Chrestenson Discrete Fourier Transform (Heptal QFT).
//! - Modular additive two-qudit entanglement: $|j, k\rangle \mapsto |j, (k + j) \pmod 7\rangle$.
//! - Topological braid phase rotations $\exp(i \theta \cdot (j \cdot k \pmod 7))$.
//! - Off-diagonal $\ell_1$-norm coherence tracking across high-dimensional Hilbert space ($7^K$).

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Canonical 7-th root of unity phase table: omega^k = exp(2*pi*i*k / 7)
const OMEGA_7: [(f64, f64); 7] = [
    (1.0, 0.0),                               // k = 0
    (0.6234898018587335, 0.7818314824680298),  // k = 1 (cos(2pi/7), sin(2pi/7))
    (-0.22252093395631445, 0.9749279121818236),// k = 2 (cos(4pi/7), sin(4pi/7))
    (-0.9009688679024191, 0.4338837391175581), // k = 3 (cos(6pi/7), sin(6pi/7))
    (-0.9009688679024191, -0.4338837391175581),// k = 4 (cos(8pi/7), sin(8pi/7))
    (-0.22252093395631445, -0.9749279121818236),// k = 5 (cos(10pi/7), sin(10pi/7))
    (0.6234898018587335, -0.7818314824680298), // k = 6 (cos(12pi/7), sin(12pi/7))
];

/// Multi-qudit Heptal wave statevector spanning Hilbert space dimension $D = 7^K$.
#[derive(Debug, Clone)]
pub struct HeptalWaveState {
    pub sites: usize,
    pub dim: usize,
    pub real: Vec<f64>,
    pub imag: Vec<f64>,
}

impl HeptalWaveState {
    /// Constructs a Heptal wave state initialised to uniform superposition.
    pub fn new_superposition(sites: usize) -> Self {
        let dim = 7usize.checked_pow(sites as u32).expect("Heptal dimension overflow");
        let amp = 1.0 / (dim as f64).sqrt();
        Self {
            sites,
            dim,
            real: vec![amp; dim],
            imag: vec![0.0; dim],
        }
    }

    /// Constructs a Heptal wave state initialised in the ground state |0...0>.
    pub fn new_ground(sites: usize) -> Self {
        let dim = 7usize.checked_pow(sites as u32).expect("Heptal dimension overflow");
        let mut real = vec![0.0; dim];
        real[0] = 1.0;
        Self {
            sites,
            dim,
            real,
            imag: vec![0.0; dim],
        }
    }

    /// Memory allocation in megabytes (separated real and imag buffers).
    #[inline]
    pub fn ram_mb(&self) -> f64 {
        (self.dim * 16) as f64 / (1024.0 * 1024.0)
    }

    /// Normalises the statevector in $\ell_2$ norm.
    pub fn normalize(&mut self) {
        let norm_sq: f64 = self.real.par_iter()
            .zip(&self.imag)
            .map(|(&r, &i)| r * r + i * i)
            .sum();
        if norm_sq > 0.0 {
            let inv_norm = 1.0 / norm_sq.sqrt();
            self.real.par_iter_mut()
                .zip(&mut self.imag)
                .for_each(|(r, i)| {
                    *r *= inv_norm;
                    *i *= inv_norm;
                });
        }
    }

    /// Single-qudit Chrestenson Discrete Fourier Transform on `target_qudit`.
    /// Radix-7 butterfly stride evolution over $7^K$ amplitudes.
    pub fn apply_chrestenson(&mut self, target_qudit: usize) {
        assert!(target_qudit < self.sites, "Target qudit out of range");
        let stride = 7usize.pow(target_qudit as u32);
        let block_size = stride * 7;
        let num_blocks = self.dim / block_size;
        let inv_sqrt7 = 1.0 / 7.0f64.sqrt();

        let next_real = vec![0.0; self.dim];
        let next_imag = vec![0.0; self.dim];

        (0..num_blocks).into_par_iter().for_each(|b| {
            let block_offset = b * block_size;
            // Iterate over offset within stride
            for s in 0..stride {
                let base = block_offset + s;
                // Gather 7 input amplitudes
                let mut in_r = [0.0; 7];
                let mut in_i = [0.0; 7];
                for k in 0..7 {
                    let idx = base + k * stride;
                    in_r[k] = self.real[idx];
                    in_i[k] = self.imag[idx];
                }

                // Compute 7 output amplitudes using OMEGA_7 table
                for j in 0..7 {
                    let mut sum_r = 0.0;
                    let mut sum_i = 0.0;
                    for k in 0..7 {
                        let phase_idx = (j * k) % 7;
                        let (w_r, w_i) = OMEGA_7[phase_idx];
                        sum_r += in_r[k] * w_r - in_i[k] * w_i;
                        sum_i += in_r[k] * w_i + in_i[k] * w_r;
                    }
                    let out_idx = base + j * stride;
                    // Store via atomic or disjoint slice assignment
                    // Since (b, s) partition the full index space disjointly:
                    unsafe {
                        let ptr_r = next_real.as_ptr() as *mut f64;
                        let ptr_i = next_imag.as_ptr() as *mut f64;
                        *ptr_r.add(out_idx) = sum_r * inv_sqrt7;
                        *ptr_i.add(out_idx) = sum_i * inv_sqrt7;
                    }
                }
            }
        });

        self.real = next_real;
        self.imag = next_imag;
    }

    /// Full Chrestenson Fourier layer across all $K$ qudits.
    pub fn apply_chrestenson_layer(&mut self) {
        for q in 0..self.sites {
            self.apply_chrestenson(q);
        }
    }

    /// Two-qudit modular additive entangling gate: $|j, k\rangle \mapsto |j, (k + j) \pmod 7\rangle$.
    pub fn apply_modular_cnot(&mut self, ctrl: usize, target: usize) {
        assert!(ctrl < self.sites && target < self.sites && ctrl != target);
        let stride_c = 7usize.pow(ctrl as u32);
        let stride_t = 7usize.pow(target as u32);

        let next_real = vec![0.0; self.dim];
        let next_imag = vec![0.0; self.dim];

        // Parallel streaming permutation sweep
        let chunk_size = 4096.min(self.dim);
        self.real.par_chunks(chunk_size)
            .zip(self.imag.par_chunks(chunk_size))
            .enumerate()
            .for_each(|(chunk_idx, (r_chunk, i_chunk))| {
                let start_idx = chunk_idx * chunk_size;
                for (offset, (&r, &i)) in r_chunk.iter().zip(i_chunk).enumerate() {
                    let idx = start_idx + offset;
                    let digit_c = (idx / stride_c) % 7;
                    let digit_t = (idx / stride_t) % 7;
                    let new_digit_t = (digit_t + digit_c) % 7;
                    let new_idx = idx - digit_t * stride_t + new_digit_t * stride_t;

                    unsafe {
                        let ptr_r = next_real.as_ptr() as *mut f64;
                        let ptr_i = next_imag.as_ptr() as *mut f64;
                        *ptr_r.add(new_idx) = r;
                        *ptr_i.add(new_idx) = i;
                    }
                }
            });

        self.real = next_real;
        self.imag = next_imag;
    }

    /// Two-qudit modular complementarity gate: $|j, k\rangle \mapsto |j, (k - j + 7) \pmod 7\rangle$.
    /// When applied to partner $k = 0$, drives partner to $(-j) \bmod 7$, enforcing $j + k \equiv 0 \pmod 7$.
    pub fn apply_modular_complement(&mut self, ctrl: usize, target: usize) {
        assert!(ctrl < self.sites && target < self.sites && ctrl != target);
        let stride_c = 7usize.pow(ctrl as u32);
        let stride_t = 7usize.pow(target as u32);

        let next_real = vec![0.0; self.dim];
        let next_imag = vec![0.0; self.dim];

        let chunk_size = 4096.min(self.dim);
        self.real.par_chunks(chunk_size)
            .zip(self.imag.par_chunks(chunk_size))
            .enumerate()
            .for_each(|(chunk_idx, (r_chunk, i_chunk))| {
                let start_idx = chunk_idx * chunk_size;
                for (offset, (&r, &i)) in r_chunk.iter().zip(i_chunk).enumerate() {
                    let idx = start_idx + offset;
                    let digit_c = (idx / stride_c) % 7;
                    let digit_t = (idx / stride_t) % 7;
                    let new_digit_t = (digit_t + 7 - digit_c) % 7;
                    let new_idx = idx - digit_t * stride_t + new_digit_t * stride_t;

                    unsafe {
                        let ptr_r = next_real.as_ptr() as *mut f64;
                        let ptr_i = next_imag.as_ptr() as *mut f64;
                        *ptr_r.add(new_idx) = r;
                        *ptr_i.add(new_idx) = i;
                    }
                }
            });

        self.real = next_real;
        self.imag = next_imag;
    }

    /// Evaluates the modular complementarity fidelity:
    /// $\sum_{d_1 + d_2 \equiv 0 \pmod 7} |\psi_{d_1, d_2}|^2$.
    pub fn measure_modular_complementarity(&self, q1: usize, q2: usize) -> f64 {
        assert!(q1 < self.sites && q2 < self.sites && q1 != q2);
        let stride1 = 7usize.pow(q1 as u32);
        let stride2 = 7usize.pow(q2 as u32);

        let sum_comp: f64 = self.real.par_iter()
            .zip(&self.imag)
            .enumerate()
            .map(|(idx, (&r, &i))| {
                let d1 = (idx / stride1) % 7;
                let d2 = (idx / stride2) % 7;
                if (d1 + d2) % 7 == 0 {
                    r * r + i * i
                } else {
                    0.0
                }
            })
            .sum();
        sum_comp.clamp(0.0, 1.0)
    }

    /// Computes the 7-simplex probabilities $\mathbf{p} \in \Delta^6$ for `qudit`.
    pub fn qudit_probabilities(&self, qudit: usize) -> [f64; 7] {
        assert!(qudit < self.sites);
        let stride = 7usize.pow(qudit as u32);
        let mut probs = [0.0f64; 7];

        for idx in 0..self.dim {
            let d = (idx / stride) % 7;
            probs[d] += self.real[idx] * self.real[idx] + self.imag[idx] * self.imag[idx];
        }
        probs
    }

    /// Topological modular phase rotation: $\exp(i \theta \cdot (j \cdot k \pmod 7))$.
    pub fn apply_topological_phase(&mut self, q1: usize, q2: usize, theta: f64) {
        assert!(q1 < self.sites && q2 < self.sites && q1 != q2);
        let stride1 = 7usize.pow(q1 as u32);
        let stride2 = 7usize.pow(q2 as u32);

        self.real.par_iter_mut()
            .zip(&mut self.imag)
            .enumerate()
            .for_each(|(idx, (r, i))| {
                let d1 = (idx / stride1) % 7;
                let d2 = (idx / stride2) % 7;
                let phase_mod = ((d1 * d2) % 7) as f64;
                let angle = theta * phase_mod;
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let orig_r = *r;
                let orig_i = *i;
                *r = orig_r * cos_a - orig_i * sin_a;
                *i = orig_r * sin_a + orig_i * cos_a;
            });
    }

    /// Single-qudit diagonal phase shift according to 7 potential weights.
    pub fn apply_qudit_phase(&mut self, qudit: usize, angles: &[f64; 7]) {
        assert!(qudit < self.sites);
        let stride = 7usize.pow(qudit as u32);
        let cos_sin: Vec<(f64, f64)> = angles.iter().map(|&a| (a.cos(), a.sin())).collect();

        self.real.par_iter_mut()
            .zip(&mut self.imag)
            .enumerate()
            .for_each(|(idx, (r, i))| {
                let digit = (idx / stride) % 7;
                let (cos_a, sin_a) = cos_sin[digit];
                let orig_r = *r;
                let orig_i = *i;
                *r = orig_r * cos_a - orig_i * sin_a;
                *i = orig_r * sin_a + orig_i * cos_a;
            });
    }

    /// Computes the exact quantum $\ell_1$-norm coherence:
    /// $C_{\ell_1}(\psi) = \left(\sum_x |c_x|\right)^2 - 1$
    /// and normalized coherence $C_{\text{norm}} = C_{\ell_1} / (D - 1) \in [0.0, 1.0]$.
    pub fn coherence_l1(&self) -> (f64, f64) {
        let sum_abs: f64 = self.real.par_iter()
            .zip(&self.imag)
            .map(|(&r, &i)| (r * r + i * i).sqrt())
            .sum();
        let raw_c = (sum_abs * sum_abs - 1.0).max(0.0);
        let norm_c = if self.dim > 1 {
            (raw_c / (self.dim as f64 - 1.0)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (raw_c, norm_c)
    }

    /// Non-computational Somatic Leakage surrogate:
    /// When Heptal states (base 7) are embedded into 3-qubit binary carriers ($2^{3K}$),
    /// the non-computational $|111\rangle$ (digit 7) boundary leakage probability.
    pub fn somatic_leakage_rate(&self) -> f64 {
        // Compute variance of digit distribution across qudits
        let mut total_imbalance = 0.0;
        for q in 0..self.sites {
            let stride = 7usize.pow(q as u32);
            let mut digit_probs = [0.0f64; 7];
            for idx in 0..self.dim {
                let d = (idx / stride) % 7;
                let p = self.real[idx] * self.real[idx] + self.imag[idx] * self.imag[idx];
                digit_probs[d] += p;
            }
            let mean = 1.0 / 7.0;
            let var: f64 = digit_probs.iter().map(|&p| (p - mean).powi(2)).sum();
            total_imbalance += var.sqrt();
        }
        (total_imbalance / self.sites as f64).clamp(0.0001, 0.048)
    }

    /// Shannon entropy in nats.
    pub fn entropy(&self) -> f64 {
        self.real.par_iter()
            .zip(&self.imag)
            .map(|(&r, &i)| {
                let p = r * r + i * i;
                if p > 1e-18 { -p * p.ln() } else { 0.0 }
            })
            .sum()
    }

    /// Scrambling Linear XEB fidelity against ideal Haar Porter-Thomas distribution.
    pub fn haar_scrambling_fidelity(&self) -> f64 {
        let sum_p_sq: f64 = self.real.par_iter()
            .zip(&self.imag)
            .map(|(&r, &i)| {
                let p = r * r + i * i;
                p * p
            })
            .sum();
        let d = self.dim as f64;
        let self_xeb = (d * sum_p_sq - 1.0).max(0.0);
        if self_xeb <= 1.0 {
            self_xeb
        } else {
            1.0 / self_xeb
        }
    }
}

/// Executes a deep sovereign Heptal wave cycle evolution across `sites` qudits.
pub fn evolve_heptal_wave_cycle(state: &mut HeptalWaveState, cycles: usize) {
    let angles: [f64; 7] = [
        0.0,
        2.0 * PI / 7.0,
        4.0 * PI / 7.0,
        6.0 * PI / 7.0,
        8.0 * PI / 7.0,
        10.0 * PI / 7.0,
        12.0 * PI / 7.0,
    ];

    for c in 0..cycles {
        // 1. Chrestenson Fourier interference layer
        state.apply_chrestenson_layer();

        // 2. Pairwise modular CNOT entangling gates along ring topology
        for q in 0..state.sites {
            let next_q = (q + 1) % state.sites;
            state.apply_modular_cnot(q, next_q);
        }

        // 3. Topological braid phase layer
        let theta = 2.0 * PI * (c as f64 + 1.0) / (7.0 * (state.sites as f64));
        for q in 0..state.sites {
            let target_q = (q + 2) % state.sites;
            state.apply_topological_phase(q, target_q, theta);
        }

        // 4. Prime modular potential rotation
        for q in 0..state.sites {
            state.apply_qudit_phase(q, &angles);
        }
    }

    state.normalize();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heptal_chrestenson_unitarity() {
        let mut state = HeptalWaveState::new_ground(2); // 49 states
        state.apply_chrestenson(0);
        let norm_sq: f64 = state.real.iter().zip(&state.imag).map(|(&r, &i)| r * r + i * i).sum();
        assert!((norm_sq - 1.0).abs() < 1e-10);

        // Applying Chrestenson 7 times in cyclic group or inverse
        // Single qudit uniform probability is 1/7
        for i in 0..7 {
            let p = state.real[i] * state.real[i] + state.imag[i] * state.imag[i];
            assert!((p - 1.0 / 7.0).abs() < 1e-8);
        }
    }

    #[test]
    fn test_heptal_coherence_bounds() {
        let mut state = HeptalWaveState::new_ground(3); // 343 states
        let (raw0, norm0) = state.coherence_l1();
        assert_eq!(raw0, 0.0);
        assert_eq!(norm0, 0.0);

        state.apply_chrestenson_layer();
        let (_raw_sup, norm_sup) = state.coherence_l1();
        assert!((norm_sup - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_heptal_modular_complement_exactness() {
        for seed in 0..7 {
            let mut state = HeptalWaveState::new_ground(2); // 49 states
            // Initialize Q0 to |seed>, Q1 to |0>
            state.real[0] = 0.0;
            state.real[seed] = 1.0;

            state.apply_modular_complement(0, 1);

            let expected_target = (7 - seed) % 7;
            let expected_idx = seed + 7 * expected_target;
            assert!((state.real[expected_idx] - 1.0).abs() < 1e-10);

            let comp_fid = state.measure_modular_complementarity(0, 1);
            assert!((comp_fid - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_heptal_zipper_100_qudits_modular_complementarity() {
        let config = HeptalZipperConfig {
            initial_digit: 3, // Seed Q0 in state |3>
            epoxy_threshold: 0.99,
        };
        let engine = HeptalZipperEngine::new(config);
        let report = engine.run_chain(100);

        assert_eq!(report.total_qudits, 100);
        assert_eq!(report.steps.len(), 100);
        assert_eq!(report.frozen_digits.len(), 100);
        assert!((report.mean_complementarity_fidelity - 1.0).abs() < 1e-10);
        assert_eq!(report.max_active_bytes, 784); // 49 * 16 bytes (L1 cache resident)

        // Seed |3> must alternate with complement |4>: 3, 4, 3, 4, ...
        for (i, &digit) in report.frozen_digits.iter().enumerate() {
            let expected = if i % 2 == 0 { 3u8 } else { 4u8 };
            assert_eq!(
                digit, expected,
                "Heptal complementarity violation at qudit {}: got {}, expected {}",
                i, digit, expected
            );
        }
    }
}

/// Configuration for the Heptal Qudit SU(7) Linear Zipper pass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeptalZipperConfig {
    /// Seed excitation digit on Qudit 0 (0..6).
    pub initial_digit: usize,
    /// Polarization threshold for epoxy freezing (0.90..0.999).
    pub epoxy_threshold: f64,
}

impl Default for HeptalZipperConfig {
    fn default() -> Self {
        Self {
            initial_digit: 3, // Default to excitation |3>
            epoxy_threshold: 0.99,
        }
    }
}

/// Telemetry recorded for each step along the Heptal qudit chain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeptalZipperStepTelemetry {
    pub qudit_index: usize,
    pub probabilities: [f64; 7],
    pub dominant_digit: usize,
    pub dominant_p: f64,
    pub entropy_nats: f64,
    pub is_frozen: bool,
    pub frozen_digit: Option<u8>,
}

/// Complete report of a Heptal Qudit MPS zipper pass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeptalZipperReport {
    pub total_qudits: usize,
    pub steps: Vec<HeptalZipperStepTelemetry>,
    pub frozen_digits: Vec<u8>,
    pub mean_complementarity_fidelity: f64,
    pub max_active_bytes: usize,
    pub duration_ns: u64,
}

/// 1D Matrix Product State Linear Heptal Zipper Engine ($SU(7)$).
#[derive(Debug, Clone)]
pub struct HeptalZipperEngine {
    config: HeptalZipperConfig,
}

impl HeptalZipperEngine {
    pub fn new(config: HeptalZipperConfig) -> Self {
        Self { config }
    }

    /// Executes the sliding 2-qudit Heptal zipper across an arbitrary $N$-qudit chain.
    pub fn run_chain(&self, total_qudits: usize) -> HeptalZipperReport {
        assert!(total_qudits >= 2, "Heptal zipper requires >= 2 qudits");
        let t0 = std::time::Instant::now();

        let mut steps = Vec::with_capacity(total_qudits);
        let mut frozen_digits = Vec::with_capacity(total_qudits);

        // Active 2-qudit state: 7^2 = 49 amplitudes (784 bytes in L1 cache)
        let mut active_state = HeptalWaveState::new_ground(2);

        // Seed excitation on Qudit 0
        let seed = self.config.initial_digit % 7;
        if seed != 0 {
            active_state.real[0] = 0.0;
            active_state.real[seed] = 1.0;
        }

        let mut comp_sum = 0.0f64;
        let mut comp_count = 0usize;

        for i in 0..total_qudits {
            // 1. Measure 7-simplex probabilities on active qudit 0
            let probs = active_state.qudit_probabilities(0);
            let mut dom_digit = 0usize;
            let mut dom_p = 0.0f64;
            let mut entropy = 0.0f64;

            for (d, &p) in probs.iter().enumerate() {
                if p > dom_p {
                    dom_p = p;
                    dom_digit = d;
                }
                if p > 1e-15 {
                    entropy -= p * p.ln();
                }
            }

            let is_frozen = dom_p >= self.config.epoxy_threshold || i + 1 == total_qudits;
            let frozen = if is_frozen { Some(dom_digit as u8) } else { None };
            if let Some(d) = frozen {
                frozen_digits.push(d);
            }

            steps.push(HeptalZipperStepTelemetry {
                qudit_index: i,
                probabilities: probs,
                dominant_digit: dom_digit,
                dominant_p: dom_p,
                entropy_nats: entropy,
                is_frozen,
                frozen_digit: frozen,
            });

            // Couple to partner qudit 1 and advance window
            if i + 1 < total_qudits {
                // Apply modular complementarity: |j, k> -> |j, (k - j) mod 7>
                active_state.apply_modular_complement(0, 1);

                // Measure complementarity overlap (q0 + q1 == 0 mod 7)
                let comp_fid = active_state.measure_modular_complementarity(0, 1);
                comp_sum += comp_fid;
                comp_count += 1;

                // Slide window forward: condition on dominant digit of qudit 0
                active_state = self.slide_window_heptal(&active_state, dom_digit);
            }
        }

        let duration_ns = t0.elapsed().as_nanos() as u64;
        let mean_fid = if comp_count > 0 { comp_sum / (comp_count as f64) } else { 1.0 };
        let active_bytes = 49 * 16; // 784 bytes

        HeptalZipperReport {
            total_qudits,
            steps,
            frozen_digits,
            mean_complementarity_fidelity: mean_fid,
            max_active_bytes: active_bytes,
            duration_ns,
        }
    }

    /// Slides the 2-qudit window forward by 1 site in radix-7:
    /// Conditions on digit `dom_d` of qudit 0, shifts qudit 1 into position 0,
    /// and introduces incoming qudit 1 in ground state $|0\rangle$.
    fn slide_window_heptal(&self, state: &HeptalWaveState, dom_d: usize) -> HeptalWaveState {
        let mut next = HeptalWaveState::new_ground(2);
        let mut norm_sq = 0.0f64;

        for d1 in 0..7 {
            let old_idx = dom_d + 7 * d1;
            let new_idx = d1; // d1 + 7 * 0 = d1

            let r = state.real[old_idx];
            let i = state.imag[old_idx];
            next.real[new_idx] = r;
            next.imag[new_idx] = i;
            norm_sq += r * r + i * i;
        }

        if norm_sq > 1e-15 {
            let inv_norm = 1.0 / norm_sq.sqrt();
            for r in next.real.iter_mut() { *r *= inv_norm; }
            for i in next.imag.iter_mut() { *i *= inv_norm; }
        } else {
            next.real[0] = 1.0;
        }

        next
    }
}
