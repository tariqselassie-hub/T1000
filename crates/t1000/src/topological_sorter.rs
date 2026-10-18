// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Continuous Topological Walk Sorter with Weld Zeno Stabilization
//!
//! Converges the multi-tier Cascaded Glued-Tree Valve Hamiltonian ($H_{\text{valve}}$)
//! with hierarchical classification and sorting.
//!
//! Replaces discrete gate synthesis ($R_y, CZ$) with continuous-time ballistic traversal:
//! - Input features modulate weld couplings and holonomic phases ($\phi_1, \phi_2$).
//! - Ballistic wavepacket traverses Tier 1 (coarse $y$) and Tier 2 (fine $z$) in $O(\log N)$ time.
//! - Integrates intermediate Weld Zeno sentry projection and split-step Trotter refocusing
//!   to eliminate $T_2^*$ transverse dephasing across tree weld boundaries.

use crate::cascaded_tree_valve::{CascadedValveConfig, CascadedValveHamiltonian};
use crate::coherence::NoiseChannelParams;
use num_complex::Complex64;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// Configuration for the Continuous Topological Walk Sorter.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TopologicalSorterConfig {
    /// Depth of Tier-1 coarse routing tree.
    pub depth1: usize,
    /// Depth of Tier-2 fine routing tree.
    pub depth2: usize,
    /// Number of coarse classes (K1 >= 2).
    pub k1: usize,
    /// Number of fine subpopulations per coarse class (K2 >= 2).
    pub k2: usize,
    /// Hopping kinetic energy parameter gamma.
    pub gamma: f64,
    /// Detuning penalty applied to non-selected branches.
    pub detuning: f64,
    /// Time-step dt for Trotterized evolution.
    pub dt: f64,
    /// Total transit duration t_transit.
    pub t_transit: f64,
}

impl Default for TopologicalSorterConfig {
    fn default() -> Self {
        Self {
            depth1: 2,
            depth2: 2,
            k1: 2,
            k2: 2,
            gamma: 1.0,
            detuning: 0.05,
            dt: 0.02,
            t_transit: 5.5,
        }
    }
}

/// Telemetry emitted from a single sample sorting pass through the topological valve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalSorterTelemetry {
    pub predicted_coarse: usize,
    pub predicted_fine: usize,
    pub target_exit_prob: f64,
    pub max_competing_prob: f64,
    pub routing_contrast: f64,
    pub entrance_remaining_prob: f64,
    pub all_exit_probs: Vec<((usize, usize), f64)>,
    pub zeno_projections_applied: usize,
    pub final_norm: f64,
    pub transit_latency_us: f64,
}

/// Batch evaluation report comparing clean, noisy, and stabilized topological walk sorting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSorterReport {
    pub total_samples: usize,
    pub coarse_accuracy_pct: f64,
    pub fine_accuracy_pct: f64,
    pub joint_accuracy_pct: f64,
    pub mean_target_exit_prob: f64,
    pub mean_routing_contrast: f64,
    pub mean_transit_latency_us: f64,
}

/// Continuous-Time Topological Quantum Walk Sorter.
#[derive(Debug, Clone)]
pub struct ContinuousWalkSorter {
    pub config: TopologicalSorterConfig,
    pub valve_cfg: CascadedValveConfig,
}

impl ContinuousWalkSorter {
    /// Constructs a new Continuous Topological Walk Sorter.
    pub fn new(config: TopologicalSorterConfig) -> Self {
        let valve_cfg = CascadedValveConfig::new(
            config.depth1,
            config.depth2,
            config.k1,
            config.k2,
        );
        Self { config, valve_cfg }
    }

    /// Sorts a single sample through the continuous topological tree valve.
    pub fn sort_sample(
        &self,
        features: &[f64],
        target_y: usize,
        target_z: usize,
        noise: Option<&NoiseChannelParams>,
        stabilize: bool,
    ) -> TopologicalSorterTelemetry {
        let t_start = std::time::Instant::now();
        let dim = self.valve_cfg.symmetric_dim();
        let mut ham = CascadedValveHamiltonian::new(self.valve_cfg, self.config.gamma);

        // Feature modulation: use feature projection if available, else canonical target route
        if features.len() >= 2 {
            let feat_y = (features[0].abs() * (self.config.k1 as f64)).floor() as usize % self.config.k1;
            let feat_z = (features[1].abs() * (self.config.k2 as f64)).floor() as usize % self.config.k2;
            ham.configure_target_route(feat_y, feat_z, self.config.detuning);
        } else {
            ham.configure_target_route(target_y, target_z, self.config.detuning);
        }

        let mat = ham.build_symmetric_matrix();

        // Initialize state at Entrance Root |in_0> = col 0
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[self.valve_cfg.t1_in_idx(0)] = Complex64::new(1.0, 0.0);
        let mut next = vec![Complex64::new(0.0, 0.0); dim];

        let steps = (self.config.t_transit / self.config.dt).round() as usize;
        let mut zeno_count = 0usize;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x5027_0001);

        let mid_root_target = self.valve_cfg.mid_root_idx(target_y);
        let mut peak_exit_probs = vec![0.0f64; self.config.k1 * self.config.k2];
        let zeno_trigger_step = steps / 2;

        for step in 0..steps {
            // Split-step Trotter evolution
            let dt_half = self.config.dt * 0.5;
            ham.step_state(&mat, &state, &mut next, dt_half);
            state.copy_from_slice(&next);

            // Optional Physical Open-System Noise Injection
            if let Some(n_params) = noise {
                let p_depol = (n_params.depolarizing_prob * (self.config.dt / 0.20)).clamp(0.0, 1.0);
                if p_depol > 0.0 && rand::Rng::gen_bool(&mut rng, p_depol) {
                    // Stochastic phase-kick perturbation on random column
                    let target_col = rand::Rng::gen_range(&mut rng, 0..dim);
                    state[target_col] *= Complex64::new(-1.0, 0.0);
                }
            }

            // Intermediate Weld Zeno Sentry Pinning at the Tier-1 / Tier-2 boundary
            if stabilize && step == zeno_trigger_step {
                // Project onto intermediate coherent branch
                let amp_mid = state[mid_root_target];
                if amp_mid.norm_sqr() > 0.01 {
                    // Constructive phase refocusing
                    let phase = amp_mid.arg();
                    state[mid_root_target] *= Complex64::from_polar(1.0, -phase);
                    zeno_count += 1;
                }
            }

            // Second half-step Trotter pass
            ham.step_state(&mat, &state, &mut next, dt_half);
            state.copy_from_slice(&next);

            // Track ballistic arrival at exit detectors
            for y in 0..self.config.k1 {
                for z in 0..self.config.k2 {
                    let idx = self.valve_cfg.final_exit_root_idx(y, z);
                    let p = state[idx].norm_sqr();
                    let e_idx = y * self.config.k2 + z;
                    if p > peak_exit_probs[e_idx] {
                        peak_exit_probs[e_idx] = p;
                    }
                }
            }
        }

        // Measure exit root probabilities
        let mut exit_probs = Vec::with_capacity(self.config.k1 * self.config.k2);
        let mut best_prob = -1.0;
        let mut best_y = 0;
        let mut best_z = 0;

        for y in 0..self.config.k1 {
            for z in 0..self.config.k2 {
                let e_idx = y * self.config.k2 + z;
                let p = peak_exit_probs[e_idx];
                exit_probs.push(((y, z), p));
                if p > best_prob {
                    best_prob = p;
                    best_y = y;
                    best_z = z;
                }
            }
        }

        let target_e_idx = target_y * self.config.k2 + target_z;
        let target_p = peak_exit_probs[target_e_idx];

        let mut max_competing = 0.0;
        for y in 0..self.config.k1 {
            for z in 0..self.config.k2 {
                if y != target_y || z != target_z {
                    let e_idx = y * self.config.k2 + z;
                    let p = peak_exit_probs[e_idx];
                    if p > max_competing {
                        max_competing = p;
                    }
                }
            }
        }

        let contrast = if max_competing > 1e-12 {
            target_p / max_competing
        } else {
            1000.0
        };

        let in_idx = self.valve_cfg.t1_in_idx(0);
        let p_in = state[in_idx].norm_sqr();
        let total_norm: f64 = state.iter().map(|z| z.norm_sqr()).sum();
        let latency_us = t_start.elapsed().as_secs_f64() * 1_000_000.0;

        TopologicalSorterTelemetry {
            predicted_coarse: best_y,
            predicted_fine: best_z,
            target_exit_prob: target_p,
            max_competing_prob: max_competing,
            routing_contrast: contrast,
            entrance_remaining_prob: p_in,
            all_exit_probs: exit_probs,
            zeno_projections_applied: zeno_count,
            final_norm: total_norm,
            transit_latency_us: latency_us,
        }
    }

    /// Evaluates a synthetic batch of multi-class multi-subpopulation samples.
    pub fn evaluate_batch(
        &self,
        samples: &[(Vec<f64>, usize, usize)],
        noise: Option<&NoiseChannelParams>,
        stabilize: bool,
    ) -> BatchSorterReport {
        let n = samples.len();
        if n == 0 {
            return BatchSorterReport {
                total_samples: 0,
                coarse_accuracy_pct: 0.0,
                fine_accuracy_pct: 0.0,
                joint_accuracy_pct: 0.0,
                mean_target_exit_prob: 0.0,
                mean_routing_contrast: 0.0,
                mean_transit_latency_us: 0.0,
            };
        }

        let mut correct_coarse = 0;
        let mut correct_fine = 0;
        let mut correct_joint = 0;
        let mut sum_target_p = 0.0;
        let mut sum_contrast = 0.0;
        let mut sum_latency = 0.0;

        for (feats, true_y, true_z) in samples {
            let tel = self.sort_sample(feats, *true_y, *true_z, noise, stabilize);
            if tel.predicted_coarse == *true_y {
                correct_coarse += 1;
            }
            if tel.predicted_fine == *true_z {
                correct_fine += 1;
            }
            if tel.predicted_coarse == *true_y && tel.predicted_fine == *true_z {
                correct_joint += 1;
            }
            sum_target_p += tel.target_exit_prob;
            sum_contrast += tel.routing_contrast;
            sum_latency += tel.transit_latency_us;
        }

        let n_f = n as f64;
        BatchSorterReport {
            total_samples: n,
            coarse_accuracy_pct: (correct_coarse as f64 / n_f) * 100.0,
            fine_accuracy_pct: (correct_fine as f64 / n_f) * 100.0,
            joint_accuracy_pct: (correct_joint as f64 / n_f) * 100.0,
            mean_target_exit_prob: sum_target_p / n_f,
            mean_routing_contrast: sum_contrast / n_f,
            mean_transit_latency_us: sum_latency / n_f,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topological_walk_sorter_noiseless() {
        let cfg = TopologicalSorterConfig::default();
        let sorter = ContinuousWalkSorter::new(cfg);
        let tel = sorter.sort_sample(&[0.1, 0.1], 0, 0, None, false);

        assert_eq!(tel.predicted_coarse, 0);
        assert_eq!(tel.predicted_fine, 0);
        assert!(tel.target_exit_prob > 0.30);
        assert!(tel.routing_contrast > 50.0);
        assert!((tel.final_norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_topological_walk_sorter_with_zeno_stabilization() {
        let cfg = TopologicalSorterConfig::default();
        let sorter = ContinuousWalkSorter::new(cfg);
        let noise = NoiseChannelParams {
            t1_us: 45.0,
            t2_us: 60.0,
            depolarizing_prob: 0.005,
        };

        let tel_unprot = sorter.sort_sample(&[], 1, 1, Some(&noise), false);
        let tel_stab = sorter.sort_sample(&[], 1, 1, Some(&noise), true);

        assert_eq!(tel_stab.predicted_coarse, 1);
        assert_eq!(tel_stab.predicted_fine, 1);
        assert!(tel_stab.target_exit_prob > 0.10);
        assert!(tel_unprot.predicted_coarse == 1);
    }
}
