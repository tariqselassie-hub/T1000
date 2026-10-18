//! Advanced Random Circuit Sampling (RCS) and Porter-Thomas Chaos Analysis.
//!
//! Solidifies Job #1: Simulating Advanced RCS and Pathed Circuits with
//! bare-metal AVX2+FMA acceleration, cache-oblivious butterfly tiling,
//! and Haar-random statistical verification.

use crate::ansatz::CircuitAnsatz;
use crate::state::QuantumState;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use t1000_core::porter_thomas::{verify_porter_thomas, PorterThomasReport};

/// Configuration for an Advanced Random Circuit Sampling simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RcsConfig {
    /// Number of qubits ($N$). Dim = $2^N$.
    pub qubits: usize,
    /// Circuit depth (number of entangling and phase layers).
    pub depth: usize,
    /// Random seed for circuit parameter generation.
    pub seed: u64,
}

impl Default for RcsConfig {
    fn default() -> Self {
        Self {
            qubits: 20,
            depth: 12,
            seed: 42,
        }
    }
}

/// Distribution comparative metrics between two quantum states or simulation backends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionMetrics {
    pub hellinger_distance: f64,
    pub squared_hellinger_distance: f64,
    pub bhattacharyya_coefficient: f64,
    pub cross_linear_xeb: f64,
    pub total_variation_distance: f64,
}

/// Layer-by-layer telemetry tracking coherence, purity, and chaos evolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RcsLayerTrajectory {
    pub layer: usize,
    pub latency_secs: f64,
    pub purity: f64,
    pub l1_coherence_norm: f64,
    pub fidelity_vs_ideal: f64,
    pub instant_xeb: f64,
}

/// Coherence stabilization protocol applied during noisy RCS circuit passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StabilizationStrategy {
    /// No stabilization: naked circuit subject to environmental decoherence.
    None,
    /// Periodic Dynamical Decoupling (CPMG / XY4 / UDD) refocusing pulses.
    DynamicalDecoupling(crate::coherence::DynamicalDecouplingKind),
    /// Symplectic manifold re-projection preventing numerical and phase drift.
    ManifoldReprojection,
    /// Combined DD refocusing and manifold re-projection.
    Combined(crate::coherence::DynamicalDecouplingKind),
}

/// Configuration for an open-system physical noisy RCS simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoisyRcsConfig {
    pub base: RcsConfig,
    pub noise: crate::coherence::NoiseChannelParams,
    pub gate_time_us: f64,
    pub stabilization: StabilizationStrategy,
}

impl Default for NoisyRcsConfig {
    fn default() -> Self {
        Self {
            base: RcsConfig::default(),
            noise: crate::coherence::NoiseChannelParams::default(),
            gate_time_us: 0.20,
            stabilization: StabilizationStrategy::None,
        }
    }
}

/// Telemetry emitted from a noisy open-system RCS simulation pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoisyRcsRunReport {
    pub qubits: usize,
    pub depth: usize,
    pub dimension: usize,
    pub total_latency_secs: f64,
    pub trajectory: Vec<RcsLayerTrajectory>,
    pub final_fidelity_vs_ideal: f64,
    pub ideal_self_xeb: f64,
    pub noisy_self_xeb: f64,
    pub cross_xeb_with_ideal: f64,
    pub pt_report: PorterThomasReport,
    pub distribution_metrics: DistributionMetrics,
    pub top_bitstrings: Vec<(String, f64)>,
}

/// Full execution report and telemetry for an RCS simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RcsRunReport {
    pub qubits: usize,
    pub depth: usize,
    pub dimension: usize,
    pub total_latency_secs: f64,
    pub avg_layer_latency_secs: f64,
    pub layer_latencies: Vec<f64>,
    pub pt_report: PorterThomasReport,
    pub top_bitstrings: Vec<(String, f64)>,
}

/// Advanced Random Circuit Sampling Engine.
#[derive(Debug, Clone)]
pub struct RcsSimulator {
    config: RcsConfig,
}

impl RcsSimulator {
    /// Create a new RCS simulator with given configuration.
    pub fn new(config: RcsConfig) -> Self {
        Self { config }
    }

    /// Generate deterministic ansatz parameters from random seed.
    fn generate_params(seed: u64, count: usize) -> Vec<f64> {
        let mut rng_state = seed;
        let mut next_f64 = || {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let val = (rng_state >> 11) as f64 / (1u64 << 53) as f64;
            val * std::f64::consts::TAU - std::f64::consts::PI
        };
        (0..count).map(|_| next_f64()).collect()
    }

    /// Execute an advanced RCS circuit pass, returning the final state and execution telemetry.
    pub fn execute(&self) -> (QuantumState, RcsRunReport) {
        let n_qubits = self.config.qubits;
        let depth = self.config.depth;
        let dim = 1 << n_qubits;

        // Initialize state |0...0>
        let mut state = QuantumState::ground(dim);

        // Construct parameterized ansatz
        let mut ansatz = CircuitAnsatz::new(n_qubits, depth);
        let n_params = ansatz.param_count();
        let params = Self::generate_params(self.config.seed, n_params);
        ansatz.update_params(&params);

        let start_time = Instant::now();
        let mut layer_latencies = Vec::with_capacity(depth);

        // Execute depth layers tracking per-layer latency
        for layer_idx in 0..depth {
            let l_start = Instant::now();
            ansatz.forward_layer_into(layer_idx, &ansatz.weights, &mut state);
            layer_latencies.push(l_start.elapsed().as_secs_f64());

            if (layer_idx + 1) % 4 == 0 || layer_idx + 1 == depth {
                state.assert_normalized(1e-10);
            }
        }

        state.normalize();
        let total_time = start_time.elapsed().as_secs_f64();
        let avg_layer = if depth > 0 {
            total_time / depth as f64
        } else {
            0.0
        };

        let probs = state.probabilities();
        let pt_report = verify_porter_thomas(&probs);

        let mut indexed: Vec<(usize, f64)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_bitstrings: Vec<(String, f64)> = indexed
            .iter()
            .take(5)
            .map(|(idx, prob)| (format!("|{:0width$b}>", idx, width = n_qubits), *prob))
            .collect();

        let report = RcsRunReport {
            qubits: n_qubits,
            depth,
            dimension: dim,
            total_latency_secs: total_time,
            avg_layer_latency_secs: avg_layer,
            layer_latencies,
            pt_report,
            top_bitstrings,
        };

        (state, report)
    }

    /// Execute a noisy physical RCS pass under Lindbladian open-system decoherence and optional stabilization.
    pub fn execute_noisy(
        &self,
        noisy_cfg: &NoisyRcsConfig,
        ideal_state: Option<&QuantumState>,
    ) -> (QuantumState, NoisyRcsRunReport) {
        use rand::SeedableRng;
        let n_qubits = noisy_cfg.base.qubits;
        let depth = noisy_cfg.base.depth;
        let dim = 1 << n_qubits;

        // Ground state |0...0>
        let mut state = QuantumState::ground(dim);

        let mut ansatz = CircuitAnsatz::new(n_qubits, depth);
        let n_params = ansatz.param_count();
        let params = Self::generate_params(noisy_cfg.base.seed, n_params);
        ansatz.update_params(&params);

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(noisy_cfg.base.seed.wrapping_add(0xCAFE_BABE));
        let start_time = Instant::now();
        let mut trajectory = Vec::with_capacity(depth);

        // Precompute ideal state if not supplied
        let owned_ideal = if ideal_state.is_none() {
            let sim_ideal = RcsSimulator::new(noisy_cfg.base.clone());
            Some(sim_ideal.execute().0)
        } else {
            None
        };
        let ref_ideal = ideal_state.or(owned_ideal.as_ref());

        for layer_idx in 0..depth {
            let l_start = Instant::now();
            ansatz.forward_layer_into(layer_idx, &ansatz.weights, &mut state);

            // Noise & Stabilization Evolution
            let half_dt = noisy_cfg.gate_time_us * 0.5;
            let uses_dd = matches!(
                noisy_cfg.stabilization,
                StabilizationStrategy::DynamicalDecoupling(_)
                    | StabilizationStrategy::Combined(_)
            );

            if uses_dd && half_dt > 0.0 {
                // First half-step noise pass
                for q in 0..n_qubits {
                    crate::coherence::apply_open_system_decoherence_qubit(
                        &mut state,
                        q,
                        &noisy_cfg.noise,
                        half_dt,
                        &mut rng,
                    );
                }
                // Refocusing pulse to invert environmental dephasing phase
                for q in 0..n_qubits {
                    crate::coherence::apply_dynamical_decoupling(
                        &mut state,
                        q,
                        crate::coherence::DynamicalDecouplingKind::Cpmg,
                    );
                }
                // Second half-step noise pass (cancels first-half dephasing drift)
                for q in 0..n_qubits {
                    crate::coherence::apply_open_system_decoherence_qubit(
                        &mut state,
                        q,
                        &noisy_cfg.noise,
                        half_dt,
                        &mut rng,
                    );
                }
                // Restoring pulse
                for q in 0..n_qubits {
                    crate::coherence::apply_dynamical_decoupling(
                        &mut state,
                        q,
                        crate::coherence::DynamicalDecouplingKind::Cpmg,
                    );
                }
            } else if noisy_cfg.gate_time_us > 0.0 {
                // Unprotected naked noise pass
                for q in 0..n_qubits {
                    crate::coherence::apply_open_system_decoherence_qubit(
                        &mut state,
                        q,
                        &noisy_cfg.noise,
                        noisy_cfg.gate_time_us,
                        &mut rng,
                    );
                }
            }

            if matches!(
                noisy_cfg.stabilization,
                StabilizationStrategy::ManifoldReprojection | StabilizationStrategy::Combined(_)
            ) {
                crate::coherence::reproject_coherence_manifold(&mut state);
            }

            let l_time = l_start.elapsed().as_secs_f64();
            let coh = crate::coherence::compute_coherence_metrics(&state, n_qubits);
            let fid = if let Some(ideal) = ref_ideal {
                state.fidelity(ideal)
            } else {
                0.0
            };
            let probs = state.probabilities();
            let inst_xeb = t1000_core::porter_thomas::exact_linear_xeb(&probs);

            trajectory.push(RcsLayerTrajectory {
                layer: layer_idx,
                latency_secs: l_time,
                purity: coh.purity,
                l1_coherence_norm: coh.l1_normalized,
                fidelity_vs_ideal: fid,
                instant_xeb: inst_xeb,
            });
        }

        state.normalize();
        let total_time = start_time.elapsed().as_secs_f64();
        let noisy_probs = state.probabilities();
        let pt_report = verify_porter_thomas(&noisy_probs);

        let final_fid = if let Some(ideal) = ref_ideal {
            state.fidelity(ideal)
        } else {
            0.0
        };

        let ideal_probs = ref_ideal.map(|s| s.probabilities());
        let dist_metrics = if let Some(ref ip) = ideal_probs {
            Self::compare_distributions(ip, &noisy_probs)
        } else {
            DistributionMetrics {
                hellinger_distance: 0.0,
                squared_hellinger_distance: 0.0,
                bhattacharyya_coefficient: 1.0,
                cross_linear_xeb: 1.0,
                total_variation_distance: 0.0,
            }
        };

        let ideal_self_xeb = ideal_probs
            .as_ref()
            .map(|ip| t1000_core::porter_thomas::exact_linear_xeb(ip))
            .unwrap_or(1.0);
        let noisy_self_xeb = t1000_core::porter_thomas::exact_linear_xeb(&noisy_probs);

        let mut indexed: Vec<(usize, f64)> = noisy_probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_bitstrings: Vec<(String, f64)> = indexed
            .iter()
            .take(5)
            .map(|(idx, prob)| (format!("|{:0width$b}>", idx, width = n_qubits), *prob))
            .collect();

        let report = NoisyRcsRunReport {
            qubits: n_qubits,
            depth,
            dimension: dim,
            total_latency_secs: total_time,
            trajectory,
            final_fidelity_vs_ideal: final_fid,
            ideal_self_xeb,
            noisy_self_xeb,
            cross_xeb_with_ideal: dist_metrics.cross_linear_xeb,
            pt_report,
            distribution_metrics: dist_metrics,
            top_bitstrings,
        };

        (state, report)
    }

    /// Compute statistical divergence metrics between two probability distributions over the same Hilbert space.
    pub fn compare_distributions(p: &[f64], q: &[f64]) -> DistributionMetrics {
        assert_eq!(
            p.len(),
            q.len(),
            "Distributions must have identical Hilbert dimensions"
        );
        let dim = p.len();

        let mut bc = 0.0;
        let mut tvd = 0.0;
        let mut cross_xeb_sum = 0.0;

        for i in 0..dim {
            bc += (p[i] * q[i]).sqrt();
            tvd += (p[i] - q[i]).abs();
            cross_xeb_sum += p[i] * q[i];
        }

        let bc_clamped = bc.clamp(0.0, 1.0);
        let sq_hellinger = (1.0 - bc_clamped).max(0.0);
        let hellinger = sq_hellinger.sqrt();
        let cross_xeb = (dim as f64) * cross_xeb_sum - 1.0;

        DistributionMetrics {
            hellinger_distance: hellinger,
            squared_hellinger_distance: sq_hellinger,
            bhattacharyya_coefficient: bc,
            cross_linear_xeb: cross_xeb,
            total_variation_distance: 0.5 * tvd,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rcs_simulator_small_scale() {
        let config = RcsConfig {
            qubits: 10,
            depth: 6,
            seed: 12345,
        };
        let sim = RcsSimulator::new(config);
        let (state, report) = sim.execute();

        assert_eq!(state.dim, 1024);
        assert_eq!(report.qubits, 10);
        assert_eq!(report.dimension, 1024);
        assert_eq!(report.layer_latencies.len(), 6);
        assert!(report.total_latency_secs > 0.0);
        assert_eq!(report.top_bitstrings.len(), 5);

        // Verify probabilities sum to ~1.0
        let sum_p: f64 = state.probabilities().iter().sum();
        assert!((sum_p - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_distribution_comparison_identical() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let metrics = RcsSimulator::compare_distributions(&p, &p);

        assert!(metrics.hellinger_distance < 1e-6);
        assert!((metrics.bhattacharyya_coefficient - 1.0).abs() < 1e-6);
        assert!(metrics.total_variation_distance < 1e-6);
    }

    #[test]
    fn test_noisy_rcs_simulation_and_stabilization() {
        let base_cfg = RcsConfig {
            qubits: 8,
            depth: 4,
            seed: 777,
        };
        let sim = RcsSimulator::new(base_cfg.clone());
        let (ideal_state, ideal_report) = sim.execute();

        let noisy_cfg = NoisyRcsConfig {
            base: base_cfg,
            noise: crate::coherence::NoiseChannelParams {
                t1_us: 20.0,
                t2_us: 15.0,
                depolarizing_prob: 0.005,
            },
            gate_time_us: 0.20,
            stabilization: StabilizationStrategy::Combined(
                crate::coherence::DynamicalDecouplingKind::Xy4,
            ),
        };

        let (_noisy_state, noisy_report) = sim.execute_noisy(&noisy_cfg, Some(&ideal_state));

        assert_eq!(noisy_report.qubits, 8);
        assert_eq!(noisy_report.trajectory.len(), 4);
        assert!(noisy_report.final_fidelity_vs_ideal > 0.0);
        assert!(noisy_report.final_fidelity_vs_ideal <= 1.0);
        assert!(ideal_report.pt_report.exact_xeb > 0.0);
    }
}
