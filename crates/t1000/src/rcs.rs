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

    /// Execute an advanced RCS circuit pass, returning the final state and execution telemetry.
    pub fn execute(&self) -> (QuantumState, RcsRunReport) {
        let n_qubits = self.config.qubits;
        let depth = self.config.depth;
        let dim = 1 << n_qubits;

        // Initialize state |0...0>
        let mut state = QuantumState::zero(dim);
        state.real[0] = 1.0;
        state.imag[0] = 0.0;

        // Construct parameterized ansatz
        let mut ansatz = CircuitAnsatz::new(n_qubits, depth);
        let n_params = ansatz.param_count();

        // Pseudo-random deterministic parameter population from seed
        let mut rng_state = self.config.seed;
        let mut next_f64 = || {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let val = (rng_state >> 11) as f64 / (1u64 << 53) as f64;
            val * std::f64::consts::TAU - std::f64::consts::PI
        };

        let params: Vec<f64> = (0..n_params).map(|_| next_f64()).collect();
        ansatz.update_params(&params);

        let start_time = Instant::now();
        let mut layer_latencies = Vec::with_capacity(depth);

        // Execute depth layers tracking per-layer latency
        for layer_idx in 0..depth {
            let l_start = Instant::now();
            ansatz.forward_layer_into(layer_idx, &ansatz.weights, &mut state);
            layer_latencies.push(l_start.elapsed().as_secs_f64());

            // Periodic L2 norm verification to verify numerical stability under deep randomized benchmarking passes
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

        // Compute probabilities
        let probs = state.probabilities();

        // Porter-Thomas analysis
        let pt_report = verify_porter_thomas(&probs);

        // Extract top bitstrings
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
}
