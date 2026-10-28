// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Unified Quantum Telemetry and Cross-Frontier Verification Schema
//!
//! Consolidates diagnostic outputs across all simulation paradigms in `T-1000`:
//! - Pure state coherence metrics ($C_{\ell_1}$, purity, bipartite entropy).
//! - Statistical Porter-Thomas chaos metrics (XEB fidelity, Kolmogorov-Smirnov distance $D_{\text{KS}}$).
//! - Circuit compiler telemetry (DAG fusion, stage count, cache pass reduction).
//! - Hardware and timing latency metrics.

use crate::coherence::{compute_coherence_metrics, CoherenceMetrics};
use crate::compiler::CompilerTelemetry;
use crate::state::QuantumState;
use serde::{Deserialize, Serialize};
use t1000_core::porter_thomas::{verify_porter_thomas, PorterThomasReport};

/// Comprehensive unified telemetry report for quantum states and circuit evolutions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantumTelemetryReport {
    /// Dimension of the Hilbert statevector ($2^n$)
    pub register_dim: usize,
    /// Number of physical qubits
    pub n_qubits: usize,
    /// Coherence and purity diagnostics
    pub coherence: CoherenceMetrics,
    /// Porter-Thomas chaos and XEB metrics (if evaluated)
    pub porter_thomas: Option<PorterThomasReport>,
    /// Circuit compiler optimizations (if derived from compiled circuit)
    pub compiler: Option<CompilerTelemetry>,
    /// Wall-clock execution time in microseconds
    pub execution_time_us: f64,
}

impl QuantumTelemetryReport {
    /// Generates telemetry from an active `QuantumState`.
    pub fn from_state(state: &QuantumState, n_qubits: usize) -> Self {
        assert_eq!(
            state.dim,
            1 << n_qubits,
            "State dim {} does not match qubit count {}",
            state.dim,
            n_qubits
        );
        let coherence = compute_coherence_metrics(state, n_qubits);
        Self {
            register_dim: state.dim,
            n_qubits,
            coherence,
            porter_thomas: None,
            compiler: None,
            execution_time_us: 0.0,
        }
    }

    /// Evaluates Porter-Thomas chaos conformity and XEB against the state's measurement distribution.
    pub fn with_porter_thomas(mut self, state: &QuantumState) -> Self {
        let probs = state.probabilities();
        self.porter_thomas = Some(verify_porter_thomas(&probs));
        self
    }

    /// Attaches circuit compiler telemetry to the report.
    pub fn with_compiler(mut self, compiler: CompilerTelemetry) -> Self {
        self.compiler = Some(compiler);
        self
    }

    /// Attaches measured execution elapsed time in microseconds.
    pub fn with_execution_time(mut self, time_us: f64) -> Self {
        self.execution_time_us = time_us;
        self
    }

    /// Serializes telemetry report to formatted JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Renders a clean Unicode box-drawing summary table adhering to Deen OS standards.
    pub fn render_ascii_summary(&self) -> String {
        let mut out = String::new();
        out.push_str("┌──────────────────────────────────────────────────────────────┐\n");
        out.push_str("│             T-1000 QUANTUM TELEMETRY REPORT                  │\n");
        out.push_str("├──────────────────────────────┬───────────────────────────────┤\n");
        out.push_str(&format!(
            "│ Qubit Count / State Dim      │ {:>2} Qubits ({:>10} states) │\n",
            self.n_qubits, self.register_dim
        ));
        out.push_str(&format!(
            "│ State Purity Tr(rho^2)       │ {:>29.6} │\n",
            self.coherence.purity
        ));
        out.push_str(&format!(
            "│ Normalized Coherence C_norm  │ {:>29.6} │\n",
            self.coherence.l1_normalized
        ));
        out.push_str(&format!(
            "│ Bipartite Entropy (bits)     │ {:>29.6} │\n",
            self.coherence.von_neumann_entropy
        ));
        out.push_str(&format!(
            "│ Shannon Entropy (nats)       │ {:>29.6} │\n",
            self.coherence.shannon_entropy_nats
        ));

        if let Some(ref pt) = self.porter_thomas {
            out.push_str("├──────────────────────────────┼───────────────────────────────┤\n");
            out.push_str(&format!(
                "│ Exact Linear XEB Fidelity    │ {:>29.6} │\n",
                pt.exact_xeb
            ));
            out.push_str(&format!(
                "│ Kolmogorov-Smirnov Distance  │ {:>29.6} │\n",
                pt.ks_statistic
            ));
            out.push_str(&format!(
                "│ Porter-Thomas Conformity     │ {:>29} │\n",
                if pt.is_porter_thomas_conforming {
                    "CONFORMS (PASSED)"
                } else {
                    "NON-CONFORMANT"
                }
            ));
        }

        if let Some(ref cmp) = self.compiler {
            out.push_str("├──────────────────────────────┼───────────────────────────────┤\n");
            out.push_str(&format!(
                "│ Raw Ops -> Fused Ops         │ {:>13} -> {:>12} │\n",
                cmp.raw_ops, cmp.fused_ops
            ));
            out.push_str(&format!(
                "│ Grouped Execution Stages     │ {:>29} │\n",
                cmp.stage_count
            ));
            out.push_str(&format!(
                "│ Cache Pass Reduction         │ {:>28.2}% │\n",
                cmp.cache_pass_reduction_pct
            ));
        }

        if self.execution_time_us > 0.0 {
            out.push_str("├──────────────────────────────┼───────────────────────────────┤\n");
            out.push_str(&format!(
                "│ Execution Latency            │ {:>26.2} us │\n",
                self.execution_time_us
            ));
        }

        out.push_str("└──────────────────────────────┴───────────────────────────────┘\n");
        out
    }
}

/// Execution timing and acceleration telemetry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionTelemetry {
    pub qubit_count: usize,
    pub active_threads: usize,
    pub latency_ms: f64,
    pub baseline_pyqrack_ms: f64,
    pub latency_multiplier: f64,
}

/// Quantum state coherence and chaos verification telemetry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoherenceTelemetry {
    pub xeb_score: f64,
    pub hog_probability: f64,
    pub phase_fidelity: f64,
}

/// Circuit optimization and wire recycling telemetry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizationTelemetry {
    pub mbu_wires_recycled: usize,
    pub toffolis_pruned: usize,
    pub gate_reduction_ratio: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub public_leaderboard_score: Option<f64>,
}

/// Continuous expectation streaming telemetry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamingTelemetry {
    pub accumulator_type: String,
    pub buffer_cache_hit_rate: f64,
    pub pauli_frame_drift: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variance_multipliers: Option<crate::measurements::VarianceMultipliers>,
}

/// Composite diagnostic metrics container.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticMetrics {
    pub execution: ExecutionTelemetry,
    pub coherence: CoherenceTelemetry,
    pub optimization: OptimizationTelemetry,
    pub streaming: StreamingTelemetry,
}

/// System operational and coherence verification lock status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TelemetryStatus {
    /// Strictly adheres to XEB >= 0.999 and HOG >= 0.975 floors
    NominalLocked,
    /// XEB or HOG below floor but above critical dephasing threshold
    CoherenceDegraded,
    /// State has suffered destructive collapse or severe dephasing
    DephasingBreached,
}

/// Standardized structured diagnostic log event for real-time telemetry streaming.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticTelemetryEvent {
    pub timestamp: String,
    pub engine: String,
    pub mode: String,
    pub architecture: String,
    pub metrics: DiagnosticMetrics,
    pub status: TelemetryStatus,
}

impl DiagnosticTelemetryEvent {
    /// Constructs a standardized diagnostic log event and evaluates coherence lock status.
    pub fn new(metrics: DiagnosticMetrics) -> Self {
        let status = if metrics.coherence.xeb_score >= 0.999 && metrics.coherence.hog_probability >= 0.975 {
            TelemetryStatus::NominalLocked
        } else if metrics.coherence.xeb_score >= 0.95 && metrics.coherence.hog_probability >= 0.85 {
            TelemetryStatus::CoherenceDegraded
        } else {
            TelemetryStatus::DephasingBreached
        };

        Self {
            timestamp: "2026-10-28T07:15:00.124Z".to_string(),
            engine: "t1000-core".to_string(),
            mode: "phase-chained-streaming".to_string(),
            architecture: "balanced-heptal-deen".to_string(),
            metrics,
            status,
        }
    }

    /// Serializes event to formatted JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Renders an executive ASCII box-drawing readout.
    pub fn render_ascii(&self) -> String {
        let mut out = String::new();
        out.push_str("┌──────────────────────────────────────────────────────────────┐\n");
        out.push_str("│             T-1000 DIAGNOSTIC TELEMETRY EVENT                │\n");
        out.push_str("├──────────────────────────────┬───────────────────────────────┤\n");
        out.push_str(&format!("│ Engine / Mode                │ {:>10} / {:>14} │\n", self.engine, self.mode));
        out.push_str(&format!("│ Architecture                 │ {:>29} │\n", self.architecture));
        out.push_str(&format!("│ Qubits / Latency             │ {:>2} Qubits ({:>10.3} ms) │\n", self.metrics.execution.qubit_count, self.metrics.execution.latency_ms));
        out.push_str(&format!("│ Latency Multiplier vs PyQrack│ {:>28.2}x │\n", self.metrics.execution.latency_multiplier));
        out.push_str(&format!("│ XEB Score / HOG Floor        │ {:>8.4} / {:>10.4} │\n", self.metrics.coherence.xeb_score, self.metrics.coherence.hog_probability));
        out.push_str(&format!("│ MBU Recycled / Pruned Gates  │ {:>6} Wires / {:>6} Gates │\n", self.metrics.optimization.mbu_wires_recycled, self.metrics.optimization.toffolis_pruned));
        out.push_str(&format!("│ Gate Reduction Ratio         │ {:>28.2}x │\n", self.metrics.optimization.gate_reduction_ratio));
        out.push_str(&format!("│ Coherence Lock Status        │ {:>29?} │\n", self.status));
        out.push_str("└──────────────────────────────┴───────────────────────────────┘\n");
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_report_generation() {
        let n_qubits = 4;
        let dim = 1 << n_qubits;
        let state = QuantumState::new(dim);

        let report = QuantumTelemetryReport::from_state(&state, n_qubits)
            .with_porter_thomas(&state)
            .with_execution_time(12.5);

        assert_eq!(report.n_qubits, 4);
        assert_eq!(report.register_dim, 16);
        assert!((report.coherence.l1_normalized - 1.0).abs() < 1e-6);
        assert!(report.porter_thomas.is_some());
        assert_eq!(report.execution_time_us, 12.5);

        let table = report.render_ascii_summary();
        assert!(table.contains("T-1000 QUANTUM TELEMETRY REPORT"));
        assert!(table.contains("4 Qubits"));

        let json = report.to_json().expect("Serialization failed");
        assert!(json.contains("coherence"));
    }

    #[test]
    fn test_diagnostic_telemetry_event_nominal_and_json() {
        let metrics = DiagnosticMetrics {
            execution: ExecutionTelemetry {
                qubit_count: 26,
                active_threads: 8,
                latency_ms: 0.042,
                baseline_pyqrack_ms: 494.014,
                latency_multiplier: 11762.23,
            },
            coherence: CoherenceTelemetry {
                xeb_score: 0.9992,
                hog_probability: 0.9754,
                phase_fidelity: 0.9981,
            },
            optimization: OptimizationTelemetry {
                mbu_wires_recycled: 217,
                toffolis_pruned: 4128,
                gate_reduction_ratio: 14.85,
                public_leaderboard_score: Some(1.28673),
            },
            streaming: StreamingTelemetry {
                accumulator_type: "avx2_non_temporal_stream".to_string(),
                buffer_cache_hit_rate: 0.991,
                pauli_frame_drift: 0.00004,
                variance_multipliers: None,
            },
        };

        let event = DiagnosticTelemetryEvent::new(metrics);
        assert_eq!(event.status, TelemetryStatus::NominalLocked);
        assert_eq!(event.metrics.execution.qubit_count, 26);
        assert_eq!(event.metrics.optimization.toffolis_pruned, 4128);

        let json = event.to_json().expect("JSON serialization failed");
        assert!(json.contains("\"latency_multiplier\": 11762.23"));
        assert!(json.contains("\"status\": \"nominal_locked\""));

        let ascii = event.render_ascii();
        assert!(ascii.contains("T-1000 DIAGNOSTIC TELEMETRY EVENT"));
        assert!(ascii.contains("11762.23x"));
    }
}
