// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Unified Quantum Telemetry and Cross-Frontier Verification Schema
//!
//! Consolidates diagnostic outputs across all simulation paradigms in `T-1000`:
//! - Pure state coherence metrics ($C_{\ell_1}$, purity, bipartite entropy).
//! - Statistical Porter-Thomas chaos metrics (XEB fidelity, Kolmogorov-Smirnov distance $D_{\text{KS}}$).
//! - Circuit compiler telemetry (DAG fusion, stage count, cache pass reduction).
//! - Hardware and timing latency metrics.

use crate::coherence::{compute_coherence_metrics, CoherenceMetrics as StateCoherenceMetrics};
use crate::compiler::CompilerTelemetry;
use crate::state::QuantumState;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;
use t1000_core::porter_thomas::{verify_porter_thomas, PorterThomasReport};

/// Comprehensive unified telemetry report for quantum states and circuit evolutions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantumTelemetryReport {
    /// Dimension of the Hilbert statevector ($2^n$)
    pub register_dim: usize,
    /// Number of physical qubits
    pub n_qubits: usize,
    /// Coherence and purity diagnostics
    pub coherence: StateCoherenceMetrics,
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
pub struct ExecutionMetrics {
    pub qubit_count: u32,
    pub active_threads: u32,
    pub latency_ms: f64,
    pub baseline_pyqrack_ms: f64,
    pub latency_multiplier: f64,
}

pub type ExecutionTelemetry = ExecutionMetrics;

/// Quantum state coherence and chaos verification telemetry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoherenceMetrics {
    pub xeb_score: f64,
    pub hog_probability: f64,
    pub phase_fidelity: f64,
}

pub type CoherenceTelemetry = CoherenceMetrics;

/// Circuit optimization and wire recycling telemetry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizationMetrics {
    pub mbu_wires_recycled: u32,
    pub toffolis_pruned: u32,
    pub gate_reduction_ratio: f64,
    pub public_leaderboard_score: f64,
}

pub type OptimizationTelemetry = OptimizationMetrics;

/// Continuous expectation streaming telemetry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamingMetrics {
    pub accumulator_type: String,
    pub buffer_cache_hit_rate: f64,
    pub pauli_frame_drift: f64,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub variance_multipliers: Option<crate::measurements::VarianceMultipliers>,
}

pub type StreamingTelemetry = StreamingMetrics;

/// Composite diagnostic metrics container.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Metrics {
    pub execution: ExecutionMetrics,
    pub coherence: CoherenceMetrics,
    pub optimization: OptimizationMetrics,
    pub streaming: StreamingMetrics,
}

pub type DiagnosticMetrics = Metrics;

/// Strongly-typed telemetry payload log structure for T-1000 diagnostic streams.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TelemetryPayload {
    pub timestamp: String,
    pub engine: String,
    pub mode: String,
    pub architecture: String,
    pub metrics: Metrics,
    pub status: String,
}

pub type DiagnosticTelemetryEvent = TelemetryPayload;

impl TelemetryPayload {
    /// Constructs a standardized diagnostic log event and evaluates coherence lock status.
    pub fn new(metrics: Metrics) -> Self {
        let status = if metrics.coherence.xeb_score >= 0.999 && metrics.coherence.hog_probability >= 0.975 {
            "nominal_locked".to_string()
        } else if metrics.coherence.xeb_score >= 0.95 && metrics.coherence.hog_probability >= 0.85 {
            "coherence_degraded".to_string()
        } else {
            "dephasing_breached".to_string()
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
        out.push_str(&format!("│ Coherence Lock Status        │ {:>29} │\n", self.status));
        out.push_str("└──────────────────────────────┴───────────────────────────────┘\n");
        out
    }
}

/// Evaluation errors encountered during telemetry ingestion and invariant verification.
#[derive(Debug)]
pub enum TelemetryError {
    JsonParseError(serde_json::Error),
    XebThresholdViolation(f64),
    LatencyMultiplierLow(f64),
}

impl fmt::Display for TelemetryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TelemetryError::JsonParseError(e) => write!(f, "JSON parse error: {}", e),
            TelemetryError::XebThresholdViolation(xeb) => {
                write!(f, "XEB score {:.4} breached safety floor (requires >= 0.999)", xeb)
            }
            TelemetryError::LatencyMultiplierLow(m) => {
                write!(f, "Latency multiplier {:.2}x below target threshold", m)
            }
        }
    }
}

impl Error for TelemetryError {}

/// Ingests raw JSON log strings, parses them into strongly-typed structures,
/// and evaluates performance and coherence invariants for T-1000.
pub fn ingest_and_evaluate_telemetry(json_data: &str) -> Result<TelemetryPayload, TelemetryError> {
    // 1. Zero-copy / fast string parsing via serde_json
    let payload: TelemetryPayload = serde_json::from_str(json_data)
        .map_err(TelemetryError::JsonParseError)?;

    // 2. Evaluate Coherence Floor (XEB >= 0.999)
    let xeb = payload.metrics.coherence.xeb_score;
    if xeb < 0.999 {
        return Err(TelemetryError::XebThresholdViolation(xeb));
    }

    // 3. Evaluate Execution Latency Multiplier against baseline
    let multiplier = payload.metrics.execution.latency_multiplier;
    if multiplier < 1000.0 {
        return Err(TelemetryError::LatencyMultiplierLow(multiplier));
    }

    // 4. Log successful validation match
    println!(
        "[T-1000 ENGINE] Telemetry validated successfully. Mode: {}, XEB: {:.4}, Multiplier: {:.2}x",
        payload.mode, xeb, multiplier
    );

    Ok(payload)
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
                public_leaderboard_score: 1.28673,
            },
            streaming: StreamingTelemetry {
                accumulator_type: "avx2_non_temporal_stream".to_string(),
                buffer_cache_hit_rate: 0.991,
                pauli_frame_drift: 0.00004,
                variance_multipliers: None,
            },
        };

        let event = DiagnosticTelemetryEvent::new(metrics);
        assert_eq!(event.status, "nominal_locked");
        assert_eq!(event.metrics.execution.qubit_count, 26);
        assert_eq!(event.metrics.optimization.toffolis_pruned, 4128);

        let json = event.to_json().expect("JSON serialization failed");
        assert!(json.contains("\"latency_multiplier\": 11762.23"));
        assert!(json.contains("\"status\": \"nominal_locked\""));

        let ascii = event.render_ascii();
        assert!(ascii.contains("T-1000 DIAGNOSTIC TELEMETRY EVENT"));
        assert!(ascii.contains("11762.23x"));

        // Test Ingestion and Evaluation Pipeline
        let ingested = ingest_and_evaluate_telemetry(&json).expect("Ingest and evaluate failed");
        assert_eq!(ingested.metrics.coherence.xeb_score, 0.9992);
        assert_eq!(ingested.metrics.execution.latency_multiplier, 11762.23);

        // Test XEB violation detection
        let bad_xeb_json = json.replace("0.9992", "0.9850");
        let err = ingest_and_evaluate_telemetry(&bad_xeb_json).unwrap_err();
        match err {
            TelemetryError::XebThresholdViolation(xeb) => assert_eq!(xeb, 0.9850),
            _ => panic!("Expected XebThresholdViolation"),
        }

        // Test Latency multiplier violation detection
        let bad_mult_json = json.replace("11762.23", "850.0");
        let err = ingest_and_evaluate_telemetry(&bad_mult_json).unwrap_err();
        match err {
            TelemetryError::LatencyMultiplierLow(m) => assert_eq!(m, 850.0),
            _ => panic!("Expected LatencyMultiplierLow"),
        }
    }
}
