// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Unified Quantum Telemetry and Cross-Frontier Verification Schema
//!
//! Consolidates diagnostic outputs across all simulation paradigms in `T-1000`:
//! - Pure state coherence metrics ($C_{\ell_1}$, purity, bipartite entropy).
//! - Statistical Porter-Thomas chaos metrics (XEB fidelity, Kolmogorov-Smirnov distance $D_{\text{KS}}$).
//! - Circuit compiler telemetry (DAG fusion, stage count, cache pass reduction).
//! - Hardware and timing latency metrics.

use serde::{Deserialize, Serialize};
use crate::coherence::{compute_coherence_metrics, CoherenceMetrics};
use crate::compiler::CompilerTelemetry;
use crate::state::QuantumState;
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
        out.push_str(&format!("│ Qubit Count / State Dim      │ {:>2} Qubits ({:>10} states) │\n", self.n_qubits, self.register_dim));
        out.push_str(&format!("│ State Purity Tr(rho^2)       │ {:>29.6} │\n", self.coherence.purity));
        out.push_str(&format!("│ Normalized Coherence C_norm  │ {:>29.6} │\n", self.coherence.l1_normalized));
        out.push_str(&format!("│ Bipartite Entropy (bits)     │ {:>29.6} │\n", self.coherence.von_neumann_entropy));
        out.push_str(&format!("│ Shannon Entropy (nats)       │ {:>29.6} │\n", self.coherence.shannon_entropy_nats));

        if let Some(ref pt) = self.porter_thomas {
            out.push_str("├──────────────────────────────┼───────────────────────────────┤\n");
            out.push_str(&format!("│ Exact Linear XEB Fidelity    │ {:>29.6} │\n", pt.exact_xeb));
            out.push_str(&format!("│ Kolmogorov-Smirnov Distance  │ {:>29.6} │\n", pt.ks_statistic));
            out.push_str(&format!("│ Porter-Thomas Conformity     │ {:>29} │\n", if pt.is_porter_thomas_conforming { "CONFORMS (PASSED)" } else { "NON-CONFORMANT" }));
        }

        if let Some(ref cmp) = self.compiler {
            out.push_str("├──────────────────────────────┼───────────────────────────────┤\n");
            out.push_str(&format!("│ Raw Ops -> Fused Ops         │ {:>13} -> {:>12} │\n", cmp.raw_ops, cmp.fused_ops));
            out.push_str(&format!("│ Grouped Execution Stages     │ {:>29} │\n", cmp.stage_count));
            out.push_str(&format!("│ Cache Pass Reduction         │ {:>28.2}% │\n", cmp.cache_pass_reduction_pct));
        }

        if self.execution_time_us > 0.0 {
            out.push_str("├──────────────────────────────┼───────────────────────────────┤\n");
            out.push_str(&format!("│ Execution Latency            │ {:>26.2} us │\n", self.execution_time_us));
        }

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
}
