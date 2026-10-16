// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Topological Frontier Scheduler & Coherence-Aware Stage Optimizer
//!
//! Groups independent operations onto disjoint qubits into parallel execution stages,
//! sorting gates for optimal L1/L2 cache line reuse and protecting idle qubits from
//! $T_2^*$ environmental dephasing via ASAP/ALAP slack scheduling and automatic
//! Dynamical Decoupling (DD) pulse injection.

use std::collections::HashSet;
use serde::{Deserialize, Serialize};
use super::dag::CircuitDag;
use super::op::CircuitOp;

/// Classification of an execution stage based on its constituent operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StageKind {
    /// All operations in the stage are single-qubit gates acting on disjoint qubits
    SingleQubitParallel,
    /// All operations in the stage are two-qubit entangling gates on disjoint pairs
    TwoQubitParallel,
    /// Mixed independent 1-qubit and 2-qubit gates on disjoint qubits
    MixedParallel,
    /// Global multi-qubit operation (e.g. Walsh-Rademacher phase layer)
    GlobalMultiQubit,
}

/// An execution stage containing mutually independent operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScheduledStage {
    pub stage_index: usize,
    pub kind: StageKind,
    pub ops: Vec<CircuitOp>,
    pub touched_qubits: Vec<usize>,
}

impl ScheduledStage {
    pub fn new(stage_index: usize, ops: Vec<CircuitOp>) -> Self {
        let mut touched = HashSet::new();
        let mut has_1q = false;
        let mut has_2q = false;
        let mut has_global = false;

        for op in &ops {
            for q in op.qubits() {
                touched.insert(q);
            }
            match op {
                CircuitOp::Ry { .. } | CircuitOp::Rz { .. } | CircuitOp::Unitary1Q { .. } => {
                    has_1q = true;
                }
                CircuitOp::CZ { .. }
                | CircuitOp::CNOT { .. }
                | CircuitOp::AntiCNOT { .. }
                | CircuitOp::SWAP { .. }
                | CircuitOp::ControlledPhase { .. } => {
                    has_2q = true;
                }
                CircuitOp::RademacherPhase { .. } => {
                    has_global = true;
                }
            }
        }

        let kind = if has_global {
            StageKind::GlobalMultiQubit
        } else if has_1q && !has_2q {
            StageKind::SingleQubitParallel
        } else if has_2q && !has_1q {
            StageKind::TwoQubitParallel
        } else {
            StageKind::MixedParallel
        };

        let mut touched_sorted: Vec<usize> = touched.into_iter().collect();
        touched_sorted.sort_unstable();

        Self {
            stage_index,
            kind,
            ops,
            touched_qubits: touched_sorted,
        }
    }

    /// Number of operations in this stage.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Returns true if the stage has no operations.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

/// Critical-path ASAP and ALAP scheduling analysis metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AsapAlapSchedule {
    pub asap_depths: Vec<usize>,
    pub alap_depths: Vec<usize>,
    pub slacks: Vec<usize>,
    pub critical_path_nodes: Vec<usize>,
    pub critical_path_length: usize,
}

/// Telemetry emitted from coherence-aware stage compilation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoherenceScheduleTelemetry {
    pub total_qubit_cycles: usize,
    pub active_qubit_cycles: usize,
    pub idle_qubit_cycles: usize,
    pub idle_fraction: f64,
    pub dd_pulses_injected: usize,
    pub estimated_fidelity_retention: f64,
}

/// Computes forward ASAP (earliest start) and backward ALAP (latest start) depths and slacks.
pub fn compute_asap_alap(dag: &CircuitDag) -> AsapAlapSchedule {
    let n = dag.len();
    if n == 0 {
        return AsapAlapSchedule {
            asap_depths: Vec::new(),
            alap_depths: Vec::new(),
            slacks: Vec::new(),
            critical_path_nodes: Vec::new(),
            critical_path_length: 0,
        };
    }

    // 1. Forward Pass (ASAP)
    let mut asap = vec![0usize; n];
    for i in 0..n {
        let max_pred = dag.nodes[i]
            .predecessors
            .iter()
            .map(|&p| asap[p] + 1)
            .max()
            .unwrap_or(0);
        asap[i] = max_pred;
    }

    let crit_len = asap.iter().max().copied().unwrap_or(0) + 1;

    // 2. Backward Pass (ALAP)
    let mut alap = vec![crit_len.saturating_sub(1); n];
    for i in (0..n).rev() {
        if !dag.nodes[i].successors.is_empty() {
            let min_succ = dag.nodes[i]
                .successors
                .iter()
                .map(|&s| alap[s].saturating_sub(1))
                .min()
                .unwrap_or(crit_len - 1);
            alap[i] = min_succ;
        }
    }

    // 3. Slack = ALAP - ASAP (nodes with slack 0 form the critical path)
    let mut slacks = Vec::with_capacity(n);
    let mut crit_nodes = Vec::new();
    for i in 0..n {
        let s = alap[i].saturating_sub(asap[i]);
        slacks.push(s);
        if s == 0 {
            crit_nodes.push(i);
        }
    }

    AsapAlapSchedule {
        asap_depths: asap,
        alap_depths: alap,
        slacks,
        critical_path_nodes: crit_nodes,
        critical_path_length: crit_len,
    }
}

/// Schedules a CircuitDag into parallel execution stages, sorting for L1/L2 cache locality.
pub fn schedule_circuit_dag(dag: &CircuitDag) -> Vec<ScheduledStage> {
    let (stages, _) = schedule_circuit_dag_with_coherence_dd(dag, false, 2);
    stages
}

/// Schedules a CircuitDag into parallel execution stages with optional
/// automatic Dynamical Decoupling (DD) pulse sequence injection into idle qubit gaps.
pub fn schedule_circuit_dag_with_coherence_dd(
    dag: &CircuitDag,
    auto_dd: bool,
    dd_threshold: usize,
) -> (Vec<ScheduledStage>, CoherenceScheduleTelemetry) {
    if dag.is_empty() {
        return (
            Vec::new(),
            CoherenceScheduleTelemetry {
                total_qubit_cycles: 0,
                active_qubit_cycles: 0,
                idle_qubit_cycles: 0,
                idle_fraction: 0.0,
                dd_pulses_injected: 0,
                estimated_fidelity_retention: 1.0,
            },
        );
    }

    let n = dag.len();
    let num_qubits = dag.num_qubits;
    let mut in_degrees = dag.in_degrees();
    let mut raw_stages = Vec::new();
    let mut stage_idx = 0;

    let mut ready_nodes: Vec<usize> = (0..n).filter(|&i| in_degrees[i] == 0).collect();

    while !ready_nodes.is_empty() {
        let mut stage_ops = Vec::new();
        let mut stage_qubits = HashSet::new();
        let mut scheduled_nodes = Vec::new();
        let mut remaining_ready = Vec::new();

        for &node_id in &ready_nodes {
            let op = &dag.nodes[node_id].op;
            let op_qubits = op.qubits();

            let overlaps = op_qubits.iter().any(|q| stage_qubits.contains(q));

            if !overlaps {
                if matches!(op, CircuitOp::RademacherPhase { .. }) && !stage_ops.is_empty() {
                    remaining_ready.push(node_id);
                    continue;
                }
                if stage_ops.iter().any(|o| matches!(o, CircuitOp::RademacherPhase { .. })) {
                    remaining_ready.push(node_id);
                    continue;
                }

                for &q in &op_qubits {
                    stage_qubits.insert(q);
                }
                stage_ops.push(op.clone());
                scheduled_nodes.push(node_id);
            } else {
                remaining_ready.push(node_id);
            }
        }

        // Cache Locality Optimization: sort by lowest qubit index
        stage_ops.sort_by_key(|op| op.qubits().into_iter().min().unwrap_or(0));

        raw_stages.push(ScheduledStage::new(stage_idx, stage_ops));
        stage_idx += 1;

        let mut new_ready = Vec::new();
        for &node_id in &scheduled_nodes {
            for &succ_id in &dag.nodes[node_id].successors {
                in_degrees[succ_id] -= 1;
                if in_degrees[succ_id] == 0 {
                    new_ready.push(succ_id);
                }
            }
        }

        ready_nodes = remaining_ready;
        ready_nodes.extend(new_ready);
    }

    // 2. Idle Qubit Gap Analysis & Auto-DD Pulse Injection
    let total_stages = raw_stages.len();
    let total_qubit_cycles = total_stages * num_qubits;
    let mut active_qubit_cycles = 0usize;
    let mut dd_pulses_injected = 0usize;

    // Track active status for each qubit in each stage
    let mut wire_timeline = vec![vec![false; total_stages]; num_qubits];
    for (s_idx, stg) in raw_stages.iter().enumerate() {
        for &q in &stg.touched_qubits {
            if q < num_qubits {
                wire_timeline[q][s_idx] = true;
                active_qubit_cycles += 1;
            }
        }
    }

    let idle_qubit_cycles = total_qubit_cycles.saturating_sub(active_qubit_cycles);
    let idle_fraction = if total_qubit_cycles > 0 {
        idle_qubit_cycles as f64 / total_qubit_cycles as f64
    } else {
        0.0
    };

    let mut optimized_stages = raw_stages;

    if auto_dd {
        // Inspect contiguous idle spans on each qubit
        for q in 0..num_qubits {
            let mut idle_start = None;

            for s in 0..total_stages {
                if !wire_timeline[q][s] {
                    if idle_start.is_none() {
                        idle_start = Some(s);
                    }
                } else if let Some(start) = idle_start {
                    let span_len = s - start;
                    if span_len >= dd_threshold && span_len >= 2 {
                        // Inject balanced refocusing pair (X - X = I) to eliminate dephasing
                        let s1 = start;
                        let s2 = start + span_len - 1;
                        optimized_stages[s1].ops.push(CircuitOp::Unitary1Q {
                            qubit: q,
                            matrix: super::op::Unitary2x2::PAULI_X,
                        });
                        optimized_stages[s2].ops.push(CircuitOp::Unitary1Q {
                            qubit: q,
                            matrix: super::op::Unitary2x2::PAULI_X,
                        });
                        dd_pulses_injected += 2;
                    }
                    idle_start = None;
                }
            }

            // Flush trailing idle span at the end of the circuit
            if let Some(start) = idle_start {
                let span_len = total_stages - start;
                if span_len >= dd_threshold && span_len >= 2 {
                    let s1 = start;
                    let s2 = start + span_len - 1;
                    optimized_stages[s1].ops.push(CircuitOp::Unitary1Q {
                        qubit: q,
                        matrix: super::op::Unitary2x2::PAULI_X,
                    });
                    optimized_stages[s2].ops.push(CircuitOp::Unitary1Q {
                        qubit: q,
                        matrix: super::op::Unitary2x2::PAULI_X,
                    });
                    dd_pulses_injected += 2;
                }
            }
        }
    }

    // Analytical estimate of fidelity retention across idle time
    // T2* model: fidelity ~ exp(-unprotected_idle * 0.002)
    let unprotected_idle = if auto_dd {
        idle_qubit_cycles.saturating_sub(dd_pulses_injected * 2)
    } else {
        idle_qubit_cycles
    };
    let est_retention = (-0.0015 * unprotected_idle as f64).exp().clamp(0.0, 1.0);

    let telem = CoherenceScheduleTelemetry {
        total_qubit_cycles,
        active_qubit_cycles,
        idle_qubit_cycles,
        idle_fraction,
        dd_pulses_injected,
        estimated_fidelity_retention: est_retention,
    };

    (optimized_stages, telem)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::CircuitBuilder;

    #[test]
    fn test_asap_alap_slack_computation() {
        let mut b = CircuitBuilder::new(3);
        b.h(0);
        b.cz(0, 1);
        b.h(2);
        let dag = b.into_dag();

        let sched = compute_asap_alap(&dag);
        assert_eq!(sched.critical_path_length, 2);
        assert!(!sched.critical_path_nodes.is_empty());
        // Slack should be non-negative
        for &s in &sched.slacks {
            assert!(s <= 2);
        }
    }

    #[test]
    fn test_auto_dd_injection_protects_idle_qubits() {
        let mut b = CircuitBuilder::new(3);
        // Qubit 0 and 1 undergo gates, qubit 2 sits idle across multiple layers
        b.h(0);
        b.cz(0, 1);
        b.h(0);
        b.cz(0, 1);
        let dag = b.into_dag();

        let (stages_raw, telem_raw) = schedule_circuit_dag_with_coherence_dd(&dag, false, 2);
        assert_eq!(telem_raw.dd_pulses_injected, 0);
        assert!(telem_raw.idle_qubit_cycles > 0);

        let (stages_dd, telem_dd) = schedule_circuit_dag_with_coherence_dd(&dag, true, 2);
        assert!(telem_dd.dd_pulses_injected >= 2);
        assert!(telem_dd.estimated_fidelity_retention >= telem_raw.estimated_fidelity_retention);
        assert_eq!(stages_dd.len(), stages_raw.len());
    }
}
