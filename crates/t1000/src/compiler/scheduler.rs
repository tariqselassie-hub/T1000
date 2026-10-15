//! Topological Frontier Scheduler and Cache-Blocked Stage Grouping Engine.
//! Groups independent operations onto disjoint qubits into parallel execution stages,
//! sorting gates for optimal L1/L2/L3 cache line reuse.

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

/// Schedules a CircuitDag into a sequence of ScheduledStages.
/// Automatically groups independent operations onto disjoint qubits together,
/// ensuring that no matter how chaotic the input code is, the hardware cache footprint
/// is fully optimized.
pub fn schedule_circuit_dag(dag: &CircuitDag) -> Vec<ScheduledStage> {
    if dag.is_empty() {
        return Vec::new();
    }

    let n = dag.len();
    let mut in_degrees = dag.in_degrees();
    let mut stages = Vec::new();
    let mut stage_idx = 0;

    // Ready queue contains all nodes whose incoming causal dependencies are satisfied (in_degree == 0)
    let mut ready_nodes: Vec<usize> = (0..n).filter(|&i| in_degrees[i] == 0).collect();

    while !ready_nodes.is_empty() {
        // From the ready set, greedily select a maximal set of operations that act on DISJOINT qubits.
        let mut stage_ops = Vec::new();
        let mut stage_qubits = HashSet::new();
        let mut scheduled_nodes = Vec::new();
        let mut remaining_ready = Vec::new();

        for &node_id in &ready_nodes {
            let op = &dag.nodes[node_id].op;
            let op_qubits = op.qubits();

            // Check if this op shares any qubits with already-selected ops in this stage
            let overlaps = op_qubits.iter().any(|q| stage_qubits.contains(q));

            if !overlaps {
                // If it's a global operation and stage already has ops, postpone to next stage
                if matches!(op, CircuitOp::RademacherPhase { .. }) && !stage_ops.is_empty() {
                    remaining_ready.push(node_id);
                    continue;
                }

                // If stage already has a global operation, no other operation can be added
                if stage_ops.iter().any(|o| matches!(o, CircuitOp::RademacherPhase { .. })) {
                    remaining_ready.push(node_id);
                    continue;
                }

                // Add to current stage
                for &q in &op_qubits {
                    stage_qubits.insert(q);
                }
                stage_ops.push(op.clone());
                scheduled_nodes.push(node_id);
            } else {
                remaining_ready.push(node_id);
            }
        }

        // Cache Locality Optimization: Sort operations within the stage
        // Placing operations on lower-index qubits first maximizes spatial L1/L2 cache line hits
        // because lower qubits stride across smaller memory deltas (inner memory blocks).
        stage_ops.sort_by_key(|op| {
            op.qubits().into_iter().min().unwrap_or(0)
        });

        stages.push(ScheduledStage::new(stage_idx, stage_ops));
        stage_idx += 1;

        // Retire scheduled nodes and decrement in-degrees of their successors
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

    stages
}
