//! Gate Fusion and Algebraic Simplification Optimizer.
//! Collapses chains of single-qubit operations into compound 2x2 unitaries,
//! reducing statevector memory bandwidth and cache evictions.

use super::dag::CircuitDag;
use super::op::CircuitOp;

/// Statistics on optimizations achieved by the fusion pass.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FusionStats {
    pub ops_before: usize,
    pub ops_after: usize,
    pub single_qubit_fused: usize,
    pub rz_merged: usize,
    pub ry_merged: usize,
    pub identities_removed: usize,
}

impl FusionStats {
    pub fn reduction_ratio(&self) -> f64 {
        if self.ops_before == 0 {
            1.0
        } else {
            self.ops_before as f64 / self.ops_after.max(1) as f64
        }
    }
}

/// Optimizes a CircuitDag by fusing consecutive operations along each qubit wire.
pub fn optimize_dag_fusion(dag: &CircuitDag) -> (CircuitDag, FusionStats) {
    let mut stats = FusionStats {
        ops_before: dag.len(),
        ..Default::default()
    };

    let mut new_dag = CircuitDag::new(dag.num_qubits);
    if dag.is_empty() {
        stats.ops_after = 0;
        return (new_dag, stats);
    }

    // Step 1: Linear wire-by-wire gate accumulation with fusion
    // Because single-qubit gates on a qubit wire between entangling boundaries
    // can be combined into a single compound Unitary2x2.
    // We walk the topological order of the original DAG.
    let mut pending_1q: Vec<Option<CircuitOp>> = vec![None; dag.num_qubits];

    for node in &dag.nodes {
        match &node.op {
            CircuitOp::Ry { qubit, .. }
            | CircuitOp::Rz { qubit, .. }
            | CircuitOp::Unitary1Q { qubit, .. } => {
                let q = *qubit;
                if let Some(existing) = pending_1q[q].take() {
                    if let Some(fused) = existing.try_fuse_single_qubit(&node.op) {
                        match (&existing, &node.op) {
                            (CircuitOp::Rz { .. }, CircuitOp::Rz { .. }) => stats.rz_merged += 1,
                            (CircuitOp::Ry { .. }, CircuitOp::Ry { .. }) => stats.ry_merged += 1,
                            _ => stats.single_qubit_fused += 1,
                        }
                        pending_1q[q] = Some(fused);
                    } else {
                        // Could not fuse, emit existing
                        flush_single_qubit(&mut new_dag, existing, &mut stats);
                        pending_1q[q] = Some(node.op.clone());
                    }
                } else {
                    pending_1q[q] = Some(node.op.clone());
                }
            }
            CircuitOp::CZ { q1, q2 } => {
                // Entangling barrier on q1 and q2: must flush any pending 1Q gates on both qubits first
                if let Some(op1) = pending_1q[*q1].take() {
                    flush_single_qubit(&mut new_dag, op1, &mut stats);
                }
                if let Some(op2) = pending_1q[*q2].take() {
                    flush_single_qubit(&mut new_dag, op2, &mut stats);
                }
                new_dag.add_op(node.op.clone());
            }
            CircuitOp::CNOT { control, target } => {
                if let Some(op_c) = pending_1q[*control].take() {
                    flush_single_qubit(&mut new_dag, op_c, &mut stats);
                }
                if let Some(op_t) = pending_1q[*target].take() {
                    flush_single_qubit(&mut new_dag, op_t, &mut stats);
                }
                new_dag.add_op(node.op.clone());
            }
            CircuitOp::RademacherPhase { .. } => {
                // Multi-qubit global barrier: flush all pending single-qubit gates
                for slot in pending_1q.iter_mut().take(dag.num_qubits) {
                    if let Some(op) = slot.take() {
                        flush_single_qubit(&mut new_dag, op, &mut stats);
                    }
                }
                new_dag.add_op(node.op.clone());
            }
        }
    }

    // Flush any remaining trailing single-qubit gates across all wires
    for slot in pending_1q.iter_mut().take(dag.num_qubits) {
        if let Some(op) = slot.take() {
            flush_single_qubit(&mut new_dag, op, &mut stats);
        }
    }

    stats.ops_after = new_dag.len();
    (new_dag, stats)
}

fn flush_single_qubit(dag: &mut CircuitDag, op: CircuitOp, stats: &mut FusionStats) {
    // If the fused unitary is equivalent to identity (e.g. Rz(0.0)), eliminate it
    if let CircuitOp::Unitary1Q { matrix, .. } = &op {
        if matrix.is_identity(1e-12) {
            stats.identities_removed += 1;
            return;
        }
    }
    if let CircuitOp::Rz { theta, .. } = &op {
        if theta.abs() < 1e-14 {
            stats.identities_removed += 1;
            return;
        }
    }
    if let CircuitOp::Ry { theta, .. } = &op {
        if theta.abs() < 1e-14 {
            stats.identities_removed += 1;
            return;
        }
    }
    dag.add_op(op);
}
