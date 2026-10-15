//! Directed Acyclic Graph (DAG) Dependency Tracker for Quantum Circuits.
//! Models read/write qubit hazards, causal dependency edges,
//! and topological critical path analysis.

use std::collections::{HashMap, HashSet, VecDeque};
use serde::{Deserialize, Serialize};
use super::op::CircuitOp;

/// A node within the circuit dependency DAG.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DagNode {
    /// Unique identifier within the DAG
    pub id: usize,
    /// The quantum operation at this node
    pub op: CircuitOp,
    /// Nodes that MUST execute before this node (incoming causal edges)
    pub predecessors: Vec<usize>,
    /// Nodes that MUST execute after this node (outgoing causal edges)
    pub successors: Vec<usize>,
    /// Topological distance from the start of the circuit (critical path depth)
    pub depth: usize,
}

impl DagNode {
    pub fn new(id: usize, op: CircuitOp) -> Self {
        Self {
            id,
            op,
            predecessors: Vec::new(),
            successors: Vec::new(),
            depth: 0,
        }
    }
}

/// Directed Acyclic Graph tracking qubit dependencies and execution causal order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitDag {
    /// Total number of qubits in the circuit
    pub num_qubits: usize,
    /// All operation nodes in topological submission order
    pub nodes: Vec<DagNode>,
    /// Mapping of qubit index to the most recent node ID that touched it
    last_writer: HashMap<usize, usize>,
    /// Ordered list of node IDs associated with each qubit
    qubit_wires: HashMap<usize, Vec<usize>>,
}

impl CircuitDag {
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            nodes: Vec::new(),
            last_writer: HashMap::with_capacity(num_qubits),
            qubit_wires: HashMap::with_capacity(num_qubits),
        }
    }

    /// Appends a new operation to the DAG, automatically resolving data dependencies
    /// and inserting directed edges between causal operations on common qubits.
    pub fn add_op(&mut self, op: CircuitOp) -> usize {
        let node_id = self.nodes.len();
        let touched_qubits = op.qubits();

        for &q in &touched_qubits {
            assert!(
                q < self.num_qubits,
                "Op touches qubit {} exceeding circuit width {}",
                q,
                self.num_qubits
            );
        }
        if let [a, b] = touched_qubits.as_slice() {
            assert_ne!(a, b, "Two-qubit op must act on distinct qubits (got {} twice)", a);
        }

        let mut node = DagNode::new(node_id, op.clone());
        let mut pred_set: HashSet<usize> = HashSet::new();

        // 1. Dependency Analysis: Inspect previous operations touching the same qubits
        for &q in &touched_qubits {
            if let Some(&prev_id) = self.last_writer.get(&q) {
                // If the previous op does not commute with this op, or to maintain wire order
                let prev_op = &self.nodes[prev_id].op;
                let commutes = prev_op.commutes_with(&op);

                if !commutes {
                    // Non-commuting hazard: Strict dependency edge
                    pred_set.insert(prev_id);
                } else {
                    // Commuting ops on same qubit (e.g. diagonal gates):
                    // Link to the last non-commuting predecessor of the previous op to permit reordering,
                    // but maintain order if it's the immediate predecessor on the wire.
                    pred_set.insert(prev_id);
                }
            }
        }

        // 2. Direct predecessors from active qubit writers
        let mut max_pred_depth = 0;
        for &pred_id in &pred_set {
            node.predecessors.push(pred_id);
            self.nodes[pred_id].successors.push(node_id);
            let p_depth = self.nodes[pred_id].depth;
            if p_depth + 1 > max_pred_depth {
                max_pred_depth = p_depth + 1;
            }
        }
        node.depth = max_pred_depth;

        // 4. Update qubit hazard tracking
        for &q in &touched_qubits {
            self.last_writer.insert(q, node_id);
            self.qubit_wires.entry(q).or_default().push(node_id);
        }

        self.nodes.push(node);
        node_id
    }


    /// Checks if node `ancestor` is an ancestor of node `descendant`.
    pub fn is_ancestor_of(&self, ancestor: usize, descendant: usize) -> bool {
        if ancestor == descendant {
            return false;
        }
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(descendant);
        visited.insert(descendant);

        while let Some(curr) = queue.pop_front() {
            for &p in &self.nodes[curr].predecessors {
                if p == ancestor {
                    return true;
                }
                if visited.insert(p) {
                    queue.push_back(p);
                }
            }
        }
        false
    }

    /// Returns true if two operations are completely independent (neither is an ancestor of the other).
    pub fn are_independent(&self, node_a: usize, node_b: usize) -> bool {
        if node_a == node_b {
            return false;
        }
        !self.is_ancestor_of(node_a, node_b) && !self.is_ancestor_of(node_b, node_a)
    }

    /// Total number of operations in the DAG.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true if the DAG has no operations.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Critical path depth of the entire circuit.
    pub fn critical_path_depth(&self) -> usize {
        self.nodes.iter().map(|n| n.depth).max().map_or(0, |d| d + 1)
    }

    /// Returns the array of current in-degrees for all nodes in the DAG.
    pub fn in_degrees(&self) -> Vec<usize> {
        self.nodes.iter().map(|n| n.predecessors.len()).collect()
    }

    /// Returns the timeline of operation IDs for a given qubit.
    pub fn wire(&self, qubit: usize) -> &[usize] {
        self.qubit_wires.get(&qubit).map_or(&[], |w| w.as_slice())
    }
}
