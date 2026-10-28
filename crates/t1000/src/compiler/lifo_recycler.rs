// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Dynamic LIFO Wire Recycling & MBU Integration
//!
//! Provides automated wire recycling and phase-reversal Measurement-Based Uncomputation (MBU)
//! for intermediate ancilla registers.
//!
//! ## Architectural Principles
//! 1. **Phase-Reversal Uncomputation (MBU):** Rather than performing destructive projective
//!    resets or discarding entangled auxiliary qubits, intermediate ancillas are routed backward
//!    through exact adjoint phase-reversal paths ($U^\dagger$). This deterministically returns the
//!    ancilla to $|0\rangle$ without probabilistic branching.
//! 2. **LIFO Wire Recycling (Hot Cache Locality):** Freed ancilla indices are maintained in a
//!    Last-In-First-Out (LIFO) stack. The most recently disentangled wire is reused immediately
//!    for the next operation, keeping stride access in hot CPU L1/L2 cache lines and collapsing
//!    the peak simulated statevector footprint from $N + K$ down to $N + \max(K_{\mathrm{concurrent}})$.

use std::collections::HashMap;
use crate::compiler::op::CircuitOp;
use serde::{Deserialize, Serialize};

/// Summary metrics of wire recycling efficiency and memory footprint reduction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecyclingStats {
    /// Number of permanent primary computational wires
    pub primary_qubits: usize,
    /// Total number of ancilla acquisition requests
    pub total_allocations: usize,
    /// Number of allocations fulfilled by reusing warm LIFO wires
    pub recycled_allocations: usize,
    /// Peak physical wire index required during circuit execution
    pub peak_wires: usize,
    /// Number of physical qubits saved compared to naive monotonic wire expansion
    pub wires_saved: usize,
    /// Number of adjoint uncomputation operations generated
    pub uncompute_ops_inserted: usize,
}

/// Dynamic LIFO Wire Recycler and MBU Synthesizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifoWireRecycler {
    primary_qubits: usize,
    lifo_stack: Vec<usize>,
    active_ancillas: Vec<usize>,
    ancilla_history: HashMap<usize, Vec<CircuitOp>>,
    total_allocations: usize,
    recycled_allocations: usize,
    peak_wires: usize,
    uncompute_ops_count: usize,
}

impl LifoWireRecycler {
    /// Initializes a recycler for a circuit with `primary_qubits` dedicated computational wires.
    pub fn new(primary_qubits: usize) -> Self {
        Self {
            primary_qubits,
            lifo_stack: Vec::new(),
            active_ancillas: Vec::new(),
            ancilla_history: HashMap::new(),
            total_allocations: 0,
            recycled_allocations: 0,
            peak_wires: primary_qubits,
            uncompute_ops_count: 0,
        }
    }

    /// Acquires an ancilla wire.
    ///
    /// If an uncomputed wire is present in the LIFO stack, pops it to maximize cache reuse.
    /// Otherwise, allocates the next available wire index and expands peak register width.
    pub fn acquire_ancilla(&mut self) -> usize {
        self.total_allocations += 1;
        let wire = if let Some(reused) = self.lifo_stack.pop() {
            self.recycled_allocations += 1;
            reused
        } else {
            let next_wire = self.peak_wires;
            self.peak_wires += 1;
            next_wire
        };

        self.active_ancillas.push(wire);
        self.ancilla_history.insert(wire, Vec::new());
        wire
    }

    /// Records an operation that interacts with an active ancilla wire.
    pub fn record_ancilla_gate(&mut self, ancilla: usize, op: CircuitOp) {
        if let Some(hist) = self.ancilla_history.get_mut(&ancilla) {
            hist.push(op);
        }
    }

    /// Synthesizes the adjoint phase-reversal sequence for an ancilla, disentangles it,
    /// pushes the freed wire index onto the LIFO cache stack, and returns the uncompute ops.
    pub fn release_and_uncompute(&mut self, ancilla: usize) -> Vec<CircuitOp> {
        self.active_ancillas.retain(|&w| w != ancilla);
        let forward_ops = self.ancilla_history.remove(&ancilla).unwrap_or_default();

        let mut uncompute_ops = Vec::with_capacity(forward_ops.len());
        for op in forward_ops.into_iter().rev() {
            let adj = Self::adjoint_of(&op);
            uncompute_ops.push(adj);
        }

        self.uncompute_ops_count += uncompute_ops.len();
        self.lifo_stack.push(ancilla);
        uncompute_ops
    }

    /// Mathematical adjoint $U^\dagger$ of a circuit operation.
    pub fn adjoint_of(op: &CircuitOp) -> CircuitOp {
        match op {
            CircuitOp::Ry { qubit, theta } => CircuitOp::Ry {
                qubit: *qubit,
                theta: -theta,
            },
            CircuitOp::Rz { qubit, theta } => CircuitOp::Rz {
                qubit: *qubit,
                theta: -theta,
            },
            CircuitOp::Unitary1Q { qubit, matrix } => {
                let adj_matrix = matrix.dagger();
                CircuitOp::Unitary1Q {
                    qubit: *qubit,
                    matrix: adj_matrix,
                }
            }
            CircuitOp::CZ { q1, q2 } => CircuitOp::CZ { q1: *q1, q2: *q2 },
            CircuitOp::CNOT { control, target } => CircuitOp::CNOT {
                control: *control,
                target: *target,
            },
            CircuitOp::AntiCNOT { control, target } => CircuitOp::AntiCNOT {
                control: *control,
                target: *target,
            },
            CircuitOp::SWAP { q1, q2 } => CircuitOp::SWAP { q1: *q1, q2: *q2 },
            CircuitOp::ControlledPhase { control, target, theta } => CircuitOp::ControlledPhase {
                control: *control,
                target: *target,
                theta: -theta,
            },
            CircuitOp::RademacherPhase { signs } => CircuitOp::RademacherPhase {
                signs: signs.clone(),
            },
        }
    }

    /// Compiles summary metrics of wire reuse and memory compression.
    pub fn stats(&self) -> RecyclingStats {
        let naive_peak = self.primary_qubits + self.total_allocations;
        let wires_saved = naive_peak.saturating_sub(self.peak_wires);
        RecyclingStats {
            primary_qubits: self.primary_qubits,
            total_allocations: self.total_allocations,
            recycled_allocations: self.recycled_allocations,
            peak_wires: self.peak_wires,
            wires_saved,
            uncompute_ops_inserted: self.uncompute_ops_count,
        }
    }
}

// Add dagger helper to Unitary2x2
impl crate::compiler::Unitary2x2 {
    pub fn dagger(&self) -> Self {
        Self {
            u00_re: self.u00_re,
            u00_im: -self.u00_im,
            u01_re: self.u10_re,
            u01_im: -self.u10_im,
            u10_re: self.u01_re,
            u10_im: -self.u01_im,
            u11_re: self.u11_re,
            u11_im: -self.u11_im,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::QuantumState;

    #[test]
    fn test_lifo_recycler_wire_reuse() {
        let mut recycler = LifoWireRecycler::new(3); // Qubits 0, 1, 2 are primary
        assert_eq!(recycler.stats().peak_wires, 3);

        // Ancilla Task 1
        let a1 = recycler.acquire_ancilla();
        assert_eq!(a1, 3);
        assert_eq!(recycler.stats().peak_wires, 4);

        // Record entangling gate on a1
        recycler.record_ancilla_gate(a1, CircuitOp::CNOT { control: 0, target: a1 });
        let uncompute1 = recycler.release_and_uncompute(a1);
        assert_eq!(uncompute1.len(), 1);

        // Ancilla Task 2: should immediately reuse wire 3 from LIFO stack!
        let a2 = recycler.acquire_ancilla();
        assert_eq!(a2, 3);
        assert_eq!(recycler.stats().recycled_allocations, 1);
        assert_eq!(recycler.stats().peak_wires, 4); // peak wires didn't increase!

        recycler.record_ancilla_gate(a2, CircuitOp::CZ { q1: 1, q2: a2 });
        let uncompute2 = recycler.release_and_uncompute(a2);
        assert_eq!(uncompute2.len(), 1);

        let stats = recycler.stats();
        assert_eq!(stats.total_allocations, 2);
        assert_eq!(stats.recycled_allocations, 1);
        assert_eq!(stats.wires_saved, 1);
    }

    #[test]
    fn test_mbu_uncomputation_state_fidelity() {
        // Test that forward entangling ops + MBU uncomputation returns the ancilla to |0>
        let mut recycler = LifoWireRecycler::new(1); // Primary wire 0, Ancilla wire 1
        let ancilla = recycler.acquire_ancilla();
        assert_eq!(ancilla, 1);

        let mut builder = crate::compiler::CircuitBuilder::new(2);

        // 1. Prepare primary qubit in |+>
        let h_q0 = CircuitOp::Unitary1Q {
            qubit: 0,
            matrix: crate::compiler::Unitary2x2::hadamard(),
        };
        builder.dag.add_op(h_q0);

        // 2. Entangle ancilla with primary qubit
        let cnot = CircuitOp::CNOT { control: 0, target: ancilla };
        builder.dag.add_op(cnot.clone());
        recycler.record_ancilla_gate(ancilla, cnot);

        // 3. Apply phase rotation on ancilla
        let rz = CircuitOp::Rz { qubit: ancilla, theta: 0.785398 };
        builder.dag.add_op(rz.clone());
        recycler.record_ancilla_gate(ancilla, rz);

        // 4. Generate MBU uncomputation sequence and add to builder
        let uncompute_ops = recycler.release_and_uncompute(ancilla);
        for op in uncompute_ops {
            builder.dag.add_op(op);
        }

        // Execute on a 2-qubit statevector
        let mut state = QuantumState::ground(4);
        builder.execute(&mut state);

        // After MBU uncomputation, ancilla (qubit 1) must be strictly in state |0>.
        // That means amplitudes for |10> (idx 2) and |11> (idx 3) must be 0.
        let p2 = state.real[2] * state.real[2] + state.imag[2] * state.imag[2];
        let p3 = state.real[3] * state.real[3] + state.imag[3] * state.imag[3];
        let prob_ancilla_is_one = p2 + p3;
        assert!(
            prob_ancilla_is_one < 1e-12,
            "Ancilla was not cleanly uncomputed to |0>: prob = {}",
            prob_ancilla_is_one
        );
    }
}
