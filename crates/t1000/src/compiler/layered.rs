// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Commutation-aware layered execution of a compiled circuit.
//!
//! Ops are assigned to layers so that two ops on a shared qubit only share a layer when both
//! are diagonal on it (Rz and CZ commute with each other). Within a layer every non-diagonal
//! single-qubit gate runs in one cache-blocked pass (`layer_fusion`), and every Rz and CZ runs
//! in one fused diagonal pass (`diagonal`). Any other op acts as a barrier on its qubits and is
//! applied on its own. The layer order is a linear extension of the circuit's non-commutation
//! order, so the result equals sequential execution up to rounding.

use super::op::CircuitOp;
use crate::layer_fusion::{apply_1q_layer, Gate1Q, LayerGate};
use crate::state::QuantumState;

/// How an op acts on each qubit it touches.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Action {
    /// Non-diagonal single-qubit gate, merged into the layer's gate pass.
    Gate,
    /// Rz or CZ, merged into the layer's diagonal pass.
    Diagonal,
    /// Anything else: applied alone, treated as non-diagonal on its qubits.
    Barrier,
}

fn action(op: &CircuitOp) -> Action {
    match op {
        CircuitOp::Ry { .. } | CircuitOp::Unitary1Q { .. } => Action::Gate,
        CircuitOp::Rz { .. } | CircuitOp::CZ { .. } => Action::Diagonal,
        _ => Action::Barrier,
    }
}

/// Layer index for each op, in input order.
fn assign_layers(ops: &[&CircuitOp], num_qubits: usize) -> Vec<usize> {
    // Next free layer on each qubit for diagonal ops / for non-diagonal ops.
    let mut after_nd = vec![0usize; num_qubits];
    let mut after_any = vec![0usize; num_qubits];
    ops.iter()
        .map(|op| {
            let diag = action(op) == Action::Diagonal;
            let qubits = op.qubits();
            let layer = qubits
                .iter()
                .map(|&q| if diag { after_nd[q] } else { after_any[q] })
                .max()
                .unwrap_or(0);
            for &q in &qubits {
                after_any[q] = after_any[q].max(layer + 1);
                if !diag {
                    after_nd[q] = layer + 1;
                }
            }
            layer
        })
        .collect()
}

/// Executes `ops` (a valid sequential order) on `state` using fused layer passes.
pub(super) fn execute_layered(ops: &[&CircuitOp], num_qubits: usize, state: &mut QuantumState) {
    let layers = assign_layers(ops, num_qubits);
    let mut order: Vec<usize> = (0..ops.len()).collect();
    order.sort_by_key(|&i| layers[i]);

    let mut gates: Vec<LayerGate> = Vec::new();
    let mut theta_rz = vec![0.0f64; num_qubits];
    let mut cz: Vec<(usize, usize)> = Vec::new();
    let mut start = 0;
    while start < order.len() {
        let layer = layers[order[start]];
        let end = start
            + order[start..]
                .iter()
                .take_while(|&&i| layers[i] == layer)
                .count();
        let mut has_rz = false;
        for &i in &order[start..end] {
            match ops[i] {
                CircuitOp::Ry { qubit, theta } => gates.push((*qubit, Gate1Q::Ry(*theta))),
                CircuitOp::Unitary1Q { qubit, matrix } => gates.push((*qubit, Gate1Q::U(*matrix))),
                CircuitOp::Rz { qubit, theta } => {
                    theta_rz[*qubit] += theta;
                    has_rz = true;
                }
                CircuitOp::CZ { q1, q2 } => cz.push((*q1, *q2)),
                barrier => super::CompiledCircuit::execute_op(state, barrier),
            }
        }
        flush_gates(state, &mut gates);
        flush_diagonal(state, &mut theta_rz, &mut cz, has_rz);
        start = end;
    }
}

fn flush_gates(state: &mut QuantumState, gates: &mut Vec<LayerGate>) {
    match gates.as_mut_slice() {
        [] => {}
        [(qubit, Gate1Q::Ry(theta))] => {
            crate::simd::apply_ry_kernel(&mut state.real, &mut state.imag, *qubit, *theta);
        }
        layer => apply_1q_layer(&mut state.real, &mut state.imag, layer),
    }
    gates.clear();
}

fn flush_diagonal(
    state: &mut QuantumState,
    theta_rz: &mut [f64],
    cz: &mut Vec<(usize, usize)>,
    has_rz: bool,
) {
    if has_rz || cz.len() > 1 {
        crate::diagonal::apply_rz_cz_layer(&mut state.real, &mut state.imag, theta_rz, cz);
    } else if let Some(&(q1, q2)) = cz.first() {
        crate::simd::apply_cz_kernel(&mut state.real, &mut state.imag, q1, q2);
    }
    theta_rz.fill(0.0);
    cz.clear();
}
