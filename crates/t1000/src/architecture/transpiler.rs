// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Quantum Chip Circuit Transpiler and Native Gate Decomposer
//!
//! Maps arbitrary [`CircuitOp`] instruction streams onto concrete [`QuantumChip`] targets,
//! executing basis gate decomposition (e.g., CNOT -> H-CZ-H / Ry-CZ-Ry), routing non-adjacent
//! couplers via shortest-path SWAP networks, and estimating physical duration and fidelity.

use std::f64::consts::PI;
use serde::{Deserialize, Serialize};

use crate::compiler::CircuitOp;
use super::{NativeGateKind, QuantumChip};

/// Summary report generated after transpiling a circuit for a target quantum chip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranspileReport {
    /// Target quantum processor.
    pub target_chip: QuantumChip,
    /// Total gates before transpilation.
    pub input_gate_count: usize,
    /// Total native hardware gates after transpilation.
    pub output_gate_count: usize,
    /// Number of single-qubit native gates.
    pub single_qubit_gate_count: usize,
    /// Number of two-qubit native entangling gates.
    pub two_qubit_gate_count: usize,
    /// Number of routing SWAP gates inserted to satisfy device connectivity.
    pub routing_swaps_inserted: usize,
    /// Estimated physical execution duration on silicon in nanoseconds (ns).
    pub estimated_duration_ns: f64,
    /// Predicted asymptotic gate fidelity before decoherence based on chip error budget.
    pub predicted_gate_fidelity: f64,
}

/// A transpiled circuit comprising physical native gates ready for hardware execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranspiledCircuit {
    /// Target chip architecture.
    pub chip: QuantumChip,
    /// Stream of physical hardware operations.
    pub ops: Vec<CircuitOp>,
    /// Analytical execution report.
    pub report: TranspileReport,
}

/// Hardware-aware circuit transpiler targeting bespoke quantum chip architectures.
#[derive(Debug, Clone, Copy)]
pub struct ChipTranspiler {
    pub chip: QuantumChip,
}

impl ChipTranspiler {
    /// Initialize a transpiler targeting the specified quantum chip.
    pub fn new(chip: QuantumChip) -> Self {
        Self { chip }
    }

    /// Transpile a slice of generic [`CircuitOp`] operations into chip-native instructions.
    pub fn transpile(&self, ops: &[CircuitOp]) -> Result<TranspiledCircuit, String> {
        let max_q = self.chip.capacity();
        for (idx, op) in ops.iter().enumerate() {
            for &q in &op.qubits() {
                if q >= max_q {
                    return Err(format!(
                        "Gate #{} ({:?}) targets qubit {} exceeding chip capacity {} on {}",
                        idx, op, q, max_q, self.chip.slug()
                    ));
                }
            }
        }

        let topo = self.chip.build_routing_topology();
        let mut native_ops = Vec::with_capacity(ops.len() * 2);
        let mut routing_swaps = 0usize;

        for op in ops {
            match op {
                CircuitOp::CNOT { control, target } => {
                    self.decompose_cnot(*control, *target, &topo, &mut native_ops, &mut routing_swaps);
                }
                CircuitOp::CZ { q1, q2 } => {
                    self.route_and_emit_2q(*q1, *q2, &topo, &mut native_ops, &mut routing_swaps, |a, b| {
                        CircuitOp::CZ { q1: a, q2: b }
                    });
                }
                CircuitOp::AntiCNOT { control, target } => {
                    // X on control, CNOT, X on control
                    native_ops.push(CircuitOp::Ry { qubit: *control, theta: PI });
                    self.decompose_cnot(*control, *target, &topo, &mut native_ops, &mut routing_swaps);
                    native_ops.push(CircuitOp::Ry { qubit: *control, theta: -PI });
                }
                CircuitOp::SWAP { q1, q2 } => {
                    // 3 CNOTs
                    self.decompose_cnot(*q1, *q2, &topo, &mut native_ops, &mut routing_swaps);
                    self.decompose_cnot(*q2, *q1, &topo, &mut native_ops, &mut routing_swaps);
                    self.decompose_cnot(*q1, *q2, &topo, &mut native_ops, &mut routing_swaps);
                }
                CircuitOp::ControlledPhase { control, target, theta } => {
                    // CP(theta) = Rz(theta/2) on control and target, then CNOT, Rz(-theta/2) on target, CNOT
                    let half = theta * 0.5;
                    native_ops.push(CircuitOp::Rz { qubit: *control, theta: half });
                    native_ops.push(CircuitOp::Rz { qubit: *target, theta: half });
                    self.decompose_cnot(*control, *target, &topo, &mut native_ops, &mut routing_swaps);
                    native_ops.push(CircuitOp::Rz { qubit: *target, theta: -half });
                    self.decompose_cnot(*control, *target, &topo, &mut native_ops, &mut routing_swaps);
                }
                _ => {
                    // Single qubit gates (Ry, Rz, Unitary1Q, RademacherPhase) pass directly
                    native_ops.push(op.clone());
                }
            }
        }

        let mut n1 = 0usize;
        let mut n2 = 0usize;
        for op in &native_ops {
            if op.is_single_qubit() {
                n1 += 1;
            } else {
                n2 += 1;
            }
        }

        let err_profile = self.chip.error_profile();
        let est_duration = (n1 as f64 * err_profile.single_qubit_duration_ns)
            + (n2 as f64 * err_profile.two_qubit_duration_ns);

        let p1 = err_profile.single_qubit_error;
        let p2 = err_profile.two_qubit_error;
        let pred_fidelity = (1.0 - p1).powi(n1 as i32) * (1.0 - p2).powi(n2 as i32);

        let report = TranspileReport {
            target_chip: self.chip,
            input_gate_count: ops.len(),
            output_gate_count: native_ops.len(),
            single_qubit_gate_count: n1,
            two_qubit_gate_count: n2,
            routing_swaps_inserted: routing_swaps,
            estimated_duration_ns: est_duration,
            predicted_gate_fidelity: pred_fidelity,
        };

        Ok(TranspiledCircuit {
            chip: self.chip,
            ops: native_ops,
            report,
        })
    }

    fn decompose_cnot(
        &self,
        ctrl: usize,
        tgt: usize,
        topo: &crate::routing::Topology,
        out: &mut Vec<CircuitOp>,
        swaps: &mut usize,
    ) {
        let has_cz = self.chip.native_gates().contains(&NativeGateKind::Cz);
        let has_cnot = self.chip.native_gates().contains(&NativeGateKind::Cnot);

        if has_cnot {
            self.route_and_emit_2q(ctrl, tgt, topo, out, swaps, |a, b| CircuitOp::CNOT {
                control: a,
                target: b,
            });
        } else if has_cz {
            // CNOT(c, t) = Ry(pi/2)_t -> CZ(c, t) -> Ry(-pi/2)_t
            out.push(CircuitOp::Ry { qubit: tgt, theta: PI * 0.5 });
            self.route_and_emit_2q(ctrl, tgt, topo, out, swaps, |a, b| CircuitOp::CZ { q1: a, q2: b });
            out.push(CircuitOp::Ry { qubit: tgt, theta: -PI * 0.5 });
        } else {
            // Default CZ fallback
            self.route_and_emit_2q(ctrl, tgt, topo, out, swaps, |a, b| CircuitOp::CZ { q1: a, q2: b });
        }
    }

    fn route_and_emit_2q<F>(
        &self,
        q1: usize,
        q2: usize,
        topo: &crate::routing::Topology,
        out: &mut Vec<CircuitOp>,
        swaps: &mut usize,
        gate_ctor: F,
    ) where
        F: FnOnce(usize, usize) -> CircuitOp,
    {
        if topo.adjacent(q1, q2) || self.chip.modality().supports_all_to_all() {
            out.push(gate_ctor(q1, q2));
            return;
        }

        // Qubits not adjacent: find shortest path on coupling graph
        let path = self.shortest_path(q1, q2, topo);
        if path.len() <= 2 {
            out.push(gate_ctor(q1, q2));
            return;
        }

        // SWAP q1 along path to meet neighbor of q2
        let n_hops = path.len() - 2;
        for i in 0..n_hops {
            let u = path[i];
            let v = path[i + 1];
            out.push(CircuitOp::SWAP { q1: u, q2: v });
            *swaps += 1;
        }

        // Execute gate between routed q1 position and q2
        let routed_q1 = path[n_hops];
        let target_q2 = path[n_hops + 1];
        out.push(gate_ctor(routed_q1, target_q2));

        // Unswap q1 back to restore canonical register mapping
        for i in (0..n_hops).rev() {
            let u = path[i];
            let v = path[i + 1];
            out.push(CircuitOp::SWAP { q1: u, q2: v });
            *swaps += 1;
        }
    }

    fn shortest_path(&self, start: usize, goal: usize, topo: &crate::routing::Topology) -> Vec<usize> {
        let n = topo.n;
        let mut prev = vec![None; n];
        let mut dist = vec![usize::MAX; n];
        let mut queue = std::collections::VecDeque::new();

        dist[start] = 0;
        queue.push_back(start);

        let edges = topo.edges();
        let mut adj = vec![Vec::new(); n];
        for &(u, v) in edges {
            adj[u].push(v);
            adj[v].push(u);
        }

        while let Some(u) = queue.pop_front() {
            if u == goal {
                break;
            }
            for &v in &adj[u] {
                if dist[v] == usize::MAX {
                    dist[v] = dist[u] + 1;
                    prev[v] = Some(u);
                    queue.push_back(v);
                }
            }
        }

        let mut path = Vec::new();
        let mut curr = goal;
        while let Some(p) = prev[curr] {
            path.push(curr);
            curr = p;
        }
        path.push(start);
        path.reverse();
        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpile_cnot_decomposition_on_nighthawk() {
        let transpiler = ChipTranspiler::new(QuantumChip::IbmNighthawkR2);
        // Coupler (0, 1) is adjacent on Nighthawk R2
        let circuit = vec![CircuitOp::CNOT { control: 0, target: 1 }];
        let res = transpiler.transpile(&circuit).expect("transpile ok");

        // IBM Nighthawk native gates: [Rz, SqrtX, X, Ecr, Cz].
        // CNOT should decompose into Ry(pi/2) -> CZ -> Ry(-pi/2)
        assert_eq!(res.report.input_gate_count, 1);
        assert_eq!(res.report.output_gate_count, 3);
        assert_eq!(res.report.two_qubit_gate_count, 1);
        assert_eq!(res.report.routing_swaps_inserted, 0);
        assert!(res.report.predicted_gate_fidelity > 0.99);
    }

    #[test]
    fn test_transpile_routing_distance_insertion() {
        let transpiler = ChipTranspiler::new(QuantumChip::T1000QpuArtix7);
        // Artix-7 is a linear chain: 0 - 1 - 2 - 3 ...
        // Gate (0, 3) requires routing hops
        let circuit = vec![CircuitOp::CZ { q1: 0, q2: 3 }];
        let res = transpiler.transpile(&circuit).expect("transpile ok");

        // Distance from 0 to 3 is 3 hops. Should insert routing SWAPs to bridge and restore.
        assert!(res.report.routing_swaps_inserted > 0);
        assert!(res.report.output_gate_count > 1);
    }

    #[test]
    fn test_exceeding_capacity_rejected() {
        let transpiler = ChipTranspiler::new(QuantumChip::T1000QpuArtix7); // capacity 14
        let circuit = vec![CircuitOp::Ry { qubit: 15, theta: 1.0 }];
        let res = transpiler.transpile(&circuit);
        assert!(res.is_err());
    }
}
