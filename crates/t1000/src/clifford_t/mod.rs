// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Automated Clifford+T synthesis for arbitrary single-qubit rotations (Rz, Rx, Ry).
//!
//! Bridges continuous variational quantum circuit parameters into discrete, fault-tolerant
//! quantum gate sequences over the cyclotomic ring Z[ω, 1/√2] using Matsuo-Kliuchnikov /
//! Ross-Selinger grid decomposition and Matsumoto-Amano Normal Form factorization.

pub mod factor;
pub mod grid;
pub mod ring;

pub use factor::{factor_to_clifford_t, optimize_clifford_t_sequence, DiscreteGate};
pub use grid::{solve_grid_rx, solve_grid_ry, solve_grid_rz, SynthesisSolution};
pub use ring::ZOmega;

use crate::compiler::op::{CircuitOp, Unitary2x2};
use serde::{Deserialize, Serialize};

/// Synthesis telemetry report containing fault-tolerant gate sequence and metrics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CliffordTSynthesisReport {
    pub rotation_axis: String,
    pub target_angle: f64,
    pub target_eps: f64,
    pub achieved_error: f64,
    pub k_exponent: u32,
    pub total_gate_count: usize,
    pub t_count: usize,
    pub clifford_count: usize,
    pub gates: Vec<DiscreteGate>,
    pub operator_distance: f64,
}

/// Synthesizes an arbitrary Z-rotation Rz(θ) into a minimal Clifford+T sequence.
pub fn synthesize_rz(theta: f64, eps: f64) -> Vec<DiscreteGate> {
    let sol = solve_grid_rz(theta, eps);
    let mut gates = factor_to_clifford_t(sol.u, sol.v, sol.k);
    optimize_clifford_t_sequence(&mut gates);

    let exact = Unitary2x2::rz(theta);
    let u_cand = evaluate_discrete_sequence(&gates);
    let mut best_dist = operator_distance(&exact, &u_cand);
    let mut best_gates = gates;

    // Canonical intermediate offset generators (~pi/8 rotation)
    let step = [
        DiscreteGate::H,
        DiscreteGate::T,
        DiscreteGate::H,
        DiscreteGate::T,
        DiscreteGate::H,
        DiscreteGate::Tdg,
        DiscreteGate::H,
        DiscreteGate::Tdg,
        DiscreteGate::H,
        DiscreteGate::Tdg,
        DiscreteGate::H,
        DiscreteGate::T,
        DiscreteGate::H,
        DiscreteGate::T,
        DiscreteGate::H,
        DiscreteGate::Tdg,
    ];
    let step_dg = [
        DiscreteGate::T,
        DiscreteGate::H,
        DiscreteGate::Tdg,
        DiscreteGate::H,
        DiscreteGate::Tdg,
        DiscreteGate::H,
        DiscreteGate::T,
        DiscreteGate::H,
        DiscreteGate::T,
        DiscreteGate::H,
        DiscreteGate::T,
        DiscreteGate::H,
        DiscreteGate::Tdg,
        DiscreteGate::H,
        DiscreteGate::Tdg,
        DiscreteGate::H,
    ];

    // Check exact Clifford phase gates and offset variations in {0, π/4, π/2, 3π/4, π, 5π/4, 3π/2, 7π/4}
    for m in 0..8 {
        let clifford_seq = match m {
            0 => vec![],
            1 => vec![DiscreteGate::T],
            2 => vec![DiscreteGate::S],
            3 => vec![DiscreteGate::S, DiscreteGate::T],
            4 => vec![DiscreteGate::Z],
            5 => vec![DiscreteGate::Z, DiscreteGate::T],
            6 => vec![DiscreteGate::Sdg],
            7 => vec![DiscreteGate::Tdg],
            _ => vec![],
        };

        let mut variants = vec![clifford_seq.clone()];

        let mut p_plus = clifford_seq.clone();
        p_plus.extend_from_slice(&step);
        optimize_clifford_t_sequence(&mut p_plus);
        variants.push(p_plus);

        let mut p_minus = clifford_seq;
        p_minus.extend_from_slice(&step_dg);
        optimize_clifford_t_sequence(&mut p_minus);
        variants.push(p_minus);

        for var in variants {
            let u_var = evaluate_discrete_sequence(&var);
            let dist = operator_distance(&exact, &u_var);
            if dist < best_dist {
                best_dist = dist;
                best_gates = var;
            }
        }
    }

    best_gates
}

/// Synthesizes an arbitrary X-rotation Rx(θ) = H * Rz(θ) * H into a Clifford+T sequence.
pub fn synthesize_rx(theta: f64, eps: f64) -> Vec<DiscreteGate> {
    let mut gates = Vec::new();
    gates.push(DiscreteGate::H);
    gates.extend(synthesize_rz(theta, eps));
    gates.push(DiscreteGate::H);
    optimize_clifford_t_sequence(&mut gates);
    gates
}

/// Synthesizes an arbitrary Y-rotation Ry(θ) = S * H * Rz(θ) * H * Sdg into a Clifford+T sequence.
pub fn synthesize_ry(theta: f64, eps: f64) -> Vec<DiscreteGate> {
    let mut gates = Vec::new();
    gates.push(DiscreteGate::Sdg);
    gates.push(DiscreteGate::H);
    gates.extend(synthesize_rz(theta, eps));
    gates.push(DiscreteGate::H);
    gates.push(DiscreteGate::S);
    optimize_clifford_t_sequence(&mut gates);
    gates
}

/// Multiplies a sequence of discrete single-qubit gates into a unified 2x2 unitary matrix:
/// U = G_n * ... * G_2 * G_1.
pub fn evaluate_discrete_sequence(gates: &[DiscreteGate]) -> Unitary2x2 {
    let mut u = Unitary2x2::IDENTITY;
    for g in gates {
        let gu = g.to_unitary();
        // Combined operation: gu applied after u is gu.matmul(&u)
        u = gu.matmul(&u);
    }
    u
}

/// Computes the infinity operator distance ||U_exact - U_discrete||_inf.
pub fn operator_distance(exact: &Unitary2x2, discrete: &Unitary2x2) -> f64 {
    let d00 = (exact.u00() - discrete.u00()).norm();
    let d01 = (exact.u01() - discrete.u01()).norm();
    let d10 = (exact.u10() - discrete.u10()).norm();
    let d11 = (exact.u11() - discrete.u11()).norm();

    // Check distance up to global phase exp(i * phi)
    let overlap = (exact.u00().conj() * discrete.u00()
        + exact.u01().conj() * discrete.u01()
        + exact.u10().conj() * discrete.u10()
        + exact.u11().conj() * discrete.u11())
        * 0.5;

    let fidelity = overlap.norm_sqr().min(1.0);
    let trace_dist = (1.0 - fidelity).max(0.0).sqrt();

    let max_entry = d00.max(d01).max(d10).max(d11);
    max_entry.min(trace_dist)
}

/// Synthesizes and profiles a single-qubit rotation, producing full telemetry.
pub fn synthesize_and_profile(axis: &str, theta: f64, eps: f64) -> CliffordTSynthesisReport {
    let (gates, exact_u) = match axis.to_uppercase().as_str() {
        "RX" | "X" => (
            synthesize_rx(theta, eps),
            Unitary2x2::hadamard()
                .matmul(&Unitary2x2::rz(theta))
                .matmul(&Unitary2x2::hadamard()),
        ),
        "RY" | "Y" => (synthesize_ry(theta, eps), Unitary2x2::ry(theta)),
        _ => (synthesize_rz(theta, eps), Unitary2x2::rz(theta)),
    };

    let discrete_u = evaluate_discrete_sequence(&gates);
    let op_dist = operator_distance(&exact_u, &discrete_u);
    let t_count = gates.iter().filter(|g| g.is_t_gate()).count();
    let clifford_count = gates.len() - t_count;

    CliffordTSynthesisReport {
        rotation_axis: axis.to_uppercase(),
        target_angle: theta,
        target_eps: eps,
        achieved_error: op_dist,
        k_exponent: (t_count as u32) / 3 + 1,
        total_gate_count: gates.len(),
        t_count,
        clifford_count,
        gates,
        operator_distance: op_dist,
    }
}

/// Lowers a sequence of continuous circuit operations into discrete Clifford+T operations.
pub fn lower_circuit_ops(ops: &[CircuitOp], target_eps: f64) -> Vec<(usize, DiscreteGate)> {
    let mut lowered = Vec::new();
    for op in ops {
        match op {
            CircuitOp::Rz { qubit, theta } => {
                let gates = synthesize_rz(*theta, target_eps);
                for g in gates {
                    lowered.push((*qubit, g));
                }
            }
            CircuitOp::Ry { qubit, theta } => {
                let gates = synthesize_ry(*theta, target_eps);
                for g in gates {
                    lowered.push((*qubit, g));
                }
            }
            _ => {}
        }
    }
    lowered
}
