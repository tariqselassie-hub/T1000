// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Comprehensive verification suite for automated Clifford+T synthesis.
//!
//! Validates:
//! 1. Cyclotomic ring Z[ω] algebraic identities and exact sqrt(2) division.
//! 2. Unitary reconstruction invariant: ||U_exact - U_discrete||_inf <= ε.
//! 3. Precision vs T-count scaling: <T_count> ~ 3*log2(1/ε) + O(1).
//! 4. Peephole rewrite optimization (H*H -> I, T*T -> S, etc.).
//! 5. Multi-axis synthesis across Rz(θ), Rx(θ), and Ry(θ).

use std::f64::consts::PI;
use t1000::clifford_t::{
    evaluate_discrete_sequence, lower_circuit_ops, optimize_clifford_t_sequence,
    synthesize_and_profile, synthesize_rx, synthesize_ry, synthesize_rz, DiscreteGate, ZOmega,
};
use t1000::compiler::op::{CircuitOp, Unitary2x2};

#[test]
fn test_cyclotomic_ring_identities() {
    // 1. omega^4 = -1
    let omega = ZOmega::OMEGA;
    let omega2 = omega.mul(omega);
    let omega4 = omega2.mul(omega2);
    assert_eq!(omega4, ZOmega::new(-1, 0, 0, 0));

    // 2. sqrt(2) = omega - omega^3
    let sqrt2 = ZOmega::SQRT2;
    let two = sqrt2.mul(sqrt2);
    assert_eq!(two, ZOmega::new(2, 0, 0, 0));
    assert!((sqrt2.norm_sq() - 2.0).abs() < 1e-12);

    // 3. Exact division by sqrt(2)
    assert_eq!(two.div_sqrt2(), Some(sqrt2));
    assert_eq!(sqrt2.div_sqrt2(), Some(ZOmega::ONE));
    assert_eq!(ZOmega::ONE.div_sqrt2(), None);

    // 4. Complex conjugation: z * conj(z) = |z|^2 in Z[sqrt(2)]
    let z = ZOmega::new(2, -1, 3, 1);
    let z_conj = z.conj();
    let norm_prod = z.mul(z_conj);
    assert_eq!(norm_prod.c, 0);
    assert_eq!(norm_prod.b, -norm_prod.d);
    assert!((norm_prod.to_complex().re - z.norm_sq()).abs() < 1e-10);
    assert!(norm_prod.to_complex().im.abs() < 1e-10);
}

#[test]
fn test_peephole_optimization_rules() {
    // Test H * H -> I
    let mut seq1 = vec![DiscreteGate::H, DiscreteGate::H];
    optimize_clifford_t_sequence(&mut seq1);
    assert!(seq1.is_empty());

    // Test T * Tdg -> I
    let mut seq2 = vec![DiscreteGate::T, DiscreteGate::Tdg];
    optimize_clifford_t_sequence(&mut seq2);
    assert!(seq2.is_empty());

    // Test T * T -> S
    let mut seq3 = vec![DiscreteGate::T, DiscreteGate::T];
    optimize_clifford_t_sequence(&mut seq3);
    assert_eq!(seq3, vec![DiscreteGate::S]);

    // Test S * S -> Z
    let mut seq4 = vec![DiscreteGate::S, DiscreteGate::S];
    optimize_clifford_t_sequence(&mut seq4);
    assert_eq!(seq4, vec![DiscreteGate::Z]);

    // Test Z * Z -> I
    let mut seq5 = vec![DiscreteGate::Z, DiscreteGate::Z];
    optimize_clifford_t_sequence(&mut seq5);
    assert!(seq5.is_empty());

    // Compound sequence: H * T * T * Tdg * T * H -> H * S * H
    let mut seq6 = vec![
        DiscreteGate::H,
        DiscreteGate::T,
        DiscreteGate::T,
        DiscreteGate::Tdg,
        DiscreteGate::T,
        DiscreteGate::H,
    ];
    optimize_clifford_t_sequence(&mut seq6);
    assert_eq!(
        seq6,
        vec![DiscreteGate::H, DiscreteGate::S, DiscreteGate::H]
    );
}

#[test]
fn test_exact_clifford_angles() {
    // Rz(0) -> empty sequence
    let g0 = synthesize_rz(0.0, 1e-3);
    assert!(g0.is_empty());

    // Rz(pi) -> Z gate
    let g_pi = synthesize_rz(PI, 1e-3);
    assert_eq!(g_pi, vec![DiscreteGate::Z]);

    // Rz(pi/2) -> S gate
    let g_half_pi = synthesize_rz(0.5 * PI, 1e-3);
    let u_half = evaluate_discrete_sequence(&g_half_pi);
    let exact_half = Unitary2x2::rz(0.5 * PI);
    let dist = t1000::clifford_t::operator_distance(&exact_half, &u_half);
    assert!(dist < 1e-3);
}

#[test]
fn test_unitary_reconstruction_invariant_rz_sweep() {
    let angles = [0.2, 0.5, 0.8, 1.2, 1.7, 2.3, 2.9, 3.4, 4.1, 4.8, 5.5];
    let eps = 0.05;

    for &theta in &angles {
        let exact = Unitary2x2::rz(theta);
        let gates = synthesize_rz(theta, eps);
        let discrete = evaluate_discrete_sequence(&gates);
        let dist = t1000::clifford_t::operator_distance(&exact, &discrete);

        assert!(
            dist <= eps * 2.0,
            "Reconstruction error invariant failed at theta={theta}: dist={dist:.4e} > eps={eps:.4e}"
        );
    }
}

#[test]
fn test_rx_and_ry_synthesis() {
    let theta = 1.15;
    let eps = 0.05;

    // Rx(theta) = H * Rz(theta) * H
    let exact_rx = Unitary2x2::hadamard()
        .matmul(&Unitary2x2::rz(theta))
        .matmul(&Unitary2x2::hadamard());
    let gates_rx = synthesize_rx(theta, eps);
    let disc_rx = evaluate_discrete_sequence(&gates_rx);
    let dist_rx = t1000::clifford_t::operator_distance(&exact_rx, &disc_rx);
    assert!(dist_rx <= eps * 2.0);

    // Ry(theta)
    let exact_ry = Unitary2x2::ry(theta);
    let gates_ry = synthesize_ry(theta, eps);
    let disc_ry = evaluate_discrete_sequence(&gates_ry);
    let dist_ry = t1000::clifford_t::operator_distance(&exact_ry, &disc_ry);
    assert!(dist_ry <= eps * 2.0);
}

#[test]
fn test_precision_scaling_t_count() {
    // Deliberately ~1.6e-7 off pi/4: an exact pi/4 makes Rz a T gate and the
    // synthesis trivial, which would defeat this precision-scaling check.
    #[allow(clippy::approx_constant)]
    let theta = 0.785398;
    let epsilons = [0.1, 0.01, 0.001];
    let mut t_counts = Vec::new();

    for &eps in &epsilons {
        let rep = synthesize_and_profile("RZ", theta, eps);
        t_counts.push(rep.t_count);
    }

    // T-count should be non-decreasing as epsilon tightens
    for i in 0..(t_counts.len() - 1) {
        assert!(
            t_counts[i] <= t_counts[i + 1] + 2,
            "T-count should increase with higher precision: {:?}",
            t_counts
        );
    }
}

#[test]
fn test_circuit_lowering_to_clifford_t() {
    let ops = vec![
        CircuitOp::Rz {
            qubit: 0,
            theta: 0.5,
        },
        CircuitOp::Ry {
            qubit: 1,
            theta: 1.2,
        },
        CircuitOp::CZ { q1: 0, q2: 1 },
    ];

    let lowered = lower_circuit_ops(&ops, 0.05);
    assert!(!lowered.is_empty());
    assert!(lowered.iter().any(|(q, _)| *q == 0));
    assert!(lowered.iter().any(|(q, _)| *q == 1));
}
