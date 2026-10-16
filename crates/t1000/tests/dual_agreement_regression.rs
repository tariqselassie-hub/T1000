// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Continuous Integration Dual-Agreement Regression Suite.
//!
//! Locks rigorous physical invariants across the scalar reference simulator
//! and the optimized AVX2 SIMD fused engine:
//! 1. Total Variation Distance (TVD) < 1e-14
//! 2. Maximum Amplitude Error (MaxErr) < 1e-16
//! 3. Kolmogorov-Smirnov Porter-Thomas Statistic D < D_critical

use num_complex::Complex64;
use t1000::clifford_t::{evaluate_discrete_sequence, operator_distance, synthesize_rz};
use t1000::compiler::{CircuitOp, Unitary2x2};
use t1000::{CircuitAnsatz, PreInitCircuit, QuantumState};
use t1000_core::porter_thomas::verify_porter_thomas;

// =========================================================================
// Textbook Scalar Reference Simulator
// =========================================================================

fn ref_apply_1q(psi: &mut [Complex64], target: usize, u: [[Complex64; 2]; 2]) {
    let mask = 1usize << target;
    for i in 0..psi.len() {
        if (i & mask) == 0 {
            let j = i | mask;
            let a = psi[i];
            let b = psi[j];
            psi[i] = u[0][0] * a + u[0][1] * b;
            psi[j] = u[1][0] * a + u[1][1] * b;
        }
    }
}

fn ref_ry(theta: f64) -> [[Complex64; 2]; 2] {
    let c = Complex64::new((theta * 0.5).cos(), 0.0);
    let s = Complex64::new((theta * 0.5).sin(), 0.0);
    [[c, -s], [s, c]]
}

fn ref_rz(theta: f64) -> [[Complex64; 2]; 2] {
    let p_neg = Complex64::from_polar(1.0, -theta * 0.5);
    let p_pos = Complex64::from_polar(1.0, theta * 0.5);
    [
        [p_neg, Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), p_pos],
    ]
}

fn ref_cz(psi: &mut [Complex64], q1: usize, q2: usize) {
    let both = (1usize << q1) | (1usize << q2);
    for i in 0..psi.len() {
        if (i & both) == both {
            psi[i] = -psi[i];
        }
    }
}

fn run_scalar_reference(ops: &[CircuitOp], init: &[Complex64]) -> Vec<Complex64> {
    let mut psi = init.to_vec();
    for op in ops {
        match op {
            CircuitOp::Ry { qubit, theta } => ref_apply_1q(&mut psi, *qubit, ref_ry(*theta)),
            CircuitOp::Rz { qubit, theta } => ref_apply_1q(&mut psi, *qubit, ref_rz(*theta)),
            CircuitOp::CZ { q1, q2 } => ref_cz(&mut psi, *q1, *q2),
            CircuitOp::RademacherPhase { signs } => {
                for (q, s) in signs.iter().enumerate() {
                    ref_apply_1q(&mut psi, q, ref_rz(s * std::f64::consts::PI));
                }
            }
            _ => {}
        }
    }
    psi
}

// =========================================================================
// Dual-Agreement Invariant Verification
// =========================================================================

#[test]
fn test_dual_agreement_scalar_vs_avx2_fused() {
    let qubits = 14;
    let depth = 12;
    let dim = 1 << qubits;

    let ansatz = CircuitAnsatz::new(qubits, depth).with_pre_init(PreInitCircuit::RademacherPi);
    let ops: Vec<CircuitOp> = ansatz
        .to_dag(&ansatz.weights)
        .nodes
        .into_iter()
        .map(|n| n.op)
        .collect();

    // Prepare normalized initial uniform state |+>^n
    let amp = 1.0 / (dim as f64).sqrt();
    let init_full = vec![Complex64::new(amp, 0.0); dim];

    // 1. Run textbook scalar reference
    let ref_psi = run_scalar_reference(&ops, &init_full);

    // 2. Run optimized AVX2 SIMD fused engine
    let mut opt_state = QuantumState::new(dim);
    ansatz.apply_pre_init(&mut opt_state);
    for layer in 0..ansatz.depth {
        ansatz.forward_layer_into(layer, &ansatz.weights, &mut opt_state);
    }

    // 3. Compute error metrics
    let mut max_err = 0.0f64;
    let mut tvd = 0.0f64;
    let mut overlap = Complex64::new(0.0, 0.0);

    for i in 0..dim {
        let o = Complex64::new(opt_state.real[i], opt_state.imag[i]);
        let r = ref_psi[i];
        let diff = (o - r).norm();
        if diff > max_err {
            max_err = diff;
        }
        overlap += r.conj() * o;
        tvd += (o.norm_sqr() - r.norm_sqr()).abs();
    }
    tvd *= 0.5;
    let fidelity = overlap.norm_sqr();

    // 4. Compute Kolmogorov-Smirnov Porter-Thomas statistic
    let probs = opt_state.probabilities();
    let pt = verify_porter_thomas(&probs);

    // Invariant 1: Total Variation Distance < 1e-14
    assert!(
        tvd < 1e-14,
        "Total Variation Distance exceeded tolerance: {} >= 1e-14",
        tvd
    );

    // Invariant 2: Maximum Amplitude Error < 1e-16
    assert!(
        max_err < 1e-16,
        "Maximum Amplitude Error exceeded tolerance: {} >= 1e-16",
        max_err
    );

    // Invariant 3: Overlap Fidelity identically 1.0 within machine precision
    assert!(
        (fidelity - 1.0).abs() < 1e-12,
        "State fidelity drift detected: {} != 1.0",
        fidelity
    );

    // Invariant 4: Porter-Thomas Kolmogorov-Smirnov test D < D_critical
    assert!(
        pt.ks_statistic < pt.ks_critical_value,
        "Porter-Thomas chaos violated: D ({}) >= D_critical ({})",
        pt.ks_statistic,
        pt.ks_critical_value
    );
    assert!(pt.passes_ks_test, "Kolmogorov-Smirnov test did not pass");
}

#[test]
fn test_clifford_t_synthesis_reconstruction_invariant() {
    let angles = [
        0.0,
        0.5,
        std::f64::consts::FRAC_PI_4,
        std::f64::consts::FRAC_PI_2,
        std::f64::consts::PI,
        2.5,
    ];
    let eps = 0.05;

    for &theta in &angles {
        let exact = Unitary2x2::rz(theta);
        let gates = synthesize_rz(theta, eps);
        let discrete = evaluate_discrete_sequence(&gates);
        let dist = operator_distance(&exact, &discrete);

        assert!(
            dist <= eps * 2.0,
            "Clifford+T decomposition operator distance drifted: {} > eps ({})",
            dist,
            eps
        );
    }
}
