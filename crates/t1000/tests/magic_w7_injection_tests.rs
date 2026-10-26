// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

use t1000::qec::magic_w7_injection::{MagicW7InjectionEngine, MAGIC_ALPHA, MAGIC_BETA};
use t1000::state::QuantumState;

#[test]
fn test_magic_state_coefficients_norm() {
    let norm_sq =
        MAGIC_ALPHA.0.powi(2) + MAGIC_ALPHA.1.powi(2) + MAGIC_BETA.0.powi(2) + MAGIC_BETA.1.powi(2);
    assert!((norm_sq - 1.0).abs() < 1e-12);
}

#[test]
fn test_bravyi_kitaev_cubic_suppression_scaling() {
    // Check cubic suppression at epsilon = 0.05: 35 * 0.05^3 = 0.004375
    let err_in_5pct = 0.05;
    let err_out_5pct = MagicW7InjectionEngine::calculate_distilled_error(err_in_5pct);
    let expected_5pct = 35.0 * 0.05f64.powi(3);
    assert!((err_out_5pct - expected_5pct).abs() < 1e-12);
    assert!(err_in_5pct / err_out_5pct > 11.0);

    // Check cubic suppression at epsilon = 0.01: 35 * 0.01^3 = 3.5e-5
    let err_in_1pct = 0.01;
    let err_out_1pct = MagicW7InjectionEngine::calculate_distilled_error(err_in_1pct);
    let expected_1pct = 35.0 * 0.01f64.powi(3);
    assert!((err_out_1pct - expected_1pct).abs() < 1e-12);
    assert!(err_in_1pct / err_out_1pct > 280.0);
}

#[test]
fn test_magic_w7_teleported_logical_t_and_code_preservation() {
    let engine = MagicW7InjectionEngine::new();
    // 18 qubits = 262,144 amplitudes
    let mut state = QuantumState::ground(1 << 18);

    let report = engine.distill_and_inject(&mut state, 0.02, 0x5EED_0042);

    assert_eq!(report.total_qubits, 18);
    assert_eq!(report.active_ancillas, 4);
    assert!(report.output_error_rate < report.input_error_rate);
    assert!(report.cubic_suppression_factor > 1.0);
    assert!(report.post_injection_code_fidelity >= 0.99);
}
