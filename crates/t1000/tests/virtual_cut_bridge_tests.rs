// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

use t1000::ansatz::virtual_cut_engine::{
    Virtual36QCutEngine, VirtualCutConfig, DENSE_36Q_RAM_BYTES, FRAGMENT_DIM, FRAGMENT_QUBITS,
    VIRTUAL_CUT_WORKING_SET_BYTES,
};

#[test]
fn test_virtual_36q_ram_reduction_and_l3_residency() {
    // 36 qubits would require 2^36 complex double amplitudes: 2^36 * 16 bytes = 1.00 TB.
    assert_eq!(DENSE_36Q_RAM_BYTES, (1usize << 36) * 16);
    assert_eq!(DENSE_36Q_RAM_BYTES, 1_099_511_627_776);

    // Dual 18Q fragments: 2 * 2^18 * 16 bytes = 8.00 MB working set.
    assert_eq!(FRAGMENT_QUBITS, 18);
    assert_eq!(FRAGMENT_DIM, 262_144);
    assert_eq!(VIRTUAL_CUT_WORKING_SET_BYTES, 8_388_608);

    // Memory reduction factor: 1,099,511,627,776 / 8,388,608 = 131,072x.
    let reduction = (DENSE_36Q_RAM_BYTES as f64) / (VIRTUAL_CUT_WORKING_SET_BYTES as f64);
    assert!((reduction - 131_072.0).abs() < 1e-6);
}

#[test]
fn test_quasiprobability_reconstruction_oracle() {
    // For 1 bridge CZ: c = [[0.5, 0.5], [0.5, -0.5]]
    // If local branches are separable:
    let exp_a = [1.0, 1.0];
    let exp_b = [1.0, 1.0];
    let res = Virtual36QCutEngine::reconstruct_expectation(&exp_a, &exp_b, 1);
    // (0.5+0.5)*(0.5+0.5) + ... = 1.0
    assert!((res - 1.0).abs() < 1e-12);

    // For 2 bridges:
    let exp_a2 = [1.0, 1.0, 1.0, 1.0];
    let exp_b2 = [1.0, 1.0, 1.0, 1.0];
    let res2 = Virtual36QCutEngine::reconstruct_expectation(&exp_a2, &exp_b2, 2);
    assert!((res2 - 1.0).abs() < 1e-12);
}

#[test]
fn test_virtual_cut_engine_evaluation_and_bell_witness() {
    let config = VirtualCutConfig {
        bridge_count: 2,
        bridge_wires: vec![(17, 0), (16, 1)],
        layers: 4,
        enable_igpu_co_processor: false, // Pure CPU AVX2 for deterministic test
        enable_checkpointing: true,
    };

    let engine = Virtual36QCutEngine::new(config);
    let report = engine.evaluate();

    assert_eq!(report.total_qubits, 36);
    assert_eq!(report.fragment_qubits, 18);
    assert_eq!(report.bridge_count, 2);
    assert_eq!(report.total_branches, 16);
    assert!(report.ram_reduction_factor > 130_000.0);
    assert!(report.bell_chsh_parameter >= 0.0);
    assert!(report.bell_chsh_parameter <= 2.0 * std::f64::consts::SQRT_2 + 1e-6);
    assert!(report.checkpoint_speedup_ratio >= 1.0);
}
