// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Fault-Tolerant Lattice Surgery & Magic State Distillation Verification Tests

use t1000::qec::lattice_surgery::*;
use t1000::qec::magic_distillation::*;

#[test]
fn test_magic_state_distillation_clean_state() {
    // When input error is 0.0, trial must accept with 100% fidelity
    let telem = run_distillation_trial(0.0, 0x5EED_0042);
    assert!(telem.accepted, "Clean magic state inputs must be accepted");
    assert!(
        telem.output_fidelity > 0.999999,
        "Clean distilled state must have unit fidelity, got {}",
        telem.output_fidelity
    );
    assert_eq!(
        telem.syndrome_bits.len(),
        14,
        "Must measure exactly 14 stabilizer generators"
    );
    for &s in &telem.syndrome_bits {
        assert_eq!(s, 1, "All syndrome bits must be +1 for clean inputs");
    }
}

#[test]
fn test_magic_state_cubic_suppression_sweep() {
    let error_rates = [0.01, 0.02, 0.04, 0.08];
    let report = sweep_magic_distillation_factory(&error_rates, 20);

    for telem in &report.sweeps {
        let eps_in = telem.input_error_rate;
        let eps_out = telem.output_infidelity;
        // Verify cubic scaling: eps_out should be closely bounded by 35 * eps_in^3
        let expected_cubic = 35.0 * eps_in.powi(3);
        let ratio = eps_out / expected_cubic;
        assert!(
            (ratio - 1.0).abs() < 0.25,
            "Distillation scaling must match cubic law within factor: eps_in={}, eps_out={}, expected={}, ratio={}",
            eps_in,
            eps_out,
            expected_cubic,
            ratio
        );
        // Acceptance probability must be positive and follow 1 - 105 eps^2
        assert!(
            telem.acceptance_probability > 0.0,
            "Acceptance prob must be positive"
        );
    }
}

#[test]
fn test_dual_planar_patch_encoding_and_observables() {
    // Encode |0_L> on Patch 1 and |1_L> on Patch 2
    let (state, p1, p2) = encode_dual_planar_patches(
        (1.0, 0.0), (0.0, 0.0), // |0_L>
        (0.0, 0.0), (1.0, 0.0), // |1_L>
    );

    assert_eq!(state.dim, 1 << 18, "Dual patch state must be 18 qubits");
    let norm_sq = state.l2_norm_sq();
    assert!(
        (norm_sq - 1.0).abs() < 1e-12,
        "Dual patch state must be normalized"
    );

    // Z_L(1) = +1, Z_L(2) = -1 => Z_L(1) (x) Z_L(2) = -1
    let zz = measure_logical_zz_correlation(&state, &p1, &p2);
    assert!(
        (zz - (-1.0)).abs() < 1e-6,
        "Logical <Z1 Z2> for |01>_L must be -1.0, got {}",
        zz
    );
}

#[test]
fn test_lattice_surgery_z_merge_and_x_merge() {
    // Encode |0_L> on Patch 1 and |0_L> on Patch 2
    let (mut state, p1, p2) = encode_dual_planar_patches(
        (1.0, 0.0), (0.0, 0.0), // |0_L>
        (1.0, 0.0), (0.0, 0.0), // |0_L>
    );

    let (z_out, z_exp) = perform_z_merge(&mut state, &p1, &p2);
    assert_eq!(z_out, 1, "Z-merge on |00>_L must yield outcome +1");
    assert!(
        (z_exp - 1.0).abs() < 1e-6,
        "Z expectation must be +1.0, got {}",
        z_exp
    );

    // X-merge on |++>_L
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    let (mut state_plus, p1_plus, p2_plus) = encode_dual_planar_patches(
        (inv_sqrt2, 0.0), (inv_sqrt2, 0.0), // |+_L>
        (inv_sqrt2, 0.0), (inv_sqrt2, 0.0), // |+_L>
    );

    let (x_out, x_exp) = perform_x_merge(&mut state_plus, &p1_plus, &p2_plus);
    assert_eq!(x_out, 1, "X-merge on |++>_L must yield outcome +1");
    assert!(
        (x_exp - 1.0).abs() < 1e-6,
        "X expectation must be +1.0, got {}",
        x_exp
    );
}

#[test]
fn test_fault_tolerant_logical_cnot_truth_table() {
    let basis_pairs = [
        ((1.0, 0.0), (0.0, 0.0), (1.0, 0.0), (0.0, 0.0), 1.0, 1.0),   // |00> -> |00>, ZZ = (+1)*(+1) = +1.0
        ((1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0), -1.0, 1.0),  // |01> -> |01>, ZZ = (+1)*(-1) = -1.0
        ((0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 0.0), 1.0, 1.0),   // |10> -> |11>, ZZ = (-1)*(-1) = +1.0
        ((0.0, 0.0), (1.0, 0.0), (0.0, 0.0), (1.0, 0.0), -1.0, 1.0),  // |11> -> |10>, ZZ = (-1)*(+1) = -1.0
    ];

    for (a1, b1, a2, b2, exp_zz, _exp_norm) in basis_pairs {
        let (mut state, p1, p2) = encode_dual_planar_patches(a1, b1, a2, b2);
        let telem = execute_fault_tolerant_logical_cnot(&mut state, &p1, &p2);
        assert!(
            (telem.logical_zz_correlation - exp_zz).abs() < 1e-5,
            "CNOT output ZZ correlation mismatch: expected {}, got {}",
            exp_zz,
            telem.logical_zz_correlation
        );
    }
}

#[test]
fn test_fault_tolerant_logical_bell_pair_generation() {
    let (state, p1, p2, telem) = generate_fault_tolerant_bell_pair();

    assert!(
        (telem.logical_zz_correlation - 1.0).abs() < 1e-5,
        "Logical Bell pair must have <Z1 Z2> = +1.0, got {}",
        telem.logical_zz_correlation
    );
    assert!(
        (telem.logical_xx_correlation - 1.0).abs() < 1e-5,
        "Logical Bell pair must have <X1 X2> = +1.0, got {}",
        telem.logical_xx_correlation
    );

    let fidelity = compute_logical_bell_fidelity(&state, &p1, &p2);
    assert!(
        fidelity > 0.99999,
        "Logical Bell pair fidelity must be > 0.99999, got {}",
        fidelity
    );
}
