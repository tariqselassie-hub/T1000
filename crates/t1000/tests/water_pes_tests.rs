// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Integration tests for the Water (H2O) 2D Energy & Linearization Barrier Engine.

use t1000::water_pes::{
    evaluate_linearization, evaluate_vibrational_manifold, evaluate_water_point, WaterPesConfig,
    WaterPesEngine, DEFAULT_R_EQUILIBRIUM, DEFAULT_THETA_EQUILIBRIUM,
};

#[test]
fn test_water_equilibrium_energy_and_geometry() {
    let pt = evaluate_water_point(DEFAULT_R_EQUILIBRIUM, DEFAULT_THETA_EQUILIBRIUM)
        .expect("Equilibrium evaluation failed");

    println!(
        "Water equilibrium: E = {:.6} Hartree, Dipole = {:.3} Debye, Iterations = {}",
        pt.total_energy, pt.dipole_moment_debye, pt.scf_iterations
    );

    assert!(pt.converged, "RHF SCF must converge at equilibrium");
    assert!(
        (pt.total_energy - (-74.965)).abs() < 0.1,
        "Total energy must match STO-3G water ground state ~ -74.96 Hartree"
    );
    assert!(
        (pt.dipole_moment_debye - 1.854).abs() < 0.05,
        "Dipole moment at equilibrium must match ~ 1.85 Debye"
    );
}

#[test]
fn test_linearization_barrier_magnitude() {
    let lin_report = evaluate_linearization(DEFAULT_R_EQUILIBRIUM, 15)
        .expect("Linearization evaluation failed");

    println!(
        "180° Linearization Barrier: {:.4} Hartree ({:.2} eV, {:.2} kcal/mol, {:.1} cm^-1)",
        lin_report.barrier_hartree,
        lin_report.barrier_ev,
        lin_report.barrier_kcal_mol,
        lin_report.barrier_cm1
    );

    assert!(
        lin_report.barrier_hartree > 0.03,
        "Linearization barrier must be positive and substantial (> 0.03 Hartree)"
    );
    assert!(
        lin_report.barrier_ev > 1.0 && lin_report.barrier_ev < 4.0,
        "Linearization barrier must fall in 1.0 - 4.0 eV physical range"
    );
}

#[test]
fn test_walsh_degeneracy_at_180() {
    let lin_report = evaluate_linearization(DEFAULT_R_EQUILIBRIUM, 21)
        .expect("Linearization evaluation failed");

    let eq_state = &lin_report.walsh_states[0]; // near 80 degrees
    let lin_state = lin_report.walsh_states.last().unwrap(); // 180 degrees

    println!(
        "Walsh splitting: at theta={:.1}° gap={:.6} Hartree; at theta={:.1}° gap={:.6} Hartree",
        eq_state.theta_deg, eq_state.degeneracy_gap, lin_state.theta_deg, lin_state.degeneracy_gap
    );

    assert!(
        eq_state.degeneracy_gap > 0.04,
        "3a1 and 1b1 must be strongly non-degenerate in bent C_2v configuration"
    );
    assert!(
        lin_state.degeneracy_gap < 1e-4,
        "3a1 and 1b1 must merge into degenerate 1pi_u shell at 180° D_inf_h geometry"
    );
}

#[test]
fn test_dipole_collapse_at_linearization() {
    let lin_report = evaluate_linearization(DEFAULT_R_EQUILIBRIUM, 11)
        .expect("Linearization evaluation failed");

    let lin_state = lin_report.walsh_states.last().unwrap();
    println!(
        "Linear configuration dipole moment: {:.6} Debye",
        lin_state.dipole_moment_debye
    );

    assert!(
        lin_state.dipole_moment_debye < 1e-6,
        "Centrosymmetric D_inf_h water must have zero dipole moment"
    );
}

#[test]
fn test_vibrational_frequencies_and_normal_modes() {
    let vib_report = evaluate_vibrational_manifold(DEFAULT_R_EQUILIBRIUM, DEFAULT_THETA_EQUILIBRIUM)
        .expect("Vibrational manifold evaluation failed");

    println!(
        "Normal vibrational frequencies: nu_1 = {:.1} cm^-1 ({}), nu_2 = {:.1} cm^-1 ({}), nu_3 = {:.1} cm^-1 ({})",
        vib_report.modes[0].frequency_cm1, vib_report.modes[0].name,
        vib_report.modes[1].frequency_cm1, vib_report.modes[1].name,
        vib_report.modes[2].frequency_cm1, vib_report.modes[2].name
    );
    println!(
        "Zero-Point Energy: {:.1} cm^-1 ({:.2} kcal/mol)",
        vib_report.total_zpe_cm1, vib_report.total_zpe_kcal_mol
    );

    assert_eq!(vib_report.modes.len(), 3, "Water must possess exactly 3 vibrational normal modes");
    assert!(
        vib_report.modes[1].frequency_cm1 > 1000.0 && vib_report.modes[1].frequency_cm1 < 3000.0,
        "Scissor bending frequency must fall in 1000 - 3000 cm^-1"
    );
    assert!(
        vib_report.modes[0].frequency_cm1 > 3000.0 && vib_report.modes[0].frequency_cm1 < 6500.0,
        "Symmetric stretch frequency must fall in 3000 - 6500 cm^-1"
    );
    assert!(
        vib_report.total_zpe_cm1 > 3000.0,
        "Harmonic zero-point energy must exceed 3000 cm^-1"
    );
}

#[test]
fn test_water_pes_engine_manifold_generation() {
    let config = WaterPesConfig {
        r_min: 0.90,
        r_max: 1.05,
        r_steps: 4,
        theta_min: 95.0,
        theta_max: 180.0,
        theta_steps: 6,
        walsh_steps: 8,
    };
    let engine = WaterPesEngine::new(config);
    let report = engine.compute_manifold().expect("Manifold generation failed");

    assert_eq!(report.grid.points.len(), 24);
    assert!(report.equilibrium_energy < -74.9);
    assert!(report.linearization_barrier_cm1 > 5000.0);
}
