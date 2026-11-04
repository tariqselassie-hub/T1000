// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Integration tests for the Quantum Universe & Cosmological Spacetime Simulator.

use t1000::cosmology::{
    CosmologyConfig, CosmologyEngine, ExpansionProfile,
};

#[test]
fn test_bunch_davies_wronskian_unitarity() {
    let config = CosmologyConfig {
        profile: ExpansionProfile::DeSitter,
        hubble_rate: 1.5,
        field_mass: 0.1,
        num_modes: 12,
        box_length: 20.0,
        t_initial: 0.0,
        t_final: 2.0,
        time_steps: 80,
    };

    let mut engine = CosmologyEngine::new(config);
    let report = engine.run_simulation();

    // Verify microscopic symplectic unitarity across the entire cosmic evolution
    println!(
        "Max Wronskian unitarity error across all modes and epochs: {:.2e}",
        report.max_unitarity_error
    );
    assert!(
        report.max_unitarity_error < 1e-4,
        "Symplectic integrator must conserve microscopic canonical commutation Wronskian"
    );
}

#[test]
fn test_minkowski_static_spacetime_zero_particle_creation() {
    // In static Minkowski spacetime (a(t) = 1.0, H = 0.0), Parker's theorem requires zero particle creation
    let config = CosmologyConfig {
        profile: ExpansionProfile::StaticMinkowski,
        hubble_rate: 0.0,
        field_mass: 0.2,
        num_modes: 8,
        box_length: 15.0,
        t_initial: 0.0,
        t_final: 2.0,
        time_steps: 60,
    };

    let mut engine = CosmologyEngine::new(config);
    let report = engine.run_simulation();

    println!(
        "Static Minkowski total particles created: {:.2e}",
        report.total_particles_created
    );
    assert!(
        report.total_particles_created < 1e-5,
        "Static spacetime must create zero particles out of Bunch-Davies vacuum"
    );
    assert_eq!(
        report.final_horizon.super_horizon_modes_count, 0,
        "Static flat spacetime has infinite horizon; no modes can be super-horizon"
    );
}

#[test]
fn test_de_sitter_super_horizon_particle_creation() {
    // Under exponential expansion, modes stretch beyond the Hubble horizon and undergo parametric amplification
    let config = CosmologyConfig {
        profile: ExpansionProfile::DeSitter,
        hubble_rate: 1.2,
        field_mass: 0.05,
        num_modes: 16,
        box_length: 25.0,
        t_initial: 0.0,
        t_final: 2.5,
        time_steps: 100,
    };

    let mut engine = CosmologyEngine::new(config);
    let report = engine.run_simulation();

    println!(
        "De Sitter expansion: scale_factor={:.2}, particles_created={:.4}, super_horizon_modes={}",
        report.final_spectrum.scale_factor,
        report.total_particles_created,
        report.final_horizon.super_horizon_modes_count
    );

    assert!(
        report.final_spectrum.scale_factor > 15.0,
        "Scale factor must expand exponentially in de Sitter"
    );
    assert!(
        report.total_particles_created > 0.01,
        "Expanding spacetime must generate non-zero Bogoliubov particle density"
    );
    assert!(
        report.final_horizon.super_horizon_modes_count > 0,
        "Long-wavelength modes must freeze out beyond the comoving Hubble horizon"
    );
}

#[test]
fn test_cosmological_entanglement_entropy_growth() {
    let config = CosmologyConfig {
        profile: ExpansionProfile::DeSitter,
        hubble_rate: 1.0,
        field_mass: 0.1,
        num_modes: 14,
        box_length: 20.0,
        t_initial: 0.0,
        t_final: 2.2,
        time_steps: 70,
    };

    let mut engine = CosmologyEngine::new(config);
    let report = engine.run_simulation();

    let initial_entropy = report.epochs.first().unwrap().horizon_entanglement_entropy;
    let final_entropy = report.final_horizon.total_entanglement_entropy;

    println!(
        "Horizon entanglement entropy: initial={:.4}, final={:.4}, peak={:.4}",
        initial_entropy, final_entropy, report.peak_entanglement_entropy
    );

    assert!(
        final_entropy >= initial_entropy,
        "Cosmological horizon entanglement entropy must accumulate as modes freeze out"
    );
}

#[test]
fn test_bouncing_cosmology_evolution() {
    // Non-singular bounce: a(t) = sqrt(1 + (H t)^2)
    let config = CosmologyConfig {
        profile: ExpansionProfile::BouncingCosmology,
        hubble_rate: 1.0,
        field_mass: 0.1,
        num_modes: 10,
        box_length: 20.0,
        t_initial: 0.0,
        t_final: 2.0,
        time_steps: 60,
    };

    let mut engine = CosmologyEngine::new(config);
    let report = engine.run_simulation();

    assert!(report.max_unitarity_error < 1e-4);
    assert!(report.final_spectrum.scale_factor > 1.0);
}

#[test]
fn test_zeldovich_cosmic_web_structure_formation() {
    let config = CosmologyConfig {
        profile: ExpansionProfile::DeSitter,
        hubble_rate: 1.2,
        field_mass: 0.1,
        num_modes: 16,
        box_length: 20.0,
        t_initial: 0.0,
        t_final: 2.5,
        time_steps: 60,
    };

    let mut engine = CosmologyEngine::new(config);
    let report = engine.run_simulation();

    let macro_struct = report
        .macro_structure
        .expect("Simulation report must generate macroscopic structure snapshot");

    println!(
        "Macro cosmic structure: nodes={}, filaments={}, clusters={}, void_fraction={:.2}%, horizon_r={:.2}",
        macro_struct.nodes.len(),
        macro_struct.filament_count,
        macro_struct.cluster_count,
        macro_struct.void_fraction * 100.0,
        macro_struct.hubble_horizon_radius
    );

    assert_eq!(macro_struct.nodes.len(), 96, "Expected 96 cosmic web tracer nodes");
    assert!(macro_struct.filament_count > 0, "Must form cosmic web filament bridges");
    assert!(macro_struct.cluster_count > 0, "Must collapse virialized halos (delta > 1.686)");
    assert!(macro_struct.hubble_horizon_radius > 0.0, "Horizon radius must be positive");
}

