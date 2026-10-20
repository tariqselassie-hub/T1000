// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Integration & Verification Tests for T-1000 Quantum Conway Engine.

use std::f64::consts::PI;
use t1000_conway::{
    build_holonomy_unitary, ffs_momentum_plateau, geometric_holonomy_phase, BoundaryCondition,
    ClassicalConwayEngine, HolonomyParams, Lattice2D, PatternFactory, PatternType,
    QuantumConwaySimulator,
};

#[test]
fn test_lattice_geometry_and_wrapping() {
    let lattice = Lattice2D::new(8, 8, BoundaryCondition::Toroidal);
    assert_eq!(lattice.total_cells, 64);
    assert_eq!(lattice.coord_to_idx(3, 4), 35);
    assert_eq!(lattice.idx_to_coord(35), (3, 4));

    // Corner cell (0, 0) should wrap to (7, 7) for top-left neighbor
    let nbrs = lattice.moore_neighbors(0);
    let top_left = nbrs[0];
    assert_eq!(lattice.idx_to_coord(top_left), (7, 7));
}

#[test]
fn test_holonomy_transfer_matrix_unitarity() {
    let params = HolonomyParams::default();

    // Test across various neighborhood occupation counts
    for nbr in [0.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 8.0] {
        let u = build_holonomy_unitary(nbr, &params);
        let err = u.unitarity_error();
        assert!(
            err < 1e-12,
            "Unitarity violated at nbr {}: error = {:.2e}",
            nbr,
            err
        );
    }
}

#[test]
fn test_geometric_holonomy_phases_and_plateau() {
    // For l = 2 (super-fermions): phase = pi/2, plateau = 0.5
    assert!((geometric_holonomy_phase(2) - PI * 0.5).abs() < 1e-14);
    assert!((ffs_momentum_plateau(2) - 0.5).abs() < 1e-14);

    // For l = 4: phase = 3*pi/4, plateau = 0.25
    assert!((geometric_holonomy_phase(4) - PI * 0.75).abs() < 1e-14);
    assert!((ffs_momentum_plateau(4) - 0.25).abs() < 1e-14);
}

#[test]
fn test_classical_conway_engine_rules() {
    let lattice = Lattice2D::new(6, 6, BoundaryCondition::Toroidal);
    let mut engine = ClassicalConwayEngine::new(6, 6);

    // Blinker at (2, 1), (2, 2), (2, 3) (vertical)
    engine.set_cell(2, 1, true);
    engine.set_cell(2, 2, true);
    engine.set_cell(2, 3, true);
    assert_eq!(engine.live_population(), 3);

    // Step 1: Should rotate to horizontal (1, 2), (2, 2), (3, 2)
    engine.step(&lattice);
    assert_eq!(engine.live_population(), 3);
    assert!(engine.grid[lattice.coord_to_idx(1, 2)]);
    assert!(engine.grid[lattice.coord_to_idx(2, 2)]);
    assert!(engine.grid[lattice.coord_to_idx(3, 2)]);

    // Step 2: Should return to vertical
    engine.step(&lattice);
    assert_eq!(engine.live_population(), 3);
    assert!(engine.grid[lattice.coord_to_idx(2, 1)]);
    assert!(engine.grid[lattice.coord_to_idx(2, 2)]);
    assert!(engine.grid[lattice.coord_to_idx(2, 3)]);
}

#[test]
fn test_quantum_conway_norm_preservation() {
    let lattice = Lattice2D::new(8, 8, BoundaryCondition::Toroidal);
    let mut sim = QuantumConwaySimulator::new(lattice, HolonomyParams::default());

    PatternFactory::inject_pattern(PatternType::Glider, &sim.lattice, &mut sim.state, 1, 1);

    for _ in 0..10 {
        let telem = sim.step();
        assert!(
            telem.max_norm_error < 1e-12,
            "State normalization error exceeded threshold: {:.2e}",
            telem.max_norm_error
        );
    }
}

#[test]
fn test_cat_blinker_superposition_preservation() {
    let lattice = Lattice2D::new(8, 8, BoundaryCondition::Toroidal);
    let mut sim = QuantumConwaySimulator::new(lattice, HolonomyParams::default());

    PatternFactory::inject_cat_blinker(&sim.lattice, &mut sim.state, 2, 2);
    let _initial_pop = sim.state.total_population();

    for _ in 0..5 {
        let telem = sim.step();
        assert!(telem.max_norm_error < 1e-12);
        assert!(telem.total_population >= 0.0);
        assert!(telem.total_population <= (sim.lattice.total_cells as f64) * 2.0);
    }

    // Population should remain coherent and bounded
    let final_pop = sim.state.total_population();
    assert!(final_pop > 0.0 && final_pop < (sim.lattice.total_cells as f64));
}

#[test]
fn test_glider_collision_interferometry() {
    let report = t1000_conway::sweep_glider_interferometry(16, 16, 8, 8);
    assert_eq!(report.phase_points, 8);
    assert!(report.max_transmission >= report.min_transmission);
    assert!(report.visibility >= 0.0 && report.visibility <= 1.0);
    assert!(report.max_entanglement_nats > 0.0);

    for point in &report.sweep_results {
        assert!(point.max_norm_error < 1e-12);
        assert!(point.transmission_prob >= 0.0 && point.transmission_prob <= 1.0);
    }
}

#[test]
fn test_simd_vs_scalar_agreement() {
    let params = HolonomyParams::default();
    let u4 = [
        build_holonomy_unitary(1.0, &params),
        build_holonomy_unitary(2.5, &params),
        build_holonomy_unitary(3.0, &params),
        build_holonomy_unitary(4.2, &params),
    ];

    let mut re0_simd = [0.8, 0.2, 0.1, 0.0];
    let mut im0_simd = [0.1, 0.0, 0.0, 0.0];
    let mut re1_simd = [0.2, 0.8, 0.7, 0.5];
    let mut im1_simd = [0.0, 0.1, 0.2, 0.0];
    let mut re2_simd = [0.0, 0.0, 0.2, 0.5];
    let mut im2_simd = [0.0, 0.0, 0.0, 0.1];

    let mut re0_scal = re0_simd;
    let mut im0_scal = im0_simd;
    let mut re1_scal = re1_simd;
    let mut im1_scal = im1_simd;
    let mut re2_scal = re2_simd;
    let mut im2_scal = im2_simd;

    let indices = [0, 1, 2, 3];

    t1000_conway::simd::apply_unitary_4cells(
        &mut re0_simd, &mut im0_simd, &mut re1_simd, &mut im1_simd, &mut re2_simd, &mut im2_simd,
        indices, &u4,
    );

    t1000_conway::simd::apply_unitary_4cells_scalar(
        &mut re0_scal, &mut im0_scal, &mut re1_scal, &mut im1_scal, &mut re2_scal, &mut im2_scal,
        indices, &u4,
    );

    for lane in 0..4 {
        assert!((re0_simd[lane] - re0_scal[lane]).abs() < 1e-14);
        assert!((im0_simd[lane] - im0_scal[lane]).abs() < 1e-14);
        assert!((re1_simd[lane] - re1_scal[lane]).abs() < 1e-14);
        assert!((im1_simd[lane] - im1_scal[lane]).abs() < 1e-14);
        assert!((re2_simd[lane] - re2_scal[lane]).abs() < 1e-14);
        assert!((im2_simd[lane] - im2_scal[lane]).abs() < 1e-14);
    }
}

#[test]
fn test_scale_to_32x32_and_64x64() {
    for &dim in &[32, 64] {
        let lattice = Lattice2D::new(dim, dim, BoundaryCondition::Toroidal);
        let mut sim = QuantumConwaySimulator::new(lattice, HolonomyParams::default());
        PatternFactory::inject_pattern(PatternType::Glider, &sim.lattice, &mut sim.state, dim / 4, dim / 4);

        for _ in 0..5 {
            let telem = sim.step();
            assert!(telem.max_norm_error < 1e-12, "Norm failed at dim {}: {:.2e}", dim, telem.max_norm_error);
        }
    }
}

#[test]
fn test_multi_rule_unitarity_invariance() {
    use t1000_conway::QcaRuleKind;

    let rules = [
        QcaRuleKind::B3S23Holonomy,
        QcaRuleKind::HighLifeReplicator,
        QcaRuleKind::GrossPitaevskiiBec,
        QcaRuleKind::FloquetKickedChaos,
        QcaRuleKind::DiracZitterbewegung,
        QcaRuleKind::CyclicPredatorPrey,
    ];

    for &rule in &rules {
        let mut params = HolonomyParams::default();
        params.rule = rule;
        params.chaos_k = 2.5;
        params.gauge_flux = 0.25;
        params.non_linear_g = 1.5;

        for nbr in [0.0, 1.2, 2.0, 3.0, 4.5, 6.0, 7.8] {
            let u = build_holonomy_unitary(nbr, &params);
            let err = u.unitarity_error();
            assert!(
                err < 1e-12,
                "Unitarity violated for rule {:?} at nbr {}: {:.2e}",
                rule,
                nbr,
                err
            );
        }
    }
}

#[test]
fn test_quantum_vortex_dipole_current_and_circulation() {
    let lattice = Lattice2D::new(16, 16, BoundaryCondition::Toroidal);
    let mut sim = QuantumConwaySimulator::new(lattice, HolonomyParams::default());
    PatternFactory::inject_pattern(PatternType::VortexDipole, &sim.lattice, &mut sim.state, 4, 8);

    let telem = sim.step();
    assert!(telem.max_norm_error < 1e-12);
    assert!(
        telem.max_vorticity > 0.05,
        "Vortex dipole should produce non-zero vorticity: {:.4}",
        telem.max_vorticity
    );
    assert!(
        telem.mean_current_magnitude > 0.0,
        "Current magnitude should be non-zero"
    );
}

#[test]
fn test_floquet_chaos_scrambling() {
    let lattice = Lattice2D::new(16, 16, BoundaryCondition::Toroidal);
    let mut params = HolonomyParams::default();
    params.rule = t1000_conway::QcaRuleKind::FloquetKickedChaos;
    params.chaos_k = 2.0;

    let mut sim = QuantumConwaySimulator::new(lattice, params);
    PatternFactory::inject_pattern(PatternType::Glider, &sim.lattice, &mut sim.state, 2, 2);

    for _ in 0..6 {
        let telem = sim.step();
        assert!(telem.max_norm_error < 1e-12);
        assert!(telem.scrambling_otoc >= 0.0);
    }
}


