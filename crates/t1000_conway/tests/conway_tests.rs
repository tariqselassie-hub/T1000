// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Integration & Verification Tests for T-1000 Quantum Conway Engine.

use t1000_conway::{
    build_holonomy_unitary, ffs_momentum_plateau, geometric_holonomy_phase,
    BoundaryCondition, ClassicalConwayEngine, HolonomyParams, Lattice2D,
    PatternFactory, PatternType, QuantumConwaySimulator,
};
use std::f64::consts::PI;

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
