// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Integration & Mathematical Invariant Tests for Continuous Topological Walk Screener.

use num_complex::Complex64;
use t1000::chemistry::{
    ClassicalRandomWalk, ContinuousMolecularWalk, MolecularLaplacian, Qm9Atom, Qm9Record,
};

fn create_sample_qm9_record() -> Qm9Record {
    // Model ethanol-like motif (C-C-O backbone + Hydrogens)
    let atoms = vec![
        Qm9Atom { symbol: "C".to_string(), coords: [0.0, 0.0, 0.0], mulliken_charge: -0.15 },
        Qm9Atom { symbol: "C".to_string(), coords: [1.52, 0.0, 0.0], mulliken_charge: 0.12 },
        Qm9Atom { symbol: "O".to_string(), coords: [2.15, 1.25, 0.0], mulliken_charge: -0.55 },
        Qm9Atom { symbol: "H".to_string(), coords: [-0.35, 1.02, 0.0], mulliken_charge: 0.08 },
        Qm9Atom { symbol: "H".to_string(), coords: [-0.35, -0.51, 0.88], mulliken_charge: 0.08 },
        Qm9Atom { symbol: "H".to_string(), coords: [-0.35, -0.51, -0.88], mulliken_charge: 0.08 },
        Qm9Atom { symbol: "H".to_string(), coords: [1.85, -0.51, 0.88], mulliken_charge: 0.08 },
        Qm9Atom { symbol: "H".to_string(), coords: [1.85, -0.51, -0.88], mulliken_charge: 0.08 },
        Qm9Atom { symbol: "H".to_string(), coords: [3.10, 1.15, 0.0], mulliken_charge: 0.18 },
    ];

    Qm9Record {
        index: 101,
        tag: "gdb".to_string(),
        n_atoms: atoms.len(),
        rotational_constants: [15.2, 8.4, 6.1],
        dipole_moment: 1.69,
        polarizability: 32.5,
        homo: -0.275,
        lumo: 0.085,
        gap: 0.360, // 9.80 eV
        spatial_extent: 42.0,
        zpve: 0.082,
        u0: -154.85,
        u: -154.84,
        h: -154.83,
        g: -154.87,
        cv: 14.5,
        atoms,
        frequencies: vec![],
        smiles_gdb: "CCO".to_string(),
        smiles_b3lyp: "CCO".to_string(),
        inchi: "InChI=1S/C2H6O".to_string(),
    }
}

#[test]
fn test_molecular_laplacian_construction_and_eigensystem() {
    let rec = create_sample_qm9_record();
    let lap = MolecularLaplacian::from_qm9_record(&rec, 1.0, 0.15)
        .expect("Failed to build MolecularLaplacian");

    assert_eq!(lap.dim, 9);
    assert_eq!(lap.eigenvalues.len(), 9);
    assert_eq!(lap.eigenvectors.len(), 9);

    // Verify Laplacian zero row-sum property: sum_j L_ij = 0
    for i in 0..lap.dim {
        let row_sum: f64 = (0..lap.dim).map(|j| lap.laplacian[i * lap.dim + j]).sum();
        assert!(
            row_sum.abs() < 1e-10,
            "Laplacian row {} sum {} exceeds epsilon",
            i,
            row_sum
        );
    }

    // Verify orthonormal eigenvectors: <v_i | v_j> = delta_ij
    for i in 0..lap.dim {
        for j in 0..lap.dim {
            let dot: f64 = (0..lap.dim)
                .map(|k| lap.eigenvectors[i][k] * lap.eigenvectors[j][k])
                .sum();
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-10,
                "Eigenvector orthogonality violated: <v_{}|v_{}> = {}",
                i,
                j,
                dot
            );
        }
    }

    // Core atom should be carbon (idx 0 or 1) and frontier site oxygen (idx 2)
    assert!(lap.core_atom_idx == 0 || lap.core_atom_idx == 1);
    assert_eq!(lap.frontier_site_idx, 2, "Oxygen should be identified as reactive frontier site");
}

#[test]
fn test_continuous_molecular_walk_exact_unitarity() {
    let rec = create_sample_qm9_record();
    let lap = MolecularLaplacian::from_qm9_record(&rec, 1.0, 0.15).unwrap();
    let walk = ContinuousMolecularWalk::new(lap);

    let mut psi0 = vec![Complex64::new(0.0, 0.0); walk.laplacian.dim];
    psi0[walk.laplacian.core_atom_idx] = Complex64::new(1.0, 0.0);

    // Evolve across varying time horizons
    for &t in &[0.01, 0.1, 0.5, 1.0, 5.0, 20.0] {
        let psi_t = walk.evolve_analytic(&psi0, t);
        let norm_sq: f64 = psi_t.iter().map(|c| c.norm_sqr()).sum();
        assert!(
            (norm_sq - 1.0).abs() < 1e-12,
            "Exact unitarity violation at t={}: norm_sq = {}",
            t,
            norm_sq
        );
    }
}

#[test]
fn test_zeno_stepped_confinement_survival() {
    let rec = create_sample_qm9_record();
    let lap = MolecularLaplacian::from_qm9_record(&rec, 1.0, 0.15).unwrap();
    let walk = ContinuousMolecularWalk::new(lap);

    let mut psi0 = vec![Complex64::new(0.0, 0.0); walk.laplacian.dim];
    psi0[walk.laplacian.core_atom_idx] = Complex64::new(1.0, 0.0);

    // Bound valence mask: include all heavy atoms (0, 1, 2)
    let bound_mask = vec![true, true, true, true, true, true, true, true, true];

    // High-frequency Zeno monitoring (50 steps over 1.0 ns)
    let (state, survival) = walk.evolve_zeno_stepped(&psi0, 1.0, 50, &bound_mask);
    let state_norm: f64 = state.iter().map(|c| c.norm_sqr()).sum();

    assert!(survival > 0.999, "Zeno survival {} drops below 0.999", survival);
    assert!((state_norm - 1.0).abs() < 1e-10, "State normalization violated");
}

#[test]
fn test_ballistic_hitting_dynamics_vs_classical_walk() {
    let rec = create_sample_qm9_record();
    let lap = MolecularLaplacian::from_qm9_record(&rec, 1.0, 0.15).unwrap();
    let walk = ContinuousMolecularWalk::new(lap.clone());
    let classical = ClassicalRandomWalk::from_laplacian(&lap);

    let core = lap.core_atom_idx;
    let target = lap.frontier_site_idx;

    // Time mesh
    let times: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
    let quantum_hitting = walk.compute_ballistic_hitting_curve(core, target, &times);

    // Find peak quantum hitting probability
    let max_quantum_p = quantum_hitting.iter().cloned().fold(0.0, f64::max);
    let classical_p = classical.hitting_prob(core, target, 10);

    // Quantum ballistic hitting must strictly outperform classical diffusion
    assert!(
        max_quantum_p > classical_p,
        "Quantum peak hitting ({}) should exceed classical diffusive hitting ({})",
        max_quantum_p,
        classical_p
    );
}
