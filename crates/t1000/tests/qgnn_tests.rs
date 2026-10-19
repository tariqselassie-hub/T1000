// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Integration tests for Geometric Quantum Graph Neural Network (QGNN).
//!
//! Validates:
//! 1. OpenSMILES graph extraction and feature scaling
//! 2. Unitarity and exact L2 norm conservation
//! 3. Permutation equivariance under node relabeling automorphisms
//! 4. Training loss reduction on molecular property regression
//! 5. Barren plateau variance resilience comparison

use t1000::qgnn::{
    compute_hea_gradient_variance, compute_qgnn_gradient_variance, EquivariantGraphConvLayer,
    GeometricQgnn, MolecularGraph,
};
use t1000::state::QuantumState;

#[test]
fn test_smiles_molecular_graph_parsing() {
    // Benzene: c1ccccc1 (6 carbons, 6 aromatic bonds)
    let graph = MolecularGraph::from_smiles("c1ccccc1", 6).expect("Benzene should parse");
    assert_eq!(graph.num_nodes, 6);
    assert_eq!(graph.edges.len(), 6);
    for edge in &graph.edges {
        assert_eq!(edge.bond_class(), 3); // Aromatic
    }

    // Aspirin: CC(=O)Oc1ccccc1C(=O)O (13 heavy atoms)
    let asp_graph = MolecularGraph::from_smiles("CC(=O)Oc1ccccc1C(=O)O", 8)
        .expect("Aspirin contracted graph should parse");
    assert_eq!(asp_graph.num_nodes, 8);
    assert!(!asp_graph.edges.is_empty());
}

#[test]
fn test_qgnn_unitarity_and_norm_conservation() {
    let graph = MolecularGraph::from_smiles("c1ccccc1", 6).unwrap();
    let dim = 1usize << graph.num_nodes;
    let mut state = QuantumState::ground(dim);

    let layer = EquivariantGraphConvLayer::new(6, 42);
    layer.forward(&mut state, &graph);

    let norm_sq = state.l2_norm_sq();
    assert!(
        (norm_sq - 1.0).abs() < 1e-12,
        "QGNN layer must preserve exact L2 unitarity, got norm_sq = {}",
        norm_sq
    );
}

#[test]
fn test_qgnn_permutation_equivariance() {
    // Benzene with canonical node order
    let graph = MolecularGraph::from_smiles("c1ccccc1", 6).unwrap();

    // Cyclic shift permutation p = [1, 2, 3, 4, 5, 0]
    let perm = vec![1, 2, 3, 4, 5, 0];
    let perm_graph = graph.permute(&perm);

    let qgnn = GeometricQgnn::new(6, 2, 1337);
    let pred_orig = qgnn.predict(&graph);
    let pred_perm = qgnn.predict(&perm_graph);

    let delta = (pred_orig - pred_perm).abs();
    assert!(
        delta < 1e-10,
        "QGNN prediction must be permutation equivariant: orig = {}, perm = {}, delta = {}",
        pred_orig,
        pred_perm,
        delta
    );
}

#[test]
fn test_qgnn_training_step() {
    let ethanol = MolecularGraph::from_smiles("CCO", 4).unwrap();
    let propanol = MolecularGraph::from_smiles("CCCO", 5).unwrap();
    let benzene = MolecularGraph::from_smiles("c1ccccc1", 6).unwrap();

    let dataset = vec![(&ethanol, -0.24), (&propanol, 0.25), (&benzene, 2.13)];

    let mut qgnn = GeometricQgnn::new(6, 2, 777);
    let initial_loss = qgnn.train_step(&dataset, 0.0);
    let trained_loss = qgnn.train_step(&dataset, 0.05);

    assert!(
        trained_loss <= initial_loss,
        "Training step must decrease or maintain MSE loss: initial = {}, trained = {}",
        initial_loss,
        trained_loss
    );
}

#[test]
fn test_barren_plateau_variance_resilience() {
    // Test on 6-qubit system: compare HEA vs QGNN
    let graph = MolecularGraph::from_smiles("c1ccccc1", 6).unwrap();
    let num_samples = 30;

    let hea_var = compute_hea_gradient_variance(6, 4, num_samples, 42);
    let qgnn_var = compute_qgnn_gradient_variance(&graph, 4, num_samples, 42);

    assert!(
        qgnn_var > 0.0,
        "QGNN gradient variance must remain strictly positive"
    );
    assert!(
        qgnn_var > hea_var * 0.5,
        "QGNN gradient variance ({:.4e}) must show robust scaling vs HEA ({:.4e})",
        qgnn_var,
        hea_var
    );
}
