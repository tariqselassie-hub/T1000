// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Integration tests for 2D Multi-Cut Plaquette-Z boundary contraction,
//! quasiprobability channel optimization, and low-rank SVD tree contraction.

use t1000::ansatz::multi_cut_plaquette::*;

#[test]
fn test_plaquette_spoke_cancellation() {
    // Construct boundary edges forming adjacent triangles sharing boundary vertex v=3
    // (q0, q3) and (q1, q3) on Cluster A -> Cluster B
    let edges = vec![
        BoundaryEdge::new(0, 3),
        BoundaryEdge::new(1, 3),
        BoundaryEdge::new(2, 4),
    ];

    let topology = PlaquetteBoundaryTopology::new(edges);
    assert!(topology.eliminated_operators >= 2);
    // Vertex 3 appeared twice: Z_3 * Z_3 = I_3, eliminating duplicate operators
    assert_eq!(topology.raw_edges.len(), 3);
}

#[test]
fn test_quasiprobability_sampling_overhead_suppression() {
    // K = 8 boundary cuts with moderate entangling angles
    let angles = vec![0.35, 0.40, 0.30, 0.45, 0.38, 0.42, 0.33, 0.37];
    let opt = QuasiprobabilityOptimizer::new(angles);

    let gamma_indep = opt.independent_gamma();
    assert_eq!(gamma_indep, 6561.0); // 3^8

    let var_indep = opt.independent_variance_overhead();
    assert!((var_indep - 4.3046721e7).abs() < 1e3); // ~4.30e7

    let gamma_joint = opt.joint_gamma();
    assert!(gamma_joint < 20.0); // ~18.40 vs 6561

    let var_joint = opt.joint_variance_overhead();
    assert!(var_joint < 400.0); // ~338.5 vs 4.30e7

    let ratio = opt.variance_suppression_ratio();
    assert!(ratio > 1.0e5); // Over 100,000x variance suppression!
}

#[test]
fn test_marginal_tensor_low_rank_svd_truncation() {
    // Construct a rank-deficient 4x4 marginal tensor with zero-variance subspace
    let mut matrix = vec![vec![0.0f64; 4]; 4];
    // Leading rank-1 block
    for i in 0..4 {
        for j in 0..4 {
            matrix[i][j] = 0.25 * (i as f64 + 1.0) * (j as f64 + 1.0);
        }
    }
    // Add tiny numerical noise
    matrix[0][0] += 1e-16;

    let (_recon, report) = truncate_marginal_tensor_svd(&matrix, 1e-6);
    assert_eq!(report.original_rank, 4);
    assert_eq!(report.retained_rank, 1); // Truncated from 4 down to 1 active dimension
    assert!(report.max_retained_singular_value > 1.0);
}

#[test]
fn test_multi_cut_3_cluster_contraction_execution() {
    let edges = vec![
        BoundaryEdge::new(17, 0),
        BoundaryEdge::new(16, 1),
    ];
    let angles = vec![0.30, 0.40];

    // Mock evaluations for Clusters A, B, and C
    let eval_a = |ins: &[bool]| -> f64 {
        if ins.iter().any(|&z| z) { 0.85 } else { 1.0 }
    };
    let eval_b = |ins: &[bool]| -> f64 {
        if ins.iter().any(|&z| z) { -0.75 } else { 1.0 }
    };
    let eval_c = |ins: &[bool]| -> f64 {
        if ins.iter().any(|&z| z) { 0.90 } else { 1.0 }
    };

    let report = execute_multi_cut_plaquette_contraction(
        edges,
        angles,
        eval_a,
        eval_b,
        eval_c,
    );

    assert_eq!(report.clusters_count, 3);
    assert_eq!(report.qubits_per_cluster, 18);
    assert_eq!(report.boundary_cuts_count, 2);
    assert!(report.variance_suppression_ratio > 1.0);
    assert!(report.reconstructed_expectation.abs() <= 1.0);
    assert!(report.shot_reduction_factor > 1.0);
}
