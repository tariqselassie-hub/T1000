// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Integration tests for holographic bulk quantum error correction and size-winding optimizer.

use t1000::wormhole::{
    evaluate_holographic_qec, ErasurePartition, OptimizerConfig, ScramblerType,
    ScramblingHamiltonian, SizeWindingOptimizer, WormholeConfig,
};

#[test]
fn test_size_winding_optimizer_improves_fidelity() {
    let opt_config = OptimizerConfig {
        num_modes: 3,
        max_iterations: 35,
        step_size: 0.15,
        perturbation: 0.06,
        target_g: 1.5,
        seed: 0xCAFE_BABE,
    };

    let initial = ScramblingHamiltonian::new(
        ScramblerType::ChaoticHeisenberg,
        opt_config.num_modes,
        opt_config.seed,
    );

    let optimizer = SizeWindingOptimizer::new(opt_config);
    let (best_h, trajectory) = optimizer.optimize(&initial);

    assert!(!trajectory.is_empty(), "Trajectory must contain optimization steps");
    let initial_fid = trajectory.first().unwrap().fidelity;
    let final_fid = trajectory.last().unwrap().fidelity;
    let max_fid = trajectory.iter().map(|p| p.fidelity).fold(0.0_f64, f64::max);

    println!(
        "Optimizer verification: initial={:.4}, final={:.4}, peak={:.4}",
        initial_fid, final_fid, max_fid
    );

    // Peak fidelity found must be strictly greater than initial unoptimized fidelity
    assert!(
        max_fid > initial_fid,
        "SPSA optimization must discover couplings with higher fidelity than baseline"
    );
    assert!(
        best_h.pairwise_couplings.len() == initial.pairwise_couplings.len(),
        "Hamiltonian parameter shape must be invariant under optimization"
    );
}

#[test]
fn test_holographic_qec_erasure_resilience() {
    let engine_config = WormholeConfig {
        num_modes: 3,
        scrambler_type: ScramblerType::ChaoticHeisenberg,
        trotter_steps: 3,
        scrambling_time: 1.5,
        coupling_g: 1.5,
        seed: 0x1337_D00D,
    };

    let hamiltonian = ScramblingHamiltonian::new(
        ScramblerType::ChaoticHeisenberg,
        engine_config.num_modes,
        engine_config.seed,
    );

    // Evaluate Left horizon erasures
    let report = evaluate_holographic_qec(
        &engine_config,
        &hamiltonian,
        ErasurePartition::LeftHorizon,
        1.5,
    );

    assert_eq!(report.num_modes, 3);
    assert_eq!(report.total_qubits, 7);
    assert_eq!(report.points.len(), 3);

    // With 0 erasures, throat is open and recovered fidelity matches clean fidelity
    let p0 = &report.points[0];
    assert_eq!(p0.erased_count, 0);
    assert!(p0.throat_open, "Clean throat must remain open");
    assert!(
        (p0.recovered_fidelity - report.clean_fidelity).abs() < 1e-6,
        "Point 0 fidelity must equal clean baseline"
    );
}

#[test]
fn test_holographic_code_distance_boundary_sweep() {
    let opt_config = OptimizerConfig {
        num_modes: 3,
        max_iterations: 30,
        step_size: 0.15,
        perturbation: 0.06,
        target_g: 1.5,
        seed: 0x4242_1000,
    };

    let initial = ScramblingHamiltonian::new(
        ScramblerType::ChaoticHeisenberg,
        opt_config.num_modes,
        opt_config.seed,
    );

    let optimizer = SizeWindingOptimizer::new(opt_config.clone());
    let (optimized_h, _) = optimizer.optimize(&initial);

    let engine_config = WormholeConfig {
        num_modes: opt_config.num_modes,
        scrambler_type: ScramblerType::ChaoticHeisenberg,
        trotter_steps: 3,
        scrambling_time: 1.5,
        coupling_g: opt_config.target_g,
        seed: opt_config.seed,
    };

    let report_right = evaluate_holographic_qec(
        &engine_config,
        &optimized_h,
        ErasurePartition::RightHorizon,
        opt_config.target_g,
    );

    assert_eq!(report_right.num_modes, 3);
    assert_eq!(report_right.points.len(), 3);

    // Holographic code distance must be >= 1
    assert!(
        report_right.holographic_code_distance >= 1,
        "Holographic code distance d_holo must be at least 1"
    );

    let p0 = &report_right.points[0];
    let p_last = report_right.points.last().unwrap();
    println!("Clean optimized fidelity: {:.4}, max erasure fidelity: {:.4}", p0.recovered_fidelity, p_last.recovered_fidelity);
    assert!(
        p0.recovered_fidelity > p_last.recovered_fidelity,
        "Optimized throat fidelity must strictly exceed total erasure noise floor"
    );
}
