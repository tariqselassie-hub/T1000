// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Holographic Quantum Error Correction & Size-Winding Optimization Benchmark
//!
//! Evaluates:
//! 1. Size-Winding Hamiltonian Learning: optimizing chaotic black hole couplings to maximize
//!    negative energy teleportation fidelity and overcome the generic size-winding gap.
//! 2. Almheiri-Dong-Harlow (ADH) Holographic Bulk Reconstruction: measuring probe recovery
//!    fidelity and entanglement wedge thresholds under boundary erasure channels.

use std::path::PathBuf;
use t1000::results;
use t1000::wormhole::{
    evaluate_holographic_qec, ErasurePartition, HolographicQecReport, HolographicWormholeEngine,
    OptimizerConfig, ScramblerType, ScramblingHamiltonian, SizeWindingOptimizer, WormholeConfig,
};

fn parse_args() -> (OptimizerConfig, PathBuf) {
    let mut config = OptimizerConfig::default();
    let mut out_path = PathBuf::from("results/experiments/holographic_qec.json");

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--modes" if i + 1 < args.len() => {
                config.num_modes = args[i + 1].parse().unwrap_or(3);
                i += 2;
            }
            "--iters" if i + 1 < args.len() => {
                config.max_iterations = args[i + 1].parse().unwrap_or(50);
                i += 2;
            }
            "--out" if i + 1 < args.len() => {
                out_path = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    (config, out_path)
}

fn print_qec_summary(
    clean_fid: f64,
    optimized_fid: f64,
    left_report: &HolographicQecReport,
    right_report: &HolographicQecReport,
) {
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("║       HOLOGRAPHIC QUANTUM ERROR CORRECTION & SIZE-WINDING BENCHMARK         ║");
    println!("║       Almheiri-Dong-Harlow Bulk Reconstruction & Boundary Erasures          ║");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ Baseline Raw Fidelity (Generic Chaos):  {:<35.4} │", clean_fid);
    println!("│ Optimized Size-Wound Bulk Fidelity:     {:<35.4} │", optimized_fid);
    println!("│ Size-Winding Fidelity Gain:             {:<35} │", format!("+{:.2}%", (optimized_fid - clean_fid) / clean_fid * 100.0));
    println!("│ Left Boundary Code Distance (d_holo):   {:<35} │", format!("{}/{} modes", left_report.holographic_code_distance, left_report.num_modes));
    println!("│ Max Tolerable Boundary Erasure Loss:    {:<35} │", format!("{:.1}% boundary loss", left_report.entanglement_wedge_threshold * 100.0));
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ BOUNDARY ERASURE SWEEP: BULK PROBE RECONSTRUCTION FIDELITY                 │");
    println!("├───────────┬───────────────┬──────────────────┬─────────────────┬────────────┤");
    println!("│ Partition │ Erased Modes  │ Erasure Fraction │ Recovered Fid   │ Throat     │");
    println!("├───────────┼───────────────┼──────────────────┼─────────────────┼────────────┤");

    for pt in &left_report.points {
        let status = if pt.throat_open { "OPEN (SURVIVED) " } else { "PINCHED (COLLAPSED)" };
        println!(
            "│ Left      │ {:2} modes     │ {:15.1}% │ {:15.4} │ {:<10} │",
            pt.erased_count, pt.erasure_fraction * 100.0, pt.recovered_fidelity, status
        );
    }
    for pt in &right_report.points {
        let status = if pt.throat_open { "OPEN (SURVIVED) " } else { "PINCHED (COLLAPSED)" };
        println!(
            "│ Right     │ {:2} modes     │ {:15.1}% │ {:15.4} │ {:<10} │",
            pt.erased_count, pt.erasure_fraction * 100.0, pt.recovered_fidelity, status
        );
    }
    println!("└───────────┴───────────────┴──────────────────┴─────────────────┴────────────┘");
}

fn main() {
    let (opt_config, out_path) = parse_args();

    let initial_hamiltonian =
        ScramblingHamiltonian::new(ScramblerType::ChaoticHeisenberg, opt_config.num_modes, opt_config.seed);

    let engine_config = WormholeConfig {
        num_modes: opt_config.num_modes,
        scrambler_type: ScramblerType::ChaoticHeisenberg,
        trotter_steps: 3,
        scrambling_time: 1.5,
        coupling_g: opt_config.target_g,
        seed: opt_config.seed,
    };

    let base_engine = HolographicWormholeEngine::new(engine_config.clone());
    let baseline_pt = base_engine.simulate(opt_config.target_g);
    let baseline_fid = baseline_pt.fidelity;

    println!("[1/2] Optimizing Hamiltonian for maximal holographic size winding...");
    let optimizer = SizeWindingOptimizer::new(opt_config.clone());
    let (optimized_hamiltonian, trajectory) = optimizer.optimize(&initial_hamiltonian);

    let mut opt_engine = HolographicWormholeEngine::new(engine_config.clone());
    opt_engine.scrambler = optimized_hamiltonian.clone();
    let opt_pt = opt_engine.simulate(opt_config.target_g);
    let opt_fid = opt_pt.fidelity;

    println!("[2/2] Evaluating bulk reconstruction under boundary erasure channels...");
    let left_qec = evaluate_holographic_qec(
        &engine_config,
        &optimized_hamiltonian,
        ErasurePartition::LeftHorizon,
        opt_config.target_g,
    );

    let right_qec = evaluate_holographic_qec(
        &engine_config,
        &optimized_hamiltonian,
        ErasurePartition::RightHorizon,
        opt_config.target_g,
    );

    print_qec_summary(baseline_fid, opt_fid, &left_qec, &right_qec);

    #[derive(serde::Serialize)]
    struct CombinedReport {
        baseline_fidelity: f64,
        optimized_fidelity: f64,
        trajectory: Vec<t1000::wormhole::OptimizationTrajectoryPoint>,
        left_erasure: HolographicQecReport,
        right_erasure: HolographicQecReport,
    }

    let combined = CombinedReport {
        baseline_fidelity: baseline_fid,
        optimized_fidelity: opt_fid,
        trajectory,
        left_erasure: left_qec,
        right_erasure: right_qec,
    };

    let serialized = serde_json::to_string_pretty(&combined).expect("Serialization failed");
    results::write(&out_path, &serialized).expect("Failed to write results");
    println!("\nTelemetry recorded to: {}", out_path.display());
}
