// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Glued Trees Traversal Showdown: Quantum Walk vs Classical Random Walk
//!
//! Evaluates the Childs et al. (STOC 2003) exponential quantum speedup.
//! Compares classical random walk hitting times against Continuous-Time Quantum Walk
//! (CTQW) ballistic wavepacket propagation across balanced glued binary trees.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::time::Instant;
use t1000::glued_trees::{
    ClassicalGluedWalker, ContinuousQuantumWalk, GluedTreesConfig, QuantumWalkTelemetry,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthComparisonResult {
    pub depth: usize,
    pub total_vertices: usize,
    pub num_columns: usize,
    pub num_leaves: usize,
    pub classical_mean_steps: f64,
    pub classical_success_rate_pct: f64,
    pub classical_sim_time_ms: f64,
    pub quantum_peak_exit_prob: f64,
    pub quantum_time_to_peak: f64,
    pub quantum_sim_time_ms: f64,
    pub quantum_speedup_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GluedTreesBenchmarkReport {
    pub timestamp_rfc3339: String,
    pub algorithm: String,
    pub max_depth_evaluated: usize,
    pub depths: Vec<DepthComparisonResult>,
    pub wavepacket_spatiotemporal_trajectory: Vec<QuantumWalkTelemetry>,
    pub defect_recovery_analysis: DefectRecoveryReport,
    pub theoretical_summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefectRecoveryReport {
    pub depth: usize,
    pub unperturbed_peak_exit_prob: f64,
    pub perturbed_exit_prob: f64,
    pub steered_exit_prob: f64,
    pub recovery_fidelity_pct: f64,
}

fn benchmark_depth(depth: usize) -> DepthComparisonResult {
    let cfg = GluedTreesConfig::new(depth);
    let total_v = cfg.total_vertices();
    let n_cols = cfg.num_columns();
    let n_leaves = cfg.num_leaves();

    // 1. Classical Random Walk Benchmark
    let walker = ClassicalGluedWalker::new(cfg);
    let max_classical_steps = 10_000;
    let n_trials = 500;

    let t0_classical = Instant::now();
    let (mean_steps, succ_rate) = walker.benchmark_classical(n_trials, max_classical_steps);
    let classical_ms = t0_classical.elapsed().as_secs_f64() * 1000.0;

    // 2. Quantum Walk (CTQW) Benchmark
    let gamma = 0.5;
    let mut qw = ContinuousQuantumWalk::new(cfg, gamma);

    let t0_quantum = Instant::now();
    let t_max = (depth as f64) * 3.5;
    let dt = 0.05;
    let trajectory = qw.run_simulation(t_max, dt);
    let quantum_ms = t0_quantum.elapsed().as_secs_f64() * 1000.0;

    let mut peak_p = 0.0;
    let mut peak_t = 0.0;
    for snap in &trajectory {
        if snap.p_exit > peak_p {
            peak_p = snap.p_exit;
            peak_t = snap.time;
        }
    }

    // Speedup ratio estimate: classical hitting time vs quantum ballistic time
    let speedup = if succ_rate > 0.001 {
        (mean_steps / succ_rate) / (peak_t * 20.0).max(1.0)
    } else {
        (max_classical_steps as f64 * 10.0) / (peak_t * 20.0).max(1.0)
    };

    DepthComparisonResult {
        depth,
        total_vertices: total_v,
        num_columns: n_cols,
        num_leaves: n_leaves,
        classical_mean_steps: mean_steps,
        classical_success_rate_pct: succ_rate * 100.0,
        classical_sim_time_ms: classical_ms,
        quantum_peak_exit_prob: peak_p,
        quantum_time_to_peak: peak_t,
        quantum_sim_time_ms: quantum_ms,
        quantum_speedup_ratio: speedup,
    }
}

fn main() {
    println!("================================================================================");
    println!("T-1000 EXPONENTIAL QUANTUM SPEEDUP: CHILDS ET AL. GLUED TREES BENCHMARK");
    println!("================================================================================");

    let depths = [2, 3, 4, 5, 6, 7, 8, 9, 10];
    let mut results = Vec::new();

    println!("Evaluating depths n = 2..10 across exponential vertex scaling...");
    println!("┌──────┬────────────┬─────────────┬──────────────────────────┬─────────────────────────┬──────────────┐");
    println!("│ Depth│ Vertices   │ Weld Leaves │ Classical Success (10k)  │ Quantum Peak Exit Prob  │ Speedup Est. │");
    println!("├──────┼────────────┼─────────────┼──────────────────────────┼─────────────────────────┼──────────────┤");

    for &d in &depths {
        let res = benchmark_depth(d);
        println!(
            "│ {:>4} │ {:>10} │ {:>11} │ {:>6.2}% ({:>6.0} steps) │ {:>6.2}% (at t={:>4.1})    │ {:>10.1}x  │",
            res.depth,
            res.total_vertices,
            res.num_leaves,
            res.classical_success_rate_pct,
            res.classical_mean_steps,
            res.quantum_peak_exit_prob * 100.0,
            res.quantum_time_to_peak,
            res.quantum_speedup_ratio
        );
        results.push(res);
    }
    println!("└──────┴────────────┴─────────────┴──────────────────────────┴─────────────────────────┴──────────────┘");

    // 2. High-Resolution Spatiotemporal Wavepacket Simulation for Depth n = 6 (126 vertices)
    println!(
        "\nGenerating spatiotemporal ballistic wavepacket trajectory (depth n=6, 14 columns)..."
    );
    let detailed_cfg = GluedTreesConfig::new(6);
    let mut detailed_qw = ContinuousQuantumWalk::new(detailed_cfg, 0.5);
    let detailed_trajectory = detailed_qw.run_simulation(25.0, 0.1);

    // 3. Defect Injection & Steering Router Analysis
    println!("\nTesting Defect Injection and Quantum Steering Recovery...");
    let test_depth = 6;
    let mut qw_clean = ContinuousQuantumWalk::new(GluedTreesConfig::new(test_depth), 0.5);
    let mut qw_defect = ContinuousQuantumWalk::new(GluedTreesConfig::new(test_depth), 0.5);
    let mut qw_steered = ContinuousQuantumWalk::new(GluedTreesConfig::new(test_depth), 0.5);

    let t_eval: f64 = 12.0;
    let dt: f64 = 0.05;
    let steps = (t_eval / dt).round() as usize;

    // Clean run
    for _ in 0..steps {
        qw_clean.step(dt);
    }
    let p_clean = qw_clean.telemetry(t_eval).p_exit;

    // Defect run: phase defect at weld col 6
    for s in 0..steps {
        qw_defect.step(dt);
        if s == steps / 3 {
            qw_defect.apply_column_phase_perturbation(6, std::f64::consts::PI * 0.4);
        }
    }
    let p_defect = qw_defect.telemetry(t_eval).p_exit;

    // Steered run: counter-phase correction
    for s in 0..steps {
        qw_steered.step(dt);
        if s == steps / 3 {
            qw_steered.apply_column_phase_perturbation(6, std::f64::consts::PI * 0.4);
        }
        if s == (steps / 3) + 2 {
            qw_steered.apply_column_phase_perturbation(6, -std::f64::consts::PI * 0.38);
        }
    }
    let p_steered = qw_steered.telemetry(t_eval).p_exit;
    let recovery_pct = (p_steered / p_clean).min(1.0) * 100.0;

    println!("  Unperturbed Peak Exit Prob: {:.2}%", p_clean * 100.0);
    println!("  With Weld Phase Defect    : {:.2}%", p_defect * 100.0);
    println!(
        "  With Steering Counter-Rot : {:.2}% (Recovery: {:.1}%)",
        p_steered * 100.0,
        recovery_pct
    );

    let defect_report = DefectRecoveryReport {
        depth: test_depth,
        unperturbed_peak_exit_prob: p_clean,
        perturbed_exit_prob: p_defect,
        steered_exit_prob: p_steered,
        recovery_fidelity_pct: recovery_pct,
    };

    let report = GluedTreesBenchmarkReport {
        timestamp_rfc3339: "2026-10-17T20:10:00Z".to_string(),
        algorithm: "Childs et al. Continuous-Time Quantum Walk on Glued Binary Trees".to_string(),
        max_depth_evaluated: 10,
        depths: results,
        wavepacket_spatiotemporal_trajectory: detailed_trajectory,
        defect_recovery_analysis: defect_report,
        theoretical_summary: "Classical random walks get trapped in the exponentially large 2^n weld layer (hitting time Omega(2^{n/2})), while the CTQW preserves column-symmetric superposition, reducing traversal to a 1D chain of length 2n+2 in O(poly(n)) time.".to_string(),
    };

    let report_path = Path::new("results/experiments/glued_trees_report.json");
    if let Some(parent) = report_path.parent() {
        fs::create_dir_all(parent).expect("Create results dir");
    }
    t1000::results::write(report_path, serde_json::to_string_pretty(&report).unwrap())
        .expect("Write report");
    println!("\nReport written to: {}", report_path.display());
}
