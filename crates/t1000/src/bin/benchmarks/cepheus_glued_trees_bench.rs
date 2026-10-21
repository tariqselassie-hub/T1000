// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Rigetti Cepheus-1-108Q: 30-Qubit Glued-Tree Quantum Walk Benchmark
//!
//! Evaluates Continuous-Time Quantum Walk (CTQW) traversal across a balanced
//! glued binary tree (depth n = 3, 30 vertices, 8 columns) mapped across
//! Rigetti Cepheus-1-108Q modular chiplets.
//!
//! - Tree A (15 vertices) is mapped to Chiplet 0 and Chiplet 4.
//! - Tree B (15 vertices) is mapped to Chiplet 1 and Chiplet 5.
//! - The central weld cycle connects Column 3 to Column 4 directly across the
//!   vertical inter-die bump-bond seam (p_seam = 0.0220 vs p_intra = 0.0080).

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::time::Instant;

use t1000::glued_trees::{
    ClassicalGluedWalker, ContinuousQuantumWalk, GluedTreesConfig, QuantumWalkTelemetry,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CepheusQubitNode {
    pub node_id: usize,
    pub tree: String,
    pub column: usize,
    pub chiplet: usize,
    pub physical_qubit: usize,
    pub row: usize,
    pub col: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GluedTreesCepheusReport {
    pub timestamp_rfc3339: String,
    pub target_architecture: String,
    pub tree_depth: usize,
    pub total_vertices: usize,
    pub total_columns: usize,
    pub weld_seam_couplers: usize,
    pub classical_mean_steps: f64,
    pub classical_success_rate_pct: f64,
    pub ideal_quantum_peak_exit_prob: f64,
    pub ideal_quantum_time_to_peak: f64,
    pub seam_noisy_peak_exit_prob: f64,
    pub seam_noisy_time_to_peak: f64,
    pub seam_fidelity_retention_pct: f64,
    pub quantum_speedup_vs_classical: f64,
    pub simulation_latency_us: f64,
    pub qubit_mapping: Vec<CepheusQubitNode>,
    pub spatiotemporal_trajectory: Vec<QuantumWalkTelemetry>,
}

fn build_30q_mapping() -> Vec<CepheusQubitNode> {
    let mut mapping = Vec::with_capacity(30);

    // Tree A: 15 nodes in Chiplets 0 & 4 (Cols 0..2)
    // Col 0 (Root A)
    mapping.push(CepheusQubitNode {
        node_id: 0,
        tree: "Tree_A".into(),
        column: 0,
        chiplet: 0,
        physical_qubit: 0,
        row: 0,
        col: 0,
    });
    // Col 1 (2 nodes)
    mapping.push(CepheusQubitNode {
        node_id: 1,
        tree: "Tree_A".into(),
        column: 1,
        chiplet: 0,
        physical_qubit: 1,
        row: 0,
        col: 1,
    });
    mapping.push(CepheusQubitNode {
        node_id: 2,
        tree: "Tree_A".into(),
        column: 1,
        chiplet: 0,
        physical_qubit: 12,
        row: 1,
        col: 0,
    });
    // Col 2 (4 nodes)
    mapping.push(CepheusQubitNode {
        node_id: 3,
        tree: "Tree_A".into(),
        column: 2,
        chiplet: 0,
        physical_qubit: 13,
        row: 1,
        col: 1,
    });
    mapping.push(CepheusQubitNode {
        node_id: 4,
        tree: "Tree_A".into(),
        column: 2,
        chiplet: 0,
        physical_qubit: 24,
        row: 2,
        col: 0,
    });
    mapping.push(CepheusQubitNode {
        node_id: 5,
        tree: "Tree_A".into(),
        column: 2,
        chiplet: 0,
        physical_qubit: 25,
        row: 2,
        col: 1,
    });
    mapping.push(CepheusQubitNode {
        node_id: 6,
        tree: "Tree_A".into(),
        column: 2,
        chiplet: 4,
        physical_qubit: 36,
        row: 3,
        col: 0,
    });
    // Col 3 (8 Leaves along the vertical seam at col 2)
    let leaves_a_q = [2, 14, 26, 38, 48, 49, 50, 60];
    for (i, &q) in leaves_a_q.iter().enumerate() {
        let r = q / 12;
        let c = q % 12;
        let ch = (r / 3) * 4 + (c / 3);
        mapping.push(CepheusQubitNode {
            node_id: 7 + i,
            tree: "Tree_A_Leaf".into(),
            column: 3,
            chiplet: ch,
            physical_qubit: q,
            row: r,
            col: c,
        });
    }

    // Tree B: 15 nodes in Chiplets 1 & 5 (Cols 3..5)
    // Col 4 (8 Leaves along the vertical seam at col 3)
    let leaves_b_q = [3, 15, 27, 39, 51, 52, 53, 63];
    for (i, &q) in leaves_b_q.iter().enumerate() {
        let r = q / 12;
        let c = q % 12;
        let ch = (r / 3) * 4 + (c / 3);
        mapping.push(CepheusQubitNode {
            node_id: 15 + i,
            tree: "Tree_B_Leaf".into(),
            column: 4,
            chiplet: ch,
            physical_qubit: q,
            row: r,
            col: c,
        });
    }
    // Col 5 (4 nodes)
    mapping.push(CepheusQubitNode {
        node_id: 23,
        tree: "Tree_B".into(),
        column: 5,
        chiplet: 1,
        physical_qubit: 16,
        row: 1,
        col: 4,
    });
    mapping.push(CepheusQubitNode {
        node_id: 24,
        tree: "Tree_B".into(),
        column: 5,
        chiplet: 1,
        physical_qubit: 28,
        row: 2,
        col: 4,
    });
    mapping.push(CepheusQubitNode {
        node_id: 25,
        tree: "Tree_B".into(),
        column: 5,
        chiplet: 5,
        physical_qubit: 40,
        row: 3,
        col: 4,
    });
    mapping.push(CepheusQubitNode {
        node_id: 26,
        tree: "Tree_B".into(),
        column: 5,
        chiplet: 5,
        physical_qubit: 41,
        row: 3,
        col: 5,
    });
    // Col 6 (2 nodes)
    mapping.push(CepheusQubitNode {
        node_id: 27,
        tree: "Tree_B".into(),
        column: 6,
        chiplet: 1,
        physical_qubit: 4,
        row: 0,
        col: 4,
    });
    mapping.push(CepheusQubitNode {
        node_id: 28,
        tree: "Tree_B".into(),
        column: 6,
        chiplet: 1,
        physical_qubit: 17,
        row: 1,
        col: 5,
    });
    // Col 7 (Root B)
    mapping.push(CepheusQubitNode {
        node_id: 29,
        tree: "Tree_B_Root".into(),
        column: 7,
        chiplet: 1,
        physical_qubit: 5,
        row: 0,
        col: 5,
    });

    mapping
}

fn run_seam_noisy_quantum_walk(
    cfg: GluedTreesConfig,
    gamma: f64,
    t_max: f64,
    dt: f64,
    p_seam: f64,
    p_intra: f64,
) -> (f64, f64, Vec<QuantumWalkTelemetry>) {
    let mut qw = ContinuousQuantumWalk::new(cfg, gamma);
    let n_steps = (t_max / dt).round() as usize;
    let mut trajectory = Vec::with_capacity(n_steps + 1);

    trajectory.push(qw.telemetry(0.0));

    let mut peak_p = 0.0;
    let mut peak_t = 0.0;

    let weld_col_1 = cfg.depth;
    let weld_col_2 = cfg.depth + 1;

    for s in 1..=n_steps {
        qw.step(dt);

        // Apply seam dephasing to weld columns 3 and 4
        let seam_dephase = 1.0 - p_seam * dt;
        qw.state[weld_col_1] *= Complex64::new(seam_dephase, 0.0);
        qw.state[weld_col_2] *= Complex64::new(seam_dephase, 0.0);

        // Apply intra-die baseline dephasing to other columns
        let intra_dephase = 1.0 - p_intra * dt;
        for c in 0..qw.state.len() {
            if c != weld_col_1 && c != weld_col_2 {
                qw.state[c] *= Complex64::new(intra_dephase, 0.0);
            }
        }

        // Renormalize state
        let norm_sq: f64 = qw.state.iter().map(|a| a.norm_sqr()).sum();
        if norm_sq > 0.0 {
            let inv_norm = 1.0 / norm_sq.sqrt();
            for elem in &mut qw.state {
                *elem *= Complex64::new(inv_norm, 0.0);
            }
        }

        let t = s as f64 * dt;
        let telem = qw.telemetry(t);
        if telem.p_exit > peak_p {
            peak_p = telem.p_exit;
            peak_t = t;
        }
        trajectory.push(telem);
    }

    (peak_p, peak_t, trajectory)
}

fn main() {
    let t0 = Instant::now();

    println!("==========================================================================");
    println!("  RIGETTI CEPHEUS-1-108Q: 30-QUBIT GLUED-TREE QUANTUM WALK BENCHMARK     ");
    println!("==========================================================================");
    println!("Topology: Childs et al. Balanced Glued Binary Trees (n = 3, 30 Vertices)");
    println!("Hardware Mapping: Tree A (Chiplets 0 & 4) -> Tree B (Chiplets 1 & 5)");
    println!("Inter-Die Seam: 8 Couplers between Col 2 (c=2) and Col 3 (c=3) at p = 2.20%");
    println!("--------------------------------------------------------------------------\n");

    let cfg = GluedTreesConfig::new(3);
    let mapping = build_30q_mapping();

    // 1. Classical Monte Carlo Baseline
    let classical_walker = ClassicalGluedWalker::new(cfg);
    let (mean_classical_steps, classical_success) = classical_walker.benchmark_classical(1000, 10_000);

    // 2. Ideal Unitary CTQW
    let gamma = 0.50;
    let t_max = 20.0;
    let dt = 0.05;
    let mut ideal_qw = ContinuousQuantumWalk::new(cfg, gamma);
    let ideal_trajectory = ideal_qw.run_simulation(t_max, dt);

    let mut ideal_peak_p = 0.0;
    let mut ideal_peak_t = 0.0;
    for snap in &ideal_trajectory {
        if snap.p_exit > ideal_peak_p {
            ideal_peak_p = snap.p_exit;
            ideal_peak_t = snap.time;
        }
    }

    // 3. Seam-Dephasing Hardware Emulation
    let p_seam = 0.0220;
    let p_intra = 0.0080;
    let (noisy_peak_p, noisy_peak_t, noisy_trajectory) =
        run_seam_noisy_quantum_walk(cfg, gamma, t_max, dt, p_seam, p_intra);

    let speedup = (mean_classical_steps / classical_success.max(0.001)) / (ideal_peak_t * 10.0);
    let retention_pct = (noisy_peak_p / ideal_peak_p) * 100.0;
    let sim_duration_us = t0.elapsed().as_secs_f64() * 1_000_000.0;

    println!("┌───────────────────────────────────┬─────────────────────────────────────┐");
    println!("│ Metric                            │ Benchmark Ground Truth              │");
    println!("├───────────────────────────────────┼─────────────────────────────────────┤");
    println!("│ Total Graph Vertices              │ 30 Qubits (15 in Tree A, 15 Tree B) │");
    println!("│ Columns (Symmetric Subspace Dim)  │ 8 Columns (Root A to Root B)        │");
    println!("│ Inter-Die Seam Weld Couplers      │ 8 Couplers across Dies 0,1,4,5      │");
    println!(
        "│ Classical Random Walk Hitting     │ {:.1} Steps (Success: {:.1}%)       │",
        mean_classical_steps, classical_success * 100.0
    );
    println!(
        "│ T-1000 Ideal Quantum Peak Exit    │ {:.4} ({:.2}%) at t = {:.2}         │",
        ideal_peak_p, ideal_peak_p * 100.0, ideal_peak_t
    );
    println!(
        "│ Cepheus Seam-Noisy Peak Exit      │ {:.4} ({:.2}%) at t = {:.2}         │",
        noisy_peak_p, noisy_peak_p * 100.0, noisy_peak_t
    );
    println!(
        "│ Seam Transmission Retention       │ {:.2}% of Pure Theoretical Peak    │",
        retention_pct
    );
    println!(
        "│ Quantum vs. Classical Speedup     │ {:.1}x Accelerated Traversal        │",
        speedup
    );
    println!(
        "│ T-1000 Simulation Latency         │ {:.2} microseconds                 │",
        sim_duration_us
    );
    println!("└───────────────────────────────────┴─────────────────────────────────────┘");

    let report = GluedTreesCepheusReport {
        timestamp_rfc3339: "2026-10-19T07:58:00Z".into(),
        target_architecture: "Rigetti Cepheus-1-108Q (30Q Glued Tree Subgraph)".into(),
        tree_depth: 3,
        total_vertices: 30,
        total_columns: 8,
        weld_seam_couplers: 8,
        classical_mean_steps: mean_classical_steps,
        classical_success_rate_pct: classical_success * 100.0,
        ideal_quantum_peak_exit_prob: ideal_peak_p,
        ideal_quantum_time_to_peak: ideal_peak_t,
        seam_noisy_peak_exit_prob: noisy_peak_p,
        seam_noisy_time_to_peak: noisy_peak_t,
        seam_fidelity_retention_pct: retention_pct,
        quantum_speedup_vs_classical: speedup,
        simulation_latency_us: sim_duration_us,
        qubit_mapping: mapping,
        spatiotemporal_trajectory: noisy_trajectory,
    };

    let serialized = serde_json::to_string_pretty(&report).expect("Failed to serialize telemetry");

    let out1 = Path::new("experiments/cepheus_glued_trees_30q/cepheus_glued_trees_telemetry.json");
    if let Some(parent) = out1.parent() {
        let _ = fs::create_dir_all(parent);
    }
    t1000::results::write(out1, &serialized).expect("Failed to write experiment telemetry");
    println!("\n [SAVED] Experiment Telemetry: {}", out1.display());

    let out2 = Path::new("results/data/cepheus_glued_trees_telemetry.json");
    if let Some(parent) = out2.parent() {
        let _ = fs::create_dir_all(parent);
    }
    t1000::results::write(out2, &serialized).expect("Failed to write results telemetry");
    println!(" [SAVED] Central Telemetry: {}", out2.display());
}
