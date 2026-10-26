// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Benchmark Binary: Second Hopf Fibration ($S^7 \to S^4 \cong \mathbb{H}P^1$) & Entanglement Sphere.
//!
//! Sweeps 2-qubit states across the full entanglement spectrum (Bell states, continuous rotation,
//! Werner states, Haar-random ensembles, and molecular eigenstates), measuring quaternionic 5D
//! coordinates, Wootters concurrence, Bell-CHSH witness parameters, and AVX2 SIMD throughput.

use std::fs;
use std::path::Path;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use t1000::hopf_s4::{
    bell_rotation_state, compute_entanglement_invariants, compute_qubit_pair_hopf_s4,
    evaluate_werner_state, project_state_s7_to_s4, EntanglementInvariant, HopfS4Coordinate,
};

#[derive(Serialize, Deserialize, Debug)]
struct BellStatePoint {
    name: String,
    hopf_coord: HopfS4Coordinate,
    invariants: EntanglementInvariant,
}

#[derive(Serialize, Deserialize, Debug)]
struct RotationTrajectoryPoint {
    theta_rad: f64,
    theta_deg: f64,
    hopf_coord: HopfS4Coordinate,
    invariants: EntanglementInvariant,
}

#[derive(Serialize, Deserialize, Debug)]
struct WernerContractionPoint {
    p: f64,
    hopf_coord: HopfS4Coordinate,
    invariants: EntanglementInvariant,
}

#[derive(Serialize, Deserialize, Debug)]
struct SimdThroughputTelemetry {
    iterations: usize,
    elapsed_ms: f64,
    latency_ns_per_eval: f64,
    throughput_m_evals_sec: f64,
    max_norm_error: f64,
}

#[derive(Serialize, Deserialize, Debug)]
struct PairwiseMolecularEntanglement {
    qubit_pair: (usize, usize),
    hopf_coord: HopfS4Coordinate,
    invariants: EntanglementInvariant,
}

#[derive(Serialize, Deserialize, Debug)]
struct HopfS4BenchmarkReport {
    timestamp: String,
    bell_basis_states: Vec<BellStatePoint>,
    rotation_sweep: Vec<RotationTrajectoryPoint>,
    werner_sweep: Vec<WernerContractionPoint>,
    throughput: SimdThroughputTelemetry,
    molecular_pairwise: Vec<PairwiseMolecularEntanglement>,
}

fn benchmark_bell_basis() -> Vec<BellStatePoint> {
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    let states = [
        ("|00>", [1.0, 0.0, 0.0, 0.0], [0.0; 4]),
        ("|01>", [0.0, 1.0, 0.0, 0.0], [0.0; 4]),
        ("|10>", [0.0, 0.0, 1.0, 0.0], [0.0; 4]),
        ("|11>", [0.0, 0.0, 0.0, 1.0], [0.0; 4]),
        ("|Phi+>", [inv_sqrt2, 0.0, 0.0, inv_sqrt2], [0.0; 4]),
        ("|Phi->", [inv_sqrt2, 0.0, 0.0, -inv_sqrt2], [0.0; 4]),
        ("|Psi+>", [0.0, inv_sqrt2, inv_sqrt2, 0.0], [0.0; 4]),
        ("|Psi->", [0.0, inv_sqrt2, -inv_sqrt2, 0.0], [0.0; 4]),
    ];

    states
        .iter()
        .map(|(name, re, im)| {
            let hopf_coord = project_state_s7_to_s4(re, im);
            let invariants = compute_entanglement_invariants(re, im);
            BellStatePoint {
                name: (*name).to_string(),
                hopf_coord,
                invariants,
            }
        })
        .collect()
}

fn benchmark_rotation_trajectory(num_steps: usize) -> Vec<RotationTrajectoryPoint> {
    (0..=num_steps)
        .map(|step| {
            let theta = (step as f64) * (std::f64::consts::PI * 0.5) / (num_steps as f64);
            let (re, im) = bell_rotation_state(theta);
            let hopf_coord = project_state_s7_to_s4(&re, &im);
            let invariants = compute_entanglement_invariants(&re, &im);
            RotationTrajectoryPoint {
                theta_rad: theta,
                theta_deg: theta.to_degrees(),
                hopf_coord,
                invariants,
            }
        })
        .collect()
}

fn benchmark_werner_sweep(num_steps: usize) -> Vec<WernerContractionPoint> {
    (0..=num_steps)
        .map(|step| {
            let p = (step as f64) / (num_steps as f64);
            let (hopf_coord, invariants) = evaluate_werner_state(p);
            WernerContractionPoint {
                p,
                hopf_coord,
                invariants,
            }
        })
        .collect()
}

fn benchmark_simd_throughput(iterations: usize) -> SimdThroughputTelemetry {
    let mut rng = StdRng::seed_from_u64(0x5EED_0042);
    let mut re_batch = Vec::with_capacity(iterations);
    let mut im_batch = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let mut r = [0.0f64; 4];
        let mut im = [0.0f64; 4];
        let mut norm_sq = 0.0;
        for k in 0..4 {
            r[k] = rng.gen_range(-1.0..1.0);
            im[k] = rng.gen_range(-1.0..1.0);
            norm_sq += r[k] * r[k] + im[k] * im[k];
        }
        let inv_norm = 1.0 / norm_sq.sqrt();
        for k in 0..4 {
            r[k] *= inv_norm;
            im[k] *= inv_norm;
        }
        re_batch.push(r);
        im_batch.push(im);
    }

    let t0 = Instant::now();
    let mut max_norm_error = 0.0f64;

    for i in 0..iterations {
        let coord = project_state_s7_to_s4(&re_batch[i], &im_batch[i]);
        let err = coord.sphere_residual;
        if err > max_norm_error {
            max_norm_error = err;
        }
    }

    let elapsed = t0.elapsed();
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let latency_ns_per_eval = (elapsed.as_secs_f64() * 1e9) / (iterations as f64);
    let throughput_m_evals_sec = (iterations as f64) / (elapsed.as_secs_f64() * 1e6);

    SimdThroughputTelemetry {
        iterations,
        elapsed_ms,
        latency_ns_per_eval,
        throughput_m_evals_sec,
        max_norm_error,
    }
}

fn benchmark_molecular_pairwise() -> Vec<PairwiseMolecularEntanglement> {
    // 6-qubit synthetic entangled molecular singlet statevector
    let n_qubits = 6;
    let dim = 1 << n_qubits;
    let mut state_re = vec![0.0; dim];
    let state_imag = vec![0.0; dim];

    // Create 3 singlet pairs: (q0, q1), (q2, q3), (q4, q5)
    // Singlet |Psi-> = (|01> - |10>) / sqrt(2)
    let inv_2_sqrt2 = 1.0 / (2.0 * std::f64::consts::SQRT_2);
    // Expand tensor product of 3 singlets
    for i in 0..dim {
        let b01 = (i >> 0) & 3;
        let b23 = (i >> 2) & 3;
        let b45 = (i >> 4) & 3;
        let mut sign = 1.0;
        let mut valid = true;

        for pair in [b01, b23, b45] {
            if pair == 1 {
                // |01> -> +
            } else if pair == 2 {
                // |10> -> -
                sign = -sign;
            } else {
                valid = false;
                break;
            }
        }
        if valid {
            state_re[i] = sign * inv_2_sqrt2;
        }
    }

    let mut results = Vec::new();
    for q0 in 0..n_qubits {
        for q1 in (q0 + 1)..n_qubits {
            let (hopf_coord, invariants) =
                compute_qubit_pair_hopf_s4(&state_re, &state_imag, q0, q1, n_qubits);
            results.push(PairwiseMolecularEntanglement {
                qubit_pair: (q0, q1),
                hopf_coord,
                invariants,
            });
        }
    }
    results
}

fn main() {
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│   T-1000 MONOGRAPH M157: SECOND HOPF FIBRATION (S7 -> S4) BENCHMARK         │");
    println!("│   256-bit AVX2 Quaternionic State Engine & Entanglement Invariants           │");
    println!("└─────────────────────────────────────────────────────────────────────────────┘");

    let bell_points = benchmark_bell_basis();
    println!("\n[1] Canonical Basis States on S4 Quaternionic Bloch Sphere:");
    println!("┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬──────────┬──────────┐");
    println!("│ State   │ y0      │ y1      │ y2      │ y3      │ y4      │ Concurr. │ Bell-CHSH│");
    println!("├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼──────────┼──────────┤");
    for p in &bell_points {
        println!(
            "│ {:<7} │ {:>7.4} │ {:>7.4} │ {:>7.4} │ {:>7.4} │ {:>7.4} │ {:>8.4} │ {:>8.4} │",
            p.name,
            p.hopf_coord.y0,
            p.hopf_coord.y1,
            p.hopf_coord.y2,
            p.hopf_coord.y3,
            p.hopf_coord.y4,
            p.invariants.concurrence,
            p.invariants.bell_chsh
        );
    }
    println!("└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴──────────┴──────────┘");

    let rot_points = benchmark_rotation_trajectory(20);
    println!("\n[2] Continuous Entanglement Rotation (|psi(theta)> = cos(th)|00> + sin(th)|11>):");
    println!(
        "  Steps: {}, Range: 0.0 deg -> 90.0 deg (Crossing Bell Equator at 45.0 deg)",
        rot_points.len()
    );
    let mid = rot_points
        .iter()
        .find(|p| (p.theta_deg - 45.0).abs() < 1e-3)
        .unwrap();
    println!(
        "  At theta = 45.0 deg: y0 = {:.6}, Concurrence = {:.6}, Bell-CHSH = {:.6}",
        mid.hopf_coord.y0, mid.invariants.concurrence, mid.invariants.bell_chsh
    );

    let werner_points = benchmark_werner_sweep(20);
    println!("\n[3] Werner State Contraction into 5D Quaternionic Bloch Ball B5:");
    let sep_pt = werner_points
        .iter()
        .find(|p| (p.p - 0.35).abs() < 0.03)
        .unwrap();
    let tsir_pt = werner_points
        .iter()
        .find(|p| (p.p - 0.70).abs() < 0.03)
        .unwrap();
    println!(
        "  Separable Threshold (p ~ 0.33): Concurrence = {:.4}, Radius = {:.4}",
        sep_pt.invariants.concurrence, sep_pt.hopf_coord.radius
    );
    println!(
        "  Tsirelson Violation Threshold (p ~ 0.707): Bell-CHSH = {:.4}, Radius = {:.4}",
        tsir_pt.invariants.bell_chsh, tsir_pt.hopf_coord.radius
    );

    let throughput = benchmark_simd_throughput(1_000_000);
    println!("\n[4] AVX2 SIMD Throughput (1,000,000 S7 -> S4 Projections):");
    println!("  Total Time     : {:.2} ms", throughput.elapsed_ms);
    println!(
        "  Latency        : {:.2} ns / projection",
        throughput.latency_ns_per_eval
    );
    println!(
        "  Throughput     : {:.2} Million projections / sec",
        throughput.throughput_m_evals_sec
    );
    println!("  Max S4 Residual: {:.3e}", throughput.max_norm_error);

    let molecular_pairs = benchmark_molecular_pairwise();
    println!("\n[5] 6-Qubit Tri-Singlet Pairwise Entanglement Topology:");
    for pair in &molecular_pairs {
        if pair.invariants.concurrence > 0.01 {
            println!(
                "  Pair (q{}, q{}): Concurrence = {:.4}, Bell-CHSH = {:.4}, Bell Distance = {:.4}",
                pair.qubit_pair.0,
                pair.qubit_pair.1,
                pair.invariants.concurrence,
                pair.invariants.bell_chsh,
                pair.invariants.bell_plane_distance
            );
        }
    }

    let report = HopfS4BenchmarkReport {
        timestamp: "2026-10-20T21:40:00Z".to_string(),
        bell_basis_states: bell_points,
        rotation_sweep: rot_points,
        werner_sweep: werner_points,
        throughput,
        molecular_pairwise: molecular_pairs,
    };

    let json_text = serde_json::to_string_pretty(&report).expect("Failed to serialize report");

    let web_data_path = Path::new("web/data/hopf_s4_entanglement_report.json");
    if let Some(parent) = web_data_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(web_data_path, &json_text).expect("Failed to write web data");
    println!("\n[✓] Telemetry written to: {}", web_data_path.display());

    let results_data_path = Path::new("results/data/hopf_s4_entanglement_report.json");
    if let Some(parent) = results_data_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(results_data_path, &json_text).expect("Failed to write results data");
    println!("[✓] Telemetry written to: {}", results_data_path.display());
}
