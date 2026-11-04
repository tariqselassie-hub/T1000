// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Traversable Holographic Wormhole Protocol (ER = EPR) Benchmark
//!
//! Evaluates the Gao-Jafferis-Wall negative-energy wormhole opening across boundary modes,
//! measuring state refocusing fidelity, gravitational sign asymmetry, and quantum chaos OTOC decay.

use std::path::PathBuf;
use t1000::results;
use t1000::wormhole::{
    HolographicWormholeEngine, ScramblerType, WormholeConfig, WormholeSweepReport,
};

fn parse_args() -> (WormholeConfig, f64, f64, usize, PathBuf) {
    let mut config = WormholeConfig::default();
    let mut g_min = -3.0;
    let mut g_max = 3.0;
    let mut num_points = 13;
    let mut out_path = PathBuf::from("results/experiments/holographic_wormhole.json");

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--modes" if i + 1 < args.len() => {
                config.num_modes = args[i + 1].parse().unwrap_or(3);
                i += 2;
            }
            "--scrambler" if i + 1 < args.len() => {
                config.scrambler_type = match args[i + 1].to_lowercase().as_str() {
                    "syk" | "majorana" => ScramblerType::MajoranaSyk,
                    "floquet" | "haar" => ScramblerType::FloquetHaar,
                    _ => ScramblerType::ChaoticHeisenberg,
                };
                i += 2;
            }
            "--steps" if i + 1 < args.len() => {
                config.trotter_steps = args[i + 1].parse().unwrap_or(3);
                i += 2;
            }
            "--time" if i + 1 < args.len() => {
                config.scrambling_time = args[i + 1].parse().unwrap_or(1.5);
                i += 2;
            }
            "--points" if i + 1 < args.len() => {
                num_points = args[i + 1].parse().unwrap_or(13);
                i += 2;
            }
            "--g-min" if i + 1 < args.len() => {
                g_min = args[i + 1].parse().unwrap_or(-3.0);
                i += 2;
            }
            "--g-max" if i + 1 < args.len() => {
                g_max = args[i + 1].parse().unwrap_or(3.0);
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

    (config, g_min, g_max, num_points, out_path)
}

fn print_matrix_table(report: &WormholeSweepReport) {
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("║            HOLOGRAPHIC WORMHOLE TELEPORTATION BENCHMARK (ER = EPR)          ║");
    println!("║            Gao-Jafferis-Wall Protocol & Negative Energy Shockwave           ║");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ Modes per Black Hole:     {:<49} │", report.num_modes);
    println!("│ Total Simulation Qubits:  {:<49} │", format!("{} (Hilbert Dim: {})", report.total_qubits, 1 << report.total_qubits));
    println!("│ Scrambling Time Duration: {:<49} │", format!("{:.3} s", report.scrambling_time));
    println!("│ Uncoupled Noise Floor:    {:<49} │", format!("{:.4}", report.uncoupled_noise_floor));
    println!("│ Optimal Shockwave Flux:   {:<49} │", format!("{:+7.4}", report.optimal_g));
    println!("│ Peak Process Fidelity:    {:<49} │", format!("{:.4}", report.peak_fidelity));
    println!("│ Max Gravitational Asym:   {:<49} │", format!("{:+7.4}", report.peak_asymmetry));
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();
    println!("┌────────────┬──────────────────────────┬─────────────────┬─────────────────┬──────────┐");
    println!("│ Coupling g │ Dual Bulk Energy Regimes │ Bell Fidelity   │ Asymmetry Delta │ OTOC(t)  │");
    println!("├────────────┼──────────────────────────┼─────────────────┼─────────────────┼──────────┤");

    for pt in &report.points {
        let regime = if pt.g > 1e-6 {
            "ANEC Violation (Negative)"
        } else if pt.g < -1e-6 {
            "Contractive (Positive)  "
        } else {
            "Causally Disconnected   "
        };
        println!(
            "│ {:+10.4} │ {:<24} │ {:15.4} │ {:+15.4} │ {:8.4} │",
            pt.g, regime, pt.fidelity, pt.asymmetry, pt.otoc
        );
    }
    println!("└────────────┴──────────────────────────┴─────────────────┴─────────────────┴──────────┘");
}

fn main() {
    let (config, g_min, g_max, num_points, out_path) = parse_args();
    let engine = HolographicWormholeEngine::new(config);

    let report = engine.sweep_coupling(g_min, g_max, num_points);
    print_matrix_table(&report);

    let serialized = serde_json::to_string_pretty(&report).expect("Serialization failed");
    results::write(&out_path, &serialized).expect("Failed to write benchmark results");
    println!("\nTelemetry successfully recorded to: {}", out_path.display());
}
