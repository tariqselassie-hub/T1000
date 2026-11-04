// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! 8-Node Gyroscopic Throwing-Jack & Wobbling Pulsar Stabilizer Benchmark
//!
//! Simulates Poinsot gyroscopic mechanics, mirrored precessional torque cancellation,
//! and decoherence-free quantum phase-locking on the 8-node throwing jack topology.

use std::path::PathBuf;
use t1000::gyroscopic::{GyroscopicConfig, GyroscopicStabilizerEngine};
use t1000::results;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let record = args.iter().any(|a| a == "--record" || a == "-r")
        || std::env::var("T1000_RECORD").map(|v| v == "1").unwrap_or(false);

    println!("[1/3] Initializing 8-Node Throwing-Jack & Wobbling Pulsar Engine...");
    let config = GyroscopicConfig {
        arm_length: 1.5,
        precession_freq_rad_s: 120.0,
        noise_freq_rad_s: 45.0,
        noise_amplitude_n: 2.5,
        quantum_noise_amplitude_rad: 1.25,
        core_coupling_strength_g: 0.85,
        total_time_s: 1.0,
        steps: 100,
    };

    println!("[2/3] Simulating 100 precessional steps with dynamic torque cancellation...");
    let engine = GyroscopicStabilizerEngine::new();
    let report = engine.simulate(&config);

    println!("[3/3] Compiling gyroscopic stability telemetry report...\n");

    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("║       8-NODE GYROSCOPIC THROWING-JACK & PULSAR STABILIZER ENGINE            ║");
    println!("║       Torque Cancellation, Center of Mass Invariance & DFS Phase-Lock       ║");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ Topology:                       8-Node Dual-Cone Hourglass / Throwing-Jack  │");
    println!("│ Center Core Fulcrum:            Node (4, 5) Shared Invariant Barycenter     │");
    println!("│ Polar Precession Velocity:      {:.2} rad/s                                 │", config.precession_freq_rad_s);
    println!("│ Polar Noise Amplitude:          {:.2} N (Kinetic Flutter)                   │", config.noise_amplitude_n);
    println!("│ Polar Noise Frequency:          {:.2} rad/s                                 │", config.noise_freq_rad_s);
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ STABILIZATION GROUND TRUTH:                                                 │");
    println!("│ Mean Torque Cancellation:       {:.6} %                                     │", report.mean_torque_cancellation_ratio * 100.0);
    println!("│ Max Residual Torque ||tau_net||: {:.2e} N*m                                  │", report.max_torque_net_norm);
    println!("│ Max Center of Mass Drift:       {:.2e} m                                    │", report.max_center_of_mass_drift);
    println!("│ Minimum Core Fidelity F_(4,5):  {:.8}                                       │", report.min_core_fidelity);
    println!("│ Mean Parity Correlator <Z_A Z_B>: {:.6}                                     │", report.mean_parity_correlator);
    println!("│ Mean Gyroscopic Q-Factor:       {:.2e}                                      │", report.mean_gyroscopic_q_factor);
    println!("└─────────────────────────────────────────────────────────────────────────────┘");

    let serialized = serde_json::to_string_pretty(&report).expect("Serialization failed");

    if record {
        let out_path = PathBuf::from("results/experiments/gyroscopic_stabilizer.json");
        results::write(&out_path, &serialized).expect("Failed to write results");
        println!("\n[RECORDED] Telemetry saved to {}", out_path.display());
    } else {
        println!("\n[SCRATCH] Run without --record; scratch run complete.");
    }
}
