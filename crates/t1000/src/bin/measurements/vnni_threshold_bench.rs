// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! AVX-VNNI Real-Time QEC Threshold Engine Benchmark.
//!
//! Executes large-scale Monte Carlo physical error injection and single-cycle
//! VNNI syndrome decoding across the [[9, 1, 3]] planar surface code to measure
//! the empirical fault-tolerant threshold and sub-threshold quadratic suppression.

use std::fs;
use std::path::Path;
use t1000::qec::vnni::{VnniBackend, VnniQecDecoder};

fn main() {
    println!("┌────────────────────────────────────────────────────────────────────────┐");
    println!("│      T-1000: AVX-VNNI REAL-TIME QEC THRESHOLD ENGINE BENCHMARK         │");
    println!("│      Closed-Loop Physical Noise Injection & Surface Code Decoding      │");
    println!("└────────────────────────────────────────────────────────────────────────┘");

    let backend = VnniBackend::detect();
    println!("  Hardware Acceleration : {}", backend.name());
    println!("  Vector Engine Active  : {}", backend.is_accelerated());
    println!();

    let decoder = VnniQecDecoder::new();

    // Sweep across 14 noise rates spanning sub-threshold to super-threshold
    let p_values = [
        0.001, 0.002, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.075, 0.100, 0.125,
        0.150,
    ];
    let shots_per_point = 250_000;
    let seed = 0xDEED_2026_0ECFu64;

    println!(
        "Executing Monte Carlo sweep ({} points, {} shots/pt, {} total shots)...",
        p_values.len(),
        shots_per_point,
        p_values.len() * shots_per_point
    );
    println!();

    let report = decoder.monte_carlo_threshold_sweep(&p_values, shots_per_point, seed);

    println!("┌──────────┬──────────┬──────────┬──────────────┬──────────────┬─────────────┐");
    println!("│  p_phys  │  Shots   │ Log.Fail │     P_L      │  Suppression │ Latency/Shot│");
    println!("├──────────┼──────────┼──────────┼──────────────┼──────────────┼─────────────┤");

    for pt in &report.points {
        let status = if pt.logical_error_rate < pt.p_phys {
            format!("{:>8.2}x", pt.suppression_ratio)
        } else {
            format!("{:>7.2}x *", pt.suppression_ratio)
        };

        println!(
            "│  {:>5.3}   │ {:>8} │ {:>8} │ {:>12.6} │ {:>12} │ {:>7.2} ns  │",
            pt.p_phys,
            pt.shots,
            pt.logical_errors,
            pt.logical_error_rate,
            status,
            pt.latency_ns_per_shot
        );
    }
    println!("└──────────┴──────────┴──────────┴──────────────┴──────────────┴─────────────┘");
    println!("* Super-threshold regime: logical error exceeds physical unencoded error.");
    println!();

    // Summary Metrics
    println!("══════════════════════════════════════════════════════════════════════════");
    println!(" EMPIRICAL THRESHOLD & HARDWARE TELEMETRY SUMMARY");
    println!("══════════════════════════════════════════════════════════════════════════");
    if let Some(pth) = report.empirical_threshold {
        println!(
            "  Empirical Fault-Tolerant Threshold (p_th) : {:.4} ({:.2}%)",
            pth,
            pth * 100.0
        );
    } else {
        println!("  Empirical Fault-Tolerant Threshold (p_th) : > 0.150 (Unbounded)");
    }
    println!(
        "  Quadratic Scaling Pre-factor (A in A*p^2) : {:.2}",
        report.fit_scaling_coefficient
    );
    println!(
        "  Total Monte Carlo Shots Evaluated         : {}",
        report.total_shots
    );
    println!(
        "  Total Wall-Clock Duration                 : {:.3} s",
        report.total_duration_seconds
    );
    println!(
        "  Decoding Pipeline Throughput              : {:.2} Million shots / sec",
        report.throughput_mshots_sec
    );
    println!("══════════════════════════════════════════════════════════════════════════");
    println!();

    // Persist JSON artifact
    let out_dir = Path::new("results/data");
    if let Err(e) = fs::create_dir_all(out_dir) {
        eprintln!("Warning: Failed to create results/data directory: {e}");
    }
    let out_path = out_dir.join("vnni_qec_threshold_report.json");
    match serde_json::to_string_pretty(&report) {
        Ok(json_str) => {
            if let Err(e) = t1000::results::write(&out_path, json_str) {
                eprintln!("Warning: Failed to write {}: {e}", out_path.display());
            } else {
                println!("[SAVED] {}", out_path.display());
            }
        }
        Err(e) => eprintln!("Warning: Failed to serialize report: {e}"),
    }
}
