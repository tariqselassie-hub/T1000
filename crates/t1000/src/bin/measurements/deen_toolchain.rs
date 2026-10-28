// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Deen Toolchain CLI Command Suite
//!
//! Autonomous sovereign tooling suite for Deen OS quantum architectures and T-1000 engine.
//! Provides commands for sidecar overhead benchmarking, shot-batched noise simulation sweeps,
//! and real-time Pauli frame telemetry verification.
//!
//! ## Subcommands
//! - `bench-sidecar`: Benchmark asynchronous PauliFrameTrackerSidecar channel overhead vs synchronous inline tracing.
//! - `run-sweep`: Execute shot-batched Monte Carlo noise run under depolarizing noise with live telemetry.
//! - `help`: Display help and command usage.

use std::env;
use std::io::Write;
use std::path::Path;

use t1000::qec::{benchmark_pauli_tracking_overhead, DeenQubitEngine};

fn print_banner() {
    println!("┌────────────────────────────────────────────────────────────────────────┐");
    println!("│ DEEN TOOLCHAIN: QUANTUM ARCHITECTURE CLI COMMAND SUITE                │");
    println!("│ Pauli Frame Sidecar Benchmarking, Noise Sweeps & Telemetry Sentry     │");
    println!("└────────────────────────────────────────────────────────────────────────┘");
}

fn print_usage() {
    print_banner();
    println!("Usage:");
    println!("  cargo run --bin deen_toolchain -- [COMMAND] [OPTIONS]");
    println!();
    println!("Commands:");
    println!("  bench-sidecar [ITERATIONS]   Benchmark async sidecar dispatch vs sync tracing (default: 100000)");
    println!("  run-sweep [BATCHES] [QUBITS] Execute shot-batched noise sweep with live telemetry");
    println!("  help                         Display this help message");
    println!();
    println!("Examples:");
    println!("  cargo run --release -p t1000 --bin deen_toolchain -- bench-sidecar 100000");
    println!("  cargo run --release -p t1000 --bin deen_toolchain -- run-sweep 1000 256");
}

fn run_bench_sidecar(iterations: usize) -> Result<(), &'static str> {
    print_banner();
    println!("[DEEN-TOOLCHAIN] Mode: Asynchronous Pauli Sidecar vs Sync Tracing");
    println!("[DEEN-TOOLCHAIN] Target Iterations: {}", iterations);
    println!();

    let res = benchmark_pauli_tracking_overhead(iterations);

    println!("\n┌──────────────────────────────────────────────────────────────┐");
    println!("│             DEEN TOOLCHAIN BENCHMARK RESULTS                 │");
    println!("├──────────────────────────────┬───────────────────────────────┤");
    println!("│ Target Iterations            │ {:>29} │", res.iterations);
    println!("│ Qubit Register Width         │ {:>25} Qubits │", res.qubit_count);
    println!("│ Synchronous Total Time       │ {:>26.3} ms │", res.sync_duration_ms);
    println!("│ Synchronous Latency          │ {:>24.2} ns/op │", res.sync_ns_per_op);
    println!("│ Asynchronous Dispatch Time   │ {:>26.3} ms │", res.async_dispatch_duration_ms);
    println!("│ Asynchronous Dispatch Latency│ {:>24.2} ns/op │", res.async_ns_per_op);
    println!("│ Background Frames Processed  │ {:>29} │", res.sidecar_frames_processed);
    println!("│ Hot-Path Dispatch Advantage  │ {:>28.2}x │", res.speedup_dispatch_vs_sync);
    println!("└──────────────────────────────┴───────────────────────────────┘");

    let results_dir = Path::new("results/experiments");
    std::fs::create_dir_all(results_dir).unwrap();
    let json_file = results_dir.join("deen_toolchain_sidecar_bench.json");
    let mut file = t1000::results::create(&json_file).unwrap();
    let json_bytes = serde_json::to_string_pretty(&res).unwrap();
    file.write_all(json_bytes.as_bytes()).unwrap();

    println!("\n[SUCCESS] Deen Toolchain Benchmark Saved: {:?}", json_file);
    Ok(())
}

fn run_sweep(batches: u64, qubits: usize) -> Result<(), &'static str> {
    print_banner();
    println!("[DEEN-TOOLCHAIN] Mode: Shot-Batched Monte Carlo Sweep");
    println!("[DEEN-TOOLCHAIN] Target Batches: {} | Qubits: {}", batches, qubits);

    let mut engine = DeenQubitEngine::new(qubits);
    let words = qubits / 64;
    let words = words.max(1);

    for i in 0..batches {
        let x_masks = vec![0xAAAAAAAAAAAAAAAA_u64; words];
        let z_masks = vec![0x5555555555555555_u64; words];

        engine.execute_shot_batch_with_tracking(x_masks, z_masks, 0.001)?;

        if i > 0 && i % (batches / 5).max(1) == 0 {
            engine.report_telemetry();
        }
    }

    std::thread::sleep(std::time::Duration::from_millis(20));
    let (processed, drift) = engine.get_telemetry();
    println!("\n[DEEN-TOOLCHAIN] Sweep completed successfully.");
    println!(
        "[FINAL TELEMETRY] Batches Processed: {} | Cumulative Pauli Drift Weight: {}",
        processed, drift
    );
    Ok(())
}

fn main() -> Result<(), &'static str> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        // Default to running bench-sidecar with 100,000 iterations
        return run_bench_sidecar(100_000);
    }

    match args[1].as_str() {
        "bench-sidecar" => {
            let iterations = if args.len() >= 3 {
                args[2].parse::<usize>().unwrap_or(100_000)
            } else {
                100_000
            };
            run_bench_sidecar(iterations)
        }
        "run-sweep" => {
            let batches = if args.len() >= 3 {
                args[2].parse::<u64>().unwrap_or(1_000)
            } else {
                1_000
            };
            let qubits = if args.len() >= 4 {
                args[3].parse::<usize>().unwrap_or(256)
            } else {
                256
            };
            run_sweep(batches, qubits)
        }
        "help" | "-h" | "--help" => {
            print_usage();
            Ok(())
        }
        other => {
            eprintln!("[ERROR] Unknown command: {}", other);
            print_usage();
            Err("Unknown deen_toolchain command")
        }
    }
}
