// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Deen Toolchain CLI Command Suite
//!
//! Autonomous sovereign tooling suite for Deen OS quantum architectures and T-1000 engine.
//! Provides commands for sidecar overhead benchmarking, shot-batched noise simulation sweeps,
//! multi-architecture chip auditing, and native hardware transpilation.
//!
//! ## Subcommands
//! - `bench-sidecar`: Benchmark asynchronous PauliFrameTrackerSidecar channel overhead vs synchronous inline tracing.
//! - `run-sweep`: Execute shot-batched Monte Carlo noise run under depolarizing noise with live telemetry.
//! - `chip-audit`: Inspect physical specifications, error budgets, and coupling maps for a [`QuantumChip`].
//! - `transpile`: Transpile a parameterized circuit into chip-native basis gates and evaluate physical noise.
//! - `help`: Display help and command usage.

use std::env;
use std::io::Write;
use std::path::Path;

use t1000::architecture::{ChipNoiseSimulator, ChipTranspiler, QuantumChip};
use t1000::compiler::CircuitOp;
use t1000::qec::{benchmark_pauli_tracking_overhead, DeenQubitEngine};

fn print_banner() {
    println!("┌────────────────────────────────────────────────────────────────────────┐");
    println!("│ DEEN TOOLCHAIN: QUANTUM ARCHITECTURE CLI COMMAND SUITE                │");
    println!("│ Multi-Chip Transpilation, Physical Noise Audit & Telemetry Sentry     │");
    println!("└────────────────────────────────────────────────────────────────────────┘");
}

fn print_usage() {
    print_banner();
    println!("Usage:");
    println!("  cargo run --bin deen_toolchain -- [COMMAND] [OPTIONS]");
    println!();
    println!("Commands:");
    println!("  bench-sidecar [ITERATIONS]           Benchmark async sidecar dispatch vs sync tracing (default: 100000)");
    println!("  run-sweep [BATCHES] [QUBITS]         Execute shot-batched noise sweep with live telemetry");
    println!("  chip-audit [CHIP_SLUG]               Audit chip calibration, topology, and native gates (default: all)");
    println!("  transpile [CHIP_SLUG] [QUBITS] [LAYERS] Transpile circuit into chip-native basis with noise audit");
    println!("  help                                 Display this help message");
    println!();
    println!("Available Chips:");
    for chip in QuantumChip::ALL {
        println!("  - {:<14} : {}", chip.slug(), chip.full_name());
    }
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
    let words = (qubits / 64).max(1);

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

fn run_chip_audit(slug: &str) -> Result<(), &'static str> {
    print_banner();

    let chips: Vec<QuantumChip> = if slug.eq_ignore_ascii_case("all") {
        QuantumChip::ALL.to_vec()
    } else {
        vec![QuantumChip::from_slug(slug).ok_or("Invalid chip slug specified")?]
    };

    println!("AUDITING {} QUANTUM PROCESSOR TARGET(S)...", chips.len());
    for chip in chips {
        let err = chip.error_profile();
        let gates = chip.native_gates();
        let gates_str = format!("{:?}", gates);

        println!("\n┌────────────────────────────────────────────────────────────────────────┐");
        println!("│ TARGET CHIP: {:<57} │", chip.full_name());
        println!("├──────────────────────────────┬─────────────────────────────────────────┤");
        println!("│ Hardware Modality            │ {:<39} │", chip.modality().name());
        println!("│ Qubit / Mode Capacity        │ {:<39} │", chip.capacity());
        println!("│ Coupler Links Count          │ {:<39} │", chip.coupler_count());
        println!("│ Native Gate Primitives       │ {:<39} │", gates_str);
        println!("│ Relaxation Time (T1)         │ {:<36.2} us │", err.t1_micros);
        println!("│ Dephasing Time (T2*)         │ {:<36.2} us │", err.t2_micros);
        println!("│ Single-Qubit Error (1-F1)    │ {:<36.4} %  │", err.single_qubit_error * 100.0);
        println!("│ Two-Qubit Error (1-F2)       │ {:<36.4} %  │", err.two_qubit_error * 100.0);
        println!("│ Inter-Chiplet Seam Penalty   │ {:<36.3}x   │", err.inter_chiplet_seam_factor);
        println!("│ Readout Confusion Error      │ {:<36.4} %  │", err.readout_error * 100.0);
        println!("│ Single-Qubit Gate Duration   │ {:<36.2} ns │", err.single_qubit_duration_ns);
        println!("│ Two-Qubit Gate Duration      │ {:<36.2} ns │", err.two_qubit_duration_ns);
        println!("└──────────────────────────────┴─────────────────────────────────────────┘");
    }
    Ok(())
}

fn run_transpile(slug: &str, n_qubits: usize, depth: usize) -> Result<(), &'static str> {
    print_banner();
    let chip = QuantumChip::from_slug(slug).ok_or("Invalid chip slug specified")?;

    if n_qubits > chip.capacity() {
        return Err("Circuit qubits exceed physical chip capacity");
    }

    println!("[DEEN-TOOLCHAIN] Transpiling Circuit: {} Qubits | Depth: {}", n_qubits, depth);
    println!("[DEEN-TOOLCHAIN] Target Architecture: {}", chip.full_name());

    // Synthesize test ansatz circuit
    let mut input_ops = Vec::new();
    for l in 0..depth {
        for q in 0..n_qubits {
            input_ops.push(CircuitOp::Ry { qubit: q, theta: 0.1 * (l + q + 1) as f64 });
            input_ops.push(CircuitOp::Rz { qubit: q, theta: 0.05 * (l + q + 1) as f64 });
        }
        for q in 0..n_qubits {
            let next_q = (q + 1) % n_qubits;
            input_ops.push(CircuitOp::CNOT { control: q, target: next_q });
        }
    }

    let transpiler = ChipTranspiler::new(chip);
    let transpiled = transpiler.transpile(&input_ops).map_err(|_| "Transpilation failed")?;

    let sim = ChipNoiseSimulator::new(chip);
    let mut ideal_dist = vec![0.0; 1 << n_qubits];
    ideal_dist[0] = 1.0;

    let (_, noise_rep) = sim.simulate_noise(
        &ideal_dist,
        n_qubits,
        transpiled.report.single_qubit_gate_count,
        transpiled.report.two_qubit_gate_count,
        0,
    );

    println!("\n┌──────────────────────────────────────────────────────────────┐");
    println!("│             TRANSPILATION & PHYSICAL NOISE REPORT            │");
    println!("├──────────────────────────────┬───────────────────────────────┤");
    println!("│ Input Gate Count             │ {:>29} │", transpiled.report.input_gate_count);
    println!("│ Native Hardware Gate Count   │ {:>29} │", transpiled.report.output_gate_count);
    println!("│ Single-Qubit Native Gates    │ {:>29} │", transpiled.report.single_qubit_gate_count);
    println!("│ Two-Qubit Native Gates       │ {:>29} │", transpiled.report.two_qubit_gate_count);
    println!("│ Routing SWAPs Inserted       │ {:>29} │", transpiled.report.routing_swaps_inserted);
    println!("│ Physical Pulse Duration      │ {:>26.2} ns │", transpiled.report.estimated_duration_ns);
    println!("│ Predicted Gate Fidelity      │ {:>28.4} % │", transpiled.report.predicted_gate_fidelity * 100.0);
    println!("│ T1 Relaxation Fidelity       │ {:>28.4} % │", noise_rep.t1_fidelity * 100.0);
    println!("│ T2* Dephasing Fidelity       │ {:>28.4} % │", noise_rep.t2_fidelity * 100.0);
    println!("│ Overall Composite Fidelity   │ {:>28.4} % │", noise_rep.composite_fidelity * 100.0);
    println!("└──────────────────────────────┴───────────────────────────────┘");

    let results_dir = Path::new("results/experiments");
    std::fs::create_dir_all(results_dir).unwrap();
    let json_file = results_dir.join(format!("deen_transpile_{}.json", chip.slug()));
    let mut file = t1000::results::create(&json_file).unwrap();
    let json_bytes = serde_json::to_string_pretty(&transpiled.report).unwrap();
    file.write_all(json_bytes.as_bytes()).unwrap();

    println!("\n[SUCCESS] Transpilation Report Saved: {:?}", json_file);
    Ok(())
}

fn main() -> Result<(), &'static str> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        return run_bench_sidecar(100_000);
    }

    match args[1].as_str() {
        "bench-sidecar" => {
            let iterations = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100_000);
            run_bench_sidecar(iterations)
        }
        "run-sweep" => {
            let batches = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1_000);
            let qubits = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(256);
            run_sweep(batches, qubits)
        }
        "chip-audit" => {
            let slug = args.get(2).map(|s| s.as_str()).unwrap_or("all");
            run_chip_audit(slug)
        }
        "transpile" => {
            let slug = args.get(2).map(|s| s.as_str()).unwrap_or("nighthawk-r2");
            let qubits = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);
            let depth = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(2);
            run_transpile(slug, qubits, depth)
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
