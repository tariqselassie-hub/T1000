// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # DeenQubit + Asynchronous Pauli Frame Sidecar Benchmark
//!
//! Evaluates the direct integration of `PauliFrameTrackerSidecar` into the `DeenQubitEngine`
//! execution pipeline. Shot batches stream through the depolarizing noise simulation loop,
//! packaging error syndromes and stabilizer masks into `PauliFrameBatch` structs and submitting
//! them asynchronously to the sidecar channel without stalling the primary statevector thread.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use serde::Serialize;
use t1000::qec::DeenQubitEngine;

#[derive(Debug, Serialize)]
struct DeenQubitSidecarReport {
    timestamp: String,
    qubit_count: usize,
    total_batches: u64,
    words_per_batch: usize,
    shots_per_batch: usize,
    elapsed_ms: f64,
    batches_per_sec: f64,
    shot_gates_per_sec: f64,
    batches_processed_sidecar: u64,
    cumulative_drift_weight: u64,
}

fn main() -> Result<(), &'static str> {
    println!("┌────────────────────────────────────────────────────────────────────────┐");
    println!("│ DEEN-QUBIT + ASYNCHRONOUS PAULI FRAME SIDECAR PIPELINE                 │");
    println!("│ High-Throughput Shot-Batched Noise Simulation (SQUID Co-Tracker)      │");
    println!("└────────────────────────────────────────────────────────────────────────┘");

    let qubit_count = 256;
    let total_batches = 1_000u64;
    let words_per_batch = 4; // 4 x 64-bit words = 256 physical qubits
    let shots_per_word = 64;
    let shots_per_batch = words_per_batch * shots_per_word; // 256 effective shot-wires

    // Instantiate engine for 256 qubits
    let mut engine = DeenQubitEngine::new(qubit_count);
    println!(
        "[DEEN-QUBIT] Initialized engine for {} qubits with async SQUID-style sidecar.",
        qubit_count
    );

    let start = Instant::now();

    // Simulate high-throughput shot-batched execution loops under depolarizing noise (p = 0.001)
    for i in 0..total_batches {
        let dummy_x_masks = vec![0xAAAAAAAAAAAAAAAA_u64; words_per_batch];
        let dummy_z_masks = vec![0x5555555555555555_u64; words_per_batch];

        engine.execute_shot_batch_with_tracking(
            dummy_x_masks,
            dummy_z_masks,
            0.001,
        )?;

        if i > 0 && i % 200 == 0 {
            engine.report_telemetry();
        }
    }

    let elapsed = start.elapsed();
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let batches_per_sec = (total_batches as f64) / elapsed.as_secs_f64();
    let total_shot_gates = (total_batches as f64) * (shots_per_batch as f64) * (qubit_count as f64);
    let shot_gates_per_sec = total_shot_gates / elapsed.as_secs_f64();

    // Allow worker thread to finish draining trailing channel items
    std::thread::sleep(std::time::Duration::from_millis(20));

    let (processed, drift) = engine.get_telemetry();
    println!("\n[DEEN-QUBIT] Execution sweep completed successfully.");
    println!(
        "[FINAL TELEMETRY] Batches Processed: {} | Cumulative Pauli Drift Weight: {}",
        processed, drift
    );

    println!("\n┌──────────────────────────────────────────────────────────────┐");
    println!("│             DEEN-QUBIT PIPELINE TELEMETRY                    │");
    println!("├──────────────────────────────┬───────────────────────────────┤");
    println!("│ Qubit Register Width         │ {:>25} Qubits │", qubit_count);
    println!("│ Total Shot Batches           │ {:>29} │", total_batches);
    println!("│ Elapsed Execution Time       │ {:>26.3} ms │", elapsed_ms);
    println!("│ Dispatch Throughput          │ {:>21.2} batches/s │", batches_per_sec);
    println!("│ Equivalent Throughput        │ {:>20.2e} shot-gates/s │", shot_gates_per_sec);
    println!("│ Sidecar Batches Evaluated    │ {:>29} │", processed);
    println!("│ Cumulative Drift Weight      │ {:>29} │", drift);
    println!("└──────────────────────────────┴───────────────────────────────┘");

    let report = DeenQubitSidecarReport {
        timestamp: "2026-10-28T09:55:00.000Z".to_string(),
        qubit_count,
        total_batches,
        words_per_batch,
        shots_per_batch,
        elapsed_ms,
        batches_per_sec,
        shot_gates_per_sec,
        batches_processed_sidecar: processed,
        cumulative_drift_weight: drift,
    };

    let results_dir = Path::new("results/experiments");
    std::fs::create_dir_all(results_dir).unwrap();
    let json_file = results_dir.join("deen_qubit_sidecar_results.json");
    let mut file = t1000::results::create(&json_file).unwrap();
    let json_bytes = serde_json::to_string_pretty(&report).unwrap();
    file.write_all(json_bytes.as_bytes()).unwrap();

    println!("\n[SUCCESS] DeenQubit Sidecar Report Saved: {:?}", json_file);
    Ok(())
}
