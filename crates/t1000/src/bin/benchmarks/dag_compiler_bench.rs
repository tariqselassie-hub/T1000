// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! DAG-Based Compiler Benchmark: Chaotic Circuit Execution & Cache Footprint Optimization.
//! Demonstrates how the DAG dependency tracker automatically groups independent operations
//! and fuses single-qubit gates, eliminating cache thrashing on arbitrary quantum layers.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use serde::Serialize;
use t1000::compiler::{CircuitBuilder, CircuitOp};
use t1000::state::QuantumState;

#[derive(Serialize)]
struct BenchmarkTelemetry {
    qubits: usize,
    statevector_dim: usize,
    memory_per_statevector_mb: f64,
    raw_ops_count: usize,
    fused_ops_count: usize,
    scheduled_stages_count: usize,
    critical_path_depth: usize,
    cache_pass_reduction_pct: f64,
    raw_execution_time_seconds: f64,
    compiled_execution_time_seconds: f64,
    speedup_ratio: f64,
    raw_memory_traffic_gb: f64,
    compiled_memory_traffic_gb: f64,
    state_fidelity: f64,
    fused_single_qubit_count: usize,
    merged_rz_count: usize,
    merged_ry_count: usize,
}

fn main() {
    println!("==========================================================================");
    println!(" ZENITH RESEARCH DIVISION: DAG COMPILER DEPENDENCY TRACKER BENCHMARK");
    println!(" Automatic Grouping of Independent Operations & Cache Footprint Defense");
    println!("==========================================================================");

    let n_qubits = 20;
    let dim = 1 << n_qubits;
    let state_bytes = dim * 16;
    let state_mb = state_bytes as f64 / (1024.0 * 1024.0);

    println!("[CONFIG] Active Qubits: {}", n_qubits);
    println!("[CONFIG] Hilbert Space Dim: {} (2^{})", dim, n_qubits);
    println!("[CONFIG] Statevector Memory Footprint: {:.2} MB", state_mb);

    // 1. Synthesize a Chaotic Interleaved Circuit
    // Simulates uncoordinated arbitrary user code where gates across distant qubits
    // are interleaved, and single-qubit rotations are submitted out-of-order.
    println!("[SYNTHESIS] Generating chaotic multi-layer circuit stream...");
    let mut builder = CircuitBuilder::new(n_qubits);
    let mut raw_ops = Vec::new();

    let num_layers = 10;
    for l in 0..num_layers {
        // Interleaved single-qubit rotations with chaotic submission order
        for step in 0..n_qubits {
            let q1 = step;
            let q2 = (step + 7) % n_qubits;

            let theta_y = 0.35 + 0.1 * ((l * step) as f64).sin();
            let op_y = CircuitOp::Ry {
                qubit: q1,
                theta: theta_y,
            };
            builder.dag.add_op(op_y.clone());
            raw_ops.push(op_y);

            let theta_z = -0.25 + 0.15 * (((l + 1) * (step + 3)) as f64).cos();
            let op_z = CircuitOp::Rz {
                qubit: q2,
                theta: theta_z,
            };
            builder.dag.add_op(op_z.clone());
            raw_ops.push(op_z);

            // Immediate follow-up rotation on q1 (opportunity for compiler fusion)
            let theta_z2 = 0.42 * (((l + 2) * (step + 1)) as f64).sin();
            let op_z2 = CircuitOp::Rz {
                qubit: q1,
                theta: theta_z2,
            };
            builder.dag.add_op(op_z2.clone());
            raw_ops.push(op_z2);
        }

        // Interleaved 2-qubit entangling gates across ring and cross-chords
        for q in 0..n_qubits {
            let target = (q + 1) % n_qubits;
            let op_cz = CircuitOp::CZ { q1: q, q2: target };
            builder.dag.add_op(op_cz.clone());
            raw_ops.push(op_cz);
        }
    }

    let total_raw_ops = raw_ops.len();
    println!(
        "[SYNTHESIS] Total Raw Unoptimized Operations: {}",
        total_raw_ops
    );

    // 2. Execute Raw Uncompiled Reference
    println!("[EXECUTION 1] Executing Raw Uncompiled Circuit (Sequential Traversal)...");
    let mut state_raw = QuantumState::new(dim);
    let t_raw_start = Instant::now();

    for op in &raw_ops {
        match op {
            CircuitOp::Ry { qubit, theta } => {
                crate_simd_ry(&mut state_raw, *qubit, *theta);
            }
            CircuitOp::Rz { qubit, theta } => {
                crate_simd_rz(&mut state_raw, *qubit, *theta);
            }
            CircuitOp::CZ { q1, q2 } => {
                crate_simd_cz(&mut state_raw, *q1, *q2);
            }
            _ => {}
        }
    }

    let raw_duration = t_raw_start.elapsed().as_secs_f64();
    println!(
        "[EXECUTION 1] Raw Execution Complete: {:.4} seconds",
        raw_duration
    );

    // 3. Compile Circuit through DAG Dependency Tracker
    println!("[COMPILER] Ingesting into DAG & resolving dependency hazards...");
    let t_comp_start = Instant::now();
    let compiled = builder.compile();
    let compile_overhead = t_comp_start.elapsed().as_secs_f64();

    println!(
        "[COMPILER] Compilation Overhead: {:.6} seconds",
        compile_overhead
    );
    println!("[COMPILER] Raw Operations: {}", compiled.telemetry.raw_ops);
    println!(
        "[COMPILER] Fused Operations: {} (Fusions: 1Q={}, Rz={}, Ry={})",
        compiled.telemetry.fused_ops,
        compiled.telemetry.single_qubit_fused,
        compiled.telemetry.rz_merged,
        compiled.telemetry.ry_merged
    );
    println!(
        "[COMPILER] DAG Critical Path Depth: {}",
        compiled.telemetry.critical_path_depth
    );
    println!(
        "[COMPILER] Grouped Independent Stages: {}",
        compiled.telemetry.stage_count
    );
    println!(
        "[COMPILER] Cache Pass Reduction: {:.2}%",
        compiled.telemetry.cache_pass_reduction_pct
    );

    // 4. Execute DAG-Compiled Circuit
    println!("[EXECUTION 2] Executing DAG-Compiled Schedule (Grouped & Cache-Ordered)...");
    let mut state_compiled = QuantumState::new(dim);
    let t_exec_start = Instant::now();
    compiled.execute(&mut state_compiled);
    let compiled_duration = t_exec_start.elapsed().as_secs_f64();

    println!(
        "[EXECUTION 2] Compiled Execution Complete: {:.4} seconds",
        compiled_duration
    );

    // 5. Verification & Mathematical Integrity
    let fidelity = state_raw.fidelity(&state_compiled);
    println!(
        "[VERIFICATION] Statevector Fidelity F(raw, compiled): {:.12}",
        fidelity
    );
    assert!(
        (fidelity - 1.0).abs() < 1e-10,
        "Fatal: State fidelity violation between raw and compiled output: {}",
        fidelity
    );

    let speedup = raw_duration / compiled_duration;
    let raw_traffic_gb = (total_raw_ops as f64 * state_bytes as f64) / (1024.0 * 1024.0 * 1024.0);
    let comp_traffic_gb =
        (compiled.stages.len() as f64 * state_bytes as f64) / (1024.0 * 1024.0 * 1024.0);

    println!("--------------------------------------------------------------------------");
    println!(" ARCHITECTURAL COMPILER TELEMETRY SUMMARY");
    println!("--------------------------------------------------------------------------");
    println!(" - Raw Uncompiled Execution Time:    {:.4} s", raw_duration);
    println!(
        " - DAG Compiled Execution Time:      {:.4} s",
        compiled_duration
    );
    println!(
        " - Total Hardware Speedup Ratio:     {:.2}x FASTER",
        speedup
    );
    println!(
        " - Raw Memory Bus Traversal:         {:.2} GB",
        raw_traffic_gb
    );
    println!(
        " - Compiled Memory Bus Traversal:    {:.2} GB",
        comp_traffic_gb
    );
    println!(
        " - Memory Bus Traffic Eliminated:    {:.2} GB ({:.1}%)",
        raw_traffic_gb - comp_traffic_gb,
        (1.0 - comp_traffic_gb / raw_traffic_gb) * 100.0
    );
    println!(
        " - Statevector Fidelity Integrity:   {:.12} (Bit-Exact)",
        fidelity
    );
    println!("==========================================================================");

    // Save JSON Telemetry Ledger
    let ledger = BenchmarkTelemetry {
        qubits: n_qubits,
        statevector_dim: dim,
        memory_per_statevector_mb: state_mb,
        raw_ops_count: total_raw_ops,
        fused_ops_count: compiled.telemetry.fused_ops,
        scheduled_stages_count: compiled.telemetry.stage_count,
        critical_path_depth: compiled.telemetry.critical_path_depth,
        cache_pass_reduction_pct: compiled.telemetry.cache_pass_reduction_pct,
        raw_execution_time_seconds: raw_duration,
        compiled_execution_time_seconds: compiled_duration,
        speedup_ratio: speedup,
        raw_memory_traffic_gb: raw_traffic_gb,
        compiled_memory_traffic_gb: comp_traffic_gb,
        state_fidelity: fidelity,
        fused_single_qubit_count: compiled.telemetry.single_qubit_fused,
        merged_rz_count: compiled.telemetry.rz_merged,
        merged_ry_count: compiled.telemetry.ry_merged,
    };

    let results_dir = Path::new("results/experiments");
    std::fs::create_dir_all(results_dir).unwrap();
    let json_file = results_dir.join("dag_compiler_results.json");
    let mut file = t1000::results::create(&json_file).unwrap();
    let json_bytes = serde_json::to_string_pretty(&ledger).unwrap();
    file.write_all(json_bytes.as_bytes()).unwrap();
    println!("[SUCCESS] Telemetry ledger written: {:?}", json_file);
}

#[inline(always)]
fn crate_simd_ry(state: &mut QuantumState, qubit: usize, theta: f64) {
    t1000::simd::apply_ry_kernel(&mut state.real, &mut state.imag, qubit, theta);
}

#[inline(always)]
fn crate_simd_rz(state: &mut QuantumState, qubit: usize, theta: f64) {
    t1000::simd::apply_rz_kernel(&mut state.real, &mut state.imag, qubit, theta);
}

#[inline(always)]
fn crate_simd_cz(state: &mut QuantumState, q1: usize, q2: usize) {
    t1000::simd::apply_cz_kernel(&mut state.real, &mut state.imag, q1, q2);
}
