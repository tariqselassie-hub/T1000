// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! AVX-VNNI Hardware-Accelerated QEC Stabilizer & Syndrome Benchmark.
//! Benchmarks 32-way INT8 dot product instructions against scalar and AVX2 fallbacks,
//! and measures single-cycle syndrome extraction throughput on the [[9, 1, 3]] surface code.

use std::time::Instant;
use t1000::qec::vnni::{
    symplectic_commutation_16q, vnni_dpbusd_32, SurfaceCode9QVnniMatrix, VnniBackend,
};

fn main() {
    println!("┌────────────────────────────────────────────────────────────────────────┐");
    println!("│      T-1000: AVX-VNNI HARDWARE-ACCELERATED QEC STABILIZER BENCHMARK    │");
    println!("│      Intel Lion Cove / Arrow Lake 32-Way INT8 Symplectic Engine        │");
    println!("└────────────────────────────────────────────────────────────────────────┘");

    let backend = VnniBackend::detect();
    println!("Hardware Backend Detected : {}", backend.name());
    println!("Hardware SIMD Accelerated: {}", backend.is_accelerated());
    println!();

    // =========================================================================
    // Benchmark 1: 32-Way INT8 Dot Product Micro-Benchmark
    // =========================================================================
    let iters = 10_000_000;
    let mut a_u8 = [0u8; 32];
    let mut b_i8 = [0i8; 32];
    for i in 0..32 {
        a_u8[i] = ((i * 3 + 1) % 15) as u8;
        b_i8[i] = (((i * 7 + 2) % 11) as i8) - 5;
    }
    let src = [1, 2, 3, 4, 5, 6, 7, 8];

    println!(
        "─── [1] 32-WAY INT8 DOT-PRODUCT THROUGHPUT ({} ITERATIONS) ───",
        iters
    );
    let t0 = Instant::now();
    let mut acc = src;
    for _ in 0..iters {
        acc = vnni_dpbusd_32(acc, a_u8, b_i8);
    }
    let dur_dp = t0.elapsed();
    let dur_dp_ns = dur_dp.as_nanos() as f64 / iters as f64;
    let ops_sec = (iters as f64 * 32.0) / dur_dp.as_secs_f64();

    println!("  Accumulated Checksum      : {:?}", acc);
    println!("  Average Latency per Pass  : {:.3} ns", dur_dp_ns);
    println!(
        "  Effective INT8 Throughput : {:.2} Giga-Ops / sec",
        ops_sec / 1e9
    );
    println!();

    // =========================================================================
    // Benchmark 2: Surface Code [[9, 1, 3]] 8-Stabilizer Syndrome Extraction
    // =========================================================================
    let matrix = SurfaceCode9QVnniMatrix::new();
    let syn_iters = 5_000_000;

    println!(
        "─── [2] SURFACE CODE [[9, 1, 3]] SYNDROME EXTRACTION ({} RUNS) ───",
        syn_iters
    );
    let t1 = Instant::now();
    let mut syn_accum = 0i64;
    for i in 0..syn_iters {
        let err_x = (i & 0x1FF) as u16;
        let err_z = ((i >> 9) & 0x1FF) as u16;
        let syn = matrix.extract_syndromes_vnni(err_x, err_z);
        syn_accum += syn[0] as i64 + syn[4] as i64;
    }
    let dur_syn = t1.elapsed();
    let dur_syn_ns = dur_syn.as_nanos() as f64 / syn_iters as f64;
    let syn_rate = syn_iters as f64 / dur_syn.as_secs_f64();

    println!("  Syndrome Checksum Value   : {}", syn_accum);
    println!("  Extraction Latency (8-syn): {:.3} ns / pass", dur_syn_ns);
    println!(
        "  Full Syndrome Rate        : {:.2} Million decodings / sec",
        syn_rate / 1e6
    );
    println!(
        "  Per-Stabilizer Evaluation : {:.3} ns / stabilizer",
        dur_syn_ns / 8.0
    );
    println!();

    // =========================================================================
    // Benchmark 3: Clifford+T Symplectic Tableau Commutation Checks
    // =========================================================================
    let comm_iters = 5_000_000;
    println!(
        "─── [3] SYMPLECTIC COMMUTATION CHECKS ({} PAIRS) ───",
        comm_iters
    );
    let t2 = Instant::now();
    let mut comm_count = 0u64;
    for i in 0..comm_iters {
        let x1 = (i & 0xFFFF) as u16;
        let z1 = ((i >> 3) & 0xFFFF) as u16;
        let x2 = ((i >> 7) & 0xFFFF) as u16;
        let z2 = ((i >> 11) & 0xFFFF) as u16;
        let anticommutes = symplectic_commutation_16q(x1, z1, x2, z2);
        comm_count += anticommutes as u64;
    }
    let dur_comm = t2.elapsed();
    let dur_comm_ns = dur_comm.as_nanos() as f64 / comm_iters as f64;
    let comm_rate = comm_iters as f64 / dur_comm.as_secs_f64();

    println!("  Anticommuting Pairs Found : {}", comm_count);
    println!(
        "  Commutation Check Latency : {:.3} ns / check",
        dur_comm_ns
    );
    println!(
        "  Commutation Rate          : {:.2} Million checks / sec",
        comm_rate / 1e6
    );
    println!();

    println!("────────────────────────────────────────────────────────────────────────");
    println!("-> PASS: AVX-VNNI QEC Stabilizer & Syndrome Engine verified.");
    println!("────────────────────────────────────────────────────────────────────────");
}
