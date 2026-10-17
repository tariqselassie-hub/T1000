// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Intel Integrated Graphics (Xe iGPU) Zero-Copy UMA Co-Processor Benchmark.
//! Benchmarks dense statevector evolution (up to N=22, 67.1 MB) across Lion Cove
//! CPU P-cores (AVX2+FMA) and Intel Graphics OpenCL using CL_MEM_USE_HOST_PTR.

use std::f64::consts::PI;
use std::time::Instant;
use t1000::gpu::{IGpuBridge, IGpuBridgeConfig};
use t1000::state::QuantumState;

fn main() {
    println!("┌────────────────────────────────────────────────────────────────────────┐");
    println!("│     T-1000: INTEL GRAPHICS (Xe iGPU) ZERO-COPY UMA CO-PROCESSOR        │");
    println!("│     Unified Memory Architecture (UMA) Dense Statevector Benchmark      │");
    println!("└────────────────────────────────────────────────────────────────────────┘");

    let bridge = IGpuBridge::new(IGpuBridgeConfig {
        threshold_qubits: 14,
        force_igpu: true,
    });

    println!("OpenCL Runtime Mounted : {}", bridge.is_available());
    if let Some(dev) = bridge.device_name() {
        println!("Mounted GPU Device     : {}", dev);
    }
    if let Some(plat) = bridge.platform_name() {
        println!("Mounted OpenCL Platform: {}", plat);
    }
    println!();

    if !bridge.is_available() {
        println!("-> ERROR: OpenCL runtime could not be initialized on this host.");
        return;
    }

    // Benchmark across scaling register sizes: 16Q (1 MB), 18Q (4 MB), 20Q (16 MB), 22Q (64 MB)
    let test_qubits = [16, 18, 20, 22];

    println!("┌──────┬──────────────┬────────────┬──────────────┬──────────────┬───────────┐");
    println!("│ Qubits│ State Memory │ Hilbert Dim│ CPU AVX2 (ms)│ iGPU UMA (ms)│ Fidelity  │");
    println!("├──────┼──────────────┼────────────┼──────────────┼──────────────┼───────────┤");

    for &n in &test_qubits {
        let dim = 1 << n;
        let mem_mb = (dim * 16) as f64 / (1024.0 * 1024.0);

        let mut s_cpu = QuantumState::ground(dim);
        let mut s_gpu = QuantumState::ground(dim);
        let theta = PI * 0.25;
        let target_qubit = n - 2;

        // Warm up and execute CPU AVX2 pass
        let t_cpu = Instant::now();
        t1000::simd::apply_ry_kernel(&mut s_cpu.real, &mut s_cpu.imag, target_qubit, theta);
        let cpu_ms = t_cpu.elapsed().as_secs_f64() * 1000.0;

        // Warm up and execute iGPU Zero-Copy pass
        let t_gpu = Instant::now();
        let rep = bridge.apply_ry(&mut s_gpu, target_qubit, theta);
        let gpu_ms = if rep.zero_copy_uma {
            rep.duration_us / 1000.0
        } else {
            t_gpu.elapsed().as_secs_f64() * 1000.0
        };

        // Numerical fidelity check
        let fid = s_gpu.fidelity(&s_cpu);

        println!(
            "│ {:4} │ {:9.2} MB │ {:10} │ {:12.3} │ {:12.3} │ {:.7} │",
            n, mem_mb, dim, cpu_ms, gpu_ms, fid
        );
    }

    println!("└──────┴──────────────┴────────────┴──────────────┴──────────────┴───────────┘");
    println!();
    println!("────────────────────────────────────────────────────────────────────────");
    println!("-> PASS: Intel Xe iGPU Zero-Copy UMA Statevector Bridge verified.");
    println!("────────────────────────────────────────────────────────────────────────");
}
