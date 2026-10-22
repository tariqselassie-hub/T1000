// SPDX-License-Identifier: MIT OR Apache-2.0
//! Sweeps register width through the real compiled-circuit executor on the exact CPU
//! backend and the FP32 iGPU backend, and shows what the Fast policy would pick.
//! Usage: backend_auto_bench [min_q] [max_q] [depth]

use t1000::backend::{BackendKind, BackendPolicy, BackendSelector, Precision};
use t1000::compiler::CircuitBuilder;
use t1000::state::QuantumState;

fn circuit(n: usize, depth: usize) -> t1000::compiler::CompiledCircuit {
    let mut b = CircuitBuilder::new(n);
    for l in 0..depth {
        for q in 0..n {
            b.ry(q, 0.37 * (1 + l + q) as f64).rz(q, 0.21 * (3 + l * 7 + q) as f64);
        }
        for q in 0..n {
            b.cz(q, (q + 1) % n);
        }
    }
    b.compile()
}

fn main() {
    let a: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    let (lo, hi, depth) = (*a.first().unwrap_or(&12), *a.get(1).unwrap_or(&26), *a.get(2).unwrap_or(&12));

    let cpu = BackendSelector::new(BackendPolicy::default());
    let gpu = BackendSelector::new(BackendPolicy { force: Some(BackendKind::IGpuFp32), ..BackendPolicy::fast() });
    let auto = BackendPolicy { precision: Precision::Fast, ..BackendPolicy::from_env() };
    println!("GPU: {} | depth {depth}\n", gpu.gpu_device().unwrap_or("none"));
    println!("{:>3} {:>9} {:>10} {:>10} {:>8} {:>10} {:>9}", "N", "state_MB", "cpu64_ms", "gpu32_ms", "speedup", "max|diff|", "auto→");

    for n in (lo..=hi).step_by(2) {
        let c = circuit(n, depth);
        let mut s_cpu = QuantumState::ground(1 << n);
        let mut s_gpu = s_cpu.clone();
        let rc = cpu.execute(&c, &mut s_cpu);
        let rg = gpu.execute(&c, &mut s_gpu);
        let diff = s_cpu.real.iter().zip(&s_gpu.real).chain(s_cpu.imag.iter().zip(&s_gpu.imag))
            .map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        let pick = match auto.choose(n) { BackendKind::IGpuFp32 => "gpu", _ => "cpu" };
        let note = if rg.used != BackendKind::IGpuFp32 { " (GPU fell back)" } else { "" };
        println!("{n:>3} {:>9} {:>10.1} {:>10.1} {:>7.2}x {diff:>10.1e} {pick:>9}{note}",
            (1usize << n) * 16 / (1 << 20), rc.duration_ms, rg.duration_ms, rc.duration_ms / rg.duration_ms);
    }
}
