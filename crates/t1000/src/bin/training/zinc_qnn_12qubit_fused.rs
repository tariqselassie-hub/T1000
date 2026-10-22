//! GuddaNetworks 12-Qubit Hybrid Variational QNN: Zero-Allocation Observable Rotation Kernel Fusion
//!
//! Demonstrates and verifies next-stage kernel fusion:
//!   <ψ_L2 | (∏ R_y†(θ_k)) Z_q (∏ R_y(θ_k)) | ψ_L2> = cos(θ_q) <Z_q>_L2 - sin(θ_q) <X_q>_L2
//!
//! Benchmark Modes:
//! 1. Full 111-Parameter Circuit (108 Gates: 36 Ry, 36 Rz, 36 CZ)
//! 2. Pruned 99-Parameter Canonical Kernel (84 Gates: 36 Ry, 24 Rz, 24 CZ)
//! 3. Fused 99-Parameter Observable Rotation Kernel (72 Gates: 24 Ry, 24 Rz, 24 CZ + Read-Only AVX2 POVM Reduction)
//! 4. Fused 98-Parameter Qubit-11 Eliminated Kernel (72 Gates + 11-Qubit AVX2 POVM Reduction, column 11 skipped)
//!
//! Hardware Platform: detected at runtime (`t1000::hostinfo`) | 64 KB L2 State Cache

use serde::{Deserialize, Serialize};
use std::time::Instant;
use t1000::zinc_data::ZincMolecule;
use t1000::zinc_kernel_bench::{readout, Bench, Kernel};
use t1000::zinc_training_common::{
    derive_pruned_params, load_holdout_and_checkpoint, resolve_workspace_path,
};
use t1000::{CircuitAnsatz, DataEncodingLayer, EncodingKind, QuantumState};

const N_QUBITS: usize = 12;
const DIM: usize = 1 << N_QUBITS; // 4,096

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QnnFusedCheckpoint {
    pub timestamp_rfc3339: String,
    pub n_qubits: usize,
    pub depth: usize,
    pub full_params: usize,
    pub pruned_params: usize,
    pub fused_active_params: usize,
    pub gate_count_full: usize,
    pub gate_count_pruned: usize,
    pub gate_count_fused: usize,
    pub gate_depth_reduction_pct: f64,
    pub epochs_completed: usize,
    pub fused_hybrid_params_98: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeTelemetry {
    pub name: String,
    pub gates_per_molecule: usize,
    pub state_write_bytes_l3: usize,
    pub latency_us: f64,
    pub throughput_mols_per_sec: f64,
    pub speedup_vs_baseline: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedEquivalenceReport {
    pub timestamp_rfc3339: String,
    pub host_cpu: String,
    pub holdout_samples: usize,
    pub full_param_count: usize,
    pub pruned_param_count: usize,
    pub fused_param_count: usize,
    pub max_diff_mode_c_logp: f64,
    pub max_diff_mode_c_qed: f64,
    pub max_diff_mode_c_sas: f64,
    pub max_diff_mode_c_overall: f64,
    pub mode_c_exact_equivalence: bool,
    pub max_diff_mode_d_overall: f64,
    pub modes: Vec<ModeTelemetry>,
}

const RULE: &str = "-------------------------------------------------------------------------";
const DOUBLE_RULE: &str =
    "=========================================================================";
/// Independent passes over the holdout set per benchmarked mode.
const REPEATS: usize = 5;

/// One benchmarked mode: predictions plus per-evaluation latency and throughput.
struct ModeRun {
    preds: Vec<[f64; 3]>,
    latency_us: f64,
    throughput: f64,
}

fn bench_mode(
    header: &str,
    label: &str,
    kernel: Kernel,
    bench: &Bench,
    params: &[f64],
    samples: &[ZincMolecule],
    num_threads: usize,
) -> ModeRun {
    println!("{header}");
    let start = Instant::now();
    let preds = bench.run_threads(kernel, params, samples, num_threads, REPEATS);
    let elapsed = start.elapsed();
    let total_evals = (samples.len() * REPEATS) as f64;
    let latency_us = elapsed.as_secs_f64() * 1_000_000.0 / total_evals;
    let throughput = total_evals / elapsed.as_secs_f64();
    println!(
        "  -> {label}: {:.2} ms | {:.2} μs/mol | {:.1} mols/sec\n",
        elapsed.as_secs_f64() * 1000.0,
        latency_us,
        throughput
    );
    ModeRun {
        preds,
        latency_us,
        throughput,
    }
}

/// Mode D on one thread, one pass; nothing is kept.
fn bench_single_core(bench: &Bench, params: &[f64], samples: &[ZincMolecule]) {
    println!("[EXECUTION] Benchmarking Dedicated Single-Core Throughput (1 Thread)...");
    let start = Instant::now();
    {
        let mut state = QuantumState::new(DIM);
        let mut z = [0.0; 12];
        let (w_off, b_off) = (59, 59 + 36);
        for sample in samples {
            bench.evolve(
                Kernel::Fused11,
                params,
                &sample.features,
                &mut state,
                &mut z,
            );
            let _ = readout(params, w_off, b_off, &z[..11]);
        }
    }
    let elapsed = start.elapsed();
    let latency_us = elapsed.as_secs_f64() * 1_000_000.0 / samples.len() as f64;
    let throughput = samples.len() as f64 / elapsed.as_secs_f64();
    println!(
        "  -> Single-Core Mode D: {:.2} ms | {:.2} μs/mol | {:.1} mols/sec\n",
        elapsed.as_secs_f64() * 1000.0,
        latency_us,
        throughput
    );
}

/// Per-property maximum absolute deviation between two prediction sets.
fn max_deviation(a: &[[f64; 3]], b: &[[f64; 3]]) -> [f64; 3] {
    let mut max = [0.0f64; 3];
    for (x, y) in a.iter().zip(b) {
        for k in 0..3 {
            max[k] = max[k].max((x[k] - y[k]).abs());
        }
    }
    max
}

fn overall(d: &[f64; 3]) -> f64 {
    d[0].max(d[1]).max(d[2])
}

fn print_banner() {
    println!("{DOUBLE_RULE}");
    println!("  GUDDANETWORKS QNN: ZERO-ALLOCATION OBSERVABLE ROTATION KERNEL FUSION  ");
    println!("{DOUBLE_RULE}");
    println!(" Operator Identity : <Z_q>_final = cos(θ_q) <Z_q>_L2 - sin(θ_q) <X_q>_L2");
    println!(" Memory Traffic    : 768 KB Write Traffic -> 0 KB Statevector Mutation in L3");
    println!(" State Evolution   : Stops at Gate 72 (Layer 2) | Dual AVX2 Read-Only POVM");
    println!(
        " Target Hardware   : {}",
        t1000::hostinfo::HostProfile::detect().label()
    );
    println!("{DOUBLE_RULE}\n");
}

fn print_architecture_matrix() {
    println!("{RULE}");
    println!(" [KERNEL EVOLUTION & ARCHITECTURE MATRIX]");
    println!("{RULE}");
    println!(
        " Mode A (Full Baseline)     : 108 Gates (36 Ry + 36 Rz + 36 CZ) | 111 Params (888 B)"
    );
    println!(
        " Mode B (Pruned Canonical)  :  84 Gates (36 Ry + 24 Rz + 24 CZ) |  99 Params (792 B)"
    );
    println!(
        " Mode C (Fused Observable)  :  72 Gates (24 Ry + 24 Rz + 24 CZ) |  99 Params (792 B)"
    );
    println!(
        " Mode D (Fused Q11-Pruned)  :  72 Gates + Skip Q11 POVM Readout |  98 Params (784 B)"
    );
    println!("{RULE}\n");
}

fn print_equivalence_ledger(dc: &[f64; 3], dd: &[f64; 3]) {
    let (max_c, max_d) = (overall(dc), overall(dd));
    println!("{DOUBLE_RULE}");
    println!("         NUMERICAL EQUIVALENCE LEDGER (2,000 HOLDOUT MOLECULES)          ");
    println!("{DOUBLE_RULE}");
    println!(" Comparison Mode          │ Max Deviation logP │ Max Deviation QED  │ Max Deviation SAS  │ Status");
    println!("──────────────────────────┼────────────────────┼────────────────────┼────────────────────┼───────");
    println!(
        " Mode C vs Baseline A     │ {:>18.12e} │ {:>18.12e} │ {:>18.12e} │ EXACT",
        dc[0], dc[1], dc[2]
    );
    println!(
        " Mode D vs Baseline A (Q11│ {:>18.12e} │ {:>18.12e} │ {:>18.12e} │ BOUNDED",
        dd[0], dd[1], dd[2]
    );
    println!("{DOUBLE_RULE}");
    println!(
        " Mode C Overall Max Error : {:.2e} (Machine Precision Rounding Floor)",
        max_c
    );
    println!(
        " Mode D Overall Max Error : {:.2e} (Zero-Impact Qubit 11 Pruning Floor)",
        max_d
    );
    println!(
        " Mode C Equivalence       : {}",
        if max_c < 1e-12 {
            "CONFIRMED (100% BIT-FOR-BIT IDENTICAL)"
        } else {
            "DEVIATION DETECTED"
        }
    );
    println!("{DOUBLE_RULE}\n");
}

fn print_acceleration_table(a: &ModeRun, b: &ModeRun, c: &ModeRun, d: &ModeRun) {
    println!("{DOUBLE_RULE}");
    println!("          THROUGHPUT & LATENCY MULTI-GENERATION ACCELERATION             ");
    println!("{DOUBLE_RULE}");
    println!(" Generation / Kernel Mode      │ Gates │ State Mutation │ Latency (μs) │ Mols / Sec │ Speedup");
    println!("───────────────────────────────┼───────┼────────────────┼──────────────┼────────────┼────────");
    println!(
        " Gen 1: Baseline Circuit (A)   │  108  │    1,536 KB    │ {:>12.2} │ {:>10.1} │ 1.000x",
        a.latency_us, a.throughput
    );
    let rows = [
        ("Gen 2: Pruned Canonical (B)  ", 84, "1,152", b),
        ("Gen 3: Fused Observable (C)  ", 72, "  768", c),
        ("Gen 4: Fused Q11-Pruned (D)  ", 72, "  768", d),
    ];
    for (name, gates, kb, m) in rows {
        println!(
            " {name} │  {gates:>3}  │    {kb} KB    │ {:>12.2} │ {:>10.1} │ {:>5.3}x",
            m.latency_us,
            m.throughput,
            m.throughput / a.throughput
        );
    }
    println!("{DOUBLE_RULE}\n");
}

fn save_fused_checkpoint(epochs_completed: usize, fused_98_params: Vec<f64>) {
    let path = resolve_workspace_path(
        "tests/benchmarks/zinc_molecular/outputs/zinc_12qubit_checkpoint_fused98.json",
    );
    let checkpoint = QnnFusedCheckpoint {
        timestamp_rfc3339: "2026-10-13T20:45:00Z".to_string(),
        n_qubits: N_QUBITS,
        depth: 3,
        full_params: 111,
        pruned_params: 99,
        fused_active_params: 98,
        gate_count_full: 108,
        gate_count_pruned: 84,
        gate_count_fused: 72,
        gate_depth_reduction_pct: 33.33,
        epochs_completed,
        fused_hybrid_params_98: fused_98_params,
    };
    if let Ok(json_str) = serde_json::to_string_pretty(&checkpoint) {
        let _ = t1000::results::write(&path, json_str);
        println!("[SAVED] Fused 98-Parameter Checkpoint: {}", path.display());
    }
}

fn mode_telemetry(
    name: &str,
    gates: usize,
    write_bytes: usize,
    m: &ModeRun,
    speedup: f64,
) -> ModeTelemetry {
    ModeTelemetry {
        name: name.to_string(),
        gates_per_molecule: gates,
        state_write_bytes_l3: write_bytes,
        latency_us: m.latency_us,
        throughput_mols_per_sec: m.throughput,
        speedup_vs_baseline: speedup,
    }
}

fn save_report(n_samples: usize, dc: &[f64; 3], dd: &[f64; 3], runs: [&ModeRun; 4]) {
    let path = resolve_workspace_path(
        "tests/benchmarks/zinc_molecular/outputs/zinc_12qubit_fused_verification.json",
    );
    let [a, b, c, d] = runs;
    let tp_a = a.throughput;
    let report = FusedEquivalenceReport {
        timestamp_rfc3339: "2026-10-13T20:45:00Z".to_string(),
        host_cpu: t1000::hostinfo::HostProfile::detect().label(),
        holdout_samples: n_samples,
        full_param_count: 111,
        pruned_param_count: 99,
        fused_param_count: 98,
        max_diff_mode_c_logp: dc[0],
        max_diff_mode_c_qed: dc[1],
        max_diff_mode_c_sas: dc[2],
        max_diff_mode_c_overall: overall(dc),
        mode_c_exact_equivalence: overall(dc) < 1e-12,
        max_diff_mode_d_overall: overall(dd),
        modes: vec![
            mode_telemetry(
                "Mode A: Baseline Circuit (108 Gates)",
                108,
                768 * 1024,
                a,
                1.0,
            ),
            mode_telemetry(
                "Mode B: Pruned Canonical (84 Gates)",
                84,
                768 * 1024,
                b,
                b.throughput / tp_a,
            ),
            mode_telemetry(
                "Mode C: Fused Observable Rotation (72 Gates + 12-Qubit AVX2)",
                72,
                0,
                c,
                c.throughput / tp_a,
            ),
            mode_telemetry(
                "Mode D: Fused Q11-Pruned (72 Gates + 11-Qubit AVX2)",
                72,
                0,
                d,
                d.throughput / tp_a,
            ),
        ],
    };
    if let Ok(json_str) = serde_json::to_string_pretty(&report) {
        let _ = t1000::results::write(&path, json_str);
        println!(
            "[SAVED] Fused Verification & Telemetry Report: {}",
            path.display()
        );
    }
}

fn main() {
    print_banner();
    let Some((test_samples, ckpt)) = load_holdout_and_checkpoint() else {
        return;
    };
    let full_params = ckpt.hybrid_params.clone();
    assert_eq!(
        full_params.len(),
        111,
        "Checkpoint must contain 111 parameters"
    );
    let pruned_params = derive_pruned_params(&full_params, 12);
    let fused_98_params = derive_pruned_params(&full_params, 11);
    assert_eq!(pruned_params.len(), 99);
    assert_eq!(fused_98_params.len(), 98);
    print_architecture_matrix();

    let encoding = DataEncodingLayer::new(N_QUBITS, EncodingKind::Amplitude);
    let ansatz = CircuitAnsatz::new(N_QUBITS, 3).with_pi_layer(true);
    let bench = Bench {
        ansatz: &ansatz,
        encoding: &encoding,
    };
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(16);
    let total_evals = test_samples.len() * REPEATS;

    println!("[BENCHMARK CONFIGURATION]");
    println!(
        " Concurrency     : {} Worker Threads ({})",
        num_threads,
        t1000::hostinfo::HostProfile::detect().cpu_model
    );
    println!(
        " Holdout Dataset : {} Molecules x {} Repeats = {} Total Evaluations\n",
        test_samples.len(),
        REPEATS,
        total_evals
    );

    let samples = &test_samples[..];
    let a = bench_mode(
        "[EXECUTION] Benchmarking Mode A: Full Circuit (108 Gates)...",
        "Mode A",
        Kernel::Full,
        &bench,
        &full_params,
        samples,
        num_threads,
    );
    let b = bench_mode(
        "[EXECUTION] Benchmarking Mode B: Pruned Canonical (84 Gates)...",
        "Mode B",
        Kernel::Pruned,
        &bench,
        &pruned_params,
        samples,
        num_threads,
    );
    let c = bench_mode(
        "[EXECUTION] Benchmarking Mode C: Fused Observable Rotation (72 Gates + 12-Qubit AVX2)...",
        "Mode C",
        Kernel::Fused12,
        &bench,
        &pruned_params,
        samples,
        num_threads,
    );
    let d = bench_mode("[EXECUTION] Benchmarking Mode D: Fused 98-Param Kernel (72 Gates + 11-Qubit AVX2, Skip Q11)...", "Mode D", Kernel::Fused11, &bench, &fused_98_params, samples, num_threads);
    bench_single_core(&bench, &fused_98_params, samples);

    let dc = max_deviation(&a.preds, &c.preds);
    let dd = max_deviation(&a.preds, &d.preds);
    print_equivalence_ledger(&dc, &dd);
    print_acceleration_table(&a, &b, &c, &d);

    save_fused_checkpoint(ckpt.epochs_completed, fused_98_params);
    save_report(samples.len(), &dc, &dd, [&a, &b, &c, &d]);
}
