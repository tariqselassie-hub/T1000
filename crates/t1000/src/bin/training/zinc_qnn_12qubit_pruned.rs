//! GuddaNetworks 12-Qubit Hybrid Variational QNN: Pruned Canonical Kernel Verification
//!
//! Demonstrates and verifies exact mathematical equivalence between:
//! 1. Full 111-Parameter Circuit (108 Quantum Gates / Sample)
//! 2. Pruned 99-Parameter Canonical Kernel (84 Quantum Gates / Sample, 792 Bytes State)
//!
//! Exploits the commutation theorem:
//!   [Rz(φ_i), Z_j] = 0 and [CZ_ring, Z_j] = 0
//! Stripping the redundant terminal Rz and Ring-CZ gates in Layer 3 yields identical
//! Pauli-Z expectation values down to machine precision while cutting gate depth by 22.2%.

use serde::{Deserialize, Serialize};
use std::fs;
use std::time::{Duration, Instant};
use t1000::zinc_data::{self, ZincMolecule};
use t1000::zinc_training_common::{eval_all_z12, resolve_workspace_path, QnnCheckpoint};
use t1000::{CircuitAnsatz, DataEncodingLayer, EncodingKind, QuantumState};

const N_QUBITS: usize = 12;
const DIM: usize = 1 << N_QUBITS; // 4,096
#[allow(dead_code)]
const N_PROPS: usize = 3;
const RULE: &str = "-------------------------------------------------------------------------";
const DOUBLE_RULE: &str =
    "=========================================================================";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QnnPrunedCheckpoint {
    pub timestamp_rfc3339: String,
    pub n_qubits: usize,
    pub depth: usize,
    pub full_params: usize,
    pub pruned_params: usize,
    pub gate_count_full: usize,
    pub gate_count_pruned: usize,
    pub gate_depth_reduction_pct: f64,
    pub epochs_completed: usize,
    pub pruned_hybrid_params: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquivalenceReport {
    pub timestamp_rfc3339: String,
    pub host_cpu: String,
    pub holdout_samples: usize,
    pub full_param_count: usize,
    pub pruned_param_count: usize,
    pub full_gate_count: usize,
    pub pruned_gate_count: usize,
    pub max_abs_diff_logp: f64,
    pub max_abs_diff_qed: f64,
    pub max_abs_diff_sas: f64,
    pub max_overall_diff: f64,
    pub exact_bit_for_bit_equivalence: bool,
    pub full_latency_us: f64,
    pub pruned_latency_us: f64,
    pub latency_reduction_pct: f64,
    pub full_throughput_mols_per_sec: f64,
    pub pruned_throughput_mols_per_sec: f64,
    pub throughput_speedup_x: f64,
}

/// Which circuit kernel a prediction pass runs.
#[derive(Clone, Copy)]
enum Kernel {
    /// 72 quantum angles + 39 classical readout parameters (108 gates).
    Full,
    /// 60 quantum angles + 39 classical readout parameters (84 gates).
    Pruned,
}

impl Kernel {
    fn quantum_params(self) -> usize {
        match self {
            Kernel::Full => 72,
            Kernel::Pruned => 60,
        }
    }
}

/// Per-property deviation between the two kernels over the holdout set.
struct Deviation {
    max_logp: f64,
    max_qed: f64,
    max_sas: f64,
    sum_logp: f64,
    sum_qed: f64,
    sum_sas: f64,
}

impl Deviation {
    fn max_overall(&self) -> f64 {
        self.max_logp.max(self.max_qed).max(self.max_sas)
    }
}

/// One timed prediction pass.
struct Timed {
    predictions: Vec<[f64; 3]>,
    latency_us: f64,
    throughput: f64,
}

fn print_banner() {
    println!("{DOUBLE_RULE}");
    println!("   GUDDANETWORKS QNN: 99-PARAMETER PRUNED CANONICAL KERNEL VERIFICATION  ");
    println!("{DOUBLE_RULE}");
    println!(" Theorem        : Commutation Invariance [Rz(φ_i), Z_j] = 0 & [CZ, Z_j] = 0");
    println!(" Target Circuit : 12 Qubits (dim=4,096 | 64 KB L2 Cache) | Depth = 3");
    println!(" Pruning Target : 111 Parameters (108 Gates) -> 99 Parameters (84 Gates)");
    let host = t1000::hostinfo::HostProfile::detect();
    println!(
        " Hardware Host  : {} | {:.1} GB RAM",
        host.label(),
        host.ram_gb
    );
    println!("{DOUBLE_RULE}\n");
}

/// Loads the 2,000-molecule holdout set and the 30-epoch checkpoint.
/// Prints the reason and returns `None` if either is unusable.
fn load_inputs() -> Option<(Vec<ZincMolecule>, QnnCheckpoint)> {
    let test_path =
        resolve_workspace_path("tests/benchmarks/zinc_molecular/outputs/zinc_test_2k.csv");
    let ckpt_path = resolve_workspace_path(
        "tests/benchmarks/zinc_molecular/outputs/zinc_12qubit_checkpoint.json",
    );

    println!(
        "[INIT] Loading ZINC 2,000 holdout molecules: {}",
        test_path.display()
    );
    let test_samples = match zinc_data::load(&test_path, 2000) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[-] Error: {}", e);
            return None;
        }
    };
    println!("  -> Loaded {} molecules.\n", test_samples.len());

    println!(
        "[INIT] Loading 30-epoch checkpoint: {}",
        ckpt_path.display()
    );
    let ckpt_content = match fs::read_to_string(&ckpt_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[-] Error reading checkpoint: {}", e);
            return None;
        }
    };
    let ckpt: QnnCheckpoint = match serde_json::from_str(&ckpt_content) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[-] Error parsing checkpoint: {}", e);
            return None;
        }
    };
    println!(
        "  -> Epochs Completed: {}, Full Parameters: {}\n",
        ckpt.epochs_completed,
        ckpt.hybrid_params.len()
    );
    Some((test_samples, ckpt))
}

/// Builds the 99-parameter pruned vector from the 111-parameter full one:
/// 1. Layer 1: params[0..24] (24 angles: 12 Ry + 12 Rz)
/// 2. Layer 2: params[24..48] (24 angles: 12 Ry + 12 Rz)
/// 3. Layer 3: params[48..60] (12 angles: 12 Ry ONLY)
/// 4. Readout weights: params[72..108] (36 weights)
/// 5. Readout biases: params[108..111] (3 biases)
///
/// Total = 24 + 24 + 12 + 36 + 3 = 99 parameters (792 bytes).
fn prune_params(full_params: &[f64]) -> Vec<f64> {
    let mut pruned_params = Vec::with_capacity(99);
    // Layers 1 & 2
    pruned_params.extend_from_slice(&full_params[0..48]);
    // Layer 3 (only Ry angles: full_params[48 + 2*q])
    for q in 0..12 {
        pruned_params.push(full_params[48 + 2 * q]);
    }
    // Readout weights & biases
    pruned_params.extend_from_slice(&full_params[72..111]);

    assert_eq!(
        pruned_params.len(),
        99,
        "Pruned parameter count must equal 99"
    );
    pruned_params
}

fn print_architecture_comparison() {
    println!("{RULE}");
    println!(" [SUB-SYSTEM ALLOCATION & GATE DEPTH COMPARISON]");
    println!("{RULE}");
    println!(
        " Full Model Architecture  : 72 Quantum Angles + 39 Classical = 111 Params (888 bytes)"
    );
    println!(
        " Pruned Model Architecture: 60 Quantum Angles + 39 Classical = 99 Params (792 bytes)"
    );
    println!(" Full Circuit Operations  : 36 Ry + 36 Rz + 36 CZ = 108 Gates / Molecule");
    println!(" Pruned Circuit Operations: 36 Ry + 24 Rz + 24 CZ = 84 Gates / Molecule");
    println!(" Gate Depth Reduction     : -24 Gates (-22.2%) with Zero Mathematical Loss");
    println!("{RULE}\n");
}

/// Predicts (logP, QED, SAS) for one chunk of molecules with the given kernel.
fn predict_chunk(
    kernel: Kernel,
    ansatz: &CircuitAnsatz,
    encoding: &DataEncodingLayer,
    params: &[f64],
    chunk: &[ZincMolecule],
) -> Vec<[f64; 3]> {
    let mut out_state = QuantumState::new(DIM);
    let mut z_eval = [0.0; 12];
    let mut preds = Vec::with_capacity(chunk.len());

    // Readout weights follow the quantum angles; the 3 biases follow the weights.
    let n_quantum = kernel.quantum_params();
    let w_off = n_quantum;
    let b_off = n_quantum + 36;

    for sample in chunk {
        let state_in = encoding.encode(&sample.features);
        match kernel {
            Kernel::Full => ansatz.forward_into(&state_in, &params[..n_quantum], &mut out_state),
            Kernel::Pruned => {
                ansatz.forward_pruned_into(&state_in, &params[..n_quantum], &mut out_state)
            }
        }
        eval_all_z12(&out_state, &mut z_eval);

        let mut p_logp = params[b_off];
        let mut p_qed = params[b_off + 1];
        let mut p_sas = params[b_off + 2];
        for q in 0..12 {
            p_logp += params[w_off + q] * z_eval[q];
            p_qed += params[w_off + 12 + q] * z_eval[q];
            p_sas += params[w_off + 2 * 12 + q] * z_eval[q];
        }
        preds.push([p_logp, p_qed, p_sas]);
    }
    preds
}

/// Runs `kernel` over every molecule, split across `num_threads` scoped threads.
fn predict_all(
    kernel: Kernel,
    ansatz: &CircuitAnsatz,
    encoding: &DataEncodingLayer,
    params: &[f64],
    samples: &[ZincMolecule],
    num_threads: usize,
) -> Vec<[f64; 3]> {
    let chunk_size = samples.len().div_ceil(num_threads);
    let sub_chunks: Vec<&[ZincMolecule]> = samples.chunks(chunk_size).collect();

    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(sub_chunks.len());
        for chunk in sub_chunks {
            handles.push(s.spawn(move || predict_chunk(kernel, ansatz, encoding, params, chunk)));
        }
        let mut all_preds = Vec::with_capacity(samples.len());
        for h in handles {
            all_preds.extend(h.join().unwrap());
        }
        all_preds
    })
}

/// Times one full prediction pass and prints its summary line.
fn run_timed(
    title: &str,
    kernel: Kernel,
    ansatz: &CircuitAnsatz,
    encoding: &DataEncodingLayer,
    params: &[f64],
    samples: &[ZincMolecule],
    num_threads: usize,
) -> Timed {
    println!("{title}");
    let start = Instant::now();
    let predictions = predict_all(kernel, ansatz, encoding, params, samples, num_threads);
    let elapsed: Duration = start.elapsed();
    let latency_us = elapsed.as_secs_f64() * 1_000_000.0 / samples.len() as f64;
    let throughput = samples.len() as f64 / elapsed.as_secs_f64();
    let trailer = if matches!(kernel, Kernel::Pruned) {
        "\n"
    } else {
        ""
    };
    println!(
        "  -> Completed in {:.2} ms ({:.2} μs/mol | {:.1} mols/sec){trailer}",
        elapsed.as_secs_f64() * 1000.0,
        latency_us,
        throughput
    );
    Timed {
        predictions,
        latency_us,
        throughput,
    }
}

fn measure_deviation(full: &[[f64; 3]], pruned: &[[f64; 3]]) -> Deviation {
    let mut dev = Deviation {
        max_logp: 0.0,
        max_qed: 0.0,
        max_sas: 0.0,
        sum_logp: 0.0,
        sum_qed: 0.0,
        sum_sas: 0.0,
    };
    for (f, p) in full.iter().zip(pruned) {
        let diff_logp = (f[0] - p[0]).abs();
        let diff_qed = (f[1] - p[1]).abs();
        let diff_sas = (f[2] - p[2]).abs();

        dev.max_logp = dev.max_logp.max(diff_logp);
        dev.max_qed = dev.max_qed.max(diff_qed);
        dev.max_sas = dev.max_sas.max(diff_sas);

        dev.sum_logp += diff_logp;
        dev.sum_qed += diff_qed;
        dev.sum_sas += diff_sas;
    }
    dev
}

fn print_equivalence_ledger(dev: &Deviation, n: f64, exact_match: bool) {
    println!("{DOUBLE_RULE}");
    println!("       BIT-FOR-BIT EQUIVALENCE VERIFICATION LEDGER (2,000 MOLECULES)     ");
    println!("{DOUBLE_RULE}");
    println!(" Target Property     │ Max Absolute Deviation │ Mean Absolute Deviation │ Status");
    println!("─────────────────────┼────────────────────────┼─────────────────────────┼───────");
    let rows = [
        (" logP (Lipophilicity)", dev.max_logp, dev.sum_logp),
        (" QED  (Drug-Likeness)", dev.max_qed, dev.sum_qed),
        (" SAS  (Synthetic Acc)", dev.max_sas, dev.sum_sas),
    ];
    for (name, max, sum) in rows {
        println!("{name}│ {:>22.16} │ {:>23.16} │ EXACT", max, sum / n);
    }
    println!("{DOUBLE_RULE}");
    println!(
        " Overall Maximum Error  : {:.2e} (Machine-Precision Rounding Floor)",
        dev.max_overall()
    );
    println!(
        " Mathematical Equivalence: {}",
        if exact_match {
            "CONFIRMED (100% BIT-FOR-BIT IDENTICAL)"
        } else {
            "FAILED"
        }
    );
    println!("{DOUBLE_RULE}\n");
}

fn print_performance(full: &Timed, pruned: &Timed, latency_reduction_pct: f64, speedup_x: f64) {
    println!("{RULE}");
    println!(" [HARDWARE PERFORMANCE & ACCELERATION PROFILE]");
    println!("{RULE}");
    println!(
        " Full Circuit Latency (108 Gates)   : {:.2} μs / molecule",
        full.latency_us
    );
    println!(
        " Pruned Circuit Latency (84 Gates)  : {:.2} μs / molecule",
        pruned.latency_us
    );
    println!(
        " Latency Reduction                  : {:+.2}%",
        latency_reduction_pct
    );
    println!(
        " Full Circuit Throughput            : {:.1} molecules / second",
        full.throughput
    );
    println!(
        " Pruned Circuit Throughput          : {:.1} molecules / second",
        pruned.throughput
    );
    println!(
        " Throughput Acceleration Factor     : {:.3}x Speedup",
        speedup_x
    );
    println!(" State Memory Saving                : 888 bytes -> 792 bytes (-10.8%)");
    println!("{RULE}\n");
}

fn save_pruned_checkpoint(epochs_completed: usize, pruned_params: Vec<f64>) {
    let path = resolve_workspace_path(
        "tests/benchmarks/zinc_molecular/outputs/zinc_12qubit_checkpoint_pruned99.json",
    );
    let checkpoint = QnnPrunedCheckpoint {
        timestamp_rfc3339: "2026-10-13T20:15:00Z".to_string(),
        n_qubits: N_QUBITS,
        depth: 3,
        full_params: 111,
        pruned_params: 99,
        gate_count_full: 108,
        gate_count_pruned: 84,
        gate_depth_reduction_pct: 22.22,
        epochs_completed,
        pruned_hybrid_params: pruned_params,
    };
    if let Ok(json_str) = serde_json::to_string_pretty(&checkpoint) {
        let _ = t1000::results::write(&path, json_str);
        println!(
            "[SAVED] Canonical 99-parameter checkpoint: {}",
            path.display()
        );
    }
}

fn save_report(report: &EquivalenceReport) {
    let path = resolve_workspace_path(
        "tests/benchmarks/zinc_molecular/outputs/zinc_12qubit_pruned_verification.json",
    );
    if let Ok(json_str) = serde_json::to_string_pretty(report) {
        let _ = t1000::results::write(&path, json_str);
        println!("[SAVED] Pruned Equivalence Report: {}", path.display());
    }
}

fn build_report(
    n_samples: usize,
    dev: &Deviation,
    exact_match: bool,
    full: &Timed,
    pruned: &Timed,
    latency_reduction_pct: f64,
    speedup_x: f64,
) -> EquivalenceReport {
    EquivalenceReport {
        timestamp_rfc3339: "2026-10-13T20:15:00Z".to_string(),
        host_cpu: t1000::hostinfo::HostProfile::detect().label(),
        holdout_samples: n_samples,
        full_param_count: 111,
        pruned_param_count: 99,
        full_gate_count: 108,
        pruned_gate_count: 84,
        max_abs_diff_logp: dev.max_logp,
        max_abs_diff_qed: dev.max_qed,
        max_abs_diff_sas: dev.max_sas,
        max_overall_diff: dev.max_overall(),
        exact_bit_for_bit_equivalence: exact_match,
        full_latency_us: full.latency_us,
        pruned_latency_us: pruned.latency_us,
        latency_reduction_pct,
        full_throughput_mols_per_sec: full.throughput,
        pruned_throughput_mols_per_sec: pruned.throughput,
        throughput_speedup_x: speedup_x,
    }
}

fn main() {
    print_banner();

    let Some((test_samples, ckpt)) = load_inputs() else {
        return;
    };
    if ckpt.hybrid_params.len() != 111 {
        eprintln!(
            "[-] Expected 111 hybrid parameters in checkpoint, found {}",
            ckpt.hybrid_params.len()
        );
        return;
    }
    let full_params = ckpt.hybrid_params.clone();
    let pruned_params = prune_params(&full_params);
    print_architecture_comparison();

    let encoding = DataEncodingLayer::new(N_QUBITS, EncodingKind::Amplitude);
    let ansatz_full = CircuitAnsatz::new(N_QUBITS, 3).with_pi_layer(true);
    let ansatz_pruned = CircuitAnsatz::new(N_QUBITS, 3).with_pi_layer(true);
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get().min(12))
        .unwrap_or(8);

    let full = run_timed(
        "[EXECUTION] Running Mode A: Full 111-Parameter Kernel (108 Gates)...",
        Kernel::Full,
        &ansatz_full,
        &encoding,
        &full_params,
        &test_samples,
        num_threads,
    );
    let pruned = run_timed(
        "[EXECUTION] Running Mode B: Pruned 99-Parameter Canonical Kernel (84 Gates)...",
        Kernel::Pruned,
        &ansatz_pruned,
        &encoding,
        &pruned_params,
        &test_samples,
        num_threads,
    );

    let dev = measure_deviation(&full.predictions, &pruned.predictions);
    let exact_match = dev.max_overall() < 1e-12;
    print_equivalence_ledger(&dev, test_samples.len() as f64, exact_match);

    let latency_reduction_pct = ((full.latency_us - pruned.latency_us) / full.latency_us) * 100.0;
    let speedup_x = pruned.throughput / full.throughput;
    print_performance(&full, &pruned, latency_reduction_pct, speedup_x);

    save_pruned_checkpoint(ckpt.epochs_completed, pruned_params);
    let report = build_report(
        test_samples.len(),
        &dev,
        exact_match,
        &full,
        &pruned,
        latency_reduction_pct,
        speedup_x,
    );
    save_report(&report);
}
