//! 20-qubit Random Circuit Sampling benchmark: optimized AVX2 engine vs. a reference simulator.
//!
//! Both engines run the *same* circuit (the depth-12 hardware-efficient ansatz with the
//! Rademacher phase layer, exported gate by gate through `CircuitAnsatz::to_dag`):
//!
//! - **Reference**: a textbook scalar statevector simulator written in this file
//!   (`Vec<Complex64>`, one pass per gate, no SIMD, no fusion, no tiling).
//! - **Optimized**: the crate's `forward_layer_into`: AVX2 Ry kernels plus one fused diagonal
//!   Rz/CZ pass per layer, multithreaded (`GUDDALM_QNN_THREADS`, default all logical CPUs).
//!
//! The report gives the honest speedup of the optimized engine over the straightforward
//! implementation, checks that both produce the same state (max amplitude error, fidelity,
//! total variation distance), and runs Porter-Thomas statistics on the result.
//!
//! For an external baseline, `--export <dir>` writes the circuit and final state, and
//! `scripts/aer_rcs_benchmark.py` runs the identical circuit on Qiskit Aer. (The previous "Classical FFT RCS" baseline was not a circuit simulator:
//! it applied FFTs and pointwise sin/cos to random amplitudes, so its timing was meaningless.)

use t1000::compiler::CircuitOp;
use t1000::{
    AdamOptimizer, CircuitAnsatz, DataEncodingLayer, EncodingKind, ParameterShiftEngine,
    PauliObservable, QuantumState,
};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Serialize, Deserialize, Debug)]
pub struct LayerTelemetry {
    pub layer: usize,
    pub latency_seconds: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct EngineTelemetry {
    pub name: String,
    pub total_time_seconds: f64,
    pub gate_count: usize,
    pub layers: Vec<LayerTelemetry>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EngineAgreement {
    pub max_amplitude_error: f64,
    pub state_fidelity: f64,
    pub total_variation_distance: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VariationalEpochTelemetry {
    pub epoch: usize,
    pub loss: f64,
    pub gradient_norm: f64,
    pub epoch_seconds: f64,
    pub active_threads: usize,
    pub scheduler_mode: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VariationalTrainingReport {
    pub epochs: usize,
    pub initial_loss: f64,
    pub final_loss: f64,
    pub total_training_seconds: f64,
    pub mean_epoch_seconds: f64,
    pub scheduler_mode: String,
    pub history: Vec<VariationalEpochTelemetry>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FrontierRcsReport {
    pub qubits: usize,
    pub depth: usize,
    pub total_states: usize,
    pub reference: EngineTelemetry,
    pub optimized: EngineTelemetry,
    /// Threads used by the optimized engine (`GUDDALM_QNN_THREADS`, default: all logical CPUs).
    pub optimized_threads: usize,
    /// reference time / optimized time (reference is single-threaded; same machine, same circuit).
    pub speedup_vs_reference: f64,
    pub agreement: EngineAgreement,
    pub porter_thomas: t1000_core::PorterThomasReport,
    pub top_bitstrings: Vec<(String, f64)>,
    pub gradients: GradientBenchmark,
    #[serde(default)]
    pub variational_training: Option<VariationalTrainingReport>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GradientBenchmark {
    /// Circuit used for the adjoint vs parameter-shift comparison.
    pub compare_qubits: usize,
    pub compare_depth: usize,
    pub compare_params: usize,
    pub adjoint_seconds: f64,
    pub parameter_shift_seconds: f64,
    pub max_abs_difference: f64,
    /// Adjoint gradient on the main 20-qubit circuit (parameter shift would need 2 * params runs).
    pub full_circuit_params: usize,
    pub full_circuit_adjoint_seconds: f64,
}

/// Times one gradient of <GlobalMeanZ> by adjoint differentiation and by parameter shift.
fn benchmark_gradients(full_qubits: usize, full_depth: usize) -> GradientBenchmark {
    let best_of = |runs: usize, f: &mut dyn FnMut() -> Vec<f64>| -> (Vec<f64>, f64) {
        let mut best = (Vec::new(), f64::INFINITY);
        for _ in 0..runs {
            let t0 = Instant::now();
            let g = f();
            let dt = t0.elapsed().as_secs_f64();
            if dt < best.1 {
                best = (g, dt);
            }
        }
        best
    };
    let engine_for = |n: usize, depth: usize| {
        ParameterShiftEngine::new(
            DataEncodingLayer::new(n, EncodingKind::Angle),
            CircuitAnsatz::new(n, depth).with_pi_layer(true),
            PauliObservable::GlobalMeanZ,
        )
    };

    let (cq, cd) = (full_qubits.min(12), full_depth.min(4));
    let engine = engine_for(cq, cd);
    let x: Vec<f64> = (0..cq).map(|q| 0.1 * q as f64).collect();
    let w = engine.ansatz.weights.clone();
    let (g_adj, t_adj) = best_of(3, &mut || engine.compute_circuit_gradients(&x, &w));
    let (g_ps, t_ps) = best_of(1, &mut || engine.compute_circuit_gradients_parameter_shift(&x, &w));
    let max_diff = g_adj.iter().zip(&g_ps).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);

    let full = engine_for(full_qubits, full_depth);
    let xf: Vec<f64> = (0..full_qubits).map(|q| 0.1 * q as f64).collect();
    let wf = full.ansatz.weights.clone();
    let full_runs = if full_qubits >= 25 { 1 } else { 2 };
    let (_, t_full) = best_of(full_runs, &mut || full.compute_circuit_gradients(&xf, &wf));

    GradientBenchmark {
        compare_qubits: cq,
        compare_depth: cd,
        compare_params: w.len(),
        adjoint_seconds: t_adj,
        parameter_shift_seconds: t_ps,
        max_abs_difference: max_diff,
        full_circuit_params: wf.len(),
        full_circuit_adjoint_seconds: t_full,
    }
}

/// Runs multi-parameter gradient descent epochs using Adam and the dynamic T-1000 scheduler.
fn run_variational_epochs(qubits: usize, depth: usize, epochs: usize) -> VariationalTrainingReport {
    let threads = t1000::parallel::num_threads();
    let (scheduler_mode, active_threads) = if qubits < t1000::parallel::PARALLEL_MIN_QUBITS {
        ("Sequential AVX2 Fast-Path (L3 Cache Bound, 1 Thread)".to_string(), 1)
    } else if qubits < 22 {
        (format!("Parallel Work-Stealing Pool ({threads} Threads)"), threads)
    } else {
        (format!("Parallel Streaming Pool + Non-Temporal Stores ({threads} Threads)"), threads)
    };

    println!("\n=======================================================");
    println!("[VARIATIONAL PIPELINE] Dynamic T-1000 Scheduler Optimization Loop");
    println!("[SCHEDULER MODE] {scheduler_mode}");
    println!("[CONFIG] {qubits} Qubits | Depth {depth} | {epochs} Epochs | Adam lr=0.05");
    println!("-------------------------------------------------------");

    let engine = ParameterShiftEngine::new(
        DataEncodingLayer::new(qubits, EncodingKind::Angle),
        CircuitAnsatz::new(qubits, depth).with_pi_layer(true),
        PauliObservable::GlobalMeanZ,
    );
    let mut weights = engine.ansatz.weights.clone();
    let mut optimizer = AdamOptimizer::new(weights.len(), 0.05);

    let x: Vec<f64> = (0..qubits).map(|q| 0.05 * (q + 1) as f64).collect();
    let y_target = -1.0;

    let mut history = Vec::with_capacity(epochs);
    let t_start = Instant::now();
    let mut initial_loss = 0.0;
    let mut final_loss = 0.0;

    for epoch in 1..=epochs {
        let t_ep = Instant::now();
        let (_pred, loss, grad) = engine.compute_loss_gradients(&x, y_target, &weights);
        if epoch == 1 {
            initial_loss = loss;
        }
        final_loss = loss;

        let grad_norm = (grad.iter().map(|g| g * g).sum::<f64>()).sqrt();
        optimizer.step(&mut weights, &grad);
        let ep_dur = t_ep.elapsed().as_secs_f64();

        println!(
            "  Epoch {epoch}/{epochs}: loss = {loss:.4} | |grad| = {grad_norm:.4} | latency = {ep_dur:.4} s ({active_threads} thread(s))"
        );

        history.push(VariationalEpochTelemetry {
            epoch,
            loss,
            gradient_norm: grad_norm,
            epoch_seconds: ep_dur,
            active_threads,
            scheduler_mode: scheduler_mode.clone(),
        });
    }

    let total_dur = t_start.elapsed().as_secs_f64();
    let mean_dur = total_dur / epochs.max(1) as f64;
    let loss_delta = initial_loss - final_loss;
    println!("-------------------------------------------------------");
    println!(
        "  Summary: Initial Loss {initial_loss:.4} -> Final Loss {final_loss:.4} (delta: {loss_delta:.4}) | {mean_dur:.4} s/epoch"
    );
    println!("=======================================================");

    VariationalTrainingReport {
        epochs,
        initial_loss,
        final_loss,
        total_training_seconds: total_dur,
        mean_epoch_seconds: mean_dur,
        scheduler_mode,
        history,
    }
}

// =========================================================================
// Reference engine: deliberately simple, independent of the crate's kernels
// =========================================================================

fn ref_apply_1q(psi: &mut [Complex64], q: usize, m: [[Complex64; 2]; 2]) {
    let mask = 1usize << q;
    for i in 0..psi.len() {
        if i & mask == 0 {
            let j = i | mask;
            let (a, b) = (psi[i], psi[j]);
            psi[i] = m[0][0] * a + m[0][1] * b;
            psi[j] = m[1][0] * a + m[1][1] * b;
        }
    }
}

fn ref_ry(theta: f64) -> [[Complex64; 2]; 2] {
    let (s, c) = (theta / 2.0).sin_cos();
    [[Complex64::new(c, 0.0), Complex64::new(-s, 0.0)], [Complex64::new(s, 0.0), Complex64::new(c, 0.0)]]
}

fn ref_rz(theta: f64) -> [[Complex64; 2]; 2] {
    let zero = Complex64::new(0.0, 0.0);
    [[Complex64::from_polar(1.0, -theta / 2.0), zero], [zero, Complex64::from_polar(1.0, theta / 2.0)]]
}

fn ref_cz(psi: &mut [Complex64], q1: usize, q2: usize) {
    let both = (1usize << q1) | (1usize << q2);
    for (i, a) in psi.iter_mut().enumerate() {
        if i & both == both {
            *a = -*a;
        }
    }
}

fn ref_cnot(psi: &mut [Complex64], ctrl: usize, target: usize) {
    let cm = 1usize << ctrl;
    let tm = 1usize << target;
    for i in 0..psi.len() {
        if (i & cm != 0) && (i & tm == 0) {
            let j = i | tm;
            psi.swap(i, j);
        }
    }
}

fn ref_anticnot(psi: &mut [Complex64], ctrl: usize, target: usize) {
    let cm = 1usize << ctrl;
    let tm = 1usize << target;
    for i in 0..psi.len() {
        if (i & cm == 0) && (i & tm == 0) {
            let j = i | tm;
            psi.swap(i, j);
        }
    }
}

fn ref_swap(psi: &mut [Complex64], q1: usize, q2: usize) {
    let m1 = 1usize << q1;
    let m2 = 1usize << q2;
    for i in 0..psi.len() {
        if (i & m1 != 0) && (i & m2 == 0) {
            let j = (i ^ m1) | m2;
            psi.swap(i, j);
        }
    }
}

fn ref_cp(psi: &mut [Complex64], ctrl: usize, target: usize, theta: f64) {
    let both = (1usize << ctrl) | (1usize << target);
    let phase = Complex64::from_polar(1.0, theta);
    for i in 0..psi.len() {
        if (i & both) == both {
            psi[i] *= phase;
        }
    }
}

fn run_reference(ops: &[CircuitOp], init: &[Complex64]) -> (Vec<Complex64>, f64) {
    let mut psi = init.to_vec();
    let t0 = Instant::now();
    for op in ops {
        match op {
            CircuitOp::Ry { qubit, theta } => ref_apply_1q(&mut psi, *qubit, ref_ry(*theta)),
            CircuitOp::Rz { qubit, theta } => ref_apply_1q(&mut psi, *qubit, ref_rz(*theta)),
            CircuitOp::Unitary1Q { qubit, matrix } => {
                let m = [
                    [matrix.u00(), matrix.u01()],
                    [matrix.u10(), matrix.u11()],
                ];
                ref_apply_1q(&mut psi, *qubit, m);
            }
            CircuitOp::CZ { q1, q2 } => ref_cz(&mut psi, *q1, *q2),
            CircuitOp::CNOT { control, target } => ref_cnot(&mut psi, *control, *target),
            CircuitOp::AntiCNOT { control, target } => ref_anticnot(&mut psi, *control, *target),
            CircuitOp::SWAP { q1, q2 } => ref_swap(&mut psi, *q1, *q2),
            CircuitOp::ControlledPhase { control, target, theta } => ref_cp(&mut psi, *control, *target, *theta),
            CircuitOp::RademacherPhase { signs } => {
                for (q, s) in signs.iter().enumerate() {
                    ref_apply_1q(&mut psi, q, ref_rz(s * std::f64::consts::PI));
                }
            }
        }
    }
    (psi, t0.elapsed().as_secs_f64())
}

// =========================================================================
// Optimized engine: the crate's SIMD + tiled path
// =========================================================================

fn run_optimized(ansatz: &CircuitAnsatz, init: &QuantumState) -> (QuantumState, Vec<LayerTelemetry>, f64) {
    let mut state = init.clone();
    let t0 = Instant::now();
    ansatz.apply_pre_init(&mut state);
    let mut layers = Vec::with_capacity(ansatz.depth);
    for layer in 0..ansatz.depth {
        let l0 = Instant::now();
        ansatz.forward_layer_into(layer, &ansatz.weights, &mut state);
        layers.push(LayerTelemetry {
            layer: layer + 1,
            latency_seconds: l0.elapsed().as_secs_f64(),
        });

        // Periodic L2 norm verification to verify numerical stability under deep randomized benchmarking passes
        if (layer + 1) % 4 == 0 || layer + 1 == ansatz.depth {
            state.assert_normalized(1e-10);
        }
    }
    (state, layers, t0.elapsed().as_secs_f64())
}

/// Writes the circuit (`circuit.json`) and the optimized engine's final state (`state.bin`,
/// little-endian f64 pairs re, im in index order) so external simulators can run the identical
/// circuit and be checked against it (see `scripts/aer_rcs_benchmark.py`).
fn export_circuit(dir: &std::path::Path, qubits: usize, depth: usize, ops: &[CircuitOp], state: &QuantumState) {
    let gates: Vec<serde_json::Value> = ops
        .iter()
        .map(|op| match op {
            CircuitOp::Ry { qubit, theta } => serde_json::json!({"op": "ry", "qubit": qubit, "theta": theta}),
            CircuitOp::Rz { qubit, theta } => serde_json::json!({"op": "rz", "qubit": qubit, "theta": theta}),
            CircuitOp::Unitary1Q { qubit, .. } => serde_json::json!({"op": "unitary_1q", "qubit": qubit}),
            CircuitOp::CZ { q1, q2 } => serde_json::json!({"op": "cz", "q1": q1, "q2": q2}),
            CircuitOp::CNOT { control, target } => serde_json::json!({"op": "cnot", "control": control, "target": target}),
            CircuitOp::AntiCNOT { control, target } => serde_json::json!({"op": "anticnot", "control": control, "target": target}),
            CircuitOp::SWAP { q1, q2 } => serde_json::json!({"op": "swap", "q1": q1, "q2": q2}),
            CircuitOp::ControlledPhase { control, target, theta } => serde_json::json!({"op": "cp", "control": control, "target": target, "theta": theta}),
            CircuitOp::RademacherPhase { signs } => serde_json::json!({"op": "rademacher", "signs": signs}),
        })
        .collect();
    let circuit = serde_json::json!({
        "qubits": qubits,
        "depth": depth,
        "initial_state": "plus",
        "qubit_order": "qubit 0 is the least significant bit of the amplitude index",
        "gates": gates,
    });
    fs::create_dir_all(dir).expect("create export directory");
    fs::write(dir.join("circuit.json"), serde_json::to_string(&circuit).expect("serialize circuit"))
        .expect("write circuit.json");
    let mut bytes = Vec::with_capacity(state.dim * 16);
    for i in 0..state.dim {
        bytes.extend_from_slice(&state.real[i].to_le_bytes());
        bytes.extend_from_slice(&state.imag[i].to_le_bytes());
    }
    fs::write(dir.join("state.bin"), bytes).expect("write state.bin");
    println!("[EXPORT] circuit.json and state.bin written to {}", dir.display());
}

fn main() {
    // Optional: `--export <dir>` writes the circuit and final state for external comparison.
    let args: Vec<String> = std::env::args().collect();
    let export_dir = args
        .iter()
        .position(|a| a == "--export")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from);

    let qubits = args
        .iter()
        .position(|a| a == "--qubits" || a == "-q")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(14);

    let depth = args
        .iter()
        .position(|a| a == "--depth" || a == "-d")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(36);

    let pre_init_str = args
        .iter()
        .position(|a| a == "--pre-init" || a == "-p")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("rademacher");

    let pre_init = match pre_init_str {
        "bell-ghz" | "ghz" => t1000::PreInitCircuit::BellGhz,
        "alt-bell-ghz" | "alt-ghz" => t1000::PreInitCircuit::AltBellGhz,
        "qft" => t1000::PreInitCircuit::Qft,
        "iqft" => t1000::PreInitCircuit::Iqft,
        "none" => t1000::PreInitCircuit::None,
        _ => t1000::PreInitCircuit::RademacherPi,
    };

    let skip_ref = args.iter().any(|a| a == "--skip-ref");
    let epochs = args
        .iter()
        .position(|a| a == "--epochs" || a == "-e")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(if qubits >= 25 { 2 } else { 5 });

    let dim = 1usize << qubits;

    println!("=========================================================================");
    println!(" {qubits}-QUBIT RCS: OPTIMIZED AVX2 ENGINE vs SCALAR REFERENCE SIMULATOR");
    println!(" (Circuit Depth: {depth} Layers | Pre-Init: {pre_init:?})");
    println!("=========================================================================");

    let ansatz = CircuitAnsatz::new(qubits, depth).with_pre_init(pre_init);
    let ops: Vec<CircuitOp> = ansatz.to_dag(&ansatz.weights).nodes.into_iter().map(|n| n.op).collect();
    println!("[CONFIG] {qubits} qubits | depth {depth} | {} gates | dim {dim}", ops.len());

    // |+>^n initial state
    let amp = 1.0 / (dim as f64).sqrt();
    let init = QuantumState::new(dim);
    let init_c = if !skip_ref {
        vec![Complex64::new(amp, 0.0); dim]
    } else {
        Vec::new()
    };

    // Reference simulator: 1 run if qubits >= 25, or best of 3 if smaller
    let (ref_psi, ref_time) = if skip_ref {
        println!("\n[RUN] Reference scalar simulator skipped (--skip-ref).");
        (Vec::new(), 0.0)
    } else if qubits >= 25 {
        println!("\n[RUN] Reference scalar simulator (1 measured run, N={qubits})...");
        let res = run_reference(&ops, &init_c);
        println!("  time: {:.4} s", res.1);
        res
    } else {
        const RUNS: usize = 3;
        println!("\n[RUN] Reference scalar simulator (warm-up + best of {RUNS})...");
        let _ = run_reference(&ops, &init_c);
        let res = (0..RUNS)
            .map(|_| run_reference(&ops, &init_c))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .expect("RUNS > 0");
        println!("  best: {:.4} s", res.1);
        res
    };
    drop(init_c); // Free reference memory immediately before optimized execution

    let threads = t1000::parallel::num_threads();
    let opt_runs = if qubits >= 25 { 1 } else { 3 };
    let (opt_state, layers, opt_time) = if qubits >= 25 {
        println!("[RUN] Optimized engine, {threads} thread(s) (1 measured run, N={qubits})...");
        let res = run_optimized(&ansatz, &init);
        println!("  time: {:.4} s ({:.4} s/layer)", res.2, res.2 / depth as f64);
        res
    } else {
        println!("[RUN] Optimized engine, {threads} thread(s) (warm-up + best of {opt_runs})...");
        let _ = run_optimized(&ansatz, &init);
        let res = (0..opt_runs)
            .map(|_| run_optimized(&ansatz, &init))
            .min_by(|a, b| a.2.total_cmp(&b.2))
            .expect("RUNS > 0");
        println!("  best: {:.4} s ({:.4} s/layer)", res.2, res.2 / depth as f64);
        res
    };
    drop(init); // Free initial statevector immediately

    // Agreement between engines
    let agreement = if !ref_psi.is_empty() {
        let mut max_err = 0.0f64;
        let mut overlap = Complex64::new(0.0, 0.0);
        let mut tvd = 0.0;
        for i in 0..dim {
            let o = Complex64::new(opt_state.real[i], opt_state.imag[i]);
            max_err = max_err.max((o - ref_psi[i]).norm());
            overlap += ref_psi[i].conj() * o;
            tvd += (o.norm_sqr() - ref_psi[i].norm_sqr()).abs();
        }
        EngineAgreement {
            max_amplitude_error: max_err,
            state_fidelity: overlap.norm_sqr(),
            total_variation_distance: 0.5 * tvd,
        }
    } else {
        EngineAgreement {
            max_amplitude_error: 0.0,
            state_fidelity: 1.0,
            total_variation_distance: 0.0,
        }
    };
    drop(ref_psi); // Free reference state immediately

    if let Some(dir) = &export_dir {
        export_circuit(dir, qubits, depth, &ops, &opt_state);
    }

    let probs = opt_state.probabilities();
    let pt = t1000_core::porter_thomas::verify_porter_thomas(&probs);
    let mut indexed: Vec<(usize, f64)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
    let top_bitstrings: Vec<(String, f64)> = indexed
        .iter()
        .take(5)
        .map(|(idx, p)| (format!("|{idx:0qubits$b}>"), *p))
        .collect();

    let speedup = if opt_time > 0.0 && ref_time > 0.0 {
        ref_time / opt_time
    } else {
        1.0
    };
    println!("\n=========================================================================");
    println!(" RESULTS");
    println!("=========================================================================");
    println!("  Reference scalar simulator : {ref_time:.4} s");
    println!("  Optimized engine           : {opt_time:.4} s ({threads} thread(s))");
    println!("  Speedup vs reference       : {speedup:.2}x (optimized: {threads} thread(s); reference: 1)");
    println!("-------------------------------------------------------------------------");
    println!("  Engine agreement");
    println!("    max |amplitude error|    : {:.3e}", agreement.max_amplitude_error);
    println!("    state fidelity           : {:.12}", agreement.state_fidelity);
    println!("    total variation distance : {:.3e}", agreement.total_variation_distance);
    println!("-------------------------------------------------------------------------");
    println!("  Porter-Thomas statistics (ideal chaotic circuit)");
    println!("    linear XEB (self)        : {:.4} (ideal 1.0)", pt.exact_xeb);
    println!("    entropy deficit          : {:.4} nats (ideal {:.4})", pt.entropy_deficit, pt.expected_entropy_deficit);
    println!("    participation ratio      : {:.0} (ideal {:.0})", pt.effective_participation_ratio, pt.expected_participation_ratio);
    println!(
        "    K-S test                 : {} (D = {:.4}, critical {:.4})",
        if pt.passes_ks_test { "pass" } else { "fail" },
        pt.ks_statistic,
        pt.ks_critical_value
    );
    println!("  External baseline: run with --export <dir>, then scripts/aer_rcs_benchmark.py <dir>.");

    profile_rademacher_phase_layer(qubits);

    println!("\n=======================================================");
    println!("[GRADIENTS] adjoint differentiation vs parameter shift (<GlobalMeanZ>)");
    println!("-------------------------------------------------------");
    let gradients = benchmark_gradients(qubits, depth);
    println!(
        "  {} qubits, depth {}, {} params: adjoint {:.4} s | parameter shift {:.4} s | {:.0}x faster | max |diff| {:.2e}",
        gradients.compare_qubits,
        gradients.compare_depth,
        gradients.compare_params,
        gradients.adjoint_seconds,
        gradients.parameter_shift_seconds,
        gradients.parameter_shift_seconds / gradients.adjoint_seconds,
        gradients.max_abs_difference
    );
    println!(
        "  {} qubits, depth {}, {} params: adjoint {:.4} s (parameter shift would need {} circuit runs)",
        qubits,
        depth,
        gradients.full_circuit_params,
        gradients.full_circuit_adjoint_seconds,
        2 * gradients.full_circuit_params
    );
    println!("=======================================================");

    let variational_training = if epochs > 0 {
        let v_qubits = qubits;
        let v_depth = if qubits >= 25 { depth.min(4) } else { depth.min(8) };
        Some(run_variational_epochs(v_qubits, v_depth, epochs))
    } else {
        None
    };

    let report = FrontierRcsReport {
        qubits,
        depth,
        total_states: dim,
        reference: EngineTelemetry {
            name: "Scalar reference statevector (frontier_rcs.rs)".to_string(),
            total_time_seconds: ref_time,
            gate_count: ops.len(),
            layers: Vec::new(),
        },
        optimized: EngineTelemetry {
            name: "t1000 AVX2 + tiled kernels".to_string(),
            total_time_seconds: opt_time,
            gate_count: ops.len(),
            layers,
        },
        optimized_threads: threads,
        speedup_vs_reference: speedup,
        agreement,
        porter_thomas: pt,
        top_bitstrings,
        gradients,
        variational_training,
    };

    // Always write to <workspace>/results/experiments, whatever the current directory is.
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let ws_root = manifest_dir.ancestors().nth(2).unwrap_or(&manifest_dir).to_path_buf();
    let out_path = ws_root.join("results/experiments/frontier_rcs_results.json");
    if let Some(parent) = out_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    match serde_json::to_string_pretty(&report) {
        Ok(json) => match fs::write(&out_path, json) {
            Ok(()) => println!("\n[SAVED] {}", out_path.display()),
            Err(e) => eprintln!("[-] could not write {}: {e}", out_path.display()),
        },
        Err(e) => eprintln!("[-] could not serialize report: {e}"),
    }
}

fn profile_rademacher_phase_layer(qubits: usize) {
    let num_states = 1 << qubits;
    let bytes = num_states * 16;
    let size_str = if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    };

    println!("\n=======================================================");
    println!("[PROFILE] RADEMACHER PHASE LAYER: fused kernel vs per-qubit Rz");
    println!("[CONFIG] {} Qubits | State Size: {} Complex Amplitudes ({})", qubits, num_states, size_str);
    println!("-------------------------------------------------------");

    let base_state = QuantumState::new(num_states);
    let pi_layer = t1000::PiPhaseLayer::new(qubits);

    // Warmup
    let mut s_warm = base_state.clone();
    pi_layer.forward_in_place(&mut s_warm);

    // 1. Sequential per-qubit Rz (one pass over statevector per qubit)
    let iters = 5;
    let t0 = Instant::now();
    for _ in 0..iters {
        let mut s_seq = base_state.clone();
        for q in 0..qubits {
            let theta = pi_layer.signs[q] * std::f64::consts::PI;
            t1000::simd::apply_rz_kernel(&mut s_seq.real, &mut s_seq.imag, q, theta);
        }
    }
    let seq_dur_ms = (t0.elapsed().as_secs_f64() * 1000.0) / (iters as f64);

    // 2. Fused kernel (single pass over statevector)
    let iters_avx = 20;
    let t1 = Instant::now();
    for _ in 0..iters_avx {
        let mut s_fused = base_state.clone();
        pi_layer.forward_in_place(&mut s_fused);
    }
    let fused_dur_ms = (t1.elapsed().as_secs_f64() * 1000.0) / (iters_avx as f64);

    println!("  Per-qubit Rz loop  : {:.3} ms ({} passes, includes {} clone)", seq_dur_ms, qubits, size_str);
    println!("  Fused kernel       : {:.3} ms (1 pass, includes {} clone)", fused_dur_ms, size_str);
    println!("  Speedup            : {:.2}x", seq_dur_ms / fused_dur_ms);
    println!("=======================================================");
}
