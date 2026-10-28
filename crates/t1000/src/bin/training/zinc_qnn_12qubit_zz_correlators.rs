// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! GuddaNetworks 12-Qubit QNN: 2-Body Entanglement Correlator Engine
//!
//! MECHANISM #1: Extracts all 66 Ising correlators ⟨Z_i Z_j⟩ alongside the 12 single-qubit
//! ⟨Z_q⟩ marginals from the fused AVX2 read-only POVM reduction kernel.
//!
//! Observable Vector Expansion:
//!   Baseline : 12 single-qubit ⟨Z_q⟩ → 3-target linear readout (36 weights + 3 biases = 39 params)
//!   Enhanced : 12 ⟨Z_q⟩ + 66 ⟨Z_i Z_j⟩ = 78 observables → 3-target readout (234 weights + 3 biases = 237 params)
//!
//! The additional 66 two-body correlators capture entanglement structure that single-qubit
//! marginals are blind to, enabling the readout head to learn degree-4 non-linearities
//! in the molecular descriptor space.
//!
//! Training Pipeline:
//!   1. Load 30-epoch pruned checkpoint (48 quantum angles + 12 terminal Ry = 60 circuit params)
//!   2. Freeze quantum circuit parameters (no parameter-shift gradient overhead)
//!   3. Extract 78-dimensional observable vectors for all training molecules
//!   4. Train linear readout W·o + b via closed-form least-squares (Ridge Regression)
//!   5. Evaluate on 2,000 holdout molecules and compare MAE/R² vs baseline

use serde::{Deserialize, Serialize};
use std::fs;
use std::time::Instant;
use t1000::zinc_data::{self, ZincMolecule};
use t1000::zinc_readout_fit::{mae_r2_per_property, ridge_regression_multi};
use t1000::zinc_training_common::{resolve_workspace_path, PrunedCheckpoint, QnnCheckpoint};
use t1000::{CircuitAnsatz, DataEncodingLayer, EncodingKind, QuantumState};

const N_QUBITS: usize = 12;
const DIM: usize = 1 << N_QUBITS; // 4,096
const N_PAIRS: usize = N_QUBITS * (N_QUBITS - 1) / 2; // 66
const N_OBS: usize = N_QUBITS + N_PAIRS; // 78
const N_PROPS: usize = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZZCorrelatorReport {
    pub timestamp_rfc3339: String,
    pub host_cpu: String,
    pub n_qubits: usize,
    pub n_single_qubit_obs: usize,
    pub n_two_body_obs: usize,
    pub n_total_obs: usize,
    pub n_readout_weights_baseline: usize,
    pub n_readout_weights_enhanced: usize,
    pub train_samples: usize,
    pub test_samples: usize,
    pub fused_zz_max_error: f64,
    pub fused_zz_verified: bool,
    pub baseline_mae_logp: f64,
    pub baseline_mae_qed: f64,
    pub baseline_mae_sas: f64,
    pub baseline_r2_logp: f64,
    pub baseline_r2_qed: f64,
    pub baseline_r2_sas: f64,
    pub enhanced_mae_logp: f64,
    pub enhanced_mae_qed: f64,
    pub enhanced_mae_sas: f64,
    pub enhanced_r2_logp: f64,
    pub enhanced_r2_qed: f64,
    pub enhanced_r2_sas: f64,
    pub mae_improvement_logp_pct: f64,
    pub mae_improvement_qed_pct: f64,
    pub mae_improvement_sas_pct: f64,
    pub throughput_baseline_mols_per_sec: f64,
    pub throughput_enhanced_mols_per_sec: f64,
    #[serde(default)]
    pub classical_baselines: Vec<t1000::baselines::BaselineResult>,
}

#[path = "zinc_qnn_12qubit_zz_correlators/report.rs"]
mod report;
use report::{
    improvement_pct, print_gain_ledger, print_results_table, print_top_corrections,
    print_zz_heatmap, save_outputs, Results,
};

const RULE: &str = "=========================================================================";
const LAMBDA: f64 = 1e-3; // Ridge regularization strength

/// Pair index for (i, j) where i > j: pair_idx = i*(i-1)/2 + j
#[inline(always)]
fn pair_index(i: usize, j: usize) -> usize {
    debug_assert!(i > j);
    i * (i - 1) / 2 + j
}

fn print_banner() {
    println!("{RULE}");
    println!("  GUDDANETWORKS QNN: 2-BODY ENTANGLEMENT CORRELATOR ENGINE              ");
    println!("{RULE}");
    println!(" Mechanism         : ⟨Z_i Z_j⟩ Ising Correlators (66 Two-Body Observables)");
    println!(" Observable Vector : 12 ⟨Z_q⟩ + 66 ⟨Z_i Z_j⟩ = 78 Total Observables");
    println!(" Readout Expansion : 39 Params (Baseline) → 237 Params (Enhanced)");
    println!(" Circuit Freeze    : 60 Quantum Angles Frozen (Zero Parameter-Shift Cost)");
    println!(" Training Mode     : Closed-Form Ridge Regression (Instant Convergence)");
    println!("{RULE}\n");
}

fn print_phase(title: &str) {
    println!("{RULE}");
    println!("{title}");
    println!("{RULE}");
}

/// Loads `rel` (the `data/` copy), falling back to `fallback_rel` (the `outputs/` copy).
/// Prints the original error and returns `None` if neither loads.
fn load_with_fallback(rel: &str, fallback_rel: &str, max_rows: usize) -> Option<Vec<ZincMolecule>> {
    match zinc_data::load(resolve_workspace_path(rel), max_rows) {
        Ok(s) => Some(s),
        Err(e) => match zinc_data::load(resolve_workspace_path(fallback_rel), max_rows) {
            Ok(s) => Some(s),
            Err(_) => {
                eprintln!("[-] Error: {}", e);
                None
            }
        },
    }
}

/// The 60 frozen circuit angles, from the pruned checkpoint or else the full one.
fn load_circuit_params() -> Option<Vec<f64>> {
    let pruned_path = resolve_workspace_path(
        "tests/benchmarks/zinc_molecular/checkpoints/zinc_12qubit_checkpoint_pruned99.json",
    );
    let full_path = resolve_workspace_path(
        "tests/benchmarks/zinc_molecular/checkpoints/zinc_12qubit_checkpoint.json",
    );
    println!("[INIT] Loading quantum circuit parameters...");

    let circuit_params: Vec<f64> = if pruned_path.exists() {
        println!(
            "  -> Using pruned 99-parameter checkpoint: {}",
            pruned_path.display()
        );
        let content = fs::read_to_string(&pruned_path).expect("Read pruned checkpoint");
        let ckpt: PrunedCheckpoint =
            serde_json::from_str(&content).expect("Parse pruned checkpoint");
        // First 60 parameters: 48 L1/L2 angles + 12 terminal Ry
        ckpt.pruned_hybrid_params[..60].to_vec()
    } else if full_path.exists() {
        println!(
            "  -> Using full 111-parameter checkpoint: {}",
            full_path.display()
        );
        let content = fs::read_to_string(&full_path).expect("Read full checkpoint");
        let ckpt: QnnCheckpoint = serde_json::from_str(&content).expect("Parse full checkpoint");
        // Pruned layout: first 48 params, then the Ry-only angles of layer 3
        let mut params = Vec::with_capacity(60);
        params.extend_from_slice(&ckpt.hybrid_params[..48]);
        for q in 0..12 {
            params.push(ckpt.hybrid_params[48 + 2 * q]);
        }
        params
    } else {
        eprintln!("[-] No checkpoint found. Please train first.");
        return None;
    };
    assert_eq!(circuit_params.len(), 60, "Expected 60 circuit parameters");
    println!(
        "  -> Circuit parameters loaded ({} frozen angles).\n",
        circuit_params.len()
    );
    Some(circuit_params)
}

/// The frozen circuit and its input encoding.
struct Frozen<'a> {
    ansatz: &'a CircuitAnsatz,
    encoding: &'a DataEncodingLayer,
    params: &'a [f64],
}

impl Frozen<'_> {
    /// Encodes `features` and evolves through layers 1-2, leaving `state` ready for readout.
    fn to_layer_2(&self, features: &[f64], state: &mut QuantumState) {
        self.encoding.encode_amplitude_into(features, state);
        self.ansatz.forward_l2_in_place(state, &self.params[..48]);
    }

    /// Fused readout of the 12 Z marginals and 66 ZZ correlators after the terminal Ry layer.
    fn full_readout(&self, state: &QuantumState, z: &mut [f64; N_QUBITS], zz: &mut [f64; N_PAIRS]) {
        t1000::simd::measure_fused_ry_full_readout(
            &state.real,
            &state.imag,
            &self.params[48..60],
            z,
            zz,
        );
    }
}

/// Phase 2: fused ZZ readout against applying the Ry gates and measuring ZZ directly.
/// Returns (max |fused - brute|, verified).
fn verify_fused_zz(circuit: &Frozen, test_samples: &[ZincMolecule]) -> (f64, bool) {
    print_phase(" [PHASE 2] FUSED ZZ CORRELATOR NUMERICAL VERIFICATION                  ");
    let n_verify = 50.min(test_samples.len());
    let mut max_zz_error = 0.0f64;
    let mut l2_state = QuantumState::new(DIM);
    let mut verify_state = QuantumState::new(DIM);
    let mut z_fused = [0.0; N_QUBITS];
    let mut zz_fused = [0.0; N_PAIRS];

    for sample in &test_samples[..n_verify] {
        // Method 1: Fused kernel (zero-allocation analytical)
        circuit.to_layer_2(&sample.features, &mut l2_state);
        circuit.full_readout(&l2_state, &mut z_fused, &mut zz_fused);

        // Method 2: Brute-force (apply Ry gates, then measure ZZ directly)
        circuit
            .encoding
            .encode_amplitude_into(&sample.features, &mut verify_state);
        circuit
            .ansatz
            .forward_pruned_in_place(&mut verify_state, circuit.params);
        let mut zz_brute = [0.0; N_PAIRS];
        t1000::simd::measure_zz_correlators(
            &verify_state.real,
            &verify_state.imag,
            N_QUBITS,
            &mut zz_brute,
        );

        for p in 0..N_PAIRS {
            max_zz_error = max_zz_error.max((zz_fused[p] - zz_brute[p]).abs());
        }
    }

    let zz_verified = max_zz_error < 1e-8;
    println!(
        " Verified {} molecules | Max |Fused - Brute| = {:.2e} | Status: {}",
        n_verify,
        max_zz_error,
        if zz_verified {
            "CONFIRMED"
        } else {
            "DEVIATION DETECTED"
        }
    );
    println!("{RULE}\n");
    (max_zz_error, zz_verified)
}

/// Observable vectors and targets for a set of molecules.
struct Observables {
    z12: Vec<[f64; N_QUBITS]>,
    obs78: Vec<Vec<f64>>,
    truth: Vec<(f64, f64, f64)>,
}

fn extract_observables(circuit: &Frozen, samples: &[ZincMolecule]) -> Observables {
    let mut state = QuantumState::new(DIM);
    let mut out = Observables {
        z12: Vec::with_capacity(samples.len()),
        obs78: Vec::with_capacity(samples.len()),
        truth: Vec::with_capacity(samples.len()),
    };
    for sample in samples {
        circuit.to_layer_2(&sample.features, &mut state);
        let mut z_out = [0.0; N_QUBITS];
        let mut zz_out = [0.0; N_PAIRS];
        circuit.full_readout(&state, &mut z_out, &mut zz_out);

        out.z12.push(z_out);
        let mut obs_78 = Vec::with_capacity(N_OBS);
        obs_78.extend_from_slice(&z_out);
        obs_78.extend_from_slice(&zz_out);
        out.obs78.push(obs_78);
        out.truth.push((sample.log_p, sample.qed, sample.sas));
    }
    out
}

/// Phase 3: observable vectors for the train and holdout molecules.
fn extract_all(
    circuit: &Frozen,
    train: &[ZincMolecule],
    test: &[ZincMolecule],
) -> (Observables, Observables) {
    println!("[PHASE 3] Extracting 78-dimensional observable vectors...");
    let t = Instant::now();
    let train_obs = extract_observables(circuit, train);
    let el = t.elapsed();
    println!(
        "  -> Training: {} molecules in {:.2} ms ({:.0} mols/sec)",
        train.len(),
        el.as_secs_f64() * 1000.0,
        train.len() as f64 / el.as_secs_f64()
    );

    let t = Instant::now();
    let test_obs = extract_observables(circuit, test);
    let el = t.elapsed();
    println!(
        "  -> Test:     {} molecules in {:.2} ms ({:.0} mols/sec)\n",
        test.len(),
        el.as_secs_f64() * 1000.0,
        test.len() as f64 / el.as_secs_f64()
    );
    (train_obs, test_obs)
}

/// Linear readout `b + Wᵀ o` for every row of `obs` (`n_feat` columns each).
fn predict<R: AsRef<[f64]>>(
    obs: &[R],
    w: &[Vec<f64>],
    b: &[f64],
    n_feat: usize,
) -> Vec<[f64; N_PROPS]> {
    obs.iter()
        .map(|row| {
            let row = row.as_ref();
            let mut pred = [0.0; N_PROPS];
            for p in 0..N_PROPS {
                pred[p] = b[p];
                for f in 0..n_feat {
                    pred[p] += w[f][p] * row[f];
                }
            }
            pred
        })
        .collect()
}

/// Everything a readout head produced.
struct Head {
    w: Vec<Vec<f64>>,
    b: Vec<f64>,
    preds: Vec<[f64; N_PROPS]>,
    mae: [f64; 3],
    r2: [f64; 3],
}

/// Phases 4 and 5: fit a ridge head on `train_x` and score it on the holdout set.
fn fit_head<R: AsRef<[f64]>>(
    fit_label: &str,
    (train_x, train_y): (&[Vec<f64>], &[Vec<f64>]),
    test_x: &[R],
    n_feat: usize,
    truth: &[(f64, f64, f64)],
) -> Head {
    let t = Instant::now();
    let (w, b) = ridge_regression_multi(train_x, train_y, n_feat, N_PROPS, LAMBDA);
    println!(
        "  -> {fit_label} ridge fit: {:.2} ms",
        t.elapsed().as_secs_f64() * 1000.0
    );
    let preds = predict(test_x, &w, &b, n_feat);
    let (mae, r2) = mae_r2_per_property(&preds, truth);
    Head {
        w,
        b,
        preds,
        mae,
        r2,
    }
}

/// Phase 7: molecules per second reading 12 vs 78 observables (3 passes over the holdout set).
fn throughput_benchmark(circuit: &Frozen, test_samples: &[ZincMolecule]) -> (f64, f64) {
    println!("[PHASE 7] Throughput Benchmark: Baseline vs Enhanced Readout...");
    let repeats = 3;
    let mut state = QuantumState::new(DIM);

    let t = Instant::now();
    for _ in 0..repeats {
        for sample in test_samples {
            circuit.to_layer_2(&sample.features, &mut state);
            let mut z_b = [0.0; N_QUBITS];
            t1000::simd::measure_fused_ry_z_readout(
                &state.real,
                &state.imag,
                &circuit.params[48..60],
                &mut z_b,
            );
        }
    }
    let tp_b = (test_samples.len() * repeats) as f64 / t.elapsed().as_secs_f64();

    let t = Instant::now();
    for _ in 0..repeats {
        for sample in test_samples {
            circuit.to_layer_2(&sample.features, &mut state);
            circuit.full_readout(&state, &mut [0.0; N_QUBITS], &mut [0.0; N_PAIRS]);
        }
    }
    let tp_e = (test_samples.len() * repeats) as f64 / t.elapsed().as_secs_f64();

    println!("  Baseline (12-obs) : {:.0} mols/sec", tp_b);
    println!("  Enhanced (78-obs) : {:.0} mols/sec", tp_e);
    println!(
        "  ZZ Overhead       : {:.1}%\n",
        (1.0 - tp_e / tp_b) * 100.0
    );
    (tp_b, tp_e)
}

/// Phase 1-2 inputs: the train and test molecules and the frozen circuit parameters.
fn load_inputs() -> Option<(Vec<ZincMolecule>, Vec<ZincMolecule>, Vec<f64>)> {
    println!("[INIT] Loading training set...");
    let train_samples = load_with_fallback(
        "tests/benchmarks/zinc_molecular/data/zinc_train_10k.csv",
        "tests/benchmarks/zinc_molecular/outputs/zinc_train_10k.csv",
        10_000,
    )?;
    println!("  -> Loaded {} training molecules.\n", train_samples.len());

    println!("[INIT] Loading test set...");
    let test_samples = load_with_fallback(
        "tests/benchmarks/zinc_molecular/data/zinc_test_2k.csv",
        "tests/benchmarks/zinc_molecular/outputs/zinc_test_2k.csv",
        2_000,
    )?;
    println!("  -> Loaded {} test molecules.\n", test_samples.len());

    let circuit_params = load_circuit_params()?;
    Some((train_samples, test_samples, circuit_params))
}

/// Phase 4: fit the 12-observable baseline head and the 78-observable enhanced head.
fn fit_both_heads(train_obs: &Observables, test_obs: &Observables) -> (Head, Head) {
    print_phase(" [PHASE 4] RIDGE REGRESSION: BASELINE (12 Observables) vs ENHANCED (78) ");
    let train_y: Vec<Vec<f64>> = train_obs
        .truth
        .iter()
        .map(|&(l, q, s)| vec![l, q, s])
        .collect();
    let train_x_12: Vec<Vec<f64>> = train_obs.z12.iter().map(|z| z.to_vec()).collect();

    println!(
        "\n[TRAINING] Baseline readout head: {} features → {} targets...",
        N_QUBITS, N_PROPS
    );
    let base = fit_head(
        "Baseline",
        (&train_x_12, &train_y),
        &test_obs.z12,
        N_QUBITS,
        &test_obs.truth,
    );
    print_results_table("BASELINE RESULTS (12 Observables)", &base.mae, &base.r2);

    println!(
        "[TRAINING] Enhanced readout head: {} features → {} targets...",
        N_OBS, N_PROPS
    );
    let enh = fit_head(
        "Enhanced",
        (&train_obs.obs78, &train_y),
        &test_obs.obs78,
        N_OBS,
        &test_obs.truth,
    );
    print_results_table(
        "ENHANCED RESULTS (78 Observables: 12 ⟨Z⟩ + 66 ⟨ZZ⟩)",
        &enh.mae,
        &enh.r2,
    );
    (base, enh)
}

fn main() {
    print_banner();
    let Some((train_samples, test_samples, circuit_params)) = load_inputs() else {
        return;
    };
    let encoding = DataEncodingLayer::new(N_QUBITS, EncodingKind::Amplitude);
    let ansatz = CircuitAnsatz::new(N_QUBITS, 3).with_pi_layer(true);
    let circuit = Frozen {
        ansatz: &ansatz,
        encoding: &encoding,
        params: &circuit_params,
    };

    let (max_zz_error, zz_verified) = verify_fused_zz(&circuit, &test_samples);
    let (train_obs, test_obs) = extract_all(&circuit, &train_samples, &test_samples);
    let (base, enh) = fit_both_heads(&train_obs, &test_obs);

    let imp = improvement_pct(&base.mae, &enh.mae);
    print_gain_ledger(&base, &enh, &imp);
    let throughput = throughput_benchmark(&circuit, &test_samples);
    print_top_corrections(&test_samples, &base, &enh);
    print_zz_heatmap(&enh.w);

    let results = Results {
        train_len: train_samples.len(),
        test_len: test_samples.len(),
        max_zz_error,
        zz_verified,
        base: &base,
        enh: &enh,
        imp,
        throughput,
    };
    save_outputs(&train_samples, &test_samples, &circuit_params, &results);
}
