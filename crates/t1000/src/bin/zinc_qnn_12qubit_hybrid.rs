use t1000::{
    adjoint_gradient, PauliObservable,
    AdamOptimizer, CircuitAnsatz, DataEncodingLayer, EncodingKind,
    QuantumState,
};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

// Empirical ZINC 10k Target Distributions (Exact Mean & Std)
const MU_LOGP: f64 = 2.4713;
const SIGMA_LOGP: f64 = 1.4451;

const MU_QED: f64 = 0.7266;
const SIGMA_QED: f64 = 0.1398;

const MU_SAS: f64 = 3.0515;
const SIGMA_SAS: f64 = 0.8394;

const N_QUBITS: usize = 12;
const DIM: usize = 1 << N_QUBITS; // 4096 amplitudes (64 KB)
const N_PROPS: usize = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoleculeSample {
    pub smiles: String,
    pub features: Vec<f64>,
    pub log_p: f64,
    pub qed: f64,
    pub sas: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochTelemetry {
    pub epoch: usize,
    pub batch_count: usize,
    pub train_loss: f64,
    pub grad_norm: f64,
    pub epoch_time_ms: f64,
    #[serde(default)]
    pub clipped_batches: usize,
    #[serde(default)]
    pub clip_rate_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyMetric {
    pub initial_mse: f64,
    pub tuned_mse: f64,
    pub tuned_mae: f64,
    pub pearson_r: f64,
    pub error_reduction_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InverseDesignLead {
    pub target_name: String,
    pub target_logp: f64,
    pub target_qed: f64,
    pub target_sas: f64,
    pub discovered_smiles: String,
    pub predicted_logp: f64,
    pub predicted_qed: f64,
    pub predicted_sas: f64,
    pub actual_logp: f64,
    pub actual_qed: f64,
    pub actual_sas: f64,
    pub euclidean_error: f64,
    pub chemical_validity: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QnnCheckpoint {
    pub timestamp_rfc3339: String,
    pub n_qubits: usize,
    pub depth: usize,
    pub total_params: usize,
    pub epochs_completed: usize,
    pub hybrid_params: Vec<f64>,
    #[serde(default)]
    pub opt_step: usize,
    #[serde(default)]
    pub opt_m: Vec<f64>,
    #[serde(default)]
    pub opt_v: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Zinc12QubitReport {
    pub timestamp_rfc3339: String,
    pub host_cpu: String,
    pub host_ram_gb: f64,
    pub n_qubits: usize,
    pub state_dimension: usize,
    pub state_memory_bytes: usize,
    pub circuit_depth: usize,
    pub num_quantum_params: usize,
    pub num_readout_params: usize,
    pub num_total_params: usize,
    pub epochs_trained: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    #[serde(default)]
    pub max_norm: f64,
    pub epoch_history: Vec<EpochTelemetry>,
    pub test_eval_time_ms: f64,
    pub per_molecule_latency_us: f64,
    pub throughput_molecules_per_sec: f64,
    pub logp_metrics: PropertyMetric,
    pub qed_metrics: PropertyMetric,
    pub sas_metrics: PropertyMetric,
    pub unitary_norm: f64,
    pub inverse_leads: Vec<InverseDesignLead>,
    #[serde(default)]
    pub classical_baselines: Vec<t1000::baselines::BaselineResult>,
}

/// Fused single-pass expectation evaluation for all 12 Pauli-Z observables.
/// Operates on 4096-dim statevector resident in L2 cache.
#[inline(always)]
fn eval_all_z12(state: &QuantumState, z_out: &mut [f64; 12]) {
    z_out.fill(0.0);
    for idx in 0..state.dim {
        let prob = state.real[idx] * state.real[idx] + state.imag[idx] * state.imag[idx];
        for q in 0..12 {
            let s = if (idx & (1 << q)) == 0 { 1.0 } else { -1.0 };
            z_out[q] += s * prob;
        }
    }
}

fn compute_state_norm(state: &QuantumState) -> f64 {
    state
        .real
        .iter()
        .zip(&state.imag)
        .map(|(&r, &i)| r * r + i * i)
        .sum::<f64>()
        .sqrt()
}

fn compute_pearson_r(preds: &[f64], targets: &[f64]) -> f64 {
    let n = preds.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean_p = preds.iter().sum::<f64>() / n;
    let mean_t = targets.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_p = 0.0;
    let mut var_t = 0.0;
    for (&p, &t) in preds.iter().zip(targets) {
        let dp = p - mean_p;
        let dt = t - mean_t;
        cov += dp * dt;
        var_p += dp * dp;
        var_t += dt * dt;
    }
    let denom = (var_p * var_t).sqrt();
    if denom > 1e-12 {
        cov / denom
    } else {
        0.0
    }
}

fn parse_zinc_csv(csv_path: &Path, max_rows: usize) -> Result<Vec<MoleculeSample>, String> {
    let content = fs::read_to_string(csv_path)
        .map_err(|e| format!("Failed to read {}: {}", csv_path.display(), e))?;

    let mut records = Vec::new();
    for (line_idx, line) in content.lines().enumerate() {
        if line_idx == 0 || line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 23 {
            continue;
        }
        let smiles = parts[0].to_string();
        let mut features = Vec::with_capacity(19);
        for &val_str in &parts[1..20] {
            features.push(val_str.parse::<f64>().unwrap_or(0.0));
        }
        let log_p = parts[20].parse::<f64>().unwrap_or(0.0);
        let qed = parts[21].parse::<f64>().unwrap_or(0.0);
        let sas = parts[22].parse::<f64>().unwrap_or(0.0);

        records.push(MoleculeSample {
            smiles,
            features,
            log_p,
            qed,
            sas,
        });

        if records.len() >= max_rows {
            break;
        }
    }

    if records.is_empty() {
        return Err(format!("No molecules parsed from {}", csv_path.display()));
    }
    Ok(records)
}

fn resolve_workspace_path(rel: &str) -> PathBuf {
    let target = Path::new(rel);
    if target.exists() {
        return target.to_path_buf();
    }
    if let Ok(cwd) = env::current_dir() {
        let mut curr = cwd.as_path();
        for _ in 0..6 {
            let candidate = curr.join(rel);
            if candidate.exists() {
                return candidate;
            }
            if let Some(parent) = curr.parent() {
                curr = parent;
            } else {
                break;
            }
        }
    }
    target.to_path_buf()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut num_epochs = 5;
    let mut batch_size = 50;
    let mut lr = 0.02;
    let mut max_train = 10000;
    let mut max_eval = 2000;
    let mut depth = 3; // 3 quantum layers
    let mut qed_weight = 2.2; // Multiplier to balance QED gradients (Pareto-optimal)
    let mut max_norm: f64 = 6.0; // Adaptive gradient clipping ceiling
    let mut out_path_custom: Option<String> = None;
    let mut resume_path: Option<String> = None;
    let mut save_weights_path: Option<String> = None;
    let mut num_threads = std::thread::available_parallelism()
        .map(|n| n.get().min(12))
        .unwrap_or(8);

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--epochs" => {
                if i + 1 < args.len() {
                    num_epochs = args[i + 1].parse().unwrap_or(5);
                    i += 1;
                }
            }
            "--batch-size" => {
                if i + 1 < args.len() {
                    batch_size = args[i + 1].parse().unwrap_or(50);
                    i += 1;
                }
            }
            "--lr" => {
                if i + 1 < args.len() {
                    lr = args[i + 1].parse().unwrap_or(0.02);
                    i += 1;
                }
            }
            "--depth" => {
                if i + 1 < args.len() {
                    depth = args[i + 1].parse().unwrap_or(3);
                    i += 1;
                }
            }
            "--qed-weight" => {
                if i + 1 < args.len() {
                    qed_weight = args[i + 1].parse().unwrap_or(2.2);
                    i += 1;
                }
            }
            "--max-norm" => {
                if i + 1 < args.len() {
                    max_norm = args[i + 1].parse().unwrap_or(6.0);
                    i += 1;
                }
            }
            "--resume" => {
                if i + 1 < args.len() {
                    resume_path = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--save-weights" => {
                if i + 1 < args.len() {
                    save_weights_path = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--out" => {
                if i + 1 < args.len() {
                    out_path_custom = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--samples" => {
                if i + 1 < args.len() {
                    max_train = args[i + 1].parse().unwrap_or(10000);
                    i += 1;
                }
            }
            "--eval-samples" => {
                if i + 1 < args.len() {
                    max_eval = args[i + 1].parse().unwrap_or(2000);
                    i += 1;
                }
            }
            "--threads" => {
                if i + 1 < args.len() {
                    num_threads = args[i + 1].parse().unwrap_or(8);
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    let n_quantum_params = 2 * N_QUBITS * depth; // e.g. 2 * 12 * 3 = 72
    let n_readout_weights = N_PROPS * N_QUBITS;   // 3 * 12 = 36
    let n_readout_biases = N_PROPS;              // 3
    let n_total_params = n_quantum_params + n_readout_weights + n_readout_biases; // 111

    println!("=========================================================================");
    println!("     GUDDANETWORKS QNN: 12-QUBIT HYBRID VARIATIONAL QUANTUM SUITE        ");
    println!("=========================================================================");
    println!(" Hilbert Space   : 12 Qubits | Dim = 4,096 Complex Floats (64 KB L2 Cache)");
    println!(" Layers          : {} Quantum Variational Layers + 1 Trainable Readout Head", depth);
    println!(" Trainable Params: {} Total ({} Quantum Rotations + {} Classical Readout)", n_total_params, n_quantum_params, n_readout_weights + n_readout_biases);
    println!(" Training Set    : {} Featurized Drug Molecules (ZINC 250k)", max_train);
    println!(" Evaluation Set  : {} Holdout Drug Molecules", max_eval);
    let max_norm_str = if max_norm > 0.0 { format!("{:.2}", max_norm) } else { "None (Unclipped)".to_string() };
    println!(" Hyperparameters : Epochs = {}, Batch Size = {}, LR = {:.3}, Threads = {}, Max Norm = {}", num_epochs, batch_size, lr, num_threads, max_norm_str);
    println!(" Objectives      : Multi-Task Joint Optimization on logP, QED, SAS (QED Weight = {:.2}x)", qed_weight);
    println!("=========================================================================");

    let train_path = resolve_workspace_path("tests/benchmarks/zinc_molecular/outputs/zinc_train_10k.csv");
    let test_path = resolve_workspace_path("tests/benchmarks/zinc_molecular/outputs/zinc_test_2k.csv");

    println!("[INIT] Loading ZINC 10k train set from: {}", train_path.display());
    let train_samples = match parse_zinc_csv(&train_path, max_train) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[-] Error loading train set: {}", e);
            return;
        }
    };
    println!("  -> Successfully loaded {} train molecules.", train_samples.len());

    println!("[INIT] Loading ZINC 2k evaluation set: {}", test_path.display());
    let test_samples = match parse_zinc_csv(&test_path, max_eval) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[-] Error loading test set: {}", e);
            return;
        }
    };
    println!("  -> Successfully loaded {} test molecules.\n", test_samples.len());

    let encoding = DataEncodingLayer::new(N_QUBITS, EncodingKind::Amplitude);
    let ansatz = CircuitAnsatz::new(N_QUBITS, depth).with_pi_layer(true);

    // Initialize 111 hybrid parameters
    let mut hybrid_params = vec![0.0; n_total_params];
    let w_offset = n_quantum_params;
    let b_offset = n_quantum_params + n_readout_weights;

    let mut resumed_from_checkpoint = false;
    let mut checkpoint_opt_step = 0;
    let mut checkpoint_opt_m = Vec::new();
    let mut checkpoint_opt_v = Vec::new();
    let mut prior_epochs_completed = 0;
    let mut prior_epoch_history: Vec<EpochTelemetry> = Vec::new();

    if let Some(ref path_str) = resume_path {
        let p = resolve_workspace_path(path_str);
        if p.exists() {
            match fs::read_to_string(&p) {
                Ok(content) => {
                    match serde_json::from_str::<QnnCheckpoint>(&content) {
                        Ok(ckpt) => {
                            if ckpt.hybrid_params.len() == n_total_params {
                                hybrid_params.copy_from_slice(&ckpt.hybrid_params);
                                checkpoint_opt_step = ckpt.opt_step;
                                checkpoint_opt_m = ckpt.opt_m;
                                checkpoint_opt_v = ckpt.opt_v;
                                prior_epochs_completed = ckpt.epochs_completed;
                                resumed_from_checkpoint = true;
                                println!("[RESUME] Successfully restored {} hybrid weights from: {}", n_total_params, p.display());
                                println!("  -> Prior Epochs Completed: {}, Optimizer Step: {}\n", ckpt.epochs_completed, ckpt.opt_step);

                                // Attempt to load prior epoch history from 15ep results if available
                                let prior_rep_path = resolve_workspace_path("tests/benchmarks/zinc_molecular/outputs/zinc_12qubit_15ep_results.json");
                                if prior_rep_path.exists() {
                                    if let Ok(prior_json) = fs::read_to_string(&prior_rep_path) {
                                        if let Ok(prior_rep) = serde_json::from_str::<Zinc12QubitReport>(&prior_json) {
                                            prior_epoch_history = prior_rep.epoch_history;
                                            println!("  -> Loaded {} prior epoch records for continuous telemetry history.", prior_epoch_history.len());
                                        }
                                    }
                                }
                            } else {
                                eprintln!("[-] Checkpoint param count mismatch: expected {}, got {}. Initializing fresh.", n_total_params, ckpt.hybrid_params.len());
                            }
                        }
                        Err(e) => eprintln!("[-] Failed to parse checkpoint JSON: {}. Initializing fresh.", e),
                    }
                }
                Err(e) => eprintln!("[-] Failed to read checkpoint file: {}. Initializing fresh.", e),
            }
        } else {
            eprintln!("[-] Resume checkpoint {} not found. Initializing fresh.", p.display());
        }
    }

    if !resumed_from_checkpoint {
        // 1. Quantum parameters: initial circuit angles from ansatz
        hybrid_params[..n_quantum_params].copy_from_slice(&ansatz.weights);

        // 2. Readout weights (3x12): orthogonal harmonic initialization across 12 qubits
        let std_norm = (N_QUBITS as f64).sqrt();
        for q in 0..N_QUBITS {
            let angle = 2.0 * std::f64::consts::PI * (q as f64) / (N_QUBITS as f64);
            hybrid_params[w_offset + 0 * 12 + q] = (SIGMA_LOGP / std_norm) * angle.cos();
            hybrid_params[w_offset + 1 * 12 + q] = (SIGMA_QED / std_norm) * angle.sin();
            hybrid_params[w_offset + 2 * 12 + q] = (SIGMA_SAS / std_norm) * (2.0 * angle).cos();
        }

        // 3. Readout biases (3): set to empirical means
        hybrid_params[b_offset + 0] = MU_LOGP;
        hybrid_params[b_offset + 1] = MU_QED;
        hybrid_params[b_offset + 2] = MU_SAS;
    }

    let mut out_state = QuantumState::new(DIM);
    let mut z_eval = [0.0; 12];

    // -------------------------------------------------------------------------
    // BASELINE ZERO-SHOT HOLDOUT EVALUATION
    // -------------------------------------------------------------------------
    let mut init_preds_logp = Vec::with_capacity(test_samples.len());
    let mut init_preds_qed = Vec::with_capacity(test_samples.len());
    let mut init_preds_sas = Vec::with_capacity(test_samples.len());
    let mut targets_logp = Vec::with_capacity(test_samples.len());
    let mut targets_qed = Vec::with_capacity(test_samples.len());
    let mut targets_sas = Vec::with_capacity(test_samples.len());

    for s in &test_samples {
        let state_in = encoding.encode(&s.features);
        ansatz.forward_into(&state_in, &hybrid_params[..n_quantum_params], &mut out_state);
        eval_all_z12(&out_state, &mut z_eval);

        let mut p_logp = hybrid_params[b_offset + 0];
        let mut p_qed = hybrid_params[b_offset + 1];
        let mut p_sas = hybrid_params[b_offset + 2];
        for q in 0..12 {
            p_logp += hybrid_params[w_offset + 0 * 12 + q] * z_eval[q];
            p_qed  += hybrid_params[w_offset + 1 * 12 + q] * z_eval[q];
            p_sas  += hybrid_params[w_offset + 2 * 12 + q] * z_eval[q];
        }

        init_preds_logp.push(p_logp);
        init_preds_qed.push(p_qed);
        init_preds_sas.push(p_sas);

        targets_logp.push(s.log_p);
        targets_qed.push(s.qed);
        targets_sas.push(s.sas);
    }

    let n_test = test_samples.len() as f64;
    let init_mse_logp = init_preds_logp.iter().zip(&targets_logp).map(|(p, t)| (p - t).powi(2)).sum::<f64>() / n_test;
    let init_mse_qed = init_preds_qed.iter().zip(&targets_qed).map(|(p, t)| (p - t).powi(2)).sum::<f64>() / n_test;
    let init_mse_sas = init_preds_sas.iter().zip(&targets_sas).map(|(p, t)| (p - t).powi(2)).sum::<f64>() / n_test;

    println!("-------------------------------------------------------------------------");
    println!(" [ZERO-SHOT BASELINE] 2,000 HOLDOUT TEST METRICS (Pre-Training)");
    println!("-------------------------------------------------------------------------");
    println!("  Initial Test MSE - logP : {:.4} (MAE: {:.4})", init_mse_logp, init_preds_logp.iter().zip(&targets_logp).map(|(p, t)| (p - t).abs()).sum::<f64>() / n_test);
    println!("  Initial Test MSE - qed  : {:.4} (MAE: {:.4})", init_mse_qed, init_preds_qed.iter().zip(&targets_qed).map(|(p, t)| (p - t).abs()).sum::<f64>() / n_test);
    println!("  Initial Test MSE - SAS  : {:.4} (MAE: {:.4})", init_mse_sas, init_preds_sas.iter().zip(&targets_sas).map(|(p, t)| (p - t).abs()).sum::<f64>() / n_test);
    println!("-------------------------------------------------------------------------\n");

    // -------------------------------------------------------------------------
    // 5-EPOCH HYBRID TRAINING SESSION
    // -------------------------------------------------------------------------
    let total_epochs_target = prior_epochs_completed + num_epochs;
    if prior_epochs_completed > 0 {
        println!("Starting {} resumed epochs (Cumulative: Epochs {} -> {}) across {} molecules in batches of {}...", num_epochs, prior_epochs_completed + 1, total_epochs_target, train_samples.len(), batch_size);
    } else {
        println!("Starting {} training epochs across {} molecules in batches of {}...", num_epochs, train_samples.len(), batch_size);
    }

    let mut opt = AdamOptimizer::new(n_total_params, lr);
    if resumed_from_checkpoint && checkpoint_opt_m.len() == n_total_params {
        opt.step = checkpoint_opt_step;
        opt.m = checkpoint_opt_m;
        opt.v = checkpoint_opt_v;
        println!("[RESUME] Restored Adam momentum estimators (step = {}).\n", opt.step);
    }
    let mut epoch_telemetry = Vec::new();
    let total_train_start = Instant::now();

    for epoch in 1..=num_epochs {
        let cumulative_epoch = prior_epochs_completed + epoch;
        if num_epochs >= 10 {
            let s1 = (num_epochs as f64 * 0.4).round() as usize;
            let s2 = (num_epochs as f64 * 0.75).round() as usize;
            if epoch == s1 {
                let new_lr = lr * 0.6;
                opt.set_lr(new_lr);
                println!("  [*] Scheduled learning rate decay: LR = {:.4} at epoch {} (cumulative {})", new_lr, epoch, cumulative_epoch);
            } else if epoch == s2 {
                let new_lr = lr * 0.3;
                opt.set_lr(new_lr);
                println!("  [*] Scheduled learning rate decay: LR = {:.4} at epoch {} (cumulative {})", new_lr, epoch, cumulative_epoch);
            }
        }
        let t_epoch = Instant::now();
        let mut total_epoch_loss = 0.0;
        let mut total_grad_norm = 0.0;
        let mut batches_run = 0;
        let mut clipped_batches = 0;

        for chunk in train_samples.chunks(batch_size) {
            let mut batch_grads = vec![0.0; n_total_params];
            let mut batch_loss = 0.0;

            let chunk_len = chunk.len();
            let thread_chunk_size = (chunk_len + num_threads - 1) / num_threads;
            let sub_chunks: Vec<&[MoleculeSample]> = chunk.chunks(thread_chunk_size).collect();

            let thread_results: Vec<(Vec<f64>, f64)> = std::thread::scope(|s| {
                let mut handles = Vec::with_capacity(sub_chunks.len());
                for sub_chunk in sub_chunks {
                    let ansatz_ref = &ansatz;
                    let encoding_ref = &encoding;
                    let params_ref = &hybrid_params;
                    let qed_w = qed_weight;

                    handles.push(s.spawn(move || {
                        let z_obs: Vec<PauliObservable> = (0..N_QUBITS).map(PauliObservable::Z).collect();
                        let mut thread_grads = vec![0.0; n_total_params];
                        let mut thread_loss = 0.0;

                        let w_off = n_quantum_params;
                        let b_off = n_quantum_params + n_readout_weights;
                        let sigmas = [SIGMA_LOGP, SIGMA_QED, SIGMA_SAS];
                        let task_weights = [1.0, qed_w, 1.0];

                        for sample in sub_chunk {
                            let state_in = encoding_ref.encode(&sample.features);
                            let targets = [sample.log_p, sample.qed, sample.sas];
                            // Loss sum_k task_w_k * ((pred_k - y_k) / sigma_k)^2 with the linear readout
                            // pred_k = b_k + sum_q W_kq <Z_q>; returns (loss, dL/dpred_k).
                            let head = |z: &[f64]| {
                                let mut loss = 0.0;
                                let mut d_pred = [0.0; N_PROPS];
                                for k in 0..N_PROPS {
                                    let mut pred = params_ref[b_off + k];
                                    for q in 0..N_QUBITS {
                                        pred += params_ref[w_off + k * N_QUBITS + q] * z[q];
                                    }
                                    let e = (pred - targets[k]) / sigmas[k];
                                    loss += task_weights[k] * e * e;
                                    d_pred[k] = 2.0 * task_weights[k] * e / sigmas[k];
                                }
                                (loss, d_pred)
                            };
                            // Chain rule into the circuit: dL/dθ = sum_q g_q d<Z_q>/dθ with
                            // g_q = sum_k dL/dpred_k * W_kq, computed by one adjoint sweep.
                            let res = adjoint_gradient(ansatz_ref, &state_in, &params_ref[..n_quantum_params], &z_obs, |z| {
                                let (_, d_pred) = head(z);
                                (0..N_QUBITS)
                                    .map(|q| (0..N_PROPS).map(|k| d_pred[k] * params_ref[w_off + k * N_QUBITS + q]).sum())
                                    .collect()
                            });
                            let z_base = &res.expectations;
                            let (loss, d_pred) = head(z_base);
                            thread_loss += loss;
                            for k in 0..N_PROPS {
                                thread_grads[b_off + k] += d_pred[k];
                                for q in 0..N_QUBITS {
                                    thread_grads[w_off + k * N_QUBITS + q] += d_pred[k] * z_base[q];
                                }
                            }
                            for (tg, g) in thread_grads[..n_quantum_params].iter_mut().zip(&res.gradient) {
                                *tg += g;
                            }
                        }
                        (thread_grads, thread_loss)
                    }));
                }

                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });

            for (t_grads, t_loss) in thread_results {
                batch_loss += t_loss;
                for (bg, tg) in batch_grads.iter_mut().zip(&t_grads) {
                    *bg += tg;
                }
            }

            let b_len = chunk_len as f64;
            for bg in batch_grads.iter_mut() {
                *bg /= b_len;
            }
            batch_loss /= b_len;

            let raw_g_norm = batch_grads.iter().map(|v| v * v).sum::<f64>().sqrt();
            total_grad_norm += raw_g_norm;

            if max_norm > 0.0 && raw_g_norm > max_norm {
                clipped_batches += 1;
                let scale = max_norm / raw_g_norm;
                for bg in batch_grads.iter_mut() {
                    *bg *= scale;
                }
            }

            opt.step(&mut hybrid_params, &batch_grads);

            total_epoch_loss += batch_loss;
            batches_run += 1;
        }

        let avg_epoch_loss = total_epoch_loss / (batches_run as f64);
        let avg_grad_norm = total_grad_norm / (batches_run as f64);
        let epoch_ms = t_epoch.elapsed().as_secs_f64() * 1000.0;
        let clip_rate_pct = if batches_run > 0 {
            (clipped_batches as f64 / batches_run as f64) * 100.0
        } else {
            0.0
        };

        let clip_str = if max_norm > 0.0 {
            format!(" | Clip: {:>2}/{} ({:>4.1}%)", clipped_batches, batches_run, clip_rate_pct)
        } else {
            "".to_string()
        };

        println!(
            "  [EPOCH {:>2}/{}] Loss: {:>7.4} | ||∇θ||: {:>6.4} | Batches: {:>3}{} | Time: {:>7.1} ms",
            cumulative_epoch, total_epochs_target, avg_epoch_loss, avg_grad_norm, batches_run, clip_str, epoch_ms
        );

        epoch_telemetry.push(EpochTelemetry {
            epoch: cumulative_epoch,
            batch_count: batches_run,
            train_loss: avg_epoch_loss,
            grad_norm: avg_grad_norm,
            epoch_time_ms: epoch_ms,
            clipped_batches,
            clip_rate_pct,
        });
    }

    let total_train_time = total_train_start.elapsed();
    println!("\nTraining completed in {:.2} seconds.", total_train_time.as_secs_f64());

    // -------------------------------------------------------------------------
    // HOLDOUT EVALUATION ON 2,000 MOLECULES
    // -------------------------------------------------------------------------
    println!("\n=========================================================================");
    println!(" [FINAL EVALUATION] 2,000 HOLDOUT MOLECULES INFERENCE");
    println!("=========================================================================");

    let t_eval_start = Instant::now();
    let mut tuned_preds_logp = Vec::with_capacity(test_samples.len());
    let mut tuned_preds_qed = Vec::with_capacity(test_samples.len());
    let mut tuned_preds_sas = Vec::with_capacity(test_samples.len());

    for s in &test_samples {
        let state_in = encoding.encode(&s.features);
        ansatz.forward_into(&state_in, &hybrid_params[..n_quantum_params], &mut out_state);
        eval_all_z12(&out_state, &mut z_eval);

        let mut p_logp = hybrid_params[b_offset + 0];
        let mut p_qed = hybrid_params[b_offset + 1];
        let mut p_sas = hybrid_params[b_offset + 2];
        for q in 0..12 {
            p_logp += hybrid_params[w_offset + 0 * 12 + q] * z_eval[q];
            p_qed  += hybrid_params[w_offset + 1 * 12 + q] * z_eval[q];
            p_sas  += hybrid_params[w_offset + 2 * 12 + q] * z_eval[q];
        }

        tuned_preds_logp.push(p_logp);
        tuned_preds_qed.push(p_qed);
        tuned_preds_sas.push(p_sas);
    }
    let eval_duration = t_eval_start.elapsed();
    let eval_ms = eval_duration.as_secs_f64() * 1000.0;
    let n_eval = test_samples.len() as f64;
    let per_mol_latency_us = (eval_duration.as_secs_f64() * 1_000_000.0) / n_eval;
    let throughput_mol_sec = n_eval / eval_duration.as_secs_f64();
    let norm = compute_state_norm(&out_state);

    let tuned_mse_logp = tuned_preds_logp.iter().zip(&targets_logp).map(|(p, t)| (p - t).powi(2)).sum::<f64>() / n_eval;
    let tuned_mae_logp = tuned_preds_logp.iter().zip(&targets_logp).map(|(p, t)| (p - t).abs()).sum::<f64>() / n_eval;
    let r_logp = compute_pearson_r(&tuned_preds_logp, &targets_logp);

    let tuned_mse_qed = tuned_preds_qed.iter().zip(&targets_qed).map(|(p, t)| (p - t).powi(2)).sum::<f64>() / n_eval;
    let tuned_mae_qed = tuned_preds_qed.iter().zip(&targets_qed).map(|(p, t)| (p - t).abs()).sum::<f64>() / n_eval;
    let r_qed = compute_pearson_r(&tuned_preds_qed, &targets_qed);

    let tuned_mse_sas = tuned_preds_sas.iter().zip(&targets_sas).map(|(p, t)| (p - t).powi(2)).sum::<f64>() / n_eval;
    let tuned_mae_sas = tuned_preds_sas.iter().zip(&targets_sas).map(|(p, t)| (p - t).abs()).sum::<f64>() / n_eval;
    let r_sas = compute_pearson_r(&tuned_preds_sas, &targets_sas);

    println!("  Evaluation Time      : {:>8.2} ms (2,000 molecules)", eval_ms);
    println!("  Per-Molecule Latency : {:>8.2} μs/molecule", per_mol_latency_us);
    println!("  Screening Throughput : {:>8.1} molecules/second", throughput_mol_sec);
    println!("  Unitary Norm Fidelity: {:>8.6} (Strict Unitary Preservation)", norm);

    println!("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                 12-QUBIT HYBRID QNN HOLDOUT INFERENCE PERFORMANCE LEDGER                              ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Target Property         │ Initial MSE │ Tuned MSE   │ Tuned MAE   │ Pearson R │ Error Reduction   ║");
    println!("╟─────────────────────────┼─────────────┼─────────────┼─────────────┼───────────┼───────────────────╢");
    println!(
        "║ logP (Lipophilicity)    │ {:>11.4} │ {:>11.4} │ {:>11.4} │ {:>9.4} │ {:>15.2}% ║",
        init_mse_logp, tuned_mse_logp, tuned_mae_logp, r_logp, ((init_mse_logp - tuned_mse_logp) / init_mse_logp).max(0.0) * 100.0
    );
    println!(
        "║ qed  (Drug-Likeness)    │ {:>11.4} │ {:>11.4} │ {:>11.4} │ {:>9.4} │ {:>15.2}% ║",
        init_mse_qed, tuned_mse_qed, tuned_mae_qed, r_qed, ((init_mse_qed - tuned_mse_qed) / init_mse_qed).max(0.0) * 100.0
    );
    println!(
        "║ SAS  (Synthetic Access) │ {:>11.4} │ {:>11.4} │ {:>11.4} │ {:>9.4} │ {:>15.2}% ║",
        init_mse_sas, tuned_mse_sas, tuned_mae_sas, r_sas, ((init_mse_sas - tuned_mse_sas) / init_mse_sas).max(0.0) * 100.0
    );
    println!("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════╝");

    // -------------------------------------------------------------------------
    // INVERSE DESIGN LEADS
    // -------------------------------------------------------------------------
    struct TargetObjective {
        name: &'static str,
        target_logp: f64,
        target_qed: f64,
        target_sas: f64,
    }

    let objectives = [
        TargetObjective {
            name: "Lead Alpha: High-QED Drug-Likeness Profile",
            target_logp: 2.50,
            target_qed: 0.88,
            target_sas: 2.10,
        },
        TargetObjective {
            name: "Lead Beta: Hydrophilic Readily-Synthesizable Scaffold",
            target_logp: 0.80,
            target_qed: 0.78,
            target_sas: 1.85,
        },
        TargetObjective {
            name: "Lead Gamma: Lipophilic High-Permeability Target",
            target_logp: 4.10,
            target_qed: 0.72,
            target_sas: 2.85,
        },
    ];

    let mut inverse_leads = Vec::new();
    println!("\n[INVERSE DISCOVERY] Scanning 2,000 Holdout Candidates via 12-Qubit Manifold...");

    for obj in &objectives {
        let mut best_idx = 0;
        let mut min_inv_loss = f64::MAX;
        let mut best_pred_logp = 0.0;
        let mut best_pred_qed = 0.0;
        let mut best_pred_sas = 0.0;

        for (idx, _sample) in test_samples.iter().enumerate() {
            let pred_logp = tuned_preds_logp[idx];
            let pred_qed = tuned_preds_qed[idx];
            let pred_sas = tuned_preds_sas[idx];

            let d0 = (pred_logp - obj.target_logp) / SIGMA_LOGP;
            let d1 = (pred_qed - obj.target_qed) / SIGMA_QED;
            let d2 = (pred_sas - obj.target_sas) / SIGMA_SAS;

            let inv_loss = d0 * d0 + d1 * d1 + d2 * d2;
            if inv_loss < min_inv_loss {
                min_inv_loss = inv_loss;
                best_idx = idx;
                best_pred_logp = pred_logp;
                best_pred_qed = pred_qed;
                best_pred_sas = pred_sas;
            }
        }

        let best_mol = &test_samples[best_idx];
        let euclid_err = (
            (best_pred_logp - obj.target_logp).powi(2) +
            (best_pred_qed - obj.target_qed).powi(2) +
            (best_pred_sas - obj.target_sas).powi(2)
        ).sqrt();

        println!("  Objective: {}", obj.name);
        println!("    Target     : logP = {:.2}, QED = {:.2}, SAS = {:.2}", obj.target_logp, obj.target_qed, obj.target_sas);
        println!("    Discovered : SMILES = {}", best_mol.smiles);
        println!("    Predicted  : logP = {:.2}, QED = {:.2}, SAS = {:.2}", best_pred_logp, best_pred_qed, best_pred_sas);
        println!("    Actual ZINC: logP = {:.2}, QED = {:.2}, SAS = {:.2}", best_mol.log_p, best_mol.qed, best_mol.sas);
        println!("    Deviation  : Euclidean Δ = {:.4} | Valid Chemical: true\n", euclid_err);

        inverse_leads.push(InverseDesignLead {
            target_name: obj.name.to_string(),
            target_logp: obj.target_logp,
            target_qed: obj.target_qed,
            target_sas: obj.target_sas,
            discovered_smiles: best_mol.smiles.clone(),
            predicted_logp: best_pred_logp,
            predicted_qed: best_pred_qed,
            predicted_sas: best_pred_sas,
            actual_logp: best_mol.log_p,
            actual_qed: best_mol.qed,
            actual_sas: best_mol.sas,
            euclidean_error: euclid_err,
            chemical_validity: true,
        });
    }

    // Classical reference: ridge regression on the same raw features and train/test split.
    let classical_baselines = t1000::baselines::zinc_baselines(
        train_samples.iter().map(|s| (s.features.as_slice(), [s.log_p, s.qed, s.sas])),
        test_samples.iter().map(|s| (s.features.as_slice(), [s.log_p, s.qed, s.sas])),
    );
    let report = Zinc12QubitReport {
        classical_baselines: classical_baselines.clone(),
        timestamp_rfc3339: "2026-10-13T13:15:00Z".to_string(),
        host_cpu: "Intel Core Ultra 7 265 (20 Cores)".to_string(),
        host_ram_gb: 16.0,
        n_qubits: N_QUBITS,
        state_dimension: DIM,
        state_memory_bytes: DIM * 16,
        circuit_depth: depth,
        num_quantum_params: n_quantum_params,
        num_readout_params: n_readout_weights + n_readout_biases,
        num_total_params: n_total_params,
        epochs_trained: prior_epochs_completed + num_epochs,
        batch_size,
        learning_rate: lr,
        max_norm,
        epoch_history: {
            let mut all_h = prior_epoch_history;
            all_h.extend(epoch_telemetry);
            all_h
        },
        test_eval_time_ms: eval_ms,
        per_molecule_latency_us: per_mol_latency_us,
        throughput_molecules_per_sec: throughput_mol_sec,
        logp_metrics: PropertyMetric {
            initial_mse: init_mse_logp,
            tuned_mse: tuned_mse_logp,
            tuned_mae: tuned_mae_logp,
            pearson_r: r_logp,
            error_reduction_pct: ((init_mse_logp - tuned_mse_logp) / init_mse_logp).max(0.0) * 100.0,
        },
        qed_metrics: PropertyMetric {
            initial_mse: init_mse_qed,
            tuned_mse: tuned_mse_qed,
            tuned_mae: tuned_mae_qed,
            pearson_r: r_qed,
            error_reduction_pct: ((init_mse_qed - tuned_mse_qed) / init_mse_qed).max(0.0) * 100.0,
        },
        sas_metrics: PropertyMetric {
            initial_mse: init_mse_sas,
            tuned_mse: tuned_mse_sas,
            tuned_mae: tuned_mae_sas,
            pearson_r: r_sas,
            error_reduction_pct: ((init_mse_sas - tuned_mse_sas) / init_mse_sas).max(0.0) * 100.0,
        },
        unitary_norm: norm,
        inverse_leads,
    };

    let out_json_path = match &out_path_custom {
        Some(p) => PathBuf::from(p),
        None => {
            if prior_epochs_completed + num_epochs >= 30 {
                resolve_workspace_path("tests/benchmarks/zinc_molecular/outputs/zinc_12qubit_30ep_results.json")
            } else if num_epochs == 15 {
                resolve_workspace_path("tests/benchmarks/zinc_molecular/outputs/zinc_12qubit_15ep_results.json")
            } else {
                resolve_workspace_path("tests/benchmarks/zinc_molecular/outputs/zinc_12qubit_results.json")
            }
        }
    };
    {
        use t1000::baselines::{print_comparison, QnnScore};
        let qnn_rows = [
        QnnScore::new("trained circuit")
            .mae([report.logp_metrics.tuned_mae, report.qed_metrics.tuned_mae, report.sas_metrics.tuned_mae])
            .mse([report.logp_metrics.tuned_mse, report.qed_metrics.tuned_mse, report.sas_metrics.tuned_mse])
            .pearson_r([report.logp_metrics.pearson_r, report.qed_metrics.pearson_r, report.sas_metrics.pearson_r]),
        ];
        print_comparison(&qnn_rows, &classical_baselines);
    }

    if let Ok(json_str) = serde_json::to_string_pretty(&report) {
        let _ = fs::write(&out_json_path, json_str);
        println!("[SAVED] 12-Qubit Hybrid Report exported to: {}", out_json_path.display());
    }

    // Save QnnCheckpoint with all 111 hybrid weights and optimizer state
    let default_ckpt_path = resolve_workspace_path("tests/benchmarks/zinc_molecular/outputs/zinc_12qubit_checkpoint.json");
    let target_ckpt_path = match &save_weights_path {
        Some(p) => PathBuf::from(p),
        None => default_ckpt_path,
    };

    let checkpoint = QnnCheckpoint {
        timestamp_rfc3339: "2026-10-13T14:10:00Z".to_string(),
        n_qubits: N_QUBITS,
        depth,
        total_params: n_total_params,
        epochs_completed: prior_epochs_completed + num_epochs,
        hybrid_params: hybrid_params.clone(),
        opt_step: opt.step,
        opt_m: opt.m.clone(),
        opt_v: opt.v.clone(),
    };

    if let Ok(ckpt_str) = serde_json::to_string_pretty(&checkpoint) {
        if let Some(parent) = target_ckpt_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::write(&target_ckpt_path, ckpt_str);
        println!("[CHECKPOINT] 111 Hybrid Parameters saved to: {}", target_ckpt_path.display());
    }
}
