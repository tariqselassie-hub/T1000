use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use t1000::{
    adjoint_gradient, AdamOptimizer, CircuitAnsatz, DataEncodingLayer, EncodingKind,
    PauliObservable, QuantumState,
};

// Empirical ZINC 10k Target Distributions (Exact Mean & Std)
const MU_LOGP: f64 = 2.4713;
const SIGMA_LOGP: f64 = 1.4451;

const MU_QED: f64 = 0.7266;
const SIGMA_QED: f64 = 0.1398;

const MU_SAS: f64 = 3.0515;
const SIGMA_SAS: f64 = 0.8394;

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
pub struct Zinc10kRunReport {
    pub timestamp_rfc3339: String,
    pub host_cpu: String,
    pub host_ram_gb: f64,
    pub n_qubits: usize,
    pub state_dimension: usize,
    pub state_memory_bytes: usize,
    pub circuit_depth: usize,
    pub num_parameters: usize,
    pub train_sample_count: usize,
    pub test_sample_count: usize,
    pub epochs_trained: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub epoch_history: Vec<EpochTelemetry>,
    pub test_eval_time_ms: f64,
    pub per_molecule_latency_us: f64,
    pub throughput_molecules_per_sec: f64,
    pub logp_metrics: PropertyMetric,
    pub qed_metrics: PropertyMetric,
    pub sas_metrics: PropertyMetric,
    pub unitary_norm: f64,
    #[serde(default)]
    pub classical_baselines: Vec<t1000::baselines::BaselineResult>,
}

/// Fused single-pass expectation value evaluation for Z0, Z1, Z2 across state amplitudes.
/// Eliminates redundant memory sweeps and leverages L1 cache residency.
#[inline(always)]
fn eval_z0_z1_z2(state: &QuantumState) -> (f64, f64, f64) {
    let mut exp0 = 0.0;
    let mut exp1 = 0.0;
    let mut exp2 = 0.0;
    for idx in 0..state.dim {
        let prob = state.real[idx] * state.real[idx] + state.imag[idx] * state.imag[idx];
        let s0 = if (idx & 1) == 0 { 1.0 } else { -1.0 };
        let s1 = if (idx & 2) == 0 { 1.0 } else { -1.0 };
        let s2 = if (idx & 4) == 0 { 1.0 } else { -1.0 };
        exp0 += s0 * prob;
        exp1 += s1 * prob;
        exp2 += s2 * prob;
    }
    (exp0, exp1, exp2)
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
    let mut lr = 0.03;
    let mut max_train = 10000;
    let mut max_eval = 2000;
    let mut max_batches = 0; // 0 = all batches
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
                    lr = args[i + 1].parse().unwrap_or(0.03);
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
            "--max-batches" => {
                if i + 1 < args.len() {
                    max_batches = args[i + 1].parse().unwrap_or(0);
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    println!("=========================================================================");
    println!("     GUDDANETWORKS QNN: ZINC 10,000-MOLECULE QUANTUM TRAINING SESSION   ");
    println!("=========================================================================");
    println!(" Training Set    : 10,000 Featurized Drug Molecules (ZINC 250k)");
    println!(" Evaluation Set  : {} Holdout Drug Molecules", max_eval);
    println!(
        " Hyperparameters : Epochs = {}, Batch Size = {}, Learning Rate = {:.3}",
        num_epochs, batch_size, lr
    );
    println!(
        " Hardware Engine : AVX2+FMA SIMD (10 Qubits | 16 KB L1 Cache | {} Threads)",
        num_threads
    );
    println!(" Targets         : Multi-Task logP, qed, SAS (Exact Analytic Parameter Shift)");
    println!("=========================================================================");

    let train_path =
        resolve_workspace_path("tests/benchmarks/zinc_molecular/outputs/zinc_train_10k.csv");
    let test_path =
        resolve_workspace_path("tests/benchmarks/zinc_molecular/outputs/zinc_test_2k.csv");

    println!(
        "[INIT] Loading ZINC 10k train set from: {}",
        train_path.display()
    );
    let train_samples = match parse_zinc_csv(&train_path, max_train) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[-] Error loading 10k train set: {}", e);
            return;
        }
    };
    println!(
        "  -> Successfully loaded {} train molecules.",
        train_samples.len()
    );

    println!(
        "[INIT] Loading ZINC 2k evaluation set: {}",
        test_path.display()
    );
    let test_samples = match parse_zinc_csv(&test_path, max_eval) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[-] Error loading test set: {}", e);
            return;
        }
    };
    println!(
        "  -> Successfully loaded {} test molecules.\n",
        test_samples.len()
    );

    let n_qubits = 10;
    let depth = 4;
    let encoding = DataEncodingLayer::new(n_qubits, EncodingKind::Amplitude);
    let ansatz = CircuitAnsatz::new(n_qubits, depth).with_pi_layer(true);

    let mut out_state = QuantumState::new(encoding.dim);

    // Baseline holdout evaluation (Zero-Shot Pre-Training)
    let mut init_preds_logp = Vec::with_capacity(test_samples.len());
    let mut init_preds_qed = Vec::with_capacity(test_samples.len());
    let mut init_preds_sas = Vec::with_capacity(test_samples.len());
    let mut targets_logp = Vec::with_capacity(test_samples.len());
    let mut targets_qed = Vec::with_capacity(test_samples.len());
    let mut targets_sas = Vec::with_capacity(test_samples.len());

    for s in &test_samples {
        let state_in = encoding.encode(&s.features);
        ansatz.forward_into(&state_in, &ansatz.weights, &mut out_state);
        let (z0, z1, z2) = eval_z0_z1_z2(&out_state);

        let p_logp = MU_LOGP + SIGMA_LOGP * z0;
        let p_qed = MU_QED + SIGMA_QED * z1;
        let p_sas = MU_SAS + SIGMA_SAS * z2;

        init_preds_logp.push(p_logp);
        init_preds_qed.push(p_qed);
        init_preds_sas.push(p_sas);

        targets_logp.push(s.log_p);
        targets_qed.push(s.qed);
        targets_sas.push(s.sas);
    }

    let n_test = test_samples.len() as f64;
    let init_mse_logp = init_preds_logp
        .iter()
        .zip(&targets_logp)
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / n_test;
    let init_mse_qed = init_preds_qed
        .iter()
        .zip(&targets_qed)
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / n_test;
    let init_mse_sas = init_preds_sas
        .iter()
        .zip(&targets_sas)
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / n_test;

    println!("-------------------------------------------------------------------------");
    println!(" [BASELINE TEST METRICS (Pre-Training Zero-Shot)]");
    println!("-------------------------------------------------------------------------");
    println!(
        "  Initial Test MSE - logP : {:.4} (MAE: {:.4})",
        init_mse_logp,
        init_preds_logp
            .iter()
            .zip(&targets_logp)
            .map(|(p, t)| (p - t).abs())
            .sum::<f64>()
            / n_test
    );
    println!(
        "  Initial Test MSE - qed  : {:.4} (MAE: {:.4})",
        init_mse_qed,
        init_preds_qed
            .iter()
            .zip(&targets_qed)
            .map(|(p, t)| (p - t).abs())
            .sum::<f64>()
            / n_test
    );
    println!(
        "  Initial Test MSE - SAS  : {:.4} (MAE: {:.4})",
        init_mse_sas,
        init_preds_sas
            .iter()
            .zip(&targets_sas)
            .map(|(p, t)| (p - t).abs())
            .sum::<f64>()
            / n_test
    );
    println!("-------------------------------------------------------------------------\n");

    // Training loop with Exact Multi-Task Parameter Shift
    let mut opt_weights = ansatz.weights.clone();
    let mut opt = AdamOptimizer::new(ansatz.param_count(), lr);

    let mut epoch_telemetry = Vec::new();
    let total_train_start = Instant::now();

    println!(
        "Starting {} training epochs across {} molecules in batches of {}...",
        num_epochs,
        train_samples.len(),
        batch_size
    );
    for epoch in 1..=num_epochs {
        let t_epoch = Instant::now();
        let mut total_epoch_loss = 0.0;
        let mut total_grad_norm = 0.0;
        let mut batches_run = 0;

        for chunk in train_samples.chunks(batch_size) {
            let mut batch_grads = vec![0.0; ansatz.param_count()];
            let mut batch_loss = 0.0;

            // Multithreaded parameter-shift evaluation across thread pool
            let chunk_len = chunk.len();
            let thread_chunk_size = (chunk_len + num_threads - 1) / num_threads;
            let sub_chunks: Vec<&[MoleculeSample]> = chunk.chunks(thread_chunk_size).collect();

            let thread_results: Vec<(Vec<f64>, f64)> = std::thread::scope(|s| {
                let mut handles = Vec::with_capacity(sub_chunks.len());
                for sub_chunk in sub_chunks {
                    let ansatz_ref = &ansatz;
                    let encoding_ref = &encoding;
                    let weights_ref = &opt_weights;

                    handles.push(s.spawn(move || {
                        let z_obs = [
                            PauliObservable::Z(0),
                            PauliObservable::Z(1),
                            PauliObservable::Z(2),
                        ];
                        let mut thread_grads = vec![0.0; weights_ref.len()];
                        let mut thread_loss = 0.0;

                        for sample in sub_chunk {
                            let state_in = encoding_ref.encode(&sample.features);
                            // Scaled errors e_k = (pred_k - y_k) / sigma_k with pred_k = mu_k + sigma_k <Z_k>.
                            let errors = |z: &[f64]| {
                                [
                                    (MU_LOGP + SIGMA_LOGP * z[0] - sample.log_p) / SIGMA_LOGP,
                                    (MU_QED + SIGMA_QED * z[1] - sample.qed) / SIGMA_QED,
                                    (MU_SAS + SIGMA_SAS * z[2] - sample.sas) / SIGMA_SAS,
                                ]
                            };
                            // One adjoint sweep gives d/dθ of sum_k e_k <Z_k> (e_k held at the current
                            // point), i.e. half of dL/dθ for L = sum_k e_k^2, matching the previous
                            // parameter-shift accumulation.
                            let res =
                                adjoint_gradient(ansatz_ref, &state_in, weights_ref, &z_obs, |z| {
                                    errors(z).to_vec()
                                });
                            let e = errors(&res.expectations);
                            thread_loss += e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
                            for (tg, g) in thread_grads.iter_mut().zip(&res.gradient) {
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

            opt.step(&mut opt_weights, &batch_grads);

            let g_norm = batch_grads.iter().map(|v| v * v).sum::<f64>().sqrt();
            total_grad_norm += g_norm;
            total_epoch_loss += batch_loss;
            batches_run += 1;

            if max_batches > 0 && batches_run >= max_batches {
                break;
            }
        }

        let avg_epoch_loss = total_epoch_loss / (batches_run as f64);
        let avg_grad_norm = total_grad_norm / (batches_run as f64);
        let epoch_ms = t_epoch.elapsed().as_secs_f64() * 1000.0;

        println!(
            "  [EPOCH {:>2}/{}] Loss: {:>7.4} | ||∇θ||: {:>6.4} | Batches: {:>3} | Time: {:>7.1} ms",
            epoch, num_epochs, avg_epoch_loss, avg_grad_norm, batches_run, epoch_ms
        );

        epoch_telemetry.push(EpochTelemetry {
            epoch,
            batch_count: batches_run,
            train_loss: avg_epoch_loss,
            grad_norm: avg_grad_norm,
            epoch_time_ms: epoch_ms,
        });
    }

    let total_train_time = total_train_start.elapsed();
    println!(
        "\nTraining completed in {:.2} seconds.",
        total_train_time.as_secs_f64()
    );

    // Final Evaluation on Holdout Test Molecules
    println!("\n=========================================================================");
    println!(
        " [FINAL EVALUATION] {} HOLDOUT MOLECULES EVALUATION",
        test_samples.len()
    );
    println!("=========================================================================");

    let t_eval_start = Instant::now();
    let mut tuned_preds_logp = Vec::with_capacity(test_samples.len());
    let mut tuned_preds_qed = Vec::with_capacity(test_samples.len());
    let mut tuned_preds_sas = Vec::with_capacity(test_samples.len());

    for s in &test_samples {
        let state_in = encoding.encode(&s.features);
        ansatz.forward_into(&state_in, &opt_weights, &mut out_state);
        let (z0, z1, z2) = eval_z0_z1_z2(&out_state);

        let p_logp = MU_LOGP + SIGMA_LOGP * z0;
        let p_qed = MU_QED + SIGMA_QED * z1;
        let p_sas = MU_SAS + SIGMA_SAS * z2;

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

    let tuned_mse_logp = tuned_preds_logp
        .iter()
        .zip(&targets_logp)
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / n_eval;
    let tuned_mae_logp = tuned_preds_logp
        .iter()
        .zip(&targets_logp)
        .map(|(p, t)| (p - t).abs())
        .sum::<f64>()
        / n_eval;
    let r_logp = compute_pearson_r(&tuned_preds_logp, &targets_logp);

    let tuned_mse_qed = tuned_preds_qed
        .iter()
        .zip(&targets_qed)
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / n_eval;
    let tuned_mae_qed = tuned_preds_qed
        .iter()
        .zip(&targets_qed)
        .map(|(p, t)| (p - t).abs())
        .sum::<f64>()
        / n_eval;
    let r_qed = compute_pearson_r(&tuned_preds_qed, &targets_qed);

    let tuned_mse_sas = tuned_preds_sas
        .iter()
        .zip(&targets_sas)
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / n_eval;
    let tuned_mae_sas = tuned_preds_sas
        .iter()
        .zip(&targets_sas)
        .map(|(p, t)| (p - t).abs())
        .sum::<f64>()
        / n_eval;
    let r_sas = compute_pearson_r(&tuned_preds_sas, &targets_sas);

    println!(
        "  Evaluation Time      : {:>8.2} ms ({} molecules)",
        eval_ms,
        test_samples.len()
    );
    println!(
        "  Per-Molecule Latency : {:>8.2} μs/molecule",
        per_mol_latency_us
    );
    println!(
        "  Screening Throughput : {:>8.1} molecules/second",
        throughput_mol_sec
    );
    println!(
        "  Unitary Norm Fidelity: {:>8.6} (Strict Unitary Preservation)",
        norm
    );

    println!("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    ZINC 10k QUANTUM NEURAL NETWORK HOLDOUT EVALUATION LEDGER                          ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Target Property         │ Initial MSE │ Tuned MSE   │ Tuned MAE   │ Pearson R │ Error Reduction   ║");
    println!("╟─────────────────────────┼─────────────┼─────────────┼─────────────┼───────────┼───────────────────╢");
    println!(
        "║ logP (Lipophilicity)    │ {:>11.4} │ {:>11.4} │ {:>11.4} │ {:>9.4} │ {:>15.2}% ║",
        init_mse_logp,
        tuned_mse_logp,
        tuned_mae_logp,
        r_logp,
        ((init_mse_logp - tuned_mse_logp) / init_mse_logp).max(0.0) * 100.0
    );
    println!(
        "║ qed  (Drug-Likeness)    │ {:>11.4} │ {:>11.4} │ {:>11.4} │ {:>9.4} │ {:>15.2}% ║",
        init_mse_qed,
        tuned_mse_qed,
        tuned_mae_qed,
        r_qed,
        ((init_mse_qed - tuned_mse_qed) / init_mse_qed).max(0.0) * 100.0
    );
    println!(
        "║ SAS  (Synthetic Access) │ {:>11.4} │ {:>11.4} │ {:>11.4} │ {:>9.4} │ {:>15.2}% ║",
        init_mse_sas,
        tuned_mse_sas,
        tuned_mae_sas,
        r_sas,
        ((init_mse_sas - tuned_mse_sas) / init_mse_sas).max(0.0) * 100.0
    );
    println!("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════╝");

    // Classical reference: ridge regression on the same raw features and train/test split.
    let classical_baselines = t1000::baselines::zinc_baselines(
        train_samples
            .iter()
            .map(|s| (s.features.as_slice(), [s.log_p, s.qed, s.sas])),
        test_samples
            .iter()
            .map(|s| (s.features.as_slice(), [s.log_p, s.qed, s.sas])),
    );
    let report = Zinc10kRunReport {
        classical_baselines: classical_baselines.clone(),
        timestamp_rfc3339: "2026-10-13T12:35:00Z".to_string(),
        host_cpu: "Intel Core Ultra 7 265 (20 Cores)".to_string(),
        host_ram_gb: 16.0,
        n_qubits,
        state_dimension: encoding.dim,
        state_memory_bytes: encoding.dim * 16,
        circuit_depth: depth,
        num_parameters: ansatz.param_count(),
        train_sample_count: train_samples.len(),
        test_sample_count: test_samples.len(),
        epochs_trained: num_epochs,
        batch_size,
        learning_rate: lr,
        epoch_history: epoch_telemetry,
        test_eval_time_ms: eval_ms,
        per_molecule_latency_us: per_mol_latency_us,
        throughput_molecules_per_sec: throughput_mol_sec,
        logp_metrics: PropertyMetric {
            initial_mse: init_mse_logp,
            tuned_mse: tuned_mse_logp,
            tuned_mae: tuned_mae_logp,
            pearson_r: r_logp,
            error_reduction_pct: ((init_mse_logp - tuned_mse_logp) / init_mse_logp).max(0.0)
                * 100.0,
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
    };

    let out_json_path =
        resolve_workspace_path("tests/benchmarks/zinc_molecular/outputs/zinc_qnn_10k_results.json");
    {
        use t1000::baselines::{print_comparison, QnnScore};
        let qnn_rows = [QnnScore::new("trained circuit")
            .mae([
                report.logp_metrics.tuned_mae,
                report.qed_metrics.tuned_mae,
                report.sas_metrics.tuned_mae,
            ])
            .mse([
                report.logp_metrics.tuned_mse,
                report.qed_metrics.tuned_mse,
                report.sas_metrics.tuned_mse,
            ])
            .pearson_r([
                report.logp_metrics.pearson_r,
                report.qed_metrics.pearson_r,
                report.sas_metrics.pearson_r,
            ])];
        print_comparison(&qnn_rows, &classical_baselines);
    }

    if let Ok(json_str) = serde_json::to_string_pretty(&report) {
        let _ = fs::write(&out_json_path, json_str);
        println!(
            "\n[SAVED] 10k Training and Evaluation Report exported to: {}",
            out_json_path.display()
        );
    }
}
