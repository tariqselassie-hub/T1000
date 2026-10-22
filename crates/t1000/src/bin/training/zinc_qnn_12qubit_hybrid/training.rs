use super::report::Zinc12QubitReport;
use super::*;
use std::fs;

/// What a successful `--resume` restored besides the weights.
#[derive(Debug)]
pub struct Resumed {
    opt_step: usize,
    opt_m: Vec<f64>,
    opt_v: Vec<f64>,
    pub prior_epochs: usize,
    pub prior_history: Vec<EpochTelemetry>,
}

/// Everything the report needs from the training session.
#[derive(Debug)]
pub struct Trained {
    pub prior_epochs: usize,
    pub prior_history: Vec<EpochTelemetry>,
    pub telemetry: Vec<EpochTelemetry>,
    pub opt: AdamOptimizer,
}

/// Restores weights and optimizer state from a checkpoint; `None` (with a diagnostic) means
/// the caller should initialise fresh. `params` is only written on success.
pub fn try_resume(path_str: &str, n_total: usize, params: &mut [f64]) -> Option<Resumed> {
    let p = resolve_workspace_path(path_str);
    if !p.exists() {
        eprintln!(
            "[-] Resume checkpoint {} not found. Initializing fresh.",
            p.display()
        );
        return None;
    }
    let content = match fs::read_to_string(&p) {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "[-] Failed to read checkpoint file: {}. Initializing fresh.",
                e
            );
            return None;
        }
    };
    let ckpt = match serde_json::from_str::<QnnCheckpoint>(&content) {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "[-] Failed to parse checkpoint JSON: {}. Initializing fresh.",
                e
            );
            return None;
        }
    };
    if ckpt.hybrid_params.len() != n_total {
        eprintln!(
            "[-] Checkpoint param count mismatch: expected {}, got {}. Initializing fresh.",
            n_total,
            ckpt.hybrid_params.len()
        );
        return None;
    }
    params.copy_from_slice(&ckpt.hybrid_params);
    println!(
        "[RESUME] Successfully restored {} hybrid weights from: {}",
        n_total,
        p.display()
    );
    println!(
        "  -> Prior Epochs Completed: {}, Optimizer Step: {}\n",
        ckpt.epochs_completed, ckpt.opt_step
    );
    Some(Resumed {
        opt_step: ckpt.opt_step,
        opt_m: ckpt.opt_m,
        opt_v: ckpt.opt_v,
        prior_epochs: ckpt.epochs_completed,
        prior_history: load_prior_history(),
    })
}

/// Prior epoch history from the 15-epoch results file, when available.
fn load_prior_history() -> Vec<EpochTelemetry> {
    let path = resolve_workspace_path(
        "tests/benchmarks/zinc_molecular/outputs/zinc_12qubit_15ep_results.json",
    );
    if !path.exists() {
        return Vec::new();
    }
    let Ok(prior_json) = fs::read_to_string(&path) else {
        return Vec::new();
    };
    let Ok(prior_rep) = serde_json::from_str::<Zinc12QubitReport>(&prior_json) else {
        return Vec::new();
    };
    let history = prior_rep.epoch_history;
    println!(
        "  -> Loaded {} prior epoch records for continuous telemetry history.",
        history.len()
    );
    history
}

/// Loss `sum_k task_w_k * ((pred_k - y_k) / sigma_k)^2` of the linear readout
/// `pred_k = b_k + sum_q W_kq <Z_q>`, with `dL/dpred_k`.
fn readout_loss(
    params: &[f64],
    l: &Layout,
    targets: &[f64; N_PROPS],
    task_weights: &[f64; N_PROPS],
    z: &[f64],
) -> (f64, [f64; N_PROPS]) {
    let sigmas = [SIGMA_LOGP, SIGMA_QED, SIGMA_SAS];
    let mut loss = 0.0;
    let mut d_pred = [0.0; N_PROPS];
    for k in 0..N_PROPS {
        let mut pred = params[l.b_offset + k];
        for q in 0..N_QUBITS {
            pred += params[l.w_offset + k * N_QUBITS + q] * z[q];
        }
        let e = (pred - targets[k]) / sigmas[k];
        loss += task_weights[k] * e * e;
        d_pred[k] = 2.0 * task_weights[k] * e / sigmas[k];
    }
    (loss, d_pred)
}

/// Gradient and summed loss of one slice of a batch.
fn slice_gradient(
    model: &Model,
    params: &[f64],
    qed_weight: f64,
    samples: &[ZincMolecule],
) -> (Vec<f64>, f64) {
    let l = model.layout;
    let z_obs: Vec<PauliObservable> = (0..N_QUBITS).map(PauliObservable::Z).collect();
    let mut thread_grads = vec![0.0; l.n_total];
    let mut thread_loss = 0.0;
    let task_weights = [1.0, qed_weight, 1.0];

    for sample in samples {
        let state_in = model.encoding.encode(&sample.features);
        let targets = [sample.log_p, sample.qed, sample.sas];
        let head = |z: &[f64]| readout_loss(params, &l, &targets, &task_weights, z);
        // Chain rule into the circuit: dL/dθ = sum_q g_q d<Z_q>/dθ with
        // g_q = sum_k dL/dpred_k * W_kq, computed by one adjoint sweep.
        let res = adjoint_gradient(
            &model.ansatz,
            &state_in,
            &params[..l.n_quantum],
            &z_obs,
            |z| {
                let (_, d_pred) = head(z);
                (0..N_QUBITS)
                    .map(|q| {
                        (0..N_PROPS)
                            .map(|k| d_pred[k] * params[l.w_offset + k * N_QUBITS + q])
                            .sum()
                    })
                    .collect()
            },
        );
        let z_base = &res.expectations;
        let (loss, d_pred) = head(z_base);
        thread_loss += loss;
        for k in 0..N_PROPS {
            thread_grads[l.b_offset + k] += d_pred[k];
            for q in 0..N_QUBITS {
                thread_grads[l.w_offset + k * N_QUBITS + q] += d_pred[k] * z_base[q];
            }
        }
        for (tg, g) in thread_grads[..l.n_quantum].iter_mut().zip(&res.gradient) {
            *tg += g;
        }
    }
    (thread_grads, thread_loss)
}

/// Mean gradient and mean loss of one batch, evaluated across a thread pool.
fn batch_gradient(
    model: &Model,
    params: &[f64],
    args: &Args,
    chunk: &[ZincMolecule],
) -> (Vec<f64>, f64) {
    let mut batch_grads = vec![0.0; model.layout.n_total];
    let mut batch_loss = 0.0;

    let chunk_len = chunk.len();
    let thread_chunk_size = chunk_len.div_ceil(args.num_threads);
    let sub_chunks: Vec<&[ZincMolecule]> = chunk.chunks(thread_chunk_size).collect();

    let thread_results: Vec<(Vec<f64>, f64)> = std::thread::scope(|s| {
        let handles: Vec<_> = sub_chunks
            .into_iter()
            .map(|sub| s.spawn(move || slice_gradient(model, params, args.qed_weight, sub)))
            .collect();
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
    (batch_grads, batch_loss)
}

/// Step-decay schedule for runs of ten or more epochs: x0.6 at 40% and x0.3 at 75%.
fn apply_lr_schedule(args: &Args, opt: &mut AdamOptimizer, epoch: usize, cumulative: usize) {
    if args.num_epochs < 10 {
        return;
    }
    let s1 = (args.num_epochs as f64 * 0.4).round() as usize;
    let s2 = (args.num_epochs as f64 * 0.75).round() as usize;
    let factor = if epoch == s1 {
        0.6
    } else if epoch == s2 {
        0.3
    } else {
        return;
    };
    let new_lr = args.lr * factor;
    opt.set_lr(new_lr);
    println!(
        "  [*] Scheduled learning rate decay: LR = {:.4} at epoch {} (cumulative {})",
        new_lr, epoch, cumulative
    );
}

/// One epoch over the training set; returns its telemetry and prints the summary line.
fn run_epoch(
    args: &Args,
    model: &Model,
    train: &[ZincMolecule],
    params: &mut [f64],
    opt: &mut AdamOptimizer,
    (cumulative, total_target): (usize, usize),
) -> EpochTelemetry {
    let t_epoch = Instant::now();
    let mut total_epoch_loss = 0.0;
    let mut total_grad_norm = 0.0;
    let mut batches_run = 0;
    let mut clipped_batches = 0;

    for chunk in train.chunks(args.batch_size) {
        let (mut batch_grads, batch_loss) = batch_gradient(model, params, args, chunk);

        let raw_g_norm = batch_grads.iter().map(|v| v * v).sum::<f64>().sqrt();
        total_grad_norm += raw_g_norm;

        if args.max_norm > 0.0 && raw_g_norm > args.max_norm {
            clipped_batches += 1;
            let scale = args.max_norm / raw_g_norm;
            for bg in batch_grads.iter_mut() {
                *bg *= scale;
            }
        }

        opt.step(params, &batch_grads);

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

    let clip_str = if args.max_norm > 0.0 {
        format!(
            " | Clip: {:>2}/{} ({:>4.1}%)",
            clipped_batches, batches_run, clip_rate_pct
        )
    } else {
        "".to_string()
    };

    println!(
        "  [EPOCH {:>2}/{}] Loss: {:>7.4} | ||∇θ||: {:>6.4} | Batches: {:>3}{} | Time: {:>7.1} ms",
        cumulative, total_target, avg_epoch_loss, avg_grad_norm, batches_run, clip_str, epoch_ms
    );

    EpochTelemetry {
        epoch: cumulative,
        batch_count: batches_run,
        train_loss: avg_epoch_loss,
        grad_norm: avg_grad_norm,
        epoch_time_ms: epoch_ms,
        clipped_batches,
        clip_rate_pct,
    }
}

/// The hybrid training session, continuing from `resumed` when given.
pub fn train(
    args: &Args,
    model: &Model,
    train_samples: &[ZincMolecule],
    params: &mut [f64],
    resumed: Option<Resumed>,
) -> Trained {
    let n_total = model.layout.n_total;
    let prior_epochs = resumed.as_ref().map_or(0, |r| r.prior_epochs);
    let total_epochs_target = prior_epochs + args.num_epochs;
    if prior_epochs > 0 {
        println!("Starting {} resumed epochs (Cumulative: Epochs {} -> {}) across {} molecules in batches of {}...", args.num_epochs, prior_epochs + 1, total_epochs_target, train_samples.len(), args.batch_size);
    } else {
        println!(
            "Starting {} training epochs across {} molecules in batches of {}...",
            args.num_epochs,
            train_samples.len(),
            args.batch_size
        );
    }

    let mut opt = AdamOptimizer::new(n_total, args.lr);
    let mut prior_history = Vec::new();
    if let Some(r) = resumed {
        prior_history = r.prior_history;
        if r.opt_m.len() == n_total {
            opt.step = r.opt_step;
            opt.m = r.opt_m;
            opt.v = r.opt_v;
            println!(
                "[RESUME] Restored Adam momentum estimators (step = {}).\n",
                opt.step
            );
        }
    }

    let mut telemetry = Vec::new();
    for epoch in 1..=args.num_epochs {
        let cumulative = prior_epochs + epoch;
        apply_lr_schedule(args, &mut opt, epoch, cumulative);
        telemetry.push(run_epoch(
            args,
            model,
            train_samples,
            params,
            &mut opt,
            (cumulative, total_epochs_target),
        ));
    }
    Trained {
        prior_epochs,
        prior_history,
        telemetry,
        opt,
    }
}
