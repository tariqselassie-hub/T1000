// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

use super::*;
use t1000::zinc_holdout_eval::{print_zero_shot, target_columns};

/// Fused single-pass expectation value evaluation for Z0, Z1, Z2 across state amplitudes.
/// Eliminates redundant memory sweeps and leverages L1 cache residency.
#[inline(always)]
pub fn eval_z0_z1_z2(state: &QuantumState) -> (f64, f64, f64) {
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

/// The circuit shared by every phase: data encoding plus trainable ansatz.
pub struct Model {
    pub encoding: DataEncodingLayer,
    pub ansatz: CircuitAnsatz,
}

impl Model {
    pub fn new() -> Self {
        Model {
            encoding: DataEncodingLayer::new(N_QUBITS, EncodingKind::Amplitude),
            ansatz: CircuitAnsatz::new(N_QUBITS, DEPTH).with_pi_layer(true),
        }
    }

    /// `<Z_0>, <Z_1>, <Z_2>` of the circuit output for every molecule.
    pub fn collect_readouts(
        &self,
        samples: &[ZincMolecule],
        weights: &[f64],
        state: &mut QuantumState,
    ) -> Vec<[f64; 3]> {
        samples
            .iter()
            .map(|m| {
                self.ansatz
                    .forward_into(&self.encoding.encode(&m.features), weights, state);
                let (z0, z1, z2) = eval_z0_z1_z2(state);
                [z0, z1, z2]
            })
            .collect()
    }

    /// Head predictions for each holdout molecule: one `Vec` per property.
    pub fn predict(
        &self,
        samples: &[ZincMolecule],
        weights: &[f64],
        head: &[AffineFit; 3],
        state: &mut QuantumState,
    ) -> [Vec<f64>; 3] {
        let mut preds = [(); 3].map(|_| Vec::with_capacity(samples.len()));
        for z in self.collect_readouts(samples, weights, state) {
            for (out, (h, zk)) in preds.iter_mut().zip(head.iter().zip(z)) {
                out.push(h.apply(zk));
            }
        }
        preds
    }
}

pub fn targets_of(m: &ZincMolecule) -> [f64; 3] {
    [m.log_p, m.qed, m.sas]
}

/// Prints and returns the pre-training (zero-shot) holdout MSE of each property.
pub fn print_baseline(test: &[ZincMolecule], model: &Model, state: &mut QuantumState) -> [f64; 3] {
    let fixed = fixed_head();
    let preds = model.predict(test, &model.ansatz.weights, &fixed, state);
    println!("{RULE}");
    println!(" [BASELINE TEST METRICS (Pre-Training Zero-Shot)]");
    println!("{RULE}");
    let init_mse = print_zero_shot(&preds, &target_columns(test));
    println!("{RULE}\n");
    init_mse
}

/// Readout head: prediction_k = intercept_k + slope_k * <Z_k>. The default is the fixed
/// (mu, sigma) head; --calibrated-head refits it by least squares on the training set.
pub fn fixed_head() -> [AffineFit; 3] {
    [
        AffineFit {
            slope: SIGMA_LOGP,
            intercept: MU_LOGP,
        },
        AffineFit {
            slope: SIGMA_QED,
            intercept: MU_QED,
        },
        AffineFit {
            slope: SIGMA_SAS,
            intercept: MU_SAS,
        },
    ]
}

pub fn fit_head(
    model: &Model,
    train: &[ZincMolecule],
    weights: &[f64],
    state: &mut QuantumState,
    n: usize,
) -> [AffineFit; 3] {
    let subset = &train[..n.min(train.len())];
    let z = model.collect_readouts(subset, weights, state);
    let y: Vec<[f64; 3]> = subset.iter().map(|m| [m.log_p, m.qed, m.sas]).collect();
    t1000::calibration::fit_columns(&z, &y)
}

/// Scaled residuals `e_k = (pred_k - y_k) / sigma_k` of the affine readout on the circuit
/// `<Z_k>`, with `pred_k = intercept_k + slope_k <Z_k>`.
#[inline]
fn scaled_errors(head: &[AffineFit; 3], y: &[f64; 3], z: &[f64]) -> [f64; 3] {
    [0, 1, 2].map(|k| (head[k].apply(z[k]) - y[k]) / SIGMAS[k])
}

/// Chain-rule coefficients of that residual loss w.r.t. the observables:
/// `dL/d<Z_k> = e_k * slope_k / sigma_k` (exactly 1 for the fixed head).
#[inline]
fn residual_coefficients(head: &[AffineFit; 3], y: &[f64; 3], z: &[f64]) -> [f64; 3] {
    let e = scaled_errors(head, y, z);
    [0, 1, 2].map(|k| e[k] * head[k].slope / SIGMAS[k])
}

/// Mean gradient and mean loss of one batch, one fused parallel adjoint sweep per sample.
///
/// Each sample is encoded straight into its own pre-allocated workspace and swept once; the
/// per-sample gradients and residuals are summed in sample order, so the accumulated
/// gradient and loss match the old sequential accumulation exactly. `ws_batch` is the
/// caller-owned workspace set sized to the batch capacity, reused across batches.
pub fn batch_gradient(
    model: &Model,
    weights: &[f64],
    head: &[AffineFit; 3],
    chunk: &[ZincMolecule],
    ws_batch: &mut BatchAdjointWorkspace,
) -> (Vec<f64>, f64) {
    let z_obs = [
        PauliObservable::Z(0),
        PauliObservable::Z(1),
        PauliObservable::Z(2),
    ];

    let mut chunk_grads = vec![0.0; model.ansatz.param_count()];
    let results = adjoint_gradient_encoded_batch(
        &model.ansatz,
        |i, state| model.encoding.encode_into(&chunk[i].features, state),
        weights,
        &z_obs,
        |i, z| residual_coefficients(head, &targets_of(&chunk[i]), z).to_vec(),
        ws_batch,
        chunk.len(),
        &mut chunk_grads,
    );

    let mut batch_grads = vec![0.0; model.ansatz.param_count()];
    let mut batch_loss = 0.0;
    for (i, result) in results.iter().enumerate() {
        let e = scaled_errors(head, &targets_of(&chunk[i]), &result.expectations);
        batch_loss += e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
        for (bg, &g) in batch_grads.iter_mut().zip(&result.gradient) {
            *bg += g;
        }
    }

    let b_len = chunk.len() as f64;
    for bg in batch_grads.iter_mut() {
        *bg /= b_len;
    }
    batch_loss /= b_len;
    (batch_grads, batch_loss)
}

/// Training loop with exact multi-task gradients. Returns the final weights, readout head
/// and per-epoch telemetry.
pub fn train(
    args: &Args,
    model: &Model,
    train_samples: &[ZincMolecule],
    state: &mut QuantumState,
) -> (Vec<f64>, [AffineFit; 3], Vec<EpochTelemetry>) {
    let mut opt_weights = model.ansatz.weights.clone();
    let mut opt = AdamOptimizer::new(model.ansatz.param_count(), args.lr);
    let mut epoch_telemetry = Vec::new();
    let mut head = fixed_head();

    println!(
        "Starting {} training epochs across {} molecules in batches of {}...",
        args.num_epochs,
        train_samples.len(),
        args.batch_size
    );

    // Pre-allocated scratch for the fused parallel adjoint sweep: one workspace set sized to
    // the batch capacity, reused by every batch of every epoch (zero per-batch allocation).
    let mut ws_batch =
        BatchAdjointWorkspace::new(model.ansatz.dim, model.ansatz.n_qubits, args.batch_size);

    for epoch in 1..=args.num_epochs {
        let t_epoch = Instant::now();
        if args.calibrated_head {
            head = fit_head(model, train_samples, &opt_weights, state, 2000);
        }
        let mut total_epoch_loss = 0.0;
        let mut total_grad_norm = 0.0;
        let mut batches_run = 0;

        for chunk in train_samples.chunks(args.batch_size) {
            let (batch_grads, batch_loss) =
                batch_gradient(model, &opt_weights, &head, chunk, &mut ws_batch);
            opt.step(&mut opt_weights, &batch_grads);

            let g_norm = batch_grads.iter().map(|v| v * v).sum::<f64>().sqrt();
            total_grad_norm += g_norm;
            total_epoch_loss += batch_loss;
            batches_run += 1;

            if args.max_batches > 0 && batches_run >= args.max_batches {
                break;
            }
        }

        let avg_epoch_loss = total_epoch_loss / (batches_run as f64);
        let avg_grad_norm = total_grad_norm / (batches_run as f64);
        let epoch_ms = t_epoch.elapsed().as_secs_f64() * 1000.0;

        println!(
            "  [EPOCH {:>2}/{}] Loss: {:>7.4} | ||∇θ||: {:>6.4} | Batches: {:>3} | Time: {:>7.1} ms",
            epoch, args.num_epochs, avg_epoch_loss, avg_grad_norm, batches_run, epoch_ms
        );

        epoch_telemetry.push(EpochTelemetry {
            epoch,
            batch_count: batches_run,
            train_loss: avg_epoch_loss,
            grad_norm: avg_grad_norm,
            epoch_time_ms: epoch_ms,
        });
    }
    (opt_weights, head, epoch_telemetry)
}
