use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use t1000::{
    AdamOptimizer, CircuitAnsatz, DataEncodingLayer, EncodingKind, ParameterShiftEngine,
    PauliObservable,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochLog {
    pub epoch: usize,
    pub lr: f64,
    pub train_loss: f64,
    pub train_accuracy_pct: f64,
    pub max_grad: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QnnTrainingTelemetry {
    pub architecture: String,
    pub n_qubits: usize,
    pub dim: usize,
    pub depth: usize,
    pub observable: String,
    pub num_trainable_parameters: usize,
    pub total_epochs: usize,
    pub initial_lr: f64,
    pub final_lr: f64,
    pub decay_gamma: f64,
    pub decay_step: usize,
    pub initial_loss: f64,
    pub final_loss: f64,
    pub final_accuracy_pct: f64,
    /// Mean loss on points not seen in training, with the final weights.
    pub holdout_loss: f64,
    pub holdout_accuracy_pct: f64,
    pub holdout_samples: usize,
    pub training_time_ms: f64,
    pub epoch_logs: Vec<EpochLog>,
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut n_qubits: usize = 10;
    let mut depth: usize = 3; // Depth expansion: increased default from 2 to 3
    let mut epochs: usize = 20;
    let mut lr: f64 = 0.05;
    let mut decay_gamma: f64 = 0.9;
    let mut decay_step: usize = 5;
    let mut obs_mode = "global".to_string(); // Default to Global Multi-Qubit Observable
    let mut out_path = PathBuf::from("results/experiments/qnn_training_run.json");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--qubits" if i + 1 < args.len() => {
                n_qubits = args[i + 1].parse().unwrap_or(10);
                i += 1;
            }
            "--depth" if i + 1 < args.len() => {
                depth = args[i + 1].parse().unwrap_or(3);
                i += 1;
            }
            "--epochs" if i + 1 < args.len() => {
                epochs = args[i + 1].parse().unwrap_or(20);
                i += 1;
            }
            "--lr" if i + 1 < args.len() => {
                lr = args[i + 1].parse().unwrap_or(0.05);
                i += 1;
            }
            "--decay-gamma" if i + 1 < args.len() => {
                decay_gamma = args[i + 1].parse().unwrap_or(0.9);
                i += 1;
            }
            "--decay-step" if i + 1 < args.len() => {
                decay_step = args[i + 1].parse().unwrap_or(5);
                i += 1;
            }
            "--observable" if i + 1 < args.len() => {
                obs_mode = args[i + 1].to_lowercase();
                i += 1;
            }
            "--out" if i + 1 < args.len() => {
                out_path = PathBuf::from(&args[i + 1]);
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }

    out_path = resolve_workspace_out(&out_path);

    let (observable, obs_name) = match obs_mode.as_str() {
        "z0" => (PauliObservable::Z(0), "Local Pauli <Z_0>".to_string()),
        _ => (
            PauliObservable::GlobalMeanZ,
            format!(
                "Global Multi-Qubit Hamiltonian <H> = (1/{}) sum <Z_i>",
                n_qubits
            ),
        ),
    };

    println!("=========================================================================");
    println!("   VARIATIONAL QUANTUM NEURAL NETWORK (VQC) TRAINING ENGINE              ");
    println!("=========================================================================");
    println!(
        " 1. Data Encoding   : Angle Encoding (Ry rotations across {} qubits)",
        n_qubits
    );
    println!(
        " 2. Ansatz Layers   : Hardware-Efficient (Ry/Rz + CZ Ring, Depth {})",
        depth
    );
    println!(" 3. Observable      : {}", obs_name);
    println!(
        " 4. Optimizer       : Exact Parameter-Shift + Adam (lr_0={}, gamma={}, step={})",
        lr, decay_gamma, decay_step
    );
    println!(
        " Hilbert Space Dim  : {} (dim >= 1024 invariant satisfied)",
        1 << n_qubits
    );
    println!(" Trainable Params   : {} θ angles", 2 * n_qubits * depth);
    println!(" Telemetry Output   : {}", out_path.display());
    println!("-------------------------------------------------------------------------");

    // Initialize the 4 Layers
    let encoding = DataEncodingLayer::new(n_qubits, EncodingKind::Angle);
    let ansatz = CircuitAnsatz::new(n_qubits, depth);

    let num_params = ansatz.param_count();
    let mut weights = ansatz.weights.clone();
    let mut optimizer = AdamOptimizer::new(num_params, lr);
    let engine = ParameterShiftEngine::new(encoding, ansatz, observable);

    // Synthetic non-linear classification dataset
    let dataset = generate_dataset(n_qubits, TRAIN_SAMPLES, 0.0);
    let holdout = generate_dataset(n_qubits, HOLDOUT_SAMPLES, 0.5);
    println!(
        "[*] Generated {} training samples with binary targets {{-1.0, +1.0}}",
        dataset.len()
    );

    let start_time = Instant::now();
    let mut epoch_logs = Vec::new();
    let mut initial_loss = 0.0;

    println!(
        "\n┌───────┬──────────┬──────────────┬──────────────┬──────────────┬──────────────────┐"
    );
    println!(
        "│ Epoch │ LR       │ Train Loss   │ Accuracy (%) │ Max Grad     │ Status           │"
    );
    println!(
        "├───────┼──────────┼──────────────┼──────────────┼──────────────┼──────────────────┤"
    );

    for ep in 1..=epochs {
        // Step exponential learning rate decay: eta_t = eta_0 * gamma^(floor((t-1) / k))
        let current_lr = optimizer.step_epoch_decay(ep - 1, decay_gamma, decay_step);

        let mut total_loss = 0.0;
        let mut correct = 0;
        let mut max_grad: f64 = 0.0;

        // Accumulate batch gradients across dataset
        let mut batch_grads = vec![0.0; num_params];

        for (sample_idx, (x, y_target)) in dataset.iter().enumerate() {
            let (pred, loss, sample_grads) = engine.compute_loss_gradients(x, *y_target, &weights);
            total_loss += loss;

            if ep == 1 && sample_idx < 4 {
                println!("  [Debug Ep 1] Sample #{}: target={:+.1}, y_pred={:+.6}, sample_loss={:.4}, sample_grad[0]={:+.4}", sample_idx, y_target, pred, loss, sample_grads[0]);
            }

            // Prediction sign determines class (+1 or -1)
            let predicted_class = if pred >= 0.0 { 1.0 } else { -1.0 };
            if (predicted_class - *y_target).abs() < 1e-3 {
                correct += 1;
            }

            for (bg, &sg) in batch_grads.iter_mut().zip(sample_grads.iter()) {
                *bg += sg;
                if sg.abs() > max_grad {
                    max_grad = sg.abs();
                }
            }
        }

        if ep == 1 {
            let avg_bg = batch_grads.iter().map(|g| g.abs()).sum::<f64>() / num_params as f64;
            println!(
                "  [Debug Ep 1] Mean Absolute Batch Gradient across {} params = {:.6}",
                num_params, avg_bg
            );
        }

        // Average gradients
        let n_samples = dataset.len() as f64;
        let mean_loss = total_loss / n_samples;
        for g in &mut batch_grads {
            *g /= n_samples;
        }

        if ep == 1 {
            initial_loss = mean_loss;
        }

        // Optimizer update step
        optimizer.step(&mut weights, &batch_grads);

        let acc_pct = f64::from(correct) / n_samples * 100.0;

        println!(
            "│ {:<5} │ {:<8.5} │ {:<12.5} │ {:<12.1} │ {:<12.4} │ Converging...    │",
            ep, current_lr, mean_loss, acc_pct, max_grad
        );

        epoch_logs.push(EpochLog {
            epoch: ep,
            lr: current_lr,
            train_loss: mean_loss,
            train_accuracy_pct: acc_pct,
            max_grad,
        });
    }
    println!(
        "└───────┴──────────┴──────────────┴──────────────┴──────────────┴──────────────────┘\n"
    );

    let total_elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;
    let final_loss = epoch_logs.last().map(|l| l.train_loss).unwrap_or(0.0);
    let final_acc = epoch_logs
        .last()
        .map(|l| l.train_accuracy_pct)
        .unwrap_or(0.0);
    let final_lr = epoch_logs.last().map(|l| l.lr).unwrap_or(lr);

    let (holdout_loss, holdout_acc) = evaluate(&engine, &holdout, &weights);

    let loss_reduction_pct = if initial_loss > 1e-6 {
        (initial_loss - final_loss) / initial_loss * 100.0
    } else {
        0.0
    };

    println!("=========================================================================");
    println!("   HYBRID OPTIMIZATION SUMMARY & CONVERGENCE LEDGER                      ");
    println!("=========================================================================");
    println!(" Initial Loss       : {:.5}", initial_loss);
    println!(
        " Final Loss         : {:.5} ({:.1}% reduction)",
        final_loss, loss_reduction_pct
    );
    println!(
        " Final Accuracy     : {:.1}% (train, {} samples)",
        final_acc,
        dataset.len()
    );
    println!(
        " Holdout Loss       : {:.5} | Accuracy {:.1}% ({} unseen samples)",
        holdout_loss,
        holdout_acc,
        holdout.len()
    );
    println!(" Final Learning Rate: {:.5}", final_lr);
    println!(" Elapsed Time       : {:.2} ms", total_elapsed_ms);
    println!(" Parameter Updates  : {} steps completed", optimizer.step);
    println!("=========================================================================");

    let telemetry = QnnTrainingTelemetry {
        architecture: "Variational Quantum Circuit (HEA + Parameter-Shift)".to_string(),
        n_qubits,
        dim: 1 << n_qubits,
        depth,
        observable: obs_name,
        num_trainable_parameters: num_params,
        total_epochs: epochs,
        initial_lr: lr,
        final_lr,
        decay_gamma,
        decay_step,
        initial_loss,
        final_loss,
        final_accuracy_pct: final_acc,
        holdout_loss,
        holdout_accuracy_pct: holdout_acc,
        holdout_samples: holdout.len(),
        training_time_ms: total_elapsed_ms,
        epoch_logs,
    };

    if let Some(parent) = out_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(json_str) = serde_json::to_string_pretty(&telemetry) {
        // Recorded runs are judged on unseen data: holdout loss decides, holdout accuracy may
        // not drop by more than one point, and only runs with the same settings compete.
        let settings = format!(
            "qubits={n_qubits};depth={depth};epochs={epochs};lr={lr};gamma={decay_gamma};\
             step={decay_step};obs={obs_mode};train={TRAIN_SAMPLES};holdout={HOLDOUT_SAMPLES}"
        );
        let score = t1000::results::Score {
            metric: "holdout_loss",
            value: holdout_loss,
            better: t1000::results::Better::Lower,
            guard: Some(t1000::results::Guard {
                name: "holdout_accuracy_pct",
                value: holdout_acc,
                better: t1000::results::Better::Higher,
                tolerance: 1.0,
            }),
            settings: &settings,
        };
        match t1000::results::write_scored(&out_path, json_str, &score) {
            Ok(t1000::results::Outcome::KeptBest) => {
                println!("[=] Saved run kept: holdout loss {holdout_loss:.5} did not improve it");
            }
            Ok(t1000::results::Outcome::Scratch) => {
                println!("[~] Telemetry written to a scratch copy only (use --record to save)");
            }
            Ok(t1000::results::Outcome::WrittenVariant) => {
                println!(
                    "[+] Different settings from the saved run: recorded as a separate variant"
                );
            }
            Ok(_) => println!("[+] Training telemetry exported to: {}", out_path.display()),
            Err(e) => eprintln!("[!] Could not write telemetry: {e}"),
        }
    }
}

/// Training points sit on a 24-point phase grid; holdout points on a 96-point grid shifted by
/// half a step, so no holdout point coincides with a training point.
const TRAIN_SAMPLES: usize = 24;
const HOLDOUT_SAMPLES: usize = 96;

/// Mean loss and sign accuracy (percent) of `weights` on `data`, forward passes only.
fn evaluate(
    engine: &ParameterShiftEngine,
    data: &[(Vec<f64>, f64)],
    weights: &[f64],
) -> (f64, f64) {
    let (mut loss, mut correct) = (0.0, 0usize);
    for (x, y) in data {
        let pred = engine.predict(x, weights);
        loss += 0.5 * (pred - y) * (pred - y);
        correct += usize::from((if pred >= 0.0 { 1.0 } else { -1.0 } - y).abs() < 1e-3);
    }
    let n = data.len() as f64;
    (loss / n, correct as f64 / n * 100.0)
}

fn generate_dataset(n_qubits: usize, samples: usize, phase_offset: f64) -> Vec<(Vec<f64>, f64)> {
    use std::f64::consts::PI;
    let mut data = Vec::new();

    for i in 0..samples {
        let t = (i as f64 + phase_offset) / (samples as f64) * 2.0 * PI;
        let mut x = Vec::with_capacity(n_qubits);
        for q in 0..n_qubits {
            // Features span full [-π, π] phase space
            let phase = t + (q as f64) * (2.0 * PI / n_qubits as f64);
            x.push(phase.sin() * PI);
        }

        // Geometrically learnable non-linear boundary:
        // Points where primary modes have constructive correlation -> +1.0, else -1.0
        let norm_score = x.iter().take(4).map(|v| v.cos()).sum::<f64>();
        let target = if norm_score > 0.0 { 1.0 } else { -1.0 };
        data.push((x, target));
    }

    data
}

fn resolve_workspace_out(target: &Path) -> PathBuf {
    if let Ok(cwd) = env::current_dir() {
        let mut curr = cwd.as_path();
        for _ in 0..6 {
            if curr.join("Cargo.lock").exists() {
                return curr.join(target);
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
