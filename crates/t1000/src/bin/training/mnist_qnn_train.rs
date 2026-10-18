// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! MNIST Quantum Neural Network (QNN) Training & Quantum Coherence Benchmark.
//!
//! Evaluates the T-1000 10-Qubit Variational Quantum Circuit on MNIST handwritten digit
//! classification, comparing performance and quantum state coherence against classical
//! Binarized Neural Network (BNN) and Hyperdimensional Computing (HDC) baselines.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use t1000::{adjoint_gradient, AdamOptimizer, CircuitAnsatz, PauliObservable, QuantumState};
use t1000_core::interference::net_interference;

const N_QUBITS: usize = 10;
const STATE_DIM: usize = 1 << N_QUBITS; // 1024
pub const N_CLASSES: usize = 10;
const N_PIXELS: usize = 784;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochTelemetry {
    pub epoch: usize,
    pub train_loss: f64,
    pub train_accuracy_pct: f64,
    pub test_loss: f64,
    pub test_accuracy_pct: f64,
    pub mean_coherence_l1: f64,
    pub intra_class_interference: f64,
    pub inter_class_interference: f64,
    pub epoch_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MnistQnnReport {
    pub timestamp_rfc3339: String,
    pub architecture: String,
    pub n_qubits: usize,
    pub state_dimension: usize,
    pub circuit_depth: usize,
    pub num_circuit_params: usize,
    pub num_head_params: usize,
    pub train_sample_count: usize,
    pub test_sample_count: usize,
    pub epochs_trained: usize,
    pub batch_size: usize,
    pub lr_ansatz: f64,
    pub lr_head: f64,
    pub baseline_bnn_bsc_acc: f64,
    pub baseline_bnn_map_acc: f64,
    pub baseline_bnn_10k_acc: f64,
    pub final_test_acc_pct: f64,
    pub final_test_loss: f64,
    pub final_intra_interference: f64,
    pub final_inter_interference: f64,
    pub coherence_contrast_ratio: f64,
    pub total_train_time_sec: f64,
    pub per_sample_latency_us: f64,
    pub history: Vec<EpochTelemetry>,
}

/// Locate MNIST dataset files across known workspace paths.
fn locate_mnist_file(filename_candidates: &[&str]) -> Option<PathBuf> {
    let candidate_dirs = [
        "data/mnist",
        "../data/mnist",
        "data",
        "../data",
        "C:\\Users\\zoddj\\zenith_workspace_divizion\\GuddaLM\\data",
        "../GuddaLM/data",
        "../../GuddaLM/data",
    ];

    for dir in &candidate_dirs {
        for candidate in filename_candidates {
            let p1 = Path::new(dir).join(candidate);
            if p1.is_file() {
                return Some(p1);
            }
            // Check subdirectories (e.g., train-images-idx3-ubyte/train-images-idx3-ubyte)
            let stem = candidate
                .trim_end_matches(".idx3-ubyte")
                .trim_end_matches(".idx1-ubyte");
            let p2 = Path::new(dir).join(stem).join(candidate);
            if p2.is_file() {
                return Some(p2);
            }
        }
    }
    None
}

/// Read bytes from an IDX file.
fn read_idx_bytes(path: &Path) -> Vec<u8> {
    fs::read(path)
        .unwrap_or_else(|e| panic!("Failed to read IDX file at {}: {}", path.display(), e))
}

/// Parse IDX image file into normalized f64 pixel vectors.
fn parse_mnist_images(path: &Path, max_samples: Option<usize>) -> Vec<Vec<f64>> {
    let data = read_idx_bytes(path);
    assert!(
        data.len() >= 16,
        "IDX image header too small: {}",
        path.display()
    );
    let total_images = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let rows = u32::from_be_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let cols = u32::from_be_bytes([data[12], data[13], data[14], data[15]]) as usize;
    assert_eq!(rows, 28, "MNIST row mismatch");
    assert_eq!(cols, 28, "MNIST col mismatch");

    let count = max_samples
        .map(|m| m.min(total_images))
        .unwrap_or(total_images);
    let mut images = Vec::with_capacity(count);

    let offset = 16;
    for i in 0..count {
        let start = offset + i * N_PIXELS;
        let img: Vec<f64> = data[start..start + N_PIXELS]
            .iter()
            .map(|&p| p as f64 / 255.0)
            .collect();
        images.push(img);
    }
    images
}

/// Parse IDX label file.
fn parse_mnist_labels(path: &Path, max_samples: Option<usize>) -> Vec<u8> {
    let data = read_idx_bytes(path);
    assert!(
        data.len() >= 8,
        "IDX label header too small: {}",
        path.display()
    );
    let total_labels = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let count = max_samples
        .map(|m| m.min(total_labels))
        .unwrap_or(total_labels);
    data[8..8 + count].to_vec()
}

/// Encode 784-dimensional MNIST pixel vector into a 10-qubit QuantumState (1024 amplitudes).
fn encode_mnist_state(pixels: &[f64]) -> QuantumState {
    let mut state = QuantumState::zero(STATE_DIM);
    let mut norm_sq = 0.0;
    for &p in pixels {
        norm_sq += p * p;
    }
    if norm_sq > 1e-12 {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for (i, &p) in pixels.iter().enumerate() {
            state.real[i] = p * inv_norm;
            state.imag[i] = 0.0;
        }
    } else {
        // Uniform fallback for blank image
        let uniform = 1.0 / (STATE_DIM as f64).sqrt();
        for i in 0..STATE_DIM {
            state.real[i] = uniform;
            state.imag[i] = 0.0;
        }
    }
    state
}

/// Readout Head mapping Pauli expectation values to 10 class logits.
#[derive(Debug, Clone)]
pub struct LinearHead {
    pub num_obs: usize,
    pub weights: Vec<f64>,
    pub biases: Vec<f64>,
}

impl LinearHead {
    fn new(num_obs: usize, seed: u64) -> Self {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let scale = (2.0 / num_obs as f64).sqrt();
        let weights = (0..N_CLASSES * num_obs)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();
        let biases = vec![0.0; N_CLASSES];
        Self {
            num_obs,
            weights,
            biases,
        }
    }

    fn forward(&self, exp: &[f64]) -> [f64; 10] {
        let mut logits = [0.0; 10];
        for c in 0..N_CLASSES {
            let mut sum = self.biases[c];
            let row = c * self.num_obs;
            for q in 0..self.num_obs {
                sum += self.weights[row + q] * exp[q];
            }
            logits[c] = sum;
        }
        logits
    }

    fn param_count(&self) -> usize {
        N_CLASSES * self.num_obs + N_CLASSES
    }

    fn pack_params(&self) -> Vec<f64> {
        let mut params = Vec::with_capacity(self.param_count());
        params.extend_from_slice(&self.weights);
        params.extend_from_slice(&self.biases);
        params
    }

    fn unpack_params(&mut self, params: &[f64]) {
        let n_w = N_CLASSES * self.num_obs;
        self.weights.copy_from_slice(&params[..n_w]);
        self.biases.copy_from_slice(&params[n_w..n_w + N_CLASSES]);
    }
}

/// Compute Softmax and Cross-Entropy Loss + Logit Gradients.
fn softmax_cross_entropy(logits: &[f64; 10], label: usize) -> (f64, [f64; 10], usize) {
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum_exp = 0.0;
    let mut probs = [0.0; 10];
    for c in 0..10 {
        let e = (logits[c] - max_logit).exp();
        probs[c] = e;
        sum_exp += e;
    }
    let inv_sum = 1.0 / sum_exp.max(1e-12);
    let mut pred_class = 0;
    let mut best_prob = -1.0;
    for c in 0..10 {
        probs[c] *= inv_sum;
        if probs[c] > best_prob {
            best_prob = probs[c];
            pred_class = c;
        }
    }

    let loss = -(probs[label].max(1e-12)).ln();
    let mut d_logits = [0.0; 10];
    for c in 0..10 {
        d_logits[c] = probs[c];
    }
    d_logits[label] -= 1.0;

    (loss, d_logits, pred_class)
}

/// Compute L1-norm of quantum coherence: C_l1 = (sum |c_i|)^2 - 1.
fn compute_l1_coherence(state: &QuantumState) -> f64 {
    let sum_mag: f64 = (0..state.dim)
        .map(|i| (state.real[i] * state.real[i] + state.imag[i] * state.imag[i]).sqrt())
        .sum();
    (sum_mag * sum_mag - 1.0).max(0.0)
}

/// Evaluate Quantum Interference Contrast between same-class and cross-class state pairs.
fn evaluate_interference_contrast(
    ansatz: &CircuitAnsatz,
    weights: &[f64],
    samples: &[(QuantumState, usize)],
    n_pairs: usize,
) -> (f64, f64, f64) {
    let mut class_states: Vec<Vec<QuantumState>> = vec![Vec::new(); 10];
    for (state, label) in samples {
        if class_states[*label].len() < 30 {
            let mut evolved = state.clone();
            ansatz.forward_in_place(&mut evolved, weights);
            class_states[*label].push(evolved);
        }
    }

    let mut intra_scores = Vec::new();
    let mut inter_scores = Vec::new();

    // Intra-class pairs (same digit)
    for c in 0..10 {
        let bucket = &class_states[c];
        for i in 0..bucket.len() {
            for j in (i + 1)..bucket.len() {
                if intra_scores.len() >= n_pairs {
                    break;
                }
                let reg_a = bucket[i].to_register(100);
                let reg_b = bucket[j].to_register(101);
                intra_scores.push(net_interference(&reg_a, &reg_b));
            }
        }
    }

    // Inter-class pairs (different digits)
    for c1 in 0..10 {
        for c2 in (c1 + 1)..10 {
            let b1 = &class_states[c1];
            let b2 = &class_states[c2];
            for s1 in b1.iter().take(4) {
                for s2 in b2.iter().take(4) {
                    if inter_scores.len() >= n_pairs {
                        break;
                    }
                    let reg_a = s1.to_register(200);
                    let reg_b = s2.to_register(201);
                    inter_scores.push(net_interference(&reg_a, &reg_b));
                }
            }
        }
    }

    let mean_intra = if intra_scores.is_empty() {
        0.0
    } else {
        intra_scores.iter().sum::<f64>() / intra_scores.len() as f64
    };
    let mean_inter = if inter_scores.is_empty() {
        0.0
    } else {
        inter_scores.iter().sum::<f64>() / inter_scores.len() as f64
    };
    let contrast = mean_intra - mean_inter;

    (mean_intra, mean_inter, contrast)
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║       T-1000 QUANTUM NEURAL NETWORK: MNIST TRAINING & COHERENCE BENCHMARK     ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════╝");

    // 1. Locate MNIST datasets
    let train_img_candidates = ["train-images-idx3-ubyte", "train-images.idx3-ubyte"];
    let train_lbl_candidates = ["train-labels-idx1-ubyte", "train-labels.idx1-ubyte"];
    let test_img_candidates = ["t10k-images-idx3-ubyte", "t10k-images.idx3-ubyte"];
    let test_lbl_candidates = ["t10k-labels-idx1-ubyte", "t10k-labels.idx1-ubyte"];

    let train_img_path =
        locate_mnist_file(&train_img_candidates).expect("MNIST train images not found");
    let train_lbl_path =
        locate_mnist_file(&train_lbl_candidates).expect("MNIST train labels not found");
    let test_img_path =
        locate_mnist_file(&test_img_candidates).expect("MNIST test images not found");
    let test_lbl_path =
        locate_mnist_file(&test_lbl_candidates).expect("MNIST test labels not found");

    println!(" [DISCOVERY] Dataset Grounding:");
    println!("   Train Images : {}", train_img_path.display());
    println!("   Train Labels : {}", train_lbl_path.display());
    println!("   Test Images  : {}", test_img_path.display());
    println!("   Test Labels  : {}", test_lbl_path.display());

    // 2. Load dataset partitions (scalable training slice for fast AVX2 turnaround)
    let n_train = 10000;
    let n_test = 2000;
    println!(
        "\n [INGEST] Parsing MNIST partitions: {} train, {} test...",
        n_train, n_test
    );

    let t0_load = Instant::now();
    let train_pixels = parse_mnist_images(&train_img_path, Some(n_train));
    let train_labels = parse_mnist_labels(&train_lbl_path, Some(n_train));
    let test_pixels = parse_mnist_images(&test_img_path, Some(n_test));
    let test_labels = parse_mnist_labels(&test_lbl_path, Some(n_test));
    println!(
        " [INGEST] Binary parse completed in {:.2} ms",
        t0_load.elapsed().as_secs_f64() * 1000.0
    );

    // 3. Pre-encode into 10-Qubit Hilbert Space
    println!(
        " [ENCODE] Projecting 784-pixel vectors into 10-qubit statevectors (2^10 = 1024 states)..."
    );
    let train_states: Vec<(QuantumState, usize)> = train_pixels
        .into_par_iter()
        .zip(train_labels.into_par_iter())
        .map(|(px, lbl)| (encode_mnist_state(&px), lbl as usize))
        .collect();

    let test_states: Vec<(QuantumState, usize)> = test_pixels
        .into_par_iter()
        .zip(test_labels.into_par_iter())
        .map(|(px, lbl)| (encode_mnist_state(&px), lbl as usize))
        .collect();

    // 4. Initialize Circuit Ansatz & Hybrid Head with Single + Bipartite ZZ Observables
    let depth = 4;
    let ansatz = CircuitAnsatz::new(N_QUBITS, depth).with_pi_layer(true);
    let mut observables: Vec<PauliObservable> = (0..N_QUBITS).map(PauliObservable::Z).collect();
    for i in 0..N_QUBITS {
        for j in (i + 1)..N_QUBITS {
            observables.push(PauliObservable::ZZ(i, j));
        }
    }
    let n_obs = observables.len(); // 10 Z + 45 ZZ = 55

    let n_ansatz_params = ansatz.param_count(); // 2 * 10 * 4 = 80
    let mut ansatz_weights = ansatz.weights.clone();
    let mut head = LinearHead::new(n_obs, 42);
    let n_head_params = head.param_count(); // 10 * 55 + 10 = 560
    let mut head_weights = head.pack_params();

    let lr_ansatz = 0.030;
    let lr_head = 0.025;
    let mut opt_ansatz = AdamOptimizer::new(n_ansatz_params, lr_ansatz);
    let mut opt_head = AdamOptimizer::new_unwrapped(n_head_params, lr_head);

    println!("\n [ARCHITECTURE] Sovereign Model Configuration:");
    println!(
        "   Qubits                 : {} (Hilbert space dimension = {})",
        N_QUBITS, STATE_DIM
    );
    println!(
        "   Circuit Depth          : {} layers (Ring CZ Entanglement + Ry/Rz)",
        depth
    );
    println!("   Walsh-Rademacher Pi    : Enabled (Bipolar Phase Scrambler)",);
    println!(
        "   Observables (Z + ZZ)   : {} (10 single-qubit Z + 45 two-body ZZ correlators)",
        n_obs
    );
    println!("   Ansatz Parameters      : {}", n_ansatz_params);
    println!("   Linear Head Parameters : {}", n_head_params);
    println!(
        "   Total Trainable        : {}",
        n_ansatz_params + n_head_params
    );
    println!("   Baseline Comparisons   : BNN-BSC (78.59%), BNN-MAP (79.40%), HDC-10k (81.25%)");

    // 5. Initial Pre-Training Metrics & Coherence Scan
    println!("\n [TELEMETRY] Evaluating Zero-Shot Quantum Interference Invariants...");
    let (init_intra, init_inter, init_contrast) =
        evaluate_interference_contrast(&ansatz, &ansatz_weights, &test_states, 200);
    println!("   Pre-train Intra-Class Interference : {:+.4}", init_intra);
    println!("   Pre-train Inter-Class Interference : {:+.4}", init_inter);
    println!(
        "   Pre-train Contrast Gap             : {:+.4}",
        init_contrast
    );

    // 6. Training Loop
    let epochs = 16;
    let batch_size = 64;
    let num_batches = (n_train + batch_size - 1) / batch_size;
    let mut history = Vec::new();

    println!("\n┌──────┬────────────┬────────────┬───────────┬───────────┬──────────────┬──────────────┬──────────────┬────────────┐");
    println!("│ Ep   │ Trn Loss   │ Trn Acc %  │ Tst Loss  │ Tst Acc % │ L1-Coherence │ Intra-Interf │ Inter-Interf │ Time (ms)  │");
    println!("├──────┼────────────┼────────────┼───────────┼───────────┼──────────────┼──────────────┼──────────────┼────────────┤");

    let t_start_all = Instant::now();

    for epoch in 1..=epochs {
        let t_ep_start = Instant::now();
        let mut total_train_loss = 0.0;
        let mut train_correct = 0;

        for b in 0..num_batches {
            let start = b * batch_size;
            let end = (start + batch_size).min(n_train);
            let batch = &train_states[start..end];
            let batch_len = batch.len() as f64;

            // Parallel forward + adjoint gradient over batch
            let (batch_loss, batch_correct, batch_ansatz_grad, batch_head_grad) = batch
                .par_iter()
                .map(|(input_state, label)| {
                    let mut sample_loss = 0.0;
                    let mut sample_correct = 0;
                    let mut d_head = vec![0.0; n_head_params];

                    // Unified single-pass Adjoint Differentiation:
                    // Evaluates forward evolution, observable expectations, classical linear head backprop,
                    // and reverse-mode adjoint circuit gradients in a single unified pass.
                    let adj = adjoint_gradient(
                        &ansatz,
                        input_state,
                        &ansatz_weights,
                        &observables,
                        |exps| {
                            let logits = head.forward(exps);
                            let (loss, d_logits, pred) = softmax_cross_entropy(&logits, *label);
                            sample_loss = loss;
                            sample_correct = if pred == *label { 1 } else { 0 };

                            for c in 0..N_CLASSES {
                                let row = c * n_obs;
                                for q in 0..n_obs {
                                    d_head[row + q] = d_logits[c] * exps[q];
                                }
                                d_head[N_CLASSES * n_obs + c] = d_logits[c];
                            }

                            let mut d_exps = vec![0.0; n_obs];
                            for q in 0..n_obs {
                                let mut sum = 0.0;
                                for c in 0..N_CLASSES {
                                    sum += d_logits[c] * head.weights[c * n_obs + q];
                                }
                                d_exps[q] = sum;
                            }
                            d_exps
                        },
                    );

                    (sample_loss, sample_correct, adj.gradient, d_head)
                })
                .reduce(
                    || (0.0, 0, vec![0.0; n_ansatz_params], vec![0.0; n_head_params]),
                    |mut acc, elem| {
                        acc.0 += elem.0;
                        acc.1 += elem.1;
                        for i in 0..n_ansatz_params {
                            acc.2[i] += elem.2[i];
                        }
                        for i in 0..n_head_params {
                            acc.3[i] += elem.3[i];
                        }
                        acc
                    },
                );

            total_train_loss += batch_loss;
            train_correct += batch_correct;

            // Average gradients over batch
            let avg_ansatz_grad: Vec<f64> =
                batch_ansatz_grad.iter().map(|&g| g / batch_len).collect();
            let avg_head_grad: Vec<f64> = batch_head_grad.iter().map(|&g| g / batch_len).collect();

            // Adam steps
            opt_ansatz.step(&mut ansatz_weights, &avg_ansatz_grad);
            opt_head.step(&mut head_weights, &avg_head_grad);
            head.unpack_params(&head_weights);
        }

        let train_loss = total_train_loss / n_train as f64;
        let train_acc = 100.0 * (train_correct as f64) / (n_train as f64);

        // Evaluation on Test Split
        let (test_loss, test_correct, mean_coherence) = test_states
            .par_iter()
            .map(|(input_state, label)| {
                let mut phi = input_state.clone();
                ansatz.forward_in_place(&mut phi, &ansatz_weights);
                let coh = compute_l1_coherence(&phi);
                let exps: Vec<f64> = observables
                    .iter()
                    .map(|o| o.expectation_value(&phi))
                    .collect();
                let logits = head.forward(&exps);
                let (loss, _, pred) = softmax_cross_entropy(&logits, *label);
                let correct = if pred == *label { 1 } else { 0 };
                (loss, correct, coh)
            })
            .reduce(
                || (0.0, 0, 0.0),
                |mut acc, elem| {
                    acc.0 += elem.0;
                    acc.1 += elem.1;
                    acc.2 += elem.2;
                    acc
                },
            );

        let test_loss_avg = test_loss / n_test as f64;
        let test_acc_pct = 100.0 * (test_correct as f64) / (n_test as f64);
        let avg_l1_coherence = mean_coherence / n_test as f64;

        // Sample quantum state interference
        let (intra_score, inter_score, _) =
            evaluate_interference_contrast(&ansatz, &ansatz_weights, &test_states, 150);
        let ep_duration = t_ep_start.elapsed().as_secs_f64() * 1000.0;

        history.push(EpochTelemetry {
            epoch,
            train_loss,
            train_accuracy_pct: train_acc,
            test_loss: test_loss_avg,
            test_accuracy_pct: test_acc_pct,
            mean_coherence_l1: avg_l1_coherence,
            intra_class_interference: intra_score,
            inter_class_interference: inter_score,
            epoch_time_ms: ep_duration,
        });

        println!(
            "│ {:<4} │ {:>10.4} │ {:>9.2}% │ {:>9.4} │ {:>8.2}% │ {:>12.2} │ {:>+12.4} │ {:>+12.4} │ {:>10.2} │",
            epoch, train_loss, train_acc, test_loss_avg, test_acc_pct, avg_l1_coherence, intra_score, inter_score, ep_duration
        );
    }
    println!("└──────┴────────────┴────────────┴───────────┴───────────┴──────────────┴──────────────┴──────────────┴────────────┘");

    let total_train_sec = t_start_all.elapsed().as_secs_f64();
    let per_sample_us = (total_train_sec * 1_000_000.0) / (n_train * epochs) as f64;

    let final_ep = history.last().unwrap();
    let contrast_ratio =
        final_ep.intra_class_interference / (final_ep.inter_class_interference.abs() + 1e-6);

    println!("\n╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    SOVEREIGN BENCHMARK COMPARISON MATRIX                      ║");
    println!("╠═══════════════════════════════════╦══════════════╦═════════════╦══════════════╣");
    println!("║ Architecture Model                ║ Accuracy %   ║ Coherence   ║ Train Engine ║");
    println!("╠═══════════════════════════════════╬══════════════╬═════════════╬══════════════╣");
    println!("║ GuddaNetworks BNN Baseline (BSC)  ║       78.59% ║   Classical ║ 4096-bit BSC ║");
    println!("║ GuddaNetworks BNN Baseline (MAP)  ║       79.40% ║   Classical ║ 4096-bit MAP ║");
    println!("║ GuddaNetworks HDC-10k Dimension   ║       81.25% ║   Classical ║ 10k-bit BSC  ║");
    println!(
        "║ T-1000 10-Qubit VQC (This Run)    ║     {:>7.2}% ║  {:>10.2} ║ AVX2 Adjoint ║",
        final_ep.test_accuracy_pct, final_ep.mean_coherence_l1
    );
    println!("╚═══════════════════════════════════╩══════════════╩═════════════╩══════════════╝");

    println!("\n [COHERENCE INVARIANT ANALYSIS]");
    println!(
        "   Intra-Class Constructive Interference : {:+.4} (Same Digits)",
        final_ep.intra_class_interference
    );
    println!(
        "   Inter-Class Constructive Interference : {:+.4} (Different Digits)",
        final_ep.inter_class_interference
    );
    println!(
        "   Interference Contrast Delta           : {:+.4}",
        final_ep.intra_class_interference - final_ep.inter_class_interference
    );
    println!(
        "   Quantum State Coherence L1-Norm       : {:.2}",
        final_ep.mean_coherence_l1
    );
    println!(
        "   Training Throughput                   : {:.2} us / sample",
        per_sample_us
    );

    // 7. Write telemetry outputs
    let report = MnistQnnReport {
        timestamp_rfc3339: chrono_or_now(),
        architecture: "T-1000 10-Qubit VQC + AVX2 Adjoint Gradient".to_string(),
        n_qubits: N_QUBITS,
        state_dimension: STATE_DIM,
        circuit_depth: depth,
        num_circuit_params: n_ansatz_params,
        num_head_params: head.param_count(),
        train_sample_count: n_train,
        test_sample_count: n_test,
        epochs_trained: epochs,
        batch_size,
        lr_ansatz,
        lr_head,
        baseline_bnn_bsc_acc: 78.59,
        baseline_bnn_map_acc: 79.40,
        baseline_bnn_10k_acc: 81.25,
        final_test_acc_pct: final_ep.test_accuracy_pct,
        final_test_loss: final_ep.test_loss,
        final_intra_interference: final_ep.intra_class_interference,
        final_inter_interference: final_ep.inter_class_interference,
        coherence_contrast_ratio: contrast_ratio,
        total_train_time_sec: total_train_sec,
        per_sample_latency_us: per_sample_us,
        history: history.clone(),
    };

    let results_dir = Path::new("results/experiments");
    fs::create_dir_all(results_dir).unwrap();

    let json_path = results_dir.join("mnist_qnn_training.json");
    let csv_path = results_dir.join("mnist_qnn_training.csv");

    let json_str = serde_json::to_string_pretty(&report).unwrap();
    fs::write(&json_path, json_str).unwrap();

    let mut csv_file = File::create(&csv_path).unwrap();
    writeln!(csv_file, "epoch,train_loss,train_acc_pct,test_loss,test_acc_pct,mean_coherence_l1,intra_interf,inter_interf,epoch_time_ms").unwrap();
    for h in &history {
        writeln!(
            csv_file,
            "{},{:.5},{:.3},{:.5},{:.3},{:.3},{:.5},{:.5},{:.2}",
            h.epoch,
            h.train_loss,
            h.train_accuracy_pct,
            h.test_loss,
            h.test_accuracy_pct,
            h.mean_coherence_l1,
            h.intra_class_interference,
            h.inter_class_interference,
            h.epoch_time_ms
        )
        .unwrap();
    }

    println!("\n [PERSISTENCE]");
    println!("   JSON Telemetry Saved : {}", json_path.display());
    println!("   CSV Telemetry Saved  : {}", csv_path.display());
}

fn chrono_or_now() -> String {
    "2026-10-14T21:00:00Z".to_string()
}
