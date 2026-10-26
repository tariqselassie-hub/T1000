// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! MNIST Hybrid Quantum-Classical Convolutional Neural Network (Hybrid QCNN).
//!
//! Fuses 10-Qubit Hilbert Space Variational Quantum Circuits (65 Pauli Z, X, and ZZ observables)
//! with a 2D Multi-Scale Spatial Convolutional & Directional Stroke Filter Stream (65 features)
//! through an end-to-end differentiable Non-Linear MLP Readout Head with AVX2 Adjoint Gradients,
//! Cosine Annealing learning rate schedule, and per-neuron activation telemetry.

#[path = "mnist_hybrid_qnn/features.rs"]
mod features;
#[path = "mnist_hybrid_qnn/mlp.rs"]
mod mlp;
#[path = "mnist_hybrid_qnn/telemetry.rs"]
mod telemetry;

use rayon::prelude::*;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use features::{
    encode_mnist_state, extract_spatial_features, locate_mnist_file, parse_mnist_images,
    parse_mnist_labels, N_SPATIAL,
};
use mlp::{softmax_cross_entropy, HybridMlpHead, N_CLASSES, N_FUSED, N_HIDDEN, N_OBSERVABLES};
use telemetry::{
    compute_l1_coherence, evaluate_interference_contrast, EpochTelemetry, MnistHybridQnnReport,
    MnistNeuralProbeReport, NeuralProbeSample, NeuronStat,
};

use t1000::adjoint::AdjointWorkspace;
use t1000::{
    adjoint_gradient_buffered, AdamOptimizer, CircuitAnsatz, PauliObservable, QuantumState,
};

const N_QUBITS: usize = 10;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║     T-1000 HYBRID QUANTUM-CLASSICAL CNN: MNIST COHESION & 90%+ BENCHMARK      ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════╝");

    let train_img_path = locate_mnist_file(&["train-images-idx3-ubyte", "train-images.idx3-ubyte"])
        .expect("MNIST train images not found");
    let train_lbl_path = locate_mnist_file(&["train-labels-idx1-ubyte", "train-labels.idx1-ubyte"])
        .expect("MNIST train labels not found");
    let test_img_path = locate_mnist_file(&["t10k-images-idx3-ubyte", "t10k-images.idx3-ubyte"])
        .expect("MNIST test images not found");
    let test_lbl_path = locate_mnist_file(&["t10k-labels-idx1-ubyte", "t10k-labels.idx1-ubyte"])
        .expect("MNIST test labels not found");

    println!(" [DISCOVERY] Ingesting MNIST Dataset:");
    println!("   Train Images: {}", train_img_path.display());
    println!("   Test Images : {}", test_img_path.display());

    let n_train = 10000;
    let n_test = 2000;
    let t0_load = Instant::now();
    let train_pixels = parse_mnist_images(&train_img_path, Some(n_train));
    let train_labels = parse_mnist_labels(&train_lbl_path, Some(n_train));
    let test_pixels = parse_mnist_images(&test_img_path, Some(n_test));
    let test_labels = parse_mnist_labels(&test_lbl_path, Some(n_test));
    let n_train = train_pixels.len().min(train_labels.len());
    let n_test = test_pixels.len().min(test_labels.len());
    println!(
        " [INGEST] {} train / {} test samples parsed in {:.2} ms",
        n_train,
        n_test,
        t0_load.elapsed().as_secs_f64() * 1000.0
    );

    let test_raw_pixels = test_pixels.clone();

    let t0_prep = Instant::now();
    let train_samples: Vec<(QuantumState, [f64; N_SPATIAL], usize)> = train_pixels
        .into_par_iter()
        .zip(train_labels.into_par_iter())
        .map(|(px, lbl)| {
            let spatial = extract_spatial_features(&px);
            let state = encode_mnist_state(&px, &spatial);
            (state, spatial, lbl as usize)
        })
        .collect();

    let test_samples: Vec<(QuantumState, [f64; N_SPATIAL], usize)> = test_pixels
        .into_par_iter()
        .zip(test_labels.into_par_iter())
        .map(|(px, lbl)| {
            let spatial = extract_spatial_features(&px);
            let state = encode_mnist_state(&px, &spatial);
            (state, spatial, lbl as usize)
        })
        .collect();
    println!(
        " [PRE-PROCESS] 2D Convolutions & Hilbert States mapped in {:.2} ms",
        t0_prep.elapsed().as_secs_f64() * 1000.0
    );

    let depth = 4;
    let ansatz = CircuitAnsatz::new(N_QUBITS, depth).with_pi_layer(true);

    // 65 Observables: 10 Pauli Z, 10 Pauli X (transverse), 45 Pauli ZZ correlators
    let mut observables: Vec<PauliObservable> = (0..N_QUBITS).map(PauliObservable::Z).collect();
    for q in 0..N_QUBITS {
        observables.push(PauliObservable::X(q));
    }
    for i in 0..N_QUBITS {
        for j in (i + 1)..N_QUBITS {
            observables.push(PauliObservable::ZZ(i, j));
        }
    }
    assert_eq!(observables.len(), N_OBSERVABLES);

    let n_ansatz_params = ansatz.param_count();
    let mut ansatz_weights = ansatz.weights.clone();
    let mut head = HybridMlpHead::new(42);
    let n_head_params = head.param_count();
    let mut head_weights = head.pack_params();

    let base_lr_ansatz = 0.025;
    let base_lr_head = 0.005;
    let mut opt_ansatz = AdamOptimizer::new(n_ansatz_params, base_lr_ansatz);
    let mut opt_head = AdamOptimizer::new_unwrapped(n_head_params, base_lr_head);

    println!("\n [ARCHITECTURE] Sovereign Hybrid Model Configuration:");
    println!(
        "   Quantum Register       : {} Qubits (1,024 Hilbert Amplitudes)",
        N_QUBITS
    );
    println!(
        "   Circuit Depth          : {} layers (Ring CZ + Ry/Rz)",
        depth
    );
    println!("   Observables (Z+X+ZZ)   : {} Correlators", N_OBSERVABLES);
    println!(
        "   Spatial Feature Stream : {} Multi-Scale 2D Features",
        N_SPATIAL
    );
    println!(
        "   Total Trainable Params : {} Quantum + {} Classical = {}",
        n_ansatz_params,
        n_head_params,
        n_ansatz_params + n_head_params
    );
    println!("   Hidden MLP Layer       : {} LeakyReLU Neurons", N_HIDDEN);
    println!("   Legacy VQC Baseline    : 84.50% (Linear Head)");

    let epochs = 16;
    let batch_size = 64;
    let num_batches = n_train.div_ceil(batch_size);
    let mut history = Vec::new();

    let mut last_firing_counts = [0usize; N_HIDDEN];
    let mut last_act_sums = [0.0f64; N_HIDDEN];
    let mut last_act_maxs = [0.0f64; N_HIDDEN];

    println!("\n┌──────┬────────────┬────────────┬───────────┬───────────┬──────────────┬──────────────┬──────────────┬────────────┐");
    println!("│ Ep   │ Trn Loss   │ Trn Acc %  │ Tst Loss  │ Tst Acc % │ L1-Coherence │ Intra-Interf │ Inter-Interf │ Time (ms)  │");
    println!("├──────┼────────────┼────────────┼───────────┼───────────┼──────────────┼──────────────┼──────────────┼────────────┤");

    let t_start_all = Instant::now();

    for epoch in 1..=epochs {
        let t_ep_start = Instant::now();

        // Cosine Annealing learning rate schedule
        let progress = (epoch - 1) as f64 / epochs as f64;
        let cosine_factor = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
        opt_ansatz.lr = (base_lr_ansatz * cosine_factor).max(0.005);
        opt_head.lr = (base_lr_head * cosine_factor).max(0.001);

        let mut total_train_loss = 0.0;
        let mut train_correct = 0;

        for b in 0..num_batches {
            let start = b * batch_size;
            let end = (start + batch_size).min(n_train);
            let batch = &train_samples[start..end];
            let batch_len = batch.len() as f64;

            let (batch_loss, batch_correct, batch_ansatz_grad, batch_head_grad) = batch
                .par_iter()
                .map(|(input_state, spatial, label)| {
                    let mut sample_loss = 0.0;
                    let mut sample_correct = 0;
                    let mut local_head_grad = vec![0.0; n_head_params];
                    let mut ws = AdjointWorkspace::new(ansatz.dim, ansatz.n_qubits);

                    let adj = adjoint_gradient_buffered(
                        &ansatz,
                        input_state,
                        &ansatz_weights,
                        &observables,
                        |exps| {
                            let mut fused = [0.0; N_FUSED];
                            fused[..N_OBSERVABLES].copy_from_slice(exps);
                            fused[N_OBSERVABLES..].copy_from_slice(spatial);

                            let (logits, z1, a1) = head.forward(&fused);
                            let (loss, d_logits, pred) = softmax_cross_entropy(&logits, *label);
                            sample_loss = loss;
                            sample_correct = if pred == *label { 1 } else { 0 };

                            let d_fused =
                                head.backward(&fused, &z1, &a1, &d_logits, &mut local_head_grad);
                            d_fused[..N_OBSERVABLES].to_vec()
                        },
                        &mut ws,
                    );

                    (sample_loss, sample_correct, adj.gradient, local_head_grad)
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

            // Apply L2 weight decay (ridge regularization) on classical head
            let weight_decay = 1e-4;
            let avg_ansatz_grad: Vec<f64> =
                batch_ansatz_grad.iter().map(|&g| g / batch_len).collect();
            let avg_head_grad: Vec<f64> = batch_head_grad
                .iter()
                .enumerate()
                .map(|(i, &g)| g / batch_len + weight_decay * head_weights[i])
                .collect();

            opt_ansatz.step(&mut ansatz_weights, &avg_ansatz_grad);
            opt_head.step(&mut head_weights, &avg_head_grad);
            head.unpack_params(&head_weights);
        }

        let train_loss = total_train_loss / n_train as f64;
        let train_acc = 100.0 * f64::from(train_correct) / (n_train as f64);

        // Test evaluation with full per-neuron activation tracking
        let (test_loss, test_correct, mean_coherence, (firing_counts, act_sums, act_maxs)) =
            test_samples
                .par_iter()
                .map(|(input_state, spatial, label)| {
                    let mut phi = input_state.clone();
                    ansatz.forward_in_place(&mut phi, &ansatz_weights);
                    let coh = compute_l1_coherence(&phi);
                    let mut fused = [0.0; N_FUSED];
                    for (i, o) in observables.iter().enumerate() {
                        fused[i] = o.expectation_value(&phi);
                    }
                    fused[N_OBSERVABLES..].copy_from_slice(spatial);

                    let (logits, _z1, a1) = head.forward(&fused);
                    let (loss, _, pred) = softmax_cross_entropy(&logits, *label);
                    let correct = if pred == *label { 1 } else { 0 };

                    let mut local_counts = [0usize; N_HIDDEN];
                    let mut local_sums = [0.0f64; N_HIDDEN];
                    let mut local_maxs = [0.0f64; N_HIDDEN];
                    for h in 0..N_HIDDEN {
                        if a1[h] > 0.0 {
                            local_counts[h] = 1;
                        }
                        local_sums[h] = a1[h];
                        local_maxs[h] = a1[h];
                    }

                    (loss, correct, coh, (local_counts, local_sums, local_maxs))
                })
                .reduce(
                    || {
                        (
                            0.0,
                            0,
                            0.0,
                            ([0; N_HIDDEN], [0.0; N_HIDDEN], [0.0; N_HIDDEN]),
                        )
                    },
                    |mut acc, elem| {
                        acc.0 += elem.0;
                        acc.1 += elem.1;
                        acc.2 += elem.2;
                        for h in 0..N_HIDDEN {
                            (acc.3).0[h] += (elem.3).0[h];
                            (acc.3).1[h] += (elem.3).1[h];
                            (acc.3).2[h] = (acc.3).2[h].max((elem.3).2[h]);
                        }
                        acc
                    },
                );

        last_firing_counts = firing_counts;
        last_act_sums = act_sums;
        last_act_maxs = act_maxs;

        let test_loss_avg = test_loss / n_test as f64;
        let test_acc_pct = 100.0 * f64::from(test_correct) / (n_test as f64);
        let avg_l1_coherence = mean_coherence / n_test as f64;

        let (intra_score, inter_score, _) =
            evaluate_interference_contrast(&ansatz, &ansatz_weights, &test_samples, 150);
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

    let dead_count = last_firing_counts.iter().filter(|&&c| c == 0).count();

    println!("\n╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    SOVEREIGN BENCHMARK COMPARISON MATRIX                      ║");
    println!("╠═══════════════════════════════════╦══════════════╦═════════════╦══════════════╣");
    println!("║ Architecture Model                ║ Accuracy %   ║ Coherence   ║ Train Engine ║");
    println!("╠═══════════════════════════════════╬══════════════╬═════════════╬══════════════╣");
    println!("║ GuddaNetworks BNN Baseline (BSC)  ║       78.59% ║   Classical ║ 4096-bit BSC ║");
    println!("║ GuddaNetworks BNN Baseline (MAP)  ║       79.40% ║   Classical ║ 4096-bit MAP ║");
    println!("║ GuddaNetworks HDC-10k Dimension   ║       81.25% ║   Classical ║ 10k-bit BSC  ║");
    println!("║ T-1000 10-Qubit VQC (Linear Head) ║       84.50% ║      350.94 ║ AVX2 Adjoint ║");
    println!(
        "║ T-1000 Hybrid QCNN (This Run)     ║     {:>7.2}% ║  {:>10.2} ║ Hybrid AVX2  ║",
        final_ep.test_accuracy_pct, final_ep.mean_coherence_l1
    );
    println!("╚═══════════════════════════════════╩══════════════╩═════════════╩══════════════╝");
    println!(
        " [NEURON HEALTH] Hidden Neurons Active: {}/128 | Dead: {}/128",
        128 - dead_count,
        dead_count
    );

    save_report(
        &history,
        n_train,
        n_test,
        depth,
        n_ansatz_params,
        n_head_params,
        epochs,
        batch_size,
        total_train_sec,
        per_sample_us,
    );

    export_neural_probe(
        &ansatz,
        &ansatz_weights,
        &head,
        &observables,
        &test_samples,
        &test_raw_pixels,
        &last_firing_counts,
        &last_act_sums,
        &last_act_maxs,
        final_ep.test_accuracy_pct,
        final_ep.test_loss,
    );
}

fn export_neural_probe(
    ansatz: &CircuitAnsatz,
    ansatz_weights: &[f64],
    head: &HybridMlpHead,
    observables: &[PauliObservable],
    test_samples: &[(QuantumState, [f64; N_SPATIAL], usize)],
    test_raw_pixels: &[Vec<f64>],
    firing_counts: &[usize; N_HIDDEN],
    act_sums: &[f64; N_HIDDEN],
    act_maxs: &[f64; N_HIDDEN],
    test_acc_pct: f64,
    test_loss: f64,
) {
    let n_test = test_samples.len();
    let mut neuron_stats = Vec::with_capacity(N_HIDDEN);
    let mut dead_neurons = Vec::new();

    for h in 0..N_HIDDEN {
        let is_dead = firing_counts[h] == 0;
        if is_dead {
            dead_neurons.push(h);
        }
        neuron_stats.push(NeuronStat {
            id: h,
            firing_rate: firing_counts[h] as f64 / n_test as f64,
            mean_activation: act_sums[h] / n_test as f64,
            max_activation: act_maxs[h],
            is_dead,
        });
    }

    let mut weights_w1 = vec![vec![0.0; N_FUSED]; N_HIDDEN];
    for h in 0..N_HIDDEN {
        for f in 0..N_FUSED {
            weights_w1[h][f] = head.w1[h * N_FUSED + f];
        }
    }

    let mut weights_w2 = vec![vec![0.0; N_HIDDEN]; N_CLASSES];
    for c in 0..N_CLASSES {
        for h in 0..N_HIDDEN {
            weights_w2[c][h] = head.w2[c * N_HIDDEN + h];
        }
    }

    let mut class_counts = [0usize; 10];
    let mut selected_samples = Vec::new();

    for (idx, (input_state, spatial, label)) in test_samples.iter().enumerate() {
        if class_counts[*label] < 5 {
            class_counts[*label] += 1;

            let mut phi = input_state.clone();
            ansatz.forward_in_place(&mut phi, ansatz_weights);
            let mut quantum_obs = vec![0.0; N_OBSERVABLES];
            let mut fused = [0.0; N_FUSED];
            for (i, o) in observables.iter().enumerate() {
                let val = o.expectation_value(&phi);
                quantum_obs[i] = val;
                fused[i] = val;
            }
            fused[N_OBSERVABLES..].copy_from_slice(spatial);

            let (logits, z1, a1) = head.forward(&fused);
            let (_, _, pred) = softmax_cross_entropy(&logits, *label);

            let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_sum: f64 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
            let probabilities: Vec<f64> = logits
                .iter()
                .map(|&l| (l - max_logit).exp() / exp_sum)
                .collect();
            let confidence = probabilities[pred];

            selected_samples.push(NeuralProbeSample {
                sample_id: idx,
                label: *label,
                prediction: pred,
                is_correct: pred == *label,
                confidence,
                probabilities,
                pixels: test_raw_pixels[idx].clone(),
                spatial_features: spatial.to_vec(),
                quantum_observables: quantum_obs,
                neuron_preactivations: z1.to_vec(),
                neuron_activations: a1.to_vec(),
            });
        }
    }

    let report = MnistNeuralProbeReport {
        timestamp_rfc3339: "2026-10-25T14:00:00Z".to_string(),
        test_accuracy_pct: test_acc_pct,
        test_loss,
        total_neurons: N_HIDDEN,
        dead_neurons_count: dead_neurons.len(),
        active_neurons_count: N_HIDDEN - dead_neurons.len(),
        dead_neurons: dead_neurons.clone(),
        neuron_stats,
        weights_w1,
        weights_w2,
        samples: selected_samples,
    };

    let json_str = serde_json::to_string_pretty(&report).unwrap();

    let _ = fs::create_dir_all("web/data");
    let _ = fs::create_dir_all("results/data");
    let _ = fs::write("web/data/mnist_neural_probe.json", &json_str);
    let _ = fs::write("results/data/mnist_neural_probe.json", &json_str);

    println!("\n [NEURAL PROBE EXPORT]");
    println!(
        "   Active Neurons : {}/{}",
        report.active_neurons_count, report.total_neurons
    );
    println!(
        "   Dead Neurons   : {} ({:?})",
        report.dead_neurons_count, report.dead_neurons
    );
    println!("   Web Telemetry  : web/data/mnist_neural_probe.json");
    println!("   Data Telemetry : results/data/mnist_neural_probe.json");
}

fn save_report(
    history: &[EpochTelemetry],
    n_train: usize,
    n_test: usize,
    depth: usize,
    n_ansatz_params: usize,
    n_head_params: usize,
    epochs: usize,
    batch_size: usize,
    total_train_sec: f64,
    per_sample_us: f64,
) {
    let final_ep = history.last().unwrap();
    let report = MnistHybridQnnReport {
        timestamp_rfc3339: "2026-10-25T14:00:00Z".to_string(),
        architecture: "T-1000 Hybrid Quantum-Classical Convolutional Network".to_string(),
        n_qubits: N_QUBITS,
        circuit_depth: depth,
        num_quantum_params: n_ansatz_params,
        num_classical_head_params: n_head_params,
        total_trainable_params: n_ansatz_params + n_head_params,
        train_samples: n_train,
        test_samples: n_test,
        epochs,
        batch_size,
        baseline_vqc_linear_acc: 84.50,
        final_test_acc_pct: final_ep.test_accuracy_pct,
        final_test_loss: final_ep.test_loss,
        final_intra_interference: final_ep.intra_class_interference,
        final_inter_interference: final_ep.inter_class_interference,
        total_train_time_sec: total_train_sec,
        per_sample_latency_us: per_sample_us,
        history: history.to_vec(),
    };

    let results_dir = Path::new("results/experiments");
    fs::create_dir_all(results_dir).unwrap();
    let json_path = results_dir.join("mnist_hybrid_qnn_training.json");
    let csv_path = results_dir.join("mnist_hybrid_qnn_training.csv");

    let json_str = serde_json::to_string_pretty(&report).unwrap();
    let settings = format!(
        "qubits={};depth={};epochs={};batch={};train={};test={}",
        report.n_qubits,
        report.circuit_depth,
        report.epochs,
        report.batch_size,
        report.train_samples,
        report.test_samples
    );
    let score = t1000::results::Score {
        metric: "test_loss",
        value: report.final_test_loss,
        better: t1000::results::Better::Lower,
        guard: Some(t1000::results::Guard {
            name: "test_accuracy_pct",
            value: report.final_test_acc_pct,
            better: t1000::results::Better::Higher,
            tolerance: 1.0,
        }),
        settings: &settings,
    };
    let outcome = t1000::results::write_scored(&json_path, json_str, &score).unwrap();
    let csv_path = if outcome == t1000::results::Outcome::WrittenVariant {
        t1000::results::variant_path(&csv_path, &settings)
    } else {
        csv_path
    };
    let mut csv_file = t1000::results::create(&csv_path).unwrap();
    writeln!(
        csv_file,
        "epoch,train_loss,train_acc_pct,test_loss,test_acc_pct,mean_coherence_l1,intra_interf,inter_interf,epoch_time_ms"
    )
    .unwrap();
    for h in history {
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
