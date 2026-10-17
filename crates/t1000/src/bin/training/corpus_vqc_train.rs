use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use t1000::{
    tokenize_text, AdamOptimizer, CircuitAnsatz, DataEncodingLayer, EncodingKind, POVMClassifier,
    ParameterShiftEngine, PauliObservable, QuantumFeatureMap, QuantumState,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocTrainingSummary {
    pub file_name: String,
    pub category: String,
    pub token_count: usize,
    pub initial_pred_category: String,
    pub final_pred_category: String,
    pub is_correct: bool,
    pub final_confidence: f64,
    pub z7_exp: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusEpochLog {
    pub epoch: usize,
    pub lr: f64,
    pub train_loss: f64,
    pub accuracy_pct: f64,
    pub max_grad: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusVqcTelemetry {
    pub task: String,
    pub n_documents: usize,
    pub categories: Vec<String>,
    pub n_qubits: usize,
    pub dim: usize,
    pub depth: usize,
    pub use_pi_layer: bool,
    pub num_trainable_parameters: usize,
    pub total_epochs: usize,
    pub initial_lr: f64,
    pub final_lr: f64,
    pub initial_loss: f64,
    pub final_loss: f64,
    pub final_accuracy_pct: f64,
    pub training_time_ms: f64,
    pub document_summaries: Vec<DocTrainingSummary>,
    pub epoch_logs: Vec<CorpusEpochLog>,
}

struct PreparedCorpusDoc {
    file_name: String,
    #[allow(dead_code)]
    path: PathBuf,
    category: String,
    category_idx: usize,
    token_count: usize,
    quantum_state: QuantumState,
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut depth: usize = 3;
    let mut epochs: usize = 20;
    let mut lr: f64 = 0.05;
    let mut decay_gamma: f64 = 0.9;
    let mut decay_step: usize = 5;
    let mut use_pi_layer: bool = false;
    let mut task_mode = "domain".to_string(); // "domain" (5-class) or "sovereign" (binary)
    let mut out_path = PathBuf::from("results/experiments/corpus_vqc_training_run.json");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
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
            "--pi-layer" => {
                use_pi_layer = true;
            }
            "--mode" if i + 1 < args.len() => {
                task_mode = args[i + 1].to_lowercase();
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
    let corpus_root = resolve_workspace_path("corpus");

    println!("=========================================================================");
    println!("   CORPUS VARIATIONAL QUANTUM CLASSIFIER (VQC) TRAINING ENGINE           ");
    println!("=========================================================================");
    println!(" Corpus Directory   : {}", corpus_root.display());
    println!(" Task Mode          : {}", task_mode);
    println!(" Circuit Depth      : {}", depth);
    println!(
        " Pi Phase Layer     : {}",
        if use_pi_layer {
            "ENABLED (±π Rademacher Mask)"
        } else {
            "DISABLED"
        }
    );
    println!(" Trainable Params   : {} θ angles", 2 * 10 * depth);
    println!(
        " LR Schedule        : lr_0={}, decay_gamma={}, step_interval={}",
        lr, decay_gamma, decay_step
    );
    println!(" Telemetry Output   : {}", out_path.display());
    println!("-------------------------------------------------------------------------");

    // 1. Ingest all corpus documents and encode into Hilbert space
    let categories = vec![
        "bridges".to_string(),
        "deen".to_string(),
        "percival".to_string(),
        "python".to_string(),
        "rust".to_string(),
    ];

    let dim = 1024;
    let n_qubits = 10;
    let feature_map = QuantumFeatureMap::new(dim, 1337);

    println!(
        "[*] Ingesting and embedding corpus documents into Hilbert space (dim={})...",
        dim
    );
    let docs = load_and_encode_corpus(&corpus_root, &categories, &feature_map);
    println!(
        "[+] Successfully prepared {} documents across {} categories.",
        docs.len(),
        categories.len()
    );

    for doc in &docs {
        println!(
            "    - [{:<8}] {:<42} ({} tokens)",
            doc.category, doc.file_name, doc.token_count
        );
    }
    println!("-------------------------------------------------------------------------");

    // 2. POVM Subspace Centroid Calibration
    let mut povm = POVMClassifier::new(dim, categories.clone());
    let mut cat_accumulators: std::collections::HashMap<String, QuantumState> =
        std::collections::HashMap::new();
    for doc in &docs {
        let acc = cat_accumulators
            .entry(doc.category.clone())
            .or_insert_with(|| QuantumState::zero(dim));
        acc.superpose(&doc.quantum_state, 1.0);
    }
    for (cat_name, centroid) in &cat_accumulators {
        if let Some(idx) = categories.iter().position(|c| c == cat_name) {
            povm.set_class_centroid(idx, centroid);
        }
    }

    // 3. Initialize VQC Architecture
    let encoding_dummy = DataEncodingLayer::new(n_qubits, EncodingKind::Angle);
    let ansatz = CircuitAnsatz::new(n_qubits, depth).with_pi_layer(use_pi_layer);
    let dummy_obs = PauliObservable::Z(0);

    let num_params = ansatz.param_count();
    let mut weights = ansatz.weights.clone();
    let mut optimizer = AdamOptimizer::new(num_params, lr);
    let engine = ParameterShiftEngine::new(encoding_dummy, ansatz, dummy_obs);

    // Multi-class observables: Qubit k measures activation for Category k
    let observables: Vec<PauliObservable> = (0..categories.len()).map(PauliObservable::Z).collect();

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
        let current_lr = optimizer.step_epoch_decay(ep - 1, decay_gamma, decay_step);

        let mut total_loss = 0.0;
        let mut correct = 0;
        let mut max_grad: f64 = 0.0;

        let mut batch_grads = vec![0.0; num_params];

        for doc in &docs {
            if task_mode == "sovereign" {
                // Binary classification: bridges/deen (+1.0) vs percival/python/rust (-1.0)
                let y_target = if doc.category == "bridges" || doc.category == "deen" {
                    1.0
                } else {
                    -1.0
                };

                let (pred, loss, grads) =
                    engine.compute_loss_gradients_for_state(&doc.quantum_state, y_target, &weights);
                total_loss += loss;

                let predicted_class = if pred >= 0.0 { 1.0 } else { -1.0 };
                if (predicted_class - y_target).abs() < 1e-3 {
                    correct += 1;
                }

                for (bg, &sg) in batch_grads.iter_mut().zip(grads.iter()) {
                    *bg += sg;
                    if sg.abs() > max_grad {
                        max_grad = sg.abs();
                    }
                }
            } else {
                // 5-class domain classification: target vector has +1.0 for true category, -1.0 for others
                let mut targets = vec![-1.0; categories.len()];
                targets[doc.category_idx] = 1.0;

                let (preds, loss, grads) = engine.compute_multi_observable_loss_gradients(
                    &doc.quantum_state,
                    &observables,
                    &targets,
                    &weights,
                );
                total_loss += loss;

                // Predict category with highest Pauli Z expectation value
                let (pred_idx, _) = preds
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or((0, &0.0));

                if pred_idx == doc.category_idx {
                    correct += 1;
                }

                for (bg, &sg) in batch_grads.iter_mut().zip(grads.iter()) {
                    *bg += sg;
                    if sg.abs() > max_grad {
                        max_grad = sg.abs();
                    }
                }
            }
        }

        let n_samples = docs.len() as f64;
        let mean_loss = total_loss / n_samples;
        for g in &mut batch_grads {
            *g /= n_samples;
        }

        if ep == 1 {
            initial_loss = mean_loss;
        }

        optimizer.step(&mut weights, &batch_grads);
        let acc_pct = (correct as f64) / n_samples * 100.0;

        println!(
            "│ {:<5} │ {:<8.5} │ {:<12.5} │ {:<12.1} │ {:<12.4} │ Converging...    │",
            ep, current_lr, mean_loss, acc_pct, max_grad
        );

        epoch_logs.push(CorpusEpochLog {
            epoch: ep,
            lr: current_lr,
            train_loss: mean_loss,
            accuracy_pct: acc_pct,
            max_grad,
        });
    }
    println!(
        "└───────┴──────────┴──────────────┴──────────────┴──────────────┴──────────────────┘\n"
    );

    let total_elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;
    let final_loss = epoch_logs.last().map(|l| l.train_loss).unwrap_or(0.0);
    let final_acc = epoch_logs.last().map(|l| l.accuracy_pct).unwrap_or(0.0);
    let final_lr = epoch_logs.last().map(|l| l.lr).unwrap_or(lr);

    let loss_reduction_pct = if initial_loss > 1e-6 {
        (initial_loss - final_loss) / initial_loss * 100.0
    } else {
        0.0
    };

    // 4. Document-by-document evaluation ledger
    let mut doc_summaries = Vec::new();
    println!("=========================================================================================================");
    println!("   FINAL QUANTUM INFERENCE & CATEGORY DISCRIMINATION LEDGER                                              ");
    println!("=========================================================================================================");
    println!(
        " {:<34} │ {:<8} │ {:<8} │ {:<5} │ {:<7} │ {:<7} │ {:<18}",
        "Document", "Target", "Pred(Z)", "Match", "Z_exp", "<Z_7>", "POVM Top (Prob)"
    );
    println!("───────────────────────────────────┼──────────┼──────────┼───────┼─────────┼─────────┼───────────────────");

    let obs_z7 = PauliObservable::Z(7);

    for doc in &docs {
        let evolved = engine.ansatz.forward(&doc.quantum_state, &weights);
        let povm_probs = povm.predict_probabilities(&evolved);
        let (povm_top_class, povm_top_prob) = povm_probs
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
            .unwrap_or(("unknown".to_string(), 0.0));

        let z7_exp = obs_z7.expectation_value(&evolved);

        if task_mode == "sovereign" {
            let pred = engine.predict_state(&doc.quantum_state, &weights);
            let target_str = if doc.category == "bridges" || doc.category == "deen" {
                "sovereign"
            } else {
                "reference"
            };
            let pred_str = if pred >= 0.0 {
                "sovereign"
            } else {
                "reference"
            };
            let is_correct = target_str == pred_str;
            let status = if is_correct { "MATCH" } else { "MISS" };

            let povm_str = format!("{} ({:.1}%)", povm_top_class, povm_top_prob * 100.0);
            println!(
                " {:<34} │ {:<8} │ {:<8} │ {:<5} │ {:<+7.3} │ {:<+7.3} │ {:<18}",
                doc.file_name, target_str, pred_str, status, pred, z7_exp, povm_str
            );

            doc_summaries.push(DocTrainingSummary {
                file_name: doc.file_name.clone(),
                category: doc.category.clone(),
                token_count: doc.token_count,
                initial_pred_category: "unknown".to_string(),
                final_pred_category: pred_str.to_string(),
                is_correct,
                final_confidence: pred.abs(),
                z7_exp,
            });
        } else {
            let preds =
                engine.predict_multi_observables(&doc.quantum_state, &observables, &weights);
            let (top_idx, &top_z) = preds
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();

            let pred_cat = &categories[top_idx];
            let is_correct = pred_cat == &doc.category;
            let status = if is_correct { "MATCH" } else { "MISS" };

            let povm_str = format!("{} ({:.1}%)", povm_top_class, povm_top_prob * 100.0);
            println!(
                " {:<34} │ {:<8} │ {:<8} │ {:<5} │ {:<+7.3} │ {:<+7.3} │ {:<18}",
                doc.file_name, doc.category, pred_cat, status, top_z, z7_exp, povm_str
            );

            doc_summaries.push(DocTrainingSummary {
                file_name: doc.file_name.clone(),
                category: doc.category.clone(),
                token_count: doc.token_count,
                initial_pred_category: "unknown".to_string(),
                final_pred_category: pred_cat.clone(),
                is_correct,
                final_confidence: top_z,
                z7_exp,
            });
        }
    }

    println!("=========================================================================");
    println!(" Initial Loss       : {:.5}", initial_loss);
    println!(
        " Final Loss         : {:.5} ({:.1}% reduction)",
        final_loss, loss_reduction_pct
    );
    println!(" Final Accuracy     : {:.1}%", final_acc);
    println!(" Final Learning Rate: {:.5}", final_lr);
    println!(" Elapsed Time       : {:.2} ms", total_elapsed_ms);
    println!(" Parameter Updates  : {} steps completed", optimizer.step);
    println!("=========================================================================");

    let telemetry = CorpusVqcTelemetry {
        task: task_mode,
        n_documents: docs.len(),
        categories,
        n_qubits,
        dim,
        depth,
        use_pi_layer,
        num_trainable_parameters: num_params,
        total_epochs: epochs,
        initial_lr: lr,
        final_lr,
        initial_loss,
        final_loss,
        final_accuracy_pct: final_acc,
        training_time_ms: total_elapsed_ms,
        document_summaries: doc_summaries,
        epoch_logs,
    };

    if let Some(parent) = out_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(json_str) = serde_json::to_string_pretty(&telemetry) {
        let _ = fs::write(&out_path, json_str);
        println!(
            "[+] Corpus VQC telemetry exported to: {}",
            out_path.display()
        );
    }
}

fn load_and_encode_corpus(
    corpus_root: &Path,
    categories: &[String],
    feature_map: &QuantumFeatureMap,
) -> Vec<PreparedCorpusDoc> {
    let mut docs = Vec::new();

    for (cat_idx, cat) in categories.iter().enumerate() {
        let cat_dir = corpus_root.join(cat);
        if !cat_dir.exists() {
            continue;
        }

        if let Ok(entries) = fs::read_dir(&cat_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                    if ext == "md" || ext == "percival" || ext == "rs" || ext == "py" {
                        if let Ok(content) = fs::read_to_string(&path) {
                            let file_name = path.file_name().unwrap().to_string_lossy().to_string();
                            let tokens = tokenize_text(&content);
                            let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();

                            let quantum_state = feature_map.encode_sequence(&token_refs);

                            docs.push(PreparedCorpusDoc {
                                file_name,
                                path,
                                category: cat.clone(),
                                category_idx: cat_idx,
                                token_count: tokens.len(),
                                quantum_state,
                            });
                        }
                    }
                }
            }
        }
    }

    docs
}

fn resolve_workspace_path(sub_path: &str) -> PathBuf {
    if let Ok(cwd) = env::current_dir() {
        let mut curr = cwd.as_path();
        for _ in 0..6 {
            let candidate = curr.join(sub_path);
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
    PathBuf::from(sub_path)
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
