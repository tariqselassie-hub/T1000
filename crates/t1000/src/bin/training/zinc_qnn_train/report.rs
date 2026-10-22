use super::*;
use t1000::zinc_holdout_eval::{holdout_metrics, print_ledger, target_columns};

/// Holdout evaluation: per-property scores plus screening speed and unitarity.
pub struct Evaluation {
    pub scores: [PropertyMetric; 3],
    pub eval_ms: f64,
    pub per_mol_latency_us: f64,
    pub throughput_mol_sec: f64,
    pub norm: f64,
}

pub fn evaluate(
    model: &Model,
    test: &[ZincMolecule],
    weights: &[f64],
    head: &[AffineFit; 3],
    init_mse: [f64; 3],
    state: &mut QuantumState,
) -> Evaluation {
    println!("\n{DOUBLE_RULE}");
    println!(
        " [FINAL EVALUATION] {} HOLDOUT MOLECULES EVALUATION",
        test.len()
    );
    println!("{DOUBLE_RULE}");

    let t_eval_start = Instant::now();
    let preds = model.predict(test, weights, head, state);
    let eval_duration = t_eval_start.elapsed();
    let eval_ms = eval_duration.as_secs_f64() * 1000.0;
    let n_eval = test.len() as f64;
    let per_mol_latency_us = (eval_duration.as_secs_f64() * 1_000_000.0) / n_eval;
    let throughput_mol_sec = n_eval / eval_duration.as_secs_f64();
    let norm = compute_state_norm(state);

    let scores = holdout_metrics(&preds, &target_columns(test), init_mse);
    Evaluation {
        scores,
        eval_ms,
        per_mol_latency_us,
        throughput_mol_sec,
        norm,
    }
}

/// Post-hoc affine calibration of the circuit readouts (fit on training molecules only),
/// then the speed lines and the final ledger.
pub fn print_evaluation(
    model: &Model,
    (train, test): (&[ZincMolecule], &[ZincMolecule]),
    weights: &[f64],
    ev: &Evaluation,
    state: &mut QuantumState,
) {
    let train_z = model.collect_readouts(train, weights, state);
    let test_z = model.collect_readouts(test, weights, state);
    let train_y: Vec<[f64; 3]> = train.iter().map(targets_of).collect();
    let test_y: Vec<[f64; 3]> = test.iter().map(targets_of).collect();
    t1000::calibration::calibrate(&train_z, &train_y, &test_z, &test_y)
        .print_table(ev.scores.each_ref().map(|s| s.tuned_mse), train.len());

    println!(
        "  Evaluation Time      : {:>8.2} ms ({} molecules)",
        ev.eval_ms,
        test.len()
    );
    println!(
        "  Per-Molecule Latency : {:>8.2} μs/molecule",
        ev.per_mol_latency_us
    );
    println!(
        "  Screening Throughput : {:>8.1} molecules/second",
        ev.throughput_mol_sec
    );
    println!(
        "  Unitary Norm Fidelity: {:>8.6} (Strict Unitary Preservation)",
        ev.norm
    );

    print_ledger(
        "║                    ZINC 10k QUANTUM NEURAL NETWORK HOLDOUT EVALUATION LEDGER                          ║",
        &ev.scores,
    );
}

pub fn load_samples(args: &Args) -> Option<(Vec<ZincMolecule>, Vec<ZincMolecule>)> {
    let train_path =
        resolve_workspace_path("tests/benchmarks/zinc_molecular/outputs/zinc_train_10k.csv");
    let test_path =
        resolve_workspace_path("tests/benchmarks/zinc_molecular/outputs/zinc_test_2k.csv");

    println!(
        "[INIT] Loading ZINC 10k train set from: {}",
        train_path.display()
    );
    let train_samples = match zinc_data::load(&train_path, args.max_train) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[-] Error loading 10k train set: {}", e);
            return None;
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
    let test_samples = match zinc_data::load(&test_path, args.max_eval) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[-] Error loading test set: {}", e);
            return None;
        }
    };
    println!(
        "  -> Successfully loaded {} test molecules.\n",
        test_samples.len()
    );
    Some((train_samples, test_samples))
}

/// Writes the report, recording it only when it beats the saved best.
pub fn save_report(
    args: &Args,
    (train, test): (&[ZincMolecule], &[ZincMolecule]),
    ev: &Evaluation,
    report: &Zinc10kRunReport,
) {
    let out_json_path =
        resolve_workspace_path("tests/benchmarks/zinc_molecular/outputs/zinc_qnn_10k_results.json");
    let Ok(json_str) = serde_json::to_string_pretty(report) else {
        return;
    };
    // Held-out MSE relative to the target variance decides; mean Pearson r guards.
    let targets: [Vec<f64>; 3] = [0, 1, 2].map(|k| test.iter().map(|s| targets_of(s)[k]).collect());
    let holdout = t1000::holdout_score::ZincHoldout::new(
        ev.scores.each_ref().map(|s| s.tuned_mse),
        ev.scores.each_ref().map(|s| s.pearson_r),
        [&targets[0], &targets[1], &targets[2]],
    );
    let settings = format!(
        "qubits={N_QUBITS};depth={DEPTH};epochs={};batch={};lr={};\
         head={};train={};test={}",
        args.num_epochs,
        args.batch_size,
        args.lr,
        if args.calibrated_head {
            "calibrated"
        } else {
            "fixed"
        },
        train.len(),
        test.len()
    );
    let score = holdout.score(&settings);
    println!(
        "\n  Recording score : holdout relative MSE {:.4} | mean Pearson r {:.4}",
        holdout.relative_mse, holdout.mean_pearson_r
    );
    match t1000::results::write_scored(&out_json_path, json_str, &score) {
        Ok(t1000::results::Outcome::KeptBest) => {
            println!("[=] Saved 10k report kept: this run did not improve it");
        }
        Ok(t1000::results::Outcome::Scratch) => {
            println!("[~] 10k report written to a scratch copy only (use --record to save)");
        }
        Ok(t1000::results::Outcome::WrittenVariant) => {
            println!("[SAVED] Different settings from the saved report: recorded as a variant");
        }
        Ok(_) => println!(
            "[SAVED] 10k Training and Evaluation Report exported to: {}",
            out_json_path.display()
        ),
        Err(e) => eprintln!("[!] Could not write the report: {e}"),
    }
}
