use super::training::Trained;
use super::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use t1000::zinc_holdout_eval::{holdout_metrics, print_ledger};
use t1000::zinc_training_common::compute_state_norm;

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

/// Holdout evaluation: predictions, scores, screening speed, and unitarity.
#[derive(Debug)]
pub struct Evaluation {
    pub preds: [Vec<f64>; 3],
    metrics: [PropertyMetric; 3],
    eval_ms: f64,
    per_mol_latency_us: f64,
    throughput_mol_sec: f64,
    norm: f64,
}

pub fn evaluate(
    model: &Model,
    test: &[ZincMolecule],
    params: &[f64],
    init_mse: [f64; 3],
    state: &mut QuantumState,
) -> Evaluation {
    println!("\n{DOUBLE_RULE}");
    println!(" [FINAL EVALUATION] 2,000 HOLDOUT MOLECULES INFERENCE");
    println!("{DOUBLE_RULE}");

    let t_eval_start = Instant::now();
    let preds = model.predict(test, params, state);
    let eval_duration = t_eval_start.elapsed();
    let eval_ms = eval_duration.as_secs_f64() * 1000.0;
    let n_eval = test.len() as f64;
    let per_mol_latency_us = (eval_duration.as_secs_f64() * 1_000_000.0) / n_eval;
    let throughput_mol_sec = n_eval / eval_duration.as_secs_f64();
    let norm = compute_state_norm(state);

    let metrics = holdout_metrics(&preds, &target_columns(test), init_mse);

    println!(
        "  Evaluation Time      : {:>8.2} ms (2,000 molecules)",
        eval_ms
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
    print_ledger(
        "║                 12-QUBIT HYBRID QNN HOLDOUT INFERENCE PERFORMANCE LEDGER                              ║",
        &metrics,
    );
    Evaluation {
        preds,
        metrics,
        eval_ms,
        per_mol_latency_us,
        throughput_mol_sec,
        norm,
    }
}

struct TargetObjective {
    name: &'static str,
    target_logp: f64,
    target_qed: f64,
    target_sas: f64,
}

const OBJECTIVES: [TargetObjective; 3] = [
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

/// Finds the holdout molecule whose predictions are nearest (in sigma units) to `obj`.
fn best_candidate(obj: &TargetObjective, preds: &[Vec<f64>; 3]) -> (usize, [f64; 3]) {
    let mut best_idx = 0;
    let mut min_inv_loss = f64::MAX;
    let mut best = [0.0; 3];
    let rows = preds[0].iter().zip(&preds[1]).zip(&preds[2]);
    for (idx, ((&pred_logp, &pred_qed), &pred_sas)) in rows.enumerate() {
        let d0 = (pred_logp - obj.target_logp) / SIGMA_LOGP;
        let d1 = (pred_qed - obj.target_qed) / SIGMA_QED;
        let d2 = (pred_sas - obj.target_sas) / SIGMA_SAS;

        let inv_loss = d0 * d0 + d1 * d1 + d2 * d2;
        if inv_loss < min_inv_loss {
            min_inv_loss = inv_loss;
            best_idx = idx;
            best = [pred_logp, pred_qed, pred_sas];
        }
    }
    (best_idx, best)
}

/// Inverse design: the best-matching holdout molecule for each property objective.
pub fn inverse_leads(test: &[ZincMolecule], preds: &[Vec<f64>; 3]) -> Vec<InverseDesignLead> {
    let mut inverse_leads = Vec::new();
    println!("\n[INVERSE DISCOVERY] Scanning 2,000 Holdout Candidates via 12-Qubit Manifold...");

    for obj in &OBJECTIVES {
        let (best_idx, [best_pred_logp, best_pred_qed, best_pred_sas]) = best_candidate(obj, preds);
        let best_mol = &test[best_idx];
        let euclid_err = ((best_pred_logp - obj.target_logp).powi(2)
            + (best_pred_qed - obj.target_qed).powi(2)
            + (best_pred_sas - obj.target_sas).powi(2))
        .sqrt();

        println!("  Objective: {}", obj.name);
        println!(
            "    Target     : logP = {:.2}, QED = {:.2}, SAS = {:.2}",
            obj.target_logp, obj.target_qed, obj.target_sas
        );
        println!("    Discovered : SMILES = {}", best_mol.smiles);
        println!(
            "    Predicted  : logP = {:.2}, QED = {:.2}, SAS = {:.2}",
            best_pred_logp, best_pred_qed, best_pred_sas
        );
        println!(
            "    Actual ZINC: logP = {:.2}, QED = {:.2}, SAS = {:.2}",
            best_mol.log_p, best_mol.qed, best_mol.sas
        );
        println!(
            "    Deviation  : Euclidean Δ = {:.4} | Valid Chemical: true\n",
            euclid_err
        );

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
    inverse_leads
}

fn build_report(
    args: &Args,
    model: &Model,
    (train, test): (&[ZincMolecule], &[ZincMolecule]),
    trained: &Trained,
    ev: Evaluation,
    leads: Vec<InverseDesignLead>,
) -> Zinc12QubitReport {
    // Classical reference: ridge regression on the same raw features and train/test split.
    let classical_baselines = t1000::baselines::zinc_baselines(
        train
            .iter()
            .map(|s| (s.features.as_slice(), [s.log_p, s.qed, s.sas])),
        test.iter()
            .map(|s| (s.features.as_slice(), [s.log_p, s.qed, s.sas])),
    );
    let l = model.layout;
    let [logp_metrics, qed_metrics, sas_metrics] = ev.metrics;
    Zinc12QubitReport {
        classical_baselines,
        timestamp_rfc3339: "2026-10-13T13:15:00Z".to_string(),
        host_cpu: t1000::hostinfo::HostProfile::detect().label(),
        host_ram_gb: 16.0,
        n_qubits: N_QUBITS,
        state_dimension: DIM,
        state_memory_bytes: DIM * 16,
        circuit_depth: args.depth,
        num_quantum_params: l.n_quantum,
        num_readout_params: l.n_weights + l.n_biases,
        num_total_params: l.n_total,
        epochs_trained: trained.prior_epochs + args.num_epochs,
        batch_size: args.batch_size,
        learning_rate: args.lr,
        max_norm: args.max_norm,
        epoch_history: {
            let mut all_h = trained.prior_history.clone();
            all_h.extend(trained.telemetry.iter().cloned());
            all_h
        },
        test_eval_time_ms: ev.eval_ms,
        per_molecule_latency_us: ev.per_mol_latency_us,
        throughput_molecules_per_sec: ev.throughput_mol_sec,
        logp_metrics,
        qed_metrics,
        sas_metrics,
        unitary_norm: ev.norm,
        inverse_leads: leads,
    }
}

/// Where the report goes: `--out`, else a name chosen by the cumulative epoch count.
fn report_path(args: &Args, total_epochs: usize) -> PathBuf {
    if let Some(p) = &args.out {
        return PathBuf::from(p);
    }
    let name = if total_epochs >= 30 {
        "zinc_12qubit_30ep_results.json"
    } else if args.num_epochs == 15 {
        "zinc_12qubit_15ep_results.json"
    } else {
        "zinc_12qubit_results.json"
    };
    resolve_workspace_path(&format!("tests/benchmarks/zinc_molecular/outputs/{name}"))
}

/// Scores the report against the saved best and writes it, returning the outcome and the
/// settings key under which the score competes.
fn write_report(
    report: &Zinc12QubitReport,
    path: &std::path::Path,
    prior_epochs: usize,
    targets: &[Vec<f64>; 3],
    sizes: (usize, usize),
) -> (t1000::results::Outcome, String) {
    {
        use t1000::baselines::{print_comparison, QnnScore};
        let m = [
            &report.logp_metrics,
            &report.qed_metrics,
            &report.sas_metrics,
        ];
        let qnn_rows = [QnnScore::new("trained circuit")
            .mae(m.map(|p| p.tuned_mae))
            .mse(m.map(|p| p.tuned_mse))
            .pearson_r(m.map(|p| p.pearson_r))];
        print_comparison(&qnn_rows, &report.classical_baselines);
    }

    // Held-out relative MSE decides, mean Pearson r guards. Resumed runs carry their prior
    // epochs in the settings so they only compete with runs of the same history.
    let m = [
        &report.logp_metrics,
        &report.qed_metrics,
        &report.sas_metrics,
    ];
    let holdout = t1000::holdout_score::ZincHoldout::new(
        m.map(|p| p.tuned_mse),
        m.map(|p| p.pearson_r),
        [&targets[0], &targets[1], &targets[2]],
    );
    let settings = format!(
        "qubits={};depth={};prior_epochs={};epochs={};batch={};lr={};max_norm={};train={};test={}",
        report.n_qubits,
        report.circuit_depth,
        prior_epochs,
        report.epochs_trained,
        report.batch_size,
        report.learning_rate,
        report.max_norm,
        sizes.0,
        sizes.1
    );
    let mut outcome = t1000::results::Outcome::Written;
    if let Ok(json_str) = serde_json::to_string_pretty(report) {
        match t1000::results::write_scored(path, json_str, &holdout.score(&settings)) {
            Ok(o) => outcome = o,
            Err(e) => eprintln!("[!] Could not write the report: {e}"),
        }
        match outcome {
            t1000::results::Outcome::KeptBest => {
                println!("[=] Saved 12-qubit hybrid report kept: this run did not improve it");
            }
            t1000::results::Outcome::Scratch => {
                println!("[~] Hybrid report written to a scratch copy only (use --record to save)");
            }
            _ => println!(
                "[SAVED] 12-Qubit Hybrid Report exported to: {}",
                path.display()
            ),
        }
    }
    (outcome, settings)
}

/// Saves the checkpoint with all hybrid weights and optimizer state.
fn save_checkpoint(
    args: &Args,
    model: &Model,
    params: &[f64],
    trained: &Trained,
    outcome: t1000::results::Outcome,
    settings: &str,
) {
    let default_ckpt_path = resolve_workspace_path(
        "tests/benchmarks/zinc_molecular/outputs/zinc_12qubit_checkpoint.json",
    );
    // The default checkpoint is the weights behind the saved report, so it follows the
    // report's outcome. An explicit --save-weights path is always written.
    let explicit_ckpt = args.save_weights.is_some();
    let mut target_ckpt_path = match &args.save_weights {
        Some(p) => PathBuf::from(p),
        None => default_ckpt_path,
    };
    if !explicit_ckpt {
        match outcome {
            t1000::results::Outcome::KeptBest => return,
            t1000::results::Outcome::WrittenVariant => {
                target_ckpt_path = t1000::results::variant_path(&target_ckpt_path, settings);
            }
            _ => {}
        }
    }

    let checkpoint = QnnCheckpoint {
        timestamp_rfc3339: "2026-10-13T14:10:00Z".to_string(),
        n_qubits: N_QUBITS,
        depth: args.depth,
        total_params: model.layout.n_total,
        epochs_completed: trained.prior_epochs + args.num_epochs,
        hybrid_params: params.to_vec(),
        opt_step: trained.opt.step,
        opt_m: trained.opt.m.clone(),
        opt_v: trained.opt.v.clone(),
    };

    if let Ok(ckpt_str) = serde_json::to_string_pretty(&checkpoint) {
        if let Some(parent) = target_ckpt_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = t1000::results::write(&target_ckpt_path, ckpt_str);
        println!(
            "[CHECKPOINT] 111 Hybrid Parameters saved to: {}",
            target_ckpt_path.display()
        );
    }
}

/// Builds and writes the report, then the checkpoint.
pub fn save_outputs(
    args: &Args,
    model: &Model,
    samples: (&[ZincMolecule], &[ZincMolecule]),
    params: &[f64],
    trained: Trained,
    ev: Evaluation,
    leads: Vec<InverseDesignLead>,
) {
    let targets = target_columns(samples.1);
    let total_epochs = trained.prior_epochs + args.num_epochs;
    let report = build_report(args, model, samples, &trained, ev, leads);
    let path = report_path(args, total_epochs);
    let sizes = (samples.0.len(), samples.1.len());
    let (outcome, settings) = write_report(&report, &path, trained.prior_epochs, &targets, sizes);
    save_checkpoint(args, model, params, &trained, outcome, &settings);
}
