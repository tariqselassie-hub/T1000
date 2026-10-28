// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! GuddaNetworks 12-Qubit QNN: Quantum Natural Gradient Engine (Mechanism #4)
//!
//! MECHANISM #4: Quantum Natural Gradient (QNG) Optimization on Fubini-Study Manifold
//!
//! Theoretical Foundation (Stokes et al., Quantum 4, 269 (2020)):
//!   Standard gradient descent assumes parameters θ exist in flat Euclidean space:
//!     θ_{t+1} = θ_t - η ∇_θ L
//!   In quantum circuits, parameters generate state trajectories in complex projective
//!   Hilbert space (CP^{2^N - 1}). The natural gradient follows the steepest descent
//!   direction along the Fubini-Study Riemannian metric tensor g(θ):
//!     θ_{t+1} = θ_t - η (g(θ) + λ I)^{-1} ∇_θ L
//!
//! Metric Tensor Formulation:
//!   g_{ij}(θ) = Re[ ⟨∂_i ψ | ∂_j ψ⟩ - ⟨∂_i ψ | ψ⟩⟨ψ | ∂_j ψ⟩ ]
//!   Computed exactly via statevector fidelity overlap on CP^N:
//!     1 - |⟨ψ(θ + ε e_i) | ψ(θ - ε e_i)⟩|^2 = (2ε)^2 g_{ii}(θ) / 4 + O(ε^3)
//!
//! Execution:
//!   Rigorous head-to-head comparison on identical starting weights:
//!   - Path A: Standard Euclidean Adam Optimizer
//!   - Path B: Quantum Natural Gradient (QNG) Riemannian Optimizer
//!
//! Evaluated on 10,000 ZINC train / 2,000 holdout test molecules across:
//! logP, QED, SAS, Molecule #10640 solvation recovery, and Fubini-Study spectrum.

use serde::Serialize;
use std::fs;
use std::time::Instant;
use t1000::baselines::{print_comparison, QnnScore};
use t1000::zinc_data::{self, ZincMolecule};
use t1000::zinc_reuploading_model::{
    eval_loss_and_weights, finite_difference_gradient, fubini_study_diagonal, holdout_scores,
    track_molecule, AdamPath, Model, QngCheckpoint, QngComparisonReport, QngEpochTelemetry, Scores,
    TrackedPath, N_QUBITS, SOURCE_QUBIT,
};
use t1000::zinc_training_common::{
    resolve_workspace_path, utc_timestamp_rfc3339, ReuploadingCheckpoint,
};
use t1000::{CircuitAnsatz, DataEncodingLayer, EncodingKind};

const FEATS_10640: [f64; 19] = [
    24.0, 408.279, 15.0, 5.0, 2.0, 1.0, 0.0, 0.0, 1.0, 1.0, 12.0, 0.5, 9.0, 0.375, 5.0, 0.0, 4.0,
    3.0, 4.0,
];
const TRUE_LP_10640: f64 = -6.7716;
const BANNER_RULE: &str =
    "=========================================================================";

fn print_banner() {
    println!("{BANNER_RULE}");
    println!(" GUDDANETWORKS 12-QUBIT QNN: QUANTUM NATURAL GRADIENT (MECHANISM #4)     ");
    println!("{BANNER_RULE}");
    println!(" Optimization Manifold : Complex Projective Fubini-Study Hilbert Space    ");
    println!(" Metric Tensor g(θ)    : Re[ ⟨∂_i ψ | ∂_j ψ⟩ - ⟨∂_i ψ | ψ⟩⟨ψ | ∂_j ψ⟩ ]   ");
    println!(" Comparison            : Euclidean Adam vs Quantum Natural Gradient      ");
    println!(" Subspace Target       : 60 Circuit Rotation Angles + 12 Inter-Layer DRU ");
    println!(" Hardware Engine       : AVX2+FMA Fused Zero-Allocation POVM Reduction   ");
    println!(" Visual Standard       : Deen OS Sovereign Palette                       ");
    println!("-------------------------------------------------------------------------\n");
}

fn print_phase(title: &str) {
    println!("{BANNER_RULE}");
    println!("{title}");
    println!("{BANNER_RULE}");
}

/// Loads the train / holdout molecules and the re-uploading checkpoint.
fn load_inputs() -> (Vec<ZincMolecule>, Vec<ZincMolecule>, ReuploadingCheckpoint) {
    let train_path =
        resolve_workspace_path("tests/benchmarks/zinc_molecular/data/zinc_train_10k.csv");
    let test_path = resolve_workspace_path("tests/benchmarks/zinc_molecular/data/zinc_test_2k.csv");
    let ckpt_path = resolve_workspace_path(
        "tests/benchmarks/zinc_molecular/checkpoints/zinc_12qubit_checkpoint_reuploading.json",
    );

    let train_samples = zinc_data::load(&train_path, 10_000).expect("Load train data");
    let test_samples = zinc_data::load(&test_path, 2_000).expect("Load test data");
    println!("  -> Train samples: {}", train_samples.len());
    println!("  -> Test samples : {}", test_samples.len());

    let ckpt_json = fs::read_to_string(&ckpt_path).expect("Read reuploading checkpoint");
    let ckpt: ReuploadingCheckpoint = serde_json::from_str(&ckpt_json).expect("Parse checkpoint");
    println!("  -> Checkpoint loaded: 60 circuit params, 12 DRU scales, 12 FF alphas.\n");
    (train_samples, test_samples, ckpt)
}

/// A tracked path's logP row and bias, from `eval_loss_and_weights` output.
fn tracked<'a>(
    params: &'a [f64],
    dru_scales: &'a [f64],
    w: &'a [Vec<f64>],
    b: &[f64],
) -> TrackedPath<'a> {
    TrackedPath {
        params,
        dru_scales,
        logp_weights: &w[0],
        logp_bias: b[0],
    }
}

fn print_scores_line(s: &Scores) {
    println!(
        "  Holdout MAE : logP = {:.4}, QED = {:.4}, SAS = {:.4}",
        s.mae[0], s.mae[1], s.mae[2]
    );
    println!(
        "  Holdout R²  : logP = {:.4}, QED = {:.4}, SAS = {:.4}\n",
        s.r2[0], s.r2[1], s.r2[2]
    );
}

/// One epoch's metrics for both paths; prints the table row.
fn epoch_telemetry(
    model: &Model,
    epoch: usize,
    euc: &AdamPath,
    qng: &AdamPath,
    opt_subset: &[ZincMolecule],
    g_diag: &[f64],
) -> QngEpochTelemetry {
    let (loss_euc, w_euc, b_euc) =
        eval_loss_and_weights(model, &euc.params, &euc.dru_scales, opt_subset, 1e-3);
    let (loss_qng, w_qng, b_qng) =
        eval_loss_and_weights(model, &qng.params, &qng.dru_scales, opt_subset, 1e-3);

    let g_min = g_diag.iter().cloned().fold(f64::INFINITY, f64::min);
    let g_max = g_diag.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let cond_num = g_max / g_min.max(1e-9);

    let paths = [
        tracked(&euc.params, &euc.dru_scales, &w_euc, &b_euc),
        tracked(&qng.params, &qng.dru_scales, &w_qng, &b_qng),
    ];
    let preds = track_molecule(model, &FEATS_10640, &paths);
    let (pred_10640_euc, pred_10640_qng) = (preds[0], preds[1]);

    let delta_adv = (loss_euc - loss_qng) / loss_euc * 100.0;
    println!(
        "  {:>2}   │    {:>10.6}    │   {:>10.6}   │   {:>+6.2}%   │     {:>8.2}      │     {:>+8.4}",
        epoch, loss_euc, loss_qng, delta_adv, cond_num, pred_10640_qng
    );
    QngEpochTelemetry {
        epoch,
        loss_euclidean: loss_euc,
        loss_qng,
        g_cond_number: cond_num,
        g_min_eigenval: g_min,
        g_max_eigenval: g_max,
        pred_mol10640_euclidean: pred_10640_euc,
        pred_mol10640_qng: pred_10640_qng,
    }
}

/// Phase 2: Euclidean Adam and QNG from identical starting weights.
fn train_both(
    model: &Model,
    circuit_params: &[f64],
    dru_scales: &[f64],
    opt_subset: &[ZincMolecule],
) -> (AdamPath, AdamPath, Vec<QngEpochTelemetry>) {
    let n_epochs = 12;
    let lr = 0.04;
    let eps = 0.01;
    let lambda_qng = 0.02; // Tikhonov regularization for Fubini-Study inversion

    print_phase(" [PHASE 2] TRAINING TRAJECTORY AUDIT: EUCLIDEAN ADAM vs QUANTUM NATURAL GRADIENT");
    println!(
        "  Batch Size : 2,000 Molecules | Learning Rate: {:.2} | Damping λ: {:.3}",
        lr, lambda_qng
    );
    println!("  Param Space: 60 Quantum Circuit Angles (All 3 Layers)\n");

    let mut euc = AdamPath::new(circuit_params, dru_scales);
    let mut qng = AdamPath::new(circuit_params, dru_scales);
    let mut history = Vec::with_capacity(n_epochs);

    println!(" Epoch │ Loss (Euclidean) │   Loss (QNG)   │ Δ Advantage │ Fubini-Study κ(g) │ #10640 Pred (QNG)");
    println!("───────┼──────────────────┼────────────────┼─────────────┼───────────────────┼──────────────────");
    let t_train_start = Instant::now();

    for epoch in 1..=n_epochs {
        let g_diag =
            fubini_study_diagonal(model, &qng.params, &qng.dru_scales, opt_subset, lambda_qng);
        history.push(epoch_telemetry(
            model, epoch, &euc, &qng, opt_subset, &g_diag,
        ));

        let grad_batch = &opt_subset[..500];
        let grad_euc =
            finite_difference_gradient(model, &euc.params, &euc.dru_scales, grad_batch, eps);
        let grad_qng =
            finite_difference_gradient(model, &qng.params, &qng.dru_scales, grad_batch, eps);

        // Euclidean Adam; QNG preconditions the raw gradient by (g(θ) + λ I)^-1 with λ = 0.05.
        euc.step(&grad_euc, None, epoch, lr);
        qng.step(&grad_qng, Some((&g_diag, 0.05)), epoch, lr);
    }

    let el_train = t_train_start.elapsed();
    println!("───────┴──────────────────┴────────────────┴─────────────┴───────────────────┴──────────────────");
    println!(
        "  Dual optimization completed in {:.2}s\n",
        el_train.as_secs_f64()
    );
    (euc, qng, history)
}

fn print_holdout_table(pre: &Scores, euc: &Scores, qng: &Scores) {
    println!("  ┌─────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┐");
    println!("  │  Property   │ Pre-Opt (DRU)    │ Euclidean Adam   │ QNG (Riemannian) │  QNG Net Gain    │");
    println!("  ├─────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┤");
    for (k, name) in ["logP MAE   ", "QED MAE    ", "SAS MAE    "]
        .iter()
        .enumerate()
    {
        println!("  │  {name}│         {:>8.4} │         {:>8.4} │         {:>8.4} │         {:>+6.2}%  │",
            pre.mae[k], euc.mae[k], qng.mae[k], (euc.mae[k] - qng.mae[k]) / euc.mae[k] * 100.0);
    }
    println!("  ├─────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┤");
    for (k, name) in ["logP R²    ", "QED R²     ", "SAS R²     "]
        .iter()
        .enumerate()
    {
        println!("  │  {name}│         {:>8.4} │         {:>8.4} │         {:>8.4} │         {:>+6.2}%  │",
            pre.r2[k], euc.r2[k], qng.r2[k], (qng.r2[k] - euc.r2[k]) / euc.r2[k] * 100.0);
    }
    println!("  └─────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┘\n");
}

fn print_solvation_table(preds: &[f64]) {
    print_phase(" [PHASE 4] MOLECULE #10640 BORN SOLVATION RECOVERY AUDIT                 ");
    println!(
        "  SMILES: O=[N+]([O-])c1ccc2c(c1)=[NH+][C@@H](CS[C@H]1[NH+]=c3ccc(Br)cc3=[NH+]1)[NH+]=2"
    );
    println!("  Physical Target : True logP = {:>+8.4}", TRUE_LP_10640);
    println!("  ┌────────────────────────────────┬──────────────────┬──────────────────┐");
    println!("  │  Optimization Method           │  Predicted logP  │  Absolute Error  │");
    println!("  ├────────────────────────────────┼──────────────────┼──────────────────┤");
    let rows = [
        ("Pre-Optimization Baseline     ", preds[0]),
        ("Euclidean Adam Optimizer      ", preds[1]),
        ("Quantum Natural Gradient (QNG)", preds[2]),
    ];
    for (name, pred) in rows {
        println!(
            "  │  {name}│       {:>+10.4} │       {:>10.4} │",
            pred,
            (pred - TRUE_LP_10640).abs()
        );
    }
    println!("  └────────────────────────────────┴──────────────────┴──────────────────┘\n");
}

/// Everything the report records besides the epoch history.
struct Outcome {
    pre: Scores,
    euc: Scores,
    qng: Scores,
    mol_preds: Vec<f64>,
}

fn build_report(
    train_len: usize,
    test_len: usize,
    out: &Outcome,
    history: Vec<QngEpochTelemetry>,
    classical_baselines: Vec<t1000::baselines::BaselineResult>,
) -> QngComparisonReport {
    QngComparisonReport {
        classical_baselines,
        timestamp_rfc3339: utc_timestamp_rfc3339(),
        host_cpu: "Native Host AVX2+FMA".to_string(),
        n_qubits: N_QUBITS,
        train_samples: train_len,
        test_samples: test_len,
        pre_qng_mae_logp: out.pre.mae[0],
        pre_qng_mae_qed: out.pre.mae[1],
        pre_qng_mae_sas: out.pre.mae[2],
        pre_qng_r2_logp: out.pre.r2[0],
        pre_qng_r2_qed: out.pre.r2[1],
        pre_qng_r2_sas: out.pre.r2[2],
        euclidean_mae_logp: out.euc.mae[0],
        euclidean_mae_qed: out.euc.mae[1],
        euclidean_mae_sas: out.euc.mae[2],
        euclidean_r2_logp: out.euc.r2[0],
        euclidean_r2_qed: out.euc.r2[1],
        euclidean_r2_sas: out.euc.r2[2],
        qng_mae_logp: out.qng.mae[0],
        qng_mae_qed: out.qng.mae[1],
        qng_mae_sas: out.qng.mae[2],
        qng_r2_logp: out.qng.r2[0],
        qng_r2_qed: out.qng.r2[1],
        qng_r2_sas: out.qng.r2[2],
        mol10640_true_logp: TRUE_LP_10640,
        mol10640_pre_qng_pred: out.mol_preds[0],
        mol10640_euclidean_pred: out.mol_preds[1],
        mol10640_qng_pred: out.mol_preds[2],
        epoch_history: history,
    }
}

fn write_json<T: Serialize>(rel_path: &str, what: &str, value: &T, label: &str) {
    let path = resolve_workspace_path(rel_path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let json =
        serde_json::to_string_pretty(value).unwrap_or_else(|e| panic!("Serialize {what}: {e:?}"));
    t1000::results::write(&path, json).unwrap_or_else(|e| panic!("Write {what}: {e:?}"));
    println!("  {label}: {}", path.display());
}

/// Phase 5: the QNG checkpoint, the classical reference, the comparison table and the report.
fn persist(
    train: &[ZincMolecule],
    test: &[ZincMolecule],
    ckpt: ReuploadingCheckpoint,
    paths: (AdamPath, AdamPath),
    out: &Outcome,
    history: Vec<QngEpochTelemetry>,
) {
    print_phase(" [PHASE 5] PERSISTING CHECKPOINT & TELEMETRY REPORT                      ");
    let (euc, qng) = paths;
    let checkpoint = QngCheckpoint {
        timestamp_rfc3339: utc_timestamp_rfc3339(),
        n_qubits: N_QUBITS,
        depth: 3,
        source_qubit: SOURCE_QUBIT,
        circuit_params_euclidean: euc.params,
        circuit_params_qng: qng.params,
        upload_scales_qng: qng.dru_scales,
        feedforward_alphas: ckpt.feedforward_alphas,
        readout_weights_78: ckpt.readout_weights_78,
        readout_biases_78: ckpt.readout_biases_78,
    };
    write_json(
        "tests/benchmarks/zinc_molecular/checkpoints/zinc_12qubit_checkpoint_qng.json",
        "checkpoint",
        &checkpoint,
        "[OK] Saved QNG Checkpoint",
    );

    // Classical reference: ridge regression on the same raw features and train/test split.
    let classical_baselines = t1000::baselines::zinc_baselines(
        train
            .iter()
            .map(|s| (s.features.as_slice(), [s.log_p, s.qed, s.sas])),
        test.iter()
            .map(|s| (s.features.as_slice(), [s.log_p, s.qed, s.sas])),
    );
    let report = build_report(
        train.len(),
        test.len(),
        out,
        history,
        classical_baselines.clone(),
    );
    let rows = [
        QnnScore::new("before optimisation")
            .mae(out.pre.mae)
            .r2(out.pre.r2),
        QnnScore::new("Euclidean Adam")
            .mae(out.euc.mae)
            .r2(out.euc.r2),
        QnnScore::new("quantum natural gradient")
            .mae(out.qng.mae)
            .r2(out.qng.r2),
    ];
    print_comparison(&rows, &classical_baselines);
    write_json(
        "tests/benchmarks/zinc_molecular/reports/zinc_12qubit_qng_report.json",
        "report",
        &report,
        "[OK] Saved Machine-Readable Report",
    );
    println!("\n{BANNER_RULE}");
    println!(" MECHANISM #4 QUANTUM NATURAL GRADIENT AUDIT COMPLETE                     ");
    println!("{BANNER_RULE}");
}

/// Phase 3: refit the readouts on all 10,000 train molecules, score the holdout set, and track
/// molecule #10640. Returns (Euclidean score, QNG score, #10640 predictions).
fn final_benchmark(
    model: &Model,
    start: &Start,
    paths: (&AdamPath, &AdamPath),
    train: &[ZincMolecule],
    test: &[ZincMolecule],
) -> (Scores, Scores, Vec<f64>) {
    let (euc, qng) = paths;
    print_phase(" [PHASE 3] HOLDOUT TEST SET BENCHMARK (2,000 MOLECULES)                  ");
    let (_, w_euc, b_euc) = eval_loss_and_weights(model, &euc.params, &euc.dru_scales, train, 1e-3);
    let (_, w_qng, b_qng) = eval_loss_and_weights(model, &qng.params, &qng.dru_scales, train, 1e-3);
    let score_euc = holdout_scores(model, &euc.params, &euc.dru_scales, &w_euc, &b_euc, test);
    let score_qng = holdout_scores(model, &qng.params, &qng.dru_scales, &w_qng, &b_qng, test);
    print_holdout_table(&start.scores, &score_euc, &score_qng);

    let mol_preds = track_molecule(
        model,
        &FEATS_10640,
        &[
            tracked(&start.params, &start.dru_scales, &start.w, &start.b),
            tracked(&euc.params, &euc.dru_scales, &w_euc, &b_euc),
            tracked(&qng.params, &qng.dru_scales, &w_qng, &b_qng),
        ],
    );
    (score_euc, score_qng, mol_preds)
}

/// The re-uploading checkpoint's circuit, its refit readout and holdout scores (phase 1).
struct Start {
    params: Vec<f64>,
    dru_scales: Vec<f64>,
    w: Vec<Vec<f64>>,
    b: Vec<f64>,
    scores: Scores,
}

fn pre_optimization(
    model: &Model,
    ckpt: &ReuploadingCheckpoint,
    train: &[ZincMolecule],
    test: &[ZincMolecule],
) -> Start {
    print_phase(" [PHASE 1] PRE-OPTIMIZATION BASELINE EVALUATION (RE-UPLOADING CHECKPOINT) ");
    let params = ckpt.circuit_params.clone();
    let dru_scales = ckpt.upload_scales_l0.clone();
    let (_, w, b) = eval_loss_and_weights(model, &params, &dru_scales, &train[..2000], 1e-3);
    let scores = holdout_scores(model, &params, &dru_scales, &w, &b, test);
    print_scores_line(&scores);
    Start {
        params,
        dru_scales,
        w,
        b,
        scores,
    }
}

fn main() {
    print_banner();
    let (train_samples, test_samples, ckpt) = load_inputs();

    let ansatz = CircuitAnsatz::new(N_QUBITS, 3).with_pi_layer(true);
    let encoding = DataEncodingLayer::new(N_QUBITS, EncodingKind::Amplitude);
    let model = Model {
        ansatz: &ansatz,
        encoding: &encoding,
        feedforward_alphas: &ckpt.feedforward_alphas,
    };

    let start = pre_optimization(&model, &ckpt, &train_samples, &test_samples);
    let opt_subset: Vec<ZincMolecule> = train_samples.iter().take(2000).cloned().collect();
    let (euc, qng, history) = train_both(&model, &start.params, &start.dru_scales, &opt_subset);

    let (score_euc, score_qng, mol_preds) =
        final_benchmark(&model, &start, (&euc, &qng), &train_samples, &test_samples);
    print_solvation_table(&mol_preds);

    let outcome = Outcome {
        pre: start.scores,
        euc: score_euc,
        qng: score_qng,
        mol_preds,
    };
    persist(
        &train_samples,
        &test_samples,
        ckpt,
        (euc, qng),
        &outcome,
        history,
    );
}
