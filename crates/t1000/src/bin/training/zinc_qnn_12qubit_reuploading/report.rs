// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

use super::phases::{Angles, Audit, Throughput};
use super::*;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use t1000::zinc_stage_common::{
    improvement, phase_header, print_angle_grid, Fit, PROPS, SMILES_10640, TRUE_LP_10640,
};
use t1000::zinc_training_common::{utc_timestamp_rfc3339, ReuploadingCheckpoint};

const RULE: &str = "=========================================================================";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReuploadingReport {
    pub timestamp_rfc3339: String,
    pub host_cpu: String,
    pub n_qubits: usize,
    pub source_qubit: usize,
    pub train_samples: usize,
    pub test_samples: usize,
    pub feedforward_alphas: Vec<f64>,
    pub upload_scales_l0: Vec<f64>,
    pub baseline_mae_logp: f64,
    pub baseline_mae_qed: f64,
    pub baseline_mae_sas: f64,
    pub baseline_r2_logp: f64,
    pub baseline_r2_qed: f64,
    pub baseline_r2_sas: f64,
    pub feedforward_mae_logp: f64,
    pub feedforward_mae_qed: f64,
    pub feedforward_mae_sas: f64,
    pub feedforward_r2_logp: f64,
    pub feedforward_r2_qed: f64,
    pub feedforward_r2_sas: f64,
    pub reuploading_mae_logp: f64,
    pub reuploading_mae_qed: f64,
    pub reuploading_mae_sas: f64,
    pub reuploading_r2_logp: f64,
    pub reuploading_r2_qed: f64,
    pub reuploading_r2_sas: f64,
    pub unified_mae_logp: f64,
    pub unified_mae_qed: f64,
    pub unified_mae_sas: f64,
    pub unified_r2_logp: f64,
    pub unified_r2_qed: f64,
    pub unified_r2_sas: f64,
    pub mol10640_true_logp: f64,
    pub mol10640_baseline_pred_logp: f64,
    pub mol10640_feedforward_pred_logp: f64,
    pub mol10640_reuploading_pred_logp: f64,
    pub mol10640_unified_pred_logp: f64,
    pub baseline_throughput_mols_per_sec: f64,
    pub reuploading_throughput_mols_per_sec: f64,
    pub unified_throughput_mols_per_sec: f64,
    #[serde(default)]
    pub classical_baselines: Vec<t1000::baselines::BaselineResult>,
}

pub fn print_banner() {
    phase_header(" GUDDANETWORKS 12-QUBIT QNN: DATA RE-UPLOADING ENGINE (MECHANISM #3)    ");
    println!(" Mode               : Universal Multi-Frequency Harmonic Expansion");
    println!(" Inter-Layer Upload : Stage 1 DRU between Layer 0 and Layer 1 CZ Block");
    println!(" Re-upload Kernel   : R_y(s_q · x̃_q) per qubit q ∈ 0..11");
    println!(
        " Feed-Forward Qubit : Qubit {} (⟨Z_{}⟩ Dynamic Terminal Modulation)",
        SOURCE_QUBIT, SOURCE_QUBIT
    );
    println!(
        " Hilbert Space Dim  : {} (12 Qubits, 64 KB L2 Cache-Resident)",
        DIM
    );
    println!(" Hardware Target    : Bare-Metal AVX2 + FMA Single-Pass Fused POVM");
    println!(" Palette            : Deen OS Sovereign (Quantum Green / Hardware Gold / Alert Red)");
    println!("-------------------------------------------------------------------------\n");
}

pub fn print_baselines(base: &Fit, ff: &Fit) {
    println!(
        "  Baseline MAE: logP = {:.4}, QED = {:.4}, SAS = {:.4} | logP R² = {:.4}",
        base.mae[0], base.mae[1], base.mae[2], base.r2[0]
    );
    println!(
        "  Mech #2  MAE: logP = {:.4}, QED = {:.4}, SAS = {:.4} | logP R² = {:.4}",
        ff.mae[0], ff.mae[1], ff.mae[2], ff.r2[0]
    );
}

pub fn print_phase2_intro() {
    phase_header(" [PHASE 2] TRAINING DATA RE-UPLOADING SCALING COEFFICIENTS (s ∈ R¹²)    ");
    println!(" Optimizing 12 inter-layer descriptor re-upload scales via Adam...");
    println!(" Re-uploading kernel: R_y(s_q · x̃_q) before Layer 1 CZ entanglement ring\n");
}

pub fn print_scale_table_header() {
    println!(" Epoch │ Composite Loss │  logP Pred (#10640) │  ||s||_2  │ Top Scales (q: s_q)");
    println!("───────┼────────────────┼─────────────────────┼───────────┼────────────────────────────────────");
}

pub fn print_scale_footer(elapsed: Duration, scales: &[f64]) {
    println!("───────┴────────────────┴─────────────────────┴───────────┴────────────────────────────────────");
    println!(
        "  DRU Optimization completed in {:.2}s\n",
        elapsed.as_secs_f64()
    );

    println!(" Final Data Re-Uploading Scales (s_q for q ∈ 0..11):");
    print_angle_grid(scales);
}

pub fn print_reuploading(base: &Fit, dru: &Fit) {
    let imp = improvement(&base.mae, &dru.mae);
    println!("  Holdout Comparison (12 Observables):");
    println!("  ┌─────────────┬──────────────────┬──────────────────┬──────────────────┐");
    println!("  │  Property   │  Baseline MAE    │  Re-Upload MAE   │  Improvement     │");
    println!("  ├─────────────┼──────────────────┼──────────────────┼──────────────────┤");
    for (k, name) in PROPS.iter().enumerate() {
        println!(
            "  │  {name:<11}│ {:>16.6} │ {:>16.6} │ {:>+14.2}%  │",
            base.mae[k], dru.mae[k], imp[k]
        );
    }
    println!("  └─────────────┴──────────────────┴──────────────────┴──────────────────┘\n");
    println!("  ┌─────────────┬──────────────────┬──────────────────┐");
    println!("  │  Property   │  Baseline R²     │  Re-Upload R²    │");
    println!("  ├─────────────┼──────────────────┼──────────────────┤");
    for (k, name) in PROPS.iter().enumerate() {
        println!(
            "  │  {name:<11}│ {:>16.6} │ {:>16.6} │",
            base.r2[k], dru.r2[k]
        );
    }
    println!("  └─────────────┴──────────────────┴──────────────────┘\n");
}

pub fn print_phase4_intro() {
    phase_header(" [PHASE 4] UNIFIED FRONTIER STACK (78 Observables: ZZ + FF + DRU)       ");
    println!(" Full integration of all 3 frontier mechanisms into a single quantum pass:\n");
}

pub fn print_ledger([base, ff, dru, uni]: [&Fit; 4]) {
    let imp = improvement(&base.mae, &uni.mae);
    println!("  Quad-Stage Architectural Evolution Ledger:");
    println!("  ┌─────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┐");
    println!("  │  Property   │  Baseline (12)   │  Mech #2 (FF)    │  Mech #3 (DRU)   │  Unified (78 Obs)│  Net Gain (%)    │");
    println!("  ├─────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┤");
    for (k, name) in PROPS.iter().enumerate() {
        println!(
            "  │  {:<11}│ {:>16.6} │ {:>16.6} │ {:>16.6} │ {:>16.6} │ {:>+14.2}%  │",
            format!("{name} MAE"),
            base.mae[k],
            ff.mae[k],
            dru.mae[k],
            uni.mae[k],
            imp[k]
        );
    }
    println!("  ├─────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┤");
    for (k, name) in PROPS.iter().enumerate() {
        println!(
            "  │  {:<11}│ {:>16.6} │ {:>16.6} │ {:>16.6} │ {:>16.6} │ {:>+14.2}%  │",
            format!("{name} R²"),
            base.r2[k],
            ff.r2[k],
            dru.r2[k],
            uni.r2[k],
            (uni.r2[k] - base.r2[k]) / base.r2[k].abs() * 100.0
        );
    }
    println!("  └─────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┘\n");
}

pub fn print_audit(a: &Audit) {
    let e_b = (a.base - TRUE_LP_10640).abs();
    let e_ff = (a.ff - TRUE_LP_10640).abs();
    let e_dru = (a.dru - TRUE_LP_10640).abs();
    let e_uni = (a.uni - TRUE_LP_10640).abs();

    println!("  SMILES: {SMILES_10640}");
    println!("  Physical Target : True logP = {:>+10.4}\n", TRUE_LP_10640);
    println!("  ┌────────────────────────────────┬──────────────────┬──────────────────┐");
    println!("  │  Architectural Stage           │  Predicted logP  │  Absolute Error  │");
    println!("  ├────────────────────────────────┼──────────────────┼──────────────────┤");
    let rows = [
        ("Baseline 12-Qubit QNN        ", a.base, e_b),
        ("+ Mech #2 (Feed-Forward)     ", a.ff, e_ff),
        ("+ Mech #3 (Data Re-Uploading)", a.dru, e_dru),
        ("+ Unified (M1 + M2 + M3: 78) ", a.uni, e_uni),
    ];
    for (label, pred, err) in rows {
        println!("  │  {label} │ {:>16.4} │ {:>16.4} │", pred, err);
    }
    println!("  └────────────────────────────────┴──────────────────┴──────────────────┘\n");
    println!(
        "  Born Solvation Error Reduction: {:.4} -> {:.4} (Gain = {:>+6.2}%)\n",
        e_b,
        e_uni,
        (1.0 - e_uni / e_b) * 100.0
    );
}

/// Phase 6: the holdout molecules whose logP error the unified stack reduces most.
pub fn print_corrections(test: &[ZincMolecule], base: &Fit, uni: &Fit) {
    phase_header(" [PHASE 6] TOP OUTLIER CORRECTIONS (Baseline vs Unified Stack)           ");

    let mut corrections: Vec<(usize, f64, f64, f64, &str)> = Vec::new();
    for (i, sample) in test.iter().enumerate() {
        let err_b = (base.preds[i][0] - sample.log_p).abs();
        let err_u = (uni.preds[i][0] - sample.log_p).abs();
        let correction = err_b - err_u;
        corrections.push((i, correction, err_b, err_u, &sample.smiles));
    }
    corrections.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!(" Top 10 logP Error Reductions across Holdout Set:");
    println!(
        " ┌────────┬────────────────┬────────────────┬────────────────┬──────────────────────────┐"
    );
    println!(
        " │  Rank  │  Baseline |ε|  │  Unified |ε|   │  Δ Correction  │  SMILES (truncated)      │"
    );
    println!(
        " ├────────┼────────────────┼────────────────┼────────────────┼──────────────────────────┤"
    );
    for (rank, &(_idx, correction, err_b, err_u, smiles)) in corrections.iter().take(10).enumerate()
    {
        let smi_short = if smiles.len() > 24 {
            &smiles[..24]
        } else {
            smiles
        };
        println!(
            " │  {:>4}  │ {:>14.4} │ {:>14.4} │ {:>+14.4} │  {:<24} │",
            rank + 1,
            err_b,
            err_u,
            correction,
            smi_short
        );
    }
    println!(" └────────┴────────────────┴────────────────┴────────────────┴──────────────────────────┘\n");
}

pub fn print_throughput(tp: &Throughput) {
    println!(
        "  Baseline Readout (12 obs) : {:>7.0} mols/sec",
        tp.baseline
    );
    println!(
        "  Re-Uploading     (12 obs) : {:>7.0} mols/sec (Overhead: {:>+.2}%)",
        tp.reupload,
        (1.0 - tp.reupload / tp.baseline) * 100.0
    );
    println!(
        "  Unified Readout  (78 obs) : {:>7.0} mols/sec\n",
        tp.unified
    );
}

fn save_checkpoint(circuit: &Circuit, (scales, alphas): Angles, dru: &Fit, uni: &Fit) {
    let checkpoint = ReuploadingCheckpoint {
        timestamp_rfc3339: utc_timestamp_rfc3339(),
        n_qubits: N_QUBITS,
        depth: 3,
        source_qubit: SOURCE_QUBIT,
        circuit_params: circuit.params.clone(),
        feedforward_alphas: alphas.to_vec(),
        upload_scales_l0: scales.to_vec(),
        readout_weights_12: dru.w.clone(),
        readout_biases_12: dru.b.clone(),
        readout_weights_78: uni.w.clone(),
        readout_biases_78: uni.b.clone(),
    };

    let path = resolve_workspace_path(
        "tests/benchmarks/zinc_molecular/checkpoints/zinc_12qubit_checkpoint_reuploading.json",
    );
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let ckpt_json = serde_json::to_string_pretty(&checkpoint).expect("Serialize checkpoint");
    t1000::results::write(&path, ckpt_json).expect("Write checkpoint");
    println!("  [ok] Saved Re-Uploading Checkpoint: {}", path.display());
}

fn build_report(
    data: &Data,
    (scales, alphas): Angles,
    [b, f, d, u]: [&Fit; 4],
    audit: &Audit,
    tp: &Throughput,
    classical_baselines: Vec<t1000::baselines::BaselineResult>,
) -> ReuploadingReport {
    ReuploadingReport {
        classical_baselines,
        timestamp_rfc3339: utc_timestamp_rfc3339(),
        host_cpu: "Native Host AVX2+FMA".to_string(),
        n_qubits: N_QUBITS,
        source_qubit: SOURCE_QUBIT,
        train_samples: data.train.len(),
        test_samples: data.test.len(),
        feedforward_alphas: alphas.to_vec(),
        upload_scales_l0: scales.to_vec(),
        baseline_mae_logp: b.mae[0],
        baseline_mae_qed: b.mae[1],
        baseline_mae_sas: b.mae[2],
        baseline_r2_logp: b.r2[0],
        baseline_r2_qed: b.r2[1],
        baseline_r2_sas: b.r2[2],
        feedforward_mae_logp: f.mae[0],
        feedforward_mae_qed: f.mae[1],
        feedforward_mae_sas: f.mae[2],
        feedforward_r2_logp: f.r2[0],
        feedforward_r2_qed: f.r2[1],
        feedforward_r2_sas: f.r2[2],
        reuploading_mae_logp: d.mae[0],
        reuploading_mae_qed: d.mae[1],
        reuploading_mae_sas: d.mae[2],
        reuploading_r2_logp: d.r2[0],
        reuploading_r2_qed: d.r2[1],
        reuploading_r2_sas: d.r2[2],
        unified_mae_logp: u.mae[0],
        unified_mae_qed: u.mae[1],
        unified_mae_sas: u.mae[2],
        unified_r2_logp: u.r2[0],
        unified_r2_qed: u.r2[1],
        unified_r2_sas: u.r2[2],
        mol10640_true_logp: TRUE_LP_10640,
        mol10640_baseline_pred_logp: audit.base,
        mol10640_feedforward_pred_logp: audit.ff,
        mol10640_reuploading_pred_logp: audit.dru,
        mol10640_unified_pred_logp: audit.uni,
        baseline_throughput_mols_per_sec: tp.baseline,
        reuploading_throughput_mols_per_sec: tp.reupload,
        unified_throughput_mols_per_sec: tp.unified,
    }
}

fn print_classical_comparison(report: &ReuploadingReport) {
    use t1000::baselines::{print_comparison, QnnScore};
    let qnn_rows = [
        QnnScore::new("circuit, no mechanisms")
            .mae([
                report.baseline_mae_logp,
                report.baseline_mae_qed,
                report.baseline_mae_sas,
            ])
            .r2([
                report.baseline_r2_logp,
                report.baseline_r2_qed,
                report.baseline_r2_sas,
            ]),
        QnnScore::new("mid-circuit feed-forward")
            .mae([
                report.feedforward_mae_logp,
                report.feedforward_mae_qed,
                report.feedforward_mae_sas,
            ])
            .r2([
                report.feedforward_r2_logp,
                report.feedforward_r2_qed,
                report.feedforward_r2_sas,
            ]),
        QnnScore::new("data re-uploading")
            .mae([
                report.reuploading_mae_logp,
                report.reuploading_mae_qed,
                report.reuploading_mae_sas,
            ])
            .r2([
                report.reuploading_r2_logp,
                report.reuploading_r2_qed,
                report.reuploading_r2_sas,
            ]),
        QnnScore::new("re-uploading + ZZ readout")
            .mae([
                report.unified_mae_logp,
                report.unified_mae_qed,
                report.unified_mae_sas,
            ])
            .r2([
                report.unified_r2_logp,
                report.unified_r2_qed,
                report.unified_r2_sas,
            ]),
    ];
    print_comparison(&qnn_rows, &report.classical_baselines);
}

/// Phase 8: write the checkpoint and the machine-readable report.
pub fn persist(
    data: &Data,
    circuit: &Circuit,
    angles: Angles,
    stages: [&Fit; 4],
    audit: &Audit,
    tp: &Throughput,
) {
    phase_header(" [PHASE 8] PERSISTING CHECKPOINT & TELEMETRY REPORT                      ");
    save_checkpoint(circuit, angles, stages[2], stages[3]);

    // Classical reference: ridge regression on the same raw features and train/test split.
    let classical_baselines = t1000::baselines::zinc_baselines(
        data.train
            .iter()
            .map(|s| (s.features.as_slice(), [s.log_p, s.qed, s.sas])),
        data.test
            .iter()
            .map(|s| (s.features.as_slice(), [s.log_p, s.qed, s.sas])),
    );
    let report = build_report(data, angles, stages, audit, tp, classical_baselines);

    let report_save_path = resolve_workspace_path(
        "tests/benchmarks/zinc_molecular/reports/zinc_12qubit_reuploading_report.json",
    );
    if let Some(parent) = report_save_path.parent() {
        fs::create_dir_all(parent).ok();
    }
    print_classical_comparison(&report);

    let report_json = serde_json::to_string_pretty(&report).expect("Serialize report");
    t1000::results::write(&report_save_path, report_json).expect("Write report");
    println!(
        "  [ok] Saved Machine-Readable Report: {}",
        report_save_path.display()
    );
    println!("\n{RULE}");
    println!(" MECHANISM #3 DATA RE-UPLOADING VERIFICATION COMPLETE                     ");
    println!("{RULE}");
}
