// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Result tables, the report JSON and the enhanced checkpoint for `zinc_qnn_12qubit_zz_correlators`.

use super::{
    pair_index, print_phase, Head, ZZCorrelatorReport, N_OBS, N_PAIRS, N_PROPS, N_QUBITS, RULE,
};
use std::fs;
use std::path::PathBuf;
use t1000::baselines::{print_comparison, QnnScore};
use t1000::zinc_data::ZincMolecule;
use t1000::zinc_training_common::{resolve_workspace_path, utc_timestamp_rfc3339};

pub fn print_results_table(title: &str, mae: &[f64; 3], r2: &[f64; 3]) {
    println!("  [{title}]");
    println!("  ┌─────────────┬──────────────┬──────────────┐");
    println!("  │  Property   │     MAE      │      R²      │");
    println!("  ├─────────────┼──────────────┼──────────────┤");
    for (k, name) in ["logP       ", "QED        ", "SAS        "]
        .iter()
        .enumerate()
    {
        println!("  │  {name}│ {:>12.6} │ {:>12.6} │", mae[k], r2[k]);
    }
    println!("  └─────────────┴──────────────┴──────────────┘\n");
}

/// Percentage MAE reduction of `enhanced` relative to `baseline`, per property.
pub fn improvement_pct(baseline: &[f64; 3], enhanced: &[f64; 3]) -> [f64; 3] {
    std::array::from_fn(|k| {
        if baseline[k] > 1e-12 {
            (1.0 - enhanced[k] / baseline[k]) * 100.0
        } else {
            0.0
        }
    })
}

pub fn print_gain_ledger(base: &Head, enh: &Head, imp: &[f64; 3]) {
    print_phase(" [PHASE 6] 2-BODY ENTANGLEMENT CORRELATOR EXPRESSIVITY GAIN LEDGER      ");
    println!(" Observable Expansion : 12 → 78 (6.5x Feature Dimensionality)");
    println!(" Readout Parameters   : 39 → 237 (6.1x Parameter Capacity)\n");
    println!(" ┌─────────────┬──────────────────┬──────────────────┬──────────────────┐");
    println!(" │  Property   │  Baseline MAE    │  Enhanced MAE    │  Improvement     │");
    println!(" ├─────────────┼──────────────────┼──────────────────┼──────────────────┤");
    let names = ["logP       ", "QED        ", "SAS        "];
    for (k, name) in names.iter().enumerate() {
        println!(
            " │  {name}│ {:>16.6} │ {:>16.6} │ {:>+14.2}%  │",
            base.mae[k], enh.mae[k], imp[k]
        );
    }
    println!(" └─────────────┴──────────────────┴──────────────────┴──────────────────┘\n");
    println!(" ┌─────────────┬──────────────────┬──────────────────┐");
    println!(" │  Property   │  Baseline R²     │  Enhanced R²     │");
    println!(" ├─────────────┼──────────────────┼──────────────────┤");
    for (k, name) in names.iter().enumerate() {
        println!(" │  {name}│ {:>16.6} │ {:>16.6} │", base.r2[k], enh.r2[k]);
    }
    println!(" └─────────────┴──────────────────┴──────────────────┘");
    println!("{RULE}\n");
}

/// Phase 8: the molecules whose logP error shrinks most with the ZZ correlators.
pub fn print_top_corrections(test_samples: &[ZincMolecule], base: &Head, enh: &Head) {
    print_phase(" [PHASE 8] TOP OUTLIER CORRECTIONS (Largest Absolute Error Reduction)    ");
    let mut corrections: Vec<(usize, f64, f64, f64, &str)> = Vec::new();
    for (i, sample) in test_samples.iter().enumerate() {
        let err_b = (base.preds[i][0] - sample.log_p).abs();
        let err_e = (enh.preds[i][0] - sample.log_p).abs();
        corrections.push((i, err_b - err_e, err_b, err_e, &sample.smiles));
    }
    corrections.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!(" Top 10 logP Error Reductions (Baseline → Enhanced):");
    println!(
        " ┌────────┬────────────────┬────────────────┬────────────────┬──────────────────────────┐"
    );
    println!(
        " │  Rank  │  Baseline |ε|  │  Enhanced |ε|  │  Δ Correction  │  SMILES (truncated)      │"
    );
    println!(
        " ├────────┼────────────────┼────────────────┼────────────────┼──────────────────────────┤"
    );
    for (rank, &(_idx, correction, err_b, err_e, smiles)) in corrections.iter().take(10).enumerate()
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
            err_e,
            correction,
            smi_short
        );
    }
    println!(" └────────┴────────────────┴────────────────┴────────────────┴──────────────────────────┘\n");
}

/// Phase 9: the logP readout weights with the largest magnitude.
pub fn print_zz_heatmap(w_78: &[Vec<f64>]) {
    print_phase(" [PHASE 9] ZZ CORRELATOR READOUT WEIGHT HEATMAP (logP)                  ");
    println!(" Top 15 Most Influential Two-Body Correlators for logP Prediction:\n");

    let mut zz_weights: Vec<(usize, usize, f64)> = Vec::new();
    for i in 1..N_QUBITS {
        for j in 0..i {
            zz_weights.push((i, j, w_78[N_QUBITS + pair_index(i, j)][0])); // logP weight
        }
    }
    zz_weights.sort_by(|a, b| {
        b.2.abs()
            .partial_cmp(&a.2.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!(" ┌────────┬────────────┬──────────────────┐");
    println!(" │  Rank  │  Pair      │  Weight (logP)   │");
    println!(" ├────────┼────────────┼──────────────────┤");
    for (rank, &(qi, qj, w)) in zz_weights.iter().take(15).enumerate() {
        println!(
            " │  {:>4}  │  Z_{} Z_{}   │ {:>+16.8} │",
            rank + 1,
            qi,
            qj,
            w
        );
    }
    println!(" └────────┴────────────┴──────────────────┘");
    println!("{RULE}\n");
}

/// What the report needs from the earlier phases.
pub struct Results<'a> {
    pub train_len: usize,
    pub test_len: usize,
    pub max_zz_error: f64,
    pub zz_verified: bool,
    pub base: &'a Head,
    pub enh: &'a Head,
    pub imp: [f64; 3],
    pub throughput: (f64, f64),
}

pub fn build_report(
    r: &Results,
    classical_baselines: Vec<t1000::baselines::BaselineResult>,
) -> ZZCorrelatorReport {
    ZZCorrelatorReport {
        classical_baselines,
        timestamp_rfc3339: utc_timestamp_rfc3339(),
        host_cpu: t1000::hostinfo::HostProfile::detect().label(),
        n_qubits: N_QUBITS,
        n_single_qubit_obs: N_QUBITS,
        n_two_body_obs: N_PAIRS,
        n_total_obs: N_OBS,
        n_readout_weights_baseline: N_QUBITS * N_PROPS + N_PROPS,
        n_readout_weights_enhanced: N_OBS * N_PROPS + N_PROPS,
        train_samples: r.train_len,
        test_samples: r.test_len,
        fused_zz_max_error: r.max_zz_error,
        fused_zz_verified: r.zz_verified,
        baseline_mae_logp: r.base.mae[0],
        baseline_mae_qed: r.base.mae[1],
        baseline_mae_sas: r.base.mae[2],
        baseline_r2_logp: r.base.r2[0],
        baseline_r2_qed: r.base.r2[1],
        baseline_r2_sas: r.base.r2[2],
        enhanced_mae_logp: r.enh.mae[0],
        enhanced_mae_qed: r.enh.mae[1],
        enhanced_mae_sas: r.enh.mae[2],
        enhanced_r2_logp: r.enh.r2[0],
        enhanced_r2_qed: r.enh.r2[1],
        enhanced_r2_sas: r.enh.r2[2],
        mae_improvement_logp_pct: r.imp[0],
        mae_improvement_qed_pct: r.imp[1],
        mae_improvement_sas_pct: r.imp[2],
        throughput_baseline_mols_per_sec: r.throughput.0,
        throughput_enhanced_mols_per_sec: r.throughput.1,
    }
}

/// The enhanced checkpoint: 60 circuit angles, then the 78x3 readout weights and 3 biases.
pub fn enhanced_checkpoint_json(circuit_params: &[f64], head: &Head) -> serde_json::Value {
    let mut enhanced_params = Vec::with_capacity(60 + N_OBS * N_PROPS + N_PROPS);
    enhanced_params.extend_from_slice(circuit_params); // 60 circuit angles
    for f in 0..N_OBS {
        for t in 0..N_PROPS {
            enhanced_params.push(head.w[f][t]);
        }
    }
    enhanced_params.extend_from_slice(&head.b); // 3 biases

    serde_json::json!({
        "timestamp_rfc3339": utc_timestamp_rfc3339(),
        "n_qubits": N_QUBITS,
        "depth": 3,
        "circuit_params": 60,
        "readout_obs": N_OBS,
        "readout_weights": N_OBS * N_PROPS,
        "readout_biases": N_PROPS,
        "total_params": enhanced_params.len(),
        "enhanced_hybrid_params": enhanced_params,
    })
}

/// Saves the report and the enhanced checkpoint, and prints the classical comparison.
pub fn save_outputs(
    train: &[ZincMolecule],
    test: &[ZincMolecule],
    circuit_params: &[f64],
    results: &Results,
) {
    let report_path = resolve_workspace_path("tests/benchmarks/zinc_molecular/reports");
    let report_file = report_path.join("zinc_12qubit_zz_correlator_report.json");
    if !report_path.exists() {
        let _ = fs::create_dir_all(&report_path);
    }

    // Classical reference: ridge regression on the same raw features and train/test split.
    let classical_baselines = t1000::baselines::zinc_baselines(
        train
            .iter()
            .map(|s| (s.features.as_slice(), [s.log_p, s.qed, s.sas])),
        test.iter()
            .map(|s| (s.features.as_slice(), [s.log_p, s.qed, s.sas])),
    );
    let report = build_report(results, classical_baselines.clone());
    let qnn_rows = [
        QnnScore::new("12 Z readout")
            .mae(results.base.mae)
            .r2(results.base.r2),
        QnnScore::new("Z + ZZ readout (78 obs)")
            .mae(results.enh.mae)
            .r2(results.enh.r2),
    ];
    print_comparison(&qnn_rows, &classical_baselines);

    if let Ok(json_str) = serde_json::to_string_pretty(&report) {
        let _ = t1000::results::write(&report_file, json_str);
        println!("[SAVED] ZZ Correlator Report: {}", report_file.display());
    }

    // Save enhanced checkpoint with 78-observable readout weights
    let enhanced_ckpt_path = match report_path.parent() {
        Some(parent) => parent.join("checkpoints/zinc_12qubit_checkpoint_zz78.json"),
        None => PathBuf::from("zinc_12qubit_checkpoint_zz78.json"),
    };
    let enhanced_ckpt = enhanced_checkpoint_json(circuit_params, results.enh);
    if let Ok(json_str) = serde_json::to_string_pretty(&enhanced_ckpt) {
        let _ = t1000::results::write(&enhanced_ckpt_path, json_str);
        println!(
            "[SAVED] Enhanced 78-Observable Checkpoint: {}",
            enhanced_ckpt_path.display()
        );
    }

    println!("\n{RULE}");
    println!("  EXECUTION COMPLETE: 2-Body Entanglement Correlators Deployed          ");
    println!("{RULE}");
}
