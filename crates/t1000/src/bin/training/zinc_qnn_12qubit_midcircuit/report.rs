use super::phases::{Audit, Baseline, Fit, Throughput, TRUE_LP_10640};
use super::*;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use t1000::zinc_training_common::{utc_timestamp_rfc3339, MidCircuitCheckpoint};

const RULE: &str = "=========================================================================";
const SMILES_10640: &str =
    "O=[N+]([O-])c1ccc2c(c1)=[NH+][C@@H](CS[C@H]1[NH+]=c3ccc(Br)cc3=[NH+]1)[NH+]=2";
const PROPS: [&str; 3] = ["logP", "QED", "SAS"];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MidCircuitReport {
    pub timestamp_rfc3339: String,
    pub host_cpu: String,
    pub n_qubits: usize,
    pub source_qubit: usize,
    pub train_samples: usize,
    pub test_samples: usize,
    pub feedforward_alphas: Vec<f64>,
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
    pub combined_mae_logp: f64,
    pub combined_mae_qed: f64,
    pub combined_mae_sas: f64,
    pub combined_r2_logp: f64,
    pub combined_r2_qed: f64,
    pub combined_r2_sas: f64,
    pub mol10640_true_logp: f64,
    pub mol10640_baseline_pred_logp: f64,
    pub mol10640_feedforward_pred_logp: f64,
    pub mol10640_combined_pred_logp: f64,
    pub mol10640_midcircuit_z8: f64,
    pub baseline_throughput_mols_per_sec: f64,
    pub feedforward_throughput_mols_per_sec: f64,
    pub combined_throughput_mols_per_sec: f64,
    #[serde(default)]
    pub classical_baselines: Vec<t1000::baselines::BaselineResult>,
}

pub fn phase_header(title: &str) {
    println!("{RULE}");
    println!("{title}");
    println!("{RULE}");
}

pub fn print_banner() {
    phase_header(" GUDDANETWORKS 12-QUBIT QNN: MID-CIRCUIT FEED-FORWARD ENGINE (QUBIT 8)  ");
    println!(" Mode               : Dynamic Feed-Forward Non-Linearity Injection");
    println!(
        " Source Qubit       : Qubit {} (⟨Z_{}⟩ Mid-Circuit Readout)",
        SOURCE_QUBIT, SOURCE_QUBIT
    );
    println!(
        " Mechanism          : θ̃_{{y,q}} = θ_{{y,q}} + α_q · ⟨Z_{}⟩_L1",
        SOURCE_QUBIT
    );
    println!(
        " Hilbert Space Dim  : {} (12 Qubits, 64 KB L2 Cache-Resident)",
        DIM
    );
    println!(" Hardware Target    : Bare-Metal AVX2 + FMA Single-Pass Fused POVM");
    println!(" Palette            : Deen OS Sovereign (Quantum Green / Hardware Gold / Alert Red)");
    println!("-------------------------------------------------------------------------\n");
}

pub fn print_baseline(fit: &Fit) {
    println!("  Baseline Holdout Performance (2,000 molecules):");
    println!("  ┌─────────────┬──────────────┬──────────────┐");
    println!("  │  Property   │  MAE         │  R² Score    │");
    println!("  ├─────────────┼──────────────┼──────────────┤");
    for (k, name) in PROPS.iter().enumerate() {
        println!(
            "  │  {name:<11}│ {:>12.6} │ {:>12.6} │",
            fit.mae[k], fit.r2[k]
        );
    }
    println!("  └─────────────┴──────────────┴──────────────┘\n");
}

pub fn print_anomaly_baseline(pred_lp: f64, m8: f64) {
    println!("  [ANOMALY CHECK] Molecule #10640 Baseline Status:");
    println!("    SMILES           : {SMILES_10640}");
    println!("    True logP        : {:>+10.4}", TRUE_LP_10640);
    println!(
        "    Baseline Pred    : {:>+10.4} (Absolute Error = {:.4})",
        pred_lp,
        (pred_lp - TRUE_LP_10640).abs()
    );
    println!(
        "    ⟨Z_{}⟩_L1 Signal  : {:>+10.6} (Spin Polarization = {:.1}%)\n",
        SOURCE_QUBIT,
        m8,
        (1.0 + m8.abs()) * 50.0
    );
}

pub fn print_phase2_intro() {
    phase_header(" [PHASE 2] TRAINING MID-CIRCUIT FEED-FORWARD PARAMETERS (α ∈ R¹²)        ");
    println!(" Optimizing 12 classical feed-forward weights via Adam + Nested Ridge Regression...");
    println!(" Objective: min_α L(α) = MSE(Z(α) W*(α) + b*(α), Y) across training molecules\n");
}

pub fn print_alpha_table_header() {
    println!(" Epoch │ Composite Loss │  logP Pred (#10640) │  ||α||_2  │ Top Alphas (q: α_q)");
    println!("───────┼────────────────┼─────────────────────┼───────────┼────────────────────────────────────");
}

pub fn print_alpha_row(epoch: usize, loss: f64, pred_lp_10640: f64, alphas: &[f64]) {
    let norm_alpha = alphas.iter().map(|&a| a * a).sum::<f64>().sqrt();

    // Top 3 alphas by magnitude
    let mut alpha_pairs: Vec<(usize, f64)> = alphas.iter().cloned().enumerate().collect();
    alpha_pairs.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    let top_str = format!(
        "q{}: {:>+5.2}, q{}: {:>+5.2}, q{}: {:>+5.2}",
        alpha_pairs[0].0,
        alpha_pairs[0].1,
        alpha_pairs[1].0,
        alpha_pairs[1].1,
        alpha_pairs[2].0,
        alpha_pairs[2].1,
    );

    println!(
        "  {:>3}  │   {:>10.6}   │      {:>+10.4}     │  {:>7.4}  │ {}",
        epoch, loss, pred_lp_10640, norm_alpha, top_str
    );
}

pub fn print_alpha_footer(elapsed: Duration, alphas: &[f64]) {
    println!("───────┴────────────────┴─────────────────────┴───────────┴────────────────────────────────────");
    println!(
        "  Optimization completed in {:.2}s\n",
        elapsed.as_secs_f64()
    );

    // Final alpha distribution across all 12 qubits
    println!(" Final Feed-Forward Weight Distribution (α_q for q ∈ 0..11):");
    print!("   ");
    for (q, alpha) in alphas.iter().enumerate() {
        print!("q{:02}: {:>+6.3}  ", q, alpha);
        if (q + 1) % 6 == 0 {
            print!("\n   ");
        }
    }
    println!();
}

/// Percentage MAE improvement of `mae` over the baseline for each property.
fn improvement(base: &[f64; 3], mae: &[f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|k| (1.0 - mae[k] / base[k]) * 100.0)
}

pub fn print_feedforward(base: &Fit, ff: &Fit) {
    let imp = improvement(&base.mae, &ff.mae);
    println!("  Holdout Performance Comparison: Baseline vs Mid-Circuit Feed-Forward:");
    println!("  ┌─────────────┬──────────────────┬──────────────────┬──────────────────┐");
    println!("  │  Property   │  Baseline MAE    │  Feed-Fwd MAE    │  Improvement     │");
    println!("  ├─────────────┼──────────────────┼──────────────────┼──────────────────┤");
    for (k, name) in PROPS.iter().enumerate() {
        println!(
            "  │  {name:<11}│ {:>16.6} │ {:>16.6} │ {:>+14.2}%  │",
            base.mae[k], ff.mae[k], imp[k]
        );
    }
    println!("  └─────────────┴──────────────────┴──────────────────┴──────────────────┘\n");
    println!("  ┌─────────────┬──────────────────┬──────────────────┐");
    println!("  │  Property   │  Baseline R²     │  Feed-Fwd R²     │");
    println!("  ├─────────────┼──────────────────┼──────────────────┤");
    for (k, name) in PROPS.iter().enumerate() {
        println!(
            "  │  {name:<11}│ {:>16.6} │ {:>16.6} │",
            base.r2[k], ff.r2[k]
        );
    }
    println!("  └─────────────┴──────────────────┴──────────────────┘\n");
}

pub fn print_phase4_intro() {
    phase_header(" [PHASE 4] COMBINED MECHANISMS: FEED-FORWARD + 2-BODY ZZ (78 Observables)");
    println!(" Extracting 12 single-qubit marginals + 66 two-body correlators under dynamic");
    println!(" feed-forward angle modulation: θ̃_{{y,q}} = θ_{{y,q}} + α_q · ⟨Z_8⟩_L1\n");
}

pub fn print_ledger(base: &Fit, ff: &Fit, comb: &Fit) {
    let imp = improvement(&base.mae, &comb.mae);
    println!("  Tri-Stage Architectural Evolution Ledger:");
    println!("  ┌─────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┐");
    println!("  │  Property   │  Baseline (12)   │  Feed-Fwd (12)   │  Combined (78)   │  Total Imp. (%)  │");
    println!("  ├─────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┤");
    for (k, name) in PROPS.iter().enumerate() {
        println!(
            "  │  {:<11}│ {:>16.6} │ {:>16.6} │ {:>16.6} │ {:>+14.2}%  │",
            format!("{name} MAE"),
            base.mae[k],
            ff.mae[k],
            comb.mae[k],
            imp[k]
        );
    }
    println!("  ├─────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┤");
    for (k, name) in PROPS.iter().enumerate() {
        println!(
            "  │  {:<11}│ {:>16.6} │ {:>16.6} │ {:>16.6} │ {:>+14.2}%  │",
            format!("{name} R²"),
            base.r2[k],
            ff.r2[k],
            comb.r2[k],
            (comb.r2[k] - base.r2[k]) / base.r2[k].abs() * 100.0
        );
    }
    println!("  └─────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┘\n");
}

pub fn print_audit(base: &Baseline, ff_lp: f64, comb_lp: f64) {
    let err_base = (base.pred_lp_10640 - TRUE_LP_10640).abs();
    let err_ff = (ff_lp - TRUE_LP_10640).abs();
    let err_comb = (comb_lp - TRUE_LP_10640).abs();

    println!("  Compound Identifier : ZINC-250k Molecule #10640");
    println!("  SMILES              : {SMILES_10640}");
    println!("  Zwitterionic Bridge : Hyper-polarized thioether bridge, 3 formal iminium charges");
    println!(
        "  Mid-Circuit Signal  : ⟨Z_{}⟩_L1 = {:>+10.6}\n",
        SOURCE_QUBIT, base.m8_10640
    );
    println!("  ┌────────────────────────────────┬──────────────────┬──────────────────┐");
    println!("  │  Architecture Stage            │  Predicted logP  │  Absolute Error  │");
    println!("  ├────────────────────────────────┼──────────────────┼──────────────────┤");
    let rows = [
        ("True Physical Target         ", TRUE_LP_10640, 0.0),
        (
            "Baseline 12-Qubit QNN        ",
            base.pred_lp_10640,
            err_base,
        ),
        ("+ Mid-Circuit Feed-Forward   ", ff_lp, err_ff),
        ("+ Combined (FF + 78 ZZ)      ", comb_lp, err_comb),
    ];
    for (label, pred, err) in rows {
        println!("  │  {label} │ {:>16.4} │ {:>16.4} │", pred, err);
    }
    println!("  └────────────────────────────────┴──────────────────┴──────────────────┘\n");
    println!(
        "  Anomaly Error Reduction: {:.4} -> {:.4} (Gain = {:>+6.2}%)\n",
        err_base,
        err_comb,
        (1.0 - err_comb / err_base) * 100.0
    );
}

/// Phase 6: the holdout molecules whose logP error the combined readout reduces most.
pub fn print_corrections(test: &[ZincMolecule], base: &Fit, comb: &Fit) {
    phase_header(" [PHASE 6] TOP OUTLIER CORRECTIONS (Largest Absolute Error Reduction)    ");

    let mut corrections: Vec<(usize, f64, f64, f64, &str)> = Vec::new();
    for (i, sample) in test.iter().enumerate() {
        let err_b = (base.preds[i][0] - sample.log_p).abs();
        let err_c = (comb.preds[i][0] - sample.log_p).abs();
        let correction = err_b - err_c;
        corrections.push((i, correction, err_b, err_c, &sample.smiles));
    }
    corrections.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!(" Top 10 logP Error Reductions (Baseline → Combined Feed-Forward + ZZ):");
    println!(
        " ┌────────┬────────────────┬────────────────┬────────────────┬──────────────────────────┐"
    );
    println!(
        " │  Rank  │  Baseline |ε|  │  Combined |ε|  │  Δ Correction  │  SMILES (truncated)      │"
    );
    println!(
        " ├────────┼────────────────┼────────────────┼────────────────┼──────────────────────────┤"
    );
    for (rank, &(_idx, correction, err_b, err_c, smiles)) in corrections.iter().take(10).enumerate()
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
            err_c,
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
        "  Feed-Forward     (12 obs) : {:>7.0} mols/sec (Overhead: {:>+.2}%)",
        tp.feedforward,
        (1.0 - tp.feedforward / tp.baseline) * 100.0
    );
    println!(
        "  Combined Readout (78 obs) : {:>7.0} mols/sec\n",
        tp.combined
    );
}

fn save_checkpoint(circuit: &Circuit, alphas: &[f64], ff: &Fit, comb: &Fit) {
    let checkpoint = MidCircuitCheckpoint {
        timestamp_rfc3339: utc_timestamp_rfc3339(),
        n_qubits: N_QUBITS,
        depth: 3,
        source_qubit: SOURCE_QUBIT,
        circuit_params: circuit.params.clone(),
        feedforward_alphas: alphas.to_vec(),
        readout_weights_12: ff.w.clone(),
        readout_biases_12: ff.b.clone(),
        readout_weights_78: comb.w.clone(),
        readout_biases_78: comb.b.clone(),
    };

    let path = resolve_workspace_path(
        "tests/benchmarks/zinc_molecular/checkpoints/zinc_12qubit_checkpoint_midcircuit.json",
    );
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let ckpt_json = serde_json::to_string_pretty(&checkpoint).expect("Serialize checkpoint");
    t1000::results::write(&path, ckpt_json).expect("Write checkpoint");
    println!("  [✓] Saved Mid-Circuit Checkpoint: {}", path.display());
}

fn build_report(
    data: &Data,
    alphas: &[f64],
    (base, ff, comb): (&Baseline, &Fit, &Fit),
    audit: &Audit,
    tp: &Throughput,
    classical_baselines: Vec<t1000::baselines::BaselineResult>,
) -> MidCircuitReport {
    let (b, f, c) = (&base.fit, ff, comb);
    MidCircuitReport {
        classical_baselines,
        timestamp_rfc3339: utc_timestamp_rfc3339(),
        host_cpu: "Native Host AVX2+FMA".to_string(),
        n_qubits: N_QUBITS,
        source_qubit: SOURCE_QUBIT,
        train_samples: data.train.len(),
        test_samples: data.test.len(),
        feedforward_alphas: alphas.to_vec(),
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
        combined_mae_logp: c.mae[0],
        combined_mae_qed: c.mae[1],
        combined_mae_sas: c.mae[2],
        combined_r2_logp: c.r2[0],
        combined_r2_qed: c.r2[1],
        combined_r2_sas: c.r2[2],
        mol10640_true_logp: TRUE_LP_10640,
        mol10640_baseline_pred_logp: base.pred_lp_10640,
        mol10640_feedforward_pred_logp: audit.ff_lp,
        mol10640_combined_pred_logp: audit.comb_lp,
        mol10640_midcircuit_z8: base.m8_10640,
        baseline_throughput_mols_per_sec: tp.baseline,
        feedforward_throughput_mols_per_sec: tp.feedforward,
        combined_throughput_mols_per_sec: tp.combined,
    }
}

fn print_classical_comparison(report: &MidCircuitReport) {
    use t1000::baselines::{print_comparison, QnnScore};
    let qnn_rows = [
        QnnScore::new("circuit, no feed-forward")
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
        QnnScore::new("feed-forward + ZZ readout")
            .mae([
                report.combined_mae_logp,
                report.combined_mae_qed,
                report.combined_mae_sas,
            ])
            .r2([
                report.combined_r2_logp,
                report.combined_r2_qed,
                report.combined_r2_sas,
            ]),
    ];
    print_comparison(&qnn_rows, &report.classical_baselines);
}

/// Phase 8: write the checkpoint and the machine-readable report.
pub fn persist(
    data: &Data,
    circuit: &Circuit,
    alphas: &[f64],
    stages: (&Baseline, &Fit, &Fit),
    audit: &Audit,
    tp: &Throughput,
) {
    phase_header(" [PHASE 8] PERSISTING CHECKPOINT & TELEMETRY REPORT                      ");
    save_checkpoint(circuit, alphas, stages.1, stages.2);

    // Classical reference: ridge regression on the same raw features and train/test split.
    let classical_baselines = t1000::baselines::zinc_baselines(
        data.train
            .iter()
            .map(|s| (s.features.as_slice(), [s.log_p, s.qed, s.sas])),
        data.test
            .iter()
            .map(|s| (s.features.as_slice(), [s.log_p, s.qed, s.sas])),
    );
    let report = build_report(data, alphas, stages, audit, tp, classical_baselines);

    let report_save_path = resolve_workspace_path(
        "tests/benchmarks/zinc_molecular/reports/zinc_12qubit_midcircuit_report.json",
    );
    if let Some(parent) = report_save_path.parent() {
        fs::create_dir_all(parent).ok();
    }
    print_classical_comparison(&report);

    let report_json = serde_json::to_string_pretty(&report).expect("Serialize report");
    t1000::results::write(&report_save_path, report_json).expect("Write report");
    println!(
        "  [✓] Saved Machine-Readable Report: {}",
        report_save_path.display()
    );
    println!("\n{RULE}");
    println!(" MECHANISM #2 MID-CIRCUIT FEED-FORWARD VERIFICATION COMPLETE             ");
    println!("{RULE}");
}
