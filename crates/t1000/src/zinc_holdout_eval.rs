// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Holdout scoring and the printed performance ledger shared by the ZINC regression trainers:
//! per-property MSE/MAE/Pearson against the zero-shot baseline, and the boxed ledger table.

use crate::calibration::pearson_r;
use crate::zinc_data::ZincMolecule;
use crate::zinc_training_common::PropertyMetric;

/// Target values per property `[logP, QED, SAS]`, one column each.
pub fn target_columns(samples: &[ZincMolecule]) -> [Vec<f64>; 3] {
    [0, 1, 2].map(|k| samples.iter().map(|s| s.targets()[k]).collect())
}

/// Mean squared error and mean absolute error of `preds` against `targets`.
pub fn mse_mae(preds: &[f64], targets: &[f64]) -> (f64, f64) {
    let n = preds.len() as f64;
    let mse = preds
        .iter()
        .zip(targets)
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / n;
    let mae = preds
        .iter()
        .zip(targets)
        .map(|(p, t)| (p - t).abs())
        .sum::<f64>()
        / n;
    (mse, mae)
}

/// Prints the zero-shot MSE/MAE of each property and returns the three MSEs.
pub fn print_zero_shot(preds: &[Vec<f64>; 3], targets: &[Vec<f64>; 3]) -> [f64; 3] {
    let mut init_mse = [0.0; 3];
    for (k, name) in ["logP", "qed ", "SAS "].iter().enumerate() {
        let (mse, mae) = mse_mae(&preds[k], &targets[k]);
        println!("  Initial Test MSE - {name} : {mse:.4} (MAE: {mae:.4})");
        init_mse[k] = mse;
    }
    init_mse
}

/// Scores tuned predictions against the holdout targets, relative to the zero-shot MSE.
pub fn holdout_metrics(
    preds: &[Vec<f64>; 3],
    targets: &[Vec<f64>; 3],
    init_mse: [f64; 3],
) -> [PropertyMetric; 3] {
    [0, 1, 2].map(|k| {
        let (mse, mae) = mse_mae(&preds[k], &targets[k]);
        PropertyMetric {
            initial_mse: init_mse[k],
            tuned_mse: mse,
            tuned_mae: mae,
            pearson_r: pearson_r(&preds[k], &targets[k]),
            error_reduction_pct: ((init_mse[k] - mse) / init_mse[k]).max(0.0) * 100.0,
        }
    })
}

/// The boxed holdout ledger; `title_row` is the full second line of the frame.
pub fn print_ledger(title_row: &str, metrics: &[PropertyMetric; 3]) {
    println!("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("{title_row}");
    println!("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Target Property         │ Initial MSE │ Tuned MSE   │ Tuned MAE   │ Pearson R │ Error Reduction   ║");
    println!("╟─────────────────────────┼─────────────┼─────────────┼─────────────┼───────────┼───────────────────╢");
    let labels = [
        "logP (Lipophilicity)   ",
        "qed  (Drug-Likeness)   ",
        "SAS  (Synthetic Access)",
    ];
    for (label, m) in labels.iter().zip(metrics) {
        println!(
            "║ {label} │ {:>11.4} │ {:>11.4} │ {:>11.4} │ {:>9.4} │ {:>15.2}% ║",
            m.initial_mse, m.tuned_mse, m.tuned_mae, m.pearson_r, m.error_reduction_pct
        );
    }
    println!("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════╝");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn molecule(log_p: f64, qed: f64, sas: f64) -> ZincMolecule {
        ZincMolecule {
            smiles: String::new(),
            features: vec![],
            log_p,
            qed,
            sas,
        }
    }

    #[test]
    fn target_columns_split_the_three_properties() {
        let cols = target_columns(&[molecule(1.0, 2.0, 3.0), molecule(4.0, 5.0, 6.0)]);
        assert_eq!(cols, [vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);
    }

    #[test]
    fn mse_and_mae_average_over_the_samples() {
        let (mse, mae) = mse_mae(&[1.0, 3.0], &[0.0, 5.0]);
        assert_eq!((mse, mae), ((1.0 + 4.0) / 2.0, (1.0 + 2.0) / 2.0));
    }

    #[test]
    fn perfect_predictions_reduce_the_error_completely() {
        let targets = [
            vec![1.0, 2.0, 3.0],
            vec![0.1, 0.4, 0.2],
            vec![5.0, 4.0, 6.0],
        ];
        let m = holdout_metrics(&targets, &targets, [2.0, 1.0, 4.0]);
        for p in &m {
            assert_eq!(
                (p.tuned_mse, p.tuned_mae, p.error_reduction_pct),
                (0.0, 0.0, 100.0)
            );
            assert!((p.pearson_r - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn a_worse_than_baseline_model_reports_zero_reduction() {
        let targets = [vec![0.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.0]];
        let preds = [vec![1.0, 1.0], vec![1.0, 1.0], vec![1.0, 1.0]];
        let m = holdout_metrics(&preds, &targets, [0.5, 0.5, 0.5]);
        assert!(m
            .iter()
            .all(|p| p.error_reduction_pct == 0.0 && p.tuned_mse == 1.0));
    }

    #[test]
    fn zero_shot_returns_the_per_property_mse() {
        let targets = [vec![0.0], vec![0.0], vec![0.0]];
        let preds = [vec![1.0], vec![2.0], vec![3.0]];
        assert_eq!(print_zero_shot(&preds, &targets), [1.0, 4.0, 9.0]);
    }
}
