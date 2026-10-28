// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Pieces shared by the 12-qubit ZINC stage trainers (mid-circuit feed-forward and data
//! re-uploading): the train/holdout data bundle, ridge readout fits, a finite-difference Adam
//! step for 12 classical angles, and the report plumbing around them.

use crate::zinc_data::{self, ZincMolecule};
use crate::zinc_training_common::{
    compute_mae_and_r2, resolve_workspace_path, solve_ridge_regression,
};
use std::time::Instant;

/// Ridge strength used for every readout fit.
pub const RIDGE: f64 = 1e-3;
/// Central finite-difference step.
pub const FD_EPS: f64 = 0.01;
const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;

/// Descriptor of the anomalous molecule #10640.
pub const FEATS_10640: [f64; 19] = [
    24.0, 408.279, 15.0, 5.0, 2.0, 1.0, 0.0, 0.0, 1.0, 1.0, 12.0, 0.5, 9.0, 0.375, 5.0, 0.0, 4.0,
    3.0, 4.0,
];
pub const TRUE_LP_10640: f64 = -6.7716;
pub const SMILES_10640: &str =
    "O=[N+]([O-])c1ccc2c(c1)=[NH+][C@@H](CS[C@H]1[NH+]=c3ccc(Br)cc3=[NH+]1)[NH+]=2";
/// Property names in target order.
pub const PROPS: [&str; 3] = ["logP", "QED", "SAS"];

const RULE: &str = "=========================================================================";

/// Prints a title between two rules.
pub fn phase_header(title: &str) {
    println!("{RULE}");
    println!("{title}");
    println!("{RULE}");
}

/// Train and holdout molecules with their (logP, QED, SAS) target rows.
#[derive(Debug)]
pub struct Data {
    pub train: Vec<ZincMolecule>,
    pub test: Vec<ZincMolecule>,
    pub y_train: Vec<Vec<f64>>,
    pub y_test: Vec<Vec<f64>>,
}

impl Data {
    pub fn new(train: Vec<ZincMolecule>, test: Vec<ZincMolecule>) -> Self {
        let rows = |s: &[ZincMolecule]| s.iter().map(|m| vec![m.log_p, m.qed, m.sas]).collect();
        Data {
            y_train: rows(&train),
            y_test: rows(&test),
            train,
            test,
        }
    }
}

/// Loads a CSV from `rel`, falling back to `fallback_rel`; reports the first error if both fail.
fn load_with_fallback(rel: &str, fallback_rel: &str, n: usize) -> Option<Vec<ZincMolecule>> {
    match zinc_data::load(resolve_workspace_path(rel), n) {
        Ok(s) => Some(s),
        Err(e) => match zinc_data::load(resolve_workspace_path(fallback_rel), n) {
            Ok(s) => Some(s),
            Err(_) => {
                eprintln!("[-] Error: {}", e);
                None
            }
        },
    }
}

/// Loads the 10k training set and the 2k holdout set, announcing each.
pub fn load_stage_data() -> Option<Data> {
    println!("[INIT] Loading training set (10,000 molecules)...");
    let train = load_with_fallback(
        "tests/benchmarks/zinc_molecular/data/zinc_train_10k.csv",
        "tests/benchmarks/zinc_molecular/outputs/zinc_train_10k.csv",
        10_000,
    )?;
    println!("  -> Loaded {} training molecules.\n", train.len());

    println!("[INIT] Loading holdout test set (2,000 molecules)...");
    let test = load_with_fallback(
        "tests/benchmarks/zinc_molecular/data/zinc_test_2k.csv",
        "tests/benchmarks/zinc_molecular/outputs/zinc_test_2k.csv",
        2_000,
    )?;
    println!("  -> Loaded {} test molecules.\n", test.len());
    Some(Data::new(train, test))
}

/// `b + Σ_q w[q] · x[q]` for each of the three properties.
pub fn linear_predict(w: &[Vec<f64>], b: &[f64], x: &[f64]) -> [f64; 3] {
    let mut p = [b[0], b[1], b[2]];
    for (wq, xq) in w.iter().zip(x) {
        for k in 0..3 {
            p[k] += wq[k] * xq;
        }
    }
    p
}

/// A ridge readout head and its holdout predictions and scores.
#[derive(Debug)]
pub struct Fit {
    pub w: Vec<Vec<f64>>,
    pub b: Vec<f64>,
    pub preds: Vec<Vec<f64>>,
    pub mae: [f64; 3],
    pub r2: [f64; 3],
}

/// Ridge fit on the train rows, scored on the holdout rows.
pub fn fit_and_score(data: &Data, x_train: &[Vec<f64>], x_test: &[Vec<f64>]) -> Fit {
    let (w, b) = solve_ridge_regression(x_train, &data.y_train, RIDGE);
    let preds: Vec<Vec<f64>> = x_test
        .iter()
        .map(|x| linear_predict(&w, &b, x).to_vec())
        .collect();
    let (mae, r2) = compute_mae_and_r2(&preds, &data.y_test);
    Fit {
        w,
        b,
        preds,
        mae,
        r2,
    }
}

/// Percentage MAE improvement of `mae` over the baseline for each property.
pub fn improvement(base: &[f64; 3], mae: &[f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|k| (1.0 - mae[k] / base[k]) * 100.0)
}

/// Composite loss of a ridge head: logP and SAS weigh 1.0, QED 5.0.
pub fn composite_loss(w: &[Vec<f64>], b: &[f64], x_mat: &[Vec<f64>], y_mat: &[Vec<f64>]) -> f64 {
    let mut total_loss = 0.0;
    for (x_row, y_row) in x_mat.iter().zip(y_mat) {
        let p = linear_predict(w, b, x_row);
        let e_lp = p[0] - y_row[0];
        let e_qed = p[1] - y_row[1];
        let e_sas = p[2] - y_row[2];
        // Calibrated weights: 1.0 for logP, 5.0 for QED, 1.0 for SAS
        total_loss += 1.0 * (e_lp * e_lp) + 5.0 * (e_qed * e_qed) + 1.0 * (e_sas * e_sas);
    }
    total_loss / x_mat.len() as f64
}

/// Central finite-difference gradient of `loss` with respect to each entry of `params`.
pub fn central_difference(params: &[f64], mut loss: impl FnMut(&[f64]) -> f64) -> Vec<f64> {
    let mut grad = vec![0.0; params.len()];
    for q in 0..params.len() {
        let mut plus = params.to_vec();
        plus[q] += FD_EPS;
        let l_plus = loss(&plus);

        let mut minus = params.to_vec();
        minus[q] -= FD_EPS;
        let l_minus = loss(&minus);

        grad[q] = (l_plus - l_minus) / (2.0 * FD_EPS);
    }
    grad
}

/// Adam moment estimates for a vector of classical angles.
#[derive(Debug)]
pub struct Adam {
    m: Vec<f64>,
    v: Vec<f64>,
    lr: f64,
}

impl Adam {
    pub fn new(n: usize, lr: f64) -> Self {
        Adam {
            m: vec![0.0; n],
            v: vec![0.0; n],
            lr,
        }
    }

    /// One update at (1-based) `epoch`, clamping each angle to [-π, π] to avoid wrap-around.
    pub fn step(&mut self, params: &mut [f64], grad: &[f64], epoch: usize) {
        for q in 0..params.len() {
            self.m[q] = BETA1 * self.m[q] + (1.0 - BETA1) * grad[q];
            self.v[q] = BETA2 * self.v[q] + (1.0 - BETA2) * grad[q] * grad[q];
            let m_hat = self.m[q] / (1.0 - BETA1.powi(epoch as i32));
            let v_hat = self.v[q] / (1.0 - BETA2.powi(epoch as i32));
            params[q] -= self.lr * m_hat / (v_hat.sqrt() + 1e-8);
            params[q] = params[q].clamp(-std::f64::consts::PI, std::f64::consts::PI);
        }
    }
}

/// One row of the angle-optimisation table.
pub fn print_angle_row(epoch: usize, loss: f64, pred_lp_10640: f64, params: &[f64]) {
    let norm = params.iter().map(|&a| a * a).sum::<f64>().sqrt();

    // Top 3 by magnitude
    let mut pairs: Vec<(usize, f64)> = params.iter().cloned().enumerate().collect();
    pairs.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    let top_str = format!(
        "q{}: {:>+5.2}, q{}: {:>+5.2}, q{}: {:>+5.2}",
        pairs[0].0, pairs[0].1, pairs[1].0, pairs[1].1, pairs[2].0, pairs[2].1,
    );

    println!(
        "  {:>3}  │   {:>10.6}   │      {:>+10.4}     │  {:>7.4}  │ {}",
        epoch, loss, pred_lp_10640, norm, top_str
    );
}

/// All angles, six per line.
pub fn print_angle_grid(params: &[f64]) {
    print!("   ");
    for (q, v) in params.iter().enumerate() {
        print!("q{:02}: {:>+6.3}  ", q, v);
        if (q + 1) % 6 == 0 {
            print!("\n   ");
        }
    }
    println!();
}

/// Molecules per second over `repeats` passes of the holdout set.
pub fn mols_per_sec(
    test: &[ZincMolecule],
    repeats: usize,
    mut run: impl FnMut(&ZincMolecule),
) -> f64 {
    let t = Instant::now();
    for _ in 0..repeats {
        test.iter().for_each(&mut run);
    }
    (test.len() * repeats) as f64 / t.elapsed().as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn molecule(x: f64) -> ZincMolecule {
        ZincMolecule {
            smiles: String::new(),
            features: vec![x],
            log_p: 2.0 * x,
            qed: x,
            sas: 1.0,
        }
    }

    #[test]
    fn data_builds_target_rows_in_logp_qed_sas_order() {
        let d = Data::new(vec![molecule(1.0)], vec![molecule(2.0)]);
        assert_eq!(d.y_train, vec![vec![2.0, 1.0, 1.0]]);
        assert_eq!(d.y_test, vec![vec![4.0, 2.0, 1.0]]);
    }

    #[test]
    fn linear_predict_adds_the_bias_and_weighted_inputs() {
        let w = vec![vec![1.0, 2.0, 3.0], vec![0.5, 0.0, -1.0]];
        let p = linear_predict(&w, &[10.0, 20.0, 30.0], &[2.0, 4.0]);
        assert_eq!(p, [10.0 + 2.0 + 2.0, 20.0 + 4.0, 30.0 + 6.0 - 4.0]);
    }

    #[test]
    fn improvement_is_positive_when_the_error_shrinks() {
        let imp = improvement(&[2.0, 1.0, 4.0], &[1.0, 1.0, 5.0]);
        assert_eq!(imp, [50.0, 0.0, -25.0]);
    }

    #[test]
    fn a_ridge_fit_recovers_an_exact_linear_readout() {
        let train: Vec<ZincMolecule> = (0..20).map(|i| molecule(f64::from(i) * 0.1)).collect();
        let test: Vec<ZincMolecule> = (0..5).map(|i| molecule(f64::from(i) * 0.3)).collect();
        let data = Data::new(train, test);
        let x = |ms: &[ZincMolecule]| -> Vec<Vec<f64>> {
            ms.iter().map(|m| m.features.clone()).collect()
        };
        let fit = fit_and_score(&data, &x(&data.train), &x(&data.test));
        assert!(fit.mae[0] < 1e-2 && fit.mae[1] < 1e-2, "mae {:?}", fit.mae);
    }

    #[test]
    fn composite_loss_weighs_qed_five_times() {
        let w = vec![vec![0.0; 3]];
        let loss = composite_loss(&w, &[0.0, 0.0, 0.0], &[vec![0.0]], &[vec![1.0, 1.0, 1.0]]);
        assert_eq!(loss, 7.0);
    }

    #[test]
    fn central_difference_matches_the_analytic_gradient_of_a_quadratic() {
        let g = central_difference(&[1.0, -2.0], |p| p[0] * p[0] + 3.0 * p[1]);
        assert!(
            (g[0] - 2.0).abs() < 1e-9 && (g[1] - 3.0).abs() < 1e-9,
            "{g:?}"
        );
    }

    #[test]
    fn adam_moves_against_the_gradient_and_clamps_to_pi() {
        let mut adam = Adam::new(2, 0.1);
        let mut params = [0.0, 3.1];
        adam.step(&mut params, &[1.0, -1e6], 1);
        assert!(params[0] < 0.0, "descends along a positive gradient");
        assert!(
            (params[1] - std::f64::consts::PI).abs() < 1e-12,
            "clamped at π"
        );
    }

    #[test]
    fn mols_per_sec_visits_every_molecule_each_repeat() {
        let test = vec![molecule(0.0), molecule(1.0)];
        let mut calls = 0;
        let rate = mols_per_sec(&test, 3, |_| calls += 1);
        assert_eq!(calls, 6);
        assert!(rate > 0.0);
    }
}
