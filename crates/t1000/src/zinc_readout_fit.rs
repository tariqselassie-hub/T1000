// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Closed-form multi-target ridge regression for the ZINC readout heads, and the per-property
//! error metrics (MAE, R^2) used to score them.

/// Ridge regression `W = (XᵀX + λI)⁻¹ XᵀY` on centred data.
///
/// `x` is `(n_samples, n_features)` and `y` is `(n_samples, n_targets)`. Returns the weights
/// `(n_features, n_targets)` and the bias per target.
#[allow(clippy::needless_range_loop)] // index form mirrors the linear algebra
pub fn ridge_regression_multi(
    x: &[Vec<f64>],
    y: &[Vec<f64>],
    n_features: usize,
    n_targets: usize,
    lambda: f64,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = x.len();

    // Center the targets
    let mut y_mean = vec![0.0; n_targets];
    let mut x_mean = vec![0.0; n_features];
    for i in 0..n {
        for t in 0..n_targets {
            y_mean[t] += y[i][t];
        }
        for f in 0..n_features {
            x_mean[f] += x[i][f];
        }
    }
    let inv_n = 1.0 / n as f64;
    for t in 0..n_targets {
        y_mean[t] *= inv_n;
    }
    for f in 0..n_features {
        x_mean[f] *= inv_n;
    }

    // Compute X^T X (n_features x n_features) + ridge term
    let mut xtx = vec![vec![0.0; n_features]; n_features];
    for i in 0..n {
        for f1 in 0..n_features {
            let v1 = x[i][f1] - x_mean[f1];
            for f2 in f1..n_features {
                let v2 = x[i][f2] - x_mean[f2];
                xtx[f1][f2] += v1 * v2;
            }
        }
    }
    // Symmetrize and add ridge
    for f1 in 0..n_features {
        for f2 in (f1 + 1)..n_features {
            xtx[f2][f1] = xtx[f1][f2];
        }
        xtx[f1][f1] += lambda;
    }

    // Compute X^T Y (n_features x n_targets)
    let mut xty = vec![vec![0.0; n_targets]; n_features];
    for i in 0..n {
        for f in 0..n_features {
            let v = x[i][f] - x_mean[f];
            for t in 0..n_targets {
                xty[f][t] += v * (y[i][t] - y_mean[t]);
            }
        }
    }

    // Solve via Cholesky decomposition
    let w = cholesky_solve_multi(&xtx, &xty, n_features, n_targets);

    // Compute bias: b = y_mean - Wᵀ x_mean
    let mut bias = y_mean.clone();
    for t in 0..n_targets {
        for f in 0..n_features {
            bias[t] -= w[f][t] * x_mean[f];
        }
    }

    (w, bias)
}

/// Cholesky solve `A X = B` for symmetric positive definite `A` (`n x n`) and `B` (`n x m`).
/// A non-positive pivot is replaced by 1e-10 so a near-singular system still returns.
#[allow(clippy::needless_range_loop)] // index form mirrors the linear algebra
fn cholesky_solve_multi(a: &[Vec<f64>], b: &[Vec<f64>], n: usize, m: usize) -> Vec<Vec<f64>> {
    // L * L^T = A (Cholesky decomposition)
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }
            if i == j {
                let diag = a[i][i] - sum;
                l[i][j] = if diag > 0.0 { diag.sqrt() } else { 1e-10 };
            } else {
                l[i][j] = (a[i][j] - sum) / l[j][j];
            }
        }
    }

    // Forward substitution: L * Z = B
    let mut z = vec![vec![0.0; m]; n];
    for i in 0..n {
        for t in 0..m {
            let mut sum = b[i][t];
            for k in 0..i {
                sum -= l[i][k] * z[k][t];
            }
            z[i][t] = sum / l[i][i];
        }
    }

    // Backward substitution: L^T * X = Z
    let mut x = vec![vec![0.0; m]; n];
    for i in (0..n).rev() {
        for t in 0..m {
            let mut sum = z[i][t];
            for k in (i + 1)..n {
                sum -= l[k][i] * x[k][t];
            }
            x[i][t] = sum / l[i][i];
        }
    }

    x
}

/// MAE and R^2 for each of (logP, QED, SAS). R^2 is 0 for a constant target.
#[allow(clippy::needless_range_loop)] // index form mirrors the per-property sums
pub fn mae_r2_per_property(preds: &[[f64; 3]], truth: &[(f64, f64, f64)]) -> ([f64; 3], [f64; 3]) {
    let n = preds.len();
    let mut mae = [0.0; 3];
    let mut ss_res = [0.0; 3];
    let mut mean_y = [0.0; 3];

    for i in 0..n {
        let t = [truth[i].0, truth[i].1, truth[i].2];
        for p in 0..3 {
            mae[p] += (preds[i][p] - t[p]).abs();
            mean_y[p] += t[p];
        }
    }
    for p in 0..3 {
        mae[p] /= n as f64;
        mean_y[p] /= n as f64;
    }

    let mut ss_tot = [0.0; 3];
    for i in 0..n {
        let t = [truth[i].0, truth[i].1, truth[i].2];
        for p in 0..3 {
            ss_res[p] += (preds[i][p] - t[p]).powi(2);
            ss_tot[p] += (t[p] - mean_y[p]).powi(2);
        }
    }

    let mut r2 = [0.0; 3];
    for p in 0..3 {
        r2[p] = if ss_tot[p] > 1e-12 {
            1.0 - ss_res[p] / ss_tot[p]
        } else {
            0.0
        };
    }

    (mae, r2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ridge_recovers_an_exact_multi_target_linear_map() {
        // y0 = 2 x0 - x1 + 0.5, y1 = -x0 + 3, y2 = 0.25 x1
        let xs: Vec<Vec<f64>> = (0..30)
            .map(|i| vec![f64::from(i) * 0.1, f64::from(i % 7)])
            .collect();
        let ys: Vec<Vec<f64>> = xs
            .iter()
            .map(|x| vec![2.0 * x[0] - x[1] + 0.5, -x[0] + 3.0, 0.25 * x[1]])
            .collect();
        let (w, b) = ridge_regression_multi(&xs, &ys, 2, 3, 1e-12);
        let expected_w = [[2.0, -1.0, 0.0], [-1.0, 0.0, 0.25]];
        for f in 0..2 {
            for t in 0..3 {
                assert!(
                    (w[f][t] - expected_w[f][t]).abs() < 1e-6,
                    "w[{f}][{t}] = {}",
                    w[f][t]
                );
            }
        }
        assert!((b[0] - 0.5).abs() < 1e-6 && (b[1] - 3.0).abs() < 1e-6 && b[2].abs() < 1e-6);
    }

    #[test]
    fn a_singular_system_still_returns_finite_weights() {
        // Duplicate columns make XᵀX singular; the ridge term keeps it solvable.
        let xs: Vec<Vec<f64>> = (0..10).map(|i| vec![f64::from(i), f64::from(i)]).collect();
        let ys: Vec<Vec<f64>> = (0..10).map(|i| vec![f64::from(i)]).collect();
        let (w, b) = ridge_regression_multi(&xs, &ys, 2, 1, 1e-3);
        assert!(w.iter().flatten().chain(&b).all(|v| v.is_finite()));
        assert!(
            ((w[0][0] + w[1][0]) - 1.0).abs() < 1e-2,
            "the pair together must carry the slope"
        );
    }

    #[test]
    fn perfect_predictions_score_zero_error_and_unit_r2() {
        let truth: Vec<(f64, f64, f64)> = (0..5)
            .map(|i| (f64::from(i), f64::from(i) * 0.5, 2.0 - f64::from(i)))
            .collect();
        let preds: Vec<[f64; 3]> = truth.iter().map(|&(a, b, c)| [a, b, c]).collect();
        let (mae, r2) = mae_r2_per_property(&preds, &truth);
        assert_eq!(mae, [0.0; 3]);
        assert_eq!(r2, [1.0; 3]);
    }

    #[test]
    fn constant_targets_give_zero_r2_and_the_plain_mae() {
        let truth = vec![(1.0, 2.0, 3.0); 4];
        let preds = vec![[1.5, 2.0, 2.0]; 4];
        let (mae, r2) = mae_r2_per_property(&preds, &truth);
        assert_eq!(mae, [0.5, 0.0, 1.0]);
        assert_eq!(r2, [0.0; 3]);
    }
}
