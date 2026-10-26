// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Affine readout calibration for three-property regression heads.
//!
//! A circuit readout `<Z>` lives in `[-1, 1]` and the head `mu + sigma * <Z>` assumes a fixed
//! scale and offset. [`calibrate`] instead fits `y ~ slope * <Z> + intercept` per property by
//! least squares on the **training** set and scores it on the held-out set, so the held-out
//! numbers stay honest.

/// Least-squares line `y = slope * x + intercept`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineFit {
    pub slope: f64,
    pub intercept: f64,
}

impl AffineFit {
    /// Ordinary least squares. A constant `x` gives slope 0 and the mean of `y`.
    pub fn fit(x: &[f64], y: &[f64]) -> Self {
        let n = x.len() as f64;
        let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
        let (mut sxy, mut sxx) = (0.0, 0.0);
        for (&a, &b) in x.iter().zip(y) {
            sxy += (a - mx) * (b - my);
            sxx += (a - mx) * (a - mx);
        }
        if sxx < 1e-300 {
            return Self {
                slope: 0.0,
                intercept: my,
            };
        }
        let slope = sxy / sxx;
        Self {
            slope,
            intercept: my - slope * mx,
        }
    }

    pub fn apply(&self, x: f64) -> f64 {
        self.slope * x + self.intercept
    }
}

/// Held-out result of calibrating each of three properties.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Calibrated {
    pub fits: [AffineFit; 3],
    pub test_mse: [f64; 3],
    pub test_mae: [f64; 3],
    /// `1 - mse / variance(test targets)`; 0 means no better than predicting the test mean.
    pub test_r2: [f64; 3],
}

/// Pearson correlation of `preds` with `targets`; 0 for fewer than two points or no variance.
pub fn pearson_r(preds: &[f64], targets: &[f64]) -> f64 {
    let n = preds.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean_p = preds.iter().sum::<f64>() / n;
    let mean_t = targets.iter().sum::<f64>() / n;
    let (mut cov, mut var_p, mut var_t) = (0.0, 0.0, 0.0);
    for (&p, &t) in preds.iter().zip(targets) {
        let (dp, dt) = (p - mean_p, t - mean_t);
        cov += dp * dt;
        var_p += dp * dp;
        var_t += dt * dt;
    }
    let denom = (var_p * var_t).sqrt();
    if denom > 1e-12 {
        cov / denom
    } else {
        0.0
    }
}

/// One least-squares line per property from readouts `z` and targets `y`.
pub fn fit_columns(z: &[[f64; 3]], y: &[[f64; 3]]) -> [AffineFit; 3] {
    let column = |rows: &[[f64; 3]], k: usize| rows.iter().map(|r| r[k]).collect::<Vec<f64>>();
    std::array::from_fn(|k| AffineFit::fit(&column(z, k), &column(y, k)))
}

/// Fits one line per property on `(train_z, train_y)` and scores it on `(test_z, test_y)`.
pub fn calibrate(
    train_z: &[[f64; 3]],
    train_y: &[[f64; 3]],
    test_z: &[[f64; 3]],
    test_y: &[[f64; 3]],
) -> Calibrated {
    let column = |rows: &[[f64; 3]], k: usize| rows.iter().map(|r| r[k]).collect::<Vec<f64>>();
    let mut out = Calibrated {
        fits: [AffineFit {
            slope: 0.0,
            intercept: 0.0,
        }; 3],
        test_mse: [0.0; 3],
        test_mae: [0.0; 3],
        test_r2: [0.0; 3],
    };
    let fits = fit_columns(train_z, train_y);
    for (k, &fit) in fits.iter().enumerate() {
        let (z, y) = (column(test_z, k), column(test_y, k));
        let n = y.len() as f64;
        let mean = y.iter().sum::<f64>() / n;
        let variance = y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
        let errors: Vec<f64> = z.iter().zip(&y).map(|(&a, &b)| fit.apply(a) - b).collect();
        out.fits[k] = fit;
        out.test_mse[k] = errors.iter().map(|e| e * e).sum::<f64>() / n;
        out.test_mae[k] = errors.iter().map(|e| e.abs()).sum::<f64>() / n;
        out.test_r2[k] = 1.0 - out.test_mse[k] / variance;
    }
    out
}

impl Calibrated {
    /// Prints the raw-versus-calibrated held-out table for logP, QED and SAS.
    pub fn print_table(&self, raw_mse: [f64; 3], train_molecules: usize) {
        println!(
            "\n  AFFINE READOUT CALIBRATION (fit on {train_molecules} training molecules, scored on held-out)"
        );
        println!("  property     raw MSE   calibrated MSE   calibrated MAE   calibrated R2   slope  intercept");
        for (k, name) in ["logP", "qed ", "SAS "].iter().enumerate() {
            println!(
                "  {name}   {:>10.4}   {:>14.4}   {:>14.4}   {:>13.4}   {:>5.3}  {:>9.4}",
                raw_mse[k],
                self.test_mse[k],
                self.test_mae[k],
                self.test_r2[k],
                self.fits[k].slope,
                self.fits[k].intercept
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_an_exact_line_and_handles_constant_input() {
        let x = [-1.0, -0.5, 0.0, 0.7, 1.0];
        let y: Vec<f64> = x.iter().map(|v| 3.0 * v + 0.25).collect();
        let fit = AffineFit::fit(&x, &y);
        assert!((fit.slope - 3.0).abs() < 1e-12 && (fit.intercept - 0.25).abs() < 1e-12);
        let flat = AffineFit::fit(&[0.3, 0.3, 0.3], &[1.0, 2.0, 3.0]);
        assert_eq!((flat.slope, flat.intercept), (0.0, 2.0));
    }

    #[test]
    fn pearson_matches_known_values() {
        let x = [1.0, 2.0, 3.0, 4.0];
        assert!((pearson_r(&x, &[2.0, 4.0, 6.0, 8.0]) - 1.0).abs() < 1e-12);
        assert!((pearson_r(&x, &[8.0, 6.0, 4.0, 2.0]) + 1.0).abs() < 1e-12);
        assert_eq!(pearson_r(&x, &[5.0; 4]), 0.0);
        assert_eq!(pearson_r(&[1.0], &[1.0]), 0.0);
    }

    #[test]
    fn scores_on_held_out_data_not_on_the_fit_data() {
        // Train: y = 2z. Test: y = 2z + 1, so the train-fitted line misses by exactly 1.
        let train_z: Vec<[f64; 3]> = (0..10).map(|i| [i as f64 * 0.1; 3]).collect();
        let train_y: Vec<[f64; 3]> = train_z.iter().map(|z| [2.0 * z[0]; 3]).collect();
        let test_z = train_z.clone();
        let test_y: Vec<[f64; 3]> = train_y.iter().map(|y| [y[0] + 1.0; 3]).collect();
        let c = calibrate(&train_z, &train_y, &test_z, &test_y);
        assert!((c.test_mse[0] - 1.0).abs() < 1e-12);
        assert!((c.test_mae[0] - 1.0).abs() < 1e-12);
    }
}
