// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Held-out scores that decide whether a recorded run replaces the saved one.
//!
//! The ZINC trainers predict logP, QED and SAS, which live on different scales, so the deciding
//! metric is held-out MSE divided by the variance of the held-out targets (1.0 means no better
//! than predicting the mean), averaged over the three properties. Mean Pearson r is the guard.

use crate::results::{Better, Guard, Score};

/// Largest drop in mean Pearson r that still lets a lower relative MSE replace a saved run.
pub const PEARSON_TOLERANCE: f64 = 0.02;

/// Aggregated held-out quality of a three-property regression run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZincHoldout {
    /// Mean over properties of `tuned_mse / variance(targets)`; lower is better.
    pub relative_mse: f64,
    /// Mean Pearson r over properties; higher is better.
    pub mean_pearson_r: f64,
}

fn variance(t: &[f64]) -> f64 {
    let mean = t.iter().sum::<f64>() / t.len() as f64;
    t.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / t.len() as f64
}

impl ZincHoldout {
    /// Builds the aggregate from per-property held-out MSE, Pearson r and the held-out targets.
    pub fn new(tuned_mse: [f64; 3], pearson_r: [f64; 3], targets: [&[f64]; 3]) -> Self {
        let relative: f64 = tuned_mse.iter().zip(targets).map(|(m, t)| m / variance(t)).sum();
        Self { relative_mse: relative / 3.0, mean_pearson_r: pearson_r.iter().sum::<f64>() / 3.0 }
    }

    /// The recording score: relative MSE decides, mean Pearson r guards.
    pub fn score<'a>(&self, settings: &'a str) -> Score<'a> {
        Score {
            metric: "holdout_relative_mse",
            value: self.relative_mse,
            better: Better::Lower,
            guard: Some(Guard {
                name: "holdout_mean_pearson_r",
                value: self.mean_pearson_r,
                better: Better::Higher,
                tolerance: PEARSON_TOLERANCE,
            }),
            settings,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_predictor_scores_one_and_perfect_scores_zero() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [0.0, 0.5, 1.0, 0.5];
        let var = |t: &[f64]| variance(t);
        let at_mean = ZincHoldout::new([var(&a), var(&b), var(&a)], [0.0; 3], [&a, &b, &a]);
        assert!((at_mean.relative_mse - 1.0).abs() < 1e-12);
        let perfect = ZincHoldout::new([0.0; 3], [1.0; 3], [&a, &b, &a]);
        assert_eq!(perfect.relative_mse, 0.0);
        assert_eq!(perfect.mean_pearson_r, 1.0);
    }
}
