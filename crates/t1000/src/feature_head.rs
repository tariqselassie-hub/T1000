// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! A fixed-circuit feature map with a closed-form ridge head: feature extraction from scaled
//! descriptors, fitting on the training molecules, and validation scoring.

use crate::baselines::regression_metrics;
use crate::datasets::Sample;
use crate::linear_head::{CircuitFeatureMap, LinearHead};
use crate::reuploading::ReuploadingMap;
use crate::zinc_data::{InputScaling, Scaler};
use std::time::Instant;

/// Circuit feature extraction, with the descriptors optionally appended (`--append-raw`).
#[derive(Debug)]
pub struct FeatureExtractor<'a> {
    pub map: CircuitFeatureMap<'a>,
    pub reupload: ReuploadingMap<'a>,
    pub use_reupload: bool,
    pub append_raw: bool,
}

impl FeatureExtractor<'_> {
    /// Readout features for max-abs / min-max scaled descriptor rows.
    pub fn matrix(&self, scaled: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let rows = scaled.iter().map(Vec::as_slice);
        let mut features = if self.use_reupload {
            self.reupload.feature_matrix(rows)
        } else {
            self.map.feature_matrix(rows)
        };
        if self.append_raw {
            features
                .iter_mut()
                .zip(scaled)
                .for_each(|(r, x)| r.extend_from_slice(x));
        }
        features
    }
}

/// Scaler and ridge head fitted on the training molecules.
#[derive(Debug)]
pub struct FittedHead {
    pub scaler: Scaler,
    pub head: LinearHead,
    pub train_x: Vec<Vec<f64>>,
    pub scaled_test: Vec<Vec<f64>>,
    pub train_ms: f64,
    pub fit_ms: f64,
}

pub fn targets_of(rows: &[Sample]) -> Vec<Vec<f64>> {
    rows.iter().map(|m| m.targets.to_vec()).collect()
}

/// Scales the descriptors, extracts circuit features and fits the ridge head.
pub fn fit_head(
    scaling: InputScaling,
    features: &FeatureExtractor,
    train: &[Sample],
    test: &[Sample],
) -> FittedHead {
    let t = Instant::now();
    let rows: Vec<&[f64]> = train.iter().map(|m| m.features.as_slice()).collect();
    let scaler = Scaler::fit_rows(scaling, &rows);
    let scaled_train: Vec<Vec<f64>> = train.iter().map(|m| scaler.apply(&m.features)).collect();
    let scaled_test: Vec<Vec<f64>> = test.iter().map(|m| scaler.apply(&m.features)).collect();
    let train_x = features.matrix(&scaled_train);
    let train_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let head = LinearHead::fit(&train_x, &targets_of(train));
    let fit_ms = t.elapsed().as_secs_f64() * 1e3;
    FittedHead {
        scaler,
        head,
        train_x,
        scaled_test,
        train_ms,
        fit_ms,
    }
}

/// Relative MSE on the validation molecules (1 - mean R2), if there are any.
/// Hyperparameters are chosen on these molecules, never on the test set.
pub fn validation_mse(
    fit: &FittedHead,
    features: &FeatureExtractor,
    validation: &[Sample],
    names: &[&str],
) -> Option<f64> {
    if validation.is_empty() {
        return None;
    }
    let scaled: Vec<Vec<f64>> = validation
        .iter()
        .map(|m| fit.scaler.apply(&m.features))
        .collect();
    let preds = fit.head.predict(&features.matrix(&scaled));
    let m = regression_metrics(&preds, &targets_of(validation), names);
    Some(1.0 - m.iter().map(|t| t.r2).sum::<f64>() / m.len() as f64)
}
