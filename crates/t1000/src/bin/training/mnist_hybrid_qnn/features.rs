// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! 2D Multi-Scale Spatial Feature Extractor & IDX Dataset Ingestion.

use std::fs;
use std::path::{Path, PathBuf};
use t1000::QuantumState;

pub const N_PIXELS: usize = 784;
pub const N_SPATIAL: usize = 65; // 49 (7x7 cell pooling) + 16 (4-quadrant directional strokes)
const STATE_DIM: usize = 1024;

pub fn locate_mnist_file(filename_candidates: &[&str]) -> Option<PathBuf> {
    let candidate_dirs = [
        "data/mnist",
        "../data/mnist",
        "data/mnist_demo",
        "../data/mnist_demo",
        "data",
        "../data",
    ];
    for dir in &candidate_dirs {
        for candidate in filename_candidates {
            let p1 = Path::new(dir).join(candidate);
            if p1.is_file() {
                return Some(p1);
            }
            let stem = candidate.trim_end_matches(".idx3-ubyte").trim_end_matches(".idx1-ubyte");
            let p2 = Path::new(dir).join(stem).join(candidate);
            if p2.is_file() {
                return Some(p2);
            }
        }
    }
    None
}

pub fn parse_mnist_images(path: &Path, max_samples: Option<usize>) -> Vec<Vec<f64>> {
    let data = fs::read(path).expect("Failed to read MNIST images");
    assert!(data.len() >= 16, "IDX image header too small");
    let total_images = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let count = max_samples.map(|m| m.min(total_images)).unwrap_or(total_images);
    let mut images = Vec::with_capacity(count);
    let offset = 16;
    for i in 0..count {
        let start = offset + i * N_PIXELS;
        let img: Vec<f64> = data[start..start + N_PIXELS]
            .iter()
            .map(|&p| f64::from(p) / 255.0)
            .collect();
        images.push(img);
    }
    images
}

pub fn parse_mnist_labels(path: &Path, max_samples: Option<usize>) -> Vec<u8> {
    let data = fs::read(path).expect("Failed to read MNIST labels");
    assert!(data.len() >= 8, "IDX label header too small");
    let total_labels = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let count = max_samples.map(|m| m.min(total_labels)).unwrap_or(total_labels);
    data[8..8 + count].to_vec()
}

/// Extract 65 2D Multi-Scale Spatial Features:
/// 49 from 7x7 cell density average-pooling, and 16 from 4-quadrant directional stroke filters.
pub fn extract_spatial_features(pixels: &[f64]) -> [f64; N_SPATIAL] {
    let mut features = [0.0; N_SPATIAL];
    // 1. 7x7 cell average pooling
    for r in 0..7 {
        for c in 0..7 {
            let mut sum = 0.0;
            for dr in 0..4 {
                for dc in 0..4 {
                    let idx = (r * 4 + dr) * 28 + (c * 4 + dc);
                    sum += pixels[idx];
                }
            }
            features[r * 7 + c] = sum / 16.0;
        }
    }

    // 2. 4 directional stroke filters across 4 quadrants
    let mut quad_strokes = [[0.0f64; 4]; 4];
    let mut quad_counts = [0.0f64; 4];

    for r in 1..27 {
        let quad_r = if r < 14 { 0 } else { 1 };
        for c in 1..27 {
            let quad_c = if c < 14 { 0 } else { 1 };
            let q_idx = quad_r * 2 + quad_c;

            let h_diff = (pixels[r * 28 + (c + 1)] - pixels[r * 28 + (c - 1)]).abs();
            let v_diff = (pixels[(r + 1) * 28 + c] - pixels[(r - 1) * 28 + c]).abs();
            let d1_diff = (pixels[(r + 1) * 28 + (c + 1)] - pixels[(r - 1) * 28 + (c - 1)]).abs();
            let d2_diff = (pixels[(r + 1) * 28 + (c - 1)] - pixels[(r - 1) * 28 + (c + 1)]).abs();

            quad_strokes[q_idx][0] += h_diff;
            quad_strokes[q_idx][1] += v_diff;
            quad_strokes[q_idx][2] += d1_diff;
            quad_strokes[q_idx][3] += d2_diff;
            quad_counts[q_idx] += 1.0;
        }
    }

    let mut out_idx = 49;
    for q in 0..4 {
        let count = quad_counts[q].max(1.0f64);
        for &stroke in &quad_strokes[q] {
            features[out_idx] = stroke / count;
            out_idx += 1;
        }
    }

    features
}

pub fn encode_mnist_state(pixels: &[f64]) -> QuantumState {
    let mut state = QuantumState::zero(STATE_DIM);
    let mut norm_sq = 0.0;
    for &p in pixels {
        norm_sq += p * p;
    }
    if norm_sq > 1e-12 {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for (i, &p) in pixels.iter().enumerate() {
            state.real[i] = p * inv_norm;
            state.imag[i] = 0.0;
        }
    } else {
        let uniform = 1.0 / (STATE_DIM as f64).sqrt();
        for i in 0..STATE_DIM {
            state.real[i] = uniform;
            state.imag[i] = 0.0;
        }
    }
    state
}
