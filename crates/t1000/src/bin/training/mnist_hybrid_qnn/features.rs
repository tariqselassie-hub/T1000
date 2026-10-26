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
            let stem = candidate
                .trim_end_matches(".idx3-ubyte")
                .trim_end_matches(".idx1-ubyte");
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
    let count = max_samples
        .map(|m| m.min(total_images))
        .unwrap_or(total_images);
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
    let count = max_samples
        .map(|m| m.min(total_labels))
        .unwrap_or(total_labels);
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

pub fn encode_mnist_state(pixels: &[f64], spatial: &[f64; N_SPATIAL]) -> QuantumState {
    let mut state = QuantumState::zero(STATE_DIM);
    let mut buffer = [0.0; STATE_DIM];

    // 1. Amplitudes 0..784: Raw normalized pixels
    buffer[..N_PIXELS].copy_from_slice(&pixels[..N_PIXELS]);

    // 2. Amplitudes 784..849 (65 values): 7x7 cell averages + 4-quadrant directional strokes
    buffer[N_PIXELS..N_PIXELS + N_SPATIAL].copy_from_slice(spatial);

    // 3. Amplitudes 849..877 (28 values): Horizontal row projection profile
    let off_h = N_PIXELS + N_SPATIAL;
    for r in 0..28 {
        let mut row_sum = 0.0;
        for c in 0..28 {
            row_sum += pixels[r * 28 + c];
        }
        buffer[off_h + r] = row_sum / 28.0;
    }

    // 4. Amplitudes 877..905 (28 values): Vertical column projection profile
    let off_v = off_h + 28;
    for c in 0..28 {
        let mut col_sum = 0.0;
        for r in 0..28 {
            col_sum += pixels[r * 28 + c];
        }
        buffer[off_v + c] = col_sum / 28.0;
    }

    // 5. Amplitudes 905..969 (64 values): 8x8 coarse pooling
    let off_8x8 = off_v + 28;
    for r8 in 0..8 {
        for c8 in 0..8 {
            let mut sum8 = 0.0;
            let mut count8 = 0.0;
            for dr in 0..3 {
                let r = (r8 * 7 + dr * 2) / 2;
                if r >= 28 {
                    continue;
                }
                for dc in 0..3 {
                    let c = (c8 * 7 + dc * 2) / 2;
                    if c >= 28 {
                        continue;
                    }
                    sum8 += pixels[r * 28 + c];
                    count8 += 1.0;
                }
            }
            buffer[off_8x8 + r8 * 8 + c8] = if count8 > 0.0 { sum8 / count8 } else { 0.0 };
        }
    }

    // 6. Amplitudes 969..1024 (55 values): Center-of-mass radial distance and boundary moments
    let off_rad = off_8x8 + 64;
    let mut total_mass = 0.0;
    let mut com_r = 0.0;
    let mut com_c = 0.0;
    for r in 0..28 {
        for c in 0..28 {
            let p = pixels[r * 28 + c];
            total_mass += p;
            com_r += (r as f64) * p;
            com_c += (c as f64) * p;
        }
    }
    if total_mass > 1e-6 {
        com_r /= total_mass;
        com_c /= total_mass;
    } else {
        com_r = 14.0;
        com_c = 14.0;
    }

    let n_radial = STATE_DIM - off_rad; // 1024 - 969 = 55 values
    for i in 0..n_radial {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n_radial as f64);
        let sample_r = ((com_r + 8.0 * angle.sin()).round() as isize).clamp(0, 27) as usize;
        let sample_c = ((com_c + 8.0 * angle.cos()).round() as isize).clamp(0, 27) as usize;
        buffer[off_rad + i] = pixels[sample_r * 28 + sample_c];
    }

    // Exact L2 normalization into 10-qubit QuantumState
    let mut norm_sq = 0.0;
    for &val in &buffer {
        norm_sq += val * val;
    }
    if norm_sq > 1e-12 {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for i in 0..STATE_DIM {
            state.real[i] = buffer[i] * inv_norm;
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
