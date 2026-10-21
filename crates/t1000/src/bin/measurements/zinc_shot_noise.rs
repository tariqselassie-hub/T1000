// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! How many measurement shots does the circuit + linear-head ZINC model need?
//!
//! Each molecule's circuit output is measured `N` times in the computational basis (sampled from
//! the exact output probabilities), the 55 Z and ZZ values are rebuilt from the sampled bitstrings,
//! and the head predicts from those noisy features. Two deployments are compared:
//!
//! * **exact-trained head**: fitted on noise-free simulated features, applied to sampled ones;
//! * **noisy-trained head**: fitted on features sampled with the same shot count.
//!
//! Usage: `zinc_shot_noise [--scale raw|minmax|maxabs] [--depth N] [--seeds K] [--noisy-train M]`

use serde::Serialize;
use std::env;
use t1000::baselines::{regression_metrics, ZINC_TARGETS};
use t1000::linear_head::{z_features_into, LinearHead, Observables};
use t1000::zinc_data::{self, InputScaling, Scaler, ZincMolecule};
use t1000::{CircuitAnsatz, DataEncodingLayer, EncodingKind, QuantumState};

const N_QUBITS: usize = 10;
const SHOTS: [usize; 8] = [30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000];
const TRAIN_CSV: &str = "tests/benchmarks/zinc_molecular/outputs/zinc_train_10k.csv";
const TEST_CSV: &str = "tests/benchmarks/zinc_molecular/outputs/zinc_test_2k.csv";

/// SplitMix64: small, fast and deterministic across platforms.
struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Output probabilities of the circuit for one input.
fn probabilities(state: &QuantumState) -> Vec<f64> {
    state.real.iter().zip(&state.imag).map(|(r, i)| r * r + i * i).collect()
}

/// Z/ZZ features estimated from `shots` samples of the distribution `cdf`.
fn sampled_features(cdf: &[f64], shots: usize, rng: &mut Rng, scratch: &mut QuantumState, counts: &mut [u32]) -> Vec<f64> {
    counts.fill(0);
    for _ in 0..shots {
        let u = rng.next_f64() * cdf[cdf.len() - 1];
        counts[cdf.partition_point(|&c| c <= u).min(cdf.len() - 1)] += 1;
    }
    // z_features_into uses |amplitude|^2, so amplitude sqrt(frequency) gives the empirical mean.
    for (r, &c) in scratch.real.iter_mut().zip(counts.iter()) {
        *r = (f64::from(c) / shots as f64).sqrt();
    }
    scratch.imag.fill(0.0);
    let mut row = vec![0.0; Observables::ZAndZz.count(N_QUBITS)];
    z_features_into(scratch, Observables::ZAndZz, &mut row);
    row
}

fn cumulative(p: &[f64]) -> Vec<f64> {
    p.iter().scan(0.0, |acc, v| { *acc += v; Some(*acc) }).collect()
}

fn score(head: &LinearHead, features: &[Vec<f64>], targets: &[Vec<f64>]) -> (f64, f64) {
    let m = regression_metrics(&head.predict(features), targets, &ZINC_TARGETS);
    (1.0 - m.iter().map(|t| t.r2).sum::<f64>() / 3.0, m.iter().map(|t| t.pearson_r).sum::<f64>() / 3.0)
}

fn mean_sd(v: &[f64]) -> (f64, f64) {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    (m, (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt())
}

#[derive(Serialize)]
struct Row {
    shots: usize,
    exact_head_relative_mse: f64,
    exact_head_sd: f64,
    exact_head_mean_r: f64,
    noisy_head_relative_mse: f64,
    noisy_head_sd: f64,
    noisy_head_mean_r: f64,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let value = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned();
    let scaling = match value("--scale").as_deref() {
        Some("raw") => InputScaling::Raw,
        Some("minmax") => InputScaling::MinMax,
        _ => InputScaling::MaxAbs,
    };
    let depth: usize = value("--depth").and_then(|v| v.parse().ok()).unwrap_or(4);
    let seeds: usize = value("--seeds").and_then(|v| v.parse().ok()).unwrap_or(3);
    let noisy_train: usize = value("--noisy-train").and_then(|v| v.parse().ok()).unwrap_or(2_000);

    let train = zinc_data::load(TRAIN_CSV, 10_000).expect("load training set");
    let test = zinc_data::load(TEST_CSV, 2_000).expect("load held-out set");
    let scaler = Scaler::fit(scaling, &train);
    let encoding = DataEncodingLayer::new(N_QUBITS, EncodingKind::Amplitude);
    let ansatz = CircuitAnsatz::new(N_QUBITS, depth).with_pi_layer(true);

    println!("Simulating the circuit (depth {depth}, {scaling:?} inputs) for {} + {} molecules...", train.len(), test.len());
    let mut state = QuantumState::new(encoding.dim);
    let mut outputs = |rows: &[ZincMolecule]| -> Vec<Vec<f64>> {
        rows.iter()
            .map(|m| {
                ansatz.forward_into(&encoding.encode(&scaler.apply(&m.features)), &ansatz.weights, &mut state);
                probabilities(&state)
            })
            .collect()
    };
    let train_p = outputs(&train);
    let test_p = outputs(&test);
    let exact = |p: &[Vec<f64>]| -> Vec<Vec<f64>> {
        let mut s = QuantumState::new(encoding.dim);
        p.iter()
            .map(|probs| {
                for (r, &v) in s.real.iter_mut().zip(probs) { *r = v.sqrt(); }
                s.imag.fill(0.0);
                let mut row = vec![0.0; Observables::ZAndZz.count(N_QUBITS)];
                z_features_into(&s, Observables::ZAndZz, &mut row);
                row
            })
            .collect()
    };
    let train_y: Vec<Vec<f64>> = train.iter().map(|m| m.targets().to_vec()).collect();
    let test_y: Vec<Vec<f64>> = test.iter().map(|m| m.targets().to_vec()).collect();
    let (train_exact, test_exact) = (exact(&train_p), exact(&test_p));
    let exact_head = LinearHead::fit(&train_exact, &train_y);
    let (ref_mse, ref_r) = score(&exact_head, &test_exact, &test_y);
    println!("Noise-free reference: relative MSE {ref_mse:.4}, mean r {ref_r:.4}\n");

    let test_cdf: Vec<Vec<f64>> = test_p.iter().map(|p| cumulative(p)).collect();
    let train_cdf: Vec<Vec<f64>> = train_p.iter().take(noisy_train).map(|p| cumulative(p)).collect();
    let mut scratch = QuantumState::new(encoding.dim);
    let mut counts = vec![0u32; encoding.dim];
    let mut rows = Vec::new();
    println!("┌─────────┬───────────────────────────────┬───────────────────────────────┐");
    println!("│  shots  │ exact-trained head            │ noisy-trained head            │");
    println!("│         │ rel. MSE (sd)       mean r    │ rel. MSE (sd)       mean r    │");
    println!("├─────────┼───────────────────────────────┼───────────────────────────────┤");
    for &shots in &SHOTS {
        let (mut a_mse, mut a_r, mut b_mse, mut b_r) = (vec![], vec![], vec![], vec![]);
        for seed in 0..seeds {
            let mut rng = Rng(0xC0FF_EE00 ^ (shots as u64) << 8 ^ seed as u64);
            let noisy_test: Vec<Vec<f64>> = test_cdf.iter().map(|c| sampled_features(c, shots, &mut rng, &mut scratch, &mut counts)).collect();
            let (m, r) = score(&exact_head, &noisy_test, &test_y);
            a_mse.push(m);
            a_r.push(r);
            let noisy_train_x: Vec<Vec<f64>> = train_cdf.iter().map(|c| sampled_features(c, shots, &mut rng, &mut scratch, &mut counts)).collect();
            let head = LinearHead::fit(&noisy_train_x, &train_y[..noisy_train_x.len()]);
            let (m, r) = score(&head, &noisy_test, &test_y);
            b_mse.push(m);
            b_r.push(r);
        }
        let ((am, asd), (bm, bsd)) = (mean_sd(&a_mse), mean_sd(&b_mse));
        let (ar, br) = (mean_sd(&a_r).0, mean_sd(&b_r).0);
        println!("│ {shots:>7} │ {am:>8.4} ({asd:.4})   {ar:>8.4}   │ {bm:>8.4} ({bsd:.4})   {br:>8.4}   │");
        rows.push(Row { shots, exact_head_relative_mse: am, exact_head_sd: asd, exact_head_mean_r: ar, noisy_head_relative_mse: bm, noisy_head_sd: bsd, noisy_head_mean_r: br });
    }
    println!("└─────────┴───────────────────────────────┴───────────────────────────────┘");
    println!("Noise-free: relative MSE {ref_mse:.4}. Mean predictor = 1.0. {seeds} seeds per row; noisy-trained head uses {} training molecules.", train_cdf.len());

    let report = serde_json::json!({ "scaling": format!("{scaling:?}"), "depth": depth, "noise_free_relative_mse": ref_mse, "noise_free_mean_r": ref_r, "seeds": seeds, "noisy_train_molecules": train_cdf.len(), "rows": rows });
    let _ = t1000::results::write("results/experiments/zinc_shot_noise.json", serde_json::to_string_pretty(&report).unwrap_or_default());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_uniform_and_deterministic() {
        let mut a = Rng(7);
        let draws: Vec<f64> = (0..100_000).map(|_| a.next_f64()).collect();
        assert!(draws.iter().all(|u| (0.0..1.0).contains(u)));
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        assert!((mean - 0.5).abs() < 0.005, "mean {mean}");
        let mut b = Rng(7);
        assert_eq!(draws[0], b.next_f64());
    }

    #[test]
    fn sampled_features_converge_to_the_exact_expectations() {
        // A 2-qubit-sized distribution embedded in the 10-qubit register: p(0)=0.7, p(1)=0.3.
        let mut p = vec![0.0; 1 << N_QUBITS];
        p[0] = 0.7;
        p[1] = 0.3;
        let cdf = cumulative(&p);
        let (mut scratch, mut counts) = (QuantumState::new(1 << N_QUBITS), vec![0u32; 1 << N_QUBITS]);
        let row = sampled_features(&cdf, 400_000, &mut Rng(1), &mut scratch, &mut counts);
        // Z0 = +1 on index 0 and -1 on index 1: 0.7 - 0.3 = 0.4; all other Z are +1.
        let sigma = (1.0f64 - 0.16).sqrt() / 400_000f64.sqrt();
        assert!((row[0] - 0.4).abs() < 5.0 * sigma, "Z0 = {}", row[0]);
        assert!((row[1] - 1.0).abs() < 1e-12);
    }
}
