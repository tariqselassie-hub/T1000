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
//! Device errors can be added to the sampled distribution: `--gate-error p2` (global depolarizing,
//! fidelity from the ansatz gate counts with single-qubit error `p2 / 10`), `--readout-error e`
//! (`P(read 0 | 1) = e`, `P(read 1 | 0) = e / 2`) and `--mitigate-readout` (invert the confusion
//! matrix on the measured frequencies).
//!
//! `--prep-rotations R --prep-cnots C` multiplies the fidelity by that of the state-preparation circuit
//! (counts from `zinc_state_prep_cost`).
//!
//! `--sweep` instead compares noise-aware head settings (wider ridge-penalty grid, feature subsets)
//! on identical sampled data.
//!
//! Usage: `zinc_shot_noise [--scale raw|minmax|maxabs] [--depth N] [--seeds K] [--noisy-train M] [--sweep]`

use serde::Serialize;
use std::env;
use t1000::baselines::{regression_metrics, ZINC_TARGETS};
use t1000::baselines::RIDGE_LAMBDAS;
use t1000::device_noise::{ansatz_fidelity, apply_readout, depolarize, invert_readout, ReadoutError};
use t1000::linear_head::{probability_features_into, project, select_features, LinearHead, Observables, NOISE_LAMBDAS};
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

/// How measured frequencies become features.
struct Plan {
    kind: Observables,
    /// If set, the confusion matrix of this readout error is inverted on the frequencies.
    mitigate: Option<ReadoutError>,
}

/// Features estimated from `shots` samples of the distribution `cdf`.
fn sampled_features(cdf: &[f64], shots: usize, rng: &mut Rng, counts: &mut [u32], plan: &Plan) -> Vec<f64> {
    counts.fill(0);
    for _ in 0..shots {
        let u = rng.next_f64() * cdf[cdf.len() - 1];
        counts[cdf.partition_point(|&c| c <= u).min(cdf.len() - 1)] += 1;
    }
    let mut freq: Vec<f64> = counts.iter().map(|&c| f64::from(c) / shots as f64).collect();
    if let Some(e) = plan.mitigate {
        invert_readout(&mut freq, N_QUBITS, e);
    }
    let mut row = vec![0.0; plan.kind.count(N_QUBITS)];
    probability_features_into(&freq, plan.kind, &mut row);
    row
}

/// The distribution a noisy device measures for the ideal output distribution `p`.
fn device_distribution(p: &[f64], fidelity: f64, readout: Option<ReadoutError>) -> Vec<f64> {
    let mut noisy = p.to_vec();
    depolarize(&mut noisy, fidelity);
    if let Some(e) = readout {
        apply_readout(&mut noisy, N_QUBITS, e);
    }
    noisy
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

/// Which of the 55 features a head is allowed to use.
#[derive(Clone, Copy)]
enum Subset {
    All,
    /// The `k` features most correlated with the targets, chosen on the training features.
    Top(usize),
    /// The ten single-qubit Z values.
    SingleZ,
}

struct Config {
    name: &'static str,
    subset: Subset,
    /// Search penalties up to 1e4 instead of 10.
    wide: bool,
}

const CONFIGS: [Config; 6] = [
    Config { name: "all 55, penalty grid to 10", subset: Subset::All, wide: false },
    Config { name: "all 55, penalty grid to 1e4", subset: Subset::All, wide: true },
    Config { name: "top 30, wide grid", subset: Subset::Top(30), wide: true },
    Config { name: "top 20, wide grid", subset: Subset::Top(20), wide: true },
    Config { name: "top 10, wide grid", subset: Subset::Top(10), wide: true },
    Config { name: "10 single-qubit Z, wide grid", subset: Subset::SingleZ, wide: true },
];
const SWEEP_SHOTS: [usize; 5] = [300, 1_000, 3_000, 10_000, 100_000];

/// Fits one configuration on `train_x` and scores it on `test_x`; returns (relative MSE, penalty).
fn run_config(cfg: &Config, train_x: &[Vec<f64>], train_y: &[Vec<f64>], test_x: &[Vec<f64>], test_y: &[Vec<f64>]) -> (f64, f64) {
    let columns: Vec<usize> = match cfg.subset {
        Subset::All => (0..train_x[0].len()).collect(),
        Subset::Top(k) => select_features(train_x, train_y, k),
        Subset::SingleZ => (0..N_QUBITS).collect(),
    };
    let grid: &[f64] = if cfg.wide { &NOISE_LAMBDAS } else { &RIDGE_LAMBDAS };
    let head = LinearHead::fit_with(&project(train_x, &columns), train_y, grid);
    (score(&head, &project(test_x, &columns), test_y).0, head.lambda())
}

fn run_sweep(exact: (&[Vec<f64>], &[Vec<f64>]), y: (&[Vec<f64>], &[Vec<f64>]), cdfs: (&[Vec<f64>], &[Vec<f64>]), seeds: usize, dim: usize, plan: &Plan) {
    let ((train_exact, test_exact), (train_y, test_y), (train_cdf, test_cdf)) = (exact, y, cdfs);
    let mut counts = vec![0u32; dim];
    let free: Vec<f64> = CONFIGS.iter().map(|c| run_config(c, train_exact, train_y, test_exact, test_y).0).collect();
    let mut mse = vec![vec![Vec::new(); SWEEP_SHOTS.len()]; CONFIGS.len()];
    let mut lam = vec![vec![Vec::new(); SWEEP_SHOTS.len()]; CONFIGS.len()];
    for (si, &shots) in SWEEP_SHOTS.iter().enumerate() {
        for seed in 0..seeds {
            let mut rng = Rng(0xDEC0_DE00 ^ (shots as u64) << 8 ^ seed as u64);
            let train_x: Vec<Vec<f64>> = train_cdf.iter().map(|c| sampled_features(c, shots, &mut rng, &mut counts, plan)).collect();
            let test_x: Vec<Vec<f64>> = test_cdf.iter().map(|c| sampled_features(c, shots, &mut rng, &mut counts, plan)).collect();
            for (ci, cfg) in CONFIGS.iter().enumerate() {
                let (m, l) = run_config(cfg, &train_x, &train_y[..train_x.len()], &test_x, test_y);
                mse[ci][si].push(m);
                lam[ci][si].push(l);
            }
        }
        eprintln!("  sampled {shots} shots");
    }
    let header: String = SWEEP_SHOTS.iter().map(|s| format!("{s:>8}")).collect();
    println!("\nRelative MSE (mean of {seeds} seeds; mean predictor = 1.0)");
    println!("{:<32} {:>10}{header}", "configuration", "noise-free");
    for (ci, cfg) in CONFIGS.iter().enumerate() {
        let cells: String = mse[ci].iter().map(|v| format!("{:>8.4}", mean_sd(v).0)).collect();
        println!("{:<32} {:>10.4}{cells}", cfg.name, free[ci]);
    }
    println!("\nRidge penalty chosen on the validation split (largest of the seeds)");
    println!("{:<32} {:>10}{header}", "configuration", "");
    for (ci, cfg) in CONFIGS.iter().enumerate() {
        let cells: String = lam[ci].iter().map(|v| format!("{:>8.0e}", v.iter().cloned().fold(0.0, f64::max))).collect();
        println!("{:<32} {:>10}{cells}", cfg.name, "");
    }
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
    let kind = match value("--observables").as_deref() {
        Some(w) if w.starts_with('w') => Observables::ParityUpTo(w[1..].parse().unwrap_or(2)),
        _ => Observables::ZAndZz,
    };
    let readout = value("--readout-error").and_then(|v| v.parse::<f64>().ok()).map(ReadoutError::typical);
    let p2: f64 = value("--gate-error").and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let prep_rotations: f64 = value("--prep-rotations").and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let prep_cnots: f64 = value("--prep-cnots").and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let prep_fidelity = (1.0 - p2 / 10.0).powf(prep_rotations) * (1.0 - p2).powf(prep_cnots);
    let ansatz_f = ansatz_fidelity(N_QUBITS, value("--depth").and_then(|v| v.parse().ok()).unwrap_or(4), p2 / 10.0, p2);
    let fidelity = ansatz_f * prep_fidelity;
    let mitigate = args.iter().any(|a| a == "--mitigate-readout");
    let plan = Plan { kind, mitigate: if mitigate { readout } else { None } };
    let noisy_train: usize = value("--noisy-train").and_then(|v| v.parse().ok()).unwrap_or(2_000);

    let train = zinc_data::load(TRAIN_CSV, 10_000).expect("load training set");
    let test = zinc_data::load(TEST_CSV, 2_000).expect("load held-out set");
    let scaler = Scaler::fit(scaling, &train);
    let encoding = DataEncodingLayer::new(N_QUBITS, EncodingKind::Amplitude);
    let ansatz = CircuitAnsatz::new(N_QUBITS, depth).with_pi_layer(true);

    if p2 > 0.0 || readout.is_some() {
        println!("Device errors: circuit fidelity {fidelity:.3} = ansatz {ansatz_f:.3} x state preparation {prep_fidelity:.3} (two-qubit error {p2}), readout {readout:?}, mitigation {mitigate}");
    }
    println!("Simulating the circuit (depth {depth}, {scaling:?} inputs, {kind:?}) for {} + {} molecules...", train.len(), test.len());
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
        p.iter()
            .map(|probs| {
                let mut row = vec![0.0; kind.count(N_QUBITS)];
                probability_features_into(probs, kind, &mut row);
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

    let test_cdf: Vec<Vec<f64>> = test_p.iter().map(|p| cumulative(&device_distribution(p, fidelity, readout))).collect();
    let train_cdf: Vec<Vec<f64>> = train_p.iter().take(noisy_train).map(|p| cumulative(&device_distribution(p, fidelity, readout))).collect();
    if args.iter().any(|a| a == "--sweep") {
        println!("Sweeping noise-aware head settings ({seeds} seeds, {} noisy training molecules)...", train_cdf.len());
        run_sweep((&train_exact, &test_exact), (&train_y, &test_y), (&train_cdf, &test_cdf), seeds, encoding.dim, &plan);
        return;
    }
    let mut counts = vec![0u32; encoding.dim];
    let mut rows = Vec::new();
    println!("┌─────────┬───────────────────────────────┬───────────────────────────────┐");
    println!("│  shots  │ exact-trained head            │ noisy-trained head            │");
    println!("│         │ rel. MSE (sd)       mean r    │ rel. MSE (sd)       mean r    │");
    println!("├─────────┼───────────────────────────────┼───────────────────────────────┤");
    let shot_list: Vec<usize> = value("--shots")
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| SHOTS.to_vec());
    for &shots in &shot_list {
        let (mut a_mse, mut a_r, mut b_mse, mut b_r) = (vec![], vec![], vec![], vec![]);
        for seed in 0..seeds {
            let mut rng = Rng(0xC0FF_EE00 ^ (shots as u64) << 8 ^ seed as u64);
            let noisy_test: Vec<Vec<f64>> = test_cdf.iter().map(|c| sampled_features(c, shots, &mut rng, &mut counts, &plan)).collect();
            let (m, r) = score(&exact_head, &noisy_test, &test_y);
            a_mse.push(m);
            a_r.push(r);
            let noisy_train_x: Vec<Vec<f64>> = train_cdf.iter().map(|c| sampled_features(c, shots, &mut rng, &mut counts, &plan)).collect();
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

    let report = serde_json::json!({ "scaling": format!("{scaling:?}"), "observables": format!("{kind:?}"), "depth": depth, "gate_error_p2": p2, "circuit_fidelity": fidelity, "ansatz_fidelity": ansatz_f, "state_preparation_fidelity": prep_fidelity, "readout_p10": readout.map(|e| e.p10), "readout_mitigated": mitigate, "noise_free_relative_mse": ref_mse, "noise_free_mean_r": ref_r, "seeds": seeds, "noisy_train_molecules": train_cdf.len(), "rows": rows });
    let obs_tag = match kind {
        Observables::ParityUpTo(k) => format!("w{k}"),
        Observables::ZAndZz => "w2".to_string(),
        Observables::SingleZ => "w1".to_string(),
    };
    let device_tag = if p2 > 0.0 || readout.is_some() {
        format!("_g{p2}_r{}{}{}", readout.map_or(0.0, |e| e.p10), if mitigate { "_m" } else { "" }, if prep_cnots > 0.0 { format!("_prep{prep_cnots:.0}") } else { String::new() })
    } else {
        String::new()
    };
    let path = format!("results/experiments/zinc_shot_noise_{}_{obs_tag}{device_tag}.json", format!("{scaling:?}").to_lowercase());
    let _ = t1000::results::write(path, serde_json::to_string_pretty(&report).unwrap_or_default());
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
        let mut counts = vec![0u32; 1 << N_QUBITS];
        let plan = Plan { kind: Observables::ZAndZz, mitigate: None };
        let row = sampled_features(&cdf, 400_000, &mut Rng(1), &mut counts, &plan);
        // Z0 = +1 on index 0 and -1 on index 1: 0.7 - 0.3 = 0.4; all other Z are +1.
        let sigma = (1.0f64 - 0.16).sqrt() / 400_000f64.sqrt();
        assert!((row[0] - 0.4).abs() < 5.0 * sigma, "Z0 = {}", row[0]);
        assert!((row[1] - 1.0).abs() < 1e-12);
    }
}
