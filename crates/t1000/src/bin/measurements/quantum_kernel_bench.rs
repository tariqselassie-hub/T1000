// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Quantum-kernel classifier against classical kernels under nested cross-validation.
//!
//! ```text
//! quantum_kernel_bench                       MNIST digit pairs from data/mnist_demo + two controls
//! quantum_kernel_bench --csv file.csv        a table of features and a last 0/1 (or -1/+1) label column
//! ```
//!
//! Every model is a kernel ridge classifier (`alpha = (K + lambda I)^-1 y`, predict `sign(K_test alpha)`).
//! Kernels: the fidelity kernel `|<phi(x)|phi(y)>|^2` of a layered angle-encoding circuit (Ry and Rz
//! rotations from the features, CZ ring, `L` repeats; features min-max scaled on the training rows and
//! multiplied by `s * pi`), an RBF kernel and a linear kernel (features z-scored on the training rows).
//! Hyperparameters (quantum `s`, `L`; RBF `gamma`; all `lambda`) are chosen by 3-fold inner cross-validation
//! inside every outer training set; the outer fold is only used to score. The quantum and RBF kernels get
//! the same number of hyperparameter combinations (48). Outer folds: 5-fold stratified, repeated.
//!
//! Two controls check the pipeline: a *planted* task whose labels are the sign of a difference of
//! quantum-kernel values (the quantum kernel should win), and the first MNIST task with *shuffled* labels
//! (every model should sit at chance).

use rayon::prelude::*;
use serde::Serialize;
use std::f64::consts::PI;
use std::fs;
use std::path::Path;

type C = (f64, f64);

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        let mut r = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        for _ in 0..4 {
            r.next();
        }
        r
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i + 1));
        }
    }
}

struct Dataset {
    name: String,
    x: Vec<Vec<f64>>,
    y: Vec<f64>, // -1 / +1
}

// ---- data -----------------------------------------------------------------------------------

fn read_idx(path: &Path) -> (Vec<u8>, usize) {
    let b = fs::read(path).unwrap_or_else(|e| panic!("{}: {e} (the demo set ships in data/mnist_demo)", path.display()));
    let n = u32::from_be_bytes([b[4], b[5], b[6], b[7]]) as usize;
    let offset = if b[3] == 3 { 16 } else { 8 };
    (b[offset..].to_vec(), n)
}

/// 28x28 images averaged over 7x7 blocks: 16 features in [0, 1].
fn pool(img: &[u8]) -> Vec<f64> {
    let mut out = Vec::with_capacity(16);
    for by in 0..4 {
        for bx in 0..4 {
            let mut s = 0.0;
            for y in 0..7 {
                for x in 0..7 {
                    s += f64::from(img[(by * 7 + y) * 28 + bx * 7 + x]);
                }
            }
            out.push(s / 49.0 / 255.0);
        }
    }
    out
}

fn mnist_pairs(dir: &Path, pairs: &[(u8, u8)]) -> Vec<Dataset> {
    let mut images = Vec::new();
    let mut labels = Vec::new();
    for (img, lab) in [("train-images.idx3-ubyte", "train-labels.idx1-ubyte"), ("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")] {
        let (ib, n) = read_idx(&dir.join(img));
        let (lb, _) = read_idx(&dir.join(lab));
        for i in 0..n {
            images.push(pool(&ib[i * 784..(i + 1) * 784]));
            labels.push(lb[i]);
        }
    }
    pairs
        .iter()
        .map(|&(a, b)| {
            let (mut x, mut y) = (Vec::new(), Vec::new());
            for (img, &l) in images.iter().zip(&labels) {
                if l == a || l == b {
                    x.push(img.clone());
                    y.push(if l == b { 1.0 } else { -1.0 });
                }
            }
            Dataset { name: format!("MNIST {a} vs {b}"), x, y }
        })
        .collect()
}

fn load_csv(path: &Path) -> Dataset {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for line in text.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let v: Vec<f64> = line.split(',').map(|t| t.trim().parse().expect("number")).collect();
        let (label, feats) = v.split_last().expect("columns");
        x.push(feats.to_vec());
        y.push(if *label > 0.5 { 1.0 } else { -1.0 });
    }
    Dataset { name: path.file_stem().unwrap().to_string_lossy().into_owned(), x, y }
}

// ---- kernels ----------------------------------------------------------------------------------

/// State of the layered feature-map circuit for features `u` (already scaled to angles).
fn feature_state(u: &[f64], layers: usize) -> Vec<C> {
    let n = u.len().div_ceil(2);
    let mut st = vec![(0.0, 0.0); 1 << n];
    st[0] = (1.0, 0.0);
    for _ in 0..layers {
        for q in 0..n {
            // Ry(u[q]) then Rz(u[q+n])
            let bit = 1usize << q;
            let (c, s) = ((u[q] / 2.0).cos(), (u[q] / 2.0).sin());
            for i0 in 0..st.len() {
                if i0 & bit == 0 {
                    let (a, b) = (st[i0], st[i0 | bit]);
                    st[i0] = (c * a.0 - s * b.0, c * a.1 - s * b.1);
                    st[i0 | bit] = (s * a.0 + c * b.0, s * a.1 + c * b.1);
                }
            }
            if let Some(&z) = u.get(q + n) {
                let (c, s) = ((z / 2.0).cos(), (z / 2.0).sin());
                for (i, v) in st.iter_mut().enumerate() {
                    let sg = if i & bit == 0 { -s } else { s };
                    *v = (c * v.0 - sg * v.1, c * v.1 + sg * v.0);
                }
            }
        }
        if n > 1 {
            for q in 0..n {
                let mask = (1usize << q) | (1usize << ((q + 1) % n));
                for (i, v) in st.iter_mut().enumerate() {
                    if i & mask == mask {
                        *v = (-v.0, -v.1);
                    }
                }
            }
        }
    }
    st
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Quantum { s: f64, layers: usize },
    Rbf { gamma: f64 },
    Linear,
}

struct Scaler {
    lo: Vec<f64>,
    hi: Vec<f64>,
    mean: Vec<f64>,
    sd: Vec<f64>,
}

impl Scaler {
    fn fit(rows: &[&Vec<f64>]) -> Self {
        let d = rows[0].len();
        let n = rows.len() as f64;
        let mut s = Scaler { lo: vec![f64::MAX; d], hi: vec![f64::MIN; d], mean: vec![0.0; d], sd: vec![0.0; d] };
        for r in rows {
            for j in 0..d {
                s.lo[j] = s.lo[j].min(r[j]);
                s.hi[j] = s.hi[j].max(r[j]);
                s.mean[j] += r[j] / n;
            }
        }
        for r in rows {
            for j in 0..d {
                s.sd[j] += (r[j] - s.mean[j]).powi(2) / n;
            }
        }
        for j in 0..d {
            s.sd[j] = s.sd[j].sqrt().max(1e-9);
        }
        s
    }
    fn unit(&self, r: &[f64]) -> Vec<f64> {
        (0..r.len()).map(|j| ((r[j] - self.lo[j]) / (self.hi[j] - self.lo[j]).max(1e-12)).clamp(0.0, 1.0)).collect()
    }
    fn z(&self, r: &[f64]) -> Vec<f64> {
        (0..r.len()).map(|j| (r[j] - self.mean[j]) / self.sd[j]).collect()
    }
}

/// Gram matrix (row-major, `rows x rows`) of `kind` over the given rows, scaled with `sc`.
fn gram(kind: Kind, rows: &[&Vec<f64>], sc: &Scaler) -> Vec<f64> {
    let n = rows.len();
    let mut g = vec![0.0; n * n];
    match kind {
        Kind::Quantum { s, layers } => {
            let states: Vec<Vec<C>> = rows
                .par_iter()
                .map(|r| feature_state(&sc.unit(r).iter().map(|u| u * PI * s).collect::<Vec<_>>(), layers))
                .collect();
            g.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
                for j in 0..n {
                    let (mut re, mut im) = (0.0, 0.0);
                    for (a, b) in states[i].iter().zip(&states[j]) {
                        re += a.0 * b.0 + a.1 * b.1;
                        im += a.0 * b.1 - a.1 * b.0;
                    }
                    row[j] = re * re + im * im;
                }
            });
        }
        Kind::Rbf { .. } | Kind::Linear => {
            let z: Vec<Vec<f64>> = rows.iter().map(|r| sc.z(r)).collect();
            let d = z[0].len() as f64;
            g.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
                for j in 0..n {
                    let dot: f64 = z[i].iter().zip(&z[j]).map(|(a, b)| a * b).sum();
                    row[j] = match kind {
                        Kind::Rbf { gamma } => {
                            let d2: f64 = z[i].iter().zip(&z[j]).map(|(a, b)| (a - b).powi(2)).sum();
                            (-gamma * d2 / d).exp()
                        }
                        _ => dot / d,
                    };
                }
            });
        }
    }
    g
}

/// Solves `(K + lambda I) alpha = y` by Cholesky; returns `None` if not positive definite.
fn ridge(k: &[f64], n: usize, y: &[f64], lambda: f64) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = k[i * n + j] + if i == j { lambda } else { 0.0 };
            for p in 0..j {
                s -= l[i * n + p] * l[j * n + p];
            }
            if i == j {
                if s <= 1e-14 {
                    return None;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    let mut z = vec![0.0; n];
    for i in 0..n {
        let mut s = y[i];
        for p in 0..i {
            s -= l[i * n + p] * z[p];
        }
        z[i] = s / l[i * n + i];
    }
    let mut a = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = z[i];
        for p in i + 1..n {
            s -= l[p * n + i] * a[p];
        }
        a[i] = s / l[i * n + i];
    }
    Some(a)
}

/// Accuracy of the ridge classifier fit on `train` and scored on `test` (indices into the Gram `g`).
fn fit_score(g: &[f64], stride: usize, train: &[usize], test: &[usize], y: &[f64], lambda: f64) -> f64 {
    let n = train.len();
    let mut k = vec![0.0; n * n];
    for (a, &i) in train.iter().enumerate() {
        for (b, &j) in train.iter().enumerate() {
            k[a * n + b] = g[i * stride + j];
        }
    }
    let yt: Vec<f64> = train.iter().map(|&i| y[i]).collect();
    let Some(alpha) = ridge(&k, n, &yt, lambda) else { return 0.0 };
    let correct = test
        .iter()
        .filter(|&&t| {
            let f: f64 = train.iter().zip(&alpha).map(|(&i, a)| g[t * stride + i] * a).sum();
            (f >= 0.0) == (y[t] > 0.0)
        })
        .count();
    correct as f64 / test.len() as f64
}

// ---- cross-validation --------------------------------------------------------------------------

fn stratified_folds(idx: &[usize], y: &[f64], k: usize, rng: &mut Rng) -> Vec<Vec<usize>> {
    let mut folds = vec![Vec::new(); k];
    for class in [-1.0, 1.0] {
        let mut members: Vec<usize> = idx.iter().copied().filter(|&i| y[i] == class).collect();
        rng.shuffle(&mut members);
        for (p, i) in members.into_iter().enumerate() {
            folds[p % k].push(i);
        }
    }
    folds
}

const LAMBDAS: [f64; 4] = [1e-3, 1e-2, 1e-1, 1.0];

fn grid(name: &str) -> Vec<Kind> {
    match name {
        "quantum" => {
            let mut v = Vec::new();
            for &s in &[0.125, 0.25, 0.5, 1.0] {
                for &layers in &[1usize, 2, 3] {
                    v.push(Kind::Quantum { s, layers });
                }
            }
            v
        }
        "rbf" => [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0].iter().map(|&gamma| Kind::Rbf { gamma }).collect(),
        _ => vec![Kind::Linear],
    }
}

/// One outer fold: chooses hyperparameters on `train` by inner CV, returns the outer test accuracy.
fn outer_fold(data: &Dataset, model: &str, train: &[usize], test: &[usize], seed: u64) -> f64 {
    let all: Vec<usize> = train.iter().chain(test).copied().collect();
    let rows: Vec<&Vec<f64>> = all.iter().map(|&i| &data.x[i]).collect();
    let train_rows: Vec<&Vec<f64>> = train.iter().map(|&i| &data.x[i]).collect();
    let sc = Scaler::fit(&train_rows);
    // positions inside `all`
    let tr: Vec<usize> = (0..train.len()).collect();
    let te: Vec<usize> = (train.len()..all.len()).collect();
    let y_all: Vec<f64> = all.iter().map(|&i| data.y[i]).collect();
    let mut rng = Rng::new(seed);
    let inner = stratified_folds(&tr, &y_all, 3, &mut rng);
    let stride = all.len();

    let mut best = (f64::MIN, Kind::Linear, 1.0);
    for kind in grid(model) {
        let g = gram(kind, &rows, &sc);
        for &lambda in &LAMBDAS {
            let score: f64 = (0..3)
                .map(|f| {
                    let val = &inner[f];
                    let fit: Vec<usize> = (0..3).filter(|&o| o != f).flat_map(|o| inner[o].iter().copied()).collect();
                    fit_score(&g, stride, &fit, val, &y_all, lambda)
                })
                .sum::<f64>()
                / 3.0;
            if score > best.0 {
                best = (score, kind, lambda);
            }
        }
    }
    let g = gram(best.1, &rows, &sc);
    fit_score(&g, stride, &tr, &te, &y_all, best.2)
}

#[derive(Serialize)]
struct ModelResult {
    model: String,
    mean_accuracy: f64,
    sd_over_outer_folds: f64,
    accuracies: Vec<f64>,
}

#[derive(Serialize)]
struct TaskResult {
    task: String,
    samples: usize,
    features: usize,
    majority_rate: f64,
    models: Vec<ModelResult>,
    quantum_minus_rbf: f64,
    quantum_minus_rbf_se: f64,
}

fn evaluate(data: &Dataset, repeats: usize) -> TaskResult {
    let n = data.x.len();
    let idx: Vec<usize> = (0..n).collect();
    let mut splits = Vec::new();
    for rep in 0..repeats {
        let mut rng = Rng::new(2026 + rep as u64);
        let folds = stratified_folds(&idx, &data.y, 5, &mut rng);
        for f in 0..5 {
            let test = folds[f].clone();
            let train: Vec<usize> = (0..5).filter(|&o| o != f).flat_map(|o| folds[o].iter().copied()).collect();
            splits.push((train, test, 17 * rep + f));
        }
    }
    let names = ["quantum", "rbf", "linear"];
    let results: Vec<Vec<f64>> = names
        .iter()
        .map(|m| splits.par_iter().map(|(tr, te, seed)| outer_fold(data, m, tr, te, *seed as u64 + 99)).collect())
        .collect();
    let models: Vec<ModelResult> = names
        .iter()
        .zip(&results)
        .map(|(m, a)| {
            let mean = a.iter().sum::<f64>() / a.len() as f64;
            let sd = (a.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (a.len() - 1) as f64).sqrt();
            ModelResult { model: m.to_string(), mean_accuracy: mean, sd_over_outer_folds: sd, accuracies: a.clone() }
        })
        .collect();
    // paired difference, standard error over repeats (outer folds within a repeat share data)
    let diff: Vec<f64> = results[0].iter().zip(&results[1]).map(|(q, r)| q - r).collect();
    let per_rep: Vec<f64> = diff.chunks(5).map(|c| c.iter().sum::<f64>() / 5.0).collect();
    let mean_diff = diff.iter().sum::<f64>() / diff.len() as f64;
    let se = if per_rep.len() > 1 {
        let m = per_rep.iter().sum::<f64>() / per_rep.len() as f64;
        (per_rep.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (per_rep.len() - 1) as f64 / per_rep.len() as f64).sqrt()
    } else {
        f64::NAN
    };
    let pos = data.y.iter().filter(|&&v| v > 0.0).count() as f64 / n as f64;
    TaskResult {
        task: data.name.clone(),
        samples: n,
        features: data.x[0].len(),
        majority_rate: pos.max(1.0 - pos),
        models,
        quantum_minus_rbf: mean_diff,
        quantum_minus_rbf_se: se,
    }
}

/// Labels are the sign of a difference of quantum-kernel values to two anchor points (median-balanced).
fn planted(n: usize, d: usize) -> Dataset {
    let mut rng = Rng::new(31337);
    let x: Vec<Vec<f64>> = (0..n).map(|_| (0..d).map(|_| rng.unit()).collect()).collect();
    let sc = Scaler { lo: vec![0.0; d], hi: vec![1.0; d], mean: vec![0.0; d], sd: vec![1.0; d] };
    let anchors: Vec<Vec<f64>> = (0..2).map(|_| (0..d).map(|_| rng.unit()).collect()).collect();
    let rows: Vec<&Vec<f64>> = x.iter().chain(&anchors).collect();
    let g = gram(Kind::Quantum { s: 1.0, layers: 2 }, &rows, &sc);
    let m = rows.len();
    let score: Vec<f64> = (0..n).map(|i| g[i * m + n] - g[i * m + n + 1]).collect();
    let mut sorted = score.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let med = sorted[n / 2];
    Dataset { name: "planted quantum-kernel labels".into(), x, y: score.iter().map(|&s| if s > med { 1.0 } else { -1.0 }).collect() }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let csv = args.iter().position(|a| a == "--csv").and_then(|i| args.get(i + 1));
    let repeats: usize = args.iter().position(|a| a == "--repeats").and_then(|i| args.get(i + 1)).map_or(3, |v| v.parse().expect("--repeats"));

    let tasks: Vec<Dataset> = if let Some(p) = csv {
        vec![load_csv(Path::new(p))]
    } else {
        let mut t = mnist_pairs(Path::new("data/mnist_demo"), &[(1, 7), (3, 8), (4, 9)]);
        let mut shuffled = Dataset { name: "MNIST 3 vs 8, labels shuffled".into(), x: t[1].x.clone(), y: t[1].y.clone() };
        Rng::new(5).shuffle(&mut shuffled.y);
        t.push(planted(240, 16));
        t.push(shuffled);
        t
    };

    println!("=========================================================================");
    println!(" QUANTUM KERNEL VS CLASSICAL KERNELS, NESTED CROSS-VALIDATION");
    println!(" outer: 5-fold x {repeats} repeats; inner: 3-fold; 48 hyperparameter combinations each for the");
    println!(" quantum and RBF kernels, 4 for the linear kernel (lambda only)");
    println!("=========================================================================");
    println!(" {:<34} {:>5} {:>5} {:>9} {:>9} {:>9} {:>9} {:>16}", "task", "n", "d", "majority", "quantum", "rbf", "linear", "quantum - rbf (se)");
    let mut out = Vec::new();
    for data in &tasks {
        let r = evaluate(data, repeats);
        println!(
            " {:<34} {:>5} {:>5} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>+10.4} ({:.4})",
            r.task, r.samples, r.features, r.majority_rate, r.models[0].mean_accuracy, r.models[1].mean_accuracy,
            r.models[2].mean_accuracy, r.quantum_minus_rbf, r.quantum_minus_rbf_se
        );
        out.push(r);
    }
    println!("\n accuracy on the outer test folds; se is over repeats (folds inside a repeat share data).");
    let dir = Path::new("results/data");
    let _ = fs::create_dir_all(dir);
    let name = if csv.is_some() { "quantum_kernel_bench_csv.json" } else { "quantum_kernel_bench.json" };
    t1000::results::write(&dir.join(name), serde_json::to_string_pretty(&out).expect("serialise")).expect("write report");
}
