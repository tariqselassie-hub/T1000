// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! 2D Molecular Correlation Spectroscopy (2D COSY NMR) Hybrid QNN Engine.
//!
//! Trains a 10-Qubit Variational Quantum Circuit on 32x32 2D COSY frequency spectra
//! to classify 10 distinct molecular scaffold families via 65 Pauli Correlators (Z, X, ZZ)
//! and AVX2 adjoint gradients with full hidden neuron diagnostics.

use rayon::prelude::*;
use std::time::Instant;

use t1000::adjoint::AdjointWorkspace;
use t1000::chemistry::nmr2d::{
    extract_nmr_features, generate_cosy_spectrum, NMR_STATE_DIM, N_NMR_SPATIAL,
};
use t1000::results::{Better, Score};
use t1000::zinc_training_common::utc_timestamp_rfc3339;
use t1000::{
    adjoint_gradient_buffered, AdamOptimizer, CircuitAnsatz, PauliObservable, QuantumState,
};

const N_QUBITS: usize = 10;
const N_CLASSES: usize = 10;
const N_OBSERVABLES: usize = 80; // 10 Z + 10 X + 45 ZZ + 5 XX mirror + 5 YY mirror + 4 XX adj + 1 AllZ
const N_FUSED: usize = N_OBSERVABLES + N_NMR_SPATIAL; // 145
const N_HIDDEN: usize = 128;
const RESULTS_PROBE_PATH: &str = "results/data/nmr2d_spectroscopy_probe.json";
const WEB_PROBE_PATH: &str = "web/data/nmr2d_spectroscopy_probe.json";

#[derive(Debug, Clone)]
pub struct MolecularSample {
    pub name: String,
    pub smiles: String,
    pub family: &'static str,
    pub label: usize,
    pub state: QuantumState,
    pub matrix: [f64; NMR_STATE_DIM],
    pub spatial: [f64; N_NMR_SPATIAL],
}

pub const FAMILIES: [&str; 10] = [
    "Aliphatic Alkanes",
    "Alcohols & Polyols",
    "Aldehydes & Ketones",
    "Carboxylic Acids & Esters",
    "Amines & Amides",
    "Aromatic Benzenes",
    "Heteroaromatics",
    "Alkenes & Dienes",
    "Terpenoids & Bicyclic",
    "Amino Acids & Peptides",
];

/// Curated molecular templates across the 10 chemical scaffold families.
pub fn generate_molecular_catalog() -> Vec<(&'static str, &'static str, usize)> {
    vec![
        // 0: Alkanes (pure aliphatic 0.8-1.5 ppm, no downfield cross-peaks)
        ("Pentane", "CCCCC", 0),
        ("Hexane", "CCCCCC", 0),
        ("Heptane", "CCCCCCC", 0),
        ("Octane", "CCCCCCCC", 0),
        ("2-Methylbutane", "CCC(C)C", 0),
        ("2,2-Dimethylbutane", "CCC(C)(C)C", 0),
        ("Cyclopentane", "C1CCCC1", 0),
        ("Cyclohexane", "C1CCCCC1", 0),
        ("Methylcyclohexane", "CC1CCCCC1", 0),
        ("2,3-Dimethylbutane", "CC(C)C(C)C", 0),
        // 1: Alcohols (strong 3.4-4.0 ppm cross-peak to alkyls)
        ("Ethanol", "CCO", 1),
        ("1-Propanol", "CCCO", 1),
        ("Isopropanol", "CC(O)C", 1),
        ("1-Butanol", "CCCCO", 1),
        ("2-Butanol", "CCC(O)C", 1),
        ("tert-Butanol", "CC(C)(O)C", 1),
        ("1,3-Propanediol", "OCCCO", 1),
        ("Cyclohexanol", "OC1CCCCC1", 1),
        ("1-Pentanol", "CCCCCO", 1),
        ("2-Methyl-1-propanol", "CC(C)CO", 1),
        // 2: Aldehydes & Ketones (alpha protons 2.1-2.5 ppm, aldehyde 9.7 ppm)
        ("Acetaldehyde", "CC=O", 2),
        ("Propanal", "CCC=O", 2),
        ("Acetone", "CC(=O)C", 2),
        ("2-Butanone", "CCC(=O)C", 2),
        ("3-Pentanone", "CCC(=O)CC", 2),
        ("Cyclopentanone", "O=C1CCCC1", 2),
        ("Cyclohexanone", "O=C1CCCCC1", 2),
        ("Butanal", "CCCC=O", 2),
        ("2-Hexanone", "CCCCC(=O)C", 2),
        ("4-Methyl-2-pentanone", "CC(C)CC(=O)C", 2),
        // 3: Carboxylic Acids & Esters (acid ~11.5 ppm, ester O-CH2 3.8-4.2 ppm)
        ("Acetic Acid", "CC(=O)O", 3),
        ("Propionic Acid", "CCC(=O)O", 3),
        ("Butyric Acid", "CCCC(=O)O", 3),
        ("Methyl Acetate", "COC(=O)C", 3),
        ("Ethyl Acetate", "CCOC(=O)C", 3),
        ("Methyl Propionate", "COC(=O)CC", 3),
        ("Ethyl Formate", "CCOC=O", 3),
        ("Isobutyl Acetate", "CC(C)COC(=O)C", 3),
        ("Valeric Acid", "CCCCC(=O)O", 3),
        ("Methyl Butyrate", "CCCC(=O)OC", 3),
        // 4: Amines & Amides (N-CH2 2.6-3.0 ppm, amide 6.8 ppm)
        ("Ethylamine", "CCN", 4),
        ("Propylamine", "CCCN", 4),
        ("Diethylamine", "CCNCC", 4),
        ("Triethylamine", "CCN(CC)CC", 4),
        ("Acetamide", "CC(=O)N", 4),
        ("N-Methylacetamide", "CNC(=O)C", 4),
        ("Pyrrolidine", "C1CCCN1", 4),
        ("Piperidine", "C1CCNCC1", 4),
        ("Butylamine", "CCCCN", 4),
        ("N,N-Dimethylformamide", "CN(C)C=O", 4),
        // 5: Aromatic Benzenes (dense 7.0-7.8 ppm ortho/meta cluster)
        ("Toluene", "Cc1ccccc1", 5),
        ("Ethylbenzene", "CCc1ccccc1", 5),
        ("Phenol", "Oc1ccccc1", 5),
        ("Aniline", "Nc1ccccc1", 5),
        ("Benzaldehyde", "O=Cc1ccccc1", 5),
        ("Benzoic Acid", "O=C(O)c1ccccc1", 5),
        ("o-Xylene", "Cc1ccccc1C", 5),
        ("m-Xylene", "Cc1cccc(C)c1", 5),
        ("p-Xylene", "Cc1ccc(C)cc1", 5),
        ("Chlorobenzene", "Clc1ccccc1", 5),
        // 6: Heteroaromatics (downfield 7.2-8.6 ppm heterocyclic split)
        ("Pyridine", "c1ccncc1", 6),
        ("Furan", "c1ccoc1", 6),
        ("Thiophene", "c1ccsc1", 6),
        ("Pyrrole", "c1cc[nH]c1", 6),
        ("2-Methylpyridine", "Cc1ccccn1", 6),
        ("3-Methylpyridine", "Cc1cccnc1", 6),
        ("Thiazole", "c1cscn1", 6),
        ("Oxazole", "c1cocn1", 6),
        ("Pyrimidine", "c1cncnc1", 6),
        ("Imidazole", "c1c[nH]cn1", 6),
        // 7: Alkenes & Dienes (vinylic 4.8-6.0 ppm cross-coupled to allylics)
        ("1-Pentene", "CCCC=C", 7),
        ("2-Pentene", "CCC=CC", 7),
        ("1-Hexene", "CCCCC=C", 7),
        ("Cyclohexene", "C1CC=CCC1", 7),
        ("Cyclopentene", "C1C=CCC1", 7),
        ("Isoprene", "CC(=C)C=C", 7),
        ("1,3-Butadiene", "C=CC=C", 7),
        ("2-Methyl-2-butene", "CC=C(C)C", 7),
        ("Cycloheptene", "C1CCCC=CC1", 7),
        ("1,4-Pentadiene", "C=CCC=C", 7),
        // 8: Terpenoids & Bicyclic (dense multi-ring bridged cross-peaks)
        ("Camphor", "CC1(C)C2CCC1(C)C(=O)C2", 8),
        ("alpha-Pinene", "CC1=CCC2CC1C2(C)C", 8),
        ("beta-Pinene", "C=C1CCC2CC1C2(C)C", 8),
        ("Limonene", "CC1=CCC(CC1)C(=C)C", 8),
        ("Norbornene", "C1CC2CC1C=C2", 8),
        ("Borneol", "CC1(C)C2CCC1(C)C(O)C2", 8),
        ("Camphene", "CC1(C)C2CCC(C2)C1=C", 8),
        ("Menthol", "CC1CCC(C(O)C1)C(C)C", 8),
        ("Carvone", "CC(=C)C1CC=C(C)C(=O)C1", 8),
        ("Fenchone", "CC1(C)C2CCC(C)(C2)C1=O", 8),
        // 9: Amino Acids & Peptides (alpha-CH 4.0-4.5 ppm coupled to beta-R)
        ("Glycine", "NCC(=O)O", 9),
        ("Alanine", "CC(N)C(=O)O", 9),
        ("Valine", "CC(C)C(N)C(=O)O", 9),
        ("Serine", "OCC(N)C(=O)O", 9),
        ("Leucine", "CC(C)CC(N)C(=O)O", 9),
        ("Isoleucine", "CCC(C)C(N)C(=O)O", 9),
        ("Proline", "O=C(O)C1CCCN1", 9),
        ("Glycyl-Alanine", "NCC(=O)NC(C)C(=O)O", 9),
        ("Alanyl-Glycine", "CC(N)C(=O)NCC(=O)O", 9),
        ("Threonine", "CC(O)C(N)C(=O)O", 9),
    ]
}

#[derive(Debug, Clone)]
pub struct SimpleMlpHead {
    pub w1: Vec<f64>,
    pub b1: Vec<f64>,
    pub w2: Vec<f64>,
    pub b2: Vec<f64>,
}

impl SimpleMlpHead {
    pub fn new(seed: u64) -> Self {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let s1 = (2.0 / N_FUSED as f64).sqrt();
        let w1 = (0..N_HIDDEN * N_FUSED)
            .map(|_| rng.gen_range(-s1..s1))
            .collect();
        let b1 = vec![0.01; N_HIDDEN];

        let s2 = (2.0 / N_HIDDEN as f64).sqrt();
        let w2 = (0..N_CLASSES * N_HIDDEN)
            .map(|_| rng.gen_range(-s2..s2))
            .collect();
        let b2 = vec![0.0; N_CLASSES];

        Self { w1, b1, w2, b2 }
    }

    pub fn forward(
        &self,
        x: &[f64; N_FUSED],
    ) -> ([f64; N_CLASSES], [f64; N_HIDDEN], [f64; N_HIDDEN]) {
        let mut z1 = [0.0; N_HIDDEN];
        let mut a1 = [0.0; N_HIDDEN];
        for (h, (z_out, a_out)) in z1.iter_mut().zip(a1.iter_mut()).enumerate() {
            let row = h * N_FUSED;
            let sum = self.b1[h]
                + self.w1[row..row + N_FUSED]
                    .iter()
                    .zip(x)
                    .map(|(w, &xi)| w * xi)
                    .sum::<f64>();
            *z_out = sum;
            *a_out = if sum > 0.0 { sum } else { 0.05 * sum };
        }

        let mut logits = [0.0; N_CLASSES];
        for (c, logit) in logits.iter_mut().enumerate() {
            let row = c * N_HIDDEN;
            *logit = self.b2[c]
                + self.w2[row..row + N_HIDDEN]
                    .iter()
                    .zip(&a1)
                    .map(|(w, &ai)| w * ai)
                    .sum::<f64>();
        }
        (logits, z1, a1)
    }

    pub fn backward(
        &self,
        x: &[f64; N_FUSED],
        z1: &[f64; N_HIDDEN],
        a1: &[f64; N_HIDDEN],
        d_logits: &[f64; N_CLASSES],
        grad_w1: &mut [f64],
        grad_b1: &mut [f64],
        grad_w2: &mut [f64],
        grad_b2: &mut [f64],
    ) -> [f64; N_FUSED] {
        for (c, &d_c) in d_logits.iter().enumerate() {
            let row = c * N_HIDDEN;
            grad_b2[c] += d_c;
            for (h, &ai) in a1.iter().enumerate() {
                grad_w2[row + h] += d_c * ai;
            }
        }

        let mut d_a1 = [0.0; N_HIDDEN];
        for (h, d_a) in d_a1.iter_mut().enumerate() {
            let sum: f64 = d_logits
                .iter()
                .enumerate()
                .map(|(c, &d_c)| d_c * self.w2[c * N_HIDDEN + h])
                .sum();
            *d_a = sum;
        }

        let mut d_z1 = [0.0; N_HIDDEN];
        for (h, (&z, &da)) in z1.iter().zip(&d_a1).enumerate() {
            let dz = if z > 0.0 { da } else { 0.05 * da };
            d_z1[h] = dz;
            grad_b1[h] += dz;
        }

        let mut d_x = [0.0; N_FUSED];
        for (&dz, w1_chunk) in d_z1.iter().zip(self.w1.chunks_exact(N_FUSED)) {
            for (dx_val, &w_val) in d_x.iter_mut().zip(w1_chunk) {
                *dx_val += dz * w_val;
            }
        }
        for (h, &dz) in d_z1.iter().enumerate() {
            let row = h * N_FUSED;
            for (gw, &xi) in grad_w1[row..row + N_FUSED].iter_mut().zip(x) {
                *gw += dz * xi;
            }
        }

        d_x
    }
}

fn softmax_cross_entropy(
    logits: &[f64; N_CLASSES],
    label: usize,
) -> (f64, [f64; N_CLASSES], usize) {
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut exp_sum = 0.0;
    let mut probs = [0.0; N_CLASSES];
    for c in 0..N_CLASSES {
        let p = (logits[c] - max_logit).exp();
        probs[c] = p;
        exp_sum += p;
    }
    for p in probs.iter_mut() {
        *p /= exp_sum;
    }

    let loss = -probs[label].max(1e-15).ln();
    let mut d_logits = probs;
    d_logits[label] -= 1.0;

    let mut pred = 0;
    let mut max_p = -1.0;
    for (c, &p) in probs.iter().enumerate() {
        if p > max_p {
            max_p = p;
            pred = c;
        }
    }

    (loss, d_logits, pred)
}

/// Number of molecule-disjoint cross-validation folds. Fold `k` holds out the catalog entries at
/// positions `2k` and `2k + 1` of every family, so each fold tests 20 molecules it never saw.
const N_FOLDS: usize = 5;
/// Each training molecule appears this many times per epoch, keeping the 800-sample epoch (and
/// the number of optimizer steps) of the original single-split run.
const TRAIN_REPLICAS: usize = 10;
const EPOCHS: u32 = 16;
const BATCH_SIZE: usize = 32;
const DEPTH: usize = 5;

/// Builds the (deterministic) COSY sample of one catalog molecule.
fn build_sample(name: &str, smiles: &str, label: usize) -> MolecularSample {
    let matrix = generate_cosy_spectrum(smiles).expect("Valid SMILES");
    let spatial = extract_nmr_features(&matrix);
    let mut state = QuantumState::zero(NMR_STATE_DIM);
    state.real.copy_from_slice(&matrix);
    MolecularSample {
        name: name.to_string(),
        smiles: smiles.to_string(),
        family: FAMILIES[label],
        label,
        state,
        matrix,
        spatial,
    }
}

/// Whether catalog entry `idx` is held out in `fold` (10 entries per family, in order).
fn is_held_out(idx: usize, fold: usize) -> bool {
    (idx % 10) / 2 == fold
}

/// Trained parameters and held-out results of one fold.
#[derive(Debug)]
struct FoldOutcome {
    ansatz_weights: Vec<f64>,
    head: SimpleMlpHead,
    train_acc_pct: f64,
    test_loss: f64,
    test_correct: usize,
    firing: [usize; N_HIDDEN],
    predictions: Vec<Prediction>,
}

/// (molecule name, true label, predicted label) of one held-out molecule.
type Prediction = (String, usize, usize);

/// Held-out loss, correct count, hidden-neuron firing counts and predictions on `test`.
fn evaluate(
    ansatz: &CircuitAnsatz,
    weights: &[f64],
    observables: &[PauliObservable],
    head: &SimpleMlpHead,
    test: &[&MolecularSample],
) -> (f64, usize, [usize; N_HIDDEN], Vec<Prediction>) {
    let mut loss_sum = 0.0;
    let mut correct = 0;
    let mut firing = [0usize; N_HIDDEN];
    let mut predictions = Vec::with_capacity(test.len());
    for sample in test {
        let mut phi = sample.state.clone();
        ansatz.forward_in_place(&mut phi, weights);
        let mut fused = [0.0; N_FUSED];
        for (f, o) in fused.iter_mut().zip(observables) {
            *f = o.expectation_value(&phi);
        }
        fused[N_OBSERVABLES..].copy_from_slice(&sample.spatial);
        let (logits, _, a1) = head.forward(&fused);
        let (loss, _, pred) = softmax_cross_entropy(&logits, sample.label);
        loss_sum += loss;
        correct += usize::from(pred == sample.label);
        for (count, &a) in firing.iter_mut().zip(&a1) {
            *count += usize::from(a > 0.0);
        }
        predictions.push((sample.name.clone(), sample.label, pred));
    }
    (loss_sum / test.len() as f64, correct, firing, predictions)
}

/// Head and ansatz gradients summed over one batch, plus its loss and correct count.
#[derive(Debug)]
struct BatchGrads {
    ansatz: Vec<f64>,
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: Vec<f64>,
    loss: f64,
    correct: usize,
}

fn batch_gradients(
    ansatz: &CircuitAnsatz,
    weights: &[f64],
    observables: &[PauliObservable],
    head: &SimpleMlpHead,
    batch: &[&MolecularSample],
    ws: &mut AdjointWorkspace,
) -> BatchGrads {
    let mut g = BatchGrads {
        ansatz: vec![0.0; weights.len()],
        w1: vec![0.0; head.w1.len()],
        b1: vec![0.0; head.b1.len()],
        w2: vec![0.0; head.w2.len()],
        b2: vec![0.0; head.b2.len()],
        loss: 0.0,
        correct: 0,
    };
    for sample in batch {
        let adj = adjoint_gradient_buffered(
            ansatz,
            &sample.state,
            weights,
            observables,
            |exps| {
                let mut fused = [0.0; N_FUSED];
                fused[..N_OBSERVABLES].copy_from_slice(exps);
                fused[N_OBSERVABLES..].copy_from_slice(&sample.spatial);
                let (logits, z1, a1) = head.forward(&fused);
                let (loss, d_logits, pred) = softmax_cross_entropy(&logits, sample.label);
                g.loss += loss;
                g.correct += usize::from(pred == sample.label);
                let d_fused = head.backward(
                    &fused, &z1, &a1, &d_logits, &mut g.w1, &mut g.b1, &mut g.w2, &mut g.b2,
                );
                d_fused[..N_OBSERVABLES].to_vec()
            },
            ws,
        );
        for (acc, d) in g.ansatz.iter_mut().zip(&adj.gradient) {
            *acc += d;
        }
    }
    g
}

/// Trains a fresh model on `train` and evaluates it on the held-out molecules of `test`.
fn train_fold(
    ansatz: &CircuitAnsatz,
    observables: &[PauliObservable],
    train: &[&MolecularSample],
    test: &[&MolecularSample],
) -> FoldOutcome {
    let mut weights = ansatz.weights.clone();
    let mut head = SimpleMlpHead::new(42);
    let mut opt_ansatz = AdamOptimizer::new(weights.len(), 0.025);
    let mut opt_w1 = AdamOptimizer::new_unwrapped(head.w1.len(), 0.005);
    let mut opt_b1 = AdamOptimizer::new_unwrapped(head.b1.len(), 0.005);
    let mut opt_w2 = AdamOptimizer::new_unwrapped(head.w2.len(), 0.005);
    let mut opt_b2 = AdamOptimizer::new_unwrapped(head.b2.len(), 0.005);
    let mut ws = AdjointWorkspace::new(ansatz.dim, ansatz.n_qubits);
    let decay = 1e-4;
    let mut train_acc_pct = 0.0;

    print_table_rule('┌', '┬', '┐');
    println!(
        "│ Ep   │ Trn Loss   │ Trn Acc %  │ Tst Loss  │ Tst Acc % │ Active Nodes │ Time (ms)  │"
    );
    print_table_rule('├', '┼', '┤');
    for epoch in 1..=EPOCHS {
        let t_ep = Instant::now();
        // Cosine annealing learning-rate schedule
        let progress = f64::from(epoch - 1) / f64::from(EPOCHS);
        let lr_mult = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
        opt_ansatz.lr = (0.025 * lr_mult).max(0.005);
        let head_lr = (0.005 * lr_mult).max(0.001);
        for opt in [&mut opt_w1, &mut opt_b1, &mut opt_w2, &mut opt_b2] {
            opt.lr = head_lr;
        }

        let (mut loss_sum, mut correct) = (0.0, 0usize);
        for batch in train.chunks(BATCH_SIZE) {
            let b_len = batch.len() as f64;
            let mut g = batch_gradients(ansatz, &weights, observables, &head, batch, &mut ws);
            loss_sum += g.loss;
            correct += g.correct;
            // Mean gradient with L2 regularization on the head weights
            g.ansatz.iter_mut().for_each(|v| *v /= b_len);
            g.b1.iter_mut().for_each(|v| *v /= b_len);
            g.b2.iter_mut().for_each(|v| *v /= b_len);
            for (v, w) in g.w1.iter_mut().zip(&head.w1) {
                *v = *v / b_len + decay * w;
            }
            for (v, w) in g.w2.iter_mut().zip(&head.w2) {
                *v = *v / b_len + decay * w;
            }
            opt_ansatz.step(&mut weights, &g.ansatz);
            opt_w1.step(&mut head.w1, &g.w1);
            opt_b1.step(&mut head.b1, &g.b1);
            opt_w2.step(&mut head.w2, &g.w2);
            opt_b2.step(&mut head.b2, &g.b2);
        }
        train_acc_pct = 100.0 * correct as f64 / train.len() as f64;

        // Monitoring only: nothing (epoch count, stopping, model choice) depends on these numbers.
        let (test_loss, test_correct, firing, _) =
            evaluate(ansatz, &weights, observables, &head, test);
        println!(
            "│ {:<4} │ {:>10.4} │ {:>9.2}% │ {:>9.4} │ {:>8.2}% │ {:>9}/128 │ {:>10.2} │",
            epoch,
            loss_sum / train.len() as f64,
            train_acc_pct,
            test_loss,
            100.0 * test_correct as f64 / test.len() as f64,
            firing.iter().filter(|&&c| c > 0).count(),
            t_ep.elapsed().as_secs_f64() * 1000.0
        );
    }
    print_table_rule('└', '┴', '┘');

    let (test_loss, test_correct, firing, predictions) =
        evaluate(ansatz, &weights, observables, &head, test);
    FoldOutcome {
        ansatz_weights: weights,
        head,
        train_acc_pct,
        test_loss,
        test_correct,
        firing,
        predictions,
    }
}

fn print_table_rule(left: char, mid: char, right: char) {
    let widths = [6, 12, 12, 11, 11, 14, 12];
    let cells: Vec<String> = widths.iter().map(|&w| "─".repeat(w)).collect();
    println!("{left}{}{right}", cells.join(&mid.to_string()));
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║       T-1000 2D MOLECULAR CORRELATION SPECTROSCOPY (2D COSY) QNN BENCHMARK    ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════╝");

    let t0 = Instant::now();
    let catalog = generate_molecular_catalog();
    // The COSY renderer is deterministic, so a molecule has exactly one spectrum. Splitting
    // replicated spectra would put copies of every test molecule in the training set; the
    // folds below split by molecule instead.
    let molecules: Vec<MolecularSample> = catalog
        .par_iter()
        .map(|&(name, smiles, label)| build_sample(name, smiles, label))
        .collect();
    println!(
        " [INGEST] {} molecules ({} families) rendered in {:.2} ms; {}-fold molecule-disjoint CV",
        molecules.len(),
        N_CLASSES,
        t0.elapsed().as_secs_f64() * 1000.0,
        N_FOLDS
    );

    let ansatz = CircuitAnsatz::new(N_QUBITS, DEPTH).with_pi_layer(true);
    // 80 targeted bipartite observables:
    //   10 Z + 10 X + 45 ZZ + 5 dyadic mirror XX + 5 dyadic mirror YY + 4 adjacent XX + 1 AllZ
    let observables = t1000::chemistry::nmr2d::build_nmr2d_80_observables();
    assert_eq!(observables.len(), N_OBSERVABLES);
    println!(
        " [ARCHITECTURE] {N_QUBITS} qubits, depth {DEPTH}, {N_OBSERVABLES} observables + \
         {N_NMR_SPATIAL} classical features -> {N_HIDDEN} LeakyReLU -> {N_CLASSES} classes"
    );

    let mut folds = Vec::with_capacity(N_FOLDS);
    for fold in 0..N_FOLDS {
        let (held_out, seen): (Vec<_>, Vec<_>) = molecules
            .iter()
            .enumerate()
            .partition(|(i, _)| is_held_out(*i, fold));
        let test: Vec<&MolecularSample> = held_out.into_iter().map(|(_, m)| m).collect();
        let seen: Vec<&MolecularSample> = seen.into_iter().map(|(_, m)| m).collect();
        let train: Vec<&MolecularSample> = (0..seen.len() * TRAIN_REPLICAS)
            .map(|i| seen[i % seen.len()])
            .collect();
        println!(
            "\n [FOLD {}/{}] {} train molecules ({} samples/epoch) | {} held-out molecules",
            fold + 1,
            N_FOLDS,
            seen.len(),
            train.len(),
            test.len()
        );
        let outcome = train_fold(&ansatz, &observables, &train, &test);
        println!(
            " [FOLD {}/{}] held-out accuracy {}/{} ({:.2}%), train accuracy {:.2}%",
            fold + 1,
            N_FOLDS,
            outcome.test_correct,
            test.len(),
            100.0 * outcome.test_correct as f64 / test.len() as f64,
            outcome.train_acc_pct
        );
        folds.push((outcome, test));
    }

    let total_correct: usize = folds.iter().map(|(o, _)| o.test_correct).sum();
    let cv_acc_pct = 100.0 * total_correct as f64 / molecules.len() as f64;
    println!(
        "\n [CROSS-VALIDATION] held-out accuracy {total_correct}/{} molecules = {cv_acc_pct:.2}%",
        molecules.len()
    );
    let (last, last_test) = folds.last().expect("at least one fold");
    let dead = last.firing.iter().filter(|&&c| c == 0).count();
    println!(
        " [NEURON HEALTH, fold {N_FOLDS}] Active {}/128 | Dead {dead}/128",
        128 - dead
    );

    export_telemetry(&ansatz, &observables, &folds, last_test, cv_acc_pct);
}

/// Writes the neural-probe telemetry of the last fold's model on its held-out molecules, plus
/// the cross-validation summary of all folds.
fn export_telemetry(
    ansatz: &CircuitAnsatz,
    observables: &[PauliObservable],
    folds: &[(FoldOutcome, Vec<&MolecularSample>)],
    test_data: &[&MolecularSample],
    cv_acc_pct: f64,
) {
    let (last, _) = folds.last().expect("at least one fold");
    let (weights, head, firing) = (&last.ansatz_weights, &last.head, &last.firing);
    let selected_samples: Vec<serde_json::Value> = test_data
        .iter()
        .map(|sample| {
            let mut phi = sample.state.clone();
            ansatz.forward_in_place(&mut phi, weights);

            let mut q_obs = vec![0.0; N_OBSERVABLES];
            let mut fused = [0.0; N_FUSED];
            for (i, o) in observables.iter().enumerate() {
                let v = o.expectation_value(&phi);
                q_obs[i] = v;
                fused[i] = v;
            }
            fused[N_OBSERVABLES..].copy_from_slice(&sample.spatial);

            let (logits, z1, a1) = head.forward(&fused);
            let (_, _, pred) = softmax_cross_entropy(&logits, sample.label);

            let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_sum: f64 = logits.iter().map(|&l| (l - max_l).exp()).sum();
            let probs: Vec<f64> = logits
                .iter()
                .map(|&l| (l - max_l).exp() / exp_sum)
                .collect();

            serde_json::json!({
                "name": sample.name,
                "smiles": sample.smiles,
                "family": sample.family,
                "label": sample.label,
                "prediction": pred,
                "confidence": probs[pred],
                "probabilities": probs,
                "matrix": sample.matrix.to_vec(),
                "spatial_features": sample.spatial.to_vec(),
                "quantum_observables": q_obs,
                "neuron_preactivations": z1.to_vec(),
                "neuron_activations": a1.to_vec(),
            })
        })
        .collect();

    let mut w2_matrix = vec![vec![0.0; N_HIDDEN]; N_CLASSES];
    for (c, row) in w2_matrix.iter_mut().enumerate() {
        for (h, val) in row.iter_mut().enumerate() {
            *val = head.w2[c * N_HIDDEN + h];
        }
    }

    let mut w1_matrix = vec![vec![0.0; N_FUSED]; N_HIDDEN];
    for (h, row) in w1_matrix.iter_mut().enumerate() {
        for (f, val) in row.iter_mut().enumerate() {
            *val = head.w1[h * N_FUSED + f];
        }
    }

    let fold_reports: Vec<serde_json::Value> = folds
        .iter()
        .enumerate()
        .map(|(k, (o, test))| {
            serde_json::json!({
                "fold": k + 1,
                "held_out_molecules": test.len(),
                "held_out_correct": o.test_correct,
                "held_out_accuracy_pct": 100.0 * o.test_correct as f64 / test.len() as f64,
                "held_out_loss": o.test_loss,
                "train_accuracy_pct": o.train_acc_pct,
                "predictions": o.predictions.iter().map(|(name, label, pred)| {
                    serde_json::json!({ "name": name, "label": label, "prediction": pred })
                }).collect::<Vec<_>>(),
            })
        })
        .collect();

    let report = serde_json::json!({
        "timestamp_rfc3339": utc_timestamp_rfc3339(),
        "evaluation": {
            "protocol": "molecule-disjoint cross-validation: each fold holds out 2 of the 10 \
                         catalog molecules per family; the COSY renderer is deterministic, so \
                         no held-out spectrum or copy of it is seen in training",
            "folds": N_FOLDS,
            "molecules": folds.iter().map(|(_, t)| t.len()).sum::<usize>(),
            "epochs": EPOCHS,
            "train_replicas_per_epoch": TRAIN_REPLICAS,
            "cv_held_out_accuracy_pct": cv_acc_pct,
            "per_fold": fold_reports,
        },
        "probe_fold": N_FOLDS,
        "total_neurons": N_HIDDEN,
        "active_neurons": firing.iter().filter(|&&c| c > 0).count(),
        "dead_neurons": firing.iter().enumerate().filter(|(_, &c)| c == 0).map(|(i, _)| i).collect::<Vec<_>>(),
        "families": FAMILIES,
        "weights_w1": w1_matrix,
        "weights_w2": w2_matrix,
        "samples": selected_samples,
    });

    let json_str = serde_json::to_string_pretty(&report).expect("Serialize telemetry");
    let settings = format!(
        "folds={N_FOLDS} epochs={EPOCHS} replicas={TRAIN_REPLICAS} batch={BATCH_SIZE} \
         depth={DEPTH} split=molecule-disjoint"
    );
    let score = Score {
        metric: "cv_held_out_accuracy_pct",
        value: cv_acc_pct,
        better: Better::Higher,
        guard: None,
        settings: &settings,
    };
    println!("\n [TELEMETRY EXPORT]");
    match t1000::results::write_scored(RESULTS_PROBE_PATH, &json_str, &score) {
        Ok(outcome) => println!("   {RESULTS_PROBE_PATH}: {outcome:?}"),
        Err(e) => eprintln!("   [ERROR] {RESULTS_PROBE_PATH}: {e}"),
    }
    match t1000::results::write(WEB_PROBE_PATH, &json_str) {
        Ok(()) => println!("   {WEB_PROBE_PATH}: written"),
        Err(e) => eprintln!("   [ERROR] {WEB_PROBE_PATH}: {e}"),
    }
}
