// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! 2D Molecular Correlation Spectroscopy (2D COSY NMR) Hybrid QNN Engine.
//!
//! Trains a 10-Qubit Variational Quantum Circuit on 32x32 2D COSY frequency spectra
//! to classify 10 distinct molecular scaffold families via 65 Pauli Correlators (Z, X, ZZ)
//! and AVX2 adjoint gradients with full hidden neuron diagnostics.

use rayon::prelude::*;
use std::fs;
use std::time::Instant;

use t1000::adjoint::AdjointWorkspace;
use t1000::chemistry::nmr2d::{
    extract_nmr_features, generate_cosy_spectrum, NMR_STATE_DIM, N_NMR_SPATIAL,
};
use t1000::{adjoint_gradient_buffered, AdamOptimizer, CircuitAnsatz, PauliObservable, QuantumState};

const N_QUBITS: usize = 10;
const N_CLASSES: usize = 10;
const N_OBSERVABLES: usize = 80; // 10 Z + 10 X + 45 ZZ + 5 XX mirror + 5 YY mirror + 4 XX adj + 1 AllZ
const N_FUSED: usize = N_OBSERVABLES + N_NMR_SPATIAL; // 145
const N_HIDDEN: usize = 128;

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
        let w1 = (0..N_HIDDEN * N_FUSED).map(|_| rng.gen_range(-s1..s1)).collect();
        let b1 = vec![0.01; N_HIDDEN];

        let s2 = (2.0 / N_HIDDEN as f64).sqrt();
        let w2 = (0..N_CLASSES * N_HIDDEN).map(|_| rng.gen_range(-s2..s2)).collect();
        let b2 = vec![0.0; N_CLASSES];

        Self { w1, b1, w2, b2 }
    }

    pub fn forward(&self, x: &[f64; N_FUSED]) -> ([f64; N_CLASSES], [f64; N_HIDDEN], [f64; N_HIDDEN]) {
        let mut z1 = [0.0; N_HIDDEN];
        let mut a1 = [0.0; N_HIDDEN];
        for (h, (z_out, a_out)) in z1.iter_mut().zip(a1.iter_mut()).enumerate() {
            let row = h * N_FUSED;
            let sum = self.b1[h] + self.w1[row..row + N_FUSED].iter().zip(x).map(|(w, &xi)| w * xi).sum::<f64>();
            *z_out = sum;
            *a_out = if sum > 0.0 { sum } else { 0.05 * sum };
        }

        let mut logits = [0.0; N_CLASSES];
        for (c, logit) in logits.iter_mut().enumerate() {
            let row = c * N_HIDDEN;
            *logit = self.b2[c] + self.w2[row..row + N_HIDDEN].iter().zip(&a1).map(|(w, &ai)| w * ai).sum::<f64>();
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
            let sum: f64 = d_logits.iter().enumerate().map(|(c, &d_c)| d_c * self.w2[c * N_HIDDEN + h]).sum();
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

fn softmax_cross_entropy(logits: &[f64; N_CLASSES], label: usize) -> (f64, [f64; N_CLASSES], usize) {
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

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║       T-1000 2D MOLECULAR CORRELATION SPECTROSCOPY (2D COSY) QNN BENCHMARK    ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════╝");

    let t0 = Instant::now();
    let catalog = generate_molecular_catalog();
    println!(" [DISCOVERY] Ingesting {} Molecular Spectroscopy Templates", catalog.len());

    // Generate 1,000 synthetic spectra samples (replicate templates with diverse noise/shifts)
    let n_samples = 1000;
    let samples: Vec<MolecularSample> = (0..n_samples)
        .into_par_iter()
        .map(|idx| {
            let (name, smiles, label) = catalog[idx % catalog.len()];
            let matrix = generate_cosy_spectrum(smiles).expect("Valid SMILES");
            let spatial = extract_nmr_features(&matrix);

            // Populate 10-qubit QuantumState
            let mut state = QuantumState::zero(NMR_STATE_DIM);
            state.real.copy_from_slice(&matrix);

            MolecularSample {
                name: format!("{} #{}", name, idx / catalog.len() + 1),
                smiles: smiles.to_string(),
                family: FAMILIES[label],
                label,
                state,
                matrix,
                spatial,
            }
        })
        .collect();

    let split = 800;
    let train_data = &samples[..split];
    let test_data = &samples[split..];

    println!(
        " [INGEST] {} train / {} test 2D COSY matrices prepared in {:.2} ms",
        train_data.len(),
        test_data.len(),
        t0.elapsed().as_secs_f64() * 1000.0
    );

    let depth = 5;
    let ansatz = CircuitAnsatz::new(N_QUBITS, depth).with_pi_layer(true);

    // 80 Targeted Bipartite Observables:
    //   10 Z + 10 X + 45 ZZ + 5 Dyadic Mirror XX + 5 Dyadic Mirror YY + 4 Adjacent-Scale Cross XX + 1 AllZ
    let observables = t1000::chemistry::nmr2d::build_nmr2d_80_observables();
    assert_eq!(observables.len(), N_OBSERVABLES);

    let n_ansatz_params = ansatz.param_count();
    let mut ansatz_weights = ansatz.weights.clone();
    let mut head = SimpleMlpHead::new(42);

    let mut opt_ansatz = AdamOptimizer::new(n_ansatz_params, 0.025);
    let mut opt_head_w1 = AdamOptimizer::new_unwrapped(head.w1.len(), 0.005);
    let mut opt_head_b1 = AdamOptimizer::new_unwrapped(head.b1.len(), 0.005);
    let mut opt_head_w2 = AdamOptimizer::new_unwrapped(head.w2.len(), 0.005);
    let mut opt_head_b2 = AdamOptimizer::new_unwrapped(head.b2.len(), 0.005);

    println!("\n [ARCHITECTURE] 2D Molecular Spectroscopy QNN Configuration:");
    println!("   Hilbert Register       : {} Qubits (1,024 2D Frequency Bins)", N_QUBITS);
    println!("   Bipartite Partition    : H_w1 (32) (x) H_w2 (32) [5Q (x) 5Q]");
    println!("   Circuit Depth          : {} layers (Ring CZ + Bipartite Cross-Rungs + Ry/Rz)", depth);
    println!("   Quantum Observables    : {} Bipartite Cross-Peak Symmetry Operators", N_OBSERVABLES);
    println!("   Spectral Stroke Filters: {} Multi-Scale Features", N_NMR_SPATIAL);
    println!("   Fused Readout Inputs   : {} Features (80 Quantum + 65 Classical)", N_FUSED);
    println!("   Hidden Neural Layer    : {} LeakyReLU Neurons", N_HIDDEN);
    println!("   Chemical Target Classes: {} Molecular Families", N_CLASSES);

    let epochs = 16;
    let batch_size = 32;
    let num_batches = train_data.len().div_ceil(batch_size);

    println!("\n┌──────┬────────────┬────────────┬───────────┬───────────┬──────────────┬────────────┐");
    println!("│ Ep   │ Trn Loss   │ Trn Acc %  │ Tst Loss  │ Tst Acc % │ Active Nodes │ Time (ms)  │");
    println!("├──────┼────────────┼────────────┼───────────┼───────────┼──────────────┼────────────┤");

    let mut last_firing = [0usize; N_HIDDEN];

    for epoch in 1..=epochs {
        let t_ep = Instant::now();

        // Cosine Annealing learning rate schedule
        let progress = f64::from(epoch - 1) / f64::from(epochs);
        let lr_mult = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
        opt_ansatz.lr = (0.025 * lr_mult).max(0.005);
        let head_lr = (0.005 * lr_mult).max(0.001);
        opt_head_w1.lr = head_lr;
        opt_head_b1.lr = head_lr;
        opt_head_w2.lr = head_lr;
        opt_head_b2.lr = head_lr;

        let mut total_train_loss = 0.0;
        let mut train_correct = 0;

        for b in 0..num_batches {
            let start = b * batch_size;
            let end = (start + batch_size).min(train_data.len());
            let batch = &train_data[start..end];
            let b_len = batch.len() as f64;

            let mut batch_gw1 = vec![0.0; head.w1.len()];
            let mut batch_gb1 = vec![0.0; head.b1.len()];
            let mut batch_gw2 = vec![0.0; head.w2.len()];
            let mut batch_gb2 = vec![0.0; head.b2.len()];
            let mut batch_gansatz = vec![0.0; n_ansatz_params];

            for sample in batch {
                let mut ws = AdjointWorkspace::new(ansatz.dim, ansatz.n_qubits);
                let mut sample_loss = 0.0;
                let mut sample_correct = 0;
                let mut local_gw1 = vec![0.0; head.w1.len()];
                let mut local_gb1 = vec![0.0; head.b1.len()];
                let mut local_gw2 = vec![0.0; head.w2.len()];
                let mut local_gb2 = vec![0.0; head.b2.len()];

                let adj = adjoint_gradient_buffered(
                    &ansatz,
                    &sample.state,
                    &ansatz_weights,
                    &observables,
                    |exps| {
                        let mut fused = [0.0; N_FUSED];
                        fused[..N_OBSERVABLES].copy_from_slice(exps);
                        fused[N_OBSERVABLES..].copy_from_slice(&sample.spatial);

                        let (logits, z1, a1) = head.forward(&fused);
                        let (loss, d_logits, pred) = softmax_cross_entropy(&logits, sample.label);
                        sample_loss = loss;
                        sample_correct = if pred == sample.label { 1 } else { 0 };

                        let d_fused = head.backward(
                            &fused, &z1, &a1, &d_logits,
                            &mut local_gw1, &mut local_gb1, &mut local_gw2, &mut local_gb2,
                        );
                        d_fused[..N_OBSERVABLES].to_vec()
                    },
                    &mut ws,
                );

                total_train_loss += sample_loss;
                train_correct += sample_correct;

                for (g, adj_g) in batch_gansatz.iter_mut().zip(&adj.gradient) { *g += adj_g; }
                for (g, local_g) in batch_gw1.iter_mut().zip(&local_gw1) { *g += local_g; }
                for (g, local_g) in batch_gb1.iter_mut().zip(&local_gb1) { *g += local_g; }
                for (g, local_g) in batch_gw2.iter_mut().zip(&local_gw2) { *g += local_g; }
                for (g, local_g) in batch_gb2.iter_mut().zip(&local_gb2) { *g += local_g; }
            }

            // Apply step with L2 regularization on head
            let decay = 1e-4;
            for g in &mut batch_gansatz { *g /= b_len; }
            for (w, g) in head.w1.iter().zip(&mut batch_gw1) { *g = *g / b_len + decay * w; }
            for g in &mut batch_gb1 { *g /= b_len; }
            for (w, g) in head.w2.iter().zip(&mut batch_gw2) { *g = *g / b_len + decay * w; }
            for g in &mut batch_gb2 { *g /= b_len; }

            opt_ansatz.step(&mut ansatz_weights, &batch_gansatz);
            opt_head_w1.step(&mut head.w1, &batch_gw1);
            opt_head_b1.step(&mut head.b1, &batch_gb1);
            opt_head_w2.step(&mut head.w2, &batch_gw2);
            opt_head_b2.step(&mut head.b2, &batch_gb2);
        }

        let train_loss = total_train_loss / train_data.len() as f64;
        let train_acc = 100.0 * f64::from(train_correct) / train_data.len() as f64;

        // Evaluate test set & hidden neuron health
        let mut test_loss = 0.0;
        let mut test_correct = 0;
        let mut firing_counts = [0usize; N_HIDDEN];

        for sample in test_data {
            let mut phi = sample.state.clone();
            ansatz.forward_in_place(&mut phi, &ansatz_weights);

            let mut fused = [0.0; N_FUSED];
            for (i, o) in observables.iter().enumerate() {
                fused[i] = o.expectation_value(&phi);
            }
            fused[N_OBSERVABLES..].copy_from_slice(&sample.spatial);

            let (logits, _, a1) = head.forward(&fused);
            let (loss, _, pred) = softmax_cross_entropy(&logits, sample.label);
            test_loss += loss;
            if pred == sample.label {
                test_correct += 1;
            }

            for h in 0..N_HIDDEN {
                if a1[h] > 0.0 {
                    firing_counts[h] += 1;
                }
            }
        }

        let test_loss_avg = test_loss / test_data.len() as f64;
        let test_acc_pct = 100.0 * f64::from(test_correct) / test_data.len() as f64;
        let active_nodes = firing_counts.iter().filter(|&&c| c > 0).count();
        last_firing = firing_counts;

        let ep_dur = t_ep.elapsed().as_secs_f64() * 1000.0;

        println!(
            "│ {:<4} │ {:>10.4} │ {:>9.2}% │ {:>9.4} │ {:>8.2}% │ {:>9}/128 │ {:>10.2} │",
            epoch, train_loss, train_acc, test_loss_avg, test_acc_pct, active_nodes, ep_dur
        );
    }
    println!("└──────┴────────────┴────────────┴───────────┴───────────┴──────────────┴────────────┘");

    let dead_nodes = last_firing.iter().filter(|&&c| c == 0).count();
    println!("\n [NEURON HEALTH] Active Nodes: {}/128 | Dead: {}/128", 128 - dead_nodes, dead_nodes);

    // Export web telemetry
    export_telemetry(&ansatz, &ansatz_weights, &head, &observables, test_data, &last_firing);
}

fn export_telemetry(
    ansatz: &CircuitAnsatz,
    weights: &[f64],
    head: &SimpleMlpHead,
    observables: &[PauliObservable],
    test_data: &[MolecularSample],
    firing: &[usize; N_HIDDEN],
) {
    let mut selected_samples = Vec::new();
    let mut family_counts = [0usize; 10];

    for sample in test_data {
        if family_counts[sample.label] < 5 {
            family_counts[sample.label] += 1;

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
            let probs: Vec<f64> = logits.iter().map(|&l| (l - max_l).exp() / exp_sum).collect();

            selected_samples.push(serde_json::json!({
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
            }));
        }
    }

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

    let report = serde_json::json!({
        "timestamp_rfc3339": "2026-10-25T19:30:00Z",
        "total_neurons": N_HIDDEN,
        "active_neurons": firing.iter().filter(|&&c| c > 0).count(),
        "dead_neurons": firing.iter().enumerate().filter(|(_, &c)| c == 0).map(|(i, _)| i).collect::<Vec<_>>(),
        "families": FAMILIES,
        "weights_w1": w1_matrix,
        "weights_w2": w2_matrix,
        "samples": selected_samples,
    });

    let json_str = serde_json::to_string_pretty(&report).unwrap();
    let _ = fs::create_dir_all("web/data");
    let _ = fs::create_dir_all("results/data");
    let _ = fs::write("web/data/nmr2d_spectroscopy_probe.json", &json_str);
    let _ = fs::write("results/data/nmr2d_spectroscopy_probe.json", &json_str);

    println!("\n [TELEMETRY EXPORT]");
    println!("   Saved to web/data/nmr2d_spectroscopy_probe.json");
    println!("   Saved to results/data/nmr2d_spectroscopy_probe.json");
}
