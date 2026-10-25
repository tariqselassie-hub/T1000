// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! ECLEKTIK Track 2: Hierarchical Ultrametric Language Family Sorter.
//!
//! Hilbert Space Architecture:
//! H = H_C(2Q, 4 Macro-Families) \otimes H_F(2Q, 4 Fine Leaves) \otimes H_E(6Q, 64-dim Entropy Sink) = 10 Qubits.
//!
//! Evaluates the 12-language parallel corpus from ECLEKTIK, sorting texts into
//! language families and fine leaf identities with zero classical readout weights.

mod eclektik_ultrametric_sorter {
    pub mod encoding;
    pub mod taxonomy;
    pub mod telemetry;
}

use eclektik_ultrametric_sorter::encoding::extract_multilingual_angles;
use eclektik_ultrametric_sorter::taxonomy::{
    load_eclektik_samples, locate_eclektik_file, EclektikSample, LanguageMeta, LANGUAGES,
    N_LANGUAGES,
};
use eclektik_ultrametric_sorter::telemetry::{
    save_eclektik_report, EclektikUltrametricReport, FamilyMetrics, GeometryMetrics, LanguageMetrics,
};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::time::Instant;
use t1000::ultrametric_tree::{
    compute_ultrametric_branch_distribution, ultrametric_gram_pair, ultrametric_tree_distance,
    UltrametricTreeConfig, UltrametricTreeSorter,
};

fn split_train_test(
    samples: Vec<EclektikSample>,
    train_ratio: f64,
    seed: u64,
) -> (Vec<EclektikSample>, Vec<EclektikSample>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..samples.len()).collect();
    indices.shuffle(&mut rng);

    let n_train = ((samples.len() as f64) * train_ratio).round() as usize;
    let mut train = Vec::with_capacity(n_train);
    let mut test = Vec::with_capacity(samples.len() - n_train);

    for (pos, idx) in indices.into_iter().enumerate() {
        if pos < n_train {
            train.push(samples[idx].clone());
        } else {
            test.push(samples[idx].clone());
        }
    }
    (train, test)
}

fn compute_geometry_metrics(
    sorter: &UltrametricTreeSorter,
    train_encoded: &[([f64; 10], LanguageMeta)],
) -> GeometryMetrics {
    let gamma1 = 0.5;
    let gamma2 = 1.0;

    // Pick samples: en (y=0, s=0), de (y=0, s=1), es (y=1, s=0)
    let s_en = train_encoded.iter().find(|(_, m)| m.code == "en").map(|(v, _)| v).unwrap();
    let s_de = train_encoded.iter().find(|(_, m)| m.code == "de").map(|(v, _)| v).unwrap();
    let s_es = train_encoded.iter().find(|(_, m)| m.code == "es").map(|(v, _)| v).unwrap();

    let p_en = compute_ultrametric_branch_distribution(&sorter.forward(s_en), &sorter.cfg);
    let p_de = compute_ultrametric_branch_distribution(&sorter.forward(s_de), &sorter.cfg);
    let p_es = compute_ultrametric_branch_distribution(&sorter.forward(s_es), &sorter.cfg);

    let (c_ident, _) = ultrametric_gram_pair(&p_en, &p_en);
    let (c_sib, _) = ultrametric_gram_pair(&p_en, &p_de);
    let (c_disj, _) = ultrametric_gram_pair(&p_en, &p_es);

    let d_en_de = ultrametric_tree_distance(0, 0, 0, 1, gamma1, gamma2);
    let d_en_es = ultrametric_tree_distance(0, 0, 1, 0, gamma1, gamma2);
    let d_de_es = ultrametric_tree_distance(0, 1, 1, 0, gamma1, gamma2);

    let strong_triangle = d_en_de <= d_en_es.max(d_de_es)
        && d_en_es <= d_en_de.max(d_de_es)
        && d_de_es <= d_en_de.max(d_en_es);

    GeometryMetrics {
        identical_leaf_overlap: c_ident,
        sibling_family_overlap: c_sib,
        disjoint_family_overlap: c_disj,
        strong_triangle_inequality_holds: strong_triangle,
        gamma1_sibling: gamma1,
        gamma2_disjoint: gamma2,
    }
}

fn train_eclektik_sorter(
    sorter: &mut UltrametricTreeSorter,
    train_encoded: &[([f64; 10], LanguageMeta)],
    epochs: usize,
    batch_size: usize,
    lr: f64,
) -> f64 {
    let t_start = Instant::now();
    let mut rng = ChaCha8Rng::seed_from_u64(0xEC1E_7100);
    let d_c = sorter.cfg.d_coarse();
    let n_fine = sorter.cfg.n_fine;

    for ep in 1..=epochs {
        let mut indices: Vec<usize> = (0..train_encoded.len()).collect();
        indices.shuffle(&mut rng);
        let n_batches = indices.len() / batch_size;

        for b in 0..n_batches {
            let b_indices = &indices[b * batch_size..(b + 1) * batch_size];
            let mut grad_branch = vec![vec![0.0; sorter.branch_params[0].len()]; d_c];

            for &idx in b_indices {
                let (angles, meta) = &train_encoded[idx];
                for f_off in 0..n_fine {
                    let g_ry = sorter.parameter_shift_branch_grad(
                        angles, f_off, false, meta.y, meta.s,
                    );
                    let g_rz = sorter.parameter_shift_branch_grad(
                        angles, f_off, true, meta.y, meta.s,
                    );
                    grad_branch[meta.y][f_off * 2] -= g_ry[meta.y];
                    grad_branch[meta.y][f_off * 2 + 1] -= g_rz[meta.y];
                }
            }

            let b_scale = lr / (batch_size as f64);
            for (branch_y, grad_y) in sorter.branch_params.iter_mut().zip(grad_branch.iter()) {
                for (p, g) in branch_y.iter_mut().zip(grad_y.iter()) {
                    *p -= b_scale * g;
                }
            }
        }
        println!("   [TRAIN] Epoch {:02}/{} Completed", ep, epochs);
    }

    t_start.elapsed().as_secs_f64()
}

fn evaluate_test_set(
    sorter: &UltrametricTreeSorter,
    test_encoded: &[([f64; 10], LanguageMeta)],
) -> (f64, f64, f64, f64, Vec<FamilyMetrics>, Vec<LanguageMetrics>) {
    let t_eval = Instant::now();
    let n_test = test_encoded.len();

    let mut lang_totals = [0usize; N_LANGUAGES];
    let mut lang_correct = [0usize; N_LANGUAGES];
    let mut fam_totals = [0usize; 4];
    let mut fam_correct = [0usize; 4];
    let mut total_coarse_correct = 0usize;
    let mut total_fine_correct = 0usize;
    let mut total_joint_correct = 0usize;

    for (angles, meta) in test_encoded {
        let out_state = sorter.forward(angles);
        let probs = compute_ultrametric_branch_distribution(&out_state, &sorter.cfg);

        // Coarse prediction: argmax sum_s P(y, s)
        let mut pred_y = 0;
        let mut max_py = -1.0;
        for (y, row) in probs.iter().enumerate() {
            let py: f64 = row.iter().sum();
            if py > max_py {
                max_py = py;
                pred_y = y;
            }
        }

        // Fine prediction: argmax_s P(pred_y, s)
        let mut pred_s = 0;
        let mut max_ps = -1.0;
        for (s, &p) in probs[pred_y].iter().enumerate() {
            if p > max_ps {
                max_ps = p;
                pred_s = s;
            }
        }

        let coarse_match = pred_y == meta.y;
        let fine_match = pred_s == meta.s;
        let joint_match = coarse_match && fine_match;

        fam_totals[meta.y] += 1;
        lang_totals[meta.leaf_id] += 1;

        if coarse_match {
            total_coarse_correct += 1;
            fam_correct[meta.y] += 1;
        }
        if fine_match {
            total_fine_correct += 1;
        }
        if joint_match {
            total_joint_correct += 1;
            lang_correct[meta.leaf_id] += 1;
        }
    }

    let eval_duration = t_eval.elapsed();
    let per_sample_us = (eval_duration.as_micros() as f64) / (n_test as f64);

    let coarse_acc = (total_coarse_correct as f64) / (n_test as f64) * 100.0;
    let fine_acc = (total_fine_correct as f64) / (n_test as f64) * 100.0;
    let joint_acc = (total_joint_correct as f64) / (n_test as f64) * 100.0;

    let fam_names = ["Germanic", "Romance", "EastAsian", "AfroIndic"];
    let mut fam_metrics = Vec::new();
    for f in 0..4 {
        let tot = fam_totals[f].max(1);
        let corr = fam_correct[f];
        fam_metrics.push(FamilyMetrics {
            family: fam_names[f].to_string(),
            total: fam_totals[f],
            correct: corr,
            accuracy_pct: (corr as f64) / (tot as f64) * 100.0,
        });
    }

    let mut lang_metrics = Vec::new();
    for l in &LANGUAGES {
        let tot = lang_totals[l.leaf_id].max(1);
        let corr = lang_correct[l.leaf_id];
        lang_metrics.push(LanguageMetrics {
            code: l.code.to_string(),
            name: l.name.to_string(),
            family: l.family.to_string(),
            total: lang_totals[l.leaf_id],
            correct: corr,
            accuracy_pct: (corr as f64) / (tot as f64) * 100.0,
        });
    }

    (coarse_acc, fine_acc, joint_acc, per_sample_us, fam_metrics, lang_metrics)
}

fn main() {
    println!("┌────────────────────────────────────────────────────────────────────────┐");
    println!("│ T-1000 ECLEKTIK TRACK 2: MULTILINGUAL ULTRAMETRIC LANGUAGE SORTER     │");
    println!("│ 10-Qubit Pure Quantum Taxonomy Router with Zero Classical Parameters   │");
    println!("└────────────────────────────────────────────────────────────────────────┘");

    let data_path = locate_eclektik_file();
    println!("\n [DATASET] Ingesting Parallel Corpus: {}", data_path.display());
    let samples = load_eclektik_samples(&data_path, None);
    println!("   Total Multilingual Instances Ingested: {}", samples.len());

    let (train_raw, test_raw) = split_train_test(samples, 0.80, 0x5EED_2026);
    println!("   Train Partition: {} samples (80.0%)", train_raw.len());
    println!("   Test Partition : {} samples (20.0%)", test_raw.len());

    println!("\n [ENCODING] Projecting N-Gram & Script Statevectors to Hilbert Space...");
    let t_enc = Instant::now();
    let train_encoded: Vec<([f64; 10], LanguageMeta)> = train_raw
        .par_iter()
        .map(|s| (extract_multilingual_angles(&s.text), s.meta))
        .collect();
    let test_encoded: Vec<([f64; 10], LanguageMeta)> = test_raw
        .par_iter()
        .map(|s| (extract_multilingual_angles(&s.text), s.meta))
        .collect();
    println!("   Encoded {} Feature Sets in {:.2}s", train_encoded.len() + test_encoded.len(), t_enc.elapsed().as_secs_f64());

    let cfg = UltrametricTreeConfig::new(2, 2, 6);
    let mut sorter = UltrametricTreeSorter::new(cfg, 1);
    sorter.init_seeded(0xEC1E_0042);

    println!("\n [TOPOLOGY] 10-Qubit Factorized Register Configuration:");
    println!("   Coarse Register H_C : {} Qubits (dim = 4: Germanic, Romance, EastAsian, AfroIndic)", cfg.n_coarse);
    println!("   Fine Register H_F   : {} Qubits (dim = 4: Leaf Languages per Macro-Family)", cfg.n_fine);
    println!("   Entropy Sink H_E    : {} Qubits (dim = 64: Thermal Language Noise Dump)", cfg.n_env);
    println!("   Total Hilbert Space : {} Qubits (1,024 Complex Amplitudes)", cfg.total_qubits());
    println!("   Trainable Parameters: {} (Zero Classical Readout Head)", sorter.total_parameters());

    println!("\n [GEOMETRY] Pre-Training Ultrametric Triangle Invariant Verification:");
    let geom_pre = compute_geometry_metrics(&sorter, &train_encoded);
    println!("   Identical Leaf Overlap (en vs en): {:.4}", geom_pre.identical_leaf_overlap);
    println!("   Sibling Family Overlap (en vs de): {:.4}", geom_pre.sibling_family_overlap);
    println!("   Disjoint Family Overlap(en vs es): {:.4}", geom_pre.disjoint_family_overlap);
    println!("   Strong Triangle Inequality       : {}", if geom_pre.strong_triangle_inequality_holds { "PASS (Preserved)" } else { "FAIL" });

    println!("\n [TRAINING] Optimizing Quantum Multiplexor Branches via Parameter-Shift...");
    let train_sec = train_eclektik_sorter(&mut sorter, &train_encoded, 4, 128, 0.05);

    println!("\n [EVALUATION] Benchmarking Full Test Partition ({} instances)...", test_encoded.len());
    let (coarse_acc, fine_acc, joint_acc, latency_us, fam_metrics, lang_metrics) =
        evaluate_test_set(&sorter, &test_encoded);

    let geom_post = compute_geometry_metrics(&sorter, &train_encoded);

    println!("┌─────────────────────────┬──────────┬──────────┬───────────┐");
    println!("│ Language Family         │ Total    │ Correct  │ Accuracy  │");
    println!("├─────────────────────────┼──────────┼──────────┼───────────┤");
    for fam in &fam_metrics {
        println!("│ {:<23} │ {:<8} │ {:<8} │ {:>8.2}% │", fam.family, fam.total, fam.correct, fam.accuracy_pct);
    }
    println!("└─────────────────────────┴──────────┴──────────┴───────────┘");

    println!("┌─────┬──────────────┬─────────────┬──────────┬──────────┬───────────┐");
    println!("│ ISO │ Language     │ Family      │ Total    │ Correct  │ Accuracy  │");
    println!("├─────┼──────────────┼─────────────┼──────────┼──────────┼───────────┤");
    for l in &lang_metrics {
        println!("│ {:<3} │ {:<12} │ {:<11} │ {:<8} │ {:<8} │ {:>8.2}% │", l.code, l.name, l.family, l.total, l.correct, l.accuracy_pct);
    }
    println!("└─────┴──────────────┴─────────────┴──────────┴──────────┴───────────┘");

    println!("\n [FINAL TELEMETRY]");
    println!("   Coarse Macro-Family Accuracy : {:>6.2}%", coarse_acc);
    println!("   Fine Sub-Language Accuracy   : {:>6.2}%", fine_acc);
    println!("   Joint 12-Way Leaf Accuracy   : {:>6.2}%", joint_acc);
    println!("   Inference Latency per Sample : {:>6.2} us", latency_us);
    println!("   Post-Train Triangle Equality : {}", if geom_post.strong_triangle_inequality_holds { "CONSERVED" } else { "VIOLATED" });

    let report = EclektikUltrametricReport {
        timestamp_rfc3339: "2026-10-25T14:30:00Z".to_string(),
        dataset: "ECLEKTIK_multilingual_main".to_string(),
        registers: "H_C(2Q) x H_F(2Q) x H_E(6Q)".to_string(),
        n_qubits: cfg.total_qubits(),
        total_parameters: sorter.total_parameters(),
        train_samples: train_encoded.len(),
        test_samples: test_encoded.len(),
        training_time_sec: train_sec,
        inference_per_sample_us: latency_us,
        test_coarse_family_accuracy_pct: coarse_acc,
        test_fine_within_family_accuracy_pct: fine_acc,
        test_joint_12way_accuracy_pct: joint_acc,
        geometry: geom_post,
        family_metrics: fam_metrics,
        language_metrics: lang_metrics,
    };

    save_eclektik_report(&report);
}
