// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! How predictable are the QM9 properties from SMILES descriptors?
//!
//! Fits mean, linear-ridge and quadratic-ridge baselines for all nine properties on two 19-column
//! descriptor sets (the ZINC pipeline's columns, and a QM9 variant with hydrogen counts) and prints
//! held-out R2, to choose the targets of the quantum feature-map study from measured difficulty.
//!
//! Usage: `qm9_explore [--train N] [--test N] [--seed S]`

use std::{env, time::Instant};
use t1000::baselines::evaluate_baselines;
use t1000::qm9_data::{self, FeatureSet, PROPERTIES};

const CSV: &str = "data/qm9/qm9.csv";

fn r2_by_model(
    set: FeatureSet,
    train_n: usize,
    test_n: usize,
    seed: u64,
) -> Vec<(String, Vec<f64>)> {
    let t = Instant::now();
    let (rows, skipped) = qm9_data::load(CSV, set, usize::MAX).expect("load QM9");
    println!(
        "  {set:?}: {} molecules featurised in {:.1} s ({skipped} skipped)",
        rows.len(),
        t.elapsed().as_secs_f64()
    );
    let (mut train, test) = qm9_data::split(rows, test_n, seed);
    train.truncate(train_n);
    let pair = |m: &qm9_data::Qm9Molecule| (m.features.clone(), m.properties.to_vec());
    let (tr, te): (Vec<_>, Vec<_>) = (
        train.iter().map(pair).collect(),
        test.iter().map(pair).collect(),
    );
    println!(
        "  {set:?}: fitting ridge on {} training / {} held-out molecules",
        tr.len(),
        te.len()
    );
    evaluate_baselines(&tr, &te, &PROPERTIES)
        .into_iter()
        .map(|b| (b.model, b.metrics.iter().map(|m| m.r2).collect()))
        .collect()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let value = |flag: &str, default: usize| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let (train_n, test_n, seed) = (
        value("--train", 20_000),
        value("--test", 10_000),
        value("--seed", 7) as u64,
    );
    println!("QM9 baselines from 19 SMILES descriptors\n");
    let zinc = r2_by_model(FeatureSet::ZincLike, train_n, test_n, seed);
    let qm9 = r2_by_model(FeatureSet::Qm9, train_n, test_n, seed);

    let get = |set: &[(String, Vec<f64>)], name: &str, i: usize| {
        set.iter()
            .find(|(m, _)| m == name)
            .map_or(f64::NAN, |(_, v)| v[i])
    };
    println!("\nHeld-out R2 (1.0 = perfect, 0 = no better than the mean)");
    println!("┌──────────┬───────────────────────────┬───────────────────────────┐");
    println!("│          │ ZINC-like descriptors     │ QM9 descriptors (with H)  │");
    println!("│ property │ linear ridge  quadratic   │ linear ridge  quadratic   │");
    println!("├──────────┼───────────────────────────┼───────────────────────────┤");
    for (i, name) in PROPERTIES.iter().enumerate() {
        println!(
            "│ {name:<8} │ {:>12.3}  {:>9.3}   │ {:>12.3}  {:>9.3}   │",
            get(&zinc, "Ridge (linear)", i),
            get(&zinc, "Ridge (quadratic)", i),
            get(&qm9, "Ridge (linear)", i),
            get(&qm9, "Ridge (quadratic)", i)
        );
    }
    println!("└──────────┴───────────────────────────┴───────────────────────────┘");
}
