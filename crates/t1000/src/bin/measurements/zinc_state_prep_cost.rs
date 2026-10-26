// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! What does it cost to load a ZINC molecule into the circuit's input state?
//!
//! The model amplitude-encodes the 19 (scaled) molecule features into amplitudes 0..18 of the
//! 10-qubit register, so they occupy qubits 0..4 (a 32-entry vector, 19 non-zero) and the other
//! five qubits stay in |0>. This builds the gate-level preparation circuit for every molecule,
//! verifies each one by simulating it against the encoder, and reports the gate counts and the
//! circuit fidelity they imply for a given gate error.
//!
//! `--dataset qm9` (with `--features zinc|qm9`) runs the same count on QM9 molecules.
//!
//! Usage: `zinc_state_prep_cost [--scale maxabs|minmax|raw] [--samples N]`

use std::env;
use t1000::datasets::DataOptions;
use t1000::device_noise::ansatz_fidelity;
use t1000::state::QuantumState;
use t1000::state_prep::{amplitude_prep, amplitude_prep_raw, counts, to_circuit, PrepGate};
use t1000::zinc_data::{InputScaling, Scaler};

const PREP_QUBITS: usize = 5;

/// The encoder's amplitudes: the scaled features, zero padded to 32 entries (the encoder
/// normalises, the preparation does too).
fn amplitudes(features: &[f64]) -> Vec<f64> {
    let mut a = vec![0.0; 1 << PREP_QUBITS];
    a[..features.len()].copy_from_slice(features);
    a
}

/// CNOT-equivalents of a circuit on a linear nearest-neighbour chain, routing each CNOT between
/// qubits at distance `d` by swapping the control next to the target and back (3 CNOTs per SWAP,
/// `2 (d - 1)` SWAPs, plus the CNOT itself). An illustration, not an optimised routing.
fn linear_chain_cnots(gates: &[PrepGate]) -> usize {
    gates
        .iter()
        .map(|g| match *g {
            PrepGate::Cnot { control, target } => 6 * (control.abs_diff(target) - 1) + 1,
            PrepGate::Ry { .. } => 0,
        })
        .sum()
}

fn summary(label: &str, v: &[usize]) {
    let mean = v.iter().sum::<usize>() as f64 / v.len() as f64;
    println!(
        "  {label:<22} mean {mean:>6.2}   min {:>3}   max {:>3}",
        v.iter().min().unwrap(),
        v.iter().max().unwrap()
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let scaling = match value("--scale").as_deref() {
        Some("raw") => InputScaling::Raw,
        Some("minmax") => InputScaling::MinMax,
        _ => InputScaling::MaxAbs,
    };
    let data = DataOptions::from_args(&args).load();
    let (train, test) = (data.train, data.test);
    let fit_rows: Vec<&[f64]> = train.iter().map(|m| m.features.as_slice()).collect();
    let scaler = Scaler::fit_rows(scaling, &fit_rows);

    let (mut raw_c, mut opt_c, mut opt_r, mut chain) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut worst_deviation = 0.0f64;
    for m in train.iter().chain(&test) {
        let amps = amplitudes(&scaler.apply(&m.features));
        let (raw, opt) = (amplitude_prep_raw(&amps), amplitude_prep(&amps));
        raw_c.push(counts(&raw).cnots);
        opt_c.push(counts(&opt).cnots);
        opt_r.push(counts(&opt).rotations);
        chain.push(linear_chain_cnots(&opt));

        // Verify the optimised circuit on the 5-qubit register against the normalised target.
        let mut state = QuantumState::ground(1 << PREP_QUBITS);
        to_circuit(&opt, PREP_QUBITS).compile().execute(&mut state);
        let norm = amps.iter().map(|a| a * a).sum::<f64>().sqrt();
        let dev = state
            .real
            .iter()
            .zip(&amps)
            .map(|(g, a)| (g - a / norm).abs())
            .fold(0.0, f64::max);
        worst_deviation = worst_deviation
            .max(dev)
            .max(state.imag.iter().fold(0.0f64, |x, v| x.max(v.abs())));
    }
    println!("State preparation of the {} input ({scaling:?} inputs), {} molecules, 19 amplitudes on {PREP_QUBITS} qubits\n", data.label, raw_c.len());
    println!("Plain construction: every molecule needs 31 rotations and 30 CNOTs.");
    println!("With free-angle zeroing and CNOT cancellation:");
    summary("CNOTs (2-qubit gates)", &opt_c);
    summary("Ry rotations", &opt_r);
    summary("on a linear chain", &chain);
    println!("\n  largest deviation of any prepared state from the encoder: {worst_deviation:.2e}");

    let mean = |v: &[usize]| v.iter().sum::<usize>() as f64 / v.len() as f64;
    let (rot, cnot) = (mean(&opt_r), mean(&opt_c));
    let stats = |v: &[usize]| serde_json::json!({ "mean": v.iter().sum::<usize>() as f64 / v.len() as f64, "min": v.iter().min(), "max": v.iter().max() });
    let mut fidelity_rows = Vec::new();
    println!(
        "\nCircuit fidelity (global depolarizing model; single-qubit error = two-qubit error / 10)"
    );
    println!("┌────────────┬──────────────┬──────────────┬──────────────┬──────────────┐");
    println!("│ 2q error   │ ansatz F     │ preparation F│ plain prep F │ total F      │");
    println!("├────────────┼──────────────┼──────────────┼──────────────┼──────────────┤");
    for p2 in [1e-3, 5e-3, 1e-2] {
        let p1 = p2 / 10.0;
        let ansatz = ansatz_fidelity(10, 4, p1, p2);
        let prep = (1.0 - p1).powf(rot) * (1.0 - p2).powf(cnot);
        let plain = (1.0 - p1).powi(31) * (1.0 - p2).powi(30);
        println!(
            "│ {p2:<10} │ {ansatz:>12.3} │ {prep:>12.3} │ {plain:>12.3} │ {:>12.3} │",
            ansatz * prep
        );
        fidelity_rows.push(serde_json::json!({ "two_qubit_error": p2, "ansatz": ansatz, "preparation": prep, "plain_preparation": plain, "total": ansatz * prep }));
    }
    println!("└────────────┴──────────────┴──────────────┴──────────────┴──────────────┘");
    let report = serde_json::json!({
        "scaling": format!("{scaling:?}"),
        "molecules": raw_c.len(),
        "preparation_qubits": PREP_QUBITS,
        "plain_construction": { "rotations": 31, "cnots": 30 },
        "optimised_cnots": stats(&opt_c),
        "optimised_rotations": stats(&opt_r),
        "linear_chain_cnot_equivalents": stats(&chain),
        "max_state_deviation": worst_deviation,
        "fidelity": fidelity_rows,
    });
    let name = if data.label.starts_with("QM9") {
        "qm9_state_prep_cost"
    } else {
        "zinc_state_prep_cost"
    };
    let _ = t1000::results::write(
        format!("results/experiments/{name}.json"),
        serde_json::to_string_pretty(&report).unwrap_or_default(),
    );
    println!("\nMean counts to use with zinc_shot_noise: --prep-rotations {rot:.1} --prep-cnots {cnot:.1}");
}
