// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Plumbing check for the circuit feature map plus closed-form linear head: when the targets are
//! exact linear combinations of the circuit's own features, the fitted head must recover them on
//! held-out rows, and a serialized head must predict identically after reloading.

use t1000::baselines::regression_metrics;
use t1000::linear_head::{CircuitFeatureMap, LinearHead, Observables};
use t1000::{CircuitAnsatz, DataEncodingLayer, EncodingKind};

const N_QUBITS: usize = 4;
const N_INPUTS: usize = 8;

/// Deterministic positive pseudo-random inputs.
fn inputs(count: usize) -> Vec<Vec<f64>> {
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        0.1 + (x >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..count).map(|_| (0..N_INPUTS).map(|_| next()).collect()).collect()
}

fn map_and_rows() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let encoding = DataEncodingLayer::new(N_QUBITS, EncodingKind::Amplitude);
    let ansatz = CircuitAnsatz::new(N_QUBITS, 2).with_pi_layer(true);
    let weights = ansatz.weights.clone();
    let map = CircuitFeatureMap { encoding: &encoding, ansatz: &ansatz, weights: &weights, observables: Observables::ZAndZz };
    let x = inputs(300);
    let rows = map.feature_matrix(x.iter().map(Vec::as_slice));
    let again = map.feature_matrix(x.iter().map(Vec::as_slice));
    assert_eq!(rows, again, "feature extraction must be deterministic");
    assert_eq!(rows[0].len(), 4 + 6);
    // Targets: exact linear combinations of three different feature columns.
    let targets = rows
        .iter()
        .map(|f| vec![2.0 * f[3] - f[7] + 0.5, f[1] + 3.0 * f[9], -f[5] + 0.25 * f[0] + 1.0])
        .collect();
    (rows, targets)
}

#[test]
fn head_recovers_targets_that_are_linear_in_the_circuit_features() {
    let (rows, targets) = map_and_rows();
    let head = LinearHead::fit(&rows[..200], &targets[..200]);
    let preds = head.predict(&rows[200..]);
    let metrics = regression_metrics(&preds, &targets[200..], &["a", "b", "c"]);
    for m in &metrics {
        assert!(m.r2 > 0.999, "{}: held-out R2 {} should be ~1 for an exactly linear target", m.target, m.r2);
    }
}

#[test]
fn saved_head_reloads_to_matching_predictions() {
    let (rows, targets) = map_and_rows();
    let head = LinearHead::fit(&rows[..200], &targets[..200]);
    let json = serde_json::to_string(&head).unwrap();
    let reloaded: LinearHead = serde_json::from_str(&json).unwrap();
    // The JSON float parser is not guaranteed bit-exact, so compare to rounding.
    let (a, b) = (head.predict(&rows[200..]), reloaded.predict(&rows[200..]));
    let worst = a.iter().flatten().zip(b.iter().flatten()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    assert!(worst < 1e-12, "reloaded head deviates by {worst}");
}
