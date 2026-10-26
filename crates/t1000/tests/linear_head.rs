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
    (0..count)
        .map(|_| (0..N_INPUTS).map(|_| next()).collect())
        .collect()
}

fn map_and_rows() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let encoding = DataEncodingLayer::new(N_QUBITS, EncodingKind::Amplitude);
    let ansatz = CircuitAnsatz::new(N_QUBITS, 2).with_pi_layer(true);
    let weights = ansatz.weights.clone();
    let map = CircuitFeatureMap {
        encoding: &encoding,
        ansatz: &ansatz,
        weights: &weights,
        observables: Observables::ZAndZz,
    };
    let x = inputs(300);
    let rows = map.feature_matrix(x.iter().map(Vec::as_slice));
    let again = map.feature_matrix(x.iter().map(Vec::as_slice));
    assert_eq!(rows, again, "feature extraction must be deterministic");
    assert_eq!(rows[0].len(), 4 + 6);
    // Targets: exact linear combinations of three different feature columns.
    let targets = rows
        .iter()
        .map(|f| {
            vec![
                2.0 * f[3] - f[7] + 0.5,
                f[1] + 3.0 * f[9],
                -f[5] + 0.25 * f[0] + 1.0,
            ]
        })
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
        assert!(
            m.r2 > 0.999,
            "{}: held-out R2 {} should be ~1 for an exactly linear target",
            m.target,
            m.r2
        );
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
    let worst = a
        .iter()
        .flatten()
        .zip(b.iter().flatten())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    assert!(worst < 1e-12, "reloaded head deviates by {worst}");
}

/// Largest fraction of any circuit feature's variance left unexplained when it is regressed on all
/// monomials up to `degree` of the normalised input (fitted on 450 inputs, scored on 150 others).
fn unexplained_variance(degree: usize) -> f64 {
    use t1000::linear_head::LinearHead;
    use t1000::polynomial::polynomial_features;

    let encoding = DataEncodingLayer::new(5, EncodingKind::Amplitude);
    let ansatz = CircuitAnsatz::new(5, 3).with_pi_layer(true);
    let weights = ansatz.weights.clone();
    let map = CircuitFeatureMap {
        encoding: &encoding,
        ansatz: &ansatz,
        weights: &weights,
        observables: Observables::ParityUpTo(4),
    };
    let inputs: Vec<Vec<f64>> = inputs(600).into_iter().map(|v| v[..6].to_vec()).collect();
    let features = map.feature_matrix(inputs.iter().map(Vec::as_slice));
    let monomials: Vec<Vec<f64>> = inputs
        .iter()
        .map(|x| {
            let n = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            polynomial_features(&x.iter().map(|v| v / n).collect::<Vec<_>>(), degree)
        })
        .collect();
    let head = LinearHead::fit(&monomials[..450], &features[..450]);
    let predicted = head.predict(&monomials[450..]);
    let mut worst = 0.0f64;
    for k in 0..features[0].len() {
        let truth: Vec<f64> = features[450..].iter().map(|r| r[k]).collect();
        let mean = truth.iter().sum::<f64>() / truth.len() as f64;
        let var = truth.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
        if var < 1e-20 {
            continue; // a constant feature is trivially reproduced
        }
        let err: f64 = predicted
            .iter()
            .zip(&truth)
            .map(|(p, t)| (p[k] - t).powi(2))
            .sum();
        worst = worst.max(err / var);
    }
    worst
}

/// Every Z-string expectation of an amplitude-encoded input passed through a fixed circuit is a
/// quadratic form in the normalised input vector, so the span of the circuit features lies inside
/// the span of the degree-2 monomials of that vector (plus a constant). This is why register size and
/// circuit depth cannot lift the circuit head above a quadratic model.
#[test]
fn circuit_features_are_quadratic_forms_in_the_normalised_input() {
    let worst = unexplained_variance(2);
    assert!(
        worst < 1e-8,
        "a circuit feature left {worst:e} of its variance unexplained by quadratic forms"
    );
}

/// Negative control for the test above: degree-1 monomials alone must leave a large share of some
/// feature unexplained, so the quadratic result is not an artefact of the regression.
#[test]
fn linear_monomials_alone_do_not_explain_the_circuit_features() {
    let worst = unexplained_variance(1);
    assert!(
        worst > 0.05,
        "only {worst:e} left unexplained by linear terms; the control is not discriminating"
    );
}
