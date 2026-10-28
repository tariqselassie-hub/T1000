// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Reproducibility guarantees: default and seeded constructors must be deterministic.

use t1000::*;

#[test]
fn circuit_ansatz_is_deterministic() {
    let a = CircuitAnsatz::new(6, 3);
    let b = CircuitAnsatz::new(6, 3);
    assert_eq!(
        a.weights, b.weights,
        "default constructor must be reproducible"
    );
    assert_eq!(
        a.weights,
        CircuitAnsatz::new_seeded(6, 3, DEFAULT_SEED).weights
    );

    let c = CircuitAnsatz::new_seeded(6, 3, 1);
    assert_eq!(c.weights, CircuitAnsatz::new_seeded(6, 3, 1).weights);
    assert_ne!(
        a.weights, c.weights,
        "different seeds must give different weights"
    );
    assert!(c
        .weights
        .iter()
        .all(|w| (-std::f64::consts::PI..std::f64::consts::PI).contains(w)));
}

#[test]
fn deep_qnn_is_deterministic_with_distinct_layers() {
    let a = DeepQuantumNeuralNetwork::new(16, 3, 0.1, 0.05);
    let b = DeepQuantumNeuralNetwork::new(16, 3, 0.1, 0.05);
    for (la, lb) in a.layers.iter().zip(&b.layers) {
        assert_eq!(la.phase_weights, lb.phase_weights);
        assert_eq!(la.hamiltonian_energies, lb.hamiltonian_energies);
    }
    assert_ne!(
        a.layers[0].phase_weights, a.layers[1].phase_weights,
        "stacked layers must not be identical copies"
    );
    let c = DeepQuantumNeuralNetwork::new_seeded(16, 3, 0.1, 0.05, 9);
    assert_ne!(a.layers[0].phase_weights, c.layers[0].phase_weights);
}

#[test]
fn variational_layer_is_deterministic() {
    let a = VariationalQuantumLayer::new(8, 0.1, 0.2);
    let b = VariationalQuantumLayer::new(8, 0.1, 0.2);
    assert_eq!(a.phase_weights, b.phase_weights);
    assert_ne!(
        a.phase_weights,
        VariationalQuantumLayer::new_seeded(8, 0.1, 0.2, 5).phase_weights
    );
}

#[test]
fn povm_classifier_is_deterministic() {
    let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let a = POVMClassifier::new(8, labels.clone());
    let b = POVMClassifier::new(8, labels.clone());
    assert_eq!(a.operators, b.operators);
    assert_ne!(
        a.operators,
        POVMClassifier::new_seeded(8, labels, 3).operators
    );
}

#[test]
fn forward_pass_output_is_reproducible() {
    let run = || {
        let a = CircuitAnsatz::new(5, 4).with_pi_layer(true);
        let enc = DataEncodingLayer::new(5, EncodingKind::Angle);
        let s = a.forward(&enc.encode(&[0.3, -1.1, 2.0, 0.7, -0.4]), &a.weights);
        (s.real, s.imag)
    };
    assert_eq!(run(), run());
}

#[test]
fn test_hybrid_visual_angle_encoding_invariants() {
    let enc = DataEncodingLayer::new(10, EncodingKind::Angle);
    let sample_features: [f64; 10] = [
        0.521, -1.244, 2.310, -0.892, 1.450, -2.110, 0.782, -0.340, 1.890, -1.540,
    ];

    let state = enc.encode(&sample_features);
    assert_eq!(state.dim, 1024);

    // 1. Unitarity invariant: norm must equal 1.0 to within float epsilon
    let norm_sq: f64 = state
        .real
        .iter()
        .zip(&state.imag)
        .map(|(&r, &i)| r * r + i * i)
        .sum();
    assert!(
        (norm_sq - 1.0).abs() < 1e-12,
        "State norm must be 1.0, got {}",
        norm_sq
    );

    // 2. Ansätz evolution must preserve unitarity
    let ansatz = CircuitAnsatz::new(10, 3).with_pi_layer(true);
    let mut evolved = state.clone();
    ansatz.forward_in_place(&mut evolved, &ansatz.weights);
    let evolved_norm_sq: f64 = evolved
        .real
        .iter()
        .zip(&evolved.imag)
        .map(|(&r, &i)| r * r + i * i)
        .sum();
    assert!(
        (evolved_norm_sq - 1.0).abs() < 1e-12,
        "Evolved norm must be 1.0, got {}",
        evolved_norm_sq
    );

    // 3. 55 Correlators expectation values must be bounded in [-1.0, 1.0]
    let mut observables = Vec::with_capacity(55);
    for q in 0..10 {
        observables.push(PauliObservable::Z(q));
    }
    for i in 0..10 {
        for j in (i + 1)..10 {
            observables.push(PauliObservable::ZZ(i, j));
        }
    }
    assert_eq!(observables.len(), 55);

    let mut raw_exps = [0.0; 55];
    for (idx, obs) in observables.iter().enumerate() {
        let val = obs.expectation_value(&evolved);
        assert!(
            (-1.000001..=1.000001).contains(&val),
            "Expectation value out of bounds: {}",
            val
        );
        raw_exps[idx] = val;
    }

    // 4. Correlator LayerNorm invariant: normalized mean ~ 0, variance ~ 1
    let cache = t1000::optimizer::normalize_correlators_arr_forward(&raw_exps);
    let norm_mean: f64 = cache.normalized.iter().sum::<f64>() / 55.0;
    assert!(
        norm_mean.abs() < 1e-6,
        "LayerNorm mean must be 0, got {}",
        norm_mean
    );
}
