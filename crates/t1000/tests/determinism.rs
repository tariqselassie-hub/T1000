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
