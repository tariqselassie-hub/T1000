// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # GuddaLM Quantum Neural Network (`t1000`)
//!
//! Solidified Dual-Mission Architecture:
//! 1. **Simulating Advanced RCS & Pathed Circuits**: Arbitrary circuit compilation (`compiler`),
//!    fused AVX2+FMA state evolution (`simd`, `tiled_butterfly`), and Random Circuit Sampling (`rcs`)
//!    with exact Porter-Thomas verification.
//! 2. **Pure SMILES String Exploration**: Zero-overhead chemical featurization, quantum state
//!    encoding, and VQC molecular property screening & latent space exploration (`smiles`).

pub mod adjoint;
pub mod ansatz;
pub mod baselines;
pub mod canonical;
pub mod coherence;
pub mod compiler;
pub mod diagonal;
pub mod feature_map;
pub mod heptal_wave;
pub mod ingest;
pub mod lattice;
pub mod measurements;
pub mod memory;
pub mod molfeatures;
pub mod optimizer;
pub mod parallel;
pub mod protocol;
pub mod qec;
pub mod rcs;
pub mod simd;
pub mod smiles;
pub mod state;
pub mod tensor_network;
pub mod tiled_butterfly;
pub mod trainer;
pub mod zipper;
pub mod qssm;
pub mod telemetry;

pub use telemetry::QuantumTelemetryReport;

// Backwards-compatible module aliases
pub use measurements::observable;
pub use measurements::povm_readout;

pub use adjoint::{adjoint_gradient, AdjointResult};
pub use ansatz::{CircuitAnsatz, DeepQuantumNeuralNetwork, PiPhaseLayer, PreInitCircuit, RademacherPhaseLayer, VariationalQuantumLayer};
pub use canonical::{
    build_bernstein_vazirani, build_grover_diffusion, build_hea_layer,
    build_quantum_phase_estimation, build_trotter_ising_step, mitigate_readout_errors,
    CanonicalAlgorithm, CanonicalExecutionReport, CanonicalRunner, EntanglerTopology,
    ReadoutConfusionMatrix,
};
pub use coherence::{
    apply_core_channel_on_qubit, apply_dynamical_decoupling, apply_open_system_decoherence,
    apply_open_system_decoherence_qubit, compute_coherence_metrics, reproject_coherence_manifold,
    CoherenceMetrics, DynamicalDecouplingKind, NoiseChannelParams, QuantumZenoSentry, ZenoTelemetry,
};
pub use compiler::{
    compute_asap_alap, schedule_circuit_dag, schedule_circuit_dag_with_coherence_dd,
    AsapAlapSchedule, CircuitBuilder, CircuitDag, CircuitOp, CoherenceScheduleTelemetry,
    CompiledCircuit, CompilerTelemetry, ScheduledStage,
};
pub use qec::{
    detect_and_recover, encode_logical_state, extract_syndrome, inject_pauli_error,
    PauliErrorKind, QecCodeKind, QecRecoveryResult, SyndromeReport,
};
pub use feature_map::{DataEncodingLayer, EncodingKind};
pub use heptal_wave::{
    evolve_heptal_wave_cycle, HeptalWaveState, HeptalZipperConfig, HeptalZipperEngine,
    HeptalZipperReport, HeptalZipperStepTelemetry,
};
pub use ingest::{tokenize_text, CorpusIngestor, IngestedDoc, IngestionTelemetry};
pub use lattice::{
    extract_lattice_frame, render_lattice_ascii, BoundaryCondition, LatticeConfig, LatticeFrame,
    LatticeTrotterEvolver, LatticeVqeEngine,
};
pub use measurements::{PauliObservable, POVMClassifier};
pub use memory::{MemoryEntry, QuantumAssociativeMemory};
pub use optimizer::{AdamOptimizer, ParameterShiftEngine};
pub use rcs::{DistributionMetrics, RcsConfig, RcsRunReport, RcsSimulator};
pub use smiles::{
    BondKind, ParsedSmiles, SmilesAtom, SmilesError, SmilesExplorer, SmilesFeatures, SmilesFeaturizer,
    SmilesPropertyProfile, SmilesQuantumEncoder, SmilesToken,
};
pub use state::{QuantumFeatureMap, QuantumState};
pub use tiled_butterfly::{apply_layer_fused, tiled_apply_cz, tiled_apply_ry, tiled_apply_rz};
pub use zipper::{BlochZipperEngine, ZipperConfig, ZipperExecutionReport, ZipperStepTelemetry};
pub use qssm::{QssmConfig, QssmForwardReport, QuantumStateSpaceAutoencoder};

/// Seed used by the plain `new` constructors so default-initialised models are reproducible.
/// Use the `new_seeded` variants to choose a different (but still deterministic) initialisation.
pub const DEFAULT_SEED: u64 = 0x5EED_0042;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_state_normalization_and_fidelity() {
        let mut s1 = QuantumState::new(1024);
        let s2 = QuantumState::new(1024);
        assert!((s1.fidelity(&s2) - 1.0).abs() < 1e-6);

        s1.real[0] = 2.0;
        s1.normalize();
        let norm_sq: f64 = s1.real.iter().zip(&s1.imag).map(|(r, i)| r * r + i * i).sum();
        assert!((norm_sq - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ansatz_forward_evolution() {
        let qnn = DeepQuantumNeuralNetwork::new(1024, 2, 0.1, 0.05);
        let s = QuantumState::new(1024);
        let evolved = qnn.forward(&s);
        assert_eq!(evolved.dim, 1024);
        let fid = s.fidelity(&evolved);
        assert!(fid > 0.0 && fid <= 1.0);
    }

    #[test]
    fn test_povm_readout() {
        let labels = vec!["class_a".to_string(), "class_b".to_string()];
        let povm = POVMClassifier::new(1024, labels);
        let s = QuantumState::new(1024);
        let probs = povm.predict_probabilities(&s);
        assert_eq!(probs.len(), 2);
        let total_prob: f64 = probs.iter().map(|(_, p)| p).sum();
        assert!((total_prob - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_quantum_associative_memory() {
        let mut qam = QuantumAssociativeMemory::new(1024);
        let s1 = QuantumState::new(1024);
        qam.insert("doc1".into(), "Title 1".into(), "cat".into(), 100, s1.clone());

        let top = qam.recall_fidelity(&s1, 1);
        assert_eq!(top.len(), 1);
        assert!((top[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_feature_map_angle_encoding() {
        let encoding = DataEncodingLayer::new(10, EncodingKind::Angle);
        let features = vec![0.5, 1.2, -0.8];
        let state = encoding.encode(&features);
        assert_eq!(state.dim, 1024);
        let norm_sq: f64 = state.real.iter().zip(&state.imag).map(|(r, i)| r * r + i * i).sum();
        assert!((norm_sq - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pauli_observable_expectation() {
        let state = QuantumState::new(1024);
        let obs_z0 = PauliObservable::Z(0);
        let exp = obs_z0.expectation_value(&state);
        // Uniform superposition: equal amplitude in |0> and |1>, so <Z> = 0.0
        assert!(exp.abs() < 1e-6);
    }

    #[test]
    fn test_parameter_shift_rule_gradient() {
        let encoding = DataEncodingLayer::new(10, EncodingKind::Angle);
        let ansatz = CircuitAnsatz::new(10, 1);
        let obs = PauliObservable::Z(0);
        let engine = ParameterShiftEngine::new(encoding, ansatz, obs);

        let x = vec![0.3; 10];
        let weights = vec![0.1; engine.ansatz.param_count()];

        let grads = engine.compute_circuit_gradients(&x, &weights);
        assert_eq!(grads.len(), weights.len());

        // Validate gradient is bounded in [-1.0, 1.0] for unitary Pauli expectation
        for &g in &grads {
            assert!(g.abs() <= 1.0001, "Gradient exceeded unitary bound: {}", g);
        }
    }

    #[test]
    fn test_global_mean_z_observable() {
        let state = QuantumState::new(1024);
        let obs_global = PauliObservable::GlobalMeanZ;
        let exp = obs_global.expectation_value(&state);
        // Uniform superposition: expectation across all qubits averages to 0.0
        assert!(exp.abs() < 1e-6);

        // Ground state |0...0> where all bits are 0: Net Z = +1.0
        let mut ground = QuantumState::zero(1024);
        ground.real[0] = 1.0;
        let exp_ground = obs_global.expectation_value(&ground);
        assert!((exp_ground - 1.0).abs() < 1e-6);

        // State |1...1> (all bits are 1): Net Z = -1.0
        let mut excited = QuantumState::zero(1024);
        excited.real[1023] = 1.0;
        let exp_excited = obs_global.expectation_value(&excited);
        assert!((exp_excited - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_adam_exponential_lr_decay() {
        let mut opt = AdamOptimizer::new(10, 0.05);
        assert!((opt.lr - 0.05).abs() < 1e-8);

        // At epoch 0 (1st epoch), decay power 0 -> lr = 0.05
        let lr_ep0 = opt.step_epoch_decay(0, 0.9, 5);
        assert!((lr_ep0 - 0.05).abs() < 1e-8);

        // At epoch 4, floor(4/5) = 0 -> lr = 0.05
        let lr_ep4 = opt.step_epoch_decay(4, 0.9, 5);
        assert!((lr_ep4 - 0.05).abs() < 1e-8);

        // At epoch 5, floor(5/5) = 1 -> lr = 0.05 * 0.9 = 0.045
        let lr_ep5 = opt.step_epoch_decay(5, 0.9, 5);
        assert!((lr_ep5 - 0.045).abs() < 1e-8);

        // At epoch 10, floor(10/5) = 2 -> lr = 0.05 * 0.81 = 0.0405
        let lr_ep10 = opt.step_epoch_decay(10, 0.9, 5);
        assert!((lr_ep10 - 0.0405).abs() < 1e-8);
    }

    #[test]
    fn test_pi_phase_layer() {
        let pi_layer = PiPhaseLayer::new(10);
        let s = QuantumState::new(1024);
        let modulated = pi_layer.forward(&s);
        assert_eq!(modulated.dim, 1024);

        // Unitary phase modulation preserves total norm
        let norm_sq: f64 = modulated.real.iter().zip(&modulated.imag).map(|(r, i)| r * r + i * i).sum();
        assert!((norm_sq - 1.0).abs() < 1e-6);

        // Fidelity between original and modulated is bounded
        let fid = s.fidelity(&modulated);
        assert!((0.0..=1.0).contains(&fid));
    }

    #[test]
    fn test_avx2_vectorization_parity() {
        let s_vec = QuantumState::new(1024);
        let weights = vec![0.75; 20 * 2]; // Depth 2
        let ansatz = CircuitAnsatz::new(10, 2);
        let evolved = ansatz.forward(&s_vec, &weights);
        assert_eq!(evolved.dim, 1024);

        let norm_sq: f64 = evolved.real.iter().zip(&evolved.imag).map(|(r, i)| r * r + i * i).sum();
        assert!((norm_sq - 1.0).abs() < 1e-6, "AVX2 evolved state violated unitary normalization");
    }

    #[test]
    fn test_rademacher_phase_layer_numerical_equivalence() {
        let n_qubits = 10;
        let dim = 1 << n_qubits;
        let base = QuantumState::new(dim);

        // 1. Sequential un-fused loop (ground-truth reference)
        let pi_layer = PiPhaseLayer::new(n_qubits);
        let mut sequential = base.clone();
        for q in 0..n_qubits {
            let theta = pi_layer.signs[q] * std::f64::consts::PI;
            simd::apply_rz_kernel(&mut sequential.real, &mut sequential.imag, q, theta);
        }

        // 2. Fused AVX2 Rademacher Kernel
        let mut fused = base.clone();
        pi_layer.forward_in_place(&mut fused);

        // Strict bit-exact / floating-point parity comparison
        for i in 0..dim {
            assert!(
                (sequential.real[i] - fused.real[i]).abs() < 1e-10,
                "Mismatch at real amplitude index {}: seq={} vs fused={}",
                i,
                sequential.real[i],
                fused.real[i]
            );
            assert!(
                (sequential.imag[i] - fused.imag[i]).abs() < 1e-10,
                "Mismatch at imag amplitude index {}: seq={} vs fused={}",
                i,
                sequential.imag[i],
                fused.imag[i]
            );
        }

        // Fidelity must be identically 1.0
        let fid = sequential.fidelity(&fused);
        assert!((fid - 1.0).abs() < 1e-12, "Fidelity between sequential and fused Rademacher was {}", fid);
    }

    #[test]
    fn test_fused_observable_rotation_equivalence() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        let n_qubits = 12;
        let dim = 1 << n_qubits; // 4096

        // Initialize state with non-trivial amplitudes
        let mut state = QuantumState::new(dim);
        for i in 0..dim {
            state.real[i] = rng.gen_range(-1.0..1.0);
            state.imag[i] = rng.gen_range(-1.0..1.0);
        }
        state.normalize();

        let theta_y: Vec<f64> = (0..n_qubits)
            .map(|_| rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI))
            .collect();

        // Method A: Sequential Ry gates + Pauli-Z accumulation
        let mut state_rot = state.clone();
        for q in 0..n_qubits {
            simd::apply_ry_kernel(&mut state_rot.real, &mut state_rot.imag, q, theta_y[q]);
        }
        let mut z_method_a = [0.0; 12];
        for q in 0..n_qubits {
            z_method_a[q] = PauliObservable::Z(q).expectation_value(&state_rot);
        }

        // Method B: Zero-Allocation Observable Rotation Fusion directly on unmutated state
        let mut z_method_b = [0.0; 12];
        simd::measure_fused_ry_z_readout(&state.real, &state.imag, &theta_y, &mut z_method_b);

        // Verify mathematical equivalence to machine precision
        for q in 0..n_qubits {
            let diff = (z_method_a[q] - z_method_b[q]).abs();
            assert!(
                diff < 1e-12,
                "Qubit {}: Method A ({}) vs Method B ({}) diff={:e} exceeded 1e-12",
                q,
                z_method_a[q],
                z_method_b[q],
                diff
            );
        }
    }
}
