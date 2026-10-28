// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! GuddaNetworks 12-Qubit QNN: Data Re-Uploading Engine (Mechanism #3)
//!
//! MECHANISM #3: Inter-layer classical feature re-encoding between Layer 0 and Layer 1:
//!   |ψ_1⟩ = U_CZ · [ ⊗_q R_y(θ_{1y, q}) R_z(θ_{1z, q}) R_y(s_q · x̃_q) ] · |ψ_0⟩
//!
//! Theoretical Foundation (Pérez-Salinas et al. 2020, Schuld et al. 2021):
//!   Single-upload unitary circuits are restricted to degree-2 quadratic forms / degree-1
//!   Fourier harmonics. Interleaving input descriptor re-uploading between parameterized
//!   layers transforms the QNN into a Universal Function Approximator by multiplying the
//!   accessible Fourier frequency spectrum: ω ∈ {-L, ..., +L}.
//!
//! Combined Unified Stack:
//!   Mechanism #1 : 66 Two-Body Ising Correlators ⟨Z_i Z_j⟩
//!   Mechanism #2 : Mid-Circuit Feed-Forward from Qubit 8 (θ̃_{y,q} = θ_{y,q} + α_q · ⟨Z_8⟩_L1)
//!   Mechanism #3 : Inter-Layer Data Re-Uploading (s_q · x̃_q before Layer 1 CZ block)

use std::fs;
use t1000::zinc_data::ZincMolecule;
use t1000::zinc_stage_common::{load_stage_data, Data};
use t1000::zinc_training_common::{resolve_workspace_path, MidCircuitCheckpoint, PrunedCheckpoint};
use t1000::{CircuitAnsatz, DataEncodingLayer, EncodingKind, QuantumState};

#[path = "zinc_qnn_12qubit_reuploading/phases.rs"]
mod phases;
#[path = "zinc_qnn_12qubit_reuploading/report.rs"]
mod report;

const N_QUBITS: usize = 12;
const DIM: usize = 1 << N_QUBITS; // 4,096
const N_PAIRS: usize = N_QUBITS * (N_QUBITS - 1) / 2; // 66
const N_OBS_FULL: usize = N_QUBITS + N_PAIRS; // 78
const SOURCE_QUBIT: usize = 8; // Qubit 8 exhibiting severe spin freeze on anomalous molecules

/// The frozen terminal-layer circuit parameters plus the circuit that consumes them.
#[derive(Debug)]
pub struct Circuit {
    ansatz: CircuitAnsatz,
    encoding: DataEncodingLayer,
    params: Vec<f64>,
}

impl Circuit {
    fn new(params: Vec<f64>) -> Self {
        Circuit {
            ansatz: CircuitAnsatz::new(N_QUBITS, 3).with_pi_layer(true),
            encoding: DataEncodingLayer::new(N_QUBITS, EncodingKind::Amplitude),
            params,
        }
    }

    /// The 12 single-qubit marginals with re-upload `scales` and feed-forward `alphas`.
    fn z12(
        &self,
        features: &[f64],
        (scales, alphas): (&[f64], &[f64]),
        state: &mut QuantumState,
    ) -> [f64; N_QUBITS] {
        self.encoding.encode_amplitude_into(features, state);
        let mut z = [0.0; N_QUBITS];
        self.ansatz.forward_reuploading_in_place(
            state,
            &self.params,
            features,
            scales,
            SOURCE_QUBIT,
            alphas,
            &mut z,
        );
        z
    }

    /// Fills the 12 single-qubit marginals and the 66 two-body correlators.
    fn readout78(
        &self,
        features: &[f64],
        (scales, alphas): (&[f64], &[f64]),
        state: &mut QuantumState,
        (z, zz): (&mut [f64; N_QUBITS], &mut [f64; N_PAIRS]),
    ) {
        self.encoding.encode_amplitude_into(features, state);
        self.ansatz.forward_reuploading_full_readout_in_place(
            state,
            &self.params,
            features,
            scales,
            SOURCE_QUBIT,
            alphas,
            z,
            zz,
        );
    }

    /// 12 single-qubit marginals followed by the 66 two-body correlators.
    fn obs78(
        &self,
        features: &[f64],
        angles: (&[f64], &[f64]),
        state: &mut QuantumState,
    ) -> Vec<f64> {
        let mut z = [0.0; N_QUBITS];
        let mut zz = [0.0; N_PAIRS];
        self.readout78(features, angles, state, (&mut z, &mut zz));
        let mut obs = Vec::with_capacity(N_OBS_FULL);
        obs.extend_from_slice(&z);
        obs.extend_from_slice(&zz);
        obs
    }
}

/// The first 60 pruned-checkpoint parameters (48 L1/L2 angles + 12 terminal Ry).
fn load_circuit_params() -> Option<Vec<f64>> {
    println!("[INIT] Loading quantum circuit parameters...");
    let path = resolve_workspace_path(
        "tests/benchmarks/zinc_molecular/checkpoints/zinc_12qubit_checkpoint_pruned99.json",
    );
    if !path.exists() {
        eprintln!("[-] Pruned checkpoint not found at: {}", path.display());
        return None;
    }
    let content = fs::read_to_string(&path).expect("Read pruned checkpoint");
    let ckpt: PrunedCheckpoint = serde_json::from_str(&content).expect("Parse pruned checkpoint");
    println!("  -> Loaded 60 pruned circuit parameters.");
    Some(ckpt.pruned_hybrid_params[..60].to_vec())
}

/// Feed-forward alphas from Mechanism #2, or zeros when that checkpoint is absent.
fn load_alphas() -> Vec<f64> {
    let path = resolve_workspace_path(
        "tests/benchmarks/zinc_molecular/checkpoints/zinc_12qubit_checkpoint_midcircuit.json",
    );
    if !path.exists() {
        println!("  -> Using default zero feed-forward alphas.");
        return vec![0.0; N_QUBITS];
    }
    let content = fs::read_to_string(&path).expect("Read midcircuit checkpoint");
    let ckpt: MidCircuitCheckpoint =
        serde_json::from_str(&content).expect("Parse midcircuit checkpoint");
    println!("  -> Loaded 12 optimized feed-forward alphas from Mechanism #2.");
    ckpt.feedforward_alphas
}

fn main() {
    report::print_banner();
    let Some(data) = load_stage_data() else {
        return;
    };
    let Some(params) = load_circuit_params() else {
        return;
    };
    let alphas = load_alphas();
    let circuit = Circuit::new(params);
    let mut state = QuantumState::new(DIM);

    let (base, ff) = phases::baselines(&circuit, &data, &alphas, &mut state);
    let scales = phases::train_scales(&circuit, &data.train, &alphas);
    let angles = (&scales[..], &alphas[..]);
    let dru = phases::reuploading(&circuit, &data, angles, &base, &mut state);
    let uni = phases::unified(&circuit, &data, angles, (&base, &ff, &dru), &mut state);
    let stages = [&base, &ff, &dru, &uni];
    let audit = phases::audit(&circuit, angles, stages, &mut state);
    report::print_corrections(&data.test, &base, &uni);
    let tp = phases::throughput(&circuit, &data.test, angles, &mut state);
    report::persist(&data, &circuit, angles, stages, &audit, &tp);
}
