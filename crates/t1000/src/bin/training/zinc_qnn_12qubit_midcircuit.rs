// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! GuddaNetworks 12-Qubit QNN: Mid-Circuit Feed-Forward Engine
//!
//! MECHANISM #2: Non-destructive mid-circuit measurement of Qubit 8 expectation ⟨Z_8⟩_L1
//! dynamically modulating terminal rotation angles before readout:
//!   θ̃_{y, q} = θ_{y, q} + α_q · ⟨Z_8⟩_L1
//!
//! Theoretical Foundation:
//!   Single-upload unitary circuits are mathematically restricted to degree-2 quadratic
//!   forms in normalized descriptors: ⟨Z_q⟩ = (x^T M_q x) / ||x||^2.
//!   Molecule #10640 (tri-cationic zwitterion, true logP = -6.772 vs pred = +2.418) induces
//!   93.4% spin freeze on Qubit 8 (⟨Z_8⟩_L1 = -0.867).
//!   Feeding ⟨Z_8⟩_L1 forward into subsequent gate angles introduces cos(α_q · ⟨Z_8⟩) and
//!   sin(α_q · ⟨Z_8⟩) terms, yielding degree-4, degree-6, and higher-order transcendental
//!   non-linearities that break the quadratic ceiling and resolve deep-regime Born solvation.

use std::fs;
use t1000::zinc_data::ZincMolecule;
use t1000::zinc_stage_common::{load_stage_data, Data};
use t1000::zinc_training_common::{resolve_workspace_path, PrunedCheckpoint};
use t1000::{CircuitAnsatz, DataEncodingLayer, EncodingKind, QuantumState};

#[path = "zinc_qnn_12qubit_midcircuit/phases.rs"]
mod phases;
#[path = "zinc_qnn_12qubit_midcircuit/report.rs"]
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

    /// The 12 single-qubit marginals under feed-forward `alphas`, and the mid-circuit
    /// ⟨Z_8⟩_L1 signal.
    fn z12(
        &self,
        features: &[f64],
        alphas: &[f64],
        state: &mut QuantumState,
    ) -> ([f64; N_QUBITS], f64) {
        self.encoding.encode_amplitude_into(features, state);
        let mut z = [0.0; N_QUBITS];
        let m8 = self.ansatz.forward_midcircuit_feedforward_in_place(
            state,
            &self.params,
            SOURCE_QUBIT,
            alphas,
            &mut z,
        );
        (z, m8)
    }

    /// Fills the 12 single-qubit marginals and the 66 two-body correlators.
    fn readout78(
        &self,
        features: &[f64],
        alphas: &[f64],
        state: &mut QuantumState,
        z: &mut [f64; N_QUBITS],
        zz: &mut [f64; N_PAIRS],
    ) {
        self.encoding.encode_amplitude_into(features, state);
        self.ansatz.forward_midcircuit_full_readout_in_place(
            state,
            &self.params,
            SOURCE_QUBIT,
            alphas,
            z,
            zz,
        );
    }

    /// 12 single-qubit marginals followed by the 66 two-body correlators.
    fn obs78(&self, features: &[f64], alphas: &[f64], state: &mut QuantumState) -> Vec<f64> {
        let mut z = [0.0; N_QUBITS];
        let mut zz = [0.0; N_PAIRS];
        self.readout78(features, alphas, state, &mut z, &mut zz);
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
    println!(
        "  -> Using pruned 99-parameter checkpoint: {}",
        path.display()
    );
    let content = fs::read_to_string(&path).expect("Read pruned checkpoint");
    let ckpt: PrunedCheckpoint = serde_json::from_str(&content).expect("Parse pruned checkpoint");
    let params = ckpt.pruned_hybrid_params[..60].to_vec();
    println!(
        "  -> Active circuit parameters: {} (48 L1/L2 angles + 12 terminal Ry)\n",
        params.len()
    );
    Some(params)
}

fn load_inputs() -> Option<(Data, Vec<f64>)> {
    let data = load_stage_data()?;
    let params = load_circuit_params()?;
    Some((data, params))
}

fn main() {
    report::print_banner();
    let Some((data, params)) = load_inputs() else {
        return;
    };
    let circuit = Circuit::new(params);
    let mut state = QuantumState::new(DIM);

    let base = phases::baseline(&circuit, &data, &mut state);
    let alphas = phases::train_alphas(&circuit, &data.train);
    let ff = phases::feedforward(&circuit, &data, &alphas, &base.fit, &mut state);
    let comb = phases::combined(&circuit, &data, &alphas, &base.fit, &ff, &mut state);
    let audit = phases::audit(&circuit, &alphas, (&base, &ff, &comb), &mut state);
    report::print_corrections(&data.test, &base.fit, &comb);
    let tp = phases::throughput(&circuit, &data.test, &alphas, &mut state);
    report::persist(&data, &circuit, &alphas, (&base, &ff, &comb), &audit, &tp);
}
