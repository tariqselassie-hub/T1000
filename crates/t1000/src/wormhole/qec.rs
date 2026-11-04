// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Holographic Bulk Quantum Error Correction & Boundary Erasure Protection.
//!
//! Under the Almheiri-Dong-Harlow (ADH) theorem, AdS/CFT bulk reconstruction functions
//! as an entanglement-assisted quantum error-correcting code. A probe passing through
//! the traversable throat is protected against boundary erasures because information is
//! encoded non-locally across the entangling horizons.
//!
//! Evaluates:
//! - Bulk probe reconstructibility under k boundary qubit erasures
//! - Critical entanglement wedge threshold: maximum loss percentage before throat pinch-off
//! - Effective holographic code distance d_holo

use super::metrics::{bell_fidelity, reduced_density_matrix_2q};
use super::scrambler::ScramblingHamiltonian;
use super::teleportation::{HolographicWormholeEngine, WormholeConfig, WormholeWireMap};
use crate::compiler::op::CircuitOp;
use crate::compiler::{CircuitBuilder, CompiledCircuit};
use crate::state::QuantumState;
use serde::{Deserialize, Serialize};

/// Which boundary partition suffers the erasure channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErasurePartition {
    /// Erasure on Left black hole boundary qubits
    LeftHorizon,
    /// Erasure on Right black hole boundary qubits
    RightHorizon,
    /// Distributed erasure across both horizons
    BilateralHorizons,
}

/// Telemetry for a specific boundary erasure count.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErasurePoint {
    /// Number of physical boundary qubits erased
    pub erased_count: usize,
    /// Indices of erased wires
    pub erased_wires: Vec<usize>,
    /// Fraction of boundary modes erased: k / N
    pub erasure_fraction: f64,
    /// Reconstructed probe Bell fidelity F(k)
    pub recovered_fidelity: f64,
    /// Gravitational sign asymmetry with erasures present
    pub asymmetry: f64,
    /// Whether the traversable throat remained open (F > 0.2500)
    pub throat_open: bool,
}

/// Comprehensive report on holographic quantum error correction under boundary loss.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HolographicQecReport {
    pub num_modes: usize,
    pub total_qubits: usize,
    pub partition: ErasurePartition,
    pub clean_fidelity: f64,
    pub points: Vec<ErasurePoint>,
    /// Effective holographic code distance d_holo (minimum erasures causing throat collapse)
    pub holographic_code_distance: usize,
    /// Maximum boundary erasure fraction tolerated with open throat
    pub entanglement_wedge_threshold: f64,
}

/// Evaluates holographic error correction resilience against boundary erasures.
pub fn evaluate_holographic_qec(
    engine_config: &WormholeConfig,
    hamiltonian: &ScramblingHamiltonian,
    partition: ErasurePartition,
    coupling_g: f64,
) -> HolographicQecReport {
    let wire_map = WormholeWireMap::new(engine_config.num_modes);
    let n = wire_map.num_modes;
    let dt = engine_config.scrambling_time / (engine_config.trotter_steps.max(1) as f64);

    // Baseline clean fidelity with zero erasures
    let mut engine = HolographicWormholeEngine::new(engine_config.clone());
    engine.scrambler = hamiltonian.clone();
    let clean_pt = engine.simulate(coupling_g);
    let clean_fid = clean_pt.fidelity;

    let mut points = Vec::new();
    let mut code_distance = n;
    let mut threshold_frac = 0.0;

    // Sweep erasure count k from 0 up to N - 1
    for k in 0..n {
        let erased_indices: Vec<usize> = match partition {
            ErasurePartition::LeftHorizon => (0..k).map(|i| wire_map.left_qubits[n - 1 - i]).collect(),
            ErasurePartition::RightHorizon => (0..k).map(|i| wire_map.right_qubits[n - 1 - i]).collect(),
            ErasurePartition::BilateralHorizons => {
                let mut wires = Vec::new();
                for i in 0..k {
                    if i % 2 == 0 {
                        wires.push(wire_map.left_qubits[n - 1 - (i / 2)]);
                    } else {
                        wires.push(wire_map.right_qubits[n - 1 - (i / 2)]);
                    }
                }
                wires
            }
        };

        // Construct erasure circuit
        let mut builder = CircuitBuilder::new(wire_map.total_qubits);

        // 1. Entangle Reference Qubit (0) and Probe Site L_1 (1) in Bell pair |\Phi^+>
        builder.hadamard(wire_map.ref_qubit);
        builder.cnot(wire_map.ref_qubit, wire_map.left_qubits[0]);

        // 2. Thermofield Double (TFD) State Preparation across remaining modes (L_k, R_k)
        for idx in 1..n {
            let l_k = wire_map.left_qubits[idx];
            let r_k = wire_map.right_qubits[idx];
            builder.hadamard(l_k);
            builder.cnot(l_k, r_k);
        }

        // 3. Forward Scrambling on Left Register: U_L(t)
        for _ in 0..engine_config.trotter_steps {
            for op in hamiltonian.build_forward_ops(&wire_map.left_qubits, dt) {
                builder.dag.add_op(op);
            }
        }

        // 4. Inject boundary erasure errors (depolarization on erased wires)
        for &erased_wire in &erased_indices {
            // Depolarizing / dephasing channel injection on erased wire
            builder.rz(erased_wire, std::f64::consts::PI * 0.5);
            builder.hadamard(erased_wire);
        }

        // 5. Bilateral Coupling (Shockwave): U(g)
        // Erased boundary modes lose their bilateral coupling channel
        if coupling_g.abs() > 1e-9 {
            let coupling_theta = -2.0 * coupling_g / (n as f64);
            for idx in 0..n {
                let l_k = wire_map.left_qubits[idx];
                let r_k = wire_map.right_qubits[idx];

                // If mode is erased, its shockwave channel is severed
                let is_erased = erased_indices.contains(&l_k) || erased_indices.contains(&r_k);
                if !is_erased {
                    builder.dag.add_op(CircuitOp::CNOT { control: l_k, target: r_k });
                    builder.dag.add_op(CircuitOp::Rz { qubit: r_k, theta: coupling_theta });
                    builder.dag.add_op(CircuitOp::CNOT { control: l_k, target: r_k });
                }
            }
        }

        // 6. Backward Unscrambling on Right
        for _ in 0..engine_config.trotter_steps {
            for op in hamiltonian.build_conjugate_ops(&wire_map.right_qubits, dt) {
                builder.dag.add_op(op);
            }
        }

        let compiled = CompiledCircuit::compile(&builder.dag);
        let dim = 1 << wire_map.total_qubits;
        let mut state = QuantumState::ground(dim);
        compiled.execute(&mut state);

        let target_r1 = wire_map.right_qubits[0];
        let rho_2q = reduced_density_matrix_2q(&state, wire_map.ref_qubit, target_r1);
        let recovered_fid = bell_fidelity(&rho_2q);

        let throat_open = recovered_fid > 0.2505;
        let frac = (k as f64) / (n as f64);

        if throat_open {
            threshold_frac = frac;
        } else if code_distance == n && k > 0 {
            code_distance = k;
        }

        points.push(ErasurePoint {
            erased_count: k,
            erased_wires: erased_indices,
            erasure_fraction: frac,
            recovered_fidelity: recovered_fid,
            asymmetry: (recovered_fid - 0.25).max(0.0),
            throat_open,
        });
    }

    HolographicQecReport {
        num_modes: n,
        total_qubits: wire_map.total_qubits,
        partition,
        clean_fidelity: clean_fid,
        points,
        holographic_code_distance: code_distance,
        entanglement_wedge_threshold: threshold_frac,
    }
}

