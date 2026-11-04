// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Holographic Wormhole Simulation Engine: Traversable ER=EPR Protocol.
//!
//! Orchestrates the complete Gao-Jafferis-Wall traversable wormhole protocol:
//! 1. Prepares the reference qubit R_ref and probe L_1 in an entangled Bell state.
//! 2. Prepares the remaining (L_k, R_k) pairs in Thermofield Double (TFD) Bell states.
//! 3. Evolves the Left register forward under a chaotic fast-scrambling Hamiltonian U_L(t).
//! 4. Applies a non-local bilateral coupling U(g) = exp(i g/N \sum Z_L Z_R) acting as a negative energy pulse.
//! 5. Evolves the Right register under conjugate time-reversal U_R(t) = U_L^*(t).
//! 6. Verifies state refocusing on the Right boundary R_1, measuring teleportation fidelity and sign asymmetry.

use super::metrics::{
    bell_fidelity, compute_operator_size_distribution, reduced_density_matrix_2q,
    single_qubit_reduced_density_matrix, state_fidelity_1q, WormholeFidelityPoint,
    WormholeSweepReport,
};
use super::scrambler::{ScramblerType, ScramblingHamiltonian};
use crate::compiler::op::CircuitOp;
use crate::compiler::{CircuitBuilder, CompiledCircuit};
use crate::state::QuantumState;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Wire indices layout for the holographic wormhole simulation circuit (width Q = 2N + 1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WormholeWireMap {
    /// Number of spatial modes N per boundary
    pub num_modes: usize,
    /// Reference qubit index (R_ref = 0)
    pub ref_qubit: usize,
    /// Left black hole register wires [L_1, ..., L_N] (1 ..= N)
    pub left_qubits: Vec<usize>,
    /// Right black hole register wires [R_1, ..., R_N] (N+1 ..= 2N)
    pub right_qubits: Vec<usize>,
    /// Total circuit width Q = 2N + 1
    pub total_qubits: usize,
}

impl WormholeWireMap {
    pub fn new(num_modes: usize) -> Self {
        assert!(num_modes >= 2, "Holographic wormhole requires at least 2 spatial modes");
        let ref_qubit = 0;
        let left_qubits: Vec<usize> = (1..=num_modes).collect();
        let right_qubits: Vec<usize> = ((num_modes + 1)..=(2 * num_modes)).collect();
        let total_qubits = 2 * num_modes + 1;

        Self {
            num_modes,
            ref_qubit,
            left_qubits,
            right_qubits,
            total_qubits,
        }
    }
}

/// Configuration parameters for the holographic wormhole experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WormholeConfig {
    /// Number of spatial boundary modes N per side (e.g. 3, 4, 5)
    pub num_modes: usize,
    /// Underlying chaotic scrambler model
    pub scrambler_type: ScramblerType,
    /// Trotterization steps for Hamiltonian evolution
    pub trotter_steps: usize,
    /// Total forward/backward scrambling time duration
    pub scrambling_time: f64,
    /// Bilateral coupling strength g
    pub coupling_g: f64,
    /// Random seed for deterministic Hamiltonian generation
    pub seed: u64,
}

impl Default for WormholeConfig {
    fn default() -> Self {
        Self {
            num_modes: 3,
            scrambler_type: ScramblerType::ChaoticHeisenberg,
            trotter_steps: 3,
            scrambling_time: 1.5,
            coupling_g: 1.5,
            seed: 42,
        }
    }
}

/// Core Holographic Wormhole Simulation Engine.
#[derive(Debug, Clone)]
pub struct HolographicWormholeEngine {
    pub config: WormholeConfig,
    pub wire_map: WormholeWireMap,
    pub scrambler: ScramblingHamiltonian,
}

impl HolographicWormholeEngine {
    /// Initializes a new holographic wormhole engine.
    pub fn new(config: WormholeConfig) -> Self {
        let wire_map = WormholeWireMap::new(config.num_modes);
        let scrambler =
            ScramblingHamiltonian::new(config.scrambler_type, config.num_modes, config.seed);
        Self {
            config,
            wire_map,
            scrambler,
        }
    }

    /// Builds the complete quantum circuit for a given coupling strength g.
    pub fn build_circuit(&self, g: f64) -> CompiledCircuit {
        let mut builder = CircuitBuilder::new(self.wire_map.total_qubits);
        let n = self.wire_map.num_modes;
        let dt = self.config.scrambling_time / (self.config.trotter_steps.max(1) as f64);

        // 1. Entangle Reference Qubit (0) and Probe Site L_1 (1) in Bell pair |\Phi^+>
        builder.hadamard(self.wire_map.ref_qubit);
        builder.cnot(self.wire_map.ref_qubit, self.wire_map.left_qubits[0]);

        // 2. Thermofield Double (TFD) State Preparation across remaining modes (L_k, R_k)
        for k in 1..n {
            let l_k = self.wire_map.left_qubits[k];
            let r_k = self.wire_map.right_qubits[k];
            builder.hadamard(l_k);
            builder.cnot(l_k, r_k);
        }

        // 3. Forward Scrambling on Left Register: U_L(t)
        for _ in 0..self.config.trotter_steps {
            let fwd_ops = self.scrambler.build_forward_ops(&self.wire_map.left_qubits, dt);
            for op in fwd_ops {
                builder.dag.add_op(op);
            }
        }

        // 4. Bilateral Coupling (Negative Energy Shockwave at t = 0): U(g) = exp(i g/N \sum Z_L Z_R)
        if g.abs() > 1e-9 {
            let coupling_theta = -2.0 * g / (n as f64);
            for k in 0..n {
                let l_k = self.wire_map.left_qubits[k];
                let r_k = self.wire_map.right_qubits[k];
                builder.dag.add_op(CircuitOp::CNOT { control: l_k, target: r_k });
                builder.dag.add_op(CircuitOp::Rz { qubit: r_k, theta: coupling_theta });
                builder.dag.add_op(CircuitOp::CNOT { control: l_k, target: r_k });
            }
        }

        // 5. Unscrambling Time-Evolution on Right Register: U_R(t) = U_L^*(t)
        for _ in 0..self.config.trotter_steps {
            let conj_ops = self.scrambler.build_conjugate_ops(&self.wire_map.right_qubits, dt);
            for op in conj_ops {
                builder.dag.add_op(op);
            }
        }

        CompiledCircuit::compile(&builder.dag)
    }

    /// Simulates wormhole traversal for a specific bilateral coupling strength g.
    pub fn simulate(&self, g: f64) -> WormholeFidelityPoint {
        let circuit = self.build_circuit(g);
        let dim = 1 << self.wire_map.total_qubits;
        let mut state = QuantumState::ground(dim);
        circuit.execute(&mut state);

        // Target receivers: R_ref (wire 0) and R_1 (first Right wire)
        let target_r1 = self.wire_map.right_qubits[0];
        let rho_2q = reduced_density_matrix_2q(&state, self.wire_map.ref_qubit, target_r1);
        let fid = bell_fidelity(&rho_2q);

        // Negative energy counterpart to evaluate asymmetry
        let fid_neg = if g.abs() > 1e-9 {
            let neg_circuit = self.build_circuit(-g.abs());
            let mut neg_state = QuantumState::ground(dim);
            neg_circuit.execute(&mut neg_state);
            let neg_rho = reduced_density_matrix_2q(&neg_state, self.wire_map.ref_qubit, target_r1);
            bell_fidelity(&neg_rho)
        } else {
            fid
        };

        let asymmetry = if g > 0.0 { fid - fid_neg } else { 0.0 };
        let otoc = self.scrambler.compute_otoc(
            0,
            self.wire_map.num_modes - 1,
            self.config.scrambling_time,
            self.config.trotter_steps,
        );

        WormholeFidelityPoint {
            g,
            fidelity: fid,
            asymmetry,
            otoc,
        }
    }

    /// Sweeps the bilateral coupling g across negative (wrong-sign) and positive (ANEC violation) regimes.
    pub fn sweep_coupling(
        &self,
        g_min: f64,
        g_max: f64,
        num_points: usize,
    ) -> WormholeSweepReport {
        assert!(num_points >= 3, "Sweep requires at least 3 sampling points");
        let step = (g_max - g_min) / ((num_points - 1) as f64);
        let mut points = Vec::with_capacity(num_points);

        let mut peak_fidelity = 0.0;
        let mut optimal_g = 0.0;
        let mut peak_asymmetry = 0.0;
        let mut uncoupled_noise_floor = 0.25;

        for i in 0..num_points {
            let g = g_min + (i as f64) * step;
            let pt = self.simulate(g);
            if g.abs() < step * 0.5 {
                uncoupled_noise_floor = pt.fidelity;
            }
            if pt.fidelity > peak_fidelity {
                peak_fidelity = pt.fidelity;
                optimal_g = pt.g;
            }
            if pt.asymmetry > peak_asymmetry {
                peak_asymmetry = pt.asymmetry;
            }
            points.push(pt);
        }

        let size_distribution = compute_operator_size_distribution(
            self.wire_map.num_modes,
            &self.scrambler,
            self.config.scrambling_time,
        );

        WormholeSweepReport {
            num_modes: self.wire_map.num_modes,
            total_qubits: self.wire_map.total_qubits,
            scrambling_time: self.config.scrambling_time,
            uncoupled_noise_floor,
            optimal_g,
            peak_fidelity,
            peak_asymmetry,
            points,
            size_distribution,
        }
    }

    /// Simulates pure single-qubit state teleportation |\psi> = cos(\theta/2)|0> + e^{i\phi} sin(\theta/2)|1>.
    pub fn simulate_pure_state(&self, g: f64, theta: f64, phi: f64) -> f64 {
        let mut builder = CircuitBuilder::new(self.wire_map.total_qubits);
        let n = self.wire_map.num_modes;
        let dt = self.config.scrambling_time / (self.config.trotter_steps.max(1) as f64);

        // 1. Prepare probe state directly on L_1
        builder.ry(self.wire_map.left_qubits[0], theta);
        builder.rz(self.wire_map.left_qubits[0], phi);

        // 2. TFD across remaining modes (L_k, R_k)
        for k in 1..n {
            let l_k = self.wire_map.left_qubits[k];
            let r_k = self.wire_map.right_qubits[k];
            builder.hadamard(l_k);
            builder.cnot(l_k, r_k);
        }

        // 3. Forward scrambling on Left: U_L(t)
        for _ in 0..self.config.trotter_steps {
            for op in self.scrambler.build_forward_ops(&self.wire_map.left_qubits, dt) {
                builder.dag.add_op(op);
            }
        }

        // 4. Bilateral Coupling
        if g.abs() > 1e-9 {
            let coupling_theta = -2.0 * g / (n as f64);
            for k in 0..n {
                let l_k = self.wire_map.left_qubits[k];
                let r_k = self.wire_map.right_qubits[k];
                builder.dag.add_op(CircuitOp::CNOT { control: l_k, target: r_k });
                builder.dag.add_op(CircuitOp::Rz { qubit: r_k, theta: coupling_theta });
                builder.dag.add_op(CircuitOp::CNOT { control: l_k, target: r_k });
            }
        }

        // 5. Unscrambling on Right
        for _ in 0..self.config.trotter_steps {
            for op in self.scrambler.build_conjugate_ops(&self.wire_map.right_qubits, dt) {
                builder.dag.add_op(op);
            }
        }

        let compiled = CompiledCircuit::compile(&builder.dag);
        let dim = 1 << self.wire_map.total_qubits;
        let mut state = QuantumState::ground(dim);
        compiled.execute(&mut state);

        let target_r1 = self.wire_map.right_qubits[0];
        let rho_1q = single_qubit_reduced_density_matrix(&state, target_r1);

        let alpha = Complex64::new((theta * 0.5).cos(), 0.0);
        let beta = Complex64::from_polar((theta * 0.5).sin(), phi);

        state_fidelity_1q(&rho_1q, alpha, beta)
    }
}
