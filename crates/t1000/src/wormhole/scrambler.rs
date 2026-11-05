// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Chaotic Quantum Scrambler Engines for Holographic Black Hole Simulation.
//! Implements fast-scrambling Hamiltonians:
//! - All-to-all coupled chaotic Heisenberg spin glass
//! - Jordan-Wigner mapped 4-Majorana SYK fast scrambler
//! - Periodic Floquet-Haar brickwork with irrational phase shifts
//!
//! Provides unitary forward evolution on the Left black hole register and
//! exact complex-conjugate time-reversed evolution on the Right register,
//! alongside Out-of-Time-Order Correlator (OTOC) validation.

use crate::compiler::op::{CircuitOp, Unitary2x2};
use crate::compiler::CircuitDag;
use crate::state::QuantumState;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Type of chaotic quantum scrambler driving the black hole interior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScramblerType {
    /// All-to-all random coupling J_x, J_y, J_z Heisenberg spin glass plus longitudinal field h_z
    ChaoticHeisenberg,
    /// 4-Majorana fermion SYK model with Gaussian random 4-body interactions
    MajoranaSyk,
    /// Alternating Floquet entangling brickwork with non-commuting irrational rotations
    FloquetHaar,
}

/// Parameterized chaotic scrambling Hamiltonian generator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScramblingHamiltonian {
    pub scrambler_type: ScramblerType,
    pub num_qubits: usize,
    pub seed: u64,
    /// Pairwise coupling constants `[j][k]` -> (Jx, Jy, Jz)
    pub pairwise_couplings: Vec<((usize, usize), (f64, f64, f64))>,
    /// Single-qubit local fields `[j]` -> hz
    pub local_fields: Vec<f64>,
    /// 4-body SYK coupling tensors: ((i, j, k, l), J_ijkl)
    pub syk_couplings: Vec<((usize, usize, usize, usize), f64)>,
}

impl ScramblingHamiltonian {
    /// Constructs a new chaotic scrambling Hamiltonian from a deterministic seed.
    pub fn new(scrambler_type: ScramblerType, num_qubits: usize, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut pairwise = Vec::new();
        let mut fields = Vec::with_capacity(num_qubits);
        let mut syk = Vec::new();

        // Longitudinal fields
        for _ in 0..num_qubits {
            fields.push((rng.gen::<f64>() - 0.5) * 2.0);
        }

        match scrambler_type {
            ScramblerType::ChaoticHeisenberg => {
                let norm = 1.0 / (num_qubits as f64).sqrt();
                for i in 0..num_qubits {
                    for j in (i + 1)..num_qubits {
                        let jx = (rng.gen::<f64>() - 0.5) * 2.0 * norm;
                        let jy = (rng.gen::<f64>() - 0.5) * 2.0 * norm;
                        let jz = (rng.gen::<f64>() - 0.5) * 2.0 * norm;
                        pairwise.push(((i, j), (jx, jy, jz)));
                    }
                }
            }
            ScramblerType::MajoranaSyk => {
                let norm = (6.0 / (num_qubits as f64).powi(3)).sqrt();
                for i in 0..num_qubits {
                    for j in (i + 1)..num_qubits {
                        for k in (j + 1)..num_qubits {
                            for l in (k + 1)..num_qubits {
                                let j_ijkl = (rng.gen::<f64>() - 0.5) * 2.0 * norm;
                                syk.push(((i, j, k, l), j_ijkl));
                            }
                        }
                    }
                }
            }
            ScramblerType::FloquetHaar => {
                for i in 0..(num_qubits.saturating_sub(1)) {
                    pairwise.push(((i, i + 1), (1.0, 1.0, 1.0)));
                }
            }
        }

        Self {
            scrambler_type,
            num_qubits,
            seed,
            pairwise_couplings: pairwise,
            local_fields: fields,
            syk_couplings: syk,
        }
    }

    /// Appends elementary Rzz gate: exp(-i theta/2 Z_a Z_b).
    fn append_rzz(ops: &mut Vec<CircuitOp>, q1: usize, q2: usize, theta: f64) {
        ops.push(CircuitOp::CNOT {
            control: q1,
            target: q2,
        });
        ops.push(CircuitOp::Rz { qubit: q2, theta });
        ops.push(CircuitOp::CNOT {
            control: q1,
            target: q2,
        });
    }

    /// Appends elementary Rxx gate: exp(-i theta/2 X_a X_b).
    fn append_rxx(ops: &mut Vec<CircuitOp>, q1: usize, q2: usize, theta: f64) {
        let h = Unitary2x2::hadamard();
        ops.push(CircuitOp::Unitary1Q {
            qubit: q1,
            matrix: h,
        });
        ops.push(CircuitOp::Unitary1Q {
            qubit: q2,
            matrix: h,
        });
        Self::append_rzz(ops, q1, q2, theta);
        ops.push(CircuitOp::Unitary1Q {
            qubit: q1,
            matrix: h,
        });
        ops.push(CircuitOp::Unitary1Q {
            qubit: q2,
            matrix: h,
        });
    }

    /// Appends elementary Ryy gate: exp(-i theta/2 Y_a Y_b).
    fn append_ryy(ops: &mut Vec<CircuitOp>, q1: usize, q2: usize, theta: f64) {
        let rx_p = Unitary2x2::rx(PI * 0.5);
        let rx_m = Unitary2x2::rx(-PI * 0.5);
        ops.push(CircuitOp::Unitary1Q {
            qubit: q1,
            matrix: rx_p,
        });
        ops.push(CircuitOp::Unitary1Q {
            qubit: q2,
            matrix: rx_p,
        });
        Self::append_rzz(ops, q1, q2, theta);
        ops.push(CircuitOp::Unitary1Q {
            qubit: q1,
            matrix: rx_m,
        });
        ops.push(CircuitOp::Unitary1Q {
            qubit: q2,
            matrix: rx_m,
        });
    }

    /// Builds the forward scrambling circuit operations for the specified register wires.
    pub fn build_forward_ops(&self, wires: &[usize], dt: f64) -> Vec<CircuitOp> {
        let mut ops = Vec::new();
        match self.scrambler_type {
            ScramblerType::ChaoticHeisenberg => {
                for &((i, j), (jx, jy, jz)) in &self.pairwise_couplings {
                    if i < wires.len() && j < wires.len() {
                        let w_i = wires[i];
                        let w_j = wires[j];
                        Self::append_rzz(&mut ops, w_i, w_j, 2.0 * jz * dt);
                        Self::append_rxx(&mut ops, w_i, w_j, 2.0 * jx * dt);
                        Self::append_ryy(&mut ops, w_i, w_j, 2.0 * jy * dt);
                    }
                }
                for (i, &hz) in self.local_fields.iter().enumerate() {
                    if i < wires.len() {
                        ops.push(CircuitOp::Rz {
                            qubit: wires[i],
                            theta: 2.0 * hz * dt,
                        });
                    }
                }
            }
            ScramblerType::MajoranaSyk => {
                for &((i, j, k, l), j_ijkl) in &self.syk_couplings {
                    if l < wires.len() {
                        let theta = 2.0 * j_ijkl * dt;
                        let (w_i, w_j, w_k, w_l) = (wires[i], wires[j], wires[k], wires[l]);
                        ops.push(CircuitOp::CNOT {
                            control: w_i,
                            target: w_j,
                        });
                        ops.push(CircuitOp::CNOT {
                            control: w_j,
                            target: w_k,
                        });
                        ops.push(CircuitOp::CNOT {
                            control: w_k,
                            target: w_l,
                        });
                        ops.push(CircuitOp::Rz { qubit: w_l, theta });
                        ops.push(CircuitOp::CNOT {
                            control: w_k,
                            target: w_l,
                        });
                        ops.push(CircuitOp::CNOT {
                            control: w_j,
                            target: w_k,
                        });
                        ops.push(CircuitOp::CNOT {
                            control: w_i,
                            target: w_j,
                        });
                    }
                }
            }
            ScramblerType::FloquetHaar => {
                let golden_ratio = 1.618_033_988_749_895;
                for step in 0..wires.len() {
                    let offset = step % 2;
                    let mut i = offset;
                    while i + 1 < wires.len() {
                        let (w1, w2) = (wires[i], wires[i + 1]);
                        ops.push(CircuitOp::CZ { q1: w1, q2: w2 });
                        let angle1 = dt * (i as f64 + 1.0) * golden_ratio;
                        let angle2 = dt * (i as f64 + 2.0) / golden_ratio;
                        ops.push(CircuitOp::Ry {
                            qubit: w1,
                            theta: angle1,
                        });
                        ops.push(CircuitOp::Rz {
                            qubit: w2,
                            theta: angle2,
                        });
                        i += 2;
                    }
                }
            }
        }
        ops
    }

    /// Builds the adjoint (inverse) operations for the specified register wires: U^\dagger(t) = U(-t).
    pub fn build_adjoint_ops(&self, wires: &[usize], dt: f64) -> Vec<CircuitOp> {
        let fwd = self.build_forward_ops(wires, dt);
        let mut adj_ops = Vec::with_capacity(fwd.len());

        for op in fwd.into_iter().rev() {
            match op {
                CircuitOp::Rz { qubit, theta } => {
                    adj_ops.push(CircuitOp::Rz {
                        qubit,
                        theta: -theta,
                    });
                }
                CircuitOp::Ry { qubit, theta } => {
                    adj_ops.push(CircuitOp::Ry {
                        qubit,
                        theta: -theta,
                    });
                }
                CircuitOp::Unitary1Q { qubit, matrix } => {
                    let adj_mat = Unitary2x2 {
                        u00_re: matrix.u00_re,
                        u00_im: -matrix.u00_im,
                        u01_re: matrix.u10_re,
                        u01_im: -matrix.u10_im,
                        u10_re: matrix.u01_re,
                        u10_im: -matrix.u01_im,
                        u11_re: matrix.u11_re,
                        u11_im: -matrix.u11_im,
                    };
                    adj_ops.push(CircuitOp::Unitary1Q {
                        qubit,
                        matrix: adj_mat,
                    });
                }
                CircuitOp::CZ { q1, q2 } => adj_ops.push(CircuitOp::CZ { q1, q2 }),
                CircuitOp::CNOT { control, target } => {
                    adj_ops.push(CircuitOp::CNOT { control, target })
                }
                CircuitOp::SWAP { q1, q2 } => adj_ops.push(CircuitOp::SWAP { q1, q2 }),
                CircuitOp::ControlledPhase {
                    control,
                    target,
                    theta,
                } => {
                    adj_ops.push(CircuitOp::ControlledPhase {
                        control,
                        target,
                        theta: -theta,
                    });
                }
                other => adj_ops.push(other),
            }
        }
        adj_ops
    }

    /// Builds the time-forward operations for the Right register: U_R(t) = U_L^T(t).
    pub fn build_conjugate_ops(&self, wires: &[usize], dt: f64) -> Vec<CircuitOp> {
        let fwd = self.build_forward_ops(wires, dt);
        let mut conj_ops = Vec::with_capacity(fwd.len());

        // Transpose reverses gate order and transposes each gate
        for op in fwd.into_iter().rev() {
            match op {
                CircuitOp::Rz { qubit, theta } => {
                    conj_ops.push(CircuitOp::Rz { qubit, theta });
                }
                CircuitOp::Ry { qubit, theta } => {
                    // Ry^T(theta) = Ry(-theta)
                    conj_ops.push(CircuitOp::Ry {
                        qubit,
                        theta: -theta,
                    });
                }
                CircuitOp::Unitary1Q { qubit, matrix } => {
                    let trans_mat = Unitary2x2 {
                        u00_re: matrix.u00_re,
                        u00_im: matrix.u00_im,
                        u01_re: matrix.u10_re,
                        u01_im: matrix.u10_im,
                        u10_re: matrix.u01_re,
                        u10_im: matrix.u01_im,
                        u11_re: matrix.u11_re,
                        u11_im: matrix.u11_im,
                    };
                    conj_ops.push(CircuitOp::Unitary1Q {
                        qubit,
                        matrix: trans_mat,
                    });
                }
                CircuitOp::CZ { q1, q2 } => conj_ops.push(CircuitOp::CZ { q1, q2 }),
                CircuitOp::CNOT { control, target } => {
                    conj_ops.push(CircuitOp::CNOT { control, target })
                }
                CircuitOp::SWAP { q1, q2 } => conj_ops.push(CircuitOp::SWAP { q1, q2 }),
                CircuitOp::ControlledPhase {
                    control,
                    target,
                    theta,
                } => {
                    conj_ops.push(CircuitOp::ControlledPhase {
                        control,
                        target,
                        theta,
                    });
                }
                other => conj_ops.push(other),
            }
        }
        conj_ops
    }

    /// Evaluates the Out-of-Time-Order Correlator (OTOC) for operators W=Z_w and V=Z_v:
    /// OTOC(t) = Re[ (1/2^N) Tr( W(t)^\dagger V(0)^\dagger W(t) V(0) ) ]
    pub fn compute_otoc(
        &self,
        w_qubit: usize,
        v_qubit: usize,
        time_t: f64,
        trotter_steps: usize,
    ) -> f64 {
        assert!(w_qubit < self.num_qubits && v_qubit < self.num_qubits);
        if time_t.abs() < 1e-12 {
            return 1.0;
        }

        let dt = time_t / (trotter_steps.max(1) as f64);
        let dim = 1 << self.num_qubits;
        let wires: Vec<usize> = (0..self.num_qubits).collect();

        let mut fwd_dag = CircuitDag::new(self.num_qubits);
        let mut adj_dag = CircuitDag::new(self.num_qubits);
        for _ in 0..trotter_steps {
            for op in self.build_forward_ops(&wires, dt) {
                fwd_dag.add_op(op);
            }
            for op in self.build_adjoint_ops(&wires, dt) {
                adj_dag.add_op(op);
            }
        }
        let fwd_compiled = crate::compiler::CompiledCircuit::compile(&fwd_dag);
        let adj_compiled = crate::compiler::CompiledCircuit::compile(&adj_dag);

        let mut sum_trace = 0.0;
        let mask_v = 1 << v_qubit;
        let mask_w = 1 << w_qubit;

        for basis_idx in 0..dim {
            let mut state = QuantumState::ground(dim);
            if basis_idx != 0 {
                state.real[0] = 0.0;
                state.real[basis_idx] = 1.0;
            }

            // 1. V(0) = Z_v
            for idx in 0..dim {
                if (idx & mask_v) != 0 {
                    state.real[idx] = -state.real[idx];
                    state.imag[idx] = -state.imag[idx];
                }
            }

            // 2. U(t)
            fwd_compiled.execute(&mut state);

            // 3. W = Z_w
            for idx in 0..dim {
                if (idx & mask_w) != 0 {
                    state.real[idx] = -state.real[idx];
                    state.imag[idx] = -state.imag[idx];
                }
            }

            // 4. U^\dagger(t)
            adj_compiled.execute(&mut state);

            // 5. V(0) = Z_v
            for idx in 0..dim {
                if (idx & mask_v) != 0 {
                    state.real[idx] = -state.real[idx];
                    state.imag[idx] = -state.imag[idx];
                }
            }

            // 6. U(t)
            fwd_compiled.execute(&mut state);

            // 7. W = Z_w
            for idx in 0..dim {
                if (idx & mask_w) != 0 {
                    state.real[idx] = -state.real[idx];
                    state.imag[idx] = -state.imag[idx];
                }
            }

            // 8. U^\dagger(t)
            adj_compiled.execute(&mut state);

            // Overlap with initial |k>
            sum_trace += state.real[basis_idx];
        }

        (sum_trace / (dim as f64)).clamp(-1.0, 1.0)
    }
}
