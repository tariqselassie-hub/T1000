//! DAG-Based Dependency Tracker and Quantum Circuit Compiler.
//!
//! Provides automatic grouping of independent operations, causal dependency analysis,
//! gate fusion, and cache-footprint minimization across arbitrary/chaotic circuit codes.

pub mod dag;
pub mod executor;
pub mod fusion;
pub(crate) mod layered;
pub mod lifo_recycler;
pub mod op;
mod permute;
pub mod scheduler;

use crate::state::QuantumState;
pub use dag::{CircuitDag, DagNode};
pub use executor::{
    apply_anticnot_op, apply_cnot_op, apply_controlled_phase_op, apply_hadamard_op, apply_swap_op,
    apply_unitary2x2_op, CompiledCircuit, CompilerTelemetry,
};
pub use fusion::{optimize_dag_fusion, FusionStats};
pub use lifo_recycler::{LifoWireRecycler, RecyclingStats};
pub use op::{CircuitOp, Unitary2x2};
pub use scheduler::{
    compute_asap_alap, schedule_circuit_dag, schedule_circuit_dag_with_coherence_dd,
    AsapAlapSchedule, CoherenceScheduleTelemetry, ScheduledStage, StageKind,
};

/// Fluent builder for constructing quantum circuits and compiling through the DAG optimizer.
#[derive(Debug, Clone)]
pub struct CircuitBuilder {
    pub dag: CircuitDag,
}

impl CircuitBuilder {
    pub fn new(num_qubits: usize) -> Self {
        Self {
            dag: CircuitDag::new(num_qubits),
        }
    }

    /// Appends a parameterized Ry rotation.
    pub fn ry(&mut self, qubit: usize, theta: f64) -> &mut Self {
        self.dag.add_op(CircuitOp::Ry { qubit, theta });
        self
    }

    /// Appends a parameterized Rz rotation.
    pub fn rz(&mut self, qubit: usize, theta: f64) -> &mut Self {
        self.dag.add_op(CircuitOp::Rz { qubit, theta });
        self
    }

    /// Appends a single-qubit Hadamard gate.
    pub fn hadamard(&mut self, qubit: usize) -> &mut Self {
        self.dag.add_op(CircuitOp::Unitary1Q {
            qubit,
            matrix: Unitary2x2::hadamard(),
        });
        self
    }

    /// Shorthand alias for hadamard.
    pub fn h(&mut self, qubit: usize) -> &mut Self {
        self.hadamard(qubit)
    }

    /// Appends a general single-qubit 2x2 complex unitary.
    pub fn unitary_1q(&mut self, qubit: usize, matrix: Unitary2x2) -> &mut Self {
        self.dag.add_op(CircuitOp::Unitary1Q { qubit, matrix });
        self
    }

    /// Appends a two-qubit Controlled-Z (CZ) gate.
    pub fn cz(&mut self, q1: usize, q2: usize) -> &mut Self {
        self.dag.add_op(CircuitOp::CZ { q1, q2 });
        self
    }

    /// Appends a two-qubit CNOT gate.
    pub fn cnot(&mut self, control: usize, target: usize) -> &mut Self {
        self.dag.add_op(CircuitOp::CNOT { control, target });
        self
    }

    /// Appends a two-qubit AntiCNOT gate (flips target if control is 0).
    pub fn anticnot(&mut self, control: usize, target: usize) -> &mut Self {
        self.dag.add_op(CircuitOp::AntiCNOT { control, target });
        self
    }

    /// Appends a two-qubit SWAP gate.
    pub fn swap(&mut self, q1: usize, q2: usize) -> &mut Self {
        self.dag.add_op(CircuitOp::SWAP { q1, q2 });
        self
    }

    /// Appends a two-qubit Controlled-Phase gate: diag(1, 1, 1, exp(i theta)).
    pub fn controlled_phase(&mut self, control: usize, target: usize, theta: f64) -> &mut Self {
        self.dag.add_op(CircuitOp::ControlledPhase {
            control,
            target,
            theta,
        });
        self
    }

    /// Appends a multi-qubit Walsh-Rademacher ±π phase layer.
    pub fn rademacher(&mut self, signs: Vec<f64>) -> &mut Self {
        self.dag.add_op(CircuitOp::RademacherPhase { signs });
        self
    }

    /// Appends a Bell/GHZ state preparation block across the specified slice of qubits:
    /// `H(qubits[0])` followed by a chain of `CNOT(qubits[i-1], qubits[i])`.
    pub fn bell_ghz(&mut self, qubits: &[usize]) -> &mut Self {
        if qubits.is_empty() {
            return self;
        }
        self.hadamard(qubits[0]);
        for i in 1..qubits.len() {
            self.cnot(qubits[i - 1], qubits[i]);
        }
        self
    }

    /// Appends an Alternate Bell/GHZ state preparation block across the specified slice of qubits:
    /// `H(qubits[0])` followed by a chain of `AntiCNOT(qubits[i-1], qubits[i])`.
    pub fn alt_bell_ghz(&mut self, qubits: &[usize]) -> &mut Self {
        if qubits.is_empty() {
            return self;
        }
        self.hadamard(qubits[0]);
        for i in 1..qubits.len() {
            self.anticnot(qubits[i - 1], qubits[i]);
        }
        self
    }

    /// Appends a Quantum Fourier Transform (QFT) block across the specified slice of qubits.
    pub fn qft(&mut self, qubits: &[usize]) -> &mut Self {
        let n = qubits.len();
        if n == 0 {
            return self;
        }
        for j in 0..n {
            self.hadamard(qubits[j]);
            for k in (j + 1)..n {
                let theta = std::f64::consts::PI / (1usize << (k - j)) as f64;
                self.controlled_phase(qubits[k], qubits[j], theta);
            }
        }
        for j in 0..(n / 2) {
            self.swap(qubits[j], qubits[n - 1 - j]);
        }
        self
    }

    /// Appends an Inverse Quantum Fourier Transform (IQFT) block across the specified slice of qubits.
    pub fn iqft(&mut self, qubits: &[usize]) -> &mut Self {
        let n = qubits.len();
        if n == 0 {
            return self;
        }
        for j in 0..(n / 2) {
            self.swap(qubits[j], qubits[n - 1 - j]);
        }
        for j in (0..n).rev() {
            for k in ((j + 1)..n).rev() {
                let theta = -std::f64::consts::PI / (1usize << (k - j)) as f64;
                self.controlled_phase(qubits[k], qubits[j], theta);
            }
            self.hadamard(qubits[j]);
        }
        self
    }

    /// Appends Grover's Diffusion Operator (inversion about the mean).
    pub fn grover_diffusion(&mut self, qubits: &[usize]) -> &mut Self {
        crate::canonical::build_grover_diffusion(self, qubits);
        self
    }

    /// Appends a Bernstein-Vazirani parity oracle and measurement preparation.
    pub fn bernstein_vazirani(
        &mut self,
        qubits: &[usize],
        ancilla: usize,
        hidden_mask: usize,
    ) -> &mut Self {
        crate::canonical::build_bernstein_vazirani(self, qubits, ancilla, hidden_mask);
        self
    }

    /// Appends Quantum Phase Estimation (QPE) kernel.
    pub fn qpe(
        &mut self,
        precision_qubits: &[usize],
        target_qubit: usize,
        phase_phi: f64,
    ) -> &mut Self {
        crate::canonical::build_quantum_phase_estimation(
            self,
            precision_qubits,
            target_qubit,
            phase_phi,
        );
        self
    }

    /// Appends a Hardware-Efficient Ansatz (HEA) variational layer.
    pub fn hea_layer(
        &mut self,
        qubits: &[usize],
        thetas_y: &[f64],
        thetas_z: &[f64],
        topology: crate::canonical::EntanglerTopology,
    ) -> &mut Self {
        crate::canonical::build_hea_layer(self, qubits, thetas_y, thetas_z, topology);
        self
    }

    /// Appends a Trotter-Suzuki time step for Transverse-Field Ising model.
    pub fn trotter_ising_step(
        &mut self,
        qubit_pairs: &[(usize, usize)],
        qubits: &[usize],
        j_coupling: f64,
        h_transverse: f64,
        dt: f64,
    ) -> &mut Self {
        crate::canonical::build_trotter_ising_step(
            self,
            qubit_pairs,
            qubits,
            j_coupling,
            h_transverse,
            dt,
        );
        self
    }

    /// Appends a 1D MPS Linear Bloch-Zipper Anti-Correlation entangler chain across `qubits`.
    ///
    /// Chains adjacent pairs (q_i, q_{i+1}) with continuous complementary rotation and AntiCNOT
    /// entangling operations to lock down anti-correlation.
    pub fn linear_bloch_zipper(&mut self, qubits: &[usize]) -> &mut Self {
        if qubits.len() < 2 {
            return self;
        }
        for i in 0..(qubits.len() - 1) {
            self.anticnot(qubits[i], qubits[i + 1]);
        }
        self
    }

    /// Consumes the builder and returns the underlying dependency DAG.
    pub fn into_dag(self) -> CircuitDag {
        self.dag
    }

    /// Compiles the circuit: resolves dependencies, fuses single-qubit gates,
    /// and groups independent operations into cache-optimized stages.
    pub fn compile(&self) -> CompiledCircuit {
        CompiledCircuit::compile(&self.dag)
    }

    /// Compiles and immediately executes the circuit in-place into `state`.
    pub fn execute(&self, state: &mut QuantumState) -> CompiledCircuit {
        let compiled = self.compile();
        compiled.execute(state);
        compiled
    }

    /// Compiles and executes the circuit with open-system decoherence injected between stages.
    pub fn execute_with_noise(
        &self,
        state: &mut QuantumState,
        noise: &crate::coherence::NoiseChannelParams,
        stage_dt_us: f64,
        seed: u64,
    ) -> CompiledCircuit {
        let compiled = self.compile();
        compiled.execute_with_noise(state, noise, stage_dt_us, seed);
        compiled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::QuantumState;

    #[test]
    fn test_dag_dependency_tracking() {
        let mut builder = CircuitBuilder::new(4);
        // Qubit 0 and Qubit 1 gates: disjoint, independent
        let id0 = builder.dag.add_op(CircuitOp::Ry {
            qubit: 0,
            theta: 0.5,
        });
        let id1 = builder.dag.add_op(CircuitOp::Ry {
            qubit: 1,
            theta: 0.8,
        });
        // CZ on (0, 1): depends on both id0 and id1
        let id2 = builder.dag.add_op(CircuitOp::CZ { q1: 0, q2: 1 });
        // Ry on qubit 2: completely independent of id0, id1, id2
        let id3 = builder.dag.add_op(CircuitOp::Ry {
            qubit: 2,
            theta: 1.2,
        });

        assert!(builder.dag.are_independent(id0, id1));
        assert!(builder.dag.are_independent(id0, id3));
        assert!(builder.dag.are_independent(id1, id3));
        assert!(builder.dag.are_independent(id2, id3));
        assert!(builder.dag.is_ancestor_of(id0, id2));
        assert!(builder.dag.is_ancestor_of(id1, id2));
    }

    #[test]
    fn test_single_qubit_gate_fusion() {
        let n_qubits = 10;
        let dim = 1 << n_qubits;
        let mut builder = CircuitBuilder::new(n_qubits);
        // Ry then Rz on same qubit: should fuse into Unitary1Q
        builder.ry(0, 0.4);
        builder.rz(0, 0.7);

        let compiled = builder.compile();
        assert_eq!(compiled.telemetry.raw_ops, 2);
        assert_eq!(compiled.telemetry.fused_ops, 1);
        assert_eq!(compiled.telemetry.single_qubit_fused, 1);

        // Verify output fidelity parity against sequential application
        let mut state_raw = QuantumState::new(dim);
        crate::simd::apply_ry_kernel(&mut state_raw.real, &mut state_raw.imag, 0, 0.4);
        crate::simd::apply_rz_kernel(&mut state_raw.real, &mut state_raw.imag, 0, 0.7);

        let mut state_compiled = QuantumState::new(dim);
        compiled.execute(&mut state_compiled);

        let fid = state_raw.fidelity(&state_compiled);
        assert!(
            (fid - 1.0).abs() < 1e-12,
            "Fidelity delta exceeds tolerance: {}",
            fid
        );
    }

    #[test]
    fn test_stage_scheduling_disjoint_grouping() {
        let mut builder = CircuitBuilder::new(4);
        // 4 independent single-qubit gates on 4 disjoint qubits
        builder.ry(0, 0.1);
        builder.ry(1, 0.2);
        builder.ry(2, 0.3);
        builder.ry(3, 0.4);

        let compiled = builder.compile();
        // All 4 should be grouped into a single parallel stage!
        assert_eq!(compiled.stages.len(), 1);
        assert_eq!(compiled.stages[0].ops.len(), 4);
        assert_eq!(compiled.stages[0].kind, StageKind::SingleQubitParallel);
    }

    #[test]
    fn test_chaotic_circuit_compilation_and_fidelity() {
        let n_qubits = 10;
        let dim = 1 << n_qubits;
        let mut builder = CircuitBuilder::new(n_qubits);

        // Submit chaotic, interleaved code:
        // Gates submitted out of order with split dependencies
        builder.ry(0, 0.5);
        builder.rz(3, 0.9);
        builder.ry(1, 0.3);
        builder.rz(0, 0.7); // Will fuse with Ry(0, 0.5)
        builder.cz(2, 4);
        builder.ry(5, 1.1);
        builder.rz(1, -0.4); // Will fuse with Ry(1, 0.3)
        builder.cz(0, 1);
        builder.ry(3, -0.6); // Will fuse with Rz(3, 0.9)
        builder.cz(3, 5);
        builder.rz(2, 0.8);
        builder.cz(1, 2);

        let compiled = builder.compile();

        // Check optimization telemetry
        assert!(compiled.telemetry.raw_ops > compiled.telemetry.fused_ops);
        assert!(compiled.telemetry.cache_pass_reduction_pct > 30.0);

        // Execute compiled circuit
        let mut state_compiled = QuantumState::new(dim);
        compiled.execute(&mut state_compiled);

        // Execute uncompiled raw reference circuit sequentially
        let mut state_raw = QuantumState::new(dim);
        crate::simd::apply_ry_kernel(&mut state_raw.real, &mut state_raw.imag, 0, 0.5);
        crate::simd::apply_rz_kernel(&mut state_raw.real, &mut state_raw.imag, 3, 0.9);
        crate::simd::apply_ry_kernel(&mut state_raw.real, &mut state_raw.imag, 1, 0.3);
        crate::simd::apply_rz_kernel(&mut state_raw.real, &mut state_raw.imag, 0, 0.7);
        crate::simd::apply_cz_kernel(&mut state_raw.real, &mut state_raw.imag, 2, 4);
        crate::simd::apply_ry_kernel(&mut state_raw.real, &mut state_raw.imag, 5, 1.1);
        crate::simd::apply_rz_kernel(&mut state_raw.real, &mut state_raw.imag, 1, -0.4);
        crate::simd::apply_cz_kernel(&mut state_raw.real, &mut state_raw.imag, 0, 1);
        crate::simd::apply_ry_kernel(&mut state_raw.real, &mut state_raw.imag, 3, -0.6);
        crate::simd::apply_cz_kernel(&mut state_raw.real, &mut state_raw.imag, 3, 5);
        crate::simd::apply_rz_kernel(&mut state_raw.real, &mut state_raw.imag, 2, 0.8);
        crate::simd::apply_cz_kernel(&mut state_raw.real, &mut state_raw.imag, 1, 2);

        let fid = state_raw.fidelity(&state_compiled);
        assert!(
            (fid - 1.0).abs() < 1e-12,
            "Chaotic code execution fidelity mismatch: {}",
            fid
        );
    }

    #[test]
    fn test_linear_bloch_zipper_builder() {
        let mut builder = CircuitBuilder::new(4);
        builder.ry(0, std::f64::consts::PI); // Prepare Q0 in |1>
        builder.linear_bloch_zipper(&[0, 1, 2, 3]);

        let compiled = builder.compile();
        let mut state = QuantumState::ground(16);
        compiled.execute(&mut state);

        // AntiCNOT chain: Q0=1 -> Q1=0 -> Q2=1 -> Q3=0 -> |0101> (idx = 1010b in little-endian wire order)
        // With wire 0 as LSB: bit 0 is 1, bit 1 is 0, bit 2 is 1, bit 3 is 0 => 1 + 4 = 5 (0101b = 5)
        let prob_5 = state.real[5] * state.real[5] + state.imag[5] * state.imag[5];
        assert!(
            (prob_5 - 1.0).abs() < 1e-6,
            "Expected state |0101> prob 1.0, got {}",
            prob_5
        );
    }
}
