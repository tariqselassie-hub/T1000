//! DAG-Based Dependency Tracker and Quantum Circuit Compiler.
//!
//! Provides automatic grouping of independent operations, causal dependency analysis,
//! gate fusion, and cache-footprint minimization across arbitrary/chaotic circuit codes.

pub mod dag;
pub mod executor;
pub mod fusion;
pub mod op;
pub mod scheduler;

pub use dag::{CircuitDag, DagNode};
pub use executor::{apply_cnot_op, apply_unitary2x2_op, CompiledCircuit, CompilerTelemetry};
pub use fusion::{optimize_dag_fusion, FusionStats};
pub use op::{CircuitOp, Unitary2x2};
pub use scheduler::{schedule_circuit_dag, ScheduledStage, StageKind};

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

    /// Appends a multi-qubit Walsh-Rademacher ±π phase layer.
    pub fn rademacher(&mut self, signs: Vec<f64>) -> &mut Self {
        self.dag.add_op(CircuitOp::RademacherPhase { signs });
        self
    }

    /// Compiles the circuit: resolves dependencies, fuses single-qubit gates,
    /// and groups independent operations into cache-optimized stages.
    pub fn compile(&self) -> CompiledCircuit {
        CompiledCircuit::compile(&self.dag)
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
        let id0 = builder.dag.add_op(CircuitOp::Ry { qubit: 0, theta: 0.5 });
        let id1 = builder.dag.add_op(CircuitOp::Ry { qubit: 1, theta: 0.8 });
        // CZ on (0, 1): depends on both id0 and id1
        let id2 = builder.dag.add_op(CircuitOp::CZ { q1: 0, q2: 1 });
        // Ry on qubit 2: completely independent of id0, id1, id2
        let id3 = builder.dag.add_op(CircuitOp::Ry { qubit: 2, theta: 1.2 });

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
        assert!((fid - 1.0).abs() < 1e-12, "Fidelity delta exceeds tolerance: {}", fid);
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
}
