//! Circuit Execution and Compiled Schedule Runtime.
//! Applies scheduled stages to QuantumState with cache-blocked traversals
//! and SIMD acceleration.

use super::dag::CircuitDag;
use super::fusion::optimize_dag_fusion;
use super::op::{CircuitOp, Unitary2x2};
use super::scheduler::{schedule_circuit_dag, ScheduledStage};
use crate::state::QuantumState;
use serde::{Deserialize, Serialize};

/// Telemetry metrics produced during circuit compilation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompilerTelemetry {
    /// Number of operations in the unoptimized raw input
    pub raw_ops: usize,
    /// Number of operations after single-qubit gate fusion and identity elimination
    pub fused_ops: usize,
    /// Critical path depth of the DAG
    pub critical_path_depth: usize,
    /// Total number of grouped execution stages
    pub stage_count: usize,
    /// Reduction in distinct memory passes: (raw_ops - stage_count) / raw_ops * 100%
    pub cache_pass_reduction_pct: f64,
    /// Gate fusion breakdown
    pub single_qubit_fused: usize,
    pub rz_merged: usize,
    pub ry_merged: usize,
    pub identities_eliminated: usize,
}

/// Fully compiled, schedule-grouped, and cache-optimized quantum circuit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledCircuit {
    pub num_qubits: usize,
    pub stages: Vec<ScheduledStage>,
    pub telemetry: CompilerTelemetry,
}

impl CompiledCircuit {
    /// Compiles a raw CircuitDag:
    /// 1. Analyzes dependency graph hazards.
    /// 2. Performs single-qubit gate fusion and algebraic merging.
    /// 3. Groups independent operations onto disjoint qubits into cache-sorted stages.
    pub fn compile(dag: &CircuitDag) -> Self {
        let raw_ops = dag.len();
        let (fused_dag, fusion_stats) = optimize_dag_fusion(dag);
        let critical_path_depth = fused_dag.critical_path_depth();
        let stages = schedule_circuit_dag(&fused_dag);
        let stage_count = stages.len();

        let cache_pass_reduction_pct = if raw_ops > 0 {
            let saved = raw_ops.saturating_sub(stage_count);
            (saved as f64 / raw_ops as f64) * 100.0
        } else {
            0.0
        };

        let telemetry = CompilerTelemetry {
            raw_ops,
            fused_ops: fused_dag.len(),
            critical_path_depth,
            stage_count,
            cache_pass_reduction_pct,
            single_qubit_fused: fusion_stats.single_qubit_fused,
            rz_merged: fusion_stats.rz_merged,
            ry_merged: fusion_stats.ry_merged,
            identities_eliminated: fusion_stats.identities_removed,
        };

        Self {
            num_qubits: dag.num_qubits,
            stages,
            telemetry,
        }
    }

    /// Executes the compiled circuit into a mutable QuantumState in-place.
    /// Reuses cache lines by executing grouped independent operations in cache order.
    /// Executes an individual operation against a mutable QuantumState.
    pub fn execute_op(state: &mut QuantumState, op: &CircuitOp) {
        match op {
            CircuitOp::Ry { qubit, theta } => {
                crate::simd::apply_ry_kernel(&mut state.real, &mut state.imag, *qubit, *theta);
            }
            CircuitOp::Rz { qubit, theta } => {
                crate::simd::apply_rz_kernel(&mut state.real, &mut state.imag, *qubit, *theta);
            }
            CircuitOp::Unitary1Q { qubit, matrix } => {
                apply_unitary2x2_op(&mut state.real, &mut state.imag, *qubit, matrix);
            }
            CircuitOp::CZ { q1, q2 } => {
                crate::simd::apply_cz_kernel(&mut state.real, &mut state.imag, *q1, *q2);
            }
            CircuitOp::CNOT { control, target } => {
                apply_cnot_op(&mut state.real, &mut state.imag, *control, *target);
            }
            CircuitOp::AntiCNOT { control, target } => {
                apply_anticnot_op(&mut state.real, &mut state.imag, *control, *target);
            }
            CircuitOp::SWAP { q1, q2 } => {
                apply_swap_op(&mut state.real, &mut state.imag, *q1, *q2);
            }
            CircuitOp::ControlledPhase {
                control,
                target,
                theta,
            } => {
                apply_controlled_phase_op(
                    &mut state.real,
                    &mut state.imag,
                    *control,
                    *target,
                    *theta,
                );
            }
            CircuitOp::RademacherPhase { signs } => {
                crate::simd::apply_rademacher_phase_kernel(&mut state.real, &mut state.imag, signs);
            }
        }
    }

    /// Executes the compiled circuit into a mutable QuantumState in-place.
    /// Reuses cache lines by executing grouped independent operations in cache order.
    pub fn execute(&self, state: &mut QuantumState) {
        assert_eq!(
            state.dim,
            1 << self.num_qubits,
            "State dimension {} does not match circuit qubit count {}",
            state.dim,
            self.num_qubits
        );

        let ops: Vec<&CircuitOp> = self.stages.iter().flat_map(|stage| stage.ops.iter()).collect();
        super::layered::execute_layered(&ops, self.num_qubits, state);
    }

    /// Executes every op as its own statevector pass. Reference path for the fused executor.
    pub fn execute_unfused(&self, state: &mut QuantumState) {
        assert_eq!(state.dim, 1 << self.num_qubits, "state does not match circuit width");
        for op in self.stages.iter().flat_map(|stage| stage.ops.iter()) {
            Self::execute_op(state, op);
        }
    }

    /// Executes the compiled circuit with stochastic open-system decoherence injected between stages.
    pub fn execute_with_noise(
        &self,
        state: &mut QuantumState,
        noise: &crate::coherence::NoiseChannelParams,
        stage_dt_us: f64,
        seed: u64,
    ) {
        assert_eq!(
            state.dim,
            1 << self.num_qubits,
            "State dimension {} does not match circuit qubit count {}",
            state.dim,
            self.num_qubits
        );
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

        for stage in &self.stages {
            for op in &stage.ops {
                Self::execute_op(state, op);
            }
            if stage_dt_us > 0.0 {
                for q in 0..self.num_qubits {
                    crate::coherence::apply_open_system_decoherence_qubit(
                        state,
                        q,
                        noise,
                        stage_dt_us,
                        &mut rng,
                    );
                }
            }
        }
    }
}

/// Applies a general 2x2 complex unitary gate to a single qubit in the statevector.
pub fn apply_unitary2x2_op(real: &mut [f64], imag: &mut [f64], qubit: usize, u: &Unitary2x2) {
    let n_qubits = crate::simd::assert_statevector(real, imag);
    crate::simd::assert_qubit(qubit, n_qubits);
    let dim = real.len();
    let mask = 1 << qubit;

    // Fast scalar implementation with optimal register reuse
    let u00_re = u.u00_re;
    let u00_im = u.u00_im;
    let u01_re = u.u01_re;
    let u01_im = u.u01_im;
    let u10_re = u.u10_re;
    let u10_im = u.u10_im;
    let u11_re = u.u11_re;
    let u11_im = u.u11_im;

    for i in 0..dim {
        if (i & mask) == 0 {
            let partner = i | mask;
            let r0 = real[i];
            let i0 = imag[i];
            let r1 = real[partner];
            let i1 = imag[partner];

            // c0' = u00 * c0 + u01 * c1
            let new_r0 = u00_re * r0 - u00_im * i0 + u01_re * r1 - u01_im * i1;
            let new_i0 = u00_re * i0 + u00_im * r0 + u01_re * i1 + u01_im * r1;

            // c1' = u10 * c0 + u11 * c1
            let new_r1 = u10_re * r0 - u10_im * i0 + u11_re * r1 - u11_im * i1;
            let new_i1 = u10_re * i0 + u10_im * r0 + u11_re * i1 + u11_im * r1;

            real[i] = new_r0;
            imag[i] = new_i0;
            real[partner] = new_r1;
            imag[partner] = new_i1;
        }
    }
}

/// Applies CNOT gate (flips target if control is 1).
pub fn apply_cnot_op(real: &mut [f64], imag: &mut [f64], control: usize, target: usize) {
    let n_qubits = crate::simd::assert_statevector(real, imag);
    crate::simd::assert_qubit_pair(control, target, n_qubits);
    let dim = real.len();
    let ctrl_mask = 1 << control;
    let target_mask = 1 << target;

    for i in 0..dim {
        if (i & ctrl_mask) != 0 && (i & target_mask) == 0 {
            let partner = i | target_mask;
            let r0 = real[i];
            let i0 = imag[i];
            let r1 = real[partner];
            let i1 = imag[partner];

            real[i] = r1;
            imag[i] = i1;
            real[partner] = r0;
            imag[partner] = i0;
        }
    }
}

/// Applies Hadamard gate: H = (1/√2) [1, 1; 1, -1]
pub fn apply_hadamard_op(real: &mut [f64], imag: &mut [f64], qubit: usize) {
    let n_qubits = crate::simd::assert_statevector(real, imag);
    crate::simd::assert_qubit(qubit, n_qubits);
    let dim = real.len();
    let mask = 1 << qubit;
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;

    for i in 0..dim {
        if (i & mask) == 0 {
            let partner = i | mask;
            let r0 = real[i];
            let i0 = imag[i];
            let r1 = real[partner];
            let i1 = imag[partner];

            real[i] = (r0 + r1) * inv_sqrt2;
            imag[i] = (i0 + i1) * inv_sqrt2;
            real[partner] = (r0 - r1) * inv_sqrt2;
            imag[partner] = (i0 - i1) * inv_sqrt2;
        }
    }
}

/// Applies AntiCNOT gate (flips target if control is 0).
pub fn apply_anticnot_op(real: &mut [f64], imag: &mut [f64], control: usize, target: usize) {
    let n_qubits = crate::simd::assert_statevector(real, imag);
    crate::simd::assert_qubit_pair(control, target, n_qubits);
    let dim = real.len();
    let ctrl_mask = 1 << control;
    let target_mask = 1 << target;

    for i in 0..dim {
        if (i & ctrl_mask) == 0 && (i & target_mask) == 0 {
            let partner = i | target_mask;
            let r0 = real[i];
            let i0 = imag[i];
            let r1 = real[partner];
            let i1 = imag[partner];

            real[i] = r1;
            imag[i] = i1;
            real[partner] = r0;
            imag[partner] = i0;
        }
    }
}

/// Applies two-qubit SWAP gate.
pub fn apply_swap_op(real: &mut [f64], imag: &mut [f64], q1: usize, q2: usize) {
    let n_qubits = crate::simd::assert_statevector(real, imag);
    crate::simd::assert_qubit_pair(q1, q2, n_qubits);
    let dim = real.len();
    let m1 = 1 << q1;
    let m2 = 1 << q2;

    for i in 0..dim {
        if (i & m1) != 0 && (i & m2) == 0 {
            let partner = (i ^ m1) | m2;
            let r0 = real[i];
            let i0 = imag[i];
            let r1 = real[partner];
            let i1 = imag[partner];

            real[i] = r1;
            imag[i] = i1;
            real[partner] = r0;
            imag[partner] = i0;
        }
    }
}

/// Applies two-qubit Controlled-Phase gate: multiplies state |11> by exp(i theta).
pub fn apply_controlled_phase_op(
    real: &mut [f64],
    imag: &mut [f64],
    control: usize,
    target: usize,
    theta: f64,
) {
    let n_qubits = crate::simd::assert_statevector(real, imag);
    crate::simd::assert_qubit_pair(control, target, n_qubits);
    let dim = real.len();
    let ctrl_mask = 1 << control;
    let target_mask = 1 << target;
    let both_mask = ctrl_mask | target_mask;

    let cos_t = theta.cos();
    let sin_t = theta.sin();

    for i in 0..dim {
        if (i & both_mask) == both_mask {
            let r = real[i];
            let im = imag[i];
            real[i] = r * cos_t - im * sin_t;
            imag[i] = r * sin_t + im * cos_t;
        }
    }
}

#[cfg(test)]
mod fused_tests {
    use super::super::CircuitBuilder;
    use crate::state::QuantumState;

    fn mixed_circuit(n: usize) -> super::CompiledCircuit {
        let mut b = CircuitBuilder::new(n);
        for l in 0..3 {
            for q in 0..n {
                b.ry(q, 0.37 * (1 + l + q) as f64).rz(q, 0.21 * (3 + l * 7 + q) as f64);
            }
            for q in 0..n {
                b.cz(q, (q + 1) % n);
            }
            b.h(l).cnot(l, n - 1 - l).ry(n - 1, 0.5).ry(n - 1, -0.2).swap(0, n - 1);
            for q in (0..n).step_by(3) {
                b.controlled_phase(q, (q + 5) % n, 0.9).rz(q, 1.3);
            }
        }
        b.compile()
    }

    #[test]
    fn fused_executor_matches_per_op_executor() {
        for n in [6, 13, 16, 17, 19] {
            let c = mixed_circuit(n);
            let mut fused = QuantumState::ground(1 << n);
            let mut plain = fused.clone();
            c.execute(&mut fused);
            c.execute_unfused(&mut plain);
            let dev = fused.real.iter().zip(&plain.real).chain(fused.imag.iter().zip(&plain.imag))
                .map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(dev < 1e-11, "n={n}: max deviation {dev}");
        }
    }
}
