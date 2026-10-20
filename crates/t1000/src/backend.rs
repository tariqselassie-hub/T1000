// SPDX-License-Identifier: MIT OR Apache-2.0
//! Dynamic execution backend selection: exact FP64 AVX2 CPU vs FP32 iGPU.
//!
//! Policy: `Precision::Exact` (default) always runs the FP64 AVX2 engine and never touches
//! OpenCL. `Precision::Fast` lets the selector route large circuits to the FP32 GPU and
//! fall back to the CPU on any GPU error or unsupported op.
//!
//! Environment overrides (read by [`BackendPolicy::from_env`]):
//! * `T1000_PRECISION` = `exact` | `fast`
//! * `T1000_BACKEND`   = `auto` | `cpu` | `gpu` (`gpu` implies fast precision)
//! * `T1000_GPU_MIN_QUBITS` = integer crossover threshold

use crate::compiler::CompiledCircuit;
use crate::gpu::fp32::Fp32GpuEngine;
use crate::state::QuantumState;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Numerical precision the caller accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Precision {
    /// FP64 only (default). Bit-for-bit the existing CPU engine.
    Exact,
    /// FP32 allowed (~1e-7 relative error) when it is faster.
    Fast,
}

/// Which engine ran (or should run) a circuit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendKind {
    CpuAvx2Fp64,
    IGpuFp32,
}

/// Selection policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendPolicy {
    pub precision: Precision,
    /// Smallest register routed to the GPU in `Fast` mode.
    /// Default 16: on the i3-1115G4 + UHD, `backend_auto_bench` shows the GPU first wins
    /// clearly (>2x, transfers included) at 16 qubits and is marginal (1.2x) at 14.
    /// Machine-specific; override with `T1000_GPU_MIN_QUBITS`.
    pub gpu_min_qubits: usize,
    /// Pin a backend regardless of size (GPU pin still requires `Fast`).
    pub force: Option<BackendKind>,
}

impl Default for BackendPolicy {
    fn default() -> Self {
        Self { precision: Precision::Exact, gpu_min_qubits: 16, force: None }
    }
}

impl BackendPolicy {
    pub fn fast() -> Self {
        Self { precision: Precision::Fast, ..Self::default() }
    }

    pub fn from_env() -> Self {
        let mut p = Self::default();
        let get = |k: &str| std::env::var(k).ok().map(|v| v.to_ascii_lowercase());
        match get("T1000_PRECISION").as_deref() {
            Some("fast") => p.precision = Precision::Fast,
            Some("exact") => p.precision = Precision::Exact,
            _ => {}
        }
        match get("T1000_BACKEND").as_deref() {
            Some("cpu") => p.force = Some(BackendKind::CpuAvx2Fp64),
            Some("gpu") => {
                p.force = Some(BackendKind::IGpuFp32);
                p.precision = Precision::Fast;
            }
            _ => {}
        }
        if let Some(n) = get("T1000_GPU_MIN_QUBITS").and_then(|v| v.parse().ok()) {
            p.gpu_min_qubits = n;
        }
        p
    }

    /// Pure routing decision, ignoring whether a GPU is actually mounted.
    pub fn choose(&self, num_qubits: usize) -> BackendKind {
        if self.precision == Precision::Exact || self.force == Some(BackendKind::CpuAvx2Fp64) {
            return BackendKind::CpuAvx2Fp64;
        }
        if self.force == Some(BackendKind::IGpuFp32) || num_qubits >= self.gpu_min_qubits {
            BackendKind::IGpuFp32
        } else {
            BackendKind::CpuAvx2Fp64
        }
    }
}

/// What actually happened during a run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendExecutionReport {
    pub requested: BackendKind,
    pub used: BackendKind,
    pub device: String,
    pub duration_ms: f64,
    /// Set when the GPU was requested but the CPU ran instead.
    pub fallback_reason: Option<String>,
}

/// Owns the (lazily mounted) GPU engine and applies the policy.
#[allow(missing_debug_implementations)]
pub struct BackendSelector {
    pub policy: BackendPolicy,
    gpu: Option<Fp32GpuEngine>,
}

impl BackendSelector {
    /// Mounts OpenCL only when the policy can ever use it (`Fast`).
    pub fn new(policy: BackendPolicy) -> Self {
        let gpu = (policy.precision == Precision::Fast && policy.force != Some(BackendKind::CpuAvx2Fp64))
            .then(Fp32GpuEngine::new)
            .flatten();
        Self { policy, gpu }
    }

    pub fn from_env() -> Self {
        Self::new(BackendPolicy::from_env())
    }

    pub fn gpu_device(&self) -> Option<&str> {
        self.gpu.as_ref().map(|g| g.device_name.as_str())
    }

    /// Executes `circuit` on the backend the policy picks, falling back to the CPU if needed.
    pub fn execute(&self, circuit: &CompiledCircuit, state: &mut QuantumState) -> BackendExecutionReport {
        let requested = self.policy.choose(circuit.num_qubits);
        let t0 = Instant::now();
        let mut fallback_reason = None;

        if requested == BackendKind::IGpuFp32 {
            match self.gpu.as_ref() {
                None => fallback_reason = Some("no usable FP32 OpenCL GPU".to_string()),
                Some(g) => match g.execute(circuit, state) {
                    Ok(()) => {
                        return BackendExecutionReport {
                            requested,
                            used: BackendKind::IGpuFp32,
                            device: g.device_name.clone(),
                            duration_ms: t0.elapsed().as_secs_f64() * 1e3,
                            fallback_reason: None,
                        };
                    }
                    Err(m) => fallback_reason = Some(m),
                },
            }
        }

        circuit.execute(state);
        BackendExecutionReport {
            requested,
            used: BackendKind::CpuAvx2Fp64,
            device: "CPU AVX2+FMA (FP64)".to_string(),
            duration_ms: t0.elapsed().as_secs_f64() * 1e3,
            fallback_reason,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::CircuitBuilder;

    #[test]
    fn exact_never_routes_to_gpu() {
        let p = BackendPolicy::default();
        for n in [4, 16, 28] {
            assert_eq!(p.choose(n), BackendKind::CpuAvx2Fp64);
        }
    }

    #[test]
    fn fast_routes_by_threshold() {
        let p = BackendPolicy::fast();
        assert_eq!(p.choose(15), BackendKind::CpuAvx2Fp64);
        assert_eq!(p.choose(16), BackendKind::IGpuFp32);
    }

    #[test]
    fn force_cpu_beats_fast() {
        let p = BackendPolicy { force: Some(BackendKind::CpuAvx2Fp64), ..BackendPolicy::fast() };
        assert_eq!(p.choose(26), BackendKind::CpuAvx2Fp64);
    }

    #[test]
    fn exact_policy_matches_cpu_bit_for_bit() {
        let mut b = CircuitBuilder::new(6);
        b.h(0).cnot(0, 1).ry(2, 0.7).cz(1, 2);
        let c = b.compile();
        let mut s1 = QuantumState::ground(1 << 6);
        let mut s2 = s1.clone();
        c.execute(&mut s1);
        let rep = BackendSelector::new(BackendPolicy::default()).execute(&c, &mut s2);
        assert_eq!(rep.used, BackendKind::CpuAvx2Fp64);
        assert_eq!(s1.real, s2.real);
        assert_eq!(s1.imag, s2.imag);
    }

    /// Differential test: every op type, GPU FP32 vs CPU FP64. Skips without an OpenCL GPU.
    #[test]
    fn gpu_fp32_matches_cpu_on_all_ops() {
        let sel = BackendSelector::new(BackendPolicy {
            force: Some(BackendKind::IGpuFp32),
            ..BackendPolicy::fast()
        });
        if sel.gpu_device().is_none() {
            eprintln!("no FP32 OpenCL GPU; skipping");
            return;
        }
        for n in [3usize, 5, 8, 11] {
            let mut b = CircuitBuilder::new(n);
            for q in 0..n {
                b.h(q).ry(q, 0.3 + q as f64).rz(q, 1.1 * q as f64 - 0.4);
            }
            for q in 0..n - 1 {
                b.cnot(q, q + 1).anticnot(q + 1, q).cz(q, q + 1);
                b.controlled_phase(q, q + 1, 0.77).swap(q, q + 1);
            }
            b.cnot(n - 1, 0).swap(0, n - 1).cnot(0, n - 1); // reversed/wrapped qubit order
            b.unitary_1q(
                n / 2,
                crate::compiler::Unitary2x2::new(
                    num_complex::Complex64::new(0.6, 0.0),
                    num_complex::Complex64::new(0.0, 0.8),
                    num_complex::Complex64::new(0.0, 0.8),
                    num_complex::Complex64::new(0.6, 0.0),
                ),
            );
            let c = b.compile();
            let mut cpu = QuantumState::ground(1 << n);
            let mut gpu = cpu.clone();
            c.execute(&mut cpu);
            let rep = sel.execute(&c, &mut gpu);
            assert_eq!(rep.used, BackendKind::IGpuFp32, "n={n}: {:?}", rep.fallback_reason);
            let max = cpu
                .real
                .iter()
                .zip(&gpu.real)
                .chain(cpu.imag.iter().zip(&gpu.imag))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(max < 1e-5, "n={n}: max diff {max}");
        }
    }

    #[test]
    fn rademacher_falls_back_to_cpu() {
        let sel = BackendSelector::new(BackendPolicy {
            force: Some(BackendKind::IGpuFp32),
            ..BackendPolicy::fast()
        });
        let mut b = CircuitBuilder::new(3);
        b.rademacher(vec![1.0, -1.0, 1.0]);
        let c = b.compile();
        let mut s = QuantumState::new(1 << 3);
        let rep = sel.execute(&c, &mut s);
        assert_eq!(rep.used, BackendKind::CpuAvx2Fp64);
        assert!(rep.fallback_reason.is_some());
    }
}
