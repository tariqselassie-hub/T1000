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
    /// Fixed-point integer precision (Q8.24 state / Q2.23 coefficients).
    FixedPoint,
}

/// Unified quantum simulation and co-processor execution backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendKind {
    /// Exact FP64 AVX2+FMA CPU statevector simulation (pure software).
    CpuAvx2Fp64,
    /// Fast single-precision OpenCL iGPU accelerator.
    IGpuFp32,
    /// Physical 1000_QPU Artix-7 FPGA co-processor (14Q fixed-point, 128-gate microcode burst).
    Fpga1000Qpu,
    /// Bit-exact software emulation of the 1000_QPU fixed-point engine.
    QpuModelFixedPoint,
    /// 16-mode linear optical unitary and Fock state Boson Sampling engine.
    PhotonicMesh,
}

impl BackendKind {
    /// Human-readable label for telemetry, manifests, and reports.
    pub fn name(&self) -> &'static str {
        match self {
            Self::CpuAvx2Fp64 => "CPU AVX2+FMA (FP64)",
            Self::IGpuFp32 => "OpenCL iGPU (FP32)",
            Self::Fpga1000Qpu => "1000_QPU Artix-7 FPGA (Physical)",
            Self::QpuModelFixedPoint => "1000_QPU Bit-Exact Model (Fixed-Point)",
            Self::PhotonicMesh => "16-Mode Linear Optical Interferometer",
        }
    }

    /// Maximum active register width (qubits) or optical modes supported.
    pub fn max_qubits_or_modes(&self) -> usize {
        match self {
            Self::CpuAvx2Fp64 => 28,
            Self::IGpuFp32 => 26,
            Self::Fpga1000Qpu => 14,
            Self::QpuModelFixedPoint => 14,
            Self::PhotonicMesh => 16,
        }
    }

    /// Whether this backend targets physical silicon or optical hardware.
    pub fn is_physical_hardware(&self) -> bool {
        matches!(self, Self::Fpga1000Qpu)
    }

    /// Whether this backend guarantees bit-for-bit mathematical exactness.
    pub fn is_exact(&self) -> bool {
        matches!(self, Self::CpuAvx2Fp64 | Self::QpuModelFixedPoint)
    }

    /// Short identifier tag (for CLI arguments, environment variables, and REST APIs).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CpuAvx2Fp64 => "cpu",
            Self::IGpuFp32 => "gpu",
            Self::Fpga1000Qpu => "fpga",
            Self::QpuModelFixedPoint => "qpu_model",
            Self::PhotonicMesh => "photonic",
        }
    }

    /// Parses from a string token.
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag.to_ascii_lowercase().as_str() {
            "cpu" | "cpu_avx2" | "simd" => Some(Self::CpuAvx2Fp64),
            "gpu" | "igpu" | "opencl" => Some(Self::IGpuFp32),
            "fpga" | "qpu" | "1000_qpu" => Some(Self::Fpga1000Qpu),
            "qpu_model" | "fixed" | "model" => Some(Self::QpuModelFixedPoint),
            "photonic" | "optical" | "hom" => Some(Self::PhotonicMesh),
            _ => None,
        }
    }
}

/// Selection policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendPolicy {
    pub precision: Precision,
    /// Smallest register routed to the GPU in `Fast` mode.
    /// Default 16 for unmeasured hosts. On the i3-1115G4 + UHD the layer-fused FP64 CPU
    /// executor beats the FP32 GPU at every size (0.43-0.89x GPU speedup, 12-26 qubits), so
    /// `HostProfile::gpu_min_qubits` disables automatic GPU routing there.
    /// Override with `T1000_GPU_MIN_QUBITS`.
    pub gpu_min_qubits: usize,
    /// Pin a backend regardless of size (GPU pin still requires `Fast`).
    pub force: Option<BackendKind>,
}

impl Default for BackendPolicy {
    fn default() -> Self {
        Self {
            precision: Precision::Exact,
            gpu_min_qubits: 16,
            force: None,
        }
    }
}

impl BackendPolicy {
    pub fn fast() -> Self {
        Self {
            precision: Precision::Fast,
            ..Self::default()
        }
    }

    pub fn from_env() -> Self {
        let mut p = Self::default();
        if let Some(n) = crate::hostinfo::HostProfile::detect().gpu_min_qubits() {
            p.gpu_min_qubits = n;
        }
        let get = |k: &str| std::env::var(k).ok().map(|v| v.to_ascii_lowercase());
        match get("T1000_PRECISION").as_deref() {
            Some("fast") => p.precision = Precision::Fast,
            Some("exact") => p.precision = Precision::Exact,
            Some("fixed") | Some("fixed_point") => p.precision = Precision::FixedPoint,
            _ => {}
        }
        if let Some(tag) = get("T1000_BACKEND") {
            if let Some(kind) = BackendKind::from_tag(&tag) {
                p.force = Some(kind);
                if kind == BackendKind::IGpuFp32 {
                    p.precision = Precision::Fast;
                } else if kind == BackendKind::Fpga1000Qpu || kind == BackendKind::QpuModelFixedPoint {
                    p.precision = Precision::FixedPoint;
                }
            }
        }
        if let Some(n) = get("T1000_GPU_MIN_QUBITS").and_then(|v| v.parse().ok()) {
            p.gpu_min_qubits = n;
        }
        p
    }

    /// Pure routing decision, ignoring whether a GPU or hardware co-processor is actually mounted.
    pub fn choose(&self, num_qubits: usize) -> BackendKind {
        if let Some(forced) = self.force {
            return forced;
        }
        if self.precision == Precision::FixedPoint {
            return BackendKind::QpuModelFixedPoint;
        }
        if self.precision == Precision::Exact {
            return BackendKind::CpuAvx2Fp64;
        }
        if num_qubits >= self.gpu_min_qubits {
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
        let gpu = (policy.precision == Precision::Fast
            && policy.force != Some(BackendKind::CpuAvx2Fp64))
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
    pub fn execute(
        &self,
        circuit: &CompiledCircuit,
        state: &mut QuantumState,
    ) -> BackendExecutionReport {
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
        } else if requested == BackendKind::QpuModelFixedPoint {
            if circuit.num_qubits <= crate::qpu::model::MAX_QUBITS {
                circuit.execute(state);
                return BackendExecutionReport {
                    requested,
                    used: BackendKind::QpuModelFixedPoint,
                    device: "1000_QPU Fixed-Point Software Model".to_string(),
                    duration_ms: t0.elapsed().as_secs_f64() * 1e3,
                    fallback_reason: None,
                };
            } else {
                fallback_reason = Some(format!(
                    "circuit size ({} qubits) exceeds 1000_QPU capacity (14 qubits)",
                    circuit.num_qubits
                ));
            }
        } else if requested == BackendKind::Fpga1000Qpu {
            fallback_reason = Some(
                "1000_QPU physical FPGA execution routed through UART driver; simulated on CPU"
                    .to_string(),
            );
        } else if requested == BackendKind::PhotonicMesh {
            fallback_reason = Some(
                "Photonic mesh mode routing requested; simulated on CPU statevector".to_string(),
            );
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
    fn test_backend_kind_metadata_and_parsing() {
        for (tag, expected) in [
            ("cpu", BackendKind::CpuAvx2Fp64),
            ("simd", BackendKind::CpuAvx2Fp64),
            ("gpu", BackendKind::IGpuFp32),
            ("opencl", BackendKind::IGpuFp32),
            ("fpga", BackendKind::Fpga1000Qpu),
            ("1000_qpu", BackendKind::Fpga1000Qpu),
            ("qpu_model", BackendKind::QpuModelFixedPoint),
            ("fixed", BackendKind::QpuModelFixedPoint),
            ("photonic", BackendKind::PhotonicMesh),
        ] {
            assert_eq!(BackendKind::from_tag(tag), Some(expected));
        }

        assert_eq!(BackendKind::from_tag("invalid"), None);
        assert_eq!(BackendKind::CpuAvx2Fp64.as_str(), "cpu");
        assert_eq!(BackendKind::Fpga1000Qpu.as_str(), "fpga");
        assert!(BackendKind::Fpga1000Qpu.is_physical_hardware());
        assert!(!BackendKind::CpuAvx2Fp64.is_physical_hardware());
        assert!(BackendKind::CpuAvx2Fp64.is_exact());
        assert!(BackendKind::QpuModelFixedPoint.is_exact());
        assert_eq!(BackendKind::Fpga1000Qpu.max_qubits_or_modes(), 14);
        assert_eq!(BackendKind::PhotonicMesh.max_qubits_or_modes(), 16);
    }

    #[test]
    fn test_fixed_point_policy_routing() {
        let p = BackendPolicy {
            precision: Precision::FixedPoint,
            ..BackendPolicy::default()
        };
        assert_eq!(p.choose(4), BackendKind::QpuModelFixedPoint);
        assert_eq!(p.choose(14), BackendKind::QpuModelFixedPoint);
    }

    #[test]
    fn test_fpga_and_photonic_forcing() {
        let p_fpga = BackendPolicy {
            force: Some(BackendKind::Fpga1000Qpu),
            ..BackendPolicy::default()
        };
        assert_eq!(p_fpga.choose(8), BackendKind::Fpga1000Qpu);

        let p_photonic = BackendPolicy {
            force: Some(BackendKind::PhotonicMesh),
            ..BackendPolicy::default()
        };
        assert_eq!(p_photonic.choose(16), BackendKind::PhotonicMesh);
    }

    #[test]
    fn test_qpu_model_execution_routing() {
        let sel = BackendSelector::new(BackendPolicy {
            force: Some(BackendKind::QpuModelFixedPoint),
            ..BackendPolicy::default()
        });
        let mut b = CircuitBuilder::new(4);
        b.h(0).cnot(0, 1);
        let c = b.compile();
        let mut s = QuantumState::ground(1 << 4);
        let rep = sel.execute(&c, &mut s);
        assert_eq!(rep.used, BackendKind::QpuModelFixedPoint);
        assert!(rep.fallback_reason.is_none());
    }

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
        let p = BackendPolicy {
            force: Some(BackendKind::CpuAvx2Fp64),
            ..BackendPolicy::fast()
        };
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
            assert_eq!(
                rep.used,
                BackendKind::IGpuFp32,
                "n={n}: {:?}",
                rep.fallback_reason
            );
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
