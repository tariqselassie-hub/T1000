// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Heterogeneous Zero-Copy iGPU Co-Processor Bridge
//!
//! Bridges large-scale statevectors ($N \ge 22$, $\ge 64\text{ MB}$) directly into the
//! Intel Graphics OpenCL runtime via `CL_MEM_USE_HOST_PTR` unified memory mappings.
//! For $N < 22$, automatically routes to the CPU L3-cache resident AVX2+FMA SIMD engine.

pub mod fp32;
pub mod opencl;

use crate::state::QuantumState;
use opencl::OpenClGpuContext;
use serde::{Deserialize, Serialize};

/// Configuration for dynamic execution routing between CPU AVX2 and iGPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IGpuBridgeConfig {
    /// Minimum qubit register width for routing to the iGPU (default: 22 qubits = 64 MB).
    pub threshold_qubits: usize,
    /// Force routing to iGPU if hardware is detected regardless of register width.
    pub force_igpu: bool,
}

impl Default for IGpuBridgeConfig {
    fn default() -> Self {
        Self {
            threshold_qubits: 22,
            force_igpu: false,
        }
    }
}

/// Execution telemetry emitted after a gate evolution pass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IGpuExecutionReport {
    pub backend_used: String,
    pub zero_copy_uma: bool,
    pub qubit: usize,
    pub dim: usize,
    pub duration_us: f64,
}

/// Zero-Copy Co-Processor Bridge coordinating Lion Cove CPU SIMD and Intel Xe Graphics.
pub struct IGpuBridge {
    pub config: IGpuBridgeConfig,
    pub context: Option<OpenClGpuContext>,
}

impl IGpuBridge {
    /// Attempts to initialize the zero-copy co-processor bridge.
    pub fn new(config: IGpuBridgeConfig) -> Self {
        let context = OpenClGpuContext::initialize();
        Self { config, context }
    }

    /// Returns `true` if an active Intel OpenCL Graphics device was mounted.
    pub fn is_available(&self) -> bool {
        self.context.is_some()
    }

    /// Descriptive device name of the mounted accelerator (e.g. "Intel(R) Graphics").
    pub fn device_name(&self) -> Option<&str> {
        self.context.as_ref().map(|ctx| ctx.device_name.as_str())
    }

    /// Descriptive platform name (e.g. "Intel(R) OpenCL Graphics").
    pub fn platform_name(&self) -> Option<&str> {
        self.context.as_ref().map(|ctx| ctx.platform_name.as_str())
    }

    /// Applies single-qubit Ry rotation with dynamic hardware routing.
    pub fn apply_ry(
        &self,
        state: &mut QuantumState,
        qubit: usize,
        theta: f64,
    ) -> IGpuExecutionReport {
        let n_qubits = state.dim.trailing_zeros() as usize;
        let dim = state.dim;
        let use_gpu = self.is_available()
            && (n_qubits >= self.config.threshold_qubits || self.config.force_igpu);

        let t0 = std::time::Instant::now();
        if use_gpu {
            if let Some(ref ctx) = self.context {
                if ctx
                    .execute_ry_zero_copy(&mut state.real, &mut state.imag, qubit, theta)
                    .is_ok()
                {
                    let dur_us = t0.elapsed().as_secs_f64() * 1e6;
                    return IGpuExecutionReport {
                        backend_used: format!(
                            "Intel Xe iGPU ({}) [Zero-Copy UMA]",
                            ctx.device_name
                        ),
                        zero_copy_uma: true,
                        qubit,
                        dim,
                        duration_us: dur_us,
                    };
                }
            }
        }

        // CPU AVX2 Fallback / L3 cache path
        crate::simd::apply_ry_kernel(&mut state.real, &mut state.imag, qubit, theta);
        let dur_us = t0.elapsed().as_secs_f64() * 1e6;
        IGpuExecutionReport {
            backend_used: "Lion Cove CPU AVX2+FMA (L3-Cache Pinned)".to_string(),
            zero_copy_uma: false,
            qubit,
            dim,
            duration_us: dur_us,
        }
    }
}

impl Default for IGpuBridge {
    fn default() -> Self {
        Self::new(IGpuBridgeConfig::default())
    }
}

impl std::fmt::Debug for IGpuBridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IGpuBridge")
            .field("config", &self.config)
            .field("is_available", &self.is_available())
            .field("device_name", &self.device_name())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_igpu_bridge_device_discovery() {
        let bridge = IGpuBridge::default();
        println!("iGPU Bridge Available : {}", bridge.is_available());
        if let Some(dev) = bridge.device_name() {
            println!("Mounted GPU Device    : {}", dev);
        }
        if let Some(plat) = bridge.platform_name() {
            println!("Mounted OpenCL Plat   : {}", plat);
        }
    }

    #[test]
    fn test_igpu_zero_copy_ry_numerical_precision() {
        let bridge = IGpuBridge::new(IGpuBridgeConfig {
            threshold_qubits: 4,
            force_igpu: true,
        });

        if !bridge.is_available() {
            println!("OpenCL runtime not mounted; skipping hardware pass.");
            return;
        }

        let mut s_gpu = QuantumState::ground(16);
        let mut s_cpu = QuantumState::ground(16);
        let theta = PI * 0.35;

        // Apply via bridge
        let rep = bridge.apply_ry(&mut s_gpu, 2, theta);
        println!("Execution Report: {:?}", rep);

        // Apply via CPU AVX2
        crate::simd::apply_ry_kernel(&mut s_cpu.real, &mut s_cpu.imag, 2, theta);

        // Verify fidelity
        let fid = s_gpu.fidelity(&s_cpu);
        println!("Cross-Backend Numerical Fidelity: {:.14}", fid);
        assert!(
            (fid - 1.0).abs() < 1e-12,
            "Cross-backend fidelity mismatch: {}",
            fid
        );
    }
}
