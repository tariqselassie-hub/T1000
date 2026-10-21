// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Runtime host identification for benchmark telemetry and backend tuning.
//!
//! Two machines are first-class ([`KnownHost`]); any other machine is read from the OS and
//! reported as [`KnownHost::Other`] with its real CPU model, core count, RAM and ISA.
//!
//! Override for testing: `T1000_HOST_CPU` replaces the detected CPU model string.

use std::fs;

/// Machines with measured, committed reference data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnownHost {
    /// Desktop reference machine: Intel Core Ultra 7 265.
    CoreUltra7_265,
    /// Laptop with UHD iGPU: Intel Core i3-1115G4.
    I3_1115G4,
    /// Anything else; values come straight from the OS.
    Other,
}

/// Best SIMD tier the running CPU supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsaTier {
    Avx512,
    Avx2Fma,
    Scalar,
}

impl IsaTier {
    pub fn label(self) -> &'static str {
        match self {
            Self::Avx512 => "512-bit AVX-512F kernels (AVX2+FMA elsewhere)",
            Self::Avx2Fma => "256-bit AVX2+FMA SIMD",
            Self::Scalar => "scalar",
        }
    }
}

/// Detected description of the machine this process is running on.
#[derive(Debug, Clone, PartialEq)]
pub struct HostProfile {
    pub cpu_model: String,
    pub logical_cores: usize,
    pub ram_gb: f64,
    pub isa: IsaTier,
    pub known: KnownHost,
}

impl HostProfile {
    /// Read the running machine. Never fails; unknown fields degrade to safe values.
    pub fn detect() -> Self {
        let cpu_model = std::env::var("T1000_HOST_CPU")
            .ok()
            .or_else(read_cpu_model)
            .unwrap_or_else(|| "Unknown CPU".to_string());
        let logical_cores = std::thread::available_parallelism().map_or(1, usize::from);
        let known = classify(&cpu_model);
        Self {
            cpu_model,
            logical_cores,
            ram_gb: read_ram_gb(),
            isa: detect_isa(),
            known,
        }
    }

    /// One-line label, e.g. `Intel Core i3-1115G4 (4 threads, 256-bit AVX2+FMA SIMD)`.
    pub fn label(&self) -> String {
        format!(
            "{} ({} threads, {})",
            self.cpu_model,
            self.logical_cores,
            self.isa.label()
        )
    }

    /// Filesystem-safe identifier used to name per-host result files.
    pub fn slug(&self) -> String {
        match self.known {
            KnownHost::CoreUltra7_265 => "core-ultra-7-265".to_string(),
            KnownHost::I3_1115G4 => "i3-1115g4".to_string(),
            KnownHost::Other => slugify(&self.cpu_model),
        }
    }

    /// True on the machine whose results are the committed canonical reference files.
    pub fn is_reference_host(&self) -> bool {
        self.known == KnownHost::CoreUltra7_265
    }

    /// Measured iGPU crossover (qubits) for this host, if one exists.
    ///
    /// `Some(usize::MAX)` means the cache-blocked FP64 CPU executor beat the FP32 iGPU at every
    /// measured size (12-26 qubits), so `Fast` precision never routes to the GPU on its own.
    pub fn gpu_min_qubits(&self) -> Option<usize> {
        match self.known {
            KnownHost::I3_1115G4 => Some(usize::MAX),
            _ => None,
        }
    }
}

impl HostProfile {
    /// Largest register (qubits) worth multithreading on this host, if measured.
    ///
    /// `None` means unmeasured: keep threading at every size.
    pub fn parallel_max_qubits(&self) -> Option<usize> {
        match self.known {
            KnownHost::I3_1115G4 => Some(PARALLEL_MAX_QUBITS_I3),
            _ => None,
        }
    }
}

/// i3-1115G4: parallel Ry beats serial at 18 qubits (4 MB) and loses from 20 (16 MB).
const PARALLEL_MAX_QUBITS_I3: usize = 18;

fn classify(model: &str) -> KnownHost {
    let m = model.to_ascii_lowercase();
    if m.contains("ultra 7 265") {
        KnownHost::CoreUltra7_265
    } else if m.contains("i3-1115g4") {
        KnownHost::I3_1115G4
    } else {
        KnownHost::Other
    }
}

fn slugify(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    out.trim_matches('-').to_string()
}

fn read_cpu_model() -> Option<String> {
    let info = fs::read_to_string("/proc/cpuinfo").ok()?;
    let line = info.lines().find(|l| l.starts_with("model name"))?;
    let raw = line.split_once(':')?.1.trim();
    // Drop the marketing clock suffix and trademark noise: "Intel(R) Core(TM) i3-... @ 3.00GHz".
    let cleaned = raw
        .split('@')
        .next()
        .unwrap_or(raw)
        .replace("(R)", "")
        .replace("(TM)", "");
    let cleaned = cleaned.replace("11th Gen ", "");
    Some(cleaned.split_whitespace().collect::<Vec<_>>().join(" "))
}

fn read_ram_gb() -> f64 {
    let kb = fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|m| {
            m.lines()
                .find(|l| l.starts_with("MemTotal:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())
        })
        .unwrap_or(0.0);
    (kb / 1024.0 / 1024.0 * 10.0).round() / 10.0
}

#[cfg(target_arch = "x86_64")]
fn detect_isa() -> IsaTier {
    // `T1000_SIMD=avx2` turns the 512-bit kernels off, so the label follows what actually runs.
    if crate::simd::avx512::enabled() {
        IsaTier::Avx512
    } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        IsaTier::Avx2Fma
    } else {
        IsaTier::Scalar
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn detect_isa() -> IsaTier {
    IsaTier::Scalar
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_known_hosts() {
        assert_eq!(
            classify("Intel Core Ultra 7 265"),
            KnownHost::CoreUltra7_265
        );
        assert_eq!(classify("Intel Core i3-1115G4"), KnownHost::I3_1115G4);
        assert_eq!(classify("AMD Ryzen 9 7950X"), KnownHost::Other);
    }

    #[test]
    fn slug_is_filesystem_safe() {
        assert_eq!(
            slugify("AMD Ryzen 9 7950X 16-Core Processor"),
            "amd-ryzen-9-7950x-16-core-processor"
        );
    }

    #[test]
    fn detect_reads_something_real() {
        let h = HostProfile::detect();
        assert!(h.logical_cores >= 1);
        assert!(!h.slug().is_empty());
    }
}
