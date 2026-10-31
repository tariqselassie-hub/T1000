// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Sovereign Multi-Architecture Quantum Chip Subsystem
//!
//! Provides a hybrid hierarchical architecture registry for quantum hardware modalities
//! and bespoke physical processor architectures:
//!
//! - [`HardwareModality`]: Fundamental physical technology class (Superconducting Transmon,
//!   Photonic Linear Optics, Neutral Atom Rydberg, Trapped Ion, FPGA Silicon Emulation).
//! - [`QuantumChip`]: Concrete processor instances (IBM Nighthawk R2, Rigetti Cepheus-1-108Q,
//!   Belenos Photonic 16-Mode, Emerald Neutral Atom 256, T1000-QPU Artix-7).
//! - [`ChipErrorProfile`]: Physical calibration figures ($T_1$, $T_2$, single- and two-qubit gate fidelities,
//!   readout error rates, inter-chiplet seam penalties).
//! - Native gate sets, coupling topologies, and hardware constraints.

use serde::{Deserialize, Serialize};

/// Fundamental quantum computing physical technology modality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HardwareModality {
    /// Superconducting transmon circuits with planar microwave resonators.
    SuperconductingTransmon,
    /// Linear optical interferometry and photon-number-resolving Fock state processing.
    PhotonicLinearOptics,
    /// Optical tweezer arrays with Rydberg excitation blockades.
    NeutralAtomRydberg,
    /// Linear segmented radio-frequency Paul traps with shuttling.
    TrappedIonShuttled,
    /// Exchange-coupled semiconductor silicon quantum dots.
    SiliconSpinQubit,
    /// High-throughput fixed-point hardware co-processor (FPGA / ASIC).
    FpgaSiliconEmulation,
}

impl HardwareModality {
    /// Returns the human-readable canonical display label.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::SuperconductingTransmon => "Superconducting Transmon",
            Self::PhotonicLinearOptics => "Linear Optical Photonic Mesh",
            Self::NeutralAtomRydberg => "Neutral Atom Rydberg Tweezer Array",
            Self::TrappedIonShuttled => "Trapped-Ion RF Paul Trap",
            Self::SiliconSpinQubit => "Silicon Spin Qubit Dot",
            Self::FpgaSiliconEmulation => "FPGA Fixed-Point Silicon Emulation",
        }
    }

    /// Whether this technology modality supports dynamic all-to-all qubit coupling.
    pub const fn supports_all_to_all(&self) -> bool {
        matches!(self, Self::NeutralAtomRydberg | Self::TrappedIonShuttled)
    }

    /// Whether the register represents continuous bosonic optical modes rather than two-level qubits.
    pub const fn is_continuous_or_photonic(&self) -> bool {
        matches!(self, Self::PhotonicLinearOptics)
    }
}

/// Native basis gate primitives recognized by physical control hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NativeGateKind {
    /// Virtual frame rotation Z (zero error, zero duration).
    Rz,
    /// Microwave X/2 pulse (SX).
    SqrtX,
    /// Microwave pi pulse (X).
    X,
    /// Resonant single-qubit Y-rotation (Ry).
    Ry,
    /// Echoed Cross-Resonance two-qubit gate (ECR).
    Ecr,
    /// Controlled-Z entangling gate (CZ).
    Cz,
    /// Controlled-NOT entangling gate (CX / CNOT).
    Cnot,
    /// Parametric iSWAP or XY interaction gate.
    Xy,
    /// Rydberg global multi-qubit entangling Hamiltonian evolution.
    RydbergBlockade,
    /// Variable beamsplitter / Mach-Zehnder interferometer phase shifter.
    BeamSplitter,
}

/// Comprehensive physical error profile and coherence calibration budget.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChipErrorProfile {
    /// Median single-qubit gate error rate ($1 - F_1$).
    pub single_qubit_error: f64,
    /// Median two-qubit entangling gate error rate ($1 - F_2$).
    pub two_qubit_error: f64,
    /// Median measurement readout error ($1 - F_{\text{meas}}$).
    pub readout_error: f64,
    /// Median energy relaxation time $T_1$ in microseconds ($\mu\text{s}$).
    pub t1_micros: f64,
    /// Median dephasing time $T_2^*$ in microseconds ($\mu\text{s}$);
    pub t2_micros: f64,
    /// Single-qubit physical gate duration in nanoseconds (ns).
    pub single_qubit_duration_ns: f64,
    /// Two-qubit physical gate duration in nanoseconds (ns).
    pub two_qubit_duration_ns: f64,
    /// Additional error factor applied across modular inter-chiplet seams (1.0 = monolithic).
    pub inter_chiplet_seam_factor: f64,
}

impl ChipErrorProfile {
    /// Effective average gate fidelity for two-qubit gates ($F_2$).
    pub fn two_qubit_fidelity(&self) -> f64 {
        1.0 - self.two_qubit_error
    }

    /// Effective two-qubit error when straddling an inter-chiplet seam boundary.
    pub fn seam_two_qubit_error(&self) -> f64 {
        (self.two_qubit_error * self.inter_chiplet_seam_factor).min(1.0)
    }
}

/// Sovereign enumeration of concrete quantum processor architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantumChip {
    /// IBM Quantum Nighthawk R2: 120-qubit monolithic heavy-hex dual Stokes lattice (218 couplers).
    IbmNighthawkR2,
    /// Rigetti Cepheus-1-108Q: 108-qubit modular processor (twelve 3x3 square chiplets).
    RigettiCepheus108Q,
    /// Belenos Photonic: 16-mode triangular/rectangular Mach-Zehnder interferometer mesh.
    BelenosPhotonic16M,
    /// Emerald Neutral Atom: 256-atom optical tweezer lattice with Rydberg blockade.
    EmeraldNeutralAtom256,
    /// T1000-QPU Artix-7: 14-qubit fixed-point FPGA co-processor (128-gate microcode burst).
    T1000QpuArtix7,
}

impl QuantumChip {
    /// All currently cataloged quantum chip targets.
    pub const ALL: &'static [QuantumChip] = &[
        Self::IbmNighthawkR2,
        Self::RigettiCepheus108Q,
        Self::BelenosPhotonic16M,
        Self::EmeraldNeutralAtom256,
        Self::T1000QpuArtix7,
    ];

    /// Returns the underlying physical technology modality.
    pub const fn modality(&self) -> HardwareModality {
        match self {
            Self::IbmNighthawkR2 | Self::RigettiCepheus108Q => HardwareModality::SuperconductingTransmon,
            Self::BelenosPhotonic16M => HardwareModality::PhotonicLinearOptics,
            Self::EmeraldNeutralAtom256 => HardwareModality::NeutralAtomRydberg,
            Self::T1000QpuArtix7 => HardwareModality::FpgaSiliconEmulation,
        }
    }

    /// Full official industry designation.
    pub const fn full_name(&self) -> &'static str {
        match self {
            Self::IbmNighthawkR2 => "IBM Quantum Nighthawk R2 (120Q Monolithic)",
            Self::RigettiCepheus108Q => "Rigetti Cepheus-1-108Q (Modular 12x Chiplets)",
            Self::BelenosPhotonic16M => "Belenos-16M Integrated Linear Photonic Mesh",
            Self::EmeraldNeutralAtom256 => "Emerald-256 Optical Tweezer Rydberg Array",
            Self::T1000QpuArtix7 => "T-1000 QPU Artix-7 Fixed-Point Co-Processor",
        }
    }

    /// Short CLI identifier slug (e.g. `nighthawk-r2`, `cepheus-108q`).
    pub const fn slug(&self) -> &'static str {
        match self {
            Self::IbmNighthawkR2 => "nighthawk-r2",
            Self::RigettiCepheus108Q => "cepheus-108q",
            Self::BelenosPhotonic16M => "belenos-16m",
            Self::EmeraldNeutralAtom256 => "emerald-256",
            Self::T1000QpuArtix7 => "t1000-qpu",
        }
    }

    /// Total number of physical data and ancilla sites (qubits or photonic modes).
    pub const fn capacity(&self) -> usize {
        match self {
            Self::IbmNighthawkR2 => 120,
            Self::RigettiCepheus108Q => 108,
            Self::BelenosPhotonic16M => 16,
            Self::EmeraldNeutralAtom256 => 256,
            Self::T1000QpuArtix7 => 14,
        }
    }

    /// Physical coupling couplers count.
    pub const fn coupler_count(&self) -> usize {
        match self {
            Self::IbmNighthawkR2 => 218,
            Self::RigettiCepheus108Q => 177, // (12 chiplets x 12 intra) + 33 inter-chiplet seams
            Self::BelenosPhotonic16M => 120, // Fully interconnected Clements/Reck interferometer mesh
            Self::EmeraldNeutralAtom256 => 256 * 255 / 2, // Dynamically reconfigurable Rydberg blockades
            Self::T1000QpuArtix7 => 13, // Linear 14-qubit nearest-neighbor systolic bus
        }
    }

    /// Native hardware gate primitives supported without decompilation.
    pub const fn native_gates(&self) -> &'static [NativeGateKind] {
        match self {
            Self::IbmNighthawkR2 => &[
                NativeGateKind::Rz,
                NativeGateKind::SqrtX,
                NativeGateKind::X,
                NativeGateKind::Ecr,
                NativeGateKind::Cz,
            ],
            Self::RigettiCepheus108Q => &[
                NativeGateKind::Rz,
                NativeGateKind::X,
                NativeGateKind::Cz,
                NativeGateKind::Xy,
            ],
            Self::BelenosPhotonic16M => &[
                NativeGateKind::Rz,
                NativeGateKind::BeamSplitter,
            ],
            Self::EmeraldNeutralAtom256 => &[
                NativeGateKind::Rz,
                NativeGateKind::Ry,
                NativeGateKind::Cz,
                NativeGateKind::RydbergBlockade,
            ],
            Self::T1000QpuArtix7 => &[
                NativeGateKind::Ry,
                NativeGateKind::Rz,
                NativeGateKind::Cz,
            ],
        }
    }

    /// Official hardware calibration figures and physical error budget.
    pub fn error_profile(&self) -> ChipErrorProfile {
        match self {
            Self::IbmNighthawkR2 => ChipErrorProfile {
                single_qubit_error: 0.00032,       // 0.032% 1Q error
                two_qubit_error: 0.0062,           // 0.62% 2Q ECR/CZ error
                readout_error: 0.0125,             // 1.25% readout error
                t1_micros: 285.0,                  // 285 us T1
                t2_micros: 210.0,                  // 210 us T2
                single_qubit_duration_ns: 20.0,
                two_qubit_duration_ns: 180.0,
                inter_chiplet_seam_factor: 1.0,    // Monolithic silicon wafer
            },
            Self::RigettiCepheus108Q => ChipErrorProfile {
                single_qubit_error: 0.0010,        // 0.10% 1Q error
                two_qubit_error: 0.0080,           // 0.80% intra-die 2Q error
                readout_error: 0.0250,             // 2.50% readout error
                t1_micros: 34.0,                   // 34 us T1
                t2_micros: 28.0,                   // 28 us T2
                single_qubit_duration_ns: 25.0,
                two_qubit_duration_ns: 90.0,
                inter_chiplet_seam_factor: 2.125,  // Seam error rises to ~1.7% across chiplet boundaries
            },
            Self::BelenosPhotonic16M => ChipErrorProfile {
                single_qubit_error: 0.00005,       // High optical phase stability
                two_qubit_error: 0.0040,           // Beamsplitter insertion loss
                readout_error: 0.0150,             // SNSPD dark counts and efficiency
                t1_micros: 1e9,                    // Photons do not undergo T1 relaxation in flight
                t2_micros: 1e9,
                single_qubit_duration_ns: 0.05,    // Optical propagation delay (speed of light in waveguide)
                two_qubit_duration_ns: 0.20,
                inter_chiplet_seam_factor: 1.0,
            },
            Self::EmeraldNeutralAtom256 => ChipErrorProfile {
                single_qubit_error: 0.0006,        // Raman beam fidelity
                two_qubit_error: 0.0045,           // Rydberg blockade fidelity
                readout_error: 0.0080,             // Fluorescence camera imaging
                t1_micros: 2_500_000.0,            // 2.5 second hyperfine state lifetime
                t2_micros: 850_000.0,              // 0.85 second dephasing lifetime
                single_qubit_duration_ns: 1200.0,  // Laser pulse width
                two_qubit_duration_ns: 350.0,      // Fast Rydberg excitation
                inter_chiplet_seam_factor: 1.0,    // Homogeneous atomic gas
            },
            Self::T1000QpuArtix7 => ChipErrorProfile {
                single_qubit_error: 1e-7,          // Fixed-point algorithmic roundoff only
                two_qubit_error: 2e-7,
                readout_error: 0.0,                // Exact digital readout
                t1_micros: 1e12,                   // Digital SRAM retention
                t2_micros: 1e12,
                single_qubit_duration_ns: 10.0,    // 100 MHz clock cycle (1 cycle)
                two_qubit_duration_ns: 30.0,       // 3 clock cycles per systolic step
                inter_chiplet_seam_factor: 1.0,
            },
        }
    }

    /// Parses a chip identifier slug case-insensitively.
    pub fn from_slug(slug: &str) -> Option<Self> {
        let norm = slug.trim().to_ascii_lowercase().replace('_', "-");
        match norm.as_str() {
            "nighthawk-r2" | "nighthawk" | "ibm-nighthawk" => Some(Self::IbmNighthawkR2),
            "cepheus-108q" | "cepheus" | "rigetti-cepheus" => Some(Self::RigettiCepheus108Q),
            "belenos-16m" | "belenos" | "photonic-16m" => Some(Self::BelenosPhotonic16M),
            "emerald-256" | "emerald" | "rydberg-256" => Some(Self::EmeraldNeutralAtom256),
            "t1000-qpu" | "artix7" | "fpga-qpu" => Some(Self::T1000QpuArtix7),
            _ => None,
        }
    }

    /// Constructs the routing [`crate::routing::Topology`] graph for this chip architecture.
    pub fn build_routing_topology(&self) -> crate::routing::Topology {
        match self {
            Self::IbmNighthawkR2 => {
                let lattice = crate::plaquette::PlaquetteLattice::build_ibm_nighthawk_r2();
                let edges: Vec<(usize, usize)> = lattice
                    .links
                    .iter()
                    .map(|l| (l.u, l.v))
                    .collect();
                crate::routing::Topology::new(self.slug(), self.capacity(), &edges)
            }
            Self::RigettiCepheus108Q => {
                let mut edges = Vec::new();
                // 12x9 grid (108 qubits)
                for r in 0..12 {
                    for c in 0..9 {
                        let u = r * 9 + c;
                        if c + 1 < 9 {
                            edges.push((u, u + 1));
                        }
                        if r + 1 < 12 {
                            edges.push((u, u + 9));
                        }
                    }
                }
                crate::routing::Topology::new(self.slug(), self.capacity(), &edges)
            }
            Self::BelenosPhotonic16M => crate::routing::Topology::all_to_all(16),
            Self::EmeraldNeutralAtom256 => {
                // Nearest-neighbor 16x16 square grid proxy for optical tweezer lattice
                crate::routing::Topology::grid(16, 16)
            }
            Self::T1000QpuArtix7 => crate::routing::Topology::line(14),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chip_catalog_integrity() {
        for chip in QuantumChip::ALL {
            assert!(chip.capacity() > 0);
            assert!(chip.coupler_count() > 0);
            assert!(!chip.native_gates().is_empty());

            let err = chip.error_profile();
            assert!(err.single_qubit_error >= 0.0);
            assert!(err.two_qubit_error >= 0.0);
            assert!(err.two_qubit_fidelity() <= 1.0);
            assert!(err.inter_chiplet_seam_factor >= 1.0);

            let slug = chip.slug();
            assert_eq!(QuantumChip::from_slug(slug), Some(*chip));
        }
    }

    #[test]
    fn test_modalities_classification() {
        assert_eq!(
            QuantumChip::IbmNighthawkR2.modality(),
            HardwareModality::SuperconductingTransmon
        );
        assert_eq!(
            QuantumChip::RigettiCepheus108Q.modality(),
            HardwareModality::SuperconductingTransmon
        );
        assert_eq!(
            QuantumChip::BelenosPhotonic16M.modality(),
            HardwareModality::PhotonicLinearOptics
        );
        assert_eq!(
            QuantumChip::EmeraldNeutralAtom256.modality(),
            HardwareModality::NeutralAtomRydberg
        );
        assert_eq!(
            QuantumChip::T1000QpuArtix7.modality(),
            HardwareModality::FpgaSiliconEmulation
        );

        assert!(HardwareModality::NeutralAtomRydberg.supports_all_to_all());
        assert!(!HardwareModality::SuperconductingTransmon.supports_all_to_all());
        assert!(HardwareModality::PhotonicLinearOptics.is_continuous_or_photonic());
    }

    #[test]
    fn test_topology_generation() {
        let nighthawk_topo = QuantumChip::IbmNighthawkR2.build_routing_topology();
        assert_eq!(nighthawk_topo.n, 120);
        assert_eq!(nighthawk_topo.edges().len(), 218);

        let artix_topo = QuantumChip::T1000QpuArtix7.build_routing_topology();
        assert_eq!(artix_topo.n, 14);
        assert_eq!(artix_topo.edges().len(), 13);
    }
}
