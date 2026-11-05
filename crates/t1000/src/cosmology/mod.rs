// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Native Quantum Universe Simulator & Cosmological Spacetime Emergence
//!
//! Models the microscopic emergence and evolution of a quantum universe:
//! - **Cosmological Metric:** FLRW background, de Sitter inflation, and comoving Hubble horizons.
//! - **Quantum Field Fluctuations:** Discretized spatial scalar fields in the Bunch-Davies vacuum.
//! - **Bogoliubov Particle Creation:** Time-dependent mode mixing creating real matter from curved spacetime.
//! - **Cosmological Entanglement Horizon:** Holographic Area Law entropy growth during super-horizon freeze-out.
//! - **Primordial Power Spectrum:** Scale-invariant Harrison-Zeldovich perturbation distributions.

pub mod bogoliubov;
pub mod field;
pub mod horizon;
pub mod metric;
pub mod simulation;
pub mod structure;

pub use bogoliubov::{compute_mode_bogoliubov, BogoliubovModeReport, CosmicParticleSpectrum};
pub use field::{DiscretizedQuantumField, QuantumFieldMode};
pub use horizon::{
    evaluate_horizon_entanglement, mode_entanglement_entropy, HorizonEntanglementReport,
};
pub use metric::{CosmologicalMetric, ExpansionProfile};
pub use simulation::{
    CosmologicalEpochSnapshot, CosmologicalSimulationReport, CosmologyConfig, CosmologyEngine,
};
pub use structure::{
    CosmicFilament, CosmicStructureSnapshot, CosmicWebNode, MacroStructureType,
    ZeldovichStructureEngine,
};
