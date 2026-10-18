// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! T-1000 Quantum Conway Game of Life Engine.
//! Sovereign, ultra-high-performance Unitary Quantum Cellular Automata (QCA)
//! driven by Fractional Fermi Seas, interaction holonomy cycles across
//! Confinement-Induced Resonances (CIR), and Haldane Generalized Exclusion Statistics.
//!
//! This crate is fully isolated from the main `t1000` library to keep Conway-specific
//! grid rules, Margolus partition operators, and soliton trackers separate from
//! the VQC and chemical screening engines.

pub mod holonomy;
pub mod lattice;
pub mod patterns;
pub mod rules;
pub mod simulator;
pub mod state;
pub mod telemetry;

pub use holonomy::{
    build_holonomy_unitary, ffs_momentum_plateau, geometric_holonomy_phase, HolonomyParams,
    HolonomyTransferMatrix,
};
pub use lattice::{BoundaryCondition, Lattice2D};
pub use patterns::{PatternFactory, PatternType};
pub use rules::{
    compute_hamming_distance, compute_neighborhood_occupations, ClassicalConwayEngine,
};
pub use simulator::{QuantumConwaySimulator, StepTelemetry};
pub use state::{CellBasis, QuantumCellStateSoA};
pub use telemetry::{colors, TelemetryVisualizer};
