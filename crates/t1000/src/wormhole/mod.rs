// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Native Traversable Holographic Wormhole Protocol (ER = EPR)
//!
//! Simulates traversable gravitational Einstein-Rosen bridges emergent from quantum entanglement:
//! - **Thermofield Double (TFD) Ground State:** Maximally entangled paired black hole horizons.
//! - **Fast Chaotic Scramblers:** All-to-all Heisenberg spin glass, Jordan-Wigner SYK, and Floquet Haar circuits.
//! - **Negative Average Null Energy (ANEC Violation):** Bilateral double-trace coupling U(g) opening the throat.
//! - **Teleportation by Size:** Coherent wavepacket refocusing driven by operator size winding.
//! - **Gravitational Sign Asymmetry:** \Delta F(g) = F(+g) - F(-g) isolating the physical wormhole signal.

pub mod metrics;
pub mod scrambler;
pub mod teleportation;

pub use metrics::{
    bell_fidelity, compute_operator_size_distribution, reduced_density_matrix_2q,
    single_qubit_reduced_density_matrix, state_fidelity_1q, WormholeFidelityPoint,
    WormholeSweepReport,
};
pub use scrambler::{ScramblerType, ScramblingHamiltonian};
pub use teleportation::{HolographicWormholeEngine, WormholeConfig, WormholeWireMap};

