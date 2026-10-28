// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Sovereign Radix-7 Heptal Qudit ($\mathbb{Z}_7$) State Spaces (`t1000::heptal`)
//!
//! Submodule namespace implementing high-dimensional Radix-7 qudit simulation across
//! discrete Matrix Product States (MPS) and continuous complementarity wave dynamics.
//!
//! In standard qubit architectures, the Hilbert space dimension scales as $2^N$. In Radix-7
//! heptal computing, each site spans $\mathcal{H}_7 = \mathbb{C}^7$, offering dense $SU(7)$
//! modular arithmetic, fault-tolerant transversal gates (e.g. $[[7, 1, 3]]$ Steane-Wheel),
//! and high-capacity chemical descriptor embeddings.
//!
//! ## Submodules
//!
//! | Module | Technical Description |
//! | :--- | :--- |
//! | [`qnn`] | Discrete Radix-7 ($\mathbb{Z}_7$) Matrix Product State (MPS) Quantum Neural Network operating in 784-byte L1 cache windows. |
//! | [`wave`] | Continuous heptal qudit zipper wave engine steering mutual complementarity fidelity across multi-site registers. |

pub mod qnn;
pub mod wave;

// Core public exports
pub use qnn::{
    HeptalMolecularZipperQnn, HeptalPrediction, HeptalQnnConfig, HeptalSiteObservables,
};
pub use wave::{
    HeptalWaveState, HeptalZipperConfig, HeptalZipperEngine, HeptalZipperReport,
    HeptalZipperStepTelemetry,
};

// Backward-compatible module aliases matching original crate layout
pub use qnn as heptal_qnn;
pub use wave as heptal_wave;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heptal_namespace_reexports() {
        let conf = HeptalQnnConfig {
            n_qudits: 7,
            entangle_strength: 0.5,
            epoxy_threshold: 0.01,
            use_chrestenson: true,
        };
        assert_eq!(conf.n_qudits, 7);
        assert!(conf.use_chrestenson);

        let zip_conf = HeptalZipperConfig {
            initial_digit: 3,
            epoxy_threshold: 0.01,
        };
        assert_eq!(zip_conf.initial_digit, 3);
    }
}
