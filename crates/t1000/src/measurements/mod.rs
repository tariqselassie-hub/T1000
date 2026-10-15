// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Sovereign Quantum Measurement Architecture
//!
//! Provides projective and generalized quantum measurement operators:
//! - [`observable`]: Pauli operators ($Z$, $X$, bipartite $ZZ$, linear combinations, and global averages).
//! - [`povm_readout`]: Generalized Positive Operator-Valued Measure (POVM) classification and interference readout.

pub mod observable;
pub mod povm_readout;

pub use observable::PauliObservable;
pub use povm_readout::POVMClassifier;
