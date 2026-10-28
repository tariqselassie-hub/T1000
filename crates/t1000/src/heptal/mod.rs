// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Sovereign Radix-7 Heptal Qudit ($\mathbb{Z}_7$) State Spaces
//!
//! Submodule namespace grouping the Radix-7 Matrix Product State QNN and
//! continuous heptal complementarity wave engines.

pub mod qnn;
pub mod wave;

// Backward-compatible module aliases matching original crate layout
pub use qnn as heptal_qnn;
pub use wave as heptal_wave;
