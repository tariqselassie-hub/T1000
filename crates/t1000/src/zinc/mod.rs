// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # ZINC Molecular Screening & Variational Training Suite
//!
//! Submodule namespace grouping the ZINC dataset loaders, holdout evaluations,
//! kernel benchmarks, readout fits, and training pipelines.

pub mod data;
pub mod holdout_eval;
pub mod kernel_bench;
pub mod readout_fit;
pub mod reuploading_model;
pub mod stage_common;
pub mod training_common;

// Backward-compatible module aliases matching original crate layout
pub use data as zinc_data;
pub use holdout_eval as zinc_holdout_eval;
pub use kernel_bench as zinc_kernel_bench;
pub use readout_fit as zinc_readout_fit;
pub use reuploading_model as zinc_reuploading_model;
pub use stage_common as zinc_stage_common;
pub use training_common as zinc_training_common;
