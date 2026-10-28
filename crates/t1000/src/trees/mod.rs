// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Topological Trees, Welded Graphs & Quantum Walk Valves
//!
//! Submodule namespace grouping continuous-time quantum walks (CTQW) on welded
//! binary trees, multi-exit quantum valves, cascaded routers, and ultrametric QMUX sorters.

pub mod cascaded_tree_valve;
pub mod directed_heptal_walk;
pub mod glued_tree_valve;
pub mod glued_trees;
pub mod ultrametric_tree;

// Backward-compatible module aliases matching original crate layout
pub use cascaded_tree_valve as cascaded_tree_valve_mod;
pub use directed_heptal_walk as directed_heptal_walk_mod;
pub use glued_tree_valve as glued_tree_valve_mod;
pub use glued_trees as glued_trees_mod;
pub use ultrametric_tree as ultrametric_tree_mod;
