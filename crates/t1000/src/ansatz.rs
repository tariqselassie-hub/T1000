// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Variational ansatz library.
//!
//! Sub-modules:
//! - [`core`]      — [`CircuitAnsatz`]: hardware-efficient VQC with fused forward paths,
//!                   mid-circuit feed-forward, data re-uploading, and DAG compilation.
//! - [`layers`]    — [`VariationalQuantumLayer`], [`DeepQuantumNeuralNetwork`],
//!                   [`PiPhaseLayer`] / [`RademacherPhaseLayer`], entangling-pair generator.
//! - [`pre_init`]  — [`PreInitCircuit`] enum and the four gate families it selects from:
//!                   Walsh-Rademacher, GHZ, and QFT/IQFT.
//! - [`cut_bridge`], [`dispatch`], [`reupload`], [`virtual_cut_engine`] — specialised circuits.

pub mod core;
pub mod cut_bridge;
pub mod dispatch;
pub mod layers;
pub mod multi_cut_plaquette;
pub mod pre_init;
pub mod reupload;
pub mod virtual_cut_engine;

// Re-export the primary public surface so call-sites are unchanged.
pub use core::CircuitAnsatz;
pub use cut_bridge::{
    eval_36q_bipartite_bridge, eval_36q_bipartite_midcircuit, eval_cut_bridge_general,
    CutBridgeReport,
};
pub use dispatch::*;
pub use layers::{
    entangling_pairs, DeepQuantumNeuralNetwork, PiPhaseLayer, RademacherPhaseLayer,
    VariationalQuantumLayer,
};
pub use multi_cut_plaquette::{
    execute_multi_cut_plaquette_contraction, generate_joint_cz_branches,
    truncate_marginal_tensor_svd, BoundaryEdge, JointBranchAssignment, MultiCutReport,
    PlaquetteBoundaryTopology, QuasiprobabilityOptimizer, SvdTruncationReport,
    MULTI_CUT_DEFAULT_CLUSTER_QUBITS,
};
pub use pre_init::{apply_alt_bell_ghz, apply_bell_ghz, apply_iqft, apply_qft, PreInitCircuit};
pub use reupload::DualAxisEncoder;
pub use virtual_cut_engine::{
    FragmentSnapshot, Virtual36QCutEngine, Virtual36QReport, VirtualCutConfig, DENSE_36Q_RAM_BYTES,
    FRAGMENT_DIM, FRAGMENT_QUBITS, VIRTUAL_CUT_WORKING_SET_BYTES,
};
