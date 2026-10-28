// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Topological Trees, Welded Graphs & Quantum Walk Valves (`t1000::trees`)
//!
//! Submodule namespace grouping Continuous-Time Quantum Walks (CTQW) on welded binary trees,
//! multi-exit topological tree valves, cascaded deep quantum routers, and ultrametric QMUX sorters.
//!
//! ## Submodules
//!
//! | Module | Technical Description |
//! | :--- | :--- |
//! | [`glued_trees`] | Childs et al. welded binary trees with symmetric column subspace reduction proving exponential quantum speedup over classical random walks. |
//! | [`glued_tree_valve`] | Multi-exit topological tree valve utilizing Aharonov-Bohm phase holonomy on central welded seams for ballistic routing. |
//! | [`cascaded_tree_valve`] | Cascaded multi-tier welded tree routers achieving sequential steering across dual welds without intermediate wave-function collapse. |
//! | [`ultrametric_tree`] | Hierarchical Quantum Multiplexor (QMUX) sorter ($\mathcal{H} = \mathcal{H}_C \otimes \mathcal{H}_F \otimes \mathcal{H}_E$) enforcing strong triangle ultrametric geometry. |
//! | [`directed_heptal_walk`] | 7-node directed graph Markov chain & CTQW routing engine with exact rational stationary state $\boldsymbol{\pi} = \frac{1}{42}(8, 5, 4, 6, 7, 4, 8)$ and analytical cubic spectrum. |

pub mod cascaded_tree_valve;
pub mod directed_heptal_walk;
pub mod glued_tree_valve;
pub mod glued_trees;
pub mod ultrametric_tree;

// Core public exports
pub use cascaded_tree_valve::{
    simulate_cascaded_classical_walk, simulate_cascaded_quantum_walk, CascadedBenchmarkResult,
    CascadedTrajectoryPoint, CascadedValveConfig, CascadedValveHamiltonian, QuantumWalkStateSoA,
    SparseCoupling, SparseSymmetricHamiltonian,
};
pub use directed_heptal_walk::{
    BipartiteSzegedyWalk, DirectedHeptalQuantumWalk, DirectedHeptalRouter, MolecularDipoleGraph,
};
pub use glued_tree_valve::{
    ClassicalKWayGluedWalker, ContinuousTreeValveWalk, GluedTreeValveConfig,
    MultiExitValveHamiltonian, ValveStepTelemetry,
};
pub use glued_trees::{
    ClassicalGluedWalker, ContinuousQuantumWalk, GluedTreesConfig, QuantumWalkTelemetry,
    SymmetricChainHamiltonian,
};
pub use ultrametric_tree::{
    UltrametricLossReport, UltrametricTreeConfig, UltrametricTreeSorter,
};

// Backward-compatible module aliases matching original crate layout
pub use cascaded_tree_valve as cascaded_tree_valve_mod;
pub use directed_heptal_walk as directed_heptal_walk_mod;
pub use glued_tree_valve as glued_tree_valve_mod;
pub use glued_trees as glued_trees_mod;
pub use ultrametric_tree as ultrametric_tree_mod;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trees_namespace_reexports() {
        let glued_conf = GluedTreesConfig::new(4);
        assert_eq!(glued_conf.depth, 4);
        assert_eq!(glued_conf.num_columns(), 10);

        let valve_conf = GluedTreeValveConfig::new(3, 2);
        assert_eq!(valve_conf.depth, 3);
        assert_eq!(valve_conf.num_exits, 2);

        let cascaded_conf = CascadedValveConfig::new(3, 3, 2, 2);
        assert_eq!(cascaded_conf.depth1, 3);
        assert_eq!(cascaded_conf.depth2, 3);

        let router = DirectedHeptalRouter::new();
        assert_eq!(router.matrix.len(), 7);
    }
}
