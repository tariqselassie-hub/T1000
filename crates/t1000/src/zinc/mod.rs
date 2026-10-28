// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # ZINC Molecular Screening & Variational Training Suite (`t1000::zinc`)
//!
//! Submodule namespace grouping the high-throughput ZINC 250k dataset loaders, holdout scoring,
//! kernel benchmarks, linear readout fitting, and variational quantum neural network (VQC) models.
//!
//! ## Submodules
//!
//! | Module | Technical Description |
//! | :--- | :--- |
//! | [`data`] | CSV loader, parsing, normalization scalers, and orthogonal wire allocations for ZINC molecules. |
//! | [`holdout_eval`] | Holdout validation and scoring evaluating mean squared error (MSE) and $R^2$ across logP, QED, and SAS. |
//! | [`kernel_bench`] | Kernel ridge regression benchmarks comparing state fidelity kernels $k(x_i, x_j) = |\langle \psi(x_i)|\psi(x_j)\rangle|^2$. |
//! | [`readout_fit`] | Closed-form Cholesky ridge regression solvers mapping quantum expectation registers to continuous chemical properties. |
//! | [`reuploading_model`] | Continuous dual-axis $SU(2)$ data re-uploading quantum neural networks with quantum natural gradient (QNG) optimizers. |
//! | [`stage_common`] | Shared dataset caching, batching, parameter clamping, and loss evaluation across training stages. |
//! | [`training_common`] | Model checkpoint serialization/deserialization, contract verification, and metric reporting. |

pub mod data;
pub mod holdout_eval;
pub mod kernel_bench;
pub mod readout_fit;
pub mod reuploading_model;
pub mod stage_common;
pub mod training_common;

// Core public exports
pub use data::{InputScaling, Scaler, ZincMolecule};
pub use kernel_bench::{Bench as KernelBench, Kernel as KernelType};
pub use reuploading_model::{
    AdamPath as ReuploadingAdamPath, Model as ReuploadingModel, QngCheckpoint, Scores as ReuploadingScores,
};
pub use stage_common::{Adam as ZincAdam, Data as ZincStageData, Fit as ZincStageFit};
pub use training_common::{
    MidCircuitCheckpoint, PrunedCheckpoint, PropertyMetric, QnnCheckpoint, ReuploadingCheckpoint,
};

// Backward-compatible module aliases matching original crate layout
pub use data as zinc_data;
pub use holdout_eval as zinc_holdout_eval;
pub use kernel_bench as zinc_kernel_bench;
pub use readout_fit as zinc_readout_fit;
pub use reuploading_model as zinc_reuploading_model;
pub use stage_common as zinc_stage_common;
pub use training_common as zinc_training_common;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zinc_namespace_reexports() {
        let mol = ZincMolecule {
            smiles: "CC(=O)O".to_string(),
            features: vec![0.5; 19],
            log_p: 0.1,
            qed: 0.5,
            sas: 2.0,
        };
        assert_eq!(mol.smiles, "CC(=O)O");
        assert_eq!(mol.features.len(), 19);

        let metric = PropertyMetric {
            initial_mse: 1.0,
            tuned_mse: 0.2,
            tuned_mae: 0.3,
            pearson_r: 0.85,
            error_reduction_pct: 80.0,
        };
        assert_eq!(metric.error_reduction_pct, 80.0);
    }
}
