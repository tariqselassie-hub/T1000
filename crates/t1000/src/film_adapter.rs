// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

use crate::backbone::{self, Quat};

#[derive(Clone, Debug)]
pub struct TaskFiLMAdapter {
    pub gamma_zinc: Quat,
    pub beta_zinc: Quat,
    pub gamma_qm9: Quat,
    pub beta_qm9: Quat,
}

#[derive(Clone, Debug)]
pub struct FiLMGradients {
    pub d_gamma_zinc: Quat,
    pub d_beta_zinc: Quat,
    pub d_gamma_qm9: Quat,
    pub d_beta_qm9: Quat,
}

impl FiLMGradients {
    pub fn zeros() -> Self {
        Self {
            d_gamma_zinc: [0.0; 4],
            d_beta_zinc: [0.0; 4],
            d_gamma_qm9: [0.0; 4],
            d_beta_qm9: [0.0; 4],
        }
    }
}

impl TaskFiLMAdapter {
    pub fn new() -> Self {
        Self {
            gamma_zinc: [1.0, 1.0, 1.0, 1.0],
            beta_zinc: [0.0, 0.0, 0.0, 0.0],
            gamma_qm9: [1.0, 1.0, 1.0, 1.0],
            beta_qm9: [0.0, 0.0, 0.0, 0.0],
        }
    }

    pub fn forward_node(&self, q: Quat, source: crate::multitask_backbone::DatasetSource) -> (Quat, f64) {
        let (gamma, beta) = match source {
            crate::multitask_backbone::DatasetSource::Zinc => (&self.gamma_zinc, &self.beta_zinc),
            crate::multitask_backbone::DatasetSource::QM9 => (&self.gamma_qm9, &self.beta_qm9),
        };
        let z = [
            gamma[0] * q[0] + beta[0],
            gamma[1] * q[1] + beta[1],
            gamma[2] * q[2] + beta[2],
            gamma[3] * q[3] + beta[3],
        ];
        let norm_z = backbone::quat_norm(z);
        let q_mod = [
            z[0] / norm_z,
            z[1] / norm_z,
            z[2] / norm_z,
            z[3] / norm_z,
        ];
        (q_mod, norm_z)
    }

    pub fn backward_node(
        &self,
        q_in: Quat,
        norm_z: f64,
        d_q_mod: Quat,
        source: crate::multitask_backbone::DatasetSource,
        grads: &mut FiLMGradients,
    ) -> Quat {
        let (gamma, _) = match source {
            crate::multitask_backbone::DatasetSource::Zinc => (&self.gamma_zinc, &self.beta_zinc),
            crate::multitask_backbone::DatasetSource::QM9 => (&self.gamma_qm9, &self.beta_qm9),
        };
        let d_z = backbone::unit_lift_backward(q_in, norm_z, d_q_mod);
        let (d_gamma, d_beta) = match source {
            crate::multitask_backbone::DatasetSource::Zinc => (&mut grads.d_gamma_zinc, &mut grads.d_beta_zinc),
            crate::multitask_backbone::DatasetSource::QM9 => (&mut grads.d_gamma_qm9, &mut grads.d_beta_qm9),
        };
        for c in 0..4 {
            d_gamma[c] += d_z[c] * q_in[c];
            d_beta[c] += d_z[c];
        }
        [
            d_z[0] * gamma[0],
            d_z[1] * gamma[1],
            d_z[2] * gamma[2],
            d_z[3] * gamma[3],
        ]
    }
}
