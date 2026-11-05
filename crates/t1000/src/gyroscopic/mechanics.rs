// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Poinsot Gyroscopic Mechanics & Dynamic Torque Cancellation Engine
//!
//! Evaluates precessional trajectories, angular momentum conservation,
//! and instantaneous torque cancellation at the central Node (4, 5) fulcrum.

use super::topology::{JackNode, JackTopology, Vector3D};
use serde::{Deserialize, Serialize};

/// Telemetry metrics for an instantaneous precessional step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MechanicsStep {
    pub time: f64,
    pub precess_angle_a_rad: f64,
    pub precess_angle_b_rad: f64,
    pub torque_jet_a: Vector3D,
    pub torque_jet_b: Vector3D,
    pub torque_net: Vector3D,
    pub torque_cancellation_ratio: f64,
    pub angular_momentum_norm: f64,
    pub center_of_mass_drift: f64,
}

/// Dynamic gyroscopic mechanics engine.
#[derive(Debug, Clone)]
pub struct GyroMechanicsEngine {
    pub precession_freq_rad_s: f64,
    pub noise_freq_rad_s: f64,
    pub noise_amplitude_n: f64,
}

impl GyroMechanicsEngine {
    /// Creates a new gyroscopic mechanics engine with specified frequencies and noise amplitude.
    pub fn new(precession_freq: f64, noise_freq: f64, noise_amp: f64) -> Self {
        Self {
            precession_freq_rad_s: precession_freq,
            noise_freq_rad_s: noise_freq,
            noise_amplitude_n: noise_amp,
        }
    }

    /// Evaluates the rotation matrix R_z(angle) applied to a 3D vector.
    #[inline]
    pub fn rotate_z(v: &Vector3D, angle: f64) -> Vector3D {
        let (sin_a, cos_a) = angle.sin_cos();
        Vector3D::new(v.x * cos_a - v.y * sin_a, v.x * sin_a + v.y * cos_a, v.z)
    }

    /// Advances the topology to time `t`, rotating the polar jets and evaluating torques.
    pub fn step(&self, topology: &mut JackTopology, t: f64) -> MechanicsStep {
        let theta = self.precession_freq_rad_s * t;

        // Perturbation noise forces at polar extremes
        let f_mag = self.noise_amplitude_n;
        let noise_phase = self.noise_freq_rad_s * t;
        let force_a = Vector3D::new(f_mag * noise_phase.cos(), f_mag * noise_phase.sin(), 0.0);
        // Equal force on antipodal arm produces equal and opposite torque: tau_B = (-r_A) x F_A = -tau_A
        let force_b = force_a;

        let mut tau_a = Vector3D::zero();
        let mut tau_b = Vector3D::zero();

        for node in &mut topology.nodes {
            match node.id {
                1 | 2 | 3 => {
                    node.current_pos = Self::rotate_z(&node.initial_pos, theta);
                    let tau_i = node.current_pos.cross(&force_a);
                    tau_a = tau_a.add(&tau_i.scale(1.0 / 3.0));
                }
                4 | 5 => {
                    // Shared fulcrum remains static at inertial center
                    node.current_pos = node.initial_pos;
                }
                6 | 7 | 8 => {
                    node.current_pos = Self::rotate_z(&node.initial_pos, theta);
                    let tau_j = node.current_pos.cross(&force_b);
                    tau_b = tau_b.add(&tau_j.scale(1.0 / 3.0));
                }
                _ => {}
            }
        }

        topology.center_of_mass = JackTopology::compute_center_of_mass(&topology.nodes);
        let com_drift = topology.center_of_mass.norm();

        let tau_net = tau_a.add(&tau_b);
        let tau_a_mag = tau_a.norm();
        let tau_net_mag = tau_net.norm();

        let cancellation_ratio = if tau_a_mag > 1e-15 {
            1.0 - (tau_net_mag / tau_a_mag).min(1.0)
        } else {
            1.0
        };

        let l_norm =
            self.compute_angular_momentum_norm(&topology.nodes, self.precession_freq_rad_s);

        MechanicsStep {
            time: t,
            precess_angle_a_rad: theta,
            precess_angle_b_rad: theta + std::f64::consts::PI,
            torque_jet_a: tau_a,
            torque_jet_b: tau_b,
            torque_net: tau_net,
            torque_cancellation_ratio: cancellation_ratio,
            angular_momentum_norm: l_norm,
            center_of_mass_drift: com_drift,
        }
    }

    /// Evaluates total angular momentum ||L|| = sum m_i (r_i x v_i).
    pub fn compute_angular_momentum_norm(&self, nodes: &[JackNode], omega: f64) -> f64 {
        let mut total_l = Vector3D::zero();
        for n in nodes {
            if !n.is_shared_core {
                let v = Vector3D::new(-omega * n.current_pos.y, omega * n.current_pos.x, 0.0);
                let li = n.current_pos.cross(&v).scale(n.mass);
                total_l = total_l.add(&li);
            }
        }
        total_l.norm()
    }
}
