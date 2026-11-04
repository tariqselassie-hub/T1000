// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Macroscopic Cosmic Structure & Zel'dovich Web Formation
//!
//! Bridges microscopic quantum fluctuations to the macroscopic universe:
//! - **Zel'dovich Approximation:** Gravitational caustic collapse x = a(t) * (q - D(t) * grad(Phi)).
//! - **Spherical Collapse Criterion:** Halos and galaxy clusters collapse when delta > delta_c = 1.686.
//! - **Cosmic Web Filaments:** 1D and 2D tidal bridges connecting high-density halos across voids.
//! - **Causal Horizon Tracking:** Horizon exit and cosmological redshift z = a^(-1) - 1.

use serde::{Deserialize, Serialize};

/// Classification of cosmic web structures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MacroStructureType {
    Void,
    Wall,
    Filament,
    ClusterNode,
}

/// A macroscopic galaxy cluster, halo, or web tracer node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CosmicWebNode {
    pub id: usize,
    pub lagrangian_q: [f64; 3],
    pub eulerian_x: [f64; 3],
    pub density_contrast: f64,
    pub structure_type: MacroStructureType,
    pub is_inside_horizon: bool,
    pub redshift: f64,
}

/// A cosmic filament bridging two gravitational nodes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CosmicFilament {
    pub from_node: usize,
    pub to_node: usize,
    pub length: f64,
    pub linear_density: f64,
}

/// Snapshot of the macroscopic universe at cosmic time t.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CosmicStructureSnapshot {
    pub cosmic_time: f64,
    pub scale_factor: f64,
    pub growth_factor: f64,
    pub hubble_horizon_radius: f64,
    pub void_fraction: f64,
    pub cluster_count: usize,
    pub filament_count: usize,
    pub nodes: Vec<CosmicWebNode>,
    pub filaments: Vec<CosmicFilament>,
}

/// Zel'dovich cosmological structure engine.
#[derive(Debug, Clone)]
pub struct ZeldovichStructureEngine {
    pub seeds: Vec<[f64; 3]>,
    pub wavevectors: Vec<[f64; 3]>,
    pub amplitudes: Vec<f64>,
    pub phases: Vec<f64>,
    pub box_scale: f64,
}

impl ZeldovichStructureEngine {
    /// Constructs the engine by seeding Lagrangian points and perturbation modes.
    pub fn new(num_seeds: usize, box_scale: f64, power_levels: &[f64]) -> Self {
        let mut seeds = Vec::with_capacity(num_seeds);
        // Distribute quasi-uniformly in a sphere of radius box_scale
        let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
        for i in 0..num_seeds {
            let theta = 2.0 * std::f64::consts::PI * (i as f64) / golden_ratio;
            let phi = ((2.0 * (i as f64) + 1.0) / (num_seeds as f64) - 1.0).acos();
            let r = box_scale * ((i as f64 + 1.0) / (num_seeds as f64)).powf(1.0 / 3.0);

            let qx = r * phi.sin() * theta.cos();
            let qy = r * phi.sin() * theta.sin();
            let qz = r * phi.cos();
            seeds.push([qx, qy, qz]);
        }

        // Generate perturbation wavevectors from power levels
        let n_modes = power_levels.len().max(1);
        let mut wavevectors = Vec::with_capacity(n_modes * 3);
        let mut amplitudes = Vec::with_capacity(n_modes * 3);
        let mut phases = Vec::with_capacity(n_modes * 3);

        for (m, &p_val) in power_levels.iter().enumerate() {
            let km = 2.0 * std::f64::consts::PI * ((m + 1) as f64) / box_scale;
            let amp = (p_val.max(1e-6)).sqrt() * 0.15;

            // 3 orthogonal axes
            wavevectors.push([km, 0.0, 0.0]);
            amplitudes.push(amp);
            phases.push((m as f64 * 1.37).sin());

            wavevectors.push([0.0, km, 0.0]);
            amplitudes.push(amp);
            phases.push((m as f64 * 2.19).cos());

            wavevectors.push([0.0, 0.0, km]);
            amplitudes.push(amp);
            phases.push((m as f64 * 3.71).sin());
        }

        Self {
            seeds,
            wavevectors,
            amplitudes,
            phases,
            box_scale,
        }
    }

    /// Evaluates the gravitational displacement field at Lagrangian position q.
    #[inline(always)]
    pub fn evaluate_displacement(&self, q: &[f64; 3]) -> [f64; 3] {
        let mut psi = [0.0; 3];
        for i in 0..self.wavevectors.len() {
            let k = &self.wavevectors[i];
            let dot = k[0] * q[0] + k[1] * q[1] + k[2] * q[2];
            let s = (dot + self.phases[i]).sin();
            let a = self.amplitudes[i];
            psi[0] += a * k[0] * s;
            psi[1] += a * k[1] * s;
            psi[2] += a * k[2] * s;
        }
        psi
    }

    /// Linear growth factor D(t) parameterized by scale factor a(t).
    #[inline(always)]
    pub fn linear_growth_factor(a: f64) -> f64 {
        // Growth factor grows with scale factor expansion
        (a - 1.0).max(0.0) / 10.0 + 0.05
    }

    /// Generates a macroscopic structure snapshot at cosmic time t.
    pub fn snapshot(
        &self,
        cosmic_time: f64,
        scale_factor: f64,
        hubble_rate: f64,
    ) -> CosmicStructureSnapshot {
        let growth = Self::linear_growth_factor(scale_factor);
        let horizon_r = if hubble_rate > 1e-6 {
            scale_factor / hubble_rate
        } else {
            self.box_scale * scale_factor * 10.0
        };
        let redshift = if scale_factor > 1e-4 {
            1.0 / scale_factor - 1.0
        } else {
            0.0
        };

        let mut nodes = Vec::with_capacity(self.seeds.len());
        let mut void_count = 0;
        let mut cluster_count = 0;

        for (id, q) in self.seeds.iter().enumerate() {
            let psi = self.evaluate_displacement(q);
            // Zel'dovich displacement: x = a * (q - D * Psi)
            let ex = scale_factor * (q[0] - growth * psi[0]);
            let ey = scale_factor * (q[1] - growth * psi[1]);
            let ez = scale_factor * (q[2] - growth * psi[2]);

            let dist_sq = ex * ex + ey * ey + ez * ez;
            let dist = dist_sq.sqrt();
            let is_inside = dist <= horizon_r;

            // Divergence of displacement ~ density contrast delta
            let psi_mag = (psi[0] * psi[0] + psi[1] * psi[1] + psi[2] * psi[2]).sqrt();
            let delta = growth * psi_mag * 5.0 - 0.2;

            let stype = if delta > 1.686 {
                cluster_count += 1;
                MacroStructureType::ClusterNode
            } else if delta > 0.4 {
                MacroStructureType::Filament
            } else if delta > -0.3 {
                MacroStructureType::Wall
            } else {
                void_count += 1;
                MacroStructureType::Void
            };

            nodes.push(CosmicWebNode {
                id,
                lagrangian_q: *q,
                eulerian_x: [ex, ey, ez],
                density_contrast: delta,
                structure_type: stype,
                is_inside_horizon: is_inside,
                redshift,
            });
        }

        let filaments = self.compute_filaments(&nodes, scale_factor);
        let filament_count = filaments.len();
        let void_fraction = (void_count as f64) / (nodes.len().max(1) as f64);

        CosmicStructureSnapshot {
            cosmic_time,
            scale_factor,
            growth_factor: growth,
            hubble_horizon_radius: horizon_r,
            void_fraction,
            cluster_count,
            filament_count,
            nodes,
            filaments,
        }
    }

    /// Computes cosmic filaments connecting neighboring high-density nodes.
    fn compute_filaments(&self, nodes: &[CosmicWebNode], scale: f64) -> Vec<CosmicFilament> {
        let mut filaments = Vec::new();
        let max_link_dist = self.box_scale * scale * 0.35;
        let max_link_sq = max_link_dist * max_link_dist;

        for i in 0..nodes.len() {
            let n1 = &nodes[i];
            if n1.structure_type == MacroStructureType::Void {
                continue;
            }

            for j in (i + 1)..nodes.len() {
                let n2 = &nodes[j];
                if n2.structure_type == MacroStructureType::Void {
                    continue;
                }

                let dx = n1.eulerian_x[0] - n2.eulerian_x[0];
                let dy = n1.eulerian_x[1] - n2.eulerian_x[1];
                let dz = n1.eulerian_x[2] - n2.eulerian_x[2];
                let d_sq = dx * dx + dy * dy + dz * dz;

                if d_sq <= max_link_sq && d_sq > 1e-6 {
                    let len = d_sq.sqrt();
                    let lin_dens = ((n1.density_contrast + n2.density_contrast) / 2.0).max(0.1);
                    filaments.push(CosmicFilament {
                        from_node: n1.id,
                        to_node: n2.id,
                        length: len,
                        linear_density: lin_dens,
                    });
                }
            }
        }
        filaments
    }
}
