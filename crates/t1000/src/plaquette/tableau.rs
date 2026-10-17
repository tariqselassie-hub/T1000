// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Wilson Flux Tableau & Discrete 2D Stokes' Theorem Engine
//!
//! Evaluates link involutions ($Z_e^2 = I$), plaquette flux observables ($W_p$),
//! boundary perimeter operators ($W_{\partial \Sigma}$), and covariance matrices.

use serde::{Deserialize, Serialize};
use super::geometry::PlaquetteLattice;

/// Wilson Loop / Flux state representation tracking gauge observables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WilsonFluxTableau {
    pub plaquette_fluxes: Vec<f64>,
    pub bulk_product_flux: f64,
    pub boundary_perimeter_flux: f64,
    pub stokes_drift: f64,
    pub mean_link_polarization: f64,
}

/// Comprehensive report of Stokes invariant pressure testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StokesPressureReport {
    pub lattice_name: String,
    pub total_qubits: usize,
    pub num_plaquettes: usize,
    pub num_internal_spokes: usize,
    pub num_boundary_links: usize,
    pub shots_tested: usize,
    pub max_shot_stokes_drift: f64,
    pub mean_shot_stokes_drift: f64,
    pub quantum_expectation_drift: f64,
    pub covariance_matrix: Vec<Vec<f64>>,
    pub max_next_nearest_covariance: f64,
    pub max_nearest_covariance: f64,
    pub diagonal_dominance_ratio: f64,
    pub c3_tripartite_frustration: Option<f64>,
}

/// Core algebraic evaluator for discrete 2D Stokes theorem.
#[derive(Debug, Clone)]
pub struct DiscreteStokesEngine {
    pub lattice: PlaquetteLattice,
}

impl DiscreteStokesEngine {
    pub fn new(lattice: PlaquetteLattice) -> Self {
        Self { lattice }
    }

    /// Evaluates link value z_l in {-1, +1} for a given bitstring shot.
    #[inline(always)]
    pub fn evaluate_link_shot(link_id: usize, shot: usize) -> f64 {
        let b = (shot >> link_id) & 1;
        if b == 0 {
            1.0
        } else {
            -1.0
        }
    }

    /// Evaluates plaquette flux W_p in {-1, +1} for a single bitstring shot.
    pub fn evaluate_plaquette_shot(&self, plaquette_idx: usize, shot: usize) -> f64 {
        let p = &self.lattice.plaquettes[plaquette_idx];
        let mut flux = 1.0;
        for &l_id in &p.link_ids {
            flux *= Self::evaluate_link_shot(l_id, shot);
        }
        flux
    }

    /// Evaluates boundary perimeter operator W_{\partial \Sigma} for a single bitstring shot.
    pub fn evaluate_boundary_shot(&self, shot: usize) -> f64 {
        let mut flux = 1.0;
        for &l_id in &self.lattice.boundary_link_ids {
            flux *= Self::evaluate_link_shot(l_id, shot);
        }
        flux
    }

    /// Tests discrete Stokes theorem shot-by-shot across an ensemble of measurement outcomes.
    pub fn stress_test_shots(&self, shots: &[usize]) -> StokesPressureReport {
        let n_plaq = self.lattice.plaquettes.len();
        let n_shots = shots.len();

        let mut max_drift = 0.0f64;
        let mut sum_drift = 0.0f64;

        let mut plaq_flux_samples = vec![vec![0.0f64; n_shots]; n_plaq];
        let mut boundary_samples = vec![0.0f64; n_shots];

        for (s_idx, &shot) in shots.iter().enumerate() {
            let mut bulk_prod = 1.0f64;
            for p_idx in 0..n_plaq {
                let w = self.evaluate_plaquette_shot(p_idx, shot);
                plaq_flux_samples[p_idx][s_idx] = w;
                bulk_prod *= w;
            }
            let bound_w = self.evaluate_boundary_shot(shot);
            boundary_samples[s_idx] = bound_w;

            let drift = (bulk_prod - bound_w).abs();
            if drift > max_drift {
                max_drift = drift;
            }
            sum_drift += drift;
        }

        let mean_drift = sum_drift / (n_shots as f64);

        // Compute Plaquette Flux Covariance Matrix C(W_i, W_j)
        let mut cov_matrix = vec![vec![0.0f64; n_plaq]; n_plaq];
        let mut plaq_means = vec![0.0f64; n_plaq];
        for i in 0..n_plaq {
            plaq_means[i] = plaq_flux_samples[i].iter().sum::<f64>() / (n_shots as f64);
        }

        for i in 0..n_plaq {
            for j in 0..n_plaq {
                let mut cov = 0.0f64;
                for k in 0..n_shots {
                    cov += (plaq_flux_samples[i][k] - plaq_means[i])
                        * (plaq_flux_samples[j][k] - plaq_means[j]);
                }
                cov_matrix[i][j] = cov / (n_shots as f64);
            }
        }

        // Measure Nearest-Neighbor vs. Next-Nearest-Neighbor Covariance
        let mut max_nn = 0.0f64;
        let mut max_nnn = 0.0f64;

        for i in 0..n_plaq {
            for j in 0..n_plaq {
                let dist = (i as isize - j as isize).abs();
                let abs_cov = cov_matrix[i][j].abs();
                if dist == 1 && abs_cov > max_nn {
                    max_nn = abs_cov;
                } else if dist >= 2 && abs_cov > max_nnn {
                    max_nnn = abs_cov;
                }
            }
        }

        let dominance_ratio = if max_nn > 1e-12 {
            max_nnn / max_nn
        } else {
            0.0
        };

        // Extract Tripartite Cumulant C3 if on Delta_3
        let c3 = if n_plaq >= 3 {
            let m_a = plaq_means[0];
            let m_b = plaq_means[1];
            let m_c = plaq_means[2];
            let mut avg_abc = 0.0;
            let mut avg_ab = 0.0;
            let mut avg_bc = 0.0;
            let mut avg_ac = 0.0;
            for k in 0..n_shots {
                let a = plaq_flux_samples[0][k];
                let b = plaq_flux_samples[1][k];
                let c = plaq_flux_samples[2][k];
                avg_abc += a * b * c;
                avg_ab += a * b;
                avg_bc += b * c;
                avg_ac += a * c;
            }
            avg_abc /= n_shots as f64;
            avg_ab /= n_shots as f64;
            avg_bc /= n_shots as f64;
            avg_ac /= n_shots as f64;
            let cumulant = avg_abc - (m_a * avg_bc + m_b * avg_ac + m_c * avg_ab) + 2.0 * m_a * m_b * m_c;
            Some(cumulant)
        } else {
            None
        };

        StokesPressureReport {
            lattice_name: self.lattice.name.clone(),
            total_qubits: self.lattice.total_qubits,
            num_plaquettes: n_plaq,
            num_internal_spokes: self.lattice.internal_spoke_ids.len(),
            num_boundary_links: self.lattice.boundary_link_ids.len(),
            shots_tested: n_shots,
            max_shot_stokes_drift: max_drift,
            mean_shot_stokes_drift: mean_drift,
            quantum_expectation_drift: max_drift,
            covariance_matrix: cov_matrix,
            max_next_nearest_covariance: max_nnn,
            max_nearest_covariance: max_nn,
            diagonal_dominance_ratio: dominance_ratio,
            c3_tripartite_frustration: c3,
        }
    }
}
