// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Cosmological Simulation Orchestrator
//!
//! Orchestrates the end-to-end evolution of quantum fields across an expanding
//! spacetime manifold. Tracks scale factor growth, horizon crossings, Bogoliubov
//! particle creation rates, and holographic entanglement entropy accumulation.

use super::bogoliubov::CosmicParticleSpectrum;
use super::field::DiscretizedQuantumField;
use super::horizon::{evaluate_horizon_entanglement, HorizonEntanglementReport};
use super::metric::{CosmologicalMetric, ExpansionProfile};
use super::structure::{CosmicStructureSnapshot, ZeldovichStructureEngine};
use serde::{Deserialize, Serialize};

/// Configuration for the quantum cosmological simulation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CosmologyConfig {
    pub profile: ExpansionProfile,
    pub hubble_rate: f64,
    pub field_mass: f64,
    pub num_modes: usize,
    pub box_length: f64,
    pub t_initial: f64,
    pub t_final: f64,
    pub time_steps: usize,
}

impl Default for CosmologyConfig {
    fn default() -> Self {
        Self {
            profile: ExpansionProfile::DeSitter,
            hubble_rate: 1.2,
            field_mass: 0.1,
            num_modes: 16,
            box_length: 20.0,
            t_initial: 0.0,
            t_final: 2.5,
            time_steps: 120,
        }
    }
}

/// Instantaneous state snapshot at an epoch in cosmic time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CosmologicalEpochSnapshot {
    pub cosmic_time: f64,
    pub scale_factor: f64,
    pub hubble_parameter: f64,
    pub total_particle_density: f64,
    pub horizon_entanglement_entropy: f64,
    pub super_horizon_modes: usize,
    pub mean_power_spectrum: f64,
}

/// Comprehensive report containing cosmic time-series and final state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CosmologicalSimulationReport {
    pub config: CosmologyConfig,
    pub epochs: Vec<CosmologicalEpochSnapshot>,
    pub final_spectrum: CosmicParticleSpectrum,
    pub final_horizon: HorizonEntanglementReport,
    pub max_unitarity_error: f64,
    pub total_particles_created: f64,
    pub peak_entanglement_entropy: f64,
    pub macro_structure: Option<CosmicStructureSnapshot>,
}

/// The sovereign Quantum Universe Simulator.
#[derive(Debug, Clone)]
pub struct CosmologyEngine {
    pub config: CosmologyConfig,
    pub metric: CosmologicalMetric,
    pub field: DiscretizedQuantumField,
}

impl CosmologyEngine {
    pub fn new(config: CosmologyConfig) -> Self {
        let metric = CosmologicalMetric::new(config.profile, config.hubble_rate);
        let field = DiscretizedQuantumField::new(
            config.num_modes,
            config.box_length,
            config.field_mass,
            metric.scale_factor(config.t_initial),
        );

        Self {
            config,
            metric,
            field,
        }
    }

    /// Executes the full cosmological expansion simulation across cosmic time.
    pub fn run_simulation(&mut self) -> CosmologicalSimulationReport {
        let steps = self.config.time_steps.max(2);
        let dt = (self.config.t_final - self.config.t_initial) / (steps as f64);
        let mut epochs = Vec::with_capacity(steps + 1);

        let mut current_t = self.config.t_initial;
        let mut conformal_eta = 0.0;
        let mut max_u_err = 0.0;

        for step in 0..=steps {
            let a = self.metric.scale_factor(current_t);
            let h = self.metric.hubble_parameter(current_t);

            // Instantaneous Bogoliubov and horizon telemetry
            let spectrum = CosmicParticleSpectrum::evaluate(
                &self.field.modes,
                self.field.field_mass,
                current_t,
                a,
                h,
                conformal_eta,
            );

            let horizon = evaluate_horizon_entanglement(&self.metric, current_t, &spectrum.mode_reports);

            let mean_p = spectrum
                .mode_reports
                .iter()
                .map(|m| m.power_spectrum)
                .sum::<f64>()
                / (spectrum.mode_reports.len().max(1) as f64);

            epochs.push(CosmologicalEpochSnapshot {
                cosmic_time: current_t,
                scale_factor: a,
                hubble_parameter: h,
                total_particle_density: spectrum.total_particle_density,
                horizon_entanglement_entropy: horizon.total_entanglement_entropy,
                super_horizon_modes: horizon.super_horizon_modes_count,
                mean_power_spectrum: mean_p,
            });

            let u_err = self.field.max_unitarity_error();
            if u_err > max_u_err {
                max_u_err = u_err;
            }

            if step < steps {
                let d_eta = dt / a.max(1e-6);
                self.field.step_field(&self.metric, current_t, d_eta);
                current_t += dt;
                conformal_eta += d_eta;
            }
        }

        let final_t = self.config.t_final;
        let final_a = self.metric.scale_factor(final_t);
        let final_h = self.metric.hubble_parameter(final_t);
        let final_spectrum = CosmicParticleSpectrum::evaluate(
            &self.field.modes,
            self.field.field_mass,
            final_t,
            final_a,
            final_h,
            conformal_eta,
        );
        let final_horizon = evaluate_horizon_entanglement(&self.metric, final_t, &final_spectrum.mode_reports);

        let total_particles = final_spectrum.total_particle_density;
        let peak_entropy = epochs
            .iter()
            .map(|e| e.horizon_entanglement_entropy)
            .fold(0.0_f64, f64::max);

        // Generate macroscopic cosmic structure snapshot from primordial power spectrum
        let power_levels: Vec<f64> = final_spectrum
            .mode_reports
            .iter()
            .map(|m| m.power_spectrum)
            .collect();
        let zeldovich = ZeldovichStructureEngine::new(96, self.config.box_length, &power_levels);
        let macro_struct = zeldovich.snapshot(final_t, final_a, final_h);

        CosmologicalSimulationReport {
            config: self.config.clone(),
            epochs,
            final_spectrum,
            final_horizon,
            max_unitarity_error: max_u_err,
            total_particles_created: total_particles,
            peak_entanglement_entropy: peak_entropy,
            macro_structure: Some(macro_struct),
        }
    }
}
