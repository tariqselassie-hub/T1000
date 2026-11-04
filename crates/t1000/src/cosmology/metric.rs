// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Cosmological Spacetime Metric & Expansion Dynamics
//!
//! Models FLRW (Friedmann-Lemaître-Robertson-Walker) geometries:
//!   ds^2 = -dt^2 + a^2(t) dx^2 = a^2(eta) [ -d(eta)^2 + dx^2 ]
//!
//! Provides scale factor a(t), conformal scale factor a(eta), physical Hubble
//! parameter H(t) = \dot{a}/a, conformal Hubble parameter \mathcal{H}(eta) = a'/a,
//! and comoving Hubble horizon tracking r_H = (a H)^(-1).

use serde::{Deserialize, Serialize};

/// Cosmological expansion profile governing scale factor evolution.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ExpansionProfile {
    /// Exponential de Sitter cosmic inflation: a(t) = exp(H * t)
    DeSitter,
    /// Radiation-dominated cosmological expansion: a(t) = (1 + 2 H t)^(1/2)
    RadiationDominated,
    /// Matter-dominated cosmological expansion: a(t) = (1 + 3/2 H t)^(2/3)
    MatterDominated,
    /// Non-singular bouncing universe: a(t) = sqrt(1 + (H t)^2)
    BouncingCosmology,
    /// Static flat Minkowski spacetime: a(t) = 1.0 (zero expansion baseline)
    StaticMinkowski,
}

/// Cosmological background spacetime metric.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CosmologicalMetric {
    pub profile: ExpansionProfile,
    pub hubble_rate: f64,
    pub initial_scale: f64,
}

impl CosmologicalMetric {
    pub fn new(profile: ExpansionProfile, hubble_rate: f64) -> Self {
        Self {
            profile,
            hubble_rate: hubble_rate.max(0.0),
            initial_scale: 1.0,
        }
    }

    /// Physical scale factor a(t) as a function of cosmic time t >= 0.
    pub fn scale_factor(&self, t: f64) -> f64 {
        let t_clamped = t.max(0.0);
        let h = self.hubble_rate;
        match self.profile {
            ExpansionProfile::StaticMinkowski => self.initial_scale,
            ExpansionProfile::DeSitter => self.initial_scale * (h * t_clamped).exp(),
            ExpansionProfile::RadiationDominated => {
                self.initial_scale * (1.0 + 2.0 * h * t_clamped).sqrt()
            }
            ExpansionProfile::MatterDominated => {
                self.initial_scale * (1.0 + 1.5 * h * t_clamped).powf(2.0 / 3.0)
            }
            ExpansionProfile::BouncingCosmology => {
                self.initial_scale * (1.0 + (h * t).powi(2)).sqrt()
            }
        }
    }

    /// First time derivative of physical scale factor \dot{a}(t).
    pub fn scale_factor_derivative(&self, t: f64) -> f64 {
        let t_clamped = t.max(0.0);
        let h = self.hubble_rate;
        match self.profile {
            ExpansionProfile::StaticMinkowski => 0.0,
            ExpansionProfile::DeSitter => self.scale_factor(t_clamped) * h,
            ExpansionProfile::RadiationDominated => {
                let a = self.scale_factor(t_clamped);
                (self.initial_scale.powi(2) * h) / a
            }
            ExpansionProfile::MatterDominated => {
                let factor = 1.0 + 1.5 * h * t_clamped;
                self.initial_scale * h * factor.powf(-1.0 / 3.0)
            }
            ExpansionProfile::BouncingCosmology => {
                let num = self.initial_scale * h * h * t;
                let den = (1.0 + (h * t).powi(2)).sqrt();
                num / den
            }
        }
    }

    /// Physical Hubble parameter H(t) = \dot{a}(t) / a(t).
    pub fn hubble_parameter(&self, t: f64) -> f64 {
        let a = self.scale_factor(t);
        if a < 1e-12 {
            return 0.0;
        }
        self.scale_factor_derivative(t) / a
    }

    /// Conformal Hubble parameter \mathcal{H} = a' / a = a * H.
    pub fn conformal_hubble(&self, t: f64) -> f64 {
        let a = self.scale_factor(t);
        let h = self.hubble_parameter(t);
        a * h
    }

    /// Effective potential term in the Mukhanov-Sasaki equation: V_eff = m^2 a^2 - a'' / a.
    /// In physical time: a''/a = a^2 ( \dot{H} + 2 H^2 ).
    pub fn effective_potential(&self, t: f64, field_mass: f64) -> f64 {
        let a = self.scale_factor(t);
        let h = self.hubble_parameter(t);
        let dt = 1e-5;
        let h_plus = self.hubble_parameter(t + dt);
        let h_minus = self.hubble_parameter((t - dt).max(0.0));
        let h_dot = (h_plus - h_minus) / (2.0 * dt);

        let geometric_pump = a * a * (h_dot + 2.0 * h * h);
        let mass_term = field_mass * field_mass * a * a;
        mass_term - geometric_pump
    }

    /// Physical Hubble horizon radius: r_H = c / H(t).
    pub fn physical_horizon(&self, t: f64) -> f64 {
        let h = self.hubble_parameter(t);
        if h > 1e-9 {
            1.0 / h
        } else {
            1e12 // Flat infinite horizon
        }
    }

    /// Comoving Hubble horizon radius: r_c = 1 / (a * H) = 1 / \mathcal{H}.
    pub fn comoving_horizon(&self, t: f64) -> f64 {
        let ch = self.conformal_hubble(t);
        if ch > 1e-9 {
            1.0 / ch
        } else {
            1e12
        }
    }

    /// Checks whether a spatial momentum mode k is super-horizon (k < \mathcal{H}).
    /// Super-horizon modes freeze out and generate squeezed quantum particle pairs.
    pub fn is_super_horizon(&self, k: f64, t: f64) -> bool {
        k < self.conformal_hubble(t)
    }
}

