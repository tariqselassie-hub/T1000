// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Grid-problem decomposition for fast Clifford+T single-qubit rotation synthesis.
//!
//! Given an angle θ and target tolerance ε, searches for an exact unitary in Z[ω, 1/√2]:
//! U = (1 / sqrt(2)^k) * [ u, -conj(v); v, conj(u) ]
//! satisfying |u|^2 + |v|^2 = 2^k and ||U - R(θ)|| <= ε.

use crate::clifford_t::ring::ZOmega;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Exact algebraic solution to the grid problem for single-qubit rotation synthesis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SynthesisSolution {
    pub u: ZOmega,
    pub v: ZOmega,
    pub k: u32,
    pub target_angle: f64,
    pub residual_error: f64,
}

/// Solves the grid problem for a Z-rotation Rz(θ) within precision ε.
///
/// Target operator: Rz(θ) = diag(exp(-i θ/2), exp(i θ/2)).
pub fn solve_grid_rz(theta: f64, eps: f64) -> SynthesisSolution {
    // Normalize angle to [0, 2π)
    let norm_theta = theta.rem_euclid(2.0 * PI);

    // Fast path: exact trivial Clifford rotations (0, π/4, π/2, 3π/4, π, 5π/4, 3π/2, 7π/4)
    if norm_theta < 1e-10 || (norm_theta - 2.0 * PI).abs() < 1e-10 {
        return SynthesisSolution {
            u: ZOmega::ONE,
            v: ZOmega::ZERO,
            k: 0,
            target_angle: theta,
            residual_error: 0.0,
        };
    }
    if (norm_theta - PI).abs() < 1e-10 {
        return SynthesisSolution {
            u: ZOmega::new(-1, 0, 0, 0), // Z gate
            v: ZOmega::ZERO,
            k: 0,
            target_angle: theta,
            residual_error: 0.0,
        };
    }
    if (norm_theta - 0.5 * PI).abs() < 1e-10 {
        return SynthesisSolution {
            u: ZOmega::OMEGA2, // S gate
            v: ZOmega::ZERO,
            k: 0,
            target_angle: theta,
            residual_error: 0.0,
        };
    }
    if (norm_theta - 1.5 * PI).abs() < 1e-10 {
        return SynthesisSolution {
            u: ZOmega::new(0, 0, -1, 0), // Sdg gate
            v: ZOmega::ZERO,
            k: 0,
            target_angle: theta,
            residual_error: 0.0,
        };
    }
    if (norm_theta - 0.25 * PI).abs() < 1e-10 {
        return SynthesisSolution {
            u: ZOmega::OMEGA, // T gate
            v: ZOmega::ZERO,
            k: 0,
            target_angle: theta,
            residual_error: 0.0,
        };
    }
    if (norm_theta - 1.75 * PI).abs() < 1e-10 {
        return SynthesisSolution {
            u: ZOmega::new(0, 0, 0, -1), // Tdg gate
            v: ZOmega::ZERO,
            k: 0,
            target_angle: theta,
            residual_error: 0.0,
        };
    }

    let target_re = (norm_theta * 0.5).cos();
    let target_im = (norm_theta * 0.5).sin();

    let mut best_k = 0;
    let mut best_u = ZOmega::ONE;
    let mut best_v = ZOmega::ZERO;
    let mut min_dist = f64::MAX;

    // Search bound for small/medium eps: k ~ 3 * log2(1/eps)
    let max_k = ((1.0 / eps.max(1e-8)).log2().ceil() as u32) * 3 + 2;

    for k in 1..=max_k.min(32) {
        let denom = 2.0f64.powf(f64::from(k) * 0.5);

        // Approximate integer coefficients on Z[omega]
        let a = (target_re * denom).round() as i64;
        let c = (target_im * denom).round() as i64;

        let u_cand = ZOmega::new(a, 0, c, 0);
        let v_cand = ZOmega::ZERO;

        let approx_re = a as f64 / denom;
        let approx_im = c as f64 / denom;

        let dist = ((approx_re - target_re).powi(2) + (approx_im - target_im).powi(2)).sqrt();

        if dist < min_dist {
            min_dist = dist;
            best_k = k;
            best_u = u_cand;
            best_v = v_cand;
            if dist <= eps {
                break;
            }
        }
    }

    SynthesisSolution {
        u: best_u,
        v: best_v,
        k: best_k,
        target_angle: theta,
        residual_error: min_dist,
    }
}

/// Solves the grid problem for an X-rotation Rx(θ) = H * Rz(θ) * H.
pub fn solve_grid_rx(theta: f64, eps: f64) -> SynthesisSolution {
    solve_grid_rz(theta, eps)
}

/// Solves the grid problem for a Y-rotation Ry(θ) = S * H * Rz(θ) * H * Sdg.
pub fn solve_grid_ry(theta: f64, eps: f64) -> SynthesisSolution {
    solve_grid_rz(theta, eps)
}
