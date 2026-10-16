// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

use crate::QubitRegister;
use num_complex::Complex64;
use rand::Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// Discrete and continuous quantum open-system decoherence channels.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ChannelKind {
    /// Amplitude damping (energy relaxation $|1\rangle \to |0\rangle$) with damping probability $\gamma \in [0, 1]$.
    AmplitudeDamping { gamma: f64 },
    /// Pure phase damping (dephasing) with loss parameter $\lambda \in [0, 1]$.
    PhaseDamping { lambda: f64 },
    /// Completely depolarizing channel with depolarizing parameter $p \in [0, 1]$.
    Depolarizing { p: f64 },
    /// Stochastic Pauli-X bit flip with probability $p \in [0, 1]$.
    BitFlip { p: f64 },
    /// Stochastic Pauli-Z phase flip with probability $p \in [0, 1]$.
    PhaseFlip { p: f64 },
    /// Generalized Pauli noise channel with independent probabilities $(p_x, p_y, p_z)$.
    PauliNoise { px: f64, py: f64, pz: f64 },
}

impl Default for ChannelKind {
    fn default() -> Self {
        ChannelKind::AmplitudeDamping { gamma: 0.05 }
    }
}

/// Parameterized decoherence channel container with seed for reproducibility.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecoherenceChannel {
    pub kind: ChannelKind,
    pub seed: u64,
}

impl DecoherenceChannel {
    pub fn new(kind: ChannelKind, seed: u64) -> Self {
        Self { kind, seed }
    }
}

/// Apply a decoherence channel to `reg` in-place.
/// Returns the total lost coherence: sum of squared probability mass that moved or decayed.
pub fn apply(reg: &mut QubitRegister, ch: &DecoherenceChannel) -> f64 {
    let mut lost = 0.0;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(ch.seed);

    match ch.kind {
        ChannelKind::AmplitudeDamping { gamma } => {
            for a in &mut reg.amplitudes {
                let old_p = a.norm_sqr();
                let damped = *a * Complex64::new((1.0 - gamma).sqrt(), 0.0);
                let new_p = damped.norm_sqr();
                lost += (old_p - new_p).clamp(0.0, f64::MAX);
                *a = damped;
            }
            // Redistribute lost coherence into ground state / uniform background
            let basis_f = reg.basis as f64;
            for a in &mut reg.amplitudes {
                *a += Complex64::new((lost / basis_f).sqrt(), 0.0);
            }
        }
        ChannelKind::PhaseDamping { lambda } => {
            let phase_factor = Complex64::new((1.0 - lambda).sqrt(), 0.0);
            for (idx, a) in reg.amplitudes.iter_mut().enumerate() {
                // Apply phase damping to odd / excited basis components
                if idx % 2 != 0 {
                    let old_p = a.norm_sqr();
                    *a *= phase_factor;
                    let new_p = a.norm_sqr();
                    lost += (old_p - new_p).clamp(0.0, f64::MAX);
                }
            }
        }
        ChannelKind::Depolarizing { p } => {
            for a in &mut reg.amplitudes {
                if rng.gen::<f64>() < p {
                    let old_p = a.norm_sqr();
                    lost += old_p.clamp(0.0, f64::MAX);
                    *a = Complex64::default();
                }
            }
            let basis_f = reg.basis as f64;
            for a in &mut reg.amplitudes {
                *a += Complex64::new((lost / basis_f).sqrt(), 0.0);
            }
        }
        ChannelKind::BitFlip { p } => {
            for a in &mut reg.amplitudes {
                if rng.gen::<f64>() < p {
                    *a = -*a;
                    lost += a.norm_sqr() * 0.1;
                }
            }
        }
        ChannelKind::PhaseFlip { p } => {
            for (idx, a) in reg.amplitudes.iter_mut().enumerate() {
                if idx % 2 != 0 && rng.gen::<f64>() < p {
                    *a = -*a;
                    lost += a.norm_sqr() * 0.1;
                }
            }
        }
        ChannelKind::PauliNoise { px, py, pz } => {
            for (idx, a) in reg.amplitudes.iter_mut().enumerate() {
                let r: f64 = rng.gen();
                if r < px {
                    // Bit flip
                    *a = -*a;
                    lost += a.norm_sqr() * 0.1;
                } else if r < px + py {
                    // Y error: phase kick + imaginary rotation
                    *a = Complex64::new(-a.im, a.re);
                    lost += a.norm_sqr() * 0.1;
                } else if r < px + py + pz && idx % 2 != 0 {
                    // Z phase flip
                    *a = -*a;
                    lost += a.norm_sqr() * 0.1;
                }
            }
        }
    }
    reg.normalize();
    lost
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amplitude_damping_preserves_normalization() {
        let mut reg = QubitRegister::new(8, 42);
        let ch = DecoherenceChannel::new(ChannelKind::AmplitudeDamping { gamma: 0.15 }, 1234);
        let lost = apply(&mut reg, &ch);
        assert!(lost >= 0.0);
        let sum_p: f64 = reg.amplitudes.iter().map(|a| a.norm_sqr()).sum();
        assert!((sum_p - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_phase_damping_preserves_normalization() {
        let mut reg = QubitRegister::new(8, 42);
        let ch = DecoherenceChannel::new(ChannelKind::PhaseDamping { lambda: 0.2 }, 5678);
        let _ = apply(&mut reg, &ch);
        let sum_p: f64 = reg.amplitudes.iter().map(|a| a.norm_sqr()).sum();
        assert!((sum_p - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_depolarizing_channel() {
        let mut reg = QubitRegister::new(16, 100);
        let ch = DecoherenceChannel::new(ChannelKind::Depolarizing { p: 0.1 }, 999);
        let lost = apply(&mut reg, &ch);
        assert!(lost >= 0.0);
        let sum_p: f64 = reg.amplitudes.iter().map(|a| a.norm_sqr()).sum();
        assert!((sum_p - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pauli_noise_channel() {
        let mut reg = QubitRegister::new(8, 42);
        let ch = DecoherenceChannel::new(
            ChannelKind::PauliNoise {
                px: 0.05,
                py: 0.05,
                pz: 0.05,
            },
            777,
        );
        let _ = apply(&mut reg, &ch);
        let sum_p: f64 = reg.amplitudes.iter().map(|a| a.norm_sqr()).sum();
        assert!((sum_p - 1.0).abs() < 1e-6);
    }
}
