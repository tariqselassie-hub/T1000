// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! VQE driven only by noisy, finite-shot energy estimates, optimised with SPSA.
//!
//! This is what a device run looks like: the optimiser never sees the exact energy or gradient,
//! only a sampled estimate with the device's attenuation. The result is judged afterwards by the
//! exact energy of the parameters it returns.

use super::driver::SolvedProblem;
use super::noise::TermStatistics;
use super::uccsd::{energy_and_gradient, prepare_state};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Settings for a noisy SPSA run.
#[derive(Debug, Clone, Copy)]
pub struct NoisyConfig {
    /// Shots spent on every Pauli term of every energy estimate.
    pub shots_per_term: u64,
    /// Circuit fidelity (1.0 for ideal gates).
    pub fidelity: f64,
    /// Per-qubit readout flip probability.
    pub readout: f64,
    /// Divide each term by its known attenuation.
    pub mitigate: bool,
    pub iterations: usize,
    pub seed: u64,
}

/// Outcome of a noisy optimisation, judged with the exact energy.
#[derive(Debug, Clone)]
pub struct NoisyRun {
    /// Exact energy of the returned parameters.
    pub energy: f64,
    pub error_vs_fci: f64,
    /// Exact energy of the iterate every `stride` iterations.
    pub trace: Vec<f64>,
    pub evaluations: usize,
}

/// A standard normal deviate (Box-Muller).
fn gaussian(rng: &mut ChaCha8Rng) -> f64 {
    let (u1, u2): (f64, f64) = (rng.gen::<f64>().max(1e-300), rng.gen());
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// One sampled energy estimate: each term's mean is attenuated, then drawn from its binomial
/// (normal approximation, clamped to ±1), and optionally mitigated.
fn sampled_energy(stats: &TermStatistics, cfg: &NoisyConfig, rng: &mut ChaCha8Rng) -> f64 {
    let mut e = stats.constant;
    for k in 0..stats.coeffs.len() {
        let a = stats.attenuation(k, cfg.fidelity, cfg.readout);
        let mean = a * stats.means[k];
        let sd = ((1.0 - mean * mean).max(0.0) / cfg.shots_per_term as f64).sqrt();
        let sample = (mean + sd * gaussian(rng)).clamp(-1.0, 1.0);
        e += stats.coeffs[k] * if cfg.mitigate { sample / a } else { sample };
    }
    e
}

/// Noisy objective at `theta`.
fn objective(
    solved: &SolvedProblem,
    theta: &[f64],
    cfg: &NoisyConfig,
    rng: &mut ChaCha8Rng,
) -> f64 {
    let psi = prepare_state(&solved.excitations, &solved.reference, theta);
    let stats = TermStatistics::new(&solved.qubit_hamiltonian, &solved.sector, &psi);
    sampled_energy(&stats, cfg, rng)
}

/// Exact energy of `theta`.
fn exact_energy(solved: &SolvedProblem, theta: &[f64]) -> f64 {
    energy_and_gradient(
        &solved.hamiltonian,
        &solved.excitations,
        &solved.reference,
        theta,
    )
    .0
}

/// SPSA from the Hartree-Fock point (`θ = 0`), averaging the last quarter of the iterates.
pub fn spsa_vqe(solved: &SolvedProblem, cfg: &NoisyConfig) -> NoisyRun {
    let n = solved.excitations.len();
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    let mut theta = vec![0.0; n];
    let (a0, c0, stab) = (1.0, 0.1, 0.1 * cfg.iterations as f64);
    let stride = (cfg.iterations / 20).max(1);
    let (mut trace, mut evaluations) = (Vec::new(), 0);
    let mut tail = vec![0.0; n];
    let tail_start = cfg.iterations - cfg.iterations / 4;
    let mut tail_count = 0.0;
    for k in 0..cfg.iterations {
        let ak = a0 / (k as f64 + 1.0 + stab).powf(0.602);
        let ck = c0 / (k as f64 + 1.0).powf(0.101);
        let delta: Vec<f64> = (0..n)
            .map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 })
            .collect();
        let shifted = |sign: f64| -> Vec<f64> {
            theta
                .iter()
                .zip(&delta)
                .map(|(t, d)| t + sign * ck * d)
                .collect()
        };
        let plus = objective(solved, &shifted(1.0), cfg, &mut rng);
        let minus = objective(solved, &shifted(-1.0), cfg, &mut rng);
        evaluations += 2;
        let scale = (plus - minus) / (2.0 * ck);
        theta
            .iter_mut()
            .zip(&delta)
            .for_each(|(t, d)| *t -= ak * scale * d);
        if k % stride == 0 {
            trace.push(exact_energy(solved, &theta));
        }
        if k >= tail_start {
            tail.iter_mut().zip(&theta).for_each(|(s, t)| *s += t);
            tail_count += 1.0;
        }
    }
    if tail_count > 0.0 {
        theta = tail.iter().map(|s| s / tail_count).collect();
    }
    let energy = exact_energy(solved, &theta);
    NoisyRun {
        energy,
        error_vs_fci: energy - solved.report.fci_energy,
        trace,
        evaluations,
    }
}

#[cfg(test)]
mod tests {
    use super::super::driver::{solve_detailed, VqeConfig};
    use super::super::molecules::by_formula;
    use super::*;

    fn h2() -> SolvedProblem {
        solve_detailed(&by_formula("H2").unwrap(), &VqeConfig::default()).unwrap()
    }

    #[test]
    fn spsa_with_many_shots_recovers_the_h2_ground_state() {
        let solved = h2();
        let cfg = NoisyConfig {
            shots_per_term: 100_000,
            fidelity: 1.0,
            readout: 0.0,
            mitigate: false,
            iterations: 400,
            seed: 3,
        };
        let run = spsa_vqe(&solved, &cfg);
        let hf_error = solved.report.hf_energy - solved.report.fci_energy;
        assert!(run.error_vs_fci >= -1e-9, "variational bound");
        assert!(
            run.error_vs_fci < 0.05 * hf_error,
            "{} vs HF error {hf_error}",
            run.error_vs_fci
        );
    }

    #[test]
    fn the_sampled_estimator_is_unbiased_with_ideal_gates_and_biased_otherwise() {
        let solved = h2();
        let psi = prepare_state(&solved.excitations, &solved.reference, &solved.theta);
        let stats = TermStatistics::new(&solved.qubit_hamiltonian, &solved.sector, &psi);
        assert!((stats.ideal_energy() - solved.report.vqe_energy).abs() < 1e-9);
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let mean = |cfg: &NoisyConfig, rng: &mut ChaCha8Rng| -> f64 {
            (0..2000)
                .map(|_| sampled_energy(&stats, cfg, rng))
                .sum::<f64>()
                / 2000.0
        };
        let ideal = NoisyConfig {
            shots_per_term: 1000,
            fidelity: 1.0,
            readout: 0.0,
            mitigate: false,
            iterations: 0,
            seed: 0,
        };
        assert!((mean(&ideal, &mut rng) - stats.ideal_energy()).abs() < 5e-4);
        let noisy = NoisyConfig {
            fidelity: 0.8,
            readout: 0.02,
            ..ideal
        };
        let biased = mean(&noisy, &mut rng);
        assert!((biased - stats.noisy_energy(0.8, 0.02)).abs() < 1e-3);
        assert!(
            (biased - stats.ideal_energy()).abs() > 5e-2,
            "noise must move the energy"
        );
        let mitigated = mean(
            &NoisyConfig {
                mitigate: true,
                ..noisy
            },
            &mut rng,
        );
        assert!(
            (mitigated - stats.ideal_energy()).abs() < 2e-3,
            "{mitigated}"
        );
    }
}
