// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! What device errors and finite sampling do to a VQE energy estimate.
//!
//! Three effects, all acting on the expectation value of each Pauli term `P_k`:
//!
//! * **Gate errors**: a global depolarising channel with the circuit fidelity
//!   `F = (1-p₁)^{n₁} (1-p₂)^{n₂}`. It shrinks every non-identity term by `F` and leaves the
//!   identity coefficient `c₀`, so `E_noisy = c₀ + F (E - c₀)`.
//! * **Readout errors**: a symmetric flip probability `e` per measured qubit multiplies a weight-`w`
//!   term by `(1-2e)^w`.
//! * **Shot noise**: each term is estimated from `N` ±1 samples, a binomial with the (noisy) mean.
//!
//! Known-noise *mitigation* divides each term by its attenuation `a_k = F (1-2e)^{w_k}`: the bias
//! vanishes in expectation and returns as extra variance.

use super::circuit::CircuitCost;
use super::fermion::Sector;
use super::pauli::QubitHamiltonian;

/// A device's error rates. `p1`/`p2` are per-gate depolarising errors, `readout` the per-qubit flip
/// probability.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeviceProfile {
    pub name: &'static str,
    pub p1: f64,
    pub p2: f64,
    pub readout: f64,
}

/// Reference device profiles.
///
/// `cepheus_intra_die` uses the 0.80 % intra-die two-qubit error and ≈2.5 % readout error from
/// the Rigetti Cepheus-1 glued-trees run (README § 22); the single-qubit error of 0.1 % is an
/// assumption, not a measurement.
pub fn profiles() -> Vec<DeviceProfile> {
    vec![
        DeviceProfile {
            name: "ideal gates",
            p1: 0.0,
            p2: 0.0,
            readout: 0.0,
        },
        DeviceProfile {
            name: "cepheus intra-die",
            p1: 1e-3,
            p2: 8e-3,
            readout: 0.025,
        },
        DeviceProfile {
            name: "10x better",
            p1: 1e-4,
            p2: 8e-4,
            readout: 2.5e-3,
        },
        DeviceProfile {
            name: "100x better",
            p1: 1e-5,
            p2: 8e-5,
            readout: 2.5e-4,
        },
    ]
}

/// Process fidelity of a compiled circuit under independent depolarising gate errors.
pub fn circuit_fidelity(cost: &CircuitCost, p1: f64, p2: f64) -> f64 {
    (1.0 - p1).powi(cost.one_qubit as i32) * (1.0 - p2).powi(cost.cnot as i32)
}

/// The largest two-qubit error (with `p1 = p2 / 10`) whose depolarising bias stays below
/// `tolerance` hartree, for ideal energy `energy` and identity coefficient `c0`.
pub fn max_two_qubit_error(cost: &CircuitCost, energy: f64, c0: f64, tolerance: f64) -> f64 {
    let gap = (energy - c0).abs();
    if gap <= tolerance {
        return 0.5;
    }
    let target = 1.0 - tolerance / gap;
    let (mut lo, mut hi): (f64, f64) = (1e-12, 0.5);
    for _ in 0..100 {
        let mid = (lo * hi).sqrt();
        if circuit_fidelity(cost, mid / 10.0, mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Ideal term expectations of the converged state, with the Hamiltonian's coefficients.
#[derive(Debug, Clone)]
pub struct TermStatistics {
    pub constant: f64,
    pub coeffs: Vec<f64>,
    pub weights: Vec<usize>,
    pub means: Vec<f64>,
}

impl TermStatistics {
    /// Statistics of `psi` (a state in `sector`) under the qubit Hamiltonian `h`.
    pub fn new(h: &QubitHamiltonian, sector: &Sector, psi: &[f64]) -> Self {
        let full = embed_sector_state(sector, psi, h.n_qubits);
        TermStatistics {
            constant: h.constant,
            coeffs: h.terms.iter().map(|t| t.coeff).collect(),
            weights: h
                .terms
                .iter()
                .map(|t| (t.x | t.z).count_ones() as usize)
                .collect(),
            means: h.term_expectations(&full),
        }
    }

    /// Attenuation `a_k = F (1 - 2e)^{w_k}` of term `k`.
    pub fn attenuation(&self, k: usize, fidelity: f64, readout: f64) -> f64 {
        fidelity * (1.0 - 2.0 * readout).powi(self.weights[k] as i32)
    }

    pub fn ideal_energy(&self) -> f64 {
        self.constant
            + self
                .coeffs
                .iter()
                .zip(&self.means)
                .map(|(c, m)| c * m)
                .sum::<f64>()
    }

    /// Expected (infinite-shot) energy on the noisy device, without mitigation.
    pub fn noisy_energy(&self, fidelity: f64, readout: f64) -> f64 {
        let terms: f64 = (0..self.coeffs.len())
            .map(|k| self.coeffs[k] * self.attenuation(k, fidelity, readout) * self.means[k])
            .sum();
        self.constant + terms
    }

    /// Per-shot standard deviation `√Σ c_k² (1 - (a_k m_k)²) / a_k²` of the energy estimate when every
    /// term gets the same number of shots; `mitigated` divides each term by its attenuation.
    pub fn std_per_shot(&self, fidelity: f64, readout: f64, mitigated: bool) -> f64 {
        let var: f64 = (0..self.coeffs.len())
            .map(|k| {
                let a = self.attenuation(k, fidelity, readout);
                let scale = if mitigated { 1.0 / a } else { 1.0 };
                let spread = (1.0 - (a * self.means[k]).powi(2)).max(0.0);
                (self.coeffs[k] * scale).powi(2) * spread
            })
            .sum();
        var.sqrt()
    }

    /// Total shots for standard error `epsilon` with the variance-optimal split across terms:
    /// `(Σ_k |c_k / a_k| σ_k)² / ε²` (mitigated) or without the `1/a_k`.
    pub fn optimal_total_shots(
        &self,
        fidelity: f64,
        readout: f64,
        mitigated: bool,
        epsilon: f64,
    ) -> f64 {
        let sum: f64 = (0..self.coeffs.len())
            .map(|k| {
                let a = self.attenuation(k, fidelity, readout);
                let scale = if mitigated { 1.0 / a } else { 1.0 };
                self.coeffs[k].abs() * scale * (1.0 - (a * self.means[k]).powi(2)).max(0.0).sqrt()
            })
            .sum();
        sum * sum / (epsilon * epsilon)
    }
}

/// Places a sector state into the full `2^n` computational basis.
pub fn embed_sector_state(sector: &Sector, psi: &[f64], n_qubits: usize) -> Vec<f64> {
    let mut full = vec![0.0; 1 << n_qubits];
    for (&det, &amp) in sector.dets.iter().zip(psi) {
        full[det as usize] = amp;
    }
    full
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats() -> TermStatistics {
        TermStatistics {
            constant: -1.0,
            coeffs: vec![0.5, -0.25, 0.1],
            weights: vec![1, 2, 4],
            means: vec![0.9, -0.5, 0.0],
        }
    }

    #[test]
    fn noiseless_attenuation_leaves_the_energy_unchanged() {
        let s = stats();
        assert!((s.noisy_energy(1.0, 0.0) - s.ideal_energy()).abs() < 1e-15);
        assert!((s.ideal_energy() - (-1.0 + 0.45 + 0.125)).abs() < 1e-15);
    }

    #[test]
    fn depolarising_shrinks_the_energy_toward_the_identity_coefficient() {
        let s = stats();
        let f = 0.8;
        let want = s.constant + f * (s.ideal_energy() - s.constant);
        assert!((s.noisy_energy(f, 0.0) - want).abs() < 1e-15);
    }

    #[test]
    fn readout_error_attenuates_by_weight() {
        let s = stats();
        let e: f64 = 0.05;
        let want = -1.0 + 0.5 * 0.9 * (1.0 - 2.0 * e) + (-0.25) * (-0.5) * (1.0 - 2.0 * e).powi(2);
        assert!((s.noisy_energy(1.0, e) - want).abs() < 1e-15);
    }

    #[test]
    fn mitigation_trades_the_bias_for_variance() {
        let s = stats();
        let (f, e) = (0.7, 0.03);
        assert!(s.std_per_shot(f, e, true) > s.std_per_shot(f, e, false));
        assert!(s.optimal_total_shots(f, e, true, 1e-3) > s.optimal_total_shots(f, e, false, 1e-3));
        // Noise-free: mitigation is a no-op.
        assert!((s.std_per_shot(1.0, 0.0, true) - s.std_per_shot(1.0, 0.0, false)).abs() < 1e-15);
    }

    #[test]
    fn roundoff_that_pushes_a_mean_past_one_does_not_produce_nan() {
        let s = TermStatistics {
            constant: 0.0,
            coeffs: vec![1.0],
            weights: vec![1],
            means: vec![1.0 + 1e-12],
        };
        assert!(s.optimal_total_shots(1.0, 0.0, false, 1e-3).is_finite());
        assert!(s.std_per_shot(1.0, 0.0, true).is_finite());
    }

    #[test]
    fn shots_scale_as_the_inverse_square_of_the_target_error() {
        let s = stats();
        let a = s.optimal_total_shots(1.0, 0.0, false, 1e-2);
        let b = s.optimal_total_shots(1.0, 0.0, false, 1e-3);
        assert!((b / a - 100.0).abs() < 1e-9);
    }

    #[test]
    fn tolerable_two_qubit_error_makes_the_bias_exactly_the_tolerance() {
        let cost = CircuitCost {
            rotations: 10,
            cnot: 200,
            one_qubit: 300,
        };
        let (energy, c0, tol) = (-7.88, -2.0, 1.594e-3);
        let p2 = max_two_qubit_error(&cost, energy, c0, tol);
        let f = circuit_fidelity(&cost, p2 / 10.0, p2);
        assert!(((1.0 - f) * (energy - c0).abs() - tol).abs() < 1e-9);
        assert!(p2 > 1e-7 && p2 < 1e-3);
    }
}
