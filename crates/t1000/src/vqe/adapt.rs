// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! ADAPT-VQE (Grimsley et al., Nat. Commun. 10, 3007, 2019): grow the ansatz one operator at a
//! time instead of fixing it in advance.
//!
//! Each step measures `∂E/∂θ = 2 ⟨Hψ|Aψ⟩` for every generator `A` in the pool at the current
//! state, appends the one with the largest magnitude (operators may repeat), and re-optimises all
//! angles from the previous optimum. The loop stops when the norm of the pool gradient falls
//! below a tolerance. The stopping rule uses only quantities a device could measure; the exact
//! energy is never consulted.

use super::fermion::SparseMatrix;
use super::optimize::bfgs;
use super::uccsd::{energy_and_gradient, prepare_state, Excitation};
use serde::{Deserialize, Serialize};

/// Stopping and optimisation settings.
#[derive(Debug, Clone)]
pub struct AdaptConfig {
    /// Stop when the Euclidean norm of the pool gradient falls below this.
    pub gradient_norm_tolerance: f64,
    /// Upper bound on the ansatz length.
    pub max_operators: usize,
    pub optimizer_iterations: usize,
    pub optimizer_tolerance: f64,
}

impl Default for AdaptConfig {
    fn default() -> Self {
        // 1e-3 is the threshold of the original ADAPT-VQE paper.
        AdaptConfig {
            gradient_norm_tolerance: 1e-3,
            max_operators: 250,
            optimizer_iterations: 400,
            optimizer_tolerance: 1e-7,
        }
    }
}

/// One growth step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptStep {
    pub operator: String,
    pub pool_gradient_norm: f64,
    pub selected_gradient: f64,
    /// Energy after re-optimising with the new operator.
    pub energy: f64,
}

/// The grown ansatz and its energy.
#[derive(Debug, Clone)]
pub struct AdaptResult {
    pub energy: f64,
    /// True when the pool gradient norm fell below the tolerance (not when `max_operators` hit).
    pub converged: bool,
    /// Pool indices in ansatz order (applied first to last); an index may appear more than once.
    pub selected: Vec<usize>,
    pub theta: Vec<f64>,
    pub steps: Vec<AdaptStep>,
    pub energy_evaluations: usize,
    pub final_gradient_norm: f64,
}

/// Pool gradients `2 ⟨Hψ|A_k ψ⟩` at state `psi`.
pub fn pool_gradients(h: &SparseMatrix, pool: &[Excitation], psi: &[f64]) -> Vec<f64> {
    let mut hpsi = vec![0.0; h.n];
    h.apply(psi, &mut hpsi);
    let mut a_psi = vec![0.0; h.n];
    pool.iter()
        .map(|a| {
            a.apply_generator(psi, &mut a_psi);
            2.0 * hpsi.iter().zip(&a_psi).map(|(x, y)| x * y).sum::<f64>()
        })
        .collect()
}

/// Runs ADAPT-VQE from `reference` with operators drawn from `pool`.
pub fn adapt_vqe(
    h: &SparseMatrix,
    pool: &[Excitation],
    reference: &[f64],
    cfg: &AdaptConfig,
) -> AdaptResult {
    let mut selected: Vec<usize> = Vec::new();
    let mut ansatz: Vec<Excitation> = Vec::new();
    let mut theta: Vec<f64> = Vec::new();
    let mut steps = Vec::new();
    let mut evaluations = 0;
    let mut energy = energy_and_gradient(h, &ansatz, reference, &theta).0;
    let mut norm = f64::INFINITY;
    let mut converged = false;
    while ansatz.len() < cfg.max_operators {
        let psi = prepare_state(&ansatz, reference, &theta);
        let g = pool_gradients(h, pool, &psi);
        norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm < cfg.gradient_norm_tolerance {
            converged = true;
            break;
        }
        let best = (0..g.len())
            .max_by(|&a, &b| g[a].abs().total_cmp(&g[b].abs()))
            .expect("non-empty pool");
        selected.push(best);
        ansatz.push(pool[best].clone());
        theta.push(0.0);
        let r = bfgs(
            |t| energy_and_gradient(h, &ansatz, reference, t),
            theta.clone(),
            cfg.optimizer_iterations,
            cfg.optimizer_tolerance,
        );
        evaluations += r.evaluations;
        theta = r.x;
        energy = r.value;
        steps.push(AdaptStep {
            operator: pool[best].label.clone(),
            pool_gradient_norm: norm,
            selected_gradient: g[best],
            energy,
        });
    }
    AdaptResult {
        energy,
        converged,
        selected,
        theta,
        steps,
        energy_evaluations: evaluations,
        final_gradient_norm: norm,
    }
}

#[cfg(test)]
mod tests {
    use super::super::driver::{electronic_structure, VqeConfig};
    use super::super::fermion::{build_hamiltonian, Sector};
    use super::super::lanczos::ground_state;
    use super::super::molecules::{by_formula, hydrogen_chain};
    use super::super::uccsd::uccsd_excitations;
    use super::*;

    fn setup(m: &super::super::molecules::Molecule) -> (SparseMatrix, Sector, Vec<f64>, f64) {
        let p = electronic_structure(m, &VqeConfig::default()).unwrap();
        let half = p.ham.n_electrons / 2;
        let sector = Sector::new(p.ham.n_orb, half, half);
        let h = build_hamiltonian(&p.ham, &sector);
        let mut reference = vec![0.0; sector.dim()];
        reference[sector.hartree_fock()] = 1.0;
        let (fci, _) = ground_state(&h, &reference, 300, 1e-12).unwrap();
        (h, sector, reference, fci)
    }

    #[test]
    fn h2_needs_one_double_excitation() {
        let (h, sector, reference, fci) = setup(&by_formula("H2").unwrap());
        let r = adapt_vqe(
            &h,
            &uccsd_excitations(&sector),
            &reference,
            &AdaptConfig::default(),
        );
        assert!(r.converged);
        assert_eq!(r.selected.len(), 1, "{:?}", r.steps);
        assert!(r.steps[0].operator.starts_with("D ab"));
        assert!((r.energy - fci).abs() < 1e-9);
    }

    #[test]
    fn pool_gradients_match_finite_differences() {
        let (h, sector, reference, _) = setup(&hydrogen_chain(4, 1.0));
        let pool = uccsd_excitations(&sector);
        let base = [pool[5].clone(), pool[20].clone()];
        let theta = [0.1, -0.2];
        let psi = prepare_state(&base, &reference, &theta);
        let g = pool_gradients(&h, &pool, &psi);
        for k in [0, 7, pool.len() - 1] {
            let ext: Vec<Excitation> = base.iter().cloned().chain([pool[k].clone()]).collect();
            let e = |x: f64| energy_and_gradient(&h, &ext, &reference, &[0.1, -0.2, x]).0;
            let fd = (e(1e-6) - e(-1e-6)) / 2e-6;
            assert!((g[k] - fd).abs() < 1e-7, "k={k}: {} vs {fd}", g[k]);
        }
    }

    #[test]
    fn stretched_h4_chain_adapt_beats_uccsd_with_fewer_parameters() {
        let (h, sector, reference, fci) = setup(&hydrogen_chain(4, 2.0));
        let pool = uccsd_excitations(&sector);
        let adapt = adapt_vqe(&h, &pool, &reference, &AdaptConfig::default());
        let uccsd = bfgs(
            |t| energy_and_gradient(&h, &pool, &reference, t),
            vec![0.0; pool.len()],
            600,
            1e-7,
        );
        assert!(adapt.energy >= fci - 1e-9, "variational");
        assert!((adapt.energy - fci).abs() < (uccsd.value - fci).abs() + 1e-6);
        assert!(adapt.selected.len() <= 2 * pool.len());
    }
}
