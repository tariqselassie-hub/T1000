// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Classical correlated baselines in the same active space as the VQE: second-order perturbation
//! theory (MP2), configuration interaction with singles and doubles (CISD) and coupled cluster
//! with singles and doubles (CCSD).
//!
//! CISD and CCSD are evaluated exactly in the determinant basis. For CCSD this is possible because
//! every excitation `T_k` squares to zero and they all commute, so `e^T = Π_k (1 + t_k T_k)` and
//! the similarity-transformed residual `e^{-T} H e^{T} |HF⟩` is a few sparse passes per iteration.
//! The amplitudes use the same singles and doubles as the UCCSD ansatz, so the two are compared
//! on identical excitation manifolds: CCSD is projective and not variational, UCCSD is variational.

use super::fermion::{Sector, SparseMatrix};
use super::lanczos::ground_state;
use super::linalg::solve_linear;
use super::mo::ActiveHamiltonian;
use super::uccsd::Excitation;

/// Canonical orbital energies of the active space: the diagonal of the closed-shell Fock matrix.
fn orbital_energies(ham: &ActiveHamiltonian) -> Vec<f64> {
    let (m, n_occ) = (ham.n_orb, ham.n_electrons / 2);
    (0..m)
        .map(|p| {
            ham.h[p * m + p]
                + (0..n_occ)
                    .map(|i| 2.0 * ham.eri(p, p, i, i) - ham.eri(p, i, i, p))
                    .sum::<f64>()
        })
        .collect()
}

/// Second-order correlation energy `Σ (ia|jb)[2(ia|jb) - (ib|ja)] / (ε_i + ε_j - ε_a - ε_b)`.
///
/// Assumes canonical Hartree-Fock orbitals (a diagonal Fock matrix), which `active_space` gives.
pub fn mp2_correlation(ham: &ActiveHamiltonian) -> f64 {
    let (m, n_occ) = (ham.n_orb, ham.n_electrons / 2);
    let eps = orbital_energies(ham);
    let mut e2 = 0.0;
    for i in 0..n_occ {
        for j in 0..n_occ {
            for a in n_occ..m {
                for b in n_occ..m {
                    let iajb = ham.eri(i, a, j, b);
                    let ibja = ham.eri(i, b, j, a);
                    e2 += iajb * (2.0 * iajb - ibja) / (eps[i] + eps[j] - eps[a] - eps[b]);
                }
            }
        }
    }
    e2
}

/// Determinants of `sector` at most `level`-fold excited from its Hartree-Fock determinant.
pub fn truncated_sector(sector: &Sector, level: u32) -> Sector {
    let hf = sector.dets[sector.hartree_fock()];
    let dets = sector
        .dets
        .iter()
        .copied()
        .filter(|d| (d ^ hf).count_ones() <= 2 * level)
        .collect();
    Sector::from_dets(sector.n_orb, sector.n_alpha, sector.n_beta, dets)
}

/// Lowest eigenvalue of the Hamiltonian projected onto the singles and doubles space (CISD).
pub fn cisd_energy(ham: &ActiveHamiltonian, sector: &Sector) -> Result<f64, String> {
    let sd = truncated_sector(sector, 2);
    let h = super::fermion::build_hamiltonian(ham, &sd);
    let mut start = vec![0.0; sd.dim()];
    start[sd.hartree_fock()] = 1.0;
    ground_state(&h, &start, 300, 1e-11).map(|(e, _)| e)
}

/// Outcome of a CCSD solve. `energy` is the last iterate even when it did not converge.
#[derive(Debug, Clone)]
pub struct CcsdResult {
    pub energy: f64,
    pub converged: bool,
    pub iterations: usize,
    /// Largest amplitude-equation residual (scaled by the diagonal denominator) at exit.
    pub max_residual: f64,
    /// Largest |t| at exit; large values mark a breakdown of the single-reference expansion.
    pub max_amplitude: f64,
}

const CC_MAX_ITER: usize = 300;
const CC_TOL: f64 = 1e-9;
const CC_DIIS: usize = 8;

/// `e^{±T} ψ` with `T = Σ t_k T_k`.
fn exp_t(excitations: &[Excitation], t: &[f64], psi: &mut [f64], sign: f64) {
    for (e, &tk) in excitations.iter().zip(t) {
        e.apply_operator(psi, sign * tk);
    }
}

/// Spin-orbital CCSD over the given excitations, solved by Jacobi updates with DIIS.
pub fn ccsd(h: &SparseMatrix, sector: &Sector, excitations: &[Excitation]) -> CcsdResult {
    let n = sector.dim();
    let hf = sector.hartree_fock();
    let diag = |i: usize| h.get(i, i);
    // The determinant each T_k makes from Hartree-Fock, with its sign, and the denominator.
    let targets: Vec<(usize, f64, f64)> = excitations
        .iter()
        .map(|e| {
            let (mu, s) = e
                .target_of(hf)
                .expect("every excitation acts on Hartree-Fock");
            (mu, s, diag(mu) - diag(hf))
        })
        .collect();
    let mut t = vec![0.0; excitations.len()];
    let mut history: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    let (mut psi, mut hpsi) = (vec![0.0; n], vec![0.0; n]);
    let mut result = CcsdResult {
        energy: f64::NAN,
        converged: false,
        iterations: 0,
        max_residual: f64::INFINITY,
        max_amplitude: 0.0,
    };
    let mut previous = f64::INFINITY;
    for it in 1..=CC_MAX_ITER {
        psi.iter_mut().for_each(|v| *v = 0.0);
        psi[hf] = 1.0;
        exp_t(excitations, &t, &mut psi, 1.0);
        h.apply(&psi, &mut hpsi);
        exp_t(excitations, &t, &mut hpsi, -1.0);
        let energy = hpsi[hf];
        // Scaled residual r_k = s_k R[μ_k] / D_k: the Jacobi step that removes it at linear order.
        let r: Vec<f64> = targets.iter().map(|&(mu, s, d)| s * hpsi[mu] / d).collect();
        let max_r = r.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        let max_t = t.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        result = CcsdResult {
            energy,
            converged: false,
            iterations: it,
            max_residual: max_r,
            max_amplitude: max_t,
        };
        if !energy.is_finite() || max_t > 50.0 {
            break;
        }
        if max_r < CC_TOL && (energy - previous).abs() < CC_TOL {
            result.converged = true;
            break;
        }
        previous = energy;
        let t_next: Vec<f64> = t.iter().zip(&r).map(|(tk, rk)| tk - rk).collect();
        history.push((t_next.clone(), r));
        if history.len() > CC_DIIS {
            history.remove(0);
        }
        t = if history.len() >= 3 {
            diis(&history).unwrap_or(t_next)
        } else {
            t_next
        };
    }
    result
}

/// Pulay extrapolation of amplitude vectors from their residuals.
fn diis(history: &[(Vec<f64>, Vec<f64>)]) -> Option<Vec<f64>> {
    let m = history.len();
    let dim = m + 1;
    let mut b = vec![0.0; dim * dim];
    for i in 0..m {
        for j in 0..m {
            b[i * dim + j] = history[i]
                .1
                .iter()
                .zip(&history[j].1)
                .map(|(x, y)| x * y)
                .sum();
        }
        b[i * dim + m] = -1.0;
        b[m * dim + i] = -1.0;
    }
    let mut rhs = vec![0.0; dim];
    rhs[m] = -1.0;
    let c = solve_linear(&b, &rhs, dim)?;
    let mut out = vec![0.0; history[0].0.len()];
    for (ci, (ti, _)) in c.iter().zip(history) {
        out.iter_mut().zip(ti).for_each(|(o, v)| *o += ci * v);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::super::driver::{electronic_structure, VqeConfig};
    use super::super::fermion::build_hamiltonian;
    use super::super::molecules::{by_formula, hydrogen_chain};
    use super::super::uccsd::uccsd_excitations;
    use super::*;

    struct Case {
        ham: ActiveHamiltonian,
        sector: Sector,
        h: SparseMatrix,
        hf: f64,
        fci: f64,
    }

    fn case(molecule: &super::super::molecules::Molecule) -> Case {
        let p = electronic_structure(molecule, &VqeConfig::default()).unwrap();
        let half = p.ham.n_electrons / 2;
        let sector = Sector::new(p.ham.n_orb, half, half);
        let h = build_hamiltonian(&p.ham, &sector);
        let mut start = vec![0.0; sector.dim()];
        start[sector.hartree_fock()] = 1.0;
        let (fci, _) = ground_state(&h, &start, 300, 1e-12).unwrap();
        let hf = h.get(sector.hartree_fock(), sector.hartree_fock());
        Case {
            ham: p.ham,
            sector,
            h,
            hf,
            fci,
        }
    }

    #[test]
    fn two_electron_cisd_and_ccsd_are_exact() {
        // With two electrons, singles and doubles span the whole space.
        let c = case(&by_formula("H2").unwrap());
        let cisd = cisd_energy(&c.ham, &c.sector).unwrap();
        let cc = ccsd(&c.h, &c.sector, &uccsd_excitations(&c.sector));
        assert!(cc.converged);
        assert!((cisd - c.fci).abs() < 1e-10, "{cisd} vs {}", c.fci);
        assert!(
            (cc.energy - c.fci).abs() < 1e-9,
            "{} vs {}",
            cc.energy,
            c.fci
        );
    }

    #[test]
    fn h2_mp2_matches_the_closed_form_two_orbital_expression() {
        // Two orbitals: E2 = (12|12)^2 / (2 ε_1 - 2 ε_2).
        let c = case(&by_formula("H2").unwrap());
        let eps = orbital_energies(&c.ham);
        let k = c.ham.eri(0, 1, 0, 1);
        let want = k * k / (2.0 * eps[0] - 2.0 * eps[1]);
        assert!((mp2_correlation(&c.ham) - want).abs() < 1e-12);
        // Hartree-Fock orbital energy of H₂ at 1.4 bohr from Szabo & Ostlund is -0.5782.
        assert!(eps[0] < -0.5 && eps[1] > 0.5);
    }

    #[test]
    fn h4_chain_baselines_bracket_the_exact_energy_near_equilibrium() {
        let c = case(&hydrogen_chain(4, 0.9));
        let cisd = cisd_energy(&c.ham, &c.sector).unwrap();
        let cc = ccsd(&c.h, &c.sector, &uccsd_excitations(&c.sector));
        let mp2 = c.hf + mp2_correlation(&c.ham);
        assert!(cc.converged, "{cc:?}");
        // CISD is variational and misses the disconnected quadruples; CCSD is close to exact.
        assert!(cisd > c.fci && cisd < c.hf);
        assert!(
            (cc.energy - c.fci).abs() < 2e-3,
            "CCSD {} FCI {}",
            cc.energy,
            c.fci
        );
        assert!((cc.energy - c.fci).abs() < (cisd - c.fci).abs());
        assert!(mp2 < c.hf);
    }
}
