// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Closed-shell restricted Hartree-Fock with Pulay DIIS convergence acceleration.

use super::integrals::AoIntegrals;
use super::linalg::{mat_mul, solve_linear, symmetric_eigh, transpose};

const MAX_ITERATIONS: usize = 200;
const DIIS_SIZE: usize = 8;
const ENERGY_TOLERANCE: f64 = 1e-11;
const ERROR_TOLERANCE: f64 = 1e-8;

/// A converged (or best-effort) restricted Hartree-Fock solution.
#[derive(Debug, Clone)]
pub struct RhfResult {
    /// Total energy including the nuclear repulsion, in hartree.
    pub energy: f64,
    /// Orbital energies, ascending.
    pub orbital_energies: Vec<f64>,
    /// MO coefficients, row-major `n_ao × n_mo`; column `p` is orbital `p`.
    pub coefficients: Vec<f64>,
    pub n_occupied: usize,
    pub iterations: usize,
    pub converged: bool,
}

/// `S^(-1/2)`.
fn inverse_sqrt(s: &[f64], n: usize) -> Result<Vec<f64>, String> {
    let (vals, vecs) = symmetric_eigh(s, n)?;
    if vals[0] < 1e-8 {
        return Err("overlap matrix is (near) singular".into());
    }
    let mut scaled = vecs.clone();
    for r in 0..n {
        for k in 0..n {
            scaled[r * n + k] /= vals[k].sqrt();
        }
    }
    Ok(mat_mul(&scaled, &transpose(&vecs, n), n))
}

/// Diagonalises `F` in the orthogonal basis `X` and returns `(ε, C = X C')`.
fn diagonalise_fock(f: &[f64], x: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    let f_orth = mat_mul(&mat_mul(&transpose(x, n), f, n), x, n);
    let (eps, c_orth) = symmetric_eigh(&f_orth, n)?;
    Ok((eps, mat_mul(x, &c_orth, n)))
}

/// Closed-shell density `P = 2 Σ_occ c cᵀ`.
fn density(c: &[f64], n: usize, n_occ: usize) -> Vec<f64> {
    let mut p = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            p[i * n + j] = 2.0 * (0..n_occ).map(|k| c[i * n + k] * c[j * n + k]).sum::<f64>();
        }
    }
    p
}

/// `F = H + Σ_kl P_kl [(ij|kl) - ½ (ik|jl)]`.
fn fock(ints: &AoIntegrals, p: &[f64]) -> Vec<f64> {
    let n = ints.n;
    let mut f = ints.h.clone();
    for i in 0..n {
        for j in 0..n {
            let mut g = 0.0;
            for k in 0..n {
                for l in 0..n {
                    g += p[k * n + l] * (ints.g(i, j, k, l) - 0.5 * ints.g(i, k, j, l));
                }
            }
            f[i * n + j] += g;
        }
    }
    f
}

fn electronic_energy(h: &[f64], f: &[f64], p: &[f64]) -> f64 {
    0.5 * p
        .iter()
        .zip(h.iter().zip(f))
        .map(|(p, (h, f))| p * (h + f))
        .sum::<f64>()
}

/// Commutator error `X^T (F P S - S P F) X`.
fn diis_error(f: &[f64], p: &[f64], s: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    let fps = mat_mul(&mat_mul(f, p, n), s, n);
    let spf = transpose(&fps, n);
    let e: Vec<f64> = fps.iter().zip(&spf).map(|(a, b)| a - b).collect();
    mat_mul(&mat_mul(&transpose(x, n), &e, n), x, n)
}

/// Pulay extrapolation of the stored Fock matrices; `None` if the DIIS system is singular.
fn diis_extrapolate(history: &[(Vec<f64>, Vec<f64>)]) -> Option<Vec<f64>> {
    let m = history.len();
    let dim = m + 1;
    let mut b = vec![0.0; dim * dim];
    for i in 0..m {
        for j in 0..m {
            b[i * dim + j] = history[i]
                .1
                .iter()
                .zip(&history[j].1)
                .map(|(a, c)| a * c)
                .sum::<f64>();
        }
        b[i * dim + m] = -1.0;
        b[m * dim + i] = -1.0;
    }
    let mut rhs = vec![0.0; dim];
    rhs[m] = -1.0;
    let c = solve_linear(&b, &rhs, dim)?;
    let mut f = vec![0.0; history[0].0.len()];
    for (ci, (fi, _)) in c.iter().zip(history) {
        f.iter_mut().zip(fi).for_each(|(acc, v)| *acc += ci * v);
    }
    Some(f)
}

/// Restricted Hartree-Fock for `n_electrons` (must be even) in the AO basis of `ints`.
pub fn rhf(ints: &AoIntegrals, n_electrons: usize) -> Result<RhfResult, String> {
    if !n_electrons.is_multiple_of(2) {
        return Err(format!(
            "RHF needs an even electron count, got {n_electrons}"
        ));
    }
    let (n, n_occ) = (ints.n, n_electrons / 2);
    if n_occ > n {
        return Err("more occupied orbitals than basis functions".into());
    }
    let x = inverse_sqrt(&ints.s, n)?;
    let (mut eps, mut c) = diagonalise_fock(&ints.h, &x, n)?;
    let mut p = density(&c, n, n_occ);
    let mut history: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    let (mut energy, mut converged, mut iterations) = (0.0, false, 0);
    for it in 1..=MAX_ITERATIONS {
        iterations = it;
        let f = fock(ints, &p);
        let e_new = electronic_energy(&ints.h, &f, &p) + ints.nuclear_repulsion;
        let err = diis_error(&f, &p, &ints.s, &x, n);
        let rms = (err.iter().map(|v| v * v).sum::<f64>() / err.len() as f64).sqrt();
        history.push((f.clone(), err));
        if history.len() > DIIS_SIZE {
            history.remove(0);
        }
        let f_used = if history.len() >= 2 {
            diis_extrapolate(&history).unwrap_or(f)
        } else {
            f
        };
        (eps, c) = diagonalise_fock(&f_used, &x, n)?;
        p = density(&c, n, n_occ);
        let delta = (e_new - energy).abs();
        energy = e_new;
        if it > 1 && delta < ENERGY_TOLERANCE && rms < ERROR_TOLERANCE {
            converged = true;
            break;
        }
    }
    Ok(RhfResult {
        energy,
        orbital_energies: eps,
        coefficients: c,
        n_occupied: n_occ,
        iterations,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::super::basis::{sto3g_basis, Atom};
    use super::super::integrals::compute_ao_integrals;
    use super::*;

    fn solve(atoms: &[Atom], electrons: usize) -> RhfResult {
        let basis = sto3g_basis(atoms).unwrap();
        let ints = compute_ao_integrals(&basis, atoms);
        rhf(&ints, electrons).unwrap()
    }

    #[test]
    fn h2_reproduces_the_szabo_ostlund_hartree_fock_solution() {
        let atoms = [
            Atom {
                z: 1,
                pos: [0.0; 3],
            },
            Atom {
                z: 1,
                pos: [0.0, 0.0, 1.4],
            },
        ];
        let r = solve(&atoms, 2);
        assert!(r.converged);
        assert!((r.energy + 1.1167).abs() < 1e-4, "E = {}", r.energy);
        assert!((r.orbital_energies[0] + 0.5782).abs() < 2e-4);
        assert!((r.orbital_energies[1] - 0.6703).abs() < 2e-4);
    }

    #[test]
    fn helium_atom_energy_matches_the_sto3g_value() {
        let r = solve(
            &[Atom {
                z: 2,
                pos: [0.0; 3],
            }],
            2,
        );
        assert!((r.energy + 2.8077840).abs() < 2e-5, "E = {}", r.energy);
    }

    #[test]
    fn hf_energy_is_invariant_under_rigid_motion_of_water() {
        let a = [
            Atom::angstrom(8, [0.0, 0.0, 0.1173]),
            Atom::angstrom(1, [0.0, 0.7572, -0.4692]),
            Atom::angstrom(1, [0.0, -0.7572, -0.4692]),
        ];
        let b = [
            Atom::angstrom(8, [0.3, -0.2, 0.9]),
            Atom::angstrom(1, [0.3 + 0.7572, -0.2 + 0.5865, 0.9 + 0.0]),
            Atom::angstrom(1, [0.3 - 0.7572, -0.2 + 0.5865, 0.9 + 0.0]),
        ];
        let (ea, eb) = (solve(&a, 10), solve(&b, 10));
        assert!(ea.converged && eb.converged);
        assert!(
            (ea.energy - eb.energy).abs() < 1e-9,
            "{} vs {}",
            ea.energy,
            eb.energy
        );
    }

    #[test]
    fn odd_electron_counts_are_rejected() {
        let atoms = [Atom {
            z: 3,
            pos: [0.0; 3],
        }];
        let basis = sto3g_basis(&atoms).unwrap();
        let ints = compute_ao_integrals(&basis, &atoms);
        assert!(rhf(&ints, 3).is_err());
    }
}
