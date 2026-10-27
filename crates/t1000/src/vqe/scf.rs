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
    /// SCF iterations over all strategies tried.
    pub iterations: usize,
    pub converged: bool,
    /// The convergence strategy that produced this result.
    pub strategy: &'static str,
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

/// Starting point for [`rhf_with`].
#[derive(Debug, Clone, Default)]
pub struct ScfOptions {
    /// MO coefficients (`n_ao × n_ao`, row-major) to start from, typically the converged orbitals
    /// of a nearby geometry. `None` starts from the core-Hamiltonian guess.
    pub guess: Option<Vec<f64>>,
}

/// One convergence strategy: a virtual-orbital level shift, density damping, and the iteration
/// from which DIIS extrapolation is used.
#[derive(Debug, Clone, Copy)]
struct Strategy {
    name: &'static str,
    level_shift: f64,
    damping: f64,
    diis_start: usize,
}

/// Tried in order until one converges. Plain DIIS is fastest near equilibrium; at stretched bonds
/// the HOMO-LUMO gap closes and DIIS oscillates, which the shift and damping suppress.
const STRATEGIES: [Strategy; 3] = [
    Strategy {
        name: "diis",
        level_shift: 0.0,
        damping: 0.0,
        diis_start: 2,
    },
    Strategy {
        name: "shift+damp",
        level_shift: 0.5,
        damping: 0.5,
        diis_start: 25,
    },
    Strategy {
        name: "strong shift+damp",
        level_shift: 1.5,
        damping: 0.7,
        diis_start: 60,
    },
];

/// Restricted Hartree-Fock for `n_electrons` (must be even) in the AO basis of `ints`.
pub fn rhf(ints: &AoIntegrals, n_electrons: usize) -> Result<RhfResult, String> {
    rhf_with(ints, n_electrons, &ScfOptions::default())
}

/// [`rhf`] from an optional starting guess, falling back to level-shifted and damped iterations
/// when plain DIIS does not converge. Returns the last attempt (unconverged) if none converges.
pub fn rhf_with(
    ints: &AoIntegrals,
    n_electrons: usize,
    opts: &ScfOptions,
) -> Result<RhfResult, String> {
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
    let cold = density(&diagonalise_fock(&ints.h, &x, n)?.1, n, n_occ);
    let Some(guess) = &opts.guess else {
        return staged(ints, &x, n_occ, cold);
    };
    if guess.len() != n * n {
        return Err("SCF guess has the wrong dimension".into());
    }
    // Following orbitals along a scan can carry the solution across a level crossing onto a
    // higher-energy SCF state (the H₄ square is the classic case), so the cold start is always
    // tried as well and the lower converged energy is kept.
    let seeded = staged(ints, &x, n_occ, density(guess, n, n_occ))?;
    let fresh = staged(ints, &x, n_occ, cold)?;
    Ok(match (seeded.converged, fresh.converged) {
        (true, true) if fresh.energy < seeded.energy - 1e-8 => fresh,
        (false, true) => fresh,
        _ => seeded,
    })
}

/// Runs the strategies in order from density `p0` until one converges.
fn staged(ints: &AoIntegrals, x: &[f64], n_occ: usize, p0: Vec<f64>) -> Result<RhfResult, String> {
    let mut last = None;
    let mut total_iterations = 0;
    for strategy in STRATEGIES {
        let mut r = scf_loop(ints, x, n_occ, p0.clone(), strategy)?;
        total_iterations += r.iterations;
        r.iterations = total_iterations;
        if r.converged {
            return Ok(r);
        }
        last = Some(r);
    }
    Ok(last.expect("at least one strategy"))
}

/// `F + b (S - ½ S P S)`: raises the virtual orbitals by `b` without moving the occupied space.
fn level_shifted(f: &[f64], s: &[f64], p: &[f64], n: usize, b: f64) -> Vec<f64> {
    if b == 0.0 {
        return f.to_vec();
    }
    let sps = mat_mul(&mat_mul(s, p, n), s, n);
    f.iter()
        .zip(s.iter().zip(&sps))
        .map(|(fv, (sv, spsv))| fv + b * (sv - 0.5 * spsv))
        .collect()
}

fn scf_loop(
    ints: &AoIntegrals,
    x: &[f64],
    n_occ: usize,
    mut p: Vec<f64>,
    st: Strategy,
) -> Result<RhfResult, String> {
    let n = ints.n;
    let mut history: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    let (mut energy, mut converged, mut iterations) = (0.0, false, 0);
    let mut f = fock(ints, &p);
    for it in 1..=MAX_ITERATIONS {
        iterations = it;
        f = fock(ints, &p);
        let e_new = electronic_energy(&ints.h, &f, &p) + ints.nuclear_repulsion;
        let err = diis_error(&f, &p, &ints.s, x, n);
        let rms = (err.iter().map(|v| v * v).sum::<f64>() / err.len() as f64).sqrt();
        let delta = (e_new - energy).abs();
        energy = e_new;
        if it > 1 && delta < ENERGY_TOLERANCE && rms < ERROR_TOLERANCE {
            converged = true;
            break;
        }
        history.push((f.clone(), err));
        if history.len() > DIIS_SIZE {
            history.remove(0);
        }
        let use_diis = it >= st.diis_start && history.len() >= 2;
        let f_next = if use_diis {
            diis_extrapolate(&history).unwrap_or_else(|| f.clone())
        } else {
            f.clone()
        };
        let (_, c) = diagonalise_fock(
            &level_shifted(&f_next, &ints.s, &p, n, st.level_shift),
            x,
            n,
        )?;
        let p_new = density(&c, n, n_occ);
        p = if use_diis || st.damping == 0.0 {
            p_new
        } else {
            p_new
                .iter()
                .zip(&p)
                .map(|(a, b)| (1.0 - st.damping) * a + st.damping * b)
                .collect()
        };
    }
    // Orbitals and energies of the unshifted Fock matrix of the final density.
    let (eps, c) = diagonalise_fock(&f, x, n)?;
    Ok(RhfResult {
        energy,
        orbital_energies: eps,
        coefficients: c,
        n_occupied: n_occ,
        iterations,
        converged,
        strategy: st.name,
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

    fn lih(r: f64) -> AoIntegrals {
        let atoms = [
            Atom::angstrom(3, [0.0; 3]),
            Atom::angstrom(1, [0.0, 0.0, r]),
        ];
        compute_ao_integrals(&sto3g_basis(&atoms).unwrap(), &atoms)
    }

    #[test]
    fn stretched_lih_converges_from_a_cold_start_through_the_fallbacks() {
        // Plain DIIS from the core guess failed here before the shifted and damped fallbacks.
        let r = rhf(&lih(4.0), 4).unwrap();
        assert!(
            r.converged,
            "strategy {} after {}",
            r.strategy, r.iterations
        );
        // The same state is reached by following the bond out from equilibrium.
        let mut guess = rhf(&lih(1.6), 4).unwrap().coefficients;
        for d in [2.0, 2.5, 3.0, 3.5, 4.0] {
            guess = rhf_with(&lih(d), 4, &ScfOptions { guess: Some(guess) })
                .unwrap()
                .coefficients;
        }
        let followed = rhf_with(&lih(4.0), 4, &ScfOptions { guess: Some(guess) }).unwrap();
        assert!(followed.converged && followed.iterations <= 4);
        assert!(
            (r.energy - followed.energy).abs() < 1e-7,
            "{} vs {}",
            r.energy,
            followed.energy
        );
    }

    #[test]
    fn a_seed_from_across_a_level_crossing_does_not_trap_the_scf() {
        // H₄ rectangle a = 1.0 Å: the occupied orbitals swap character at the square (b = a).
        let rect = |b: f64| {
            let atoms: Vec<Atom> = [
                [-0.5, -b / 2.0],
                [0.5, -b / 2.0],
                [0.5, b / 2.0],
                [-0.5, b / 2.0],
            ]
            .iter()
            .map(|&[x, y]| Atom::angstrom(1, [x, y, 0.0]))
            .collect();
            compute_ao_integrals(&sto3g_basis(&atoms).unwrap(), &atoms)
        };
        let seed = rhf(&rect(0.9), 4).unwrap().coefficients;
        let ints = rect(1.2);
        let x = inverse_sqrt(&ints.s, ints.n).unwrap();
        let trapped = staged(&ints, &x, 2, density(&seed, ints.n, 2)).unwrap();
        let chosen = rhf_with(&ints, 4, &ScfOptions { guess: Some(seed) }).unwrap();
        let cold = rhf(&ints, 4).unwrap();
        assert!(chosen.converged && cold.converged);
        assert!((chosen.energy - cold.energy).abs() < 1e-8);
        assert!(
            // The seed alone lands on a higher SCF state; that is the trap this guards against.
            trapped.converged && trapped.energy > chosen.energy + 1e-3,
            "{} vs {}",
            chosen.energy,
            trapped.energy
        );
    }

    #[test]
    fn level_shift_leaves_the_occupied_space_alone() {
        // At a converged density F and F + b(S - SPS/2) have the same occupied orbitals.
        let ints = lih(1.6);
        let r = rhf(&ints, 4).unwrap();
        let n = ints.n;
        let p = density(&r.coefficients, n, 2);
        let x = inverse_sqrt(&ints.s, n).unwrap();
        let f = fock(&ints, &p);
        let (eps0, _) = diagonalise_fock(&f, &x, n).unwrap();
        let (eps1, c1) = diagonalise_fock(&level_shifted(&f, &ints.s, &p, n, 0.8), &x, n).unwrap();
        assert!((eps0[0] - eps1[0]).abs() < 1e-8 && (eps0[1] - eps1[1]).abs() < 1e-8);
        assert!((eps1[2] - eps0[2] - 0.8).abs() < 1e-8);
        let p1 = density(&c1, n, 2);
        assert!(p.iter().zip(&p1).all(|(a, b)| (a - b).abs() < 1e-8));
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
