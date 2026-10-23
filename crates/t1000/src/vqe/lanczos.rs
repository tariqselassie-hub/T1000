// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Lanczos ground-state solver for the sparse determinant-basis Hamiltonian: the exact
//! (full configuration interaction) reference the variational energy is judged against.

use super::fermion::SparseMatrix;
use super::linalg::symmetric_eigh;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Lowest eigenvalue of tridiagonal `(alpha, beta)` and its eigenvector.
fn lowest_ritz_pair(alpha: &[f64], beta: &[f64]) -> Result<(f64, Vec<f64>), String> {
    let k = alpha.len();
    let mut t = vec![0.0; k * k];
    for i in 0..k {
        t[i * k + i] = alpha[i];
        if i + 1 < k {
            t[i * k + i + 1] = beta[i];
            t[(i + 1) * k + i] = beta[i];
        }
    }
    let (vals, vecs) = symmetric_eigh(&t, k)?;
    Ok((vals[0], (0..k).map(|r| vecs[r * k]).collect()))
}

/// Lowest eigenvalue and eigenvector of `h`, starting the Krylov space at `start`.
///
/// Full reorthogonalisation keeps the basis orthonormal; the iteration stops when the lowest
/// Ritz value moves by less than `tol` between checks or the Krylov space is exhausted.
pub fn ground_state(
    h: &SparseMatrix,
    start: &[f64],
    max_iter: usize,
    tol: f64,
) -> Result<(f64, Vec<f64>), String> {
    let n = h.n;
    let max_iter = max_iter.min(n);
    let norm = dot(start, start).sqrt();
    let mut basis: Vec<Vec<f64>> = vec![start.iter().map(|v| v / norm).collect()];
    let (mut alpha, mut beta) = (Vec::new(), Vec::new());
    let mut w = vec![0.0; n];
    let mut previous = f64::INFINITY;
    for k in 0..max_iter {
        h.apply(&basis[k], &mut w);
        alpha.push(dot(&w, &basis[k]));
        for v in &basis {
            let c = dot(&w, v);
            w.iter_mut().zip(v).for_each(|(wi, vi)| *wi -= c * vi);
        }
        // Second pass for numerical orthogonality.
        for v in &basis {
            let c = dot(&w, v);
            w.iter_mut().zip(v).for_each(|(wi, vi)| *wi -= c * vi);
        }
        let b = dot(&w, &w).sqrt();
        let exhausted = b < 1e-11 || k + 1 == max_iter;
        if exhausted || (k + 1) % 8 == 0 {
            let (value, _) = lowest_ritz_pair(&alpha, &beta)?;
            if exhausted || (previous - value).abs() < tol {
                return ritz_vector(&basis, &alpha, &beta);
            }
            previous = value;
        }
        beta.push(b);
        basis.push(w.iter().map(|v| v / b).collect());
    }
    ritz_vector(&basis, &alpha, &beta)
}

/// The lowest Ritz value and the corresponding vector in the original basis.
fn ritz_vector(basis: &[Vec<f64>], alpha: &[f64], beta: &[f64]) -> Result<(f64, Vec<f64>), String> {
    let (value, y) = lowest_ritz_pair(alpha, &beta[..alpha.len() - 1])?;
    let mut x = vec![0.0; basis[0].len()];
    for (yi, v) in y.iter().zip(basis) {
        x.iter_mut().zip(v).for_each(|(xi, vi)| *xi += yi * vi);
    }
    let norm = dot(&x, &x).sqrt();
    x.iter_mut().for_each(|v| *v /= norm);
    Ok((value, x))
}

#[cfg(test)]
mod tests {
    use super::super::fermion::{build_hamiltonian, Sector};
    use super::super::mo::ActiveHamiltonian;
    use super::*;

    /// A two-orbital model Hamiltonian with a closed-form ground state.
    fn hubbard_dimer(t: f64, u: f64) -> ActiveHamiltonian {
        // Orthonormal sites with hopping -t and on-site repulsion U: (pp|pp) = U.
        let mut g = vec![0.0; 16];
        g[0] = u;
        g[15] = u;
        ActiveHamiltonian {
            n_orb: 2,
            n_electrons: 2,
            constant: 0.0,
            h: vec![0.0, -t, -t, 0.0],
            g,
        }
    }

    #[test]
    fn lanczos_finds_the_hubbard_dimer_ground_energy() {
        let (t, u) = (1.0, 4.0);
        let sector = Sector::new(2, 1, 1);
        let h = build_hamiltonian(&hubbard_dimer(t, u), &sector);
        let mut start = vec![0.0; sector.dim()];
        start[sector.hartree_fock()] = 1.0;
        let (e, v) = ground_state(&h, &start, 50, 1e-12).unwrap();
        // E0 = U/2 - sqrt((U/2)^2 + 4 t^2) for the half-filled two-site Hubbard model.
        let exact = u / 2.0 - ((u / 2.0).powi(2) + 4.0 * t * t).sqrt();
        assert!((e - exact).abs() < 1e-10, "{e} vs {exact}");
        let mut hv = vec![0.0; v.len()];
        h.apply(&v, &mut hv);
        let resid: f64 = hv
            .iter()
            .zip(&v)
            .map(|(a, b)| (a - e * b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(resid < 1e-8, "{resid}");
    }

    #[test]
    fn full_space_lanczos_agrees_with_dense_diagonalisation() {
        let sector = Sector::new(3, 1, 1);
        let mut g = vec![0.0; 81];
        for p in 0..3 {
            for q in 0..3 {
                g[((p * 3 + p) * 3 + q) * 3 + q] = 0.3 + 0.1 * (p + q) as f64;
            }
        }
        let ham = ActiveHamiltonian {
            n_orb: 3,
            n_electrons: 2,
            constant: 0.5,
            h: vec![-1.0, 0.2, 0.1, 0.2, -0.4, 0.3, 0.1, 0.3, 0.2],
            g,
        };
        let h = build_hamiltonian(&ham, &sector);
        let n = sector.dim();
        let dense: Vec<f64> = (0..n * n).map(|k| h.get(k / n, k % n)).collect();
        let (vals, _) = symmetric_eigh(&dense, n).unwrap();
        let mut start = vec![0.0; n];
        start[sector.hartree_fock()] = 1.0;
        start[1] += 0.3;
        let (e, _) = ground_state(&h, &start, 100, 1e-13).unwrap();
        assert!((e - vals[0]).abs() < 1e-10, "{e} vs {}", vals[0]);
    }
}
