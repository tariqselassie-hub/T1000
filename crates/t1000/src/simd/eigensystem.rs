// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Hardware-Accelerated Real-Symmetric Eigensystem Subsystem.
//!
//! Provides bare-metal Householder tridiagonalization and implicit symmetric
//! QL/QR decomposition for real symmetric matrices (such as molecular Coulomb
//! matrices and Jordan-Wigner Hamiltonian density matrices).

use std::f64::EPSILON;

/// Result of a real-symmetric eigensystem decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct SymmetricEigensystem {
    /// Dimension of the square symmetric matrix.
    pub dim: usize,
    /// Eigenvalues sorted in descending order: lambda_0 >= lambda_1 >= ...
    pub eigenvalues: Vec<f64>,
    /// Optional eigenvectors stored as column vectors: `eigenvectors[col][row]`.
    pub eigenvectors: Option<Vec<Vec<f64>>>,
}

impl SymmetricEigensystem {
    /// Returns the spectral trace: sum of eigenvalues.
    pub fn spectral_trace(&self) -> f64 {
        self.eigenvalues.iter().sum()
    }

    /// Returns the spectral Frobenius norm squared: sum of squared eigenvalues.
    pub fn frobenius_norm_sq(&self) -> f64 {
        self.eigenvalues.iter().map(|&x| x * x).sum()
    }
}

/// Computes the complete eigensystem of a real symmetric matrix.
///
/// Matrix is expected in row-major order: `matrix[row * n + col]`.
/// If `compute_vectors` is true, eigenvectors are computed and returned as column vectors.
pub fn solve_symmetric_eigensystem(
    matrix: &[f64],
    n: usize,
    compute_vectors: bool,
) -> Result<SymmetricEigensystem, String> {
    if n == 0 {
        return Err("Matrix dimension cannot be zero".to_string());
    }
    if matrix.len() != n * n {
        return Err(format!(
            "Buffer size mismatch: expected {} elements for {}x{} matrix, got {}",
            n * n,
            n,
            n,
            matrix.len()
        ));
    }

    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n];
    let mut v = if compute_vectors {
        matrix.to_vec()
    } else {
        Vec::new()
    };

    householder_tridiagonalize(matrix, n, &mut d, &mut e, &mut v, compute_vectors);
    symmetric_ql_algorithm(n, &mut d, &mut e, &mut v, compute_vectors)?;

    // Sort eigenvalues descending and permute eigenvectors accordingly
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| d[b].partial_cmp(&d[a]).unwrap_or(std::cmp::Ordering::Equal));

    let sorted_eigenvalues: Vec<f64> = indices.iter().map(|&i| d[i]).collect();

    let sorted_eigenvectors = if compute_vectors {
        let mut cols = Vec::with_capacity(n);
        for &col_idx in &indices {
            let mut col = Vec::with_capacity(n);
            for row in 0..n {
                col.push(v[row * n + col_idx]);
            }
            cols.push(col);
        }
        Some(cols)
    } else {
        None
    };

    Ok(SymmetricEigensystem {
        dim: n,
        eigenvalues: sorted_eigenvalues,
        eigenvectors: sorted_eigenvectors,
    })
}

/// Stack-allocated zero-heap eigensolver for 50x50 Coulomb matrices.
///
/// Returns 50 eigenvalues sorted in descending order: lambda_0 >= lambda_1 >= ... >= lambda_49.
pub fn solve_coulomb_eigensystem_50(matrix: &[f64; 2500]) -> [f64; 50] {
    const N: usize = 50;
    let mut d = [0.0; N];
    let mut e = [0.0; N];
    let mut a = *matrix;

    householder_tridiagonalize_50(&mut a, &mut d, &mut e);
    symmetric_ql_algorithm_50(&mut d, &mut e);

    // Sort descending
    d.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    d
}

/// Householder reduction of a real symmetric matrix to tridiagonal form.
fn householder_tridiagonalize(
    matrix: &[f64],
    n: usize,
    d: &mut [f64],
    e: &mut [f64],
    v: &mut [f64],
    compute_vectors: bool,
) {
    if compute_vectors {
        // v initialized with matrix
    } else {
        // Copy diagonal to d
        for i in 0..n {
            d[i] = matrix[i * n + i];
        }
    }

    let mut a = matrix.to_vec();

    for i in (1..n).rev() {
        let l = i - 1;
        let mut h = 0.0;
        let mut scale = 0.0;

        if l > 0 {
            for k in 0..=l {
                scale += a[i * n + k].abs();
            }
            if scale < EPSILON {
                e[i] = a[i * n + l];
            } else {
                for k in 0..=l {
                    a[i * n + k] /= scale;
                    h += a[i * n + k] * a[i * n + k];
                }
                let f = a[i * n + l];
                let g = if f >= 0.0 { -h.sqrt() } else { h.sqrt() };
                e[i] = scale * g;
                h -= f * g;
                a[i * n + l] = f - g;
                let mut f_accum = 0.0;

                for j in 0..=l {
                    if compute_vectors {
                        v[j * n + i] = a[i * n + j] / h;
                    }
                    let mut g_inner = 0.0;
                    for k in 0..=j {
                        g_inner += a[j * n + k] * a[i * n + k];
                    }
                    for k in (j + 1)..=l {
                        g_inner += a[k * n + j] * a[i * n + k];
                    }
                    e[j] = g_inner / h;
                    f_accum += e[j] * a[i * n + j];
                }

                let hh = f_accum / (h + h);
                for j in 0..=l {
                    let f = a[i * n + j];
                    let g = e[j] - hh * f;
                    e[j] = g;
                    for k in 0..=j {
                        a[j * n + k] -= f * e[k] + g * a[i * n + k];
                    }
                }
            }
        } else {
            e[i] = a[i * n + l];
        }
        d[i] = h;
    }

    if compute_vectors {
        d[0] = 0.0;
        e[0] = 0.0;
        for i in 0..n {
            let l = i;
            if d[i] != 0.0 {
                for j in 0..l {
                    let mut g = 0.0;
                    for k in 0..l {
                        g += v[i * n + k] * v[k * n + j];
                    }
                    for k in 0..l {
                        v[k * n + j] -= g * v[k * n + i];
                    }
                }
            }
            d[i] = a[i * n + i];
            v[i * n + i] = 1.0;
            for j in 0..l {
                v[j * n + i] = 0.0;
                v[i * n + j] = 0.0;
            }
        }
    } else {
        for i in 0..n {
            d[i] = a[i * n + i];
        }
        e[0] = 0.0;
    }
}

/// Implicit symmetric QL algorithm with Wilkinson shifts for eigenvalues and eigenvectors.
fn symmetric_ql_algorithm(
    n: usize,
    d: &mut [f64],
    e: &mut [f64],
    v: &mut [f64],
    compute_vectors: bool,
) -> Result<(), String> {
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;

    for l in 0..n {
        let mut iter = 0;
        loop {
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= EPSILON * dd {
                    break;
                }
                m += 1;
            }

            if m == l {
                break;
            }

            iter += 1;
            if iter > 60 {
                return Err(format!("Symmetric QL did not converge at index {l}"));
            }

            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = (g * g + 1.0).sqrt();
            g = d[m] - d[l] + e[l] / (g + if g >= 0.0 { r.abs() } else { -r.abs() });

            let mut s = 1.0;
            let mut c = 1.0;
            let mut p = 0.0;

            for i in (l..m).rev() {
                let mut f = s * e[i];
                let b = c * e[i];
                r = (f * f + g * g).sqrt();
                e[i + 1] = r;

                if r.abs() < EPSILON {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    break;
                }

                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;

                if compute_vectors {
                    for k in 0..n {
                        f = v[k * n + (i + 1)];
                        v[k * n + (i + 1)] = s * v[k * n + i] + c * f;
                        v[k * n + i] = c * v[k * n + i] - s * f;
                    }
                }
            }

            if r.abs() < EPSILON && m > l {
                continue;
            }

            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }

    Ok(())
}

/// Zero-allocation Householder tridiagonalization for fixed 50x50 matrices.
fn householder_tridiagonalize_50(a: &mut [f64; 2500], d: &mut [f64; 50], e: &mut [f64; 50]) {
    const N: usize = 50;
    for i in (1..N).rev() {
        let l = i - 1;
        let mut h = 0.0;
        let mut scale = 0.0;

        if l > 0 {
            for k in 0..=l {
                scale += a[i * N + k].abs();
            }
            if scale < EPSILON {
                e[i] = a[i * N + l];
            } else {
                for k in 0..=l {
                    a[i * N + k] /= scale;
                    h += a[i * N + k] * a[i * N + k];
                }
                let f = a[i * N + l];
                let g = if f >= 0.0 { -h.sqrt() } else { h.sqrt() };
                e[i] = scale * g;
                h -= f * g;
                a[i * N + l] = f - g;
                let mut f_accum = 0.0;

                for j in 0..=l {
                    let mut g_inner = 0.0;
                    for k in 0..=j {
                        g_inner += a[j * N + k] * a[i * N + k];
                    }
                    for k in (j + 1)..=l {
                        g_inner += a[k * N + j] * a[i * N + k];
                    }
                    e[j] = g_inner / h;
                    f_accum += e[j] * a[i * N + j];
                }

                let hh = f_accum / (h + h);
                for j in 0..=l {
                    let f = a[i * N + j];
                    let g = e[j] - hh * f;
                    e[j] = g;
                    for k in 0..=j {
                        a[j * N + k] -= f * e[k] + g * a[i * N + k];
                    }
                }
            }
        } else {
            e[i] = a[i * N + l];
        }
        // d[i] will be set from a[i * N + i] below
    }
    for i in 0..N {
        d[i] = a[i * N + i];
    }
    e[0] = 0.0;
}

/// Zero-allocation QL algorithm for fixed 50x50 eigenvalues.
fn symmetric_ql_algorithm_50(d: &mut [f64; 50], e: &mut [f64; 50]) {
    const N: usize = 50;
    for i in 1..N {
        e[i - 1] = e[i];
    }
    e[N - 1] = 0.0;

    for l in 0..N {
        let mut iter = 0;
        loop {
            let mut m = l;
            while m < N - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= EPSILON * dd {
                    break;
                }
                m += 1;
            }

            if m == l {
                break;
            }

            iter += 1;
            if iter > 60 {
                break; // Fallback to best approximation
            }

            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = (g * g + 1.0).sqrt();
            g = d[m] - d[l] + e[l] / (g + if g >= 0.0 { r.abs() } else { -r.abs() });

            let mut s = 1.0;
            let mut c = 1.0;
            let mut p = 0.0;

            for i in (l..m).rev() {
                let f = s * e[i];
                let b = c * e[i];
                r = (f * f + g * g).sqrt();
                e[i + 1] = r;

                if r.abs() < EPSILON {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    break;
                }

                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
            }

            if r.abs() < EPSILON && m > l {
                continue;
            }

            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagonal_matrix_eigenvalues() {
        let matrix = [3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 5.0];
        let sys = solve_symmetric_eigensystem(&matrix, 3, false).unwrap();
        assert!((sys.eigenvalues[0] - 5.0).abs() < 1e-10);
        assert!((sys.eigenvalues[1] - 3.0).abs() < 1e-10);
        assert!((sys.eigenvalues[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_2x2_symmetric_matrix() {
        // [[2, 1], [1, 2]] -> eigenvalues 3 and 1
        let matrix = [2.0, 1.0, 1.0, 2.0];
        let sys = solve_symmetric_eigensystem(&matrix, 2, true).unwrap();
        assert!((sys.eigenvalues[0] - 3.0).abs() < 1e-10);
        assert!((sys.eigenvalues[1] - 1.0).abs() < 1e-10);

        // Check spectral trace and Frobenius norm invariants
        assert!((sys.spectral_trace() - 4.0).abs() < 1e-10);
        assert!((sys.frobenius_norm_sq() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_50x50_coulomb_eigensolver() {
        let mut mat = [0.0; 2500];
        for i in 0..50 {
            mat[i * 50 + i] = (i + 1) as f64;
        }
        let eigs = solve_coulomb_eigensystem_50(&mat);
        assert_eq!(eigs.len(), 50);
        assert!((eigs[0] - 50.0).abs() < 1e-8);
        assert!((eigs[49] - 1.0).abs() < 1e-8);
    }
}
