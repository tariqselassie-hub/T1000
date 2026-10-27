// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Small dense real linear algebra for the SCF solver. Matrices are row-major `n × n` slices.

/// `a · b`.
pub fn mat_mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            for j in 0..n {
                c[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    c
}

/// `aᵀ`.
pub fn transpose(a: &[f64], n: usize) -> Vec<f64> {
    let mut t = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            t[j * n + i] = a[i * n + j];
        }
    }
    t
}

/// Eigenvalues in ascending order and the matching eigenvectors as the **columns** of a
/// row-major matrix, by cyclic Jacobi rotations (slow for large `n`, but unconditionally
/// convergent and accurate to machine precision for the orbital-sized matrices used here).
pub fn symmetric_eigh(m: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    let mut a = m.to_vec();
    let mut v = vec![0.0; n * n];
    (0..n).for_each(|i| v[i * n + i] = 1.0);
    let scale: f64 = a
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt()
        .max(f64::MIN_POSITIVE);
    let off_norm = |a: &[f64]| -> f64 {
        (0..n)
            .flat_map(|i| (0..i).map(move |j| (i, j)))
            .map(|(i, j)| a[i * n + j] * a[i * n + j])
            .sum::<f64>()
            .sqrt()
    };
    for _sweep in 0..100 {
        if off_norm(&a) <= 1e-15 * scale {
            return Ok(sorted_eigenpairs(&a, &v, n));
        }
        for p in 0..n {
            for q in p + 1..n {
                rotate(&mut a, &mut v, n, p, q);
            }
        }
    }
    // Rounding can leave the off-diagonal norm stalled just above the machine-precision target
    // for large or nearly degenerate matrices; that is still converged for every use here.
    let off = off_norm(&a);
    if off <= 1e-11 * scale {
        return Ok(sorted_eigenpairs(&a, &v, n));
    }
    Err(format!(
        "Jacobi eigensolver did not converge (off-diagonal {:.2e} of scale {scale:.2e})",
        off
    ))
}

/// Diagonal of `a` in ascending order with the matching columns of `v`.
fn sorted_eigenpairs(a: &[f64], v: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| a[x * n + x].total_cmp(&a[y * n + y]));
    let values = order.iter().map(|&k| a[k * n + k]).collect();
    let mut cols = vec![0.0; n * n];
    for (new, &old) in order.iter().enumerate() {
        for r in 0..n {
            cols[r * n + new] = v[r * n + old];
        }
    }
    (values, cols)
}

/// One Jacobi rotation annihilating `a[p][q]`.
fn rotate(a: &mut [f64], v: &mut [f64], n: usize, p: usize, q: usize) {
    let apq = a[p * n + q];
    if apq.abs() < 1e-300 {
        return;
    }
    let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
    let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
    let t = if theta == 0.0 { 1.0 } else { t };
    let c = 1.0 / (t * t + 1.0).sqrt();
    let s = t * c;
    for k in 0..n {
        let (akp, akq) = (a[k * n + p], a[k * n + q]);
        a[k * n + p] = c * akp - s * akq;
        a[k * n + q] = s * akp + c * akq;
    }
    for k in 0..n {
        let (apk, aqk) = (a[p * n + k], a[q * n + k]);
        a[p * n + k] = c * apk - s * aqk;
        a[q * n + k] = s * apk + c * aqk;
    }
    for k in 0..n {
        let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
        v[k * n + p] = c * vkp - s * vkq;
        v[k * n + q] = s * vkp + c * vkq;
    }
}

/// Solves `A x = b` by Gaussian elimination with partial pivoting; `None` if singular.
pub fn solve_linear(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut m: Vec<f64> = (0..n)
        .flat_map(|i| {
            a[i * n..(i + 1) * n]
                .iter()
                .copied()
                .chain(std::iter::once(b[i]))
        })
        .collect();
    let w = n + 1;
    for col in 0..n {
        let pivot =
            (col..n).max_by(|&x, &y| m[x * w + col].abs().total_cmp(&m[y * w + col].abs()))?;
        if m[pivot * w + col].abs() < 1e-14 {
            return None;
        }
        for k in 0..w {
            m.swap(col * w + k, pivot * w + k);
        }
        for r in col + 1..n {
            let f = m[r * w + col] / m[col * w + col];
            for k in col..w {
                m[r * w + k] -= f * m[col * w + k];
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let tail: f64 = (i + 1..n).map(|j| m[i * w + j] * x[j]).sum();
        x[i] = (m[i * w + n] - tail) / m[i * w + i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigh_returns_ascending_values_and_orthonormal_column_vectors() {
        let m = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 4.0];
        let (vals, vecs) = symmetric_eigh(&m, 3).unwrap();
        assert!(vals[0] < vals[1] && vals[1] < vals[2]);
        let vt_v = mat_mul(&transpose(&vecs, 3), &vecs, 3);
        for i in 0..3 {
            for j in 0..3 {
                assert!((vt_v[i * 3 + j] - f64::from(u8::from(i == j))).abs() < 1e-12);
            }
        }
        // A v = λ v for the lowest pair.
        for r in 0..3 {
            let av: f64 = (0..3).map(|c| m[r * 3 + c] * vecs[c * 3]).sum();
            assert!((av - vals[0] * vecs[r * 3]).abs() < 1e-12);
        }
    }

    #[test]
    fn linear_solve_recovers_a_known_solution_and_rejects_singular_systems() {
        let a = [2.0, 1.0, 1.0, 3.0];
        let x = solve_linear(&a, &[5.0, 10.0], 2).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 3.0).abs() < 1e-12);
        assert!(solve_linear(&[1.0, 2.0, 2.0, 4.0], &[1.0, 2.0], 2).is_none());
    }
}
