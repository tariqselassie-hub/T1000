use num_complex::Complex64;

/// Dense row-major complex matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Complex64>,
}

impl ComplexMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![Complex64::new(0.0, 0.0); rows * cols],
        }
    }

    pub fn from_flat(rows: usize, cols: usize, data: Vec<Complex64>) -> Self {
        assert_eq!(data.len(), rows * cols, "data len must equal rows * cols");
        Self { rows, cols, data }
    }

    pub fn identity(n: usize) -> Self {
        let mut mat = Self::new(n, n);
        for i in 0..n {
            mat[(i, i)] = Complex64::new(1.0, 0.0);
        }
        mat
    }

    #[inline(always)]
    pub fn get(&self, r: usize, c: usize) -> Complex64 {
        self.data[r * self.cols + c]
    }

    #[inline(always)]
    pub fn set(&mut self, r: usize, c: usize, val: Complex64) {
        self.data[r * self.cols + c] = val;
    }

    pub fn conjugate_transpose(&self) -> Self {
        let mut adj = Self::new(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                adj[(c, r)] = self[(r, c)].conj();
            }
        }
        adj
    }

    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows, "incompatible matrix dimensions for matmul");
        let mut out = Self::new(self.rows, other.cols);
        for r in 0..self.rows {
            for k in 0..self.cols {
                let a = self[(r, k)];
                if a.norm_sqr() < 1e-28 {
                    continue;
                }
                for c in 0..other.cols {
                    out[(r, c)] += a * other[(k, c)];
                }
            }
        }
        out
    }

    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
    }
}

impl std::ops::Index<(usize, usize)> for ComplexMatrix {
    type Output = Complex64;
    #[inline(always)]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0 * self.cols + index.1]
    }
}

impl std::ops::IndexMut<(usize, usize)> for ComplexMatrix {
    #[inline(always)]
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0 * self.cols + index.1]
    }
}

/// Result of Singular Value Decomposition: M = U * diag(S) * V^dagger
#[derive(Debug, Clone)]
pub struct SvdResult {
    /// Left singular vectors (rows x rank)
    pub u: ComplexMatrix,
    /// Singular values in non-increasing order (rank)
    pub s: Vec<f64>,
    /// Right singular vectors V^dagger (rank x cols)
    pub v_dag: ComplexMatrix,
    /// Sum of squares of discarded singular values (truncation error)
    pub discarded_weight: f64,
}

/// Compute SVD of a complex matrix M = U * diag(S) * V^dagger using one-sided Jacobi rotations.
/// If max_rank is specified, singular values are truncated to min(max_rank, available_rank).
pub fn svd(m: &ComplexMatrix, max_rank: Option<usize>) -> SvdResult {
    let rows = m.rows;
    let cols = m.cols;

    if rows == 0 || cols == 0 {
        return SvdResult {
            u: ComplexMatrix::new(rows, 0),
            s: Vec::new(),
            v_dag: ComplexMatrix::new(0, cols),
            discarded_weight: 0.0,
        };
    }

    // If rows < cols, decompose M^dagger: M^dagger = V * S * U^dagger => M = U * S * V^dagger
    if rows < cols {
        let m_adj = m.conjugate_transpose();
        let adj_res = svd(&m_adj, max_rank);
        return SvdResult {
            u: adj_res.v_dag.conjugate_transpose(),
            s: adj_res.s,
            v_dag: adj_res.u.conjugate_transpose(),
            discarded_weight: adj_res.discarded_weight,
        };
    }

    // rows >= cols: one-sided Hestenes Jacobi SVD on columns of A
    let mut a = m.clone();
    let mut v = ComplexMatrix::identity(cols);

    let max_sweeps = 60;
    let tol = 1e-13;

    for _sweep in 0..max_sweeps {
        let mut max_off_diag = 0.0;

        for j in 0..cols {
            for k in (j + 1)..cols {
                // Compute alpha = ||a_j||^2, beta = ||a_k||^2, gamma = <a_j, a_k>
                let mut alpha = 0.0;
                let mut beta = 0.0;
                let mut gamma = Complex64::new(0.0, 0.0);

                for r in 0..rows {
                    let aj = a[(r, j)];
                    let ak = a[(r, k)];
                    alpha += aj.norm_sqr();
                    beta += ak.norm_sqr();
                    gamma += aj.conj() * ak;
                }

                let gamma_abs = gamma.norm();
                let denom = (alpha * beta).sqrt();
                if denom > 1e-28 {
                    let rel_corr = gamma_abs / denom;
                    if rel_corr > max_off_diag {
                        max_off_diag = rel_corr;
                    }
                }

                if gamma_abs < tol * denom.max(1e-14) {
                    continue;
                }

                // 2x2 complex Jacobi rotation:
                // gamma = |gamma| * exp(i * psi)
                let psi_exp = gamma / gamma_abs; // exp(i * psi)
                let zeta = (beta - alpha) / (2.0 * gamma_abs);
                let t = if zeta >= 0.0 {
                    1.0 / (zeta + (1.0 + zeta * zeta).sqrt())
                } else {
                    -1.0 / (-zeta + (1.0 + zeta * zeta).sqrt())
                };

                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let s_phase_conj = Complex64::new(s, 0.0) * psi_exp.conj();
                let s_phase = Complex64::new(s, 0.0) * psi_exp;

                // Update columns of A:
                // a'_j = c * a_j - s * exp(-i*psi) * a_k
                // a'_k = s * exp(i*psi) * a_j + c * a_k
                for r in 0..rows {
                    let aj = a[(r, j)];
                    let ak = a[(r, k)];
                    a[(r, j)] = Complex64::new(c, 0.0) * aj - s_phase_conj * ak;
                    a[(r, k)] = s_phase * aj + Complex64::new(c, 0.0) * ak;
                }

                // Accumulate right singular vectors V:
                for r in 0..cols {
                    let vj = v[(r, j)];
                    let vk = v[(r, k)];
                    v[(r, j)] = Complex64::new(c, 0.0) * vj - s_phase_conj * vk;
                    v[(r, k)] = s_phase * vj + Complex64::new(c, 0.0) * vk;
                }
            }
        }

        if max_off_diag < tol {
            break;
        }
    }

    // Extract singular values as column norms
    let mut s_vals = Vec::with_capacity(cols);
    for c in 0..cols {
        let mut norm_sq = 0.0;
        for r in 0..rows {
            norm_sq += a[(r, c)].norm_sqr();
        }
        s_vals.push((norm_sq.sqrt(), c));
    }

    // Sort descending by singular value
    s_vals.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));

    let full_rank = cols;
    let target_rank = match max_rank {
        Some(k) => k.min(full_rank).max(1),
        None => full_rank,
    };

    let mut discarded_weight = 0.0;
    for &(sigma, _) in s_vals.iter().take(full_rank).skip(target_rank) {
        discarded_weight += sigma.powi(2);
    }

    let mut u_mat = ComplexMatrix::new(rows, target_rank);
    let mut s_out = Vec::with_capacity(target_rank);
    let mut v_dag = ComplexMatrix::new(target_rank, cols);

    for (new_col, &(sigma, orig_col)) in s_vals.iter().take(target_rank).enumerate() {
        s_out.push(sigma);
        let inv_sigma = if sigma > 1e-14 { 1.0 / sigma } else { 0.0 };

        for r in 0..rows {
            u_mat[(r, new_col)] = a[(r, orig_col)] * inv_sigma;
        }

        for c in 0..cols {
            // Row new_col of V^dagger is (column orig_col of V)^dagger
            v_dag[(new_col, c)] = v[(c, orig_col)].conj();
        }
    }

    SvdResult {
        u: u_mat,
        s: s_out,
        v_dag,
        discarded_weight,
    }
}

/// Compute QR decomposition of M = Q * R using Householder reflections or Gram-Schmidt.
pub fn qr(m: &ComplexMatrix) -> (ComplexMatrix, ComplexMatrix) {
    let rows = m.rows;
    let cols = m.cols;
    let k = rows.min(cols);

    let mut q = ComplexMatrix::new(rows, k);
    let mut r = ComplexMatrix::new(k, cols);

    // Modified Gram-Schmidt
    let mut v_cols: Vec<Vec<Complex64>> = (0..cols)
        .map(|c| (0..rows).map(|r_idx| m[(r_idx, c)]).collect())
        .collect();

    for j in 0..k {
        let norm_sq: f64 = v_cols[j].iter().take(rows).map(|v| v.norm_sqr()).sum();
        let norm = norm_sq.sqrt();
        r[(j, j)] = Complex64::new(norm, 0.0);

        if norm > 1e-14 {
            for i in 0..rows {
                q[(i, j)] = v_cols[j][i] / norm;
            }
        }

        for col in (j + 1)..cols {
            let mut dot = Complex64::new(0.0, 0.0);
            for i in 0..rows {
                dot += q[(i, j)].conj() * v_cols[col][i];
            }
            r[(j, col)] = dot;
            for i in 0..rows {
                v_cols[col][i] -= dot * q[(i, j)];
            }
        }
    }

    (q, r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_matmul_and_identity() {
        let eye = ComplexMatrix::identity(3);
        let mut a = ComplexMatrix::new(3, 3);
        a[(0, 1)] = Complex64::new(2.0, -1.0);
        a[(1, 2)] = Complex64::new(0.5, 3.0);
        a[(2, 0)] = Complex64::new(-4.0, 0.0);

        let prod = a.matmul(&eye);
        assert_eq!(prod, a);
    }

    #[test]
    fn test_svd_reconstruction_square() {
        let mut a = ComplexMatrix::new(3, 3);
        a[(0, 0)] = Complex64::new(1.0, 2.0);
        a[(0, 1)] = Complex64::new(3.0, 0.0);
        a[(1, 1)] = Complex64::new(0.0, -4.0);
        a[(2, 0)] = Complex64::new(5.0, 1.0);
        a[(2, 2)] = Complex64::new(2.0, -2.0);

        let res = svd(&a, None);
        assert_eq!(res.s.len(), 3);
        assert!(res.s[0] >= res.s[1] && res.s[1] >= res.s[2]);

        // Reconstruct M_rec = U * diag(S) * V^dag
        let mut s_mat = ComplexMatrix::new(3, 3);
        for i in 0..3 {
            s_mat[(i, i)] = Complex64::new(res.s[i], 0.0);
        }
        let u_s = res.u.matmul(&s_mat);
        let rec = u_s.matmul(&res.v_dag);

        for r in 0..3 {
            for c in 0..3 {
                let diff = (a[(r, c)] - rec[(r, c)]).norm();
                assert!(diff < 1e-10, "SVD reconstruction error at ({r}, {c}): {diff}");
            }
        }

        // Check U^dag * U = I
        let u_dag_u = res.u.conjugate_transpose().matmul(&res.u);
        let eye = ComplexMatrix::identity(3);
        assert!((u_dag_u.frobenius_norm() - eye.frobenius_norm()).abs() < 1e-10);
    }

    #[test]
    fn test_svd_truncation() {
        let mut a = ComplexMatrix::new(4, 2);
        a[(0, 0)] = Complex64::new(10.0, 0.0);
        a[(1, 0)] = Complex64::new(5.0, 0.0);
        a[(2, 1)] = Complex64::new(1.0, 0.0);
        a[(3, 1)] = Complex64::new(0.1, 0.0);

        let res = svd(&a, Some(1));
        assert_eq!(res.s.len(), 1);
        assert!(res.discarded_weight > 0.0);
    }

    #[test]
    fn test_qr_decomposition() {
        let mut a = ComplexMatrix::new(3, 2);
        a[(0, 0)] = Complex64::new(1.0, 0.0);
        a[(1, 0)] = Complex64::new(2.0, 1.0);
        a[(2, 1)] = Complex64::new(3.0, -1.0);

        let (q, r) = qr(&a);
        let rec = q.matmul(&r);
        for i in 0..3 {
            for j in 0..2 {
                assert!((a[(i, j)] - rec[(i, j)]).norm() < 1e-10);
            }
        }
        let q_dag_q = q.conjugate_transpose().matmul(&q);
        let eye = ComplexMatrix::identity(2);
        for i in 0..2 {
            for j in 0..2 {
                assert!((q_dag_q[(i, j)] - eye[(i, j)]).norm() < 1e-10);
            }
        }
    }
}
