// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

use std::ops::Neg;

#[derive(Debug)]
pub enum Side {
    A,
    B,
}

/// Binary correlation matrix between two subspaces with Schmidt-like coefficients.
#[derive(Debug, Clone)]
pub struct EntanglementTensor {
    pub dim_a: usize,
    pub dim_b: usize,
    pub correlations: Vec<Vec<f64>>,
    pub schmidt_coeffs: Vec<f64>,
}

impl EntanglementTensor {
    /// Build from paired event logs: rows are subsystem A indices, cols B indices.
    /// Correlation is symmetric co-activation count.
    pub fn from_events(a: &[u64], b: &[u64]) -> Self {
        let dim_a = (a.iter().cloned().max().unwrap_or(0) + 1) as usize;
        let dim_b = (b.iter().cloned().max().unwrap_or(0) + 1) as usize;
        let mut corr = vec![vec![0.0; dim_b]; dim_a];
        for (&i, &j) in a.iter().zip(b.iter()) {
            corr[i as usize][j as usize] += 1.0;
        }
        // Schmidt coefficients via SVD-lite: singular values sqrt(eigenvalues of A * A^T)
        let coeffs = schmidt_like(&corr);
        Self {
            dim_a,
            dim_b,
            correlations: corr,
            schmidt_coeffs: coeffs,
        }
    }

    pub fn schmidt_rank(&self) -> usize {
        self.schmidt_coeffs.iter().filter(|c| **c > 1e-9).count()
    }

    /// Mutual information proxy: -sum p log p from normalized Schmidt coeffs.
    pub fn mutual_information(&self) -> f64 {
        self.schmidt_coeffs
            .iter()
            .map(|c| {
                let p = c.powi(2);
                if p > 0.0 {
                    p * p.ln()
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            .neg()
    }

    /// Partial trace: reduce to the kept side by summing out the other.
    pub fn partial_trace(&self, keep: Side) -> Vec<f64> {
        match keep {
            Side::A => (0..self.dim_a)
                .map(|i| self.correlations[i].iter().sum())
                .collect(),
            Side::B => (0..self.dim_b)
                .map(|j| self.correlations.iter().map(|row| row[j]).sum())
                .collect(),
        }
    }
}

fn schmidt_like(corr: &[Vec<f64>]) -> Vec<f64> {
    // Power iteration on correlation matrix for top singular values.
    // For small matrices this is sufficient; correctness is "good enough" for
    // ranking entanglement between subspaces.
    let m = corr.len();
    if m == 0 || corr[0].is_empty() {
        return Vec::new();
    }
    let mut coeffs = Vec::new();
    let mut residual = corr.to_vec();
    for _ in 0..m.min(8) {
        let mut v = vec![1.0 / (m as f64).sqrt(); m];
        for _ in 0..12 {
            let mut new_v = vec![0.0; m];
            for i in 0..m {
                for (j, row_j) in residual.iter().enumerate() {
                    if j < row_j.len() {
                        new_v[i] += residual[i][j] * v[j];
                    }
                }
            }
            let norm = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for x in &mut new_v {
                    *x /= norm;
                }
            }
            v = new_v;
        }
        // Rayleigh quotient = v^T * corr * v
        let mut rq = 0.0;
        for i in 0..m {
            for j in 0..m.min(residual[0].len()) {
                rq += v[i] * residual[i][j] * v[j];
            }
        }
        coeffs.push(rq.abs().sqrt());
        // Deflate
        for i in 0..m {
            for j in 0..m.min(residual[0].len()) {
                residual[i][j] -= rq * v[i] * v[j];
            }
        }
        if rq.abs() < 1e-9 {
            break;
        }
    }
    coeffs
}
