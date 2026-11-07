// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Reed-Solomon Code Formulation & Classical Baselines
//!
//! Implements generalized Reed-Solomon Maximum Distance Separable (MDS) codes
//! over $\mathbb{F}_p$, syndrome generation, classical Berlekamp-Welch unique
//! decoding, and brute-force Maximum-LINSAT search baselines.

use super::field::{field_dot, field_nullspace, FieldElement};
use serde::{Deserialize, Serialize};

/// Maximum Distance Separable (MDS) Reed-Solomon Code over $\mathbb{F}_p$.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReedSolomonCode {
    /// Field characteristic / modulus $p$.
    pub p: u32,
    /// Codeword block length $n \le p$.
    pub n: usize,
    /// Message dimension $k \le n$ (polynomial degree $\le k-1$).
    pub k: usize,
    /// Distinct evaluation points $\alpha_0, \dots, \alpha_{n-1} \in \mathbb{F}_p$.
    pub eval_points: Vec<FieldElement>,
    /// Generator matrix $G \in \mathbb{F}_p^{k \times n}$.
    pub generator_matrix: Vec<Vec<FieldElement>>,
    /// Parity-check matrix $H \in \mathbb{F}_p^{(n - k) \times n}$.
    pub parity_matrix: Vec<Vec<FieldElement>>,
}

#[allow(clippy::needless_range_loop, clippy::cast_lossless)]
impl ReedSolomonCode {
    /// Constructs a new Reed-Solomon code with standard evaluation points $\alpha_i = i \pmod p$.
    pub fn new(p: u32, n: usize, k: usize) -> Self {
        assert!(k >= 1, "Message dimension k must be >= 1");
        assert!(n >= k, "Block length n must be >= k");
        assert!(n <= p as usize, "Block length n must not exceed field size p");

        let eval_points: Vec<FieldElement> = (0..n)
            .map(|i| FieldElement::new(i as u32, p))
            .collect();

        // Generator matrix: G[j][i] = alpha_i^j for j in 0..k
        let mut generator_matrix = vec![vec![FieldElement::zero(p); n]; k];
        for j in 0..k {
            for i in 0..n {
                generator_matrix[j][i] = eval_points[i].pow(j as u64);
            }
        }

        // Parity-check matrix H from nullspace of G
        let parity_matrix = field_nullspace(&generator_matrix, k, n);
        assert_eq!(
            parity_matrix.len(),
            n - k,
            "Parity-check matrix row count must be exactly n - k"
        );

        Self {
            p,
            n,
            k,
            eval_points,
            generator_matrix,
            parity_matrix,
        }
    }

    /// Minimum Hamming distance $d_{\min} = n - k + 1$ (Singleton bound).
    #[inline(always)]
    pub fn minimum_distance(&self) -> usize {
        self.n - self.k + 1
    }

    /// Classical unique decoding error radius $t_{\text{unique}} = \lfloor(n - k)/2\rfloor$.
    #[inline(always)]
    pub fn unique_decoding_radius(&self) -> usize {
        (self.n - self.k) / 2
    }

    /// Guruswami-Sudan list decoding radius $t_{\text{GS}} \approx n - \sqrt{n(k-1)}$.
    pub fn guruswami_sudan_radius(&self) -> usize {
        let bound = (self.n as f64) - ((self.n * (self.k - 1)) as f64).sqrt();
        bound.max(0.0).floor() as usize
    }

    /// Evaluates polynomial $P(x) = \sum_{j=0}^{k-1} c_j x^j$ at point $x$.
    pub fn evaluate_poly(&self, coeffs: &[FieldElement], x: FieldElement) -> FieldElement {
        let mut sum = FieldElement::zero(self.p);
        for (j, &c) in coeffs.iter().enumerate() {
            sum = sum.add(c.mul(x.pow(j as u64)));
        }
        sum
    }

    /// Encodes a message $\vec{m} \in \mathbb{F}_p^k$ into codeword $\vec{c} \in \mathbb{F}_p^n$.
    pub fn encode(&self, message: &[FieldElement]) -> Vec<FieldElement> {
        assert_eq!(message.len(), self.k, "Message length must equal k");
        let mut codeword = Vec::with_capacity(self.n);
        for &pt in &self.eval_points {
            codeword.push(self.evaluate_poly(message, pt));
        }
        codeword
    }

    /// Computes the syndrome vector $\vec{s} = H \vec{v} \in \mathbb{F}_p^{n - k}$.
    pub fn compute_syndrome(&self, vector: &[FieldElement]) -> Vec<FieldElement> {
        assert_eq!(vector.len(), self.n);
        self.parity_matrix
            .iter()
            .map(|row| field_dot(row, vector))
            .collect()
    }

    /// Alias for compute_syndrome.
    #[inline(always)]
    pub fn syndrome(&self, vector: &[FieldElement]) -> Vec<FieldElement> {
        self.compute_syndrome(vector)
    }

    /// Returns all $p^k$ possible degree $< k$ polynomials over $\mathbb{F}_p$.
    pub fn all_polynomials(&self) -> Vec<Vec<FieldElement>> {
        let total = (self.p as usize).pow(self.k as u32);
        let mut list = Vec::with_capacity(total);
        for idx in 0..total {
            let mut cur = idx;
            let mut poly = Vec::with_capacity(self.k);
            for _ in 0..self.k {
                poly.push(FieldElement::new((cur % self.p as usize) as u32, self.p));
                cur /= self.p as usize;
            }
            list.push(poly);
        }
        list
    }

    /// Checks if a vector is a valid codeword ($H \vec{v} == \vec{0}$).
    pub fn is_codeword(&self, vector: &[FieldElement]) -> bool {
        self.compute_syndrome(vector)
            .iter()
            .all(|s| s.value == 0)
    }

    /// Calculates agreement / intersection count $|\{i : P(\alpha_i) = y_i\}|$.
    pub fn agreement_score(&self, coeffs: &[FieldElement], observed: &[FieldElement]) -> usize {
        assert_eq!(observed.len(), self.n);
        let mut matches = 0;
        for i in 0..self.n {
            if self.evaluate_poly(coeffs, self.eval_points[i]) == observed[i] {
                matches += 1;
            }
        }
        matches
    }

    /// Classical Berlekamp-Welch unique decoder.
    /// Returns `Some(coeffs)` if error weight $t \le t_{\text{unique}}$, else `None`.
    pub fn decode_berlekamp_welch(&self, received: &[FieldElement]) -> Option<Vec<FieldElement>> {
        let t = self.unique_decoding_radius();
        let n = self.n;
        let k = self.k;

        // Berlekamp-Welch sets up: N(x_i) = r_i * E(x_i)
        // where deg(N) <= e + k - 1, deg(E) <= e, with E monic.
        // Unknowns: (e + k) coeffs for N, e coeffs for E.
        let num_unknowns = (t + k) + t;
        if n < num_unknowns {
            return None;
        }

        let mut matrix = vec![vec![FieldElement::zero(self.p); num_unknowns + 1]; n];
        for i in 0..n {
            let x_i = self.eval_points[i];
            let y_i = received[i];

            // N(x_i) coefficients: x_i^0, ..., x_i^{t + k - 1}
            for j in 0..(t + k) {
                matrix[i][j] = x_i.pow(j as u64);
            }
            // - y_i * E(x_i) for degrees 0..(t - 1)
            for j in 0..t {
                let term = y_i.mul(x_i.pow(j as u64));
                matrix[i][(t + k) + j] = FieldElement::zero(self.p).sub(term);
            }
            // RHS: y_i * x_i^t (since E is monic: E(x) = x^t + ...)
            matrix[i][num_unknowns] = y_i.mul(x_i.pow(t as u64));
        }

        // Gaussian elimination on augmented matrix
        let solution = solve_linear_system(&mut matrix, n, num_unknowns, self.p)?;

        // Extract candidate polynomial N(x)
        let n_coeffs = &solution[0..(t + k)];
        let mut e_coeffs = vec![FieldElement::zero(self.p); t + 1];
        for j in 0..t {
            e_coeffs[j] = solution[(t + k) + j];
        }
        e_coeffs[t] = FieldElement::one(self.p);

        // Divide N(x) by E(x)
        let poly = polynomial_division(n_coeffs, &e_coeffs, self.p)?;
        if poly.len() <= k {
            let mut padded = poly;
            padded.resize(k, FieldElement::zero(self.p));
            // Verify agreement
            if self.agreement_score(&padded, received) >= n - t {
                return Some(padded);
            }
        }
        None
    }

    /// Classical exhaustive search (brute-force) solving Maximum-LINSAT over all $p^k$ polynomials.
    /// Demonstrates $O(p^k)$ classical complexity.
    pub fn classical_brute_force_search(
        &self,
        received: &[FieldElement],
    ) -> (Vec<FieldElement>, usize, u64) {
        let total_candidates = (self.p as u64).pow(self.k as u32);
        let mut best_coeffs = vec![FieldElement::zero(self.p); self.k];
        let mut max_agreement = 0;
        let mut evaluations = 0u64;

        for idx in 0..total_candidates {
            evaluations += 1;
            let mut cur = idx;
            let mut candidate = Vec::with_capacity(self.k);
            for _ in 0..self.k {
                candidate.push(FieldElement::new((cur % self.p as u64) as u32, self.p));
                cur /= self.p as u64;
            }

            let score = self.agreement_score(&candidate, received);
            if score > max_agreement {
                max_agreement = score;
                best_coeffs = candidate;
            }
        }

        (best_coeffs, max_agreement, evaluations)
    }

    /// Decodes by exhaustive brute force, returning the candidate with maximum agreement.
    pub fn decode_brute_force(&self, received: &[FieldElement]) -> RsBruteForceResult {
        let (poly, score, evals) = self.classical_brute_force_search(received);
        RsBruteForceResult {
            best_polynomial: poly,
            max_agreement: score,
            total_candidates_searched: evals as usize,
        }
    }
}

/// Result of classical exhaustive search over all $p^k$ polynomials.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RsBruteForceResult {
    /// Best polynomial coefficients.
    pub best_polynomial: Vec<FieldElement>,
    /// Maximum agreement score achieved.
    pub max_agreement: usize,
    /// Total candidates evaluated.
    pub total_candidates_searched: usize,
}

/// Solves an augmented system $A x = b$ over $\mathbb{F}_p$.
#[allow(clippy::needless_range_loop)]
fn solve_linear_system(
    matrix: &mut [Vec<FieldElement>],
    rows: usize,
    cols: usize,
    p: u32,
) -> Option<Vec<FieldElement>> {
    let mut lead = 0;
    for r in 0..rows {
        if lead >= cols {
            break;
        }
        let mut i = r;
        while matrix[i][lead].value == 0 {
            i += 1;
            if i == rows {
                i = r;
                lead += 1;
                if lead == cols {
                    break;
                }
            }
        }
        if lead == cols {
            break;
        }

        matrix.swap(i, r);
        let lead_val = matrix[r][lead];
        let inv_lead = lead_val.inv()?;
        for c in 0..=cols {
            matrix[r][c] = matrix[r][c].mul(inv_lead);
        }

        for j in 0..rows {
            if j != r {
                let factor = matrix[j][lead];
                for c in 0..=cols {
                    let sub = factor.mul(matrix[r][c]);
                    matrix[j][c] = matrix[j][c].sub(sub);
                }
            }
        }
        lead += 1;
    }

    let mut solution = vec![FieldElement::zero(p); cols];
    for r in 0..cols.min(rows) {
        solution[r] = matrix[r][cols];
    }
    Some(solution)
}

/// Polynomial long division $N(x) / E(x)$ over $\mathbb{F}_p$.
fn polynomial_division(
    num: &[FieldElement],
    den: &[FieldElement],
    p: u32,
) -> Option<Vec<FieldElement>> {
    let mut rem = num.to_vec();
    while rem.len() > 1 && rem.last().is_some_and(|x| x.value == 0) {
        rem.pop();
    }
    let mut den_trimmed = den.to_vec();
    while den_trimmed.len() > 1 && den_trimmed.last().is_some_and(|x| x.value == 0) {
        den_trimmed.pop();
    }

    if den_trimmed.is_empty() || (den_trimmed.len() == 1 && den_trimmed[0].value == 0) {
        return None;
    }
    if rem.len() < den_trimmed.len() {
        return Some(vec![FieldElement::zero(p)]);
    }

    let deg_num = rem.len() - 1;
    let deg_den = den_trimmed.len() - 1;
    let mut quo = vec![FieldElement::zero(p); deg_num - deg_den + 1];

    let lead_den_inv = den_trimmed.last().unwrap().inv()?;

    for i in (0..=deg_num - deg_den).rev() {
        let cur_deg = i + deg_den;
        if cur_deg >= rem.len() {
            continue;
        }
        let coeff = rem[cur_deg].mul(lead_den_inv);
        quo[i] = coeff;

        for j in 0..=deg_den {
            let sub = coeff.mul(den_trimmed[j]);
            rem[i + j] = rem[i + j].sub(sub);
        }
    }

    // Verify zero remainder
    if rem.iter().any(|r| r.value != 0) {
        return None;
    }

    Some(quo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reed_solomon_encoding_and_syndrome() {
        let p = 7;
        let rs = ReedSolomonCode::new(p, 5, 2);
        assert_eq!(rs.minimum_distance(), 4);
        assert_eq!(rs.unique_decoding_radius(), 1);

        let msg = vec![FieldElement::new(3, p), FieldElement::new(2, p)]; // 3 + 2x
        let codeword = rs.encode(&msg);
        assert_eq!(codeword.len(), 5);
        assert!(rs.is_codeword(&codeword));

        // Corrupt with 1 error
        let mut corrupted = codeword.clone();
        corrupted[1] = corrupted[1].add(FieldElement::new(1, p));
        assert!(!rs.is_codeword(&corrupted));

        // Classical Berlekamp-Welch decodes 1 error
        let decoded = rs.decode_berlekamp_welch(&corrupted).expect("Must decode within radius");
        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_beyond_unique_decoding_failure() {
        let p = 7;
        let rs = ReedSolomonCode::new(p, 5, 2); // t_unique = 1
        let msg = vec![FieldElement::new(3, p), FieldElement::new(2, p)];
        let codeword = rs.encode(&msg);

        // Corrupt with 2 errors (> t_unique = 1)
        let mut corrupted = codeword.clone();
        corrupted[0] = corrupted[0].add(FieldElement::new(1, p));
        corrupted[1] = corrupted[1].add(FieldElement::new(1, p));

        // Berlekamp-Welch must fail
        let decoded = rs.decode_berlekamp_welch(&corrupted);
        assert!(decoded.is_none(), "Classical unique decoder must fail when t > t_unique");

        // Classical brute force finds it with O(p^k) = 49 trials
        let (bf_poly, agreement, evals) = rs.classical_brute_force_search(&corrupted);
        assert_eq!(bf_poly, msg);
        assert_eq!(agreement, 3); // 5 - 2 = 3 matches
        assert_eq!(evals, 49);
    }
}
