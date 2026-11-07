// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Prime Finite Field Arithmetic ($\mathbb{F}_p$)
//!
//! Provides zero-allocation arithmetic operations over prime Galois fields $\mathbb{F}_p$:
//! addition, subtraction, multiplication, modular inversion via Extended Euclidean
//! Algorithm, and vector/matrix algebra over $\mathbb{F}_p$.

use serde::{Deserialize, Serialize};

/// Element of a prime finite field $\mathbb{F}_p$.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FieldElement {
    pub value: u32,
    pub modulus: u32,
}

#[allow(clippy::should_implement_trait, clippy::cast_lossless)]
impl FieldElement {
    /// Creates a new field element $v \pmod p$.
    #[inline(always)]
    pub fn new(value: u32, modulus: u32) -> Self {
        assert!(modulus >= 2, "Field modulus must be at least 2");
        Self {
            value: value % modulus,
            modulus,
        }
    }

    /// Zero element of $\mathbb{F}_p$.
    #[inline(always)]
    pub fn zero(modulus: u32) -> Self {
        Self { value: 0, modulus }
    }

    /// Multiplicative identity of $\mathbb{F}_p$.
    #[inline(always)]
    pub fn one(modulus: u32) -> Self {
        Self { value: 1, modulus }
    }

    /// Field addition $(a + b) \pmod p$.
    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        debug_assert_eq!(self.modulus, other.modulus);
        Self {
            value: (self.value + other.value) % self.modulus,
            modulus: self.modulus,
        }
    }

    /// Field subtraction $(a - b) \pmod p$.
    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        debug_assert_eq!(self.modulus, other.modulus);
        let val = if self.value >= other.value {
            self.value - other.value
        } else {
            self.value + self.modulus - other.value
        };
        Self {
            value: val % self.modulus,
            modulus: self.modulus,
        }
    }

    /// Field multiplication $(a \cdot b) \pmod p$.
    #[inline(always)]
    pub fn mul(self, other: Self) -> Self {
        debug_assert_eq!(self.modulus, other.modulus);
        Self {
            value: ((self.value as u64 * other.value as u64) % self.modulus as u64) as u32,
            modulus: self.modulus,
        }
    }

    /// Field modular inverse $a^{-1} \pmod p$ via Extended Euclidean Algorithm.
    pub fn inv(self) -> Option<Self> {
        if self.value == 0 {
            return None;
        }
        let (g, x, _) = extended_gcd(self.value as i64, self.modulus as i64);
        if g != 1 {
            return None;
        }
        let p = self.modulus as i64;
        let inv = ((x % p) + p) % p;
        Some(Self {
            value: inv as u32,
            modulus: self.modulus,
        })
    }

    /// Field division $a / b \pmod p$.
    #[inline(always)]
    pub fn div(self, other: Self) -> Option<Self> {
        other.inv().map(|inv_other| self.mul(inv_other))
    }

    /// Exponentiation $a^e \pmod p$.
    pub fn pow(self, mut exp: u64) -> Self {
        let mut base = self;
        let mut result = Self::one(self.modulus);
        while exp > 0 {
            if exp & 1 == 1 {
                result = result.mul(base);
            }
            base = base.mul(base);
            exp >>= 1;
        }
        result
    }
}

/// Extended Euclidean Algorithm returning $(g, x, y)$ such that $a x + b y = g = \gcd(a, b)$.
fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if a == 0 {
        (b, 0, 1)
    } else {
        let (g, x1, y1) = extended_gcd(b % a, a);
        let x = y1 - (b / a) * x1;
        let y = x1;
        (g, x, y)
    }
}

/// Computes the dot product of two vectors over $\mathbb{F}_p$.
pub fn field_dot(a: &[FieldElement], b: &[FieldElement]) -> FieldElement {
    assert_eq!(a.len(), b.len(), "Vector lengths must match");
    if a.is_empty() {
        return FieldElement::zero(2);
    }
    let p = a[0].modulus;
    let mut sum = FieldElement::zero(p);
    for i in 0..a.len() {
        sum = sum.add(a[i].mul(b[i]));
    }
    sum
}

/// Performs Gaussian elimination to compute the nullspace (kernel) of matrix $M \in \mathbb{F}_p^{r \times c}$.
/// Returns a basis of vectors $v \in \mathbb{F}_p^c$ such that $M v = \vec{0}$.
#[allow(clippy::needless_range_loop)]
pub fn field_nullspace(matrix: &[Vec<FieldElement>], rows: usize, cols: usize) -> Vec<Vec<FieldElement>> {
    if rows == 0 || cols == 0 {
        return Vec::new();
    }
    let p = matrix[0][0].modulus;
    let mut a = matrix.to_vec();

    // Reduced row echelon form
    let mut lead = 0;
    let mut pivot_cols = Vec::new();

    for r in 0..rows {
        if lead >= cols {
            break;
        }
        let mut i = r;
        while a[i][lead].value == 0 {
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

        a.swap(i, r);
        let lead_val = a[r][lead];
        if let Some(inv_lead) = lead_val.inv() {
            for c in 0..cols {
                a[r][c] = a[r][c].mul(inv_lead);
            }
        }

        for j in 0..rows {
            if j != r {
                let factor = a[j][lead];
                for c in 0..cols {
                    let sub = factor.mul(a[r][c]);
                    a[j][c] = a[j][c].sub(sub);
                }
            }
        }
        pivot_cols.push(lead);
        lead += 1;
    }

    // Free columns
    let free_cols: Vec<usize> = (0..cols).filter(|c| !pivot_cols.contains(c)).collect();
    let mut null_basis = Vec::with_capacity(free_cols.len());

    for &free_col in &free_cols {
        let mut basis_vec = vec![FieldElement::zero(p); cols];
        basis_vec[free_col] = FieldElement::one(p);

        for (pivot_row, &pivot_col) in pivot_cols.iter().enumerate() {
            let coeff = a[pivot_row][free_col];
            basis_vec[pivot_col] = FieldElement::zero(p).sub(coeff);
        }
        null_basis.push(basis_vec);
    }

    null_basis
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finite_field_arithmetic() {
        let p = 7;
        let a = FieldElement::new(5, p);
        let b = FieldElement::new(4, p);

        assert_eq!(a.add(b).value, 2); // 5 + 4 = 9 = 2 mod 7
        assert_eq!(a.sub(b).value, 1); // 5 - 4 = 1 mod 7
        assert_eq!(a.mul(b).value, 6); // 5 * 4 = 20 = 6 mod 7

        let inv_a = a.inv().expect("Inverse must exist for non-zero in prime field");
        assert_eq!(a.mul(inv_a).value, 1); // 5 * 3 = 15 = 1 mod 7
    }

    #[test]
    fn test_nullspace_parity_check() {
        let p = 7;
        // Generator matrix for [3, 2] code:
        // G = [[1, 1, 1],
        //      [0, 1, 2]]
        let g = vec![
            vec![FieldElement::new(1, p), FieldElement::new(1, p), FieldElement::new(1, p)],
            vec![FieldElement::new(0, p), FieldElement::new(1, p), FieldElement::new(2, p)],
        ];
        let h_basis = field_nullspace(&g, 2, 3);
        assert_eq!(h_basis.len(), 1, "Dual space dimension must be n - k = 3 - 2 = 1");

        // Verify H * G^T = 0
        let h = &h_basis[0];
        let syn0 = field_dot(h, &g[0]);
        let syn1 = field_dot(h, &g[1]);
        assert_eq!(syn0.value, 0);
        assert_eq!(syn1.value, 0);
    }
}
