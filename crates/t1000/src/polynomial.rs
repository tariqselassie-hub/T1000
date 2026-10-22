// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Polynomial feature expansion for classical baselines.

/// All monomials of the inputs from degree 1 up to `degree`, grouped by degree, with the factors of
/// each monomial in non-decreasing index order (so `x_i x_j` with `i <= j`, `x_i x_j x_k` with
/// `i <= j <= k`). Degree 2 reproduces `baselines::quadratic_features`.
pub fn polynomial_features(x: &[f64], degree: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(feature_count(x.len(), degree));
    // `current[d]` holds the monomials of degree `d` with their largest factor index, for the
    // recurrence "multiply a degree d-1 monomial by x_j with j >= its largest factor index".
    let mut previous: Vec<(f64, usize)> = x.iter().enumerate().map(|(i, v)| (*v, i)).collect();
    out.extend(previous.iter().map(|(v, _)| *v));
    for _ in 2..=degree {
        let next: Vec<(f64, usize)> = previous
            .iter()
            .flat_map(|&(value, last)| (last..x.len()).map(move |j| (value * x[j], j)))
            .collect();
        out.extend(next.iter().map(|(v, _)| *v));
        previous = next;
    }
    out
}

/// Number of monomials from degree 1 to `degree` in `n` inputs: `C(n + degree, degree) - 1`.
pub fn feature_count(n: usize, degree: usize) -> usize {
    let (mut c, top) = (1u128, n + degree);
    for i in 0..degree {
        c = c * (top - i) as u128 / (i + 1) as u128;
    }
    c as usize - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baselines::quadratic_features;

    #[test]
    fn degree_two_matches_the_quadratic_features() {
        let x = [0.5, -1.25, 2.0, 0.0, 3.5];
        assert_eq!(polynomial_features(&x, 2), quadratic_features(&x));
    }

    #[test]
    fn counts_follow_the_binomial_formula() {
        assert_eq!(feature_count(19, 1), 19);
        assert_eq!(feature_count(19, 2), 19 + 190);
        assert_eq!(feature_count(19, 3), 19 + 190 + 1330);
        for (n, d) in [(3, 3), (5, 4), (19, 3)] {
            assert_eq!(polynomial_features(&vec![1.0; n], d).len(), feature_count(n, d));
        }
    }

    #[test]
    fn cubic_terms_are_the_ordered_products() {
        let x = [2.0, 3.0];
        // degree 1: 2, 3; degree 2: 4, 6, 9; degree 3: 8, 12, 18, 27.
        assert_eq!(polynomial_features(&x, 3), vec![2.0, 3.0, 4.0, 6.0, 9.0, 8.0, 12.0, 18.0, 27.0]);
    }
}
