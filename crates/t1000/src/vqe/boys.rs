// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! The Boys function `F_n(x) = ∫₀¹ t^(2n) exp(-x t²) dt`, the radial kernel of every Coulomb
//! integral over Gaussians.

/// Below this `x` the convergent series is used; above it the asymptotic closed form.
const SERIES_LIMIT: f64 = 35.0;

/// `F_0(x), …, F_n_max(x)`.
///
/// Small `x` uses the series for the top order and the stable downward recursion
/// `F_(n-1) = (2x F_n + e^-x) / (2n - 1)`; large `x` starts from `F_0 = ½ √(π/x)` and recurses
/// upward, which is stable there.
pub fn boys_all(n_max: usize, x: f64) -> Vec<f64> {
    let mut f = vec![0.0; n_max + 1];
    if x < 1e-13 {
        for (n, v) in f.iter_mut().enumerate() {
            *v = 1.0 / (2 * n + 1) as f64;
        }
        return f;
    }
    let exp_neg = (-x).exp();
    if x < SERIES_LIMIT {
        f[n_max] = boys_series(n_max, x, exp_neg);
        for n in (1..=n_max).rev() {
            f[n - 1] = (2.0 * x * f[n] + exp_neg) / (2 * n - 1) as f64;
        }
    } else {
        f[0] = 0.5 * (std::f64::consts::PI / x).sqrt();
        for n in 0..n_max {
            f[n + 1] = ((2 * n + 1) as f64 * f[n] - exp_neg) / (2.0 * x);
        }
    }
    f
}

/// `F_n(x) = e^-x Σ_k (2x)^k / ((2n+1)(2n+3)…(2n+2k+1))`.
fn boys_series(n: usize, x: f64, exp_neg: f64) -> f64 {
    let mut term = 1.0 / (2 * n + 1) as f64;
    let mut sum = term;
    let mut k = 0usize;
    while term > 1e-17 * sum && k < 400 {
        k += 1;
        term *= 2.0 * x / (2 * n + 2 * k + 1) as f64;
        sum += term;
    }
    exp_neg * sum
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Composite Simpson quadrature of the defining integral.
    fn quadrature(n: usize, x: f64) -> f64 {
        let steps = 20_000i32;
        let h = 1.0 / f64::from(steps);
        let f = |t: f64| t.powi(2 * n as i32) * (-x * t * t).exp();
        let mut sum = f(0.0) + f(1.0);
        for i in 1..steps {
            sum += f(f64::from(i) * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
        }
        sum * h / 3.0
    }

    #[test]
    fn matches_numerical_quadrature_across_the_series_and_asymptotic_ranges() {
        for &x in &[1e-9, 0.01, 0.5, 3.0, 12.0, 34.9, 35.1, 80.0] {
            let f = boys_all(6, x);
            for (n, &fn_x) in f.iter().enumerate() {
                let q = quadrature(n, x);
                assert!(
                    (fn_x - q).abs() < 1e-10 * q.max(1e-30) + 1e-14,
                    "F_{n}({x}) = {fn_x} vs {q}"
                );
            }
        }
    }

    #[test]
    fn zero_argument_gives_the_reciprocal_odd_integers() {
        let f = boys_all(4, 0.0);
        for (n, v) in f.iter().enumerate() {
            assert!((v - 1.0 / (2 * n + 1) as f64).abs() < 1e-15);
        }
    }

    #[test]
    fn f0_has_the_error_function_closed_form_at_large_x() {
        // F_0(x) → ½ √(π/x) once erf(√x) = 1 to machine precision.
        let x = 50.0;
        assert!((boys_all(0, x)[0] - 0.5 * (std::f64::consts::PI / x).sqrt()).abs() < 1e-17);
    }
}
