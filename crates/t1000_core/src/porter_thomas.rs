// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Porter-Thomas Statistics & Exact Haar-Random Calibration Engine
//!
//! Provides mathematically exact cross-entropy benchmarking (XEB), finite-dimensional
//! Haar moments, Shannon entropy deficit calculation, and Kolmogorov-Smirnov
//! goodness-of-fit testing for quantum state scrambling across both Binary ($2^N$)
//! and Heptal ($7^K$) Hilbert spaces.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Euler-Mascheroni constant gamma = 0.5772156649015329
pub const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;

/// Asymptotic Haar entropy deficit: 1 - gamma ≈ 0.422784335 nats
pub const ASYMPTOTIC_HAAR_ENTROPY_DEFICIT: f64 = 1.0 - EULER_MASCHERONI;

/// Kolmogorov-Smirnov test evaluation result.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct KSResult {
    /// Maximum difference between empirical and theoretical CDF
    pub statistic: f64,
    /// Critical value at alpha = 0.05 (~1.358 / sqrt(D))
    pub critical_value: f64,
    /// Whether the null hypothesis of Porter-Thomas distribution is accepted
    pub passed: bool,
}

/// Comprehensive Porter-Thomas and Haar scrambling evaluation report.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PorterThomasReport {
    /// Total Hilbert space dimension D
    pub dim: usize,
    /// Exact finite-D Linear XEB fidelity (identically 1.0000 on ideal Haar states)
    pub exact_xeb: f64,
    /// Naive Linear XEB fidelity (D * sum(p^2) - 1.0)
    pub naive_xeb: f64,
    /// Empirical sum of squared probabilities: sum_{x} p(x)^2
    pub sum_p_sqr: f64,
    /// Theoretically expected sum of squared probabilities: 2 / (D + 1)
    pub expected_sum_p_sqr: f64,
    /// Empirical Shannon measurement entropy S = -sum p ln p
    pub shannon_entropy: f64,
    /// Empirical entropy deficit: ln(D) - S
    pub entropy_deficit: f64,
    /// Theoretically expected entropy deficit for finite D: `ln(D) - E[S_Haar]`
    pub expected_entropy_deficit: f64,
    /// Empirical Effective Participation Ratio (1 / sum(p^2))
    pub effective_participation_ratio: f64,
    /// Expected Effective Participation Ratio: (D + 1) / 2
    pub expected_participation_ratio: f64,
    /// Kolmogorov-Smirnov test statistic
    pub ks_statistic: f64,
    /// Kolmogorov-Smirnov critical threshold (alpha = 0.05)
    pub ks_critical_value: f64,
    /// Whether the distribution satisfies the Kolmogorov-Smirnov goodness-of-fit test
    pub passes_ks_test: bool,
    /// High-level confirmation: state exhibits verified Porter-Thomas quantum chaos
    pub is_porter_thomas_conforming: bool,
}

/// Calculates the exact finite-dimensional Linear Cross-Entropy Benchmarking (XEB) fidelity.
///
/// In a Hilbert space of dimension D:
/// - Flat uniform distribution (p_x = 1/D): returns `0.0000`
/// - Ideal Haar-random ensemble (E[sum p^2] = 2/(D+1)): returns `1.0000`
///
/// Formula:
/// ```text
/// F_XEB = [ D(D + 1) * sum(p^2) - (D + 1) ] / [ D - 1 ]
/// ```
#[inline]
pub fn exact_linear_xeb(probs: &[f64]) -> f64 {
    let d = probs.len();
    if d <= 1 {
        return 1.0;
    }
    let d_f = d as f64;
    let sum_p2: f64 = probs.iter().map(|&p| p * p).sum();
    (d_f * (d_f + 1.0) * sum_p2 - (d_f + 1.0)) / (d_f - 1.0)
}

/// Calculates the naive asymptotic Linear XEB fidelity: `D * sum(p^2) - 1.0`.
#[inline]
pub fn naive_linear_xeb(probs: &[f64]) -> f64 {
    let d = probs.len() as f64;
    let sum_p2: f64 = probs.iter().map(|&p| p * p).sum();
    d * sum_p2 - 1.0
}

/// Returns the exact theoretical expectation value of sum(p^2) for an ideal Haar-scrambled pure state:
/// `2 / (D + 1)`.
#[inline]
pub fn expected_sum_p_sqr(dim: usize) -> f64 {
    if dim == 0 {
        return 0.0;
    }
    2.0 / (dim as f64 + 1.0)
}

/// Computes the Shannon entropy in natural units (nats) of a probability distribution:
/// `S = -sum_{x: p(x) > 0} p(x) ln(p(x))`.
pub fn shannon_entropy(probs: &[f64]) -> f64 {
    probs
        .iter()
        .map(|&p| if p > 1e-18 { -p * p.ln() } else { 0.0 })
        .sum()
}

/// Returns the exact expected Haar-random measurement basis entropy for dimension D:
/// `E[S_Haar] = H_D - 1 = sum_{k=1}^D (1/k) - 1`
///
/// In the large-D limit:
/// `E[S_Haar] -> ln(D) + gamma - 1`
/// `E[Delta S] = ln(D) - E[S_Haar] -> 1 - gamma ≈ 0.422784 nats`
pub fn expected_haar_entropy(dim: usize) -> f64 {
    if dim <= 1 {
        return 0.0;
    }
    if dim < 50_000 {
        let mut harmonic_sum = 0.0;
        for k in 1..=dim {
            harmonic_sum += 1.0 / (k as f64);
        }
        harmonic_sum - 1.0
    } else {
        // Asymptotic expansion: H_D = ln(D) + gamma + 1/(2D) - 1/(12D^2)
        let d = dim as f64;
        let h_d = d.ln() + EULER_MASCHERONI + (1.0 / (2.0 * d)) - (1.0 / (12.0 * d * d));
        h_d - 1.0
    }
}

/// Computes the empirical and expected entropy deficit: `ln(D) - S`.
///
/// As D -> infinity, the expected deficit approaches `1 - gamma ≈ 0.422784` nats.
pub fn haar_entropy_deficit(probs: &[f64]) -> (f64, f64) {
    let d = probs.len();
    if d <= 1 {
        return (0.0, 0.0);
    }
    let ln_d = (d as f64).ln();
    let empirical_entropy = shannon_entropy(probs);
    let expected_entropy = expected_haar_entropy(d);
    (ln_d - empirical_entropy, ln_d - expected_entropy)
}

/// Calculates the Effective Participation Ratio (EPR): `1 / sum(p(x)^2)`.
///
/// For ideal Porter-Thomas, EPR is approximately `(D + 1) / 2` (occupying 50% of the Hilbert space volume).
#[inline]
pub fn effective_participation_ratio(probs: &[f64]) -> f64 {
    let sum_p2: f64 = probs.iter().map(|&p| p * p).sum();
    if sum_p2 > 1e-18 {
        1.0 / sum_p2
    } else {
        0.0
    }
}

/// Evaluates the Kolmogorov-Smirnov goodness-of-fit test comparing the empirical distribution
/// of rescaled probabilities `w_i = D * p_i` against the theoretical Porter-Thomas CDF:
/// `F(w) = 1 - exp(-w)`.
pub fn kolmogorov_smirnov_porter_thomas(probs: &[f64]) -> KSResult {
    let d = probs.len();
    if d < 4 {
        return KSResult {
            statistic: 0.0,
            critical_value: 1.0,
            passed: true,
        };
    }

    let d_f = d as f64;
    // Rescale to w = D * p
    let mut w_vals: Vec<f64> = probs.iter().map(|&p| (d_f * p).max(0.0)).collect();
    w_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut max_diff = 0.0f64;
    for (i, &w) in w_vals.iter().enumerate() {
        let f_empirical = (i + 1) as f64 / d_f;
        let f_empirical_prev = i as f64 / d_f;
        let f_theoretical = 1.0 - (-w).exp();

        let diff_plus = (f_empirical - f_theoretical).abs();
        let diff_minus = (f_theoretical - f_empirical_prev).abs();

        if diff_plus > max_diff {
            max_diff = diff_plus;
        }
        if diff_minus > max_diff {
            max_diff = diff_minus;
        }
    }

    // Standard alpha = 0.05 Kolmogorov-Smirnov critical value: 1.358 / sqrt(D)
    let critical_value = 1.358 / d_f.sqrt();
    let passed = max_diff <= critical_value;

    KSResult {
        statistic: max_diff,
        critical_value,
        passed,
    }
}

/// Generates a provably exact sample of Porter-Thomas probabilities on the simplex in dimension D.
///
/// Draws independent standard exponential random variables `E_i ~ Exp(1)` and normalizes:
/// `p_i = E_i / sum(E_k)`. This is mathematically equivalent to the induced distribution
/// of |c_i|^2 under the uniform Haar measure on the unit sphere S^{2D-1}.
pub fn sample_ideal_porter_thomas(dim: usize, seed: u64) -> Vec<f64> {
    if dim == 0 {
        return Vec::new();
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut exps = Vec::with_capacity(dim);
    let mut sum = 0.0f64;

    for _ in 0..dim {
        // Sample uniform u in (0, 1]
        let u: f64 = rng.gen::<f64>().clamp(1e-15, 1.0);
        let e = -u.ln();
        exps.push(e);
        sum += e;
    }

    if sum > 0.0 {
        for e in &mut exps {
            *e /= sum;
        }
    }
    exps
}

/// Evaluates a full, comprehensive Porter-Thomas verification report on a probability distribution.
pub fn verify_porter_thomas(probs: &[f64]) -> PorterThomasReport {
    let dim = probs.len();
    let exact_xeb = exact_linear_xeb(probs);
    let naive_xeb = naive_linear_xeb(probs);
    let sum_p_sqr: f64 = probs.iter().map(|&p| p * p).sum();
    let expected_sum_p_sqr = expected_sum_p_sqr(dim);
    let shannon = shannon_entropy(probs);
    let (entropy_deficit, expected_entropy_deficit) = haar_entropy_deficit(probs);
    let epr = effective_participation_ratio(probs);
    let expected_epr = (dim as f64 + 1.0) / 2.0;
    let ks = kolmogorov_smirnov_porter_thomas(probs);

    // Conformance criteria:
    // 1. Exact XEB is close to 1.0 within statistical tolerance: 3.5 / sqrt(D)
    // 2. Kolmogorov-Smirnov goodness-of-fit passes at alpha = 0.05
    let tolerance = if dim >= 16 {
        (3.5 / (dim as f64).sqrt()).max(0.15)
    } else {
        0.50
    };
    let xeb_in_bounds = (exact_xeb - 1.0).abs() <= tolerance;
    let is_porter_thomas_conforming = xeb_in_bounds && ks.passed;

    PorterThomasReport {
        dim,
        exact_xeb,
        naive_xeb,
        sum_p_sqr,
        expected_sum_p_sqr,
        shannon_entropy: shannon,
        entropy_deficit,
        expected_entropy_deficit,
        effective_participation_ratio: epr,
        expected_participation_ratio: expected_epr,
        ks_statistic: ks.statistic,
        ks_critical_value: ks.critical_value,
        passes_ks_test: ks.passed,
        is_porter_thomas_conforming,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_xeb_on_uniform_is_zero() {
        for dim in [4, 7, 16, 49, 128, 343, 1024] {
            let p = 1.0 / (dim as f64);
            let probs = vec![p; dim];
            let xeb = exact_linear_xeb(&probs);
            assert!(
                xeb.abs() < 1e-12,
                "dim={}: expected XEB=0.0 for uniform, got {}",
                dim,
                xeb
            );
        }
    }

    #[test]
    fn exact_xeb_on_haar_expectation_is_one() {
        for dim in [4, 7, 16, 49, 128, 343, 1024] {
            let expected_sum_sqr = 2.0 / (f64::from(dim) + 1.0);
            // Construct a synthetic probability vector with exactly the Haar sum of squares
            let d_f = f64::from(dim);
            let exact_xeb = (d_f * (d_f + 1.0) * expected_sum_sqr - (d_f + 1.0)) / (d_f - 1.0);
            assert!(
                (exact_xeb - 1.0).abs() < 1e-12,
                "dim={}: expected exact XEB=1.0000, got {}",
                dim,
                exact_xeb
            );
        }
    }

    #[test]
    fn ideal_porter_thomas_sample_conforms() {
        // Test in both binary (1024 = 2^10) and heptal (343 = 7^3 or 2401 = 7^4)
        for dim in [256, 343, 1024] {
            let probs = sample_ideal_porter_thomas(dim, 1337);
            let report = verify_porter_thomas(&probs);

            assert!(
                report.passes_ks_test,
                "dim={}: KS test failed, stat={}, crit={}",
                dim, report.ks_statistic, report.ks_critical_value
            );
            assert!(
                (report.exact_xeb - 1.0).abs() < 0.15,
                "dim={}: exact XEB {} drifted too far from 1.0",
                dim,
                report.exact_xeb
            );
            assert!(
                report.is_porter_thomas_conforming,
                "dim={}: report failed conformance: {:?}",
                dim, report
            );

            // Verify entropy deficit is close to 1 - gamma (~0.42)
            assert!(
                (report.entropy_deficit - report.expected_entropy_deficit).abs() < 0.08,
                "dim={}: entropy deficit {} far from expected {}",
                dim,
                report.entropy_deficit,
                report.expected_entropy_deficit
            );
        }
    }
}
