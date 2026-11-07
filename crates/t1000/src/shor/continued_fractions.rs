// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Continued Fractions Algorithm & Order Extraction
//!
//! Converts measured phase estimates $\phi \in [0, 1)$ into continued fraction
//! convergents $p_k / q_k \approx \phi$, identifying candidate order $r = q_k$.

use super::modular_arithmetic::mod_pow;
use serde::{Deserialize, Serialize};

/// Convergent rational fraction $p / q$.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Convergent {
    pub numerator: u64,
    pub denominator: u64,
}

/// Continued fraction expansion and convergent generator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuedFractions;

impl ContinuedFractions {
    /// Computes convergents of phase $\phi$ up to denominator ceiling `max_denominator`.
    pub fn convergents(phase: f64, max_denominator: u64) -> Vec<Convergent> {
        let mut convergents = Vec::new();
        let mut x = phase;

        let mut p_prev2 = 0i128;
        let mut p_prev1 = 1i128;
        let mut q_prev2 = 1i128;
        let mut q_prev1 = 0i128;

        for _ in 0..64 {
            let a = x.floor() as i128;
            let p = a * p_prev1 + p_prev2;
            let q = a * q_prev1 + q_prev2;

            if q > max_denominator as i128 {
                break;
            }

            if q > 0 {
                convergents.push(Convergent {
                    numerator: p as u64,
                    denominator: q as u64,
                });
            }

            let remainder = x - x.floor();
            if remainder.abs() < 1e-12 {
                break;
            }
            x = 1.0 / remainder;

            p_prev2 = p_prev1;
            p_prev1 = p;
            q_prev2 = q_prev1;
            q_prev1 = q;
        }

        convergents
    }

    /// Extracts candidate multiplicative order $r$ from measured phase for base $a$ and modulus $N$.
    pub fn extract_order(phase: f64, a: u64, modulus: u64) -> Option<u64> {
        let convs = Self::convergents(phase, modulus);

        for conv in &convs {
            let r = conv.denominator;
            if r > 0 && r < modulus && mod_pow(a, r, modulus) == 1 {
                return Some(r);
            }
            // Check small multiples m * r up to modulus (in case gcd(s, r) > 1 canceled)
            for m in 2..=8 {
                let mr = r * m;
                if mr < modulus && mod_pow(a, mr, modulus) == 1 {
                    return Some(mr);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continued_fractions_convergents() {
        // Phase 0.25 = 1/4
        let convs = ContinuedFractions::convergents(0.25, 15);
        assert!(convs.iter().any(|c| c.numerator == 1 && c.denominator == 4));

        // Phase 0.75 = 3/4
        let convs = ContinuedFractions::convergents(0.75, 15);
        assert!(convs.iter().any(|c| c.numerator == 3 && c.denominator == 4));

        // Phase 0.3333333 approx 1/3
        let convs = ContinuedFractions::convergents(1.0 / 3.0, 20);
        assert!(convs.iter().any(|c| c.numerator == 1 && c.denominator == 3));
    }

    #[test]
    fn test_order_extraction_from_phase() {
        // For a=7, N=15, order is r=4.
        // QPE measures phases s / 4 in {0.0, 0.25, 0.50, 0.75}
        assert_eq!(ContinuedFractions::extract_order(0.25, 7, 15), Some(4));
        assert_eq!(ContinuedFractions::extract_order(0.75, 7, 15), Some(4));
        assert_eq!(ContinuedFractions::extract_order(0.50, 7, 15), Some(4));
    }
}
