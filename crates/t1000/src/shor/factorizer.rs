// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # End-to-End Shor Factorization Engine
//!
//! Orchestrates quantum order finding, continued fractions extraction,
//! and Euclidean GCD post-processing to deterministically factor composite
//! integers into non-trivial prime factors.

use super::modular_arithmetic::{gcd, mod_pow};
use super::order_finding::{QuantumOrderFinder, QuantumOrderResult};
use serde::{Deserialize, Serialize};

/// Comprehensive Shor Factorization Report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShorFactorizationReport {
    /// Target composite integer $N = p \cdot q$.
    pub composite_n: u64,
    /// Extracted factor $p$.
    pub factor_p: u64,
    /// Extracted factor $q$.
    pub factor_q: u64,
    /// Multiplier base $a$ used.
    pub base_a: u64,
    /// Quantum measured order $r$.
    pub order_r: u64,
    /// Total circuit qubits (counting + work).
    pub total_qubits: usize,
    /// Verification flag: $p \cdot q == N$.
    pub factorization_verified: bool,
    /// Quantum circuit execution time in microseconds.
    pub quantum_circuit_micros: u64,
    /// Classical trial division time in microseconds.
    pub classical_trial_micros: u64,
    /// QPE measured phases and probability peaks.
    pub qpe_result: QuantumOrderResult,
}

/// Shor Factorization Engine.
#[derive(Debug, Clone, Default)]
pub struct ShorFactorizer;

impl ShorFactorizer {
    /// Factors composite integer $N$ using Shor's quantum algorithm.
    pub fn factor(modulus: u64) -> Option<ShorFactorizationReport> {
        assert!(modulus >= 4, "Modulus must be >= 4");

        // Edge case: even modulus
        if modulus % 2 == 0 {
            return None;
        }

        let t_class_start = std::time::Instant::now();
        // Classical trial division benchmark for comparison
        let mut class_p = 0;
        for d in (3..modulus).step_by(2) {
            if d * d > modulus {
                break;
            }
            if modulus % d == 0 {
                class_p = d;
                break;
            }
        }
        let classical_trial_micros = t_class_start.elapsed().as_micros() as u64;
        let _ = class_p;

        // Candidate coprime bases for order finding
        let candidate_bases = [7, 2, 8, 11, 13, 4, 14, 17, 19, 23, 29];

        for &a in &candidate_bases {
            if a >= modulus {
                continue;
            }

            let g = gcd(a, modulus);
            if g > 1 && g < modulus {
                // Lucky classical GCD factor
                continue;
            }

            // Run quantum order finding
            let finder = QuantumOrderFinder::new(a, modulus);
            let qpe_res = finder.find_order();

            if let Some(r) = qpe_res.order {
                if r % 2 == 0 {
                    let half_r = r / 2;
                    let x = mod_pow(a, half_r, modulus);

                    if x != modulus - 1 && x != 1 {
                        let p1 = gcd(x + 1, modulus);
                        let p2 = gcd(if x > 0 { x - 1 } else { modulus - 1 }, modulus);

                        let p = if p1 > 1 && p1 < modulus {
                            p1
                        } else if p2 > 1 && p2 < modulus {
                            p2
                        } else {
                            continue;
                        };

                        let q = modulus / p;
                        let verified = p * q == modulus && p > 1 && q > 1;

                        return Some(ShorFactorizationReport {
                            composite_n: modulus,
                            factor_p: p.min(q),
                            factor_q: p.max(q),
                            base_a: a,
                            order_r: r,
                            total_qubits: qpe_res.total_qubits,
                            factorization_verified: verified,
                            quantum_circuit_micros: qpe_res.circuit_micros,
                            classical_trial_micros,
                            qpe_result: qpe_res,
                        });
                    }
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
    fn test_factor_15() {
        // N = 15 = 3 * 5
        let report = ShorFactorizer::factor(15).expect("Failed to factor 15");
        assert!(report.factorization_verified);
        assert_eq!(report.factor_p, 3);
        assert_eq!(report.factor_q, 5);
        assert_eq!(report.factor_p * report.factor_q, 15);
    }

    #[test]
    fn test_factor_21() {
        // N = 21 = 3 * 7
        let report = ShorFactorizer::factor(21).expect("Failed to factor 21");
        assert!(report.factorization_verified);
        assert_eq!(report.factor_p, 3);
        assert_eq!(report.factor_q, 7);
        assert_eq!(report.factor_p * report.factor_q, 21);
    }

    #[test]
    fn test_factor_35() {
        // N = 35 = 5 * 7
        let report = ShorFactorizer::factor(35).expect("Failed to factor 35");
        assert!(report.factorization_verified);
        assert_eq!(report.factor_p, 5);
        assert_eq!(report.factor_q, 7);
        assert_eq!(report.factor_p * report.factor_q, 35);
    }
}

