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
    ///
    /// Automatically routes to the unified semiclassical zipper engine when
    /// the modulus exceeds 16 counting+work qubits to prevent exponential
    /// $2^{3n}$ RAM exhaustion.
    pub fn factor(modulus: u64) -> Option<ShorFactorizationReport> {
        assert!(modulus >= 4, "Modulus must be >= 4");
        if modulus % 2 == 0 {
            return None;
        }

        let t_class_start = std::time::Instant::now();
        let classical_trial_micros = t_class_start.elapsed().as_micros() as u64;

        let mut nw = 1;
        while (1u64 << nw) <= modulus {
            nw += 1;
        }

        // If textbook QPE requires > 16 qubits, dispatch directly to Zipper
        if nw * 3 > 16 {
            return Self::factor_with_zipper(modulus, nw, classical_trial_micros);
        }

        Self::factor_with_textbook_qpe(modulus, classical_trial_micros)
    }

    /// Helper for textbook QPE factorization on small moduli.
    fn factor_with_textbook_qpe(modulus: u64, classical_trial_micros: u64) -> Option<ShorFactorizationReport> {
        let candidate_bases = [7, 2, 8, 11, 13, 4, 14, 17, 19, 23, 29];
        for &a in &candidate_bases {
            if a >= modulus || gcd(a, modulus) > 1 {
                continue;
            }

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

    /// Helper for unified semiclassical zipper factorization.
    fn factor_with_zipper(
        modulus: u64,
        nw: usize,
        classical_trial_micros: u64,
    ) -> Option<ShorFactorizationReport> {
        let zip_rep = super::zipper_factoring::ZipperShorFactorizer::factor(modulus, 0x5EED_0042)?;
        let p = zip_rep.factor_p?;
        let q = zip_rep.factor_q?;
        let r = zip_rep.order_r?;

        Some(ShorFactorizationReport {
            composite_n: modulus,
            factor_p: p.min(q),
            factor_q: p.max(q),
            base_a: zip_rep.base_a,
            order_r: r,
            total_qubits: zip_rep.total_active_qubits,
            factorization_verified: zip_rep.factoring_verified,
            quantum_circuit_micros: zip_rep.wall_time_micros,
            classical_trial_micros,
            qpe_result: QuantumOrderResult {
                a: zip_rep.base_a,
                modulus,
                counting_qubits: zip_rep.counting_qubits,
                work_qubits: nw,
                total_qubits: zip_rep.total_active_qubits,
                order: Some(r),
                measured_phases: vec![zip_rep.estimated_phase],
                peak_probabilities: vec![zip_rep.mean_coherence],
                quantum_success: zip_rep.factoring_verified,
                circuit_micros: zip_rep.wall_time_micros,
            },
        })
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

    #[test]
    fn test_factor_77_zipper_default() {
        // N = 77 = 7 * 11 (requires nw = 7, textbook QPE = 21 qubits -> routes to zipper)
        let report = ShorFactorizer::factor(77).expect("Failed to factor 77");
        assert!(report.factorization_verified);
        assert_eq!(report.factor_p, 7);
        assert_eq!(report.factor_q, 11);
        assert_eq!(report.factor_p * report.factor_q, 77);
    }

    #[test]
    fn test_factor_221_zipper_default() {
        // N = 221 = 13 * 17 (nw = 8, textbook QPE = 24 qubits -> routes to zipper)
        let report = ShorFactorizer::factor(221).expect("Failed to factor 221");
        assert!(report.factorization_verified);
        assert_eq!(report.factor_p, 13);
        assert_eq!(report.factor_q, 17);
        assert_eq!(report.factor_p * report.factor_q, 221);
    }
}

