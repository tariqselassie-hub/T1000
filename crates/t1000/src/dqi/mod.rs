// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Decoded Quantum Interferometry (DQI) Subsystem
//!
//! Provides the complete mathematical and computational implementation of
//! Decoded Quantum Interferometry (Target 01 from *The Compendium of Super-Quadratic
//! Computational Quantum Advantages*, arXiv:2610.32595).
//!
//! DQI bridges the Quantum Fourier Transform with classical dual-space error-correcting
//! codes to decode Reed-Solomon codewords and solve Optimal Polynomial Intersection (OPI)
//! beyond the classical unique decoding radius $\lfloor(n-k)/2\rfloor$ and into the
//! classically NP-hard Maximum-LINSAT regime.

pub mod field;
pub mod interferometer;
pub mod reed_solomon;

pub use field::{field_dot, field_nullspace, FieldElement};
pub use interferometer::{DqiInterferenceReport, DqiInterferometer, DqiStatevector};
pub use reed_solomon::{ReedSolomonCode, RsBruteForceResult};

use serde::{Deserialize, Serialize};

/// High-level showdown benchmark report comparing Classical vs Quantum decoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DqiShowdownReport {
    /// Field characteristic $p$.
    pub p: u32,
    /// Block length $n$.
    pub n: usize,
    /// Message dimension $k$.
    pub k: usize,
    /// Number of injected errors $t$.
    pub errors: usize,
    /// Unique decoding threshold $\lfloor(n-k)/2\rfloor$.
    pub unique_threshold: usize,
    /// Johnson / Guruswami-Sudan bound $n - \sqrt{nk}$.
    pub guruswami_sudan_bound: f64,
    /// Classical Berlekamp-Welch success flag.
    pub classical_bw_success: bool,
    /// Classical brute force candidate evaluation count $p^k$.
    pub classical_search_space: usize,
    /// Classical brute force wall-clock duration in microseconds.
    pub classical_search_micros: u64,
    /// Quantum DQI success flag.
    pub dqi_success: bool,
    /// Quantum DQI constructive amplification ratio.
    pub dqi_amplification: f64,
    /// Planted codeword probability under DQI.
    pub planted_codeword_prob: f64,
    /// Quantum circuit execution wall-clock duration in microseconds.
    pub dqi_circuit_micros: u64,
    /// Agreement of DQI top candidate with noisy input.
    pub agreement_count: usize,
}

/// Executes a comparative showdown between Classical decoders and Quantum DQI.
pub fn run_dqi_showdown(
    p: u32,
    n: usize,
    k: usize,
    errors: usize,
    planted_poly: &[FieldElement],
) -> DqiShowdownReport {
    let code = ReedSolomonCode::new(p, n, k);
    let planted_codeword = code.encode(planted_poly);

    // Inject deterministic error pattern of weight `errors`
    let mut received = planted_codeword.clone();
    let num_err = errors.min(n);
    for (i, r) in received.iter_mut().enumerate().take(num_err) {
        let err_val = FieldElement::new((i as u32 % (p - 1)) + 1, p);
        *r = r.add(err_val);
    }

    // 1. Classical Berlekamp-Welch
    let bw_res = code.decode_berlekamp_welch(&received);
    let classical_bw_success = if let Some(poly) = bw_res {
        poly == planted_poly
    } else {
        false
    };

    // 2. Classical Brute Force Search
    let t0 = std::time::Instant::now();
    let bf_res = code.decode_brute_force(&received);
    let classical_search_micros = t0.elapsed().as_micros() as u64;

    // 3. Quantum DQI
    let interferometer = DqiInterferometer::new(code.clone());
    let t1 = std::time::Instant::now();
    let report = interferometer.run_dqi(&received, &planted_codeword, planted_poly, errors);
    let dqi_circuit_micros = t1.elapsed().as_micros() as u64;

    let gs_bound = (n as f64) - ((n * k) as f64).sqrt();

    DqiShowdownReport {
        p,
        n,
        k,
        errors,
        unique_threshold: code.unique_decoding_radius(),
        guruswami_sudan_bound: gs_bound,
        classical_bw_success,
        classical_search_space: bf_res.total_candidates_searched,
        classical_search_micros,
        dqi_success: report.dqi_success,
        dqi_amplification: report.amplification_factor,
        planted_codeword_prob: report.planted_codeword_prob,
        dqi_circuit_micros,
        agreement_count: report.candidate_agreement,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dqi_showdown_execution() {
        let p = 7;
        let n = 5;
        let k = 2;
        let planted_poly = vec![FieldElement::new(4, p), FieldElement::new(1, p)];

        // Test with 2 errors (beyond unique decoding threshold = 1)
        let report = run_dqi_showdown(p, n, k, 2, &planted_poly);

        assert_eq!(report.unique_threshold, 1);
        assert!(!report.classical_bw_success);
        assert!(report.dqi_success);
        assert!(report.dqi_amplification > 1.0);
    }
}
