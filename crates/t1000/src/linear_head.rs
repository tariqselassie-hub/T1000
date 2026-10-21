// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! A fixed circuit as a feature map with a closed-form linear head.
//!
//! The circuit maps a molecule to a state; the features are the Z and ZZ expectation values
//! of that state (all of them come from one computational-basis measurement), and a ridge
//! regression fitted on the training set reads them out. No gradient training is involved, so
//! the head is exactly reproducible.

use crate::baselines::{fit_ridge_validated, RidgeRegression, RIDGE_LAMBDAS};
use crate::{CircuitAnsatz, DataEncodingLayer, QuantumState};

/// Largest register supported by the stack-allocated sign table.
const MAX_QUBITS: usize = 32;

/// Which Pauli-Z expectation values form the feature vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Observables {
    /// `Z_q` only: `n` features.
    SingleZ,
    /// `Z_q` followed by `Z_q Z_r` for `q < r`: `n + n(n-1)/2` features.
    ZAndZz,
}

impl Observables {
    pub fn count(self, n_qubits: usize) -> usize {
        match self {
            Self::SingleZ => n_qubits,
            Self::ZAndZz => n_qubits + n_qubits * (n_qubits - 1) / 2,
        }
    }
}

/// Writes the expectation values of `state` into `out` (`out.len() == kind.count(n)`).
pub fn z_features_into(state: &QuantumState, kind: Observables, out: &mut [f64]) {
    let n = state.dim.trailing_zeros() as usize;
    assert!(n <= MAX_QUBITS && 1usize << n == state.dim, "state must hold 2^n amplitudes");
    assert_eq!(out.len(), kind.count(n), "feature buffer has the wrong length");
    out.fill(0.0);
    let mut sign = [0.0f64; MAX_QUBITS];
    for i in 0..state.dim {
        let p = state.real[i] * state.real[i] + state.imag[i] * state.imag[i];
        if p == 0.0 {
            continue;
        }
        for (q, s) in sign.iter_mut().enumerate().take(n) {
            *s = 1.0 - 2.0 * ((i >> q) & 1) as f64;
        }
        out[..n].iter_mut().zip(&sign).for_each(|(o, s)| *o += p * s);
        if kind == Observables::ZAndZz {
            let mut k = n;
            for q in 0..n {
                for r in q + 1..n {
                    out[k] += p * sign[q] * sign[r];
                    k += 1;
                }
            }
        }
    }
}

/// Encodes an input, runs the circuit and reads out the features.
#[derive(Debug)]
pub struct CircuitFeatureMap<'a> {
    pub encoding: &'a DataEncodingLayer,
    pub ansatz: &'a CircuitAnsatz,
    pub weights: &'a [f64],
    pub observables: Observables,
}

impl CircuitFeatureMap<'_> {
    pub fn n_features(&self) -> usize {
        self.observables.count(self.encoding.n_qubits)
    }

    /// Feature rows for every input, reusing one scratch state.
    pub fn feature_matrix<'x>(&self, inputs: impl IntoIterator<Item = &'x [f64]>) -> Vec<Vec<f64>> {
        let mut scratch = QuantumState::new(self.encoding.dim);
        inputs
            .into_iter()
            .map(|x| {
                self.ansatz.forward_into(&self.encoding.encode(x), self.weights, &mut scratch);
                let mut row = vec![0.0; self.n_features()];
                z_features_into(&scratch, self.observables, &mut row);
                row
            })
            .collect()
    }
}

fn standardize(row: &[f64], center: &[f64], spread: &[f64]) -> Vec<f64> {
    row.iter().zip(center).zip(spread).map(|((v, c), s)| (v - c) / s).collect()
}

/// Standardisation followed by a ridge regression: the complete, self-contained readout.
///
/// Circuit features can sit near a constant (a Z value near -1 with a spread of 1e-4), and the
/// shared ridge solver forms its Gram matrix from raw second moments, which loses precision for
/// such columns and can make a small penalty numerically singular. Centering and scaling here first
/// keeps the solver's inputs well conditioned and makes a saved head reusable on its own.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LinearHead {
    pub center: Vec<f64>,
    pub spread: Vec<f64>,
    pub ridge: RidgeRegression,
}

impl LinearHead {
    /// Fits on `features` (rows) and `targets`, choosing the penalty on a validation split.
    pub fn fit(features: &[Vec<f64>], targets: &[Vec<f64>]) -> Self {
        let n = features.len() as f64;
        let d = features[0].len();
        let center: Vec<f64> = (0..d).map(|j| features.iter().map(|r| r[j]).sum::<f64>() / n).collect();
        let spread: Vec<f64> = (0..d)
            .map(|j| {
                let var = features.iter().map(|r| (r[j] - center[j]).powi(2)).sum::<f64>() / n;
                if var > 1e-24 { var.sqrt() } else { 1.0 }
            })
            .collect();
        let scaled: Vec<Vec<f64>> = features.iter().map(|r| standardize(r, &center, &spread)).collect();
        let ridge = fit_ridge_validated(&scaled, targets, &RIDGE_LAMBDAS);
        Self { center, spread, ridge }
    }

    pub fn predict_one(&self, row: &[f64]) -> Vec<f64> {
        self.ridge.predict_one(&standardize(row, &self.center, &self.spread))
    }

    pub fn predict(&self, rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
        rows.iter().map(|r| self.predict_one(r)).collect()
    }

    /// The ridge penalty chosen on the validation split.
    pub fn lambda(&self) -> f64 {
        self.ridge.lambda
    }
}

/// Standard deviation of every feature column across the rows.
pub fn feature_spreads(rows: &[Vec<f64>]) -> Vec<f64> {
    let n = rows.len() as f64;
    (0..rows[0].len())
        .map(|j| {
            let mean = rows.iter().map(|r| r[j]).sum::<f64>() / n;
            (rows.iter().map(|r| (r[j] - mean).powi(2)).sum::<f64>() / n).sqrt()
        })
        .collect()
}

/// Shots per molecule at which sampling noise `1 / sqrt(shots)` equals `spread`.
pub fn shots_for_noise_equal_to(spread: f64) -> f64 {
    1.0 / (spread * spread).max(1e-300)
}

/// Short stable fingerprint of a weight vector (FNV-1a over the bit patterns).
pub fn weights_fingerprint(weights: &[f64]) -> String {
    let hash = weights
        .iter()
        .flat_map(|w| w.to_bits().to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3));
    format!("{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn basis(n: usize, index: usize) -> QuantumState {
        let mut s = QuantumState::zero(1 << n);
        s.real[index] = 1.0;
        s
    }

    #[test]
    fn basis_state_expectations_are_the_bit_signs() {
        let n = 3;
        let mut out = vec![0.0; Observables::ZAndZz.count(n)];
        z_features_into(&basis(n, 0b001), Observables::ZAndZz, &mut out);
        // Z0 = -1, Z1 = Z2 = +1; ZZ pairs in order (0,1), (0,2), (1,2).
        assert_eq!(out, vec![-1.0, 1.0, 1.0, -1.0, -1.0, 1.0]);
    }

    #[test]
    fn uniform_superposition_has_zero_expectations() {
        let n = 3;
        let mut s = QuantumState::zero(1 << n);
        s.real.fill((1.0f64 / 8.0).sqrt());
        let mut out = vec![1.0; Observables::ZAndZz.count(n)];
        z_features_into(&s, Observables::ZAndZz, &mut out);
        assert!(out.iter().all(|v| v.abs() < 1e-12));
    }

    #[test]
    fn single_z_is_the_leading_block_of_the_full_set() {
        let n = 4;
        let s = basis(n, 0b0110);
        let (mut full, mut z) = (vec![0.0; Observables::ZAndZz.count(n)], vec![0.0; n]);
        z_features_into(&s, Observables::ZAndZz, &mut full);
        z_features_into(&s, Observables::SingleZ, &mut z);
        assert_eq!(&full[..n], &z[..]);
    }

    #[test]
    fn spreads_and_shot_estimate() {
        let rows = vec![vec![1.0, 5.0], vec![3.0, 5.0]];
        assert_eq!(feature_spreads(&rows), vec![1.0, 0.0]);
        assert_eq!(shots_for_noise_equal_to(0.01), 1.0e4);
    }

    #[test]
    fn fingerprint_changes_with_any_weight() {
        let a = weights_fingerprint(&[0.1, 0.2, 0.3]);
        assert_eq!(a, weights_fingerprint(&[0.1, 0.2, 0.3]));
        assert_ne!(a, weights_fingerprint(&[0.1, 0.2, 0.3000000001]));
    }
}
