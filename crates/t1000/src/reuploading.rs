// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! A data re-uploading feature map.
//!
//! The inputs are split into chunks of one value per qubit. Upload `l` applies `Ry(scale * x_j)`
//! to qubit `p` for every input `j` of chunk `l mod n_chunks` (position `p` in the chunk), and is
//! followed by layer `l` of the circuit ansatz. The features are the Z-string expectation values of
//! the final state, so each is a trigonometric polynomial of the inputs: an input uploaded `k`
//! times enters with frequencies up to `k`, and different inputs interact to every order.
//!
//! This differs from amplitude encoding, where every expectation value is a quadratic form in the
//! normalised input vector and so cannot exceed degree 2 whatever the register size or depth.

use crate::linear_head::{z_features_into, Observables};
use crate::state::QuantumState;
use crate::CircuitAnsatz;

/// Feature map with `uploads` data uploads, each followed by one ansatz layer.
#[derive(Debug)]
pub struct ReuploadingMap<'a> {
    pub ansatz: &'a CircuitAnsatz,
    pub weights: &'a [f64],
    /// Number of uploads; must not exceed the ansatz depth.
    pub uploads: usize,
    /// Rotation angle per unit of input.
    pub angle_scale: f64,
    pub observables: Observables,
}

impl ReuploadingMap<'_> {
    pub fn n_features(&self) -> usize {
        self.observables.count(self.ansatz.n_qubits)
    }

    /// The indices of the inputs uploaded in upload `l`, for `n_inputs` inputs.
    pub fn chunk(&self, l: usize, n_inputs: usize) -> std::ops::Range<usize> {
        let n = self.ansatz.n_qubits;
        let n_chunks = n_inputs.div_ceil(n);
        let start = (l % n_chunks) * n;
        start..(start + n).min(n_inputs)
    }

    /// Runs the circuit for input `x`, leaving the final state in `state`.
    pub fn run(&self, x: &[f64], state: &mut QuantumState) {
        assert!(self.uploads <= self.ansatz.depth, "{} uploads need an ansatz of at least that depth", self.uploads);
        state.real.fill(0.0);
        state.imag.fill(0.0);
        state.real[0] = 1.0;
        for l in 0..self.uploads {
            for (position, j) in self.chunk(l, x.len()).enumerate() {
                crate::simd::apply_ry_kernel(&mut state.real, &mut state.imag, position, self.angle_scale * x[j]);
            }
            self.ansatz.forward_layer_into(l, self.weights, state);
        }
    }

    /// Feature rows for every input, reusing one scratch state.
    pub fn feature_matrix<'x>(&self, inputs: impl IntoIterator<Item = &'x [f64]>) -> Vec<Vec<f64>> {
        let mut state = QuantumState::new(1 << self.ansatz.n_qubits);
        inputs
            .into_iter()
            .map(|x| {
                self.run(x, &mut state);
                let mut row = vec![0.0; self.n_features()];
                z_features_into(&state, self.observables, &mut row);
                row
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map_for<'a>(ansatz: &'a CircuitAnsatz, weights: &'a [f64], uploads: usize) -> ReuploadingMap<'a> {
        ReuploadingMap { ansatz, weights, uploads, angle_scale: 1.0, observables: Observables::ZAndZz }
    }

    fn features(map: &ReuploadingMap, x: &[f64]) -> Vec<f64> {
        map.feature_matrix([x]).remove(0)
    }

    #[test]
    fn chunks_cover_the_inputs_cyclically() {
        let ansatz = CircuitAnsatz::new(4, 4);
        let map = map_for(&ansatz, &ansatz.weights, 4);
        assert_eq!(map.chunk(0, 6), 0..4);
        assert_eq!(map.chunk(1, 6), 4..6);
        assert_eq!(map.chunk(2, 6), 0..4);
        // 10 inputs on 4 qubits make three chunks, and upload 3 wraps to the first.
        assert_eq!(map.chunk(2, 10), 8..10);
        assert_eq!(map.chunk(3, 10), 0..4);
    }

    #[test]
    fn a_single_qubit_upload_gives_cosine_of_the_angle() {
        // One upload with a zero-weight ansatz layer: Z on the uploaded qubit is cos(angle).
        let ansatz = CircuitAnsatz::new(2, 1);
        let zero = vec![0.0; ansatz.param_count()];
        let map = map_for(&ansatz, &zero, 1);
        let f = features(&map, &[0.7, 1.9]);
        assert!((f[0] - 0.7f64.cos()).abs() < 1e-12 && (f[1] - 1.9f64.cos()).abs() < 1e-12, "{f:?}");
    }

    /// Sum over `period` equally spaced shifts of input `j` by `2 pi / period`, as a feature vector.
    fn shifted_sum(map: &ReuploadingMap, x: &[f64], j: usize, period: usize) -> Vec<f64> {
        let mut total = vec![0.0; map.n_features()];
        for k in 0..period {
            let mut y = x.to_vec();
            y[j] += std::f64::consts::TAU * k as f64 / period as f64;
            total.iter_mut().zip(features(map, &y)).for_each(|(t, f)| *t += f);
        }
        total
    }

    fn max_difference(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    }

    #[test]
    fn an_input_uploaded_once_has_frequency_at_most_one() {
        // With one upload per input, summing a feature over two shifts by pi removes the cos/sin
        // terms, leaving a value independent of the input's own angle.
        let ansatz = CircuitAnsatz::new(4, 2);
        let map = map_for(&ansatz, &ansatz.weights, 2);
        let x = [0.4, 1.1, 0.8, 0.3, 0.9, 0.5];
        for j in 0..6 {
            let mut other = x;
            other[j] += 0.7;
            let (a, b) = (shifted_sum(&map, &x, j, 2), shifted_sum(&map, &other, j, 2));
            assert!(max_difference(&a, &b) < 1e-10, "input {j} enters with a frequency above 1");
        }
    }

    #[test]
    fn uploading_an_input_twice_raises_its_frequency_to_two() {
        // Four uploads cover every input twice: the two-shift sum no longer removes the dependence
        // (frequency 2), but a four-shift sum does (frequencies 1 to 3 cancel).
        let ansatz = CircuitAnsatz::new(4, 4);
        let map = map_for(&ansatz, &ansatz.weights, 4);
        let x = [0.4, 1.1, 0.8, 0.3, 0.9, 0.5];
        let mut worst_two_shift = 0.0f64;
        for j in 0..6 {
            let mut other = x;
            other[j] += 0.7;
            worst_two_shift = worst_two_shift.max(max_difference(&shifted_sum(&map, &x, j, 2), &shifted_sum(&map, &other, j, 2)));
            assert!(max_difference(&shifted_sum(&map, &x, j, 4), &shifted_sum(&map, &other, j, 4)) < 1e-10, "input {j} has a frequency above 3");
        }
        assert!(worst_two_shift > 1e-3, "re-uploading should add a frequency-2 term, got {worst_two_shift:e}");
    }

    #[test]
    #[should_panic(expected = "uploads need an ansatz")]
    fn more_uploads_than_layers_is_rejected() {
        let ansatz = CircuitAnsatz::new(3, 2);
        let map = map_for(&ansatz, &ansatz.weights, 3);
        features(&map, &[0.1, 0.2, 0.3]);
    }
}
