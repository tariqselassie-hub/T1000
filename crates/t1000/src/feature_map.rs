// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

use crate::state::QuantumState;
use serde::{Deserialize, Serialize};

/// Canonical Quantum Feature Map Encoding Strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncodingKind {
    /// Each feature x_i specifies rotation angle Ry(x_i) on qubit i.
    Angle,
    /// Compresses 2^n normalized features directly into amplitudes of n qubits.
    Amplitude,
}

/// Data Encoding Layer mapping classical vectors into quantum states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataEncodingLayer {
    pub n_qubits: usize,
    pub dim: usize,
    pub kind: EncodingKind,
}

impl DataEncodingLayer {
    pub fn new(n_qubits: usize, kind: EncodingKind) -> Self {
        assert!(
            (1..usize::BITS as usize).contains(&n_qubits),
            "DataEncodingLayer needs between 1 and {} qubits (got {})",
            usize::BITS - 1,
            n_qubits
        );
        let dim = 1 << n_qubits;
        Self {
            n_qubits,
            dim,
            kind,
        }
    }

    /// Embed classical feature vector `x` into quantum state |psi(x)>.
    pub fn encode(&self, x: &[f64]) -> QuantumState {
        match self.kind {
            EncodingKind::Angle => self.encode_angle(x),
            EncodingKind::Amplitude => self.encode_amplitude(x),
        }
    }

    /// Angle Encoding: applies Ry(x_i) to each qubit i initialized in |0>.
    /// State on qubit i: cos(x_i / 2) |0> + sin(x_i / 2) |1>
    /// Joint state: tensor product over all n qubits.
    fn encode_angle(&self, x: &[f64]) -> QuantumState {
        let mut real = vec![0.0; self.dim];
        let imag = vec![0.0; self.dim];

        // Precompute cos(x_i/2) and sin(x_i/2) for each qubit
        let mut cos_sin = Vec::with_capacity(self.n_qubits);
        for i in 0..self.n_qubits {
            let angle = if i < x.len() { x[i] } else { 0.0 };
            let (s, c) = (angle * 0.5).sin_cos();
            cos_sin.push((c, s));
        }

        for (idx, r) in real.iter_mut().enumerate() {
            let mut amp = 1.0;
            for (q, &(c, s)) in cos_sin.iter().enumerate() {
                let bit = (idx >> q) & 1;
                amp *= if bit == 0 { c } else { s };
            }
            *r = amp;
        }

        QuantumState {
            dim: self.dim,
            real: real.into(),
            imag: imag.into(),
        }
    }

    /// Amplitude Encoding: normalizes vector x and assigns directly to state amplitudes.
    fn encode_amplitude(&self, x: &[f64]) -> QuantumState {
        let mut real = vec![0.0; self.dim];
        let imag = vec![0.0; self.dim];
        // Features beyond 2^n are truncated, so only the kept prefix contributes to the norm.
        let norm_sq: f64 = x.iter().take(self.dim).map(|v| v * v).sum();
        let norm = norm_sq.sqrt().max(1e-12);

        for (idx, &val) in x.iter().enumerate().take(self.dim) {
            real[idx] = val / norm;
        }

        QuantumState {
            dim: self.dim,
            real: real.into(),
            imag: imag.into(),
        }
    }

    /// Zero-allocation in-place encoding into pre-allocated state buffer.
    pub fn encode_into(&self, x: &[f64], state: &mut QuantumState) {
        assert_eq!(self.dim, state.dim);
        match self.kind {
            EncodingKind::Angle => self.encode_angle_into(x, state),
            EncodingKind::Amplitude => self.encode_amplitude_into(x, state),
        }
    }

    /// Angle encoding written directly into `state`, with no intermediate statevector.
    fn encode_angle_into(&self, x: &[f64], state: &mut QuantumState) {
        let mut cos_sin = [(1.0f64, 0.0f64); 64];
        assert!(
            self.n_qubits <= cos_sin.len(),
            "{0} qubits exceed the {1}-entry angle table",
            self.n_qubits,
            cos_sin.len()
        );
        for (i, cs) in cos_sin.iter_mut().enumerate().take(self.n_qubits) {
            let angle = if i < x.len() { x[i] } else { 0.0 };
            let (s, c) = (angle * 0.5).sin_cos();
            *cs = (c, s);
        }

        for (idx, r) in state.real.iter_mut().enumerate() {
            let mut amp = 1.0;
            for (q, &(c, s)) in cos_sin.iter().enumerate().take(self.n_qubits) {
                amp *= if (idx >> q) & 1 == 0 { c } else { s };
            }
            *r = amp;
        }
        state.imag.fill(0.0);
    }

    /// Zero-allocation in-place amplitude encoding directly into target statevector:
    pub fn encode_amplitude_into(&self, x: &[f64], state: &mut QuantumState) {
        assert_eq!(self.dim, state.dim);
        state.real.fill(0.0);
        state.imag.fill(0.0);
        // Features beyond 2^n are truncated, so only the kept prefix contributes to the norm.
        let norm_sq: f64 = x.iter().take(self.dim).map(|v| v * v).sum();
        let norm = norm_sq.sqrt().max(1e-12);
        let inv_norm = 1.0 / norm;

        for (idx, &val) in x.iter().enumerate().take(self.dim) {
            state.real[idx] = val * inv_norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_same(a: &QuantumState, b: &QuantumState, context: &str) {
        assert_eq!(a.dim, b.dim);
        for k in 0..a.dim {
            assert!(
                (a.real[k] - b.real[k]).abs() < 1e-15 && (a.imag[k] - b.imag[k]).abs() < 1e-15,
                "{context}: amplitude {k} differs ({} vs {})",
                b.real[k],
                a.real[k]
            );
        }
    }

    #[test]
    fn encode_into_matches_allocating_encode() {
        for n_qubits in [1usize, 3, 6, 9] {
            for kind in [EncodingKind::Angle, EncodingKind::Amplitude] {
                let layer = DataEncodingLayer::new(n_qubits, kind);
                let dim = layer.dim;
                let mut buf = QuantumState::new(dim);
                // Fill the buffer with junk first: encode_into must fully overwrite it.
                for k in 0..dim {
                    buf.real[k] = 0.25 * k as f64;
                    buf.imag[k] = -0.5 * k as f64;
                }
                let exact: Vec<f64> = (0..n_qubits).map(|q| 0.4 * q as f64 - 0.3).collect();
                let short: Vec<f64> = exact.iter().take(n_qubits / 2).copied().collect();
                let mut long = exact.clone();
                long.extend([1.5, -2.5]);
                for (case, x) in ["exact", "short", "long"]
                    .into_iter()
                    .zip([&exact, &short, &long])
                {
                    layer.encode_into(x, &mut buf);
                    assert_same(
                        &layer.encode(x),
                        &buf,
                        &format!("{n_qubits}q {kind:?} {case}"),
                    );
                }
            }
        }
    }
}
