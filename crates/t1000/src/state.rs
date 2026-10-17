use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use t1000_core::QubitRegister;

/// Panics unless `dim` is a power of two >= 2 (i.e. a whole number of qubits).
/// The SIMD gate kernels rely on this layout invariant.
fn assert_valid_dim(dim: usize) {
    assert!(
        dim >= 2 && dim.is_power_of_two(),
        "Hilbert space dimension must be a power of two >= 2 (got {})",
        dim
    );
}

/// High-dimensional Hilbert space state vector.
/// Structure-of-Arrays (SOA) memory layout enabling contiguous 256-bit AVX2 vectorization.
/// Dimension is always a power of two (n qubits => 2^n amplitudes).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumState {
    pub dim: usize,
    pub real: Vec<f64>,
    pub imag: Vec<f64>,
}

impl QuantumState {
    /// Initialize state in uniform equal superposition |+>^n:
    /// Real amplitude = 1 / sqrt(dim), Imaginary = 0.0
    pub fn new(dim: usize) -> Self {
        assert_valid_dim(dim);
        let base_amp = 1.0 / (dim as f64).sqrt();
        Self {
            dim,
            real: vec![base_amp; dim],
            imag: vec![0.0; dim],
        }
    }

    /// Initialize empty zero state:
    pub fn zero(dim: usize) -> Self {
        assert_valid_dim(dim);
        Self {
            dim,
            real: vec![0.0; dim],
            imag: vec![0.0; dim],
        }
    }

    /// Initialize computational basis ground state |0...0>:
    /// Real amplitude of index 0 is 1.0; all others are 0.0.
    pub fn ground(dim: usize) -> Self {
        assert_valid_dim(dim);
        let mut real = vec![0.0; dim];
        real[0] = 1.0;
        Self {
            dim,
            real,
            imag: vec![0.0; dim],
        }
    }

    /// Construct state from interleaved (real, imag) amplitude slice:
    pub fn from_amplitudes(dim: usize, amps: &[(f64, f64)]) -> Self {
        assert_valid_dim(dim);
        assert_eq!(dim, amps.len());
        let mut real = Vec::with_capacity(dim);
        let mut imag = Vec::with_capacity(dim);
        for &(r, i) in amps {
            real.push(r);
            imag.push(i);
        }
        Self { dim, real, imag }
    }

    #[inline(always)]
    pub fn amplitude(&self, idx: usize) -> (f64, f64) {
        (self.real[idx], self.imag[idx])
    }

    #[inline(always)]
    pub fn set_amplitude(&mut self, idx: usize, r: f64, i: f64) {
        self.real[idx] = r;
        self.imag[idx] = i;
    }

    /// Returns interleaved (real, imag) vector for legacy compatibility.
    pub fn amplitudes(&self) -> Vec<(f64, f64)> {
        self.real
            .iter()
            .zip(self.imag.iter())
            .map(|(&r, &i)| (r, i))
            .collect()
    }

    /// In-place copy without heap re-allocations:
    #[inline(always)]
    pub fn copy_from(&mut self, other: &Self) {
        assert_eq!(self.dim, other.dim);
        self.real.copy_from_slice(&other.real);
        self.imag.copy_from_slice(&other.imag);
    }

    pub fn to_register(&self, seed: u64) -> QubitRegister {
        let mut reg = QubitRegister::new(self.dim, seed);
        reg.amplitudes = self
            .real
            .iter()
            .zip(self.imag.iter())
            .map(|(&r, &i)| Complex64::new(r, i))
            .collect();
        reg.normalize();
        reg
    }

    pub fn from_register(reg: &QubitRegister) -> Self {
        assert_valid_dim(reg.basis);
        let mut real = Vec::with_capacity(reg.basis);
        let mut imag = Vec::with_capacity(reg.basis);
        for a in &reg.amplitudes {
            real.push(a.re);
            imag.push(a.im);
        }
        Self {
            dim: reg.basis,
            real,
            imag,
        }
    }

    /// Converts the SOA amplitudes into an interleaved `Vec<Complex64>`.
    pub fn to_complex_vec(&self) -> Vec<Complex64> {
        self.real
            .iter()
            .zip(self.imag.iter())
            .map(|(&r, &i)| Complex64::new(r, i))
            .collect()
    }

    /// Constructs a `QuantumState` directly from a slice of `Complex64` amplitudes.
    pub fn from_complex_slice(slice: &[Complex64]) -> Self {
        let dim = slice.len();
        assert_valid_dim(dim);
        let mut real = Vec::with_capacity(dim);
        let mut imag = Vec::with_capacity(dim);
        for c in slice {
            real.push(c.re);
            imag.push(c.im);
        }
        Self { dim, real, imag }
    }

    /// Traces out subsystem B using `t1000_core::PartialTrace`, returning the reduced density matrix of subsystem A.
    pub fn partial_trace_b(&self, dim_a: usize, dim_b: usize) -> Vec<Complex64> {
        assert_eq!(
            self.dim,
            dim_a * dim_b,
            "State dimension must equal dim_a * dim_b"
        );
        let complex_state = self.to_complex_vec();
        t1000_core::PartialTrace::trace_out_b(&complex_state, dim_a, dim_b)
    }

    /// Applies a `t1000_core::DecoherenceChannel` to this state in-place.
    pub fn apply_channel(&mut self, channel: &t1000_core::DecoherenceChannel) -> f64 {
        let mut reg = self.to_register(channel.seed);
        let lost = t1000_core::channel::apply(&mut reg, channel);
        let new_state = Self::from_register(&reg);
        self.copy_from(&new_state);
        lost
    }

    /// Hardware-vectorized L2 norm normalization (AVX2 + FMA).
    pub fn normalize(&mut self) {
        crate::simd::normalize_kernel(&mut self.real, &mut self.imag);
    }

    /// Returns the L2 norm squared: \sum_i (|re_i|^2 + |im_i|^2) via hardware-accelerated AVX2+FMA.
    #[inline(always)]
    pub fn l2_norm_sq(&self) -> f64 {
        let (re, _) =
            crate::simd::inner_product_kernel(&self.real, &self.imag, &self.real, &self.imag);
        re
    }

    /// Returns the L2 norm: \sqrt{\sum_i (|re_i|^2 + |im_i|^2)}.
    #[inline(always)]
    pub fn l2_norm(&self) -> f64 {
        self.l2_norm_sq().sqrt()
    }

    /// Asserts that the statevector is normalized within numerical tolerance epsilon.
    #[inline(always)]
    pub fn assert_normalized(&self, epsilon: f64) {
        let norm_sq = self.l2_norm_sq();
        assert!(
            (norm_sq - 1.0).abs() <= epsilon,
            "L2 norm numerical stability assertion failed: ||psi||^2 = {:.14} (diff = {:.2e}, tolerance = {:.2e})",
            norm_sq,
            (norm_sq - 1.0).abs(),
            epsilon
        );
    }

    /// Hardware-vectorized inner product <self | other> (AVX2 + FMA).
    pub fn inner_product(&self, other: &Self) -> Complex64 {
        assert_eq!(self.dim, other.dim);
        let (re, im) =
            crate::simd::inner_product_kernel(&self.real, &self.imag, &other.real, &other.imag);
        Complex64::new(re, im)
    }

    /// Quantum fidelity |<self | other>|^2
    pub fn fidelity(&self, other: &Self) -> f64 {
        let (re, im) =
            crate::simd::inner_product_kernel(&self.real, &self.imag, &other.real, &other.imag);
        re * re + im * im
    }

    /// Returns a zero-allocation read-only SIMD view over this statevector.
    #[inline(always)]
    pub fn as_simd(&self) -> crate::simd::SimdStateView<'_> {
        crate::simd::SimdStateView::new(&self.real, &self.imag)
    }

    /// Returns a zero-allocation mutable SIMD view over this statevector.
    #[inline(always)]
    pub fn as_simd_mut(&mut self) -> crate::simd::SimdStateViewMut<'_> {
        crate::simd::SimdStateViewMut::new(&mut self.real, &mut self.imag)
    }

    /// Born rule measurement probabilities $p_i = |\psi_i|^2 = \text{Re}_i^2 + \text{Im}_i^2$.
    #[inline]
    pub fn probabilities(&self) -> Vec<f64> {
        self.real
            .iter()
            .zip(self.imag.iter())
            .map(|(&r, &i)| r * r + i * i)
            .collect()
    }

    /// Circular permutation (directional temporal flow)
    pub fn permute(&self, shift: usize) -> Self {
        let k = shift % self.dim;
        let mut r_new = vec![0.0; self.dim];
        let mut i_new = vec![0.0; self.dim];
        for i in 0..self.dim {
            let target = (i + k) % self.dim;
            r_new[target] = self.real[i];
            i_new[target] = self.imag[i];
        }
        Self {
            dim: self.dim,
            real: r_new,
            imag: i_new,
        }
    }

    /// Elementwise complex multiplication (FHRR phasor binding)
    pub fn bind(&self, other: &Self) -> Self {
        assert_eq!(self.dim, other.dim);
        let mut r_new = Vec::with_capacity(self.dim);
        let mut i_new = Vec::with_capacity(self.dim);
        for i in 0..self.dim {
            let (r1, im1) = (self.real[i], self.imag[i]);
            let (r2, im2) = (other.real[i], other.imag[i]);
            // (r1 + i im1) * (r2 + i im2) = (r1*r2 - im1*im2) + i (r1*im2 + im1*r2)
            r_new.push(r1 * r2 - im1 * im2);
            i_new.push(r1 * im2 + im1 * r2);
        }
        let mut res = Self {
            dim: self.dim,
            real: r_new,
            imag: i_new,
        };
        res.normalize();
        res
    }

    /// Quantum superposition bundling: sum of states with normalization
    pub fn superpose(&mut self, other: &Self, weight: f64) {
        assert_eq!(self.dim, other.dim);
        for i in 0..self.dim {
            self.real[i] += other.real[i] * weight;
            self.imag[i] += other.imag[i] * weight;
        }
        self.normalize();
    }

    /// Returns the exact quantum $\ell_1$-norm coherence metrics (raw and normalized).
    pub fn coherence_l1(&self) -> (f64, f64) {
        let n_qubits = (self.dim as f64).log2().round() as usize;
        let m = crate::coherence::compute_coherence_metrics(self, n_qubits);
        (m.l1_raw, m.l1_normalized)
    }

    /// Re-projects the state onto the complex projective unit hypersphere,
    /// eliminating numerical phase creep and gauge drift over deep gate passes.
    pub fn stabilize_manifold(&mut self) {
        crate::coherence::reproject_coherence_manifold(self);
    }
}

/// Parameterized Quantum Feature Map mapping tokens/strings into Hilbert space.
#[derive(Debug, Clone)]
pub struct QuantumFeatureMap {
    pub dim: usize,
    seed: u64,
}

impl QuantumFeatureMap {
    pub fn new(dim: usize, seed: u64) -> Self {
        assert!(
            dim >= 1024 && dim.is_power_of_two(),
            "QuantumFeatureMap requires a power-of-two dim >= 1024 (got {})",
            dim
        );
        Self { dim, seed }
    }

    /// Deterministically hash a token string to an angle vector and create normalized state.
    pub fn encode_token(&self, token: &str) -> QuantumState {
        let mut real = Vec::with_capacity(self.dim);
        let mut imag = Vec::with_capacity(self.dim);
        let base_mag = 1.0 / (self.dim as f64).sqrt();

        for k in 0..self.dim {
            let phase = self.hash_token_phase(token, k);
            let (s, c) = phase.sin_cos();
            real.push(base_mag * c);
            imag.push(base_mag * s);
        }

        QuantumState {
            dim: self.dim,
            real,
            imag,
        }
    }

    /// Ingest a sequence of tokens into a single bound holographic trajectory state:
    /// Ψ_seq = sum_t Π^t(Ψ(token_t))
    pub fn encode_sequence(&self, tokens: &[&str]) -> QuantumState {
        if tokens.is_empty() {
            return QuantumState::new(self.dim);
        }

        let mut doc_state = QuantumState::zero(self.dim);

        for (pos, &token) in tokens.iter().enumerate() {
            let tok_state = self.encode_token(token);
            let permuted = tok_state.permute(pos + 1);
            for i in 0..self.dim {
                doc_state.real[i] += permuted.real[i];
                doc_state.imag[i] += permuted.imag[i];
            }
        }

        doc_state.normalize();
        doc_state
    }

    fn hash_token_phase(&self, token: &str, index: usize) -> f64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.seed.hash(&mut hasher);
        token.hash(&mut hasher);
        index.hash(&mut hasher);
        let h = hasher.finish();

        let frac = (h as f64) / (u64::MAX as f64);
        frac * 2.0 * PI
    }
}

// =========================================================================
// Bidirectional Substrate Conversions (t1000_core <-> t1000)
// =========================================================================

impl From<QubitRegister> for QuantumState {
    fn from(reg: QubitRegister) -> Self {
        Self::from_register(&reg)
    }
}

impl From<&QubitRegister> for QuantumState {
    fn from(reg: &QubitRegister) -> Self {
        Self::from_register(reg)
    }
}

impl From<QuantumState> for QubitRegister {
    fn from(state: QuantumState) -> Self {
        state.to_register(crate::DEFAULT_SEED)
    }
}

impl From<&QuantumState> for QubitRegister {
    fn from(state: &QuantumState) -> Self {
        state.to_register(crate::DEFAULT_SEED)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_state_register_roundtrip() {
        let original = QuantumState::new(16);
        let reg: QubitRegister = (&original).into();
        assert_eq!(reg.basis, 16);

        let restored: QuantumState = (&reg).into();
        assert_eq!(restored.dim, 16);
        assert!((original.fidelity(&restored) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_from_complex_slice_and_partial_trace() {
        // Prepare maximally entangled 2-qubit Bell state: (|00> + |11>) / sqrt(2)
        let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
        let bell_amps = vec![
            Complex64::new(inv_sqrt2, 0.0), // |00>
            Complex64::default(),           // |01>
            Complex64::default(),           // |10>
            Complex64::new(inv_sqrt2, 0.0), // |11>
        ];
        let state = QuantumState::from_complex_slice(&bell_amps);
        assert_eq!(state.dim, 4);

        // Tracing out qubit B (dim_b = 2) leaves maximally mixed qubit A: rho_A = 0.5 * I
        let rho_a = state.partial_trace_b(2, 2);
        assert_eq!(rho_a.len(), 4);
        assert!((rho_a[0].re - 0.5).abs() < 1e-10); // <0|rho|0> = 0.5
        assert!((rho_a[3].re - 0.5).abs() < 1e-10); // <1|rho|1> = 0.5
        assert!(rho_a[1].norm() < 1e-10); // off-diagonal = 0
        assert!(rho_a[2].norm() < 1e-10);
    }

    #[test]
    fn test_apply_channel_on_quantum_state() {
        let mut state = QuantumState::ground(8);
        let channel = t1000_core::DecoherenceChannel::new(
            t1000_core::ChannelKind::AmplitudeDamping { gamma: 0.1 },
            42,
        );
        let lost = state.apply_channel(&channel);
        assert!(lost >= 0.0);
        state.assert_normalized(1e-6);
    }
}
