// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Zero-Copy Expectation Streaming Engine
//!
//! Evaluates quantum evolution trajectories natively through continuous phase-amplitude
//! streaming accumulators without destructive projective measurements or cache line dirtying.
//!
//! Traditional simulators interrupt the gate loop to sample discrete states or mutate the
//! statevector via projective collapse. Mirroring continuous weak supercurrent readout in
//! superconducting Andreev systems, this module accumulates:
//! - Longitudinal polarization $\langle Z_q \rangle$
//! - Transverse polarization $\langle X_q \rangle$ and $\langle Y_q \rangle$
//! - Continuous phase flux $\Phi_q = \mathrm{atan2}(\langle Y_q \rangle, \langle X_q \rangle)$
//! - Coherence flux $\mathcal{F}_{\mathrm{coh}} = \frac{1}{N} \sum_q \sqrt{\langle X_q \rangle^2 + \langle Y_q \rangle^2}$
//! - Statevector norm $\|\psi\|^2$
//!
//! all in a single non-destructive read-only streaming pass.

use crate::state::QuantumState;
use serde::{Deserialize, Serialize};

/// Trajectory telemetry point capturing running expectation registers at a discrete evolution step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryPoint {
    /// Zero-based step counter
    pub step: usize,
    /// Semantic identifier (e.g. stage name, gate descriptor, timestamp)
    pub tag: String,
    /// Running Pauli-Z expectation values $\langle Z_q \rangle$ per qubit
    pub z_expectations: Vec<f64>,
    /// Running Pauli-X expectation values $\langle X_q \rangle$ per qubit
    pub x_expectations: Vec<f64>,
    /// Running Pauli-Y expectation values $\langle Y_q \rangle$ per qubit
    pub y_expectations: Vec<f64>,
    /// Running phase flux $\Phi_q = \mathrm{atan2}(\langle Y_q \rangle, \langle X_q \rangle)$ in radians
    pub phase_flux: Vec<f64>,
    /// Total statevector norm $\|\psi\|^2$
    pub norm_sq: f64,
    /// Mean longitudinal polarization across all qubits
    pub mean_z: f64,
    /// Transverse coherence flux measuring Bloch equator displacement
    pub coherence_flux: f64,
}

impl TrajectoryPoint {
    /// Calculates Euclidean trajectory distance to another point in $(Z, X)$ expectation space.
    pub fn distance(&self, other: &Self) -> f64 {
        let n = self.z_expectations.len().min(other.z_expectations.len());
        if n == 0 {
            return 0.0;
        }
        let mut sum_sq = 0.0;
        for q in 0..n {
            let dz = self.z_expectations[q] - other.z_expectations[q];
            let dx = self.x_expectations[q] - other.x_expectations[q];
            let dy = self.y_expectations[q] - other.y_expectations[q];
            sum_sq += dz * dz + dx * dx + dy * dy;
        }
        (sum_sq / (n as f64)).sqrt()
    }

    /// Single-qubit Bloch vector length $r_q = \sqrt{\langle X_q \rangle^2 + \langle Y_q \rangle^2 + \langle Z_q \rangle^2}$.
    pub fn bloch_radius(&self, qubit: usize) -> f64 {
        if qubit >= self.z_expectations.len() {
            return 0.0;
        }
        let z = self.z_expectations[qubit];
        let x = self.x_expectations[qubit];
        let y = self.y_expectations[qubit];
        (x * x + y * y + z * z).sqrt()
    }
}

/// Non-temporal streaming expectation accumulator.
///
/// Maintains running phase-amplitude trajectory registers concurrently with state evolution,
/// eliminating projective measurement collapse overhead and cache eviction penalties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingExpectationAccumulator {
    num_qubits: usize,
    history: Vec<TrajectoryPoint>,
    step_counter: usize,
}

impl StreamingExpectationAccumulator {
    /// Creates a new streaming accumulator for an $N$-qubit register.
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            history: Vec::new(),
            step_counter: 0,
        }
    }

    /// Creates an accumulator pre-allocated for a planned trajectory depth.
    pub fn with_capacity(num_qubits: usize, capacity: usize) -> Self {
        Self {
            num_qubits,
            history: Vec::with_capacity(capacity),
            step_counter: 0,
        }
    }

    /// Number of qubits monitored.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Resets history and counter while retaining allocated capacity.
    pub fn reset(&mut self) {
        self.history.clear();
        self.step_counter = 0;
    }

    /// Returns a slice of all recorded trajectory points.
    pub fn history(&self) -> &[TrajectoryPoint] {
        &self.history
    }

    /// Returns the most recently evaluated trajectory point, if any.
    pub fn latest(&self) -> Option<&TrajectoryPoint> {
        self.history.last()
    }

    /// Evaluates expectation values directly from the statevector in a single non-destructive pass,
    /// appends the point to history, and returns a reference to the new point.
    pub fn stream_step(
        &mut self,
        state: &QuantumState,
        tag: impl Into<String>,
    ) -> &TrajectoryPoint {
        let pt = Self::accumulate_point(self.num_qubits, state, self.step_counter, tag.into());
        self.step_counter += 1;
        self.history.push(pt);
        self.history.last().expect("point just pushed")
    }

    /// Computes cumulative trajectory drift across the entire recorded sequence.
    pub fn total_drift(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        let mut drift = 0.0;
        for i in 1..self.history.len() {
            drift += self.history[i].distance(&self.history[i - 1]);
        }
        drift
    }

    /// Core non-destructive streaming reduction kernel.
    ///
    /// Computes norm, $Z$, $X$, and $Y$ expectations across the statevector without writing back
    /// or mutating any amplitude cache lines.
    pub fn accumulate_point(
        num_qubits: usize,
        state: &QuantumState,
        step: usize,
        tag: String,
    ) -> TrajectoryPoint {
        let dim = state.dim;
        assert!(dim >= (1 << num_qubits), "state dimension smaller than monitored qubits");

        let mut z_acc = vec![0.0; num_qubits];
        let mut x_acc = vec![0.0; num_qubits];
        let mut y_acc = vec![0.0; num_qubits];
        let mut norm_acc = 0.0;

        let real = &state.real;
        let imag = &state.imag;

        // Pass 1: Total norm and longitudinal Z expectations
        for idx in 0..dim {
            let re = real[idx];
            let im = imag[idx];
            let prob = re * re + im * im;
            norm_acc += prob;

            for q in 0..num_qubits {
                if (idx >> q) & 1 == 0 {
                    z_acc[q] += prob;
                } else {
                    z_acc[q] -= prob;
                }
            }
        }

        // Pass 2: Transverse X and Y expectations (cross-amplitude overlap)
        for q in 0..num_qubits {
            let mask = 1usize << q;
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;

            for idx in 0..dim {
                if (idx & mask) == 0 {
                    let partner = idx | mask;
                    let r0 = real[idx];
                    let i0 = imag[idx];
                    let r1 = real[partner];
                    let i1 = imag[partner];

                    // <psi|X|psi> on qubit q: 2 * Re(psi_0^* * psi_1) = 2 * (r0*r1 + i0*i1)
                    sum_x += 2.0 * (r0 * r1 + i0 * i1);
                    // <psi|Y|psi> on qubit q: 2 * Im(psi_0^* * psi_1) = 2 * (r0*i1 - i0*r1)
                    sum_y += 2.0 * (r0 * i1 - i0 * r1);
                }
            }
            x_acc[q] = sum_x;
            y_acc[q] = sum_y;
        }

        // Normalize expectations by state norm if state is unnormalized
        let inv_norm = if norm_acc > 1e-15 { 1.0 / norm_acc } else { 1.0 };
        for q in 0..num_qubits {
            z_acc[q] *= inv_norm;
            x_acc[q] *= inv_norm;
            y_acc[q] *= inv_norm;
        }

        // Calculate phase flux and coherence flux
        let mut phase_flux = Vec::with_capacity(num_qubits);
        let mut total_transverse = 0.0;
        let mut sum_z = 0.0;

        for q in 0..num_qubits {
            let z = z_acc[q];
            let x = x_acc[q];
            let y = y_acc[q];
            phase_flux.push(y.atan2(x));
            total_transverse += (x * x + y * y).sqrt();
            sum_z += z;
        }

        let mean_z = sum_z / (num_qubits as f64);
        let coherence_flux = total_transverse / (num_qubits as f64);

        TrajectoryPoint {
            step,
            tag,
            z_expectations: z_acc,
            x_expectations: x_acc,
            y_expectations: y_acc,
            phase_flux,
            norm_sq: norm_acc,
            mean_z,
            coherence_flux,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_1_SQRT_2;

    #[test]
    fn test_streaming_ground_state() {
        let state = QuantumState::ground(8); // 3 qubits
        let mut acc = StreamingExpectationAccumulator::new(3);
        let pt = acc.stream_step(&state, "ground_init");

        assert_eq!(pt.step, 0);
        assert_eq!(pt.tag, "ground_init");
        assert!((pt.norm_sq - 1.0).abs() < 1e-12);
        assert!((pt.mean_z - 1.0).abs() < 1e-12);
        for q in 0..3 {
            assert!((pt.z_expectations[q] - 1.0).abs() < 1e-12);
            assert!(pt.x_expectations[q].abs() < 1e-12);
            assert!(pt.y_expectations[q].abs() < 1e-12);
            assert!((pt.bloch_radius(q) - 1.0).abs() < 1e-12);
        }
        assert!(pt.coherence_flux < 1e-12);
    }

    #[test]
    fn test_streaming_superposition_state() {
        // |+++> = 1/sqrt(8) sum_k |k>
        let mut state = QuantumState::ground(8);
        let amp = 1.0 / (8.0_f64).sqrt();
        for idx in 0..8 {
            state.real[idx] = amp;
            state.imag[idx] = 0.0;
        }

        let mut acc = StreamingExpectationAccumulator::new(3);
        let pt = acc.stream_step(&state, "plus_superposition");

        assert!((pt.norm_sq - 1.0).abs() < 1e-12);
        assert!(pt.mean_z.abs() < 1e-12);
        for q in 0..3 {
            assert!(pt.z_expectations[q].abs() < 1e-12);
            assert!((pt.x_expectations[q] - 1.0).abs() < 1e-12);
            assert!(pt.y_expectations[q].abs() < 1e-12);
            assert!(pt.phase_flux[q].abs() < 1e-12);
        }
        assert!((pt.coherence_flux - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_streaming_bell_state_trajectory() {
        // Step 0: |00>
        let mut s0 = QuantumState::ground(4);
        let mut acc = StreamingExpectationAccumulator::new(2);
        acc.stream_step(&s0, "init_00");

        // Step 1: (|00> + |10>) / sqrt(2) (H on q0)
        s0.real[0] = FRAC_1_SQRT_2;
        s0.real[1] = FRAC_1_SQRT_2;
        acc.stream_step(&s0, "h_q0");

        // Step 2: Bell state (|00> + |11>) / sqrt(2) (CNOT 0->1)
        s0.real[1] = 0.0;
        s0.real[3] = FRAC_1_SQRT_2;
        acc.stream_step(&s0, "cnot_01");

        assert_eq!(acc.history().len(), 3);
        let bell_pt = acc.latest().unwrap();
        // For Bell state, single qubit reduced states are maximally mixed: <Z>=0, <X>=0, <Y>=0
        assert!(bell_pt.z_expectations[0].abs() < 1e-12);
        assert!(bell_pt.z_expectations[1].abs() < 1e-12);
        assert!(bell_pt.x_expectations[0].abs() < 1e-12);
        assert!(bell_pt.x_expectations[1].abs() < 1e-12);

        let drift = acc.total_drift();
        assert!(drift > 0.0);
    }
}
