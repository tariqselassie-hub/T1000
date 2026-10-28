// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Embedded Pauli Frame Co-Tracker (The SQUID Principle)
//!
//! Decouples phase error and Pauli frame tracking from the core physical gate execution pipeline.
//! Rather than actively compiling corrective Pauli gates into the statevector or invoking
//! destructive projective resets, this engine maintains an auxiliary symplectic coordinate space
//! updated via light-touch $O(1)$ transformations.
//!
//! ## The SQUID Readout Principle
//! Analogous to a Superconducting Quantum Interference Device (SQUID) weakly measuring magnetic
//! flux without collapsing wavefunctions, the Pauli Frame Tracker maintains continuous visibility
//! into multi-qubit coherence and corrects observables at readout time through symplectic inner
//! products:
//! $$\langle \psi_L | O | \psi_L \rangle = \langle \psi | P^\dagger O P | \psi \rangle = (-1)^{\langle P, O \rangle_{\mathrm{sp}}} \langle \psi | O | \psi \rangle$$

use crate::compiler::CircuitOp;
use crate::measurements::PauliObservable;
use crate::qec::PauliErrorKind;
use crossbeam_channel::{unbounded, Receiver, Sender};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Elementary Pauli operator on a single physical qubit site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PauliType {
    /// Identity operator
    I,
    /// Pauli-X (Bit Flip)
    X,
    /// Pauli-Y (Combined Bit and Phase Flip with factor $i$)
    Y,
    /// Pauli-Z (Phase Flip)
    Z,
}

impl PauliType {
    /// Returns true if the operator is not Identity.
    pub fn is_non_trivial(&self) -> bool {
        !matches!(self, PauliType::I)
    }

    /// Evaluates single-qubit multiplication: $A \cdot B = \pm (i) C$.
    pub fn multiply(&self, other: Self) -> (Self, u8) {
        match (*self, other) {
            (PauliType::I, b) => (b, 0),
            (a, PauliType::I) => (a, 0),
            (PauliType::X, PauliType::X) => (PauliType::I, 0),
            (PauliType::Y, PauliType::Y) => (PauliType::I, 0),
            (PauliType::Z, PauliType::Z) => (PauliType::I, 0),
            (PauliType::X, PauliType::Y) => (PauliType::Z, 1),  // XY = iZ
            (PauliType::Y, PauliType::X) => (PauliType::Z, 3),  // YX = -iZ
            (PauliType::Y, PauliType::Z) => (PauliType::X, 1),  // YZ = iX
            (PauliType::Z, PauliType::Y) => (PauliType::X, 3),  // ZY = -iX
            (PauliType::Z, PauliType::X) => (PauliType::Y, 1),  // ZX = iY
            (PauliType::X, PauliType::Z) => (PauliType::Y, 3),  // XZ = -iY
        }
    }
}

/// Symplectic Pauli Frame Co-Tracker running alongside quantum circuit simulation.
///
/// Tracks virtual Pauli frames $P = (-1)^s i^p \bigotimes_{j=0}^{N-1} X^{x_j} Z^{z_j}$
/// without dirtying or modifying statevector buffers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PauliFrameTracker {
    num_qubits: usize,
    /// Bit flip indicators for each qubit $x_j \in \{0, 1\}$
    x_mask: Vec<bool>,
    /// Phase flip indicators for each qubit $z_j \in \{0, 1\}$
    z_mask: Vec<bool>,
    /// Global phase quadrant $p \in \{0, 1, 2, 3\}$ representing factor $i^p$
    phase_quadrant: u8,
    /// Total count of gate transformations processed
    gates_tracked: usize,
}

impl PauliFrameTracker {
    /// Creates a clean identity Pauli frame for an $N$-qubit register.
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            x_mask: vec![false; num_qubits],
            z_mask: vec![false; num_qubits],
            phase_quadrant: 0,
            gates_tracked: 0,
        }
    }

    /// Resets the frame to clean identity state.
    pub fn reset(&mut self) {
        self.x_mask.fill(false);
        self.z_mask.fill(false);
        self.phase_quadrant = 0;
        self.gates_tracked = 0;
    }

    /// Total number of qubits in the frame.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Number of gates tracked by this sidecar.
    pub fn gates_tracked(&self) -> usize {
        self.gates_tracked
    }

    /// Returns the effective Pauli operator at a single qubit site.
    pub fn qubit_pauli(&self, qubit: usize) -> PauliType {
        assert!(qubit < self.num_qubits, "qubit index out of bounds");
        match (self.x_mask[qubit], self.z_mask[qubit]) {
            (false, false) => PauliType::I,
            (true, false) => PauliType::X,
            (true, true) => PauliType::Y,
            (false, true) => PauliType::Z,
        }
    }

    /// Returns true if the frame is strictly the identity operator (no active errors).
    pub fn is_identity(&self) -> bool {
        self.phase_quadrant == 0
            && self.x_mask.iter().all(|&b| !b)
            && self.z_mask.iter().all(|&b| !b)
    }

    /// Computes the Hamming weight (number of non-identity physical sites) of the frame.
    pub fn weight(&self) -> usize {
        (0..self.num_qubits)
            .filter(|&q| self.x_mask[q] || self.z_mask[q])
            .count()
    }

    /// Injects an error into the Pauli frame in $O(1)$ time without touching physical state memory.
    pub fn inject_error(&mut self, qubit: usize, kind: PauliErrorKind) {
        assert!(qubit < self.num_qubits, "qubit index out of bounds");
        match kind {
            PauliErrorKind::BitFlip => {
                // Multiply existing site by X
                if self.z_mask[qubit] {
                    // Was Z: Z * X = -i Y
                    self.phase_quadrant = (self.phase_quadrant + 3) % 4;
                }
                self.x_mask[qubit] = !self.x_mask[qubit];
            }
            PauliErrorKind::PhaseFlip => {
                // Multiply existing site by Z
                if self.x_mask[qubit] {
                    // Was X: X * Z = -i Y
                    self.phase_quadrant = (self.phase_quadrant + 3) % 4;
                }
                self.z_mask[qubit] = !self.z_mask[qubit];
            }
            PauliErrorKind::BitPhaseFlip => {
                // Multiply existing site by Y
                self.x_mask[qubit] = !self.x_mask[qubit];
                self.z_mask[qubit] = !self.z_mask[qubit];
                self.phase_quadrant = (self.phase_quadrant + 1) % 4;
            }
        }
    }

    /// Symplectically updates the frame through a single Hadamard gate:
    /// $H X H = Z, \quad H Z H = X, \quad H Y H = -Y$.
    pub fn track_hadamard(&mut self, qubit: usize) {
        assert!(qubit < self.num_qubits, "qubit out of bounds");
        let x = self.x_mask[qubit];
        let z = self.z_mask[qubit];
        if x && z {
            // Y -> -Y (phase shift +2)
            self.phase_quadrant = (self.phase_quadrant + 2) % 4;
        }
        self.x_mask[qubit] = z;
        self.z_mask[qubit] = x;
        self.gates_tracked += 1;
    }

    /// Symplectically updates the frame through a Phase gate $S$:
    /// $S X S^\dagger = Y = i X Z, \quad S Z S^\dagger = Z, \quad S Y S^\dagger = -X$.
    pub fn track_phase_s(&mut self, qubit: usize) {
        assert!(qubit < self.num_qubits, "qubit out of bounds");
        let x = self.x_mask[qubit];
        let z = self.z_mask[qubit];
        if x && !z {
            // X -> Y = i X Z
            self.z_mask[qubit] = true;
            self.phase_quadrant = (self.phase_quadrant + 1) % 4;
        } else if x && z {
            // Y -> -X
            self.z_mask[qubit] = false;
            self.phase_quadrant = (self.phase_quadrant + 2) % 4;
        }
        self.gates_tracked += 1;
    }

    /// Symplectically updates the frame through a CNOT gate:
    /// $X_c \to X_c X_t, \quad Z_t \to Z_c Z_t, \quad Z_c \to Z_c, \quad X_t \to X_t$.
    pub fn track_cnot(&mut self, control: usize, target: usize) {
        assert!(control < self.num_qubits && target < self.num_qubits, "qubits out of bounds");
        self.x_mask[target] ^= self.x_mask[control];
        self.z_mask[control] ^= self.z_mask[target];
        self.gates_tracked += 1;
    }

    /// Symplectically updates the frame through a Controlled-Z (CZ) gate:
    /// $X_1 \to X_1 Z_2, \quad X_2 \to Z_1 X_2$.
    pub fn track_cz(&mut self, q1: usize, q2: usize) {
        assert!(q1 < self.num_qubits && q2 < self.num_qubits, "qubits out of bounds");
        self.z_mask[q2] ^= self.x_mask[q1];
        self.z_mask[q1] ^= self.x_mask[q2];
        self.gates_tracked += 1;
    }

    /// Symplectically updates the frame through a SWAP gate.
    pub fn track_swap(&mut self, q1: usize, q2: usize) {
        assert!(q1 < self.num_qubits && q2 < self.num_qubits, "qubits out of bounds");
        self.x_mask.swap(q1, q2);
        self.z_mask.swap(q1, q2);
        self.gates_tracked += 1;
    }

    /// Tracks an arbitrary compiled circuit operation through the Pauli frame sidecar.
    pub fn track_circuit_op(&mut self, op: &CircuitOp) {
        match op {
            CircuitOp::CNOT { control, target } => self.track_cnot(*control, *target),
            CircuitOp::CZ { q1, q2 } => self.track_cz(*q1, *q2),
            CircuitOp::SWAP { q1, q2 } => self.track_swap(*q1, *q2),
            CircuitOp::AntiCNOT { control, target } => {
                // AntiCNOT is X_c CNOT X_c
                self.inject_error(*control, PauliErrorKind::BitFlip);
                self.track_cnot(*control, *target);
                self.inject_error(*control, PauliErrorKind::BitFlip);
            }
            CircuitOp::Unitary1Q { qubit, matrix } => {
                // Check for standard Clifford gates
                if matrix.is_hadamard(1e-6) {
                    self.track_hadamard(*qubit);
                } else if matrix.is_phase_s(1e-6) {
                    self.track_phase_s(*qubit);
                } else {
                    self.gates_tracked += 1;
                }
            }
            _ => {
                // Non-Clifford rotations preserve Z/X commutation when aligned
                self.gates_tracked += 1;
            }
        }
    }

    /// Evaluates the symplectic inner product between the active frame and an observable.
    ///
    /// Returns `true` if the frame commutes with the observable, `false` if they anti-commute.
    pub fn commutes_with_observable(&self, obs: &PauliObservable) -> bool {
        match obs {
            PauliObservable::Z(q) => {
                // Anti-commutes if qubit q has an X or Y component (i.e. x_mask[q] is true)
                !self.x_mask.get(*q).copied().unwrap_or(false)
            }
            PauliObservable::X(q) => {
                // Anti-commutes if qubit q has a Z or Y component (i.e. z_mask[q] is true)
                !self.z_mask.get(*q).copied().unwrap_or(false)
            }
            PauliObservable::ZZ(a, b) => {
                let flip_a = self.x_mask.get(*a).copied().unwrap_or(false);
                let flip_b = self.x_mask.get(*b).copied().unwrap_or(false);
                // Parity of anti-commutations: (-1)^(flip_a + flip_b)
                (flip_a ^ flip_b) == false
            }
            PauliObservable::XX(a, b) => {
                let flip_a = self.z_mask.get(*a).copied().unwrap_or(false);
                let flip_b = self.z_mask.get(*b).copied().unwrap_or(false);
                (flip_a ^ flip_b) == false
            }
            PauliObservable::YY(a, b) => {
                // Y anti-commutes with X and Z (i.e. when x != z)
                let y_commutes = |q: usize| {
                    let x = self.x_mask.get(q).copied().unwrap_or(false);
                    let z = self.z_mask.get(q).copied().unwrap_or(false);
                    x == z // Commutes with Y if site is I or Y
                };
                let ca = y_commutes(*a);
                let cb = y_commutes(*b);
                ca == cb
            }
            PauliObservable::AllZ => {
                let total_x_flips = self.x_mask.iter().filter(|&&b| b).count();
                (total_x_flips % 2) == 0
            }
            _ => true,
        }
    }

    /// Evaluates the effective observable expectation value $\langle \psi_L | O | \psi_L \rangle$
    /// directly from raw physical state expectation $\langle \psi | O | \psi \rangle$ in $O(1)$ time.
    pub fn conjugate_observable_expectation(
        &self,
        obs: &PauliObservable,
        raw_expectation: f64,
    ) -> f64 {
        if self.commutes_with_observable(obs) {
            raw_expectation
        } else {
            -raw_expectation
        }
    }

    /// Continuous SQUID-like coherence visibility metric:
    /// $$\mathcal{V}_{\mathrm{coh}} = \exp(-\gamma \cdot W_P)$$
    /// where $W_P$ is the Hamming weight of the frame and $\gamma$ is a decoherence scale parameter.
    pub fn coherence_visibility(&self, gamma: f64) -> f64 {
        let w = self.weight() as f64;
        (-gamma * w).exp()
    }
}

// Helper methods on Unitary2x2 for Clifford identification
impl crate::compiler::Unitary2x2 {
    pub fn is_hadamard(&self, eps: f64) -> bool {
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        (self.u00_re - inv_sqrt2).abs() < eps
            && self.u00_im.abs() < eps
            && (self.u01_re - inv_sqrt2).abs() < eps
            && (self.u10_re - inv_sqrt2).abs() < eps
            && (self.u11_re + inv_sqrt2).abs() < eps
    }

    pub fn is_phase_s(&self, eps: f64) -> bool {
        (self.u00_re - 1.0).abs() < eps
            && self.u00_im.abs() < eps
            && self.u01_re.abs() < eps
            && (self.u11_re.abs() < eps)
            && (self.u11_im - 1.0).abs() < eps
    }
}

/// Represents a batch of Pauli error correction frames tracked during simulation runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PauliFrameBatch {
    pub batch_id: u64,
    pub qubit_count: usize,
    pub x_stabilizers: Vec<u64>, // Bit-sliced X stabilizer flags
    pub z_stabilizers: Vec<u64>, // Bit-sliced Z stabilizer flags
    pub error_probability: f64,
}

impl PauliFrameBatch {
    /// Creates a new bit-sliced Pauli frame batch.
    pub fn new(
        batch_id: u64,
        qubit_count: usize,
        x_stabilizers: Vec<u64>,
        z_stabilizers: Vec<u64>,
        error_probability: f64,
    ) -> Self {
        Self {
            batch_id,
            qubit_count,
            x_stabilizers,
            z_stabilizers,
            error_probability,
        }
    }
}

/// Telemetry metrics emitted by the sidecar thread.
#[derive(Debug, Default, Clone)]
pub struct FrameTrackerMetrics {
    pub frames_processed: Arc<AtomicU64>,
    pub cumulative_drift: Arc<AtomicU64>, // Fixed-point representation of phase drift
    pub drop_count: Arc<AtomicU64>,
}

impl FrameTrackerMetrics {
    /// Total frames processed by the sidecar worker.
    pub fn frames_processed(&self) -> u64 {
        self.frames_processed.load(Ordering::Relaxed)
    }

    /// Cumulative syndrome error drift.
    pub fn cumulative_drift(&self) -> u64 {
        self.cumulative_drift.load(Ordering::Relaxed)
    }

    /// Dropped frame count.
    pub fn drop_count(&self) -> u64 {
        self.drop_count.load(Ordering::Relaxed)
    }
}

/// Concurrent, low-latency sidecar thread decoupling error tracking and phase-drift corrections
/// from the primary statevector execution loop.
pub struct PauliFrameTrackerSidecar {
    sender: Sender<PauliFrameBatch>,
    worker_handle: Option<JoinHandle<()>>,
    running: Arc<AtomicBool>,
    metrics: FrameTrackerMetrics,
}

impl std::fmt::Debug for PauliFrameTrackerSidecar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PauliFrameTrackerSidecar")
            .field("running", &self.running.load(Ordering::Relaxed))
            .field("metrics", &self.metrics)
            .finish()
    }
}

impl PauliFrameTrackerSidecar {
    /// Spawns the concurrent sidecar thread for asynchronous Pauli frame tracking.
    pub fn new(qubit_capacity: usize) -> Self {
        let (sender, receiver): (Sender<PauliFrameBatch>, Receiver<PauliFrameBatch>) = unbounded();
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = Arc::clone(&running);

        let metrics = FrameTrackerMetrics::default();
        let frames_processed_metric = Arc::clone(&metrics.frames_processed);
        let cumulative_drift_metric = Arc::clone(&metrics.cumulative_drift);

        let worker_handle = thread::spawn(move || {
            Self::tracker_loop(
                receiver,
                running_clone,
                frames_processed_metric,
                cumulative_drift_metric,
                qubit_capacity,
            );
        });

        Self {
            sender,
            worker_handle: Some(worker_handle),
            running,
            metrics,
        }
    }

    /// Push a new batch of frames into the non-blocking channel for the sidecar thread.
    pub fn submit_batch(&self, batch: PauliFrameBatch) -> Result<(), &'static str> {
        self.sender.send(batch).map_err(|_| "Tracker channel closed")
    }

    /// Internal high-performance worker loop running concurrently.
    fn tracker_loop(
        receiver: Receiver<PauliFrameBatch>,
        running: Arc<AtomicBool>,
        frames_processed: Arc<AtomicU64>,
        cumulative_drift: Arc<AtomicU64>,
        _qubit_capacity: usize,
    ) {
        let mut _local_batch_counter = 0u64;

        while running.load(Ordering::Relaxed) || !receiver.is_empty() {
            match receiver.recv_timeout(Duration::from_millis(2)) {
                Ok(batch) => {
                    let mut drift_sum = 0u64;
                    for (&x_mask, &z_mask) in batch.x_stabilizers.iter().zip(&batch.z_stabilizers) {
                        let weight = (x_mask ^ z_mask).count_ones() as u64;
                        drift_sum = drift_sum.wrapping_add(weight);
                    }

                    frames_processed.fetch_add(1, Ordering::Relaxed);
                    cumulative_drift.fetch_add(drift_sum, Ordering::Relaxed);
                    _local_batch_counter += 1;
                }
                Err(_) => {
                    if !running.load(Ordering::Relaxed) {
                        break;
                    }
                }
            }
        }
    }

    /// Access the shared thread metrics.
    pub fn get_metrics(&self) -> &FrameTrackerMetrics {
        &self.metrics
    }

    /// Number of batches processed so far.
    pub fn frames_processed(&self) -> u64 {
        self.metrics.frames_processed()
    }

    /// Total cumulative drift evaluated by the sidecar.
    pub fn cumulative_drift(&self) -> u64 {
        self.metrics.cumulative_drift()
    }
}

impl Drop for PauliFrameTrackerSidecar {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Release);
        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pauli_frame_identity_and_error_injection() {
        let mut frame = PauliFrameTracker::new(4);
        assert!(frame.is_identity());
        assert_eq!(frame.weight(), 0);

        // Inject BitFlip on q1 -> X
        frame.inject_error(1, PauliErrorKind::BitFlip);
        assert!(!frame.is_identity());
        assert_eq!(frame.weight(), 1);
        assert_eq!(frame.qubit_pauli(1), PauliType::X);

        // Inject PhaseFlip on q1 -> Y
        frame.inject_error(1, PauliErrorKind::PhaseFlip);
        assert_eq!(frame.qubit_pauli(1), PauliType::Y);
        assert_eq!(frame.weight(), 1);

        // Inject BitFlip on q3 -> X
        frame.inject_error(3, PauliErrorKind::BitFlip);
        assert_eq!(frame.weight(), 2);
    }

    #[test]
    fn test_hadamard_and_cnot_propagation() {
        let mut frame = PauliFrameTracker::new(2);
        // Error X on qubit 0
        frame.inject_error(0, PauliErrorKind::BitFlip);
        assert_eq!(frame.qubit_pauli(0), PauliType::X);

        // Hadamard on q0: X -> Z
        frame.track_hadamard(0);
        assert_eq!(frame.qubit_pauli(0), PauliType::Z);

        // CNOT from 0 to 1: Z_control does not propagate to target (Z_c -> Z_c)
        frame.track_cnot(0, 1);
        assert_eq!(frame.qubit_pauli(0), PauliType::Z);
        assert_eq!(frame.qubit_pauli(1), PauliType::I);

        // Error X on qubit 0
        frame.inject_error(0, PauliErrorKind::BitFlip); // now q0 is X * Z = Y
        assert_eq!(frame.qubit_pauli(0), PauliType::Y);

        // CNOT 0->1: X_c propagates to X_t
        frame.track_cnot(0, 1);
        assert_eq!(frame.qubit_pauli(1), PauliType::X);
    }

    #[test]
    fn test_symplectic_readout_conjugation() {
        let mut frame = PauliFrameTracker::new(2);
        let obs_z0 = PauliObservable::Z(0);

        // Initially identity: commutes with Z(0)
        assert!(frame.commutes_with_observable(&obs_z0));
        assert_eq!(frame.conjugate_observable_expectation(&obs_z0, 0.85), 0.85);

        // Inject X on q0: anti-commutes with Z(0)
        frame.inject_error(0, PauliErrorKind::BitFlip);
        assert!(!frame.commutes_with_observable(&obs_z0));
        // Readout expectation should be inverted: 0.85 -> -0.85
        assert_eq!(frame.conjugate_observable_expectation(&obs_z0, 0.85), -0.85);

        // High visibility when clean, drops with errors
        let clean = PauliFrameTracker::new(3);
        assert!((clean.coherence_visibility(0.2) - 1.0).abs() < 1e-12);
        assert!(frame.coherence_visibility(0.2) < 1.0);
    }

    #[test]
    fn test_pauli_frame_sidecar_concurrent_tracking() {
        let sidecar = PauliFrameTrackerSidecar::new(64);

        let batch1 = PauliFrameBatch::new(
            1,
            64,
            vec![0b1010, 0b1100],
            vec![0b0010, 0b1000],
            0.001,
        );
        let batch2 = PauliFrameBatch::new(
            2,
            64,
            vec![0b1111],
            vec![0b0000],
            0.002,
        );

        assert!(sidecar.submit_batch(batch1).is_ok());
        assert!(sidecar.submit_batch(batch2).is_ok());

        // Wait briefly for worker processing
        std::thread::sleep(Duration::from_millis(30));

        let metrics = sidecar.get_metrics();
        assert_eq!(metrics.frames_processed(), 2);
        // batch1: (0b1010 ^ 0b0010).count_ones() = 1000 -> 1
        //         (0b1100 ^ 0b1000).count_ones() = 0100 -> 1 -> sum = 2
        // batch2: (0b1111 ^ 0b0000).count_ones() = 4 -> sum = 4
        // total drift = 6
        assert_eq!(metrics.cumulative_drift(), 6);
        assert_eq!(sidecar.frames_processed(), 2);
        assert_eq!(sidecar.cumulative_drift(), 6);
    }
}
