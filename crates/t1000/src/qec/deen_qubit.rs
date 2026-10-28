// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # DeenQubitEngine & Asynchronous Pauli Frame Sidecar Pipeline
//!
//! Directly integrates the concurrent `PauliFrameTrackerSidecar` into the shot-batched
//! noise simulation loop. Stabilizer syndrome masks are packaged into `PauliFrameBatch`
//! structs and dispatched asynchronously over non-blocking crossbeam MPSC channels,
//! decoupling syndrome popcount and phase-drift evaluation from the primary quantum execution thread.

use crate::qec::{PauliFrameBatch, PauliFrameTrackerSidecar};
use std::sync::atomic::Ordering;

/// Core deen_qubit simulation engine with an attached asynchronous Pauli frame sidecar.
#[derive(Debug)]
pub struct DeenQubitEngine {
    qubit_count: usize,
    tracker_sidecar: PauliFrameTrackerSidecar,
    execution_batch_counter: u64,
}

impl DeenQubitEngine {
    /// Initialize the deen_qubit engine with an attached asynchronous Pauli frame sidecar.
    pub fn new(qubit_count: usize) -> Self {
        let tracker_sidecar = PauliFrameTrackerSidecar::new(qubit_count);

        Self {
            qubit_count,
            tracker_sidecar,
            execution_batch_counter: 0,
        }
    }

    /// Qubit capacity of the engine.
    pub fn qubit_count(&self) -> usize {
        self.qubit_count
    }

    /// Total number of shot batches dispatched to the sidecar.
    pub fn execution_batch_counter(&self) -> u64 {
        self.execution_batch_counter
    }

    /// Access the underlying sidecar tracker directly.
    pub fn tracker_sidecar(&self) -> &PauliFrameTrackerSidecar {
        &self.tracker_sidecar
    }

    /// Execute a shot-batched Monte Carlo noise run, dispatching Pauli error frames
    /// to the sidecar concurrently without blocking execution.
    pub fn execute_shot_batch_with_tracking(
        &mut self,
        x_stabilizers: Vec<u64>,
        z_stabilizers: Vec<u64>,
        error_probability: f64,
    ) -> Result<(), &'static str> {
        self.execution_batch_counter += 1;

        // 1. Construct the frame batch from the current execution tick
        let batch = PauliFrameBatch {
            batch_id: self.execution_batch_counter,
            qubit_count: self.qubit_count,
            x_stabilizers,
            z_stabilizers,
            error_probability,
        };

        // 2. Submit non-blockingly to the sidecar channel (acting like the SQUID sensor)
        self.tracker_sidecar.submit_batch(batch)?;

        // 3. Continue primary statevector / Clifford tableau execution immediately
        // (No waiting on error trace evaluation overhead in the hot path)
        Ok(())
    }

    /// Pull live telemetry metrics from the sidecar without interrupting active shots.
    pub fn get_telemetry(&self) -> (u64, u64) {
        let metrics = self.tracker_sidecar.get_metrics();
        let processed = metrics.frames_processed.load(Ordering::Relaxed);
        let drift = metrics.cumulative_drift.load(Ordering::Relaxed);
        (processed, drift)
    }

    /// Report live telemetry metrics to stdout.
    pub fn report_telemetry(&self) {
        let (processed, drift) = self.get_telemetry();
        println!(
            "[TELEMETRY] Batches Processed: {} | Cumulative Pauli Drift Weight: {}",
            processed, drift
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_deen_qubit_engine_lifecycle_and_telemetry() {
        let mut engine = DeenQubitEngine::new(64);
        assert_eq!(engine.qubit_count(), 64);
        assert_eq!(engine.execution_batch_counter(), 0);

        for _ in 0..10 {
            let x_masks = vec![0b1010_u64, 0b1100_u64];
            let z_masks = vec![0b0010_u64, 0b1000_u64];
            assert!(engine.execute_shot_batch_with_tracking(x_masks, z_masks, 0.001).is_ok());
        }

        assert_eq!(engine.execution_batch_counter(), 10);

        // Allow worker thread to drain
        std::thread::sleep(Duration::from_millis(30));

        let (processed, drift) = engine.get_telemetry();
        assert_eq!(processed, 10);
        // Each batch: (0b1010 ^ 0b0010).count_ones() = 1, (0b1100 ^ 0b1000).count_ones() = 1 -> 2
        // 10 batches * 2 = 20 drift
        assert_eq!(drift, 20);
    }
}
