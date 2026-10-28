// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # DeenQubitEngine & Asynchronous Pauli Frame Sidecar Pipeline
//!
//! Directly integrates the concurrent `PauliFrameTrackerSidecar` into the shot-batched
//! noise simulation loop. Stabilizer syndrome masks are packaged into `PauliFrameBatch`
//! structs and dispatched asynchronously over non-blocking crossbeam MPSC channels,
//! decoupling syndrome popcount and phase-drift evaluation from the primary quantum execution thread.

use crate::qec::{PauliFrameBatch, PauliFrameTrackerSidecar};
use serde::{Deserialize, Serialize};
use std::sync::atomic::Ordering;
use std::time::Instant;

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

/// Synchronous reference implementation for inline Pauli error tracing.
#[inline(always)]
pub fn process_pauli_frame_sync(x_masks: &[u64], z_masks: &[u64]) -> u64 {
    let mut drift_sum = 0u64;
    for (&x, &z) in x_masks.iter().zip(z_masks.iter()) {
        drift_sum = drift_sum.wrapping_add((x ^ z).count_ones() as u64);
    }
    drift_sum
}

/// Telemetry metrics comparing synchronous inline tracing against asynchronous sidecar dispatch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PauliTrackingBenchmarkResult {
    pub iterations: usize,
    pub qubit_count: usize,
    pub sync_duration_ms: f64,
    pub sync_ns_per_op: f64,
    pub sync_checksum: u64,
    pub async_dispatch_duration_ms: f64,
    pub async_ns_per_op: f64,
    pub sidecar_frames_processed: u64,
    pub speedup_dispatch_vs_sync: f64,
}

/// Benchmarks the overhead of asynchronous Pauli frame sidecar channel dispatch
/// against synchronous inline popcount evaluation across high-volume iteration streams.
pub fn benchmark_pauli_tracking_overhead(iterations: usize) -> PauliTrackingBenchmarkResult {
    println!(
        "[BENCHMARK] Starting Pauli Frame tracking comparison over {} iterations...",
        iterations
    );

    let qubit_count = 256;
    let dummy_x = vec![0xAAAAAAAAAAAAAAAA_u64; 4];
    let dummy_z = vec![0x5555555555555555_u64; 4];

    // ==========================================
    // 1. Benchmark Synchronous Inline Tracing
    // ==========================================
    let start_sync = Instant::now();
    let mut sync_checksum = 0u64;
    for _ in 0..iterations {
        sync_checksum = sync_checksum.wrapping_add(process_pauli_frame_sync(&dummy_x, &dummy_z));
    }
    let duration_sync = start_sync.elapsed();
    let sync_ns_per_op = duration_sync.as_nanos() as f64 / iterations as f64;
    println!(
        "   -> Synchronous Tracing:    {:?} total ({:.2} ns/op) [Checksum: {}]",
        duration_sync, sync_ns_per_op, sync_checksum
    );

    // ==========================================
    // 2. Benchmark Asynchronous Sidecar Dispatch
    // ==========================================
    let sidecar = PauliFrameTrackerSidecar::new(qubit_count);
    let start_async = Instant::now();

    for i in 0..iterations {
        let batch = PauliFrameBatch {
            batch_id: i as u64,
            qubit_count,
            x_stabilizers: dummy_x.clone(),
            z_stabilizers: dummy_z.clone(),
            error_probability: 0.001,
        };
        // Measure the non-blocking channel send overhead
        sidecar.submit_batch(batch).expect("Channel open");
    }

    let duration_async = start_async.elapsed();
    let async_ns_per_op = duration_async.as_nanos() as f64 / iterations as f64;

    // Allow background worker a brief moment to drain the queue for clean metrics
    std::thread::sleep(std::time::Duration::from_millis(50));

    println!(
        "   -> Asynchronous Sidecar:   {:?} total ({:.2} ns/op dispatch overhead)",
        duration_async, async_ns_per_op
    );

    let metrics = sidecar.get_metrics();
    let processed = metrics.frames_processed.load(Ordering::Relaxed);
    println!(
        "   -> Sidecar Worker Stats:   {} frames successfully processed in background.",
        processed
    );

    let speedup = if async_ns_per_op > 0.0 {
        sync_ns_per_op / async_ns_per_op
    } else {
        1.0
    };

    PauliTrackingBenchmarkResult {
        iterations,
        qubit_count,
        sync_duration_ms: duration_sync.as_secs_f64() * 1000.0,
        sync_ns_per_op,
        sync_checksum,
        async_dispatch_duration_ms: duration_async.as_secs_f64() * 1000.0,
        async_ns_per_op,
        sidecar_frames_processed: processed,
        speedup_dispatch_vs_sync: speedup,
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

    #[test]
    fn test_benchmark_pauli_tracking_overhead() {
        let res = benchmark_pauli_tracking_overhead(500);
        assert_eq!(res.iterations, 500);
        assert_eq!(res.qubit_count, 256);
        assert!(res.sync_checksum > 0);
        assert!(res.sidecar_frames_processed >= 450); // almost all or all processed
    }
}

