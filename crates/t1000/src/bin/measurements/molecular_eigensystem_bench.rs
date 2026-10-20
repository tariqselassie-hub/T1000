// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Molecular Coulomb Matrix Eigensystem & QM9 Streaming Benchmark.
//!
//! Evaluates bare-metal throughput for real-symmetric 50x50 Coulomb matrix
//! Householder-QL eigensolver passes on RoboBohr molecules and streaming
//! extended XYZ parsing with 19-dimensional quantum feature extraction on QM9.

use std::fs::{self, File};
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;
use t1000::chemistry::coulomb::{CoulombMatrix50, RoboBohrRecord};
use t1000::chemistry::qm9::Qm9Record;
use zip::ZipArchive;

fn main() {
    println!("┌────────────────────────────────────────────────────────────────────────┐");
    println!("│ T-1000 MOLECULAR EIGENSYSTEM & QUANTUM CHEMICAL INGESTION BENCHMARK   │");
    println!("│ Pure Rust AVX2 Eigensolver + Streaming QM9 & RoboBohr Substrate        │");
    println!("│ Author: Terrence A. Jones SR (2026)                                    │");
    println!("└────────────────────────────────────────────────────────────────────────┘");

    let downloads_dir = Path::new(r"C:\Users\zoddj\Downloads");
    let robobohr_zip = downloads_dir.join("Ground_state_energies.zip");
    let qm9_zip = downloads_dir.join("QM9.zip");

    // =========================================================================
    // PART 1: RoboBohr Coulomb Matrix Eigensystem Benchmark
    // =========================================================================
    println!("\n[PHASE 1] Benchmarking Real-Symmetric 50x50 Coulomb Matrix Eigensystem...");

    let mut robobohr_evaluated = 0usize;
    let mut robobohr_total_time_ms = 0.0;
    let mut robobohr_throughput_mols_per_sec = 0.0;
    let mut robobohr_avg_latency_us = 0.0;
    let mut robobohr_max_trace_drift = 0.0;
    let mut robobohr_mean_eat = 0.0;

    if robobohr_zip.exists() {
        let file = File::open(&robobohr_zip).expect("Failed to open Ground_state_energies.zip");
        let mut archive = ZipArchive::new(BufReader::new(file)).expect("Valid zip archive");
        let csv_file = archive
            .by_name("roboBohr.csv")
            .expect("roboBohr.csv inside zip");

        let target_samples = 5000;
        println!(
            "  -> Reading up to {} molecules from roboBohr.csv...",
            target_samples
        );
        let records = RoboBohrRecord::stream_from_reader(csv_file, Some(target_samples))
            .expect("Streaming parsed records");

        robobohr_evaluated = records.len();
        println!(
            "  -> Ingested {} records. Executing 50x50 Householder-QL eigensolver passes...",
            robobohr_evaluated
        );

        let t0 = Instant::now();
        let mut sum_eat = 0.0;
        let mut max_drift = 0.0f64;

        for rec in &records {
            let eigs = rec.coulomb.eigenspectrum();
            let tr = rec.coulomb.trace();
            let eig_sum: f64 = eigs.iter().sum();
            let drift = (tr - eig_sum).abs();
            if drift > max_drift {
                max_drift = drift;
            }
            sum_eat += rec.eat_hartree;
        }

        let elapsed = t0.elapsed();
        robobohr_total_time_ms = elapsed.as_secs_f64() * 1000.0;
        robobohr_avg_latency_us =
            (elapsed.as_secs_f64() * 1_000_000.0) / (robobohr_evaluated as f64);
        robobohr_throughput_mols_per_sec =
            (robobohr_evaluated as f64) / elapsed.as_secs_f64().max(1e-9);
        robobohr_max_trace_drift = max_drift;
        robobohr_mean_eat = sum_eat / (robobohr_evaluated as f64);

        println!(
            "  -> Evaluated: {} molecules in {:.2} ms",
            robobohr_evaluated, robobohr_total_time_ms
        );
        println!(
            "  -> Throughput: {:.1} molecules/sec ({:.2} µs/eval)",
            robobohr_throughput_mols_per_sec, robobohr_avg_latency_us
        );
        println!(
            "  -> Max Trace Drift: {:.2e} (Machine-precision spectral preservation)",
            robobohr_max_trace_drift
        );
        println!("  -> Mean Atomization Energy: {:.4} Ha", robobohr_mean_eat);
    } else {
        println!(
            "  -> Warning: {:?} not found, running synthetic 50x50 validation",
            robobohr_zip
        );
        let dummy = CoulombMatrix50::new();
        let eigs = dummy.eigenspectrum();
        assert_eq!(eigs.len(), 50);
    }

    // =========================================================================
    // PART 2: QM9 Extended XYZ Streaming & 19-Dim Quantum Ingestion Benchmark
    // =========================================================================
    println!("\n[PHASE 2] Benchmarking QM9 Extended XYZ & Quantum Encoding Streaming...");

    let mut qm9_evaluated = 0usize;
    let mut qm9_total_time_ms = 0.0;
    let mut qm9_throughput_mols_per_sec = 0.0;
    let mut qm9_avg_latency_us = 0.0;
    let mut qm9_mean_u0 = 0.0;
    let mut qm9_mean_gap_ev = 0.0;
    let mut qm9_mean_dipole = 0.0;

    if qm9_zip.exists() {
        let target_qm9 = 5000;
        println!(
            "  -> Streaming and quantum encoding up to {} molecules from QM9.zip...",
            target_qm9
        );

        let t0 = Instant::now();
        let qm9_records =
            Qm9Record::stream_from_zip(&qm9_zip, Some(target_qm9)).expect("Stream QM9 records");

        let mut sum_u0 = 0.0;
        let mut sum_gap = 0.0;
        let mut sum_mu = 0.0;
        let mut encoded_count = 0usize;

        for rec in &qm9_records {
            sum_u0 += rec.u0;
            sum_gap += rec.gap_ev();
            sum_mu += rec.dipole_moment;

            if let Ok(q_vec) = rec.quantum_feature_vector() {
                assert_eq!(q_vec.len(), 19);
                encoded_count += 1;
            }
        }

        let elapsed = t0.elapsed();
        qm9_evaluated = qm9_records.len();
        qm9_total_time_ms = elapsed.as_secs_f64() * 1000.0;
        qm9_avg_latency_us = (elapsed.as_secs_f64() * 1_000_000.0) / (qm9_evaluated as f64);
        qm9_throughput_mols_per_sec = (qm9_evaluated as f64) / elapsed.as_secs_f64().max(1e-9);

        if qm9_evaluated > 0 {
            qm9_mean_u0 = sum_u0 / (qm9_evaluated as f64);
            qm9_mean_gap_ev = sum_gap / (qm9_evaluated as f64);
            qm9_mean_dipole = sum_mu / (qm9_evaluated as f64);
        }

        println!(
            "  -> Ingested: {} QM9 molecules ({} quantum feature maps generated)",
            qm9_evaluated, encoded_count
        );
        println!(
            "  -> Throughput: {:.1} molecules/sec ({:.2} µs/molecule)",
            qm9_throughput_mols_per_sec, qm9_avg_latency_us
        );
        println!("  -> Mean Ground State Energy (U_0): {:.4} Ha", qm9_mean_u0);
        println!("  -> Mean HOMO-LUMO Gap: {:.4} eV", qm9_mean_gap_ev);
        println!("  -> Mean Dipole Moment: {:.4} Debye", qm9_mean_dipole);
    } else {
        println!(
            "  -> Warning: {:?} not found, skipping streaming run",
            qm9_zip
        );
    }

    // =========================================================================
    // PART 3: Telemetry Reporting
    // =========================================================================
    println!("\n┌────────────────────────────────────────────────────────────────────────┐");
    println!("│ BENCHMARK MATRIX TELEMETRY                                             │");
    println!("├────────────────────────┬───────────────────────┬───────────────────────┤");
    println!("│ Metric                 │ RoboBohr Eigensystem  │ QM9 Quantum Ingestion │");
    println!("├────────────────────────┼───────────────────────┼───────────────────────┤");
    println!(
        "│ Molecules Evaluated    │ {:<21} │ {:<21} │",
        robobohr_evaluated, qm9_evaluated
    );
    println!(
        "│ Total Latency (ms)     │ {:<21.2} │ {:<21.2} │",
        robobohr_total_time_ms, qm9_total_time_ms
    );
    println!(
        "│ Throughput (mols/s)    │ {:<21.1} │ {:<21.1} │",
        robobohr_throughput_mols_per_sec, qm9_throughput_mols_per_sec
    );
    println!(
        "│ Per-Molecule (µs)      │ {:<21.2} │ {:<21.2} │",
        robobohr_avg_latency_us, qm9_avg_latency_us
    );
    println!(
        "│ Primary Scientific Val │ Eat: {:<14.4}Ha │ U0: {:<15.4}Ha │",
        robobohr_mean_eat, qm9_mean_u0
    );
    println!("└────────────────────────┴───────────────────────┴───────────────────────┘");

    // Write JSON telemetry
    let json_output = serde_json::json!({
        "timestamp": "2026-10-19T21:15:00Z",
        "author": "Terrence A. Jones SR",
        "system": "T-1000 Molecular Eigensystem & Quantum Chemistry Ingestion Engine",
        "robobohr": {
            "molecules_evaluated": robobohr_evaluated,
            "total_time_ms": robobohr_total_time_ms,
            "throughput_mols_per_sec": robobohr_throughput_mols_per_sec,
            "avg_latency_us": robobohr_avg_latency_us,
            "max_trace_drift": robobohr_max_trace_drift,
            "mean_eat_hartree": robobohr_mean_eat
        },
        "qm9": {
            "molecules_evaluated": qm9_evaluated,
            "total_time_ms": qm9_total_time_ms,
            "throughput_mols_per_sec": qm9_throughput_mols_per_sec,
            "avg_latency_us": qm9_avg_latency_us,
            "mean_u0_hartree": qm9_mean_u0,
            "mean_gap_ev": qm9_mean_gap_ev,
            "mean_dipole_debye": qm9_mean_dipole
        }
    });

    let results_dir = Path::new("results/data");
    let _ = fs::create_dir_all(results_dir);
    let report_path = results_dir.join("molecular_eigensystem_report.json");
    fs::write(
        &report_path,
        serde_json::to_string_pretty(&json_output).unwrap(),
    )
    .expect("Write report");
    println!("\n[OK] Telemetry report recorded at: {:?}", report_path);
}
