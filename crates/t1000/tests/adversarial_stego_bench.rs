// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Integration test verifying Adversarial Cryptographic Steganography & Anomaly Radar.

#[path = "../src/bin/measurements/adversarial_stego_bench.rs"]
mod bench;

#[test]
fn test_adversarial_steganography_and_anomaly_radar() {
    bench::run_adversarial_stego_benchmark();
}
