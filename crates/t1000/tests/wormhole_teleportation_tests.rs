// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

use std::f64::consts::PI;
use t1000::wormhole::{
    HolographicWormholeEngine, ScramblerType, WormholeConfig,
};

#[test]
fn test_wormhole_uncoupled_noise_floor() {
    let config = WormholeConfig {
        num_modes: 3,
        scrambler_type: ScramblerType::ChaoticHeisenberg,
        trotter_steps: 3,
        scrambling_time: 1.5,
        coupling_g: 0.0,
        seed: 42,
    };
    let engine = HolographicWormholeEngine::new(config);
    let pt = engine.simulate(0.0);

    // At g=0, Left and Right remain causally disconnected.
    // The Bell process fidelity must lie exactly at the 2-qubit classical noise floor (0.2500).
    assert!(
        (pt.fidelity - 0.25).abs() < 1e-4,
        "Uncoupled noise floor expected exactly 0.2500, got {}",
        pt.fidelity
    );
    assert_eq!(pt.asymmetry, 0.0);
}

#[test]
fn test_wormhole_traversal_negative_energy_advantage() {
    let config = WormholeConfig {
        num_modes: 3,
        scrambler_type: ScramblerType::ChaoticHeisenberg,
        trotter_steps: 3,
        scrambling_time: 1.5,
        coupling_g: 1.5,
        seed: 42,
    };
    let engine = HolographicWormholeEngine::new(config);

    let pt_pos = engine.simulate(1.5);
    let pt_neg = engine.simulate(-1.5);

    println!(
        "Wormhole g=+1.5 Fid={:.4}, g=-1.5 Fid={:.4}, Asym={:.4}",
        pt_pos.fidelity, pt_neg.fidelity, pt_pos.asymmetry
    );

    // ANEC violation / negative energy pulse (g > 0) must produce positive fidelity enhancement
    // over the uncoupled floor (0.2500), while positive energy (g < 0) suppresses it.
    assert!(
        pt_pos.fidelity > 0.2500,
        "Negative energy fidelity ({}) must exceed 0.2500",
        pt_pos.fidelity
    );
    assert!(
        pt_pos.fidelity > pt_neg.fidelity,
        "Negative energy ({}) must exceed wrong-sign positive energy ({})",
        pt_pos.fidelity,
        pt_neg.fidelity
    );
    assert!(
        pt_pos.asymmetry > 0.02,
        "Gravitational sign asymmetry must be positive, got {}",
        pt_pos.asymmetry
    );
}

#[test]
fn test_quantum_chaos_otoc_decay() {
    let config = WormholeConfig {
        num_modes: 4,
        scrambler_type: ScramblerType::ChaoticHeisenberg,
        trotter_steps: 4,
        scrambling_time: 2.0,
        coupling_g: 1.5,
        seed: 12345,
    };
    let engine = HolographicWormholeEngine::new(config);

    let otoc_early = engine.scrambler.compute_otoc(0, 3, 0.0, 1);
    let otoc_late = engine.scrambler.compute_otoc(0, 3, 2.0, 4);

    println!("OTOC early: {:.4}, late: {:.4}", otoc_early, otoc_late);
    assert!(
        (otoc_early - 1.0).abs() < 1e-4,
        "OTOC at t=0 must be exactly 1.0, got {}",
        otoc_early
    );
    assert!(
        otoc_late < otoc_early,
        "Chaotic OTOC must decay with time, got early={}, late={}",
        otoc_early,
        otoc_late
    );
}

#[test]
fn test_operator_size_distribution() {
    let config = WormholeConfig {
        num_modes: 4,
        scrambler_type: ScramblerType::ChaoticHeisenberg,
        trotter_steps: 3,
        scrambling_time: 1.5,
        coupling_g: 1.5,
        seed: 42,
    };
    let engine = HolographicWormholeEngine::new(config);
    let dist = t1000::wormhole::compute_operator_size_distribution(
        4,
        &engine.scrambler,
        1.5,
    );

    let total_prob: f64 = dist.iter().map(|(_, p)| *p).sum();
    assert!(
        (total_prob - 1.0).abs() < 1e-6,
        "Operator size distribution must sum to 1.0, got {}",
        total_prob
    );
    assert_eq!(dist.len(), 4);
}

#[test]
fn test_pure_state_teleportation() {
    let config = WormholeConfig {
        num_modes: 3,
        scrambler_type: ScramblerType::ChaoticHeisenberg,
        trotter_steps: 3,
        scrambling_time: 1.5,
        coupling_g: 1.5,
        seed: 42,
    };
    let engine = HolographicWormholeEngine::new(config);

    let fid_plus = engine.simulate_pure_state(1.5, PI * 0.5, 0.0);
    let fid_zero = engine.simulate_pure_state(1.5, 0.0, 0.0);

    println!("Pure state |+> fid: {:.4}, |0> fid: {:.4}", fid_plus, fid_zero);
    assert!(
        fid_plus > 0.0 && fid_zero > 0.0,
        "Pure state fidelities must be positive"
    );
}

#[test]
fn test_all_scrambler_modalities() {
    for scrambler_type in [
        ScramblerType::ChaoticHeisenberg,
        ScramblerType::MajoranaSyk,
        ScramblerType::FloquetHaar,
    ] {
        let config = WormholeConfig {
            num_modes: 4, // 4 modes to support 4-Majorana SYK
            scrambler_type,
            trotter_steps: 2,
            scrambling_time: 0.8,
            coupling_g: 1.5,
            seed: 999,
        };
        let engine = HolographicWormholeEngine::new(config);
        let pt = engine.simulate(1.5);
        assert!(
            pt.fidelity > 0.0 && pt.fidelity <= 1.0,
            "Scrambler {:?} produced invalid fidelity {}",
            scrambler_type,
            pt.fidelity
        );
    }
}
