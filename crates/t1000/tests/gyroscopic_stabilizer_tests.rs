// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

use t1000::gyroscopic::{
    GyroMechanicsEngine, GyroscopicConfig, GyroscopicStabilizerEngine, JackTopology,
    QuantumLockEngine,
};

#[test]
fn test_throwing_jack_center_of_mass_invariance() {
    let mut topology = JackTopology::new(1.5);
    let initial_com = topology.center_of_mass;
    assert!(initial_com.norm() < 1e-15, "Initial center of mass must be at the origin");

    let engine = GyroMechanicsEngine::new(100.0, 50.0, 2.0);

    // Test across several precessional time steps
    for step in 0..50 {
        let t = step as f64 * 0.01;
        let m_step = engine.step(&mut topology, t);
        assert!(
            m_step.center_of_mass_drift < 1e-14,
            "Center of mass drift {:e} exceeded threshold at t={}",
            m_step.center_of_mass_drift,
            t
        );
    }
}

#[test]
fn test_exact_precessional_torque_cancellation() {
    let mut topology = JackTopology::new(1.5);
    let engine = GyroMechanicsEngine::new(120.0, 45.0, 3.5);

    for step in 1..=40 {
        let t = step as f64 * 0.02;
        let m_step = engine.step(&mut topology, t);

        assert!(
            m_step.torque_net.norm() < 1e-14,
            "Net torque {:e} failed to cancel at t={}",
            m_step.torque_net.norm(),
            t
        );
        assert!(
            (m_step.torque_cancellation_ratio - 1.0).abs() < 1e-6,
            "Torque cancellation ratio {} regressed at t={}",
            m_step.torque_cancellation_ratio,
            t
        );
    }
}

#[test]
fn test_angular_momentum_conservation() {
    let mut topology = JackTopology::new(1.5);
    let engine = GyroMechanicsEngine::new(80.0, 20.0, 1.5);

    let m_step_0 = engine.step(&mut topology, 0.0);
    let l_0 = m_step_0.angular_momentum_norm;
    assert!(l_0 > 0.0, "Angular momentum must be strictly positive");

    for step in 1..=30 {
        let t = step as f64 * 0.05;
        let m_step = engine.step(&mut topology, t);
        let delta_l = (m_step.angular_momentum_norm - l_0).abs();
        assert!(
            delta_l < 1e-12,
            "Angular momentum drift {:e} observed at t={}",
            delta_l,
            t
        );
    }
}

#[test]
fn test_quantum_phase_lock_core_fidelity() {
    let q_engine = QuantumLockEngine::new(2.5, 60.0, 0.95);

    for step in 0..50 {
        let t = step as f64 * 0.01;
        let q_step = q_engine.evaluate_step(t);

        assert!(
            q_step.core_fidelity >= 0.99999999,
            "Core fidelity {} dropped below unity at t={}",
            q_step.core_fidelity,
            t
        );
        assert!(
            q_step.dfs_confinement_error < 1e-14,
            "DFS confinement error {:e} exceeded threshold at t={}",
            q_step.dfs_confinement_error,
            t
        );
    }
}

#[test]
fn test_parity_correlator_anti_phase() {
    let q_engine = QuantumLockEngine::new(1.8, 30.0, 1.0);

    for step in 0..40 {
        let t = step as f64 * 0.025;
        let q_step = q_engine.evaluate_step(t);

        assert!(
            (q_step.parity_correlator_zab - (-1.0)).abs() < 1e-14,
            "Parity correlator {} deviated from -1.0 at t={}",
            q_step.parity_correlator_zab,
            t
        );
    }
}

#[test]
fn test_gyroscopic_stabilizer_full_simulation() {
    let config = GyroscopicConfig::default();
    let engine = GyroscopicStabilizerEngine::new();
    let report = engine.simulate(&config);

    assert_eq!(report.mechanics_steps.len(), config.steps + 1);
    assert_eq!(report.quantum_steps.len(), config.steps + 1);
    assert!(report.max_torque_net_norm < 1e-14);
    assert!((report.mean_torque_cancellation_ratio - 1.0).abs() < 1e-6);
    assert!(report.min_core_fidelity >= 0.99999999);
    assert!((report.mean_parity_correlator - (-1.0)).abs() < 1e-14);
    assert!(report.max_center_of_mass_drift < 1e-14);
}
