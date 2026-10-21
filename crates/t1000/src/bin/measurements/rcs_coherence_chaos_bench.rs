// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # High-Depth Physical Noise RCS Chaos & Coherence Transition Benchmark
//!
//! Evaluates the breakdown of quantum speckle patterns and Porter-Thomas chaos
//! under layer-by-layer open-system Lindbladian decoherence ($T_1, T_2^*, p_{\text{depol}}$).
//! Quantifies the critical noise horizon and tests whether compiler-injected Dynamical
//! Decoupling (XY4) and Symplectic Manifold Re-projection restore Cross-Entropy Benchmarking (XEB).

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use t1000::coherence::{DynamicalDecouplingKind, NoiseChannelParams};
use t1000::rcs::{NoisyRcsConfig, RcsConfig, RcsSimulator, StabilizationStrategy};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthPointTelemetry {
    pub qubits: usize,
    pub depth: usize,
    pub ideal_xeb: f64,
    pub ideal_ks_passed: bool,
    pub unprot_fidelity: f64,
    pub unprot_xeb: f64,
    pub unprot_l1_norm: f64,
    pub unprot_ks_passed: bool,
    pub stab_fidelity: f64,
    pub stab_xeb: f64,
    pub stab_l1_norm: f64,
    pub stab_ks_passed: bool,
    pub fidelity_gain_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseSweepPointTelemetry {
    pub depol_prob: f64,
    pub unprot_fidelity: f64,
    pub unprot_cross_xeb: f64,
    pub unprot_ks_stat: f64,
    pub unprot_ks_passed: bool,
    pub stab_fidelity: f64,
    pub stab_cross_xeb: f64,
    pub stab_ks_stat: f64,
    pub stab_ks_passed: bool,
    pub xeb_recovery_delta: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RcsCoherenceChaosReport {
    pub timestamp: String,
    pub base_seed: u64,
    pub noise_model: NoiseChannelParams,
    pub gate_time_us: f64,
    pub depth_sweep: Vec<DepthPointTelemetry>,
    pub noise_sweep: Vec<NoiseSweepPointTelemetry>,
}

fn run_single_depth_point(
    qubits: usize,
    depth: usize,
    seed: u64,
    noise: NoiseChannelParams,
    gate_time_us: f64,
) -> DepthPointTelemetry {
    let base_cfg = RcsConfig {
        qubits,
        depth,
        seed,
    };
    let sim = RcsSimulator::new(base_cfg.clone());
    let (ideal_state, ideal_report) = sim.execute();

    let unprot_cfg = NoisyRcsConfig {
        base: base_cfg.clone(),
        noise,
        gate_time_us,
        stabilization: StabilizationStrategy::None,
    };
    let (_unprot_state, unprot_rep) = sim.execute_noisy(&unprot_cfg, Some(&ideal_state));

    let stab_cfg = NoisyRcsConfig {
        base: base_cfg,
        noise,
        gate_time_us,
        stabilization: StabilizationStrategy::Combined(DynamicalDecouplingKind::Xy4),
    };
    let (_stab_state, stab_rep) = sim.execute_noisy(&stab_cfg, Some(&ideal_state));

    let fid_unprot = unprot_rep.final_fidelity_vs_ideal;
    let fid_stab = stab_rep.final_fidelity_vs_ideal;
    let gain_pct = if fid_unprot > 1e-12 {
        ((fid_stab - fid_unprot) / fid_unprot) * 100.0
    } else {
        0.0
    };

    let unprot_l1 = unprot_rep
        .trajectory
        .last()
        .map(|t| t.l1_coherence_norm)
        .unwrap_or(0.0);
    let stab_l1 = stab_rep
        .trajectory
        .last()
        .map(|t| t.l1_coherence_norm)
        .unwrap_or(0.0);

    DepthPointTelemetry {
        qubits,
        depth,
        ideal_xeb: ideal_report.pt_report.exact_xeb,
        ideal_ks_passed: ideal_report.pt_report.passes_ks_test,
        unprot_fidelity: fid_unprot,
        unprot_xeb: unprot_rep.cross_xeb_with_ideal,
        unprot_l1_norm: unprot_l1,
        unprot_ks_passed: unprot_rep.pt_report.passes_ks_test,
        stab_fidelity: fid_stab,
        stab_xeb: stab_rep.cross_xeb_with_ideal,
        stab_l1_norm: stab_l1,
        stab_ks_passed: stab_rep.pt_report.passes_ks_test,
        fidelity_gain_pct: gain_pct,
    }
}

fn run_single_noise_point(
    qubits: usize,
    depth: usize,
    seed: u64,
    depol_prob: f64,
    gate_time_us: f64,
) -> NoiseSweepPointTelemetry {
    let base_cfg = RcsConfig {
        qubits,
        depth,
        seed,
    };
    let sim = RcsSimulator::new(base_cfg.clone());
    let (ideal_state, _ideal_report) = sim.execute();

    let noise = NoiseChannelParams {
        t1_us: 45.0,
        t2_us: 60.0,
        depolarizing_prob: depol_prob,
    };

    let unprot_cfg = NoisyRcsConfig {
        base: base_cfg.clone(),
        noise,
        gate_time_us,
        stabilization: StabilizationStrategy::None,
    };
    let (_unprot_state, unprot_rep) = sim.execute_noisy(&unprot_cfg, Some(&ideal_state));

    let stab_cfg = NoisyRcsConfig {
        base: base_cfg,
        noise,
        gate_time_us,
        stabilization: StabilizationStrategy::Combined(DynamicalDecouplingKind::Xy4),
    };
    let (_stab_state, stab_rep) = sim.execute_noisy(&stab_cfg, Some(&ideal_state));

    let delta = stab_rep.cross_xeb_with_ideal - unprot_rep.cross_xeb_with_ideal;

    NoiseSweepPointTelemetry {
        depol_prob,
        unprot_fidelity: unprot_rep.final_fidelity_vs_ideal,
        unprot_cross_xeb: unprot_rep.cross_xeb_with_ideal,
        unprot_ks_stat: unprot_rep.pt_report.ks_statistic,
        unprot_ks_passed: unprot_rep.pt_report.passes_ks_test,
        stab_fidelity: stab_rep.final_fidelity_vs_ideal,
        stab_cross_xeb: stab_rep.cross_xeb_with_ideal,
        stab_ks_stat: stab_rep.pt_report.ks_statistic,
        stab_ks_passed: stab_rep.pt_report.passes_ks_test,
        xeb_recovery_delta: delta,
    }
}

fn print_depth_sweep_table(points: &[DepthPointTelemetry]) {
    println!("┌────┬───────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐");
    println!("│ Q  │ Depth │ Ideal XEB    │ Unprot F_id  │ Unprot XEB   │ Stab F_id    │ Stab XEB     │ Gain (%)     │");
    println!("├────┼───────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤");
    for p in points {
        println!(
            "│ {:2} │ {:5} │ {:12.4} │ {:12.6} │ {:12.6} │ {:12.6} │ {:12.6} │ {:+11.2}% │",
            p.qubits,
            p.depth,
            p.ideal_xeb,
            p.unprot_fidelity,
            p.unprot_xeb,
            p.stab_fidelity,
            p.stab_xeb,
            p.fidelity_gain_pct
        );
    }
    println!("└────┴───────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘");
}

fn print_noise_sweep_table(points: &[NoiseSweepPointTelemetry]) {
    println!("┌──────────────┬──────────────┬──────────────┬──────────┬──────────────┬──────────────┬──────────┬──────────────┐");
    println!("│ Depol P      │ Unprot F_id  │ Unprot XEB   │ Unprot KS│ Stab F_id    │ Stab XEB     │ Stab KS  │ Delta XEB    │");
    println!("├──────────────┼──────────────┼──────────────┼──────────┼──────────────┼──────────────┼──────────┼──────────────┤");
    for p in points {
        println!(
            "│ {:12.5} │ {:12.6} │ {:12.6} │ {:8} │ {:12.6} │ {:12.6} │ {:8} │ {:+12.6} │",
            p.depol_prob,
            p.unprot_fidelity,
            p.unprot_cross_xeb,
            if p.unprot_ks_passed { "PASS" } else { "FAIL" },
            p.stab_fidelity,
            p.stab_cross_xeb,
            if p.stab_ks_passed { "PASS" } else { "FAIL" },
            p.xeb_recovery_delta
        );
    }
    println!("└──────────────┴──────────────┴──────────────┴──────────┴──────────────┴──────────────┴──────────┴──────────────┘");
}

fn main() {
    println!("==========================================================================================");
    println!("        T-1000 SOVEREIGN BENCHMARK: RCS CHAOS & PHYSICAL NOISE COHERENCE HORIZON          ");
    println!("==========================================================================================");
    println!(" Microarchitecture: AVX2+FMA Double-Precision SOA Statevector                             ");
    println!(" Physical Model   : T1=45us, T2*=60us, Lindbladian Continuous Decay, Depolarizing Kick    ");
    println!(" Stabilization    : Interleaved Symmetrized XY4 Dynamical Decoupling + Manifold Pinning   ");
    println!("------------------------------------------------------------------------------------------");

    let seed = 0x5EED_0042;
    let gate_time_us = 0.20;
    let standard_noise = NoiseChannelParams {
        t1_us: 45.0,
        t2_us: 60.0,
        depolarizing_prob: 0.001,
    };

    println!("\n[PART 1] DEPTH SCRAMBLING & QUANTUM FIDELITY CLIFF (14 QUBITS & 16 QUBITS)");
    let depths = vec![4, 8, 12, 16, 20, 24];
    let mut depth_points = Vec::new();

    for &q in &[14, 16] {
        for &d in &depths {
            let p = run_single_depth_point(q, d, seed, standard_noise, gate_time_us);
            depth_points.push(p);
        }
    }
    print_depth_sweep_table(&depth_points);

    println!(
        "\n[PART 2] NOISE RESILIENCE & KS GOODNESS-OF-FIT SWEEP (N = 14 QUBITS, D = 16 LAYERS)"
    );
    let noise_levels = vec![0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02];
    let mut noise_points = Vec::new();

    for &prob in &noise_levels {
        let p = run_single_noise_point(14, 16, seed, prob, gate_time_us);
        noise_points.push(p);
    }
    print_noise_sweep_table(&noise_points);

    let report = RcsCoherenceChaosReport {
        timestamp: "2026-10-18T05:00:00Z".to_string(),
        base_seed: seed,
        noise_model: standard_noise,
        gate_time_us,
        depth_sweep: depth_points,
        noise_sweep: noise_points,
    };

    let results_dir = Path::new("results");
    if !results_dir.exists() {
        let _ = fs::create_dir_all(results_dir);
    }
    let json_path = results_dir.join("rcs_physical_noise_chaos.json");
    if let Ok(serialized) = serde_json::to_string_pretty(&report) {
        let _ = t1000::results::write(&json_path, serialized);
        println!("\nTelemetry written to: {}", json_path.display());
    }
}
