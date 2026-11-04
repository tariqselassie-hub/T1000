// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Water (H2O) 2D Energy & Linearization Barrier Benchmark
//!
//! Maps ab-initio potential energy surface, analyzes 180° planar linearization barrier,
//! tracks Walsh molecular orbital correlation curves, and calculates harmonic vibrational modes.

use std::path::PathBuf;
use t1000::results;
use t1000::water_pes::{WaterPesConfig, WaterPesEngine};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let record = args.iter().any(|a| a == "--record" || a == "-r")
        || std::env::var("T1000_RECORD").map(|v| v == "1").unwrap_or(false);

    println!("[1/3] Initializing Water (H2O) 2D Potential Energy Surface calculation...");
    let config = WaterPesConfig {
        r_min: 0.80,
        r_max: 1.30,
        r_steps: 11,
        theta_min: 80.0,
        theta_max: 180.0,
        theta_steps: 21,
        walsh_steps: 25,
    };

    println!("[2/3] Solving ab-initio STO-3G Restricted Hartree-Fock across 231 grid points...");
    let engine = WaterPesEngine::new(config);
    let report = engine
        .compute_manifold()
        .expect("Water PES manifold computation failed");

    println!("[3/3] Compiling vibrational normal modes and Walsh correlation report...\n");

    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("║       WATER (H2O) 2D ENERGY SURFACE & LINEARIZATION BARRIER ENGINE          ║");
    println!("║       Ab-Initio Electronic Structure, Walsh Correlation & Vibrations        ║");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ Molecular Symmetry Group:       C_2v (Bent) -> D_inf_h (Linear)             │");
    println!("│ Active Basis Set:               STO-3G (7 AOs, 14 Spin-Orbitals, 10 e-)      │");
    println!("│ Equilibrium Bond Length r_OH:   {:.4} Angstroms                             │", report.equilibrium_r_oh);
    println!("│ Equilibrium Bond Angle theta:   {:.2}°                                      │", report.equilibrium_theta_deg);
    println!("│ Ground State Electronic Energy: {:.6} Hartree                              │", report.equilibrium_energy);
    println!("│ Equilibrium Dipole Moment:      {:.3} Debye                                 │", report.grid.min_energy_point.dipole_moment_debye);
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ 180° LINEARIZATION BARRIER METRICS:                                         │");
    println!("│ Linear Configuration Energy:    {:.6} Hartree                              │", report.linearization.linear_energy);
    println!("│ Barrier Height (Hartree):       {:.4} Hartree                               │", report.linearization.barrier_hartree);
    println!("│ Barrier Height (eV):            {:.3} eV                                    │", report.linearization_barrier_ev);
    println!("│ Barrier Height (kcal/mol):      {:.2} kcal/mol                              │", report.linearization_barrier_kcal_mol);
    println!("│ Barrier Height (wavenumbers):   {:.1} cm^-1                                 │", report.linearization_barrier_cm1);
    println!("│ Linear Dipole Moment (180°):    {:.6} Debye (Centrosymmetric Inversion)    │", report.linearization.dipole_moment_at_180);
    println!("│ Walsh 3a1 - 1b1 Degeneracy Gap: {:.2e} Hartree (1pi_u Convergence)         │", report.linearization.degeneracy_gap_at_180);
    println!("└─────────────────────────────────────────────────────────────────────────────┘\n");

    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ VIBRATIONAL NORMAL MODES & ZERO-POINT ENERGY (WILSON GF MATRIX)             │");
    println!("├───────┬──────────────────────┬──────────┬──────────────┬────────────────────┤");
    println!("│ Mode  │ Description          │ Symmetry │ Frequency    │ Frequency (THz)    │");
    println!("├───────┼──────────────────────┼──────────┼──────────────┼────────────────────┤");
    for m in &report.vibrations.modes {
        println!(
            "│ nu_{:<2} │ {:<20} │ {:<8} │ {:>8.1} cm^-1│ {:>14.2} THz │",
            m.mode_index, m.name, m.symmetry_label, m.frequency_cm1, m.frequency_thz
        );
    }
    println!("├───────┴──────────────────────┴──────────┴──────────────┴────────────────────┤");
    println!("│ Harmonic Zero-Point Energy:     {:.1} cm^-1 ({:.2} kcal/mol)               │", report.zpe_cm1, report.vibrations.total_zpe_kcal_mol);
    println!("│ Bond Stretching Force Constant: {:.4} Hartree / A^2                        │", report.vibrations.force_kr);
    println!("│ Angle Bending Force Constant:   {:.4} Hartree / rad^2                      │", report.vibrations.force_ktheta);
    println!("└─────────────────────────────────────────────────────────────────────────────┘\n");

    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ WALSH ORBITAL CORRELATION: 3a_1 vs 1b_1 DEGENERACY CONVERGENCE             │");
    println!("├────────────┬──────────────┬──────────────┬──────────────┬───────────────────┤");
    println!("│ Theta (°)  │ Total Energy │ eps(3a_1)    │ eps(1b_1)    │ Degeneracy Gap    │");
    println!("├────────────┼──────────────┼──────────────┼──────────────┼───────────────────┤");
    for (i, s) in report.linearization.walsh_states.iter().enumerate() {
        if i % 4 == 0 || i == report.linearization.walsh_states.len() - 1 {
            println!(
                "│ {:>9.1}° │ {:>12.6} │ {:>12.6} │ {:>12.6} │ {:>15.2e} │",
                s.theta_deg, s.total_energy, s.orbital_3a1, s.orbital_1b1, s.degeneracy_gap
            );
        }
    }
    println!("└────────────┴──────────────┴──────────────┴──────────────┴───────────────────┘");

    let serialized = serde_json::to_string_pretty(&report).expect("Serialization failed");

    if record {
        let out_path = PathBuf::from("results/experiments/water_pes.json");
        results::write(&out_path, &serialized).expect("Failed to write results");
        println!("\nWater PES telemetry recorded to: {}", out_path.display());
    } else {
        println!("\n[bench] Pass --record (or T1000_RECORD=1) to save official results.");
    }
}
