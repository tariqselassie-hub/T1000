// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Quantum Universe & Cosmological Spacetime Simulation Benchmark
//!
//! Simulates microscopic quantum field fluctuations evolving on expanding FLRW
//! geometries, tracking Bogoliubov particle creation, super-horizon freeze-out,
//! and cosmological entanglement entropy accumulation across the Hubble horizon.

use std::path::PathBuf;
use t1000::cosmology::{CosmologyConfig, CosmologyEngine, ExpansionProfile};
use t1000::results;

fn parse_args() -> (CosmologyConfig, PathBuf) {
    let mut config = CosmologyConfig::default();
    let mut out_path = PathBuf::from("results/experiments/universe_cosmology.json");

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--profile" if i + 1 < args.len() => {
                config.profile = match args[i + 1].to_lowercase().as_str() {
                    "radiation" => ExpansionProfile::RadiationDominated,
                    "matter" => ExpansionProfile::MatterDominated,
                    "bounce" => ExpansionProfile::BouncingCosmology,
                    "minkowski" => ExpansionProfile::StaticMinkowski,
                    _ => ExpansionProfile::DeSitter,
                };
                i += 2;
            }
            "--hubble" if i + 1 < args.len() => {
                config.hubble_rate = args[i + 1].parse().unwrap_or(1.2);
                i += 2;
            }
            "--modes" if i + 1 < args.len() => {
                config.num_modes = args[i + 1].parse().unwrap_or(16);
                i += 2;
            }
            "--steps" if i + 1 < args.len() => {
                config.time_steps = args[i + 1].parse().unwrap_or(120);
                i += 2;
            }
            "--out" if i + 1 < args.len() => {
                out_path = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    (config, out_path)
}

fn print_cosmology_summary(report: &t1000::cosmology::CosmologicalSimulationReport) {
    let profile_name = match report.config.profile {
        ExpansionProfile::DeSitter => "de Sitter Cosmic Inflation (Exponential)",
        ExpansionProfile::RadiationDominated => "Radiation-Dominated Expansion (a ~ t^1/2)",
        ExpansionProfile::MatterDominated => "Matter-Dominated Expansion (a ~ t^2/3)",
        ExpansionProfile::BouncingCosmology => "Non-Singular Quantum Bounce",
        ExpansionProfile::StaticMinkowski => "Static Flat Minkowski Spacetime",
    };

    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("║       QUANTUM UNIVERSE & COSMOLOGICAL SPACETIME SIMULATION                  ║");
    println!("║       Bogoliubov Vacuum Particle Creation & Horizon Entanglement            ║");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ Metric Expansion Profile:       {:<43} │", profile_name);
    println!("│ Hubble Expansion Parameter:     {:<43.4} │", report.config.hubble_rate);
    println!("│ Spatial Resolution:             {:<43} │", format!("{} modes (Box L = {:.1})", report.config.num_modes, report.config.box_length));
    println!("│ Final Cosmic Scale Factor a(t): {:<43.4} │", report.final_spectrum.scale_factor);
    println!("│ Total Particles Created N_tot:  {:<43.4} │", report.total_particles_created);
    println!("│ Gibbons-Hawking Temperature:    {:<43.6} │", report.final_spectrum.gibbons_hawking_temp);
    println!("│ Peak Horizon Entanglement (S):  {:<43.4} │", report.peak_entanglement_entropy);
    println!("│ Super-Horizon Frozen Modes:     {:<43} │", format!("{}/{} modes", report.final_horizon.super_horizon_modes_count, report.config.num_modes));
    println!("│ Max Wronskian Unitarity Error:  {:<43.2e} │", report.max_unitarity_error);
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ MOMENTUM SPECTRUM: CREATED PARTICLES & SQUEEZING AT FINAL EPOCH             │");
    println!("├───────┬──────────────┬───────────────┬───────────────┬──────────────────────┤");
    println!("│ Mode  │ Momentum (k) │ Particles (n) │ Squeezing (r) │ Horizon Status       │");
    println!("├───────┼──────────────┼───────────────┼───────────────┼──────────────────────┤");

    for m in report.final_spectrum.mode_reports.iter().take(10) {
        let is_super = m.momentum_k < report.final_spectrum.scale_factor * report.final_spectrum.hubble_parameter;
        let status = if is_super { "SUPER-HORIZON (FROZEN)" } else { "SUB-HORIZON (OSCILL)" };
        println!(
            "│ {:<5} │ {:12.4} │ {:13.4} │ {:13.4} │ {:<20} │",
            m.mode_index, m.momentum_k, m.particle_number, m.squeezing_param, status
        );
    }
    println!("└───────┴──────────────┴───────────────┴───────────────┴──────────────────────┘");
}

fn main() {
    let (config, out_path) = parse_args();

    println!("[1/2] Initializing quantum field modes in Bunch-Davies vacuum...");
    let mut engine = CosmologyEngine::new(config);

    println!("[2/2] Evolving quantum field on dynamic curved spacetime metric...");
    let report = engine.run_simulation();

    print_cosmology_summary(&report);

    let serialized = serde_json::to_string_pretty(&report).expect("Failed to serialize telemetry");
    results::write(&out_path, &serialized).expect("Failed to write results");
    println!("\nCosmological telemetry recorded to: {}", out_path.display());
}
