// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Quantum Conway Game of Life Simulation & Benchmark Driver.
//! Demonstrates unitary QCA evolution, Fractional Fermi Sea topological protection,
//! Bethe-string glider solitons, and coherent Schrödinger Cat blinker revivals.

use serde::Serialize;
use std::fs;
use std::path::Path;
use std::time::Instant;
use t1000_conway::{
    colors, BoundaryCondition, HolonomyParams, Lattice2D, PatternFactory, PatternType,
    QuantumConwaySimulator, StepTelemetry, TelemetryVisualizer,
};

#[derive(Debug, Serialize)]
struct ScenarioTelemetry {
    scenario: String,
    total_steps: usize,
    elapsed_us: u128,
    final_population: f64,
    final_ffs_population: f64,
    max_norm_error: f64,
    entanglement_entropy: f64,
    mean_momentum_density: f64,
    history: Vec<StepTelemetry>,
}

#[derive(Debug, Serialize)]
struct QgolRunResults {
    timestamp: String,
    license: String,
    grid_size: (usize, usize),
    scenarios: Vec<ScenarioTelemetry>,
}

fn run_glider_soliton_scenario(width: usize, height: usize) -> ScenarioTelemetry {
    println!("{}[SCENARIO 1: Bethe-String Glider Soliton Propagation]{}", colors::POLAR_WHITE, colors::RESET);
    let lattice = Lattice2D::new(width, height, BoundaryCondition::Toroidal);
    let mut sim = QuantumConwaySimulator::new(lattice, HolonomyParams::default());
    PatternFactory::inject_pattern(PatternType::Glider, &sim.lattice, &mut sim.state, 2, 2);

    println!("{}", TelemetryVisualizer::render_grid(&sim.lattice, &sim.state));
    let start = Instant::now();
    let mut history = Vec::new();

    for _ in 0..12 {
        let telem = sim.step();
        history.push(telem);
    }
    let elapsed = start.elapsed().as_micros();

    let last = history.last().unwrap();
    println!("{}", TelemetryVisualizer::render_grid(&sim.lattice, &sim.state));
    println!("{}", TelemetryVisualizer::format_banner(last));

    ScenarioTelemetry {
        scenario: "Bethe-String Glider Soliton".to_string(),
        total_steps: 12,
        elapsed_us: elapsed,
        final_population: last.total_population,
        final_ffs_population: last.ffs_population,
        max_norm_error: last.max_norm_error,
        entanglement_entropy: last.entanglement_entropy,
        mean_momentum_density: last.mean_momentum_density,
        history,
    }
}

fn run_ffs_overpopulation_protection_scenario(width: usize, height: usize) -> ScenarioTelemetry {
    println!("{}[SCENARIO 2: Fractional Fermi Sea Topological Protection]{}", colors::POLAR_WHITE, colors::RESET);
    let lattice = Lattice2D::new(width, height, BoundaryCondition::Toroidal);
    let mut sim = QuantumConwaySimulator::new(lattice, HolonomyParams::default());
    PatternFactory::inject_pattern(PatternType::FractionalFermiSeaBubble, &sim.lattice, &mut sim.state, 6, 6);

    println!("{}", TelemetryVisualizer::render_grid(&sim.lattice, &sim.state));
    let start = Instant::now();
    let mut history = Vec::new();

    for _ in 0..8 {
        let telem = sim.step();
        history.push(telem);
    }
    let elapsed = start.elapsed().as_micros();

    let last = history.last().unwrap();
    println!("{}", TelemetryVisualizer::render_grid(&sim.lattice, &sim.state));
    println!("{}", TelemetryVisualizer::format_banner(last));

    ScenarioTelemetry {
        scenario: "FFS Overpopulation Protection".to_string(),
        total_steps: 8,
        elapsed_us: elapsed,
        final_population: last.total_population,
        final_ffs_population: last.ffs_population,
        max_norm_error: last.max_norm_error,
        entanglement_entropy: last.entanglement_entropy,
        mean_momentum_density: last.mean_momentum_density,
        history,
    }
}

fn run_cat_blinker_scenario(width: usize, height: usize) -> ScenarioTelemetry {
    println!("{}[SCENARIO 3: Coherent Cat Blinker Revival]{}", colors::POLAR_WHITE, colors::RESET);
    let lattice = Lattice2D::new(width, height, BoundaryCondition::Toroidal);
    let mut sim = QuantumConwaySimulator::new(lattice, HolonomyParams::default());
    PatternFactory::inject_cat_blinker(&sim.lattice, &mut sim.state, 5, 5);

    println!("{}", TelemetryVisualizer::render_grid(&sim.lattice, &sim.state));
    let start = Instant::now();
    let mut history = Vec::new();

    for _ in 0..10 {
        let telem = sim.step();
        history.push(telem);
    }
    let elapsed = start.elapsed().as_micros();

    let last = history.last().unwrap();
    println!("{}", TelemetryVisualizer::render_grid(&sim.lattice, &sim.state));
    println!("{}", TelemetryVisualizer::format_banner(last));

    ScenarioTelemetry {
        scenario: "Coherent Cat Blinker".to_string(),
        total_steps: 10,
        elapsed_us: elapsed,
        final_population: last.total_population,
        final_ffs_population: last.ffs_population,
        max_norm_error: last.max_norm_error,
        entanglement_entropy: last.entanglement_entropy,
        mean_momentum_density: last.mean_momentum_density,
        history,
    }
}

fn main() {
    let width = 16;
    let height = 16;

    println!("{}[T-1000 Sovereign Quantum Conway Game of Life Engine]{}", colors::POLAR_WHITE, colors::RESET);
    println!("Lattice Geometry: {}x{} Toroidal | Generalized Exclusion Statistics: alpha = 2", width, height);

    let s1 = run_glider_soliton_scenario(width, height);
    let s2 = run_ffs_overpopulation_protection_scenario(width, height);
    let s3 = run_cat_blinker_scenario(width, height);

    let results = QgolRunResults {
        timestamp: "2026-10-18T13:30:00Z".to_string(),
        license: "MIT OR Apache-2.0".to_string(),
        grid_size: (width, height),
        scenarios: vec![s1, s2, s3],
    };

    let out_dir = Path::new("results/experiments");
    if !out_dir.exists() {
        let _ = fs::create_dir_all(out_dir);
    }
    let json_path = out_dir.join("qgol_simulation_telemetry.json");
    let json_data = serde_json::to_string_pretty(&results).expect("Failed to serialize telemetry");
    fs::write(&json_path, json_data).expect("Failed to write telemetry JSON");
    println!("Saved telemetry results to: {}", json_path.display());
}
