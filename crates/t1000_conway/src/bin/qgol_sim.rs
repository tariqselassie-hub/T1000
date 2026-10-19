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


fn run_glider_soliton_scenario(width: usize, height: usize) -> ScenarioTelemetry {
    println!(
        "{}[SCENARIO 1: Bethe-String Glider Soliton Propagation]{}",
        colors::POLAR_WHITE,
        colors::RESET
    );
    let lattice = Lattice2D::new(width, height, BoundaryCondition::Toroidal);
    let mut sim = QuantumConwaySimulator::new(lattice, HolonomyParams::default());
    PatternFactory::inject_pattern(PatternType::Glider, &sim.lattice, &mut sim.state, 2, 2);

    println!(
        "{}",
        TelemetryVisualizer::render_grid(&sim.lattice, &sim.state)
    );
    let start = Instant::now();
    let mut history = Vec::new();

    for _ in 0..12 {
        let telem = sim.step();
        history.push(telem);
    }
    let elapsed = start.elapsed().as_micros();

    let last = history.last().unwrap();
    println!(
        "{}",
        TelemetryVisualizer::render_grid(&sim.lattice, &sim.state)
    );
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
    println!(
        "{}[SCENARIO 2: Fractional Fermi Sea Topological Protection]{}",
        colors::POLAR_WHITE,
        colors::RESET
    );
    let lattice = Lattice2D::new(width, height, BoundaryCondition::Toroidal);
    let mut sim = QuantumConwaySimulator::new(lattice, HolonomyParams::default());
    PatternFactory::inject_pattern(
        PatternType::FractionalFermiSeaBubble,
        &sim.lattice,
        &mut sim.state,
        6,
        6,
    );

    println!(
        "{}",
        TelemetryVisualizer::render_grid(&sim.lattice, &sim.state)
    );
    let start = Instant::now();
    let mut history = Vec::new();

    for _ in 0..8 {
        let telem = sim.step();
        history.push(telem);
    }
    let elapsed = start.elapsed().as_micros();

    let last = history.last().unwrap();
    println!(
        "{}",
        TelemetryVisualizer::render_grid(&sim.lattice, &sim.state)
    );
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
    println!(
        "{}[SCENARIO 3: Coherent Cat Blinker Revival]{}",
        colors::POLAR_WHITE,
        colors::RESET
    );
    let lattice = Lattice2D::new(width, height, BoundaryCondition::Toroidal);
    let mut sim = QuantumConwaySimulator::new(lattice, HolonomyParams::default());
    PatternFactory::inject_cat_blinker(&sim.lattice, &mut sim.state, 5, 5);

    println!(
        "{}",
        TelemetryVisualizer::render_grid(&sim.lattice, &sim.state)
    );
    let start = Instant::now();
    let mut history = Vec::new();

    for _ in 0..10 {
        let telem = sim.step();
        history.push(telem);
    }
    let elapsed = start.elapsed().as_micros();

    let last = history.last().unwrap();
    println!(
        "{}",
        TelemetryVisualizer::render_grid(&sim.lattice, &sim.state)
    );
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

fn run_quantum_glider_collision_scenario(width: usize, height: usize) -> ScenarioTelemetry {
    println!(
        "{}[SCENARIO 4: Quantum Glider Collision & Beam-Splitter Interference (Δθ = π/2)]{}",
        colors::POLAR_WHITE,
        colors::RESET
    );
    let lattice = Lattice2D::new(width, height, BoundaryCondition::Toroidal);
    let mut sim = QuantumConwaySimulator::new(lattice, HolonomyParams::default());
    let phase_rad = std::f64::consts::PI * 0.5;

    // Launch Glider A at (2, 2) SE, Glider B at (width - 7, 2) SW
    t1000_conway::setup_glider_collision(&mut sim, 2, 2, width.saturating_sub(7), 2, phase_rad);

    println!(
        "{}",
        TelemetryVisualizer::render_grid(&sim.lattice, &sim.state)
    );
    let start = Instant::now();
    let mut history = Vec::new();

    for _ in 0..16 {
        let telem = sim.step();
        history.push(telem);
    }
    let elapsed = start.elapsed().as_micros();

    let last = history.last().unwrap();
    println!(
        "{}",
        TelemetryVisualizer::render_grid(&sim.lattice, &sim.state)
    );
    println!("{}", TelemetryVisualizer::format_banner(last));

    ScenarioTelemetry {
        scenario: "Quantum Glider Beam-Splitter Collision".to_string(),
        total_steps: 16,
        elapsed_us: elapsed,
        final_population: last.total_population,
        final_ffs_population: last.ffs_population,
        max_norm_error: last.max_norm_error,
        entanglement_entropy: last.entanglement_entropy,
        mean_momentum_density: last.mean_momentum_density,
        history,
    }
}

#[derive(Debug, Serialize)]
struct ScaleBenchmarkResult {
    grid_dimension: String,
    total_cells: usize,
    latency_us_per_step: f64,
    throughput_cells_per_sec: f64,
    norm_conservation_error: f64,
}

fn run_multi_scale_simd_benchmark() -> Vec<ScaleBenchmarkResult> {
    println!("\n┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│             T-1000 CONWAY MULTI-SCALE AVX2 SIMD ACCELERATION BENCHMARK       │");
    println!("└──────────────────────────────────────────────────────────────────────────────┘");

    let scales = [(16, 16), (32, 32), (64, 64)];
    let mut results = Vec::new();

    for (w, h) in scales {
        let total = w * h;
        let lattice = Lattice2D::new(w, h, BoundaryCondition::Toroidal);
        let mut sim = QuantumConwaySimulator::new(lattice, HolonomyParams::default());
        PatternFactory::inject_pattern(PatternType::Glider, &sim.lattice, &mut sim.state, w / 4, h / 4);

        // Warmup
        for _ in 0..5 {
            let _ = sim.step();
        }

        let iters = 20;
        let t0 = Instant::now();
        let mut last_err = 0.0f64;
        for _ in 0..iters {
            let telem = sim.step();
            last_err = telem.max_norm_error;
        }
        let total_us = t0.elapsed().as_micros() as f64;
        let us_per_step = total_us / f64::from(iters);
        let tput_cells = (f64::from(total as u32) * 1_000_000.0) / us_per_step;

        println!(
            "  Grid {:2}x{:2} ({:4} cells) │ Latency: {:8.2} us/tick │ Throughput: {:10.1} cells/s │ Err: {:.2e}",
            w, h, total, us_per_step, tput_cells, last_err
        );

        results.push(ScaleBenchmarkResult {
            grid_dimension: format!("{}x{}", w, h),
            total_cells: total,
            latency_us_per_step: us_per_step,
            throughput_cells_per_sec: tput_cells,
            norm_conservation_error: last_err,
        });
    }

    results
}

fn main() {
    let width = 16;
    let height = 16;

    println!(
        "{}[T-1000 Sovereign Quantum Conway Game of Life Engine]{}",
        colors::POLAR_WHITE,
        colors::RESET
    );
    println!(
        "Lattice Geometry: {}x{} Toroidal | Generalized Exclusion Statistics: alpha = 2",
        width, height
    );

    let s1 = run_glider_soliton_scenario(width, height);
    let s2 = run_ffs_overpopulation_protection_scenario(width, height);
    let s3 = run_cat_blinker_scenario(width, height);
    let s4 = run_quantum_glider_collision_scenario(24, 24);

    let bench_results = run_multi_scale_simd_benchmark();

    // S-Matrix interferometry sweep
    println!("\n--- Mach-Zehnder Glider Interferometry Sweep (16 Phase Points) ---");
    let report = t1000_conway::sweep_glider_interferometry(24, 24, 16, 12);
    println!(
        "  Visibility V: {:.4} │ Trans range: [{:.3}, {:.3}] │ Max Entanglement: {:.4} nats",
        report.visibility, report.min_transmission, report.max_transmission, report.max_entanglement_nats
    );

    let results = serde_json::json!({
        "timestamp": "2026-10-19T12:00:00Z",
        "license": "MIT OR Apache-2.0",
        "grid_size": (width, height),
        "scenarios": vec![s1, s2, s3, s4],
        "multi_scale_benchmark": bench_results,
        "interferometry_sweep": report,
    });

    let out_dir = Path::new("results/experiments");
    if !out_dir.exists() {
        let _ = fs::create_dir_all(out_dir);
    }
    let json_path = out_dir.join("qgol_simulation_telemetry.json");
    let json_data = serde_json::to_string_pretty(&results).expect("Failed to serialize telemetry");
    fs::write(&json_path, json_data).expect("Failed to write telemetry JSON");
    println!("\n[+] Saved Conway telemetry & S-matrix results to: {}", json_path.display());
}
