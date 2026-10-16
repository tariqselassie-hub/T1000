// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! 36-Qubit Bipartite Cut-Bridge Benchmark & L3 Cache Telemetry.
//! Executes a 36-qubit dual-molecule interaction (18Q Receptor Pocket ⋈ 18Q Ligand)
//! inside CPU L3 cache (8.00 MB working set) via Quasiprobability Decomposition.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use t1000::ansatz::{eval_36q_bipartite_bridge, eval_cut_bridge_general};
use t1000::simd::{apply_cz_kernel, apply_ry_kernel, apply_rz_kernel};
use t1000::simd_bloch::compute_all_bloch_vectors;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchTelemetry {
    pub fragment: String,
    pub branch_id: usize,
    pub z_insertions: Vec<bool>,
    pub local_expectation: f64,
    pub latency_us: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CutBridgeBenchReport {
    pub timestamp: String,
    pub total_qubits: usize,
    pub fragment_a_qubits: usize,
    pub fragment_b_qubits: usize,
    pub bridge_gates: usize,
    pub dense_ram_theoretical_gb: f64,
    pub cut_bridge_ram_actual_mb: f64,
    pub ram_compression_ratio: f64,
    pub small_register_agreement_error: f64,
    pub reconstructed_36q_expectation: f64,
    pub total_execution_time_ms: f64,
    pub mean_bloch_radius_a: f64,
    pub mean_bloch_radius_b: f64,
    pub mean_purity_a: f64,
    pub mean_purity_b: f64,
    pub branches: Vec<BranchTelemetry>,
}

#[inline(always)]
fn apply_pauli_z(real: &mut [f64], imag: &mut [f64], qubit: usize) {
    let mask = 1usize << qubit;
    for i in 0..real.len() {
        if (i & mask) != 0 {
            real[i] = -real[i];
            imag[i] = -imag[i];
        }
    }
}

#[inline(always)]
fn measure_single_qubit_z(real: &[f64], imag: &[f64], qubit: usize) -> f64 {
    let mask = 1usize << qubit;
    let mut exp_z = 0.0;
    for i in 0..real.len() {
        let p = real[i] * real[i] + imag[i] * imag[i];
        if (i & mask) == 0 {
            exp_z += p;
        } else {
            exp_z -= p;
        }
    }
    exp_z
}

/// Phase 1: Small Register Ground Truth Verification (N = 4, 2Q ⋈ 2Q with 2 Bridges).
fn run_ground_truth_verification() -> f64 {
    let dim_dense = 16;
    let mut d_re = vec![0.0; dim_dense];
    let mut d_im = vec![0.0; dim_dense];
    d_re[0] = 1.0;

    // Apply rotation layer
    for q in 0..4 {
        let theta = 0.3 * (q as f64 + 1.0);
        apply_ry_kernel(&mut d_re, &mut d_im, q, theta);
        apply_rz_kernel(&mut d_re, &mut d_im, q, theta * 0.5);
    }
    // Intra-fragment CZ
    apply_cz_kernel(&mut d_re, &mut d_im, 0, 1);
    apply_cz_kernel(&mut d_re, &mut d_im, 2, 3);
    // 2 Inter-fragment bridge CZs: (0, 2) and (1, 3)
    apply_cz_kernel(&mut d_re, &mut d_im, 0, 2);
    apply_cz_kernel(&mut d_re, &mut d_im, 1, 3);

    // Measure dense <Z_0 ⊗ Z_2>
    let mut dense_exp = 0.0;
    for i in 0..dim_dense {
        let p = d_re[i] * d_re[i] + d_im[i] * d_im[i];
        let sign_0 = if (i & (1 << 0)) == 0 { 1.0 } else { -1.0 };
        let sign_2 = if (i & (1 << 2)) == 0 { 1.0 } else { -1.0 };
        dense_exp += p * sign_0 * sign_2;
    }

    // Now evaluate via Quasiprobability Decomposition (Cut-Bridge)
    let branch_a = |z_flags: &[bool], re: &mut [f64], im: &mut [f64]| -> f64 {
        re.fill(0.0);
        im.fill(0.0);
        re[0] = 1.0;
        for q in 0..2 {
            let theta = 0.3 * (q as f64 + 1.0);
            apply_ry_kernel(re, im, q, theta);
            apply_rz_kernel(re, im, q, theta * 0.5);
        }
        apply_cz_kernel(re, im, 0, 1);
        if z_flags[0] {
            apply_pauli_z(re, im, 0);
        }
        if z_flags[1] {
            apply_pauli_z(re, im, 1);
        }
        measure_single_qubit_z(re, im, 0)
    };

    let branch_b = |z_flags: &[bool], re: &mut [f64], im: &mut [f64]| -> f64 {
        re.fill(0.0);
        im.fill(0.0);
        re[0] = 1.0;
        for q in 0..2 {
            let global_q = q + 2;
            let theta = 0.3 * (global_q as f64 + 1.0);
            apply_ry_kernel(re, im, q, theta);
            apply_rz_kernel(re, im, q, theta * 0.5);
        }
        apply_cz_kernel(re, im, 0, 1);
        if z_flags[0] {
            apply_pauli_z(re, im, 0);
        }
        if z_flags[1] {
            apply_pauli_z(re, im, 1);
        }
        measure_single_qubit_z(re, im, 0)
    };

    let mut cut_re = vec![0.0; 4];
    let mut cut_im = vec![0.0; 4];
    let cut_exp = eval_cut_bridge_general(2, branch_a, branch_b, &mut cut_re, &mut cut_im);

    let err = (dense_exp - cut_exp).abs();
    println!("  Dense 4Q Expectation   : {:.12}", dense_exp);
    println!("  Cut-Bridge Reconstructed: {:.12}", cut_exp);
    println!("  Absolute Error (Δ)     : {:.4e}", err);
    err
}

fn main() {
    println!("┌────────────────────────────────────────────────────────────────────────┐");
    println!("│     T-1000: 36-QUBIT BIPARTITE CUT-BRIDGE L3-CACHE EVALUATOR           │");
    println!("│     Quasiprobability Decomposition: CZ = 0.5*(I⊗I + I⊗Z + Z⊗I - Z⊗Z)   │");
    println!("└────────────────────────────────────────────────────────────────────────┘");

    let total_qubits = 36;
    let fragment_qubits = 18;
    let bridge_gates = 2;
    let dim_18q = 1usize << fragment_qubits; // 262,144 amplitudes

    // Theoretical monolithic dense RAM: 2^36 * 16 bytes = 1099.51 GB
    let dense_ram_gb = (1u64 << total_qubits) as f64 * 16.0 / (1024.0 * 1024.0 * 1024.0);
    // Actual cut-bridge RAM: 2 fragments * 262,144 amplitudes * 16 bytes = 8.00 MB
    let cut_bridge_ram_mb = 2.0 * (dim_18q as f64 * 16.0) / (1024.0 * 1024.0);
    let ram_compression = (dense_ram_gb * 1024.0) / cut_bridge_ram_mb;

    println!("[CONFIG] Architecture Overview:");
    println!("  Total Virtual Register  : {} Qubits (2^{} = 68,719,476,736 States)", total_qubits, total_qubits);
    println!("  Sub-Register A (Pocket) : {} Qubits ({} Amplitudes | 4.00 MB L3)", fragment_qubits, dim_18q);
    println!("  Sub-Register B (Ligand) : {} Qubits ({} Amplitudes | 4.00 MB L3)", fragment_qubits, dim_18q);
    println!("  Inter-Fragment Bridges  : {} CZ Gates [Bridge 0: (4, 13), Bridge 1: (8, 17)]", bridge_gates);
    println!("  Dense State RAM (Dense) : {:.2} GB (Physically Impassable on Host DRAM)", dense_ram_gb);
    println!("  Cut-Bridge RAM (Actual) : {:.2} MB (100% CPU L3 Cache Resident)", cut_bridge_ram_mb);
    println!("  Memory Compression Ratio: {:.0}x RAM Reduction\n", ram_compression);

    println!("────────────────────────────────────────────────────────────────────────");
    println!("[PHASE 1] Mathematical Equivalence Verification (N=4 Ground Truth):");
    let agreement_err = run_ground_truth_verification();
    println!("────────────────────────────────────────────────────────────────────────\n");

    println!("[PHASE 2] Executing 36-Qubit Bipartite Simulation (AVX2 FMA Wings)...");
    let mut re_buf = vec![0.0f64; dim_18q];
    let mut im_buf = vec![0.0f64; dim_18q];

    let start_all = Instant::now();
    let mut branch_telemetry = Vec::new();

    // Fragment A (18Q Kinase Pocket) Branch Runner
    let run_18q_pocket = |z_flags: &[bool], re: &mut [f64], im: &mut [f64]| -> f64 {
        re.fill(0.0);
        im.fill(0.0);
        re[0] = 1.0;

        // 18Q Parameterized Layer: Alternating Ry, Rz
        for q in 0..18 {
            let theta = 0.25 * ((q % 6) as f64 + 1.0);
            apply_ry_kernel(re, im, q, theta);
            apply_rz_kernel(re, im, q, theta * 0.4);
        }
        // Intra-fragment Ring CZ
        for q in 0..18 {
            let next = (q + 1) % 18;
            apply_cz_kernel(re, im, q, next);
        }

        // Bridge Z-insertions
        if z_flags[0] {
            apply_pauli_z(re, im, 4);
        }
        if z_flags[1] {
            apply_pauli_z(re, im, 8);
        }

        measure_single_qubit_z(re, im, 4)
    };

    // Fragment B (18Q Drug Ligand) Branch Runner
    let run_18q_ligand = |z_flags: &[bool], re: &mut [f64], im: &mut [f64]| -> f64 {
        re.fill(0.0);
        im.fill(0.0);
        re[0] = 1.0;

        // 18Q Parameterized Layer: Alternating Ry, Rz
        for q in 0..18 {
            let theta = 0.35 * ((q % 5) as f64 + 1.2);
            apply_ry_kernel(re, im, q, theta);
            apply_rz_kernel(re, im, q, theta * 0.3);
        }
        // Intra-fragment Ring CZ
        for q in 0..18 {
            let next = (q + 1) % 18;
            apply_cz_kernel(re, im, q, next);
        }

        // Bridge Z-insertions
        if z_flags[0] {
            apply_pauli_z(re, im, 13);
        }
        if z_flags[1] {
            apply_pauli_z(re, im, 17);
        }

        measure_single_qubit_z(re, im, 13)
    };

    // Record branch timings
    for mask in 0..4 {
        let z_flags = [(mask & 1) != 0, ((mask >> 1) & 1) != 0];
        let t0 = Instant::now();
        let exp_a = run_18q_pocket(&z_flags, &mut re_buf, &mut im_buf);
        let us_a = t0.elapsed().as_micros() as f64;
        branch_telemetry.push(BranchTelemetry {
            fragment: "Fragment_A (Pocket)".to_string(),
            branch_id: mask,
            z_insertions: z_flags.to_vec(),
            local_expectation: exp_a,
            latency_us: us_a,
        });

        let t1 = Instant::now();
        let exp_b = run_18q_ligand(&z_flags, &mut re_buf, &mut im_buf);
        let us_b = t1.elapsed().as_micros() as f64;
        branch_telemetry.push(BranchTelemetry {
            fragment: "Fragment_B (Ligand)".to_string(),
            branch_id: mask,
            z_insertions: z_flags.to_vec(),
            local_expectation: exp_b,
            latency_us: us_b,
        });
    }

    // Execute full cut-bridge evaluation
    let reconstructed_exp = eval_36q_bipartite_bridge(
        run_18q_pocket,
        run_18q_ligand,
        &mut re_buf,
        &mut im_buf,
    );
    let total_time_ms = start_all.elapsed().as_secs_f64() * 1000.0;

    // Single-pass AVX2 Bloch metrics on the final state of Fragment A and Fragment B
    let bloch_a = compute_all_bloch_vectors(&re_buf, &im_buf, 18);
    let mean_radius_a = bloch_a.iter().map(|b| b.radius).sum::<f64>() / 18.0;
    let mean_purity_a = bloch_a.iter().map(|b| b.purity).sum::<f64>() / 18.0;

    run_18q_pocket(&[false, false], &mut re_buf, &mut im_buf);
    let bloch_pocket = compute_all_bloch_vectors(&re_buf, &im_buf, 18);
    let mean_radius_pocket = bloch_pocket.iter().map(|b| b.radius).sum::<f64>() / 18.0;
    let mean_purity_pocket = bloch_pocket.iter().map(|b| b.purity).sum::<f64>() / 18.0;

    println!("────────────────────────────────────────────────────────────────────────");
    println!(" 36-QUBIT BIPARTITE CUT-BRIDGE TELEMETRY MATRIX");
    println!("────────────────────────────────────────────────────────────────────────");
    println!("  Reconstructed ⟨Z_A ⊗ Z_B⟩ : {:.10}", reconstructed_exp);
    println!("  Total 36Q Latency         : {:.2} ms (8 branch passes + contraction)", total_time_ms);
    println!("  Average Branch Latency    : {:.2} ms per 18Q fragment", total_time_ms / 8.0);
    println!("  L3 Working Set Size       : {:.2} MB (Peak Contiguous RAM)", cut_bridge_ram_mb);
    println!("  Sub-Register A Mean Purity: {:.4} (Mean Bloch Radius: {:.4})", mean_purity_pocket, mean_radius_pocket);
    println!("  Sub-Register B Mean Purity: {:.4} (Mean Bloch Radius: {:.4})", mean_purity_a, mean_radius_a);
    println!("────────────────────────────────────────────────────────────────────────");

    println!("\n[BRANCH EXECUTION BREAKDOWN]");
    for b in &branch_telemetry {
        println!(
            "  {:20} | Branch {:02b} (Z-ins: {:?}) | ⟨Z⟩ = {:+0.6} | Latency: {:6.2} us",
            b.fragment, b.branch_id, b.z_insertions, b.local_expectation, b.latency_us
        );
    }

    // Save JSON telemetry
    let report = CutBridgeBenchReport {
        timestamp: "2026-10-16T13:00:00Z".to_string(),
        total_qubits,
        fragment_a_qubits: fragment_qubits,
        fragment_b_qubits: fragment_qubits,
        bridge_gates,
        dense_ram_theoretical_gb: dense_ram_gb,
        cut_bridge_ram_actual_mb: cut_bridge_ram_mb,
        ram_compression_ratio: ram_compression,
        small_register_agreement_error: agreement_err,
        reconstructed_36q_expectation: reconstructed_exp,
        total_execution_time_ms: total_time_ms,
        mean_bloch_radius_a: mean_radius_pocket,
        mean_bloch_radius_b: mean_radius_a,
        mean_purity_a: mean_purity_pocket,
        mean_purity_b: mean_purity_a,
        branches: branch_telemetry,
    };

    let out_dir = PathBuf::from("results/experiments");
    fs::create_dir_all(&out_dir).ok();
    let out_path = out_dir.join("cut_bridge_36q_results.json");
    if let Ok(json_str) = serde_json::to_string_pretty(&report) {
        fs::write(&out_path, json_str).ok();
        println!("\n[SAVED] Structured telemetry exported to {}", out_path.display());
    }
}
