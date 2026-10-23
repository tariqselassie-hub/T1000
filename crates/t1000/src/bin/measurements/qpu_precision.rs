// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Fixed-point precision study for the 1000_QPU engine.
//!
//! Each converged UCCSD circuit is compiled to the engine's gate set and run through the
//! bit-exact model (`t1000::qpu`) at several state scales. The energy of the resulting state is
//! compared with the same circuit in double precision, so the error isolates the integer
//! arithmetic (25-bit coefficients, exact accumulate, 32-bit saturating components).

use serde::Serialize;
use std::fs;
use std::path::Path;
use t1000::qpu::compile::compile_ansatz;
use t1000::qpu::fixed::{dequantize_state, quantize_state};
use t1000::qpu::model::{QpuModel, MAX_QUBITS};
use t1000::vqe::circuit::prepare_qubit_state;
use t1000::vqe::driver::{solve_detailed, VqeConfig, CHEMICAL_ACCURACY};
use t1000::vqe::molecules::catalog;
use t1000::vqe::pauli::QubitHamiltonian;

const CLOCK_HZ: f64 = 100e6;
const FRAC_BITS: [u32; 5] = [16, 20, 24, 28, 30];

#[derive(Serialize)]
struct Row {
    formula: String,
    qubits: usize,
    gates: usize,
    cnot: usize,
    fraction_bits: u32,
    energy_error_mha: f64,
    norm_drift: f64,
    max_imag: f64,
    saturated_amplitudes: usize,
    fabric_seconds_per_circuit: f64,
}

#[derive(Serialize)]
struct Skipped {
    formula: String,
    qubits: usize,
}

fn normalised_energy(h: &QubitHamiltonian, re: &[f64], im: &[f64]) -> (f64, f64, f64) {
    let norm: f64 = re.iter().zip(im).map(|(a, b)| a * a + b * b).sum();
    let max_imag = im.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    (h.expectation(re) / norm, norm, max_imag)
}

fn main() {
    println!("=========================================================================");
    println!(" 1000_QPU FIXED-POINT PRECISION STUDY (25-bit coefficients, 32-bit state)");
    println!("=========================================================================");
    let cfg = VqeConfig::default();
    let mut rows = Vec::new();
    let mut skipped = Vec::new();
    for molecule in catalog() {
        let s = solve_detailed(&molecule, &cfg).expect("solve");
        let n = s.report.n_qubits;
        if n > MAX_QUBITS {
            skipped.push(Skipped { formula: s.report.formula.clone(), qubits: n });
            continue;
        }
        let hf = s.sector.dets[s.sector.hartree_fock()];
        let gates = compile_ansatz(&s.excitations, &s.theta);
        let cnot = gates.iter().filter(|g| g.ctrl != 0).count();
        let (fre, fim) = prepare_qubit_state(n, hf, &s.excitations, &s.theta);
        let (e_float, _, _) = normalised_energy(&s.qubit_hamiltonian, &fre, &fim);
        let seconds = gates.len() as f64 * (1u64 << (n - 1)) as f64 / CLOCK_HZ;
        for &frac in &FRAC_BITS {
            let mut q = QpuModel::new();
            q.set_qubits(n);
            q.reset(0);
            q.load(hf as usize, &[(quantize_state(1.0, frac), 0)]);
            for g in &gates {
                assert!(q.apply_gate(g));
            }
            let amps = q.read(0, 1 << n).expect("read");
            let re: Vec<f64> = amps.iter().map(|a| dequantize_state(a.0, frac)).collect();
            let im: Vec<f64> = amps.iter().map(|a| dequantize_state(a.1, frac)).collect();
            let saturated = amps
                .iter()
                .filter(|a| a.0.abs() >= i64::from(i32::MAX) || a.1.abs() >= i64::from(i32::MAX))
                .count();
            let (e, norm, max_imag) = normalised_energy(&s.qubit_hamiltonian, &re, &im);
            rows.push(Row {
                formula: s.report.formula.clone(),
                qubits: n,
                gates: gates.len(),
                cnot,
                fraction_bits: frac,
                energy_error_mha: (e - e_float) * 1e3,
                norm_drift: norm - 1.0,
                max_imag,
                saturated_amplitudes: saturated,
                fabric_seconds_per_circuit: seconds,
            });
        }
    }

    println!(
        "\n chemical accuracy = {:.2} mHa; energy error is fixed point minus double precision, same circuit",
        CHEMICAL_ACCURACY * 1e3
    );
    println!(
        " {:<5} {:>3} {:>7} {:>6} {:>5} {:>13} {:>11} {:>10} {:>5} {:>10}",
        "mol", "q", "gates", "CNOT", "frac", "dE (mHa)", "norm-1", "max |im|", "sat", "fabric s"
    );
    for r in &rows {
        println!(
            " {:<5} {:>3} {:>7} {:>6} {:>5} {:>13.6} {:>11.2e} {:>10.2e} {:>5} {:>10.3}",
            r.formula,
            r.qubits,
            r.gates,
            r.cnot,
            r.fraction_bits,
            r.energy_error_mha,
            r.norm_drift,
            r.max_imag,
            r.saturated_amplitudes,
            r.fabric_seconds_per_circuit
        );
    }
    for k in &skipped {
        println!(" {:<5} {:>3}  exceeds the engine's {} qubits: not run", k.formula, k.qubits, MAX_QUBITS);
    }

    let dir = Path::new("results/data");
    let _ = fs::create_dir_all(dir);
    let path = dir.join("qpu_precision_report.json");
    let json = serde_json::json!({ "rows": rows, "skipped": skipped });
    t1000::results::write(&path, serde_json::to_string_pretty(&json).expect("serialise"))
        .expect("write report");
    println!("\n [OK] report: {}", path.display());
}
