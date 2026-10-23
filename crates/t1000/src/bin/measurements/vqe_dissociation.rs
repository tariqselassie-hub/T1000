// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Potential-energy curves for H₂ and LiH: Hartree-Fock, exact FCI and UCCSD-VQE along the bond.
//!
//! Stretching a bond is where single-determinant Hartree-Fock fails (it dissociates H₂ to the
//! wrong energy) and correlation becomes essential, so it is the standard test of a VQE.

use std::fs;
use std::path::Path;
use t1000::vqe::driver::{solve, VqeConfig, VqeReport};
use t1000::vqe::molecules::diatomic;

struct Curve {
    label: &'static str,
    formula: &'static str,
    z: [u32; 2],
    bond_lengths: Vec<f64>,
}

fn curves() -> Vec<Curve> {
    let range = |lo: f64, hi: f64, n: usize| -> Vec<f64> {
        (0..n).map(|k| lo + (hi - lo) * k as f64 / (n - 1) as f64).collect()
    };
    vec![
        Curve { label: "H2", formula: "H2", z: [1, 1], bond_lengths: range(0.4, 3.0, 14) },
        Curve { label: "LiH", formula: "LiH", z: [3, 1], bond_lengths: range(1.0, 4.0, 13) },
    ]
}

fn scan(curve: &Curve) -> Vec<(f64, VqeReport)> {
    let cfg = VqeConfig { max_iterations: 600, ..VqeConfig::default() };
    curve
        .bond_lengths
        .iter()
        .filter_map(|&r| {
            let molecule = diatomic(curve.label, curve.formula, curve.z, r, 0);
            match solve(&molecule, &cfg) {
                Ok(report) => Some((r, report)),
                Err(e) => {
                    println!("  r = {r:.3} Å: {e}");
                    None
                }
            }
        })
        .collect()
}

fn print_curve(label: &str, points: &[(f64, VqeReport)]) {
    println!("\n {label}");
    println!(
        " {:>7} {:>13} {:>13} {:>13} {:>10} {:>10} {:>6}",
        "R (Å)", "E(HF)", "E(FCI)", "E(VQE)", "HF err mHa", "VQE err mHa", "conv"
    );
    for (r, p) in points {
        println!(
            " {:>7.3} {:>13.6} {:>13.6} {:>13.6} {:>10.2} {:>11.4} {:>6}",
            r,
            p.hf_energy,
            p.fci_energy,
            p.vqe_energy,
            (p.hf_energy - p.fci_energy) * 1e3,
            p.vqe_error_vs_fci * 1e3,
            p.converged
        );
    }
}

fn main() {
    println!("=========================================================================");
    println!(" T-1000 BOND DISSOCIATION: HF vs UCCSD-VQE vs exact FCI (STO-3G)");
    println!("=========================================================================");
    let mut out = serde_json::Map::new();
    for curve in curves() {
        let points = scan(&curve);
        print_curve(curve.label, &points);
        let worst = points.iter().map(|(_, p)| p.vqe_error_vs_fci.abs()).fold(0.0, f64::max);
        println!(" worst VQE error along the curve: {:.4} mHa", worst * 1e3);
        let rows: Vec<_> = points
            .iter()
            .map(|(r, p)| serde_json::json!({ "bond_length_angstrom": r, "report": p }))
            .collect();
        out.insert(curve.label.to_string(), serde_json::Value::Array(rows));
    }
    let dir = Path::new("results/data");
    let _ = fs::create_dir_all(dir);
    let path = dir.join("vqe_dissociation_report.json");
    let json = serde_json::to_string_pretty(&out).expect("serialise report");
    t1000::results::write(&path, json).expect("write report");
    println!("\n [OK] report: {}", path.display());
}
