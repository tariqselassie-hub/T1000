// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Does the ADAPT-VQE trap on stretched N₂ come from the operator pool or from the reference?
//!
//! `vqe_strong_correlation` found ADAPT stuck 82 mHa above FCI on N₂ beyond 2.8 Å with every
//! occupied-to-virtual pool gradient below 1e-4, starting from the lowest-energy RHF solution. This
//! binary reruns the stretched region with three pools (occupied-to-virtual fermionic, generalised
//! fermionic, generalised qubit excitations) and two Hartree-Fock references (the lowest SCF state,
//! and the state followed out from equilibrium), all at the default 1e-3 gradient tolerance.
//!
//! ```text
//! cargo run --release -p t1000 --bin vqe_adapt_pools              # N2 and H6
//! cargo run --release -p t1000 --bin vqe_adapt_pools N2 --record  # one system, saved
//! ```

use serde::Serialize;
use std::time::Instant;
use t1000::results::{Better, Score};
use t1000::vqe::adapt::{adapt_vqe, AdaptConfig};
use t1000::vqe::circuit::circuit_cost;
use t1000::vqe::driver::{electronic_structure, VqeConfig, CHEMICAL_ACCURACY};
use t1000::vqe::fermion::{build_hamiltonian, Sector, SparseMatrix};
use t1000::vqe::lanczos::ground_state;
use t1000::vqe::molecules::{diatomic, hydrogen_chain, Molecule};
use t1000::vqe::uccsd::{generalized_excitations, uccsd_excitations, Excitation};

const REPORT_PATH: &str = "results/data/vqe_adapt_pools_report.json";

#[derive(Debug, Clone, Copy, PartialEq)]
enum Pool {
    OccVirt,
    Generalized,
    QubitGeneralized,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Reference {
    Lowest,
    Followed,
}

struct Variant {
    key: &'static str,
    pool: Pool,
    reference: Reference,
}

const VARIANTS: [Variant; 5] = [
    Variant {
        key: "occ-virt",
        pool: Pool::OccVirt,
        reference: Reference::Lowest,
    },
    Variant {
        key: "GSD",
        pool: Pool::Generalized,
        reference: Reference::Lowest,
    },
    Variant {
        key: "qubit GSD",
        pool: Pool::QubitGeneralized,
        reference: Reference::Lowest,
    },
    Variant {
        key: "occ-virt/fol",
        pool: Pool::OccVirt,
        reference: Reference::Followed,
    },
    Variant {
        key: "GSD/fol",
        pool: Pool::Generalized,
        reference: Reference::Followed,
    },
];

struct System {
    key: &'static str,
    label: &'static str,
    grid: Vec<f64>,
    /// Points before the first reported one, used only to follow the SCF branch out.
    lead_in: Vec<f64>,
    build: fn(f64) -> Molecule,
}

fn systems() -> Vec<System> {
    vec![
        System {
            key: "N2",
            label: "N2 dissociation (16 qubits)",
            grid: vec![1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
            lead_in: vec![1.0, 1.2, 1.4],
            build: |r| diatomic("dinitrogen", "N2", [7, 7], r, 0),
        },
        System {
            key: "H6",
            label: "H6 linear chain (12 qubits)",
            grid: vec![1.4, 1.8, 2.2, 2.6, 3.0],
            lead_in: vec![0.8, 1.0, 1.2],
            build: |d| hydrogen_chain(6, d),
        },
    ]
}

#[derive(Debug, Clone, Serialize)]
struct Run {
    variant: &'static str,
    energy: f64,
    error_mha: f64,
    operators: usize,
    unique_operators: usize,
    cnot: usize,
    /// "naive JW" for fermionic operators, "qubit-excitation circuits" for the qubit pool.
    cnot_model: &'static str,
    final_gradient_norm: f64,
    converged: bool,
    time_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
struct Point {
    coordinate: f64,
    fci: f64,
    hf_lowest: f64,
    hf_followed: f64,
    pool_sizes: [usize; 3],
    runs: Vec<Run>,
}

/// One Hartree-Fock branch: the SCF, its determinant Hamiltonian and FCI.
struct Branch {
    h: SparseMatrix,
    sector: Sector,
    reference: Vec<f64>,
    hf: f64,
    fci: f64,
}

fn branch(
    molecule: &Molecule,
    orbitals: &mut Option<Vec<f64>>,
    seed_only: bool,
) -> Result<Branch, String> {
    let cfg = VqeConfig {
        scf_guess: orbitals.clone(),
        scf_seed_only: seed_only,
        ..VqeConfig::default()
    };
    let p = electronic_structure(molecule, &cfg)?;
    if !p.scf.converged {
        return Err(format!("SCF did not converge ({})", p.scf.strategy));
    }
    *orbitals = Some(p.scf.coefficients.clone());
    let half = p.ham.n_electrons / 2;
    let sector = Sector::new(p.ham.n_orb, half, half);
    let h = build_hamiltonian(&p.ham, &sector);
    let mut reference = vec![0.0; sector.dim()];
    reference[sector.hartree_fock()] = 1.0;
    let (fci, _) = ground_state(&h, &reference, 400, 1e-12)?;
    let hf = h.get(sector.hartree_fock(), sector.hartree_fock());
    Ok(Branch {
        h,
        sector,
        reference,
        hf,
        fci,
    })
}

/// CNOTs of the qubit-excitation circuits of Yordanov et al. (PRA 102, 062612, 2020): 2 per
/// single and 13 per double excitation, with no Jordan-Wigner parity ladders.
fn qubit_excitation_cnots(ansatz: &[Excitation]) -> usize {
    ansatz
        .iter()
        .map(|e| if e.is_double() { 13 } else { 2 })
        .sum()
}

fn run_variant(v: &Variant, b: &Branch, pools: &[Vec<Excitation>; 3], cfg: &AdaptConfig) -> Run {
    let pool = match v.pool {
        Pool::OccVirt => &pools[0],
        Pool::Generalized => &pools[1],
        Pool::QubitGeneralized => &pools[2],
    };
    let t = Instant::now();
    let r = adapt_vqe(&b.h, pool, &b.reference, cfg);
    let ansatz: Vec<Excitation> = r.selected.iter().map(|&k| pool[k].clone()).collect();
    let mut unique = r.selected.clone();
    unique.sort_unstable();
    unique.dedup();
    let (cnot, cnot_model) = if v.pool == Pool::QubitGeneralized {
        (qubit_excitation_cnots(&ansatz), "qubit-excitation circuits")
    } else {
        (circuit_cost(&ansatz).cnot, "naive JW")
    };
    Run {
        variant: v.key,
        energy: r.energy,
        error_mha: (r.energy - b.fci) * 1e3,
        operators: ansatz.len(),
        unique_operators: unique.len(),
        cnot,
        cnot_model,
        final_gradient_norm: r.final_gradient_norm,
        converged: r.converged,
        time_ms: t.elapsed().as_secs_f64() * 1e3,
    }
}

fn scan(sys: &System, cfg: &AdaptConfig) -> Vec<Point> {
    let (mut low_orb, mut fol_orb) = (None, None);
    for &x in &sys.lead_in {
        let m = (sys.build)(x);
        // Both branches start together near equilibrium, where the SCF solution is unique.
        if let Err(e) = branch(&m, &mut low_orb, false).and(branch(&m, &mut fol_orb, true)) {
            println!("  lead-in {x:.2}: {e}");
        }
    }
    let mut points = Vec::new();
    for &x in &sys.grid {
        let m = (sys.build)(x);
        let (low, fol) = match (
            branch(&m, &mut low_orb, false),
            branch(&m, &mut fol_orb, true),
        ) {
            (Ok(l), Ok(f)) => (l, f),
            (Err(e), _) | (_, Err(e)) => {
                println!("  {x:.2}: {e}");
                continue;
            }
        };
        let occ = uccsd_excitations(&low.sector);
        let gen = generalized_excitations(&low.sector);
        let qgen: Vec<Excitation> = gen.iter().map(Excitation::to_qubit).collect();
        let pools = [occ, gen, qgen];
        let runs: Vec<Run> = VARIANTS
            .iter()
            .map(|v| {
                run_variant(
                    v,
                    if v.reference == Reference::Lowest {
                        &low
                    } else {
                        &fol
                    },
                    &pools,
                    cfg,
                )
            })
            .collect();
        points.push(Point {
            coordinate: x,
            fci: low.fci,
            hf_lowest: low.hf,
            hf_followed: fol.hf,
            pool_sizes: [pools[0].len(), pools[1].len(), pools[2].len()],
            runs,
        });
        print_point(points.last().expect("just pushed"));
    }
    points
}

fn print_header() {
    print!(" {:>5} {:>9} |", "coord", "dHF mHa");
    for v in &VARIANTS {
        print!(" {:>22} |", v.key);
    }
    println!();
}

fn print_point(p: &Point) {
    print!(
        " {:>5.2} {:>9.2} |",
        p.coordinate,
        (p.hf_followed - p.hf_lowest) * 1e3
    );
    for r in &p.runs {
        let mark = if r.error_mha.abs() < CHEMICAL_ACCURACY * 1e3 {
            ' '
        } else {
            '*'
        };
        print!(
            " {:>8.2}{mark} {:>4} {:>6} |",
            r.error_mha, r.operators, r.cnot
        );
    }
    println!();
}

fn main() {
    let filter: Option<String> = std::env::args().skip(1).find(|a| !a.starts_with("--"));
    let cfg = AdaptConfig::default();
    println!("==============================================================================================================");
    println!(" T-1000 ADAPT-VQE POOLS AND REFERENCES ON STRETCHED BONDS (STO-3G, frozen core, ADAPT tol 1e-3)");
    println!("==============================================================================================================");
    println!(" per variant: error vs FCI (mHa, * outside chemical accuracy), operators, CNOTs");
    println!(
        " dHF = E_HF(followed branch) - E_HF(lowest branch); /fol = ADAPT from the followed branch"
    );
    let mut report = serde_json::Map::new();
    let mut worst_gsd = 0.0f64;
    let mut keys = Vec::new();
    for sys in systems() {
        if filter
            .as_ref()
            .is_some_and(|f| f.split(',').all(|k| k != sys.key))
        {
            continue;
        }
        keys.push(sys.key);
        println!("\n {}", sys.label);
        print_header();
        let points = scan(&sys, &cfg);
        for p in &points {
            worst_gsd = worst_gsd.max(p.runs[1].error_mha.abs());
        }
        report.insert(
            sys.key.into(),
            serde_json::json!({ "label": sys.label, "points": points }),
        );
    }
    report.insert(
        "settings".into(),
        serde_json::json!({
            "basis": "STO-3G",
            "frozen_core": true,
            "adapt_gradient_norm_tolerance": cfg.gradient_norm_tolerance,
            "adapt_max_operators": cfg.max_operators,
            "pools": {
                "occ-virt": "fermionic singles and doubles from occupied to virtual orbitals",
                "GSD": "fermionic generalised singles and doubles on distinct spin orbitals",
                "qubit GSD": "the GSD excitations as qubit excitations (no parity strings)"
            },
            "references": {
                "lowest": "lowest converged RHF of a seeded and a cold start (as in vqe_strong_correlation)",
                "followed": "RHF reached from the previous point's orbitals only"
            },
            "cnot_models": {
                "naive JW": "2(w-1) CNOTs per weight-w Pauli rotation of the fermionic excitation",
                "qubit-excitation circuits": "2 per single, 13 per double (Yordanov et al. 2020)"
            }
        }),
    );
    let settings = format!(
        "systems={} tol={}",
        keys.join(","),
        cfg.gradient_norm_tolerance
    );
    let score = Score {
        metric: "gsd_max_abs_error_mha",
        value: worst_gsd,
        better: Better::Lower,
        guard: None,
        settings: &settings,
    };
    let json = serde_json::to_string_pretty(&report).expect("serialise report");
    match t1000::results::write_scored(REPORT_PATH, json, &score) {
        Ok(outcome) => println!("\n [OK] {REPORT_PATH}: {outcome:?}"),
        Err(e) => eprintln!("\n [ERROR] {REPORT_PATH}: {e}"),
    }
}
