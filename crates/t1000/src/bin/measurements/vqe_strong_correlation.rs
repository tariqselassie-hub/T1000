// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Strong-correlation benchmark: where do UCCSD-VQE, ADAPT-VQE and the classical baselines (MP2,
//! CISD, CCSD) lose chemical accuracy as bonds stretch?
//!
//! Systems (STO-3G, frozen core, exact simulation): the linear H₄ and H₆ chains stretched
//! uniformly, the H₄ rectangle passing through the square, and N₂ dissociation. Every method runs
//! in the same active space and is judged against exact FCI from the same Hamiltonian.
//!
//! ```text
//! cargo run --release -p t1000 --bin vqe_strong_correlation             # all systems
//! cargo run --release -p t1000 --bin vqe_strong_correlation H4,N2       # a selection
//! cargo run --release -p t1000 --bin vqe_strong_correlation -- --record # save the report
//! ```

use serde::Serialize;
use std::time::Instant;
use t1000::results::{Better, Score};
use t1000::vqe::adapt::{adapt_vqe, AdaptConfig};
use t1000::vqe::circuit::circuit_cost;
use t1000::vqe::classical::{ccsd, cisd_energy, mp2_correlation};
use t1000::vqe::driver::{electronic_structure, VqeConfig, CHEMICAL_ACCURACY};
use t1000::vqe::fermion::{build_hamiltonian, Sector};
use t1000::vqe::lanczos::ground_state;
use t1000::vqe::molecules::{diatomic, hydrogen_chain, hydrogen_rectangle, Molecule};
use t1000::vqe::optimize::bfgs;
use t1000::vqe::uccsd::{energy_and_gradient, uccsd_excitations, Excitation};

const REPORT_PATH: &str = "results/data/vqe_strong_correlation_report.json";

struct System {
    key: &'static str,
    label: &'static str,
    coordinate: &'static str,
    grid: Vec<f64>,
    build: fn(f64) -> Molecule,
}

fn grid(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| lo + (hi - lo) * k as f64 / (n - 1) as f64)
        .collect()
}

fn systems() -> Vec<System> {
    vec![
        System {
            key: "H4",
            label: "H4 linear chain",
            coordinate: "H-H spacing (Å)",
            grid: grid(0.6, 3.0, 13),
            build: |d| hydrogen_chain(4, d),
        },
        System {
            key: "H4rect",
            label: "H4 rectangle, a = 1.0 Å",
            coordinate: "side b (Å)",
            grid: grid(0.7, 1.3, 13),
            build: |b| hydrogen_rectangle(1.0, b),
        },
        System {
            key: "H6",
            label: "H6 linear chain",
            coordinate: "H-H spacing (Å)",
            grid: grid(0.6, 3.0, 13),
            build: |d| hydrogen_chain(6, d),
        },
        System {
            key: "N2",
            label: "N2 dissociation",
            coordinate: "N-N distance (Å)",
            grid: grid(0.8, 3.0, 12),
            build: |r| diatomic("dinitrogen", "N2", [7, 7], r, 0),
        },
    ]
}

#[derive(Debug, Clone, Serialize)]
struct Method {
    energy: f64,
    error_mha: f64,
    converged: bool,
    parameters: usize,
    cnot: usize,
    time_ms: f64,
    /// A variational energy is an upper bound even when its optimiser stopped early; a
    /// projective one (CCSD) means nothing unless its equations converged.
    variational: bool,
}

impl Method {
    /// Whether the energy can be compared with FCI at all.
    fn usable(&self) -> bool {
        self.converged || self.variational
    }
}

#[derive(Debug, Clone, Serialize)]
struct Point {
    coordinate: f64,
    qubits: usize,
    determinants: usize,
    scf_strategy: String,
    hf: Method,
    mp2: Method,
    cisd: Method,
    ccsd: Method,
    ccsd_max_amplitude: f64,
    uccsd: Method,
    adapt: Method,
    adapt_unique_operators: usize,
    fci: f64,
}

/// State carried along a scan: SCF orbitals and UCCSD angles of the previous point.
#[derive(Default)]
struct Carry {
    orbitals: Option<Vec<f64>>,
    uccsd_theta: Option<Vec<f64>>,
}

fn method(
    energy: f64,
    fci: f64,
    converged: bool,
    parameters: usize,
    cnot: usize,
    t: Instant,
) -> Method {
    Method {
        energy,
        error_mha: (energy - fci) * 1e3,
        converged,
        parameters,
        cnot,
        time_ms: t.elapsed().as_secs_f64() * 1e3,
        variational: true,
    }
}

/// UCCSD from zero and, when available, from the previous point's angles; the lower energy wins
/// (both are variational, so the lower one is the better optimum of the same ansatz).
fn run_uccsd(
    h: &t1000::vqe::fermion::SparseMatrix,
    pool: &[Excitation],
    reference: &[f64],
    warm: Option<&Vec<f64>>,
) -> (f64, bool, Vec<f64>) {
    let starts: Vec<Vec<f64>> = std::iter::once(vec![0.0; pool.len()])
        .chain(warm.filter(|w| w.len() == pool.len()).cloned())
        .collect();
    starts
        .into_iter()
        .map(|x0| {
            bfgs(
                |t| energy_and_gradient(h, pool, reference, t),
                x0,
                800,
                1e-7,
            )
        })
        .map(|r| (r.value, r.converged, r.x))
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .expect("at least one start")
}

fn solve_point(
    molecule: &Molecule,
    coordinate: f64,
    carry: &mut Carry,
    adapt_cfg: &AdaptConfig,
) -> Result<Point, String> {
    let cfg = VqeConfig {
        scf_guess: carry.orbitals.clone(),
        ..VqeConfig::default()
    };
    let t = Instant::now();
    let p = electronic_structure(molecule, &cfg)?;
    if !p.scf.converged {
        return Err(format!("SCF did not converge ({})", p.scf.strategy));
    }
    carry.orbitals = Some(p.scf.coefficients.clone());
    let half = p.ham.n_electrons / 2;
    let sector = Sector::new(p.ham.n_orb, half, half);
    let h = build_hamiltonian(&p.ham, &sector);
    let mut reference = vec![0.0; sector.dim()];
    reference[sector.hartree_fock()] = 1.0;
    let (fci, _) = ground_state(&h, &reference, 400, 1e-12)?;
    let e_hf = h.get(sector.hartree_fock(), sector.hartree_fock());
    let hf = method(e_hf, fci, true, 0, 0, t);

    let t = Instant::now();
    let mp2 = method(e_hf + mp2_correlation(&p.ham), fci, true, 0, 0, t);
    let t = Instant::now();
    let cisd = method(cisd_energy(&p.ham, &sector)?, fci, true, 0, 0, t);

    let pool = uccsd_excitations(&sector);
    let t = Instant::now();
    let cc = ccsd(&h, &sector, &pool);
    let ccsd_m = Method {
        variational: false,
        ..method(cc.energy, fci, cc.converged, pool.len(), 0, t)
    };

    let t = Instant::now();
    let (e_u, conv_u, theta_u) = run_uccsd(&h, &pool, &reference, carry.uccsd_theta.as_ref());
    carry.uccsd_theta = Some(theta_u);
    let uccsd = method(e_u, fci, conv_u, pool.len(), circuit_cost(&pool).cnot, t);

    let t = Instant::now();
    let ad = adapt_vqe(&h, &pool, &reference, adapt_cfg);
    let ansatz: Vec<Excitation> = ad.selected.iter().map(|&k| pool[k].clone()).collect();
    let mut unique = ad.selected.clone();
    unique.sort_unstable();
    unique.dedup();
    let adapt = method(
        ad.energy,
        fci,
        ad.converged,
        ansatz.len(),
        circuit_cost(&ansatz).cnot,
        t,
    );

    Ok(Point {
        coordinate,
        qubits: 2 * p.ham.n_orb,
        determinants: sector.dim(),
        scf_strategy: format!("{} ({})", p.scf.strategy, p.scf.iterations),
        hf,
        mp2,
        cisd,
        ccsd: ccsd_m,
        ccsd_max_amplitude: cc.max_amplitude,
        uccsd,
        adapt,
        adapt_unique_operators: unique.len(),
        fci,
    })
}

fn flag(m: &Method) -> &'static str {
    if !m.converged {
        "!"
    } else if m.error_mha.abs() < CHEMICAL_ACCURACY * 1e3 {
        " "
    } else {
        "*"
    }
}

fn print_table(sys: &System, points: &[Point]) {
    println!(
        "\n {}  [{} qubits, {} determinants]",
        sys.label, points[0].qubits, points[0].determinants
    );
    println!(
        " {:>6} {:>12} | error vs FCI (mHa): {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} | {:>6} {:>6} | {:>6} {:>6}",
        "coord", "E(FCI)", "HF", "MP2", "CISD", "CCSD", "UCCSD", "ADAPT", "n_UCC", "n_ADA", "CX_UCC", "CX_ADA"
    );
    for p in points {
        let e = |m: &Method| {
            if m.usable() {
                format!("{:>7.2}{}", m.error_mha, flag(m))
            } else {
                format!("{:>8}", "n/c")
            }
        };
        println!(
            " {:>6.3} {:>12.6} |                     {} {} {} {} {} {} | {:>6} {:>6} | {:>6} {:>6}",
            p.coordinate,
            p.fci,
            e(&p.hf),
            e(&p.mp2),
            e(&p.cisd),
            e(&p.ccsd),
            e(&p.uccsd),
            e(&p.adapt),
            p.uccsd.parameters,
            p.adapt.parameters,
            p.uccsd.cnot,
            p.adapt.cnot
        );
    }
    println!(
        " (* outside chemical accuracy 1.594 mHa; ! optimiser stopped early, energy still an upper bound)"
    );
    println!(" (n/c CCSD did not converge; n = parameters, CX = CNOTs)");
}

#[derive(Debug, Serialize)]
struct Summary {
    method: &'static str,
    max_abs_error_mha: f64,
    points_within_chemical_accuracy: usize,
    points: usize,
    not_converged: usize,
}

/// A method name and how to read its result from a point.
type MethodPick = (&'static str, fn(&Point) -> &Method);

fn summarise(points: &[Point]) -> Vec<Summary> {
    let pick: [MethodPick; 6] = [
        ("HF", |p| &p.hf),
        ("MP2", |p| &p.mp2),
        ("CISD", |p| &p.cisd),
        ("CCSD", |p| &p.ccsd),
        ("UCCSD", |p| &p.uccsd),
        ("ADAPT", |p| &p.adapt),
    ];
    pick.iter()
        .map(|&(name, get)| Summary {
            method: name,
            max_abs_error_mha: points
                .iter()
                .filter(|p| get(p).usable())
                .map(|p| get(p).error_mha.abs())
                .fold(0.0, f64::max),
            points_within_chemical_accuracy: points
                .iter()
                .filter(|p| get(p).usable() && get(p).error_mha.abs() < CHEMICAL_ACCURACY * 1e3)
                .count(),
            points: points.len(),
            not_converged: points.iter().filter(|p| !get(p).converged).count(),
        })
        .collect()
}

fn main() {
    let filter: Option<Vec<String>> = std::env::args()
        .skip(1)
        .scan(false, |skip_next, a| {
            let skip = *skip_next || a.starts_with("--");
            *skip_next = a == "--adapt-tol";
            Some((skip, a))
        })
        .find(|(skip, _)| !skip)
        .map(|(_, a)| a)
        .map(|a| a.split(',').map(str::to_string).collect());
    println!(
        "========================================================================================="
    );
    println!(" T-1000 STRONG CORRELATION: MP2 / CISD / CCSD / UCCSD-VQE / ADAPT-VQE vs exact FCI (STO-3G)");
    println!(
        "========================================================================================="
    );
    let args: Vec<String> = std::env::args().collect();
    let adapt_cfg = AdaptConfig {
        gradient_norm_tolerance: args
            .iter()
            .position(|a| a == "--adapt-tol")
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(AdaptConfig::default().gradient_norm_tolerance),
        ..AdaptConfig::default()
    };
    let mut report = serde_json::Map::new();
    let mut worst_adapt = 0.0f64;
    let mut keys = Vec::new();
    for sys in systems() {
        if filter
            .as_ref()
            .is_some_and(|f| !f.iter().any(|k| k == sys.key))
        {
            continue;
        }
        keys.push(sys.key);
        let mut carry = Carry::default();
        let mut points = Vec::new();
        for &x in &sys.grid {
            match solve_point(&(sys.build)(x), x, &mut carry, &adapt_cfg) {
                Ok(p) => points.push(p),
                Err(e) => println!("  {} = {x:.3}: {e}", sys.coordinate),
            }
        }
        if points.is_empty() {
            continue;
        }
        print_table(&sys, &points);
        let summary = summarise(&points);
        for s in &summary {
            println!(
                "   {:<6} max |error| {:>9.3} mHa, within chemical accuracy at {:>2}/{} points{}",
                s.method,
                s.max_abs_error_mha,
                s.points_within_chemical_accuracy,
                s.points,
                if s.not_converged > 0 {
                    format!(", {} not converged", s.not_converged)
                } else {
                    String::new()
                }
            );
        }
        worst_adapt = worst_adapt.max(summary[5].max_abs_error_mha);
        report.insert(
            sys.key.to_string(),
            serde_json::json!({
                "label": sys.label,
                "coordinate": sys.coordinate,
                "summary": summary,
                "points": points,
            }),
        );
    }
    report.insert(
        "settings".into(),
        serde_json::json!({
            "basis": "STO-3G",
            "frozen_core": true,
            "adapt_pool": "UCCSD singles and doubles (occupied to virtual), operators may repeat",
            "adapt_gradient_norm_tolerance": adapt_cfg.gradient_norm_tolerance,
            "adapt_max_operators": adapt_cfg.max_operators,
            "uccsd_starts": "zero and the previous point's angles; lower energy kept",
            "chemical_accuracy_mha": CHEMICAL_ACCURACY * 1e3,
        }),
    );
    let settings = format!(
        "systems={} adapt_tol={}",
        keys.join(","),
        adapt_cfg.gradient_norm_tolerance
    );
    let score = Score {
        metric: "adapt_max_abs_error_mha",
        value: worst_adapt,
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
