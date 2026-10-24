// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Checks the analytic noise model of `vqe::noise` against Monte-Carlo gate-level Pauli noise.
//!
//! The analytic model assumes every error anywhere in the circuit destroys every non-identity term:
//! `E_noisy = c0 + F (E - c0)` with `F = (1-p1)^{n1} (1-p2)^{n2}`. Here the compiled UCCSD circuit
//! (basis changes, CNOT ladders, Rz; `qpu::compile`) is simulated as a state vector with a random
//! Pauli inserted after each gate with probability `p1` (one-qubit gates, X/Y/Z equally likely) or
//! `p2` (CNOT, one of the 15 two-qubit Paulis), `p1 = p2 / 10`. The exact energy of each trajectory is
//! averaged to give the energy of the noisy device. The analytic model is evaluated at the same gate
//! counts as the simulated circuit.
//!
//! Errors are rare at the rates of interest, so plain sampling would mostly return the error-free
//! energy. Instead every trajectory is conditioned on at least one error: the gate of the first error
//! is drawn in proportion to its probability of being first, later gates fail independently, and
//! `bias = P(any error) * (mean energy of conditioned trajectories - error-free energy)`.

use rayon::prelude::*;
use serde::Serialize;
use std::fs;
use std::path::Path;
use t1000::qpu::compile::compile_ansatz;
use t1000::qpu::fixed::dequantize_coef;
use t1000::qpu::model::Gate;
use t1000::vqe::driver::{solve_detailed, VqeConfig, CHEMICAL_ACCURACY};
use t1000::vqe::molecules::by_formula;
use t1000::vqe::noise::TermStatistics;
use t1000::vqe::pauli::PauliTerm;
use t1000::vqe::uccsd::prepare_state;

type C = (f64, f64);

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        let mut r = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        for _ in 0..4 {
            r.next();
        }
        r
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

struct Op {
    target: usize,
    ctrl: usize,
    m: [C; 4],
}

fn to_op(g: &Gate) -> Op {
    let c = |k: usize| (dequantize_coef(g.m[2 * k]), dequantize_coef(g.m[2 * k + 1]));
    Op { target: g.target as usize, ctrl: g.ctrl as usize, m: [c(0), c(1), c(2), c(3)] }
}

fn apply_gate(state: &mut [C], op: &Op) {
    let bit = 1usize << op.target;
    let mul = |m: C, v: C| (m.0 * v.0 - m.1 * v.1, m.0 * v.1 + m.1 * v.0);
    for i0 in 0..state.len() {
        if i0 & bit != 0 || i0 & op.ctrl != op.ctrl {
            continue;
        }
        let i1 = i0 | bit;
        let (a, b) = (state[i0], state[i1]);
        let (p, q, r, s) = (mul(op.m[0], a), mul(op.m[1], b), mul(op.m[2], a), mul(op.m[3], b));
        state[i0] = (p.0 + q.0, p.1 + q.1);
        state[i1] = (r.0 + s.0, r.1 + s.1);
    }
}

/// Applies X (1), Y (2) or Z (3) to qubit `q`; 0 is the identity.
fn apply_pauli(state: &mut [C], q: usize, kind: u64) {
    let bit = 1usize << q;
    match kind {
        1 | 2 => {
            for i0 in 0..state.len() {
                if i0 & bit == 0 {
                    let i1 = i0 | bit;
                    let (a, b) = (state[i0], state[i1]);
                    if kind == 1 {
                        state[i0] = b;
                        state[i1] = a;
                    } else {
                        // Y|0> = i|1>, Y|1> = -i|0>
                        state[i0] = (b.1, -b.0);
                        state[i1] = (-a.1, a.0);
                    }
                }
            }
        }
        3 => {
            for (i, v) in state.iter_mut().enumerate() {
                if i & bit != 0 {
                    *v = (-v.0, -v.1);
                }
            }
        }
        _ => {}
    }
}

/// `<psi|P|psi>` for `P = i^{|x∧z|} X^x Z^z` on a complex state.
fn pauli_expectation(psi: &[C], t: &PauliTerm) -> f64 {
    let (pr, pi) = match (t.x & t.z).count_ones() % 4 {
        0 => (1.0, 0.0),
        1 => (0.0, 1.0),
        2 => (-1.0, 0.0),
        _ => (0.0, -1.0),
    };
    let mut acc = 0.0;
    for j in 0..psi.len() {
        let src = j ^ t.x as usize;
        let sign = if ((src as u32) & t.z).count_ones() % 2 == 0 { 1.0 } else { -1.0 };
        let v = psi[src];
        let (wr, wi) = (sign * (pr * v.0 - pi * v.1), sign * (pr * v.1 + pi * v.0));
        acc += psi[j].0 * wr + psi[j].1 * wi;
    }
    acc
}

/// Energy of the normalised state: the 25-bit gate coefficients are not exactly unitary, so the norm
/// of the simulated state drifts from 1 over thousands of gates.
fn energy(psi: &[C], constant: f64, terms: &[PauliTerm]) -> f64 {
    let norm: f64 = psi.iter().map(|v| v.0 * v.0 + v.1 * v.1).sum();
    constant + terms.iter().map(|t| t.coeff * pauli_expectation(psi, t)).sum::<f64>() / norm
}

fn random_error(state: &mut [C], op: &Op, rng: &mut Rng) {
    if op.ctrl == 0 {
        apply_pauli(state, op.target, 1 + rng.below(3));
    } else {
        let r = 1 + rng.below(15);
        apply_pauli(state, op.target, r / 4);
        apply_pauli(state, op.ctrl.trailing_zeros() as usize, r % 4);
    }
}

/// One trajectory with the first error forced at gate `first`; every later gate fails independently
/// with its own probability `p[i]`. With `first = None` the circuit runs without noise.
fn trajectory(
    init: &[C],
    ops: &[Op],
    p: &[f64],
    first: Option<usize>,
    seed: u64,
    constant: f64,
    terms: &[PauliTerm],
) -> f64 {
    let mut rng = Rng::new(seed);
    let mut state = init.to_vec();
    for (i, op) in ops.iter().enumerate() {
        apply_gate(&mut state, op);
        if let Some(f) = first {
            if i == f || (i > f && rng.unit() < p[i]) {
                random_error(&mut state, op, &mut rng);
            }
        }
    }
    energy(&state, constant, terms)
}

#[derive(Serialize)]
struct Row {
    formula: String,
    p2: f64,
    p1: f64,
    one_qubit_gates: usize,
    cnots: usize,
    fidelity_analytic: f64,
    probability_of_any_error: f64,
    bias_analytic_mha: f64,
    bias_monte_carlo_mha: f64,
    bias_monte_carlo_stderr_mha: f64,
    trajectories: usize,
    ratio_mc_over_analytic: f64,
}

/// p2 at which |bias| reaches `target`, by log-log interpolation between bracketing grid points.
fn crossing(points: &[(f64, f64)], target: f64) -> Option<f64> {
    points.windows(2).find_map(|w| {
        let ((x0, y0), (x1, y1)) = (w[0], w[1]);
        (y0 < target && y1 >= target).then(|| {
            let t = (target.ln() - y0.ln()) / (y1.ln() - y0.ln());
            (x0.ln() + t * (x1.ln() - x0.ln())).exp()
        })
    })
}

fn main() {
    let cases: [(&str, &[f64], usize); 4] = [
        ("H2", &[1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 1e-3, 1e-2], 6000),
        ("HeH+", &[1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 1e-3, 1e-2], 6000),
        ("LiH", &[1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 1e-4, 1e-3], 3000),
        ("H2O", &[1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 1e-5, 5e-5], 600),
    ];
    println!("=========================================================================");
    println!(" ANALYTIC DEPOLARISING MODEL VS MONTE-CARLO PAULI NOISE (compiled UCCSD circuit)");
    println!("=========================================================================");
    let cfg = VqeConfig::default();
    let target = CHEMICAL_ACCURACY * 1e3;
    let mut rows: Vec<Row> = Vec::new();
    let mut tolerances = Vec::new();
    for (formula, grid, trajectories) in cases {
        let s = solve_detailed(&by_formula(formula).expect("molecule"), &cfg).expect("solve");
        let h = &s.qubit_hamiltonian;
        let n = h.n_qubits;
        let psi = prepare_state(&s.excitations, &s.reference, &s.theta);
        let stats = TermStatistics::new(h, &s.sector, &psi);
        let hf = s.sector.dets[s.sector.hartree_fock()] as usize;
        let ops: Vec<Op> = compile_ansatz(&s.excitations, &s.theta).iter().map(to_op).collect();
        let cnots = ops.iter().filter(|o| o.ctrl != 0).count();
        let one_q = ops.len() - cnots;
        let mut init = vec![(0.0, 0.0); 1 << n];
        init[hf] = (1.0, 0.0);
        let e_circuit = trajectory(&init, &ops, &[], None, 1, h.constant, &h.terms);
        let e_ideal = stats.ideal_energy();
        println!(
            "\n {formula}: {n} qubits, {} gates ({cnots} CNOT, {one_q} one-qubit); circuit energy {e_circuit:.8} vs ideal {e_ideal:.8} (diff {:.1e})",
            ops.len(),
            (e_circuit - e_ideal).abs()
        );
        println!(
            " {:>8} {:>9} {:>14} {:>22} {:>8}",
            "p2", "F", "analytic mHa", "Monte-Carlo mHa (se)", "MC/an."
        );
        let mut mc_points = Vec::new();
        for &p2 in grid {
            let p1 = p2 / 10.0;
            let f = (1.0 - p1).powi(one_q as i32) * (1.0 - p2).powi(cnots as i32);
            let analytic = (f - 1.0) * (e_ideal - h.constant);
            let probs: Vec<f64> = ops.iter().map(|o| if o.ctrl == 0 { p1 } else { p2 }).collect();
            // P(first error at gate i) = prod_{j<i}(1 - p_j) p_i; the total is 1 - F.
            let mut cum = Vec::with_capacity(ops.len());
            let (mut alive, mut total) = (1.0f64, 0.0f64);
            for &pj in &probs {
                total += alive * pj;
                alive *= 1.0 - pj;
                cum.push(total);
            }
            let any_error = total;
            let samples: Vec<f64> = (0..trajectories)
                .into_par_iter()
                .map(|i| {
                    let mut r = Rng::new(7919 + i as u64);
                    let u = r.unit() * any_error;
                    let first = cum.partition_point(|&c| c < u).min(ops.len() - 1);
                    trajectory(&init, &ops, &probs, Some(first), 1000 + i as u64, h.constant, &h.terms)
                })
                .collect();
            let n_s = samples.len() as f64;
            let mean = samples.iter().sum::<f64>() / n_s;
            let var = samples.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (n_s - 1.0);
            let bias = any_error * (mean - e_circuit);
            let se = any_error * (var / n_s).sqrt();
            println!(
                " {:>8.0e} {:>9.4} {:>14.3} {:>14.3} ({:>5.3}) {:>8.3}",
                p2,
                f,
                analytic * 1e3,
                bias * 1e3,
                se * 1e3,
                bias / analytic
            );
            mc_points.push((p2, bias.abs() * 1e3));
            rows.push(Row {
                formula: formula.to_string(),
                p2,
                p1,
                one_qubit_gates: one_q,
                cnots,
                fidelity_analytic: f,
                probability_of_any_error: any_error,
                bias_analytic_mha: analytic * 1e3,
                bias_monte_carlo_mha: bias * 1e3,
                bias_monte_carlo_stderr_mha: se * 1e3,
                trajectories,
                ratio_mc_over_analytic: bias / analytic,
            });
        }
        // analytic crossing by bisection on p2
        let (mut lo, mut hi) = (1e-12f64, 0.5f64);
        for _ in 0..80 {
            let mid = (lo * hi).sqrt();
            let f = (1.0 - mid / 10.0).powi(one_q as i32) * (1.0 - mid).powi(cnots as i32);
            if ((1.0 - f) * (e_ideal - h.constant)).abs() * 1e3 < target {
                lo = mid
            } else {
                hi = mid
            }
        }
        tolerances.push((formula, lo, crossing(&mc_points, target)));
    }
    println!("\n Largest two-qubit error (p1 = p2/10) keeping the energy bias under {target:.2} mHa:");
    for (f, a, m) in &tolerances {
        match m {
            Some(m) => println!("  {f:<5} analytic {a:.1e}   Monte-Carlo (interpolated) {m:.1e}   ({:.1}x higher)", m / a),
            None => println!("  {f:<5} analytic {a:.1e}   Monte-Carlo: not bracketed by the grid"),
        }
    }
    let dir = Path::new("results/data");
    let _ = fs::create_dir_all(dir);
    let path = dir.join("vqe_noise_check_report.json");
    t1000::results::write(&path, serde_json::to_string_pretty(&rows).expect("serialise")).expect("write report");
    println!("\n [OK] report: {}", path.display());
}
