// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! How the fixed-point error of the 1000_QPU arithmetic grows with qubit count and circuit depth.
//!
//! Random circuits (layers of one random single-qubit gate on every qubit followed by n/2 random
//! CZ or CNOT gates) run through the bit-exact model at several state scales and are compared with
//! the same circuit in double precision with the ideal (unrounded) gate matrices. The infidelity
//! is `1 - |<ref|fixed>|^2 / (<ref|ref><fixed|fixed>)`. Median of three circuits per cell.

use serde::Serialize;
use std::f64::consts::{FRAC_1_SQRT_2, PI};
use std::fs;
use std::path::Path;
use t1000::qpu::fixed::{dequantize_state, quantize_coef, quantize_state};
use t1000::qpu::model::{Gate, QpuModel};

type C = (f64, f64);
type Mat = [C; 4];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

struct Op {
    target: usize,
    ctrl: usize, // control mask
    m: Mat,
}

fn single(rng: &mut Rng) -> Mat {
    let theta = rng.unit() * 2.0 * PI;
    let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    match rng.below(6) {
        0 => [(c, 0.0), (-s, 0.0), (s, 0.0), (c, 0.0)],  // RY
        1 => [(c, -s), (0.0, 0.0), (0.0, 0.0), (c, s)],  // RZ
        2 => [(c, 0.0), (0.0, -s), (0.0, -s), (c, 0.0)], // RX
        3 => [
            (FRAC_1_SQRT_2, 0.0),
            (FRAC_1_SQRT_2, 0.0),
            (FRAC_1_SQRT_2, 0.0),
            (-FRAC_1_SQRT_2, 0.0),
        ], // H
        4 => [(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 1.0)], // S
        _ => [
            (1.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (FRAC_1_SQRT_2, FRAC_1_SQRT_2),
        ], // T
    }
}

fn circuit(n: usize, layers: usize, rng: &mut Rng) -> Vec<Op> {
    let mut ops = Vec::new();
    for _ in 0..layers {
        for q in 0..n {
            ops.push(Op {
                target: q,
                ctrl: 0,
                m: single(rng),
            });
        }
        for _ in 0..n / 2 {
            let a = rng.below(n as u64) as usize;
            let mut b = rng.below(n as u64) as usize;
            while b == a {
                b = rng.below(n as u64) as usize;
            }
            let m = if rng.below(2) == 0 {
                [(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (-1.0, 0.0)] // Z (CZ)
            } else {
                [(0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 0.0)] // X (CNOT)
            };
            ops.push(Op {
                target: a,
                ctrl: 1 << b,
                m,
            });
        }
    }
    ops
}

fn apply_f64(state: &mut [C], op: &Op) {
    let bit = 1usize << op.target;
    for i0 in 0..state.len() {
        if i0 & bit != 0 || i0 & op.ctrl != op.ctrl {
            continue;
        }
        let i1 = i0 | bit;
        let (a, b) = (state[i0], state[i1]);
        let mul = |m: C, v: C| (m.0 * v.0 - m.1 * v.1, m.0 * v.1 + m.1 * v.0);
        let (p, q, r, s) = (
            mul(op.m[0], a),
            mul(op.m[1], b),
            mul(op.m[2], a),
            mul(op.m[3], b),
        );
        state[i0] = (p.0 + q.0, p.1 + q.1);
        state[i1] = (r.0 + s.0, r.1 + s.1);
    }
}

fn gate_of(op: &Op) -> Gate {
    let mut m = [0i64; 8];
    for (k, c) in op.m.iter().enumerate() {
        m[2 * k] = quantize_coef(c.0);
        m[2 * k + 1] = quantize_coef(c.1);
    }
    Gate {
        target: op.target as u8,
        ctrl: op.ctrl as u16,
        m,
    }
}

fn infidelity(reference: &[C], fixed: &[C]) -> f64 {
    let (mut re, mut im, mut nr, mut nf) = (0.0, 0.0, 0.0, 0.0);
    for (r, f) in reference.iter().zip(fixed) {
        re += r.0 * f.0 + r.1 * f.1;
        im += r.0 * f.1 - r.1 * f.0;
        nr += r.0 * r.0 + r.1 * r.1;
        nf += f.0 * f.0 + f.1 * f.1;
    }
    (1.0 - (re * re + im * im) / (nr * nf)).max(0.0)
}

#[derive(Serialize)]
struct Row {
    qubits: usize,
    layers: usize,
    gates: usize,
    fraction_bits: u32,
    median_infidelity: f64,
    worst_infidelity: f64,
    median_norm_drift: f64,
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn main() {
    const FRACS: [u32; 5] = [16, 20, 24, 28, 30];
    const SEEDS: usize = 3;
    println!("=========================================================================");
    println!(" 1000_QPU FIXED-POINT ERROR VS QUBITS AND DEPTH (median of {SEEDS} random circuits)");
    println!("=========================================================================");
    print!(" {:>3} {:>6} {:>7}", "n", "layers", "gates");
    for f in FRACS {
        print!(" {:>11}", format!("1-F @ {f}b"));
    }
    println!("   (infidelity vs double precision)");
    let mut rows = Vec::new();
    for n in [4usize, 6, 8, 10, 12, 14] {
        for layers in [4usize, 16, 64] {
            let mut cells: Vec<Vec<f64>> = vec![Vec::new(); FRACS.len()];
            let mut drift: Vec<Vec<f64>> = vec![Vec::new(); FRACS.len()];
            let mut gates = 0;
            for seed in 0..SEEDS {
                let mut rng = Rng(0x2026_0A11_5EED_0000
                    ^ ((n as u64) << 32)
                    ^ ((layers as u64) << 16)
                    ^ seed as u64
                    ^ 1);
                let ops = circuit(n, layers, &mut rng);
                gates = ops.len();
                let mut reference = vec![(0.0, 0.0); 1 << n];
                reference[0] = (1.0, 0.0);
                for op in &ops {
                    apply_f64(&mut reference, op);
                }
                for (fi, &frac) in FRACS.iter().enumerate() {
                    let mut q = QpuModel::new();
                    q.set_qubits(n);
                    q.reset(quantize_state(1.0, frac));
                    for op in &ops {
                        assert!(q.apply_gate(&gate_of(op)));
                    }
                    let amps = q.read(0, 1 << n).expect("read");
                    let fixed: Vec<C> = amps
                        .iter()
                        .map(|a| (dequantize_state(a.0, frac), dequantize_state(a.1, frac)))
                        .collect();
                    cells[fi].push(infidelity(&reference, &fixed));
                    drift[fi].push(
                        (fixed.iter().map(|a| a.0 * a.0 + a.1 * a.1).sum::<f64>() - 1.0).abs(),
                    );
                }
            }
            print!(" {:>3} {:>6} {:>7}", n, layers, gates);
            for (fi, &frac) in FRACS.iter().enumerate() {
                let worst = cells[fi].iter().cloned().fold(0.0, f64::max);
                let med = median(&mut cells[fi]);
                let md = median(&mut drift[fi]);
                print!(" {:>11.2e}", med);
                rows.push(Row {
                    qubits: n,
                    layers,
                    gates,
                    fraction_bits: frac,
                    median_infidelity: med,
                    worst_infidelity: worst,
                    median_norm_drift: md,
                });
            }
            println!();
        }
    }
    let dir = Path::new("results/data");
    let _ = fs::create_dir_all(dir);
    let path = dir.join("qpu_scaling_report.json");
    t1000::results::write(
        &path,
        serde_json::to_string_pretty(&rows).expect("serialise"),
    )
    .expect("write report");
    println!("\n [OK] report: {}", path.display());
}
