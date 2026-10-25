// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Golden test vectors for the 1000_QPU engine.
//!
//! One command stream and the exact reply the bit-exact model gives to it. The simulation
//! testbench and the hardware conformance script replay the same stream, so RTL, board and model
//! are compared byte for byte.
//!
//! ```text
//! qpu_vectors <out_dir>          small suite for simulation
//! qpu_vectors <out_dir> --full   adds the full LiH UCCSD circuit (hardware only)
//! ```
//!
//! Files: `stim.bin`/`expect.bin` (raw bytes), `stim.mem`/`expect.mem` (one hex byte per line for
//! `$readmemh`) and `sizes.mem` (their lengths and the frame
//! count) and `frames.mem` (frame length and reply length of every frame).

use std::fs;
use std::path::PathBuf;
use t1000::qpu::compile::{cnot, compile_ansatz, h, rz, s};
use t1000::qpu::fixed::{quantize_coef, quantize_state, COEF_MAX, COEF_MIN};
use t1000::qpu::model::Gate;
use t1000::qpu::protocol::*;
use t1000::vqe::driver::{solve_detailed, VqeConfig};
use t1000::vqe::molecules::by_formula;

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

    fn signed(&mut self, bits: u32) -> i64 {
        (self.next() as i64) >> (64 - bits)
    }
}

/// Accumulates the stream and the model's reply.
struct Suite {
    session: Session,
    stim: Vec<u8>,
    expect: Vec<u8>,
    /// (frame length, reply length) of every frame, in order.
    frames: Vec<(usize, usize)>,
}

impl Suite {
    fn send(&mut self, frame: Vec<u8>) {
        let reply = self.session.process(&frame);
        self.frames.push((frame.len(), reply.len()));
        self.expect.extend(reply);
        self.stim.extend(frame);
    }

    fn gate(&mut self, g: &Gate) {
        self.send(frame_gate(g));
    }

    fn dump(&mut self, qubits: usize) {
        self.send(frame_read(0, (1u32 << qubits) as u16));
        for mask in 0..(1u32 << qubits.min(4)) {
            self.send(frame_expect_z(mask as u16));
        }
    }
}

fn random_state(rng: &mut Rng, count: usize, bits: u32) -> Vec<(i64, i64)> {
    (0..count).map(|_| (rng.signed(bits), rng.signed(bits))).collect()
}

fn random_gate(rng: &mut Rng, n: usize) -> Gate {
    let target = rng.below(n as u64) as u8;
    let mut ctrl = (rng.next() as u16) & ((1u16 << n) - 1) & !(1u16 << target);
    if rng.below(3) != 0 {
        ctrl &= (rng.next() as u16) & (rng.next() as u16);
    }
    let mut m = [0i64; 8];
    for c in &mut m {
        *c = match rng.below(8) {
            0 => COEF_MAX,
            1 => COEF_MIN,
            2 => 0,
            3 => rng.signed(25),
            _ => quantize_coef((rng.signed(20) as f64) / f64::from(1u32 << 19)),
        };
    }
    Gate { target, ctrl, m }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out_dir = PathBuf::from(args.next().expect("usage: qpu_vectors <out_dir> [--full]"));
    let full = args.any(|a| a == "--full");
    let mut rng = Rng(0x1000_9D0C_A11E_5EED);
    let mut su = Suite { session: Session::new(), stim: Vec::new(), expect: Vec::new(), frames: Vec::new() };
    let one = quantize_state(1.0, 28);

    // 1. identity and unknown opcodes
    su.send(frame_ping());
    su.send(frame_version());
    su.send(vec![0x7F]);
    su.send(vec![0x00]);

    // 2. load, read back, Z strings on 3 qubits
    su.send(frame_set_n(3));
    su.send(frame_reset(one));
    su.send(frame_read(0, 8));
    let data = random_state(&mut rng, 8, 31);
    su.send(frame_load(0, &data));
    su.dump(3);
    su.send(frame_load(5, &[(i64::from(i32::MAX), i64::from(i32::MIN)), (-1, 1)]));
    su.send(frame_read(4, 4));
    su.send(frame_load(2, &[]));
    su.send(frame_read(8, 0));

    // 3. illegal operations leave the state alone
    su.send(frame_set_n(0));
    su.send(frame_set_n(15));
    su.send(frame_load(7, &[(9, 9), (9, 9)]));
    su.send(frame_read(7, 2));
    su.send(frame_read(0, 9));
    su.send(frame_expect_z(8));
    for (target, ctrl) in [(3u8, 0u16), (0, 0b001), (1, 0b1000), (200, 0)] {
        su.gate(&Gate { target, ctrl, ..h(0) });
    }
    su.dump(3);

    // 4. random controlled gates with saturating coefficients
    for n in [1usize, 2, 3, 5, 6] {
        su.send(frame_set_n(n as u8));
        su.send(frame_load(0, &random_state(&mut rng, 1 << n, 30)));
        for _ in 0..24 {
            let g = random_gate(&mut rng, n);
            su.gate(&g);
        }
        su.dump(n);
    }

    // 5. the converged H2 UCCSD circuit from the Hartree-Fock state
    let cfg = VqeConfig::default();
    let circuit = |formula: &str, su: &mut Suite| {
        let solved = solve_detailed(&by_formula(formula).expect("molecule"), &cfg).expect("solve");
        let n = solved.report.n_qubits;
        let hf = solved.sector.dets[solved.sector.hartree_fock()];
        su.send(frame_set_n(n as u8));
        su.send(frame_reset(0));
        su.send(frame_load(hf as u16, &[(one, 0)]));
        for g in compile_ansatz(&solved.excitations, &solved.theta) {
            su.gate(&g);
        }
        su.send(frame_read(0, (1u32 << n.min(4)) as u16));
        su.send(frame_read(hf as u16, 1));
        for mask in [0u16, 1, 3, 5, (1 << n) - 1] {
            su.send(frame_expect_z(mask));
        }
    };
    circuit("H2", &mut su);

    // 6. a 14-qubit smoke test touching the high address bits
    su.send(frame_set_n(14));
    su.send(frame_reset(one));
    for q in 0..14 {
        su.gate(&h(q));
    }
    su.gate(&cnot(0, 13));
    su.gate(&cnot(13, 6));
    su.gate(&rz(7, 0.4));
    su.gate(&s(12));
    su.gate(&random_gate(&mut rng, 14));
    su.send(frame_read(0, 6));
    su.send(frame_read(16378, 6));
    su.send(frame_read(8189, 5));
    su.send(frame_expect_z(0));
    su.send(frame_expect_z(0x3FFF));
    su.send(frame_expect_z(0x2001));

    // 7. hardware burst circuit execution
    su.send(frame_set_n(3));
    su.send(frame_reset(one));
    let burst_circuit = vec![
        h(0),
        cnot(0, 1),
        rz(1, 0.4),
        h(2),
        cnot(1, 2),
        s(0),
    ];
    su.send(frame_load_circuit(&burst_circuit));
    su.send(frame_run_burst());
    su.dump(3);
    su.send(frame_reset(one));
    su.send(frame_run_burst_expect(0b011));
    su.send(frame_run_burst_expect(0b101));

    // 8. recovery
    su.send(vec![0xFF]);
    su.send(frame_ping());

    if full {
        circuit("LiH", &mut su);
        su.send(frame_ping());
    }

    fs::create_dir_all(&out_dir).expect("create output directory");
    let mem = |bytes: &[u8]| bytes.iter().map(|b| format!("{b:02x}\n")).collect::<String>();
    fs::write(out_dir.join("stim.bin"), &su.stim).expect("write");
    fs::write(out_dir.join("expect.bin"), &su.expect).expect("write");
    fs::write(out_dir.join("stim.mem"), mem(&su.stim)).expect("write");
    fs::write(out_dir.join("expect.mem"), mem(&su.expect)).expect("write");
    fs::write(
        out_dir.join("sizes.mem"),
        format!("{:08x}
{:08x}
{:08x}
", su.stim.len(), su.expect.len(), su.frames.len()),
    )
    .expect("write");
    let table: String = su.frames.iter().map(|(f, r)| format!("{f:08x}
{r:08x}
")).collect();
    fs::write(out_dir.join("frames.mem"), table).expect("write");
    println!(
        "wrote {} stimulus bytes and {} expected reply bytes to {}",
        su.stim.len(),
        su.expect.len(),
        out_dir.display()
    );
}
