<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Terrence A. Jones SR. All rights reserved. -->

# 1000_QPU: fixed-point statevector engine on the Basys 3

Target: Digilent Basys 3, AMD Artix-7 XC7A35T-1CPG236C (20,800 LUTs, 90 DSP48E1, 50 BRAM36), 100 MHz,
FT2232HQ USB-UART at 3 Mbaud (`COM6` on the development machine).

The design is a small, exactly specified accelerator: a 14-qubit complex statevector held in block
RAM, a controlled 2x2 gate that streams one amplitude pair per clock, and a UART protocol to load,
run and read back. Its arithmetic is defined by a bit-exact Rust model, and the RTL is checked
against that model byte for byte. It is a verification platform for fixed-point quantum
simulation, not a faster simulator than the host CPU (see "What it is not").

## 1. Specification chain

| Layer | File |
| :--- | :--- |
| Arithmetic (coefficients, rounding, saturation) | `crates/t1000/src/qpu/fixed.rs`, `model.rs` |
| Byte protocol and reference session | `crates/t1000/src/qpu/protocol.rs` |
| UCCSD to gate lists | `crates/t1000/src/qpu/compile.rs` |
| Golden vectors | `cargo run --release -p t1000 --bin qpu_vectors -- fpga/1000_qpu/sim/vectors` |
| RTL | `fpga/1000_qpu/rtl/{qpu_top,qpu_gate,qpu_bank}.v` |
| Simulation | `python fpga/1000_qpu/sim/run_sim.py` |
| Hardware conformance | `python scripts/fpga/verify_1000_qpu.py` |

## 2. Arithmetic

* State component: signed 32-bit integer. The engine does not know the scale; the host chooses the
  number of fractional bits (at least 24 is accurate enough for every molecule up to 14 qubits,
  below).
* Gate coefficient: signed 25-bit, scaled by 2^23 (Q2.23), the width of a DSP48E1 multiplier port.
* A gate applies `out0 = m00 a0 + m01 a1`, `out1 = m10 a0 + m11 a1` (complex) with exact products
  and sums, then `round_half_up(sum / 2^23)` saturated to 32 bits.
* The gate acts on `target` and only on pairs whose index has all `ctrl` bits set. CNOT, CZ and
  Toffoli-style controls come from the mask; `Rz`, `H`, `S` are single-qubit matrices.
* `EXPECT_Z(mask)` returns the exact 80-bit sum of `(-1)^popcount(k & mask) (re_k^2 + im_k^2)`.
  `mask = 0` is the squared norm.

## 3. Protocol

All fields little-endian. Every successful command ends its reply with `0xEE`. A command that is
not valid (qubit count, address range, gate operands) is consumed whole, changes nothing, and is
answered with `0xE1`. An unknown opcode is answered with `0xE0`.

| Opcode | Command | Arguments | Reply before `0xEE` |
| :--- | :--- | :--- | :--- |
| `0xAA` | PING | none | `0x10` only (no `0xEE`) |
| `0x01` | VERSION | none | `51 50 55 01` |
| `0x02` | SET_N | `n` (1 byte, 1..14) | none |
| `0x03` | RESET | `one` (i32) | none; zeroes the active register, amplitude 0 = `one` |
| `0x04` | LOAD | `addr` u16, `count` u16, `count` x (re i32, im i32) | none |
| `0x05` | READ | `addr` u16, `count` u16 | `count` x (re i32, im i32) |
| `0x06` | GATE | `target` u8, `ctrl` u16, 8 x i32 (`m00r m00i m01r m01i m10r m10i m11r m11i`; low 25 bits used) | none |
| `0x07` | EXPECT_Z | `mask` u16 | 10-byte signed sum |
| `0x08` | LOAD_CIRCUIT | `count` u16, `count` x (`target` u8, `ctrl` u16, 8 x i32 `m`) | none; stores up to 128 gates in microcode RAM |
| `0x09` | RUN_BURST | none | none; executes the loaded microcode circuit at fabric speed |
| `0x0A` | RUN_BURST_EXPECT | `mask` u16 | 10-byte signed sum; runs burst circuit and measures expectation value |

Flow control: the device has a 512-byte receive FIFO and no handshake. Frames whose reply is longer
than one byte (READ, EXPECT_Z, VERSION) must be sent one at a time, waiting for the reply; GATE,
LOAD, RESET and SET_N frames may be streamed back to back (the engine is faster than the UART).
`led[13]` latches if the FIFO ever overflowed.

## 4. Microarchitecture

* Amplitude `k` lives in bank `parity(k)` at address `k >> 1` (two banks of 8192 x 64 bits, 32 BRAM36).
  The two elements of a gate pair differ in one bit, so they are always in opposite banks: each
  bank sees one read and one write per cycle and a pair is processed every clock.
* Pipeline: address, bank read, capture, 16 products, four sums, round and saturate, write
  (about 7 stages). Pairs of one gate are disjoint, so there is no hazard; the next command starts
  after the pipeline drains.
* A gate takes `2^(n-1)` cycles plus about 12: 82 microseconds at 14 qubits, 0.1 microsecond at 4.
* 16 multipliers of 25 x 32 bits use DSP48E1 slices (40 in total with the `EXPECT_Z` squarers).

## 5. Results

Fixed-point precision (`qpu_precision`, committed in `results/hosts/unknown-cpu/data/qpu_precision_report.json`):
energy of the compiled UCCSD circuit minus the same circuit in double precision.

| Molecule | Qubits | Gates | dE at 16 bits | at 24 bits | at 28 bits |
| :--- | ---: | ---: | ---: | ---: | ---: |
| H2 | 4 | 196 | 1e-6 mHa | 0 | 0 |
| LiH | 10 | 3,152 | 3.6e-5 mHa | 0 | 0 |
| H2O | 12 | 15,104 | -2.2e-3 mHa | 2.5e-5 mHa | 4e-6 mHa |
| BH3 | 14 | 37,264 | 2.9e-2 mHa | 1.2e-4 mHa | 2e-6 mHa |
| NH3 | 14 | 37,264 | 0.14 mHa | 4.3e-5 mHa | 2.1e-5 mHa |

Error against qubit count and depth (`qpu_scaling`, random circuits of single-qubit rotations,
H, S, T and CZ/CNOT, median of three per cell, infidelity against double precision with the ideal
gate matrices):

| Qubits | Gates | 16 fraction bits | 20 | 24 | 28 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 384 | 1.2e-7 | 4.3e-10 | 1.7e-12 | 2.4e-13 |
| 8 | 768 | 3.8e-6 | 1.3e-8 | 5.2e-11 | 6.9e-13 |
| 12 | 1,152 | 8.6e-5 | 3.3e-7 | 1.3e-9 | 5.9e-12 |
| 14 | 1,344 | 4.0e-4 | 1.6e-6 | 6.1e-9 | 2.5e-11 |

The infidelity follows `1 - F ≈ c · gates · 2^n · 4^{-F_bits}` with `c ≈ 0.07` (fitted over 51 cells
with the three smallest word lengths; individual cells lie between 0.01 and 0.09): it grows
linearly with the number of gates, doubles per added qubit (random states spread amplitude as
`2^{-n/2}`, so a fixed absolute rounding step is a larger relative error), and falls by a factor of
256 for every four extra fraction bits. The floor at 28 to 30 bits (1e-14 to 1e-11) is the 25-bit
coefficient rounding. For a target infidelity `ε` at 14 qubits and `G` gates, about
`½ log2(0.07 · G · 2^14 / ε)` fraction bits are needed: 23 bits for the 37,264-gate UCCSD circuit at
`ε = 1e-6`, consistent with the molecule study above. This is an empirical rule from random
circuits up to 14 qubits, not a bound; structured circuits with concentrated amplitudes behave better.

Chemical accuracy is 1.59 mHa. The norm of the state drifts by up to 1.6e-3 over 37,000 gates
because the 25-bit coefficients are not exactly unitary; this does not enter the normalised energy.

Build (Vivado 2026.1, `scripts/build_bitstream.tcl`): Artix-7 XC7A35T implementation with 128-gate
microcode RAM buffer (`1000_qpu.bit`), 32 BRAM36 blocks, 40 DSP48E1 slices, and multi-bank dual-port streaming.

Simulation (`sim/run_sim.py`, xsim): 2,716 reply bytes exactly match the Rust model for the 13,838-byte
golden test suite across all 472 frames (including `LOAD_CIRCUIT`, `RUN_BURST`, and `RUN_BURST_EXPECT`),
with 0 error bytes.

Hardware burst throughput (Basys 3 physical silicon on `COM6`, `scripts/fpga/verify_1000_qpu.py`):
* 128 gates at 4 qubits: 1,676.2 µs total (13.1 µs per gate) vs 15,360.0 µs UART serial baseline (**9.2x speedup**).
* Eliminates the 120.0 µs UART transmission floor per gate, enabling direct microcode execution at fabric clock rate.

## 6. What it is not

* It is not faster than the host: a 14-qubit UCCSD circuit is 37,264 gates, about 3 seconds of
  fabric time plus 37,264 x 36 bytes on the UART (about 4.5 seconds at 3 Mbaud), against
  milliseconds on the CPU. Molecules above 14 qubits (CH4 at 16) do not fit.
* It is not a quantum device. Its value is hardware-in-the-loop validation of the integer
  arithmetic and a reproducible target for fixed-point studies.
* The hardware conformance result for the board is recorded in `docs/CLAIMS.md`; the simulation
  result does not stand in for it.
