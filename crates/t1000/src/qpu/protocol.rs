// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Byte protocol of the 1000_QPU engine on the 3 Mbaud UART, and the reference session that
//! answers it from the bit-exact model.
//!
//! All multi-byte fields are little-endian. A frame is an opcode followed by fixed arguments;
//! every command that succeeds ends its reply with [`ACK`]. A command that is understood but
//! illegal (register size, address range, gate operands) is answered with the single byte
//! [`ERR`] after the device has consumed the whole frame and changed nothing; an unknown opcode
//! is answered with [`UNKNOWN`].
//!
//! | opcode | name      | arguments                                   | reply (before ACK)         |
//! |--------|-----------|---------------------------------------------|----------------------------|
//! | `0xAA` | PING      | none                                        | `0x10` only (no ACK)       |
//! | `0x01` | VERSION   | none                                        | `0x51 0x50 0x55 0x01` ("QPU", 1) |
//! | `0x02` | SET_N     | `n` (1 byte, 1..=14)                        | none                       |
//! | `0x03` | RESET     | `one` (i32): amplitude 0 real part          | none                       |
//! | `0x04` | LOAD      | `addr` u16, `count` u16, `count` x (re, im) i32 | none                   |
//! | `0x05` | READ      | `addr` u16, `count` u16                     | `count` x (re, im) i32     |
//! | `0x06` | GATE      | `target` u8, `ctrl` u16, 8 x i32 coefficients | none                     |
//! | `0x07` | EXPECT_Z  | `mask` u16                                  | 10-byte signed sum (mask 0 is the norm) |

use super::model::{Gate, QpuModel};

pub const PING: u8 = 0xAA;
pub const PING_REPLY: u8 = 0x10;
pub const VERSION: u8 = 0x01;
pub const SET_N: u8 = 0x02;
pub const RESET: u8 = 0x03;
pub const LOAD: u8 = 0x04;
pub const READ: u8 = 0x05;
pub const GATE: u8 = 0x06;
pub const EXPECT_Z: u8 = 0x07;

pub const ACK: u8 = 0xEE;
pub const ERR: u8 = 0xE1;
pub const UNKNOWN: u8 = 0xE0;
pub const IDENT: [u8; 4] = [0x51, 0x50, 0x55, 0x01];

/// Encodes a signed 80-bit value as 10 little-endian bytes.
pub fn encode_i80(v: i128) -> [u8; 10] {
    let mut out = [0u8; 10];
    out.copy_from_slice(&v.to_le_bytes()[..10]);
    out
}

/// Decodes 10 little-endian bytes as a signed 80-bit value.
pub fn decode_i80(b: &[u8]) -> i128 {
    let mut raw = [0u8; 16];
    raw[..10].copy_from_slice(&b[..10]);
    (i128::from_le_bytes(raw) << 48) >> 48
}

pub fn frame_ping() -> Vec<u8> {
    vec![PING]
}

pub fn frame_version() -> Vec<u8> {
    vec![VERSION]
}

pub fn frame_set_n(n: u8) -> Vec<u8> {
    vec![SET_N, n]
}

pub fn frame_reset(one: i64) -> Vec<u8> {
    let mut f = vec![RESET];
    f.extend((one as i32).to_le_bytes());
    f
}

pub fn frame_load(addr: u16, data: &[(i64, i64)]) -> Vec<u8> {
    let mut f = vec![LOAD];
    f.extend(addr.to_le_bytes());
    f.extend((data.len() as u16).to_le_bytes());
    for &(r, i) in data {
        f.extend((r as i32).to_le_bytes());
        f.extend((i as i32).to_le_bytes());
    }
    f
}

pub fn frame_read(addr: u16, count: u16) -> Vec<u8> {
    let mut f = vec![READ];
    f.extend(addr.to_le_bytes());
    f.extend(count.to_le_bytes());
    f
}

pub fn frame_gate(g: &Gate) -> Vec<u8> {
    let mut f = vec![GATE, g.target];
    f.extend(g.ctrl.to_le_bytes());
    for &c in &g.m {
        f.extend((c as i32).to_le_bytes());
    }
    f
}

pub fn frame_expect_z(mask: u16) -> Vec<u8> {
    let mut f = vec![EXPECT_Z];
    f.extend(mask.to_le_bytes());
    f
}

fn le32(b: &[u8]) -> i64 {
    i64::from(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn le16(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}

/// A device session: the model plus the protocol on top of it.
#[derive(Debug, Default, Clone)]
pub struct Session {
    pub model: QpuModel,
}

impl Session {
    pub fn new() -> Self {
        Self::default()
    }

    /// Answers every complete frame in `input`; an incomplete trailing frame is ignored.
    pub fn process(&mut self, input: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut at = 0;
        while at < input.len() {
            match self.frame(&input[at..], &mut out) {
                Some(used) => at += used,
                None => break,
            }
        }
        out
    }

    /// Handles the frame at the start of `b`; returns the bytes it used, or `None` if incomplete.
    fn frame(&mut self, b: &[u8], out: &mut Vec<u8>) -> Option<usize> {
        let need = |n: usize| (b.len() >= n).then_some(n);
        match b[0] {
            PING => {
                out.push(PING_REPLY);
                Some(1)
            }
            VERSION => {
                out.extend(IDENT);
                out.push(ACK);
                Some(1)
            }
            SET_N => {
                need(2)?;
                out.push(if self.model.set_qubits(usize::from(b[1])) { ACK } else { ERR });
                Some(2)
            }
            RESET => {
                need(5)?;
                self.model.reset(le32(&b[1..5]));
                out.push(ACK);
                Some(5)
            }
            LOAD => {
                need(5)?;
                let (addr, count) = (usize::from(le16(&b[1..3])), usize::from(le16(&b[3..5])));
                let total = need(5 + 8 * count)?;
                let data: Vec<(i64, i64)> = (0..count)
                    .map(|k| (le32(&b[5 + 8 * k..]), le32(&b[9 + 8 * k..])))
                    .collect();
                out.push(if self.model.load(addr, &data) { ACK } else { ERR });
                Some(total)
            }
            READ => {
                need(5)?;
                let (addr, count) = (usize::from(le16(&b[1..3])), usize::from(le16(&b[3..5])));
                match self.model.read(addr, count) {
                    Some(amps) => {
                        for (r, i) in amps {
                            out.extend((r as i32).to_le_bytes());
                            out.extend((i as i32).to_le_bytes());
                        }
                        out.push(ACK);
                    }
                    None => out.push(ERR),
                }
                Some(5)
            }
            GATE => {
                need(36)?;
                let mut m = [0i64; 8];
                for (k, c) in m.iter_mut().enumerate() {
                    *c = le32(&b[4 + 4 * k..]);
                }
                let g = Gate { target: b[1], ctrl: le16(&b[2..4]), m };
                out.push(if self.model.apply_gate(&g) { ACK } else { ERR });
                Some(36)
            }
            EXPECT_Z => {
                need(3)?;
                match self.model.expect_z(usize::from(le16(&b[1..3]))) {
                    Some(v) => {
                        out.extend(encode_i80(v));
                        out.push(ACK);
                    }
                    None => out.push(ERR),
                }
                Some(3)
            }
            _ => {
                out.push(UNKNOWN);
                Some(1)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::compile::{cnot, h};
    use super::super::fixed::quantize_state;
    use super::*;

    #[test]
    fn ping_version_and_unknown_opcodes() {
        let mut s = Session::new();
        assert_eq!(s.process(&frame_ping()), vec![PING_REPLY]);
        assert_eq!(s.process(&frame_version()), vec![0x51, 0x50, 0x55, 1, ACK]);
        assert_eq!(s.process(&[0x7F, 0x00]), vec![UNKNOWN, UNKNOWN]);
    }

    #[test]
    fn a_bell_pair_round_trips_through_frames() {
        let one = quantize_state(1.0, 28);
        let mut s = Session::new();
        let mut cmds = frame_set_n(2);
        cmds.extend(frame_reset(one));
        cmds.extend(frame_gate(&h(0)));
        cmds.extend(frame_gate(&cnot(0, 1)));
        cmds.extend(frame_read(0, 4));
        cmds.extend(frame_expect_z(0));
        let out = s.process(&cmds);
        assert_eq!(&out[..4], &[ACK, ACK, ACK, ACK]);
        let amps = &out[4..4 + 32];
        let re = |k: usize| i32::from_le_bytes(amps[8 * k..8 * k + 4].try_into().unwrap());
        let r = std::f64::consts::FRAC_1_SQRT_2;
        assert!((f64::from(re(0)) / 2f64.powi(28) - r).abs() < 1e-6);
        assert_eq!(re(1), 0);
        assert!((f64::from(re(3)) / 2f64.powi(28) - r).abs() < 1e-6);
        assert_eq!(out[36], ACK);
        let norm = decode_i80(&out[37..47]);
        assert!((norm as f64 / 2f64.powi(56) - 1.0).abs() < 1e-5);
        assert_eq!(out[47], ACK);
    }

    #[test]
    fn illegal_frames_are_consumed_whole_and_change_nothing() {
        let mut s = Session::new();
        s.process(&frame_set_n(3));
        s.process(&frame_reset(1 << 20));
        let before = s.model.read(0, 8).unwrap();
        let bad = frame_load(7, &[(5, 5), (6, 6)]);
        assert_eq!(s.process(&bad), vec![ERR]);
        assert_eq!(s.process(&frame_read(7, 2)), vec![ERR]);
        assert_eq!(s.process(&frame_set_n(15)), vec![ERR]);
        assert_eq!(s.process(&frame_expect_z(8)), vec![ERR]);
        assert_eq!(s.model.read(0, 8).unwrap(), before);
    }

    #[test]
    fn eighty_bit_values_round_trip() {
        for v in [0i128, 1, -1, (1 << 77) - 5, -(1 << 77)] {
            assert_eq!(decode_i80(&encode_i80(v)), v);
        }
    }

    #[test]
    fn incomplete_trailing_frames_are_ignored() {
        let mut s = Session::new();
        let mut cmds = frame_ping();
        cmds.extend(&frame_gate(&h(0))[..10]);
        assert_eq!(s.process(&cmds), vec![PING_REPLY]);
    }
}
