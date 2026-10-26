// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Compilation of UCCSD Pauli rotations to the gate set of the 1000_QPU engine.
//!
//! `exp(-i φ P)` for a Pauli string `P` becomes: a basis change to Z on every non-Z qubit
//! (`H` for X, `S†` then `H` for Y), a CNOT ladder gathering the parity on the highest qubit, the
//! diagonal `Rz(2φ) = diag(e^{-iφ}, e^{iφ})` there, then the ladder and basis changes reversed.

use super::fixed::quantize_coef;
use super::model::Gate;
use crate::vqe::circuit::{rotations, PauliRotation};
use crate::vqe::uccsd::Excitation;

const S: f64 = std::f64::consts::FRAC_1_SQRT_2;

fn single(target: u8, m: [f64; 8]) -> Gate {
    Gate {
        target,
        ctrl: 0,
        m: m.map(quantize_coef),
    }
}

/// Hadamard.
pub fn h(t: u8) -> Gate {
    single(t, [S, 0.0, S, 0.0, S, 0.0, -S, 0.0])
}

/// `S = diag(1, i)`.
pub fn s(t: u8) -> Gate {
    single(t, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
}

/// `S† = diag(1, -i)`.
pub fn sdg(t: u8) -> Gate {
    single(t, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
}

/// Controlled NOT.
pub fn cnot(control: u8, target: u8) -> Gate {
    Gate {
        target,
        ctrl: 1 << control,
        m: [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0].map(quantize_coef),
    }
}

/// `diag(e^{-iφ}, e^{iφ})`, the rotation `exp(-i φ Z)`.
pub fn rz(t: u8, phi: f64) -> Gate {
    let (sn, cs) = phi.sin_cos();
    single(t, [cs, -sn, 0.0, 0.0, 0.0, 0.0, cs, sn])
}

/// The gates of `exp(-i φ P)`.
pub fn compile_rotation(r: &PauliRotation, phi: f64) -> Vec<Gate> {
    let support: Vec<u8> = (0..32u8).filter(|&q| (r.x | r.z) >> q & 1 == 1).collect();
    let Some((&top, rest)) = support.split_last() else {
        return Vec::new();
    };
    let mut enter = Vec::new();
    for &q in &support {
        match (r.x >> q & 1, r.z >> q & 1) {
            (1, 0) => enter.push(h(q)),
            (1, 1) => {
                enter.push(sdg(q));
                enter.push(h(q));
            }
            _ => {}
        }
    }
    let ladder: Vec<Gate> = rest.iter().map(|&q| cnot(q, top)).collect();
    let mut gates = enter.clone();
    gates.extend(&ladder);
    gates.push(rz(top, phi));
    gates.extend(ladder.iter().rev());
    // Undo the basis changes: inverse of (S† then H) is (H then S).
    for &q in support.iter().rev() {
        match (r.x >> q & 1, r.z >> q & 1) {
            (1, 0) => gates.push(h(q)),
            (1, 1) => {
                gates.push(h(q));
                gates.push(s(q));
            }
            _ => {}
        }
    }
    gates
}

/// The gates of the whole UCCSD circuit `Π_k Π_t exp(-i φ_{k,t} P_{k,t})`.
pub fn compile_ansatz(excitations: &[Excitation], theta: &[f64]) -> Vec<Gate> {
    excitations
        .iter()
        .zip(theta)
        .flat_map(|(e, &t)| {
            rotations(e)
                .into_iter()
                .flat_map(move |r| compile_rotation(&r, r.coeff * t))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::fixed::{dequantize_state, quantize_state};
    use super::super::model::QpuModel;
    use super::*;
    use crate::vqe::circuit::apply_rotation;

    const F: u32 = 28;

    fn run(n: usize, basis_state: usize, gates: &[Gate]) -> QpuModel {
        let mut q = QpuModel::new();
        q.set_qubits(n);
        q.reset(0);
        q.load(basis_state, &[(quantize_state(1.0, F), 0)]);
        for g in gates {
            assert!(q.apply_gate(g));
        }
        q
    }

    fn start(n: usize, basis_state: usize) -> (Vec<f64>, Vec<f64>) {
        let mut re = vec![0.0; 1 << n];
        re[basis_state] = 1.0;
        (re, vec![0.0; 1 << n])
    }

    #[test]
    fn every_pauli_letter_matches_the_floating_point_rotation() {
        // Strings over qubits 0..4 using X, Y and Z, started from a non-trivial basis state with
        // a Hadamard to spread amplitude, so phases are exercised.
        let cases = [
            PauliRotation {
                x: 0b0001,
                z: 0b0000,
                coeff: 1.0,
            },
            PauliRotation {
                x: 0b0010,
                z: 0b0010,
                coeff: 1.0,
            },
            PauliRotation {
                x: 0b0000,
                z: 0b0100,
                coeff: 1.0,
            },
            PauliRotation {
                x: 0b1011,
                z: 0b0110,
                coeff: 1.0,
            },
            PauliRotation {
                x: 0b0111,
                z: 0b1101,
                coeff: 1.0,
            },
        ];
        for r in cases {
            let phi = 0.37;
            let pre = [h(0), h(2), s(1)];
            let mut q = run(4, 0b0101, &pre);
            for g in compile_rotation(&r, phi) {
                assert!(q.apply_gate(&g));
            }
            let (mut re, mut im) = start(4, 0b0101);
            let reference = run(4, 0b0101, &pre);
            for k in 0..16 {
                let v = reference.read(k, 1).unwrap()[0];
                re[k] = dequantize_state(v.0, F);
                im[k] = dequantize_state(v.1, F);
            }
            apply_rotation(&mut re, &mut im, &r, phi);
            for k in 0..16 {
                let v = q.read(k, 1).unwrap()[0];
                assert!(
                    (dequantize_state(v.0, F) - re[k]).abs() < 1e-5,
                    "{r:?} re[{k}]"
                );
                assert!(
                    (dequantize_state(v.1, F) - im[k]).abs() < 1e-5,
                    "{r:?} im[{k}]"
                );
            }
        }
    }

    #[test]
    fn gate_count_matches_the_documented_ladder_cost() {
        let r = PauliRotation {
            x: 0b1111,
            z: 0b1000,
            coeff: 1.0,
        };
        let gates = compile_rotation(&r, 0.1);
        let cnots = gates.iter().filter(|g| g.ctrl != 0).count();
        assert_eq!(cnots, 2 * (r.weight() - 1));
    }
}
