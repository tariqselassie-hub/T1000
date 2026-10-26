// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Gate-level preparation of a real, non-negative amplitude vector.
//!
//! The vector `a` of `2^m` amplitudes is split down a binary tree: for each qubit from the most
//! significant down, an Ry rotation with angle `2 atan2(norm of the 1-half, norm of the 0-half)`
//! is applied, multiplexed over the more significant qubits (a different angle for every value
//! of those qubits). Each multiplexed rotation with `k` controls becomes `2^k` single-qubit
//! rotations interleaved with `2^k` CNOTs on a Gray-code schedule (Möttönen et al., 2004), so
//! the whole preparation needs `2^m - 1` rotations and at most `2^m - 2` CNOTs.
//!
//! Sparse vectors allow two savings, both exact. The angle of a branch whose amplitudes are all
//! zero is a free choice, and since the rotations of a multiplexor are a Walsh transform of the
//! angles, those free angles can be chosen to make as many rotations vanish as possible. CNOTs
//! that share a target (or a control) commute, so once rotations between them are gone, pairs of
//! identical CNOTs cancel.
//!
//! Qubit `q` is bit `q` of the amplitude index, matching the rest of the engine.

use crate::compiler::CircuitBuilder;

/// One gate of a preparation circuit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrepGate {
    Ry { qubit: usize, theta: f64 },
    Cnot { control: usize, target: usize },
}

/// Gate counts of a preparation circuit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GateCounts {
    pub rotations: usize,
    pub cnots: usize,
}

pub fn counts(gates: &[PrepGate]) -> GateCounts {
    gates.iter().fold(GateCounts::default(), |mut c, g| {
        match g {
            PrepGate::Ry { .. } => c.rotations += 1,
            PrepGate::Cnot { .. } => c.cnots += 1,
        }
        c
    })
}

/// Rotation angles of the tree level for target qubit `t`, one per value of the qubits above it.
fn level_angles(amps: &[f64], m: usize, t: usize) -> (Vec<f64>, Vec<bool>) {
    let controls = m - 1 - t;
    let mut free = Vec::new();
    let angles = (0..1usize << controls)
        .map(|j| {
            let norm = |bit: usize| -> f64 {
                (0..1usize << t)
                    .map(|low| {
                        let idx = (j << (t + 1)) | (bit << t) | low;
                        amps[idx] * amps[idx]
                    })
                    .sum::<f64>()
                    .sqrt()
            };
            let (zero, one) = (norm(0), norm(1));
            free.push(zero + one < 1e-300);
            if zero + one < 1e-300 {
                0.0
            } else {
                2.0 * one.atan2(zero)
            }
        })
        .collect();
    (angles, free)
}

fn gray(i: usize) -> usize {
    i ^ (i >> 1)
}

/// `theta_i = 2^-k * sum_j (-1)^popcount(j & gray(i)) * alpha_j` (rotation angles from branch angles).
fn walsh_entry(i: usize, j: usize, n: usize) -> f64 {
    (if (j & gray(i)).count_ones().is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }) / n as f64
}

/// Chooses the free branch angles so that the rotations listed first in `order` vanish, as far as
/// the free angles allow. Returns the completed branch angles.
fn zero_with_free_angles(angles: &[f64], free: &[bool], order: &[usize]) -> Vec<f64> {
    let n = angles.len();
    let free_idx: Vec<usize> = (0..n).filter(|&j| free[j]).collect();
    if free_idx.is_empty() {
        return angles.to_vec();
    }
    // Keep a row only if it is independent of the rows kept so far (over the free columns).
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut rows: Vec<usize> = Vec::new();
    for &i in order {
        let mut r: Vec<f64> = free_idx.iter().map(|&j| walsh_entry(i, j, n)).collect();
        for b in &basis {
            let pivot = b.iter().position(|v| v.abs() > 1e-12).unwrap_or(0);
            let factor = r[pivot] / b[pivot];
            r.iter_mut().zip(b).for_each(|(x, y)| *x -= factor * y);
        }
        if r.iter().any(|v| v.abs() > 1e-9) && rows.len() < free_idx.len() {
            basis.push(r);
            rows.push(i);
        }
    }
    // Solve  sum_{j free} A[i][j] x_j = - sum_{j fixed} A[i][j] alpha_j  for i in rows.
    let mut aug: Vec<Vec<f64>> = rows
        .iter()
        .map(|&i| {
            let mut row: Vec<f64> = free_idx.iter().map(|&j| walsh_entry(i, j, n)).collect();
            row.push(
                -(0..n)
                    .filter(|&j| !free[j])
                    .map(|j| walsh_entry(i, j, n) * angles[j])
                    .sum::<f64>(),
            );
            row
        })
        .collect();
    let mut pivots = Vec::new();
    for r in 0..aug.len() {
        let col = (0..free_idx.len())
            .filter(|c| !pivots.contains(c))
            .max_by(|a, b| aug[r][*a].abs().total_cmp(&aug[r][*b].abs()));
        let Some(col) = col else { break };
        let div = aug[r][col];
        aug[r].iter_mut().for_each(|v| *v /= div);
        for q in 0..aug.len() {
            if q != r {
                let f = aug[q][col];
                let pivot_row = aug[r].clone();
                aug[q]
                    .iter_mut()
                    .zip(&pivot_row)
                    .for_each(|(x, y)| *x -= f * y);
            }
        }
        pivots.push(col);
    }
    let mut out = angles.to_vec();
    for (r, &col) in pivots.iter().enumerate() {
        out[free_idx[col]] = aug[r][free_idx.len()];
    }
    out
}

/// The gate sequence for a Ry on `target` multiplexed over `controls` (angle `angles[j]` when the
/// control qubits, least significant first, hold the value `j`).
fn multiplexed_ry(angles: &[f64], target: usize, controls: &[usize], out: &mut Vec<PrepGate>) {
    let k = controls.len();
    if k == 0 {
        out.push(PrepGate::Ry {
            qubit: target,
            theta: angles[0],
        });
        return;
    }
    let n = 1usize << k;
    for i in 0..n {
        let theta = (0..n)
            .map(|j| walsh_entry(i, j, n) * angles[j])
            .sum::<f64>();
        out.push(PrepGate::Ry {
            qubit: target,
            theta,
        });
        // The control that flips between gray(i) and gray(i + 1); the last wraps to the top bit.
        let flipped = if i + 1 < n {
            (gray(i) ^ gray(i + 1)).trailing_zeros() as usize
        } else {
            k - 1
        };
        out.push(PrepGate::Cnot {
            control: controls[flipped],
            target,
        });
    }
}

/// Preparation circuit for `amps` (length `2^m`, non-negative, any scale) from the plain
/// construction, with no use of zero amplitudes.
pub fn amplitude_prep_raw(amps: &[f64]) -> Vec<PrepGate> {
    build(amps, false)
}

fn build(amps: &[f64], use_free_angles: bool) -> Vec<PrepGate> {
    let m = amps.len().trailing_zeros() as usize;
    assert_eq!(
        1usize << m,
        amps.len(),
        "amplitude count must be a power of two"
    );
    assert!(
        amps.iter().all(|a| *a >= 0.0),
        "amplitudes must be non-negative"
    );
    let mut gates = Vec::new();
    for t in (0..m).rev() {
        let controls: Vec<usize> = (t + 1..m).collect();
        let (angles, free) = level_angles(amps, m, t);
        if !use_free_angles || !free.iter().any(|f| *f) {
            multiplexed_ry(&angles, t, &controls, &mut gates);
            continue;
        }
        // Try several orders of which rotations to eliminate and keep the cheapest level.
        let n = angles.len();
        let orders: [Vec<usize>; 4] = [
            (0..n).collect(),
            (0..n).rev().collect(),
            (0..n).step_by(2).chain((1..n).step_by(2)).collect(),
            (1..n).step_by(2).chain((0..n).step_by(2)).collect(),
        ];
        let best = orders
            .iter()
            .map(|order| {
                let mut level = Vec::new();
                multiplexed_ry(
                    &zero_with_free_angles(&angles, &free, order),
                    t,
                    &controls,
                    &mut level,
                );
                optimize(&level)
            })
            .min_by_key(|g| {
                let c = counts(g);
                (c.cnots, c.rotations)
            })
            .unwrap_or_default();
        gates.extend(best);
    }
    gates
}

/// Drops rotations with a vanishing angle and cancels pairs of identical CNOTs until stable.
///
/// Two CNOTs commute unless the control of one is the target of the other, so a CNOT may be moved
/// past other CNOTs (and past gates on unrelated qubits) to meet its twin.
pub fn optimize(gates: &[PrepGate]) -> Vec<PrepGate> {
    let mut gates: Vec<PrepGate> = gates
        .iter()
        .copied()
        .filter(|g| !matches!(g, PrepGate::Ry { theta, .. } if theta.abs() < 1e-12))
        .collect();
    loop {
        let mut removed = false;
        'scan: for i in 0..gates.len() {
            let PrepGate::Cnot { control, target } = gates[i] else {
                continue;
            };
            for j in i + 1..gates.len() {
                match gates[j] {
                    g if g == gates[i] => {
                        gates.remove(j);
                        gates.remove(i);
                        removed = true;
                        break 'scan;
                    }
                    PrepGate::Ry { qubit, .. } if qubit == control || qubit == target => break,
                    PrepGate::Cnot {
                        control: c,
                        target: t,
                    } if c == target || t == control => break,
                    _ => {}
                }
            }
        }
        if !removed {
            return gates;
        }
    }
}

/// Optimised preparation circuit for `amps`.
pub fn amplitude_prep(amps: &[f64]) -> Vec<PrepGate> {
    optimize(&build(amps, true))
}

/// Builds the engine circuit for `gates` on `n_qubits` qubits.
pub fn to_circuit(gates: &[PrepGate], n_qubits: usize) -> CircuitBuilder {
    let mut b = CircuitBuilder::new(n_qubits);
    for g in gates {
        match *g {
            PrepGate::Ry { qubit, theta } => b.ry(qubit, theta),
            PrepGate::Cnot { control, target } => b.cnot(control, target),
        };
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::QuantumState;

    fn pseudo(len: usize, seed: u64, sparsity: usize) -> Vec<f64> {
        let mut x = seed;
        (0..len)
            .map(|i| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                if i < sparsity {
                    0.05 + (x >> 11) as f64 / (1u64 << 53) as f64
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Largest deviation between the prepared state and the normalised target.
    fn deviation(amps: &[f64], gates: &[PrepGate], n_qubits: usize) -> f64 {
        let mut state = QuantumState::ground(1 << n_qubits);
        to_circuit(gates, n_qubits).compile().execute(&mut state);
        let norm = amps.iter().map(|a| a * a).sum::<f64>().sqrt();
        let mut worst = state.imag.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (i, got) in state.real.iter().enumerate() {
            let want = amps.get(i).map_or(0.0, |a| a / norm);
            worst = worst.max((got - want).abs());
        }
        worst
    }

    #[test]
    fn prepares_the_normalised_vector_for_dense_and_sparse_inputs() {
        for (m, sparsity, seed) in [
            (1, 2, 3),
            (2, 4, 5),
            (3, 8, 7),
            (4, 16, 11),
            (5, 32, 13),
            (5, 19, 17),
            (5, 5, 19),
        ] {
            let amps = pseudo(1 << m, seed, sparsity);
            for gates in [amplitude_prep_raw(&amps), amplitude_prep(&amps)] {
                assert!(
                    deviation(&amps, &gates, m) < 1e-10,
                    "m={m} sparsity={sparsity}"
                );
            }
        }
    }

    #[test]
    fn the_prepared_register_can_sit_inside_a_larger_one() {
        // 19 amplitudes on qubits 0..4 of a 10-qubit register; qubits 5..9 stay in |0>.
        let amps = pseudo(32, 23, 19);
        let gates = amplitude_prep(&amps);
        assert!(deviation(&amps, &gates, 10) < 1e-10);
        assert!(gates.iter().all(|g| match *g {
            PrepGate::Ry { qubit, .. } => qubit < 5,
            PrepGate::Cnot { control, target } => control < 5 && target < 5,
        }));
    }

    #[test]
    fn gate_counts_follow_the_construction() {
        let dense = amplitude_prep_raw(&pseudo(32, 29, 32));
        assert_eq!(
            counts(&dense),
            GateCounts {
                rotations: 31,
                cnots: 30
            }
        );
    }

    #[test]
    fn optimisation_never_adds_gates_and_helps_sparse_vectors() {
        let sparse = pseudo(32, 31, 19);
        let (raw, opt) = (
            counts(&amplitude_prep_raw(&sparse)),
            counts(&amplitude_prep(&sparse)),
        );
        assert!(opt.cnots <= raw.cnots && opt.rotations <= raw.rotations);
        assert!(
            opt.cnots < raw.cnots,
            "a 19-of-32 vector should lose CNOTs: {raw:?} -> {opt:?}"
        );
    }

    #[test]
    fn zero_amplitude_branches_need_no_rotation() {
        // |01> (qubit 0 set) is a single Ry(pi) on qubit 0.
        let amps = [0.0, 1.0, 0.0, 0.0];
        let gates = amplitude_prep(&amps);
        assert_eq!(
            counts(&gates),
            GateCounts {
                rotations: 1,
                cnots: 0
            }
        );
        assert!(deviation(&amps, &gates, 2) < 1e-10);
    }
}
