// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! The Jordan-Wigner qubit Hamiltonian of the active-space electronic structure problem.
//!
//! Operators are expanded in the basis `B(x, z) = X^x Z^z` (bit masks over qubits). Products obey
//! `B(x₁,z₁) B(x₂,z₂) = (-1)^{|z₁ ∧ x₂|} B(x₁⊕x₂, z₁⊕z₂)`, so every coefficient stays real, and a
//! real molecular Hamiltonian only has terms with an even number of `Y`s.

use super::mo::ActiveHamiltonian;
use std::collections::HashMap;

/// One Hermitian Pauli term `coeff · P` with `P = i^{|x∧z|} X^x Z^z` (Y where both bits are set).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PauliTerm {
    pub x: u32,
    pub z: u32,
    pub coeff: f64,
}

/// The qubit Hamiltonian `constant·I + Σ coeff_k P_k`.
#[derive(Debug, Clone)]
pub struct QubitHamiltonian {
    pub n_qubits: usize,
    pub constant: f64,
    pub terms: Vec<PauliTerm>,
}

type Key = (u32, u32);
type Sum = HashMap<Key, f64>;

/// `B(x₁,z₁) · B(x₂,z₂)` as `(key, sign)`.
fn multiply(a: Key, b: Key) -> (Key, f64) {
    let sign = if (a.1 & b.0).count_ones().is_multiple_of(2) {
        1.0
    } else {
        -1.0
    };
    ((a.0 ^ b.0, a.1 ^ b.1), sign)
}

/// `a_j` (`creation = false`) or `a†_j` (`true`) as two `B` terms.
fn ladder(j: usize, creation: bool) -> [(Key, f64); 2] {
    let below = (1u32 << j) - 1;
    let x = 1u32 << j;
    let second = if creation { 0.5 } else { -0.5 };
    [((x, below), 0.5), ((x, below | x), second)]
}

/// Adds `coeff · ladder₁ ladder₂ …` to `sum`.
fn add_product(sum: &mut Sum, coeff: f64, ops: &[(usize, bool)]) {
    let mut partial: Vec<(Key, f64)> = vec![((0, 0), coeff)];
    for &(j, creation) in ops {
        let next = ladder(j, creation);
        let mut out = Vec::with_capacity(partial.len() * 2);
        for &(key, c) in &partial {
            for &(lk, lc) in &next {
                let (nk, sign) = multiply(key, lk);
                out.push((nk, c * lc * sign));
            }
        }
        partial = out;
    }
    for (key, c) in partial {
        *sum.entry(key).or_insert(0.0) += c;
    }
}

/// Jordan-Wigner transform of the active-space Hamiltonian; qubit `2p + spin` is spin orbital
/// `(p, spin)`. Terms below `1e-12` are dropped.
pub fn jordan_wigner(ham: &ActiveHamiltonian) -> QubitHamiltonian {
    let m = ham.n_orb;
    let mut sum = Sum::new();
    for p in 0..m {
        for q in 0..m {
            let h = ham.h[p * m + q];
            if h.abs() < 1e-14 {
                continue;
            }
            for spin in 0..2 {
                add_product(&mut sum, h, &[(2 * p + spin, true), (2 * q + spin, false)]);
            }
        }
    }
    for p in 0..m {
        for q in 0..m {
            for r in 0..m {
                for s in 0..m {
                    let g = ham.eri(p, q, r, s);
                    if g.abs() < 1e-14 {
                        continue;
                    }
                    for sigma in 0..2 {
                        for tau in 0..2 {
                            let ops = [
                                (2 * p + sigma, true),
                                (2 * r + tau, true),
                                (2 * s + tau, false),
                                (2 * q + sigma, false),
                            ];
                            add_product(&mut sum, 0.5 * g, &ops);
                        }
                    }
                }
            }
        }
    }
    let mut constant = ham.constant;
    let mut terms = Vec::new();
    for ((x, z), c) in sum {
        if c.abs() < 1e-12 {
            continue;
        }
        debug_assert!(
            (x & z).count_ones().is_multiple_of(2),
            "a real Hamiltonian has an even number of Y"
        );
        let phase = if (x & z).count_ones().is_multiple_of(4) {
            1.0
        } else {
            -1.0
        };
        if x == 0 && z == 0 {
            constant += c;
        } else {
            terms.push(PauliTerm {
                x,
                z,
                coeff: c * phase,
            });
        }
    }
    terms.sort_by_key(|t| (t.x, t.z));
    QubitHamiltonian {
        n_qubits: 2 * m,
        constant,
        terms,
    }
}

impl QubitHamiltonian {
    /// Pauli string such as `"X0 Z1 Y3"` (qubits not listed carry the identity).
    pub fn label(&self, term: &PauliTerm) -> String {
        (0..self.n_qubits)
            .filter_map(|q| match (term.x >> q & 1, term.z >> q & 1) {
                (0, 0) => None,
                (1, 0) => Some(format!("X{q}")),
                (0, 1) => Some(format!("Z{q}")),
                _ => Some(format!("Y{q}")),
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Largest number of qubits any term acts on.
    pub fn max_weight(&self) -> usize {
        self.terms
            .iter()
            .map(|t| (t.x | t.z).count_ones() as usize)
            .max()
            .unwrap_or(0)
    }

    /// `⟨bra|H|ket⟩` between computational basis states.
    pub fn matrix_element(&self, bra: u32, ket: u32) -> f64 {
        let mut e = if bra == ket { self.constant } else { 0.0 };
        for t in &self.terms {
            if ket ^ t.x != bra {
                continue;
            }
            // Hermitian P = i^{|x∧z|} B, and |x∧z| is even, so P = (-1)^{|x∧z|/2} B.
            let phase = if (t.x & t.z).count_ones().is_multiple_of(4) {
                1.0
            } else {
                -1.0
            };
            let sign = if (ket & t.z).count_ones().is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            e += t.coeff * phase * sign;
        }
        e
    }

    /// `⟨ψ|H|ψ⟩` for a real state over all `2^n` computational basis states.
    pub fn expectation(&self, psi: &[f64]) -> f64 {
        let mut e = self.constant * psi.iter().map(|v| v * v).sum::<f64>();
        for t in &self.terms {
            let phase = if (t.x & t.z).count_ones().is_multiple_of(4) {
                1.0
            } else {
                -1.0
            };
            let mut acc = 0.0;
            for (i, &v) in psi.iter().enumerate() {
                if v == 0.0 {
                    continue;
                }
                let sign = if (i as u32 & t.z).count_ones().is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                };
                acc += sign * v * psi[i ^ t.x as usize];
            }
            e += t.coeff * phase * acc;
        }
        e
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `{a_i, a†_j} = δ_ij` and `{a_i, a_j} = 0` in the B-basis algebra.
    #[test]
    fn jordan_wigner_ladder_operators_obey_the_canonical_anticommutation_relations() {
        let anticommutator = |a: (usize, bool), b: (usize, bool)| -> Sum {
            let mut s = Sum::new();
            add_product(&mut s, 1.0, &[a, b]);
            add_product(&mut s, 1.0, &[b, a]);
            s.retain(|_, c| c.abs() > 1e-12);
            s
        };
        for i in 0..4 {
            for j in 0..4 {
                let aa = anticommutator((i, false), (j, true));
                if i == j {
                    assert_eq!(aa.len(), 1);
                    assert!((aa[&(0, 0)] - 1.0).abs() < 1e-12);
                } else {
                    assert!(aa.is_empty(), "{{a{i}, a†{j}}} = {aa:?}");
                }
                assert!(anticommutator((i, false), (j, false)).is_empty());
                assert!(anticommutator((i, true), (j, true)).is_empty());
            }
        }
    }

    #[test]
    fn number_operator_is_the_projector_onto_an_occupied_qubit() {
        let mut s = Sum::new();
        add_product(&mut s, 1.0, &[(2, true), (2, false)]);
        s.retain(|_, c| c.abs() > 1e-12);
        // n = (I - Z)/2
        assert!((s[&(0, 0)] - 0.5).abs() < 1e-12 && (s[&(0, 1 << 2)] + 0.5).abs() < 1e-12);
        assert_eq!(s.len(), 2);
    }
}
