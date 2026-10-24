// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Measuring Hamiltonian terms in groups.
//!
//! Terms that act with the same Pauli (or the identity) on every qubit they share are
//! qubit-wise commuting: one product-basis measurement yields all of them at once. A group `G` with
//! operator `H_G = Σ c_k P_k` is estimated from its own shots with single-shot variance
//! `⟨H_G²⟩ - ⟨H_G⟩²`, which includes the covariances between its terms. With the variance-optimal
//! split of shots between groups, the total for standard error `ε` is `(Σ_G σ_G)² / ε²`, against
//! `(Σ_k |c_k| σ_k)² / ε²` when every term is measured on its own (`noise::TermStatistics`).

use super::pauli::{PauliTerm, QubitHamiltonian};

/// Whether two terms can be read out in the same product basis.
pub fn qubit_wise_commute(a: &PauliTerm, b: &PauliTerm) -> bool {
    let overlap = (a.x | a.z) & (b.x | b.z);
    ((a.x ^ b.x) | (a.z ^ b.z)) & overlap == 0
}

/// Greedy first-fit grouping, largest coefficients first. Returns indices into `h.terms`.
pub fn qwc_groups(h: &QubitHamiltonian) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..h.terms.len()).collect();
    order.sort_by(|&i, &j| h.terms[j].coeff.abs().total_cmp(&h.terms[i].coeff.abs()));
    // (basis x, basis z, members): the basis is the union of the members' supports.
    let mut groups: Vec<(u32, u32, Vec<usize>)> = Vec::new();
    for k in order {
        let t = &h.terms[k];
        let sup = t.x | t.z;
        let slot = groups.iter_mut().find(|(bx, bz, _)| {
            let overlap = sup & (*bx | *bz);
            ((t.x ^ *bx) | (t.z ^ *bz)) & overlap == 0
        });
        match slot {
            Some((bx, bz, members)) => {
                *bx |= t.x;
                *bz |= t.z;
                members.push(k);
            }
            None => groups.push((t.x, t.z, vec![k])),
        }
    }
    groups.into_iter().map(|g| g.2).collect()
}

/// Adds `c · P ψ` to `(re, im)` for the Hermitian Pauli `P = i^{|x∧z|} X^x Z^z` and a real state.
fn add_pauli_applied(re: &mut [f64], im: &mut [f64], t: &PauliTerm, psi: &[f64]) {
    let (pr, pi) = match (t.x & t.z).count_ones() % 4 {
        0 => (1.0, 0.0),
        1 => (0.0, 1.0),
        2 => (-1.0, 0.0),
        _ => (0.0, -1.0),
    };
    for j in 0..psi.len() {
        let src = j ^ t.x as usize;
        let sign = if ((src as u32) & t.z).count_ones() % 2 == 0 { 1.0 } else { -1.0 };
        let v = t.coeff * sign * psi[src];
        re[j] += pr * v;
        im[j] += pi * v;
    }
}

/// Mean and single-shot variance of every group's operator on the real state `psi` (full `2^n` basis).
pub fn group_moments(h: &QubitHamiltonian, groups: &[Vec<usize>], psi: &[f64]) -> Vec<(f64, f64)> {
    groups
        .iter()
        .map(|g| {
            let mut re = vec![0.0; psi.len()];
            let mut im = vec![0.0; psi.len()];
            for &k in g {
                add_pauli_applied(&mut re, &mut im, &h.terms[k], psi);
            }
            let norm: f64 = psi.iter().map(|v| v * v).sum();
            let mean = psi.iter().zip(&re).map(|(a, b)| a * b).sum::<f64>() / norm;
            let second = re.iter().zip(&im).map(|(a, b)| a * a + b * b).sum::<f64>() / norm;
            (mean, (second - mean * mean).max(0.0))
        })
        .collect()
}

/// Total shots for standard error `epsilon` with the variance-optimal split between groups.
pub fn grouped_total_shots(moments: &[(f64, f64)], epsilon: f64) -> f64 {
    let s: f64 = moments.iter().map(|m| m.1.sqrt()).sum();
    s * s / (epsilon * epsilon)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn term(x: u32, z: u32, coeff: f64) -> PauliTerm {
        PauliTerm { x, z, coeff }
    }

    #[test]
    fn commutation_is_per_qubit() {
        assert!(qubit_wise_commute(&term(0, 0b11, 1.0), &term(0, 0b01, 1.0))); // ZZ, ZI
        assert!(!qubit_wise_commute(&term(0b11, 0, 1.0), &term(0, 0b11, 1.0))); // XX, ZZ
        assert!(qubit_wise_commute(&term(0b01, 0, 1.0), &term(0b10, 0, 1.0))); // XI, IX
    }

    #[test]
    fn groups_cover_every_term_once_and_the_moments_match_the_matrix() {
        let h = QubitHamiltonian {
            n_qubits: 2,
            constant: 0.0,
            terms: vec![term(0, 0b11, 0.7), term(0b11, 0, 0.4), term(0, 0b01, -0.3)],
        };
        let groups = qwc_groups(&h);
        let mut all: Vec<usize> = groups.iter().flatten().copied().collect();
        all.sort();
        assert_eq!(all, vec![0, 1, 2]);
        assert_eq!(groups.len(), 2); // {ZZ, ZI} and {XX}
        // |00>: Z-type group is an eigenstate (variance 0), <XX> = 0 so the XX group has variance 0.4².
        let psi = [1.0, 0.0, 0.0, 0.0];
        let m = group_moments(&h, &groups, &psi);
        for (g, (mean, var)) in groups.iter().zip(&m) {
            if g.contains(&1) {
                assert!(mean.abs() < 1e-12);
                assert!((var - 0.16).abs() < 1e-12, "{var}");
            } else {
                assert!((mean - 0.4).abs() < 1e-12); // 0.7 - 0.3
                assert!(var.abs() < 1e-12);
            }
        }
        // Bell state (|00> + |11>)/√2: <XX> = <ZZ> = 1, <ZI> = 0.
        let r = std::f64::consts::FRAC_1_SQRT_2;
        let m = group_moments(&h, &groups, &[r, 0.0, 0.0, r]);
        for (g, (mean, var)) in groups.iter().zip(&m) {
            if g.contains(&1) {
                assert!((mean - 0.4).abs() < 1e-12 && var.abs() < 1e-12);
            } else {
                // ZZ and ZI are measured together: mean 0.7, variance 0.3² (from ZI, ZZ fixed at +1).
                assert!((mean - 0.7).abs() < 1e-12);
                assert!((var - 0.09).abs() < 1e-12, "{var}");
            }
        }
    }
}
