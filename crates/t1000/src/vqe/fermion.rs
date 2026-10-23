// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! The fixed-particle-number sector of Fock space as Slater determinants, the electronic
//! Hamiltonian in that basis, and its sparse matrix-vector product.
//!
//! Determinants are bit strings over spin orbitals ordered `2p + spin` (spin 0 = α, 1 = β),
//! the same ordering the Jordan-Wigner map uses for qubits, so a determinant is also a
//! computational basis state.

use super::mo::ActiveHamiltonian;
use std::collections::HashMap;

/// The determinants with `n_alpha` α and `n_beta` β electrons in `n_orb` spatial orbitals.
#[derive(Debug, Clone)]
pub struct Sector {
    pub n_orb: usize,
    pub n_alpha: usize,
    pub n_beta: usize,
    pub dets: Vec<u32>,
    index: HashMap<u32, usize>,
}

impl Sector {
    pub fn new(n_orb: usize, n_alpha: usize, n_beta: usize) -> Self {
        assert!(n_orb <= 16, "determinants are 32-bit");
        let dets: Vec<u32> = (0..1u32 << (2 * n_orb))
            .filter(|d| spin_count(*d, 0) == n_alpha && spin_count(*d, 1) == n_beta)
            .collect();
        let index = dets.iter().enumerate().map(|(i, &d)| (d, i)).collect();
        Sector {
            n_orb,
            n_alpha,
            n_beta,
            dets,
            index,
        }
    }

    pub fn dim(&self) -> usize {
        self.dets.len()
    }

    pub fn index_of(&self, det: u32) -> Option<usize> {
        self.index.get(&det).copied()
    }

    /// Index of the Hartree-Fock determinant: the lowest orbitals doubly occupied.
    pub fn hartree_fock(&self) -> usize {
        let mut det = 0u32;
        for p in 0..self.n_alpha {
            det |= 1 << (2 * p);
        }
        for p in 0..self.n_beta {
            det |= 1 << (2 * p + 1);
        }
        self.index_of(det)
            .expect("the Hartree-Fock determinant lies in its own sector")
    }
}

/// Number of occupied orbitals of one spin.
fn spin_count(det: u32, spin: usize) -> usize {
    (0..16).filter(|p| det >> (2 * p + spin) & 1 == 1).count()
}

/// `a_j |det⟩ = sign · |det'⟩`, or `None` if orbital `j` is empty.
pub fn annihilate(det: u32, j: usize) -> Option<(u32, f64)> {
    if det >> j & 1 == 0 {
        return None;
    }
    let sign = if (det & ((1 << j) - 1)).count_ones().is_multiple_of(2) {
        1.0
    } else {
        -1.0
    };
    Some((det & !(1 << j), sign))
}

/// `a†_j |det⟩ = sign · |det'⟩`, or `None` if orbital `j` is occupied.
pub fn create(det: u32, j: usize) -> Option<(u32, f64)> {
    if det >> j & 1 == 1 {
        return None;
    }
    let sign = if (det & ((1 << j) - 1)).count_ones().is_multiple_of(2) {
        1.0
    } else {
        -1.0
    };
    Some((det | (1 << j), sign))
}

/// A real symmetric sparse matrix in compressed-row form.
#[derive(Debug, Clone)]
pub struct SparseMatrix {
    pub n: usize,
    row_ptr: Vec<usize>,
    cols: Vec<u32>,
    vals: Vec<f64>,
}

impl SparseMatrix {
    /// `y = A x`.
    pub fn apply(&self, x: &[f64], y: &mut [f64]) {
        for (i, out) in y.iter_mut().enumerate().take(self.n) {
            let mut acc = 0.0;
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                acc += self.vals[k] * x[self.cols[k] as usize];
            }
            *out = acc;
        }
    }

    pub fn nnz(&self) -> usize {
        self.vals.len()
    }

    /// Dense element `A[i][j]` (for tests).
    pub fn get(&self, i: usize, j: usize) -> f64 {
        (self.row_ptr[i]..self.row_ptr[i + 1])
            .filter(|&k| self.cols[k] as usize == j)
            .map(|k| self.vals[k])
            .sum()
    }
}

/// Occupied spin-orbital indices of a determinant.
fn occupied(det: u32) -> impl Iterator<Item = usize> {
    (0..32).filter(move |j| det >> j & 1 == 1)
}

/// Adds `value * |det⟩` to the column being built if `det` lies in the sector.
fn push(sector: &Sector, out: &mut Vec<(usize, f64)>, det: u32, value: f64) {
    if let Some(i) = sector.index_of(det) {
        out.push((i, value));
    }
}

/// `H |det⟩` as unmerged `(row, value)` entries.
fn apply_hamiltonian(ham: &ActiveHamiltonian, sector: &Sector, det: u32) -> Vec<(usize, f64)> {
    let m = ham.n_orb;
    let mut out = Vec::new();
    push(sector, &mut out, det, ham.constant);
    // One-body: Σ h_pq a†_{pσ} a_{qσ}.
    for jq in occupied(det) {
        let (q, spin) = (jq / 2, jq % 2);
        let (d1, s1) = annihilate(det, jq).expect("occupied");
        for p in 0..m {
            if let Some((d2, s2)) = create(d1, 2 * p + spin) {
                push(sector, &mut out, d2, s1 * s2 * ham.h[p * m + q]);
            }
        }
    }
    // Two-body: ½ Σ (pq|rs) a†_{pσ} a†_{rτ} a_{sτ} a_{qσ}.
    for jq in occupied(det) {
        let (q, sigma) = (jq / 2, jq % 2);
        let (d1, s1) = annihilate(det, jq).expect("occupied");
        for js in occupied(d1) {
            let (s, tau) = (js / 2, js % 2);
            let (d2, s2) = annihilate(d1, js).expect("occupied");
            for r in 0..m {
                let Some((d3, s3)) = create(d2, 2 * r + tau) else {
                    continue;
                };
                for p in 0..m {
                    let Some((d4, s4)) = create(d3, 2 * p + sigma) else {
                        continue;
                    };
                    let v = 0.5 * ham.eri(p, q, r, s);
                    push(sector, &mut out, d4, s1 * s2 * s3 * s4 * v);
                }
            }
        }
    }
    out
}

/// The Hamiltonian matrix in the determinant basis of `sector`.
pub fn build_hamiltonian(ham: &ActiveHamiltonian, sector: &Sector) -> SparseMatrix {
    let n = sector.dim();
    let mut row_ptr = Vec::with_capacity(n + 1);
    let (mut cols, mut vals) = (Vec::new(), Vec::new());
    row_ptr.push(0);
    for &det in &sector.dets {
        // The matrix is symmetric, so the column H|det⟩ doubles as the row for `det`.
        let mut entries = apply_hamiltonian(ham, sector, det);
        entries.sort_by_key(|e| e.0);
        let mut k = 0;
        while k < entries.len() {
            let col = entries[k].0;
            let mut sum = 0.0;
            while k < entries.len() && entries[k].0 == col {
                sum += entries[k].1;
                k += 1;
            }
            if sum.abs() > 1e-13 {
                cols.push(col as u32);
                vals.push(sum);
            }
        }
        row_ptr.push(cols.len());
    }
    SparseMatrix {
        n,
        row_ptr,
        cols,
        vals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sector_sizes_are_products_of_binomials() {
        assert_eq!(Sector::new(2, 1, 1).dim(), 4);
        assert_eq!(Sector::new(6, 2, 2).dim(), 225);
        assert_eq!(Sector::new(4, 2, 1).dim(), 24);
    }

    #[test]
    fn hartree_fock_occupies_the_lowest_orbitals_of_each_spin() {
        let s = Sector::new(3, 2, 2);
        assert_eq!(s.dets[s.hartree_fock()], 0b001111);
    }

    #[test]
    fn ladder_operators_carry_jordan_wigner_signs_and_obey_exclusion() {
        let det = 0b0110u32;
        assert_eq!(annihilate(det, 1), Some((0b0100, 1.0)));
        assert_eq!(annihilate(det, 2), Some((0b0010, -1.0)));
        assert_eq!(annihilate(det, 0), None);
        assert_eq!(create(det, 0), Some((0b0111, 1.0)));
        assert_eq!(create(det, 3), Some((0b1110, 1.0)));
        assert_eq!(create(det, 1), None);
        // a†_j a_j is the number operator; a_j a†_j = 1 - n_j.
        let (d, s1) = annihilate(det, 2).unwrap();
        let (back, s2) = create(d, 2).unwrap();
        assert_eq!((back, s1 * s2), (det, 1.0));
    }

    #[test]
    fn the_two_electron_h2_hamiltonian_is_symmetric_with_the_hf_diagonal_element() {
        use super::super::basis::{sto3g_basis, Atom};
        use super::super::integrals::compute_ao_integrals;
        use super::super::mo::{active_space, hartree_fock_energy};
        use super::super::scf::rhf;
        let atoms = [
            Atom {
                z: 1,
                pos: [0.0; 3],
            },
            Atom {
                z: 1,
                pos: [0.0, 0.0, 1.4],
            },
        ];
        let basis = sto3g_basis(&atoms).unwrap();
        let ints = compute_ao_integrals(&basis, &atoms);
        let scf = rhf(&ints, 2).unwrap();
        let ham = active_space(&ints, &scf, 0, None).unwrap();
        let sector = Sector::new(2, 1, 1);
        let h = build_hamiltonian(&ham, &sector);
        for i in 0..sector.dim() {
            for j in 0..sector.dim() {
                assert!((h.get(i, j) - h.get(j, i)).abs() < 1e-12);
            }
        }
        let hf = sector.hartree_fock();
        assert!((h.get(hf, hf) - hartree_fock_energy(&ham)).abs() < 1e-10);
        assert!((h.get(hf, hf) - scf.energy).abs() < 1e-9);
    }
}
