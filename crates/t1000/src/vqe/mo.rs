// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Molecular-orbital integrals and the frozen-core active-space Hamiltonian.

use super::integrals::AoIntegrals;
use super::scf::RhfResult;

/// The electronic Hamiltonian over an active set of spatial orbitals, in chemists' notation:
/// `H = constant + Σ h_pq E_pq + ½ Σ (pq|rs) (E_pq E_rs - δ_qr E_ps)`.
#[derive(Debug, Clone)]
pub struct ActiveHamiltonian {
    /// Number of active spatial orbitals.
    pub n_orb: usize,
    /// Electrons occupying the active orbitals.
    pub n_electrons: usize,
    /// Nuclear repulsion plus the frozen-core energy.
    pub constant: f64,
    /// Effective one-electron integrals (core Hamiltonian plus frozen-core field), `n_orb²`.
    pub h: Vec<f64>,
    /// `(pq|rs)` flattened as `((p n + q) n + r) n + s`.
    pub g: Vec<f64>,
}

impl ActiveHamiltonian {
    /// `(pq|rs)`.
    pub fn eri(&self, p: usize, q: usize, r: usize, s: usize) -> f64 {
        self.g[((p * self.n_orb + q) * self.n_orb + r) * self.n_orb + s]
    }
}

/// `Cᵀ h C` for the AO matrix `h`.
fn one_electron_to_mo(h: &[f64], c: &[f64], n: usize) -> Vec<f64> {
    let mut hc = vec![0.0; n * n];
    for mu in 0..n {
        for q in 0..n {
            hc[mu * n + q] = (0..n).map(|nu| h[mu * n + nu] * c[nu * n + q]).sum();
        }
    }
    let mut out = vec![0.0; n * n];
    for p in 0..n {
        for q in 0..n {
            out[p * n + q] = (0..n).map(|mu| c[mu * n + p] * hc[mu * n + q]).sum();
        }
    }
    out
}

/// One quarter-transformation of a four-index tensor along `axis`, using `c[ao * n + mo]`.
fn transform_index(t: &[f64], c: &[f64], n: usize, axis: usize) -> Vec<f64> {
    let stride = n.pow((3 - axis) as u32);
    let mut out = vec![0.0; t.len()];
    for (i, o) in out.iter_mut().enumerate() {
        let mo = (i / stride) % n;
        let base = i - mo * stride;
        *o = (0..n)
            .map(|ao| c[ao * n + mo] * t[base + ao * stride])
            .sum();
    }
    out
}

/// All AO two-electron integrals transformed to the MO basis.
fn eri_to_mo(ints: &AoIntegrals, c: &[f64]) -> Vec<f64> {
    let n = ints.n;
    let mut t = ints.eri.clone();
    for axis in 0..4 {
        t = transform_index(&t, c, n, axis);
    }
    t
}

/// Builds the active-space Hamiltonian: the first `n_frozen` molecular orbitals are held doubly
/// occupied, the next `n_active` (all remaining if `None`) are correlated.
pub fn active_space(
    ints: &AoIntegrals,
    scf: &RhfResult,
    n_frozen: usize,
    n_active: Option<usize>,
) -> Result<ActiveHamiltonian, String> {
    let n = ints.n;
    let n_active = n_active.unwrap_or(n - n_frozen);
    if n_frozen + n_active > n || n_frozen > scf.n_occupied {
        return Err("active space exceeds the basis or freezes virtual orbitals".into());
    }
    let h_mo = one_electron_to_mo(&ints.h, &scf.coefficients, n);
    let g_mo = eri_to_mo(ints, &scf.coefficients);
    let g = |p: usize, q: usize, r: usize, s: usize| g_mo[((p * n + q) * n + r) * n + s];

    let mut constant = ints.nuclear_repulsion;
    for i in 0..n_frozen {
        constant += 2.0 * h_mo[i * n + i];
        for j in 0..n_frozen {
            constant += 2.0 * g(i, i, j, j) - g(i, j, j, i);
        }
    }
    let m = n_active;
    let mut h = vec![0.0; m * m];
    for p in 0..m {
        for q in 0..m {
            let (pp, qq) = (n_frozen + p, n_frozen + q);
            let field: f64 = (0..n_frozen)
                .map(|i| 2.0 * g(pp, qq, i, i) - g(pp, i, i, qq))
                .sum();
            h[p * m + q] = h_mo[pp * n + qq] + field;
        }
    }
    let mut gm = vec![0.0; m * m * m * m];
    for p in 0..m {
        for q in 0..m {
            for r in 0..m {
                for s in 0..m {
                    gm[((p * m + q) * m + r) * m + s] =
                        g(n_frozen + p, n_frozen + q, n_frozen + r, n_frozen + s);
                }
            }
        }
    }
    Ok(ActiveHamiltonian {
        n_orb: m,
        n_electrons: 2 * (scf.n_occupied - n_frozen),
        constant,
        h,
        g: gm,
    })
}

/// Hartree-Fock energy of the active-space Hamiltonian (closed-shell determinant).
pub fn hartree_fock_energy(ham: &ActiveHamiltonian) -> f64 {
    let n_occ = ham.n_electrons / 2;
    let m = ham.n_orb;
    let mut e = ham.constant;
    for i in 0..n_occ {
        e += 2.0 * ham.h[i * m + i];
        for j in 0..n_occ {
            e += 2.0 * ham.eri(i, i, j, j) - ham.eri(i, j, j, i);
        }
    }
    e
}

#[cfg(test)]
mod tests {
    use super::super::basis::{sto3g_basis, Atom};
    use super::super::integrals::compute_ao_integrals;
    use super::super::scf::rhf;
    use super::*;

    fn setup(atoms: &[Atom], electrons: usize) -> (AoIntegrals, RhfResult) {
        let basis = sto3g_basis(atoms).unwrap();
        let ints = compute_ao_integrals(&basis, atoms);
        let scf = rhf(&ints, electrons).unwrap();
        (ints, scf)
    }

    #[test]
    fn the_full_space_determinant_energy_equals_the_scf_energy() {
        let atoms = [
            Atom::angstrom(3, [0.0; 3]),
            Atom::angstrom(1, [0.0, 0.0, 1.595]),
        ];
        let (ints, scf) = setup(&atoms, 4);
        let ham = active_space(&ints, &scf, 0, None).unwrap();
        assert!((hartree_fock_energy(&ham) - scf.energy).abs() < 1e-9);
    }

    #[test]
    fn freezing_the_core_preserves_the_hartree_fock_energy() {
        let atoms = [
            Atom::angstrom(3, [0.0; 3]),
            Atom::angstrom(1, [0.0, 0.0, 1.595]),
        ];
        let (ints, scf) = setup(&atoms, 4);
        let ham = active_space(&ints, &scf, 1, None).unwrap();
        assert_eq!((ham.n_orb, ham.n_electrons), (5, 2));
        assert!((hartree_fock_energy(&ham) - scf.energy).abs() < 1e-9);
        let small = active_space(&ints, &scf, 1, Some(3)).unwrap();
        assert!((hartree_fock_energy(&small) - scf.energy).abs() < 1e-9);
    }

    #[test]
    fn mo_integrals_keep_the_eightfold_symmetry_and_orthonormality() {
        let atoms = [
            Atom::angstrom(8, [0.0, 0.0, 0.1]),
            Atom::angstrom(1, [0.0, 0.76, -0.5]),
        ];
        let (ints, scf) = setup(&atoms, 8);
        let ham = active_space(&ints, &scf, 0, None).unwrap();
        for &(p, q, r, s) in &[(0, 1, 2, 3), (4, 4, 1, 0), (2, 5, 3, 1)] {
            let v = ham.eri(p, q, r, s);
            for w in [
                ham.eri(q, p, r, s),
                ham.eri(p, q, s, r),
                ham.eri(r, s, p, q),
            ] {
                assert!((v - w).abs() < 1e-11);
            }
        }
    }

    #[test]
    fn invalid_active_spaces_are_rejected() {
        let atoms = [
            Atom::angstrom(3, [0.0; 3]),
            Atom::angstrom(1, [0.0, 0.0, 1.595]),
        ];
        let (ints, scf) = setup(&atoms, 4);
        assert!(active_space(&ints, &scf, 3, None).is_err());
        assert!(active_space(&ints, &scf, 1, Some(9)).is_err());
    }
}
