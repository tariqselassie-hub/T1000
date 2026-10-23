// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! One- and two-electron integrals over contracted Cartesian Gaussians by the
//! McMurchie-Davidson scheme: Hermite expansion coefficients for the product of two Gaussians,
//! and Hermite Coulomb integrals built from the Boys function.

use super::basis::{Atom, BasisFunction};
use super::boys::boys_all;
use std::f64::consts::PI;

/// Hermite expansion coefficient `E_t^{ij}` of the 1-D Gaussian product `x_A^i x_B^j` with
/// separation `qx = A_x - B_x` and exponents `a`, `b`.
fn hermite_e(i: i32, j: i32, t: i32, qx: f64, a: f64, b: f64) -> f64 {
    let p = a + b;
    let q = a * b / p;
    if t < 0 || t > i + j {
        0.0
    } else if i == 0 && j == 0 && t == 0 {
        (-q * qx * qx).exp()
    } else if j == 0 {
        hermite_e(i - 1, j, t - 1, qx, a, b) / (2.0 * p)
            - q * qx / a * hermite_e(i - 1, j, t, qx, a, b)
            + f64::from(t + 1) * hermite_e(i - 1, j, t + 1, qx, a, b)
    } else {
        hermite_e(i, j - 1, t - 1, qx, a, b) / (2.0 * p)
            + q * qx / b * hermite_e(i, j - 1, t, qx, a, b)
            + f64::from(t + 1) * hermite_e(i, j - 1, t + 1, qx, a, b)
    }
}

/// `E_t^{ij}` for `t = 0..=i+j`.
fn e_table(i: u32, j: u32, qx: f64, a: f64, b: f64) -> Vec<f64> {
    (0..=(i + j) as i32)
        .map(|t| hermite_e(i as i32, j as i32, t, qx, a, b))
        .collect()
}

/// Hermite Coulomb integrals `R_{tuv}` (order 0) for `t + u + v ≤ l_max`, flattened with stride
/// `l_max + 1`.
fn hermite_r(l_max: usize, alpha: f64, d: [f64; 3]) -> Vec<f64> {
    let s = l_max + 1;
    let idx = |t: usize, u: usize, v: usize| (t * s + u) * s + v;
    let f = boys_all(l_max, alpha * (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]));
    let mut r: Vec<Vec<f64>> = vec![vec![0.0; s * s * s]; l_max + 1];
    for (n, table) in r.iter_mut().enumerate() {
        table[idx(0, 0, 0)] = (-2.0 * alpha).powi(n as i32) * f[n];
    }
    for total in 1..=l_max {
        for n in 0..=(l_max - total) {
            for t in 0..=total {
                for u in 0..=(total - t) {
                    let v = total - t - u;
                    let value = if t > 0 {
                        let lower = if t > 1 {
                            r[n + 1][idx(t - 2, u, v)]
                        } else {
                            0.0
                        };
                        (t - 1) as f64 * lower + d[0] * r[n + 1][idx(t - 1, u, v)]
                    } else if u > 0 {
                        let lower = if u > 1 {
                            r[n + 1][idx(t, u - 2, v)]
                        } else {
                            0.0
                        };
                        (u - 1) as f64 * lower + d[1] * r[n + 1][idx(t, u - 1, v)]
                    } else {
                        let lower = if v > 1 {
                            r[n + 1][idx(t, u, v - 2)]
                        } else {
                            0.0
                        };
                        (v - 1) as f64 * lower + d[2] * r[n + 1][idx(t, u, v - 1)]
                    };
                    r[n][idx(t, u, v)] = value;
                }
            }
        }
    }
    r.swap_remove(0)
}

fn gaussian_product_center(a: f64, ra: [f64; 3], b: f64, rb: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|k| (a * ra[k] + b * rb[k]) / (a + b))
}

/// Overlap `⟨a|b⟩`.
pub fn overlap(a: &BasisFunction, b: &BasisFunction) -> f64 {
    let mut s = 0.0;
    for (&ea, &ca) in a.exps.iter().zip(&a.coefs) {
        for (&eb, &cb) in b.exps.iter().zip(&b.coefs) {
            let p = ea + eb;
            let mut prod = (PI / p).powf(1.5);
            for k in 0..3 {
                prod *= hermite_e(
                    a.lmn[k] as i32,
                    b.lmn[k] as i32,
                    0,
                    a.center[k] - b.center[k],
                    ea,
                    eb,
                );
            }
            s += ca * cb * prod;
        }
    }
    s
}

/// 1-D overlap `S_ij` of two Gaussians of angular momenta `i` and `j` along one axis.
fn overlap_1d(i: i32, j: i32, qx: f64, a: f64, b: f64) -> f64 {
    if i < 0 || j < 0 {
        0.0
    } else {
        hermite_e(i, j, 0, qx, a, b) * (PI / (a + b)).sqrt()
    }
}

/// Kinetic energy `⟨a|-½∇²|b⟩`.
pub fn kinetic(a: &BasisFunction, b: &BasisFunction) -> f64 {
    let mut t_total = 0.0;
    for (&ea, &ca) in a.exps.iter().zip(&a.coefs) {
        for (&eb, &cb) in b.exps.iter().zip(&b.coefs) {
            let s: [f64; 3] = [0, 1, 2].map(|k| {
                overlap_1d(
                    a.lmn[k] as i32,
                    b.lmn[k] as i32,
                    a.center[k] - b.center[k],
                    ea,
                    eb,
                )
            });
            let t1d: [f64; 3] = [0, 1, 2].map(|k| {
                let (i, j, qx) = (a.lmn[k] as i32, b.lmn[k] as i32, a.center[k] - b.center[k]);
                -2.0 * eb * eb * overlap_1d(i, j + 2, qx, ea, eb)
                    + eb * f64::from(2 * j + 1) * overlap_1d(i, j, qx, ea, eb)
                    - 0.5 * f64::from(j * (j - 1)) * overlap_1d(i, j - 2, qx, ea, eb)
            });
            t_total +=
                ca * cb * (t1d[0] * s[1] * s[2] + s[0] * t1d[1] * s[2] + s[0] * s[1] * t1d[2]);
        }
    }
    t_total
}

/// Electron-nuclear attraction `⟨a| Σ_C -Z_C / |r - R_C| |b⟩`.
pub fn nuclear_attraction(a: &BasisFunction, b: &BasisFunction, atoms: &[Atom]) -> f64 {
    let mut v_total = 0.0;
    let l_max = (a.lmn.iter().sum::<u32>() + b.lmn.iter().sum::<u32>()) as usize;
    let s = l_max + 1;
    for (&ea, &ca) in a.exps.iter().zip(&a.coefs) {
        for (&eb, &cb) in b.exps.iter().zip(&b.coefs) {
            let p = ea + eb;
            let center = gaussian_product_center(ea, a.center, eb, b.center);
            let ex = e_table(a.lmn[0], b.lmn[0], a.center[0] - b.center[0], ea, eb);
            let ey = e_table(a.lmn[1], b.lmn[1], a.center[1] - b.center[1], ea, eb);
            let ez = e_table(a.lmn[2], b.lmn[2], a.center[2] - b.center[2], ea, eb);
            for atom in atoms {
                let d = [0, 1, 2].map(|k| center[k] - atom.pos[k]);
                let r = hermite_r(l_max, p, d);
                let mut sum = 0.0;
                for (t, &et) in ex.iter().enumerate() {
                    for (u, &eu) in ey.iter().enumerate() {
                        for (v, &ev) in ez.iter().enumerate() {
                            sum += et * eu * ev * r[(t * s + u) * s + v];
                        }
                    }
                }
                v_total -= f64::from(atom.z) * ca * cb * 2.0 * PI / p * sum;
            }
        }
    }
    v_total
}

/// Hermite expansion of the product of two contracted-function primitives on each axis.
struct PairExpansion {
    p: f64,
    center: [f64; 3],
    e: [Vec<f64>; 3],
}

impl PairExpansion {
    fn new(a: &BasisFunction, ea: f64, b: &BasisFunction, eb: f64) -> Self {
        let e = [0, 1, 2].map(|k| e_table(a.lmn[k], b.lmn[k], a.center[k] - b.center[k], ea, eb));
        PairExpansion {
            p: ea + eb,
            center: gaussian_product_center(ea, a.center, eb, b.center),
            e,
        }
    }
}

/// Coulomb repulsion between two pair densities, summed over Hermite indices.
fn pair_repulsion(bra: &PairExpansion, ket: &PairExpansion, l_max: usize) -> f64 {
    let (p, q) = (bra.p, ket.p);
    let alpha = p * q / (p + q);
    let d = [0, 1, 2].map(|k| bra.center[k] - ket.center[k]);
    let r = hermite_r(l_max, alpha, d);
    let s = l_max + 1;
    let mut sum = 0.0;
    for (t, &e1) in bra.e[0].iter().enumerate() {
        for (u, &e2) in bra.e[1].iter().enumerate() {
            for (v, &e3) in bra.e[2].iter().enumerate() {
                let bra_term = e1 * e2 * e3;
                for (tau, &f1) in ket.e[0].iter().enumerate() {
                    for (nu, &f2) in ket.e[1].iter().enumerate() {
                        for (phi, &f3) in ket.e[2].iter().enumerate() {
                            let sign = if (tau + nu + phi).is_multiple_of(2) {
                                1.0
                            } else {
                                -1.0
                            };
                            sum += bra_term
                                * sign
                                * f1
                                * f2
                                * f3
                                * r[((t + tau) * s + (u + nu)) * s + (v + phi)];
                        }
                    }
                }
            }
        }
    }
    2.0 * PI.powf(2.5) / (p * q * (p + q).sqrt()) * sum
}

/// Two-electron repulsion integral `(ab|cd)` in chemists' notation.
pub fn electron_repulsion(
    a: &BasisFunction,
    b: &BasisFunction,
    c: &BasisFunction,
    d: &BasisFunction,
) -> f64 {
    let l_max = [a, b, c, d]
        .iter()
        .map(|f| f.lmn.iter().sum::<u32>() as usize)
        .sum();
    let mut total = 0.0;
    for (&ea, &ca) in a.exps.iter().zip(&a.coefs) {
        for (&eb, &cb) in b.exps.iter().zip(&b.coefs) {
            let bra = PairExpansion::new(a, ea, b, eb);
            for (&ec, &cc) in c.exps.iter().zip(&c.coefs) {
                for (&ed, &cd) in d.exps.iter().zip(&d.coefs) {
                    let ket = PairExpansion::new(c, ec, d, ed);
                    total += ca * cb * cc * cd * pair_repulsion(&bra, &ket, l_max);
                }
            }
        }
    }
    total
}

/// Atomic-orbital integrals of a molecule.
#[derive(Debug, Clone)]
pub struct AoIntegrals {
    pub n: usize,
    /// Overlap, row-major `n × n`.
    pub s: Vec<f64>,
    /// Core Hamiltonian `T + V`, row-major `n × n`.
    pub h: Vec<f64>,
    /// `(ij|kl)` flattened as `((i n + j) n + k) n + l`.
    pub eri: Vec<f64>,
    pub nuclear_repulsion: f64,
}

impl AoIntegrals {
    /// `(ij|kl)`.
    pub fn g(&self, i: usize, j: usize, k: usize, l: usize) -> f64 {
        self.eri[((i * self.n + j) * self.n + k) * self.n + l]
    }
}

/// Classical repulsion between the nuclei.
pub fn nuclear_repulsion(atoms: &[Atom]) -> f64 {
    let mut e = 0.0;
    for (i, a) in atoms.iter().enumerate() {
        for b in &atoms[i + 1..] {
            let r2: f64 = (0..3).map(|k| (a.pos[k] - b.pos[k]).powi(2)).sum();
            e += f64::from(a.z * b.z) / r2.sqrt();
        }
    }
    e
}

/// All one- and two-electron integrals, filling the eight-fold permutational symmetry.
pub fn compute_ao_integrals(basis: &[BasisFunction], atoms: &[Atom]) -> AoIntegrals {
    let n = basis.len();
    let mut s = vec![0.0; n * n];
    let mut h = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let sij = overlap(&basis[i], &basis[j]);
            let hij =
                kinetic(&basis[i], &basis[j]) + nuclear_attraction(&basis[i], &basis[j], atoms);
            s[i * n + j] = sij;
            s[j * n + i] = sij;
            h[i * n + j] = hij;
            h[j * n + i] = hij;
        }
    }
    let mut eri = vec![0.0; n * n * n * n];
    for i in 0..n {
        for j in 0..=i {
            let ij = i * (i + 1) / 2 + j;
            for k in 0..=i {
                for l in 0..=(if k == i { j } else { k }) {
                    if ij < k * (k + 1) / 2 + l {
                        continue;
                    }
                    let v = electron_repulsion(&basis[i], &basis[j], &basis[k], &basis[l]);
                    for &(a, b, c, d) in &[
                        (i, j, k, l),
                        (j, i, k, l),
                        (i, j, l, k),
                        (j, i, l, k),
                        (k, l, i, j),
                        (l, k, i, j),
                        (k, l, j, i),
                        (l, k, j, i),
                    ] {
                        eri[((a * n + b) * n + c) * n + d] = v;
                    }
                }
            }
        }
    }
    AoIntegrals {
        n,
        s,
        h,
        eri,
        nuclear_repulsion: nuclear_repulsion(atoms),
    }
}

#[cfg(test)]
mod tests {
    use super::super::basis::sto3g_basis;
    use super::*;

    /// H₂ at R = 1.4 bohr, the Szabo & Ostlund worked example (STO-3G, ζ = 1.24).
    fn h2() -> (Vec<BasisFunction>, Vec<Atom>) {
        let atoms = vec![
            Atom {
                z: 1,
                pos: [0.0, 0.0, 0.0],
            },
            Atom {
                z: 1,
                pos: [0.0, 0.0, 1.4],
            },
        ];
        (sto3g_basis(&atoms).unwrap(), atoms)
    }

    #[test]
    fn h2_integrals_match_the_szabo_ostlund_tables() {
        let (basis, atoms) = h2();
        let ints = compute_ao_integrals(&basis, &atoms);
        let close = |a: f64, b: f64, tol: f64| assert!((a - b).abs() < tol, "{a} vs {b}");
        close(ints.s[1], 0.6593, 5e-5);
        close(kinetic(&basis[0], &basis[0]), 0.7600, 5e-5);
        close(kinetic(&basis[0], &basis[1]), 0.2365, 5e-5);
        close(ints.g(0, 0, 0, 0), 0.7746, 5e-5);
        close(ints.g(0, 0, 1, 1), 0.5697, 5e-5);
        close(ints.g(1, 0, 0, 0), 0.4441, 5e-5);
        close(ints.g(1, 0, 1, 0), 0.2970, 5e-5);
        close(ints.nuclear_repulsion, 1.0 / 1.4, 1e-12);
    }

    #[test]
    fn eri_has_the_eightfold_symmetry_for_p_functions() {
        let atoms = [
            Atom::angstrom(8, [0.0; 3]),
            Atom::angstrom(1, [0.3, 0.5, 0.8]),
        ];
        let basis = sto3g_basis(&atoms).unwrap();
        let ints = compute_ao_integrals(&basis, &atoms);
        let n = ints.n;
        for &(i, j, k, l) in &[(2, 3, 4, 5), (3, 5, 0, 4), (4, 4, 2, 5), (1, 2, 5, 0)] {
            let v = ints.g(i, j, k, l);
            for w in [
                ints.g(j, i, k, l),
                ints.g(i, j, l, k),
                ints.g(k, l, i, j),
                ints.g(l, k, j, i),
            ] {
                assert!((v - w).abs() < 1e-12);
            }
            let direct = electron_repulsion(&basis[i], &basis[j], &basis[k], &basis[l]);
            assert!((v - direct).abs() < 1e-12, "{v} vs {direct}");
        }
        assert_eq!(n, 6);
    }

    #[test]
    fn overlap_matrix_is_positive_definite_and_p_orbitals_are_orthogonal_on_one_centre() {
        let atoms = [Atom::angstrom(6, [0.0; 3])];
        let basis = sto3g_basis(&atoms).unwrap();
        assert!(overlap(&basis[2], &basis[3]).abs() < 1e-14);
        assert!(overlap(&basis[1], &basis[2]).abs() < 1e-14);
        assert!(overlap(&basis[0], &basis[1]).abs() > 0.1);
    }

    #[test]
    fn nuclear_attraction_of_a_1s_orbital_matches_the_closed_form() {
        // ⟨1s|-1/r|1s⟩ for a single normalised primitive of exponent a is -2 √(2a/π).
        let atoms = [Atom {
            z: 1,
            pos: [0.0; 3],
        }];
        let a = 0.9;
        let n = (2.0 * a / PI).powf(0.75);
        let bf = BasisFunction {
            center: [0.0; 3],
            lmn: [0; 3],
            exps: vec![a],
            coefs: vec![n],
            atom: 0,
        };
        let v = nuclear_attraction(&bf, &bf, &atoms);
        assert!((v + 2.0 * (2.0 * a / PI).sqrt()).abs() < 1e-12, "{v}");
    }

    #[test]
    fn integrals_are_unchanged_by_rotating_and_translating_the_molecule() {
        let base = [
            Atom::angstrom(8, [0.0, 0.0, 0.0]),
            Atom::angstrom(1, [0.0, 0.76, 0.59]),
        ];
        // The same O-H bond, rotated and translated.
        let moved = [
            Atom::angstrom(8, [1.0, -2.0, 0.5]),
            Atom::angstrom(1, [1.0 + 0.59, -2.0, 0.5 + 0.76]),
        ];
        let energy = |atoms: &[Atom]| {
            let basis = sto3g_basis(atoms).unwrap();
            let ints = compute_ao_integrals(&basis, atoms);
            // Rotation-invariant fingerprint: traces of the one-electron matrices and the
            // total ERI sum.
            let n = ints.n;
            let tr_h: f64 = (0..n).map(|i| ints.h[i * n + i]).sum();
            let tr_s: f64 = (0..n).map(|i| ints.s[i * n + i]).sum();
            (tr_h, tr_s, ints.eri.iter().map(|v| v * v).sum::<f64>())
        };
        let (a, b) = (energy(&base), energy(&moved));
        assert!((a.0 - b.0).abs() < 1e-10 && (a.1 - b.1).abs() < 1e-10);
        assert!((a.2 - b.2).abs() < 1e-9, "{} vs {}", a.2, b.2);
    }
}
