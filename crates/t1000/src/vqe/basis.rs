// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Atoms and the STO-3G Gaussian basis for hydrogen through fluorine.

use super::integrals::overlap;

/// Bohr radii per Ångström.
pub const ANGSTROM_TO_BOHR: f64 = 1.889_726_124_625_770_2;

/// A nucleus: atomic number and position in bohr.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Atom {
    pub z: u32,
    pub pos: [f64; 3],
}

impl Atom {
    /// An atom at a position given in Ångström.
    pub fn angstrom(z: u32, pos: [f64; 3]) -> Self {
        Atom {
            z,
            pos: pos.map(|c| c * ANGSTROM_TO_BOHR),
        }
    }
}

/// A contracted Cartesian Gaussian `Σ_k coef_k · x^l y^m z^n · exp(-exp_k |r - center|²)`.
///
/// `coefs` already include the primitive and contraction normalisation, so the function has
/// unit self-overlap.
#[derive(Debug, Clone)]
pub struct BasisFunction {
    pub center: [f64; 3],
    pub lmn: [u32; 3],
    pub exps: Vec<f64>,
    pub coefs: Vec<f64>,
    /// Index of the atom the function sits on.
    pub atom: usize,
}

/// STO-3G contraction coefficients for the 1s, 2s and 2p shells.
const D_1S: [f64; 3] = [0.154_328_967_3, 0.535_328_142_3, 0.444_634_542_2];
const D_2S: [f64; 3] = [-0.099_967_229_19, 0.399_512_826_1, 0.700_115_468_9];
const D_2P: [f64; 3] = [0.155_916_275_0, 0.607_683_718_6, 0.391_957_393_1];

/// STO-3G exponents for Slater exponent 1; an element scales them by its `ζ²`.
const E_1S: [f64; 3] = [2.227_660_584, 0.405_771_156_2, 0.109_817_510_4];
const E_2SP: [f64; 3] = [0.994_202_729_6, 0.231_031_333_3, 0.075_138_56];

/// Standard STO-3G Slater exponents `(ζ_1s, ζ_2sp)` per element.
fn zetas(z: u32) -> Option<(f64, Option<f64>)> {
    Some(match z {
        1 => (1.24, None),
        2 => (1.69, None),
        3 => (2.69, Some(0.80)),
        4 => (3.68, Some(1.15)),
        5 => (4.68, Some(1.50)),
        6 => (5.67, Some(1.72)),
        7 => (6.67, Some(1.95)),
        8 => (7.66, Some(2.25)),
        9 => (8.65, Some(2.55)),
        _ => return None,
    })
}

fn double_factorial(n: i32) -> f64 {
    if n <= 0 {
        1.0
    } else {
        (1..=n).rev().step_by(2).map(f64::from).product()
    }
}

/// Normalisation of a primitive Cartesian Gaussian.
fn primitive_norm(a: f64, lmn: [u32; 3]) -> f64 {
    let l = lmn.iter().sum::<u32>() as i32;
    let den: f64 = lmn
        .iter()
        .map(|&k| double_factorial(2 * k as i32 - 1))
        .product();
    (2.0 * a / std::f64::consts::PI).powf(0.75) * (4.0 * a).powf(f64::from(l) / 2.0) / den.sqrt()
}

/// One contracted shell member with unit self-overlap.
fn contracted(
    center: [f64; 3],
    lmn: [u32; 3],
    exps: [f64; 3],
    d: [f64; 3],
    atom: usize,
) -> BasisFunction {
    let coefs = (0..3)
        .map(|k| d[k] * primitive_norm(exps[k], lmn))
        .collect();
    let mut bf = BasisFunction {
        center,
        lmn,
        exps: exps.to_vec(),
        coefs,
        atom,
    };
    let norm = overlap(&bf, &bf).sqrt();
    bf.coefs.iter_mut().for_each(|c| *c /= norm);
    bf
}

/// The STO-3G basis of a molecule, atom by atom: 1s, then 2s, 2px, 2py, 2pz for Li–F.
pub fn sto3g_basis(atoms: &[Atom]) -> Result<Vec<BasisFunction>, String> {
    let mut basis = Vec::new();
    for (i, atom) in atoms.iter().enumerate() {
        let (z1s, z2sp) =
            zetas(atom.z).ok_or_else(|| format!("no STO-3G data for Z = {}", atom.z))?;
        let e1 = E_1S.map(|e| e * z1s * z1s);
        basis.push(contracted(atom.pos, [0, 0, 0], e1, D_1S, i));
        if let Some(z2) = z2sp {
            let e2 = E_2SP.map(|e| e * z2 * z2);
            basis.push(contracted(atom.pos, [0, 0, 0], e2, D_2S, i));
            for lmn in [[1, 0, 0], [0, 1, 0], [0, 0, 1]] {
                basis.push(contracted(atom.pos, lmn, e2, D_2P, i));
            }
        }
    }
    Ok(basis)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponents_follow_the_published_hydrogen_and_oxygen_sets() {
        let h = sto3g_basis(&[Atom::angstrom(1, [0.0; 3])]).unwrap();
        let want = [3.425_250_914, 0.623_913_729_8, 0.168_855_404];
        for (a, b) in h[0].exps.iter().zip(want) {
            assert!((a - b).abs() < 1e-7, "{a} vs {b}");
        }
        let o = sto3g_basis(&[Atom::angstrom(8, [0.0; 3])]).unwrap();
        assert!((o[0].exps[0] - 130.709_32).abs() < 1e-3);
        assert!((o[1].exps[0] - 5.033_151_3).abs() < 1e-5);
    }

    #[test]
    fn function_counts_follow_the_shell_structure() {
        let water = [
            Atom::angstrom(8, [0.0; 3]),
            Atom::angstrom(1, [0.0, 0.76, 0.59]),
            Atom::angstrom(1, [0.0, -0.76, 0.59]),
        ];
        assert_eq!(sto3g_basis(&water).unwrap().len(), 7);
        assert!(sto3g_basis(&[Atom::angstrom(11, [0.0; 3])]).is_err());
    }

    #[test]
    fn every_function_has_unit_self_overlap() {
        let atoms = [
            Atom::angstrom(6, [0.0; 3]),
            Atom::angstrom(8, [0.0, 0.0, 1.13]),
        ];
        for bf in sto3g_basis(&atoms).unwrap() {
            assert!((overlap(&bf, &bf) - 1.0).abs() < 1e-12);
        }
    }
}
