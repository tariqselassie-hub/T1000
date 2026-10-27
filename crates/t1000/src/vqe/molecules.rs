// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! The benchmark molecules: equilibrium geometries for the nine Period 1-2 species.

use super::basis::Atom;

/// A molecule: its nuclei and net charge.
#[derive(Debug, Clone)]
pub struct Molecule {
    pub name: &'static str,
    pub formula: &'static str,
    pub atoms: Vec<Atom>,
    pub charge: i32,
}

impl Molecule {
    /// Electron count: nuclear charge minus the net charge.
    pub fn electrons(&self) -> usize {
        let nuclear: i32 = self.atoms.iter().map(|a| a.z as i32).sum();
        (nuclear - self.charge) as usize
    }

    /// Doubly occupied core orbitals to freeze: one 1s per atom heavier than helium.
    pub fn core_orbitals(&self) -> usize {
        self.atoms.iter().filter(|a| a.z > 2).count()
    }
}

fn atoms_at(z: &[u32], pos: &[[f64; 3]]) -> Vec<Atom> {
    z.iter()
        .zip(pos)
        .map(|(&z, &p)| Atom::angstrom(z, p))
        .collect()
}

/// Hydrides `X-H_n` with the heavy atom at the origin and hydrogens given by `h`.
fn hydride(z: u32, h: &[[f64; 3]]) -> Vec<Atom> {
    let mut atoms = vec![Atom::angstrom(z, [0.0; 3])];
    atoms.extend(h.iter().map(|&p| Atom::angstrom(1, p)));
    atoms
}

/// Hydrogens of a pyramidal `XH₃` at bond length `r` and H-X-H angle `theta` (degrees).
fn pyramid(r: f64, theta_deg: f64) -> Vec<[f64; 3]> {
    let cos_theta = theta_deg.to_radians().cos();
    let cos_a = ((cos_theta + 0.5) / 1.5).sqrt();
    let sin_a = (1.0 - cos_a * cos_a).sqrt();
    (0..3)
        .map(|k| {
            let phi = 120f64.to_radians() * f64::from(k);
            [r * sin_a * phi.cos(), r * sin_a * phi.sin(), -r * cos_a]
        })
        .collect()
}

/// A diatomic along the z axis with bond length `r` ångström.
pub fn diatomic(
    name: &'static str,
    formula: &'static str,
    z: [u32; 2],
    r: f64,
    charge: i32,
) -> Molecule {
    Molecule {
        name,
        formula,
        atoms: atoms_at(&z, &[[0.0; 3], [0.0, 0.0, r]]),
        charge,
    }
}

/// A linear chain of `n` hydrogen atoms (n = 2, 4, 6 or 8) spaced `d` ångström along z.
///
/// Stretching the chain uniformly drives it from weak to strong correlation: at large `d` every
/// atom carries one unpaired electron and a single determinant is a poor reference.
pub fn hydrogen_chain(n: usize, d: f64) -> Molecule {
    let formula = match n {
        2 => "H2",
        4 => "H4",
        6 => "H6",
        8 => "H8",
        _ => panic!("hydrogen_chain supports n = 2, 4, 6, 8"),
    };
    let pos: Vec<[f64; 3]> = (0..n).map(|k| [0.0, 0.0, d * k as f64]).collect();
    Molecule {
        name: "hydrogen chain",
        formula,
        atoms: atoms_at(&vec![1; n], &pos),
        charge: 0,
    }
}

/// Four hydrogens on a rectangle with sides `a` and `b` ångström in the xy plane.
///
/// At `a = b` (the square) the two lowest determinants are degenerate by symmetry, the textbook
/// multireference case; for `a ≠ b` the system is two weakly coupled H₂ molecules.
pub fn hydrogen_rectangle(a: f64, b: f64) -> Molecule {
    let (x, y) = (a / 2.0, b / 2.0);
    Molecule {
        name: "hydrogen rectangle",
        formula: "H4",
        atoms: atoms_at(
            &[1; 4],
            &[[-x, -y, 0.0], [x, -y, 0.0], [x, y, 0.0], [-x, y, 0.0]],
        ),
        charge: 0,
    }
}

/// The nine benchmark molecules at their equilibrium geometries.
pub fn catalog() -> Vec<Molecule> {
    let r_h2o = 0.9572;
    let half = 104.52f64.to_radians() / 2.0;
    let r_bh = 1.19;
    let t = 1.087 / 3f64.sqrt();
    vec![
        Molecule {
            name: "Molecular Hydrogen",
            formula: "H2",
            atoms: atoms_at(&[1, 1], &[[0.0; 3], [0.0, 0.0, 0.7414]]),
            charge: 0,
        },
        Molecule {
            name: "Helium Hydride Ion",
            formula: "HeH+",
            atoms: atoms_at(&[2, 1], &[[0.0; 3], [0.0, 0.0, 0.772]]),
            charge: 1,
        },
        Molecule {
            name: "Lithium Hydride",
            formula: "LiH",
            atoms: atoms_at(&[3, 1], &[[0.0; 3], [0.0, 0.0, 1.595]]),
            charge: 0,
        },
        Molecule {
            name: "Hydrogen Fluoride",
            formula: "HF",
            atoms: atoms_at(&[9, 1], &[[0.0; 3], [0.0, 0.0, 0.9168]]),
            charge: 0,
        },
        Molecule {
            name: "Beryllium Dihydride",
            formula: "BeH2",
            atoms: hydride(4, &[[0.0, 0.0, 1.3264], [0.0, 0.0, -1.3264]]),
            charge: 0,
        },
        Molecule {
            name: "Water",
            formula: "H2O",
            atoms: hydride(
                8,
                &[
                    [r_h2o * half.sin(), 0.0, r_h2o * half.cos()],
                    [-r_h2o * half.sin(), 0.0, r_h2o * half.cos()],
                ],
            ),
            charge: 0,
        },
        Molecule {
            name: "Borane",
            formula: "BH3",
            atoms: hydride(
                5,
                &(0..3)
                    .map(|k| {
                        let phi = 120f64.to_radians() * f64::from(k);
                        [r_bh * phi.cos(), r_bh * phi.sin(), 0.0]
                    })
                    .collect::<Vec<_>>(),
            ),
            charge: 0,
        },
        Molecule {
            name: "Ammonia",
            formula: "NH3",
            atoms: hydride(7, &pyramid(1.012, 106.67)),
            charge: 0,
        },
        Molecule {
            name: "Methane",
            formula: "CH4",
            atoms: hydride(6, &[[t, t, t], [t, -t, -t], [-t, t, -t], [-t, -t, t]]),
            charge: 0,
        },
    ]
}

/// A catalog molecule by formula.
pub fn by_formula(formula: &str) -> Option<Molecule> {
    catalog()
        .into_iter()
        .find(|m| m.formula.eq_ignore_ascii_case(formula))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn electron_counts_follow_the_nuclear_charge_and_net_charge() {
        let counts: Vec<(String, usize)> = catalog()
            .iter()
            .map(|m| (m.formula.to_string(), m.electrons()))
            .collect();
        let want = [
            ("H2", 2),
            ("HeH+", 2),
            ("LiH", 4),
            ("HF", 10),
            ("BeH2", 6),
            ("H2O", 10),
            ("BH3", 8),
            ("NH3", 10),
            ("CH4", 10),
        ];
        for (formula, n) in want {
            assert_eq!(
                counts.iter().find(|(f, _)| f == formula).unwrap().1,
                n,
                "{formula}"
            );
        }
    }

    #[test]
    fn diatomic_places_the_second_atom_on_the_z_axis() {
        let m = diatomic("test", "LiH", [3, 1], 2.0, 0);
        assert!((m.atoms[1].pos[2] - 2.0 * super::super::basis::ANGSTROM_TO_BOHR).abs() < 1e-12);
        assert_eq!(m.electrons(), 4);
    }

    #[test]
    fn geometries_have_the_stated_bond_lengths_and_angles() {
        let dist = |a: &Atom, b: &Atom| {
            (0..3)
                .map(|k| (a.pos[k] - b.pos[k]).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let to_a = |bohr: f64| bohr / super::super::basis::ANGSTROM_TO_BOHR;
        let nh3 = by_formula("NH3").unwrap();
        for h in &nh3.atoms[1..] {
            assert!((to_a(dist(&nh3.atoms[0], h)) - 1.012).abs() < 1e-9);
        }
        let hh = to_a(dist(&nh3.atoms[1], &nh3.atoms[2]));
        let angle = (2.0 * (hh / 2.0 / 1.012).asin()).to_degrees();
        assert!((angle - 106.67).abs() < 1e-6, "{angle}");
        let ch4 = by_formula("CH4").unwrap();
        for h in &ch4.atoms[1..] {
            assert!((to_a(dist(&ch4.atoms[0], h)) - 1.087).abs() < 1e-9);
        }
        assert_eq!(by_formula("ch4").unwrap().core_orbitals(), 1);
        assert_eq!(by_formula("HeH+").unwrap().core_orbitals(), 0);
    }

    /// STO-3G restricted Hartree-Fock energies at the catalog geometries. LiH, BeH2, CH4 and
    /// NH3 agree with the published STO-3G values (-7.8620, -15.5603, -39.7269, -55.4554) to a
    /// few millihartree (geometry-dependent); the rest pin this engine's results.
    #[test]
    fn hartree_fock_energies_agree_with_published_sto3g_values() {
        use super::super::basis::sto3g_basis;
        use super::super::integrals::compute_ao_integrals;
        use super::super::scf::rhf;
        let published = [
            ("H2", -1.11668, 5e-5),
            ("HeH+", -2.8418, 1e-3),
            ("LiH", -7.8620, 1e-3),
            ("BeH2", -15.5603, 1e-3),
            ("H2O", -74.9630, 3e-3),
            ("HF", -98.5707, 3e-3),
            ("NH3", -55.4540, 3e-3),
            ("CH4", -39.7268, 1e-3),
            ("BH3", -26.0690, 3e-3),
        ];
        for (formula, want, tol) in published {
            let m = by_formula(formula).unwrap();
            let basis = sto3g_basis(&m.atoms).unwrap();
            let ints = compute_ao_integrals(&basis, &m.atoms);
            let r = rhf(&ints, m.electrons()).unwrap();
            assert!(r.converged, "{formula} did not converge");
            assert!(
                (r.energy - want).abs() < tol,
                "{formula}: {} vs {want}",
                r.energy
            );
        }
    }
}
