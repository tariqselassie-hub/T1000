// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! The unitary coupled-cluster singles-and-doubles ansatz `|ψ(θ)⟩ = Π_k exp(θ_k (T_k - T_k†)) |HF⟩`
//! with exact energies and analytic gradients.
//!
//! Each fermionic excitation generator `A = T - T†` rotates disjoint pairs of determinants
//! `(D, TD)` by the angle `θ`, so `exp(θ A)` is exact and costs one pass over the pairs. This is
//! the same state a Jordan-Wigner qubit circuit of commuting Pauli rotations prepares.

use super::fermion::{annihilate, create, Sector, SparseMatrix};

/// One excitation operator as its determinant pairs `(from, to, sign)` with `A|from⟩ = sign·|to⟩`.
#[derive(Debug, Clone)]
pub struct Excitation {
    pub label: String,
    /// Spin orbitals emptied and filled: `T = Π a†_to Π a_from`.
    pub from: Vec<usize>,
    pub to: Vec<usize>,
    pairs: Vec<(u32, u32, f64)>,
}

impl Excitation {
    /// `ψ ← exp(θ A) ψ`.
    pub fn rotate(&self, psi: &mut [f64], theta: f64) {
        let (s, c) = theta.sin_cos();
        for &(d, e, sign) in &self.pairs {
            let (x, y) = (psi[d as usize], psi[e as usize]);
            psi[d as usize] = c * x - sign * s * y;
            psi[e as usize] = c * y + sign * s * x;
        }
    }

    /// `out = A ψ`.
    pub fn apply_generator(&self, psi: &[f64], out: &mut [f64]) {
        out.iter_mut().for_each(|v| *v = 0.0);
        for &(d, e, sign) in &self.pairs {
            out[e as usize] += sign * psi[d as usize];
            out[d as usize] -= sign * psi[e as usize];
        }
    }

    pub fn pair_count(&self) -> usize {
        self.pairs.len()
    }

    /// `ψ ← (1 + t T) ψ` for the bare excitation `T` (not `T - T†`). In place is safe: a
    /// determinant `T` reaches has the `from` orbitals empty, so it is never a source of `T`.
    pub fn apply_operator(&self, psi: &mut [f64], t: f64) {
        for &(d, e, sign) in &self.pairs {
            psi[e as usize] += t * sign * psi[d as usize];
        }
    }

    /// `T|det⟩ = sign·|target⟩` for the determinant at sector index `det`, if `T` reaches one.
    pub fn target_of(&self, det: usize) -> Option<(usize, f64)> {
        self.pairs
            .iter()
            .find(|p| p.0 as usize == det)
            .map(|&(_, e, sign)| (e as usize, sign))
    }

    /// True for a two-electron excitation.
    pub fn is_double(&self) -> bool {
        self.from.len() == 2
    }

    /// The qubit excitation on the same spin orbitals: the same determinant pairs with every
    /// sign `+1`, i.e. the Jordan-Wigner parity strings dropped (Yordanov et al., PRA 102,
    /// 062612, 2020). Still particle- and spin-conserving, but not a fermionic operator.
    pub fn to_qubit(&self) -> Excitation {
        Excitation {
            label: format!("Q{}", self.label),
            from: self.from.clone(),
            to: self.to.clone(),
            pairs: self.pairs.iter().map(|&(d, e, _)| (d, e, 1.0)).collect(),
        }
    }
}

/// Applies the string `ops` (rightmost first) of `(create?, spin orbital)` to a determinant.
fn apply_string(det: u32, ops: &[(bool, usize)]) -> Option<(u32, f64)> {
    let (mut d, mut sign) = (det, 1.0);
    for &(is_create, j) in ops.iter().rev() {
        let (nd, s) = if is_create {
            create(d, j)?
        } else {
            annihilate(d, j)?
        };
        d = nd;
        sign *= s;
    }
    Some((d, sign))
}

/// The excitation `T = Π a†_to Π a_from` as determinant pairs over the sector.
fn excitation(sector: &Sector, label: String, from: &[usize], to: &[usize]) -> Excitation {
    // T = a†_to[0] a†_to[1] a_from[1] a_from[0]; the rightmost operator acts first.
    let mut ops: Vec<(bool, usize)> = to.iter().map(|&j| (true, j)).collect();
    ops.extend(from.iter().rev().map(|&j| (false, j)));
    let mut pairs = Vec::new();
    for (i, &det) in sector.dets.iter().enumerate() {
        if let Some((target, sign)) = apply_string(det, &ops) {
            if let Some(j) = sector.index_of(target) {
                pairs.push((i as u32, j as u32, sign));
            }
        }
    }
    Excitation {
        label,
        from: from.to_vec(),
        to: to.to_vec(),
        pairs,
    }
}

/// All spin-conserving single and double excitations out of the closed-shell reference with
/// `n_occ` doubly occupied spatial orbitals.
pub fn uccsd_excitations(sector: &Sector) -> Vec<Excitation> {
    let (m, n_occ) = (sector.n_orb, sector.n_alpha);
    let spin_orbital = |p: usize, s: usize| 2 * p + s;
    let mut out = Vec::new();
    for s in 0..2 {
        for i in 0..n_occ {
            for a in n_occ..m {
                let label = format!("S {i}->{a} ({})", if s == 0 { "a" } else { "b" });
                out.push(excitation(
                    sector,
                    label,
                    &[spin_orbital(i, s)],
                    &[spin_orbital(a, s)],
                ));
            }
        }
    }
    for i in 0..n_occ {
        for j in 0..n_occ {
            for a in n_occ..m {
                for b in n_occ..m {
                    let label = format!("D ab {i}{j}->{a}{b}");
                    out.push(excitation(
                        sector,
                        label,
                        &[spin_orbital(i, 0), spin_orbital(j, 1)],
                        &[spin_orbital(a, 0), spin_orbital(b, 1)],
                    ));
                }
            }
        }
    }
    for s in 0..2 {
        for i in 0..n_occ {
            for j in i + 1..n_occ {
                for a in n_occ..m {
                    for b in a + 1..m {
                        let label =
                            format!("D {} {i}{j}->{a}{b}", if s == 0 { "aa" } else { "bb" });
                        out.push(excitation(
                            sector,
                            label,
                            &[spin_orbital(i, s), spin_orbital(j, s)],
                            &[spin_orbital(a, s), spin_orbital(b, s)],
                        ));
                    }
                }
            }
        }
    }
    out
}

/// Generalised singles and doubles: spin-conserving excitations between any orbitals, not only
/// from occupied to virtual. Each generator `T - T†` appears once (the reverse excitation is the
/// same rotation with the opposite angle); doubles act on four distinct spin orbitals, and
/// operators that cannot act inside the sector are dropped.
pub fn generalized_excitations(sector: &Sector) -> Vec<Excitation> {
    let m = sector.n_orb;
    let so = |p: usize, s: usize| 2 * p + s;
    let mut out = Vec::new();
    let mut keep = |e: Excitation| {
        if e.pair_count() > 0 {
            out.push(e);
        }
    };
    for s in 0..2 {
        for p in 0..m {
            for q in p + 1..m {
                let label = format!("GS {p}->{q} ({})", if s == 0 { "a" } else { "b" });
                keep(excitation(sector, label, &[so(p, s)], &[so(q, s)]));
            }
        }
    }
    // Mixed spin: (p_α, q_β) -> (r_α, t_β) with p ≠ r and q ≠ t, each unordered pair of pairs once.
    let pairs_ab: Vec<(usize, usize)> = (0..m).flat_map(|p| (0..m).map(move |q| (p, q))).collect();
    for (u, &(p, q)) in pairs_ab.iter().enumerate() {
        for &(r, t) in &pairs_ab[u + 1..] {
            if p != r && q != t {
                let label = format!("GD ab {p}{q}->{r}{t}");
                keep(excitation(
                    sector,
                    label,
                    &[so(p, 0), so(q, 1)],
                    &[so(r, 0), so(t, 1)],
                ));
            }
        }
    }
    // Same spin: (p<q) -> (r<t) on four distinct orbitals, each unordered pair of pairs once.
    let pairs_ss: Vec<(usize, usize)> = (0..m)
        .flat_map(|p| (p + 1..m).map(move |q| (p, q)))
        .collect();
    for s in 0..2 {
        for (u, &(p, q)) in pairs_ss.iter().enumerate() {
            for &(r, t) in &pairs_ss[u + 1..] {
                if p != r && p != t && q != r && q != t {
                    let label = format!("GD {} {p}{q}->{r}{t}", if s == 0 { "aa" } else { "bb" });
                    keep(excitation(
                        sector,
                        label,
                        &[so(p, s), so(q, s)],
                        &[so(r, s), so(t, s)],
                    ));
                }
            }
        }
    }
    out
}

/// The UCCSD state `Π exp(θ_k A_k)|reference⟩`, applying `A_1` first.
pub fn prepare_state(excitations: &[Excitation], reference: &[f64], theta: &[f64]) -> Vec<f64> {
    let mut psi = reference.to_vec();
    for (e, &t) in excitations.iter().zip(theta) {
        e.rotate(&mut psi, t);
    }
    psi
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Energy `⟨ψ|H|ψ⟩` and its exact gradient with respect to every angle.
///
/// Reverse sweep: with `λ_k = (U_{k+1}…U_N)ᵀ H ψ_N`, `∂E/∂θ_k = 2 λ_k · (A_k ψ_k)`; stepping back
/// from `k` to `k-1` applies `U_kᵀ` to both `ψ` and `λ`.
pub fn energy_and_gradient(
    h: &SparseMatrix,
    excitations: &[Excitation],
    reference: &[f64],
    theta: &[f64],
) -> (f64, Vec<f64>) {
    let n = h.n;
    let mut psi = prepare_state(excitations, reference, theta);
    let mut lambda = vec![0.0; n];
    h.apply(&psi, &mut lambda);
    let energy = dot(&psi, &lambda);
    let mut grad = vec![0.0; theta.len()];
    let mut a_psi = vec![0.0; n];
    for k in (0..excitations.len()).rev() {
        excitations[k].apply_generator(&psi, &mut a_psi);
        grad[k] = 2.0 * dot(&lambda, &a_psi);
        excitations[k].rotate(&mut psi, -theta[k]);
        excitations[k].rotate(&mut lambda, -theta[k]);
    }
    (energy, grad)
}

#[cfg(test)]
mod tests {
    use super::super::basis::{sto3g_basis, Atom};
    use super::super::fermion::build_hamiltonian;
    use super::super::integrals::compute_ao_integrals;
    use super::super::mo::active_space;
    use super::super::scf::rhf;
    use super::*;

    fn h2_problem() -> (Sector, SparseMatrix) {
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
        (sector, h)
    }

    #[test]
    fn excitation_counts_follow_the_occupied_virtual_structure() {
        // 6 orbitals, 2 occupied: singles 2*2*4, mixed-spin doubles 2*2*4*4, same-spin 0 + 2*1*6.
        let s = Sector::new(6, 2, 2);
        let ex = uccsd_excitations(&s);
        assert_eq!(ex.len(), 16 + 64 + 2 * 6);
        let h2 = uccsd_excitations(&Sector::new(2, 1, 1));
        assert_eq!(h2.len(), 2 + 1);
    }

    #[test]
    fn rotations_are_orthogonal_and_the_reference_is_stationary_for_singles() {
        let (sector, _) = h2_problem();
        let ex = uccsd_excitations(&sector);
        let mut reference = vec![0.0; sector.dim()];
        reference[sector.hartree_fock()] = 1.0;
        let psi = prepare_state(&ex, &reference, &[0.3, -0.7, 1.1]);
        assert!((dot(&psi, &psi) - 1.0).abs() < 1e-14);
        // A single-excitation rotation of the closed-shell reference changes nothing at first
        // order in the energy (Brillouin's theorem): the gradient of E at θ = 0 vanishes.
        let (sector, h) = h2_problem();
        let ex = uccsd_excitations(&sector);
        let (_, g) = energy_and_gradient(&h, &ex, &reference, &[0.0; 3]);
        assert!(g[0].abs() < 1e-10 && g[1].abs() < 1e-10, "{g:?}");
        assert!(g[2].abs() > 1e-3);
    }

    #[test]
    fn analytic_gradient_matches_central_finite_differences() {
        let (sector, h) = h2_problem();
        let ex = uccsd_excitations(&sector);
        let mut reference = vec![0.0; sector.dim()];
        reference[sector.hartree_fock()] = 1.0;
        let theta = [0.21, -0.13, 0.37];
        let (_, g) = energy_and_gradient(&h, &ex, &reference, &theta);
        for k in 0..theta.len() {
            let (mut up, mut dn) = (theta, theta);
            up[k] += 1e-6;
            dn[k] -= 1e-6;
            let fd = (energy_and_gradient(&h, &ex, &reference, &up).0
                - energy_and_gradient(&h, &ex, &reference, &dn).0)
                / 2e-6;
            assert!((g[k] - fd).abs() < 1e-7, "k={k}: {} vs {fd}", g[k]);
        }
    }

    #[test]
    fn generalized_pool_contains_the_uccsd_pool() {
        // H₂: one single per spin, and the two mixed-spin doubles on four distinct spin orbitals.
        assert_eq!(generalized_excitations(&Sector::new(2, 1, 1)).len(), 4);
        let sector = Sector::new(4, 2, 2);
        let gen = generalized_excitations(&sector);
        for e in uccsd_excitations(&sector) {
            assert!(
                gen.iter().any(|g| g.from == e.from && g.to == e.to),
                "{} missing from the generalised pool",
                e.label
            );
        }
        assert!(gen.len() > uccsd_excitations(&sector).len());
    }

    #[test]
    fn qubit_excitations_drop_only_the_signs_and_stay_orthogonal() {
        let sector = Sector::new(4, 2, 2);
        let mut reference = vec![0.0; sector.dim()];
        reference[sector.hartree_fock()] = 1.0;
        let gen = generalized_excitations(&sector);
        let q: Vec<Excitation> = gen.iter().map(Excitation::to_qubit).collect();
        let theta: Vec<f64> = (0..q.len())
            .map(|k| 0.05 * ((k % 7) as f64 - 3.0))
            .collect();
        let psi = prepare_state(&q, &reference, &theta);
        assert!((dot(&psi, &psi) - 1.0).abs() < 1e-12);
        // Same determinant pairs, some fermionic signs negative, all qubit signs positive.
        assert!(gen
            .iter()
            .zip(&q)
            .all(|(f, b)| f.pair_count() == b.pair_count()));
        assert!(gen.iter().any(|f| f.pairs.iter().any(|p| p.2 < 0.0)));
        assert!(q.iter().all(|b| b.pairs.iter().all(|p| p.2 == 1.0)));
    }
}
