// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Compilation of the UCCSD ansatz to Pauli rotations, with a gate-count estimate.
//!
//! Each fermionic excitation `exp(θ (T - T†))` becomes a product of commuting rotations
//! `exp(-i φ_k P_k)`, one per Jordan-Wigner Pauli term (2 for a single excitation, 8 for a double).
//! A rotation of weight `w` compiles to basis changes, a CNOT ladder (`2(w-1)` CNOTs) and one `Rz`.
//! No cancellation between neighbouring ladders is attempted, so the counts are upper bounds.

use super::pauli::excitation_generator;
use super::uccsd::Excitation;

/// One rotation `exp(-i · coeff·θ · P)` of the Hermitian Pauli `P = i^{|x∧z|} X^x Z^z`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PauliRotation {
    pub x: u32,
    pub z: u32,
    /// Rotation angle per unit of the ansatz parameter `θ`.
    pub coeff: f64,
}

impl PauliRotation {
    /// Number of qubits the rotation acts on.
    pub fn weight(&self) -> usize {
        (self.x | self.z).count_ones() as usize
    }
}

/// The rotations whose product is `exp(θ (T - T†))` for one excitation.
pub fn rotations(excitation: &Excitation) -> Vec<PauliRotation> {
    excitation_generator(&excitation.from, &excitation.to)
        .into_iter()
        .map(|(x, z, c)| {
            // B = i^{-k} P with k = |x∧z| odd: k ≡ 1 (mod 4) gives -i·P, k ≡ 3 gives +i·P, so
            // exp(θ c B) = exp(-i φ P) with φ = ±θ c.
            let sign = if (x & z).count_ones() % 4 == 1 {
                1.0
            } else {
                -1.0
            };
            PauliRotation {
                x,
                z,
                coeff: sign * c,
            }
        })
        .collect()
}

/// Gate counts of a compiled circuit.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CircuitCost {
    pub rotations: usize,
    pub cnot: usize,
    pub one_qubit: usize,
}

impl CircuitCost {
    /// Cost of one rotation: `2(w-1)` CNOTs; two basis-change gates per X or Y qubit plus the Rz.
    pub fn of(rotation: &PauliRotation) -> Self {
        CircuitCost {
            rotations: 1,
            cnot: 2 * (rotation.weight() - 1),
            one_qubit: 2 * rotation.x.count_ones() as usize + 1,
        }
    }

    pub fn add(&mut self, other: CircuitCost) {
        self.rotations += other.rotations;
        self.cnot += other.cnot;
        self.one_qubit += other.one_qubit;
    }

    pub fn two_qubit_and_one_qubit(&self) -> (usize, usize) {
        (self.cnot, self.one_qubit)
    }
}

/// Total cost of the UCCSD circuit for all `excitations`.
pub fn circuit_cost(excitations: &[Excitation]) -> CircuitCost {
    let mut total = CircuitCost::default();
    for e in excitations {
        rotations(e)
            .iter()
            .for_each(|r| total.add(CircuitCost::of(r)));
    }
    total
}

/// Applies `exp(-i φ P)` to a complex state over all `2^n` computational basis states.
pub fn apply_rotation(re: &mut [f64], im: &mut [f64], r: &PauliRotation, phi: f64) {
    let (s, c) = phi.sin_cos();
    let k = (r.x & r.z).count_ones();
    // P = i^k X^x Z^z; (Pψ)[j] = i^k (-1)^{|(j⊕x)∧z|} ψ[j⊕x].
    let (pr, pi) = match k % 4 {
        0 => (1.0, 0.0),
        1 => (0.0, 1.0),
        2 => (-1.0, 0.0),
        _ => (0.0, -1.0),
    };
    let (old_re, old_im) = (re.to_vec(), im.to_vec());
    for j in 0..re.len() {
        let src = j ^ r.x as usize;
        let sign = if ((src as u32) & r.z).count_ones().is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        // (Pψ)[j] as a complex number.
        let (ar, ai) = (sign * old_re[src], sign * old_im[src]);
        let (mr, mi) = (pr * ar - pi * ai, pr * ai + pi * ar);
        // ψ' = cos φ ψ - i sin φ Pψ
        re[j] = c * old_re[j] + s * mi;
        im[j] = c * old_im[j] - s * mr;
    }
}

/// The state `Π_k Π_t exp(-i φ_{k,t} P_{k,t}) |reference⟩` over all `2^n` basis states.
pub fn prepare_qubit_state(
    n_qubits: usize,
    reference: u32,
    excitations: &[Excitation],
    theta: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let mut re = vec![0.0; 1 << n_qubits];
    let mut im = vec![0.0; 1 << n_qubits];
    re[reference as usize] = 1.0;
    for (e, &t) in excitations.iter().zip(theta) {
        for r in rotations(e) {
            apply_rotation(&mut re, &mut im, &r, r.coeff * t);
        }
    }
    (re, im)
}

#[cfg(test)]
mod tests {
    use super::super::basis::{sto3g_basis, Atom};
    use super::super::fermion::Sector;
    use super::super::integrals::compute_ao_integrals;
    use super::super::mo::active_space;
    use super::super::scf::rhf;
    use super::super::uccsd::{prepare_state, uccsd_excitations};
    use super::*;

    #[test]
    fn h2_circuit_has_the_expected_pauli_structure_and_cost() {
        let sector = Sector::new(2, 1, 1);
        let ex = uccsd_excitations(&sector);
        // Two single excitations (2 weight-3 terms each) and one double (8 weight-4 terms).
        let terms: Vec<usize> = ex.iter().map(|e| rotations(e).len()).collect();
        assert_eq!(terms, vec![2, 2, 8]);
        assert!(ex.iter().all(|e| rotations(e)
            .iter()
            .all(|r| (r.x & r.z).count_ones() % 2 == 1)));
        let cost = circuit_cost(&ex);
        assert_eq!(cost.rotations, 12);
        assert_eq!(cost.cnot, 2 * 2 * 4 + 8 * 6);
    }

    /// The compiled Pauli rotations prepare exactly the determinant-space UCCSD state.
    #[test]
    fn compiled_circuit_prepares_the_same_state_as_the_determinant_rotations() {
        let atoms = [
            Atom::angstrom(3, [0.0; 3]),
            Atom::angstrom(1, [0.0, 0.0, 1.595]),
        ];
        let basis = sto3g_basis(&atoms).unwrap();
        let ints = compute_ao_integrals(&basis, &atoms);
        let scf = rhf(&ints, 4).unwrap();
        let ham = active_space(&ints, &scf, 1, None).unwrap();
        let sector = Sector::new(ham.n_orb, 1, 1);
        let ex = uccsd_excitations(&sector);
        let theta: Vec<f64> = (0..ex.len())
            .map(|k| 0.05 * (k as f64 + 1.0).sin())
            .collect();

        let mut reference = vec![0.0; sector.dim()];
        reference[sector.hartree_fock()] = 1.0;
        let want = prepare_state(&ex, &reference, &theta);

        let hf_det = sector.dets[sector.hartree_fock()];
        let (re, im) = prepare_qubit_state(2 * ham.n_orb, hf_det, &ex, &theta);
        for (i, &det) in sector.dets.iter().enumerate() {
            assert!(
                (re[det as usize] - want[i]).abs() < 1e-10,
                "det {det:b}: {} vs {}",
                re[det as usize],
                want[i]
            );
            assert!(im[det as usize].abs() < 1e-10);
        }
        let norm: f64 = re.iter().zip(&im).map(|(a, b)| a * a + b * b).sum();
        assert!((norm - 1.0).abs() < 1e-12);
    }
}
