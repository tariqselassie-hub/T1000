// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Atomic Baseline Decomposition & Chemical Binding Energy Decoder.
//!
//! Separates extensive atomic nuclear core energies from molecular binding energies,
//! allowing bounded VQC Pauli expectation values <Z_q> in [-1, +1] to decode
//! ab initio ground-state energies with numerical stability and zero gradient explosion.

use serde::{Deserialize, Serialize};

/// Standard CODATA conversion constant from Hartrees to electron-volts.
pub const HARTREE_TO_EV: f64 = 27.211386245988;

/// Standard conversion from Hartrees to kilocalories per mole.
pub const HARTREE_TO_KCAL_MOL: f64 = 627.509474;

/// Standard B3LYP/6-31G(2df,p) single-atom electronic ground-state reference energies (Hartrees).
/// Sourced from standard QM9 ab initio benchmark specifications (Ramakrishnan et al., 2014).
pub const B3LYP_E_ATOM_H: f64 = -0.500272782;
pub const B3LYP_E_ATOM_C: f64 = -37.846772;
pub const B3LYP_E_ATOM_N: f64 = -54.583861;
pub const B3LYP_E_ATOM_O: f64 = -75.064579;
pub const B3LYP_E_ATOM_F: f64 = -99.718730;

/// Atomic baseline reference lookup table for standard chemical elements.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AtomicReferenceTable;

impl AtomicReferenceTable {
    /// Returns the B3LYP/6-31G(2df,p) single-atom reference energy (in Hartrees) by atomic number Z.
    #[inline(always)]
    pub fn energy_by_z(z: usize) -> f64 {
        match z {
            1 => B3LYP_E_ATOM_H,
            6 => B3LYP_E_ATOM_C,
            7 => B3LYP_E_ATOM_N,
            8 => B3LYP_E_ATOM_O,
            9 => B3LYP_E_ATOM_F,
            // Fallback empirical screening approximation: E(Z) ≈ -0.5 * Z^2.4
            other => -0.5 * (other as f64).powf(2.4),
        }
    }

    /// Returns the reference energy by element symbol string.
    #[inline(always)]
    pub fn energy_by_symbol(symbol: &str) -> f64 {
        match symbol.trim().to_uppercase().as_str() {
            "H" => B3LYP_E_ATOM_H,
            "C" => B3LYP_E_ATOM_C,
            "N" => B3LYP_E_ATOM_N,
            "O" => B3LYP_E_ATOM_O,
            "F" => B3LYP_E_ATOM_F,
            _ => 0.0,
        }
    }

    /// Converts element symbol to atomic number Z.
    #[inline(always)]
    pub fn symbol_to_z(symbol: &str) -> usize {
        match symbol.trim().to_uppercase().as_str() {
            "H" => 1,
            "HE" => 2,
            "LI" => 3,
            "BE" => 4,
            "B" => 5,
            "C" => 6,
            "N" => 7,
            "O" => 8,
            "F" => 9,
            "NE" => 10,
            "NA" => 11,
            "MG" => 12,
            "AL" => 13,
            "SI" => 14,
            "P" => 15,
            "S" => 16,
            "CL" => 17,
            "AR" => 18,
            "BR" => 35,
            "I" => 53,
            _ => 6,
        }
    }
}

/// High-precision atomic baseline decomposition engine.
/// Decouples extensive core energy scale from intensive chemical covalent binding energy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AtomicBaselineDecoder {
    /// Learnable or calibrated scaling coefficient for Coulomb trace baseline.
    pub coulomb_scale: f64,
}

impl Default for AtomicBaselineDecoder {
    fn default() -> Self {
        Self { coulomb_scale: 1.0 }
    }
}

impl AtomicBaselineDecoder {
    /// Creates a new baseline decoder with default calibration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Computes the sum of non-interacting free atomic ground-state energies (Hartrees)
    /// for a collection of atomic numbers.
    #[inline]
    pub fn compute_atomic_sum_by_z(z_numbers: &[usize]) -> f64 {
        let mut total = 0.0;
        for &z in z_numbers {
            if z > 0 {
                total += AtomicReferenceTable::energy_by_z(z);
            }
        }
        total
    }

    /// Computes the sum of free atomic ground-state energies (Hartrees) from element symbols.
    #[inline]
    pub fn compute_atomic_sum_by_symbols(symbols: &[&str]) -> f64 {
        let mut total = 0.0;
        for &sym in symbols {
            total += AtomicReferenceTable::energy_by_symbol(sym);
        }
        total
    }

    /// Computes the true atomization / binding energy (Hartrees):
    /// ΔU_0 = U_0 - ∑ E_atom(Z_i)
    /// By physical convention, atomization energy is negative (bound state is lower in energy than free atoms).
    #[inline(always)]
    pub fn compute_atomization_energy(u0: f64, z_numbers: &[usize]) -> f64 {
        let e_atom_sum = Self::compute_atomic_sum_by_z(z_numbers);
        u0 - e_atom_sum
    }

    /// Reconstructs the total electronic ground-state energy U_0 (Hartrees)
    /// from a VQC-decoded atomization energy ΔU_0 and the atomic composition.
    #[inline(always)]
    pub fn reconstruct_total_energy(atomization_energy: f64, z_numbers: &[usize]) -> f64 {
        let e_atom_sum = Self::compute_atomic_sum_by_z(z_numbers);
        e_atom_sum + atomization_energy
    }

    /// Computes Coulomb trace baseline proxy energy from sum of diagonal self-energies.
    /// E_base ≈ -1.026 * Tr(C).
    #[inline(always)]
    pub fn coulomb_trace_proxy_energy(&self, trace: f64) -> f64 {
        -1.0263 * self.coulomb_scale * trace
    }

    /// Converts energy in Hartrees to electron-volts.
    #[inline(always)]
    pub fn hartree_to_ev(hartree: f64) -> f64 {
        hartree * HARTREE_TO_EV
    }

    /// Converts energy in Hartrees to kcal/mol.
    #[inline(always)]
    pub fn hartree_to_kcal_mol(hartree: f64) -> f64 {
        hartree * HARTREE_TO_KCAL_MOL
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_methane_atomization_energy() {
        // Methane CH4: 1 Carbon + 4 Hydrogens
        // Total QM9 DFT U_0 = -40.47893 Ha
        let u0_methane = -40.47893;
        let z_methane = [6, 1, 1, 1, 1];

        let e_atomic = AtomicBaselineDecoder::compute_atomic_sum_by_z(&z_methane);
        let expected_atomic = B3LYP_E_ATOM_C + 4.0 * B3LYP_E_ATOM_H;
        assert!((e_atomic - expected_atomic).abs() < 1e-6);

        let atomization = AtomicBaselineDecoder::compute_atomization_energy(u0_methane, &z_methane);
        // Atomization energy should be approximately -0.63107 Ha (~ -17.17 eV)
        assert!((atomization - (-0.631065)).abs() < 1e-4);

        let atomization_ev = AtomicBaselineDecoder::hartree_to_ev(atomization);
        assert!((atomization_ev - (-17.172)).abs() < 0.05);

        // Round-trip reconstruction
        let reconstructed =
            AtomicBaselineDecoder::reconstruct_total_energy(atomization, &z_methane);
        assert!((reconstructed - u0_methane).abs() < 1e-12);
    }

    #[test]
    fn test_water_atomization_energy() {
        // Water H2O: 1 Oxygen + 2 Hydrogens
        // Total QM9 DFT U_0 = -76.39327 Ha
        let u0_water = -76.39327;
        let z_water = [8, 1, 1];

        let atomization = AtomicBaselineDecoder::compute_atomization_energy(u0_water, &z_water);
        // Atomization energy should be approximately -0.32814 Ha (~ -8.93 eV)
        let atomization_ev = AtomicBaselineDecoder::hartree_to_ev(atomization);
        assert!((atomization_ev - (-8.929)).abs() < 0.05);

        let reconstructed = AtomicBaselineDecoder::reconstruct_total_energy(atomization, &z_water);
        assert!((reconstructed - u0_water).abs() < 1e-12);
    }
}
