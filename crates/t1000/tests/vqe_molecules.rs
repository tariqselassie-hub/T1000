//! End-to-end checks of the ab initio VQE against known physics.

use t1000::vqe::driver::{solve, VqeConfig};
use t1000::vqe::molecules::{by_formula, catalog, diatomic};

/// STO-3G energy of a hydrogen atom, hartree.
const E_H_ATOM: f64 = -0.466_581_85;

#[test]
fn h2_dissociates_to_two_hydrogen_atoms_where_hartree_fock_fails() {
    let cfg = VqeConfig::default();
    let far = solve(&diatomic("H2", "H2", [1, 1], 6.0, 0), &cfg).unwrap();
    // Exact: twice the atomic energy. Restricted Hartree-Fock: far above it.
    assert!(
        (far.fci_energy - 2.0 * E_H_ATOM).abs() < 5e-5,
        "FCI {}",
        far.fci_energy
    );
    assert!(far.vqe_error_vs_fci.abs() < 1e-7);
    assert!(
        far.hf_energy - far.fci_energy > 0.2,
        "RHF dissociation error"
    );
}

#[test]
fn vqe_never_falls_below_the_exact_energy_and_never_rises_above_hartree_fock() {
    for formula in ["H2", "HeH+", "LiH", "H2O"] {
        let r = solve(&by_formula(formula).unwrap(), &VqeConfig::default()).unwrap();
        assert!(
            r.vqe_energy >= r.fci_energy - 1e-9,
            "{formula} below the variational bound"
        );
        assert!(
            r.vqe_energy <= r.hf_energy + 1e-9,
            "{formula} above Hartree-Fock"
        );
        assert!(
            r.within_chemical_accuracy,
            "{formula}: {} mHa",
            r.vqe_error_vs_fci * 1e3
        );
    }
}

#[test]
fn freezing_the_core_changes_the_energy_by_a_small_correlation_contribution_only() {
    let molecule = by_formula("LiH").unwrap();
    let frozen = solve(&molecule, &VqeConfig::default()).unwrap();
    let all = solve(
        &molecule,
        &VqeConfig {
            freeze_core: false,
            ..VqeConfig::default()
        },
    )
    .unwrap();
    assert_eq!((frozen.n_qubits, all.n_qubits), (10, 12));
    assert!(
        (frozen.hf_energy - all.hf_energy).abs() < 1e-9,
        "same HF energy either way"
    );
    // Core correlation lowers the full-space FCI energy by a few millihartree at most.
    let shift = frozen.fci_energy - all.fci_energy;
    assert!(shift > 0.0 && shift < 5e-3, "core correlation {shift}");
}

#[test]
fn every_catalog_molecule_runs_and_reaches_chemical_accuracy() {
    // The two largest (BH3, CH4) take a few seconds in debug builds.
    for molecule in catalog() {
        let r = solve(&molecule, &VqeConfig::default()).unwrap();
        assert!(r.converged, "{} did not converge", r.formula);
        assert!(
            r.within_chemical_accuracy,
            "{}: {} mHa",
            r.formula,
            r.vqe_error_vs_fci * 1e3
        );
        assert!(r.correlation_recovered_pct > 98.0, "{}", r.formula);
    }
}
