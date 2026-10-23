# Ab initio molecular VQE (`t1000::vqe`)

A variational quantum eigensolver whose Hamiltonian is computed from the molecular geometry, with
no tabulated energies and no training data. The physics supplies the objective: the energy of
the state is the loss.

```
geometry ─► STO-3G Gaussian integrals ─► restricted Hartree-Fock ─► frozen-core active space
         ─► second-quantised Hamiltonian ─┬─► Jordan-Wigner qubit Hamiltonian (Pauli terms)
                                          ├─► determinant-basis matrix ─► exact FCI (Lanczos)
                                          └─► UCCSD ansatz + BFGS ─► VQE energy
```

Run it:

```bash
cargo run --release -p t1000 --bin vqe_molecules                 # all nine molecules
cargo run --release -p t1000 --bin vqe_molecules LiH,H2O         # a selection
cargo run --release -p t1000 --bin vqe_molecules CH4 --active 6  # correlate 6 orbitals only
cargo run --release -p t1000 --bin vqe_dissociation              # H2 and LiH bond-stretch curves
```

## Modules

| Module | Role |
| :--- | :--- |
| `boys` | Boys function `F_n(x)` (series, downward recursion; asymptotic form above `x = 35`). |
| `basis` | Atoms and the STO-3G basis for H-F, with unit-normalised contracted functions. |
| `integrals` | McMurchie-Davidson overlap, kinetic, nuclear attraction and electron repulsion integrals. |
| `scf` | Closed-shell restricted Hartree-Fock with Pulay DIIS. |
| `mo` | Molecular-orbital integrals and the frozen-core active-space Hamiltonian. |
| `pauli` | Jordan-Wigner transform to a real Pauli-term Hamiltonian; term and weight statistics. |
| `fermion` | The fixed-`(N_α, N_β)` determinant sector and the sparse Hamiltonian in it. |
| `lanczos` | Ground-state solver: the exact (full CI) reference. |
| `uccsd` | UCCSD excitations as exact determinant-pair rotations, with analytic gradients. |
| `optimize` | BFGS with an Armijo line search. |
| `driver` | The pipeline and the serialisable `VqeReport`. |

The ansatz state is the one a Jordan-Wigner circuit of commuting Pauli rotations prepares: each
fermionic excitation generator `T - T†` rotates disjoint pairs of determinants, so its exponential
is exact and one pass over the pairs. Working in the particle-number sector (4,900 determinants for
CH₄ instead of 65,536 amplitudes) is what makes the full catalog run in seconds. The qubit
Hamiltonian is built independently and checked against the determinant matrix.

## Results (STO-3G, frozen core, exact simulation)

From `results/hosts/unknown-cpu/data/vqe_molecules_report.json`. Energies in hartree.

| Molecule | Qubits | Pauli terms | Parameters | E(HF) | E(FCI) | E(VQE) | Error (mHa) | Correlation recovered |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H₂ | 4 | 14 | 3 | -1.116684 | -1.137270 | -1.137270 | 0.0000 | 100.00 % |
| HeH⁺ | 4 | 26 | 3 | -2.841382 | -2.851024 | -2.851024 | 0.0000 | 100.00 % |
| LiH | 10 | 275 | 24 | -7.862024 | -7.882175 | -7.882175 | 0.0000 | 100.00 % |
| HF | 10 | 275 | 24 | -98.570758 | -98.596566 | -98.596566 | 0.0000 | 100.00 % |
| BeH₂ | 12 | 326 | 92 | -15.560312 | -15.594838 | -15.594470 | 0.3681 | 98.93 % |
| H₂O | 12 | 550 | 92 | -74.962928 | -75.012326 | -75.012225 | 0.1006 | 99.80 % |
| BH₃ | 14 | 1533 | 204 | -26.068996 | -26.121892 | -26.121527 | 0.3649 | 99.31 % |
| NH₃ | 14 | 2325 | 204 | -55.454046 | -55.518933 | -55.518759 | 0.1741 | 99.73 % |
| CH₄ | 16 | 4688 | 360 | -39.726810 | -39.805262 | -39.805077 | 0.1853 | 99.76 % |

All nine are within chemical accuracy (1.594 mHa) of the exact energy. UCCSD is exact when only
two electrons are correlated (H₂, HeH⁺, LiH with its frozen core) and matches to the printed precision
for HF; the remaining 0.1-0.4 mHa in the larger molecules is the truncation of UCCSD at doubles.

Bond stretching (`vqe_dissociation`) is where Hartree-Fock fails. For H₂ the restricted
Hartree-Fock error grows from 10 mHa at 0.4 Å to 278 mHa at 3.0 Å, while the VQE stays on the exact
curve to better than 1e-4 mHa. The exact energy at 6 Å equals twice the STO-3G hydrogen-atom
energy (-0.46658 hartree) to 5e-5, an independent physical check of the whole stack
(`tests/vqe_molecules.rs`).

## How it is validated

* **Integrals.** The H₂ overlap, kinetic and two-electron integrals match the Szabo & Ostlund
  tables to 4 decimals; the Boys function matches Simpson quadrature across both of its regimes;
  a one-centre closed form checks the nuclear attraction; p-function ERIs have the full eight-fold
  symmetry and the integrals are unchanged by rotating and translating the molecule.
* **Hartree-Fock.** H₂ (-1.1167, orbital energies -0.5782 and 0.6703) reproduces Szabo & Ostlund,
  helium matches -2.80778, and the energies of all nine molecules agree with the published STO-3G
  values to 1-3 mHa (geometry-dependent); water is identical under rigid motion.
* **Active space.** The frozen-core and reduced-space determinant energies equal the full SCF energy.
* **Qubit Hamiltonian.** The Jordan-Wigner ladder operators obey the canonical anticommutation
  relations, and every matrix element of the LiH qubit Hamiltonian between determinants equals the
  Slater-Condon matrix to 1e-10, as does its expectation value on a random state.
* **Solver.** Lanczos reproduces the analytic Hubbard-dimer energy and a dense diagonalisation; the
  UCCSD gradient equals central finite differences to 1e-7; Brillouin's theorem holds (the singles
  gradient vanishes at the Hartree-Fock point); BFGS minimises Rosenbrock to 1e-6.

## Limits

* **Exact simulation.** There is no shot noise, gate noise or measurement error; this is the
  noiseless ideal a device would approach. `t1000::device_noise` and the Cepheus calibration are the
  natural next step for noisy VQE.
* **Basis and reference.** STO-3G only (H-F), closed-shell restricted Hartree-Fock, at most 16 spatial
  orbitals (32-bit determinants). Open-shell species and larger basis sets need an unrestricted
  reference and more primitives.
* **SCF at long bond lengths.** Restricted Hartree-Fock with DIIS does not converge for LiH beyond
  about 3.5 Å (`vqe_dissociation` reports the points it skips).
* **UCCSD, not FCI.** The ansatz stops at double excitations, so strongly correlated cases will not
  reach chemical accuracy; the exact reference makes that visible.
* **Hardware cost.** The Pauli-term counts above are what a QPU would have to measure per energy
  estimate; a circuit-level compilation of the UCCSD exponentials is not included.

The earlier `chemistry` module keeps tabulated energies for the same nine species at a basis and
geometry that are not stated; they do not agree with these STO-3G values (for example CH₄ -40.143 against
-39.727 hartree at Hartree-Fock level). See `docs/CLAIMS.md`.
