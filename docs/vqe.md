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
| `driver` | The pipeline and the serialisable `VqeReport` (`solve_detailed` keeps the intermediates). |
| `circuit` | UCCSD as Pauli rotations, CNOT / one-qubit gate counts, and a qubit-state simulator that checks them. |
| `grouping` | Qubit-wise commuting measurement groups and their variances (shot budgets with grouping). |
| `noise` | Depolarising gate error, readout error, shot noise, mitigation, shot budgets, tolerable error rates. |
| `noisy_vqe` | SPSA driven only by sampled, attenuated energies. |

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

## Noise, shots and gate counts (`vqe_noise`)

```bash
cargo run --release -p t1000 --bin vqe_noise
```

`circuit` compiles each UCCSD exponential to commuting Pauli rotations (2 terms per single
excitation, 8 per double; the compiled rotations are checked to prepare exactly the determinant-space
state). A weight-`w` rotation costs `2(w-1)` CNOTs, so the counts below are **naive upper bounds**:
no CNOT cancellation, no qubit tapering, no pruning of small amplitudes.

`noise` then applies a global depolarising channel with circuit fidelity `F = (1-p₁)^{n₁}(1-p₂)^{n₂}`
(which leaves only the identity coefficient `c₀`: `E_noisy = c₀ + F(E - c₀)`), a symmetric readout
flip `e` per qubit (a weight-`w` term shrinks by `(1-2e)^w`), and binomial shot noise. Known-noise
mitigation divides each term by its attenuation: the bias disappears in expectation and returns as
variance. The reference profile `cepheus intra-die` uses the 0.80 % intra-die two-qubit error and
≈2.5 % readout error from the Cepheus run (README § 22); its 0.1 % single-qubit error is an assumption.

From `results/hosts/unknown-cpu/data/vqe_noise_report.json`:

| Molecule | Qubits | Pauli terms | CNOTs | 1-qubit gates | Shots for 1.59 mHa (ideal device) | Max p₂ for 1.59 mHa bias |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| H₂ | 4 | 14 | 64 | 92 | 4.9e4 | 2.1e-5 |
| HeH⁺ | 4 | 26 | 64 | 92 | 2.8e5 | 1.7e-5 |
| LiH | 10 | 275 | 1,376 | 1,232 | 3.2e6 | 4.9e-7 |
| HF | 10 | 275 | 1,376 | 1,232 | 3.1e7 | 1.6e-7 |
| BeH₂ | 12 | 326 | 6,976 | 5,632 | 1.3e7 | 7.0e-8 |
| H₂O | 12 | 550 | 6,976 | 5,632 | 6.3e7 | 4.3e-8 |
| BH₃ | 14 | 1,533 | 18,208 | 13,200 | 7.5e7 | 2.2e-8 |
| NH₃ | 14 | 2,325 | 18,208 | 13,200 | 1.4e8 | 1.9e-8 |
| CH₄ | 16 | 4,688 | 36,480 | 23,936 | 2.6e8 | 9.1e-9 |

Shots are the total over all terms with the variance-optimal split and no commuting-term grouping,
so they overstate what grouped measurement needs. The next section measures by how much.

### Measurement grouping (`vqe_grouping`)

```bash
cargo run --release -p t1000 --bin vqe_grouping
```

Terms that agree on every qubit they share (qubit-wise commuting) are read out together from one
product-basis measurement. `grouping` builds the groups greedily (largest coefficients first) and
computes each group's exact single-shot variance on the converged UCCSD state, covariances between
its terms included; the total for standard error `ε` with the variance-optimal split is
`(Σ_G σ_G)² / ε²`. The sum of group means plus `c₀` reproduces the energy to 1e-13, so the grouped
estimator is unbiased. From `results/hosts/unknown-cpu/data/vqe_grouping_report.json`:

| Molecule | Pauli terms | Groups | Shots, term by term | Shots, grouped | Reduction |
| :--- | ---: | ---: | ---: | ---: | ---: |
| H₂ | 14 | 5 | 4.9e4 | 4.9e4 | 1.0x |
| HeH⁺ | 26 | 9 | 2.8e5 | 4.1e4 | 6.9x |
| LiH | 275 | 83 | 3.2e6 | 2.3e5 | 14.2x |
| HF | 275 | 75 | 3.1e7 | 1.2e6 | 25.3x |
| BeH₂ | 326 | 107 | 1.3e7 | 1.7e6 | 7.3x |
| H₂O | 550 | 187 | 6.3e7 | 5.0e6 | 12.7x |
| BH₃ | 1,533 | 547 | 7.5e7 | 1.2e7 | 6.3x |
| NH₃ | 2,325 | 869 | 1.4e8 | 2.1e7 | 7.0x |
| CH₄ | 4,688 | 1,569 | 2.6e8 | 3.5e7 | 7.4x |

Grouping cuts the ideal-device shot budget by 6x to 25x (H₂ gains nothing). The grouping is greedy
qubit-wise; general commuting groups or classical shadows could cut further, at the price of deeper
measurement circuits. The shot counts are for the ideal device; they are not changed by gate error.

What it says:

* **Gate error, not shots, is the barrier.** At the Cepheus profile the circuit fidelity is 0.55 for
  H₂ and 5e-6 for LiH, giving energy errors of 509 mHa and 2,149 mHa; beyond LiH it is zero to many
  digits. Mitigating that attenuation needs more than 1e18 shots for LiH, so mitigation does not rescue it.
* **The tolerable error is far below today's devices**, but this is a worst case: global depolarising
  mixes toward the maximally mixed state of the *full* `2^n` space, whose energy `c₀` is far above the
  ground state, and the circuits are uncompiled. Tapering the Z₂ symmetries, dropping tiny
  excitations and cancelling CNOT ladders can cut the gate count substantially, and
  symmetry verification (post-selecting on particle number and spin) removes much of the leakage
  into wrong sectors. None of that is modelled here.
* **Shot noise alone is mild.** SPSA on sampled energies finds good parameters: H₂ ends within
  0.26 mHa of FCI with 100 shots per term and 0.001 mHa with 1e5; HeH⁺ within 0.11 mHa at 1,000 and
  0.005 mHa at 1e5; LiH, with 24 parameters, needs about 1e5 shots per term (0.65 mHa; 68 mHa at 1,000).
* **Depolarising noise rescales the landscape but does not move its minimum**, so an optimiser on a
  noisy device still finds good parameters; what is wrong is the energy it *reports*. The SPSA errors
  above are the exact energies of the returned parameters, not the device's own readings.

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
