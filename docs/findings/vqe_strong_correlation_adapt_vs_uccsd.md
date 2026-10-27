<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Terrence A. Jones SR. All rights reserved. -->

# T-1000 Scientific Finding: Strong Correlation in Molecular VQE - ADAPT-VQE, UCCSD and Classical Baselines Against Exact FCI

**Document ID:** `T1000-FINDING-2026-042`
**Deen Monograph:** `M168`
**Author:** Terrence A. Jones SR
**Date:** 2026-10-26
**Classification:** Ab initio quantum chemistry / variational quantum eigensolvers
**Canonical Repository:** `https://github.com/tariqselassie-hub/T1000`

---

## 1. Question

UCCSD-VQE reaches chemical accuracy for all nine equilibrium molecules in `docs/vqe.md`. Stretching bonds makes a single Hartree-Fock determinant a poor reference (strong correlation), and that is where both the standard quantum ansatz (UCCSD) and the standard classical methods (MP2, CISD, CCSD) are expected to fail. Where exactly does each method lose chemical accuracy (1.594 mHa), and does ADAPT-VQE, which grows its circuit operator by operator, hold up there and at what circuit cost?

Every method runs in the same STO-3G frozen-core active space and is judged against exact full CI from the same Hamiltonian. Nothing is fitted to the exact energy: ADAPT stops on its own pool-gradient norm.

## 2. Systems and methods

| System | Coordinate | Points | Qubits | Determinants |
| :--- | :--- | ---: | ---: | ---: |
| H₄ linear chain | uniform H-H spacing 0.6-3.0 Å | 13 | 8 | 36 |
| H₄ rectangle, a = 1.0 Å | side b 0.7-1.3 Å (square at b = 1.0) | 13 | 8 | 36 |
| H₆ linear chain | uniform H-H spacing 0.6-3.0 Å | 13 | 12 | 400 |
| N₂ (2 core orbitals frozen) | N-N distance 0.8-3.0 Å | 12 | 16 | 3,136 |

* **Classical (`t1000::vqe::classical`):** MP2 from canonical orbital energies; CISD as the lowest eigenvalue in the singles-and-doubles determinant space; CCSD solved exactly in the determinant basis (`e^T = Π(1 + t_k T_k)`, Jacobi updates with DIIS) over the same excitation list as UCCSD. CCSD is projective and not variational; a run whose amplitude equations do not converge is reported as a failure, not as a number.
* **UCCSD-VQE:** all occupied-to-virtual singles and doubles, BFGS with exact gradients, started from zero and from the previous point's angles (the lower variational energy is kept, so a failure is the ansatz and not a bad start).
* **ADAPT-VQE (`t1000::vqe::adapt`, Grimsley et al. 2019):** pool = the UCCSD singles and doubles; each step appends the operator with the largest `|∂E/∂θ|` (repeats allowed) and re-optimises all angles; stops when the pool-gradient norm is below 1e-3 (the original paper's threshold) or at 250 operators.
* **Circuit cost:** CNOTs of each excitation compiled naively as Jordan-Wigner Pauli rotations, `2(w-1)` per weight-`w` string (`t1000::vqe::circuit`); real compilers do better, but the ratio between ansätze is what matters here.

## 3. Results

From `results/hosts/unknown-cpu/data/vqe_strong_correlation_report.json` (ADAPT tolerance 1e-3). "Within" counts points inside chemical accuracy; n/c counts CCSD runs whose equations did not converge (excluded from the maximum).

| System | HF max err | MP2 max err | CISD max / within | CCSD max / within / n/c | UCCSD max / within | ADAPT max / within |
| :--- | ---: | ---: | :---: | :---: | :---: | :---: |
| H₄ chain | 554 mHa | 175 mHa | 43.1 / 3 of 13 | 30.6 / 5 of 13 / 0 | 5.15 / 7 of 13 | **2.03 / 12 of 13** |
| H₄ rectangle | 154 | 95.3 | 4.64 / 3 of 13 | 1.12 / 8 of 13 / 5 | **1.09 / 13 of 13** | **1.09 / 13 of 13** |
| H₆ chain | 830 | 266 | 281 / 1 of 13 | 314 / 5 of 13 / 3 | 14.8 / 4 of 13 | **5.92 / 11 of 13** |
| N₂ | 650 | 426 | 314 / 0 of 12 | 12.7 / 1 of 12 / 8 | 6.88 / 4 of 12 | 84.5 / 4 of 12 |
| N₂, ADAPT tol 1e-4 | | | | | | 82.6 / **9 of 12** |

The N₂ row with tolerance 1e-4 is `vqe_strong_correlation_report.84aad6c6.json` (same run otherwise).

### 3.1 What fails where

* **CCSD breaks first and worst.** Because it is not variational, at stretched geometries it falls *below* the exact energy (H₄ chain: -30.6 mHa at 2.6 Å; H₆: -314 mHa at 2.6 Å) or its equations stop converging (5 of 13 rectangle points around the square, 8 of 12 N₂ points). It reaches chemical accuracy only near equilibrium.
* **CISD is variational but not size-extensive:** for H₆ and N₂ it never gets inside chemical accuracy beyond the shortest bonds (281 and 314 mHa at the stretched end).
* **UCCSD degrades smoothly and stays an upper bound,** but only near equilibrium is it within chemical accuracy (H₄ chain 7 of 13, H₆ 4 of 13, N₂ 4 of 12). On the H₄ chain its error peaks at 5.15 mHa around 2.2-2.4 Å and falls again toward dissociation.
* **ADAPT-VQE is the most accurate method on the hydrogen systems:** H₄ chain 12 of 13 (the miss is 2.03 mHa at 1.6 Å, where the 1e-3 stopping rule ended the growth early), the rectangle 13 of 13, H₆ 11 of 13.
* **The square H₄ is benign for the variational methods here** (both 1.09 mHa at the square) and the worst case for CCSD, which does not converge across the crossing.

### 3.2 Circuit cost

ADAPT is not uniformly cheaper. Near equilibrium it needs about half of UCCSD's parameters and CNOTs (H₄ chain 14 vs 26 parameters, 736 vs 1,312 CNOTs; H₆ 59 vs 117 and 4,560 vs 9,024; N₂ 53 vs 315 and 5,296 vs 31,760). In the strongly correlated region of H₆ it grows *past* UCCSD (up to 161 operators and 12,704 CNOTs at 1.8 Å) because it repeats operators, and that is what buys its accuracy there. For N₂ at tolerance 1e-4 the accurate middle of the curve costs 150-250 operators (15,040-25,536 CNOTs), still below UCCSD's 31,760.

### 3.3 N₂: a stopping rule and a genuine trap

With the default tolerance ADAPT reaches chemical accuracy at only 4 of 12 N₂ points. Two different things are going on:

1. **The 1e-3 stopping rule is too loose for N₂.** At 1e-4 ADAPT is inside chemical accuracy at 9 of 12 points, with errors of 0.04-0.11 mHa from 1.0 to 2.0 Å.
2. **At 2.8 and 3.0 Å ADAPT is trapped.** It stops at 82.4-82.6 mHa with 46 operators even at tolerance 1e-4: every operator in the pool has a gradient below 1e-4 at a state 82 mHa above the exact energy. This is a gradient trough of the occupied-to-virtual pool on that Hartree-Fock reference, not an early stop; UCCSD from zero and warm starts reaches 0.26-0.62 mHa at the same points.

The trap depends on the reference. Beyond 2.2 Å the lowest-energy restricted Hartree-Fock solution is a different SCF state from the one obtained by following the orbitals out from equilibrium (its Hartree-Fock energy is lower). In an earlier, unrecorded run that kept the followed state, ADAPT stayed at 0.2-0.7 mHa out to 3.0 Å. The lower-energy SCF state is therefore not automatically the better reference for a correlated method. Remedies that this study does not test: a generalised (all-orbital) pool, qubit-excitation pools, or restarting ADAPT from a different reference when the pool gradient vanishes far above a variational estimate.

## 4. Infrastructure fixed on the way (M1)

* **Hartree-Fock at stretched bonds** (`t1000::vqe::scf::rhf_with`): a starting guess from the previous geometry, and fallbacks with a virtual-orbital level shift and density damping when DIIS oscillates. LiH, which failed beyond 3.5 Å, now converges to 5.0 Å (`vqe_dissociation`); its FCI energy approaches the sum of the separated STO-3G atoms.
* **Level crossings:** following orbitals across the H₄ square left the SCF on a higher state (HF error 525 mHa instead of about 87 mHa at b = 1.3 Å, and every correlated method inherited the bad reference). A seeded SCF now also runs a cold start and keeps the lower converged energy; `scf::tests::a_seed_from_across_a_level_crossing_does_not_trap_the_scf` checks that the trap is real and avoided.
* **Eigensolver:** the Jacobi solver demanded an off-diagonal norm of 1e-15 of the matrix scale and failed for H₆ at short spacing when rounding stalled just above it; a stalled result within 1e-11 of the scale is now accepted.

## 5. Limits

* STO-3G minimal basis and frozen core: the FCI reference is exact only within this active space.
* Exact state-vector simulation: no shot noise or gate error. The CNOT counts are what a device would have to run.
* One ADAPT pool (fermionic occupied-to-virtual singles and doubles). The N₂ trap may not occur with other pools.
* CCSD is solved in the determinant basis, which is exact for these sizes but scales exponentially; it is a reference implementation, not a production CCSD.

## 6. Reproduce

```bash
cargo run --release -p t1000 --bin vqe_strong_correlation                      # all systems, ADAPT tol 1e-3
cargo run --release -p t1000 --bin vqe_strong_correlation N2 --adapt-tol 1e-4  # the N2 tolerance study
cargo run --release -p t1000 --bin vqe_dissociation                            # H2 and LiH to 5 Å
cargo test --release -p t1000 --lib vqe::                                      # 64 unit tests
```

Add `--record` to save; reports go to `results/hosts/<host>/data/`.
