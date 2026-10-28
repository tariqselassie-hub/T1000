# Google Quantum AI — Verifiable Quantum Circuit Compilation
## Competition Architecture, Mathematical Specifications & Optimization Engineering

SPDX-License-Identifier: MIT OR Apache-2.0  
Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

---

### Executive Overview

This repository directory contains the mathematical models, simulation pipelines, circuit synthesis engines, and submission artifacts for the **Google Quantum AI - Verifiable Quantum Circuit Compilation** challenge on Kaggle (arXiv:2610.23832).

The competition evaluates the fault-tolerant compilation of three bottleneck quantum subroutines constrained to verifiable structures (reversible classical logic, Clifford+T, arbitrary single-qubit rotations, and measurement-based uncomputation via the `kickmix` simulator).

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 COMPETITION TAXONOMY                                   │
├───────────────┬─────────────────────────────┬──────────────────────────┬───────────────┤
│ Benchmark     │ Subroutine Target           │ Mathematical Object      │ Metric Model  │
├───────────────┼─────────────────────────────┼──────────────────────────┼───────────────┤
│ Benchmark I   │ Coulomb Potential Phasing   │ 1 / sqrt(x), x in [1,M]  │ Q * C         │
│ Benchmark II  │ Decoded Quantum Interf.     │ Reed-Solomon [70,10,61]  │ R(eps+) * Q^2*T│
│ Benchmark III │ FeMoCo Hamiltonian Block Enc│ Givens QROM Angles (75)  │ lambda*T_tot*Q│
└───────────────┴─────────────────────────────┴──────────────────────────┴───────────────┘
```

The overall tournament score is the geometric mean of speedup over baseline across all three benchmarks:
$$\text{Score} = \left( \frac{\text{Baseline}_I}{\text{Candidate}_I} \times \frac{\text{Baseline}_{II}}{\text{Candidate}_{II}} \times \frac{\text{Baseline}_{III}}{\text{Candidate}_{III}} \right)^{1/3}$$

---

### Submission & Tournament Ledger

| Submission Iteration | Benchmark I (`invsqrt`) | Benchmark II (`dqi`) | Benchmark III (`dfthcsos`) | Tournament Score | Status |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **V1 (Reference Baseline)** | $Q=260$, $C=7950.2$ ($2,067,055$) | $Q=1893$, $T=5.725\times 10^6$ | $T=93237$, $Q=730$ ($3.356\times 10^9$) | **1.00000** | Reference Base |
| **V7 (Certified Cand4)** | $Q=258$, $C=7671.5$ ($1,979,259$) | Baseline ($886\text{ MB}$) | Baseline ($79,480\text{ Toff}$) | **1.01444** | **VERIFIED (Public LB)** |
| **V8 (Certified Cand5)** | $Q=255$, $C=7470.4$ ($1,904,940$) | Baseline ($886\text{ MB}$) | Baseline ($79,480\text{ Toff}$) | **1.02760** | **IN EVALUATION** |
| **Target (Quad-B1 + J9)** | $Q=238$, $C=6770.0$ ($1,611,260$) | Baseline | $T=85289$, $Q=742$ ($3.120\times 10^9$) | **1.11320** | Development |

---

### Benchmark Mathematical Contracts & Attack Vectors

#### 1. Benchmark I: `invsqrt` (Coulomb Potential Phasing)
- **Target Unitary:** $|x\rangle|0\rangle \to \exp(2\pi i / \sqrt{x}) |x\rangle|0\rangle$ for all $x \in [1, 3133452]$ on a 22-qubit register `x`.
- **Tolerance Requirement:** Phase error must satisfy $| \Delta \theta(x) | \le 2^{-24} / \sqrt{x}$ turns mod 1.
- **Cost Formula:** $\text{Score}_I = Q \times \left( T_{\text{offoli}} + T_{\text{gate}} / 4 \right)$.
- **Status & Results:**
  - V7: Passed edge test $x = 2,684,415$ with $33\%$ margin ($2.444 \times 10^{-11} < 3.638 \times 10^{-11}$).
  - V8 (Cand5): Qubits dropped to $255$, Toffolis to $7,470.4$, edge error $1.026 \times 10^{-12}$ ($35\times$ margin).
  - Quadratic Breakthrough: Proved degree-2 polynomial on 4 sub-intervals meets $2^{-24}$ tolerance, permanently eliminating $c_3$ (17 qubits) and Horner step 1 (~700 Toffolis).
- **Full Documentation:** [docs/benchmark_1_invsqrt.md](docs/benchmark_1_invsqrt.md)

#### 2. Benchmark II: `dqi` (Decoded Quantum Interferometry)
- **Target Code:** Reed-Solomon $[70, 10, 61]$ over $\mathbb{F}_{2^{12}}$ correcting up to $l = 35$ errors.
- **Contract:** Input syndromes $S_0 \dots S_{69}$ and classical MBUC registers $E_0 \dots E_{4094}$. Applies phase $(-1)^{\sum_j \text{popcount}(e_j \wedge M_j)}$ and restores all syndrome qubits.
- **Cost Formula:** $\text{Score}_{II} = R(\epsilon_+) \times Q^2 \times T$.
- **Discovery:**
  - The $Q^2$ quadratic scaling makes qubit trimming $2\times$ to $4\times$ more impactful than Toffoli pruning.
  - Proved that synchronized EEA contains an over-provisioned iteration margin (e.g., $n=4$ reduces from 24 to 21 iters with zero errors, saving $12.3\%$ Toffolis).
- **Full Documentation:** [docs/benchmark_2_dqi.md](docs/benchmark_2_dqi.md)

#### 3. Benchmark III: `dfthcsos` (Electronic Structure Block Encoding)
- **Target Hamiltonian:** FeMoCo Active Space ($N = 76$ spatial orbitals, $\eta = 113$ electrons).
- **Contract:** Family of $K = \lceil 75 / J \rceil$ circuits loading $75$ Givens rotation angle words across $3,976$ vector addresses.
- **Cost Formula:** $\text{Score}_{III} = \lambda_{\text{eff}} \times T_{\text{total}} \times Q_{\text{total}}$.
- **Discovery:**
  - $85.2\%$ of Toffolis ($79,480$ of $93,237$) come from the 10 QROM tables.
  - Setting $J = 9$ reduces circuits from 10 to 9, saving $7,948$ Toffolis.
  - With an 8-ancilla unary iteration tree, total circuit qubits is $12 + 108 + 8 = 128 \le 128$ (`QUBIT_CAP`).
  - Factorization $\lambda_{\text{eff}}$ can be reduced from 49.3 towards the 19.19 theoretical floor ($2.56\times$ ceiling).
- **Full Documentation:** [docs/benchmark_3_dfthcsos.md](docs/benchmark_3_dfthcsos.md)

---

### Directory Architecture

```
quantum-contest/
├── README.md                           # Master specification, tournament ledger & taxonomy
├── notebooks/
│   └── kaggle_certified_pipeline.py    # Self-contained Kaggle submission notebook code
└── docs/
    ├── benchmark_1_invsqrt.md          # Numerical analysis, precision tuning & degree-2 proof
    ├── benchmark_2_dqi.md              # Reed-Solomon MBUC decoder & Q^2 quadratic scaling
    ├── benchmark_3_dfthcsos.md         # QROM unary segment tree analysis & J=9 architecture
    └── kaggle_skill.md                 # Kaggle operational handbook & platform protocols
```
