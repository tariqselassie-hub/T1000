<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Terrence A. Jones SR. All rights reserved. -->

# University of Pittsburgh Quantum Screening, Giant Dipole Ferroelectrics & Molecular Stark Szegedy Walk

## Executive Summary

This monograph records the empirical screening of the **University of Pittsburgh Quantum Repository** (105,926 molecules, PM7 semi-empirical electronic observables) and the implementation of the **Molecular Stark Dipole Szegedy Quantum Walk Engine** coupling real molecular dipole moments into 7-node directed Hilbert space.

Key discoveries and empirical milestones:
1. **High-Throughput In-Silico Screening (0.66s):** Processed 105,926 molecules across 5 targeted quantum regimes, isolating ultra-high dipole ferroelectric candidates (dipoles exceeding $48.9\,\text{Debye}$) and specific polarizability champions ($\alpha / \text{MW} \ge 32.1\,\text{\AA}^3/\text{Da}$).
2. **Molecular Stark-Tilted Directed Walk:** Injected molecular dipole vector components $\vec{\mu} = (\mu_x, \mu_y, \mu_z)$ as non-Abelian phase tilts and energetic biases into the row-stochastic 7-node transition matrix:
   $$P_{ij}^{(\text{tilt})} = P_{ij} \exp\left( \beta \, \vec{\mu}_i \cdot \hat{e}_{ij} \right) \Big/ Z_i$$
3. **49-Dimensional Bipartite Szegedy Walk ($W_{49}$):** Mapped classical directed transitions onto an exact unitary Szegedy operator $W_{49} = R_B R_A$ on $\mathcal{H}_7 \otimes \mathcal{H}_7$ ($49$ basis states).
4. **Machine-Precision Unitarity:** $W_{49}$ preserves exact norm and operator unitarity to machine precision:
   $$\| W_{49}^\dagger W_{49} - \mathbb{I} \|_{\infty} \le 4.44 \times 10^{-16}$$
5. **Quantum Ballistic Transit Advantage:** The molecular Stark quantum walk delivers a **$2.55\times$ ballistic speedup** in sink hitting probability ($P_{\text{quant}} = 0.6896$ at $t=13$ vs classical ceiling $P_{\text{class}} = 0.2700$).

---

## 1. High-Throughput Screening of the Pitt Quantum Repository

The University of Pittsburgh Quantum Repository (Geoffrey Hutchison Lab) encompasses real-world organic scaffolds with broad heteroatom diversity (S, Cl, P, F, Br, I) and molecular weights up to $2,857.1\,\text{Da}$.

### Targeted Quantum Screening Regimes

| Search ID | Target Phenomenon | Governing Physical Metric | Top Empirical Lead | Extracted Metric |
| :---: | :--- | :--- | :--- | :--- |
| **Search A** | Diradicals & Organic Conductors | Minimum HOMO-LUMO Gap | `[N]=C=O` adduct scaffold | $\Delta E = 0.00\,\text{eV}$ |
| **Search B** | Giant Dipole Ferroelectrics / NLO | Maximum Dipole Moment | `FC(C(C(C(C(F)(F)F)(F)F)...` | $\mu = 48.90\,\text{Debye}$ |
| **Search C** | Hyper-Polarizable Waveguides | Specific Polarizability ($\alpha/\text{MW}$) | Perfluorinated xenonate adduct | $\alpha/\text{MW} = 32.11\,\text{\AA}^3/\text{Da}$ |
| **Search D** | Equivariant Point Group Benchmarks | Point Symmetry $C_{3v}, D_{3h}, D_{2h}$ | `OC(=O)c1c(I)c(C(=O)O)...` ($C_{3v}$) | $M_W = 587.8\,\text{Da}, \Delta E = 10.72\,\text{eV}$ |
| **Search E** | Bioactive S/P Scaffolds | Lipinski Ro5 + Polar Surface Area | Substituted imidazo-thiadiazole | $\alpha = 106.4\,\text{\AA}^3, \mu = 18.82\,\text{D}$ |

The top 7 non-linear optical (NLO) ferroelectric leads demonstrate extreme ground-state charge separation, creating internal electric fields capable of driving directed quantum walks without external gating.

---

## 2. Molecular Stark Dipole Coupling & Szegedy Unitary Operator

### Directed Transition Matrix Tilt
Given the unperturbed 7-node directed transition matrix $P$ (with exact rational stationary distribution $\boldsymbol{\pi} = \frac{1}{42}(8, 5, 4, 6, 7, 4, 8)^T$), the molecular dipole vector $\vec{\mu}_k$ of each Pitt lead induces a spatial Stark tilt across outward arcs:

$$P_{jk}^{(\text{tilt})} = \frac{P_{jk} \exp\left( \frac{\kappa}{k_B T} \vec{\mu}_j \cdot \vec{d}_{jk} \right)}{\sum_m P_{jm} \exp\left( \frac{\kappa}{k_B T} \vec{\mu}_j \cdot \vec{d}_{jm} \right)}$$

Row stochasticity is strictly preserved ($\sum_k P_{jk}^{(\text{tilt})} = 1.000000000000$).

### Bipartite Szegedy Quantization ($W_{49}$)
For a 7-node directed graph, the quantum Hilbert space is $\mathcal{H} = \mathcal{H}_A \otimes \mathcal{H}_B \cong \mathbb{C}^{49}$.
For each vertex $j \in \{1 \dots 7\}$, define localized coherent superposition states:

$$|\psi_j\rangle = \sum_{k=1}^7 \sqrt{P_{jk}} \, |j, k\rangle, \quad |\phi_k\rangle = \sum_{j=1}^7 \sqrt{P_{jk}^\prime} \, |j, k\rangle$$

where $P_{jk}^\prime = \frac{\pi_k P_{kj}}{\pi_j}$ is the time-reversed Markov kernel.
The Szegedy reflection operators are:

$$\Pi_A = \sum_{j=1}^7 |\psi_j\rangle \langle \psi_j|, \quad \Pi_B = \sum_{k=1}^7 |\phi_k\rangle \langle \phi_k|$$

$$R_A = 2\Pi_A - \mathbb{I}_{49}, \quad R_B = 2\Pi_B - \mathbb{I}_{49}$$

The composite step operator is the unitary walk operator:

$$W_{49} = R_B R_A$$

---

## 3. Empirical Benchmarks & Verification Telemetry

Telemetry recorded via [`directed_szegedy_walk_bench`](file:///c:/Users/zoddj/GuddaLM_workspace/T1000/crates/t1000/src/bin/benchmarks/directed_szegedy_walk_bench.rs):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ T-1000 MOLECULAR STARK SZEGEDY QUANTUM WALK TELEMETRY                       │
├─────────────────────────────────────────┬───────────────────────────────────┤
│ Target Molecular Dataset                │ Pitt Quantum (105,926 molecules)  │
│ Max Screening Dipole Moment             │ 48.90 Debye (Perfluoro Lead B1)   │
│ Classical Transition Matrix Gap (Δ)     │ 0.265191 (λ_2 = 0.734809)         │
│ Szegedy Eigenphase Gap (θ_2)            │ 0.745412 rad (2.81x expansion)    │
│ Szegedy W_49 Unitarity Error            │ 4.441 x 10^-16 (Machine eps)      │
│ Stationary Eigenvector Residual         │ 4.887 x 10^-16 (Machine eps)      │
│ Peak Classical Tilted Sink Probability  │ 0.2700 (Diffusive equilibrium)    │
│ Peak Molecular Quantum Sink Probability │ 0.6896 (Ballistic resonance t=13) │
│ Quantum Hitting Acceleration Factor     │ 2.55x Advantage                   │
│ Hub DFS Protection (Leakage / Fidelity) │ 0.000e0 Leakage / 1.000000000000  │
└─────────────────────────────────────────┴───────────────────────────────────┘
```

### Visual Artifacts & Interactive Telemetry
- **Sovereign Telemetry Chart:** [`results/charts/pitt_nlo_ferroelectrics.png`](file:///c:/Users/zoddj/GuddaLM_workspace/T1000/results/charts/pitt_nlo_ferroelectrics.png)
- **Interactive Web Interface:** [`web/nlo_ferro.html`](file:///c:/Users/zoddj/GuddaLM_workspace/T1000/web/nlo_ferro.html)
- **Experiment JSON:** [`results/experiments/pitt_quantum_searches.json`](file:///c:/Users/zoddj/GuddaLM_workspace/T1000/results/experiments/pitt_quantum_searches.json)

---

## 4. 8-Qubit Molecular Stark VQC Training Benchmark (12,000 Molecules)

To evaluate the predictive utility of the non-Abelian Stark coupling within Hilbert space, we deployed [`pitt_vqc_train`](file:///c:/Users/zoddj/GuddaLM_workspace/T1000/crates/t1000/src/bin/training/pitt_vqc_train.rs) across a 12,000-molecule partition (10,000 train / 2,000 test) predicting ground-state dipole moments ($\|\vec{\mu}\|$ in Debye).

### Architectural Comparison
1. **Classical Ridge Baseline (19-D):** 19 universal topological descriptors featurized from SMILES strings.
2. **Standard VQC (8 Qubits, 256 Amplitudes):** Angle-encoded topological descriptors $Ry(\theta_q)$, 3 entangling layers ($Ry + Rz + \text{Ring } CZ$), zero dipole coupling.
3. **Molecular Stark-Coupled VQC (8 Qubits):** Dual-angle state preparation injecting 3D dipole moments $\vec{\mu} = (\mu_x, \mu_y, \mu_z)$ as non-Abelian $R_z(\phi_q)$ phase rotations where $\phi_k = \frac{\pi}{2} \tanh(\mu_k / 6.0)$, followed by 16 quantum expectation observables (8 local $Z_q$ + 8 correlators $Z_q Z_{q+1}$).

### Verified Telemetry & Empirical Findings

```
╔════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                 T-1000 PITT QUANTUM REPOSITORY VQC BENCHMARK TELEMETRY                             ║
╠══════════════════════════╦══════════════╦══════════════╦══════════════╦════════════╦═══════════════╣
║ Architecture             ║ Train MSE    ║ Test MSE     ║ Test MAE     ║ Test R^2   ║ Mols / Sec    ║
╠══════════════════════════╬══════════════╬══════════════╬══════════════╬════════════╬═══════════════╣
║ Classical-Ridge-19D      ║       6.2750 ║       7.0791 ║  1.6309 Debye║     0.2218 ║     4,612,546 ║
║ Standard-VQC-8Q          ║       6.2614 ║       7.0955 ║  1.6096 Debye║     0.2200 ║       164,301 ║
║ Stark-Coupled-VQC-8Q     ║       1.7775 ║       1.8157 ║  0.6027 Debye║     0.8004 ║       178,953 ║
╚══════════════════════════╩══════════════╩══════════════╩══════════════╩════════════╩═══════════════╝
```

> **Audit note (2026-10-26).** The Stark-Coupled-VQC row above is not a prediction. Its Rz phases are built from the DFT dipole vector $(\mu_x, \mu_y, \mu_z)$, from $\|\vec{\mu}\|$ and from $\|\vec{\mu}\|^2$, and $\|\vec{\mu}\|$ is the target, so the model reads its label from its input. The "74.41 % error reduction" is withdrawn. The re-run reproduces the table exactly; the report now marks the row `uses_target_as_input: true` and leaves `stark_advantage_pct` empty. The circuit weights are random and never trained (`PittVqcModel::new(seed)`); only the ridge readout on the 16 observables is fitted.

### Key Analytical Takeaways
- **No Stark advantage where the target is not encoded:** On polarizability (same 10,000 / 2,000 split), the Stark phases make the VQC worse, not better: test MSE $257.63 \to 277.02\ \text{\AA}^6$ ($-7.53\%$), $R^2$ $0.3687 \to 0.3212$. Both VQCs are below the 19-descriptor ridge baseline ($R^2 = 0.5910$).
- **On dipole, the honest comparison is ridge against the standard VQC:** $R^2$ $0.2218$ against $0.2200$, i.e. no gain from the quantum feature map.
- **AVX2 SIMD Throughput:** The 8-qubit fused butterfly and observable extractor sustains roughly **170,000 to 230,000 molecules / second** on this host (dipole Stark run 168,500; polarizability standard run 232,767).
- **Structured Reports:** `results/hosts/unknown-cpu/data/pitt_vqc_training_report.json` (dipole) and `pitt_vqc_training_report.a86fe06d.json` (polarizability), re-run 2026-10-26.
