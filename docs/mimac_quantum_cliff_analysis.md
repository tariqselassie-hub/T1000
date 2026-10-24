// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

# T-1000 MIMAC Quantum Activity Cliff & Attractor Convergence Report

## Executive Summary
This document provides the full empirical and theoretical analysis of the Molecular Interaction Manifold with Activity Cliffs (MIMAC) dataset screened through the T-1000 Variational Quantum Circuit (VQC) and Quantum Neural Network engine.

Through Search A (Quantum Activity Cliff & SAR Discontinuity Hunter), T-1000 identified that three distinct low-potency precursors independently converge onto the exact same active lead molecule: **`NCc1cccc(CO)c1` (Molecule #109, [3-(aminomethyl)phenyl]methanol)** with bioactivity leaps spanning $3.7\times$ to $9.3\times$.

---

## 1. Dataset Ingestion & Normalization Protocol

### Ingested Data Topology
- **Dataset:** `Molecular_Interaction_Manifold_Dataset.zip` (MIMAC)
- **Total Molecules:** 325 compounds across 5 structural clusters
- **Activity Cliffs:** 584 annotated discontinuity pairs
- **Graph Topology:** 1,625 KNN affinity edges
- **Fingerprint Tensor:** Shape `(325, 1024)` in `fingerprints_multiview.npy`

### Mathematical Normalization
Every molecule was mapped across 10 continuous physicochemical features:
1. `raw_bioactivity` (Original scalar affinity)
2. `logp` (Wildman-Crippen lipophilicity)
3. `tpsa` (Topological Polar Surface Area)
4. `molwt` (Exact molecular weight)
5. `dipole` (Calculated electric dipole moment)
6. `charge` (Partial charge density)
7. `balaban_j` (Balaban topological index)
8. `bertz_ct` (Bertz molecular complexity)
9. `manifold_x` (Manifold coordinate 1)
10. `manifold_y` (Manifold coordinate 2)

Normalized tensors were generated and exported to `data/mimac/normalized/`:
- **Z-Score Normalization:** $z_i = (x_i - \mu_i) / \sigma_i$
- **Hilbert Space Injection Angles:** $\theta_i = \pi \cdot \tilde{x}_i - \frac{\pi}{2} \in \left[-\frac{\pi}{2}, +\frac{\pi}{2}\right]$ for single-qubit parameter rotations $R_y(\theta_i)$.

---

## 2. Search A: Quantum Activity Cliff Discovery

The hybrid discontinuity index between molecules $A$ and $B$ is defined by:

$$S_{\text{hybrid}}(A, B) = \alpha \cdot F_Q(A, B) + (1 - \alpha) \cdot T_{\text{tanimoto}}(A, B)$$

$$\text{Cliff Metric} = \frac{|\Delta \text{Activity}|}{1 - S_{\text{hybrid}}(A, B) + \epsilon}$$

Where $F_Q(A, B) = |\langle\psi(A)|\psi(B)\rangle|^2$ is the statevector overlap computed across 10-qubit Hilbert space ($\mathcal{H}^{\otimes 10}$, dimension $2^{10} = 1024$).

### Top 5 Discovered Quantum Activity Cliffs
1. **Cliff #1:** Molecule #94 (`c1cc(N)ccc1N`) $\to$ Molecule #319 (`c1ccc(N)c(CO)c1`), $\Delta A = +7.21$ ($6.70\times$ leap, $F_Q = 0.908$).
2. **Cliff #2:** Molecule #297 (`NCc1cccc(N)c1`) $\to$ Molecule #109 (`NCc1cccc(CO)c1`), $\Delta A = +7.64$ ($9.31\times$ leap, $F_Q = 0.931$).
3. **Cliff #3:** Molecule #304 (`CCc1cccc(CN)c1`) $\to$ Molecule #38 (`CCc1cc(N)ccc1`), $\Delta A = +5.66$ ($3.92\times$ leap, $F_Q = 0.977$).
4. **Cliff #4:** Molecule #304 (`CCc1cccc(CN)c1`) $\to$ Molecule #109 (`NCc1cccc(CO)c1`), $\Delta A = +6.62$ ($4.41\times$ leap, $F_Q = 0.756$).
5. **Cliff #5:** Molecule #93 (`NCc1cccc(F)c1`) $\to$ Molecule #109 (`NCc1cccc(CO)c1`), $\Delta A = +6.24$ ($3.70\times$ leap, $F_Q = 0.816$).

Notice that Cliffs #2, #4, and #5 all terminate at **Molecule #109**.

---

## 3. Deep Convergence Analysis: Molecule #109 as an Attractor

### Comparative Structural Matrix
All four compounds share the core meta-substituted benzylamine scaffold `[R]-c1cccc(CN)c1`:

| Parameter | Lead #109 | Precursor #297 | Precursor #304 | Precursor #93 |
| :--- | :--- | :--- | :--- | :--- |
| **SMILES** | `NCc1cccc(CO)c1` | `NCc1cccc(N)c1` | `CCc1cccc(CN)c1` | `NCc1cccc(F)c1` |
| **Meta Substituent [R]** | **`-CH2OH`** | **`-NH2`** | **`-CH2CH3`** | **`-F`** |
| **Bioactivity** | **8.558** | 0.919 | 1.942 | 2.314 |
| **Activity Ratio** | Reference | **$9.31\times$** | **$4.41\times$** | **$3.70\times$** |
| **Manifold $(X, Y)$** | $(7.773, 6.236)$ | $(7.842, 6.211)$ | $(7.836, 6.332)$ | $(7.761, 6.222)$ |
| **Hamming Bit Match** | $1024 / 1024$ | $1012 / 1024$ ($98.8\%$) | $1014 / 1024$ ($99.0\%$) | $1013 / 1024$ ($98.9\%$) |
| **Quantum Fidelity $F_Q$** | $1.000$ | $0.9309$ | $0.7565$ | $0.8160$ |
| **Mol Weight ($\text{g/mol}$)**| $137.08$ | $122.08$ | $135.10$ | $125.06$ |
| **TPSA ($\text{\AA}^2$)** | $46.25$ | $52.04$ | $26.02$ | $26.02$ |
| **Calculated LogP** | $0.638$ | $0.728$ | $1.708$ | $1.284$ |

### Physical & Chemical Mechanism
1. **The Methylene Dihedral Freedom:** The single $sp^3$ methylene link (`-CH2-`) in `-CH2OH` breaks aromatic $\pi$-conjugation, giving full unconstrained rotational dihedral freedom to orient the hydrogen-bond donor/acceptor into the catalytic cleft.
2. **Failure of Aniline (`-NH2`):** Direct ring conjugation forces planar $sp^2$ geometry, freezing orientation and shortening reach by $1.54\,\text{\AA}$.
3. **Failure of Alkyl (`-Et`):** Matches steric volume and flexible chain length, but lacks lone pairs or polar protons for hydrogen bonding.
4. **Failure of Halogen (`-F`):** Directly on the ring, poor acceptor, zero donor capability.

---

## 4. MIMAC Manifold: Activity Cliff Vector Field Analysis & Latent Regularization

Empirical verification of the 2D topological manifold (`data/mimac/Activity Cliff Vector Feild Analysis.png`) details the global bioactivity landscape and localized activity cliff vector fields.

### Coordinate Space & Primary Activity Basin
- **Global Coordinates:** Latent Coordinate $X \in [-8, 35]$, Latent Coordinate $Y \in [-10, 22]$.
- **Primary Activity Basin:** Densest concentration sits in the upper-left quadrant ($X \in [-5, 0], Y \in [5, 12]$), bifurcating into:
  1. *Upper Manifold ($Y \approx 10\text{--}12$):* Tightly packed structural cluster containing high-density local web connections.
  2. *Lower Sub-Cluster ($Y \approx 5\text{--}8$):* Dense region harboring the primary attractor basin containing Lead #109 at $(7.773, 6.236)$ and precursor duds.

### Vector Field Bridges & Activity Cliffs
Directed vector transitions across short latent distances quantify steep structure-activity relationship (SAR) discontinuities:
- **Vertical Master Bridge ($Y \approx 11.5 \leftrightarrow 7.0$):** A prominent directed vector linking $(X \approx -0.5, Y \approx 11.5)$ down to $(X \approx -1.0, Y \approx 7.0)$. This vector constitutes the critical topological pivot where minimal functional alterations drive exponential potency shifts.
- **Intra-Basin Discontinuity Web:** Dense localized gradient networks clustered along the upper-right flank of the primary basin, highlighting micro-switches in binding affinity.

### Isolated Satellite Basins & Scaffold Hopping Candidates
Sparse peripheral clusters demonstrate orthogonal chemical space representation:
- **Far Upper-Right Outlier:** High-potency doublet at $(X \approx 33.5, Y \approx 22)$ exhibiting $\text{pIC}_{50} \sim 7\text{--}8$. Because this island resides far outside the primary topological manifold support, it represents a candidate for scaffold hopping and alternative binding modes.
- **Mid-Field & Lower-Right Islands:** Dispersed clusters at $(11, 15)$, $(9.5, 12.5)$, $(15, 8)$, $(12, -3.5)$, $(6, -7)$, and $(22, -10)$.
- **Left-Edge Chain:** Vertical triplet near $(X \approx -5, Y \in [-0.5, 1.5])$.

### Bioactivity Field Envelopes & Latent Discontinuities
- **Potency Gradient ($0\text{--}10$ Scale):** Maps bioactivity from baseline duds ($0\text{--}3$) to nanomolar inhibitors ($8\text{--}10$).
- **Local Field Envelopes:** Polygonal/hexagonal density halos tinted by potency demonstrate non-smooth, non-convex gradient propagation.
- **Latent Regularization Directives:** Co-localization of low-activity states ($2\text{--}4$) adjacent to ultra-potent leads ($7\text{--}9+$) necessitates:
  1. *Trajectory Interpolation:* Systematic latent vector sampling along the $Y \approx 11.5 \to 7.0$ vector to isolate the functional group switch.
  2. *Continuity Penalization:* Enforcing activity-aware contrastive losses or Lipschitz regularization bounds ($\|\nabla_{\mathbf{z}} \mathcal{A}(\mathbf{z})\| \le L$) to govern surrogate manifold stability during generative optimization.
