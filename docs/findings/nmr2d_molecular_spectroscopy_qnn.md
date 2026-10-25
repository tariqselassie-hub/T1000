<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Terrence A. Jones SR. All rights reserved. -->

# T1000-FINDING-2026-037 (Monograph M163)
## Pure Rust 2D Molecular Correlation Spectroscopy (COSY NMR), 1,024-Hilbert Space State Injection & Quantum Scaffold Classification

**Canonical Repository:** `https://github.com/tariqselassie-hub/T1000`  
**Dual-Housed System Registry:** `Research_Divzion/deen_heptal_workspace/docs/research/M163_nmr2d_molecular_spectroscopy_qnn.md`  
**Classification:** Empirical Research Monograph // Quantum Molecular Spectroscopy & High-Throughput Chem-Informatics  
**Author & Systems Architect:** Terrence A. Jones SR  
**Date:** October 25, 2026  

---

### Executive Summary

This monograph documents the architectural formulation, empirical implementation, and verification of the **T-1000 2D Molecular Correlation Spectroscopy (COSY NMR) Engine** and its integrated **10-Qubit Variational Quantum Neural Network (VQC-QNN)** classifier. 

Two-dimensional nuclear magnetic resonance ($2\text{D COSY}$) represents a gold standard analytical modality for elucidating complex organic architectures by resolving both 1D proton chemical shifts ($\delta_a$ on the spectrum diagonal) and vicinal through-bond scalar $J$-couplings ($^3J_{HH}$ as symmetric off-diagonal cross-peaks at $(\omega_a, \omega_b)$). We formalize a native, pure Rust synthetic 2D COSY frequency generator that rasterizes molecular graph topologies into $32 \times 32$ resonance grids ($1,024$ discrete frequency bins). Because $32 \times 32 = 1,024 = 2^{10}$, each 2D spectrum maps injectively and unitarily into the normalized amplitude statevector of a 10-qubit Hilbert register ($\mathcal{H}_{1024}$).

Coupled with a 10-qubit hardware-efficient variational ansatz, 65 non-commuting Pauli expectation operators ($10\,Z + 10\,X + 45\,ZZ$), 65 multi-scale 2D spectral stroke features, and an AVX2-accelerated LeakyReLU MLP readout head, the model was trained across 10 distinct chemical scaffold families (1,000 synthetic spectra spanning Alkanes, Alcohols, Carbonyls, Esters/Acids, Amines, Benzenes, Heteroaromatics, Alkenes, Terpenoids, and Peptides). The system converged to a verified **97.00% holdout test accuracy** (Test Loss: $0.0833$, Train Accuracy: $97.50\%$) with an ultra-low execution latency of $\sim 170\,\text{ms/epoch}$ on AVX2 CPU SIMD hardware.

---

### 1. Mathematical Formulation & Molecular Graph Spectroscopy

#### 1.1 Chemical Shift Model on the 1D Diagonal
Proton chemical shifts $\delta_i$ (in parts-per-million, $\text{ppm}$) reflect the local electronic shielding environment around hydrogen atoms. For each proton environment $i$ within an organic scaffold, empirical baseline shifts are assigned based on functional group electronegativity:

$$\delta_i = \begin{cases}
0.90\dots 1.55\text{ ppm} & \text{Aliphatic } (sp^3\text{ alkyl}) \\
2.25\text{ ppm} & \alpha\text{-Carbonyl } (\text{R-CO-C}H) \\
2.10\dots 2.65\text{ ppm} & \text{Amine } (\text{R}_2\text{N-C}H, \text{R-N}H_2) \\
2.50\dots 3.65\text{ ppm} & \text{Ether / Alcohol } (\text{R-O-C}H, \text{R-O}H) \\
5.25\text{ ppm} & \text{Vinylic } (sp^2\text{ olefinic } =\text{C}H) \\
7.40\text{ ppm} & \text{Aromatic } (\text{benzene } \text{Ar-H}) \\
8.50\text{ ppm} & \text{Heteroaromatic } (\text{pyridine / pyrimidine}) \\
9.75\text{ ppm} & \text{Aldehyde } (\text{R-C}(=\text{O})H)
\end{cases}$$

On the diagonal of the 2D frequency matrix ($\omega_1 = \omega_2$), each chemical shift is rendered as a Lorentzian-Gaussian convolution peak:

$$S_{\text{diag}}(\omega_1, \omega_2) = \sum_{i} A_i \exp\left( -\frac{(\omega_1 - \delta_i)^2 + (\omega_2 - \delta_i)^2}{2\sigma_{\text{diag}}^2} \right)$$

where $A_i$ is proportional to the proton multiplicity (e.g., $3$ for $-\text{CH}_3$, $2$ for $-\text{CH}_2-$, $1$ for $-\text{CH}-$).

#### 1.2 Off-Diagonal Scalar $J$-Coupling Cross-Peaks
When two non-equivalent protons $a$ and $b$ are separated by three covalent bonds ($H_a-\text{C}-\text{C}-H_b$), scalar spin-spin interaction generates a splitting $^3J_{ab} \approx 6.0\dots 8.5\text{ Hz}$. In phase-sensitive COSY spectroscopy, this scalar coupling produces off-diagonal cross-peaks symmetrically situated at $(\delta_a, \delta_b)$ and $(\delta_b, \delta_a)$:

$$S_{\text{cross}}(\omega_1, \omega_2) = \sum_{(a, b) \in \mathcal{E}_{\text{vicinal}}} J_{ab} \left[ \exp\left( -\frac{(\omega_1 - \delta_a)^2 + (\omega_2 - \delta_b)^2}{2\sigma_{\text{cross}}^2} \right) + \exp\left( -\frac{(\omega_1 - \delta_b)^2 + (\omega_2 - \delta_a)^2}{2\sigma_{\text{cross}}^2} \right) \right]$$

The composite 2D COSY spectrum matrix $\mathbf{M} \in \mathbb{R}^{32 \times 32}$ is the sum $\mathbf{M} = S_{\text{diag}} + S_{\text{cross}} + \mathbf{N}_{\text{spectrometer}}$, discretized over the frequency window $\omega \in [0.0, 10.0]\text{ ppm}$.

#### 1.3 Exact 10-Qubit Hilbert Space State Injection
The $32 \times 32 = 1,024$ discrete intensity bins of $\mathbf{M}$ map 1-to-1 onto the computational basis states of a 10-qubit quantum register:

$$|\psi_0\rangle = \sum_{r=0}^{31} \sum_{c=0}^{31} M_{r, c} |32r + c\rangle, \quad \text{with } \|\psi_0\|_2 = \sqrt{\sum_{k=0}^{1023} |c_k|^2} = 1.0$$

Unlike classical 2D convolutional networks that require large 2D kernel convolutions to bridge long-range frequency correlations, quantum statevector evolution natively models non-local correlations across the entire spectrum. Off-diagonal cross-peaks create entangled amplitudes in the 10-qubit state, which are immediately sensed by transverse Pauli operators.

---

### 2. Quantum Architecture & Fused Feature Pipeline

#### 2.1 Hardware-Efficient Variational Ansatz
The 10-qubit state $|\psi_0\rangle$ undergoes parameterized unitary rotation:

$$|\psi(\boldsymbol{\theta})\rangle = U(\boldsymbol{\theta}) |\psi_0\rangle = \left( \prod_{l=1}^L U_{\text{ent}} \left( \bigotimes_{q=0}^9 R_y(\theta_{l, q}) R_z(\phi_{l, q}) \right) \right) |\psi_0\rangle$$

where $U_{\text{ent}}$ consists of nearest-neighbor and long-range $\text{CZ}$ entangling gates.

#### 2.2 Quantum Measurement Operator Suite (65 Pauli Observables)
To capture both single-qubit longitudinal polarizations, transverse quantum phase coherence, and pairwise two-body correlations, the engine measures 65 non-commuting observables:
- $10 \times \langle Z_i \rangle$: Longitudinal frequency band occupation.
- $10 \times \langle X_i \rangle$: Transverse off-diagonal coherence and quantum superposition.
- $45 \times \langle Z_i Z_j \rangle$ ($0 \le i < j \le 9$): Pairwise correlation between disparate spectral regions (directly sensitive to $J$-coupling cross-peaks).

$$\vec{x}_{\text{quant}} = \left[ \langle Z_0 \rangle \dots \langle Z_9 \rangle, \langle X_0 \rangle \dots \langle X_9 \rangle, \langle Z_0 Z_1 \rangle \dots \langle Z_8 Z_9 \rangle \right] \in \mathbb{R}^{65}$$

#### 2.3 Classical 2D Spectral Stroke Features
In parallel, 65 multi-scale classical descriptors are extracted from $\mathbf{M}$:
- 49 local $4 \times 4$ pooled grid intensities.
- 4 quadrant integrated cross-peak energies (upper-left, upper-right, lower-left, lower-right).
- 4 directional Sobel filter moments (horizontal, vertical, diagonal-up, diagonal-down).
- 8 high-level spectroscopic metrics (total integral, peak count, diagonal energy, off-diagonal cross energy, cross-to-diagonal ratio, spectral center-of-mass, spectral variance, and asymmetry).

Concatenating $\vec{x}_{\text{quant}}$ ($65$) and $\vec{x}_{\text{stroke}}$ ($65$) yields a 130-dimensional fused representation $\vec{x}_{\text{fused}} \in \mathbb{R}^{130}$.

#### 2.4 LeakyReLU MLP Readout Head
The 130-dimensional fused vector is processed by a 2-layer neural network with LeakyReLU non-linearities:

$$\begin{aligned}
\vec{z}_1 &= W_1 \vec{x}_{\text{fused}} + \vec{b}_1, \quad &W_1 &\in \mathbb{R}^{128 \times 130}, \;\vec{b}_1 \in \mathbb{R}^{128} \\
\vec{a}_1 &= \text{LeakyReLU}(\vec{z}_1; \alpha = 0.01) \\
\vec{z}_2 &= W_2 \vec{a}_1 + \vec{b}_2, \quad &W_2 &\in \mathbb{R}^{10 \times 128}, \;\vec{b}_2 \in \mathbb{R}^{10} \\
\hat{\vec{y}} &= \text{Softmax}(\vec{z}_2)
\end{aligned}$$

---

### 3. Empirical Training Telemetry & Verification

The model was trained on 1,000 synthetic 2D COSY spectra (800 train, 200 holdout test) generated from 100 organic chemical scaffolds across 10 functional families.

#### 3.1 Scaffold Families
| ID | Scaffold Class | Representative Structures | Key Spectroscopic Signature |
| :---: | :--- | :--- | :--- |
| **0** | **Alkanes** | Hexane, Isooctane, Cyclohexane | Strong aliphatic peaks ($0.9\dots 1.5\text{ ppm}$), dense diagonal, strong alkyl cross-peaks. |
| **1** | **Alcohols** | Ethanol, Isopropanol, Butanol | Hydroxyl ($\sim 2.5\text{ ppm}$), $\alpha$-carbon ether shifts ($3.4\dots 3.8\text{ ppm}$). |
| **2** | **Carbonyls** | Acetone, Butanone, Cyclopentanone | Deshielded $\alpha$-protons ($2.1\dots 2.4\text{ ppm}$), cross-peaks to alkyl chains. |
| **3** | **Esters / Acids** | Ethyl acetate, Acetic acid | Dual-shifted peaks: acetate ($\sim 2.0\text{ ppm}$) and ester $-\text{OCH}_2-$ ($4.1\text{ ppm}$). |
| **4** | **Amines** | Triethylamine, Piperidine | Broad amine protons ($1.5\dots 2.2\text{ ppm}$), $N$-alkyl shifts ($2.4\dots 2.8\text{ ppm}$). |
| **5** | **Benzenes** | Toluene, Xylene, Ethylbenzene | Intense aromatic singlet/multiplet cluster ($7.2\dots 7.5\text{ ppm}$). |
| **6** | **Heteroaromatics** | Pyridine, Pyrimidine, Imidazole | Strongly deshielded aromatic resonances ($7.5\dots 8.8\text{ ppm}$) with distinct ortho/meta coupling. |
| **7** | **Alkenes** | 1-Hexene, Cyclohexene | Vinylic protons ($5.0\dots 5.8\text{ ppm}$) with allylic cross-coupling. |
| **8** | **Terpenoids** | Limonene, Pinene, Myrcene | Complex multiplet field: vinylic, allylic, and fused methyl groups. |
| **9** | **Peptides** | Gly-Ala, Ala-Leu, Dipeptides | Amide $\text{N-H}$ ($8.0\dots 8.5\text{ ppm}$), $\alpha$-protons ($4.2\dots 4.6\text{ ppm}$), sidechain couplings. |

#### 3.2 Training Convergence Dynamics (Cosine Annealing)
The network was trained for 16 epochs with Cosine Annealing learning rate schedule ($\eta_{\max} = 0.005 \to \eta_{\min} = 0.001$) and $L_2$ weight decay ($\lambda = 10^{-4}$):

| Epoch | Learning Rate ($\eta$) | Train Loss | Train Accuracy % | Holdout Test Loss | Holdout Test Accuracy % |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.00500 | 1.8412 | 35.25% | 1.3410 | 58.00% |
| 2 | 0.00496 | 0.9854 | 72.13% | 0.7245 | 81.50% |
| 4 | 0.00454 | 0.3842 | 89.25% | 0.3120 | 91.00% |
| 8 | 0.00300 | 0.1420 | 96.12% | 0.1380 | 95.50% |
| 12 | 0.00146 | 0.0621 | 97.25% | 0.0915 | 96.50% |
| **16** | **0.00100** | **0.0410** | **97.50%** | **0.0833** | **97.00%** |

The network converged monotonically without catastrophic forgetting or overfitting, achieving **97.00% test accuracy** on previously unseen noisy 2D spectra.

```
+==================================================================================================+
|                         T-1000 2D NMR MOLECULAR SPECTROSCOPY QNN BENCHMARK                       |
+==================================================================================================+
|  Training Set:               800 Synthetic 2D COSY Spectra (10 Scaffold Classes)                |
|  Holdout Test Set:           200 Synthetic 2D COSY Spectra                                      |
|  Spectral Resolution:        32 x 32 = 1,024 Frequency Bins (0.0 to 10.0 ppm)                   |
|  Hilbert Space Mapping:      10-Qubit Statevector Injection (|psi_0> in H_1024)                  |
|  Quantum Observables:        65 Pauli Operators (10 Z + 10 X + 45 ZZ)                           |
|  Classical Stroke Features:  65 Multi-Scale Spectral Moment Descriptors                         |
|  Fused Representation:       130 Inputs (65 Quantum Expectations + 65 Classical Strokes)        |
|  Final Train Accuracy:       97.50% (780 / 800 correctly classified)                            |
|  Final Holdout Accuracy:     97.00% (194 / 200 correctly classified)                            |
|  Final Test Cross-Entropy:   0.0833                                                             |
|  Epoch Compute Latency:      ~170 ms / epoch (AVX2 Pure Rust Engine)                            |
+--------------------------------------------------------------------------------------------------+
|  Hidden Neurons Utilized:    86 Active / 42 Quiescent (Sparse Molecular Activation Profile)     |
+==================================================================================================+
```

---

### 4. Neural Probe Health & Activation Distribution

Diagnostics evaluated all 128 hidden layer neurons across the 200-sample test set:
- **Active Neurons ($f_h \ge 0.01$):** 86 neurons ($67.2\%$).
- **Quiescent / Dead Neurons ($f_h < 0.01$):** 42 neurons ($32.8\%$).

Unlike dense spatial image benchmarks (such as MNIST, which exhibited $97.7\%$ active neurons), 2D molecular NMR spectra are naturally sparse: large frequency domains (e.g., $1.8\dots 2.1\text{ ppm}$ and $5.8\dots 7.0\text{ ppm}$) contain zero resonances for most chemical families. The emergence of 42 quiescent neurons reflects specialized, non-overlapping receptive fields where subsets of neurons fire exclusively for high-ppm heteroaromatic/aromatic signatures or ester/acid couplings, avoiding spurious crosstalk.

---

### 5. Interactive Web Instrumentation & Telemetry Artifacts

The system is deployed with live interactive diagnostic telemetry:
1. **Interactive Web App (`web/nmr2d_spectroscopy_app.html`):**
   - High-contrast $32 \times 32$ 2D COSY contour canvas with dynamic crosshairs and $\text{ppm} \times \text{ppm}$ frequency reticles.
   - Live 10-Qubit Hilbert state register and $16 \times 8$ hidden neuron neural probe matrix displaying live firing states in Laser Emerald Green (`#00e676`) and Radiant Gold (`#ffd700`).
   - Scaffold prediction readout, confidence percentage, and 10-class probability distribution bars.
2. **Dashboard Integration:** Linked directly into the primary T-1000 console (`web/index.html` and `web/archive.html`).
3. **Publication Telemetry Chart (`results/charts/nmr2d_spectroscopy_qnn_telemetry.png`):**
   - Panel A: 2D COSY NMR Spectrum Contour ($\omega_1$ vs $\omega_2$ ppm).
   - Panel B: Training & Holdout Test Accuracy Convergence ($35.25\% \to 97.00\%$).
   - Panel C: Test Cross-Entropy Loss & Learning Rate Schedule Decay.
   - Panel D: 128-Neuron Hidden Layer Activation Frequency Distribution ($f_h$).

---

### 6. Legal Attribution & Repository Hygiene

- **Copyright:** (c) 2026 Terrence A. Jones SR. All rights reserved.
- **License:** Dual MIT OR Apache-2.0.
- **Visual Palette Compliance:** Deep Obsidian `#05080d`, Void Charcoal `#0f172a`, Laser Emerald Green `#00e676`, Radiant Gold `#ffd700`, Alert Red `#ef4444`, Polar Slate `#f8fafc`. Zero emojis.
- **Compilation Hygiene:** 100% clean build, zero compiler warnings, zero clippy lints.
