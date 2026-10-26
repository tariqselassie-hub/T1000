<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Terrence A. Jones SR. All rights reserved. -->

# T1000-FINDING-2026-037 (Monograph M163)
## Bipartite Tensor Hilbert State Injection, 80 Targeted Cross-Peak Symmetry Observables & 2D COSY NMR Scaffold Classification

**Canonical Repository:** `https://github.com/tariqselassie-hub/T1000`  
**Dual-Housed System Registry:** `Research_Divzion/deen_heptal_workspace/docs/research/M163_nmr2d_molecular_spectroscopy_qnn.md`  
**Classification:** Empirical Research Monograph // Quantum Molecular Spectroscopy & Bipartite Symmetry Operators  
**Author & Systems Architect:** Terrence A. Jones SR  
**Date:** October 25, 2026  

> **Audit note (2026-10-26).** The first version reported 98.00 % holdout accuracy on 200 of 1,000 spectra. The COSY renderer is deterministic and the 1,000 spectra were 10 identical copies of each of 100 molecules, so all 200 test spectra were copies of training spectra. The binary now runs 5-fold molecule-disjoint cross-validation; the held-out accuracy is **82.00 %** (82/100 molecules). Sections 3 and 4 are rewritten; the web app header, the dashboard label and the telemetry chart still show the old figure until they are regenerated.

---

### Executive Summary

This monograph documents the architectural formulation, empirical implementation, and verification of the **T-1000 2D Molecular Correlation Spectroscopy (COSY NMR) Engine** and its upgraded **10-Qubit Bipartite Variational Quantum Neural Network (VQC-QNN)** classifier. 

Two-dimensional nuclear magnetic resonance ($2\text{D COSY}$) represents the gold standard analytical modality for elucidating complex organic architectures by resolving both 1D proton chemical shifts ($\delta_a$ on the spectrum diagonal) and vicinal through-bond scalar $J$-couplings ($^3J_{HH}$ as symmetric off-diagonal cross-peaks at $(\omega_a, \omega_b)$ and $(\omega_b, \omega_a)$). By explicitly factorizing the 10-qubit Hilbert space $\mathcal{H}_{1024}$ into a bipartite tensor product of two 5-qubit registers:

$$\mathcal{H}_{1024} = \mathcal{H}_{\omega_1} \otimes \mathcal{H}_{\omega_2} = \mathcal{H}_{32} \otimes \mathcal{H}_{32}$$

and upgrading the quantum measurement suite from 65 to **80 targeted Pauli observables**, the engine directly measures the two-qubit $\text{SWAP}$ symmetry between row ($\omega_1$, vertical shift $r \in [0, 31]$) and column ($\omega_2$, horizontal shift $c \in [0, 31]$) coordinates. 

Coupled with a 5-layer variational ansatz incorporating cross-register bipartite CNOT rungs, 65 multi-scale 2D spectral stroke features (145 fused inputs), and an AVX2-accelerated LeakyReLU MLP readout head, the model was trained across 10 distinct chemical scaffold families (100 molecules spanning Alkanes, Alcohols, Carbonyls, Esters/Acids, Amines, Benzenes, Heteroaromatics, Alkenes, Terpenoids, and Peptides, one synthetic spectrum each). Under 5-fold molecule-disjoint cross-validation it classifies **82.00% of held-out molecules** correctly (82/100; folds 70 % to 95 %; train accuracy 97 % to 99 %) at $\sim 220\,\text{ms/epoch}$ on AVX2 CPU SIMD hardware.

---

### 1. Bipartite Register Topology & Mathematical Formulation

#### 1.1 Dyadic Frequency Coordinate Decomposition ($32 \times 32 \to \mathcal{H}_{32} \otimes \mathcal{H}_{32}$)
In the flat 1,024-element amplitude statevector $|\psi\rangle = \sum_{k=0}^{1023} c_k |k\rangle$, each basis index $k = 32r + c$ uniquely decomposes into two 5-bit binary frequency coordinates:
- **Row Register ($\omega_1$, qubits $q_9 \dots q_5$):** Encodes the indirect vertical chemical shift axis $r \in [0, 31]$, from coarse quadrant ($q_9$, MSB) down to sub-ppm multiplet envelope ($q_5$, LSB).
- **Column Register ($\omega_2$, qubits $q_4 \dots q_0$):** Encodes the direct horizontal chemical shift axis $c \in [0, 31]$, from coarse quadrant ($q_4$, MSB) down to fine resolution ($q_0$, LSB).

| Scale Level $s$ | Row Qubit ($\omega_1$) | Column Qubit ($\omega_2$) | Spectral Resolution ($\Delta\delta$ on $[0, 10]\,\text{ppm}$) | Bitmask Pair $(M_r, M_c)$ |
| :---: | :---: | :---: | :---: | :---: |
| **$s = 4$ (MSB)** | $q_9$ | $q_4$ | $5.000\,\text{ppm}$ (Upfield vs. Downfield split) | $(1 \ll 9, 1 \ll 4) = (512, 16)$ |
| **$s = 3$** | $q_8$ | $q_3$ | $2.500\,\text{ppm}$ (Aromatic / Vinyl / Aliphatic) | $(1 \ll 8, 1 \ll 3) = (256, 8)$ |
| **$s = 2$** | $q_7$ | $q_2$ | $1.250\,\text{ppm}$ (Functional group sub-bands) | $(1 \ll 7, 1 \ll 2) = (128, 4)$ |
| **$s = 1$** | $q_6$ | $q_1$ | $0.625\,\text{ppm}$ (Multiplet envelope) | $(1 \ll 6, 1 \ll 1) = (64, 2)$ |
| **$s = 0$ (LSB)** | $q_5$ | $q_0$ | $0.3125\,\text{ppm}$ (Fine vicinal $J$-splitting) | $(1 \ll 5, 1 \ll 0) = (32, 1)$ |

#### 1.2 Transverse $XX$ and $YY$ Observables as COSY Symmetry Witnesses
A phase-sensitive 2D COSY spectrum is Hermitian and symmetric across the main diagonal ($A_{r, c} = A_{c, r}$), meaning the quantum state is invariant under the 5-pair swap operator $\mathcal{S} = \prod_{s=0}^4 \text{SWAP}(q_{s+5}, q_s)$. For each matched scale pair $(q_{s+5}, q_s)$, the local two-qubit swap expands in the Pauli basis as:

$$\text{SWAP}_{s+5, s} = \frac{1}{2}\left(I + Z_{s+5}Z_s + X_{s+5}X_s + Y_{s+5}Y_s\right)$$

Evaluating the sum and difference of the transverse operators isolates the Bell-state coherence between symmetric cross-peaks $|r\rangle|c\rangle$ and $|c\rangle|r\rangle$:

$$\begin{aligned}
\frac{1}{2}\left(X_{s+5}X_s + Y_{s+5}Y_s\right) &= |01\rangle\langle 10| + |10\rangle\langle 01| \\
\frac{1}{2}\left(X_{s+5}X_s - Y_{s+5}Y_s\right) &= |00\rangle\langle 11| + |11\rangle\langle 00|
\end{aligned}$$

- **$X_{s+5}X_s + Y_{s+5}Y_s$ (Symmetric Cross-Peak Exchange):** Directly measures off-diagonal coherence between coupled protons ($\langle r, c | \rho | c, r \rangle$). Diagonal peaks ($r = c$) contribute identically to $ZZ$, whereas off-diagonal cross-peaks drive $XX + YY$.
- **$X_{s+6}X_s$ (Adjacent-Scale Cross Coupling):** Captures vicinal $J$-couplings that straddle a dyadic bin boundary between scale $s+1$ and scale $s$.

---

### 2. Quantum Architecture & 80-Observable Specification

#### 2.1 Complete 80-Observable Suite
The upgraded observable suite measures 80 targeted operators evaluated in a single $\mathcal{O}(1024)$ pass over the statevector:

| Index Range | Count | Operator Family | Exact Definition | Physical Role in 2D COSY NMR |
| :---: | :---: | :--- | :--- | :--- |
| **`Obs[0..10]`** | 10 | Single-Qubit $Z$ | $Z_i \quad \forall i \in \{0\dots 9\}$ | 1D marginal chemical shift projections along $\omega_1$ and $\omega_2$. |
| **`Obs[10..20]`** | 10 | Single-Qubit $X$ | $X_i \quad \forall i \in \{0\dots 9\}$ | Local transverse phase readout across adjacent dyadic spectral bins. |
| **`Obs[20..65]`** | 45 | All-Pairs $ZZ$ | $Z_i Z_j \quad (0 \le i < j \le 9)$ | 10 intra-$\omega_1$, 10 intra-$\omega_2$, and 25 inter-register intensity correlations. |
| **`Obs[65..70]`** | 5 | Dyadic Mirror $XX$ | $X_{s+5} X_s \quad \forall s \in \{0\dots 4\}$ | Real symmetric cross-peak bit-flip interference between $\omega_1$ and $\omega_2$. |
| **`Obs[70..75]`** | 5 | Dyadic Mirror $YY$ | $Y_{s+5} Y_s \quad \forall s \in \{0\dots 4\}$ | Phase-inverted cross-peak interference; pairs with $XX$ to reconstruct $\text{SWAP}$. |
| **`Obs[75..79]`** | 4 | Adjacent-Scale $XX$ | $X_{s+6} X_s \quad \forall s \in \{0\dots 3\}$ | Vicinal $J$-couplings straddling scale boundaries ($s+1 \leftrightarrow s$). |
| **`Obs[79]`** | 1 | Global Parity $Z^{\otimes 10}$ | $\bigotimes_{i=0}^9 Z_i$ | Full 10-qubit parity anchor stabilizing global statevector entanglement. |

#### 2.2 Circuit Ansatz with Bipartite Cross-Rungs
To ensure the new $XX$ and $YY$ operators receive direct gradient flow during backpropagation, the ansatz incorporates 5 cross-register bipartite CNOTs ($\text{CNOT}(s+5, s)$ for $s \in \{0\dots 4\}$) at layer 4 (depth 5) right before the final $R_y/R_z$ variational layer. This shortens the entanglement circuit depth between corresponding $\omega_1$ and $\omega_2$ frequency scales from 5 hops down to 1 hop.

#### 2.3 Fused Feature Representation & MLP Readout Head
Concatenating $\vec{x}_{\text{quant}} \in \mathbb{R}^{80}$ with $\vec{x}_{\text{stroke}} \in \mathbb{R}^{65}$ yields a 145-dimensional fused feature vector $\vec{x}_{\text{fused}} \in \mathbb{R}^{145}$. The classical MLP head uses LeakyReLU non-linearities:

$$\begin{aligned}
\vec{z}_1 &= W_1 \vec{x}_{\text{fused}} + \vec{b}_1, \quad &W_1 &\in \mathbb{R}^{128 \times 145}, \;\vec{b}_1 \in \mathbb{R}^{128} \\
\vec{a}_1 &= \text{LeakyReLU}(\vec{z}_1; \alpha = 0.05) \\
\vec{z}_2 &= W_2 \vec{a}_1 + \vec{b}_2, \quad &W_2 &\in \mathbb{R}^{10 \times 128}, \;\vec{b}_2 \in \mathbb{10} \\
\hat{\vec{y}} &= \text{Softmax}(\vec{z}_2)
\end{aligned}$$

---

### 3. Empirical Training Telemetry & Verification

The catalog holds 100 organic molecules, 10 per functional family. The COSY renderer is deterministic, so each molecule has exactly one spectrum. The model is evaluated by **5-fold molecule-disjoint cross-validation**: fold $k$ holds out the catalog entries at positions $2k$ and $2k+1$ of every family (20 molecules, 2 per family) and trains a fresh model on the other 80. Each training molecule is presented 10 times per epoch, which keeps the 800-sample epoch and the 400 optimizer steps of the original run. Every molecule is held out exactly once, so the cross-validated accuracy is over 100 molecules none of which the model that scored it had seen.

#### 3.1 Training Configuration
16 epochs per fold, batch 32, cosine-annealed learning rates (ansatz $0.025 \to 0.005$, head $0.005 \to 0.001$), $L_2$ weight decay $\lambda = 10^{-4}$ on the head. The per-epoch holdout accuracy printed by the binary is monitoring only; the epoch count and the model were fixed in advance and nothing is selected on it.

#### 3.2 Cross-Validated Held-Out Accuracy (`results/hosts/unknown-cpu/data/nmr2d_spectroscopy_probe.json`)

| Fold | Held-out molecules | Correct | Held-out accuracy | Held-out loss | Train accuracy (final epoch) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 20 | 16 | 80.00% | 0.8321 | 96.88% |
| 2 | 20 | 19 | 95.00% | 0.2820 | 98.75% |
| 3 | 20 | 14 | 70.00% | 2.4393 | 96.88% |
| 4 | 20 | 17 | 85.00% | 0.6263 | 98.88% |
| 5 | 20 | 16 | 80.00% | 0.6735 | 98.75% |
| **All** | **100** | **82** | **82.00%** | — | — |

Per family (correct out of 10 held-out molecules): Alkanes 10, Alcohols 9, Aldehydes & Ketones 9, Acids & Esters 7, Amines & Amides 8, Benzenes 9, Heteroaromatics 6, Alkenes 10, Terpenoids 8, Amino Acids 6. The errors are chemically plausible neighbours: methylpyridines and thiophene called benzenes, short acids called carbonyls, amides called amines or amino acids, proline called an amine.

With 20 molecules per fold, one molecule is 5 points of fold accuracy; the fold spread (70 to 95 %) is the uncertainty to keep in mind. The gap between train (97 to 99 %) and held-out (82 %) accuracy is the generalisation gap the original single split could not show.

There is no classical-only baseline yet (for example the same head on the 65 classical spectral features alone), so this result does not say how much the 80 quantum observables add.

```
+==================================================================================================+
|                  T-1000 2D NMR BIPARTITE SPECTROSCOPY QNN BENCHMARK (80 OBSERVABLES)             |
+==================================================================================================+
|  Molecules:                  100 (10 Scaffold Classes x 10), one deterministic spectrum each     |
|  Evaluation:                 5-fold molecule-disjoint CV (80 train / 20 held-out per fold)       |
|  Spectral Resolution:        32 x 32 = 1,024 Frequency Bins (0.0 to 10.0 ppm)                   |
|  Hilbert Space Mapping:      10-Qubit Statevector Injection: H_w1(32) x H_w2(32)                |
|  Quantum Observables:        80 Bipartite Pauli Operators (10 Z + 10 X + 45 ZZ + 15 XX/YY/AllZ)|
|  Classical Stroke Features:  65 Multi-Scale Spectral Moment Descriptors                         |
|  Fused Representation:       145 Inputs (80 Quantum Expectations + 65 Classical Strokes)        |
|  Train Accuracy (per fold):  96.88% to 98.88%                                                   |
|  Held-Out Accuracy (CV):     82.00% (82 / 100 molecules); folds 70% to 95%                      |
|  Epoch Compute Latency:      ~220 ms / epoch (median of 80 epochs, this host)                   |
+--------------------------------------------------------------------------------------------------+
|  Hidden Neurons (fold 5):    82 Active / 46 Quiescent on its 20 held-out molecules              |
+==================================================================================================+
```

#### 3.3 Superseded Result
The first version split 1,000 spectra into 800 train / 200 test, but the 1,000 spectra were 10 identical copies of each of the 100 molecules, so every test spectrum was also in the training set. Its 98.00 % "holdout" accuracy (196/200) measured recall of training molecules and is withdrawn. The comparison with the 65-observable model ($+1.00\%$ accuracy) was made on the same split and is withdrawn with it.

---

### 4. Neural Probe Health & Activation Distribution

The probe uses the fold-5 model and its 20 held-out molecules (2 per family):
- **Active Neurons (fire on at least one held-out molecule):** 82 neurons ($64.1\%$).
- **Quiescent Neurons (fire on none):** 46 neurons ($35.9\%$).

These counts depend on the evaluation set; the 86 and 90 active-neuron figures of the first version were measured on duplicated training spectra and are not comparable.

---

### 5. Interactive Web Instrumentation & Telemetry Artifacts

1. **Interactive Web App (`web/nmr2d_spectroscopy_app.html`):**
   - High-contrast $32 \times 32$ 2D COSY contour canvas with dynamic crosshairs and $\text{ppm} \times \text{ppm}$ frequency reticles.
   - Live 10-Qubit Hilbert state register and $16 \times 8$ hidden neuron neural probe matrix displaying live firing states in Laser Emerald Green (`#00e676`) and Radiant Gold (`#ffd700`).
   - Scaffold prediction readout, confidence percentage, and 10-class probability distribution bars.
   - Header pills show accuracy, active neurons and observable count. (They were written for the withdrawn 98.00 % / 90-neuron run; the held-out figure is 82.00 %.)
2. **Dashboard Integration:** Linked directly into the primary T-1000 console (`web/index.html` updated to `2d nmr cosy (98%)` and `web/archive.html`).
3. **Publication Telemetry Chart (`results/charts/nmr2d_spectroscopy_qnn_telemetry.png`):**
   - Panel 1: 2D COSY Bipartite QNN Convergence (drawn from the withdrawn duplicated-spectra run; regenerate).
   - Panel 2: 2D COSY Spectrum Contour ($\omega_1$ vs $\omega_2$ ppm).
   - Panel 3: 128-Neuron Activation Matrix ($90$ active, $38$ quiescent).
   - Panel 4: Inferred Molecular Scaffold Confidence Bars.

---

### 6. Legal Attribution & Repository Hygiene

- **Copyright:** (c) 2026 Terrence A. Jones SR. All rights reserved.
- **License:** Dual MIT OR Apache-2.0.
- **Visual Palette Compliance:** Deep Obsidian `#05080d`, Void Charcoal `#0f172a`, Laser Emerald Green `#00e676`, Radiant Gold `#ffd700`, Alert Red `#ef4444`, Polar Slate `#f8fafc`. Zero emojis.
- **Compilation Hygiene:** 100% clean build, zero compiler warnings, zero clippy lints.
