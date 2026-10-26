<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Terrence A. Jones SR. All rights reserved. -->

# T-1000 Scientific Finding: Interaction-Free Quantum Zeno Anomaly Detection and Ghost Boundary Sentry Filtering on Financial Transactions

> **Audit note (2026-10-26).** The first version picked the F1-maximising threshold (0.52) on the 10,000 holdout transactions and reported precision, recall, F1 and balanced accuracy at that threshold, so those four numbers had seen the test labels. The binary now picks the threshold on the 15,000 training transactions (0.53) and applies it unchanged to the holdout set: F1 17.91 % -> 17.31 %, balanced accuracy 60.37 % -> 59.37 %, recall 35.39 % -> 32.40 %, precision 11.98 % -> 11.81 %. AUROC (0.6777) and the alert means do not depend on a threshold and are unchanged. The report keeps the test-tuned F1 as `oracle_test_f1_score_pct`, for reference only. Two other corrections: only the 10 feature-scale parameters are trained (the 10 phase and 16 typology parameters stay at their initial values), and inference runs on all cores, not one.

**Document ID:** `T1000-FINDING-2026-040`  
**Deen Monograph:** `M166`  
**Author:** Terrence A. Jones SR  
**Date:** 2026-10-26  
**Status:** Verified on Disk & Hardware Grounded  
**Classification:** Quantum Zeno Dynamics & Interaction-Free Anomaly Sentry  
**Canonical Repository:** `https://github.com/tariqselassie-hub/T1000`  

---

## 1. Executive Summary

Traditional fraud detection architectures rely on deep classical neural networks, gradient-boosted decision trees, or dense autoencoders that invasively process, score, and buffer transactions, incurring significant computational overhead, calibration drift, and vulnerability to severe class imbalance (typically $<6\%$ fraud prevalence).

This finding introduces the **Quantum Zeno Sentry & Ghost Anomaly Detector**, implementing interaction-free boundary confinement on the 1,000,000-transaction financial banking dataset (`bank_fraud.csv`, 55,255 fraud instances = 5.5255% prevalence). Grounded in the **Ghost Zeno Effect** (`zeno_networking` protocol), the 10-qubit Hilbert space ($\mathcal{H}_{1024}$) is partitioned into two mutually orthogonal subspaces:

$$\mathcal{H}_{1024} = \mathcal{H}_{\text{flux}} \oplus \mathcal{H}_{\text{boundary}}$$

- **Authorized Flux Subspace ($\mathcal{H}_{\text{flux}} = \{|0000000000\rangle\}$):** Pristine ground-state amplitude representing undisturbed, coherent, legitimate transaction dynamics ($P_{\text{flux}} = |state[0]|^2$).
- **Forbidden Boundary Strata ($\mathcal{H}_{\text{boundary}} = \{|k\rangle : k > 0\}$):** 1,023 excited orthogonal basis states where anomalous transaction indicators (unusual velocity, nighttime PIN reset, multiple failed authorization attempts, international routing, atypical balance drain) scatter probability mass ($P_{\text{alert}} = 1.0 - |state[0]|^2$).

### Key Empirical Results (15,000 Train / 10,000 Holdout Test Samples)
- **Sentry Routing Contrast:** Clean transactions achieve a baseline leakage of $P_{\text{alert}} = 0.3385$ ($66.2\%$ confined in $\mathcal{H}_{\text{flux}}$), whereas fraudulent vectors elevate alert probability to $P_{\text{alert}} = 0.4639$, yielding a **$1.37:1$ routing contrast ratio** ($+37.1\%$ signal amplification into the boundary strata).
- **Discrimination Capability:** Evaluated on 10,000 holdout transactions ($5.34\%$ positive fraud prevalence), the system achieves an **AUROC of $0.6777$** and a **Balanced Accuracy of $59.37\%$** at threshold $0.53$, chosen on the training set ($32.40\%$ recall, $11.81\%$ precision, $F_1 = 17.31\%$).
- **Sub-Microsecond Latency:** Full 10-qubit unitary evolution and Zeno boundary projection averages **$0.85\,\mu\text{s/sample}$** over the holdout batch evaluated in parallel on all cores (about $1.18$ M transactions/sec on this host).
- **Zero Classical Readout Weights:** Requires **zero classical linear layers**, operating purely via Hilbert space subspace projection. The circuit declares $36$ parameters, of which the $10$ feature-scale parameters are trained; 3-epoch parameter-shift training takes **$5.82\text{ seconds}$** in pure Rust.

---

## 2. Hilbert Space Topology & Zeno Confinement

The 10-qubit state space ($2^{10} = 1,024$ amplitudes) is factorized into an operational hierarchy:

$$\mathcal{H} = \mathcal{H}_{\text{Sentry}}(1Q) \otimes \mathcal{H}_{\text{Typology}}(3Q) \otimes \mathcal{H}_{\text{Env}}(6Q)$$

| Register | Wires | Dimension | Operational Sentry Function |
| :--- | :---: | :---: | :--- |
| **Sentry Register ($\mathcal{H}_{\text{Sentry}}$)** | $q_0$ | $d=2$ | Sentry probe monitoring boundary leakage ($|0\rangle_{\text{sentry}}$ vs $|1\rangle_{\text{sentry}}$). |
| **Typology Register ($\mathcal{H}_{\text{Typology}}$)** | $q_1..q_3$ | $d=8$ | Fraud archetype subspace (Phishing, Cloning, Synthetic ID, Takeover). |
| **Environment Cooler ($\mathcal{H}_{\text{Env}}$)** | $q_4..q_9$ | $d=64$ | Entanglement sink cooling benign payment noise and transaction variance. |

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   QUANTUM ZENO BANK FRAUD SENTRY PERIMETER                 │
│                                                                             │
│  Forbidden Boundary Strata H_boundary (1,023 excited eigenstates: k > 0)    │
│  └── Multi-Channel CZ Coupling: Failed x Intl, Velocity x BalRatio, Night    │
│      ├── Dark Fringe: Clean Baseline -> P_flux = 66.2% (No Sentry Alarm)     │
│      └── Sentry Click: Fraud Vector -> P_alert = 46.4% (Boundary Leakage)   │
│                                                                             │
│  Authorized Flux Subspace H_flux (|0000000000>)                             │
│  └── Ground State Amplitude state[0] (0.80 us inline screening)             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Mathematical Formulation of Zeno Boundary Projection

### 3.1 Feature Angle Injection
Given a 10-dimensional continuous transaction descriptor vector $\boldsymbol{x} \in \mathbb{R}^{10}$, each feature is mapped to a unipolar deviation angle $\theta_q(\boldsymbol{x}) \in [0, \pi/2]$ representing departure from the normal baseline:

$$\theta_q(\boldsymbol{x}) = \text{clamp}\left(w_q \cdot \Delta_q(\boldsymbol{x}), 0, \frac{\pi}{2}\right)$$

For pristine clean transactions (normal spending, local location, daytime, 0 failed attempts), $\theta_q \approx 0$, so $R_y(\theta_q)|0\rangle \approx |0\rangle$.

### 3.2 Multi-Channel Interaction-Free Entanglement
Non-linear risk combinations are captured without classical multiplication by applying Controlled-$Z$ ($CZ$) entangling gates between correlated risk channels:

$$U_{\text{ent}} = CZ(0, 1) \cdot CZ(0, 2) \cdot CZ(1, 2) \cdot CZ(3, 4) \cdot CZ(0, 5) \cdot CZ(1, 5) \cdot CZ(6, 7)$$

When concurrent anomalies fire (e.g. international IP AND nighttime PIN reset), $CZ$ flips the phase of $|11\rangle$, inducing destructive interference on the ground state and accelerating probability expulsion into the boundary strata.

### 3.3 Zeno Probability Evaluation
The boundary leakage probability is evaluated via the orthogonal subspace projection:

$$P_{\text{flux}} = |\langle 0^{\otimes 10} | \psi(\boldsymbol{x})\rangle|^2 = |state[0]|^2$$

$$P_{\text{alert}}(\boldsymbol{x}) = \sum_{k=1}^{1023} |\langle k | \psi(\boldsymbol{x})\rangle|^2 = 1.0 - P_{\text{flux}}$$

---

## 4. Empirical Benchmark Data & Telemetry

Hardware execution on 15,000 training transactions and 10,000 holdout test transactions produced the following verified telemetry (`results/hosts/unknown-cpu/experiments/bank_fraud_zeno_report.json`):

| Telemetry Metric | Measured Value | Baseline / Theoretical Limit | Sovereign Physical Assessment |
| :--- | :---: | :---: | :--- |
| **Sentry Routing Contrast** | **$1.37:1$** | $1.00:1$ (Uniform Chance) | $+37.1\%$ probability elevation in $\mathcal{H}_{\text{boundary}}$ for fraud vectors. |
| **Clean Baseline $P_{\text{alert}}$** | **$0.3385$** ($33.8\%$) | — | $66.2\%$ of clean transactions remain pinned in $\mathcal{H}_{\text{flux}}$. |
| **Fraud Vector $P_{\text{alert}}$** | **$0.4639$** ($46.4\%$) | — | Detectable phase scattering into excited strata. |
| **AUROC Score** | **$0.6777$** | $0.5000$ (Random Guess) | Non-trivial separation without classical readout weights. |
| **Balanced Accuracy** | **$59.37\%$** | $50.00\%$ (Random Chance) | At the training-set threshold; class-balanced, so the $5.34\%$ prevalence does not inflate it. |
| **Operating Threshold** | **$0.53$** | — | Swept on the 15,000 training transactions to maximize $F_1$ (train $F_1 = 20.16\%$), then fixed. |
| **Fraud Recall** | **$32.40\%$** | — | 173 of 534 holdout fraud cases flagged. |
| **Fraud Precision** | **$11.81\%$** | $5.34\%$ (Base Prevalence) | $2.21\times$ enrichment over prevalence; about 1,465 alerts, of which about 1,290 are false positives. |
| **Fraud $F_1$** | **$17.31\%$** | — | A threshold tuned on the holdout labels would give $17.91\%$ (not a valid estimate). |
| **Inference Latency** | **$0.85\,\mu\text{s/sample}$** | $< 10.0\,\mu\text{s}$ Target | Mean over the holdout batch, evaluated in parallel on all cores (`par_iter`). |
| **Training Duration** | **$5.82\text{ sec}$** | — | 3 epochs of parameter-shift updates on the 10 feature-scale parameters. |
| **Classical Readout Weights** | **$0$** | Hundreds (Standard ML) | Pure unitary quantum mechanics. |

---

## 5. Visual Artifact Manifest

The complete scientific visual analysis is compiled in [`results/charts/bank_fraud_zeno_telemetry.png`](file:///c:/Users/zoddj/GuddaLM_workspace/T1000/results/charts/bank_fraud_zeno_telemetry.png):
- **Panel 1: Sentry Routing Contrast:** Bar chart illustrating the $0.3385 \to 0.4639$ contrast separation ($1.37:1$ ratio).
- **Panel 2: ROC Curve & Frontier:** Empirical ROC curve demonstrating AUROC $= 0.6777$ and the operating cut. The chart reads `optimal_threshold` from the report; regenerate it after the 2026-10-26 audit so it shows the training-set threshold 0.53.
- **Panel 3: Subspace Strata Decomposition:** Stacked probability distribution comparing $\mathcal{H}_{\text{flux}}$ vs $\mathcal{H}_{\text{boundary}}$.
- **Panel 4: Latency & Parameter Economy:** Readouts validating $0.80\,\mu\text{s}$ per transaction and 36 total parameters.

---

## 6. Scientific Attribution & Standards Compliance

- **Author & Copyright:** Terrence A. Jones SR (2026).
- **Dual Licensing:** MIT License OR Apache License, Version 2.0.
- **Code Standard:** Zero warnings (`cargo check`), 100% test pass rate (`cargo test -p t1000 --lib bank_fraud_data`), LoC limits strictly respected ($\le 500$ LoC binary, $\le 800$ LoC module).
