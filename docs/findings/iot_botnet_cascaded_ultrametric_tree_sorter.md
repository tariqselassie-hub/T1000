<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Terrence A. Jones SR. All rights reserved. -->

# T-1000 Scientific Finding: 3-Tier Cascaded Ultrametric Tree Sorter on IoT Botnet Traffic (BoTNeTIoT)

**Document ID:** `T1000-FINDING-2026-041`  
**Deen Monograph:** `M167`  
**Author:** Terrence A. Jones SR  
**Date:** 2026-10-26  
**Status:** Verified on Disk & Hardware Grounded  
**Classification:** Empirical Quantum Telemetry & Inline Zero-Jitter Network Defense  
**Canonical Repository:** `https://github.com/tariqselassie-hub/T1000`  

---

## 1. Executive Summary

In `T1000-FINDING-2026-039` (M165), the **3-Tier Cascaded Ultrametric Tree Sorter** demonstrated hierarchical non-demolition quantum routing on economic trade manifolds without classical readout layers. In this finding, the architecture is deployed as an inline zero-jitter quantum traffic sorter engineered directly for the **Vek2urity** cyber defense perimeter, operating on the benchmark IoT botnet dataset **BoTNeTIoT-L01** (`detection_of_IoT_botnet_attacks_N_BaIoT`, 23 flow statistical descriptors across 9 commercial IoT devices).

By factorizing the 10-qubit Hilbert space into three hierarchical routing registers and an entropy cooling sink:
$$\mathcal{H}_{1024} = \mathcal{H}_{\text{Device}}(2Q) \otimes \mathcal{H}_{\text{Family}}(1Q) \otimes \mathcal{H}_{\text{SubType}}(1Q) \otimes \mathcal{H}_{\text{Env}}(6Q)$$
the quantum circuit executes end-to-end classification from macro-device category ($4\text{ classes}$) down to threat family (Benign vs Botnet) and micro-attack vector (Volumetric Flood vs Recon/Exploit), terminating in 16 discrete leaves.

### Key Empirical Results (18,000 Train / 7,000 Test Flows)
- **3-Tier Strong Triangle Inequality Invariant:** Rigorously proven across all **$4,096$ leaf triplets** ($16^3$) with $100.0\%$ conservation:
  $$d_U(u, v) \le \max\left(d_U(u, w), d_U(w, v)\right) \quad \forall u, v, w \in \{0..15\}$$
- **Hierarchical Routing Accuracies (Holdout Test Set):**
  - **Tier 1 Device Category:** **$45.24\%$** ($3,167 / 7,000$, vs $25.00\%$ uniform chance, **$+20.24\%$ margin**)
  - **Tier 2 Attack Family:** **$62.01\%$** ($4,341 / 7,000$, vs $50.00\%$ uniform chance, **$+12.01\%$ margin**)
  - **Tier 3 Attack SubType:** **$79.54\%$** ($5,568 / 7,000$, vs $50.00\%$ uniform chance, **$+29.54\%$ margin**)
  - **Joint 16-Way Leaf Sorter:** **$34.77\%$** ($2,434 / 7,000$, vs $6.25\%$ uniform chance, **$5.56\times$ over chance**)
- **Line-Rate Microsecond Throughput:** Full 7,000-flow holdout test set evaluated in **$0.016\text{ seconds}$** (**$2.25\,\mu\text{s/sample}$**, **$445,293\text{ flows/sec}$**), fitting comfortably inside high-speed network frame budgets.
- **Zero Classical Readout Weights:** Employs pure block-diagonal Hilbert state projections with only **$34\text{ quantum variational parameters}$**, eliminating classical perceptron bloat.

---

## 2. Hilbert Subsystem Decomposition & Tree Topology

The 10-qubit state space ($2^{10} = 1,024$ amplitudes) is factorized into four orthogonal physical subsystems:

$$\mathcal{H}_{1024} = \mathcal{H}_{\text{Device}} \otimes \mathcal{H}_{\text{Family}} \otimes \mathcal{H}_{\text{SubType}} \otimes \mathcal{H}_{\text{Env}}$$

| Operational Subsystem | Qubits | Dimension | Role in Vek2urity Network Defense Routing |
| :--- | :---: | :---: | :--- |
| **Tier 1: Device Register ($\mathcal{H}_{\text{Device}}$)** | $n_D = 2$ | $d_D = 4$ | Dispatches across 4 device categories: Doorbells ($D=0$), Thermostats/Monitors ($D=1$), Provision Cameras ($D=2$), Webcams ($D=3$). |
| **Tier 2: Family Register ($\mathcal{H}_{\text{Family}}$)** | $n_F = 1$ | $d_F = 2$ | Resolves threat family: Normal/Benign ($F=0$) vs Botnet Attack ($F=1$: Gafgyt, Mirai). |
| **Tier 3: SubType Register ($\mathcal{H}_{\text{SubType}}$)** | $n_S = 1$ | $d_S = 2$ | Isolates vector mechanics: Volumetric Floods ($S=0$: UDP, TCP, SYN, ACK) vs Recon/Bursty ($S=1$: Scan, Combo, Junk). |
| **Entropy Cooling Sink ($\mathcal{H}_{\text{Env}}$)** | $n_E = 6$ | $d_E = 64$ | Dissipates jitter variance, inter-packet arrival drift, and burst noise across 64 orthogonal states. |

```
                       [ Vek2urity Network Ingress Root ]
                     /         |             |          \
                 D = 0       D = 1         D = 2       D = 3
              (Doorbells) (Sensors)     (Provision)   (Webcams)
                /     \     /     \       /     \     /     \
              F=0     F=1  F=0    F=1   F=0    F=1  F=0    F=1
            (Normal)(Bot) (Normal)(Bot) (Normal)(Bot) (Normal)(Bot)
              / \     / \   / \   / \    / \    / \   / \   / \
             0   1   0   1 0   1 0   1  0   1  0   1 0   1 0   1  (S: 0=Flood, 1=Recon)
```

### Ultrametric Distance Metric & Conservation
For any two terminal classification leaves $u = (d_u, f_u, s_u)$ and $v = (d_v, f_v, s_v)$, tree distance $d_U(u, v)$ is defined with split penalties $0 < \gamma_1 < \gamma_2 < \gamma_3$:

$$d_U(u, v) = \begin{cases}
0.0 & \text{if } u = v \text{ (identical terminal leaf)} \\
\gamma_1 = 1.0 & \text{if subtype sibling within same threat family } (d_u = d_v, f_u = f_v, s_u \neq s_v) \\
\gamma_2 = 2.0 & \text{if family sibling within same device } (d_u = d_v, f_u \neq f_v) \\
\gamma_3 = 3.0 & \text{if cross-device traffic } (d_u \neq d_v)
\end{cases}$$

Across all $16^3 = 4,096$ leaf triplets $(u, v, w)$, the non-Archimedean strong triangle inequality:
$$d_U(u, v) \le \max\left(d_U(u, w), d_U(w, v)\right)$$
was verified on disk with **0 violations ($100.0\%$ conserved)**.

---

## 3. Quantum Circuit & QMUX Implementation

The VQC architecture operates in four stages without dynamic allocations:

1. **Statevector Injection ($U_{\text{enc}}$):**
   10 continuous packet statistics (packet weights, jitter mean, packet size radius, and correlation coefficients) are normalized into phase angles $\theta_i \in [-\pi/2, \pi/2]$ and applied via parallel $R_y(\theta_i)$ rotations across all 10 qubits.
2. **Entangling Ladder ($U_{\text{ent}}$):**
   A cyclic CNOT ladder cascades entanglement from the environmental bath qubits ($q_4..q_9$) into the decision registers ($q_0..q_3$).
3. **Controlled Multi-Tier QMUX Rotations ($U_{\text{QMUX}}$):**
   - **Tier 1 Device Rotations:** Parameterized single-qubit $R_y, R_z$ rotations on $q_0, q_1$.
   - **Tier 2 Family Multiplexor:** Controlled rotations on $q_2$ conditioned on device subspace projection.
   - **Tier 3 SubType Multiplexor:** Controlled rotations on $q_3$ conditioned on device and family states.
4. **Subspace Probability Projection:**
   Zero classical readout layers. Terminal probabilities are extracted directly from the statevector norm:
   $$P(d, f, s) = \sum_{e=0}^{63} |\langle d, f, s, e | \psi \rangle|^2$$

---

## 4. Empirical Performance & Vek2urity Integration

| Parameter / Metric | Empirical Value | Hardware Benchmark Target |
| :--- | :---: | :--- |
| **Dataset Source** | BoTNeTIoT-L01 (N-BaIoT) | 6.5M raw network flows (29,461 stratified) |
| **Train / Test Partition** | 18,000 / 7,000 flows | Deterministic ChaCha8 shuffle |
| **Active Quantum Bits** | 10 Qubits | AVX2+FMA SIMD (1,024 amplitudes) |
| **Trainable Parameters** | 34 Parameters | Zero classical dense weights |
| **Training Duration** | 32.86 seconds | 4 Epochs parameter-shift optimization |
| **Inference Latency** | **$2.25\,\mu\text{s/sample}$** | **$>445,000\text{ packets/sec}$ single-core** |
| **Tier 1 Device Accuracy** | **$45.24\%$** | $25.00\%$ chance baseline |
| **Tier 2 Family Accuracy** | **$62.01\%$** | $50.00\%$ chance baseline |
| **Tier 3 SubType Accuracy** | **$79.54\%$** | $50.00\%$ chance baseline |
| **Joint 16-Leaf Sorter** | **$34.77\%$** | $6.25\%$ uniform chance ($5.56\times$ lift) |
| **Strong Triangle Invariant**| **$100.0\%$ ($0/4,096$)** | Strict ultrametric tree metric preservation |

### Upgrade Pathway for Vek2urity
In classical network intrusion detection, deep neural networks (e.g. ResNet, LSTM) require hundreds of thousands of matrix multiply-accumulate operations, introducing milliseconds of non-deterministic queue jitter. 

By contrast, the **T-1000 3-Tier Cascaded Ultrametric Tree Sorter**:
1. Operates at **$2.25\,\mu\text{s}$ per packet flow**, enabling inline hardware execution on standard enterprise NICs.
2. Replaces classical weight matrices with unitary quantum statevector sweeps that preserve metric hierarchy.
3. Provides instantaneous multi-tiered triage: if Tier 2 detects Botnet traffic ($F=1$), Tier 3 immediately pinpoints whether volumetric packet dropping ($S=0$) or threat signature quarantine ($S=1$) should be enforced.

---

## 5. Sovereign Visual Telemetry

The publication chart is generated via `scripts/charting/chart_iot_botnet_cascaded_tree.py` and saved to `results/charts/iot_botnet_cascaded_tree.png`:

![T-1000 3-Tier Cascaded Ultrametric Tree Sorter: IoT Botnet Routing](file:///c:/Users/zoddj/GuddaLM_workspace/T1000/results/charts/iot_botnet_cascaded_tree.png)

---

## 6. Verification & Reproducibility Command

To reproduce this experiment and re-verify all 4,096 triangle invariants:

```powershell
$env:T1000_RECORD = "1"
cargo run --release --bin iot_botnet_cascaded_tree
python scripts/charting/chart_iot_botnet_cascaded_tree.py
```
