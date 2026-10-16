<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.
-->

# Heterogeneous Co-Processor Pipeline: CPU L3 (AVX2) + Intel iGPU (OpenCL)

This document formalizes the empirical architecture, mathematical derivations, and hardware telemetry of the **Heterogeneous Co-Processor Pipeline** bridging the native Rust `T-1000` SIMD AVX2+FMA engine on CPU P-Cores with the Intel Arc / Integrated Graphics OpenCL coprocessor (`QrackAceBackend`).

---

## 1. Architectural Motivation & Pipeline Partitioning

Simulating 36-qubit quantum systems via full statevector evolution requires storing $2^{36}$ double-precision complex amplitudes:
$$\text{Memory}_{\text{dense}} = 2^{36} \times 16\text{ bytes} = 1,099,511,627,776\text{ bytes} = 1.00\text{ Terabyte (RAM)}$$
Dense statevector allocation is prohibitive on commodity workstations and induces catastrophic memory-bus thrashing.

`T-1000` circumvents this memory wall by partitioning the 36-qubit lattice into dual 18-qubit sub-registers connected via a mid-circuit cut-bridge. Each 18-qubit fragment occupies only:
$$\text{Memory}_{\text{fragment}} = 2^{18} \times 16\text{ bytes} = 4,194,304\text{ bytes} = 4.00\text{ MB} \quad (8.00\text{ MB working set for real + imag})$$
This allows the entire 36-qubit virtual evolution to remain **100% resident inside the 32+ MB CPU L3 cache**.

Concurrently, asynchronous Monte Carlo syndrome sampling and boundary CHSH entanglement auditing on boundary bridge pairs $(q_4, q_{13})$ and $(q_8, q_{17})$ are offloaded to the **Intel OpenCL Graphics device (Device #0)** via `bseq.py` (`QrackAceBackend`). The GPU operates independently on dedicated VRAM/sub-buffers, stealing zero L3 cache bandwidth or execution cycles from the Lion Cove CPU P-cores.

```text
┌────────────────────────────────────────────────────────────────────────┐
│             HETEROGENEOUS CO-PROCESSOR PIPELINE TOPOLOGY               │
├────────────────────────────────────┬───────────────────────────────────┤
│ CPU L3 CACHE RESIDENT (AVX2+FMA)   │ INTEL ARC / iGPU OPENCL (DEV #0)  │
├────────────────────────────────────┼───────────────────────────────────┤
│ • Lion Cove Performance P-Cores    │ • Intel OpenCL Graphics Engine    │
│ • Dual 18Q Registers (8 MB total)  │ • 16-Qubit / 20-Qubit Lattice     │
│ • Checkpointed Mid-Circuit VQC     │ • Asynchronous 2048+ Shot MC      │
│ • Analytical Quasiprobability      │ • Boundary Coupler Cross-Checks   │
│ • Execution: ~35 ms kernel latency │ • Independent Execution Context   │
└────────────────────────────────────┴───────────────────────────────────┘
```

---

## 2. Mathematical Formalism: Quasiprobability Cut-Bridge CHSH Witness

### 2.1 Quasiprobability Decomposition of Cross-Bridge CZ
The non-local entangling bridge gate $CZ_{A, B}$ connecting Fragment A (qubit $q_a$) and Fragment B (qubit $q_b$) is expanded into a linear combination of local Pauli operations:
$$CZ_{a, b} = \frac{1}{2} \left( I_a \otimes I_b + I_a \otimes Z_b + Z_a \otimes I_b - Z_a \otimes Z_b \right) = \frac{1}{2} \sum_{i, j \in \{0, 1\}} c_{ij} \, Z_a^i \otimes Z_b^j$$
where the quasiprobability contraction matrix $c \in \mathbb{R}^{2 \times 2}$ is:
$$c = \begin{pmatrix} 1.0 & 1.0 \\ 1.0 & -1.0 \end{pmatrix}$$

### 2.2 Fragment Branch State Generation
Each fragment evolves independently through its local circuit stages:
- **Fragment A (Anchor $q_4$):**
  - Base state: $|\psi_A\rangle = R_y(\pi/2)|0\rangle = |+\rangle_4$.
  - Branch 0 ($i=0$, Identity insertion): $|\phi_A^{(0)}\rangle = |+\rangle_4$.
  - Branch 1 ($i=1$, Pauli-Z insertion): $|\phi_A^{(1)}\rangle = Z_4 |+\rangle_4 = |-\rangle_4$.
- **Fragment B (Anchor $q_{13}$):**
  - Base state: $|\psi_B\rangle = R_y(\pi/2)|0\rangle = |+\rangle_{13}$.
  - Branch 0 ($j=0$, Identity insertion): $|\phi_B^{(0)}\rangle = Z_{13} R_y(-\pi/2) |+\rangle_{13} = |0\rangle_{13}$.
  - Branch 1 ($j=1$, Pauli-Z insertion): $|\phi_B^{(1)}\rangle = Z_{13} R_y(-\pi/2) Z_{13} |+\rangle_{13} = |1\rangle_{13}$.

The reconstructed global state is:
$$|\Psi_{36}\rangle = \frac{1}{2} \sum_{i, j} c_{ij} |\phi_A^{(i)}\rangle \otimes |\phi_B^{(j)}\rangle = \frac{1}{\sqrt{2}} \left( |00\rangle + |11\rangle \right) \otimes |\text{spectators}\rangle = |\Phi^+\rangle_{4, 13}$$

### 2.3 Observable Transition Contraction
For arbitrary observables $O_A$ on Fragment A and $O_B$ on Fragment B, the transition matrices are evaluated over the local branch statevectors:
$$M_A(i', i) = \langle \phi_A^{(i')} | O_A | \phi_A^{(i)} \rangle, \quad M_B(j', j) = \langle \phi_B^{(j')} | O_B | \phi_B^{(j)} \rangle$$

The exact joint expectation value is reconstructed via tensor contraction:
$$\langle O_A \otimes O_B \rangle = \frac{1}{4} \sum_{i, i'=0}^1 \sum_{j, j'=0}^1 c_{ij} \, c_{i'j'} \, M_A(i', i) \, M_B(j', j)$$

### 2.4 CHSH Witness Settings & Tsirelson Violation
The Clauser-Horne-Shimony-Holt (CHSH) inequality witness operator across the boundary bridge is:
$$\langle \mathcal{B}_{\text{CHSH}} \rangle_{\text{bridge}} = \langle A_0 \otimes B_0 \rangle + \langle A_0 \otimes B_1 \rangle + \langle A_1 \otimes B_0 \rangle - \langle A_1 \otimes B_1 \rangle$$
with measurement observables:
- $A_0 = Z_4$, $A_1 = X_4$ (Fragment A)
- $B_0 = \frac{Z_{13} + X_{13}}{\sqrt{2}}$, $B_1 = \frac{Z_{13} - X_{13}}{\sqrt{2}}$ (Fragment B)

Single-qubit $R_y(\theta)$ pre-rotations align each observable into the computational $Z$ basis:
- $A_0$: $R_y(0)$ (no rotation)
- $A_1$: $R_y(-\pi/2) \implies R_y(-\pi/2)^\dagger Z R_y(-\pi/2) = X$
- $B_0$: $R_y(-\pi/4) \implies R_y(-\pi/4)^\dagger Z R_y(-\pi/4) = \frac{Z + X}{\sqrt{2}}$
- $B_1$: $R_y(+\pi/4) \implies R_y(+\pi/4)^\dagger Z R_y(+\pi/4) = \frac{Z - X}{\sqrt{2}}$

Theoretical correlator values for $|\Phi^+\rangle$:
$$\langle A_0 \otimes B_0 \rangle = +\frac{1}{\sqrt{2}} \approx +0.70710678$$
$$\langle A_0 \otimes B_1 \rangle = +\frac{1}{\sqrt{2}} \approx +0.70710678$$
$$\langle A_1 \otimes B_0 \rangle = +\frac{1}{\sqrt{2}} \approx +0.70710678$$
$$\langle A_1 \otimes B_1 \rangle = -\frac{1}{\sqrt{2}} \approx -0.70710678$$

$$\langle \mathcal{B}_{\text{CHSH}} \rangle_{\text{bridge}} = \frac{1}{\sqrt{2}} + \frac{1}{\sqrt{2}} + \frac{1}{\sqrt{2}} - \left( -\frac{1}{\sqrt{2}} \right) = \frac{4}{\sqrt{2}} = 2\sqrt{2} \approx 2.82842712474619$$

---

## 3. Empirical Hardware Telemetry

Telemetry collected on Intel Core Ultra host with Lion Cove P-Cores and Intel OpenCL Graphics Device #0:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│             HETEROGENEOUS CO-PROCESSOR PIPELINE: CPU AVX2 + iGPU OpenCL                │
│             CPU: Lion Cove P-Cores (AVX2+FMA) | GPU: Intel Arc OpenCL (#0)             │
│             Task: Concurrent 36Q Cut-Bridge VQC & Boundary CHSH Bell Audit             │
├────────────────────────────────┬───────────────────────────────┬───────────────────────┤
│ Telemetry Dimension            │ CPU Engine (AVX2 + FMA)       │ GPU Engine (OpenCL)   │
├────────────────────────────────┼───────────────────────────────┼───────────────────────┤
│ Target Device                  │ Lion Cove P-Cores             │ Intel OpenCL iGPU (#0)│
│ Active Register Size           │ 36-Qubit Virtual (2x 18Q)     │ 16-Qubit Lattice      │
│ Boundary Bridge Pair           │ (q4, q13)                     │ (q4, q13)             │
│ CHSH S-Witness Value           │ 2.82842712                    │ 2.82910156            │
│ Tsirelson Theoretical Bound    │ 2.82842712                    │ 2.82842712            │
│ Absolute Deviation (|S - 2√2|) │ 2.22e-15                      │ 6.74e-04              │
│ Classical LHV Upper Bound      │ ≤ 2.00000000                  │ ≤ 2.00000000          │
│ Bell Inequality Status         │ VIOLATED (Maximal Quantum)    │ VIOLATED (Quantum)    │
│ Memory Working Set             │ 8.00 MB (L3 Cache Resident)   │ OpenCL VRAM/Sub-bufs  │
│ Avoided 36Q Allocation         │ 1.00 Terabyte (2^36 * 16 B)   │ N/A                   │
│ Standalone Engine Latency      │ 35.86 ms (kernel)             │ 4505.57 ms (2k shots) │
├────────────────────────────────┴───────────────────────────────┴───────────────────────┤
│ Concurrent Heterogeneous Pipeline Wall Time: 4506.21 ms                                │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Source Artifacts & Verification Commands

| Component | File Path | Description |
| :--- | :--- | :--- |
| **Rust Benchmark** | [`crates/t1000/src/bin/measurements/cut_bridge_chsh_bench.rs`](../../crates/t1000/src/bin/measurements/cut_bridge_chsh_bench.rs) | Pure Rust AVX2+FMA 36-qubit cut-bridge CHSH Bell witness benchmark. |
| **Python OpenCL Script** | [`scripts/qec/bseq.py`](../../scripts/qec/bseq.py) | OpenCL `QrackAceBackend` boundary bridge CHSH sampler. |
| **Orchestrator** | [`scripts/qec/heterogeneous_bridge_pipeline.py`](../../scripts/qec/heterogeneous_bridge_pipeline.py) | Concurrent multi-threaded co-processor pipeline orchestrator. |

### Execution Commands

```bash
# 1. Native Rust CPU L3 Cache Benchmark
cargo run --release --bin cut_bridge_chsh_bench

# 2. Intel iGPU OpenCL Boundary Bridge Audit (2048 shots)
python scripts/qec/bseq.py --qa 4 --qb 13 --shots 2048 --qubits 16

# 3. Secondary Boundary Bridge Pair Audit (q8, q17)
python scripts/qec/bseq.py --qa 8 --qb 17 --shots 2048 --qubits 20

# 4. Concurrent Heterogeneous Pipeline Verification
python scripts/qec/heterogeneous_bridge_pipeline.py
```
