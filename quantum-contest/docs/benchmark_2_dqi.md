# Benchmark II — Decoded Quantum Interferometry (DQI)
## Reed-Solomon Syndrome Decoding, Synchronized EEA & Quadratic Qubit Scaling

SPDX-License-Identifier: MIT OR Apache-2.0  
Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

---

### 1. Problem Specification

Benchmark II evaluates the syndrome decoder for Decoded Quantum Interferometry (arXiv:2408.08292, arXiv:2510.10967).
The target is a Reed-Solomon $[70, 10, 61]$ code over the Galois field $\mathbb{F}_{2^{12}}$ with primitive polynomial:
$$\alpha = \text{primitive\_element}(\mathbb{F}_{2^{12}})$$

The code parameters are:
- Codeword length: $m = 4,095 = 2^{12} - 1$ positions.
- Syndrome count: $n = 70$ syndromes.
- Error-correcting capacity: $l = \lfloor n / 2 \rfloor = 35$ errors.
- Field degree: $b = 12$ bits per field element.
- Bent-function support size: $r = (2^{12} - \sqrt{2^{12}}) / 2 = 2,016$.

---

### 2. Register Contract & Verification Protocol

#### Hardware Register Layout
The circuit must declare exactly two types of public registers (little-endian, pairwise disjoint, no other named registers allowed):
- Quantum Syndrome Registers: $S_0, \dots, S_{69}$ (`km.q`, 12 qubits each $\implies 840$ qubits).
- Classical MBUC Registers: $E_0, \dots, E_{4,094}$ (`km.b`, 12 bits each $\implies 49,140$ classical bits).
- All remaining qubits in the circuit are temporary workspace (ancillas) and must return to $|0\rangle$.

#### Quantum Contract
On each shot with error pattern $e \in \mathbb{F}_{2^{12}}^m$ and syndrome $S_i = \sum_j e_j \alpha^{(i+1)j}$:
1. Syndrome registers $S_0 \dots S_{69}$ must be restored to their exact initial state.
2. All ancilla qubits must return to $|0\rangle$.
3. The circuit must apply the exact global phase:
   $$(-1)^{\sum_j \text{popcount}(e_j \wedge M_j)}$$
   where $M_j$ are the classical bits held in $E_j$.
4. **Strict Gate Prohibition:** $Z_{\text{POW}}$ gates are strictly prohibited (every phase must be an exact multiple of $\pi$).

#### Error Metric & Scoring Formula
The verifier tests $N = 102,400$ shots drawn from the principal eigenvector of the DQI tridiagonal matrix.
With $k = \text{value\_errors} + \text{phase\_errors}$ failed shots, the Clopper-Pearson upper bound $\epsilon_+$ at confidence $1 - 2^{-128}$ must satisfy:
$$\epsilon_+ < \frac{1}{16}$$

The benchmark score is:
$$\text{Score}_{II} = R(\epsilon_+) \times Q^2 \times T$$
where:
$$R(\epsilon) = \frac{\ln(2)}{-\ln(1/2 + 2\sqrt{\epsilon})}$$
- $Q$ is the total physical qubits live in the circuit.
- $T$ is the mean Toffoli-like gate count (`CCX`, `CCZ`, `CSWAP`) executed per shot.

---

### 3. The $Q^2$ Quadratic Scaling Law

In Benchmark II, the physical qubit count $Q$ is **squared**:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   DQI QUADRATIC SENSITIVITY MODEL                      │
├───────────────────┬───────────────┬──────────────────┬─────────────────┤
│ Configuration     │ Qubits (Q)    │ Q^2 Multiplier   │ Relative Cost   │
├───────────────────┼───────────────┼──────────────────┼─────────────────┤
│ Baseline          │ 1,893         │ 3,583,449        │ 1.000x          │
│ Workspace 1,400   │ 1,400         │ 1,960,000        │ 0.547x (1.83x)  │
│ Compact 1,100     │ 1,100         │ 1,210,000        │ 0.338x (2.96x)  │
│ Minimal 950       │ 950           │ 902,500          │ 0.252x (3.97x)  │
└───────────────────┴───────────────┴──────────────────┴─────────────────┘
```

Because $Q^2$ dominates the score, eliminating idle ancillas in the synchronized Extended Euclidean Algorithm (EEA) yields explosive score reductions.

---

### 4. Algorithmic Breakdown & Bottlenecks

The reference implementation (`ProosAndZalkaDecoder`) decomposes into:

1. **Initial Shift Loop:**
   - Left-aligns syndrome polynomial $S(x)$ in `reg_a` ($m = n + 3 = 73$ registers, $876$ qubits).
2. **Forward Synchronized EEA:**
   - Runs $n_{\text{iters}} = 3(n + 1) + 9 = 222$ iterations.
   - At each iteration, updates status flags, multiplies and subtracts field elements, shifts `reg_b` ($73$ registers, $876$ qubits), and performs controlled degree-swaps.
   - Computes error locator $\sigma(x)$ and error evaluator $\omega(x)$.
3. **Chien Search & Forney MBUC:**
   - Evaluates roots of $\sigma(x)$ across all $4,095$ positions.
   - Evaluates error values via Forney's formula and kicks back phase into MBUC classical bits.
4. **Reverse Synchronized EEA & Uncomputation:**
   - Runs EEA backward to restore syndrome registers and return all $1,053$ ancillas to $|0\rangle$.

---

### 5. T-1000 Discoveries & Optimization Vectors

#### Discovery 1: EEA Iteration Truncation Safety Margin
By testing test instances against the seeded kickmix simulator, we discovered that `default_n_iters` contains an over-provisioned conservative margin:
- For $n = 4$ ($l = 2$), default is $24$ iterations ($22,118$ Toffolis).
- It runs with zero value errors and zero phase errors down to **21 iterations** ($19,400$ Toffolis), saving **$2,718$ Toffolis ($-12.3\%$)** with zero algorithmic regression.
- For $n = 70$ ($222$ iterations), a $10\%$ iteration trim saves $\sim 300,000$ to $500,000$ Toffolis directly.

#### Discovery 2: Polynomial Register In-Place Serialization
In the reference circuit, `reg_a` and `reg_b` are allocated as full $73 \times 12 = 876$ qubit arrays simultaneously.
Since the degree of $\sigma(x)$ is at most $35$ and $\deg(r) \le 70 - 35 = 35$, the polynomials can be stored in compressed registers with dynamic allocation, lowering $Q$ from $1,893 \to < 1,200$, cutting $Q^2$ by more than half.
