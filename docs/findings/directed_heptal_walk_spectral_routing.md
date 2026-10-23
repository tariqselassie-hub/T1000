# Directed 7-Node Graph Walk & Analytical Spectral Routing Engine

```
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.
```

---

## Executive Summary

This document specifies the exact mathematical derivation, invariant subspaces, closed-form eigenvalue spectrum, and dynamical simulation capabilities of the **7-Node Directed Graph Markov Chain and Continuous-Time Quantum Walk (CTQW) Routing Engine** implemented in [`crates/t1000/src/directed_heptal_walk.rs`](file:///c:/Users/zoddj/GuddaLM_workspace/T1000/crates/t1000/src/directed_heptal_walk.rs).

While the $[[7, 1, 3]]$ Steane-Wheel quantum error correction code in [`crates/t1000/src/qec/heptal_qudit.rs`](file:///c:/Users/zoddj/GuddaLM_workspace/T1000/crates/t1000/src/qec/heptal_qudit.rs) requires a fully symmetric, undirected 6-fold wheel graph $W_7$ to maintain distance-3 single-qudit Pauli syndrome decodability, this custom directed/weighted graph provides an asymmetric routing kernel characterized by:
1. **Exact Rational Stationary Distribution:** $\boldsymbol{\pi} = \frac{1}{42}(8, 5, 4, 6, 7, 4, 8)^T$.
2. **Instantaneous 1-Step Memory Nullspace ($\lambda_7 = 0$):** Annihilates differences between twin nodes $v_2$ and $v_5$.
3. **Decoupled Anti-Phase Rim Resonance ($\lambda = -1/2$, multiplicity 2):** Localized on the $v_3 - v_6$ rim 2-cycle and the symmetric quotient branch.
4. **All-Real Spectral Relaxation:** Despite strong directionality and asymmetry ($P \neq P^T$), all 7 eigenvalues are strictly real, eliminating oscillatory spiral phase rotation in classical Markov relaxation.
5. **Continuous-Time Quantum Walk (CTQW) Unitary Dynamics:** Enables coherent ballistic wavepacket transit with tunable chiral drift across the 7-node network.

---

## 1. Matrix Transcription & Graph Topology

The $7 \times 7$ row-stochastic transition matrix $P$ is defined over node basis $(v_1, v_2, v_3, v_4, v_5, v_6, v_7)$:

$$P = \begin{pmatrix}
0   & 1/4 & 1/4 & 0   & 0   & 1/4 & 1/4 \\
0   & 0   & 0   & 1/2 & 0   & 0   & 1/2 \\
1/2 & 0   & 0   & 0   & 0   & 1/2 & 0   \\
0   & 1/2 & 0   & 0   & 1/2 & 0   & 0   \\
0   & 0   & 0   & 1/2 & 0   & 0   & 1/2 \\
1/2 & 0   & 1/2 & 0   & 0   & 0   & 0   \\
1/2 & 0   & 0   & 0   & 1/2 & 0   & 0
\end{pmatrix}$$

```
┌────────────────────────────────────────────────────────────────────────┐
│                   7-NODE DIRECTED ROUTING TOPOLOGY                     │
│                                                                        │
│            [v3] <==== 1/2 ====> [v6]                                   │
│             ^ \                  ^ \                                   │
│         1/4 │  \ 1/2         1/4 │  \ 1/2                              │
│             │   v                │   v                                 │
│            [      v1 (Central Dispatch Hub)      ]                     │
│             │                    ^                                     │
│         1/4 │                1/2 │                                     │
│             v                    │                                     │
│            [v2] --- 1/2 ---> [v4] <--- 1/2 --- [v5]                    │
│             │                 │                 ^                      │
│         1/2 │                 │ 1/2             │ 1/2                  │
│             v                 v                 │                      │
│            [      v7 (Sink / Feedback Hub)       ]                     │
└────────────────────────────────────────────────────────────────────────┘
```

### Key Structural Symmetries
- **Row-Stochasticity:** Every row sums to $1.0$ ($\sum_j P_{ij} = 1.0$). This guarantees the principal stationary Markov eigenvalue $\lambda_1 = 1$.
- **Identical Twin Rows ($v_2$ and $v_5$):** Rows 2 and 5 are identical: $[0, 0, 0, 1/2, 0, 0, 1/2]$. Thus $\text{rank}(P) \le 6$, guaranteeing the null mode $\lambda_7 = 0$ with eigenvector $\mathbf{x} = \mathbf{e}_2 - \mathbf{e}_5$.
- **Decoupled 2-Cycle ($v_3$ and $v_6$):** Nodes $v_3$ and $v_6$ cross-feed each other with weight $1/2$ and feed the hub $v_1$ with weight $1/2$.
- **Bipartite-like Branch ($v_2, v_4, v_5, v_7$):** Forms a cyclic loop with alternating feedforward from $v_1$ and feedback to $v_1$.

---

## 2. Analytical Eigenvalue Spectrum

The characteristic polynomial factors into exact analytical components:

$$p(\lambda) = \det(\lambda I_7 - P) = \lambda (\lambda - 1)\left(\lambda + \frac{1}{2}\right)^2 \left(16\lambda^3 - 10\lambda + 1\right) = 0$$

### Eigenvalue Summary Table

| Mode | Exact Closed-Form Expression | Numerical Value | Multiplicity | Dynamical / Physical Interpretation |
| :--- | :--- | :---: | :---: | :--- |
| $\lambda_1$ | $1$ | $+1.000000$ | 1 | Stationary invariant Markov state ($\boldsymbol{\pi} P = \boldsymbol{\pi}$) |
| $\lambda_2$ | $\text{Root}_1(16\lambda^3 - 10\lambda + 1)$ | $+0.734809$ | 1 | Dominant sub-unity relaxation mode; sets mixing rate |
| $\lambda_3$ | $\text{Root}_2(16\lambda^3 - 10\lambda + 1)$ | $+0.101682$ | 1 | Intermediate radial exchange mode between hub and rim |
| $\lambda_4$ | $0$ | $0.000000$ | 1 | Nullspace projection mode: annihilates $v_2 - v_5$ in 1 step |
| $\lambda_5$ | $-1/2$ | $-0.500000$ | 2 | Anti-phase local rim oscillation ($\mathbf{e}_3 - \mathbf{e}_6$) and quotient branch |
| $\lambda_6$ | $\text{Root}_3(16\lambda^3 - 10\lambda + 1)$ | $-0.836491$ | 1 | High-frequency alternating rim/hub exchange mode |

### Spectral Gap & Mixing Time
The spectral gap of the transition matrix is:
$$\Delta = 1 - \lambda_2 \approx 1 - 0.734809 = 0.265191$$
The asymptotic mixing time to reach total variation distance $d_{\text{TV}} < \epsilon$ scales as:
$$\tau_{\text{mix}} \sim \frac{1}{\Delta} \approx 3.77 \text{ steps}$$

---

## 3. Stationary Probability Vector $\boldsymbol{\pi}$

Solving the global balance equations $\boldsymbol{\pi} P = \boldsymbol{\pi}$ subject to $\sum_{i=1}^7 \pi_i = 1$:

$$\boldsymbol{\pi} = \frac{1}{42} \begin{pmatrix} 8 & 5 & 4 & 6 & 7 & 4 & 8 \end{pmatrix}^T \approx \begin{pmatrix} 0.190476 \\ 0.119048 \\ 0.095238 \\ 0.142857 \\ 0.166667 \\ 0.095238 \\ 0.190476 \end{pmatrix}$$

### Node Occupancy Analysis
- **Hub & Feedback Concentrations:** The central hub $v_1$ and boundary sink $v_7$ accumulate the highest stationary probability ($\pi_1 = \pi_7 = 8/42 \approx 19.05\%$).
- **Intermediate Branch:** Node $v_5$ has higher occupancy than its twin $v_2$ ($\pi_5 = 7/42$ vs $\pi_2 = 5/42$) because $v_5$ receives incoming probability from both $v_4$ and $v_7$.
- **Rim 2-Cycle Buffer:** Nodes $v_3$ and $v_6$ maintain identical low occupancy ($\pi_3 = \pi_6 = 4/42 \approx 9.52\%$).

---

## 4. Continuous-Time Quantum Walk (CTQW) Hamiltonian

To model coherent quantum transport on this directed graph without violating unitarity, the engine constructs a Hermitian Hamiltonian parameterized by symmetric hopping kinetic energy $\gamma$ and chiral drift $\mu$:

$$H_{jk} = \frac{\gamma}{2} (P_{jk} + P_{kj}) + i \frac{\mu}{2} (P_{jk} - P_{kj})$$

Because $H = H^\dagger$, the time evolution $|\psi(t)\rangle = e^{-i H t} |\psi(0)\rangle$ strictly preserves the statevector $\ell_2$-norm:
$$\sum_{j=1}^7 |\psi_j(t)|^2 = 1.000000000000$$

Integration is implemented via a 4th-order Runge-Kutta (RK4) numerical integrator ([`DirectedHeptalQuantumWalk::step_rk4`](file:///c:/Users/zoddj/GuddaLM_workspace/T1000/crates/t1000/src/directed_heptal_walk.rs#L215-L255)), supporting localized state initialization, ballistic transport tracking, and Shannon entropy monitoring.

---

## 5. Implementation Verification

All mathematical invariants are unit-tested and verified in [`crates/t1000/src/directed_heptal_walk.rs`](file:///c:/Users/zoddj/GuddaLM_workspace/T1000/crates/t1000/src/directed_heptal_walk.rs):

```
$ cargo test -p t1000 --lib directed_heptal_walk -- --nocapture
running 7 tests
Exact cubic roots: [0.7348087170290185, 0.10168210689845253, -0.8364908239274711]
test directed_heptal_walk::tests::test_discrete_walk_convergence_to_stationary ... ok
test directed_heptal_walk::tests::test_rim_antiphase_eigenvalue ... ok
test directed_heptal_walk::tests::test_row_stochasticity ... ok
test directed_heptal_walk::tests::test_characteristic_polynomial_zeros ... ok
test directed_heptal_walk::tests::test_stationary_distribution_invariance ... ok
test directed_heptal_walk::tests::test_twin_row_nullspace_cancellation ... ok
test directed_heptal_walk::tests::test_ctqw_unitarity_and_entropy ... ok

test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; finished in 0.00s
```
