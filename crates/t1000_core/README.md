# `t1000_core` — Sovereign Mathematical Substrate for Quantum Simulation

[![Rust](https://img.shields.io/badge/rust-2021_edition-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)

`t1000_core` provides the pure mathematical substrate, quantum state registers, statistical verification tools, and open quantum system mechanics for the T-1000 Quantum Neural Network & Mimetic Simulation Engine.

Engineered with zero unsafe memory abstractions in its public interface and minimal dependencies (`num-complex`, `rand`, `rand_chacha`, `serde`), it functions as a lightweight, fully deterministic foundational library.

---

## Subsystem Architecture

| Module | Purpose & Mathematical Role |
| :--- | :--- |
| **[`register`](src/register.rs)** | Multi-radix quantum state vectors (`QubitRegister`, `QuditRegister`) with normalization invariants and support for prime Galois field qudits ($\mathbb{Z}_d$). |
| **[`porter_thomas`](src/porter_thomas.rs)** | Quantum chaos and quantum supremacy verification: Cross-Entropy Benchmarking (linear XEB), Kolmogorov-Smirnov test statistics against Haar-random distributions, Participation Ratio, and Shannon entropy deficits. |
| **[`partial_trace`](src/partial_trace.rs)** | Subsystem reduction: Traces out subsystem $B$ from bipartite states $\mathcal{H}_A \otimes \mathcal{H}_B$, returning exact density matrices $\rho_A = \text{Tr}_B(|\psi\rangle\langle\psi|)$. |
| **[`channel`](src/channel.rs)** | Open quantum system dynamics: Krauss-form channels including Amplitude Damping ($T_1$), Phase Damping ($T_2$), Depolarizing, and Pauli noise. |
| **[`povm`](src/povm.rs)** | Positive Operator-Valued Measure (POVM) generalized measurement operators with soft continuous probabilities. |
| **[`hamiltonian`](src/hamiltonian.rs)** | Unitary time evolution $\exp(-i H \Delta t)$ for diagonal Hamiltonians. |
| **[`phase_gate`](src/phase_gate.rs)** | Parameterized global phase gates $\text{diag}(e^{i\theta_0}, \dots, e^{i\theta_{d-1}})$. |
| **[`interference`](src/interference.rs)** | Quantum state interference scoring: Constructive, destructive, and net interference amplitude tracking. |
| **[`simulated_bifurcation`](src/simulated_bifurcation.rs)** | Simulated Quantum Bifurcation Machine (SBM) algorithms for combinatorial Ising optimization. |

---

## Verification & Testing

Run all unit tests in the core substrate:
```bash
cargo test -p t1000_core
```

---

## Legal & Dual Licensing

Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

Dual-licensed under either:
* Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
* MIT License ([LICENSE-MIT](../../LICENSE-MIT))
