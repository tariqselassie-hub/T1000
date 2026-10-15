# T-1000: Quantum Neural Network & Mimetic Circuit Simulator

[![Rust](https://img.shields.io/badge/rust-2021_edition-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)
[![Build Status](https://img.shields.io/badge/status-active%20development-brightgreen.svg)](#)

> *"Like the liquid metal assassin, T-1000 adapts and mimics any quantum device or classical architecture. Feed it parameters; it handles the rest."*

`T-1000` is an ultra-high-performance Variational Quantum Circuit (VQC) and Quantum Neural Network engine engineered in pure Rust. Built for extreme SIMD throughput, arbitrary circuit DAG scheduling, and chemical latent space exploration, it runs completely standalone with zero legacy bloat.

---

## Architecture Overview

```
                                      T-1000
                        ┌─────────────────────────────────┐
                        │   Shape-Shifting Quantum Core   │
                        └───────────────┬─────────────────┘
                                        │
         ┌──────────────────────────────┼──────────────────────────────┐
         ▼                              ▼                              ▼
  SIMD Acceleration             Circuit Compiler              Chemical Exploration
  • 64-bit complex SOA          • Per-qubit DAG scheduling    • OpenSMILES parser
  • AVX2 + FMA Ry / Rz kernels  • Gate & stage fusion         • 19-dim quantum encoding
  • Fused diagonal CZ passes    • Disjoint wire execution     • ZINC 250k drug screening
  • Multi-threaded (Rayon)      • Commutation analysis        • Hilbert-space fidelity
```

### Workspace Crates

| Crate | Purpose |
| :--- | :--- |
| **[`t1000_core`](crates/t1000_core)** | Pure mathematical substrate: state registers, partial trace, POVM projections, Hamiltonian evolution, Porter-Thomas chaos metrics. |
| **[`t1000`](crates/t1000)** | The high-performance QNN engine: SIMD AVX2 kernels, circuit DAG compiler, SMILES molecular encoder, parameter-shift optimizers, and benchmarks. |

---

## Key Capabilities

### 1. Shape-Shifting Circuit Compilation & Scheduling (`t1000::compiler`)
* **Per-qubit dependency DAG (`CircuitDag`):** Tracks gate commutation and dependencies along wire order.
* **Gate Fusion:** Greedy grouping of non-overlapping single-qubit rotations and multi-qubit diagonal interactions into unified stages.
* **Fused Diagonal Kernel:** Executes entire layers of $R_z$ and $CZ$ gates in a single streaming memory sweep over the statevector.

### 2. AVX2 + FMA SIMD Statevector Engine (`t1000::simd`, `t1000::tiled_butterfly`)
* **Structure-of-Arrays (SOA):** Separated contiguous arrays of double-precision real and imaginary amplitudes.
* **L2-Resident Tiled Butterfly Kernels:** Cache-conscious stride access for high qubit indices ($q \ge 16$).
* **Beats Industry Baselines:** Up to **1.9x faster than Qiskit Aer** at 8 threads on deep Random Circuit Sampling benchmarks.

### 3. SMILES Molecular Exploration (`t1000::smiles`, `t1000::molfeatures`)
* **Chemical Featurizer:** Native parser for OpenSMILES strings generating 19 molecular descriptors (atom counts, aromaticity, ring closures, molecular weight, heteroatom ratio).
* **Quantum Feature Map:** Direct parameterized injection into Hilbert space ($|\psi(\text{SMILES})\rangle$) via parameterized $R_y(x_i)$ rotations.
* **Drug Discovery Screening:** Scalable screening across molecular repositories like ZINC 250k.

### 4. Statistical Quantum Verification (`t1000::rcs`)
* **Frontier RCS Benchmark (`frontier_rcs`):** Deep Random Circuit Sampling (default: 14 qubits, 36 layers, 1,513 gates) benchmarking the optimized 256-bit AVX2 engine against a textbook scalar reference simulator and Qiskit Aer.
* **Dual-Engine Precision:** Machine precision numerical agreement (fidelity $1.000000000000$, max amplitude error $< 10^{-16}$).
* **Cross-Entropy Benchmarking (XEB):** Validates linear XEB convergence ($F_{\text{XEB}} \approx 1.0101$) against ideal Haar-random Porter-Thomas distributions.
* **Kolmogorov-Smirnov & Entropy Tests:** Automated confirmation of Porter-Thomas chaos ($D_{\text{KS}} = 0.0045 < D_{\text{critical}} = 0.0106$, $\Delta S = 0.4243\text{ nats} \to 1 - \gamma$).
* **Adjoint Differentiation:** Evaluates analytical gradients across 1,008 parameters in $86.5\text{ ms}$ ($231\times$ speedup over parameter shift).
* **Coherence Frontier Push (`coherence_frontier_push`):** Dual-mark physical experiment tracking quantum $\ell_1$-norm coherence capacity ($C_{\text{norm}}$) up to 26 qubits dense ($2^{26}$ amplitudes, 1.02 GB RAM), 64 qubits via 2D tensor network belief propagation ($2^{64}$ states), and sovereign Heptal wave qudits ($7^9$ states, $d=7$).

### 5. Quantum Vision & BNN Benchmark (`mnist_qnn_train`)
* **Amplitude Encoding:** Exact injection of 784-dimensional image spaces directly into a 10-qubit Hilbert space ($2^{10} = 1024$ amplitudes).
* **Bipartite Entanglement Observables:** 55 quantum expectation operators (10 single-qubit $\langle Z_i \rangle$ + 45 two-body $\langle Z_i Z_j \rangle$ correlators) with analytical reverse-mode `adjoint_gradient`.
* **Surpasses Classical BNN & HDC Baselines:**

| Architecture Model | Test Accuracy % | Space Representation | State Coherence $C_{\ell_1}$ | Throughput |
| :--- | :---: | :---: | :---: | :---: |
| **GuddaNetworks BNN Baseline (BSC)** | 78.59% | 4096-bit Binary Sparse Coding | Classical | Classical BSC |
| **GuddaNetworks BNN Baseline (MAP)** | 79.40% | 4096-dim Multiply-Add-Permute | Classical | Classical MAP |
| **GuddaNetworks HDC-10k Dimension** | 81.25% | 10,000-bit BSC Space | Classical | Classical BSC |
| **T-1000 10-Qubit VQC (AVX2)** | **84.50%** | **1024-dim ($2^{10}$) Hilbert Space** | **350.94** | **34.38 $\mu$s / sample** |

---

## Quick Start for Developers

Clone and build the entire workspace with standard Cargo tools:

```bash
git clone https://github.com/tariqselassie-hub/T1000.git
cd T1000

# Run all unit, integration, and oracle verification tests
cargo test

# Run 10-qubit MNIST Quantum Neural Network & Coherence Benchmark
cargo run --release -p t1000 --bin mnist_qnn_train

# Execute 14-qubit, 36-depth Random Circuit Sampling frontier benchmark
cargo run --release -p t1000 --bin frontier_rcs -- --qubits 14 --depth 36

# Run sovereign Quantum Coherence Frontier Push (Dense 26Q + TN 64Q + Heptal 7^9 Qudits)
cargo run --release -p t1000 --bin coherence_frontier_push

# Run chemical SMILES quantum featurization & screening
cargo run --release -p t1000 --bin zinc_protocol

# Generate all scientific research charts and telemetry figures
python scripts/charting/chart_all.py
```

---

## Architecture Governance

Autonomous agents and contributors must follow the sovereign engineering, visual palette, and dual-licensing governance defined in [AGENTS.md](AGENTS.md).

---

## Contributing & Development Roadmap

We are actively welcoming quantum computing researchers, Rust performance engineers, and machine learning architects to collaborate:

- [ ] **AVX-512 & ARM NEON Kernels:** Expanding SIMD kernels beyond AVX2 for Apple Silicon (M-series) and modern server CPUs.
- [ ] **GPU Backend:** WGPU / CUDA kernels for scaling beyond 26 qubits.
- [ ] **Clifford+T Decomposition:** Automated synthesis of continuous rotation angles into discrete fault-tolerant gate sets.
- [ ] **Graph Neural Network (GNN) Hybridization:** Fusing T-1000 latent molecular embeddings with classical GNN message passing.

Feel free to open issues, submit pull requests, or share benchmark results on different architectures!

---

## License & Attribution

Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

Dual-licensed under either of:
* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option. Unless explicitly stated otherwise, any contribution intentionally submitted for inclusion in this work by you shall be dual licensed as above, without any additional terms or conditions.
