<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Terrence A. Jones SR. All rights reserved. -->

# `t1000` — High-Performance Quantum Neural Network & Simulation Engine

[![Rust](https://img.shields.io/badge/rust-2021_edition-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)

`t1000` is the primary high-performance engine of the T-1000 system. Engineered in pure Rust for x86-64 AVX2+FMA vectorization, it scales statevector evolution, variational circuit differentiation, discrete 2D Stokes plaquette contractions, fault-tolerant quantum error correction, and molecular latent space exploration.

---

## Architecture & Subsystems

```
                                    crates/t1000
                        ┌─────────────────────────────────┐
                        │   Shape-Shifting Quantum Core   │
                        └───────────────┬─────────────────┘
                                        │
         ┌───────────────┬──────────────┼───────────────┬───────────────┐
         ▼               ▼              ▼               ▼               ▼
     t1000::simd    t1000::compiler t1000::qec      t1000::chemistry t1000::adjoint
     AVX2+FMA Gate  Per-Qubit DAG   VNNI Syndrome   Coulomb Eigensys In-Place Adjoint
     Butterfly Core Fusion Sweeps   Lattice Surgery QM9 Ingestion    Differentiation
```

### Core Modules

| Module | Technical Specification |
| :--- | :--- |
| **[`simd`](src/simd/)** | Structure-of-Arrays (SOA) 64-bit complex amplitude buffers vectorized with 256-bit AVX2+FMA. Subdivided into modular domains (`guards`, `rotations`, `algebra`, `measurements`, `model`, `eigensystem`). |
| **[`compiler`](src/compiler/)** | Dependency DAG (`CircuitDag`), commutation analysis, stage grouping, and single-pass fused diagonal streaming sweeps ($R_z$, $CZ$). |
| **[`adjoint`](src/adjoint.rs)** | Single-pass analytic reverse-mode differentiation (`AdjointWorkspace`) with in-place costate accumulation eliminating heap churn. |
| **[`ansatz`](src/ansatz/)** | Hardware-Efficient Ansatz (HEA), bipartite Schmidt cut-bridges (`cut_bridge.rs`), and continuous dual-axis SU(2) re-uploading (`reupload.rs`). |
| **[`optimizer`](src/optimizer.rs)** | Two-Speed AdamW optimizer (`TwoSpeedHybridOptimizer`) and exact analytical correlator LayerNorm Jacobian backpropagation (`normalize_correlators_arr_backward`). |
| **[`plaquette`](src/plaquette/)** | Discrete 2D Stokes' theorem engine contracting planar lattices (e.g. 120-qubit IBM Nighthawk R2 and Kagome strips) with zero Stokes drift ($\Delta = 0.000000$) in 384-byte L1 cache windows. |
| **[`qec::vnni`](src/qec/vnni.rs)** | 32-way INT8 vector dot-product syndrome engine (`{{vex}} vpdpbusd`) evaluating surface code stabilizer checks at **$37.6\text{ Million decodings/sec}$**. |
| **[`qec::lattice_surgery`](src/qec/lattice_surgery.rs)** | Dual-patch rotated surface code lattice surgery (Z-merge and X-merge) generating logical Bell pairs $|\Phi^+\rangle_L$ ($F = 1.000000000000$). |
| **[`qec::magic_distillation`](src/qec/magic_distillation.rs)** | Bravyi-Kitaev 15-to-1 Reed-Muller code magic state distillation achieving $1,145.4\times$ cubic error suppression ($\epsilon_{\text{out}} = 35\epsilon_{\text{in}}^3$). |
| **[`qec::heptal_qudit`](src/qec/heptal_qudit.rs)** | [[7, 1, 3]] Steane-Wheel ($W_7$) code with hub mediator and bitwise transversal logical $\text{CZ}_L = \bigotimes_{k=0}^6 \text{CZ}(k, k+7)$ in $\mathcal{O}(1)$ depth. |
| **[`chemistry`](src/chemistry/)** | Ab initio molecular quantum chemistry: 50x50 Coulomb eigensystem (`coulomb.rs`), atomic extensive energy decoupling (`baseline.rs`), and extended XYZ QM9 streaming ingestion (`qm9.rs`). |
| **[`qgnn`](src/qgnn.rs)** | Lie-algebraic geometric Quantum Graph Neural Network with polynomial dynamical Lie algebra $\dim(\mathfrak{g}) \sim \mathcal{O}(N)$ guaranteeing barren plateau resilience. |
| **[`heptal_qnn`](src/heptal_qnn.rs)** | Sovereign Radix-7 qudit ($\mathbb{Z}_7$) Matrix Product State (MPS) QNN for chemical descriptor space (784-byte L1 resident). |
| **[`ultrametric_tree`](src/ultrametric_tree.rs)** | Multi-register hierarchical Quantum Multiplexor (QMUX) sorter ($\mathcal{H} = \mathcal{H}_C \otimes \mathcal{H}_F \otimes \mathcal{H}_E$) enforcing ultrametric geometry and parallel branch parameter shifts. |
| **[`glued_trees`](src/glued_trees.rs)** | Continuous-Time Quantum Walk (CTQW) engine on welded binary trees (Childs et al.) demonstrating exponential quantum speedup via column-symmetric subspace invariance. |
| **[`glued_tree_valve`](src/glued_tree_valve.rs)** | Multi-Exit Topological Tree Valve & Quantum Router ($K$-way welded tree) utilizing Aharonov-Bohm phase holonomy for ballistic routing. |
| **[`cascaded_tree_valve`](src/cascaded_tree_valve.rs)** | Cascaded Multi-Tier Glued-Tree Valve & Deep Router demonstrating multi-stage coherent routing through sequential dual welds without measurement collapse. |
| **[`molfeatures`](src/molfeatures.rs)** | Native OpenSMILES parser with 19 structural descriptors mapped to parameterized rotation angles $R_y(x_i)$. |

---

## Executable Binaries (`src/bin/`)

The crate organizes CLI binaries into four domain subdirectories:
* **`src/bin/training/`**: QNN variational training pipelines (`qm9_transversal_w7_train`, `qm9_fault_tolerant_train`, `qm9_vqc_train`, `mnist_qnn_train`, `mnist_ultrametric_sorter`, `birds_hybrid_qnn_train`, `zinc_qnn_train`).
* **`src/bin/benchmarks/`**: Reproducible architectural showdowns (`molecular_eigensystem_bench`, `qgnn_molecular_bench`, `vnni_threshold_bench`, `full_qnn_suite_bench`, `glued_trees_bench`, `glued_tree_valve_bench`, `cascaded_tree_valve_bench`, `ultrametric_tree_sorter_bench`, `fidelity_orthogonality_bench`, `path_a_vs_c_sorter_bench`, `zinc_scaled_14q_16q_bench`, `zinc_heptal_qnn_bench`).
* **`src/bin/measurements/`**: Quantum hardware telemetry and algorithms (`frontier_rcs`, `vnni_syndrome_bench`, `cut_bridge_chsh_bench`, `coherence_frontier_push`, `nighthawk_r2_bench`).
* **`src/bin/screening/`**: High-throughput chemical discovery (`zinc_dual_search`, `zinc_250k_screen`, `eval_single_mol`).

---

## Verification & Testing

Execute all unit, integration, and mathematical oracle tests:
```bash
cargo test -p t1000 -- --test-threads=4
```

---

## Legal & Dual Licensing

Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

Dual-licensed under either:
* Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
* MIT License ([LICENSE-MIT](../../LICENSE-MIT))
