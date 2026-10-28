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
| **[`ansatz`](src/ansatz/)** | Hardware-Efficient Ansatz (HEA), bipartite Schmidt cut-bridges (`cut_bridge.rs`), 36-qubit virtual cut engine (`virtual_cut_engine.rs`), and continuous dual-axis SU(2) re-uploading (`reupload.rs`). |
| **[`optimizer`](src/optimizer.rs)** | Two-Speed AdamW optimizer (`TwoSpeedHybridOptimizer`) and exact analytical correlator LayerNorm Jacobian backpropagation (`normalize_correlators_arr_backward`). |
| **[`plaquette`](src/plaquette/)** | Discrete 2D Stokes' theorem engine contracting planar lattices (e.g. 120-qubit IBM Nighthawk R2 and Kagome strips) with zero Stokes drift ($\Delta = 0.000000$) in 384-byte L1 cache windows. |
| **[`qec::vnni`](src/qec/vnni.rs)** | 32-way INT8 vector dot-product syndrome engine (`{{vex}} vpdpbusd`) evaluating surface code stabilizer checks at **$37.6\text{ Million decodings/sec}$**. |
| **[`qec::lattice_surgery`](src/qec/lattice_surgery.rs)** | Dual-patch rotated surface code lattice surgery (Z-merge and X-merge) generating logical Bell pairs $|\Phi^+\rangle_L$ ($F = 1.000000000000$). |
| **[`qec::magic_distillation`](src/qec/magic_distillation.rs)** | Bravyi-Kitaev 15-to-1 Reed-Muller code magic state distillation achieving $1,145.4\times$ cubic error suppression ($\epsilon_{\text{out}} = 35\epsilon_{\text{in}}^3$). |
| **[`qec::heptal_qudit`](src/qec/heptal_qudit.rs)** | [[7, 1, 3]] Steane-Wheel ($W_7$) code with hub mediator and bitwise transversal logical $\text{CZ}_L = \bigotimes_{k=0}^6 \text{CZ}(k, k+7)$ in $\mathcal{O}(1)$ depth. |
| **[`qec::magic_w7_injection`](src/qec/magic_w7_injection.rs)** | Eastin-Knill bypass: teleported non-Clifford logical $T_L$ injection into active in-register physical ancillas with adaptive Clifford $S_L$ phase frame updates. |
| **[`chemistry`](src/chemistry/)** | Ab initio molecular quantum chemistry: 50x50 Coulomb eigensystem (`coulomb.rs`), continuous topological walk on molecular graph Laplacians (`topological_walk.rs`), atomic extensive energy decoupling (`baseline.rs`), and extended XYZ QM9 streaming ingestion (`qm9.rs`). |
| **[`datasets`](src/datasets/)** | Multi-domain dataset loaders and streaming adapters: MoleculeNet QM9 (`qm9.rs`), Pitt Quantum 106k (`pitt.rs`), Bank Fraud Graph (`bank_fraud.rs`), IoT Botnet telemetry (`iot_botnet.rs`), and Harmonized Shipments (`harmonized.rs`). |
| **[`heptal`](src/heptal/)** | Sovereign Radix-7 qudit ($\mathbb{Z}_7$) state spaces: Matrix Product State QNN (`qnn.rs`) and continuous complementarity wave engines (`wave.rs`). |
| **[`photonic_interferometer`](src/photonic_interferometer/)** | 16-mode linear optical interferometer & multi-photon Fock state compiler with pure 50:50 directional couplers, Glynn permanents, HOM quantum bunching, and SOH electro-optic modulation. |
| **[`trees`](src/trees/)** | Continuous-Time Quantum Walks (CTQW) on welded binary trees (`glued_trees.rs`), multi-exit topological tree valves (`glued_tree_valve.rs`), cascaded deep routers (`cascaded_tree_valve.rs`), ultrametric QMUX sorters (`ultrametric_tree.rs`), and 7-node directed Szegedy walks (`directed_heptal_walk.rs`). |
| **[`zinc`](src/zinc/)** | High-throughput ZINC molecular screening & variational training suite: dataset loaders (`data.rs`), holdout scoring (`holdout_eval.rs`), fidelity kernels (`kernel_bench.rs`), linear readout fits (`readout_fit.rs`), and re-uploading models (`reuploading_model.rs`). |
| **[`molfeatures`](src/molfeatures.rs)** | Native OpenSMILES parser with 19 structural descriptors mapped to parameterized rotation angles $R_y(x_i)$. |
| **[`q_krr`](src/q_krr.rs)** | Quantum Kernel Ridge Regression (Q-KRR) computing exact parameter-free state fidelity kernels $k(x_i, x_j) = |\langle \psi(x_i)|\psi(x_j)\rangle|^2$ and parallel Rayon Cholesky linear solves bypassing barren plateaus. |
| **[`backend`](src/backend.rs)** | Consolidated backend execution engine (`BackendKind`) routing across CPU AVX2 FP64, OpenCL iGPU FP32, Artix-7 1000_QPU on `COM6`, fixed-point QPU model, and photonic meshes. |
| **[`qpu`](src/qpu/)** | 14-qubit fixed-point statevector co-processor specification: bit-exact model (`model.rs`), Q8.24/Q2.23 arithmetic (`fixed.rs`), UCCSD gate compiler (`compile.rs`), and protocol (`protocol.rs`). |


---

## Executable Binaries (`src/bin/`)

The crate organizes CLI binaries into four domain subdirectories:
* **`src/bin/training/`**: QNN variational training and kernel regression pipelines (`pitt_qkrr_train`, `pitt_vqc_train`, `qm9_magic_w7_ft_train`, `qm9_transversal_w7_train`, `qm9_fault_tolerant_train`, `qm9_vqc_train`, `mnist_qnn_train`, `mnist_ultrametric_sorter`, `birds_hybrid_qnn_train`, `zinc_qnn_train`).
* **`src/bin/benchmarks/`**: Reproducible architectural showdowns (`molecular_eigensystem_bench`, `qgnn_molecular_bench`, `vnni_threshold_bench`, `full_qnn_suite_bench`, `glued_trees_bench`, `glued_tree_valve_bench`, `cascaded_tree_valve_bench`, `ultrametric_tree_sorter_bench`, `fidelity_orthogonality_bench`, `path_a_vs_c_sorter_bench`, `zinc_scaled_14q_16q_bench`, `zinc_heptal_qnn_bench`).
* **`src/bin/measurements/`**: Quantum hardware telemetry and algorithms (`virtual_36q_cut_bridge_bench`, `frontier_rcs`, `vnni_syndrome_bench`, `cut_bridge_chsh_bench`, `coherence_frontier_push`, `nighthawk_r2_bench`).
* **`src/bin/screening/`**: High-throughput chemical discovery (`continuous_walk_qm9_filter`, `zinc_dual_search`, `zinc_250k_screen`, `eval_single_mol`).

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
