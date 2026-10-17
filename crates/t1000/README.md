# `t1000` — High-Performance Quantum Neural Network & Simulation Engine

[![Rust](https://img.shields.io/badge/rust-2021_edition-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)

`t1000` is the primary high-performance engine of the T-1000 system. Engineered in pure Rust for x86-64 AVX2+FMA vectorization, it scales statevector evolution, variational circuit differentiation, discrete 2D Stokes plaquette contractions, and chemical molecular latent space screening.

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
     t1000::simd    t1000::compiler t1000::adjoint  t1000::plaquette t1000::molfeatures
     AVX2+FMA Gate  Per-Qubit DAG   In-Place Adjoint Discrete 2D     OpenSMILES Parser
     Butterfly Core Fusion Sweeps   Differentiation  Stokes Engine   19-D Feature Maps
```

### Core Modules

| Module | Technical Specification |
| :--- | :--- |
| **[`simd`](src/simd/)** | Structure-of-Arrays (SOA) 64-bit complex amplitude buffers vectorized with 256-bit AVX2+FMA. Subdivided into modular domains (`guards`, `rotations`, `algebra`, `measurements`, `model`). |
| **[`compiler`](src/compiler/)** | Dependency DAG (`CircuitDag`), commutation analysis, stage grouping, and single-pass fused diagonal streaming sweeps ($R_z$, $CZ$). |
| **[`adjoint`](src/adjoint.rs)** | Single-pass analytic reverse-mode differentiation (`AdjointWorkspace`) with in-place costate accumulation eliminating heap churn. |
| **[`ansatz`](src/ansatz/)** | Hardware-Efficient Ansatz (HEA), bipartite Schmidt cut-bridges (`cut_bridge.rs`), and continuous dual-axis SU(2) re-uploading (`reupload.rs`). |
| **[`optimizer`](src/optimizer.rs)** | Two-Speed AdamW optimizer (`TwoSpeedHybridOptimizer`) and exact analytical correlator LayerNorm Jacobian backpropagation (`normalize_correlators_arr_backward`). |
| **[`plaquette`](src/plaquette/)** | Discrete 2D Stokes' theorem engine contracting planar lattices (e.g. 120-qubit IBM Nighthawk R2 and Kagome strips) with zero Stokes drift ($\Delta = 0.000000$) in 384-byte L1 cache windows. |
| **[`qec::vnni`](src/qec/vnni.rs)** | 32-way INT8 vector dot-product syndrome engine (`{{vex}} vpdpbusd`) evaluating surface code stabilizer checks at **$37.6\text{ Million decodings/sec}$**. |
| **[`heptal_qnn`](src/heptal_qnn.rs)** | Sovereign Radix-7 qudit ($\mathbb{Z}_7$) Matrix Product State (MPS) QNN for chemical descriptor space (784-byte L1 resident). |
| **[`molfeatures`](src/molfeatures.rs)** | Native OpenSMILES parser with 19 structural descriptors mapped to parameterized rotation angles $R_y(x_i)$. |

---

## Executable Binaries (`src/bin/`)

The crate organizes CLI binaries into four domain subdirectories:
* **`src/bin/training/`**: QNN variational training pipelines (`mnist_qnn_train`, `birds_hybrid_qnn_train`, `zinc_qnn_train`, `corpus_vqc_train`).
* **`src/bin/benchmarks/`**: Reproducible architectural showdowns (`fidelity_orthogonality_bench`, `path_a_vs_c_sorter_bench`, `zinc_scaled_14q_16q_bench`, `zinc_heptal_qnn_bench`).
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
