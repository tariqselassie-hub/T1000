# AGENTS.md — T-1000 Quantum Neural Network & Mimetic Simulation Engine

This document defines the sovereign technical specifications, architectural standards, visual palette, and legal attribution governance for all autonomous agents, human engineers, and subagents operating within the `T-1000` workspace.

---

## 1. System Identity & Mission

`T-1000` is an ultra-high-performance Variational Quantum Circuit (VQC) and Quantum Neural Network engine engineered in pure Rust. Built for extreme SIMD throughput, arbitrary circuit DAG scheduling, and chemical latent space exploration, it runs completely standalone with zero legacy bloat.

### Dual-Mission Core
1. **Quantum Circuit & RCS Simulation:** Arbitrary circuit compilation (`t1000::compiler`), fused AVX2+FMA state evolution (`t1000::simd`, `t1000::tiled_butterfly`), and Random Circuit Sampling (`t1000::rcs`) with rigorous Porter-Thomas chaos and Cross-Entropy Benchmarking (XEB).
2. **Molecular Latent Space Exploration:** Native OpenSMILES parsing, 19-dimensional quantum feature map injection into Hilbert space ($|\psi(\text{SMILES})\rangle$), and high-throughput molecular screening (e.g., ZINC 250k).

---

## 2. Technology Stack & Architectural Standards

| Subsystem | Technology Choice | Technical Specification |
| :--- | :--- | :--- |
| **Language & Toolchain** | Rust (2021 Edition) | Pure Rust workspace; zero unvetted dependencies; strict `clippy` and `cargo test` hygiene. |
| **SIMD Vectorization** | 256-bit AVX2 + FMA | Structure-of-Arrays (SOA) separated double-precision real and imaginary amplitude buffers. |
| **Cache Architecture** | L2 Tiled Butterfly Kernels | Cache-conscious 64-byte aligned blocks optimizing stride access for deep registers ($q \ge 16$). |
| **Multithreading** | Rayon 1.x | Task-stealing threadpool across non-overlapping wire blocks and statevector chunks. |
| **Circuit Compilation** | Dependency DAG (`CircuitDag`) | Wire-order gate commutation analysis, greedy stage grouping, and single-pass fused diagonal streaming sweeps ($R_z$, $CZ$). |
| **Quantum Mathematics** | `t1000_core` | POVM generalized measurements, partial trace, Hamiltonian time-evolution, Kolmogorov-Smirnov statistics. |
| **Chemical Featurizer** | `t1000::molfeatures` | OpenSMILES parser with 19 structural descriptors mapped to parameterized rotation angles $R_y(x_i)$. |

---

## 3. Sovereign Visual Palette & Aesthetics

All generated charts, visual rendering scripts, SVG diagrams, Matplotlib/Python tools, web frontends, and terminal telemetry MUST strictly follow the Deen OS Sovereign Visual Specification.

### Approved Color Palette
| Visual Role | Color Name | Hex Code | Usage |
| :--- | :--- | :--- | :--- |
| **Root Canvas** | Deep Obsidian | `#05080d` / `#080c14` | Viewport backdrops, root canvases, terminal background |
| **Card / Contrast** | Void Charcoal | `#0f172a` / `#111827` | Subplot panels, card backdrops, reticle containers |
| **Quantum Telemetry** | Laser Emerald Green | `#00e676` / `#00ff88` / `#10b981` | State amplitudes, wavefunctions, quantum fidelity, positive flux |
| **Hardware / FPGA** | Radiant Amber / Gold | `#ffb300` / `#ffd700` | Clock cycles, memory bus, registers, attestation hashes |
| **Alert / Threshold** | Crimson Red | `#ef4444` / `#dc2626` | Bottlenecks, decoherence spikes, threshold alerts |
| **Grid / Reticles** | Muted Slate Grid | `#1e293b` / `#334155` | Coordinate axes, crosshairs, wire dividers |
| **Monospace Readout**| Polar Slate / White | `#f8fafc` / `#e2e8f0` | Numeric readouts, labels, table headers |

### Absolute Aesthetic Rules
- **PROHIBITED COLORS (Zero Tolerance):** NEVER use cyan, teal, or sky blue (`#00ffff`, `#00f0ff`, `#38bdf8`, `#06b6d4`, `#0ea5e9`, `#0284c7`). NEVER use purple, violet, or magenta (`#a855f7`, `#8b5cf6`, `#9333ea`, `#d946ef`). NEVER use pastel or neon party tints.
- **NO EMOJIS:** Absolutely zero emojis in source code, commit messages, terminal banners, markdown reports, or documentation.
- **Unicode Box Drawing:** Use clean single-line or double-line box-drawing characters (`┌─┐`, `║═║`, `├─┤`) for all structured tables, banners, and diagrams.

---

## 4. Legal Attribution & Dual Licensing Governance

`T-1000` is dual-licensed under both the **MIT License** and the **Apache License, Version 2.0**.

### Attribution & Copyright Entity
- **Copyright Holder:** `Terrence A. Jones SR`
- **Copyright Year:** `2026`
- **Canonical Repository:** `https://github.com/tariqselassie-hub/T1000`

### Mandatory Source File Header
Every newly created Rust source file (`.rs`), script, or core module MUST include the standard SPDX header and copyright notice at the top of the file:

```rust
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.
```

### License Files & Manifests
- [LICENSE-MIT](LICENSE-MIT): Standard MIT License text with copyright to `Terrence A. Jones SR`.
- [LICENSE-APACHE](LICENSE-APACHE): Full text of Apache License, Version 2.0 with copyright notice.
- [NOTICE](NOTICE): Apache 2.0 mandatory attribution document attributing development to `Terrence A. Jones SR`.
- `Cargo.toml`: Root and subcrate manifests MUST specify `license = "MIT OR Apache-2.0"` and `authors = ["Terrence A. Jones SR"]`.

---

## 5. Engineering & Operational Directives

1. **Live Disk-Grounded Verification:** Never claim a build or test passes without executing `cargo check --workspace` and `cargo test --workspace`. Always report actual benchmark numbers directly from stdout or disk logs.
2. **Action-First Execution:** Implement full, working, drop-in solutions without placeholders, mock stubs, or unfinished `todo!()` blocks in production crates.
3. **Zero-Warning Hygiene:** Code must compile cleanly with zero compiler warnings and zero broken documentation links.
4. **Performance Integrity:** Avoid unnecessary allocations, clone calls on statevector buffers, or dynamic dispatch on hot inner-loop butterfly kernels.
