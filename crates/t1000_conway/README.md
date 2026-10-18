# `t1000_conway` — Sovereign Quantum Conway Game of Life & Fractional Fermi Sea QCA Engine

[![Rust](https://img.shields.io/badge/rust-2021_edition-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)

`t1000_conway` is an isolated, high-performance Unitary Quantum Cellular Automata (QCA) engine engineered in pure Rust. It implements a non-dissipative formulation of Conway's Game of Life (B3/S23) driven by Fractional Fermi Seas (FFS), Confinement-Induced Resonances (CIR), and Haldane Generalized Exclusion Statistics (GES).

All cellular automaton topology, Moore neighborhood accumulators, pattern factories, and holonomy transfer matrices are strictly quarantined within this crate to keep the primary VQC and chemical screening engines in `t1000` completely unpolluted.

---

## Architecture & Subsystems

```
                                 crates/t1000_conway
                         ┌─────────────────────────────────┐
                         │   Quantum Cellular Automata     │
                         └───────────────┬─────────────────┘
                                         │
          ┌───────────────┬──────────────┼───────────────┬───────────────┐
          ▼               ▼              ▼               ▼               ▼
    t1000_conway::   t1000_conway:: t1000_conway:: t1000_conway:: t1000_conway::
       lattice          state          holonomy       simulator       telemetry
    2D Toroidal    SoA Amplitudes   CIR Resonance  Split-Step QCA  Deen OS Matrix
    Moore Neighbor (|0>, |1>, |2>)  Holonomy Gates Exact Unitarity & Visualizer
```

### Core Modules

| Module | Technical Specification |
| :--- | :--- |
| **[`lattice`](src/lattice.rs)** | 2D Toroidal and Open lattice geometry, flat index conversion, precomputed $O(1)$ 8-neighbor Moore tables, and Margolus $2 \times 2$ block plaquettes. |
| **[`state`](src/state.rs)** | Structure-of-Arrays (SoA) double-precision real and imaginary buffers for 3 local basis sectors: Vacuum $|0\rangle$, Living Fermi $|1\rangle$, and Super-Fermionic FFS $|2\rangle$. |
| **[`holonomy`](src/holonomy.rs)** | Unitary $3 \times 3$ interaction holonomy transfer matrices across Confinement-Induced Resonances ($g_{1D}: 0 \to +\infty \to -\infty \to 0$), accumulating geometric Berry phases $\gamma = \pi/2$ and realizing Haldane exclusion plateaus $n(k) = 1/\ell = 0.5000$. |
| **[`rules`](src/rules.rs)** | Classical Conway B3/S23 reference execution engine and continuous Moore neighborhood density accumulation $\bar{N}_i = \sum_{j \in \text{Moore}(i)} \langle \hat{n}_j \rangle$. |
| **[`patterns`](src/patterns.rs)** | Canonical pattern injection: Bethe-string glider solitons, period-2 blinkers/toads, $2 \times 2$ still-life blocks, and Schrödinger Cat superposition blinkers $(|H\rangle + |V\rangle)/\sqrt{2}$. |
| **[`simulator`](src/simulator.rs)** | Split-step checkerboard decomposition ($U_{\text{step}} = U_{\mathcal{B}} \cdot U_{\mathcal{A}}$) preserving global spatial reversibility and machine-precision unitarity ($\le 8.88 \times 10^{-16}$). |
| **[`telemetry`](src/telemetry.rs)** | Deen OS Sovereign ANSI terminal visualizer rendering obsidian backdrops, laser emerald live cells, and radiant amber FFS excitations. |

---

## Executable Driver & Interactive Visualizer

### 1. Bare-Metal Rust Benchmark Binary
Execute the standalone simulation and benchmark driver:
```bash
cargo run -p t1000_conway --bin qgol_sim
```
Runs 3 sovereign scenarios (Glider Soliton, FFS Overpopulation Protection, and Cat Blinker), outputting real-time ANSI matrices and logging structured telemetry to `results/experiments/qgol_simulation_telemetry.json`.

### 2. Single-View Web Visualizer Deck
A dedicated GPU-accelerated client-side visualizer is available in `web/`:
* **Direct URL:** Open `web/qgol.html` or visit `http://localhost:8000/qgol.html`
* **Single-View Switcher:**
  * **View 1 (Key `1`):** 3D Isometric Columns on Substrate with rotating phase needles $\theta = \arg(c_1)$.
  * **View 2 (Key `2`):** Dual-Domain Wavefield with Friedel ripples and Reciprocal Momentum Sea $n(k) = 0.5000$.
  * **View 3 (Key `3`):** Confined 3D Hilbert Space Chamber ($\mathbb{C}P^2$ Bloch simplex) with CIR geodesic orbits.

---

## Verification & Testing

Execute all unit, integration, and unitarity verification tests:
```bash
cargo test -p t1000_conway
```

---

## Legal & Dual Licensing

Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

Dual-licensed under either:
* Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
* MIT License ([LICENSE-MIT](../../LICENSE-MIT))
