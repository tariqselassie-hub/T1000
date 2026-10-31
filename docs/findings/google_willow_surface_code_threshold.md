# T1000-FINDING-2026-039: Google Willow 105Q Surface-Code Threshold Mapping & Plaquette Teleportation Benchmark

> **Audit note (2026-10-30).** Simulated with pure Rust AVX2+FMA density matrix evolution based on empirical Google Willow calibration figures ($p_1 = 0.035\%$, $p_2 = 0.14\%$ RCS / $0.33\%$ QEC, readout error $0.67\%$).

**Author:** Terrence A. Jones SR  
**Date:** 2026-10-30  
**Registry ID:** `T1000-FINDING-2026-039`  
**Subsystems:** `t1000::architecture`, `t1000::plaquette`, `t1000::density_matrix`, `t1000::simd`  
**License:** MIT OR Apache-2.0  

---

## 1. Executive Summary & Problem Formulation

Google Willow represents a landmark breakthrough in scalable quantum computing, demonstrating physical two-qubit gate error rates ($p_2 = 0.14\%$) well below the surface-code fault-tolerance threshold ($p_{\text{th}} \approx 0.57\%$).

This finding documents the rigorous modeling, calibration, and benchmarking of Google Willow in `T-1000`:
* **Planar Lattice Architecture:** Modeled as an $11 \times 10$ planar square lattice with 5 symmetric defect/boundary sites removed (`{(1, 2), (1, 3), (9, 2), (9, 3), (5, 0)}`), yielding exactly **105 qubits** and **182 tunable couplers** (mean node degree 3.47).
* **76-Plaquette Full Sweep:** Evaluates all 76 elementary $2 \times 2$ defect-free sub-grids across both Random Circuit Sampling (RCS) mode ($p_2 = 0.14\%$) and active Quantum Error Correction (QEC) syndrome cycle mode ($p_2 = 0.33\%$).
* **Active Pauli Frame Teleportation:** Achieves **$0.9944$ Bell fidelity** in RCS mode and **$0.9869$** in QEC mode with a **$99.5\%$ syndrome acceptance rate** using active Pauli-$Z$ software frame tracking.
* **Three-Way Architecture Showdown:** Willow 105Q significantly outperforms earlier-generation transmon processors (IBM Nighthawk R2 at $0.9606$ and Rigetti Cepheus-1-108Q at $0.9515$).

```text
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 GOLDEN BENCHMARK GROUND TRUTH                                   │
├───────────────────────────────────┬──────────────┬──────────────────┬───────────────────────────┤
│ Architecture / Benchmark Regime   │ SWAP (1>2>4) │ 4-Edge Teleport  │ QEC Syndrome Acceptance   │
├───────────────────────────────────┼──────────────┼──────────────────┼───────────────────────────┤
│ Google Willow 105Q (RCS Mode)     │ 0.9906       │ 0.9944           │ 99.5% (Sub-Threshold)     │
│ Google Willow 105Q (QEC Mode)     │ 0.9782       │ 0.9869           │ 99.5% (Syndrome Sweep)    │
├───────────────────────────────────┼──────────────┼──────────────────┼───────────────────────────┤
│ IBM Nighthawk R2 (120Q Monolithic)│ 0.9253       │ 0.9606           │ 96.8%                     │
│ Rigetti Cepheus-1-108Q (Modular)  │ 0.9082       │ 0.9515           │ 95.8%                     │
└───────────────────────────────────┴──────────────┴──────────────────┴───────────────────────────┘
```

---

## 2. Google Willow 105Q Topological & Calibration Specifications

* **Physical Qubits:** 105 superconducting transmon qubits.
* **Couplers:** 182 tunable capacitive couplers.
* **Coherence Times:** $T_1 = 98.0\,\mu\text{s}$, $T_2^* = 85.0\,\mu\text{s}$.
* **Pulse Durations:** Single-qubit microwave: $25\text{ ns}$; Two-qubit fast CZ: $20\text{ ns}$.
* **Native Basis Gates:** Virtual $R_z$, microwave $\sqrt{X}$ (SX), $X$, fast tunable $CZ$, and parametric $XY$.
* **Experiment Artifacts:**
  * Benchmark CLI: [`crates/t1000/src/bin/benchmarks/willow_105q_bench.rs`](../../crates/t1000/src/bin/benchmarks/willow_105q_bench.rs)
  * Publication Chart: [`experiments/golden_willow_105q/willow_105q_golden_showdown.png`](../../experiments/golden_willow_105q/willow_105q_golden_showdown.png)
  * Telemetry JSON: [`results/data/willow_105q_telemetry.json`](../../results/data/willow_105q_telemetry.json)
