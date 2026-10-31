# T-1000 Golden Experiment: Google Willow 105Q Surface-Code Threshold Benchmark

**Author:** Terrence A. Jones SR  
**Date:** 2026-10-30  
**Target Architecture:** Google Willow 105Q (Planar Superconducting Transmon Processor)  
**License:** MIT OR Apache-2.0  
**Subsystems:** `t1000::architecture`, `t1000::plaquette`, `t1000::density_matrix`, `t1000::simd`  

---

## 1. Executive Summary & Golden Experiment Mission

This repository subdirectory encapsulates the complete, self-contained **Golden Experiment** testing the **Google Willow 105Q** planar processor architecture against the **T-1000 Pure-Rust Simulation Engine** (AVX2+FMA density matrix evolution).

Google Willow demonstrates sub-threshold quantum error correction performance with physical two-qubit error rates ($p_2 = 0.14\%$) well below the surface-code fault-tolerance threshold ($p_{\text{th}} \approx 0.57\%$). This experiment benchmarks 76 elementary $2 \times 2$ plaquettes, evaluating raw sequential SWAP routing, 4-edge plaquette teleportation with active Pauli frame recovery, and head-to-head comparison with IBM Nighthawk R2 (120Q) and Rigetti Cepheus-1-108Q.

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

## 2. Google Willow 105Q Physical Topology

Google Willow arranges 105 superconducting transmon qubits on an $11 \times 10$ planar bounding lattice with 5 symmetric boundary sites removed (`{(1, 2), (1, 3), (9, 2), (9, 3), (5, 0)}`):

* **Total Physical Qubits:** 105
* **Total Planar Couplers:** 182 tunable capacitive couplers (mean connectivity 3.47)
* **Valid $2 \times 2$ Plaquettes:** 76 defect-free 4-qubit sub-grids
* **Pulse Durations:** Single-qubit microwave: 25 ns; Two-qubit fast CZ: 20 ns
* **Coherence Times:** $T_1 = 98.0\,\mu\text{s}$, $T_2^* = 85.0\,\mu\text{s}$

---

## 3. Directory Contents

* `willow_105q_golden_showdown.png`: 4-panel publication visual diagram in Deen OS visual palette.
* `chart_golden_willow.py`: Python visualizer rendering the publication diagram.
* `willow_105q_telemetry.json`: Raw physical telemetry and per-plaquette Bell fidelities.
* `willow_hardware_spec.json`: Machine-readable hardware specifications and error profile.
* `runner.py`: Unified pipeline orchestrator.

