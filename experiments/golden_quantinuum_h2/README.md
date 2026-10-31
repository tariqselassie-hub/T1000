# T-1000 Golden Experiment: Quantinuum System Model H2-1 Trapped-Ion Benchmark

**Author:** Terrence A. Jones SR  
**Date:** 2026-10-30  
**Target Architecture:** Quantinuum System Model H2-1 (56Q All-to-All QCCD Trapped-Ion QPU)  
**License:** MIT OR Apache-2.0  
**Subsystems:** `t1000::architecture`, `t1000::density_matrix`, `t1000::simd`, `t1000::routing`  

---

## 1. Executive Summary & Golden Experiment Mission

This repository subdirectory encapsulates the complete, self-contained **Golden Experiment** testing the **Quantinuum System Model H2-1** trapped-ion processor architecture against the **T-1000 Pure-Rust Simulation Engine** (AVX2+FMA density matrix evolution).

Unlike planar superconducting transmon lattices (such as Google Willow 105Q, IBM Nighthawk R2, and Rigetti Cepheus 108Q) which constrain two-qubit operations to fixed nearest-neighbor links, the Quantinuum H2-1 leverages a **Quantum Charge-Coupled Device (QCCD) linear racetrack architecture**. By dynamically shuttling $^{171}\text{Yb}^+$ data ions through 4 parallel laser interaction zones, any arbitrary pair of qubits across the 56-qubit register can be directly entangled via Mølmer-Sørensen gates ($ZZ(\theta)$) with **zero SWAP routing penalty**.

```text
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 GOLDEN BENCHMARK GROUND TRUTH                                   │
├──────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────────────────────┤
│ Hops │ QuantinuumH2 │ Willow 105Q  │ Nighthawk R2 │ Cepheus 108Q │ Physical Routing Penalty     │
├──────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────────────────────┤
│ 1    │ 0.9964       │ 0.9972       │ 0.9877       │ 0.9841       │ 0 SWAPs (Local Nearest-Neigh)│
│ 2    │ 0.9964       │ 0.9926       │ 0.9676       │ 0.9580       │ 1 SWAP (3 CNOT Pulses)       │
│ 5    │ 0.9964       │ 0.9791       │ 0.9106       │ 0.8852       │ 4 SWAPs (12 CNOT Pulses)     │
│ 10   │ 0.9964       │ 0.9570       │ 0.8256       │ 0.7801       │ 9 SWAPs (27 CNOT Pulses)     │
│ 15   │ 0.9964       │ 0.9357       │ 0.7514       │ 0.6924       │ 14 SWAPs (42 CNOT Pulses)    │
│ 20   │ 0.9964       │ 0.9149       │ 0.6869       │ 0.6192       │ 19 SWAPs (57 CNOT Pulses)    │
└──────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────────────────────┘
```

---

## 2. Quantinuum H2-1 Physical Architecture

* **Physical Qubits:** 56 $^{171}\text{Yb}^+$ hyperfine clock-state qubits with interspersed $^{138}\text{Ba}^+$ sympathetic cooling ions.
* **Topology:** All-to-all connectivity ($K_{56}$ complete graph, 1,540 couplers) enabled by physical ion shuttling along a linear racetrack surface electrode geometry.
* **Concurrent Gate Zones:** 4 independent interaction zones operating simultaneously.
* **Native Basis Gates:** Virtual $R_z$, microwave/Raman $U_{1q}$ ($R_x, R_y$), and arbitrary-angle Mølmer-Sørensen entangling gate $ZZ(\theta)$.
* **Coherence Times:** $T_1 = 100\text{ s}$, $T_2^* = 3.0\text{ s}$.
* **Gate Durations:** Single-qubit pulse: $10.0\,\mu\text{s}$; Two-qubit Mølmer-Sørensen gate: $150.0\,\mu\text{s}$; Shuttling roundtrip: $250.0\,\mu\text{s}$.

---

## 3. Directory Contents

* `quantinuum_h2_golden_showdown.png`: 4-panel publication visual diagram in Deen OS visual palette.
* `chart_golden_quantinuum_h2.py`: Python visualizer rendering the publication diagram.
* `quantinuum_h2_telemetry.json`: Raw physical telemetry and distance sweep fidelities.
* `quantinuum_h2_hardware_spec.json`: Machine-readable hardware specifications and error profile.
* `runner.py`: Unified pipeline orchestrator.

