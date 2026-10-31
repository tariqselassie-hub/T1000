# T1000-FINDING-2026-040: Quantinuum System Model H2-1 Trapped-Ion All-to-All Architecture & Zero-SWAP Distance Scaling

> **Audit note (2026-10-30).** Simulated with pure Rust AVX2+FMA density matrix evolution based on published Quantinuum H2-1 specifications ($p_1 = 0.0020\%$, $p_2 = 0.180\%$, SPAM $0.200\%$, $T_1 = 100\text{ s}$, $T_2^* = 3.0\text{ s}$).

**Author:** Terrence A. Jones SR  
**Date:** 2026-10-30  
**Registry ID:** `T1000-FINDING-2026-040`  
**Subsystems:** `t1000::architecture`, `t1000::density_matrix`, `t1000::simd`, `t1000::routing`  
**License:** MIT OR Apache-2.0  

---

## 1. Executive Summary & Problem Formulation

In superconducting quantum processors, qubit connectivity is constrained by physical lithographic wiring on a 2D planar wafer (e.g. nearest-neighbor square grids or heavy-hex lattices). Executing two-qubit entangling gates between non-adjacent qubits requires inserting chains of SWAP gates ($3\text{ CNOTs}$ per hop), which causes exponential degradation of state fidelity across distance.

The **Quantinuum System Model H2-1** resolves this constraint through a **Quantum Charge-Coupled Device (QCCD) linear racetrack architecture**:
* **Dynamic All-to-All Shuttling:** Individual $^{171}\text{Yb}^+$ ions are dynamically shuttled via radio-frequency Paul trap surface electrodes into any of 4 concurrent laser interaction zones.
* **Complete Graph Connectivity:** Full all-to-all connectivity ($K_{56} = 1,540$ couplers) enables direct Mølmer-Sørensen entangling gates between any two arbitrary qubits with **zero SWAP gates**.
* **Distance Scaling Invariance:** Across graph routing distances $d = 1 \dots 20$ hops, Quantinuum H2-1 maintains a flat **$0.9964$ Bell fidelity**, whereas planar transmon lattices experience severe decay (Willow: $0.9972 \to 0.9149$, Nighthawk: $0.9877 \to 0.6869$, Cepheus: $0.9841 \to 0.6192$).

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

## 2. Quantinuum H2-1 Technical Specifications

* **Qubit Species:** $^{171}\text{Yb}^+$ hyperfine clock states with $^{138}\text{Ba}^+$ sympathetic cooling ions.
* **Physical Qubits:** 56 data qubits.
* **Couplers:** 1,540 all-to-all pairs.
* **Parallel Interaction Zones:** 4 concurrent two-qubit zones.
* **Coherence Times:** $T_1 = 100.0\text{ s}$, $T_2^* = 3.0\text{ s}$.
* **Gate Durations:** Single-qubit pulse: $10.0\,\mu\text{s}$; Two-qubit Mølmer-Sørensen gate: $150.0\,\mu\text{s}$; Shuttling latency: $250.0\,\mu\text{s}$.
* **Native Basis Gates:** Virtual $R_z$, microwave/Raman $U_{1q}$ ($R_x, R_y$), Mølmer-Sørensen $ZZ(\theta)$.
* **Experiment Artifacts:**
  * Benchmark CLI: [`crates/t1000/src/bin/benchmarks/quantinuum_h2_bench.rs`](../../crates/t1000/src/bin/benchmarks/quantinuum_h2_bench.rs)
  * Publication Chart: [`experiments/golden_quantinuum_h2/quantinuum_h2_golden_showdown.png`](../../experiments/golden_quantinuum_h2/quantinuum_h2_golden_showdown.png)
  * Telemetry JSON: [`results/data/quantinuum_h2_telemetry.json`](../../results/data/quantinuum_h2_telemetry.json)
