<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Terrence A. Jones SR. All rights reserved. -->

# T-1000 Architectural & Scientific Documentation

This directory houses the sovereign architectural specifications, hardware manuals, Web HUD guides, and empirical research monographs for the **T-1000 Quantum Neural Network & Mimetic Simulation Engine**.

---

## 1. System Documentation Map

```
docs/
├── README.md                                 # Master documentation navigation & system index
├── dashboard_and_circuit_composer.md         # Tri-Deck Web HUD, 3D molecular manifold & Lego-block composer
├── cohesion_architecture.md                  # Substrate bridges (t1000_core <-> t1000) & true cohesion
├── fpga_architecture.md                      # 1000_QPU fixed-point statevector engine (Artix-7 XC7A35T)
├── zinc_quantum_screening.md                 # ZINC 250k whole-library VQC screening & Pareto drug mining
├── bloch_zipper_mps_specification.md         # 1D Matrix Product State (MPS) linear Bloch-zipper theory
├── mimac_quantum_cliff_analysis.md           # Molecular activity cliffs & chemical latent space dispersion
└── findings/                                 # Sovereign Empirical Research Registry (Monographs M123–M156)
    ├── README.md                             # Monograph index table & verification criteria
    ├── universal_ft_qnn_magic_state_w7_injection.md                     # M156: Universal FT-QNN & Magic State Distillation
    ├── virtual_36q_cut_bridge_igpu.md                                   # M155: 36Q Virtual Cut-Bridge & Co-Processor
    ├── qm9_continuous_topological_walk_screening.md                     # M154: Continuous Topological Walk Screener
    ├── qm9_transversal_w7_fault_tolerant_training.md                     # M153: [[7, 1, 3]] Steane-Wheel FT-QNN
    ├── qgol_geometric_holonomy_and_topological_solitons.md              # M152: Unitary QCA & Berry Phase Solitons
    ├── qm9_fault_tolerant_vqc_training_and_midcircuit_vnni_filtering.md  # M151: AVX-VNNI Syndrome Confinement
    ├── qm9_ab_initio_vqc_training_and_baseline_decomposition.md         # M150: Core Decoupling & Dual Phase Map
    └── ... (27 earlier monographs)
```

---

## 2. Architectural Specifications

### [Dashboard & Lego-Block Circuit Composer](dashboard_and_circuit_composer.md)
* **Tri-Deck Cockpit:** Deck 1 (3D Molecular Manifold & Force-Directed Distance Geometry), Deck 2 (Lego-Block Circuit Composer on $q_0 \dots q_{11}$ wires), and Deck 3 (Run Telemetry Matrix).
* **Hardware Bridge:** UART handshake over `COM6` @ 3.0 MBaud with the Artix-7 FPGA board; circuits run on the AVX2+FMA host CPU.
* **Cohesion Protocols:** Quantum Zeno sentry subspace pinning, XY4 dynamical decoupling echoes, and zero-noise extrapolation.

### [True Cohesion Architecture](cohesion_architecture.md)
* **Substrate Unification:** Zero-allocation bidirectional conversions between `t1000_core::QuantumState` and `t1000::QubitRegister`.
* **Open-System Noise:** Microscopic Lindblad/Kraus models (amplitude damping, phase damping, symmetric depolarizing, and Pauli noise).
* **Facaded Engine:** High-level `T1000Engine` session facade orchestrating compilation, noise mitigation, and telemetry attestation.

### [Fixed-Point Statevector Engine on the FPGA](fpga_architecture.md)
* **Target Hardware:** AMD/Xilinx Artix-7 XC7A35T-1CPG236C (Digilent Basys 3) operating at 100 MHz.
* **Compute Matrix:** controlled 2x2 complex gate streaming one amplitude pair per clock over two parity-banked block-RAM banks (14 qubits, 25-bit coefficients, 32-bit saturating state).
* **Protocol:** SET_N, RESET, LOAD, READ, GATE and EXPECT_Z over UART, specified by the bit-exact Rust model `t1000::qpu` and checked against it in simulation.

### [ZINC 250k Whole-Library Quantum Screening](zinc_quantum_screening.md)
* **Scale:** Evaluated entire ZINC 250k clean drug universe ($N = 249,455$ molecules) in $1.169\text{ s}$ ($213,350.6\text{ mols/s}$).
* **Frontier Mining:** Multi-objective Pareto scoring (QED, SAS, logP) and quantum chemical anomaly detection identifying novel zwitterionic and pharmacological motifs.

### [Bloch-Zipper & MPS Specification](bloch_zipper_mps_specification.md)
* **Theoretical Framework:** 1D Matrix Product State factorization of linear entanglement chains up to $N=200$ qubits ($70.5\text{ ns/qubit}$) and radix-7 qudits with 128-byte L1 cache residency.

### [MIMAC Quantum Activity Cliff Analysis](mimac_quantum_cliff_analysis.md)
* **Structure-Activity Landscapes:** Identification of molecular activity cliffs where minor chemical modifications produce radical shifts in quantum Hilbert space projection.

---

## 3. Sovereign Research Monographs (`docs/findings/`)

The `findings/` directory contains 31 formally registered, immutable research monographs documenting empirical discoveries across the T-1000 engine. All monographs are dual-housed with the Deen OS Sovereign System Archive:

* **M153:** 18-Qubit Transversal [[7, 1, 3]] Steane-Wheel FT-QNN & In-Register Physical Ancillas ([`qm9_transversal_w7_fault_tolerant_training.md`](findings/qm9_transversal_w7_fault_tolerant_training.md))
* **M152:** Non-Abelian Geometric Holonomy, Quantized Vortex Dipoles & Floquet Chaos Scrambling in Unitary QCA ([`qgol_geometric_holonomy_and_topological_solitons.md`](findings/qgol_geometric_holonomy_and_topological_solitons.md))
* **M151:** 18-Qubit Fault-Tolerant QM9 VQC Training & Mid-Circuit AVX-VNNI Syndrome Filtering ([`qm9_fault_tolerant_vqc_training_and_midcircuit_vnni_filtering.md`](findings/qm9_fault_tolerant_vqc_training_and_midcircuit_vnni_filtering.md))
* **M150:** QM9 Ab Initio VQC Training, Atomic Baseline Decomposition & Dual-Angle Quantum Phase Map ([`qm9_ab_initio_vqc_training_and_baseline_decomposition.md`](findings/qm9_ab_initio_vqc_training_and_baseline_decomposition.md))
* **M149:** Real-Symmetric Molecular Eigensystem & Zero-Allocation QM9 Streaming Ingestion ([`molecular_eigensystem_and_qm9_ingestion.md`](findings/molecular_eigensystem_and_qm9_ingestion.md))
* **M148:** Closed-Loop AVX-VNNI Real-Time Surface Code Threshold Dynamics ([`vnni_qec_threshold_dynamics.md`](findings/vnni_qec_threshold_dynamics.md))
* **M147:** Fault-Tolerant Lattice Surgery & Magic State Distillation ([`fault_tolerant_lattice_surgery_and_magic_distillation.md`](findings/fault_tolerant_lattice_surgery_and_magic_distillation.md))
* **M146:** Comprehensive 7-Pillar Full Quantum Neural Network Suite Showdown ([`full_quantum_neural_network_suite_benchmarks.md`](findings/full_quantum_neural_network_suite_benchmarks.md))
* **M123–M145:** Topological walk sorters, glued trees, quantum cellular automata, IBM Nighthawk plaquette mapping, and heterogeneous CPU-iGPU cut-bridges.

See [`docs/findings/README.md`](findings/README.md) for the full registry table and telemetry metrics.

---

## 4. Documentation & Verification Standards

1. **Exact Reproduction:** Every document describing an empirical benchmark includes exact CLI invocations and links to corresponding source binaries in `crates/t1000/src/bin/` or `crates/t1000_conway/src/bin/`.
2. **Visual Palette Compliance:** All embedded figures, charts, and diagrams adhere strictly to the Deen OS Sovereign Visual Specification (`#05080d` obsidian backdrop, `#00e676` laser green, `#ffb300` radiant gold, `#ef4444` alert red, polar slate typography).
3. **Dual Licensing Governance:** All documents include SPDX and copyright headers attributing authorship to `Terrence A. Jones SR` under MIT OR Apache-2.0.
