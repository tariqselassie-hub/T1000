<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Terrence A. Jones SR. All rights reserved. -->

# T-1000 Operational Scripts Directory

This directory contains autonomous tooling, empirical benchmarking drivers, data ingestion pipelines, hardware co-processor bridges, and sovereign visualization suites.

---

## Directory Organization

```
scripts/
├── charting/       # 36 Sovereign research charting modules & master suite (chart_all.py)
├── data/           # High-throughput dataset feature extractors (MobileNetV3 PCA)
├── fpga/           # Artix-7 XC7A35T hardware UART drivers & chaos verification
├── qec/            # Heterogeneous CPU-GPU cut-bridge pipelines (OpenCL + AVX2)
├── screening/      # Chemical latent space screening & molecule search drivers
└── benchmarks/     # High-level Python comparative benchmark drivers (Qrack vs T-1000)
```

---

## Subsystem Details

### 1. `scripts/charting/`
Visual telemetry suite adhering strictly to the Deen OS Sovereign Visual Specification (Obsidian `#05080d`, Laser Emerald Green `#00e676`, Radiant Amber Gold `#ffb300`, Alert Crimson Red `#ef4444`, Polar Slate `#f8fafc`).

* **`deen_theme.py`**: Shared typography, color tokens, and card styling utilities.
* **`chart_all.py`**: Master orchestrator executing all primary scientific visualization modules.
* **`chart_qm9_continuous_walk.py`**: M154 Continuous topological walk on ab initio Coulomb Laplacians & AVX-VNNI Zeno confinement.
* **`chart_qm9_transversal_w7_training.py`**: M153 Transversal [[7, 1, 3]] Steane-Wheel FT-QNN dynamics & active ancilla QEC.
* **`chart_qgol_holonomy.py`**: M152 Unitary QCA non-Abelian holonomy, quantized vortex dipoles & Floquet chaos.
* **`chart_qm9_fault_tolerant_training.py`**: M151 18-Qubit fault-tolerant QM9 VQC training & AVX-VNNI filtering.
* **`chart_qm9_vqc_training.py`**: M150 QM9 ab initio VQC training, atomic baseline decomposition & dual-angle phase map.
* **`chart_molecular_eigensystem.py`**: M149 Real-symmetric molecular Coulomb eigensystem & extended XYZ ingestion.
* **`chart_vnni_threshold.py`**: M148 AVX-VNNI real-time surface code threshold & quadratic error suppression.
* **`chart_qec_fault_tolerance.py`**: M147 Fault-tolerant lattice surgery & Bravyi-Kitaev magic state distillation.
* **`chart_full_qnn_suite.py`**: M146 Comprehensive 7-pillar full QNN suite showdown.
* **`chart_cepheus_108q.py`**: M144 Rigetti Cepheus-1-108Q physical QPU 30-qubit glued-tree run comparison.
* **`chart_qgnn_showdown.py`**: M143 Lie-algebraic geometric QGNN barren plateau resilience & permutation equivariance.
* **`chart_orthogonal_proxy_showdown.py`**: M142 Orthogonal K-patch proxy RCS: PyQrack vs T-1000 bare-metal AVX2.
* **`chart_zinc_topological_screen.py`**: M141 Continuous topological quantum walk screening on ZINC 250k.
* **`chart_topological_walk_sorter.py`**: M140 Continuous topological walk sorter with weld Zeno sentries.
* **`chart_rcs_coherence_chaos.py`**: M139 High-depth RCS chaos & physical noise coherence horizon.
* **`chart_cascaded_tree_valve.py`**: M136 Cascaded multi-tier glued-tree valve deep routing telemetry.
* **`chart_glued_tree_valve.py`**: M135 Multi-exit topological glued-tree valve ballistic router.
* **`chart_glued_trees.py`**: M134 Childs et al. continuous-time quantum walk exponential speedup.
* **`chart_ultrametric_tree.py`**: M133 Multi-register QMUX ultrametric tree sorter architecture & runtime telemetry.
* **`chart_birds_hybrid.py`**: M132 Waterbirds subpopulation robustness and accuracy trajectories.
* **`chart_nighthawk_99_plaquettes.py`**: M124 IBM Quantum Nighthawk 99-plaquette contraction & Qrack comparison.
* **`chart_nighthawk_layout.py`**: 120-qubit IBM Nighthawk R2 and Kagome strip layout rendering.
* **`chart_exascale_showdown.py`**: Desktop CPU/FPGA vs supercomputing infrastructure comparison.
* **`chart_pc_vs_fpga.py`**: AVX2 CPU vs Artix-7 FPGA execution cycles and throughput scaling.
* **`chart_fpga_26q.py`**: 26-qubit hardware co-processor UART streaming telemetry.
* **`chart_frontier_rcs.py`**: 14-qubit, 36-depth random circuit sampling with Porter-Thomas chaos.
* **`chart_frontier_rcs_fpga.py`**: FPGA cycle-accurate execution vs simulated statevector.
* **`chart_lattice_phase.py`**: 2D transverse Ising lattice quantum phase transition.
* **`chart_lattice_quench.py`**: Non-equilibrium quantum quench and Lieb-Robinson magnon spreading.
* **`chart_lead_109_attractor.py`**: Convergence dynamics of the 3 magic molecular attractor switches.
* **`chart_mimac_activity_cliffs.py`**: Chemical latent space activity cliffs and structure-activity landscapes.
* **`chart_mnist_qnn.py`**: 10-qubit MNIST VQC optimization and coherence decay.
* **`chart_qnn_scaling.py`**: Dense AVX2 statevector evolution scaling from 10 to 26 qubits.
* **`chart_zinc_anomalies.py`**: ZINC 250k Search 3: Zwitterion and topological quantum anomalies.
* **`chart_zinc_pareto.py`**: ZINC 250k Search 1: Frontier Pareto drug leads and binding affinities.

### 2. `scripts/data/`
* **`extract_birds_features.py`**: Deep visual transfer feature extraction from full-resolution Waterbirds images using PyTorch MobileNetV3-Small. Performs CPU-accelerated inference (~245 img/s), extracts 576-D embeddings, computes PCA projection down to 10 principal components ($37.48\%$ visual variance), and writes standardized $[-\pi, \pi]^{10}$ binary tensors to `data/birds/`.
* **`extract_birds.py`**: Utility to extract and verify the Waterbirds image archive.

### 3. `scripts/fpga/`
* **`test_chaos_streaming.py`**: Connects to the physical Artix-7 FPGA over 3.0 MBaud UART (`COM6`) to verify cycle-accurate Porter-Thomas chaos streaming and hardware attestation hashes.
* **`bench_pc_vs_fpga_scaling.py`**: Benchmarks CPU AVX2 vs FPGA Artix-7 across circuit widths $N=10\dots 26$.
* **`frontier_rcs_fpga_bench.py`**: Evaluates random circuit sampling on physical hardware vs reference simulation.
* **`nist_randomness_audit.py`**: Runs NIST SP 800-22 statistical entropy tests on FPGA streamed output.
* **`run_fpga_26q_telemetry.py`**: Records continuous telemetry across 26-qubit scaled circuits.
* **`test_1000_qpu.py`**: Hardware handshake and 7-segment display verification ("1000").
* **`test_19s_chaos_burst.py`**: High-speed burst test of chaotic entropy generation.

### 4. `scripts/qec/`
* **`heterogeneous_bridge_pipeline.py`**: Orchestrates concurrent dual-register evaluation across CPU L3 cache (AVX2+FMA) and Intel Arc Xe iGPU (OpenCL), evaluating non-local Tsirelson CHSH Bell inequality witnesses ($S = 2.82842712$).
* **`bseq.py`**: Syndrome sequence and stabilizer matrix generator for planar and rotated surface codes.

### 5. `scripts/screening/`
* **`analyze_and_normalize_mimac.py`**: Data normalization pipeline for MIMAC chemical library.
* **`analyze_lead_109_attractor.py`**: Attractor basin analysis for prime drug candidate #109.
* **`mimac_quantum_cliff_search.py`**: Identifies steep activity cliffs in chemical descriptor manifolds.
* **`simulate_period_1_2_atoms.py`**: Computes Hartree-Fock Coulomb matrices for Period 1 and 2 atoms ($Z=1\dots 10$).
* **`simulate_period_1_2_molecules.py`**: Evaluates ground-state molecular geometries for Period 1 and 2 hydrides.

---

## Usage Examples

```bash
# Render all master scientific charts
python scripts/charting/chart_all.py

# Render M153 Transversal FT-QNN Dynamics
python scripts/charting/chart_qm9_transversal_w7_training.py

# Run FPGA cycle-accurate chaos verification
python scripts/fpga/test_chaos_streaming.py --port COM6 --baud 3000000

# Run heterogeneous CPU+iGPU cut-bridge pipeline
python scripts/qec/heterogeneous_bridge_pipeline.py
```
