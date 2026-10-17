# T-1000 Scientific Results & Empirical Telemetry Gallery

This directory stores reproducible experiment reports, benchmark telemetry JSON/CSV files, and publication-ready charts rendered according to the Deen OS Sovereign Visual Specification.

---

## Directory Organization

```
results/
├── charts/         # 16 High-resolution scientific figures (Obsidian & Laser Green)
├── experiments/    # Hardware and simulation JSON/CSV telemetry reports
└── data/           # Intermediate screening metrics and latent coordinate tables
```

---

## Visual Gallery Manifest (`results/charts/`)

| # | Scientific Figure | Focus / Experiment | Resolution / Size |
| :-: | :--- | :--- | :--- |
| **01** | **`mnist_qnn_learning_dynamics.png`** | 10-Qubit MNIST VQC optimization and coherence decay. | ~640 KB |
| **02** | **`qnn_scaling_26_qubits.png`** | Dense AVX2 statevector evolution scaling from 10 to 26 qubits. | ~497 KB |
| **03** | **`lattice_phase_transition.png`** | 2D transverse Ising lattice quantum phase transition. | ~457 KB |
| **04** | **`lattice_quench_dynamics.png`** | Non-equilibrium quantum quench and Lieb-Robinson magnon spreading. | ~475 KB |
| **05** | **`frontier_rcs_14q_36d_telemetry.png`** | 14-Qubit, 36-Depth Random Circuit Sampling with Porter-Thomas chaos. | ~572 KB |
| **06** | **`pc_vs_fpga_26q_scaling.png`** | AVX2 CPU vs. Artix-7 FPGA execution cycles and throughput. | ~578 KB |
| **07** | **`fpga_26q_telemetry.png`** | 26-Qubit hardware co-processor UART streaming telemetry. | ~437 KB |
| **08** | **`frontier_rcs_fpga_comparison.png`** | FPGA cycle-accurate execution vs. simulated statevector. | ~393 KB |
| **09** | **`zinc_quantum_anomalies.png`** | ZINC 250k Search 3: Zwitterion and topological quantum anomalies. | ~814 KB |
| **10** | **`zinc_pareto_leads.png`** | ZINC 250k Search 1: Frontier Pareto drug leads and binding affinities. | ~962 KB |
| **11** | **`mimac_quantum_activity_cliffs.png`** | Chemical latent space activity cliffs and structure-activity landscapes. | ~687 KB |
| **12** | **`lead_109_attractor_analysis.png`** | Convergence dynamics of the 3 magic molecular attractor switches. | ~862 KB |
| **13** | **`exascale_desktop_showdown.png`** | T-1000 desktop AVX2/FPGA throughput vs. Frontier & Summit supercomputers. | ~667 KB |
| **14** | **`nighthawk_r2_kagome_layout.png`** | 120-qubit IBM Quantum Nighthawk R2 & Kagome strip plaquette layout. | ~2,032 KB |
| **15** | **`waterbirds_hybrid_qnn_telemetry.png`** | Waterbirds 11k visual transfer QNN robustness and subpopulation generalization. | ~534 KB |
| **16** | **`ultrametric_tree_sorter_telemetry.png`** | Multi-register QMUX Ultrametric Tree Sorter architecture and runtime telemetry. | ~464 KB |

---

## Experiment Reports Manifest (`results/experiments/`)

* **`ultrametric_tree_sorter_report.json`**: Multi-register hierarchical QMUX benchmark on Waterbirds ($4.14\,\mu\text{s/sample}$, $70.69\%$ coarse accuracy, strong triangle inequality verified).
* **`path_a_vs_c_comparison_report.json`**: Direct comparison of Headless Sorter (Path A) vs. Data Re-Uploading (Path C).
* **`fidelity_orthogonality_report.json`**: Mathematical proof and empirical verification of the Unitarity Rigidity Theorem ($W^\dagger W = \mathbb{I}$).
* **`cut_bridge_36q_results.json`**: Mid-circuit checkpointed 36-qubit cut-bridge simulation in CPU L3 cache.
* **`chsh_qrack_ace_results.json`**: Tsirelson CHSH witness verification ($S = 2.82842712$).
* **`zinc_dual_search_summary.json`**: Whole-library screening summary over 250,000 SMILES molecules.

---

## Reproducibility Command

To regenerate all 16 scientific figures from current disk telemetry:
```bash
python scripts/charting/chart_all.py
```
