# T-1000: Quantum Neural Network & Mimetic Circuit Simulator

[![Rust](https://img.shields.io/badge/rust-2021_edition-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)
[![Build Status](https://img.shields.io/badge/status-active%20development-brightgreen.svg)](#)

> *"Like the liquid metal assassin, T-1000 adapts and mimics any quantum device or classical architecture. Feed it parameters; it handles the rest."*

`T-1000` is an ultra-high-performance Variational Quantum Circuit (VQC) and Quantum Neural Network engine engineered in pure Rust. Built for extreme SIMD throughput, arbitrary circuit DAG scheduling, and chemical latent space exploration, it runs completely standalone with zero legacy bloat.

---

> Where each headline number comes from, and how to regenerate it: [`docs/CLAIMS.md`](docs/CLAIMS.md) (`python scripts/check_claims.py`).

## Architecture Overview

```
                                      T-1000
                        ┌─────────────────────────────────┐
                        │   Shape-Shifting Quantum Core   │
                        └───────────────┬─────────────────┘
                                        │
         ┌──────────────────────────────┼──────────────────────────────┐
         ▼                              ▼                              ▼
  SIMD Acceleration             Circuit Compiler              Chemical Exploration
  • 64-bit complex SOA          • Per-qubit DAG scheduling    • OpenSMILES parser
  • AVX2 + FMA Ry / Rz kernels  • Gate & stage fusion         • 19-dim quantum encoding
  • Fused diagonal CZ passes    • Disjoint wire execution     • ZINC 250k drug screening
  • Multi-threaded (Rayon)      • Commutation analysis        • Hilbert-space fidelity
```

### Workspace Crates

| Crate | Purpose |
| :--- | :--- |
| **[`t1000_core`](crates/t1000_core)** | Pure mathematical substrate: state registers, partial trace, POVM projections, Hamiltonian evolution, Porter-Thomas chaos metrics. |
| **[`t1000`](crates/t1000)** | The high-performance QNN engine: SIMD AVX2 kernels, circuit DAG compiler, SMILES molecular encoder, parameter-shift optimizers, and benchmarks. |
| **[`t1000_conway`](crates/t1000_conway)** | Sovereign Unitary Quantum Cellular Automata (QCA) engine: Fractional Fermi Seas, interaction holonomy cycles across Confinement-Induced Resonances (CIR), and 3D web telemetry visualizer. |

---

## Key Capabilities

### 1. Shape-Shifting Circuit Compilation & Scheduling (`t1000::compiler`)
* **Per-qubit dependency DAG (`CircuitDag`):** Tracks gate commutation and dependencies along wire order.
* **Gate Fusion:** Greedy grouping of non-overlapping single-qubit rotations and multi-qubit diagonal interactions into unified stages.
* **Fused Diagonal Kernel (`t1000::diagonal`):** Vectorized 256-bit AVX2+FMA streaming memory sweep (`_mm256_fmadd_pd`, dual 8-amplitude unrolling) executing entire layers of $R_z$ and $CZ$ gates in a single pass (**$1.7\text{–}1.8\times$ faster** than per-qubit loops on that diagonal layer, re-measured; the gain on a whole 18Q depth-12 adjoint pass is much smaller, reported earlier as $0.1002\text{ s}$ to $0.0921\text{ s}$ and not re-run).

### 2. AVX2 + FMA SIMD Statevector Engine (`t1000::simd`, `t1000::tiled_butterfly`)
* **Structure-of-Arrays (SOA):** Separated contiguous arrays of double-precision real and imaginary amplitudes.
* **Modular Architectural Decomposition:** Subsystem partitioned into focused domain modules (`guards`, `rotations`, `algebra`, `measurements`, `model`), all conforming strictly to $\le 800$ LoC per file and $\le 80$ LoC per function.
* **Strongly-Typed SIMD Domain Models (`SimdStateView`, `SimdStateViewMut`, `SimdBackend`):** Zero-copy, chainable unitary evolution (`.ry()`, `.rz()`, `.cz()`, `.rademacher_phase()`, `.normalize()`) and direct expectation reductions directly on statevector buffers.
* **L2-Resident Tiled Butterfly Kernels:** Cache-conscious stride access for high qubit indices ($q \ge 16$).
* **Speed:** the 18-qubit, 36-layer RCS circuit runs $11\text{–}14\times$ faster than the repository's textbook scalar reference simulator on 20 threads (`frontier_rcs`). A comparison with Qiskit Aer is not reproducible from this repository (`frontier_rcs --export` writes the circuit, but `scripts/aer_rcs_benchmark.py` is missing), so no Aer figure is claimed.

### 3. SMILES Molecular Exploration & Whole-Library 250k Screening (`zinc_dual_search`)
* **Chemical Featurizer:** Native parser for OpenSMILES strings generating 19 molecular descriptors (atom counts, aromaticity, ring closures, molecular weight, heteroatom ratio).
* **Quantum Feature Map:** Direct parameterized injection into Hilbert space ($|\psi(\text{SMILES})\rangle$) via parameterized $R_y(x_i)$ rotations.
* **Whole-Library ZINC 250k Engine:** Screens all 249,455 drug candidates in **1.17 seconds** (**213,351 molecules/sec**) across 20 threads (the author's run; not re-run here because the ZINC 250k table is not shipped).
* **Search 3 (Quantum Anomalies & Outliers):** Mines extreme zwitterions, highly charged amidinium/guanidinium motifs, and topological outliers based on Hilbert space dispersion.
* **Search 1 (Frontier Pareto Drug Candidates):** Discovers top golden leads with $\text{QED} > 0.88$, $\text{SAS} < 1.45$, and $\text{logP} \in [2.4, 3.1]$.
* **Period 1 & 2 molecular shells (`t1000::chemistry`):** tabulated reference energies and fixed-formula Hamiltonians for 9 molecules, not computed from molecular integrals. Use §29 (`t1000::vqe`) for ab initio energies.
* Detailed Documentation: [`docs/zinc_quantum_screening.md`](docs/zinc_quantum_screening.md).

### 4. Statistical Quantum Verification (`t1000::rcs`)
* **Frontier RCS Benchmark (`frontier_rcs`):** Deep Random Circuit Sampling (default: 18 qubits, 36 layers, 1,945 gates, $262,144$ states) benchmarking the optimized 256-bit AVX2 engine against a textbook scalar reference simulator and Qiskit Aer.
* **Dual-Engine Precision:** Machine precision numerical agreement (fidelity $1.000000000000$, max amplitude error $< 3.1 \times 10^{-17}$).
* **Cross-Entropy Benchmarking (XEB):** Validates linear XEB convergence ($F_{\text{XEB}} \approx 1.0045$) against ideal Haar-random Porter-Thomas distributions.
* **Kolmogorov-Smirnov & Entropy Tests:** Automated confirmation of Porter-Thomas chaos ($D_{\text{KS}} = 0.0021 < D_{\text{critical}} = 0.0027$, $\Delta S = 0.4243\text{ nats} \to 1 - \gamma$).
* **Adjoint Differentiation:** analytical gradients for 1,296 parameters on 18 qubits in about $0.12\text{ s}$ (re-measured; parameter shift would need 2,592 circuit runs). At 12 qubits and 96 parameters the adjoint gradient is $47\text{–}61\times$ faster than parameter shift and agrees to $3\times 10^{-16}$.
* **Coherence Frontier Push (`coherence_frontier_push`):** Dual-mark physical experiment tracking quantum $\ell_1$-norm coherence capacity ($C_{\text{norm}}$) up to 26 qubits dense ($2^{26}$ amplitudes, 1.02 GB RAM). The 2D tensor-network stage to 64 qubits (approximate, bond-dimension-truncated) and the heptal stage ($7^9$ states) did not finish within a 150 s cap in the audit run.
* **Physical Noise & Coherence Horizon (`rcs_coherence_chaos_bench`):** Layer-by-layer open-system Lindbladian decoherence ($T_1 = 45\,\mu\text{s}, T_2^* = 60\,\mu\text{s}, p_{\text{depol}} \in [10^{-4}, 0.02]$) across $N = 14$ and $N = 16$ qubits. Discovers the quantum speckle fidelity cliff and demonstrates that split-step CPMG/XY4 refocusing paired with symplectic manifold re-projection rescues cross-XEB fidelity, prevents negative speckle correlation collapse ($-0.0025 \to +0.0124$), and raises state fidelity at 14 qubits from $0.0004$ to $0.021$. The percentage gains (up to $+5,147\%$ and $+13,543\%$) are relative to a near-zero baseline, absolute fidelity stays near 2%, and the sweep also has negative entries. Documented in [`docs/findings/rcs_physical_noise_chaos_transition.md`](docs/findings/rcs_physical_noise_chaos_transition.md).

### 5. Quantum Vision & Hybrid QCNN Benchmark (`mnist_qnn_train`, `mnist_hybrid_qnn`)
* **Amplitude & Spatial Encoding:** Dual-stream injection coupling 10-qubit Hilbert space ($2^{10} = 1024$ amplitudes) with a 65-dimensional 2D multi-scale spatial convolution & directional stroke stream.
* **Bipartite Entanglement Observables:** 55 quantum expectation operators (10 single-qubit $\langle Z_i \rangle$ + 45 two-body $\langle Z_i Z_j \rangle$ correlators) with analytical reverse-mode `adjoint_gradient`.
* **Breakthrough 95%+ Hybrid Horizon:** Fusing Hilbert space quadratic invariants with spatial edge inductive biases through a differentiable non-linear MLP head elevates test accuracy to **95.45% (Peak 95.95%)** in **3.8 seconds total training time** (**22.80 $\mu$s / sample**), decisively surpassing classical baselines (logistic regression 87.0%, k-NN 92.35%).
* Detailed Documentation: [`docs/qcnn_adjoint_microarchitecture.md`](docs/qcnn_adjoint_microarchitecture.md).

| Architecture Model | Test Accuracy % | Space Representation | State Coherence $C_{\ell_1}$ | Throughput |
| :--- | :---: | :---: | :---: | :---: |
| **GuddaNetworks BNN Baseline (BSC)** | 78.59% | 4096-bit Binary Sparse Coding | Classical | Classical BSC |
| **GuddaNetworks BNN Baseline (MAP)** | 79.40% | 4096-dim Multiply-Add-Permute | Classical | Classical MAP |
| **GuddaNetworks HDC-10k Dimension** | 81.25% | 10,000-bit BSC Space | Classical | Classical BSC |
| **T-1000 10-Qubit VQC (Linear Head)** | **84.50%** | **1024-dim ($2^{10}$) Hilbert Space** | **350.94** | **34.38 $\mu$s / sample** |
| **T-1000 Hybrid QCNN (AVX2 Adjoint)** | **95.45%** | **120-dim Fused Latent Representation** | **450.69** | **22.80 $\mu$s / sample** |

### 6. Fixed-Point Statevector Engine on the Basys 3 (`fpga/1000_qpu`)
* **Target hardware:** AMD/Xilinx Artix-7 XC7A35T-1CPG236C (Digilent Basys 3), 100 MHz, UART at 3 Mbaud on `COM6`.
* **Design:** 14-qubit complex statevector in block RAM (2 x 8192 x 64 bits), a controlled 2x2 gate that processes one amplitude pair per clock, and load, read-back and Z-string expectation commands. 25-bit coefficients, exact accumulate, 32-bit saturating state.
* **Specification:** a bit-exact Rust model (`t1000::qpu`). In simulation the design's 2,536 reply bytes equal the model's for the golden stream. Build: 1,904 LUTs, 32 BRAM36, 40 DSP48E1, +0.372 ns slack at 100 MHz.
* **Precision study:** UCCSD circuits for molecules up to 14 qubits reproduce the double-precision energy to under 1e-3 mHa with 24 or more fractional bits (`qpu_precision`).
* **Not an accelerator over the CPU:** a 14-qubit UCCSD circuit is 37,264 gates, several seconds of fabric and UART time. The board is a hardware-in-the-loop check of the fixed-point arithmetic.
* Details and hardware status: [`docs/fpga_architecture.md`](docs/fpga_architecture.md), [`docs/CLAIMS.md`](docs/CLAIMS.md).

### 7. Fault-Tolerant Quantum Error Correction & Planar Surface Code (`t1000::qec`)
* **3x3 Planar Surface Code $[[9, 1, 3]]$:** Rotated lattice topology embedding 9 physical data qubits with 8 syndrome measurement plaquettes (4 weight-2/4 $X$-checks and 4 weight-2/4 $Z$-checks).
* **Closed-Form Transversal Code Space:** Exact stabilizer subspace generation via generator algebra (`[27, 432, 6, 192]`) guaranteeing 16 closed-form logical codewords with zero runtime allocation.
* **Standard Stabilizer Suite:** Native support for 3-qubit bit-flip $[[3,1,1]]$, 3-qubit phase-flip $[[3,1,1]]$, 5-qubit perfect code $[[5,1,3]]$, 7-qubit Steane code $[[7,1,3]]$, and 9-qubit Shor code $[[9,1,3]]$.
* **AVX-VNNI Symplectic Syndrome Decoder (`t1000::qec::vnni`):** Hardware-accelerated 32-way INT8 vector dot-product engine (`{{vex}} vpdpbusd`) evaluating all 8 surface code plaquettes simultaneously in about $27\text{ ns}$ (about $37\text{ Million decodings/sec}$, $3.4\text{ ns}$ per stabilizer, re-measured) and executing Clifford+$T$ symplectic commutation checks at about $34\text{ Million checks/sec}$ (`vnni_syndrome_bench`).

### 8. Canonical Quantum Algorithms & Readout Error Mitigation (`t1000::canonical`)
* **Foundational Algorithmic Kernels:**
  * **Grover's Diffusion Operator:** Inversion about the mean across arbitrary register widths with phase reflection.
  * **Bernstein-Vazirani Parity Oracle:** Deterministic single-query recovery of hidden bitstrings $s \in \{0, 1\}^n$.
  * **Quantum Phase Estimation (QPE):** Controlled unitary power stages with inverse Quantum Fourier Transform (IQFT) precision readouts.
  * **Hardware-Efficient Ansatz (HEA):** Parameterized $R_y(\theta) \cdot R_z(\phi)$ layers with Linear, Circular, or All-to-All entangling topologies.
  * **Trotter-Suzuki Spin Chains:** First-order Lie-Trotter evolution for Transverse-Field 1D Ising chains ($H = -J \sum Z_i Z_{i+1} - h \sum X_i$).
* **Readout Error Mitigation:** Full $2^n \times 2^n$ calibration confusion matrix inversion (`ReadoutConfusionMatrix`) with least-squares regularization and physical probability simplex projection.

### 9. Coherence-Aware ASAP/ALAP DAG Scheduler & Auto-DD (`t1000::compiler::scheduler`)
* **Causal Dependency Scheduling:** Analyzes Directed Acyclic Graphs (DAGs) to identify critical paths, earliest start times (ASAP), and latest start times (ALAP).
* **Decoherence Minimization:** Computes per-qubit idle latency bubbles during non-uniform gate delays and multi-qubit entangling stages.
* **Automatic Dynamical Decoupling (Auto-DD):** Injects calibrated CPMG or XY-4 refocusing $\pi$-pulse pairs into idle qubit cycles exceeding coherence thresholds, boosting simulated fidelity retention under $T_1/T_2$ noise channels.

### 10. 1D Matrix Product State (MPS) Linear Bloch-Zipper Engine (`t1000::zipper`)
* **Continuous Bloch Probability Steering:** Dynamically computes single-qubit polar rotation angles $\theta_i = \pi \cdot P_i(0)$ directly from amplitude geometry without projective wavefunction collapse.
* **Sliding Window Cache Residency:** Sliding local statevector window ($W \le 4$, 8–16 complex amplitudes = 128–256 bytes) remaining $100\%$ resident in the 32 KB CPU L1 data cache with zero DRAM bus traffic.
* **Partial-Trace Epoxy Freezing:** As each qubit locks into its deterministic pole ($|\langle Z_i \rangle| \ge 0.999$), the boundary density matrix is reduced via partial trace, advancing the front and releasing scratch memory.
* **Linear scaling:** 200 qubits take $22\,\mu\text{s}$ with a 3-qubit window and $47\,\mu\text{s}$ with a 4-qubit window (about $110$ and $236\text{ ns/qubit}$, re-measured, constant from 14 to 200 qubits). The benchmark reports only the fraction of neighbouring pairs found anti-correlated, not an entanglement measure.
* Detailed Documentation: [`docs/bloch_zipper_mps_specification.md`](docs/bloch_zipper_mps_specification.md).

### 11. Heptal Qudit $SU(7)$ Modular Complementarity Engine (`t1000::heptal_wave`)
* **Radix-7 Hilbert State Evolution:** Native $d = 7$ qudit simulation simulating high-dimensional quantum systems ($7^N$ state spaces).
* **Modular Inversion Unitaries:** Sliding 2-qudit window (49 amplitudes = 784 bytes, strictly L1 resident) applying cyclic shift and reflection operations: $|j, k\rangle \mapsto |j, (k - j) \pmod 7\rangle$.
* **Exact Modular Complementarity:** Enforces deterministic pairwise complementary locking across the qudit chain: $(q_i + q_{i+1}) \equiv 0 \pmod 7$.
* **Throughput:** 200 qudits in about $4.9\text{ ms}$ ($24\,\mu\text{s/qudit}$, re-measured) with $100\%$ modular complementarity. Only a 49-amplitude window is simulated at a time; $7^{200}$ is the nominal size of the space the chain would span, not a state held or evolved in full.

### 12. Quantum State-Space Autoencoder & Sequence Classifier (`t1000::qssm`)
* **Quantum Mamba / Q-SSM Architecture:** Combines streaming 1D MPS sliding window recurrence ($W=3$, 128 bytes in L1 cache) with a dual-head latent bottleneck.
* **Dual-Head Topology:**
  * **Latent Classifier Head:** Extracts expectation observables $\langle Z_k \rangle$ from the compressed quantum state to output sequence class probabilities.
  * **Unzipping Decoder Head:** Reverses continuous Bloch steering to autoregressively reconstruct the input feature sequence $\hat{X}$ from the latent bottleneck.
* **Streaming throughput:** sequence lengths $T \in [10, 500]$ at $64\text{–}84\text{ ns per token}$ ($12\text{–}15.6\text{ million tokens/second}$, re-measured), with 100% test accuracy on a synthetic coherent-versus-chaotic burst task. The harder quench variant (`qssm_quench_bench`) ranges from 70.8% to 100% test accuracy across epochs. The full-suite benchmark's $1.8\,\mu\text{s}$ per token includes classification overhead.

### 13. Automated Clifford+T Single-Qubit Synthesis (`t1000::clifford_t`)
* **Cyclotomic Ring $\mathbb{Z}[\omega]$ Arithmetic:** Exact integer coordinate representation modulo $\omega^4 + 1 = 0$ ($\omega = e^{i\pi/4} = \frac{1+i}{\sqrt{2}}$) with algebraic norm, conjugation, and exact division by $\lambda = 1+\omega$.
* **Hierarchical Grid Solver:** Synthesizes arbitrary continuous single-qubit rotations ($R_z(\theta), R_x(\theta), R_y(\theta)$) into minimal-length discrete gate sequences over $\{H, S, S^\dagger, T, T^\dagger, X, Y, Z\}$.
* **Matsumoto-Amano Normal Form & Peephole Optimization:** Deterministic parity state-machine reduction with multi-pass peephole cancellations ($H^2 \to I$, $T T^\dagger \to I$, $T^2 \to S$, $S^2 \to Z$, $Z^2 \to I$), guaranteeing $\|U_{\text{exact}} - U_{\text{discrete}}\|_\infty \le \varepsilon$.
### 14. Heterogeneous Co-Processor Pipeline & 36-Qubit Cut-Bridge Verification (`cut_bridge_chsh_bench`, `cut_bridge_36q_bench`)
* **L3-Cache Resident Virtual 36-Qubit VQC:** Partitions a 36-qubit lattice into dual 18-qubit sub-registers joined by a single cut $CZ$ (this circuit form only, not general 36-qubit circuits) resident in 8.38 MB working set inside CPU L3 cache (avoiding 1.00 Terabyte dense statevector RAM).
* **Quasiprobability Contraction:** Decomposes cross-fragment $CZ$ bridge into local Pauli-$Z$ insertions, reconstructing exact non-local expectation values.
* **Exact Tsirelson Bell Witness Violation:** Evaluates $\langle \mathcal{B}_{\text{CHSH}} \rangle_{\text{bridge}} = \langle A_0 \otimes B_0 \rangle + \langle A_0 \otimes B_1 \rangle + \langle A_1 \otimes B_0 \rangle - \langle A_1 \otimes B_1 \rangle = 2.82842712$ across boundary bridge pair $(q_4, q_{13})$ ($|S - 2\sqrt{2}| = 2.22 \times 10^{-15}$) in **$35.86\text{ ms}$** inner AVX2 latency.
* **Mid-Circuit Checkpointing (`cut_bridge_36q_bench`):** Evaluates a 12-layer sandwich topology (`[Layers 1..6] -> [Cut CZ] -> [Layers 7..12]`) pre-computing prefix evolution once in **$316.50\text{ ms}$** and cloning cache-resident 18Q statevectors across 4 branches, confirming non-separable 36Q entanglement ($\Delta = +0.0001056229$) in **$1.608\text{ s}$**.
* **Intel iGPU OpenCL Offload & Native Rust Bridge (`t1000::gpu`):** Dynamically wires `QuantumState` SOA buffers directly into the Intel Xe Graphics OpenCL runtime using `CL_MEM_USE_HOST_PTR` unified memory mappings. Automatically routes dense registers $N \ge 22$ ($64\text{ MB}\text{–}1\text{ GB}$) to the iGPU, completing in about $9.5\text{ ms}$ against $17.4\text{ ms}$ on the CPU at 22 qubits ($1.8\times$, re-measured; the iGPU is slower below 20 qubits) with $1.0000000$ cross-backend fidelity (`igpu_bridge_bench`).
* Detailed Documentation: [`docs/findings/heterogeneous_co_processor_pipeline.md`](docs/findings/heterogeneous_co_processor_pipeline.md) and [`docs/findings/cut_bridge_36q_checkpointing.md`](docs/findings/cut_bridge_36q_checkpointing.md).

### 15. Quantum Vision & Subpopulation Sorter Engine (`birds_hybrid_qnn_train`, `path_a_vs_c_sorter_bench`)
* **Unitarity Rigidity Theorem ($W^{\dagger} W = \mathbb{I}$):** Rigorous mathematical proof and empirical confirmation that a parameterized unitary circuit cannot alter relative Hilbert state distances globally ($F = 2.5105$ invariant).
* **Active Measurement Orientation Sorter:** Demonstrates that the VQC acts as a coordinate frame rotator, aligning rigid state clusters with the eigenspaces of physical Pauli correlators ($+2.70\%$ separation gain in observable space $\mathbb{R}^{55}$).
* **Headless Pure Quantum Sorter (Path A):** Eliminating the 112 classical head parameters entirely and sorting samples directly via global magnetization $\text{sign}(\frac{1}{10}\sum \langle Z_i \rangle)$ boosts Worst-Group Accuracy (WGA) to **$61.91\%$** ($+5.10\%$), surging to **$71.81\%$** on the hardest minority confounder ($G_2$, Waterbirds on Land) while halving training latency to **$1.09\text{ s}$** (**$16.19\,\mu\text{s}$ per sample**).
* Detailed Documentation: [`docs/findings/quantum_sorter_vs_classifier_waterbirds.md`](docs/findings/quantum_sorter_vs_classifier_waterbirds.md).

### 16. Ultrametric Tree Sorter & Quantum Multiplexor (QMUX) Engine (`ultrametric_tree_sorter_bench`)
* **Hierarchical Factorization:** Partitions the Hilbert space into three explicit functional registers: $\mathcal{H} = \mathcal{H}_C \otimes \mathcal{H}_F \otimes \mathcal{H}_E$ (Coarse Class, Fine Subpopulation, and Spurious Entropy Sink).
* **Block-Diagonal QMUX Unitarity:** Coarse register $\mathcal{H}_C$ coherently routes class-conditioned sorting unitaries $W_F^{(y)}$ onto $\mathcal{H}_F$ ($U_{\text{QMUX}} = \sum_y |y\rangle\langle y|_C \otimes W_F^{(y)} \otimes \mathbb{I}_E$) preserving global unitarity ($W^\dagger W = \mathbb{I}$) while enforcing an ultrametric tree distance geometry.
* **Strong Triangle Inequality Verified:** Overlap matrices validate the strong triangle inequality ($d_U(u, v) \le \max(d_U(u, w), d_U(w, v))$) across identical, sibling, and disjoint tree leaves.
* **Extreme Subpopulation Throughput:** With only 45 quantum parameters (zero classical weights), reaches **$70.69\%$** coarse accuracy on the Waterbirds 11k test set with **$4.14\,\mu\text{s/sample}$** inference latency (**241,500 samples/sec**).
* Detailed Documentation: [`docs/findings/ultrametric_tree_sorter_qmux.md`](docs/findings/ultrametric_tree_sorter_qmux.md).

### 17. Glued-Trees Quantum Walk (`glued_trees_bench`)
* **Welded-tree walk (Childs et al. STOC 2003):** Continuous-Time Quantum Walk (CTQW) on two balanced binary trees welded at leaves by a random cycle ($2 \cdot (2^{n+1}-1)$ vertices).
* **Column-Symmetric Invariant Subspace:** Destructive interference cancels all paths into the orthogonal subspace, collapsing the exponential $2^n$ weld bottleneck into an exact 1D chain of length $2n+2$.
* **Hitting Time Scaling:** At depth $n = 10$ ($4,094$ vertices), classical hitting success drops to **$49.40\%$** within 10,000 steps ($7,373$ mean steps), while the simulated quantum walk reaches the exit root at $t=17.3$ with **$54.16\%$ peak probability**. The comparison is with an unbiased classical random walker on an explicitly given graph; a classical algorithm that reads the graph solves it in linear time, so this is not a demonstration of quantum speedup (the oracle-model separation of Childs et al. is a different claim).
* **Defect Steering Recovery:** Unitary phase counter-rotation restores coherence across the weld, achieving **$100.0\%$ defect recovery fidelity**.
* Detailed Documentation: [`docs/findings/glued_trees_exponential_speedup.md`](docs/findings/glued_trees_exponential_speedup.md).

### 18. Multi-Exit Topological Glued-Tree Valve & Quantum Router (`glued_tree_valve_bench`)
* **Multi-Branch Topological Routing:** Fuses the Ultrametric Sorter with Continuous-Time Quantum Walks on a $K$-way welded tree ($1\text{ Entrance Tree} \to K\text{ Exit Trees}$, dimension $(K+1)(n+1)$).
* **Aharonov-Bohm Dark-Subspace Extinction:** Modulating weld holonomy and phase coupling extinguishes transmission into non-target trees through destructive phase cancellation ($< 0.05\%$ leakage, contrast ratio $> 1,700:1$ at $K=2$, $> 520:1$ at $K=4$).
* **Deterministic Sorting vs Classical $1/K$ Dispersion:** Classical random walk hits target exit with only $49.0\%$ ($K=2$) or $25.0\%$ ($K=4$) probability; the Quantum Valve sorts deterministically with **$77.16\%$ to $58.68\%$** ($K=2$) and **$77.03\%$ to $61.29\%$** ($K=4$) peak probability.
* **Invariant Linear Group Velocity:** Longitudinal propagation speed $v_g \approx 1.33$ columns/unit time is strictly invariant under multi-branch fanout ($t_{\text{peak}} = 6.70 \to 14.15$ for $n=3 \to 8$).
* Detailed Documentation: [`docs/findings/glued_tree_topological_valve.md`](docs/findings/glued_tree_topological_valve.md).

### 19. Cascaded Multi-Tier Glued-Tree Valve & Deep Quantum Routing (`cascaded_tree_valve_bench`)
* **Multi-Stage Sequential Ballistic Routing:** Cascades coarse entrance trees through intermediate roots into fine subpopulation exit trees across dual sequential welds without intermediate measurement collapse.
* **Dual-Barrier Dark Subspace Extinction:** Cross-class and sibling non-target branches are extinguished in sequence ($< 0.04\%$ leakage), achieving contrast ratios of **$2,489 : 1$** ($2\times 2$, 4 exits) and **$831 : 1$** ($2\times 4$, 8 exits).
* **Quantum Peak vs Classical Trapping:** Quantum walker hits target exit with **$60.31\% \to 44.61\%$** peak probability, while classical random walk hits the $25.0\%$ ($2\times 2$) or $12.5\%$ ($2\times 4$) floor.
* **Linear Latency Scaling:** Multi-tier propagation latency scales linearly as $t_{\text{peak}} \approx \sqrt{2}\sum n_i$ with zero group velocity degradation across sequential welds.
* Detailed Documentation: [`docs/findings/cascaded_tree_valve_deep_routing.md`](docs/findings/cascaded_tree_valve_deep_routing.md).

### 20. Continuous Topological Walk Sorter with Weld Zeno Sentry (`topological_walk_sorter_bench`)
* **Continuous Hamiltonian Routing vs Discrete Gate Circuits:** Replaces discrete variational gate layers ($R_y, CZ$) with continuous-time Hamiltonian traversal ($e^{-i H_{\text{valve}} t}$) through a multi-tier cascaded glued-tree valve.
* **Immunity to Discrete Gate Noise Accumulation:** Under physical open-system dephasing ($T_1 = 45\,\mu\text{s}, T_2^* = 60\,\mu\text{s}, p_{\text{depol}} = 0.0020$), joint subpopulation classification accuracy jumps from **$24.50\%$** (discrete circuit QMUX) to **$82.00\%$** (continuous topological walk), with **$90.00\%$ coarse accuracy** and **$92.00\%$ fine accuracy**.
* **$719\times$ Contrast Explosion:** Extinction of incorrect branches increases from $2.85:1$ (discrete QMUX) to **$2,050.00:1$**.
* **Intermediate Weld Zeno Sentry:** Intersperse high-frequency projective monitoring at the intermediate weld roots, increasing target ballistic exit hitting probability from $0.4992 \to 0.5089$ and escaping the classical random walk dispersion trap ($25.20\%$).
### 21. 2D Plaquette Routing, Multi-Chiplet Seam Mitigation & Architecture Showdown (`plaquette_1234_bench`, `nighthawk_99_plaquette_bench`, `cepheus_108q_bench`)
* **Scope:** the fidelities in this section come from a noise-model simulation with assumed error rates (for example 0.8% intra-die and 2.2% seam two-qubit error); they are not hardware measurements. The only hardware data is §22.
* **Measurement-Based 4-Edge Plaquette Teleportation:** Replaces lossy multi-hop sequential SWAP chains ($1 \to 2 \to 4$) with single-tick 4-edge measurement-based teleportation across intermediate diagonal mediators ($1+(2,3)=4$).
* **Active Classical Pauli-Z Frame Recovery:** Software frame tracking decodes $X$-basis syndrome outcomes $s_2 \in \{0, 1\}$ with zero physical gate delay, doubling post-selection acceptance from **$48.4\% \to 96.8\%$** while locking Bell state fidelity at **$0.9606$**.
* **Global 20-Hop Cross-Chip Diagonal Routing:** Across the 120-qubit monolithic **IBM Quantum Nighthawk R2** ($12 \times 10$ grid), standard 60-CNOT SWAP cascades crash link fidelity to **$0.2119$**; 4-edge plaquette teleportation sustains **$0.6687$** (**$+45.7\%$ boost**) entirely within 384 bytes of CPU L1 cache.
* **Rigetti Cepheus-1-108Q Multi-Die Seam Mitigation:** Evaluates 108 qubits across twelve $3 \times 3$ modular chiplets (195 couplers with 51 inter-die bump-bond seams at $p=2.20\%$ vs $p=0.80\%$ intra-die). Constant-depth teleportation insulates the circuit from cascading seam errors, yielding a **$+5.58\%$ absolute purification boost** on 4-chiplet corner intersections ($0.8588 \to 0.9146$).
* Detailed Documentation: [`docs/findings/plaquette_routing_and_cepheus_seam_mitigation.md`](docs/findings/plaquette_routing_and_cepheus_seam_mitigation.md).

### 22. Empirical Physical QPU Quantum Walk: Rigetti Cepheus-1-108Q (`experiments/cepheus_glued_trees_30q`)
* **Physical Hardware Execution:** 1,000 physical projective measurement shots dispatched to **Rigetti Cepheus-1-108Q** (Modular 12-Chiplet Superconducting Transmon QPU, Job ID: `ef387a4a-1898-4214-a686-8ee1c088942a`, Backend: `7433acb8-ae52-4bc9-9030-6df68c696538`).
* **Childs et al. Welded-Tree Traversal:** 30-qubit continuous-time quantum walk across dual balanced binary trees ($n=3$, 8 columns) welded at leaves, collapsed into an 8-column symmetric invariant subspace.
* **Exit-wire marginal:** the physical probability of finding the terminal exit wire ($q_7$) excited is **$65.20\%$** ($652/1000$ shots). The eight wire marginals ($0.478, 0.427, 0.378, 0.240, 0.269, 0.382, 0.663, 0.652$) sum to $3.49$, where one walker on eight wires would give $1$: the register is nearly random (a fair coin gives $0.5$ on every wire), and the exit wire is no higher than its neighbour $q_6$ ($66.3\%$). Comparing it with the $\sim 1.67\%$ single-walker hitting probability of a classical random walk is not like for like, so this run does not demonstrate ballistic transport or a quantum advantage.
* **Simulation comparison:** T-1000 predicted a single-walker exit probability of $77.22\%$ at $t=6.80$; the measured wire marginal was $65.20\%$ (ratio $84.43\%$, $|\Delta P| = 12.02\%$). The two quantities are different observables (see above), so the ratio is not a predictive accuracy; the large departure is consistent with transmon relaxation ($T_1 = 32.5\,\mu\text{s}$), inter-die seam error ($p_{\text{seam}} = 0.0220$) and readout noise ($\approx 2.5\%$).
* **Sovereign Visual Suite (300 DPI):**
  - **Overview Showdown:** [`cepheus_glued_trees_showdown.png`](experiments/cepheus_glued_trees_30q/cepheus_glued_trees_showdown.png) (1.34 MB)
  - **Hardware Alone:** [`cepheus_physical_alone.png`](experiments/cepheus_glued_trees_30q/cepheus_physical_alone.png) (1.15 MB)
  - **Side-by-Side Comparison:** [`cepheus_sim_vs_run_side_by_side.png`](experiments/cepheus_glued_trees_30q/cepheus_sim_vs_run_side_by_side.png) (839 KB)
  - **Heatmap Showdown:** [`cepheus_sim_vs_run_heatmap.png`](experiments/cepheus_glued_trees_30q/cepheus_sim_vs_run_heatmap.png) (1.02 MB) with continuous spatiotemporal density, $8\times 8$ empirical Pauli correlation matrix $\langle Z_i Z_j \rangle$, and Frobenius error norm.
* Detailed Documentation: [`docs/findings/cepheus_physical_glued_trees_qpu_run.md`](docs/findings/cepheus_physical_glued_trees_qpu_run.md) and [`experiments/cepheus_glued_trees_30q/README.md`](experiments/cepheus_glued_trees_30q/README.md).

### 23. Lie-Algebraic Geometric Quantum Graph Neural Network (`t1000::qgnn`)
* **Gradient variance (not evidence of barren-plateau resilience):** maps OpenSMILES molecular bond graphs into Lie-algebraic equivariant graph convolutions with shared weights. The measured gradient variance of one parameter is $4.29\times 10^{-2}$ at every size from 4 to 14 qubits (a local observable on shared-weight layers, so it cannot depend on size), and the hardware-efficient baseline's variance ($0.08\text{–}0.16$) is larger at every size tested and does not decay in this range either. The sweep therefore does not show resilience.
* **Exact Permutation Equivariance:** Invariant node projection and Heisenberg exchange entangling layers ($U = \exp(-i \frac{\phi}{2}(XX+YY+\Delta ZZ))$) guarantee exact permutation invariance under graph automorphisms ($|\Delta y| = 0.0000 \times 10^0$).
* **Inference latency:** about $2.7\,\mu\text{s}$ per molecule (about $370{,}000$ molecules/second, re-measured on a 6-node benzene graph).
* **Sovereign Publication Visual:** 4-panel showdown chart rendered at 300 DPI ([`results/charts/qgnn_geometric_showdown.png`](results/charts/qgnn_geometric_showdown.png)).
* Detailed Documentation: [`docs/findings/geometric_qgnn_barren_plateau_resilience.md`](docs/findings/geometric_qgnn_barren_plateau_resilience.md).

### 24. Full Quantum Neural Network (QNN) Suite Benchmark (`full_qnn_suite_bench`)
* **Seven-pillar suite (re-measured on the audit host; latencies vary by run):**
  1. *Geometric QGNN:* about $2.6\,\mu\text{s}$ per molecule (benzene graph), permutation drift $0$.
  2. *Continuous walk sorter:* about $37\,\mu\text{s}$ per sample, $400:1$ routing contrast, $45.7\%$ ballistic peak.
  3. *Ultrametric tree sorter:* about $0.07\,\mu\text{s}$ per sample, $0$ strong-triangle-inequality violations.
  4. *Q-SSM (1D MPS):* about $1.8\,\mu\text{s}$ per token including classification, $128\text{ B}$ window, reconstruction MSE $0.19$.
  5. *Radix-7 heptal MPS:* about $420\text{–}470\,\mu\text{s}$ per molecule, $784\text{ B}$ active window.
  6. *Dense molecular VQC (14Q):* about $6\text{–}7\,\mu\text{s}$ per pass, $262\text{ KB}$ state, norm drift $0$.
  7. *Adjoint differentiation:* 96 parameters on 12 qubits in about $0.4\text{ ms}$, about $50\times$ faster than parameter shift (measured; maximum difference $3\times 10^{-16}$). An earlier version of this benchmark computed that speedup from a formula ($153.6\times$); the figure was removed.
### 25. Unitary Quantum Cellular Automata via Fractional Fermi Seas (`t1000_conway`)
* **Physical Basis:** Inspired by Guardado-Sanchez et al. (Innsbruck, arXiv:2602.17657v2), couples local Conway cell states to closed interaction holonomy cycles across a Confinement-Induced Resonance ($g_{1D}: 0 \to +\infty \to -\infty \to 0$), realizing Haldane Generalized Exclusion Statistics ($\alpha = \ell = 2$).
* **Exact Global Unitarity:** Split-step checkerboard decomposition ($U_{\text{step}} = U_{\mathcal{B}} \cdot U_{\mathcal{A}}$) maintains exact machine-precision unitarity ($|\|\psi\|^2 - 1.0| \le 1.55 \times 10^{-15}$) across thousands of steps with zero numerical dissipation.
* **Quantum Glider Collision S-Matrix:** Soliton wave-packet scattering matrix with phase-tunable interferometry $\Delta \theta \in [0, 2\pi]$ generating maximal bipartite entanglement $S = \ln 2 \approx 0.6931\text{ nats}$ (1 ebit) at $\Delta \theta = \pi/2$ (Mach-Zehnder beam splitter).
* **Multi-Scale AVX2+FMA SIMD:** 256-bit vectorized 4-cell batch evolution delivering up to **$1,513,897\text{ cells/second}$** on $16\times 16$ ($5.4\times$ speedup) and scaling stably to $64\times 64$ ($4,096\text{ cells}$).
* **Zero Core Workspace Pollution:** Completely quarantined in isolated sub-crate `crates/t1000_conway`.
* **Interactive 3D Web Visualizer:** Zero-dependency HTML5/Canvas dashboard (`web/qgol.html`) featuring 3D isometric column rendering, continuous wavefield/momentum sea ($n(k) = 0.5000$ plateau), $\mathbb{CP}^2$ Hilbert simplex projection, interactive click-to-paint cell basis selection, and real-time phase tuning.
* Detailed Documentation: [`docs/findings/qgol_fractional_fermi_sea_qca.md`](docs/findings/qgol_fractional_fermi_sea_qca.md).

### 26. Fault-Tolerant Planar Surface Code Lattice Surgery & Magic Distillation (`t1000::qec`)
* **18-Qubit Dual-Patch Lattice Surgery:** Complete statevector evolution across two distance-3 planar surface code patches ($q_0\dots q_8$ and $q_9\dots q_{17}$, $262,144$ states resident in 4 MB L3 cache). Synthesizes non-transversal logical CNOT via boundary $Z$-merge ($M_{ZZ}$) and $X$-merge ($M_{XX}$) cycles without physical defect routing.
* **Exact Truth Table & Bell Pair Generation:** Verifies all 4 computational basis transitions ($|00\rangle_L \mapsto |00\rangle_L, |01\rangle_L \mapsto |01\rangle_L, |10\rangle_L \mapsto |11\rangle_L, |11\rangle_L \mapsto |10\rangle_L$) with exact machine-precision norm preservation ($1.000000000000$), generating a maximally entangled logical Bell pair $|\Phi^+\rangle_L$ with unit fidelity ($F = 1.000000000000$) and logical correlations $\langle Z_1 Z_2 \rangle = +1.0, \langle X_1 X_2 \rangle = +1.0$ in $25.15\text{ ms}$.
* **Bravyi-Kitaev 15-to-1 Distillation Factory:** Punctured quantum Reed-Muller $[[15, 1, 3]]$ CSS code with 14 commuting stabilizer generators. Achieves **$1,145.4\times$ cubic error suppression** ($\epsilon_{\text{out}} = 35\epsilon_{\text{in}}^3$, suppressing $0.5\%$ input error down to $4.365 \times 10^{-6}$) at $244.8\text{ trials/second}$.
* **Active Boundary Error Correction:** Injects physical Pauli bit-flips on boundary data qubits, extracts discrete syndromes, and restores $100\%$ logical state fidelity ($F = 1.000000000000$).
* **Sovereign Publication Visual:** 4-panel showdown chart rendered at 300 DPI ([`results/charts/qec_lattice_surgery_showdown.png`](results/charts/qec_lattice_surgery_showdown.png)).
* Detailed Documentation: [`docs/findings/fault_tolerant_lattice_surgery_and_magic_distillation.md`](docs/findings/fault_tolerant_lattice_surgery_and_magic_distillation.md).

### 27. Real-Time Surface Code Threshold Engine & AVX-VNNI Minimum-Weight Decoder (`vnni_threshold_bench`)
* **Closed-Loop Monte Carlo Threshold Simulation:** Full physical depolarizing error channel sweeping error rates $p \in [0.001, 0.150]$ across 3.5 Million Monte Carlo shots on the rotated $[[9, 1, 3]]$ planar surface code.
* **Pseudo-threshold ($8.14\%$):** the logical error rate of the distance-3 code crosses the failure probability of nine unencoded qubits, $P_{\text{unencoded}} = 1 - (1 - \frac{2}{3}p)^9 \approx 6p$, at $p = 8.14\%$. Against a single unencoded qubit ($2p/3$) the crossing is near 5%. This is a code-capacity result (perfect syndrome extraction), not a circuit-level fault-tolerance threshold, which needs several code distances.
* **Quadratic Error Suppression ($d = 3$):** In the sub-threshold regime ($p \le 0.05$), logical error scales strictly quadratically as $P_L \approx 24.0 \cdot p^2$, delivering up to **$41.67\times$ error suppression** at $p = 0.001$ ($P_L = 0.014\%$ vs $P_{\text{unencoded}} = 0.60\%$).
* **Extreme Throughput Real-Time Decoding:** Evaluates syndrome extraction and minimum-weight homologous correction lookup in about $42\text{–}58\text{ ns per shot}$ (about $21\text{ Million decodings/sec}$, re-measured), completely bypassing slow graph-matching heuristic trees and enabling real-time QEC loop closure at hardware clock speeds.
* **Zero-Allocation L1-Resident Architecture:** Pre-allocated 16-element boundary/bulk correction lookup tables fit entirely in $128\text{ bytes}$ of CPU L1 data cache with zero dynamic heap allocation in inner decoding loops.
* **Sovereign Publication Visual:** 4-panel showdown chart rendered at 300 DPI ([`results/plots/vnni_qec_threshold_dynamics.png`](results/plots/vnni_qec_threshold_dynamics.png)).
* Detailed Documentation: [`docs/findings/vnni_qec_threshold_dynamics.md`](docs/findings/vnni_qec_threshold_dynamics.md).

### 28. Real-Symmetric Molecular Eigensystem Engine & Zero-Allocation QM9 Streaming Ingestion (`molecular_eigensystem_bench`)
* **Data required:** the figures below come from the author's run with the roboBohr table (`Ground_state_energies.zip`) and `QM9.zip`. Without them `molecular_eigensystem_bench` evaluates 0 molecules, so they are not reproducible from this repository.
* **50x50 Coulomb Matrix Householder-QL Eigensolver (`t1000::simd::eigensystem`, `t1000::chemistry::coulomb`):** Vectorized bare-metal symmetric reduction of dense $50 \times 50$ Coulomb matrices ($1,275$ packed upper-triangular entries) into exact 50-dimensional eigenspectra at **$14,352.5\text{ molecules/second}$** ($69.67\,\mu\text{s}$ per molecule).
* **Machine-Precision Spectral Preservation:** Guaranteed spectral trace conservation ($\sum \lambda_i = \mathrm{Tr}(C)$) with maximum drift bounded at **$2.73 \times 10^{-12}$**, verifying exact mathematical rigor across thousands of real molecules from `roboBohr.csv`.
* **Complete Invariance under $SO(3) \times S_N$:** Reconstructed eigenspectra are strictly invariant under molecular 3D rotation, translation, and identical atom permutation, providing an ideal coordinate-free representation for quantum VQC and QGNN training.
* **Zero-Allocation QM9 Streaming Ingestion Substrate (`t1000::chemistry::qm9`):** Reads compressed `.xyz` structures directly from `QM9.zip` in memory without temporary disk extraction, ingesting 15 scalar B3LYP DFT properties ($U_0, \Delta\epsilon, \mu, \alpha, C_v$), Cartesian geometries, and compiling 19-dimensional quantum Hilbert feature maps at **$4,056.4\text{ molecules/second}$** ($246.53\,\mu\text{s}$ per molecule).
* **Sovereign Publication Visual:** 4-panel showdown chart rendered at 300 DPI ([`results/plots/molecular_eigensystem_dynamics.png`](results/plots/molecular_eigensystem_dynamics.png)).
* Detailed Documentation: [`docs/findings/molecular_eigensystem_and_qm9_ingestion.md`](docs/findings/molecular_eigensystem_and_qm9_ingestion.md).

---

### 29. Ab Initio Molecular VQE (`vqe_molecules`, `vqe_dissociation`)
* **Computed, not tabulated:** STO-3G Gaussian integrals from the geometry (McMurchie-Davidson), restricted Hartree-Fock with DIIS, a frozen-core active space, and the Jordan-Wigner qubit Hamiltonian (14 to 4,688 Pauli terms on 4 to 16 qubits).
* **Exact reference:** full configuration interaction by Lanczos in the particle-number sector.
* **UCCSD-VQE with analytic gradients:** all nine Period 1-2 molecules (H₂, HeH⁺, LiH, HF, BeH₂, H₂O, BH₃, NH₃, CH₄) land within chemical accuracy (1.59 mHa) of FCI; the largest error is 0.37 mHa (BeH₂).
* **Bond stretching:** H₂ Hartree-Fock is 278 mHa off at 3 Å while the VQE tracks the exact curve; at 6 Å the exact energy equals twice the hydrogen-atom energy.
* **Strong correlation (`vqe_strong_correlation`):** stretched H₄ and H₆ chains, the H₄ rectangle through the square, and N₂ dissociation, against FCI. ADAPT-VQE is inside chemical accuracy at 12 of 13 H₄-chain points and 11 of 13 H₆ points, UCCSD at 7 and 4, CCSD at 5 and 5 (it overshoots below the exact energy or stops converging). On N₂ ADAPT needs a 1e-4 gradient tolerance (9 of 12) and is trapped at 82 mHa beyond 2.8 Å. Details: [`docs/findings/vqe_strong_correlation_adapt_vs_uccsd.md`](docs/findings/vqe_strong_correlation_adapt_vs_uccsd.md).
* **Noise study (`vqe_noise`):** UCCSD compiled to Pauli rotations (64 to 36,480 CNOTs, naive), depolarising/readout/shot noise, shot budgets (4.9e4 to 2.6e8 for 1.59 mHa) and SPSA on sampled energies; gate error rather than shot noise is the limit.
* Validated against the Szabo & Ostlund H₂ tables, published STO-3G Hartree-Fock energies, and an independent determinant-basis Hamiltonian. Detailed documentation: [`docs/vqe.md`](docs/vqe.md).

## System Startup & Execution Guide

> **Where results go.** Benchmark and training binaries write to a git-ignored scratch
> directory (`target/bench-scratch/<host>/`) by default, so a run never overwrites a saved
> result. Add `--record` (or set `T1000_RECORD=1`) to save: the reference machine writes the
> canonical `results/` path, any other machine writes `results/hosts/<host>/`. Trainers keep a
> recorded run only if its held-out metric improves. See
> [docs/host_tuning_and_result_recording.md](docs/host_tuning_and_result_recording.md) for host
> tuning, the fused execution paths, measured numbers and how to onboard a new machine.

The `T-1000` sovereign engine comprises 10 specialized quantum and neural systems. Below are the exact startup, execution, and verification commands for each system:

---

### System 1: Lab Bench (web dashboard)
A bench for running things: quantum algorithms with shots, a gate-list circuit composer that runs on the host or the Basys 3, probability tools (Born-rule sampling with chi-square tests, Porter-Thomas statistics, quantum against classical walks on graphs you define), exact number theory with digit roots and the times-table circle, and geometry (Bloch sphere, Hopf fibres, 4-polytopes). Every panel names the source of its numbers and each source is tested. Recorded results of earlier benchmark runs are on a separate archive page.

```bash
python web/serve.py              # opens http://localhost:8000/
python web/serve.py --no-browser
T1000_QPU_PORT=COM6 python web/serve.py   # serial port of the Basys 3

node web/test_lab_nt.js && node web/test_lab_geom.js && python web/test_lab_api.py
```

* **Pages:** lab bench `http://localhost:8000/`; recorded results `http://localhost:8000/archive.html`; Quantum Life QCA simulator `http://localhost:8000/qgol.html`; health `http://localhost:8000/api/status`.
* Details, endpoints and the tests behind each tab: [`docs/dashboard_and_circuit_composer.md`](docs/dashboard_and_circuit_composer.md).

---

### System 2: Fixed-Point Statevector Engine (AMD Artix-7 Basys 3)
Bit-exact model, golden vectors, simulation and hardware conformance for the board on `COM6 @ 3,000,000 Baud`.

```bash
# Regenerate the golden vectors from the Rust model
cargo run --release -p t1000 --bin qpu_vectors -- fpga/1000_qpu/sim/vectors

# Fixed-point precision study over the molecule catalogue
cargo run --release -p t1000 --bin qpu_precision

# Simulate the RTL against the vectors (Vivado xsim)
python fpga/1000_qpu/sim/run_sim.py

# Replay the same vectors on the board
.venv\Scripts\python.exe scripts\fpga\verify_1000_qpu.py --port COM6
```

---

### System 3: Core SIMD AVX2 Quantum Circuit & Random Circuit Sampling (RCS) Engine
High-throughput fused 256-bit AVX2+FMA state evolution and statistical chaos verification.

```bash
# Execute deep Random Circuit Sampling frontier benchmark (default: 18Q, 36 layers, 1,945 gates)
cargo run --release -p t1000 --bin frontier_rcs -- --qubits 18 --depth 36

# Run customized fast RCS benchmark (e.g. 14 qubits, 20 layers)
cargo run --release -p t1000 --bin frontier_rcs -- --qubits 14 --depth 20

# Run open-system Lindbladian decoherence & physical noise transition benchmark (M149)
cargo run --release -p t1000 --bin rcs_coherence_chaos_bench

# Render 300 DPI Sovereign Visual of RCS Physical Noise Transition
python scripts/charting/chart_rcs_coherence_chaos.py

# Run Quantum Coherence Frontier Push (Dense 26Q + Tensor Network 64Q + Heptal 7^9 qudits)
cargo run --release -p t1000 --bin coherence_frontier_push

# Run Orthogonal K-Patch Proxy RCS Showdown (PyQrack QrackAceBackend vs T-1000)
python scripts/benchmarks/t1000_orthogonal_proxy_showdown.py

# Render 300 DPI Sovereign Visual of Orthogonal Proxy Showdown
python scripts/charting/chart_orthogonal_proxy_showdown.py
```

---

### System 4: Fault-Tolerant QEC & Magic State Distillation Systems (M151, M153, M156)
Universal Eastin-Knill bypass, transversal Steane-wheel Clifford gates, and real-time surface code decoders.

```bash
# Run Universal FT-QNN Training with Bravyi-Kitaev 15-to-1 Magic State Distillation (Monograph M156)
cargo run --release -p t1000 --bin qm9_magic_w7_ft_train

# Render Sovereign Universal FT-QNN Magic State Distillation Chart (Monograph M156)
python scripts/charting/chart_qm9_magic_w7_training.py

# Run 18-Qubit Transversal [[7, 1, 3]] Steane-Wheel FT-QNN Training with In-Register Ancillas (Monograph M153)
cargo run --release -p t1000 --bin qm9_transversal_w7_train

# Render Sovereign Transversal FT-QNN Dynamics Chart (Monograph M153)
python scripts/charting/chart_qm9_transversal_w7_training.py

# Run 18-Qubit Fault-Tolerant QM9 VQC Training with Mid-Circuit VNNI Filtering (Monograph M151)
cargo run --release -p t1000 --bin qm9_fault_tolerant_train

# Render Sovereign Fault-Tolerant QM9 VQC Dynamics Chart (Monograph M151)
python scripts/charting/chart_qm9_fault_tolerant_training.py

# Run Closed-Loop AVX-VNNI Real-Time Surface Code Threshold Sweep (3.5M Monte Carlo shots)
cargo run --release -p t1000 --bin vnni_threshold_bench

# Render 300 DPI Sovereign Visual of VNNI QEC Threshold Dynamics
python scripts/charting/chart_vnni_threshold.py

# Run AVX-VNNI hardware-accelerated syndrome and stabilizer benchmark (CPU INT8 VNNI)
cargo run --release -p t1000 --bin vnni_syndrome_bench

# Run Fault-Tolerant Dual-Patch Lattice Surgery & Bell Pair Generation Benchmark
cargo run --release -p t1000 --bin qec_fault_tolerance_bench

# Render 300 DPI Sovereign Visual of Lattice Surgery Dynamics
python scripts/charting/chart_qec_fault_tolerance.py
```

---

### System 5: 36-Qubit Virtual Cut-Bridge & Heterogeneous Co-Processor Pipeline (M155)
$131,072\times$ L3 cache compression ($1.00\text{ TB} \to 8\text{ MB}$) and Intel Xe Graphics OpenCL UMA zero-copy bridge.

```bash
# Run 36-Qubit Virtual Cut-Bridge & Heterogeneous Co-Processor Benchmark (Monograph M155)
cargo run --release -p t1000 --bin virtual_36q_cut_bridge_bench

# Render Sovereign 36-Qubit Virtual Cut-Bridge Dynamics Chart (Monograph M155)
python scripts/charting/chart_virtual_36q_cut_bridge.py

# Run 36-Qubit Mid-Circuit Checkpointed Cut-Bridge Benchmark (CPU AVX2+FMA)
cargo run --release -p t1000 --bin cut_bridge_36q_bench

# Run 36-Qubit Cross-Bridge CHSH Bell Inequality Verification Benchmark (CPU AVX2+FMA)
cargo run --release -p t1000 --bin cut_bridge_chsh_bench

# Run Intel Graphics (Xe iGPU) Zero-Copy UMA Dense Statevector Benchmark (OpenCL)
cargo run --release -p t1000 --bin igpu_bridge_bench

# Run concurrent Heterogeneous Co-Processor Pipeline (CPU AVX2 + Intel iGPU OpenCL)
python scripts/qec/heterogeneous_bridge_pipeline.py
```

---

### System 6: Molecular Latent Space & Continuous-Time Quantum Walk (CTQW) (M154)
Chemical featurization, Coulomb Hamiltonian eigensolvers, and ballistic topological walk screeners.

```bash
# Run QM9 Continuous-Time Topological Quantum Walk Chemical Screener & Zeno Filter (Monograph M154)
cargo run --release -p t1000 --bin continuous_walk_qm9_filter

# Render Sovereign QM9 Continuous Topological Walk Screening Dynamics Chart (Monograph M154)
python scripts/charting/chart_qm9_continuous_walk.py

# Run Continuous Topological Walk Molecular Screening on 1,000 ZINC Drug Candidates
cargo run --release -p t1000 --bin zinc_topological_walk_screen

# Render Sovereign ZINC Topological Screening Dynamics Chart
python scripts/charting/chart_zinc_topological_screen.py

# Execute ZINC 250k Whole-Library Dual Search (Search 1 Pareto Leads + Search 3 Anomalies)
cargo run --release -p t1000 --bin zinc_dual_search

# Render Sovereign Visuals for ZINC Pareto Leads and Quantum Anomalies
python scripts/charting/chart_zinc_pareto.py
python scripts/charting/chart_zinc_anomalies.py

# Run Real-Symmetric Molecular Eigensystem & Zero-Allocation QM9 Ingestion Benchmark
cargo run --release -p t1000 --bin molecular_eigensystem_bench

# Render 300 DPI Sovereign Visual of Molecular Eigensystem Dynamics
python scripts/charting/chart_molecular_eigensystem.py

# Run Cascaded Multi-Tier Tree Valve Benchmark (Dual-Weld Deep Quantum Routing)
cargo run --release -p t1000 --bin cascaded_tree_valve_bench

# Render 300 DPI Sovereign Visual of Cascaded Tree Valve Dynamics
python scripts/charting/chart_cascaded_tree_valve.py

# Run Multi-Exit Glued-Tree Valve & Quantum Router Benchmark (K-Way Routing)
cargo run --release -p t1000 --bin glued_tree_valve_bench

# Render 300 DPI Sovereign Visual of Glued-Tree Valve Dynamics
python scripts/charting/chart_glued_tree_valve.py

# Run Glued Trees Traversal Exponential Speedup Benchmark (Childs et al. CTQW)
cargo run --release -p t1000 --bin glued_trees_bench

# Render 300 DPI Sovereign Visual of Glued-Trees Traversal Dynamics
python scripts/charting/chart_glued_trees.py
```

---

### System 7: Quantum Vision, Ultrametric Sorters & Sequence Autoencoders
Headless geometric coordinate sorters, non-Archimedean tree factorizations, and quantum state-space models.

```bash
# Run Full Quantum Neural Network (QNN) Suite Benchmark (All 7 Mathematical Pillars)
cargo run --release -p t1000 --bin full_qnn_suite_bench

# Render 300 DPI Sovereign Full QNN Suite Showdown Visual
python scripts/charting/chart_full_qnn_suite.py

# Run Lie-Algebraic Geometric QGNN Molecular Benchmark & Barren Plateau Showdown
cargo run --release -p t1000 --bin qgnn_molecular_bench

# Render 300 DPI Sovereign QGNN Showdown Visual
python scripts/charting/chart_qgnn_showdown.py

# Run Ultrametric Tree Sorter QMUX Benchmark (Waterbirds Hierarchical Multi-Register)
cargo run --release -p t1000 --bin ultrametric_tree_sorter_bench

# Render 300 DPI Sovereign Visual of Ultrametric Tree Sorter Dynamics
python scripts/charting/chart_ultrametric_tree.py

# Run Headless Pure Quantum Sorter vs Data Re-Uploading Showdown (Waterbirds 11k)
cargo run --release -p t1000 --bin path_a_vs_c_sorter_bench

# Run Hilbert Space Fidelity & Observable Orientation Sorter Benchmark
cargo run --release -p t1000 --bin fidelity_orthogonality_bench

# Run 10-Qubit Waterbirds Hybrid Transfer Quantum Neural Network Training (15 Epochs)
cargo run --release -p t1000 --bin birds_hybrid_qnn_train -- 15

# Render 300 DPI Sovereign Visual of Waterbirds Hybrid QNN Dynamics
python scripts/charting/chart_birds_hybrid.py

# Run 10-Qubit MNIST Quantum Neural Network & Coherence Benchmark (Adjoint AVX2)
cargo run --release -p t1000 --bin mnist_qnn_train

# Render 300 DPI Sovereign Visual of MNIST QNN Training Dynamics
python scripts/charting/chart_mnist_qnn.py

# Run Quantum State-Space Autoencoder (Q-SSM Mamba) Streaming Benchmark
cargo run --release -p t1000 --bin qssm_autoencoder_bench

# Run Zero-Data Real-Time Quantum Quench Autoencoder & Dynamical Phase Classifier
cargo run --release -p t1000 --bin qssm_quench_bench

# Render 300 DPI Sovereign Visuals of Lattice Quench and Phase Transition Dynamics
python scripts/charting/chart_lattice_quench.py
python scripts/charting/chart_lattice_phase.py

# Run 1D MPS Bloch-Zipper and Heptal SU(7) Complementarity Benchmarks (14 to 200 qubits/qudits)
cargo run --release -p t1000 --bin bloch_zipper_bench
```

---

### System 8: Superconducting 2D Plaquettes & Cloud QPU Dispatch (IBM Nighthawk & Rigetti Cepheus)
Dual-graph error mitigation, boundary spoke cancellation, and live cloud hardware dispatch.

```bash
# Run IBM Nighthawk R2 99-Plaquette Monolithic Architecture Benchmark
cargo run --release -p t1000 --bin nighthawk_99_plaquette_bench

# Render 300 DPI Sovereign Visual of IBM Nighthawk 99-Plaquette Heatmap & Teleportation
python scripts/charting/chart_nighthawk_99_plaquettes.py
python scripts/charting/chart_nighthawk_layout.py

# Run Rigetti Cepheus-1-108Q 12-Chiplet Modular Seam Mitigation Benchmark
cargo run --release -p t1000 --bin cepheus_108q_bench

# Render 300 DPI Sovereign Visual of Rigetti Cepheus 108Q Seam Mitigation
python scripts/charting/chart_cepheus_108q.py

# Run Elementary 4-Qubit Plaquette Routing Benchmark Suite (Protocols 1-6)
cargo run --release -p t1000 --bin plaquette_1234_bench

# Run 30-Qubit Glued-Tree Quantum Walk T-1000 Simulation Pass
python experiments/cepheus_glued_trees_30q/runner.py --target t1000

# Dispatch / Poll Physical Run on Rigetti Cepheus-1-108Q (OpenQuantum Cloud)
python experiments/cepheus_glued_trees_30q/dispatch_to_openquantum_cepheus.py

# Render 300 DPI Sovereign Visual Suite for Cepheus (Physical Alone, Side-by-Side, Heatmap)
python experiments/cepheus_glued_trees_30q/chart_cepheus_physical_alone.py
python experiments/cepheus_glued_trees_30q/chart_cepheus_side_by_side.py
python experiments/cepheus_glued_trees_30q/chart_cepheus_heatmap_showdown.py
```

---

### System 9: Unitary Quantum Cellular Automata & Fractional Fermi Solitons (M152)
Interaction holonomy cycles across Confinement-Induced Resonances and Mach-Zehnder soliton collision interferometry.

```bash
# Run Quantum Conway QCA Multi-Scale Benchmark & Soliton S-Matrix Simulation
cargo run --release -p t1000_conway --bin qgol_sim

# Render Sovereign QCA Holonomy & Topological Soliton Chart (Monograph M152)
python scripts/charting/chart_qgol_holonomy.py

# Interactive 3D Web Visualizer:
# Start the web server (python web/serve.py) and navigate in any WebGL browser to:
# http://localhost:8000/qgol.html
```

---

### System 10: Automated Test Suite & Full Scientific Charting Generation
Live disk-grounded verification of workspace integrity and batch generation of publication graphics.

```bash
# Run all unit tests, integration tests, and oracle verifications across the entire workspace
cargo test --workspace

# Fast workspace compilation and lint hygiene check
cargo check --workspace

# Batch generate all 39 publication-grade 300 DPI scientific charts in one command
python scripts/charting/chart_all.py
```

* **Universal Fault-Tolerant Quantum Neural Networks: Eastin-Knill Bypass via Magic State Distillation (Monograph M156):** [`docs/findings/universal_ft_qnn_magic_state_w7_injection.md`](docs/findings/universal_ft_qnn_magic_state_w7_injection.md).
* **36-Qubit Virtual Cut-Bridge & Heterogeneous Co-Processor Pipeline (Monograph M155):** [`docs/findings/virtual_36q_cut_bridge_igpu.md`](docs/findings/virtual_36q_cut_bridge_igpu.md).
* **Continuous-Time Topological Quantum Walk on Ab Initio Coulomb Laplacians & AVX-VNNI Zeno Confinement (Monograph M154):** [`docs/findings/qm9_continuous_topological_walk_screening.md`](docs/findings/qm9_continuous_topological_walk_screening.md).
* **18-Qubit Transversal [[7, 1, 3]] Steane-Wheel FT-QNN & In-Register Physical Ancillas (Monograph M153):** [`docs/findings/qm9_transversal_w7_fault_tolerant_training.md`](docs/findings/qm9_transversal_w7_fault_tolerant_training.md).
* **Non-Abelian Geometric Holonomy, Quantized Vortex Dipoles & Floquet Chaos Scrambling in Unitary QCA (Monograph M152):** [`docs/findings/qgol_geometric_holonomy_and_topological_solitons.md`](docs/findings/qgol_geometric_holonomy_and_topological_solitons.md).
* **18-Qubit Fault-Tolerant QM9 VQC Training & Mid-Circuit AVX-VNNI Syndrome Filtering (Monograph M151):** [`docs/findings/qm9_fault_tolerant_vqc_training_and_midcircuit_vnni_filtering.md`](docs/findings/qm9_fault_tolerant_vqc_training_and_midcircuit_vnni_filtering.md).
* **QM9 Ab Initio VQC Training, Atomic Baseline Decomposition & Dual-Angle Quantum Phase Map (Monograph M150):** [`docs/findings/qm9_ab_initio_vqc_training_and_baseline_decomposition.md`](docs/findings/qm9_ab_initio_vqc_training_and_baseline_decomposition.md).
* **Real-Symmetric Molecular Eigensystem & Zero-Allocation QM9 Ingestion:** [`docs/findings/molecular_eigensystem_and_qm9_ingestion.md`](docs/findings/molecular_eigensystem_and_qm9_ingestion.md).
* **Closed-Loop AVX-VNNI Real-Time Surface Code Threshold Dynamics:** [`docs/findings/vnni_qec_threshold_dynamics.md`](docs/findings/vnni_qec_threshold_dynamics.md).
* **Fault-Tolerant Lattice Surgery & Magic State Distillation:** [`docs/findings/fault_tolerant_lattice_surgery_and_magic_distillation.md`](docs/findings/fault_tolerant_lattice_surgery_and_magic_distillation.md).
* **Full Quantum Neural Network (QNN) Suite Benchmark & Showdown:** [`docs/findings/full_quantum_neural_network_suite_benchmarks.md`](docs/findings/full_quantum_neural_network_suite_benchmarks.md).
* **Lie-Algebraic Geometric QGNN & Barren Plateau Resilience:** [`docs/findings/geometric_qgnn_barren_plateau_resilience.md`](docs/findings/geometric_qgnn_barren_plateau_resilience.md).
* **Rigetti Cepheus-1-108Q Physical QPU Quantum Walk Findings:** [`docs/findings/cepheus_physical_glued_trees_qpu_run.md`](docs/findings/cepheus_physical_glued_trees_qpu_run.md).
* **30-Qubit Glued-Tree Hardware Protocol & Telemetry:** [`experiments/cepheus_glued_trees_30q/README.md`](experiments/cepheus_glued_trees_30q/README.md).
* **2D Plaquette Routing & Cepheus-108Q Seam Mitigation Findings:** [`docs/findings/plaquette_routing_and_cepheus_seam_mitigation.md`](docs/findings/plaquette_routing_and_cepheus_seam_mitigation.md).
* **IBM Nighthawk R2 Plaquette Mapping & Qrack Comparison:** [`docs/findings/ibm_nighthawk_r2_plaquette_mapping.md`](docs/findings/ibm_nighthawk_r2_plaquette_mapping.md).
* **Orthogonal K-Patch Proxy RCS Showdown (Qrack vs T-1000):** [`docs/findings/orthogonal_proxy_rcs_showdown.md`](docs/findings/orthogonal_proxy_rcs_showdown.md).
* **ZINC Continuous Topological Walk Molecular Screening:** [`docs/findings/zinc_topological_walk_screening.md`](docs/findings/zinc_topological_walk_screening.md).
* **Continuous Topological Walk Sorter with Weld Zeno Sentry:** [`docs/findings/continuous_topological_walk_sorter.md`](docs/findings/continuous_topological_walk_sorter.md).
* **RCS Chaos & Physical Noise Coherence Horizon:** [`docs/findings/rcs_physical_noise_chaos_transition.md`](docs/findings/rcs_physical_noise_chaos_transition.md).
* **Cascaded Multi-Tier Glued-Tree Valve Findings:** [`docs/findings/cascaded_tree_valve_deep_routing.md`](docs/findings/cascaded_tree_valve_deep_routing.md).
* **Multi-Exit Topological Glued-Tree Valve & Router Findings:** [`docs/findings/glued_tree_topological_valve.md`](docs/findings/glued_tree_topological_valve.md).
* **Childs et al. Glued-Trees Exponential Speedup Findings:** [`docs/findings/glued_trees_exponential_speedup.md`](docs/findings/glued_trees_exponential_speedup.md).
* **Ultrametric Tree Sorter & QMUX Architecture Findings:** [`docs/findings/ultrametric_tree_sorter_qmux.md`](docs/findings/ultrametric_tree_sorter_qmux.md).
* **Quantum Sorter vs. Classifier Theory & Findings:** [`docs/findings/quantum_sorter_vs_classifier_waterbirds.md`](docs/findings/quantum_sorter_vs_classifier_waterbirds.md).
* **Heterogeneous Co-Processor Pipeline Specification:** [`docs/findings/heterogeneous_co_processor_pipeline.md`](docs/findings/heterogeneous_co_processor_pipeline.md).
* **Zero-Allocation Dense VQC & Heptal Qudit MPS QNN Findings:** [`docs/findings/avx2_zero_allocation_and_heptal_qnn.md`](docs/findings/avx2_zero_allocation_and_heptal_qnn.md).
* **Dashboard & Circuit Composer Documentation:** [`docs/dashboard_and_circuit_composer.md`](docs/dashboard_and_circuit_composer.md).
* **Bloch-Zipper & MPS Derivation:** [`docs/bloch_zipper_mps_specification.md`](docs/bloch_zipper_mps_specification.md).
* **Empirical Research & Findings Registry (Dual-Housed with Deen OS Monographs M123–M156):** [`docs/findings/README.md`](docs/findings/README.md).

---

## Architecture Governance

Autonomous agents and contributors must follow the sovereign engineering, visual palette, and dual-licensing governance defined in [AGENTS.md](AGENTS.md).

---

## Contributing & Development Roadmap

We are actively welcoming quantum computing researchers, Rust performance engineers, and machine learning architects to collaborate:

- [ ] **AVX-512 & ARM NEON Kernels:** Expanding SIMD kernels beyond AVX2 for Apple Silicon (M-series) and modern server CPUs.
- [ ] **GPU Backend:** WGPU / CUDA kernels for scaling beyond 26 qubits.
- [ ] **Graph Neural Network (GNN) Hybridization:** Fusing T-1000 latent molecular embeddings with classical GNN message passing.

Feel free to open issues, submit pull requests, or share benchmark results on different architectures!

---

## License & Attribution

Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

Dual-licensed under either of:
* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option. Unless explicitly stated otherwise, any contribution intentionally submitted for inclusion in this work by you shall be dual licensed as above, without any additional terms or conditions.
