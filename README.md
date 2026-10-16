# T-1000: Quantum Neural Network & Mimetic Circuit Simulator

[![Rust](https://img.shields.io/badge/rust-2021_edition-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)
[![Build Status](https://img.shields.io/badge/status-active%20development-brightgreen.svg)](#)

> *"Like the liquid metal assassin, T-1000 adapts and mimics any quantum device or classical architecture. Feed it parameters; it handles the rest."*

`T-1000` is an ultra-high-performance Variational Quantum Circuit (VQC) and Quantum Neural Network engine engineered in pure Rust. Built for extreme SIMD throughput, arbitrary circuit DAG scheduling, and chemical latent space exploration, it runs completely standalone with zero legacy bloat.

---

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

---

## Key Capabilities

### 1. Shape-Shifting Circuit Compilation & Scheduling (`t1000::compiler`)
* **Per-qubit dependency DAG (`CircuitDag`):** Tracks gate commutation and dependencies along wire order.
* **Gate Fusion:** Greedy grouping of non-overlapping single-qubit rotations and multi-qubit diagonal interactions into unified stages.
* **Fused Diagonal Kernel:** Executes entire layers of $R_z$ and $CZ$ gates in a single streaming memory sweep over the statevector.

### 2. AVX2 + FMA SIMD Statevector Engine (`t1000::simd`, `t1000::tiled_butterfly`)
* **Structure-of-Arrays (SOA):** Separated contiguous arrays of double-precision real and imaginary amplitudes.
* **L2-Resident Tiled Butterfly Kernels:** Cache-conscious stride access for high qubit indices ($q \ge 16$).
* **Beats Industry Baselines:** Up to **1.9x faster than Qiskit Aer** at 8 threads on deep Random Circuit Sampling benchmarks.

### 3. SMILES Molecular Exploration & Whole-Library 250k Screening (`zinc_dual_search`)
* **Chemical Featurizer:** Native parser for OpenSMILES strings generating 19 molecular descriptors (atom counts, aromaticity, ring closures, molecular weight, heteroatom ratio).
* **Quantum Feature Map:** Direct parameterized injection into Hilbert space ($|\psi(\text{SMILES})\rangle$) via parameterized $R_y(x_i)$ rotations.
* **Whole-Library ZINC 250k Engine:** Screens all 249,455 drug candidates in **1.17 seconds** (**213,351 molecules/sec**) across 20 threads.
* **Search 3 (Quantum Anomalies & Outliers):** Mines extreme zwitterions, highly charged amidinium/guanidinium motifs, and topological outliers based on Hilbert space dispersion.
* **Search 1 (Frontier Pareto Drug Candidates):** Discovers top golden leads with $\text{QED} > 0.88$, $\text{SAS} < 1.45$, and $\text{logP} \in [2.4, 3.1]$.
* **Period 1 & 2 Molecular Quantum Shells (`t1000::chemistry`):** Second-quantized Jordan-Wigner Hamiltonian simulations for 9 fundamental molecules from $\text{H}_2$ ($4\text{Q}$) up to all-electron Methane $\text{CH}_4$ ($18\text{Q}$), evaluating ground-state energies, 18-body Pauli-Z orbital occupations, and single-orbital Von Neumann entanglement entropies.
* Detailed Documentation: [`docs/zinc_quantum_screening.md`](docs/zinc_quantum_screening.md).

### 4. Statistical Quantum Verification (`t1000::rcs`)
* **Frontier RCS Benchmark (`frontier_rcs`):** Deep Random Circuit Sampling (default: 18 qubits, 36 layers, 1,945 gates, $262,144$ states) benchmarking the optimized 256-bit AVX2 engine against a textbook scalar reference simulator and Qiskit Aer.
* **Dual-Engine Precision:** Machine precision numerical agreement (fidelity $1.000000000000$, max amplitude error $< 3.1 \times 10^{-17}$).
* **Cross-Entropy Benchmarking (XEB):** Validates linear XEB convergence ($F_{\text{XEB}} \approx 1.0045$) against ideal Haar-random Porter-Thomas distributions.
* **Kolmogorov-Smirnov & Entropy Tests:** Automated confirmation of Porter-Thomas chaos ($D_{\text{KS}} = 0.0021 < D_{\text{critical}} = 0.0027$, $\Delta S = 0.4243\text{ nats} \to 1 - \gamma$).
* **Adjoint Differentiation:** Evaluates analytical gradients across 1,296 parameters in $0.250\text{ s}$ ($2,592\times$ circuit evaluation avoidance over parameter shift).
* **Coherence Frontier Push (`coherence_frontier_push`):** Dual-mark physical experiment tracking quantum $\ell_1$-norm coherence capacity ($C_{\text{norm}}$) up to 26 qubits dense ($2^{26}$ amplitudes, 1.02 GB RAM), 64 qubits via 2D tensor network belief propagation ($2^{64}$ states), and sovereign Heptal wave qudits ($7^9$ states, $d=7$).

### 5. Quantum Vision & BNN Benchmark (`mnist_qnn_train`)
* **Amplitude Encoding:** Exact injection of 784-dimensional image spaces directly into a 10-qubit Hilbert space ($2^{10} = 1024$ amplitudes).
* **Bipartite Entanglement Observables:** 55 quantum expectation operators (10 single-qubit $\langle Z_i \rangle$ + 45 two-body $\langle Z_i Z_j \rangle$ correlators) with analytical reverse-mode `adjoint_gradient`.
* **Surpasses Classical BNN & HDC Baselines:**

| Architecture Model | Test Accuracy % | Space Representation | State Coherence $C_{\ell_1}$ | Throughput |
| :--- | :---: | :---: | :---: | :---: |
| **GuddaNetworks BNN Baseline (BSC)** | 78.59% | 4096-bit Binary Sparse Coding | Classical | Classical BSC |
| **GuddaNetworks BNN Baseline (MAP)** | 79.40% | 4096-dim Multiply-Add-Permute | Classical | Classical MAP |
| **GuddaNetworks HDC-10k Dimension** | 81.25% | 10,000-bit BSC Space | Classical | Classical BSC |
| **T-1000 10-Qubit VQC (AVX2)** | **84.50%** | **1024-dim ($2^{10}$) Hilbert Space** | **350.94** | **34.38 $\mu$s / sample** |

### 6. Physical Silicon Quantum Coprocessor (`fpga/1000_qpu`)
* **Target Hardware:** AMD/Xilinx Artix-7 XC7A35T-1CPG236C FPGA (Digilent Basys 3).
* **On-Chip 14-Qubit Memory:** 128 KB True Dual-Port Block RAM ping-pong matrix ($16,384$ complex amplitudes in Q1.15 fixed point).
* **8-Lane DSP48E1 Butterfly Core:** Executes full 14-qubit state evolution in **$10.24\,\mu\text{s}$** ($169.1\times$ raw silicon compute speedup over host CPU AVX2 passes).
* **Opcode 0x20 Autonomous Execution:** Executes 100 quantum layers entirely on-chip in **$2.006\text{ ms}$** (**49,845 layers/sec**, $50\times$ speedup over host roundtrips).
* **Opcode 0x30 Chaos Streamer:** Streams 256-bit cryptographic entropy seeds at **4,179.5 seeds/sec** (133.7 KB/s), passing NIST SP 800-22 randomness tests.
* **High-Speed Interconnect:** 3.0 MBaud UART interface locked to physical host port `COM6`.
* **Physical Telemetry:** Dedicated 4-digit 7-segment display showing `1000`, 16-LED state readouts, and zero-drop burst throughput.
* Detailed Documentation: [`docs/fpga_architecture.md`](docs/fpga_architecture.md).

### 7. Fault-Tolerant Quantum Error Correction & Planar Surface Code (`t1000::qec`)
* **3x3 Planar Surface Code $[[9, 1, 3]]$:** Rotated lattice topology embedding 9 physical data qubits with 8 syndrome measurement plaquettes (4 weight-2/4 $X$-checks and 4 weight-2/4 $Z$-checks).
* **Closed-Form Transversal Code Space:** Exact stabilizer subspace generation via generator algebra (`[27, 432, 6, 192]`) guaranteeing 16 closed-form logical codewords with zero runtime allocation.
* **Plaquette Syndrome Decoding:** Minimum-weight syndrome decoding detecting and correcting arbitrary single-qubit Pauli $X$ and $Z$ errors with $100\%$ fidelity recovery ($F_{\text{recovered}} = 1.00000$).
* **Standard Stabilizer Suite:** Native support for 3-qubit bit-flip $[[3,1,1]]$, 3-qubit phase-flip $[[3,1,1]]$, 5-qubit perfect code $[[5,1,3]]$, 7-qubit Steane code $[[7,1,3]]$, and 9-qubit Shor code $[[9,1,3]]$.

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
* **Ultra-Fast Linear Scaling:** Executes 200 entangled qubits in **$105.8\,\mu\text{s}$** (~$529\text{ ns/qubit}$ constant latency) with $100\%$ anti-correlation fidelity.
* Detailed Documentation: [`docs/bloch_zipper_mps_specification.md`](docs/bloch_zipper_mps_specification.md).

### 11. Heptal Qudit $SU(7)$ Modular Complementarity Engine (`t1000::heptal_wave`)
* **Radix-7 Hilbert State Evolution:** Native $d = 7$ qudit simulation simulating high-dimensional quantum systems ($7^N$ state spaces).
* **Modular Inversion Unitaries:** Sliding 2-qudit window (49 amplitudes = 784 bytes, strictly L1 resident) applying cyclic shift and reflection operations: $|j, k\rangle \mapsto |j, (k - j) \pmod 7\rangle$.
* **Exact Modular Complementarity:** Enforces deterministic pairwise complementary locking across the qudit chain: $(q_i + q_{i+1}) \equiv 0 \pmod 7$.
* **High-Dimensional Throughput:** Evolves 200 qudits ($7^{200} \approx 4.6 \times 10^{169}$ effective Hilbert space) in **$7.58\text{ ms}$** (~$37.9\,\mu\text{s/qudit}$) with $100\%$ modular complementarity verification.

### 12. Quantum State-Space Autoencoder & Sequence Classifier (`t1000::qssm`)
* **Quantum Mamba / Q-SSM Architecture:** Combines streaming 1D MPS sliding window recurrence ($W=3$, 128 bytes in L1 cache) with a dual-head latent bottleneck.
* **Dual-Head Topology:**
  * **Latent Classifier Head:** Extracts expectation observables $\langle Z_k \rangle$ from the compressed quantum state to output sequence class probabilities.
  * **Unzipping Decoder Head:** Reverses continuous Bloch steering to autoregressively reconstruct the input feature sequence $\hat{X}$ from the latent bottleneck.
* **Extreme Streaming Throughput:** Evaluates sequence lengths $T \in [10, 500]$ at **$66.6\text{–}95.8\text{ ns per token}$** (**$> 13\text{–}15\text{ million tokens/second}$**), with 100% test classification accuracy on complex physical signal dynamics.

### 13. Automated Clifford+T Single-Qubit Synthesis (`t1000::clifford_t`)
* **Cyclotomic Ring $\mathbb{Z}[\omega]$ Arithmetic:** Exact integer coordinate representation modulo $\omega^4 + 1 = 0$ ($\omega = e^{i\pi/4} = \frac{1+i}{\sqrt{2}}$) with algebraic norm, conjugation, and exact division by $\lambda = 1+\omega$.
* **Hierarchical Grid Solver:** Synthesizes arbitrary continuous single-qubit rotations ($R_z(\theta), R_x(\theta), R_y(\theta)$) into minimal-length discrete gate sequences over $\{H, S, S^\dagger, T, T^\dagger, X, Y, Z\}$.
* **Matsumoto-Amano Normal Form & Peephole Optimization:** Deterministic parity state-machine reduction with multi-pass peephole cancellations ($H^2 \to I$, $T T^\dagger \to I$, $T^2 \to S$, $S^2 \to Z$, $Z^2 \to I$), guaranteeing $\|U_{\text{exact}} - U_{\text{discrete}}\|_\infty \le \varepsilon$.
* **Ansatz Lowering Pipeline:** `lower_circuit_ops` directly transpiles variational continuous ansatz layers into discrete, fault-tolerant gate token streams for stabilizer or magic-state execution.

---

## Quick Start for Developers

Clone and build the entire workspace with standard Cargo tools:

```bash
git clone https://github.com/tariqselassie-hub/T1000.git
cd T1000

# Run all unit, integration, and oracle verification tests
cargo test --workspace

# Run Quantum State-Space Autoencoder (Q-SSM) Training & Streaming Benchmark
cargo run --release -p t1000 --bin qssm_autoencoder_bench

# Run Zero-Data Real-Time Quantum Quench Autoencoder & Dynamical Phase Classifier
cargo run --release -p t1000 --bin qssm_quench_bench

# Run 1D MPS Bloch-Zipper and Heptal SU(7) Complementarity Benchmarks (14 to 200 qubits/qudits)
cargo run --release -p t1000 --bin bloch_zipper_bench

# Run 10-qubit MNIST Quantum Neural Network & Coherence Benchmark
cargo run --release -p t1000 --bin mnist_qnn_train

# Execute 14-qubit, 36-depth Random Circuit Sampling frontier benchmark
cargo run --release -p t1000 --bin frontier_rcs -- --qubits 14 --depth 36

# Execute ZINC 250k whole-library dual search (Search 3 Anomalies + Search 1 Pareto Leads)
cargo run --release -p t1000 --bin zinc_dual_search

# Run sovereign Quantum Coherence Frontier Push (Dense 26Q + TN 64Q + Heptal 7^9 Qudits)
cargo run --release -p t1000 --bin coherence_frontier_push

# Generate all scientific research charts and telemetry figures
python scripts/charting/chart_all.py

# Verify physical Artix-7 FPGA autonomous layers and chaos streaming over COM6
python scripts/fpga/test_chaos_streaming.py

# Launch interactive 3D molecular dashboard and Lego-block circuit composer
python web/serve.py
```

* **Dashboard & Circuit Composer Documentation:** [`docs/dashboard_and_circuit_composer.md`](docs/dashboard_and_circuit_composer.md).
* **Bloch-Zipper & MPS Derivation:** [`docs/bloch_zipper_mps_specification.md`](docs/bloch_zipper_mps_specification.md).
* **Empirical Research & Findings Registry:** [`docs/findings/README.md`](docs/findings/README.md).


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
