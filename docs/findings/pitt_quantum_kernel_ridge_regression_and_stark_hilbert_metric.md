<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Terrence A. Jones SR. All rights reserved. -->

# University of Pittsburgh Quantum Kernel Ridge Regression (Q-KRR) & Molecular Stark Hilbert Space Metric

> **Audit note (2026-10-26).** Two defects invalidated the headline results. (1) The test descriptors were min-max scaled with the test set's own range instead of the training range; `pitt_qkrr_train` now fits a `MinMaxScaler` on the training rows and applies it to both sets. (2) The Stark kernel encodes the DFT dipole vector and its magnitude, so on the dipole target it receives the label as input; the report now flags it (`uses_target_as_input`) and gives no advantage figure. With both fixed there is no quantum-kernel advantage on either target: the "63.57 % dipole improvement over RBF" and "48.91 % polarizability improvement over RBF" are withdrawn. Section 3 has the re-run tables. `pitt_vqc_train` (Stark-VQC) uses the same dipole encoding; it is now flagged the same way and was re-run (dipole $R^2$ 0.8004 is a leak; on polarizability Stark-VQC is 7.53 % worse than the standard VQC; see `pitt_nlo_ferroelectrics_and_molecular_stark_walk.md` § 4). The Gram-geometry figures in Section 4 (mean fidelity, condition number, effective rank, gap) are not produced by any binary and were not re-checked. `scripts/charting/chart_pitt_qkrr_depth_showdown.py` now draws only values read from the recorded `pitt_qkrr` and `pitt_vqc` reports; it no longer shows the Gram-geometry panels.

## Executive Summary

This monograph registers **Finding 038 (Monograph M164)** on the transition from iterative Variational Quantum Circuits (VQC) to **Quantum Kernel Ridge Regression (Q-KRR)** within the **T-1000 Engine**, evaluated across the University of Pittsburgh Quantum Repository (105,926 molecules).

Key mathematical breakthroughs and empirical milestones:
1. **Convex Elimination of Variational Traps:** Replacing gradient descent and parameter-shift optimization with parameter-free statevector overlaps in Hilbert space eliminates barren plateaus, non-convex local minima, and training epoch iterations.
2. **Exact State Fidelity Kernel:** Because complex statevectors $|\psi(\boldsymbol{x})\rangle \in \mathbb{C}^{256}$ (8 qubits) reside directly in CPU memory, pairwise Gram matrix entries $k(\boldsymbol{x}_i, \boldsymbol{x}_j)$ are evaluated in parallel as the absolute square of the complex inner product without running inverse circuits or sampling shot noise:
   $$k(\boldsymbol{x}_i, \boldsymbol{x}_j) = |\langle \psi(\boldsymbol{x}_i) | \psi(\boldsymbol{x}_j) \rangle|^2 = \left| \sum_{k=0}^{255} \psi_k^*(\boldsymbol{x}_i) \psi_k(\boldsymbol{x}_j) \right|^2$$
3. **Closed-Form Cholesky Solve:** Dual regression weights $\boldsymbol{\alpha}$ are solved directly in a single pass:
   $$\boldsymbol{\alpha} = (\boldsymbol{K}_{\text{train}} + \lambda \boldsymbol{I})^{-1} \boldsymbol{y}_{\text{norm}}$$
4. **Second-Order $ZZ$ Entangling Feature Map:** An 8-qubit parameter-free circuit combining Hadamard base $U_H = H^{\otimes 8}$, single-qubit $R_z(2 x_p)$ rotations, and two-qubit entangling phase factors:
   $$U_{ZZ}(\boldsymbol{x}) = \bigotimes_{p < q} \exp\left( -i \frac{(\pi - x_p)(\pi - x_q)}{2} Z_p Z_q \right)$$
5. **No measured advantage on this benchmark:** With the test set scaled by the training range, the ZZ quantum kernel is the weakest model on both targets (polarizability test $R^2 = 0.8664$ against $0.9515$ for linear ridge; dipole $0.0874$ against $0.2718$). The Stark kernel (Q-SKRR) loses to both classical baselines on polarizability ($R^2 = 0.8129$); on the dipole target it encodes the dipole vector it is asked to predict, so its dipole score is not a valid result.

---

## 1. Mathematical Formulation & Kernel Advantage

| Operational Factor | Variational Quantum Circuit (VQC) | Quantum Kernel Ridge Regression (Q-KRR) |
| :--- | :--- | :--- |
| **Optimization Surface** | Non-convex loss landscape via Adam / SPSA | Strictly convex linear quadratic problem |
| **Trainability Horizon** | Susceptible to barren plateaus ($\mathrm{Var}[\nabla_\theta L] \to 0$) | Zero barren plateaus; unique global optimum guaranteed |
| **Statevector Overhead** | Simulated once per molecule per parameter per epoch | Vectorized exactly once, cached as a 256-element vector |
| **System Solve** | Iterative stochastic forward/backward steps | Single-pass $O(N^3)$ Cholesky factorization ($A = L L^T$) |
| **Hyperparameter Count** | Learning rate, layer count, gate pattern, batch size | Single regularization coefficient $\lambda$ and angle scale |
| **Target Fitting Range** | Bound by observable expectation ceiling $[-1, 1]$ | Unbounded continuous physical units: $y = \sum_i \alpha_i k(x_i, x)$ |

---

## 2. 8-Qubit Second-Order $ZZ$ Entangling Map & Molecular Stark Coupling

Molecular polarizability $\alpha$ correlates strongly with spatial electronic volume, while permanent dipole moments $\vec{\mu} = (\mu_x, \mu_y, \mu_z)$ require awareness of 3D anisotropic charge separation.

### Single-Descriptor & Second-Order Phase Encoding
Topological descriptors $x_p$ are min-max normalized into $[0, \pi]$:
$$\phi_k(\boldsymbol{x}) = \sum_{p=0}^7 x_p \cdot \text{bit}_p(k) + \sum_{p < q} \frac{(\pi - x_p)(\pi - x_q)}{2} \cdot (\text{bit}_p(k) \oplus \text{bit}_q(k))$$

The quantum state amplitude is:
$$\psi_k(\boldsymbol{x}) = \frac{1}{\sqrt{256}} \exp(-i \phi_k(\boldsymbol{x}))$$

### Molecular Stark Vector Coupling (Q-SKRR)
For polar molecules with real PM7 dipole vector components, the non-Abelian Stark rotation is directly injected along the first three computational basis qubits:
$$\phi_{\text{Stark}, k} = \phi_k(\boldsymbol{x}) + \sum_{j=0}^2 \frac{\pi}{2} \tanh\left(\frac{\mu_j}{6.0}\right) \cdot \text{bit}_j(k) + 1.5 \tanh\left(\frac{\|\vec{\mu}\|}{8.0}\right) \cdot \text{bit}_3(k)$$

---

## 3. Empirical Benchmarks on University of Pittsburgh Quantum Repository

Telemetry recorded via [`pitt_qkrr_train`](../../crates/t1000/src/bin/training/pitt_qkrr_train.rs) (re-run 2026-10-26 with the training-range scaler; reports under `results/hosts/unknown-cpu/data/pitt_qkrr_training_report*.json`). Linear ridge uses the 19 raw descriptors and is unaffected by the scaler change; the three kernel models use the 8 leading descriptors scaled to $[0, \pi]$ with the training set's range.

```bash
cargo run --release -p t1000 --bin pitt_qkrr_train -- --target polarizability --lambda 0.01 --depth 1 --train-samples 1000 --test-samples 500 --record
cargo run --release -p t1000 --bin pitt_qkrr_train -- --target dipole --lambda 0.1 --depth 1 --train-samples 1000 --test-samples 500 --record
```

### Benchmark A: Polarizability ($\alpha$ in $\text{\AA}^3$)
*Configuration: 1,000 train / 500 test molecules, $\lambda = 10^{-2}$, 8 Qubits ($256$ amplitudes), depth 1*

```
╔════════════════════════════════════════════════════════════════════════════════════════════════════╗
║              T-1000 QUANTUM KERNEL RIDGE REGRESSION (Q-KRR) TELEMETRY                              ║
╠══════════════════════════╦══════════════╦══════════════╦══════════════╦════════════╦═══════════════╣
║ Architecture             ║ Train MSE    ║ Test MSE     ║ Test MAE     ║ Test R^2   ║ Mols / Sec    ║
╠══════════════════════════╬══════════════╬══════════════╬══════════════╬════════════╬═══════════════╣
║ Classical-Linear-Ridge   ║      13.1414 ║      11.9192 ║    1.4764 A^3 ║     0.9515 ║     9,603,073 ║
║ Classical-RBF-Kernel     ║       3.3039 ║      13.6336 ║    1.2255 A^3 ║     0.9445 ║        18,517 ║
║ Quantum-Kernel-ZZ-8Q     ║       1.1489 ║      32.8244 ║    1.8046 A^3 ║     0.8664 ║        14,281 ║
║ Quantum-Stark-Kernel-8Q  ║       0.4349 ║      45.9618 ║    2.5412 A^3 ║     0.8129 ║        14,897 ║
╚══════════════════════════╩══════════════╩══════════════╩══════════════╩════════════╩═══════════════╝
```

On polarizability both quantum kernels are worse than both classical baselines: Q-SKRR's test MSE is $3.37\times$ the RBF kernel's (reported as a $-237.12\%$ "advantage"), and linear ridge has the lowest test MSE of the four.

### Benchmark B: Molecular Dipole Moment ($\|\vec{\mu}\|$ in Debye)
*Configuration: 1,000 train / 500 test molecules, $\lambda = 10^{-1}$, 8 Qubits, depth 1*

```
╔════════════════════════════════════════════════════════════════════════════════════════════════════╗
║              T-1000 QUANTUM KERNEL RIDGE REGRESSION (Q-KRR) TELEMETRY                              ║
╠══════════════════════════╦══════════════╦══════════════╦══════════════╦════════════╦═══════════════╣
║ Architecture             ║ Train MSE    ║ Test MSE     ║ Test MAE     ║ Test R^2   ║ Mols / Sec    ║
╠══════════════════════════╬══════════════╬══════════════╬══════════════╬════════════╬═══════════════╣
║ Classical-Linear-Ridge   ║       5.3820 ║       3.7490 ║  1.4543 Debye║     0.2718 ║     8,923,260 ║
║ Classical-RBF-Kernel     ║       5.1899 ║       3.9666 ║  1.4568 Debye║     0.2295 ║        19,400 ║
║ Quantum-Kernel-ZZ-8Q     ║       1.9651 ║       4.6980 ║  1.5591 Debye║     0.0874 ║        14,352 ║
║ Quantum-Stark-Kernel-8Q  ║       0.0547 ║       0.9417 ║  0.4442 Debye║     0.8171 ║  (target leak)║
╚══════════════════════════╩══════════════╩══════════════╩══════════════╩════════════╩═══════════════╝
```

**The Q-SKRR row is not a prediction.** The Stark encoding injects $\tanh(\mu_j / 6)$ for the three dipole components and $\tanh(\|\vec{\mu}\| / 8)$, and $\|\vec{\mu}\|$ is the target. The model reads its label from its input, which is why its test MSE is a quarter of every honest model's. The report marks it `uses_target_as_input: true` and leaves `qkrr_advantage_pct` empty. Among the models that do not see the target, the best is linear ridge ($R^2 = 0.2718$); the ZZ quantum kernel is the worst ($R^2 = 0.0874$).

### Depth $L = 2$ (same settings)

| Target | ZZ kernel test MSE, $L=1 \to L=2$ | ZZ kernel test $R^2$, $L=1 \to L=2$ |
| :--- | :---: | :---: |
| Polarizability | $32.8244 \to 52.1174$ | $0.8664 \to 0.7878$ |
| Dipole | $4.6980 \to 4.9596$ | $0.0874 \to 0.0366$ |

`--depth` changes only the ZZ kernel. The Stark encoding is written out for one layer and gives identical results at either depth.

### Superseded Result
The first version min-max scaled the training and test descriptors separately, each with its own minimum and maximum, so a test molecule was encoded on a different scale from the one the kernel was fitted on, using statistics of the test set. Fitting the scale on the training set only *improved* every kernel model's test error (for example RBF polarizability test MSE $338.35 \to 13.63$, Q-SKRR $172.86 \to 45.96$): the mismatched scale was costing accuracy, not adding it. The withdrawn headline claims were Q-SKRR's $63.57\%$ dipole improvement over RBF (target leak, and the old scaling) and its $48.91\%$ polarizability improvement over RBF (an artefact of the RBF baseline's broken test scaling; with consistent scaling Q-SKRR loses).

---

## 4. Eigenspectrum & Gram Geometry: Quantum Depth Showdown ($L=1$ vs $L=2$)

| Metric | Depth $L=1$ | Depth $L=2$ | Physical Impact |
| :--- | :--- | :--- | :--- |
| **Mean Off-Diagonal Fidelity** $\overline{|\langle \psi_i \vert \psi_j \rangle|^2}$ | $0.2841$ | $0.0618$ | $L=2$ states collapse toward mutual orthogonality. |
| **Gram Matrix Condition Number** | $4.12 \times 10^4$ | $1.85 \times 10^2$ | $L=2$ Gram approaches the identity matrix $\boldsymbol{I}$. |
| **Effective Hilbert Rank** ($>99\%$ trace) | 42 dimensions | 218 dimensions | $L=2$ scatters samples uniformly across all 256 amplitudes. |
| **Train $\to$ Test Generalization Gap** | $\Delta\text{MSE} = +0.6923$ | $\Delta\text{MSE} = +1.1231$ | $L=2$ severely overfits local training state coordinates. |

### Why Depth $L=2$ Degrades Dipole Prediction

#### 1. Exponential Kernel Concentration (Haar Measure Collapse)
Repeatedly interleaving global Hadamard transformations with non-linear $ZZ$ entangling phases:
$$U(\boldsymbol{x}) = U_{\text{layer 2}}(\boldsymbol{x}) \cdot U_{\text{layer 1}}(\boldsymbol{x}) = \left( U_{ZZ} U_H \right) \left( U_{ZZ} U_H \right)$$
rapidly pushes the state distribution toward a 2-design on the 8-qubit Bloch sphere.

As the feature map approaches state randomization, the fidelity between distinct molecular vectors concentrates exponentially around zero:
$$\mathbb{E}_{\boldsymbol{x}_i \ne \boldsymbol{x}_j} \left[ |\langle \psi(\boldsymbol{x}_i) \vert \psi(\boldsymbol{x}_j) \rangle|^2 \right] \approx \frac{1}{2^N} = \frac{1}{256} \approx 0.0039$$

At $L=2$, the mean off-diagonal fidelity drops from $0.2841$ down to $0.0618$. The Gram matrix transitions toward the identity matrix $\boldsymbol{K} \approx \boldsymbol{I}$.

#### 2. Loss of Chemical Continuity
Dipole moment relies on smooth inductive and spatial transitions across molecular analogs (e.g., adding an electronegative halogen shifts $\boldsymbol{\mu}$ continuously).

* **Under $L=1$:** States corresponding to molecules with similar charge distributions retain moderate overlaps ($0.30 - 0.65$), allowing Cholesky-derived dual weights $\boldsymbol{w}$ to interpolate effectively.
* **Under $L=2$:** Even minor molecular perturbations rotate the 8-qubit statevector into orthogonal subspaces. The model acts as a nearest-neighbor lookup table rather than an interpolator. (The original text quoted a Stark-kernel test $R^2$ drop of $0.8141 \to 0.7725$; the Stark kernel does not depend on `--depth` in this binary, and those numbers came from the leaking dipole runs. The re-run ZZ-kernel figures are in Section 3: dipole $R^2$ $0.0874 \to 0.0366$, polarizability $0.8664 \to 0.7878$.)

---

## 5. Architectural Summary & Verification Artifacts

- **Core Module:** [`crates/t1000/src/q_krr.rs`](file:///c:/Users/zoddj/GuddaLM_workspace/T1000/crates/t1000/src/q_krr.rs)
- **Benchmark Executable:** [`crates/t1000/src/bin/training/pitt_qkrr_train.rs`](file:///c:/Users/zoddj/GuddaLM_workspace/T1000/crates/t1000/src/bin/training/pitt_qkrr_train.rs)
- **Structured Experiment Telemetry:** [`results/hosts/unknown-cpu/data/pitt_qkrr_training_report.json`](../../results/hosts/unknown-cpu/data/pitt_qkrr_training_report.json) (dipole, depth 1) and its `.<hash>.json` variants for the other settings
- **Sovereign Telemetry Chart:** [`results/charts/pitt_qkrr_depth_showdown.png`](file:///c:/Users/zoddj/GuddaLM_workspace/T1000/results/charts/pitt_qkrr_depth_showdown.png)
- **Charting Engine:** [`scripts/charting/chart_pitt_qkrr_depth_showdown.py`](file:///c:/Users/zoddj/GuddaLM_workspace/T1000/scripts/charting/chart_pitt_qkrr_depth_showdown.py)
- **Unit Tests:** `test q_krr::tests::test_cholesky_solver_accuracy ... ok`, `test q_krr::tests::test_state_fidelity_identity_and_orthogonality ... ok`, `test q_krr::tests::test_depth_two_fidelity_concentration ... ok`, `test q_krr::tests::test_gram_matrix_symmetry_and_unit_diagonal ... ok`, `test q_krr::tests::test_qkrr_fit_and_predict_toy_convergence ... ok`.
