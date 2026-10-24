<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Terrence A. Jones SR. All rights reserved. -->

# Quantum-kernel classifier against classical kernels

```bash
cargo run --release -p t1000 --bin quantum_kernel_bench                     # MNIST demo pairs + controls
cargo run --release -p t1000 --bin quantum_kernel_bench -- --csv table.csv  # your own table, label in the last column
```

Recent comparisons find that quantum models do not yet beat classical baselines when the protocol is
careful (for example [Quantum vs. Classical Machine Learning: A Unified Empirical Comparison](https://arxiv.org/abs/2607.01197),
2026, and a nested-cross-validation study of quantum-kernel SVMs, [arXiv:2604.18837](https://www.emergentmind.com/papers/2604.18837)).
This benchmark applies the same discipline to our simulator.

## Protocol

* **Models.** Kernel ridge classifier, `alpha = (K + λI)⁻¹ y`, prediction `sign(K_test alpha)`. Kernels:
  the fidelity kernel `|⟨φ(x)|φ(y)⟩|²` of a layered circuit (one Ry and one Rz rotation per qubit from the
  features, a CZ ring, `L` repeats, 16 features on 8 qubits), an RBF kernel and a linear kernel.
* **Selection.** Nested cross-validation: the outer 5-fold stratified split (repeated) only scores; the
  hyperparameters are chosen by 3-fold inner cross-validation inside every outer training set. Quantum:
  angle scale `s ∈ {0.125, 0.25, 0.5, 1}` × layers `L ∈ {1, 2, 3}` × `λ ∈ {1e-3, 1e-2, 0.1, 1}`; RBF: 12 values
  of `γ` × the same four `λ`. Both kernels get 48 combinations. Features are scaled on the training rows only.
* **Data.** Pairs of MNIST digits from the bundled `data/mnist_demo` (240 images per pair, 4 x 4 average-pooled
  to 16 features).
* **Controls.** A *planted* task whose labels are the sign of a difference of quantum-kernel values to two
  anchor points (the quantum kernel should win) and the 3-vs-8 task with *shuffled* labels (everything should
  sit at chance).

## Results (10 repeats, accuracy on the outer test folds)

From `results/hosts/unknown-cpu/data/quantum_kernel_bench.json`:

| Task | Quantum | RBF | Linear | Quantum − RBF (se over repeats) |
| :--- | ---: | ---: | ---: | ---: |
| MNIST 1 vs 7 | 0.959 | 0.972 | 0.952 | −0.013 (0.003) |
| MNIST 3 vs 8 | 0.838 | 0.933 | 0.935 | −0.095 (0.005) |
| MNIST 4 vs 9 | 0.866 | 0.896 | 0.883 | −0.030 (0.006) |
| Planted quantum-kernel labels | 0.753 | 0.578 | 0.547 | +0.175 (0.008) |
| MNIST 3 vs 8, labels shuffled | 0.446 | 0.447 | 0.468 | −0.001 (0.006) |

* **The pipeline can see an advantage when one exists**: on the planted task the quantum kernel beats RBF by
  17.5 points, and with shuffled labels every model is at chance (below 0.5, as stratified cross-validation of
  random labels gives).
* **On real digits the quantum kernel is worse**: 1.3, 9.5 and 3.0 points behind RBF on the three pairs. Its
  worst case (3 vs 8, where the pixel pattern is subtle) is also the largest gap. The linear kernel is within 2 points of RBF on
  all three pairs, so these tasks do not need much non-linearity.
* **Limits.** 240 images per pair and 16 pooled features; one circuit family (layered Ry/Rz + CZ, up to 3
  repeats); kernel ridge, not an SVM; the grid is small. A different feature map could do better, and nothing
  here says quantum kernels cannot win on a task built for them: the planted task shows they can. The
  standard errors are over repeats, because the folds inside a repeat share data.
