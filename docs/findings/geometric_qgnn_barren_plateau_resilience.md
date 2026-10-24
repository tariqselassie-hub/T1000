# Lie-Equivariant Quantum Graph Neural Network (QGNN): equivariance and a gradient-variance sweep

**Author:** Terrence A. Jones SR
**Date:** 2026-10-19; corrected 2026-10-23
**Subsystems:** `t1000::qgnn`, `t1000::smiles`, `t1000::simd`, `qgnn_molecular_bench`
**License:** MIT OR Apache-2.0

> **Audit note (2026-10-23).** The first version of this document was titled "Barren Plateau Resilience" and
> reported a "non-vanishing gradient variance of 4.2931e-2 at 14 qubits" as an advantage over the
> hardware-efficient ansatz. The sweep does not show that: the QGNN variance is identical at every size
> (the observable is local on shared-weight layers, so it cannot depend on the number of qubits) and the
> baseline's variance is larger at every size tested. The title, the "theorem" and the chart were removed or
> reworded below.

---

## 1. What the circuit is

Layers of graph-structured rotations: a node rotation per atom and a Heisenberg exchange
`exp(-i phi/2 (XX + YY + Delta ZZ))` on every bond of the OpenSMILES graph, with weights shared across
nodes and bonds. Sharing makes the prediction exactly invariant under relabelling of the atoms.

## 2. What was measured (re-run 2026-10-23)

* **Permutation equivariance.** Benzene (6 nodes) under a non-trivial automorphism: prediction
  0.79473525 before and after, drift 0 (machine precision). This follows from the weight sharing and is
  verified numerically.
* **Latency.** 2.6 to 2.7 us per molecule on the 6-node benzene graph (about 370,000 to 380,000 per second).
  Larger molecules cost more; the figure is for a small graph.
* **Gradient variance, one parameter, 40 random initialisations (`qgnn_molecular_bench`):**

| Qubits | Hardware-efficient ansatz | Equivariant QGNN |
| ---: | ---: | ---: |
| 4 | 1.42e-1 | 4.29e-2 |
| 6 | 1.18e-1 | 4.29e-2 |
| 8 | 1.60e-1 | 4.29e-2 |
| 10 | 1.20e-1 | 4.29e-2 |
| 12 | 8.03e-2 | 4.29e-2 |
| 14 | 1.02e-1 | 4.29e-2 |

  The hardware-efficient baseline does not decay either in this range (barren plateaus show up at larger
  sizes and greater depth), and the QGNN number is smaller than the baseline's at every size. The constant
  value is a property of the construction: the measured qubit only sees its graph neighbourhood. This sweep
  therefore neither shows resilience nor refutes it.
* **Training.** Regressing logP on eight small molecules, the mean squared error fell from 1.157 to 1.092
  over 20 epochs (RMSE 1.076 to 1.045). No baseline is printed (a constant predictor at the mean has an RMSE
  equal to the targets' standard deviation), so this is not evidence that the model learns anything useful.

## 3. What is argued, not shown

For a circuit whose dynamical Lie algebra has polynomial dimension, results in the literature (for example
Ragone et al., 2023) bound the variance of cost gradients from below by an inverse polynomial in the
algebra's dimension, under conditions on the initial state and observable. This is a reason to expect
trainability at large sizes. Nothing in this repository checks those conditions or tests sizes where the
unstructured ansatz has plateaued.
