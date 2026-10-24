# Full Quantum Neural Network (QNN) Suite Benchmark

**Author:** Terrence A. Jones SR  
**Date:** 2026-10-19; re-measured and corrected 2026-10-23  
**Subsystems:** `t1000::qgnn`, `t1000::topological_sorter`, `t1000::ultrametric_tree`, `t1000::qssm`, `t1000::heptal_qnn`, `t1000::simd`, `t1000::adjoint`  
**License:** MIT OR Apache-2.0  
**Benchmark Binary:** `crates/t1000/src/bin/benchmarks/full_qnn_suite_bench.rs`  
**Telemetry File:** `results/data/full_qnn_suite_telemetry.json`

> **Audit note (2026-10-23).** The first version of this document quoted a "153.6x speedup" for the
> adjoint pillar and a 14-qubit barren-plateau variance. Both were wrong: the benchmark computed the
> speedup from the formula `2 * parameters * 0.8` and printed the variance as a literal, and the adjoint
> pillar crashed at run time. The benchmark now measures the speedup, the numbers below are from a re-run,
> and the showdown chart (which also contained hard-coded curves) was removed.

---

## 1. Re-measured results

Core Ultra 7 265, 20 threads, one run of `cargo run --release -p t1000 --bin full_qnn_suite_bench`
(latencies vary by about 10 to 30 % between runs).

| ID | Pillar | Latency | Throughput | Note |
| :--- | :--- | ---: | ---: | :--- |
| P1 | Geometric QGNN (Lie-equivariant graph convolution) | 2.6 us per molecule | about 380,000 molecules/s | 6-node benzene graph; permutation drift 0 |
| P2 | Continuous walk sorter | 37 us per sample | about 27,000 samples/s | routing contrast 400:1, ballistic peak 45.7 % |
| P3 | Ultrametric tree sorter | 0.07 us per sample | about 13 million samples/s | no strong-triangle-inequality violations |
| P4 | Q-SSM (1D MPS, window 3) | 1.8 us per token | about 550,000 tokens/s | 128-byte window; reconstruction MSE 0.19 |
| P5 | Radix-7 heptal MPS | 420 to 470 us per molecule | about 2,100 to 2,400 molecules/s | 784-byte active window |
| P6 | Dense molecular VQC, 14 qubits | 6 to 7 us per pass | about 140,000 to 160,000 passes/s | 262 KB state; norm drift 0 |
| P7 | Adjoint differentiation, 96 parameters, 12 qubits | 0.4 ms | n/a | 50x faster than parameter shift (measured); agrees with it to 3e-16 |

## 2. What the table does and does not show

* **P1.** The permutation drift of exactly 0 is the point of the equivariant construction. The earlier
  claim of "barren-plateau resilience" is not supported: the measured gradient variance
  (`qgnn_molecular_bench`) is 4.29e-2 at every size from 4 to 14 qubits because the observable is local on
  shared-weight layers, and the hardware-efficient baseline's variance (0.08 to 0.16) is larger at every
  size tested and does not decay in this range either. Barren plateaus appear at larger sizes than this.
* **P4.** Per-token cost depends on what is timed: 64 to 84 ns in `qssm_autoencoder_bench` (inference only)
  and 1.8 us here (with classification).
* **P5.** The heptal MPS keeps a 49-amplitude window. The full space of `7^19` amplitudes is never held, so
  the earlier "2.3e14x compression" compares with an object the method does not represent.
* **P7.** The comparison with parameter shift uses the same circuit and observable; the gradients agree to
  rounding error, so the speedup is a like-for-like timing.
* Throughputs are for small, fixed workloads on one host; they are not scaling results.
