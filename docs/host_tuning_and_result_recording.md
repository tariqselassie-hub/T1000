# Host Tuning, Fused Execution and Result Recording

This document covers three connected pieces of the engine: how it identifies the machine it
runs on, the cache-blocked execution paths that make large registers fast on a
memory-bandwidth-limited CPU, and the rules that keep benchmark and training runs from
overwriting saved results.

All timings below were measured on the Intel Core i3-1115G4 laptop (2 cores, 4 threads, 6 MB L3,
19 GB RAM, about 19 GB/s DRAM bandwidth, UHD iGPU) unless a row says otherwise. The Core Ultra 7
265 reference machine has not been re-measured with these changes.

---

## 1. Host detection (`t1000::hostinfo`)

`HostProfile::detect()` reads the machine at run time. Nothing about the CPU is hardcoded in
telemetry any more.

| Field | Source |
| :--- | :--- |
| CPU model | `/proc/cpuinfo` (`T1000_HOST_CPU` overrides it, for testing) |
| Logical threads | `std::thread::available_parallelism` |
| RAM | `/proc/meminfo` |
| SIMD tier | CPUID: AVX-512, AVX2+FMA or scalar |

Two machines are first-class (`KnownHost`): the Core Ultra 7 265 (the reference host) and the
i3-1115G4. Any other machine is reported as `Other` with its real values, and gets a file-name
slug derived from its CPU model.

When the CPU supports AVX-512F the hot kernels use 512-bit vectors (section 2.1); the label says
which width is active, and follows `T1000_SIMD`.

### Per-host tuning

| Setting | i3-1115G4 | Core Ultra 7 265 / other |
| :--- | :--- | :--- |
| Largest register run multithreaded | 18 qubits | unbounded (not measured) |
| Automatic iGPU routing in `Fast` precision | disabled | threshold 16 qubits (not measured) |

Why the i3 limits threading: one thread already reaches the DRAM ceiling (about 19 GB/s) on a
low-qubit gate, and from 20 qubits the 4-thread path measured 1.7 to 2 times slower than serial.

Environment overrides:

| Variable | Effect |
| :--- | :--- |
| `GUDDALM_QNN_THREADS` | Worker thread count (default: all logical CPUs) |
| `GUDDALM_QNN_PAR_MAX_QUBITS` | Largest register split across threads |
| `T1000_SIMD` | `avx2` forces the 256-bit kernels on an AVX-512 machine (for comparison); default uses AVX-512F when present |
| `T1000_PRECISION` | `exact` (default) or `fast` |
| `T1000_BACKEND` | `auto`, `cpu` or `gpu` (`gpu` implies fast) |
| `T1000_GPU_MIN_QUBITS` | Crossover for automatic GPU routing |

---

## 2. Fused execution

Above the last-level cache every gate pass is limited by DRAM bandwidth, so the number of
passes over the statevector is the cost. The following paths cut the pass count. Each is checked
against the earlier per-gate kernels in unit tests, and the oracle tests (parameter-shift and
finite-difference gradients) still pass.

| Module | What it does |
| :--- | :--- |
| `layer_fusion` | Applies a layer of single-qubit gates (Ry or any 2x2 unitary) in cache-sized blocks: qubits 0 to 11 together on 64 KB blocks, the rest four at a time over 4 KB chunks. A 26-qubit layer makes about 5 passes instead of 26. |
| `compiler::layered` | Assigns compiled-circuit ops to layers using commutation (Rz and CZ commute), then runs each layer's gates as one `layer_fusion` pass and its Rz/CZ gates as one diagonal pass. `CompiledCircuit::execute_unfused` keeps the per-op path as a reference. |
| `adjoint_fusion` | Backward sweep of the adjoint gradient: all Rz gradients in one pass, Ry gradients fused with the Ry undo. Sums are reduced in a fixed task order, so results do not depend on the thread count. |
| `compiler::permute` | CNOT, AntiCNOT, SWAP and controlled-phase move contiguous runs instead of branching per index. Gain is modest (at most about 30%, high qubits). |

The block, chunk and group sizes were swept (ten combinations) and are flat within noise.

### Measured results (i3-1115G4)

Forward ansatz, `scale_26_qubits` (depth 12, with the AVX-512 kernels of section 2.1):

| Qubits | Session start | Now |
| :--- | :--- | :--- |
| 20 | 954 ms | 125 ms |
| 22 | 4.85 s | 603 ms |
| 24 | 21.2 s | 2.52 s |
| 26 | 92.1 s | 12.1 s |

Compiled-circuit executor, CPU FP64 against the FP32 iGPU (`backend_auto_bench`, AVX-512 build):

| Qubits | CPU | iGPU | iGPU speed relative to CPU |
| :--- | :--- | :--- | :--- |
| 16 | 5.0 ms | 9.6 ms | 0.52x |
| 20 | 123 ms | 423 ms | 0.29x |
| 24 | 3.0 s | 8.4 s | 0.35x |
| 26 | 14.7 s | 30.7 s | 0.48x |

Adjoint gradient, depth 6:

| Qubits | Before | After |
| :--- | :--- | :--- |
| 12 | 2.02 ms | 0.77 ms |
| 16 | 35.5 ms | 10.2 ms |
| 20 | 729 ms | 317 ms |
| 22 | 3352 ms | 1613 ms |

End-to-end training, session-start build against current (identical final losses):

| Workload | Before | Now |
| :--- | :--- | :--- |
| `qnn_train` 12 qubits, 20 epochs | 0.50 s | 0.20 s |
| `qnn_train` 16 qubits, 3 epochs | 1.36 s | 0.48 s |
| `qnn_train` 20 qubits, 1 epoch | 17.78 s | 6.32 s |
| `zinc_qnn_train`, 10k molecules, 3 epochs | 4.89 s | 2.91 s |
| `zinc_qnn_12qubit_fused` per-molecule latency | about 48 us | about 48 us (unchanged) |

The last row did not change: that binary uses its own fused observable path.

Single benchmark runs on this laptop vary by roughly 15 to 30%; compare best-of-N numbers.

### 2.1 Low-qubit and AVX-512 kernels

Two separate improvements, measured on the i3-1115G4 (AVX-512F, 48 KB L1, 1.25 MB L2 per core):

* **Qubits 0 and 1 were scalar.** The AVX2 Ry/Rz kernels vectorized only qubit 2 and up. The
  block-local pass applies qubits 0 to 11 to every 4096-amplitude block, so those two scalar
  qubits cost nearly half of that pass. `simd::lowq` now splits each pair of vectors into the
  elements with the qubit bit 0 and 1 (one unpack or 128-bit permute), rotates them as ordinary
  vectors and merges back. This helps any AVX2 machine, not only AVX-512 ones.
* **AVX-512F kernels** (`simd::avx512`, `simd::unitary`): Ry, Rz, real and complex pair rotations,
  complex gates on qubits 0 to 2 (two `vpermt2pd` per vector pair), and the gradient cross-sum.
  Single-gate Ry/Rz use them for states up to 2^21 amplitudes; larger states keep the AVX2
  kernels with streaming stores. Block and chunk kernels always use them when available.

Ry on one cache-resident 4096-amplitude block, time per call:

| Qubit | Before | AVX2 + low-qubit fix | AVX-512 |
| :--- | :--- | :--- | :--- |
| 0 | 4605 ns | 1085 ns | 955 ns |
| 1 | 4068 ns | 1108 ns | 961 ns |
| 2 | 876 ns | 872 ns | 953 ns |
| 5 | 987 ns | 984 ns | 900 ns |
| 11 | 996 ns | 982 ns | 835 ns |

For qubit 3 and up AVX-512 gains 0 to 17% over AVX2: a 4096-amplitude block is 64 KB, which does
not fit the 48 KB L1, so the loop is limited by L2 bandwidth, not by vector width.

Best of N, previous commit against the new code (same machine, same inputs):

| Path | Qubits | Previous | AVX2 + fix | AVX-512 |
| :--- | :--- | :--- | :--- | :--- |
| Forward ansatz, depth 12 | 14 | 1.40 ms | 1.09 ms | 0.98 ms |
| | 18 | 13.84 ms | 11.94 ms | 11.23 ms |
| | 20 | 141 ms | 126 ms | 123 ms |
| Adjoint gradient, depth 6 | 12 | 0.78 ms | 0.69 ms | 0.64 ms |
| | 16 | 9.7 ms | 8.6 ms | 8.3 ms |
| | 20 | 490 ms | 420 ms | 302 ms |
| Compiled-circuit executor, depth 12 | 14 | 1.61 ms | 1.22 ms | 1.07 ms |
| | 18 | 16.1 ms | 13.3 ms | 10.5 ms |
| | 20 | 237 ms | 206 ms | 163 ms |

End-to-end trainers (best of 3, includes the held-out evaluation; results identical to every
printed digit with `T1000_SIMD=avx2` and with AVX-512):

| Workload | Previous | Now |
| :--- | :--- | :--- |
| `qnn_train` 12 qubits, 20 epochs | 0.20 s | 0.17 s |
| `qnn_train` 16 qubits, 3 epochs | 0.66 s | 0.59 s |
| `qnn_train` 20 qubits, 1 epoch | 14.20 s | 8.99 s |
| `zinc_qnn_train`, 10k molecules, 3 epochs | 2.73 s | 2.42 s |

The Rz/CZ diagonal pass was not ported: it is about 15% of a layer at 18 qubits, so a wider
vector would gain a few percent at most. The Ry layer runs at about 69 GB/s at 18 qubits, which is
L3 bandwidth, so little SIMD headroom remains there.

### Known limits

* CNOT, SWAP, controlled-phase and Rademacher ops are barriers inside the layered executor, so
  each is still its own pass. On CNOT-heavy circuits the iGPU is about even with the CPU
  (22 qubits, CNOT ladder: 1.78 s iGPU, 2.24 s CPU).
* A 26-qubit layer still costs about 9 memory passes.
* Nothing here was measured on the Core Ultra 7 265. It has no AVX-512, so it runs the AVX2
  kernels, which gain the qubit-0/1 fix but not the 512-bit paths.

---

## 3. Result recording (`t1000::results`)

Binaries write results through `t1000::results::{write, create, write_scored, write_if_better}`,
drop-in replacements for `fs::write` and `File::create`. A run never touches a saved result
unless it is told to record.

| Mode | Where a path inside the workspace goes |
| :--- | :--- |
| Default | `target/bench-scratch/<host>/<path>` (git-ignored) |
| `--record` or `T1000_RECORD=1`, reference host | the path itself |
| `--record` or `T1000_RECORD=1`, any other host | `results/hosts/<host>/<path without "results/">` |

Paths outside the workspace (an explicit `--out /tmp/x.json`) are always used as given.
Programs may still print the path they were asked to write; the line starting `[results]` on
stderr says where the file really went.

### Keep-best rules

`write_scored` stores a `.best` sidecar next to each recorded file:

```
settings qubits=12;depth=3;epochs=20;lr=0.05;...
metric holdout_loss 3.026890506961245e-1
guard holdout_accuracy_pct 8.958333333333334e1
```

A recorded run replaces the saved one only when:

1. its deciding metric improves (equal does not count), and
2. the guard metric has not regressed by more than its tolerance.

A run whose settings differ from the saved run never competes with it. It is saved beside it as
`<name>.<settings-hash>.<ext>` with its own sidecar. `--force` or `T1000_RECORD_FORCE=1`
replaces the saved run unconditionally. Metrics must be held-out numbers, not training numbers.

| Binary | Deciding metric (lower is better) | Guard |
| :--- | :--- | :--- |
| `qnn_train` | holdout loss on 96 unseen phase-shifted points | holdout accuracy may drop at most 1 point |
| `zinc_qnn_train` (settings include the head mode), `zinc_linear_head_train` (settings include depth and a fingerprint of the circuit weights), `zinc_qnn_12qubit_hybrid`, `zinc_qnn_20ep_eval` | held-out MSE divided by target variance, mean of logP, QED, SAS (1.0 means no better than the mean); shared code in `t1000::holdout_score` | mean Pearson r may drop at most 0.02 |
| `birds_qnn_train`, `birds_hybrid_qnn_train` | test loss of the best checkpoint | worst-group accuracy may drop at most 1 point |
| `mnist_qnn_train` | final test loss | test accuracy may drop at most 1 point |
| `scale_26_qubits` | 26-qubit latency | none |

Companion files follow the report. `zinc_qnn_12qubit_hybrid` skips rewriting its default checkpoint
when the report is kept, and writes it under the variant name when the report becomes a variant;
`mnist_qnn_train` does the same for its CSV. An explicit `--save-weights` path is always written.

All other binaries use the plain scratch-or-record behavior. The ablation reports
(`zinc_qnn_12qubit_fused`, `_pruned`, `_midcircuit`, `_qng`, `_reuploading`, `_zz_correlators`)
put several variants side by side in one file, so there is no single score to keep-best on.
`mnist_ultrametric_sorter` reports accuracy only, and `qm9_*`, `corpus_vqc_train`,
`qnn_protocol_train` and `qgnn_molecular_bench` report no clearly held-out number.

Caveats: the birds and MNIST trainers could not be run on the i3 (their data files are not in the
repository), so their recording path is compile-checked only. The birds trainers pick the "best
checkpoint" by test-set worst-group accuracy, so their test numbers are optimistic.

### Examples

```bash
# Experiment freely: nothing saved is touched.
cargo run --release --bin qnn_train -- --qubits 12 --epochs 20

# Save this machine's result (kept only if it improves on the saved run).
cargo run --release --bin qnn_train -- --qubits 12 --epochs 20 --record

# Replace the saved run regardless.
T1000_RECORD=1 T1000_RECORD_FORCE=1 cargo run --release --bin qnn_train -- --qubits 12 --record
```

### Using it in a new binary

```rust
let score = t1000::results::Score {
    metric: "holdout_loss",
    value: holdout_loss,
    better: t1000::results::Better::Lower,
    guard: Some(t1000::results::Guard {
        name: "holdout_accuracy_pct",
        value: holdout_acc,
        better: t1000::results::Better::Higher,
        tolerance: 1.0,
    }),
    settings: &settings_string, // everything that makes two runs comparable
};
match t1000::results::write_scored(&out_path, json, &score)? {
    t1000::results::Outcome::Written => {}        // recorded
    t1000::results::Outcome::WrittenVariant => {} // recorded beside the saved run
    t1000::results::Outcome::KeptBest => {}       // saved run is better
    t1000::results::Outcome::Scratch => {}        // not recording
}
```

---

## 4. Setting up a new machine

1. Build and test: `cargo check --workspace && cargo test --workspace`.
2. Run `cargo run --release --bin scale_26_qubits -- --record` and
   `cargo run --release --bin backend_auto_bench`. Results land in `results/hosts/<slug>/`.
3. If the register size where threads stop helping differs from 18, or the iGPU wins at some
   size, add the measured constants to `HostProfile` (`parallel_max_qubits`, `gpu_min_qubits`).
   Unmeasured hosts keep threading at every size and the default GPU threshold.
4. Do not build with `-C target-cpu=native` when more than one machine is in use: the i3 has
   AVX-512 and the Core Ultra 7 265 does not. The kernels already select instructions at run
   time.
5. Leave the build profile alone: fat LTO with one codegen unit measured no difference.
6. Keep the default thread count (all logical CPUs); on the i3, 4 threads were best or equal in a
   1/2/3/4-thread sweep.

---

## 5. ZINC readout calibration (`t1000::calibration`)

`zinc_qnn_train` predicts logP, QED and SAS as `mu + sigma * <Z_k>` on qubits 0 to 2, with `mu`
and `sigma` fixed from the data. After 20 epochs the circuit outputs correlate with the targets
(Pearson r 0.46, 0.21, 0.41) but the squared error stays at the level of predicting the mean,
because `<Z>` is tiny (it has a spread of order 1e-3 to 1e-2) so `sigma * <Z>` barely moves the
prediction. Two tests, both scored on the 2,000 held-out molecules:

* **Post-hoc affine fit** (always printed): fit `y ~ slope * <Z> + intercept` per property by least
  squares on the 10,000 training molecules only. It recovers about r squared and no more.
* **Calibrated-head training** (`--calibrated-head`, off by default): refit that line on 2,000
  training molecules at the start of every epoch and train the circuit against it. The adjoint
  gradient uses `slope / sigma` as the coefficient of each readout, which is exactly 1 for the
  default head, so the default path is bit-for-bit unchanged.

Held-out results, 20 epochs, i3-1115G4 (R squared is `1 - MSE / variance`):

| Model | Relative MSE | Mean r | R2 logP | R2 QED | R2 SAS |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Default head (saved run) | 0.9915 | 0.359 | 0.021 | 0.002 | 0.003 |
| Default head + post-hoc affine fit | 0.860 | 0.359 | 0.209 | 0.043 | 0.169 |
| `--calibrated-head`, lr 0.03 | 0.828 | 0.435 | 0.141 | 0.080 | 0.298 |
| `--calibrated-head`, lr 0.01 (saved variant) | 0.7985 | 0.436 | 0.126 | 0.064 | 0.415 |
| Quadratic ridge (same features) | 0.386 | n/a | 0.634 | 0.561 | 0.646 |

The two saved rows are computed from the saved JSON files; the others are from single runs
printed to the terminal.

Learning rate 0.1 and 40 epochs gave 0.83 and 0.84, so tuning does not change the picture. The
calibrated head helps (every property now beats the mean predictor) but the circuit still
explains only 13%, 6% and 42% of the held-out variance of logP, QED and SAS, against 63%, 56% and
65% for quadratic ridge regression.

**Caveats**

* The fitted slopes are in the hundreds (logP's is negative): the head amplifies a wiggle of order
  1e-3 around a learned mean of `<Z>`. This is exact in the simulator. On hardware the readout
  would have to resolve that wiggle against shot noise of about `1 / sqrt(shots)`, which suggests
  of the order of a million shots per readout (an estimate, not measured).
* Training is deterministic for a given setting, but different learning rates move the
  per-property R squared by several hundredths (logP 0.13 to 0.26, SAS 0.20 to 0.42 across the
  sweep), so treat single-property differences as noisy.
* Saved on this host: `results/hosts/i3-1115g4/tests/benchmarks/zinc_molecular/outputs/`
  `zinc_qnn_10k_results.json` (default head, 20 epochs) and the `.50641bcc.json` variant
  (calibrated head, lr 0.01, 20 epochs).

### 5.1 Wider learned readout (`zinc_qnn_12qubit_hybrid`)

The hybrid trainer uses 12 qubits and a learned classical readout over the circuit's observables
(111 parameters: 72 quantum rotations and 39 readout weights). It does not close the gap that
calibration leaves, and it is unstable. Fresh runs (no `--resume`), 10,000 training and 2,000
held-out molecules, relative MSE as in section 5 (computed from the printed held-out MSE):

| Run | MSE logP / QED / SAS | r logP / QED / SAS | Relative MSE |
| :--- | :--- | :--- | :--- |
| defaults, 20 epochs | 2.134 / 0.0199 / 0.669 | 0.28 / 0.21 / 0.48 | 0.994 |
| defaults, 10 epochs | 2.192 / 0.0202 / 0.662 | -0.25 / 0.17 / 0.56 | 1.005 |
| `--max-norm 100`, 10 epochs | 2.132 / 0.0177 / 0.724 | 0.43 / 0.33 / -0.56 | 0.983 |
| `--lr 0.005`, 10 epochs | 2.169 / 0.0181 / 0.684 | -0.10 / 0.29 / 0.15 | 0.976 |
| `--lr 0.002 --max-norm 100`, 10 epochs | 2.120 / 0.0187 / 0.655 | 0.47 / 0.23 / 0.60 | 0.965 |

For comparison, the 10-qubit trainer with `--calibrated-head` reaches 0.7985 and quadratic ridge
0.386. Two things stand out. With the default `--max-norm 6`, 39% to 65% of batches are clipped
and the training loss stays flat (4.21 at epoch 1, 4.13 at epoch 20). And the correlation changes
sign between settings (SAS r from +0.60 to -0.56), which points to a poorly conditioned
optimization and not a capacity limit. A wider readout alone is therefore not the missing piece.

### 5.2 Where the ZINC signal is lost (`zinc_feature_probe`)

`cargo run --release --bin zinc_feature_probe -- --weights <zinc_qnn_train report>` fits the same
ridge regression (penalty chosen on a validation split) to several feature sets on the identical
10,000 / 2,000 split, so differences come from the features alone. The trained circuit is the
`--calibrated-head`, lr 0.01, 20-epoch model of section 5 (weights are stored in the report as
`final_weights`). Relative MSE is `1 - mean(R2)`; lower is better.

| Feature set | n | Model | Relative MSE | R2 logP | R2 QED | R2 SAS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| raw features | 19 | linear ridge | 0.495 | 0.601 | 0.345 | 0.569 |
| raw features | 19 | quadratic ridge | 0.386 | 0.634 | 0.561 | 0.646 |
| encoded amplitudes, raw inputs | 19 | linear ridge | 0.503 | 0.588 | 0.331 | 0.571 |
| encoded amplitudes, raw inputs | 19 | quadratic ridge | 0.414 | 0.628 | 0.495 | 0.633 |
| encoded amplitudes, standardized inputs | 19 | linear ridge | 0.525 | 0.558 | 0.331 | 0.536 |
| encoded amplitudes, standardized inputs | 19 | quadratic ridge | 0.416 | 0.620 | 0.509 | 0.623 |
| Z/ZZ of encoded state, raw inputs | 55 | linear ridge | 0.619 | 0.485 | 0.183 | 0.473 |
| Z/ZZ after untrained circuit, raw inputs | 55 | linear ridge | 0.471 | 0.605 | 0.363 | 0.619 |
| Z/ZZ after trained circuit, raw inputs | 55 | linear ridge | 0.473 | 0.603 | 0.358 | 0.620 |
| Z0..Z2 only, untrained circuit, raw inputs | 3 | linear ridge | 0.786 | 0.260 | 0.059 | 0.324 |
| all 10 single-qubit Z, untrained circuit, raw inputs | 10 | linear ridge | 0.584 | 0.504 | 0.197 | 0.545 |
| Z0..Z2 only, trained circuit, raw inputs | 3 | linear ridge | 0.733 | 0.231 | 0.089 | 0.482 |
| all 10 single-qubit Z, trained circuit, raw inputs | 10 | linear ridge | 0.600 | 0.535 | 0.105 | 0.559 |
| Z/ZZ of encoded state, standardized inputs | 55 | linear ridge | 0.900 | 0.094 | 0.141 | 0.064 |
| Z/ZZ after untrained circuit, standardized inputs | 55 | linear ridge | 0.780 | 0.244 | 0.232 | 0.185 |

Saved for this host in `results/hosts/i3-1115g4/experiments/zinc_feature_probe.json`.

What the rows say:

* **The encoding is not the problem.** The circuit receives 19 amplitudes (the L2-normalised
  features). A ridge fitted to those scores almost the same as on the raw features (0.503 against
  0.495 linear, 0.414 against 0.386 quadratic), although one feature (molecular weight) carries
  99.1% of the squared norm. Normalising away the scale costs almost nothing.
* **The circuit is not destroying information.** A linear ridge on the 55 Z and ZZ expectation
  values of the circuit output scores 0.471 for an untrained circuit and 0.473 for the trained one,
  slightly better than linear ridge on the raw features (0.495).
* **The readout width is the bottleneck.** The same trained circuit read through 3 observables
  (Z0 to Z2, all the training model uses) gives 0.733, through the 10 single-qubit Z values 0.600,
  and through all 55 Z and ZZ values 0.473. The model's own readout, one number per property,
  reaches 0.7985.
* **Gradient training of the circuit adds little to what a linear head can already extract.** For
  the 55 features the trained and untrained circuits score the same (0.473 and 0.471). Training
  helps the narrow readouts (3 observables: 0.786 untrained to 0.733 trained) by concentrating
  information into them.
* **Standardising the inputs hurts** (0.78 to 0.90). Z and ZZ depend only on the squared
  amplitudes, and standardised features are signed, so their sign is lost. Keep the inputs
  non-negative (the raw features are) or read the state in more than one basis.

**Implication.** A fixed circuit with a closed-form linear head over all Z and ZZ values should
reach a relative MSE near 0.47 with no gradient training, against 0.7985 for the best trained
single-readout model. The probe fits that head on the training set only and scores it on
held-out molecules, so the comparison is fair; it has not been turned into a trainer yet.
Caveat: on hardware the 55 values come from one set of computational-basis shots, but the
informative variation in each is small (raw-input Z values sit near -1), so the shot count needed
has not been measured.

### 5.3 Circuit feature map with a closed-form linear head (`zinc_linear_head_train`)

Section 5.2 showed that the circuit's 55 Z and ZZ outputs hold the signal and that the
single-readout models lose it. `zinc_linear_head_train` turns that into a trainer: amplitude-encode
each molecule, run the circuit, read out all Z and ZZ expectation values (one computational-basis
measurement gives all of them), and fit a ridge regression on the training molecules. There is no
gradient training. The circuit is either the default initialisation or the weights stored in a
`zinc_qnn_train` report (`--weights report.json`). The fitted head (standardisation plus ridge) and
the circuit weights are stored in the report, so the model is reproducible and reloadable.

```bash
cargo run --release --bin zinc_linear_head_train                    # scratch copy
cargo run --release --bin zinc_linear_head_train -- --depth 6 --record
```

Options: `--depth N` (default 4), `--weights report.json`, `--observables w4|wK|all|z` (default `w4`,
section 5.6; `all` is the 55 Z and ZZ values of this section, `z` the 10 single-qubit values),
`--samples N`, `--eval-samples N`. Library code: `t1000::linear_head` (`z_features_into`,
`CircuitFeatureMap`, `LinearHead`), `t1000::zinc_data` (CSV loader).

Held-out results on 2,000 molecules, 10,000 training molecules, i3-1115G4 (relative MSE as in
section 5). Rows 1 to 2 use the raw features as the circuit input, which is the original
behaviour; section 5.4 shows that rescaling the inputs improves on them:

| Model | Relative MSE | Mean r | R2 logP / QED / SAS |
| :--- | :--- | :--- | :--- |
| Circuit (depth 4, default init) + linear head, 55 observables | 0.4655 | 0.727 | 0.611 / 0.368 / 0.625 |
| Same, circuit trained by `zinc_qnn_train` (calibrated head) | 0.4684 | 0.725 | 0.606 / 0.366 / 0.623 |
| Same, 10 single-qubit Z observables only | 0.5845 | 0.631 | n/a |
| Best gradient-trained single-readout model (section 5) | 0.7985 | 0.436 | 0.126 / 0.064 / 0.415 |
| Linear ridge on the raw features | 0.4952 | n/a | 0.601 / 0.345 / 0.569 |
| Quadratic ridge on the raw features | 0.386 | n/a | 0.634 / 0.561 / 0.646 |

Circuit depth (default initialisation, 55 observables):

| Depth | 1 | 2 | 3 | 4 | 6 | 8 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Relative MSE | 0.5520 | 0.5275 | 0.4818 | 0.4655 | 0.4562 | 0.4621 |
| Mean r | 0.659 | 0.679 | 0.715 | 0.727 | 0.733 | 0.729 |

Depth 6 is saved as a variant for this host. Findings:

* **It beats linear ridge on the raw features** (0.4655 against 0.4952; logP MSE 0.843 against
  0.864, QED 0.0123 against 0.0127, SAS 0.258 against 0.296) but not quadratic ridge (0.386).
* **Cost.** Feature extraction takes 0.78 s for 10,000 molecules, the head fit 32 ms, and
  inference is 78 us per molecule including the circuit, against about 16 s for 20 epochs of
  gradient training. The result is bit-reproducible.
* **Training the circuit does not help** (0.4684 against 0.4655), consistent with section 5.2.
* **Depth saturates** around 4 to 6.
* **Width matters:** 10 observables give 0.5845, 55 give 0.4655.

**Numerical note.** The shared ridge solver forms its Gram matrix from raw second moments, which
loses precision for near-constant columns, and circuit outputs are exactly that (a Z value near -1
with a spread of 1e-4). At depths 2 and 3 this made the Cholesky factorisation fail for the smallest
penalty. `LinearHead` therefore standardises the features before the ridge fit; the depths that
worked before give identical results. The chosen penalty sits at the lower edge of the grid
(1e-7) for the default circuit, so a smaller penalty might help slightly.

**Hardware caveat.** The 55 outputs vary across molecules with a median standard deviation of only
2.2e-3 for raw inputs. An earlier estimate here (about 2e5 shots, "several million" for a comfortable
margin) was a bound from those spreads alone and is superseded by the sampling simulation in section
5.5, which shows the head amplifies sampling noise much more than the spreads suggest.

### 5.4 Input scaling for the amplitude encoding

Amplitude encoding L2-normalises the feature vector, and one raw feature (molecular weight, in the
hundreds next to counts below 25) carries 99.1% of the squared norm. The Z and ZZ readouts depend
only on squared amplitudes, so signed (standardised) inputs lose information (section 5.2), but
non-negative rescaling is safe. `--scale` on `zinc_linear_head_train` fits the scaling on the
training molecules only and clamps held-out values to [0, 1]: `maxabs` is `x / max`, `minmax` is
`(x - min) / (max - min)`, `raw` is the original behaviour. Depth 4, 55 observables, default
circuit, 10,000 training and 2,000 held-out molecules:

| Input scaling | Relative MSE | Mean r | Median feature spread |
| :--- | :--- | :--- | :--- |
| raw | 0.4655 | 0.727 | 2.2e-3 |
| minmax | 0.4415 | 0.745 | 4.3e-2 |
| maxabs (default) | **0.4312** | **0.752** | 3.9e-2 |

Depth sweep with `maxabs`:

| Depth | 1 | 2 | 3 | 4 | 6 | 8 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Relative MSE | 0.6430 | 0.5226 | 0.4546 | 0.4312 | 0.4463 | 0.4550 |
| Mean r | 0.591 | 0.686 | 0.736 | 0.752 | 0.742 | 0.736 |
| Median feature spread | 7.4e-2 | 5.1e-2 | 5.1e-2 | 3.9e-2 | 1.8e-2 | 1.5e-2 |

Depth 4 is the best, and deeper circuits shrink the feature spreads. With only the 10 single-qubit Z
observables the result is 0.6188. The default is now `maxabs` with weight-4 parities (section 5.6);
the 55-observable `maxabs` run and the raw-input runs are saved beside it as variants.

### 5.5 Shot noise (`zinc_shot_noise`)

`zinc_shot_noise` samples each molecule's circuit output `N` times in the computational basis,
rebuilds the 55 Z and ZZ values from the sampled bitstrings, and predicts from them. Two heads are
compared: one fitted on exact (noise-free) features and applied to sampled ones, and one fitted on
features sampled with the same `N`. Depth 4, `maxabs` inputs, noisy-trained head fitted on all 10,000
training molecules, 2 seeds per row (relative MSE; the mean predictor is 1.0 and the noise-free
reference is 0.4312):

| Shots per molecule | Exact-trained head | Noisy-trained head | Mean r (noisy-trained) |
| :--- | :--- | :--- | :--- |
| 30 | 19,437 | 0.942 | 0.232 |
| 100 | 5,567 | 0.872 | 0.344 |
| 300 | 1,865 | 0.787 | 0.445 |
| 1,000 | 563 | 0.685 | 0.544 |
| 3,000 | 183 | 0.611 | 0.610 |
| 10,000 | 58 | 0.555 | 0.657 |
| 30,000 | 19.1 | 0.523 | 0.682 |
| 100,000 | 6.3 | 0.499 | 0.701 |

* **A head fitted on exact simulation cannot be deployed on sampled features.** It puts large
  weights on nearly cancelling features and amplifies the sampling noise by orders of magnitude
  (still 6.3 at 100,000 shots). The head has to be fitted on features with the deployment noise.
* **A noisy-trained head degrades gracefully.** It beats the mean predictor from about 30 shots, beats
  the best gradient-trained simulator model (0.7985) from about 300 shots, and reaches 0.499 at
  100,000 shots, close to linear ridge on the raw features (0.4952) but still short of the
  noise-free 0.4312. Approaching the noise-free value takes well over 100,000 shots per molecule.
* **Input scaling matters on hardware.** With raw inputs the same experiment (2,000 noisy training
  molecules, 3 seeds) gives 0.865 at 10,000 shots and 0.715 at 100,000 shots, against 0.565 and
  0.509 for `maxabs`.
* The training-size effect is small: the noise-free fit on 2,000 molecules scores 0.4383 against
  0.4312 on 10,000, and the noisy-trained head at 100,000 shots scores 0.509 with 2,000 training
  molecules against 0.499 with 10,000.

This models only sampling noise on an ideal device. Gate errors, decoherence and readout errors are
not included, and the circuit is simulated exactly.

### 5.6 Noise-aware head: what helps and what does not

Section 5.5 showed that the head must be fitted on sampled features and that 100,000 shots only
reach 0.499. Three ideas were tested on identical sampled data (`zinc_shot_noise --sweep`, depth 4,
`maxabs`, 3,000 noisy training molecules, 2 seeds; relative MSE):

| Configuration | Noise-free | 300 shots | 1,000 | 3,000 | 10,000 | 100,000 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| all 55, penalty grid to 10 | 0.4312 | 0.7821 | 0.6826 | 0.6070 | 0.5636 | 0.5027 |
| all 55, penalty grid to 1e4 | 0.4312 | 0.7821 | 0.6826 | 0.6070 | 0.5636 | 0.5039 |
| top 30 by correlation | 0.4757 | 0.8229 | 0.7384 | 0.6708 | 0.6245 | 0.5491 |
| top 20 by correlation | 0.5210 | 0.8504 | 0.7985 | 0.7727 | 0.7242 | 0.6058 |
| top 10 by correlation | 0.6893 | 0.8954 | 0.8500 | 0.7954 | 0.7664 | 0.7150 |
| 10 single-qubit Z | 0.6188 | 0.9009 | 0.8477 | 0.7954 | 0.7422 | 0.6654 |

* **A stronger ridge penalty does not help.** Widening the grid changes nothing: the validation
  split never picks a penalty above 1e-2, even at 300 shots.
* **Fewer features hurt at every shot count.** The information is spread over many features, and
  the head's averaging over many noisy values is what keeps it stable.
* **More features from the same shots help.** One set of computational-basis samples yields every
  parity (Z-string) over any subset of the qubits, and a fast Walsh-Hadamard transform computes all
  1,023 of them. `Observables::ParityUpTo(k)` uses those of weight at most `k`.

Noise-free held-out accuracy by maximum parity weight (depth 4, `maxabs`; weight 2 is the 55 Z and ZZ
values):

| Maximum weight | 1 | 2 | 3 | 4 | 5 | 10 (all) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Features | 10 | 55 | 175 | 385 | 637 | 1,023 |
| Relative MSE | 0.6188 | 0.4312 | 0.3963 | **0.3959** | 0.4008 | 0.4099 |
| Mean r | 0.604 | 0.752 | 0.777 | 0.777 | 0.774 | 0.768 |

Weight 4 (385 features) is the default. It is within 0.01 of quadratic ridge on the raw features
(0.386) and well ahead of linear ridge (0.4952). Under sampling noise the head fitted on sampled
features, using all 10,000 training molecules and 2 seeds (saved for this host as
`zinc_shot_noise_maxabs_w4.json` and `_w6.json`):

| Shots per molecule | 55 features (w2) | 385 features (w4) | 847 features (w6) |
| :--- | :--- | :--- | :--- |
| 30 | 0.942 | 0.892 | 0.886 |
| 100 | 0.872 | 0.768 | 0.752 |
| 300 | 0.787 | 0.666 | 0.653 |
| 1,000 | 0.685 | 0.573 | 0.563 |
| 3,000 | 0.611 | 0.523 | 0.510 |
| 10,000 | 0.555 | 0.484 | 0.476 |
| 30,000 | 0.523 | 0.458 | 0.453 |
| 100,000 | 0.499 | 0.442 | 0.435 |

* **Roughly 7 to 8 times fewer shots for the same accuracy** (by log interpolation: relative MSE
  0.555 takes 10,000 shots with 55 features, about 1,500 with 385 and about 1,200 with 847).
* **Linear ridge on the raw features (0.4952) is reached at about 7,000 shots** with 385 features
  and about 5,000 with 847, against more than 100,000 with 55.
* Higher weights keep helping under noise even though they stop helping noise-free (weight 6 is
  worse than weight 4 without noise), because the extra features average the noise out. Weight 10
  gives the same as weight 6 and costs more to fit.
* Quadratic ridge (0.386) is not reached at 100,000 shots (0.435 with 847 features).

This still models only sampling noise on an ideal device with the circuit simulated exactly.

### 5.7 Device errors on top of sampling noise (`t1000::device_noise`)

`zinc_shot_noise` can add device errors to the sampled distribution: `--gate-error p2` (a global
depolarizing channel with fidelity `F = (1 - p2/10)^80 (1 - p2)^40` for the depth-4, 10-qubit
ansatz), `--readout-error e` (a measured bit flips with `P(0 to 1) = e/2` and `P(1 to 0) = e`, as an
exact per-qubit confusion matrix) and `--mitigate-readout` (invert that matrix on the measured
frequencies). Analytic checks are in the tests: under symmetric readout error `e` a weight-`w` parity
shrinks by `(1 - 2e)^w`, and every non-trivial parity scales by `F` under depolarizing.

Three regimes, with error rates in the range usually quoted for current hardware (the exact
numbers are illustrative, not measurements of a device):

| Scenario | Two-qubit error | Circuit fidelity F | Readout error (P(1 to 0)) |
| :--- | :--- | :--- | :--- |
| Good (ion-trap-like) | 1e-3 | 0.953 | 0.5% |
| Typical (superconducting-like) | 5e-3 | 0.786 | 2% |
| Poor | 1e-2 | 0.618 | 4% |

Relative MSE of the head fitted on the noisy features (depth 4, `maxabs`, 5,000 noisy training
molecules, 2 seeds; the mean predictor is 1.0). The noise-free values are 0.4312 (55 features), 0.3959
(385) and 0.4086 (847):

| Scenario | Shots | 55 features (w2) | 385 features (w4) | 847 features (w6) |
| :--- | :--- | :--- | :--- | :--- |
| Ideal device | 1,000 | 0.688 | 0.587 | 0.571 |
| | 10,000 | 0.559 | 0.491 | 0.489 |
| | 100,000 | 0.501 | 0.448 | 0.445 |
| Good | 1,000 | 0.701 | 0.605 | 0.583 |
| | 10,000 | 0.561 | 0.494 | 0.496 |
| | 100,000 | 0.504 | 0.453 | 0.448 |
| Typical | 1,000 | 0.733 | 0.631 | 0.614 |
| | 10,000 | 0.581 | 0.519 | 0.512 |
| | 100,000 | 0.517 | 0.467 | 0.454 |
| Poor | 1,000 | 0.797 | 0.690 | 0.672 |
| | 10,000 | 0.625 | 0.552 | 0.537 |
| | 100,000 | 0.531 | 0.477 | 0.476 |

* **The advantage of higher-order parities survives device errors.** At 10,000 shots 385 features
  beat 55 by 0.07 on a good device, 0.06 on a typical one and 0.07 on a poor one.
* **Readout mitigation is unnecessary** when the head is fitted on data from the same device. On the
  poor device, weight-4 gives 0.6892 / 0.5512 / 0.4764 with mitigation against 0.6899 / 0.5521 /
  0.4766 without. The head standardises every feature, and readout error maps each parity onto itself
  plus lower-weight parities, which are all in the feature set, so the distortion is absorbed by the
  fit. (This relies on the head being trained on device data; a head fitted on a simulator would
  still be unusable, section 5.5.)
* **The cost of device errors is an effective loss of shots.** For weight 4 at 10,000 shots, a good
  device behaves like an ideal one at about 9,000 shots, a typical one like about 4,400 and a poor one
  like about 2,000 (log interpolation). That is worse than the `F^2` expected from the gate error alone
  (0.91, 0.62, 0.38) because readout error also shrinks the weight-4 and weight-6 parities (by
  `(1 - 2e)^w` for symmetric error `e`).
* **Linear ridge on the raw features (0.4952) is still reached at 100,000 shots in every scenario**
  with 385 or 847 features (0.453, 0.467, 0.477 for 385), and at 10,000 shots only on the good device
  with 385 features (0.494; 847 features give 0.496).
* Saved for this host: `zinc_shot_noise_maxabs_w4_g<p2>_r<e>.json` for each scenario.

**What is not modelled.** The circuit is simulated from an ideal amplitude-encoded input. On hardware
the 19-amplitude state would have to be prepared by a circuit of its own, plausibly of the order of a
hundred two-qubit gates; at the typical error rate that alone would roughly halve the fidelity again
(the "poor" scenario's F of 0.62 is about what typical gate errors plus such a preparation would
give, but that is an estimate, not a compiled circuit). Gate errors are modelled as a global
depolarizing channel, not as the structured errors a device has; crosstalk, drift and decoherence
during measurement are not included.

