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
| `zinc_qnn_train`, `zinc_qnn_12qubit_hybrid`, `zinc_qnn_20ep_eval` | held-out MSE divided by target variance, mean of logP, QED, SAS (1.0 means no better than the mean); shared code in `t1000::holdout_score` | mean Pearson r may drop at most 0.02 |
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
