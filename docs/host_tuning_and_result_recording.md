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

The engine's kernels are 256-bit AVX2+FMA even when the CPU reports AVX-512; the label says so.

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

Forward ansatz, `scale_26_qubits` (depth 12):

| Qubits | Session start | Now |
| :--- | :--- | :--- |
| 20 | 954 ms | 158 ms |
| 22 | 4.85 s | 829 ms |
| 24 | 21.2 s | 3.06 s |
| 26 | 92.1 s | 14.4 s |

Compiled-circuit executor, CPU FP64 against the FP32 iGPU (`backend_auto_bench`):

| Qubits | CPU | iGPU | iGPU speed relative to CPU |
| :--- | :--- | :--- | :--- |
| 16 | 6.6 ms | 10.3 ms | 0.64x |
| 20 | 170 ms | 233 ms | 0.73x |
| 24 | 3.4 s | 7.5 s | 0.45x |
| 26 | 15.7 s | 34.7 s | 0.45x |

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

### Known limits

* CNOT, SWAP, controlled-phase and Rademacher ops are barriers inside the layered executor, so
  each is still its own pass. On CNOT-heavy circuits the iGPU is about even with the CPU
  (22 qubits, CNOT ladder: 1.78 s iGPU, 2.24 s CPU).
* A 26-qubit layer still costs about 9 memory passes.
* Nothing here was measured on the Core Ultra 7 265.

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
