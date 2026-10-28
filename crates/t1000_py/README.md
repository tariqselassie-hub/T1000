<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Terrence A. Jones SR. All rights reserved. -->

# `t1000` Python wheel

Python bindings for the T-1000 engine, built with [PyO3](https://pyo3.rs) and
[maturin](https://www.maturin.rs). Long computations run in Rust with the GIL released.

```python
import numpy as np
import t1000

# Variational quantum circuit classifier, scikit-learn style. One qubit per feature; features
# are rotation angles, so scale them to about [0, pi]. Any two label values work.
clf = t1000.VqcClassifier(depth=3, epochs=30, lr=0.05, seed=11).fit(X_train, y_train)
clf.predict(X_test); clf.decision_function(X_test); clf.score(X_test, y_test)

# State-vector simulation from |0...0>. Qubit 0 is the least significant bit.
state = t1000.run_circuit(2, [("H", 0), ("CX", 0, 1)])         # complex128, length 4
probs = t1000.probabilities(3, [("RY", 0, 0.7), ("CP", 0, 1, np.pi / 2)])

# Ab initio VQE (STO-3G, RHF, UCCSD) with the exact FCI reference, returned as a dict.
t1000.vqe_catalog()                                            # ['H2', 'HeH+', 'LiH', ...]
report = t1000.vqe_solve("LiH")                                # report["fci_energy"], ...
t1000.vqe_solve(atoms=[(1, (0, 0, 0)), (1, (0, 0, 0.74))])     # any geometry, in angstrom
```

Gates: `H X Y Z S T`, `RX RY RZ` (angle in radians), `CX`/`CNOT`, `CZ`, `SWAP`, `CP`
(controlled phase, with an angle). Registers up to 26 qubits.

`VqcClassifier` is the same model as the `csv_fold_vqc` binary (`t1000::vqc_classifier`): on a
Titanic fold the two agree to the 9 decimals the binary prints. `vqe_solve` gives energies
identical to the `vqe_molecules` report.

## Install a built wheel

Wheels are `abi3`: one file per platform serves every Python from 3.9 on.

```bash
pip install t1000-0.1.0-cp39-abi3-win_amd64.whl          # Windows
pip install t1000-0.1.0-cp39-abi3-manylinux_*.whl         # Linux, including Kaggle
```

CI (`.github/workflows/wheels.yml`) builds the Windows and manylinux wheels on every change
under `crates/` and tests each on Python 3.11 and 3.13; download them from the run's artifacts.

### In a Kaggle notebook

Kaggle notebooks have no Rust toolchain and competition notebooks often run offline, so use a
prebuilt Linux wheel:

1. Upload the `manylinux` wheel as a Kaggle dataset (for example `t1000-wheel`).
2. Attach the dataset to the notebook and install it without the internet:

```python
!pip install --no-index --no-deps /kaggle/input/t1000-wheel/t1000-*-manylinux*.whl
import t1000
```

(`--no-deps` because Kaggle already has numpy.)

## Build from source

```bash
cd crates/t1000_py
uv tool run --from "maturin>=1.7,<2" maturin build --release      # wheel in target/wheels/
python -m pytest tests                                             # after installing the wheel
```

The crate is outside the root Cargo workspace on purpose: a PyO3 extension module only links
inside maturin's build, so `cargo test --workspace` leaves it alone.

## Performance

The wheel targets baseline x86-64 so it installs on any 64-bit machine, and still runs the AVX2,
FMA and AVX-512 kernels: the engine checks the CPU at run time (`is_x86_feature_detected!`) and
picks them whenever they are present. The two compile-time `target_feature = "avx2"` switches in
`simd/` only skip that check; they do not change which kernel runs.

Alternatives measured on the reference host (Core Ultra 7 265, medians of 5 to 21 runs, 2026-10-28):

| Workload | Portable wheel | `target-cpu=x86-64-v3` | Portable + fat LTO, 1 codegen unit |
| :--- | ---: | ---: | ---: |
| VQC fit, 712 rows x 8 qubits, 30 epochs | 81 ms | 86 ms | 82-86 ms |
| VQC decision_function, 20,000 rows | 18 ms | 18 ms | 19 ms |
| run_circuit, 20 qubits, 300 gates | 89-91 ms | 85 ms | 88 ms |
| run_circuit, 23 qubits, 60 gates | 282 ms | 270 ms | 280 ms |
| run_circuit, 24 qubits, 60 gates | 517 ms | 498 ms | 515 ms |
| vqe_solve H2O (12 qubits) | 14.9 ms | 15.1 ms | 13.9 ms |
| vqe_solve CH4 (16 qubits) | 332-339 ms | 331 ms | 326 ms |

An AVX2-only build gains about 4 % on the largest circuits and nothing elsewhere, and would stop
the wheel from installing on older CPUs (pip cannot choose wheels by CPU level). LTO trades a
few percent between workloads and only shrinks the wheel (490 KB to 462 KB). The default
portable build is therefore kept. The three builds agree to about 1e-12: circuits and VQE are
bit-identical, VQC outputs differ in the last bits from different instruction selection.
