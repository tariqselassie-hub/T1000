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

## Performance note

The portable wheel targets baseline x86-64. Kernels that pick AVX2, FMA, AVX-512 or AVX-VNNI at
run time (most of the engine) still use them. The few code paths gated at compile time on
`target_feature = "avx2"` take their scalar fallback in this build; a wheel built with
`RUSTFLAGS="-C target-cpu=x86-64-v3"` would enable them for AVX2 machines, including Kaggle's
CPUs, at the cost of not running on older processors.
