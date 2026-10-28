# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.
"""Python bindings for the T-1000 quantum simulation engine.

    import numpy as np, t1000

    clf = t1000.VqcClassifier(depth=3, epochs=30).fit(X_angles, y)   # scikit-learn style
    state = t1000.run_circuit(2, [("H", 0), ("CX", 0, 1)])           # Bell state vector
    report = t1000.vqe_solve("LiH")                                   # ab initio VQE, as a dict

Everything runs in the compiled Rust engine; long calls release the GIL.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from . import _t1000

__version__: str = _t1000.__version__
__all__ = ["VqcClassifier", "run_circuit", "probabilities", "vqe_solve", "vqe_catalog", "__version__"]

_ONE_QUBIT = {"H", "X", "Y", "Z", "S", "T", "RX", "RY", "RZ"}
_TWO_QUBIT = {"CX", "CNOT", "CZ", "SWAP", "CP"}
_ANGLED = {"RX", "RY", "RZ", "CP"}


class VqcClassifier:
    """Binary classifier: a variational quantum circuit with one qubit per feature.

    Each feature is used as a rotation angle, so scale features to about [0, pi] first. The
    readout <Z> lies in [-1, 1]; its sign picks the class. Training is full-batch Adam on the
    squared error, with exact gradients, and is deterministic for a given ``seed``.

    Any two label values work (for example 0/1 or "died"/"survived"): the smaller one (in sorted
    order) maps to -1 and the larger to +1.
    """

    def __init__(self, depth: int = 3, epochs: int = 30, lr: float = 0.05, seed: int = 11):
        self.depth = depth
        self.epochs = epochs
        self.lr = lr
        self.seed = seed

    # scikit-learn compatible parameter access, so the class works in Pipeline and GridSearchCV.
    def get_params(self, deep: bool = True) -> dict:
        return {"depth": self.depth, "epochs": self.epochs, "lr": self.lr, "seed": self.seed}

    def set_params(self, **params: Any) -> "VqcClassifier":
        for key, value in params.items():
            if key not in self.get_params():
                raise ValueError(f"unknown parameter {key!r}")
            setattr(self, key, value)
        return self

    def fit(self, X, y) -> "VqcClassifier":
        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2-D (n_samples, n_features)")
        y = np.asarray(y)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"need exactly two classes, got {len(classes)}")
        self.classes_ = classes
        signs = np.where(y == classes[1], 1.0, -1.0)
        self.model_ = _t1000.vqc_train(X, signs, self.depth, self.epochs, self.lr, self.seed)
        self.n_features_in_ = X.shape[1]
        return self

    def decision_function(self, X) -> np.ndarray:
        self._check_fitted()
        return self.model_.decision_function(np.ascontiguousarray(X, dtype=np.float64))

    def predict(self, X) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0.0, self.classes_[1], self.classes_[0])

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

    @property
    def weights_(self) -> np.ndarray:
        self._check_fitted()
        return self.model_.weights

    @property
    def final_loss_(self) -> float:
        self._check_fitted()
        return self.model_.final_loss

    def _check_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise RuntimeError("call fit() first")

    def __repr__(self) -> str:
        return f"VqcClassifier(depth={self.depth}, epochs={self.epochs}, lr={self.lr}, seed={self.seed})"


def _normalize_gate(gate: Sequence[Any]) -> tuple:
    """("H", 0) | ("RY", 0, 0.5) | ("CX", 0, 1) | ("CP", 0, 1, 0.5) -> (name, [qubits], angle)."""
    if not gate:
        raise ValueError("empty gate")
    name = str(gate[0]).upper()
    args = list(gate[1:])
    if name in _ONE_QUBIT:
        n_qubits = 1
    elif name in _TWO_QUBIT:
        n_qubits = 2
    else:
        raise ValueError(f"unknown gate {gate[0]!r}")
    qubits = [int(q) for q in args[:n_qubits]]
    rest = args[n_qubits:]
    if name in _ANGLED:
        if len(rest) != 1:
            raise ValueError(f"{name} needs {n_qubits} qubit(s) and one angle: {gate!r}")
        return name, qubits, float(rest[0])
    if rest or len(qubits) != n_qubits:
        raise ValueError(f"{name} needs exactly {n_qubits} qubit(s): {gate!r}")
    return name, qubits, None


def run_circuit(n_qubits: int, gates: Iterable[Sequence[Any]]) -> np.ndarray:
    """Applies ``gates`` to |0...0> and returns the complex state vector (length 2**n_qubits).

    Gates: H X Y Z S T, RX RY RZ (with an angle in radians), CX/CNOT CZ SWAP, and CP (controlled
    phase, with an angle). Qubit 0 is the least significant bit of the state index.
    """
    return _t1000.run_circuit(int(n_qubits), [_normalize_gate(g) for g in gates])


def probabilities(n_qubits: int, gates: Iterable[Sequence[Any]]) -> np.ndarray:
    """Measurement probabilities |amplitude|^2 of :func:`run_circuit`."""
    return np.abs(run_circuit(n_qubits, gates)) ** 2


def vqe_solve(
    formula: Optional[str] = None,
    atoms: Optional[Sequence[tuple]] = None,
    charge: int = 0,
    freeze_core: bool = True,
    active_orbitals: Optional[int] = None,
) -> dict:
    """Ab initio VQE (STO-3G, restricted Hartree-Fock, UCCSD) with the exact FCI reference.

    Pass either ``formula`` (one of :func:`vqe_catalog`) or ``atoms`` as ``[(Z, (x, y, z)), ...]``
    in angstrom. Returns the engine's report: energies in hartree (``hf_energy``, ``fci_energy``,
    ``vqe_energy``), ``vqe_error_vs_fci``, qubit and parameter counts, timings and the optimizer
    history.
    """
    atom_list = None
    if atoms is not None:
        atom_list = [(int(z), tuple(float(c) for c in pos)) for z, pos in atoms]
    text = _t1000.vqe_solve_json(formula, atom_list, charge, freeze_core, active_orbitals)
    return json.loads(text)


def vqe_catalog() -> list:
    """Formulas of the built-in equilibrium molecule catalog."""
    return list(_t1000.vqe_catalog())
