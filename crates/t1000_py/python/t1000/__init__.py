# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.
"""Python bindings for the T-1000 quantum simulation engine.

    import numpy as np, t1000

    clf = t1000.VqcClassifier(depth=3, epochs=30).fit(X_angles, y)   # scikit-learn style
    state = t1000.run_circuit(2, [("H", 0), ("CX", 0, 1)])           # Bell state vector
    report = t1000.vqe_solve("LiH")                                   # ab initio VQE, as a dict
    sim = t1000.State(2).h(0).cx(0, 1)                                 # stateful simulator
    sim.expectation("Z0 Z1"), sim.counts(1000, seed=1, p2=0.01)       # Pauli <P>, noisy shots
    H = t1000.hamiltonian("H2")                                        # qubit Hamiltonian

Everything runs in the compiled Rust engine; long calls release the GIL.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from . import _t1000

__version__: str = _t1000.__version__
__all__ = [
    "VqcClassifier",
    "State",
    "run_circuit",
    "probabilities",
    "counts",
    "vqe_solve",
    "vqe_catalog",
    "hamiltonian",
    "classical_baselines",
    "adapt_vqe",
    "vqe_scan",
    "diatomic_scan",
    "chain",
    "__version__",
]

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


def counts(samples, n_qubits: int) -> dict:
    """Shot outcomes (basis-state indices) as ``{"bitstring": count}``, qubit 0 rightmost."""
    values, freq = np.unique(np.asarray(samples), return_counts=True)
    return {format(int(v), f"0{n_qubits}b"): int(c) for v, c in zip(values, freq)}


class State:
    """A state vector you apply gates to step by step, then measure, sample or take <P> of.

    ``State(n)`` starts in |0...0> (``basis_state=k`` starts in |k>). Gate methods return the
    state, so they chain: ``State(2).h(0).cx(0, 1)``. Qubit 0 is the least significant bit.

    Noise is applied at readout, not per gate: ``p1``/``p2`` are depolarizing error rates per one-
    and two-qubit gate applied so far, giving a global depolarizing channel with fidelity
    ``(1 - p1)^n1 (1 - p2)^n2``; ``readout=(p01, p10)`` flips each measured bit independently.
    """

    def __init__(self, n_qubits: int, basis_state: int = 0):
        self._s = _t1000.State(int(n_qubits), int(basis_state))

    @classmethod
    def from_amplitudes(cls, amplitudes) -> "State":
        """A state from a normalized complex vector of length 2**n."""
        obj = cls.__new__(cls)
        obj._s = _t1000.State.from_amplitudes(np.ascontiguousarray(amplitudes, dtype=np.complex128))
        return obj

    def apply(self, gates: Iterable[Sequence[Any]]) -> "State":
        """Applies gates in the :func:`run_circuit` format."""
        self._s.apply([_normalize_gate(g) for g in gates])
        return self

    def gate(self, name: str, *args: Any) -> "State":
        return self.apply([(name, *args)])

    def h(self, q: int) -> "State":
        return self.gate("H", q)

    def x(self, q: int) -> "State":
        return self.gate("X", q)

    def y(self, q: int) -> "State":
        return self.gate("Y", q)

    def z(self, q: int) -> "State":
        return self.gate("Z", q)

    def s(self, q: int) -> "State":
        return self.gate("S", q)

    def t(self, q: int) -> "State":
        return self.gate("T", q)

    def rx(self, q: int, theta: float) -> "State":
        return self.gate("RX", q, theta)

    def ry(self, q: int, theta: float) -> "State":
        return self.gate("RY", q, theta)

    def rz(self, q: int, theta: float) -> "State":
        return self.gate("RZ", q, theta)

    def cx(self, control: int, target: int) -> "State":
        return self.gate("CX", control, target)

    def cz(self, a: int, b: int) -> "State":
        return self.gate("CZ", a, b)

    def swap(self, a: int, b: int) -> "State":
        return self.gate("SWAP", a, b)

    def cp(self, a: int, b: int, phi: float) -> "State":
        return self.gate("CP", a, b, phi)

    @property
    def n_qubits(self) -> int:
        return self._s.n_qubits

    @property
    def gate_counts(self) -> tuple:
        """(one-qubit gates, two-qubit gates) applied so far."""
        return self._s.gate_counts

    @property
    def amplitudes(self) -> np.ndarray:
        return self._s.amplitudes

    def probabilities(self, p1: float = 0.0, p2: float = 0.0, readout=None) -> np.ndarray:
        return self._s.probabilities(p1, p2, readout)

    def expectation(self, pauli: str, p1: float = 0.0, p2: float = 0.0) -> float:
        """<P> for a Pauli string such as ``"X0 Z1 Y3"`` (``""`` is the identity)."""
        return self._s.expectation(pauli, p1, p2)

    def energy(self, hamiltonian: dict, p1: float = 0.0, p2: float = 0.0) -> float:
        """<H> for a Hamiltonian from :func:`hamiltonian` (or any dict with constant and terms)."""
        terms = [(float(c), str(label)) for c, label in hamiltonian["terms"]]
        return self._s.expectation_sum(terms, float(hamiltonian.get("constant", 0.0)), p1, p2)

    def sample(self, shots: int, seed: int = 0, p1: float = 0.0, p2: float = 0.0, readout=None):
        """``shots`` basis-state indices, reproducible for a given ``seed``."""
        return self._s.sample(int(shots), int(seed), p1, p2, readout)

    def counts(self, shots: int, seed: int = 0, p1: float = 0.0, p2: float = 0.0, readout=None):
        return counts(self.sample(shots, seed, p1, p2, readout), self.n_qubits)

    def copy(self) -> "State":
        obj = State.__new__(State)
        obj._s = self._s.copy()
        return obj

    def __repr__(self) -> str:
        return repr(self._s)


def _molecule_args(formula, atoms):
    if atoms is None:
        return formula, None
    return formula, [(int(z), tuple(float(c) for c in pos)) for z, pos in atoms]


def hamiltonian(
    formula: Optional[str] = None,
    atoms: Optional[Sequence[tuple]] = None,
    charge: int = 0,
    freeze_core: bool = True,
    active_orbitals: Optional[int] = None,
) -> dict:
    """The Jordan-Wigner qubit Hamiltonian of a molecule's active space.

    Returns ``n_qubits``, ``n_electrons``, ``constant``, ``terms`` as ``[(coeff, "X0 Z1 ..."), ...]``,
    ``hf_state`` (the Hartree-Fock basis index) and ``hf_energy``. Qubit ``2p + s`` is spatial
    orbital ``p`` with spin ``s``. Feed it to :meth:`State.energy`.
    """
    f, a = _molecule_args(formula, atoms)
    return json.loads(_t1000.hamiltonian_json(f, a, charge, freeze_core, active_orbitals))


def classical_baselines(
    formula: Optional[str] = None,
    atoms: Optional[Sequence[tuple]] = None,
    charge: int = 0,
    freeze_core: bool = True,
    active_orbitals: Optional[int] = None,
) -> dict:
    """HF, MP2, CISD, CCSD and exact FCI energies (hartree) in the same active space."""
    f, a = _molecule_args(formula, atoms)
    return json.loads(_t1000.classical_json(f, a, charge, freeze_core, active_orbitals))


def adapt_vqe(
    formula: Optional[str] = None,
    atoms: Optional[Sequence[tuple]] = None,
    charge: int = 0,
    freeze_core: bool = True,
    active_orbitals: Optional[int] = None,
    pool: str = "occ-virt",
    gradient_tolerance: float = 1e-3,
    max_operators: int = 250,
) -> dict:
    """ADAPT-VQE against exact FCI.

    ``pool`` is ``"occ-virt"`` (UCCSD singles and doubles), ``"gsd"`` (generalized singles and
    doubles) or ``"qubit-gsd"`` (GSD as qubit excitations, much cheaper circuits). The result
    has the energy, ``error_mha`` versus FCI, the chosen operators and angles, and a CNOT count.
    """
    f, a = _molecule_args(formula, atoms)
    text = _t1000.adapt_json(
        f, a, charge, freeze_core, active_orbitals, pool, gradient_tolerance, max_operators
    )
    return json.loads(text)


def vqe_scan(
    geometries: Sequence[Sequence[tuple]],
    charge: int = 0,
    freeze_core: bool = True,
    active_orbitals: Optional[int] = None,
    follow_scf: bool = True,
    seed_only: bool = False,
) -> list:
    """UCCSD-VQE reports along a list of geometries (each ``[(Z, (x, y, z)), ...]``, angstrom).

    ``follow_scf`` starts each SCF from the previous point's orbitals, which keeps a scan on one
    solution branch through bond stretching; ``seed_only`` keeps that followed branch instead of
    the lowest SCF solution found. A point that fails reports ``{"error": ...}``.
    """
    geoms = [_molecule_args(None, g)[1] for g in geometries]
    return json.loads(
        _t1000.scan_json(geoms, charge, freeze_core, active_orbitals, follow_scf, seed_only)
    )


def diatomic_scan(z_a: int, z_b: int, distances: Iterable[float], **kwargs: Any) -> list:
    """:func:`vqe_scan` of a diatomic (atomic numbers ``z_a``, ``z_b``) along the z axis."""
    geoms = [[(z_a, (0.0, 0.0, 0.0)), (z_b, (0.0, 0.0, float(d)))] for d in distances]
    return vqe_scan(geoms, **kwargs)


def chain(z: int, n: int, spacing: float) -> list:
    """A linear chain of ``n`` atoms of element ``z`` along the z axis (e.g. ``chain(1, 4, 1.0)``)."""
    return [(z, (0.0, 0.0, i * float(spacing))) for i in range(n)]
