# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.
"""Smoke and correctness tests for the t1000 wheel. Run with: python -m pytest crates/t1000_py/tests"""

import math

import numpy as np
import pytest

import t1000


def test_version_and_catalog():
    assert t1000.__version__.count(".") == 2
    catalog = t1000.vqe_catalog()
    assert "H2" in catalog and "LiH" in catalog and len(catalog) == 9


def test_x_on_qubit_zero_sets_the_lowest_bit():
    state = t1000.run_circuit(3, [("X", 0)])
    assert state.dtype == np.complex128 and state.shape == (8,)
    assert abs(state[1] - 1) < 1e-12 and np.allclose(np.delete(state, 1), 0)


def test_bell_state():
    state = t1000.run_circuit(2, [("H", 0), ("CX", 0, 1)])
    expected = np.array([1, 0, 0, 1]) / math.sqrt(2)
    assert np.allclose(state, expected, atol=1e-12)


def test_rotation_probabilities_follow_the_closed_form():
    theta = 0.7
    p = t1000.probabilities(1, [("RY", 0, theta)])
    assert np.allclose(p, [math.cos(theta / 2) ** 2, math.sin(theta / 2) ** 2], atol=1e-12)
    # RX and RY give the same probabilities from |0>; S, T and RZ change only phases.
    assert np.allclose(t1000.probabilities(1, [("RX", 0, theta)]), p, atol=1e-12)
    q = t1000.probabilities(2, [("H", 0), ("H", 1), ("S", 0), ("T", 1), ("RZ", 0, 0.3), ("CZ", 0, 1)])
    assert np.allclose(q, 0.25, atol=1e-12)


def test_swap_and_controlled_phase():
    state = t1000.run_circuit(2, [("X", 0), ("SWAP", 0, 1)])
    assert abs(state[2] - 1) < 1e-12
    state = t1000.run_circuit(2, [("X", 0), ("X", 1), ("CP", 0, 1, math.pi / 2)])
    assert abs(state[3] - 1j) < 1e-12


@pytest.mark.parametrize(
    "gates",
    [[("Q", 0)], [("H", 5)], [("CX", 0, 0)], [("RY", 0)], [("H", 0, 1)], [("CP", 0, 1)]],
)
def test_bad_gates_are_rejected(gates):
    with pytest.raises(ValueError):
        t1000.run_circuit(2, gates)


def _separable(n=60, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    x0 = np.where(y == 0, 0.4, 2.7) + rng.normal(0, 0.15, n)
    X = np.column_stack([x0, rng.uniform(0, math.pi, n)])
    return X, y


def test_classifier_learns_a_separable_rule_with_any_two_labels():
    X, y = _separable()
    clf = t1000.VqcClassifier(depth=2, epochs=60, lr=0.1, seed=3).fit(X, y)
    assert clf.score(X, y) >= 0.9
    assert set(np.unique(clf.predict(X))) <= {0, 1}
    names = np.where(y == 1, "yes", "no")
    clf2 = t1000.VqcClassifier(depth=2, epochs=60, lr=0.1, seed=3).fit(X, names)
    assert np.array_equal(clf2.decision_function(X), clf.decision_function(X))
    assert clf2.predict(X[:2]).dtype.kind == "U"


def test_classifier_is_deterministic_and_validates_input():
    X, y = _separable(20)
    a = t1000.VqcClassifier(epochs=5).fit(X, y)
    b = t1000.VqcClassifier(epochs=5).fit(X, y)
    assert np.array_equal(a.weights_, b.weights_)
    with pytest.raises(ValueError):
        a.decision_function(X[:, :1])
    with pytest.raises(ValueError):
        t1000.VqcClassifier().fit(X, np.zeros(len(X)))
    with pytest.raises(RuntimeError):
        t1000.VqcClassifier().predict(X)
    assert t1000.VqcClassifier().set_params(depth=4).get_params()["depth"] == 4


def test_vqe_h2_matches_the_published_sto3g_fci_energy():
    report = t1000.vqe_solve("H2")
    assert abs(report["fci_energy"] + 1.13727) < 2e-5
    assert abs(report["vqe_error_vs_fci"]) < 1e-8
    assert report["n_qubits"] == 4
    explicit = t1000.vqe_solve(atoms=[(1, (0.0, 0.0, 0.0)), (1, (0.0, 0.0, 0.7414))])
    assert abs(explicit["fci_energy"] - report["fci_energy"]) < 1e-6


def test_vqe_rejects_bad_requests():
    with pytest.raises(ValueError):
        t1000.vqe_solve("Unobtainium")
    with pytest.raises(ValueError):
        t1000.vqe_solve()
