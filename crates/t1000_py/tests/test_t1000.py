# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.
"""Smoke and correctness tests for the t1000 wheel. Run with: python -m pytest crates/t1000_py/tests"""

import json
import math
from pathlib import Path

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


# ── Stateful simulator ──────────────────────────────────────────────────────────────────────


def test_bell_pauli_expectations():
    s = t1000.State(2).h(0).cx(0, 1)
    assert s.gate_counts == (1, 1)
    assert abs(s.expectation("Z0 Z1") - 1) < 1e-12
    assert abs(s.expectation("X0 X1") - 1) < 1e-12
    assert abs(s.expectation("Y0 Y1") + 1) < 1e-12
    assert abs(s.expectation("Z0")) < 1e-12 and abs(s.expectation("") - 1) < 1e-12


def test_y_phase_convention():
    # RX(-pi/2)|0> = (|0> + i|1>)/sqrt(2), the +1 eigenstate of Y.
    s = t1000.State(1).rx(0, -math.pi / 2)
    assert abs(s.expectation("Y0") - 1) < 1e-12


def test_pauli_label_errors():
    s = t1000.State(2)
    for bad in ["Z2", "Z0 X0", "Q1"]:
        with pytest.raises(ValueError):
            s.expectation(bad)


def test_sampling_is_reproducible_and_matches_probabilities():
    s = t1000.State(2).h(0).cx(0, 1)
    a, b = s.sample(4000, seed=7), s.sample(4000, seed=7)
    assert np.array_equal(a, b) and set(np.unique(a)) <= {0, 3}
    c = s.counts(4000, seed=7)
    assert set(c) <= {"00", "11"} and abs(c["00"] / 4000 - 0.5) < 0.03


def test_noise_scales_expectations_and_spreads_probabilities():
    s = t1000.State(2).h(0).cx(0, 1)
    f = (1 - 0.01) * (1 - 0.05)
    assert abs(s.expectation("Z0 Z1", p1=0.01, p2=0.05) - f) < 1e-12
    p = s.probabilities(p1=0.01, p2=0.05)
    assert abs(p.sum() - 1) < 1e-12 and p[1] > 0 and abs(p[1] - (1 - f) / 4) < 1e-12
    r = t1000.State(1).probabilities(readout=(0.1, 0.0))
    assert np.allclose(r, [0.9, 0.1], atol=1e-12)


def test_from_amplitudes_round_trip():
    v = t1000.run_circuit(3, [("H", 0), ("CX", 0, 2), ("RY", 1, 0.4)])
    s = t1000.State.from_amplitudes(v)
    assert np.allclose(s.amplitudes, v) and s.copy().n_qubits == 3
    with pytest.raises(ValueError):
        t1000.State.from_amplitudes(np.array([1.0, 1.0]))


# ── Chemistry ───────────────────────────────────────────────────────────────────────────────


def test_hamiltonian_hf_state_energy_is_hartree_fock():
    h = t1000.hamiltonian("LiH")
    s = t1000.State(h["n_qubits"], basis_state=h["hf_state"])
    assert abs(s.energy(h) - h["hf_energy"]) < 1e-9


def test_classical_baselines_on_h2_are_exact():
    c = t1000.classical_baselines("H2")
    # Two electrons: CISD and CCSD are exact.
    assert abs(c["cisd"] - c["fci"]) < 1e-9 and abs(c["ccsd"]["energy"] - c["fci"]) < 1e-9
    assert c["hf"] > c["mp2"] > c["fci"]


def _recorded(name):
    root = Path(__file__).resolve().parents[3] / "results" / "hosts" / "unknown-cpu" / "data"
    path = root / name
    if not path.exists():
        pytest.skip(f"{name} not recorded on this checkout")
    return json.loads(path.read_text(encoding="utf-8"))


def test_h4_chain_matches_the_recorded_strong_correlation_report():
    point = _recorded("vqe_strong_correlation_report.json")["H4"]["points"][0]
    atoms = t1000.chain(1, 4, point["coordinate"])
    c = t1000.classical_baselines(atoms=atoms)
    assert abs(c["fci"] - point["fci"]) < 1e-8
    assert abs(c["ccsd"]["energy"] - point["ccsd"]["energy"]) < 1e-8
    a = t1000.adapt_vqe(atoms=atoms)
    assert abs(a["energy"] - point["adapt"]["energy"]) < 1e-8 and a["cnot"] == point["adapt"]["cnot"]


def test_adapt_pools_match_the_recorded_h6_point():
    point = _recorded("vqe_adapt_pools_report.json")["H6"]["points"][0]
    atoms = t1000.chain(1, 6, point["coordinate"])
    runs = {r["variant"]: r for r in point["runs"]}
    for pool, variant in [("occ-virt", "occ-virt"), ("qubit-gsd", "qubit GSD")]:
        a = t1000.adapt_vqe(atoms=atoms, pool=pool)
        assert abs(a["energy"] - runs[variant]["energy"]) < 1e-8, pool
        assert a["cnot"] == runs[variant]["cnot"] and a["parameters"] == runs[variant]["operators"]


def test_diatomic_scan_follows_the_bond():
    pts = t1000.diatomic_scan(1, 1, [0.7, 1.5, 2.5])
    assert len(pts) == 3 and all("error" not in p for p in pts)
    assert all(p["vqe_error_vs_fci"] < 1e-6 for p in pts)
