# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""Checks of lab_api against closed forms and brute force. Run: python web/test_lab_api.py"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab_api as L  # noqa: E402

passed = 0


def check(name, cond):
    global passed
    assert cond, name
    passed += 1
    print("ok  " + name)


# ---- Grover
g = L.grover(4, [5])
theta = math.asin(math.sqrt(1 / 16))
check("grover n=4: optimal iterations is 3", g["optimal_iterations"] == 3)
check("grover n=4: success equals sin^2(7 theta)", abs(g["success_probability"] - math.sin(7 * theta) ** 2) < 1e-12)
check("grover: simulation matches the closed form at every step", g["max_abs_error_vs_closed_form"] < 1e-12)
check("grover: distribution sums to 1", abs(sum(g["probabilities"]) - 1) < 1e-12)
g3 = L.grover(6, [1, 17, 40])
check("grover with 3 marked items: matches the closed form", g3["max_abs_error_vs_closed_form"] < 1e-12)
over = L.grover(4, [5], iterations=3, max_iterations=9)
# the success probability oscillates with period pi/(2 theta) = 6.2 iterations: peak at 3, trough near 6, next peak near 9
check("grover overshoot: success falls after the optimum and recovers at the next peak",
      over["success_by_iteration"][6] < 0.05 and over["success_by_iteration"][9] > over["success_by_iteration"][6])

# ---- Bernstein-Vazirani, Deutsch-Jozsa
bv = L.bernstein_vazirani(6, 37)
check("bernstein-vazirani recovers the secret with certainty", bv["recovered"] == 37 and abs(bv["success_probability"] - 1) < 1e-12)
check("deutsch-jozsa constant", abs(L.deutsch_jozsa(4, "constant1")["probability_all_zeros"] - 1) < 1e-12)
check("deutsch-jozsa balanced (parity)", L.deutsch_jozsa(4, "balanced_parity")["probability_all_zeros"] < 1e-12)
check("deutsch-jozsa balanced (random)", L.deutsch_jozsa(5, "balanced_random", 7)["verdict"] == "balanced")

# ---- QPE
q = L.qpe(5 / 16, 4)
check("qpe with an exactly representable phase is exact", q["best_outcome"] == 5 and abs(q["probability_of_best"] - 1) < 1e-12)
q = L.qpe(1 / 3, 6)
check("qpe of 1/3: best outcome within 1/2^t, probability >= 4/pi^2", q["abs_error"] <= 1 / 64 and q["probability_of_best"] >= 4 / math.pi ** 2 - 1e-9)

# ---- Shor order finding: FFT construction against brute force
a, big_n, t = 7, 15, 4
dim = 1 << t
state = np.zeros((dim, 16), dtype=np.complex128)
for x in range(dim):
    state[x, pow(a, x, big_n)] = 1 / math.sqrt(dim)
qft_inv = np.array([[np.exp(-2j * np.pi * j * k / dim) for k in range(dim)] for j in range(dim)]) / math.sqrt(dim)
brute = np.sum(np.abs(qft_inv @ state) ** 2, axis=1)
fast = L.shor_order(a, big_n, t, 10)["probabilities"]
check("shor: FFT construction equals the explicit QFT matrix", np.allclose(brute, fast, atol=1e-12))
s = L.shor_order(7, 15, 8, 400, 3)
check("shor 7 mod 15: order 4, peaks at multiples of 64", s["true_order"] == 4 and s["peak_spacing"] == 64.0
      and all(item["index"] % 64 == 0 for item in s["top"][:4]) and abs(sum(i["p"] for i in s["top"][:4]) - 1) < 1e-9)
check("shor 7 mod 15: factors 3 and 5 found", s["factors_found"] == [3, 5] and s["estimated_order"] == 4)
s9 = L.shor_order(2, 9, 0, 300, 1)
check("order of 2 mod 9 is 6 (the 1-2-4-8-7-5 cycle)", s9["true_order"] == 6 and s9["estimated_order"] == 6)
check("shor 2 mod 21: order 6", L.shor_order(2, 21, 0, 300, 2)["estimated_order"] == 6)

# ---- shots
p = np.array([0.5, 0.25, 0.125, 0.125])
r = L.sample_shots(p.tolist(), 1_000_000, 5)
check("shots: empirical distribution within sampling error", r["total_variation_distance"] < 0.005)
check("shots: chi-square statistic is of the order of its degrees of freedom", r["chi_square"] < 25)

check("chi2 survival function against tabulated values",
      all(abs(L.chi2_sf(x, k) - 0.05) < 1e-5 for x, k in ((3.841459, 1), (16.918978, 9), (15.507313, 8), (18.307038, 10))))
rep = L.sample_shots([0.2] * 5, 500, 9, repeats=400)["repeat_p_values"]
check("p-values of repeated null experiments are uniform (mean 0.5, 5% below 0.05)",
      abs(np.mean(rep) - 0.5) < 0.05 and 0.01 < np.mean(np.array(rep) < 0.05) < 0.10)

# ---- random circuits and Porter-Thomas
rc = L.random_circuit(10, 20, 4)
check("random circuit: linear XEB near 1", abs(rc["linear_xeb_self"] - 1) < 0.25)
check("random circuit: KS distance under twice the 5% critical value", rc["ks_statistic"] < 2 * rc["ks_critical_5pct"])
check("random circuit: entropy close to Porter-Thomas value", abs(rc["entropy_nats"] - rc["ideal_entropy_nats"]) < 0.1)

# ---- walks
w = L.walk({"kind": "path", "n": 2}, 0, 1, 3.0, 61)
t_grid = np.array(w["times"])
check("quantum walk on two nodes: return probability is cos^2(t)", np.allclose(w["quantum"]["return_probability"], np.cos(t_grid) ** 2, atol=1e-10))
check("classical walk on two nodes: return probability is (1 + exp(-2t))/2", np.allclose(w["classical"]["return_probability"], (1 + np.exp(-2 * t_grid)) / 2, atol=1e-10))
w = L.walk({"kind": "cycle", "n": 16}, 0, 8, 12.0)
check("walks conserve probability", w["norm_check_quantum"] < 1e-9 and w["norm_check_classical"] < 1e-9)
check("classical walk spreads to uniform; quantum walk does not", w["classical"]["tv_to_uniform"][-1] < w["quantum"]["tv_to_uniform"][-1])
dbl = L.build_graph({"kind": "unit_multiplication", "n": 9, "k": 2})
check("doubling graph mod 9 has 7 edges (a 6-cycle and the pair 3-6)", int(dbl.sum() / 2) == 7)
gt = L.build_graph({"kind": "glued_trees", "depth": 3})
deg = gt.sum(axis=1)
check("glued trees depth 3: 30 vertices, roots have degree 2, all others 3", len(gt) == 30 and sorted(deg)[:2] == [2, 2] and set(deg) == {2.0, 3.0})
check("hypercube d=4: regular of degree 4", set(L.build_graph({"kind": "hypercube", "d": 4}).sum(axis=1)) == {4.0})

# ---- input validation
check("bad input is reported, not raised", "error" in L.handle("/api/lab/algorithm", {"name": "grover", "n": 4, "marked": [99]}))
check("unknown endpoint is reported", "error" in L.handle("/api/lab/nope", {}))

print(f"\n{passed} checks passed")
