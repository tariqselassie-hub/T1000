# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""Backend of the T-1000 lab bench: exact small-scale simulations with numpy.

Every function here computes its result from the stated inputs (exact state vectors, exact matrix
exponentials, exact integer arithmetic) and returns plain JSON-serialisable data. Nothing is stored
or hard-coded. Limits keep each request under a few seconds on a laptop.
"""

import math
from typing import Any, Dict, List

import numpy as np

MAX_QUBITS = 18


class LabError(ValueError):
    """Bad request (the message is shown to the user)."""


# ------------------------------------------------------------------------------------------------
# state-vector helpers
# ------------------------------------------------------------------------------------------------

def _apply_1q(state: np.ndarray, u: np.ndarray, q: int, n: int) -> np.ndarray:
    s = state.reshape(1 << (n - 1 - q), 2, 1 << q)
    return np.einsum("ab,ibj->iaj", u, s).reshape(-1)


H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2.0)


def _hadamard_all(state: np.ndarray, n: int) -> np.ndarray:
    for q in range(n):
        state = _apply_1q(state, H, q, n)
    return state


def _check_n(n: int, lo: int = 1, hi: int = MAX_QUBITS) -> int:
    n = int(n)
    if not lo <= n <= hi:
        raise LabError(f"qubit count must be between {lo} and {hi}")
    return n


# ------------------------------------------------------------------------------------------------
# algorithms
# ------------------------------------------------------------------------------------------------

def grover(n: int, marked: List[int], iterations: int = -1, max_iterations: int = 0) -> Dict[str, Any]:
    """Grover search by explicit oracle and diffusion; reports the success probability after every step."""
    n = _check_n(n, 2, 16)
    big_n = 1 << n
    marked = sorted({int(m) for m in marked})
    if not marked or any(m < 0 or m >= big_n for m in marked):
        raise LabError(f"marked items must be integers in 0..{big_n - 1}")
    if len(marked) >= big_n:
        raise LabError("at least one item must be unmarked")
    m = len(marked)
    theta = math.asin(math.sqrt(m / big_n))
    optimal = max(0, int(math.floor(math.pi / (4 * theta))))
    k = optimal if iterations < 0 else int(iterations)
    steps = max(k, int(max_iterations), 1)
    steps = min(steps, 4 * optimal + 8)
    mark_mask = np.zeros(big_n, dtype=bool)
    mark_mask[marked] = True
    state = np.full(big_n, 1.0 / math.sqrt(big_n), dtype=np.complex128)
    success = [float(np.sum(np.abs(state[mark_mask]) ** 2))]
    for _ in range(steps):
        state[mark_mask] *= -1.0                      # phase oracle
        state = 2.0 * state.mean() - state            # inversion about the mean
        success.append(float(np.sum(np.abs(state[mark_mask]) ** 2)))
    theory = [math.sin((2 * j + 1) * theta) ** 2 for j in range(len(success))]
    # distribution after exactly k iterations
    state = np.full(big_n, 1.0 / math.sqrt(big_n), dtype=np.complex128)
    for _ in range(k):
        state[mark_mask] *= -1.0
        state = 2.0 * state.mean() - state
    probs = np.abs(state) ** 2
    return {
        "algorithm": "grover", "qubits": n, "marked": marked, "optimal_iterations": optimal,
        "iterations_used": k, "success_by_iteration": success, "closed_form": theory,
        "max_abs_error_vs_closed_form": float(max(abs(a - b) for a, b in zip(success, theory))),
        "success_probability": float(probs[mark_mask].sum()),
        "probabilities": probs.tolist() if big_n <= 1024 else None,
        "top": _top(probs),
        "model": "exact state vector: phase oracle and inversion about the mean",
    }


def bernstein_vazirani(n: int, secret: int) -> Dict[str, Any]:
    """H^n, phase oracle (-1)^(s.x), H^n: the input register ends in |s> with certainty."""
    n = _check_n(n, 1, 16)
    s = int(secret)
    if not 0 <= s < (1 << n):
        raise LabError(f"secret must be in 0..{(1 << n) - 1}")
    state = np.zeros(1 << n, dtype=np.complex128)
    state[0] = 1.0
    state = _hadamard_all(state, n)
    x = np.arange(1 << n)
    parity = np.array([bin(int(v) & s).count("1") & 1 for v in x])
    state = state * (1 - 2 * parity)
    state = _hadamard_all(state, n)
    probs = np.abs(state) ** 2
    best = int(np.argmax(probs))
    return {"algorithm": "bernstein_vazirani", "qubits": n, "secret": s, "recovered": best,
            "success_probability": float(probs[s]), "classical_queries_needed": n, "quantum_queries": 1,
            "top": _top(probs), "model": "exact state vector, phase-oracle form of the circuit"}


def deutsch_jozsa(n: int, kind: str, seed: int = 0) -> Dict[str, Any]:
    n = _check_n(n, 1, 16)
    x = np.arange(1 << n)
    if kind == "constant0":
        f = np.zeros(1 << n, dtype=int)
    elif kind == "constant1":
        f = np.ones(1 << n, dtype=int)
    elif kind == "balanced_parity":
        f = np.array([bin(int(v)).count("1") & 1 for v in x])
    elif kind == "balanced_random":
        rng = np.random.default_rng(int(seed))
        f = np.zeros(1 << n, dtype=int)
        f[rng.permutation(1 << n)[: (1 << n) // 2]] = 1
    else:
        raise LabError("kind must be constant0, constant1, balanced_parity or balanced_random")
    state = np.zeros(1 << n, dtype=np.complex128)
    state[0] = 1.0
    state = _hadamard_all(state, n) * (1 - 2 * f)
    state = _hadamard_all(state, n)
    p0 = float(abs(state[0]) ** 2)
    return {"algorithm": "deutsch_jozsa", "qubits": n, "kind": kind, "probability_all_zeros": p0,
            "verdict": "constant" if p0 > 0.5 else "balanced", "top": _top(np.abs(state) ** 2),
            "model": "exact state vector"}


def qpe(phase: float, t: int) -> Dict[str, Any]:
    """Phase estimation of U = diag(1, exp(2 pi i phase)) acting on |1>, with t counting qubits."""
    t = _check_n(t, 1, 16)
    phase = float(phase)
    big_n = 1 << t
    x = np.arange(big_n)
    v = np.exp(2j * np.pi * phase * x) / math.sqrt(big_n)
    amp = np.fft.fft(v) / math.sqrt(big_n)             # inverse QFT on the counting register
    probs = np.abs(amp) ** 2
    k = int(np.argmax(probs))
    return {"algorithm": "qpe", "counting_qubits": t, "phase": phase, "best_outcome": k,
            "estimate": k / big_n, "abs_error": abs(k / big_n - (phase % 1.0)),
            "probability_of_best": float(probs[k]), "probabilities": probs.tolist() if big_n <= 4096 else None,
            "top": _top(probs), "model": "exact: inverse QFT of the phase-kickback state"}


def _order_classical(a: int, n: int) -> int:
    r, v = 1, a % n
    while v != 1:
        v = (v * a) % n
        r += 1
        if r > n:
            return 0
    return r


def shor_order(a: int, big_n: int, t: int = 0, shots: int = 200, seed: int = 0) -> Dict[str, Any]:
    """Quantum order finding for a mod N, simulated exactly.

    The state after the controlled modular exponentiations is sum_x |x>|a^x mod N>; the inverse QFT is an
    FFT over x. This is the output of the ideal circuit, built without gate-by-gate simulation of the
    modular multiplier.
    """
    a, big_n = int(a), int(big_n)
    if not 2 <= big_n <= 256:
        raise LabError("N must be between 2 and 256")
    if math.gcd(a, big_n) != 1 or not 1 < a < big_n:
        raise LabError("a must satisfy 1 < a < N and gcd(a, N) = 1")
    n_work = max(1, (big_n - 1).bit_length())
    t = int(t) if t else min(16, 2 * n_work)
    t = _check_n(t, 2, 16)
    big_t = 1 << t
    xs = np.arange(big_t)
    ys = np.empty(big_t, dtype=np.int64)
    v = 1
    for i in range(big_t):
        ys[i] = v
        v = (v * a) % big_n
    probs = np.zeros(big_t)
    for y in np.unique(ys):
        indicator = (ys == y).astype(np.complex128)
        probs += np.abs(np.fft.fft(indicator)) ** 2
    probs /= float(big_t) ** 2
    probs /= probs.sum()
    rng = np.random.default_rng(int(seed))
    outcomes = rng.choice(big_t, size=int(shots), p=probs)
    r_true = _order_classical(a, big_n)
    found: Dict[int, int] = {}
    factors = set()
    good = 0
    for k in outcomes:
        r = _denominator_candidate(int(k), big_t, a, big_n)
        if r:
            found[r] = found.get(r, 0) + 1
            if r == r_true:
                good += 1
            if r % 2 == 0:
                x = pow(a, r // 2, big_n)
                if x != big_n - 1:
                    for d in (math.gcd(x - 1, big_n), math.gcd(x + 1, big_n)):
                        if 1 < d < big_n:
                            factors.add(d)
    best = max(found.items(), key=lambda kv: kv[1])[0] if found else None
    return {"algorithm": "shor_order", "a": a, "N": big_n, "counting_qubits": t, "work_qubits": n_work,
            "true_order": r_true, "estimated_order": best, "shots": int(shots),
            "shots_giving_true_order": good, "order_candidates": {str(k): v for k, v in sorted(found.items())},
            "factors_found": sorted(factors), "peak_spacing": big_t / r_true if r_true else None,
            "probabilities": probs.tolist() if big_t <= 4096 else None, "top": _top(probs, 24),
            "model": "exact: sum_x |x>|a^x mod N> followed by an inverse QFT (FFT); shots sampled from the exact distribution"}


def _denominator_candidate(k: int, big_t: int, a: int, big_n: int) -> int:
    """Smallest denominator of a continued-fraction convergent of k/2^t that is a valid order of a mod N."""
    if k == 0:
        return 0
    num, den = k, big_t
    p0, p1, q0, q1 = 0, 1, 1, 0
    while den:
        q = num // den
        num, den = den, num - q * den
        p0, p1 = p1, q * p1 + p0
        q0, q1 = q1, q * q1 + q0
        if 1 < q1 <= big_n and pow(a, q1, big_n) == 1:
            return q1
        if q1 > big_n:
            break
    return 0


def _top(probs: np.ndarray, k: int = 16) -> List[Dict[str, Any]]:
    idx = np.argsort(probs)[::-1][:k]
    return [{"index": int(i), "p": float(probs[i])} for i in idx if probs[i] > 1e-12]


# ------------------------------------------------------------------------------------------------
# probability tools
# ------------------------------------------------------------------------------------------------

def chi2_sf(x: float, k: int) -> float:
    """Survival function of the chi-square distribution with k degrees of freedom (regularised upper gamma)."""
    a, x = k / 2.0, x / 2.0
    if x <= 0:
        return 1.0
    if x < a + 1:
        ap, total = a, 1.0 / a
        delta = total
        for _ in range(500):
            ap += 1
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * 1e-15:
                break
        return 1.0 - total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b, c = x + 1 - a, 1e300
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = d if abs(d) > 1e-300 else 1e-300
        c = b + an / c
        c = c if abs(c) > 1e-300 else 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def sample_shots(probs: List[float], shots: int, seed: int = 0, repeats: int = 0) -> Dict[str, Any]:
    p = np.asarray(probs, dtype=float)
    if p.ndim != 1 or len(p) < 2 or np.any(p < -1e-12):
        raise LabError("probabilities must be a non-negative list")
    p = np.clip(p, 0, None)
    p /= p.sum()
    shots = int(shots)
    if not 1 <= shots <= 50_000_000:
        raise LabError("shots must be between 1 and 50,000,000")
    rng = np.random.default_rng(int(seed))
    expected = shots * p
    mask = expected > 0
    dof = int(mask.sum() - 1)

    def one() -> tuple:
        counts = rng.multinomial(shots, p)
        stat = float(np.sum((counts[mask] - expected[mask]) ** 2 / expected[mask]))
        return counts, stat

    counts, stat = one()
    out = {"shots": shots, "counts": counts.tolist() if len(p) <= 4096 else None,
           "top": [{"index": int(i), "count": int(counts[i])} for i in np.argsort(counts)[::-1][:16] if counts[i]],
           "chi_square": stat, "degrees_of_freedom": dof, "p_value": chi2_sf(stat, dof) if dof > 0 else None,
           "total_variation_distance": float(0.5 * np.abs(counts / shots - p).sum())}
    repeats = int(repeats)
    if repeats > 0:
        if repeats > 2000:
            raise LabError("repeats must be at most 2000")
        out["repeat_p_values"] = [chi2_sf(one()[1], dof) for _ in range(repeats)] if dof > 0 else []
    return out


def _haar_1q(rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def random_circuit(n: int, depth: int, seed: int = 0, bins: int = 30) -> Dict[str, Any]:
    """Haar-random single-qubit layers plus a brickwork of CZ; Porter-Thomas statistics of the output."""
    n = _check_n(n, 2, MAX_QUBITS)
    depth = int(depth)
    if not 1 <= depth <= 60:
        raise LabError("depth must be between 1 and 60")
    rng = np.random.default_rng(int(seed))
    big_n = 1 << n
    state = np.zeros(big_n, dtype=np.complex128)
    state[0] = 1.0
    idx = np.arange(big_n)
    for layer in range(depth):
        for q in range(n):
            state = _apply_1q(state, _haar_1q(rng), q, n)
        for q in range(layer % 2, n - 1, 2):
            both = ((idx >> q) & 1) & ((idx >> (q + 1)) & 1)
            state = state * (1 - 2 * both)
    p = np.abs(state) ** 2
    x = np.sort(big_n * p)
    cdf_model = 1 - np.exp(-x)
    ks = float(np.max(np.abs(np.arange(1, big_n + 1) / big_n - cdf_model)))
    hist, edges = np.histogram(big_n * p, bins=bins, range=(0, 6), density=False)
    return {"qubits": n, "depth": depth, "seed": int(seed),
            "linear_xeb_self": float(big_n * np.sum(p ** 2) - 1.0),
            "entropy_nats": float(-np.sum(p[p > 0] * np.log(p[p > 0]))),
            "ideal_entropy_nats": float(math.log(big_n) - 1 + 0.5772156649),
            "ks_statistic": ks, "ks_critical_5pct": 1.358 / math.sqrt(big_n),
            "histogram": hist.tolist(), "bin_edges": edges.tolist(),
            "exponential_expected": [float(big_n * (math.exp(-edges[i]) - math.exp(-edges[i + 1]))) for i in range(bins)],
            "model": "exact state vector; Porter-Thomas law is p(N*prob) = exp(-N*prob)"}


# ------------------------------------------------------------------------------------------------
# graphs and walks
# ------------------------------------------------------------------------------------------------

def build_graph(spec: Dict[str, Any]) -> np.ndarray:
    kind = spec.get("kind", "cycle")
    n = int(spec.get("n", 8))
    if kind in ("cycle", "path", "complete", "star", "circulant", "unit_multiplication") and not 2 <= n <= 256:
        raise LabError("n must be between 2 and 256")
    a = np.zeros((n, n))
    if kind == "cycle":
        for i in range(n):
            a[i, (i + 1) % n] = a[(i + 1) % n, i] = 1
    elif kind == "path":
        for i in range(n - 1):
            a[i, i + 1] = a[i + 1, i] = 1
    elif kind == "complete":
        a = np.ones((n, n)) - np.eye(n)
    elif kind == "star":
        a[0, 1:] = a[1:, 0] = 1
    elif kind == "hypercube":
        d = int(spec.get("d", 3))
        if not 1 <= d <= 8:
            raise LabError("hypercube dimension must be between 1 and 8")
        n = 1 << d
        a = np.zeros((n, n))
        for i in range(n):
            for b in range(d):
                a[i, i ^ (1 << b)] = 1
    elif kind == "grid":
        rows, cols = int(spec.get("rows", 4)), int(spec.get("cols", 4))
        if not (1 <= rows <= 16 and 1 <= cols <= 16):
            raise LabError("grid sides must be between 1 and 16")
        n = rows * cols
        a = np.zeros((n, n))
        for r in range(rows):
            for c in range(cols):
                i = r * cols + c
                if c + 1 < cols:
                    a[i, i + 1] = a[i + 1, i] = 1
                if r + 1 < rows:
                    a[i, i + cols] = a[i + cols, i] = 1
    elif kind == "circulant":
        gens = [int(g) for g in spec.get("generators", [1])]
        for i in range(n):
            for g in gens:
                j = (i + g) % n
                if j != i:
                    a[i, j] = a[j, i] = 1
    elif kind == "unit_multiplication":
        k = int(spec.get("k", 2))
        off = int(spec.get("offset", 0))
        for x in range(n):
            y = (k * x + off) % n
            if y != x:
                a[x, y] = a[y, x] = 1
    elif kind == "glued_trees":
        depth = int(spec.get("depth", 3))
        if not 1 <= depth <= 5:
            raise LabError("glued-tree depth must be between 1 and 5")
        return _glued_trees(depth, int(spec.get("seed", 0)))
    else:
        raise LabError("unknown graph kind " + str(kind))
    return a


def _glued_trees(depth: int, seed: int) -> np.ndarray:
    tree = (1 << (depth + 1)) - 1
    n = 2 * tree
    a = np.zeros((n, n))
    for base in (0, tree):
        for i in range(tree):
            for c in (2 * i + 1, 2 * i + 2):
                if c < tree:
                    a[base + i, base + c] = a[base + c, base + i] = 1
    leaves1 = list(range(tree - (1 << depth), tree))
    leaves2 = [tree + v for v in leaves1]
    rng = np.random.default_rng(seed)
    for perm in (rng.permutation(len(leaves2)), rng.permutation(len(leaves2))):
        for i, j in enumerate(perm):
            a[leaves1[i], leaves2[j]] = a[leaves2[j], leaves1[i]] = 1
    return a


def walk(graph: Dict[str, Any], start: int, target: int, t_max: float, steps: int = 120) -> Dict[str, Any]:
    """Continuous-time quantum walk exp(-iAt), classical continuous-time walk exp(-Lt), and the discrete walk."""
    a = build_graph(graph)
    n = len(a)
    start, target = int(start) % n, int(target) % n
    steps = max(2, min(int(steps), 400))
    times = np.linspace(0.0, float(t_max), steps)
    deg = a.sum(axis=1)
    lam, vec = np.linalg.eigh(a)
    e = np.zeros(n)
    e[start] = 1.0
    coeff = vec.T @ e
    quantum = np.empty((steps, n))
    for i, t in enumerate(times):
        psi = vec @ (np.exp(-1j * lam * t) * coeff)
        quantum[i] = np.abs(psi) ** 2
    lap = np.diag(deg) - a
    lam_l, vec_l = np.linalg.eigh(lap)
    coeff_l = vec_l.T @ e
    classical = np.empty((steps, n))
    for i, t in enumerate(times):
        classical[i] = np.clip(vec_l @ (np.exp(-lam_l * t) * coeff_l), 0, None)
    uniform = np.full(n, 1.0 / n)

    def stats(m: np.ndarray) -> Dict[str, Any]:
        tv = 0.5 * np.abs(m - uniform).sum(axis=1)
        ent = -np.sum(np.where(m > 1e-15, m * np.log(np.where(m > 1e-15, m, 1)), 0.0), axis=1)
        return {"return_probability": m[:, start].tolist(), "target_probability": m[:, target].tolist(),
                "tv_to_uniform": tv.tolist(), "entropy_nats": ent.tolist(),
                "peak_target": float(m[:, target].max()), "time_of_peak": float(times[int(np.argmax(m[:, target]))])}

    return {"nodes": n, "edges": int(a.sum() / 2), "start": start, "target": target, "times": times.tolist(),
            "adjacency_eigenvalues": lam.tolist() if n <= 64 else None,
            "quantum": stats(quantum), "classical": stats(classical),
            "heat_quantum": quantum.tolist() if n * steps <= 60000 else None,
            "heat_classical": classical.tolist() if n * steps <= 60000 else None,
            "norm_check_quantum": float(np.abs(quantum.sum(axis=1) - 1).max()),
            "norm_check_classical": float(np.abs(classical.sum(axis=1) - 1).max()),
            "model": "exact spectral evolution: exp(-iAt) and exp(-(D-A)t) by eigendecomposition"}


# ------------------------------------------------------------------------------------------------
# dispatcher
# ------------------------------------------------------------------------------------------------

def handle(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if path == "/api/lab/algorithm":
            name = payload.get("name", "grover")
            if name == "grover":
                return grover(payload.get("n", 4), payload.get("marked", [3]), payload.get("iterations", -1),
                              payload.get("max_iterations", 0))
            if name == "bernstein_vazirani":
                return bernstein_vazirani(payload.get("n", 5), payload.get("secret", 11))
            if name == "deutsch_jozsa":
                return deutsch_jozsa(payload.get("n", 4), payload.get("kind", "balanced_parity"), payload.get("seed", 0))
            if name == "qpe":
                return qpe(payload.get("phase", 0.3125), payload.get("t", 6))
            if name == "shor_order":
                return shor_order(payload.get("a", 2), payload.get("N", 15), payload.get("t", 0),
                                  payload.get("shots", 200), payload.get("seed", 0))
            raise LabError("unknown algorithm " + str(name))
        if path == "/api/lab/shots":
            return sample_shots(payload.get("probabilities", [0.5, 0.5]), payload.get("shots", 1000), payload.get("seed", 0), payload.get("repeats", 0))
        if path == "/api/lab/random_circuit":
            return random_circuit(payload.get("n", 10), payload.get("depth", 12), payload.get("seed", 0))
        if path == "/api/lab/walk":
            return walk(payload.get("graph", {"kind": "cycle", "n": 16}), payload.get("start", 0),
                        payload.get("target", 8), payload.get("t_max", 12.0), payload.get("steps", 120))
        raise LabError("unknown endpoint " + path)
    except LabError as e:
        return {"error": str(e)}
