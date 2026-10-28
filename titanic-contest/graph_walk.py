# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

r"""Can a walk on the passenger graph organise the data? Classical spreading against a quantum walk.

    .venv\Scripts\python.exe titanic\graph_walk.py

Nodes are all 1,309 passengers (train and test; test labels do not exist, so nothing leaks).
Edges join passengers on the same ticket (weight 1), the same cabin (1) and the same surname (0.5);
a second graph also joins each passenger to their 10 nearest neighbours in (sex, class, age, fare,
family size). With S = D^-1/2 A D^-1/2 (eigenvalues in [-1, 1]) and a signal s holding the centred
training labels (0 on unlabelled nodes), three spreading filters are applied:

  heat       exp(-t (I - S)) s            classical diffusion
  resolvent  (1 - a)(I - a S)^-1 s        classical label spreading
  quantum    exp(-i t S) s                continuous-time quantum walk; Re, Im and |amplitude|^2

For a training passenger the passenger's own label is removed exactly (the filters are linear, so
f_i minus K_ii s_i), so the feature never sees its own answer. The spreading time or strength is
chosen inside each training fold by correlation of that leave-one-out score with the labels. The
scores are then added to the best feature set (group survival + role-split group survival) and
cross-validated on the same folds as the other experiments.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402
from analysis import FOLDS, REPEATS, group_features  # noqa: E402
from relations import role_columns, role_group_features  # noqa: E402

warnings.filterwarnings("ignore")

TIMES = [0.5, 1.0, 2.0, 4.0, 8.0]
ALPHAS = [0.5, 0.8, 0.95]


def pair_adjacency(keys, weight, n):
    """Weight `weight` between all distinct pairs sharing a key (NaN keys excluded)."""
    a = np.zeros((n, n))
    for members in pd.Series(np.arange(n)).groupby(keys.values).groups.values():
        m = np.asarray(list(members))
        if len(m) > 1:
            a[np.ix_(m, m)] = weight
    np.fill_diagonal(a, 0.0)
    return a


def build_graphs(raw, feat):
    n = len(raw)
    cabin = raw["Cabin"].where(raw["Cabin"].notna(), None)
    a_rel = np.zeros((n, n))
    a_rel = np.maximum(a_rel, pair_adjacency(feat["ticket"], 1.0, n))
    has_cabin = cabin.notna()
    a_cab = np.zeros((n, n))
    idx = np.where(has_cabin)[0]
    a_cab[np.ix_(idx, idx)] = pair_adjacency(cabin[has_cabin].reset_index(drop=True), 1.0, len(idx))
    a_rel = np.maximum(a_rel, a_cab)
    a_rel = np.maximum(a_rel, pair_adjacency(feat["surname"], 0.5, n))

    cols = np.column_stack([feat["sex"] * 3.0, feat["pclass"], feat["age"] / 20.0, feat["log_fare"], feat["family"] / 2.0])
    z = (cols - cols.mean(0)) / cols.std(0)
    z[:, 0] *= 2.0
    nn = NearestNeighbors(n_neighbors=11).fit(z)
    _, nb = nn.kneighbors(z)
    a_knn = np.zeros((n, n))
    for i in range(n):
        a_knn[i, nb[i, 1:]] = 1.0
    a_knn = np.maximum(a_knn, a_knn.T)
    a_both = a_rel + 0.3 * a_knn
    return {"relationship graph": a_rel, "relationship + similarity graph": a_both}


def spectrum(a):
    d = a.sum(1)
    inv = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-12)), 0.0)
    s = a * inv[:, None] * inv[None, :]
    lam, v = np.linalg.eigh(s)
    return lam, v, (d > 0)


def filters(lam):
    out = {}
    for t in TIMES:
        out[("heat", t)] = np.exp(-t * (1.0 - lam))
    for al in ALPHAS:
        out[("resolvent", al)] = (1.0 - al) / (1.0 - al * lam)
    for t in TIMES:
        out[("quantum", t)] = np.exp(-1j * t * lam)
    return out


def apply(v, g, s_vec, diag_vec=None):
    """f = V g V^T s and the diagonal K_ii of V g V^T."""
    f = v @ (g * (v.T @ s_vec))
    kii = (v * v) @ g
    return f, kii


def graph_scores(spec, labelled_idx, y_lab, train_idx, eval_idx, prior):
    """Chooses one parameter per filter family inside the training fold; returns feature columns."""
    lam, v, connected = spec
    n = len(lam)
    s = np.zeros(n)
    s[labelled_idx] = y_lab - prior
    flt = filters(lam)
    cache = {}
    for key, g in flt.items():
        f, kii = apply(v, g, s)
        cache[key] = (f, kii)

    def loo_value(key, kind):
        f, kii = cache[key]
        loo = f[train_idx] - kii[train_idx] * s[train_idx]
        return {"re": loo.real, "im": loo.imag, "pr": np.abs(loo) ** 2 - 0.0}[kind] if np.iscomplexobj(loo) else loo

    def best(family, kind="re"):
        ok = connected[train_idx]
        yy = (s[train_idx] + prior)[ok]
        scores = {}
        for key in flt:
            if key[0] != family:
                continue
            val = loo_value(key, kind)[ok]
            scores[key] = abs(stats.pearsonr(val, yy)[0]) if np.std(val) > 1e-12 else 0.0
        return max(scores, key=scores.get)

    cols_t, cols_e, names = {}, {}, {}
    for family in ("heat", "resolvent", "quantum"):
        key = best(family)
        f, kii = cache[key]
        f_t = f[train_idx] - kii[train_idx] * s[train_idx]
        f_e = f[eval_idx]
        if family == "quantum":
            cols_t["quantum"] = np.column_stack([f_t.real, f_t.imag, np.abs(f_t) ** 2])
            cols_e["quantum"] = np.column_stack([f_e.real, f_e.imag, np.abs(f_e) ** 2])
            cols_t["quantum_re"] = f_t.real[:, None]
            cols_e["quantum_re"] = f_e.real[:, None]
        else:
            cols_t[family] = f_t.real[:, None]
            cols_e[family] = f_e.real[:, None]
        names[family] = key
    return cols_t, cols_e, names


def main():
    raw, n_train = features.load()
    feat = features.build(raw)
    base_all = features.numeric_matrix(feat)
    base = base_all.iloc[:n_train]
    _, woman_child = role_columns(raw, feat)
    role = woman_child.iloc[:n_train]
    y = raw["Survived"].iloc[:n_train].astype(int)
    keys = [feat["ticket"].iloc[:n_train], feat["surname"].iloc[:n_train]]
    graphs = {k: spectrum(a) for k, a in build_graphs(raw, feat).items()}
    pos = {ix: i for i, ix in enumerate(raw.index)}

    splitter = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=2026)
    makers = {
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=0),
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
    }
    variants = ["heat", "resolvent", "quantum_re", "quantum", "all three"]
    rows = []
    chosen = {}
    for fold, (tr, ev) in enumerate(splitter.split(base, y)):
        tr_idx, ev_idx = base.index[tr], base.index[ev]
        prior = float(y.loc[tr_idx].mean())
        gt = [group_features(k, y, tr_idx, ev_idx, prior) for k in keys]
        rg_t, rg_e = role_group_features(keys, y, role, tr_idx, ev_idx, prior)
        xt0 = np.hstack([base.loc[tr_idx].values] + [g[0] for g in gt] + [rg_t])
        xe0 = np.hstack([base.loc[ev_idx].values] + [g[1] for g in gt] + [rg_e])
        ytr, yev = y.loc[tr_idx].values, y.loc[ev_idx].values
        tr_pos = np.array([pos[i] for i in tr_idx])
        ev_pos = np.array([pos[i] for i in ev_idx])
        per_graph = {}
        for gname, spec in graphs.items():
            ct, ce, names = graph_scores(spec, tr_pos, ytr, tr_pos, ev_pos, prior)
            per_graph[gname] = (ct, ce)
            for fam, key in names.items():
                chosen.setdefault((gname, fam), []).append(key[1])
        for mname, make in makers.items():
            base_acc = float((make().fit(xt0, ytr).predict(xe0) == yev).mean())
            row = {"fold": fold, "model": mname, "graph": "none", "variant": "base", "acc": base_acc}
            rows.append(row)
            for gname, (ct, ce) in per_graph.items():
                for var in variants:
                    fams = ["heat", "resolvent", "quantum"] if var == "all three" else [var]
                    add_t = np.hstack([ct[f] for f in fams])
                    add_e = np.hstack([ce[f] for f in fams])
                    m = make().fit(np.hstack([xt0, add_t]), ytr)
                    acc = float((m.predict(np.hstack([xe0, add_e])) == yev).mean())
                    rows.append({"fold": fold, "model": mname, "graph": gname, "variant": var, "acc": acc})

    df = pd.DataFrame(rows)
    print(f" {FOLDS}-fold x {REPEATS} repeats; base = group survival + role-split group survival")
    for mname in makers:
        d = df[df.model == mname]
        base_acc = d[d.variant == "base"].set_index("fold")["acc"]
        print(f"\n {mname}: base {base_acc.mean():.4f}")
        print(f" {'graph':<34} {'walk':<12} {'accuracy':>9} {'gain':>8} {'se':>7}")
        for gname in graphs:
            for var in variants:
                a = d[(d.graph == gname) & (d.variant == var)].set_index("fold")["acc"]
                diff = (a - base_acc).values
                se = diff.reshape(REPEATS, FOLDS).mean(axis=1).std(ddof=1) / np.sqrt(REPEATS)
                print(f" {gname:<34} {var:<12} {a.mean():9.4f} {diff.mean():+8.4f} {se:7.4f}")
    print("\n parameter chosen inside the training folds (median over folds):")
    for (gname, fam), vals in chosen.items():
        print(f"  {gname:<34} {fam:<10} {np.median(vals)}")


if __name__ == "__main__":
    main()
