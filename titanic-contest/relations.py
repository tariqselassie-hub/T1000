# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

r"""Name and relationship structure: roles inside families and how each role fared.

    .venv\Scripts\python.exe titanic\relations.py

Label-free columns: detailed title, role (woman/child or man), the mix of roles on the same ticket,
likely mother / spouse / child-with-parent. Label-aware columns (inside each fold, training labels
only): survival of the other women-and-children and of the other men in the passenger's ticket and
surname groups. Same folds as improve.py; prints the gain over the base columns.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402
from analysis import FOLDS, REPEATS, group_features  # noqa: E402

warnings.filterwarnings("ignore")


def role_columns(raw, feat):
    """Label-free relationship columns."""
    c = pd.DataFrame(index=raw.index)
    title = raw["Name"].str.extract(r",\s*([^\.]+)\.")[0].str.strip()
    for t in ("Rev", "Dr", "Col", "Mlle", "Major"):
        c[f"title_{t}"] = (title == t).astype(float)
    woman_child = ((raw["Sex"] == "female") | (feat["age"] <= 14) | (feat["title"] == "Master")).astype(int)
    c["woman_child"] = woman_child.astype(float)
    c["is_man"] = 1.0 - c["woman_child"]
    g = raw["Ticket"]
    size = feat["ticket_size"]
    c["ticket_women_children_others"] = woman_child.groupby(g).transform("sum") - woman_child
    c["ticket_men_others"] = (size - woman_child.groupby(g).transform("sum")) - (1 - woman_child)
    c["mother"] = ((raw["Sex"] == "female") & (raw["Parch"] > 0) & (feat["age"] > 18) & (title.isin(["Mrs", "Dr"]))).astype(float)
    c["child_with_parent"] = ((feat["age"] <= 18) & (raw["Parch"] > 0)).astype(float)
    sur = feat["surname"]
    mrs = ((raw["Sex"] == "female") & (title == "Mrs")).astype(int)
    mr = ((raw["Sex"] == "male") & (feat["age"] > 18)).astype(int)
    c["couple_on_ticket"] = ((mrs.groupby([g, sur]).transform("sum") > 0) & (mr.groupby([g, sur]).transform("sum") > 0)).astype(float)
    c["maiden_name"] = raw["Name"].str.contains(r"\(").astype(float)
    return c, woman_child


def role_group_features(keys, y, role, train_idx, eval_idx, prior):
    """Survival of the other women/children and of the other men in each group (training labels)."""
    out_t, out_e = [], []
    for want in (1, 0):
        mask = (role.reindex(y.index) == want)
        y_masked = y.where(mask)  # NaN where the member is not in this role
        for k in keys:
            lab = pd.DataFrame({"k": k.values, "y": y_masked.values}, index=k.index).loc[train_idx].dropna()
            s, n = lab.groupby("k")["y"].sum(), lab.groupby("k")["y"].count()

            def stats(idx, self_in_train):
                kk = k.loc[idx]
                ss, nn = kk.map(s).fillna(0.0).values, kk.map(n).fillna(0.0).values
                if self_in_train:  # remove the passenger's own label if they are in this role
                    own = y_masked.loc[idx].values
                    has = ~np.isnan(own)
                    ss = ss - np.where(has, own, 0.0)
                    nn = nn - has
                return np.column_stack([nn, np.where(nn > 0, ss / np.maximum(nn, 1), prior)])

            out_t.append(stats(train_idx, True))
            out_e.append(stats(eval_idx, False))
    return np.hstack(out_t), np.hstack(out_e)


def run(x, y, keys, role, extra, make, splitter, use_role_groups):
    acc = []
    for tr, ev in splitter.split(x, y):
        tr_idx, ev_idx = x.index[tr], x.index[ev]
        prior = float(y.loc[tr_idx].mean())
        gt = [group_features(k, y, tr_idx, ev_idx, prior) for k in keys]
        parts_t = [x.loc[tr_idx].values] + [g[0] for g in gt]
        parts_e = [x.loc[ev_idx].values] + [g[1] for g in gt]
        if extra is not None:
            parts_t.append(extra.loc[tr_idx].values)
            parts_e.append(extra.loc[ev_idx].values)
        if use_role_groups:
            rt, re = role_group_features(keys, y, role, tr_idx, ev_idx, prior)
            parts_t.append(rt)
            parts_e.append(re)
        m = make().fit(np.hstack(parts_t), y.loc[tr_idx].values)
        acc.append(float((m.predict(np.hstack(parts_e)) == y.loc[ev_idx].values).mean()))
    return np.array(acc)


def main():
    raw, n_train = features.load()
    feat = features.build(raw)
    base = features.numeric_matrix(feat).iloc[:n_train]
    roles_df, woman_child = role_columns(raw, feat)
    roles_df = roles_df.iloc[:n_train]
    role = woman_child.iloc[:n_train]
    y = raw["Survived"].iloc[:n_train].astype(int)
    keys = [feat["ticket"].iloc[:n_train], feat["surname"].iloc[:n_train]]
    splitter = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=2026)
    makers = {
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=0),
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
    }
    variants = {
        "relationship columns": (roles_df, False),
        "role-split group survival": (None, True),
        "both": (roles_df, True),
    }
    print(f" {FOLDS}-fold x {REPEATS} repeats; group survival features in every column")
    print(f" {'model':<24} {'base':>7}  " + "  ".join(f"{n:>34}" for n in variants))
    for name, make in makers.items():
        a = run(base, y, keys, role, None, make, splitter, False)
        cells = []
        for extra, rg in variants.values():
            b = run(base, y, keys, role, extra, make, splitter, rg)
            se = (b - a).reshape(REPEATS, FOLDS).mean(axis=1).std(ddof=1) / np.sqrt(REPEATS)
            cells.append(f"{b.mean():.4f} ({b.mean() - a.mean():+.4f} +/- {se:.4f})")
        print(f" {name:<24} {a.mean():7.4f}  " + "  ".join(f"{c:>34}" for c in cells))


if __name__ == "__main__":
    main()
