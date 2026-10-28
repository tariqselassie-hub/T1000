# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""Does adding the columns found in the residual search improve cross-validated accuracy?

    .venv\\Scripts\\python.exe titanic\\improve.py

Caveat printed with the result: those columns were chosen by looking at this same training data,
so the gain measured here is optimistic. The honest test is the Kaggle public score.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402
from analysis import group_features, FOLDS, REPEATS  # noqa: E402

warnings.filterwarnings("ignore")


def extras(raw, feat):
    e = feat[[]].copy()
    e["name_length"] = raw["Name"].str.len()
    e["maiden_name_in_brackets"] = raw["Name"].str.contains(r"\(").astype(int)
    e["nickname_in_quotes"] = raw["Name"].str.contains('"').astype(int)
    e["is_oldest_in_ticket"] = (raw.groupby("Ticket")["Age"].transform("max") == raw["Age"]).astype(int)
    return e


def evaluate(x, y, keys, make, splitter):
    acc = []
    for tr, ev in splitter.split(x, y):
        tr_idx, ev_idx = x.index[tr], x.index[ev]
        prior = float(y.loc[tr_idx].mean())
        gt = [group_features(k, y, tr_idx, ev_idx, prior) for k in keys]
        xt = np.hstack([x.loc[tr_idx].values] + [g[0] for g in gt])
        xe = np.hstack([x.loc[ev_idx].values] + [g[1] for g in gt])
        m = make().fit(xt, y.loc[tr_idx].values)
        acc.append(float((m.predict(xe) == y.loc[ev_idx].values).mean()))
    return np.array(acc)


def main():
    raw, n_train = features.load()
    feat = features.build(raw)
    base = features.numeric_matrix(feat)
    plus = base.join(extras(raw, feat))
    y = raw["Survived"].iloc[:n_train].astype(int)
    keys = [feat["ticket"].iloc[:n_train], feat["surname"].iloc[:n_train]]
    splitter = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=2026)
    makers = {
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=0),
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
    }
    print(f" {FOLDS}-fold x {REPEATS} repeats, group features included in both columns")
    print(f" {'model':<26} {'base':>8} {'+ residual columns':>20} {'gain':>8} {'se':>8}")
    for name, make in makers.items():
        a = evaluate(base.iloc[:n_train], y, keys, make, splitter)
        b = evaluate(plus.iloc[:n_train], y, keys, make, splitter)
        per_rep = (b - a).reshape(REPEATS, FOLDS).mean(axis=1)
        se = per_rep.std(ddof=1) / np.sqrt(REPEATS)
        print(f" {name:<26} {a.mean():8.4f} {b.mean():20.4f} {b.mean() - a.mean():+8.4f} {se:8.4f}")
    print("\n These columns were selected on this data, so the gain is optimistic.")


if __name__ == "__main__":
    main()
