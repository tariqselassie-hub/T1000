# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""Repeated cross-validation of classical baselines on the Titanic training set.

    .venv\\Scripts\\python.exe titanic\\analysis.py

Same folds for every model (5 folds x 10 repeats, fixed seeds). Label-derived "how did the rest of
the group fare" features are computed inside each fold from that fold's training labels only.
Writes results.json next to this file.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
REPEATS, FOLDS = 10, 5


def group_features(keys, y, train_idx, eval_idx, prior):
    """Leave-one-out survival of the other labelled members of the same group.

    `keys` is a Series aligned to the rows; only rows in `train_idx` contribute their labels.
    Returns (n_others, mean_others) for the rows in train_idx (excluding self) and eval_idx.
    """
    lab = pd.DataFrame({"k": keys.values, "y": y.values}, index=keys.index)
    lab = lab.loc[train_idx]
    g_sum = lab.groupby("k")["y"].sum()
    g_cnt = lab.groupby("k")["y"].count()

    def stats(idx, self_in_train):
        k = keys.loc[idx]
        s = k.map(g_sum).fillna(0.0).values
        c = k.map(g_cnt).fillna(0.0).values
        if self_in_train:
            s = s - y.loc[idx].values
            c = c - 1.0
        mean = np.where(c > 0, s / np.maximum(c, 1.0), prior)
        return np.column_stack([c, mean])

    return stats(train_idx, True), stats(eval_idx, False)


def models():
    return {
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
        "random_forest": lambda: RandomForestClassifier(500, min_samples_leaf=3, random_state=0, n_jobs=-1),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=0
        ),
    }


def main():
    df, n_train = features.load()
    feat = features.build(df)
    x_all = features.numeric_matrix(feat)
    y = df["Survived"].iloc[:n_train].astype(int)
    x = x_all.iloc[:n_train]
    keys = {"ticket": feat["ticket"].iloc[:n_train], "surname": feat["surname"].iloc[:n_train]}
    splitter = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=2026)

    names = ["sex_rule"] + [f"{m}" for m in models()] + [f"{m}+group" for m in models()]
    scores = {n: [] for n in names}
    for fold, (tr, ev) in enumerate(splitter.split(x, y)):
        tr_idx, ev_idx = x.index[tr], x.index[ev]
        scores["sex_rule"].append(float((x.loc[ev_idx, "sex"].values == y.loc[ev_idx].values).mean()))
        prior = float(y.loc[tr_idx].mean())
        gt, ge = {}, {}
        for name, k in keys.items():
            gt[name], ge[name] = group_features(k, y, tr_idx, ev_idx, prior)
        xt_g = np.hstack([x.loc[tr_idx].values, gt["ticket"], gt["surname"]])
        xe_g = np.hstack([x.loc[ev_idx].values, ge["ticket"], ge["surname"]])
        for name, make in models().items():
            m = make().fit(x.loc[tr_idx].values, y.loc[tr_idx].values)
            scores[name].append(float((m.predict(x.loc[ev_idx].values) == y.loc[ev_idx].values).mean()))
            m = make().fit(xt_g, y.loc[tr_idx].values)
            scores[f"{name}+group"].append(float((m.predict(xe_g) == y.loc[ev_idx].values).mean()))

    summary = {}
    best = max(names, key=lambda n: np.mean(scores[n]))
    print(f" {FOLDS}-fold x {REPEATS} repeats, {len(scores[best])} evaluations per model (accuracy)")
    print(f" {'model':<32} {'mean':>7} {'sd(folds)':>10} {'vs best (paired, mean diff)':>30}")
    for n in sorted(names, key=lambda n: -np.mean(scores[n])):
        a, b = np.array(scores[n]), np.array(scores[best])
        diff = a - b
        # Repeats share data, so the fold-level paired standard error understates the real one;
        # a per-repeat mean is the unit that is roughly independent.
        per_rep = diff.reshape(REPEATS, FOLDS).mean(axis=1)
        se = per_rep.std(ddof=1) / np.sqrt(REPEATS) if n != best else 0.0
        summary[n] = {"mean": float(a.mean()), "sd_folds": float(a.std(ddof=1)),
                      "diff_vs_best": float(diff.mean()), "diff_se": float(se)}
        print(f" {n:<32} {a.mean():7.4f} {a.std(ddof=1):10.4f} {diff.mean():+14.4f} (se {se:.4f})")
    (HERE / "results.json").write_text(json.dumps({"folds": FOLDS, "repeats": REPEATS, "models": summary}, indent=2))


if __name__ == "__main__":
    main()
