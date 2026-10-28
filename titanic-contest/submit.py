# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""Cross-validates an average of three classical models with group features (ticket and surname
survival, and the same split by role: women/children and men), then writes
submission.csv (PassengerId,Survived) for the Kaggle test set.

    .venv\\Scripts\\python.exe titanic\\submit.py

For the test passengers, the group features use the labels of the whole training set.
"""

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
from analysis import group_features  # noqa: E402
from relations import role_columns, role_group_features  # noqa: E402

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent


def makers():
    return [
        make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
        RandomForestClassifier(500, min_samples_leaf=3, random_state=0, n_jobs=-1),
        HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0),
    ]


def ensemble_proba(xt, yt, xe):
    return np.mean([m.fit(xt, yt).predict_proba(xe)[:, 1] for m in makers()], axis=0)


def main():
    raw, n_train = features.load()
    feat = features.build(raw)
    x_all = features.numeric_matrix(feat)
    y = raw["Survived"].iloc[:n_train].astype(int)
    x = x_all.iloc[:n_train]
    keys = [feat["ticket"], feat["surname"]]
    _, role = role_columns(raw, feat)

    acc = []
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=2026)
    for tr, ev in splitter.split(x, y):
        tr_idx, ev_idx = x.index[tr], x.index[ev]
        prior = float(y.loc[tr_idx].mean())
        gt = [group_features(k.iloc[:n_train], y, tr_idx, ev_idx, prior) for k in keys]
        rt, re = role_group_features([k.iloc[:n_train] for k in keys], y, role.iloc[:n_train], tr_idx, ev_idx, prior)
        xt = np.hstack([x.loc[tr_idx].values] + [g[0] for g in gt] + [rt])
        xe = np.hstack([x.loc[ev_idx].values] + [g[1] for g in gt] + [re])
        p = ensemble_proba(xt, y.loc[tr_idx].values, xe)
        acc.append(float(((p > 0.5) == y.loc[ev_idx].values).mean()))
    acc = np.array(acc)
    print(f" ensemble (logistic + random forest + gradient boosting, group + role-split group features): "
          f"{acc.mean():.4f} +/- {acc.std(ddof=1):.4f} over {len(acc)} held-out folds")

    # Final fit on all labelled rows; the test rows' groups use every training label.
    tr_idx = x.index
    te_idx = x_all.index[n_train:]
    ylab = raw["Survived"].copy()
    ylab.iloc[n_train:] = 0
    ylab = ylab.astype(int)
    prior = float(y.mean())
    gt = [group_features(k, ylab, tr_idx, te_idx, prior) for k in keys]
    rt, re = role_group_features(keys, ylab, role, tr_idx, te_idx, prior)
    xt = np.hstack([x.values] + [g[0] for g in gt] + [rt])
    xe = np.hstack([x_all.loc[te_idx].values] + [g[1] for g in gt] + [re])
    p = ensemble_proba(xt, y.values, xe)
    out = pd.DataFrame({"PassengerId": raw.loc[te_idx, "PassengerId"].astype(int), "Survived": (p > 0.5).astype(int)})
    out.to_csv(features.DATA_DIR / "submission.csv", index=False)
    print(f" wrote {len(out)} rows to titanic-contest/data/submission.csv; predicted survival rate {out['Survived'].mean():.3f}")


if __name__ == "__main__":
    main()
