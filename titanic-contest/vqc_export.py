# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""Writes the cross-validation folds for the quantum model (`titanic_vqc`) to vqc_folds/.

    .venv\\Scripts\\python.exe titanic\\vqc_export.py

Eight features, each scaled to [0, pi] (an Ry encoding over a wider range would map both ends of a binary feature to the same state) with that fold's training minimum and maximum, so the
quantum circuit and the classical comparison in vqc_compare.py see exactly the same inputs. The
two group-survival features come from the fold's training labels only. Files per fold k:
fold_k_train.csv and fold_k_eval.csv (x0..x7,y with y in {-1,+1}).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402
from analysis import group_features  # noqa: E402

HERE = Path(__file__).resolve().parent
FOLDS = 5
NAMES = ["sex", "pclass", "age", "log_fare", "family", "is_master", "ticket_group_survival", "surname_group_survival"]


def base_matrix(feat):
    m = pd.DataFrame(index=feat.index)
    m["sex"] = feat["sex"]
    m["pclass"] = feat["pclass"]
    m["age"] = feat["age"]
    m["log_fare"] = feat["log_fare"]
    m["family"] = feat["family"]
    m["is_master"] = (feat["title"] == "Master").astype(int)
    return m


def fold_matrices(feat, y, tr_idx, ev_idx):
    base = base_matrix(feat).iloc[: len(y)]
    prior = float(y.loc[tr_idx].mean())
    keys = [feat["ticket"].iloc[: len(y)], feat["surname"].iloc[: len(y)]]
    gt = [group_features(k, y, tr_idx, ev_idx, prior) for k in keys]
    xt = np.column_stack([base.loc[tr_idx].values, gt[0][0][:, 1], gt[1][0][:, 1]])
    xe = np.column_stack([base.loc[ev_idx].values, gt[0][1][:, 1], gt[1][1][:, 1]])
    return xt, xe


def to_angles(xt, xe):
    lo, hi = xt.min(axis=0), xt.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    f = lambda a: np.clip((a - lo) / span, 0.0, 1.0) * np.pi
    return f(xt), f(xe)


def main():
    default_sub = "data/vqc_folds" if (HERE / "data").exists() else "vqc_folds"
    out_dir = HERE / (sys.argv[1] if len(sys.argv) > 1 else default_sub)
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
    repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    raw, n_train = features.load()
    feat = features.build(raw)
    y = raw["Survived"].iloc[:n_train].astype(int)
    x = feat.iloc[:n_train]
    out_dir.mkdir(exist_ok=True)
    splitter = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=repeats, random_state=seed)
    for k, (tr, ev) in enumerate(splitter.split(x, y)):
        tr_idx, ev_idx = x.index[tr], x.index[ev]
        xt, xe = fold_matrices(feat, y, tr_idx, ev_idx)
        at, ae = to_angles(xt, xe)
        for name, a, idx in (("train", at, tr_idx), ("eval", ae, ev_idx)):
            df = pd.DataFrame(a, columns=[f"x{i}" for i in range(a.shape[1])])
            df["y"] = np.where(y.loc[idx].values == 1, 1, -1)
            df.to_csv(out_dir / f"fold_{k}_{name}.csv", index=False, float_format="%.9f")
    print(f"wrote {FOLDS * repeats} folds to {out_dir} using features: {', '.join(NAMES)}")


if __name__ == "__main__":
    main()
