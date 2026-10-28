# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

r"""Does the quantum circuit help as a member of the ensemble?

    .venv\Scripts\python.exe titanic\vqc_export.py vqc_final 777 5
    target\release\titanic_vqc.exe --dir titanic\vqc_final --epochs 600 --lr 0.1 --depth 4
    .venv\Scripts\python.exe titanic\quantum_member.py

Uses the same 25 folds (seed 777) the circuit was scored on. The classical ensemble is the one in
submit.py (logistic + random forest + gradient boosting on all columns, group and role-split group
survival). The circuit's readout r in [-1, 1] becomes a probability (1 + r) / 2. Reports accuracy
of the ensemble alone, the circuit alone, and blends with weight w on the circuit.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402
from analysis import group_features  # noqa: E402
from relations import role_columns, role_group_features  # noqa: E402
from submit import ensemble_proba  # noqa: E402

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]


def main():
    folds_dir = HERE / "data" / "vqc_final" if (HERE / "data" / "vqc_final").exists() else HERE / "vqc_final"
    raw, n_train = features.load()
    feat = features.build(raw)
    x = features.numeric_matrix(feat).iloc[:n_train]
    y = raw["Survived"].iloc[:n_train].astype(int)
    keys = [feat["ticket"].iloc[:n_train], feat["surname"].iloc[:n_train]]
    _, role = role_columns(raw, feat)
    role = role.iloc[:n_train]
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=777)

    acc = {w: [] for w in WEIGHTS}
    both_wrong, ens_wrong, vqc_wrong = [], [], []
    for k, (tr, ev) in enumerate(splitter.split(x, y)):
        tr_idx, ev_idx = x.index[tr], x.index[ev]
        prior = float(y.loc[tr_idx].mean())
        gt = [group_features(kk, y, tr_idx, ev_idx, prior) for kk in keys]
        rt, re = role_group_features(keys, y, role, tr_idx, ev_idx, prior)
        xt = np.hstack([x.loc[tr_idx].values] + [g[0] for g in gt] + [rt])
        xe = np.hstack([x.loc[ev_idx].values] + [g[1] for g in gt] + [re])
        p_ens = ensemble_proba(xt, y.loc[tr_idx].values, xe)
        r = pd.read_csv(folds_dir / f"fold_{k}_pred.csv")["pred"].values
        assert len(r) == len(ev_idx), (k, len(r), len(ev_idx))
        p_vqc = np.clip((1.0 + r) / 2.0, 0.0, 1.0)
        truth = y.loc[ev_idx].values
        for w in WEIGHTS:
            p = (1 - w) * p_ens + w * p_vqc
            acc[w].append(float(((p > 0.5) == truth).mean()))
        e_wrong, v_wrong = (p_ens > 0.5) != truth, (p_vqc > 0.5) != truth
        ens_wrong.append(e_wrong.mean()); vqc_wrong.append(v_wrong.mean()); both_wrong.append((e_wrong & v_wrong).mean())

    base = np.array(acc[0.0])
    print(f" 25 folds (seed 777, 5 repeats); circuit = 8 qubits, depth 4, 8 features")
    print(f" {'weight on circuit':<20} {'accuracy':>9} {'vs ensemble':>12} {'se':>7}")
    for w in WEIGHTS:
        a = np.array(acc[w])
        d = a - base
        per_rep = d.reshape(5, 5).mean(axis=1)
        se = per_rep.std(ddof=1) / np.sqrt(5) if w else 0.0
        tag = "  (ensemble alone)" if w == 0 else ("  (circuit alone)" if w == 1 else "")
        print(f" {w:<20.1f} {a.mean():9.4f} {d.mean():+12.4f} {se:7.4f}{tag}")
    print(f"\n error rates: ensemble {np.mean(ens_wrong):.4f}, circuit {np.mean(vqc_wrong):.4f}, "
          f"wrong together {np.mean(both_wrong):.4f}")
    indep = np.mean(ens_wrong) * np.mean(vqc_wrong)
    print(f" if their mistakes were independent they would be wrong together {indep:.4f} of the time")


if __name__ == "__main__":
    main()
