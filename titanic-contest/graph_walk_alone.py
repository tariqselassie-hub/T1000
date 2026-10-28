# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

r"""How well does each walk, on its own, separate survivors from the dead? (leave-one-out AUC)

    .venv\Scripts\python.exe titanic\graph_walk_alone.py

Uses all 891 training labels with each passenger's own label removed exactly. Reports the AUC of
the spread score over the passengers who have at least one neighbour, for every filter and setting.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402
from graph_walk import apply, build_graphs, filters, spectrum  # noqa: E402

raw, n_train = features.load()
feat = features.build(raw)
y = raw["Survived"].iloc[:n_train].astype(int).values
prior = y.mean()
for gname, a in build_graphs(raw, feat).items():
    lam, v, connected = spectrum(a)
    s = np.zeros(len(lam))
    s[:n_train] = y - prior
    idx = np.arange(n_train)
    deg_ok = connected[:n_train] & (a[:n_train].sum(1) > 0)
    print(f"\n {gname}: {int(deg_ok.sum())} of {n_train} training passengers have a neighbour")
    print(f" {'filter':<12} {'setting':>8} {'AUC (Re)':>9} {'AUC (-Im)':>10} {'AUC (|amp|^2)':>14}")
    for (fam, par), g in filters(lam).items():
        f, kii = apply(v, g, s)
        loo = f[:n_train] - kii[:n_train] * s[:n_train]
        auc = roc_auc_score(y[deg_ok], loo.real[deg_ok])
        auc2 = roc_auc_score(y[deg_ok], (np.abs(loo) ** 2)[deg_ok]) if fam == "quantum" else float("nan")
        aucim = roc_auc_score(y[deg_ok], -loo.imag[deg_ok]) if fam == "quantum" else float("nan")
        print(f" {fam:<12} {par:>8} {auc:9.4f} {aucim:10.4f} {auc2:14.4f}")
