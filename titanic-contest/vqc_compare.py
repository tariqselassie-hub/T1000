# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""Scores the quantum circuit against classical models on the same folds and the same inputs.

    .venv\\Scripts\\python.exe titanic\\vqc_compare.py [folds_dir]

Reads fold_k_train.csv, fold_k_eval.csv and fold_k_pred.csv (written by `titanic_vqc`). Every
classical model gets the identical eight scaled features.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent


def main():
    default_dir = HERE / "data" / "vqc_folds" if (HERE / "data" / "vqc_folds").exists() else HERE / "vqc_folds"
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir
    makers = {
        "logistic": lambda: LogisticRegression(C=1.0, max_iter=2000),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=0),
        "mlp (16 hidden)": lambda: MLPClassifier((16,), max_iter=1500, random_state=0),
    }
    acc = {"vqc (quantum)": [], **{k: [] for k in makers}}
    k = 0
    while (directory / f"fold_{k}_pred.csv").exists():
        tr = pd.read_csv(directory / f"fold_{k}_train.csv")
        ev = pd.read_csv(directory / f"fold_{k}_eval.csv")
        pred = pd.read_csv(directory / f"fold_{k}_pred.csv")["pred"].values
        feats = [c for c in tr.columns if c != "y"]
        y_ev = ev["y"].values
        acc["vqc (quantum)"].append(float(((pred >= 0) == (y_ev > 0)).mean()))
        for name, make in makers.items():
            m = make().fit(tr[feats].values, tr["y"].values > 0)
            acc[name].append(float((m.predict(ev[feats].values) == (y_ev > 0)).mean()))
        k += 1
    q = np.array(acc["vqc (quantum)"])
    print(f" {k} folds from {directory.name}; accuracy on held-out rows, identical inputs")
    print(f" {'model':<26} {'mean':>7} {'sd':>7} {'vs vqc':>9} {'se':>7}")
    for name, a in acc.items():
        a = np.array(a)
        d = a - q
        se = d.std(ddof=1) / np.sqrt(len(d)) if name != "vqc (quantum)" else 0.0
        print(f" {name:<26} {a.mean():7.4f} {a.std(ddof=1):7.4f} {d.mean():+9.4f} {se:7.4f}")
    base = float(max((1 - tr["y"].gt(0).mean()), tr["y"].gt(0).mean()))
    print(f"\n majority-class rate (last fold's training set): {base:.4f}")


if __name__ == "__main__":
    main()
