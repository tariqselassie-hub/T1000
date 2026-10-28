# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

r"""How many fare tiers does the Titanic data show? Run: .venv\Scripts\python.exe titanic\fares.py"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402

warnings.filterwarnings("ignore")
raw, n_train = features.load()
feat = features.build(raw)
fare = raw["Fare"].fillna(raw["Fare"].median())
raw["pp"] = fare / feat["ticket_size"]
raw["fare_total"] = fare

print("Ticket price (whole ticket) and per-person price by class, all 1,309 passengers")
print(raw.groupby("Pclass")[["fare_total", "pp"]].describe(percentiles=[.1, .5, .9]).round(1).T.to_string())

print("\nMost common exact ticket prices (GBP, shared by many passengers = published tariff)")
vc = fare[fare > 0].round(4).value_counts().head(14)
cls = raw.assign(f=fare.round(4)).groupby("f")["Pclass"].agg(lambda s: s.mode()[0])
for f, c in vc.items():
    print(f"  {f:9.4f}  {c:4d} passengers  class {cls[f]}")
print(f"  fare = 0: {(fare == 0).sum()} passengers (crew / guarantee group / upgrades)")

print("\nMixture models on log(per-person price): number of clusters by BIC")
x = np.log1p(raw["pp"].values).reshape(-1, 1)
bic = {k: GaussianMixture(k, n_init=5, random_state=0).fit(x).bic(x) for k in range(1, 9)}
for k, b in bic.items():
    print(f"  k={k}: BIC {b:9.1f}")
best = min(bic, key=bic.get)
gm = GaussianMixture(best, n_init=10, random_state=0).fit(x)
order = np.argsort(gm.means_.ravel())
raw["tier"] = pd.Series(gm.predict(x)).map({o: i + 1 for i, o in enumerate(order)}).values
print(f"\nBest k = {best}. Tiers (low to high): per-person price, class mix, survival (training rows only)")
tr = raw.iloc[:n_train]
for t in range(1, best + 1):
    a = raw[raw["tier"] == t]
    s = tr[tr["tier"] == t]
    mix = a["Pclass"].value_counts(normalize=True).sort_index().round(2).to_dict()
    print(f"  tier {t}: median {a['pp'].median():7.2f} (range {a['pp'].min():.1f}-{a['pp'].max():.1f}), n={len(a):4d}, "
          f"class mix {mix}, survived {s['Survived'].mean():.3f} (n={len(s)})")

print("\nSurvival by class and sex, training rows")
print(tr.pivot_table(index="Pclass", columns="Sex", values="Survived", aggfunc=["mean", "count"]).round(3).to_string())

print("\nWithin each class: survival by fare quartile (training rows)")
for c in (1, 2, 3):
    g = tr[tr["Pclass"] == c].copy()
    g["q"] = pd.qcut(g["pp"], 4, duplicates="drop")
    print(f"  class {c}:", g.groupby("q", observed=True)["Survived"].agg(["mean", "count"]).round(3).to_dict("index"))
