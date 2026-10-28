# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

r"""Do fare tiers or ticket-mate cabin information improve cross-validated accuracy?

    .venv\Scripts\python.exe titanic\angles.py

All added columns are label-free (built from the passengers' own columns and ticket groups), so
they are computed once. Same folds and same group features as improve.py.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402
from analysis import FOLDS, REPEATS  # noqa: E402
from improve import evaluate  # noqa: E402

warnings.filterwarnings("ignore")


def added_columns(raw, feat):
    fare = raw["Fare"].fillna(raw["Fare"].median())
    pp = fare / feat["ticket_size"]
    x = np.log1p(pp.values).reshape(-1, 1)
    gm = GaussianMixture(5, n_init=10, random_state=0).fit(x)
    order = np.argsort(gm.means_.ravel())
    rank = {o: i for i, o in enumerate(order)}
    tier = pd.Series(gm.predict(x)).map(rank).values
    tiers = pd.get_dummies(pd.Series(tier, index=raw.index), prefix="fare_tier").astype(float)

    deck_own = raw["Cabin"].str[0]
    # deck shared by anyone on the same ticket who has a cabin
    deck_ticket = deck_own.groupby(raw["Ticket"]).transform(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    deck_filled = deck_own.fillna(deck_ticket)
    order_deck = {c: i for i, c in enumerate("ABCDEFG")}
    deck_num = deck_filled.map(order_deck)
    cols = pd.DataFrame(index=raw.index)
    cols["deck_filled_known"] = deck_filled.notna().astype(float)
    cols["deck_filled_level"] = deck_num.fillna(-1.0)
    cols["cabin_from_ticketmate"] = (deck_own.isna() & deck_filled.notna()).astype(float)
    sets = {
        "fare tiers": tiers,
        "deck filled from ticket-mates": cols,
        "both": tiers.join(cols),
    }
    return sets


def main():
    raw, n_train = features.load()
    feat = features.build(raw)
    base = features.numeric_matrix(feat)
    y = raw["Survived"].iloc[:n_train].astype(int)
    keys = [feat["ticket"].iloc[:n_train], feat["surname"].iloc[:n_train]]
    splitter = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=2026)
    makers = {
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=0),
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
    }
    sets = added_columns(raw, feat)
    print(f" {FOLDS}-fold x {REPEATS} repeats, group features in every column; gain over the base columns")
    print(f" {'model':<24} {'base':>7}  " + "  ".join(f"{n:>30}" for n in sets))
    for name, make in makers.items():
        a = evaluate(base.iloc[:n_train], y, keys, make, splitter)
        cells = []
        for extra in sets.values():
            b = evaluate(base.join(extra).iloc[:n_train], y, keys, make, splitter)
            se = (b - a).reshape(REPEATS, FOLDS).mean(axis=1).std(ddof=1) / np.sqrt(REPEATS)
            cells.append(f"{b.mean():.4f} ({b.mean() - a.mean():+.4f} +/- {se:.4f})")
        print(f" {name:<24} {a.mean():7.4f}  " + "  ".join(f"{c:>30}" for c in cells))


if __name__ == "__main__":
    main()
