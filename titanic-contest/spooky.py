# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""Looks for structure the best classical model leaves behind.

    .venv\\Scripts\\python.exe titanic\\spooky.py

Takes out-of-fold predictions of the gradient-boosting model with group features (averaged over
repeated 5-fold runs), forms residuals (survived minus predicted probability), and tests each of
many candidate columns for association with the residuals using a permutation test. Because many
columns are tried, p-values are Bonferroni-corrected over the whole battery: a column is only
interesting if it survives that.
"""

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402
from analysis import group_features  # noqa: E402

warnings.filterwarnings("ignore")
REPEATS, PERMS = 8, 20000
rng = np.random.default_rng(7)


def oof_probabilities(x, y, keys):
    prob = np.zeros(len(y))
    for rep in range(REPEATS):
        for tr, ev in StratifiedKFold(5, shuffle=True, random_state=100 + rep).split(x, y):
            tr_idx, ev_idx = x.index[tr], x.index[ev]
            prior = float(y.loc[tr_idx].mean())
            gt = [group_features(k, y, tr_idx, ev_idx, prior) for k in keys]
            xt = np.hstack([x.loc[tr_idx].values] + [g[0] for g in gt])
            xe = np.hstack([x.loc[ev_idx].values] + [g[1] for g in gt])
            m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=rep)
            m.fit(xt, y.loc[tr_idx].values)
            prob[ev] += m.predict_proba(xe)[:, 1] / REPEATS
    return prob


def candidates(raw, feat):
    c = {}
    c["PassengerId"] = raw["PassengerId"]
    c["name_length"] = raw["Name"].str.len()
    c["name_words"] = raw["Name"].str.split().str.len()
    c["maiden_name_in_brackets"] = raw["Name"].str.contains(r"\(").astype(int)
    c["nickname_in_quotes"] = raw["Name"].str.contains('"').astype(int)
    c["surname_length"] = feat["surname"].str.len()
    c["surname_alphabet_position"] = feat["surname"].str[0].map(lambda ch: ord(ch.upper()) - 64)
    t = raw["Ticket"].fillna("")
    c["ticket_number"] = pd.to_numeric(t.str.extract(r"(\d+)\s*$")[0], errors="coerce")
    c["ticket_has_prefix"] = t.str.contains(r"[A-Za-z]").astype(int)
    c["ticket_length"] = t.str.len()
    c["ticket_last_digit"] = pd.to_numeric(t.str.extract(r"(\d)\s*$")[0], errors="coerce")
    cab = raw["Cabin"].fillna("")
    num = pd.to_numeric(cab.str.extract(r"(\d+)")[0], errors="coerce")
    c["cabin_number"] = num
    c["cabin_number_odd"] = (num % 2).where(num.notna())
    c["cabin_count"] = cab.map(lambda s: len(s.split()) if s else 0)
    fare = raw["Fare"].fillna(raw["Fare"].median())
    c["fare_is_round"] = ((fare * 100) % 100 == 0).astype(int)
    c["fare_cents"] = (fare * 100) % 100
    c["fare_is_zero"] = (fare == 0).astype(int)
    age = raw["Age"]
    c["age_has_fraction"] = ((age % 1) != 0).astype(int).where(age.notna())
    c["age_ends_in_zero"] = ((age % 10) == 0).astype(int).where(age.notna())
    c["age_missing"] = age.isna().astype(int)
    c["sibsp_x_parch"] = raw["SibSp"] * raw["Parch"]
    c["family_over_4"] = ((raw["SibSp"] + raw["Parch"]) >= 4).astype(int)
    c["embarked_C"] = (raw["Embarked"] == "C").astype(int)
    c["embarked_Q"] = (raw["Embarked"] == "Q").astype(int)
    c["ticket_size"] = feat["ticket_size"]
    c["surname_group_size"] = feat.groupby("surname")["surname"].transform("count")
    c["rank_in_ticket_by_age"] = raw.groupby("Ticket")["Age"].rank(method="first")
    c["is_oldest_in_ticket"] = (raw.groupby("Ticket")["Age"].transform("max") == raw["Age"]).astype(int)
    return pd.DataFrame(c)


def main():
    raw, n_train = features.load()
    feat = features.build(raw)
    x = features.numeric_matrix(feat).iloc[:n_train]
    y = raw["Survived"].iloc[:n_train].astype(int)
    keys = [feat["ticket"].iloc[:n_train], feat["surname"].iloc[:n_train]]
    prob = oof_probabilities(x, y, keys)
    resid = y.values - prob
    acc = float(((prob > 0.5) == y.values).mean())
    print(f" out-of-fold accuracy of the model whose mistakes we study: {acc:.4f}")
    print(f" residual sd {resid.std():.3f}; testing for structure in what it misses\n")

    cand = candidates(raw, feat).iloc[:n_train]
    rows = []
    for name in cand.columns:
        v = cand[name]
        ok = v.notna().values
        if ok.sum() < 100 or v[ok].nunique() < 2:
            continue
        r_obs = stats.spearmanr(v[ok], resid[ok])[0]
        ranks = stats.rankdata(v[ok])
        # permutation distribution of the Spearman correlation with the residuals
        perm = np.array([stats.pearsonr(ranks, rng.permutation(stats.rankdata(resid[ok])))[0] for _ in range(PERMS // 10)])
        p = (np.sum(np.abs(perm) >= abs(r_obs)) + 1) / (len(perm) + 1)
        rows.append((name, int(ok.sum()), r_obs, p))
    rows.sort(key=lambda r: r[3])
    m = len(rows)
    print(f" {m} candidate columns tested; Bonferroni threshold p < {0.05 / m:.5f}")
    print(f" {'column':<30} {'n':>5} {'spearman r':>11} {'perm p':>9} {'Bonferroni p':>13}")
    for name, n, r, p in rows[:12]:
        print(f" {name:<30} {n:5d} {r:+11.4f} {p:9.4f} {min(1.0, p * m):13.4f}")

    # Do group members share residual beyond what the model already encodes?
    res = pd.Series(resid, index=x.index)
    for label, key in (("ticket", keys[0]), ("surname", keys[1])):
        g = pd.DataFrame({"k": key, "r": res})
        size = g.groupby("k")["r"].transform("count")
        multi = g[size > 1]
        total = multi["r"].sum()
        loo = (multi.groupby("k")["r"].transform("sum") - multi["r"]) / (size[size > 1] - 1)
        rr = stats.pearsonr(multi["r"], loo)[0]
        print(f"\n residual of a passenger vs mean residual of the others in the same {label} group: "
              f"r = {rr:+.3f} over {len(multi)} passengers in {multi['k'].nunique()} groups")


if __name__ == "__main__":
    main()
