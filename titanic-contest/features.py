# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""Feature construction for the Kaggle Titanic data (train.csv and test.csv in this directory).

Everything here uses only the passengers' own columns and counts over train and test together
(group sizes), never the Survived label, so it can be computed once for all rows.
Label-derived features (how the rest of a family fared) are built per cross-validation fold in
analysis.py.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" if (HERE / "data").exists() else HERE

TITLE_MAP = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare", "Capt": "Rare",
    "Sir": "Rare", "Lady": "Rare", "Countess": "Rare", "Don": "Rare", "Dona": "Rare",
    "Jonkheer": "Rare",
}


def load():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    test["Survived"] = np.nan
    return pd.concat([train, test], ignore_index=True), len(train)


def build(df):
    out = pd.DataFrame(index=df.index)
    title = df["Name"].str.extract(r",\s*([^\.]+)\.")[0].str.strip().map(TITLE_MAP).fillna("Rare")
    out["title"] = title
    out["surname"] = df["Name"].str.split(",").str[0].str.strip()
    out["sex"] = (df["Sex"] == "female").astype(int)
    out["pclass"] = df["Pclass"]
    out["sibsp"] = df["SibSp"]
    out["parch"] = df["Parch"]
    out["family"] = df["SibSp"] + df["Parch"] + 1
    out["alone"] = (out["family"] == 1).astype(int)
    fare = df["Fare"].fillna(df.groupby("Pclass")["Fare"].transform("median"))
    out["ticket"] = df["Ticket"]
    out["ticket_size"] = df.groupby("Ticket")["Ticket"].transform("count")
    out["fare_pp"] = fare / out["ticket_size"]
    out["log_fare"] = np.log1p(fare)
    age = df["Age"].copy()
    out["age_known"] = age.notna().astype(int)
    median_by_group = df.assign(title=title).groupby(["title", "Pclass"])["Age"].transform("median")
    out["age"] = age.fillna(median_by_group).fillna(df["Age"].median())
    out["child"] = (out["age"] < 14).astype(int)
    out["cabin_known"] = df["Cabin"].notna().astype(int)
    out["deck"] = df["Cabin"].fillna("U").str[0]
    out["embarked"] = df["Embarked"].fillna("S")
    out["surname_ticket"] = out["surname"] + "|" + out["ticket"]
    return out


def numeric_matrix(feat):
    """One-hot encodes the categorical columns; returns a float DataFrame."""
    cols = ["sex", "pclass", "sibsp", "parch", "family", "alone", "ticket_size", "fare_pp", "log_fare",
            "age", "age_known", "child", "cabin_known"]
    x = feat[cols].astype(float).copy()
    for c in ("title", "deck", "embarked"):
        x = x.join(pd.get_dummies(feat[c], prefix=c).astype(float))
    return x
