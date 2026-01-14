#!/usr/bin/env python3
"""
Evaluate the construction battery using cross-corpus PR-AUC.

Reads out/cx_battery_candidates.csv, trains logistic regression models on
cue features, and reports cross-corpus transfer for each construction with
ablation variants.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score

IN_PATH = os.path.join("out", "cx_battery_candidates.csv")
OUT_EVAL = os.path.join("out", "cx_battery_eval.csv")
OUT_PREV = os.path.join("out", "cx_battery_prevalence.csv")

FEATURE_SETS = {
    "full": ["cue1", "cue2", "cue3"],
    "no_cue1": ["cue2", "cue3"],
    "no_cue2": ["cue1", "cue3"],
    "no_cue3": ["cue1", "cue2"],
}

MIN_CANDIDATES = 20
MIN_POSITIVES = 10


def train_eval(train: pd.DataFrame, test: pd.DataFrame, features: List[str]) -> Dict[str, float]:
    X_tr = train[features].astype(int).values
    y_tr = train["label"].astype(int).values
    X_te = test[features].astype(int).values
    y_te = test["label"].astype(int).values
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_tr, y_tr)
    probs = clf.predict_proba(X_te)[:, 1]
    pr_auc = average_precision_score(y_te, probs) if len(np.unique(y_te)) > 1 else 0.0
    y_hat = (probs >= 0.5).astype(int)
    return {
        "pr_auc": float(pr_auc),
        "precision": float(precision_score(y_te, y_hat, zero_division=0)),
        "recall": float(recall_score(y_te, y_hat, zero_division=0)),
        "f1": float(f1_score(y_te, y_hat, zero_division=0)),
    }


def main() -> None:
    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(f"Missing candidates file: {IN_PATH}")
    df = pd.read_csv(IN_PATH)
    # prevalence summary
    prev_rows = []
    for (construction, corpus), sub in df.groupby(["construction", "corpus"]):
        prev_rows.append({
            "construction": construction,
            "corpus": corpus,
            "n_candidates": int(len(sub)),
            "n_positive": int(sub["label"].sum()),
            "prevalence": float(sub["label"].mean()) if len(sub) else 0.0,
        })
    os.makedirs(os.path.dirname(OUT_PREV), exist_ok=True)
    pd.DataFrame(prev_rows).to_csv(OUT_PREV, index=False)

    rows: List[Dict[str, float]] = []
    corpora = sorted(df["corpus"].unique())
    constructions = sorted(df["construction"].unique())

    for construction in constructions:
        sub = df[df["construction"] == construction]
        for train_c in corpora:
            for test_c in corpora:
                if train_c == test_c:
                    continue
                train = sub[sub["corpus"] == train_c]
                test = sub[sub["corpus"] == test_c]
                n_train = len(train)
                n_test = len(test)
                pos_train = int(train["label"].sum()) if n_train else 0
                pos_test = int(test["label"].sum()) if n_test else 0
                status = "ok"
                if n_train < MIN_CANDIDATES or n_test < MIN_CANDIDATES or pos_train < MIN_POSITIVES or pos_test < MIN_POSITIVES:
                    status = "not_estimable"
                for model, feats in FEATURE_SETS.items():
                    if status != "ok" or len(np.unique(train["label"])) < 2 or len(np.unique(test["label"])) < 2:
                        rows.append({
                            "construction": construction,
                            "train": train_c,
                            "test": test_c,
                            "model": model,
                            "n_train": n_train,
                            "n_test": n_test,
                            "pos_train": pos_train,
                            "pos_test": pos_test,
                            "status": status,
                            "pr_auc": np.nan,
                            "precision": np.nan,
                            "recall": np.nan,
                            "f1": np.nan,
                        })
                        continue
                    metrics = train_eval(train, test, feats)
                    rows.append({
                        "construction": construction,
                        "train": train_c,
                        "test": test_c,
                        "model": model,
                        "n_train": n_train,
                        "n_test": n_test,
                        "pos_train": pos_train,
                        "pos_test": pos_test,
                        "status": status,
                        **metrics,
                    })

    os.makedirs(os.path.dirname(OUT_EVAL), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_EVAL, index=False)
    print(f"Saved evaluation metrics to {OUT_EVAL}")
    print(f"Saved prevalence summary to {OUT_PREV}")


if __name__ == "__main__":
    main()
