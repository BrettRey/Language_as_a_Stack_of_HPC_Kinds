#!/usr/bin/env python3
"""
Compute precision-recall curves and evaluation metrics for the let alone cue bundle.

This script reads the anchor-present candidate set produced by
`11_extract_let_alone.py`, trains regularized logistic models on the cue
bundle (anchor+parallelism+licensing) with ablations (drop parallelism;
drop licensing), and evaluates cross-corpus transfer (GUM->EWT and EWT->GUM).

Outputs:
  - out/let_alone_eval.csv      : summary metrics table
  - images/let_alone_prcurve.pdf: PR curves for GUM->EWT
  - images/appendix/let_alone_prcurve_ewt2gum.png: PR curves for EWT->GUM
  - out/let_alone_errors.tsv    : top misclassifications by predicted score
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_style import MODEL_COLORS, set_plot_style
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score


IN_PATH = os.path.join("out", "let_alone_anchor_present_eval.csv")
OUT_EVAL = os.path.join("out", "let_alone_eval.csv")
OUT_ERRS = os.path.join("out", "let_alone_errors.tsv")

FEATURE_SETS = {
    "full": ["anchor_present", "parallelism", "licensing"],
    "no_parallelism": ["anchor_present", "licensing"],
    "no_licensing": ["anchor_present", "parallelism"],
}


def load_data() -> pd.DataFrame:
    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(f"Missing evaluation file: {IN_PATH}")
    df = pd.read_csv(IN_PATH)
    required = {"corpus", "label", "anchor_present", "parallelism", "licensing"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {IN_PATH}: {sorted(missing)}")
    return df


def split_corpora(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gum = df[df["corpus"].str.lower() == "gum"].copy()
    ewt = df[df["corpus"].str.lower() == "ewt"].copy()
    return gum, ewt


def train_predict(train: pd.DataFrame, test: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X_tr = train[features].astype(int).values
    y_tr = train["label"].astype(int).values
    X_te = test[features].astype(int).values
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_tr, y_tr)
    probs = clf.predict_proba(X_te)[:, 1]
    return probs, test["label"].astype(int).values


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    if len(np.unique(y_true)) < 2:
        return {
            "pr_auc": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "prec_curve": np.array([]),
            "rec_curve": np.array([]),
        }
    pr_auc = average_precision_score(y_true, y_prob)
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
    y_hat = (y_prob >= 0.5).astype(int)
    return {
        "pr_auc": float(pr_auc),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "prec_curve": prec_curve,
        "rec_curve": rec_curve,
    }


def plot_curves(curves: Dict[str, Tuple[np.ndarray, np.ndarray]], path: str, title: str) -> None:
    set_plot_style()
    plt.figure(figsize=(6, 4))
    for label, (rec, prec) in curves.items():
        color = MODEL_COLORS.get(label)
        plt.plot(rec, prec, label=label, color=color)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="upper right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()


def main() -> None:
    df = load_data()
    gum, ewt = split_corpora(df)
    rows: List[Dict[str, float]] = []

    # Evaluate GUM -> EWT
    curves = {}
    for name, feats in FEATURE_SETS.items():
        if len(np.unique(gum["label"])) < 2:
            metrics = {
                "pr_auc": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "prec_curve": np.array([]),
                "rec_curve": np.array([]),
            }
        else:
            probs, y_true = train_predict(gum, ewt, feats)
            metrics = compute_metrics(y_true, probs)
        rows.append({
            "train": "gum",
            "test": "ewt",
            "model": name,
            "n_train": int(len(gum)),
            "n_test": int(len(ewt)),
            "pos_train": int(gum["label"].sum()),
            "pos_test": int(ewt["label"].sum()),
            "pr_auc": metrics["pr_auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        })
        if metrics["rec_curve"].size and metrics["prec_curve"].size:
            curves[name] = (metrics["rec_curve"], metrics["prec_curve"])
    if curves:
        plot_curves(curves, os.path.join("images", "let_alone_prcurve.pdf"), "PR curves: GUM -> EWT")

    # Evaluate EWT -> GUM
    curves = {}
    for name, feats in FEATURE_SETS.items():
        if len(np.unique(ewt["label"])) < 2:
            metrics = {
                "pr_auc": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "prec_curve": np.array([]),
                "rec_curve": np.array([]),
            }
        else:
            probs, y_true = train_predict(ewt, gum, feats)
            metrics = compute_metrics(y_true, probs)
        rows.append({
            "train": "ewt",
            "test": "gum",
            "model": name,
            "n_train": int(len(ewt)),
            "n_test": int(len(gum)),
            "pos_train": int(ewt["label"].sum()),
            "pos_test": int(gum["label"].sum()),
            "pr_auc": metrics["pr_auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        })
        if metrics["rec_curve"].size and metrics["prec_curve"].size:
            curves[name] = (metrics["rec_curve"], metrics["prec_curve"])
    if curves:
        plot_curves(curves, os.path.join("images", "appendix", "let_alone_prcurve_ewt2gum.png"), "PR curves: EWT -> GUM")

    os.makedirs(os.path.dirname(OUT_EVAL), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_EVAL, index=False)
    print(f"Saved evaluation metrics to {OUT_EVAL}")

    # Save top misclassifications for GUM->EWT (full model) if estimable
    if len(np.unique(gum["label"])) >= 2:
        probs, y_true = train_predict(gum, ewt, FEATURE_SETS["full"])
        errors = ewt.copy()
        errors["pred_prob"] = probs
        errors["pred_label"] = (probs >= 0.5).astype(int)
        errors = errors[errors["pred_label"] != errors["label"]].copy()
        errors = errors.sort_values("pred_prob", ascending=False).head(25)
        out_cols = ["corpus", "sent_id", "text", "label", "pred_label", "pred_prob"]
        errors[out_cols].to_csv(OUT_ERRS, sep="\t", index=False)
        print(f"Saved error analysis to {OUT_ERRS}")


if __name__ == "__main__":
    main()
