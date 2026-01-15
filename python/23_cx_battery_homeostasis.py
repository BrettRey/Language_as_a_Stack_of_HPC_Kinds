#!/usr/bin/env python3
"""
Compute construction battery homeostasis diagnostics.

Outputs:
  - out/cx_battery_cue_cov.csv / .tex  (pairwise cue phi with bootstrap CIs)
  - out/cx_battery_downsample.csv / .tex (downsampling sensitivity on PR--AUC)
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

IN_CANDIDATES = os.path.join("out", "cx_battery_candidates.csv")
OUT_COV = os.path.join("out", "cx_battery_cue_cov.csv")
OUT_COV_TEX = os.path.join("out", "cx_battery_cue_cov.tex")
OUT_DOWNSAMPLE = os.path.join("out", "cx_battery_downsample.csv")
OUT_DOWNSAMPLE_TEX = os.path.join("out", "cx_battery_downsample.tex")

FEATURES = ["cue1", "cue2", "cue3"]
PAIRS = [("cue1", "cue2"), ("cue1", "cue3"), ("cue2", "cue3")]

MIN_CANDIDATES = 20
MIN_POSITIVES = 10

BOOTSTRAP_N = 500
BOOTSTRAP_SEED = 42
DOWN_FRAC = 0.25
DOWN_REPS = 100
DOWN_SEED = 123


def phi_coeff(x: np.ndarray, y: np.ndarray) -> float:
    a = int(np.sum((x == 1) & (y == 1)))
    b = int(np.sum((x == 1) & (y == 0)))
    c = int(np.sum((x == 0) & (y == 1)))
    d = int(np.sum((x == 0) & (y == 0)))
    denom = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if denom == 0:
        return float("nan")
    return float((a * d - b * c) / denom)


def bootstrap_phi(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan")
    vals: List[float] = []
    idx = np.arange(n)
    for _ in range(n_boot):
        sample = rng.choice(idx, size=n, replace=True)
        val = phi_coeff(x[sample], y[sample])
        if not np.isnan(val):
            vals.append(val)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def train_eval(train: pd.DataFrame, test: pd.DataFrame, features: List[str]) -> float:
    X_tr = train[features].astype(int).values
    y_tr = train["label"].astype(int).values
    X_te = test[features].astype(int).values
    y_te = test["label"].astype(int).values
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_tr, y_tr)
    probs = clf.predict_proba(X_te)[:, 1]
    if len(np.unique(y_te)) < 2:
        return 0.0
    return float(average_precision_score(y_te, probs))


def stratified_sample(df: pd.DataFrame, frac: float, rng: np.random.Generator) -> pd.DataFrame:
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return df.sample(frac=frac, replace=False, random_state=int(rng.integers(1, 1_000_000)))
    n_pos = max(1, int(round(len(pos) * frac)))
    n_neg = max(1, int(round(len(neg) * frac)))
    pos_idx = rng.choice(pos.index.to_numpy(), size=n_pos, replace=len(pos) < n_pos)
    neg_idx = rng.choice(neg.index.to_numpy(), size=n_neg, replace=len(neg) < n_neg)
    return df.loc[np.concatenate([pos_idx, neg_idx])]


def make_covariance(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for (construction, corpus), sub in df.groupby(["construction", "corpus"]):
        pos = sub[sub["label"] == 1]
        n_pos = len(pos)
        for c1, c2 in PAIRS:
            x = pos[c1].to_numpy(dtype=int)
            y = pos[c2].to_numpy(dtype=int)
            phi = phi_coeff(x, y)
            lo, hi = bootstrap_phi(x, y, BOOTSTRAP_N, BOOTSTRAP_SEED)
            rows.append({
                "construction": construction,
                "corpus": corpus,
                "pair": f"{c1}_{c2}",
                "n_pos": int(n_pos),
                "phi": phi,
                "ci_low": lo,
                "ci_high": hi,
            })
    return pd.DataFrame(rows).sort_values(["construction", "corpus", "pair"])


def make_downsample(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    corpora = sorted(df["corpus"].unique())
    constructions = sorted(df["construction"].unique())
    rng = np.random.default_rng(DOWN_SEED)
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
                if status != "ok" or len(np.unique(train["label"])) < 2 or len(np.unique(test["label"])) < 2:
                    rows.append({
                        "construction": construction,
                        "train": train_c,
                        "test": test_c,
                        "n_train": n_train,
                        "n_test": n_test,
                        "pos_train": pos_train,
                        "pos_test": pos_test,
                        "status": status,
                        "full_pr_auc": np.nan,
                        "down_pr_auc": np.nan,
                        "down_pr_std": np.nan,
                        "delta": np.nan,
                    })
                    continue
                full_pr = train_eval(train, test, FEATURES)
                vals: List[float] = []
                for _ in range(DOWN_REPS):
                    sampled = stratified_sample(train, DOWN_FRAC, rng)
                    if len(np.unique(sampled["label"])) < 2:
                        continue
                    vals.append(train_eval(sampled, test, FEATURES))
                if vals:
                    down_mean = float(np.mean(vals))
                    down_std = float(np.std(vals))
                    delta = float(full_pr - down_mean)
                else:
                    down_mean = float("nan")
                    down_std = float("nan")
                    delta = float("nan")
                rows.append({
                    "construction": construction,
                    "train": train_c,
                    "test": test_c,
                    "n_train": n_train,
                    "n_test": n_test,
                    "pos_train": pos_train,
                    "pos_test": pos_test,
                    "status": status,
                    "full_pr_auc": full_pr,
                    "down_pr_auc": down_mean,
                    "down_pr_std": down_std,
                    "delta": delta,
                })
    return pd.DataFrame(rows).sort_values(["construction", "train", "test"])


def fmt_float(val: float) -> str:
    if pd.isna(val):
        return "--"
    return f"{val:.3f}"


def latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def write_cov_tex(df: pd.DataFrame) -> None:
    lines = []
    lines.append("% Auto-generated by python/23_cx_battery_homeostasis.py")
    lines.append("\\begin{table}[t]")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append("  \\caption{Cue covariance (phi) among positive instances in the construction battery. Bootstrap 95\\% CIs are reported in the CSV.}")
    lines.append("  \\label{tab:cx-battery-cov}")
    lines.append("  \\begin{tabular}{l l r r r r}")
    lines.append("    \\toprule")
    lines.append("    Construction & Corpus & $n_{pos}$ & $\\phi_{12}$ & $\\phi_{13}$ & $\\phi_{23}$ \\\\")
    lines.append("    \\midrule")
    for (construction, corpus), sub in df.groupby(["construction", "corpus"]):
        n_pos = int(sub["n_pos"].iloc[0]) if len(sub) else 0
        if n_pos < MIN_POSITIVES:
            continue
        phis = {row["pair"]: row["phi"] for _, row in sub.iterrows()}
        row = [
            latex_escape(str(construction)),
            corpus.upper(),
            f"{n_pos}",
            fmt_float(phis.get("cue1_cue2", float("nan"))),
            fmt_float(phis.get("cue1_cue3", float("nan"))),
            fmt_float(phis.get("cue2_cue3", float("nan"))),
        ]
        lines.append("    " + " & ".join(row) + " \\\\")
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    with open(OUT_COV_TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_downsample_tex(df: pd.DataFrame) -> None:
    lines = []
    lines.append("% Auto-generated by python/23_cx_battery_homeostasis.py")
    lines.append("\\begin{table}[t]")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append("  \\caption{Downsampling sensitivity for estimable construction pairs (train set reduced to 25\\%).}")
    lines.append("  \\label{tab:cx-battery-downsample}")
    lines.append("  \\begin{tabular}{l l r r r r}")
    lines.append("    \\toprule")
    lines.append("    Construction & Direction & $n_{tr}$ & $n_{te}$ & PR--AUC & $\\Delta$ down \\\\")
    lines.append("    \\midrule")
    ok = df[df["status"] == "ok"]
    for _, row in ok.iterrows():
        direction = f"{row['train']}\\ensuremath{{\\to}}{row['test']}"
        n_tr = f"{int(row['n_train'])}({int(row['pos_train'])})"
        n_te = f"{int(row['n_test'])}({int(row['pos_test'])})"
        pr = fmt_float(row["full_pr_auc"])
        delta = fmt_float(row["delta"])
        construction = latex_escape(str(row["construction"]))
        lines.append(f"    {construction} & {direction} & {n_tr} & {n_te} & {pr} & {delta} \\\\")
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    with open(OUT_DOWNSAMPLE_TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    if not os.path.exists(IN_CANDIDATES):
        raise FileNotFoundError(IN_CANDIDATES)
    df = pd.read_csv(IN_CANDIDATES)
    cov = make_covariance(df)
    cov.to_csv(OUT_COV, index=False)
    write_cov_tex(cov)
    down = make_downsample(df)
    down.to_csv(OUT_DOWNSAMPLE, index=False)
    write_downsample_tex(down)
    print(f"Wrote {OUT_COV}, {OUT_COV_TEX}")
    print(f"Wrote {OUT_DOWNSAMPLE}, {OUT_DOWNSAMPLE_TEX}")


if __name__ == "__main__":
    main()
