#!/usr/bin/env python3
"""
Create the cue profile figure for the *or even* construction.

Reads `out/or_even_features.csv` and outputs figures to
`images/or_even_profile.pdf` and `images/or_even_profile.png`.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_SEED = 20250901
BOOTSTRAP_REPS = 1000


def bootstrap_ci(data: np.ndarray, reps: int = BOOTSTRAP_REPS, seed: int = RANDOM_SEED) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(data)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    means = np.empty(reps)
    for i in range(reps):
        sample = rng.choice(data, size=n, replace=True)
        means[i] = sample.mean()
    lower = np.percentile(means, 2.5)
    upper = np.percentile(means, 97.5)
    return (data.mean(), lower, upper)


def bootstrap_distribution(df: pd.DataFrame, column: str, categories: List[str], reps: int = BOOTSTRAP_REPS, seed: int = RANDOM_SEED) -> Dict[str, Tuple[float, float, float]]:
    rng = np.random.default_rng(seed)
    n = len(df)
    counts = df[column].value_counts().reindex(categories).fillna(0)
    props = counts / n if n > 0 else counts
    boot_means: Dict[str, List[float]] = {cat: [] for cat in categories}
    for _ in range(reps):
        sample_idxs = rng.choice(df.index, size=n, replace=True)
        sample = df.loc[sample_idxs, column]
        for cat in categories:
            boot_means[cat].append((sample == cat).mean() if n > 0 else 0.0)
    ci: Dict[str, Tuple[float, float, float]] = {}
    for cat in categories:
        arr = np.array(boot_means[cat])
        ci[cat] = (props[cat], np.percentile(arr, 2.5), np.percentile(arr, 97.5))
    return ci


def main() -> None:
    feat_path = os.path.join("out", "or_even_features.csv")
    df = pd.read_csv(feat_path)
    df["corpus"] = df["corpus"].astype(str)
    corpora = ["gum", "ewt", "gumreddit"]
    categories = ["VERB", "NOUN", "ADJ", "OTHER"]
    stats = {}
    for corpus in corpora:
        subset = df[df["corpus"] == corpus]
        parallel = subset["parallelism"].values.astype(float)
        parallel_mean, parallel_low, parallel_high = bootstrap_ci(parallel)
        licensing = subset["licensing"].values.astype(float)
        lic_mean, lic_low, lic_high = bootstrap_ci(licensing)
        dist_ci = bootstrap_distribution(subset, "upos_y", categories)
        stats[corpus] = {
            "parallel": (parallel_mean, parallel_low, parallel_high),
            "licensing": (lic_mean, lic_low, lic_high),
            "dist": dist_ci,
            "n": len(subset),
        }

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={'width_ratios': [1, 1.2, 1]})

    labels = [c.upper() for c in corpora]
    means = [stats[c]['parallel'][0] for c in corpora]
    lows = [stats[c]['parallel'][1] for c in corpora]
    highs = [stats[c]['parallel'][2] for c in corpora]
    yerr = [np.array(means) - np.array(lows), np.array(highs) - np.array(means)]
    axes[0].bar(labels, means, yerr=yerr, color=["#4C72B0", "#55A868", "#C44E52"], capsize=4)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Parallelism rate")
    axes[0].set_title("Parallelism")

    bottoms = np.zeros(len(corpora))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
    for i, cat in enumerate(categories):
        vals = [stats[c]['dist'][cat][0] for c in corpora]
        axes[1].bar(labels, vals, bottom=bottoms, color=colors[i], label=cat)
        bottoms += vals
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Proportion of Y-heads")
    axes[1].set_title("Distribution of Y-head UPOS")
    axes[1].legend(title="UPOS", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    lic_means = [stats[c]['licensing'][0] for c in corpora]
    lic_lows = [stats[c]['licensing'][1] for c in corpora]
    lic_highs = [stats[c]['licensing'][2] for c in corpora]
    lic_yerr = [np.array(lic_means) - np.array(lic_lows), np.array(lic_highs) - np.array(lic_means)]
    axes[2].bar(labels, lic_means, yerr=lic_yerr, color=["#4C72B0", "#55A868", "#C44E52"], capsize=4)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Scalar/contrast cue rate")
    axes[2].set_title("Scalar/contrast cues")

    for i, corpus in enumerate(corpora):
        n = stats[corpus]['n']
        axes[0].text(i, means[i] + 0.03, f"N={n}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    os.makedirs("images", exist_ok=True)
    fig_path_pdf = os.path.join("images", "or_even_profile.pdf")
    fig_path_png = os.path.join("images", "or_even_profile.png")
    fig.savefig(fig_path_pdf)
    fig.savefig(fig_path_png, dpi=300)
    print(f"Saved profile figure to {fig_path_pdf} and {fig_path_png}")


if __name__ == "__main__":
    main()
