#!/usr/bin/env python3
"""
Multi-lexeme drift evaluation using HistWords COHA lemma SGNS embeddings.

Outputs:
  out/word_drift_lexemes.csv
  out/word_drift_summary.csv
  out/word_drift_summary.tex
  out/word_drift_lexemes.tex
  out/word_drift_shuffle.csv
"""

from __future__ import annotations

import math
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


DATA_DIR = Path("data/histwords/coha-lemma_sgns/sgns")
STATS_DIR = Path("data/histwords_full/coha-lemma")
VOLSTATS_PATH = STATS_DIR / "volstats" / "vols.pkl"
FREQS_PATH = STATS_DIR / "freqs.pkl"
OUT_DIR = Path("out")

DECADES = list(range(1900, 2010, 10))  # 1900..2000
TRAIN_DECADES = list(range(1900, 1950, 10))  # 1900..1940
TEST_DECADES = list(range(1950, 2010, 10))  # 1950..2000

N_TARGETS = 20
N_CONTROLS = 20
K_NEIGHBORS = 50
SEED = 42

FORCE_INCLUDE = {"egregious", "gay", "awful", "nice", "broadcast", "sick"}
MIN_AVG_FREQ = 1e-6
TARGET_POS = {"ADJ"}


@dataclass
class DecadeData:
    vocab: List[str]
    index: Dict[str, int]
    vectors: np.ndarray
    norms: np.ndarray


def load_decade(decade: int) -> DecadeData:
    vocab_path = DATA_DIR / f"{decade}-vocab.pkl"
    vec_path = DATA_DIR / f"{decade}-w.npy"
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    vectors = np.load(vec_path, mmap_mode="r")
    norms = np.linalg.norm(vectors, axis=1)
    index = {w: i for i, w in enumerate(vocab)}
    return DecadeData(vocab=vocab, index=index, vectors=vectors, norms=norms)


def cosine(vec_a: np.ndarray, vec_b: np.ndarray, norm_a: float, norm_b: float) -> float:
    denom = norm_a * norm_b
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def eligible_word(word: str) -> bool:
    return word.isalpha() and len(word) >= 3


def nonzero_word(word: str, data: Dict[int, DecadeData]) -> bool:
    for d in DECADES:
        dd = data[d]
        if dd.norms[dd.index[word]] == 0:
            return False
    return True


def load_volstats_words() -> set:
    if not VOLSTATS_PATH.exists():
        return set()
    vols = pickle.load(open(VOLSTATS_PATH, "rb"), encoding="latin1")
    return set(vols.keys())


def load_avg_freq(words: Iterable[str]) -> Dict[str, float]:
    if not FREQS_PATH.exists():
        return {}
    freqs = pickle.load(open(FREQS_PATH, "rb"), encoding="latin1")
    avg = {}
    for word in words:
        series = freqs.get(word, {})
        avg[word] = float(np.mean([series.get(d, 0.0) for d in DECADES]))
    return avg


def coarse_pos(tag: str) -> str:
    if not tag:
        return "OTHER"
    tag = tag.lower()
    if tag.startswith("n"):
        return "N"
    if tag.startswith("v"):
        return "V"
    if tag.startswith("j"):
        return "ADJ"
    if tag.startswith("r"):
        return "ADV"
    return "OTHER"


def load_pos(words: Iterable[str]) -> Dict[str, str]:
    pos_counts: Dict[str, Counter] = {w: Counter() for w in words}
    for decade in DECADES:
        pos_path = STATS_DIR / "pos" / f"{decade}-pos_counts.pkl"
        if not pos_path.exists():
            continue
        data = pickle.load(open(pos_path, "rb"), encoding="latin1")
        for word in pos_counts:
            if word in data:
                pos_counts[word].update(data[word])
    pos_map = {}
    for word, counts in pos_counts.items():
        if not counts:
            pos_map[word] = "OTHER"
            continue
        top_tag = counts.most_common(1)[0][0]
        pos_map[word] = coarse_pos(top_tag)
    return pos_map


def drift_score(word: str, data: Dict[int, DecadeData]) -> float:
    cosines = []
    for d1, d2 in zip(DECADES[:-1], DECADES[1:]):
        dd1, dd2 = data[d1], data[d2]
        i1, i2 = dd1.index[word], dd2.index[word]
        v1, v2 = dd1.vectors[i1], dd2.vectors[i2]
        c = cosine(v1, v2, dd1.norms[i1], dd2.norms[i2])
        cosines.append(c)
    return float(np.mean([1.0 - c for c in cosines]))


def neighbor_set(word: str, decade: int, data: Dict[int, DecadeData], k: int) -> List[str]:
    dd = data[decade]
    idx = dd.index[word]
    vec = dd.vectors[idx]
    sims = dd.vectors @ vec
    denom = dd.norms * dd.norms[idx]
    denom = np.where(denom == 0, 1.0, denom)
    sims = sims / denom
    sims[idx] = -np.inf
    top_idx = np.argpartition(-sims, k)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return [dd.vocab[i] for i in top_idx]


def neighbor_overlap(word: str, data: Dict[int, DecadeData], k: int) -> float:
    overlaps = []
    prev = None
    for decade in DECADES:
        current = set(neighbor_set(word, decade, data, k))
        if prev is not None:
            overlaps.append(len(prev & current) / k)
        prev = current
    return float(np.mean(overlaps))


def self_cosine_mean(word: str, data: Dict[int, DecadeData]) -> float:
    vals = []
    for d1, d2 in zip(DECADES[:-1], DECADES[1:]):
        dd1, dd2 = data[d1], data[d2]
        i1, i2 = dd1.index[word], dd2.index[word]
        v1, v2 = dd1.vectors[i1], dd2.vectors[i2]
        vals.append(cosine(v1, v2, dd1.norms[i1], dd2.norms[i2]))
    return float(np.mean(vals))


def normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n == 0:
        return vec
    return vec / n


def build_prototypes(words: List[str], data: Dict[int, DecadeData]) -> np.ndarray:
    protos = []
    for word in words:
        vecs = []
        for d in TRAIN_DECADES:
            dd = data[d]
            idx = dd.index[word]
            vecs.append(normalize(dd.vectors[idx]))
        proto = np.mean(vecs, axis=0)
        protos.append(normalize(proto))
    return np.vstack(protos)


def classify_words(
    words: List[str], data: Dict[int, DecadeData], protos: np.ndarray
) -> Tuple[pd.DataFrame, float]:
    label_to_idx = {w: i for i, w in enumerate(words)}
    y_true = []
    y_pred = []
    per_word_correct = {w: 0 for w in words}
    per_word_total = {w: 0 for w in words}

    for word in words:
        for d in TEST_DECADES:
            dd = data[d]
            idx = dd.index[word]
            vec = normalize(dd.vectors[idx])
            sims = protos @ vec
            pred_idx = int(np.argmax(sims))
            pred_word = words[pred_idx]
            y_true.append(label_to_idx[word])
            y_pred.append(pred_idx)
            per_word_total[word] += 1
            if pred_word == word:
                per_word_correct[word] += 1

    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    acc_rows = []
    for word in words:
        acc = per_word_correct[word] / max(1, per_word_total[word])
        acc_rows.append({"word": word, "project_acc": acc})
    return pd.DataFrame(acc_rows), macro_f1


def temporal_mae(word: str, data: Dict[int, DecadeData]) -> float:
    vecs = []
    for d in DECADES:
        dd = data[d]
        idx = dd.index[word]
        vecs.append(normalize(dd.vectors[idx]))
    mat = np.vstack(vecs)
    sims = mat @ mat.T
    np.fill_diagonal(sims, -np.inf)
    best_idx = np.argmax(sims, axis=1)
    errors = [abs(DECADES[i] - DECADES[j]) / 10.0 for i, j in enumerate(best_idx)]
    return float(np.mean(errors))


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 1000) -> Tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = samples.mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> None:
    if not DATA_DIR.exists():
        raise SystemExit("HistWords data not found. Run python/04_download_histwords.py first.")
    if not VOLSTATS_PATH.exists():
        raise SystemExit("HistWords stats not found. Run python/04_download_histwords_stats.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    data = {d: load_decade(d) for d in DECADES}
    common_vocab = set(data[DECADES[0]].vocab)
    for d in DECADES[1:]:
        common_vocab &= set(data[d].vocab)

    vol_words = load_volstats_words()
    eligible = [
        w
        for w in common_vocab
        if eligible_word(w) and nonzero_word(w, data) and w in vol_words
    ]
    avg_freqs = load_avg_freq(eligible)
    pos_map = load_pos(eligible)
    stats = []
    for word in eligible:
        avg_freq = avg_freqs.get(word, 0.0)
        if avg_freq <= 0:
            continue
        stats.append(
            {
                "word": word,
                "avg_freq": avg_freq,
                "log_freq": math.log10(max(avg_freq, 1e-12)),
                "drift_score": drift_score(word, data),
                "length": len(word),
                "pos": pos_map.get(word, "OTHER"),
            }
        )
    stats_df = pd.DataFrame(stats)

    pool = stats_df[
        (stats_df["avg_freq"] >= MIN_AVG_FREQ) & (stats_df["pos"].isin(TARGET_POS))
    ].copy()
    top_cut = pool["drift_score"].quantile(0.9)
    bot_cut = pool["drift_score"].quantile(0.1)

    targets_pool = pool[pool["drift_score"] >= top_cut].copy()
    targets_pool = targets_pool.sort_values("drift_score", ascending=False)
    forced_keep = [w for w in FORCE_INCLUDE if w in pool["word"].values]
    for forced in FORCE_INCLUDE:
        if forced in common_vocab and not nonzero_word(forced, data):
            print(f"[warn] {forced} excluded (zero vectors)")
        elif forced in common_vocab and forced not in pool["word"].values:
            print(f"[warn] {forced} excluded (pos/frequency filter)")
    targets = forced_keep + [w for w in targets_pool["word"].tolist() if w not in forced_keep]
    targets = targets[: N_TARGETS]

    controls_pool = pool[pool["drift_score"] <= bot_cut].copy()
    controls_pool = controls_pool.sort_values("drift_score", ascending=True)

    controls = []
    used_controls = set()
    for word in targets:
        t_row = pool[pool["word"] == word].iloc[0]
        candidates = controls_pool[
            (controls_pool["pos"] == t_row["pos"])
            & (abs(controls_pool["log_freq"] - t_row["log_freq"]) <= 0.5)
            & (abs(controls_pool["length"] - t_row["length"]) <= 3)
        ]
        if candidates.empty:
            candidates = controls_pool[
                (controls_pool["pos"] == t_row["pos"])
                & (abs(controls_pool["log_freq"] - t_row["log_freq"]) <= 0.7)
                & (abs(controls_pool["length"] - t_row["length"]) <= 4)
            ]
        if candidates.empty:
            candidates = controls_pool[
                (controls_pool["pos"] == t_row["pos"])
                & (abs(controls_pool["log_freq"] - t_row["log_freq"]) <= 1.0)
            ]
        if candidates.empty:
            candidates = controls_pool[controls_pool["pos"] == t_row["pos"]]
        if candidates.empty:
            continue
        if candidates.empty:
            continue
        candidate = candidates.iloc[0]
        if candidate["word"] in used_controls:
            # pick next available
            for _, row in candidates.iterrows():
                if row["word"] not in used_controls:
                    candidate = row
                    break
        if candidate["word"] in used_controls:
            continue
        controls.append(candidate["word"])
        used_controls.add(candidate["word"])
        if len(controls) >= N_CONTROLS:
            break

    words = targets + controls

    protos = build_prototypes(words, data)
    acc_df, macro_f1 = classify_words(words, data, protos)

    # Shuffle-label baseline
    shuffle_f1 = []
    for _ in range(200):
        perm = rng.permutation(len(words))
        shuffled = protos[perm]
        _, f1_val = classify_words(words, data, shuffled)
        shuffle_f1.append(f1_val)
    shuffle_f1 = np.array(shuffle_f1)
    shuffle_df = pd.DataFrame({"f1": shuffle_f1})
    shuffle_df.to_csv(OUT_DIR / "word_drift_shuffle.csv", index=False)

    rows = []
    for word in words:
        group = "target" if word in targets else "control"
        row = pool[pool["word"] == word].iloc[0]
        rows.append(
            {
                "word": word,
                "group": group,
                "drift_score": row["drift_score"],
                "avg_freq": row["avg_freq"],
                "log_freq": row["log_freq"],
                "pos": row["pos"],
                "self_cosine": self_cosine_mean(word, data),
                "neighbor_overlap": neighbor_overlap(word, data, K_NEIGHBORS),
                "temporal_mae": temporal_mae(word, data),
            }
        )
    metrics_df = pd.DataFrame(rows).merge(acc_df, on="word", how="left")
    metrics_df["group"] = pd.Categorical(metrics_df["group"], categories=["target", "control"], ordered=True)
    metrics_df = metrics_df.sort_values(["group", "drift_score"], ascending=[True, False])
    metrics_df.to_csv(OUT_DIR / "word_drift_lexemes.csv", index=False)

    summary_rows = []
    for group, gdf in metrics_df.groupby("group", observed=True):
        for metric in ["neighbor_overlap", "self_cosine", "project_acc", "temporal_mae"]:
            values = gdf[metric].to_numpy()
            lo, hi = bootstrap_ci(values, rng)
            summary_rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "word_drift_summary.csv", index=False)

    # Build LaTeX summary table
    summary_pivot = summary_df.pivot(index="group", columns="metric", values="mean")
    summary_ci_low = summary_df.pivot(index="group", columns="metric", values="ci_low")
    summary_ci_high = summary_df.pivot(index="group", columns="metric", values="ci_high")

    def fmt(group: str, metric: str) -> str:
        mean = summary_pivot.loc[group, metric]
        lo = summary_ci_low.loc[group, metric]
        hi = summary_ci_high.loc[group, metric]
        return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"

    summary_table = pd.DataFrame(
        {
            "Group": ["Targets", "Controls"],
            "n": [
                int(summary_df[summary_df["group"] == "target"]["n"].iloc[0]),
                int(summary_df[summary_df["group"] == "control"]["n"].iloc[0]),
            ],
            "Neighbor overlap": [fmt("target", "neighbor_overlap"), fmt("control", "neighbor_overlap")],
            "Self cosine": [fmt("target", "self_cosine"), fmt("control", "self_cosine")],
            "Project acc": [fmt("target", "project_acc"), fmt("control", "project_acc")],
            "Temporal MAE": [fmt("target", "temporal_mae"), fmt("control", "temporal_mae")],
        }
    )

    tex = summary_table.to_latex(index=False, escape=True)
    shuffle_mean = float(shuffle_f1.mean())
    shuffle_lo, shuffle_hi = np.quantile(shuffle_f1, [0.025, 0.975])
    caption = (
        "  \\caption{Word-level drift summary (COHA lemma SGNS, 1900--2000). "
        "Cells show mean with 95\\% bootstrap CI. "
        "Project acc is per-lexeme word-identity accuracy on held-out decades; "
        f"macro-F1 = {macro_f1:.3f}; "
        f"shuffled-label baseline mean = {shuffle_mean:.3f} "
        f"(95\\% CI [{shuffle_lo:.3f}, {shuffle_hi:.3f}])."
        "}"
    )
    tex_lines = [
        "% Auto-generated by python/05_word_drift_multilexeme.py",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\small",
        caption,
        "  \\label{tab:word-drift-summary}",
        "  " + tex.replace("\\begin{tabular}", "\\begin{tabular}").strip(),
        "\\end{table}",
        "",
    ]
    (OUT_DIR / "word_drift_summary.tex").write_text("\n".join(tex_lines))

    # Lexeme table (top 40)
    display_cols = [
        "word",
        "group",
        "pos",
        "avg_freq",
        "drift_score",
        "neighbor_overlap",
        "self_cosine",
        "project_acc",
        "temporal_mae",
    ]
    lex_df = metrics_df[display_cols].copy()
    lex_df["avg_freq"] = lex_df["avg_freq"].map(lambda x: f"{x:.2e}")
    lex_df["drift_score"] = lex_df["drift_score"].map(lambda x: f"{x:.3f}")
    lex_df["neighbor_overlap"] = lex_df["neighbor_overlap"].map(lambda x: f"{x:.3f}")
    lex_df["self_cosine"] = lex_df["self_cosine"].map(lambda x: f"{x:.3f}")
    lex_df["project_acc"] = lex_df["project_acc"].map(lambda x: f"{x:.3f}")
    lex_df["temporal_mae"] = lex_df["temporal_mae"].map(lambda x: f"{x:.3f}")

    lex_tex = lex_df.to_latex(index=False, escape=True)
    lex_lines = [
        "% Auto-generated by python/05_word_drift_multilexeme.py",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\small",
        "  \\caption{Per-lexeme drift metrics (targets vs matched controls). Project accuracy is word-identity accuracy on held-out decades.}",
        "  \\label{tab:word-drift-lexemes}",
        "  " + lex_tex.replace("\\begin{tabular}", "\\begin{tabular}").strip(),
        "\\end{table}",
        "",
    ]
    (OUT_DIR / "word_drift_lexemes.tex").write_text("\n".join(lex_lines))

    print(f"[targets] {targets}")
    print(f"[controls] {controls}")
    print(f"[macro_f1] {macro_f1:.3f}")
    print(f"[shuffle_f1] mean={shuffle_f1.mean():.3f} ci=[{np.quantile(shuffle_f1,0.025):.3f},{np.quantile(shuffle_f1,0.975):.3f}]")


if __name__ == "__main__":
    main()
