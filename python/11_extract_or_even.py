#!/usr/bin/env python3
"""
Extract *or even* instances from UD treebanks and compute cue features.

Outputs:
* `out/or_even_features.csv` -- All extracted instances with features.
* `out/or_even_stats.csv` -- Summary statistics per corpus.
* `out/or_even_anchor_present_eval.csv` -- Anchor-present candidates with labels.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Any

import pandas as pd

# ensure python directory is on sys.path for utils_ud
sys.path.insert(0, os.path.dirname(__file__))
import utils_ud  # type: ignore


CORPORA = ["gum", "ewt", "gumreddit"]


def load_corpus_features(corpus: str) -> pd.DataFrame:
    base_dir = os.path.join("data", "ud", corpus)
    if not os.path.isdir(base_dir) or not any(fn.endswith(".conllu") for fn in os.listdir(base_dir)):
        print(f"[WARN] No .conllu files found for corpus '{corpus}'. Skipping.")
        return pd.DataFrame()
    files = [fn for fn in os.listdir(base_dir) if fn.endswith(".conllu")]
    rows: List[Dict[str, Any]] = []
    for filename in files:
        path = os.path.join(base_dir, filename)
        print(f"Parsing {path}...")
        sentences = utils_ud.load_conllu(path)
        for sent in sentences:
            feats = utils_ud.extract_or_even_features(sent)
            for row in feats:
                row["corpus"] = corpus
            rows.extend(feats)
    return pd.DataFrame(rows)


def load_corpus_candidates(corpus: str) -> pd.DataFrame:
    base_dir = os.path.join("data", "ud", corpus)
    if not os.path.isdir(base_dir) or not any(fn.endswith(".conllu") for fn in os.listdir(base_dir)):
        print(f"[WARN] No .conllu files found for corpus '{corpus}'. Skipping candidates.")
        return pd.DataFrame()
    files = [fn for fn in os.listdir(base_dir) if fn.endswith(".conllu")]
    rows: List[Dict[str, Any]] = []
    for filename in files:
        path = os.path.join(base_dir, filename)
        print(f"Parsing {path} for candidates...")
        sentences = utils_ud.load_conllu(path)
        for sent in sentences:
            feats = utils_ud.extract_or_even_candidates(sent)
            for row in feats:
                row["corpus"] = corpus
            rows.extend(feats)
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs("out", exist_ok=True)
    df_list = [load_corpus_features(c) for c in CORPORA]
    df_all = pd.concat([df for df in df_list if not df.empty], ignore_index=True)
    feat_path = os.path.join("out", "or_even_features.csv")
    if not df_all.empty:
        df_all.to_csv(feat_path, index=False)
        print(f"Saved features to {feat_path}")

    # summary stats per corpus
    rows = []
    for corpus in CORPORA:
        df = df_all[df_all.get("corpus") == corpus] if not df_all.empty else pd.DataFrame()
        n_tokens = len(df)
        parallelism_rate = df['parallelism'].mean() if n_tokens > 0 else 0.0
        licensing_rate = df['licensing'].mean() if n_tokens > 0 else 0.0
        rows.append({
            "corpus": corpus,
            "n_tokens": int(n_tokens),
            "parallelism_rate": parallelism_rate,
            "licensing_rate": licensing_rate,
        })
    stats_df = pd.DataFrame(rows)
    stats_path = os.path.join("out", "or_even_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved stats to {stats_path}")

    # candidates
    cand_list = [load_corpus_candidates(c) for c in CORPORA]
    cand_all = pd.concat([df for df in cand_list if not df.empty], ignore_index=True) if any(not df.empty for df in cand_list) else pd.DataFrame()
    cand_path = os.path.join("out", "or_even_anchor_present_eval.csv")
    if not cand_all.empty:
        cand_all.to_csv(cand_path, index=False)
        print(f"Saved anchor-present candidates to {cand_path}")
    else:
        print("[WARN] No candidates extracted; evaluation file not written.")


if __name__ == "__main__":
    main()
