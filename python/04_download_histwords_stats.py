#!/usr/bin/env python3
"""
Download and extract HistWords COHA lemma stats (word lists, volstats, freqs).

This pulls the full archive but extracts only the metadata needed to select
non-stop, non-proper words and drift scores.
"""

from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://snap.stanford.edu/historical_embeddings/coha-lemma.zip"
DEST_DIR = Path("data/histwords_full")
ZIP_PATH = DEST_DIR / "coha-lemma.zip"
EXTRACT_DIR = DEST_DIR / "coha-lemma"

KEEP_PREFIXES = (
    "word_lists/",
    "volstats/",
    "pos/",
)
KEEP_FILES = (
    "freqs.pkl",
    "avg_freqs.pkl",
)


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(100.0, downloaded * 100.0 / total_size)
    mb = downloaded / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    sys.stderr.write(f"\r[download] {pct:5.1f}% ({mb:,.1f}MB/{total_mb:,.1f}MB)")
    sys.stderr.flush()


def should_extract(name: str) -> bool:
    if any(name.endswith(suffix) for suffix in KEEP_FILES):
        return True
    return any(name.startswith(prefix) for prefix in KEEP_PREFIXES)


def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    if EXTRACT_DIR.exists():
        if (EXTRACT_DIR / "word_lists").exists() and (EXTRACT_DIR / "volstats").exists():
            print(f"[skip] {EXTRACT_DIR} already has stats")
            return

    if not ZIP_PATH.exists():
        print(f"[download] {URL}")
        urllib.request.urlretrieve(URL, ZIP_PATH.as_posix(), _progress)
        sys.stderr.write("\n")
    else:
        print(f"[skip] {ZIP_PATH} already downloaded")

    print(f"[extract] stats -> {EXTRACT_DIR}")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        for name in zf.namelist():
            # Normalize any leading folder (e.g., coha-lemma/word_lists/...)
            trimmed = name
            if trimmed.startswith("coha-lemma/"):
                trimmed = trimmed[len("coha-lemma/") :]
            if trimmed == "" or trimmed.endswith("/"):
                continue
            if should_extract(trimmed):
                target_path = EXTRACT_DIR / trimmed
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, open(target_path, "wb") as dst:
                    dst.write(src.read())

    print("[done] histwords stats ready")


if __name__ == "__main__":
    main()
