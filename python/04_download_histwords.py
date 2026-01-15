#!/usr/bin/env python3
"""
Download and unpack HistWords COHA lemma SGNS embeddings.

This uses the smaller SGNS-only archive to avoid multi-GB downloads.
Outputs are stored under data/histwords/coha-lemma_sgns/.
"""

from __future__ import annotations

import sys
import time
import urllib.request
import zipfile
from pathlib import Path

URL = "https://snap.stanford.edu/historical_embeddings/coha-lemma_sgns.zip"
DEST_DIR = Path("data/histwords")
ZIP_PATH = DEST_DIR / "coha-lemma_sgns.zip"
EXTRACT_DIR = DEST_DIR / "coha-lemma_sgns"


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(100.0, downloaded * 100.0 / total_size)
    mb = downloaded / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    sys.stderr.write(f"\r[download] {pct:5.1f}% ({mb:,.1f}MB/{total_mb:,.1f}MB)")
    sys.stderr.flush()


def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    if EXTRACT_DIR.exists():
        # Assume already extracted if sgns folder exists.
        if (EXTRACT_DIR / "sgns").exists():
            print(f"[skip] {EXTRACT_DIR} already present")
            return

    if not ZIP_PATH.exists():
        print(f"[download] {URL}")
        urllib.request.urlretrieve(URL, ZIP_PATH.as_posix(), _progress)
        sys.stderr.write("\n")
    else:
        print(f"[skip] {ZIP_PATH} already downloaded")

    print(f"[extract] {ZIP_PATH} -> {EXTRACT_DIR}")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(EXTRACT_DIR)

    # Some archives nest an extra folder; normalize if needed.
    nested = EXTRACT_DIR / "coha-lemma_sgns"
    if nested.exists() and (nested / "sgns").exists():
        for item in nested.iterdir():
            target = EXTRACT_DIR / item.name
            if target.exists():
                continue
            item.rename(target)
        try:
            nested.rmdir()
        except OSError:
            pass

    print("[done] histwords embeddings ready")


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"[timing] {elapsed:0.1f}s")
