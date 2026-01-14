#!/usr/bin/env python3
"""
Download UD treebanks for a set of English UD corpora.

This script retrieves version‑controlled CoNLL‑U files and accompanying
license/README texts from the official Universal Dependencies GitHub
repositories.  We pin the downloads to a specific release tag (``r2.16``)
to ensure reproducibility.  Files are saved under ``data/ud/{gum,ewt}/``.

The script attempts to use the internal API tool (available at
``http://localhost:8674``) to fetch files via the GitHub connector.  If the
API tool is unavailable or fails, it falls back to downloading from
``raw.githubusercontent.com``.  See ``DATA_SOURCES.md`` for licensing
details.

Usage (from repository root)::

    python3 src/10_download_ud.py

Each file is only downloaded if it does not already exist.  Progress
messages are printed to standard output.  Any unrecoverable errors
will terminate the script with a non‑zero exit code.
"""

from __future__ import annotations

import json
import os
import sys
import requests
from typing import Optional, List, Tuple

# UD release tag to use. Pinning ensures the same corpus version is
# downloaded across runs. Update this value to use a newer release.
VERSION = "r2.16"

# Fully qualified repository names.  Keys correspond to the short
# corpus identifiers used in downstream code.
REPOS = {
    "gum": "UniversalDependencies/UD_English-GUM",
    "ewt": "UniversalDependencies/UD_English-EWT",
    "gumreddit": "UniversalDependencies/UD_English-GUMReddit",
    "esl": "UniversalDependencies/UD_English-ESL",
    "eslspok": "UniversalDependencies/UD_English-ESLSpok",
    "gentle": "UniversalDependencies/UD_English-GENTLE",
    "childes": "UniversalDependencies/UD_English-CHILDES",
    "lines": "UniversalDependencies/UD_English-LinES",
    "pud": "UniversalDependencies/UD_English-PUD",
    "partut": "UniversalDependencies/UD_English-ParTUT",
    "unidive": "UniversalDependencies/UD_English-UniDive",
    "atis": "UniversalDependencies/UD_English-Atis",
    "littleprince": "UniversalDependencies/UD_English-LittlePrince",
    "pronouns": "UniversalDependencies/UD_English-Pronouns",
    "ctetex": "UniversalDependencies/UD_English-CTeTex",
    "pcedt": "UniversalDependencies/UD_English-PCEDT",
}

# Optional explicit file lists per corpus. When absent, files are discovered
# via the GitHub API and all root-level .conllu files are downloaded along
# with LICENSE/README documents if present.
FILES: dict = {}


def get_default_branch(repo: str, session: requests.Session) -> str:
    """Fetch the default branch name for a GitHub repo."""
    resp = session.get(f"https://api.github.com/repos/{repo}", timeout=30)
    if resp.status_code == 200:
        data = resp.json()
        return data.get("default_branch", "master")
    return "master"


def list_repo_files(repo: str, session: requests.Session) -> Tuple[List[str], str]:
    """List .conllu files (and root README/LICENSE) for a repo, with fallback.

    Returns (files, ref_used) where files may include subdirectory paths.
    """
    def fetch_contents(path: str, ref: Optional[str]) -> List[dict]:
        url = f"https://api.github.com/repos/{repo}/contents"
        if path:
            url = f"{url}/{path}"
        params = {"ref": ref} if ref else None
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            return []
        data = resp.json()
        return data if isinstance(data, list) else []

    def walk(path: str, ref: Optional[str], depth: int, max_depth: int, files: List[str]) -> None:
        entries = fetch_contents(path, ref)
        for entry in entries:
            etype = entry.get("type")
            name = entry.get("name", "")
            lower = name.lower()
            entry_path = entry.get("path", name)
            if etype == "file":
                if lower.endswith(".conllu"):
                    files.append(entry_path)
                elif depth == 0 and (lower.startswith("readme") or lower.startswith("license")):
                    files.append(entry_path)
            elif etype == "dir" and depth < max_depth:
                # Skip non-release and large auxiliary directories
                if lower in {"not-to-release", "not_to_release"}:
                    continue
                walk(entry_path, ref, depth + 1, max_depth, files)

    for ref in (VERSION, None):
        files: List[str] = []
        root_entries = fetch_contents("", ref)
        root_files: List[str] = []
        for entry in root_entries:
            if entry.get("type") != "file":
                continue
            name = entry.get("name", "")
            lower = name.lower()
            if lower.endswith(".conllu") or lower.startswith("readme") or lower.startswith("license"):
                root_files.append(entry.get("path", name))
        # If we already have root-level .conllu files, use them and avoid recursion.
        if any(f.lower().endswith(".conllu") for f in root_files):
            files = root_files
        else:
            walk("", ref, 0, 2, files)
        if files:
            if ref is None:
                default_branch = get_default_branch(repo, session)
                print(f"[WARN] {repo} has no tag {VERSION}; using default branch '{default_branch}'.")
                return files, default_branch
            return files, ref
    return [], ""


def ensure_dir(path: str) -> None:
    """Create the parent directory for a given file path if needed."""
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def download_from_raw(repo: str, filename: str, dest_path: str, ref: str) -> bool:
    """
    Attempt to download a file using the raw.githubusercontent.com URL.

    Parameters
    ----------
    repo : str
        Fully qualified repository name (``owner/repo``).
    filename : str
        Path to the file within the repository root.
    dest_path : str
        Local filesystem path where the file should be saved.

    Returns
    -------
    bool
        ``True`` if the file was successfully downloaded and saved; ``False``
        otherwise.
    """
    raw_url = f"https://raw.githubusercontent.com/{repo}/{ref}/{filename}"
    try:
        print(f"[RAW]  {raw_url}")
        resp = requests.get(raw_url, timeout=60)
        if resp.status_code == 200:
            ensure_dir(dest_path)
            with open(dest_path, "wb") as fh:
                fh.write(resp.content)
            print(f"[OK]   Saved to {dest_path}")
            return True
        else:
            print(f"[WARN] Raw download returned status {resp.status_code} for {filename}")
            return False
    except Exception as exc:
        print(f"[WARN] Raw download failed for {filename}: {exc}")
        return False


def fetch_api_tool_name(session: requests.Session, action_keyword: str) -> Optional[str]:
    """
    Query the API tool registry to find the fully qualified name of a GitHub
    action (e.g. ``fetch_file`` or ``fetch_blob``).

    Parameters
    ----------
    session : requests.Session
        A persistent HTTP session for API requests.
    action_keyword : str
        A substring that should occur in the action name (e.g., ``fetch_file``).

    Returns
    -------
    Optional[str]
        The full action name (including the ``/GitHub/...`` prefix) if found,
        otherwise ``None``.
    """
    try:
        query_url = "http://localhost:8674/search_available_apis"
        params = {"query": action_keyword}
        resp = session.get(query_url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # The API returns a list of entries under ``search_available_apis``
        for entry in data:
            name = entry.get("name")
            if name and name.startswith("/GitHub/") and action_keyword in name:
                return name
        return None
    except Exception as exc:
        print(f"[WARN] Could not query API tool for {action_keyword}: {exc}")
        return None


def download_via_api(repo: str, filename: str, dest_path: str, session: requests.Session, ref: str) -> bool:
    """
    Download a file from GitHub using the internal API tool (if available).

    This function performs two API calls: first to ``fetch_file`` to retrieve
    the blob SHA, and then to ``fetch_blob`` to obtain the file content.

    Parameters
    ----------
    repo : str
        Fully qualified repository name (``owner/repo``).
    filename : str
        File to retrieve from the repository root.
    dest_path : str
        Destination path on the local filesystem.
    session : requests.Session
        Shared HTTP session to reuse connections.

    Returns
    -------
    bool
        ``True`` if the download succeeded; ``False`` otherwise.
    """
    # Look up API endpoints lazily; this avoids repeated network calls for
    # each file if the API tool is unavailable.  If either endpoint is
    # missing, we bail out immediately.
    fetch_file_name = fetch_api_tool_name(session, "fetch_file")
    fetch_blob_name = fetch_api_tool_name(session, "fetch_blob")
    if not fetch_file_name or not fetch_blob_name:
        return False
    try:
        # Step 1: request file metadata to obtain the blob SHA
        meta_params = {
            "name": fetch_file_name,
            "params": json.dumps({
                "repository_full_name": repo,
                "path": filename,
                "ref": ref,
            }),
        }
        meta_resp = session.get(
            "http://localhost:8674/call_api", params=meta_params, timeout=60
        )
        meta_resp.raise_for_status()
        meta_json = meta_resp.json()
        sha = meta_json.get("result", {}).get("sha")
        if not sha:
            print(f"[WARN] No SHA returned for {filename} via API tool")
            return False
        # Step 2: request the blob content itself
        blob_params = {
            "name": fetch_blob_name,
            "params": json.dumps({
                "repository_full_name": repo,
                "blob_sha": sha,
            }),
        }
        blob_resp = session.get(
            "http://localhost:8674/call_api", params=blob_params, timeout=300
        )
        blob_resp.raise_for_status()
        blob_json = blob_resp.json()
        content = blob_json.get("result", {}).get("content")
        if not isinstance(content, str):
            print(f"[WARN] Invalid content returned for {filename} via API tool")
            return False
        ensure_dir(dest_path)
        with open(dest_path, "w", encoding="utf-8") as fh:
            fh.write(content)
        print(f"[OK]   Saved to {dest_path} via API tool")
        return True
    except Exception as exc:
        print(f"[WARN] API tool download failed for {filename}: {exc}")
        return False


def main() -> None:
    """Entry point: download all required files for both treebanks."""
    session = requests.Session()
    for corpus, repo in REPOS.items():
        dest_dir = os.path.join("data", "ud", corpus)
        if os.path.isdir(dest_dir) and any(fn.endswith(".conllu") for fn in os.listdir(dest_dir)):
            print(f"[SKIP] data/ud/{corpus} already has .conllu files")
            continue
        file_list = FILES.get(corpus)
        ref_used = VERSION
        if not file_list:
            file_list, ref_used = list_repo_files(repo, session)
        if not file_list or not ref_used:
            print(f"[WARN] No file list available for {corpus} ({repo}); skipping.")
            continue
        for filename in file_list:
            dest_path = os.path.join(dest_dir, filename)
            # Skip if file already exists and is non‑empty
            if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
                print(f"[SKIP] {dest_path} already exists")
                continue
            # Try the API tool first
            ok = download_via_api(repo, filename, dest_path, session, ref_used)
            if ok:
                continue
            # Fallback: raw.githubusercontent.com
            ok = download_from_raw(repo, filename, dest_path, ref_used)
            if not ok:
                print(
                    f"[ERROR] Failed to download {filename} from both API tool and raw URLs",
                    file=sys.stderr,
                )
                sys.exit(1)
    print("All files downloaded.")


if __name__ == "__main__":  # pragma: no cover
    main()
