# Repository Guidelines

## Project Structure & Module Organization
- `main.tex` is the primary manuscript; `house-style-and-preamble.tex` holds shared LaTeX setup and macros.
- `refs.bib` is the bibliography; `images/` stores figures referenced by the paper; `data/` stores tabular outputs.
- `python/` contains numbered analysis scripts (`01_...`, `02_...`, etc.); `notes/` contains working notes.
- Generated artifacts include `main.pdf` and auxiliary LaTeX files (`.aux`, `.bcf`, `.log`, `.run.xml`).

## Build, Test, and Development Commands
- Build the PDF (requires `biber` and `biblatex-unified`):
  `xelatex main.tex && biber main && xelatex main.tex && xelatex main.tex`
- Phoneme pipeline (run from repo root so outputs land in top-level folders):
  `python python/01_download_phoible.py`, `python python/02_make_ridgelines.py`, `python python/03_model_y.py`
- Construction pipeline (flagship *or even*):
  `python python/10_download_ud.py`, `python python/11_extract_or_even.py`, `python python/12_or_even_profile_plot.py`, `python python/13_or_even_prcurve.py`
- Outputs: tables in `out/` (created by scripts) and figures in `images/` or `figs/` depending on the script.

## Coding Style & Naming Conventions
- LaTeX: keep manuscript text in `main.tex`, macros in `house-style-and-preamble.tex`, and cite with `\citep`/`\citet` using keys from `refs.bib`.
- Python: 4-space indentation, snake_case names, module docstring header. Script names follow `NN_description.py`.
- Prefer editing the canonical scripts (no `(1)` suffix or `.bak`), which are retained as historical backups.

## Testing Guidelines
- No automated test suite. Validate changes by re-running the relevant scripts and rebuilding `main.pdf`, then spot-check the referenced figures/tables.

## Commit & Pull Request Guidelines
- Commit messages follow short, imperative summaries (e.g., `Add Gelman-style checks for construction battery`, `Fix construction battery LaTeX tables`). Keep them concise and action-led.
- In PRs, describe the analytical change, list regenerated outputs (tables/figures), and note any dataset version updates.

## Data & Reproducibility Notes
- Scripts download external datasets (e.g., PHOIBLE, Universal Dependencies). Ensure network access and record version/pinning decisions in `STATUS.md` when changes affect results.
