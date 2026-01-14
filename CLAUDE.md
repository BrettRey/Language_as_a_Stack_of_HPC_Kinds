# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project Overview

**Title:** Language as a Stack of Homeostatic Property-Cluster Kinds: From Phonemes to Constructions
**Author:** Brett Reynolds
**Status:** Preprint published; empirical sections preliminary

This paper develops two operational diagnostics (projectibility and homeostasis) for deciding when linguistic categories warrant treatment as HPC kinds. It applies these to three levels: phonemes (PHOIBLE), words (semantic drift), and constructions (let alone).

## Build

```bash
# Requires biber backend and biblatex-unified style
xelatex main.tex && biber main && xelatex main.tex && xelatex main.tex
```

Note: Uses `biblatex-unified` style (linguistic journals), not APA.

## Structure

- `main.tex` - single-file article
- `refs.bib` - bibliography
- `house-style-and-preamble.tex` - LaTeX setup
- `python/` - analysis scripts
- `images/` - generated figures
- `data/` - CSV/TSV outputs

## Python Analysis

Scripts numbered by workflow order:

```bash
cd python
python 01_download_phoible.py   # Get PHOIBLE data
python 02_make_ridgelines.py    # Inventory size visualization
python 03_model_y.py            # /y/ scaling model

python 10_download_ud.py        # Universal Dependencies
python 11_extract_let_alone.py  # Extract construction instances
python 12_profile_plot.py       # Cue profile visualization
python 13_predict_prcurve.py    # Cross-corpus evaluation
```

## Key Concepts

- **Projectibility:** Out-of-sample prediction (PR-AUC, F1, cross-corpus transfer)
- **Homeostasis:** Identifiable mechanisms maintaining cluster (frequency, norms, cue redundancy)
- **Failure modes:** Thin (nonce), fat (cross-linguistic umbrella), negative (complement class)

## Connections to Other Projects

- HPC book Ch 12 uses similar phoneme/register cases
- `Labels_to_Stabilisers/` develops stabiliser taxonomy
- `linguistics-metrology/` applies measurement science

## Cautions

- Small sample sizes for construction case (GUM n=12, EWT n=15)
- Word-level analysis uses single lexeme (egregious); needs expansion
- Results marked as preliminary pending full verification

## Multi-Agent Dispatch

Before dispatching multiple agents, ALWAYS ask Brett:
1. **Which model(s)?** Claude, Codex, Gemini, Copilot
2. **Redundant outputs?** Multiple models on same task?

See portfolio-level `CLAUDE.md` for CLI command patterns.
