# Language as a Stack of Homeostatic Property-Cluster Kinds

Preprint manuscript and analysis scripts supporting: *Language as a Stack of Homeostatic Property-Cluster Kinds: From Phonemes to Constructions*.

## What's Here
- `main.tex` + `house-style-and-preamble.tex`: LaTeX source
- `refs.bib`: bibliography
- `images/`: figures referenced by the paper
- `data/`: tabular outputs used in the manuscript
- `python/`: analysis scripts (numbered by workflow order)

## Build the PDF
Requires `xelatex`, `biber`, and the `biblatex-unified` style.

```bash
xelatex main.tex && biber main && xelatex main.tex && xelatex main.tex
```

## Run Analyses
Phoneme/PHOIBLE pipeline (run from repo root):

```bash
python python/01_download_phoible.py
python python/02_make_ridgelines.py
python python/03_model_y.py
```

Construction/UD pipeline (flagship *or even*):

```bash
python python/10_download_ud.py
python python/11_extract_or_even.py
python python/12_or_even_profile_plot.py
python python/13_or_even_prcurve.py
```

Outputs land in `out/`, `images/`, or `figs/` depending on the script.

Word-level drift pipeline (HistWords COHA lemma SGNS):

```bash
python python/04_download_histwords.py
python python/04_download_histwords_stats.py
python python/05_word_drift_multilexeme.py
```

Outputs land in `out/word_drift_*` and Appendix~A tables.

## License
- Paper, figures, and data: CC BY 4.0 (see `LICENSE`)
- Code: MIT (see `python/LICENSE`)
