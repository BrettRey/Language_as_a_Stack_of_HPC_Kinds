# Language as a Stack of HPC Kinds

**Status:** Preprint published (LingBuzz + Zenodo); conceptual claims ready, empirical sections preliminary
**Type:** Article
**Target journal:** TBD
**Last updated:** 2026-01-08

---

## Progress

- [x] Conceptual framework (projectibility + homeostasis diagnostics)
- [x] Case A: Phonemes (PHOIBLE, /y/ scaling) - preliminary
- [x] Case B: Words (semantic drift, egregious) - preliminary
- [x] Case C: Constructions (let alone, cross-corpus transfer) - preliminary
- [x] Failure modes taxonomy (thin/fat/negative)
- [x] Python analysis code
- [ ] Full verification of empirical results
- [ ] Word-level multi-lexeme evaluation
- [ ] Target journal identified
- [ ] Final submission

---

## Key Claims

1. Two operational diagnostics for HPC kinds: **projectibility** (out-of-sample prediction) and **homeostasis** (identifiable stabilizing mechanisms)
2. Phonemes, words, and constructions can qualify as HPC kinds when both diagnostics succeed
3. Failure modes: thin (nonce items), fat (cross-linguistic umbrellas), negative (complement classes)

---

## Empirical Components

| Case | Data | Status | Notes |
|------|------|--------|-------|
| Phonemes | PHOIBLE 2.0 | Preliminary | /y/ scaling, inventory ridgelines |
| Words | Historical corpora | Preliminary | Egregious drift; needs multi-lexeme expansion |
| Constructions | UD GUM + EWT | Preliminary | let alone cross-corpus; small n (12, 15) |

---

## Code

Python scripts in `python/`:
- `01_download_phoible.py` - PHOIBLE data
- `02_make_ridgelines.py` - inventory visualization
- `03_model_y.py` - /y/ scaling model
- `10_download_ud.py` - Universal Dependencies
- `11_extract_let_alone.py` - construction extraction
- `12_profile_plot.py` - cue profiles
- `13_predict_prcurve.py` - cross-corpus evaluation

---

## Connections

- Feeds into HPC book (especially Ch 12: non-grammar worked cases)
- Related to: `Labels_to_Stabilisers/`, `linguistics-metrology/`
- Cites: Miller 2021, Ekstrom 2025, Boyd 1991/1999

---

## Session Log

- **2026-01-08**: Imported from Overleaf; STATUS.md created
