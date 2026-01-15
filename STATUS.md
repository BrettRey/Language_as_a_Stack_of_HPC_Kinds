# Language as a Stack of HPC Kinds

**Status:** Preprint published (LingBuzz + Zenodo); conceptual claims ready, empirical sections preliminary
**Type:** Article
**Target journal:** TBD
**Last updated:** 2026-01-15

---

## Progress

- [x] Conceptual framework (projectibility + homeostasis diagnostics)
- [x] Case A: Phonemes (PHOIBLE, /y/ scaling) - preliminary
- [x] Case B: Words (semantic drift, mixed outcomes) - preliminary
- [x] Case C: Constructions (or even, cross-corpus transfer) - preliminary
- [x] Construction homeostasis diagnostics (cue covariance + downsampling) - preliminary
- [x] Construction labeling decoupled from cues (reduced PR--AUC saturation) - preliminary
- [x] Expanded UD English palette for construction battery - preliminary
- [x] Construction heuristics broadened to increase estimable pairs - preliminary
- [x] Gelman-style checks (bootstrap CIs + shuffled-label baselines) - preliminary
- [x] Construction spot-check sample (all-cleft, way, comparative correlative) - preliminary
- [x] Comparative-correlative heuristic tightened with UD Degree=Cmp + "than" filter - preliminary
- [x] Construction heuristics frozen after Gelman-style checks (comparative correlatives treated as non-evaluable at this grain)
- [x] Construction tier declared complete for this round (frozen unless new task design)
- [x] Failure modes taxonomy (thin/fat/negative)
- [x] Python analysis code
- [ ] Full verification of empirical results
- [x] Word-level multi-lexeme evaluation (HistWords COHA lemma SGNS; high-drift adjectives + matched controls)
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
| Words | Historical corpora (HistWords COHA lemma SGNS) | Preliminary | Multi-lexeme adjectives + matched controls; egregious excluded (zero vectors) |
| Constructions | UD English palette (GUM, EWT, GUMReddit, ESL/ESLSpok, GENTLE, CHILDES, LinES, PUD, ParTUT, UniDive, ATIS, LittlePrince, Pronouns, CTeTex, PCEDT) | Preliminary | Cue proxies decoupled from labels; estimable set now N--P--N + resultatives; homeostasis tables added |

---

## Code

Python scripts in `python/`:
- `01_download_phoible.py` - PHOIBLE data
- `02_make_ridgelines.py` - inventory visualization
- `03_model_y.py` - /y/ scaling model
- `10_download_ud.py` - Universal Dependencies
- `11_extract_let_alone.py` - legacy let alone extraction
- `11_extract_or_even.py` - or even extraction
- `12_profile_plot.py` - legacy let alone cue profiles
- `12_or_even_profile_plot.py` - or even cue profiles
- `13_predict_prcurve.py` - legacy let alone evaluation
- `13_or_even_prcurve.py` - or even evaluation
- `20_extract_construction_battery.py` - construction battery candidates
- `21_eval_construction_battery.py` - battery evaluation

---

## Connections

- Feeds into HPC book (especially Ch 12: non-grammar worked cases)
- Related to: `Labels_to_Stabilisers/`, `linguistics-metrology/`
- Cites: Miller 2021, Ekstrom 2025, Boyd 1991/1999

---

## Session Log

- **2026-01-08**: Imported from Overleaf; STATUS.md created
- **2026-01-14**: Switched flagship construction to *or even* due to low *let alone* counts in UD
- **2026-01-15**: Relaxed construction heuristics so cues predict (not define) labels; added cue covariance + downsampling tables and stratified resultatives
- **2026-01-15**: Expanded construction battery to additional UD English treebanks; recomputed prevalence and transfer tables
- **2026-01-15**: Broadened construction heuristics to increase estimable pairs (all-cleft, comparative correlative, way now estimable)
- **2026-01-15**: Added Gelman-style robustness checks (bootstrap CIs, shuffled-label baselines) for construction battery
- **2026-01-15**: Spot-checked 10 pos/10 neg samples for all-cleft, way, comparative correlative; notes saved to `notes/cx_battery_spotcheck.md`
- **2026-01-15**: Tightened comparative-correlative heuristic (comparative detection + than filter); remains below evaluability thresholds
- **2026-01-15**: Froze construction heuristics after Gelman-style checks; comparative correlatives recorded as non-evaluable at this grain
- **2026-01-15**: Declared construction tier complete for this round; no further heuristic tweaks without new task design
- **2026-01-15**: Downloaded HistWords COHA lemma SGNS + stats; ran multi-lexeme drift evaluation for high-drift adjectives with matched controls
- **2026-01-15**: Added word-level summary and per-lexeme tables to Appendix A; updated Case B narrative to reflect mixed pass/fail outcomes
- **2026-01-15**: Clarified Case B writeup (mixed evidence framing, removed egregious-specific illustration, updated summary phrasing)
- **2026-01-15**: Normalized non-ASCII punctuation in manuscript and small output logs; left corpus-extracted candidate CSV untouched to preserve source fidelity
- **2026-01-15**: Replaced em-dash usage with spaced en-dash style (word~-- word); updated house style guidance

---

## Decision Log

- **2026-01-15**: Prioritize expanding UD corpora coverage for construction analyses
- **2026-01-15**: Log decisions in STATUS.md going forward
- **2026-01-15**: Checked UD English treebank comparison list; current palette already covers all listed English UD treebanks (no new additions)
- **2026-01-15**: Restrict corpora expansion to English only (no non-English UD)
- **2026-01-15**: No additional corpora for now; keep current UD English palette
- **2026-01-15**: Broaden cue/label heuristics to raise estimable construction counts before adding new corpora
- **2026-01-15**: Run Gelman-style model checks before further construction expansion
- **2026-01-15**: Perform and record manual spot-checks for newly estimable constructions
- **2026-01-15**: Prioritize precision over recall for comparative correlatives to avoid false positives
- **2026-01-15**: Stop further heuristic tweaking; treat comparative correlatives as a failure mode unless a new task design is introduced
- **2026-01-15**: Treat construction tier as closed for now; move effort to other sections
- **2026-01-15**: Use HistWords COHA lemma SGNS embeddings + stats (volstats/freqs/pos) for word-level drift evaluation
- **2026-01-15**: Restrict word targets to high-drift adjectives from the non-stop, non-proper list; force-include nice/sick/gay/awful when available
- **2026-01-15**: Exclude egregious and broadcast from multi-lexeme evaluation due to zero vectors in SGNS release
- **2026-01-15**: Allow control matching to relax to POS-only if log-frequency matching fails; report group sizes explicitly
