# Construction battery spot-check (2026-01-15)

Sample
- Stratified sample: 10 positives + 10 negatives each for all_cleft, way_construction, comparative_correlative.
- Source: `out/cx_battery_candidates.csv`, seed 17.
- Annotated file: `out/cx_battery_spotcheck_annotated.csv`.

Findings (manual)
- all_cleft: 8/10 positives look like true all-clefts; false positives are mainly `after all` discourse-marker cases.
- way_construction: 6/8 clear positives (precision ~0.75 after excluding 2 unclear), with false positives from `on my way`/`on our ways` and borderline cases like `learn/know my way`.
- comparative_correlative: 2/10 positives are true correlatives; many false positives come from generic `the` sequences. One negative example was actually a correlative (false negative), indicating the current label heuristic misses some multi-clause cases.

Implications
- all_cleft and way_construction heuristics appear usable at this grain but need `after all` and `on my way` exclusions for higher precision.
- comparative_correlative requires tighter structural cues (e.g., two comparative phrases linked by clause boundary) to reduce false positives and catch true correlatives.
