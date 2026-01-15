# Construction battery spot-check (2026-01-15)

Sample
- Stratified sample: 10 positives + 10 negatives each for all_cleft, way_construction, comparative_correlative.
- Source: `out/cx_battery_candidates.csv`, seed 17.
- Annotated file: `out/cx_battery_spotcheck_annotated.csv`.

Findings (manual)
- all_cleft: 8/10 positives look like true all-clefts; false positives are mainly `after all` discourse-marker cases.
- way_construction: 6/8 clear positives (precision ~0.75 after excluding 2 unclear), with false positives from `on my way`/`on our ways` and borderline cases like `learn/know my way`.
- comparative_correlative: 9/10 positives look like true correlatives after the latest tightening, but several negatives are still true correlatives (mostly elliptical patterns like `the sooner the better`), so recall remains low.

Implications
- all_cleft and way_construction heuristics appear usable at this grain but need `after all` and `on my way` exclusions for higher precision.
- comparative_correlative precision improved, but the heuristic is still too conservative (misses ellipsis and some multi-clause cases), so prevalence stays below evaluability thresholds.
