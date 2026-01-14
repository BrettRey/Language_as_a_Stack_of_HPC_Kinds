# Corpus Design: Fixed Palette for Case C

**Date:** 2026-01-14  
**Status:** RESOLVED

## Decision

Adopt a **fixed corpus palette** with register as an explicit experimental factor, rather than varying corpora per construction.

## Corpus Palette

| Corpus | Register profile | UD link |
|--------|------------------|---------|
| **GUM** | Broad multi-genre (academic, news, fiction, social, spoken) | [UD_English-GUM](https://universaldependencies.org/treebanks/en_gum/index.html) |
| **EWT** | Web genres (blogs, newsgroups, emails, reviews, Yahoo answers) | [UD_English-EWT](https://universaldependencies.org/treebanks/en_ewt/index.html) |
| **GUMReddit** | Informal social media (Reddit) | [UD_English-GUMReddit](https://universaldependencies.org/treebanks/en_gumreddit/index.html) |

## Three-Step Reporting Protocol

For **every** construction in the battery:

1. **Prevalence map**: Count anchor-/frame-candidates and validated instances per corpus (or genre slice). Zeros are informative as distributional scope, not test failures.

2. **In-register projectibility**: Transfer between corpora/slices in the same register class (informal<->informal; edited<->edited), provided minimum-count thresholds are met.

3. **Cross-register stress test**: Train in informal, test in edited (and optionally reverse). This makes "register-local homeostasis" falsifiable.

## Minimum Evaluability Thresholds

- Anchor-present candidate set: >=20 items in both train and test
- Positive class: >=10 instances in both train and test
- When thresholds aren't met: report prevalence, mark as "not estimable"

## Why This Works

- Same corpus grid for all constructions -> no opportunistic asymmetry
- Sparsity is informative (scope/extent of kind), not a classifier failure
- "Projection fails outside informal dialogue" means *given adequate data*, not "no tokens available"
