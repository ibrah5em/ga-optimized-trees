# Provenance

Every result artifact committed to this repository must be attributable to the commit,
config, and seed that produced it. Anything that is not gets deleted.

Audited **2026-08-06**. Nothing in this directory is currently tracked except `.gitkeep`
placeholders.

______________________________________________________________________

## The rule

A result artifact may be committed only if all four hold:

1. A command on `main` regenerates it.
1. The config that produced it is committed next to it.
1. The seed that produced it is committed next to it.
1. Its numbers appear nowhere in source code as literals.

If a figure or table cannot be regenerated from committed data, it is not evidence — it is
a picture of a claim.

______________________________________________________________________

## What was removed, and why

### `results/tables/paper-results.csv`

Source of the retired "24–77% smaller trees" target in `configs/paper.yaml`
(reductions computed from it: iris 32%, wine 24%, breast cancer 79%). No function on `main`
writes `results/tables/` — `scripts/experiment.py:519` writes
`results/result-{config}-{date}.csv` instead. Produced by code that was never merged.

### `results/tables/results-{accuracy_focused,balanced,fast,interpretability-focused,optimized}.csv`

Same unmerged write path. These appear to record real runs, but with no committed config or
seed and no code on `main` that produces them, they cannot be tied to anything.

### `results/figures/*.png` (6 files)

Generated from **hardcoded literals in `scripts/visualize_comprehensive.py`**, not from any
data file. `tree_size_comparison.png` renders the title "GA Produces 2-7× Smaller Trees" with
"46% smaller" and "49% smaller" annotations baked into the pixels.

### `results/statistics/*.xlsx` (4 files)

Nothing in `scripts/` or `src/` writes `.xlsx` — the codebase only ever reads that format
(`dataset_loader.py:456`). Unattributable at any commit. `.gitignore:150` already declared
these should not be tracked; they predate the rule.

______________________________________________________________________

## The four inconsistent number sets

Worth recording, because the disagreement between them is the reason none of the published
claims held up. All four coexisted in the repository at the same commit:

| Set                                                      | iris GA acc | iris GA nodes | Fed                          |
| -------------------------------------------------------- | ----------- | ------------- | ---------------------------- |
| `PAPER_RESULTS` literal, `visualize_comprehensive.py:50` | 94.55%      | 7.4           | README + all docs claims     |
| `RESULTS` literal, `visualize_comprehensive.py:29`       | 95.33%      | 7.4           | the committed figures        |
| `results/tables/paper-results.csv`                       | 93.75%      | 11.2          | `paper.yaml`'s 24–77% target |
| `paper/evidence/result-paper-2026-02-27.csv`             | 92.59%      | 5.9           | nothing — never published    |

The published "46–82% smaller trees" was assembled from two different sets: **46%** is the
iris annotation in `tree_size_comparison.png` (from `RESULTS`), **82%** is
`PAPER_RESULTS["breast_cancer"]["size_reduction_pct"]`. The docs pages reported 55/48/82 from
`PAPER_RESULTS` alone. No single run ever produced the published range.

Note what this means: the headline numbers were not the stale output of an old run. They were
**literals typed into a figure script**. `paper/CLAIMS.md` originally attributed them to
`results/tables/paper-results.csv`; that was wrong, and the row has been corrected.

______________________________________________________________________

## Audit evidence

`paper/evidence/` holds the 2026-02-27 run — the most recent real run, and the evidence base
for `paper/CLAIMS.md`. It lives outside `results/` because everything under `results/` is
ignored: the only numbers backing the audit were untracked and would not have survived a
fresh clone.

Those files are **frozen evidence, not results**. They come from the pre-audit protocol —
resubstitution fitness, unseeded GA, invalid cross-fold t-tests — so they are not valid
outcomes either. They are retained solely so the claims audit is checkable by someone else.

An earlier run also exists locally under `results/benchmark/` (2026-02-26), untracked and
ignored. It is a fourth inconsistent set and is not preserved.

______________________________________________________________________

## Still to fix

`scripts/visualize_comprehensive.py` on `main` still contains both hardcoded dicts. Running
it regenerates figures asserting the withdrawn claims from numbers with no data behind them.
Deleting the output while leaving the generator in place fixes nothing.
