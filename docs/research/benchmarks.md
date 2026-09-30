# Benchmark Results

This page used to publish a summary table across iris, wine and breast cancer. Those
numbers are withdrawn; the reasons are below. The replacement benchmark has run, and its
results are on the [Results](results.md) page.

## Why the table was withdrawn

**The size reduction was measured against the wrong baseline.** CART ran as
`DecisionTreeClassifier(max_depth=6, random_state=42)` — untuned, with no `ccp_alpha` and no
`min_samples_*` matching the GA's tree constraints. Comparing an evolved tree penalised for
node count against an unpruned greedy tree measures the absence of pruning, not the value of
evolutionary search. The fair comparison against cost-complexity-pruned CART was never run.

**"Statistical equivalence" did not follow from the p-values.** Failure to reject a null
hypothesis is not evidence for it. Establishing equivalence requires an equivalence test
against a pre-specified margin — TOST, or a Bayesian test with a region of practical
equivalence — and none was run. Separately, the paired t-test was computed across
cross-validation folds, which share training data and are therefore not independent
(Dietterich 1998), so the p-values were not valid to begin with.

**The headline figures were not reproducible.** The 46–82% range came from
`results/tables/paper-results.csv`, produced by code that never landed on `main`. No commit
in this repository regenerates it.

**The newest run points the other way.** On the most recent 8-dataset run the GA loses to
depth-matched CART on 7 of 8 datasets, with two of those losses significant after Bonferroni
correction (`digits` p_adj = 0.0153, d = −1.03; `banknote` p_adj = 0.00078, d = −1.32).

## What replaced it

The protocol in `paper/PLAN.md`, run in August–September 2026:

- Outer 10-fold stratified cross-validation (× 3 repeats for the frontier run); inner 5-fold
  for all hyperparameter selection, applied the same way to every method
- Baselines: CART's full cost-complexity pruning path, tuned CART, and random search over
  the same tree space at an exactly matched evaluation budget
- 20 datasets from OpenML-CC18, listed in `paper/DATASETS.md`
- Holm-corrected Wilcoxon signed-rank tests across datasets for K1; TOST against a 2-point
  accuracy margin for K3
- Seeded end to end, with every run committed alongside its config, seeds and command in
  `paper/evidence/`

Hypotheses, thresholds and kill criteria were fixed in advance in
`paper/PREREGISTRATION.md`; deviations are logged there with dates.

## Reporting

Interpretability is reported as node count, leaves, mean decision-path length and distinct
features used. The composite interpretability score is a search heuristic and appears in no
reported result.
