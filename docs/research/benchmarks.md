# Benchmark Results

> **⚠️ Results under revision.** This page previously published a summary table across
> iris, wine, and breast cancer. Those numbers have been withdrawn. See `paper/CLAIMS.md`
> in the repository for the per-claim audit.

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

## What replaces it

`paper/PLAN.md` specifies the rebuilt protocol:

- Outer 10-fold stratified cross-validation × 3 repeats for reporting; inner 5-fold for all
  hyperparameter selection, applied identically to every method
- Baselines: CART tuned by inner CV over `ccp_alpha`, unconstrained CART, and random search
  over the same tree space at a matched evaluation budget
- Roughly 20 datasets from OpenML CC-18 — iris, wine, and breast cancer are saturated
- Wilcoxon signed-rank and Friedman + Nemenyi across datasets; equivalence by TOST against a
  pre-registered 2% absolute-accuracy margin
- Seeded end to end, with every reported number committed alongside its config and seed

Hypotheses, decision thresholds, and kill criteria are fixed in advance in
`paper/PREREGISTRATION.md`.

## Reporting

Interpretability will be reported as number of leaves, mean weighted decision-path length,
and number of distinct features used. The composite interpretability score is a search
heuristic and will not appear as a reported outcome measure.
