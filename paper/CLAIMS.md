# Claims Ledger

Every public claim, its status, and the evidence it rests on.
Audited **2026-08-04** against `main` @ `5f77b56`. Amended **2026-08-06** after tracing
artifact provenance — see `results/PROVENANCE.md`.

Supporting run data lives in `paper/evidence/`. It sits outside `results/` because everything
under `results/` is ignored, which meant the numbers backing this audit were untracked and
would not have survived a fresh clone.

Status values: `SUPPORTED` · `STALE` · `UNSUPPORTED` · `FALSE` · `FABRICATED`

`FABRICATED` was added 2026-08-06 after tracing artifact provenance (see
`results/PROVENANCE.md`). It is distinct from `STALE`: a stale claim is the output of an old
run, whereas these numbers were **literals typed into a figure script** and were never the
output of any run. The original audit assumed the former. Four mutually inconsistent number
sets coexisted at the same commit; the published claims match none of the real ones.

______________________________________________________________________

## Accuracy claims

| Claim                                             | Location                                                               | Status         | Evidence                                                                                                                                                  |
| ------------------------------------------------- | ---------------------------------------------------------------------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| "statistically equivalent accuracy (p > 0.05)"    | `README.md:9`                                                          | **FALSE**      | `paper/evidence/stats-paper-2026-02-27.csv`: digits p_adj=0.0153 (d=-1.03), banknote p_adj=0.00078 (d=-1.32) — both significant *losses* after Bonferroni |
| Iris GA 94.55% vs CART 92.41%, p=0.186            | `README.md`, `docs/research/benchmarks.md`, `docs/research/results.md` | **FABRICATED** | Both figures are `PAPER_RESULTS` literals. Newest real run: GA 0.9259 vs CART 0.9580 (`paper/evidence/result-paper-2026-02-27.csv`), p=0.0545             |
| Wine GA 88.19% vs CART 87.22%, p=0.683            | same                                                                   | **STALE**      | Newest run: GA 0.8646 vs CART 0.8722, p=0.816                                                                                                             |
| Breast cancer GA 91.05% vs CART 91.57%, p=0.640   | same                                                                   | **STALE**      | Newest run: GA 0.8979 vs CART 0.9140, p=0.308                                                                                                             |
| "All p-values > 0.05" → "Statistical Equivalence" | `docs/research/benchmarks.md`                                          | **FALSE**      | Failure to reject ≠ equivalence. No equivalence test was ever run. Two p-values are now \< 0.05 anyway.                                                   |

**Overall:** on the newest 8-dataset run the GA loses to depth-matched CART on **7 of 8**
datasets (only `heart` wins: 0.7833 vs 0.7417), and loses to unconstrained CART on 7 of 8.

______________________________________________________________________

## Size claims

| Claim                              | Location                           | Status          | Note                                                                                                                                                                                                                                                                                                                                        |
| ---------------------------------- | ---------------------------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| "46–82% smaller trees"             | `README.md:9`                      | **FABRICATED**  | **Corrected 2026-08-06.** Not from any CSV. `46` is the iris annotation in `tree_size_comparison.png`, generated from the hardcoded `RESULTS` dict at `scripts/visualize_comprehensive.py:29`; `82` is `PAPER_RESULTS["breast_cancer"]["size_reduction_pct"]` at line 50. Two different literal dicts, spliced. No run produced this range. |
| 55% / 48% / 82% per-dataset        | `docs/research/*`, `docs/index.md` | **FABRICATED**  | Read directly off `PAPER_RESULTS` literals (`visualize_comprehensive.py:50-74`), which no data file backs                                                                                                                                                                                                                                   |
| "Target: 24-77% smaller trees"     | `configs/paper.yaml` header        | **UNSUPPORTED** | Traces to `results/tables/paper-results.csv` (iris 32%, wine 24%, breast cancer 79%) — produced by code not on `main`, now deleted                                                                                                                                                                                                          |
| Size reduction figures per dataset | `docs/research/results.md`         | **STALE**       | Baseline is *unpruned* CART, not cost-complexity-pruned CART. The fair comparison was never run.                                                                                                                                                                                                                                            |

______________________________________________________________________

## Methodology claims

| Claim                                                                     | Location                                     | Status               | Reality                                                                                                                                                                                                                                                                                     |
| ------------------------------------------------------------------------- | -------------------------------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| "Random seed: 42 (reproducibility)"                                       | `docs/research/methodology.md:15`            | **FIXED 2026-08-06** | Was FALSE: `GAConfig` was built without `random_state`, so the seed reached only the CV splitter. Now seeded per fold via `derive_fold_seed(base, dataset, method, fold)`, with the seeds used written to `results/seeds-*.json`. Regression tests in `tests/unit/test_reproducibility.py`. |
| "20-fold stratified CV… same folds for GA and baselines (paired testing)" | `docs/research/methodology.md`               | **SUPPORTED**        | Folds are shared, but the paired t-test across them is invalid (Dietterich 1998)                                                                                                                                                                                                            |
| `classification_metric: accuracy`                                         | `configs/paper.yaml`, `configs/default.yaml` | **FIXED 2026-08-06** | Was silently ignored. Now passed to `FitnessCalculator`; an unrecognised value raises rather than falling back to accuracy. Verified end to end.                                                                                                                                            |
| `early_stopping_rounds` in configs                                        | —                                            | **CORRECTED**        | The original row was wrong: this key is **not present in any `configs/*.yaml`**, so nothing was being dropped. The plumbing now honours it if set (absent = disabled); it is deliberately left unset, since enabling it changes the search budget.                                          |
| Tree constraints `min_samples_split=8, min_samples_leaf=3`                | `configs/paper.yaml`                         | **FALSE**            | Enforced only at initialization. `expand_leaf` (`engine.py:357`) and crossover never re-check against data, so evolved trees can violate them.                                                                                                                                              |
| CART baseline = `DecisionTreeClassifier(max_depth=6, random_state=42)`    | `docs/research/methodology.md:34`            | **SUPPORTED**        | …and that's the problem — untuned, no `ccp_alpha`, no matching `min_samples_*`                                                                                                                                                                                                              |
| "CART (unconstrained)" results                                            | `paper/evidence/result-paper-2026-02-27.csv` | **UNSUPPORTED**      | No function on `main` produces this row                                                                                                                                                                                                                                                     |

______________________________________________________________________

## Fitness claim

| Claim                                                            | Location                   | Status                      | Reality                                                                                                                                                                                                                               |
| ---------------------------------------------------------------- | -------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LDD-3 "Optional validation set for generalization-aware fitness" | `fitness/calculator.py:12` | **UNSUPPORTED in practice** | `X_val`/`y_val` exist but `scripts/experiment.py` never passes them. Leaves are fit on `X` (line 417) and scored on the same `X` (line 425). Fitness is **resubstitution accuracy**; `GAEngine` selects the champion on training fit. |

______________________________________________________________________

## Interpretability claims

| Claim                                       | Location                                              | Status                       | Note                                                                                                                                    |
| ------------------------------------------- | ----------------------------------------------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Composite score measures "interpretability" | `README.md`, `docs/core-concepts/interpretability.md` | **UNSUPPORTED**              | No grounding in literature, no human study. `semantic_coherence` (std of feature depths) is invented and weighted 0.30 in `paper.yaml`. |
| `feature_coherence` rewards fewer features  | `fitness/calculator.py:263`                           | **SUPPORTED but confounded** | `1 − n_used/n_total` scales with dataset dimensionality — 30-feature datasets get a free high score                                     |

______________________________________________________________________

## Action

All `FALSE` and `STALE` rows must be removed from public docs in Phase 0. See
`paper/PLAN.md`.

______________________________________________________________________

## Claims the paper makes (added 2026-09-29)

Every claim in `paper/gecco/main.tex`, with the evidence it rests on. All numbers in the
paper are generated from these files by `scripts/paper_assets.py`.

| Claim                                                                                     | Status        | Evidence                                                                 |
| ----------------------------------------------------------------------------------------- | ------------- | ------------------------------------------------------------------------ |
| GA frontier beats budget-matched random search (+0.66 HV, p_holm 0.032, 15/20)           | **SUPPORTED** | `evidence/frontier-2026-08-07/` (pre-registered K1)                       |
| GA frontier does not dominate CART's pruning path (45% < 60%)                             | **SUPPORTED** | same (pre-registered K2)                                                  |
| Tuned GA tree is not equivalent to tuned CART within ±2 points (8/20 lose > 2)           | **SUPPORTED** | `evidence/k3-2026-09-29/` (pre-registered K3/H2)                          |
| GA trees are smaller than tuned CART's (6.3 vs 18.4 leaves)                               | **SUPPORTED** | same — *at lower accuracy on 8/20 datasets; never stated without that*   |
| GA fronts are truncated at small trees; the HV gap tracks the best-accuracy gap          | SUPPORTED, exploratory | frontier evidence + `scripts/paper_assets.py`                  |
| Validation split and budget each explain ~30% of the truncation; small-tree bias none    | SUPPORTED, exploratory | `evidence/frontier-diagnostic-2026-09-29/` (4 datasets, 10 folds) |
| Verdicts unchanged by constraint repair                                                   | SUPPORTED, sensitivity | `evidence/frontier-repair-2026-09-29/`                        |
| Verdict unchanged against a depth-capped CART path (40%)                                  | SUPPORTED, exploratory | `evidence/cart-depth6-2026-09-29/`                            |
| GOSDT does not dominate CART's path either; GA and GOSDT indistinguishable               | SUPPORTED, exploratory | `evidence/gosdt-2026-09-29/` (19/20 datasets, 10 folds)       |
| 8–27% of evaluated GA trees violate sample-count constraints                              | SUPPORTED     | `evidence/constraint-violations-2026-09-29/` (one fold of four datasets)  |

The configured constraint claim above ("min_samples_split=8, min_samples_leaf=3", FALSE)
is now **fixable**: `tree.repair_constraints: true` enforces it after variation. It stays
off in `configs/paper.yaml` so the pre-registered run remains reproducible.

